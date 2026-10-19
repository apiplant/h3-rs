//! Tensor equivalents of the CUDA-only fused kernels, for Metal and CPU parity tests.
use candle_core::{DType, Device, Result, Tensor};

fn error(e: anyhow::Error) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}
fn floats(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()
}

pub fn dequant_i8(q: &Tensor, scale: &Tensor, out: DType) -> Result<Tensor> {
    let (n, k) = q.dims2()?;
    let bytes = q.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    let s = floats(scale)?;
    if s.len() != 1 && s.len() != n {
        candle_core::bail!("int8 scale must be scalar or per-row")
    }
    let v: Vec<f32> = bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| (b as i8) as f32 * s[if s.len() == 1 { 0 } else { i / k }])
        .collect();
    Tensor::from_vec(v, (n, k), q.device())?.to_dtype(out)
}

fn fp8(v: u8) -> f32 {
    let e = (v >> 3) & 15;
    let m = v & 7;
    let x = if e == 0 {
        (m as f32 / 8.0) * 2f32.powi(-6)
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if v & 128 != 0 {
        -x
    } else {
        x
    }
}

pub fn dequant_nvfp4(q: &Tensor, bs: &Tensor, global: &Tensor, out: DType) -> Result<Tensor> {
    let (n, kp) = q.dims2()?;
    let k = 2 * kp;
    let qh = q.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    let sh = bs.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    let g = floats(global)?[0];
    let vals = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let ncb = (k / 16).div_ceil(4);
    let mut v = Vec::with_capacity(n * k);
    for r in 0..n {
        for c in 0..k {
            let b = c / 16;
            let off = (r / 128 * ncb + b / 4) * 512 + (r % 32) * 16 + (r % 128 / 32) * 4 + b % 4;
            let scale = *sh
                .get(off)
                .ok_or_else(|| candle_core::Error::Msg("truncated NVFP4 scale table".into()))?;
            let byte = qh[r * kp + c / 2];
            let nib = if c % 2 == 0 { byte >> 4 } else { byte & 15 };
            v.push(vals[nib as usize] * fp8(scale) * g);
        }
    }
    Tensor::from_vec(v, (n, k), q.device())?.to_dtype(out)
}

pub fn rope_table(angles: &Tensor) -> Result<Tensor> {
    let a = angles.to_dtype(DType::F32)?;
    Tensor::stack(&[a.cos()?, a.sin()?], 0)?.contiguous()
}

#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(
    x: &Tensor,
    off: usize,
    heads: usize,
    dim: usize,
    norm: Option<(Option<&Tensor>, f32)>,
    table: Option<&Tensor>,
    rot: usize,
    out: DType,
) -> Result<Tensor> {
    qk_norm_rope_strided(x, off, dim, heads, dim, norm, table, rot, out)
}

#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_strided(
    x: &Tensor,
    off: usize,
    stride: usize,
    heads: usize,
    dim: usize,
    norm: Option<(Option<&Tensor>, f32)>,
    table: Option<&Tensor>,
    rot: usize,
    out: DType,
) -> Result<Tensor> {
    let s = x.dim(0)?;
    let indices: Vec<u32> = (0..heads)
        .flat_map(|h| (0..dim).map(move |d| (off + h * stride + d) as u32))
        .collect();
    let indices = Tensor::from_vec(indices, heads * dim, x.device())?;
    let mut y = x
        .index_select(&indices, 1)?
        .reshape((s, heads, dim))?
        .to_dtype(DType::F32)?;
    if let Some((w, eps)) = norm {
        let inv = (y.sqr()?.mean_keepdim(2)? + eps as f64)?.sqrt()?;
        y = y.broadcast_div(&inv)?;
        if let Some(w) = w {
            y = y.broadcast_mul(&w.to_dtype(DType::F32)?)?;
        }
    }
    if let Some(t) = table {
        let half = rot / 2;
        if half > 0 {
            let a = y.narrow(2, 0, half)?;
            let b = y.narrow(2, half, half)?;
            let c = t.get(0)?.unsqueeze(1)?;
            let sn = t.get(1)?.unsqueeze(1)?;
            let left = (a.broadcast_mul(&c)? - b.broadcast_mul(&sn)?)?;
            let right = (a.broadcast_mul(&sn)? + b.broadcast_mul(&c)?)?;
            let mut parts = vec![left, right];
            if rot < dim {
                parts.push(y.narrow(2, rot, dim - rot)?);
            }
            y = Tensor::cat(&parts, 2)?;
        }
    }
    y.to_dtype(out)?.contiguous()
}

pub fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let sc = scale.index_select(idx, 0)?.to_dtype(x.dtype())?;
    let sh = shift.index_select(idx, 0)?.to_dtype(x.dtype())?;
    x.broadcast_mul(&(sc + 1.0)?)?.broadcast_add(&sh)
}
pub fn rms_modulate(
    x: &Tensor,
    w: &Tensor,
    eps: f32,
    scale: &Tensor,
    shift: &Tensor,
    idx: &Tensor,
) -> Result<Tensor> {
    modulate(
        &crate::ops::rms_norm(x, Some(w), eps).map_err(error)?,
        scale,
        shift,
        idx,
    )
}
pub fn gate_add(x: &Tensor, y: &Tensor, gate: &Tensor, idx: &Tensor) -> Result<Tensor> {
    x.add(&y.broadcast_mul(&gate.index_select(idx, 0)?.to_dtype(x.dtype())?)?)
}

pub fn act1d(x: &Tensor, fu: &Tensor, fd: &Tensor, la: &Tensor, lb: &Tensor) -> Result<Tensor> {
    let (n, c, l) = x.dims3()?;
    let up = floats(fu)?;
    let down = floats(fd)?;
    if up.len() != 12 || down.len() != 12 {
        candle_core::bail!("act1d expects 12-tap filters")
    }
    let mut u = Tensor::zeros((n, c, 2 * l), DType::F32, x.device())?;
    // Equivalent to replicate pad, upsample/filter and crop in BigVGAN's Activation1d.
    for (kk, &filter) in up.iter().enumerate() {
        let ids: Vec<u32> = (0..2 * l)
            .map(|m| (((m as i64 + 15 - kk as i64) / 2 - 5).clamp(0, l as i64 - 1)) as u32)
            .collect();
        let mask: Vec<f32> = (0..2 * l)
            .map(|m| {
                if (m + 15 - kk) % 2 == 0 {
                    2.0 * filter
                } else {
                    0.0
                }
            })
            .collect();
        let ids = Tensor::from_vec(ids, 2 * l, x.device())?;
        let mask = Tensor::from_vec(mask, (1, 1, 2 * l), x.device())?;
        u = (u + x
            .to_dtype(DType::F32)?
            .index_select(&ids, 2)?
            .broadcast_mul(&mask)?)?;
    }
    let alpha = la.exp()?.reshape((1, c, 1))?;
    let beta = (lb.exp()? + 1e-9)?.reshape((1, c, 1))?;
    let a = (&u
        + u.broadcast_mul(&alpha)?
            .sin()?
            .sqr()?
            .broadcast_div(&beta)?)?;
    let mut y = Tensor::zeros((n, c, l), DType::F32, x.device())?;
    for (kk, &filter) in down.iter().enumerate() {
        let ids: Vec<u32> = (0..l)
            .map(|i| (2 * i as i64 + kk as i64 - 5).clamp(0, 2 * l as i64 - 1) as u32)
            .collect();
        y = (y + (a.index_select(&Tensor::from_vec(ids, l, x.device())?, 2)? * filter as f64)?)?;
    }
    y.to_dtype(x.dtype())
}

pub fn can_fuse_rotation(_: usize, _: usize) -> bool {
    false
}
pub fn int8_linear(
    x: &Tensor,
    w: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    out: DType,
) -> Result<Tensor> {
    int8_linear_ex(x, w, scale, bias, out, false, false)
}
pub fn int8_linear_ex(
    x: &Tensor,
    w: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    out: DType,
    swiglu: bool,
    rotate: bool,
) -> Result<Tensor> {
    let x = if rotate {
        crate::weights::rotate_activation(x, 256).map_err(error)?
    } else {
        x.clone()
    };
    let mut y = x.matmul(&dequant_i8(w, scale, x.dtype())?.t()?)?;
    if let Some(b) = bias {
        y = y.broadcast_add(&b.to_dtype(y.dtype())?)?;
    }
    if swiglu {
        y = crate::ops::swiglu(&y).map_err(error)?;
    }
    y.to_dtype(out)
}
pub fn keep_pool_memory(_: &Device) {}
pub fn trim_pool() {
    #[cfg(feature = "mlx")]
    {
        let _ = mlx_rs::memory::clear_cache();
    }
}

pub fn group_norm_silu(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    groups: usize,
    eps: f32,
) -> Result<Tensor> {
    let shape = x.dims();
    let (t, c) = (shape[0], shape[1]);
    let s = x.elem_count() / (t * c);
    let grouped = x
        .to_dtype(DType::F32)?
        .reshape((t, groups, c / groups, s))?
        .permute((1, 0, 2, 3))?
        .reshape((groups, ()))?;
    let mean = grouped.mean_keepdim(1)?;
    let xc = grouped.broadcast_sub(&mean)?;
    let norm = xc.broadcast_div(&(xc.sqr()?.mean_keepdim(1)? + eps as f64)?.sqrt()?)?;
    let y = norm
        .reshape((groups, t, c / groups, s))?
        .permute((1, 0, 2, 3))?
        .reshape((t, c, s))?
        .broadcast_mul(&gamma.to_dtype(DType::F32)?.reshape((1, c, 1))?)?
        .broadcast_add(&beta.to_dtype(DType::F32)?.reshape((1, c, 1))?)?;
    candle_nn::ops::silu(&y)?
        .reshape(shape)?
        .to_dtype(x.dtype())
}

pub fn dw_time(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let shape = x.dims();
    let (t, c) = (shape[0], shape[1]);
    let s = x.elem_count() / (t * c);
    let k = w.elem_count() / c;
    let x3 = x.reshape((t, c, s))?;
    let mut y = b
        .reshape((1, c, 1))?
        .broadcast_as((t, c, s))?
        .to_dtype(DType::F32)?;
    for i in 0..k {
        let delta = i as i64 - k as i64 / 2;
        let ids: Vec<u32> = (0..t)
            .map(|j| (j as i64 + delta).clamp(0, t as i64 - 1) as u32)
            .collect();
        let mask: Vec<f32> = (0..t)
            .map(|j| {
                if (0..t as i64).contains(&(j as i64 + delta)) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let v = x3
            .index_select(&Tensor::from_vec(ids, t, x.device())?, 0)?
            .to_dtype(DType::F32)?;
        let wt = w.reshape((c, k))?.narrow(1, i, 1)?.reshape((1, c, 1))?;
        y = (y + v.broadcast_mul(&wt)?.broadcast_mul(&Tensor::from_vec(
            mask,
            (t, 1, 1),
            x.device(),
        )?)?)?;
    }
    y.reshape(shape)?.to_dtype(x.dtype())
}

pub fn bilinear_hw(x: &Tensor, ho: usize, wo: usize) -> Result<Tensor> {
    let shape = x.dims();
    let (h, w) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let n = x.elem_count() / (h * w);
    let mut out = shape.to_vec();
    let rank = out.len();
    out[rank - 2] = ho;
    out[rank - 1] = wo;
    x.reshape((n, 1, h, w))?
        .upsample_bilinear2d(ho, wo, false)?
        .reshape(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nvfp4_decodes_high_nibble_and_swizzled_block_scales() -> Result<()> {
        // Four row groups within a 128-row cuBLAS scale tile.
        let q = Tensor::from_vec(vec![0x12u8; 128 * 32], (128, 32), &Device::Cpu)?;
        let mut scales = vec![0u8; 512];
        for r in 0..128 {
            for block in 0..4 {
                let off = (r % 32) * 16 + (r / 32) * 4 + block;
                scales[off] = if r < 32 { 0x38 } else { 0x40 }; // e4m3: 1 or 2
            }
        }
        let s = Tensor::from_vec(scales, 512, &Device::Cpu)?;
        let global = Tensor::new(&[3.0f32], &Device::Cpu)?;
        let y = dequant_nvfp4(&q, &s, &global, DType::F32)?.to_vec2::<f32>()?;
        for (r, ys) in y.iter().enumerate() {
            for (c, &v) in ys.iter().enumerate() {
                let expected =
                    if c % 2 == 0 { 0.5 } else { 1.0 } * if r < 32 { 1.0 } else { 2.0 } * 3.0;
                assert_eq!(v, expected);
            }
        }
        Ok(())
    }
    #[test]
    fn portable_modulation_rope_and_resize_run_on_metal() -> anyhow::Result<()> {
        #[cfg(feature = "mlx")]
        let dev = crate::backend::new_device(0)?;
        #[cfg(not(feature = "mlx"))]
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &dev)?;
        let idx = Tensor::from_slice(&[1u32, 0], 2, &dev)?;
        let scale = Tensor::from_slice(&[0f32, 0., 1., 1.], (2, 2), &dev)?;
        let shift = Tensor::zeros((2, 2), DType::F32, &dev)?;
        assert_eq!(
            modulate(&x, &scale, &shift, &idx)?.to_vec2::<f32>()?,
            vec![vec![2., 4.], vec![3., 4.]]
        );
        let angles = Tensor::zeros((2, 1), DType::F32, &dev)?;
        let y = qk_norm_rope(
            &x,
            0,
            1,
            2,
            None,
            Some(&rope_table(&angles)?),
            2,
            DType::F32,
        )?;
        assert_eq!(y.flatten_all()?.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);
        let y = bilinear_hw(&x.reshape((1, 1, 2, 2))?, 3, 3)?;
        assert!((y.flatten_all()?.to_vec1::<f32>()?[4] - 2.5).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn audio_activation_and_video_temporal_ops_run_on_metal() -> anyhow::Result<()> {
        #[cfg(feature = "mlx")]
        let dev = crate::backend::new_device(0)?;
        #[cfg(not(feature = "mlx"))]
        let dev = Device::Cpu;
        let x = Tensor::ones((1, 1, 3), DType::F32, &dev)?;
        let mut up = vec![0f32; 12];
        up[5] = 0.5;
        up[6] = 0.5;
        let mut down = vec![0f32; 12];
        down[5] = 1.0;
        let zero = Tensor::zeros(1, DType::F32, &dev)?;
        let y = act1d(
            &x,
            &Tensor::from_vec(up, 12, &dev)?,
            &Tensor::from_vec(down, 12, &dev)?,
            &zero,
            &zero,
        )?;
        for value in y.flatten_all()?.to_vec1::<f32>()? {
            assert!((value - (1.0 + 1f32.sin().powi(2))).abs() < 1e-5);
        }
        let x = Tensor::from_slice(&[1f32, 2., 3.], (3, 1, 1, 1), &dev)?;
        let w = Tensor::from_slice(&[1f32, 2., 1.], (1, 3), &dev)?;
        let b = Tensor::new(&[0.5f32], &dev)?;
        assert_eq!(
            dw_time(&x, &w, &b)?.flatten_all()?.to_vec1::<f32>()?,
            vec![4.5, 8.5, 8.5]
        );
        let g = Tensor::ones(1, DType::F32, &dev)?;
        let y = group_norm_silu(&x, &g, &zero, 1, 1e-5)?;
        let expected = [-1f32, 0., 1.].map(|v| {
            let n = v / (2f32 / 3.0 + 1e-5).sqrt();
            n / (1.0 + (-n).exp())
        });
        for (a, b) in y.flatten_all()?.to_vec1::<f32>()?.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5);
        }
        Ok(())
    }
}
