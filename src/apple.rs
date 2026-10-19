//! MLX GPU operations. Candle/MLX interoperability currently materializes an activation at
//! each boundary. Both allocators use unified memory, but these boundaries are not zero-copy.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use mlx_rs::{Array, Dtype};

pub fn init() -> Result<()> {
    static INIT: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    INIT.get_or_init(|| {
        (|| -> Result<()> {
            // Release archives carry the Metal library beside the executable. Homebrew installs
            // it in share/h3-rs. Configure this before MLX initializes the Metal device.
            if let Ok(exe) = std::env::current_exe() {
                if let Some(dir) = exe.parent() {
                    for path in [
                        dir.join("mlx.metallib"),
                        dir.join("../share/h3-rs/mlx.metallib"),
                    ] {
                        if path.is_file() {
                            mlx_rs::metal::set_metallib_path(path.to_string_lossy())?;
                            break;
                        }
                    }
                }
            }
            mlx_rs::Device::set_default(&mlx_rs::Device::new(mlx_rs::DeviceType::Gpu, 0));
            let total = physical_memory()?;
            let limit = match std::env::var("H3_UNIFIED_MEMORY_GB") {
                Ok(v) => {
                    let gb: f64 = v
                        .parse()
                        .context("H3_UNIFIED_MEMORY_GB must be a positive number")?;
                    anyhow::ensure!(
                        gb.is_finite() && gb > 0.0 && gb * (1u64 << 30) as f64 <= total as f64,
                        "H3_UNIFIED_MEMORY_GB must fit physical unified memory"
                    );
                    (gb * (1u64 << 30) as f64) as usize
                }
                Err(_) => total / 5 * 4,
            };
            mlx_rs::memory::set_memory_limit(limit)?;
            mlx_rs::memory::set_cache_limit((limit / 16).min(512 << 20))?;
            Ok(())
        })()
        .map_err(|e| format!("{e:#}"))
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!(e.clone()))?;
    Ok(())
}

fn physical_memory() -> Result<usize> {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "could not query unified memory size"
    );
    Ok(std::str::from_utf8(&output.stdout)?.trim().parse()?)
}

pub fn memory_info() -> Result<(usize, usize)> {
    init()?;
    let total = mlx_rs::memory::memory_limit()?;
    let mut used = mlx_rs::memory::active_memory()?;
    // Candle allocations share the same physical pool; count them in automatic placement.
    if let Device::Metal(d) = Device::new_metal(0)? {
        used = used.max(d.device().current_allocated_size() as usize);
    }
    Ok((total.saturating_sub(used), total))
}

pub(crate) fn from_tensor(t: &Tensor) -> Result<Array> {
    init()?;
    let t = t.to_device(&Device::Cpu)?.contiguous()?.flatten_all()?;
    let a = match t.dtype() {
        DType::F16 => Array::from_slice(&t.to_vec1::<half::f16>()?, &[t.elem_count() as i32]),
        DType::BF16 => Array::from_slice(&t.to_vec1::<half::bf16>()?, &[t.elem_count() as i32]),
        _ => Array::from_slice(
            &t.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            &[t.elem_count() as i32],
        ),
    };
    Ok(a)
}

pub(crate) fn array(t: &Tensor) -> Result<Array> {
    let shape: Vec<i32> = t.dims().iter().map(|&v| v as i32).collect();
    Ok(from_tensor(t)?.reshape(&shape)?)
}

pub(crate) fn tensor(a: &Array, dev: &Device, dtype: DType) -> Result<Tensor> {
    let shape: Vec<usize> = a.shape().iter().map(|&v| v as usize).collect();
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?.reshape(&[-1])?;
    Ok(Tensor::from_slice(a.try_as_slice::<f32>()?, shape.as_slice(), dev)?.to_dtype(dtype)?)
}

/// Lossless signed int8 -> MLX affine uint32 packing. No re-quantization or dense weight copy.
pub struct Int8Weight {
    q: Array,
    scale: Array,
    bias: Array,
}

impl Int8Weight {
    pub fn new(w: &Tensor, scale: &Tensor) -> Result<Self> {
        init()?;
        let (n, k) = w.dims2()?;
        anyhow::ensure!(
            k % 64 == 0,
            "MLX int8 GEMM requires an input width divisible by 64"
        );
        let bytes = w.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
        let packed: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0] ^ 128, b[1] ^ 128, b[2] ^ 128, b[3] ^ 128]))
            .collect();
        let scales = scale
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        anyhow::ensure!(
            scales.len() == 1 || scales.len() == n,
            "int8 scale must be scalar or per-row"
        );
        let scales: Vec<f32> = (0..n)
            .flat_map(|r| {
                std::iter::repeat_n(scales[if scales.len() == 1 { 0 } else { r }], k / 64)
            })
            .collect();
        let scale = Array::from_slice(&scales, &[n as i32, (k / 64) as i32]);
        let bias = scale.multiply(Array::from(-128.0f32))?;
        let q = Array::from_slice(&packed, &[n as i32, (k / 4) as i32]);
        q.eval()?;
        scale.eval()?;
        bias.eval()?;
        Ok(Self { q, scale, bias })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = mlx_rs::ops::quantized_matmul(
            array(x)?,
            &self.q,
            &self.scale,
            &self.bias,
            true,
            64,
            8,
        )?;
        tensor(&y, x.device(), x.dtype())
    }
}

pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let qa = array(q)?.transpose_axes(&[0, 2, 1, 3])?;
    let ka = array(k)?.transpose_axes(&[0, 2, 1, 3])?;
    let va = array(v)?.transpose_axes(&[0, 2, 1, 3])?;
    let mask = causal.then_some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal);
    let y = mlx_rs::fast::scaled_dot_product_attention(
        &qa,
        &ka,
        &va,
        1.0 / (q.dim(3)? as f32).sqrt(),
        mask,
        None,
    )?;
    tensor(&y.transpose_axes(&[0, 2, 1, 3])?, q.device(), q.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_int8_preserves_signed_weights_and_row_scales() -> Result<()> {
        let dev = crate::backend::new_device(0)?;
        let bytes: Vec<u8> = (0..128).map(|i| (i as i8).wrapping_mul(3) as u8).collect();
        let w = Tensor::from_vec(bytes, (2, 64), &dev)?;
        let scales = Tensor::from_slice(&[0.125f32, 0.25], (2, 1), &dev)?;
        let input: Vec<f32> = (0..192).map(|i| (i as f32 * 0.13).sin()).collect();
        let x = Tensor::from_vec(input, (3, 64), &dev)?;
        let got = Int8Weight::new(&w, &scales)?
            .forward(&x)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reference = x
            .matmul(&crate::kernels::dequant_i8(&w, &scales, DType::F32)?.t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (a, b) in got.iter().zip(reference) {
            assert!((a - b).abs() < 1e-3, "{a} != {b}");
        }
        Ok(())
    }

    #[test]
    fn mlx_attention_matches_chunked_gqa_with_causal_mask() -> Result<()> {
        let dev = crate::backend::new_device(0)?;
        let q = Tensor::from_vec(
            (0..4 * 2 * 64)
                .map(|i| (i as f32 * 0.07).sin())
                .collect::<Vec<_>>(),
            (1, 4, 2, 64),
            &dev,
        )?;
        let k = Tensor::from_vec(
            (0..4 * 64)
                .map(|i| (i as f32 * 0.05).cos())
                .collect::<Vec<_>>(),
            (1, 4, 1, 64),
            &dev,
        )?;
        let v = (&k * 0.7)?;
        let got = attention(&q, &k, &v, true)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reference = crate::ops::attention_fallback(&q, &k, &v, 0.125, true)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (a, b) in got.iter().zip(reference) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
        Ok(())
    }
}
