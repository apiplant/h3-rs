//! Shared tensor ops: norms, attention, activations.

use anyhow::Result;
use candle_core::{DType, Tensor, D};

/// RMSNorm over the last dim, weight cast to the input dtype (fused candle kernel).
pub fn rms_norm(x: &Tensor, w: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let x = x.contiguous()?;
    let dim = x.dim(D::Minus1)?;
    let w = match w {
        Some(w) => w.to_dtype(x.dtype())?,
        None => Tensor::ones(dim, x.dtype(), x.device())?,
    };
    Ok(candle_nn::ops::rms_norm(&x, &w, eps)?)
}

/// LayerNorm over the last dim (computed in f32, returned in the input dtype).
pub fn layer_norm(x: &Tensor, w: Option<&Tensor>, b: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let mut y = xc.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
    if let Some(w) = w {
        y = y.broadcast_mul(&w.to_dtype(DType::F32)?)?;
    }
    if let Some(b) = b {
        y = y.broadcast_add(&b.to_dtype(DType::F32)?)?;
    }
    Ok(y.to_dtype(dt)?)
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

/// `silu(first half) * second half` of the last dim.
pub fn swiglu(x: &Tensor) -> Result<Tensor> {
    let n = x.dim(D::Minus1)? / 2;
    let a = x.narrow(D::Minus1, 0, n)?;
    let b = x.narrow(D::Minus1, n, n)?;
    Ok((candle_nn::ops::silu(&a.contiguous()?)? * b)?)
}

pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

pub fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu_erf()?)
}

/// Scaled dot-product attention on `[B, S, H, D]` tensors (k/v may have fewer heads: GQA).
/// Returns `[B, S, H, D]`.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let d = q.dim(D::Minus1)?;
    let scale = 1.0 / (d as f32).sqrt();
    #[cfg(feature = "flash-attn")]
    {
        if q.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16) && d <= 256 {
            let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
            return Ok(candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)?);
        }
    }
    attention_fallback(q, k, v, scale, causal)
}

/// Chunked softmax attention (no flash kernel): processes queries in blocks to bound memory.
pub fn attention_fallback(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
    let (b, s, h, d) = q.dims4()?;
    let hk = k.dim(2)?;
    let rep = h / hk;
    let expand = |t: &Tensor| -> Result<Tensor> {
        let t = t.transpose(1, 2)?.contiguous()?; // [B, Hk, S, D]
        if rep == 1 {
            return Ok(t);
        }
        let (b, hk, s, d) = t.dims4()?;
        Ok(t.unsqueeze(2)?.expand((b, hk, rep, s, d))?.reshape((b, hk * rep, s, d))?)
    };
    let kt = expand(k)?;
    let vt = expand(v)?;
    let sk = kt.dim(2)?;
    let qt = q.transpose(1, 2)?.contiguous()?;
    let chunk = (64 * 1024 * 1024 / (sk * h * 4).max(1)).clamp(1, s);
    let mut outs = vec![];
    let mut start = 0;
    while start < s {
        let n = chunk.min(s - start);
        let qc = qt.narrow(2, start, n)?.to_dtype(DType::F32)?;
        let mut att = (qc.matmul(&kt.to_dtype(DType::F32)?.t()?)? * scale as f64)?;
        if causal {
            let mask: Vec<f32> = (0..n)
                .flat_map(|i| (0..sk).map(move |j| if j > start + i + (sk - s) { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            let mask = Tensor::from_vec(mask, (n, sk), q.device())?;
            att = att.broadcast_add(&mask)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        outs.push(att.matmul(&vt.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?);
        start += n;
    }
    let o = Tensor::cat(&outs, 2)?; // [B, H, S, D]
    let _ = (b, d);
    Ok(o.transpose(1, 2)?.contiguous()?)
}
