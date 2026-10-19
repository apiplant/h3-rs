//! CUDA fast paths are unavailable on this backend. Attention uses MLX SDPA via ops.
use candle_core::{Device, Result, Tensor};
pub fn available_arch(_: &Device) -> bool {
    false
}
pub fn available(_: &Tensor) -> bool {
    false
}
pub fn gemm_s8(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: Option<&Tensor>,
    _: bool,
    _: bool,
) -> Result<Tensor> {
    candle_core::bail!("gemm_s8 requires the cuda feature")
}
pub fn attention_qkv(q: &Tensor, k: &Tensor, qkv: &Tensor, v_col: usize) -> Result<Tensor> {
    let (s, h, d) = q.dims3()?;
    let v = qkv.narrow(1, v_col, h * d)?.reshape((s, h, d))?;
    attention(q, k, &v)
}
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    crate::ops::attention(&q.unsqueeze(0)?, &k.unsqueeze(0)?, &v.unsqueeze(0)?, false)
        .and_then(|x| Ok(x.squeeze(0)?))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}
