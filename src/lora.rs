//! ComfyUI-style LoRA files for the H3 DiT (`diffusion_model.<module>.lora_A/B` or
//! `lora_down/up`, optional `.alpha`, optional `.reshape_weight` pad-and-add head banks).

use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::weights::SafeFile;

pub struct Lora {
    file: Arc<SafeFile>,
    prefix: String,
}

pub struct Delta {
    /// `[r, in]`
    pub a: Tensor,
    /// `[out, r]`
    pub b: Tensor,
    pub scale: f64,
    pub reshape: Option<Vec<usize>>,
}

impl Lora {
    pub fn open(path: &str) -> Result<Self> {
        let file = SafeFile::open(path)?;
        let prefix = if file.tensors.keys().any(|k| k.starts_with("diffusion_model.")) { "diffusion_model." } else { "" };
        Ok(Self { file, prefix: prefix.to_string() })
    }

    /// Bytes of the low-rank factors once loaded (bf16).
    pub fn device_bytes(&self) -> usize {
        self.file.tensors.values().filter(|t| t.dtype == "BF16" || t.dtype == "F32" || t.dtype == "F16").map(|t| t.shape.iter().product::<usize>() * 2).sum()
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.file.metadata
    }

    /// The low-rank pair for `<module>` (module name without `.weight`).
    pub fn delta(&self, module: &str, dev: &Device) -> Result<Option<Delta>> {
        let p = format!("{}{module}", self.prefix);
        let (up, down) = if self.file.has(&format!("{p}.lora_B.weight")) {
            (format!("{p}.lora_B.weight"), format!("{p}.lora_A.weight"))
        } else if self.file.has(&format!("{p}.lora_up.weight")) {
            (format!("{p}.lora_up.weight"), format!("{p}.lora_down.weight"))
        } else {
            return Ok(None);
        };
        // low-rank factors stay in bf16 (they run in the compute dtype anyway)
        let b = self.file.load_as(&up, dev, DType::BF16)?;
        let a = self.file.load_as(&down, dev, DType::BF16)?;
        let a = a.flatten_from(1)?;
        let b = b.flatten_from(1)?;
        let rank = a.dim(0)? as f64;
        let alpha_key = format!("{p}.alpha");
        let scale = if self.file.has(&alpha_key) {
            self.file.load_as(&alpha_key, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0] as f64 / rank
        } else {
            1.0
        };
        let rk = format!("{p}.reshape_weight");
        let reshape = if self.file.has(&rk) {
            Some(self.file.load(&rk, &Device::Cpu)?.to_vec1::<i64>()?.into_iter().map(|v| v as usize).collect())
        } else {
            None
        };
        Ok(Some(Delta { a, b, scale, reshape }))
    }

    /// Merge into a dense weight + bias, padding to the LoRA's `reshape_weight` shape if given
    /// (PDD head banks). Returns the weights unchanged when the LoRA has no entry.
    pub fn merge_padded(&self, module: &str, w: &Tensor, b: &Tensor, strength: f64, dev: &Device) -> Result<(Tensor, Tensor)> {
        let merge = |name: &str, base: &Tensor| -> Result<Tensor> {
            match self.delta(name, dev)? {
                None => Ok(base.clone()),
                Some(d) => {
                    let mut base = base.to_dtype(DType::F32)?;
                    if let Some(shape) = &d.reshape {
                        base = pad_to(&base, shape)?;
                    }
                    let diff = d.b.to_dtype(DType::F32)?.matmul(&d.a.to_dtype(DType::F32)?)?.reshape(base.shape())?;
                    Ok((base + (diff * (d.scale * strength))?)?)
                }
            }
        };
        Ok((merge(module, w)?, merge(&format!("{module}.bias"), b)?))
    }
}

fn pad_to(t: &Tensor, shape: &[usize]) -> Result<Tensor> {
    let mut t = t.clone();
    for (d, &n) in shape.iter().enumerate() {
        let cur = t.dim(d)?;
        if n > cur {
            let mut zs = t.dims().to_vec();
            zs[d] = n - cur;
            t = Tensor::cat(&[&t, &Tensor::zeros(zs, t.dtype(), t.device())?], d)?;
        }
    }
    Ok(t)
}
