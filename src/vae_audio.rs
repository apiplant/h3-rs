//! MiniMax H3 stereo audio VAE decoder (BigVGAN, 32 kHz; port of `comfy/ldm/minimax/audio_vae.py`).
//! Latents `[1, 32, 2, T]` at 40 Hz -> waveform `[2, T*800]`.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::dit::log;
use crate::kernels;
use crate::weights::SafeFile;

struct Act {
    up: Tensor,
    down: Tensor,
    alpha: Tensor,
    beta: Tensor,
}

impl Act {
    fn load(f: &SafeFile, p: &str, dev: &Device) -> Result<Self> {
        let l = |n: &str| f.load_as(&format!("{p}.{n}"), dev, DType::F32);
        Ok(Self { up: l("upsample.filter")?, down: l("downsample.lowpass.filter")?, alpha: l("act.alpha")?, beta: l("act.beta")? })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        Ok(kernels::act1d(x, &self.up, &self.down, &self.alpha, &self.beta)?)
    }
}

struct Conv1 {
    w: Tensor,
    b: Option<Tensor>,
    pad: usize,
    dil: usize,
}

impl Conv1 {
    fn load(f: &SafeFile, p: &str, dev: &Device, dil: usize) -> Result<Self> {
        let w = f.load_as(&format!("{p}.weight"), dev, DType::F32)?;
        let k = w.dim(2)?;
        let b = if f.has(&format!("{p}.bias")) { Some(f.load_as(&format!("{p}.bias"), dev, DType::F32)?) } else { None };
        Ok(Self { w, b, pad: (k * dil - dil) / 2, dil })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d(&self.w, self.pad, 1, self.dil, 1)?;
        Ok(match &self.b {
            Some(b) => y.broadcast_add(&b.reshape((1, (), 1))?)?,
            None => y,
        })
    }
}

struct AmpBlock {
    convs1: Vec<Conv1>,
    convs2: Vec<Conv1>,
    acts: Vec<Act>,
}

pub struct AudioVae {
    dev: Device,
    mean: Tensor,
    std: Tensor,
    dec_in: Conv1,
    conv_pre: Conv1,
    ups: Vec<(Tensor, Tensor, usize, usize)>, // weight, bias, stride, padding
    res: Vec<AmpBlock>,
    act_post: Act,
    conv_post: Conv1,
    pub sample_rate: u32,
}

impl AudioVae {
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        let f = SafeFile::open(path)?;
        let l = |n: &str| f.load_as(n, dev, DType::F32);
        let rates = [5usize, 5, 2, 2, 2, 2, 2];
        let kernels_ = [9usize, 9, 4, 4, 4, 4, 4];
        let mut ups = vec![];
        for (i, (&u, &k)) in rates.iter().zip(&kernels_).enumerate() {
            let p = format!("decoder.ups.{i}.0");
            ups.push((l(&format!("{p}.weight"))?, l(&format!("{p}.bias"))?, u, (k - u) / 2));
        }
        let ks = [3usize, 7, 11];
        let mut res = vec![];
        for i in 0..rates.len() * ks.len() {
            let p = format!("decoder.resblocks.{i}");
            let mut b = AmpBlock { convs1: vec![], convs2: vec![], acts: vec![] };
            for (j, d) in [1usize, 3, 5].into_iter().enumerate() {
                b.convs1.push(Conv1::load(&f, &format!("{p}.convs1.{j}"), dev, d)?);
                b.convs2.push(Conv1::load(&f, &format!("{p}.convs2.{j}"), dev, 1)?);
            }
            for j in 0..6 {
                b.acts.push(Act::load(&f, &format!("{p}.activations.{j}"), dev)?);
            }
            res.push(b);
        }
        let sr = f.metadata.get("minimax_h3_audio_vae").and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok()).and_then(|v| v["sample_rate"].as_u64()).unwrap_or(32000) as u32;
        log::info(&format!("audio VAE: BigVGAN decoder, {sr} Hz stereo"));
        Ok(Self {
            dev: dev.clone(),
            mean: l("latents_mean")?.reshape((1, 32, 1))?,
            std: l("latents_std")?.reshape((1, 32, 1))?,
            dec_in: Conv1::load(&f, "dec_in_proj", dev, 1)?,
            conv_pre: Conv1::load(&f, "decoder.conv_pre", dev, 1)?,
            ups,
            res,
            act_post: Act::load(&f, "decoder.activation_post", dev)?,
            conv_post: Conv1::load(&f, "decoder.conv_post", dev, 1)?,
            sample_rate: sr,
        })
    }

    /// Normalized latents `[1, 32, 2, T]` -> stereo waveform `[2, T*800]` (CPU, f32 in [-1, 1]).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = z.to_device(&self.dev)?.to_dtype(DType::F32)?;
        let (_, c, s, t) = z.dims4()?;
        let z = z.permute((0, 2, 1, 3))?.reshape((s, c, t))?.contiguous()?;
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let mut x = self.conv_pre.fwd(&self.dec_in.fwd(&z)?)?;
        for (i, (w, b, stride, pad)) in self.ups.iter().enumerate() {
            x = x.conv_transpose1d(w, *pad, 0, *stride, 1, 1)?.broadcast_add(&b.reshape((1, (), 1))?)?;
            let mut acc: Option<Tensor> = None;
            for j in 0..3 {
                let blk = &self.res[i * 3 + j];
                let mut h = x.clone();
                for k in 0..3 {
                    let xt = blk.acts[2 * k].fwd(&h)?;
                    let xt = blk.convs1[k].fwd(&xt)?;
                    let xt = blk.acts[2 * k + 1].fwd(&xt)?;
                    let xt = blk.convs2[k].fwd(&xt)?;
                    h = (xt + h)?;
                }
                acc = Some(match acc {
                    None => h,
                    Some(a) => (a + h)?,
                });
            }
            x = (acc.unwrap() / 3.0)?;
        }
        let x = self.conv_post.fwd(&self.act_post.fwd(&x)?)?.clamp(-1.0, 1.0)?;
        Ok(x.reshape((s, ()))?.to_device(&Device::Cpu)?)
    }
}

// ----------------------------------------------------------------------------------------------
// Encoder: DAC waveform encoder + causal attention projection (posterior mean only).

fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let s = x.broadcast_mul(alpha)?.sin()?.sqr()?;
    Ok((x + s.broadcast_div(&(alpha + 1e-9)?)?)?)
}

struct StridedConv {
    w: Tensor,
    b: Tensor,
    stride: usize,
    pad: usize,
    dil: usize,
}

impl StridedConv {
    fn load(f: &SafeFile, p: &str, dev: &Device, stride: usize, pad: usize, dil: usize) -> Result<Self> {
        Ok(Self { w: f.load_as(&format!("{p}.weight"), dev, DType::F32)?, b: f.load_as(&format!("{p}.bias"), dev, DType::F32)?, stride, pad, dil })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.conv1d(&self.w, self.pad, self.stride, self.dil, 1)?.broadcast_add(&self.b.reshape((1, (), 1))?)?)
    }
}

pub struct AudioEncoder {
    dev: Device,
    conv0: StridedConv,
    /// per encoder block: residual units (alpha1, conv7, alpha2, conv1), block snake alpha, strided conv
    blocks: Vec<(Vec<(Tensor, StridedConv, Tensor, StridedConv)>, Tensor, StridedConv)>,
    alpha_out: Tensor,
    conv_out: StridedConv,
    pre: std::collections::HashMap<String, Tensor>,
    mean_proj: StridedConv,
    mean: Tensor,
    std: Tensor,
}

impl AudioEncoder {
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        let f = SafeFile::open(path)?;
        let l = |n: &str| f.load_as(n, dev, DType::F32);
        let strides = [2usize, 4, 4, 5, 5];
        let mut blocks = vec![];
        for (i, &s) in strides.iter().enumerate() {
            let p = format!("encoder.block.{}", i + 1);
            let mut rus = vec![];
            for (j, d) in [1usize, 3, 9].into_iter().enumerate() {
                let rp = format!("{p}.block.{j}.block");
                rus.push((
                    l(&format!("{rp}.0.alpha"))?,
                    StridedConv::load(&f, &format!("{rp}.1"), dev, 1, 3 * d, d)?,
                    l(&format!("{rp}.2.alpha"))?,
                    StridedConv::load(&f, &format!("{rp}.3"), dev, 1, 0, 1)?,
                ));
            }
            blocks.push((rus, l(&format!("{p}.block.3.alpha"))?, StridedConv::load(&f, &format!("{p}.block.4"), dev, s, s.div_ceil(2), 1)?));
        }
        let mut pre = std::collections::HashMap::new();
        for k in f.names_with_prefix("pre_block.") {
            pre.insert(k.trim_start_matches("pre_block.").to_string(), l(&k)?);
        }
        Ok(Self {
            dev: dev.clone(),
            conv0: StridedConv::load(&f, "encoder.block.0", dev, 1, 3, 1)?,
            blocks,
            alpha_out: l("encoder.block.6.alpha")?,
            conv_out: StridedConv::load(&f, "encoder.block.7", dev, 1, 1, 1)?,
            pre,
            mean_proj: StridedConv::load(&f, "mean_proj", dev, 1, 0, 1)?,
            mean: l("latents_mean")?.reshape((1, 32, 1))?,
            std: l("latents_std")?.reshape((1, 32, 1))?,
        })
    }

    fn p(&self, k: &str) -> &Tensor {
        &self.pre[k]
    }

    fn lin(&self, x: &Tensor, w: &str, b: Option<&str>) -> Result<Tensor> {
        let y = x.broadcast_matmul(&self.p(w).t()?)?;
        Ok(match b {
            Some(b) => y.broadcast_add(self.p(b))?,
            None => y,
        })
    }

    /// Stereo waveform `[2, L]` at 32 kHz -> normalized latents `[1, 32, 2, T]` (T = ceil(L/800)).
    pub fn encode(&self, wave: &Tensor) -> Result<Tensor> {
        let (s, len) = wave.dims2()?;
        let t = len.div_ceil(800);
        let x = wave.to_device(&self.dev)?.to_dtype(DType::F32)?.pad_with_zeros(1, 0, t * 800 - len)?.reshape((s, 1, t * 800))?;
        let mut x = self.conv0.fwd(&x)?;
        for (rus, alpha, conv) in &self.blocks {
            for (a1, c1, a2, c2) in rus {
                let y = c2.fwd(&snake(&c1.fwd(&snake(&x, a1)?)?, a2)?)?;
                x = (y + &x)?;
            }
            x = conv.fwd(&snake(&x, alpha)?)?;
        }
        let x = self.conv_out.fwd(&snake(&x, &self.alpha_out)?)?; // [s, 2048, T]
        let x = x.transpose(1, 2)?.contiguous()?; // [s, T, 2048]
        let ln = |x: &Tensor, n: &str| crate::ops::layer_norm(x, Some(self.p(&format!("{n}.weight"))), Some(self.p(&format!("{n}.bias"))), 1e-5);
        // causal attention, mean over heads, pooled 256 -> 32
        let h = ln(&x, "norm1")?;
        let bias = Tensor::cat(&[self.p("attn.q_bias"), self.p("attn.zero_k_bias"), self.p("attn.v_bias")], 0)?;
        let qkv = h.broadcast_matmul(&self.p("attn.qkv.weight").t()?)?.broadcast_add(&bias)?; // [s, T, 6144]
        let (heads, hd) = (8usize, 256usize);
        let qkv = qkv.reshape((s, t, 3, heads, hd))?;
        let g = |i: usize| qkv.narrow(2, i, 1).and_then(|z| z.squeeze(2)).and_then(|z| z.contiguous());
        let a = crate::ops::attention_fallback(&g(0)?, &g(1)?, &g(2)?, 1.0 / (hd as f32).sqrt(), true)?; // [s, T, 8, 256]
        let a = a.mean(2)?; // [s, T, 256]
        let a = a.reshape((s, t, 32, hd / 32))?.mean(3)?; // adaptive avg pool 256 -> 32
        let a = self.lin(&a, "attn.proj.weight", Some("attn.proj.bias"))?;
        let x2 = (self.lin(&ln(&x, "norm3")?, "proj.weight", Some("proj.bias"))? + a)?;
        // GeGLU MLP on norm2(x): w2(gelu_tanh(w0(n)) * w1(n)), n = mlp.norm(norm2(x))
        let n = ln(&ln(&x2, "norm2")?, "mlp.norm")?;
        let g0 = self.lin(&n, "mlp.w0.weight", Some("mlp.w0.bias"))?.gelu()?;
        let g1 = self.lin(&n, "mlp.w1.weight", Some("mlp.w1.bias"))?;
        let m = self.lin(&(g0 * g1)?, "mlp.w2.weight", Some("mlp.w2.bias"))?;
        let x = (x2 + m)?.transpose(1, 2)?.contiguous()?; // [s, 32, T]
        let z = self.mean_proj.fwd(&x)?;
        let z = z.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?;
        Ok(z.unsqueeze(0)?.permute((0, 2, 1, 3))?.contiguous()?)
    }
}
