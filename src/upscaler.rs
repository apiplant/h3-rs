//! MiniMax H3 latent upscaler (LBH-123-AI/Minimax_h3_latent_Upscaler, "3D conv v1"): a learned
//! spatial upscaler for the video VAE's 24-channel latents. Port of the ComfyUI node's
//! `LatentResizer3D`: conv_in -> 12 x (ResBlockEmb3D [+ TemporalConv every 2nd]) -> trilinear resize
//! (H and W only) -> 12 more blocks -> GroupNorm, SiLU, conv_out, conditioned on `scale - 1`.
//!
//! Tensors are frames-first `[T, C, H, W]` internally so every 3x3x3 conv is three candle conv2d taps over
//! the frame batch; GroupNorm + SiLU, the temporal depthwise conv and the resize are NVRTC kernels
//! (`kernels.rs`). The network runs in f16 (the checkpoint's dtype), statistics in f32/f64.

use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::kernels;
use crate::weights::SafeFile;

/// Latent channels of the video VAE; the per-channel statistics below normalize them for the network.
pub const CHANNELS: usize = 24;
/// Pixels per latent cell (the video VAE's spatial stride).
pub const LATENT_STRIDE: usize = 16;
/// The checkpoint was trained on scales 1.0x - 4.0x.
pub const MAX_SCALE: f64 = 4.0;

pub const MEAN: [f32; CHANNELS] = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
];
pub const STD: [f32; CHANNELS] = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
];

/// Frames per temporal chunk of the reference node; longer videos run in overlapping chunks.
const CHUNK: usize = 32;
const GROUPS: usize = 32;
const EPS: f32 = 1e-5;
/// im2col budget of one conv2d call.
const IM2COL_BYTES: usize = 384 << 20;

struct Conv3 {
    /// One `[Co, Ci, k, k]` kernel per temporal tap.
    taps: Vec<Tensor>,
    bias: Tensor, // [1, Co, 1, 1]
    pad: usize,
}

struct Norm {
    w: Tensor, // f32 [C]
    b: Tensor,
}

struct Lin {
    w: Tensor, // f32 [out, in]
    b: Tensor,
}

enum Block {
    Res { n1: Norm, c1: Conv3, emb: Lin, n2: Norm, c2: Conv3 },
    Temporal { norm: Norm, dw_w: Tensor, dw_b: Tensor, pw_w: Tensor, pw_b: Tensor },
}

/// Per-forward conditioning of a residual block: `out_norm` affine with the scale embedding folded in.
struct Mod {
    gamma: Tensor,
    beta: Tensor,
}

pub struct LatentUpscaler {
    conv_in: Conv3,
    embed: (Lin, Lin),
    in_blocks: Vec<Block>,
    out_blocks: Vec<Block>,
    norm_out: Norm,
    conv_out: Conv3,
    dev: Device,
    dtype: DType,
    temporal_kernel: usize,
}

fn conv3(f: &SafeFile, p: &str, dev: &Device, dt: DType) -> Result<Conv3> {
    let w = f.load_as(&format!("{p}.weight"), dev, dt)?;
    let (co, _, kt, k, k2) = w.dims5()?;
    if k != k2 || !(kt == 1 || kt == 3) || k % 2 == 0 {
        bail!("{p}: unsupported kernel {kt}x{k}x{k2}");
    }
    let taps = (0..kt).map(|i| Ok(w.narrow(2, i, 1)?.squeeze(2)?.contiguous()?)).collect::<Result<Vec<_>>>()?;
    let bias = f.load_as(&format!("{p}.bias"), dev, dt)?.reshape((1, co, 1, 1))?;
    Ok(Conv3 { taps, bias, pad: k / 2 })
}

fn norm(f: &SafeFile, p: &str, dev: &Device) -> Result<Norm> {
    Ok(Norm { w: f.load_as(&format!("{p}.weight"), dev, DType::F32)?, b: f.load_as(&format!("{p}.bias"), dev, DType::F32)? })
}

fn lin(f: &SafeFile, p: &str, dev: &Device) -> Result<Lin> {
    Ok(Lin { w: f.load_as(&format!("{p}.weight"), dev, DType::F32)?, b: f.load_as(&format!("{p}.bias"), dev, DType::F32)? })
}

impl Lin {
    /// `x [1, in]` -> `[1, out]`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.w.t()?)?.broadcast_add(&self.b)?)
    }
}

fn load_blocks(f: &SafeFile, prefix: &str, dev: &Device, dt: DType) -> Result<Vec<Block>> {
    let mut blocks = vec![];
    for i in 0.. {
        let p = format!("{prefix}.{i}");
        if f.has(&format!("{p}.in_layers.0.weight")) {
            blocks.push(Block::Res {
                n1: norm(f, &format!("{p}.in_layers.0"), dev)?,
                c1: conv3(f, &format!("{p}.in_layers.2"), dev, dt)?,
                emb: lin(f, &format!("{p}.emb_layers.1"), dev)?,
                n2: norm(f, &format!("{p}.out_norm"), dev)?,
                c2: conv3(f, &format!("{p}.out_layers.2"), dev, dt)?,
            });
        } else if f.has(&format!("{p}.dwconv.weight")) {
            let dw = f.load_as(&format!("{p}.dwconv.weight"), dev, DType::F32)?; // [C, 1, K, 1, 1]
            let (c, _, k, _, _) = dw.dims5()?;
            let pw = f.load_as(&format!("{p}.pwconv.weight"), dev, dt)?; // [C, C, 1, 1, 1]
            blocks.push(Block::Temporal {
                norm: norm(f, &format!("{p}.norm"), dev)?,
                dw_w: dw.reshape((c, k))?,
                dw_b: f.load_as(&format!("{p}.dwconv.bias"), dev, DType::F32)?,
                pw_w: pw.reshape((c, c))?.unsqueeze(0)?,
                pw_b: f.load_as(&format!("{p}.pwconv.bias"), dev, dt)?.reshape((1, c, 1))?,
            });
        } else {
            break;
        }
    }
    if blocks.is_empty() {
        bail!("{prefix}: no blocks found, is this a MiniMax H3 latent upscaler checkpoint?");
    }
    Ok(blocks)
}

impl Conv3 {
    /// 3D conv (zero padded) of frames-first `x [T, Ci, H, W]`: each temporal tap is a conv2d over all
    /// frames, shifted onto its neighbours.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (t, ci, h, w) = x.dims4()?;
        let kt = self.taps.len();
        let mid = kt / 2;
        let mut out = self.conv2d(x, &self.taps[mid], ci, h, w)?;
        for (dt, tap) in self.taps.iter().enumerate() {
            if dt == mid || t == 1 {
                continue;
            }
            // out[t] += conv(x[t + dt - mid]) with zero frames outside the clip
            let y = if dt < mid { self.conv2d(&x.narrow(0, 0, t - 1)?, tap, ci, h, w)? } else { self.conv2d(&x.narrow(0, 1, t - 1)?, tap, ci, h, w)? };
            out = if dt < mid {
                Tensor::cat(&[&out.narrow(0, 0, 1)?, &(out.narrow(0, 1, t - 1)? + y)?], 0)?
            } else {
                Tensor::cat(&[&(out.narrow(0, 0, t - 1)? + y)?, &out.narrow(0, t - 1, 1)?], 0)?
            };
        }
        Ok(out.broadcast_add(&self.bias)?)
    }

    fn conv2d(&self, x: &Tensor, tap: &Tensor, ci: usize, h: usize, w: usize) -> Result<Tensor> {
        let n = x.dim(0)?;
        let k = tap.dim(2)?;
        let per_frame = (h * w * ci * k * k * x.dtype().size_in_bytes()).max(1);
        let chunk = (IM2COL_BYTES / per_frame).clamp(1, n);
        let mut outs = vec![];
        let mut s = 0;
        while s < n {
            let m = chunk.min(n - s);
            outs.push(x.narrow(0, s, m)?.contiguous()?.conv2d(tap, self.pad, 1, 1, 1)?);
            s += m;
        }
        let y = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };
        Ok(y)
    }
}

impl LatentUpscaler {
    /// Open a `.safetensors` checkpoint (fp16 / bf16 / fp32) and upload it as f16.
    pub fn load(path: impl AsRef<Path>, dev: &Device) -> Result<Self> {
        Self::load_as(path, dev, DType::F16)
    }

    /// Like [`load`](Self::load) with an explicit compute dtype (`F16`, `BF16` or `F32`).
    pub fn load_as(path: impl AsRef<Path>, dev: &Device, dt: DType) -> Result<Self> {
        let path = path.as_ref();
        if path.extension().is_some_and(|e| e == "pth") {
            bail!("{}: only .safetensors upscaler checkpoints are supported (the repository ships fp16 and bf16 ones)", path.display());
        }
        let f = SafeFile::open(path).with_context(|| format!("opening {}", path.display()))?;
        let conv_in = conv3(&f, "conv_in", dev, dt)?;
        if conv_in.taps[0].dim(1)? != CHANNELS {
            bail!("{}: expects {} latent channels, got {}", path.display(), CHANNELS, conv_in.taps[0].dim(1)?);
        }
        let in_blocks = load_blocks(&f, "in_blocks", dev, dt)?;
        let out_blocks = load_blocks(&f, "out_blocks", dev, dt)?;
        let temporal_kernel = in_blocks
            .iter()
            .find_map(|b| match b {
                Block::Temporal { dw_w, .. } => dw_w.dim(1).ok(),
                _ => None,
            })
            .unwrap_or(0);
        Ok(Self {
            conv_in,
            embed: (lin(&f, "embed.0", dev)?, lin(&f, "embed.2", dev)?),
            in_blocks,
            out_blocks,
            norm_out: norm(&f, "norm_out", dev)?,
            conv_out: conv3(&f, "conv_out", dev, dt)?,
            dev: dev.clone(),
            dtype: dt,
            temporal_kernel,
        })
    }

    /// Effective scale of a latent resize, as the reference node computes it for a target size.
    pub fn scale_for(from: (usize, usize), to: (usize, usize)) -> f64 {
        (to.1 as f64 / from.1 as f64 + to.0 as f64 / from.0 as f64) / 2.0
    }

    /// Upscale a video latent `[1, 24, T, H, W]` (f32, the VAE's raw latent) to `h_out x w_out` latent cells.
    /// `scale` conditions the network (1.0 - 4.0); `None` derives it from the sizes.
    pub fn upscale(&self, z: &Tensor, h_out: usize, w_out: usize, scale: Option<f64>) -> Result<Tensor> {
        let (b, c, t, h, w) = z.dims5()?;
        if b != 1 || c != CHANNELS {
            bail!("latent must be [1, {CHANNELS}, T, H, W], got {:?}", z.dims());
        }
        if (h_out, w_out) == (h, w) {
            return Ok(z.clone());
        }
        if h_out < h || w_out < w {
            bail!("the upscaler only upscales: latent {w}x{h} -> {w_out}x{h_out}");
        }
        let scale = scale.unwrap_or_else(|| Self::scale_for((h, w), (h_out, w_out)));
        if !(1.0..=MAX_SCALE + 1e-6).contains(&scale) {
            bail!("upscale factor {scale:.2} is outside the model's trained 1.0 - {MAX_SCALE} range");
        }
        let dev = &self.dev;
        let stat = |v: &[f32]| Tensor::from_slice(v, (1, CHANNELS, 1, 1), dev);
        let (mean, std) = (stat(&MEAN)?, stat(&STD)?);
        let x = z.to_device(dev)?.to_dtype(DType::F32)?.squeeze(0)?.permute((1, 0, 2, 3))?.contiguous()?; // [T, 24, H, W]
        let x = x.broadcast_sub(&mean)?.broadcast_div(&std)?.to_dtype(self.dtype)?;
        let mods = self.conditioning(scale)?;
        let y = if t <= CHUNK { self.forward_seg(&x, &mods, h_out, w_out)?.to_dtype(DType::F32)? } else { self.forward_chunked(&x, &mods, h_out, w_out)? };
        let y = y.broadcast_mul(&std)?.broadcast_add(&mean)?;
        Ok(y.permute((1, 0, 2, 3))?.unsqueeze(0)?.contiguous()?)
    }

    /// `(1 + scale, shift)` of every residual block for this upscale factor, folded into its `out_norm`.
    fn conditioning(&self, scale: f64) -> Result<(Vec<Mod>, Vec<Mod>)> {
        let s = Tensor::from_slice(&[(scale - 1.0) as f32], (1, 1), &self.dev)?;
        let emb = self.embed.1.forward(&crate::ops::silu(&self.embed.0.forward(&s)?)?)?;
        let emb = crate::ops::silu(&emb)?;
        let build = |blocks: &[Block]| -> Result<Vec<Mod>> {
            let mut v = vec![];
            for b in blocks {
                if let Block::Res { emb: l, n2, .. } = b {
                    let o = l.forward(&emb)?.squeeze(0)?; // [2C]
                    let c = n2.w.dim(0)?;
                    let (sc, sh) = (o.narrow(0, 0, c)?, o.narrow(0, c, c)?);
                    let one_plus = (sc + 1.0)?;
                    v.push(Mod { gamma: (&n2.w * &one_plus)?, beta: ((&n2.b * &one_plus)? + sh)? });
                }
            }
            Ok(v)
        };
        Ok((build(&self.in_blocks)?, build(&self.out_blocks)?))
    }

    fn run_blocks(&self, blocks: &[Block], mods: &[Mod], mut h: Tensor) -> Result<Tensor> {
        let mut mi = 0;
        for b in blocks {
            h = match b {
                Block::Res { n1, c1, c2, .. } => {
                    let a = c1.forward(&kernels::group_norm_silu(&h, &n1.w, &n1.b, GROUPS, EPS)?)?;
                    let m = &mods[mi];
                    mi += 1;
                    let a = c2.forward(&kernels::group_norm_silu(&a, &m.gamma, &m.beta, GROUPS, EPS)?)?;
                    (h + a)?
                }
                Block::Temporal { norm, dw_w, dw_b, pw_w, pw_b } => {
                    let (t, c, hh, ww) = h.dims4()?;
                    let a = kernels::dw_time(&kernels::group_norm_silu(&h, &norm.w, &norm.b, GROUPS, EPS)?, dw_w, dw_b)?;
                    let a = pw_w.broadcast_matmul(&a.reshape((t, c, hh * ww))?)?.broadcast_add(pw_b)?.reshape((t, c, hh, ww))?;
                    (h + a)?
                }
            };
        }
        Ok(h)
    }

    /// The network on one clip segment `[T, 24, H, W]` (normalized, f16).
    fn forward_seg(&self, x: &Tensor, mods: &(Vec<Mod>, Vec<Mod>), h_out: usize, w_out: usize) -> Result<Tensor> {
        let h = self.conv_in.forward(x)?;
        let h = self.run_blocks(&self.in_blocks, &mods.0, h)?;
        let h = kernels::bilinear_hw(&h, h_out, w_out)?;
        let h = self.run_blocks(&self.out_blocks, &mods.1, h)?;
        let h = kernels::group_norm_silu(&h, &self.norm_out.w, &self.norm_out.b, GROUPS, EPS)?;
        self.conv_out.forward(&h)
    }

    /// Long clips: segments of `CHUNK` frames with `overlap` frames of context (replicate padded at the
    /// clip ends) on each side, cross-faded where neighbours overlap. Same scheme as the reference node.
    fn forward_chunked(&self, x: &Tensor, mods: &(Vec<Mod>, Vec<Mod>), h_out: usize, w_out: usize) -> Result<Tensor> {
        let t = x.dim(0)?;
        let overlap = self.temporal_kernel;
        let first = x.narrow(0, 0, 1)?.repeat((overlap, 1, 1, 1))?;
        let last = x.narrow(0, t - 1, 1)?.repeat((overlap, 1, 1, 1))?;
        let padded = Tensor::cat(&[&first, x, &last], 0)?;
        let mut out = Tensor::zeros((t, CHANNELS, h_out, w_out), DType::F32, &self.dev)?;
        let mut weight_sum = vec![0f32; t];
        let mut start = 0;
        while start < t {
            let (seg_start, seg_end) = (start, (start + CHUNK).min(t));
            let out_start = seg_start.saturating_sub(overlap);
            let out_end = (seg_end + overlap).min(t);
            let lo = out_start.saturating_sub(overlap);
            let hi = (out_end + overlap).min(t + 2 * overlap);
            let seg = padded.narrow(0, lo, hi - lo)?.contiguous()?;
            let seg_out = self.forward_seg(&seg, mods, h_out, w_out)?;
            let s0 = out_start + overlap - lo;
            let n = out_end - out_start;
            let valid = seg_out.narrow(0, s0, n)?.to_dtype(DType::F32)?;
            let mut wgt = vec![1f32; n];
            if seg_start > out_start {
                let blend = seg_start - out_start;
                for (i, v) in wgt.iter_mut().take(blend).enumerate() {
                    *v = (i + 1) as f32 / (blend + 1) as f32;
                }
            }
            if out_end > seg_end {
                let blend = out_end - seg_end;
                for (i, v) in wgt.iter_mut().skip(n - blend).enumerate() {
                    *v = (blend - i) as f32 / (blend + 1) as f32;
                }
            }
            for (i, v) in wgt.iter().enumerate() {
                weight_sum[out_start + i] += v;
            }
            let wt = Tensor::from_vec(wgt, (n, 1, 1, 1), &self.dev)?;
            let merged = (out.narrow(0, out_start, n)? + valid.broadcast_mul(&wt)?)?;
            out = out.slice_assign(&[out_start..out_end, 0..CHANNELS, 0..h_out, 0..w_out], &merged)?;
            start += CHUNK;
        }
        let ws = Tensor::from_vec(weight_sum.into_iter().map(|v| v.max(1e-8)).collect::<Vec<_>>(), (t, 1, 1, 1), &self.dev)?;
        Ok(out.broadcast_div(&ws)?)
    }
}

/// Upscaled latent size for a pixel `target` canvas `(w, h)`: latent cells, at least the source size.
pub fn target_cells(target: (usize, usize), from: (usize, usize)) -> (usize, usize) {
    ((target.1 / LATENT_STRIDE).max(from.0), (target.0 / LATENT_STRIDE).max(from.1))
}

impl LatentUpscaler {
    /// Debugging aid for parity checks: intermediate activations (frames-first, f32) of an fp32-normalized
    /// input `x [T, 24, H, W]`: after `conv_in`, after the first three input blocks, after all input blocks,
    /// after the resize, after all output blocks.
    #[doc(hidden)]
    pub fn stages(&self, x: &Tensor, scale: f64, h_out: usize, w_out: usize) -> Result<Vec<(String, Tensor)>> {
        let mods = self.conditioning(scale)?;
        let mut v = vec![];
        let mut h = self.conv_in.forward(&x.to_dtype(self.dtype)?)?;
        v.push(("conv_in".to_string(), h.clone()));
        for i in 0..self.in_blocks.len() {
            let mi = self.in_blocks[..i].iter().filter(|b| matches!(b, Block::Res { .. })).count();
            h = self.run_blocks(&self.in_blocks[i..=i], &mods.0[mi..], h)?;
            if i < 3 {
                v.push((format!("in{i}"), h.clone()));
            }
        }
        v.push(("in_all".into(), h.clone()));
        h = kernels::bilinear_hw(&h, h_out, w_out)?;
        v.push(("interp".into(), h.clone()));
        h = self.run_blocks(&self.out_blocks, &mods.1, h)?;
        v.push(("out_all".into(), h.clone()));
        Ok(v)
    }
}
