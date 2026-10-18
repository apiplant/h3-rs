//! MiniMax H3 Fun ControlNet (port of `comfy/ldm/minimax/controlnet.py` and `MiniMaxH3FunControlPatch` in
//! `comfy_extras/nodes_minimax_h3.py`).
//!
//! A ControlNet-Union branch of DiT blocks (same block layout as the base model) that reads a hint latent: a
//! control video (depth, pose, edges...) and/or an inpaint pair (visibility mask + masked source video). At the base
//! block where it branches off, the hint is projected into the packed stream's video rows, and each control block
//! adds its `after_proj` output to the base block it sits after, on every non-audio row.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::dit::{patchify_video, Block, ControlCond, Dit, PackedLayout, RunControl, Seg, StepCtx};
use crate::media;
use crate::vae_video::{VideoVae, IMAGENET_MEAN};
use crate::weights::{Held, Placement, QFormat, QLinear, SafeFile};

/// Input channels of the hint: 24 control-video latent + 1 visibility + 24 masked-source latent.
pub const CONTROL_IN_DIM: usize = 49;
const PATCH: usize = 4;
const BASE_BLOCKS: usize = 50;

/// What the checkpoint says about itself (ComfyUI reads the same safetensors metadata).
#[derive(Debug, Clone)]
pub struct ControlInfo {
    /// Base blocks each control block is added after (the first is 0).
    pub injection: Vec<usize>,
    /// v2 checkpoints hole the masked source at mid-gray after VAE normalization instead of black.
    pub inpaint_post_norm: bool,
    /// adaLN in the curve basis form (matches a `adaln_t_table` base checkpoint).
    pub adaln_basis: bool,
}

pub fn is_control_file(f: &SafeFile) -> bool {
    ["control_proj_in.weight", "control_blocks.0.adaln_proj.linear.weight", "control_blocks.0.after_proj.weight", "control_blocks.0.before_proj.weight", "control_blocks.0.attn.qkv_proj.weight", "control_blocks.0.attn.q_norm.weight", "control_blocks.0.mlp.fc1.weight"]
        .iter()
        .all(|k| f.has(k))
}

/// Base blocks the control blocks follow: from the checkpoint's `control_blocks_places` metadata, else spread
/// evenly over the 50 base blocks (v1 has 5, every 10th; v2 has 10, every 5th).
pub fn injection_layers(n: usize, places: Option<&str>) -> Result<Vec<usize>> {
    if n == 0 {
        bail!("the ControlNet has no control blocks");
    }
    let mut injection: Vec<usize> = (0..BASE_BLOCKS).step_by((BASE_BLOCKS / n).max(1)).collect();
    if let Some(places) = places {
        injection = serde_json::from_str(places).context("control_blocks_places metadata")?;
        if injection.len() != n {
            bail!("MiniMax H3 Fun control_blocks_places metadata does not match the checkpoint");
        }
    }
    if injection.len() != n || injection[0] != 0 || injection.windows(2).any(|w| w[0] >= w[1]) {
        bail!("MiniMax H3 Fun control injection layers must start at 0 and increase, got {injection:?}");
    }
    Ok(injection)
}

pub fn inspect(f: &SafeFile) -> Result<ControlInfo> {
    if !is_control_file(f) {
        bail!("{} is not a MiniMax H3 Fun ControlNet", f.path);
    }
    let n = (0..).take_while(|i| f.has(&format!("control_blocks.{i}.after_proj.weight"))).count();
    let injection = injection_layers(n, f.metadata.get("control_blocks_places").map(String::as_str))?;
    Ok(ControlInfo {
        injection,
        inpaint_post_norm: f.metadata.get("inpaint_masked_pixel_mode").map(String::as_str) == Some("post_norm"),
        adaln_basis: f.metadata.get("minimax_h3_fun_controlnet").map(String::as_str) == Some("adaln_basis"),
    })
}

pub struct Control {
    pub path: String,
    pub info: ControlInfo,
    pub injection: Vec<usize>,
    proj_in: QLinear,
    blocks: Vec<Block>,
    before: QLinear,
    after: Vec<QLinear>,
    dev: Device,
}

fn dense_f32(f: &Arc<SafeFile>, prefix: &str, dev: &Device) -> Result<QLinear> {
    let mut l = QLinear::load(f, prefix, Placement::Gpu, dev, DType::F32)?;
    if let QFormat::Dense = l.fmt {
        l.weight = Held::from_tensor(l.weight.get(dev)?.to_dtype(DType::F32)?);
    }
    Ok(l)
}

impl Control {
    /// Load the checkpoint. `place` is where the control blocks' weights wait (`Gpu` when the base DiT is fully
    /// resident, else the memory config's offload target).
    pub fn load(path: &str, dev: &Device, place: Placement, dtype: DType) -> Result<Self> {
        let f = SafeFile::open(path)?;
        let info = inspect(&f)?;
        let mut blocks = vec![];
        let mut after = vec![];
        for i in 0..info.injection.len() {
            let p = format!("control_blocks.{i}");
            let ad = if info.adaln_basis { dense_f32(&f, &format!("{p}.adaln_proj.linear"), dev)? } else { QLinear::load(&f, &format!("{p}.adaln_proj.linear"), place, dev, dtype)? };
            let ld = |n: &str| f.load_as(&format!("{p}.{n}"), dev, dtype);
            blocks.push(Block {
                norm1: ld("norm1.weight")?,
                norm2: ld("norm2.weight")?,
                q_norm: ld("attn.q_norm.weight")?,
                k_norm: ld("attn.k_norm.weight")?,
                qkv: QLinear::load(&f, &format!("{p}.attn.qkv_proj"), place, dev, dtype)?,
                out: QLinear::load(&f, &format!("{p}.attn.out_proj"), place, dev, dtype)?,
                fc1: QLinear::load(&f, &format!("{p}.mlp.fc1"), place, dev, dtype)?,
                fc2: QLinear::load(&f, &format!("{p}.mlp.fc2"), place, dev, dtype)?,
                adaln: ad,
            });
            if crate::weights::int8_enabled() {
                blocks[i].fc1.interleave_swiglu()?;
            }
            after.push(QLinear::load(&f, &format!("{p}.after_proj"), Placement::Gpu, dev, dtype)?);
        }
        Ok(Self {
            path: path.to_string(),
            injection: info.injection.clone(),
            info,
            proj_in: dense_f32(&f, "control_proj_in", dev)?,
            blocks,
            before: QLinear::load(&f, "control_blocks.0.before_proj", Placement::Gpu, dev, dtype)?,
            after,
            dev: dev.clone(),
        })
    }

    /// Per-run rows: the hint latent as control patches for the target video rows (zeros for keyframe and reference
    /// rows), plus the per-row strength (0 on audio rows).
    pub(crate) fn prepare_run(&self, cond: &ControlCond, layout: &PackedLayout) -> Result<RunControl> {
        let patch_dim = CONTROL_IN_DIM * PATCH;
        let (_, c, _, h, w) = cond.latent.dims5()?;
        if c > CONTROL_IN_DIM || h % 2 != 0 || w % 2 != 0 {
            bail!("control hint {:?} does not fit the ControlNet's {CONTROL_IN_DIM}-channel, 2x2-patch input", cond.latent.dims());
        }
        let target = patchify_video(&cond.latent.to_device(&self.dev)?.to_dtype(DType::F32)?)?;
        let target = if target.dim(1)? < patch_dim { target.pad_with_zeros(1, 0, patch_dim - target.dim(1)?)? } else { target };
        // video-kind rows in packed order: keyframe / reference rows first, the target video last
        let (va, vb) = layout.find(Seg::Video);
        let before_target: usize = layout.segments.iter().filter(|s| matches!(s.2, Seg::Cond | Seg::RefImg)).map(|s| s.1 - s.0).sum();
        if target.dim(0)? != vb - va {
            bail!("control hint has {} rows, the video {}", target.dim(0)?, vb - va);
        }
        let rows = if before_target > 0 { Tensor::cat(&[&Tensor::zeros((before_target, patch_dim), DType::F32, &self.dev)?, &target], 0)? } else { target };
        let mut scale = vec![cond.strength as f32; layout.seq_len];
        for &(a, b, k) in &layout.segments {
            if matches!(k, Seg::Audio | Seg::CondAudio | Seg::RefAudio) {
                scale[a..b].fill(0.0);
            }
        }
        Ok(RunControl { cond: cond.clone(), rows, scale: Tensor::from_vec(scale, (layout.seq_len, 1), &self.dev)? })
    }

    /// The control stream entering the first block: the base stream with its video rows replaced by the projected
    /// hint, through `before_proj`, plus the base stream.
    pub(crate) fn init_stream(&self, h: &Tensor, rows: &Tensor, layout: &PackedLayout) -> Result<Tensor> {
        let proj = self.proj_in.forward(rows)?.to_dtype(h.dtype())?;
        let mut parts = vec![];
        let mut off = 0;
        for &(a, b, k) in &layout.segments {
            if matches!(k, Seg::Cond | Seg::RefImg | Seg::Video) {
                parts.push(proj.narrow(0, off, b - a)?);
                off += b - a;
            } else {
                parts.push(h.narrow(0, a, b - a)?);
            }
        }
        let c = Tensor::cat(&parts, 0)?;
        Ok((self.before.forward(&c)?.to_dtype(h.dtype())? + h)?)
    }

    /// Control block `ci` on the stream; returns the new stream and the residual for the base stream.
    pub(crate) fn step(&self, dit: &Dit, ci: usize, c: Tensor, cx: &StepCtx) -> Result<(Tensor, Tensor)> {
        let b = &self.blocks[ci];
        let raws = [b.qkv.fetch(&self.dev)?, b.out.fetch(&self.dev)?, b.fc1.fetch(&self.dev)?, b.fc2.fetch(&self.dev)?];
        let c = dit.run_block(b, raws, c, cx)?;
        let skip = self.after[ci].forward(&c)?;
        Ok((c, skip))
    }

    /// Check the checkpoint's adaLN form against the base model's (`width` = base timestep-embedding width).
    pub fn check_base(&self, base_t_dim: usize, base_curve: bool) -> Result<()> {
        let w = self.blocks[0].adaln.in_features;
        if w != base_t_dim || self.info.adaln_basis != base_curve {
            bail!(
                "MiniMax H3 controlnet adaln width {w} does not match the base model's timestep embedding width {base_t_dim}: the controlnet and base checkpoint use different adaln forms (curve basis vs full), convert the controlnet to match the base model."
            );
        }
        Ok(())
    }
}

/// Pixel inputs of a hint (CPU tensors).
pub struct HintInputs<'a> {
    /// Control video frames `[F, h, w, 3]` at the canvas size, if any.
    pub video: Option<&'a Tensor>,
    /// Mask `[F, H, W]` (white = regenerate, as read by [`media::load_mask`]), any size.
    pub mask: Option<&'a Tensor>,
    /// Source video `[F, h, w, 3]` at the canvas size behind the mask, if any (black where unknown).
    pub source: Option<&'a Tensor>,
}

/// The hint latent for a canvas: `target` is the video latent shape `[1, 24, T, H, W]`.
pub fn encode_hint(vae: &VideoVae, post_norm: bool, target: [usize; 5], inputs: &HintInputs) -> Result<Tensor> {
    let (lt, lh, lw) = (target[2], target[3], target[4]);
    let frame_count = (lt.saturating_sub(2) / 5) * 17 + 5;
    let (w, h) = (lw * 16, lh * 16);
    let fit = |frames: &Tensor| -> Result<Tensor> {
        // repeat the last frame to the length, like ComfyUI's `_fit_frames`
        let n = frames.dim(0)?;
        let idx: Vec<u32> = (0..frame_count).map(|i| i.min(n - 1) as u32).collect();
        Ok(frames.index_select(&Tensor::from_vec(idx, frame_count, frames.device())?, 0)?)
    };
    let encode = |frames: &Tensor| -> Result<Tensor> {
        let z = vae.encode_frames(frames)?;
        if z.dims() != &target[..] {
            bail!("MiniMax H3 Fun VAE output shape {:?} does not match the target {:?}", z.dims(), target);
        }
        Ok(z.to_dtype(DType::F32)?)
    };
    let mut hint = match inputs.video {
        Some(v) => Some(encode(&fit(v)?.contiguous()?)?),
        None => None,
    };
    if let Some(mask) = inputs.mask {
        let (mf, mh, mw) = mask.dims3()?;
        // binarize, fit to the canvas and the length, binarize again: visibility is 1 where the source stays
        let m = mask.gt(0.5f64)?.to_dtype(DType::F32)?.reshape((mf, 1, mh, mw))?;
        let m = fit(&m)?;
        let m = media::resize_bilinear_crop(&m, w, h)?;
        let vis = (m.gt(0.5f64)?.to_dtype(DType::F32)?.neg()? + 1.0)?; // [F, 1, h, w]
        let vis_hwc = vis.permute((0, 2, 3, 1))?.contiguous()?; // [F, h, w, 1]
        let source = match inputs.source {
            Some(s) => fit(s)?,
            None => Tensor::zeros((frame_count, h, w, 3), DType::F32, vis.device())?,
        };
        let mut masked = source.broadcast_mul(&vis_hwc)?;
        if post_norm {
            // the pixel the VAE normalizes to zero, i.e. holes at mid-gray instead of black
            let mean = Tensor::new(&IMAGENET_MEAN, masked.device())?.reshape((1, 1, 1, 3))?;
            masked = (masked + (vis_hwc.neg()? + 1.0)?.broadcast_mul(&mean)?)?;
        }
        let masked_latent = encode(&masked.contiguous()?)?;
        let hint_v = match hint.take() {
            Some(h) => h,
            None => masked_latent.zeros_like()?,
        };
        let vis_vec = vis.flatten_all()?.to_vec1::<f32>()?;
        let vis_lat = media::resize_trilinear(&vis_vec, (frame_count, h, w), (lt, lh, lw));
        let vis_lat = Tensor::from_vec(vis_lat, (1, 1, lt, lh, lw), masked_latent.device())?;
        hint = Some(Tensor::cat(&[&hint_v, &vis_lat, &masked_latent], 1)?);
    }
    hint.context("a ControlNet needs a control video or a mask")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injection_layers_follow_the_checkpoint() {
        assert_eq!(injection_layers(5, None).unwrap(), vec![0, 10, 20, 30, 40]);
        assert_eq!(injection_layers(10, None).unwrap(), vec![0, 5, 10, 15, 20, 25, 30, 35, 40, 45]);
        assert_eq!(injection_layers(3, Some("[0, 7, 31]")).unwrap(), vec![0, 7, 31]);
        assert!(injection_layers(3, Some("[0, 7]")).is_err());
        assert!(injection_layers(2, Some("[3, 9]")).is_err());
        assert!(injection_layers(0, None).is_err());
    }
}
