//! MiniMax H3 audio-video DiT (port of `comfy/ldm/minimax/model.py`).
//!
//! Single-stream packed transformer over `[text | cond rows | audio | video]`, conditioned on
//! Qwen3-VL hidden states, denoising 24ch video latents (patch 1x2x2) and 32ch stereo audio
//! latents (40 Hz) jointly. Video and audio run on different flow shifts (12 / 3); the sampler
//! carries the audio scaled onto the video schedule and [`Dit::forward`] converts in and out.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};

use crate::config::{resolve_resident, MemoryConfig};
use crate::kernels;
use crate::lora::Lora;
use crate::ops;
use crate::weights::{Held, Placement, QFormat, QLinear, SafeFile};

pub const FRAME_PER_TOKEN: [usize; 5] = [1, 4, 4, 4, 4];
pub const FRAME_RESCALE: f64 = 5.0 / 3.0;
pub const VISUAL_COND_TIMESTEP: f64 = 0.999;
pub const AUDIO_COND_TIMESTEP: f64 = 1.0;

pub fn time_shift_sigma(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * base / (1.0 + (to_shift - 1.0) * base)
}

// ----------------------------------------------------------------------------------------------
// Packed layout
// ----------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seg {
    Text,
    Cond,
    CondAudio,
    RefImg,
    RefAudio,
    Audio,
    Video,
}

impl Seg {
    fn tag(self) -> u32 {
        match self {
            Seg::Text => 1,
            Seg::Video | Seg::Cond | Seg::RefImg => 0,
            Seg::Audio | Seg::CondAudio | Seg::RefAudio => 2,
        }
    }
    fn is_video(self) -> bool {
        matches!(self, Seg::Cond | Seg::RefImg | Seg::Video)
    }
}

/// Keyframe condition shape info (fl2va / AddGuide).
#[derive(Debug, Clone)]
pub struct KeyframeSpec {
    pub frame_index: usize,
    pub video_t: Option<usize>,
    pub audio_t: Option<usize>,
}

/// Reference block shape info (ref2va).
#[derive(Debug, Clone)]
pub enum RefSpec {
    Image { lh: usize, lw: usize },
    Audio { rt: usize },
    Video { lt: usize, lh: usize, lw: usize, rt: usize },
}

impl RefSpec {
    fn t_span(&self) -> f64 {
        match self {
            RefSpec::Image { .. } => 1.0,
            RefSpec::Audio { rt } => *rt as f64,
            RefSpec::Video { lt, rt, .. } => (*rt as f64).max(video_t_spans(*lt).iter().sum()),
        }
    }
}

fn axis_from_sqrt_area(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / patch;
    (0..n).map(|k| (k as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0).collect()
}

/// Area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows + the w axis.
fn frame_grid(h: usize, w: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let area = ((h * w) as f64).sqrt();
    let hh = axis_from_sqrt_area(h, 2, area);
    let ww = axis_from_sqrt_area(w, 2, area);
    let mut rows = Vec::with_capacity(hh.len() * ww.len());
    for &a in &hh {
        for &b in &ww {
            rows.push([a, b]);
        }
    }
    (rows, ww)
}

fn video_t_spans(n: usize) -> Vec<f64> {
    (0..n).map(|k| FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] as f64).collect()
}

fn video_t_grid(n: usize, origin: f64) -> Vec<f64> {
    let spans = video_t_spans(n);
    let mut out = Vec::with_capacity(n);
    let mut acc = 0.0;
    for s in spans {
        out.push(origin + acc);
        acc += s;
    }
    out
}

fn audio_grid(cursor: f64, t: usize, w_low: f64, w_high: f64) -> Vec<[f64; 3]> {
    let mut g = Vec::with_capacity(2 * t);
    for ch in 0..2 {
        for k in 0..t {
            g.push([cursor + k as f64, 0.0, if ch == 0 { w_low } else { w_high }]);
        }
    }
    g
}

fn video_grid(vt: usize, frame: &[[f64; 2]], cursor: f64) -> Vec<[f64; 3]> {
    let tg = video_t_grid(vt, cursor);
    let mut g = Vec::with_capacity(vt * frame.len());
    for t in tg {
        for f in frame {
            g.push([t, f[0], f[1]]);
        }
    }
    g
}

#[derive(Debug, Clone)]
pub struct PackedLayout {
    pub seq_len: usize,
    pub positions: Vec<[f64; 3]>,
    /// Contiguous `(start, stop, kind)` segments covering the sequence.
    pub segments: Vec<(usize, usize, Seg)>,
}

impl PackedLayout {
    pub fn new(
        text_len: usize,
        latent_t: usize,
        latent_h: usize,
        latent_w: usize,
        audio_t: usize,
        keyframes: &[KeyframeSpec],
        refs: &[RefSpec],
    ) -> Self {
        let (frame, w_grid) = frame_grid(latent_h, latent_w);
        let mut segs: Vec<(Seg, usize)> = vec![(Seg::Text, text_len)];
        let mut pos: Vec<[f64; 3]> = (0..text_len).map(|k| [k as f64, 0.0, 0.0]).collect();
        let target_w = (w_grid[0], *w_grid.last().unwrap());
        let mut cursor = text_len as f64 + refs.iter().map(|r| r.t_span()).sum::<f64>();

        for kf in keyframes {
            let cond_t = cursor + FRAME_RESCALE * kf.frame_index as f64;
            if let Some(vt) = kf.video_t {
                segs.push((Seg::Cond, vt * frame.len()));
                pos.extend(video_grid(vt, &frame, cond_t));
            }
            if let Some(rt) = kf.audio_t {
                segs.push((Seg::CondAudio, rt * 2));
                pos.extend(audio_grid(cond_t, rt, target_w.0, target_w.1));
            }
        }

        if !refs.is_empty() {
            let mut c = text_len as f64;
            for r in refs {
                match *r {
                    RefSpec::Image { lh, lw } => {
                        let (rf, _) = frame_grid(lh, lw);
                        segs.push((Seg::RefImg, rf.len()));
                        pos.extend(rf.iter().map(|f| [c, f[0], f[1]]));
                        c += 1.0;
                    }
                    RefSpec::Audio { rt } => {
                        if rt > 0 {
                            segs.push((Seg::RefAudio, rt * 2));
                            pos.extend(audio_grid(c, rt, target_w.0, target_w.1));
                        }
                        c += rt as f64;
                    }
                    RefSpec::Video { lt, lh, lw, rt } => {
                        let (rf, rw) = frame_grid(lh, lw);
                        if rt > 0 {
                            segs.push((Seg::RefAudio, rt * 2));
                            pos.extend(audio_grid(c, rt, rw[0], *rw.last().unwrap()));
                        }
                        segs.push((Seg::RefImg, lt * rf.len()));
                        pos.extend(video_grid(lt, &rf, c));
                        c += (rt as f64).max(video_t_spans(lt).iter().sum());
                    }
                }
            }
            cursor = c;
        }

        segs.push((Seg::Audio, audio_t * 2));
        pos.extend(audio_grid(cursor, audio_t, target_w.0, target_w.1));
        segs.push((Seg::Video, latent_t * frame.len()));
        pos.extend(video_grid(latent_t, &frame, cursor));

        let mut segments = vec![];
        let mut off = 0;
        for (k, n) in segs {
            segments.push((off, off + n, k));
            off += n;
        }
        Self { seq_len: off, positions: pos, segments }
    }

    pub fn find(&self, kind: Seg) -> (usize, usize) {
        let s = self.segments.iter().find(|s| s.2 == kind).unwrap();
        (s.0, s.1)
    }
}

// ----------------------------------------------------------------------------------------------
// Patchify helpers
// ----------------------------------------------------------------------------------------------

/// `[1, C, T, H, W]` -> `[T*h*w, C*2*2]` (patch 1x2x2).
pub fn patchify_video(x: &Tensor) -> Result<Tensor> {
    let (b, c, t, h, w) = x.dims5()?;
    let x = x.reshape(&[b, c, t, 1, h / 2, 2, w / 2, 2][..])?;
    let x = x.permute(vec![0, 2, 4, 6, 1, 3, 5, 7])?.contiguous()?;
    Ok(x.reshape((b * t * (h / 2) * (w / 2), c * 4))?)
}

pub fn unpatchify_video(rows: &Tensor, t: usize, h: usize, w: usize, c: usize) -> Result<Tensor> {
    let x = rows.reshape(&[1, t, h, w, c, 1, 2, 2][..])?;
    let x = x.permute(vec![0, 4, 1, 5, 2, 6, 3, 7])?.contiguous()?;
    Ok(x.reshape((1, c, t, h * 2, w * 2))?)
}

/// `[1, C, 2, T]` -> `[2T, C]` channel-major.
pub fn pack_audio(x: &Tensor) -> Result<Tensor> {
    let (_, c, ch, t) = x.dims4()?;
    Ok(x.get(0)?.permute((1, 2, 0))?.contiguous()?.reshape((ch * t, c))?)
}

pub fn unpack_audio(rows: &Tensor) -> Result<Tensor> {
    let (n, c) = rows.dims2()?;
    Ok(rows.reshape((2, n / 2, c))?.permute((2, 0, 1))?.contiguous()?.unsqueeze(0)?)
}

// ----------------------------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DitConfig {
    pub hidden: usize,
    pub layers: usize,
    pub refiner_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub latents_dim: usize,
    /// Stacked PDD output heads in the checkpoint itself (1 = plain heads; a LoRA can grow them later).
    pub pdd_heads: usize,
    pub audio_dim: usize,
    pub text_dim: usize,
    pub time_embed_dim: usize,
    pub curve_grid: Option<usize>,
    pub timestep_input_dim: usize,
    pub eps: f32,
    pub shift_video: f64,
    pub shift_audio: f64,
}

impl DitConfig {
    /// Same detection as `comfy/model_detection.py` for MiniMax H3.
    pub fn detect(f: &SafeFile) -> Result<Self> {
        if !f.has("video_patch_proj.weight") || !f.has("audio_patch_proj.weight") {
            bail!("{} is not a MiniMax H3 diffusion model", f.path);
        }
        let count = |p: &str, suffix: &str| (0..).take_while(|i| f.has(&format!("{p}{i}.{suffix}"))).count();
        let hidden = f.shape("video_patch_proj.weight")?[0];
        let head_dim = f.shape("blocks.0.attn.q_norm.weight")?[0];
        let qkv = f.shape("blocks.0.attn.qkv_proj.weight")?[0];
        let (time_embed_dim, curve_grid, tin) = if f.has("adaln_t_table") {
            let s = f.shape("adaln_t_table")?;
            (s[1], Some(s[0]), 256)
        } else {
            (f.shape("time_embedder.proj_out.weight")?[0], None, f.shape("time_embedder.proj_in.weight")?[1])
        };
        // a native PDD checkpoint ships `pdd_steps` stacked output heads (block 0 full, the rest offsets)
        let pdd_heads = f
            .metadata
            .get("config")
            .and_then(|c| serde_json::from_str::<serde_json::Value>(c).ok())
            .and_then(|c| c["transformer"]["pdd_steps"].as_u64())
            .map_or(1, |n| (n as usize).max(1));
        Ok(Self {
            hidden,
            pdd_heads,
            layers: count("blocks.", "norm1.weight"),
            refiner_layers: count("token_refiner.blocks.", "norm1.weight"),
            heads: qkv / (3 * head_dim),
            head_dim,
            ffn: f.shape("blocks.0.mlp.fc1.weight")?[0] / 2,
            latents_dim: f.shape("final_layer.video_out.weight")?[0] / pdd_heads / 4,
            audio_dim: f.shape("final_layer.audio_out.weight")?[0] / pdd_heads,
            text_dim: f.shape("condition_proj.weight")?[1],
            time_embed_dim,
            curve_grid,
            timestep_input_dim: tin,
            eps: 1e-5,
            shift_video: 12.0,
            shift_audio: 3.0,
        })
    }
}

pub(crate) struct Block {
    pub(crate) norm1: Tensor,
    pub(crate) norm2: Tensor,
    pub(crate) q_norm: Tensor,
    pub(crate) k_norm: Tensor,
    pub(crate) qkv: QLinear,
    pub(crate) out: QLinear,
    pub(crate) fc1: QLinear,
    pub(crate) fc2: QLinear,
    pub(crate) adaln: QLinear,
}

/// Per-forward state a block needs besides its weights.
pub(crate) struct StepCtx<'a> {
    /// Timestep embeddings of the distinct timesteps, `[U, t_dim]` f32.
    pub t_emb: &'a Tensor,
    /// Curve-form adaLN (no SiLU before the projection).
    pub curve: bool,
    /// Per-token row of the `[3U, hidden]` modulation tables.
    pub idx: &'a Tensor,
    pub rope: &'a Tensor,
    pub u: usize,
}

struct RefinerBlock {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    qkv: QLinear,
    out: QLinear,
    fc1: QLinear,
    fc2: QLinear,
}

/// Largest fused qkv (bytes) a block's attention materializes at once; longer sequences go in head groups.
const ATTN_QKV_BUDGET: usize = 1_500_000_000;

pub struct Dit {
    pub cfg: DitConfig,
    dev: Device,
    dtype: DType,
    mem: MemoryConfig,
    video_patch_proj: QLinear,
    audio_patch_proj: QLinear,
    condition_proj: QLinear,
    adaln_t_table: Option<Tensor>,
    time_embedder: Option<(QLinear, QLinear)>,
    inv_freq: Tensor,
    refiner: Vec<RefinerBlock>,
    refiner_norm: Tensor,
    blocks: Vec<Block>,
    /// Blocks whose weights sit in VRAM now, and how many were at load (`set_resident` moves in between).
    resident: usize,
    base_resident: usize,
    block_bytes: usize,
    final_norm: Tensor,
    final_adaln: QLinear,
    video_out: QLinear,
    audio_out: QLinear,
    /// PDD head bank size of the output heads (1 = plain heads).
    pdd_heads: usize,
    control: Option<crate::control::Control>,
    /// Debug: block indices whose outputs are recorded into `captured`.
    pub capture_blocks: Vec<usize>,
    pub captured: std::cell::RefCell<Vec<(usize, Tensor)>>,
}

/// Conditioning for one sampling run.
#[derive(Clone)]
pub struct DitCond {
    /// Qwen3-VL hidden states `[L, text_dim]` (or already refined `[L, hidden]`).
    pub text: Tensor,
    /// Per-text-token adaLN tag (0 = vision block, 1 = text).
    pub tags: Option<Vec<u32>>,
    pub keyframes: Vec<KeyframeCond>,
    pub refs: Vec<RefCond>,
    pub seed: u64,
    pub visual_cond_noise_aug: f64,
    pub audio_cond_noise_aug: f64,
    /// Denoise mask on the 2x2-patch token grid (see [`TokenMasks`]).
    pub denoise: Option<TokenMasks>,
    /// Fun ControlNet hint and schedule (needs [`Dit::set_control`]).
    pub control: Option<ControlCond>,
}

/// A denoise mask (1 = generate, 0 = keep) pooled to the DiT's token grid, as ComfyUI's
/// `MiniMaxH3._token_grid_masks`: video max-pooled per 2x2 latent patch, every value rounded up to a multiple of
/// 1/256. A stream is `None` when it generates everywhere. Rows with a mask value below 1 run at their own
/// timestep and their velocity is scaled by the value.
#[derive(Clone, Default)]
pub struct TokenMasks {
    /// `[T, H, W]` latent cells, row-major.
    pub video: Option<Vec<f32>>,
    /// One value per audio latent frame.
    pub audio: Option<Vec<f32>>,
}

/// A Fun ControlNet hint for one sampling pass.
#[derive(Clone)]
pub struct ControlCond {
    /// Hint latent `[1, C, T, H, W]` (24 channels, or 49 with the inpaint visibility mask and masked source).
    pub latent: Tensor,
    pub strength: f64,
    /// The ControlNet runs while `sigma_end <= sigma <= sigma_start`.
    pub sigma_start: f64,
    pub sigma_end: f64,
}

#[derive(Clone)]
pub struct KeyframeCond {
    pub frame_index: usize,
    /// Normalized video latent `[1, 24, t, h, w]`.
    pub video: Option<Tensor>,
    /// Normalized audio latent `[1, 32, 2, t]`.
    pub audio: Option<Tensor>,
}

#[derive(Clone)]
pub enum RefCond {
    Image { latent: Tensor },
    Audio { latent: Tensor },
    Video { latent: Tensor, audio: Option<Tensor> },
}

/// Per-run state: packed layout, rope table, refined text and condition rows.
pub struct DitRun {
    layout: PackedLayout,
    rope: Tensor,
    text: Tensor,
    tags: Option<Vec<u32>>,
    cond_video_rows: Option<Tensor>,
    cond_audio_rows: Option<Tensor>,
    vis_aug: f64,
    aud_aug: f64,
    latent_t: usize,
    lat_h: usize,
    lat_w: usize,
    audio_t: usize,
    /// Per-2x2-patch-row mask values of the video stream, and per-audio-row (`2 * audio_t`, channel-major), for
    /// the per-row timesteps; `*_out` scale the predicted velocity (`[1, 1, T, H, W]`, `[1, 1, 2, audio_t]`).
    vmask_rows: Option<Vec<f32>>,
    vmask_out: Option<Tensor>,
    amask_rows: Option<Vec<f32>>,
    amask_out: Option<Tensor>,
    control: Option<RunControl>,
    pub sample_sigmas: Vec<f64>,
}

impl DitRun {
    /// Tokens of the packed sequence (video, audio, text, keyframe and reference rows).
    pub fn seq_len(&self) -> usize {
        self.layout.seq_len
    }
}

pub(crate) struct RunControl {
    pub(crate) cond: ControlCond,
    /// Control patch rows for every video-kind row of the packed sequence (zeros for keyframe / reference rows).
    pub(crate) rows: Tensor,
    /// `[S, 1]`: the strength for rows the control adds to, 0 for audio rows.
    pub(crate) scale: Tensor,
}

fn ld(f: &Arc<SafeFile>, name: &str, dev: &Device, dt: DType) -> Result<Tensor> {
    f.load_as(name, dev, dt)
}

fn dense_f32(f: &Arc<SafeFile>, prefix: &str, dev: &Device) -> Result<QLinear> {
    let mut l = QLinear::load(f, prefix, Placement::Gpu, dev, DType::F32)?;
    if let QFormat::Dense = l.fmt {
        l.weight = Held::from_tensor(l.weight.get(dev)?.to_dtype(DType::F32)?);
    }
    Ok(l)
}

impl Dit {
    pub fn load(path: &str, dev: &Device, mem: &MemoryConfig, lora: Option<(&Lora, f64)>) -> Result<Self> {
        let f = SafeFile::open(path)?;
        let cfg = DitConfig::detect(&f)?;
        let dtype = DType::BF16;
        log::info(&format!(
            "DiT: {} blocks, hidden {}, {} heads x {}, adaln {}",
            cfg.layers,
            cfg.hidden,
            cfg.heads,
            cfg.head_dim,
            if cfg.curve_grid.is_some() { "curve" } else { "time embedder" }
        ));
        let block_bytes: usize = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]
            .iter()
            .map(|p| f.bytes(&format!("blocks.0.{p}.weight")).map(|b| b.len()).unwrap_or(0))
            .sum();
        let lora_bytes = lora.map(|(l, _)| l.device_bytes()).unwrap_or(0);
        let resident = resolve_resident(mem.dit_gpu_blocks, cfg.layers, block_bytes, (mem.reserve_gb * 1e9) as usize + lora_bytes + 2_000_000_000);
        log::info(&format!(
            "DiT: {resident}/{} blocks resident on GPU ({:.1} GB), rest streamed from {}",
            cfg.layers,
            (resident * block_bytes) as f64 / 1e9,
            mem.offload
        ));
        let mut blocks = vec![];
        for i in 0..cfg.layers {
            if std::env::var("H3_DEBUG").is_ok() {
                let (free, _) = crate::config::vram_info().unwrap_or((0, 0));
                log::info(&format!("loading block {i}, free VRAM {:.2} GB", free as f64 / 1e9));
            }
            let place = if i < resident { Placement::Gpu } else { mem.offload };
            let p = format!("blocks.{i}");
            let ad = if cfg.curve_grid.is_some() { dense_f32(&f, &format!("{p}.adaln_proj.linear"), dev)? } else {
                QLinear::load(&f, &format!("{p}.adaln_proj.linear"), place, dev, dtype)?
            };
            blocks.push(Block {
                norm1: ld(&f, &format!("{p}.norm1.weight"), dev, dtype)?,
                norm2: ld(&f, &format!("{p}.norm2.weight"), dev, dtype)?,
                q_norm: ld(&f, &format!("{p}.attn.q_norm.weight"), dev, dtype)?,
                k_norm: ld(&f, &format!("{p}.attn.k_norm.weight"), dev, dtype)?,
                qkv: QLinear::load(&f, &format!("{p}.attn.qkv_proj"), place, dev, dtype)?,
                out: QLinear::load(&f, &format!("{p}.attn.out_proj"), place, dev, dtype)?,
                fc1: QLinear::load(&f, &format!("{p}.mlp.fc1"), place, dev, dtype)?,
                fc2: QLinear::load(&f, &format!("{p}.mlp.fc2"), place, dev, dtype)?,
                adaln: ad,
            });
        }
        let ref_place = if resident == cfg.layers { Placement::Gpu } else { mem.offload };
        let mut refiner = vec![];
        for i in 0..cfg.refiner_layers {
            let p = format!("token_refiner.blocks.{i}");
            refiner.push(RefinerBlock {
                norm1: ld(&f, &format!("{p}.norm1.weight"), dev, dtype)?,
                norm2: ld(&f, &format!("{p}.norm2.weight"), dev, dtype)?,
                q_norm: ld(&f, &format!("{p}.attn.q_norm.weight"), dev, dtype)?,
                k_norm: ld(&f, &format!("{p}.attn.k_norm.weight"), dev, dtype)?,
                qkv: QLinear::load(&f, &format!("{p}.attn.qkv_proj"), ref_place, dev, dtype)?,
                out: QLinear::load(&f, &format!("{p}.attn.out_proj"), ref_place, dev, dtype)?,
                fc1: QLinear::load(&f, &format!("{p}.mlp.fc1"), ref_place, dev, dtype)?,
                fc2: QLinear::load(&f, &format!("{p}.mlp.fc2"), ref_place, dev, dtype)?,
            });
        }
        let (adaln_t_table, time_embedder) = if cfg.curve_grid.is_some() {
            (Some(ld(&f, "adaln_t_table", dev, DType::F32)?), None)
        } else {
            (None, Some((dense_f32(&f, "time_embedder.proj_in", dev)?, dense_f32(&f, "time_embedder.proj_out", dev)?)))
        };
        let mut dit = Self {
            dev: dev.clone(),
            dtype,
            mem: mem.clone(),
            video_patch_proj: dense_f32(&f, "video_patch_proj", dev)?,
            audio_patch_proj: dense_f32(&f, "audio_patch_proj", dev)?,
            condition_proj: QLinear::load(&f, "condition_proj", Placement::Gpu, dev, dtype)?,
            adaln_t_table,
            time_embedder,
            inv_freq: ld(&f, "rope.inv_freq", dev, DType::F32)?,
            refiner,
            refiner_norm: ld(&f, "token_refiner.final_norm.weight", dev, dtype)?,
            blocks,
            resident,
            base_resident: resident,
            block_bytes,
            final_norm: ld(&f, "final_layer.norm.weight", dev, dtype)?,
            final_adaln: dense_f32(&f, "final_layer.adaln_proj.linear", dev)?,
            video_out: dense_f32(&f, "final_layer.video_out", dev)?,
            audio_out: dense_f32(&f, "final_layer.audio_out", dev)?,
            pdd_heads: cfg.pdd_heads,
            control: None,
            capture_blocks: vec![],
            captured: Default::default(),
            cfg,
        };
        if let Some((l, strength)) = lora {
            dit.apply_lora(l, strength)?;
        }
        if lora.is_some() && std::env::var("H3_LORA_RUNTIME").is_err() {
            let t = std::time::Instant::now();
            for b in &mut dit.blocks {
                for l in [&mut b.qkv, &mut b.out, &mut b.fc1, &mut b.fc2, &mut b.adaln] {
                    l.merge_loras(dev)?;
                }
            }
            for b in &mut dit.refiner {
                for l in [&mut b.qkv, &mut b.out, &mut b.fc1, &mut b.fc2] {
                    l.merge_loras(dev)?;
                }
            }
            log::info(&format!("LoRA merged into the weights in {:.1}s", t.elapsed().as_secs_f64()));
        }
        if crate::weights::int8_enabled() {
            for b in &mut dit.blocks {
                b.fc1.interleave_swiglu()?;
            }
        }
        Ok(dit)
    }

    /// Blocks resident in VRAM now.
    pub fn resident_blocks(&self) -> usize {
        self.resident
    }

    /// Blocks resident when the DiT was loaded (the most `set_resident` brings back).
    pub fn base_resident_blocks(&self) -> usize {
        self.base_resident
    }

    /// VRAM held by one block's weights.
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// Keep only the first `n` of the loaded-resident blocks in VRAM: the rest stream from pinned RAM every
    /// step, like the blocks that never fit. Used to open up room for a run with more tokens than the load-time
    /// budget assumed (the second, larger pass of an upscaled request), and to take the blocks back afterwards.
    pub fn set_resident(&mut self, n: usize) -> Result<()> {
        let n = n.min(self.base_resident);
        let dev = self.dev.clone();
        for (i, b) in self.blocks.iter_mut().enumerate().take(self.base_resident) {
            for l in [&mut b.qkv, &mut b.out, &mut b.fc1, &mut b.fc2] {
                if i >= n {
                    l.weight.demote(&dev)?;
                } else {
                    l.weight.promote(&dev)?;
                }
            }
        }
        self.resident = n;
        Ok(())
    }

    fn apply_lora(&mut self, lora: &Lora, strength: f64) -> Result<()> {
        let dev = self.dev.clone();
        let mut attached = 0usize;
        let mut skipped = vec![];
        let mut attach = |lin: &mut QLinear, name: &str| -> Result<()> {
            if let Some(d) = lora.delta(name, &dev)? {
                if d.a.dim(1)? != lin.in_features || d.b.dim(0)? != lin.out_features || d.reshape.is_some() {
                    skipped.push(name.to_string());
                    return Ok(());
                }
                lin.loras.push(crate::weights::LoraDelta { a: d.a, b: d.b, scale: d.scale * strength });
                attached += 1;
            }
            Ok(())
        };
        for (i, b) in self.blocks.iter_mut().enumerate() {
            attach(&mut b.qkv, &format!("blocks.{i}.attn.qkv_proj"))?;
            attach(&mut b.out, &format!("blocks.{i}.attn.out_proj"))?;
            attach(&mut b.fc1, &format!("blocks.{i}.mlp.fc1"))?;
            attach(&mut b.fc2, &format!("blocks.{i}.mlp.fc2"))?;
            attach(&mut b.adaln, &format!("blocks.{i}.adaln_proj.linear"))?;
        }
        for (i, b) in self.refiner.iter_mut().enumerate() {
            attach(&mut b.qkv, &format!("token_refiner.blocks.{i}.attn.qkv_proj"))?;
            attach(&mut b.out, &format!("token_refiner.blocks.{i}.attn.out_proj"))?;
            attach(&mut b.fc1, &format!("token_refiner.blocks.{i}.mlp.fc1"))?;
            attach(&mut b.fc2, &format!("token_refiner.blocks.{i}.mlp.fc2"))?;
        }
        attach(&mut self.final_adaln, "final_layer.adaln_proj.linear")?;
        // output heads: merged (pad-and-add PDD head banks)
        for (lin, name) in [(&mut self.video_out, "final_layer.video_out"), (&mut self.audio_out, "final_layer.audio_out")] {
            let w = lin.weight.get(&dev)?.to_dtype(DType::F32)?;
            let b = lin.bias.clone().context("output head bias")?;
            let (w2, b2) = lora.merge_padded(name, &w, &b, strength, &dev)?;
            if w2.dim(0)? != w.dim(0)? {
                self.pdd_heads = w2.dim(0)? / lin.out_features;
            }
            lin.weight = Held::from_tensor(w2);
            lin.bias = Some(b2);
        }
        if !skipped.is_empty() {
            log::warn(&format!(
                "LoRA: {} targets skipped (shape mismatch with this checkpoint, e.g. {}); {} attached",
                skipped.len(),
                skipped[0],
                attached
            ));
        } else {
            log::info(&format!("LoRA: {attached} layers attached, PDD heads: {}", self.pdd_heads));
        }
        Ok(())
    }

    pub fn audio_scale(&self) -> f64 {
        self.cfg.shift_video / self.cfg.shift_audio
    }

    /// condition_proj + token refiner, once per run.
    pub fn refine_text(&self, text: &Tensor) -> Result<Tensor> {
        Ok(self.refine_text_stages(text)?.pop().unwrap())
    }

    /// Refiner with intermediates: [condition_proj, (attn residual, block out) per block, final].
    pub fn refine_text_stages(&self, text: &Tensor) -> Result<Vec<Tensor>> {
        let mut stages = vec![];
        let x = text.to_dtype(self.dtype)?.to_device(&self.dev)?;
        if x.dim(D::Minus1)? == self.cfg.hidden {
            return Ok(vec![x]);
        }
        let mut x = self.condition_proj.forward(&x)?;
        stages.push(x.clone());
        let (h, d) = (self.cfg.heads, self.cfg.head_dim);
        for b in &self.refiner {
            let n1 = ops::rms_norm(&x, Some(&b.norm1), self.cfg.eps)?;
            let qkv = b.qkv.forward(&n1)?;
            let a = self.attend(&qkv, Some((&b.q_norm, &b.k_norm)), None, 0, h, d)?;
            x = (b.out.forward(&a)? + &x)?;
            stages.push(x.clone());
            let n2 = ops::rms_norm(&x, Some(&b.norm2), self.cfg.eps)?;
            x = (b.fc2.forward(&ops::swiglu(&b.fc1.forward(&n2)?)?)? + &x)?;
            stages.push(x.clone());
        }
        stages.push(ops::rms_norm(&x, Some(&self.refiner_norm), self.cfg.eps)?);
        Ok(stages)
    }

    /// q/k norm (+ rope) out of a fused qkv `[S, 3*H*D]`, attention, `[S, H*D]`.
    fn attend(&self, qkv: &Tensor, norms: Option<(&Tensor, &Tensor)>, rope: Option<&Tensor>, rot: usize, h: usize, d: usize) -> Result<Tensor> {
        let s = qkv.dim(0)?;
        let inner = h * d;
        let (qn, kn) = match norms {
            Some((q, k)) => (Some((Some(q), self.cfg.eps)), Some((Some(k), self.cfg.eps))),
            None => (None, None),
        };
        let q = kernels::qk_norm_rope(qkv, 0, h, d, qn, rope, rot, self.dtype)?;
        let k = kernels::qk_norm_rope(qkv, inner, h, d, kn, rope, rot, self.dtype)?;
        if crate::attn8::available(&q) {
            return Ok(crate::attn8::attention_qkv(&q, &k, qkv, 2 * inner)?.reshape((s, inner))?);
        }
        let v = qkv.narrow(1, 2 * inner, inner)?.contiguous()?.reshape((s, h, d))?;
        let o = ops::attention(&q.unsqueeze(0)?, &k.unsqueeze(0)?, &v.unsqueeze(0)?, false)?;
        Ok(o.reshape((s, inner))?)
    }

    /// How many head groups the attention of a block runs in. The fused qkv is 3x a token stream, and with q, k and
    /// the f16 copy of v alive next to it that is most of the memory of a long sequence (an upscaled request's second
    /// pass has ~150k tokens). Head groups project, rope and attend `heads / groups` heads at a time, so the full
    /// qkv never exists. Results are the same; short sequences keep the single fused projection.
    fn attn_groups(&self, tokens: usize, lin: &QLinear) -> usize {
        if let Some(g) = std::env::var("H3_ATTN_GROUPS").ok().and_then(|v| v.parse::<usize>().ok()) {
            return if g >= 1 && self.cfg.heads % g == 0 { g } else { 1 };
        }
        let qkv_bytes = 3 * tokens * self.cfg.hidden * 2;
        let ok = lin.loras.is_empty() && !lin.interleaved && lin.block_scale.is_none() && crate::attn8::available_arch(&self.dev) && self.cfg.head_dim == 128;
        if !ok || qkv_bytes <= ATTN_QKV_BUDGET {
            return 1;
        }
        let want = qkv_bytes.div_ceil(ATTN_QKV_BUDGET);
        (want..=self.cfg.heads).find(|g| self.cfg.heads % g == 0).unwrap_or(self.cfg.heads)
    }

    /// Attention of one block, `groups` heads groups at a time (see [`attn_groups`](Self::attn_groups)); consumes
    /// `n1`, the modulated input of the qkv projection.
    #[allow(clippy::too_many_arguments)]
    fn attend_grouped(&self, b: &Block, n1: Tensor, r_qkv: &crate::weights::Raw, rope: &Tensor, rot: usize, groups: usize, chunk: usize) -> Result<Tensor> {
        let (h, d) = (self.cfg.heads, self.cfg.head_dim);
        let inner = h * d;
        let (hg, w) = (h / groups, h / groups * d);
        let s = n1.dim(0)?;
        let proj = |base: usize, g: usize| -> Result<Tensor> {
            let r0 = base + g * w;
            b.qkv.rows(r0, w, &self.dev)?.forward_raw(&n1, &r_qkv.rows(r0, w)?, chunk)
        };
        let mut outs = Vec::with_capacity(groups);
        for g in 0..groups {
            let qt = prof::time("qkv linear", &self.dev, || proj(0, g))?;
            let q = kernels::qk_norm_rope(&qt, 0, hg, d, Some((Some(&b.q_norm), self.cfg.eps)), Some(rope), rot, self.dtype)?;
            drop(qt);
            let kt = prof::time("qkv linear", &self.dev, || proj(inner, g))?;
            let k = kernels::qk_norm_rope(&kt, 0, hg, d, Some((Some(&b.k_norm), self.cfg.eps)), Some(rope), rot, self.dtype)?;
            drop(kt);
            let vt = prof::time("qkv linear", &self.dev, || proj(2 * inner, g))?;
            let o = prof::time("attention", &self.dev, || crate::attn8::attention_qkv(&q, &k, &vt, 0))?;
            outs.push(o.reshape((s, w))?);
        }
        drop(n1);
        Ok(if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 1)? })
    }

    fn rope_angles(&self, layout: &PackedLayout) -> Result<Tensor> {
        // [S, 3] positions (f32) * inv_freq per axis -> [S, 48] (t | h | w)
        let inv: Vec<f32> = self.inv_freq.to_vec1()?;
        let nf = inv.len();
        let mut a = Vec::with_capacity(layout.seq_len * 3 * nf);
        for p in &layout.positions {
            for ax in p {
                let v = *ax as f32;
                for f in &inv {
                    a.push(v * f);
                }
            }
        }
        let t = Tensor::from_vec(a, (layout.seq_len, 3 * nf), &self.dev)?;
        // the reference stores the rotation table at compute dtype (bf16)
        let tab = kernels::rope_table(&t)?;
        Ok(tab.to_dtype(self.dtype)?.to_dtype(DType::F32)?)
    }

    fn cond_rows(&self, z: &Tensor, video: bool, aug: f64, seed: u64) -> Result<Tensor> {
        let z = z.to_device(&self.dev)?.to_dtype(DType::F32)?;
        let r = if video { patchify_video(&z)? } else { pack_audio(&z)? };
        if aug < 1.0 {
            let noise = crate::rng::randn(seed, r.dims(), &self.dev)?;
            Ok(((r * aug)? + (noise * (1.0 - aug))?)?)
        } else {
            Ok(r)
        }
    }

    /// Build the per-run state for a target latent shape.
    pub fn prepare(&self, cond: &DitCond, video_shape: &[usize], audio_t: usize, sample_sigmas: Vec<f64>) -> Result<DitRun> {
        let text = self.refine_text(&cond.text)?;
        let text_len = text.dim(0)?;
        let (lt, lh, lw) = (video_shape[2], video_shape[3].div_ceil(2) * 2, video_shape[4].div_ceil(2) * 2);
        let mut kf_specs = vec![];
        let mut vrows = vec![];
        let mut arows = vec![];
        for kf in &cond.keyframes {
            kf_specs.push(KeyframeSpec {
                frame_index: kf.frame_index,
                video_t: kf.video.as_ref().map(|v| v.dims()[2]),
                audio_t: kf.audio.as_ref().map(|a| a.dims()[3]),
            });
            if let Some(v) = &kf.video {
                vrows.push(self.cond_rows(v, true, cond.visual_cond_noise_aug, cond.seed)?);
            }
            if let Some(a) = &kf.audio {
                arows.push(self.cond_rows(a, false, cond.audio_cond_noise_aug, cond.seed + 1)?);
            }
        }
        let mut ref_specs = vec![];
        for r in &cond.refs {
            match r {
                RefCond::Image { latent } => {
                    let d = latent.dims();
                    ref_specs.push(RefSpec::Image { lh: d[3], lw: d[4] });
                    vrows.push(self.cond_rows(latent, true, cond.visual_cond_noise_aug, cond.seed)?);
                }
                RefCond::Audio { latent } => {
                    ref_specs.push(RefSpec::Audio { rt: latent.dims()[3] });
                    arows.push(self.cond_rows(latent, false, cond.audio_cond_noise_aug, cond.seed + 1)?);
                }
                RefCond::Video { latent, audio } => {
                    let d = latent.dims();
                    let rt = audio.as_ref().map(|a| a.dims()[3]).unwrap_or(0);
                    ref_specs.push(RefSpec::Video { lt: d[2], lh: d[3], lw: d[4], rt });
                    vrows.push(self.cond_rows(latent, true, cond.visual_cond_noise_aug, cond.seed)?);
                    if let Some(a) = audio {
                        arows.push(self.cond_rows(a, false, cond.audio_cond_noise_aug, cond.seed + 1)?);
                    }
                }
            }
        }
        let layout = PackedLayout::new(text_len, lt, lh, lw, audio_t, &kf_specs, &ref_specs);
        let rope = self.rope_angles(&layout)?;
        let (vmask_rows, vmask_out, amask_rows, amask_out) = self.mask_rows(cond.denoise.as_ref(), lt, lh, lw, audio_t)?;
        let control = match (&cond.control, &self.control) {
            (Some(c), Some(model)) => Some(model.prepare_run(c, &layout)?),
            (Some(_), None) => bail!("a control hint needs a loaded ControlNet (Dit::set_control)"),
            _ => None,
        };
        if let Some(t) = &cond.tags {
            if t.len() != text_len {
                bail!("token tags ({}) do not match text length ({text_len})", t.len());
            }
        }
        Ok(DitRun {
            rope,
            text,
            tags: cond.tags.clone(),
            cond_video_rows: if vrows.is_empty() { None } else { Some(Tensor::cat(&vrows, 0)?) },
            cond_audio_rows: if arows.is_empty() { None } else { Some(Tensor::cat(&arows, 0)?) },
            vis_aug: cond.visual_cond_noise_aug,
            aud_aug: cond.audio_cond_noise_aug,
            latent_t: lt,
            lat_h: lh,
            lat_w: lw,
            audio_t,
            layout,
            vmask_rows,
            vmask_out,
            amask_rows,
            amask_out,
            control,
            sample_sigmas,
        })
    }

    /// Attach (or drop) a Fun ControlNet; requests with a [`ControlCond`] then run it.
    pub fn set_control(&mut self, control: Option<crate::control::Control>) -> Result<()> {
        if let Some(c) = &control {
            c.check_base(self.cfg.time_embed_dim, self.cfg.curve_grid.is_some())?;
        }
        self.control = control;
        Ok(())
    }

    pub fn has_control(&self) -> bool {
        self.control.is_some()
    }

    /// Checkpoint of the attached ControlNet.
    pub fn control_path(&self) -> Option<&str> {
        self.control.as_ref().map(|c| c.path.as_str())
    }

    /// Whether the weights are all resident (the ControlNet then loads resident too).
    pub fn fully_resident(&self) -> bool {
        self.resident == self.cfg.layers
    }

    /// Row-level views of a [`TokenMasks`]: per-row values for the timesteps, and the velocity scale tensors.
    #[allow(clippy::type_complexity)]
    fn mask_rows(&self, m: Option<&TokenMasks>, lt: usize, lh: usize, lw: usize, audio_t: usize) -> Result<(Option<Vec<f32>>, Option<Tensor>, Option<Vec<f32>>, Option<Tensor>)> {
        let Some(m) = m else { return Ok((None, None, None, None)) };
        let (mut vrows, mut vout) = (None, None);
        if let Some(v) = &m.video {
            if v.len() != lt * lh * lw {
                bail!("video mask has {} cells, the latent {}", v.len(), lt * lh * lw);
            }
            let mut rows = Vec::with_capacity(lt * (lh / 2) * (lw / 2));
            for t in 0..lt {
                for y in 0..lh / 2 {
                    for x in 0..lw / 2 {
                        let at = |dy: usize, dx: usize| v[(t * lh + 2 * y + dy) * lw + 2 * x + dx];
                        rows.push(at(0, 0).max(at(0, 1)).max(at(1, 0)).max(at(1, 1)));
                    }
                }
            }
            vrows = Some(rows);
            vout = Some(Tensor::from_vec(v.clone(), (1, 1, lt, lh, lw), &self.dev)?);
        }
        let (mut arows, mut aout) = (None, None);
        if let Some(a) = &m.audio {
            if a.len() != audio_t {
                bail!("audio mask has {} frames, the latent {audio_t}", a.len());
            }
            arows = Some([a.as_slice(), a.as_slice()].concat());
            aout = Some(Tensor::from_vec([a.as_slice(), a.as_slice()].concat(), (1, 1, 2, audio_t), &self.dev)?);
        }
        Ok((vrows, vout, arows, aout))
    }

    /// One transformer block over the packed stream `x` (shared by the base blocks and the ControlNet's), given its
    /// weights already fetched as `[qkv, out, fc1, fc2]`.
    pub(crate) fn run_block(&self, b: &Block, raws: [crate::weights::Raw; 4], mut x: Tensor, cx: &StepCtx) -> Result<Tensor> {
        let dev = &self.dev;
        let [r_qkv, r_out, r_fc1, r_fc2] = raws;
        let (h, d, hidden) = (self.cfg.heads, self.cfg.head_dim, self.cfg.hidden);
        let rot = cx.rope.dim(2)? * 2;
        let chunk = self.mem.token_chunk;
        let te = if cx.curve { cx.t_emb.clone() } else { ops::silu(cx.t_emb)?.to_dtype(self.dtype)? };
        let m = b.adaln.forward(&te)?.to_dtype(DType::F32)?; // [U, 6*hidden*3]
        let m = m.reshape((cx.u * 3, 6 * hidden))?;
        let c = |i: usize| m.narrow(1, i * hidden, hidden).and_then(|t| t.contiguous());
        let (shift_msa, scale_msa, gate_msa) = (c(0)?, c(1)?, c(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (c(3)?, c(4)?, c(5)?);

        let n1 = prof::time("norm/mod", dev, || -> Result<Tensor> { Ok(kernels::rms_modulate(&x, &b.norm1, self.cfg.eps, &scale_msa, &shift_msa, cx.idx)?) })?;
        let groups = self.attn_groups(n1.dim(0)?, &b.qkv);
        let att = if groups > 1 {
            self.attend_grouped(b, n1, &r_qkv, cx.rope, rot, groups, chunk)?
        } else {
            let qkv = prof::time("qkv linear", dev, || b.qkv.forward_raw(&n1, &r_qkv, chunk))?;
            drop(n1);
            let att = prof::time("attention", dev, || self.attend(&qkv, Some((&b.q_norm, &b.k_norm)), Some(cx.rope), rot, h, d))?;
            drop(qkv);
            att
        };
        let o = prof::time("out linear", dev, || b.out.forward_raw(&att, &r_out, chunk))?;
        drop(att);
        x = prof::time("norm/mod", dev, || kernels::gate_add(&x, &o, &gate_msa, cx.idx))?;
        drop(o);

        let n2 = prof::time("norm/mod", dev, || -> Result<Tensor> { Ok(kernels::rms_modulate(&x, &b.norm2, self.cfg.eps, &scale_mlp, &shift_mlp, cx.idx)?) })?;
        let w1 = prof::time("dequant", dev, || b.fc1.prepare(r_fc1.clone(), dev, self.dtype))?;
        let w2 = prof::time("dequant", dev, || b.fc2.prepare(r_fc2.clone(), dev, self.dtype))?;
        drop((r_qkv, r_out, r_fc1, r_fc2));
        let s = n2.dim(0)?;
        let mut outs = vec![];
        let mc = self.mem.mlp_chunk.max(1);
        let mut st = 0;
        prof::time("mlp", dev, || -> Result<()> {
            while st < s {
                let n = mc.min(s - st);
                let hc = b.fc1.run_swiglu(&n2.narrow(0, st, n)?, &w1, usize::MAX)?;
                outs.push(b.fc2.run(&hc, &w2, usize::MAX)?);
                st += n;
            }
            Ok(())
        })?;
        drop((w1, w2, n2));
        let mo = Tensor::cat(&outs, 0)?;
        drop(outs);
        x = kernels::gate_add(&x, &mo, &gate_mlp, cx.idx)?;

        Ok(x)
    }

    fn time_embed(&self, t_vals: &[f64]) -> Result<Tensor> {
        let t = Tensor::from_vec(t_vals.iter().map(|v| *v as f32).collect::<Vec<_>>(), t_vals.len(), &self.dev)?;
        if let Some(table) = &self.adaln_t_table {
            let g = table.dim(0)?;
            let mut i0 = vec![];
            let mut fr = vec![];
            for v in t_vals {
                let pos = (*v as f32).clamp(0.0, 1.0) * (g - 1) as f32;
                let i = (pos.floor() as usize).min(g - 2);
                i0.push(i as u32);
                fr.push(pos - i as f32);
            }
            let n = t_vals.len();
            let a = table.index_select(&Tensor::from_vec(i0.clone(), n, &self.dev)?, 0)?;
            let b = table.index_select(&Tensor::from_vec(i0.iter().map(|i| i + 1).collect::<Vec<_>>(), n, &self.dev)?, 0)?;
            let w = Tensor::from_vec(fr, (n, 1), &self.dev)?;
            return Ok((&a + (b - &a)?.broadcast_mul(&w)?)?);
        }
        let (pin, pout) = self.time_embedder.as_ref().unwrap();
        let half = self.cfg.timestep_input_dim / 2;
        let freqs: Vec<f32> = (0..half).map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp()).collect();
        let freqs = Tensor::from_vec(freqs, (1, half), &self.dev)?;
        let args = t.unsqueeze(1)?.broadcast_mul(&freqs)?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], 1)?;
        let h = ops::silu(&pin.forward(&emb)?)?;
        Ok(pout.forward(&h)?.to_dtype(self.dtype)?.to_dtype(DType::F32)?)
    }

    /// One model evaluation. `x_audio` is the sampler's carried audio (scaled onto the video
    /// schedule); the returned audio velocity is in the same carried space. `sigma` is the video
    /// sigma. Returns the flow outputs `[video, audio]` (CONST parameterization).
    pub fn forward(&self, run: &DitRun, x_video: &Tensor, x_audio: &Tensor, sigma: f64) -> Result<(Tensor, Tensor)> {
        let scale = self.audio_scale();
        let (sv, sa) = (self.cfg.shift_video, self.cfg.shift_audio);
        let sigma_v = sigma.max(1e-6);
        let sigma_a = time_shift_sigma(sigma_v, sv, sa);
        let carry = sigma_a / sigma_v;
        let audio_own = (x_audio * carry)?;
        let (v, a) = self.forward_inner(run, x_video, &audio_own, sigma_v, sigma_a)?;
        let a = if scale != 1.0 { ((&audio_own * (1.0 - scale))? + (a * (1.0 + (scale - 1.0) * sigma_a))?)? } else { a };
        Ok((v, a))
    }

    fn forward_inner(&self, run: &DitRun, x_video: &Tensor, x_audio: &Tensor, sigma_v: f64, sigma_a: f64) -> Result<(Tensor, Tensor)> {
        let dev = &self.dev;
        let (_, _, ot, oh, ow) = x_video.dims5()?;
        let xv = x_video.to_dtype(DType::F32)?;
        let xv = if oh % 2 != 0 || ow % 2 != 0 { xv.pad_with_same(3, 0, oh % 2)?.pad_with_same(4, 0, ow % 2)? } else { xv };
        let layout = &run.layout;
        let t_v = 1.0 - sigma_v;
        let t_a = 1.0 - sigma_a;
        let seg_t = |k: Seg| -> f64 {
            match k {
                Seg::Text | Seg::Video => t_v,
                Seg::Audio => t_a,
                Seg::Cond | Seg::RefImg => t_v.max(run.vis_aug),
                Seg::CondAudio | Seg::RefAudio => t_a.max(run.aud_aug),
            }
        };
        // masked rows run at their own strength: a row with mask value m sits at sigma = m * sigma_stream, so its
        // label is 1 - m * sigma, held at the cond timestep for rows that are kept
        let (t_pin_v, t_pin_a) = (t_v.max(VISUAL_COND_TIMESTEP), t_a.max(AUDIO_COND_TIMESTEP));
        let v_rows_t: Option<Vec<f64>> = run.vmask_rows.as_ref().map(|m| m.iter().map(|&v| (1.0 - v as f64 * sigma_v).min(t_pin_v)).collect());
        let a_rows_t: Option<Vec<f64>> = run.amask_rows.as_ref().map(|m| m.iter().map(|&v| (1.0 - v as f64 * sigma_a).min(t_pin_a)).collect());
        let mut unique: Vec<f64> = vec![t_v, t_a];
        for s in &layout.segments {
            unique.push(seg_t(s.2));
        }
        unique.extend(v_rows_t.iter().chain(a_rows_t.iter()).flatten());
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup();
        let t_row = |t: f64| unique.binary_search_by(|u| u.partial_cmp(&t).unwrap()).unwrap() as u32;

        // per-token mod-row index into the [3U, hidden] block tables
        let mut idx = vec![0u32; layout.seq_len];
        for &(a, b, k) in &layout.segments {
            let rows_t = match k {
                Seg::Video => v_rows_t.as_deref(),
                Seg::Audio => a_rows_t.as_deref(),
                _ => None,
            };
            if let Some(rt) = rows_t {
                for (slot, &t) in idx[a..b].iter_mut().zip(rt) {
                    *slot = t_row(t) * 3 + k.tag();
                }
                continue;
            }
            let base = t_row(seg_t(k)) * 3;
            for (i, slot) in idx[a..b].iter_mut().enumerate() {
                *slot = if k == Seg::Text { base + run.tags.as_ref().map(|t| t[i]).unwrap_or(1) } else { base + k.tag() };
            }
        }
        let idx = Tensor::from_vec(idx, layout.seq_len, dev)?;

        // embed + pack
        let video_rows = patchify_video(&xv)?;
        let audio_rows = pack_audio(&x_audio.to_dtype(DType::F32)?)?;
        let all_v = match &run.cond_video_rows {
            Some(c) => Tensor::cat(&[c, &video_rows], 0)?,
            None => video_rows,
        };
        let all_a = match &run.cond_audio_rows {
            Some(c) => Tensor::cat(&[c, &audio_rows], 0)?,
            None => audio_rows,
        };
        let ve = self.video_patch_proj.forward(&all_v)?.to_dtype(self.dtype)?;
        let ae = self.audio_patch_proj.forward(&all_a)?.to_dtype(self.dtype)?;
        let mut parts = vec![];
        let (mut vo, mut ao) = (0, 0);
        for &(a, b, k) in &layout.segments {
            let n = b - a;
            match k {
                Seg::Text => parts.push(run.text.clone()),
                k if k.is_video() => {
                    parts.push(ve.narrow(0, vo, n)?);
                    vo += n;
                }
                _ => {
                    parts.push(ae.narrow(0, ao, n)?);
                    ao += n;
                }
            }
        }
        let mut x = Tensor::cat(&parts, 0)?;
        drop((parts, ve, ae, all_v, all_a));
        // the ControlNet branches off the stream as it enters the first block
        let ctrl = match (&self.control, &run.control) {
            (Some(model), Some(rc)) if rc.cond.sigma_end <= sigma_v && sigma_v <= rc.cond.sigma_start => Some((model, rc)),
            _ => None,
        };
        let pristine = ctrl.map(|_| x.clone());
        let mut cstream: Option<Tensor> = None;

        let t_emb = self.time_embed(&unique)?; // [U, t_dim] f32
        let curve = self.adaln_t_table.is_some();
        let hidden = self.cfg.hidden;
        let cx = StepCtx { t_emb: &t_emb, curve, idx: &idx, rope: &run.rope, u: unique.len() };
        let fetch = |b: &Block| -> Result<[crate::weights::Raw; 4]> { Ok([b.qkv.fetch(dev)?, b.out.fetch(dev)?, b.fc1.fetch(dev)?, b.fc2.fetch(dev)?]) };
        let mut next = Some(fetch(&self.blocks[0])?);
        let mut prev_marker: Option<crate::weights::Marker> = None;
        for (bi, b) in self.blocks.iter().enumerate() {
            crate::progress::check()?;
            let [r_qkv, r_out, r_fc1, r_fc2] = next.take().unwrap();
            x = self.run_block(b, [r_qkv, r_out, r_fc1, r_fc2], x, &cx)?;
            if let (Some((model, rc)), Some(ci)) = (ctrl, ctrl.and_then(|(m, _)| m.injection.iter().position(|&l| l == bi))) {
                if ci == 0 {
                    cstream = Some(model.init_stream(pristine.as_ref().unwrap(), &rc.rows, layout)?);
                }
                let (c, skip) = model.step(self, ci, cstream.take().unwrap(), &cx)?;
                cstream = Some(c);
                x = (x + skip.to_dtype(self.dtype)?.broadcast_mul(&rc.scale.to_dtype(self.dtype)?)?)?;
            }
            // block bi is enqueued: once the GPU is past block bi-1, start uploading block bi+1
            if let Some(m) = prev_marker.take() {
                m.wait();
            }
            prev_marker = Some(crate::weights::Marker::record(dev));
            if bi + 1 < self.blocks.len() {
                next = Some(fetch(&self.blocks[bi + 1])?);
            }
            if self.capture_blocks.contains(&bi) {
                self.captured.borrow_mut().push((bi, x.clone()));
            }
        }

        // final layer
        let fm = self.final_adaln.forward(&t_emb)?.to_dtype(DType::F32)?; // [U, 2*hidden]
        let shift = fm.narrow(1, 0, hidden)?;
        let fscale = fm.narrow(1, hidden, hidden)?;
        let head_in = |a: usize, b: usize, row: u32| -> Result<Tensor> {
            let n = ops::rms_norm(&x.narrow(0, a, b - a)?, Some(&self.final_norm), self.cfg.eps)?.to_dtype(DType::F32)?;
            let sc = fscale.narrow(0, row as usize, 1)?;
            let sh = shift.narrow(0, row as usize, 1)?;
            Ok(n.broadcast_mul(&(sc + 1.0)?)?.broadcast_add(&sh)?)
        };
        let (va, vb) = layout.find(Seg::Video);
        let (aa, ab) = layout.find(Seg::Audio);
        let head_in_rows = |a: usize, b: usize, rows_t: &[f64]| -> Result<Tensor> {
            let n = ops::rms_norm(&x.narrow(0, a, b - a)?, Some(&self.final_norm), self.cfg.eps)?.to_dtype(DType::F32)?;
            let rows = Tensor::from_vec(rows_t.iter().map(|&t| t_row(t)).collect::<Vec<u32>>(), rows_t.len(), dev)?;
            let sc = fscale.contiguous()?.index_select(&rows, 0)?;
            let sh = shift.contiguous()?.index_select(&rows, 0)?;
            Ok((((sc + 1.0)? * n)? + sh)?)
        };
        let hv = match &v_rows_t {
            Some(r) => head_in_rows(va, vb, r)?,
            None => head_in(va, vb, t_row(t_v))?,
        };
        let ha = match &a_rows_t {
            Some(r) => head_in_rows(aa, ab, r)?,
            None => head_in(aa, ab, t_row(t_a))?,
        };
        let (v, a) = if self.pdd_heads == 1 {
            (self.video_out.forward(&hv)?, self.audio_out.forward(&ha)?)
        } else {
            let sig = &run.sample_sigmas;
            let n = self.pdd_heads;
            let i = sig
                .iter()
                .enumerate()
                .min_by(|x, y| (x.1 - sigma_v).abs().partial_cmp(&(y.1 - sigma_v).abs()).unwrap())
                .map(|x| x.0)
                .unwrap_or(0);
            let sigma_next = sig[(i + 1).min(sig.len() - 1)];
            let to_idx = |s: f64| ((1.0 - time_shift_sigma(s, self.cfg.shift_video, 1.0)) * n as f64).round() as usize;
            let start = to_idx(sigma_v).min(n - 1);
            let stop = to_idx(sigma_next).max(start + 1);
            (
                pdd_head(&self.video_out, &hv, n, start, stop, self.cfg.shift_video)?,
                pdd_head(&self.audio_out, &ha, n, start, stop, self.cfg.shift_audio)?,
            )
        };
        let video = unpatchify_video(&v, run.latent_t, run.lat_h / 2, run.lat_w / 2, self.cfg.latents_dim)?;
        let video = video.narrow(2, 0, ot)?.narrow(3, 0, oh)?.narrow(4, 0, ow)?;
        let audio = unpack_audio(&a)?;
        // rows that are partly kept predict at mask * sigma: scale their velocity to match
        let video = match &run.vmask_out {
            Some(m) => video.broadcast_mul(m)?,
            None => video,
        };
        let audio = match &run.amask_out {
            Some(m) => audio.broadcast_mul(m)?,
            None => audio,
        };
        let _ = run.audio_t;
        Ok((video.neg()?, audio.neg()?))
    }
}

/// PDD head bank: row block 0 is a full head, later blocks are offsets from it; a step uses the
/// dt-weighted mean of the heads it spans.
fn pdd_head(head: &QLinear, h: &Tensor, n: usize, start: usize, stop: usize, flow_shift: f64) -> Result<Tensor> {
    let w = head.weight.get(h.device())?.to_dtype(DType::F32)?;
    let b = head.bias.as_ref().unwrap().to_dtype(DType::F32)?;
    let k = w.dim(1)?;
    let rows = w.reshape((n, w.dim(0)? / n, k))?;
    let brows = b.reshape((n, b.dim(0)? / n))?;
    let grid: Vec<f64> = (0..=n).map(|i| 1.0 - i as f64 / n as f64).collect();
    let f: Vec<f64> = grid.iter().map(|g| 1.0 - flow_shift * g / (1.0 + (flow_shift - 1.0) * g)).collect();
    let dt: Vec<f64> = (start..stop).map(|i| f[i + 1] - f[i]).collect();
    let tot: f64 = dt.iter().sum();
    let first = start.max(1);
    let mut wm = rows.get(0)?;
    let mut bm = brows.get(0)?;
    for (j, i) in (first..stop).enumerate() {
        let wt = dt[first - start + j] / tot;
        wm = (wm + (rows.get(i)? * wt)?)?;
        bm = (bm + (brows.get(i)? * wt)?)?;
    }
    Ok(h.matmul(&wm.t()?)?.broadcast_add(&bm)?)
}

pub mod prof {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    thread_local!(static T: RefCell<BTreeMap<&'static str, f64>> = RefCell::new(BTreeMap::new()));
    pub fn on() -> bool {
        std::env::var("H3_PROFILE").is_ok()
    }
    /// Time `f` (device-synchronized) under `name` when H3_PROFILE is set.
    pub fn time<R>(name: &'static str, dev: &candle_core::Device, f: impl FnOnce() -> R) -> R {
        if !on() {
            return f();
        }
        let _ = dev.synchronize();
        let t = std::time::Instant::now();
        let r = f();
        let _ = dev.synchronize();
        T.with(|m| *m.borrow_mut().entry(name).or_default() += t.elapsed().as_secs_f64());
        r
    }
    pub fn dump() {
        T.with(|m| {
            for (k, v) in m.borrow_mut().iter() {
                eprintln!("[h3] profile {k:12} {:8.1} ms", v * 1e3);
            }
            m.borrow_mut().clear();
        });
    }
}

pub mod log {
    pub fn info(s: &str) {
        eprintln!("[h3] {s}");
    }
    pub fn warn(s: &str) {
        eprintln!("[h3] WARNING: {s}");
    }
}
