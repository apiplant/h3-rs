//! End-to-end generation: prompt (+ optional first / last keyframes) -> MP4 with stereo audio.
//! Stages run sequentially (text encoder -> VAE encode -> DiT -> VAE decodes) and, with
//! `MemoryConfig::sequential`, each model is dropped before the next one loads.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::config::{vram_info, MemoryConfig};
use crate::control::{self, Control, HintInputs};
use crate::dit::{log, ControlCond, Dit, DitCond, KeyframeCond, RefCond, AUDIO_COND_TIMESTEP, FRAME_PER_TOKEN, FRAME_RESCALE, VISUAL_COND_TIMESTEP};
use crate::inpaint::{Inpaint, MaskAudio, Masks};
use crate::lora::Lora;
use crate::media;
use crate::progress::{self, Stage};
use crate::sampling::{self, Sampler};
use crate::text::{RefItem, TextEncoder, Tokenizer};
use crate::upscaler::{self, LatentUpscaler};
use crate::vae_audio::{AudioEncoder, AudioVae};
use crate::vae_video::VideoVae;

pub const FPS: usize = 24;
/// DiT activation memory of a forward over `tokens` (hidden 5376): ~135 KB per token with the fused qkv, ~110 KB per
/// token once attention runs in head groups (long sequences, see `Dit::attn_groups`; measured on a 24 GB card with
/// 153k tokens: 13 resident blocks ran out of memory, 9 fit), plus the streamed weights' buffers. Used to size the
/// VRAM an upscaled request's second pass needs; too low is caught by the out-of-memory retry.
fn dit_activation_bytes(tokens: usize) -> usize {
    if 3 * tokens * 5376 * 2 <= 1_500_000_000 {
        tokens * 135_000 + (3 << 29)
    } else {
        tokens * 110_000 + (2 << 30)
    }
}
/// Upscaler activation memory per output latent voxel of a 32-frame segment (512 channels, f16, ~8 live tensors).
const UPSCALER_BYTES_PER_VOXEL: usize = 8_500;
const CHUNK_FRAMES: usize = 42;
pub const AUDIO_LATENT_FPS: usize = 40;

/// Model files, ComfyUI layout (`models/{diffusion_models,text_encoders,vae,loras}`).
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub dit: PathBuf,
    pub text_encoder: PathBuf,
    pub video_vae: PathBuf,
    pub audio_vae: PathBuf,
    /// Latent upscaler checkpoint (`models/latent_upscale_models/`), when one was found.
    pub upscaler: Option<PathBuf>,
    /// Where missing known models get downloaded to (ComfyUI layout)
    pub models_dir: PathBuf,
}

/// Latent upscaler looked up by default.
pub const DEFAULT_UPSCALER: &str = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors";

impl ModelPaths {
    /// Resolve each name against `<models_dir>/<subdir>/` unless it is already a path to a file.
    pub fn resolve(models_dir: &Path, dit: &str, te: &str, vvae: &str, avae: &str) -> Result<Self> {
        let r = |sub: &str, n: &str| -> Result<PathBuf> {
            let p = Path::new(n);
            if p.is_file() {
                return Ok(p.to_path_buf());
            }
            crate::download::ensure(&models_dir.join(sub), n)
        };
        Ok(Self { dit: r("diffusion_models", dit)?, text_encoder: r("text_encoders", te)?, video_vae: r("vae", vvae)?, audio_vae: r("vae", avae)?, upscaler: None, models_dir: models_dir.to_path_buf() })
    }

    /// Look for the latent upscaler `name` (a path, or a file in `<models_dir>/latent_upscale_models/`).
    /// Without `required`, a missing file just leaves upscaling unavailable.
    pub fn with_upscaler(mut self, models_dir: &Path, name: &str, required: bool) -> Result<Self> {
        let direct = Path::new(name);
        let in_dir = models_dir.join("latent_upscale_models").join(name);
        self.upscaler = if direct.is_file() {
            Some(direct.to_path_buf())
        } else if in_dir.is_file() {
            Some(in_dir)
        } else if required {
            Some(crate::download::ensure(&models_dir.join("latent_upscale_models"), name)?)
        } else {
            None
        };
        Ok(self)
    }
}

/// Two-pass generation through the latent upscaler (the ComfyUI "Minimax H3 Latent Upscaler" workflow):
/// the first `split` of the sampling steps run at the requested canvas, the clean estimate is upscaled
/// `scale` times in latent space (no VAE round trip), re-noised to where the schedule stands, and the remaining
/// steps refine it at the larger canvas, which is what gets delivered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Upscale {
    /// Factor between the requested canvas (sampled first) and the delivered one, in (1, 4].
    pub scale: f64,
    /// Fraction of the steps sampled at the low resolution, in (0, 1).
    pub split: f64,
}

impl Default for Upscale {
    fn default() -> Self {
        Self { scale: 1.5, split: 0.5 }
    }
}

#[derive(Debug, Clone)]
pub struct Request {
    pub prompt: String,
    /// Explicit canvas; when unset it follows the first keyframe's (else `aspect`'s) aspect ratio
    /// at `megapixels`, or defaults to 1344x768.
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub megapixels: f64,
    /// Aspect ratio `(w, h)` for an unset canvas without a keyframe.
    pub aspect: Option<(f64, f64)>,
    /// How keyframes are fitted to the canvas (default: first stretched, last center-cropped,
    /// like ComfyUI).
    pub fit: Option<Fit>,
    /// ref2va references (images, videos, audio), in tag order per type.
    pub refs: Vec<RefInput>,
    pub ref_image_size: RefImageSize,
    /// Frame count at 24 fps; snapped up to the 17k+5 grid.
    pub length: usize,
    pub steps: usize,
    pub seed: u64,
    pub sampler: Sampler,
    pub shift_video: f64,
    pub shift_audio: f64,
    pub first_frame: Option<String>,
    pub last_frame: Option<String>,
    pub lora: Option<(String, f64)>,
    /// Images, clips and audio anchored at arbitrary frames (ComfyUI `MiniMaxH3AddGuide`), in order.
    pub guides: Vec<Guide>,
    /// Video (or image) to edit: with `mask`, the regions the mask keeps are taken from it, and its soundtrack is
    /// kept (see `mask_audio`) unless `source_mute`. The video length must not exceed the source's.
    pub source: Option<String>,
    pub source_mute: bool,
    /// Denoise mask over `source` (image or video, white = regenerate, grey = partly).
    pub mask: Option<String>,
    pub mask_audio: MaskAudio,
    /// Fun ControlNet: a control video and/or the source + mask as an inpainting hint.
    pub control: Option<ControlRequest>,
    /// Sample at the requested size, upscale the latent, refine at `scale` times that size (see [`Upscale`]).
    pub upscale: Option<Upscale>,
    pub output: String,
    pub save_latents: Option<String>,
    pub crf: u32,
}

/// One `MiniMaxH3AddGuide`: media anchored at a pixel frame of the video.
#[derive(Debug, Clone)]
pub struct Guide {
    /// An image, a clip (video file; cut to 5, 22, 39... = 17k+5 frames), or audio.
    pub path: String,
    /// Frame the guide starts at; negative counts from the end of the video.
    pub frame: i64,
    pub kind: GuideKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuideKind {
    /// Image or clip, fitted to the canvas by center crop. Only its pictures: its soundtrack is a separate
    /// `Audio` guide.
    Visual,
    /// Soundtrack starting at the frame, cropped to the video's remaining duration.
    Audio,
}

impl Guide {
    /// Parse `PATH[@FRAME]` (frame defaults to 0).
    pub fn parse(spec: &str, kind: GuideKind) -> Guide {
        match spec.rsplit_once('@').and_then(|(p, f)| f.trim().parse::<i64>().ok().map(|f| (p, f))) {
            Some((path, frame)) => Guide { path: path.to_string(), frame, kind },
            None => Guide { path: spec.to_string(), frame: 0, kind },
        }
    }
}

/// A MiniMax H3 Fun ControlNet applied to the base model.
#[derive(Debug, Clone)]
pub struct ControlRequest {
    /// Checkpoint (`models/model_patches/`).
    pub model: PathBuf,
    /// Control video (depth, pose, edges...), repeated to the length if shorter.
    pub video: Option<String>,
    pub strength: f64,
    /// Share of the sampling (0 = first step, 1 = last) the ControlNet is active over.
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Fit {
    Stretch,
    Crop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RefImageSize {
    /// Scale each reference (down only) to the generation's pixel area.
    Match,
    /// 2048 px short edge (best identity fidelity, slower).
    Max,
}

#[derive(Debug, Clone)]
pub enum RefInput {
    /// `<Picture i>`
    Image(String),
    /// `<Video k>`; `with_audio` also uses its soundtrack as `<Audio j>` (or a separate file).
    Video { path: String, audio: Option<String> },
    /// `<Audio j>`
    Audio(String),
}

/// Round to a multiple of 32 (at least 32).
fn r32(v: f64) -> usize {
    ((v / 32.0).round() as usize * 32).max(32)
}

/// ComfyUI `ImageScaleToTotalPixels`: keep the aspect ratio at `megapixels` (x 1024^2), step 32.
pub fn canvas_for_aspect(w: f64, h: f64, megapixels: f64) -> (usize, usize) {
    let s = (megapixels * 1024.0 * 1024.0 / (w * h)).sqrt();
    (r32(w * s), r32(h * s))
}

/// The reference pipeline's video canvas: 768 short edge, 768*1344 area cap, step 32.
fn adapt_canvas(width: usize, height: usize) -> (usize, usize) {
    let ratio = width as f64 / height as f64;
    let (mut w, mut h) = if ratio >= 1.0 { (768.0 * ratio, 768.0) } else { (768.0, 768.0 / ratio) };
    let max = 768.0 * 1344.0;
    if w * h > max {
        let s = (max / (w * h)).sqrt();
        w *= s;
        h *= s;
    }
    (r32(w), r32(h))
}

impl Default for Request {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: None,
            height: None,
            megapixels: 1.0,
            aspect: None,
            fit: None,
            refs: vec![],
            ref_image_size: RefImageSize::Match,
            length: 124,
            steps: 20,
            seed: 1,
            sampler: Sampler::ResMultistep,
            shift_video: 12.0,
            shift_audio: 3.0,
            first_frame: None,
            last_frame: None,
            lora: None,
            guides: vec![],
            source: None,
            source_mute: false,
            mask: None,
            mask_audio: MaskAudio::default(),
            control: None,
            upscale: None,
            output: "h3_output.mp4".into(),
            save_latents: None,
            crf: 18,
        }
    }
}

pub fn align_frame_count(mut n: usize) -> usize {
    while n % 17 != 5 {
        n += 1;
    }
    n
}

pub fn video_latent_t(frames: usize) -> usize {
    if frames <= 5 {
        2
    } else {
        ((frames - 5) / 17) * 5 + 2
    }
}

/// (frame_count, latent_t, audio_t)
pub fn temporal_shape(length: usize) -> (usize, usize, usize) {
    let fc = align_frame_count(length.max(5));
    (fc, video_latent_t(fc), (fc as f64 / FPS as f64 * AUDIO_LATENT_FPS as f64).round() as usize)
}

/// Frames of an AddGuide clip: 1 for a picture or fewer than 5 frames, else cut down to 17k+5.
pub fn guide_frames(n: usize) -> usize {
    if n < 5 {
        1
    } else {
        n - (n - 5) % 17
    }
}

/// The guide's start frame in the video (negative counts from the end), checked to leave room for `frames`.
pub fn resolve_guide_frame(frame: i64, frame_count: usize, frames: usize) -> Result<usize> {
    let r = if frame >= 0 { frame } else { frame_count as i64 + frame };
    if r < 0 || r as usize + frames > frame_count {
        if frames == 1 {
            bail!("guide frame {frame} is outside the video's {frame_count} frames");
        }
        bail!("a {frames} frame guide clip at frame {frame} does not fit in the video's {frame_count} frames");
    }
    Ok(r as usize)
}

/// Frames `[n, H, W, 3]` center-cropped and Lanczos-resized to the canvas.
fn fit_frames(frames: &Tensor, n: usize, w: usize, h: usize) -> Result<Vec<Tensor>> {
    (0..n).map(|i| media::resize_lanczos(&frames.get(i)?, w, h, true)).collect()
}

/// ComfyUI `ModelSampling.percent_to_sigma` for the flow schedule.
fn percent_to_sigma(percent: f64, shift: f64) -> f64 {
    if percent <= 0.0 {
        1.0
    } else if percent >= 1.0 {
        0.0
    } else {
        sampling::time_snr_shift(shift, 1.0 - percent)
    }
}

/// Source video, mask and soundtrack of a masked request, ready for one canvas.
struct MaskedSource {
    latent: Tensor,
    masks: Masks,
}

/// Loaded models, reused across requests when `keep` is set (the server); otherwise each model is
/// dropped as soon as its stage finishes (the CLI, `MemoryConfig::sequential`).
pub struct Engine {
    pub paths: ModelPaths,
    /// DiT used when a request has references.
    pub ref_dit: Option<PathBuf>,
    pub mem: MemoryConfig,
    pub dev: Device,
    pub keep: bool,
    tok: Option<Tokenizer>,
    te: Option<TextEncoder>,
    dit: Option<((PathBuf, Option<(String, u64)>), Dit)>,
    vvae: Option<VideoVae>,
    avae: Option<AudioVae>,
    aenc: Option<AudioEncoder>,
}

impl Engine {
    pub fn new(paths: ModelPaths, ref_dit: Option<PathBuf>, mem: MemoryConfig, dev: Device, keep: bool) -> Self {
        Self { paths, ref_dit, mem, dev, keep, tok: None, te: None, dit: None, vvae: None, avae: None, aenc: None }
    }

    /// Load everything a plain t2va/fl2va request needs (server warm-up). The DiT loads last so its
    /// automatic GPU residency sees what the other models left free.
    pub fn preload_with(&mut self, lora: Option<&(String, f64)>) -> Result<()> {
        self.tokenizer()?;
        self.text_encoder()?;
        self.video_vae()?;
        self.audio_vae()?;
        self.audio_encoder()?;
        let p = self.paths.dit.clone();
        self.dit(&p, lora)?;
        Ok(())
    }

    fn tokenizer(&mut self) -> Result<&Tokenizer> {
        if self.tok.is_none() {
            self.tok = Some(Tokenizer::load_default()?);
        }
        Ok(self.tok.as_ref().unwrap())
    }

    fn text_encoder(&mut self) -> Result<&TextEncoder> {
        if self.te.is_none() {
            self.te = Some(TextEncoder::load(&self.paths.text_encoder.display().to_string(), &self.dev, &self.mem)?);
            log::info(&format!("text encoder loaded ({})", vram()));
        }
        Ok(self.te.as_ref().unwrap())
    }

    pub fn video_vae(&mut self) -> Result<&VideoVae> {
        if self.vvae.is_none() {
            self.vvae = Some(VideoVae::load(&self.paths.video_vae.display().to_string(), &self.dev, &self.mem)?);
        }
        Ok(self.vvae.as_ref().unwrap())
    }

    fn audio_vae(&mut self) -> Result<&AudioVae> {
        if self.avae.is_none() {
            self.avae = Some(AudioVae::load(&self.paths.audio_vae.display().to_string(), &self.dev)?);
        }
        Ok(self.avae.as_ref().unwrap())
    }

    fn audio_encoder(&mut self) -> Result<&AudioEncoder> {
        if self.aenc.is_none() {
            self.aenc = Some(AudioEncoder::load(&self.paths.audio_vae.display().to_string(), &self.dev)?);
        }
        Ok(self.aenc.as_ref().unwrap())
    }

    fn dit(&mut self, path: &Path, lora: Option<&(String, f64)>) -> Result<&mut Dit> {
        let key = (path.to_path_buf(), lora.map(|(p, s)| (p.clone(), s.to_bits())));
        if self.dit.as_ref().map(|(k, _)| k != &key).unwrap_or(true) {
            self.dit = None; // free the old one first
            let t = Instant::now();
            let l = lora.map(|(p, s)| Lora::open(p).map(|l| (l, *s))).transpose()?;
            let d = Dit::load(&path.display().to_string(), &self.dev, &self.mem, l.as_ref().map(|(l, s)| (l, *s)))?;
            log::info(&format!("DiT loaded in {:.1}s ({})", t.elapsed().as_secs_f64(), vram()));
            self.dit = Some((key, d));
        }
        Ok(&mut self.dit.as_mut().unwrap().1)
    }

    /// Drop a stage's model unless the engine keeps models loaded.
    fn release(&mut self, what: &str) {
        if self.keep {
            return;
        }
        match what {
            "te" => self.te = None,
            "dit" => self.dit = None,
            "vvae" => self.vvae = None,
            "avae" => self.avae = None,
            "aenc" => self.aenc = None,
            _ => {}
        }
    }
}

fn vram() -> String {
    match vram_info() {
        Ok((f, t)) => format!("{:.1}/{:.1} GB VRAM used", (t - f) as f64 / 1e9, t as f64 / 1e9),
        Err(_) => String::new(),
    }
}

pub fn generate(req: &Request, paths: &ModelPaths, mem: &MemoryConfig, dev: &Device) -> Result<()> {
    let mut eng = Engine::new(paths.clone(), None, mem.clone(), dev.clone(), false);
    eng.generate(req)
}

impl Engine {
pub fn generate(&mut self, req: &Request) -> Result<()> {
    let dev = &self.dev.clone();
    let t_start = Instant::now();
    let first = req.first_frame.as_deref().map(media::load_image).transpose()?;
    let last = req.last_frame.as_deref().map(media::load_image).transpose()?;
    let src_aspect = first.as_ref().or(last.as_ref()).map(|i| (i.dim(1).unwrap() as f64, i.dim(0).unwrap() as f64)).or(req.aspect);
    let (w, h) = match (req.width, req.height, src_aspect) {
        (Some(w), Some(h), _) => (w / 32 * 32, h / 32 * 32),
        (Some(w), None, a) => {
            let (aw, ah) = a.unwrap_or((16.0, 9.0));
            (w / 32 * 32, r32(w as f64 * ah / aw))
        }
        (None, Some(h), a) => {
            let (aw, ah) = a.unwrap_or((16.0, 9.0));
            (r32(h as f64 * aw / ah), h / 32 * 32)
        }
        (None, None, Some((aw, ah))) => canvas_for_aspect(aw, ah, req.megapixels),
        (None, None, None) => (1344, 768),
    };
    if w == 0 || h == 0 {
        bail!("width/height must be at least 32");
    }
    // `(w, h)` is the requested canvas, sampled first; with an upscale `(fw, fh)` is the larger one delivered
    let (fw, fh) = match &req.upscale {
        Some(u) => {
            if !(u.scale > 1.0 && u.scale <= upscaler::MAX_SCALE) {
                bail!("upscale must be above 1 and at most {}", upscaler::MAX_SCALE);
            }
            if !(u.split > 0.0 && u.split < 1.0) || req.steps < 2 {
                bail!("upscale needs at least 2 steps and a split strictly between 0 and 1");
            }
            let (fw, fh) = (r32(w as f64 * u.scale), r32(h as f64 * u.scale));
            if fw <= w && fh <= h {
                bail!("upscale {} leaves a {w}x{h} canvas unchanged", u.scale);
            }
            (fw, fh)
        }
        None => (w, h),
    };
    let (frame_count, latent_t, audio_t) = temporal_shape(req.length);
    progress::check()?;
    progress::stage(Stage::Encoding);
    log::info(&format!(
        "{fw}x{fh}{}, {frame_count} frames ({:.2} s) -> latent {latent_t}x{}x{}, audio {audio_t}",
        if req.upscale.is_some() { format!(" (sampled at {w}x{h}, then upscaled {:.2}x)", fw as f64 / w as f64) } else { String::new() },
        frame_count as f64 / FPS as f64,
        h / 16,
        w / 16
    ));

    // keyframe images on the canvas
    let mut images = vec![];
    let mut kf_index = vec![];
    if let Some(img) = &first {
        images.push(media::resize_lanczos(img, w, h, req.fit == Some(Fit::Crop))?);
        kf_index.push(0usize);
    }
    if let Some(img) = &last {
        images.push(media::resize_lanczos(img, w, h, req.fit != Some(Fit::Stretch))?);
        kf_index.push(frame_count - 1);
    }

    // the same keyframes on the delivered canvas, for the refining pass
    let mut images_final = vec![];
    if req.upscale.is_some() {
        if let Some(img) = &first {
            images_final.push(media::resize_lanczos(img, fw, fh, req.fit == Some(Fit::Crop))?);
        }
        if let Some(img) = &last {
            images_final.push(media::resize_lanczos(img, fw, fh, req.fit != Some(Fit::Stretch))?);
        }
    }

    // references (MiniMaxH3ReferenceToVideo): images, then videos (soundtrack label first), then audio
    let mut ref_items = vec![];
    let mut ref_imgs: Vec<Tensor> = vec![];
    let mut ref_vids: Vec<(Tensor, Option<Tensor>)> = vec![];
    let mut ref_auds: Vec<Tensor> = vec![];
    let ordered: Vec<&RefInput> = req
        .refs
        .iter()
        .filter(|r| matches!(r, RefInput::Image(_)))
        .chain(req.refs.iter().filter(|r| matches!(r, RefInput::Video { .. })))
        .chain(req.refs.iter().filter(|r| matches!(r, RefInput::Audio(_))))
        .collect();
    for r in ordered {
        match r {
            RefInput::Image(p) => {
                let img = media::load_image(p)?;
                let (ih, iw) = (img.dim(0)? as f64, img.dim(1)? as f64);
                let scale = match req.ref_image_size {
                    RefImageSize::Match => (((w * h) as f64) / (iw * ih)).sqrt().min(1.0),
                    RefImageSize::Max => (2048.0 / iw.min(ih)).min(1.0),
                };
                let img = media::resize_lanczos(&img, r32(iw * scale), r32(ih * scale), false)?;
                ref_items.push(RefItem::Image(img.clone()));
                ref_imgs.push(img);
            }
            RefInput::Video { path, audio } => {
                let (vw, vh) = media::video_size(path)?;
                let (mut cw, mut ch) = adapt_canvas(vw, vh);
                if vw * vh < cw * ch {
                    (cw, ch) = (r32(vw as f64), r32(vh as f64));
                }
                let raw = media::load_video(path, FPS as f64, frame_count)?;
                let mut n = raw.dim(0)?;
                if n < 5 {
                    bail!("reference video {path} needs at least 5 frames (~0.2 s at 24 fps)");
                }
                while n % 17 != 5 {
                    n -= 1;
                }
                let mut fr = vec![];
                for i in 0..n {
                    fr.push(media::resize_lanczos(&raw.get(i)?, cw, ch, false)?);
                }
                let frames = Tensor::stack(&fr, 0)?; // [n, ch, cw, 3]
                let sound = match audio {
                    Some(a) => Some(media::load_audio(a, 32000, None)?),
                    None => None,
                };
                if sound.is_some() {
                    ref_items.push(RefItem::Audio);
                }
                // Qwen sees the clip at 2 fps with timestamps
                let idx: Vec<usize> = (0..n).step_by(FPS / 2).collect();
                let q = Tensor::stack(&idx.iter().map(|&i| frames.get(i)).collect::<candle_core::Result<Vec<_>>>()?, 0)?;
                ref_items.push(RefItem::Video { frames: q, timestamps: (0..idx.len()).map(|i| i as f64 / 2.0).collect() });
                log::info(&format!("reference video {path}: {n} frames at {cw}x{ch}{}", if sound.is_some() { " + soundtrack" } else { "" }));
                ref_vids.push((frames, sound));
            }
            RefInput::Audio(p) => {
                ref_items.push(RefItem::Audio);
                ref_auds.push(media::load_audio(p, 32000, None)?);
            }
        }
    }
    let ref_mode = !req.refs.is_empty();

    // 1. text (+ vision) conditioning
    let t0 = Instant::now();
    let tok = self.tokenizer()?;
    let pieces = if ref_mode {
        if !images.is_empty() {
            bail!("keyframes (--image/--last-image) and references can't be combined");
        }
        tok.ref_presentation(&req.prompt, &ref_items)?
    } else {
        tok.presentation(&req.prompt, &images)?
    };
    let enc = self.text_encoder()?.encode(&pieces)?;
    self.release("te");
    log::info(&format!("conditioning: {} tokens in {:.1}s", enc.hidden.dim(0)?, t0.elapsed().as_secs_f64()));

    // 2. keyframe / guide / reference / source latents
    if req.mask.is_some() != req.source.is_some() {
        bail!("a denoise mask needs a source video (--source) and the other way round");
    }
    if let Some(c) = &req.control {
        if c.video.is_none() && req.mask.is_none() {
            bail!("a ControlNet needs a control video, or a source and mask to inpaint");
        }
    }
    let masked = req.source.is_some();
    // pixel inputs that outlive the encoder: guide pictures, the source video, the mask, the control video
    let guide_media: Vec<Option<Tensor>> = req
        .guides
        .iter()
        .map(|g| match g.kind {
            GuideKind::Visual => media::load_frames(&g.path, FPS as f64, frame_count).map(Some),
            GuideKind::Audio => Ok(None),
        })
        .collect::<Result<_>>()?;
    let source_raw = match &req.source {
        Some(p) => {
            let raw = media::load_frames(p, FPS as f64, frame_count)?;
            if raw.dim(0)? < frame_count {
                bail!("source {p} has {} frames, the video needs {frame_count}: pass a shorter --length / --seconds", raw.dim(0)?);
            }
            Some(raw)
        }
        None => None,
    };
    let mask_px = req.mask.as_deref().map(|p| media::load_mask(p, FPS as f64, frame_count)).transpose()?;
    let control_info = req.control.as_ref().map(|c| -> Result<_> {
        let f = crate::weights::SafeFile::open(&c.model)?;
        control::inspect(&f)
    }).transpose()?;
    let control_video = match &req.control {
        Some(ControlRequest { video: Some(v), .. }) => {
            let raw = media::load_frames(v, FPS as f64, frame_count)?;
            // bilinear center crop to each canvas, like ComfyUI's `_fit_frames`
            Some(raw)
        }
        _ => None,
    };
    let to_canvas_bilinear = |raw: &Tensor, cw: usize, ch: usize| -> Result<Tensor> {
        let nchw = raw.permute((0, 3, 1, 2))?.contiguous()?;
        Ok(media::resize_bilinear_crop(&nchw, cw, ch)?.permute((0, 2, 3, 1))?.contiguous()?)
    };

    let mut keyframes = vec![];
    let mut keyframes_final = vec![];
    // guides in the chain's order: where each one lands, filled in with its latents below
    let mut guide_kfs: Vec<KeyframeCond> = vec![];
    let mut guide_video_final: Vec<Option<Tensor>> = vec![None; req.guides.len()];
    for (g, frames) in req.guides.iter().zip(&guide_media) {
        let gf = frames.as_ref().map(|f| f.dim(0).map(guide_frames)).transpose()?.unwrap_or(1);
        let at = resolve_guide_frame(g.frame, frame_count, gf)?;
        log::info(&format!("guide {}: {} at frame {at}", g.path, if g.kind == GuideKind::Audio { "audio".to_string() } else { format!("{gf} frame{}", if gf == 1 { "" } else { "s" }) }));
        guide_kfs.push(KeyframeCond { frame_index: at, video: None, audio: None });
    }
    let mut refs = vec![];
    let mut source_video: Option<(MaskedSource, Option<MaskedSource>)> = None;
    let mut hints: (Option<Tensor>, Option<Tensor>) = (None, None);
    let canvases: Vec<(usize, usize)> = if req.upscale.is_some() { vec![(w, h), (fw, fh)] } else { vec![(w, h)] };
    let need_vae = !images.is_empty() || !ref_imgs.is_empty() || !ref_vids.is_empty() || guide_media.iter().any(|g| g.is_some()) || masked || req.control.is_some();
    let guide_frame_counts: Vec<usize> = guide_media.iter().map(|g| g.as_ref().map(|f| f.dim(0).map(guide_frames)).transpose().map(|n| n.unwrap_or(1))).collect::<candle_core::Result<_>>()?;
    // audio of the source, encoded below; its length (in latent frames) decides what the mask can keep
    let src_has_audio = !req.source_mute && req.source.as_deref().is_some_and(media::has_audio);
    if req.mask_audio == MaskAudio::Keep && masked && !src_has_audio {
        bail!("--mask-audio keep needs a source with a soundtrack");
    }
    // the source soundtrack, encoded first: its length enters the audio mask
    let mut src_audio: Option<Tensor> = None;
    let mut src_audio_kept: Option<usize> = None;
    if src_has_audio {
        let wave = media::load_audio(req.source.as_deref().unwrap(), 32000, Some(frame_count as f64 / FPS as f64))?;
        let z = self.audio_encoder()?.encode(&wave)?;
        self.release("aenc");
        let kept = z.dim(3)?.min(audio_t);
        let z = z.narrow(3, 0, kept)?.contiguous()?;
        // silence-padded to the video's audio length; the mask always generates the padding
        src_audio = Some(if kept < audio_t { Tensor::cat(&[&z, &z.narrow(3, 0, 1)?.zeros_like()?.repeat((1, 1, 1, audio_t - kept))?], 3)? } else { z });
        src_audio_kept = Some(kept);
        log::info(&format!("source soundtrack: {:.2}s", kept as f64 / AUDIO_LATENT_FPS as f64));
    }
    if need_vae {
        // a kept server holds the DiT's resident blocks and the last render's cached buffers; a video encode needs
        // more room than they leave, so the blocks stream from RAM until sampling
        if let Some((_, d)) = self.dit.as_mut() {
            d.set_resident(0)?;
        }
        crate::kernels::trim_pool();
        let vae = self.video_vae()?;
        for (img, idx) in images.iter().zip(&kf_index) {
            keyframes.push(KeyframeCond { frame_index: *idx, video: Some(vae.encode_image(img)?), audio: None });
        }
        for (img, idx) in images_final.iter().zip(&kf_index) {
            keyframes_final.push(KeyframeCond { frame_index: *idx, video: Some(vae.encode_image(img)?), audio: None });
        }
        for img in &ref_imgs {
            refs.push(RefCond::Image { latent: vae.encode_image(img)? });
        }
        for (frames, _) in &ref_vids {
            refs.push(RefCond::Video { latent: vae.encode_frames(frames)?, audio: None });
        }
        // guides: pictures here, soundtracks below
        for (i, frames) in guide_media.iter().enumerate() {
            let Some(frames) = frames else { continue };
            let gf = guide_frame_counts[i];
            let enc = |cw: usize, ch: usize| -> Result<Tensor> {
                let fr = fit_frames(frames, gf, cw, ch)?;
                if gf == 1 {
                    Ok(vae.encode_image(&fr[0])?)
                } else {
                    Ok(vae.encode_frames(&Tensor::stack(&fr, 0)?)?)
                }
            };
            guide_kfs[i].video = Some(enc(w, h)?);
            if req.upscale.is_some() {
                guide_video_final[i] = Some(enc(fw, fh)?);
            }
        }
        // the source video on each canvas, the mask on the latent grid
        if let (Some(raw), Some(mask)) = (&source_raw, &mask_px) {
            let mut per_canvas = vec![];
            let mut canvas_frames = vec![];
            for &(cw, ch) in &canvases {
                let frames = Tensor::stack(&fit_frames(raw, frame_count, cw, ch)?, 0)?;
                let t = Instant::now();
                let latent = vae.encode_frames(&frames)?;
                if latent.dim(2)? != latent_t {
                    bail!("source encodes to {} latent frames, the video has {latent_t}", latent.dim(2)?);
                }
                log::info(&format!("source video encoded at {cw}x{ch} in {:.1}s", t.elapsed().as_secs_f64()));
                let masks = Masks::new(mask, (latent_t, ch / 16, cw / 16), audio_t, frame_count, req.mask_audio, src_audio_kept)?;
                per_canvas.push(MaskedSource { latent, masks });
                canvas_frames.push(frames);
            }
            let mut it = per_canvas.into_iter();
            source_video = Some((it.next().unwrap(), it.next()));
            // ControlNet hints on each canvas (the source behind the mask is the one the sampler uses)
            if let (Some(info), Some(_)) = (&control_info, &req.control) {
                let mut out = vec![];
                for (i, &(cw, ch)) in canvases.iter().enumerate() {
                    let cv = control_video.as_ref().map(|v| to_canvas_bilinear(v, cw, ch)).transpose()?;
                    let target = [1, 24, latent_t, ch / 16, cw / 16];
                    let inputs = HintInputs { video: cv.as_ref(), mask: Some(mask), source: Some(&canvas_frames[i]) };
                    out.push(control::encode_hint(vae, info.inpaint_post_norm, target, &inputs)?);
                }
                let mut it = out.into_iter();
                hints = (it.next(), it.next());
            }
        } else if let (Some(info), Some(_)) = (&control_info, &req.control) {
            let mut out = vec![];
            for &(cw, ch) in &canvases {
                let cv = control_video.as_ref().map(|v| to_canvas_bilinear(v, cw, ch)).transpose()?;
                let inputs = HintInputs { video: cv.as_ref(), mask: None, source: None };
                out.push(control::encode_hint(vae, info.inpaint_post_norm, [1, 24, latent_t, ch / 16, cw / 16], &inputs)?);
            }
            let mut it = out.into_iter();
            hints = (it.next(), it.next());
        }
        self.release("vvae");
    }
    if req.guides.iter().any(|g| g.kind == GuideKind::Audio) {
        for (i, g) in req.guides.iter().enumerate().filter(|(_, g)| g.kind == GuideKind::Audio) {
            let at = guide_kfs[i].frame_index;
            let wave = media::load_audio(&g.path, 32000, None)?;
            let z = self.audio_encoder()?.encode(&wave)?;
            // the streams share one time axis: FRAME_RESCALE per pixel frame, 1.0 per audio latent frame
            let max_rt = (audio_t as f64 - FRAME_RESCALE * at as f64).floor();
            if max_rt < 1.0 {
                bail!("guide frame {} is past the end of the video's audio track", g.frame);
            }
            let rt = z.dim(3)?.min(max_rt as usize);
            let z = z.narrow(3, 0, rt)?.contiguous()?;
            log::info(&format!("audio guide {}: {:.2}s at frame {at}", g.path, rt as f64 / AUDIO_LATENT_FPS as f64));
            guide_kfs[i].audio = Some(z);
        }
        self.release("aenc");
    }
    if req.upscale.is_some() {
        for (kf, video) in guide_kfs.iter().zip(guide_video_final) {
            keyframes_final.push(KeyframeCond { video, ..kf.clone() });
        }
    }
    keyframes.extend(guide_kfs);
    if ref_vids.iter().any(|v| v.1.is_some()) || !ref_auds.is_empty() {
        let aenc = self.audio_encoder()?;
        let mut k = 0;
        for r in refs.iter_mut() {
            if let RefCond::Video { audio, .. } = r {
                if let Some(s) = &ref_vids[k].1 {
                    *audio = Some(aenc.encode(s)?);
                }
                k += 1;
            }
        }
        for a in &ref_auds {
            refs.push(RefCond::Audio { latent: aenc.encode(a)? });
        }
        self.release("aenc");
    }

    // 3. sample (with an upscale: first part small, latent upscale, rest at the delivered size)
    if ref_mode && !self.ref_dit.as_ref().is_some_and(|p| p.is_file()) {
        match &self.ref_dit {
            Some(p) => crate::variants::ensure_path(p)?,
            None => {
                let dir = self.paths.models_dir.join("diffusion_models");
                self.ref_dit = Some(crate::download::ensure(&dir, "minimax_h3_ref2va_pruned_int8_convrot.safetensors")?);
            }
        }
    }
    let dit_path = if ref_mode { self.ref_dit.clone().unwrap_or(self.paths.dit.clone()) } else { self.paths.dit.clone() };
    if req.upscale.is_some() && self.paths.upscaler.is_none() {
        self.paths.upscaler = Some(crate::download::ensure(&self.paths.models_dir.join("latent_upscale_models"), DEFAULT_UPSCALER)?);
    }
    let upscaler_path = self.paths.upscaler.clone();
    let dev_for_upscaler = self.dev.clone();
    let keep_models = self.keep;
    let offload = self.mem.offload;
    let dit = self.dit(&dit_path, req.lora.as_ref())?;
    if dit.resident_blocks() != dit.base_resident_blocks() {
        dit.set_resident(dit.base_resident_blocks())?;
    }
    dit.cfg.shift_video = req.shift_video;
    dit.cfg.shift_audio = req.shift_audio;
    // the Fun ControlNet rides on the DiT; it stays attached while requests keep using the same checkpoint
    match &req.control {
        Some(c) => {
            let path = c.model.display().to_string();
            if dit.control_path() != Some(path.as_str()) {
                let t = Instant::now();
                let size = std::fs::metadata(&c.model).map(|m| m.len() as usize).unwrap_or(0);
                let free = vram_info().map(|(f, _)| f).unwrap_or(0);
                let place = if dit.fully_resident() && free > size + (6 << 30) { crate::weights::Placement::Gpu } else { offload };
                dit.set_control(Some(Control::load(&path, dev, place, candle_core::DType::BF16)?))?;
                log::info(&format!("ControlNet {} loaded in {:.1}s ({})", c.model.display(), t.elapsed().as_secs_f64(), vram()));
            }
        }
        None if dit.has_control() => {
            dit.set_control(None)?;
            crate::kernels::trim_pool();
        }
        None => {}
    }
    let sigmas = sampling::simple_sigmas(req.shift_video, req.steps);
    // steps sampled at the first size
    let split = match &req.upscale {
        Some(u) => ((req.steps as f64 * u.split).round() as usize).clamp(1, req.steps - 1),
        None => req.steps,
    };
    let cond = DitCond {
        text: enc.hidden,
        tags: Some(enc.tags),
        keyframes,
        refs,
        seed: req.seed,
        visual_cond_noise_aug: VISUAL_COND_TIMESTEP,
        audio_cond_noise_aug: AUDIO_COND_TIMESTEP,
        denoise: source_video.as_ref().map(|(s, _)| s.masks.token_masks()),
        control: req.control.as_ref().zip(hints.0.take()).map(|(c, latent)| ControlCond {
            latent,
            strength: c.strength,
            sigma_start: percent_to_sigma(c.start, req.shift_video),
            sigma_end: percent_to_sigma(c.end, req.shift_video),
        }),
    };
    let cond_final = req.upscale.map(|_| DitCond {
        keyframes: std::mem::take(&mut keyframes_final),
        denoise: source_video.as_ref().and_then(|(_, f)| f.as_ref()).map(|s| s.masks.token_masks()),
        control: cond.control.as_ref().zip(hints.1.take()).map(|(c, latent)| ControlCond { latent, ..c.clone() }),
        ..cond.clone()
    });
    let vshape = [1usize, 24, latent_t, h / 16, w / 16];
    let run = dit.prepare(&cond, &vshape, audio_t, sigmas[..=split].to_vec())?;
    drop(cond);
    // noise for the packed latent (audio carried on the video schedule): sigma_max = 1
    // same stream as ComfyUI's RandomNoise: one torch.manual_seed(seed) generator, video then audio
    let mut gen = crate::rng::TorchGen::new(req.seed);
    let nv = gen.randn_tensor(&vshape, dev)?;
    let xa = gen.randn_tensor(&[1, 32, 2, audio_t], dev)?;
    let xv = (&nv * sigmas[0])?;
    let xa = (xa * sigmas[0])?;
    let inpaint = match &source_video {
        Some((s, _)) => Some(Inpaint::new(&s.masks, &s.latent, src_audio.as_ref(), &nv, req.shift_video, req.shift_audio, dev)?),
        None => None,
    };
    drop(nv);
    let t2 = Instant::now();
    let mut last = Instant::now();
    progress::count(Stage::Sampling, 0, req.steps);
    let step_log = |done: usize, last: &mut Instant| -> Result<()> {
        progress::count(Stage::Sampling, done, req.steps);
        log::info(&format!("step {done}/{}  {:.3}s/it  ({})", req.steps, last.elapsed().as_secs_f64(), vram()));
        *last = Instant::now();
        progress::check()
    };
    let first_pass = sampling::sample_denoised(&*dit, &run, xv, xa, &sigmas[..=split], req.sampler, inpaint.as_ref(), |i, _| step_log(i, &mut last))?;
    drop(inpaint);
    drop(run);
    let (zv, za) = match (&req.upscale, cond_final, upscaler_path) {
        (Some(_), Some(cond_final), Some(up_path)) => {
            drop(first_pass.xv);
            // The DiT was loaded with headroom for the first canvas. The upscaler's activations and then a pass with
            // ~scale^2 times the tokens (activations grow with them) need more: stream some resident blocks from
            // RAM to make room, and stream more and redo the stage if the estimate was short.
            let cells = (fh / 16, fw / 16);
            let tokens = latent_t * (fh / 32) * (fw / 32);
            let upscaler_need = (CHUNK_FRAMES.min(latent_t + 10) * cells.0 * cells.1) * UPSCALER_BYTES_PER_VOXEL + (6 << 28);
            let need = dit_activation_bytes(tokens).max(upscaler_need);
            crate::kernels::trim_pool();
            let free = vram_info().map(|(f, _)| f).unwrap_or(usize::MAX);
            let base_resident = dit.resident_blocks();
            let mut keep = base_resident;
            if free < need {
                keep = base_resident.saturating_sub((need - free).div_ceil(dit.block_bytes().max(1)));
                log::info(&format!("upscaling and a {tokens}-token second pass need ~{:.1} GB: {keep}/{base_resident} DiT blocks stay resident", need as f64 / 1e9));
                dit.set_resident(keep)?;
                crate::kernels::trim_pool();
            }
            let is_oom = |e: &anyhow::Error| format!("{e:?}").contains("OUT_OF_MEMORY");
            let mut attempt = 0;
            // the clean estimate of the first pass, upscaled in latent space
            let zv_up = loop {
                attempt += 1;
                match upscale_video_latent(&up_path, &dev_for_upscaler, &first_pass.denoised_v, cells) {
                    Ok(z) => break z,
                    Err(e) if keep > 0 && attempt < 6 && is_oom(&e) => {
                        keep = keep.saturating_sub(keep.div_ceil(4).max(4));
                        log::warn(&format!("out of memory upscaling, retrying with {keep} resident DiT blocks"));
                        dit.set_resident(keep)?;
                        crate::kernels::trim_pool();
                    }
                    Err(e) => return Err(e),
                }
            };
            let sig = &sigmas[split..];
            let vshape = [1usize, 24, latent_t, fh / 16, fw / 16];
            attempt = 0;
            let out = loop {
                attempt += 1;
                let r = (|| -> Result<(Tensor, Tensor)> {
                    // re-noise to where the schedule stands (CONST: x = sigma * noise + (1 - sigma) * latent), then finish
                    let run = dit.prepare(&cond_final, &vshape, audio_t, sig.to_vec())?;
                    let mut gen = crate::rng::TorchGen::new(req.seed);
                    let nv = gen.randn_tensor(&vshape, dev)?;
                    let na = gen.randn_tensor(&[1, 32, 2, audio_t], dev)?;
                    let xv = ((&nv * sig[0])? + (&zv_up * (1.0 - sig[0]))?)?;
                    let xa = ((na * sig[0])? + (&first_pass.denoised_a * (1.0 - sig[0]))?)?;
                    let inpaint = match source_video.as_ref().and_then(|(_, f)| f.as_ref()) {
                        Some(s) => Some(Inpaint::new(&s.masks, &s.latent, src_audio.as_ref(), &nv, req.shift_video, req.shift_audio, dev)?),
                        None => None,
                    };
                    progress::count(Stage::Sampling, split, req.steps);
                    last = Instant::now();
                    sampling::sample(&*dit, &run, xv, xa, sig, req.sampler, inpaint.as_ref(), |i, _| step_log(split + i, &mut last))
                })();
                match r {
                    Ok(o) => break o,
                    Err(e) if keep > 0 && attempt < 6 && is_oom(&e) => {
                        keep = keep.saturating_sub(keep.div_ceil(4).max(4));
                        log::warn(&format!("out of memory in the second pass, retrying with {keep} resident DiT blocks"));
                        dit.set_resident(keep)?;
                        crate::kernels::trim_pool();
                    }
                    Err(e) => return Err(e),
                }
            };
            drop(cond_final);
            // the next request starts with the blocks it was loaded with
            if keep_models {
                crate::kernels::trim_pool();
                dit.set_resident(base_resident)?;
            }
            out
        }
        _ => (first_pass.xv, first_pass.xa),
    };
    let audio_scale = dit.audio_scale();
    log::info(&format!("sampling done in {:.1}s", t2.elapsed().as_secs_f64()));
    self.release("dit");
    let za = (za / audio_scale)?; // carried -> own audio latent

    if let Some(p) = &req.save_latents {
        save_latents(p, &zv, &za)?;
        log::info(&format!("latents saved to {p}"));
    }

    self.decode_and_write(&zv, &za, frame_count, &req.output, req.crf)?;
    log::info(&format!("done in {:.1}s -> {}", t_start.elapsed().as_secs_f64(), req.output));
    Ok(())
}

/// Decode a video latent to `[F, 3, H, W]` frames in [0, 1] (at most `frame_count` of them).
pub fn decode_frames(&mut self, zv: &Tensor, frame_count: usize) -> Result<Tensor> {
    let t = Instant::now();
    progress::check()?;
    progress::stage(Stage::Decoding);
    let frames = self.video_vae()?.decode(zv)?;
    self.release("vvae");
    let nf = frames.dim(1)?.min(frame_count);
    let frames = frames.narrow(1, 0, nf)?.permute((1, 0, 2, 3))?.contiguous()?; // [F, 3, H, W]
    log::info(&format!("video decoded: {nf} frames in {:.1}s", t.elapsed().as_secs_f64()));
    Ok(frames)
}

/// Decode latents and mux them into an MP4.
pub fn decode_and_write(&mut self, zv: &Tensor, za: &Tensor, frame_count: usize, out: &str, crf: u32) -> Result<()> {
    let frames = self.decode_frames(zv, frame_count)?;
    let nf = frames.dim(0)?;
    let t = Instant::now();
    progress::check()?;
    progress::stage(Stage::Muxing);
    let avae = self.audio_vae()?;
    let sr = avae.sample_rate;
    let wave = avae.decode(za)?;
    self.release("avae");
    let samples = ((nf as f64 / FPS as f64) * sr as f64).round() as usize;
    let wave = wave.narrow(1, 0, samples.min(wave.dim(1)?))?;
    log::info(&format!("audio decoded: {:.2}s in {:.1}s", wave.dim(1)? as f64 / sr as f64, t.elapsed().as_secs_f64()));
    media::write_mp4(out, &frames, FPS as f64, Some((&wave, sr)), crf)?;
    if out.ends_with(".mp4") {
        let _ = media::write_wav(&out.replace(".mp4", ".wav"), &wave, sr);
    }
    Ok(())
}
}

/// Decode latents and mux them into an MP4 (one-shot).
#[allow(clippy::too_many_arguments)]
pub fn decode_and_write(zv: &Tensor, za: &Tensor, frame_count: usize, paths: &ModelPaths, mem: &MemoryConfig, dev: &Device, out: &str, crf: u32) -> Result<()> {
    Engine::new(paths.clone(), None, mem.clone(), dev.clone(), false).decode_and_write(zv, za, frame_count, out, crf)
}

/// Frame count implied by a video latent with `t` frames (5k+2 -> 17k+5).
pub fn frames_for_latent(t: usize) -> usize {
    (0..t).map(|k| FRAME_PER_TOKEN[k % 5]).sum()
}

pub fn load_latents(path: &str, dev: &Device) -> Result<(Tensor, Tensor)> {
    let m = candle_core::safetensors::load(path, dev)?;
    Ok((m.get("video").context("video latent")?.to_dtype(DType::F32)?, m.get("audio").context("audio latent")?.to_dtype(DType::F32)?))
}

/// Write sampled latents (`video` `[1, 24, T, H, W]`, `audio` `[1, 32, 2, T]`) for `h3 decode` / `h3 upscale`.
pub fn save_latents(path: &str, zv: &Tensor, za: &Tensor) -> Result<()> {
    let m: std::collections::HashMap<String, Tensor> =
        [("video".to_string(), zv.to_device(&Device::Cpu)?), ("audio".to_string(), za.to_device(&Device::Cpu)?)].into_iter().collect();
    candle_core::safetensors::save(&m, path)?;
    Ok(())
}

/// Upscale a video latent `[1, 24, T, H, W]` to `cells = (height, width)` latent cells with the upscaler
/// checkpoint at `path`. The model is loaded for the call and dropped again, so it never competes with the DiT.
pub fn upscale_video_latent(path: &Path, dev: &Device, zv: &Tensor, cells: (usize, usize)) -> Result<Tensor> {
    let t = Instant::now();
    progress::check()?;
    progress::stage(Stage::Upscaling);
    let up = LatentUpscaler::load(path, dev)?;
    let from = (zv.dim(3)?, zv.dim(4)?);
    let out = up.upscale(zv, cells.0, cells.1, None)?;
    dev.synchronize()?;
    log::info(&format!(
        "latent upscaled {}x{} -> {}x{} ({:.2}x) in {:.1}s ({})",
        from.1,
        from.0,
        cells.1,
        cells.0,
        LatentUpscaler::scale_for(from, cells),
        t.elapsed().as_secs_f64(),
        vram()
    ));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guide_clip_lengths() {
        // 1 picture, or 17k+5 frames: batches are cut down to the grid, shorter than 5 use the first image
        assert_eq!([1, 4, 5, 21, 22, 38, 39, 124].map(guide_frames), [1, 1, 5, 5, 22, 22, 39, 124]);
    }

    #[test]
    fn guide_frames_resolve_like_comfyui() {
        assert_eq!(resolve_guide_frame(0, 124, 1).unwrap(), 0);
        assert_eq!(resolve_guide_frame(-1, 124, 1).unwrap(), 123);
        assert_eq!(resolve_guide_frame(-22, 124, 22).unwrap(), 102);
        assert_eq!(resolve_guide_frame(50, 124, 39).unwrap(), 50);
        assert!(resolve_guide_frame(124, 124, 1).is_err());
        assert!(resolve_guide_frame(-125, 124, 1).is_err());
        assert!(resolve_guide_frame(100, 124, 39).unwrap_err().to_string().contains("39 frame guide clip"));
    }

    #[test]
    fn guide_specs() {
        let g = Guide::parse("a/b.png@-12", GuideKind::Visual);
        assert_eq!((g.path.as_str(), g.frame), ("a/b.png", -12));
        let g = Guide::parse("voice.wav", GuideKind::Audio);
        assert_eq!((g.path.as_str(), g.frame), ("voice.wav", 0));
        // an @ that is not followed by a frame number belongs to the path
        let g = Guide::parse("me@home/clip.mp4", GuideKind::Visual);
        assert_eq!((g.path.as_str(), g.frame), ("me@home/clip.mp4", 0));
    }

    #[test]
    fn control_window_in_sigma() {
        assert_eq!(percent_to_sigma(0.0, 12.0), 1.0);
        assert_eq!(percent_to_sigma(1.0, 12.0), 0.0);
        // halfway through the schedule sits at the shifted 0.5
        assert!((percent_to_sigma(0.5, 12.0) - 12.0 * 0.5 / (1.0 + 11.0 * 0.5)).abs() < 1e-12);
    }
}
