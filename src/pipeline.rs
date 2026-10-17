//! End-to-end generation: prompt (+ optional first / last keyframes) -> MP4 with stereo audio.
//! Stages run sequentially (text encoder -> VAE encode -> DiT -> VAE decodes) and, with
//! `MemoryConfig::sequential`, each model is dropped before the next one loads.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::config::{vram_info, MemoryConfig};
use crate::dit::{log, Dit, DitCond, KeyframeCond, RefCond, AUDIO_COND_TIMESTEP, FRAME_PER_TOKEN, VISUAL_COND_TIMESTEP};
use crate::lora::Lora;
use crate::media;
use crate::progress::{self, Stage};
use crate::sampling::{self, Sampler};
use crate::text::{RefItem, TextEncoder, Tokenizer};
use crate::vae_audio::{AudioEncoder, AudioVae};
use crate::vae_video::VideoVae;

pub const FPS: usize = 24;
pub const AUDIO_LATENT_FPS: usize = 40;

/// Model files, ComfyUI layout (`models/{diffusion_models,text_encoders,vae,loras}`).
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub dit: PathBuf,
    pub text_encoder: PathBuf,
    pub video_vae: PathBuf,
    pub audio_vae: PathBuf,
}

impl ModelPaths {
    /// Resolve each name against `<models_dir>/<subdir>/` unless it is already a path to a file.
    pub fn resolve(models_dir: &Path, dit: &str, te: &str, vvae: &str, avae: &str) -> Result<Self> {
        let r = |sub: &str, n: &str| -> Result<PathBuf> {
            let p = Path::new(n);
            if p.is_file() {
                return Ok(p.to_path_buf());
            }
            let c = models_dir.join(sub).join(n);
            if c.is_file() {
                return Ok(c);
            }
            bail!("model file {n} not found (looked in {})", models_dir.join(sub).display())
        };
        Ok(Self { dit: r("diffusion_models", dit)?, text_encoder: r("text_encoders", te)?, video_vae: r("vae", vvae)?, audio_vae: r("vae", avae)? })
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
    /// Audio anchored at frame 0 (ComfyUI `MiniMaxH3AddGuide` with audio): the generated video and its
    /// soundtrack follow it, e.g. speech for a talking avatar. Cropped to the video's duration.
    pub audio_guide: Option<String>,
    pub output: String,
    pub save_latents: Option<String>,
    pub crf: u32,
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
            audio_guide: None,
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

    fn video_vae(&mut self) -> Result<&VideoVae> {
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
    let (frame_count, latent_t, audio_t) = temporal_shape(req.length);
    progress::stage(Stage::Encoding);
    log::info(&format!(
        "{w}x{h}, {frame_count} frames ({:.2} s) -> latent {latent_t}x{}x{}, audio {audio_t}",
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

    // 2. keyframe / reference latents
    let mut keyframes = vec![];
    let mut refs = vec![];
    if !images.is_empty() || !ref_imgs.is_empty() || !ref_vids.is_empty() {
        let vae = self.video_vae()?;
        for (img, idx) in images.iter().zip(&kf_index) {
            keyframes.push(KeyframeCond { frame_index: *idx, video: Some(vae.encode_image(img)?), audio: None });
        }
        for img in &ref_imgs {
            refs.push(RefCond::Image { latent: vae.encode_image(img)? });
        }
        for (frames, _) in &ref_vids {
            refs.push(RefCond::Video { latent: vae.encode_frames(frames)?, audio: None });
        }
        self.release("vvae");
    }
    if let Some(path) = &req.audio_guide {
        let wave = media::load_audio(path, 32000, None)?;
        let z = self.audio_encoder()?.encode(&wave)?;
        // the streams share one time axis; at frame 0 the guide may span the whole soundtrack
        let rt = z.dim(3)?.min(audio_t);
        let z = z.narrow(3, 0, rt)?.contiguous()?;
        log::info(&format!("audio guide {path}: {:.2}s", rt as f64 / AUDIO_LATENT_FPS as f64));
        match keyframes.iter_mut().find(|k| k.frame_index == 0) {
            Some(kf) => kf.audio = Some(z),
            None => keyframes.insert(0, KeyframeCond { frame_index: 0, video: None, audio: Some(z) }),
        }
        self.release("aenc");
    }
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

    // 3. sample
    let dit_path = if ref_mode { self.ref_dit.clone().unwrap_or(self.paths.dit.clone()) } else { self.paths.dit.clone() };
    let dit = self.dit(&dit_path, req.lora.as_ref())?;
    dit.cfg.shift_video = req.shift_video;
    dit.cfg.shift_audio = req.shift_audio;
    let dit = &*dit;
    let sigmas = sampling::simple_sigmas(req.shift_video, req.steps);
    let cond = DitCond {
        text: enc.hidden,
        tags: Some(enc.tags),
        keyframes,
        refs,
        seed: req.seed,
        visual_cond_noise_aug: VISUAL_COND_TIMESTEP,
        audio_cond_noise_aug: AUDIO_COND_TIMESTEP,
    };
    let vshape = [1usize, 24, latent_t, h / 16, w / 16];
    let run = dit.prepare(&cond, &vshape, audio_t, sigmas.clone())?;
    drop(cond);
    // noise for the packed latent (audio carried on the video schedule): sigma_max = 1
    // same stream as ComfyUI's RandomNoise: one torch.manual_seed(seed) generator, video then audio
    let mut gen = crate::rng::TorchGen::new(req.seed);
    let xv = gen.randn_tensor(&vshape, dev)?;
    let xa = gen.randn_tensor(&[1, 32, 2, audio_t], dev)?;
    let xv = (xv * sigmas[0])?;
    let xa = (xa * sigmas[0])?;
    let t2 = Instant::now();
    let mut last = Instant::now();
    progress::count(Stage::Sampling, 0, req.steps);
    let (zv, za) = sampling::sample(&dit, &run, xv, xa, &sigmas, req.sampler, |i, n| {
        progress::count(Stage::Sampling, i, n);
        log::info(&format!("step {i}/{n}  {:.3}s/it  ({})", last.elapsed().as_secs_f64(), vram()));
        last = Instant::now();
    })?;
    let audio_scale = dit.audio_scale();
    log::info(&format!("sampling done in {:.1}s", t2.elapsed().as_secs_f64()));
    drop(run);
    self.release("dit");
    let za = (za / audio_scale)?; // carried -> own audio latent

    if let Some(p) = &req.save_latents {
        let m: std::collections::HashMap<String, Tensor> =
            [("video".to_string(), zv.to_device(&Device::Cpu)?), ("audio".to_string(), za.to_device(&Device::Cpu)?)].into_iter().collect();
        candle_core::safetensors::save(&m, p)?;
        log::info(&format!("latents saved to {p}"));
    }

    self.decode_and_write(&zv, &za, frame_count, &req.output, req.crf)?;
    log::info(&format!("done in {:.1}s -> {}", t_start.elapsed().as_secs_f64(), req.output));
    Ok(())
}

/// Decode latents and mux them into an MP4.
pub fn decode_and_write(&mut self, zv: &Tensor, za: &Tensor, frame_count: usize, out: &str, crf: u32) -> Result<()> {
    let t = Instant::now();
    progress::stage(Stage::Decoding);
    let frames = self.video_vae()?.decode(zv)?;
    self.release("vvae");
    let nf = frames.dim(1)?.min(frame_count);
    let frames = frames.narrow(1, 0, nf)?.permute((1, 0, 2, 3))?.contiguous()?; // [F, 3, H, W]
    log::info(&format!("video decoded: {nf} frames in {:.1}s", t.elapsed().as_secs_f64()));
    let t = Instant::now();
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
