use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use h3_rs::config::{MemoryConfig, Preset};
use h3_rs::pipeline::{self, ModelPaths, Request};
use h3_rs::sampling::Sampler;
use h3_rs::weights::Placement;

#[derive(Parser)]
#[command(name = "h3", version, about = "MiniMax H3 audio+video generation (ComfyUI checkpoints, candle/CUDA)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// `./models` or `~/ComfyUI/models` when one exists, else `~/.cache/h3-rs/models`.
fn default_models_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    [PathBuf::from("models"), home.join("ComfyUI/models")].into_iter().find(|p| p.join("diffusion_models").is_dir()).unwrap_or_else(|| home.join(".cache/h3-rs/models"))
}

#[derive(clap::Args, Clone)]
struct Models {
    /// ComfyUI `models` directory (model names below resolve against its subfolders).
    #[arg(long, default_value_os_t = default_models_dir(), env = "H3_MODELS_DIR")]
    models_dir: PathBuf,
    /// Model to run: h3, fasth3 or singularity (each lives in `diffusion_models/<name>/`, downloaded when missing;
    /// see `h3 models`). `serve` asks when this is left out on a terminal.
    #[arg(long, env = "H3_MODEL")]
    model: Option<String>,
    /// DiT checkpoint file or path; overrides --model (default: fl2va, or ref2va when references are given).
    #[arg(long)]
    dit: Option<String>,
    #[arg(long, default_value = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")]
    text_encoder: String,
    #[arg(long, default_value = "minimax_h3_video_vae_fp16.safetensors")]
    video_vae: String,
    #[arg(long, default_value = "minimax_h3_audio_vae_fp32.safetensors")]
    audio_vae: String,
    /// Latent upscaler (models/latent_upscale_models) for `--upscale` and `h3 upscale`.
    #[arg(long, default_value = pipeline::DEFAULT_UPSCALER)]
    upscaler: String,
}

impl Models {
    /// The chosen `--model`, if any and no explicit `--dit` overrides it.
    fn variant(&self) -> Result<Option<&'static h3_rs::variants::Variant>> {
        match (&self.model, &self.dit) {
            (Some(m), None) => Ok(Some(h3_rs::variants::find(m)?)),
            _ => Ok(None),
        }
    }
}

#[derive(clap::Args, Clone)]
struct Mem {
    /// Memory preset (see `h3 presets`).
    #[arg(long, value_enum, default_value = "gb24")]
    preset: Preset,
    /// TOML file with a [memory] table; overrides the preset.
    #[arg(long)]
    config: Option<String>,
    /// Override: DiT blocks resident on the GPU (-1 = auto).
    #[arg(long, allow_hyphen_values = true)]
    dit_gpu_blocks: Option<i64>,
    /// Override: text encoder layers resident on the GPU (-1 = auto).
    #[arg(long, allow_hyphen_values = true)]
    te_gpu_layers: Option<i64>,
    /// Override: where non-resident weights wait (ram | disk).
    #[arg(long)]
    offload: Option<String>,
    /// Override: max token rows per DiT MLP chunk.
    #[arg(long)]
    mlp_chunk: Option<usize>,
    #[arg(long, default_value_t = 0)]
    device: usize,
}

impl Mem {
    fn resolve(&self) -> Result<MemoryConfig> {
        let mut m = match &self.config {
            Some(p) => MemoryConfig::from_toml(p)?,
            None => self.preset.config(),
        };
        if let Some(v) = self.dit_gpu_blocks {
            m.dit_gpu_blocks = v;
        }
        if let Some(v) = self.te_gpu_layers {
            m.te_gpu_layers = v;
        }
        if let Some(v) = &self.offload {
            m.offload = match v.as_str() {
                "ram" => Placement::Ram,
                "disk" => Placement::Disk,
                "gpu" => Placement::Gpu,
                o => anyhow::bail!("--offload must be ram or disk, got {o}"),
            };
        }
        if let Some(v) = self.mlp_chunk {
            m.mlp_chunk = v;
        }
        Ok(m)
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate a video with audio.
    Generate {
        #[arg(long, short)]
        prompt: String,
        /// Canvas width (default: from the keyframe's aspect ratio, else 1344).
        #[arg(long)]
        width: Option<usize>,
        /// Canvas height (default: from the keyframe's aspect ratio, else 768).
        #[arg(long)]
        height: Option<usize>,
        /// Canvas area when width/height follow an aspect ratio (x 1024^2 pixels, step 32).
        #[arg(long, default_value_t = 1.0)]
        megapixels: f64,
        /// Aspect ratio W:H for text-to-video without explicit size (e.g. 9:16).
        #[arg(long)]
        aspect: Option<String>,
        /// Keyframe fitting when the canvas aspect differs (default: first stretch, last crop).
        #[arg(long, value_enum)]
        fit: Option<pipeline::Fit>,
        /// Reference image for ref2va (<Picture i>, repeatable, up to 9).
        #[arg(long)]
        ref_image: Vec<String>,
        /// Reference video for ref2va (<Video k>, repeatable, up to 3); its soundtrack becomes <Audio j>.
        #[arg(long)]
        ref_video: Vec<String>,
        /// Ignore reference videos' soundtracks.
        #[arg(long)]
        ref_video_mute: bool,
        /// Reference audio for ref2va (<Audio j>, repeatable, up to 3).
        #[arg(long)]
        ref_audio: Vec<String>,
        #[arg(long, value_enum, default_value = "match")]
        ref_image_size: pipeline::RefImageSize,
        /// Frame count at 24 fps (snapped up to 17k+5; 124 = ~5 s).
        #[arg(long, default_value_t = 124)]
        length: usize,
        /// Duration in seconds (overrides --length).
        #[arg(long)]
        seconds: Option<f64>,
        /// Sampling steps (default: the model's own, 20 unless --model says otherwise).
        #[arg(long)]
        steps: Option<usize>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, value_enum, default_value = "res-multistep")]
        sampler: Sampler,
        #[arg(long, default_value_t = 12.0)]
        shift_video: f64,
        #[arg(long, default_value_t = 3.0)]
        shift_audio: f64,
        /// First keyframe image (fl2va).
        #[arg(long)]
        image: Option<String>,
        /// Last keyframe image (fl2va).
        #[arg(long)]
        last_image: Option<String>,
        /// Image or clip anchored at a frame, `PATH[@FRAME]` (repeatable; ComfyUI MiniMaxH3AddGuide). A video
        /// file is cut to 5, 22, 39... (17k+5) frames; FRAME defaults to 0 and counts from the end if negative.
        #[arg(long)]
        guide: Vec<String>,
        /// Audio anchored at a frame, `PATH[@FRAME]` (repeatable): speech for a talking image, music... Cropped
        /// to the video's remaining duration; set --seconds to its length. A video file gives its soundtrack.
        #[arg(long, alias = "audio-guide")]
        guide_audio: Vec<String>,
        /// Video (or image) to edit with --mask: the cells the mask keeps come from it, along with its soundtrack.
        #[arg(long, requires = "mask")]
        source: Option<String>,
        /// Ignore the source's soundtrack (the audio is then generated).
        #[arg(long)]
        source_mute: bool,
        /// Denoise mask over --source: an image or video, white = regenerate, black = keep, grey = partly.
        #[arg(long, requires = "source")]
        mask: Option<String>,
        /// What the mask does to the source's audio.
        #[arg(long, value_enum, default_value = "follow")]
        mask_audio: h3_rs::inpaint::MaskAudio,
        /// Fun ControlNet checkpoint (models/model_patches): steers the video with --control-video and/or the
        /// --source + --mask pair as an inpainting hint.
        #[arg(long)]
        controlnet: Option<String>,
        /// Control video for --controlnet (depth, pose, edges...).
        #[arg(long, requires = "controlnet")]
        control_video: Option<String>,
        #[arg(long, default_value_t = 1.0, requires = "controlnet")]
        control_strength: f64,
        /// Share of the sampling at which the ControlNet starts (0 = first step).
        #[arg(long, default_value_t = 0.0, requires = "controlnet")]
        control_start: f64,
        /// Share of the sampling at which the ControlNet stops (1 = last step).
        #[arg(long, default_value_t = 1.0, requires = "controlnet")]
        control_end: f64,
        /// LoRA file (path, or name in models/loras).
        #[arg(long)]
        lora: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        lora_strength: f64,
        /// Sample at the canvas size, upscale the latent SCALE times, and refine the rest of the steps at that
        /// larger size, which is what you get (1 < SCALE <= 4; needs the latent upscaler model).
        #[arg(long)]
        upscale: Option<f64>,
        /// With --upscale: fraction of the steps sampled at the low resolution.
        #[arg(long, default_value_t = 0.5)]
        upscale_split: f64,
        #[arg(long, short, default_value = "h3_output.mp4")]
        output: String,
        /// Also save the sampled latents (safetensors) for `h3 decode` / `h3 upscale`.
        #[arg(long)]
        save_latents: Option<String>,
        #[arg(long, default_value_t = 18)]
        crf: u32,
        #[command(flatten)]
        models: Models,
        #[command(flatten)]
        mem: Mem,
    },
    /// Decode saved latents to an MP4.
    Decode {
        latents: String,
        #[arg(long, short, default_value = "h3_output.mp4")]
        output: String,
        #[command(flatten)]
        models: Models,
        #[command(flatten)]
        mem: Mem,
    },
    /// Upscale a video file (its audio is kept) or saved latents (`generate --save-latents`) with the latent
    /// upscaler: encode, upscale in latent space, decode. This is the raw learned upscale;
    /// `generate --upscale` also re-noises and refines it with the DiT.
    Upscale {
        /// A video file, or latents saved by `generate --save-latents` (`.safetensors`).
        input: String,
        /// Upscale factor (1 - 4) applied to the input's size (default 1.5).
        #[arg(long, conflicts_with_all = ["width", "height"])]
        scale: Option<f64>,
        /// Target width in pixels (rounded to 32); the height follows the aspect unless --height is given.
        #[arg(long)]
        width: Option<usize>,
        #[arg(long)]
        height: Option<usize>,
        #[arg(long, short, default_value = "h3_upscaled.mp4")]
        output: String,
        /// Also save the upscaled latents (latents input only).
        #[arg(long)]
        save_latents: Option<String>,
        #[arg(long, default_value_t = 18)]
        crf: u32,
        #[command(flatten)]
        models: Models,
        #[command(flatten)]
        mem: Mem,
    },
    /// Serve an HTTP API (POST /v1/generate -> MP4) with all models kept loaded; see `h3_rs::serve`.
    Serve {
        /// Address to listen on
        #[arg(long, default_value = "127.0.0.1:8188")]
        addr: String,
        /// Default sampling steps for requests that leave it out (default: the model's own)
        #[arg(long)]
        steps: Option<usize>,
        /// Default LoRA for requests that leave `lora` out (name in models/loras or a path)
        #[arg(long)]
        lora: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        lora_strength: f64,
        #[command(flatten)]
        models: Models,
        #[command(flatten)]
        mem: Mem,
    },
    /// List the models h3 can run and which are downloaded.
    Models {
        #[arg(long, default_value_os_t = default_models_dir(), env = "H3_MODELS_DIR")]
        models_dir: PathBuf,
    },
    /// Download models (h3, fasth3, singularity) into `diffusion_models/<name>/` without running anything.
    Download {
        /// One or more of: h3, fasth3, singularity
        #[arg(required = true)]
        names: Vec<String>,
        #[arg(long, default_value_os_t = default_models_dir(), env = "H3_MODELS_DIR")]
        models_dir: PathBuf,
    },
    /// Print the memory presets as TOML.
    Presets,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Models { models_dir } => {
            println!("models in {}\n", models_dir.display());
            for l in h3_rs::variants::listing(&models_dir) {
                println!("  {l}");
            }
            println!("\nrun one with `h3 serve --model <name>`; it is downloaded on first use (or `h3 download <name>`)");
        }
        Cmd::Download { names, models_dir } => {
            for n in names {
                h3_rs::variants::find(&n)?.ensure_all(&models_dir)?;
            }
        }
        Cmd::Presets => {
            for p in [Preset::Gb24, Preset::Gb16, Preset::Gb12, Preset::Gb8, Preset::Minimal, Preset::Auto] {
                #[derive(serde::Serialize)]
                struct F {
                    memory: MemoryConfig,
                }
                println!("# --preset {:?}\n{}", p, toml::to_string(&F { memory: p.config() })?);
            }
        }
        Cmd::Generate {
            prompt,
            width,
            height,
            megapixels,
            aspect,
            fit,
            ref_image,
            ref_video,
            ref_video_mute,
            ref_audio,
            ref_image_size,
            length,
            seconds,
            steps,
            seed,
            sampler,
            shift_video,
            shift_audio,
            image,
            last_image,
            guide,
            guide_audio,
            source,
            source_mute,
            mask,
            mask_audio,
            controlnet,
            control_video,
            control_strength,
            control_start,
            control_end,
            lora,
            lora_strength,
            upscale,
            upscale_split,
            output,
            save_latents,
            crf,
            models,
            mem,
        } => {
            let memc = mem.resolve()?;
            let dev = h3_rs::backend::new_device(mem.device)?;
            h3_rs::kernels::keep_pool_memory(&dev);
            let mut refs: Vec<pipeline::RefInput> = ref_image.into_iter().map(pipeline::RefInput::Image).collect();
            for v in ref_video {
                let audio = if !ref_video_mute && h3_rs::media::has_audio(&v) { Some(v.clone()) } else { None };
                refs.push(pipeline::RefInput::Video { path: v, audio });
            }
            refs.extend(ref_audio.into_iter().map(pipeline::RefInput::Audio));
            let default_dit = if refs.is_empty() { "minimax_h3_fl2va_pruned_int8_convrot.safetensors" } else { "minimax_h3_ref2va_pruned_int8_convrot.safetensors" };
            let variant = models.variant()?;
            let variant_dit = variant
                .map(|v| -> Result<PathBuf> {
                    if refs.is_empty() {
                        return v.ensure_main(&models.models_dir);
                    }
                    let p = v.reference_path(&models.models_dir);
                    h3_rs::variants::ensure_path(&p)?;
                    Ok(p)
                })
                .transpose()?
                .map(|p| p.display().to_string());
            let steps = steps.or(variant.map(|v| v.steps)).unwrap_or(20);
            let paths = ModelPaths::resolve(&models.models_dir, variant_dit.as_deref().or(models.dit.as_deref()).unwrap_or(default_dit), &models.text_encoder, &models.video_vae, &models.audio_vae)?
                .with_upscaler(&models.models_dir, &models.upscaler, upscale.is_some())?;
            let aspect = match aspect {
                Some(a) => {
                    let (x, y) = a.split_once(':').ok_or_else(|| anyhow::anyhow!("--aspect must look like 16:9"))?;
                    Some((x.trim().parse::<f64>()?, y.trim().parse::<f64>()?))
                }
                None => None,
            };
            let lora = match lora {
                Some(l) => {
                    let p = if std::path::Path::new(&l).is_file() { l } else { h3_rs::download::ensure(&models.models_dir.join("loras"), &l)?.display().to_string() };
                    Some((p, lora_strength))
                }
                None => None,
            };
            let guides: Vec<pipeline::Guide> = guide
                .iter()
                .map(|g| pipeline::Guide::parse(g, pipeline::GuideKind::Visual))
                .chain(guide_audio.iter().map(|g| pipeline::Guide::parse(g, pipeline::GuideKind::Audio)))
                .collect();
            let control = controlnet
                .map(|c| -> anyhow::Result<_> {
                    Ok(pipeline::ControlRequest {
                        model: if std::path::Path::new(&c).is_file() { c.into() } else { h3_rs::download::ensure(&models.models_dir.join("model_patches"), &c)? },
                        video: control_video,
                        strength: control_strength,
                        start: control_start,
                        end: control_end,
                    })
                })
                .transpose()?;
            let length = seconds.map(|s| (s * pipeline::FPS as f64).round() as usize).unwrap_or(length);
            let req = Request {
                prompt,
                width,
                height,
                megapixels,
                aspect,
                fit,
                refs,
                ref_image_size,
                length,
                steps,
                seed,
                sampler,
                shift_video,
                shift_audio,
                first_frame: image,
                last_frame: last_image,
                lora,
                guides,
                source,
                source_mute,
                mask,
                mask_audio,
                control,
                upscale: upscale.map(|scale| pipeline::Upscale { scale, split: upscale_split }),
                output,
                save_latents,
                crf,
            };
            pipeline::generate(&req, &paths, &memc, &dev)?;
        }
        Cmd::Serve { addr, steps, lora, lora_strength, mut models, mem } => {
            let mut memc = mem.resolve()?;
            // every model stays loaded: stream the text encoder and VAE decoder, give the DiT the rest
            if mem.te_gpu_layers.is_none() && mem.config.is_none() {
                memc.te_gpu_layers = 0;
            }
            if mem.config.is_none() {
                memc.vae_resident = false;
            }
            if models.model.is_none() && models.dit.is_none() {
                if let Some(v) = h3_rs::variants::choose(&models.models_dir)? {
                    models.model = Some(v.id.to_string());
                }
            }
            let variant = models.variant()?;
            let steps = steps.or(variant.map(|v| v.steps)).unwrap_or(20);
            let dit = match variant {
                Some(v) => v.ensure_main(&models.models_dir)?.display().to_string(),
                None => models.dit.clone().unwrap_or_else(|| "minimax_h3_fl2va_pruned_int8_convrot.safetensors".into()),
            };
            let paths = ModelPaths::resolve(&models.models_dir, &dit, &models.text_encoder, &models.video_vae, &models.audio_vae)?.with_upscaler(&models.models_dir, &models.upscaler, false)?;
            let ref_dit = match variant {
                Some(v) => v.reference_path(&models.models_dir),
                None => models.models_dir.join("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
            };
            let ref_dit = Some(ref_dit); // fetched on the first request with references
            let loras_dir = models.models_dir.join("loras");
            let lora = lora
                .map(|l| -> anyhow::Result<_> {
                    let p = if std::path::Path::new(&l).is_file() { l } else { h3_rs::download::ensure(&loras_dir, &l)?.display().to_string() };
                    Ok((p, lora_strength))
                })
                .transpose()?;
            let info = serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "model": variant.map(|v| v.id),
                "dit": paths.dit, "ref_dit": ref_dit, "text_encoder": paths.text_encoder,
                "video_vae": paths.video_vae, "audio_vae": paths.audio_vae, "upscaler": paths.upscaler,
                "default_steps": steps, "default_lora": lora.as_ref().map(|l| l.0.clone()),
                "memory": memc,
            });
            let defaults = h3_rs::serve::Defaults { steps, lora: lora.clone(), loras_dir, patches_dir: models.models_dir.join("model_patches") };
            let device = mem.device;
            h3_rs::serve::run(
                move || {
                    let dev = h3_rs::backend::new_device(device)?;
                    h3_rs::kernels::keep_pool_memory(&dev);
                    let mut eng = h3_rs::pipeline::Engine::new(paths, ref_dit, memc, dev, true);
                    eng.preload_with(lora.as_ref())?;
                    Ok(eng)
                },
                defaults,
                &addr,
                info,
            )?;
        }
        Cmd::Upscale { input, scale, width, height, output, save_latents, crf, models, mem } => {
            let memc = mem.resolve()?;
            let dev = h3_rs::backend::new_device(mem.device)?;
            h3_rs::kernels::keep_pool_memory(&dev);
            let paths = ModelPaths::resolve(&models.models_dir, models.dit.as_deref().unwrap_or("minimax_h3_fl2va_pruned_int8_convrot.safetensors"), &models.text_encoder, &models.video_vae, &models.audio_vae)?
                .with_upscaler(&models.models_dir, &models.upscaler, true)?;
            let is_video = !input.ends_with(".safetensors");
            // a video is encoded to latents first (whole 17-frame clips, centre-cropped to a multiple of 16 px)
            let (zv, za, source) = if is_video {
                let raw = h3_rs::media::load_video(&input, pipeline::FPS as f64, pipeline::FPS * 30)?;
                let mut n = raw.dim(0)?;
                if n < 5 {
                    anyhow::bail!("{input} has fewer than 5 frames");
                }
                while n % 17 != 5 {
                    n -= 1;
                }
                let (vh, vw) = (raw.dim(1)?, raw.dim(2)?);
                let (ch, cw) = (vh / 16 * 16, vw / 16 * 16);
                let frames = raw.narrow(0, 0, n)?.narrow(1, (vh - ch) / 2, ch)?.narrow(2, (vw - cw) / 2, cw)?.contiguous()?;
                let mut eng = pipeline::Engine::new(paths.clone(), None, memc.clone(), dev.clone(), false);
                let t = std::time::Instant::now();
                let z = eng.video_vae()?.encode_frames(&frames)?;
                eprintln!("[h3] encoded {n} frames at {cw}x{ch} in {:.1}s", t.elapsed().as_secs_f64());
                (z, None, Some((frames.dim(0)?, input.clone())))
            } else {
                let (zv, za) = pipeline::load_latents(&input, &dev)?;
                (zv, Some(za), None)
            };
            let (h, w) = (zv.dim(3)?, zv.dim(4)?);
            // pixel target rounded to 32, like the canvas of a generation
            let r32 = |v: f64| ((v / 32.0).round() as usize * 32).max(32);
            let (tw, th) = match (scale, width, height) {
                (_, Some(tw), Some(th)) => (r32(tw as f64), r32(th as f64)),
                (_, Some(tw), None) => (r32(tw as f64), r32(tw as f64 * h as f64 / w as f64)),
                (_, None, Some(th)) => (r32(th as f64 * w as f64 / h as f64), r32(th as f64)),
                (s, None, None) => {
                    let s = s.unwrap_or(1.5);
                    (r32(w as f64 * 16.0 * s), r32(h as f64 * 16.0 * s))
                }
            };
            let zv = pipeline::upscale_video_latent(paths.upscaler.as_ref().unwrap(), &dev, &zv, (th / 16, tw / 16))?;
            let frame_count = pipeline::frames_for_latent(zv.dim(2)?);
            match (source, za) {
                (Some((n, path)), _) => {
                    // video in: decode, and carry the original soundtrack over untouched
                    let mut eng = pipeline::Engine::new(paths.clone(), None, memc.clone(), dev.clone(), false);
                    let frames = eng.decode_frames(&zv, n)?;
                    let nf = frames.dim(0)?;
                    let wave = if h3_rs::media::has_audio(&path) {
                        let wv = h3_rs::media::load_audio(&path, 32000, Some(nf as f64 / pipeline::FPS as f64))?;
                        Some(wv)
                    } else {
                        None
                    };
                    h3_rs::media::write_mp4(&output, &frames, pipeline::FPS as f64, wave.as_ref().map(|w| (w, 32000)), crf)?;
                }
                (None, Some(za)) => {
                    if let Some(p) = &save_latents {
                        pipeline::save_latents(p, &zv, &za)?;
                    }
                    pipeline::decode_and_write(&zv, &za, frame_count, &paths, &memc, &dev, &output, crf)?;
                }
                (None, None) => unreachable!(),
            }
        }
        Cmd::Decode { latents, output, models, mem } => {
            let memc = mem.resolve()?;
            let dev = h3_rs::backend::new_device(mem.device)?;
            h3_rs::kernels::keep_pool_memory(&dev);
            let paths = ModelPaths::resolve(&models.models_dir, models.dit.as_deref().unwrap_or("minimax_h3_fl2va_pruned_int8_convrot.safetensors"), &models.text_encoder, &models.video_vae, &models.audio_vae)?;
            let (zv, za) = pipeline::load_latents(&latents, &dev)?;
            let frames = pipeline::frames_for_latent(zv.dim(2)?);
            pipeline::decode_and_write(&zv, &za, frames, &paths, &memc, &dev, &output, 18)?;
        }
    }
    Ok(())
}
