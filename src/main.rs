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
    /// DiT checkpoint (default: fl2va, or ref2va when references are given).
    #[arg(long)]
    dit: Option<String>,
    #[arg(long, default_value = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")]
    text_encoder: String,
    #[arg(long, default_value = "minimax_h3_video_vae_fp16.safetensors")]
    video_vae: String,
    #[arg(long, default_value = "minimax_h3_audio_vae_fp32.safetensors")]
    audio_vae: String,
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
        #[arg(long, default_value_t = 20)]
        steps: usize,
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
        /// Audio anchored at frame 0 (speech for a talking image, music...). Use with --image; set
        /// --seconds to the audio length.
        #[arg(long)]
        audio_guide: Option<String>,
        /// LoRA file (path, or name in models/loras).
        #[arg(long)]
        lora: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        lora_strength: f64,
        #[arg(long, short, default_value = "h3_output.mp4")]
        output: String,
        /// Also save the sampled latents (safetensors) for `h3 decode`.
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
    /// Serve an HTTP API (POST /v1/generate -> MP4) with all models kept loaded; see `h3_rs::serve`.
    Serve {
        /// Address to listen on
        #[arg(long, default_value = "127.0.0.1:8188")]
        addr: String,
        /// Default sampling steps for requests that leave it out
        #[arg(long, default_value_t = 20)]
        steps: usize,
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
    /// Print the memory presets as TOML.
    Presets,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
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
            audio_guide,
            lora,
            lora_strength,
            output,
            save_latents,
            crf,
            models,
            mem,
        } => {
            let memc = mem.resolve()?;
            let dev = candle_core::Device::new_cuda(mem.device)?;
            h3_rs::kernels::keep_pool_memory(&dev);
            let mut refs: Vec<pipeline::RefInput> = ref_image.into_iter().map(pipeline::RefInput::Image).collect();
            for v in ref_video {
                let audio = if !ref_video_mute && h3_rs::media::has_audio(&v) { Some(v.clone()) } else { None };
                refs.push(pipeline::RefInput::Video { path: v, audio });
            }
            refs.extend(ref_audio.into_iter().map(pipeline::RefInput::Audio));
            let default_dit = if refs.is_empty() { "minimax_h3_fl2va_pruned_int8_convrot.safetensors" } else { "minimax_h3_ref2va_pruned_int8_convrot.safetensors" };
            let paths = ModelPaths::resolve(&models.models_dir, models.dit.as_deref().unwrap_or(default_dit), &models.text_encoder, &models.video_vae, &models.audio_vae)?;
            let aspect = match aspect {
                Some(a) => {
                    let (x, y) = a.split_once(':').ok_or_else(|| anyhow::anyhow!("--aspect must look like 16:9"))?;
                    Some((x.trim().parse::<f64>()?, y.trim().parse::<f64>()?))
                }
                None => None,
            };
            let lora = match lora {
                Some(l) => {
                    let p = if std::path::Path::new(&l).is_file() { l } else { models.models_dir.join("loras").join(&l).display().to_string() };
                    Some((p, lora_strength))
                }
                None => None,
            };
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
                audio_guide,
                output,
                save_latents,
                crf,
            };
            pipeline::generate(&req, &paths, &memc, &dev)?;
        }
        Cmd::Serve { addr, steps, lora, lora_strength, models, mem } => {
            let mut memc = mem.resolve()?;
            // every model stays loaded: stream the text encoder and VAE decoder, give the DiT the rest
            if mem.te_gpu_layers.is_none() && mem.config.is_none() {
                memc.te_gpu_layers = 0;
            }
            if mem.config.is_none() {
                memc.vae_resident = false;
            }
            let dit = models.dit.clone().unwrap_or_else(|| "minimax_h3_fl2va_pruned_int8_convrot.safetensors".into());
            let paths = ModelPaths::resolve(&models.models_dir, &dit, &models.text_encoder, &models.video_vae, &models.audio_vae)?;
            let ref_dit = models.models_dir.join("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors");
            let ref_dit = ref_dit.is_file().then_some(ref_dit);
            let loras_dir = models.models_dir.join("loras");
            let lora = lora.map(|l| {
                let p = if std::path::Path::new(&l).is_file() { l } else { loras_dir.join(&l).display().to_string() };
                (p, lora_strength)
            });
            let info = serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "dit": paths.dit, "ref_dit": ref_dit, "text_encoder": paths.text_encoder,
                "video_vae": paths.video_vae, "audio_vae": paths.audio_vae,
                "default_steps": steps, "default_lora": lora.as_ref().map(|l| l.0.clone()),
                "memory": memc,
            });
            let defaults = h3_rs::serve::Defaults { steps, lora: lora.clone(), loras_dir };
            let device = mem.device;
            h3_rs::serve::run(
                move || {
                    let dev = candle_core::Device::new_cuda(device)?;
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
        Cmd::Decode { latents, output, models, mem } => {
            let memc = mem.resolve()?;
            let dev = candle_core::Device::new_cuda(mem.device)?;
            h3_rs::kernels::keep_pool_memory(&dev);
            let paths = ModelPaths::resolve(&models.models_dir, models.dit.as_deref().unwrap_or("minimax_h3_fl2va_pruned_int8_convrot.safetensors"), &models.text_encoder, &models.video_vae, &models.audio_vae)?;
            let (zv, za) = pipeline::load_latents(&latents, &dev)?;
            let frames = pipeline::frames_for_latent(zv.dim(2)?);
            pipeline::decode_and_write(&zv, &za, frames, &paths, &memc, &dev, &output, 18)?;
        }
    }
    Ok(())
}
