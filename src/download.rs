//! Fetch model files on demand. The required set is downloaded when the engine starts; optional files
//! (reference DiT, LoRAs, ControlNets, the latent upscaler) the first time something asks for them.
//! Downloads resume from `<file>.part`. Set `H3_NO_DOWNLOAD=1` to turn this off, `HF_ENDPOINT` to use a mirror.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};

use crate::dit::log;
use crate::progress::{self, Stage};

const HF: &str = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main";
const UPSCALER_URL: &str = "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors";

struct Known {
    sub: &'static str,
    name: &'static str,
    /// Full URL, or None for a file in the Comfy-Org repo at `<sub>/<name>`
    url: Option<&'static str>,
    mb: u32,
}

const fn k(sub: &'static str, name: &'static str, mb: u32) -> Known {
    Known { sub, name, url: None, mb }
}

const KNOWN: &[Known] = &[
    k("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", 21_000),
    k("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15_700),
    k("vae", "minimax_h3_video_vae_fp16.safetensors", 5_210),
    k("vae", "minimax_h3_audio_vae_fp32.safetensors", 605),
    k("vae", "minimax_h3_video_vae_int8_convrot.safetensors", 2_810),
    k("diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", 21_000),
    k("loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1_960),
    k("loras", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 1_960),
    k("loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 1_960),
    k("model_patches", "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors", 0),
    k("model_patches", "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors", 0),
    Known { sub: "latent_upscale_models", name: "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors", url: Some(UPSCALER_URL), mb: 691 },
];

/// Files this library knows how to download into `<models_dir>/<sub>/`.
pub fn known(sub: &str) -> impl Iterator<Item = &'static str> + '_ {
    KNOWN.iter().filter(move |f| f.sub == sub).map(|f| f.name)
}

/// `<dir>/<name>` if it exists, downloading it first when it's a known model that is missing.
/// `dir` is `<models_dir>/<sub>`; `name` is a bare file name.
pub fn ensure(dir: &Path, name: &str) -> Result<PathBuf> {
    let path = dir.join(name);
    if path.is_file() {
        return Ok(path);
    }
    let sub = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let Some(f) = KNOWN.iter().find(|f| f.sub == sub && f.name == name) else {
        bail!("model file {name} not found in {}", dir.display());
    };
    if std::env::var_os("H3_NO_DOWNLOAD").is_some_and(|v| v != "0" && !v.is_empty()) {
        bail!("model file {name} not found in {} (downloads disabled by H3_NO_DOWNLOAD)", dir.display());
    }
    let url = match f.url {
        Some(u) => u.to_string(),
        None => format!("{}/{}/{}", HF.replacen("https://huggingface.co", &endpoint(), 1), f.sub, f.name),
    };
    std::fs::create_dir_all(dir)?;
    fetch(&url, &path, f.mb).with_context(|| format!("downloading {name}"))?;
    Ok(path)
}

/// Download `url` to `dest` (resumable, honours `H3_NO_DOWNLOAD` and `HF_ENDPOINT`) unless it is already there.
pub fn ensure_url(dest: &Path, url: &str, approx_mb: u32) -> Result<()> {
    if dest.is_file() {
        return Ok(());
    }
    let name = dest.file_name().and_then(|s| s.to_str()).unwrap_or("model");
    if std::env::var_os("H3_NO_DOWNLOAD").is_some_and(|v| v != "0" && !v.is_empty()) {
        bail!("model file {} not found (downloads disabled by H3_NO_DOWNLOAD)", dest.display());
    }
    let url = url.replacen("https://huggingface.co", &endpoint(), 1);
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    fetch(&url, dest, approx_mb).with_context(|| format!("downloading {name}"))
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|s| !s.is_empty()).map(|s| s.trim_end_matches('/').to_string()).unwrap_or_else(|| "https://huggingface.co".into())
}

fn fetch(url: &str, dest: &Path, approx_mb: u32) -> Result<()> {
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(30)).timeout_read(Duration::from_secs(60)).build();
    let mut req = agent.get(url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        // the .part is already complete
        Err(ureq::Error::Status(416, _)) if have > 0 => {
            std::fs::rename(&part, dest)?;
            return Ok(());
        }
        Err(e) => return Err(anyhow!("{e}")),
    };
    let resumed = resp.status() == 206;
    let remaining: u64 = resp.header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let start = if resumed { have } else { 0 };
    let total = if remaining > 0 { start + remaining } else { approx_mb as u64 * 1_000_000 };
    let name = dest.file_name().and_then(|s| s.to_str()).unwrap_or("model");
    log::info(&format!("downloading {name} ({:.1} GB){}", total as f64 / 1e9, if resumed { format!(", resuming at {:.1} GB", have as f64 / 1e9) } else { String::new() }));

    let mut out = std::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part)?;
    let mut body = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = start;
    let mut last_log = Instant::now();
    loop {
        progress::check()?;
        let n = body.read(&mut buf).context("connection lost (run again to resume)")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        progress::count(Stage::Downloading, (done / 1_000_000) as usize, (total / 1_000_000) as usize);
        if last_log.elapsed() >= Duration::from_secs(5) {
            log::info(&format!("  {name}: {:.1} / {:.1} GB", done as f64 / 1e9, total as f64 / 1e9));
            last_log = Instant::now();
        }
    }
    out.flush()?;
    drop(out);
    if remaining > 0 && done != total {
        bail!("download of {name} ended early ({done} of {total} bytes); run again to resume");
    }
    std::fs::rename(&part, dest)?;
    log::info(&format!("downloaded {name}"));
    Ok(())
}
