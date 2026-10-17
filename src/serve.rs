//! HTTP server: `POST /v1/generate` renders a prompt (+ optional keyframes / references) to an MP4 with
//! stereo audio, keeping every model loaded between requests.
//!
//! The GPU engine is single-tenant, so one worker thread owns the [`Engine`] and requests queue for it; the
//! ntex handlers only parse requests and forward the result. Endpoints:
//!
//! - `POST /v1/generate`: JSON `{"prompt": ..., "width", "height", "megapixels", "aspect": "9:16", "seconds",
//!   "length", "steps", "seed", "sampler": "res_multistep"|"euler", "shift_video", "shift_audio",
//!   "image": b64, "last_image": b64, "fit": "stretch"|"crop", "ref_images": [b64], "ref_videos": [b64],
//!   "ref_video_mute", "ref_audios": [b64], "ref_image_size": "match"|"max", "lora": name,
//!   "lora_strength", "crf"}` (only `prompt` is required; media fields are base64 file contents, a
//!   `data:` URL prefix is accepted). Returns `video/mp4`. A plain-text body is taken as the prompt.
//! - `GET /v1/info`: models, memory settings, available LoRAs, queue length. `GET /v1/status`: what the engine is
//!   doing right now (stage and step), for progress bars. `GET /health`: `ok`.
//! - `GET /`: a web UI for all of the above (built from `ui/` and embedded). `GET /openapi.json`: the API as
//!   OpenAPI 3.1, and `GET /docs` renders it (Scalar, loaded from a CDN).

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use ntex::http::StatusCode;
use ntex::util::Bytes;
use ntex::web::{self, App, HttpRequest, HttpResponse, HttpServer};
use serde::Deserialize;
use tokio::sync::oneshot;

use crate::pipeline::{Engine, Fit, RefImageSize, RefInput, Request};
use crate::sampling::Sampler;

/// Request defaults (from the `serve` command line).
#[derive(Clone)]
pub struct Defaults {
    pub steps: usize,
    pub lora: Option<(String, f64)>,
    pub loras_dir: PathBuf,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct GenRequest {
    prompt: String,
    width: Option<usize>,
    height: Option<usize>,
    megapixels: Option<f64>,
    aspect: Option<String>,
    seconds: Option<f64>,
    length: Option<usize>,
    steps: Option<usize>,
    seed: Option<u64>,
    sampler: Option<String>,
    shift_video: Option<f64>,
    shift_audio: Option<f64>,
    image: Option<String>,
    last_image: Option<String>,
    fit: Option<String>,
    #[serde(default)]
    ref_images: Vec<String>,
    #[serde(default)]
    ref_videos: Vec<String>,
    #[serde(default)]
    ref_video_mute: bool,
    #[serde(default)]
    ref_audios: Vec<String>,
    ref_image_size: Option<String>,
    /// LoRA file name in models/loras ("" or "none" disables the server default)
    lora: Option<String>,
    lora_strength: Option<f64>,
    crf: Option<u32>,
}

struct Job {
    req: GenRequest,
    done: oneshot::Sender<Result<Vec<u8>>>,
}

/// Per-request scratch directory, removed on drop.
struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn decode_b64(s: &str) -> Result<Vec<u8>> {
    let s = s.split_once("base64,").map(|(_, b)| b).unwrap_or(s);
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => anyhow::bail!("invalid base64"),
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

fn to_request(g: GenRequest, d: &Defaults, dir: &std::path::Path) -> Result<Request> {
    let mut n = 0;
    let mut save = |b64: &str, ext: &str| -> Result<String> {
        n += 1;
        let p = dir.join(format!("in{n}.{ext}"));
        std::fs::write(&p, decode_b64(b64)?)?;
        Ok(p.display().to_string())
    };
    let first_frame = g.image.as_deref().map(|b| save(b, "img")).transpose()?;
    let last_frame = g.last_image.as_deref().map(|b| save(b, "img")).transpose()?;
    let mut refs = vec![];
    for b in &g.ref_images {
        refs.push(RefInput::Image(save(b, "img")?));
    }
    for b in &g.ref_videos {
        let p = save(b, "mp4")?;
        let audio = if !g.ref_video_mute && crate::media::has_audio(&p) { Some(p.clone()) } else { None };
        refs.push(RefInput::Video { path: p, audio });
    }
    for b in &g.ref_audios {
        refs.push(RefInput::Audio(save(b, "audio")?));
    }
    let lora = match g.lora.as_deref() {
        Some("") | Some("none") => None,
        Some(name) => {
            let p = if std::path::Path::new(name).is_absolute() { PathBuf::from(name) } else { d.loras_dir.join(name) };
            if !p.is_file() {
                anyhow::bail!("LoRA {name} not found in {}", d.loras_dir.display());
            }
            Some((p.display().to_string(), g.lora_strength.unwrap_or(1.0)))
        }
        None => d.lora.clone().map(|(p, s)| (p, g.lora_strength.unwrap_or(s))),
    };
    let aspect = g
        .aspect
        .as_deref()
        .map(|a| -> Result<(f64, f64)> {
            let (x, y) = a.split_once(':').ok_or_else(|| anyhow!("aspect must look like 16:9"))?;
            Ok((x.trim().parse()?, y.trim().parse()?))
        })
        .transpose()?;
    let sampler = match g.sampler.as_deref() {
        None | Some("res_multistep") | Some("res-multistep") => Sampler::ResMultistep,
        Some("euler") => Sampler::Euler,
        Some(s) => anyhow::bail!("unknown sampler {s}"),
    };
    let fit = match g.fit.as_deref() {
        None => None,
        Some("stretch") => Some(Fit::Stretch),
        Some("crop") => Some(Fit::Crop),
        Some(f) => anyhow::bail!("fit must be stretch or crop, got {f}"),
    };
    let ref_image_size = match g.ref_image_size.as_deref() {
        None | Some("match") => RefImageSize::Match,
        Some("max") => RefImageSize::Max,
        Some(v) => anyhow::bail!("ref_image_size must be match or max, got {v}"),
    };
    let length = g.seconds.map(|s| (s * crate::pipeline::FPS as f64).round() as usize).or(g.length).unwrap_or(124);
    if length > 24 * 30 {
        anyhow::bail!("at most 30 s per request");
    }
    Ok(Request {
        prompt: g.prompt,
        width: g.width,
        height: g.height,
        megapixels: g.megapixels.unwrap_or(1.0),
        aspect,
        fit,
        refs,
        ref_image_size,
        length,
        steps: g.steps.unwrap_or(d.steps),
        seed: g.seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|t| t.as_nanos() as u64).unwrap_or(0)),
        sampler,
        shift_video: g.shift_video.unwrap_or(12.0),
        shift_audio: g.shift_audio.unwrap_or(3.0),
        first_frame,
        last_frame,
        lora,
        audio_guide: None,
        output: dir.join("out.mp4").display().to_string(),
        save_latents: None,
        crf: g.crf.unwrap_or(18),
    })
}

/// The engine thread: runs jobs one at a time.
fn worker(mut eng: Engine, defaults: Defaults, jobs: std_mpsc::Receiver<Job>, pending: Arc<AtomicUsize>) {
    static N: AtomicUsize = AtomicUsize::new(0);
    for job in jobs {
        let dir = std::env::temp_dir().join(format!("h3-serve-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _scratch = Scratch(dir.clone());
        let r = std::fs::create_dir_all(&dir).map_err(anyhow::Error::from).and_then(|_| {
            let req = to_request(job.req, &defaults, &dir)?;
            eng.generate(&req)?;
            Ok(std::fs::read(&req.output)?)
        });
        if let Err(e) = &r {
            eprintln!("[h3] request failed: {e:#}");
        }
        crate::progress::idle();
        pending.fetch_sub(1, Ordering::Relaxed);
        let _ = job.done.send(r);
    }
}

fn error(status: StatusCode, msg: impl std::fmt::Display) -> HttpResponse {
    HttpResponse::builder(status).content_type("text/plain; charset=utf-8").body(format!("{msg}\n"))
}

async fn generate(jobs: std_mpsc::Sender<Job>, pending: Arc<AtomicUsize>, req: HttpRequest, body: Bytes) -> HttpResponse {
    let is_json = req.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.contains("json")) || body.first() == Some(&b'{');
    let g = if is_json {
        match serde_json::from_slice::<GenRequest>(&body) {
            Ok(r) => r,
            Err(e) => return error(StatusCode::BAD_REQUEST, format!("bad JSON: {e}")),
        }
    } else {
        GenRequest { prompt: String::from_utf8_lossy(&body).into_owned(), ..Default::default() }
    };
    if g.prompt.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "no prompt");
    }
    let (done, rx) = oneshot::channel();
    pending.fetch_add(1, Ordering::Relaxed);
    if jobs.send(Job { req: g, done }).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match rx.await {
        Ok(Ok(mp4)) => HttpResponse::Ok().content_type("video/mp4").body(mp4),
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
}

/// The web UI's files (`ui/dist`) as `(path, bytes)`, embedded by build.rs; empty when it was not built.
static UI: &[(&str, &[u8])] = include!(concat!(env!("OUT_DIR"), "/ui.rs"));

const DOCS_HTML: &str = r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>h3 API</title>
  </head>
  <body>
    <script id="api-reference" data-url="/openapi.json"></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@1"></script>
  </body>
</html>
"#;

fn openapi() -> String {
    include_str!("openapi.json").replacen("\"VERSION\"", &format!("{:?}", env!("CARGO_PKG_VERSION")), 1)
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("json") => "application/json",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

async fn ui(req: HttpRequest) -> HttpResponse {
    let path = req.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match UI.iter().find(|(p, _)| *p == path) {
        Some((p, bytes)) => {
            let cache = if p.starts_with("assets/") { "public, max-age=31536000, immutable" } else { "no-cache" };
            HttpResponse::Ok().content_type(content_type(p)).header("cache-control", cache).body(Bytes::from_static(bytes))
        }
        None if UI.is_empty() && path == "index.html" => error(
            StatusCode::NOT_FOUND,
            "this h3 was built without its web UI: run `pnpm install && pnpm build` in ui/, then rebuild",
        ),
        None => error(StatusCode::NOT_FOUND, "not found"),
    }
}

/// Names of the LoRA files the server can load (`models/loras/*.safetensors`), sorted.
fn list_loras(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();
    names
}

fn status_json(pending: &AtomicUsize) -> serde_json::Value {
    let queued = pending.load(Ordering::Relaxed);
    let (stage, done, total) = crate::progress::snapshot();
    let state = match (queued, stage) {
        (0, _) => "idle",
        (_, crate::progress::Stage::Idle) => "starting",
        _ => "running",
    };
    serde_json::json!({ "state": state, "queued": queued, "stage": stage.name(), "done": done, "total": total })
}

/// Load the engine with `load` on a dedicated thread (CUDA state stays on the thread that created it), then
/// serve it on `addr` until the process is stopped (Ctrl-C).
pub fn run<F>(load: F, defaults: Defaults, addr: &str, info: serde_json::Value) -> Result<()>
where
    F: FnOnce() -> Result<Engine> + Send + 'static,
{
    let loras_dir = defaults.loras_dir.clone();
    let (jobs, rx) = std_mpsc::channel::<Job>();
    let (ready_tx, ready) = std_mpsc::channel::<Result<()>>();
    let pending = Arc::new(AtomicUsize::new(0));
    let p2 = pending.clone();
    std::thread::Builder::new().name("h3-engine".into()).spawn(move || match load() {
        Ok(eng) => {
            let _ = ready_tx.send(Ok(()));
            worker(eng, defaults, rx, p2)
        }
        Err(e) => {
            let _ = ready_tx.send(Err(e));
        }
    })?;
    ready.recv().map_err(|_| anyhow!("engine thread died while loading"))??;
    serve(jobs, pending, addr.to_string(), info, loras_dir)?;
    Ok(())
}

/// The startup banner: where to point a browser, and what is loaded.
fn banner(addr: &str, info: &serde_json::Value) {
    use std::io::IsTerminal;
    let tty = std::io::stderr().is_terminal();
    let c = |code: &str, t: &str| if tty { format!("\x1b[{code}m{t}\x1b[0m") } else { t.to_string() };
    let host = addr.replace("0.0.0.0", "localhost").replace("[::]", "localhost");
    let file = |k: &str| info[k].as_str().and_then(|p| std::path::Path::new(p).file_name()).map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| "-".into());
    let lora = info["default_lora"].as_str().and_then(|p| std::path::Path::new(p).file_name()).map(|f| f.to_string_lossy().into_owned());
    let ver = info["version"].as_str().unwrap_or("");
    eprintln!();
    eprintln!("  {}  {}", c("1;38;5;215", "▐▌ h3"), c("2", &format!("v{ver} · MiniMax H3 video + audio")));
    eprintln!();
    eprintln!("  {}  {}", c("38;5;215", "web UI   "), c("4", &format!("http://{host}/")));
    eprintln!("  {}  {}", c("38;5;215", "API docs "), c("4", &format!("http://{host}/docs")));
    eprintln!();
    eprintln!("  {}  {}", c("2", "dit      "), file("dit"));
    eprintln!("  {}  {}", c("2", "encoder  "), file("text_encoder"));
    eprintln!("  {}  {} steps{}", c("2", "defaults "), info["default_steps"], lora.map(|l| format!(", lora {l}")).unwrap_or_default());
    eprintln!();
}

#[ntex::main(name = "h3", signals = true)]
async fn serve(jobs: std_mpsc::Sender<Job>, pending: Arc<AtomicUsize>, addr: String, info: serde_json::Value, loras_dir: PathBuf) -> std::io::Result<()> {
    banner(&addr, &info);
    HttpServer::new(async move |_| {
        let (j1, p1, p2, p3) = (jobs.clone(), pending.clone(), pending.clone(), pending.clone());
        let info = info.clone();
        let loras_dir = loras_dir.clone();
        // keyframes and reference media arrive base64-encoded in the JSON body
        let cfg = web::WebAppConfig::new().set_state(web::types::PayloadConfig::new(512 << 20));
        App::new()
            .with_config(cfg)
            .service(web::resource("/health").to(async || "ok\n"))
            .service(web::resource("/openapi.json").to(async || HttpResponse::Ok().content_type("application/json").body(openapi())))
            .service(web::resource("/docs").to(async || HttpResponse::Ok().content_type("text/html; charset=utf-8").body(DOCS_HTML)))
            .service(web::resource("/v1/info").route(web::get().to(move || {
                let mut i = info.clone();
                i["queued"] = serde_json::json!(p2.load(Ordering::Relaxed));
                i["loras"] = serde_json::json!(list_loras(&loras_dir));
                async move { HttpResponse::Ok().json(&i) }
            })))
            .service(web::resource("/v1/status").route(web::get().to(move || {
                let s = status_json(&p3);
                async move { HttpResponse::Ok().header("cache-control", "no-store").json(&s) }
            })))
            .service(
                web::resource("/v1/generate").route(web::post().to(move |req: HttpRequest, body: Bytes| generate(j1.clone(), p1.clone(), req, body))),
            )
            .default_service(web::get().to(ui))
    })
    .bind(addr.as_str(), ntex::SharedCfg::new("h3"))?
    .workers(1)
    .run()
    .await
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_roundtrip() {
        assert_eq!(super::decode_b64("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        assert_eq!(super::decode_b64("data:image/png;base64,aGk=").unwrap(), b"hi");
    }
}
