//! Where the engine is in the current request. The pipeline reports into a few atomics as it works;
//! `h3 serve` publishes a [`snapshot`] at `GET /v1/status` so the web UI can draw a progress bar.
//!
//! One request runs at a time (the engine is single-tenant), so a process-wide state is enough.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Stage {
    Idle = 0,
    /// Reading the prompt and any keyframes / references (text encoder, VAE encoders)
    Encoding = 1,
    /// The DiT denoising loop: `done` of `total` steps
    Sampling = 2,
    /// Video VAE decode: `done` of `total` chunks
    Decoding = 3,
    /// Audio decode and MP4 mux
    Muxing = 4,
    /// Latent upscaler between the two sampling passes of an upscaled request
    Upscaling = 5,
    /// Fetching a missing model file: `done` of `total` megabytes
    Downloading = 6,
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Idle => "idle",
            Stage::Encoding => "encoding",
            Stage::Sampling => "sampling",
            Stage::Decoding => "decoding",
            Stage::Muxing => "muxing",
            Stage::Upscaling => "upscaling",
            Stage::Downloading => "downloading",
        }
    }
}

static STAGE: AtomicU8 = AtomicU8::new(0);
static DONE: AtomicU32 = AtomicU32::new(0);
static TOTAL: AtomicU32 = AtomicU32::new(0);

/// Enter a stage that has no countable steps.
pub fn stage(s: Stage) {
    DONE.store(0, Relaxed);
    TOTAL.store(0, Relaxed);
    STAGE.store(s as u8, Relaxed);
}

/// Enter (or advance within) a stage that counts: `done` of `total` finished.
pub fn count(s: Stage, done: usize, total: usize) {
    DONE.store(done as u32, Relaxed);
    TOTAL.store(total as u32, Relaxed);
    STAGE.store(s as u8, Relaxed);
}

pub fn idle() {
    stage(Stage::Idle)
}

/// `(stage, done, total)`; `total` is 0 for stages that don't count.
pub fn snapshot() -> (Stage, u32, u32) {
    let s = match STAGE.load(Relaxed) {
        1 => Stage::Encoding,
        2 => Stage::Sampling,
        3 => Stage::Decoding,
        4 => Stage::Muxing,
        5 => Stage::Upscaling,
        6 => Stage::Downloading,
        _ => Stage::Idle,
    };
    (s, DONE.load(Relaxed), TOTAL.load(Relaxed))
}

// Cancellation: `cancel_all` bumps a generation; a job stamped with an older generation than the current
// one is cancelled, either before it starts or at the next `check` while it runs.
static GEN: AtomicU64 = AtomicU64::new(0);
static RUN: AtomicU64 = AtomicU64::new(0);

/// The generation to stamp a newly queued job with.
pub fn generation() -> u64 {
    GEN.load(Relaxed)
}

/// Cancel the running job and everything queued so far.
pub fn cancel_all() {
    GEN.fetch_add(1, Relaxed);
}

/// Whether a job stamped with `gen` has been cancelled.
pub fn is_cancelled(gen: u64) -> bool {
    GEN.load(Relaxed) != gen
}

/// Mark `gen` as the job now using the engine.
pub fn begin(gen: u64) {
    RUN.store(gen, Relaxed);
}

/// Bail out if the running job was cancelled. Called between steps / chunks / stages.
pub fn check() -> anyhow::Result<()> {
    if is_cancelled(RUN.load(Relaxed)) {
        anyhow::bail!("cancelled");
    }
    Ok(())
}
