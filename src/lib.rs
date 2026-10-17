//! MiniMax H3 (audio + video generation) in Rust, loading ComfyUI-format checkpoints.

pub mod attn8;
pub mod config;
pub mod dit;
pub mod kernels;
pub mod lora;
pub mod media;
pub mod ops;
pub mod pipeline;
pub mod progress;
pub mod rng;
pub mod sampling;
#[cfg(feature = "serve")]
pub mod serve;
pub mod text;
pub mod vae_audio;
pub mod vae_video;
pub mod weights;
