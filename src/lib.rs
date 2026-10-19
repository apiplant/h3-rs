//! MiniMax H3 audio/video generation with CUDA or Apple Silicon MLX + Metal.

#[cfg(all(feature = "cuda", feature = "mlx"))]
compile_error!("select one backend: --no-default-features --features mlx,serve");
#[cfg(all(feature = "mlx", not(all(target_os = "macos", target_arch = "aarch64"))))]
compile_error!("MLX requires Apple Silicon macOS");

#[cfg(feature = "cuda")]
pub mod attn8;
#[cfg(not(feature = "cuda"))]
#[path = "portable/attn8.rs"]
pub mod attn8;
#[cfg(feature = "mlx")]
pub mod apple;
pub mod backend;
pub mod config;
pub mod control;
pub mod dit;
pub mod download;
pub mod inpaint;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(not(feature = "cuda"))]
#[path = "portable/kernels.rs"]
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
pub mod upscaler;
pub mod vae_audio;
pub mod vae_video;
pub mod weights;
