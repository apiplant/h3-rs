/** Everything the home page says about the project, in one place. */

export interface Card {
  name: string;
  href: string;
  tagline: string;
  body: string;
}

export interface Feature {
  title: string;
  body: string;
}

export const SITE = {
  pkg: "h3-rs",
  hasCuda: false,
  bins: ["h3"],
  badges: ["Rust", "CUDA", "int8", "Apache-2.0"],
  hero: { pre: "MiniMax H3 video with sound, ", accent: "on your GPU", post: "." },
  lead: "Text, keyframes or reference pictures in; a clip with stereo audio out, from one model and one binary. h3-rs loads the ComfyUI checkpoints as they are, keeps them quantized on the card, and renders as fast as ComfyUI. No Python at run time.",
  heroNote: "13.0 s per DiT step at 1344×768 × 124 frames on an RTX 4090, against 15.6 s in ComfyUI. The clip above was rendered by h3-rs.",
  demo: undefined as { href: string; label: string } | undefined,
  cargo: `cargo install h3-rs                          # needs the CUDA 13 toolkit (nvcc)
CUDA_COMPUTE_CAP=86 cargo install h3-rs        # pick your GPU's compute capability`,
  terminal: {
    title: "h3 serve",
    command: `h3 serve --preset minimal --steps 8`,
    outputLang: "text",
    output: `  ▐▌ h3  v0.1.0 · MiniMax H3 video + audio

  web UI     http://127.0.0.1:8188/
  API docs   http://127.0.0.1:8188/docs

[h3] 672x384, 73 frames (3.04 s) -> latent 22x24x42, audio 122
[h3] conditioning: 22 tokens in 2.5s
[h3] step 1/8  2.252s/it
[h3] step 2/8  2.298s/it
...
[h3] step 8/8  2.386s/it
[h3] sampling done in 18.5s
[h3] video decoded: 73 frames in 5.9s
[h3] audio decoded: 3.04s in 0.1s
[h3] done in 27.6s`,
  },
  cardsTitle: "One binary, one crate",
  cardsLead: "A command-line renderer, a server with a web studio, and the engine behind them.",
  cards: [
    {
      name: "h3 generate",
      href: "/docs/cli",
      tagline: "Text, frames and references to MP4",
      body: "Text to video, first and last keyframes, up to nine reference pictures, three videos and three voices. Turbo LoRAs, saved latents, and memory presets from 8 GB to 24 GB cards.",
    },
    {
      name: "h3 serve",
      href: "/docs/server",
      tagline: "A studio in your browser",
      body: "Keeps every model loaded and renders requests one at a time. Comes with a web UI, live progress, an OpenAPI spec and a one-line curl API.",
    },
    {
      name: "h3-rs",
      href: "/docs/library",
      tagline: "The engine as a library",
      body: "Build a Request, hand it to the Engine, get an MP4. The DiT, text encoder and both VAEs are separate modules.",
    },
  ] as Card[],
  featuresTitle: "ComfyUI's checkpoints, without ComfyUI",
  featuresLead: "Quantized weights stay quantized in memory and are dequantized layer by layer by NVRTC kernels.",
  features: [
    {
      title: "Video and audio together",
      body: "One DiT denoises a video latent and a stereo audio latent, so the sound is generated with the picture, not added afterwards.",
    },
    {
      title: "Loads the files you already have",
      body: "int8 + ConvRot DiT, NVFP4 + AWQ Qwen3-VL text encoder, fp16 video VAE, fp32 audio VAE: straight from your ComfyUI models folder.",
    },
    {
      title: "Faster than ComfyUI",
      body: "A fused int8 tensor-core GEMM and an int8 flash-style attention kernel: 13.0 s per step at 1344×768 against 15.6 s.",
    },
    {
      title: "Same noise as ComfyUI",
      body: "Torch mt19937 randn, video then audio from one generator: a seed gives the same starting latent as in a ComfyUI workflow.",
    },
    {
      title: "Three ways to start",
      body: "A prompt, a first and/or last keyframe, or reference pictures, videos and voices addressed as <Picture 1>, <Video 1>, <Audio 1>.",
    },
    {
      title: "LoRAs cost nothing",
      body: "Merged into the int8 weights at load. The official turbo LoRAs cut a render to 8 steps, or 4 for references.",
    },
    {
      title: "From 8 GB to 24 GB",
      body: "Presets decide what lives on the GPU and what streams from RAM or disk. A measured peak of 6.5 GB at full size on the smallest.",
    },
    {
      title: "Checked against ComfyUI",
      body: "Dequantization, text encoder, DiT blocks and both VAEs are compared tensor by tensor with dumps from ComfyUI: cosines of 0.9996 to 0.999998.",
    },
    {
      title: "No Python at run time",
      body: "One binary plus CUDA and ffmpeg. The tokenizer and web UI are compiled in.",
    },
  ] as Feature[],
  lib: {
    lead: "The engine is a library: describe a Request, load an Engine once, and render as many clips as you like. The server sits behind a feature.",
    add: `cargo add h3-rs --no-default-features`,
    caption: "src/main.rs",
    snippet: `use h3_rs::config::Preset;
use h3_rs::pipeline::{Engine, ModelPaths, Request};

let paths = ModelPaths::resolve("/models".as_ref(), "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors")?;
let dev = candle_core::Device::new_cuda(0)?;
let mut engine = Engine::new(paths, None, Preset::Gb24.config(), dev, true);

let req = Request { prompt: "A red fox trots through snow.".into(), output: "fox.mp4".into(), ..Default::default() };
engine.generate(&req)?;`,
  },
};
