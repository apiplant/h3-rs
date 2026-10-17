//! Memory configuration: presets for different VRAM / RAM budgets, loadable from TOML.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::weights::Placement;

/// How the pipeline spends VRAM and RAM. Every field has a preset default; see `configs/*.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    /// DiT transformer blocks (of 50, ~0.39 GB each as int8) kept resident on the GPU. The rest
    /// are uploaded from `offload` every sampling step. `-1` = as many as fit (auto).
    pub dit_gpu_blocks: i64,
    /// Qwen3-VL text encoder layers (of 50, ~0.31 GB each as nvfp4) resident while encoding.
    /// `-1` = auto.
    pub te_gpu_layers: i64,
    /// Where non-resident weights wait between uses: `ram` (a copy in process memory, fastest
    /// upload) or `disk` (mmapped file / page cache, lowest RAM use, slowest when RAM is short).
    pub offload: Placement,
    /// Placement of the text encoder's non-resident layers (default: same as `offload`). It runs once per
    /// request, so `disk` keeps it out of pinned RAM at little cost.
    pub te_offload: Option<Placement>,
    /// Free each model as soon as its stage is done (text encoder -> DiT -> VAEs), so only one
    /// stage's weights are resident at a time.
    pub sequential: bool,
    /// Max token rows per GEMM in the DiT (bounds activation memory of the big projections).
    pub token_chunk: usize,
    /// Max token rows per MLP chunk in the DiT (the fc1 output is 5.3x wider than the stream).
    pub mlp_chunk: usize,
    /// VRAM kept free for activations when `-1` (auto) budgets are resolved, in GB.
    pub reserve_gb: f64,
    /// Video VAE tiles (256 px) computed together per decoder layer.
    pub vae_tile_batch: usize,
    /// Keep the video VAE decoder (~4.8 GB fp16) resident; otherwise its layers stream from
    /// `offload` once per 17-frame chunk.
    pub vae_resident: bool,
    /// Compute dtype of the video VAE: `f16` (ComfyUI default), `bf16` or `f32`.
    pub vae_dtype: String,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Preset::Gb24.config()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Preset {
    /// Auto-fit to the current free VRAM, weights offloaded to RAM.
    Auto,
    /// 24 GB GPU (4090 / 3090) with >= 48 GB RAM.
    Gb24,
    /// 16 GB GPU with >= 32 GB RAM.
    Gb16,
    /// 12 GB GPU with >= 32 GB RAM.
    Gb12,
    /// 8 GB GPU; weights stream from disk, low RAM (>= 16 GB) use.
    Gb8,
    /// Smallest footprint: nothing resident, everything streamed from disk, small chunks.
    Minimal,
}

impl Preset {
    pub fn config(self) -> MemoryConfig {
        let base = MemoryConfig {
            dit_gpu_blocks: -1,
            te_gpu_layers: 50,
            offload: Placement::Ram,
            te_offload: None,
            sequential: true,
            token_chunk: 16384,
            mlp_chunk: 8192,
            reserve_gb: 6.0,
            vae_tile_batch: 4,
            vae_resident: true,
            vae_dtype: "f16".into(),
        };
        match self {
            Preset::Auto => MemoryConfig { dit_gpu_blocks: -1, te_gpu_layers: -1, ..base },
            Preset::Gb24 => base,
            Preset::Gb16 => MemoryConfig { dit_gpu_blocks: 14, te_gpu_layers: 30, token_chunk: 8192, mlp_chunk: 4096, vae_tile_batch: 2, ..base },
            Preset::Gb12 => MemoryConfig { dit_gpu_blocks: 4, te_gpu_layers: 16, token_chunk: 8192, mlp_chunk: 2048, vae_tile_batch: 1, ..base },
            Preset::Gb8 => MemoryConfig {
                dit_gpu_blocks: 0,
                te_gpu_layers: 4,
                offload: Placement::Disk,
                token_chunk: 4096,
                mlp_chunk: 1024,
                vae_tile_batch: 1,
                ..base
            },
            Preset::Minimal => MemoryConfig {
                dit_gpu_blocks: 0,
                te_gpu_layers: 0,
                offload: Placement::Disk,
                token_chunk: 2048,
                mlp_chunk: 512,
                vae_tile_batch: 1,
                vae_resident: false,
                ..base
            },
        }
    }
}

impl MemoryConfig {
    pub fn from_toml(path: &str) -> Result<Self> {
        let s = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        #[derive(Deserialize)]
        struct File {
            memory: Option<MemoryConfig>,
        }
        let f: File = toml::from_str(&s).with_context(|| format!("parsing {path}"))?;
        f.memory.context("config file needs a [memory] table")
    }

    pub fn vae_dtype(&self) -> Result<candle_core::DType> {
        Ok(match self.vae_dtype.as_str() {
            "f16" | "fp16" => candle_core::DType::F16,
            "bf16" => candle_core::DType::BF16,
            "f32" | "fp32" => candle_core::DType::F32,
            d => bail!("vae_dtype must be f16, bf16 or f32, got {d}"),
        })
    }
}

/// Free / total VRAM in bytes on the current CUDA context.
pub fn vram_info() -> Result<(usize, usize)> {
    crate::kernels::trim_pool();
    candle_core::cuda_backend::cudarc::driver::result::mem_get_info().map_err(|e| anyhow::anyhow!("mem_get_info: {e:?}"))
}

/// Resolve a "count of resident units" budget: explicit values clamp to `total`, `-1` fits as many
/// `unit_bytes` units as free VRAM allows after `reserve` bytes.
pub fn resolve_resident(requested: i64, total: usize, unit_bytes: usize, reserve: usize) -> usize {
    if requested >= 0 {
        return (requested as usize).min(total);
    }
    let free = vram_info().map(|(f, _)| f).unwrap_or(0);
    (free.saturating_sub(reserve) / unit_bytes.max(1)).min(total)
}
