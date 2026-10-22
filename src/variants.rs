//! The DiT variants h3 can serve (`--model`). Each one lives in its own folder,
//! `<models_dir>/diffusion_models/<id>/`, so several can sit side by side; the text encoder and VAEs are shared.
//! A file already sitting directly in `diffusion_models/` (a ComfyUI install) is used as is, and a missing one is
//! downloaded into the variant's folder.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

pub struct File {
    pub name: &'static str,
    pub url: &'static str,
    pub mb: u32,
}

pub struct Variant {
    pub id: &'static str,
    pub title: &'static str,
    /// Sampling steps the model is meant to run at.
    pub steps: usize,
    /// Text/first-frame to video.
    pub main: File,
    /// DiT used when the request carries references (`None`: `main` handles them too).
    pub reference: Option<File>,
}

pub const VARIANTS: &[Variant] = &[
    Variant {
        id: "h3",
        title: "MiniMax H3 (official, ComfyUI int8)",
        steps: 20,
        main: File { name: "minimax_h3_fl2va_pruned_int8_convrot.safetensors", url: "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", mb: 21_000 },
        reference: Some(File { name: "minimax_h3_ref2va_pruned_int8_convrot.safetensors", url: "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", mb: 21_000 }),
    },
    Variant {
        id: "fasth3",
        title: "FastH3 Omni (8-step distilled, text + references)",
        steps: 8,
        main: File { name: "fasth3_omniref_8step_pruned_int8_convrot.safetensors", url: "https://huggingface.co/FastVideo/FastH3-Omni-Comfy/resolve/main/diffusion_models/fasth3_omniref_8step_pruned_int8_convrot.safetensors", mb: 22_214 },
        reference: None,
    },
    Variant {
        id: "singularity",
        title: "Singularity v2 (H3 ref2va fine-tune, int8)",
        steps: 20,
        main: File { name: "Minimax-h3_Singularity_ref2va_v2_pruned_int8_convrot.safetensors", url: "https://huggingface.co/WAS/Minimax-h3_Singularity_v2_pruned/resolve/main/Minimax-h3_Singularity_ref2va_v2_pruned_int8_convrot.safetensors", mb: 20_968 },
        reference: None,
    },
];

pub fn find(id: &str) -> Result<&'static Variant> {
    let id = id.trim().to_ascii_lowercase();
    VARIANTS.iter().find(|v| v.id == id).ok_or_else(|| anyhow::anyhow!("unknown model `{id}` (choose one of: {})", VARIANTS.iter().map(|v| v.id).collect::<Vec<_>>().join(", ")))
}

impl Variant {
    /// Where `file` goes when it has to be downloaded.
    fn home(&self, models_dir: &Path, file: &File) -> PathBuf {
        models_dir.join("diffusion_models").join(self.id).join(file.name)
    }

    /// The file on disk, in the variant's folder or (a ComfyUI install) directly in `diffusion_models/`.
    fn present(&self, models_dir: &Path, file: &File) -> Option<PathBuf> {
        [self.home(models_dir, file), models_dir.join("diffusion_models").join(file.name)].into_iter().find(|p| p.is_file())
    }

    fn ensure_file(&self, models_dir: &Path, file: &File) -> Result<PathBuf> {
        if let Some(p) = self.present(models_dir, file) {
            return Ok(p);
        }
        let dest = self.home(models_dir, file);
        crate::download::ensure_url(&dest, file.url, file.mb)?;
        Ok(dest)
    }

    /// The text/first-frame DiT, downloaded first when it is missing.
    pub fn ensure_main(&self, models_dir: &Path) -> Result<PathBuf> {
        self.ensure_file(models_dir, &self.main)
    }

    /// Where the reference DiT is (or will be fetched to, on the first request that needs it).
    pub fn reference_path(&self, models_dir: &Path) -> PathBuf {
        match &self.reference {
            Some(f) => self.present(models_dir, f).unwrap_or_else(|| self.home(models_dir, f)),
            None => self.present(models_dir, &self.main).unwrap_or_else(|| self.home(models_dir, &self.main)),
        }
    }

    pub fn downloaded(&self, models_dir: &Path) -> bool {
        self.present(models_dir, &self.main).is_some()
    }

    /// Download everything the variant needs.
    pub fn ensure_all(&self, models_dir: &Path) -> Result<()> {
        self.ensure_file(models_dir, &self.main)?;
        if let Some(f) = &self.reference {
            self.ensure_file(models_dir, f)?;
        }
        Ok(())
    }

    pub fn size_gb(&self) -> f64 {
        (self.main.mb + self.reference.as_ref().map_or(0, |f| f.mb)) as f64 * 1.048576 / 1000.0
    }
}

/// Download a variant's reference DiT when `path` is one of the known files and is missing.
pub fn ensure_path(path: &Path) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    for v in VARIANTS {
        for f in std::iter::once(&v.main).chain(v.reference.iter()) {
            if f.name == name {
                return crate::download::ensure_url(path, f.url, f.mb);
            }
        }
    }
    bail!("model file {} not found", path.display())
}

/// One line per variant: `h3  MiniMax H3 …  42.0 GB  downloaded`.
pub fn listing(models_dir: &Path) -> Vec<String> {
    VARIANTS
        .iter()
        .map(|v| format!("{:<12} {:<52} {:>5.1} GB  {}", v.id, v.title, v.size_gb(), if v.downloaded(models_dir) { "downloaded" } else { "not downloaded" }))
        .collect()
}

/// Ask on the terminal which variant to serve. `None` when there's no terminal to ask on.
pub fn choose(models_dir: &Path) -> Result<Option<&'static Variant>> {
    use std::io::{BufRead, IsTerminal, Write};
    if !std::io::stdin().is_terminal() {
        return Ok(None);
    }
    eprintln!("\nWhich model?\n");
    for (i, l) in listing(models_dir).iter().enumerate() {
        eprintln!("  {}) {l}", i + 1);
    }
    loop {
        eprint!("\nnumber or name [1]: ");
        std::io::stderr().flush()?;
        let mut s = String::new();
        if std::io::stdin().lock().read_line(&mut s)? == 0 {
            return Ok(None);
        }
        let s = s.trim();
        if s.is_empty() {
            return Ok(Some(&VARIANTS[0]));
        }
        if let Some(v) = s.parse::<usize>().ok().and_then(|n| VARIANTS.get(n.wrapping_sub(1))).or_else(|| find(s).ok()) {
            return Ok(Some(v));
        }
        eprintln!("no such model: {s}");
    }
}
