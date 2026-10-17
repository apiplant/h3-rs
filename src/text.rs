//! MiniMax H3 conditioning: Qwen3-VL-32B truncated to 50 layers (port of
//! `comfy/text_encoders/minimax.py`, `qwen3vl.py`, `llama.py`, `qwen35.py` vision tower).
//!
//! The presentation is not chat-templated: raw prompt / label text with vision blocks spliced in.
//! The conditioning is the unnormalized hidden state after the last (50th) layer, plus per-token
//! adaLN tags (0 for vision blocks incl. their start/end tokens, 1 for text).

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::config::{resolve_resident, MemoryConfig};
use crate::dit::log;
use crate::kernels;
use crate::ops;
use crate::weights::{Placement, QLinear, SafeFile};

pub const VISION_START: u32 = 151652;
pub const VISION_END: u32 = 151653;
pub const PAD: u32 = 151643;

/// One element of the presentation.
pub enum Piece {
    Token(u32),
    /// An image `[H, W, 3]` f32 in [0, 1] (CPU), encoded by the vision tower.
    Image(Tensor),
    /// A 2-frame temporal block `[2, H, W, 3]` (reference videos).
    VideoBlock(Tensor),
}

pub struct Tokenizer {
    tk: tokenizers::Tokenizer,
}

/// The tokenizer (`assets/tokenizer.json`), compiled into the binary so an installed `h3` needs no source tree.
static EMBEDDED_TOKENIZER: &[u8] = include_bytes!("../assets/tokenizer.json");

impl Tokenizer {
    /// `$H3_TOKENIZER` if set, else the embedded tokenizer.
    pub fn load_default() -> Result<Self> {
        match std::env::var("H3_TOKENIZER").ok().filter(|p| !p.is_empty()) {
            Some(path) => Self::load(&path),
            None => Ok(Self { tk: tokenizers::Tokenizer::from_bytes(EMBEDDED_TOKENIZER).map_err(|e| anyhow::anyhow!("embedded tokenizer: {e}"))? }),
        }
    }

    pub fn load(path: &str) -> Result<Self> {
        Ok(Self { tk: tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer {path}: {e}"))? })
    }

    pub fn encode(&self, s: &str) -> Result<Vec<u32>> {
        if s.is_empty() {
            return Ok(vec![]);
        }
        Ok(self.tk.encode(s, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?.get_ids().to_vec())
    }

    fn text(&self, out: &mut Vec<Piece>, s: &str) -> Result<()> {
        out.extend(self.encode(s)?.into_iter().map(Piece::Token));
        Ok(())
    }

    fn vision(out: &mut Vec<Piece>, p: Piece) {
        out.push(Piece::Token(VISION_START));
        out.push(p);
        out.push(Piece::Token(VISION_END));
    }

    /// t2va / fl2va: `"<Picture i>: " <vision>` per keyframe image, then the prompt.
    pub fn presentation(&self, prompt: &str, images: &[Tensor]) -> Result<Vec<Piece>> {
        let mut out = vec![];
        for (i, img) in images.iter().enumerate() {
            self.text(&mut out, &format!("<Picture {}>: ", i + 1))?;
            Self::vision(&mut out, Piece::Image(img.clone()));
        }
        self.text(&mut out, prompt)?;
        if out.is_empty() {
            out.push(Piece::Token(PAD));
        }
        Ok(out)
    }

    /// ref2va: images, then videos (soundtrack label before each video), then standalone audio.
    pub fn ref_presentation(&self, prompt: &str, items: &[RefItem]) -> Result<Vec<Piece>> {
        let mut out = vec![];
        let (mut ni, mut na, mut nv) = (0, 0, 0);
        for it in items {
            match it {
                RefItem::Image(img) => {
                    ni += 1;
                    self.text(&mut out, &format!("<Picture {ni}>: "))?;
                    Self::vision(&mut out, Piece::Image(img.clone()));
                }
                RefItem::Audio => {
                    na += 1;
                    self.text(&mut out, &format!("<Audio {na}>: "))?;
                }
                RefItem::Video { frames, timestamps } => {
                    nv += 1;
                    let mut frames = frames.clone();
                    let mut ts = timestamps.clone();
                    if frames.dim(0)? % 2 == 1 {
                        let last = frames.narrow(0, frames.dim(0)? - 1, 1)?;
                        frames = Tensor::cat(&[&frames, &last], 0)?;
                        ts.push(*ts.last().unwrap_or(&0.0));
                    }
                    self.text(&mut out, &format!("<Video {nv}>: "))?;
                    for i in (0..frames.dim(0)?).step_by(2) {
                        self.text(&mut out, &format!("<{:.1} seconds>", (ts[i] + ts[i + 1]) / 2.0))?;
                        Self::vision(&mut out, Piece::VideoBlock(frames.narrow(0, i, 2)?));
                    }
                }
            }
        }
        self.text(&mut out, prompt)?;
        if out.is_empty() {
            out.push(Piece::Token(PAD));
        }
        Ok(out)
    }
}

pub enum RefItem {
    Image(Tensor),
    Audio,
    /// Frames `[T, H, W, 3]` sampled at 2 fps with their timestamps.
    Video { frames: Tensor, timestamps: Vec<f64> },
}

// ----------------------------------------------------------------------------------------------

struct Layer {
    in_norm: Tensor,
    post_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

struct VisionBlock {
    n1: (Tensor, Tensor),
    n2: (Tensor, Tensor),
    qkv: QLinear,
    proj: QLinear,
    fc1: QLinear,
    fc2: QLinear,
}

struct Merger {
    norm: (Tensor, Tensor),
    fc1: QLinear,
    fc2: QLinear,
    post_shuffle: bool,
}

struct Vision {
    patch: QLinear,
    pos_embed: Tensor,
    blocks: Vec<VisionBlock>,
    merger: Merger,
    deepstack: Vec<(usize, Merger)>,
}

pub struct TextEncoder {
    dev: Device,
    dtype: DType,
    file: Arc<SafeFile>,
    layers: Vec<Layer>,
    vision: Vision,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    theta: f64,
}

pub struct Encoded {
    /// `[L, 5120]` hidden states after the last layer.
    pub hidden: Tensor,
    pub tags: Vec<u32>,
}

fn dense(f: &Arc<SafeFile>, p: &str, dev: &Device, dt: DType) -> Result<QLinear> {
    QLinear::load(f, p, Placement::Gpu, dev, dt)
}

impl TextEncoder {
    pub fn load(path: &str, dev: &Device, mem: &MemoryConfig) -> Result<Self> {
        let f = SafeFile::open(path)?;
        let dtype = DType::BF16;
        let n_layers = (0..).take_while(|i| f.has(&format!("model.layers.{i}.input_layernorm.weight"))).count();
        if n_layers == 0 {
            bail!("{path} is not a Qwen3-VL text encoder");
        }
        let layer_bytes: usize = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
            .iter()
            .map(|p| f.bytes(&format!("model.layers.0.{p}.weight")).map(|b| b.len() * 9 / 8).unwrap_or(0))
            .sum();
        let resident = resolve_resident(mem.te_gpu_layers, n_layers, layer_bytes, (mem.reserve_gb * 0.5e9) as usize);
        log::info(&format!(
            "text encoder: Qwen3-VL {n_layers} layers, {resident} resident ({:.1} GB), rest from {:?}",
            (resident * layer_bytes) as f64 / 1e9,
            mem.te_offload.unwrap_or(mem.offload)
        ));
        let ld = |n: &str| f.load_as(n, dev, dtype);
        let mut layers = vec![];
        for i in 0..n_layers {
            let place = if i < resident { Placement::Gpu } else { mem.te_offload.unwrap_or(mem.offload) };
            let p = format!("model.layers.{i}");
            let q = |n: &str| QLinear::load(&f, &format!("{p}.{n}"), place, dev, dtype);
            layers.push(Layer {
                in_norm: ld(&format!("{p}.input_layernorm.weight"))?,
                post_norm: ld(&format!("{p}.post_attention_layernorm.weight"))?,
                q_norm: ld(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: ld(&format!("{p}.self_attn.k_norm.weight"))?,
                q: q("self_attn.q_proj")?,
                k: q("self_attn.k_proj")?,
                v: q("self_attn.v_proj")?,
                o: q("self_attn.o_proj")?,
                gate: q("mlp.gate_proj")?,
                up: q("mlp.up_proj")?,
                down: q("mlp.down_proj")?,
            });
        }
        let ln = |p: &str| -> Result<(Tensor, Tensor)> { Ok((ld(&format!("{p}.weight"))?, ld(&format!("{p}.bias"))?)) };
        let mut blocks = vec![];
        let depth = (0..).take_while(|i| f.has(&format!("visual.blocks.{i}.norm1.weight"))).count();
        for i in 0..depth {
            let p = format!("visual.blocks.{i}");
            blocks.push(VisionBlock {
                n1: ln(&format!("{p}.norm1"))?,
                n2: ln(&format!("{p}.norm2"))?,
                qkv: dense(&f, &format!("{p}.attn.qkv"), dev, dtype)?,
                proj: dense(&f, &format!("{p}.attn.proj"), dev, dtype)?,
                fc1: dense(&f, &format!("{p}.mlp.linear_fc1"), dev, dtype)?,
                fc2: dense(&f, &format!("{p}.mlp.linear_fc2"), dev, dtype)?,
            });
        }
        let merger = |p: &str, post: bool| -> Result<Merger> {
            Ok(Merger {
                norm: ln(&format!("{p}.norm"))?,
                fc1: dense(&f, &format!("{p}.linear_fc1"), dev, dtype)?,
                fc2: dense(&f, &format!("{p}.linear_fc2"), dev, dtype)?,
                post_shuffle: post,
            })
        };
        let ds_idx = [8usize, 16, 24];
        let mut deepstack = vec![];
        for (k, idx) in ds_idx.iter().enumerate() {
            let p = format!("visual.deepstack_merger_list.{k}");
            if f.has(&format!("{p}.norm.weight")) {
                deepstack.push((*idx, merger(&p, true)?));
            }
        }
        let pw = f.load_as("visual.patch_embed.proj.weight", dev, dtype)?;
        let pw = pw.reshape((pw.dim(0)?, pw.elem_count() / pw.dim(0)?))?;
        let mut patch = dense(&f, "visual.patch_embed.proj", dev, dtype)?;
        patch.weight = crate::weights::Held::from_tensor(pw);
        patch.in_features = 1536;
        let vision = Vision {
            patch,
            pos_embed: ld("visual.pos_embed.weight")?,
            blocks,
            merger: merger("visual.merger", false)?,
            deepstack,
        };
        let hidden = f.shape("model.layers.0.input_layernorm.weight")?[0];
        let head_dim = f.shape("model.layers.0.self_attn.q_norm.weight")?[0];
        let q_out = f.shape("model.layers.0.self_attn.q_proj.weight")?[0];
        let k_out = f.shape("model.layers.0.self_attn.k_proj.weight")?[0];
        Ok(Self {
            dev: dev.clone(),
            dtype,
            file: f,
            layers,
            vision,
            hidden,
            heads: q_out / head_dim,
            kv_heads: k_out / head_dim,
            head_dim,
            eps: 1e-6,
            theta: 5_000_000.0,
        })
    }

    /// int8 embedding rows gathered straight from the mmapped table.
    fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        let q = self.file.bytes("model.embed_tokens.weight")?;
        let s = self.file.bytes("model.embed_tokens.weight_scale")?;
        let h = self.hidden;
        let mut out = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            let id = id as usize;
            let scale = f32::from_le_bytes(s[id * 4..id * 4 + 4].try_into()?);
            out.extend(q[id * h..(id + 1) * h].iter().map(|b| (*b as i8) as f32 * scale));
        }
        Ok(Tensor::from_vec(out, (ids.len(), h), &self.dev)?)
    }

    pub fn encode(&self, pieces: &[Piece]) -> Result<Encoded> {
        // embeds: text runs from the table, vision blocks from the vision tower
        let mut parts: Vec<Tensor> = vec![];
        let mut run: Vec<u32> = vec![];
        let mut images: Vec<(usize, usize, (usize, usize, usize), Vec<Tensor>)> = vec![]; // (index, size, grid, deepstack)
        let mut pos = 0usize;
        for p in pieces {
            match p {
                Piece::Token(t) => {
                    run.push(*t);
                    pos += 1;
                }
                Piece::Image(img) | Piece::VideoBlock(img) => {
                    if !run.is_empty() {
                        parts.push(self.embed(&run)?);
                        run.clear();
                    }
                    let video = matches!(p, Piece::VideoBlock(_));
                    let (patches, grid) = qwen_patches(img, video)?;
                    let (merged, ds) = self.vision_forward(&patches.to_device(&self.dev)?, grid)?;
                    let n = merged.dim(0)?;
                    images.push((pos, n, grid, ds));
                    parts.push(merged.to_dtype(DType::F32)?);
                    pos += n;
                }
            }
        }
        if !run.is_empty() {
            parts.push(self.embed(&run)?);
        }
        let mut x = Tensor::cat(&parts, 0)?.to_dtype(self.dtype)?;
        let seq = x.dim(0)?;

        let mut tags = vec![1u32; seq];
        for (idx, size, _, _) in &images {
            for t in &mut tags[idx.saturating_sub(1)..(idx + size + 1).min(seq)] {
                *t = 0;
            }
        }

        let rope = self.rope_table(seq, &images)?;
        // deepstack features scattered onto the image positions
        let ds_count = images.first().map(|i| i.3.len()).unwrap_or(0);
        let mut ds_full = vec![];
        for k in 0..ds_count {
            let mut rows = vec![];
            let mut cur = 0;
            for (idx, size, _, ds) in &images {
                if *idx > cur {
                    rows.push(Tensor::zeros((idx - cur, self.hidden), self.dtype, &self.dev)?);
                }
                rows.push(ds[k].to_dtype(self.dtype)?);
                cur = idx + size;
            }
            if seq > cur {
                rows.push(Tensor::zeros((seq - cur, self.hidden), self.dtype, &self.dev)?);
            }
            ds_full.push(Tensor::cat(&rows, 0)?);
        }

        let (h, kvh, d) = (self.heads, self.kv_heads, self.head_dim);
        let prefetch = |l: &Layer| {
            for q in [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down] {
                q.prefetch();
            }
        };
        if let Some(l) = self.layers.first() {
            prefetch(l);
        }
        for (i, l) in self.layers.iter().enumerate() {
            if let Some(next) = self.layers.get(i + 1) {
                prefetch(next);
            }
            let n = ops::rms_norm(&x, Some(&l.in_norm), self.eps)?;
            let q = kernels::qk_norm_rope(&l.q.forward(&n)?, 0, h, d, Some((Some(&l.q_norm), self.eps)), Some(&rope), d, self.dtype)?;
            let k = kernels::qk_norm_rope(&l.k.forward(&n)?, 0, kvh, d, Some((Some(&l.k_norm), self.eps)), Some(&rope), d, self.dtype)?;
            let v = l.v.forward(&n)?.reshape((seq, kvh, d))?;
            let a = ops::attention(&q.unsqueeze(0)?, &k.unsqueeze(0)?, &v.unsqueeze(0)?, true)?.reshape((seq, h * d))?;
            x = (&x + l.o.forward(&a)?)?;
            let n = ops::rms_norm(&x, Some(&l.post_norm), self.eps)?;
            let g = ops::silu(&l.gate.forward(&n)?)?;
            let u = l.up.forward(&n)?;
            x = (&x + l.down.forward(&(g * u)?)?)?;
            if i < ds_full.len() {
                x = (&x + &ds_full[i])?;
            }
        }
        Ok(Encoded { hidden: x, tags })
    }

    /// `[2, S, 64]` cos/sin for the LM: plain 1D positions for text, interleaved MRoPE with images.
    fn rope_table(&self, seq: usize, images: &[(usize, usize, (usize, usize, usize), Vec<Tensor>)]) -> Result<Tensor> {
        let half = self.head_dim / 2;
        let inv: Vec<f32> = (0..half).map(|j| 1.0 / (self.theta as f32).powf(2.0 * j as f32 / self.head_dim as f32)).collect();
        let pos = mrope_positions(seq, images.iter().map(|i| (i.0, i.1, i.2)).collect());
        let mrope_dims = [24usize, 20, 20];
        let mut ang = Vec::with_capacity(seq * half);
        for s in 0..seq {
            for (j, f) in inv.iter().enumerate() {
                let axis = match &pos {
                    None => 0,
                    Some(_) if j % 3 == 1 && j < mrope_dims[1] * 3 => 1,
                    Some(_) if j % 3 == 2 && j < mrope_dims[2] * 3 => 2,
                    _ => 0,
                };
                let p = match &pos {
                    None => s as f32,
                    Some(p) => p[axis][s],
                };
                ang.push(p * f);
            }
        }
        kernels::rope_table(&Tensor::from_vec(ang, (seq, half), &self.dev)?).map_err(Into::into)
    }

    fn vision_forward(&self, patches: &Tensor, grid: (usize, usize, usize)) -> Result<(Tensor, Vec<Tensor>)> {
        let v = &self.vision;
        let (gt, gh, gw) = grid;
        let x = v.patch.forward(&patches.to_dtype(self.dtype)?)?;
        let x = (x + self.vision_pos_embed(grid)?)?;
        let n = x.dim(0)?;
        // rotary: (row, col) per patch in merge-block order, 18 freqs each, theta 1e4
        let hd = 72usize;
        let rd = hd / 2 / 2; // 18
        let inv: Vec<f32> = (0..rd).map(|j| 1.0 / 10000f32.powf(2.0 * j as f32 / (hd / 2) as f32)).collect();
        let mut ang = Vec::with_capacity(n * rd * 2);
        for _t in 0..gt {
            for bh in 0..gh / 2 {
                for bw in 0..gw / 2 {
                    for ih in 0..2 {
                        for iw in 0..2 {
                            let (r, c) = ((bh * 2 + ih) as f32, (bw * 2 + iw) as f32);
                            ang.extend(inv.iter().map(|f| r * f));
                            ang.extend(inv.iter().map(|f| c * f));
                        }
                    }
                }
            }
        }
        let rope = kernels::rope_table(&Tensor::from_vec(ang, (n, rd * 2), &self.dev)?)?;
        let heads = 16usize;
        let per_frame = gh * gw;
        let mut x = x;
        let mut ds = vec![];
        for (li, b) in v.blocks.iter().enumerate() {
            let hn = ops::layer_norm(&x, Some(&b.n1.0), Some(&b.n1.1), 1e-6)?;
            let qkv = b.qkv.forward(&hn)?;
            let q = kernels::qk_norm_rope(&qkv, 0, heads, hd, None, Some(&rope), hd, self.dtype)?;
            let k = kernels::qk_norm_rope(&qkv, heads * hd, heads, hd, None, Some(&rope), hd, self.dtype)?;
            let vv = qkv.narrow(1, 2 * heads * hd, heads * hd)?.contiguous()?.reshape((n, heads, hd))?;
            let mut outs = vec![];
            for t in 0..gt {
                let sl = |z: &Tensor| z.narrow(0, t * per_frame, per_frame).and_then(|z| z.unsqueeze(0));
                outs.push(ops::attention(&sl(&q)?, &sl(&k)?, &sl(&vv)?, false)?.squeeze(0)?);
            }
            let a = Tensor::cat(&outs, 0)?.reshape((n, heads * hd))?;
            x = (&x + b.proj.forward(&a)?)?;
            let hn = ops::layer_norm(&x, Some(&b.n2.0), Some(&b.n2.1), 1e-6)?;
            x = (&x + b.fc2.forward(&ops::gelu_tanh(&b.fc1.forward(&hn)?)?)?)?;
            if let Some((_, m)) = v.deepstack.iter().find(|(i, _)| *i == li) {
                ds.push(merger_forward(m, &x)?);
            }
        }
        Ok((merger_forward(&v.merger, &x)?, ds))
    }

    /// Bilinear interpolation of the 48x48 learned position grid, in merge-block order.
    fn vision_pos_embed(&self, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (gt, h, w) = grid;
        let side = 48usize;
        let lin = |n: usize| -> Vec<f32> {
            if n == 1 {
                return vec![0.0];
            }
            (0..n).map(|i| i as f32 * (side - 1) as f32 / (n - 1) as f32).collect()
        };
        let (hs, ws) = (lin(h), lin(w));
        let mut idx = [vec![], vec![], vec![], vec![]];
        let mut wts = [vec![], vec![], vec![], vec![]];
        for &hy in &hs {
            for &wx in &ws {
                let (hf, wf) = (hy as usize, wx as usize);
                let (hc, wc) = ((hf + 1).min(side - 1), (wf + 1).min(side - 1));
                let (dh, dw) = (hy - hf as f32, wx - wf as f32);
                let cells = [(hf, wf, (1.0 - dh) * (1.0 - dw)), (hf, wc, (1.0 - dh) * dw), (hc, wf, dh * (1.0 - dw)), (hc, wc, dh * dw)];
                for (k, (a, b, wt)) in cells.iter().enumerate() {
                    idx[k].push((a * side + b) as u32);
                    wts[k].push(*wt);
                }
            }
        }
        let table = &self.vision.pos_embed;
        let mut acc: Option<Tensor> = None;
        for k in 0..4 {
            let n = idx[k].len();
            let rows = table.index_select(&Tensor::from_vec(idx[k].clone(), n, &self.dev)?, 0)?;
            // reference keeps the weights in the table dtype
            let wt = Tensor::from_vec(wts[k].clone(), (n, 1), &self.dev)?.to_dtype(table.dtype())?;
            let r = rows.broadcast_mul(&wt)?;
            acc = Some(match acc {
                None => r,
                Some(a) => (a + r)?,
            });
        }
        let pe = acc.unwrap(); // [h*w, C] row-major
        let c = pe.dim(1)?;
        let pe = pe.reshape((h / 2, 2, w / 2, 2, c))?.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((h * w, c))?;
        let pe = if gt > 1 { pe.repeat((gt, 1))? } else { pe };
        Ok(pe.to_dtype(self.dtype)?)
    }
}

fn merger_forward(m: &Merger, x: &Tensor) -> Result<Tensor> {
    let n = x.dim(0)?;
    let c = x.dim(1)?;
    let h = if m.post_shuffle {
        ops::layer_norm(&x.reshape((n / 4, c * 4))?, Some(&m.norm.0), Some(&m.norm.1), 1e-6)?
    } else {
        ops::layer_norm(x, Some(&m.norm.0), Some(&m.norm.1), 1e-6)?.reshape((n / 4, c * 4))?
    };
    m.fc2.forward(&ops::gelu_erf(&m.fc1.forward(&h)?)?)
}

/// `(3, S)` T/H/W MRoPE positions (`qwen2vl_mrope_position_ids`), `None` without images.
fn mrope_positions(seq: usize, images: Vec<(usize, usize, (usize, usize, usize))>) -> Option<[Vec<f32>; 3]> {
    if images.is_empty() {
        return None;
    }
    let mut p = [vec![0f32; seq], vec![0f32; seq], vec![0f32; seq]];
    let mut offset = 0i64;
    for (n, (start, size, (gt, gh, gw))) in images.into_iter().enumerate() {
        let (start_i, end) = (start as i64, start + size);
        if n == 0 {
            for ax in p.iter_mut() {
                for (i, v) in ax.iter_mut().enumerate().take(start) {
                    *v = i as f32;
                }
            }
        }
        let len_max = (gt.max(gh).max(gw) / 2) as i64;
        let start_next = len_max + start_i;
        for ax in p.iter_mut() {
            for (k, v) in ax[end..].iter_mut().enumerate() {
                *v = (start_next + offset + k as i64) as f32;
            }
        }
        let (mh, mw) = (gh / 2, gw / 2);
        for k in 0..size {
            p[0][start + k] = (start_i + offset) as f32;
            p[1][start + k] = (start_i + offset + (k / mw.max(1)) as i64 % mh.max(1) as i64) as f32;
            p[2][start + k] = (start_i + offset + (k % mw.max(1)) as i64) as f32;
        }
        offset += len_max - size as i64;
    }
    Some(p)
}

/// Qwen3-VL patch extraction: resize to multiples of 32, normalize to [-1, 1], flatten
/// `[gh*gw, 3*2*16*16]` in merge-block order. `img` is `[H, W, 3]` (or `[2, H, W, 3]` for a video
/// block, which fills the temporal patch instead of repeating one frame).
pub fn qwen_patches(img: &Tensor, video_block: bool) -> Result<(Tensor, (usize, usize, usize))> {
    let frames = if video_block { img.clone() } else { img.unsqueeze(0)? };
    let frames = frames.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let (_, height, width, _) = frames.dims4()?;
    let factor = 32usize;
    let (min_px, max_px) = (3136f64, 12845056f64);
    let round = |v: f64| ((v / factor as f64).round() as usize * factor).max(factor);
    let (mut hb, mut wb) = (round(height as f64), round(width as f64));
    if (hb * wb) as f64 > max_px {
        let beta = ((height * width) as f64 / max_px).sqrt();
        hb = (((height as f64 / beta) / factor as f64).floor() as usize * factor).max(factor);
        wb = (((width as f64 / beta) / factor as f64).floor() as usize * factor).max(factor);
    } else if ((hb * wb) as f64) < min_px {
        let beta = (min_px / (height * width) as f64).sqrt();
        hb = ((height as f64 * beta) / factor as f64).ceil() as usize * factor;
        wb = ((width as f64 * beta) / factor as f64).ceil() as usize * factor;
    }
    let chw = frames.permute((0, 3, 1, 2))?.contiguous()?; // [T, 3, H, W]
    let chw = if (hb, wb) != (height, width) { crate::media::resize_bilinear(&chw, hb, wb)? } else { chw };
    let x = ((chw - 0.5)? / 0.5)?;
    let x = if video_block { x } else { Tensor::cat(&[&x, &x], 0)? }; // [2, 3, H, W]
    let (gh, gw) = (hb / 16, wb / 16);
    let x = x.reshape(&[1, 2, 3, gh / 2, 2, 16, gw / 2, 2, 16][..])?;
    let x = x.permute(vec![0, 3, 6, 4, 7, 2, 1, 5, 8])?.contiguous()?;
    Ok((x.reshape((gh * gw, 3 * 2 * 16 * 16))?, (1, gh, gw)))
}

pub fn tokenizer_path() -> Result<String> {
    let candidates = [
        std::env::var("H3_TOKENIZER").unwrap_or_default(),
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/tokenizer.json").to_string(),
        "assets/tokenizer.json".to_string(),
    ];
    candidates.into_iter().find(|p| !p.is_empty() && std::path::Path::new(p).exists()).context("assets/tokenizer.json not found (set H3_TOKENIZER)")
}
