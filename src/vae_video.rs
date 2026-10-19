//! MiniMax H3 video VAE (port of `comfy/ldm/minimax/vae.py`): 3D causal CNN encoder (single-frame
//! path, used for image keyframes) and ViT3D decoder with the reference's 256 px spatial tiling and
//! 17-frame temporal chunking. All tiles of a temporal chunk run through the decoder layer by
//! layer, so each decoder block is fetched once per chunk (cheap streaming on small GPUs).

use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, D};

use crate::config::MemoryConfig;
use crate::dit::log;
use crate::kernels;
use crate::ops;
use crate::weights::{Placement, QLinear, SafeFile};

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const RATIO: usize = 16;
const RATIO_T: usize = 4;
const CLIP_LENGTH: usize = 17;
const TOKEN_DROP: usize = 3;
const TILE: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;
const TOKENS_CHUNK: usize = 5; // ceil(17 / 4)
const TOKEN_OVERLAP: usize = 2; // (-3) % 5
const FRAME_PRE_PADDING: usize = 3; // (-17) % 4
const FRAME_OVERLAP: usize = 5; // 2 * 4 - 3

struct DecBlock {
    norm1: Tensor,
    norm2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    qkv: QLinear,
    out: QLinear,
    w1: QLinear,
    w2: QLinear,
}

struct Decoder {
    x_embedder: QLinear,
    register_tokens: Tensor,
    blocks: Vec<DecBlock>,
    norm_out: (Tensor, Tensor),
    proj_out: QLinear,
    post_quant: QLinear,
}

struct Conv {
    w: Tensor, // [out, in, kt, k, k]
    b: Tensor,
    k: usize,
    kt: usize,
}

enum ResBlock {
    Full { n1: (Tensor, Tensor), c1: Conv, n2: (Tensor, Tensor), c2: Conv, short: Option<Conv> },
    Shortcut(Option<Conv>),
}

struct Encoder {
    conv_in: Conv,
    levels: Vec<(Vec<ResBlock>, Option<(Conv, usize)>)>,
    norm_out: (Tensor, Tensor),
    conv_out: Conv,
    quant_conv: Conv,
}

pub struct VideoVae {
    dev: Device,
    dtype: DType,
    tile_batch: usize,
    dec: Option<Decoder>,
    enc: Option<Encoder>,
    mean: Tensor,
    std: Tensor,
}

impl VideoVae {
    pub fn load(path: &str, dev: &Device, mem: &MemoryConfig) -> Result<Self> {
        let f = SafeFile::open(path)?;
        let dtype = mem.vae_dtype()?;
        let place = if mem.vae_resident { Placement::Gpu } else { mem.offload };
        let ld = |n: &str| f.load_as(n, dev, dtype);
        let n_layers = (0..).take_while(|i| f.has(&format!("decoder.transformer_blocks.{i}.scale1"))).count();
        let dec = if n_layers > 0 {
            let mut blocks = vec![];
            for i in 0..n_layers {
                let p = format!("decoder.transformer_blocks.{i}");
                let q = |n: &str| QLinear::load(&f, &format!("{p}.{n}"), place, dev, dtype);
                blocks.push(DecBlock {
                    norm1: ld(&format!("{p}.norm1.weight"))?,
                    norm2: ld(&format!("{p}.norm2.weight"))?,
                    scale1: ld(&format!("{p}.scale1"))?,
                    scale2: ld(&format!("{p}.scale2"))?,
                    qkv: q("attn.to_qkv")?,
                    out: q("attn.to_out")?,
                    w1: q("ff.w1")?,
                    w2: q("ff.w2")?,
                });
            }
            let mut pq = QLinear::load(&f, "post_quant_conv", Placement::Gpu, dev, DType::F32)?;
            let w = f.load_as("post_quant_conv.weight", dev, DType::F32)?;
            pq.weight = crate::weights::Held::from_tensor(w.reshape((w.dim(0)?, w.dim(1)?))?);
            Some(Decoder {
                x_embedder: QLinear::load(&f, "decoder.x_embedder", Placement::Gpu, dev, dtype)?,
                register_tokens: ld("decoder.register_tokens")?,
                blocks,
                norm_out: (ld("decoder.norm_out.weight")?, ld("decoder.norm_out.bias")?),
                proj_out: QLinear::load(&f, "decoder.proj_out", Placement::Gpu, dev, dtype)?,
                post_quant: pq,
            })
        } else {
            None
        };
        let enc = if f.has("encoder.conv_in.weight") { Some(load_encoder(&f, dev)?) } else { None };
        let stats = |n: &str| -> Result<Tensor> { Ok(f.load_as(n, dev, DType::F32)?.reshape((1, 24, 1, 1, 1))?) };
        log::info(&format!(
            "video VAE: decoder {n_layers} layers ({:?}, {}), encoder {}",
            dtype,
            if mem.vae_resident { "resident" } else { "streamed" },
            enc.is_some()
        ));
        Ok(Self { dev: dev.clone(), dtype, tile_batch: mem.vae_tile_batch.max(1), dec, enc, mean: stats("latents_mean")?, std: stats("latents_std")? })
    }

    // ------------------------------------------------------------------------------------------
    // decode

    /// Normalized latents `[1, 24, T, h, w]` -> pixels `[3, F, H, W]` f32 in [0, 1] on the CPU.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let dec = self.dec.as_ref().ok_or_else(|| anyhow::anyhow!("this VAE file has no decoder"))?;
        let z = z.to_device(&self.dev)?.to_dtype(DType::F32)?;
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let (_, _, t, lh, lw) = z.dims5()?;
        let (h, w) = (lh * RATIO, lw * RATIO);
        if t == 1 {
            let zz = z.repeat((1, 1, TOKENS_CHUNK, 1, 1))?;
            let px = self.tiled_decode(dec, &zz)?;
            let px = slice_frames(&px, FRAME_PRE_PADDING, FRAME_PRE_PADDING + 1);
            return finalize(px);
        }
        let (pad_tokens, num_chunks) = decode_temporal_chunks(t);
        let z = if pad_tokens > 0 { Tensor::cat(&[z.clone(), z.narrow(2, t - 1, 1)?.repeat((1, 1, pad_tokens, 1, 1))?], 2)? } else { z };
        let zl = t + pad_tokens;
        let total = decode_temporal_frame_plan(zl, num_chunks, pad_tokens);
        let mut out = CpuVid::zeros(3, total, h, w);
        let mut write_pos = 0usize;
        let mut dec_overlap: Option<CpuVid> = None;
        let chunk_dec = TOKENS_CHUNK * RATIO_T;
        for i in 0..num_chunks {
            let t0 = i * TOKENS_CHUNK;
            let t1 = (t0 + TOKENS_CHUNK + TOKEN_OVERLAP).min(zl);
            let clip = self.tiled_decode(dec, &z.narrow(2, t0, t1 - t0)?)?;
            log::info(&format!("VAE decode chunk {}/{num_chunks}", i + 1));
            crate::progress::count(crate::progress::Stage::Decoding, i + 1, num_chunks);
            crate::progress::check()?;
            for j in 0..2 {
                let f0 = j * chunk_dec;
                let f1 = (f0 + chunk_dec).min(clip.f);
                if f1 <= f0 + FRAME_PRE_PADDING {
                    if j == 1 {
                        dec_overlap = None;
                    }
                    continue;
                }
                let mut part = slice_frames(&clip, f0 + FRAME_PRE_PADDING, f1);
                if j == 0 {
                    if let Some(prev) = dec_overlap.take() {
                        part = blend_frames(&prev, &part, FRAME_OVERLAP);
                    }
                    write_part(&mut out, &mut write_pos, part);
                } else {
                    dec_overlap = Some(part);
                }
            }
            if i == num_chunks - 1 {
                if let Some(prev) = dec_overlap.take() {
                    write_part(&mut out, &mut write_pos, prev);
                }
            }
        }
        out.into_tensor()
    }

    /// Spatially tiled decode of one temporal clip: `[1, 24, Tc, lh, lw]` (denormalized) ->
    /// raw decoder pixels `[3, Tc*4, H, W]`.
    fn tiled_decode(&self, dec: &Decoder, z: &Tensor) -> Result<CpuVid> {
        let (_, _, _, lh, lw) = z.dims5()?;
        let (height, width) = (lh * RATIO, lw * RATIO);
        let (yi, yl, yo) = split_tiles(height);
        let (xi, xl, xo) = split_tiles(width);
        let mut lat = vec![];
        for (&ip, &il) in yi.iter().zip(&yl) {
            for (&jp, &jl) in xi.iter().zip(&xl) {
                lat.push(z.narrow(3, ip / RATIO, il / RATIO)?.narrow(4, jp / RATIO, jl / RATIO)?);
            }
        }
        let tiles = self.decode_tiles(dec, &Tensor::cat(&lat, 0)?)?;
        let mut canvas: Option<CpuVid> = None;
        let mut strip: Option<CpuVid> = None;
        let mut out_y = 0;
        let mut k = 0;
        for i in 0..yi.len() {
            let mut new_strip: Option<CpuVid> = None;
            let mut left_tail: Option<CpuVid> = None;
            let mut out_x = 0;
            let mut tile_h = 0;
            for j in 0..xi.len() {
                let mut tile = tiles[k].clone();
                k += 1;
                if i > 0 {
                    let s = strip.as_ref().unwrap().crop(0, usize::MAX, xi[j], xi[j] + xl[j]);
                    tile = blend_h(&s, &tile, yo[i - 1]);
                }
                if j > 0 {
                    tile = blend_w(left_tail.as_ref().unwrap(), &tile, xo[j - 1]);
                }
                let last_x = j == xi.len() - 1;
                left_tail = if !last_x { Some(tile.crop(0, usize::MAX, tile.w - xo[j], tile.w)) } else { None };
                if !last_x {
                    tile = tile.crop(0, usize::MAX, 0, tile.w - xo[j]);
                }
                let c = canvas.get_or_insert_with(|| CpuVid::zeros(tile.c, tile.f, height, width));
                if i < yi.len() - 1 {
                    let ns = new_strip.get_or_insert_with(|| CpuVid::zeros(tile.c, tile.f, yo[i], width));
                    ns.paste(&tile.crop(tile.h - yo[i], tile.h, 0, usize::MAX), 0, out_x);
                    tile = tile.crop(0, tile.h - yo[i], 0, usize::MAX);
                }
                c.paste(&tile, out_y, out_x);
                tile_h = tile.h;
                out_x += tile.w;
            }
            strip = new_strip;
            out_y += tile_h;
        }
        Ok(canvas.unwrap())
    }

    /// Run the ViT3D decoder over a batch of same-size latent tiles `[N, 24, T, h, w]`.
    fn decode_tiles(&self, dec: &Decoder, z: &Tensor) -> Result<Vec<CpuVid>> {
        let (n, c, t, h, w) = z.dims5()?;
        let dt = self.dtype;
        let tokens = z.reshape((n, c, t * h * w))?.transpose(1, 2)?.contiguous()?; // [N, P, 24]
        let tokens = dec.post_quant.forward(&tokens)?.to_dtype(dt)?;
        let x = dec.x_embedder.forward(&tokens)?; // [N, P, 2048]
        let p = t * h * w;
        let dim = x.dim(2)?;
        let reg = dec.register_tokens.to_dtype(dt)?.expand((n, 4, dim))?;
        let zero = Tensor::zeros((n, 1, dim), dt, &self.dev)?;
        let x = Tensor::cat(&[&x, &reg, &zero], 1)?;
        let s = p + 5;
        let rope = decoder_rope(t, h, w, &self.dev, dt)?; // [2, S, 24]

        // layer-major over the whole tile batch, compute in sub-batches
        let sb = self.tile_batch;
        let mut xs: Vec<Tensor> = (0..n.div_ceil(sb)).map(|k| x.narrow(0, k * sb, sb.min(n - k * sb))).collect::<candle_core::Result<_>>()?;
        let tables: Vec<Tensor> = xs.iter().map(|xb| rope.repeat((1, xb.dim(0).unwrap(), 1))).collect::<candle_core::Result<_>>()?;
        for blk in &dec.blocks {
            let wqkv = blk.qkv.dequant(&self.dev, dt)?;
            let wout = blk.out.dequant(&self.dev, dt)?;
            let w1 = blk.w1.dequant(&self.dev, dt)?;
            let w2 = blk.w2.dequant(&self.dev, dt)?;
            for (xb, tab) in xs.iter_mut().zip(&tables) {
                let b = xb.dim(0)?;
                let flat = xb.reshape((b * s, dim))?;
                let qkv = blk.qkv.apply(&ops::rms_norm(&flat, Some(&blk.norm1), 1e-5)?, &wqkv, usize::MAX)?;
                let (heads, hd) = (32usize, 64usize);
                let norm = Some((None, 1e-5f32));
                let q = kernels::qk_norm_rope_strided(&qkv, 0, 3 * hd, heads, hd, norm, Some(tab), 48, dt)?;
                let k = kernels::qk_norm_rope_strided(&qkv, hd, 3 * hd, heads, hd, norm, Some(tab), 48, dt)?;
                let v = qkv.reshape((b * s, heads, 3 * hd))?.narrow(2, 2 * hd, hd)?.contiguous()?;
                let r = |t: Tensor| t.reshape((b, s, heads, hd));
                let o = ops::attention(&r(q)?, &r(k)?, &r(v)?, false)?.reshape((b * s, dim))?;
                let o = nan_to_num(&o)?;
                let o = blk.out.apply(&o, &wout, usize::MAX)?;
                let flat = (flat + o.broadcast_mul(&blk.scale1)?)?;
                let hdn = blk.w1.apply(&ops::rms_norm(&flat, Some(&blk.norm2), 1e-5)?, &w1, usize::MAX)?;
                let ff = blk.w2.apply(&ops::swiglu(&hdn)?, &w2, usize::MAX)?;
                *xb = (flat + ff.broadcast_mul(&blk.scale2)?)?.reshape((b, s, dim))?;
            }
        }
        let mut out = vec![];
        for xb in xs {
            let b = xb.dim(0)?;
            let y = ops::layer_norm(&xb, Some(&dec.norm_out.0), Some(&dec.norm_out.1), 1e-5)?;
            let y = dec.proj_out.forward(&y.narrow(1, 0, p)?.contiguous()?)?; // [b, P, 3*4*16*16]
            let y = y.reshape(&[b, t, h, w, 3, RATIO_T, RATIO, RATIO][..])?;
            let y = y.permute(vec![0, 4, 1, 5, 2, 6, 3, 7])?.contiguous()?;
            let y = y.reshape((b, 3, t * RATIO_T, h * RATIO, w * RATIO))?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
            for i in 0..b {
                out.push(CpuVid::from_tensor(&y.get(i)?)?);
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------------------------------
    // encode

    /// Encode one image `[H, W, 3]` in [0, 1] -> normalized latent `[1, 24, 1, H/16, W/16]`.
    pub fn encode_image(&self, img: &Tensor) -> Result<Tensor> {
        self.encode_frames(&img.unsqueeze(0)?)
    }

    /// Encode frames `[T, H, W, 3]` in [0, 1] (T = 1 or 17k+5) -> normalized latent
    /// `[1, 24, T_lat, H/16, W/16]` (17-frame causal clips, last 3 tokens dropped).
    pub fn encode_frames(&self, frames: &Tensor) -> Result<Tensor> {
        let enc = self.enc.as_ref().ok_or_else(|| anyhow::anyhow!("this VAE file has no encoder"))?;
        let x = frames.to_device(&self.dev)?.to_dtype(DType::F32)?.permute((3, 0, 1, 2))?.unsqueeze(0)?; // [1,3,T,H,W]
        let mean = Tensor::new(&IMAGENET_MEAN, &self.dev)?.reshape((1, 3, 1, 1, 1))?;
        let std = Tensor::new(&IMAGENET_STD, &self.dev)?.reshape((1, 3, 1, 1, 1))?;
        let x = x.broadcast_sub(&mean)?.broadcast_div(&std)?;
        let (_, _, t, height, width) = x.dims5()?;
        if height % RATIO != 0 || width % RATIO != 0 {
            bail!("frame size must be a multiple of 16");
        }
        let moments = if t == 1 {
            let m = self.tiled_encode(enc, &x)?;
            let n = m.dim(2)?;
            m.narrow(2, n - 1, 1)?
        } else {
            let mut zs = vec![];
            for i in 0..t.div_ceil(CLIP_LENGTH) {
                let n = CLIP_LENGTH.min(t - i * CLIP_LENGTH);
                let mut clip = x.narrow(2, i * CLIP_LENGTH, n)?;
                if n < CLIP_LENGTH {
                    let last = clip.narrow(2, n - 1, 1)?.repeat((1, 1, CLIP_LENGTH - n, 1, 1))?;
                    clip = Tensor::cat(&[&clip, &last], 2)?;
                }
                zs.push(self.tiled_encode(enc, &clip)?);
                log::info(&format!("VAE encode clip {}/{}", i + 1, t.div_ceil(CLIP_LENGTH)));
            }
            let z = Tensor::cat(&zs, 2)?;
            let n = z.dim(2)?;
            z.narrow(2, 0, n - TOKEN_DROP)?
        };
        let mean_z = moments.narrow(1, 0, 24)?;
        Ok(mean_z.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?)
    }

    fn tiled_encode(&self, enc: &Encoder, x: &Tensor) -> Result<Tensor> {
        let (_, _, _, height, width) = x.dims5()?;
        let (yi, yl, yo) = split_tiles(height);
        let (xi, xl, xo) = split_tiles(width);
        let mut rows: Vec<Vec<Tensor>> = vec![];
        for (&ip, &il) in yi.iter().zip(&yl) {
            let mut row = vec![];
            for (&jp, &jl) in xi.iter().zip(&xl) {
                row.push(encode_tile(enc, &x.narrow(3, ip, il)?.narrow(4, jp, jl)?)?);
            }
            rows.push(row);
        }
        let ly: Vec<usize> = yo.iter().map(|o| o / RATIO).collect();
        let lx: Vec<usize> = xo.iter().map(|o| o / RATIO).collect();
        let mut out_rows = vec![];
        for (i, row) in rows.iter().enumerate() {
            let mut res = vec![];
            for (j, tile) in row.iter().enumerate() {
                crate::progress::check()?;
                let mut t = tile.clone();
                if i > 0 {
                    t = blend_t(&rows[i - 1][j], &t, ly[i - 1], 3)?;
                }
                if j > 0 {
                    t = blend_t(&row[j - 1], &t, lx[j - 1], 4)?;
                }
                if i < rows.len() - 1 {
                    t = t.narrow(3, 0, t.dim(3)? - ly[i])?;
                }
                if j < row.len() - 1 {
                    t = t.narrow(4, 0, t.dim(4)? - lx[j])?;
                }
                res.push(t);
            }
            out_rows.push(Tensor::cat(&res, 4)?);
        }
        Ok(Tensor::cat(&out_rows, 3)?)
    }
}


fn nan_to_num(x: &Tensor) -> Result<Tensor> {
    let max = match x.dtype() {
        DType::F16 => 65504.0,
        DType::BF16 => 3.38e38,
        _ => 3.4e38,
    };
    let clean = x.eq(x)?.where_cond(x, &x.zeros_like()?)?;
    Ok(clean.clamp(-max, max)?)
}

/// ViT3D rope: coords in [-1, 1] per axis, 8 freqs per axis (base 100), angle * 2pi; suffix
/// tokens at 0. Returns `[2, S, 24]` cos/sin rounded to the compute dtype.
fn decoder_rope(t: usize, h: usize, w: usize, dev: &Device, dt: DType) -> Result<Tensor> {
    let inv: Vec<f32> = (0..8).map(|i| 1.0 / 100f32.powf(i as f32 * 0.125)).collect();
    let coord = |i: usize, n: usize| 2.0 * ((i as f32 + 0.5) / n as f32) - 1.0;
    let mut ang = Vec::with_capacity((t * h * w + 5) * 24);
    for a in 0..t {
        for b in 0..h {
            for c in 0..w {
                for (v, _) in [(coord(a, t), 0), (coord(b, h), 1), (coord(c, w), 2)] {
                    let v = half_round(v, dt);
                    ang.extend(inv.iter().map(|f| std::f32::consts::TAU * v * f));
                }
            }
        }
    }
    ang.extend(std::iter::repeat_n(0.0, 5 * 24));
    let tab = kernels::rope_table(&Tensor::from_vec(ang, (t * h * w + 5, 24), dev)?)?;
    Ok(tab.to_dtype(dt)?.to_dtype(DType::F32)?)
}

fn half_round(v: f32, dt: DType) -> f32 {
    match dt {
        DType::F16 => half::f16::from_f32(v).to_f32(),
        DType::BF16 => half::bf16::from_f32(v).to_f32(),
        _ => v,
    }
}

pub fn split_tiles(len: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if TILE >= len {
        return (vec![0], vec![len], vec![]);
    }
    let mut n = len.div_ceil(TILE);
    let (mut overlaps, remaining) = loop {
        let ov = vec![TILE_OVERLAP_MIN; n - 1];
        let r = (TILE * n) as i64 - ov.iter().sum::<usize>() as i64 - len as i64;
        if r < 0 {
            n += 1;
        } else {
            break (ov, r as usize);
        }
    };
    for i in 0..remaining / RATIO {
        overlaps[i % (n - 1)] += RATIO;
    }
    let mut idx = vec![0];
    for o in overlaps.iter().take(n - 1) {
        let last = *idx.last().unwrap();
        idx.push(last + TILE - o);
    }
    (idx, vec![TILE; n], overlaps)
}

fn decode_temporal_chunks(z_len: usize) -> (usize, usize) {
    let mut pseudo = z_len + TOKEN_DROP;
    let mut pad = (TOKENS_CHUNK - pseudo % TOKENS_CHUNK) % TOKENS_CHUNK;
    pseudo += pad;
    let mut chunks = pseudo / TOKENS_CHUNK - 1;
    if chunks < 1 {
        pad += TOKENS_CHUNK;
        chunks += 1;
    }
    (pad, chunks)
}

fn decode_temporal_pad_frames(z_len: usize, pad: usize) -> usize {
    if pad == 0 {
        return 0;
    }
    let before = z_len - pad;
    (0..pad).map(|k| if (before + k) % TOKENS_CHUNK == 0 { CLIP_LENGTH % RATIO_T } else { RATIO_T }).sum()
}

fn decode_temporal_frame_plan(z_len: usize, chunks: usize, pad: usize) -> usize {
    let chunk_dec = TOKENS_CHUNK * RATIO_T;
    let (mut total, mut fin) = (0, 0);
    for i in 0..chunks {
        let t0 = i * TOKENS_CHUNK;
        let t1 = t0 + TOKENS_CHUNK + TOKEN_OVERLAP;
        let clip_frames = (t1.min(z_len).saturating_sub(t0.min(z_len))) * RATIO_T;
        for j in 0..2 {
            let f0 = j * chunk_dec;
            let f1 = (f0 + chunk_dec).min(clip_frames);
            let n = f1.saturating_sub(f0).saturating_sub(FRAME_PRE_PADDING);
            if j == 0 {
                total += n;
            } else {
                fin = n;
            }
        }
    }
    total + fin - decode_temporal_pad_frames(z_len, pad)
}

// ----------------------------------------------------------------------------------------------
// encoder: 3D causal CNN (reflect spatial padding, zero front temporal padding). A single-frame
// input keeps only each conv's last temporal tap, like the reference's causal_zero autopad.

fn conv3(f: &Arc<SafeFile>, p: &str, dev: &Device) -> Result<Conv> {
    let w = f.load_as(&format!("{p}.weight"), dev, DType::F32)?;
    let (k, kt) = (w.dim(3)?, w.dim(2)?);
    Ok(Conv { w, b: f.load_as(&format!("{p}.bias"), dev, DType::F32)?, k, kt })
}

fn load_encoder(f: &Arc<SafeFile>, dev: &Device) -> Result<Encoder> {
    let gn = |p: &str| -> Result<(Tensor, Tensor)> { Ok((f.load_as(&format!("{p}.weight"), dev, DType::F32)?, f.load_as(&format!("{p}.bias"), dev, DType::F32)?)) };
    let mut levels = vec![];
    for l in 0.. {
        let p = format!("encoder.down.{l}");
        if !f.tensors.keys().any(|k| k.starts_with(&format!("{p}."))) {
            break;
        }
        let mut blocks = vec![];
        for b in 0..2 {
            let bp = format!("{p}.block.{b}");
            let short = if f.has(&format!("{bp}.nin_shortcut.weight")) { Some(conv3(f, &format!("{bp}.nin_shortcut"), dev)?) } else { None };
            if f.has(&format!("{bp}.conv1.weight")) {
                blocks.push(ResBlock::Full {
                    n1: gn(&format!("{bp}.norm1"))?,
                    c1: conv3(f, &format!("{bp}.conv1"), dev)?,
                    n2: gn(&format!("{bp}.norm2"))?,
                    c2: conv3(f, &format!("{bp}.conv2"), dev)?,
                    short,
                });
            } else {
                blocks.push(ResBlock::Shortcut(short));
            }
        }
        let tstride = if l == 1 || l == 2 { 2 } else { 1 };
        let ds = if f.has(&format!("{p}.downsample.conv.weight")) { Some((conv3(f, &format!("{p}.downsample.conv"), dev)?, tstride)) } else { None };
        levels.push((blocks, ds));
    }
    Ok(Encoder {
        conv_in: conv3(f, "encoder.conv_in", dev)?,
        levels,
        norm_out: gn("encoder.norm_out")?,
        conv_out: conv3(f, "encoder.conv_out", dev)?,
        quant_conv: conv3(f, "quant_conv", dev)?,
    })
}

/// Reflect-pad the last two dims of a 4D or 5D tensor by 1-px steps.
fn reflect_pad(x: &Tensor, l: usize, r: usize, t: usize, b: usize) -> Result<Tensor> {
    let nd = x.rank();
    let (dh, dw) = (nd - 2, nd - 1);
    let mut x = x.clone();
    let w = x.dim(dw)?;
    if l > 0 || r > 0 {
        let mut parts = vec![];
        for i in (1..=l).rev() {
            parts.push(x.narrow(dw, i, 1)?);
        }
        parts.push(x.clone());
        for i in 0..r {
            parts.push(x.narrow(dw, w - 2 - i, 1)?);
        }
        x = Tensor::cat(&parts, dw)?;
    }
    let h = x.dim(dh)?;
    if t > 0 || b > 0 {
        let mut parts = vec![];
        for i in (1..=t).rev() {
            parts.push(x.narrow(dh, i, 1)?);
        }
        parts.push(x.clone());
        for i in 0..b {
            parts.push(x.narrow(dh, h - 2 - i, 1)?);
        }
        x = Tensor::cat(&parts, dh)?;
    }
    Ok(x)
}

/// conv2d over a batch of frames, chunked so im2col stays around 256 MB.
fn conv2d_chunked(x: &Tensor, w: &Tensor, stride: usize) -> Result<Tensor> {
    let (n, c, h, wd) = x.dims4()?;
    let k = w.dim(2)?;
    let per = (c * k * k * h * wd * 4).max(1);
    let chunk = (256usize << 20).div_ceil(per).max(1).min(n);
    if chunk >= n {
        return Ok(x.conv2d(w, 0, stride, 1, 1)?);
    }
    let mut outs = vec![];
    let mut s = 0;
    while s < n {
        let m = chunk.min(n - s);
        outs.push(x.narrow(0, s, m)?.conv2d(w, 0, stride, 1, 1)?);
        s += m;
    }
    Ok(Tensor::cat(&outs, 0)?)
}

/// Causal conv3d on `[B, C, T, H, W]` with explicit spatial padding (l, r, t, b) applied already
/// or not (`pad_spatial`), temporal stride `ts`, spatial stride `ss`.
fn causal_conv(c: &Conv, x: &Tensor, spatial: (usize, usize, usize, usize), ts: usize, ss: usize) -> Result<Tensor> {
    let (b, ci, t, _, _) = x.dims5()?;
    let x = if spatial != (0, 0, 0, 0) { reflect_pad(x, spatial.0, spatial.1, spatial.2, spatial.3)? } else { x.clone() };
    let (h, w) = (x.dim(3)?, x.dim(4)?);
    let co = c.w.dim(0)?;
    let (x, taps, t_out) = if t == 1 || c.kt == 1 {
        (x, vec![c.kt - 1], t)
    } else {
        let front = c.kt - 1;
        let xp = Tensor::cat(&[&Tensor::zeros((b, ci, front, h, w), x.dtype(), x.device())?, &x], 2)?;
        let t_out = (t + front - c.kt) / ts + 1;
        (xp, (0..c.kt).collect(), t_out)
    };
    let mut acc: Option<Tensor> = None;
    for k in taps {
        let idx: Vec<u32> = (0..t_out).map(|o| (o * ts + if t == 1 || c.kt == 1 { 0 } else { k }) as u32).collect();
        let frames = x.contiguous()?.index_select(&Tensor::from_vec(idx, t_out, x.device())?, 2)?; // [B, C, To, H, W]
        let frames = frames.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((b * t_out, ci, h, w))?;
        let wk = c.w.narrow(2, k, 1)?.squeeze(2)?.contiguous()?;
        let y = conv2d_chunked(&frames, &wk, ss)?;
        acc = Some(match acc {
            None => y,
            Some(a) => (a + y)?,
        });
    }
    let y = acc.unwrap();
    let (_, _, ho, wo) = y.dims4()?;
    let y = y.reshape((b, t_out, co, ho, wo))?.permute((0, 2, 1, 3, 4))?.contiguous()?;
    Ok(y.broadcast_add(&c.b.reshape((1, co, 1, 1, 1))?)?)
}

fn conv_same(c: &Conv, x: &Tensor) -> Result<Tensor> {
    let p = c.k / 2;
    causal_conv(c, x, (p, p, p, p), 1, 1)
}

/// Per-frame GroupNorm(32, eps 1e-6) + SiLU on `[B, C, T, H, W]`.
fn group_norm_silu(x: &Tensor, gn: &(Tensor, Tensor)) -> Result<Tensor> {
    let (b, c, t, h, w) = x.dims5()?;
    let g = 32;
    let xf = x.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((b * t, g, c / g * h * w))?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let y = xc.broadcast_div(&(var + 1e-6)?.sqrt()?)?.reshape((b * t, c, h, w))?;
    let y = y.broadcast_mul(&gn.0.reshape((1, c, 1, 1))?)?.broadcast_add(&gn.1.reshape((1, c, 1, 1))?)?;
    let y = ops::silu(&y)?;
    Ok(y.reshape((b, t, c, h, w))?.permute((0, 2, 1, 3, 4))?.contiguous()?)
}

fn encode_tile(e: &Encoder, x: &Tensor) -> Result<Tensor> {
    let mut h = conv_same(&e.conv_in, x)?;
    for (blocks, ds) in &e.levels {
        for b in blocks {
            h = match b {
                ResBlock::Full { n1, c1, n2, c2, short } => {
                    let y = conv_same(c1, &group_norm_silu(&h, n1)?)?;
                    let sc = match short {
                        Some(s) => causal_conv(s, &h, (0, 0, 0, 0), 1, 1)?,
                        None => h.clone(),
                    };
                    (conv_same(c2, &group_norm_silu(&y, n2)?)? + sc)?
                }
                ResBlock::Shortcut(Some(s)) => causal_conv(s, &h, (0, 0, 0, 0), 1, 1)?,
                ResBlock::Shortcut(None) => h,
            };
        }
        if let Some((d, ts)) = ds {
            h = causal_conv(d, &h, (0, 1, 0, 1), *ts, 2)?;
        }
    }
    let h = conv_same(&e.conv_out, &group_norm_silu(&h, &e.norm_out)?)?;
    causal_conv(&e.quant_conv, &h, (0, 0, 0, 0), 1, 1)
}


fn blend_t(a: &Tensor, b: &Tensor, extent: usize, dim: usize) -> Result<Tensor> {
    let e = extent.min(a.dim(dim)?).min(b.dim(dim)?);
    if e == 0 {
        return Ok(b.clone());
    }
    let wb: Vec<f32> = (0..e).map(|p| p as f32 / e as f32).collect();
    let mut shape = vec![1usize; a.rank()];
    shape[dim] = e;
    let wb = Tensor::from_vec(wb, shape, a.device())?;
    let wa = (1.0 - &wb)?;
    let an = a.dim(dim)?;
    let bl = (a.narrow(dim, an - e, e)?.broadcast_mul(&wa)? + b.narrow(dim, 0, e)?.broadcast_mul(&wb)?)?;
    let bn = b.dim(dim)?;
    if e < bn {
        Ok(Tensor::cat(&[&bl, &b.narrow(dim, e, bn - e)?], dim)?)
    } else {
        Ok(bl)
    }
}

// ----------------------------------------------------------------------------------------------
// CPU video buffer [C, F, H, W] for tile / chunk blending

#[derive(Clone)]
struct CpuVid {
    c: usize,
    f: usize,
    h: usize,
    w: usize,
    d: Vec<f32>,
}

impl CpuVid {
    fn zeros(c: usize, f: usize, h: usize, w: usize) -> Self {
        Self { c, f, h, w, d: vec![0.0; c * f * h * w] }
    }
    fn from_tensor(t: &Tensor) -> Result<Self> {
        let (c, f, h, w) = t.dims4()?;
        Ok(Self { c, f, h, w, d: t.flatten_all()?.to_vec1::<f32>()? })
    }
    fn planes(&self) -> usize {
        self.c * self.f
    }
    fn crop(&self, y0: usize, y1: usize, x0: usize, x1: usize) -> Self {
        let (y1, x1) = (y1.min(self.h), x1.min(self.w));
        let (h, w) = (y1 - y0, x1 - x0);
        let mut d = Vec::with_capacity(self.planes() * h * w);
        for p in 0..self.planes() {
            for y in y0..y1 {
                let o = p * self.h * self.w + y * self.w;
                d.extend_from_slice(&self.d[o + x0..o + x1]);
            }
        }
        Self { c: self.c, f: self.f, h, w, d }
    }
    fn paste(&mut self, t: &Self, y0: usize, x0: usize) {
        for p in 0..self.planes() {
            for y in 0..t.h {
                let o = p * self.h * self.w + (y0 + y) * self.w + x0;
                let s = p * t.h * t.w + y * t.w;
                self.d[o..o + t.w].copy_from_slice(&t.d[s..s + t.w]);
            }
        }
    }
    fn into_tensor(self) -> Result<Tensor> {
        Ok(Tensor::from_vec(self.d, (self.c, self.f, self.h, self.w), &Device::Cpu)?)
    }
}

fn blend_h(a: &CpuVid, b: &CpuVid, extent: usize) -> CpuVid {
    let e = extent.min(a.h).min(b.h);
    let mut out = b.clone();
    for p in 0..b.planes() {
        for y in 0..e {
            let wb = y as f32 / e as f32;
            let ay = a.h - e + y;
            for x in 0..b.w {
                let o = p * b.h * b.w + y * b.w + x;
                out.d[o] = a.d[p * a.h * a.w + ay * a.w + x] * (1.0 - wb) + b.d[o] * wb;
            }
        }
    }
    out
}

fn blend_w(a: &CpuVid, b: &CpuVid, extent: usize) -> CpuVid {
    let e = extent.min(a.w).min(b.w);
    let mut out = b.clone();
    for p in 0..b.planes() {
        for y in 0..b.h {
            for x in 0..e {
                let wb = x as f32 / e as f32;
                let o = p * b.h * b.w + y * b.w + x;
                out.d[o] = a.d[p * a.h * a.w + y * a.w + a.w - e + x] * (1.0 - wb) + b.d[o] * wb;
            }
        }
    }
    out
}

fn slice_frames(v: &CpuVid, f0: usize, f1: usize) -> CpuVid {
    let f1 = f1.min(v.f);
    let n = f1.saturating_sub(f0);
    let fs = v.h * v.w;
    let mut d = Vec::with_capacity(v.c * n * fs);
    for c in 0..v.c {
        d.extend_from_slice(&v.d[(c * v.f + f0) * fs..(c * v.f + f1) * fs]);
    }
    CpuVid { c: v.c, f: n, h: v.h, w: v.w, d }
}

fn blend_frames(a: &CpuVid, b: &CpuVid, extent: usize) -> CpuVid {
    let e = extent.min(a.f).min(b.f);
    let mut out = b.clone();
    let fs = b.h * b.w;
    for c in 0..b.c {
        for k in 0..e {
            let wb = k as f32 / e as f32;
            let ao = (c * a.f + a.f - e + k) * fs;
            let bo = (c * b.f + k) * fs;
            for i in 0..fs {
                out.d[bo + i] = a.d[ao + i] * (1.0 - wb) + b.d[bo + i] * wb;
            }
        }
    }
    out
}

fn finalize_inplace(v: &mut CpuVid) {
    let fs = v.f * v.h * v.w;
    for c in 0..v.c {
        for x in &mut v.d[c * fs..(c + 1) * fs] {
            *x = (*x * IMAGENET_STD[c] + IMAGENET_MEAN[c]).clamp(0.0, 1.0);
        }
    }
}

fn finalize(mut v: CpuVid) -> Result<Tensor> {
    finalize_inplace(&mut v);
    v.into_tensor()
}

fn write_part(out: &mut CpuVid, pos: &mut usize, mut part: CpuVid) {
    finalize_inplace(&mut part);
    let n = part.f.min(out.f.saturating_sub(*pos));
    let fs = out.h * out.w;
    for c in 0..out.c {
        let dst = (c * out.f + *pos) * fs;
        let src = c * part.f * fs;
        out.d[dst..dst + n * fs].copy_from_slice(&part.d[src..src + n * fs]);
    }
    *pos += n;
}
