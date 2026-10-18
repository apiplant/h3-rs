//! Denoise masks (ComfyUI `SetLatentNoiseMask` on a MiniMax H3 AV latent): regenerate part of a source video and its
//! soundtrack and keep the rest.
//!
//! A mask holds 1 where the video is regenerated and 0 where the source is kept; in between, the region is
//! partly regenerated. Kept regions are injected before every model call at the DiT's condition strength (so the
//! network sees them as clean, like keyframes), the model's prediction for them is replaced by the source after
//! it, and masked rows run at their own timestep (see `Dit`). This is ComfyUI's `KSamplerX0Inpaint` plus
//! `MiniMaxH3.scale_latent_inpaint` / `_token_grid_masks`.

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};

use crate::dit::{time_shift_sigma, TokenMasks, FRAME_PER_TOKEN, VISUAL_COND_TIMESTEP};
use crate::media;

/// What happens to the source's soundtrack under a video mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum MaskAudio {
    /// Regenerate the audio over the frames where the mask is set, keep it elsewhere.
    #[default]
    Follow,
    /// Keep the source's soundtrack whatever the video does.
    Keep,
    /// Regenerate all of the audio.
    Generate,
}

/// Masks on the latent grid: per video latent cell `[T, H, W]` and per audio latent frame.
#[derive(Debug, Clone)]
pub struct Masks {
    pub dims: (usize, usize, usize),
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

/// Video latent frames of `frames` pixel frames.
pub fn latent_frames(frames: usize) -> usize {
    ((frames.max(5) - 5) / 17) * 5 + 2
}

impl Masks {
    /// Resample a pixel mask `[F, H, W]` to the latent grid `(t, h, w)` the way ComfyUI does (trilinear,
    /// `align_corners=False`, no antialiasing), and derive the audio mask over `audio_t` latent frames.
    ///
    /// `audio_kept`: audio latent frames `[0, n)` hold the source's soundtrack; later ones have none and are
    /// always generated. `None` = the source has no audio.
    pub fn new(mask: &Tensor, grid: (usize, usize, usize), audio_t: usize, frame_count: usize, mode: MaskAudio, audio_kept: Option<usize>) -> Result<Self> {
        let (mf, mh, mw) = mask.dims3()?;
        let flat = mask.flatten_all()?.to_vec1::<f32>()?;
        let video: Vec<f32> = media::resize_trilinear(&flat, (mf, mh, mw), grid).into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
        // per pixel frame: the most-set pixel
        let per_frame: Vec<f32> = (0..mf).map(|f| flat[f * mh * mw..(f + 1) * mh * mw].iter().fold(0f32, |a, &b| a.max(b))).collect();
        let mut audio = vec![1f32; audio_t];
        if let Some(kept) = audio_kept {
            for (j, a) in audio.iter_mut().enumerate().take(kept.min(audio_t)) {
                *a = match mode {
                    MaskAudio::Generate => 1.0,
                    MaskAudio::Keep => 0.0,
                    MaskAudio::Follow => {
                        // audio latent frame j spans pixel frames [j * 0.6, (j + 1) * 0.6) at 24 fps / 40 Hz
                        let (p0, p1) = (j as f64 * 0.6, (j + 1) as f64 * 0.6);
                        let (f0, f1) = ((p0.floor() as usize).min(frame_count - 1), (p1.ceil() as usize).clamp(1, frame_count));
                        let m = (f0..f1.max(f0 + 1)).map(|p| per_frame[(p * mf / frame_count).min(mf - 1)]).fold(0f32, f32::max);
                        m.clamp(0.0, 1.0)
                    }
                };
            }
        }
        Ok(Self { dims: grid, video, audio })
    }

    /// Video mask pooled to the DiT's token grid: max over each 2x2 patch, rounded up to a multiple of 1/256.
    pub fn video_tokens(&self) -> Vec<f32> {
        let (t, h, w) = self.dims;
        let mut out = vec![0f32; t * h * w];
        for ti in 0..t {
            for y in (0..h).step_by(2) {
                for x in (0..w).step_by(2) {
                    let mut m = 0f32;
                    for dy in 0..2.min(h - y) {
                        for dx in 0..2.min(w - x) {
                            m = m.max(self.video[(ti * h + y + dy) * w + x + dx]);
                        }
                    }
                    let m = (m * 256.0).ceil() / 256.0;
                    for dy in 0..2.min(h - y) {
                        for dx in 0..2.min(w - x) {
                            out[(ti * h + y + dy) * w + x + dx] = m;
                        }
                    }
                }
            }
        }
        out
    }

    pub fn audio_tokens(&self) -> Vec<f32> {
        self.audio.iter().map(|m| (m * 256.0).ceil() / 256.0).collect()
    }

    /// What the DiT needs: a stream is left out when it generates everywhere.
    pub fn token_masks(&self) -> TokenMasks {
        let some_kept = |v: Vec<f32>| if v.iter().copied().fold(1f32, f32::min) < 1.0 - 1e-3 { Some(v) } else { None };
        TokenMasks { video: some_kept(self.video_tokens()), audio: some_kept(self.audio_tokens()) }
    }
}

/// Sampler-side state of a masked run (all tensors on the device; masks broadcast over channels).
pub struct Inpaint {
    dm_v: Tensor,
    keep_v: Tensor,
    w_v: Tensor,
    inj_v: Tensor,
    lat_v: Tensor,
    dm_a: Tensor,
    keep_a: Tensor,
    w_a: Tensor,
    /// Source audio on the sampler's carried scale (`audio_scale` x the audio latent).
    lat_a: Tensor,
    shift_v: f64,
    shift_a: f64,
    scale: f64,
}

impl Inpaint {
    /// `src_v` `[1, 24, T, H, W]` and `src_a` `[1, 32, 2, Ta]` (own scale; `None` = silence) are the source
    /// latents, `noise_v` the run's video noise (kept regions carry 0.1% of it, as ComfyUI's condition
    /// injection does).
    pub fn new(masks: &Masks, src_v: &Tensor, src_a: Option<&Tensor>, noise_v: &Tensor, shift_v: f64, shift_a: f64, dev: &Device) -> Result<Self> {
        let (t, h, w) = masks.dims;
        if src_v.dims() != noise_v.dims() || src_v.dim(2)? != t || src_v.dim(3)? != h || src_v.dim(4)? != w {
            bail!("source latent {:?} does not fit the {t}x{h}x{w} mask grid", src_v.dims());
        }
        let audio_t = masks.audio.len();
        let scale = shift_v / shift_a;
        let tok_v = masks.video_tokens();
        let tok_a = masks.audio_tokens();
        // weight of the current x in kept cells that share a token with a regenerated one
        let weight = |tok: &[f32], dm: &[f32]| -> Vec<f32> { tok.iter().zip(dm).map(|(&t, &d)| if d < 1.0 { ((t - d) / (1.0 - d).max(1e-6)).clamp(0.0, 1.0) } else { 0.0 }).collect() };
        let w_v = weight(&tok_v, &masks.video);
        let w_a = weight(&tok_a, &masks.audio);
        let vol = |v: &[f32]| Tensor::from_slice(v, (1, 1, t, h, w), dev);
        let aud = |v: &[f32]| Tensor::from_vec([v, v].concat(), (1, 1, 2, audio_t), dev);
        let keep = |v: &[f32]| -> Vec<f32> { v.iter().map(|d| 1.0 - d).collect() };
        let src_v = src_v.to_device(dev)?.to_dtype(candle_core::DType::F32)?;
        let aug = VISUAL_COND_TIMESTEP;
        let lat_a = match src_a {
            Some(a) => (a.to_device(dev)?.to_dtype(candle_core::DType::F32)? * scale)?,
            None => Tensor::zeros((1, 32, 2, audio_t), candle_core::DType::F32, dev)?,
        };
        Ok(Self {
            dm_v: vol(&masks.video)?,
            keep_v: vol(&keep(&masks.video))?,
            w_v: vol(&w_v)?,
            inj_v: ((&src_v * aug)? + (noise_v.to_device(dev)?.to_dtype(candle_core::DType::F32)? * (1.0 - aug))?)?,
            lat_v: src_v,
            dm_a: aud(&masks.audio)?,
            keep_a: aud(&keep(&masks.audio))?,
            w_a: aud(&w_a)?,
            lat_a,
            shift_v,
            shift_a,
            scale,
        })
    }

    fn blend(x: &Tensor, inj: &Tensor, w: &Tensor, dm: &Tensor, keep: &Tensor) -> Result<Tensor> {
        let i = (inj + (x - inj)?.broadcast_mul(w)?)?;
        Ok((x.broadcast_mul(dm)? + i.broadcast_mul(keep)?)?)
    }

    /// The model input: regenerated cells stay as sampled, kept cells are replaced by the (nearly) clean source.
    pub fn inject(&self, xv: &Tensor, xa: &Tensor, sigma: f64) -> Result<(Tensor, Tensor)> {
        let sigma_v = sigma.max(1e-6);
        let sigma_a = time_shift_sigma(sigma_v, self.shift_v, self.shift_a);
        // the sampler carries the audio as (sigma_v / sigma_a) x its own latent
        let inj_a = (&self.lat_a * ((sigma_v / sigma_a) / self.scale))?;
        Ok((Self::blend(xv, &self.inj_v, &self.w_v, &self.dm_v, &self.keep_v)?, Self::blend(xa, &inj_a, &self.w_a, &self.dm_a, &self.keep_a)?))
    }

    /// The denoised prediction with the source put back in the kept cells.
    pub fn merge(&self, dv: &Tensor, da: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok(((dv.broadcast_mul(&self.dm_v)? + self.lat_v.broadcast_mul(&self.keep_v)?)?, (da.broadcast_mul(&self.dm_a)? + self.lat_a.broadcast_mul(&self.keep_a)?)?))
    }
}

/// Latent frames the `FRAME_PER_TOKEN` pattern needs for `frames` pixel frames (sanity check helper).
pub fn pixel_frames(latent_t: usize) -> usize {
    (0..latent_t).map(|k| FRAME_PER_TOKEN[k % 5]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn mask(frames: usize, h: usize, w: usize, f: impl Fn(usize, usize, usize) -> f32) -> Tensor {
        let v: Vec<f32> = (0..frames).flat_map(|t| (0..h).flat_map(move |y| (0..w).map(move |x| (t, y, x)))).map(|(t, y, x)| f(t, y, x)).collect();
        Tensor::from_vec(v, (frames, h, w), &Device::Cpu).unwrap()
    }

    #[test]
    fn latent_frame_math() {
        assert_eq!(latent_frames(124), 37);
        assert_eq!(pixel_frames(latent_frames(124)), 124);
        assert_eq!(latent_frames(5), 2);
    }

    #[test]
    fn trilinear_identity_and_pool() {
        // same size: values are untouched
        let m = mask(2, 4, 4, |t, y, x| (t * 16 + y * 4 + x) as f32 / 40.0);
        let masks = Masks::new(&m, (2, 4, 4), 4, 5, MaskAudio::Generate, None).unwrap();
        assert_eq!(masks.video[5], 5.0 / 40.0);
        // token pooling takes the max of each 2x2 cell and rounds up to 1/256
        let tok = masks.video_tokens();
        assert_eq!(tok[0], tok[1]);
        assert_eq!(tok[0], tok[4]);
        assert_eq!(tok[0], ((5.0f32 / 40.0) * 256.0).ceil() / 256.0);
    }

    #[test]
    fn token_masks_skip_full_generation() {
        let m = mask(1, 4, 4, |_, _, _| 1.0);
        let masks = Masks::new(&m, (1, 4, 4), 4, 5, MaskAudio::Generate, Some(4)).unwrap();
        let t = masks.token_masks();
        assert!(t.video.is_none() && t.audio.is_none());
        let m = mask(1, 4, 4, |_, _, x| if x < 2 { 0.0 } else { 1.0 });
        let masks = Masks::new(&m, (1, 4, 4), 4, 5, MaskAudio::Keep, Some(4)).unwrap();
        let t = masks.token_masks();
        assert!(t.video.is_some() && t.audio.is_some());
    }

    #[test]
    fn audio_follows_the_mask_in_time() {
        // 48 pixel frames (2 s), the mask set from frame 24 on: audio latent frames 40.. regenerate
        let m = mask(48, 2, 2, |t, _, _| if t >= 24 { 1.0 } else { 0.0 });
        let masks = Masks::new(&m, (11, 2, 2), 80, 48, MaskAudio::Follow, Some(80)).unwrap();
        assert_eq!(masks.audio[0], 0.0);
        assert_eq!(masks.audio[39], 0.0);
        assert_eq!(masks.audio[41], 1.0);
        assert_eq!(masks.audio[79], 1.0);
        // audio beyond the source's soundtrack is always generated
        let masks = Masks::new(&m, (11, 2, 2), 80, 48, MaskAudio::Keep, Some(30)).unwrap();
        assert_eq!((masks.audio[29], masks.audio[30]), (0.0, 1.0));
    }

    /// Parity with ComfyUI on a dump from `tools/ref_inpaint.py` (set `H3_REF_INPAINT`).
    #[test]
    #[ignore]
    fn matches_comfyui() {
        let path = std::env::var("H3_REF_INPAINT").expect("H3_REF_INPAINT=<tools/ref_inpaint.py output>");
        let dev = Device::Cpu;
        let r = candle_core::safetensors::load(path, &dev).unwrap();
        let max_diff = |a: &Tensor, b: &Tensor| (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        let (_, _, t, h, w) = r["nv"].dims5().unwrap();
        let audio_t = r["na"].dim(3).unwrap();
        let masks = Masks::new(&r["mask"], (t, h, w), audio_t, 22, MaskAudio::Generate, None).unwrap();
        let dm = Tensor::from_vec(masks.video.clone(), (1, 1, t, h, w), &dev).unwrap();
        assert!(max_diff(&dm, &r["dm_v"].narrow(1, 0, 1).unwrap()) < 1e-6);
        let tok = Tensor::from_vec(masks.video_tokens(), (1, 1, t, h, w), &dev).unwrap();
        assert!(max_diff(&tok, &r["tok_v"].narrow(1, 0, 1).unwrap()) < 1e-6, "token grid mask");
        let ip = Inpaint::new(&masks, &r["Lv"], Some(&r["La"]), &r["nv"], 12.0, 3.0, &dev).unwrap();
        let sigma = r["sigma"].to_vec1::<f32>().unwrap()[0] as f64;
        let (xv, xa) = ip.inject(&r["xv"], &r["xa"], sigma).unwrap();
        let (dv, da) = (max_diff(&xv, &r["x_in_v"]), max_diff(&xa, &r["x_in_a"]));
        println!("x_in video {dv:e}, audio {da:e}");
        assert!(dv < 1e-5 && da < 1e-5);
    }
}
