//! Image input, resizing, and MP4 (h264 + AAC) / WAV output.

use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

/// Load an image as `[H, W, 3]` f32 in [0, 1] (CPU).
pub fn load_image(path: &str) -> Result<Tensor> {
    // sniff the format from the contents (uploads are saved without a meaningful extension)
    let img = image::ImageReader::open(path)
        .with_context(|| format!("opening image {path}"))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("decoding image {path}"))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let data: Vec<f32> = img.into_raw().into_iter().map(|v| v as f32 / 255.0).collect();
    Ok(Tensor::from_vec(data, (h as usize, w as usize, 3), &Device::Cpu)?)
}

fn to_rgb8(img: &Tensor) -> Result<image::RgbImage> {
    let (h, w, _) = img.dims3()?;
    let v: Vec<u8> = img.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    image::RgbImage::from_raw(w as u32, h as u32, v).context("image buffer")
}

/// ComfyUI `common_upscale(..., "lanczos", crop)`: optional center crop to the target aspect, then
/// a Lanczos resize. `img` is `[H, W, 3]` in [0, 1].
pub fn resize_lanczos(img: &Tensor, width: usize, height: usize, center_crop: bool) -> Result<Tensor> {
    let (h, w, _) = img.dims3()?;
    let mut src = img.clone();
    if center_crop {
        let old = w as f64 / h as f64;
        let new = width as f64 / height as f64;
        let (mut x, mut y, mut cw, mut ch) = (0, 0, w, h);
        if old > new {
            x = ((w as f64 - w as f64 * (new / old)) / 2.0).round() as usize;
            cw = w - 2 * x;
        } else if old < new {
            y = ((h as f64 - h as f64 * (old / new)) / 2.0).round() as usize;
            ch = h - 2 * y;
        }
        src = src.narrow(0, y, ch)?.narrow(1, x, cw)?;
    }
    let rgb = to_rgb8(&src.contiguous()?)?;
    let out = image::imageops::resize(&rgb, width as u32, height as u32, image::imageops::FilterType::Lanczos3);
    let data: Vec<f32> = out.into_raw().into_iter().map(|v| v as f32 / 255.0).collect();
    Ok(Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?)
}

/// Bilinear resize (`align_corners=False`, no antialias) of `[N, C, H, W]` f32 on the CPU.
pub fn resize_bilinear(x: &Tensor, oh: usize, ow: usize) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    let src = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; n * c * oh * ow];
    let (sy, sx) = (h as f32 / oh as f32, w as f32 / ow as f32);
    for p in 0..n * c {
        let base = p * h * w;
        for oy in 0..oh {
            let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
            let y0 = (fy as usize).min(h - 1);
            let y1 = (y0 + 1).min(h - 1);
            let dy = fy - y0 as f32;
            for ox in 0..ow {
                let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
                let x0 = (fx as usize).min(w - 1);
                let x1 = (x0 + 1).min(w - 1);
                let dx = fx - x0 as f32;
                let v = src[base + y0 * w + x0] * (1.0 - dy) * (1.0 - dx)
                    + src[base + y0 * w + x1] * (1.0 - dy) * dx
                    + src[base + y1 * w + x0] * dy * (1.0 - dx)
                    + src[base + y1 * w + x1] * dy * dx;
                out[p * oh * ow + oy * ow + ox] = v;
            }
        }
    }
    Ok(Tensor::from_vec(out, (n, c, oh, ow), &Device::Cpu)?)
}

/// Write stereo f32 audio `[2, L]` as 16-bit PCM WAV.
pub fn write_wav(path: &str, audio: &Tensor, sample_rate: u32) -> Result<()> {
    let (ch, len) = audio.dims2()?;
    let data = audio.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let spec = hound::WavSpec { channels: ch as u16, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec)?;
    for i in 0..len {
        for c in data.iter() {
            w.write_sample((c[i].clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
        }
    }
    w.finalize()?;
    Ok(())
}

/// Mux frames `[T, 3, H, W]` (f32 in [0, 1]) and optional stereo audio into an MP4 via ffmpeg.
pub fn write_mp4(path: &str, frames: &Tensor, fps: f64, audio: Option<(&Tensor, u32)>, crf: u32) -> Result<()> {
    let (t, c, h, w) = frames.dims4()?;
    if c != 3 {
        bail!("expected RGB frames");
    }
    let wav = audio.map(|_| format!("{path}.tmp.wav"));
    if let (Some(wp), Some((a, sr))) = (&wav, audio) {
        write_wav(wp, a, sr)?;
    }
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", &fps.to_string(), "-i", "-"]);
    if let Some(wp) = &wav {
        cmd.args(["-i", wp, "-c:a", "aac", "-b:a", "192k"]);
    }
    cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", &crf.to_string(), "-preset", "medium", "-movflags", "+faststart", path]);
    let mut child = cmd.stdin(Stdio::piped()).spawn().context("running ffmpeg (is it installed?)")?;
    {
        let stdin = child.stdin.as_mut().unwrap();
        let hwc = frames.to_dtype(DType::F32)?.permute((0, 2, 3, 1))?.contiguous()?;
        for i in 0..t {
            let f: Vec<u8> = hwc.get(i)?.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
            stdin.write_all(&f)?;
        }
    }
    let st = child.wait()?;
    if let Some(wp) = &wav {
        let _ = std::fs::remove_file(wp);
    }
    if !st.success() {
        bail!("ffmpeg failed with {st}");
    }
    Ok(())
}

/// Probe a media file's video size with ffprobe.
pub fn video_size(path: &str) -> Result<(usize, usize)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path])
        .output()
        .context("running ffprobe")?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.trim().split(',').map(|v| v.parse::<usize>());
    match (it.next(), it.next()) {
        (Some(Ok(w)), Some(Ok(h))) => Ok((w, h)),
        _ => bail!("could not read video size of {path}"),
    }
}

/// Decode a video file to frames `[T, H, W, 3]` f32 in [0, 1] at `fps` (CPU), at most `max_frames`.
pub fn load_video(path: &str, fps: f64, max_frames: usize) -> Result<Tensor> {
    let (w, h) = video_size(path)?;
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i", path, "-vf", &format!("fps={fps}"), "-frames:v", &max_frames.to_string(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .context("running ffmpeg")?;
    if !out.status.success() {
        bail!("ffmpeg could not decode {path}: {}", String::from_utf8_lossy(&out.stderr));
    }
    let n = out.stdout.len() / (w * h * 3);
    if n == 0 {
        bail!("no frames decoded from {path}");
    }
    let data: Vec<f32> = out.stdout[..n * w * h * 3].iter().map(|v| *v as f32 / 255.0).collect();
    Ok(Tensor::from_vec(data, (n, h, w, 3), &Device::Cpu)?)
}

/// True when the file has an audio stream.
pub fn has_audio(path: &str) -> bool {
    Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path])
        .output()
        .map(|o| !String::from_utf8_lossy(&o.stdout).trim().is_empty())
        .unwrap_or(false)
}

/// Decode any audio (or a video's soundtrack) to stereo `[2, L]` f32 at `sample_rate` (mono is
/// duplicated, like ComfyUI's replicate channel padding).
pub fn load_audio(path: &str, sample_rate: u32, max_seconds: Option<f64>) -> Result<Tensor> {
    let mut args: Vec<String> = vec!["-v".into(), "error".into(), "-i".into(), path.into(), "-vn".into()];
    if let Some(t) = max_seconds {
        args.extend(["-t".into(), format!("{t}")]);
    }
    args.extend(["-ac".into(), "2".into(), "-ar".into(), sample_rate.to_string(), "-f".into(), "f32le".into(), "-".into()]);
    let out = Command::new("ffmpeg").args(&args).output().context("running ffmpeg")?;
    if !out.status.success() {
        bail!("ffmpeg could not decode audio from {path}: {}", String::from_utf8_lossy(&out.stderr));
    }
    let s: Vec<f32> = out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let n = s.len() / 2;
    if n == 0 {
        bail!("no audio in {path}");
    }
    let t = Tensor::from_vec(s[..n * 2].to_vec(), (n, 2), &Device::Cpu)?;
    Ok(t.t()?.contiguous()?)
}
