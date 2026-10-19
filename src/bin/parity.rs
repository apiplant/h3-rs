//! Parity checks against reference dumps written by `tools/ref_*.py` (ComfyUI).
//!
//! usage: parity <ref_dir> <check>...   checks: dequant, dit, te, vae, audio, upscaler, upscaler_stages

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use h3_rs::config::Preset;
use h3_rs::weights::{QLinear, SafeFile};

const M: &str = "/mnt/extra/ai/ComfyUI/models";

fn report(name: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    let g = got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?;
    let w = want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?;
    let diff = (&g - &w)?.abs()?;
    let max = diff.max(0)?.to_scalar::<f32>()?;
    let mean = diff.mean_all()?.to_scalar::<f32>()?;
    let scale = w.abs()?.mean_all()?.to_scalar::<f32>()?;
    let dot = (&g * &w)?.sum_all()?.to_scalar::<f32>()?;
    let ng = g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nw = w.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    println!(
        "{name:28} max|d| {max:.3e}  mean|d| {mean:.3e}  mean|ref| {scale:.3e}  rel {:.3e}  cos {:.6}",
        mean / scale.max(1e-12),
        dot / (ng * nw).max(1e-12)
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let dev = h3_rs::backend::new_device(0)?;
    h3_rs::kernels::keep_pool_memory(&dev);
    for check in &args[2..] {
        match check.as_str() {
            "dequant" => {
                let r = SafeFile::open(format!("{dir}/dequant.safetensors"))?;
                let f = SafeFile::open(format!("{M}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"))?;
                let l = QLinear::load(&f, "blocks.0.attn.qkv_proj", h3_rs::weights::Placement::Gpu, &dev, DType::BF16)?;
                report("dit qkv dequant (plain)", &l.dequant_plain(&dev, DType::F32)?, &r.load("dit_qkv_plain", &dev)?)?;
                let f = SafeFile::open(format!("{M}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"))?;
                let l = QLinear::load(&f, "model.layers.0.mlp.down_proj", h3_rs::weights::Placement::Gpu, &dev, DType::BF16)?;
                report("te down_proj nvfp4", &l.dequant(&dev, DType::F32)?, &r.load("te_down_plain", &dev)?)?;
            }
            "dit" => {
                let r = SafeFile::open(format!("{dir}/dit_fwd.safetensors"))?;
                let mem = Preset::Gb24.config();
                let mut dit = h3_rs::dit::Dit::load(&format!("{M}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"), &dev, &mem, None)?;
                dit.capture_blocks = vec![0, 1, 49];
                let ctx = r.load("ctx", &dev)?.get(0)?;
                let refined = dit.refine_text(&ctx)?;
                if let Ok(t) = SafeFile::open(format!("{dir}/refiner_stages.safetensors")) {
                    for (i, st) in dit.refine_text_stages(&ctx)?.iter().enumerate() {
                        report(&format!("refiner stage {i}"), st, &t.load(&format!("s{i}"), &dev)?)?;
                    }
                }
                report("refined text", &refined, &r.load("text_refined", &dev)?)?;
                let v = r.load("v", &dev)?;
                let a = r.load("a", &dev)?;
                let cond = h3_rs::dit::DitCond {
                    text: ctx,
                    tags: None,
                    keyframes: vec![],
                    refs: vec![],
                    seed: 0,
                    visual_cond_noise_aug: h3_rs::dit::VISUAL_COND_TIMESTEP,
                    audio_cond_noise_aug: h3_rs::dit::AUDIO_COND_TIMESTEP,
                    denoise: None,
                    control: None,
                };
                let run = dit.prepare(&cond, v.dims(), a.dim(3)?, vec![])?;
                let sigma = r.load("sigma", &Device::Cpu)?.to_vec1::<f32>()?[0] as f64;
                let (ov, oa) = dit.forward(&run, &v, &a, sigma)?;
                report("dit out video", &ov, &r.load("out_v", &dev)?)?;
                report("dit out audio", &oa, &r.load("out_a", &dev)?)?;
                if r.has("mask_v") {
                    // under a denoise mask: per-row timesteps and scaled velocity
                    let flat = |t: Tensor| -> Result<Vec<f32>> { Ok(t.flatten_all()?.to_vec1::<f32>()?) };
                    let mv = flat(r.load("mask_v", &Device::Cpu)?)?;
                    let ma = flat(r.load("mask_a", &Device::Cpu)?.narrow(2, 0, 1)?)?;
                    let masked = h3_rs::dit::DitCond { denoise: Some(h3_rs::dit::TokenMasks { video: Some(mv), audio: Some(ma) }), ..cond.clone() };
                    let run = dit.prepare(&masked, v.dims(), a.dim(3)?, vec![])?;
                    let (ov, oa) = dit.forward(&run, &v, &a, sigma)?;
                    report("dit out video (masked)", &ov, &r.load("out_mv", &dev)?)?;
                    report("dit out audio (masked)", &oa, &r.load("out_ma", &dev)?)?;
                }
                for (i, t) in dit.captured.borrow().iter() {
                    let rf = r.load(&format!("block{i}"), &dev)?;
                    report(&format!("block {i} out"), t, &rf)?;
                    for (nm, a, n) in [("text", 0, 40), ("audio", 40, 16), ("video", 56, t.dim(0)? - 56)] {
                        report(&format!("  block {i} {nm}"), &t.narrow(0, a, n)?, &rf.narrow(0, a, n)?)?;
                    }
                }
            }
            "vae" => {
                let r = SafeFile::open(format!("{dir}/vae_vae_enc.safetensors"))?;
                let mem = Preset::Gb24.config();
                let vae = h3_rs::vae_video::VideoVae::load(&format!("{M}/vae/minimax_h3_video_vae_fp16.safetensors"), &dev, &mem)?;
                let px = vae.decode(&r.load("z", &dev)?)?; // [3, F, H, W]
                let want = r.load("pixels", &Device::Cpu)?.get(0)?.permute((3, 0, 1, 2))?; // [3, F, H, W]
                println!("decoded {:?} vs ref {:?}", px.dims(), want.dims());
                report("vae decode", &px, &want)?;
                let img = r.load("image", &Device::Cpu)?;
                let lat = vae.encode_image(&img)?;
                report("vae encode image", &lat, &r.load("latent", &dev)?)?;
            }
            "audio" => {
                let r = SafeFile::open(format!("{dir}/audio.safetensors"))?;
                let a = h3_rs::vae_audio::AudioVae::load(&format!("{M}/vae/minimax_h3_audio_vae_fp32.safetensors"), &dev)?;
                let w = a.decode(&r.load("z", &dev)?)?;
                report("audio decode", &w, &r.load("wave", &Device::Cpu)?)?;
            }
            "te" => {
                let r = SafeFile::open(format!("{dir}/te.safetensors"))?;
                let mem = Preset::Gb24.config();
                let tok = h3_rs::text::Tokenizer::load_default()?;
                let te = h3_rs::text::TextEncoder::load(&format!("{M}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"), &dev, &mem)?;
                let prompt = "A red fox trots through fresh snow at dawn, breath steaming, soft wind and crunching footsteps.";
                for name in ["t2v", "i2v"] {
                    let images = if name == "i2v" { vec![r.load(&format!("{name}_image"), &Device::Cpu)?] } else { vec![] };
                    let pieces = tok.presentation(prompt, &images)?;
                    let ids: Vec<i64> = pieces.iter().filter_map(|p| if let h3_rs::text::Piece::Token(t) = p { Some(*t as i64) } else { None }).collect();
                    let want_ids = r.load(&format!("{name}_ids"), &Device::Cpu)?.to_vec1::<i64>()?;
                    println!("{name}: token ids match: {}", ids == want_ids);
                    let enc = te.encode(&pieces)?;
                    let want_tags = r.load(&format!("{name}_tags"), &Device::Cpu)?.to_vec1::<i64>()?;
                    let tags: Vec<i64> = enc.tags.iter().map(|t| *t as i64).collect();
                    println!("{name}: {} tokens, tags match: {}", tags.len(), tags == want_tags);
                    report(&format!("{name} hidden"), &enc.hidden, &r.load(&format!("{name}_hidden"), &dev)?)?;
                }
            }
            "venc" => {
                let r = SafeFile::open(format!("{dir}/venc.safetensors"))?;
                let vae = h3_rs::vae_video::VideoVae::load(&format!("{M}/vae/minimax_h3_video_vae_fp16.safetensors"), &dev, &Preset::Gb24.config())?;
                let lat = vae.encode_frames(&r.load("frames", &Device::Cpu)?)?;
                report("vae encode 22 frames", &lat, &r.load("latent", &dev)?)?;
            }
            "aenc" => {
                let r = SafeFile::open(format!("{dir}/aenc.safetensors"))?;
                let e = h3_rs::vae_audio::AudioEncoder::load(&format!("{M}/vae/minimax_h3_audio_vae_fp32.safetensors"), &dev)?;
                report("audio encode", &e.encode(&r.load("wave", &dev)?)?, &r.load("z", &dev)?)?;
            }
            "upscaler" => {
                let r = SafeFile::open(format!("{dir}/upscaler.safetensors"))?;
                let up = h3_rs::upscaler::LatentUpscaler::load(format!("{M}/latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"), &dev)?;
                for c in ["a", "b", "c"] {
                    let z = r.load(&format!("{c}_z"), &dev)?;
                    let want = r.load(&format!("{c}_f16"), &dev)?;
                    let (ho, wo) = (want.dim(3)?, want.dim(4)?);
                    let scale = r.load(&format!("{c}_scale"), &Device::Cpu)?.to_vec1::<f32>()?[0] as f64;
                    let t = std::time::Instant::now();
                    let got = up.upscale(&z, ho, wo, Some(scale))?;
                    dev.synchronize()?;
                    println!("case {c}: {:?} -> {:?} in {:.2}s", z.dims(), got.dims(), t.elapsed().as_secs_f64());
                    report(&format!("upscaler {c} vs torch f16"), &got, &want)?;
                    report(&format!("upscaler {c} vs torch f32"), &got, &r.load(&format!("{c}_f32"), &dev)?)?;
                    report(&format!("  (torch f16 vs torch f32)"), &want, &r.load(&format!("{c}_f32"), &dev)?)?;
                }
            }
            "upscaler_stages" => {
                let r = SafeFile::open(format!("{dir}/upscaler_stages.safetensors"))?;
                let dt = if std::env::var("H3_F32").is_ok() { DType::F32 } else { DType::F16 };
                let up = h3_rs::upscaler::LatentUpscaler::load_as(format!("{M}/latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"), &dev, dt)?;
                let x = r.load("s_x", &dev)?.squeeze(0)?.permute((1, 0, 2, 3))?.contiguous()?;
                let scale = r.load("s_scale", &Device::Cpu)?.to_vec1::<f32>()?[0] as f64;
                for (name, got) in up.stages(&x, scale, 17, 22)? {
                    let want = r.load(&format!("s_{name}"), &dev)?.squeeze(0)?.permute((1, 0, 2, 3))?;
                    report(&format!("stage {name}"), &got, &want)?;
                }
            }
            c => anyhow::bail!("unknown check {c}"),
        }
    }
    Ok(())
}
