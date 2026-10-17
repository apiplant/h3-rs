use candle_core::{DType, Device, Tensor};
fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    h3_rs::kernels::keep_pool_memory(&dev);
    let (s, h) = (std::env::args().nth(1).map(|v| v.parse().unwrap()).unwrap_or(37700usize), 56usize);
    // q/k with a per-channel offset (outliers like real K) and moderate logits
    let off = (Tensor::randn(0f32, 3.0, (1, h, 128), &dev)?).broadcast_as((s, h, 128))?;
    let q = Tensor::randn(0f32, 1.0, (s, h, 128), &dev)?.to_dtype(DType::BF16)?;
    let k = (Tensor::randn(0f32, 1.0, (s, h, 128), &dev)? + off)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1.0, (s, h, 128), &dev)?.to_dtype(DType::BF16)?;
    let mut ref_o = None;
    for name in ["flash", "int8"] {
        let mut o = None;
        for it in 0..3 {
            dev.synchronize()?;
            let t = std::time::Instant::now();
            let r = if name == "flash" {
                h3_rs::ops::attention(&q.unsqueeze(0)?, &k.unsqueeze(0)?, &v.unsqueeze(0)?, false)?.squeeze(0)?
            } else {
                h3_rs::attn8::attention(&q, &k, &v)?
            };
            dev.synchronize()?;
            let dt = t.elapsed().as_secs_f64();
            if it == 2 {
                let fl = 4.0 * (s * s) as f64 * 128.0 * h as f64;
                println!("{name}: {:.1} ms  {:.0} TFLOPS", dt * 1e3, fl / dt / 1e12);
            }
            o = Some(r);
        }
        let o = o.unwrap().to_dtype(DType::F32)?;
        if let Some(r) = &ref_o {
            let r: &Tensor = r;
            let d = (&o - r)?.abs()?.mean_all()?.to_scalar::<f32>()?;
            let m = r.abs()?.mean_all()?.to_scalar::<f32>()?;
            let cos = (&o * r)?.sum_all()?.to_scalar::<f32>()? / (o.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() * r.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt());
            println!("int8 vs flash: rel {:.3e}  cos {cos:.6}", d / m);
        } else {
            ref_o = Some(o);
        }
    }
    Ok(())
}
