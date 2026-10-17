use candle_core::{DType, Device, Tensor};
fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    h3_rs::kernels::keep_pool_memory(&dev);
    let args: Vec<usize> = std::env::args().skip(1).map(|v| v.parse().unwrap()).collect();
    let (m, k, n) = if args.len() == 3 { (args[0], args[1], args[2]) } else { (8192, 5376, 28672) };
    let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::BF16)?;
    let wf = Tensor::randn(0f32, 1.0, (n, k), &dev)?;
    let w8 = ((wf.clamp(-3.0, 3.0)? * 40.0)?.round()? + 128.0)?.to_dtype(DType::U8)?;
    let s = (Tensor::rand(0.5f32, 1.5, (n, 1), &dev)? * 0.01)?;
    let bias = Tensor::randn(0f32, 1.0, n, &dev)?;
    let flops = 2.0 * (m * k * n) as f64;
    let mut outs = vec![];
    for name in ["cublaslt", "fused"] {
        let mut y = None;
        for it in 0..5 {
            dev.synchronize()?;
            let t = std::time::Instant::now();
            y = Some(if name == "cublaslt" {
                h3_rs::kernels::int8_linear(&x, &w8, &s, Some(&bias), DType::BF16)?
            } else {
                h3_rs::attn8::gemm_s8(&x, &w8, &s, Some(&bias), false, false)?
            });
            dev.synchronize()?;
            if it == 4 {
                let dt = t.elapsed().as_secs_f64();
                println!("{name}: {:.2} ms  {:.0} TOPS", dt * 1e3, flops / dt / 1e12);
            }
        }
        outs.push(y.unwrap().to_dtype(DType::F32)?);
    }
    let d = (&outs[0] - &outs[1])?.abs()?.max_all()?.to_scalar::<f32>()?;
    let mag = outs[0].abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("max |cublaslt - fused| = {d:.4e} (mean |y| {mag:.3})");
    Ok(())
}
