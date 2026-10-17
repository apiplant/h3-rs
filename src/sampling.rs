//! Flow-matching schedule and samplers (ComfyUI `simple` scheduler over `ModelSamplingAV`,
//! `euler` and `res_multistep`).

use anyhow::Result;
use candle_core::Tensor;

use crate::dit::{Dit, DitRun};

pub fn time_snr_shift(alpha: f64, t: f64) -> f64 {
    if alpha == 1.0 {
        return t;
    }
    alpha * t / (1.0 + (alpha - 1.0) * t)
}

/// ComfyUI `simple` scheduler on the shifted 1000-step flow grid; ends with 0.
pub fn simple_sigmas(shift: f64, steps: usize) -> Vec<f64> {
    let n = 1000usize;
    let table: Vec<f64> = (1..=n).map(|i| time_snr_shift(shift, i as f64 / n as f64)).collect();
    let ss = n as f64 / steps as f64;
    let mut s: Vec<f64> = (0..steps).map(|x| table[n - 1 - (x as f64 * ss) as usize]).collect();
    s.push(0.0);
    s
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Sampler {
    Euler,
    ResMultistep,
}

/// Run the sampler over the packed (video, carried audio) latent. `on_step(i, n)` is called
/// after each step.
pub fn sample(
    dit: &Dit,
    run: &DitRun,
    mut xv: Tensor,
    mut xa: Tensor,
    sigmas: &[f64],
    sampler: Sampler,
    mut on_step: impl FnMut(usize, usize),
) -> Result<(Tensor, Tensor)> {
    let mut old: Option<(Tensor, Tensor)> = None;
    let mut old_sigma_down: Option<f64> = None;
    let n = sigmas.len() - 1;
    for i in 0..n {
        let (s, s_next) = (sigmas[i], sigmas[i + 1]);
        let (ov, oa) = dit.forward(run, &xv, &xa, s)?;
        // CONST: denoised = x - out * sigma
        let dv = (&xv - (&ov * s)?)?;
        let da = (&xa - (&oa * s)?)?;
        let euler = sampler == Sampler::Euler || s_next == 0.0 || old.is_none();
        if euler {
            xv = (&xv + (&ov * (s_next - s))?)?;
            xa = (&xa + (&oa * (s_next - s))?)?;
        } else {
            let tf = |x: f64| -x.ln();
            let (t, t_old, t_next, t_prev) = (tf(s), tf(old_sigma_down.unwrap()), tf(s_next), tf(sigmas[i - 1]));
            let h = t_next - t;
            let c2 = (t_prev - t_old) / h;
            let phi1 = (-h).exp_m1() / -h;
            let phi2 = (phi1 - 1.0) / -h;
            let nz = |v: f64| if v.is_nan() { 0.0 } else { v };
            let (b1, b2) = (nz(phi1 - phi2 / c2), nz(phi2 / c2));
            let (odv, oda) = old.as_ref().unwrap();
            let e = (-h).exp();
            xv = ((&xv * e)? + ((&dv * (h * b1))? + (odv * (h * b2))?)?)?;
            xa = ((&xa * e)? + ((&da * (h * b1))? + (oda * (h * b2))?)?)?;
        }
        old = Some((dv, da));
        old_sigma_down = Some(s_next);
        on_step(i + 1, n);
        crate::dit::prof::dump();
    }
    Ok((xv, xa))
}
