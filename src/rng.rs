//! Noise identical to ComfyUI: `torch.randn(shape, generator=torch.manual_seed(seed), device="cpu")`
//! (mt19937 + PyTorch's CPU `normal_fill`), with one generator stream shared by consecutive draws.

use anyhow::Result;
use candle_core::{Device, Tensor};

/// PyTorch's `at::mt19937`.
pub struct TorchGen {
    state: [u32; 624],
    idx: usize,
    cached: Option<f64>,
}

impl TorchGen {
    /// `torch.manual_seed(seed)` / `torch.Generator().manual_seed(seed)`.
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = (seed & 0xffff_ffff) as u32;
        for i in 1..624 {
            state[i] = 1812433253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Self { state, idx: 624, cached: None }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            let mut v = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            self.state[i] = v;
        }
        self.idx = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.state[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn next_u64(&mut self) -> u64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }

    fn uniform(&mut self) -> f32 {
        (self.next_u32() & ((1 << 24) - 1)) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// One standard normal (scalar path, used by torch for tensors with < 16 elements).
    fn normal_scalar(&mut self) -> f32 {
        // at::normal_distribution<double>: Box-Muller on two 53-bit uniforms, sin half cached
        if let Some(c) = self.cached.take() {
            return c as f32;
        }
        let u = |g: &mut Self| -> f64 {
            let x = g.next_u64();
            (x & ((1u64 << 53) - 1)) as f64 * (1.0 / (1u64 << 53) as f64)
        };
        let u1 = u(self);
        let u2 = u(self);
        let r = (-2.0 * (-u2).ln_1p()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u1;
        self.cached = Some(r * th.sin());
        (r * th.cos()) as f32
    }

    /// `torch.randn(n)` for float32.
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        if n < 16 {
            return (0..n).map(|_| self.normal_scalar()).collect();
        }
        let mut d: Vec<f32> = (0..n).map(|_| self.uniform()).collect();
        let mut i = 0;
        while i + 16 <= n {
            fill16(&mut d[i..i + 16]);
            i += 16;
        }
        if n % 16 != 0 {
            let s = n - 16;
            for v in &mut d[s..] {
                *v = self.uniform();
            }
            fill16(&mut d[s..]);
        }
        d
    }

    pub fn randn_tensor(&mut self, shape: &[usize], dev: &Device) -> Result<Tensor> {
        let n = shape.iter().product();
        Ok(Tensor::from_vec(self.randn(n), shape, dev)?)
    }
}

fn fill16(d: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0 - d[j];
        let u2 = d[j + 8];
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f32::consts::PI * u2;
        d[j] = r * th.cos();
        d[j + 8] = r * th.sin();
    }
}

/// `torch.randn(shape, generator=torch.Generator().manual_seed(seed))`.
pub fn randn(seed: u64, shape: &[usize], dev: &Device) -> Result<Tensor> {
    TorchGen::new(seed).randn_tensor(shape, dev)
}
