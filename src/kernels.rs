//! Custom CUDA kernels (compiled once with NVRTC): dequantization of the ComfyUI quantized
//! formats, fused per-head RMSNorm + split-half RoPE, and adaLN modulation/gating by per-token
//! table rows.

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, WrapErr};
use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor};

const SRC: &str = r#"
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned int u32;

__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u16 f2bf(float f) {
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
__device__ __forceinline__ float h2f(u16 v) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(v)); return f; }
__device__ __forceinline__ u16 f2h(float f) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }

// kind: 0 = f32, 1 = f16, 2 = bf16
__device__ __forceinline__ float ldf(const void* p, int kind, size_t i) {
    if (kind == 0) return ((const float*)p)[i];
    if (kind == 1) return h2f(((const u16*)p)[i]);
    return bf2f(((const u16*)p)[i]);
}
__device__ __forceinline__ void stf(void* p, int kind, size_t i, float v) {
    if (kind == 0) ((float*)p)[i] = v;
    else if (kind == 1) ((u16*)p)[i] = f2h(v);
    else ((u16*)p)[i] = f2bf(v);
}

// int8 rows * (per-row or scalar) f32 scale -> out
extern "C" __global__ void dequant_i8(const signed char* q, const float* scale, void* out,
                                      unsigned long long rows, unsigned long long cols, int per_row, int okind) {
    unsigned long long n = rows * cols;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned long long r = i / cols;
        float s = per_row ? scale[r] : scale[0];
        stf(out, okind, i, (float)q[i] * s);
    }
}

__device__ __forceinline__ float fp8e4m3(u8 v) {
    int sgn = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float r;
    if (e == 0) r = ldexpf((float)m / 8.0f, -6);
    else if (e == 15 && m == 7) r = __int_as_float(0x7fc00000);
    else r = ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return sgn ? -r : r;
}

__constant__ float E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// NVFP4: q [N, K/2] packed hi-first, block scales fp8 e4m3 in the cuBLAS 128x4 swizzled layout,
// global f32 scale. out [N, K].
extern "C" __global__ void dequant_nvfp4(const u8* q, const u8* bscale, const float* gscale, void* out,
                                         unsigned long long nrows, unsigned long long k, int okind) {
    unsigned long long n = nrows * k;
    unsigned long long nb = k / 16;
    unsigned long long ncb = (nb + 3) / 4;
    float g = gscale[0];
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned long long r = i / k, c = i % k;
        u8 byte = q[r * (k / 2) + c / 2];
        int nib = (c & 1) ? (byte & 15) : (byte >> 4);
        unsigned long long b = c / 16;
        unsigned long long rb = r / 128, rin = r % 128, cb = b / 4, cin = b % 4;
        unsigned long long off = (rb * ncb + cb) * 512 + (rin % 32) * 16 + (rin / 32) * 4 + cin;
        stf(out, okind, i, E2M1[nib] * fp8e4m3(bscale[off]) * g);
    }
}

// Per (token, head) row: optional RMSNorm (optional weight), then split-half RoPE over the first
// rot dims. x is read at x + s*row_stride + col_off + h*D (elements); out is contiguous [S, H, D].
// cs: [S, rot/2] cos then sin tables as f32 (cs = cos, cs + S*rot/2 = sin). D <= 256.
extern "C" __global__ void qk_norm_rope(const void* x, int xkind, unsigned long long row_stride,
                                        unsigned long long col_off, unsigned long long head_stride,
                                        const void* w, int wkind, int has_w,
                                        int do_norm, float eps, const float* cs, int rot, void* out,
                                        int okind, int S, int H, int D) {
    __shared__ float buf[256];
    __shared__ float red[32];
    int row = blockIdx.x;
    int s = row / H, h = row % H;
    int d = threadIdx.x;
    float v = 0.f;
    if (d < D) v = ldf(x, xkind, (size_t)s * row_stride + col_off + (size_t)h * head_stride + d);
    if (do_norm) {
        float sq = v * v;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, o);
        if ((d & 31) == 0) red[d >> 5] = sq;
        __syncthreads();
        if (d < 32) {
            float t = (d < (blockDim.x + 31) / 32) ? red[d] : 0.f;
            for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffff, t, o);
            if (d == 0) red[0] = t;
        }
        __syncthreads();
        float inv = rsqrtf(red[0] / (float)D + eps);
        v = v * inv * (has_w && d < D ? ldf(w, wkind, d) : 1.f);
    }
    buf[d] = v;
    __syncthreads();
    if (d < D) {
        float o = v;
        int half = rot / 2;
        if (d < rot) {
            int j = d % half;
            float c = cs[(size_t)s * half + j];
            float sn = cs[(size_t)S * half + (size_t)s * half + j];
            if (d < half) o = v * c - buf[d + half] * sn;
            else o = v * c + buf[d - half] * sn;
        }
        stf(out, okind, (size_t)row * D + d, o);
    }
}

// BigVGAN anti-aliased activation, fused: x [B*C, L] -> down2(snakebeta(up2(x))) [B*C, L].
// up2: replicate pad 5, conv_transpose stride 2 with fu[12] * 2, crop 15/15.
// down2: replicate pad (5, 6), conv stride 2 with fd[12]. alpha/beta are log-scale per channel.
extern "C" __global__ void act1d(const float* x, const float* fu, const float* fd, const float* la,
                                 const float* lb, float* out, int C, int L, int rows) {
    unsigned long long n_out = (unsigned long long)rows * L;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n_out;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        int r = i / L, n = i % L, c = r % C;
        const float* xr = x + (size_t)r * L;
        float alpha = expf(la[c]), beta = expf(lb[c]);
        float inv_b = 1.0f / (beta + 1e-9f);
        float acc = 0.f;
        for (int k = 0; k < 12; ++k) {
            int m = 2 * n + k - 5;
            m = m < 0 ? 0 : (m > 2 * L - 1 ? 2 * L - 1 : m);
            // up2 sample m: 2 * sum_i xp[i] * fu[m + 15 - 2i], xp[i] = x[clamp(i - 5)]
            float u = 0.f;
            // taps kk = m + 15 - 2i in [0, 11]  <=>  i in [ceil((m+4)/2), floor((m+15)/2)]
            int i_lo = (m + 5) / 2;
            int i_hi = (m + 15) / 2;
            for (int ii = i_lo; ii <= i_hi; ++ii) {
                int kk = m + 15 - 2 * ii;
                if (kk < 0 || kk > 11) continue;
                int xi = ii - 5;
                xi = xi < 0 ? 0 : (xi > L - 1 ? L - 1 : xi);
                u += xr[xi] * fu[kk];
            }
            u *= 2.0f;
            float sn = sinf(alpha * u);
            acc += (u + inv_b * sn * sn) * fd[k];
        }
        out[i] = acc;
    }
}

// per-row int8 quantization with a fused ConvRot rotation (256-group regular Hadamard, applied as
// four radix-4 butterfly stages). blockDim = 256, K = G * 256 with G <= 64.
extern "C" __global__ void quant_rows_i8_rot(const void* x, int kind, signed char* q, float* scale, int K) {
    __shared__ float sm[256];
    __shared__ float red[8];
    const int r = blockIdx.x, i = threadIdx.x, G = K / 256;
    const size_t base = (size_t)r * K;
    float v[64];
    float m = 0.f;
    #pragma unroll
    for (int g = 0; g < 64; ++g) {
        if (g < G) {
            sm[i] = ldf(x, kind, base + g * 256 + i);
            __syncthreads();
            #pragma unroll
            for (int st = 0, stride = 1; st < 4; ++st, stride *= 4) {
                const int p = (i / stride) & 3;
                const int i0 = i - p * stride;
                const float a0 = sm[i0], a1 = sm[i0 + stride], a2 = sm[i0 + 2 * stride], a3 = sm[i0 + 3 * stride];
                // rows of [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
                float o;
                if (p == 0) o = a0 + a1 + a2 - a3;
                else if (p == 1) o = a0 + a1 - a2 + a3;
                else if (p == 2) o = a0 - a1 + a2 + a3;
                else o = -a0 + a1 + a2 + a3;
                __syncthreads();
                sm[i] = o;
                __syncthreads();
            }
            v[g] = sm[i] * (1.f / 16.f);
            m = fmaxf(m, fabsf(v[g]));
            __syncthreads();
        }
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((i & 31) == 0) red[i >> 5] = m;
    __syncthreads();
    if (i < 32) {
        float t = i < 8 ? red[i] : 0.f;
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffff, t, o));
        if (i == 0) red[0] = t;
    }
    __syncthreads();
    const float s = fmaxf(red[0] / 127.f, 1e-30f), inv = 1.f / s;
    if (i == 0) scale[r] = s;
    #pragma unroll
    for (int g = 0; g < 64; ++g) {
        if (g < G) q[base + g * 256 + i] = (signed char)fminf(fmaxf(rintf(v[g] * inv), -128.f), 127.f);
    }
}

// Warp-level ConvRot rotation + per-row int8 quantization. One block (8 warps) per row; each warp
// transforms 256-groups held as 8 contiguous values per lane. Regular Hadamard H4 applied per
// base-4 digit of the in-group index: out_p = sum(a) - 2 * a_{3-p}.
__device__ __forceinline__ float h4(float s, float partner) { return s - 2.f * partner; }

extern "C" __global__ void quant_rows_i8_rot2(const u16* x, signed char* q, float* scale, int K) {
    __shared__ float red[8];
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = K / 256;
    const size_t base = (size_t)r * K;
    float v[7][8];
    float m = 0.f;
    #pragma unroll
    for (int gi = 0; gi < 7; ++gi) {
        const int g = warp + gi * 8;
        if (g >= G) break;
        uint4 raw = *(const uint4*)(x + base + g * 256 + lane * 8);
        float a[8];
        a[0] = bf2f(raw.x & 0xffff); a[1] = bf2f(raw.x >> 16); a[2] = bf2f(raw.y & 0xffff); a[3] = bf2f(raw.y >> 16);
        a[4] = bf2f(raw.z & 0xffff); a[5] = bf2f(raw.z >> 16); a[6] = bf2f(raw.w & 0xffff); a[7] = bf2f(raw.w >> 16);
        // digit 0 (e & 3): in-thread
        #pragma unroll
        for (int hb = 0; hb < 8; hb += 4) {
            float s = a[hb] + a[hb + 1] + a[hb + 2] + a[hb + 3];
            float b0 = h4(s, a[hb + 3]), b1 = h4(s, a[hb + 2]), b2 = h4(s, a[hb + 1]), b3 = h4(s, a[hb]);
            a[hb] = b0; a[hb + 1] = b1; a[hb + 2] = b2; a[hb + 3] = b3;
        }
        // digit 1 = (e >> 2) + 2 * (lane & 1): partners e ^ 4 in-thread, lane ^ 1
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            float lo = a[e], hi = a[e + 4];
            float s = lo + hi;
            s += __shfl_xor_sync(0xffffffff, s, 1);
            float plo = __shfl_xor_sync(0xffffffff, hi, 1); // partner of (lo) is (hi) on lane^1
            float phi = __shfl_xor_sync(0xffffffff, lo, 1);
            a[e] = h4(s, plo);
            a[e + 4] = h4(s, phi);
        }
        // digit 2 = (lane >> 1) & 3, digit 3 = (lane >> 3) & 3
        #pragma unroll
        for (int d = 0; d < 2; ++d) {
            const int sh = d == 0 ? 1 : 3;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                float s = a[e];
                s += __shfl_xor_sync(0xffffffff, s, 1 << sh);
                s += __shfl_xor_sync(0xffffffff, s, 2 << sh);
                float p = __shfl_xor_sync(0xffffffff, a[e], 3 << sh);
                a[e] = h4(s, p);
            }
        }
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[gi][e] = a[e] * (1.f / 16.f);
            m = fmaxf(m, fabsf(v[gi][e]));
        }
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    float mx = red[0];
    #pragma unroll
    for (int i = 1; i < 8; ++i) mx = fmaxf(mx, red[i]);
    const float s = fmaxf(mx / 127.f, 1e-30f), inv = 1.f / s;
    if (threadIdx.x == 0) scale[r] = s;
    #pragma unroll
    for (int gi = 0; gi < 7; ++gi) {
        const int g = warp + gi * 8;
        if (g >= G) break;
        u32 lo = 0, hi = 0;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            int qv = __float2int_rn(v[gi][e] * inv);
            qv = max(-128, min(127, qv)) & 0xff;
            if (e < 4) lo |= (u32)qv << (8 * e); else hi |= (u32)qv << (8 * (e - 4));
        }
        *(uint2*)(q + base + g * 256 + lane * 8) = make_uint2(lo, hi);
    }
}

// per-row symmetric int8 quantization: one block per row
extern "C" __global__ void quant_rows_i8(const void* x, int kind, signed char* q, float* scale, int K) {
    __shared__ float red[32];
    int r = blockIdx.x;
    size_t base = (size_t)r * K;
    float m = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) m = fmaxf(m, fabsf(ldf(x, kind, base + i)));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffff, t, o));
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    float s = fmaxf(red[0] / 127.f, 1e-30f);
    if (threadIdx.x == 0) scale[r] = s;
    float inv = 1.f / s;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = rintf(ldf(x, kind, base + i) * inv);
        q[base + i] = (signed char)fminf(fmaxf(v, -128.f), 127.f);
    }
}

// out[m, n] = acc[m, n] * xs[m] * ws[n or 0] (+ bias[n]); grid (ceil(N/4/256), M), 4 columns per thread
__device__ __forceinline__ void st4(void* out, int okind, size_t i, float a, float b, float c, float d) {
    if (okind == 2) {
        uint2 v; v.x = (u32)f2bf(a) | ((u32)f2bf(b) << 16); v.y = (u32)f2bf(c) | ((u32)f2bf(d) << 16);
        *(uint2*)((u16*)out + i) = v;
    } else if (okind == 1) {
        uint2 v; v.x = (u32)f2h(a) | ((u32)f2h(b) << 16); v.y = (u32)f2h(c) | ((u32)f2h(d) << 16);
        *(uint2*)((u16*)out + i) = v;
    } else {
        *(float4*)((float*)out + i) = make_float4(a, b, c, d);
    }
}

extern "C" __global__ void int8_epilogue(const int* acc, const float* xs, const float* ws, int ws_per_row,
                                         const float* bias, int has_bias, void* out, int okind,
                                         unsigned long long M, unsigned long long N) {
    const int m = blockIdx.y;
    const int n = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (n >= (int)N) return;
    const float x = xs[m];
    const int4 a = *(const int4*)(acc + (size_t)m * N + n);
    float4 w = ws_per_row ? *(const float4*)(ws + n) : make_float4(ws[0], ws[0], ws[0], ws[0]);
    float4 bb = has_bias ? *(const float4*)(bias + n) : make_float4(0.f, 0.f, 0.f, 0.f);
    st4(out, okind, (size_t)m * N + n, a.x * x * w.x + bb.x, a.y * x * w.y + bb.y, a.z * x * w.z + bb.z, a.w * x * w.w + bb.w);
}

__device__ __forceinline__ float silu_r(float g, int okind) {
    if (okind == 2) g = bf2f(f2bf(g));
    float s = g / (1.f + __expf(-g));
    return okind == 2 ? bf2f(f2bf(s)) : s;
}

// swiglu epilogue: acc [M, 2F] -> out[m, j] = silu(gate) * up (gate = col j, up = col F + j)
extern "C" __global__ void int8_epilogue_swiglu(const int* acc, const float* xs, const float* ws, int ws_per_row,
                                                const float* bias, int has_bias, void* out, int okind,
                                                unsigned long long M, unsigned long long F) {
    const int m = blockIdx.y;
    const int j = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (j >= (int)F) return;
    const float x = xs[m];
    const int* row = acc + (size_t)m * 2 * F;
    const int4 ga = *(const int4*)(row + j);
    const int4 ua = *(const int4*)(row + F + j);
    float4 wg = ws_per_row ? *(const float4*)(ws + j) : make_float4(ws[0], ws[0], ws[0], ws[0]);
    float4 wu = ws_per_row ? *(const float4*)(ws + F + j) : wg;
    float4 bg = has_bias ? *(const float4*)(bias + j) : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 bu = has_bias ? *(const float4*)(bias + F + j) : make_float4(0.f, 0.f, 0.f, 0.f);
    auto up = [&](float v) { return okind == 2 ? bf2f(f2bf(v)) : v; };
    st4(out, okind, (size_t)m * F + j,
        silu_r(ga.x * x * wg.x + bg.x, okind) * up(ua.x * x * wu.x + bu.x),
        silu_r(ga.y * x * wg.y + bg.y, okind) * up(ua.y * x * wu.y + bu.y),
        silu_r(ga.z * x * wg.z + bg.z, okind) * up(ua.z * x * wu.z + bu.z),
        silu_r(ga.w * x * wg.w + bg.w, okind) * up(ua.w * x * wu.w + bu.w));
}

// fused RMSNorm(weight w) + adaLN modulate, one block per row; rounding matches the unfused path
extern "C" __global__ void rms_modulate(const void* x, int kind, const void* w, float eps, const float* scale,
                                        const float* shift, const u32* idx, void* out, int C) {
    __shared__ float red[32];
    const int r = blockIdx.x;
    const size_t base = (size_t)r * C;
    float sq = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { float v = ldf(x, kind, base + c); sq += v * v; }
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffff, t, o);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)C + eps);
    const size_t row = (size_t)idx[r] * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float n = ldf(x, kind, base + c) * inv * ldf(w, kind, c);
        float sc = scale[row + c], sh = shift[row + c];
        if (kind == 2) { n = bf2f(f2bf(n)); sc = bf2f(f2bf(sc)); sh = bf2f(f2bf(sh)); }
        else if (kind == 1) { n = h2f(f2h(n)); sc = h2f(f2h(sc)); sh = h2f(f2h(sh)); }
        stf(out, kind, base + c, n * (1.f + sc) + sh);
    }
}

// out[s, c] = x[s, c] * (1 + scale[idx[s], c]) + shift[idx[s], c]; tables f32 [R, C]
extern "C" __global__ void modulate(const void* x, int kind, const float* scale, const float* shift,
                                    const u32* idx, void* out, unsigned long long S, unsigned long long C) {
    unsigned long long n = S * C;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned long long s = i / C, c = i % C;
        unsigned long long r = idx[s];
        float sc = bf2f(f2bf(scale[r * C + c])), sh = bf2f(f2bf(shift[r * C + c]));
        if (kind == 1) { sc = h2f(f2h(scale[r * C + c])); sh = h2f(f2h(shift[r * C + c])); }
        stf(out, kind, i, ldf(x, kind, i) * (1.f + sc) + sh);
    }
}

// out[s, c] = x[s, c] + y[s, c] * gate[idx[s], c]
extern "C" __global__ void gate_add(const void* x, const void* y, int kind, const float* gate,
                                    const u32* idx, void* out, unsigned long long S, unsigned long long C) {
    unsigned long long n = S * C;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned long long s = i / C, c = i % C;
        float g = gate[(unsigned long long)idx[s] * C + c];
        g = (kind == 1) ? h2f(f2h(g)) : bf2f(f2bf(g));
        stf(out, kind, i, ldf(x, kind, i) + ldf(y, kind, i) * g);
    }
}

// ---- latent upscaler (frames-first x[T, C, S]: S = H*W) ----

// GroupNorm statistics: block (g * split + k) sums a strided share of group g over all T and S
// (double accumulators: single pass, no cancellation). Group g is channels [g*cg, (g+1)*cg).
extern "C" __global__ void gn_partial(const void* x, int kind, unsigned long long T, unsigned long long C,
                                      unsigned long long S, int groups, int split, double* part) {
    int g = blockIdx.x / split, k = blockIdx.x % split;
    unsigned long long cg = C / groups, per = cg * S, n = T * per;
    double s1 = 0.0, s2 = 0.0;
    for (unsigned long long e = (unsigned long long)k * blockDim.x + threadIdx.x; e < n;
         e += (unsigned long long)split * blockDim.x) {
        unsigned long long t = e / per, r = e % per;
        double v = (double)ldf(x, kind, t * C * S + (unsigned long long)g * cg * S + r);
        s1 += v;
        s2 += v * v;
    }
    __shared__ double sh1[256], sh2[256];
    sh1[threadIdx.x] = s1;
    sh2[threadIdx.x] = s2;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            sh1[threadIdx.x] += sh1[threadIdx.x + o];
            sh2[threadIdx.x] += sh2[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        part[2 * blockIdx.x] = sh1[0];
        part[2 * blockIdx.x + 1] = sh2[0];
    }
}

// mr[2g] = mean, mr[2g+1] = 1/sqrt(var + eps)
extern "C" __global__ void gn_final(const double* part, int groups, int split, double n, float eps, float* mr) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    double s1 = 0.0, s2 = 0.0;
    for (int k = 0; k < split; k++) {
        s1 += part[2 * (g * split + k)];
        s2 += part[2 * (g * split + k) + 1];
    }
    double mean = s1 / n, var = s2 / n - mean * mean;
    if (var < 0.0) var = 0.0;
    mr[2 * g] = (float)mean;
    mr[2 * g + 1] = (float)rsqrt(var + (double)eps);
}

// out = silu((x - mean) * rstd * gamma[c] + beta[c])
extern "C" __global__ void gn_silu(const void* x, int kind, unsigned long long T, unsigned long long C,
                                   unsigned long long S, int groups, const float* mr, const float* gamma,
                                   const float* beta, void* out) {
    unsigned long long n = T * C * S, cg = C / groups;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned long long c = (i / S) % C;
        int g = (int)(c / cg);
        float y = (ldf(x, kind, i) - mr[2 * g]) * mr[2 * g + 1] * gamma[c] + beta[c];
        stf(out, kind, i, y / (1.f + __expf(-y)));
    }
}

// depthwise conv over T (zero padded, K taps centered), per channel weights w[C, K] and bias b[C]
extern "C" __global__ void dw_time(const void* x, int kind, const float* w, const float* b, void* out,
                                   unsigned long long T, unsigned long long C, unsigned long long S, int K) {
    unsigned long long n = T * C * S;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned long long s = i % S, c = (i / S) % C;
        long long t = (long long)(i / (S * C));
        float acc = b[c];
        for (int k = 0; k < K; k++) {
            long long tt = t + k - K / 2;
            if (tt >= 0 && tt < (long long)T) acc += w[c * K + k] * ldf(x, kind, ((unsigned long long)tt * C + c) * S + s);
        }
        stf(out, kind, i, acc);
    }
}

// bilinear resize of planes x[N, H, W] -> [N, Ho, Wo], align_corners = false (PyTorch)
extern "C" __global__ void bilinear_hw(const void* x, int kind, void* out, unsigned long long N, int H, int W,
                                       int Ho, int Wo) {
    unsigned long long n = N * Ho * Wo;
    float sy = (float)H / (float)Ho, sx = (float)W / (float)Wo;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        int ox = (int)(i % Wo), oy = (int)((i / Wo) % Ho);
        unsigned long long p = i / ((unsigned long long)Wo * Ho);
        float fy = fmaxf(sy * ((float)oy + 0.5f) - 0.5f, 0.f), fx = fmaxf(sx * ((float)ox + 0.5f) - 0.5f, 0.f);
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float ly = fy - (float)y0, lx = fx - (float)x0;
        const unsigned long long base = p * (unsigned long long)H * W;
        float a = ldf(x, kind, base + (unsigned long long)y0 * W + x0), bb = ldf(x, kind, base + (unsigned long long)y0 * W + x1);
        float c = ldf(x, kind, base + (unsigned long long)y1 * W + x0), d = ldf(x, kind, base + (unsigned long long)y1 * W + x1);
        stf(out, kind, i, (a * (1.f - lx) + bb * lx) * (1.f - ly) + (c * (1.f - lx) + d * lx) * ly);
    }
}
"#;

fn ptx() -> Result<&'static str> {
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let r = PTX.get_or_init(|| {
        let opts = candle_core::cuda_backend::cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(false),
            options: vec!["--std=c++17".into()],
            ..Default::default()
        };
        candle_core::cuda_backend::cudarc::nvrtc::compile_ptx_with_opts(SRC, opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("nvrtc: {e:?}"))
    });
    r.as_deref().map_err(|e| candle_core::Error::Msg(e.clone()))
}

const MODULE: &str = "h3_kernels";

fn kind_of(dt: DType) -> Result<i32> {
    Ok(match dt {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        d => candle_core::bail!("unsupported kernel dtype {d:?}"),
    })
}

fn grid_for(n: usize) -> LaunchConfig {
    let threads = 256u32;
    let blocks = n.div_ceil(threads as usize).clamp(1, 65535 * 8) as u32;
    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

/// Raw device pointer (as u64) of a storage, honoring the layout start offset.
fn dev_ptr(st: &CudaStorage, l: &Layout) -> Result<u64> {
    use candle_core::cuda_backend::CudaStorageSlice as S;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let stream = st.device().cuda_stream();
    let off = l.start_offset();
    macro_rules! p {
        ($s:expr, $t:ty) => {{
            let (ptr, _g) = $s.device_ptr(&stream);
            ptr + (off * std::mem::size_of::<$t>()) as u64
        }};
    }
    Ok(match &st.slice {
        S::U8(s) => p!(s, u8),
        S::U32(s) => p!(s, u32),
        S::I64(s) => p!(s, i64),
        S::BF16(s) => p!(s, half::bf16),
        S::F16(s) => p!(s, half::f16),
        S::F32(s) => p!(s, f32),
        _ => candle_core::bail!("unsupported storage for raw pointer"),
    })
}

fn alloc_out(dev: &candle_core::CudaDevice, dt: DType, n: usize) -> Result<CudaStorage> {
    Ok(match dt {
        DType::F32 => CudaStorage::wrap_cuda_slice(unsafe { dev.alloc::<f32>(n)? }, dev.clone()),
        DType::F16 => CudaStorage::wrap_cuda_slice(unsafe { dev.alloc::<half::f16>(n)? }, dev.clone()),
        DType::BF16 => CudaStorage::wrap_cuda_slice(unsafe { dev.alloc::<half::bf16>(n)? }, dev.clone()),
        d => candle_core::bail!("alloc_out {d:?}"),
    })
}

fn out_ptr(st: &CudaStorage) -> Result<u64> {
    dev_ptr(st, &Layout::contiguous(1))
}

fn need_contig(l: &Layout, what: &str) -> Result<()> {
    if !l.is_contiguous() {
        candle_core::bail!("{what}: expected a contiguous tensor")
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------

struct DequantI8 {
    out: DType,
}
impl CustomOp2 for DequantI8 {
    fn name(&self) -> &'static str {
        "dequant_i8"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("dequant_i8: cuda only")
    }
    fn cuda_fwd(&self, q: &CudaStorage, ql: &Layout, s: &CudaStorage, sl: &Layout) -> Result<(CudaStorage, Shape)> {
        need_contig(ql, "dequant_i8 q")?;
        need_contig(sl, "dequant_i8 scale")?;
        let dims = ql.shape().dims2()?;
        let n = dims.0 * dims.1;
        let per_row = (sl.shape().elem_count() == dims.0 && dims.0 > 1) as i32;
        let dev = q.device().clone();
        let func = dev.get_or_load_custom_func("dequant_i8", MODULE, ptx()?)?;
        let out = alloc_out(&dev, self.out, n)?;
        let (qp, sp, op) = (dev_ptr(q, ql)?, dev_ptr(s, sl)?, out_ptr(&out)?);
        let (r, c, ok) = (dims.0 as u64, dims.1 as u64, kind_of(self.out)?);
        let mut b = func.builder();
        b.arg(&qp).arg(&sp).arg(&op).arg(&r).arg(&c).arg(&per_row).arg(&ok);
        unsafe { b.launch(grid_for(n)) }.w()?;
        Ok((out, ql.shape().clone()))
    }
}

/// int8 weight stored as U8 bytes `[N, K]` * f32 scale (`[N, 1]` or scalar) -> `[N, K]` of `out`.
pub fn dequant_i8(q: &Tensor, scale: &Tensor, out: DType) -> Result<Tensor> {
    q.apply_op2_no_bwd(&scale.to_dtype(DType::F32)?.contiguous()?, &DequantI8 { out })
}

struct DequantNvfp4 {
    out: DType,
    k: usize,
}
impl CustomOp3 for DequantNvfp4 {
    fn name(&self) -> &'static str {
        "dequant_nvfp4"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("dequant_nvfp4: cuda only")
    }
    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        bs: &CudaStorage,
        bl: &Layout,
        g: &CudaStorage,
        gl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        need_contig(ql, "nvfp4 q")?;
        need_contig(bl, "nvfp4 block scale")?;
        let rows = ql.shape().dims()[0];
        let n = rows * self.k;
        let dev = q.device().clone();
        let func = dev.get_or_load_custom_func("dequant_nvfp4", MODULE, ptx()?)?;
        let out = alloc_out(&dev, self.out, n)?;
        let (qp, bp, gp, op) = (dev_ptr(q, ql)?, dev_ptr(bs, bl)?, dev_ptr(g, gl)?, out_ptr(&out)?);
        let (r, k, ok) = (rows as u64, self.k as u64, kind_of(self.out)?);
        let mut b = func.builder();
        b.arg(&qp).arg(&bp).arg(&gp).arg(&op).arg(&r).arg(&k).arg(&ok);
        unsafe { b.launch(grid_for(n)) }.w()?;
        Ok((out, Shape::from((rows, self.k))))
    }
}

/// NVFP4 packed weight `[N, K/2]` (U8), swizzled fp8 block scales (U8 bytes), f32 global scale.
pub fn dequant_nvfp4(q: &Tensor, block_scale: &Tensor, global: &Tensor, out: DType) -> Result<Tensor> {
    let k = q.dim(1)? * 2;
    q.apply_op3_no_bwd(block_scale, &global.to_dtype(DType::F32)?.reshape(1)?, &DequantNvfp4 { out, k })
}

struct QkNormRope {
    heads: usize,
    head_dim: usize,
    row_stride: usize,
    col_off: usize,
    head_stride: usize,
    rot: usize,
    eps: f32,
    do_norm: bool,
    weight: Option<Tensor>,
    out: DType,
}
impl CustomOp2 for QkNormRope {
    fn name(&self) -> &'static str {
        "qk_norm_rope"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("qk_norm_rope: cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, cs: &CudaStorage, csl: &Layout) -> Result<(CudaStorage, Shape)> {
        need_contig(csl, "rope table")?;
        let s = xl.shape().dims()[0];
        let dev = x.device().clone();
        let func = dev.get_or_load_custom_func("qk_norm_rope", MODULE, ptx()?)?;
        let n = s * self.heads * self.head_dim;
        let out = alloc_out(&dev, self.out, n)?;
        let (wp, wk, has_w) = match &self.weight {
            Some(w) => {
                let (st, wl) = w.storage_and_layout();
                let p = match &*st {
                    Storage::Cuda(c) => dev_ptr(c, wl)?,
                    _ => candle_core::bail!("weight not on cuda"),
                };
                (p, kind_of(w.dtype())?, 1i32)
            }
            None => (0u64, 0i32, 0i32),
        };
        let (xp, cp, op) = (dev_ptr(x, xl)?, dev_ptr(cs, csl)?, out_ptr(&out)?);
        let xk = kind_of(x.dtype())?;
        let (rs, co, hs) = (self.row_stride as u64, self.col_off as u64, self.head_stride as u64);
        let (dn, eps, rot, ok) = (self.do_norm as i32, self.eps, self.rot as i32, kind_of(self.out)?);
        let (si, hi, di) = (s as i32, self.heads as i32, self.head_dim as i32);
        let threads = self.head_dim.next_multiple_of(32) as u32;
        let cfg = LaunchConfig { grid_dim: ((s * self.heads) as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let mut b = func.builder();
        b.arg(&xp).arg(&xk).arg(&rs).arg(&co).arg(&hs).arg(&wp).arg(&wk).arg(&has_w).arg(&dn).arg(&eps).arg(&cp).arg(&rot);
        b.arg(&op).arg(&ok).arg(&si).arg(&hi).arg(&di);
        unsafe { b.launch(cfg) }.w()?;
        Ok((out, Shape::from((s, self.heads, self.head_dim))))
    }
}

/// Rope tables for [`qk_norm_rope`]: `angles` is `[S, rot/2]` (f32) -> `[2, S, rot/2]` cos/sin.
pub fn rope_table(angles: &Tensor) -> Result<Tensor> {
    let a = angles.to_dtype(DType::F32)?;
    Tensor::stack(&[a.cos()?, a.sin()?], 0)?.contiguous()
}

/// Per-head RMSNorm + split-half rope reading heads straight out of a fused projection.
///
/// `x` is a 2-D `[S, row_width]` tensor whose rows hold this tensor's `heads * head_dim` values
/// starting at column `col_off` (e.g. the q part of a fused qkv). Returns contiguous `[S, H, D]`.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(
    x: &Tensor,
    col_off: usize,
    heads: usize,
    head_dim: usize,
    norm: Option<(Option<&Tensor>, f32)>,
    table: Option<&Tensor>,
    rot: usize,
    out: DType,
) -> Result<Tensor> {
    qk_norm_rope_strided(x, col_off, head_dim, heads, head_dim, norm, table, rot, out)
}

/// [`qk_norm_rope`] with an explicit per-head column stride (for per-head interleaved qkv).
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_strided(
    x: &Tensor,
    col_off: usize,
    head_stride: usize,
    heads: usize,
    head_dim: usize,
    norm: Option<(Option<&Tensor>, f32)>,
    table: Option<&Tensor>,
    rot: usize,
    out: DType,
) -> Result<Tensor> {
    let (_, width) = x.dims2()?;
    if x.stride()[1] != 1 || x.stride()[0] != width {
        candle_core::bail!("qk_norm_rope expects a row-contiguous 2-D input");
    }
    let dummy;
    let table = match table {
        Some(t) => t,
        None => {
            dummy = Tensor::zeros((2, 1), DType::F32, x.device())?;
            &dummy
        }
    };
    let (do_norm, weight, eps) = match norm {
        Some((w, eps)) => (true, w.cloned(), eps),
        None => (false, None, 0.0),
    };
    let op = QkNormRope {
        heads,
        head_dim,
        row_stride: width,
        col_off,
        head_stride,
        rot: if table.dims()[0] == 2 && table.rank() == 3 { rot } else { 0 },
        eps,
        do_norm,
        weight,
        out,
    };
    x.apply_op2_no_bwd(table, &op)
}

struct Modulate {
    shift: Tensor,
    idx: Tensor,
}
impl CustomOp2 for Modulate {
    fn name(&self) -> &'static str {
        "modulate"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("modulate: cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, sc: &CudaStorage, scl: &Layout) -> Result<(CudaStorage, Shape)> {
        need_contig(xl, "modulate x")?;
        need_contig(scl, "modulate scale")?;
        let (s, c) = xl.shape().dims2()?;
        let dev = x.device().clone();
        let func = dev.get_or_load_custom_func("modulate", MODULE, ptx()?)?;
        let out = alloc_out(&dev, x.dtype(), s * c)?;
        let shp = cuda_ptr_of(&self.shift)?;
        let ip = cuda_ptr_of(&self.idx)?;
        let (xp, sp, op) = (dev_ptr(x, xl)?, dev_ptr(sc, scl)?, out_ptr(&out)?);
        let k = kind_of(x.dtype())?;
        let (su, cu) = (s as u64, c as u64);
        let mut b = func.builder();
        b.arg(&xp).arg(&k).arg(&sp).arg(&shp).arg(&ip).arg(&op).arg(&su).arg(&cu);
        unsafe { b.launch(grid_for(s * c)) }.w()?;
        Ok((out, xl.shape().clone()))
    }
}

fn cuda_ptr_of(t: &Tensor) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    match &*st {
        Storage::Cuda(c) => {
            need_contig(l, "table")?;
            dev_ptr(c, l)
        }
        _ => candle_core::bail!("expected a cuda tensor"),
    }
}

/// `x * (1 + scale[idx]) + shift[idx]` with f32 tables `[R, C]` and u32 per-token `idx` `[S]`.
pub fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let op = Modulate { shift: shift.contiguous()?, idx: idx.contiguous()? };
    x.contiguous()?.apply_op2_no_bwd(&scale.contiguous()?, &op)
}

struct GateAdd {
    gate: Tensor,
    idx: Tensor,
}
impl CustomOp2 for GateAdd {
    fn name(&self) -> &'static str {
        "gate_add"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("gate_add: cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, y: &CudaStorage, yl: &Layout) -> Result<(CudaStorage, Shape)> {
        need_contig(xl, "gate_add x")?;
        need_contig(yl, "gate_add y")?;
        let (s, c) = xl.shape().dims2()?;
        let dev = x.device().clone();
        let func = dev.get_or_load_custom_func("gate_add", MODULE, ptx()?)?;
        let out = alloc_out(&dev, x.dtype(), s * c)?;
        let gp = cuda_ptr_of(&self.gate)?;
        let ip = cuda_ptr_of(&self.idx)?;
        let (xp, yp, op) = (dev_ptr(x, xl)?, dev_ptr(y, yl)?, out_ptr(&out)?);
        let k = kind_of(x.dtype())?;
        let (su, cu) = (s as u64, c as u64);
        let mut b = func.builder();
        b.arg(&xp).arg(&yp).arg(&k).arg(&gp).arg(&ip).arg(&op).arg(&su).arg(&cu);
        unsafe { b.launch(grid_for(s * c)) }.w()?;
        Ok((out, xl.shape().clone()))
    }
}

/// `x + y * gate[idx]` (gated residual accumulate).
pub fn gate_add(x: &Tensor, y: &Tensor, gate: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let op = GateAdd { gate: gate.contiguous()?, idx: idx.contiguous()? };
    x.contiguous()?.apply_op2_no_bwd(&y.to_dtype(x.dtype())?.contiguous()?, &op)
}

struct Act1d {
    fu: Tensor,
    fd: Tensor,
    la: Tensor,
    lb: Tensor,
}
impl CustomOp1 for Act1d {
    fn name(&self) -> &'static str {
        "act1d"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("act1d: cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        need_contig(xl, "act1d x")?;
        let (b, c, l) = xl.shape().dims3()?;
        let dev = x.device().clone();
        let func = dev.get_or_load_custom_func("act1d", MODULE, ptx()?)?;
        let n = b * c * l;
        let out = alloc_out(&dev, DType::F32, n)?;
        let (xp, op) = (dev_ptr(x, xl)?, out_ptr(&out)?);
        let (fu, fd, la, lb) = (cuda_ptr_of(&self.fu)?, cuda_ptr_of(&self.fd)?, cuda_ptr_of(&self.la)?, cuda_ptr_of(&self.lb)?);
        let (ci, li, ri) = (c as i32, l as i32, (b * c) as i32);
        let mut bld = func.builder();
        bld.arg(&xp).arg(&fu).arg(&fd).arg(&la).arg(&lb).arg(&op).arg(&ci).arg(&li).arg(&ri);
        unsafe { bld.launch(grid_for(n)) }.w()?;
        Ok((out, xl.shape().clone()))
    }
}

/// BigVGAN `Activation1d(SnakeBeta)`: `[B, C, L]` f32, filters `[12]`, log alpha/beta `[C]`.
pub fn act1d(x: &Tensor, up_filter: &Tensor, down_filter: &Tensor, log_alpha: &Tensor, log_beta: &Tensor) -> Result<Tensor> {
    let f = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.contiguous();
    let op = Act1d { fu: f(up_filter)?, fd: f(down_filter)?, la: f(log_alpha)?, lb: f(log_beta)? };
    x.to_dtype(DType::F32)?.contiguous()?.apply_op1_no_bwd(&op)
}

// ---------------------------------------------------------------------------------------------
// int8 x int8 -> int32 GEMM (cuBLASLt IMMA), dynamic per-row activation quantization

mod lt {
    use candle_core::cuda_backend::cudarc::cublaslt::{result, sys};
    use std::cell::RefCell;

    pub struct Plan {
        pub desc: sys::cublasLtMatmulDesc_t,
        pub la: sys::cublasLtMatrixLayout_t,
        pub lb: sys::cublasLtMatrixLayout_t,
        pub lc: sys::cublasLtMatrixLayout_t,
        pub algo: sys::cublasLtMatmulAlgo_t,
    }

    pub struct Handle {
        pub h: sys::cublasLtHandle_t,
        pub ws: candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
        /// (m, n, k) -> descriptors + chosen algorithm (heuristics are slow, shapes repeat)
        pub plans: std::collections::HashMap<(usize, usize, usize), Plan>,
    }

    thread_local!(pub static H: RefCell<Option<Handle>> = const { RefCell::new(None) });

    pub fn sys_err(e: impl std::fmt::Debug) -> candle_core::Error {
        candle_core::Error::Msg(format!("cublasLt: {e:?}"))
    }
    pub use result::*;
    pub use sys::*;
}

pub const LT_WORKSPACE: usize = 32 << 20;

struct Int8Linear {
    ws: Tensor,
    bias: Option<Tensor>,
    out: DType,
    swiglu: bool,
    /// fused ConvRot rotation of the activations (group 256)
    rotate: bool,
}

impl CustomOp2 for Int8Linear {
    fn name(&self) -> &'static str {
        "int8_linear"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("int8_linear: cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        need_contig(xl, "int8_linear x")?;
        need_contig(wl, "int8_linear w")?;
        let (m, k) = xl.shape().dims2()?;
        let (n, k2) = wl.shape().dims2()?;
        if k != k2 || k % 4 != 0 {
            candle_core::bail!("int8_linear: bad shapes {m}x{k} @ {n}x{k2}");
        }
        let dev = x.device().clone();
        let stream = dev.cuda_stream();
        // 1. quantize rows
        let q = unsafe { dev.alloc::<u8>(m * k)? };
        let xs = unsafe { dev.alloc::<f32>(m)? };
        let rot = self.rotate && k % 256 == 0 && k / 256 <= 64;
        let quant = dev.get_or_load_custom_func(if rot { "quant_rows_i8_rot" } else { "quant_rows_i8" }, MODULE, ptx()?)?;
        let (qp, _g1) = q.device_ptr(&stream);
        let (sp, _g2) = xs.device_ptr(&stream);
        let xp = dev_ptr(x, xl)?;
        let (xk, ki) = (kind_of(x.dtype())?, k as i32);
        let mut b = quant.builder();
        b.arg(&xp).arg(&xk).arg(&qp).arg(&sp).arg(&ki);
        unsafe { b.launch(LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        // 2. GEMM: D[n, m] (col-major) = W^T(op T on [k, n]) * Xq([k, m])
        let acc = unsafe { dev.alloc::<i32>(m * n)? };
        let (ap, _g3) = acc.device_ptr(&stream);
        let wp = dev_ptr(w, wl)?;
        lt::H.with(|cell| -> Result<()> {
            let mut c = cell.borrow_mut();
            if c.is_none() {
                let h = lt::create_handle().map_err(lt::sys_err)?;
                let ws = unsafe { dev.alloc::<u8>(LT_WORKSPACE)? };
                *c = Some(lt::Handle { h, ws, plans: Default::default() });
            }
            let hd = c.as_mut().unwrap();
            unsafe {
                use lt::*;
                if !hd.plans.contains_key(&(m, n, k)) {
                let desc = create_matmul_desc(cublasComputeType_t::CUBLAS_COMPUTE_32I, cudaDataType::CUDA_R_32I).map_err(sys_err)?;
                let opt = candle_core::cuda_backend::cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
                set_matmul_desc_attribute(desc, cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &opt as *const _ as *const _, std::mem::size_of_val(&opt)).map_err(sys_err)?;
                let la = create_matrix_layout(cudaDataType::CUDA_R_8I, k as u64, n as u64, k as i64).map_err(sys_err)?;
                let lb = create_matrix_layout(cudaDataType::CUDA_R_8I, k as u64, m as u64, k as i64).map_err(sys_err)?;
                let lc = create_matrix_layout(cudaDataType::CUDA_R_32I, n as u64, m as u64, n as i64).map_err(sys_err)?;
                let pref = create_matmul_pref().map_err(sys_err)?;
                let wsz = LT_WORKSPACE as u64;
                set_matmul_pref_attribute(pref, cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz as *const _ as *const _, 8).map_err(sys_err)?;
                let heur = get_matmul_algo_heuristic(hd.h, desc, la, lb, lc, lc, pref).map_err(sys_err)?;
                let _ = destroy_matmul_pref(pref);
                hd.plans.insert((m, n, k), Plan { desc, la, lb, lc, algo: heur.algo });
                }
                let Plan { desc, la, lb, lc, algo } = &hd.plans[&(m, n, k)];
                let (desc, la, lb, lc) = (*desc, *la, *lb, *lc);
                let (alpha, beta) = (1i32, 0i32);
                let (wsp, _g4) = hd.ws.device_ptr(&stream);
                let r = matmul(
                    hd.h,
                    desc,
                    &alpha as *const _ as *const _,
                    &beta as *const _ as *const _,
                    wp as *const _,
                    la,
                    qp as *const _,
                    lb,
                    ap as *const _,
                    lc,
                    ap as *mut _,
                    lc,
                    algo,
                    wsp as *mut _,
                    LT_WORKSPACE,
                    stream.cu_stream() as _,
                );
                r.map_err(sys_err)?;
            }
            Ok(())
        })?;
        // 3. scale + bias (+ swiglu) -> out
        let n_out = if self.swiglu { n / 2 } else { n };
        let out = alloc_out(&dev, self.out, m * n_out)?;
        let op = out_ptr(&out)?;
        let ep = dev.get_or_load_custom_func(if self.swiglu { "int8_epilogue_swiglu" } else { "int8_epilogue" }, MODULE, ptx()?)?;
        let wsp = cuda_ptr_of(&self.ws)?;
        let per_row = (self.ws.elem_count() == n && n > 1) as i32;
        let (bp, hb) = match &self.bias {
            Some(t) => (cuda_ptr_of(t)?, 1i32),
            None => (0u64, 0i32),
        };
        let (mu, nu, ok) = (m as u64, n_out as u64, kind_of(self.out)?);
        let mut b = ep.builder();
        b.arg(&ap).arg(&sp).arg(&wsp).arg(&per_row).arg(&bp).arg(&hb).arg(&op).arg(&ok).arg(&mu).arg(&nu);
        if n_out % 4 != 0 {
            candle_core::bail!("int8_linear: output width must be a multiple of 4");
        }
        let cfg = LaunchConfig { grid_dim: ((n_out / 4).div_ceil(256) as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(cfg) }.w()?;
        drop((_g1, _g2, _g3));
        Ok((out, Shape::from((m, n_out))))
    }
}

/// `x @ dequant(w)^T + bias` on int8 tensor cores: x `[M, K]` is quantized per row on the fly,
/// `w` is the stored int8 weight `[N, K]` (U8 bytes) with f32 `scale` (`[N, 1]` or scalar).
pub fn int8_linear(x: &Tensor, w: &Tensor, scale: &Tensor, bias: Option<&Tensor>, out: DType) -> Result<Tensor> {
    int8_linear_ex(x, w, scale, bias, out, false, false)
}

/// Whether [`int8_linear_ex`] can fuse the ConvRot rotation for an input width.
/// Off by default: the block-per-row butterfly is slower than the separate rotation GEMM on
/// RTX 4090 (`H3_FUSED_ROT=1` to try it).
pub fn can_fuse_rotation(k: usize, group: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("H3_FUSED_ROT").is_ok_and(|v| v == "1")) && group == 256 && k % 256 == 0 && k / 256 <= 64
}

/// [`int8_linear`], optionally rotating activations (ConvRot, group 256) inside the quantizer and
/// following with a fused `silu(first half) * second half`.
pub fn int8_linear_ex(x: &Tensor, w: &Tensor, scale: &Tensor, bias: Option<&Tensor>, out: DType, swiglu: bool, rotate: bool) -> Result<Tensor> {
    let op = Int8Linear {
        rotate,
        swiglu,
        ws: scale.to_dtype(DType::F32)?.flatten_all()?.contiguous()?,
        bias: bias.map(|b| b.to_dtype(DType::F32).and_then(|b| b.contiguous())).transpose()?,
        out,
    };
    x.contiguous()?.apply_op2_no_bwd(&w.contiguous()?, &op)
}

/// Keep freed device memory in the stream-ordered pool instead of returning it to the driver on
/// every synchronization (the default threshold of 0 makes large per-layer buffers re-allocate).
pub fn keep_pool_memory(dev: &candle_core::Device) {
    use candle_core::cuda_backend::cudarc::driver::sys;
    if let candle_core::Device::Cuda(c) = dev {
        let ctx = c.cuda_stream().context().clone();
        unsafe {
            let mut pool = std::mem::MaybeUninit::uninit();
            if sys::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), ctx.cu_device()) == sys::CUresult::CUDA_SUCCESS {
                let mut thr: u64 = u64::MAX;
                let _ = sys::cuMemPoolSetAttribute(pool.assume_init(), sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &mut thr as *mut u64 as *mut _);
            }
        }
    }
}

/// Return pooled-but-free device memory to the driver (before measuring free VRAM).
pub fn trim_pool() {
    use candle_core::cuda_backend::cudarc::driver::sys;
    unsafe {
        let mut dev = 0;
        if sys::cuCtxGetDevice(&mut dev) != sys::CUresult::CUDA_SUCCESS {
            return;
        }
        let _ = sys::cuCtxSynchronize();
        let mut pool = std::mem::MaybeUninit::uninit();
        if sys::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), dev) == sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuMemPoolTrimTo(pool.assume_init(), 0);
        }
    }
}

/// Quantize a row-major `[m, k]` activation (raw device pointer, dtype `kind`) to int8 rows + f32
/// per-row scales, optionally with the fused ConvRot rotation.
pub(crate) fn quant_rows_raw(
    dev: &candle_core::CudaDevice,
    x_ptr: u64,
    kind_code: i32,
    m: usize,
    k: usize,
    rotate: bool,
) -> Result<(candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>, candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>)> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let stream = dev.cuda_stream();
    let q = unsafe { dev.alloc::<u8>(m * k)? };
    let xs = unsafe { dev.alloc::<f32>(m)? };
    let rot2 = rotate && kind_code == 2 && k % 256 == 0 && k / 256 <= 56;
    let name = if rot2 { "quant_rows_i8_rot2" } else if rotate { "quant_rows_i8_rot" } else { "quant_rows_i8" };
    let f = dev.get_or_load_custom_func(name, MODULE, ptx()?)?;
    {
        let (qp, _a) = q.device_ptr(&stream);
        let (sp, _b) = xs.device_ptr(&stream);
        let ki = k as i32;
        let mut b = f.builder();
        if rot2 {
            b.arg(&x_ptr).arg(&qp).arg(&sp).arg(&ki);
        } else {
            b.arg(&x_ptr).arg(&kind_code).arg(&qp).arg(&sp).arg(&ki);
        }
        unsafe { b.launch(LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
    }
    Ok((q, xs))
}

pub(crate) fn raw_ptr(st: &CudaStorage, l: &Layout) -> Result<u64> {
    dev_ptr(st, l)
}

pub(crate) fn kind_code(dt: DType) -> Result<i32> {
    kind_of(dt)
}

struct RmsModulate {
    w: Tensor,
    eps: f32,
    shift: Tensor,
    idx: Tensor,
}
impl CustomOp2 for RmsModulate {
    fn name(&self) -> &'static str {
        "rms_modulate"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("rms_modulate: cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, sc: &CudaStorage, scl: &Layout) -> Result<(CudaStorage, Shape)> {
        need_contig(xl, "rms_modulate x")?;
        let (s, c) = xl.shape().dims2()?;
        let dev = x.device().clone();
        let f = dev.get_or_load_custom_func("rms_modulate", MODULE, ptx()?)?;
        let out = alloc_out(&dev, x.dtype(), s * c)?;
        let (xp, sp, op) = (dev_ptr(x, xl)?, dev_ptr(sc, scl)?, out_ptr(&out)?);
        let (wp, shp, ip) = (cuda_ptr_of(&self.w)?, cuda_ptr_of(&self.shift)?, cuda_ptr_of(&self.idx)?);
        let (k, ci, eps) = (kind_of(x.dtype())?, c as i32, self.eps);
        let mut b = f.builder();
        b.arg(&xp).arg(&k).arg(&wp).arg(&eps).arg(&sp).arg(&shp).arg(&ip).arg(&op).arg(&ci);
        unsafe { b.launch(LaunchConfig { grid_dim: (s as u32, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        Ok((out, xl.shape().clone()))
    }
}

/// `modulate(rms_norm(x, w, eps), scale, shift, idx)` in one pass.
pub fn rms_modulate(x: &Tensor, w: &Tensor, eps: f32, scale: &Tensor, shift: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let op = RmsModulate { w: w.to_dtype(x.dtype())?.contiguous()?, eps, shift: shift.contiguous()?, idx: idx.contiguous()? };
    x.contiguous()?.apply_op2_no_bwd(&scale.contiguous()?, &op)
}

// ---------------------------------------------------------------------------------------------
// latent upscaler kernels (see `upscaler.rs`); tensors are frames-first `[T, C, H, W]`.

const GN_SPLIT: usize = 64;

struct GroupNormSilu {
    groups: usize,
    eps: f32,
}
impl CustomOp3 for GroupNormSilu {
    fn name(&self) -> &'static str {
        "group_norm_silu"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("group_norm_silu: cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, g: &CudaStorage, gl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        need_contig(xl, "group_norm x")?;
        need_contig(gl, "group_norm gamma")?;
        need_contig(bl, "group_norm beta")?;
        let (t, c, h, w) = xl.shape().dims4()?;
        let s = h * w;
        if c % self.groups != 0 {
            candle_core::bail!("group_norm: {c} channels not divisible by {} groups", self.groups);
        }
        let dev = x.device().clone();
        let n = t * c * s;
        let part = unsafe { dev.alloc::<f64>(2 * self.groups * GN_SPLIT)? };
        let mr = unsafe { dev.alloc::<f32>(2 * self.groups)? };
        let out = alloc_out(&dev, x.dtype(), n)?;
        let stream = dev.cuda_stream();
        let (pp, mp) = (part.device_ptr(&stream).0, mr.device_ptr(&stream).0);
        let (xp, gp, bp, op) = (dev_ptr(x, xl)?, dev_ptr(g, gl)?, dev_ptr(b, bl)?, out_ptr(&out)?);
        let (tu, cu, su, grp, split, k) = (t as u64, c as u64, s as u64, self.groups as i32, GN_SPLIT as i32, kind_of(x.dtype())?);
        let f_part = dev.get_or_load_custom_func("gn_partial", MODULE, ptx()?)?;
        let mut bd = f_part.builder();
        bd.arg(&xp).arg(&k).arg(&tu).arg(&cu).arg(&su).arg(&grp).arg(&split).arg(&pp);
        let cfg = LaunchConfig { grid_dim: ((self.groups * GN_SPLIT) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { bd.launch(cfg) }.w()?;
        let f_fin = dev.get_or_load_custom_func("gn_final", MODULE, ptx()?)?;
        let per = (t * (c / self.groups) * s) as f64;
        let mut bd = f_fin.builder();
        bd.arg(&pp).arg(&grp).arg(&split).arg(&per).arg(&self.eps).arg(&mp);
        unsafe { bd.launch(grid_for(self.groups)) }.w()?;
        let f_app = dev.get_or_load_custom_func("gn_silu", MODULE, ptx()?)?;
        let mut bd = f_app.builder();
        bd.arg(&xp).arg(&k).arg(&tu).arg(&cu).arg(&su).arg(&grp).arg(&mp).arg(&gp).arg(&bp).arg(&op);
        unsafe { bd.launch(grid_for(n)) }.w()?;
        Ok((out, xl.shape().clone()))
    }
}

/// `silu(group_norm(x) * gamma + beta)` on frames-first `x [T, C, H, W]`: `groups` groups of
/// contiguous channels, statistics over all frames and pixels of a group (3D GroupNorm, as in the
/// reference's `nn.GroupNorm` on `[1, C, T, H, W]`). `gamma`/`beta` are f32 `[C]`.
pub fn group_norm_silu(x: &Tensor, gamma: &Tensor, beta: &Tensor, groups: usize, eps: f32) -> Result<Tensor> {
    x.contiguous()?.apply_op3_no_bwd(&gamma.contiguous()?, &beta.contiguous()?, &GroupNormSilu { groups, eps })
}

struct DwTime;
impl CustomOp3 for DwTime {
    fn name(&self) -> &'static str {
        "dw_time"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("dw_time: cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
        need_contig(xl, "dw_time x")?;
        need_contig(wl, "dw_time w")?;
        need_contig(bl, "dw_time b")?;
        let (t, c, h, wd) = xl.shape().dims4()?;
        let k = wl.shape().dims2()?.1;
        let dev = x.device().clone();
        let n = t * c * h * wd;
        let func = dev.get_or_load_custom_func("dw_time", MODULE, ptx()?)?;
        let out = alloc_out(&dev, x.dtype(), n)?;
        let (xp, wp, bp, op) = (dev_ptr(x, xl)?, dev_ptr(w, wl)?, dev_ptr(b, bl)?, out_ptr(&out)?);
        let (kind, tu, cu, su, ki) = (kind_of(x.dtype())?, t as u64, c as u64, (h * wd) as u64, k as i32);
        let mut bd = func.builder();
        bd.arg(&xp).arg(&kind).arg(&wp).arg(&bp).arg(&op).arg(&tu).arg(&cu).arg(&su).arg(&ki);
        unsafe { bd.launch(grid_for(n)) }.w()?;
        Ok((out, xl.shape().clone()))
    }
}

/// Depthwise conv over the frame axis of `x [T, C, H, W]`: `w` f32 `[C, K]`, `b` f32 `[C]`, zero padded.
pub fn dw_time(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.contiguous()?.apply_op3_no_bwd(&w.contiguous()?, &b.contiguous()?, &DwTime)
}

struct BilinearHw {
    ho: usize,
    wo: usize,
}
impl CustomOp1 for BilinearHw {
    fn name(&self) -> &'static str {
        "bilinear_hw"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("bilinear_hw: cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        need_contig(xl, "bilinear x")?;
        let (t, c, h, w) = xl.shape().dims4()?;
        let dev = x.device().clone();
        let n = t * c * self.ho * self.wo;
        let func = dev.get_or_load_custom_func("bilinear_hw", MODULE, ptx()?)?;
        let out = alloc_out(&dev, x.dtype(), n)?;
        let (xp, op) = (dev_ptr(x, xl)?, out_ptr(&out)?);
        let (kind, planes, hi, wi, ho, wo) = (kind_of(x.dtype())?, (t * c) as u64, h as i32, w as i32, self.ho as i32, self.wo as i32);
        let mut bd = func.builder();
        bd.arg(&xp).arg(&kind).arg(&op).arg(&planes).arg(&hi).arg(&wi).arg(&ho).arg(&wo);
        unsafe { bd.launch(grid_for(n)) }.w()?;
        Ok((out, Shape::from((t, c, self.ho, self.wo))))
    }
}

/// Bilinear resize (PyTorch `align_corners=False`) of the last two dims of `x [T, C, H, W]`.
pub fn bilinear_hw(x: &Tensor, ho: usize, wo: usize) -> Result<Tensor> {
    x.contiguous()?.apply_op1_no_bwd(&BilinearHw { ho, wo })
}
