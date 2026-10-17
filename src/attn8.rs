//! SageAttention-style attention for head_dim 128 on sm_80+: Q·Kᵀ on int8 tensor cores (per-token
//! int8 Q and K, K smoothed by its per-channel mean), online softmax in registers, P·V in fp16 with
//! fp32 accumulation. Non-causal, no mask. Opt out with `H3_ATTN=flash`.

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, WrapErr};
use candle_core::{CpuStorage, CustomOp1, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor, D};

const SRC: &str = r#"
typedef unsigned short u16;
typedef unsigned int u32;
typedef signed char s8;

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#define BM 64
#define BN 64
#define HD 128
#define KPAD 128
#define VPAD 128

__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u16 f2bf(float f) {
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
__device__ __forceinline__ u32 smem_addr(const void* p) {
    u32 a; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p)); return a;
}
__device__ __forceinline__ void cp16(void* s, const void* g, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_addr(s)), "l"(g), "r"(bytes));
}
__device__ __forceinline__ u32 pack_h2(float lo, float hi) {
    u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// Column sums of x [S, C] (bf16) into acc [C] (f32, pre-zeroed); grid.y splits the rows.
extern "C" __global__ void col_sum(const u16* x, float* acc, int S, int C, int rows_per) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int r0 = blockIdx.y * rows_per, r1 = min(S, r0 + rows_per);
    float s = 0.f;
    for (int r = r0; r < r1; ++r) s += bf2f(x[(size_t)r * C + c]);
    atomicAdd(&acc[c], s);
}

// [S, H, 128] bf16 (row stride `row_stride` elements, column offset `col_off`) -> [H, S, 128] f16
extern "C" __global__ void v_to_f16(const u16* x, unsigned long long row_stride, unsigned long long col_off, u16* out, int S, int H) {
    unsigned long long n = (unsigned long long)S * H * HD / 2;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned long long e = i * 2;
        int d = e % HD, sh = e / HD, h = sh % H, s = sh / H;
        u32 two = *(const u32*)(x + (size_t)s * row_stride + col_off + (size_t)h * HD + d);
        float a = bf2f(two & 0xffff), b = bf2f(two >> 16);
        u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a));
        *(u32*)(out + ((size_t)h * S + s) * HD + d) = r;
    }
}

// One warp per (token, head) row: subtract optional per-(head, dim) mean, per-row int8 quantization.
// x: [S, H, 128] (bf16), out: [H, S, 128] int8, scale: [H, S]
extern "C" __global__ void quant_heads(const u16* x, const float* mean, s8* out, float* scale, int S, int H) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= S * H) return;
    int lane = threadIdx.x & 31;
    int s = row / H, h = row % H;
    const u16* xr = x + (size_t)row * HD;
    float v[4];
    float m = 0.f;
    for (int i = 0; i < 4; ++i) {
        int d = lane * 4 + i;
        v[i] = bf2f(xr[d]) - (mean ? mean[h * HD + d] : 0.f);
        m = fmaxf(m, fabsf(v[i]));
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    float sc = fmaxf(m / 127.f, 1e-20f);
    float inv = 1.f / sc;
    u32 packed = 0;
    for (int i = 0; i < 4; ++i) {
        int q = __float2int_rn(v[i] * inv);
        q = max(-127, min(127, q));
        packed |= ((u32)(q & 0xff)) << (8 * i);
    }
    *(u32*)(out + ((size_t)h * S + s) * HD + lane * 4) = packed;
    if (lane == 0) scale[h * S + s] = sc;
}

__device__ __forceinline__ void mma_s8(int* c, const u32* a, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float* c, const u32* a, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// q8/k8: [H, S, 128] int8, qs/ks: [H, S], v: [H, S, 128] f16, out: [S, H, 128] bf16.
extern "C" __global__ void __launch_bounds__(128) attn_qk8(const s8* __restrict__ q8, const float* __restrict__ qs,
        const s8* __restrict__ k8, const float* __restrict__ ks, const u16* __restrict__ v, u16* __restrict__ out,
        int S, int H, float scale_log2) {
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char (*Ks)[BN * KPAD] = (unsigned char (*)[BN * KPAD])smem;
    u16 (*Vs)[BN * VPAD] = (u16 (*)[BN * VPAD])(smem + 2 * BN * KPAD);
    float (*Kscl)[BN] = (float (*)[BN])(smem + 2 * BN * KPAD + 2 * BN * VPAD * 2);
    const int h = blockIdx.y, qb = blockIdx.x * BM;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int r0 = qb + warp * 16 + g, r1 = r0 + 8;
    const s8* kh = k8 + (size_t)h * S * HD;
    const u16* vh = v + (size_t)h * S * HD;

    u32 qa[4][4];
    for (int k = 0; k < 4; ++k) {
        int col = k * 32 + t * 4;
        const s8* q0 = q8 + ((size_t)h * S + min(r0, S - 1)) * HD;
        const s8* q1 = q8 + ((size_t)h * S + min(r1, S - 1)) * HD;
        qa[k][0] = r0 < S ? *(const u32*)(q0 + col) : 0u;
        qa[k][1] = r1 < S ? *(const u32*)(q1 + col) : 0u;
        qa[k][2] = r0 < S ? *(const u32*)(q0 + col + 16) : 0u;
        qa[k][3] = r1 < S ? *(const u32*)(q1 + col + 16) : 0u;
    }
    const float qs0 = r0 < S ? qs[h * S + r0] * scale_log2 : 0.f;
    const float qs1 = r1 < S ? qs[h * S + r1] * scale_log2 : 0.f;

    float o[16][4];
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const int nt = (S + BN - 1) / BN;

    auto load = [&](int buf, int tile) {
        for (int c = tid; c < BN * 8; c += 128) {
            int row = c >> 3, ch = c & 7, key = tile * BN + row;
            cp16(&Ks[buf][row * KPAD + ((ch ^ (row & 7)) * 16)], kh + (size_t)min(key, S - 1) * HD + ch * 16, key < S ? 16 : 0);
        }
        for (int c = tid; c < BN * 16; c += 128) {
            int row = c >> 4, ch = c & 15, key = tile * BN + row;
            cp16(&Vs[buf][row * VPAD + ((ch ^ (row & 7)) * 8)], vh + (size_t)min(key, S - 1) * HD + ch * 8, key < S ? 16 : 0);
        }
        if (tid < BN) {
            int key = tile * BN + tid;
            Kscl[buf][tid] = key < S ? ks[h * S + key] : 0.f;
        }
        asm volatile("cp.async.commit_group;");
    };

    load(0, 0);
    for (int tile = 0; tile < nt; ++tile) {
        const int buf = tile & 1;
        if (tile + 1 < nt) {
            load(buf ^ 1, tile + 1);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        int sc[8][4];
        for (int j = 0; j < 8; ++j) {
            sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0;
            // ldmatrix (b16 view of int8): matrix i = rows (keys) j*8.., 16-byte chunk c0 + i,
            // which is exactly the m16n8k32 B fragment (n = key, 4 consecutive k bytes per thread)
            const int key = j * 8 + (lane & 7);
            for (int k = 0; k < 4; k += 2) {
                const int chunk = 2 * k + (lane >> 3);
                u32 b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                             : "r"(smem_addr(&Ks[buf][key * KPAD + ((chunk ^ (key & 7)) * 16)])));
                mma_s8(sc[j], qa[k], b0, b1);
                mma_s8(sc[j], qa[k + 1], b2, b3);
            }
        }
        float p[8][4];
        float mx0 = -INFINITY, mx1 = -INFINITY;
        for (int j = 0; j < 8; ++j) {
            int kc = j * 8 + t * 2;
            int key = tile * BN + kc;
            float ka = Kscl[buf][kc], kb = Kscl[buf][kc + 1];
            p[j][0] = key < S ? (float)sc[j][0] * qs0 * ka : -INFINITY;
            p[j][1] = key + 1 < S ? (float)sc[j][1] * qs0 * kb : -INFINITY;
            p[j][2] = key < S ? (float)sc[j][2] * qs1 * ka : -INFINITY;
            p[j][3] = key + 1 < S ? (float)sc[j][3] * qs1 * kb : -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(p[j][0], p[j][1]));
            mx1 = fmaxf(mx1, fmaxf(p[j][2], p[j][3]));
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, off));
        }
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float a0 = mn0 == -INFINITY ? 1.f : ex2(m0 - mn0);
        const float a1 = mn1 == -INFINITY ? 1.f : ex2(m1 - mn1);
        const float b0 = mn0 == -INFINITY ? 0.f : mn0, b1 = mn1 == -INFINITY ? 0.f : mn1;
        float s0 = 0.f, s1 = 0.f;
        for (int j = 0; j < 8; ++j) {
            p[j][0] = ex2(p[j][0] - b0); p[j][1] = ex2(p[j][1] - b0);
            p[j][2] = ex2(p[j][2] - b1); p[j][3] = ex2(p[j][3] - b1);
            s0 += p[j][0] + p[j][1];
            s1 += p[j][2] + p[j][3];
        }
        l0 = l0 * a0 + s0;
        l1 = l1 * a1 + s1;
        m0 = mn0;
        m1 = mn1;
        for (int n = 0; n < 16; ++n) {
            o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1;
        }

        for (int kk = 0; kk < 4; ++kk) {
            u32 a[4];
            a[0] = pack_h2(p[2 * kk][0], p[2 * kk][1]);
            a[1] = pack_h2(p[2 * kk][2], p[2 * kk][3]);
            a[2] = pack_h2(p[2 * kk + 1][0], p[2 * kk + 1][1]);
            a[3] = pack_h2(p[2 * kk + 1][2], p[2 * kk + 1][3]);
            const int key = kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
            for (int n = 0; n < 16; n += 2) {
                const int dim = (((n + (lane >> 4)) ^ (key & 7))) * 8;
                u32 r0_, r1_, r2_, r3_;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
                             : "r"(smem_addr(&Vs[buf][key * VPAD + dim])));
                mma_f16(o[n], a, r0_, r1_);
                mma_f16(o[n + 1], a, r2_, r3_);
            }
        }
        __syncthreads();
    }
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffff, l0, off);
        l1 += __shfl_xor_sync(0xffffffff, l1, off);
    }
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    for (int n = 0; n < 16; ++n) {
        int dim = n * 8 + t * 2;
        if (r0 < S) *(u32*)(out + ((size_t)r0 * H + h) * HD + dim) = (u32)f2bf(o[n][0] * i0) | ((u32)f2bf(o[n][1] * i0) << 16);
        if (r1 < S) *(u32*)(out + ((size_t)r1 * H + h) * HD + dim) = (u32)f2bf(o[n][2] * i1) | ((u32)f2bf(o[n][3] * i1) << 16);
    }
}
// ---------------------------------------------------------------------------------------------
// int8 GEMM with fused epilogue: out[m, n] = bf16(sum_k A[m,k] B[n,k] * xs[m] * ws[n] + bias[n]).
// A [M, K], B [N, K] int8 (K % 64 == 0). 128x128 CTA tile, BK = 64, 8 warps (2 x 4, 64x32 each),
// 3-stage cp.async pipeline, swizzled 64-byte smem rows. swiglu: B rows interleave (gate, up) pairs,
// out[m, n/2] = silu(gate) * up.
#define GBM 128
#define GBN 128
#define GBK 64
#define GST 3
__device__ __forceinline__ int sw64(int row, int ch) { return row * 64 + ((ch ^ ((row >> 1) & 3)) << 4); }

extern "C" __global__ void __launch_bounds__(256, 2) gemm_s8(const s8* __restrict__ A, const s8* __restrict__ B,
        const float* __restrict__ xs, const float* __restrict__ ws, int ws_per_row, const float* __restrict__ bias,
        int has_bias, u16* __restrict__ out, int M, int N, int K, int swiglu) {
    extern __shared__ __align__(16) unsigned char gsm[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    // grouped rasterization: GROUP_M tile-rows sweep the N tiles together so A and B stay in L2
    const int num_m = (M + GBM - 1) / GBM, num_n = (N + GBN - 1) / GBN;
    const int pid = blockIdx.x, GROUP_M = 8;
    const int in_group = GROUP_M * num_n, group = pid / in_group;
    const int first_m = group * GROUP_M, gsz = min(num_m - first_m, GROUP_M);
    const int tm = first_m + (pid % in_group) % gsz, tn = (pid % in_group) / gsz;
    const int m0 = tm * GBM, n0 = tn * GBN;
    const int nk = K / GBK;
    auto sA = [&](int st) { return gsm + st * (GBM + GBN) * GBK; };
    auto sB = [&](int st) { return gsm + st * (GBM + GBN) * GBK + GBM * GBK; };
    auto load = [&](int st, int kt) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 256, row = c >> 2, ch = c & 3;
            int gm = m0 + row, gn = n0 + row;
            cp16(sA(st) + sw64(row, ch), A + (size_t)min(gm, M - 1) * K + kt * GBK + ch * 16, gm < M ? 16 : 0);
            cp16(sB(st) + sw64(row, ch), B + (size_t)min(gn, N - 1) * K + kt * GBK + ch * 16, gn < N ? 16 : 0);
        }
        asm volatile("cp.async.commit_group;");
    };
    int acc[4][4][4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) for (int c = 0; c < 4; ++c) acc[a][b][c] = 0;
    load(0, 0);
    if (nk > 1) load(1, 1); else asm volatile("cp.async.commit_group;");
    for (int kt = 0; kt < nk; ++kt) {
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        if (kt + 2 < nk) load((kt + 2) % GST, kt + 2); else asm volatile("cp.async.commit_group;");
        const unsigned char* a_s = sA(kt % GST);
        const unsigned char* b_s = sB(kt % GST);
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            u32 af[4][4], bf[4][2];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                int row = wm * 64 + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                int ch = kk * 2 + (lane >> 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(smem_addr(a_s + sw64(row, ch))));
            }
            #pragma unroll
            for (int np = 0; np < 2; ++np) {
                int row = wn * 32 + np * 16 + (lane >> 4) * 8 + (lane & 7);
                int ch = kk * 2 + ((lane >> 3) & 1);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(bf[np * 2][0]), "=r"(bf[np * 2][1]), "=r"(bf[np * 2 + 1][0]), "=r"(bf[np * 2 + 1][1])
                             : "r"(smem_addr(b_s + sw64(row, ch))));
            }
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt) mma_s8(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
        }
    }
    asm volatile("cp.async.wait_group 0;");
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = m0 + wm * 64 + mt * 16 + g + half * 8;
            if (row >= M) continue;
            const float x = xs[row];
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int col = n0 + wn * 32 + nt * 8 + t * 2;
                if (col >= N) continue;
                float w0 = ws_per_row ? ws[col] : ws[0], w1 = ws_per_row ? ws[col + 1] : ws[0];
                float v0 = acc[mt][nt][half * 2] * x * w0 + (has_bias ? bias[col] : 0.f);
                float v1 = acc[mt][nt][half * 2 + 1] * x * w1 + (has_bias ? bias[col + 1] : 0.f);
                if (swiglu) {
                    float gt = bf2f(f2bf(v0)), up = bf2f(f2bf(v1));
                    float sg = bf2f(f2bf(gt / (1.f + __expf(-gt))));
                    out[(size_t)row * (N / 2) + col / 2] = f2bf(sg * up);
                } else {
                    *(u32*)(out + (size_t)row * N + col) = (u32)f2bf(v0) | ((u32)f2bf(v1) << 16);
                }
            }
        }
    }
}

"#;

fn ptx(arch: &str) -> Result<&'static str> {
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let r = PTX.get_or_init(|| {
        let opts = candle_core::cuda_backend::cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(false),
            arch: Some(Box::leak(arch.to_string().into_boxed_str())),
            options: vec!["--std=c++17".into()],
            ..Default::default()
        };
        candle_core::cuda_backend::cudarc::nvrtc::compile_ptx_with_opts(SRC, opts).map(|p| p.to_src()).map_err(|e| format!("nvrtc attn8: {e:?}"))
    });
    r.as_deref().map_err(|e| candle_core::Error::Msg(e.clone()))
}

const MODULE: &str = "h3_attn8";

fn arch_of(dev: &candle_core::CudaDevice) -> Result<(String, i32)> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute as A;
    let ctx = dev.cuda_stream().context().clone();
    let major = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).w()?;
    let minor = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).w()?;
    Ok((format!("compute_{major}{minor}"), major * 10 + minor))
}

fn ptr(st: &CudaStorage, l: &Layout) -> Result<u64> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice as Sl;
    let stream = st.device().cuda_stream();
    let off = l.start_offset();
    Ok(match &st.slice {
        Sl::U8(s) => s.device_ptr(&stream).0 + off as u64,
        Sl::F32(s) => s.device_ptr(&stream).0 + (off * 4) as u64,
        Sl::BF16(s) => s.device_ptr(&stream).0 + (off * 2) as u64,
        Sl::F16(s) => s.device_ptr(&stream).0 + (off * 2) as u64,
        _ => candle_core::bail!("attn8: unsupported storage"),
    })
}

fn tptr(t: &Tensor) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    match &*st {
        Storage::Cuda(c) => ptr(c, l),
        _ => candle_core::bail!("attn8: expected a cuda tensor"),
    }
}

/// `[S, H, 128]` bf16 -> (int8 `[H, S, 128]` as U8, f32 scales `[H, S]`), optional mean `[H, 128]`.
struct QuantHeads {
    mean: Option<Tensor>,
}
impl CustomOp1 for QuantHeads {
    fn name(&self) -> &'static str {
        "quant_heads"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        if !xl.is_contiguous() {
            candle_core::bail!("quant_heads: contiguous input required");
        }
        let (s, h, d) = xl.shape().dims3()?;
        let dev = x.device().clone();
        let (arch, _) = arch_of(&dev)?;
        let f = dev.get_or_load_custom_func("quant_heads", MODULE, ptx(&arch)?)?;
        // one buffer: int8 values then f32 scales
        let qbytes = s * h * d;
        let buf = unsafe { dev.alloc::<u8>(qbytes + s * h * 4)? };
        let out = CudaStorage::wrap_cuda_slice(buf, dev.clone());
        let base = ptr(&out, &Layout::contiguous(1))?;
        let (xp, mp) = (ptr(x, xl)?, self.mean.as_ref().map(tptr).transpose()?.unwrap_or(0));
        let (op, sp) = (base, base + qbytes as u64);
        let (si, hi) = (s as i32, h as i32);
        let rows = (s * h) as u32;
        let mut b = f.builder();
        b.arg(&xp).arg(&mp).arg(&op).arg(&sp).arg(&si).arg(&hi);
        unsafe { b.launch(LaunchConfig { grid_dim: (rows.div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        Ok((out, Shape::from(qbytes + s * h * 4)))
    }
}

struct ColMean;
impl CustomOp1 for ColMean {
    fn name(&self) -> &'static str {
        "col_mean"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        let (s, h, d) = xl.shape().dims3()?;
        let c = h * d;
        let dev = x.device().clone();
        let (arch, _) = arch_of(&dev)?;
        let f = dev.get_or_load_custom_func("col_sum", MODULE, ptx(&arch)?)?;
        let acc = dev.alloc_zeros::<f32>(c)?;
        let out = CudaStorage::wrap_cuda_slice(acc, dev.clone());
        let op = ptr(&out, &Layout::contiguous(1))?;
        let xp = ptr(x, xl)?;
        let parts = 128usize.min(s);
        let rows_per = s.div_ceil(parts);
        let (si, ci, ri) = (s as i32, c as i32, rows_per as i32);
        let mut b = f.builder();
        b.arg(&xp).arg(&op).arg(&si).arg(&ci).arg(&ri);
        unsafe { b.launch(LaunchConfig { grid_dim: (c.div_ceil(256) as u32, s.div_ceil(rows_per) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        Ok((out, Shape::from((h, d))))
    }
}

struct VToF16 {
    h: usize,
    col_off: usize,
}
impl CustomOp1 for VToF16 {
    fn name(&self) -> &'static str {
        "v_to_f16"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        let (s, width) = xl.shape().dims2()?;
        if xl.stride() != [width, 1] {
            candle_core::bail!("v_to_f16: row-contiguous 2D input required");
        }
        let dev = x.device().clone();
        let (arch, _) = arch_of(&dev)?;
        let f = dev.get_or_load_custom_func("v_to_f16", MODULE, ptx(&arch)?)?;
        let out = unsafe { dev.alloc::<half::f16>(s * self.h * 128)? };
        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        let (xp, op) = (ptr(x, xl)?, ptr(&out, &Layout::contiguous(1))?);
        let (rs, co, si, hi) = (width as u64, self.col_off as u64, s as i32, self.h as i32);
        let n = (s * self.h * 64) as u32;
        let mut b = f.builder();
        b.arg(&xp).arg(&rs).arg(&co).arg(&op).arg(&si).arg(&hi);
        unsafe { b.launch(LaunchConfig { grid_dim: (n.div_ceil(256).min(65535 * 4), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.w()?;
        Ok((out, Shape::from((self.h, s, 128))))
    }
}

struct Attn {
    s: usize,
    h: usize,
    scale_log2: f32,
}
impl CustomOp3 for Attn {
    fn name(&self) -> &'static str {
        "attn_qk8"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, q: &CudaStorage, ql: &Layout, k: &CudaStorage, kl: &Layout, v: &CudaStorage, vl: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = q.device().clone();
        let (arch, _) = arch_of(&dev)?;
        let f = dev.get_or_load_custom_func("attn_qk8", MODULE, ptx(&arch)?)?;
        let (s, h) = (self.s, self.h);
        let qbytes = (s * h * 128) as u64;
        let (qp, kp, vp) = (ptr(q, ql)?, ptr(k, kl)?, ptr(v, vl)?);
        let (qsp, ksp) = (qp + qbytes, kp + qbytes);
        let out = unsafe { dev.alloc::<half::bf16>(s * h * 128)? };
        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        let op = ptr(&out, &Layout::contiguous(1))?;
        let (si, hi, sc) = (s as i32, h as i32, self.scale_log2);
        let mut b = f.builder();
        b.arg(&qp).arg(&qsp).arg(&kp).arg(&ksp).arg(&vp).arg(&op).arg(&si).arg(&hi).arg(&sc);
        const SMEM: u32 = 2 * 64 * 128 + 2 * 64 * 128 * 2 + 2 * 64 * 4;
        f.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM as i32).w()?;
        let cfg = LaunchConfig { grid_dim: (s.div_ceil(64) as u32, h as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: SMEM };
        unsafe { b.launch(cfg) }.w()?;
        Ok((out, Shape::from((s, h, 128))))
    }
}

struct Gemm {
    ws: Tensor,
    bias: Option<Tensor>,
    swiglu: bool,
    rotate: bool,
}
impl candle_core::CustomOp2 for Gemm {
    fn name(&self) -> &'static str {
        "gemm_s8"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        let (m, k) = xl.shape().dims2()?;
        let (n, _) = bl.shape().dims2()?;
        let dev = x.device().clone();
        let (qa, xsa) = crate::kernels::quant_rows_raw(&dev, crate::kernels::raw_ptr(x, xl)?, crate::kernels::kind_code(x.dtype())?, m, k, self.rotate)?;
        let stream = dev.cuda_stream();
        let (ap, _ga) = qa.device_ptr(&stream);
        let (xsp, _gx) = xsa.device_ptr(&stream);
        let (arch, _) = arch_of(&dev)?;
        let f = dev.get_or_load_custom_func("gemm_s8", MODULE, ptx(&arch)?)?;
        const SMEM: u32 = 3 * (128 + 128) * 64;
        f.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM as i32).w()?;
        let n_out = if self.swiglu { n / 2 } else { n };
        let out = unsafe { dev.alloc::<half::bf16>(m * n_out)? };
        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        let (bp, op) = (ptr(b, bl)?, ptr(&out, &Layout::contiguous(1))?);
        let wsp = tptr(&self.ws)?;
        let per_row = (self.ws.elem_count() == n && n > 1) as i32;
        let (bp2, hb) = match &self.bias {
            Some(t) => (tptr(t)?, 1i32),
            None => (0u64, 0i32),
        };
        let (mi, ni, ki, sw) = (m as i32, n as i32, k as i32, self.swiglu as i32);
        let mut bld = f.builder();
        bld.arg(&ap).arg(&bp).arg(&xsp).arg(&wsp).arg(&per_row).arg(&bp2).arg(&hb).arg(&op).arg(&mi).arg(&ni).arg(&ki).arg(&sw);
        let cfg = LaunchConfig { grid_dim: ((n.div_ceil(128) * m.div_ceil(128)) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: SMEM };
        unsafe { bld.launch(cfg) }.w()?;
        Ok((out, Shape::from((m, n_out))))
    }
}

/// bf16 `x [M, K]` (quantized per row on the fly) times int8 weight `w [N, K]` (U8 bytes, scale `ws`
/// `[N, 1]` or scalar) + bias -> bf16 `[M, N]`, or `[M, N/2]` when `swiglu` (rows of `w` interleave
/// gate/up pairs).
pub fn gemm_s8(x: &Tensor, w: &Tensor, ws: &Tensor, bias: Option<&Tensor>, swiglu: bool, rotate: bool) -> Result<Tensor> {
    if x.dim(1)? % 64 != 0 || x.dtype() != DType::BF16 {
        candle_core::bail!("gemm_s8: bf16 input with K % 64 == 0 required");
    }
    let op = Gemm {
        ws: ws.to_dtype(DType::F32)?.flatten_all()?.contiguous()?,
        bias: bias.map(|b| b.to_dtype(DType::F32).and_then(|b| b.contiguous())).transpose()?,
        swiglu,
        rotate,
    };
    x.contiguous()?.apply_op2_no_bwd(&w.contiguous()?, &op)
}

/// Whether the tensor-core kernels in this module run on `dev` (CUDA, sm_80+).
pub fn available_arch(dev: &candle_core::Device) -> bool {
    match dev {
        candle_core::Device::Cuda(c) => arch_of(c).map(|(_, cc)| cc >= 80).unwrap_or(false),
        _ => false,
    }
}

/// Whether the int8 attention kernel applies (CUDA sm_80+, head_dim 128, `H3_ATTN` != "flash").
pub fn available(q: &Tensor) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("H3_ATTN").map(|v| v != "flash").unwrap_or(true));
    if !on || q.dim(D::Minus1).ok() != Some(128) || q.dtype() != DType::BF16 {
        return false;
    }
    match q.device() {
        candle_core::Device::Cuda(c) => arch_of(c).map(|(_, cc)| cc >= 80).unwrap_or(false),
        _ => false,
    }
}

/// [`attention`] with V read straight out of a fused qkv `[S, 3*H*128]` at column `v_col`.
pub fn attention_qkv(q: &Tensor, k: &Tensor, qkv: &Tensor, v_col: usize) -> Result<Tensor> {
    let h = q.dim(1)?;
    let v16 = qkv.apply_op1_no_bwd(&VToF16 { h, col_off: v_col })?;
    attention_f16v(q, k, &v16)
}

/// Non-causal attention on `[S, H, 128]` bf16 q/k/v (same head count). Returns `[S, H, 128]` bf16.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let v16 = v.transpose(0, 1)?.to_dtype(DType::F16)?.contiguous()?; // [H, S, D]
    attention_f16v(q, k, &v16)
}

fn attention_f16v(q: &Tensor, k: &Tensor, v16: &Tensor) -> Result<Tensor> {
    let (s, h, d) = q.dims3()?;
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    // smoothing K by its mean over tokens shifts every score of a query by the same amount
    let kmean = (k.apply_op1_no_bwd(&ColMean)? / s as f64)?; // [H, D]
    let q8 = q.apply_op1_no_bwd(&QuantHeads { mean: None })?;
    let k8 = k.apply_op1_no_bwd(&QuantHeads { mean: Some(kmean) })?;
    let scale_log2 = (1.0 / (d as f32).sqrt()) * std::f32::consts::LOG2_E;
    q8.apply_op3_no_bwd(&k8, v16, &Attn { s, h, scale_log2 })
}
