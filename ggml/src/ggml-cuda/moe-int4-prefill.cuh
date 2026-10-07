// delphi.cpp: prefill (more than 16 rows) kernels of the INT4 routed GLM FFN. Weights are the PACKED layout of moe-int4-decode.cuh.
//
// The results are bitwise equal to the decode kernels (moe-int4-decode.cu) for every slot, whatever the other rows of the batch are:
//   * weights are the mma A operand (16 columns x 16 k, one int32 of the packed layout per lane), tokens the B operand;
//   * the k16 chain of every accumulator is ascending, W13 sums four slices of 64 k16 rows in the fixed order 0..3, W2 is one chain;
//   * dequantization, scale multiply, activation and epilogue rounding are the code of moe-int4-common.cuh and the act/epilogue
//     expressions of the decode kernels.
// Only the data movement differs: a CTA of 4 warps owns 32 slots of one expert (a tile) and 64-column weight tiles; the activation
// tile goes through shared memory (cp.async, ldmatrix), each warp streams its own weight fragments through a private cp.async ring.
//
//   routing   hist -> scan -> scatter: tiles of 32 slots of one expert, ascending expert order, padding = n_slots
//   w13       gate and up in one CTA (2 warps each), tot += acc after every slice, fused clipped SwiGLU -> h[slot][FF]
//   w2        down projection, epilogue bf16(bf16(acc) * bf16(router weight)) -> c3[slot][HIDDEN]
//   sum       the slot sum of the decode path
#pragma once

#include "moe-int4-common.cuh"

#include <algorithm>
#include <cstddef>

extern "C" __device__ float __nv_expf(float); // libdevice: the precise expf, whatever -use_fast_math says

namespace {

constexpr int PF_HIDDEN = 4096;                        // W13: K, W2: N
constexpr int PF_FF     = 2048;                        // gate/up N, W2: K
constexpr int PF_TOPK   = 8;
constexpr int PF_TILE   = 32;                          // slots per tile (the mma N dimension of a warp: PF_NT n8 tiles)
constexpr int PF_NT     = PF_TILE / 8;
constexpr int PF_BK     = 64;                          // k per pipeline stage (4 k16 rows)
constexpr int PF_STAGES = 3;
constexpr int PF_WARPS  = 4;
constexpr int PF_THREADS = PF_WARPS * 32;

__device__ __forceinline__ void pf_ldsm_x4(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// ---- routing -----------------------------------------------------------------------------------------------------------------

// Counts the slots of every expert (cnt, zero on entry) and fills the sorted table with the padding value.
__global__ void __launch_bounds__(256) pf_hist(const int32_t * __restrict__ ids, const int64_t ids_stride, const int n_slots, const int n_experts,
        int32_t * __restrict__ cnt, int32_t * __restrict__ sorted, const int sorted_len) {
    extern __shared__ int pf_sh[];
    for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
        pf_sh[i] = 0;
    }
    __syncthreads();
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t first  = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t s = first; s < n_slots; s += stride) {
        int e = ids[(s / PF_TOPK) * ids_stride + s % PF_TOPK];
        e = min(max(e, 0), n_experts - 1);
        atomicAdd(&pf_sh[e], 1);
    }
    for (int64_t i = first; i < sorted_len; i += stride) {
        sorted[i] = n_slots;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
        if (pf_sh[i] != 0) {
            atomicAdd(&cnt[i], pf_sh[i]);
        }
    }
}

// One block of 1024 threads, one expert per thread (n_experts <= 1024): the first tile of every expert, the expert of every tile, the tile count.
__global__ void __launch_bounds__(1024) pf_scan(const int32_t * __restrict__ cnt, const int n_experts, int32_t * __restrict__ tile_base,
        int32_t * __restrict__ fill, int32_t * __restrict__ eids, int32_t * __restrict__ ntiles, const int cap) {
    __shared__ int wsum[32];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int nb = t < n_experts ? (cnt[t] + PF_TILE - 1) / PF_TILE : 0;
    int incl = nb;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) {
            incl += u;
        }
    }
    if (lane == 31) {
        wsum[warp] = incl;
    }
    __syncthreads();
    int woff = 0, total = 0;
    for (int w = 0; w < 32; ++w) {
        woff  += w < warp ? wsum[w] : 0;
        total += wsum[w];
    }
    if (t < n_experts) {
        const int base = woff + incl - nb;
        tile_base[t] = base;
        fill[t]      = 0;
        for (int i = 0; i < nb; ++i) {
            eids[base + i] = t;
        }
    }
    for (int i = total + t; i < cap; i += blockDim.x) {
        eids[i] = -1;
    }
    if (t == 0) {
        ntiles[0] = total;
    }
}

// Slot s goes to the next free position of its expert (the order inside an expert never changes a result: every slot is computed alone).
__global__ void __launch_bounds__(256) pf_scatter(const int32_t * __restrict__ ids, const int64_t ids_stride, const int n_slots, const int n_experts,
        const int32_t * __restrict__ tile_base, int32_t * __restrict__ fill, int32_t * __restrict__ sorted) {
    const int64_t s = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) {
        return;
    }
    int e = ids[(s / PF_TOPK) * ids_stride + s % PF_TOPK];
    e = min(max(e, 0), n_experts - 1);
    const int pos = atomicAdd(&fill[e], 1);
    sorted[(int64_t) tile_base[e] * PF_TILE + pos] = (int32_t) s;
}

__global__ void __launch_bounds__(256) pf_x2bf16(const float * __restrict__ x, const int64_t x_stride, const int n_rows, __nv_bfloat16 * __restrict__ xb) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i < (int64_t) n_rows * PF_HIDDEN) {
        const int64_t row = i / PF_HIDDEN, col = i % PF_HIDDEN;
        const float4 v = *reinterpret_cast<const float4 *>(x + row * x_stride + col);
        __nv_bfloat16 r[4] = { __float2bfloat16(v.x), __float2bfloat16(v.y), __float2bfloat16(v.z), __float2bfloat16(v.w) };
        *reinterpret_cast<uint2 *>(xb + i) = *reinterpret_cast<const uint2 *>(r);
    }
}

// ---- GEMM building blocks ----------------------------------------------------------------------------------------------------

// One k16 row of a warp tile: 64 weight columns (4 fragments, the int4 `wv` of the lane) x PF_TILE slots. bs: the stage of the activation tile,
// PF_TILE rows of 64 k (128 bytes) with the 16-byte chunks of a row XOR-swizzled by the row number (conflict-free ldmatrix and cp.async).
__device__ __forceinline__ void pf_k16(float (&acc)[4][PF_NT][4], const int4 wv, const int4 sc, const unsigned char * bs, const int r, const int lane) {
    uint32_t b[PF_NT][2];
#pragma unroll
    for (int p = 0; p < PF_NT / 2; ++p) {
        const int mi    = lane >> 3;
        const int row   = 8 * (2 * p + (mi >> 1)) + (lane & 7);
        const int chunk = (2 * r + (mi & 1)) ^ (lane & 7); // row & 7 == lane & 7
        pf_ldsm_x4(b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1], bs + row * 128 + chunk * 16);
    }
    const uint32_t * qv = reinterpret_cast<const uint32_t *>(&wv);
    const __nv_bfloat162 * s2 = reinterpret_cast<const __nv_bfloat162 *>(&sc);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t qj = qv[j];
        const __nv_bfloat162 sl = __low2bfloat162(s2[j]);   // column g   (position 2j)
        const __nv_bfloat162 sh = __high2bfloat162(s2[j]);  // column g+8 (position 2j+1)
        const uint32_t a0 = scl(deq2(qj), sl), a1 = scl(deq2(qj >> 8), sh), a2 = scl(deq2(qj >> 4), sl), a3 = scl(deq2(qj >> 12), sh);
#pragma unroll
        for (int nt = 0; nt < PF_NT; ++nt) {
            mma_bf16(acc[j][nt], a0, a1, a2, a3, b[nt][0], b[nt][1]);
        }
    }
}

// The activation rows of a tile: thread t copies the 16-byte chunks (t, t + 128, ...) of the PF_TILE x 64 stage. Rows of padding slots read row 0.
struct pf_bload {
    static constexpr int N = PF_TILE * 8 / PF_THREADS; // chunks per thread
    const __nv_bfloat16 * src[N];
    int dst[N];
};

template <bool W13>
__device__ __forceinline__ void pf_bload_init(pf_bload & b, const __nv_bfloat16 * a, const int32_t * sorted, const int tile, const int n_slots, const int tid) {
#pragma unroll
    for (int i = 0; i < pf_bload::N; ++i) {
        const int idx = tid + PF_THREADS * i, row = idx >> 3, chunk = idx & 7;
        const int sid = sorted[tile * PF_TILE + row];
        const int64_t arow = sid < n_slots ? (W13 ? sid / PF_TOPK : sid) : 0;
        b.src[i] = a + arow * (W13 ? PF_HIDDEN : PF_FF) + chunk * 8;
        b.dst[i] = row * 128 + ((chunk ^ (row & 7)) * 16);
    }
}

// ---- W13: gate and up, fused clipped SwiGLU ----------------------------------------------------------------------------------

// Grid (PF_FF / 128, tiles). CTA: warps 0, 1 take the gate tiles 2 * blockIdx.x + warp, warps 2, 3 the up tiles 2 * blockIdx.x + warp - 2; all
// four use the same activation stages. Per warp: acc = the current slice (64 k16 rows), tot = the fixed-order sum of the finished slices.
__global__ void __launch_bounds__(PF_THREADS, 2) pf_w13(const __nv_bfloat16 * __restrict__ xb, const int4 * __restrict__ gate, const int4 * __restrict__ up,
        const int4 * __restrict__ gate_s, const int4 * __restrict__ up_s, const int64_t estride, const int32_t * __restrict__ sorted,
        const int32_t * __restrict__ eids, const int32_t * __restrict__ ntiles, const int n_slots, const float limit, __nv_bfloat16 * __restrict__ h) {
#if __CUDA_ARCH__ >= 800
    constexpr int K = PF_HIDDEN, NIT = K / PF_BK, WROW = PF_FF / 2, SROW = PF_FF / 8, NG = K / 128, SLICE_IT = 64 * 16 / PF_BK;
    __shared__ __align__(16) unsigned char sB[PF_STAGES][PF_TILE * 128];
    __shared__ __align__(16) int4 sA[PF_STAGES][PF_WARPS][4][32];
    const int tile = blockIdx.y;
    if (tile >= ntiles[0]) {
        return;
    }
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
    const int e = eids[tile];
    const int mat = warp >> 1, tl = 2 * blockIdx.x + (warp & 1);
    const int4 * wp = (mat ? up : gate) + (int64_t) e * estride + tl * 32 + lane;
    const int4 * sp = (mat ? up_s : gate_s) + (int64_t) e * estride + tl * 8 + g;

    pf_bload bl;
    pf_bload_init<true>(bl, xb, sorted, tile, n_slots, tid);
    auto load_stage = [&](const int it, const int sb) {
#pragma unroll
        for (int i = 0; i < pf_bload::N; ++i) {
            cp_async16(&sB[sb][bl.dst[i]], bl.src[i] + it * PF_BK, true);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            cp_async16(&sA[sb][warp][r][lane], wp + (int64_t) (it * 4 + r) * WROW, true);
        }
    };

    float acc[4][PF_NT][4], tot[4][PF_NT][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int nt = 0; nt < PF_NT; ++nt)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                acc[j][nt][v] = 0.f;
                tot[j][nt][v] = 0.f;
            }

    int4 sc_cur = __ldg(sp), sc_nxt = __ldg(sp + SROW);
#pragma unroll
    for (int s = 0; s < PF_STAGES - 1; ++s) {
        load_stage(s, s);
        cp_async_commit();
    }
    for (int it = 0; it < NIT; ++it) {
        cp_async_wait<PF_STAGES - 2>();
        __syncthreads();
        if (it + PF_STAGES - 1 < NIT) {
            load_stage(it + PF_STAGES - 1, (it + PF_STAGES - 1) % PF_STAGES);
        }
        cp_async_commit();
        if ((it & 1) == 0 && it > 0) {
            sc_cur = sc_nxt;
            if (it / 2 + 1 < NG) {
                sc_nxt = __ldg(sp + (int64_t) (it / 2 + 1) * SROW);
            }
        }
        const int cb = it % PF_STAGES;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            pf_k16(acc, sA[cb][warp][r][lane], sc_cur, sB[cb], r, lane);
        }
        if ((it % SLICE_IT) == SLICE_IT - 1) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int nt = 0; nt < PF_NT; ++nt)
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        tot[j][nt][v] += acc[j][nt][v];
                        acc[j][nt][v] = 0.f;
                    }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // gate warps hand over bf16(G) (the layout of the registers: warp-private columns, no bank conflicts), up warps finish the activation
    __nv_bfloat16 * gx = reinterpret_cast<__nv_bfloat16 *>(&sB[0][0]);
    if (mat == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int nt = 0; nt < PF_NT; ++nt)
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    gx[((warp * 64 + (j * PF_NT + nt) * 4 + v)) * 32 + lane] = __float2bfloat16(tot[j][nt][v]);
                }
    }
    __syncthreads();
    if (mat == 1) {
        const int gw = warp - 2;
#pragma unroll
        for (int nt = 0; nt < PF_NT; ++nt)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int sid = sorted[tile * PF_TILE + 8 * nt + 2 * c + q];
                if (sid >= n_slots) {
                    continue;
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        const int v = 2 * hh + q;
                        const float gf = fminf(__bfloat162float(gx[((gw * 64 + (j * PF_NT + nt) * 4 + v)) * 32 + lane]), limit);
                        const float uf = fmaxf(fminf(__bfloat162float(__float2bfloat16(tot[j][nt][v])), limit), -limit);
                        const float sl = __bfloat162float(__float2bfloat16(__fdiv_rn(gf, 1.0f + __nv_expf(-gf))));
                        h[(int64_t) sid * PF_FF + tl * 64 + 16 * j + g + 8 * hh] = __float2bfloat16(sl * uf);
                    }
            }
    }
#else
    __trap();
#endif
}

// ---- W2: down projection with router weights ---------------------------------------------------------------------------------

// Grid (PF_HIDDEN / 256, tiles). Warp w of the CTA takes the 64-column tile 4 * blockIdx.x + w; one chain of 128 k16 rows.
__global__ void __launch_bounds__(PF_THREADS, 2) pf_w2(const __nv_bfloat16 * __restrict__ h, const int4 * __restrict__ down, const int4 * __restrict__ down_s,
        const int64_t estride, const int32_t * __restrict__ sorted, const int32_t * __restrict__ eids, const int32_t * __restrict__ ntiles,
        const float * __restrict__ topk_w, const int n_slots, __nv_bfloat16 * __restrict__ c3) {
#if __CUDA_ARCH__ >= 800
    constexpr int K = PF_FF, NIT = K / PF_BK, WROW = PF_HIDDEN / 2, SROW = PF_HIDDEN / 8, NG = K / 128;
    __shared__ __align__(16) unsigned char sB[PF_STAGES][PF_TILE * 128];
    __shared__ __align__(16) int4 sA[PF_STAGES][PF_WARPS][4][32];
    const int tile = blockIdx.y;
    if (tile >= ntiles[0]) {
        return;
    }
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
    const int e = eids[tile];
    const int tl = 4 * blockIdx.x + warp;
    const int4 * wp = down + (int64_t) e * estride + tl * 32 + lane;
    const int4 * sp = down_s + (int64_t) e * estride + tl * 8 + g;

    pf_bload bl;
    pf_bload_init<false>(bl, h, sorted, tile, n_slots, tid);
    auto load_stage = [&](const int it, const int sb) {
#pragma unroll
        for (int i = 0; i < pf_bload::N; ++i) {
            cp_async16(&sB[sb][bl.dst[i]], bl.src[i] + it * PF_BK, true);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            cp_async16(&sA[sb][warp][r][lane], wp + (int64_t) (it * 4 + r) * WROW, true);
        }
    };

    float acc[4][PF_NT][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int nt = 0; nt < PF_NT; ++nt)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                acc[j][nt][v] = 0.f;
            }

    int4 sc_cur = __ldg(sp), sc_nxt = __ldg(sp + SROW);
#pragma unroll
    for (int s = 0; s < PF_STAGES - 1; ++s) {
        load_stage(s, s);
        cp_async_commit();
    }
    for (int it = 0; it < NIT; ++it) {
        cp_async_wait<PF_STAGES - 2>();
        __syncthreads();
        if (it + PF_STAGES - 1 < NIT) {
            load_stage(it + PF_STAGES - 1, (it + PF_STAGES - 1) % PF_STAGES);
        }
        cp_async_commit();
        if ((it & 1) == 0 && it > 0) {
            sc_cur = sc_nxt;
            if (it / 2 + 1 < NG) {
                sc_nxt = __ldg(sp + (int64_t) (it / 2 + 1) * SROW);
            }
        }
        const int cb = it % PF_STAGES;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            pf_k16(acc, sA[cb][warp][r][lane], sc_cur, sB[cb], r, lane);
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int nt = 0; nt < PF_NT; ++nt)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int sid = sorted[tile * PF_TILE + 8 * nt + 2 * c + q];
            if (sid >= n_slots) {
                continue;
            }
            const float w = __bfloat162float(__float2bfloat16(topk_w[sid]));
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    c3[(int64_t) sid * PF_HIDDEN + tl * 64 + 16 * j + g + 8 * hh] =
                        __float2bfloat16(__bfloat162float(__float2bfloat16(acc[j][nt][2 * hh + q])) * w);
                }
        }
#else
    __trap();
#endif
}

// dst[t][n] = bf16(sum over k = 0..7 in slot order of c3[t*8+k][n]), as F32 (identical to moe_int4_sum of the decode path).
__global__ void __launch_bounds__(128) pf_sum(const __nv_bfloat16 * __restrict__ c3, float * __restrict__ dst) {
    const int64_t n   = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int64_t row = blockIdx.y;
    float acc[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
    for (int k = 0; k < PF_TOPK; ++k) {
        const uint4 v = *reinterpret_cast<const uint4 *>(c3 + (row * PF_TOPK + k) * PF_HIDDEN + n);
        const __nv_bfloat16 * p = reinterpret_cast<const __nv_bfloat16 *>(&v);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[i] += __bfloat162float(p[i]);
        }
    }
    float r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = __bfloat162float(__float2bfloat16(acc[i]));
    }
    float4 * o = reinterpret_cast<float4 *>(dst + row * PF_HIDDEN + n);
    o[0] = make_float4(r[0], r[1], r[2], r[3]);
    o[1] = make_float4(r[4], r[5], r[6], r[7]);
}

// ---- host side ---------------------------------------------------------------------------------------------------------------

// Scratch of one prefill region, carved from one allocation.
struct pf_scratch {
    int32_t *       cnt, * tile_base, * fill, * ntiles, * eids, * sorted;
    __nv_bfloat16 * xb, * h, * c3;
    int             cap;     // tiles at most
    size_t          bytes;
};

static inline int pf_tile_cap(const int n_rows, const int n_experts) {
    return (n_rows * PF_TOPK + PF_TILE - 1) / PF_TILE + n_experts;
}

static inline pf_scratch pf_scratch_layout(char * base, const int n_rows, const int n_experts) {
    pf_scratch s = {};
    s.cap = pf_tile_cap(n_rows, n_experts);
    size_t off = 0;
    auto take = [&](size_t n) { char * r = base ? base + off : nullptr; off += (n + 255) / 256 * 256; return r; };
    s.cnt       = (int32_t *) take(n_experts * sizeof(int32_t));
    s.tile_base = (int32_t *) take(n_experts * sizeof(int32_t));
    s.fill      = (int32_t *) take(n_experts * sizeof(int32_t));
    s.ntiles    = (int32_t *) take(sizeof(int32_t));
    s.eids      = (int32_t *) take((size_t) s.cap * sizeof(int32_t));
    s.sorted    = (int32_t *) take((size_t) s.cap * PF_TILE * sizeof(int32_t));
    s.xb        = (__nv_bfloat16 *) take((size_t) n_rows * PF_HIDDEN * sizeof(__nv_bfloat16));
    s.h         = (__nv_bfloat16 *) take((size_t) n_rows * PF_TOPK * PF_FF * sizeof(__nv_bfloat16));
    s.c3        = (__nv_bfloat16 *) take((size_t) n_rows * PF_TOPK * PF_HIDDEN * sizeof(__nv_bfloat16));
    s.bytes     = off;
    return s;
}

struct pf_weights {          // PACKED expert tensors (moe-int4-decode.cuh): expert e at base + e * estride; scales follow the 4 MiB of codes
    const int4 * gate, * up, * down;
    int64_t      estride;    // int4 units
};

static constexpr int64_t PF_CODES_INT4 = (int64_t) PF_HIDDEN * PF_FF / 2 / 16;

static inline void pf_route(cudaStream_t stream, const pf_scratch & s, const int32_t * ids, const int64_t ids_stride, const int n_rows, const int n_experts,
        const float * x, const int64_t x_stride) {
    const int n_slots = n_rows * PF_TOPK;
    cudaMemsetAsync(s.cnt, 0, n_experts * sizeof(int32_t), stream);
    pf_x2bf16<<<(unsigned) (((int64_t) n_rows * PF_HIDDEN / 4 + 255) / 256), 256, 0, stream>>>(x, x_stride, n_rows, s.xb);
    pf_hist<<<std::min(256, (n_slots + 255) / 256 + 1), 256, n_experts * sizeof(int), stream>>>(ids, ids_stride, n_slots, n_experts, s.cnt, s.sorted, s.cap * PF_TILE);
    pf_scan<<<1, 1024, 0, stream>>>(s.cnt, n_experts, s.tile_base, s.fill, s.eids, s.ntiles, s.cap);
    pf_scatter<<<(n_slots + 255) / 256, 256, 0, stream>>>(ids, ids_stride, n_slots, n_experts, s.tile_base, s.fill, s.sorted);
}

static inline void pf_launch_w13(cudaStream_t stream, const pf_scratch & s, const pf_weights & w, const int n_rows, const float limit) {
    pf_w13<<<dim3(PF_FF / 128, s.cap), PF_THREADS, 0, stream>>>(s.xb, w.gate, w.up, w.gate + PF_CODES_INT4, w.up + PF_CODES_INT4, w.estride, s.sorted, s.eids,
        s.ntiles, n_rows * PF_TOPK, limit, s.h);
}

static inline void pf_launch_w2(cudaStream_t stream, const pf_scratch & s, const pf_weights & w, const int n_rows, const float * topk_w) {
    pf_w2<<<dim3(PF_HIDDEN / 256, s.cap), PF_THREADS, 0, stream>>>(s.h, w.down, w.down + PF_CODES_INT4, w.estride, s.sorted, s.eids, s.ntiles, topk_w,
        n_rows * PF_TOPK, s.c3);
}

static inline void pf_launch_sum(cudaStream_t stream, const pf_scratch & s, const int n_rows, float * dst) {
    pf_sum<<<dim3(PF_HIDDEN / 8 / 128, n_rows), 128, 0, stream>>>(s.c3, dst);
}

} // namespace
