// Standalone bench and bitwise test of the INT4 prefill kernels (ggml/src/ggml-cuda/moe-int4-prefill.cuh) on random PACKED weights.
//
//   nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -extended-lambda -lineinfo -I ggml/src/ggml-cuda tools/marlin-compare/prefill-bench.cu -o prefill-bench
//   prefill-bench ROWS [SKEW=0.8] [ITERATIONS=10] [N_EXPERTS=256] [--no-check]
//
// Routing: every row picks 8 distinct experts with Zipf-like popularity 1/(rank+1)^SKEW (SKEW 0: uniform). The reference kernels below compute
// every slot in the arithmetic order of the decode kernels with plain loads (no pipelining), one warp per (8 slots, 16 columns); h, c3 and the
// output of the fast path must be bitwise equal to them. Timings are CUDA events per stage; TFLOPs count the valid slots only.
#include "moe-int4-prefill.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

// Random packed experts: codes are random words (every nibble pattern is a valid weight), scales random positive bf16 in [0.01, 0.05).
__global__ void fill_experts(char * base, const int64_t slot_bytes, const int64_t codes_bytes, const int64_t n_experts, const uint32_t seed) {
    const int64_t n = n_experts * slot_bytes / 4;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t byte = i * 4, e = byte / slot_bytes, off = byte % slot_bytes;
        const uint32_t hsh = hash32((uint32_t) i * 2654435761u + seed);
        if (off < codes_bytes) {
            ((uint32_t *) base)[i] = hsh;
        } else {
            __nv_bfloat162 s = __floats2bfloat162_rn(0.01f + 0.04f * (hsh & 0xffff) / 65536.f, 0.01f + 0.04f * (hsh >> 16) / 65536.f);
            ((__nv_bfloat162 *) base)[i] = s;
        }
        (void) e;
    }
}

__global__ void fill_x(float * x, const int64_t n, const uint32_t seed) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const uint32_t a = hash32((uint32_t) i + seed), b = hash32((uint32_t) i * 7919u + seed + 1);
        const float u1 = (a + 1.f) / 4294967296.f, u2 = b / 4294967296.f;
        x[i] = sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
    }
}

// ---- reference: decode-order arithmetic, plain loads --------------------------------------------------------------------------

// grid (128 fragments of 16 columns of one matrix, 4 * tiles), 32 threads. One warp: 8 slots (n tile `nt` of a tile) x 16 columns (fragment j of 64-column tile tl),
// gate and up together.
__global__ void ref_w13(const __nv_bfloat16 * xb, const int4 * gate, const int4 * up, const int4 * gate_s, const int4 * up_s, const int64_t estride,
        const int32_t * sorted, const int32_t * eids, const int32_t * ntiles, const int n_slots, const float limit, __nv_bfloat16 * h) {
    const int tile = blockIdx.y >> 2, nt = blockIdx.y & 3;
    if (tile >= ntiles[0]) { return; }
    const int lane = threadIdx.x, g = lane >> 2, c = lane & 3;
    const int tl = blockIdx.x >> 2, j = blockIdx.x & 3, e = eids[tile];
    const int sid = sorted[tile * PF_TILE + nt * 8 + g];
    const int64_t trow = sid < n_slots ? sid / PF_TOPK : 0;
    float tg[4] = { 0.f, 0.f, 0.f, 0.f }, tu[4] = { 0.f, 0.f, 0.f, 0.f };
    for (int ks = 0; ks < 4; ++ks) {
        float ag[4] = { 0.f, 0.f, 0.f, 0.f }, au[4] = { 0.f, 0.f, 0.f, 0.f };
        for (int r = ks * 64; r < ks * 64 + 64; ++r) {
            const uint32_t * bp = reinterpret_cast<const uint32_t *>(xb + trow * PF_HIDDEN + 16 * r + 2 * c);
            const uint32_t b0 = bp[0], b1 = bp[4];
            for (int m = 0; m < 2; ++m) {
                const int4 wv = (m ? up : gate)[(int64_t) e * estride + (int64_t) r * (PF_FF / 2) + tl * 32 + lane];
                const int4 sc = (m ? up_s : gate_s)[(int64_t) e * estride + (int64_t) (r >> 3) * (PF_FF / 8) + tl * 8 + g];
                const uint32_t qj = reinterpret_cast<const uint32_t *>(&wv)[j];
                const __nv_bfloat162 * s2 = reinterpret_cast<const __nv_bfloat162 *>(&sc);
                const __nv_bfloat162 sl = __low2bfloat162(s2[j]), sh = __high2bfloat162(s2[j]);
                mma_bf16(m ? au : ag, scl(deq2(qj), sl), scl(deq2(qj >> 8), sh), scl(deq2(qj >> 4), sl), scl(deq2(qj >> 12), sh), b0, b1);
            }
        }
        for (int v = 0; v < 4; ++v) { tg[v] += ag[v]; tu[v] += au[v]; }
    }
    for (int v = 0; v < 4; ++v) {
        const int s = sorted[tile * PF_TILE + nt * 8 + 2 * c + (v & 1)];
        if (s >= n_slots) { continue; }
        const float gf = fminf(__bfloat162float(__float2bfloat16(tg[v])), limit);
        const float uf = fmaxf(fminf(__bfloat162float(__float2bfloat16(tu[v])), limit), -limit);
        const float sl = __bfloat162float(__float2bfloat16(__fdiv_rn(gf, 1.0f + __nv_expf(-gf))));
        h[(int64_t) s * PF_FF + tl * 64 + 16 * j + g + 8 * (v >> 1)] = __float2bfloat16(sl * uf);
    }
}

// grid (256 fragments, 4 * tiles), 32 threads.
__global__ void ref_w2(const __nv_bfloat16 * h, const int4 * down, const int4 * down_s, const int64_t estride, const int32_t * sorted, const int32_t * eids,
        const int32_t * ntiles, const float * topk_w, const int n_slots, __nv_bfloat16 * c3) {
    const int tile = blockIdx.y >> 2, nt = blockIdx.y & 3;
    if (tile >= ntiles[0]) { return; }
    const int lane = threadIdx.x, g = lane >> 2, c = lane & 3;
    const int tl = blockIdx.x >> 2, j = blockIdx.x & 3, e = eids[tile];
    const int sid = sorted[tile * PF_TILE + nt * 8 + g];
    const int64_t trow = sid < n_slots ? sid : 0;
    float acc[4] = { 0.f, 0.f, 0.f, 0.f };
    for (int r = 0; r < PF_FF / 16; ++r) {
        const uint32_t * bp = reinterpret_cast<const uint32_t *>(h + trow * PF_FF + 16 * r + 2 * c);
        const uint32_t b0 = bp[0], b1 = bp[4];
        const int4 wv = down[(int64_t) e * estride + (int64_t) r * (PF_HIDDEN / 2) + tl * 32 + lane];
        const int4 sc = down_s[(int64_t) e * estride + (int64_t) (r >> 3) * (PF_HIDDEN / 8) + tl * 8 + g];
        const uint32_t qj = reinterpret_cast<const uint32_t *>(&wv)[j];
        const __nv_bfloat162 * s2 = reinterpret_cast<const __nv_bfloat162 *>(&sc);
        const __nv_bfloat162 sl = __low2bfloat162(s2[j]), sh = __high2bfloat162(s2[j]);
        mma_bf16(acc, scl(deq2(qj), sl), scl(deq2(qj >> 8), sh), scl(deq2(qj >> 4), sl), scl(deq2(qj >> 12), sh), b0, b1);
    }
    for (int v = 0; v < 4; ++v) {
        const int s = sorted[tile * PF_TILE + nt * 8 + 2 * c + (v & 1)];
        if (s >= n_slots) { continue; }
        const float w = __bfloat162float(__float2bfloat16(topk_w[s]));
        c3[(int64_t) s * PF_HIDDEN + tl * 64 + 16 * j + g + 8 * (v >> 1)] = __float2bfloat16(__bfloat162float(__float2bfloat16(acc[v])) * w);
    }
}

// ---- host ---------------------------------------------------------------------------------------------------------------------

static size_t count_diff(const void * a, const void * b, const size_t n_bytes, const char * what) {
    std::vector<uint16_t> x(n_bytes / 2), y(n_bytes / 2);
    CK(cudaMemcpy(x.data(), a, n_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b, n_bytes, cudaMemcpyDeviceToHost));
    size_t bad = 0, first = (size_t) -1;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] != y[i]) { if (first == (size_t) -1) { first = i; } ++bad; }
    }
    printf("%-26s %s: %zu of %zu elements differ%s", what, bad ? "MISMATCH" : "bitwise equal", bad, x.size(), bad ? "" : "\n");
    if (bad) { printf(" (first at %zu: %04x vs %04x)\n", first, x[first], y[first]); }
    return bad;
}

template <class F>
static float time_ms(F f, const int iters) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) { f(); }
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    return ms / iters;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: prefill-bench ROWS [SKEW] [ITERATIONS] [N_EXPERTS] [--no-check]\n"); return 1; }
    const int rows = atoi(argv[1]);
    const double skew = argc > 2 ? atof(argv[2]) : 0.8;
    const int iters = argc > 3 ? atoi(argv[3]) : 10;
    const int n_experts = argc > 4 ? atoi(argv[4]) : 256;
    bool check = true;
    for (int i = 1; i < argc; ++i) { if (!strcmp(argv[i], "--no-check")) { check = false; } }
    const int n_slots = rows * PF_TOPK;

    // routing fixture
    std::vector<int32_t> ids((size_t) rows * PF_TOPK);
    std::vector<float> router((size_t) rows * PF_TOPK);
    {
        std::mt19937 rng(1234);
        std::vector<int> perm(n_experts); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<double> pop(n_experts);
        for (int r = 0; r < n_experts; ++r) { pop[perm[r]] = 1.0 / std::pow(r + 1.0, skew); }
        std::uniform_real_distribution<double> uni(1e-12, 1.0);
        std::normal_distribution<double> nrm(0.0, 1.0);
        for (int t = 0; t < rows; ++t) {
            std::vector<std::pair<double, int>> key(n_experts);
            for (int e = 0; e < n_experts; ++e) { key[e] = { std::log(uni(rng)) / pop[e], e }; } // Efraimidis-Spirakis
            std::partial_sort(key.begin(), key.begin() + PF_TOPK, key.end(), [](auto & a, auto & b) { return a.first > b.first; });
            double mx = -1e30, sum = 0, v[PF_TOPK];
            for (int k = 0; k < PF_TOPK; ++k) { ids[t * PF_TOPK + k] = key[k].second; v[k] = 2.5 * nrm(rng); mx = std::max(mx, v[k]); }
            for (int k = 0; k < PF_TOPK; ++k) { v[k] = std::exp(v[k] - mx); sum += v[k]; }
            for (int k = 0; k < PF_TOPK; ++k) { router[t * PF_TOPK + k] = (float) (v[k] / sum); }
        }
    }
    {
        std::vector<int> cnt(n_experts, 0); for (int e : ids) { cnt[e]++; }
        int mx = 0, used = 0; long tiles = 0; for (int c : cnt) { mx = std::max(mx, c); used += c > 0; tiles += (c + PF_TILE - 1) / PF_TILE; }
        printf("rows %d, slots %d, experts used %d of %d, max slots of an expert %d, tiles %ld (%.0f%% of the tile slots are real)\n", rows, n_slots, used, n_experts,
            mx, tiles, 100.0 * n_slots / (tiles * PF_TILE));
    }

    const int64_t codes_bytes = (int64_t) PF_HIDDEN * PF_FF / 2, slot_bytes = codes_bytes + (int64_t) PF_HIDDEN / 128 * PF_FF * 2;
    char * wbuf[3];
    for (int i = 0; i < 3; ++i) {
        CK(cudaMalloc(&wbuf[i], n_experts * slot_bytes));
        fill_experts<<<4096, 256>>>(wbuf[i], slot_bytes, codes_bytes, n_experts, 1000 + i);
    }
    pf_weights w = { (const int4 *) wbuf[0], (const int4 *) wbuf[1], (const int4 *) wbuf[2], slot_bytes / 16 };
    float * x; CK(cudaMalloc(&x, (size_t) rows * PF_HIDDEN * sizeof(float)));
    fill_x<<<4096, 256>>>(x, (int64_t) rows * PF_HIDDEN, 77);
    int32_t * d_ids; float * d_router; float * dst;
    CK(cudaMalloc(&d_ids, ids.size() * 4)); CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_router, router.size() * 4)); CK(cudaMemcpy(d_router, router.data(), router.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dst, (size_t) rows * PF_HIDDEN * 4));
    const pf_scratch lay = pf_scratch_layout(nullptr, rows, n_experts);
    char * sbase; CK(cudaMalloc(&sbase, lay.bytes));
    const pf_scratch s = pf_scratch_layout(sbase, rows, n_experts);
    const float limit = 10.0f;
    CK(cudaDeviceSynchronize());
    printf("scratch %.1f MiB, tile cap %d\n", s.bytes / 1048576.0, s.cap);

    cudaStream_t st = 0;
    pf_route(st, s, d_ids, PF_TOPK, rows, n_experts, x, PF_HIDDEN);
    pf_launch_w13(st, s, w, rows, limit);
    pf_launch_w2(st, s, w, rows, d_router);
    pf_launch_sum(st, s, rows, dst);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    size_t bad = 0;
    if (check) {
        __nv_bfloat16 * h_ref, * c3_ref, * h_keep; float * dst_ref;
        const size_t hb = (size_t) n_slots * PF_FF * 2, cb = (size_t) n_slots * PF_HIDDEN * 2;
        CK(cudaMalloc(&h_ref, hb)); CK(cudaMalloc(&c3_ref, cb)); CK(cudaMalloc(&h_keep, hb)); CK(cudaMalloc(&dst_ref, (size_t) rows * PF_HIDDEN * 4));
        CK(cudaMemset(h_ref, 0xff, hb)); CK(cudaMemset(c3_ref, 0xff, cb));
        CK(cudaMemcpy(h_keep, s.h, hb, cudaMemcpyDeviceToDevice));
        ref_w13<<<dim3(PF_FF / 16, 4 * s.cap), 32>>>(s.xb, w.gate, w.up, w.gate + PF_CODES_INT4, w.up + PF_CODES_INT4, w.estride, s.sorted, s.eids, s.ntiles, n_slots, limit, h_ref);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        bad += count_diff(h_keep, h_ref, hb, "W13 + SwiGLU (h)");
        // W2 on the same h: the reference reads the reference h (equal when the line above is clean)
        CK(cudaMemcpy(s.h, h_ref, hb, cudaMemcpyDeviceToDevice));
        pf_launch_w2(st, s, w, rows, d_router);
        ref_w2<<<dim3(PF_HIDDEN / 16, 4 * s.cap), 32>>>(h_ref, w.down, w.down + PF_CODES_INT4, w.estride, s.sorted, s.eids, s.ntiles, d_router, n_slots, c3_ref);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        bad += count_diff(s.c3, c3_ref, cb, "W2 (c3)");
        CK(cudaMemcpy(s.h, h_keep, hb, cudaMemcpyDeviceToDevice));
        printf("%s\n", bad ? "FAILED" : "all stages bitwise equal to the decode-order reference");
        CK(cudaFree(h_ref)); CK(cudaFree(c3_ref)); CK(cudaFree(h_keep)); CK(cudaFree(dst_ref));
    }

    const double macs13 = (double) n_slots * 2 * PF_HIDDEN * PF_FF, macs2 = (double) n_slots * PF_FF * PF_HIDDEN;
    const float t_route = time_ms([&]() { pf_route(st, s, d_ids, PF_TOPK, rows, n_experts, x, PF_HIDDEN); }, iters);
    const float t13 = time_ms([&]() { pf_launch_w13(st, s, w, rows, limit); }, iters);
    const float t2  = time_ms([&]() { pf_launch_w2(st, s, w, rows, d_router); }, iters);
    const float tsum = time_ms([&]() { pf_launch_sum(st, s, rows, dst); }, iters);
    CK(cudaGetLastError());
    printf("route %.3f ms | W13+act %.3f ms (%.1f TFLOPs) | W2 %.3f ms (%.1f TFLOPs) | sum %.3f ms | total %.3f ms\n", t_route, t13, 2 * macs13 / t13 / 1e9, t2,
        2 * macs2 / t2 / 1e9, tsum, t_route + t13 + t2 + tsum);
    return bad ? 2 : 0;
}
