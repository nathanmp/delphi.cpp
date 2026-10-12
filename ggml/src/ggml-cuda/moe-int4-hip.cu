// SPDX-License-Identifier: MIT
// delphi.cpp: compact G128 routed FFNs on RDNA4. Adapted from the tested tools/rdna4-int4 prototypes at d448ff4df.
// F32 boundaries, symmetric Q8 activation groups of 128, exact integer group dots, and F32 scale/reduction arithmetic.
#include "moe-int4-decode.cuh"
#include "moe-schedule.cuh"

#ifdef GGML_HIP_MOE_INT4_AVAILABLE

#include "ggml-impl.h"
#include <atomic>
#include <map>
#include <set>
#include <mutex>
#include <vector>
#include <tuple>
#if defined(__x86_64__) && !defined(__HIP_DEVICE_COMPILE__)
#include <immintrin.h>
#define MOE_INT4_PACK_AVX2 1
#endif

namespace {

constexpr int MAX_ROWS = 16;
constexpr int MAX_TOPK = 16;
constexpr int MAX_SLOTS = MAX_ROWS * MAX_TOPK;
constexpr int BATCH_ROWS_DEFAULT = 4096;
constexpr int TILE_ROWS = 32;
constexpr int BATCH_MIN_ROWS = 64;

struct bank {
    const char * data;
    size_t stride;
    int k, n, experts;
    const int * packed;
};

struct routes {
    int groups;
    int experts[MAX_SLOTS], counts[MAX_SLOTS], slots[MAX_SLOTS * MAX_ROWS];
};

struct weight_state {
    int device, k, n, experts;
    char * data;
    size_t stride;
    int * flag;
    bool packed;
};

struct scratch {
    int device, h, f;
    void * base;
    size_t bytes;
    routes * routing;
    int8_t * xq, * hq;
    float * xd, * hd, * hidden, * down;
};

struct batch_scratch {
    int device, rows;
    void * base;
    size_t bytes;
    int * offsets, * slots, * schedule, * group_offsets, * xsum, * hsum;
    int8_t * xq, * hq;
    float * xd, * hd, * hidden, * down;
};

std::mutex state_mutex;
std::vector<weight_state> states;
std::map<std::tuple<const ggml_backend_cuda_context *, int, int, int>, scratch> scratches;
std::map<std::tuple<const ggml_backend_cuda_context *, int, int, int, int, int>, batch_scratch> batch_scratches;
std::atomic<int> override_mode{-1};
std::atomic<bool> have_g128{false};
std::atomic<int64_t> packs{0}, pack_us{0}, runs{0}, fallbacks{0}, wide_runs{0}, scratch_bytes{0};
std::atomic<int64_t> batch_runs{0}, batch_rows{0};

__device__ __forceinline__ int code_at(const char * src, int k, int n, int out, int kk, bool packed) {
    if (packed) {
        const int lane = out % 16 + 16 * ((kk % 16) / 8);
        const size_t wi = (size_t(out / 16) * (k / 16) + kk / 16) * 32 + lane;
        const unsigned q = reinterpret_cast<const unsigned *>(src)[wi];
        return (q >> (8 * (kk % 4) + 4 * ((kk % 8) / 4))) & 15;
    }
    const int j = kk % 128;
    const uint8_t * group = reinterpret_cast<const uint8_t *>(src) + (size_t(out) * (k / 128) + kk / 128) * 66;
    return (group[2 + (j / 32) * 16 + j % 16] >> (j % 32 >= 16 ? 4 : 0)) & 15;
}

// Packed scales are [N/16][K/128][16]: the 16 outputs of a tile and group are adjacent, so half a wave loads its 8 at once.
__host__ __device__ __forceinline__ size_t packed_scale_index(int k, int out, int group) {
    return (size_t(out / 16) * (k / 128) + group) * 16 + out % 16;
}

__device__ __forceinline__ float scale_at(const char * src, int k, int n, int out, int group, bool packed) {
    const uint16_t * d = packed ? reinterpret_cast<const uint16_t *>(src + size_t(k) * n / 2) + packed_scale_index(k, out, group) :
        reinterpret_cast<const uint16_t *>(src + (size_t(out) * (k / 128) + group) * 66);
    return __uint_as_float(unsigned(*d) << 16);
}

// N (4 or 8) adjacent packed scales of one group; out must be a multiple of N.
template <int N> __device__ __forceinline__ void scales_at(const char * src, int k, int n, int out, int group, float (&s)[N]) {
    const uint16_t * d = reinterpret_cast<const uint16_t *>(src + size_t(k) * n / 2) + packed_scale_index(k, out, group);
    unsigned w[N / 2];
    if constexpr (N == 4) { const uint2 v = *reinterpret_cast<const uint2 *>(d); w[0] = v.x; w[1] = v.y; }
    else { static_assert(N == 8, "4 or 8 scales"); const uint4 v = *reinterpret_cast<const uint4 *>(d); w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w; }
#pragma unroll
    for (int i = 0; i < N; ++i) { s[i] = __uint_as_float(i % 2 ? w[i / 2] & 0xffff0000u : w[i / 2] << 16); }
}

__device__ __forceinline__ void unpack8(unsigned q, int & lo, int & hi) {
    const unsigned a = q & 0x0f0f0f0fu, b = (q >> 4) & 0x0f0f0f0fu;
    lo = int(((a | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
    hi = int(((b | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
}

__device__ __forceinline__ float activate(float gate, float up, float limit) {
    if (limit > 0.0f) {
        gate = fminf(gate, limit);
        up = fminf(limit, fmaxf(-limit, up));
    }
    return (gate / (1.0f + expf(-gate))) * up;
}

// Packing and its inverse preserve every canonical code and BF16 scale. One temporary expert slice suffices.
// Packs whole experts, one block per 16-row x 128-weight tile: its 16 canonical groups (1 KB) are read
// coalesced into LDS, and its 256 code words (1 KB, contiguous) and 16 scales are written. (code_at with packed = true reads it back).
__global__ void pack_tiles(const char * src, char * dst, size_t stride, int k, int n) {
    __shared__ uint16_t g[16 * 33];
    const int kgs = k / 128, tile = blockIdx.x / kgs, kg = blockIdx.x % kgs;
    const char * s = src + blockIdx.y * stride;
    char * d = dst + blockIdx.y * stride;
    for (int i = threadIdx.x; i < 16 * 33; i += blockDim.x) {
        const int r = i / 33;
        g[i] = reinterpret_cast<const uint16_t *>(s + (size_t(tile * 16 + r) * kgs + kg) * 66)[i % 33];
    }
    __syncthreads();
    const int lane = threadIdx.x % 32, t8 = threadIdx.x / 32;
    const int j = t8 * 16 + (lane / 16) * 8; // the word holds codes j..j+7 of the group for output row lane % 16
    const uint8_t * b = reinterpret_cast<const uint8_t *>(g) + (lane % 16) * 66 + 2 + (j / 32) * 16 + j % 16;
    const unsigned x = b[0] | b[1] << 8 | b[2] << 16 | unsigned(b[3]) << 24;
    const unsigned y = b[4] | b[5] << 8 | b[6] << 16 | unsigned(b[7]) << 24;
    reinterpret_cast<unsigned *>(d)[(size_t(tile) * (k / 16) + kg * 8 + t8) * 32 + lane] = j % 32 >= 16 ?
        ((x >> 4) & 0x0f0f0f0fu) | (y & 0xf0f0f0f0u) : (x & 0x0f0f0f0fu) | ((y & 0x0f0f0f0fu) << 4);
    if (threadIdx.x < 16) {
        reinterpret_cast<uint16_t *>(d + size_t(k) * n / 2)[packed_scale_index(k, tile * 16 + threadIdx.x, kg)] = g[threadIdx.x * 33];
    }
}

__global__ void unpack_groups(const char * src, char * dst, size_t stride, int k, int n, int experts) {
    const size_t groups = size_t(k / 128) * n;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < groups * experts; i += size_t(gridDim.x) * blockDim.x) {
        const size_t e = i / groups, g = i % groups;
        const int out = g / (k / 128), kg = g % (k / 128);
        char * p = dst + e * stride + g * 66;
        *reinterpret_cast<uint16_t *>(p) = reinterpret_cast<const uint16_t *>(src + e * stride + size_t(k) * n / 2)[packed_scale_index(k, out, kg)];
        for (int b = 0; b < 4; ++b) {
            for (int j = 0; j < 16; ++j) {
                p[2 + b * 16 + j] = code_at(src + e * stride, k, n, out, kg * 128 + b * 32 + j, true) |
                    (code_at(src + e * stride, k, n, out, kg * 128 + b * 32 + j + 16, true) << 4);
            }
        }
    }
}

// The first occurrence owns a group. Split repeated IDs into tiles of 16, including duplicate IDs within a token.
__global__ void group_routes(const int * ids, routes * r, int slots, int group_rows, int topk, int ids_stride, int experts) {
    __shared__ int blocks[MAX_SLOTS];
    __shared__ int expert_ids[MAX_SLOTS];
    const int s = threadIdx.x;
    if (s < slots) { expert_ids[s] = ids[(s / topk) * ids_stride + s % topk]; }
    __syncthreads();
    const auto id = [&](int i) { return expert_ids[i]; };
    const bool valid = s < slots && id(s) >= 0 && id(s) < experts;
    int first = valid, count = 0;
    if (s < slots) {
        for (int j = 0; j < slots; ++j) {
            if (id(j) == id(s)) { count++; if (j < s) { first = 0; } }
        }
    }
    blocks[s] = first ? (count + group_rows - 1) / group_rows : 0;
    __syncthreads();
    if (s >= slots) { return; }
    int group = 0, row = 0, total = 0;
    for (int j = 0; j < slots; ++j) {
        total += blocks[j];
        if (s < slots) {
            if (blocks[j] && id(j) < id(s)) { group += blocks[j]; }
            if (j < s && id(j) == id(s)) { row++; }
        }
    }
    if (s == 0) { r->groups = total; }
    if (valid) {
        group += row / group_rows;
        r->slots[group * MAX_ROWS + row % group_rows] = s;
        if (row % group_rows == 0) {
            r->experts[group] = id(s);
            r->counts[group] = min(group_rows, count - row);
        }
    }
}

// sums (optional): each group's integer code sum, the zero-point term of batch_tiles.
__global__ void quantize_i8(const float * x, int8_t * q, float * d, int k, size_t row_stride, int rows_per_token, size_t token_stride,
        int * sums = nullptr) {
    __shared__ float a[128];
    __shared__ int c[4];
    const int j = threadIdx.x, g = blockIdx.x, row = blockIdx.y;
    const float * input = reinterpret_cast<const float *>(reinterpret_cast<const char *>(x) + (row % rows_per_token) * row_stride +
        (row / rows_per_token) * token_stride);
    const float v = input[g * 128 + j];
    a[j] = fabsf(v);
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (j < stride) { a[j] = fmaxf(a[j], a[j + stride]); }
        __syncthreads();
    }
    const float scale = a[0] / 127.0f, inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    const int8_t code = int8_t(fminf(127.0f, fmaxf(-127.0f, rintf(v * inv))));
    q[size_t(row) * k + g * 128 + j] = code;
    if (j == 0) { d[size_t(row) * (k / 128) + g] = scale; }
    if (sums) {
        int sum = code;
        for (int o = 16; o; o /= 2) { sum += __shfl_xor(sum, o, 32); }
        if (j % 32 == 0) { c[j / 32] = sum; }
        __syncthreads();
        if (j == 0) { sums[size_t(row) * (k / 128) + g] = c[0] + c[1] + c[2] + c[3]; }
    }
}

// Four waves divide K128 groups; lanes share coalesced K8 loads. Reuse a weight word across routed rows.
template <bool GATE_UP, int ROWS, bool PACKED = true, int FILTER_ROWS = 0>
__global__ void vector_projection(bank w, bank u, const int8_t * x, const float * dx, const routes * r,
        const int * ids, float * y, int topk, int ids_stride, int input_rows, float limit, size_t out_slot_stride, size_t out_token_stride) {
    const int group = blockIdx.y;
    if (r && group >= r->groups) { return; }
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, tile = blockIdx.x;
    const int out = tile * 16 + lane % 16;
    const int expert = r ? r->experts[group] : ids[(group / topk) * ids_stride + group % topk], nr = r ? r->counts[group] : 1;
    if constexpr (FILTER_ROWS > 0) { if (nr > FILTER_ROWS) { return; } }
    if (expert < 0 || expert >= w.experts) {
        if (!r && threadIdx.x < 16) {
            auto * output = reinterpret_cast<float *>(reinterpret_cast<char *>(y) + (group / topk) * out_token_stride +
                    (group % topk) * out_slot_stride);
            output[tile * 16 + threadIdx.x] = 0.0f;
        }
        return;
    }
    const char * wp = w.data + expert * w.stride, * up = GATE_UP ? u.data + expert * u.stride : nullptr;
    const bool packed = PACKED || *w.packed;
    float sum[ROWS] = {}, sumu[ROWS] = {};
    for (int kg = wave; kg < w.k / 128; kg += 4) {
        int acc[ROWS] = {}, accu[ROWS] = {};
        for (int t = 0; t < 8; ++t) {
            const size_t wi = (size_t(tile) * (w.k / 16) + kg * 8 + t) * 32 + lane;
            unsigned q = 0, uq = 0;
            const int kk = kg * 128 + t * 16 + (lane / 16) * 8;
            if (packed) { q = reinterpret_cast<const unsigned *>(wp)[wi]; }
            else {
                for (int j = 0; j < 8; ++j) { q |= unsigned(code_at(wp, w.k, w.n, out, kk + j, false)) << (8 * (j % 4) + 4 * (j / 4)); }
            }
            if constexpr (GATE_UP) { uq = reinterpret_cast<const unsigned *>(up)[wi]; }
            int lo, hi, ul = 0, uh = 0;
            unpack8(q, lo, hi);
            if constexpr (GATE_UP) { unpack8(uq, ul, uh); }
            for (int row = 0; row < ROWS; ++row) {
                if (row < nr) {
                    const int slot = r ? r->slots[group * MAX_ROWS + row] : group;
                    const int xr = GATE_UP ? slot / topk : (r ? slot : (slot / topk) * input_rows + (slot % topk) % input_rows);
                    const int * xp = reinterpret_cast<const int *>(x + size_t(xr) * w.k + kk);
                    acc[row] = ggml_cuda_dp4a(lo, xp[0], acc[row]);
                    acc[row] = ggml_cuda_dp4a(hi, xp[1], acc[row]);
                    if constexpr (GATE_UP) { accu[row] = ggml_cuda_dp4a(ul, xp[0], accu[row]); accu[row] = ggml_cuda_dp4a(uh, xp[1], accu[row]); }
                }
            }
        }
        const float ws = scale_at(wp, w.k, w.n, out, kg, packed);
        const float us = GATE_UP ? scale_at(up, u.k, u.n, out, kg, true) : 0.0f;
        for (int row = 0; row < ROWS; ++row) {
            if (row < nr) {
                const int slot = r ? r->slots[group * MAX_ROWS + row] : group;
                const int xr = GATE_UP ? slot / topk : (r ? slot : (slot / topk) * input_rows + (slot % topk) % input_rows);
                const int dot = acc[row] + __shfl_xor(acc[row], 16);
                const float xs = dx[size_t(xr) * (w.k / 128) + kg];
                sum[row] += float(dot) * (ws * xs);
                if constexpr (GATE_UP) { const int udot = accu[row] + __shfl_xor(accu[row], 16); sumu[row] += float(udot) * (us * xs); }
            }
        }
    }
    __shared__ float partial[4][ROWS][16], partialu[GATE_UP ? 4 : 1][ROWS][16];
    if (lane < 16) {
        for (int row = 0; row < ROWS; ++row) {
            if (row < nr) { partial[wave][row][lane] = sum[row]; if constexpr (GATE_UP) { partialu[wave][row][lane] = sumu[row]; } }
        }
    }
    __syncthreads();
    if (wave == 0 && lane < 16) {
        for (int row = 0; row < ROWS; ++row) {
            if (row < nr) {
                float v = partial[0][row][lane], uv = GATE_UP ? partialu[0][row][lane] : 0.0f;
                for (int j = 1; j < 4; ++j) { v += partial[j][row][lane]; if constexpr (GATE_UP) { uv += partialu[j][row][lane]; } }
                const int slot = r ? r->slots[group * MAX_ROWS + row] : group;
                float * output = reinterpret_cast<float *>(reinterpret_cast<char *>(y) + (slot / topk) * out_token_stride +
                    (slot % topk) * out_slot_stride);
                output[out] = GATE_UP ? activate(v, uv, limit) : v;
            }
        }
    }
}

// Wide-load comparator from the same G128 prototypes: four adjacent channels per aligned load.
template<bool GATE_UP,int ROWS,int FILTER_ROWS=0,int WAVES=4> __device__ __forceinline__ void vector_wide_body(
        bank wb,bank ub,const signed char * x,const float * dx,const routes * r,float * y,int topk,float limit) {
    const int n=wb.n,k=wb.k;
    int group=blockIdx.y;if(group>=r->groups)return;
    int lane=threadIdx.x&31,wave=threadIdx.x/32,tile=WAVES==1?blockIdx.x*4+wave:blockIdx.x;
    int base=tile*16+(lane%4)*4,expert=r->experts[group],nr=r->counts[group];
    const unsigned * w=reinterpret_cast<const unsigned *>(wb.data+expert*wb.stride);
    const unsigned * u=GATE_UP?reinterpret_cast<const unsigned *>(ub.data+expert*ub.stride):nullptr;
    if constexpr(FILTER_ROWS>0)if(nr>FILTER_ROWS)return;
    float sum[ROWS][4]={},sumu[ROWS][4]={};
    for(int kg=WAVES==1?0:wave;kg<k/128;kg+=WAVES){
        int acc[ROWS][4]={},accu[ROWS][4]={};
#pragma unroll
        for(int t=0;t<2;++t){
            size_t wi=((size_t(tile))*(k/16)+kg*8+t*4+lane/8)*32+(lane%4)*4+16*((lane/4)%2);
            uint4 packed=*reinterpret_cast<const uint4 *>(w+wi),up{};
            if constexpr(GATE_UP)up=*reinterpret_cast<const uint4 *>(u+wi);
            unsigned codes[4]={packed.x,packed.y,packed.z,packed.w},ucodes[4]={up.x,up.y,up.z,up.w};
            int kk=kg*128+t*64+(lane/4)*8;
#pragma unroll
            for(int row=0;row<ROWS;++row)if(row<nr){
                int slot=r->slots[group*MAX_ROWS+row],xr=GATE_UP?slot/topk:slot;
                auto xp=reinterpret_cast<const int *>(x+size_t(xr)*k+kk);int a=xp[0],b=xp[1];
#pragma unroll
                for(int col=0;col<4;++col){
                    int lo,hi;unpack8(codes[col],lo,hi);
                    acc[row][col]=ggml_cuda_dp4a(lo,a,acc[row][col]);
                    acc[row][col]=ggml_cuda_dp4a(hi,b,acc[row][col]);
                    if constexpr(GATE_UP){unpack8(ucodes[col],lo,hi);accu[row][col]=ggml_cuda_dp4a(lo,a,accu[row][col]);accu[row][col]=ggml_cuda_dp4a(hi,b,accu[row][col]);}
                }
            }
        }
#pragma unroll
        for(int row=0;row<ROWS;++row)if(row<nr){
            int slot=r->slots[group*MAX_ROWS+row],xr=GATE_UP?slot/topk:slot;float xs=dx[size_t(xr)*(k/128)+kg];
            float ws[4],us[4];scales_at(wb.data+expert*wb.stride,k,n,base,kg,ws);if constexpr(GATE_UP)scales_at(ub.data+expert*ub.stride,k,n,base,kg,us);
#pragma unroll
            for(int col=0;col<4;++col){
                int dot=acc[row][col],udot=accu[row][col];
#pragma unroll
                for(int stride=4;stride<=16;stride*=2){dot+=__shfl_xor(dot,stride);if constexpr(GATE_UP)udot+=__shfl_xor(udot,stride);}
                sum[row][col]+=float(dot)*(ws[col]*xs);
                if constexpr(GATE_UP){sumu[row][col]+=float(udot)*(us[col]*xs);}
            }
        }
    }
    if constexpr(WAVES==1){
        if(lane<4){
#pragma unroll
            for(int row=0;row<ROWS;++row)if(row<nr){
                int slot=r->slots[group*MAX_ROWS+row];
#pragma unroll
                for(int col=0;col<4;++col)y[size_t(slot)*n+base+col]=GATE_UP?activate(sum[row][col],sumu[row][col],limit):sum[row][col];
            }
        }
        return;
    }
    __shared__ float partial[WAVES][ROWS][16];
    __shared__ float partialu[GATE_UP?WAVES:1][ROWS][16];
    if(lane<4){
#pragma unroll
        for(int row=0;row<ROWS;++row)if(row<nr){
#pragma unroll
            for(int col=0;col<4;++col){int j=lane*4+col;partial[wave][row][j]=sum[row][col];if constexpr(GATE_UP)partialu[wave][row][j]=sumu[row][col];}
        }
    }
    __syncthreads();
    if(wave==0 && lane<16){
#pragma unroll
        for(int row=0;row<ROWS;++row)if(row<nr){
            float v=partial[0][row][lane],uv=GATE_UP?partialu[0][row][lane]:0;
#pragma unroll
            for(int j=1;j<WAVES;++j){v+=partial[j][row][lane];if constexpr(GATE_UP)uv+=partialu[j][row][lane];}
            int slot=r->slots[group*MAX_ROWS+row];y[size_t(slot)*n+tile*16+lane]=GATE_UP?activate(v,uv,limit):v;
        }
    }
}


template <bool GATE_UP, int ROWS> __global__ void vector_wide_projection(
        bank w, bank u, const int8_t * x, const float * dx, const routes * r, float * y, int topk, float limit) {
    vector_wide_body<GATE_UP, ROWS>(w, u, x, dx, r, y, topk, limit);
}

// Both schedules use the same W4A8 arithmetic. Slots retain token/top-k order for the final reduction.
template <bool GATE_UP>
__device__ __forceinline__ void wmma_tile_body(bank w, bank u, const int8_t * x, const float * dx,
        const int * slots, int expert, int nr, float * y, int topk, float limit) {
#if defined(RDNA4)
    using i2 = int __attribute__((ext_vector_type(2)));
    using i8 = int __attribute__((ext_vector_type(8)));
    using f8 = float __attribute__((ext_vector_type(8)));
    const int lane = threadIdx.x % 32, tile = blockIdx.x * 4 + threadIdx.x / 32, col = lane % 16;
    if (tile * 16 >= w.n) { return; }
    const int slot = col < nr ? slots[col] : 0, xr = GATE_UP ? slot / topk : slot;
    const char * wp = w.data + expert * w.stride, * up = GATE_UP ? u.data + expert * u.stride : nullptr;
    f8 sum{}, sumu{};
    for (int kg = 0; kg < w.k / 128; ++kg) {
        i8 acc{}, accu{};
        for (int t = 0; t < 8; ++t) {
            const size_t wi = (size_t(tile) * (w.k / 16) + kg * 8 + t) * 32 + lane;
            i2 a, b{}, ua{};
            int lo, hi;
            unpack8(reinterpret_cast<const unsigned *>(wp)[wi], lo, hi); a[0] = lo; a[1] = hi;
            if constexpr (GATE_UP) { unpack8(reinterpret_cast<const unsigned *>(up)[wi], lo, hi); ua[0] = lo; ua[1] = hi; }
            if (col < nr) {
                const int * xp = reinterpret_cast<const int *>(x + size_t(xr) * w.k + kg * 128 + t * 16 + (lane / 16) * 8);
                b[0] = xp[0]; b[1] = xp[1];
            }
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc, false);
            if constexpr (GATE_UP) { accu = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, ua, true, b, accu, false); }
        }
        const float xs = col < nr ? dx[size_t(xr) * (w.k / 128) + kg] : 0.0f;
        float ws[8], us[8];
        scales_at(wp, w.k, w.n, tile * 16 + (lane / 16) * 8, kg, ws);
        if constexpr (GATE_UP) { scales_at(up, u.k, u.n, tile * 16 + (lane / 16) * 8, kg, us); }
        for (int j = 0; j < 8; ++j) {
            sum[j] += float(acc[j]) * (ws[j] * xs);
            if constexpr (GATE_UP) { sumu[j] += float(accu[j]) * (us[j] * xs); }
        }
    }
    if (col < nr) {
        for (int j = 0; j < 8; ++j) { y[size_t(slot) * w.n + tile * 16 + (lane / 16) * 8 + j] = GATE_UP ? activate(sum[j], sumu[j], limit) : sum[j]; }
    }
#else
    GGML_UNUSED_VARS(w, u, x, dx, slots, expert, nr, y, topk, limit);
#endif
}

// A WMMA tile shares its packed weight fragments across up to 16 routed rows.
template <bool GATE_UP, int MIN_ROWS = 0>
__device__ __forceinline__ void wmma_body(bank w, bank u, const int8_t * x, const float * dx, const routes * r,
        float * y, int topk, float limit) {
    const int group = blockIdx.y;
    if (group >= r->groups || r->counts[group] < MIN_ROWS) { return; }
    wmma_tile_body<GATE_UP>(w, u, x, dx, r->slots + group * MAX_ROWS, r->experts[group], r->counts[group], y, topk, limit);
}

// Expert offsets come from the shared stable counting sort. Only the small group-count scan is new.
__global__ void batch_group_offsets(const int * offsets, int * groups, int experts, int rows) {
    __shared__ int scan[256];
    const int chunk = (experts + 255) / 256;
    const int begin = min(int(threadIdx.x) * chunk, experts), end = min(begin + chunk, experts);
    int sum = 0;
    for (int e = begin; e < end; ++e) { sum += (offsets[e + 1] - offsets[e] + rows - 1) / rows; }
    scan[threadIdx.x] = sum;
    __syncthreads();
    for (int step = 1; step < 256; step *= 2) {
        const int v = int(threadIdx.x) >= step ? scan[threadIdx.x - step] : 0;
        __syncthreads();
        scan[threadIdx.x] += v;
        __syncthreads();
    }
    int base = scan[threadIdx.x] - sum;
    for (int e = begin; e < end; ++e) {
        groups[e] = base;
        base += (offsets[e + 1] - offsets[e] + rows - 1) / rows;
    }
    if (threadIdx.x == 255) { groups[experts] = scan[255]; }
}

template <bool GATE_UP> __global__ void batch_projection(bank w, bank u, const int8_t * x, const float * dx,
        const int * offsets, const int * groups, const int * slots, int experts, float * y, int topk, float limit) {
    const int group = blockIdx.y;
    if (group >= groups[experts]) { return; }
    int lo = 0, hi = experts;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (groups[mid + 1] <= group) { lo = mid + 1; } else { hi = mid; }
    }
    const int start = offsets[lo] + (group - groups[lo]) * MAX_ROWS;
    const int count = min(MAX_ROWS, offsets[lo + 1] - start);
    wmma_tile_body<GATE_UP>(w, u, x, dx, slots + start, lo, count, y, topk, limit);
}

// Register-blocked G128 x Q8 batch tiles. Each wave owns MT routed-row tiles x NT output tiles (x2 for gate/up):
// one weight fragment feeds MT WMMAs and one activation fragment feeds NT (2NT) WMMAs. Codes enter the WMMA
// unsigned (q in 0..15) and the zero point is the accumulator start, -8 * (sum of the group's activation codes),
// so the integer group dot is unchanged; the F32 scale/accumulate order matches wmma_tile_body exactly.
// QUANT (gate/up only): the block's WAVES x NT tiles are one 128-wide group, which the epilogue quantizes to Q8
// as quantize_i8 would, writing codes, scale and code sum for the down projection.
template <bool GATE_UP, int MT, int NT, int WAVES, bool QUANT>
__global__ __launch_bounds__(WAVES * 32) void batch_tiles(bank w, bank u, const int8_t * x, const float * dx, const int * xsum,
        const int * offsets, const int * groups, const int * slots, int experts, float * y, int8_t * yq, float * yd, int * ysum,
        int topk, float limit) {
#if defined(RDNA4)
    using i2 = int __attribute__((ext_vector_type(2)));
    using i8 = int __attribute__((ext_vector_type(8)));
    using f8 = float __attribute__((ext_vector_type(8)));
    using u4 = unsigned __attribute__((ext_vector_type(4)));
    constexpr int ROWS = MT * 16, G = GATE_UP ? 2 : 1;
    static_assert(!QUANT || (GATE_UP && WAVES * NT * 16 == 128), "a quantizing block covers one 128-wide group");
    const int group = blockIdx.y;
    if (group >= groups[experts]) { return; }
    int lo = 0, hi = experts;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (groups[mid + 1] <= group) { lo = mid + 1; } else { hi = mid; }
    }
    const int start = offsets[lo] + (group - groups[lo]) * ROWS;
    const int count = min(ROWS, offsets[lo + 1] - start);
    const int lane = threadIdx.x % 32, col = lane % 16, half = lane / 16, wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
    const int ntiles = w.n / 16, tile0 = (blockIdx.x * WAVES + wave) * NT;
    if (!QUANT && tile0 >= ntiles) { return; }
    const int K = w.k, KG = K / 128;
    const char * wp[G];
    wp[0] = w.data + size_t(lo) * w.stride;
    if constexpr (GATE_UP) { wp[1] = u.data + size_t(lo) * u.stride; }
    int slot[MT];
    const int8_t * xp[MT];
    const float * xsp[MT];
    const int * xcp[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int r = i * 16 + col, s = slots[start + min(r, count - 1)], xr = GATE_UP ? s / topk : s;
        slot[i] = r < count ? s : -1;
        xp[i] = x + size_t(xr) * K + half * 8;
        xsp[i] = dx + size_t(xr) * KG;
        xcp[i] = xsum + size_t(xr) * KG;
    }
    const unsigned * wq[G][NT];
    const uint16_t * ws[G][NT];
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int t = min(tile0 + j, ntiles - 1);
            wq[g][j] = reinterpret_cast<const unsigned *>(wp[g]) + size_t(t) * (K / 16) * 32 + lane;
            ws[g][j] = reinterpret_cast<const uint16_t *>(wp[g] + size_t(K) * w.n / 2) + size_t(t) * KG * 16 + half * 8;
        }
    }
    f8 sum[G][NT][MT];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int i = 0; i < MT; ++i) { sum[g][j][i] = f8{}; }
    for (int kg = 0; kg < KG; ++kg) {
        unsigned word[G][NT][8];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int t = 0; t < 8; ++t) { word[g][j][t] = wq[g][j][(kg * 8 + t) * 32]; }
        u4 sc[G][NT];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int j = 0; j < NT; ++j) { sc[g][j] = *reinterpret_cast<const u4 *>(ws[g][j] + kg * 16); }
        i8 acc[G][NT][MT], init[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) { const int c = -8 * xcp[i][kg]; init[i] = i8{c, c, c, c, c, c, c, c}; }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            i2 b[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int * p = reinterpret_cast<const int *>(xp[i] + kg * 128 + t * 16);
                b[i][0] = p[0]; b[i][1] = p[1];
            }
#pragma unroll
            for (int g = 0; g < G; ++g)
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const unsigned q = word[g][j][t];
                    i2 a; a[0] = int(q & 0x0f0f0f0fu); a[1] = int((q >> 4) & 0x0f0f0f0fu);
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        acc[g][j][i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, b[i], t == 0 ? init[i] : acc[g][j][i], false);
                    }
                }
        }
        float xs[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) { xs[i] = xsp[i][kg]; }
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned pair = sc[g][j][e / 2];
                    const float s = __uint_as_float(e % 2 ? pair & 0xffff0000u : pair << 16);
#pragma unroll
                    for (int i = 0; i < MT; ++i) { sum[g][j][i][e] += float(acc[g][j][i][e]) * (s * xs[i]); }
                }
    }
    if constexpr (QUANT) {
        __shared__ float amax[WAVES][ROWS];
        __shared__ int csum[WAVES][ROWS];
        float v[NT][MT][8], m[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            m[i] = 0.0f;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[j][i][e] = activate(sum[0][j][i][e], sum[1][j][i][e], limit); m[i] = fmaxf(m[i], fabsf(v[j][i][e])); }
            m[i] = fmaxf(m[i], __shfl_xor(m[i], 16, 32));
            if (half == 0) { amax[wave][i * 16 + col] = m[i]; }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            float a = amax[0][i * 16 + col];
#pragma unroll
            for (int o = 1; o < WAVES; ++o) { a = fmaxf(a, amax[o][i * 16 + col]); }
            const float scale = a / 127.0f, inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            int c = 0;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                unsigned packed[2] = {0, 0};
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int q = int(int8_t(fminf(127.0f, fmaxf(-127.0f, rintf(v[j][i][e] * inv)))));
                    c += q;
                    packed[e / 4] |= unsigned(q & 0xff) << (8 * (e % 4));
                }
                if (slot[i] >= 0) {
                    *reinterpret_cast<uint2 *>(yq + size_t(slot[i]) * w.n + (tile0 + j) * 16 + half * 8) = make_uint2(packed[0], packed[1]);
                }
            }
            c += __shfl_xor(c, 16, 32);
            if (half == 0) { csum[wave][i * 16 + col] = c; }
            if (wave == 0 && half == 0 && slot[i] >= 0) { yd[size_t(slot[i]) * (w.n / 128) + blockIdx.x] = scale; }
        }
        __syncthreads();
        if (wave == 0 && half == 0) {
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (slot[i] < 0) { continue; }
                int c = 0;
#pragma unroll
                for (int o = 0; o < WAVES; ++o) { c += csum[o][i * 16 + col]; }
                ysum[size_t(slot[i]) * (w.n / 128) + blockIdx.x] = c;
            }
        }
        return;
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        if (tile0 + j >= ntiles) { break; }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            if (slot[i] < 0) { continue; }
            float * o = y + size_t(slot[i]) * w.n + (tile0 + j) * 16 + half * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) { o[e] = GATE_UP ? activate(sum[0][j][i][e], sum[G - 1][j][i][e], limit) : sum[0][j][i][e]; }
        }
    }
#else
    GGML_UNUSED_VARS(w, u, x, dx, xsum, offsets, groups, slots, experts, y, yq, yd, ysum, topk, limit);
#endif
}

template <bool GATE_UP, int MIN_ROWS = 0> __global__ void wmma_projection(
        bank w, bank u, const int8_t * x, const float * dx, const routes * r, float * y, int topk, float limit) {
    wmma_body<GATE_UP, MIN_ROWS>(w, u, x, dx, r, y, topk, limit);
}

// Match the tile geometry so singleton/vector and reused/WMMA groups share one launch.
template <bool GATE_UP> __global__ void singleton_projection(
        bank w, bank u, const int8_t * x, const float * dx, const routes * r, float * y, int topk, float limit) {
    const int group = blockIdx.y;
    if (group >= r->groups) { return; }
    if (r->counts[group] == 1) {
        vector_wide_body<GATE_UP, 1, 0, 1>(w, u, x, dx, r, y, topk, limit);
    } else {
        wmma_body<GATE_UP>(w, u, x, dx, r, y, topk, limit);
    }
}

__global__ void reduce_slots(const float * x, const float * rw, float * y, int h, int topk) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (j >= h) { return; }
    float sum = x[size_t(row) * topk * h + j] * rw[row * topk];
    for (int s = 1; s < topk; ++s) { sum += x[(size_t(row) * topk + s) * h + j] * rw[row * topk + s]; }
    y[size_t(row) * h + j] = sum;
}

bool capturing(ggml_backend_cuda_context & ctx) {
    hipStreamCaptureStatus status;
    CUDA_CHECK(hipStreamIsCapturing(ctx.stream(), &status));
    return status != hipStreamCaptureStatusNone;
}

weight_state * find_state(int device, const void * data) {
    for (auto & s : states) { if (s.device == device && s.data == data) { return &s; } }
    return nullptr;
}

bool supported_weight(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_Q4_G128 && t->ne[0] % 128 == 0 && t->ne[1] % 16 == 0 &&
        t->ne[3] == 1 && !t->view_src && ggml_is_contiguous(t) && t->data && t->nb[2] % 16 == 0;
}

bank get_bank(int device, const ggml_tensor * t) {
    std::lock_guard<std::mutex> lock(state_mutex);
    auto * s = find_state(device, t->data);
    GGML_ASSERT(s);
    return {s->data, s->stride, s->k, s->n, s->experts, s->flag};
}

// Keep flags alive until the weight buffer is freed. Captured generic kernels observe subsequent layout changes.
void register_weight(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (find_state(ctx.device, t->data)) { return; }
    GGML_ASSERT(!capturing(ctx));
    int * flag = nullptr;
    CUDA_CHECK(cudaMalloc(&flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(flag, 0, sizeof(int)));
    states.push_back({ctx.device, int(t->ne[0]), int(t->ne[1]), int(t->ne[2]), static_cast<char *>(t->data), t->nb[2], flag, false});
}

void change_layout(weight_state & s, bool packed) {
    if (s.packed == packed) { return; }
    ggml_cuda_set_device(s.device);
    CUDA_CHECK(cudaDeviceSynchronize());
    char * tmp = nullptr;
    const int chunk = std::min(s.experts, 4);
    CUDA_CHECK(cudaMalloc(&tmp, chunk * s.stride));
    const cudaStream_t stream = cudaStreamPerThread;
    for (int e = 0; e < s.experts; e += chunk) {
        const int ce = std::min(chunk, s.experts - e);
        char * dst = s.data + e * s.stride;
        CUDA_CHECK(cudaMemcpyAsync(tmp, dst, ce * s.stride, cudaMemcpyDeviceToDevice, stream));
        if (packed) {
            pack_tiles<<<dim3((s.n / 16) * (s.k / 128), ce), 256, 0, stream>>>(tmp, dst, s.stride, s.k, s.n);
        } else {
            const unsigned grid = std::min<size_t>((size_t(ce) * s.k * s.n / 128 + 255) / 256, 65535);
            unpack_groups<<<grid, 256, 0, stream>>>(tmp, dst, s.stride, s.k, s.n, ce);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    const int flag = packed ? 1 : 0;
    CUDA_CHECK(cudaMemcpyAsync(s.flag, &flag, sizeof(flag), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(tmp));
    s.packed = packed;
}

bool overlaps(const weight_state & s, int device, const void * base, size_t bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(base), b = reinterpret_cast<uintptr_t>(s.data);
    return bytes && s.device == device && a < b + s.stride * s.experts && b < a + bytes;
}

scratch * get_scratch(ggml_backend_cuda_context & ctx, int h, int f, bool create) {
    const auto key = std::make_tuple(&ctx, ctx.curr_stream_no, h, f);
    std::lock_guard<std::mutex> lock(state_mutex);
    auto it = scratches.find(key);
    if (it != scratches.end()) { return &it->second; }
    if (!create || capturing(ctx)) { return nullptr; }
    scratch s{};
    s.device = ctx.device; s.h = h; s.f = f;
    const size_t xq = MAX_ROWS * h, hq = MAX_SLOTS * f;
    const size_t xd = MAX_ROWS * (h / 128) * sizeof(float), hd = MAX_SLOTS * (f / 128) * sizeof(float);
    s.bytes = sizeof(routes) + xq + hq + xd + hd + size_t(MAX_SLOTS) * (h + f) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&s.base, s.bytes));
    char * p = static_cast<char *>(s.base);
    s.routing = reinterpret_cast<routes *>(p); p += sizeof(routes);
    s.xq = reinterpret_cast<int8_t *>(p); p += xq;
    s.hq = reinterpret_cast<int8_t *>(p); p += hq;
    s.xd = reinterpret_cast<float *>(p); p += xd;
    s.hd = reinterpret_cast<float *>(p); p += hd;
    s.hidden = reinterpret_cast<float *>(p); p += size_t(MAX_SLOTS) * f * sizeof(float);
    s.down = reinterpret_cast<float *>(p);
    scratch_bytes += s.bytes;
    return &scratches.emplace(key, s).first->second;
}

bool batch_enabled(int rows) {
    static const bool enabled = [] { const char * p = getenv("GGML_HIP_MOE_INT4_BATCH"); return !p || atoi(p) != 0; }();
    return enabled && rows >= BATCH_MIN_ROWS;
}

// GGML_HIP_MOE_INT4_TILES=0 restores the 16-row single-tile batch kernels (with their float hidden buffer).
bool tiles_enabled() {
    static const bool enabled = [] { const char * p = getenv("GGML_HIP_MOE_INT4_TILES"); return !p || atoi(p) != 0; }();
    return enabled;
}

// Rows per batch call: experts are re-read once per call, scratch grows by about topk * (h + f) * 5 bytes per row.
int batch_rows_per_call() {
    static const int rows = [] { const char * p = getenv("GGML_HIP_MOE_INT4_BATCH_ROWS"); return p ? std::max(BATCH_MIN_ROWS, atoi(p)) : BATCH_ROWS_DEFAULT; }();
    return rows;
}

// Sized for the largest batch seen, up to batch_rows_per_call(): a smaller scratch only means more calls per batch.
batch_scratch * get_batch_scratch(ggml_backend_cuda_context & ctx, int h, int f, int experts, int topk, int rows, bool create) {
    const auto key = std::make_tuple(&ctx, ctx.curr_stream_no, h, f, experts, topk);
    const int want = std::min(batch_rows_per_call(), (rows + 255) / 256 * 256);
    std::lock_guard<std::mutex> lock(state_mutex);
    auto it = batch_scratches.find(key);
    if (it != batch_scratches.end()) {
        if (it->second.rows >= want || !create || capturing(ctx)) { return &it->second; }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaFree(it->second.base));
        scratch_bytes -= it->second.bytes;
        batch_scratches.erase(it);
    }
    if (!create || capturing(ctx)) { return nullptr; }
    batch_scratch s{};
    s.device = ctx.device;
    s.rows = want;
    const bool tiles = tiles_enabled();
    const size_t slots = size_t(s.rows) * topk;
    const size_t schedule = ggml_cuda_moe_schedule_scratch_size(experts, slots);
    const size_t xq = size_t(s.rows) * h, hq = slots * f;
    const size_t xd = size_t(s.rows) * (h / 128), hd = slots * (f / 128);
    const size_t integers = 2 * (experts + 1) + slots + schedule + xd + hd;
    s.bytes = integers * sizeof(int) + xq + hq + (xd + hd + slots * (h + (tiles ? 0 : f))) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&s.base, s.bytes));
    char * p = static_cast<char *>(s.base);
    s.offsets = reinterpret_cast<int *>(p); p += (experts + 1) * sizeof(int);
    s.group_offsets = reinterpret_cast<int *>(p); p += (experts + 1) * sizeof(int);
    s.slots = reinterpret_cast<int *>(p); p += slots * sizeof(int);
    s.schedule = reinterpret_cast<int *>(p); p += schedule * sizeof(int);
    s.xsum = reinterpret_cast<int *>(p); p += xd * sizeof(int);
    s.hsum = reinterpret_cast<int *>(p); p += hd * sizeof(int);
    s.xq = reinterpret_cast<int8_t *>(p); p += xq;
    s.hq = reinterpret_cast<int8_t *>(p); p += hq;
    s.xd = reinterpret_cast<float *>(p); p += xd * sizeof(float);
    s.hd = reinterpret_cast<float *>(p); p += hd * sizeof(float);
    if (!tiles) { s.hidden = reinterpret_cast<float *>(p); p += slots * f * sizeof(float); }
    s.down = reinterpret_cast<float *>(p);
    scratch_bytes += s.bytes;
    GGML_LOG_WARN("%s: device %d: %d-row G128 batch scratch, %.1f MiB (%s kernels)\n", __func__, ctx.device, s.rows, s.bytes / 1048576.0,
        tiles ? "register-tiled" : "single-tile");
    return &batch_scratches.emplace(key, s).first->second;
}

void run_batch(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f, batch_scratch & s, int start, int m) {
    const int h = f.gate->ne[0], ff = f.gate->ne[1], experts = f.gate->ne[2], topk = f.ids->ne[0], slots = m * topk;
    const auto stream = ctx.stream();
    const bank gate = get_bank(ctx.device, f.gate), up = get_bank(ctx.device, f.up), down = get_bank(ctx.device, f.down);
    const int ids_stride = f.ids->nb[1] / sizeof(int);
    const int * ids = static_cast<const int *>(f.ids->data) + start * ids_stride;
    const float * x = static_cast<const float *>(f.x->data) + size_t(start) * h;
    const float * rw = static_cast<const float *>(f.weights->data) + size_t(start) * topk;
    float * out = static_cast<float *>(f.dst->data) + size_t(start) * h;
    if (f.partial) {
        // the schedule skips out-of-range ids (another part's experts), so their down rows are never written: the slot
        // reduction must read zeros there
        CUDA_CHECK(cudaMemsetAsync(s.down, 0, size_t(slots) * h * sizeof(float), stream));
    }
    ggml_cuda_moe_schedule_build(ids, ids_stride, m, topk, experts, s.offsets, s.slots, nullptr, s.schedule, stream);
    if (tiles_enabled()) {
        // gate/up blocks are one 128-wide group of 32 routed rows; down blocks four waves of 2 x 4 tiles
        batch_group_offsets<<<1, 256, 0, stream>>>(s.offsets, s.group_offsets, experts, TILE_ROWS);
        const int groups = std::min(slots, (slots + TILE_ROWS - 1) / TILE_ROWS + experts);
        quantize_i8<<<dim3(h / 128, m), 128, 0, stream>>>(x, s.xq, s.xd, h, h * sizeof(float), 1, h * sizeof(float), s.xsum);
        batch_tiles<true, 2, 2, 4, true><<<dim3(ff / 128, groups), 128, 0, stream>>>(gate, up, s.xq, s.xd, s.xsum,
                s.offsets, s.group_offsets, s.slots, experts, nullptr, s.hq, s.hd, s.hsum, topk, f.limit);
        batch_tiles<false, 2, 4, 4, false><<<dim3((h / 16 + 15) / 16, groups), 128, 0, stream>>>(down, {}, s.hq, s.hd, s.hsum,
                s.offsets, s.group_offsets, s.slots, experts, s.down, nullptr, nullptr, nullptr, topk, 0.0f);
        reduce_slots<<<dim3((h + 255) / 256, m), 256, 0, stream>>>(s.down, rw, out, h, topk);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    batch_group_offsets<<<1, 256, 0, stream>>>(s.offsets, s.group_offsets, experts, MAX_ROWS);
    const int groups = std::min(slots, (slots + MAX_ROWS - 1) / MAX_ROWS + experts);
    quantize_i8<<<dim3(h / 128, m), 128, 0, stream>>>(x, s.xq, s.xd, h, h * sizeof(float), 1, h * sizeof(float));
    batch_projection<true><<<dim3((ff + 63) / 64, groups), 128, 0, stream>>>(gate, up, s.xq, s.xd,
            s.offsets, s.group_offsets, s.slots, experts, s.hidden, topk, f.limit);
    quantize_i8<<<dim3(ff / 128, slots), 128, 0, stream>>>(s.hidden, s.hq, s.hd, ff, ff * sizeof(float), 1, ff * sizeof(float));
    batch_projection<false><<<dim3((h + 63) / 64, groups), 128, 0, stream>>>(down, {}, s.hq, s.hd,
            s.offsets, s.group_offsets, s.slots, experts, s.down, topk, 0.0f);
    reduce_slots<<<dim3((h + 255) / 256, m), 256, 0, stream>>>(s.down, rw, out, h, topk);
    CUDA_CHECK(cudaGetLastError());
}

bool eligible(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f) {
    if (!ggml_cuda_moe_int4_type_supported(ctx.device) || !supported_weight(f.gate) || !supported_weight(f.up) ||
            !supported_weight(f.down) || !f.x || !f.ids || !f.weights || !f.dst) { return false; }
    const int h = f.gate->ne[0], ff = f.gate->ne[1], e = f.gate->ne[2], topk = f.ids->ne[0], m = f.x->ne[2];
    if (h % 128 || ff % 128 || topk < 2 || topk > MAX_TOPK || m < 1 ||
            f.up->ne[0] != h || f.up->ne[1] != ff || f.up->ne[2] != e ||
            f.down->ne[0] != ff || f.down->ne[1] != h || f.down->ne[2] != e) { return false; }
    for (const ggml_tensor * t : {f.gate, f.up, f.down}) {
        if (t->op != GGML_OP_NONE || t->view_src || !t->buffer ||
                ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cuda_buffer_type(ctx.device) ||
                ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) { return false; }
    }
    return f.x->type == GGML_TYPE_F32 && f.x->ne[0] == h && f.x->ne[1] == 1 && f.x->ne[3] == 1 && ggml_is_contiguous(f.x) &&
        f.ids->type == GGML_TYPE_I32 && f.ids->ne[1] == m && f.ids->nb[0] == sizeof(int32_t) && f.ids->ne[2] == 1 && f.ids->ne[3] == 1 &&
        f.weights->type == GGML_TYPE_F32 && ggml_is_contiguous(f.weights) && ggml_nelements(f.weights) == int64_t(m) * topk &&
        f.dst->type == GGML_TYPE_F32 && ggml_is_contiguous(f.dst) && ggml_nelements(f.dst) == int64_t(m) * h;
}

template <int ROWS>
void run_chunk(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f, scratch & s, int start, int m) {
    const int h = f.gate->ne[0], ff = f.gate->ne[1], topk = f.ids->ne[0], slots = m * topk;
    const auto stream = ctx.stream();
    const bank gate = get_bank(ctx.device, f.gate), up = get_bank(ctx.device, f.up), down = get_bank(ctx.device, f.down);
    const float * x = static_cast<const float *>(f.x->data) + size_t(start) * h;
    const int ids_stride = f.ids->nb[1] / sizeof(int32_t);
    const int * ids = static_cast<const int *>(f.ids->data) + start * ids_stride;
    const float * rw = static_cast<const float *>(f.weights->data) + start * topk;
    float * out = static_cast<float *>(f.dst->data) + size_t(start) * h;
    if (f.partial) {
        CUDA_CHECK(cudaMemsetAsync(s.hidden, 0, size_t(slots) * ff * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(s.down, 0, size_t(slots) * h * sizeof(float), stream));
    }
    group_routes<<<1, MAX_SLOTS, 0, stream>>>(ids, s.routing, slots, ROWS, topk, ids_stride, f.gate->ne[2]);
    quantize_i8<<<dim3(h / 128, m), 128, 0, stream>>>(x, s.xq, s.xd, h, h * sizeof(float), 1, h * sizeof(float));
    // Qwen uses wide vector loads at one row and selects vector/WMMA per expert during verification.
    // Keep explicit comparators and leave other geometries on the previously tested vector mechanism.
    static const int path = [] {
        const char * p = getenv("GGML_HIP_MOE_INT4_PATH");
        if (!p || strcmp(p, "auto") == 0) { return -1; }
        if (strcmp(p, "vector") == 0) { return 0; }
        if (p && strcmp(p, "wmma") == 0) { return 1; }
        if (p && strcmp(p, "singleton") == 0) { return 4; }
        if (p && strcmp(p, "mixed") == 0) { return 2; }
        if (p && strcmp(p, "wide") == 0) { return 3; }
        GGML_ABORT("GGML_HIP_MOE_INT4_PATH must be auto, vector, wide, wmma, singleton or mixed");
    }();
    const int selected = path < 0 ? ((h == 2560 && ff == 640) ? (m == 1 ? 3 : 4) : 0) : path;
    if (selected == 4) {
        singleton_projection<true><<<dim3(ff / 64, slots), 128, 0, stream>>>(gate, up, s.xq, s.xd, s.routing, s.hidden, topk, f.limit);
    } else if (selected == 1) {
        wmma_projection<true><<<dim3((ff + 63) / 64, slots), 128, 0, stream>>>(gate, up, s.xq, s.xd, s.routing, s.hidden, topk, f.limit);
    } else if (selected == 3) {
        vector_wide_projection<true, ROWS><<<dim3(ff / 16, slots), 128, 0, stream>>>(gate, up, s.xq, s.xd, s.routing, s.hidden, topk, f.limit);
    } else if (selected == 2 && m > 1) {
        vector_projection<true, 2, true, 2><<<dim3(ff / 16, slots), 128, 0, stream>>>(gate, up, s.xq, s.xd, s.routing, nullptr,
            s.hidden, topk, ids_stride, 1, f.limit, ff * sizeof(float), size_t(topk) * ff * sizeof(float));
        wmma_projection<true, 3><<<dim3((ff + 63) / 64, slots), 128, 0, stream>>>(gate, up, s.xq, s.xd, s.routing, s.hidden, topk, f.limit);
    } else {
        vector_projection<true, ROWS><<<dim3(ff / 16, slots), 128, 0, stream>>>(gate, up, s.xq, s.xd, s.routing, nullptr,
            s.hidden, topk, ids_stride, 1, f.limit, ff * sizeof(float), size_t(topk) * ff * sizeof(float));
    }
    quantize_i8<<<dim3(ff / 128, slots), 128, 0, stream>>>(s.hidden, s.hq, s.hd, ff, ff * sizeof(float), 1, ff * sizeof(float));
    if (selected == 4) {
        singleton_projection<false><<<dim3(h / 64, slots), 128, 0, stream>>>(down, {}, s.hq, s.hd, s.routing, s.down, topk, 0.0f);
    } else if (selected == 1) {
        wmma_projection<false><<<dim3((h + 63) / 64, slots), 128, 0, stream>>>(down, {}, s.hq, s.hd, s.routing, s.down, topk, 0.0f);
    } else if (selected == 3) {
        vector_wide_projection<false, ROWS><<<dim3(h / 16, slots), 128, 0, stream>>>(down, {}, s.hq, s.hd, s.routing, s.down, topk, 0.0f);
    } else if (selected == 2 && m > 1) {
        vector_projection<false, 2, true, 2><<<dim3(h / 16, slots), 128, 0, stream>>>(down, {}, s.hq, s.hd, s.routing, nullptr,
            s.down, topk, ids_stride, topk, 0.0f, h * sizeof(float), size_t(topk) * h * sizeof(float));
        wmma_projection<false, 3><<<dim3((h + 63) / 64, slots), 128, 0, stream>>>(down, {}, s.hq, s.hd, s.routing, s.down, topk, 0.0f);
    } else {
        vector_projection<false, ROWS><<<dim3(h / 16, slots), 128, 0, stream>>>(down, {}, s.hq, s.hd, s.routing, nullptr,
            s.down, topk, ids_stride, topk, 0.0f, h * sizeof(float), size_t(topk) * h * sizeof(float));
    }
    reduce_slots<<<dim3((h + 255) / 256, m), 256, 0, stream>>>(s.down, rw, out, h, topk);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool ggml_cuda_moe_int4_enabled() {
    const int mode = override_mode.load(std::memory_order_relaxed);
    if (mode >= 0) { return mode != 0; }
    static const bool enabled = [] { const char * e = getenv("GGML_HIP_MOE_INT4_DECODE"); return !e || atoi(e) != 0; }();
    // Keep ordinary graphs outside the G128 preparation and matcher scans.
    return enabled && have_g128.load(std::memory_order_relaxed);
}

bool ggml_cuda_moe_int4_type_supported(int device) {
    return GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[device].cc);
}

ggml_type ggml_cuda_moe_int4_dispatch_type(const ggml_tensor * src0) {
    return src0->type == GGML_TYPE_Q4_G128 ? GGML_TYPE_Q4_0 : src0->type;
}

void ggml_hip_moe_int4_prepare_graph(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph) {
    if (!have_g128.load(std::memory_order_relaxed) && !ggml_cuda_moe_int4_enabled()) { return; }
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID && supported_weight(node->src[0])) { register_weight(ctx, node->src[0]); }
    }
}

void ggml_hip_moe_int4_note_tensor(const ggml_tensor * tensor) {
    if (tensor->type == GGML_TYPE_Q4_G128) { have_g128.store(true, std::memory_order_relaxed); }
}

bool ggml_cuda_moe_int4_prepare(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f) {
    if (!eligible(ctx, f)) { return false; }
    const int64_t start = ggml_time_us();
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        for (const ggml_tensor * t : {f.gate, f.up, f.down}) {
            auto * s = find_state(ctx.device, t->data);
            GGML_ASSERT(s);
            if (!s->packed) { change_layout(*s, true); packs++; }
        }
    }
    pack_us += ggml_time_us() - start;
    if (batch_enabled(f.x->ne[2])) {
        return get_batch_scratch(ctx, f.gate->ne[0], f.gate->ne[1], f.gate->ne[2], f.ids->ne[0], f.x->ne[2], true) != nullptr;
    }
    return get_scratch(ctx, f.gate->ne[0], f.gate->ne[1], true) != nullptr;
}

bool ggml_cuda_moe_int4_ready(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f) {
    if (!eligible(ctx, f)) { return false; }
    if (batch_enabled(f.x->ne[2])) {
        if (!get_batch_scratch(ctx, f.gate->ne[0], f.gate->ne[1], f.gate->ne[2], f.ids->ne[0], f.x->ne[2], !capturing(ctx))) { return false; }
    } else if (!get_scratch(ctx, f.gate->ne[0], f.gate->ne[1], !capturing(ctx))) { return false; }
    std::lock_guard<std::mutex> lock(state_mutex);
    for (const ggml_tensor * t : {f.gate, f.up, f.down}) { auto * s = find_state(ctx.device, t->data); if (!s || !s->packed) { return false; } }
    return true;
}

void ggml_cuda_moe_int4_run(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f) {
    const int rows = f.x->ne[2];
    if (batch_enabled(rows)) {
        auto * s = get_batch_scratch(ctx, f.gate->ne[0], f.gate->ne[1], f.gate->ne[2], f.ids->ne[0], rows, false);
        GGML_ASSERT(s);
        for (int start = 0; start < rows; start += s->rows) {
            run_batch(ctx, f, *s, start, std::min(s->rows, rows - start));
        }
        runs++; batch_runs++; batch_rows += rows;
        return;
    }
    scratch * s = get_scratch(ctx, f.gate->ne[0], f.gate->ne[1], false);
    GGML_ASSERT(s);
    for (int start = 0; start < rows; start += MAX_ROWS) {
        const int m = std::min(MAX_ROWS, rows - start);
        if (m == 1) { run_chunk<1>(ctx, f, *s, start, m); }
        else if (m <= 4) { run_chunk<4>(ctx, f, *s, start, m); }
        else if (m <= 8) { run_chunk<8>(ctx, f, *s, start, m); }
        else { run_chunk<16>(ctx, f, *s, start, m); }
    }
    runs++;
    if (rows > MAX_ROWS) { wide_runs++; }
}

void ggml_hip_moe_int4_mul_mat_id(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0], * x = dst->src[1], * ids = dst->src[2];
    GGML_ASSERT(supported_weight(w) && x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
        dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float) && ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    const int k = w->ne[0], topk = ids->ne[0], m = ids->ne[1], input_rows = x->ne[1];
    const int chunk_rows = std::min(m, std::min(1024, 65535 / std::max(topk, input_rows)));
    GGML_ASSERT(chunk_rows > 0);
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool());
    ggml_cuda_pool_alloc<float> d(ctx.pool());
    q.alloc(size_t(k) * input_rows * chunk_rows); d.alloc(size_t(k / 128) * input_rows * chunk_rows);
    const bank weights = get_bank(ctx.device, w);
    for (int start = 0; start < m; start += chunk_rows) {
        const int rows = std::min(chunk_rows, m - start);
        const float * input = reinterpret_cast<const float *>(static_cast<const char *>(x->data) + start * x->nb[2]);
        const int * selected = reinterpret_cast<const int *>(static_cast<const char *>(ids->data) + start * ids->nb[1]);
        float * output = reinterpret_cast<float *>(static_cast<char *>(dst->data) + start * dst->nb[2]);
        quantize_i8<<<dim3(k / 128, input_rows * rows), 128, 0, ctx.stream()>>>(input, q.get(), d.get(),
            k, x->nb[1], input_rows, x->nb[2]);
        vector_projection<false, 1, false><<<dim3(w->ne[1] / 16, topk * rows), 128, 0, ctx.stream()>>>(weights, {},
            q.get(), d.get(), nullptr, selected, output, topk, ids->nb[1] / sizeof(int32_t), input_rows, 0.0f, dst->nb[1], dst->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());
    fallbacks++;
}

void ggml_cuda_moe_int4_shadow(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, ggml_cuda_pool_alloc<char> & storage,
        ggml_tensor & src0_q, ggml_tensor & dst_q) {
    GGML_UNUSED_VARS(ctx, dst, storage, src0_q, dst_q);
    GGML_ABORT("HIP G128 nodes use the direct implementation");
}

void ggml_cuda_moe_int4_before_write(int device, const void * base, size_t bytes) {
    std::lock_guard<std::mutex> lock(state_mutex);
    for (auto & s : states) { if (overlaps(s, device, base, bytes)) { change_layout(s, false); } }
}

void ggml_cuda_moe_int4_check_not_packed(int device, const void * data, size_t bytes, const char * what) {
    GGML_UNUSED(what);
    ggml_cuda_moe_int4_before_write(device, data, bytes);
}

void ggml_cuda_moe_int4_forget(int device, const void * base, size_t bytes) {
    std::lock_guard<std::mutex> lock(state_mutex);
    for (auto it = states.begin(); it != states.end();) {
        if (!overlaps(*it, device, base, bytes)) { ++it; continue; }
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(it->flag));
        it = states.erase(it);
    }
}

void ggml_cuda_moe_int4_free_scratch(const ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (const char * e = getenv("GGML_HIP_MOE_INT4_STATS"); e && atoi(e)) {
        GGML_LOG_WARN("hip g128: states %zu, packs %lld (%.1f ms), fused runs %lld (batch %lld, rows %lld), unfused nodes %lld\n",
                states.size(), (long long) packs.load(), pack_us.load() / 1000.0, (long long) runs.load(),
                (long long) batch_runs.load(), (long long) batch_rows.load(), (long long) fallbacks.load());
    }
    for (auto it = scratches.begin(); it != scratches.end();) {
        if (std::get<0>(it->first) != ctx) { ++it; continue; }
        ggml_cuda_set_device(it->second.device);
        CUDA_CHECK(cudaFree(it->second.base));
        scratch_bytes -= it->second.bytes;
        it = scratches.erase(it);
    }
    for (auto it = batch_scratches.begin(); it != batch_scratches.end();) {
        if (std::get<0>(it->first) != ctx) { ++it; continue; }
        ggml_cuda_set_device(it->second.device);
        CUDA_CHECK(cudaFree(it->second.base));
        scratch_bytes -= it->second.bytes;
        it = batch_scratches.erase(it);
    }
}

// A packed bank stays packed: the expert is packed from its device copy into its slot. Other experts are not touched.
void ggml_cuda_moe_int4_set_expert(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int64_t expert, const void * src) {
    std::lock_guard<std::mutex> lock(state_mutex);
    weight_state * s = find_state(ctx.device, t->data);
    char * dst = static_cast<char *>(t->data) + expert * t->nb[2];
    if (!s || !s->packed) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, t->nb[2], cudaMemcpyDeviceToDevice, ctx.stream()));
        return;
    }
    GGML_ASSERT(expert >= 0 && expert < s->experts && size_t(t->nb[2]) == s->stride);
    pack_tiles<<<dim3((s->n / 16) * (s->k / 128), 1), 256, 0, ctx.stream()>>>(static_cast<const char *>(src), dst, s->stride, s->k, s->n);
    CUDA_CHECK(cudaGetLastError());
}

// Staging uploads that arrive packed (GGML_HIP_MOE_INT4_PACK_ON_UPLOAD=0: off), so the compute stream never repacks. Through
// bounce buffers, the staging threads pack while they copy (ggml_cuda_moe_int4_stage_pack_host) and the chunk goes straight
// into the slot. Direct uploads from page-locked weights (or GGML_SCHED_STAGE_PACK_HOST=0) go to a device scratch on the copy
// stream, and a pack stream writes them packed into the slot.
// 2026-10-09, MiMo prefill (8192-token ubatches, 16K tokens): the old in-place pack took ~54 of ~191 ms per layer (640-726
// t/s). In place with pack_tiles 692-810, device pack on upload 748-792, host pack 853-856 t/s (65K tokens: 842).
// The device pack kernels queue behind the compute kernels, which fill the device, so four scratch buffers let the uploads run
// ahead meanwhile (two made the ring upload-bound). A high-priority pack stream slowed the compute (prefill 831 -> 580-734).
#define STAGE_PACK_SCRATCH 4
struct stage_packer {
    cudaStream_t stream = nullptr;
    char * scratch[STAGE_PACK_SCRATCH] = {};
    size_t cap = 0;
    cudaEvent_t uploaded[STAGE_PACK_SCRATCH] = {}, packed[STAGE_PACK_SCRATCH] = {};
    int next = 0;
};
std::map<int, stage_packer> stage_packers;            // per device; used by that device's staging thread
std::set<std::pair<int, const void *>> staged_packed; // slots whose pending upload was packed (state_mutex)

static bool pack_on_upload_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_HIP_MOE_INT4_PACK_ON_UPLOAD");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// The layout of pack_tiles, computed on the host over whole experts (tiles [ith/nth, (ith+1)/nth) of the chunk), so that the
// staging threads can pack while they copy into the bounce buffers.
static void moe_int4_pack_host(int k, int n, size_t stride, void * dst, const void * src, size_t size, int ith, int nth) {
    const int kgs = k / 128;
    const size_t per_expert = size_t(n / 16) * kgs, total = size / stride * per_expert;
    const size_t t0 = total * ith / nth, t1 = total * (ith + 1) / nth;
    const uint64_t m = 0x0f0f0f0f0f0f0f0full;
    for (size_t t = t0; t < t1; ++t) {
        const size_t e = t / per_expert;
        const int tile = int((t % per_expert) / kgs), kg = int((t % per_expert) % kgs);
        const uint8_t * s = static_cast<const uint8_t *>(src) + e * stride;
        uint8_t * d = static_cast<uint8_t *>(dst) + e * stride;
        uint32_t * words = reinterpret_cast<uint32_t *>(d) + (size_t(tile) * (k / 16) + kg * 8) * 32;
        uint16_t * scales = reinterpret_cast<uint16_t *>(d + size_t(k) * n / 2);
        for (int r = 0; r < 16; ++r) {
            const size_t gi = size_t(tile * 16 + r) * kgs + kg;
            const uint8_t * g = s + gi * 66;
            memcpy(scales + packed_scale_index(k, tile * 16 + r, kg), g, 2);
            for (int sb = 0; sb < 4; ++sb) {
                uint64_t a, b;
                memcpy(&a, g + 2 + sb * 16, 8);
                memcpy(&b, g + 2 + sb * 16 + 8, 8);
                const uint64_t la = a & m, lb = b & m, ha = (a >> 4) & m, hb = (b >> 4) & m;
                words[(2 * sb) * 32 + r]          = uint32_t(la) | uint32_t(la >> 32) << 4;
                words[(2 * sb) * 32 + r + 16]     = uint32_t(lb) | uint32_t(lb >> 32) << 4;
                words[(2 * sb + 1) * 32 + r]      = uint32_t(ha) | uint32_t(ha >> 32) << 4;
                words[(2 * sb + 1) * 32 + r + 16] = uint32_t(hb) | uint32_t(hb >> 32) << 4;
            }
        }
    }
}

#ifdef MOE_INT4_PACK_AVX2
// moe_int4_pack_host with AVX2, a whole tile (16 consecutive rows, one contiguous run of the source) per step while the next
// tile is prefetched, and the packed groups streamed out of L1: ~2x the scalar rate (16 threads 46 vs 23 GB/s; memcpy 50).
// Per row, a 32-byte load holds two 16-byte sub-blocks; the nibble masks, a 28-bit shift that brings byte i+4's nibble next to
// byte i's, and an even-dword permute give the words {la, lb} of each sub-block, which 4x4 transposes place across 4 rows.
__attribute__((target("avx2")))
static void moe_int4_pack_host_avx2(int k, int n, size_t stride, void * dst, const void * src, size_t size, int ith, int nth) {
    const int kgs = k / 128;
    const size_t tiles = size_t(n / 16), total = size / stride * tiles;
    const size_t t0 = total * ith / nth, t1 = total * (ith + 1) / nth;
    const size_t rs = size_t(kgs) * 66, tile_bytes = 16 * rs;
    const __m256i m = _mm256_set1_epi8(0x0f);
    const __m256i even = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    alignas(64) uint32_t blk[256];
    alignas(64) uint16_t sc[16];
    const uint8_t * src8 = static_cast<const uint8_t *>(src);
    for (size_t t = t0; t < t1; ++t) {
        const size_t e = t / tiles;
        const int tile = int(t % tiles);
        uint8_t * d = static_cast<uint8_t *>(dst) + e * stride;
        const uint8_t * rows = src8 + e * stride + size_t(tile) * tile_bytes;
        const uint8_t * next = t + 1 < t1 ? src8 + (t + 1) / tiles * stride + (t + 1) % tiles * tile_bytes : nullptr;
        for (int kg = 0; kg < kgs; ++kg) {
            if (next != nullptr) {
                for (size_t o = size_t(kg) * tile_bytes / kgs; o < size_t(kg + 1) * tile_bytes / kgs; o += 64) {
                    _mm_prefetch((const char *) next + o, _MM_HINT_T0);
                }
            }
            const uint8_t * g0 = rows + size_t(kg) * 66;
            for (int r0 = 0; r0 < 16; r0 += 4) {
                __m128i v[4][4]; // [row][lo, hi of sub-blocks 0-1, lo, hi of sub-blocks 2-3]
                for (int i = 0; i < 4; ++i) {
                    const uint8_t * g = g0 + (r0 + i) * rs;
                    memcpy(sc + r0 + i, g, 2);
                    for (int h = 0; h < 2; ++h) {
                        const __m256i x = _mm256_loadu_si256((const __m256i *) (g + 2 + 32 * h));
                        __m256i lo = _mm256_and_si256(x, m), hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), m);
                        lo = _mm256_permutevar8x32_epi32(_mm256_or_si256(lo, _mm256_srli_epi64(lo, 28)), even);
                        hi = _mm256_permutevar8x32_epi32(_mm256_or_si256(hi, _mm256_srli_epi64(hi, 28)), even);
                        v[i][2 * h]     = _mm256_castsi256_si128(lo);
                        v[i][2 * h + 1] = _mm256_castsi256_si128(hi);
                    }
                }
                for (int j = 0; j < 4; ++j) {
                    const __m128i a = _mm_unpacklo_epi32(v[0][j], v[1][j]), b = _mm_unpackhi_epi32(v[0][j], v[1][j]);
                    const __m128i c = _mm_unpacklo_epi32(v[2][j], v[3][j]), dd = _mm_unpackhi_epi32(v[2][j], v[3][j]);
                    const int line = (j & 1) + 4 * (j >> 1); // words of sub-block 2*(j>>1), lo (j even) or hi nibbles
                    _mm_store_si128((__m128i *) (blk + line * 32 + r0),            _mm_unpacklo_epi64(a, c));
                    _mm_store_si128((__m128i *) (blk + line * 32 + 16 + r0),       _mm_unpackhi_epi64(a, c));
                    _mm_store_si128((__m128i *) (blk + (line + 2) * 32 + r0),      _mm_unpacklo_epi64(b, dd));
                    _mm_store_si128((__m128i *) (blk + (line + 2) * 32 + 16 + r0), _mm_unpackhi_epi64(b, dd));
                }
            }
            __m256i * words = reinterpret_cast<__m256i *>(reinterpret_cast<uint32_t *>(d) + (size_t(tile) * (k / 16) + kg * 8) * 32);
            for (int i = 0; i < 32; ++i) {
                _mm256_stream_si256(words + i, _mm256_load_si256((const __m256i *) blk + i));
            }
            uint16_t * scales = reinterpret_cast<uint16_t *>(d + size_t(k) * n / 2) + packed_scale_index(k, tile * 16, kg);
            _mm256_stream_si256((__m256i *) scales, _mm256_load_si256((const __m256i *) sc));
        }
    }
    // the streamed stores must be visible before the upload reads the bounce buffer
    _mm_sfence();
}
#endif

void ggml_cuda_moe_int4_stage_pack_host(const ggml_tensor * w, void * dst, const void * src, size_t size, int ith, int nth) {
    GGML_ASSERT(size % w->nb[2] == 0);
#ifdef MOE_INT4_PACK_AVX2
    static const bool avx2 = __builtin_cpu_supports("avx2") && getenv("GGML_HIP_MOE_INT4_PACK_SCALAR") == nullptr;
    if (avx2 && (uintptr_t(dst) | w->nb[2]) % 32 == 0) {
        moe_int4_pack_host_avx2(int(w->ne[0]), int(w->ne[1]), w->nb[2], dst, src, size, ith, nth);
        return;
    }
#endif
    moe_int4_pack_host(int(w->ne[0]), int(w->ne[1]), w->nb[2], dst, src, size, ith, nth);
}

size_t ggml_cuda_moe_int4_stage_pack_unit(int device, const ggml_tensor * w) {
    return pack_on_upload_enabled() && supported_weight(w) && ggml_cuda_moe_int4_type_supported(device) ? size_t(w->nb[2]) : 0;
}

void ggml_cuda_moe_int4_stage_upload_packed(int device, cudaStream_t copy, const ggml_tensor * w, char * slot, size_t offset,
        const void * src, size_t size, bool prepacked, cudaEvent_t copied, cudaEvent_t done) {
    const size_t stride = w->nb[2];
    GGML_ASSERT(offset % stride == 0 && size % stride == 0 && size > 0);
    stage_packer * p;
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        p = &stage_packers[device];
        staged_packed.insert({device, slot});
    }
    if (prepacked) {
        // packed on the host (ggml_cuda_moe_int4_stage_pack_host): straight into the slot
        CUDA_CHECK(cudaMemcpyAsync(slot + offset, src, size, cudaMemcpyHostToDevice, copy));
        if (copied) {
            CUDA_CHECK(cudaEventRecord(copied, copy));
        }
        if (done) {
            CUDA_CHECK(cudaEventRecord(done, copy));
        }
        return;
    }
    if (p->stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking));
        for (int b = 0; b < STAGE_PACK_SCRATCH; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&p->uploaded[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&p->packed[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(p->packed[b], p->stream));
        }
    }
    if (size > p->cap) {
        CUDA_CHECK(cudaStreamSynchronize(p->stream));
        CUDA_CHECK(cudaStreamSynchronize(copy));
        for (char *& b : p->scratch) {
            if (b) { CUDA_CHECK(cudaFree(b)); }
            CUDA_CHECK(cudaMalloc(&b, size));
        }
        p->cap = size;
    }
    // the upload of a chunk overlaps the packing of the ones before
    const int b = p->next;
    p->next = (b + 1) % STAGE_PACK_SCRATCH;
    // a host wait: a copy-stream wait on the pack stream slowed the uploads from 25.7 to 19.6 GB/s; with four buffers the pack is
    // rarely still running
    CUDA_CHECK(cudaEventSynchronize(p->packed[b]));
    CUDA_CHECK(cudaMemcpyAsync(p->scratch[b], src, size, cudaMemcpyHostToDevice, copy));
    CUDA_CHECK(cudaEventRecord(p->uploaded[b], copy));
    if (copied) {
        CUDA_CHECK(cudaEventRecord(copied, copy));
    }
    CUDA_CHECK(cudaStreamWaitEvent(p->stream, p->uploaded[b], 0));
    const int k = int(w->ne[0]), n = int(w->ne[1]);
    pack_tiles<<<dim3((n / 16) * (k / 128), unsigned(size / stride)), 256, 0, p->stream>>>(p->scratch[b], slot + offset, stride, k, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(p->packed[b], p->stream));
    if (done) {
        CUDA_CHECK(cudaEventRecord(done, p->stream));
    }
}

// A staging slot holds a new expert tensor in the canonical layout at every use, unless its upload packed it. Pack it in place
// on the compute stream so that the fused kernels read it; its state stays packed, because every later use is packed again.
void ggml_cuda_moe_int4_stage_prepare(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    if (t->type != GGML_TYPE_Q4_G128 || !supported_weight(t) || !ggml_cuda_moe_int4_type_supported(ctx.device)) { return; }
    // staged experts are the only G128 tensors of a model whose experts stay in host memory: enable the matcher
    have_g128.store(true, std::memory_order_relaxed);
    register_weight(ctx, t);
    const int64_t start = ggml_time_us();
    std::lock_guard<std::mutex> lock(state_mutex);
    weight_state * s = find_state(ctx.device, t->data);
    GGML_ASSERT(s && s->stride == size_t(t->nb[2]) && s->experts == t->ne[2]);
    GGML_ASSERT(s->k == t->ne[0] && s->n == t->ne[1]);
    const cudaStream_t stream = ctx.stream();
    if (staged_packed.erase({ctx.device, t->data}) > 0) {
        if (!s->packed) {
            const int flag = 1;
            CUDA_CHECK(cudaMemcpyAsync(s->flag, &flag, sizeof(flag), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream)); // once per slot: flag lives on the host stack
            s->packed = true;
        }
        packs++;
        pack_us += ggml_time_us() - start;
        return;
    }
    const int chunk = std::min(s->experts, 16);
    ggml_cuda_pool_alloc<char> tmp(ctx.pool(), chunk * s->stride);
    for (int e = 0; e < s->experts; e += chunk) {
        const int ce = std::min(chunk, s->experts - e);
        char * dst = s->data + e * s->stride;
        CUDA_CHECK(cudaMemcpyAsync(tmp.get(), dst, ce * s->stride, cudaMemcpyDeviceToDevice, stream));
        pack_tiles<<<dim3((s->n / 16) * (s->k / 128), ce), 256, 0, stream>>>(tmp.get(), dst, s->stride, s->k, s->n);
    }
    CUDA_CHECK(cudaGetLastError());
    if (!s->packed) {
        const int flag = 1;
        CUDA_CHECK(cudaMemcpyAsync(s->flag, &flag, sizeof(flag), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream)); // once per slot: flag lives on the host stack
        s->packed = true;
    }
    packs++;
    pack_us += ggml_time_us() - start;
}

void ggml_cuda_moe_int4_forget_inside(int device, const ggml_tensor * t) {
    std::lock_guard<std::mutex> lock(state_mutex);
    const char * base = static_cast<const char *>(t->data);
    for (auto it = states.begin(); it != states.end();) {
        if (it->device != device || it->data == base || it->data < base || it->data >= base + ggml_nbytes(t)) { ++it; continue; }
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(it->flag));
        it = states.erase(it);
    }
}

void ggml_backend_cuda_moe_int4_set_mode(int mode) { override_mode.store(mode < 0 ? -1 : mode != 0); }

void ggml_backend_cuda_moe_int4_get_stats(int64_t * out) {
    std::lock_guard<std::mutex> lock(state_mutex);
    for (int i = 0; i < 16; ++i) { out[i] = 0; }
    for (const auto & s : states) { if (s.packed) { out[0]++; out[3] += s.stride * s.experts; } }
    out[1] = packs; out[2] = pack_us; out[4] = runs; out[5] = scratch_bytes; out[6] = fallbacks; out[15] = wide_runs;
    out[13] = batch_runs; out[14] = batch_rows;
}

bool ggml_backend_cuda_moe_int4_get_pack(const void * data, int which, const void ** ptr, size_t * bytes, size_t * stride) {
    std::lock_guard<std::mutex> lock(state_mutex);
    const auto * s = find_state(ggml_cuda_get_device(), data);
    if (!s || !s->packed) { return false; }
    const size_t codes = size_t(s->k) * s->n / 2;
    *ptr = s->data + (which ? codes : 0); *bytes = which ? s->stride - codes : codes; *stride = s->stride;
    return true;
}

#endif // GGML_HIP_MOE_INT4_AVAILABLE
