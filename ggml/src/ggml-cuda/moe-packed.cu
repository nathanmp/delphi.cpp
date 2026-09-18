#include "moe-packed.cuh"
#include "moe-grouped.cuh"
#include "convert.cuh"
#include "cp-async.cuh"
#include "mmq.cuh"
#include "ggml-impl.h"

#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#ifdef GGML_CUDA_MOE_PACKED_AVAILABLE

#include "moe-packed-q.cuh"

#define MOE_PACKED_SA      16384 // weight bytes per row tile and k chunk
#define MOE_PACKED_SD      2048  // i8 scale bytes per row tile and k chunk
#define MOE_PACKED_SY      144   // shared memory bytes per column and k chunk: 64 f16 + 16 B padding, or one block_q8_1_mmq

template <int fmt> static constexpr __host__ __device__ int moe_packed_bk() { return fmt == GGML_CUDA_MOE_PACKED_F16 ? 64 : 128; }

// Weight bytes per shared memory stage (without the i8 scales). Lossless formats: one chunk of codes and the metadata of its QK_K block.
template <int fmt> static constexpr __host__ __device__ int moe_packed_sa() {
    if constexpr (moe_packed_is_q(fmt)) {
        using pq = moe_packed_q<fmt>;
        return pq::sb_bytes - pq::chunk_bytes + MOE_PACKED_I*moe_packed_row_meta_bytes<fmt>();
    } else {
        return MOE_PACKED_SA;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Packing

// One thread per lane of one A fragment (8 halves). src holds rows of ncols halves.
static __global__ void moe_pack_f16(const half * __restrict__ src, uint4 * __restrict__ dst, const int64_t n, const int nrows, const int ncols) {
    const int64_t p = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n) {
        return;
    }
    const int kc_n = ncols / 64;
    const int nt   = nrows / MOE_PACKED_I;

    int64_t q = p;
    const int lane = q % 32; q /= 32;
    const int ks   = q % 4;  q /= 4;
    const int mg   = q % 8;  q /= 8;
    const int kc   = q % kc_n; q /= kc_n;
    const int it   = q % nt;
    const int64_t e = q / nt;

    const int g = lane / 4;
    const int t = lane % 4;
    const half * r0 = src + (e*nrows + it*MOE_PACKED_I + mg*16 + g)*ncols + kc*64 + ks*16 + 2*t;
    const half * r1 = r0 + 8*ncols;

    alignas(16) half v[8] = { r0[0], r0[1], r1[0], r1[1], r0[8], r0[9], r1[8], r1[9] };
    dst[p] = *(const uint4 *) v;
}

// int8 code k (0..255) of one QK_K block, as MMQ builds it in shared memory.
template <ggml_type type>
static __device__ __forceinline__ int moe_pack_code(const void * vb, const int k) {
    if constexpr (type == GGML_TYPE_IQ4_XS) {
        const block_iq4_xs * b = (const block_iq4_xs *) vb;
        const int ib = k / 32;
        const int j  = k % 32;
        return kvalues_iq4nl[(b->qs[16*ib + j%16] >> (4*(j/16))) & 0xF];
    } else {
        static_assert(type == GGML_TYPE_IQ3_S, "bad type");
        const block_iq3_s * b = (const block_iq3_s *) vb;
        const int ib = k / 32;
        const int il = (k % 32) / 8;
        const int j  = k % 8;
        const int idx = b->qs[8*ib + 2*il + j/4] | ((b->qh[ib] << (8 - 2*il - j/4)) & 256);
        const int g = ((const uint8_t *) (iq3s_grid + idx))[j % 4];
        return b->signs[4*ib + il] & kmask_iq2xs[j] ? -g : g;
    }
}

template <ggml_type type>
static __device__ __forceinline__ float moe_pack_scale(const void * vb, const int ib) {
    if constexpr (type == GGML_TYPE_IQ4_XS) {
        const block_iq4_xs * b = (const block_iq4_xs *) vb;
        const int ls = ((b->scales_l[ib/2] >> (4*(ib%2))) & 0xF) | (((b->scales_h >> (2*ib)) & 3) << 4);
        return __half2float(b->d) * (ls - 32);
    } else {
        const block_iq3_s * b = (const block_iq3_s *) vb;
        const int ls = 1 + 2*((b->scales[ib/2] >> (4*(ib%2))) & 0xF);
        return ls * __half2float(b->d);
    }
}

// One thread per lane of one A fragment (16 int8 codes), plus the scales of the fragment rows.
template <ggml_type type, typename block_t>
static __global__ void moe_pack_i8(const char * __restrict__ src, const int64_t stride_row, uint4 * __restrict__ dst, float * __restrict__ dst_d,
        const int64_t n, const int nrows, const int ncols) {
    const int64_t p = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n) {
        return;
    }
    const int kc_n = ncols / 128;
    const int nt   = nrows / MOE_PACKED_I;

    int64_t q = p;
    const int lane = q % 32; q /= 32;
    const int ks   = q % 4;  q /= 4;
    const int mg   = q % 8;  q /= 8;
    const int kc   = q % kc_n; q /= kc_n;
    const int it   = q % nt;
    const int64_t e = q / nt;

    const int g = lane / 4;
    const int t = lane % 4;
    const int k32 = kc*128 + ks*32;       // first value of the k step
    const int kb  = k32 / QK_K;
    const int64_t row0 = e*nrows + it*MOE_PACKED_I + mg*16 + g;

    alignas(16) int8_t v[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const block_t * b = (const block_t *) (src + (row0 + 8*h)*stride_row) + kb;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[4*(h + 2*r) + i] = moe_pack_code<type>(b, k32 % QK_K + 16*r + 4*t + i);
            }
        }
        if (t == 0) {
            dst_d[p/32*16 + g + 8*h] = moe_pack_scale<type>(b, (k32 % QK_K) / 32);
        }
    }
    dst[p] = *(const uint4 *) v;
}

// One thread per lane and 16-row group (mg) of one superblock (row tile, QK_K block) of a lossless format: rows g and g + 8.
// Superblocks are numbered over all experts: the rows of src are contiguous across experts.
template <int fmt>
static __global__ void moe_pack_q(const char * __restrict__ src, const int64_t stride_row, char * __restrict__ dst, const int64_t n,
        const int nrows, const int ncols) {
    using pq      = moe_packed_q<fmt>;
    using block_t = typename pq::block_t;
    constexpr int n_sub = QK_K/ggml_cuda_type_traits<pq::type>::qk;
    const int64_t p = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n) {
        return;
    }
    const int kb_n = ncols / QK_K;
    const int lane = p % 32;
    const int mg   = (p / 32) % 8;
    const int64_t sb = p / 256;
    const int kb   = sb % kb_n;
    const int nt = nrows/MOE_PACKED_I;
    const int64_t it = (sb / kb_n) % nt;
    const int64_t e = (sb / kb_n) / nt;
    const int64_t row0 = (e*nrows + it*MOE_PACKED_I + mg*16 + lane/4);

    const block_t * b0 = (const block_t *) (src + row0*stride_row + moe_packed_row_meta_bytes<fmt>()) + kb*n_sub;
    const block_t * b1 = (const block_t *) (src + (row0 + 8)*stride_row + moe_packed_row_meta_bytes<fmt>()) + kb*n_sub;
    pq::pack(b0, b1, dst + e*nrows*stride_row + (it*kb_n + kb)*pq::sb_bytes, mg, lane);
}

template <int fmt>
static __global__ void moe_pack_q_row_meta(const char * __restrict__ src, const int64_t stride_row, char * __restrict__ dst,
        const int64_t n_rows, const int nrows, const int ncols) {
    using pq = moe_packed_q<fmt>;
    constexpr int rm = moe_packed_row_meta_bytes<fmt>();
    const int64_t p = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n_rows*rm) {
        return;
    }
    const int64_t row = p/rm;
    const int byte = p % rm;
    const int64_t e = row/nrows;
    const int r = row % nrows;
    const int64_t payload = (int64_t) nrows*(ncols/QK_K)*pq::sb_bytes/MOE_PACKED_I;
    dst[e*nrows*stride_row + payload + (int64_t) r*rm + byte] = src[row*stride_row + byte];
}

// One thread per GGUF block: rebuilds the block from the superblocks in packed and counts the bytes that differ from src.
template <int fmt>
static __global__ void moe_unpack_q_check(const char * __restrict__ packed, const char * __restrict__ src, const int64_t stride_row,
        const int64_t n, const int nrows, const int ncols, unsigned long long * __restrict__ n_bad) {
    using pq      = moe_packed_q<fmt>;
    using block_t = typename pq::block_t;
    constexpr int n_sub = QK_K/ggml_cuda_type_traits<pq::type>::qk;
    const int64_t p = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n) {
        return;
    }
    const int kb_n = ncols / QK_K;
    const int rem  = p % (MOE_PACKED_I*n_sub);
    const int r    = rem / n_sub; // row in the row tile
    const int ib   = rem % n_sub; // GGUF block within the QK_K span
    const int64_t sb = p / (MOE_PACKED_I*n_sub);
    const int kb   = sb % kb_n;
    const int nt = nrows/MOE_PACKED_I;
    const int64_t it = (sb/kb_n) % nt;
    const int64_t e = (sb/kb_n) / nt;

    block_t b;
    if constexpr (n_sub == 1) {
        pq::unpack(packed + e*nrows*stride_row + (it*kb_n + kb)*pq::sb_bytes, r, b);
    } else {
        pq::unpack(packed + e*nrows*stride_row + (it*kb_n + kb)*pq::sb_bytes, r, b, ib);
    }

    const uint8_t * ref = (const uint8_t *) ((const block_t *) (src + (e*nrows + it*MOE_PACKED_I + r)*stride_row + moe_packed_row_meta_bytes<fmt>()) + kb*n_sub + ib);
    const uint8_t * got = (const uint8_t *) &b;
    int bad = 0;
    for (int i = 0; i < (int) sizeof(block_t); ++i) {
        bad += got[i] != ref[i];
    }
    if (bad) {
        atomicAdd(n_bad, (unsigned long long) bad);
    }
}

template <int fmt>
static __global__ void moe_unpack_q_row_meta_check(const char * __restrict__ packed, const char * __restrict__ src,
        const int64_t stride_row, const int64_t n_rows, const int nrows, const int ncols, unsigned long long * __restrict__ n_bad) {
    using pq = moe_packed_q<fmt>;
    constexpr int rm = moe_packed_row_meta_bytes<fmt>();
    const int64_t p = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n_rows*rm) {
        return;
    }
    const int64_t row = p/rm;
    const int byte = p % rm;
    const int64_t e = row/nrows;
    const int r = row % nrows;
    const int64_t payload = (int64_t) nrows*(ncols/QK_K)*pq::sb_bytes/MOE_PACKED_I;
    if (packed[e*nrows*stride_row + payload + (int64_t) r*rm + byte] != src[row*stride_row + byte]) {
        atomicAdd(n_bad, 1ULL);
    }
}

struct moe_packed_entry {
    int         device;
    int         fmt;
    const void * src;
    ggml_type   type;
    int64_t     ne[3];
    char      * w;
    float     * d;
    int64_t     stride_w;
    int64_t     stride_d;
    bool        inplace; // w is src: the tensor data was overwritten with the packed layout
};

static std::mutex                    moe_packed_mutex;
static std::vector<moe_packed_entry> moe_packed_cache;
static std::atomic<bool>             moe_packed_any_inplace{false};

// The lossless format of a GGUF type, GGML_CUDA_MOE_PACKED_NONE if it has none.
static int moe_packed_q_fmt(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:   return GGML_CUDA_MOE_PACKED_Q80;
        case GGML_TYPE_Q5_0:   return GGML_CUDA_MOE_PACKED_Q50;
        case GGML_TYPE_IQ4_NL: return GGML_CUDA_MOE_PACKED_IQ4NL;
        case GGML_TYPE_MXFP4:  return GGML_CUDA_MOE_PACKED_MXFP4;
        case GGML_TYPE_Q5_K:   return GGML_CUDA_MOE_PACKED_Q5K;
        case GGML_TYPE_IQ3_S:  return GGML_CUDA_MOE_PACKED_IQ3S;
        case GGML_TYPE_IQ4_XS: return GGML_CUDA_MOE_PACKED_IQ4XS;
        case GGML_TYPE_Q4_K:   return GGML_CUDA_MOE_PACKED_Q4K;
        case GGML_TYPE_Q6_K:   return GGML_CUDA_MOE_PACKED_Q6K;
        case GGML_TYPE_Q4_1:   return GGML_CUDA_MOE_PACKED_Q41;
        case GGML_TYPE_Q5_1:   return GGML_CUDA_MOE_PACKED_Q51;
        case GGML_TYPE_Q3_K:   return GGML_CUDA_MOE_PACKED_Q3K;
        case GGML_TYPE_IQ3_XXS:return GGML_CUDA_MOE_PACKED_IQ3XXS;
        case GGML_TYPE_IQ2_XXS:return GGML_CUDA_MOE_PACKED_IQ2XXS;
        case GGML_TYPE_IQ2_XS: return GGML_CUDA_MOE_PACKED_IQ2XS;
        case GGML_TYPE_IQ2_S:  return GGML_CUDA_MOE_PACKED_IQ2S;
        case GGML_TYPE_IQ2_K:  return GGML_CUDA_MOE_PACKED_IQ2K;
        case GGML_TYPE_IQ3_K:  return GGML_CUDA_MOE_PACKED_IQ3K;
        case GGML_TYPE_IQ4_K:  return GGML_CUDA_MOE_PACKED_IQ4K;
        case GGML_TYPE_IQ5_K:  return GGML_CUDA_MOE_PACKED_IQ5K;
        case GGML_TYPE_IQ2_KS: return GGML_CUDA_MOE_PACKED_IQ2KS;
        case GGML_TYPE_IQ3_KS: return GGML_CUDA_MOE_PACKED_IQ3KS;
        case GGML_TYPE_IQ4_KS: return GGML_CUDA_MOE_PACKED_IQ4KS;
        case GGML_TYPE_IQ5_KS: return GGML_CUDA_MOE_PACKED_IQ5KS;
        default:               return GGML_CUDA_MOE_PACKED_NONE;
    }
}

// GGML_CUDA_MOE_PACKED_TYPES: the GGUF types that the packed mode packs, as a comma-separated list of type names (default: all).
static bool moe_packed_type_enabled(ggml_type type) {
    static const std::string types = [] {
        const char * s = getenv("GGML_CUDA_MOE_PACKED_TYPES");
        return s ? "," + std::string(s) + "," : std::string();
    }();
    return types.empty() || types.find("," + std::string(ggml_type_name(type)) + ",") != std::string::npos;
}

static bool moe_packed_supported(int fmt, const ggml_tensor * src0) {
    if (!ggml_is_contiguous(src0) || src0->ne[3] != 1 || src0->ne[1] % MOE_PACKED_I != 0) {
        return false;
    }
    if (fmt == GGML_CUDA_MOE_PACKED_F16) {
        return src0->ne[0] % 64 == 0 && ggml_get_to_fp16_cuda(src0->type) != nullptr;
    }
    if (fmt == GGML_CUDA_MOE_PACKED_I8) {
        return src0->ne[0] % QK_K == 0 && (src0->type == GGML_TYPE_IQ3_S || src0->type == GGML_TYPE_IQ4_XS);
    }
    if (moe_packed_is_q(fmt)) {
        return src0->ne[0] % QK_K == 0 && moe_packed_q_fmt(src0->type) == fmt;
    }
    return false;
}

static bool moe_packed_env_flag(const char * name) {
    const char * s = getenv(name);
    return s != nullptr && atoi(s) != 0;
}

static bool moe_packed_nocache() {
    static const bool nocache = moe_packed_env_flag("GGML_CUDA_MOE_PACKED_NOCACHE");
    return nocache;
}

static std::atomic<int> moe_packed_inplace_override{-1};

static bool moe_packed_inplace() {
    static const bool env = moe_packed_env_flag("GGML_CUDA_MOE_PACKED_INPLACE");
    const int o = moe_packed_inplace_override.load(std::memory_order_relaxed);
    return o >= 0 ? o != 0 : env;
}

int ggml_cuda_moe_packed_select(const ggml_tensor * src0, bool warn) {
    int fmt;
    switch (ggml_cuda_moe_grouped_packed_mode()) {
        case GGML_CUDA_MOE_GROUPED_PACKED_F16: fmt = GGML_CUDA_MOE_PACKED_F16; break;
        case GGML_CUDA_MOE_GROUPED_PACKED_I8:  fmt = GGML_CUDA_MOE_PACKED_I8;  break;
        case GGML_CUDA_MOE_GROUPED_PACKED:
            // the lossless format of the type; other types use grouped MMQ without a warning
            fmt = moe_packed_type_enabled(src0->type) ? moe_packed_q_fmt(src0->type) : GGML_CUDA_MOE_PACKED_NONE;
            return fmt != GGML_CUDA_MOE_PACKED_NONE && moe_packed_supported(fmt, src0) ? fmt : GGML_CUDA_MOE_PACKED_NONE;
        default:
            return GGML_CUDA_MOE_PACKED_NONE;
    }
    if (!moe_packed_supported(fmt, src0)) {
        if (warn) {
            GGML_LOG_WARN("%s: no packed path for %s [%" PRId64 ", %" PRId64 "], using grouped MMQ\n", __func__,
                ggml_type_name(src0->type), src0->ne[0], src0->ne[1]);
        }
        return GGML_CUDA_MOE_PACKED_NONE;
    }
    return fmt;
}

static bool moe_packed_verify() {
    static const bool verify = moe_packed_env_flag("GGML_CUDA_MOE_PACKED_VERIFY");
    return verify;
}

static const char * moe_packed_fmt_name(int fmt) {
    switch (fmt) {
        case GGML_CUDA_MOE_PACKED_Q80:  return "q8_0";
        case GGML_CUDA_MOE_PACKED_Q50:  return "q5_0";
        case GGML_CUDA_MOE_PACKED_IQ4NL:return "iq4_nl";
        case GGML_CUDA_MOE_PACKED_MXFP4:return "mxfp4";
        case GGML_CUDA_MOE_PACKED_Q5K:  return "q5_k";
        case GGML_CUDA_MOE_PACKED_F16:  return "f16";
        case GGML_CUDA_MOE_PACKED_I8:   return "i8";
        case GGML_CUDA_MOE_PACKED_IQ3S:  return "iq3s";
        case GGML_CUDA_MOE_PACKED_IQ4XS: return "iq4xs";
        case GGML_CUDA_MOE_PACKED_Q4K:   return "q4k";
        case GGML_CUDA_MOE_PACKED_Q6K:   return "q6k";
        case GGML_CUDA_MOE_PACKED_Q41:   return "q4_1";
        case GGML_CUDA_MOE_PACKED_Q51:   return "q5_1";
        case GGML_CUDA_MOE_PACKED_Q3K:   return "q3_k";
        case GGML_CUDA_MOE_PACKED_IQ3XXS:return "iq3_xxs";
        case GGML_CUDA_MOE_PACKED_IQ2XXS:return "iq2_xxs";
        case GGML_CUDA_MOE_PACKED_IQ2XS: return "iq2_xs";
        case GGML_CUDA_MOE_PACKED_IQ2S:  return "iq2_s";
        case GGML_CUDA_MOE_PACKED_IQ2K:  return "iq2_k";
        case GGML_CUDA_MOE_PACKED_IQ3K:  return "iq3_k";
        case GGML_CUDA_MOE_PACKED_IQ4K:  return "iq4_k";
        case GGML_CUDA_MOE_PACKED_IQ5K:  return "iq5_k";
        case GGML_CUDA_MOE_PACKED_IQ2KS: return "iq2_ks";
        case GGML_CUDA_MOE_PACKED_IQ3KS: return "iq3_ks";
        case GGML_CUDA_MOE_PACKED_IQ4KS: return "iq4_ks";
        case GGML_CUDA_MOE_PACKED_IQ5KS: return "iq5_ks";
        default:                         return "none";
    }
}

// Packs the rows of n_experts experts from src into dst in the lossless format fmt. With n_bad, the packed blocks are unpacked again
// and the bytes that differ from src are counted.
template <int fmt>
static void moe_packed_build_q_fmt(const char * src, char * dst, int64_t n_experts, int64_t stride_row, int64_t nrows, int64_t ncols,
        unsigned long long * n_bad, cudaStream_t stream) {
    using pq = moe_packed_q<fmt>;
    const int64_t n_blocks = n_experts*nrows*(ncols/QK_K);
    const int64_t n        = n_blocks/MOE_PACKED_I*256;
    moe_pack_q<fmt><<<(n + 255)/256, 256, 0, stream>>>(src, stride_row, dst, n, (int) nrows, (int) ncols);
    CUDA_CHECK(cudaGetLastError());
    if constexpr (moe_packed_row_meta_bytes<fmt>() != 0) {
        const int64_t n_meta = n_experts*nrows*moe_packed_row_meta_bytes<fmt>();
        moe_pack_q_row_meta<fmt><<<(n_meta + 255)/256, 256, 0, stream>>>(src, stride_row, dst, n_experts*nrows, (int) nrows, (int) ncols);
        CUDA_CHECK(cudaGetLastError());
    }
    if (n_bad) {
        constexpr int n_sub = QK_K/ggml_cuda_type_traits<pq::type>::qk;
        moe_unpack_q_check<fmt><<<(n_blocks*n_sub + 255)/256, 256, 0, stream>>>(dst, src, stride_row, n_blocks*n_sub, (int) nrows,
            (int) ncols, n_bad);
        CUDA_CHECK(cudaGetLastError());
        if constexpr (moe_packed_row_meta_bytes<fmt>() != 0) {
            const int64_t n_meta = n_experts*nrows*moe_packed_row_meta_bytes<fmt>();
            moe_unpack_q_row_meta_check<fmt><<<(n_meta + 255)/256, 256, 0, stream>>>(dst, src, stride_row, n_experts*nrows,
                (int) nrows, (int) ncols, n_bad);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

static void moe_packed_build_q(int fmt, const char * src, char * dst, int64_t n_experts, int64_t stride_row, int64_t nrows, int64_t ncols,
        unsigned long long * n_bad, cudaStream_t stream) {
    switch (fmt) {
        case GGML_CUDA_MOE_PACKED_Q80:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_Q80>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_Q50:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_Q50>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ4NL:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ4NL>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_MXFP4:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_MXFP4>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_Q5K:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_Q5K>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ3S:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ3S>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ4XS:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ4XS>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_Q4K:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_Q4K>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_Q6K:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_Q6K>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_Q41:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_Q41>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_Q51:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_Q51>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_Q3K:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_Q3K>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ3XXS:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ3XXS>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ2XXS:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ2XXS>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ2XS:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ2XS>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ2S:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ2S>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ2K:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ2K>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ3K:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ3K>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ4K:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ4K>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ5K:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ5K>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ2KS:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ2KS>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ3KS:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ3KS>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ4KS:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ4KS>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        case GGML_CUDA_MOE_PACKED_IQ5KS:
            moe_packed_build_q_fmt<GGML_CUDA_MOE_PACKED_IQ5KS>(src, dst, n_experts, stride_row, nrows, ncols, n_bad, stream);
            break;
        default:
            GGML_ABORT("packed MoE: %d is not a lossless format", fmt);
    }
}

// f16 dequantizes tmp_experts experts at a time into tmp.
static void moe_packed_build(int fmt, const ggml_tensor * src0, char * w, float * d, half * tmp, int64_t tmp_experts, cudaStream_t stream) {
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = src0->ne[1];
    const int64_t ne2   = src0->ne[2];
    if (fmt == GGML_CUDA_MOE_PACKED_F16) {
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(src0->type);
        for (int64_t e0 = 0; e0 < ne2; e0 += tmp_experts) {
            const int64_t ce = std::min(tmp_experts, ne2 - e0);
            to_fp16((const char *) src0->data + e0*src0->nb[2], tmp, ce*nrows, ncols, stream);
            const int64_t n = ce*nrows*ncols/8;
            moe_pack_f16<<<(n + 255)/256, 256, 0, stream>>>(tmp, (uint4 *) (w + e0*nrows*ncols*sizeof(half)), n, nrows, ncols);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        const int64_t n = ne2*nrows*ncols/16;
        if (src0->type == GGML_TYPE_IQ3_S) {
            moe_pack_i8<GGML_TYPE_IQ3_S, block_iq3_s><<<(n + 255)/256, 256, 0, stream>>>
                ((const char *) src0->data, src0->nb[1], (uint4 *) w, d, n, nrows, ncols);
        } else {
            moe_pack_i8<GGML_TYPE_IQ4_XS, block_iq4_xs><<<(n + 255)/256, 256, 0, stream>>>
                ((const char *) src0->data, src0->nb[1], (uint4 *) w, d, n, nrows, ncols);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

moe_packed_weights ggml_cuda_moe_packed_get(ggml_cuda_pool_alloc<char> & storage, int fmt, const ggml_tensor * src0, cudaStream_t stream) {
    const bool    is_q     = moe_packed_is_q(fmt);
    const int64_t ncols    = src0->ne[0];
    const int64_t nrows    = src0->ne[1];
    const int64_t ne2      = src0->ne[2];
    const int64_t stride_w = is_q ? (int64_t) src0->nb[2] : nrows*ncols*(fmt == GGML_CUDA_MOE_PACKED_F16 ? sizeof(half) : 1);
    const int64_t stride_d = fmt == GGML_CUDA_MOE_PACKED_I8 ? nrows*ncols/32 : 0;
    const size_t  nbytes_w = ne2*stride_w;
    const size_t  nbytes_d = ne2*stride_d*sizeof(float);

    if (moe_packed_nocache()) {
        GGML_ASSERT(!moe_packed_inplace() && "GGML_CUDA_MOE_PACKED_NOCACHE and GGML_CUDA_MOE_PACKED_INPLACE exclude each other");
        // w, d, then the fp16 dequantization of all experts
        const size_t nbytes_tmp = fmt == GGML_CUDA_MOE_PACKED_F16 ? nbytes_w : 0;
        char * p = storage.alloc(nbytes_w + nbytes_d + nbytes_tmp);
        if (is_q) {
            moe_packed_build_q(fmt, (const char *) src0->data, p, ne2, src0->nb[1], nrows, ncols, nullptr, stream);
        } else {
            moe_packed_build(fmt, src0, p, (float *) (p + nbytes_w), (half *) (p + nbytes_w + nbytes_d), ne2, stream);
        }
        return { p, (const float *) (p + nbytes_w), stride_w, stride_d };
    }

    const int device = ggml_cuda_get_device();
    std::lock_guard<std::mutex> lock(moe_packed_mutex);
    for (const moe_packed_entry & c : moe_packed_cache) {
        if (c.device != device || c.src != src0->data) {
            continue;
        }
        // data packed in place can only be read in its own format
        GGML_ASSERT(!c.inplace || c.fmt == fmt);
        if (c.fmt == fmt && c.type == src0->type && c.ne[0] == src0->ne[0] && c.ne[1] == src0->ne[1] && c.ne[2] == src0->ne[2]) {
            return { c.w, c.d, c.stride_w, c.stride_d };
        }
    }

    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    GGML_ASSERT(status == cudaStreamCaptureStatusNone && "packed MoE weights must be built before graph capture");

    const bool inplace = is_q && moe_packed_inplace();
    moe_packed_entry c = { device, fmt, src0->data, src0->type, { ncols, nrows, ne2 }, nullptr, nullptr, stride_w, stride_d, inplace };
    const int64_t t0 = ggml_time_us();

    unsigned long long * n_bad = nullptr;
    if (is_q && moe_packed_verify()) {
        CUDA_CHECK(cudaMallocAsync(&n_bad, sizeof(*n_bad), stream));
        CUDA_CHECK(cudaMemsetAsync(n_bad, 0, sizeof(*n_bad), stream));
    }
    if (inplace) {
        // an expert keeps its size, so each expert is packed from a temporary copy into its own bytes
        const int64_t chunk = 16; // experts per temporary copy
        char * tmp = nullptr;
        CUDA_CHECK(cudaMallocAsync(&tmp, std::min(chunk, ne2)*stride_w, stream));
        c.w = (char *) src0->data;
        for (int64_t e0 = 0; e0 < ne2; e0 += chunk) {
            const int64_t ce = std::min(chunk, ne2 - e0);
            CUDA_CHECK(cudaMemcpyAsync(tmp, c.w + e0*stride_w, ce*stride_w, cudaMemcpyDeviceToDevice, stream));
            moe_packed_build_q(fmt, tmp, c.w + e0*stride_w, ce, src0->nb[1], nrows, ncols, n_bad, stream);
        }
        CUDA_CHECK(cudaFreeAsync(tmp, stream));
    } else {
        CUDA_CHECK(cudaMalloc(&c.w, nbytes_w));
        if (nbytes_d > 0) {
            CUDA_CHECK(cudaMalloc(&c.d, nbytes_d));
        }
        if (is_q) {
            moe_packed_build_q(fmt, (const char *) src0->data, c.w, ne2, src0->nb[1], nrows, ncols, n_bad, stream);
        } else {
            const int64_t chunk = 8; // experts per fp16 dequantization
            half * tmp = nullptr;
            if (fmt == GGML_CUDA_MOE_PACKED_F16) {
                CUDA_CHECK(cudaMallocAsync(&tmp, chunk*stride_w, stream));
            }
            moe_packed_build(fmt, src0, c.w, c.d, tmp, chunk, stream);
            if (tmp) {
                CUDA_CHECK(cudaFreeAsync(tmp, stream));
            }
        }
    }
    unsigned long long h_bad = 0;
    if (n_bad) {
        CUDA_CHECK(cudaMemcpyAsync(&h_bad, n_bad, sizeof(h_bad), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaFreeAsync(n_bad, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_bad != 0) {
        GGML_ABORT("%s: %llu bytes of the unpacked %s blocks of %s differ from the source", __func__, h_bad, ggml_type_name(src0->type),
            src0->name);
    }
    GGML_LOG_INFO("%s: CUDA%d packed %s %s %s [%" PRId64 ", %" PRId64 ", %" PRId64 "] %s %.1f MiB in %.1f ms%s\n", __func__, device,
        moe_packed_fmt_name(fmt), ggml_type_name(src0->type), src0->name, ncols, nrows, ne2, inplace ? "in place," : "into",
        (ne2*c.stride_w + ne2*c.stride_d*sizeof(float))/1048576.0, (ggml_time_us() - t0)/1000.0,
        n_bad ? ", unpacked blocks identical to the source" : "");

    moe_packed_cache.push_back(c);
    if (inplace) {
        moe_packed_any_inplace.store(true, std::memory_order_relaxed);
    }
    return { c.w, c.d, c.stride_w, c.stride_d };
}

// An entry of a tensor packed in place stays: its data is still in the packed layout.
void ggml_backend_cuda_moe_packed_clear(void) {
    std::lock_guard<std::mutex> lock(moe_packed_mutex);
    std::vector<moe_packed_entry> keep;
    for (const moe_packed_entry & c : moe_packed_cache) {
        if (c.inplace) {
            keep.push_back(c);
            continue;
        }
        ggml_cuda_set_device(c.device);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(c.w));
        if (c.d) {
            CUDA_CHECK(cudaFree(c.d));
        }
    }
    moe_packed_cache = std::move(keep);
}

void ggml_cuda_moe_packed_forget(int device, const void * base, size_t size) {
    std::lock_guard<std::mutex> lock(moe_packed_mutex);
    const char * lo = (const char *) base;
    for (size_t i = 0; i < moe_packed_cache.size();) {
        const moe_packed_entry & c = moe_packed_cache[i];
        const char * p = (const char *) c.src;
        if (c.device != device || p < lo || p >= lo + size) {
            ++i;
            continue;
        }
        if (!c.inplace) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaFree(c.w));
            if (c.d) {
                CUDA_CHECK(cudaFree(c.d));
            }
        }
        moe_packed_cache.erase(moe_packed_cache.begin() + i);
    }
}

int ggml_cuda_moe_packed_inplace_fmt(const ggml_tensor * src0) {
    if (!moe_packed_any_inplace.load(std::memory_order_relaxed)) {
        return GGML_CUDA_MOE_PACKED_NONE;
    }
    const int device = ggml_cuda_get_device();
    std::lock_guard<std::mutex> lock(moe_packed_mutex);
    for (const moe_packed_entry & c : moe_packed_cache) {
        if (c.inplace && c.device == device && c.src == src0->data) {
            return c.fmt;
        }
    }
    return GGML_CUDA_MOE_PACKED_NONE;
}

void ggml_cuda_moe_packed_check_not_inplace(const ggml_tensor * src0) {
    if (ggml_cuda_moe_packed_inplace_fmt(src0) != GGML_CUDA_MOE_PACKED_NONE) {
        GGML_ABORT("%s: %s was packed in place (GGML_CUDA_MOE_PACKED_INPLACE), and this path reads only the GGUF layout", __func__, src0->name);
    }
}

void ggml_cuda_moe_packed_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    if (ggml_cuda_moe_grouped_packed_mode() == GGML_CUDA_MOE_GROUPED_OLD || moe_packed_nocache()) {
        return;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(ctx.device);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        // weights only: a tensor that a graph computes, or a view, has no final data yet
        const ggml_tensor * src0 = node->src[0];
        if (src0->op != GGML_OP_NONE || src0->view_src != nullptr || src0->buffer == nullptr ||
                ggml_backend_buffer_get_type(src0->buffer) != buft ||
                ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            continue;
        }
        const int fmt = ggml_cuda_moe_packed_select(src0, /*warn =*/ false);
        if (fmt != GGML_CUDA_MOE_PACKED_NONE) {
            ggml_cuda_pool_alloc<char> storage(ctx.pool());
            ggml_cuda_moe_packed_get(storage, fmt, src0, ctx.stream());
        }
    }
}

void ggml_backend_cuda_moe_packed_set_inplace(int inplace) {
    moe_packed_inplace_override.store(inplace < 0 ? -1 : inplace != 0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------------------------------
// Activations

static __global__ void moe_packed_y_f16_kernel(const float * __restrict__ x, const int32_t * __restrict__ ids, half * __restrict__ y,
        const int64_t ne10, const int64_t stride_row, const int n_expert_used, const bool scatter) {
    const int64_t i0 = 4*((int64_t) blockIdx.y*blockDim.x + threadIdx.x);
    if (i0 >= ne10) {
        return;
    }
    const int64_t src_row = scatter ? blockIdx.x : ids[blockIdx.x];
    const float4 v = *(const float4 *) (x + src_row*stride_row + i0);
    alignas(8) const half2 h[2] = { __floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w) };
    const int n = scatter ? n_expert_used : 1;
    for (int k = 0; k < n; ++k) {
        const int64_t dst_row = scatter ? ids[(int64_t) blockIdx.x*n_expert_used + k] : blockIdx.x;
        *(uint2 *) (y + dst_row*ne10 + i0) = *(const uint2 *) h;
    }
}

void ggml_cuda_moe_packed_y_f16(const float * x, const int32_t * ids, half * y, int64_t ne10, int64_t stride_row,
        int64_t n_src_rows, int n_expert_used, bool scatter, cudaStream_t stream) {
    GGML_ASSERT(ne10 % 4 == 0 && stride_row % 4 == 0);
    const dim3 grid(n_src_rows, (ne10/4 + 255)/256);
    moe_packed_y_f16_kernel<<<grid, 256, 0, stream>>>(x, ids, y, ne10, stride_row, n_expert_used, scatter);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// Kernel

static __device__ __forceinline__ void moe_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif
}

template <int n>
static __device__ __forceinline__ void moe_cp_async_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif
}

static int moe_packed_alias() {
    static const int alias = [] {
        const char * s = getenv("GGML_CUDA_MOE_PACKED_ALIAS");
        return s ? atoi(s) : 0;
    }();
    return alias;
}

static bool moe_packed_rowtile_outer() {
    static const bool outer = moe_packed_env_flag("GGML_CUDA_MOE_PACKED_ROWTILE_OUTER");
    return outer;
}

// int32 mma sum -> fp32. magic: an add of 1.5*2^23 instead of I2F, exact for |c| < 2^22 (32 int8 products: |c| <= 32*128*127).
template <bool magic>
static __device__ __forceinline__ float moe_packed_cvt(const int c) {
    return magic ? __int_as_float(c + 0x4B400000) - 12582912.0f : (float) c;
}

// Block: 128 rows x J columns, NW warps as NWR x NWC, each warp 16*MW rows x J/NWC columns, ST pipeline stages.
// The (unit, k chunk) iterations of a block form one cp.async pipeline, so a new unit does not drain it.
// A unit is (row tile, list entry). The row tiles of one entry are adjacent units, so its activations stay in L2;
// rowtile_outer gives the other order (each activation column is read from HBM once per row tile).
// magic: moe_packed_cvt converts the int32 sums without I2F.
template <int fmt, int J, int NW, int MW, int ST, bool magic>
__launch_bounds__(NW*WARP_SIZE, 1)
static __global__ void moe_packed_mm(const moe_packed_launch_args a, const int alias, const bool rowtile_outer) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    using pq = moe_packed_q<fmt>;
    constexpr bool is_i8 = fmt == GGML_CUDA_MOE_PACKED_I8;
    constexpr bool is_q  = moe_packed_is_q(fmt);
    constexpr bool y_q8  = fmt != GGML_CUDA_MOE_PACKED_F16; // activations as block_q8_1_mmq, else fp16 rows
    constexpr int  BK    = moe_packed_bk<fmt>();
    constexpr int  NWR   = MOE_PACKED_I / (16*MW);
    constexpr int  NWC   = NW / NWR;
    static_assert(NWR*NWC == NW, "bad warp layout");
    constexpr int  JW    = J / NWC;
    constexpr int  NI    = JW / 8;
    constexpr int  SA    = moe_packed_sa<fmt>();
    constexpr int  SA_G  = is_q ? pq::sb_bytes/2 : MOE_PACKED_SA; // weight bytes per row tile and k chunk in global memory
    constexpr int  SD    = is_i8 ? MOE_PACKED_SD : 0;
    constexpr int  RM    = is_q ? MOE_PACKED_I*moe_packed_row_meta_bytes<fmt>() : 0;
    constexpr int  SSTAGE = SA + SD + J*MOE_PACKED_SY;
    constexpr int  YOFF   = y_q8 ? 16 : 0; // qs of block_q8_1_mmq follow the scales
    static_assert(JW % 8 == 0, "bad tile width");

    extern __shared__ __align__(16) char smem[];
    // A KS row scale applies to every QK_K block of one row tile. Cache one plane for each in-flight unit and copy it into the
    // ordinary stage tail only after its async load has completed; this avoids rereading the global tail for every k chunk.
    __shared__ __align__(16) char row_meta_cache[RM ? ST*RM : 1];

    // the table of the format, visible after the first __syncthreads of the loop
    __shared__ uint32_t tab_s[pq::tab_words > 0 ? pq::tab_words : 1];
    if constexpr (pq::tab_words > 0) {
        for (int i = threadIdx.x; i < pq::tab_words; i += NW*WARP_SIZE) {
            tab_s[i] = pq::tab(i);
        }
    }

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int wr   = warp % NWR;
    const int wc   = warp / NWR;
    const int g    = lane / 4;
    const int t    = lane % 4;

    const int n_tiles = *a.n_tiles;
    const int nt      = a.nrows / MOE_PACKED_I;
    const int kc_n    = a.ncols_k / BK;
    const int n_units = n_tiles*nt;
    const int n_mine  = (int) blockIdx.x < n_units ? (n_units - 1 - blockIdx.x) / gridDim.x + 1 : 0;
    const int n_iter  = n_mine*kc_n;

    // Row tile of unit u; the list entry is returned in tile.
    auto unit = [&](const int u, int & tile) {
        const int it = rowtile_outer ? u / n_tiles : u % nt;
        tile = rowtile_outer ? u - it*n_tiles : u / nt;
        return it;
    };

    // Source pointers of the unit that is loading, at k chunk 0.
    int           ld_q = -1;
    const char  * ld_w = nullptr;
    const char  * ld_meta = nullptr;
    const float * ld_d = nullptr;
    const char  * ld_y = nullptr;

    auto load_stage = [&](const int iter) {
        const int q  = iter / kc_n;
        const int kc = iter - q*kc_n;
        if (q != ld_q) {
            int tile;
            const int  it = unit(blockIdx.x + q*gridDim.x, tile);
            const int2 tl = a.tiles[tile];
            const int  col_low = a.expert_bounds[tl.x] + tl.y;
            const int64_t e  = alias >= 1 ? 0 : tl.x;
            const int64_t iw = (int64_t) (alias >= 2 ? 0 : it)*kc_n;
            ld_w = a.w.w + e*a.w.stride_w + iw*SA_G;
            if constexpr (is_q && moe_packed_row_meta_bytes<fmt>() != 0) {
                // kc_n counts BK=128-value k-chunks, two per QK_K superblock; the payload plane is sized in superblocks.
                const int64_t payload = (int64_t) a.nrows*(a.ncols_k/QK_K)*pq::sb_bytes/MOE_PACKED_I;
                ld_meta = a.w.w + e*a.w.stride_w + payload + (int64_t) it*MOE_PACKED_I*moe_packed_row_meta_bytes<fmt>();
            }
            ld_d = is_i8 ? a.w.d + e*a.w.stride_d + iw*(MOE_PACKED_SD/sizeof(float)) : nullptr;
            ld_y = a.y + (int64_t) col_low*(y_q8 ? (int64_t) sizeof(block_q8_1_mmq) : a.ncols_k*(int64_t) sizeof(half));
            ld_q = q;
        }

        char * s = smem + (iter % ST)*SSTAGE;
        if constexpr (is_q) {
            // the codes of the chunk, then the metadata of its QK_K block and, for IQ*_KS, the shared row scale plane
            constexpr int CH = pq::chunk_bytes;
            constexpr int LM = pq::sb_bytes - 2*CH;
            const char * srcb = ld_w + (int64_t) (kc/2)*pq::sb_bytes;
            const char * srcc = srcb + (kc % 2)*CH;
            for (int c = threadIdx.x; c < SA/16; c += NW*WARP_SIZE) {
                const char * src;
                if (c < CH/16) {
                    src = srcc + 16*c;
                } else if (c < (CH + LM)/16) {
                    src = srcb + 2*CH + 16*(c - CH/16);
                } else {
                    continue;
                }
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(s + 16*c), src);
            }
            if constexpr (RM != 0) {
                if (kc == 0) {
                    char * cache = row_meta_cache + (q % ST)*RM;
                    for (int c = threadIdx.x; c < RM/16; c += NW*WARP_SIZE) {
                        cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(cache + 16*c), ld_meta + 16*c);
                    }
                }
            }
        } else {
            const char * srcw = ld_w + (int64_t) kc*MOE_PACKED_SA;
#pragma unroll
            for (int c = threadIdx.x; c < MOE_PACKED_SA/16; c += NW*WARP_SIZE) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(s + 16*c), srcw + 16*c);
            }
        }
        if constexpr (is_i8) {
            const float * srcd = ld_d + (int64_t) kc*(MOE_PACKED_SD/sizeof(float));
            if (threadIdx.x < MOE_PACKED_SD/16) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(s + MOE_PACKED_SA + 16*threadIdx.x), srcd + 4*threadIdx.x);
            }
        }
        char * sy = s + SA + SD;
        const char * srcy = ld_y + (y_q8 ? (int64_t) kc*a.ncols_y*(int64_t) sizeof(block_q8_1_mmq) : (int64_t) kc*BK*(int64_t) sizeof(half));
        const int64_t stride_y = y_q8 ? (int64_t) sizeof(block_q8_1_mmq) : a.ncols_k*(int64_t) sizeof(half);
        constexpr int nchunk = y_q8 ? MOE_PACKED_SY/16 : BK*2/16;
        for (int c = threadIdx.x; c < J*nchunk; c += NW*WARP_SIZE) {
            const int j    = c / nchunk;
            const int part = c - j*nchunk;
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(sy + j*MOE_PACKED_SY + 16*part), srcy + j*stride_y + 16*part);
        }
    };

    float acc[MW][NI][4];
#pragma unroll
    for (int mi = 0; mi < MW; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[mi][ni][l] = 0.0f;
            }
        }
    }

#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < n_iter) {
            load_stage(s);
        }
        moe_cp_async_commit();
    }

    for (int iter = 0; iter < n_iter; ++iter) {
        moe_cp_async_wait<ST - 2>();
        __syncthreads();
        if constexpr (RM != 0) {
            constexpr int CH = pq::chunk_bytes;
            constexpr int LM = pq::sb_bytes - 2*CH;
            char * s_meta = smem + (iter % ST)*SSTAGE + CH + LM;
            const char * cache = row_meta_cache + ((iter/kc_n) % ST)*RM;
            for (int c = threadIdx.x; c < RM/16; c += NW*WARP_SIZE) {
                *(uint4 *) (s_meta + 16*c) = *(const uint4 *) (cache + 16*c);
            }
            __syncthreads();
        }
        if (iter + ST - 1 < n_iter) {
            load_stage(iter + ST - 1);
        }
        moe_cp_async_commit();

        const char  * s  = smem + (iter % ST)*SSTAGE;
        const float * sd = (const float *) (s + SA);
        const char  * sy = s + SA + SD;
        const unsigned int sy_u = ggml_cuda_cvta_generic_to_shared((void *) sy);

        // lossless formats: the metadata of rows g and g + 8 of each 16-row group of the warp
        [[maybe_unused]] typename pq::frag_state st[MW];
        if constexpr (is_q) {
#pragma unroll
            for (int mi = 0; mi < MW; ++mi) {
                st[mi] = pq::load_state(s, wr*MW + mi, lane, iter % kc_n);
            }
        }

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint4 af[MW];
#pragma unroll
            for (int mi = 0; mi < MW; ++mi) {
                if constexpr (is_q) {
                    af[mi] = pq::fragment(s, tab_s, st[mi], wr*MW + mi, ks, lane);
                } else {
                    af[mi] = *(const uint4 *) (s + (((wr*MW + mi)*4 + ks)*WARP_SIZE + lane)*16);
                }
            }

            uint32_t bf[NI][2];
#pragma unroll
            for (int ni = 0; ni < NI; ni += 2) {
                if (ni + 1 < NI) {
                    const int m = lane / 8;
                    const int n = wc*JW + (ni + m/2)*8 + lane % 8;
                    const unsigned int addr = sy_u + n*MOE_PACKED_SY + YOFF + ks*32 + (m % 2)*16;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                        : "=r"(bf[ni][0]), "=r"(bf[ni][1]), "=r"(bf[ni + 1][0]), "=r"(bf[ni + 1][1]) : "r"(addr));
                } else {
                    const int m = (lane % 16) / 8;
                    const int n = wc*JW + ni*8 + lane % 8;
                    const unsigned int addr = sy_u + n*MOE_PACKED_SY + YOFF + ks*32 + m*16;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                        : "=r"(bf[ni][0]), "=r"(bf[ni][1]) : "r"(addr));
                }
            }

            if constexpr (!y_q8) {
#pragma unroll
                for (int mi = 0; mi < MW; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NI; ++ni) {
                        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                            : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                            : "r"(af[mi].x), "r"(af[mi].y), "r"(af[mi].z), "r"(af[mi].w), "r"(bf[ni][0]), "r"(bf[ni][1]));
                    }
                }
            } else if constexpr (pq::family == MOE_PACKED_FAMILY_D4) {
                float da[MW][2];
#pragma unroll
                for (int mi = 0; mi < MW; ++mi) {
                    if constexpr (is_q) {
                        da[mi][0] = pq::scale(st[mi], 0, ks);
                        da[mi][1] = pq::scale(st[mi], 1, ks);
                    } else {
                        da[mi][0] = sd[((wr*MW + mi)*4 + ks)*16 + g];
                        da[mi][1] = sd[((wr*MW + mi)*4 + ks)*16 + g + 8];
                    }
                }
                float db[NI][2];
#pragma unroll
                for (int ni = 0; ni < NI; ++ni) {
                    const int n = wc*JW + ni*8 + 2*t;
                    db[ni][0] = ((const block_q8_1_mmq *) (sy + (n + 0)*MOE_PACKED_SY))->d4[ks];
                    db[ni][1] = ((const block_q8_1_mmq *) (sy + (n + 1)*MOE_PACKED_SY))->d4[ks];
                }
#pragma unroll
                for (int mi = 0; mi < MW; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NI; ++ni) {
                        int c[4] = { 0, 0, 0, 0 };
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                            : "r"(af[mi].x), "r"(af[mi].y), "r"(af[mi].z), "r"(af[mi].w), "r"(bf[ni][0]), "r"(bf[ni][1]));
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            acc[mi][ni][l] += da[mi][l/2]*db[ni][l%2]*moe_packed_cvt<magic>(c[l]);
                        }
                    }
                }
            } else if constexpr (pq::family == MOE_PACKED_FAMILY_DM) {
                // as vec_dot_q8_1_q8_1_mma: (d*sc, -dmin*m) of the rows, (d, s) of the DS4 activations
                float2 da[MW][2];
#pragma unroll
                for (int mi = 0; mi < MW; ++mi) {
                    da[mi][0] = pq::scale(st[mi], 0, ks);
                    da[mi][1] = pq::scale(st[mi], 1, ks);
                }
                float2 db[NI][2];
#pragma unroll
                for (int ni = 0; ni < NI; ++ni) {
                    const int n = wc*JW + ni*8 + 2*t;
                    db[ni][0] = __half22float2(((const block_q8_1_mmq *) (sy + (n + 0)*MOE_PACKED_SY))->ds4[ks]);
                    db[ni][1] = __half22float2(((const block_q8_1_mmq *) (sy + (n + 1)*MOE_PACKED_SY))->ds4[ks]);
                }
#pragma unroll
                for (int mi = 0; mi < MW; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NI; ++ni) {
                        int c[4] = { 0, 0, 0, 0 };
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                            : "r"(af[mi].x), "r"(af[mi].y), "r"(af[mi].z), "r"(af[mi].w), "r"(bf[ni][0]), "r"(bf[ni][1]));
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            acc[mi][ni][l] += da[mi][l/2].x*db[ni][l%2].x*moe_packed_cvt<magic>(c[l]);
                            acc[mi][ni][l] += da[mi][l/2].y*db[ni][l%2].y;
                        }
                    }
                }
            } else {
                static_assert(pq::family == MOE_PACKED_FAMILY_D16, "bad family");
                // as vec_dot_q8_0_16_q8_1_mma: one m16n8k16 per 16 values (registers x, y and z, w of the m16n8k32 fragments)
                float2 da[MW][2];
#pragma unroll
                for (int mi = 0; mi < MW; ++mi) {
                    da[mi][0] = pq::scale(st[mi], 0, ks);
                    da[mi][1] = pq::scale(st[mi], 1, ks);
                }
                float db[NI][2];
#pragma unroll
                for (int ni = 0; ni < NI; ++ni) {
                    const int n = wc*JW + ni*8 + 2*t;
                    db[ni][0] = ((const block_q8_1_mmq *) (sy + (n + 0)*MOE_PACKED_SY))->d4[ks];
                    db[ni][1] = ((const block_q8_1_mmq *) (sy + (n + 1)*MOE_PACKED_SY))->d4[ks];
                }
#pragma unroll
                for (int mi = 0; mi < MW; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NI; ++ni) {
                        // magic: the mma adds the 0x4B400000 of moe_packed_cvt, and fma(c0 + 1.5*2^23, s0, -1.5*2^23*s0) is the rounded
                        // c0*s0 in one FFMA (1.5*2^23*s0 is exact: s0 = d*sc has at most 19 significant bits)
                        constexpr int c_init = magic ? 0x4B400000 : 0;
                        int c0[4] = { c_init, c_init, c_init, c_init };
                        int c1[4] = { c_init, c_init, c_init, c_init };
                        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                            : "+r"(c0[0]), "+r"(c0[1]), "+r"(c0[2]), "+r"(c0[3]) : "r"(af[mi].x), "r"(af[mi].y), "r"(bf[ni][0]));
                        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                            : "+r"(c1[0]), "+r"(c1[1]), "+r"(c1[2]), "+r"(c1[3]) : "r"(af[mi].z), "r"(af[mi].w), "r"(bf[ni][1]));
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const float2 s = da[mi][l/2];
                            float t0;
                            float f1;
                            if constexpr (magic) {
                                t0 = __fmaf_rn(__int_as_float(c0[l]), s.x, -12582912.0f*s.x);
                                f1 = __int_as_float(c1[l]) - 12582912.0f;
                            } else {
                                t0 = (float) c0[l]*s.x;
                                f1 = (float) c1[l];
                            }
                            acc[mi][ni][l] += db[ni][l%2]*__fmaf_rn(f1, s.y, t0);
                        }
                    }
                }
            }
        }

        if ((iter + 1) % kc_n != 0) {
            continue;
        }

        // Write back the finished unit in assignment order, like mul_mat_q_process_tile.
        int tile;
        const int  it  = unit(blockIdx.x + (iter / kc_n)*gridDim.x, tile);
        const int2 tl  = a.tiles[tile];
        const int  col_low  = a.expert_bounds[tl.x] + tl.y;
        const int  col_diff = a.expert_bounds[tl.x + 1] - col_low;
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int j = wc*JW + ni*8 + 2*t + h;
                if (j < col_diff) {
                    float * d = a.dst + (int64_t) a.ids_dst[col_low + j]*a.stride_col_dst + it*MOE_PACKED_I + wr*MW*16 + g;
#pragma unroll
                    for (int mi = 0; mi < MW; ++mi) {
                        d[mi*16 + 0] = acc[mi][ni][h + 0];
                        d[mi*16 + 8] = acc[mi][ni][h + 2];
                    }
                }
#pragma unroll
                for (int mi = 0; mi < MW; ++mi) {
                    acc[mi][ni][h + 0] = 0.0f;
                    acc[mi][ni][h + 2] = 0.0f;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(a, alias, rowtile_outer);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// Also for the lossless formats, which have the same epilogue.
static bool moe_packed_i8_magic() {
    static const bool magic = [] {
        const char * s = getenv("GGML_CUDA_MOE_PACKED_I8_MAGIC");
        return s == nullptr || atoi(s) != 0;
    }();
    return magic;
}

static int moe_packed_variant() {
    static const int variant = [] {
        const char * s = getenv("GGML_CUDA_MOE_PACKED_VARIANT");
        return s ? atoi(s) : -1;
    }();
    return variant;
}

template <int fmt, int J, int NW, int MW, int ST, bool magic>
static void moe_packed_launch_kernel(const moe_packed_launch_args & args, cudaStream_t stream) {
    constexpr int nbytes_shared = ST*(moe_packed_sa<fmt>() + (fmt == GGML_CUDA_MOE_PACKED_I8 ? MOE_PACKED_SD : 0) + J*MOE_PACKED_SY);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    static bool configured[GGML_CUDA_MAX_DEVICES] = {};
    static int  occupancy[GGML_CUDA_MAX_DEVICES]  = {};
    if (!configured[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(moe_packed_mm<fmt, J, NW, MW, ST, magic>, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy[id], moe_packed_mm<fmt, J, NW, MW, ST, magic>,
            NW*WARP_SIZE, nbytes_shared));
        GGML_ASSERT(occupancy[id] > 0);
        GGML_LOG_INFO("%s: CUDA%d packed %s%s J=%d: %d warps, %d rows per warp, %d stages: %d blocks per SM, %d B shared memory per block\n",
            __func__, id, moe_packed_fmt_name(fmt), magic ? " magic" : "", J, NW, 16*MW, ST, occupancy[id], nbytes_shared);
        configured[id] = true;
    }

    moe_packed_mm<fmt, J, NW, MW, ST, magic><<<nsm*occupancy[id], NW*WARP_SIZE, nbytes_shared, stream>>>(args, moe_packed_alias(), moe_packed_rowtile_outer());
    CUDA_CHECK(cudaGetLastError());
}

template <int fmt, int J, int NW, int MW, int ST>
static void moe_packed_launch_magic(const moe_packed_launch_args & args, cudaStream_t stream) {
    if (fmt != GGML_CUDA_MOE_PACKED_F16 && moe_packed_i8_magic()) {
        moe_packed_launch_kernel<fmt, J, NW, MW, ST, fmt != GGML_CUDA_MOE_PACKED_F16>(args, stream);
    } else {
        moe_packed_launch_kernel<fmt, J, NW, MW, ST, false>(args, stream);
    }
}

// GGML_CUDA_MOE_PACKED_VARIANT: 0 = 8 warps, 4 stages; 1 = 8 warps, 3 stages; 2 = 4 warps, 3 stages; 3 = 4 warps, 4 stages.
// Default: 4 warps and 3 stages, but 8 warps for the int8 formats above J=64, where the 4-warp i8 instances spill registers.
template <int fmt, int J>
static void moe_packed_launch_J(const moe_packed_launch_args & args, cudaStream_t stream) {
    // 8 warps: two warps per column range when a warp would get at least 8 columns, so each B fragment feeds two A fragments.
    constexpr int MW8 = J % 16 == 0 && J >= 32 ? 2 : 1;
    // 4 warps: 64 x J/2 or 32 x J warp tiles, fewer shared memory reads per mma.
    constexpr int MW4 = J % 16 == 0 && J >= 32 ? 4 : 2;
    switch (moe_packed_variant()) {
        case 1:  moe_packed_launch_magic<fmt, J, 8, MW8, 3>(args, stream); break;
        case 2:  moe_packed_launch_magic<fmt, J, 4, MW4, 3>(args, stream); break;
        case 3:  moe_packed_launch_magic<fmt, J, 4, MW4, 4>(args, stream); break;
        case 0:  moe_packed_launch_magic<fmt, J, 8, MW8, 4>(args, stream); break;
        default:
            if constexpr (fmt != GGML_CUDA_MOE_PACKED_F16 && J > 64) {
                moe_packed_launch_magic<fmt, J, 8, MW8, 3>(args, stream);
            } else {
                moe_packed_launch_magic<fmt, J, 4, MW4, 3>(args, stream);
            }
            break;
    }
}

template <int fmt>
static void moe_packed_switch_J(int J, const moe_packed_launch_args & args, cudaStream_t stream) {
    switch (J) {
        case   8: moe_packed_launch_J<fmt,   8>(args, stream); break;
        case  16: moe_packed_launch_J<fmt,  16>(args, stream); break;
        case  24: moe_packed_launch_J<fmt,  24>(args, stream); break;
        case  32: moe_packed_launch_J<fmt,  32>(args, stream); break;
        case  48: moe_packed_launch_J<fmt,  48>(args, stream); break;
        case  64: moe_packed_launch_J<fmt,  64>(args, stream); break;
        case  96: moe_packed_launch_J<fmt,  96>(args, stream); break;
        case 128: moe_packed_launch_J<fmt, 128>(args, stream); break;
        default:
            GGML_ABORT("packed MoE: no instance for J=%d", J);
    }
}

void ggml_cuda_moe_packed_launch(int fmt, int J, const moe_packed_launch_args & args, cudaStream_t stream) {
    switch (fmt) {
        case GGML_CUDA_MOE_PACKED_F16:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_F16> (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_I8:   moe_packed_switch_J<GGML_CUDA_MOE_PACKED_I8>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_Q80:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_Q80> (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_Q50:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_Q50> (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ4NL:moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ4NL>(J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_MXFP4:moe_packed_switch_J<GGML_CUDA_MOE_PACKED_MXFP4>(J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_Q5K:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_Q5K>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ3S:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ3S> (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ4XS: moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ4XS>(J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_Q4K:   moe_packed_switch_J<GGML_CUDA_MOE_PACKED_Q4K>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_Q6K:   moe_packed_switch_J<GGML_CUDA_MOE_PACKED_Q6K>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_Q41:   moe_packed_switch_J<GGML_CUDA_MOE_PACKED_Q41>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_Q51:   moe_packed_switch_J<GGML_CUDA_MOE_PACKED_Q51>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_Q3K:   moe_packed_switch_J<GGML_CUDA_MOE_PACKED_Q3K>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ3XXS:moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ3XXS>(J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ2XXS:moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ2XXS>(J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ2XS: moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ2XS> (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ2S:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ2S>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ2K:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ2K>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ3K:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ3K>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ4K:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ4K>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ5K:  moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ5K>  (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ2KS: moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ2KS> (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ3KS: moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ3KS> (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ4KS: moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ4KS> (J, args, stream); break;
        case GGML_CUDA_MOE_PACKED_IQ5KS: moe_packed_switch_J<GGML_CUDA_MOE_PACKED_IQ5KS> (J, args, stream); break;
        default:
            GGML_ABORT("packed MoE: bad format %d", fmt);
    }
}

#else

int ggml_cuda_moe_packed_select(const ggml_tensor * src0, bool warn) {
    GGML_UNUSED(src0);
    GGML_UNUSED(warn);
    return GGML_CUDA_MOE_PACKED_NONE;
}

void ggml_cuda_moe_packed_forget(int device, const void * base, size_t size) {
    GGML_UNUSED(device);
    GGML_UNUSED(base);
    GGML_UNUSED(size);
}

int ggml_cuda_moe_packed_inplace_fmt(const ggml_tensor * src0) {
    GGML_UNUSED(src0);
    return GGML_CUDA_MOE_PACKED_NONE;
}

void ggml_cuda_moe_packed_check_not_inplace(const ggml_tensor * src0) {
    GGML_UNUSED(src0);
}

void ggml_cuda_moe_packed_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
}

moe_packed_weights ggml_cuda_moe_packed_get(ggml_cuda_pool_alloc<char> & storage, int fmt, const ggml_tensor * src0, cudaStream_t stream) {
    GGML_UNUSED(storage);
    GGML_UNUSED(fmt);
    GGML_UNUSED(src0);
    GGML_UNUSED(stream);
    GGML_ABORT("packed MoE weights are not available on this backend");
}

void ggml_cuda_moe_packed_y_f16(const float * x, const int32_t * ids, half * y, int64_t ne10, int64_t stride_row,
        int64_t n_src_rows, int n_expert_used, bool scatter, cudaStream_t stream) {
    GGML_UNUSED_VARS(x, ids, y, ne10, stride_row, n_src_rows, n_expert_used, scatter, stream);
    GGML_ABORT("packed MoE weights are not available on this backend");
}

void ggml_cuda_moe_packed_launch(int fmt, int J, const moe_packed_launch_args & args, cudaStream_t stream) {
    GGML_UNUSED_VARS(fmt, J, args, stream);
    GGML_ABORT("packed MoE weights are not available on this backend");
}

void ggml_backend_cuda_moe_packed_clear(void) {
}

void ggml_backend_cuda_moe_packed_set_inplace(int inplace) {
    GGML_UNUSED(inplace);
}

#endif // GGML_CUDA_MOE_PACKED_AVAILABLE
