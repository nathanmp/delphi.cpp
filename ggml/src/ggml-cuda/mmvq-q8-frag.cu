#include "mmvq-q8-frag.cuh"
#include "convert.cuh"

#include <algorithm>
#include <cinttypes>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Layout of one row tile (16 rows, K = 32*nkt values, nkg = nkt/4), in the 16*34*nkt bytes of its rows:
//   [T*512, T*512 + 512)      tile T, lane l = 4g + t: int4 {row g k 4t.., row g+8 k 4t.., row g k 16+4t.., row g+8 k 16+4t..}
//   [nkt*512 + kg*128, +128)  scales of tiles 4kg..4kg+3: [g][4] half2 {d row g, d row g+8}
// A row tile only depends on its own rows, so a weight made of adjacent tensors (glm5next_adjacent_rows) and its parts
// read the same bytes.
// Activations of a pass of 8*NT columns: B [nt][kg][lane][4][2] int (column 8nt + g: k 4t.. and k 16+4t.. of tile 4kg + i),
// YS [nt][kg][t][4][2] float (columns 8nt + 2t and 8nt + 2t + 1).

static constexpr int Q8F_MIN_K     = 1024;
static constexpr int Q8F_MIN_ROWS  = 256;
static constexpr int Q8F_PASS_COLS = 32;

// widest product of the fragment kernel; wider products dequantize to F16 for cuBLAS
static int q8f_max_cols() {
    static const int v = [] {
        const char * s = getenv("GGML_CUDA_Q8_FRAG_MAX_COLS");
        return s != nullptr ? atoi(s) : 64;
    }();
    return v;
}

#if !defined(GGML_USE_HIP)

static __device__ __forceinline__ float q8f_i2f(const int x) { // exact for |x| < 2^22, without the quarter-rate I2F
    return __int_as_float(x + 0x4B400000) - 12582912.0f;
}

// one thread per (row tile, k tile, lane); lanes with t == 0 also write the scales
static __global__ void q8f_pack(const char * __restrict__ src, char * __restrict__ dst, const int64_t row_bytes, const int nkt,
        const int64_t n_tiles) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_tiles*nkt*32) {
        return;
    }
    const int     lane = i % 32;
    const int     T    = (i / 32) % nkt;
    const int64_t R    = i / (32*nkt);
    const int g = lane / 4, t = lane % 4;
    const char * tile_src = src + R*16*row_bytes;
    char       * tile_dst = dst + R*16*row_bytes;

    const int rows[4] = {g, g + 8, g, g + 8};
    const int ks[4]   = {4*t, 4*t, 16 + 4*t, 16 + 4*t};
    int v[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const uint8_t * q = (const uint8_t *) tile_src + rows[r]*row_bytes + T*34 + 2 + ks[r];
        v[r] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
    }
    *(int4 *) (tile_dst + (int64_t) T*512 + lane*16) = make_int4(v[0], v[1], v[2], v[3]);

    if (t == 0) {
        const half d0 = *(const half *) (tile_src + g*row_bytes + T*34);
        const half d1 = *(const half *) (tile_src + (g + 8)*row_bytes + T*34);
        *(half2 *) (tile_dst + (int64_t) nkt*512 + (T/4)*128 + g*16 + (T%4)*4) = __halves2half2(d0, d1);
    }
}

// packed rows -> F16 rows; a block of 128 threads per (row tile, group of 4 k tiles) goes through shared memory, so that
// 8 threads write the 256 bytes of a row
static __global__ void q8f_unpack_f16(const char * __restrict__ src, half * __restrict__ dst, const int64_t row_bytes, const int nkt) {
    const int     kg   = blockIdx.x;
    const int64_t R    = blockIdx.y;
    const int     tid  = threadIdx.x;
    const int     lane = tid % 32, T = kg*4 + tid/32;
    const int     g = lane / 4, t = lane % 4;
    const int64_t K = (int64_t) nkt*32;
    const char * tile = src + R*16*row_bytes;

    __shared__ __align__(16) int8_t q[16][128 + 16];
    __shared__ float  d[16][4];
    const int4 v = *(const int4 *) (tile + (int64_t) T*512 + lane*16);
    const int  qv[4]   = {v.x, v.y, v.z, v.w};
    const int  rows[4] = {g, g + 8, g, g + 8};
    const int  ks[4]   = {4*t, 4*t, 16 + 4*t, 16 + 4*t};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        *(int *) &q[rows[r]][(tid/32)*32 + ks[r]] = qv[r];
    }
    if (t == 0) {
        const half2 h = *(const half2 *) (tile + (int64_t) nkt*512 + kg*128 + g*16 + (tid/32)*4);
        d[g][tid/32]     = __low2float(h);
        d[g + 8][tid/32] = __high2float(h);
    }
    __syncthreads();
    const int row = tid / 8, seg = tid % 8; // 16 values of the row
    const float dr = d[row][seg/2];
    const int4  qq = *(const int4 *) &q[row][seg*16];
    const int   qw[4] = {qq.x, qq.y, qq.z, qq.w};
    half2 out[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int w = qw[j/2] >> (16*(j%2));
        out[j] = __floats2half2_rn(dr*(float) (int8_t) w, dr*(float) (int8_t) (w >> 8));
    }
    int4 * y = (int4 *) (dst + (R*16 + row)*K + kg*128 + seg*16);
    y[0] = *(const int4 *) &out[0];
    y[1] = *(const int4 *) &out[4];
}

// one warp per (column, 32-value block) with the rounding of quantize_q8_1, into the B / YS layout of a pass of 8*NT
// columns (gridDim.y). Columns >= ncols are zero.
static __global__ void q8f_quantize(const float * __restrict__ x, int8_t * __restrict__ B, float * __restrict__ YS, const int64_t s_col,
        const int ncols, const int nkt) {
    const int lane = threadIdx.x;
    const int T    = blockIdx.x*blockDim.y + threadIdx.y;
    const int c    = blockIdx.y;
    if (T >= nkt) {
        return;
    }
    const float  xi   = c < ncols ? x[c*s_col + T*32 + lane] : 0.0f;
    const float  amax = warp_reduce_max<32>(fabsf(xi));
    const float  d    = amax/127.0f;
    const int8_t q    = amax == 0.0f ? 0 : (int8_t) roundf(xi/d);

    const int nkg = nkt/4;
    const int nt = c/8, cg = c%8, kg = T/4, i = T%4;
    const int tt = (lane % 16)/4;
    B[((((int64_t) (nt*nkg + kg)*32 + cg*4 + tt)*4 + i)*2 + (lane >= 16))*4 + lane%4] = q;
    if (lane == 0) {
        YS[(((int64_t) (nt*nkg + kg)*4 + cg/2)*4 + i)*2 + cg%2] = d;
    }
}

struct q8f_group {
    int4  a[4];
    uint4 s;
};

// block: nw warps on one row tile, warp w takes k groups w, w + nw, ...; NT n-tiles of 8 columns
template <int NT, int nw>
static __global__ void __launch_bounds__(nw*32) q8f_mul_mat(const char * __restrict__ x, const int4 * __restrict__ B,
        const float4 * __restrict__ YS, float * __restrict__ dst, const int64_t row_bytes, const int nkt, const int nrows,
        const int ncols, const int64_t stride_col_dst) {
#if defined(AMPERE_MMA_AVAILABLE)
    const int lane = threadIdx.x, w = threadIdx.y;
    const int g = lane >> 2, t = lane & 3;
    const int nkg = nkt/4;
    const int64_t R = blockIdx.x;
    const char  * tile = x + R*16*row_bytes;
    const int4  * a = (const int4 *) tile + lane;
    const uint4 * s = (const uint4 *) (tile + (int64_t) nkt*512) + g;

    float acc[NT][4] = {};
    q8f_group cur;
    if (w < nkg) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            cur.a[i] = __ldcs(a + (w*4 + i)*32);
        }
        cur.s = __ldcs(s + w*8);
    }
    for (int kg = w; kg < nkg; kg += nw) {
        const q8f_group xg = cur;
        const int kn = kg + nw;
        if (kn < nkg) { // the next group of this warp is in flight while this one is multiplied
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                cur.a[i] = __ldcs(a + (kn*4 + i)*32);
            }
            cur.s = __ldcs(s + kn*8);
        }
        const uint32_t sv[4] = {xg.s.x, xg.s.y, xg.s.z, xg.s.w};
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int64_t ib = ((int64_t) (nt*nkg + kg)*32 + lane)*2;
            const int64_t iy = ((int64_t) (nt*nkg + kg)*4 + t)*2;
            const int4   b0 = __ldg(B + ib);
            const int4   b1 = __ldg(B + ib + 1);
            const float4 y0 = __ldg(YS + iy);
            const float4 y1 = __ldg(YS + iy + 1);
            const int   bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            const float yv[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int c[4] = {0, 0, 0, 0};
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                    : "r"(xg.a[i].x), "r"(xg.a[i].y), "r"(xg.a[i].z), "r"(xg.a[i].w), "r"(bv[2*i]), "r"(bv[2*i + 1]));
                const float2 d = __half22float2(*(const half2 *) &sv[i]);
                acc[nt][0] += d.x*yv[2*i]    *q8f_i2f(c[0]);
                acc[nt][1] += d.x*yv[2*i + 1]*q8f_i2f(c[1]);
                acc[nt][2] += d.y*yv[2*i]    *q8f_i2f(c[2]);
                acc[nt][3] += d.y*yv[2*i + 1]*q8f_i2f(c[3]);
            }
        }
    }

    __shared__ float part[nw][NT][4][32];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            part[w][nt][i][lane] = acc[nt][i];
        }
    }
    __syncthreads();
    // fixed summation order: the result of a column does not depend on ncols
    for (int j = w; j < NT*4; j += nw) {
        const int nt = j / 4, i = j % 4;
        float r = 0.0f;
#pragma unroll
        for (int v = 0; v < nw; ++v) {
            r += part[v][nt][i][lane];
        }
        const int64_t row = R*16 + g + (i >= 2 ? 8 : 0);
        const int     col = nt*8 + 2*t + (i & 1);
        if (col < ncols && row < nrows) {
            dst[col*stride_col_dst + row] = r;
        }
    }
#else
    GGML_UNUSED_VARS(x, B, YS, dst, row_bytes, nkt, nrows, ncols, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE)
}

// ---------------------------------------------------------------------------------------------------------------------
// packed address ranges per device

namespace {

struct q8f_range {
    int          device;
    const char * lo;
    const char * hi;
    int64_t      row_bytes = 0;   // the packed layout is local to 16-row tiles of rows of this size
    std::string  name;            // the weight that was packed (diagnostics; graph tensor names do not outlive the graph)
};

std::mutex                              q8f_mutex;
std::vector<q8f_range>                  q8f_ranges;  // packed weights, no two overlap
std::vector<q8f_range>                  q8f_blocked; // weights that other ops read: never packed
std::set<std::tuple<int, const void *, size_t>> q8f_seen; // (device, data, size) of MUL_MAT weights already decided
std::atomic<bool>                       q8f_any{false};

static bool overlaps(const q8f_range & r, int device, const char * lo, const char * hi) {
    return r.device == device && r.lo < hi && lo < r.hi;
}

static size_t q8f_covered_locked(int device, const char * lo, const char * hi) {
    size_t n = 0;
    for (const q8f_range & r : q8f_ranges) {
        if (overlaps(r, device, lo, hi)) {
            n += std::min(hi, r.hi) - std::max(lo, r.lo);
        }
    }
    return n;
}

} // namespace

bool ggml_cuda_q8f_enabled(int cc) {
    static const int env = [] {
        const char * s = getenv("GGML_CUDA_Q8_FRAG");
        return s == nullptr ? -1 : atoi(s);
    }();
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE || ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    return env < 0 ? cc == GGML_CUDA_CC_AMPERE : env != 0;
}

bool ggml_cuda_q8f_packed(const ggml_tensor * t) {
    if (!q8f_any.load(std::memory_order_relaxed) || t->type != GGML_TYPE_Q8_0 || t->data == nullptr) {
        return false;
    }
    const int    device = ggml_cuda_get_device();
    const char * lo     = (const char *) t->data;
    const char * hi     = lo + ggml_nbytes(t);
    std::lock_guard<std::mutex> lock(q8f_mutex);
    const size_t n = q8f_covered_locked(device, lo, hi);
    if (n != 0 && n != (size_t) (hi - lo)) {
        GGML_ABORT("%s: %s is only partly in packed Q8_0 weights", __func__, t->name);
    }
    return n != 0;
}

void ggml_cuda_q8f_check_not_packed(int device, const void * data, size_t size, const char * what) {
    if (!q8f_any.load(std::memory_order_relaxed)) {
        return;
    }
    const char * lo = (const char *) data;
    std::lock_guard<std::mutex> lock(q8f_mutex);
    if (q8f_covered_locked(device, lo, lo + size) != 0) {
        GGML_ABORT("%s: %s touches Q8_0 weights in the packed fragment layout (GGML_CUDA_Q8_FRAG=0 disables the packing)", __func__, what);
    }
}

void ggml_cuda_q8f_forget(int device, const void * base, size_t size) {
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    std::lock_guard<std::mutex> lock(q8f_mutex);
    auto gone = [&](const q8f_range & r) { return r.device == device && r.lo >= lo && r.hi <= hi; };
    q8f_ranges.erase(std::remove_if(q8f_ranges.begin(), q8f_ranges.end(), gone), q8f_ranges.end());
    q8f_blocked.erase(std::remove_if(q8f_blocked.begin(), q8f_blocked.end(), gone), q8f_blocked.end());
    for (auto it = q8f_seen.begin(); it != q8f_seen.end();) {
        const char * p = (const char *) std::get<1>(*it);
        it = std::get<0>(*it) == device && p >= lo && p < hi ? q8f_seen.erase(it) : std::next(it);
    }
}

static bool q8f_is_weight(const ggml_tensor * t, ggml_backend_buffer_type_t buft) {
    return t != nullptr && t->op == GGML_OP_NONE && t->view_src == nullptr && t->buffer != nullptr && t->extra == nullptr &&
        ggml_backend_buffer_get_type(t->buffer) == buft &&
        ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
}

static bool q8f_eligible(const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    return src0->type == GGML_TYPE_Q8_0 && node->type == GGML_TYPE_F32 && ggml_get_op_params_i32(node, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && ggml_is_contiguous(src0) && src0->ne[0] % 128 == 0 && src0->ne[0] >= Q8F_MIN_K &&
        src0->ne[1] % 16 == 0 && src0->ne[1] >= Q8F_MIN_ROWS && (uintptr_t) src0->data % 16 == 0 &&
        src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && node->nb[0] == sizeof(float);
}

void ggml_cuda_q8f_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_q8f_enabled(cc)) {
        return;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(ctx.device);
    std::vector<const ggml_tensor *> todo;
    {
        std::lock_guard<std::mutex> lock(q8f_mutex);
        // Q8_0 weights that another op reads keep the GGUF layout
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                const ggml_tensor * s = node->src[j];
                if (s == nullptr || s->type != GGML_TYPE_Q8_0 || (node->op == GGML_OP_MUL_MAT && j == 0)) {
                    continue;
                }
                const ggml_tensor * base = s->view_src ? s->view_src : s;
                if (!q8f_is_weight(base, buft)) {
                    continue;
                }
                const char * lo = (const char *) base->data;
                const char * hi = lo + ggml_nbytes(base);
                if (q8f_covered_locked(ctx.device, lo, hi) != 0) {
                    GGML_ABORT("%s: %s reads %s, which is packed for MUL_MAT (GGML_CUDA_Q8_FRAG=0 disables the packing)", __func__,
                        ggml_op_desc(node), base->name);
                }
                q8f_blocked.push_back({ ctx.device, lo, hi });
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node->op != GGML_OP_MUL_MAT || node->src[0]->type != GGML_TYPE_Q8_0 || !q8f_is_weight(node->src[0], buft)) {
                continue;
            }
            const ggml_tensor * src0 = node->src[0];
            if (!q8f_seen.insert({ ctx.device, src0->data, ggml_nbytes(src0) }).second) {
                continue;
            }
            const char * lo = (const char *) src0->data;
            const char * hi = lo + ggml_nbytes(src0);
            bool blocked = false;
            for (const q8f_range & r : q8f_blocked) {
                blocked = blocked || overlaps(r, ctx.device, lo, hi);
            }
            if (blocked || !q8f_eligible(node)) {
                // the old paths cannot read a weight that is partly packed (a part of adjacent tensors)
                if (q8f_covered_locked(ctx.device, lo, hi) != 0) {
                    GGML_ABORT("%s: %s is partly packed but not eligible", __func__, src0->name);
                }
                continue;
            }
            todo.push_back(src0);
        }
    }
    if (todo.empty()) {
        return;
    }

    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    GGML_ASSERT(status == cudaStreamCaptureStatusNone && "packed Q8_0 weights must be built before graph capture");

    const int64_t t0      = ggml_time_us();
    const size_t  tmp_max = 64ull << 20;
    size_t nbytes = 0;
    char * tmp    = nullptr;
    CUDA_CHECK(cudaMallocAsync(&tmp, tmp_max, stream));
    std::lock_guard<std::mutex> lock(q8f_mutex);
    for (const ggml_tensor * src0 : todo) {
        const int64_t row_bytes  = src0->nb[1];
        const int64_t tile_bytes = 16*row_bytes;
        const int     nkt        = src0->ne[0]/32;
        const char *  lo         = (const char *) src0->data;
        const char *  hi         = lo + ggml_nbytes(src0);
        GGML_ASSERT((size_t) tile_bytes <= tmp_max);
        // the row tiles that are not packed yet: a part of an adjacent weight can be packed already
        std::vector<q8f_range> in;
        for (const q8f_range & r : q8f_ranges) {
            if (overlaps(r, ctx.device, lo, hi)) {
                in.push_back(r);
            }
        }
        std::sort(in.begin(), in.end(), [](const q8f_range & a, const q8f_range & b) { return a.lo < b.lo; });
        // A packed range can also be larger than src0: a graph can multiply by a tensor that spans adjacent weights (a fused
        // gate/up made of ffn_gate and ffn_up, for example) before it multiplies by one of them. The layout of a row tile does
        // not depend on the tensor around it, so the part inside src0 is valid if it is made of whole row tiles of src0.
        std::vector<std::pair<const char *, const char *>> gaps;
        const char * p = lo;
        for (const q8f_range & r : in) {
            const char * a = std::max(r.lo, lo);
            const char * b = std::min(r.hi, hi);
            if (r.row_bytes != row_bytes || (a - lo) % tile_bytes != 0 || (b - lo) % tile_bytes != 0) {
                GGML_ABORT("%s: %s (CUDA%d %p..%p, rows of %" PRId64 " B) overlaps packed %s (%p..%p, rows of %" PRId64 " B) not in whole "
                    "row tiles", __func__, src0->name, ctx.device, (const void *) lo, (const void *) hi, row_bytes,
                    r.name.c_str(), (const void *) r.lo, (const void *) r.hi, r.row_bytes);
            }
            if (a > p) {
                gaps.push_back({ p, a });
            }
            p = b;
        }
        if (p < hi) {
            gaps.push_back({ p, hi });
        }
        const int64_t chunk_tiles = (int64_t) (tmp_max / tile_bytes);
        for (const auto & gap : gaps) {
            for (const char * c = gap.first; c < gap.second; c += chunk_tiles*tile_bytes) {
                const int64_t nt   = std::min<int64_t>(chunk_tiles, (gap.second - c)/tile_bytes);
                const int64_t nthr = nt*nkt*32;
                CUDA_CHECK(cudaMemcpyAsync(tmp, c, nt*tile_bytes, cudaMemcpyDeviceToDevice, stream));
                q8f_pack<<<(nthr + 255)/256, 256, 0, stream>>>(tmp, const_cast<char *>(c), row_bytes, nkt, nt);
                CUDA_CHECK(cudaGetLastError());
            }
            q8f_ranges.push_back({ ctx.device, gap.first, gap.second, row_bytes, src0->name });
            nbytes += gap.second - gap.first;
        }
    }
    CUDA_CHECK(cudaFreeAsync(tmp, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    q8f_any.store(true, std::memory_order_relaxed);
    GGML_LOG_INFO("%s: CUDA%d packed %zu Q8_0 weights in place (%.1f MiB) in %.1f ms\n", __func__, ctx.device, todo.size(),
        nbytes/1048576.0, (ggml_time_us() - t0)/1000.0);
}

// ---------------------------------------------------------------------------------------------------------------------

template <int NT>
static void q8f_launch(const char * x, const int4 * B, const float4 * YS, float * dst, int64_t row_bytes, int nkt, int nrows, int ncols,
        int64_t stride_col_dst, cudaStream_t stream) {
    // about 1024 warps on the GPU keep enough loads in flight; more warps per row tile only add reduction work
    const int  ntiles = nrows/16;
    const int  nkg    = nkt/4;
    const dim3 grid(ntiles);
    if (ntiles >= 512 || nkg < 4) {
        q8f_mul_mat<NT, 2><<<grid, dim3(32, 2), 0, stream>>>(x, B, YS, dst, row_bytes, nkt, nrows, ncols, stride_col_dst);
    } else if (ntiles >= 128 || nkg < 8) {
        q8f_mul_mat<NT, 4><<<grid, dim3(32, 4), 0, stream>>>(x, B, YS, dst, row_bytes, nkt, nrows, ncols, stride_col_dst);
    } else {
        q8f_mul_mat<NT, 8><<<grid, dim3(32, 8), 0, stream>>>(x, B, YS, dst, row_bytes, nkt, nrows, ncols, stride_col_dst);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_q8f_uses_cublas(int64_t ncols) {
    return ncols > q8f_max_cols();
}

// nc columns of x (stride s_col floats) into nc columns of dst (stride stride_col_dst floats)
static void q8f_mul_mat_cols(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const float * x, int64_t s_col, int64_t nc,
        float * dst, int64_t stride_col_dst) {
    cudaStream_t stream = ctx.stream();
    const int nkt   = src0->ne[0]/32;
    const int nkg   = nkt/4;
    const int nrows = src0->ne[1];
    ggml_cuda_pool_alloc<char> tmp(ctx.pool());
    for (int64_t c0 = 0; c0 < nc; c0 += Q8F_PASS_COLS) {
        const int n  = (int) std::min<int64_t>(Q8F_PASS_COLS, nc - c0);
        const int NT = n <= 8 ? 1 : n <= 16 ? 2 : 4;
        const size_t nb_B  = (size_t) NT*nkg*32*4*2*4;
        const size_t nb_YS = (size_t) NT*nkg*4*4*2*sizeof(float);
        // the activations of a pass are shared by the products with the same input (the q8_1 cache of MMVQ, under a key
        // that no q8_1 copy has: nb[3] = 0x51F8)
        ggml_tensor key = {};
        key.type  = GGML_TYPE_F32;
        key.data  = const_cast<float *>(x + c0*s_col);
        key.ne[0] = (int64_t) nkt*32; key.ne[1] = n; key.ne[2] = 1; key.ne[3] = 1;
        key.nb[0] = sizeof(float); key.nb[1] = s_col*sizeof(float); key.nb[2] = n*s_col*sizeof(float); key.nb[3] = 0x51F8;
        char * buf = nullptr;
        const bool hit = ctx.q8_1_cache_get(&key, nb_B + nb_YS, &buf);
        if (buf == nullptr) {
            buf = tmp.ptr != nullptr ? tmp.get() : tmp.alloc((size_t) 4*nkg*(32*4*2*4 + 4*4*2*sizeof(float)));
        }
        int8_t * B  = (int8_t *) buf;
        float  * YS = (float *) (buf + nb_B);
        if (!hit) {
            const dim3 qgrid((nkt + 7)/8, 8*NT);
            q8f_quantize<<<qgrid, dim3(32, 8), 0, stream>>>(x + c0*s_col, B, YS, s_col, n, nkt);
            CUDA_CHECK(cudaGetLastError());
        }
        const char * xw = (const char *) src0->data;
        switch (NT) {
            case 1:  q8f_launch<1>(xw, (const int4 *) B, (const float4 *) YS, dst + c0*stride_col_dst, src0->nb[1], nkt, nrows, n, stride_col_dst, stream); break;
            case 2:  q8f_launch<2>(xw, (const int4 *) B, (const float4 *) YS, dst + c0*stride_col_dst, src0->nb[1], nkt, nrows, n, stride_col_dst, stream); break;
            default: q8f_launch<4>(xw, (const int4 *) B, (const float4 *) YS, dst + c0*stride_col_dst, src0->nb[1], nkt, nrows, n, stride_col_dst, stream); break;
        }
    }
}

// wide products: F16 row chunks of the weight and cuBLAS, as in mm-dense-chunked.cu
static void q8f_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const float * x, int64_t s_col, int64_t nc,
        float * dst, int64_t stride_col_dst) {
    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int     nkt  = ne00/32;

    ggml_cuda_pool_alloc<half> xh(ctx.pool(), ne00*nc);
    const to_fp16_cuda_t to_f16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    if (s_col == ne00) {
        to_f16(x, xh.get(), 1, ne00*nc, stream);
    } else {
        for (int64_t c = 0; c < nc; ++c) {
            to_f16(x + c*s_col, xh.get() + c*ne00, 1, ne00, stream);
        }
    }
    int64_t rows = (int64_t) ((64ull << 20) / (ne00*sizeof(half)));
    rows = std::min(ne01, std::max<int64_t>(128, rows & ~(int64_t) 127));
    ggml_cuda_pool_alloc<half> w(ctx.pool(), rows*ne00);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    for (int64_t r0 = 0; r0 < ne01; r0 += rows) {
        const int64_t nr   = std::min(rows, ne01 - r0);
        q8f_unpack_f16<<<dim3(nkt/4, nr/16), 128, 0, stream>>>((const char *) src0->data + r0*src0->nb[1], w.get(), src0->nb[1], nkt);
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                nr, nc, ne00,
                &alpha, w.get(),  CUDA_R_16F, ne00,
                        xh.get(), CUDA_R_16F, ne00,
                &beta,  dst + r0, CUDA_R_32F, stride_col_dst,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

void ggml_cuda_q8f_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ne02 == 1 && ne03 == 1);
    GGML_ASSERT(nb10 == sizeof(float) && nb0 == sizeof(float));

    const float * x  = (const float *) src1->data;
    float       * d  = (float *) dst->data;
    const int64_t s1 = nb11/sizeof(float), s2 = nb12/sizeof(float), s3 = nb13/sizeof(float);
    const int64_t d1 = nb1/sizeof(float),  d2 = nb2/sizeof(float),  d3 = nb3/sizeof(float);
    // all columns as one 2D product when the strides of both sides allow it
    const bool    flat    = s2 == s1*ne11 && s3 == s2*ne12 && d2 == d1*ne1 && d3 == d2*ne2;
    const int64_t n_outer = flat ? 1 : ne12*ne13;
    const int64_t nc      = flat ? ne11*ne12*ne13 : ne11;
    for (int64_t o = 0; o < n_outer; ++o) {
        const int64_t i12 = o % ne12, i13 = o / ne12;
        const float * xo = x + i12*s2 + i13*s3;
        float       * dsto = d + i12*d2 + i13*d3;
        if (nc <= q8f_max_cols()) {
            q8f_mul_mat_cols(ctx, src0, xo, s1, nc, dsto, d1);
        } else {
            q8f_mul_mat_cublas(ctx, src0, xo, s1, nc, dsto, d1);
        }
    }
}

#else // GGML_USE_HIP

bool ggml_cuda_q8f_enabled(int cc) { GGML_UNUSED(cc); return false; }
bool ggml_cuda_q8f_packed(const ggml_tensor * t) { GGML_UNUSED(t); return false; }
void ggml_cuda_q8f_check_not_packed(int device, const void * data, size_t size, const char * what) { GGML_UNUSED_VARS(device, data, size, what); }
void ggml_cuda_q8f_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) { GGML_UNUSED_VARS(ctx, cgraph); }
void ggml_cuda_q8f_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    GGML_ABORT("packed Q8_0 weights are CUDA only");
}
bool ggml_cuda_q8f_uses_cublas(int64_t ncols) { GGML_UNUSED(ncols); return false; }
void ggml_cuda_q8f_forget(int device, const void * base, size_t size) { GGML_UNUSED_VARS(device, base, size); }

#endif // GGML_USE_HIP
