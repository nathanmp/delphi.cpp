#include "moe-grouped.cuh"
#include "moe-packed.cuh"
#include "mmq-grouped.cuh"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <string>

#define MOE_GROUPED_MAX_CLASSES 8
#define MOE_GROUPED_TILES_BLOCK 256

// Class c takes the experts with rows_max[c - 1] < rows <= rows_max[c]; its tiles start at base[c] in the tile array.
struct moe_grouped_classes {
    int n;
    int J[MOE_GROUPED_MAX_CLASSES];
    int rows_max[MOE_GROUPED_MAX_CLASSES];
    int base[MOE_GROUPED_MAX_CLASSES];
};

// Exclusive scan over the block. total gets the sum of all v.
static __device__ int moe_grouped_block_scan(int v, int * sh, int & total) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int nw   = blockDim.x / warp_size;

    int incl = v;
#pragma unroll
    for (int off = 1; off < warp_size; off <<= 1) {
        const int x = __shfl_up_sync(0xFFFFFFFF, incl, off, warp_size);
        if (lane >= off) {
            incl += x;
        }
    }
    if (lane == warp_size - 1) {
        sh[warp] = incl;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < nw; ++w) {
            sh[w] += sh[w - 1];
        }
    }
    __syncthreads();
    const int excl = incl - v + (warp > 0 ? sh[warp - 1] : 0);
    total = sh[nw - 1];
    __syncthreads();
    return excl;
}

// One block. Writes the column tiles (expert, first row inside the expert) of every class in expert order, so the tile order
// and therefore the stream-k split, and the result, depend only on the routing.
static __global__ void moe_grouped_tiles(
        const int32_t * __restrict__ offsets, const int n_experts, const moe_grouped_classes cls,
        int2 * __restrict__ tiles, int32_t * __restrict__ n_tiles) {
    __shared__ int sh[MOE_GROUPED_TILES_BLOCK/16];

    int count[MOE_GROUPED_MAX_CLASSES] = {0};

    for (int e0 = 0; e0 < n_experts; e0 += blockDim.x) {
        const int e    = e0 + threadIdx.x;
        const int rows = e < n_experts ? offsets[e + 1] - offsets[e] : 0;

        int c = 0;
        while (c < cls.n - 1 && rows > cls.rows_max[c]) {
            c++;
        }

        for (int ci = 0; ci < cls.n; ++ci) {
            const int nt = rows > 0 && c == ci ? (rows + cls.J[ci] - 1) / cls.J[ci] : 0;
            int total;
            const int pos = moe_grouped_block_scan(nt, sh, total);
            for (int q = 0; q < nt; ++q) {
                tiles[cls.base[ci] + count[ci] + pos + q] = make_int2(e, q*cls.J[ci]);
            }
            count[ci] += total;
        }
    }

    for (int ci = 0; ci < cls.n; ++ci) {
        if (threadIdx.x == ci) {
            n_tiles[ci] = count[ci];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Mode and counters

static std::atomic<int> moe_grouped_mode_override{-1};

static int moe_grouped_env_mode() {
    static const int mode = [] {
        // delphi: packed by default; it applies to NVIDIA Ampere and newer, other devices keep the old path
        const char * s = getenv("GGML_CUDA_MOE_GROUPED");
        if (s == nullptr) {
            return (int) GGML_CUDA_MOE_GROUPED_PACKED;
        }
        if (strcmp(s, "old") == 0 || strcmp(s, "0") == 0) {
            return (int) GGML_CUDA_MOE_GROUPED_OLD;
        }
        if (strcmp(s, "auto") == 0 || strcmp(s, "1") == 0) {
            return (int) GGML_CUDA_MOE_GROUPED_AUTO;
        }
        if (strcmp(s, "grouped") == 0 || strcmp(s, "2") == 0) {
            return (int) GGML_CUDA_MOE_GROUPED_GROUPED;
        }
        if (strcmp(s, "packed-f16") == 0) {
            return (int) GGML_CUDA_MOE_GROUPED_PACKED_F16;
        }
        if (strcmp(s, "packed-i8") == 0) {
            return (int) GGML_CUDA_MOE_GROUPED_PACKED_I8;
        }
        if (strcmp(s, "packed") == 0) {
            return (int) GGML_CUDA_MOE_GROUPED_PACKED;
        }
        GGML_LOG_WARN("%s: unknown GGML_CUDA_MOE_GROUPED=%s, using old\n", __func__, s);
        return (int) GGML_CUDA_MOE_GROUPED_OLD;
    }();
    return mode;
}

static int moe_grouped_mode() {
    const int mode = moe_grouped_mode_override.load(std::memory_order_relaxed);
    return mode >= 0 ? mode : moe_grouped_env_mode();
}

// delphi: the packed default was validated on NVIDIA only; other vendors need an explicit GGML_CUDA_MOE_GROUPED
static bool moe_grouped_mode_is_default() {
    static const bool unset = getenv("GGML_CUDA_MOE_GROUPED") == nullptr;
    return unset && moe_grouped_mode_override.load(std::memory_order_relaxed) < 0;
}

int ggml_cuda_moe_grouped_packed_mode(void) {
    const int mode = moe_grouped_mode();
    return mode >= GGML_CUDA_MOE_GROUPED_PACKED_F16 ? mode : (int) GGML_CUDA_MOE_GROUPED_OLD;
}

static int64_t moe_grouped_min_tokens() {
    static const int64_t n = [] {
        const char * s = getenv("GGML_CUDA_MOE_GROUPED_MIN_TOKENS");
        return s ? (int64_t) atoll(s) : int64_t(128);
    }();
    return n;
}

enum moe_grouped_count {
    MOE_GROUPED_COUNT_GROUPED,
    MOE_GROUPED_COUNT_BACKEND,     // not NVIDIA, or not compiled for this backend
    MOE_GROUPED_COUNT_TYPE,        // no grouped kernel for the type
    MOE_GROUPED_COUNT_SHAPE,       // src1 layout not supported by the schedule consumer
    MOE_GROUPED_COUNT_FEW_TOKENS,  // auto mode below GGML_CUDA_MOE_GROUPED_MIN_TOKENS
    MOE_GROUPED_COUNT_N,
};

static std::atomic<int64_t> moe_grouped_counts[GGML_CUDA_MAX_DEVICES][MOE_GROUPED_COUNT_N];

// GGML_CUDA_MOE_GROUPED_TIMING=1 times the tile-list kernel and every class launch with events, with one synchronization per call.
// It is for measurement only and does nothing while a stream is being captured. Class statistics are kept per J.
#define MOE_GROUPED_J_SLOTS (128/8 + 1)

struct moe_grouped_timing_stats {
    int64_t calls    = 0;
    double  tiles_us = 0.0;
    int64_t class_calls[MOE_GROUPED_J_SLOTS] = {};
    int64_t class_tiles[MOE_GROUPED_J_SLOTS] = {};
    double  class_us[MOE_GROUPED_J_SLOTS]    = {};
    double  fixup_us[MOE_GROUPED_J_SLOTS]    = {};
};

static moe_grouped_timing_stats moe_grouped_timing[GGML_CUDA_MAX_DEVICES];

static bool moe_grouped_timing_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_MOE_GROUPED_TIMING");
        return s != nullptr && atoi(s) != 0;
    }();
    return enabled;
}

static void moe_grouped_timing_log(int device) {
    moe_grouped_timing_stats & ts = moe_grouped_timing[device];
    if (ts.calls == 0) {
        return;
    }
    GGML_LOG_INFO("%s: CUDA%d grouped timing: %" PRId64 " calls, tile list %.2f us per call\n",
        __func__, device, ts.calls, ts.tiles_us/ts.calls);
    for (int s = 0; s < MOE_GROUPED_J_SLOTS; ++s) {
        if (ts.class_calls[s] == 0) {
            continue;
        }
        GGML_LOG_INFO("%s: CUDA%d   J=%3d: %" PRId64 " launches, %.1f list entries, kernel %.1f us, fixup %.1f us per launch, kernel %.3f us per entry\n",
            __func__, device, 8*s, ts.class_calls[s], double(ts.class_tiles[s])/ts.class_calls[s], ts.class_us[s]/ts.class_calls[s],
            ts.fixup_us[s]/ts.class_calls[s], ts.class_tiles[s] > 0 ? ts.class_us[s]/ts.class_tiles[s] : 0.0);
    }
    ts = moe_grouped_timing_stats();
}

static bool moe_grouped_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_G128:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_moe_grouped_select(int device, ggml_type type, int cc, int64_t n_tokens, bool shape_ok) {
    const int mode = moe_grouped_mode();
    if (mode == GGML_CUDA_MOE_GROUPED_OLD || (!GGML_CUDA_CC_IS_NVIDIA(cc) && moe_grouped_mode_is_default())) {
        return false;
    }

    int reason = MOE_GROUPED_COUNT_GROUPED;
#ifdef GGML_CUDA_MMQ_GROUPED_AVAILABLE
    if (!GGML_CUDA_CC_IS_NVIDIA(cc)) {
        reason = MOE_GROUPED_COUNT_BACKEND;
    } else if (!moe_grouped_type_supported(type)) {
        reason = MOE_GROUPED_COUNT_TYPE;
    } else if (!shape_ok) {
        reason = MOE_GROUPED_COUNT_SHAPE;
    } else if (mode == GGML_CUDA_MOE_GROUPED_AUTO && n_tokens < moe_grouped_min_tokens()) {
        reason = MOE_GROUPED_COUNT_FEW_TOKENS;
    }
#else
    reason = MOE_GROUPED_COUNT_BACKEND;
    GGML_UNUSED_VARS(type, cc, n_tokens, shape_ok);
#endif // GGML_CUDA_MMQ_GROUPED_AVAILABLE

    moe_grouped_counts[device][reason].fetch_add(1, std::memory_order_relaxed);
    return reason == MOE_GROUPED_COUNT_GROUPED;
}

// Counts since the previous log of the device, so that each backend reports its own calls. A call on the old path is a warning:
// a run that requested the grouped path must not look grouped while part of the routed work is not.
void ggml_cuda_moe_grouped_log_stats(int device) {
    static int64_t logged[GGML_CUDA_MAX_DEVICES][MOE_GROUPED_COUNT_N] = {};

    moe_grouped_timing_log(device);

    int64_t c[MOE_GROUPED_COUNT_N];
    int64_t total = 0;
    for (int i = 0; i < MOE_GROUPED_COUNT_N; ++i) {
        const int64_t now = moe_grouped_counts[device][i].load(std::memory_order_relaxed);
        c[i] = now - logged[device][i];
        logged[device][i] = now;
        total += c[i];
    }
    if (total == 0) {
        return;
    }
    const int64_t n_old = total - c[MOE_GROUPED_COUNT_GROUPED];
    const char * fmt = "%s: CUDA%d MUL_MAT_ID MMQ: %" PRId64 " grouped, %" PRId64 " old path (type %" PRId64 ", shape %" PRId64
        ", few tokens %" PRId64 ", backend %" PRId64 ")\n";
    if (n_old > 0) {
        GGML_LOG_WARN(fmt, __func__, device, c[MOE_GROUPED_COUNT_GROUPED], n_old,
            c[MOE_GROUPED_COUNT_TYPE], c[MOE_GROUPED_COUNT_SHAPE], c[MOE_GROUPED_COUNT_FEW_TOKENS], c[MOE_GROUPED_COUNT_BACKEND]);
    } else {
        GGML_LOG_INFO(fmt, __func__, device, c[MOE_GROUPED_COUNT_GROUPED], n_old,
            c[MOE_GROUPED_COUNT_TYPE], c[MOE_GROUPED_COUNT_SHAPE], c[MOE_GROUPED_COUNT_FEW_TOKENS], c[MOE_GROUPED_COUNT_BACKEND]);
    }
}

void ggml_backend_cuda_moe_grouped_set_mode(int mode) {
    moe_grouped_mode_override.store(mode < 0 ? -1 : std::min(mode, (int) GGML_CUDA_MOE_GROUPED_PACKED), std::memory_order_relaxed);
}

bool ggml_backend_cuda_moe_grouped_get_counts(int device, int64_t * counts) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    int64_t total = 0;
    for (int i = 0; i < MOE_GROUPED_COUNT_N; ++i) {
        total += moe_grouped_counts[device][i].load(std::memory_order_relaxed);
    }
    counts[0] = moe_grouped_counts[device][MOE_GROUPED_COUNT_GROUPED].load(std::memory_order_relaxed);
    counts[1] = total - counts[0];
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Execution

#ifdef GGML_CUDA_MMQ_GROUPED_AVAILABLE

struct moe_grouped_class_spec {
    int n = 0;
    int J[MOE_GROUPED_MAX_CLASSES];
    int rows_max[MOE_GROUPED_MAX_CLASSES];
};

static bool moe_grouped_J_instantiated(int J) {
    for (int j : { MMQ_GROUPED_J_LIST }) {
        if (j == J) {
            return true;
        }
    }
    return false;
}

static moe_grouped_class_spec moe_grouped_parse_classes() {
    moe_grouped_class_spec spec;
    const char * s = getenv("GGML_CUDA_MOE_GROUPED_CLASSES");
    const std::string text = s ? s : "8:8,16:16,24:24,32:32,48:48,64:64,96:96,128";
    size_t pos = 0;
    while (pos < text.size() && spec.n < MOE_GROUPED_MAX_CLASSES) {
        const size_t end  = std::min(text.find(',', pos), text.size());
        const std::string item = text.substr(pos, end - pos);
        const size_t colon = item.find(':');
        const int J = atoi(item.substr(0, colon).c_str());
        if (!moe_grouped_J_instantiated(J)) {
            GGML_ABORT("GGML_CUDA_MOE_GROUPED_CLASSES: no grouped instance for J=%d", J);
        }
        spec.J[spec.n]        = J;
        spec.rows_max[spec.n] = colon == std::string::npos ? INT32_MAX : atoi(item.substr(colon + 1).c_str());
        spec.n++;
        pos = end + 1;
    }
    GGML_ASSERT(spec.n > 0);
    spec.rows_max[spec.n - 1] = INT32_MAX;
    return spec;
}

static void moe_grouped_switch_type(ggml_type type, const mmq_grouped_args & args, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q1_0:    mul_mat_q_grouped_case<GGML_TYPE_Q1_0>   (args, stream); break;
        case GGML_TYPE_Q2_0:    mul_mat_q_grouped_case<GGML_TYPE_Q2_0>   (args, stream); break;
        case GGML_TYPE_Q4_0:    mul_mat_q_grouped_case<GGML_TYPE_Q4_0>   (args, stream); break;
        case GGML_TYPE_Q4_G128: mul_mat_q_grouped_case<GGML_TYPE_Q4_G128>(args, stream); break;
        case GGML_TYPE_Q4_1:    mul_mat_q_grouped_case<GGML_TYPE_Q4_1>   (args, stream); break;
        case GGML_TYPE_Q5_0:    mul_mat_q_grouped_case<GGML_TYPE_Q5_0>   (args, stream); break;
        case GGML_TYPE_Q5_1:    mul_mat_q_grouped_case<GGML_TYPE_Q5_1>   (args, stream); break;
        case GGML_TYPE_Q8_0:    mul_mat_q_grouped_case<GGML_TYPE_Q8_0>   (args, stream); break;
        case GGML_TYPE_Q2_K:    mul_mat_q_grouped_case<GGML_TYPE_Q2_K>   (args, stream); break;
        case GGML_TYPE_Q3_K:    mul_mat_q_grouped_case<GGML_TYPE_Q3_K>   (args, stream); break;
        case GGML_TYPE_Q4_K:    mul_mat_q_grouped_case<GGML_TYPE_Q4_K>   (args, stream); break;
        case GGML_TYPE_Q5_K:    mul_mat_q_grouped_case<GGML_TYPE_Q5_K>   (args, stream); break;
        case GGML_TYPE_Q6_K:    mul_mat_q_grouped_case<GGML_TYPE_Q6_K>   (args, stream); break;
        case GGML_TYPE_IQ1_S:   mul_mat_q_grouped_case<GGML_TYPE_IQ1_S>  (args, stream); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_grouped_case<GGML_TYPE_IQ2_XXS>(args, stream); break;
        case GGML_TYPE_IQ2_XS:  mul_mat_q_grouped_case<GGML_TYPE_IQ2_XS> (args, stream); break;
        case GGML_TYPE_IQ2_S:   mul_mat_q_grouped_case<GGML_TYPE_IQ2_S>  (args, stream); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_grouped_case<GGML_TYPE_IQ3_XXS>(args, stream); break;
        case GGML_TYPE_IQ3_S:   mul_mat_q_grouped_case<GGML_TYPE_IQ3_S>  (args, stream); break;
        case GGML_TYPE_IQ4_NL:  mul_mat_q_grouped_case<GGML_TYPE_IQ4_NL> (args, stream); break;
        case GGML_TYPE_IQ4_XS:  mul_mat_q_grouped_case<GGML_TYPE_IQ4_XS> (args, stream); break;
        default:
            GGML_ABORT("grouped MMQ: unsupported type %s", ggml_type_name(type));
    }
}

static bool moe_grouped_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

// ev: [0] before the tile list, [1] after it, [2 + 2*c] between the kernel and the fixup of class c, [3 + 2*c] after the fixup.
static void moe_grouped_timing_record(int device, const moe_grouped_classes & cls, const int32_t * n_tiles_d, const cudaEvent_t * ev,
        cudaStream_t stream) {
    int32_t n_tiles[MOE_GROUPED_MAX_CLASSES];
    CUDA_CHECK(cudaMemcpyAsync(n_tiles, n_tiles_d, cls.n*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    moe_grouped_timing_stats & ts = moe_grouped_timing[device];
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
    ts.calls++;
    ts.tiles_us += 1e3*ms;
    for (int c = 0; c < cls.n; ++c) {
        const int s = cls.J[c]/8;
        float kernel_ms, fixup_ms;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev[1 + 2*c], ev[2 + 2*c]));
        CUDA_CHECK(cudaEventElapsedTime(&fixup_ms,  ev[2 + 2*c], ev[3 + 2*c]));
        ts.class_calls[s]++;
        ts.class_tiles[s] += n_tiles[c];
        ts.class_us[s]    += 1e3*kernel_ms;
        ts.fixup_us[s]    += 1e3*fixup_ms;
    }
}

// A class whose J is not usable (J_ok) moves to the next wider usable J; adjacent classes with the same J merge.
template <typename J_ok_t>
static moe_grouped_classes moe_grouped_make_classes(const J_ok_t & J_ok, const int64_t n_assign, const int n_experts, int64_t & n_tiles_max) {
    static const moe_grouped_class_spec spec = moe_grouped_parse_classes();

    moe_grouped_classes cls;
    cls.n = 0;
    n_tiles_max = 0;
    for (int c = 0; c < spec.n; ++c) {
        int J = spec.J[c];
        for (; J <= 128; J += 8) {
            if (moe_grouped_J_instantiated(J) && J_ok(J)) {
                break;
            }
        }
        GGML_ASSERT(J <= 128);
        if (cls.n > 0 && cls.J[cls.n - 1] == J) {
            cls.rows_max[cls.n - 1] = spec.rows_max[c];
            continue;
        }
        cls.J[cls.n]        = J;
        cls.rows_max[cls.n] = spec.rows_max[c];
        cls.base[cls.n]     = (int) n_tiles_max;
        // sum over the experts of ceil(rows/J) <= n_assign/J + n_experts
        n_tiles_max += (n_assign + J - 1)/J + n_experts;
        cls.n++;
    }
    GGML_ASSERT(n_tiles_max <= INT32_MAX);
    return cls;
}

static cudaEvent_t moe_grouped_events[GGML_CUDA_MAX_DEVICES][2 + 2*MOE_GROUPED_MAX_CLASSES] = {};

// Builds the tile lists and calls launch(c, tiles, n_tiles, max_tiles, ev_mid) for every class. The launcher records ev_mid
// (if not null) between its kernel and its fixup kernel.
template <typename launch_t>
static void moe_grouped_run(ggml_backend_cuda_context & ctx, const moe_grouped_classes & cls, const int64_t n_tiles_max,
        const int32_t * expert_bounds, const int n_experts, cudaStream_t stream, const launch_t & launch) {
    const int id = ggml_cuda_get_device();

    ggml_cuda_pool_alloc<int2>    tiles(ctx.pool(), n_tiles_max);
    ggml_cuda_pool_alloc<int32_t> n_tiles(ctx.pool(), cls.n);

    const bool    timing = moe_grouped_timing_enabled() && !moe_grouped_capturing(stream);
    cudaEvent_t * ev     = moe_grouped_events[id];
    if (timing && ev[0] == nullptr) {
        for (int k = 0; k < 2 + 2*MOE_GROUPED_MAX_CLASSES; ++k) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ev[k], cudaEventDefault));
        }
    }
    if (timing) {
        CUDA_CHECK(cudaEventRecord(ev[0], stream));
    }

    moe_grouped_tiles<<<1, MOE_GROUPED_TILES_BLOCK, 0, stream>>>(expert_bounds, n_experts, cls, tiles.get(), n_tiles.get());
    CUDA_CHECK(cudaGetLastError());
    if (timing) {
        CUDA_CHECK(cudaEventRecord(ev[1], stream));
    }

    for (int c = 0; c < cls.n; ++c) {
        const int64_t max_tiles = (c + 1 < cls.n ? cls.base[c + 1] : n_tiles_max) - cls.base[c];
        launch(c, tiles.get() + cls.base[c], n_tiles.get() + c, max_tiles, timing ? ev[2 + 2*c] : nullptr);
        if (timing) {
            CUDA_CHECK(cudaEventRecord(ev[3 + 2*c], stream));
        }
    }

    if (timing) {
        moe_grouped_timing_record(id, cls, n_tiles.get(), ev, stream);
    }
}

void ggml_cuda_mul_mat_q_grouped(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int  id        = ggml_cuda_get_device();
    const int  cc        = ggml_cuda_info().devices[id].cc;
    const int  nsm       = ggml_cuda_info().devices[id].nsm;
    const bool fallback  = args.nrows_x % 128 != 0;
    const int  n_experts = (int) args.nchannels_x;

    int64_t n_tiles_max;
    const moe_grouped_classes cls = moe_grouped_make_classes([&](int J) {
        return ggml_cuda_mmq_get_config(args.type_x, J, fallback, cc).type != GGML_TYPE_COUNT;
    }, args.ncols_y, n_experts, n_tiles_max);

    int J_max = 0;
    int I     = 0;
    for (int c = 0; c < cls.n; ++c) {
        J_max = std::max(J_max, cls.J[c]);
        I     = std::max(I, ggml_cuda_mmq_get_config(args.type_x, cls.J[c], fallback, cc).I);
    }
    ggml_cuda_pool_alloc<float> tmp_fixup(ctx.pool(), (size_t) nsm*J_max*I);

    moe_grouped_run(ctx, cls, n_tiles_max, args.expert_bounds, n_experts, stream,
            [&](int c, const int2 * tiles, const int32_t * n_tiles, int64_t max_tiles, cudaEvent_t ev_mid) {
        const mmq_grouped_args gargs = {
            args.x, args.y, args.ids_dst, args.expert_bounds, args.dst,
            tiles, n_tiles, max_tiles, tmp_fixup.get(),
            args.ncols_x, args.nrows_x, args.stride_row_x, args.ncols_y, args.nrows_dst, args.stride_channel_x,
            cls.J[c], ev_mid };
        moe_grouped_switch_type(args.type_x, gargs, stream);
    });
}

void ggml_cuda_mul_mat_id_packed(ggml_backend_cuda_context & ctx, int fmt, const moe_packed_weights & w, const char * y, int64_t ncols_y,
        const int32_t * ids_dst, const int32_t * expert_bounds, float * dst, int64_t nrows, int64_t ncols_k, int n_experts,
        int64_t n_assign, int64_t stride_col_dst, cudaStream_t stream) {
    int64_t n_tiles_max;
    const moe_grouped_classes cls = moe_grouped_make_classes([](int) { return true; }, n_assign, n_experts, n_tiles_max);

    moe_grouped_run(ctx, cls, n_tiles_max, expert_bounds, n_experts, stream,
            [&](int c, const int2 * tiles, const int32_t * n_tiles, int64_t max_tiles, cudaEvent_t ev_mid) {
        GGML_UNUSED(max_tiles);
        const moe_packed_launch_args pargs = { w, y, ncols_y, ids_dst, expert_bounds, tiles, n_tiles, dst, nrows, ncols_k, stride_col_dst };
        ggml_cuda_moe_packed_launch(fmt, cls.J[c], pargs, stream);
        if (ev_mid) {
            CUDA_CHECK(cudaEventRecord(ev_mid, stream));
        }
    });
}

#else

void ggml_cuda_mul_mat_q_grouped(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    GGML_UNUSED_VARS(ctx, args, stream);
    GGML_ABORT("grouped MMQ is not available for this backend");
}

void ggml_cuda_mul_mat_id_packed(ggml_backend_cuda_context & ctx, int fmt, const moe_packed_weights & w, const char * y, int64_t ncols_y,
        const int32_t * ids_dst, const int32_t * expert_bounds, float * dst, int64_t nrows, int64_t ncols_k, int n_experts,
        int64_t n_assign, int64_t stride_col_dst, cudaStream_t stream) {
    GGML_UNUSED_VARS(ctx, fmt, w, y, ncols_y, ids_dst, expert_bounds, dst, nrows, ncols_k, n_experts, n_assign, stride_col_dst, stream);
    GGML_ABORT("grouped MMQ is not available for this backend");
}

#endif // GGML_CUDA_MMQ_GROUPED_AVAILABLE
