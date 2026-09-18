#include "moe-schedule.cuh"
#include "mmid.cuh"

#include "ggml-backend-impl.h"

#include <atomic>
#include <cstring>
#include <vector>

// Assignments per block of the count and place kernels. Nothing here depends on the warp size.
#define MOE_SCHEDULE_BLOCK 256
// Threads of the single-block scan. Each thread owns a contiguous chunk of experts.
#define MOE_SCHEDULE_SCAN  1024

// The schedule is a stable counting sort of the assignments by expert, blocked so that it needs no global atomics:
// count builds one histogram per block, scan turns the histograms into per-block write bases, and place writes each
// assignment at offsets[e] + base[b][e] + its rank among earlier assignments of the same expert in the block.
// Work is O(n_assign + n_experts*n_blocks); mm_ids_helper instead scans all assignments once per expert.

static __device__ __forceinline__ int32_t moe_schedule_id(
        const int32_t * __restrict__ ids, const int64_t ids_s1, const int top_k, const int n_experts, const int a) {
    const int32_t e = ids[(int64_t) (a / top_k)*ids_s1 + a % top_k];
    return e >= 0 && e < n_experts ? e : -1;
}

// hist[b*n_experts + e] = assignments of block b routed to expert e. Every entry is written, so no clearing pass is needed.
static __global__ void moe_schedule_count(
        const int32_t * __restrict__ ids, const int64_t ids_s1, const int n_assign, const int top_k, const int n_experts,
        int32_t * __restrict__ hist) {
    extern __shared__ int32_t sh_count[];

    for (int e = threadIdx.x; e < n_experts; e += MOE_SCHEDULE_BLOCK) {
        sh_count[e] = 0;
    }
    __syncthreads();

    const int a = blockIdx.x*MOE_SCHEDULE_BLOCK + threadIdx.x;
    if (a < n_assign) {
        const int32_t e = moe_schedule_id(ids, ids_s1, top_k, n_experts, a);
        if (e >= 0) {
            atomicAdd(&sh_count[e], 1);
        }
    }
    __syncthreads();

    for (int e = threadIdx.x; e < n_experts; e += MOE_SCHEDULE_BLOCK) {
        hist[(int64_t) blockIdx.x*n_experts + e] = sh_count[e];
    }
}

// One block. Replaces hist[b][e] by the first row of block b inside expert e, and writes offsets.
// offsets[e] holds the total of expert e until the same thread overwrites it with the expert base.
static __global__ void moe_schedule_scan(
        int32_t * __restrict__ hist, const int n_blocks, const int n_experts, int32_t * __restrict__ offsets) {
    __shared__ int32_t sh_scan[MOE_SCHEDULE_SCAN];

    const int chunk   = (n_experts + MOE_SCHEDULE_SCAN - 1) / MOE_SCHEDULE_SCAN;
    const int e_begin = min((int) threadIdx.x*chunk, n_experts);
    const int e_end   = min(e_begin + chunk, n_experts);

    int32_t sum = 0;
    for (int e = e_begin; e < e_end; ++e) {
        int32_t run = 0;
#pragma unroll 8
        for (int b = 0; b < n_blocks; ++b) {
            const int32_t count = hist[(int64_t) b*n_experts + e];
            hist[(int64_t) b*n_experts + e] = run;
            run += count;
        }
        offsets[e] = run;
        sum += run;
    }

    sh_scan[threadIdx.x] = sum;
    __syncthreads();
    for (int off = 1; off < MOE_SCHEDULE_SCAN; off <<= 1) {
        const int32_t x = (int) threadIdx.x >= off ? sh_scan[threadIdx.x - off] : 0;
        __syncthreads();
        sh_scan[threadIdx.x] += x;
        __syncthreads();
    }

    int32_t base = sh_scan[threadIdx.x] - sum;
    for (int e = e_begin; e < e_end; ++e) {
        const int32_t total = offsets[e];
        offsets[e] = base;
        base += total;
    }
    if (threadIdx.x == MOE_SCHEDULE_SCAN - 1) {
        offsets[n_experts] = sh_scan[threadIdx.x];
    }
}

static __global__ void moe_schedule_place(
        const int32_t * __restrict__ ids, const int64_t ids_s1, const int n_assign, const int top_k, const int n_experts,
        const int32_t * __restrict__ hist, const int32_t * __restrict__ offsets,
        int32_t * __restrict__ slot, int32_t * __restrict__ inverse) {
    __shared__ int32_t sh_expert[MOE_SCHEDULE_BLOCK];

    const int     a = blockIdx.x*MOE_SCHEDULE_BLOCK + threadIdx.x;
    const int32_t e = a < n_assign ? moe_schedule_id(ids, ids_s1, top_k, n_experts, a) : -1;
    sh_expert[threadIdx.x] = e;
    __syncthreads();

    if (e < 0) {
        if (a < n_assign && inverse) {
            inverse[a] = -1;
        }
        return;
    }

    int rank = 0;
    for (int j = 0; j < (int) threadIdx.x; ++j) {
        rank += sh_expert[j] == e;
    }

    const int32_t r = offsets[e] + hist[(int64_t) blockIdx.x*n_experts + e] + rank;
    slot[r] = a;
    if (inverse) {
        inverse[a] = r;
    }
}

size_t ggml_cuda_moe_schedule_scratch_size(int n_experts, int64_t n_assign) {
    const int64_t n_blocks = (n_assign + MOE_SCHEDULE_BLOCK - 1) / MOE_SCHEDULE_BLOCK;
    return (size_t) std::max<int64_t>(n_blocks, 1)*n_experts;
}

void ggml_cuda_moe_schedule_build(
        const int32_t * ids, int64_t ids_s1, int64_t n_tokens, int top_k, int n_experts,
        int32_t * offsets, int32_t * slot, int32_t * inverse, int32_t * scratch, cudaStream_t stream) {
    const int64_t n_assign = n_tokens*top_k;
    GGML_ASSERT(n_experts > 0 && top_k > 0 && n_assign <= INT32_MAX);

    if (n_assign == 0) {
        CUDA_CHECK(cudaMemsetAsync(offsets, 0, ((size_t) n_experts + 1)*sizeof(int32_t), stream));
        return;
    }

    const int    n_blocks      = (int) ((n_assign + MOE_SCHEDULE_BLOCK - 1) / MOE_SCHEDULE_BLOCK);
    const size_t nbytes_shared = (size_t) n_experts*sizeof(int32_t);
    GGML_ASSERT(nbytes_shared <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    moe_schedule_count<<<n_blocks, MOE_SCHEDULE_BLOCK, nbytes_shared, stream>>>(
        ids, ids_s1, (int) n_assign, top_k, n_experts, scratch);
    moe_schedule_scan<<<1, MOE_SCHEDULE_SCAN, 0, stream>>>(scratch, n_blocks, n_experts, offsets);
    moe_schedule_place<<<n_blocks, MOE_SCHEDULE_BLOCK, 0, stream>>>(
        ids, ids_s1, (int) n_assign, top_k, n_experts, scratch, offsets, slot, inverse);
    CUDA_CHECK(cudaGetLastError());
}

static std::atomic<int> moe_schedule_mode{-1};

bool ggml_cuda_moe_schedule_enabled(void) {
    const int mode = moe_schedule_mode.load(std::memory_order_relaxed);
    if (mode >= 0) {
        return mode != 0;
    }
    static const bool env = [] {
        const char * s = getenv("GGML_CUDA_MOE_SCHEDULE");
        return s != nullptr && atoi(s) != 0;
    }();
    return env;
}

void ggml_backend_cuda_moe_schedule_set_mode(int mode) {
    moe_schedule_mode.store(mode < 0 ? -1 : (mode != 0), std::memory_order_relaxed);
}

bool ggml_backend_cuda_moe_schedule_build_host(
        ggml_backend_t backend, int arm, const int32_t * ids, int64_t n_tokens, int top_k, int n_experts,
        int32_t * offsets, int32_t * slot, int32_t * inverse, int reps, float * device_us) {
    if (!ggml_backend_is_cuda(backend) || (arm != 0 && arm != 1) || n_tokens <= 0 || top_k <= 0 || n_experts <= 0) {
        return false;
    }

    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);
    cudaStream_t stream = ctx->stream();

    // mm_ids_helper keeps one int32 per token in shared memory and packs the top-k position into 10 bits
    if (arm == 1 && (n_tokens*sizeof(int32_t) > ggml_cuda_info().devices[ctx->device].smpbo || top_k >= (1 << 10))) {
        return false;
    }

    // One allocation with a guard band after every region, so a write out of bounds shows up even where compute-sanitizer does not run.
    // Regions: ids, offsets, slot, inverse, scratch.
    const int64_t  n_assign = n_tokens*top_k;
    const size_t   guard    = 64;
    const uint32_t pattern  = 0x5a5a5a5a;
    const size_t   sizes[5] = { (size_t) n_assign, (size_t) n_experts + 1, (size_t) n_assign, (size_t) n_assign,
                                ggml_cuda_moe_schedule_scratch_size(n_experts, n_assign) };
    size_t begin[5];
    size_t total = guard;
    for (int i = 0; i < 5; ++i) {
        begin[i] = total;
        total   += sizes[i] + guard;
    }
    ggml_cuda_pool_alloc<int32_t> buf(ctx->pool(), total);
    CUDA_CHECK(cudaMemsetAsync(buf.get(), pattern & 0xff, total*sizeof(int32_t), stream));
    int32_t * d_ids     = buf.get() + begin[0];
    int32_t * d_offsets = buf.get() + begin[1];
    int32_t * d_slot    = buf.get() + begin[2];
    int32_t * d_inverse = buf.get() + begin[3];
    int32_t * d_scratch = buf.get() + begin[4];

    CUDA_CHECK(cudaMemcpyAsync(d_ids, ids, n_assign*sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    auto build = [&]() {
        if (arm == 0) {
            ggml_cuda_moe_schedule_build(d_ids, top_k, n_tokens, top_k, n_experts, d_offsets, d_slot, d_inverse, d_scratch, stream);
        } else {
            ggml_cuda_launch_mm_ids_helper(d_ids, d_inverse, d_slot, d_offsets,
                n_experts, (int) n_tokens, top_k, /*nchannels_y =*/ 1, /*si1 =*/ top_k, /*sis1 =*/ 1, /*write_inverse =*/ true, stream);
        }
    };
    build();

    if (device_us && reps > 0) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&stop, cudaEventDefault));
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) {
            build();
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        *device_us = 1000.0f*ms/reps;
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    std::vector<int32_t> host(total);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), buf.get(), total*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < 5; ++i) {
        const size_t g0 = i == 0 ? 0 : begin[i - 1] + sizes[i - 1];
        for (size_t j = g0; j < begin[i]; ++j) {
            if ((uint32_t) host[j] != pattern) {
                GGML_LOG_ERROR("%s: arm %d wrote before region %d\n", __func__, arm, i);
                return false;
            }
        }
    }
    for (size_t j = begin[4] + sizes[4]; j < total; ++j) {
        if ((uint32_t) host[j] != pattern) {
            GGML_LOG_ERROR("%s: arm %d wrote after the last region\n", __func__, arm);
            return false;
        }
    }

    memcpy(offsets, host.data() + begin[1], sizes[1]*sizeof(int32_t));
    memcpy(slot,    host.data() + begin[2], sizes[2]*sizeof(int32_t));
    memcpy(inverse, host.data() + begin[3], sizes[3]*sizeof(int32_t));
    return true;
}
