#include "moe-devsort.cuh"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

// Block size for the routing kernels. Chosen only for occupancy; nothing here depends on the
// warp/wave size, so the same value is correct on RDNA (wave32), CDNA (wave64) and NVIDIA.
#define MOE_ROUTE_BLOCK 256

// Crossover for GGML_CUDA_MOE_ROUTE=auto, in routed slots (n_tokens * n_expert_used).
//
// Measured on gfx1201 the device arm costs 96-129 us per call and barely grows with the shape,
// while the host arm ranges from 120 us at its cheapest to 3.5 ms at 16384 slots. The device arm is
// therefore cheaper at routing in every shape that can reach this path, and the default routes
// everything on the device.
//
// Slot count, not the host loop's iteration count (n_experts * n_slots), is what tracks end-to-end
// prefill: measured across three models the effect is flat to slightly negative below 4096 slots
// (worst case -3.4% on a 128-expert F16 model at 2048 slots, where device routing is nonetheless
// 2.6x cheaper, so the loss is launch and pool overhead rather than routing) and a large win from
// 8192 slots up (+17% to +39%). Two models with the same n_experts * n_slots landed on opposite
// sides of that, so the knob counts slots.
#define MOE_ROUTE_MIN_WORK_DEFAULT 0

// ids layout, shared by every reader here and by the host arm in ggml-cuda.cu
static __device__ __forceinline__ int32_t moe_ids_at(const char * ids, size_t nb0, size_t nb1, int i12, int iex) {
    return *(const int32_t *)(ids + (size_t) i12 * nb1 + (size_t) iex * nb0);
}

// The routing is a stable counting sort over the (token, slot) pairs, keyed by expert. Stability
// matters: the per-expert GEMM that consumes the sorted rows is not invariant to the order of its
// rows on this hardware, so an arbitrary within-expert order would make every run produce slightly
// different logits. Sorting stably by slot reproduces the host loop's order exactly, which keeps the
// two arms bit-identical and keeps a run reproducible.
//
// A plain atomicAdd cursor per expert cannot do that, so the sort is blocked: each block builds its
// own histogram, the histograms are scanned per expert across blocks, and each slot then lands at
// expert_base + block_base + its rank inside the block.

// Per-block histogram, written to block_hist[expert][block]. Every entry is written, so no separate
// clearing pass is needed. Uses n_experts int32 of dynamic shared memory.
static __global__ void k_moe_block_hist(const char * __restrict__ ids, int n_slots, int n_expert_used, int n_experts,
                                        int n_blocks, size_t nb0, size_t nb1, int32_t * __restrict__ block_hist) {
    extern __shared__ int32_t sh_hist[];

    for (int i = threadIdx.x; i < n_experts; i += MOE_ROUTE_BLOCK) {
        sh_hist[i] = 0;
    }
    __syncthreads();

    const int slot = blockIdx.x * MOE_ROUTE_BLOCK + threadIdx.x;
    if (slot < n_slots) {
        const int32_t e = moe_ids_at(ids, nb0, nb1, slot / n_expert_used, slot % n_expert_used);
        // A malformed expert index is a release-stripped assert on the host arm but an out-of-bounds
        // device write here, so it is dropped instead. Well-formed ids never trip this.
        if (e >= 0 && e < n_experts) {
            atomicAdd(&sh_hist[e], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n_experts; i += MOE_ROUTE_BLOCK) {
        block_hist[(size_t) i * n_blocks + blockIdx.x] = sh_hist[i];
    }
}

// Serial exclusive scan of one row of block_hist, one block per expert. The row length is the block
// count, which stays small because each block covers MOE_ROUTE_BLOCK slots.
static __global__ void k_moe_scan_blocks(int32_t * __restrict__ block_hist, int n_blocks,
                                         int32_t * __restrict__ counts) {
    __shared__ int32_t tile[MOE_ROUTE_BLOCK];
    __shared__ int32_t carry;

    int32_t * row = block_hist + (size_t) blockIdx.x * n_blocks;

    if (threadIdx.x == 0) {
        carry = 0;
    }
    __syncthreads();

    for (int base = 0; base < n_blocks; base += MOE_ROUTE_BLOCK) {
        const int i = base + threadIdx.x;
        tile[threadIdx.x] = i < n_blocks ? row[i] : 0;
        __syncthreads();

        if (threadIdx.x == 0) {
            int32_t acc = carry;
            for (int j = 0; j < MOE_ROUTE_BLOCK; ++j) {
                const int32_t v = tile[j];
                tile[j] = acc;
                acc += v;
            }
            carry = acc;
        }
        __syncthreads();

        if (i < n_blocks) {
            row[i] = tile[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        counts[blockIdx.x] = carry;
    }
}

// Exclusive prefix sum over the experts, in one block. The expert count is small (<= 256 on current
// models) so a serial scan over a shared-memory tile is enough; a multi-block scan would need a
// second launch to carry the block sums.
static __global__ void k_moe_scan_experts(const int32_t * __restrict__ in, int32_t * __restrict__ out, int n) {
    __shared__ int32_t tile[MOE_ROUTE_BLOCK];
    __shared__ int32_t carry;

    if (threadIdx.x == 0) {
        carry = 0;
    }
    __syncthreads();

    for (int base = 0; base < n; base += MOE_ROUTE_BLOCK) {
        const int i = base + threadIdx.x;
        tile[threadIdx.x] = i < n ? in[i] : 0;
        __syncthreads();

        if (threadIdx.x == 0) {
            int32_t acc = carry;
            for (int j = 0; j < MOE_ROUTE_BLOCK; ++j) {
                const int32_t v = tile[j];
                tile[j] = acc;
                acc += v;
            }
            carry = acc;
        }
        __syncthreads();

        if (i < n) {
            out[i] = tile[threadIdx.x];
        }
        __syncthreads();
    }
}

// Final placement. Each slot's rank inside its block is counted directly from the block's expert
// list, which is what makes the sort stable.
static __global__ void k_moe_place(const char * __restrict__ ids, int n_slots, int n_expert_used, int n_experts,
                                   int n_blocks, int ne11, size_t nb0, size_t nb1,
                                   const int32_t * __restrict__ expert_base, const int32_t * __restrict__ block_base,
                                   int32_t * __restrict__ to_sorted, int32_t * __restrict__ from_sorted) {
    __shared__ int32_t sh_expert[MOE_ROUTE_BLOCK];

    const int slot = blockIdx.x * MOE_ROUTE_BLOCK + threadIdx.x;
    const int i12  = slot / n_expert_used;
    const int iex  = slot % n_expert_used;

    int32_t e = -1;
    if (slot < n_slots) {
        e = moe_ids_at(ids, nb0, nb1, i12, iex);
        if (e < 0 || e >= n_experts) {
            e = -1; // same guard as k_moe_block_hist: this slot was never counted
        }
    }
    sh_expert[threadIdx.x] = e;
    __syncthreads();

    if (e < 0) {
        return;
    }

    int rank = 0;
    for (int j = 0; j < (int) threadIdx.x; ++j) {
        rank += sh_expert[j] == e;
    }

    const int pos     = expert_base[e] + block_base[(size_t) e * n_blocks + blockIdx.x] + rank;
    to_sorted[pos]    = i12 * ne11 + iex % ne11;
    from_sorted[slot] = pos;
}

static void moe_route_device(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         ids,
        int64_t ne02, int64_t ne11, int64_t ne12, int64_t n_expert_used,
        int32_t *                   ids_buf_dev,
        std::vector<int32_t> &      tokens_per_expert,
        cudaStream_t                stream) {
    const int n_slots   = (int) (ne12 * n_expert_used);
    const int n_experts = (int) ne02;
    const int n_blocks  = (n_slots + MOE_ROUTE_BLOCK - 1) / MOE_ROUTE_BLOCK;

    GGML_ASSERT(n_slots > 0 && n_experts > 0); // an empty grid is an invalid launch configuration
    GGML_ASSERT((size_t) n_experts*sizeof(int32_t) <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    ggml_cuda_pool_alloc<int32_t> scratch(ctx.pool(), (size_t) n_experts*(n_blocks + 2));
    int32_t * block_hist  = scratch.ptr;                                // per block count, expert-major
    int32_t * counts      = scratch.ptr + (size_t) n_experts*n_blocks;  // tokens per expert, read back
    int32_t * expert_base = counts + n_experts;                         // exclusive scan of counts

    k_moe_block_hist<<<n_blocks, MOE_ROUTE_BLOCK, n_experts*sizeof(int32_t), stream>>>(
        (const char *) ids->data, n_slots, (int) n_expert_used, n_experts, n_blocks,
        ids->nb[0], ids->nb[1], block_hist);

    k_moe_scan_blocks<<<n_experts, MOE_ROUTE_BLOCK, 0, stream>>>(block_hist, n_blocks, counts);

    k_moe_scan_experts<<<1, MOE_ROUTE_BLOCK, 0, stream>>>(counts, expert_base, n_experts);

    k_moe_place<<<n_blocks, MOE_ROUTE_BLOCK, 0, stream>>>(
        (const char *) ids->data, n_slots, (int) n_expert_used, n_experts, n_blocks, (int) ne11,
        ids->nb[0], ids->nb[1], expert_base, block_hist, ids_buf_dev, ids_buf_dev + n_slots);

    CUDA_CHECK(cudaGetLastError());

    // Only the per-expert counts have to reach the host; they size the per-expert GEMMs.
    CUDA_CHECK(cudaMemcpyAsync(tokens_per_expert.data(), counts, n_experts*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Upstream host routing, unchanged apart from being moved out of ggml_cuda_mul_mat_id.
static void moe_route_host(
        const ggml_tensor *    ids,
        int64_t ne02, int64_t ne11, int64_t ne12, int64_t n_expert_used,
        int32_t *              ids_buf_dev,
        std::vector<int32_t> & tokens_per_expert,
        cudaStream_t           stream) {
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Check the device routing against the routing the host arm would produce, element by element.
//
// The device sort is stable, so the two arms must agree exactly - not merely up to a permutation
// inside each expert. That is worth checking strictly: the per-expert GEMM is not invariant to the
// order of its rows on this hardware, so a grouping that is only correct up to a permutation would
// still change the model's output.
static void moe_route_verify(
        const ggml_tensor *          ids,
        int64_t ne02, int64_t ne11, int64_t ne12, int64_t n_expert_used,
        const int32_t *              ids_buf_dev,
        const std::vector<int32_t> & tokens_per_expert,
        cudaStream_t                 stream) {
    const int64_t n_slots = ne12 * n_expert_used;

    std::vector<char> ids_host(ggml_nbytes(ids));
    std::vector<int32_t> buf(2*n_slots);
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.data(), ids_buf_dev, 2*n_slots*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The host arm's loop, verbatim, against the same ids.
    std::vector<int32_t> to_sorted_ref;
    to_sorted_ref.reserve(n_slots);
    std::vector<int32_t> from_sorted_ref(n_slots);
    std::vector<int32_t> counts_ref(ne02, 0);
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t e = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                GGML_ASSERT(e >= 0 && e < ne02);
                if (e == i02) {
                    from_sorted_ref[i12*n_expert_used + iex] = to_sorted_ref.size();
                    to_sorted_ref.push_back(i12*ne11 + iex % ne11);
                    counts_ref[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(to_sorted_ref.size() == (size_t) n_slots);

    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (counts_ref[i02] != tokens_per_expert[i02]) {
            GGML_LOG_ERROR("%s: expert %" PRId64 " count %d, host reference %d\n",
                __func__, i02, tokens_per_expert[i02], counts_ref[i02]);
            GGML_ABORT("MoE device routing count mismatch");
        }
    }
    for (int64_t i = 0; i < n_slots; ++i) {
        if (buf[i] != to_sorted_ref[i]) {
            GGML_LOG_ERROR("%s: ids_to_sorted[%" PRId64 "] = %d, host reference %d\n",
                __func__, i, buf[i], to_sorted_ref[i]);
            GGML_ABORT("MoE device routing gather mismatch");
        }
        if (buf[n_slots + i] != from_sorted_ref[i]) {
            GGML_LOG_ERROR("%s: ids_from_sorted[%" PRId64 "] = %d, host reference %d\n",
                __func__, i, buf[n_slots + i], from_sorted_ref[i]);
            GGML_ABORT("MoE device routing scatter mismatch");
        }
    }
}

enum moe_route_mode { MOE_ROUTE_AUTO, MOE_ROUTE_DEVICE, MOE_ROUTE_HOST };

struct moe_route_config {
    moe_route_mode mode     = MOE_ROUTE_AUTO;
    int64_t        min_work = MOE_ROUTE_MIN_WORK_DEFAULT;
    bool           debug    = false;
    bool           verify   = false;
};

// Accumulated per-arm cost of the routing section. Debug mode drains the stream before the arm runs
// and reports that wait separately: the drain is queued GPU work that both arms pay for, while the
// remainder is the extra stall the device arm is meant to remove.
struct moe_route_stats {
    std::atomic<uint64_t> calls   [2];
    std::atomic<uint64_t> slots   [2];
    std::atomic<uint64_t> ns_drain[2];
    std::atomic<uint64_t> ns_route[2];
    bool                  report = false;

    // Written from a static destructor, so it goes straight to stderr: the ggml log callback that
    // llama.cpp installs may already be gone by then.
    ~moe_route_stats() {
        if (!report) {
            return;
        }
        for (int device = 0; device < 2; ++device) {
            const uint64_t n = calls[device].load();
            if (n == 0) {
                continue;
            }
            fprintf(stderr, "ggml_cuda_moe_build_routing: %s arm: %" PRIu64 " calls, %" PRIu64 " slots, "
                "routing %.3f ms (%.1f us/call), queue drain %.3f ms (%.1f us/call)\n",
                device ? "device" : "host", n, slots[device].load(),
                ns_route[device].load()/1e6, ns_route[device].load()/1e3/n,
                ns_drain[device].load()/1e6, ns_drain[device].load()/1e3/n);
        }
    }
};

static moe_route_stats g_moe_route_stats;

static const moe_route_config & moe_route_get_config() {
    static moe_route_config cfg = [] {
        moe_route_config c;
        if (const char * s = getenv("GGML_CUDA_MOE_ROUTE")) {
            if (strcmp(s, "device") == 0) {
                c.mode = MOE_ROUTE_DEVICE;
            } else if (strcmp(s, "host") == 0) {
                c.mode = MOE_ROUTE_HOST;
            } else if (strcmp(s, "auto") != 0) {
                GGML_LOG_WARN("%s: unknown GGML_CUDA_MOE_ROUTE=%s, using auto\n", __func__, s);
            }
        }
        if (const char * s = getenv("GGML_CUDA_MOE_ROUTE_MIN_WORK")) {
            c.min_work = atoll(s);
        }
        c.debug  = getenv("GGML_CUDA_MOE_ROUTE_DEBUG")  != nullptr;
        c.verify = getenv("GGML_CUDA_MOE_ROUTE_VERIFY") != nullptr;
        g_moe_route_stats.report = c.debug;
        return c;
    }();
    return cfg;
}

// One line per distinct shape, so a long run does not drown in log output.
static void moe_route_log_shape(const ggml_tensor * dst, int64_t n_slots, bool device) {
    static std::mutex mutex;
    static std::set<std::array<int64_t, 6>> seen;

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const std::array<int64_t, 6> key = {
        (int64_t) src0->type, src0->ne[0], src0->ne[1], src0->ne[2], src1->ne[2], ids->ne[0]
    };

    std::lock_guard<std::mutex> lock(mutex);
    if (!seen.insert(key).second) {
        return;
    }
    GGML_LOG_INFO("ggml_cuda_moe_build_routing: %s arm: mul_mat_id fallback, src0 %s [%" PRId64 " x %" PRId64 "] x %" PRId64 " experts, %" PRId64 " tokens, top-%" PRId64 ", %" PRId64 " slots\n",
        device ? "device" : "host", ggml_type_name(src0->type),
        src0->ne[0], src0->ne[1], src0->ne[2], src1->ne[2], ids->ne[0], n_slots);
}

void ggml_cuda_moe_build_routing(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         dst,
        int32_t *                   ids_buf_dev,
        std::vector<int32_t> &      tokens_per_expert,
        cudaStream_t                stream) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int64_t ne02          = src0->ne[2];
    const int64_t ne11          = src1->ne[1];
    const int64_t ne12          = src1->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_slots       = ne12 * n_expert_used;

    const moe_route_config & cfg = moe_route_get_config();

    const bool device = cfg.mode == MOE_ROUTE_DEVICE ||
                       (cfg.mode == MOE_ROUTE_AUTO && n_slots >= cfg.min_work);

    GGML_ASSERT(tokens_per_expert.size() == (size_t) ne02);
    GGML_ASSERT(n_slots <= INT32_MAX);

    tokens_per_expert.assign(ne02, 0); // the host arm accumulates into it

    const auto t0 = std::chrono::steady_clock::now();
    if (cfg.debug) {
        // Both arms end in a synchronize that also waits for the work already queued on the stream.
        // Draining first separates that wait from the routing cost itself. This is behavior-neutral
        // apart from exposing the device arm's kernel launch latency, which the drain would
        // otherwise hide, so it is only done when the measurement is asked for.
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (device) {
        moe_route_device(ctx, ids, ne02, ne11, ne12, n_expert_used, ids_buf_dev, tokens_per_expert, stream);
    } else {
        moe_route_host(ids, ne02, ne11, ne12, n_expert_used, ids_buf_dev, tokens_per_expert, stream);
    }
    const auto t2 = std::chrono::steady_clock::now();

    if (cfg.debug) {
        const int i = device ? 1 : 0;
        g_moe_route_stats.calls   [i].fetch_add(1);
        g_moe_route_stats.slots   [i].fetch_add(n_slots);
        g_moe_route_stats.ns_drain[i].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        g_moe_route_stats.ns_route[i].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
        moe_route_log_shape(dst, n_slots, device);
    }

    if (cfg.verify && device) {
        moe_route_verify(ids, ne02, ne11, ne12, n_expert_used, ids_buf_dev, tokens_per_expert, stream);
    }
}
