#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "repack.h"
#include "traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"

#include <cctype>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#    if defined(__linux__)
#        include <sys/mman.h>
#    endif
#endif

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

ggml_backend_buffer_type_t ggml_backend_cpu_numa_buffer_type(void);

static const char * ggml_backend_cpu_numa_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "CPU_NUMA";
}

static void ggml_backend_cpu_numa_buffer_free(ggml_backend_buffer_t buffer) {
#if defined(__linux__)
    if (munmap(buffer->context, buffer->size) != 0) {
        GGML_LOG_ERROR("%s: munmap failed: %s\n", __func__, strerror(errno));
    }
#else
    ggml_aligned_free(buffer->context, buffer->size);
#endif
}

static void * ggml_backend_cpu_numa_buffer_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void ggml_backend_cpu_numa_buffer_memset(
        ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
        uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memset((char *) tensor->data + offset, value, size);
}

static void ggml_backend_cpu_numa_buffer_set(
        ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void ggml_backend_cpu_numa_buffer_get(
        ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static bool ggml_backend_cpu_numa_buffer_cpy(
        ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (!ggml_backend_buffer_is_host(src->buffer)) {
        return false;
    }
    memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}

static void ggml_backend_cpu_numa_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_numa_buffer_iface = {
    /* .free_buffer   = */ ggml_backend_cpu_numa_buffer_free,
    /* .get_base      = */ ggml_backend_cpu_numa_buffer_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_cpu_numa_buffer_memset,
    /* .set_tensor    = */ ggml_backend_cpu_numa_buffer_set,
    /* .get_tensor    = */ ggml_backend_cpu_numa_buffer_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_cpu_numa_buffer_cpy,
    /* .clear         = */ ggml_backend_cpu_numa_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_cpu_numa_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
#if defined(__linux__)
    void * data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
        GGML_LOG_ERROR("%s: mmap failed for %zu bytes: %s\n", __func__, size, strerror(errno));
        return nullptr;
    }
    // Keep the default 4 KiB pages: an expert stores its nodes' row ranges back to back.
    // A 2 MiB page often spans two of those ranges, and first touch would put it on one node only.
    return ggml_backend_buffer_init(buft, ggml_backend_cpu_numa_buffer_iface, data, size);
#else
    void * data = ggml_aligned_malloc(size);
    return data ? ggml_backend_buffer_init(buft, ggml_backend_cpu_numa_buffer_iface, data, size) : nullptr;
#endif
}

static size_t ggml_backend_cpu_numa_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
}

static bool ggml_backend_cpu_numa_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

ggml_backend_buffer_type_t ggml_backend_cpu_numa_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name       = */ ggml_backend_cpu_numa_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_cpu_numa_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_cpu_numa_buffer_type_get_alignment,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ nullptr,
            /* .is_host        = */ ggml_backend_cpu_numa_buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ nullptr,
    };
    return &buft;
}

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        if (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();

    GGML_UNUSED(device);
}

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    for (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra == buft) {
            return true;
        }
    }
    return false;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool                use_ref;  // use reference implementation

    // MUL_MAT_ID calibration is shape-dependent, not only type-dependent: the routed
    // tile geometry (rows, cols, top-k, tokens) changes the achievable throughput, and MoE
    // layers alternate between projection shapes. A single per-type slot therefore thrashes
    // and re-runs the benchmark for every layer. Key by the full shape so one calibration is
    // reused across layers and ubatches.
    struct offload_mul_mat_id_shape {
        int64_t rows;
        int64_t cols;
        int64_t top_k;
        int64_t tokens;

        bool operator<(const offload_mul_mat_id_shape & other) const {
            if (rows   != other.rows)   return rows   < other.rows;
            if (cols   != other.cols)   return cols   < other.cols;
            if (top_k  != other.top_k)  return top_k  < other.top_k;
            return tokens < other.tokens;
        }
    };

    std::mutex          offload_cost_mutex;
    int                 offload_cost_threads = 0;
    double              offload_memory_bytes_per_second = 0.0;
    double              offload_dot_operations_per_second[GGML_TYPE_COUNT] = {};
    std::map<offload_mul_mat_id_shape, double> offload_mul_mat_id_operations_per_second;
};

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    return ggml_graph_compute(cgraph, &cplan);
}

static double ggml_backend_cpu_measure_memory_bandwidth(int n_threads) {
    const size_t size = 128u*1024u*1024u;
    std::vector<uint8_t> src(size, 0x5a);
    std::vector<uint8_t> dst(size, 0);
    n_threads = std::max(1, std::min<int>(n_threads, 64));

    double best = 0.0;
    for (int repetition = 0; repetition < 2; ++repetition) {
        std::atomic<int> ready(0);
        std::atomic<bool> start(false);
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        for (int thread = 0; thread < n_threads; ++thread) {
            workers.emplace_back([&, thread] {
                const size_t begin = size*(size_t) thread/n_threads;
                const size_t end   = size*(size_t) (thread + 1)/n_threads;
                ready.fetch_add(1, std::memory_order_release);
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                std::memcpy(dst.data() + begin, src.data() + begin, end - begin);
            });
        }
        while (ready.load(std::memory_order_acquire) != n_threads) {
            std::this_thread::yield();
        }
        const auto before = std::chrono::steady_clock::now();
        start.store(true, std::memory_order_release);
        for (std::thread & worker : workers) {
            worker.join();
        }
        const auto after = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(after - before).count();
        best = std::max(best, size/seconds);
        std::swap(src, dst);
    }
    return best;
}

static double ggml_backend_cpu_measure_dot_throughput(enum ggml_type type) {
    const ggml_type_traits_cpu * weight_traits = ggml_get_type_traits_cpu(type);
    if (!weight_traits->from_float || !weight_traits->vec_dot) {
        return 0.0;
    }

    const int64_t n = 4096;
    const ggml_type activation_type = weight_traits->vec_dot_type;
    const ggml_type_traits_cpu * activation_traits = ggml_get_type_traits_cpu(activation_type);
    if (!activation_traits->from_float) {
        return 0.0;
    }

    std::vector<float> source(n, 0.125f);
    std::vector<uint8_t> weight(ggml_row_size(type, n));
    std::vector<uint8_t> activation(ggml_row_size(activation_type, n));
    weight_traits->from_float(source.data(), weight.data(), n);
    activation_traits->from_float(source.data(), activation.data(), n);

    float result = 0.0f;
    weight_traits->vec_dot(n, &result, 0, weight.data(), 0, activation.data(), 0, 1);

    uint64_t iterations = 1;
    double seconds = 0.0;
    do {
        iterations *= 2;
        const auto before = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < iterations; ++i) {
            weight_traits->vec_dot(n, &result, 0, weight.data(), 0, activation.data(), 0, 1);
        }
        const auto after = std::chrono::steady_clock::now();
        seconds = std::chrono::duration<double>(after - before).count();
    } while (seconds < 0.01 && iterations < (1u << 20));

    return 2.0*n*iterations/seconds;
}

// Measure the actual routed-expert kernel using the already loaded weight tensor.
// This preserves any CPU-specific packed layout and does not retain model data.
// delphi: the MUL_MAT_ID calibration runs the whole op three times, so measuring every token count again
// re-ran it for each new prompt length (seconds per request for large CPU experts). Measure and cache at
// the token count rounded to the nearest power of two instead; throughput follows tokens per expert, which
// that resolves well enough.
static int64_t ggml_backend_cpu_offload_token_bucket(int64_t n_tokens) {
    if (n_tokens <= 1) {
        return 1;
    }
    int64_t lower = 1;
    while (2*lower <= n_tokens) {
        lower *= 2;
    }
    return n_tokens - lower < 2*lower - n_tokens ? lower : 2*lower;
}

static double ggml_backend_cpu_measure_mul_mat_id_throughput(
        ggml_backend_t backend, const struct ggml_backend_cpu_context * cpu_ctx,
        const struct ggml_tensor * op, const struct ggml_tensor * weight, int64_t n_tokens) {
    if (op->op != GGML_OP_MUL_MAT_ID || !op->src[1] || !op->src[2] || weight->ne[2] <= 0) {
        return 0.0;
    }

    const int64_t top_k = op->src[2]->ne[0];
    if (n_tokens <= 0 || top_k <= 0) {
        return 0.0;
    }

    // Calibration runs during graph reserve, before lazy weights are resident.
    // Reading weight data here would fault on NULL and crash. Skip the direct
    // measurement and fall back to the vector-dot estimate.
    if (!weight->data) {
        return 0.0;
    }

    const struct ggml_init_params params = {
        /*.mem_size   =*/ 3*ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return 0.0;
    }

    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, op->src[1]->type,
            weight->ne[0], top_k, n_tokens);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, n_tokens);
    struct ggml_tensor * result = ggml_mul_mat_id(ctx, const_cast<struct ggml_tensor *>(weight), b, ids);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, result);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return 0.0;
    }

    std::vector<uint8_t> zeros(std::max(ggml_nbytes(b), ggml_nbytes(result)), 0);
    std::vector<int32_t> selected((size_t) top_k*n_tokens);
    for (size_t i = 0; i < selected.size(); ++i) {
        selected[i] = i % weight->ne[2];
    }
    ggml_backend_tensor_set(b, zeros.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(ids, selected.data(), 0, ggml_nbytes(ids));

    struct ggml_cplan plan = ggml_graph_plan(graph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    plan.abort_callback = cpu_ctx->abort_callback;
    plan.abort_callback_data = cpu_ctx->abort_callback_data;
    plan.use_ref = cpu_ctx->use_ref;
    if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return 0.0;
    }

    const uint64_t operations = 2*(uint64_t) weight->ne[0]*weight->ne[1]*top_k*n_tokens;
    double longest_seconds = 0.0;
    for (int repetition = 0; repetition < 2; ++repetition) {
        const auto before = std::chrono::steady_clock::now();
        const enum ggml_status status = ggml_graph_compute(graph, &plan);
        const auto after = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return 0.0;
        }
        longest_seconds = std::max(longest_seconds, std::chrono::duration<double>(after - before).count());
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    const double throughput = longest_seconds > 0.0 ? operations/longest_seconds : 0.0;
    return throughput;
}

static bool ggml_backend_cpu_get_op_cost(
        ggml_backend_t backend, const struct ggml_tensor * op, const struct ggml_tensor * weight,
        uint64_t operations, uint64_t active_weight_bytes, struct ggml_backend_op_cost * cost) {
    if (operations == 0 || active_weight_bytes == 0) {
        return false;
    }

    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *) backend->context;
    const int n_threads = std::max(1, cpu_ctx->n_threads);
    double dot_operations_per_second = 0.0;
    double mul_mat_id_operations_per_second = 0.0;
    {
        std::lock_guard<std::mutex> lock(cpu_ctx->offload_cost_mutex);
        if (cpu_ctx->offload_cost_threads != n_threads || cpu_ctx->offload_memory_bytes_per_second <= 0.0) {
            cpu_ctx->offload_memory_bytes_per_second = ggml_backend_cpu_measure_memory_bandwidth(n_threads);
            cpu_ctx->offload_cost_threads = n_threads;
            cpu_ctx->offload_mul_mat_id_operations_per_second.clear();
            GGML_LOG_INFO("CPU offload calibration: threads = %d, memory = %.2f GB/s\n",
                    n_threads, cpu_ctx->offload_memory_bytes_per_second/1.0e9);
        }
        if (cpu_ctx->offload_dot_operations_per_second[weight->type] <= 0.0) {
            cpu_ctx->offload_dot_operations_per_second[weight->type] =
                    ggml_backend_cpu_measure_dot_throughput(weight->type);
            GGML_LOG_INFO("CPU offload calibration: type = %s, vector dot = %.2f GOPS/thread\n",
                    ggml_type_name(weight->type),
                    cpu_ctx->offload_dot_operations_per_second[weight->type]/1.0e9);
        }
        dot_operations_per_second = cpu_ctx->offload_dot_operations_per_second[weight->type];
        if (op->op == GGML_OP_MUL_MAT_ID && op->src[1] != nullptr && op->src[2] != nullptr) {
            const struct ggml_backend_cpu_context::offload_mul_mat_id_shape shape = {
                /*.rows   =*/ weight->ne[0],
                /*.cols   =*/ weight->ne[1],
                /*.top_k  =*/ op->src[2]->ne[0],
                /*.tokens =*/ ggml_backend_cpu_offload_token_bucket(op->src[1]->ne[2]),
            };
            auto it = cpu_ctx->offload_mul_mat_id_operations_per_second.find(shape);
            if (it != cpu_ctx->offload_mul_mat_id_operations_per_second.end()) {
                mul_mat_id_operations_per_second = it->second;
            } else {
                const double measured = ggml_backend_cpu_measure_mul_mat_id_throughput(backend, cpu_ctx, op, weight, shape.tokens);
                // A lazy weight that is not resident yet measures 0; do not cache that so a
                // later call with resident data can still calibrate.
                if (measured > 0.0) {
                    cpu_ctx->offload_mul_mat_id_operations_per_second.emplace(shape, measured);
                    mul_mat_id_operations_per_second = measured;
                    GGML_LOG_INFO("CPU offload calibration: type = %s, MUL_MAT_ID = %.2f GOPS\n",
                            ggml_type_name(weight->type), measured/1.0e9);
                }
            }
        }
    }

    if (dot_operations_per_second <= 0.0 && mul_mat_id_operations_per_second <= 0.0) {
        return false;
    }

    // The two measured limits make this a roofline estimate. The dot benchmark is cache-hot;
    // the memory calibration limits it when the full expert working set must come from RAM.
    const double arithmetic_intensity = (double) operations/active_weight_bytes;
    const double memory_limit = cpu_ctx->offload_memory_bytes_per_second*arithmetic_intensity;
    const double compute_limit = mul_mat_id_operations_per_second > 0.0 ?
            mul_mat_id_operations_per_second : n_threads*dot_operations_per_second;

    *cost = {};
    // MUL_MAT_ID has an operation-family calibration with its routed tile shape.
    // Do not replace that measurement with the generic memcpy roofline.
    cost->operations_per_second = mul_mat_id_operations_per_second > 0.0 ?
            compute_limit : std::min(memory_limit, compute_limit);
    cost->numa_node = -1;
    cost->calibrated = 1;
    return cost->operations_per_second > 0.0;
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->use_ref             = false;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    return "CPU";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return src1->type == GGML_TYPE_F32 || src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        case GGML_OP_GET_ROWS_BACK:
            return src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SSM_SCAN:
            return ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        default:
            return true;
    }
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_cpu_device_context ctx;
    static ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_get_rvv_vlen() > 0) {
            static std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_get_op_cost") == 0) {
        return (void *) ggml_backend_cpu_get_op_cost;
    }
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa_tensor_mode") == 0) {
        return (void *)ggml_numa_is_tensor_mode;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_load_expert_tensor") == 0) {
        return (void *)ggml_numa_load_expert_tensor;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_buffer_type") == 0) {
        return (void *)ggml_backend_cpu_numa_buffer_type;
    }
    if (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        return (void *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
