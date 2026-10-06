#include "ggml-exec-trace.h"

#include "ggml-version.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <time.h>
#include <unistd.h>
#endif

#define GGML_EXEC_TRACE_STRINGIFY_(value) #value
#define GGML_EXEC_TRACE_STRINGIFY(value) GGML_EXEC_TRACE_STRINGIFY_(value)

namespace {

constexpr uint64_t DEFAULT_EVENT_CAPACITY = 1024 * 1024;
constexpr uint64_t MAX_EVENT_CAPACITY = 128 * 1024 * 1024;
constexpr uint64_t EVENT_RESERVATION_SIZE = 256;
constexpr uint32_t DEFAULT_GPU_TIMING_EVENT_CAPACITY = 4096;
constexpr uint32_t MAX_GPU_TIMING_EVENT_CAPACITY = 1024 * 1024;

#pragma pack(push, 1)
struct trace_file_header {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint32_t event_size;
    uint32_t flags;
    uint64_t start_time_ns;
    uint64_t event_count;
    uint64_t string_count;
    uint64_t string_bytes;
    uint64_t event_capacity;
    uint64_t attempted_event_count;
    uint64_t dropped_event_count;
    uint64_t buffer_growth_count;
    uint64_t serialization_duration_ns;
};

struct integrity_file_header {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint64_t graph_count;
    uint64_t node_count;
    uint64_t split_count;
    uint64_t graph_sum;
    uint64_t graph_xor;
    uint64_t assignment_sum;
    uint64_t assignment_xor;
    uint64_t split_sum;
    uint64_t split_xor;
    uint64_t copy_count;
    uint64_t copy_sum;
    uint64_t copy_xor;
    uint64_t dispatch_count;
    uint64_t dispatch_sum;
    uint64_t dispatch_xor;
    uint64_t gpu_graph_event_count;
    uint64_t gpu_graph_sum;
    uint64_t gpu_graph_xor;
    uint64_t gpu_graph_capture_count;
    uint64_t gpu_graph_update_count;
    uint64_t gpu_graph_replay_count;
    uint64_t gpu_graph_fallback_count;
    uint64_t eval_callback_count;
    uint64_t build_flags;
    uint64_t runtime_flags;
};

struct trace_string_header {
    uint32_t id;
    uint32_t size;
};
#pragma pack(pop)

static_assert(sizeof(ggml_exec_trace_event) == 80, "trace event layout changed");
static_assert(sizeof(trace_file_header) == 96, "trace header layout changed");
static_assert(sizeof(integrity_file_header) == 216, "integrity header layout changed");

struct integrity_accumulator {
    uint64_t count = 0;
    uint64_t sum = 0;
    uint64_t xor_value = 0;
};

enum integrity_category {
    INTEGRITY_GRAPH,
    INTEGRITY_ASSIGNMENT,
    INTEGRITY_SPLIT,
    INTEGRITY_COPY,
    INTEGRITY_DISPATCH,
    INTEGRITY_GPU_GRAPH,
    INTEGRITY_CATEGORY_COUNT,
};

struct integrity_thread_state {
    integrity_accumulator accumulators[INTEGRITY_CATEGORY_COUNT];
};

struct integrity_state {
    std::atomic<bool> enabled { false };
    std::string path;
    std::mutex flush_mutex;
    std::mutex thread_mutex;
    std::vector<integrity_thread_state *> threads;
    std::atomic<uint64_t> node_count { 0 };
    std::atomic<uint64_t> split_count { 0 };
    std::atomic<uint64_t> gpu_graph_state_counts[4] = {};
    std::atomic<uint64_t> eval_callback_count { 0 };
    uint64_t build_flags = 0;
    uint64_t runtime_flags = 0;
};

struct trace_state {
    std::atomic<bool> enabled { false };
    std::atomic<bool> truncated { false };
    std::atomic<uint64_t> reserved_event_count { 0 };
    std::atomic<uint64_t> dropped_event_count { 0 };
    std::atomic<uint32_t> next_execution_id { 1 };
    std::atomic<uint32_t> next_ubatch_id { 1 };
    std::atomic<uint32_t> next_graph_id { 1 };
    std::atomic<uint32_t> next_node_id { 1 };
    std::atomic<uint32_t> next_gpu_graph_id { 1 };
    std::atomic<uint32_t> next_thread_id { 1 };
    std::atomic<uint32_t> next_gpu_timing_epoch_id { 1 };
    std::atomic<uint64_t> gpu_timing_started { 0 };
    std::atomic<uint64_t> gpu_timing_released { 0 };

    std::mutex init_mutex;
    std::mutex string_mutex;
    std::mutex flush_mutex;
    std::atomic<bool> initialized { false };
    uint64_t start_time_ns = 0;
    uint64_t capacity = 0;
    std::string path;
    ggml_exec_trace_event * events = nullptr;
    std::vector<std::string> strings;
    std::unordered_map<std::string, uint32_t> string_ids;
    bool detailed = false;
    bool events_compacted = false;
    uint64_t compacted_event_count = 0;
    ggml_exec_trace_gpu_timing_mode gpu_timing_mode = GGML_EXEC_TRACE_GPU_TIMING_OFF;
    uint32_t gpu_timing_event_capacity = 0;
};

trace_state g_trace;
integrity_state g_integrity;
thread_local uint32_t t_execution_id = 0;
thread_local enum ggml_exec_trace_execution_phase t_execution_phase = GGML_EXEC_TRACE_PHASE_UNKNOWN;
thread_local uint32_t t_ubatch_id = 0;
thread_local uint32_t t_thread_id = 0;
thread_local integrity_thread_state * t_integrity = nullptr;
thread_local uint64_t t_event_next = 0;
thread_local uint64_t t_event_end = 0;
thread_local bool t_event_exhausted = false;

struct trace_backend_context {
    uint32_t graph_id = 0;
    uint32_t split_id = 0;
    uint32_t backend_id = 0;
    uint32_t device_id = 0;
    const uint32_t * node_ids = nullptr;
    uint32_t n_nodes = 0;
    uint32_t node_index = 0;
    uint64_t dispatch_timestamp_ns = 0;
};

thread_local trace_backend_context t_backend;
thread_local ggml_exec_trace_dispatch_record * t_dispatch_capture_records = nullptr;
thread_local uint32_t t_dispatch_capture_capacity = 0;
thread_local uint32_t * t_dispatch_capture_count = nullptr;

const char * const DISPATCH_NAMES[GGML_EXEC_TRACE_DISPATCH_COUNT] = {
    "unknown",
    "GPU_CUBLAS",
    "GPU_HIPBLAS",
    "GPU_MMVF",
    "GPU_MMVQ",
    "GPU_MMQ",
    "GPU_MMF",
    "GPU_FWHT",
    "GPU_MUL_MAT_ID_FALLBACK",
    "CPU_LLAMAFILE",
    "CPU_IQP",
    "CPU_VEC_DOT",
    "CPU_REPACK_GEMV",
    "CPU_REPACK_GEMM",
    "CPU_REPACK_GEMM_GEMV",
    "CPU_FWHT",
    "CPU_IQP_VEC_DOT",
    "CPU_IQK",
    "GPU_MOE_INT4",
};

const char * const DISPATCH_VARIANT_NAMES[GGML_EXEC_TRACE_DISPATCH_VARIANT_COUNT] = {
    "",
    "default",
    "bad_padding_or_type",
    "transposed",
    "fused",
    "sorted_fallback",
    "per_expert",
    "unpacked",
};

uint32_t g_dispatch_name_ids[GGML_EXEC_TRACE_DISPATCH_COUNT] = {};
uint32_t g_dispatch_variant_name_ids[GGML_EXEC_TRACE_DISPATCH_VARIANT_COUNT] = {};

ggml_exec_trace_event make_dispatch_event(
        uint32_t node_index,
        uint32_t op,
        uint32_t implementation_id,
        uint32_t variant_id,
        uint32_t config_0,
        uint32_t config_1) {
    ggml_exec_trace_event event = {};
    event.kind = GGML_EXEC_TRACE_EVENT_DISPATCH;
    event.timestamp_ns = t_backend.dispatch_timestamp_ns;
    event.flags = op;
    event.graph_id = t_backend.graph_id;
    event.node_id = t_backend.node_ids[node_index];
    event.split_id = t_backend.split_id;
    event.backend_id = t_backend.backend_id;
    event.device_id = t_backend.device_id;
    event.args[0] = implementation_id;
    event.args[1] = implementation_id < GGML_EXEC_TRACE_DISPATCH_COUNT ? g_dispatch_name_ids[implementation_id] : 0;
    event.args[2] = ((uint64_t) (variant_id < GGML_EXEC_TRACE_DISPATCH_VARIANT_COUNT ?
            g_dispatch_variant_name_ids[variant_id] : 0) << 32) | variant_id;
    event.args[3] = ((uint64_t) config_1 << 32) | config_0;
    return event;
}

uint64_t parse_capacity(const char * value) {
    if (!value || !value[0]) {
        return DEFAULT_EVENT_CAPACITY;
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0 || parsed > MAX_EVENT_CAPACITY) {
        std::fprintf(stderr, "ggml_exec_trace: invalid GGML_EXEC_TRACE_MAX_EVENTS '%s'; using %llu\n",
                value, (unsigned long long) DEFAULT_EVENT_CAPACITY);
        return DEFAULT_EVENT_CAPACITY;
    }
    return (uint64_t) parsed;
}

ggml_exec_trace_gpu_timing_mode parse_gpu_timing_mode(const char * value) {
    if (!value || !value[0] || std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0) {
        return GGML_EXEC_TRACE_GPU_TIMING_OFF;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "submit") == 0) {
        return GGML_EXEC_TRACE_GPU_TIMING_SUBMIT;
    }
    if (std::strcmp(value, "op") == 0) {
        return GGML_EXEC_TRACE_GPU_TIMING_OP;
    }
    std::fprintf(stderr, "ggml_exec_trace: invalid GGML_EXEC_TRACE_GPU_TIMING '%s'; device timing is disabled\n", value);
    return GGML_EXEC_TRACE_GPU_TIMING_OFF;
}

uint32_t parse_gpu_timing_event_capacity(const char * value) {
    if (!value || !value[0]) {
        return DEFAULT_GPU_TIMING_EVENT_CAPACITY;
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 2 || parsed > MAX_GPU_TIMING_EVENT_CAPACITY) {
        std::fprintf(stderr, "ggml_exec_trace: invalid GGML_EXEC_TRACE_GPU_TIMING_EVENTS '%s'; using %u\n",
                value, DEFAULT_GPU_TIMING_EVENT_CAPACITY);
        return DEFAULT_GPU_TIMING_EVENT_CAPACITY;
    }
    // Each span uses one start event and one end event.
    return (uint32_t) parsed & ~1u;
}

const char * gpu_timing_mode_name(ggml_exec_trace_gpu_timing_mode mode) {
    switch (mode) {
        case GGML_EXEC_TRACE_GPU_TIMING_SUBMIT: return "submit";
        case GGML_EXEC_TRACE_GPU_TIMING_OP:     return "op";
        default:                                return "off";
    }
}

uint32_t intern_locked(const char * value) {
    if (!value || !value[0]) {
        return 0;
    }

    auto it = g_trace.string_ids.find(value);
    if (it != g_trace.string_ids.end()) {
        return it->second;
    }

    const uint32_t id = (uint32_t) g_trace.strings.size() + 1;
    g_trace.strings.emplace_back(value);
    g_trace.string_ids.emplace(g_trace.strings.back(), id);
    return id;
}

std::string cpu_model() {
#if defined(__linux__)
    FILE * file = std::fopen("/proc/cpuinfo", "rb");
    if (!file) {
        return "unknown";
    }

    char line[512];
    std::string result = "unknown";
    while (std::fgets(line, sizeof(line), file)) {
        const char * prefix = "model name";
        if (std::strncmp(line, prefix, std::strlen(prefix)) != 0) {
            continue;
        }
        const char * value = std::strchr(line, ':');
        if (value) {
            value++;
            while (*value == ' ' || *value == '\t') {
                value++;
            }
            result = value;
            while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                result.pop_back();
            }
        }
        break;
    }
    std::fclose(file);
    return result;
#else
    return "unknown";
#endif
}

std::string cpu_affinity() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return "unknown";
    }

    std::string result;
    int range_start = -1;
    int previous = -1;
    auto append_range = [&]() {
        if (range_start < 0) {
            return;
        }
        if (!result.empty()) {
            result += ',';
        }
        result += std::to_string(range_start);
        if (previous != range_start) {
            result += '-' + std::to_string(previous);
        }
    };

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) {
            continue;
        }
        if (range_start < 0) {
            range_start = previous = cpu;
        } else if (cpu == previous + 1) {
            previous = cpu;
        } else {
            append_range();
            range_start = previous = cpu;
        }
    }
    append_range();
    return result.empty() ? "unknown" : result;
#else
    return "unknown";
#endif
}

void record_clock_sync() {
#if defined(__linux__)
    struct timespec raw_before;
    struct timespec monotonic;
    struct timespec raw_after;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &raw_before) == 0 &&
            clock_gettime(CLOCK_MONOTONIC, &monotonic) == 0 &&
            clock_gettime(CLOCK_MONOTONIC_RAW, &raw_after) == 0) {
        ggml_exec_trace_event event = {};
        event.timestamp_ns = (uint64_t) raw_before.tv_sec * 1000000000ULL + (uint64_t) raw_before.tv_nsec;
        event.args[0] = (uint64_t) monotonic.tv_sec * 1000000000ULL + (uint64_t) monotonic.tv_nsec;
        event.args[1] = (uint64_t) raw_after.tv_sec * 1000000000ULL + (uint64_t) raw_after.tv_nsec;
        event.kind = GGML_EXEC_TRACE_EVENT_CLOCK_SYNC;
        event.flags = GGML_EXEC_TRACE_CLOCK_MONOTONIC;
        ggml_exec_trace_record(event);
    }
#endif
}

void flush_at_exit() {
    ggml_exec_trace_flush();
}

bool host_is_little_endian() {
    const uint16_t value = 1;
    return *(const uint8_t *) &value == 1;
}

bool write_u32_le(FILE * file, uint32_t value) {
    uint8_t bytes[4];
    for (uint32_t i = 0; i < 4; ++i) {
        bytes[i] = (uint8_t) (value >> (i * 8));
    }
    return std::fwrite(bytes, sizeof(bytes), 1, file) == 1;
}

bool write_u64_le(FILE * file, uint64_t value) {
    uint8_t bytes[8];
    for (uint32_t i = 0; i < 8; ++i) {
        bytes[i] = (uint8_t) (value >> (i * 8));
    }
    return std::fwrite(bytes, sizeof(bytes), 1, file) == 1;
}

bool write_header(FILE * file, const trace_file_header & header) {
    if (host_is_little_endian()) {
        return std::fwrite(&header, sizeof(header), 1, file) == 1;
    }
    return std::fwrite(header.magic, sizeof(header.magic), 1, file) == 1 &&
            write_u32_le(file, header.version) &&
            write_u32_le(file, header.header_size) &&
            write_u32_le(file, header.event_size) &&
            write_u32_le(file, header.flags) &&
            write_u64_le(file, header.start_time_ns) &&
            write_u64_le(file, header.event_count) &&
            write_u64_le(file, header.string_count) &&
            write_u64_le(file, header.string_bytes) &&
            write_u64_le(file, header.event_capacity) &&
            write_u64_le(file, header.attempted_event_count) &&
            write_u64_le(file, header.dropped_event_count) &&
            write_u64_le(file, header.buffer_growth_count) &&
            write_u64_le(file, header.serialization_duration_ns);
}

uint64_t hash_values(std::initializer_list<uint64_t> values) {
    uint64_t hash = ggml_exec_trace_hash_init();
    for (uint64_t value : values) {
        hash = ggml_exec_trace_hash_u64(hash, value);
    }
    return hash;
}

void integrity_add(enum integrity_category category, uint64_t value) {
    if (!t_integrity) {
        integrity_thread_state * state = new (std::nothrow) integrity_thread_state;
        if (!state) {
            g_integrity.enabled.store(false, std::memory_order_release);
            std::fprintf(stderr, "ggml_exec_trace: failed to allocate a per-thread integrity summary\n");
            return;
        }
        try {
            std::lock_guard<std::mutex> lock(g_integrity.thread_mutex);
            g_integrity.threads.push_back(state);
        } catch (...) {
            delete state;
            g_integrity.enabled.store(false, std::memory_order_release);
            std::fprintf(stderr, "ggml_exec_trace: failed to register a per-thread integrity summary\n");
            return;
        }
        t_integrity = state;
    }
    integrity_accumulator & accumulator = t_integrity->accumulators[category];
    accumulator.count++;
    accumulator.sum += value;
    accumulator.xor_value ^= value;
}

integrity_accumulator integrity_total(enum integrity_category category) {
    integrity_accumulator total;
    std::lock_guard<std::mutex> lock(g_integrity.thread_mutex);
    for (const integrity_thread_state * state : g_integrity.threads) {
        const integrity_accumulator & accumulator = state->accumulators[category];
        total.count += accumulator.count;
        total.sum += accumulator.sum;
        total.xor_value ^= accumulator.xor_value;
    }
    return total;
}

uint64_t build_flags() {
    uint64_t flags = 0;
#if !defined(NDEBUG)
    flags |= GGML_EXEC_TRACE_INTEGRITY_BUILD_ASSERTS;
#endif
#if !defined(_MSC_VER) && !defined(__OPTIMIZE__)
    flags |= GGML_EXEC_TRACE_INTEGRITY_BUILD_NO_OPT;
#endif
#if defined(__SANITIZE_ADDRESS__)
    flags |= GGML_EXEC_TRACE_INTEGRITY_BUILD_ASAN;
#endif
#if defined(__SANITIZE_THREAD__)
    flags |= GGML_EXEC_TRACE_INTEGRITY_BUILD_TSAN;
#endif
#if defined(__SANITIZE_UNDEFINED__)
    flags |= GGML_EXEC_TRACE_INTEGRITY_BUILD_UBSAN;
#endif
    return flags;
}

uint64_t runtime_flags() {
    uint64_t flags = 0;
    flags |= std::getenv("GGML_CUDA_DISABLE_GRAPHS") ? GGML_EXEC_TRACE_INTEGRITY_RUNTIME_CUDA_GRAPHS_DISABLED : 0;
    flags |= std::getenv("GGML_HIP_DISABLE_GRAPHS") ? GGML_EXEC_TRACE_INTEGRITY_RUNTIME_HIP_GRAPHS_DISABLED : 0;
    flags |= std::getenv("GGML_CUDA_OP_TRACE") ? GGML_EXEC_TRACE_INTEGRITY_RUNTIME_CUDA_OP_TRACE : 0;
    flags |= std::getenv("GGML_HIP_OP_TRACE") ? GGML_EXEC_TRACE_INTEGRITY_RUNTIME_HIP_OP_TRACE : 0;
    const char * detail = std::getenv("GGML_EXEC_TRACE_DETAIL");
    flags |= detail && std::strcmp(detail, "full") == 0 ? GGML_EXEC_TRACE_INTEGRITY_RUNTIME_FULL_DETAIL : 0;
    return flags;
}

bool write_integrity_header(FILE * file, const integrity_file_header & header) {
    if (host_is_little_endian()) {
        return std::fwrite(&header, sizeof(header), 1, file) == 1;
    }
    if (std::fwrite(header.magic, sizeof(header.magic), 1, file) != 1 ||
            !write_u32_le(file, header.version) || !write_u32_le(file, header.header_size)) {
        return false;
    }
    const uint64_t * values = &header.graph_count;
    constexpr size_t value_count = (sizeof(header) - 16) / sizeof(uint64_t);
    for (size_t i = 0; i < value_count; ++i) {
        if (!write_u64_le(file, values[i])) {
            return false;
        }
    }
    return true;
}

bool flush_integrity() {
    if (!g_integrity.enabled.load(std::memory_order_acquire) || g_integrity.path.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_integrity.flush_mutex);
    const integrity_accumulator graph = integrity_total(INTEGRITY_GRAPH);
    const integrity_accumulator assignment = integrity_total(INTEGRITY_ASSIGNMENT);
    const integrity_accumulator split = integrity_total(INTEGRITY_SPLIT);
    const integrity_accumulator copy = integrity_total(INTEGRITY_COPY);
    const integrity_accumulator dispatch = integrity_total(INTEGRITY_DISPATCH);
    const integrity_accumulator gpu_graph = integrity_total(INTEGRITY_GPU_GRAPH);
    integrity_file_header header = {};
    std::memcpy(header.magic, "GEXINT", 6);
    header.version = GGML_EXEC_TRACE_INTEGRITY_SCHEMA_VERSION;
    header.header_size = sizeof(header);
    header.graph_count = graph.count;
    header.node_count = g_integrity.node_count.load(std::memory_order_relaxed);
    header.split_count = g_integrity.split_count.load(std::memory_order_relaxed);
    header.graph_sum = graph.sum;
    header.graph_xor = graph.xor_value;
    header.assignment_sum = assignment.sum;
    header.assignment_xor = assignment.xor_value;
    header.split_sum = split.sum;
    header.split_xor = split.xor_value;
    header.copy_count = copy.count;
    header.copy_sum = copy.sum;
    header.copy_xor = copy.xor_value;
    header.dispatch_count = dispatch.count;
    header.dispatch_sum = dispatch.sum;
    header.dispatch_xor = dispatch.xor_value;
    header.gpu_graph_event_count = gpu_graph.count;
    header.gpu_graph_sum = gpu_graph.sum;
    header.gpu_graph_xor = gpu_graph.xor_value;
    for (uint32_t i = 0; i < 4; ++i) {
        (&header.gpu_graph_capture_count)[i] = g_integrity.gpu_graph_state_counts[i].load(std::memory_order_relaxed);
    }
    header.eval_callback_count = g_integrity.eval_callback_count.load(std::memory_order_relaxed);
    header.build_flags = g_integrity.build_flags;
    header.runtime_flags = g_integrity.runtime_flags;

    FILE * file = std::fopen(g_integrity.path.c_str(), "wb");
    bool ok = file && write_integrity_header(file, header);
    if (file) {
        ok = std::fclose(file) == 0 && ok;
        file = nullptr;
    }
    if (!ok) {
        std::fprintf(stderr, "ggml_exec_trace: failed to write integrity summary '%s'\n", g_integrity.path.c_str());
    }
    return ok;
}

bool write_event(FILE * file, const ggml_exec_trace_event & event) {
    if (!write_u64_le(file, event.timestamp_ns)) {
        return false;
    }
    for (uint32_t i = 0; i < 4; ++i) {
        if (!write_u64_le(file, event.args[i])) {
            return false;
        }
    }
    return write_u32_le(file, event.execution_id) &&
            write_u32_le(file, event.ubatch_id) &&
            write_u32_le(file, event.graph_id) &&
            write_u32_le(file, event.node_id) &&
            write_u32_le(file, event.kind) &&
            write_u32_le(file, event.flags) &&
            write_u32_le(file, event.split_id) &&
            write_u32_le(file, event.backend_id) &&
            write_u32_le(file, event.device_id) &&
            write_u32_le(file, event.thread_id);
}

uint32_t trace_thread_id() {
    if (t_thread_id == 0) {
        t_thread_id = g_trace.next_thread_id.fetch_add(1, std::memory_order_relaxed);
    }
    return t_thread_id;
}

bool reserve_event_slot(uint64_t & index) {
    if (t_event_exhausted) {
        g_trace.dropped_event_count.fetch_add(1, std::memory_order_relaxed);
        g_trace.truncated.store(true, std::memory_order_relaxed);
        return false;
    }
    // Reserve in blocks so CPU worker threads do not contend on one event counter for every dispatch.
    if (t_event_next == t_event_end) {
        t_event_next = g_trace.reserved_event_count.fetch_add(EVENT_RESERVATION_SIZE, std::memory_order_relaxed);
        t_event_end = std::min(t_event_next + EVENT_RESERVATION_SIZE, g_trace.capacity);
        if (t_event_next >= t_event_end) {
            t_event_exhausted = true;
            g_trace.dropped_event_count.fetch_add(1, std::memory_order_relaxed);
            g_trace.truncated.store(true, std::memory_order_relaxed);
            return false;
        }
    }
    index = t_event_next++;
    return true;
}

} // namespace

void ggml_exec_trace_init(void) {
    if (g_trace.initialized.load(std::memory_order_acquire)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_trace.init_mutex);
    if (g_trace.initialized.load(std::memory_order_relaxed)) {
        return;
    }
    g_trace.initialized.store(true, std::memory_order_release);

    const ggml_exec_trace_gpu_timing_mode gpu_timing_mode = parse_gpu_timing_mode(std::getenv("GGML_EXEC_TRACE_GPU_TIMING"));

    const char * integrity_path = std::getenv("GGML_EXEC_TRACE_INTEGRITY");
    if (integrity_path && integrity_path[0]) {
        try {
            g_integrity.path = integrity_path;
            g_integrity.build_flags = build_flags();
            g_integrity.runtime_flags = runtime_flags();
            g_integrity.enabled.store(true, std::memory_order_release);
        } catch (...) {
            std::fprintf(stderr, "ggml_exec_trace: failed to initialize integrity reporting\n");
        }
    }

    const char * path = std::getenv("GGML_EXEC_TRACE");
    if (!path || !path[0]) {
        if (ggml_exec_trace_integrity_enabled()) {
            std::atexit(flush_at_exit);
        }
        return;
    }

    try {
        g_trace.path = path;
        const char * detail = std::getenv("GGML_EXEC_TRACE_DETAIL");
        g_trace.detailed = detail && std::strcmp(detail, "full") == 0;
    } catch (...) {
        std::fprintf(stderr, "ggml_exec_trace: failed to store the output path; tracing is disabled\n");
        return;
    }

    g_trace.capacity = parse_capacity(std::getenv("GGML_EXEC_TRACE_MAX_EVENTS"));
    g_trace.events = new (std::nothrow) ggml_exec_trace_event[g_trace.capacity]();
    if (!g_trace.events) {
        std::fprintf(stderr, "ggml_exec_trace: failed to allocate %llu trace events; tracing is disabled\n",
                (unsigned long long) g_trace.capacity);
        return;
    }

    FILE * file = std::fopen(g_trace.path.c_str(), "wb");
    if (!file) {
        std::fprintf(stderr, "ggml_exec_trace: cannot open '%s': %s; tracing is disabled\n",
                g_trace.path.c_str(), std::strerror(errno));
        delete[] g_trace.events;
        g_trace.events = nullptr;
        return;
    }
    std::fclose(file);

    g_trace.start_time_ns = ggml_exec_trace_now_ns();
    // Device timing needs the raw trace; an integrity-only witness never records timing events.
    g_trace.gpu_timing_mode = gpu_timing_mode;
    if (gpu_timing_mode != GGML_EXEC_TRACE_GPU_TIMING_OFF) {
        g_trace.gpu_timing_event_capacity = parse_gpu_timing_event_capacity(std::getenv("GGML_EXEC_TRACE_GPU_TIMING_EVENTS"));
        g_integrity.runtime_flags |= GGML_EXEC_TRACE_INTEGRITY_RUNTIME_GPU_TIMING;
    }
    g_trace.enabled.store(true, std::memory_order_release);
    std::atexit(flush_at_exit);

    record_clock_sync();

    for (uint32_t i = 0; i < GGML_EXEC_TRACE_DISPATCH_COUNT; ++i) {
        g_dispatch_name_ids[i] = ggml_exec_trace_intern(DISPATCH_NAMES[i]);
    }
    for (uint32_t i = 1; i < GGML_EXEC_TRACE_DISPATCH_VARIANT_COUNT; ++i) {
        g_dispatch_variant_name_ids[i] = ggml_exec_trace_intern(DISPATCH_VARIANT_NAMES[i]);
    }

    ggml_exec_trace_metadata("schema", "ggml-exec-trace-v" GGML_EXEC_TRACE_STRINGIFY(GGML_EXEC_TRACE_SCHEMA_VERSION));
#if defined(__linux__)
    ggml_exec_trace_metadata("clock", "CLOCK_MONOTONIC_RAW");
#else
    ggml_exec_trace_metadata("clock", "steady_clock");
#endif
    ggml_exec_trace_metadata("build_commit", GGML_COMMIT);
    try {
        const std::string model = cpu_model();
        ggml_exec_trace_metadata("host_cpu", model.c_str());
        const std::string affinity = cpu_affinity();
        ggml_exec_trace_metadata("cpu_affinity", affinity.c_str());
    } catch (...) {
        ggml_exec_trace_disable("failed to collect host metadata");
    }
    ggml_exec_trace_metadata("trace_enabled", "true");
    ggml_exec_trace_metadata("trace_detail", g_trace.detailed ? "full" : "production");
    ggml_exec_trace_metadata("gpu_timing", gpu_timing_mode_name(g_trace.gpu_timing_mode));
    if (g_trace.gpu_timing_mode != GGML_EXEC_TRACE_GPU_TIMING_OFF) {
        try {
            const std::string capacity = std::to_string(g_trace.gpu_timing_event_capacity);
            ggml_exec_trace_metadata("gpu_timing_event_capacity", capacity.c_str());
        } catch (...) {
            ggml_exec_trace_disable("failed to record GPU timing metadata");
        }
    }
}

bool ggml_exec_trace_enabled(void) {
    return g_trace.enabled.load(std::memory_order_acquire);
}

bool ggml_exec_trace_integrity_enabled(void) {
    return g_integrity.enabled.load(std::memory_order_acquire);
}

bool ggml_exec_trace_observing(void) {
    return ggml_exec_trace_enabled() || ggml_exec_trace_integrity_enabled();
}

bool ggml_exec_trace_detailed(void) {
    return ggml_exec_trace_enabled() && g_trace.detailed;
}

void ggml_exec_trace_disable(const char * reason) {
    if (g_trace.enabled.exchange(false, std::memory_order_acq_rel)) {
        g_trace.truncated.store(true, std::memory_order_relaxed);
        std::fprintf(stderr, "ggml_exec_trace: %s; tracing is disabled\n", reason ? reason : "trace failure");
    }
}

uint64_t ggml_exec_trace_now_ns(void) {
#if defined(__linux__)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0) {
        return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
    }
#endif
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint32_t ggml_exec_trace_intern(const char * value) {
    if (!ggml_exec_trace_enabled() || !value || !value[0]) {
        return 0;
    }
    try {
        std::lock_guard<std::mutex> lock(g_trace.string_mutex);
        return intern_locked(value);
    } catch (...) {
        ggml_exec_trace_disable("failed to intern trace metadata");
        return 0;
    }
}

void ggml_exec_trace_metadata(const char * key, const char * value) {
    if (!ggml_exec_trace_enabled()) {
        return;
    }
    ggml_exec_trace_event event = {};
    event.kind = GGML_EXEC_TRACE_EVENT_RUN_METADATA;
    event.args[0] = ggml_exec_trace_intern(key);
    event.args[1] = ggml_exec_trace_intern(value);
    ggml_exec_trace_record(event);
}

void ggml_exec_trace_record(struct ggml_exec_trace_event event) {
    uint64_t index = 0;
    if (!ggml_exec_trace_enabled() || !reserve_event_slot(index)) {
        return;
    }

    if (event.timestamp_ns == 0) {
        event.timestamp_ns = ggml_exec_trace_now_ns();
    }
    if (event.execution_id == 0) {
        event.execution_id = t_execution_id;
    }
    if (event.ubatch_id == 0) {
        event.ubatch_id = t_ubatch_id;
    }
    if (event.thread_id == 0) {
        event.thread_id = trace_thread_id();
    }
    g_trace.events[index] = event;
}

uint32_t ggml_exec_trace_execution_begin(
        enum ggml_exec_trace_execution_kind kind,
        uint32_t n_tokens,
        uint32_t n_batch,
        uint32_t n_ubatch,
        uint32_t n_threads,
        uint32_t n_threads_batch) {
    if (!ggml_exec_trace_enabled()) {
        return 0;
    }

    const uint32_t id = g_trace.next_execution_id.fetch_add(1, std::memory_order_relaxed);
    t_execution_id = id;

    ggml_exec_trace_event event = {};
    event.kind = GGML_EXEC_TRACE_EVENT_EXECUTION_BEGIN;
    event.execution_id = id;
    event.flags = (uint32_t) kind | ((uint32_t) t_execution_phase << 8);
    event.args[0] = n_tokens;
    event.args[1] = n_batch;
    event.args[2] = n_ubatch;
    event.args[3] = ((uint64_t) n_threads_batch << 32) | n_threads;
    ggml_exec_trace_record(event);
    return id;
}

enum ggml_exec_trace_execution_phase ggml_exec_trace_set_execution_phase(
        enum ggml_exec_trace_execution_phase phase) {
    const enum ggml_exec_trace_execution_phase previous = t_execution_phase;
    t_execution_phase = phase;
    return previous;
}

void ggml_exec_trace_execution_end(uint32_t execution_id, int32_t status) {
    if (execution_id == 0) {
        return;
    }
    ggml_exec_trace_event event = {};
    event.kind = GGML_EXEC_TRACE_EVENT_EXECUTION_END;
    event.execution_id = execution_id;
    event.args[0] = (uint32_t) status;
    ggml_exec_trace_record(event);
    if (t_execution_id == execution_id) {
        t_execution_id = 0;
    }
}

uint32_t ggml_exec_trace_ubatch_begin(
        uint32_t n_tokens,
        uint32_t n_sequences,
        uint32_t n_unique_sequences,
        uint32_t n_outputs) {
    if (!ggml_exec_trace_enabled()) {
        return 0;
    }
    const uint32_t id = g_trace.next_ubatch_id.fetch_add(1, std::memory_order_relaxed);
    t_ubatch_id = id;

    ggml_exec_trace_event event = {};
    event.kind = GGML_EXEC_TRACE_EVENT_UBATCH_BEGIN;
    event.ubatch_id = id;
    event.args[0] = n_tokens;
    event.args[1] = n_sequences;
    event.args[2] = n_unique_sequences;
    event.args[3] = n_outputs;
    ggml_exec_trace_record(event);
    return id;
}

void ggml_exec_trace_ubatch_end(uint32_t ubatch_id, int32_t status) {
    if (ubatch_id == 0) {
        return;
    }
    ggml_exec_trace_event event = {};
    event.kind = GGML_EXEC_TRACE_EVENT_UBATCH_END;
    event.ubatch_id = ubatch_id;
    event.args[0] = (uint32_t) status;
    ggml_exec_trace_record(event);
    if (t_ubatch_id == ubatch_id) {
        t_ubatch_id = 0;
    }
}

uint32_t ggml_exec_trace_next_graph_id(void) {
    return ggml_exec_trace_observing() ? g_trace.next_graph_id.fetch_add(1, std::memory_order_relaxed) : 0;
}

uint32_t ggml_exec_trace_next_node_id(void) {
    return ggml_exec_trace_enabled() ? g_trace.next_node_id.fetch_add(1, std::memory_order_relaxed) : 0;
}

uint32_t ggml_exec_trace_next_gpu_graph_id(void) {
    return ggml_exec_trace_observing() ? g_trace.next_gpu_graph_id.fetch_add(1, std::memory_order_relaxed) : 0;
}

void ggml_exec_trace_backend_begin(
        uint32_t graph_id,
        uint32_t split_id,
        uint32_t backend_id,
        uint32_t device_id,
        const uint32_t * node_ids,
        uint32_t n_nodes) {
    if (!ggml_exec_trace_observing()) {
        return;
    }
    t_backend.graph_id = graph_id;
    t_backend.split_id = split_id;
    t_backend.backend_id = backend_id;
    t_backend.device_id = device_id;
    t_backend.node_ids = node_ids;
    t_backend.n_nodes = n_nodes;
    t_backend.node_index = 0;
    t_backend.dispatch_timestamp_ns = ggml_exec_trace_enabled() ? ggml_exec_trace_now_ns() : 0;
}

void ggml_exec_trace_backend_end(void) {
    ggml_exec_trace_dispatch_capture_end();
    t_backend = {};
}

void ggml_exec_trace_set_node(uint32_t node_index) {
    t_backend.node_index = node_index;
}

thread_local std::vector<trace_backend_context> t_nested_saved;
thread_local std::vector<std::vector<uint32_t>> t_nested_ids;

void ggml_exec_trace_nested_begin(const uint32_t * map, uint32_t n_nodes) {
    const size_t depth = t_nested_saved.size();
    t_nested_saved.push_back(t_backend);
    if (!t_backend.node_ids) {
        return;
    }
    if (t_nested_ids.size() <= depth) {
        t_nested_ids.resize(depth + 1);
    }
    std::vector<uint32_t> & ids = t_nested_ids[depth];
    ids.resize(n_nodes);
    for (uint32_t i = 0; i < n_nodes; ++i) {
        // node ID 0 is never assigned, so it marks work without a traced node
        ids[i] = map[i] < t_backend.n_nodes ? t_backend.node_ids[map[i]] : 0;
    }
    t_backend.node_ids   = ids.data();
    t_backend.n_nodes    = n_nodes;
    t_backend.node_index = 0;
}

void ggml_exec_trace_nested_end(void) {
    GGML_ASSERT(!t_nested_saved.empty());
    t_backend = t_nested_saved.back();
    t_nested_saved.pop_back();
}

uint64_t ggml_exec_trace_cpu_node_begin(void) {
    return ggml_exec_trace_enabled() && g_trace.gpu_timing_mode == GGML_EXEC_TRACE_GPU_TIMING_OP &&
            t_backend.node_ids && t_backend.node_index < t_backend.n_nodes ? ggml_exec_trace_now_ns() : 0;
}

void ggml_exec_trace_cpu_node_end(uint64_t start_ns, uint32_t node_index, uint32_t node_count) {
    if (start_ns == 0 || !t_backend.node_ids || node_index >= t_backend.n_nodes) {
        return;
    }
    ggml_exec_trace_event event = {};
    event.timestamp_ns = start_ns;
    event.kind = GGML_EXEC_TRACE_EVENT_CPU_NODE_TIMING;
    event.graph_id = t_backend.graph_id;
    event.node_id = t_backend.node_ids[node_index];
    event.split_id = t_backend.split_id;
    event.backend_id = t_backend.backend_id;
    event.device_id = t_backend.device_id;
    event.args[0] = ggml_exec_trace_now_ns();
    event.args[1] = node_count;
    ggml_exec_trace_record(event);
}

void ggml_exec_trace_dispatch(
        enum ggml_op op,
        enum ggml_exec_trace_dispatch_impl implementation,
        enum ggml_exec_trace_dispatch_variant variant,
        uint32_t config_0,
        uint32_t config_1) {
    if (!ggml_exec_trace_observing() || t_backend.node_index >= t_backend.n_nodes) {
        return;
    }

    const uint32_t implementation_id = (uint32_t) implementation;
    const uint32_t variant_id = (uint32_t) variant;

    if (ggml_exec_trace_integrity_enabled()) {
        integrity_add(INTEGRITY_DISPATCH, hash_values({
                t_backend.split_id, t_backend.backend_id, t_backend.device_id, t_backend.node_index,
                (uint32_t) op, implementation_id, variant_id, config_0, config_1 }));
    }

    if (t_dispatch_capture_records && t_dispatch_capture_count && *t_dispatch_capture_count < t_dispatch_capture_capacity) {
        ggml_exec_trace_dispatch_record & record = t_dispatch_capture_records[(*t_dispatch_capture_count)++];
        record.node_index = t_backend.node_index;
        record.op = (uint32_t) op;
        record.implementation = implementation_id;
        record.variant = variant_id;
        record.config_0 = config_0;
        record.config_1 = config_1;
    }

    if (ggml_exec_trace_enabled() && t_backend.node_ids) {
        ggml_exec_trace_record(make_dispatch_event(
                t_backend.node_index, (uint32_t) op, implementation_id, variant_id, config_0, config_1));
    }
}

void ggml_exec_trace_dispatch_capture_begin(
        struct ggml_exec_trace_dispatch_record * records,
        uint32_t capacity,
        uint32_t * count) {
    t_dispatch_capture_records = records;
    t_dispatch_capture_capacity = capacity;
    t_dispatch_capture_count = count;
    if (count) {
        *count = 0;
    }
}

void ggml_exec_trace_dispatch_capture_end(void) {
    t_dispatch_capture_records = nullptr;
    t_dispatch_capture_capacity = 0;
    t_dispatch_capture_count = nullptr;
}

void ggml_exec_trace_dispatch_replay(
        const struct ggml_exec_trace_dispatch_record * records,
        uint32_t count) {
    if (!ggml_exec_trace_observing() || !records) {
        return;
    }

    for (uint32_t i = 0; i < count; ++i) {
        const ggml_exec_trace_dispatch_record & record = records[i];
        if (record.node_index >= t_backend.n_nodes) {
            continue;
        }
        if (ggml_exec_trace_integrity_enabled()) {
            integrity_add(INTEGRITY_DISPATCH, hash_values({
                    t_backend.split_id, t_backend.backend_id, t_backend.device_id, record.node_index,
                    record.op, record.implementation, record.variant, record.config_0, record.config_1 }));
        }
    }
    if (!ggml_exec_trace_enabled() || !t_backend.node_ids) {
        return;
    }

    uint32_t valid_count = 0;
    for (uint32_t i = 0; i < count; ++i) {
        valid_count += records[i].node_index < t_backend.n_nodes;
    }
    if (valid_count == 0) {
        return;
    }

    // A replay repeats decisions made during capture. Reserve them together and use one host timestamp so
    // replay tracing does not add a clock read and atomic reservation for every node in a captured graph.
    const uint64_t first_index = g_trace.reserved_event_count.fetch_add(valid_count, std::memory_order_relaxed);
    if (first_index >= g_trace.capacity) {
        g_trace.dropped_event_count.fetch_add(valid_count, std::memory_order_relaxed);
        g_trace.truncated.store(true, std::memory_order_relaxed);
        return;
    }
    const uint32_t writable_count = (uint32_t) std::min<uint64_t>(valid_count, g_trace.capacity - first_index);
    if (writable_count != valid_count) {
        g_trace.dropped_event_count.fetch_add(valid_count - writable_count, std::memory_order_relaxed);
        g_trace.truncated.store(true, std::memory_order_relaxed);
    }

    const uint64_t timestamp_ns = ggml_exec_trace_now_ns();
    if (t_thread_id == 0) {
        t_thread_id = g_trace.next_thread_id.fetch_add(1, std::memory_order_relaxed);
    }

    uint32_t output_index = 0;
    for (uint32_t i = 0; i < count && output_index < writable_count; ++i) {
        const ggml_exec_trace_dispatch_record & record = records[i];
        if (record.node_index >= t_backend.n_nodes) {
            continue;
        }
        ggml_exec_trace_event event = make_dispatch_event(
                record.node_index, record.op, record.implementation, record.variant, record.config_0, record.config_1);
        event.timestamp_ns = timestamp_ns;
        event.execution_id = t_execution_id;
        event.ubatch_id = t_ubatch_id;
        event.thread_id = t_thread_id;
        g_trace.events[first_index + output_index++] = event;
    }
}

void ggml_exec_trace_gpu_graph(
        enum ggml_exec_trace_gpu_graph_state state,
        uint32_t gpu_graph_id,
        uint32_t node_count,
        enum ggml_exec_trace_gpu_graph_reason reason) {
    if (ggml_exec_trace_integrity_enabled()) {
        integrity_add(INTEGRITY_GPU_GRAPH, hash_values({
                (uint32_t) state, gpu_graph_id, node_count, (uint32_t) reason,
                t_backend.split_id, t_backend.backend_id, t_backend.device_id }));
        if (state >= GGML_EXEC_TRACE_GPU_GRAPH_CAPTURE && state <= GGML_EXEC_TRACE_GPU_GRAPH_FALLBACK) {
            g_integrity.gpu_graph_state_counts[state - GGML_EXEC_TRACE_GPU_GRAPH_CAPTURE].fetch_add(
                    1, std::memory_order_relaxed);
        }
    }
    if (!ggml_exec_trace_enabled()) {
        return;
    }
    ggml_exec_trace_event event = {};
    event.kind = GGML_EXEC_TRACE_EVENT_GPU_GRAPH;
    event.flags = (uint32_t) state;
    event.graph_id = t_backend.graph_id;
    event.split_id = t_backend.split_id;
    event.backend_id = t_backend.backend_id;
    event.device_id = t_backend.device_id;
    event.args[0] = gpu_graph_id;
    event.args[1] = node_count;
    event.args[2] = (uint32_t) reason;
    ggml_exec_trace_record(event);
}

enum ggml_exec_trace_gpu_timing_mode ggml_exec_trace_gpu_timing_get_mode(void) {
    return ggml_exec_trace_enabled() ? g_trace.gpu_timing_mode : GGML_EXEC_TRACE_GPU_TIMING_OFF;
}

uint32_t ggml_exec_trace_gpu_timing_event_capacity(void) {
    return g_trace.gpu_timing_event_capacity;
}

uint32_t ggml_exec_trace_next_gpu_timing_epoch_id(void) {
    return g_trace.next_gpu_timing_epoch_id.fetch_add(1, std::memory_order_relaxed);
}

void ggml_exec_trace_gpu_timing_prepare(
        struct ggml_exec_trace_event * event,
        enum ggml_exec_trace_gpu_timing_scope scope,
        uint32_t node_index,
        uint32_t detail) {
    *event = {};
    event->timestamp_ns = ggml_exec_trace_now_ns();
    event->args[0] = event->timestamp_ns;
    event->args[3] = detail;
    event->execution_id = t_execution_id;
    event->ubatch_id = t_ubatch_id;
    event->graph_id = t_backend.graph_id;
    event->split_id = t_backend.split_id;
    event->backend_id = t_backend.backend_id;
    event->device_id = t_backend.device_id;
    if (scope == GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP && t_backend.node_ids && node_index < t_backend.n_nodes) {
        event->node_id = t_backend.node_ids[node_index];
    }
    event->kind = GGML_EXEC_TRACE_EVENT_GPU_TIMING;
    event->flags = (uint32_t) scope | ((uint32_t) GGML_EXEC_TRACE_TIMING_DEVICE_EVENT << 16);
    event->thread_id = trace_thread_id();
    g_trace.gpu_timing_started.fetch_add(1, std::memory_order_relaxed);
}

void ggml_exec_trace_gpu_timing_report(
        struct ggml_exec_trace_event * event,
        enum ggml_exec_trace_gpu_timing_status status,
        uint32_t epoch_id,
        uint64_t start_offset_ns,
        uint64_t duration_ns) {
    event->flags = (event->flags & ~0xff00u) | (((uint32_t) status & 0xff) << 8);
    event->args[1] = start_offset_ns;
    event->args[2] = duration_ns;
    event->args[3] = ((uint64_t) epoch_id << 32) | (event->args[3] & 0xffffffff);
    g_trace.gpu_timing_released.fetch_add(1, std::memory_order_relaxed);

    uint64_t index = 0;
    if (ggml_exec_trace_enabled() && reserve_event_slot(index)) {
        g_trace.events[index] = *event;
    }
}

void ggml_exec_trace_gpu_timing_epoch(
        const struct ggml_exec_trace_event * anchor_span,
        uint32_t epoch_id,
        uint64_t complete_ns,
        uint32_t span_count,
        enum ggml_exec_trace_gpu_timing_completion completion) {
    ggml_exec_trace_event event = *anchor_span;
    event.kind = GGML_EXEC_TRACE_EVENT_GPU_TIMING_EPOCH;
    event.flags = (uint32_t) GGML_EXEC_TRACE_TIMING_DEVICE_EVENT << 16;
    event.node_id = 0;
    event.args[0] = complete_ns;
    event.args[1] = epoch_id;
    event.args[2] = span_count;
    event.args[3] = (uint32_t) completion;

    uint64_t index = 0;
    if (ggml_exec_trace_enabled() && reserve_event_slot(index)) {
        g_trace.events[index] = event;
    }
}

void ggml_exec_trace_integrity_graph(
        uint64_t ops_shapes_hash,
        uint64_t backend_hash,
        uint64_t split_hash,
        uint64_t node_count,
        uint64_t split_count,
        uint64_t cross_backend_edges) {
    if (!ggml_exec_trace_integrity_enabled()) {
        return;
    }
    integrity_add(INTEGRITY_GRAPH, hash_values({ ops_shapes_hash, node_count }));
    integrity_add(INTEGRITY_ASSIGNMENT, hash_values({ backend_hash, node_count }));
    integrity_add(INTEGRITY_SPLIT, hash_values({ split_hash, split_count, cross_backend_edges }));
    g_integrity.node_count.fetch_add(node_count, std::memory_order_relaxed);
    g_integrity.split_count.fetch_add(split_count, std::memory_order_relaxed);
}

void ggml_exec_trace_integrity_copy(
        enum ggml_exec_trace_copy_path path,
        uint32_t split_id,
        uint32_t src_backend_id,
        uint32_t dst_backend_id,
        uint64_t bytes) {
    if (!ggml_exec_trace_integrity_enabled()) {
        return;
    }
    integrity_add(INTEGRITY_COPY, hash_values({
            (uint32_t) path, split_id, src_backend_id, dst_backend_id, bytes }));
}

void ggml_exec_trace_integrity_eval_callback(void) {
    if (ggml_exec_trace_integrity_enabled()) {
        g_integrity.eval_callback_count.fetch_add(1, std::memory_order_relaxed);
    }
}

uint64_t ggml_exec_trace_hash_init(void) {
    return 14695981039346656037ULL;
}

uint64_t ggml_exec_trace_hash_bytes(uint64_t hash, const void * data, uint64_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    for (uint64_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t ggml_exec_trace_hash_u64(uint64_t hash, uint64_t value) {
    uint8_t bytes[8];
    for (uint32_t i = 0; i < 8; ++i) {
        bytes[i] = (uint8_t) (value >> (i * 8));
    }
    return ggml_exec_trace_hash_bytes(hash, bytes, sizeof(bytes));
}

bool ggml_exec_trace_flush(void) {
    flush_integrity();
    if (!g_trace.events || g_trace.path.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> flush_lock(g_trace.flush_mutex);
    if (!g_trace.events_compacted) {
        record_clock_sync();
        if (g_trace.gpu_timing_mode != GGML_EXEC_TRACE_GPU_TIMING_OFF) {
            // A started span that was never released belongs to a backend that was not freed before the trace was written.
            try {
                const std::string started = std::to_string(g_trace.gpu_timing_started.load(std::memory_order_relaxed));
                const std::string released = std::to_string(g_trace.gpu_timing_released.load(std::memory_order_relaxed));
                ggml_exec_trace_metadata("gpu_timing_spans_started", started.c_str());
                ggml_exec_trace_metadata("gpu_timing_spans_released", released.c_str());
            } catch (...) {
                std::fprintf(stderr, "ggml_exec_trace: failed to record GPU timing counters\n");
            }
        }
    }
    std::lock_guard<std::mutex> string_lock(g_trace.string_mutex);

    // Per-thread reservations can leave unused zeroed slots. Remove them after measured execution.
    if (!g_trace.events_compacted) {
        const uint64_t reserved_event_count = std::min(
                g_trace.reserved_event_count.load(std::memory_order_acquire), g_trace.capacity);
        uint64_t output_index = 0;
        for (uint64_t input_index = 0; input_index < reserved_event_count; ++input_index) {
            if (g_trace.events[input_index].kind != 0) {
                if (output_index != input_index) {
                    g_trace.events[output_index] = g_trace.events[input_index];
                }
                output_index++;
            }
        }
        g_trace.compacted_event_count = output_index;
        g_trace.events_compacted = true;
    }
    const uint64_t event_count = g_trace.compacted_event_count;
    uint64_t string_bytes = 0;
    for (const std::string & value : g_trace.strings) {
        string_bytes += sizeof(trace_string_header) + value.size();
    }

    trace_file_header header = {};
    std::memcpy(header.magic, "GEXECTR", 7);
    header.version = GGML_EXEC_TRACE_SCHEMA_VERSION;
    header.header_size = sizeof(header);
    header.event_size = sizeof(ggml_exec_trace_event);
    header.flags = g_trace.truncated.load(std::memory_order_relaxed) ? GGML_EXEC_TRACE_FILE_TRUNCATED : 0;
    header.start_time_ns = g_trace.start_time_ns;
    const uint64_t dropped_event_count = g_trace.dropped_event_count.load(std::memory_order_acquire);
    const uint64_t attempted_event_count = event_count + dropped_event_count;
    header.event_count = event_count;
    header.string_count = g_trace.strings.size();
    header.string_bytes = string_bytes;
    header.event_capacity = g_trace.capacity;
    header.attempted_event_count = attempted_event_count;
    header.dropped_event_count = dropped_event_count;
    header.buffer_growth_count = 0;

    const uint64_t serialization_start_ns = ggml_exec_trace_now_ns();
    FILE * file = std::fopen(g_trace.path.c_str(), "wb");
    if (!file) {
        std::fprintf(stderr, "ggml_exec_trace: cannot write '%s': %s\n", g_trace.path.c_str(), std::strerror(errno));
        return false;
    }

    bool ok = write_header(file, header);
    if (ok && event_count > 0) {
        if (host_is_little_endian()) {
            ok = std::fwrite(g_trace.events, sizeof(ggml_exec_trace_event), event_count, file) == event_count;
        } else {
            for (uint64_t i = 0; ok && i < event_count; ++i) {
                ok = write_event(file, g_trace.events[i]);
            }
        }
    }
    for (uint32_t i = 0; ok && i < g_trace.strings.size(); ++i) {
        const std::string & value = g_trace.strings[i];
        const trace_string_header string_header = { i + 1, (uint32_t) value.size() };
        ok = host_is_little_endian() ?
                std::fwrite(&string_header, sizeof(string_header), 1, file) == 1 :
                write_u32_le(file, string_header.id) && write_u32_le(file, string_header.size);
        ok = ok && (value.empty() || std::fwrite(value.data(), 1, value.size(), file) == value.size());
    }
    header.serialization_duration_ns = ggml_exec_trace_now_ns() - serialization_start_ns;
    if (ok && std::fseek(file, 0, SEEK_SET) == 0) {
        ok = write_header(file, header);
    } else {
        ok = false;
    }
    ok = std::fclose(file) == 0 && ok;

    if (!ok) {
        std::fprintf(stderr, "ggml_exec_trace: failed while writing '%s'\n", g_trace.path.c_str());
    }
    return ok;
}
