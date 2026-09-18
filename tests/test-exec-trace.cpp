#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-exec-trace.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#pragma pack(push, 1)
struct test_trace_header {
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

struct test_integrity_header {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint64_t values[25];
};
#pragma pack(pop)

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    GGML_ASSERT(_putenv_s(name, value) == 0);
#else
    GGML_ASSERT(setenv(name, value, 1) == 0);
#endif
}

static test_trace_header read_header(const char * path) {
    test_trace_header header = {};
    std::ifstream input(path, std::ios::binary);
    GGML_ASSERT(input.good());
    input.read((char *) &header, sizeof(header));
    GGML_ASSERT(input.gcount() == sizeof(header));
    GGML_ASSERT(std::memcmp(header.magic, "GEXECTR", 7) == 0);
    GGML_ASSERT(header.version == GGML_EXEC_TRACE_SCHEMA_VERSION);
    GGML_ASSERT(header.header_size == sizeof(header));
    GGML_ASSERT(header.event_size == sizeof(ggml_exec_trace_event));
    return header;
}

static std::vector<ggml_exec_trace_event> read_events(const char * path, const test_trace_header & header) {
    std::ifstream input(path, std::ios::binary);
    input.seekg(header.header_size);
    std::vector<ggml_exec_trace_event> events(header.event_count);
    input.read((char *) events.data(), events.size() * sizeof(events[0]));
    GGML_ASSERT((uint64_t) input.gcount() == events.size() * sizeof(events[0]));
    return events;
}

static std::vector<std::string> read_strings(const char * path, const test_trace_header & header) {
    std::ifstream input(path, std::ios::binary);
    input.seekg(header.header_size + header.event_count * header.event_size);
    std::vector<std::string> strings(header.string_count + 1);
    for (uint64_t i = 0; i < header.string_count; ++i) {
        uint32_t id_size[2] = {};
        input.read((char *) id_size, sizeof(id_size));
        GGML_ASSERT(input.gcount() == sizeof(id_size) && id_size[0] > 0 && id_size[0] <= header.string_count);
        strings[id_size[0]].resize(id_size[1]);
        input.read(strings[id_size[0]].data(), id_size[1]);
        GGML_ASSERT((uint32_t) input.gcount() == id_size[1]);
    }
    return strings;
}

static std::string metadata_value(
        const std::vector<ggml_exec_trace_event> & events, const std::vector<std::string> & strings, const char * key) {
    for (const ggml_exec_trace_event & event : events) {
        if (event.kind == GGML_EXEC_TRACE_EVENT_RUN_METADATA && event.args[0] < strings.size() &&
                strings[event.args[0]] == key && event.args[1] < strings.size()) {
            return strings[event.args[1]];
        }
    }
    return "";
}

static void run_tiny_graph() {
    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * experts = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 3, 2);
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 1, 2);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 2);
    ggml_set_name(experts, "trace-experts");
    ggml_set_name(input, "trace-input");
    ggml_set_name(ids, "trace-ids");
    ggml_set_input(experts);
    ggml_set_input(input);
    ggml_set_input(ids);
    ggml_tensor * result = ggml_mul_mat_id(ctx.get(), experts, input, ids);
    ggml_set_name(result, "trace-result");
    ggml_set_output(result);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(graph, result);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);
    ggml_backend_t backend_raw = backend.get();
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_raw);
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(&backend_raw, &buft, 1, 8, false, true));
    GGML_ASSERT(sched);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    float expert_data[4 * 3 * 2] = {};
    float input_data[4 * 1 * 2] = {};
    int32_t id_data[1 * 2] = { 0, 1 };
    ggml_backend_tensor_set(experts, expert_data, 0, sizeof(expert_data));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(ids, id_data, 0, sizeof(id_data));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
}

static int test_disabled() {
    ggml_exec_trace_init();
    GGML_ASSERT(!ggml_exec_trace_enabled());
    ggml_exec_trace_event event = {};
    event.kind = GGML_EXEC_TRACE_EVENT_GRAPH;
    ggml_exec_trace_record(event);
    run_tiny_graph();
    GGML_ASSERT(!ggml_exec_trace_flush());
    return 0;
}

static int test_enabled(const char * path, bool detailed) {
    std::remove(path);
    set_env("GGML_EXEC_TRACE", path);
    if (detailed) {
        set_env("GGML_EXEC_TRACE_DETAIL", "full");
    }
    ggml_exec_trace_init();
    GGML_ASSERT(ggml_exec_trace_enabled());
    GGML_ASSERT(ggml_exec_trace_detailed() == detailed);

    const uint32_t first = ggml_exec_trace_intern("same-value");
    const uint32_t second = ggml_exec_trace_intern("same-value");
    const uint32_t third = ggml_exec_trace_intern("different-value");
    GGML_ASSERT(first != 0 && first == second && first != third);

    const uint64_t hash_a = ggml_exec_trace_hash_u64(ggml_exec_trace_hash_init(), 0x12345678);
    const uint64_t hash_b = ggml_exec_trace_hash_u64(ggml_exec_trace_hash_init(), 0x12345678);
    const uint64_t hash_c = ggml_exec_trace_hash_u64(ggml_exec_trace_hash_init(), 0x12345679);
    GGML_ASSERT(hash_a == hash_b && hash_a != hash_c);

    const uint32_t execution_id = ggml_exec_trace_execution_begin(
            GGML_EXEC_TRACE_EXECUTION_DECODE, 1, 32, 16, 4, 8);
    const uint32_t ubatch_id = ggml_exec_trace_ubatch_begin(1, 1, 1, 1);

    const uint64_t copy_start = ggml_exec_trace_now_ns();
    ggml_exec_trace_event copy_event = {};
    copy_event.timestamp_ns = copy_start;
    copy_event.args[0] = copy_start + 1000;
    copy_event.args[1] = 41;
    copy_event.args[2] = 4096;
    copy_event.args[3] = ((uint64_t) 7 << 32) | 6;
    copy_event.graph_id = 37;
    copy_event.node_id = 40;
    copy_event.kind = GGML_EXEC_TRACE_EVENT_COPY_HANDLING;
    copy_event.flags = GGML_EXEC_TRACE_COPY_FALLBACK_SYNC;
    copy_event.split_id = 3;
    copy_event.backend_id = 8;
    copy_event.device_id = 9;
    ggml_exec_trace_record(copy_event);

    const uint32_t backend_node_ids[] = { 41, 42 };
    ggml_exec_trace_dispatch_record captured_dispatches[2] = {};
    uint32_t captured_dispatch_count = 0;
    ggml_exec_trace_backend_begin(37, 3, 8, 9, backend_node_ids, 2);
    ggml_exec_trace_set_node(1);
    ggml_exec_trace_dispatch_capture_begin(captured_dispatches, 2, &captured_dispatch_count);
    ggml_exec_trace_dispatch(GGML_OP_MUL_MAT_ID, GGML_EXEC_TRACE_DISPATCH_GPU_MMVQ,
            GGML_EXEC_TRACE_DISPATCH_VARIANT_FUSED, 1201, 64);
    ggml_exec_trace_dispatch_capture_end();
    GGML_ASSERT(captured_dispatch_count == 1 && captured_dispatches[0].node_index == 1);

    const uint32_t gpu_graph_id = ggml_exec_trace_next_gpu_graph_id();
    const uint32_t other_gpu_graph_id = ggml_exec_trace_next_gpu_graph_id();
    GGML_ASSERT(gpu_graph_id != 0 && other_gpu_graph_id != gpu_graph_id);
    ggml_exec_trace_gpu_graph(GGML_EXEC_TRACE_GPU_GRAPH_CAPTURE, gpu_graph_id, 2,
            GGML_EXEC_TRACE_GPU_GRAPH_REASON_NONE);
    ggml_exec_trace_gpu_graph(GGML_EXEC_TRACE_GPU_GRAPH_REPLAY, gpu_graph_id, 2,
            GGML_EXEC_TRACE_GPU_GRAPH_REASON_NONE);
    ggml_exec_trace_gpu_graph(GGML_EXEC_TRACE_GPU_GRAPH_REPLAY, gpu_graph_id, 2,
            GGML_EXEC_TRACE_GPU_GRAPH_REASON_NONE);
    ggml_exec_trace_backend_end();

    ggml_exec_trace_backend_begin(38, 4, 10, 11, backend_node_ids, 2);
    ggml_exec_trace_dispatch_replay(captured_dispatches, captured_dispatch_count);
    ggml_exec_trace_gpu_graph(GGML_EXEC_TRACE_GPU_GRAPH_FALLBACK, 0, 2,
            GGML_EXEC_TRACE_GPU_GRAPH_REASON_DISABLED);
    ggml_exec_trace_backend_end();

    std::vector<std::thread> writers;
    for (uint32_t thread_id = 0; thread_id < 4; ++thread_id) {
        writers.emplace_back([thread_id]() {
            for (uint32_t i = 0; i < 32; ++i) {
                ggml_exec_trace_event event = {};
                event.kind = GGML_EXEC_TRACE_EVENT_NODE_DIMS;
                event.args[0] = thread_id;
                event.args[1] = i;
                ggml_exec_trace_record(event);
            }
        });
    }
    for (std::thread & writer : writers) {
        writer.join();
    }

    run_tiny_graph();
    ggml_exec_trace_ubatch_end(ubatch_id, GGML_STATUS_SUCCESS);
    ggml_exec_trace_execution_end(execution_id, 0);
    GGML_ASSERT(ggml_exec_trace_flush());

    const test_trace_header header = read_header(path);
    GGML_ASSERT((header.flags & GGML_EXEC_TRACE_FILE_TRUNCATED) == 0);
    GGML_ASSERT(header.event_count > 128);
    GGML_ASSERT(header.string_count >= 3);
    GGML_ASSERT(header.string_bytes > 0);
    GGML_ASSERT(header.event_capacity >= header.event_count);
    GGML_ASSERT(header.attempted_event_count == header.event_count);
    GGML_ASSERT(header.dropped_event_count == 0);
    GGML_ASSERT(header.buffer_growth_count == 0);

    const std::vector<ggml_exec_trace_event> events = read_events(path, header);
    const std::vector<std::string> strings = read_strings(path, header);
    GGML_ASSERT(metadata_value(events, strings, "schema") ==
            "ggml-exec-trace-v" + std::to_string(GGML_EXEC_TRACE_SCHEMA_VERSION));
    auto has_kind = [&](uint32_t kind) {
        return std::any_of(events.begin(), events.end(), [kind](const ggml_exec_trace_event & event) {
            return event.kind == kind;
        });
    };
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_EXECUTION_BEGIN));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_UBATCH_BEGIN));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_GRAPH));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_NODE));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_NODE_SOURCE));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_NODE_MATMUL));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_SCHED_SPLIT));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_FINGERPRINT));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_CPU_COMPUTE));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_BACKEND_WAIT));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_COPY_HANDLING));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_DISPATCH));
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_GPU_GRAPH));
#if defined(__linux__)
    GGML_ASSERT(has_kind(GGML_EXEC_TRACE_EVENT_CLOCK_SYNC));
#endif

    uint32_t cpu_compute_count = 0;
    uint32_t backend_wait_count = 0;
    bool found_copy_event = false;
    uint32_t manual_dispatch_count = 0;
    uint32_t gpu_graph_replay_count = 0;
    uint32_t clock_sync_count = 0;
    uint64_t dispatch_name_id = 0;
    for (const ggml_exec_trace_event & event : events) {
        if (event.kind == GGML_EXEC_TRACE_EVENT_NODE_SOURCE) {
            GGML_ASSERT(event.args[1] != GGML_EXEC_TRACE_ID_NONE);
            GGML_ASSERT(event.flags != GGML_EXEC_TRACE_ID_NONE);
        } else if (event.kind == GGML_EXEC_TRACE_EVENT_NODE_MATMUL) {
            GGML_ASSERT(event.flags == GGML_OP_MUL_MAT_ID);
            GGML_ASSERT(event.args[0] == 3);
            GGML_ASSERT(event.args[1] == 2);
            GGML_ASSERT(event.args[2] == 4);
            GGML_ASSERT(event.args[3] == 1);
        } else if (event.kind == GGML_EXEC_TRACE_EVENT_CLOCK_SYNC) {
            GGML_ASSERT(event.flags == GGML_EXEC_TRACE_CLOCK_MONOTONIC);
            GGML_ASSERT(event.args[0] > 0);
            GGML_ASSERT(event.args[1] >= event.timestamp_ns);
            clock_sync_count++;
        }
        if (event.kind >= GGML_EXEC_TRACE_EVENT_BACKEND_SUBMIT && event.kind <= GGML_EXEC_TRACE_EVENT_COPY_BODY) {
            GGML_ASSERT(event.args[0] >= event.timestamp_ns);
            GGML_ASSERT(event.execution_id == execution_id);
            GGML_ASSERT(event.ubatch_id == ubatch_id);
        }
        if (event.kind == GGML_EXEC_TRACE_EVENT_CPU_COMPUTE) {
            cpu_compute_count++;
        } else if (event.kind == GGML_EXEC_TRACE_EVENT_BACKEND_WAIT) {
            backend_wait_count++;
        } else if (event.kind == GGML_EXEC_TRACE_EVENT_COPY_HANDLING && event.graph_id == 37) {
            GGML_ASSERT(event.flags == GGML_EXEC_TRACE_COPY_FALLBACK_SYNC);
            GGML_ASSERT(event.node_id == 40 && event.args[1] == 41);
            GGML_ASSERT(event.args[2] == 4096);
            GGML_ASSERT((event.args[3] & 0xffffffff) == 6);
            GGML_ASSERT((event.args[3] >> 32) == 7);
            GGML_ASSERT(event.split_id == 3);
            GGML_ASSERT(event.backend_id == 8 && event.device_id == 9);
            found_copy_event = true;
        } else if (event.kind == GGML_EXEC_TRACE_EVENT_DISPATCH &&
                (event.graph_id == 37 || event.graph_id == 38)) {
            GGML_ASSERT(event.node_id == 42);
            GGML_ASSERT(event.flags == GGML_OP_MUL_MAT_ID);
            GGML_ASSERT(event.args[0] == GGML_EXEC_TRACE_DISPATCH_GPU_MMVQ);
            GGML_ASSERT((event.args[2] & 0xffffffff) == GGML_EXEC_TRACE_DISPATCH_VARIANT_FUSED);
            GGML_ASSERT((event.args[3] & 0xffffffff) == 1201 && (event.args[3] >> 32) == 64);
            GGML_ASSERT(event.backend_id == (event.graph_id == 37 ? 8 : 10));
            GGML_ASSERT(event.device_id == (event.graph_id == 37 ? 9 : 11));
            if (dispatch_name_id == 0) {
                dispatch_name_id = event.args[1];
            }
            GGML_ASSERT(event.args[1] == dispatch_name_id);
            manual_dispatch_count++;
        } else if (event.kind == GGML_EXEC_TRACE_EVENT_GPU_GRAPH &&
                event.args[0] == gpu_graph_id && event.flags == GGML_EXEC_TRACE_GPU_GRAPH_REPLAY) {
            GGML_ASSERT(event.backend_id == 8 && event.device_id == 9 && event.args[1] == 2);
            gpu_graph_replay_count++;
        }
    }
#if defined(__linux__)
    GGML_ASSERT(clock_sync_count == 2);
#endif
    GGML_ASSERT(cpu_compute_count == 2);
    GGML_ASSERT(backend_wait_count >= 2);
    GGML_ASSERT(found_copy_event);
    GGML_ASSERT(manual_dispatch_count == 2 && dispatch_name_id != 0);
    GGML_ASSERT(gpu_graph_replay_count == 2);

    std::vector<ggml_exec_trace_event> graphs;
    std::vector<ggml_exec_trace_event> fingerprints;
    for (const ggml_exec_trace_event & event : events) {
        if (event.kind == GGML_EXEC_TRACE_EVENT_GRAPH) {
            graphs.push_back(event);
        } else if (event.kind == GGML_EXEC_TRACE_EVENT_FINGERPRINT) {
            fingerprints.push_back(event);
        }
    }
    GGML_ASSERT(graphs.size() == 2 && graphs[0].flags == 0 && graphs[1].flags == 1);
    GGML_ASSERT(graphs[1].args[0] == graphs[0].graph_id);
    GGML_ASSERT(fingerprints.size() == 2);
    GGML_ASSERT(std::equal(std::begin(fingerprints[0].args), std::end(fingerprints[0].args), std::begin(fingerprints[1].args)));
    return 0;
}

static int test_truncation(const char * path) {
    std::remove(path);
    set_env("GGML_EXEC_TRACE", path);
    set_env("GGML_EXEC_TRACE_MAX_EVENTS", "4");
    ggml_exec_trace_init();
    GGML_ASSERT(ggml_exec_trace_enabled());
    for (int i = 0; i < 16; ++i) {
        ggml_exec_trace_event event = {};
        event.kind = GGML_EXEC_TRACE_EVENT_GRAPH;
        ggml_exec_trace_record(event);
    }
    GGML_ASSERT(ggml_exec_trace_flush());
    const test_trace_header header = read_header(path);
    GGML_ASSERT(header.event_count == 4);
    GGML_ASSERT(header.flags & GGML_EXEC_TRACE_FILE_TRUNCATED);
    GGML_ASSERT(header.attempted_event_count > header.event_count);
    GGML_ASSERT(header.dropped_event_count == header.attempted_event_count - header.event_count);
    return 0;
}

static int test_integrity(const char * path) {
    std::remove(path);
    set_env("GGML_EXEC_TRACE_INTEGRITY", path);
    ggml_exec_trace_init();
    GGML_ASSERT(!ggml_exec_trace_enabled());
    GGML_ASSERT(ggml_exec_trace_integrity_enabled());
    GGML_ASSERT(ggml_exec_trace_observing());

    run_tiny_graph();
    GGML_ASSERT(!ggml_exec_trace_flush());

    test_integrity_header header = {};
    std::ifstream input(path, std::ios::binary);
    GGML_ASSERT(input.good());
    input.read((char *) &header, sizeof(header));
    GGML_ASSERT(input.gcount() == sizeof(header));
    GGML_ASSERT(std::memcmp(header.magic, "GEXINT", 6) == 0);
    GGML_ASSERT(header.version == GGML_EXEC_TRACE_INTEGRITY_SCHEMA_VERSION);
    GGML_ASSERT(header.header_size == sizeof(header));
    GGML_ASSERT(header.values[0] == 2); // graph_count
    GGML_ASSERT(header.values[1] > 0);  // node_count
    GGML_ASSERT(header.values[2] > 0);  // split_count
    GGML_ASSERT(header.values[12] > 0); // dispatch_count
    return 0;
}

static uint32_t gpu_timing_scope(const ggml_exec_trace_event & event) {
    return event.flags & 0xff;
}

static uint32_t gpu_timing_status(const ggml_exec_trace_event & event) {
    return (event.flags >> 8) & 0xff;
}

static uint32_t gpu_timing_provenance(const ggml_exec_trace_event & event) {
    return (event.flags >> 16) & 0xff;
}

static int test_gpu_timing_api(const char * path) {
    std::remove(path);
    set_env("GGML_EXEC_TRACE", path);
    set_env("GGML_EXEC_TRACE_GPU_TIMING", "op");
    set_env("GGML_EXEC_TRACE_GPU_TIMING_EVENTS", "7");
    ggml_exec_trace_init();
    GGML_ASSERT(ggml_exec_trace_gpu_timing_get_mode() == GGML_EXEC_TRACE_GPU_TIMING_OP);
    GGML_ASSERT(ggml_exec_trace_gpu_timing_event_capacity() == 6);

    const uint32_t execution_id = ggml_exec_trace_execution_begin(GGML_EXEC_TRACE_EXECUTION_DECODE, 1, 1, 1, 1, 1);
    const uint32_t node_ids[] = { 51, 52 };
    ggml_exec_trace_event submit;
    ggml_exec_trace_event op;
    ggml_exec_trace_event dropped;
    ggml_exec_trace_backend_begin(47, 5, 12, 13, node_ids, 2);
    ggml_exec_trace_gpu_timing_prepare(&submit, GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_SUBMIT, 0, 2);
    ggml_exec_trace_gpu_timing_prepare(&op, GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP, 1, 1);
    ggml_exec_trace_gpu_timing_prepare(&dropped, GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP, 0, 1);
    ggml_exec_trace_backend_end();
    ggml_exec_trace_execution_end(execution_id, 0);

    // Spans are reported after the execution ends; they must keep the IDs captured by prepare.
    const uint32_t epoch_id = ggml_exec_trace_next_gpu_timing_epoch_id();
    ggml_exec_trace_gpu_timing_report(&submit, GGML_EXEC_TRACE_GPU_TIMING_RESOLVED, epoch_id, 0, 5000);
    ggml_exec_trace_gpu_timing_report(&op, GGML_EXEC_TRACE_GPU_TIMING_RESOLVED, epoch_id, 1000, 2000);
    ggml_exec_trace_gpu_timing_epoch(&submit, epoch_id, submit.args[0] + 10000, 2,
            GGML_EXEC_TRACE_GPU_TIMING_COMPLETION_SYNCHRONIZE);
    ggml_exec_trace_gpu_timing_report(&dropped, GGML_EXEC_TRACE_GPU_TIMING_NOT_RECORDED, 0, 0, 0);
    GGML_ASSERT(ggml_exec_trace_flush());

    const test_trace_header header = read_header(path);
    const std::vector<ggml_exec_trace_event> events = read_events(path, header);
    const std::vector<std::string> strings = read_strings(path, header);
    GGML_ASSERT(metadata_value(events, strings, "gpu_timing") == "op");
    GGML_ASSERT(metadata_value(events, strings, "gpu_timing_event_capacity") == "6");
    GGML_ASSERT(metadata_value(events, strings, "gpu_timing_spans_started") == "3");
    GGML_ASSERT(metadata_value(events, strings, "gpu_timing_spans_released") == "3");

    uint32_t span_count = 0;
    uint32_t epoch_count = 0;
    for (const ggml_exec_trace_event & event : events) {
        if (event.kind == GGML_EXEC_TRACE_EVENT_GPU_TIMING) {
            span_count++;
            GGML_ASSERT(gpu_timing_provenance(event) == GGML_EXEC_TRACE_TIMING_DEVICE_EVENT);
            GGML_ASSERT(event.execution_id == execution_id);
            GGML_ASSERT(event.graph_id == 47 && event.split_id == 5 && event.backend_id == 12 && event.device_id == 13);
            if (gpu_timing_status(event) == GGML_EXEC_TRACE_GPU_TIMING_NOT_RECORDED) {
                GGML_ASSERT(event.node_id == 51 && (event.args[3] >> 32) == 0);
            } else if (gpu_timing_scope(event) == GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP) {
                GGML_ASSERT(event.node_id == 52);
                GGML_ASSERT(event.args[1] == 1000 && event.args[2] == 2000);
                GGML_ASSERT((event.args[3] >> 32) == epoch_id && (event.args[3] & 0xffffffff) == 1);
            } else {
                GGML_ASSERT(gpu_timing_scope(event) == GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_SUBMIT);
                GGML_ASSERT(event.node_id == 0 && event.args[2] == 5000 && (event.args[3] & 0xffffffff) == 2);
            }
        } else if (event.kind == GGML_EXEC_TRACE_EVENT_GPU_TIMING_EPOCH) {
            epoch_count++;
            GGML_ASSERT(event.timestamp_ns == submit.timestamp_ns && event.args[0] == submit.args[0] + 10000);
            GGML_ASSERT(event.args[1] == epoch_id && event.args[2] == 2);
            GGML_ASSERT(event.args[3] == GGML_EXEC_TRACE_GPU_TIMING_COMPLETION_SYNCHRONIZE);
            GGML_ASSERT(((event.flags >> 16) & 0xff) == GGML_EXEC_TRACE_TIMING_DEVICE_EVENT);
        }
    }
    GGML_ASSERT(span_count == 3 && epoch_count == 1);
    return 0;
}

static ggml_backend_dev_t find_timed_gpu() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const std::string reg_name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                (reg_name == "CUDA" || reg_name == "ROCm" || reg_name == "MUSA")) {
            return dev;
        }
    }
    return nullptr;
}

static int test_gpu_timing_backend(const char * path, bool small_pool) {
    std::remove(path);
    set_env("GGML_EXEC_TRACE", path);
    set_env("GGML_EXEC_TRACE_GPU_TIMING", "op");
    if (small_pool) {
        // One span per completion point: later spans in the same epoch must be reported as not recorded.
        set_env("GGML_EXEC_TRACE_GPU_TIMING_EVENTS", "2");
    }
    ggml_exec_trace_init();
    ggml_backend_dev_t dev = find_timed_gpu();
    if (!dev) {
        std::printf("%s: no CUDA or HIP device, skipping\n", __func__);
        return 0;
    }

    uint32_t execution_id = 0;
    std::vector<uint32_t> node_ids;
    {
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        GGML_ASSERT(backend);
        ggml_init_params params = {
            /* .mem_size   = */ 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        const int64_t n = 1 << 16;
        ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
        ggml_tensor * b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
        ggml_tensor * c = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
        ggml_tensor * result = ggml_scale(ctx.get(), ggml_add(ctx.get(), ggml_mul(ctx.get(), a, b), c), 0.5f);
        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, result);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        GGML_ASSERT(buffer);
        const std::vector<float> data(n, 1.0f);
        for (ggml_tensor * tensor : { a, b, c }) {
            ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
        }

        node_ids.resize(ggml_graph_n_nodes(graph));
        for (size_t i = 0; i < node_ids.size(); ++i) {
            node_ids[i] = 101 + (uint32_t) i;
        }
        execution_id = ggml_exec_trace_execution_begin(GGML_EXEC_TRACE_EXECUTION_DECODE, 1, 1, 1, 1, 1);
        // The first compute runs eagerly; later computes can capture and replay a GPU graph.
        for (int i = 0; i < 4; ++i) {
            ggml_exec_trace_backend_begin(91, 1, 1, 1, node_ids.data(), (uint32_t) node_ids.size());
            GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
            ggml_exec_trace_backend_end();
        }
        ggml_exec_trace_execution_end(execution_id, 0);
    }
    GGML_ASSERT(ggml_exec_trace_flush());

    const test_trace_header header = read_header(path);
    const std::vector<ggml_exec_trace_event> events = read_events(path, header);
    const std::vector<std::string> strings = read_strings(path, header);
    GGML_ASSERT(metadata_value(events, strings, "gpu_timing") == "op");
    GGML_ASSERT(metadata_value(events, strings, "gpu_timing_spans_started") ==
            metadata_value(events, strings, "gpu_timing_spans_released"));

    uint32_t counts[4][5] = {};
    std::vector<ggml_exec_trace_event> epochs;
    for (const ggml_exec_trace_event & event : events) {
        if (event.kind == GGML_EXEC_TRACE_EVENT_GPU_TIMING) {
            const uint32_t scope = gpu_timing_scope(event);
            const uint32_t status = gpu_timing_status(event);
            GGML_ASSERT(gpu_timing_provenance(event) == GGML_EXEC_TRACE_TIMING_DEVICE_EVENT);
            GGML_ASSERT(scope >= 1 && scope <= 3 && status >= 1 && status <= 4);
            GGML_ASSERT(event.execution_id == execution_id && event.graph_id == 91);
            GGML_ASSERT(event.args[0] >= event.timestamp_ns);
            if (scope == GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP) {
                GGML_ASSERT(event.node_id >= 101 && event.node_id < 101 + node_ids.size());
            }
            counts[scope][status]++;
        } else if (event.kind == GGML_EXEC_TRACE_EVENT_GPU_TIMING_EPOCH) {
            GGML_ASSERT(event.args[3] == GGML_EXEC_TRACE_GPU_TIMING_COMPLETION_SYNCHRONIZE);
            epochs.push_back(event);
        }
    }
    for (uint32_t scope = 1; scope <= 3; ++scope) {
        GGML_ASSERT(counts[scope][GGML_EXEC_TRACE_GPU_TIMING_UNRESOLVED] == 0);
        GGML_ASSERT(counts[scope][GGML_EXEC_TRACE_GPU_TIMING_FAILED] == 0);
    }
    GGML_ASSERT(counts[GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_SUBMIT][GGML_EXEC_TRACE_GPU_TIMING_RESOLVED] >= 1);
    if (small_pool) {
        GGML_ASSERT(counts[GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP][GGML_EXEC_TRACE_GPU_TIMING_NOT_RECORDED] >= 1);
        GGML_ASSERT(counts[GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP][GGML_EXEC_TRACE_GPU_TIMING_RESOLVED] == 0);
    } else {
        GGML_ASSERT(counts[GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP][GGML_EXEC_TRACE_GPU_TIMING_RESOLVED] >= 1);
        for (uint32_t scope = 1; scope <= 3; ++scope) {
            GGML_ASSERT(counts[scope][GGML_EXEC_TRACE_GPU_TIMING_NOT_RECORDED] == 0);
        }
    }
    // Every compute synchronizes once, and each compute records a submission or graph-launch span.
    GGML_ASSERT(epochs.size() == 4);

    // The device reaches each start and end event after the host clock sample taken before its record call, and before the
    // synchronization returns. These bounds constrain the unknown device time of the epoch anchor; they must leave a non-empty range.
    for (const ggml_exec_trace_event & epoch : epochs) {
        int64_t lower = INT64_MIN;
        int64_t upper = INT64_MAX;
        for (const ggml_exec_trace_event & event : events) {
            if (event.kind != GGML_EXEC_TRACE_EVENT_GPU_TIMING || (event.args[3] >> 32) != epoch.args[1] ||
                    gpu_timing_status(event) != GGML_EXEC_TRACE_GPU_TIMING_RESOLVED) {
                continue;
            }
            const int64_t offset = (int64_t) event.args[1];
            const int64_t duration = (int64_t) event.args[2];
            lower = std::max(lower, (int64_t) event.timestamp_ns - offset);
            lower = std::max(lower, (int64_t) event.args[0] - offset - duration);
            upper = std::min(upper, (int64_t) epoch.args[0] - offset - duration);
        }
        std::printf("%s: epoch %llu spans=%llu anchor placement range=%lld ns\n", __func__,
                (unsigned long long) epoch.args[1], (unsigned long long) epoch.args[2], (long long) (upper - lower));
        GGML_ASSERT(lower <= upper);
    }
    std::printf("%s: resolved submit=%u graph=%u op=%u not_recorded_op=%u\n", __func__,
            counts[GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_SUBMIT][GGML_EXEC_TRACE_GPU_TIMING_RESOLVED],
            counts[GGML_EXEC_TRACE_GPU_TIMING_SCOPE_GRAPH_LAUNCH][GGML_EXEC_TRACE_GPU_TIMING_RESOLVED],
            counts[GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP][GGML_EXEC_TRACE_GPU_TIMING_RESOLVED],
            counts[GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP][GGML_EXEC_TRACE_GPU_TIMING_NOT_RECORDED]);
    return 0;
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc >= 2);
    const std::string mode = argv[1];
    if (mode == "disabled") {
        return test_disabled();
    }
    GGML_ASSERT(argc == 3);
    if (mode == "enabled") {
        return test_enabled(argv[2], false);
    }
    if (mode == "detailed") {
        return test_enabled(argv[2], true);
    }
    if (mode == "truncation") {
        return test_truncation(argv[2]);
    }
    if (mode == "integrity") {
        return test_integrity(argv[2]);
    }
    if (mode == "gpu-timing") {
        return test_gpu_timing_api(argv[2]);
    }
    if (mode == "gpu-timing-backend") {
        return test_gpu_timing_backend(argv[2], false);
    }
    if (mode == "gpu-timing-pool") {
        return test_gpu_timing_backend(argv[2], true);
    }
    GGML_ABORT("unknown mode: %s", mode.c_str());
}
