#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_EXEC_TRACE_SCHEMA_VERSION 5
#define GGML_EXEC_TRACE_ID_NONE 0

#define GGML_EXEC_TRACE_INTEGRITY_SCHEMA_VERSION 1

enum ggml_exec_trace_event_kind {
    GGML_EXEC_TRACE_EVENT_RUN_METADATA = 1,
    GGML_EXEC_TRACE_EVENT_EXECUTION_BEGIN,
    GGML_EXEC_TRACE_EVENT_EXECUTION_END,
    GGML_EXEC_TRACE_EVENT_UBATCH_BEGIN,
    GGML_EXEC_TRACE_EVENT_UBATCH_END,
    GGML_EXEC_TRACE_EVENT_BACKEND,
    GGML_EXEC_TRACE_EVENT_GRAPH,
    GGML_EXEC_TRACE_EVENT_NODE,
    GGML_EXEC_TRACE_EVENT_NODE_DIMS,
    GGML_EXEC_TRACE_EVENT_NODE_SOURCE,
    GGML_EXEC_TRACE_EVENT_NODE_SOURCE_DIMS,
    GGML_EXEC_TRACE_EVENT_NODE_VIEW,
    GGML_EXEC_TRACE_EVENT_NODE_MATMUL,
    GGML_EXEC_TRACE_EVENT_SCHED_SPLIT,
    GGML_EXEC_TRACE_EVENT_SCHED_SPLIT_INPUT,
    GGML_EXEC_TRACE_EVENT_FINGERPRINT,
    // Timeline events store the host start in timestamp_ns and the host end in args[0].
    // Copy spans use node_id for the source tensor and backend_id/device_id for the destination; args[3] packs the source device/backend IDs.
    GGML_EXEC_TRACE_EVENT_BACKEND_SUBMIT,
    GGML_EXEC_TRACE_EVENT_CPU_COMPUTE,
    GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
    GGML_EXEC_TRACE_EVENT_COPY_HANDLING,
    GGML_EXEC_TRACE_EVENT_COPY_WAIT_SRC,
    GGML_EXEC_TRACE_EVENT_COPY_WAIT_DST,
    GGML_EXEC_TRACE_EVENT_COPY_BODY,
    GGML_EXEC_TRACE_EVENT_DISPATCH,
    GGML_EXEC_TRACE_EVENT_GPU_GRAPH,
    // CLOCK_SYNC stores a trace-clock sample before the secondary clock in timestamp_ns, the secondary sample in args[0], and a trace-clock sample after it in args[1].
    GGML_EXEC_TRACE_EVENT_CLOCK_SYNC,
    // GPU_TIMING is a device interval measured with timing events, not a host span.
    // timestamp_ns and args[0] are host times read before the start and end events are recorded, so the device cannot reach an event earlier. args[1] is the device start offset from the epoch anchor, args[2] is the device duration,
    // and args[3] packs the epoch ID (high 32 bits) with the scope detail (GPU graph ID or node count). flags packs scope (bits 0-7), status (bits 8-15), and provenance (bits 16-23).
    GGML_EXEC_TRACE_EVENT_GPU_TIMING,
    // GPU_TIMING_EPOCH closes a group of GPU_TIMING spans that share one anchor event. timestamp_ns is the host enqueue time of the anchor, args[0] is the host time when
    // completion was known, args[1] is the epoch ID, args[2] is the span count, and args[3] is the completion source.
    GGML_EXEC_TRACE_EVENT_GPU_TIMING_EPOCH,
    // OFFLOAD_DECISION stores CPU compute, CPU transfer, candidate compute, and candidate transfer
    // nanoseconds in args[0..3].
    // backend_id/device_id identify the candidate. flags is the selected side.
    GGML_EXEC_TRACE_EVENT_OFFLOAD_DECISION,
    // OFFLOAD_BYTES stores candidate weight H2D bytes, CPU input D2H bytes, CPU output H2D bytes, and operation count in args[0..3].
    GGML_EXEC_TRACE_EVENT_OFFLOAD_BYTES,
    // OFFLOAD_CALIBRATION stores CPU operations/s, candidate operations/s, H2D bytes/s, and D2H bytes/s
    // in args[0..3]. flags is NUMA node + 1, or zero if unknown.
    GGML_EXEC_TRACE_EVENT_OFFLOAD_CALIBRATION,
    // CPU_NODE_TIMING is a host interval for one unfused CPU graph node. It is emitted only in op timing mode.
    // node_id identifies the node; args[0] is the end timestamp and args[1] is the number of graph nodes covered.
    GGML_EXEC_TRACE_EVENT_CPU_NODE_TIMING,
};

enum ggml_exec_trace_execution_phase {
    GGML_EXEC_TRACE_PHASE_UNKNOWN = 0,
    GGML_EXEC_TRACE_PHASE_PROMPT,
    GGML_EXEC_TRACE_PHASE_GENERATION_TARGET,
    GGML_EXEC_TRACE_PHASE_GENERATION_DRAFT,
};

enum ggml_exec_trace_offload_choice {
    GGML_EXEC_TRACE_OFFLOAD_CPU = 1,
    GGML_EXEC_TRACE_OFFLOAD_CANDIDATE,
};

// Timing provenance. Host spans are HOST_SAFE. VENDOR_EXTERNAL marks offline profiler data and never appears in a raw trace.
enum ggml_exec_trace_timing_provenance {
    GGML_EXEC_TRACE_TIMING_HOST_SAFE = 1,
    GGML_EXEC_TRACE_TIMING_DEVICE_EVENT,
    GGML_EXEC_TRACE_TIMING_VENDOR_EXTERNAL,
};

enum ggml_exec_trace_gpu_timing_mode {
    GGML_EXEC_TRACE_GPU_TIMING_OFF = 0,
    // one span for each eager backend submission and each GPU graph launch
    GGML_EXEC_TRACE_GPU_TIMING_SUBMIT,
    // SUBMIT plus one span for each eagerly launched node or fused node group; graph launches stay whole-graph spans
    GGML_EXEC_TRACE_GPU_TIMING_OP,
};

enum ggml_exec_trace_gpu_timing_scope {
    GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_SUBMIT = 1,
    GGML_EXEC_TRACE_GPU_TIMING_SCOPE_GRAPH_LAUNCH,
    GGML_EXEC_TRACE_GPU_TIMING_SCOPE_EAGER_OP,
};

enum ggml_exec_trace_gpu_timing_status {
    GGML_EXEC_TRACE_GPU_TIMING_RESOLVED = 1,
    // completion was not known when the span had to be released; no wait was added
    GGML_EXEC_TRACE_GPU_TIMING_UNRESOLVED,
    // the event pool was full; no events were recorded
    GGML_EXEC_TRACE_GPU_TIMING_NOT_RECORDED,
    // the runtime rejected an event record or query
    GGML_EXEC_TRACE_GPU_TIMING_FAILED,
};

enum ggml_exec_trace_gpu_timing_completion {
    // an existing backend stream synchronization returned
    GGML_EXEC_TRACE_GPU_TIMING_COMPLETION_SYNCHRONIZE = 1,
    // a non-blocking event query reported completion
    GGML_EXEC_TRACE_GPU_TIMING_COMPLETION_QUERY,
    // completion was not known
    GGML_EXEC_TRACE_GPU_TIMING_COMPLETION_NONE,
};

enum ggml_exec_trace_clock_kind {
    GGML_EXEC_TRACE_CLOCK_MONOTONIC = 1,
};

enum ggml_exec_trace_dispatch_impl {
    GGML_EXEC_TRACE_DISPATCH_UNKNOWN = 0,
    GGML_EXEC_TRACE_DISPATCH_GPU_CUBLAS,
    GGML_EXEC_TRACE_DISPATCH_GPU_HIPBLAS,
    GGML_EXEC_TRACE_DISPATCH_GPU_MMVF,
    GGML_EXEC_TRACE_DISPATCH_GPU_MMVQ,
    GGML_EXEC_TRACE_DISPATCH_GPU_MMQ,
    GGML_EXEC_TRACE_DISPATCH_GPU_MMF,
    GGML_EXEC_TRACE_DISPATCH_GPU_FWHT,
    GGML_EXEC_TRACE_DISPATCH_GPU_MUL_MAT_ID_FALLBACK,
    GGML_EXEC_TRACE_DISPATCH_CPU_LLAMAFILE,
    GGML_EXEC_TRACE_DISPATCH_CPU_IQP,
    GGML_EXEC_TRACE_DISPATCH_CPU_VEC_DOT,
    GGML_EXEC_TRACE_DISPATCH_CPU_REPACK_GEMV,
    GGML_EXEC_TRACE_DISPATCH_CPU_REPACK_GEMM,
    GGML_EXEC_TRACE_DISPATCH_CPU_REPACK_GEMM_GEMV,
    GGML_EXEC_TRACE_DISPATCH_CPU_FWHT,
    GGML_EXEC_TRACE_DISPATCH_CPU_IQP_VEC_DOT,
    GGML_EXEC_TRACE_DISPATCH_CPU_IQK,
    GGML_EXEC_TRACE_DISPATCH_COUNT,
};

enum ggml_exec_trace_dispatch_variant {
    GGML_EXEC_TRACE_DISPATCH_VARIANT_NONE = 0,
    GGML_EXEC_TRACE_DISPATCH_VARIANT_DEFAULT,
    GGML_EXEC_TRACE_DISPATCH_VARIANT_BAD_PADDING_OR_TYPE,
    GGML_EXEC_TRACE_DISPATCH_VARIANT_TRANSPOSED,
    GGML_EXEC_TRACE_DISPATCH_VARIANT_FUSED,
    GGML_EXEC_TRACE_DISPATCH_VARIANT_SORTED_FALLBACK,
    GGML_EXEC_TRACE_DISPATCH_VARIANT_PER_EXPERT,
    GGML_EXEC_TRACE_DISPATCH_VARIANT_COUNT,
};

enum ggml_exec_trace_gpu_graph_state {
    GGML_EXEC_TRACE_GPU_GRAPH_CAPTURE = 1,
    GGML_EXEC_TRACE_GPU_GRAPH_UPDATE,
    GGML_EXEC_TRACE_GPU_GRAPH_REPLAY,
    GGML_EXEC_TRACE_GPU_GRAPH_FALLBACK,
};

enum ggml_exec_trace_gpu_graph_reason {
    GGML_EXEC_TRACE_GPU_GRAPH_REASON_NONE = 0,
    GGML_EXEC_TRACE_GPU_GRAPH_REASON_DISABLED,
    GGML_EXEC_TRACE_GPU_GRAPH_REASON_INCOMPATIBLE_MUL_MAT_ID,
    GGML_EXEC_TRACE_GPU_GRAPH_REASON_WARMUP,
    GGML_EXEC_TRACE_GPU_GRAPH_REASON_PROPERTIES_CHANGED,
    GGML_EXEC_TRACE_GPU_GRAPH_REASON_UPDATE_REINSTANTIATED,
};

enum ggml_exec_trace_wait_reason {
    GGML_EXEC_TRACE_WAIT_PREVIOUS_SPLIT = 1,
    GGML_EXEC_TRACE_WAIT_MOE_IDS,
    GGML_EXEC_TRACE_WAIT_EVAL_CALLBACK,
    GGML_EXEC_TRACE_WAIT_SCHEDULER_SYNCHRONIZE,
    GGML_EXEC_TRACE_WAIT_GRAPH_REALLOCATION,
    GGML_EXEC_TRACE_WAIT_COPY_DST_REUSE,
};

enum ggml_exec_trace_wait_api {
    GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE = 1,
    GGML_EXEC_TRACE_WAIT_API_EVENT_SYNCHRONIZE,
};

enum ggml_exec_trace_copy_path {
    GGML_EXEC_TRACE_COPY_USER_INPUT_SYNC = 1,
    GGML_EXEC_TRACE_COPY_MOE_EXPERT_ASYNC,
    GGML_EXEC_TRACE_COPY_BACKEND_ASYNC,
    GGML_EXEC_TRACE_COPY_FALLBACK_SYNC,
};

enum ggml_exec_trace_execution_kind {
    GGML_EXEC_TRACE_EXECUTION_DECODE = 1,
    GGML_EXEC_TRACE_EXECUTION_PREFILL,
    GGML_EXEC_TRACE_EXECUTION_ENCODE,
};

enum ggml_exec_trace_file_flags {
    GGML_EXEC_TRACE_FILE_TRUNCATED = 1 << 0,
};

enum ggml_exec_trace_integrity_build_flags {
    GGML_EXEC_TRACE_INTEGRITY_BUILD_ASSERTS   = 1 << 0,
    GGML_EXEC_TRACE_INTEGRITY_BUILD_NO_OPT    = 1 << 1,
    GGML_EXEC_TRACE_INTEGRITY_BUILD_ASAN      = 1 << 2,
    GGML_EXEC_TRACE_INTEGRITY_BUILD_TSAN      = 1 << 3,
    GGML_EXEC_TRACE_INTEGRITY_BUILD_UBSAN     = 1 << 4,
};

enum ggml_exec_trace_integrity_runtime_flags {
    GGML_EXEC_TRACE_INTEGRITY_RUNTIME_CUDA_GRAPHS_DISABLED = 1 << 0,
    GGML_EXEC_TRACE_INTEGRITY_RUNTIME_HIP_GRAPHS_DISABLED  = 1 << 1,
    GGML_EXEC_TRACE_INTEGRITY_RUNTIME_CUDA_OP_TRACE        = 1 << 2,
    GGML_EXEC_TRACE_INTEGRITY_RUNTIME_HIP_OP_TRACE         = 1 << 3,
    GGML_EXEC_TRACE_INTEGRITY_RUNTIME_FULL_DETAIL          = 1 << 4,
    GGML_EXEC_TRACE_INTEGRITY_RUNTIME_GPU_TIMING           = 1 << 5,
};

// Fixed-size events keep the append path allocation-free after trace initialization.
struct ggml_exec_trace_event {
    uint64_t timestamp_ns;
    uint64_t args[4];
    uint32_t execution_id;
    uint32_t ubatch_id;
    uint32_t graph_id;
    uint32_t node_id;
    uint32_t kind;
    uint32_t flags;
    uint32_t split_id;
    uint32_t backend_id;
    uint32_t device_id;
    uint32_t thread_id;
};

struct ggml_exec_trace_dispatch_record {
    uint32_t node_index;
    uint32_t op;
    uint32_t implementation;
    uint32_t variant;
    uint32_t config_0;
    uint32_t config_1;
};

GGML_API void     ggml_exec_trace_init(void);
GGML_API bool     ggml_exec_trace_enabled(void);
GGML_API bool     ggml_exec_trace_integrity_enabled(void);
GGML_API bool     ggml_exec_trace_observing(void);
GGML_API bool     ggml_exec_trace_detailed(void);
GGML_API void     ggml_exec_trace_disable(const char * reason);
GGML_API uint64_t ggml_exec_trace_now_ns(void);

GGML_API uint32_t ggml_exec_trace_intern(const char * value);
GGML_API void     ggml_exec_trace_metadata(const char * key, const char * value);
GGML_API void     ggml_exec_trace_record(struct ggml_exec_trace_event event);

GGML_API uint32_t ggml_exec_trace_execution_begin(
        enum ggml_exec_trace_execution_kind kind,
        uint32_t n_tokens,
        uint32_t n_batch,
        uint32_t n_ubatch,
        uint32_t n_threads,
        uint32_t n_threads_batch);
GGML_API void ggml_exec_trace_execution_end(uint32_t execution_id, int32_t status);

// The phase is thread-local and is copied into EXECUTION_BEGIN. Returns the previous phase for scoped use.
GGML_API enum ggml_exec_trace_execution_phase ggml_exec_trace_set_execution_phase(
        enum ggml_exec_trace_execution_phase phase);

GGML_API uint32_t ggml_exec_trace_ubatch_begin(
        uint32_t n_tokens,
        uint32_t n_sequences,
        uint32_t n_unique_sequences,
        uint32_t n_outputs);
GGML_API void ggml_exec_trace_ubatch_end(uint32_t ubatch_id, int32_t status);

GGML_API uint32_t ggml_exec_trace_next_graph_id(void);
GGML_API uint32_t ggml_exec_trace_next_node_id(void);
GGML_API uint32_t ggml_exec_trace_next_gpu_graph_id(void);

// Backend context is valid only during the matching graph-compute call. Node IDs are indexed in backend graph order.
GGML_API void ggml_exec_trace_backend_begin(
        uint32_t graph_id,
        uint32_t split_id,
        uint32_t backend_id,
        uint32_t device_id,
        const uint32_t * node_ids,
        uint32_t n_nodes);
GGML_API void ggml_exec_trace_backend_end(void);
GGML_API void ggml_exec_trace_set_node(uint32_t node_index);
GGML_API uint64_t ggml_exec_trace_cpu_node_begin(void);
GGML_API void ggml_exec_trace_cpu_node_end(uint64_t start_ns, uint32_t node_index, uint32_t node_count);
GGML_API void ggml_exec_trace_dispatch(
        enum ggml_op op,
        enum ggml_exec_trace_dispatch_impl implementation,
        enum ggml_exec_trace_dispatch_variant variant,
        uint32_t config_0,
        uint32_t config_1);
GGML_API void ggml_exec_trace_dispatch_capture_begin(
        struct ggml_exec_trace_dispatch_record * records,
        uint32_t capacity,
        uint32_t * count);
GGML_API void ggml_exec_trace_dispatch_capture_end(void);
GGML_API void ggml_exec_trace_dispatch_replay(
        const struct ggml_exec_trace_dispatch_record * records,
        uint32_t count);
GGML_API void ggml_exec_trace_gpu_graph(
        enum ggml_exec_trace_gpu_graph_state state,
        uint32_t gpu_graph_id,
        uint32_t node_count,
        enum ggml_exec_trace_gpu_graph_reason reason);

// Device-event timing. Backends own the timing events; these functions only fill and record trace events.
// The mode is OFF unless GGML_EXEC_TRACE is active.
GGML_API enum ggml_exec_trace_gpu_timing_mode ggml_exec_trace_gpu_timing_get_mode(void);
GGML_API uint32_t ggml_exec_trace_gpu_timing_event_capacity(void);
GGML_API uint32_t ggml_exec_trace_next_gpu_timing_epoch_id(void);
// Fills a pending GPU_TIMING event from the current backend context and the host clock. node_index is used only for EAGER_OP spans.
GGML_API void ggml_exec_trace_gpu_timing_prepare(
        struct ggml_exec_trace_event * event,
        enum ggml_exec_trace_gpu_timing_scope scope,
        uint32_t node_index,
        uint32_t detail);
// Records a prepared span. It keeps the IDs captured by prepare, so it can run later on another thread.
GGML_API void ggml_exec_trace_gpu_timing_report(
        struct ggml_exec_trace_event * event,
        enum ggml_exec_trace_gpu_timing_status status,
        uint32_t epoch_id,
        uint64_t start_offset_ns,
        uint64_t duration_ns);
GGML_API void ggml_exec_trace_gpu_timing_epoch(
        const struct ggml_exec_trace_event * anchor_span,
        uint32_t epoch_id,
        uint64_t complete_ns,
        uint32_t span_count,
        enum ggml_exec_trace_gpu_timing_completion completion);

GGML_API void ggml_exec_trace_integrity_graph(
        uint64_t ops_shapes_hash,
        uint64_t backend_hash,
        uint64_t split_hash,
        uint64_t node_count,
        uint64_t split_count,
        uint64_t cross_backend_edges);
GGML_API void ggml_exec_trace_integrity_copy(
        enum ggml_exec_trace_copy_path path,
        uint32_t split_id,
        uint32_t src_backend_id,
        uint32_t dst_backend_id,
        uint64_t bytes);
GGML_API void ggml_exec_trace_integrity_eval_callback(void);

GGML_API uint64_t ggml_exec_trace_hash_init(void);
GGML_API uint64_t ggml_exec_trace_hash_bytes(uint64_t hash, const void * data, uint64_t size);
GGML_API uint64_t ggml_exec_trace_hash_u64(uint64_t hash, uint64_t value);

// Flush is intended for shutdown and tests. It does not synchronize backend work.
GGML_API bool ggml_exec_trace_flush(void);

#ifdef __cplusplus
}
#endif
