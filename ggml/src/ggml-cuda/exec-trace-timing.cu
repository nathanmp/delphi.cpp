#include "exec-trace-timing.cuh"

#include <cmath>

static uint64_t ggml_cuda_exec_trace_ms_to_ns(float ms) {
    return (uint64_t) std::llround((double) ms * 1e6);
}

ggml_cuda_exec_trace_timer::ggml_cuda_exec_trace_timer(int device, uint32_t event_capacity, ggml_exec_trace_gpu_timing_mode mode) :
    device(device), mode(mode) {
    ggml_cuda_set_device(device);
    events.reserve(event_capacity);
    for (uint32_t i = 0; i < event_capacity; ++i) {
        cudaEvent_t event = nullptr;
        if (cudaEventCreateWithFlags(&event, cudaEventDefault) != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_WARN("%s: created %u of %u timing events on device %d; later spans are not recorded\n",
                    __func__, i, event_capacity, device);
            break;
        }
        events.push_back(event);
    }
    if (events.size() % 2 != 0) {
        (void) cudaEventDestroy(events.back());
        events.pop_back();
    }
    spans.resize(events.size() / 2);
    span_failed.resize(spans.size());
}

ggml_cuda_exec_trace_timer::~ggml_cuda_exec_trace_timer() {
    if (n_spans > 0) {
        // Release must not wait. On one stream, completion of the newest event implies completion of all older events.
        const cudaError_t status = cudaEventQuery(last_recorded);
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
        }
        const bool complete = status == cudaSuccess;
        report_epoch(complete ? GGML_EXEC_TRACE_GPU_TIMING_COMPLETION_QUERY : GGML_EXEC_TRACE_GPU_TIMING_COMPLETION_NONE, complete);
    }
    for (cudaEvent_t event : events) {
        (void) cudaEventDestroy(event);
    }
    (void) cudaGetLastError();
}

bool ggml_cuda_exec_trace_timer::record(cudaEvent_t event, cudaStream_t stream) {
    if (cudaEventRecord(event, stream) != cudaSuccess) {
        // Clear the error so that the next kernel launch check does not report it.
        (void) cudaGetLastError();
        return false;
    }
    last_recorded = event;
    return true;
}

int ggml_cuda_exec_trace_timer::begin(cudaStream_t stream, ggml_exec_trace_gpu_timing_scope scope, uint32_t node_index, uint32_t detail) {
    if (n_spans == spans.size()) {
        // Events become reusable only at the next completion point. Report the gap instead of waiting.
        ggml_exec_trace_event event;
        ggml_exec_trace_gpu_timing_prepare(&event, scope, node_index, detail);
        ggml_exec_trace_gpu_timing_report(&event, GGML_EXEC_TRACE_GPU_TIMING_NOT_RECORDED, 0, 0, 0);
        return -1;
    }

    const uint32_t span = n_spans;
    ggml_exec_trace_gpu_timing_prepare(&spans[span], scope, node_index, detail);
    if (!record(events[2*span], stream)) {
        ggml_exec_trace_gpu_timing_report(&spans[span], GGML_EXEC_TRACE_GPU_TIMING_FAILED, 0, 0, 0);
        return -1;
    }
    span_failed[span] = 0;
    n_spans++;
    return (int) span;
}

void ggml_cuda_exec_trace_timer::end(cudaStream_t stream, int span, uint32_t detail) {
    if (span < 0) {
        return;
    }
    ggml_exec_trace_event & event = spans[span];
    // Read the host clock before the record call. The device can reach the event before the call returns.
    event.args[0] = ggml_exec_trace_now_ns();
    if (!record(events[2*span + 1], stream)) {
        span_failed[span] = 1;
    }
    if (detail != 0) {
        event.args[3] = detail;
    }
}

void ggml_cuda_exec_trace_timer::resolve(ggml_exec_trace_gpu_timing_completion completion) {
    if (n_spans > 0) {
        report_epoch(completion, true);
    }
}

void ggml_cuda_exec_trace_timer::report_epoch(ggml_exec_trace_gpu_timing_completion completion, bool complete) {
    const uint64_t complete_ns = ggml_exec_trace_now_ns();
    const uint32_t epoch_id = ggml_exec_trace_next_gpu_timing_epoch_id();
    for (uint32_t i = 0; i < n_spans; ++i) {
        ggml_exec_trace_gpu_timing_status status = complete ?
                GGML_EXEC_TRACE_GPU_TIMING_RESOLVED : GGML_EXEC_TRACE_GPU_TIMING_UNRESOLVED;
        float offset_ms = 0.0f;
        float duration_ms = 0.0f;
        if (span_failed[i]) {
            status = GGML_EXEC_TRACE_GPU_TIMING_FAILED;
        } else if (complete) {
            // Elapsed-time queries do not block; they return cudaErrorNotReady for incomplete events.
            cudaError_t err = i == 0 ? cudaSuccess : cudaEventElapsedTime(&offset_ms, events[0], events[2*i]);
            if (err == cudaSuccess) {
                err = cudaEventElapsedTime(&duration_ms, events[2*i], events[2*i + 1]);
            }
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                status = err == cudaErrorNotReady ? GGML_EXEC_TRACE_GPU_TIMING_UNRESOLVED : GGML_EXEC_TRACE_GPU_TIMING_FAILED;
            } else if (offset_ms < 0.0f || duration_ms < 0.0f) {
                status = GGML_EXEC_TRACE_GPU_TIMING_FAILED;
            }
        }
        const bool resolved = status == GGML_EXEC_TRACE_GPU_TIMING_RESOLVED;
        ggml_exec_trace_gpu_timing_report(&spans[i], status, epoch_id,
                resolved ? ggml_cuda_exec_trace_ms_to_ns(offset_ms) : 0,
                resolved ? ggml_cuda_exec_trace_ms_to_ns(duration_ms) : 0);
    }
    ggml_exec_trace_gpu_timing_epoch(&spans[0], epoch_id, complete_ns, n_spans, completion);
    n_spans = 0;
    last_recorded = nullptr;
}
