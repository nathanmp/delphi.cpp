#pragma once

#include "common.cuh"
#include "ggml-exec-trace.h"

#include <vector>

// Device-event timing for the execution trace, shared by CUDA and HIP.
// All events are created with timing enabled when the backend context is created. They are recorded only on the
// context main stream and never during stream capture. Spans between two completion points form one epoch, and each
// span is measured from the first start event of its epoch. The timer never waits: it reads elapsed times after an
// existing stream synchronization, or at release when a non-blocking query reports completion.
struct ggml_cuda_exec_trace_timer {
    ggml_cuda_exec_trace_timer(int device, uint32_t event_capacity, ggml_exec_trace_gpu_timing_mode mode);
    ~ggml_cuda_exec_trace_timer();

    ggml_cuda_exec_trace_timer(const ggml_cuda_exec_trace_timer &) = delete;
    ggml_cuda_exec_trace_timer & operator=(const ggml_cuda_exec_trace_timer &) = delete;

    bool op_spans() const { return mode == GGML_EXEC_TRACE_GPU_TIMING_OP; }

    // Returns a span index, or -1 if no start event was recorded.
    int begin(cudaStream_t stream, ggml_exec_trace_gpu_timing_scope scope, uint32_t node_index, uint32_t detail);
    // A non-zero detail replaces the value given to begin.
    void end(cudaStream_t stream, int span, uint32_t detail);
    // The caller guarantees that all work submitted to the timed stream is complete.
    void resolve(ggml_exec_trace_gpu_timing_completion completion);

    bool record(cudaEvent_t event, cudaStream_t stream);
    void report_epoch(ggml_exec_trace_gpu_timing_completion completion, bool complete);

    const int device;
    const ggml_exec_trace_gpu_timing_mode mode;
    // span i uses events[2*i] and events[2*i + 1]
    std::vector<cudaEvent_t> events;
    std::vector<ggml_exec_trace_event> spans;
    std::vector<uint8_t> span_failed;
    uint32_t n_spans = 0;
    cudaEvent_t last_recorded = nullptr;
};
