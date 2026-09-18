#pragma once

#include "common.cuh"

// delphi.cpp: device builder for the grouped routed-MoE schedule defined in ggml-moe-schedule.h.
// It does no host synchronization and its grids depend only on n_tokens*top_k and n_experts, so it can be captured in a CUDA graph.

// Scratch for ggml_cuda_moe_schedule_build, in int32 elements.
size_t ggml_cuda_moe_schedule_scratch_size(int n_experts, int64_t n_assign);

// ids_s1 is the row stride of ids in elements. inverse may be nullptr if it is not needed.
void ggml_cuda_moe_schedule_build(
        const int32_t * ids, int64_t ids_s1, int64_t n_tokens, int top_k, int n_experts,
        int32_t * offsets, int32_t * slot, int32_t * inverse, int32_t * scratch, cudaStream_t stream);

// GGML_CUDA_MOE_SCHEDULE=1 makes MUL_MAT_ID MMQ take its expert grouping from ggml_cuda_moe_schedule_build instead of mm_ids_helper.
bool ggml_cuda_moe_schedule_enabled(void);

// Entry points for tests and benchmarks, returned by the CUDA backend registry's get_proc_address.

// -1: use GGML_CUDA_MOE_SCHEDULE, 0: off, 1: on.
void ggml_backend_cuda_moe_schedule_set_mode(int mode);

// Builds a schedule from host ids on the device of backend and copies it back. arm 0 is ggml_cuda_moe_schedule_build, arm 1 is
// mm_ids_helper (distinct valid ids per token only). If device_us is not null, the build is repeated reps times and the mean device
// time of one build is written there.
bool ggml_backend_cuda_moe_schedule_build_host(
        ggml_backend_t backend, int arm, const int32_t * ids, int64_t n_tokens, int top_k, int n_experts,
        int32_t * offsets, int32_t * slot, int32_t * inverse, int reps, float * device_us);
