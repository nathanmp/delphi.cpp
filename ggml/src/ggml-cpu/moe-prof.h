#pragma once

#include <stdbool.h>
#include <stdint.h>

// GGML internal header

// Phase profiler for the CPU routed-expert path, enabled with GGML_MOE_PROF=1.
// Diagnostic only: when the variable is unset every site is one predictable
// branch on a cached flag, so instrumented builds stay usable for benchmarks.
// Counters are per thread and are summed into a report at process exit.

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_moe_prof_phase {
    GGML_MOE_PROF_QUANT_SRC1, // src1 f32 -> vec_dot_type into the node's wdata
    GGML_MOE_PROF_ROW_MAP,    // build matrix_row_counts / matrix_rows (thread 0 only)
    GGML_MOE_PROF_BARRIER,    // the barrier that publishes both of the above
    GGML_MOE_PROF_REPACK,     // IQK weight conversion to the transient Q8_K_R8
    GGML_MOE_PROF_GEMM,       // IQK expert GEMM
    GGML_MOE_PROF_TOTAL,      // whole MUL_MAT_ID op; the report derives "other" from it
    GGML_MOE_PROF_COUNT,
};

bool ggml_moe_prof_enabled(void);

void ggml_moe_prof_add(enum ggml_moe_prof_phase phase, int64_t us);

// one sample per (expert, MUL_MAT_ID node) with routed rows > 0, to show how far
// the per expert batch is from the IQK repack threshold
void ggml_moe_prof_add_expert_rows(int64_t rows);

#ifdef __cplusplus
}
#endif
