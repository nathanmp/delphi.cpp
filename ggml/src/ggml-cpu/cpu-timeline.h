#pragma once

#include <stdbool.h>
#include <stdint.h>

// GGML internal header

// Per-thread timeline of CPU graph computes, enabled with GGML_CPU_TIMELINE=<file>. Diagnostic only: when the variable is
// unset every site is one predictable branch on a cached flag. Recording starts when <file>.start exists (checked every 64
// graph computes) and stops when the buffers fill or <file>.stop exists; the file is then written once. Analyze it with
// tools/cpu-timeline/analyze.py.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cgraph;
struct ggml_tensor;

// Main thread, before the workers start: returns the record id of this graph compute, or -1 when not recording.
int32_t ggml_cpu_timeline_begin(const struct ggml_cgraph * cgraph, int n_threads);
// Main thread, after the workers joined and the affinity was cleared.
void ggml_cpu_timeline_end(int32_t call);

// Worker side; call is the id from begin(), never -1.
void ggml_cpu_timeline_enter(int32_t call, int ith);
// tensor is the node on the main thread (ith 0) and NULL elsewhere; t0/t1/t2: compute start, compute end, barrier exit.
void ggml_cpu_timeline_node(int32_t call, int ith, int node, int n_fused, const struct ggml_tensor * tensor,
                            int64_t t0, int64_t t1, int64_t t2);
void ggml_cpu_timeline_exit(int32_t call, int ith);

int64_t ggml_cpu_timeline_now(void);

#ifdef __cplusplus
}
#endif
