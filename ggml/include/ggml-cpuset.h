#pragma once

#include "ggml.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

    // delphi: the CPUs that the helper threads may pin to (staging copy threads, scheduler workers, expert cache refill
    // threads, NUMA loader threads). A CPU is allowed when it is in the affinity mask that the process had at start-up (taskset,
    // cgroup) and, if set, in the application limit (--cpu-mask / --cpu-range). Compute threads follow the thread pool
    // parameters instead.
    //
    // Without Linux affinity calls, every CPU is allowed.

    // set the application limit: mask[i] for CPU i, n entries; NULL removes the limit
    GGML_API void ggml_cpuset_restrict(const bool * mask, int n);

    GGML_API bool ggml_cpuset_allowed(int cpu);

    // keep the allowed CPUs of cpus (in order); returns the new count
    GGML_API int  ggml_cpuset_filter(int * cpus, int n);

    // all allowed CPUs below max_cpus in increasing order; returns the count (at most n)
    GGML_API int  ggml_cpuset_list(int * cpus, int n);

#ifdef __cplusplus
}
#endif
