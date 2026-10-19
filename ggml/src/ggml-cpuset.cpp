#include "ggml-cpuset.h"

#include <mutex>
#include <vector>

#if defined(__linux__)
#include <sched.h>

// the process mask at the load of the library, before any thread of ours pins itself
static const cpu_set_t g_process_mask = [] {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        for (int i = 0; i < CPU_SETSIZE; i++) {
            CPU_SET(i, &set);
        }
    }
    return set;
}();
#endif

static std::mutex         g_mutex;
static std::vector<bool>  g_limit; // empty: no limit

void ggml_cpuset_restrict(const bool * mask, int n) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_limit.clear();
    if (mask != nullptr) {
        g_limit.assign(mask, mask + n);
    }
}

bool ggml_cpuset_allowed(int cpu) {
    if (cpu < 0) {
        return false;
    }
#if defined(__linux__)
    if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &g_process_mask)) {
        return false;
    }
#endif
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_limit.empty() || ((size_t) cpu < g_limit.size() && g_limit[cpu]);
}

int ggml_cpuset_filter(int * cpus, int n) {
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (ggml_cpuset_allowed(cpus[i])) {
            cpus[k++] = cpus[i];
        }
    }
    return k;
}

int ggml_cpuset_list(int * cpus, int n) {
    int k = 0;
#if defined(__linux__)
    const int max_cpu = CPU_SETSIZE;
#else
    const int max_cpu = 1024;
#endif
    for (int c = 0; c < max_cpu && k < n; c++) {
        if (ggml_cpuset_allowed(c)) {
            cpus[k++] = c;
        }
    }
    return k;
}
