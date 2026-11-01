#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "iqp.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "ggml-exec-trace.h"
#include "ggml-cpuset.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "common.h"
#include "iqk/iqk_mul_mat.h"
#include "moe-prof.h"
#include "cpu-timeline.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <syscall.h>
#include <unistd.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

static bool ggml_cpu_iqk_mul_mat_enabled(void) {
    const char * value = getenv("GGML_CPU_IQK_MUL_MAT");
    return value == NULL || strcmp(value, "0") != 0;
}

// GGML_CPU_BATCHED_VEC_DOT=0 restores the per-row expert dots for A/B tests.
static bool ggml_cpu_batched_vec_dot_enabled(void) {
    const char * value = getenv("GGML_CPU_BATCHED_VEC_DOT");
    return value == NULL || strcmp(value, "0") != 0;
}

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
float ggml_table_f32_ue4m3[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_row_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_G128] = {
        .from_float               = quantize_row_q4_g128,
        .vec_dot                  = ggml_vec_dot_q4_g128_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
    [GGML_TYPE_IQ2_K] = {
        .from_float               = quantize_row_iq2_k,
        .vec_dot                  = ggml_vec_dot_iq2_k_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_K] = {
        .from_float               = quantize_row_iq3_k,
        .vec_dot                  = ggml_vec_dot_iq3_k_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_K] = {
        .from_float               = quantize_row_iq4_k,
        .vec_dot                  = ggml_vec_dot_iq4_k_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ5_K] = {
        .from_float               = quantize_row_iq5_k,
        .vec_dot                  = ggml_vec_dot_iq5_k_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ6_K] = {
        .from_float               = quantize_row_iq6_k,
        .vec_dot                  = ggml_vec_dot_iq6_k_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_KS] = {
        .from_float               = quantize_row_iq2_ks,
        .vec_dot                  = ggml_vec_dot_iq2_ks_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_KS] = {
        .from_float               = quantize_row_iq3_ks,
        .vec_dot                  = ggml_vec_dot_iq3_ks_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_KS] = {
        .from_float               = quantize_row_iq4_ks,
        .vec_dot                  = ggml_vec_dot_iq4_ks_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ5_KS] = {
        .from_float               = quantize_row_iq5_ks,
        .vec_dot                  = ggml_vec_dot_iq5_ks_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// --numa tensors: the state of one node's workers, in that node's memory (ggml_numa_node_state_alloc)
struct ggml_numa_node_state {
    atomic_int GGML_CACHE_ALIGN n_barrier;        // ggml_numa_node_barrier
    atomic_int GGML_CACHE_ALIGN n_barrier_passed; // also the generation of the node's MUL_MAT_ID work (piece stealing)
    atomic_int GGML_CACHE_ALIGN frontier;         // highest expert the node's own workers have started in this op, or -1
    size_t   scratch_size;
    uint8_t * scratch;                            // node-local MUL_MAT_ID work memory, see ggml_numa_mmid_layout
};

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;

    int32_t timeline_call;    // GGML_CPU_TIMELINE record of the running graph compute, or -1

    // --numa tensors only (numa_n_nodes is 0 otherwise)
    int numa_n_nodes;
    int numa_graph_threads;                       // the worker count of the node ranks below (ggml_numa_prepare_workers)
    uint32_t * numa_cpus;                         // [node*GGML_NUMA_MAX_CPUS + i]: the CPUs each node may use
    uint32_t numa_node_cpus[GGML_NUMA_MAX_NODES];
    int numa_node_workers[GGML_NUMA_MAX_NODES];
    int numa_steal_order[GGML_NUMA_MAX_NODES][GGML_NUMA_MAX_NODES]; // the other nodes, nearest first
    struct ggml_numa_node_state * numa_state[GGML_NUMA_MAX_NODES];
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
    int numa_node;
    int numa_cpu;
    int numa_rank;
    int numa_n_threads;
    bool numa_node_leader;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
    uint32_t id; // physical Linux NUMA node id
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
    double tp_share[GGML_NUMA_MAX_NODES]; // GGML_NUMA_STRATEGY_TENSORS column-split weights
    uint32_t distance[GGML_NUMA_MAX_NODES][GGML_NUMA_MAX_NODES]; // SLIT distance between nodes[i] and nodes[j] (tensor mode)
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

static void ggml_numa_init_tp_shares(void) {
    const uint32_t n_nodes = g_state.numa.n_nodes;
    double shares[GGML_NUMA_MAX_NODES];

    if (!ggml_numa_tp_parse_shares(getenv("GGML_NUMA_TP_SHARES"), n_nodes, shares)) {
        for (uint32_t n = 0; n < n_nodes; ++n) {
            shares[n] = 1.0;
        }
    }

    double total = 0.0;
    for (uint32_t n = 0; n < n_nodes; ++n) {
        total += shares[n];
    }

    GGML_LOG_INFO("numa tensors: column-split shares:");
    for (uint32_t n = 0; n < n_nodes; ++n) {
        g_state.numa.tp_share[n] = shares[n];
        GGML_LOG_CONT(" node%u=%.1f%%", g_state.numa.nodes[n].id, 100.0 * shares[n] / total);
    }
    GGML_LOG_CONT("\n");
}

#if defined(__gnu_linux__)
// Distances between the active nodes, from /sys/devices/system/node/nodeN/distance (one value per system node).
// If a file cannot be read, a node is at 10 from itself and at 20 from the others.
static void ggml_numa_init_distances(void) {
    const uint32_t n_nodes = g_state.numa.n_nodes;
    for (uint32_t i = 0; i < n_nodes; ++i) {
        uint32_t row[GGML_NUMA_MAX_NODES];
        uint32_t n_row = 0;
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/distance", g_state.numa.nodes[i].id);
        FILE * f = fopen(path, "r");
        if (f) {
            while (n_row < GGML_NUMA_MAX_NODES && fscanf(f, "%u", &row[n_row]) == 1) {
                ++n_row;
            }
            fclose(f);
        }
        for (uint32_t j = 0; j < n_nodes; ++j) {
            const uint32_t id = g_state.numa.nodes[j].id;
            g_state.numa.distance[i][j] = id < n_row ? row[id] : (i == j ? 10 : 20);
        }
    }
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate physical nodes
    uint32_t n_system_nodes = 0;
    while (n_system_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", n_system_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++n_system_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", n_system_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (n_system_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    uint32_t n_active_nodes = 0;
    for (uint32_t n = 0; n < n_system_nodes; ++n) {
        struct ggml_numa_node node = {0};
        node.id = n;
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0 &&
                    (numa_flag != GGML_NUMA_STRATEGY_TENSORS || CPU_ISSET(c, &g_state.numa.cpuset))) {
                node.cpus[node.n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");

        if (numa_flag == GGML_NUMA_STRATEGY_TENSORS) {
            if (node.n_cpus > 0) {
                g_state.numa.nodes[n_active_nodes++] = node;
            }
        } else {
            g_state.numa.nodes[n] = node;
        }
    }
    g_state.numa.n_nodes = (numa_flag == GGML_NUMA_STRATEGY_TENSORS) ? n_active_nodes : n_system_nodes;

    if (numa_flag == GGML_NUMA_STRATEGY_TENSORS) {
        if (g_state.numa.n_nodes < 2) {
            GGML_ABORT("--numa tensors requires at least two NUMA nodes allowed by the process affinity mask");
        }
        GGML_LOG_INFO("numa tensors: active nodes and affinity-allowed CPUs:\n");
        for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
            const struct ggml_numa_node * node = &g_state.numa.nodes[n];
            GGML_LOG_INFO("numa tensors: node %u CPUs", node->id);
            for (uint32_t c = 0; c < node->n_cpus; ++c) {
                GGML_LOG_CONT(" %u", node->cpus[c]);
            }
            GGML_LOG_CONT("\n");
        }
        ggml_numa_init_tp_shares();
        ggml_numa_init_distances();
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    if (numa_flag == GGML_NUMA_STRATEGY_TENSORS) {
        GGML_ABORT("--numa tensors is supported only on Linux");
    }
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

bool ggml_numa_is_tensor_mode(void) {
    return g_state.numa.numa_strategy == GGML_NUMA_STRATEGY_TENSORS;
}

uint32_t ggml_numa_get_node_count(void) {
    return ggml_numa_is_tensor_mode() ? g_state.numa.n_nodes : 0;
}

void ggml_numa_tp_row_range_for_node(
        uint32_t node, int64_t n_rows, int64_t align, int64_t * row_start, int64_t * row_end) {
    ggml_numa_tp_row_range(g_state.numa.tp_share, g_state.numa.n_nodes, node, n_rows, align, row_start, row_end);
}

// GGML_NUMA_NODE_LOCAL=0: MUL_MAT_ID under --numa tensors keeps its per-node data in the shared work buffer and waits for
// all nodes after the src1 conversion (A/B). It also turns off GGML_NUMA_STEAL.
static bool ggml_numa_node_local_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_NUMA_NODE_LOCAL");
        enabled = env ? atoi(env) != 0 : 1;
    }
    return enabled != 0;
}

// GGML_NUMA_STEAL=0: a node that has no more pieces of an expert does not take pieces of the other nodes (A/B).
static bool ggml_numa_steal_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_NUMA_STEAL");
        enabled = env ? atoi(env) != 0 : 1;
    }
    return enabled != 0;
}

#if defined(__gnu_linux__)
// Anonymous pages that fault in on `node` (MPOL_PREFERRED: another node if this one is full), whoever touches them first.
static void * ggml_numa_node_mmap(uint32_t node, size_t size) {
    void * data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
        return NULL;
    }
    unsigned long node_mask = 1UL << g_state.numa.nodes[node].id;
    if (syscall(SYS_mbind, data, size, MPOL_PREFERRED, &node_mask, sizeof(node_mask) * CHAR_BIT, 0) != 0) {
        GGML_LOG_WARN("numa tensors: mbind of %zu bytes to node %u failed: %s\n",
                size, g_state.numa.nodes[node].id, strerror(errno));
    }
    return data;
}

static void ggml_numa_node_munmap(void * data, size_t size) {
    if (data && munmap(data, size) != 0) {
        GGML_LOG_ERROR("numa tensors: munmap failed: %s\n", strerror(errno));
    }
}
#else
static void * ggml_numa_node_mmap(uint32_t node, size_t size) {
    UNUSED(node);
    return calloc(1, size);
}

static void ggml_numa_node_munmap(void * data, size_t size) {
    UNUSED(size);
    free(data);
}
#endif

static struct ggml_numa_node_state * ggml_numa_node_state_alloc(uint32_t node) {
    struct ggml_numa_node_state * st = ggml_numa_node_mmap(node, sizeof(struct ggml_numa_node_state));
    if (st) {
        atomic_store_explicit(&st->n_barrier, 0, memory_order_relaxed);
        atomic_store_explicit(&st->n_barrier_passed, 0, memory_order_relaxed);
        atomic_store_explicit(&st->frontier, -1, memory_order_relaxed);
        st->scratch_size = 0;
        st->scratch = NULL;
    }
    return st;
}

static void ggml_numa_node_state_free(struct ggml_numa_node_state * st) {
    if (st) {
        ggml_numa_node_munmap(st->scratch, st->scratch_size);
        ggml_numa_node_munmap(st, sizeof(struct ggml_numa_node_state));
    }
}

// Make the node's scratch at least `size` bytes. Call only while no worker runs.
static void ggml_numa_node_scratch_reserve(struct ggml_numa_node_state * st, uint32_t node, size_t size) {
    if (st->scratch_size >= size) {
        return;
    }
    size = GGML_PAD(size + size / 2, 4096); // grow by half again, so a slowly growing batch does not remap every time
    ggml_numa_node_munmap(st->scratch, st->scratch_size);
    st->scratch = ggml_numa_node_mmap(node, size);
    st->scratch_size = st->scratch ? size : 0;
}

// Barrier of one node's workers: it does not wait for the other nodes. Each pass increments n_barrier_passed, also when
// the node has only one worker.
static void ggml_numa_node_barrier(struct ggml_numa_node_state * st, int n_workers) {
    const int n_passed = atomic_load_explicit(&st->n_barrier_passed, memory_order_relaxed);

    if (atomic_fetch_add_explicit(&st->n_barrier, 1, memory_order_seq_cst) == n_workers - 1) {
        atomic_store_explicit(&st->n_barrier, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&st->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    while (atomic_load_explicit(&st->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }
#ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&st->n_barrier_passed, 0, memory_order_seq_cst);
#else
    atomic_thread_fence(memory_order_seq_cst);
#endif
}

#if defined(__gnu_linux__)
struct ggml_numa_load_worker {
    struct ggml_tensor * tensor;
    const void * data;
    uint32_t node;
    uint32_t rank;      // index of this worker inside its node
    uint32_t n_workers; // workers on this node
    int result;
};

// Loader threads per node. One thread per node makes the copy and repack single-threaded per socket, which is slow for a large model.
#define GGML_NUMA_LOAD_WORKERS_PER_NODE 8

static void * ggml_numa_load_expert_worker(void * userdata) {
    struct ggml_numa_load_worker * worker = (struct ggml_numa_load_worker *) userdata;
    const struct ggml_numa_node * node = &g_state.numa.nodes[worker->node];

    // a CPU of the node inside the allowed set (ggml-cpuset.h); the memory policy below sets where the pages go, so any
    // allowed CPU does when the node has none
    uint32_t n_ok = 0;
    for (uint32_t i = 0; i < node->n_cpus; ++i) {
        n_ok += ggml_cpuset_allowed((int) node->cpus[i]);
    }
    uint32_t cpu = node->cpus[worker->rank % node->n_cpus];
    if (n_ok > 0) {
        uint32_t k = worker->rank % n_ok;
        for (uint32_t i = 0; i < node->n_cpus; ++i) {
            if (ggml_cpuset_allowed((int) node->cpus[i]) && k-- == 0) {
                cpu = node->cpus[i];
                break;
            }
        }
    } else {
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (ggml_cpuset_allowed(c)) {
                cpu = (uint32_t) c;
                break;
            }
        }
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    const int rv = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (rv != 0) {
        GGML_LOG_ERROR("numa tensors: failed to pin loader for node %u to CPU %u: %s\n",
                node->id, cpu, strerror(rv));
        worker->result = rv;
        return NULL;
    }
    // Linux memory policy is per-thread, so this bind makes first touch land on the wanted node.
    // The policy goes away with the loader thread and never moves a page that exists.
    unsigned long node_mask = 1UL << node->id;
    const long policy_result = syscall(
            SYS_set_mempolicy, MPOL_BIND, &node_mask, sizeof(node_mask) * CHAR_BIT);
    if (policy_result != 0) {
        GGML_LOG_ERROR("numa tensors: failed to bind loader memory policy to node %u: %s\n",
                node->id, strerror(errno));
        worker->result = errno;
        return NULL;
    }

    // column split: this node touches every expert, but only its own row range of each one
    int64_t row_start, row_end;
    ggml_numa_tp_row_range(g_state.numa.tp_share, g_state.numa.n_nodes, worker->node,
            worker->tensor->ne[1], GGML_NUMA_TP_PLAIN_ALIGN, &row_start, &row_end);

    if (row_start >= row_end) {
        return NULL;
    }

    const size_t expert_size = worker->tensor->nb[2];
    const size_t row_size    = worker->tensor->nb[1];

    for (int64_t expert = worker->rank; expert < worker->tensor->ne[2]; expert += worker->n_workers) {
        memcpy(
                (char *) worker->tensor->data + expert * expert_size + row_start * row_size,
                (const char *) worker->data + expert * expert_size + row_start * row_size,
                (size_t) (row_end - row_start) * row_size);
    }

    return NULL;
}

#endif

bool ggml_numa_load_expert_tensor(struct ggml_tensor * tensor, const void * data, size_t size) {
    if (!ggml_numa_is_tensor_mode() || !tensor || !data || tensor->ne[2] < 1 ||
            tensor->ne[3] != 1 || size != ggml_nbytes(tensor)) {
        return false;
    }

#if defined(__gnu_linux__)
    struct ggml_numa_load_worker workers[GGML_NUMA_MAX_NODES * GGML_NUMA_LOAD_WORKERS_PER_NODE] = {0};
    pthread_t threads[GGML_NUMA_MAX_NODES * GGML_NUMA_LOAD_WORKERS_PER_NODE] = {0};

    uint32_t n_workers = 0;
    for (uint32_t node = 0; node < g_state.numa.n_nodes; ++node) {
        const uint32_t per_node = MIN(
                (uint32_t) GGML_NUMA_LOAD_WORKERS_PER_NODE, g_state.numa.nodes[node].n_cpus);
        for (uint32_t rank = 0; rank < per_node; ++rank, ++n_workers) {
            workers[n_workers].tensor    = tensor;
            workers[n_workers].data      = data;
            workers[n_workers].node      = node;
            workers[n_workers].rank      = rank;
            workers[n_workers].n_workers = per_node;
            const int result = pthread_create(
                    &threads[n_workers], NULL, ggml_numa_load_expert_worker, &workers[n_workers]);
            if (result != 0) {
                GGML_ABORT("numa tensors: failed to create loader for node %u: %s",
                        g_state.numa.nodes[node].id, strerror(result));
            }
        }
    }

    bool ok = true;
    for (uint32_t i = 0; i < n_workers; ++i) {
        pthread_join(threads[i], NULL);
        ok = ok && workers[i].result == 0;
    }
    return ok;
#else
    UNUSED(tensor);
    UNUSED(data);
    UNUSED(size);
    return false;
#endif
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        if (params->ith == 0) {
            ggml_exec_trace_dispatch(dst->op, GGML_EXEC_TRACE_DISPATCH_CPU_FWHT,
                    GGML_EXEC_TRACE_DISPATCH_VARIANT_DEFAULT, params->nth, 0);
        }
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    // spec-invariant: 2-8 columns as separate 1-column products, so each takes the 1-token path
    if (src1->ne[1] >= 2 && src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_cpu_spec_inv(src1->ne[1])) {
        for (int64_t j = 0; j < src1->ne[1]; j++) {
            struct ggml_tensor s1 = *src1;
            s1.ne[1] = 1;
            s1.data  = (char *) src1->data + j*src1->nb[1];
            struct ggml_tensor d1 = *dst;
            d1.ne[1]  = 1;
            d1.data   = (char *) dst->data + j*dst->nb[1];
            d1.src[1] = &s1;
            ggml_compute_forward_mul_mat(params, &d1);
            ggml_barrier(params->threadpool);
        }
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        if (ith == 0) {
            ggml_exec_trace_dispatch(dst->op, GGML_EXEC_TRACE_DISPATCH_CPU_LLAMAFILE,
                    GGML_EXEC_TRACE_DISPATCH_VARIANT_DEFAULT, nth, 0);
        }
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

    // Try the optional IQK implementation before falling back to the stock CPU matmul. Unsupported
    // layouts/types return false and continue below. Like the IQ panel gemm underneath, this must come after
    // the barrier above: when src1 needs conversion it reads the q8_K rows from the work buffer, which the
    // loop above fills cooperatively.
    if (!params->use_ref && ggml_cpu_iqk_mul_mat_enabled() && dst->type == GGML_TYPE_F32) {
        const void * iqk_src1 = src1->type == vec_dot_type ? src1->data : params->wdata;
        const size_t iqk_nb11 = src1->type == vec_dot_type ? nb11 : ggml_row_size(vec_dot_type, ne10);
        const size_t iqk_nb12 = src1->type == vec_dot_type ? nb12 : iqk_nb11 * ne11;
        const size_t iqk_nb13 = src1->type == vec_dot_type ? nb13 : iqk_nb12 * ne12;

        if (iqk_mul_mat_4d(ne01, ne11, ne00,
                ne02, ne03, ne12, ne13,
                nb02, nb03, iqk_nb12, iqk_nb13,
                nb2 / sizeof(float), nb3 / sizeof(float),
                src0->type, src0->data, nb01,
                vec_dot_type, iqk_src1, iqk_nb11,
                dst->data, nb1 / sizeof(float), ith, nth)) {
            return;
        }
    }

    // IQ panel gemm (see iqp.h) - must come after the barrier above, it consumes the q8_K rows
    // of src1 from the work buffer
    if (ggml_cpu_iqp_supports_mul_mat(dst) && !params->use_ref) {
        if (ith == 0) {
            ggml_exec_trace_dispatch(dst->op, GGML_EXEC_TRACE_DISPATCH_CPU_IQP,
                    GGML_EXEC_TRACE_DISPATCH_VARIANT_DEFAULT, nth, 0);
        }
        ggml_compute_forward_mul_mat_iqp(params, dst);
        return;
    }

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        if (ith == 0) {
            ggml_exec_trace_dispatch(dst->op, GGML_EXEC_TRACE_DISPATCH_CPU_LLAMAFILE,
                    GGML_EXEC_TRACE_DISPATCH_VARIANT_DEFAULT, nth, 0);
        }
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    if (ith == 0) {
        ggml_exec_trace_dispatch(dst->op, GGML_EXEC_TRACE_DISPATCH_CPU_VEC_DOT,
                GGML_EXEC_TRACE_DISPATCH_VARIANT_DEFAULT, nth,
                ((uint32_t) MIN(nchunk1, UINT16_MAX) << 16) | (uint32_t) MIN(nchunk0, UINT16_MAX));
    }

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

// GGML_CPU_G128_ROWS=0 restores the per-row vec_dot loop for G128 experts (A/B).
static bool ggml_cpu_g128_rows_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_CPU_G128_ROWS");
        enabled = env ? atoi(env) != 0 : 1;
    }
    return enabled != 0;
}

static bool ggml_cpu_numa_quant_flat_enabled(void) {
    const char * env = getenv("GGML_NUMA_QUANT_FLAT");
    return !env || atoi(env) != 0;
}

// GGML_CPU_G128_PANEL=0 recomputes the activation correction and scale per weight row (A/B).
static bool ggml_cpu_g128_panel_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_CPU_G128_PANEL");
        enabled = env ? atoi(env) != 0 : 1;
    }
    return enabled != 0;
}

// GGML_CPU_NUMA_CHUNKS: pieces per worker of a node's rows of an expert under --numa tensors (0: one static slice each).
static int ggml_cpu_numa_chunks_per_thread(void) {
    static int chunks = -1;
    if (chunks < 0) {
        const char * env = getenv("GGML_CPU_NUMA_CHUNKS");
        chunks = env ? MAX(0, atoi(env)) : 2;
    }
    return chunks;
}

static bool ggml_cpu_g128_rows_supported(const struct ggml_tensor * src0) {
    return src0->type == GGML_TYPE_Q4_G128 && type_traits_cpu[src0->type].vec_dot_type == GGML_TYPE_Q8_0 &&
        ggml_cpu_g128_rows_enabled();
}

// Bytes of one thread's G128 activation panels (GGML_Q4_G128_MAX_ROWS rows of ne00 values).
static size_t ggml_cpu_g128_panels_size(int64_t ne00) {
    return GGML_Q4_G128_MAX_ROWS * GGML_PAD(GGML_Q4_G128_PANEL_SIZE(ne00), 64);
}

// GGML_CPU_G128_GEMM: fewest rows (tokens) of one expert that run the G128 tile GEMM, 0 = off. Fewer rows (decode, MTP
// verification) keep the rows kernel, whose results are bitwise those of single-row decode.
static int ggml_cpu_g128_gemm_min_rows(void) {
#if defined(__AVX2__)
    static int rows = -1;
    if (rows < 0) {
        const char * env = getenv("GGML_CPU_G128_GEMM");
        rows = env ? MAX(0, atoi(env)) : 4;
    }
    return rows;
#else
    return 0;
#endif
}

// A G128 MUL_MAT_ID with enough tokens for an expert to reach the GEMM: every src1 row also gets a GEMM panel.
static bool ggml_cpu_g128_gemm_node(const struct ggml_tensor * node) {
    const int min_rows = ggml_cpu_g128_gemm_min_rows();
    return min_rows > 0 && ggml_cpu_g128_rows_supported(node->src[0]) && node->src[1]->type == GGML_TYPE_F32 &&
        node->src[2]->ne[1] >= min_rows && node->src[0]->ne[0] / QK4_G128 <= GGML_Q4_G128_GEMM_MAX_GROUPS;
}

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata,
    char * g128_panels,
    const char * gemm_panels,
    ggml_vec_dot_batch_rows_t batch_rows) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    if (gemm_panels && ir1_end - ir1_start >= ggml_cpu_g128_gemm_min_rows()) {
        // G128 tile GEMM over this expert's rows, in blocks of up to 64 rows
        const size_t panel_size = GGML_Q4_G128_GEMM_PANEL_SIZE(ne10);
        for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += 64) {
            const int ny = (int) MIN(64, ir1_end - iir1);
            const void * panels[64];
            float * dsts[64];
            for (int r = 0; r < ny; ++r) {
                const struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1 + r);
                const int64_t i11 = row_mapping.i1 % ne11;
                const int64_t i12 = row_mapping.i2;
                panels[r] = gemm_panels + (i11 + i12*ne11)*panel_size;
                dsts[r] = (float *) ((char *) dst->data + (row_mapping.i1*nb1 + i12*nb2)) + ir0_start;
            }
            ggml_gemm_q4_g128_panels(ne00, src0_cur + ir0_start*nb01, nb01, (int) (ir0_end - ir0_start), panels, dsts, ny);
        }
        return;
    }

    if (g128_panels) {
        // Decode each weight row once for up to GGML_Q4_G128_MAX_ROWS of this expert's tokens (MTP verification and small
        // batches route a few tokens to the same expert), with each token's activation panel built once for all weight rows.
        // The results equal the per-row vec_dot.
        const size_t panel_stride = ggml_cpu_g128_panels_size(ne00) / GGML_Q4_G128_MAX_ROWS;
        const bool use_panel = ggml_cpu_g128_panel_enabled();
        for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += GGML_Q4_G128_MAX_ROWS) {
            const int ny = (int) MIN(GGML_Q4_G128_MAX_ROWS, ir1_end - iir1);
            // a single row is memory-bound: building its panel would only add work
            const bool panel = use_panel && ny > 1;
            const void * cols[GGML_Q4_G128_MAX_ROWS];
            const void * panels[GGML_Q4_G128_MAX_ROWS];
            float * dsts[GGML_Q4_G128_MAX_ROWS];
            for (int r = 0; r < ny; ++r) {
                const struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1 + r);
                const int64_t i11 = row_mapping.i1 % ne11;
                const int64_t i12 = row_mapping.i2;
                cols[r] = (const char *) wdata + (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12*ne11)*row_size
                        : (i11*nb11 + i12*nb12));
                dsts[r] = (float *) ((char *) dst->data + (row_mapping.i1*nb1 + i12*nb2));
                panels[r] = g128_panels + r * panel_stride;
                if (panel) {
                    ggml_q4_g128_panel(ne00, cols[r], g128_panels + r * panel_stride);
                }
            }
            for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                float out[GGML_Q4_G128_MAX_ROWS];
                ggml_vec_dot_q4_g128_q8_0_rows(ne00, out, src0_cur + ir0*nb01, cols, panel ? panels : NULL, ny);
                for (int r = 0; r < ny; ++r) {
                    dsts[r][ir0] = out[r];
                }
            }
        }
        return;
    }

    // Reuse each decoded weight block across up to four activation rows.
    if (batch_rows && ir0_start < ir0_end && ir1_end - ir1_start > 1) {
        for (int64_t first = ir1_start; first < ir1_end; first += 4) {
            const int nr = (int) MIN(4, ir1_end - first);
            const void * ys[4];
            float * outs[4];
            for (int r = 0; r < nr; ++r) {
                const struct mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, first + r);
                const int64_t i11 = rm.i1 % ne11;
                ys[r] = (const char *) wdata + (src1_cont || src1->type != vec_dot_type
                    ? (i11 + rm.i2 * ne11) * row_size : i11 * nb11 + rm.i2 * nb12);
                outs[r] = (float *) ((char *) dst->data + rm.i1 * nb1 + rm.i2 * nb2) + ir0_start;
            }
            batch_rows(ne00, outs, src0_cur + ir0_start * nb01, nb01, ys, nr, ir0_end - ir0_start);
        }
        return;
    }

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

// --numa tensors: where a MUL_MAT_ID node keeps its per-node data in each node's scratch (ggml_numa_node_state). All nodes
// use the same offsets, so a node can find the chunk counters of another node.
struct ggml_numa_mmid_layout {
    size_t src1;   // src1 converted to vec_dot_type (empty when src1 has that type)
    size_t gemm;   // G128 GEMM panels of every src1 row (empty without the G128 GEMM)
    size_t counts; // matrix_row_counts [n_as]
    size_t rows;   // matrix_rows [n_as][n_ids * n_tokens]
    size_t chunks; // chunk counters [n_as], one cache line each
    size_t size;
};

static struct ggml_numa_mmid_layout ggml_numa_mmid_layout_of(const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    const struct ggml_tensor * ids  = node->src[2];
    const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
    const size_t n_as = (size_t) src0->ne[2];

    struct ggml_numa_mmid_layout l;
    size_t off = 0;

    l.src1 = off;
    if (src1->type != vec_dot_type) {
        off += ggml_row_size(vec_dot_type, ggml_nelements(src1));
    }
    off = GGML_PAD(off, CACHE_LINE_SIZE);

    l.gemm = off;
    if (ggml_cpu_g128_gemm_node(node)) {
        off += ggml_nrows(src1) * GGML_Q4_G128_GEMM_PANEL_SIZE(src1->ne[0]);
    }
    off = GGML_PAD(off, CACHE_LINE_SIZE);

    l.counts = off;
    off = GGML_PAD(off + n_as * sizeof(int64_t), CACHE_LINE_SIZE);

    l.rows = off;
    off = GGML_PAD(off + n_as * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping), CACHE_LINE_SIZE);

    l.chunks = off;
    off += CACHE_LINE_SIZE * n_as;

    l.size = off;
    return l;
}

// --numa tensors: true when MUL_MAT_ID keeps its per-node data in node memory and syncs each node alone after the src1
// conversion. That needs the whole gang (a node barrier counts the node's workers) and every node's scratch reserved by
// ggml_graph_compute. Every thread gets the same answer.
static bool ggml_numa_mmid_node_local(const struct ggml_compute_params * params, size_t size) {
    const struct ggml_threadpool * tp = params->threadpool;
    if (!ggml_numa_node_local_enabled() || !tp || tp->numa_n_nodes < 2 || params->nth != tp->numa_graph_threads ||
            params->numa_n_threads == 0 ||
            (uint32_t) tp->numa_n_nodes != ggml_numa_get_node_count()) {
        return false;
    }
    for (int node = 0; node < tp->numa_n_nodes; ++node) {
        if (!tp->numa_state[node] || tp->numa_state[node]->scratch_size < size) {
            return false;
        }
    }
    return true;
}

// The pieces of one expert for the n_workers workers of a node: nchunk0 x nchunk1 pieces of dr0 x dr1 rows over nr0 weight
// rows and nr1 token rows. Worker i takes piece i first; the others claim the rest from the node's counter.
static void ggml_mmid_chunking(
        int64_t nr0, int64_t nr1, int n_workers, int numa_chunks, bool g128_fused, bool disable_chunking,
        int64_t * nchunk0, int64_t * nchunk1, int64_t * dr0, int64_t * dr1) {
    int chunk_size = 16;
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    int64_t nc0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nc1 = (nr1 + chunk_size - 1) / chunk_size;

    if (numa_chunks > 0 && nr0 > nr1) {
        // a node's workers claim its rows of the expert in pieces, so a worker that starts late or runs slow does not
        // hold up the node (static slices: the slowest worker ends ~10% after the median one in MiMo MTP decode). Short
        // pieces stream worse: MiMo CPU expert time per verify pass with 1/2/4/8 pieces per worker 39.0/37.9/41.4/46.5 ms,
        // and a static slice plus a tail of 16-row pieces 39.5 ms.
        nc0 = MIN(nr0, (int64_t) n_workers * numa_chunks);
        nc1 = 1;
    } else if (nc0 * nc1 < n_workers * 4 || disable_chunking) {
        nc0 = nr0 > nr1 ? n_workers : 1;
        nc1 = nr0 > nr1 ? 1 : n_workers;
    }

    int64_t d0 = (nr0 + nc0 - 1) / nc0;
    const int64_t d1 = (nr1 + nc1 - 1) / nc1;
    if ((numa_chunks > 0 && nr0 > nr1) || g128_fused) {
        // whole 8-row pieces: the fused SwiGLU below then takes the vector path for every row, as the unfused GLU does
        d0 = GGML_PAD(d0, 8);
        nc0 = (nr0 + d0 - 1) / d0;
    }

    *nchunk0 = nc0;
    *nchunk1 = nc1;
    *dr0 = d0;
    *dr1 = d1;
}

// One piece of expert cur_a: weight rows [ir0_start, ir0_end) for its token rows [ir1_start, ir1_end). With g128_fused,
// also the same rows of up (fuse_up) and the SwiGLU of both into fuse_dst.
static void ggml_mmid_piece(
        struct ggml_tensor * dst, struct ggml_tensor * fuse_up, struct ggml_tensor * fuse_dst, bool g128_fused,
        int64_t cur_a, int64_t ir0_start, int64_t ir0_end, int64_t ir1_start, int64_t ir1_end,
        const struct mmid_row_mapping * matrix_rows, size_t row_size, bool src1_cont, const void * wdata,
        char * g128_panels, const char * gemm_panels, ggml_vec_dot_batch_rows_t batch_rows) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];

    ggml_compute_forward_mul_mat_id_one_chunk(
        dst, src0, src1, ids, cur_a,
        ir0_start, ir0_end, ir1_start, ir1_end,
        (const char *) src0->data + cur_a * src0->nb[2],
        matrix_rows, row_size, src1_cont, wdata, g128_panels, gemm_panels, batch_rows
    );
    if (g128_fused) {
        // the same rows of up, then SwiGLU over them: one node and one barrier instead of three
        ggml_compute_forward_mul_mat_id_one_chunk(
            fuse_up, fuse_up->src[0], src1, ids, cur_a,
            ir0_start, ir0_end, ir1_start, ir1_end,
            (const char *) fuse_up->src[0]->data + cur_a * fuse_up->src[0]->nb[2],
            matrix_rows, row_size, src1_cont, wdata, g128_panels, gemm_panels, batch_rows
        );
        for (int64_t ir1 = ir1_start; ir1 < ir1_end; ++ir1) {
            const struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);
            const size_t off = row_mapping.i1*dst->nb[1] + row_mapping.i2*dst->nb[2];
            ggml_vec_swiglu_f32((int) (ir0_end - ir0_start),
                (float *) ((char *) fuse_dst->data + row_mapping.i1*fuse_dst->nb[1] + row_mapping.i2*fuse_dst->nb[2]) + ir0_start,
                (const float *) ((const char *) dst->data + off) + ir0_start,
                (const float *) ((const char *) fuse_up->data + off) + ir0_start);
        }
    }
}

// Expert prefetch for --numa tensors decode (GGML_CPU_EXPERT_PREFETCH=<experts per MUL_MAT_ID>, default 6, 0 = off).
// The CPU MUL_MAT_ID records which experts had rows in its last two calls. Before the inputs of the next CPU split
// arrive (the GPU still runs the dense part of the layer), each worker reads the rows of the predicted experts that it
// computes in the IQK static split, so that the expert kernel finds them in L2/L3 and DRAM is used while it is idle.
#define GGML_CPU_PF_MAX_AS 512
#define GGML_CPU_PF_WORDS  (GGML_CPU_PF_MAX_AS/64)
#define GGML_CPU_PF_SLOTS  1024
#define GGML_CPU_PF_MAX_NY 16
#define GGML_CPU_PF_MAX_HINT 64 // larger batches repack the experts (IQK), with another row split

struct ggml_cpu_pf_entry {
    const void * key; // src0->data
    uint64_t bits[2][GGML_CPU_PF_WORDS];
    int last;   // index of the newest set in bits
    int n_recs; // 0, 1 or 2 valid sets
    int32_t hint[GGML_CPU_PF_MAX_HINT]; // experts that a pre-gate predicts for the next call (ggml_cpu_expert_hint)
    int n_hint;
    int32_t pf[GGML_CPU_PF_MAX_HINT];   // the experts read for the next call, for the statistics
    int n_pf;
};

static struct ggml_cpu_pf_entry ggml_cpu_pf_table[GGML_CPU_PF_SLOTS];
static _Thread_local const struct ggml_compute_params * ggml_cpu_tls_params = NULL;

// statistics, written by the main thread or worker 0 only
static int64_t ggml_cpu_pf_stat_hits, ggml_cpu_pf_stat_used, ggml_cpu_pf_stat_pred, ggml_cpu_pf_stat_recs;
static int64_t ggml_cpu_pf_stat_calls, ggml_cpu_pf_stat_us, ggml_cpu_pf_stat_bytes;
// the wait for the split inputs after a prefetch: near 0 means that the prefetch took longer than the GPU work before the split
static int64_t ggml_cpu_pf_end_us, ggml_cpu_pf_stat_wait_us, ggml_cpu_pf_stat_waits, ggml_cpu_pf_stat_short;
static int64_t ggml_cpu_pf_stat_split_us; // compute time of the CPU splits after a prefetch
static bool    ggml_cpu_pf_dry;   // GGML_CPU_EXPERT_PREFETCH_DRY=1: predict, but do not read (measures the window)
static bool    ggml_cpu_pf_stats; // GGML_CPU_EXPERT_PREFETCH_STATS=1: print the statistics now and then, and at the exit


static void ggml_cpu_pf_print_stats(void) {
    const double recs = (double) MAX(1, ggml_cpu_pf_stat_recs);
    const double calls = (double) MAX(1, ggml_cpu_pf_stat_calls);
    fprintf(stderr, "cpu expert prefetch: %lld of %lld used experts predicted (%.1f %%), per MUL_MAT_ID %.1f used, %.1f predicted; "
            "%lld prefetch calls, %.1f us and %.1f MiB per call\n",
            (long long) ggml_cpu_pf_stat_hits, (long long) ggml_cpu_pf_stat_used,
            100.0*ggml_cpu_pf_stat_hits/MAX(1, ggml_cpu_pf_stat_used),
            ggml_cpu_pf_stat_used/recs, ggml_cpu_pf_stat_pred/recs,
            (long long) ggml_cpu_pf_stat_calls, ggml_cpu_pf_stat_us/calls, ggml_cpu_pf_stat_bytes/1048576.0/calls);
    fprintf(stderr, "cpu expert prefetch%s: input wait after the prefetch %.1f us mean, %lld of %lld waits below 20 us, "
            "split compute %.1f us mean\n", ggml_cpu_pf_dry ? " (dry)" : "",
            ggml_cpu_pf_stat_wait_us/(double) MAX(1, ggml_cpu_pf_stat_waits),
            (long long) ggml_cpu_pf_stat_short, (long long) ggml_cpu_pf_stat_waits,
            ggml_cpu_pf_stat_split_us/(double) MAX(1, ggml_cpu_pf_stat_waits));
}

int ggml_cpu_expert_prefetch_budget(void) {
    static int budget = -1;
    if (budget < 0) {
        // 6 experts per MUL_MAT_ID fit the GPU attention of a DeepSeek V4.1 layer on an R9700 (~580 us at 48 threads)
        const char * s = getenv("GGML_CPU_EXPERT_PREFETCH");
        budget = s ? MAX(0, atoi(s)) : 6;
        const char * d = getenv("GGML_CPU_EXPERT_PREFETCH_DRY");
        ggml_cpu_pf_dry = d && atoi(d) != 0;
        const char * st = getenv("GGML_CPU_EXPERT_PREFETCH_STATS");
        ggml_cpu_pf_stats = st && atoi(st) != 0;
        if (budget > 0 && ggml_cpu_pf_stats) {
            atexit(ggml_cpu_pf_print_stats);
        }
    }
    return budget;
}

static struct ggml_cpu_pf_entry * ggml_cpu_pf_find(const void * key, bool insert) {
    const size_t h = (size_t) ((((uintptr_t) key >> 6) * 0x9E3779B97F4A7C15ull) >> 54);
    for (int i = 0; i < GGML_CPU_PF_SLOTS; i++) {
        struct ggml_cpu_pf_entry * e = &ggml_cpu_pf_table[(h + i) & (GGML_CPU_PF_SLOTS - 1)];
        if (e->key == key) {
            return e;
        }
        if (e->key == NULL) {
            if (!insert) {
                return NULL;
            }
            e->key = key;
            return e;
        }
    }
    return NULL;
}

// the predicted experts in priority order: in both of the last two calls, then in the last one, then in the one before
static int ggml_cpu_pf_select(const struct ggml_cpu_pf_entry * e, int n_as, int budget, int32_t * out) {
    const uint64_t * a = e->bits[e->last];
    const uint64_t * b = e->bits[e->last ^ 1];
    const bool has_b = e->n_recs > 1;
    int n = 0;
    for (int pass = 0; pass < (has_b ? 3 : 1) && n < budget; pass++) {
        for (int w = 0; w < (n_as + 63)/64 && n < budget; w++) {
            uint64_t m = pass == 0 ? (has_b ? a[w] & b[w] : a[w]) : pass == 1 ? a[w] & ~b[w] : b[w] & ~a[w];
            while (m != 0 && n < budget) {
                const int bit = __builtin_ctzll(m);
                m &= m - 1;
                if (w*64 + bit < n_as) {
                    out[n++] = w*64 + bit;
                }
            }
        }
    }
    return n;
}

static void ggml_cpu_pf_record(const void * key, const int64_t * counts, int n_as, bool stats) {
    struct ggml_cpu_pf_entry * e = ggml_cpu_pf_find(key, true);
    if (e == NULL || n_as > GGML_CPU_PF_MAX_AS) {
        return;
    }
    if (stats && e->n_pf > 0) {
        for (int i = 0; i < e->n_pf; i++) {
            ggml_cpu_pf_stat_hits += counts[e->pf[i]] > 0;
        }
        for (int i = 0; i < n_as; i++) {
            ggml_cpu_pf_stat_used += counts[i] > 0;
        }
        ggml_cpu_pf_stat_pred += e->n_pf;
        ggml_cpu_pf_stat_recs++;
    }
    e->n_pf = 0;
    e->n_hint = 0; // a hint is for one call
    e->last ^= 1;
    memset(e->bits[e->last], 0, sizeof(e->bits[e->last]));
    for (int i = 0; i < n_as; i++) {
        if (counts[i] > 0) {
            e->bits[e->last][i/64] |= 1ull << (i%64);
        }
    }
    e->n_recs = MIN(2, e->n_recs + 1);
}

// The experts that the next MUL_MAT_ID of src0 will likely use, most likely first. They replace the history for that call.
void ggml_cpu_expert_hint(const void * src0_data, const int32_t * experts, int n) {
    if (ggml_cpu_expert_prefetch_budget() <= 0) {
        return;
    }
    struct ggml_cpu_pf_entry * e = ggml_cpu_pf_find(src0_data, true);
    if (e == NULL) {
        return;
    }
    e->n_hint = MIN(n, GGML_CPU_PF_MAX_HINT);
    memcpy(e->hint, experts, e->n_hint*sizeof(int32_t));
}

struct ggml_cpu_pf_item {
    const char * base; // expert matrix
    size_t       nb01;
    int64_t      ne01;
};

struct ggml_cpu_pf_job {
    int n_items;
    struct ggml_cpu_pf_item * items;
};

static volatile uint64_t ggml_cpu_pf_sink;

static void ggml_cpu_pf_run(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst); GGML_UNUSED(ith); GGML_UNUSED(nth);
    const struct ggml_cpu_pf_job * job = (const struct ggml_cpu_pf_job *) userdata;
    const struct ggml_compute_params * p = ggml_cpu_tls_params;
    if (p == NULL || p->numa_n_threads <= 0) {
        return;
    }
    uint64_t acc = 0;
    for (int i = 0; i < job->n_items; i++) {
        const struct ggml_cpu_pf_item * it = &job->items[i];
        int64_t s, e;
        ggml_numa_tp_row_range_for_node((uint32_t) p->numa_node, it->ne01, GGML_NUMA_TP_PLAIN_ALIGN, &s, &e);
        // the split of iqk_mul_mat_moe (no repack): ceil(rows/workers) rows per worker of the node
        const int64_t per = (e - s + p->numa_n_threads - 1)/p->numa_n_threads;
        const int64_t r0 = MIN(e, s + p->numa_rank*per);
        const int64_t r1 = MIN(e, r0 + per);
        const char * x  = it->base + r0*it->nb01;
        const char * xe = it->base + r1*it->nb01;
        for (; x + 256 <= xe; x += 256) {
            acc += *(const uint64_t *) x + *(const uint64_t *) (x + 64) + *(const uint64_t *) (x + 128) + *(const uint64_t *) (x + 192);
        }
        for (; x < xe; x += 64) {
            acc += *(const uint8_t *) x;
        }
    }
    if (acc == 1) {
        ggml_cpu_pf_sink = acc; // keeps the loads
    }
}

// Run by the CPU backend before the inputs of a split are copied. Returns false when nothing was prefetched.
bool ggml_cpu_prefetch_experts(struct ggml_cgraph * graph, int n_threads, struct ggml_threadpool * threadpool) {
    const int budget = ggml_cpu_expert_prefetch_budget();
    if (budget <= 0 || !ggml_numa_is_tensor_mode() || n_threads < (int) ggml_numa_get_node_count()) {
        return false;
    }
    const int64_t t0 = ggml_time_us();

    struct ggml_cpu_pf_item items[256];
    int n_items = 0;
    size_t bytes = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        const struct ggml_tensor * node = graph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] == NULL || node->src[2] == NULL ||
            (node->src[0]->flags & GGML_TENSOR_FLAG_NUMA_EXPERT) == 0) {
            continue;
        }
        if (node->src[2]->ne[1] > GGML_CPU_PF_MAX_NY) {
            continue;
        }
        const struct ggml_tensor * src0 = node->src[0];
        struct ggml_cpu_pf_entry * e = ggml_cpu_pf_find(src0->data, false);
        if (e == NULL || (e->n_recs == 0 && e->n_hint == 0) || src0->ne[2] > GGML_CPU_PF_MAX_AS) {
            continue;
        }
        int32_t pred[GGML_CPU_PF_MAX_AS];
        int np;
        if (e->n_hint > 0) {
            np = MIN(budget, e->n_hint);
            memcpy(pred, e->hint, np*sizeof(int32_t));
        } else {
            np = ggml_cpu_pf_select(e, (int) src0->ne[2], budget, pred);
        }
        e->n_pf = MIN(np, GGML_CPU_PF_MAX_HINT);
        memcpy(e->pf, pred, e->n_pf*sizeof(int32_t));
        for (int j = 0; j < np && n_items < (int) (sizeof(items)/sizeof(items[0])); j++) {
            items[n_items++] = (struct ggml_cpu_pf_item) { (const char *) src0->data + pred[j]*src0->nb[2], src0->nb[1], src0->ne[1] };
            bytes += src0->nb[2];
        }
    }
    if (n_items == 0) {
        return false;
    }

    if (ggml_cpu_pf_dry) {
        n_items = 0;
    }
    struct ggml_cpu_pf_job job = { n_items, items };
    _Alignas(64) char mem[16384];
    GGML_ASSERT(2*ggml_tensor_overhead() + ggml_graph_overhead_custom(4, false) <= sizeof(mem));
    struct ggml_init_params ip = { sizeof(mem), mem, true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * t = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, NULL, 0, ggml_cpu_pf_run, GGML_N_TASKS_MAX, &job);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(gf, t);
    struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, threadpool);
    GGML_ASSERT(cplan.work_size == 0);
    ggml_graph_compute(gf, &cplan);
    ggml_free(ctx);

    ggml_cpu_pf_stat_calls++;
    ggml_cpu_pf_stat_bytes += bytes;
    if (ggml_cpu_pf_stats) {
        ggml_cpu_pf_end_us = ggml_time_us();
        ggml_cpu_pf_stat_us += ggml_cpu_pf_end_us - t0;
    }
    return true;
}

// called when the CPU backend starts a graph: returns the start time when the graph follows a prefetch, else 0
int64_t ggml_cpu_prefetch_note_compute(void) {
    if (ggml_cpu_pf_end_us == 0) {
        return 0;
    }
    const int64_t now = ggml_time_us();
    const int64_t w = now - ggml_cpu_pf_end_us;
    ggml_cpu_pf_stat_wait_us += w;
    ggml_cpu_pf_stat_waits++;
    ggml_cpu_pf_stat_short += w < 20;
    ggml_cpu_pf_end_us = 0;
    return now;
}

// called when that graph is done
void ggml_cpu_prefetch_note_compute_end(int64_t t_start) {
    ggml_cpu_pf_stat_split_us += ggml_time_us() - t_start;
    if (ggml_cpu_pf_stat_waits % 65536 == 0) {
        ggml_cpu_pf_print_stats();
    }
}

// `fuse_up` and `fuse_dst` are the up-projection MUL_MAT_ID node and the GLU node of a
// fused MoE up+gate subgraph, or NULL for a plain MUL_MAT_ID. When set, `dst` supplies the
// gate weights and every other shape, and the routed rows are written to `fuse_dst`
// combined by SwiGLU or swiglu_clamp; `dst` itself is never written.
static void ggml_compute_forward_mul_mat_id_impl(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst,
              struct ggml_tensor * fuse_up,
              struct ggml_tensor * fuse_dst,
        float fuse_limit) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool prof = ggml_moe_prof_enabled();
    const int64_t prof_t_op = prof ? ggml_time_us() : 0;

    const int ith = params->ith;
    const int nth = params->nth;
    // a graph with fewer workers than nodes has no node ranks (numa_n_threads 0) and uses the ordinary schedule
    const bool numa_tensors = ggml_numa_is_tensor_mode() && params->numa_n_threads > 0 &&
        (src0->flags & GGML_TENSOR_FLAG_NUMA_EXPERT) != 0;
    const int work_ith = numa_tensors ? params->numa_rank : ith;
    const int work_nth = numa_tensors ? params->numa_n_threads : nth;
    const uint32_t n_nodes = numa_tensors ? ggml_numa_get_node_count() : 1;
    const uint32_t numa_node = numa_tensors ? (uint32_t) params->numa_node : 0;

    const enum ggml_type type = src0->type;

    // gate + up + SwiGLU of G128 experts through the row kernels (ggml_cpu_moe_fuse_up_gate_match)
    const bool g128_fused = fuse_up != NULL && ggml_cpu_g128_rows_supported(src0);

    const bool src1_cont = ggml_is_contiguous(src1);

    const ggml_vec_dot_batch_rows_t batch_rows = !params->use_ref && ggml_cpu_batched_vec_dot_enabled()
        ? ggml_get_vec_dot_batch_rows(type) : NULL;

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;
    size_t src1_scratch_stride = 0;

    if (src1->type != vec_dot_type) {
        src1_scratch_stride = GGML_PAD(
                ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
        incr_ptr_aligned(&wdata_cur, src1_scratch_stride * n_nodes, sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    // Every node claims chunks over the same experts, so each node needs its own counter block.
    // The block is node-major to keep a node's counters in its own cache lines.
    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_nodes][n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as * n_nodes, CACHE_LINE_SIZE);
    char (*node_current_chunk)[CACHE_LINE_SIZE] = atomic_current_chunk + (size_t) numa_node * n_as;

    // IQ panel gemm (see iqp.h); per expert eligibility is decided below, but the work buffer is
    // reserved for the whole node (ggml_graph_plan sizes it without params, use_ref only skips the dispatch)
    const bool iqp = ggml_cpu_iqp_supports_mul_mat_id(dst) && !params->use_ref;

    char * iqp_panels = NULL;

    if (iqp) {
        iqp_panels = incr_ptr_aligned(&wdata_cur, nth * ggml_cpu_iqp_scratch_size(dst), 64);
    }

    char * g128_panels = NULL;

    if (ggml_cpu_g128_rows_supported(src0)) {
        const size_t size = ggml_cpu_g128_panels_size(ne00);
        g128_panels = (char *) incr_ptr_aligned(&wdata_cur, nth * size, 64) + (size_t) params->ith * size;
    }

    // G128 GEMM panels of every src1 row, one copy per node (the work buffer is planned with use_ref unknown, as for IQP)
    const bool g128_gemm_node = ggml_cpu_g128_gemm_node(dst);
    const bool g128_gemm = g128_gemm_node && !params->use_ref;
    const size_t gemm_panel_size = GGML_Q4_G128_GEMM_PANEL_SIZE(ne10);
    const size_t gemm_node_stride = g128_gemm_node ? ggml_nrows(src1) * gemm_panel_size : 0;
    char * gemm_panels = NULL;

    if (g128_gemm_node) {
        gemm_panels = (char *) incr_ptr_aligned(&wdata_cur, gemm_node_stride * n_nodes, 64) + numa_node * gemm_node_stride;
    }

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    // src1 converted to vec_dot_type, one copy per node
    char * src1_conv = (char *) params->wdata + numa_node * src1_scratch_stride;

    // --numa tensors: the node's src1 copy, GEMM panels, row map and counters go to the node's own memory, and the node's
    // workers wait only for each other after the conversion (ggml_numa_mmid_node_local). With the global barrier and the
    // shared work buffer, every node waited for the slowest node, and its copies sat in one node's memory.
    struct ggml_threadpool * tp = params->threadpool;
    const struct ggml_numa_mmid_layout numa_layout = numa_tensors ? ggml_numa_mmid_layout_of(dst) : (struct ggml_numa_mmid_layout) {0};
    const bool node_local = numa_tensors && ggml_numa_mmid_node_local(params, numa_layout.size);

    if (node_local) {
        uint8_t * scratch = tp->numa_state[numa_node]->scratch;
        src1_conv          = (char *) scratch + numa_layout.src1;
        matrix_row_counts  = (int64_t *) (scratch + numa_layout.counts);
        matrix_rows        = (struct mmid_row_mapping *) (scratch + numa_layout.rows);
        node_current_chunk = (char (*)[CACHE_LINE_SIZE]) (scratch + numa_layout.chunks);
        if (g128_gemm_node) {
            gemm_panels = (char *) scratch + numa_layout.gemm;
        }
    }

    const int64_t prof_t_quant = prof ? ggml_time_us() : 0;

    if (src1->type != vec_dot_type) {
        char * wdata = src1_conv;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= src1_scratch_stride * n_nodes);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        if (g128_gemm) {
            // whole rows per worker: a worker makes the GEMM panel of each of its Q8_0 rows, so no barrier is needed between
            const int64_t nr1 = ne11*ne12*ne13;
            for (int64_t ir = work_ith; ir < nr1; ir += work_nth) {
                const int64_t i11 = ir % ne11;
                const int64_t i12 = (ir / ne11) % ne12;
                const int64_t i13 = ir / (ne11*ne12);
                void * q = wdata + i13*nbw3 + i12*nbw2 + i11*nbw1;
                from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), q, ne10);
                ggml_q4_g128_gemm_panel(ne10, q, gemm_panels + ir*gemm_panel_size);
            }
        } else if (numa_tensors && vec_dot_type == GGML_TYPE_Q8_K && ggml_cpu_numa_quant_flat_enabled()) {
            // Split blocks across all rows. A short row can have fewer blocks than this node has workers.
            const int64_t bs = ggml_blck_size(vec_dot_type);
            const int64_t blocks_per_row = ne10 / bs;
            const int64_t blocks = blocks_per_row * ne11 * ne12 * ne13;
            const int64_t end = blocks * (work_ith + 1) / work_nth;
            for (int64_t block = blocks * work_ith / work_nth; block < end;) {
                const int64_t row = block / blocks_per_row;
                const int64_t first = block % blocks_per_row;
                const int64_t count = MIN(blocks_per_row - first, end - block);
                const int64_t i11 = row % ne11;
                const int64_t i12 = (row / ne11) % ne12;
                const int64_t i13 = row / (ne11 * ne12);
                from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + first*bs*nb10),
                           wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + first*nbw0, count * bs);
                block += count;
            }
        } else

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (work_ith * ne10/bs) / work_nth;
                    int64_t ne10_block_end   = ((work_ith + 1) * ne10/bs) / work_nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    if (prof) {
        ggml_moe_prof_add(GGML_MOE_PROF_QUANT_SRC1, ggml_time_us() - prof_t_quant);
    }

    const int64_t prof_t_rows = prof ? ggml_time_us() : 0;

    // the row map is shared, or one per node when node-local
    if (node_local ? params->numa_node_leader : ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        const bool partial = ggml_mul_mat_id_is_partial(dst);

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                if (partial && (i02 < 0 || i02 >= n_as)) {
                    continue; // zeroed by all threads below
                }
                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // Rows of another part's experts are zero: the threads share them, instead of the first thread clearing them all while the
    // others wait (a partial MUL_MAT_ID of the expert cache leaves ~half of its rows, 16 KB each, to the GPU). A fused node
    // clears only its SwiGLU output, the one that is read.
    if (ggml_mul_mat_id_is_partial(dst)) {
        struct ggml_tensor * zt = fuse_dst ? fuse_dst : dst;
        int k = 0;
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                if ((i02 < 0 || i02 >= n_as) && k++ % nth == ith) {
                    memset((char *) zt->data + id*zt->nb[1] + iid1*zt->nb[2], 0, zt->ne[0]*sizeof(float));
                }
            }
        }
    }

    // reset the per-expert work counter - each node writes its own block, so the stores stay local
    for (int cur_a = work_ith; cur_a < n_as; cur_a += work_nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(node_current_chunk + cur_a);
        *current_chunk_ctr = work_nth;
    }
    if (node_local && params->numa_node_leader) {
        atomic_store_explicit(&tp->numa_state[numa_node]->frontier, -1, memory_order_relaxed);
    }

    if (prof) {
        ggml_moe_prof_add(GGML_MOE_PROF_ROW_MAP, ggml_time_us() - prof_t_rows);
    }

    const int64_t prof_t_barrier = prof ? ggml_time_us() : 0;

    if (node_local) {
        ggml_numa_node_barrier(tp->numa_state[numa_node], work_nth);
    } else {
        ggml_barrier(params->threadpool);
    }

    if (prof) {
        ggml_moe_prof_add(GGML_MOE_PROF_BARRIER, ggml_time_us() - prof_t_barrier);
    }

    if (ith == 0 && numa_tensors && ggml_cpu_expert_prefetch_budget() > 0) {
        ggml_cpu_pf_record(src0->data, matrix_row_counts, (int) n_as, true);
        if (fuse_up) {
            ggml_cpu_pf_record(fuse_up->src[0]->data, matrix_row_counts, (int) n_as, false);
        }
    }

    // column split: this node owns rows [numa_row0_start, numa_row0_end) of every expert, and its workers share that range
    int64_t numa_row0_start = 0;
    int64_t numa_row0_end   = ne01;
    if (numa_tensors) {
        ggml_numa_tp_row_range_for_node(
                numa_node, ne01, GGML_NUMA_TP_PLAIN_ALIGN, &numa_row0_start, &numa_row0_end);
    }

    if (ith == 0 && ggml_exec_trace_observing()) {
        int active_experts = 0;
        int iqp_experts = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            const int64_t cne1 = matrix_row_counts[cur_a];
            if (cne1 > 0) {
                active_experts++;
                iqp_experts += iqp && ggml_cpu_iqp_mul_mat_id_min_batch(cne1);
            }
        }
        // IQP and IQK cover disjoint src0 types, so a node either runs IQP (possibly
        // mixed with vec_dot when an expert is below the batch threshold) or IQK.
        const bool iqk_node = !params->use_ref && ggml_cpu_iqk_mul_mat_enabled() &&
                ne13 == 1 && dst->type == GGML_TYPE_F32 &&
                (src1_cont || src1->type != vec_dot_type) &&
                iqk_mul_mat_moe_supported(ne00, src0->type, vec_dot_type);
        const enum ggml_exec_trace_dispatch_impl implementation = iqp_experts == 0 ?
                (iqk_node ? GGML_EXEC_TRACE_DISPATCH_CPU_IQK : GGML_EXEC_TRACE_DISPATCH_CPU_VEC_DOT) :
                iqp_experts == active_experts ?
                GGML_EXEC_TRACE_DISPATCH_CPU_IQP : GGML_EXEC_TRACE_DISPATCH_CPU_IQP_VEC_DOT;
        ggml_exec_trace_dispatch(dst->op, implementation,
                fuse_up ? GGML_EXEC_TRACE_DISPATCH_VARIANT_FUSED : GGML_EXEC_TRACE_DISPATCH_VARIANT_PER_EXPERT,
                nth, ((uint32_t) MIN(iqp_experts, UINT16_MAX) << 16) | (uint32_t) MIN(active_experts, UINT16_MAX));
    }

    const int numa_chunks = numa_tensors ? ggml_cpu_numa_chunks_per_thread() : 0;

    // pieces of another node can be taken only where its owners claim them from the counter: the per-row code below,
    // not the IQK paths (a static split; they need Q8_K activations), and not the experts that go to the IQ panel GEMM
    const bool steal_path = !(fuse_up && !g128_fused) &&
        (g128_fused || params->use_ref || vec_dot_type != GGML_TYPE_Q8_K || !ggml_cpu_iqk_mul_mat_enabled() ||
         ne13 != 1 || dst->type != GGML_TYPE_F32 || !(src1_cont || src1->type != vec_dot_type));

    for (int cur_a = 0; cur_a < n_as && numa_row0_start < numa_row0_end; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        if (node_local) {
            // tell the other nodes' workers that this expert is started (only the first worker here writes the line)
            atomic_int * frontier = &tp->numa_state[numa_node]->frontier;
            if (atomic_load_explicit(frontier, memory_order_relaxed) < cur_a) {
                atomic_store_explicit(frontier, cur_a, memory_order_relaxed);
            }
        }

        if (prof && work_ith == 0) {
            ggml_moe_prof_add_expert_rows(cne1);
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : src1_conv;

        if (iqp && ggml_cpu_iqp_mul_mat_id_min_batch(cne1)) {
            ggml_compute_forward_mul_mat_id_iqp(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0),
                                                iqp_panels, wdata,
                                                numa_row0_start, numa_row0_end, work_ith, work_nth);

            continue;
        }
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = numa_row0_end - numa_row0_start;
        const int64_t nr1 = cne1;

        if (fuse_up && !g128_fused) {
            // Both weight shards are offset by numa_row0_start exactly like the unfused
            // call, so a socket still reads only its own output-row range of gate and up.
            const struct ggml_tensor * up0 = fuse_up->src[0];
            const char * up0_cur = (const char *) up0->data + cur_a * up0->nb[2];
            const bool fused = iqk_moe_fused_up_gate(nr0, nr1, ne00, ne11,
                src0->type,
                src0_cur + numa_row0_start * nb01,
                up0_cur  + numa_row0_start * nb01, nb01,
                vec_dot_type, wdata, row_size,
                (float *) fuse_dst->data + numa_row0_start, fuse_dst->nb[1], fuse_dst->nb[2],
                matrix_rows + cur_a * ids->ne[0] * ids->ne[1],
                ggml_get_glu_op(fuse_dst) == GGML_GLU_OP_SWIGLU_CLAMP, fuse_limit, cur_a, work_ith, work_nth);
            // There is no fallback: the eligibility test in ggml_cpu_try_fuse_ops already
            // ran the same kernel checks, so a refusal here means they disagree.
            GGML_ASSERT(fused);
            continue;
        }

        // The IQK MoE kernel always computes rows [0, nr0) of the pointers it is
        // given, so the `--numa tensors` column split is expressed by offsetting
        // src0 and dst rather than by an explicit row range. Under --numa tensors
        // the remaining split must be over this node's own workers.
        if (!g128_fused && !params->use_ref && ggml_cpu_iqk_mul_mat_enabled() &&
            ne13 == 1 && dst->type == GGML_TYPE_F32 &&
            (src1_cont || src1->type != vec_dot_type) &&
            iqk_mul_mat_moe(nr0, nr1, ne00, ne11,
                src0->type, src0_cur + numa_row0_start * nb01, nb01,
                vec_dot_type, wdata, row_size,
                (float *) dst->data + numa_row0_start, nb1, nb2,
                matrix_rows + cur_a * ids->ne[0] * ids->ne[1], cur_a, work_ith, work_nth)) {
            continue;
        }

        // disable for NUMA, except for --numa tensors, where the counters and the rows belong to the node
        const bool disable_chunking = ggml_is_numa() && !numa_tensors;

        int64_t nchunk0, nchunk1, dr0, dr1;
        ggml_mmid_chunking(nr0, nr1, work_nth, numa_chunks, g128_fused, disable_chunking, &nchunk0, &nchunk1, &dr0, &dr1);

        int current_chunk = work_ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(node_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = numa_row0_start + dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, numa_row0_end);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_mmid_piece(dst, fuse_up, fuse_dst, g128_fused, cur_a, ir0_start, ir0_end, ir1_start, ir1_end,
                matrix_rows, row_size, src1_cont, wdata, g128_panels, g128_gemm ? gemm_panels : NULL, batch_rows);

            if (work_nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }

    // --numa tensors: this worker has no more pieces of its own node, so it helps the other nodes, nearest node first. It
    // takes spare pieces (those after the first piece of each owner) of the experts that the owners have not started:
    // then it works beside the owners, and a slow node (a core shared with a VM, a GPU's DMA, the OS) does not hold up the
    // op. The expert that the owners work on is left to them, as their next piece starts sooner than a remote one ends.
    // A piece is claimed from its owner's counter, so it is still computed once and by the same code: only its weight
    // rows come from the other node's memory, and the result is bitwise the same.
    if (node_local && steal_path && ggml_numa_steal_enabled()) {
        // the generation of this op's node barriers: a node that has another one has not yet reset its counters
        const int gen = atomic_load_explicit(&tp->numa_state[numa_node]->n_barrier_passed, memory_order_relaxed);

        const void * wdata = (src1->type == vec_dot_type) ? src1->data : src1_conv;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int k = 0; k + 1 < tp->numa_n_nodes; ++k) {
            const int victim = tp->numa_steal_order[numa_node][k];
            struct ggml_numa_node_state * vs = tp->numa_state[victim];
            if (atomic_load_explicit(&vs->n_barrier_passed, memory_order_acquire) != gen) {
                continue;
            }

            int64_t v_row0_start, v_row0_end;
            ggml_numa_tp_row_range_for_node(victim, ne01, GGML_NUMA_TP_PLAIN_ALIGN, &v_row0_start, &v_row0_end);
            if (v_row0_start >= v_row0_end) {
                continue;
            }

            const int v_nth = tp->numa_node_workers[victim];
            char (*v_chunk)[CACHE_LINE_SIZE] = (char (*)[CACHE_LINE_SIZE]) (vs->scratch + numa_layout.chunks);

            // from the last expert down to the one the owners work on
            for (int cur_a = n_as - 1; cur_a > atomic_load_explicit(&vs->frontier, memory_order_relaxed); --cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0 || (iqp && ggml_cpu_iqp_mul_mat_id_min_batch(cne1))) {
                    continue;
                }

                int64_t nchunk0, nchunk1, dr0, dr1;
                ggml_mmid_chunking(v_row0_end - v_row0_start, cne1, v_nth, numa_chunks, g128_fused, false,
                        &nchunk0, &nchunk1, &dr0, &dr1);
                if (v_nth >= nchunk0 * nchunk1) {
                    continue; // no spare pieces: one per owner
                }

                atomic_int * ctr = (atomic_int *) (v_chunk + cur_a);
                if (atomic_load_explicit(ctr, memory_order_relaxed) >= nchunk0 * nchunk1) {
                    continue; // other thieves took them
                }

                int chunk = atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
                while (chunk < nchunk0 * nchunk1) {
                    const int64_t ith0 = chunk % nchunk0;
                    const int64_t ith1 = chunk / nchunk0;

                    const int64_t ir0_start = v_row0_start + dr0 * ith0;
                    const int64_t ir0_end = MIN(ir0_start + dr0, v_row0_end);

                    const int64_t ir1_start = dr1 * ith1;
                    const int64_t ir1_end = MIN(ir1_start + dr1, cne1);

                    ggml_mmid_piece(dst, fuse_up, fuse_dst, g128_fused, cur_a, ir0_start, ir0_end, ir1_start, ir1_end,
                        matrix_rows, row_size, src1_cont, wdata, g128_panels, g128_gemm ? gemm_panels : NULL, batch_rows);

                    chunk = atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
                }
            }
        }
    }

    if (prof) {
        ggml_moe_prof_add(GGML_MOE_PROF_TOTAL, ggml_time_us() - prof_t_op);
    }
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    ggml_compute_forward_mul_mat_id_impl(params, dst, NULL, NULL, 0.0f);
}

// --numa tensors: make every node's scratch large enough for the largest routed-expert MUL_MAT_ID of the graph. Call only
// while no worker runs.
static void ggml_numa_reserve_scratch(struct ggml_threadpool * tp, const struct ggml_cgraph * cgraph) {
    if (tp->numa_n_nodes < 2 || !ggml_numa_node_local_enabled()) {
        return;
    }

    size_t size = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID && (node->src[0]->flags & GGML_TENSOR_FLAG_NUMA_EXPERT) != 0) {
            size = MAX(size, ggml_numa_mmid_layout_of(node).size);
        }
    }

    for (int node = 0; size > 0 && node < tp->numa_n_nodes; ++node) {
        ggml_numa_node_scratch_reserve(tp->numa_state[node], (uint32_t) node, size);
    }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } break;
        case GGML_OP_KPOOL_EXPAND:
            {
                ggml_compute_forward_kpool_expand(params, tensor);
            } break;
        case GGML_OP_INTERVAL_MASK:
            {
                ggml_compute_forward_interval_mask(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_MIX:
            {
                ggml_compute_forward_dsv4_hc_mix(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_ALLREDUCE:
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
// CPU this thread is pinned to under --numa tensors, -1 when not pinned
static _Thread_local int ggml_numa_pinned_cpu = -1;

static void set_numa_thread_affinity(const struct ggml_compute_state * state) {
    if (!ggml_is_numa()) {
        return;
    }

    const int thread_n = state->ith;
    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        case GGML_NUMA_STRATEGY_TENSORS: {
            // A worker keeps its CPU for the life of the threadpool, but this runs once per graph split.
            // The syscall costs more than the node split saves, so only call it when the CPU changes.
            if (ggml_numa_pinned_cpu == state->numa_cpu) {
                return;
            }
            cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
            CPU_ZERO_S(setsize, cpus);
            CPU_SET_S(state->numa_cpu, setsize, cpus);
            rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
            if (rv) {
                fprintf(stderr, "warning: failed to pin NUMA tensor worker %d to CPU %d: %s\n",
                        thread_n, state->numa_cpu, strerror(rv));
            } else {
                ggml_numa_pinned_cpu = state->numa_cpu;
            }
            CPU_FREE(cpus);
            return;
        }
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }
    // delphi: with --numa tensors the calling thread keeps its CPU between graphs. Pinning it again for each graph (one CPU
    // split) costs two syscalls and often a migration. GGML_NUMA_UNPIN_MAIN=1 unpins it after each graph as before
    static int unpin = -1;
    if (unpin < 0) {
        const char * env = getenv("GGML_NUMA_UNPIN_MAIN");
        unpin = env != NULL && atoi(env) != 0;
    }
    if (ggml_numa_is_tensor_mode() && !unpin) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        if (!ggml_numa_is_tensor_mode() || CPU_ISSET(i, &g_state.numa.cpuset)) {
            CPU_SET_S(i, setsize, cpus);
        }
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }
    ggml_numa_pinned_cpu = -1;

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(const struct ggml_compute_state * state) { UNUSED(state); }
static void clear_numa_thread_affinity(void) {}
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_DSV4_HC_MIX:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                // delphi: except for many rows: a gather from a lazily mapped table can fault on nearly every
                // row, and one thread then waits for each disk read in turn
                const int64_t n_rows = ggml_nelements(node->src[1]);
                n_tasks = n_rows >= 256 ? (int) MIN((int64_t) n_threads, n_rows) : 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_COL2IM_1D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_LIGHTNING_INDEXER:
        case GGML_OP_KPOOL_EXPAND:
        case GGML_OP_INTERVAL_MASK:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_ALLREDUCE:
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        // delphi: stay inside the allowed CPUs (ggml-cpuset.h)
        if (mask[i] && ggml_cpuset_allowed((int) i)) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

// --numa tensors: split the n_threads workers of the next graph over the nodes, in proportion to their CPUs. Workers
// 0..n_threads-1 are the ones that run, so a split made for the whole pool would leave the later nodes without workers
// and their rows of every expert without an owner. With fewer workers than nodes no worker gets a node rank, and
// MUL_MAT_ID uses the ordinary schedule. Call only while no worker runs.
static void ggml_numa_prepare_workers(struct ggml_threadpool * tp, int n_threads) {
#if defined(__gnu_linux__)
    if (!ggml_numa_is_tensor_mode() || tp->numa_cpus == NULL || tp->numa_graph_threads == n_threads) {
        return;
    }
    struct ggml_compute_state * workers = tp->workers;
    const uint32_t n_nodes = g_state.numa.n_nodes;
    tp->numa_graph_threads = n_threads;

    uint32_t node_workers[GGML_NUMA_MAX_NODES] = {0};
    const bool split = (uint32_t) n_threads >= n_nodes;
    if (split) {
        GGML_ASSERT(ggml_numa_partition_workers(tp->numa_node_cpus, n_nodes, n_threads, node_workers));
    }

    uint32_t ith = 0;
    for (uint32_t node = 0; node < n_nodes; ++node) {
        tp->numa_node_workers[node] = (int) node_workers[node];
        const uint32_t * cpus = tp->numa_cpus + (size_t) node*GGML_NUMA_MAX_CPUS;
        const uint32_t n_ranks = split ? node_workers[node] : (ith < (uint32_t) n_threads ? 1 : 0);
        for (uint32_t rank = 0; rank < n_ranks; ++rank, ++ith) {
            const uint32_t cpu = cpus[rank % tp->numa_node_cpus[node]];
            workers[ith].numa_node        = (int) node;
            workers[ith].numa_cpu         = (int) cpu;
            workers[ith].numa_rank        = split ? (int) rank : 0;
            workers[ith].numa_n_threads   = split ? (int) node_workers[node] : 0;
            workers[ith].numa_node_leader = split && rank == 0;
            memset(workers[ith].cpumask, 0, sizeof(workers[ith].cpumask));
            workers[ith].cpumask[cpu] = true;
        }
    }
#else
    UNUSED(tp);
    UNUSED(n_threads);
#endif
}

static void ggml_numa_assign_workers(
        const struct ggml_threadpool_params * tpp,
        struct ggml_compute_state * workers) {
    if (!ggml_numa_is_tensor_mode()) {
        return;
    }

#if defined(__gnu_linux__)
    const bool has_user_mask = ggml_thread_cpumask_is_valid(tpp->cpumask);
    uint32_t allowed_cpus[GGML_NUMA_MAX_NODES][GGML_NUMA_MAX_CPUS] = {{0}};
    uint32_t node_cpus[GGML_NUMA_MAX_NODES] = {0};

    for (uint32_t node = 0; node < g_state.numa.n_nodes; ++node) {
        const struct ggml_numa_node * topology = &g_state.numa.nodes[node];
        for (uint32_t i = 0; i < topology->n_cpus; ++i) {
            const uint32_t cpu = topology->cpus[i];
            if (!has_user_mask || (cpu < GGML_MAX_N_THREADS && tpp->cpumask[cpu])) {
                allowed_cpus[node][node_cpus[node]++] = cpu;
            }
        }
        if (node_cpus[node] == 0) {
            GGML_ABORT("--numa tensors CPU mask leaves NUMA node %u without a usable CPU", topology->id);
        }
    }

    // the pool keeps the CPU lists: a graph can run fewer workers than the pool has (cplan.n_threads is at most the
    // largest task count of its nodes, and a context can plan decode with fewer threads than its pool), and then
    // ggml_numa_prepare_workers splits that count over the nodes again
    struct ggml_threadpool * tp = workers[0].threadpool;
    const uint32_t n_nodes = g_state.numa.n_nodes;
    tp->numa_cpus = (uint32_t *) calloc((size_t) n_nodes*GGML_NUMA_MAX_CPUS, sizeof(uint32_t));
    GGML_ASSERT(tp->numa_cpus != NULL);
    for (uint32_t node = 0; node < n_nodes; ++node) {
        tp->numa_node_cpus[node] = node_cpus[node];
        memcpy(tp->numa_cpus + (size_t) node*GGML_NUMA_MAX_CPUS, allowed_cpus[node], node_cpus[node]*sizeof(uint32_t));
    }
    tp->numa_graph_threads = -1;
    ggml_numa_prepare_workers(tp, tpp->n_threads);

    uint32_t ith = 0;
    for (uint32_t node = 0; node < n_nodes && (uint32_t) tpp->n_threads >= n_nodes; ++node) {
        const uint32_t first = ith;
        GGML_LOG_INFO("numa tensors: node %u workers %u-%u CPUs",
                g_state.numa.nodes[node].id, first, first + tp->numa_node_workers[node] - 1);
        for (int rank = 0; rank < tp->numa_node_workers[node]; ++rank, ++ith) {
            GGML_LOG_CONT(" %d", workers[ith].numa_cpu);
        }
        GGML_LOG_CONT("\n");
    }

    // per-node state of MUL_MAT_ID, and the order in which a node takes pieces from the others: nearest first, and at
    // equal distance node+1, node+2, ... so that the nodes of one socket do not all go to the same node first
    tp->numa_n_nodes = (int) n_nodes;
    for (uint32_t node = 0; node < n_nodes; ++node) {
        tp->numa_state[node] = ggml_numa_node_state_alloc(node);
        if (!tp->numa_state[node]) {
            GGML_ABORT("--numa tensors: failed to allocate the state of NUMA node %u", g_state.numa.nodes[node].id);
        }
        int * order = tp->numa_steal_order[node];
        for (uint32_t step = 1; step < n_nodes; ++step) {
            const int other = (int) ((node + step) % n_nodes);
            int k = (int) step - 1;
            while (k > 0 && g_state.numa.distance[node][order[k - 1]] > g_state.numa.distance[node][other]) {
                order[k] = order[k - 1];
                --k;
            }
            order[k] = other;
        }
    }
    GGML_LOG_INFO("numa tensors: MUL_MAT_ID node-local scratch and barrier %s, piece stealing %s\n",
            ggml_numa_node_local_enabled() ? "on" : "off (GGML_NUMA_NODE_LOCAL=0)",
            !ggml_numa_node_local_enabled() ? "off (needs GGML_NUMA_NODE_LOCAL)" :
            ggml_numa_steal_enabled() ? "on" : "off (GGML_NUMA_STEAL=0)");
#else
    UNUSED(tpp);
    UNUSED(workers);
#endif
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    for (int node = 0; node < threadpool->numa_n_nodes; ++node) {
        ggml_numa_node_state_free(threadpool->numa_state[node]);
    }
    free(threadpool->numa_cpus);

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#if defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }

                        // the IQ panel path needs one scratch panel per thread past the q8_K rows
                        if (ggml_cpu_iqp_supports_mul_mat(node)) {
                            cur = GGML_PAD(cur, 64) + n_tasks * ggml_cpu_iqp_scratch_size(node);
                        }
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // --numa tensors gives every NUMA node a private src1 conversion and chunk counter block
                        const bool numa_tensors = ggml_numa_is_tensor_mode() &&
                            (src0->flags & GGML_TENSOR_FLAG_NUMA_EXPERT) != 0;
                        const size_t n_nodes = numa_tensors ? ggml_numa_get_node_count() : 1;
                        // src1
                        if (src1->type != vec_dot_type) {
                            const size_t src1_size = ggml_row_size(vec_dot_type, ggml_nelements(src1));
                            cur += numa_tensors
                                ? GGML_PAD(src1_size, sizeof(int64_t)) * n_nodes
                                : src1_size + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as*n_nodes + CACHE_LINE_SIZE;
                        // the IQ panel path needs one scratch panel per thread on top of that
                        if (ggml_cpu_iqp_supports_mul_mat_id(node)) {
                            cur += n_tasks * ggml_cpu_iqp_scratch_size(node) + 64;
                        }
                        // G128 activation panels, one set per thread
                        if (ggml_cpu_g128_rows_supported(src0)) {
                            cur += n_tasks * ggml_cpu_g128_panels_size(src0->ne[0]) + 64;
                        }
                        // G128 GEMM panels of every src1 row, one copy per node
                        if (ggml_cpu_g128_gemm_node(node)) {
                            cur += ggml_nrows(src1) * GGML_Q4_G128_GEMM_PANEL_SIZE(src1->ne[0]) * n_nodes + 64;
                        }
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type) ||
                            node->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SET_ROWS:
                    {
                        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In
                        const int64_t ne13 = node->src[1]->ne[3]; // Batch

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = ggml_get_op_params_i32(node, 0);
                        const int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                case GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        const int64_t ne10 = node->src[1]->ne[0];
                        cur += sizeof(float)*ne10*n_tasks;
                    } break;
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
static bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

bool ggml_backend_cpu_is_repack_tensor(const struct ggml_tensor * tensor);

// Whether the three nodes at `node_n` are a routed MoE gate+up+SwiGLU triple that the fused
// IQK kernel can compute in one pass. The two matmuls are told apart by identity from the
// GLU sources rather than by graph order, which the builder is free to choose.
static bool ggml_cpu_moe_fuse_up_gate_match(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        struct ggml_tensor ** gate_out,
        struct ggml_tensor ** up_out,
        struct ggml_tensor ** glu_out,
        float * limit_out) {

    const char * disable = getenv("GGML_CPU_MOE_FUSE_UP_GATE");
    if (disable && strcmp(disable, "0") == 0) {
        return false;
    }

    const enum ggml_op fuse_ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    const int outputs[] = { node_n + 2 };
    // ggml_can_fuse cannot be used: the two matmuls are siblings, not a chain.
    if (!ggml_can_fuse_subgraph(cgraph, node_n, 3, fuse_ops, outputs, 1)) {
        return false;
    }

    struct ggml_tensor * mm0 = cgraph->nodes[node_n + 0];
    struct ggml_tensor * mm1 = cgraph->nodes[node_n + 1];
    struct ggml_tensor * glu = cgraph->nodes[node_n + 2];

    // an empty node (no output rows in this ubatch) is skipped by the unfused path; the fused one must not run it either
    if (ggml_is_empty(mm0) || ggml_is_empty(mm1)) {
        return false;
    }

    if (ggml_get_op_params_i32(glu, 1) != 0 || !glu->src[1]) {
        return false;
    }

    struct ggml_tensor * gate = glu->src[0];
    struct ggml_tensor * up   = glu->src[1];
    if (!((gate == mm0 && up == mm1) || (gate == mm1 && up == mm0))) {
        return false;
    }

    const struct ggml_tensor * gate0 = gate->src[0];
    const struct ggml_tensor * up0   = up->src[0];

    // G128 experts fuse through the row kernels (ggml_compute_forward_mul_mat_id_impl), the others through IQK
    const bool g128 = ggml_cpu_g128_rows_supported(gate0) && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU;
    const bool clamp = ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP;
    if (!g128 && (!ggml_cpu_iqk_mul_mat_enabled() || (!clamp && ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU))) {
        return false;
    }
    if (ggml_mul_mat_id_is_partial(mm0) != ggml_mul_mat_id_is_partial(mm1)) {
        return false;
    }

    // MXFP4 experts live in the repack extra buffer, which has its own NUMA-aware
    // MUL_MAT_ID and runs from ggml_cpu_extra_compute_forward. Fusing would bypass it.
    if (ggml_backend_cpu_is_repack_tensor(gate0) || ggml_backend_cpu_is_repack_tensor(up0)) {
        return false;
    }

    if (gate->src[1] != up->src[1] || gate->src[2] != up->src[2] ||
        gate0->type != up0->type ||
        !ggml_are_same_shape(gate0, up0) ||
        gate0->nb[1] != up0->nb[1] || gate0->nb[2] != up0->nb[2] ||
        (gate0->flags & GGML_TENSOR_FLAG_NUMA_EXPERT) != (up0->flags & GGML_TENSOR_FLAG_NUMA_EXPERT)) {
        return false;
    }

    if (gate->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(gate, glu) || glu->nb[0] != sizeof(float) ||
        gate->src[1]->ne[3] != 1) {
        return false;
    }

    const enum ggml_type vec_dot_type = type_traits_cpu[gate0->type].vec_dot_type;
    if (!ggml_is_contiguous(gate->src[1]) && gate->src[1]->type == vec_dot_type) {
        return false;
    }
    if (!g128 && !iqk_mul_mat_moe_supported(gate0->ne[0], gate0->type, vec_dot_type)) {
        return false;
    }

    *gate_out  = gate;
    *up_out    = up;
    *glu_out   = glu;
    *limit_out = clamp ? ggml_get_op_params_f32(glu, 3) : 0.0f;
    return true;
}


static int ggml_cpu_try_fuse_ops(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params,
        const struct ggml_cplan * cplan) {

    if (ggml_cpu_disable_fusion || cplan->use_ref) {
        return 0;
    }

    struct ggml_tensor * node = cgraph->nodes[node_n];

    if (node->op == GGML_OP_MUL_MAT_ID) {
        // MUL_MAT_ID + MUL_MAT_ID + GLU fusion for CPU-routed MoE experts
        struct ggml_tensor * gate = NULL;
        struct ggml_tensor * up   = NULL;
        struct ggml_tensor * glu  = NULL;
        float limit = 0.0f;
        if (ggml_cpu_moe_fuse_up_gate_match(cgraph, node_n, &gate, &up, &glu, &limit)) {
            ggml_compute_forward_mul_mat_id_impl(params, gate, up, glu, limit);
            return 2;
        }
    }

    if (node->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        const enum ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse(cgraph, node_n, fuse_ops, 2)) {
            struct ggml_tensor * mul_node = cgraph->nodes[node_n + 1];
            const struct ggml_tensor * mul_w = (mul_node->src[0] == node)
                ? mul_node->src[1] : mul_node->src[0];
            if (node->src[0]->type  == GGML_TYPE_F32 &&
                mul_node->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == node->ne[0]   &&
                mul_w->nb[0]        == sizeof(float)) {

                ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);
                return 1;
            }
        }
    }

    return 0;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#else
    set_numa_thread_affinity(state);
#endif

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.numa_node  =*/ state->numa_node,
        /*.numa_rank  =*/ state->numa_rank,
        /*.numa_n_threads =*/ state->numa_n_threads,
        /*.numa_node_leader =*/ state->numa_node_leader,
        /*.use_ref    =*/ cplan->use_ref,
    };
    ggml_cpu_tls_params = &params;

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    const bool trace_nodes = state->ith == 0 && ggml_exec_trace_observing();
    const int32_t tl_call = tp->timeline_call;
    const struct ggml_tensor * tl_last = NULL; // the last computed node when it was not followed by a barrier
    int tl_last_n = 0, tl_last_fused = 0;
    int64_t tl_last_t0 = 0, tl_last_t1 = 0;
    if (tl_call >= 0) {
        ggml_cpu_timeline_enter(tl_call, state->ith);
    }
    uint64_t trace_cpu_start = 0;
    uint32_t trace_cpu_node = 0;
    uint32_t trace_cpu_node_count = 0;

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (trace_nodes) {
            ggml_exec_trace_set_node(node_n);
        }

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (trace_nodes) {
            trace_cpu_node = node_n;
            trace_cpu_node_count = 1;
            trace_cpu_start = ggml_exec_trace_cpu_node_begin();
        }

        const int64_t tl_t0 = tl_call >= 0 ? ggml_cpu_timeline_now() : 0;
        const int tl_node = node_n;

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            if (trace_nodes) {
                trace_cpu_node_count += n_fused;
            }
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        const int64_t tl_t1 = tl_call >= 0 ? ggml_cpu_timeline_now() : 0;
        if (tl_call >= 0) {
            tl_last = node; tl_last_n = tl_node; tl_last_fused = n_fused; tl_last_t0 = tl_t0; tl_last_t1 = tl_t1;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
            if (tl_call >= 0) {
                ggml_cpu_timeline_node(tl_call, state->ith, tl_node, n_fused, state->ith == 0 ? node : NULL,
                        tl_t0, tl_t1, ggml_cpu_timeline_now());
                tl_last = NULL;
            }
            if (trace_nodes) {
                ggml_exec_trace_cpu_node_end(trace_cpu_start, trace_cpu_node, trace_cpu_node_count);
                trace_cpu_start = 0;
            }
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

    if (tl_call >= 0) {
        if (tl_last) {
            ggml_cpu_timeline_node(tl_call, state->ith, tl_last_n, tl_last_fused, state->ith == 0 ? tl_last : NULL,
                    tl_last_t0, tl_last_t1, ggml_cpu_timeline_now());
        }
        ggml_cpu_timeline_exit(tl_call, state->ith);
    }

    if (trace_nodes && trace_cpu_start != 0) {
        ggml_exec_trace_cpu_node_end(trace_cpu_start, trace_cpu_node, trace_cpu_node_count);
    }

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
        threadpool->timeline_call    = -1;
        threadpool->numa_n_nodes     = 0;
        threadpool->numa_graph_threads = -1;
        threadpool->numa_cpus        = NULL;
        memset(threadpool->numa_state, 0, sizeof(threadpool->numa_state));
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
        workers[j].numa_cpu   = -1;
    }

    threadpool->workers = workers;
    ggml_numa_assign_workers(tpp, workers);

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        if (!ggml_numa_is_tensor_mode()) {
            ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
        }
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        if (!ggml_numa_is_tensor_mode()) {
            ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
        }

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    if (!ggml_numa_is_tensor_mode()) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);
    }

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }
    ggml_numa_reserve_scratch(threadpool, cgraph);
    threadpool->timeline_call = ggml_cpu_timeline_begin(cgraph, n_threads);

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
                ggml_numa_prepare_workers(threadpool, n_threads);
            }

            // Apply thread CPU mask and priority
            int ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            // --numa tensors pins in set_numa_thread_affinity() and caches it, so do not apply the mask again here
            if (!ggml_numa_is_tensor_mode() && ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_numa_prepare_workers(threadpool, 1);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    ggml_numa_prepare_workers(threadpool, n_threads);

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    if (threadpool->timeline_call >= 0) {
        ggml_cpu_timeline_end(threadpool->timeline_call);
        threadpool->timeline_call = -1;
    }

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme2(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            const char * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != NULL && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
