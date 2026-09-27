#include "allreduce-p2p.cuh"

#include <cstdlib>

#if defined(GGML_USE_HIP)

// Two slots are enough: a rank can only start call N+2 after every peer arrived at call N+1, and a
// peer arrives at N+1 only after its kernel for call N (which reads slot N%2) has finished.
static constexpr int GGML_CUDA_AR_P2P_SLOTS       = 2;
static constexpr int GGML_CUDA_AR_P2P_MAX_DEVICES = 8;
static constexpr int GGML_CUDA_AR_P2P_BLOCKS      = 2;
// ints between flags, so every flag has its own cache line
static constexpr int GGML_CUDA_AR_P2P_FLAG_STRIDE = 16;
static constexpr size_t GGML_CUDA_AR_P2P_MAX_BYTES_DEFAULT = 256 * 1024;

struct ggml_cuda_ar_p2p {
    int    n_devices = 0;
    int    devices[GGML_CUDA_AR_P2P_MAX_DEVICES] = {};
    // receive buffers and flags live on their owning device, in fine-grained memory so that peer
    // writes become visible to a running kernel. rbuf: [slot][src rank][max_count]; flags: [src rank][block]
    float * rbuf [GGML_CUDA_AR_P2P_MAX_DEVICES] = {};
    int   * flags[GGML_CUDA_AR_P2P_MAX_DEVICES] = {};
    size_t max_count = 0;
    // tokens only increase, so flags never need to be reset
    int    token = 0;
};

struct ggml_cuda_ar_p2p_args {
    float * buf;
    float * peer_rbuf [GGML_CUDA_AR_P2P_MAX_DEVICES];
    int   * peer_flags[GGML_CUDA_AR_P2P_MAX_DEVICES];
    int     n_devices;
    int     rank;
    int     count4;
    size_t  max_count;
    int     slot;
    int     token;
};

static __global__ void ggml_cuda_ar_p2p_kernel(const ggml_cuda_ar_p2p_args a) {
    const int gtid = blockIdx.x*blockDim.x + threadIdx.x;
    const int gnt  = gridDim.x*blockDim.x;

    const size_t slot_offset = (size_t) a.slot*GGML_CUDA_AR_P2P_MAX_DEVICES*a.max_count;
    float4 * buf = (float4 *) a.buf;

    // Push this rank's data into every peer. Each block handles the same stripe on every rank, so it
    // only has to wait for the matching block of each peer.
    for (int p = 0; p < a.n_devices; ++p) {
        if (p == a.rank) {
            continue;
        }
        float4 * dst = (float4 *) (a.peer_rbuf[p] + slot_offset + (size_t) a.rank*a.max_count);
        for (int i = gtid; i < a.count4; i += gnt) {
            dst[i] = buf[i];
        }
    }
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int p = 0; p < a.n_devices; ++p) {
            if (p != a.rank) {
                int * flag = a.peer_flags[p] + (a.rank*GGML_CUDA_AR_P2P_BLOCKS + blockIdx.x)*GGML_CUDA_AR_P2P_FLAG_STRIDE;
                __hip_atomic_store(flag, a.token, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
        int * own_flags = a.peer_flags[a.rank];
        for (int p = 0; p < a.n_devices; ++p) {
            if (p == a.rank) {
                continue;
            }
            const int * flag = own_flags + (p*GGML_CUDA_AR_P2P_BLOCKS + blockIdx.x)*GGML_CUDA_AR_P2P_FLAG_STRIDE;
            while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < a.token) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();

    // Sum in rank order on every rank so that all copies of the result are bit-identical.
    const float * own_rbuf = a.peer_rbuf[a.rank] + slot_offset;
    for (int i = gtid; i < a.count4; i += gnt) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int p = 0; p < a.n_devices; ++p) {
            const float4 v = p == a.rank ? buf[i] : ((const float4 *) (own_rbuf + (size_t) p*a.max_count))[i];
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        buf[i] = sum;
    }
}

ggml_cuda_ar_p2p * ggml_cuda_ar_p2p_init(const int * devices, size_t n_devices) {
    const char * env = getenv("GGML_CUDA_AR_P2P_MAX_BYTES");
    const size_t max_bytes = env ? (size_t) atoll(env) : GGML_CUDA_AR_P2P_MAX_BYTES_DEFAULT;
    if (max_bytes == 0 || n_devices < 2 || n_devices > GGML_CUDA_AR_P2P_MAX_DEVICES) {
        return nullptr;
    }

    for (size_t i = 0; i < n_devices; ++i) {
        for (size_t j = 0; j < n_devices; ++j) {
            int can_access = 0;
            if (i != j && (cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]) != cudaSuccess || !can_access)) {
                (void) cudaGetLastError();
                GGML_LOG_INFO("%s: no peer access between devices %d and %d, P2P AllReduce disabled\n", __func__, devices[i], devices[j]);
                return nullptr;
            }
        }
    }

    auto * p2p = new ggml_cuda_ar_p2p;
    p2p->n_devices = (int) n_devices;
    p2p->max_count = max_bytes / sizeof(float);

    const size_t rbuf_size  = (size_t) GGML_CUDA_AR_P2P_SLOTS*GGML_CUDA_AR_P2P_MAX_DEVICES*p2p->max_count*sizeof(float);
    const size_t flags_size = (size_t) GGML_CUDA_AR_P2P_MAX_DEVICES*GGML_CUDA_AR_P2P_BLOCKS*GGML_CUDA_AR_P2P_FLAG_STRIDE*sizeof(int);

    bool ok = true;
    for (size_t i = 0; i < n_devices && ok; ++i) {
        p2p->devices[i] = devices[i];
        ggml_cuda_set_device(devices[i]);
        for (size_t j = 0; j < n_devices; ++j) {
            if (i != j) {
                const cudaError_t err = cudaDeviceEnablePeerAccess(devices[j], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    ok = false;
                }
                (void) cudaGetLastError();
            }
        }
        ok = ok && hipExtMallocWithFlags((void **) &p2p->rbuf[i],  rbuf_size,  hipDeviceMallocFinegrained) == hipSuccess;
        ok = ok && hipExtMallocWithFlags((void **) &p2p->flags[i], flags_size, hipDeviceMallocFinegrained) == hipSuccess;
        ok = ok && cudaMemset(p2p->flags[i], 0, flags_size) == cudaSuccess;
    }
    if (!ok) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: failed to set up P2P AllReduce buffers, P2P AllReduce disabled\n", __func__);
        ggml_cuda_ar_p2p_free(p2p);
        return nullptr;
    }
    GGML_LOG_INFO("%s: P2P AllReduce enabled for %zu devices, up to %zu bytes\n", __func__, n_devices, max_bytes);
    return p2p;
}

void ggml_cuda_ar_p2p_free(ggml_cuda_ar_p2p * p2p) {
    if (p2p == nullptr) {
        return;
    }
    for (int i = 0; i < GGML_CUDA_AR_P2P_MAX_DEVICES; ++i) {
        if (p2p->rbuf[i]) {
            ggml_cuda_set_device(p2p->devices[i]);
            CUDA_CHECK(cudaFree(p2p->rbuf[i]));
        }
        if (p2p->flags[i]) {
            ggml_cuda_set_device(p2p->devices[i]);
            CUDA_CHECK(cudaFree(p2p->flags[i]));
        }
    }
    delete p2p;
}

bool ggml_cuda_ar_p2p_allreduce(ggml_cuda_ar_p2p * p2p, ggml_backend_t * backends, ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne % 4 != 0 || (size_t) ne > p2p->max_count) {
        return false;
    }
    for (int i = 0; i < p2p->n_devices; ++i) {
        const ggml_tensor * t = tensors[i];
        if (t == nullptr || t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne ||
                !ggml_is_contiguously_allocated(t) || ((uintptr_t) t->data & 0xF) != 0) {
            return false;
        }
    }
    if (ne == 0) {
        return true;
    }

    p2p->token++;
    ggml_cuda_ar_p2p_args args = {};
    args.n_devices = p2p->n_devices;
    args.count4    = (int) (ne / 4);
    args.max_count = p2p->max_count;
    args.slot      = p2p->token % GGML_CUDA_AR_P2P_SLOTS;
    args.token     = p2p->token;
    for (int i = 0; i < p2p->n_devices; ++i) {
        args.peer_rbuf[i]  = p2p->rbuf[i];
        args.peer_flags[i] = p2p->flags[i];
    }

    for (int i = 0; i < p2p->n_devices; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[i]->context;
        ggml_cuda_set_device(cuda_ctx->device);
        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            // this rank holds an empty slice and contributes zeros
            CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
        }
        args.buf  = (float *) tensors[i]->data;
        args.rank = i;
        ggml_cuda_ar_p2p_kernel<<<GGML_CUDA_AR_P2P_BLOCKS, 256, 0, cuda_ctx->stream()>>>(args);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

#else

ggml_cuda_ar_p2p * ggml_cuda_ar_p2p_init(const int * devices, size_t n_devices) {
    GGML_UNUSED(devices);
    GGML_UNUSED(n_devices);
    return nullptr;
}

void ggml_cuda_ar_p2p_free(ggml_cuda_ar_p2p * p2p) {
    GGML_UNUSED(p2p);
}

bool ggml_cuda_ar_p2p_allreduce(ggml_cuda_ar_p2p * p2p, ggml_backend_t * backends, ggml_tensor ** tensors) {
    GGML_UNUSED(p2p);
    GGML_UNUSED(backends);
    GGML_UNUSED(tensors);
    return false;
}

#endif // defined(GGML_USE_HIP)
