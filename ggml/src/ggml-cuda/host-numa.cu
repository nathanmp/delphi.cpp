#include "host-numa.cuh"
#include "common.cuh"

#include <cerrno>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__linux__)
static size_t ggml_cuda_host_numa_len(size_t size) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    return (size + page - 1) / page * page;
}
#endif

void * ggml_cuda_host_numa_alloc(size_t size) {
#if defined(__linux__)
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    // the nodes that the process can allocate on (cpuset), 1024 bits as the kernel needs a mask of all possible nodes
    unsigned long mask[16] = {};
    const unsigned long maxnode = 8*sizeof(mask);
    if (syscall(SYS_get_mempolicy, nullptr, mask, maxnode, nullptr, MPOL_F_MEMS_ALLOWED) != 0) {
        return nullptr;
    }
    int n_nodes = 0;
    for (unsigned long word : mask) {
        n_nodes += __builtin_popcountl(word);
    }
    if (n_nodes < 2) {
        return nullptr;
    }

    const size_t len = ggml_cuda_host_numa_len(size);
    void * ptr = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        return nullptr;
    }
    // The policy is set before any page exists, so the registration below faults the pages in on all nodes in turn.
    if (syscall(SYS_mbind, ptr, len, MPOL_INTERLEAVE, mask, maxnode, 0) != 0) {
        GGML_LOG_DEBUG("%s: mbind of %.2f MiB failed: %s\n", __func__, len/1024.0/1024.0, strerror(errno));
        munmap(ptr, len);
        return nullptr;
    }
    const cudaError_t err = cudaHostRegister(ptr, len, cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_DEBUG("%s: cannot pin %.2f MiB: %s\n", __func__, len/1024.0/1024.0, cudaGetErrorString(err));
        munmap(ptr, len);
        return nullptr;
    }
    return ptr;
#else
    GGML_UNUSED(size);
    return nullptr;
#endif
}

void ggml_cuda_host_numa_free(void * ptr, size_t size) {
#if defined(__linux__)
    CUDA_CHECK(cudaHostUnregister(ptr));
    munmap(ptr, ggml_cuda_host_numa_len(size));
#else
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
#endif
}
