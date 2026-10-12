#pragma once

#include <cstddef>

// Pinned host memory with its pages interleaved over the NUMA nodes that the process can use. Returns nullptr when the
// process can use fewer than 2 nodes or when the allocation, the interleave policy or the registration fails.
void * ggml_cuda_host_numa_alloc(size_t size);
void   ggml_cuda_host_numa_free(void * ptr, size_t size);
