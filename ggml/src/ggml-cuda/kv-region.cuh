#pragma once

#include <cstddef>

// delphi: a buffer whose address range mixes VRAM and pinned host memory (CUDA VMM).
// Kernels read and write the whole range with plain pointers; the host parts go over PCIe.

// mapping granularity of a region on this device, 0 if regions are not supported
size_t ggml_cuda_kv_region_granularity(int device);

// map [0, size) on the device; host_ranges holds n pairs (begin, end), sorted, granularity-aligned, that are backed
// by pinned host memory instead. Returns the region (nullptr on failure) and its base address in *base.
void * ggml_cuda_kv_region_create(int device, size_t size, const size_t * host_ranges, int n, void ** base);

void ggml_cuda_kv_region_free(void * region);
