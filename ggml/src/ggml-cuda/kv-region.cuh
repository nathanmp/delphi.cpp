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

// delphi: the granularity of ggml_cuda_kv_region_swap, 0 if the region was not mapped one granule per allocation
size_t ggml_cuda_kv_region_swap_granularity(void * region);

// swap the memory behind the granule-aligned, disjoint ranges [a, a + len) and [b, b + len) of the region (offsets from
// its base). keep_a / keep_b: copy the data, so that the range keeps its contents. Synchronizes the device.
bool ggml_cuda_kv_region_swap(void * region, size_t a, size_t b, size_t len, bool keep_a, bool keep_b);
