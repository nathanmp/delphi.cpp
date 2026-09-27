#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Small-message AllReduce over direct peer-to-peer writes, for any number of GPUs in one process.
// Each rank pushes its data into every peer's receive buffer and signals through flags in the peer's
// memory, so one kernel per GPU does the whole reduction. Only available on HIP.
struct ggml_cuda_ar_p2p;

// Returns nullptr if peer access or fine-grained memory is unavailable.
ggml_cuda_ar_p2p * ggml_cuda_ar_p2p_init(const int * devices, size_t n_devices);

void ggml_cuda_ar_p2p_free(ggml_cuda_ar_p2p * p2p);

// In-place sum over tensors[0..n_devices-1]. Returns false without enqueueing anything if the
// tensors are not contiguous F32 of a supported size, so the caller can use another path.
bool ggml_cuda_ar_p2p_allreduce(ggml_cuda_ar_p2p * p2p, ggml_backend_t * backends, ggml_tensor ** tensors);
