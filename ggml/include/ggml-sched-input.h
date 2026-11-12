#ifndef GGML_SCHED_INPUT_H
#define GGML_SCHED_INPUT_H

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// delphi: allow host graph inputs to upload on the destination backend's stream. The caller must keep their data
// unchanged and allocated until ggml_backend_sched_wait_graph() or ggml_backend_sched_synchronize() completes.
// Set before graph allocation; disabled by default. Device inputs and backends without asynchronous uploads retain the regular copy path.
GGML_API void ggml_backend_sched_set_input_async(ggml_backend_sched_t sched, bool enable);

// Copy host graph inputs into scheduler-owned pinned buffers before asynchronous upload. The caller can reuse its
// input data when graph_compute_async returns. Backends without pinned buffers or upload events keep synchronous copies.
GGML_API void ggml_backend_sched_set_input_staging(ggml_backend_sched_t sched, bool enable);

#ifdef __cplusplus
}
#endif

#endif
