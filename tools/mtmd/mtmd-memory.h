#pragma once

#include "mtmd.h"

// delphi: LLAMA_MMPROJ_PHASE=1 lets the server release idle language-model compute before a vision phase.
// Weights stay in RAM and transient encoder GPU allocations are released before the embeddings are consumed. No expert-cache storage is borrowed.
// Returns the phase GPU, or nullptr when phase vision is disabled.
MTMD_API ggml_backend_dev_t mtmd_phase_device(const mtmd_context * ctx);
