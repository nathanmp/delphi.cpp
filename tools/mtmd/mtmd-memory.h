#pragma once

#include "mtmd.h"

// delphi: the server defaults to phase vision when its expert cache uses the vision GPU. LLAMA_MMPROJ_PHASE=0/1 overrides it.
// Weights stay in RAM and transient encoder GPU allocations are released before the embeddings are consumed. No expert-cache storage is borrowed.
MTMD_API mtmd_context * mtmd_init_from_file_phase(const char * mmproj_fname, const llama_model * text_model,
        mtmd_context_params ctx_params, bool phase_default);

// Returns the phase GPU, or nullptr when phase vision is disabled.
MTMD_API ggml_backend_dev_t mtmd_phase_device(const mtmd_context * ctx);
