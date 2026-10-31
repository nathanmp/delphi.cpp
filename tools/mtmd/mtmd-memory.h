#pragma once

#include "mtmd.h"

// delphi: dynamic vision (llama-server --mmproj-memory dynamic, the default) loads the encoder to the GPU for each encoding.
// Weights stay in RAM and transient encoder GPU allocations are released before the embeddings are consumed. No expert-cache storage is borrowed.
MTMD_API mtmd_context * mtmd_init_from_file_dynamic(const char * mmproj_fname, const llama_model * text_model,
        mtmd_context_params ctx_params, bool dynamic);

// Returns the GPU of dynamic vision, or nullptr when the encoder is resident (or on the CPU).
MTMD_API ggml_backend_dev_t mtmd_dynamic_device(const mtmd_context * ctx);
