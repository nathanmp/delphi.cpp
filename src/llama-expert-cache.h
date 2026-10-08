#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include <memory>
#include <vector>

struct llama_model;

// Experimental static residency. Host experts remain authoritative; the banks never change during execution.
struct llama_expert_cache {
    struct entry {
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
        ggml_tensor * hot_ids = nullptr;
        ggml_tensor * cold_ids = nullptr;
    };

    static std::unique_ptr<llama_expert_cache> create(const llama_model & model, const std::vector<ggml_backend_ptr> & backends);
    const entry * layer(int il) const;
    ggml_backend_t backend() const { return gpu; }

private:
    ggml_backend_t gpu = nullptr;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    std::vector<entry> layers;
};
