#pragma once

#include "ggml.h"
#include "ggml-cpp.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

struct llama_context;
struct llama_memory_context_i;
struct llm_graph_result;
class llama_kv_cache_dsv4_context;

struct llama_dsv41_prefill_tensor {
    ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, 4> ne = {};
    std::vector<uint8_t> data;

    ggml_tensor * input(ggml_context * ctx, llm_graph_result * res, const char * name) const;
    void store(ggml_tensor * tensor);
};

struct llama_dsv41_prefill_chunk {
    // HC stream and pending mix after the last layer that ran, in the device state buffer. Each encoder graph reads and overwrites them.
    ggml_tensor * hc = nullptr;
    ggml_tensor * mix = nullptr;
    // shared index selections, in host memory: only the index source layers write them
    llama_dsv41_prefill_tensor top_k;
    llama_dsv41_prefill_tensor cand;
};

// One partial graph. Output tensors belong to that graph, not to the host checkpoint.
struct llama_dsv41_prefill_graph {
    int first = 0;
    int last = 0;
    llama_dsv41_prefill_chunk * chunk = nullptr;
    ggml_tensor * top_k = nullptr;
    ggml_tensor * cand = nullptr;
};

class llama_dsv41_prefill_window {
public:
    static std::unique_ptr<llama_dsv41_prefill_window> create(llama_context & ctx, llama_memory_context_i * mctx);
    ~llama_dsv41_prefill_window();

    // Device memory for the HC state of a full window, before the expert cache takes the free memory.
    static void reserve(llama_context & ctx);

    ggml_status encode();
    void decoder(size_t chunk);

private:
    static bool context_eligible(const llama_context & ctx, int mode);
    llama_dsv41_prefill_window(llama_context & ctx, llama_kv_cache_dsv4_context & mctx, bool retain);

    llama_context & ctx;
    llama_kv_cache_dsv4_context & mctx;
    const bool retain;
    std::vector<llama_dsv41_prefill_chunk> chunks;
    ggml_context_ptr state_ctx;
    llama_dsv41_prefill_graph graph;
};
