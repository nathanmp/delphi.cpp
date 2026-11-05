#pragma once

#include "ggml.h"
#include "ggml-cpp.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

struct llama_context;
class llama_dsv41_prefill_window;
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
    // shared index selections: only the index source layers write them. In the device state buffer when it has room, else in host memory.
    ggml_tensor * top_k_dev = nullptr;
    ggml_tensor * cand_dev = nullptr;
    llama_dsv41_prefill_tensor top_k;
    llama_dsv41_prefill_tensor cand;
    // attention inputs (masks, cache indices), the same for every layer of the chunk
    std::vector<std::vector<uint8_t>> inputs;
};

// One partial graph. Output tensors belong to that graph, not to the host checkpoint.
struct llama_dsv41_prefill_graph {
    int first = 0;
    int last = 0;
    llama_dsv41_prefill_chunk * chunk = nullptr;
    llama_dsv41_prefill_window * window = nullptr;
    // outputs for the host checkpoint
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
    // Set the decoder graph of window chunk `chunk`. A ubatch after the window runs the full graph.
    void decoder(size_t chunk);
    bool covers(size_t chunk) const { return chunk < chunks.size(); }

    // `slot`, or a new device state tensor with the type and shape of `like`; nullptr if the state buffer has no room
    ggml_tensor * persist(ggml_tensor *& slot, const ggml_tensor * like);

private:
    static bool context_eligible(const llama_context & ctx, int mode);
    llama_dsv41_prefill_window(llama_context & ctx, llama_kv_cache_dsv4_context & mctx, size_t n_chunks, bool retain);

    llama_context & ctx;
    llama_kv_cache_dsv4_context & mctx;
    const bool retain;
    std::vector<llama_dsv41_prefill_chunk> chunks;
    ggml_context_ptr state_ctx;
    char * state_next = nullptr; // free part of the state buffer
    char * state_end = nullptr;
    llama_dsv41_prefill_graph graph;
};
