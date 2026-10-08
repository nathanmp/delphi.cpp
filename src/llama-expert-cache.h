#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;

// Experimental GPU-hot / CPU-cold routed experts. Host experts remain authoritative. All layers share one slot arena, so a
// membership change rewrites slots and maps but never changes a graph. Membership is fixed while a response decodes; it is
// rebuilt at the first small batch after a prompt, from the routing of the most recent rows.
struct llama_expert_cache {
    // Batches with more rows use the original routed experts (and only record their routing).
    static constexpr int64_t max_rows = 16;

    struct entry {
        llama_expert_cache * owner = nullptr;
        int il = -1;
        ggml_tensor * hot_ids = nullptr; // GPU I32 [1, n_expert]: expert -> arena slot, or -1
        std::vector<int32_t> cold;       // expert -> expert, or -1 when resident
        std::vector<int32_t> slot;       // expert -> arena slot, or -1
        std::vector<uint32_t> count;     // selections in the window
        std::vector<float> prior;        // profile selections per row
        std::vector<uint16_t> ring;      // window rows, top_k ids each
        size_t head = 0;
        int64_t rows = 0;
    };

    static std::unique_ptr<llama_expert_cache> create(const llama_model & model, const std::vector<ggml_backend_ptr> & backends);
    ~llama_expert_cache();

    const entry * layer(int il) const;
    ggml_backend_t backend() const { return gpu; }
    ggml_tensor * gate() const { return banks[0]; }
    ggml_tensor * up() const { return banks[1]; }
    ggml_tensor * down() const { return banks[2]; }

    // CPU op: records the routed ids (F32 [top_k, n_tokens]) unless warmup, and returns I32 ids with resident experts set to -1.
    ggml_tensor * build_cold_ids(ggml_context * ctx, ggml_tensor * ids, int il);
    // CPU op: records the routed ids (I32 [top_k, n_tokens]) of a batch that does not use the arena. Null if not recorded.
    ggml_tensor * build_record(ggml_context * ctx, ggml_tensor * ids, int il);

    // Call before each decoded batch while no graph is running. Rebuilds membership at the first small batch after a prompt.
    void before_batch(ggml_backend_sched_t sched, int64_t n_tokens, bool is_warmup);

private:
    void record(entry & l, const char * data, size_t nb0, size_t nb1, int64_t k, int64_t rows, bool is_f32);
    void refresh();
    void upload(int il, int e, int s);
    static void op_cold_ids(ggml_tensor * dst, int ith, int nth, void * userdata);
    static void op_record(ggml_tensor * dst, int ith, int nth, void * userdata);

    const llama_model * model = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * banks[3] = {};
    std::vector<entry> layers;
    std::vector<std::pair<int, int>> owner_of; // arena slot -> (layer, expert), or (-1, -1)
    bool (*set_expert)(const ggml_tensor *, int64_t, const void *) = nullptr;

    int64_t window = 2048;     // rows; 0 keeps every row, negative never refreshes
    bool record_prompts = true;
    double hysteresis = 0.25;  // score bonus of a resident expert
    bool pending = false;      // a prompt was processed since the last refresh
    bool warmup = false;       // warmup rows are not routing evidence

    // Decode hit statistics since the last refresh: routed selections and distinct experts per (ubatch, layer).
    int64_t hit_sel = 0, all_sel = 0, hit_distinct = 0, all_distinct = 0;
    std::vector<uint32_t> stamp; // expert -> last (ubatch, layer) that counted it
    uint32_t stamp_gen = 0;
    int64_t n_refresh = 0, n_uploaded = 0;
    double refresh_ms = 0.0;
};
