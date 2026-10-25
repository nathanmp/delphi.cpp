#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama-expert-device.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

struct llama_model;

// Experimental GPU-hot / CPU-cold routed experts. Host experts remain authoritative. The slots live in tiers: device buffers
// on any GPU, each with one slot arena per combination of gate/up/down expert types (a group). Tier 0 is the static budget
// (LLAMA_MOE_CACHE_MIB) on the layer GPU; more tiers can be added and dropped at run time (VRAM sharing between servers), which
// changes the graphs (generation()). Within a tier every layer gets the same number of slots, and a membership change rewrites
// slots and maps but never changes a graph. Membership is rebuilt at the first small batch after a prompt, from the routing of
// the most recent rows, and while a response decodes: every `every` small batches a worker copies the experts that entered the
// window's top set to a device scratch on its own stream, and a later batch boundary packs them into the slots of the experts
// that left it. A prompt that streams experts (GGML_SCHED_STAGE=1) borrows the top of the largest tier on each staging GPU for
// the staging ring; the refresh after the prompt fills it again.
struct llama_expert_cache {
    // Batches with more rows are prompt batches: they use the original routed experts (and only record their routing), unless
    // they have at most graph_rows() rows.
    static constexpr int64_t max_rows = 16;
    // Most rows of a batch that runs the cache graph. LLAMA_MOE_CACHE_PROMPT_ROWS raises it above max_rows, so that prompt
    // batches the CPU runs (below GGML_SCHED_STAGE_MIN_TOKENS) also run their hot experts on the GPU beside the CPU part.
    int64_t graph_rows() const { return std::max(max_rows, prompt_rows); }
    bool prefill_enabled(int64_t rows) const { return rows >= prefill_min_rows && rows <= prefill_max_rows; }
    bool prefill_staging(int64_t rows, int il) const;
    ggml_tensor * prefill_bank(int il, int p) const;
    ggml_tensor * prefill_route(int il) const;
    // The plan starts selected uploads. The second CPU op waits, or preserves a split boundary for the submission worker.
    ggml_tensor * build_prefill_ready(ggml_context * ctx, ggml_tensor * plan, int il);
    bool prefill_overlap() const;
    static bool prefill_worker_ready(const ggml_cgraph * graph, bool ask, void * userdata);
    // run-time tiers; each adds a partial expert pass per layer to the graph
    static constexpr int max_tiers = 8;

    struct entry {
        llama_expert_cache * owner = nullptr;
        int il = -1;
        int group = -1;
        std::vector<int32_t> cold;       // expert -> expert, or -1 when resident
        std::vector<int32_t> slot;       // expert -> global slot, or -1
        std::vector<uint32_t> count;     // selections in the window
        std::vector<float> prior;        // profile selections per row
        std::vector<uint16_t> ring;      // window rows, top_k ids each
        size_t head = 0;
        int64_t rows = 0;
        std::vector<float> heat;         // selections of every recorded row (halved now and then), for the static tiers
        int64_t heat_rows = 0;
    };

    static std::unique_ptr<llama_expert_cache> create(const llama_model & model, const std::vector<ggml_backend_ptr> & backends,
            bool allow_prefill_overlap, const llama_expert_device_policy & device_policy = {});
    // LLAMA_MOE_CACHE_MIB unset or "auto" (and no LLAMA_MOE_CACHE_SHARED): the cache is on when the model allows it, and its
    // tiers are sized from the free memory of the GPUs at the first decode (llama_context::expert_cache_auto). An unsupported
    // model then gets no cache instead of an error.
    static bool auto_requested();
    bool auto_size = false; // tiers not added yet
    ~llama_expert_cache();

    const entry * layer(int il) const;
    // true if any tier holds slots: the graph then uses the cache for small batches
    bool active() const { return !tiers.empty(); }
    // the GPU of the first cached layer (prefill sharing and the LLAMA_MOE_CACHE_MIB tier use it)
    ggml_backend_t backend() const { return gpu; }
    // the GPU of layer il, where its tiers are local
    ggml_backend_t backend(int il) const { return il >= 0 && il < (int) layer_gpu.size() && layer_gpu[il] ? layer_gpu[il] : gpu; }
    // a tier on b serves a layer on another GPU (its splits then need a scheduler worker)
    bool serves_remote(ggml_backend_t b) const;
    // a cached layer is on dev
    bool is_home(ggml_backend_dev_t dev) const;
    bool auto_device(ggml_backend_dev_t dev) const;

    int n_tiers() const { return (int) tiers.size(); }
    ggml_backend_t tier_backend(int t) const { return tiers[t].backend; }
    // GPU I32 [1, n_expert] of tier t for layer il: expert -> slot in the tier's bank, or -1. Null if the tier has no slots for
    // the layer's group. Pinned tiers also serve layers outside their device's domain.
    ggml_tensor * hot_ids(int t, int il) const;
    // gate (0), up (1) or down (2) bank of tier t for the layer's group
    ggml_tensor * bank(int t, int il, int p) const { return tiers[t].groups[layers[il].group].banks[p]; }
    // bumped whenever the tiers change: graphs built before must not be reused
    uint64_t generation() const { return gen; }

    // VRAM sharing: add a tier of up to `bytes` on `backend`, filled at the next refresh. Returns its id, or -1 if the size gives
    // no slot to every layer or the allocation fails.
    int add_tier(ggml_backend_t backend, size_t bytes);
    // Drop tier `id` and free its memory. The caller has synchronized and reclaimed the staging slots of every scheduler.
    // Returns the bytes freed (0 if there is no such tier).
    size_t drop_tier(int id);
    // ids, devices and sizes of the tiers
    std::vector<std::pair<int, ggml_backend_t>> tier_list() const;
    size_t tier_bytes(int id) const;
    // free the decode-time refresh buffers (allocated again when needed); returns the device bytes freed
    size_t release_scratch();
    // the caller reclaimed the staging slots outside a refresh: the lent slots are free again
    void reclaimed();

    // CPU op: records the routed ids (F32 [top_k, n_tokens]) unless warmup, and returns I32 ids with resident experts set to -1.
    ggml_tensor * build_cold_ids(ggml_context * ctx, ggml_tensor * ids, int il);
    // CPU op: records the routed ids (I32 [top_k, n_tokens]) of a batch that does not use the arena. Null if not recorded.
    ggml_tensor * build_record(ggml_context * ctx, ggml_tensor * ids, int il);
    // CPU op: the sum over the expert slots of experts (F32 [n_embd, top_k, n_tokens]) times weights (F32 [1, top_k, n_tokens]),
    // F32 [n_embd, n_tokens]. The same sums in the same order as ggml_mul and a chain of ggml_add, in one op (one thread barrier
    // instead of top_k)
    static ggml_tensor * build_weighted_sum(ggml_context * ctx, ggml_tensor * experts, ggml_tensor * weights);

    // Call before each decoded batch while no graph is running. True at the first small batch after a prompt (or after a tier
    // was added): then the caller synchronizes, reclaims lent staging slots from every scheduler, and calls refresh(). Other
    // small batches may start or apply a decode-time refresh here (applying one synchronizes the GPUs).
    bool needs_refresh(int64_t n_tokens, bool is_warmup);
    void refresh();

    // ggml_backend_sched_stage_lend_t: evict the experts in the top of the largest tier on `backend` and lend it as staging slots
    static bool lend(void * user_data, ggml_backend_t backend, int n_slots, size_t slot_size, void ** slots);

private:
    struct upload_job { int il, e, s; }; // s: global slot
    // Layers with one combination of expert types, in one tier. Its slots are the global slots [base, base + n); local slot =
    // global - base.
    struct group {
        ggml_tensor * banks[3] = {};
        int64_t base = 0, n = 0;
        int64_t lent_lo = 0; // local slots from lent_lo up are lent to the staging ring
    };
    struct tier {
        int id = -1;
        ggml_backend_t backend = nullptr;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        std::vector<group> groups;          // per type group
        std::vector<ggml_tensor *> hot_ids; // per layer, null for layers without routed experts
        bool fixed = false;                 // static tier (LLAMA_MOE_CACHE_STATIC): the hottest experts by heat
        bool pinned = false;                // populated members cannot be replaced or lent to staging
    };
    struct candidate { double score; int il, e; };
    struct async_refresh;
    double window_score(const entry & l, int e) const;
    double heat_score(const entry & l, int e) const;
    // The membership change toward the top sets: the static tiers take the experts of most heat, the other tiers the best
    // window scores of the rest. Entering experts, best first, take free slots and then the slots of members that left the
    // set, weakest first. At most max_jobs uploads, of which at most max_swaps replace static members. jobs[i] evicts
    // evict[i] (or (-1, -1)); an expert that moves into a static tier leaves its slot in the other tier.
    void plan(int64_t max_jobs, int64_t max_swaps, std::vector<upload_job> & jobs, std::vector<std::pair<int, int>> & evict) const;
    // apply one planned job to the maps (the slot data is written apart)
    void apply_job(const upload_job & j, std::pair<int, int> evicted, std::vector<bool> & changed);
    bool slot_usable(int64_t s) const;
    // slot pools: one per group, or with local_only one per group and GPU
    int pool_of_slot(int64_t s) const;
    int pool_of_layer(int il) const;
    int n_pools() const { return (int) group_types.size() * (local_only ? (int) homes.size() + 1 : 1); }
    void sync_homes() const;
    size_t max_slab() const;
    bool any_lent() const;
    // tier index and group of a global slot
    std::pair<int, int> locate(int64_t s) const { return where[s]; }
    void renumber();
    bool make_tier(tier & t, ggml_backend_t backend, size_t bytes);
    void tick();
    void cancel_async();
    void record(entry & l, const char * data, size_t nb0, size_t nb1, int64_t k, int64_t rows, bool is_f32);
    void upload(const std::vector<upload_job> & jobs);
    void push_maps(const std::vector<bool> & changed);
    void forget(const tier & t) const;
    static void op_cold_ids(ggml_tensor * dst, int ith, int nth, void * userdata);
    static void op_record(ggml_tensor * dst, int ith, int nth, void * userdata);
    static void op_prefill_plan(ggml_tensor * dst, int ith, int nth, void * userdata);
    static void op_prefill_ready(ggml_tensor * dst, int ith, int nth, void * userdata);
    static void op_weighted_sum(ggml_tensor * dst, int ith, int nth, void * userdata);

    struct prefill_share;
    std::unique_ptr<prefill_share> prefill;
    int64_t prefill_min_rows = 512; // LLAMA_MOE_PREFILL_MIN_ROWS
    int64_t prefill_max_rows = 0; // LLAMA_MOE_PREFILL_SHARE, LLAMA_MOE_PREFILL_MAX_ROWS
    void init_prefill(uint32_t min_reuse, bool gpu_only, bool allow_overlap);

    const llama_model * model = nullptr;
    llama_expert_device_policy device_policy;
    ggml_backend_t gpu = nullptr;
    std::vector<ggml_backend_t> layer_gpu; // per layer, null for layers that are not cached
    std::vector<ggml_backend_t> homes;     // the GPUs of the cached layers
    bool local_only = false;               // LLAMA_MOE_CACHE_LOCAL=1: a layer takes slots in the tiers on its GPU only
    std::vector<tier> tiers;
    int next_tier_id = 0;
    uint64_t gen = 0;
    std::vector<entry> layers;
    // type groups (shared by all tiers)
    const ggml_tensor * shape[3] = {};
    std::vector<std::array<ggml_type, 3>> group_types;
    std::vector<int64_t> group_layers;
    std::vector<size_t> group_bytes;
    size_t all_layer_bytes = 0;
    int n_expert = 0;
    std::vector<std::pair<int, int>> owner_of; // global slot -> (layer, expert), or (-1, -1)
    std::vector<std::pair<int, int>> where;    // global slot -> (tier index, group)
    void (*set_expert)(ggml_backend_t, const ggml_tensor *, int64_t, const void *) = nullptr;
    void (*forget_inside)(ggml_backend_t, const ggml_tensor *) = nullptr;
    ggml_backend_buffer_ptr pinned; // refill bounce buffers
    std::vector<int> cpus;          // the CPUs next to the GPU, for the refill threads

    int64_t window = 256;      // rows; 0 keeps every row, negative never refreshes
    bool record_prompts = true;
    int64_t prompt_rows = 0;   // LLAMA_MOE_CACHE_PROMPT_ROWS, see graph_rows()
    int64_t prompt_hit_sel = 0, prompt_all_sel = 0;
    double hysteresis = 1.0;   // score bonus of a resident expert
    int64_t static_swaps = 8;  // LLAMA_MOE_CACHE_STATIC_SWAPS: static members replaced per prompt
    int64_t refresh_max = -1;  // LLAMA_MOE_CACHE_REFRESH_MAX: uploads of the refresh after a prompt (-1: all); the rest
                               // fill free slots at the next small batches
    bool fill_idle = false;    // a planned fill found nothing for the free slots: wait for the next `every`
    int64_t every = 8;         // small batches between decode-time refreshes; 0 refreshes only after prompts
    int64_t max_async = 32;    // experts per decode-time refresh (device scratch and pinned buffer: 3 slabs each)
    int64_t n_small = 0;
    std::unique_ptr<async_refresh> async;
    int64_t n_async = 0, n_async_uploaded = 0;
    double async_apply_ms = 0.0;
    bool pending = false;      // a prompt was processed (or a tier added) since the last refresh
    bool warmup = false;       // warmup rows are not routing evidence

    // Decode hit statistics since the last refresh: routed selections and distinct experts per (ubatch, layer).
    int64_t hit_sel = 0, all_sel = 0, hit_distinct = 0, all_distinct = 0;
    std::vector<uint32_t> stamp; // expert -> last (ubatch, layer) that counted it
    uint32_t stamp_gen = 0;
    int64_t n_refresh = 0, n_uploaded = 0;
    double refresh_ms = 0.0;
    int64_t upload_copy_us = 0, upload_wait_us = 0;
    // LLAMA_MOE_CACHE_DUMP=<file>: every routed row, batch start and refresh, for offline policy replay
    // (tools/expert-cache/replay.py).
    FILE * dump = nullptr;
    uint32_t batch_seq = 0;
    void dump_record(int il, int kind, int64_t rows, int64_t k, const char * data, size_t nb0, size_t nb1, bool is_f32);
};
