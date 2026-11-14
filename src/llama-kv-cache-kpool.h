#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-graph.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

struct llama_ubatch;
class llama_kv_cache;
class llama_kv_cache_context;
class llama_model;

// Persistent F32 storage for completed GLM-5-Next pool keys. Pool identities are
// tracked host-side, while the key tensors live beside their model layers.
class llama_kpool_cache {
public:
    struct pool_id {
        llama_seq_id seq;
        int64_t      block;

        bool operator==(const pool_id & other) const {
            return seq == other.seq && block == other.block;
        }
    };

    struct stream_plan {
        std::vector<int32_t> slots;
        std::vector<uint8_t> cached;
        std::vector<int32_t> scratch;
    };

    using layer_filter_cb = std::function<bool(uint32_t)>;

    llama_kpool_cache(
            const llama_model & model,
            bool                offload,
            bool                unified,
            uint32_t            kv_size,
            uint32_t            n_seq_max,
            uint32_t            kpool,
            uint32_t            n_embd,
            const layer_filter_cb & filter);

    ~llama_kpool_cache();

    void clear(bool data);
    void invalidate();
    // delphi: invalidate the pools of one sequence only; with one stream (unified) or seq_id < 0, all pools
    void invalidate(llama_seq_id seq_id);
    void seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);

    bool needs_rebuild(uint32_t stream0, uint32_t n_stream) const;
    void finish_rebuild(uint32_t stream0, uint32_t n_stream);

    void set_mtp_index_reuse(bool reuse);
    bool get_mtp_index_reuse() const;
    void mark_dirty(uint32_t stream, llama_seq_id seq_id, llama_pos pos);

    stream_plan prepare_stream(
            uint32_t stream,
            const std::vector<pool_id> & ids,
            size_t n_scratch,
            bool rebuild);

    void mark_cached(uint32_t stream, const std::vector<int32_t> & slots);

    uint32_t get_max_slots() const;
    uint32_t get_n_stream() const;
    uint32_t get_max_uncached(uint32_t stream0, uint32_t n_stream) const;

    ggml_tensor * get(
            ggml_context * ctx,
            int32_t il,
            uint32_t stream0,
            uint32_t n_stream) const;

    ggml_tensor * store(
            ggml_context * ctx,
            ggml_tensor * cur,
            ggml_tensor * idxs,
            int32_t il,
            uint32_t stream0,
            uint32_t n_stream) const;

    ggml_tensor * get_mtp_selection(
            ggml_context * ctx,
            int32_t il,
            int64_t n_selected,
            uint32_t stream0,
            uint32_t n_stream,
            bool mask) const;

    ggml_tensor * store_mtp_selection(
            ggml_context * ctx,
            ggml_tensor * cur,
            int32_t il,
            uint32_t stream0,
            uint32_t n_stream,
            bool mask) const;

    ggml_tensor * get_topk_hint(
            ggml_context * ctx,
            int32_t il,
            int64_t n_pools,
            uint32_t stream0,
            uint32_t n_stream) const;

    ggml_tensor * store_topk_hint(
            ggml_context * ctx,
            ggml_tensor * cur,
            int32_t il,
            uint32_t stream0,
            uint32_t n_stream) const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // delphi: ring slots. Without a unified KV cache a stream holds one sequence, and the slot of its pool block b is
    // b % max_slots: the live blocks of a sequence span at most max_slots, so no map is needed, and when the packed
    // run starts at a multiple of max_slots (b_base == 0 in practice) the slot of packed pool p is p itself, which
    // lets the indexer read the keys in place. ring_blocks()[slot] is the block that slot holds (-1: none), and
    // ring_cached()[slot] says whether its key is current.
    bool      is_ring() const;
    int64_t * ring_blocks(uint32_t stream);
    uint8_t * ring_cached(uint32_t stream);
    // slots >= ring_hi(stream) hold no block
    uint32_t & ring_hi(uint32_t stream);
    // the slot of a block: (block - origin) mod max_slots; a rebase empties the stream's ring (and rebuilds it) with
    // a new origin, so that the packed ordinals of the stream's run are its slots again
    int64_t  ring_origin(uint32_t stream) const;
    uint32_t ring_slot(uint32_t stream, int64_t block) const;
    void     ring_rebase(uint32_t stream, int64_t origin);

    // delphi: device-resident pool tables (ring mode only), one per buffer type that holds pool keys, so a decode
    // step uploads the rows that changed instead of two n_kv-sized tables per device (GLM-5.3 at 250K: ~1.5 MiB per
    // device per step over PCIe, with the devices idle). The shadows mirror what the device holds.
    struct dev_table {
        ggml_tensor * cells = nullptr;       // I32 [kpool*max_slots, n_stream], the members of packed pool p
        ggml_tensor * iv    = nullptr;       // F32 [max_slots, n_stream, 2]: p if complete else BIG; p
        std::vector<int32_t> shadow_cells;   // [n_stream][kpool*max_slots]
        std::vector<uint8_t> shadow_valid;   // [n_stream][max_slots]
    };
    size_t      n_tables() const;
    dev_table & get_table(size_t i);
    // the table on the device of layer il's pool keys, -1 if none
    int32_t     table_of_layer(int32_t il) const;

    // the first n_pools slots of layer il's keys [n_embd, n_pools, n_stream], written by a copy of cur
    ggml_tensor * store_prefix(
            ggml_context * ctx,
            ggml_tensor * cur,
            int32_t il,
            uint32_t stream0,
            uint32_t n_stream) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// GLM-5-Next indexer pooling. the position -> cell map is built host side because
// find_slot's cell order is arbitrary. no input may hold a negative index: ggml_set_rows
// asserts i1 >= 0 and ggml_get_rows has no sentinel, so unusable entries are clamped into
// range and neutralised by the additive masks instead.

// pool slots for `n_kv` cells shared by `n_seqs` sequences: n_kv/kpool, exact only while
// the sequences' cells are disjoint, plus 2 per sequence for rebasing.
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs = 1);

// select_k of modular_glm5_next.py, Glm5NextTextIndexer.forward. must run over POOLS, not
// cells: relu ties span pool boundaries, so a cell-level cut takes partial pools.
uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool);

// Mode 0 uses dense attention, mode 1 selects by KV and batch size, and mode 2 forces indexed attention.
bool llama_kpool_indexed_attn_enabled(int64_t n_kv, int64_t n_tps);

// Build the dense sparse-attention masks on the device (ggml_interval_mask) from
// per-cell and per-query scalars instead of uploading two O(n_kv*n_tps) host masks
// to every device, and the same for the pool bias of the indexer. LLAMA_GLM5_DEVICE_MASKS=0 restores the
// host masks.
bool llama_kpool_device_masks_enabled();

// `kv` must be the ATTENTION (MLA) cache; the indexer cache shares its slot layout.
//   cell_pool  I32 [n_kv, n_stream]                 per-cell view, optional, unused here
//   pool_cells I32 [kpool*n_pools, n_stream]        pool member -> cell, 0 if not resident
//   bias       F32 [n_kv, n_tps, n_stream]          per-cell view, optional, unused here
//   pool_bias  F32 [n_pools, n_tps, n_stream]       pool_valid & pool_visible, -INFINITY
//       outside the query's own sequence run; computed, not gathered from `bias` at the
//       last member, which an incomplete pool lacks and would inherit cell 0's validity
//   sel_mask   F16/F32 [n_kv, n_batch, 1, n_stream] 0.0f on the always-selected tail only
//   cand_mask  F16/F32 [n_kv, n_batch, 1, n_stream] max(bias, sel_mask); bounds the top-k
//       spills that a partial seq_rm would otherwise let escape the candidate set
//   pool_iv_cells, pool_iv_queries: the scalars of the device-built pool_bias (llm_graph_input_kpool)
void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
        const llama_ubatch   * ubatch,
              uint32_t         kpool,
              uint32_t         n_kv,
              ggml_tensor    * compact_tail_cells = nullptr,
              ggml_tensor    * compact_tail_mask  = nullptr,
              llama_kpool_cache * pool_cache      = nullptr,
              ggml_tensor    * pool_cache_slots   = nullptr,
              ggml_tensor    * pool_store_src     = nullptr,
              ggml_tensor    * pool_store_dst     = nullptr,
              ggml_tensor    * pool_update_cells  = nullptr,
              ggml_tensor    * pool_update_dst    = nullptr,
              bool             rebuild_pool_cache = false,
              uint32_t         stream0            = 0,
              ggml_tensor    * mask_cells         = nullptr,
              ggml_tensor    * mask_queries       = nullptr,
              ggml_tensor    * pool_iv_cells      = nullptr,
              ggml_tensor    * pool_iv_queries    = nullptr);

// delphi: device tables, see llm_graph_input_kpool::fast. LLAMA_GLM5_DEVICE_TABLES=0 keeps the host tables.
bool llama_kpool_device_tables_enabled();

class llm_graph_input_kpool;

// Compute the pool tables of a device-table ubatch and the rows of each device table that differ (into inp.fast_*).
// Read-only on the caches: graph build and can_reuse call it, set_input commits. n_update and n_delta are the shapes
// the graph needs (with floors, so that a reservation covers the real ubatches).
// the static conditions of the device-table path (the same for reservations and real ubatches, so that their graphs have
// the same topology)
bool llama_kpool_fast_eligible(
        const llama_kpool_cache * cache,
        const llama_ubatch      & ubatch,
        int64_t                   n_stream,
        int64_t                   n_pools,
        bool                      attn_indexed_or_compact,
        bool                      mtp_reuse);

// rebase the ring of every stream of the ubatch whose packed run no longer starts at its origin (a removed prefix):
// that stream rebuilds its pools. Call before the rebuild decision of the graph.
void llama_kpool_fast_rebase(
        llama_kpool_cache    * cache,
        const llama_kv_cache * kv,
        const llama_ubatch   & ubatch,
        uint32_t               stream0,
        int64_t                n_stream,
        int64_t                n_pools,
        uint32_t               kpool);

bool llama_kpool_fast_prepare(
        llm_graph_input_kpool & inp,
        const llama_kv_cache  * kv,
        llama_kpool_cache     * cache,
        const llama_ubatch    & ubatch,
        uint32_t                stream0,
        int64_t                 n_kv,
        int64_t                 n_stream,
        int64_t                 n_pools,
        bool                    rebuild,
        int64_t               & n_update,
        int64_t               & n_delta);

// One pooling map per ubatch; rebuilding it per indexer layer costs O(n_kv * n_tokens)
// host writes and dominates prefill. sharing is valid only while every indexer layer sees
// the same candidate set - true for glm5next (indexer_types all "full"), not for windowed.
class llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(
            const llama_kv_cache_context * mctx_attn,
            const llama_kv_cache_context * mctx_idx,
            llama_kpool_cache * pool_cache,
            bool rebuild_pool_cache,
            bool mtp_index_reuse,
            uint32_t stream0,
            uint32_t kpool) :
        mctx_attn(mctx_attn),
        mctx_idx(mctx_idx),
        pool_cache(pool_cache),
        rebuild_pool_cache(rebuild_pool_cache),
        mtp_index_reuse(mtp_index_reuse),
        stream0(stream0),
        kpool(kpool) {}

    ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * k_idxs     = nullptr;   // I32 [n_tokens]
    ggml_tensor * pool_cells = nullptr;   // I32 [kpool*n_pools, n_stream]
    ggml_tensor * pool_bias  = nullptr;   // F32 [n_pools, n_tps, n_stream]

    // exact, since pool_bias only holds 0.0f or -INFINITY. nullptr if the fused path is off
    ggml_tensor * pool_bias_f16 = nullptr; // F16 [n_pools, n_tps, 1, n_stream]

    ggml_tensor * sel_mask   = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]
    ggml_tensor * cand_mask  = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]

    // Device-built replacement for sel_mask/cand_mask (one sequence per stream).
    // Cell rows: pos, pos, (complete pool ? BIG : pos); BIG for cells outside the
    // stream's sequence. Query rows: q, tail_start. With ggml_interval_mask,
    // rows (0, 1) give sel (tail_start <= pos <= q) and rows (1, 2) give cand
    // (pos <= q && (pooled || pos >= tail_start)), which implies the causal KQ mask.
    ggml_tensor * mask_cells   = nullptr; // F32 [n_kv, 3, 1, n_stream]
    ggml_tensor * mask_queries = nullptr; // F32 [n_tps, 2, 1, n_stream]

    // graph-build cache: one (sel, cand) pair per device that runs sparse attention
    std::map<ggml_backend_dev_t, std::pair<ggml_tensor *, ggml_tensor *>> dev_masks;

    // Device-built replacement for pool_bias and pool_bias_f16. The host bias is O(n_pools*n_tps) per pipeline
    // copy (GLM 5.3 at -c 500000 -ub 1024: 8 x 489 MiB on the first GPU). Pool rows: the slot of a complete
    // pool of a sequence run, BIG otherwise. Query rows: the last slot that the query sees in its own run, the
    // first slot of that run. ggml_interval_mask gives pool_valid & pool_visible from them.
    ggml_tensor * pool_iv_cells   = nullptr; // F32 [n_pools, 2, 1, n_stream]
    ggml_tensor * pool_iv_queries = nullptr; // F32 [n_tps, 2, 1, n_stream]

    // graph-build cache: the pool bias of each device that scores pools (F16 for the fused indexer, else F32)
    std::map<ggml_backend_dev_t, ggml_tensor *> dev_pool_bias;

    // The direct compact-attention suffix: the real incomplete tail followed by
    // masked cell-zero padding up to the CUDA FA stride. Shared by every layer.
    ggml_tensor * compact_tail_cells = nullptr; // I32 [n_tail_pad, n_tps, n_stream]
    ggml_tensor * compact_tail_mask  = nullptr; // F16 [n_tail_pad, n_tps, n_stream]

    ggml_tensor * pool_cache_slots  = nullptr; // I32 [n_pools, n_stream]
    ggml_tensor * pool_store_src    = nullptr; // I32 [n_pools, n_stream], rebuild only
    ggml_tensor * pool_store_dst    = nullptr; // I32 [n_pools, n_stream], rebuild only
    ggml_tensor * pool_update_cells = nullptr; // I32 [kpool*n_update, n_stream], incremental
    ggml_tensor * pool_update_dst   = nullptr; // I32 [n_update, n_stream], incremental

    // delphi: device tables (LLAMA_GLM5_DEVICE_TABLES, default on). Ring-mode pool cache, one sequence per stream,
    // packed run at slot 0, device masks, indexed or compact attention. Then pool_cells, pool_iv_cells and
    // pool_cache_slots are null: the pool members and the validity rows live on each device (llama_kpool_cache::
    // dev_table) and get the rows that changed by set_rows, and the pool keys are read in place (slot = ordinal).
    struct fast_stream {
        llama_seq_id seq = -1;
        int64_t b_base = 0;
        int64_t n_run  = 0;
        bool    aligned = true;       // b_base is the ring origin of the stream
        std::vector<int32_t> cells;   // [kpool*n_pools], the pool_cells row of the legacy path
        std::vector<uint8_t> valid;   // [n_pools]
        std::vector<int32_t> filled;  // [n_pools], scratch
    };
    struct dev_update {
        int32_t table = -1;
        ggml_tensor * idx   = nullptr; // I32 [n_delta]: stream*max_slots + p
        ggml_tensor * cells = nullptr; // F32 [kpool, n_delta]: the I32 bits of the members
        ggml_tensor * iv    = nullptr; // F32 [1, n_delta]: row 0 of the validity rows
        ggml_tensor * cells_out = nullptr; // I32 [kpool*n_pools, n_stream], contiguous per stream
        ggml_tensor * iv_out    = nullptr; // F32 [n_pools, 2, 1, n_stream]
    };
    bool    fast         = false;
    int64_t fast_n_pools = 0;
    int64_t fast_n_delta = 0; // rows of each delta input
    int64_t fast_n_need  = 0; // pools whose key this step must compute, max over the streams
    std::vector<fast_stream> fast_streams;
    std::vector<std::vector<int32_t>> fast_dirty; // per table: the rows (stream*max_slots + p) that differ from its shadow
    std::vector<dev_update> dev_updates;

    int64_t get_n_pools() const;

    // the per-device views of the tables, built once per graph and device; the set_rows that apply the deltas are
    // expanded into gf right away so that set_input can commit the shadows
    ggml_tensor * get_pool_cells   (ggml_context * ctx, ggml_cgraph * gf, int32_t il);
    ggml_tensor * get_pool_iv_cells(ggml_context * ctx, ggml_cgraph * gf, int32_t il);

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;
    llama_kpool_cache * pool_cache;

    const bool rebuild_pool_cache;
    const bool mtp_index_reuse;
    const uint32_t stream0;
    const uint32_t kpool;
};
