#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   rep_cells I32 [n_blocks, ns]       cell whose row holds the block's pooled key: the cell of its first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns], -inf for a block
    // that is not complete before the token's tail, so a top-k over blocks needs no attention mask
    //   tail      I32 [ratio, n_tokens/ns, ns] (blk_bias only) the token's own cell, then the cells
    //             of its incomplete tail, then -1
    // The ubatch pools n_wr blocks again and writes their keys into the pooled half of the rows (see qsa_full):
    //   wr_cells  I32 [ratio*n_wr, ns]     cells making up each pooled block, a block's own cell for a missing slot
    //   wr_pos    I32 [4*n_wr*ns]          mrope position rows of each pooled block
    //   wr_idx    I32 [n_wr*ns]            row the pooled key goes to, s*kv_size + cell; extra rows repeat the first
    // full: all complete blocks (n_wr = n_blocks), else the blocks that hold a token of the ubatch (n_wr = n_tokens/ns)
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * rep_cells,
                       ggml_tensor * wr_cells, ggml_tensor * wr_pos, ggml_tensor * wr_idx,
                       ggml_tensor * bias, ggml_tensor * tail, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool full, uint32_t stream_base) const;

    // delphi: a block's pooled key stays in its first cell's row, written by the ubatches that change the block. An
    // operation that changes blocks or positions outside a ubatch (copy, keep, shift, restore, or a removal that is not
    // the end of a single-sequence stream) sets this, and the next ubatch pools all blocks again.
    bool qsa_full() const { return qsa_dirty; }

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // several sequences can share a stream, so a removal can join the cells of two blocks into one
    const bool multi_seq_stream;

    // set_input_qsa clears it when a full ubatch runs
    mutable bool qsa_dirty = true;

    // set_input_qsa's buffers, kept between calls (a thread_local cost a __tls_get_addr call per access in the loops)
    struct qsa_work {
        std::vector<int32_t>  blk_of;
        std::vector<int32_t>  cell_grp;
        std::vector<int32_t>  grp_head;
        std::vector<int32_t>  grp_next;
        std::vector<int32_t>  grp_first;
        std::vector<int32_t>  grp_slot0;
        std::vector<uint64_t> grp_slots;
        std::vector<int32_t>  grp_bid;
        std::vector<int32_t>  grp_cells;
        std::vector<int32_t>  grp_pb;
        std::vector<int32_t>  bid_idx;
        std::vector<int32_t>  bid_cell;
        std::vector<int32_t>  bid_slot0;
        std::vector<int32_t>  bid_pos;
        std::vector<int32_t>  blk_cells_s;
        std::vector<int32_t>  wr_grp;
        std::vector<uint8_t>  grp_wr;
        std::vector<int32_t>  order;
        std::vector<int32_t>  rank;
        std::vector<int32_t>  rep_s;       // rep_cells of one stream

        // the groups above describe these cells, so a ubatch that only adds cells updates them (set_input_qsa)
        // every sequence operation, clear and state_read clear inc_valid
        bool                    inc_valid    = false;
        const llama_kv_cells *  inc_cells    = nullptr;
        llama_seq_id            inc_seq      = -1;
        int64_t                 inc_n_kv     = 0;
        int64_t                 inc_n_blocks = 0;
        int64_t                 inc_r        = 0;
        uint32_t                inc_used     = 0;
        int32_t                 inc_n_bid    = 0;
    };

    mutable qsa_work qsa_buf;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * rep_cells,
                       ggml_tensor * wr_cells, ggml_tensor * wr_pos, ggml_tensor * wr_idx,
                       ggml_tensor * bias, ggml_tensor * tail, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool full) const;

    // see llama_memory_hybrid_idx::qsa_full
    bool qsa_full() const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
