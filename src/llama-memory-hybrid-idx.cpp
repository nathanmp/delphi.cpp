#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a row holds the raw key, then the pooled key of the block that starts at the cell (set_input_qsa)
        hparams_idx.n_embd_head_k_full = 2*model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = 2*model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = 2*model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()),
    multi_seq_stream(unified && n_seq_max > 1) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    qsa_buf.inc_valid = false;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);

        // any removal changes the groups, also one that leaves the pooled keys valid
        qsa_buf.inc_valid = false;

        // cutting the end of a sequence leaves the blocks before the cut, and the ubatches that fill the cut block pool it
        // again; with other sequences in the stream, a cell that loses the sequence can join a block of another one
        if ((p1 >= 0 && p1 != std::numeric_limits<llama_pos>::max()) || seq_id < 0 || multi_seq_stream) {
            qsa_dirty = true;
        }
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        qsa_dirty = true;
        qsa_buf.inc_valid = false;
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        qsa_dirty = true;
        qsa_buf.inc_valid = false;
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
        qsa_dirty = true;
        qsa_buf.inc_valid = false;
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        qsa_dirty = true;
        qsa_buf.inc_valid = false;
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    // restored rows hold the pooled keys of the writer's blocks, which need not be the blocks here
    qsa_dirty = true;
    qsa_buf.inc_valid = false;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    qsa_buf.inc_valid = false;

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * rep_cells,
        ggml_tensor * wr_cells,
        ggml_tensor * wr_pos,
        ggml_tensor * wr_idx,
        ggml_tensor * bias,
        ggml_tensor * tail,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        bool full,
        uint32_t stream_base) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    // the block top-k does not read cell_blk, so the graph prunes it and leaves it unallocated
    GGML_ASSERT(cell_blk->buffer == nullptr || ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = rep_cells->ne[0];
    const int64_t n_wr     = wr_idx->ne[0]/n_ns;
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;
    const int64_t kv_size  = get_mem_idx()->get_size();

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = cell_blk->buffer ? (int32_t *) cell_blk->data : nullptr;
    // only the top-k over blocks reads blk_cells, so the graph can prune it as well
    GGML_ASSERT(blk_cells->buffer == nullptr || ggml_backend_buffer_is_host(blk_cells->buffer));
    int32_t * dst_blk_cells = blk_cells->buffer ? (int32_t *) blk_cells->data : nullptr;
    int32_t * dst_rep       = (int32_t *) rep_cells->data;
    int32_t * dst_wr_cells  = (int32_t *) wr_cells->data;
    int32_t * dst_wr_pos    = (int32_t *) wr_pos->data;
    int32_t * dst_wr_idx    = (int32_t *) wr_idx->data;
    float   * dst_bias      = (float   *) bias->data;

    GGML_ASSERT(n_wr == (full ? n_blocks : n_tokens/n_ns));

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // this runs per ubatch and is O(n_kv) per stream. Built from empty vectors, the groups took 2.5 ms of a 3.7 ms call at
    // 100k context (Qwen 3.8 decode on a CMP, GPU idle meanwhile), so the buffers keep their capacity between calls
    qsa_work & w = qsa_buf;

    std::vector<int32_t> &  blk_of      = w.blk_of;
    std::vector<int32_t> &  cell_grp    = w.cell_grp;
    std::vector<int32_t> &  grp_head    = w.grp_head;
    std::vector<int32_t> &  grp_next    = w.grp_next;
    std::vector<int32_t> &  grp_first   = w.grp_first;
    std::vector<int32_t> &  grp_slot0   = w.grp_slot0;
    std::vector<uint64_t> & grp_slots   = w.grp_slots;
    std::vector<int32_t> &  grp_bid     = w.grp_bid;
    std::vector<int32_t> &  grp_cells   = w.grp_cells;
    std::vector<int32_t> &  grp_pb      = w.grp_pb;
    std::vector<int32_t> &  bid_idx     = w.bid_idx;
    std::vector<int32_t> &  bid_cell    = w.bid_cell;
    std::vector<int32_t> &  bid_slot0   = w.bid_slot0;
    std::vector<int32_t> &  bid_pos     = w.bid_pos;    // 4 mrope rows per block
    std::vector<int32_t> &  blk_cells_s = w.blk_cells_s; // blk_cells of one stream
    std::vector<int32_t> &  wr_grp      = w.wr_grp;     // groups pooled by this ubatch
    std::vector<uint8_t> &  grp_wr      = w.grp_wr;
    std::vector<int32_t> &  order       = w.order;
    std::vector<int32_t> &  rank        = w.rank;
    std::vector<int32_t> &  rep_s       = w.rep_s;      // rep_cells of one stream

    // a 64-bit division per cell took 1.4 ms of the scan at 100k cells (EPYC 7532); ratios are powers of two in practice
    int r_shift = 0;
    while ((int64_t(1) << r_shift) < r) {
        r_shift++;
    }
    const bool r_pow2 = (int64_t(1) << r_shift) == r;

    // blk_of (block of each cell) is read only by the per-cell bias and by cell_blk
    const bool need_blk_of = !blk_bias || dst_cell_blk != nullptr;

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk ? dst_cell_blk + s*n_kv : nullptr;

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // the groups of one stream, one sequence and plain positions can carry over to the next ubatch (inc_valid)
        const bool inc_config = n_ns == 1 && one_seq && blk_bias && !need_blk_of && r_pow2;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            // only blk_of reads cell_grp
            if (need_blk_of) {
                cell_grp.assign(n_kv, -1);
            }
            grp_head.assign(n_blocks, -1);

            oor = false;
            dup = false;

            // one sequence and plain positions: group g is the block at index bucket g, so no search and no growth
            if (one_seq && !ranked) {
                grp_next .assign(n_blocks, -1);
                grp_first.assign(n_blocks, -1);
                grp_slot0.assign(n_blocks, -1);
                grp_slots.assign(n_blocks, 0);
                grp_bid  .assign(n_blocks, -1);
                grp_cells.assign(n_blocks*r, -1);
                grp_pb   .resize(n_blocks);

                for (int64_t pb = 0; pb < n_blocks; ++pb) {
                    grp_pb[pb] = (int32_t) pb;
                }

                // raw pointers and local flags: through the closure, each cell reloaded every data pointer
                // (0.55 ms per call at 100k cells)
                int32_t  * GGML_RESTRICT p_cell_grp = need_blk_of ? cell_grp.data() : nullptr;
                int32_t  * GGML_RESTRICT p_first    = grp_first.data();
                int32_t  * GGML_RESTRICT p_slot0    = grp_slot0.data();
                uint64_t * GGML_RESTRICT p_slots    = grp_slots.data();
                int32_t  * GGML_RESTRICT p_cells    = grp_cells.data();

                bool l_oor = false;
                bool l_dup = false;

                const int64_t l_r      = r;
                const int64_t l_blocks = n_blocks;
                const int     l_shift  = r_shift;
                const bool    l_pow2   = r_pow2;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j)) {
                        continue;
                    }

                    const int64_t idx  = cells.pos_get(j);
                    const int64_t pb   = l_pow2 ? idx >> l_shift : idx/l_r;
                    const int64_t slot = l_pow2 ? idx & (l_r - 1) : idx%l_r;

                    if (pb >= l_blocks) {
                        l_oor = true;
                        continue;
                    }

                    if (p_first[pb] < 0) {
                        p_first[pb] = (int32_t) j;
                    }

                    const uint64_t bit = uint64_t(1) << slot;

                    l_dup |= (p_slots[pb] & bit) != 0;

                    if (p_cell_grp) {
                        p_cell_grp[j] = (int32_t) pb;
                    }
                    p_slots[pb] |= bit;

                    p_cells[pb*l_r + slot] = (int32_t) j;

                    if (slot == 0) {
                        p_slot0[pb] = (int32_t) j;
                    }
                }

                for (int64_t pb = 0; pb < n_blocks; ++pb) {
                    if (p_first[pb] >= 0) {
                        grp_head[pb] = (int32_t) pb;
                    }
                }

                oor = l_oor;
                dup = l_dup;

                return;
            }

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();
            grp_cells.clear();
            grp_pb   .clear();

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);
                    grp_cells.insert(grp_cells.end(), r, -1);
                    grp_pb   .push_back((int32_t) pb);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                grp_cells[g*r + idx%r] = (int32_t) j;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        // the mrope rows of a group: its block start, or with ranked cells the position of its first token
        auto group_pos = [&](int32_t g, int32_t * sec_pos) {
            const int32_t start = grp_pb[g]*(int32_t) r;

            std::fill(sec_pos, sec_pos + 4, start);

            if (ranked) {
                const int32_t   c = grp_slot0[g] >= 0 ? grp_slot0[g] : grp_first[g];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }
        };

        int32_t n_bid = 0;

        // a block takes the next block id when its last slot fills; padding blocks read the pooled row of block 0
        auto add_bid = [&](int32_t g, int64_t pb) {
            GGML_ASSERT(n_bid < n_blocks);

            grp_bid[g] = n_bid;

            bid_idx  [n_bid] = (int32_t) (pb*r);
            bid_cell [n_bid] = grp_first[g];
            bid_slot0[n_bid] = grp_slot0[g];

            group_pos(g, bid_pos.data() + 4*n_bid);

            const int32_t * src = grp_cells.data() + g*r;
            int32_t *       dst = blk_cells_s.data() + n_bid*r;
            for (int64_t k = 0; k < r; ++k) {
                dst[k] = src[k];
            }

            n_bid++;
        };

        // a ubatch that only adds cells of the sequence: each token fills one slot of its group, in O(1).
        // At 100k cells the full scan below took 0.6 ms of every decode step (Qwen 3.8 on a CMP, GPU idle meanwhile).
        // Any other change (a removal, an overwritten cell, a duplicate position, a block that completes out of order)
        // returns false, and the full scan rebuilds the groups.
        auto add_cells = [&]() -> bool {
            n_bid = w.inc_n_bid;

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const llama_pos q = ubatch->pos[s*n_tps + ii];
                const int32_t   j = q < 0 ? -1 : cells.seq_pos_cell(seq_of_stream, q);

                if (j < 0 || j >= n_kv) {
                    return false;
                }

                const int64_t  pb   = int64_t(q) >> r_shift;
                const int64_t  slot = int64_t(q) & (r - 1);
                const uint64_t bit  = uint64_t(1) << slot;

                if (pb >= n_blocks || (grp_slots[pb] & bit) != 0) {
                    return false;
                }

                grp_head[pb]  = (int32_t) pb;
                grp_first[pb] = grp_first[pb] < 0 ? j : std::min(grp_first[pb], j);
                grp_slots[pb] |= bit;

                grp_cells[pb*r + slot] = j;

                if (slot == 0) {
                    grp_slot0[pb] = j;
                }

                if (grp_slots[pb] == slots_full) {
                    if (n_bid > 0 && bid_idx[n_bid - 1] >= pb*r) {
                        return false;
                    }

                    add_bid((int32_t) pb, pb);

                    if (n_bid == 1) {
                        std::fill(rep_s.begin(), rep_s.end(), bid_slot0[0]);
                    } else {
                        rep_s[n_bid - 1] = bid_slot0[n_bid - 1];
                    }
                }
            }

            return true;
        };

        const bool inc = inc_config && w.inc_valid && w.inc_cells == &cells && w.inc_seq == seq_of_stream &&
            w.inc_n_kv == n_kv && w.inc_n_blocks == n_blocks && w.inc_r == r &&
            cells.get_used() == w.inc_used + (uint32_t) n_tps && add_cells();

        if (!inc) {
            // a failed incremental update leaves partial groups, which the scan overwrites
            n_bid = 0;

            blk_cells_s.assign(r*n_blocks, 0);

            group_cells();

            // mrope repeats one position across an image, so rank cells instead of using the position
            if (dup && ubatch->is_pos_2d() && one_seq) {
                order.clear();
                order.reserve(n_kv);

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (!cells.is_empty(j)) {
                        order.push_back((int32_t) j);
                    }
                }

                // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
                std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                    const llama_pos pa = cells.pos_get(a);
                    const llama_pos pb = cells.pos_get(b);

                    if (pa != pb) {
                        return pa < pb;
                    }

                    const auto & ea = cells.ext_get(a);

                    return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
                });

                rank.assign(n_kv, -1);

                for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                    rank[order[k]] = (int32_t) k;
                }

                ranked = true;

                group_cells();
            }

            GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

            // a complete group fills every slot of its block
            // indexed writes, not push_back and a std::copy (a memmove call) per block: 25600 blocks at 100k cells
            bid_idx  .resize(n_blocks);
            bid_cell .resize(n_blocks);
            bid_slot0.resize(n_blocks);
            bid_pos  .resize(4*(size_t) n_blocks);

            for (int64_t pb = 0; pb < n_blocks; ++pb) {
                for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                    if (grp_slots[g] == slots_full) {
                        add_bid(g, pb);
                    }
                }
            }

            // a padding block reads the pooled row of a real block, or of cell 0 (zeroed, then written by a padding row)
            rep_s.resize(n_blocks);

            for (int64_t b = 0; b < n_blocks; ++b) {
                rep_s[b] = b < n_bid ? bid_slot0[b] : (n_bid > 0 ? bid_slot0[0] : 0);
            }
        }

        w.inc_valid    = inc_config && !ranked && !dup && !oor;
        w.inc_cells    = &cells;
        w.inc_seq      = seq_of_stream;
        w.inc_n_kv     = n_kv;
        w.inc_n_blocks = n_blocks;
        w.inc_r        = r;
        w.inc_used     = cells.get_used();
        w.inc_n_bid    = n_bid;

        std::copy(rep_s.begin(), rep_s.end(), dst_rep + s*n_blocks);

        if (dst_blk_cells) {
            std::copy(blk_cells_s.begin(), blk_cells_s.end(), dst_blk_cells + s*(r*n_blocks));
        }

        const int32_t * cur_blk_cells = blk_cells_s.data();

        wr_grp.clear();
        grp_wr.assign(grp_first.size(), 0);

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        if (need_blk_of) {
            blk_of.resize(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t g = cell_grp[j];

                blk_of[j] = g < 0 ? -1 : grp_bid[g];

                if (cur_cell_blk) {
                    cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
                }
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the token's group changes with the token, so this ubatch pools it again
            if (!full && q/r < n_blocks) {
                for (int32_t g = grp_head[q/r]; g >= 0; g = grp_next[g]) {
                    const int32_t c = grp_cells[g*r + q%r];

                    if (c >= 0 && (one_seq || cells.seq_has((uint32_t) c, seq_id))) {
                        if (!grp_wr[g]) {
                            grp_wr[g] = 1;
                            wr_grp.push_back(g);
                        }
                        break;
                    }
                }
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block that is complete before the tail is visible to the whole token row; the tail and
                // later blocks are not candidates, so the top-k over blocks needs no attention mask
                float * cur_blk_bias = dst_bias + i*n_blocks;

                if (one_seq) {
                    // every block holds the token's sequence, and the blocks go up by index: the candidates are a prefix
                    const int64_t n_cand = std::lower_bound(bid_idx.begin(), bid_idx.begin() + n_bid, (int32_t) tail_start) -
                        bid_idx.begin();

                    std::fill(cur_blk_bias,          cur_blk_bias + n_cand,   0.0f);
                    std::fill(cur_blk_bias + n_cand, cur_blk_bias + n_blocks, -INFINITY);
                } else {
                    for (int64_t b = 0; b < n_blocks; ++b) {
                        const bool cand = b < n_bid && bid_idx[b] < tail_start &&
                            cells.seq_has((uint32_t) bid_cell[b], seq_id);

                        cur_blk_bias[b] = cand ? 0.0f : -INFINITY;
                    }
                }

                // the token's own block holds its own cell and the tail; groups key on the sequence set,
                // so the slots of one sequence can sit in more than one group
                int32_t own[64];
                std::fill(own, own + r, -1);

                for (int32_t g = grp_head[q/r]; g >= 0; g = grp_next[g]) {
                    if (!one_seq && !cells.seq_has((uint32_t) grp_first[g], seq_id)) {
                        continue;
                    }
                    for (int64_t k = 0; k < r; ++k) {
                        if (grp_cells[g*r + k] >= 0) {
                            own[k] = grp_cells[g*r + k];
                        }
                    }
                }

                GGML_ASSERT(own[q%r] >= 0 && "qsa: the token has no cell");
                for (int64_t k = tail_start; k <= q; ++k) {
                    GGML_ASSERT(own[k%r] >= 0 && "qsa: a tail position has no cell");
                }

                int32_t * cur_tail = (int32_t *) tail->data + i*r;

                // -1 marks a slot past the tail; the graph fills it with a cell it selects anyway
                cur_tail[0] = own[q%r];
                for (int64_t k = 1; k < r; ++k) {
                    const int64_t slot = k - 1;
                    cur_tail[k] = tail_start + slot <= q ? own[slot] : -1;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }

        // the pooled rows: a block's members (its first cell for a missing slot), mrope rows and the row its key goes to
        int32_t * cur_wr_cells = dst_wr_cells + s*(r*n_wr);
        int32_t * cur_wr_idx   = dst_wr_idx   + s*n_wr;

        const int64_t row_base = ((int64_t) stream_base + s)*kv_size;

        int64_t n_rows = 0;

        auto put_row = [&](int64_t row, const int32_t * members, const int32_t * sec_pos, int32_t target) {
            for (int64_t k = 0; k < r; ++k) {
                cur_wr_cells[row*r + k] = members[k];
            }
            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_wr_pos[sec*(n_wr*n_ns) + s*n_wr + row] = sec_pos[sec];
            }
            cur_wr_idx[row] = (int32_t) (row_base + target);
        };

        int32_t members[64];
        int32_t sec_pos[4];

        if (full) {
            for (int32_t b = 0; b < n_bid; ++b) {
                put_row(n_rows++, cur_blk_cells + b*r, bid_pos.data() + 4*b, bid_slot0[b]);
            }
        } else {
            GGML_ASSERT((int64_t) wr_grp.size() <= n_wr);

            for (const int32_t g : wr_grp) {
                for (int64_t k = 0; k < r; ++k) {
                    members[k] = grp_cells[g*r + k] >= 0 ? grp_cells[g*r + k] : grp_first[g];
                }
                group_pos(g, sec_pos);

                // a group without its first token is not a block yet: its key goes to a cell of its own, which no block reads
                put_row(n_rows++, members, sec_pos, grp_slot0[g] >= 0 ? grp_slot0[g] : grp_first[g]);
            }
        }

        if (n_rows == 0) {
            std::fill(members, members + r, 0);
            std::fill(sec_pos, sec_pos + 4, 0);

            put_row(n_rows++, members, sec_pos, 0);
        }

        // padding repeats the first row, so a row written twice gets the same key twice
        for (int64_t row = n_rows; row < n_wr; ++row) {
            for (int64_t k = 0; k < r; ++k) {
                cur_wr_cells[row*r + k] = cur_wr_cells[k];
            }
            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_wr_pos[sec*(n_wr*n_ns) + s*n_wr + row] = dst_wr_pos[sec*(n_wr*n_ns) + s*n_wr];
            }
            cur_wr_idx[row] = cur_wr_idx[0];
        }
    }

    if (full) {
        qsa_dirty = false;
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

bool llama_memory_hybrid_idx_context::qsa_full() const {
    GGML_ASSERT(mem != nullptr);

    return mem->qsa_full();
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * rep_cells,
        ggml_tensor * wr_cells,
        ggml_tensor * wr_pos,
        ggml_tensor * wr_idx,
        ggml_tensor * bias,
        ggml_tensor * tail,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        bool full) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, rep_cells, wr_cells, wr_pos, wr_idx, bias, tail, ubatch, ratio, blk_bias, full,
            get_idx()->get_stream_base());
}
