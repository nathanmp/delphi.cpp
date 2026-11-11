#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

//
// llama_memory_hybrid
//

// delphi: only the pooled sparse attention (glm5next, build_attn_sparse) stages or gathers the cells of a KV window.
// Other models would read the host cells in place on every step, so they ignore the window variables.
static const char * llama_kv_window_env(const llama_model & model, bool has_kpool, const char * env_name) {
    const char * env = getenv(env_name);
    if (env == nullptr || *env == '\0') {
        // LLAMA_KV_RESIDENT_CELLS=auto also windows the indexer cache (llama_kv_resident_cells)
        const char * attn = getenv("LLAMA_KV_RESIDENT_CELLS");
        if (attn == nullptr || strcmp(attn, "auto") != 0 || strcmp(env_name, "LLAMA_KV_IDX_RESIDENT_CELLS") != 0) {
            return nullptr;
        }
    }
    if (!has_kpool) {
        LLAMA_LOG_WARN("%s: %s is set, but %s has no KV window support (only the pooled sparse attention of glm5next): "
                "the cache stays in VRAM\n", __func__, env_name, llm_arch_name(model.arch));
        return nullptr;
    }
    return env_name;
}

llama_memory_hybrid::llama_memory_hybrid(
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
    const layer_filter_cb & filter_idx,
                ggml_type   type_idx) :
    hparams(model.hparams),
    hparams_idx(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr,
        "",
        false,
        /* kv_window_env */ llama_kv_window_env(model, filter_idx != nullptr && model.hparams.indexer_kpool > 0,
                "LLAMA_KV_RESIDENT_CELLS")
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // a *pooling* indexer (indexer_kpool > 0, glm5next only) needs a second head for
        // the compressor gate, or the pool cannot be rebuilt once its tokens leave the
        // batch. every other arch leaves indexer_kpool 0 and is unchanged.
        const uint32_t n_head_idx = model.hparams.indexer_kpool > 0 ? 2 : 1;

        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), n_head_idx);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells, %u x %u per cell, type = %s\n",
                __func__, kv_size, n_head_idx, hparams_idx.n_embd_head_k_full, ggml_type_name(type_idx));

        // the indexer cache is read only to compress pools: all of them after an invalidation, else the cells of the
        // pools that a batch completes. Scoring reads the pool keys (mem_kpool, resident), so nearly all of this cache
        // can sit in host memory (LLAMA_KV_IDX_RESIDENT_CELLS)
        return new llama_kv_cache(
            model, hparams_idx, type_idx, type_idx, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "", false,
            llama_kv_window_env(model, model.hparams.indexer_kpool > 0, "LLAMA_KV_IDX_RESIDENT_CELLS"));
    }()),
    mem_kpool(filter_idx == nullptr || model.hparams.indexer_kpool == 0 ? nullptr :
        new llama_kpool_cache(
            model, offload, unified, kv_size, n_seq_max,
            model.hparams.indexer_kpool, model.hparams.indexer_head_size, filter_idx)) {}

uint32_t llama_memory_hybrid::get_n_seq_split_tokens() const {
    if (mem_kpool == nullptr || mem_attn->get_n_stream() <= 1) {
        return 0;
    }
    static const uint32_t n = [] {
        const char * env = getenv("LLAMA_HYBRID_SEQ_SPLIT_TOKENS");
        return env != nullptr ? (uint32_t) atoi(env) : 16u;
    }();
    return n;
}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        // delphi: a prompt of one sequence takes its own ubatches, ahead of the other sequences. An equal split cut it
        // into pieces as short as the other sequences' parts, and a prompt ubatch over several KV streams stages the
        // KV window of every stream (GLM at 3 x 512K: 1.5 GiB per layer) and needs its own reservation.
        if (const uint32_t n_split = get_n_seq_split_tokens(); n_split > 0 && !embd_all) {
            const llama_batch & batch = balloc.get_batch();
            std::vector<uint32_t> n_seq_tokens(LLAMA_MAX_SEQ, 0);
            bool single = true;
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                single = single && batch.n_seq_id[i] == 1;
                n_seq_tokens[batch.seq_id[i][0]]++;
            }
            const uint32_t n_rs_seq = mem_recr->n_rs_seq;
            const uint32_t n_tail = n_rs_seq > 0 ? n_rs_seq + 1 : 0;
            // LLAMA_STAGGERED_VERIFY=1: every sequence takes its own ubatches, also a short verify. The scheduler can then
            // pipeline the ubatches of different sequences over the devices of a layer split.
            static const bool staggered = [] {
                const char * env = getenv("LLAMA_STAGGERED_VERIFY");
                return env != nullptr && atoi(env) != 0;
            }();
            // LLAMA_STAGGERED_MIXED_UBATCH=N: when a batch has prompt and verify sequences, cut the prompts into ubatches of
            // at most N tokens, so that they pipeline over the devices and the verify of the batch completes earlier
            static const uint32_t n_ubatch_mixed = [] {
                const char * env = getenv("LLAMA_STAGGERED_MIXED_UBATCH");
                return env != nullptr ? (uint32_t) std::max(0, atoi(env)) : 0u;
            }();
            bool has_short = false;
            for (llama_seq_id seq = 0; seq < LLAMA_MAX_SEQ; ++seq) {
                has_short = has_short || (n_seq_tokens[seq] > 0 && n_seq_tokens[seq] <= n_split);
            }
            const uint32_t n_ubatch_seq = staggered && has_short && n_ubatch_mixed > 0 ? std::min(n_ubatch, n_ubatch_mixed) : n_ubatch;
            for (llama_seq_id seq = 0; single && seq < LLAMA_MAX_SEQ; ++seq) {
                if (n_seq_tokens[seq] == 0 || (!staggered && n_seq_tokens[seq] <= n_split)) {
                    continue;
                }
                for (uint32_t left = n_seq_tokens[seq]; left > 0;) {
                    uint32_t n = std::min(n_ubatch_seq, left);
                    // the trailing (1 + n_rs_seq) tokens of the seq stay in one ubatch
                    if (n < left && left - n < n_tail && n_tail < n_ubatch_seq) {
                        n = left - n_tail;
                    }
                    auto ubatch = balloc.split_seq_id(n, seq);
                    GGML_ASSERT(ubatch.n_tokens == n);
                    left -= n;
                    ubatches.push_back(std::move(ubatch)); // NOLINT
                }
            }
        }

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

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
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer takes the attention cache's slot layout rather than finding its
        // own: allocated separately the two drift apart when the context is rewritten
        // between turns, and the top-k indices would then point at the wrong cells
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches), std::move(heads_idx));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_full_n_seq(
        uint32_t n_seq, uint32_t n_kv, bool kpool_rebuild) {
    return std::make_unique<llama_memory_hybrid_context>(this, n_seq, n_kv, kpool_rebuild);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    if (mem_idx && !mem_idx->get_can_shift()) {
        return false;
    }
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::set_mtp_index_reuse(bool reuse) {
    if (mem_kpool) {
        mem_kpool->set_mtp_index_reuse(reuse);
    }
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    if (mem_idx) mem_idx->clear(data);
    if (mem_kpool) mem_kpool->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.  An attention-only hybrid
    // cache has no recurrent state to constrain partial rollback.
    if (mem_recr->has_layers() && !mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (mem_idx) mem_idx->seq_rm(seq_id, p0, p1);
    const bool result = mem_attn->seq_rm(seq_id, p0, p1);
    if (result && mem_kpool) mem_kpool->seq_rm(seq_id, p0, p1);
    return result;
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_idx) mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_recr->has_layers()) mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_kpool) mem_kpool->invalidate(seq_id_dst);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    if (mem_idx) mem_idx->seq_keep(seq_id);
    if (mem_recr->has_layers()) mem_recr->seq_keep(seq_id);
    if (mem_kpool) mem_kpool->invalidate();
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    if (mem_idx) mem_idx->seq_add(seq_id, p0, p1, shift);
    if (mem_recr->has_layers()) mem_recr->seq_add(seq_id, p0, p1, shift);
    if (mem_kpool) mem_kpool->invalidate(seq_id);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    if (mem_idx) mem_idx->seq_div(seq_id, p0, p1, d);
    if (mem_recr->has_layers()) mem_recr->seq_div(seq_id, p0, p1, d);
    if (mem_kpool) mem_kpool->invalidate(seq_id);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    if (!mem_recr->has_layers()) {
        return mem_attn->seq_pos_min(seq_id);
    }
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    if (!mem_recr->has_layers()) {
        return mem_attn->seq_pos_max(seq_id);
    }
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    if (mem_kpool) {
        for (const auto & buft_size : mem_kpool->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
        // indexer keys are not recomputable; skipping them here misselects on restore
        if (mem_idx) mem_idx->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
        if (mem_idx) mem_idx->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
    if (mem_kpool) mem_kpool->invalidate(seq_id);
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_kv_cache * llama_memory_hybrid::get_mem_idx() const {
    return mem_idx.get();
}

llama_kpool_cache * llama_memory_hybrid::get_kpool_cache() const {
    return mem_kpool.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : mem(nullptr), status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr : mem->get_mem_idx()->init_full()),
    mem(mem),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
                  uint32_t   n_seq,
                  uint32_t   n_kv,
                      bool   kpool_rebuild) :
    ctx_attn(n_kv == 0 ?
        new llama_kv_cache_context(mem->get_mem_attn(), n_seq) :
        new llama_kv_cache_context(mem->get_mem_attn(), n_seq, n_kv)),
    // n_seq counts KV streams; a unified KV cache has one stream for all sequences, so the recurrent state must cover all of them
    ctx_recr(mem->get_mem_recr()->init_full_n_seq(
        mem->get_mem_attn()->get_n_stream() == 1 ? mem->get_mem_recr()->size : n_seq)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        (n_kv == 0 ?
            new llama_kv_cache_context(mem->get_mem_idx(), n_seq) :
            new llama_kv_cache_context(mem->get_mem_idx(), n_seq, n_kv))),
    mem(mem),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())),
    kpool_rebuild_override(kpool_rebuild ? 1 : 0) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    // indexer keys carry no positional encoding, but the pending per-cell delta must
    // still be cleared or the two caches disagree about whether a shift is outstanding.
    // safe because an indexer only exists for LLAMA_ROPE_TYPE_NONE archs, where
    // llama_kv_cache::update skips the K-shift graph and does only that
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr : mem->get_mem_idx()->init_update(lctx, optimize)),
    mem(mem),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches,
                  slot_info_vec_t   sinfos_idx) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), this->ubatches)),
    mem(mem),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();
    if (ctx_idx) ctx_idx->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();

        // a top-k over indexer cells is meaningful only if both caches cover the same
        // window
        if (!ubatches.empty()) {
            GGML_ASSERT(get_idx()->get_n_kv()     == get_attn()->get_n_kv());
            GGML_ASSERT(get_idx()->get_n_stream() == get_attn()->get_n_stream());
        }
    }

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

llama_kpool_cache * llama_memory_hybrid_context::get_kpool_cache() const {
    return mem == nullptr ? nullptr : mem->get_kpool_cache();
}

int llama_memory_hybrid_context::get_kpool_rebuild_override() const {
    return kpool_rebuild_override;
}
