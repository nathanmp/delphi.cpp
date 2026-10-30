#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"
#include "llama-engram.h"
#include "llama-expert-cache.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// DeepSeek-V4.1.
//
// Shares almost everything with DeepSeek-V4: the hyper-connection stream, the MoE, the latent
// attention and the compressed KV cache. Three things differ.
//
// 1. The hyper-connection coefficients lag by one sublayer. A sublayer computes the mix that the
//    NEXT one consumes, so attention uses what the previous layer's FFN produced. V4 computes and
//    consumes in the same sublayer.
// 2. There is no learned hyper-connection head. Nothing is left to collapse the copies with in V4,
//    which is why it needs one; here the last layer's FFN mix is still unused and does the job.
//    These two facts explain each other, and the file ships no output_hc_* tensors.
// 3. The engram tables: n-gram keyed lookups added into the stream at a few layers.
//
// The sparse attention also differs. V4 compresses KV on every layer at one of two fixed ratios;
// V4.1 compresses on a few source layers and the layers after each source read the same rows, and
// it derives index keys from that shared latent rather than from a second compressor. See
// build_attention_v41().
//
// The indexer picks in two levels. The candidate source keeps its best blocks of compressed positions (build_candidate_mask()), and the index sources after it pick positions only inside those blocks.

// mean over the hyper-connection copies; deepseek4.cpp keeps its own copy of this
static ggml_tensor * dsv41_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

int llama_model_deepseek41::engram_index(int il) const {
    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] == (uint32_t) il) {
            return (int) e;
        }
    }
    return -1;
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    ml.get_arr_n(LLM_KV_ENGRAM_LAYER_IDS, engram_n_layer);
    if (engram_n_layer == 0 || engram_n_layer > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("DeepSeek-V4.1 engram layer count %u is out of range", engram_n_layer));
    }
    ml.get_arr(LLM_KV_ENGRAM_LAYER_IDS, hparams.engram_layer_ids);

    ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,     hparams.engram_n_head);
    ml.get_key(LLM_KV_ENGRAM_KEY_LENGTH,     hparams.engram_key_length);
    ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.engram_max_ngram_size);
    ml.get_key(LLM_KV_ENGRAM_PAD_ID,         engram_pad_id);

    if (hparams.engram_n_head == 0 || hparams.engram_max_ngram_size < 2) {
        throw std::runtime_error("DeepSeek-V4.1 engram needs at least one head and a 2-gram");
    }

    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] >= hparams.n_layer()) {
            throw std::runtime_error(format("engram layer %u is out of range", hparams.engram_layer_ids[e]));
        }
    }

    ml.get_arr(LLM_KV_ENGRAM_MULTIPLIERS, engram_multipliers);
    ml.get_arr(LLM_KV_ENGRAM_PRIMES,      engram_primes);
    ml.get_arr(LLM_KV_ENGRAM_OFFSETS,     engram_offsets);
    ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,   engram_token_map);

    // the hash indexes straight into these, so a short array would read past the end
    const size_t n_bucket = (size_t) (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;

    if (engram_multipliers.size() != (size_t) engram_n_layer * hparams.engram_max_ngram_size) {
        throw std::runtime_error("engram multiplier count does not match layers * ngram size");
    }
    if (engram_primes.size() != (size_t) engram_n_layer * n_bucket ||
        engram_offsets.size() != engram_primes.size()) {
        throw std::runtime_error("engram prime or offset count does not match layers * buckets");
    }
    for (uint64_t p : engram_primes) {
        if (p == 0) {
            throw std::runtime_error("engram prime of zero would divide by zero in the hash");
        }
    }

    // The first level of the indexer. Without the keys the model runs without it, which matches the reference only while the candidate stream holds at most block_size * topk_blocks rows.
    uint32_t cand_source = 0;
    if (ml.get_key(LLM_KV_ATTENTION_CANDIDATE_SOURCE_LAYER, cand_source, false)) {
        ml.get_key(LLM_KV_ATTENTION_CANDIDATE_BLOCK_SIZE,  hparams.dsv41_candidate_block_size);
        ml.get_key(LLM_KV_ATTENTION_CANDIDATE_TOPK_BLOCKS, hparams.dsv41_candidate_topk_blocks);

        if (cand_source >= hparams.n_layer()) {
            throw std::runtime_error(format("candidate source layer %u is out of range", cand_source));
        }
        if (hparams.dsv41_candidate_block_size == 0 || hparams.dsv41_candidate_topk_blocks == 0) {
            throw std::runtime_error("the candidate block size and block count must not be zero");
        }
        // the compressed caches are padded to a multiple of 256 rows, and a block must not cross the end of the cache
        if (256 % hparams.dsv41_candidate_block_size != 0) {
            throw std::runtime_error(format("candidate block size %u does not divide 256", hparams.dsv41_candidate_block_size));
        }
        hparams.dsv41_candidate_source = (int32_t) cand_source;
    } else {
        LLAMA_LOG_WARN("%s: the file has no %s, so the indexer runs without its first level. Output differs from the reference above candidate_block_size * candidate_topk_blocks compressed positions (16384 tokens for DeepSeek-V4.1-Flash). Convert the model again to add the keys.\n",
                __func__, ml.llm_kv(LLM_KV_ATTENTION_CANDIDATE_SOURCE_LAYER).c_str());
    }
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head   = hparams.n_embd_head_k();
    const int64_t o_groups      = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank   = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult       = hparams.dsv4_hc_mult;
    const int64_t hc_dim        = hc_mult * n_embd;
    const int64_t hc_mix_dim    = (2 + hc_mult) * hc_mult;
    const int64_t n_embd_indexer = hparams.indexer_head_size;

    if ((size_t) n_vocab > engram_token_map.size()) {
        throw std::runtime_error(format("engram token map has %zu entries, too few for %" PRId64 " tokens",
                                        engram_token_map.size(), n_vocab));
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        // the file lays wo_a out as (n_head * n_embd_head / o_groups, o_lora_rank * o_groups),
        // so reshape at load and keep the graph free of it
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        // Only the KV source layers carry a compressor, and only those with a ratio above 1 pool
        // with a gate, so both are optional rather than keyed off the ratio the way V4 does it.
        layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, TENSOR_NOT_REQUIRED);

        // An index source scores queries against shared index keys. Only a layer that also
        // compresses its own KV builds those keys; the rest read what an earlier layer published.
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, TENSOR_NOT_REQUIRED);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd_head, n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer}, TENSOR_NOT_REQUIRED);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        // vision variant only: routing bias for image tokens
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);

        const int eg = engram_index(i);
        if (eg >= 0) {
            const int64_t n_cols = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
            const int64_t key_len = hparams.engram_key_length;

            // The table has hundreds of millions of rows and is far too large to hold in memory,
            // but each token only touches n_cols of them, so read those rows on demand.
            const std::string embd_name = tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i).str();

            // a row index is a bucket offset plus a hash, so the last bucket has to end inside
            uint64_t max_row = 0;
            for (int64_t b = 0; b < n_cols; ++b) {
                const size_t k = (size_t) eg*n_cols + b;
                max_row = std::max(max_row, engram_offsets[k] + engram_primes[k]);
            }

            // A model built without a file (llama_model_init_from_user) has no weight to size the
            // table from, so it gets exactly the rows the buckets address. With a file, a missing
            // table still fails in create_tensor() below.
            const auto * embd_w = ml.get_weight(embd_name.c_str());
            const int64_t n_rows = embd_w ? embd_w->tensor->ne[1] : (int64_t) max_row;

            if ((int64_t) max_row > n_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the engram buckets (%" PRIu64 ")",
                                                embd_name.c_str(), n_rows, max_row));
            }

            layer.engram_embd = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i), {key_len, n_rows}, TENSOR_READ_LAZY);
            layer.engram_wkv  = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV,  "weight", i), {n_cols * key_len, n_embd * (hc_mult + 1)}, 0);
            layer.engram_q    = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,    "weight", i), {n_embd, hc_mult}, 0);
            layer.engram_k    = create_tensor(tn(LLM_TENSOR_ENGRAM_K,    "weight", i), {n_embd, hc_mult}, 0);
        }

    }

    // Work out which layer publishes the stream each layer reads. Only a source carries a
    // compressor, only an index key owner carries indexer_attn_k, and only an index source
    // carries indexer_attn_q_b, so the file itself says which layer plays which role.
    hparams.dsv41_kv_source.fill(-1);
    hparams.dsv41_index_key_source.fill(-1);
    hparams.dsv41_topk_source.fill(-1);

    int32_t last_kv_source    = -1;
    int32_t last_key_owner    = -1;
    int32_t last_index_source = -1;

    for (int i = 0; i < n_layer; ++i) {
        const auto & layer = layers[i];

        if (layer.attn_comp_wkv)    { last_kv_source    = i; }
        if (layer.indexer_attn_k)   { last_key_owner    = i; }
        if (layer.indexer_attn_q_b) { last_index_source = i; }

        if (hparams.dsv4_compress_ratios[i] == 0) {
            // pure sliding window, no compressed stream to read
            continue;
        }

        if (last_kv_source < 0 || last_key_owner < 0 || last_index_source < 0) {
            throw std::runtime_error(format("layer %d reads a compressed stream before any layer publishes one", i));
        }

        // the row layout of a stream follows the ratio it was compressed at, so a reader that
        // disagrees with its source would index into rows that stand for different positions
        // the same holds for the index keys and for a top-k pick, which name positions in a stream
        for (const int32_t src : { last_kv_source, last_key_owner, last_index_source }) {
            if (hparams.dsv4_compress_ratios[i] != hparams.dsv4_compress_ratios[src]) {
                throw std::runtime_error(format("layer %d compresses at ratio %u but reads layer %d, compressed at %u",
                                                i, hparams.dsv4_compress_ratios[i],
                                                src, hparams.dsv4_compress_ratios[src]));
            }
        }

        hparams.dsv41_kv_source[i]        = last_kv_source;
        hparams.dsv41_index_key_source[i] = last_key_owner;
        hparams.dsv41_topk_source[i]      = last_index_source;
    }

    {
        std::string kv_sources;
        std::string index_sources;
        for (int i = 0; i < n_layer; ++i) {
            if (hparams.dsv41_is_kv_source(i)) {
                kv_sources += (kv_sources.empty() ? "" : ",") + std::to_string(i);
            }
            if (hparams.dsv41_is_index_source(i)) {
                index_sources += (index_sources.empty() ? "" : ",") + std::to_string(i);
            }
        }
        LLAMA_LOG_INFO("%s: DeepSeek-V4.1 compressed stream sources: kv = [%s], index = [%s]\n",
                __func__, kv_sources.c_str(), index_sources.c_str());
    }

    // a compressor with no gate only makes sense where there is nothing to pool
    for (int i = 0; i < n_layer; ++i) {
        if (hparams.dsv41_is_kv_source(i) && !layers[i].attn_comp_wgate && hparams.dsv4_compress_ratios[i] != 1) {
            throw std::runtime_error(format("layer %d compresses %u tokens per row but has no pooling gate",
                                            i, hparams.dsv4_compress_ratios[i]));
        }
    }

    // The candidate mask covers the stream of the candidate source. So only an index source can make it, and an index source after it can only apply it when both read a stream of the same ratio.
    const int32_t cand_src = hparams.dsv41_candidate_source;
    if (cand_src >= 0) {
        if (!hparams.dsv41_is_index_source(cand_src)) {
            throw std::runtime_error(format("candidate source layer %d is not an index source", cand_src));
        }
        for (int i = cand_src + 1; i < n_layer; ++i) {
            if (hparams.dsv41_is_index_source(i) && hparams.dsv4_compress_ratios[i] != hparams.dsv4_compress_ratios[cand_src]) {
                throw std::runtime_error(format("index source layer %d compresses at ratio %u, but candidate source layer %d compresses at %u",
                                                i, hparams.dsv4_compress_ratios[i], cand_src, hparams.dsv4_compress_ratios[cand_src]));
            }
        }
        LLAMA_LOG_INFO("%s: DeepSeek-V4.1 candidate blocks: source = %d, block size = %u, top-k blocks = %u\n",
                __func__, cand_src, hparams.dsv41_candidate_block_size, hparams.dsv41_candidate_topk_blocks);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Engram n-gram hash: each token gathers n_cols rows of this layer's table.
//   rolling_i = (t[0]*m[0]) ^ ... ^ (t[i]*m[i]);  row = rolling_i % prime[i][h] + offset[i][h]
// The hash runs host-side because ggml has no 64 bit integers and no xor. Look-back stops at the
// start of the sequence, and the compressed token map folds case and accents together first.
class llm_graph_input_engram : public llm_graph_input_i {
public:
    llm_graph_input_engram(const llama_model_deepseek41 & pmodel,
                           const llama_kv_cache_dsv4_raw_context * mctx,
                           int eg) : pmodel(pmodel), mctx(mctx), eg(eg) {}
    virtual ~llm_graph_input_engram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx)->get_raw();
        const int64_t n_cols = (pmodel.hparams.engram_max_ngram_size - 1) * pmodel.hparams.engram_n_head;
        return rows->ne[0] == n_cols * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [n_cols * n_tokens]
    const ggml_tensor * table = nullptr;
    llama_engram_reader * reader = nullptr;
    llama_engram_reader::job_ptr lookup;

    static void collect(ggml_tensor * dst, int ith, int nth, void * userdata) {
        GGML_UNUSED(ith);
        GGML_UNUSED(nth);
        auto * input = static_cast<llm_graph_input_engram *>(userdata);
        llama_engram_reader::collect(input->lookup, (float *) dst->data, ggml_nbytes(dst));
    }

    const llama_model_deepseek41 & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_dsv4_raw_context * mctx;

    // which engram layer this is, so the right multipliers and buckets are used
    const int eg;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_engram::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.engram_max_ngram_size;
    const int64_t n_heads  = hp.engram_n_head;
    const int64_t n_cols   = (n_gram - 1) * n_heads;
    const int64_t n_prev   = n_gram - 1;

    const uint64_t * mult = pmodel.engram_multipliers.data() + (size_t) eg*n_gram;
    const uint64_t * prime = pmodel.engram_primes.data()  + (size_t) eg*n_cols;
    const uint64_t * offset = pmodel.engram_offsets.data() + (size_t) eg*n_cols;

    // an image arrives as an embd batch, so ubatch->token is null; the graph skips the engram for
    // those batches, so the padding token here is only a safe default
    const int32_t pad = (int32_t) pmodel.engram_pad_id;
    auto map_of = [&](llama_token t) -> uint64_t {
        if (t < 0 || (size_t) t >= pmodel.engram_token_map.size()) {
            return (uint64_t) pad;
        }
        return (uint64_t) pmodel.engram_token_map[t];
    };

    std::vector<int32_t> idx(n_cols * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "engram n-gram lookups do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // look-back stops at the start of the sequence; everything from there on reads as padding
        std::vector<uint64_t> ctx(n_gram);
        ctx[0] = ubatch->token ? map_of(ubatch->token[i]) : (uint64_t) pad;
        bool blocked = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = blocked ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            blocked = blocked || t < 0;
            ctx[s] = blocked ? (uint64_t) pad : map_of(t);
        }

        // compressed ids stay under 2^17 and the multipliers under 2^37, so no product overflows
        uint64_t rolling = ctx[0] * mult[0];
        for (int64_t s = 1; s < n_gram; ++s) {
            rolling ^= ctx[s] * mult[s];

            for (int64_t h = 0; h < n_heads; ++h) {
                const int64_t b = (s - 1)*n_heads + h;
                idx[i*n_cols + b] = (int32_t) (rolling % prime[b] + offset[b]);
            }
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
    if (reader) {
        lookup = reader->submit(table, std::move(idx));
    }
}

ggml_tensor * llama_model_deepseek41::graph::build_inp_engram(
        const llama_model & model,
        int il) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    const int64_t n_cols  = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
    const int64_t key_len = hparams.engram_key_length;

    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_engram>(pmodel, mctx_cur->get_raw(), pmodel.engram_index(il));

    inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cols * n_tokens);
    ggml_set_input(inp->rows);
    ggml_tensor * rows = inp->rows;
    // gather then flatten, laying the buckets out slowest, as the reference does
    ggml_tensor * emb;
    const auto * table = model.layers[il].engram_embd;
    if (engram_reader && n_tokens <= 16 && llama_engram_reader::supports(table)) {
        inp->reader = engram_reader;
        inp->table = table;
        ggml_tensor * args[] = { rows };
        emb = ggml_custom_4d(ctx0, GGML_TYPE_F32, key_len, n_cols*n_tokens, 1, 1,
                args, 1, llm_graph_input_engram::collect, 1, inp.get());
        cb(emb, "engram_lookup", il);
        ggml_backend_sched_set_tensor_backend(sched, emb, backend_cpu);
    } else {
        emb = ggml_get_rows(ctx0, model.layers[il].engram_embd, rows);
    }
    res->add_input(std::move(inp));
    emb = ggml_reshape_2d(ctx0, emb, key_len * n_cols, n_tokens);
    cb(emb, "engram_embd", il);

    return emb;
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model & model,
        ggml_tensor * x,
        ggml_tensor * emb,
        int il) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // one projection makes a key per hc copy plus one value they all share
    ggml_tensor * kv = build_lora_mm(model.layers[il].engram_wkv, emb);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, hc_dim, nt, kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc_dim*kv->nb[0]));

    // The gate scales reach ggml_mul, which takes only f32, and a file quantized before
    // llama-quant.cpp learned to skip them carries them quantized. get_rows dequantizes.
    auto as_f32 = [&](ggml_tensor * w) {
        if (w->type == GGML_TYPE_F32) {
            return w;
        }
        ggml_tensor * ids = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) w->ne[1], 1.0f), GGML_TYPE_I32);
        return ggml_get_rows(ctx0, w, ids);
    };

    // normalized per (token, hc copy) over n_embd, not jointly over the copies. The reference
    // keeps engram_q and engram_k apart but only ever uses their product, so applying one to each
    // side of the dot product gives the same result.
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        t = ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
        t = ggml_rms_norm(ctx0, t, norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, nt);
        t = ggml_mul(ctx0, t, ggml_reshape_2d(ctx0, w, hc_dim, 1));
        return ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
    };

    ggml_tensor * k = grouped_norm(key, as_f32(model.layers[il].engram_k));
    ggml_tensor * q = grouped_norm(x,   as_f32(model.layers[il].engram_q));

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, k, q));
    s = ggml_scale(ctx0, s, 1.0f/sqrtf((float) n_embd));

    // signed square root before the sigmoid, matching the training kernel.
    // The reference uses copysign, which treats +0 as positive where ggml_sgn gives 0, so a dot
    // product of exactly zero gates 0.5 here against 0.50025 there. qwen4exp's PLE gate is built
    // the same way.
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "engram_gate", il);

    // the value is shared across the copies, only the gate differs
    ggml_tensor * v = ggml_reshape_3d(ctx0, value, n_embd, 1, nt);
    v = ggml_repeat_4d(ctx0, v, n_embd, hc, nt, 1);

    return ggml_add(ctx0, x, ggml_mul(ctx0, v, gate));
}

// Rope settings for one layer. A layer that reads a compressed stream rotates with the
// compressor's base and YaRN; a plain sliding window layer rotates with the model's.
struct dsv41_rope_cfg {
    float   base;
    float   scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
    int32_t n_ctx_orig;
};

dsv41_rope_cfg llama_model_deepseek41::graph::rope_cfg(int il) const {
    if (hparams.dsv4_compress_ratios[il] == 0) {
        return { freq_base, 1.0f, 0.0f, dsv4_rope_attn_factor(1.0f, 0.0f), 0.0f, 0.0f, 0 };
    }

    return {
        hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
        dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow, n_ctx_orig,
    };
}

// Undo the rotation the query carried into attention, then the grouped output projection. wo_a is
// block diagonal over groups, each projecting only its own heads, hence a batched mul_mat.
ggml_tensor * llama_model_deepseek41::graph::build_attention_tail(
        const llama_model & model,
        ggml_tensor * out,
        ggml_tensor * inp_pos,
        int64_t nt,
        int il) const {
    ggml_tensor * oa = build_attention_wo_a(model, out, inp_pos, nt, 0, hparams.dsv4_o_group_count, il);

    out = build_lora_mm(model.layers[il].wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

ggml_tensor * llama_model_deepseek41::graph::build_attention_wo_a(
        const llama_model & model,
        ggml_tensor * out,
        ggml_tensor * inp_pos,
        int64_t nt,
        int64_t g0,
        int64_t ng,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t n_heads_group    = n_head/n_groups;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;

    const dsv41_rope_cfg rc = rope_cfg(il);

    out = ggml_reshape_3d(ctx0, out, n_embd_head, ng*n_heads_group, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, ng, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);

    ggml_tensor * wo_a = layer.wo_a;
    if (g0 > 0 && cparams.attn_split_w && cparams.attn_split_w->wo_a[il]) {
        wo_a = cparams.attn_split_w->wo_a[il];
        GGML_ASSERT(g0 + ng == n_groups && wo_a->ne[2] == ng);
    } else if (ng != n_groups) {
        wo_a = ggml_view_3d(ctx0, wo_a, wo_a->ne[0], wo_a->ne[1], ng, wo_a->nb[1], wo_a->nb[2], g0*wo_a->nb[2]);
    }
    ggml_tensor * oa = ggml_mul_mat(ctx0, wo_a, out);
    cb(oa, "attn_wo_a", il);

    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*ng, nt);

    return oa;
}

ggml_tensor * llama_model_deepseek41::graph::build_query(
        const llama_model & model,
        ggml_tensor * qr,
        ggml_tensor * inp_pos,
        int64_t h0,
        int64_t nh,
        int il) const {
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t nt               = qr->ne[1];

    const dsv41_rope_cfg rc = rope_cfg(il);

    ggml_tensor * wq_b = model.layers[il].wq_b;
    if (h0 > 0 && cparams.attn_split_w && cparams.attn_split_w->wq_b[il]) {
        // the heads of the split device, copied into its memory
        wq_b = cparams.attn_split_w->wq_b[il];
        GGML_ASSERT(h0 + nh == n_head && wq_b->ne[1] == nh*n_embd_head);
    } else if (nh != n_head) {
        wq_b = ggml_view_2d(ctx0, wq_b, wq_b->ne[0], nh*n_embd_head, wq_b->nb[1], h0*n_embd_head*wq_b->nb[1]);
    }
    ggml_tensor * q = build_lora_mm(wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, nh, nt);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    q = ggml_rope_set_offset(q, n_embd_head - n_embd_head_rope);
    cb(q, "q", il);

    return q;
}

// Heads split over two devices (LLAMA_ATTN_SPLIT). The split device computes the queries, attention and wo_a of the last
// attn_split_groups output groups, this device those of the others, and wo_b runs here on both. K has one head and the
// top-k is per query, so both devices use the same K and top-k; the split device builds its own mask from the top-k, since
// the mask is as wide as the compressed stream. It takes every input from here before its first kernel: an input copied
// later would wait on this device for all the work queued before it (see ggml_backend_sched_compute_splits).
ggml_tensor * llama_model_deepseek41::graph::build_attention_split(
        const llama_model & model,
        ggml_tensor * qr,
        ggml_tensor * k,
        ggml_tensor * k_rot,
        ggml_tensor * raw_mask,
        ggml_tensor * comp_mask,
        ggml_tensor * top_k,
        int64_t n_kv_max,
        ggml_tensor * inp_pos,
        int64_t nt,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head   = hparams.n_embd_head_k();
    const int64_t n_groups      = hparams.dsv4_o_group_count;
    const int64_t n_heads_group = n_head/n_groups;
    const int64_t ng_b          = cparams.attn_split_groups;
    const float   kq_scale      = 1.0f/sqrtf(float(n_embd_head));

    // qr crosses as F16: wq_b quantizes it to 8 bits anyway
    ggml_tensor * qr16 = ggml_cast(ctx0, qr, GGML_TYPE_F16);
    cb(qr16, "attn_xa_qr16", il);
    // everything that both parts read is in the graph before them, or it would be pinned with the first part
    ggml_build_forward_expand(gf, qr);
    ggml_build_forward_expand(gf, qr16);
    ggml_build_forward_expand(gf, k);
    if (top_k) {
        ggml_build_forward_expand(gf, top_k);
    }

    auto part = [&](int64_t g0, int64_t ng, bool remote) {
        const int n0 = ggml_graph_n_nodes(gf);
        const int64_t h0 = g0*n_heads_group;
        const int64_t nh = ng*n_heads_group;

        ggml_tensor * q_in = qr;
        ggml_tensor * k_in = k;
        ggml_tensor * top_k_in = top_k;
        if (remote) {
            q_in = ggml_cast(ctx0, qr16, GGML_TYPE_F32);
            ggml_build_forward_expand(gf, q_in);
            k_in = ggml_cont(ctx0, k);
            ggml_build_forward_expand(gf, k_in);
            if (top_k) {
                top_k_in = ggml_cont(ctx0, top_k);
                ggml_build_forward_expand(gf, top_k_in);
            }
        }

        ggml_tensor * q = build_query(model, q_in, inp_pos, h0, nh, il);
        if (k_rot) {
            q = llama_mul_mat_hadamard(ctx0, q, k_rot);
        }

        ggml_tensor * kq_mask = raw_mask;
        if (comp_mask) {
            ggml_tensor * m = top_k_in ? build_top_k_mask(comp_mask, top_k_in, "comp_top_k_mask", il) : comp_mask;
            kq_mask = ggml_concat(ctx0, raw_mask, m, 0);
            cb(kq_mask, "kq_mask", il);
        }

        ggml_tensor * sinks = remote && cparams.attn_split_w && cparams.attn_split_w->sinks[il]
            ? cparams.attn_split_w->sinks[il]
            : ggml_view_1d(ctx0, layer.attn_sinks, nh, h0*ggml_element_size(layer.attn_sinks));

        ggml_tensor * out = build_attn_mha(q, k_in, k_in, nullptr, kq_mask, sinks, nullptr, n_kv_max, kq_scale, il);
        if (k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, k_rot);
        }
        cb(out, "attn_out_raw", il);

        ggml_tensor * oa = build_attention_wo_a(model, out, inp_pos, nt, g0, ng, il);
        if (remote) {
            // the result crosses as F16 too: wo_b quantizes it to 8 bits
            oa = ggml_cast(ctx0, oa, GGML_TYPE_F16);
        }
        ggml_build_forward_expand(gf, oa);
        pin_split_nodes(n0, remote, il);
        return oa;
    };

    ggml_tensor * oa_b = part(n_groups - ng_b, ng_b, true);
    ggml_tensor * oa_a = part(0, n_groups - ng_b, false);

    // wo_b as two partial products over its input columns, so that this device multiplies its own half while the other
    // half is still on its way
    ggml_tensor * wo_b = layer.wo_b;
    const int64_t n_in_a = oa_a->ne[0];
    GGML_ASSERT(n_in_a % ggml_blck_size(wo_b->type) == 0);
    ggml_tensor * wo_b_a = ggml_view_2d(ctx0, wo_b, n_in_a,               wo_b->ne[1], wo_b->nb[1], 0);
    ggml_tensor * wo_b_b = ggml_view_2d(ctx0, wo_b, wo_b->ne[0] - n_in_a, wo_b->ne[1], wo_b->nb[1],
            ggml_row_size(wo_b->type, n_in_a));
    ggml_tensor * out_a = ggml_mul_mat(ctx0, wo_b_a, oa_a);
    cb(out_a, "attn_xa_out", il);
    ggml_build_forward_expand(gf, out_a);

    oa_b = ggml_cast(ctx0, oa_b, GGML_TYPE_F32);
    ggml_tensor * out = ggml_add(ctx0, out_a, ggml_mul_mat(ctx0, wo_b_b, oa_b));
    cb(out, "attn_out", il);

    return out;
}

// The first level of the indexer: keep the candidate_topk_blocks blocks with the best score, where the score of a block is the best score of its positions. The result is a block mask [1, n_blocks, n_rows, n_stream] that holds 0 on the kept blocks and -inf on the others, so that a reader adds it to its score with a broadcast and never makes a mask as wide as the stream.
//
// The reference also drops a kept block when its score is -inf, and pads the stream to a whole number of blocks. Neither is necessary here. A block with a -inf score has only positions that the query can not see, and those are -inf in the score of the reader too. The stream is padded to 256 rows, which the block size divides (load_arch_hparams()).
ggml_tensor * llama_model_deepseek41::graph::build_candidate_mask(
        ggml_tensor * score,
        ggml_tensor * cand_pin,
        int il) const {
    if (cand_pin == nullptr) {
        // all blocks fit in the top-k, so the selection keeps all of them
        return nullptr;
    }

    const int64_t n_pos    = score->ne[0];
    const int64_t n_rows   = score->ne[1];
    const int64_t n_stream = score->ne[3];
    const int64_t n_blocks = cand_pin->ne[0];

    GGML_ASSERT(n_pos % n_blocks == 0);
    GGML_ASSERT(score->ne[2] == 1 && cand_pin->ne[1] == n_rows && cand_pin->ne[3] == n_stream);

    const int64_t block  = n_pos/n_blocks;
    const int64_t n_keep = std::min<int64_t>(hparams.dsv41_candidate_topk_blocks, n_blocks);

    // pool_2d with a (block, 1) kernel, because pool_1d has no GPU kernel
    ggml_tensor * block_score = ggml_is_contiguous(score) ? score : ggml_cont(ctx0, score);
    block_score = ggml_pool_2d(ctx0, block_score, GGML_OP_POOL_MAX, (int) block, 1, (int) block, 1, 0.0f, 0.0f);
    cb(block_score, "cand_block_score", il);

    // the block with the newest position of the query is only partly full and an older full block can have a better score, so it is always kept
    block_score = ggml_add(ctx0, block_score, cand_pin);

    ggml_tensor * keep_idx = ggml_cont(ctx0, ggml_top_k(ctx0, block_score, (int) n_keep));
    keep_idx = ggml_reshape_3d(ctx0, keep_idx, n_keep, n_rows, n_stream);
    cb(keep_idx, "cand_keep_idx", il);

    // -inf on all blocks, then 0 on the kept ones
    ggml_tensor * keep  = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blocks, n_rows, n_stream), -INFINITY);
    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_keep,   n_rows, n_stream), 0.0f);
    keep = ggml_set_rows(ctx0, keep, zeros, keep_idx);
    cb(keep, "cand_mask", il);

    return keep;
}

// Score this layer's queries against the shared index keys and keep the best compressed positions.
// The keys were published by an earlier layer, so this only builds the query side.
ggml_tensor * llama_model_deepseek41::graph::build_indexer_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & inp_comp,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor ** shared_cand,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_idx_head      = hparams.indexer_n_head;
    const int64_t n_idx_head_dim  = hparams.indexer_head_size;
    const int64_t n_idx_head_rope = hparams.n_rot();
    const int64_t n_idx_head_nope = n_idx_head_dim - n_idx_head_rope;
    const int64_t nt              = cur->ne[1];

    GGML_ASSERT(inp_comp.kq_mask);
    GGML_ASSERT(n_idx_head_dim >= n_idx_head_rope);

    ggml_tensor * idx_k_rot = inp_dsv4->get_lid().k_rot;

    // one weight per head, scaled so the score matches the reference's
    // softmax_scale * n_heads**-0.5
    ggml_tensor * idx_w = build_lora_mm(layer.indexer_proj, cur);
    idx_w = ggml_scale(ctx0, idx_w, 1.0f/sqrtf(float(n_idx_head_dim*n_idx_head)));
    cb(idx_w, "idx_weights", il);

    ggml_tensor * idx_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= idx_k->ne[2]);

    idx_k = ggml_view_4d(ctx0, idx_k,
            idx_k->ne[0], idx_k->ne[1], n_comp, idx_k->ne[3],
            idx_k->nb[1], idx_k->nb[2], idx_k->nb[3], 0);
    cb(idx_k, "idx_k", il);

    const int64_t n_stream = idx_k->ne[3];
    const int32_t cand_src = hparams.dsv41_candidate_source;

    // the top-k of the query rows [r0, r0 + nr), from their qr and weights
    auto rows_top_k = [&](ggml_tensor * qr_r, ggml_tensor * w_r, ggml_tensor * k_r, int64_t r0, int64_t nr) {
        ggml_tensor * pos_r = nr == nt ? inp_pos : ggml_view_1d(ctx0, inp_pos, nr, r0*ggml_element_size(inp_pos));
        ggml_tensor * idx_q = build_lora_mm(layer.indexer_attn_q_b, qr_r);
        idx_q = ggml_reshape_3d(ctx0, idx_q, n_idx_head_dim, n_idx_head, nr);
        idx_q = ggml_rope_ext(ctx0, idx_q, pos_r, nullptr, n_idx_head_rope, rope_type, n_ctx_orig,
                hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
        idx_q = ggml_rope_set_offset(idx_q, n_idx_head_nope);
        cb(idx_q, "idx_q", il);

        if (idx_k_rot) {
            idx_q = llama_mul_mat_hadamard(ctx0, idx_q, idx_k_rot);
            cb(idx_q, "idx_q_rot", il);
        }

        idx_q = ggml_view_4d(ctx0, idx_q,
                idx_q->ne[0], idx_q->ne[1], idx_q->ne[2]/n_stream, n_stream,
                idx_q->nb[1], idx_q->nb[2], idx_q->nb[3]/n_stream, 0);
        w_r = ggml_view_4d(ctx0, w_r,
                w_r->ne[0], w_r->ne[1]/n_stream, w_r->ne[2], n_stream,
                w_r->nb[1], w_r->nb[2]/n_stream, w_r->nb[3]/n_stream, 0);

        // the stream mask is F16 only when flash attention is on. it only ever holds 0 or -inf, so a
        // cast to the type that the score path needs is exact.
        ggml_tensor * mask = inp_comp.kq_mask;
        if (nr != nt) {
            mask = ggml_view_4d(ctx0, mask, mask->ne[0], nr, mask->ne[2], mask->ne[3],
                    mask->nb[1], mask->nb[2], mask->nb[3], r0*mask->nb[1]);
        }

        ggml_tensor * score = nullptr;
        if (cparams.fused_lid) {
            if (mask->type != GGML_TYPE_F16) {
                mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
            }
            score = ggml_lightning_indexer(ctx0, idx_q, k_r, w_r, mask);
            res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
        } else {
            idx_q = ggml_permute(ctx0, idx_q, 0, 2, 1, 3);
            ggml_tensor * k_p = ggml_permute(ctx0, k_r, 0, 2, 1, 3);

            score = ggml_mul_mat(ctx0, k_p, idx_q);
            score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

            score = ggml_relu(ctx0, score);
            score = ggml_mul(ctx0, score, w_r);
            score = ggml_sum_rows(ctx0, score);
            score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

            if (mask->type != score->type) {
                mask = ggml_cast(ctx0, mask, score->type);
            }
            score = ggml_add(ctx0, score, mask);
        }
        cb(score, "idx_score", il);

        // The candidate source publishes its block mask and does not apply it. The index sources after it add the mask before they pick positions.
        if (cand_src == il) {
            *shared_cand = build_candidate_mask(score, inp_comp.cand_pin, il);
        } else if (cand_src >= 0 && il > cand_src && *shared_cand != nullptr) {
            ggml_tensor * cand = *shared_cand;
            if (nr != nt) {
                cand = ggml_view_4d(ctx0, cand, cand->ne[0], cand->ne[1], nr, cand->ne[3],
                        cand->nb[1], cand->nb[2], cand->nb[3], r0*cand->nb[2]);
            }
            const int64_t n_blocks = cand->ne[1];

            GGML_ASSERT(ggml_is_contiguous(score) && score->ne[2] == 1);
            GGML_ASSERT(score->ne[0] % n_blocks == 0 && score->ne[1] == cand->ne[2] && score->ne[3] == cand->ne[3]);

            score = ggml_reshape_4d(ctx0, score, score->ne[0]/n_blocks, n_blocks, score->ne[1], score->ne[3]);
            score = ggml_add(ctx0, score, cand);
            score = ggml_reshape_4d(ctx0, score, score->ne[0]*n_blocks, score->ne[2], 1, score->ne[3]);
            cb(score, "idx_score_cand", il);
        }

        const uint32_t n_top_k = score->ne[0] < hparams.indexer_top_k ? score->ne[0] : hparams.indexer_top_k;

        ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
        cb(top_k, "idx_top_k", il);

        return top_k;
    };

    // LLAMA_ATTN_SPLIT_INDEXER=1 (with LLAMA_ATTN_SPLIT): the split device scores the second half of the query rows, which
    // is as much work as the first on long prompts, and sends back only their top-k. The candidate source keeps all rows,
    // since it pools over them. Opt-in: the greedy output changes (12.8k-token prompt), although each row should score the
    // same, so this is not validated yet.
    const char * env_idx = getenv("LLAMA_ATTN_SPLIT_INDEXER");
    const bool split = cparams.attn_split_groups > 0 && cparams.moe_split_n > 0 && nt >= (int64_t) cparams.moe_split_min_tokens &&
            n_stream == 1 && cand_src != il && env_idx != nullptr && atoi(env_idx) != 0;
    if (!split) {
        return rows_top_k(qr, idx_w, idx_k, 0, nt);
    }

    const int64_t nr_a = nt/2;
    const int64_t nr_b = nt - nr_a;
    ggml_build_forward_expand(gf, qr);
    ggml_build_forward_expand(gf, idx_w);
    if (*shared_cand != nullptr) {
        ggml_build_forward_expand(gf, *shared_cand);
    }

    int n0 = ggml_graph_n_nodes(gf);
    ggml_tensor * qr_b = ggml_cont(ctx0, ggml_view_2d(ctx0, qr,    qr->ne[0],    nr_b, qr->nb[1],    nr_a*qr->nb[1]));
    ggml_tensor * w_b  = ggml_cont(ctx0, ggml_view_2d(ctx0, idx_w, idx_w->ne[0], nr_b, idx_w->nb[1], nr_a*idx_w->nb[1]));
    ggml_tensor * k_b  = ggml_cont(ctx0, idx_k);
    ggml_build_forward_expand(gf, qr_b);
    ggml_build_forward_expand(gf, w_b);
    ggml_build_forward_expand(gf, k_b);
    ggml_tensor * top_k_b = rows_top_k(qr_b, w_b, k_b, nr_a, nr_b);
    ggml_build_forward_expand(gf, top_k_b);
    pin_split_nodes(n0, true, il);

    n0 = ggml_graph_n_nodes(gf);
    ggml_tensor * qr_a = ggml_view_2d(ctx0, qr,    qr->ne[0],    nr_a, qr->nb[1],    0);
    ggml_tensor * w_a  = ggml_view_2d(ctx0, idx_w, idx_w->ne[0], nr_a, idx_w->nb[1], 0);
    ggml_tensor * top_k_a = rows_top_k(qr_a, w_a, idx_k, 0, nr_a);
    ggml_build_forward_expand(gf, top_k_a);
    pin_split_nodes(n0, false, il);

    ggml_tensor * top_k = ggml_concat(ctx0, top_k_a, top_k_b, 1);
    cb(top_k, "idx_top_k", il);

    return top_k;
}

// Sparse attention on HIP cannot compact a finite mask inside FlashAttention. Decode already has
// the selected compressed rows, so gather them before attention instead. Keep the raw sliding
// window in cache order, then append the selected compressed rows in top-k order. This preserves
// the attention sinks and works with the ordinary FlashAttention kernels.
static bool dsv41_compact_attn_enabled(
        int64_t n_comp,
        int64_t n_selected,
        int64_t n_query,
        int64_t n_stream) {
    const char * env = std::getenv("LLAMA_DSV41_COMPACT_ATTN");
    const int mode = env ? std::atoi(env) : 1;

    if (mode == 0 || n_query != n_stream) {
        return false;
    }

    return mode >= 2 || (n_stream == 1 && n_comp >= 2*n_selected);
}

// DSpark verification issues a few queries per stream, and each query selects different
// compressed rows, so one shared compact K/V tensor is not valid. The batch is tiny, so
// materialize one compact K/V per query instead. Prefill has far too many queries for one
// attention call per query; it stays on the dense fallback.
//
// Mode 3 forces this path wherever the query count fits. Unlike decode there is no
// profitability heuristic: at short context the extra attention calls can cost more than
// they save, while at long context they win by a wide margin. Compare mode 0 against
// mode 3 at the target context length instead.
static bool dsv41_verify_compact_enabled(
        int64_t n_query,
        int64_t n_stream) {
    const char * env = std::getenv("LLAMA_DSV41_COMPACT_ATTN");
    const int mode = env ? std::atoi(env) : 1;

    if (mode < 3 || n_stream < 1 || n_query % n_stream != 0) {
        return false;
    }

    const int64_t n_tps = n_query/n_stream;

    return n_tps >= 2 && n_tps <= 16;
}

// Prefill issues far too many queries for one attention call per query. Pass the
// per-query selections as indices and let FlashAttention dereference the cache rows
// inside the kernel, like build_attn_sparse does. Only the CUDA/HIP indexed kernel
// honors indices fast (the CPU kernel does too); Metal and Vulkan reject such ops and
// fall back to the CPU, so this path is opt-in (mode 4) and stays off by default.
static bool dsv41_prefill_indexed_enabled(
        const llama_hparams & hparams,
        bool flash_attn,
        int64_t n_comp,
        int64_t n_selected,
        int64_t n_query,
        int64_t n_stream) {
    const char * env = std::getenv("LLAMA_DSV41_COMPACT_ATTN");
    const int mode = env ? std::atoi(env) : 1;

    if (mode < 4 || !flash_attn || n_stream < 1 || n_query % n_stream != 0) {
        return false;
    }

    // The indexed kernel takes no alibi bias or logit softcap.
    if (hparams.f_max_alibi_bias != 0.0f || hparams.attn_soft_cap) {
        return false;
    }

    const int64_t n_tps = n_query/n_stream;

    // The indexed kernel is scalar per row while dense prefill uses the MMA tile
    // kernel, so it needs a clear rows-ratio to win. Measured: break-even near
    // n_comp == 2*top_k, clear wins from n_comp == 4*top_k. Compare mode 0 against
    // mode 4 at the target context length instead of lowering this.
    return n_tps > 16 && n_comp >= 4*n_selected;
}

// DeepSeek-V4.1 attention: a sliding window of raw KV, plus, where the layer uses one, the
// compressed positions the indexer picked, concatenated into a single masked attention.
//
// Only a source layer compresses. The layers after it read the same rows, which the KV cache
// hands them through the reuse callback, so a reader builds no compressor at all.
ggml_tensor * llama_model_deepseek41::graph::build_attention_v41(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor ** shared_top_k,
        ggml_tensor ** shared_cand,
        int il,
        ggml_tensor * full_cur,
        uint32_t n_skip) const {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t nt               = cur->ne[1];
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % hparams.dsv4_o_group_count == 0);

    const dsv41_rope_cfg rc = rope_cfg(il);

    ggml_tensor * k_idxs  = inp_attn->get_k_idxs(il);
    ggml_tensor * kq_mask = inp_attn->get_kq_mask(il);

    // Query. V4 normalizes again after wq_b; V4.1 normalizes only the low rank part.
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr", il);

    // LLAMA_ATTN_SPLIT: the queries are built per device in build_attention_split
    const bool split = cparams.attn_split_groups > 0 && cparams.attn_split_groups < (int32_t) hparams.dsv4_o_group_count &&
            cparams.moe_split_n > 0 && nt >= (int64_t) cparams.moe_split_min_tokens;

    ggml_tensor * q = split ? nullptr : build_query(model, qr, inp_pos, 0, n_head, il);

    // the sliding window KV, which every layer keeps for itself
    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
    cb(kv, "kv", il);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * out = nullptr;

    if (ratio == 0 && split) {
        ggml_tensor * k_rot = inp_attn->self_k_rot;
        if (k_rot) {
            kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
        }
        ggml_build_forward_expand(gf, kv);
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, k_idxs, il));

        return build_attention_split(model, qr, inp_attn->mctx->get_k(ctx0, il), k_rot, kq_mask,
                nullptr, nullptr, 0, inp_pos, nt, il);
    }

    if (ratio == 0) {
        // no compressed stream, so this layer sees only its own window
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);

        return build_attention_tail(model, out, inp_pos, nt, il);
    }

    // The plan slot follows the ratio, since a plan encodes how many tokens make a row. The rows
    // themselves always live in the CSA cache, and the index keys in the indexer cache, whichever
    // plan produced them.
    const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

    auto inp_comp = use_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();
    if (n_skip > 0) {
        inp_comp.kq_mask = ggml_view_4d(ctx0, inp_comp.kq_mask,
                inp_comp.kq_mask->ne[0], nt, 1, 1,
                inp_comp.kq_mask->nb[1], inp_comp.kq_mask->nb[2], inp_comp.kq_mask->nb[3],
                n_skip*inp_comp.kq_mask->nb[1]);
        if (inp_comp.cand_pin) {
            inp_comp.cand_pin = ggml_view_4d(ctx0, inp_comp.cand_pin,
                    inp_comp.cand_pin->ne[0], nt, 1, 1,
                    inp_comp.cand_pin->nb[1], inp_comp.cand_pin->nb[2], inp_comp.cand_pin->nb[3],
                    n_skip*inp_comp.cand_pin->nb[1]);
        }
    }

    const llama_dsv4_comp_state * comp_state = use_csa
        ? inp_dsv4->mctx->get_csa_state()
        : inp_dsv4->mctx->get_hca_state();

    GGML_ASSERT(inp_comp.kq_mask && "a compressed layer needs a plan for its ratio");

    if (hparams.dsv41_is_kv_source(il) && inp_comp.state_pos) {
        // CED keeps the full global KV stream while only decoder queries are compacted.
        ggml_tensor * source = full_cur ? full_cur : cur;
        ggml_tensor * state_kv = build_lora_mm(layer.attn_comp_wkv, source);
        cb(state_kv, "comp_state_kv", il);

        // At ratio 1 there is nothing to pool and the file carries no gate. The softmax below
        // then runs over a single element and returns 1.0 whatever the score holds, so the
        // values reach the cache unweighted, which is what a plain projection means.
        ggml_tensor * state_score = layer.attn_comp_wgate
            ? build_lora_mm(layer.attn_comp_wgate, source)
            : state_kv;
        cb(state_score, "comp_state_score", il);

        const dsv4_state_tensors restored = dsv4_build_state_restore(ctx0, inp_comp, comp_state, il);

        ggml_tensor * base_kv = dsv4_view_2d(
                ctx0, restored.kv, restored.kv->ne[0], comp_state->get_n_rows(), 0);
        ggml_tensor * base_score = dsv4_view_2d(
                ctx0, restored.score, restored.score->ne[0], comp_state->get_n_rows(), 0);

        ggml_tensor * source_kv    = ggml_concat(ctx0, base_kv,    state_kv,    1);
        ggml_tensor * source_score = ggml_concat(ctx0, base_score, state_score, 1);

        // the indexer reads the latent before it is rotated, so ask for both forms at once
        ggml_tensor * latent_pre = nullptr;

        ggml_tensor * latent = build_hca_compressed_kv_from_state(
                source_kv,
                source_score,
                inp_comp.state_read_idxs,
                inp_comp.state_write_pos,
                layer.attn_comp_norm,
                ratio,
                n_embd_head,
                "comp_kv",
                il,
                &latent_pre);

        if (hparams.dsv41_owns_index_k(il)) {
            const int64_t n_idx_head_dim  = hparams.indexer_head_size;
            const int64_t n_idx_head_rope = hparams.n_rot();

            ggml_tensor * idx_k = build_lora_mm(layer.indexer_attn_k, latent_pre);
            idx_k = build_norm(idx_k, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
            idx_k = ggml_rope_ext(ctx0, idx_k, inp_comp.state_write_pos, nullptr, n_idx_head_rope,
                    rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                    dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
            idx_k = ggml_rope_set_offset(idx_k, n_idx_head_dim - n_idx_head_rope);
            cb(idx_k, "idx_k_new", il);

            if (inp_dsv4->get_lid().k_rot) {
                idx_k = llama_mul_mat_hadamard(ctx0, idx_k, inp_dsv4->get_lid().k_rot);
            }

            inp_dsv4->mctx->get_lid()->cpy_k_mirrored(ctx0, gf, idx_k, inp_comp.state_write_idxs, il);
        }

        if (inp_dsv4->get_csa().k_rot) {
            latent = llama_mul_mat_hadamard(ctx0, latent, inp_dsv4->get_csa().k_rot);
            cb(latent, "comp_kv_rot", il);
        }

        inp_dsv4->mctx->get_csa()->cpy_k_mirrored(ctx0, gf, latent, inp_comp.state_write_idxs, il);

        // carry whatever did not complete a row into the next ubatch
        ggml_tensor * snapshot_kv    = ggml_concat(ctx0, restored.kv,    state_kv,    1);
        ggml_tensor * snapshot_score = ggml_concat(ctx0, restored.score, state_score, 1);

        const dsv4_state_tensors snapshot = dsv4_build_state_snapshot(
                ctx0, inp_comp, comp_state, snapshot_kv, snapshot_score, il);
        if (snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, snapshot.kv);
        }
        if (snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, snapshot.score);
        }

        ggml_tensor * persist_kv = ggml_get_rows(ctx0, state_kv, inp_comp.state_persist_src_idxs);
        ggml_tensor * persist_score = ggml_get_rows(ctx0, state_score, inp_comp.state_persist_src_idxs);

        ggml_build_forward_expand(gf, comp_state->cpy_kv(
                    ctx0, persist_kv, inp_comp.state_persist_dst_idxs, il));
        ggml_build_forward_expand(gf, comp_state->cpy_score(
                    ctx0, persist_score, inp_comp.state_persist_dst_idxs, il));
    }

    // An index source picks the positions. The layers after it reuse that pick, as the reference
    // passes topk_idxs through a layer without an indexer; attending to every compressed row there
    // is only the same once the stream has no more rows than the top-k.
    ggml_tensor * top_k = nullptr;
    if (hparams.dsv41_is_index_source(il)) {
        top_k = build_indexer_top_k(model, inp_dsv4, inp_comp, qr, cur, inp_pos, shared_cand, il);
        *shared_top_k = top_k;
    } else if (hparams.dsv41_topk_source[il] >= 0) {
        GGML_ASSERT(*shared_top_k != nullptr && "the top-k source must build before its readers");
        top_k = *shared_top_k;
    }

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        if (q) {
            q = llama_mul_mat_hadamard(ctx0, q, k_rot);
        }
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    if (q) {
        ggml_build_forward_expand(gf, q);
    }
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, k_idxs, il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "raw_k", il);

    ggml_tensor * comp_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= comp_k->ne[2]);

    comp_k = ggml_view_4d(ctx0, comp_k,
            comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
            comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
    cb(comp_k, "comp_k", il);

    ggml_tensor * raw_mask = kq_mask;
    const int64_t n_stream = raw_k->ne[3];
    const bool use_compact = !split && top_k != nullptr && cparams.flash_attn &&
            dsv41_compact_attn_enabled(n_comp, top_k->ne[0], nt, n_stream);
    const bool use_verify_compact = !split && !use_compact && top_k != nullptr && cparams.flash_attn &&
            dsv41_verify_compact_enabled(nt, n_stream);
    const bool use_prefill_indexed = !split && !use_compact && !use_verify_compact && top_k != nullptr &&
            dsv41_prefill_indexed_enabled(hparams, cparams.flash_attn, n_comp, top_k->ne[0], nt, n_stream);

    if (use_compact) {
        const int64_t n_raw = raw_k->ne[2];
        const int64_t n_selected = top_k->ne[0];
        const int64_t n_compact = n_raw + n_selected;

        GGML_ASSERT(raw_mask->ne[0] == n_raw && raw_mask->ne[1] == 1 &&
                raw_mask->ne[2] == 1 && raw_mask->ne[3] == n_stream);
        GGML_ASSERT(top_k->ne[1] == 1 && top_k->ne[2] == 1 && top_k->ne[3] == n_stream);

        // ggml_get_rows addresses dimension 1, while the KV cache stores rows in
        // dimension 2. The selected compressed rows are query-specific in general;
        // this decode path has one query per stream, so one compact K/V tensor is valid.
        ggml_tensor * top_k_rows = ggml_reshape_2d(ctx0, top_k, n_selected, n_stream);
        ggml_tensor * comp_rows = ggml_view_3d(ctx0, comp_k,
                comp_k->ne[0], n_comp, n_stream, comp_k->nb[2], comp_k->nb[3], 0);
        ggml_tensor * comp_selected = ggml_get_rows(ctx0, comp_rows, top_k_rows);
        comp_selected = ggml_reshape_4d(ctx0, comp_selected,
                comp_k->ne[0], 1, n_selected, n_stream);
        if (comp_selected->type != comp_k->type) {
            comp_selected = ggml_cast(ctx0, comp_selected, comp_k->type);
        }
        cb(comp_selected, "comp_k_selected", il);

        ggml_tensor * compact_k = ggml_concat(ctx0, raw_k, comp_selected, 2);
        cb(compact_k, "k_compact", il);

        // Top-k can contain masked fill entries when a stream has fewer compressed
        // positions than the configured width. Gather the original validity mask so
        // those entries cannot contribute to the compact attention result.
        ggml_tensor * mask_rows = ggml_reshape_4d(ctx0, inp_comp.kq_mask,
                1, n_comp, 1, n_stream);
        ggml_tensor * comp_mask = ggml_get_rows(ctx0, mask_rows, top_k_rows);
        comp_mask = ggml_reshape_4d(ctx0, comp_mask, n_selected, 1, 1, n_stream);
        if (comp_mask->type != raw_mask->type) {
            comp_mask = ggml_cast(ctx0, comp_mask, raw_mask->type);
        }

        ggml_tensor * compact_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
        cb(compact_mask, "kq_compact_mask", il);

        out = build_attn_mha(q, compact_k, compact_k, nullptr, compact_mask, layer.attn_sinks,
                nullptr, n_compact, kq_scale, il);
    } else if (use_verify_compact) {
        // Each query selects different compressed rows, so materialize one compact K/V
        // per query with the decode gather pattern and run one ordinary attention per
        // query. The raw window is shared per stream; only the selected rows differ.
        // Per-query causality already lives in the per-query mask columns, and the
        // fill-entry validity rule from the decode path applies unchanged.
        const int64_t n_raw = raw_k->ne[2];
        const int64_t n_selected = top_k->ne[0];
        const int64_t n_tps = nt/n_stream;
        const int64_t n_compact = n_raw + n_selected;

        GGML_ASSERT(nt == n_tps*n_stream);
        GGML_ASSERT(top_k->ne[1] == n_tps && top_k->ne[2] == 1 && top_k->ne[3] == n_stream);
        GGML_ASSERT(raw_mask->ne[0] == n_raw && raw_mask->ne[1] == n_tps &&
                raw_mask->ne[2] == 1 && raw_mask->ne[3] == n_stream);
        GGML_ASSERT(inp_comp.kq_mask->ne[0] == n_comp && inp_comp.kq_mask->ne[1] == n_tps &&
                inp_comp.kq_mask->ne[2] == 1 && inp_comp.kq_mask->ne[3] == n_stream);

        ggml_tensor * top_k_flat = ggml_reshape_2d(ctx0, top_k, n_selected, nt);

        out = nullptr;
        for (int64_t t = 0; t < nt; ++t) {
            const int64_t s  = t/n_tps;
            const int64_t qi = t % n_tps;

            ggml_tensor * q_t = ggml_view_3d(ctx0, q,
                    q->ne[0], q->ne[1], 1, q->nb[1], q->nb[2], t*q->nb[2]);
            ggml_tensor * raw_k_s = ggml_view_4d(ctx0, raw_k,
                    raw_k->ne[0], raw_k->ne[1], n_raw, 1,
                    raw_k->nb[1], raw_k->nb[2], raw_k->nb[3], s*raw_k->nb[3]);

            ggml_tensor * top_k_col = ggml_view_2d(ctx0, top_k_flat,
                    n_selected, 1, top_k_flat->nb[1], t*top_k_flat->nb[1]);

            ggml_tensor * comp_rows = ggml_view_3d(ctx0, comp_k,
                    comp_k->ne[0], n_comp, 1, comp_k->nb[2], comp_k->nb[3], s*comp_k->nb[3]);
            ggml_tensor * comp_selected = ggml_get_rows(ctx0, comp_rows, top_k_col);
            comp_selected = ggml_reshape_4d(ctx0, comp_selected,
                    comp_k->ne[0], 1, n_selected, 1);
            if (comp_selected->type != comp_k->type) {
                comp_selected = ggml_cast(ctx0, comp_selected, comp_k->type);
            }

            ggml_tensor * compact_k = ggml_concat(ctx0, raw_k_s, comp_selected, 2);

            ggml_tensor * raw_mask_qi = ggml_view_4d(ctx0, raw_mask,
                    n_raw, 1, 1, 1,
                    raw_mask->nb[1], raw_mask->nb[2], raw_mask->nb[3],
                    qi*raw_mask->nb[1] + s*raw_mask->nb[3]);

            // The selection addresses the compressed rows in mask dimension 0, so
            // stride the view with nb[0] to gather them through dimension 1.
            ggml_tensor * mask_col = ggml_view_4d(ctx0, inp_comp.kq_mask,
                    1, n_comp, 1, 1,
                    inp_comp.kq_mask->nb[0], inp_comp.kq_mask->nb[2], inp_comp.kq_mask->nb[3],
                    qi*inp_comp.kq_mask->nb[1] + s*inp_comp.kq_mask->nb[3]);
            ggml_tensor * comp_mask = ggml_get_rows(ctx0, mask_col, top_k_col);
            comp_mask = ggml_reshape_4d(ctx0, comp_mask, n_selected, 1, 1, 1);
            if (comp_mask->type != raw_mask->type) {
                comp_mask = ggml_cast(ctx0, comp_mask, raw_mask->type);
            }

            ggml_tensor * compact_mask = ggml_concat(ctx0, raw_mask_qi, comp_mask, 0);

            ggml_tensor * out_t = build_attn_mha(q_t, compact_k, compact_k, nullptr, compact_mask,
                    layer.attn_sinks, nullptr, n_compact, kq_scale, il);
            char out_name[64];
            snprintf(out_name, sizeof(out_name), "attn_out_verify_q%" PRId64, t);
            cb(out_t, out_name, il);

            out = out == nullptr ? out_t : ggml_concat(ctx0, out, out_t, 1);
        }
    } else if (use_prefill_indexed) {
        // One kernel dereferences every query's rows through indices. The compressed
        // rows come first so the per-query selections need no offset; the raw window
        // follows, identical for every query, so build its indices once with arange
        // and repeat. The validity mask carries the gathered compressed entries first,
        // then the raw mask with its per-query causality.
        const int64_t n_raw = raw_k->ne[2];
        const int64_t n_selected = top_k->ne[0];
        const int64_t n_tps = nt/n_stream;
        const int64_t n_compact = n_raw + n_selected;

        GGML_ASSERT(nt == n_tps*n_stream);
        GGML_ASSERT(top_k->ne[1] == n_tps && top_k->ne[2] == 1 && top_k->ne[3] == n_stream);
        GGML_ASSERT(raw_mask->ne[0] == n_raw && raw_mask->ne[1] == n_tps &&
                raw_mask->ne[2] == 1 && raw_mask->ne[3] == n_stream);
        GGML_ASSERT(inp_comp.kq_mask->ne[0] == n_comp && inp_comp.kq_mask->ne[1] == n_tps &&
                inp_comp.kq_mask->ne[2] == 1 && inp_comp.kq_mask->ne[3] == n_stream);
        GGML_ASSERT(comp_k->type == raw_k->type &&
                (comp_k->type == GGML_TYPE_F16 || comp_k->type == GGML_TYPE_Q8_0));

        ggml_tensor * k_all = ggml_concat(ctx0, comp_k, raw_k, 2);
        cb(k_all, "k_all_indexed", il);

        ggml_tensor * raw_idx = ggml_arange(ctx0, (float) n_comp, (float) (n_comp + n_raw), 1.0f);
        raw_idx = ggml_cast(ctx0, raw_idx, GGML_TYPE_I32);
        raw_idx = ggml_reshape_3d(ctx0, raw_idx, n_raw, 1, 1);
        ggml_tensor * raw_idx_tpl = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, n_raw, n_tps, n_stream);
        raw_idx = ggml_repeat(ctx0, raw_idx, raw_idx_tpl);
        cb(raw_idx, "raw_idx", il);

        ggml_tensor * comp_idx = ggml_reshape_3d(ctx0, top_k, n_selected, n_tps, n_stream);

        ggml_tensor * kv_indices = ggml_concat(ctx0, comp_idx, raw_idx, 0);
        cb(kv_indices, "kq_indices", il);

        ggml_tensor * mask_rows = ggml_reshape_3d(ctx0, inp_comp.kq_mask, 1, n_comp, nt);
        ggml_tensor * top_k_2d = ggml_reshape_2d(ctx0, top_k, n_selected, nt);
        ggml_tensor * comp_valid = ggml_get_rows(ctx0, mask_rows, top_k_2d);
        comp_valid = ggml_reshape_4d(ctx0, comp_valid, n_selected, n_tps, 1, n_stream);
        if (comp_valid->type != raw_mask->type) {
            comp_valid = ggml_cast(ctx0, comp_valid, raw_mask->type);
        }

        ggml_tensor * indexed_mask = ggml_concat(ctx0, comp_valid, raw_mask, 0);
        cb(indexed_mask, "kq_indexed_mask", il);

        out = build_attn_mha(q, k_all, k_all, nullptr, indexed_mask, layer.attn_sinks,
                nullptr, n_compact, kq_scale, il, kv_indices);
    } else {
        ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
        cb(k_all, "k_all", il);

        ggml_tensor * comp_mask = top_k
            ? build_top_k_mask(inp_comp.kq_mask, top_k, "comp_top_k_mask", il)
            : inp_comp.kq_mask;

        ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
        cb(kq_mask, "kq_mask", il);

        const int64_t n_kv_max = top_k
            ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + top_k->ne[0]
            : 0;

        if (split) {
            return build_attention_split(model, qr, k_all, k_rot, raw_mask, inp_comp.kq_mask, top_k, n_kv_max, inp_pos, nt, il);
        }

        out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks,
                nullptr, n_kv_max, kq_scale, il);
    }
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_out_raw", il);

    return build_attention_tail(model, out, inp_pos, nt, il);
}

llama_model_deepseek41::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    const auto ubatch_dec = inp_attn->mctx->decoder_ubatch(ubatch);
    const uint32_t n_skip = ubatch.n_tokens - ubatch_dec.n_tokens;
    ggml_tensor * inp_out_ids = build_inp_out_ids(n_skip);
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // Layer 0 has no previous sublayer to take a mix from, so the reference hands it a one-hot
    // that selects the first copy.
    ggml_tensor * pre_mix = ggml_concat(ctx0,
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1,      n_tokens), 1.0f),
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f), 0);
    cb(pre_mix, "hc_pre_init", -1);

    ggml_tensor * shared_top_k = nullptr;
    ggml_tensor * shared_cand  = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * full_cur = nullptr;
        if (il == (int) cparams.n_layer_prefill && n_skip > 0) {
            // Adapt DeepInfer's consumer boundary: retain full normalized inputs for
            // the global-KV projection, then compact the decoder trunk and its mix.
            full_cur = build_hc_pre(inpL, pre_mix, il);
            full_cur = build_norm(full_cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(full_cur, "ced_full_norm", il);
            inpL = ggml_cont(ctx0, ggml_view_3d(ctx0, inpL, n_embd, hc, ubatch_dec.n_tokens,
                    inpL->nb[1], inpL->nb[2], n_skip*inpL->nb[2]));
            pre_mix = ggml_cont(ctx0, ggml_view_2d(ctx0, pre_mix, hc, ubatch_dec.n_tokens,
                    pre_mix->nb[1], n_skip*pre_mix->nb[1]));
            inp_pos = ggml_view_1d(ctx0, inp_pos, ubatch_dec.n_tokens, n_skip*ggml_element_size(inp_pos));
        }
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv41_hc_mean(ctx0, inpL);
            if (n_skip > 0 && il >= (int) cparams.n_layer_prefill) {
                // The host feature API keeps batch row indices. DSpark only retains
                // the suffix; pad discarded rows so it cannot read stale features.
                auto * pad = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_skip), 0.0f);
                res->t_layer_inp[il] = ggml_concat(ctx0, pad, res->t_layer_inp[il], 1);
            }
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        // the engram sits before the block and writes straight into the stream
        // an embd batch is an mtmd image span: the reference shuts the engram gate for those positions
        if (pmodel.engram_index(il) >= 0 && ubatch.embd == nullptr) {
            inpL = build_engram(model, inpL, build_inp_engram(model, il), il);
            cb(inpL, "engram_out", il);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * attn_pre = nullptr;
        ggml_tensor * post     = nullptr;
        ggml_tensor * comb     = nullptr;

        // this sublayer's mixes are for the next one, so the collapse uses the incoming mix
        build_hc_mixes(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &attn_pre, &post, &comb, il);

        cur = build_hc_pre(inpL, pre_mix, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention_v41(model, inp_dsv4, cur, inp_pos, &shared_top_k, &shared_cand, il,
                full_cur, il >= (int) cparams.n_layer_prefill ? n_skip : 0);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;

        // the FFN mix is what the next layer's attention collapses with
        build_hc_mixes(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &pre_mix, &post, &comb, il);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl if the input is from mtmd
        if (ubatch.embd != nullptr && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        ggml_tensor * ffn_shexp = nullptr;
        const auto build_shared = [&] {
            ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp, nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);
        };
        static const bool shared_overlap = [] {
            const char * env = std::getenv("LLAMA_MOE_SHARED_OVERLAP");
            return env == nullptr || std::atoi(env) != 0;
        }();
        std::function<void()> before_cpu_ffn;
        if (shared_overlap && expert_cache && expert_cache->layer(il) && cur->ne[1] <= llama_expert_cache::max_rows) {
            auto * home = expert_cache->backend(il);
            const auto on_home = [&](const ggml_tensor * weight) {
                return weight && weight->buffer && !ggml_backend_buffer_is_host(weight->buffer) &&
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(weight->buffer)) == ggml_backend_get_device(home);
            };
            if (on_home(layer.ffn_gate_shexp) && on_home(layer.ffn_up_shexp) && on_home(layer.ffn_down_shexp)) {
                before_cpu_ffn = [&, home] {
                    build_shared();
                    const int n0 = ggml_graph_n_nodes(gf);
                    ggml_build_forward_expand(gf, ffn_shexp);
                    // Packet/ID nodes precede this branch. Keep its supported ops on the layer GPU.
                    for (int i = n0; i < ggml_graph_n_nodes(gf); ++i) {
                        auto * node = ggml_graph_node(gf, i);
                        if (ggml_backend_supports_op(home, node)) {
                            ggml_backend_sched_set_tensor_backend(sched, node, home);
                        }
                    }
                };
            }
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                /* probs_in            */ nullptr,
                /* gate_up_exps        */ nullptr,
                /* up_exps_s           */ nullptr,
                /* gate_exps_s         */ nullptr,
                /* down_exps_s         */ nullptr,
                /* selected_experts_in */ nullptr,
                /* post_exp_norm       */ nullptr,
                /* post_exp_norm_eps   */ 1e-5f,
                before_cpu_ffn);
        cb(moe_out, "ffn_moe_out", il);

        if (!ffn_shexp) {
            build_shared();
        }

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv41_hc_mean(ctx0, inpL);
        if (n_skip > 0) {
            auto * pad = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_skip), 0.0f);
            res->t_layer_inp[n_layer] = ggml_concat(ctx0, pad, res->t_layer_inp[n_layer], 1);
        }
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, inpL->ne[2]);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
    }

    // The last layer's FFN mix is the one nothing has consumed, and it collapses the copies here.
    // This is what a learned hyper-connection head does in V4, which is why this model has none.
    cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
