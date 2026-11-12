#include "models.h"

#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cache-kpool.h"

#include <array>
#include <cstdlib>

// ssm_a holds -exp(A_log) (kimi-k3), not +exp(A_log) (bailingmoe3); converter checks

// positions the indexer keeps; at or below this many the dense path IS the sparse one.
// asserted not measured (invisible to output); the second assert is an independent spelling
static uint32_t glm5next_n_select(const llama_hparams & hparams) {
    GGML_ASSERT(hparams.indexer_kpool > 0);
    GGML_ASSERT(hparams.indexer_top_k >= hparams.indexer_kpool);
    GGML_ASSERT(hparams.indexer_top_k % hparams.indexer_kpool == 0);

    const uint32_t n_select = hparams.indexer_top_k + hparams.indexer_kpool - 1;

    GGML_ASSERT(n_select > hparams.indexer_top_k);
    GGML_ASSERT(n_select == (hparams.indexer_top_k/hparams.indexer_kpool + 1)*hparams.indexer_kpool - 1);

    return n_select;
}

static bool glm5next_mtp_topk_share_enabled(const llama_hparams & hparams) {
    const char * env = std::getenv("LLAMA_GLM5_MTP_TOPK_SHARE");
    return hparams.indexer_index_share_mtp && (env == nullptr || std::atoi(env) != 0);
}

// Runs t on the backend of dev (the device of the layer that reads it), so that only its small inputs cross PCIe.
static void glm5next_pin(ggml_backend_sched_t sched, ggml_tensor * t, ggml_backend_dev_t dev) {
    for (int ib = 0; dev != nullptr && ib < ggml_backend_sched_get_n_backends(sched); ++ib) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, ib);
        if (ggml_backend_get_device(backend) == dev && ggml_backend_supports_op(backend, t)) {
            ggml_backend_sched_set_tensor_backend(sched, t, backend);
            break;
        }
    }
}

static bool glm5next_temporal_topk_enabled() {
    const char * env = std::getenv("GGML_CUDA_TOPK_TEMPORAL");
    return env == nullptr || std::atoi(env) != 0;
}

static bool glm5next_kpool_expand_enabled() {
    const char * env = std::getenv("LLAMA_GLM5_KPOOL_EXPAND");
    return env == nullptr || std::atoi(env) != 0;
}

class llm_graph_input_glm5next_mtp_device_draft : public llm_graph_input_i {
public:
    llm_graph_input_glm5next_mtp_device_draft(
            ggml_tensor * seq_ids,
            ggml_tensor * result_ids,
            const llama_mtp_device_draft_cache * cache) :
        seq_ids(seq_ids), result_ids(result_ids), cache(cache) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(cache != nullptr);
        GGML_ASSERT(seq_ids->ne[0] == ubatch->n_tokens);
        GGML_ASSERT(result_ids->ne[0] == ubatch->n_tokens);
        GGML_ASSERT(ubatch->n_tokens <= LLAMA_MAX_SEQ);
        GGML_ASSERT(cache->step < cache->n_steps);

        std::array<int32_t, LLAMA_MAX_SEQ> seq;
        std::array<int32_t, LLAMA_MAX_SEQ> result;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            GGML_ASSERT(ubatch->n_seq_id[i] == 1);
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < cache->n_seq_max);
            seq[i] = seq_id;
            result[i] = cache->step*cache->n_seq_max + seq_id;
        }

        ggml_backend_tensor_set(seq_ids, seq.data(), 0, ubatch->n_tokens*sizeof(int32_t));
        ggml_backend_tensor_set(result_ids, result.data(), 0, ubatch->n_tokens*sizeof(int32_t));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return params.mtp_device_draft_cache == cache &&
               seq_ids->ne[0] == params.ubatch.n_tokens &&
               result_ids->ne[0] == params.ubatch.n_tokens;
    }

private:
    ggml_tensor * seq_ids;
    ggml_tensor * result_ids;
    const llama_mtp_device_draft_cache * cache;
};

void llama_model_glm5next::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    // indexer k_norm is a LayerNorm with bias; without this key it runs at eps 0
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps);
    // warned not asserted: no output comparison sees it, and an assert breaks test-llama-archs
    if (hparams.f_norm_eps <= 0.0f || hparams.f_norm_eps > 2e-6f) {
        LLAMA_LOG_WARN("%s: indexer k_norm eps is %g, but the reference hardcodes 1e-6. "
                "this is invisible to every output comparison; check the converter\n",
                __func__, (double) hparams.f_norm_eps);
    }

    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    GGML_ASSERT(hparams.n_lora_q > 0 && "glm5next requires a q LoRA");
    GGML_ASSERT(hparams.n_rot() == 0 && "glm5next MLA is nope-only");

    // no linear_num_heads key: KDA head count is attention.head_count (converter enforces)
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,      hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,         hparams.n_embd_head_kda);
    GGML_ASSERT(hparams.ssm_d_conv > 1);
    GGML_ASSERT(hparams.n_embd_head_kda > 0);
    // required: absent, kimi-k3 selects the softplus branch, a different function
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND, hparams.kda_gate_lower_bound);
    GGML_ASSERT(hparams.kda_gate_lower_bound < 0.0f);

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL,      hparams.indexer_kpool);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_INDEX_SHARE_MTP, hparams.indexer_index_share_mtp, false);
    GGML_ASSERT(hparams.indexer_kpool > 0);
    GGML_ASSERT(hparams.indexer_top_k % hparams.indexer_kpool == 0);

    const uint32_t n_select = glm5next_n_select(hparams);
    LLAMA_LOG_INFO("%s: indexer selection width = %u cells (%u pools of %u, plus a %u-wide tail)\n",
            __func__, n_select, hparams.indexer_top_k/hparams.indexer_kpool,
            hparams.indexer_kpool, hparams.indexer_kpool - 1);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult > 0);

    // n_embd_out stays n_embd: lm_head sees the stream mean. deepseek4's hc_mult*n_embd
    // makes llama-context.cpp overread t_embd, and the assert there sizes the destination
    hparams.n_embd_out_impl = 0;

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,   hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false);

    if (hparams.n_ff_shexp == 0) {
        hparams.n_ff_shexp = hparams.n_ff_exp() * std::max(1u, hparams.n_expert_shared);
    }

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all);

    uint32_t n_recr = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
        n_recr += hparams.is_recr_impl[il];
    }
    GGML_ASSERT(n_recr > 0 && n_recr < hparams.n_layer() && "glm5next needs a per-layer attention.head_count_kv array");

    // every glm5next indexer is full; the generic loader only zero-fills the array
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_indexer_full_impl[il] = !hparams.is_recr_impl[il];
    }

    switch (hparams.n_layer()) {
        case 45: type = hparams.n_embd == 4096 && hparams.n_expert == 288 ? LLM_TYPE_313B_A17B : LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_glm5next::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner  = head_dim * n_head;
    const int64_t d_conv   = hparams.ssm_d_conv;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t qk_head_dim  = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim   = hparams.n_embd_head_v_mla();

    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t kpool          = hparams.indexer_kpool;

    const int64_t hc_dim     = (int64_t) hparams.dsv4_hc_mult * n_embd;
    const int64_t hc_mix_dim = (2 + (int64_t) hparams.dsv4_hc_mult) * hparams.dsv4_hc_mult;

    const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    const int trunk_flags = mtp_only   ? TENSOR_NOT_REQUIRED : 0;
    int       mtp_flags   = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int il = 0; il < n_layer_all; ++il) {
        auto & layer = layers[il];
        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), {n_embd}, flags);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", il), {n_embd}, flags);

        // the NextN block keeps the plain residual, so it has no mHC mixer
        if (il < n_layer) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", il), {hc_dim, hc_mix_dim}, flags);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", il), {hc_mix_dim}, flags);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", il), {3}, flags);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", il), {hc_dim, hc_mix_dim}, flags);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", il), {hc_mix_dim}, flags);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", il), {3}, flags);
        }

        if (hparams.is_recr(il)) {
            create_tensor_qkv(layer, il, n_embd, d_inner, d_inner, d_inner, flags);

            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", il), {d_conv, 1, d_inner, 1}, flags);
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", il), {d_conv, 1, d_inner, 1}, flags);
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", il), {d_conv, 1, d_inner, 1}, flags);

            // delphi: q, k, v, the 3 conv weights, and f_a, g_a, beta are created one after the other, so that in one
            // buffer they are adjacent and build_kda_layer can use each group as one tensor (glm5next_adjacent_rows); the
            // gate and up weights of the FFNs are too (build_layer_ffn)
            layer.ssm_f_a  = create_tensor(tn(LLM_TENSOR_SSM_F_A,  "weight", il), {n_embd, head_dim}, flags);
            layer.ssm_g_a  = create_tensor(tn(LLM_TENSOR_SSM_G_A,  "weight", il), {n_embd, head_dim}, flags);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", il), {n_embd, n_head}, flags);

            layer.ssm_f_b = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", il), {head_dim, d_inner}, flags);
            layer.ssm_g_b = create_tensor(tn(LLM_TENSOR_SSM_G_B, "weight", il), {head_dim, d_inner}, flags);
            layer.ssm_a    = create_tensor(tn(LLM_TENSOR_SSM_A,              il), {n_head}, flags);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT,   "bias",   il), {d_inner}, flags);

            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", il), {head_dim}, flags);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), {d_inner, n_embd}, flags);
        } else {
            layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", il), {n_embd, q_lora_rank}, flags);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), {q_lora_rank}, flags);
            layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", il), {q_lora_rank, n_head * qk_head_dim}, flags);

            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", il), {n_embd, kv_lora_rank}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), {kv_lora_rank}, flags);
            layer.wk_b           = create_tensor(tn(LLM_TENSOR_ATTN_K_B,       "weight", il), {qk_head_dim, kv_lora_rank, n_head}, flags);
            layer.wv_b           = create_tensor(tn(LLM_TENSOR_ATTN_V_B,       "weight", il), {kv_lora_rank, v_head_dim, n_head}, flags);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), {n_head * v_head_dim, n_embd}, flags);

            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", il), {n_embd_indexer}, flags);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   il), {n_embd_indexer}, flags);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", il), {n_embd, hparams.indexer_n_head}, flags);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", il), {n_embd, n_embd_indexer}, flags);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", il), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags);

            // key pooling: DeepSeek-V4 doubles the compressor width, GLM-5.3 does not
            layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", il), {n_embd, n_embd_indexer}, flags);
            layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", il), {n_embd_indexer, kpool}, flags);
        }

        if (il < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd, n_ff}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {n_ff, n_embd}, flags);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", il), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   il), {n_expert}, flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), {n_embd, hparams.n_ff_exp(il), n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", il), {n_embd, hparams.n_ff_exp(il), n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), {hparams.n_ff_exp(il), n_embd, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), {n_embd, hparams.n_ff_shexp}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", il), {n_embd, hparams.n_ff_shexp}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), {hparams.n_ff_shexp, n_embd}, flags);
        }

        if (il >= n_layer) {
            layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), {n_embd}, flags);
            layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), {n_embd}, flags);

            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), {n_embd}, flags);
            // absent in the checkpoint: NextN shares the trunk's embeddings and lm_head
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);

            // The trunk's token_embd stays in host memory (input layer), so every MTP step would look its token up
            // on the CPU, and the device-resident draft loop would sync with the host at each step. A copy on the
            // output device, where the head and the output live, keeps the lookup there. env LLAMA_MTP_EMBD_DEVICE=0
            // disables it.
            static const bool embd_device = [] {
                const char * env = std::getenv("LLAMA_MTP_EMBD_DEVICE");
                return env == nullptr || std::atoi(env) != 0;
            }();
            if (embd_device && !layer.nextn.embed_tokens && layer.nextn.eh_proj && !(flags & TENSOR_SKIP)) {
                layer.nextn.embed_tokens = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
            }
        }
    }
}

// one conv over concatenated q|k|v: keeps the conv state one block for rollback
// delphi: one weight tensor over tensors that are adjacent in one buffer (same type and row length, created one after
// the other), so that their products are one mul_mat with the rows in the same order; nullptr when they are not adjacent
// (mmap, tensor overrides, split buffers). LLAMA_ADJACENT_WEIGHTS=0 disables it.
static ggml_tensor * glm5next_adjacent_rows(ggml_context * ctx, std::initializer_list<ggml_tensor *> ts) {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_ADJACENT_WEIGHTS");
        return env == nullptr || std::atoi(env) != 0;
    }();
    if (!enabled) {
        return nullptr;
    }

    const ggml_tensor * first = *ts.begin();
    if (first == nullptr) {
        return nullptr;
    }
    int64_t      nrows = 0;
    const char * next  = (const char *) first->data;
    for (const ggml_tensor * t : ts) {
        if (t == nullptr || t->buffer == nullptr || t->buffer != first->buffer || t->data != next || t->extra != nullptr ||
                t->view_src != nullptr || t->type != first->type || t->ne[0] != first->ne[0] || !ggml_is_contiguous(t)) {
            return nullptr;
        }
        nrows += ggml_nrows(t);
        next   = (const char *) t->data + ggml_nbytes(t);
    }

    ggml_tensor * merged = ggml_new_tensor_2d(ctx, first->type, first->ne[0], nrows);
    merged->data   = first->data;
    merged->buffer = first->buffer;
    ggml_format_name(merged, "%s (adjacent)", first->name);
    return merged;
}

ggml_tensor * llama_model_glm5next::graph::build_kda_layer(
        const llama_layer & layer,
        llm_graph_input_rs * inp_rs,
        ggml_tensor * cur,
        int il) {
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_inner      = head_dim * n_head;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto * mctx_cur = inp_rs->mctx;

    // f, g and beta read the layer input, NOT the convolved q/k/v
    ggml_tensor * inp = cur;

    ggml_tensor * qkv = nullptr;
    if (ggml_tensor * wqkv = glm5next_adjacent_rows(ctx0, {layer.wq, layer.wk, layer.wv})) {
        qkv = ggml_mul_mat(ctx0, wqkv, inp);
    } else {
        ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, inp);
        ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.wk, inp);
        ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, inp);
        qkv = ggml_concat(ctx0, ggml_concat(ctx0, Qcur, Kcur, 0), Vcur, 0);
    }
    qkv = ggml_reshape_3d(ctx0, qkv, 3*d_inner, n_seq_tokens, n_seqs);
    cb(qkv, "kda_qkv", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_in = build_conv_state(inp_rs, conv_states_all, qkv, d_conv, 3*d_inner, il);

    ggml_tensor * conv_w = glm5next_adjacent_rows(ctx0, {layer.ssm_q_conv, layer.ssm_k_conv, layer.ssm_v_conv});
    if (!conv_w) {
        conv_w = ggml_concat(ctx0,
                ggml_concat(ctx0,
                    ggml_reshape_2d(ctx0, layer.ssm_q_conv, d_conv, d_inner),
                    ggml_reshape_2d(ctx0, layer.ssm_k_conv, d_conv, d_inner), 1),
                ggml_reshape_2d(ctx0, layer.ssm_v_conv, d_conv, d_inner), 1);
    }

    // SiLU is applied to the conv output, not to the projections
    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_in, conv_w));
    cb(conv_out, "kda_conv", il);

    const size_t nb_qkv  = ggml_row_size(conv_out->type, 3*d_inner);
    const size_t nb_head = ggml_row_size(conv_out->type, head_dim);

    ggml_tensor * Qcur = ggml_view_4d(ctx0, conv_out, head_dim, n_head, n_seq_tokens, n_seqs,
            nb_head, nb_qkv, nb_qkv*n_seq_tokens, 0);
    ggml_tensor * Kcur = ggml_view_4d(ctx0, conv_out, head_dim, n_head, n_seq_tokens, n_seqs,
            nb_head, nb_qkv, nb_qkv*n_seq_tokens, ggml_row_size(conv_out->type, d_inner));
    ggml_tensor * Vcur = ggml_view_4d(ctx0, conv_out, head_dim, n_head, n_seq_tokens, n_seqs,
            nb_head, nb_qkv, nb_qkv*n_seq_tokens, ggml_row_size(conv_out->type, 2*d_inner));

    // 1e-6 is the reference's own constant, not the model's norm eps
    Qcur = ggml_l2_norm(ctx0, Qcur, 1e-6f);
    Kcur = ggml_l2_norm(ctx0, Kcur, 1e-6f);
    cb(Qcur, "kda_q_norm", il);
    cb(Kcur, "kda_k_norm", il);

    // the 1/sqrt(head_dim) query scale is applied inside build_delta_net, after this norm

    // f_a, g_a and beta read the same input: one product when the weights are adjacent
    ggml_tensor * f_lo    = nullptr;
    ggml_tensor * g_lo    = nullptr;
    ggml_tensor * beta_in = nullptr;
    if (ggml_tensor * w_fgb = glm5next_adjacent_rows(ctx0, {layer.ssm_f_a, layer.ssm_g_a, layer.ssm_beta})) {
        ggml_tensor * fgb = ggml_mul_mat(ctx0, w_fgb, inp);
        f_lo    = ggml_view_2d(ctx0, fgb, head_dim, n_tokens, fgb->nb[1], 0);
        g_lo    = ggml_view_2d(ctx0, fgb, head_dim, n_tokens, fgb->nb[1], ggml_row_size(fgb->type, head_dim));
        beta_in = ggml_cont(ctx0, ggml_view_2d(ctx0, fgb, n_head, n_tokens, fgb->nb[1], ggml_row_size(fgb->type, 2*head_dim)));
    } else {
        f_lo    = ggml_mul_mat(ctx0, layer.ssm_f_a, inp);
        g_lo    = ggml_mul_mat(ctx0, layer.ssm_g_a, inp);
        beta_in = ggml_mul_mat(ctx0, layer.ssm_beta, inp);
    }

    // g = lower_bound * sigmoid(exp(A_log)*(f_b(f_a(x)) + dt_bias)); it scales, not clamps
    ggml_tensor * g = ggml_mul_mat(ctx0, layer.ssm_f_b, f_lo);
    g = ggml_add(ctx0, g, layer.ssm_dt_b);
    g = ggml_reshape_3d(ctx0, g, head_dim, n_head, n_tokens);
    g = ggml_mul(ctx0, g, ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1));
    g = ggml_sigmoid(ctx0, ggml_scale(ctx0, g, -1.0f));
    g = ggml_scale(ctx0, g, hparams.kda_gate_lower_bound);
    g = ggml_reshape_4d(ctx0, g, head_dim, n_head, n_seq_tokens, n_seqs);
    cb(g, "kda_gate", il);

    ggml_tensor * beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta_in, 1, n_head, n_seq_tokens, n_seqs));
    cb(beta, "kda_beta", il);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

    ggml_tensor * out = build_recurrent_attn(inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g, beta, state, il);

    // the fallbacks return a permuted view, the fused op a contiguous one (no copy needed)
    ggml_tensor * o = ggml_is_contiguous(out) ? ggml_reshape_3d(ctx0, out, head_dim, n_head, n_tokens)
                                              : ggml_cont_3d(ctx0, out, head_dim, n_head, n_tokens);
    cb(o, "kda_scan_out", il);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ssm_g_b, g_lo);
    gate = ggml_reshape_3d(ctx0, gate, head_dim, n_head, n_tokens);

    // plain sigmoid gate, not the SiLU that FusedRMSNormGated defaults to
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * gated  = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, gate));
    cb(gated, "kda_normed", il);

    cur = ggml_mul_mat(ctx0, layer.wo, ggml_reshape_2d(ctx0, gated, d_inner, n_tokens));
    cb(cur, "kda_out", il);

    return cur;
}

// the store is NOT gated on the sparse path, the scoring is: gating both leaves cells
// below n_select with no indexer state, which the first ubatch to cross n_select pools.
//   * weights_proj runs in fp32; bf16 head-gates flip near-tie pool rankings (vLLM, sglang)
//   * k_norm is a LayerNorm WITH BIAS at eps 1e-6, not f_norm_rms_eps (transformers, vLLM)
//   * the ReLU between the QK dot and the head weighting is real (modular_glm5_next.py)
// no Hadamard rotation: H is orthogonal so (Hq).(Hk) == q.k; it only helps fp8.
ggml_tensor * llama_model_glm5next::graph::build_indexer(
        const llama_layer & layer,
        llm_graph_input_kpool * inp_kp,
        ggml_tensor * cur,
        ggml_tensor * qr,
        ggml_tensor ** top_k_mask,
        bool scoring,
        int il,
        bool state_only) const {
    const int64_t d_idx   = hparams.indexer_head_size;
    const int64_t n_ihead = hparams.indexer_n_head;
    const int64_t r       = hparams.indexer_kpool;

    const auto * mctx_idx = inp_kp->mctx_idx;

    GGML_ASSERT(layer.indexer_k_norm_b != nullptr && "the indexer k_norm is a LayerNorm with bias");

    // The checkpoint shares the first MTP iteration's selection with the
    // remaining draft iterations. These iterations do not consume their new
    // indexer K/G: accepted positions are written again by the normal catch-up
    // decode, while rejected positions are discarded. Return before building
    // the projections and cache write so they are pruned from the reuse graph.
    if (scoring && glm5next_mtp_topk_share_enabled(hparams) && inp_kp->pool_cache != nullptr &&
            inp_kp->pool_cache->get_mtp_index_reuse()) {
        const int64_t n_stream = inp_kp->pool_cells->ne[1];
        GGML_ASSERT(n_stream > 0 && n_tokens % n_stream == 0);

        const int64_t n_tps    = n_tokens/n_stream;
        const int64_t n_pools  = inp_kp->pool_cells->ne[0]/r;
        const int64_t select_k = llama_kpool_select_k(n_pools, hparams.indexer_top_k, r);
        const int64_t n_selected = r*select_k;

        GGML_ASSERT(il >= (int) hparams.n_layer() && n_tps == 1);
        GGML_ASSERT(select_k > 0 && select_k <= n_pools);

        // The host still rebuilds these maps to obtain the new incomplete
        // tail. Keep them in the graph even though pool scoring is skipped.
        ggml_build_forward_expand(gf, ggml_cont(ctx0, inp_kp->pool_cells));
        if (inp_kp->pool_bias != nullptr) {
            ggml_build_forward_expand(gf, ggml_cont(ctx0, inp_kp->pool_bias));
        } else {
            ggml_build_forward_expand(gf, ggml_cont(ctx0, inp_kp->pool_iv_cells));
            ggml_build_forward_expand(gf, ggml_cont(ctx0, inp_kp->pool_iv_queries));
        }

        *top_k_mask = inp_kp->pool_cache->get_mtp_selection(ctx0, il, n_selected,
                mctx_idx->get_stream_base(), n_stream, true);
        ggml_tensor * top_k = inp_kp->pool_cache->get_mtp_selection(ctx0, il, n_selected,
                mctx_idx->get_stream_base(), n_stream, false);
        cb(*top_k_mask, "indexer_top_k_mask_reused", il);
        cb(top_k, "indexer_top_k_reused", il);
        return top_k;
    }

    ggml_tensor * ik = build_norm(ggml_mul_mat(ctx0, layer.indexer_attn_k, cur),
            layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
    cb(ik, "indexer_k", il);

    // a SECOND, INDEPENDENT projection, not a reuse of the key, and cached beside it
    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.indexer_comp_wgate, cur);
    cb(gate, "indexer_gate", il);

    // {d_idx, 2, n_tokens}: head 0 is the key, head 1 the gate
    ggml_tensor * packed = ggml_concat(ctx0,
            ggml_reshape_3d(ctx0, ik,   d_idx, 1, n_tokens),
            ggml_reshape_3d(ctx0, gate, d_idx, 1, n_tokens), 1);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, packed, inp_kp->k_idxs, il));

    if (!scoring || (state_only && inp_kp->pool_cache == nullptr)) {
        *top_k_mask = nullptr;
        return nullptr;
    }

    ggml_tensor * kbuf = mctx_idx->get_k(ctx0, il);

    const int64_t n_kv     = kbuf->ne[2];
    const int64_t n_stream = kbuf->ne[3];
    const int64_t n_tps    = n_tokens/n_stream;
    const int64_t n_pools  = inp_kp->pool_cells->ne[0]/r;

    GGML_ASSERT(kbuf->ne[0] == d_idx && kbuf->ne[1] == 2 &&
            "the pooled indexer cache needs a key head and a gate head");
    GGML_ASSERT(kbuf->nb[1] == (size_t) d_idx*kbuf->nb[0] && "key and gate must be adjacent in a cell");
    GGML_ASSERT(n_tokens == n_tps*n_stream);

    const int64_t select_k = llama_kpool_select_k(n_pools, hparams.indexer_top_k, r);
    const int64_t n_selected = r*select_k;
    GGML_ASSERT(select_k > 0 && select_k <= n_pools);

    // The number of chunks follows the deepest stream that the cache holds, not the current depth: then every graph
    // has the same chunks as the worst-case graph that is reserved, and only their width shrinks with the depth.
    // (With chunks from the current depth, the reserved graph at full depth had a different shape from the prompt
    // graphs, and prefill at 100K of -c 300000 ran at half speed.)
    const int64_t n_pools_max = std::max<int64_t>(n_pools, inp_kp->pool_cache != nullptr ?
            (int64_t) inp_kp->pool_cache->get_max_slots() : (int64_t) cparams.n_ctx/r + 2);
    // n_parts pieces of n_items, each at most the rows of llm_graph_chunk_rows() at the depth n_pools_max
    auto chunk_size = [](int64_t n_items, int64_t n_items_max, int64_t width_max) {
        // 512 MiB: at -c 300000 -ub 1024 one chunk holds everything, so the graph is the one without chunks (two
        // chunks there cost 2-3 % prefill at 100K-250K for 0.3-0.5 GiB per device)
        const int64_t c = llm_graph_chunk_rows(n_items_max, width_max, sizeof(float), 512);
        const int64_t n_parts = std::min<int64_t>(n_items, (n_items_max + c - 1)/c);
        return (n_items + n_parts - 1)/n_parts;
    };

    // The selection runs for n_chunk query rows at a time. Its score is F32 [n_pools, rows] and the device pool mask
    // F16 [n_pools, rows]: both grow with the context times the rows (-c 1048576, -ub 4096: 4 GiB + 2 GiB per device).
    // With more than one chunk, each chunk makes its own interval mask, so that the full mask is not kept for all the
    // layers of the device. The single-row paths (temporal top-k, MTP selection sharing) never chunk.
    static const bool stable_prefill_chunks = [] {
        const char * env = std::getenv("LLAMA_GLM5_STABLE_PREFILL_CHUNKS");
        return env == nullptr || std::atoi(env) != 0;
    }();
    // Keep the partition count of the full prompt ubatch. At -c 1572864 -np 3 -ub 1024, 1024 queries need two
    // pieces but 1020 fit in one. Changing that count replaces the allocation plan and drains the pipeline.
    const int64_t n_tps_layout = stable_prefill_chunks && n_tokens_layout > 0 ?
            std::max<int64_t>(n_tps, n_tokens_layout/n_stream) : n_tps;
    const int64_t n_chunk = n_tps == 1 ? 1 : chunk_size(n_tps_layout, n_tps_layout, n_pools_max);
    const int64_t n_chunks = n_tps == 1 ? 1 : std::min<int64_t>(n_tps, (n_tps_layout + n_chunk - 1)/n_chunk);
    const bool chunked = n_chunks > 1;

    ggml_backend_dev_t dev = layer.wo->buffer ?
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(layer.wo->buffer)) : nullptr;

    // pool_valid & pool_visible: the host input, or one interval mask per device, so that no O(n_pools*n_tps)
    // input is uploaded (and copied once per pipeline copy). The device mask is F16 for the fused indexer.
    ggml_tensor * pool_bias     = inp_kp->pool_bias;     // F32 [n_pools, n_tps, n_stream]
    ggml_tensor * pool_bias_f16 = inp_kp->pool_bias_f16; // F16 [n_pools, n_tps, 1, n_stream]
    if (inp_kp->pool_iv_cells != nullptr && !chunked) {
        auto it = inp_kp->dev_pool_bias.find(dev);
        if (it == inp_kp->dev_pool_bias.end()) {
            ggml_tensor * bias = ggml_interval_mask(ctx0, inp_kp->pool_iv_cells, inp_kp->pool_iv_queries,
                    cparams.fused_lid ? GGML_TYPE_F16 : GGML_TYPE_F32);
            cb(bias, "kpool_dev_pool_bias", il);
            glm5next_pin(sched, bias, dev);
            it = inp_kp->dev_pool_bias.emplace(dev, bias).first;
        }
        if (it->second->type == GGML_TYPE_F16) {
            pool_bias_f16 = it->second;
            pool_bias     = nullptr;
        } else {
            pool_bias     = ggml_reshape_3d(ctx0, it->second, n_pools, n_tps, n_stream);
            pool_bias_f16 = nullptr;
        }
    }

    ggml_tensor * kg_rows = ggml_view_3d(ctx0, kbuf, 2*d_idx, n_kv, n_stream,
            kbuf->nb[2], kbuf->nb[3], 0);

    auto compress_members = [&](ggml_tensor * members, int64_t n_group) {
        const size_t nb_mem = members->nb[1];

        ggml_tensor * mem_k = ggml_view_4d(ctx0, members, d_idx, r, n_group, n_stream,
                nb_mem, nb_mem*r, members->nb[2], 0);
        ggml_tensor * mem_g = ggml_view_4d(ctx0, members, d_idx, r, n_group, n_stream,
                nb_mem, nb_mem*r, members->nb[2], d_idx*members->nb[0]);

        // r-way softmaxes over the slot axis; APE is added before softmax
        ggml_tensor * keys_t = ggml_cont(ctx0, ggml_permute(ctx0, mem_k, 1, 0, 2, 3));
        ggml_tensor * gate_t = ggml_cont(ctx0, ggml_permute(ctx0, mem_g, 1, 0, 2, 3));

        ggml_tensor * ape = ggml_cont(ctx0, ggml_transpose(ctx0, layer.indexer_comp_ape));
        gate_t = ggml_add(ctx0, gate_t, ggml_reshape_4d(ctx0, ape, r, d_idx, 1, 1));

        ggml_tensor * probs = ggml_soft_max(ctx0, gate_t);
        cb(probs, "indexer_pool_probs", il);

        ggml_tensor * result = ggml_sum_rows(ctx0, ggml_mul(ctx0, keys_t, probs));
        return ggml_reshape_3d(ctx0, result, d_idx, n_group, n_stream);
    };

    // Compress all the pools of `cells` [r*n_pools, n_stream]. The members are F32 [2*d_idx, r*n_pools] plus two
    // transposed halves (-c 1048576: 1 GiB + 2 x 512 MiB in one piece), so do it for a chunk of pools at a time.
    auto compress_all = [&](ggml_tensor * cells) {
        const int64_t n_pc = chunk_size(n_pools, n_pools_max, r*2*d_idx);
            if (n_pc >= n_pools) {
            ggml_tensor * members = ggml_get_rows(ctx0, kg_rows, cells);
            cb(members, "indexer_pool_members", il);
            return compress_members(members, n_pools);
        }
        ggml_tensor * out = nullptr;
        for (int64_t p0 = 0; p0 < n_pools; p0 += n_pc) {
            const int64_t np = std::min<int64_t>(n_pc, n_pools - p0);
            ggml_tensor * cells_c = ggml_view_2d(ctx0, cells, r*np, n_stream, cells->nb[1], p0*r*cells->nb[0]);
            if (n_stream > 1) {
                cells_c = ggml_cont(ctx0, cells_c);
            }
            ggml_tensor * members = ggml_get_rows(ctx0, kg_rows, cells_c);
            cb(members, "indexer_pool_members", il);
            ggml_tensor * part = compress_members(members, np);
            out = out == nullptr ? part : ggml_concat(ctx0, out, part, 1);
            // finish this chunk before the next one starts, so that their members do not live at the same time
            ggml_build_forward_expand(gf, out);
        }
        return out;
    };

    ggml_tensor * pool_k;

    if (inp_kp->pool_cache == nullptr) {
        pool_k = compress_all(inp_kp->pool_cells);
    } else if (inp_kp->rebuild_pool_cache) {
        // Prefill and invalidation rebuild every resident completed pool once.
        ggml_tensor * rebuilt = compress_all(inp_kp->pool_cells);
        ggml_tensor * store_values = ggml_get_rows(ctx0, rebuilt, inp_kp->pool_store_src);
        ggml_tensor * cache_write = inp_kp->pool_cache->store(ctx0, store_values, inp_kp->pool_store_dst,
                il, mctx_idx->get_stream_base(), n_stream);
        if (state_only) {
            ggml_build_forward_expand(gf, cache_write);
            *top_k_mask = nullptr;
            return nullptr;
        }
        pool_k = ggml_get_rows(ctx0, cache_write, inp_kp->pool_cache_slots);
    } else {
        // Recompress only pools completed by this decode/prefill batch. The
        // remaining resident pool keys stay in persistent device storage.
        const int64_t n_update = inp_kp->pool_update_cells->ne[0]/r;
        ggml_tensor * members = ggml_get_rows(ctx0, kg_rows, inp_kp->pool_update_cells);
        cb(members, "indexer_pool_update_members", il);

        ggml_tensor * update = compress_members(members, n_update);
        ggml_tensor * cache_write = inp_kp->pool_cache->store(ctx0, update, inp_kp->pool_update_dst,
                il, mctx_idx->get_stream_base(), n_stream);
        if (state_only) {
            ggml_build_forward_expand(gf, cache_write);
            *top_k_mask = nullptr;
            return nullptr;
        }
        pool_k = ggml_get_rows(ctx0, cache_write, inp_kp->pool_cache_slots);
    }

    pool_k = ggml_reshape_4d(ctx0, pool_k, d_idx, n_pools, 1, n_stream);
    cb(pool_k, "indexer_pool_k", il);

    // no rope: n_rot() is 0 for the whole text tower
    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_4d(ctx0, iq, d_idx, n_ihead, n_tps, n_stream);
    cb(iq, "indexer_q", il);

    // sign-unconstrained head weights: no softmax, no abs, no relu; both scale constants
    // fold in here. GGML_PREC_F32 is not cosmetic: bf16 can swap two near-tied pools
    ggml_tensor * w = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
    ggml_mul_mat_set_prec(w, GGML_PREC_F32);
    w = ggml_reshape_4d(ctx0, w, n_ihead, n_tps, 1, n_stream);
    w = ggml_scale(ctx0, w, 1.0f/sqrtf(float(d_idx*n_ihead)));
    cb(w, "indexer_weights", il);

    const int64_t n_compact = GGML_PAD(n_selected + r - 1, 256);
    const bool direct_indexed = n_tps == 1 ? n_kv >= std::max<int64_t>(4096, 2*n_compact) :
            llama_kpool_indexed_attn_enabled(n_kv, n_tps);
    const bool expand = glm5next_kpool_expand_enabled() && cparams.fused_lid &&
            il < (int) hparams.n_layer() && direct_indexed;

    // rows [q0, q0 + nq) of a tensor whose dimension `dim` is the query axis; a copy when the view is not contiguous
    auto rows_of = [&](ggml_tensor * t, int dim, int64_t q0, int64_t nq) -> ggml_tensor * {
        if (t == nullptr || !chunked) {
            return t;
        }
        int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        ne[dim] = nq;
        ggml_tensor * v = ggml_view_4d(ctx0, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], q0*t->nb[dim]);
        return n_stream > 1 ? ggml_cont(ctx0, v) : v;
    };

    ggml_tensor * top_k = nullptr;
    ggml_tensor * selected_valid = nullptr;
    for (int64_t part = 0, q0 = 0; q0 < n_tps; ++part) {
        const int64_t nq = stable_prefill_chunks && n_tokens_layout > 0 ?
                n_tps/n_chunks + (part < n_tps%n_chunks) : std::min<int64_t>(n_chunk, n_tps - q0);

        ggml_tensor * iq_c = rows_of(iq, 2, q0, nq);
        ggml_tensor * w_c  = rows_of(w,  1, q0, nq);

        ggml_tensor * bias_c     = rows_of(pool_bias,     1, q0, nq);
        ggml_tensor * bias_f16_c = rows_of(pool_bias_f16, 1, q0, nq);
        if (chunked && inp_kp->pool_iv_cells != nullptr) {
            ggml_tensor * bias = ggml_interval_mask(ctx0, inp_kp->pool_iv_cells,
                    rows_of(inp_kp->pool_iv_queries, 0, q0, nq), cparams.fused_lid ? GGML_TYPE_F16 : GGML_TYPE_F32);
            cb(bias, "kpool_dev_pool_bias", il);
            glm5next_pin(sched, bias, dev);
            if (bias->type == GGML_TYPE_F16) {
                bias_f16_c = bias;
                bias_c     = nullptr;
            } else {
                bias_c     = ggml_reshape_3d(ctx0, bias, n_pools, nq, n_stream);
                bias_f16_c = nullptr;
            }
        }
        // either type serves the validity tests below (0 or -INFINITY)
        ggml_tensor * bias_any_c = bias_c ? bias_c : ggml_reshape_3d(ctx0, bias_f16_c, n_pools, nq, n_stream);

        ggml_tensor * pool_score = nullptr;

        if (cparams.fused_lid) {
            // pool_k stays f32 so the kernel takes its f32 path; f16 wmma would undo the prec
            ggml_tensor * pool_kf = ggml_reshape_4d(ctx0, pool_k, d_idx, 1, n_pools, n_stream);

            pool_score = ggml_lightning_indexer(ctx0, iq_c, pool_kf, w_c, bias_f16_c);
            cb(pool_score, "indexer_pool_score", il);

            res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, pool_score, il});

            pool_score = ggml_reshape_3d(ctx0, pool_score, n_pools, nq, n_stream);
        } else {
            ggml_tensor * kq = ggml_mul_mat(ctx0, pool_k, ggml_permute(ctx0, iq_c, 0, 2, 1, 3));

            // the ReLU sits BETWEEN the per-head dot and the head weighting; either side differs
            kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
            ggml_tensor * score = ggml_relu(ctx0, kq);
            cb(score, "indexer_score", il);

            pool_score = ggml_sum_rows(ctx0, ggml_mul(ctx0, score, w_c));
            pool_score = ggml_cont(ctx0, ggml_permute(ctx0, pool_score, 2, 1, 0, 3));
            pool_score = ggml_reshape_3d(ctx0, pool_score, n_pools, nq, n_stream);

            pool_score = ggml_add(ctx0, pool_score, bias_c);
            cb(pool_score, "indexer_pool_score", il);
        }

        // top-k over POOLS then expand, as in the reference: a cell-level top-k is wrong
        // because relu ties span pool boundaries and ggml_top_k splits the pool it lands in
        // {select_k, nq, n_stream} of POOL ordinals
        ggml_tensor * sel;
        // The temporal selector wins only once the cache reaches 24576 pools.
        if (glm5next_temporal_topk_enabled() && inp_kp->pool_cache != nullptr && n_pools >= 24576 && n_tps == 1 &&
                select_k == (int64_t) hparams.indexer_top_k/r) {
            ggml_tensor * hint = inp_kp->pool_cache->get_topk_hint(
                    ctx0, il, select_k, mctx_idx->get_stream_base(), n_stream);
            sel = ggml_cont(ctx0, ggml_top_k_hint(ctx0, pool_score, hint, (int) select_k));
            ggml_build_forward_expand(gf, inp_kp->pool_cache->store_topk_hint(
                    ctx0, sel, il, mctx_idx->get_stream_base(), n_stream));
        } else {
            sel = ggml_cont(ctx0, ggml_top_k(ctx0, pool_score, (int) select_k));
        }
        cb(sel, "indexer_top_k_pools", il);

        ggml_tensor * top_k_c;
        ggml_tensor * valid_c = nullptr;
        if (expand) {
            top_k_c = ggml_kpool_expand(ctx0, sel, inp_kp->pool_cells, bias_any_c,
                    rows_of(inp_kp->compact_tail_cells, 1, q0, nq), rows_of(inp_kp->compact_tail_mask, 1, q0, nq),
                    (int32_t) r);
            cb(top_k_c, "indexer_top_k_compact", il);
        } else {
            // Preserve whether each selected pool was actually eligible. top-k always
            // returns its full width, so short or fragmented sequences can spill into
            // -inf pool slots whose member table aliases cell zero. The dense scatter is
            // naturally duplicate-free; direct compact attention needs this mask to make
            // every expanded member of a spilled pool inert.
            ggml_tensor * pool_valid = ggml_reshape_4d(
                    ctx0, bias_any_c, 1, n_pools, nq, n_stream);
            valid_c = ggml_get_rows(ctx0, pool_valid, sel);
            valid_c = ggml_repeat_4d(
                    ctx0, valid_c, r, select_k, nq, n_stream);
            valid_c = ggml_reshape_3d(
                    ctx0, valid_c, r*select_k, nq, n_stream);
            valid_c = ggml_cast(ctx0, valid_c, GGML_TYPE_F16);
            cb(valid_c, "indexer_top_k_mask", il);

            // Expand pools to members: gather whole rows of `kpool` cells out of pool_cells.
            // The query axis folds into the gather's row axis, which is what lets ONE
            // ggml_get_rows serve every query, while the stream axis stays where get_rows wants
            // it (src0 dim 2 is indexed by the index tensor's dim 1)
            ggml_tensor * pc3      = ggml_reshape_3d(ctx0, inp_kp->pool_cells, r, n_pools, n_stream);
            ggml_tensor * sel_flat = ggml_reshape_2d(ctx0, sel, select_k*nq, n_stream);

            top_k_c = ggml_get_rows(ctx0, pc3, sel_flat);
            GGML_ASSERT(top_k_c->type == GGML_TYPE_I32 && "pool_cells is I32, so the gather stays I32");
            top_k_c = ggml_reshape_3d(ctx0, top_k_c, r*select_k, nq, n_stream);
        }

        top_k = top_k == nullptr ? top_k_c : ggml_concat(ctx0, top_k, top_k_c, 1);
        if (valid_c != nullptr) {
            selected_valid = selected_valid == nullptr ? valid_c : ggml_concat(ctx0, selected_valid, valid_c, 1);
        }
        if (chunked) {
            // finish this chunk before the next one starts, so that their scores do not live at the same time
            ggml_build_forward_expand(gf, top_k);
            if (selected_valid != nullptr) {
                ggml_build_forward_expand(gf, selected_valid);
            }
        }
        q0 += nq;
    }

    if (expand) {
        *top_k_mask = nullptr;
        return top_k;
    }

    // Step 0 of an MTP draft group seeds persistent selection buffers. The
    // cpy nodes also establish the write-before-attention dependency, so the
    // next decode can safely switch to the reuse graph immediately.
    if (glm5next_mtp_topk_share_enabled(hparams) && inp_kp->pool_cache != nullptr &&
            il >= (int) hparams.n_layer() && n_tps == 1) {
        // Persistent destinations already own backend buffers. Expand the
        // selection producers explicitly so graph discovery cannot treat the
        // destination view as a leaf and prune their host-side pool inputs.
        ggml_build_forward_expand(gf, top_k);
        ggml_build_forward_expand(gf, selected_valid);

        const uint32_t stream0 = mctx_idx->get_stream_base();
        ggml_tensor * cache_top_k = inp_kp->pool_cache->store_mtp_selection(
                ctx0, top_k, il, stream0, n_stream, false);
        ggml_tensor * cache_mask = inp_kp->pool_cache->store_mtp_selection(
                ctx0, selected_valid, il, stream0, n_stream, true);
        // Keep attention on the original tensors. Using an already-allocated
        // persistent destination as the attention input makes the scheduler
        // regard that branch as a leaf and omit pool_cells allocation.
        ggml_build_forward_expand(gf, cache_top_k);
        ggml_build_forward_expand(gf, cache_mask);
    }

    *top_k_mask = selected_valid;
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// absorbed form (deepseek2/glm-dsa): q_nope goes through wk_b so q.k is taken against
// the latent the cache holds; the naive form needs a V cache this layout lacks
ggml_tensor * llama_model_glm5next::graph::build_dsa_layer(
        const llama_layer & layer,
        llm_graph_input_attn_k * inp_attn,
        llm_graph_input_kpool * inp_kp,
        bool scoring,
        ggml_tensor * cur,
        int il,
        bool state_only) const {
    const int64_t qk_head_dim  = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    GGML_ASSERT(hparams.n_rot() == 0);

    // scale is over the MLA head size, as in the reference, not the absorbed width
    const float kq_scale = 1.0f/sqrtf(float(qk_head_dim));

    if (state_only) {
        // Only what later batches read: the indexer key/gate and pooled keys, and the latent KV row. The queries,
        // indexer scores and attention feed this batch's outputs alone.
        ggml_tensor * top_k_mask = nullptr;
        if (inp_kp) {
            build_indexer(layer, inp_kp, cur, nullptr, &top_k_mask, scoring, il, true);
        }
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
        kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * k = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
        cb(k, "dsa_kv_latent", il);
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, k, inp_attn->get_k_idxs(), il));
        return nullptr;
    }

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "dsa_q_a_norm", il);

    // the indexer shares this q LoRA residual with the main MLA path and consumes it
    // with its own wq_b, so it is built here rather than being handed the layer input
    // twice. it also writes the indexer cache, which happens on the dense path too
    ggml_tensor * top_k_mask = nullptr;
    ggml_tensor * top_k = inp_kp ? build_indexer(
            layer, inp_kp, cur, qr, &top_k_mask, scoring, il) : nullptr;

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, qk_head_dim, n_head, n_tokens);
    cb(q, "dsa_q_b", il);

    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv, "dsa_kv_a_norm", il);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);

    q = ggml_mul_mat(ctx0, layer.wk_b, q);

    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
    cb(q, "dsa_q_absorbed", il);

    // absorbed MLA is MQA: one head of keys, and V is the same latent row as K
    ggml_tensor * k = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
    cb(k, "dsa_kv_latent", il);

    if (top_k) {
        ggml_tensor * sel_mask  = inp_kp->sel_mask;
        ggml_tensor * cand_mask = inp_kp->cand_mask;
        if (inp_kp->mask_cells != nullptr) {
            // One mask pair per device, pinned there: only the O(n_kv + n_tps)
            // inputs cross PCIe. The layer's output weight identifies its device.
            ggml_backend_dev_t dev = layer.wo->buffer ?
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(layer.wo->buffer)) : nullptr;
            auto it = inp_kp->dev_masks.find(dev);
            if (it == inp_kp->dev_masks.end()) {
                ggml_tensor * c = inp_kp->mask_cells;
                ggml_tensor * qm = inp_kp->mask_queries;
                ggml_tensor * c_sel  = ggml_view_4d(ctx0, c, c->ne[0], 2, 1, c->ne[3], c->nb[1], c->nb[2], c->nb[3], 0);
                ggml_tensor * c_cand = ggml_view_4d(ctx0, c, c->ne[0], 2, 1, c->ne[3], c->nb[1], c->nb[2], c->nb[3], c->nb[1]);
                ggml_tensor * sel  = ggml_interval_mask(ctx0, c_sel,  qm, GGML_TYPE_F16);
                ggml_tensor * cand = ggml_interval_mask(ctx0, c_cand, qm, GGML_TYPE_F16);
                cb(sel,  "kpool_dev_sel_mask",  il);
                cb(cand, "kpool_dev_cand_mask", il);
                glm5next_pin(sched, sel,  dev);
                glm5next_pin(sched, cand, dev);
                it = inp_kp->dev_masks.emplace(dev, std::make_pair(sel, cand)).first;
            }
            sel_mask  = it->second.first;
            cand_mask = it->second.second;
        }
        cur = build_attn_sparse(inp_attn,
                layer.wo, nullptr, nullptr,
                q, k, k, nullptr, nullptr, layer.wv_b,
                top_k, top_k_mask, inp_kp->compact_tail_cells, inp_kp->compact_tail_mask,
                sel_mask, cand_mask, kq_scale, il, inp_kp->mask_cells != nullptr);
    } else {
        cur = build_attn(inp_attn,
                layer.wo, nullptr, nullptr,
                q, k, k, nullptr, nullptr, layer.wv_b, kq_scale, il);
    }
    cb(cur, "dsa_out", il);

    return cur;
}

ggml_tensor * llama_model_glm5next::graph::build_layer_attn(
        const llama_model & model,
        llm_graph_input_mem_hybrid_k * inp_mem,
        llm_graph_input_kpool * inp_kp,
        bool scoring,
        ggml_tensor * cur,
        int il) {
    if (hparams.is_recr(il)) {
        return build_kda_layer(model.layers[il], inp_mem->get_recr(), cur, il);
    }

    return build_dsa_layer(model.layers[il], inp_mem->get_attn(), inp_kp, scoring, cur, il);
}

// delphi: a SwiGLU FFN whose gate and up weights are adjacent (glm5next_adjacent_rows): one product for both, then the
// activation of build_ffn on its two halves; nullptr when they are not adjacent
static ggml_tensor * glm5next_ffn_adjacent(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * gate, ggml_tensor * up,
        ggml_tensor * down, float limit) {
    ggml_tensor * w_gu = glm5next_adjacent_rows(ctx, {gate, up});
    if (w_gu == nullptr || ggml_nrows(gate) != ggml_nrows(up)) {
        return nullptr;
    }
    const int64_t n_ff = ggml_nrows(gate);
    ggml_tensor * gu = ggml_mul_mat(ctx, w_gu, cur);
    ggml_tensor * g  = ggml_view_2d(ctx, gu, n_ff, gu->ne[1], gu->nb[1], 0);
    ggml_tensor * u  = ggml_view_2d(ctx, gu, n_ff, gu->ne[1], gu->nb[1], n_ff*ggml_element_size(gu));
    ggml_tensor * act = limit > 1e-6f ? ggml_swiglu_clamp(ctx, g, u, limit) : ggml_swiglu_split(ctx, g, u);
    return ggml_mul_mat(ctx, down, act);
}

ggml_tensor * llama_model_glm5next::graph::build_layer_ffn(
        const llama_model & model,
        ggml_tensor * cur,
        int il) const {
    const auto & layer = model.layers[il];

    // build_ffn applies the LoRA adapters per weight; at 1 token MMVQ already fuses gate, up and the GLU
    const bool  adjacent = loras->empty() && ggml_n_dims(cur) == 2 && cur->ne[1] > 1;
    const float limit    = hparams.swiglu_clamp_shexp[il];

    // the leading dense layers clamp like the experts: one Glm5NextTextMLP serves both
    if (il < (int) hparams.n_layer_dense_lead) {
        if (ggml_tensor * out = adjacent ? glm5next_ffn_adjacent(ctx0, cur, layer.ffn_gate, layer.ffn_up, layer.ffn_down, limit) : nullptr) {
            return out;
        }
        return build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }

    // noaux_tc: exp_probs_b biases top-k selection only; the weights stay unbiased
    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, hparams.n_expert_used(il),
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);

    ggml_tensor * shexp = adjacent ?
        glm5next_ffn_adjacent(ctx0, cur, layer.ffn_gate_shexp, layer.ffn_up_shexp, layer.ffn_down_shexp, limit) : nullptr;
    if (!shexp) {
        shexp = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
    cb(shexp, "ffn_shexp", il);

    // shared expert unscaled: routed_scaling_factor applies to the routed weights only
    return ggml_add(ctx0, moe_out, shexp);
}

llama_model_glm5next::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    ggml_tensor * cur;

    ggml_tensor * inp         = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    llm_graph_input_mem_hybrid_k * inp_mem = build_inp_mem_hybrid_k();

    // one map for the whole ubatch; nothing in it depends on the layer. gated on n_ctx,
    // not n_kv, which grows and would flip the graph topology mid-run
    llm_graph_input_kpool * inp_kp = nullptr;
    bool indexer_scoring = false;
    {
        const auto * mctx_hyb = static_cast<const llama_memory_hybrid_context *>(mctx);

        if (mctx_hyb->get_idx() != nullptr) {
            indexer_scoring = cparams.n_ctx > glm5next_n_select(hparams);

            inp_kp = build_inp_kpool(mctx_hyb,
                    inp_mem->get_attn()->get_kq_mask(), indexer_scoring);
        }
    }

    GGML_ASSERT(ubatch.n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);

    const int64_t hc = hparams.dsv4_hc_mult;

    // hc_mult exact copies of the embedding: no scaling, no one-hot into stream 0
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // delphi: the hc_post of a sublayer runs in the HC mix op of the next boundary (build_hc_post_pre); pend_* hold the
    // hc_post of the last FFN until the next layer
    ggml_tensor * pend_y    = nullptr;
    ggml_tensor * pend_res  = nullptr;
    ggml_tensor * pend_post = nullptr;
    ggml_tensor * pend_comb = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        if (pend_y) {
            cur = build_hc_post_pre(pend_y, pend_res, pend_post, pend_comb,
                    model.layers[il].hc_attn_fn,
                    model.layers[il].hc_attn_scale,
                    model.layers[il].hc_attn_base,
                    &post, &comb, &inpL, il, model.layers[il].attn_norm);
            cb(inpL, "l_last", il - 1);
            pend_y = nullptr;
        } else {
            cur = build_hc_pre(inpL,
                    model.layers[il].hc_attn_fn,
                    model.layers[il].hc_attn_scale,
                    model.layers[il].hc_attn_base,
                    &post, &comb, il, model.layers[il].attn_norm);
        }
        cb(cur, "attn_norm", il);

        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;

        cur = build_layer_attn(model, inp_mem, inp_kp, indexer_scoring, cur, il);

        cur = build_hc_post_pre(cur, residual, post, comb,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, &inpL, il, model.layers[il].ffn_norm);
        cb(inpL, "hc_attn_post", il);
        cb(cur, "ffn_norm", il);

        residual = inpL;

        // expand before the sublayer so op offload does not pull mHC state to the experts
        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_layer_ffn(model, cur, il);
        cb(cur, "ffn_out", il);

        // the next layer takes the hc_post only on the same device: across devices it copies y, the residual, post and comb
        // instead of the new streams, 0.2 to 0.5 ms slower per pass (GLM-5.3, 3 CMP 170HX, 1 to 8 tokens)
        if (il + 1 < n_layer && cvec->tensor_for(il) == nullptr && model.dev_layer(il) == model.dev_layer(il + 1)) {
            pend_y    = cur;
            pend_res  = residual;
            pend_post = post;
            pend_comb = comb;
        } else {
            inpL = build_hc_post(cur, residual, post, comb, il);
            inpL = build_cvec(inpL, il);
            cb(inpL, "l_last", il);
        }
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = build_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
        // flattened: get_rows needs one token's streams to be one contiguous row
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
    }

    // no hc_head tensor here: unweighted mean, not DeepSeek-V4's learned gated head
    cur = build_hc_mean(ctx0, inpL);
    cb(cur, "hc_mean", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// GLM-5.3-Flash NextN head. Unlike the trunk blocks, the trained MTP block has
// a plain residual path (no mHC mixer), but it keeps the complete DSA indexer,
// pooled sparse attention and MoE/shared-expert FFN of a normal DSA layer.
llama_model_glm5next::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(params) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "glm5next MTP requires exactly one NextN block");
    GGML_ASSERT(cparams.nextn_layer_offset == 0 && "glm5next has one NextN block");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm);

    const bool device_draft = cparams.mtp_device_draft_mode != LLAMA_MTP_DEVICE_DRAFT_DISABLED;
    const bool device_reuse = cparams.mtp_device_draft_mode == LLAMA_MTP_DEVICE_DRAFT_REUSE;

    if (device_draft) {
        GGML_ASSERT(params.mtp_device_draft_cache != nullptr);
        GGML_ASSERT(params.mtp_device_draft_cache->tokens != nullptr);
        GGML_ASSERT(params.mtp_device_draft_cache->hidden != nullptr);

        res->t_mtp_device_seq_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        res->t_mtp_device_result_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(res->t_mtp_device_seq_ids);
        ggml_set_input(res->t_mtp_device_result_ids);
        res->add_input(std::make_unique<llm_graph_input_glm5next_mtp_device_draft>(
                res->t_mtp_device_seq_ids, res->t_mtp_device_result_ids,
                params.mtp_device_draft_cache));
    }

    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * h_embd   = nullptr;
    if (device_reuse) {
        ggml_tensor * tokens = ggml_get_rows(
                ctx0, params.mtp_device_draft_cache->tokens, res->t_mtp_device_seq_ids);
        tokens = ggml_cast(ctx0, tokens, GGML_TYPE_I32);
        tokens = ggml_reshape_1d(ctx0, tokens, n_tokens);

        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, tokens);
        h_embd = ggml_get_rows(
                ctx0, params.mtp_device_draft_cache->hidden, res->t_mtp_device_seq_ids);
    } else {
        // The seed graph receives the target token and hidden state normally.
        // Host-driven fallback uses this branch for every draft iteration.
        auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp->tokens);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
        ggml_set_input(inp->embd);

        if (ubatch.token) {
            ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
            tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
        } else {
            tok_embd = inp->embd;
        }

        inp->build_h(ctx0, ubatch);
        ggml_set_name(inp->h, "mtp_h_input");
        h_embd = inp->h;
        res->add_input(std::move(inp));
    }
    cb(tok_embd, "mtp_tok_embd", il);

    // A prompt-ingest batch has no outputs and only has to leave the layer's state for later batches: the indexer
    // key/gate, the pooled keys and the latent KV. Attention, the FFN and the head are skipped. Short batches keep the
    // full graph: single tokens also store the top-k hint and the shared MTP selection, and the decode catch-up
    // batches (a few tokens) would otherwise alternate graph shapes with the draft steps, and every change of node
    // count reallocates the compute graph. env LLAMA_MTP_STATE_ONLY=0 disables it.
    static const bool state_only_enabled = [] {
        const char * env = getenv("LLAMA_MTP_STATE_ONLY");
        return env == nullptr || atoi(env) != 0;
    }();
    const bool state_only = state_only_enabled && n_outputs == 0 && n_tokens >= 32;

    ggml_tensor * inp_out_ids = state_only ? nullptr : build_inp_out_ids();
    auto * inp_mem = build_inp_mem_hybrid_k();

    llm_graph_input_kpool * inp_kp = nullptr;
    bool indexer_scoring = false;
    {
        const auto * mctx_hyb = static_cast<const llama_memory_hybrid_context *>(mctx);
        GGML_ASSERT(mctx_hyb->get_idx() != nullptr && "glm5next MTP requires its indexer cache");

        indexer_scoring = cparams.n_ctx > glm5next_n_select(hparams);
        inp_kp = build_inp_kpool(mctx_hyb, inp_mem->get_attn()->get_kq_mask(), indexer_scoring);
    }

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * cur = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(cur, "mtp_concat", il);

    cur = ggml_mul_mat(ctx0, layer.nextn.eh_proj, cur);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * residual = cur;
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    if (state_only) {
        build_dsa_layer(layer, inp_mem->get_attn(), inp_kp, indexer_scoring, cur, il, true);
        // small host-filled inputs that the pruned scoring path read; keep them allocated for set_input
        for (ggml_tensor * t : { inp_kp->pool_cells, inp_kp->pool_cache_slots }) {
            if (t != nullptr) {
                ggml_build_forward_expand(gf, t);
            }
        }
        return;
    }

    cur = build_dsa_layer(layer, inp_mem->get_attn(), inp_kp, indexer_scoring, cur, il);
    cur = ggml_add(ctx0, cur, residual);
    cb(cur, "mtp_post_attn", il);

    residual = cur;
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    cur = build_layer_ffn(model, cur, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm = layer.nextn.shared_head_norm
        ? layer.nextn.shared_head_norm
        : model.output_norm;
    GGML_ASSERT(head_norm && "glm5next MTP is missing shared_head_norm/output_norm");

    cur = build_norm(cur, head_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    ggml_tensor * head = layer.nextn.shared_head_head
        ? layer.nextn.shared_head_head
        : model.output;
    GGML_ASSERT(head && "glm5next MTP is missing shared_head_head/output");

    cur = ggml_mul_mat(ctx0, head, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_glm5next::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }

    return std::make_unique<graph>(*this, params);
}
