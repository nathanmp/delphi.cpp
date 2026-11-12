#include "llama-context.h"
#include "llama-dsv41-prefill.h"
#include "llama-engram.h"

#include "ggml-cpuset.h"
#include "ggml-exec-trace.h"
#include "ggml-sched-input.h"
#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-kpool.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

//
// llama_context
//

// true if the CPU backend runs --numa tensors (as in llama_model_loader)
static bool cpu_numa_tensor_mode() {
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        return false;
    }
    auto * is_tensor_mode = (bool (*)(void))
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_dev), "ggml_backend_cpu_is_numa_tensor_mode");
    return is_tensor_mode && is_tensor_mode();
}

// DeepSeek V4.1 with parallel sequences in separate KV streams: the KV cache gives a long prompt its own ubatches (one
// sequence, compacted decoder rows), and a ubatch of several sequences holds at most n_keep_dec tokens of each (see
// llama_kv_cache_dsv4::init_batch). The largest multi-sequence ubatch, or 0 when that split does not apply.
static uint32_t dsv4_multi_seq_tokens(const llama_cparams & cparams, const llama_memory_i * memory) {
    if (cparams.kv_unified || cparams.n_seq_max <= 1) {
        return 0;
    }
    const auto * kv = dynamic_cast<const llama_kv_cache_dsv4 *>(memory);
    return kv ? kv->get_n_keep_dec() * cparams.n_seq_max : 0;
}

// The largest multi-sequence ubatch when the memory gives a long prompt its own ubatches (DeepSeek V4.1, GLM with
// separate KV streams), or 0 when that split does not apply.
static uint32_t multi_seq_split_tokens(const llama_cparams & cparams, const llama_memory_i * memory) {
    if (const uint32_t n = dsv4_multi_seq_tokens(cparams, memory); n > 0) {
        return n;
    }
    if (cparams.kv_unified || cparams.n_seq_max <= 1) {
        return 0;
    }
    const auto * hybrid = dynamic_cast<const llama_memory_hybrid *>(memory);
    return hybrid ? hybrid->get_n_seq_split_tokens() * cparams.n_seq_max : 0;
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_mix_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_MIX,
    /*.name             =*/ "fused DeepSeek V4 HC mix",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    ggml_exec_trace_init();

    expert_device_policy = llama_expert_device_policy::from_env();
    std::vector<std::string> gpu_names;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { gpu_names.emplace_back(ggml_backend_dev_name(dev)); }
    }
    expert_device_policy.validate(gpu_names);

    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    // LLAMA_MOE_ROUTING_LOG=<file>: append the router's top-k expert ids of every MoE layer to a binary file, for offline
    // expert-placement profiling. Draft contexts (MTP, or with another context) are skipped, and so are ubatches of more
    // than LLAMA_MOE_ROUTING_LOG_MAX_TOKENS tokens (default 16: decode and verify only). The eval callback
    // copies one small tensor per layer to the host, which slows decode, but leaves the computation unchanged.
    if (const char * path = getenv("LLAMA_MOE_ROUTING_LOG");
            path != nullptr && *path != '\0' && params.cb_eval == nullptr && params.ctx_other == nullptr &&
            params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
        moe_routing_log = fopen(path, "ab");
        if (moe_routing_log == nullptr) {
            throw std::runtime_error(format("LLAMA_MOE_ROUTING_LOG: cannot open %s", path));
        }
        if (const char * env = getenv("LLAMA_MOE_ROUTING_LOG_MAX_TOKENS")) {
            moe_routing_log_max_tokens = atoll(env);
        }
        cparams.cb_eval           = moe_routing_log_cb;
        cparams.cb_eval_user_data = this;
        LLAMA_LOG_WARN("%s: logging MoE routing to %s\n", __func__, path);
    }

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH || model.arch == LLM_ARCH_QWEN4EXP) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    {
        // the fused kernel sums heads in a different order, so a near-tied top-k can differ
        const char * LLAMA_FUSED_LID_DISABLE = getenv("LLAMA_FUSED_LID_DISABLE");
        if (LLAMA_FUSED_LID_DISABLE && atoi(LLAMA_FUSED_LID_DISABLE) != 0) {
            cparams.fused_lid = false;
            cparams.auto_flid = false;
        }
    }

    cparams.n_layer_prefill = hparams.n_layer();
    cparams.ced_compact = false;
    if (model.arch == LLM_ARCH_DEEPSEEK41) {
        const char * LLAMA_DSV41_TRIM = getenv("LLAMA_DSV41_TRIM");
        const int mode = LLAMA_DSV41_TRIM ? atoi(LLAMA_DSV41_TRIM) : 1;
        if (mode == 1 || mode == 2) {
            const uint32_t split = hparams.n_layer()/2;
            bool supported = split > 0 && hparams.dsv41_candidate_source == (int32_t) split &&
                    hparams.dsv41_is_kv_source(split) && hparams.dsv41_owns_index_k(split);
            for (uint32_t il = split + 1; il < hparams.n_layer(); ++il) {
                supported &= !hparams.dsv41_is_kv_source(il) && !hparams.dsv41_owns_index_k(il);
            }
            for (uint32_t il = split; il < hparams.n_layer(); ++il) {
                supported &= model.layers[il].engram_embd == nullptr;
            }
            if (supported) {
                cparams.n_layer_prefill = split;
                cparams.ced_compact = mode == 1;
                LLAMA_LOG_INFO("%s: CED prefill: %s, encoder layers = %u\n", __func__, mode == 1 ? "compact" : "bounded mask", split);
            } else {
                LLAMA_LOG_WARN("%s: CED prefill disabled: unsupported decoder source layout\n", __func__);
            }
        }
    }

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;
    // delphi: the hyper-connection mixes in F32 and the norm of the layer input in one op (ggml_dsv4_hc_mix);
    // LLAMA_FUSED_HC_MIX=0 keeps the separate ops and the q8_1 activations of a quantized hc_fn
    {
        const char * LLAMA_FUSED_HC_MIX = getenv("LLAMA_FUSED_HC_MIX");
        cparams.fused_dsv4_hc_mix = LLAMA_FUSED_HC_MIX == nullptr || atoi(LLAMA_FUSED_HC_MIX) != 0;
    }

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.moe_split_n = 0;
    cparams.moe_split_min_tokens = 1024;
    cparams.attn_split_groups = 0;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }

        const char * LLAMA_GRAPH_ALT_MAX_TOKENS = getenv("LLAMA_GRAPH_ALT_MAX_TOKENS");
        if (LLAMA_GRAPH_ALT_MAX_TOKENS) {
            graph_alt_max_tokens = (uint32_t) std::max(0, atoi(LLAMA_GRAPH_ALT_MAX_TOKENS));
        }
        const char * LLAMA_GRAPH_ALT_SLOTS = getenv("LLAMA_GRAPH_ALT_SLOTS");
        if (LLAMA_GRAPH_ALT_SLOTS) {
            graph_alt_slots = (uint32_t) std::max(0, atoi(LLAMA_GRAPH_ALT_SLOTS));
        }
        if (graph_reuse_disable || graph_alt_slots == 0) {
            graph_alt_max_tokens = 0;
        }
        graph_slots.resize(1 + (graph_alt_max_tokens > 0 ? graph_alt_slots : 0));
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    if (cparams.kv_unified && cparams.n_seq_max > 1 &&
            (model.arch == LLM_ARCH_DEEPSEEK4 || model.arch == LLM_ARCH_DEEPSEEK41)) {
        LLAMA_LOG_WARN("%s: %s with a unified KV cache reserves its compute buffers for all %u sequences at the full "
                "context depth; without --kv-unified each sequence has its own KV stream and much smaller buffers\n",
                __func__, llm_arch_name(model.arch), cparams.n_seq_max);
    }
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // LLAMA_MOE_SPLIT=<device>[:<n>]: compute n routed experts (default half) of each host-resident MoE layer on another
        // device, while the rest run where the scheduler puts them (see llm_graph_context::build_moe_ffn_split)
        if (const char * env = getenv("LLAMA_MOE_SPLIT"); env != nullptr && *env != '\0' && hparams.n_expert > 1) {
            std::string spec = env;
            int32_t n = (int32_t) hparams.n_expert / 2;
            if (const size_t colon = spec.find(':'); colon != std::string::npos) {
                n = atoi(spec.c_str() + colon + 1);
                spec.resize(colon);
            }
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(spec.c_str());
            if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU || n <= 0 || n >= (int32_t) hparams.n_expert) {
                throw std::runtime_error(format("LLAMA_MOE_SPLIT=%s: expected <GPU device>[:<experts, 1..%u>]", env, hparams.n_expert - 1));
            }
            for (auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == dev) {
                    backend_moe_split = backend.get();
                }
            }
            if (backend_moe_split == nullptr) {
                backend_moe_split = ggml_backend_dev_init(dev, nullptr);
                if (backend_moe_split == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", spec.c_str()));
                }
                backends.emplace_back(backend_moe_split);
            }
            cparams.moe_split_n = n;
            // the two devices hand over twice per layer: order those copies on the devices, not with host waits
            // (see GGML_SCHED_XDEV_EVENTS in ggml_backend_sched_new)
            if (getenv("GGML_SCHED_XDEV_EVENTS") == nullptr) {
                setenv("GGML_SCHED_XDEV_EVENTS", "1", 0);
            }
            if (const char * e = getenv("LLAMA_MOE_SPLIT_MIN_TOKENS")) {
                cparams.moe_split_min_tokens = (uint32_t) std::max(1, atoi(e));
            }
            LLAMA_LOG_INFO("%s: MoE split: %d of %u routed experts on %s for ubatches of %u+ tokens\n", __func__,
                    n, hparams.n_expert, spec.c_str(), cparams.moe_split_min_tokens);
            // LLAMA_ATTN_SPLIT=<groups>: the same device also computes the attention heads of that many output groups
            // (DeepSeek V4.1, see llama_model_deepseek41::graph::build_attention_split)
            if (const char * e = getenv("LLAMA_ATTN_SPLIT"); e != nullptr && atoi(e) > 0) {
                cparams.attn_split_groups = atoi(e);
                LLAMA_LOG_INFO("%s: attention split: %d output groups on %s\n", __func__, cparams.attn_split_groups, spec.c_str());
                const char * ec = getenv("LLAMA_ATTN_SPLIT_COPY"); // 0: read slices of the layer device's weights
                if (model.arch == LLM_ARCH_DEEPSEEK41 && cparams.attn_split_groups < (int32_t) hparams.dsv4_o_group_count &&
                        (ec == nullptr || atoi(ec) != 0)) {
                    init_attn_split_weights();
                }
            }
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        mem_params = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(mem_params, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                // Under --numa tensors the CPU threads of all nodes read and write this buffer. Pinned host memory is
                // on the node of the GPU, and the memory channels of that node then carry all of that traffic. Thus
                // spread its pages over the nodes, if the device can (LLAMA_NUMA_HOST_INTERLEAVE=0 disables this).
                const char * interleave = getenv("LLAMA_NUMA_HOST_INTERLEAVE");
                if (host_buft && (!interleave || atoi(interleave) != 0) && cpu_numa_tensor_mode()) {
                    auto * numa_buft_fn = (ggml_backend_buffer_type_t (*)(void))
                        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev.dev), "ggml_backend_host_numa_buffer_type");
                    if (numa_buft_fn) {
                        host_buft = numa_buft_fn();
                    }
                }
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        // tensor buffer overrides used to disable it unconditionally (upstream #11397), but
        // overrides that only relocate embedding lookup tables do not change layer compute
        // placement. LLAMA_PP_ALLOW_EMBEDDING_OVERRIDES=0 restores the upstream gate for A/B.
        bool pp_blocked_by_overrides = model.has_tensor_overrides();
        {
            const char * LLAMA_PP_ALLOW_EMBEDDING_OVERRIDES = getenv("LLAMA_PP_ALLOW_EMBEDDING_OVERRIDES");
            const bool allow_embedding_overrides =
                LLAMA_PP_ALLOW_EMBEDDING_OVERRIDES ? atoi(LLAMA_PP_ALLOW_EMBEDDING_OVERRIDES) != 0 : true;
            if (allow_embedding_overrides && !model.has_non_embedding_tensor_overrides()) {
                pp_blocked_by_overrides = false;
            }
            // LLAMA_PP_ALLOW_OVERRIDES=1: keep pipeline parallelism with any override, e.g. some layers' experts on the CPU
            // and the rest of a layer split over GPUs, where each ubatch otherwise runs the GPUs in series
            const char * LLAMA_PP_ALLOW_OVERRIDES = getenv("LLAMA_PP_ALLOW_OVERRIDES");
            if (LLAMA_PP_ALLOW_OVERRIDES && atoi(LLAMA_PP_ALLOW_OVERRIDES) != 0) {
                pp_blocked_by_overrides = false;
            }
        }

        // tensor split runs every layer on all GPUs at once, so there is no pipeline across devices, but
        // the extra input copies still let the next ubatch's CPU splits and input copies proceed while
        // the Meta device is busy, instead of draining every GPU at each CPU -> GPU copy
        const bool split_mode_pp =
            (model.split_mode() == LLAMA_SPLIT_MODE_LAYER  && model.n_devices() > 1) ||
            (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR && model.get_split_state_ud.n_devices > 1);

        bool pipeline_parallel =
            split_mode_pp &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            cparams.offload_kqv &&
            !pp_blocked_by_overrides;

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        const bool engram_model = model.arch == LLM_ARCH_DEEPSEEK41 ||
            (model.arch == LLM_ARCH_QWEN4EXP && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT);
        if (engram_model && cparams.cb_eval == nullptr) {
            const char * env = getenv("LLAMA_ENGRAM_PREFETCH");
            const int n_readers = env ? std::min(32, std::max(0, atoi(env))) : 8;
            const auto host_engram = [](const llama_layer & layer) {
                return llama_engram_reader::supports(layer.engram_embd);
            };
            const bool host_table = model.arch == LLM_ARCH_QWEN4EXP ?
                llama_engram_reader::supports(model.per_layer_tok_embd) :
                std::any_of(model.layers.begin(), model.layers.end(), host_engram);
            if (n_readers > 0 && !model.hparams.no_alloc && host_table) {
                engram_reader = std::make_unique<llama_engram_reader>(n_readers);
                LLAMA_LOG_INFO("%s: Engram row prefetch with %d readers (up to 16 tokens)\n", __func__, n_readers);
            }
        }

        if (params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && params.ctx_other == nullptr) {
            try {
                // Eval callbacks execute subgraphs. Preserve their original CPU readiness boundary.
                expert_cache = llama_expert_cache::create(model, backends, cparams.cb_eval == nullptr, expert_device_policy);
            } catch (const std::exception & e) {
                if (!llama_expert_cache::auto_requested()) {
                    throw;
                }
                LLAMA_LOG_INFO("%s: expert cache off: %s\n", __func__, e.what());
            }
        }
        staggered_verify_init();
        verify_streams_init();
        sched_reserve();

        // LLAMA_MOE_CACHE_TIERS=<dev>:<MiB>[,...]: run-time tiers added at once, for tests without a VRAM sharing daemon
        if (const char * env = getenv("LLAMA_MOE_CACHE_TIERS"); env && expert_cache) {
            std::string list = env;
            for (size_t pos = 0; pos < list.size();) {
                const size_t end = std::min(list.find(',', pos), list.size());
                const std::string item = list.substr(pos, end - pos);
                const size_t colon = item.find(':');
                if (colon != std::string::npos) {
                    llama_share_add_tier(this, item.substr(0, colon).c_str(), size_t(atoll(item.c_str() + colon + 1)) << 20);
                }
                pos = end + 1;
            }
        }

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

// One record per (MoE layer, ubatch): int32 layer, int32 n_tokens, int32 top_k, then n_tokens*top_k int32 expert ids,
// token-major. The top-k ids are usually a view of a sorted index tensor, so rows are read with their stride.
bool llama_context::moe_routing_log_cb(ggml_tensor * t, bool ask, void * user_data) {
    static const char prefix[] = "ffn_moe_topk-";
    const bool match = strncmp(t->name, prefix, sizeof(prefix) - 1) == 0;
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }

    auto * ctx = (llama_context *) user_data;
    const int64_t k        = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    if (t->type != GGML_TYPE_I32 || t->nb[0] != sizeof(int32_t) || t->ne[2] != 1 || t->ne[3] != 1 ||
            (ctx->moe_routing_log_max_tokens > 0 && n_tokens > ctx->moe_routing_log_max_tokens)) {
        return true;
    }

    const size_t span = (size_t) (n_tokens - 1)*t->nb[1] + (size_t) k*sizeof(int32_t);
    ctx->moe_routing_buf.resize(span);
    ggml_backend_tensor_get(t, ctx->moe_routing_buf.data(), 0, span);

    const int32_t hdr[3] = { (int32_t) atoi(t->name + sizeof(prefix) - 1), (int32_t) n_tokens, (int32_t) k };
    fwrite(hdr, sizeof(hdr), 1, ctx->moe_routing_log);
    for (int64_t i = 0; i < n_tokens; ++i) {
        fwrite(ctx->moe_routing_buf.data() + i*t->nb[1], sizeof(int32_t), k, ctx->moe_routing_log);
    }
    // a client attributes records to its requests by the file size
    fflush(ctx->moe_routing_log);
    return true;
}

llama_context::~llama_context() {
    if (moe_routing_log) { fclose(moe_routing_log); }
    // asynchronous state copies whose destination may already be gone
    state_async_pending.clear();

    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();
    verify_streams_free();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            // the expectation is for the main scheduler, the alternate one only holds small graphs
            const size_t size_act = ggml_backend_sched_get_buffer_size(graph_sched_main(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            // The scheduler can place a fused node that has no weights of its own on the device of its inputs, which
            // at a layer split is the previous layer's device. That is not missing support: only disable the op when
            // the layer's device cannot run it.
            if (device_fused != device_layer && device_layer && ggml_backend_dev_supports_op(device_layer, node.tensor)) {
                LLAMA_LOG_DEBUG("%s: layer %d: %s is on device %s, layer on %s, which supports it\n", func, node.il,
                        probe.name, device_fused ? ggml_backend_dev_name(device_fused) : "none",
                        ggml_backend_dev_name(device_layer));
                continue;
            }

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        resolve(llm_fused_op_dsv4_hc_mix_probe,  cparams.fused_dsv4_hc_mix);
        cparams.auto_fhc = false;
    }
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve && sampler_sig() == sampler_sig_reserved) {
        return;
    }

    sched_need_reserve = false;
    sampler_sig_reserved = sampler_sig();

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    // Backend placement can depend on CPU throughput. Configure the batch thread
    // count before the scheduler assigns the reserve graphs, not only before it
    // executes them. Otherwise the first graph uses the backend default and its
    // placement can be cached for the whole context.
    set_backend_n_threads(cparams.n_threads_batch);

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    // the alternate schedulers hold graphs of the old reservation
    graph_alt_reset();
    for (auto & slot : graph_slots) {
        slot.sched.reset();
        for (auto & res : slot.gf_res_prev) {
            res.reset();
        }
    }
    max_prefill_sched = nullptr;
    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    sched_set_stage_lender();

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        llama_memory_context_ptr mctx_pp;
        uint32_t n_tokens_pp = n_tokens;
        uint32_t n_seqs_pp = n_seqs;
        uint32_t n_outputs_pp_first = n_outputs_pp;

        if (dsv4_multi_seq_tokens(cparams, memory.get()) > 0) {
            // a prompt ubatch has one sequence: reserve that shape over one stream
            n_seqs_pp = 1;
            mctx_pp = memory->init_full_n_seq(1);
        } else if (model.arch == LLM_ARCH_GLM5NEXT && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
            // The largest startup CUDA workspace occurs on an intermediate
            // prompt chunk: one sequence, a nearly-full ubatch, a two-ubatch
            // dense KV span, and no output rows. A full ubatch or even a
            // one-output graph has a different allocation layout and
            // underestimates this case. Growing the buffer during inference would
            // require the old and new CUDA allocations to coexist transiently.
            // A partial ubatch can have a different allocation layout from an
            // exactly full one. Use the largest partial shape as the bound.
            n_tokens_pp = n_tokens > 1 ? n_tokens - 1 : n_tokens;
            n_seqs_pp = 1;
            n_outputs_pp_first = 0;
            mctx_pp = memory->init_full_n_seq(
                    n_seqs_pp, std::min<uint32_t>(cparams.n_ctx_seq, 2*n_tokens), false);
            if (!mctx_pp) {
                throw std::runtime_error("failed to initialize dense-prefill memory module");
            }
        }

        const llama_memory_context_i * mctx_pp_ptr = mctx_pp ? mctx_pp.get() : mctx.get();
        auto * gf = graph_reserve(n_tokens_pp, n_seqs_pp, n_outputs_pp_first, mctx_pp_ptr,
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                staggered_verify_init();
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                sched_set_stage_lender();
                gf = graph_reserve(n_tokens_pp, n_seqs_pp, n_outputs_pp_first, mctx_pp_ptr);
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // In no-alloc mode every synthetic graph must contribute to the projected
    // buffer sizes used by auto-fit. The regular allocator retains the maximum
    // across reserve calls, while reserve_size reports only the current graph.
    auto graph_reserve_extra = [&](uint32_t n_tokens_cur, uint32_t n_seqs_cur,
            uint32_t n_outputs_cur, const llama_memory_context_i * mctx_cur) {
        if (!model.hparams.no_alloc) {
            return graph_reserve(n_tokens_cur, n_seqs_cur, n_outputs_cur, mctx_cur);
        }

        std::vector<size_t> sizes(backend_buf_exp_size.size());
        auto * gf = graph_reserve(
                n_tokens_cur, n_seqs_cur, n_outputs_cur, mctx_cur, true, sizes.data());
        if (gf) {
            for (size_t i = 0; i < sizes.size(); ++i) {
                backend_buf_exp_size[i] = std::max(backend_buf_exp_size[i], sizes[i]);
            }
        }
        return gf;
    };

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve_extra(n_seqs, n_seqs, n_seqs, mctx.get());
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // delphi: with a KV window, the widest ubatch that gathers its selected cells (a speculative verify) holds the
    // largest gather buffers; reserve it here, so that a context that does not fit fails at load, not at the first
    // verify past the indexer threshold
    {
        const char * env = getenv("LLAMA_KV_RESIDENT_CELLS");
        const uint32_t n_gather = std::min(cparams.n_ubatch, llama_kv_window_stage_min_tokens() - 1);
        if (env != nullptr && *env != '\0' && model.arch == LLM_ARCH_GLM5NEXT && n_gather > n_seqs) {
            if (!graph_reserve_extra(n_gather, n_seqs, n_gather, mctx.get())) {
                throw std::runtime_error("failed to allocate compute buffers for the KV window gather");
            }
        }
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve_extra(n_tokens, 1, n_outputs_pp, mctx.get());
                break;
            case LLM_ARCH_GLM5NEXT: {
                // Cover the simultaneous two-slot graph after the largest dense
                // single-slot allocation has already established buffer capacity.
                // With separate KV streams, a prompt takes its own ubatches and the
                // multi-sequence ubatches hold a few tokens of each sequence.
                const uint32_t n_multi = std::min(n_tokens, multi_seq_split_tokens(cparams, memory.get()));
                const uint32_t n_tokens_multi   = n_multi > 0 ? n_multi : n_tokens;
                const uint32_t n_outputs_multi  = n_multi > 0 ? std::min(n_multi, cparams.n_outputs_max) : n_outputs_pp;
                gf = graph_reserve_extra(n_tokens_multi, n_seqs, n_outputs_multi, mctx.get());
                if (!gf) {
                    break;
                }

                const uint32_t n_stream_full = cparams.kv_unified ? 1 : n_seqs;
                auto mctx_multi_inc = memory->init_full_n_seq(n_stream_full, 0, false);
                if (!mctx_multi_inc) {
                    throw std::runtime_error("failed to initialize incremental multi-sequence memory module");
                }
                gf = graph_reserve_extra(n_tokens_multi, n_seqs, n_outputs_multi, mctx_multi_inc.get());
                if (!gf) {
                    break;
                }

                // Indexed-attention workspaces are largest when one sequence uses
                // the entire ubatch. Build a matching one-stream full-cache context;
                // reusing the n_seq_max context gives the indexer incompatible shapes.
                auto mctx_single = memory->init_full_n_seq(1, 0, false);
                if (!mctx_single) {
                    throw std::runtime_error("failed to initialize single-sequence memory module");
                }
                gf = graph_reserve_extra(n_tokens, 1, n_outputs_pp, mctx_single.get());
                if (!gf) {
                    break;
                }

                // Also cover the largest KV length which still selects dense attention.
                auto mctx_dense = memory->init_full_n_seq(
                        1, std::min<uint32_t>(cparams.n_ctx_seq, 32768 - 256), false);
                if (!mctx_dense) {
                    throw std::runtime_error("failed to initialize dense-prefill memory module");
                }
                gf = graph_reserve_extra(n_tokens, 1, n_outputs_pp, mctx_dense.get());
                if (!gf) {
                    break;
                }

                // The first request reaches this dense transition on its second
                // prefill chunk (for example n_tokens=952, n_kv=2048 with a
                // 1024-token ubatch). It can require substantially more CUDA
                // workspace than both the first chunk and indexed attention.
                auto mctx_transition = memory->init_full_n_seq(
                        1, std::min<uint32_t>(cparams.n_ctx_seq, 2*n_tokens), false);
                if (!mctx_transition) {
                    throw std::runtime_error("failed to initialize transition-prefill memory module");
                }
                gf = graph_reserve_extra(n_tokens, 1, n_outputs_pp, mctx_transition.get());
                if (!gf) {
                    break;
                }

                // Finish on the short-KV graph most likely to be used at startup.
                auto mctx_short = memory->init_full_n_seq(
                        1, std::min<uint32_t>(cparams.n_ctx_seq, n_tokens + 256), false);
                if (!mctx_short) {
                    throw std::runtime_error("failed to initialize short-prefill memory module");
                }
                gf = graph_reserve_extra(n_tokens, 1, n_outputs_pp, mctx_short.get());
                break;
            }
            default:
                if (const uint32_t n_multi = std::min(n_tokens, dsv4_multi_seq_tokens(cparams, memory.get())); n_multi > 0) {
                    // the largest multi-sequence ubatch, then the one-sequence prompt ubatch
                    gf = graph_reserve_extra(n_multi, n_seqs, std::min(n_multi, cparams.n_outputs_max), mctx.get());
                    if (gf) {
                        auto mctx_single = memory->init_full_n_seq(1);
                        gf = graph_reserve_extra(n_tokens, 1, n_outputs_pp, mctx_single.get());
                    }
                    break;
                }
                gf = graph_reserve_extra(n_tokens, n_seqs, n_outputs_pp, mctx.get());
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
    if (share_rereserve) {
        LLAMA_LOG_WARN("%s: compute buffers reserved again in %.1f ms\n", __func__, (t_end_us - t_start_us)/1000.0);
        share_rereserve = false;
    }
}

// true if a prompt ubatch of this attention variant and padded KV length fits the sizes of the last max prefill reservation
bool llama_context::max_prefill_fits(int variant, uint32_t n_kv) const {
    return variant == max_prefill_variant && (variant < 0 || n_kv <= max_prefill_n_kv);
}

// true if the main scheduler still has the layout of the last max prefill reservation
bool llama_context::max_prefill_in_place() const {
    ggml_backend_sched_t sched_main = graph_sched_main();
    return sched_main == max_prefill_sched && ggml_backend_sched_get_n_reserves(sched_main) == max_prefill_reserves;
}

void llama_context::reserve_max_prefill(
        const llama_ubatch & ubatch, const llama_memory_context_i * mctx,
        uint32_t n_tokens_all, uint32_t n_tokens_prev) {
    // The disease (prefill ubatches outgrowing the recorded compute reservation
    // every time n_kv / pool sizes grow) is generic; only the variant decision
    // depends on the memory type. See the per-variant checks below.
    {
        const char * LLAMA_MAX_RESERVE = getenv("LLAMA_MAX_RESERVE");
        const char * LLAMA_GLM5_MAX_RESERVE = getenv("LLAMA_GLM5_MAX_RESERVE");
        if ((LLAMA_MAX_RESERVE && atoi(LLAMA_MAX_RESERVE) == 0)
                || (LLAMA_GLM5_MAX_RESERVE && atoi(LLAMA_GLM5_MAX_RESERVE) == 0)) {
            return;
        }
    }
    // DeepSeek's prefill is CPU routed-expert bound and ubatch-invariant, so the
    // reallocation barriers are noise there; it stays out of the generic path.
    if (model.arch == LLM_ARCH_DEEPSEEK4 || model.arch == LLM_ARCH_DEEPSEEK41) {
        return;
    }
    // Prompt-sized ubatches of one sequence only. A llama_decode call with a single ubatch also takes it: the server
    // splits a prompt into several calls (it ends one before the last ubatch for a checkpoint), and a short ubatch
    // (2044 tokens) recorded sizes that the full ubatch of the next call (2048) outgrew. That reallocation
    // synchronized every backend and drained the ubatch still in flight from the previous call. Later ubatches of a
    // call are already covered, except one that switches the attention variant (below).
    GGML_UNUSED(n_tokens_all);
    if (2*ubatch.n_tokens < cparams.n_ubatch || ubatch.n_seqs_unq != 1 || ubatch.n_seq_tokens != ubatch.n_tokens) {
        return;
    }
    if (model.hparams.no_alloc) {
        return;
    }
    // The incoming ubatches must all use the same reservation variant. Memory
    // types with a single prefill variant need no check. The pooled indexer is
    // the only known case that can switch shape mid-prefill: a cold pool cache
    // changes shape after the first ubatch, so that case keeps today's behavior
    // instead of recording a reservation nothing else fits. llama_memory_hybrid_idx
    // shadows the base get_idx() with its own indexer cache and has no pool cache,
    // so the base accessor being null covers it.
    const auto * mctx_hyb = dynamic_cast<const llama_memory_hybrid_context *>(mctx);
    const bool is_hyb_idx = dynamic_cast<const llama_memory_hybrid_idx_context *>(mctx) != nullptr;
    if (mctx_hyb != nullptr && mctx_hyb->get_idx() != nullptr) {
        auto * cache = mctx_hyb->get_kpool_cache();
        if (cache != nullptr) {
            const char * pool_cache_env = getenv("LLAMA_GLM5_POOL_CACHE");
            if (pool_cache_env != nullptr && atoi(pool_cache_env) == 0) {
                cache = nullptr;
            }
        }
        if (cache != nullptr) {
            const uint32_t n_select = model.hparams.indexer_top_k + model.hparams.indexer_kpool - 1;
            const bool scoring = cparams.n_ctx > n_select;
            const uint32_t stream0 = mctx_hyb->get_idx()->get_stream_base();
            const uint32_t n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
            if (scoring && cache->needs_rebuild(stream0, n_stream)) {
                return;
            }
        }
    }

    // simulate the largest prefill ubatch in the variant that later ubatches will
    // use: the single-stream warm pool-cache shape for plain hybrid memory; the
    // worst-case full cache for everything else, including the derived hybrid-idx
    // context type and non-hybrid memories (whose init_full_n_seq defaults here)
    // GLM-5 kpool memory: prefill ubatches use dense masked attention below the indexed-attention crossover and
    // indexed attention above it (llama_kpool_indexed_attn_enabled), two graph topologies. A reservation of one
    // fits no ubatch of the other, so each of those reallocated, and every reallocation synchronizes all devices,
    // which serializes a pipeline-parallel prefill: a 49K-token prompt at -c 53248 reallocated on every ubatch
    // below 32K (the full-cache reservation is indexed). Reserve the largest graph of the variant this ubatch
    // uses, and reserve again when a later ubatch switches variant (once per crossing).
    int variant = -1;
    uint32_t n_kv_max = cparams.n_ctx_seq;
    if (mctx_hyb != nullptr && !is_hyb_idx && mctx_hyb->get_kpool_cache() != nullptr && ubatch.n_tokens > 0) {
        const uint32_t pad = 256;
        const int64_t n_tps = ubatch.n_seq_tokens;
        const int64_t n_kv_ubatch = GGML_PAD((int64_t) ubatch.pos[ubatch.n_tokens - 1] + 1, pad);
        variant = llama_kpool_indexed_attn_enabled(n_kv_ubatch, n_tps) ? 1 : 0;
        if (variant == 0) {
            // the largest padded KV length that stays dense
            while (n_kv_max > pad && llama_kpool_indexed_attn_enabled(n_kv_max, n_tps)) {
                n_kv_max = (n_kv_max - 1)/pad*pad;
            }
        }
    }
    // LLAMA_VERIFY_STREAMS: the final KV length of a background prompt (set_prompt_background_end) bounds the indexed
    // reservation. The full-context one has masks of n_ctx_seq cells per query, more than the free memory of a long context.
    const uint32_t n_kv_ubatch = ubatch.n_tokens > 0 ? (uint32_t) GGML_PAD((int64_t) ubatch.pos[ubatch.n_tokens - 1] + 1, 256) : 0;
    if (variant == 1 && decode_bg && bg_pos_end > 0) {
        n_kv_max = std::min<uint32_t>(n_kv_max, std::max<uint32_t>(n_kv_ubatch, GGML_PAD((uint32_t) bg_pos_end + cparams.n_ubatch, bg_n_kv_step)));
    }
    const bool fits = max_prefill_fits(variant, n_kv_ubatch);
    if (n_tokens_prev != 0 && fits) {
        return;
    }
    // Only plain hybrid memory gives a one-stream full-cache context. For other memories init_full() exposes all
    // n_seq_max streams. A one-sequence ubatch against that context splits its rows over the streams, so the
    // attention output does not reshape back to n_tokens rows (GGML_ASSERT in ggml_reshape_3d with -np 3 and a
    // DSpark draft). The regular reservation already covers n_ubatch tokens over n_seq_max streams.
    if ((mctx_hyb == nullptr || is_hyb_idx) && !cparams.kv_unified && cparams.n_seq_max > 1) {
        return;
    }

    auto mctx_max = (mctx_hyb != nullptr && !is_hyb_idx)
            ? memory->init_full_n_seq(1, n_kv_max, false)
            : memory->init_full();
    if (!mctx_max) {
        LLAMA_LOG_WARN("%s: failed to initialize max-prefill memory module\n", __func__);
        return;
    }

    const uint32_t n_tokens_max  = std::min(cparams.n_ctx, cparams.n_ubatch);

    // Reserve with this ubatch's own output count. The output rows change the
    // graph topology (and backend sampling adds per-output nodes), so a
    // reservation with a different count does not match the ubatches that
    // follow: the first of them then re-reserves for its own, still short, KV
    // length and every later ubatch reallocates again as n_kv grows. MTP
    // targets cap n_outputs_max at the draft length, which exposed this.
    uint32_t n_outputs = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n_outputs += ubatch.output[i] != 0;
    }
    // Without backend samplers, a ubatch with no outputs builds the same nodes as one with a single output, only
    // with empty output tensors, so it fits a single-output reservation. The last ubatch of a prompt carries the one
    // output: against a no-output reservation its out_ids (and logits) outgrew the recorded sizes, and the
    // reallocation synchronized every backend, draining the pipeline ahead of the prompt's last ubatch.
    if (n_outputs == 0 && sampling.samplers.empty()) {
        n_outputs = 1;
    }

    // An earlier logical batch of this prompt (or an earlier prompt) left the reservation in place. Reserving again
    // would synchronize every backend and drain the ubatches still in flight from the previous llama_decode call.
    if (fits && max_prefill_in_place()) {
        return;
    }

    LLAMA_LOG_INFO("%s: pre-recording max prefill reservation for %u ubatch tokens (KV length %u)\n", __func__, n_tokens_max, n_kv_max);

    static const bool build_log = getenv("LLAMA_GRAPH_BUILD_LOG") != nullptr;
    const int64_t t_reserve_us = build_log ? ggml_time_us() : 0;
    // with verify streams, the alternate schedulers have their own backends and buffers: their graphs stay valid, and the
    // generating slots do not build and capture their verify graphs again
    if (!graph_reserve(n_tokens_max, 1, n_outputs, mctx_max.get(), false, nullptr, verify_streams)) {
        LLAMA_LOG_WARN("%s: failed to reserve max prefill graph\n", __func__);
        max_prefill_sched   = nullptr;
        max_prefill_variant = variant; // do not retry for every later ubatch
        max_prefill_n_kv    = UINT32_MAX;
    } else {
        // graph_reserve reserved on the main scheduler, which is `sched` again now
        max_prefill_sched    = sched.get();
        max_prefill_reserves = ggml_backend_sched_get_n_reserves(sched.get());
        max_prefill_variant  = variant;
        max_prefill_n_kv     = n_kv_max;
    }
    if (build_log) {
        LLAMA_LOG_WARN("max prefill reserve: n_tokens %u n_kv %u variant %d background %d: %.2f ms\n", n_tokens_max, n_kv_max, variant, (int) decode_bg,
                (ggml_time_us() - t_reserve_us)/1e3);
    }
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    if (verify_streams) {
        // synchronize_outputs() leaves the background prompt work of the main scheduler running
        verify_streams_sync_alt();
        if (!sync_outputs_only || main_fg_queued) {
            verify_streams_sync_main();
        }
    } else {
        ggml_backend_sched_synchronize(sched.get());
    }

    if (!state_async_pending.empty()) {
        state_async_finish();
    }

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        // the memory module uses the main scheduler
        graph_alt_reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;

        // A backend-resident MTP iteration has exactly one output token per
        // active sequence. A pending cache operation (for example a context
        // shift) resets the scheduler while that mode is active, so reserve a
        // representative all-output draft graph rather than the normal
        // worst-case prefill graph. The latter may have fewer outputs than
        // tokens and is not a valid device-draft topology.
        const bool mtp_device_draft =
                cparams.mtp_device_draft_mode != LLAMA_MTP_DEVICE_DRAFT_DISABLED;
        const uint32_t n_tokens = mtp_device_draft ?
                n_seqs : std::min(cparams.n_ctx, cparams.n_ubatch);
        const uint32_t n_outputs_max = mtp_device_draft ?
                n_tokens : std::min(n_tokens, cparams.n_outputs_max);

        ggml_cgraph * gf = nullptr;
        if (const uint32_t n_multi = std::min(n_tokens, multi_seq_split_tokens(cparams, memory.get())); n_multi > 0 && !mtp_device_draft) {
            // the shapes of sched_reserve: a full ubatch over several streams is larger and never occurs
            gf = graph_reserve(n_multi, n_seqs, std::min(n_multi, cparams.n_outputs_max), mctx.get());
            if (gf) {
                auto mctx_single = memory->init_full_n_seq(1);
                gf = graph_reserve(n_tokens, 1, n_outputs_max, mctx_single.get());
            }
        } else {
            gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        }
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

ggml_tensor * llama_context::get_embeddings_nextn_tensor() {
    return embd_nextn_backend_rows > 0 ? &embd_nextn_backend_view : nullptr;
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

void llama_context::set_sampler_lazy(bool enable) {
    if (cparams.sampler_lazy != enable) {
        synchronize();
        cparams.sampler_lazy = enable;
        sampler_changed();
        sched_need_reserve = true;
    }
}

bool llama_context::get_sampled_prob_ith(int32_t idx, llama_token token, float * prob) {
    if (!cparams.sampler_lazy || !prob || token < 0 || (uint32_t) token >= model.vocab.n_tokens()) {
        return false;
    }
    synchronize_outputs();
    output_reorder();
    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.lazy_rows.size() || !sampling.lazy_rows[row].probs_by_token) {
            return false;
        }
        ggml_backend_tensor_get(sampling.lazy_rows[row].probs_by_token, prob, (size_t) token*sizeof(float), sizeof(float));
        return true;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid output id %d, reason: %s\n", __func__, idx, err.what());
        return false;
    }
}

static void sampling_read_lazy(ggml_tensor * & tensor, void * dst) {
    if (tensor) {
        ggml_backend_tensor_get(tensor, dst, 0, ggml_nbytes(tensor));
        tensor = nullptr;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        sampling_read_lazy(sampling.lazy_rows[row].probs, sampling.probs.data + row*model.vocab.n_tokens());
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        sampling_read_lazy(sampling.lazy_rows[row].logits, sampling.logits.data + row*model.vocab.n_tokens());
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            sampling_read_lazy(sampling.lazy_rows[row].candidates, sampling.candidates.data + row*model.vocab.n_tokens());
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn == value && cparams.embeddings_nextn_masked == masked) {
        return;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;

    // Unmasked NextN extraction keeps full-length hidden states alive until
    // the head instead of gathering output rows before it. This changes the
    // allocation topology and can substantially increase prefill workspace.
    sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn_host(bool value) {
    embd_nextn_host = value;
}

// delphi: optional backend procs that download layer features on a side stream, so that the PCIe transfer does not
// hold up the next ubatch of the compute stream (ggml-cuda: ggml_backend_cuda_get_tensor_async_side)
typedef void (*llama_get_tensor_async_side_t)(ggml_backend_t, const ggml_tensor *, void *, size_t, size_t);
typedef void (*llama_side_join_t)(ggml_backend_t);

static void * feature_side_proc(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

static llama_get_tensor_async_side_t feature_side_get(ggml_backend_t backend) {
    return (llama_get_tensor_async_side_t) feature_side_proc(backend, "ggml_backend_get_tensor_async_side");
}

static llama_side_join_t feature_side_join(ggml_backend_t backend) {
    return (llama_side_join_t) feature_side_proc(backend, "ggml_backend_side_join");
}

void llama_context::set_feature_handoff(bool enable) {
    std::lock_guard<std::mutex> lock(handoff_mutex);
    handoff_enabled = enable;
}

void llama_context::feature_handoff_acquire() {
    std::unique_lock<std::mutex> lock(handoff_mutex);

    // a claimed set stays untouched until released; a decode after two claimed calls waits here
    if (handoff_slots[handoff_cur].taken) {
        handoff_cur ^= 1;
    }
    handoff_cv.wait(lock, [&] { return !handoff_slots[handoff_cur].taken; });

    auto & slot = handoff_slots[handoff_cur];

    const size_t align = 256;
    auto padded = [&](size_t n_floats) { return GGML_PAD(n_floats*sizeof(float), align); };

    size_t need = padded(embd_nextn.size);
    for (const auto & v : embd_layer_inp) {
        need += padded(v.size);
    }
    if (need == 0) {
        return;
    }

    if (!slot.buf || ggml_backend_buffer_get_size(slot.buf.get()) < need) {
        auto * buft = ggml_backend_cpu_buffer_type();
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        slot.buf.reset(ggml_backend_buft_alloc_buffer(buft, need));
        GGML_ASSERT(slot.buf && "failed to allocate the feature hand-off buffer");
        slot.pinned = buft != ggml_backend_cpu_buffer_type();
    }

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(slot.buf.get());
    size_t offset = 0;

    slot.nextn = embd_nextn.size > 0 ? (float *) (base + offset) : nullptr;
    offset += padded(embd_nextn.size);
    if (embd_nextn.size > 0) {
        embd_nextn.data = slot.nextn;
    }

    slot.layer_inp.assign(embd_layer_inp.size(), nullptr);
    for (size_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (embd_layer_inp[il].size > 0) {
            slot.layer_inp[il] = (float *) (base + offset);
            embd_layer_inp[il].data = slot.layer_inp[il];
            offset += padded(embd_layer_inp[il].size);
        }
    }
}

int32_t llama_context::feature_handoff_take() {
    std::lock_guard<std::mutex> lock(handoff_mutex);

    auto & slot = handoff_slots[handoff_cur];
    if (!handoff_enabled || slot.taken || !slot.buf) {
        return -1;
    }

    // the feature copies of the last decode are queued on the backend streams (or side streams joined here); mark their end
    for (auto & backend : backends) {
        if (auto * join = feature_side_join(backend.get())) {
            join(backend.get());
        }
    }
    if (slot.events.empty()) {
        slot.events.resize(backends.size());
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backends[i].get());
            slot.events[i].reset(dev ? ggml_backend_event_new(dev) : nullptr);
        }
    }
    for (size_t i = 0; i < backends.size(); ++i) {
        if (slot.events[i]) {
            ggml_backend_event_record(slot.events[i].get(), backends[i].get());
        }
    }

    slot.taken = true;
    return handoff_cur;
}

const float * llama_context::feature_handoff_nextn(int32_t ticket) {
    GGML_ASSERT(ticket == 0 || ticket == 1);
    auto & slot = handoff_slots[ticket];
    GGML_ASSERT(slot.taken);
    for (auto & ev : slot.events) {
        if (ev) {
            ggml_backend_event_synchronize(ev.get());
        }
    }
    return slot.nextn;
}

const float * llama_context::feature_handoff_layer_inp(int32_t ticket, uint32_t lid) {
    GGML_ASSERT(ticket == 0 || ticket == 1);
    auto & slot = handoff_slots[ticket];
    GGML_ASSERT(slot.taken && lid < slot.layer_inp.size() && slot.layer_inp[lid] != nullptr);
    for (auto & ev : slot.events) {
        if (ev) {
            ggml_backend_event_synchronize(ev.get());
        }
    }
    return slot.layer_inp[lid];
}

void llama_context::feature_handoff_release(int32_t ticket) {
    GGML_ASSERT(ticket == 0 || ticket == 1);
    {
        std::lock_guard<std::mutex> lock(handoff_mutex);
        handoff_slots[ticket].taken = false;
    }
    handoff_cv.notify_all();
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;

    if (cparams.embeddings_layer_inp_dev) {
        layer_inp_dev_alloc();
    }
}

void llama_context::set_embeddings_layer_inp_dev(bool value) {
    if (cparams.embeddings_layer_inp_dev == value) {
        return;
    }
    cparams.embeddings_layer_inp_dev = value;
    sched_need_reserve = true;
    layer_inp_dev_alloc();
}

void llama_context::layer_inp_dev_alloc() {
    synchronize();

    layer_inp_dev     = nullptr;
    buf_layer_inp_dev.reset();
    ctx_layer_inp_dev.reset();

    if (!cparams.embeddings_layer_inp_dev) {
        return;
    }

    int64_t n_layer_inp = 0;
    for (bool enabled : cparams.embeddings_layer_inp) {
        n_layer_inp += enabled ? 1 : 0;
    }
    if (n_layer_inp == 0) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_layer_inp_dev.reset(ggml_init(params));

    layer_inp_dev = ggml_new_tensor_2d(ctx_layer_inp_dev.get(), GGML_TYPE_F32, n_layer_inp*model.hparams.n_embd, cparams.n_batch);
    ggml_set_name(layer_inp_dev, "layer_inp_dev");

    auto * layer_inp_dev_dev = model.dev_output();
    auto * buft = layer_inp_dev_dev ? ggml_backend_dev_buffer_type(layer_inp_dev_dev) : ggml_backend_cpu_buffer_type();

    buf_layer_inp_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_layer_inp_dev.get(), buft));
    if (!buf_layer_inp_dev) {
        // e.g. a large n_batch that does not fit next to the model, keep the host copies
        LLAMA_LOG_WARN("%s: failed to allocate the layer input device buffer, using host copies\n", __func__);
        cparams.embeddings_layer_inp_dev = false;
        layer_inp_dev     = nullptr;
        ctx_layer_inp_dev.reset();
        return;
    }

    LLAMA_LOG_INFO("%s: %10s layer input buffer size = %8.2f MiB (%d layers)\n", __func__,
            ggml_backend_buffer_name(buf_layer_inp_dev.get()), ggml_backend_buffer_get_size(buf_layer_inp_dev.get())/1024.0/1024.0, (int) n_layer_inp);
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_mtp_index_reuse(bool reuse) {
    memory->set_mtp_index_reuse(reuse);
}

bool llama_context::mtp_device_draft_begin(uint32_t n_steps) {
    const char * env = std::getenv("LLAMA_MTP_DEVICE_DRAFT");
    if (env != nullptr && std::atoi(env) == 0) {
        return false;
    }

    const auto & hparams = model.hparams;
    if (n_steps == 0 || cparams.n_seq_max == 0 || model.arch != LLM_ARCH_GLM5NEXT ||
            cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || hparams.n_layer_nextn != 1 ||
            sampling.samplers.size() != cparams.n_seq_max) {
        LLAMA_LOG_DEBUG("%s: unavailable (steps=%u, seq=%u, arch=%s, ctx_type=%d, nextn=%u, samplers=%zu)\n",
                __func__, n_steps, cparams.n_seq_max, llm_arch_name(model.arch), (int) cparams.ctx_type,
                hparams.n_layer_nextn, sampling.samplers.size());
        return false;
    }

    // Token ids are stored temporarily as F32 because SET_ROWS does not accept
    // an I32 destination on all supported backends.
    if (model.vocab.n_tokens() > (1 << 24)) {
        return false;
    }

    // The device loop currently reproduces the MTP driver's top-10 greedy
    // selection. Any other chain keeps using the host-driven fallback.
    for (uint32_t seq_id = 0; seq_id < cparams.n_seq_max; ++seq_id) {
        const auto it = sampling.samplers.find(seq_id);
        if (it == sampling.samplers.end() || strcmp(llama_sampler_name(it->second), "chain") != 0 ||
                llama_sampler_chain_n(it->second) != 1) {
            return false;
        }

        const auto * sampler = llama_sampler_chain_get(it->second, 0);
        if (sampler == nullptr || strcmp(llama_sampler_name(sampler), "+top-k") != 0) {
            LLAMA_LOG_DEBUG("%s: seq_id=%u has no backend top-k sampler (name=%s)\n",
                    __func__, seq_id, sampler ? llama_sampler_name(sampler) : "none");
            return false;
        }
    }

    sched_reserve(); // returns at once when nothing changed

    if ((size_t) n_steps > std::numeric_limits<size_t>::max()/cparams.n_seq_max) {
        return false;
    }
    const size_t n_results = (size_t) n_steps*cparams.n_seq_max;
    if (n_results > (size_t) std::numeric_limits<int32_t>::max()) {
        return false;
    }

    const bool need_alloc = !mtp_device_draft_buf ||
            (size_t) mtp_device_draft_cache.result_tokens->ne[1] < n_results;
    if (need_alloc) {
        synchronize();
        mtp_device_draft_buf.reset();
        mtp_device_draft_ctx.reset();
        mtp_device_draft_cache = {};

        ggml_backend_t cache_backend = nullptr;
        const ggml_backend_dev_t cache_dev = model.dev_output();
        for (auto & backend : backends) {
            if (ggml_backend_get_device(backend.get()) == cache_dev) {
                cache_backend = backend.get();
                break;
            }
        }
        if (!cache_backend) {
            return false;
        }

        ggml_init_params cache_params = {
            /*.mem_size   =*/ 4*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        mtp_device_draft_ctx.reset(ggml_init(cache_params));
        if (!mtp_device_draft_ctx) {
            return false;
        }

        mtp_device_draft_cache.tokens = ggml_new_tensor_2d(
                mtp_device_draft_ctx.get(), GGML_TYPE_F32, 1, cparams.n_seq_max);
        mtp_device_draft_cache.hidden = ggml_new_tensor_2d(
                mtp_device_draft_ctx.get(), GGML_TYPE_F32, hparams.n_embd_out(), cparams.n_seq_max);
        mtp_device_draft_cache.result_tokens = ggml_new_tensor_2d(
                mtp_device_draft_ctx.get(), GGML_TYPE_F32, 1, n_results);
        mtp_device_draft_cache.result_probs = ggml_new_tensor_2d(
                mtp_device_draft_ctx.get(), GGML_TYPE_F32, 1, n_results);
        ggml_set_name(mtp_device_draft_cache.tokens,        "mtp_device_tokens");
        ggml_set_name(mtp_device_draft_cache.hidden,        "mtp_device_hidden");
        ggml_set_name(mtp_device_draft_cache.result_tokens, "mtp_device_result_tokens");
        ggml_set_name(mtp_device_draft_cache.result_probs,  "mtp_device_result_probs");

        auto * cache_buft = ggml_backend_dev_buffer_type(cache_dev);
        if (!cache_buft) {
            mtp_device_draft_ctx.reset();
            mtp_device_draft_cache = {};
            return false;
        }
        mtp_device_draft_buf.reset(
                ggml_backend_alloc_ctx_tensors_from_buft(mtp_device_draft_ctx.get(), cache_buft));
        if (!mtp_device_draft_buf) {
            mtp_device_draft_ctx.reset();
            mtp_device_draft_cache = {};
            return false;
        }

        // Graph results retain pointers to the persistent destinations.
        graph_alt_reset();

        LLAMA_LOG_INFO("%s: %10s MTP device draft cache size = %.2f KiB\n", __func__,
                ggml_backend_buffer_name(mtp_device_draft_buf.get()),
                ggml_backend_buffer_get_size(mtp_device_draft_buf.get())/1024.0);
    }

    mtp_device_draft_cache.n_steps   = n_steps;
    mtp_device_draft_cache.n_seq_max = cparams.n_seq_max;
    mtp_device_draft_cache.step      = 0;
    cparams.mtp_device_draft_mode    = LLAMA_MTP_DEVICE_DRAFT_SEED;
    return true;
}

void llama_context::mtp_device_draft_advance() {
    GGML_ASSERT(cparams.mtp_device_draft_mode != LLAMA_MTP_DEVICE_DRAFT_DISABLED);
    GGML_ASSERT(mtp_device_draft_cache.step + 1 < mtp_device_draft_cache.n_steps);

    ++mtp_device_draft_cache.step;
    cparams.mtp_device_draft_mode = LLAMA_MTP_DEVICE_DRAFT_REUSE;
}

bool llama_context::mtp_device_draft_finish(
        llama_token * tokens, float * probs, size_t n_results) {
    if (cparams.mtp_device_draft_mode == LLAMA_MTP_DEVICE_DRAFT_DISABLED) {
        return false;
    }

    const size_t expected = (size_t) mtp_device_draft_cache.n_steps*mtp_device_draft_cache.n_seq_max;
    if (tokens == nullptr || probs == nullptr || n_results != expected) {
        cparams.mtp_device_draft_mode = LLAMA_MTP_DEVICE_DRAFT_DISABLED;
        mtp_device_draft_cache.step = 0;
        return false;
    }

    synchronize();

    std::vector<float> token_values(expected);
    ggml_backend_tensor_get(
            mtp_device_draft_cache.result_tokens, token_values.data(), 0, expected*sizeof(float));
    ggml_backend_tensor_get(
            mtp_device_draft_cache.result_probs, probs, 0, expected*sizeof(float));
    for (size_t i = 0; i < expected; ++i) {
        tokens[i] = (llama_token) token_values[i];
    }

    cparams.mtp_device_draft_mode = LLAMA_MTP_DEVICE_DRAFT_DISABLED;
    mtp_device_draft_cache.step = 0;
    return true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

std::string llama_context::sampler_sig() const {
    std::string sig;
    for (const auto & [seq_id, smpl] : sampling.samplers) {
        sig += std::to_string(seq_id);
        sig += ':';
        const int n = llama_sampler_chain_n(smpl);
        for (int i = 0; i < n; ++i) {
            sig += llama_sampler_name(llama_sampler_chain_get(smpl, i));
            sig += ',';
        }
        sig += ';';
    }
    return sig;
}

void llama_context::sampler_changed() {
    static const bool always = [] {
        const char * e = getenv("LLAMA_SAMPLER_RESERVE_ALWAYS");
        return e && atoi(e) != 0;
    }();

    samplers_gen++;

    // without the switch, sched_reserve() compares the chains with those of the last reservation (the server sets a new
    // sampler object for each request and removes it at the end; a full reservation recreates the scheduler and its buffers)
    if (always) {
        sched_need_reserve = true;
    }
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sampler_changed();
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto it = sampling.samplers.find(seq_id);
        if (it != sampling.samplers.end() && it->second == sampler) {
            // the same object stays attached (the server keeps it between requests with equal parameters): its
            // backend state and the graphs built with it remain valid
            return true;
        }

        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sampler_changed();

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sampler_changed();
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sampler_changed();

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    const uint32_t trace_ubatch_id = ggml_exec_trace_enabled() ? ggml_exec_trace_ubatch_begin(
            ubatch.n_tokens, ubatch.n_seqs, ubatch.n_seqs_unq, n_outputs) : 0;

    static const bool build_log = getenv("LLAMA_GRAPH_BUILD_LOG") != nullptr;
    const int64_t t_apply_us0 = build_log ? ggml_time_us() : 0;
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        ggml_exec_trace_ubatch_end(trace_ubatch_id, ret);
        return nullptr;
    }

    set_backend_n_threads(ubatch.n_tokens > 1 ? cparams.n_threads_batch : cparams.n_threads);

    auto * res = get_gf_res_prev();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    auto gparams = graph_params(res, ubatch, mctx, gtype);

    const int64_t t_apply_us = build_log ? ggml_time_us() - t_apply_us0 : 0;
    int64_t t_build_us = -1;
    int64_t t_alloc_us = -1;

    auto can_reuse = [&]() {
        return !graph_reuse_disable && gf_res_prev_active == res && gf_res_prev_threads == backend_n_threads && res->can_reuse(gparams);
    };

    bool reuse = can_reuse();
    const bool small = graph_alt_max_tokens > 0 && ubatch.n_tokens <= graph_alt_max_tokens;
    if (!reuse && (graph_slot_live != 0 || small)) {
        // large ubatches always go to the main scheduler, which is reserved for them. A small one takes the
        // alternate scheduler that holds its graph; else the one whose stale graph has the same topology (e.g.
        // after the KV view grew), whose buffers and device graphs fit it; else the least recently used one.
        int target = 0;
        if (small) {
            target = -1;
            int same = -1;
            int lru  = -1;
            for (int id = 1; id < (int) graph_slots.size() && target < 0; ++id) {
                if (id != graph_slot_live && !graph_slots[id].sched) {
                    lru = id; // never used (last_use 0): nothing to reuse, and the later ones are unused too
                    break;
                }
                graph_slot_activate(id);
                res     = get_gf_res_prev();
                gparams = graph_params(res, ubatch, mctx, gtype);
                if (can_reuse()) {
                    target = id;
                } else {
                    if (same < 0 && gf_res_prev_active == res && res->same_topology(gparams)) {
                        same = id;
                    }
                    if (lru < 0 || graph_slots[id].last_use < graph_slots[lru].last_use) {
                        lru = id;
                    }
                }
            }
            if (target < 0) {
                target = same >= 0 ? same : lru;
            }
        }
        graph_slot_activate(target);
        res     = get_gf_res_prev();
        gparams = graph_params(res, ubatch, mctx, gtype);
        reuse   = can_reuse();
    }
    graph_slots[graph_slot_live].last_use = ++graph_slot_clock;
    verify_streams_ubatch(ubatch, n_outputs);
    const bool async_inputs = verify_input_async && (graph_slot_live != 0 || model.arch == LLM_ARCH_DFLASH);
    auto * gf = res->get_gf();

    if (reuse) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        // with staggered verify, the graphs of the other sequences are in other graph slots (own schedulers and
        // inputs), so wait only for this scheduler's last graph and let their work continue on the devices
        if (async_inputs || (cparams.pipeline_parallel && staggered_verify)) {
            ggml_backend_sched_wait_graph(sched.get());
        } else if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        // The old graph's host inputs remain in use until its asynchronous uploads complete. This also covers
        // single-device draft contexts, which do not use pipeline parallelism.
        if (async_inputs) {
            ggml_backend_sched_wait_graph(sched.get());
        }
        if (build_log) {
            const bool fresh = gf_res_prev_active != res || res->get_gf() == nullptr;
            LLAMA_LOG_WARN("graph build: ctx %p n_tokens %u slot %d reason %s\n", (void *) this, ubatch.n_tokens, graph_slot_live,
                    fresh ? "slot" : gf_res_prev_threads != backend_n_threads ? "threads" : res->reuse_fail_reason(gparams).c_str());
        }
        gf_res_prev_active = nullptr;
        res->reset();
        n_graph_built++;

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        t_build_us = ggml_time_us();
        gf = model.build_graph(gparams);
        t_build_us = ggml_time_us() - t_build_us;

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            ggml_exec_trace_ubatch_end(trace_ubatch_id, ret);
            return nullptr;
        }

        t_alloc_us = ggml_time_us();
        const uint64_t n_reserves0 = ggml_backend_sched_get_n_reserves(sched.get());
        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            ggml_exec_trace_ubatch_end(trace_ubatch_id, ret);
            return nullptr;
        }
        t_alloc_us = ggml_time_us() - t_alloc_us;
        if (build_log && ggml_backend_sched_get_n_reserves(sched.get()) != n_reserves0) {
            LLAMA_LOG_WARN("graph build: ctx %p n_tokens %u slot %d new compute layout\n", (void *) this, ubatch.n_tokens, graph_slot_live);
        }

        gf_res_prev_active  = res;
        gf_res_prev_threads = backend_n_threads;
    }

    int64_t t_inputs_us = 0;
    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        const int64_t t_inputs_us0 = build_log ? ggml_time_us() : 0;
        res->set_inputs(&ubatch);
        t_inputs_us = build_log ? ggml_time_us() - t_inputs_us0 : 0;

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const int64_t t_compute_us0 = build_log ? ggml_time_us() : 0;
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        ggml_exec_trace_ubatch_end(trace_ubatch_id, ret);
        return nullptr;
    }
    if (build_log && t_build_us >= 0) {
        // host times; the first compute of a new graph includes its eager run or CUDA/HIP graph capture
        LLAMA_LOG_WARN("graph build time: ctx %p n_tokens %u apply %.2f build %.2f alloc %.2f inputs %.2f compute %.2f ms\n", (void *) this,
                ubatch.n_tokens, t_apply_us/1e3, t_build_us/1e3, t_alloc_us/1e3, t_inputs_us/1e3, (ggml_time_us() - t_compute_us0)/1e3);
    }

    ret = GGML_STATUS_SUCCESS;
    if (trace_ubatch_id) {
        ggml_exec_trace_ubatch_end(trace_ubatch_id, ret);
    }

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    if (memory_released) {
        share_restore_memory();
    }
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd || batch_inp.embd_tensor);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_embd = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, model.hparams.n_embd_out(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn_host && embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    if (memory_released) {
        share_restore_memory();
    }
    // LLAMA_MOE_CACHE_TIER_CYCLE=N (test): every N decodes, drop the newest run-time tier and add it again
    static const int tier_cycle = [] { const char * e = getenv("LLAMA_MOE_CACHE_TIER_CYCLE"); return e ? atoi(e) : 0; }();
    if (tier_cycle > 0 && expert_cache && ++share_test_decodes % tier_cycle == 0 && expert_cache->n_tiers() > 1) {
        const auto [id, backend] = expert_cache->tier_list().back();
        const size_t bytes = expert_cache->tier_bytes(id);
        share_drop_tier(id);
        share_add_tier(ggml_backend_get_device(backend), bytes);
    }
    // LLAMA_SHARE_TEST_RELEASE_COMPUTE=N (test): every N decodes, release the compute buffers
    static const int release_cycle = [] { const char * e = getenv("LLAMA_SHARE_TEST_RELEASE_COMPUTE"); return e ? atoi(e) : 0; }();
    if (release_cycle > 0 && ++share_test_releases % release_cycle == 0) {
        share_release_compute();
    }
    GGML_ASSERT(batch_inp.token || batch_inp.embd || batch_inp.embd_tensor);

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    // an MTP batch with embd and no embd_h carries the hidden state in embd; with embd_h, embd holds input (media) embeddings
    const bool    mtp_embd = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && batch_inp.embd && !batch_inp.embd_h;
    // DFlash embd batches carry the fused target features at the encoder input width
    const bool    dflash_embd = model.arch == LLM_ARCH_DFLASH && (batch_inp.embd || batch_inp.embd_tensor);
    const int64_t n_embd  = mtp_embd ? hparams.n_embd_out() : dflash_embd ? hparams.n_embd_inp_enc() : hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // embedding contexts output every token even when batch.logits is not set
    if (has_samplers && (output_all || batch_inp.logits)) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (!output_all && batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, hparams.n_embd_out(), n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();
    embd_nextn_backend_rows = 0;
    verify_streams_decode_begin(n_tokens_all);

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();

    sched_reserve();

    if (expert_cache && expert_cache->needs_refresh(n_tokens_all, cparams.warmup)) {
        // the staging ring of any scheduler may hold arena slots that the refresh fills again
        ggml_backend_sched_synchronize(sched.get());
        ggml_backend_sched_stage_reclaim(sched.get());
        for (auto & slot : graph_slots) {
            if (slot.sched) { ggml_backend_sched_stage_reclaim(slot.sched.get()); }
        }
        expert_cache->refresh();
    }

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    if (handoff_enabled) {
        feature_handoff_acquire();
    }

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;
    int64_t n_tokens_prev_main = 0; // tokens of the earlier ubatches of this call that ran on the main scheduler

    auto prefill_window = llama_dsv41_prefill_window::create(*this, mctx.get());
    if (prefill_window) {
        ggml_status status;
        try {
            status = prefill_window->encode();
        } catch (const std::bad_alloc &) {
            LLAMA_LOG_ERROR("%s: failed to allocate the encoder window checkpoints\n", __func__);
            status = GGML_STATUS_ALLOC_FAILED;
        }
        if (status != GGML_STATUS_SUCCESS) {
            // Some encoder layers have overwritten their windows. No partial prefix can be reused after failure.
            synchronize();
            memory->seq_rm(mctx->get_ubatch().seq_id_unq[0], -1, -1);
            return status == GGML_STATUS_ABORTED ? 2 : status == GGML_STATUS_ALLOC_FAILED ? -2 : -3;
        }
    }
    size_t prefill_chunk = 0;
    do {
        // a window covers the leading ubatches of one sequence; the ubatches after it run the full graph
        const bool in_window = prefill_window && prefill_window->covers(prefill_chunk);
        if (prefill_window) {
            prefill_window->decoder(prefill_chunk++);
        }
        const auto & ubatch = mctx->get_ubatch();

        // pre-record the largest prefill reservation ahead of a growing
        // multi-ubatch prefill, so later ubatches fit without reallocating
        if (!prefill_window) {
            // a background decode starts with the verify ubatches of the generating slots (alternate schedulers): they
            // do not cover the prompt ubatch that follows
            reserve_max_prefill(ubatch, mctx.get(), n_tokens_all, decode_bg ? n_tokens_prev_main : n_tokens_prev);
        }

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            if (prefill_window) {
                synchronize();
            }
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                const llama_pos remove_from = in_window ? -1 : pos_min[s];
                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, remove_from);

                memory->seq_rm(s, remove_from, -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        extract_layer_inputs(res, ubatch, n_tokens_prev);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (cparams.mtp_device_draft_mode == LLAMA_MTP_DEVICE_DRAFT_DISABLED &&
                    !embd_nextn_host && t_h_nextn && n_rows > 0 &&
                    !store_embeddings_nextn_backend(t_h_nextn, offset, n_rows)) {
                LLAMA_LOG_ERROR("%s: failed to preserve backend nextn embeddings\n", __func__);
                return -2;
            }

            if (cparams.mtp_device_draft_mode == LLAMA_MTP_DEVICE_DRAFT_DISABLED &&
                    embd_nextn_host && embd_nextn.data && t_h_nextn && n_rows > 0 &&
                    cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
            }
        }

        if (has_samplers && cparams.mtp_device_draft_mode == LLAMA_MTP_DEVICE_DRAFT_DISABLED) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            // with staggered verify each sequence has its own ubatch, so the lazy rows of a ubatch start at n_outputs_prev
            const bool lazy = cparams.sampler_lazy &&
                    (staggered_verify || (n_outputs_prev == 0 && n_outputs == n_outputs_all &&
                                         n_tokens_prev + ubatch.n_tokens == balloc->get_n_tokens())) &&
                    std::any_of(res->t_sampled_probs_by_token.begin(), res->t_sampled_probs_by_token.end(),
                            [](ggml_tensor * tensor) { return tensor != nullptr; });
            if (lazy) {
                for (size_t i0 = 0; i0 < res->t_sampled.size(); ++i0) {
                    const size_t i = n_outputs_prev + i0;
                    auto & row = sampling.lazy_rows[i];
                    while (i >= sampling.lazy_storage.size()) {
                        sampling.lazy_storage.emplace_back(new std::array<sampling_info::lazy_tensor, 4>);
                    }
                    bool allocation_failed = false;
                    auto store = [&](ggml_tensor * src, size_t index) -> ggml_tensor * {
                        if (!src) {
                            return nullptr;
                        }
                        auto & dst = (*sampling.lazy_storage[i])[index];
                        auto backend = ggml_backend_sched_get_tensor_backend(sched.get(), src);
                        auto buft = ggml_backend_get_default_buffer_type(backend);
                        const size_t size = ggml_backend_buft_get_alloc_size(buft, src);
                        if (!dst.buffer || ggml_backend_buffer_get_type(dst.buffer.get()) != buft ||
                                ggml_backend_buffer_get_size(dst.buffer.get()) < size) {
                            dst.buffer.reset(ggml_backend_buft_alloc_buffer(buft, size));
                            if (!dst.buffer) {
                                allocation_failed = true;
                                return nullptr;
                            }
                        }
                        dst.tensor = *src;
                        dst.tensor.buffer = dst.buffer.get();
                        dst.tensor.data = ggml_backend_buffer_get_base(dst.buffer.get());
                        dst.tensor.view_src = nullptr;
                        dst.tensor.view_offs = 0;
                        dst.tensor.extra = nullptr;
                        ggml_backend_tensor_copy_async(backend, backend, src, &dst.tensor);
                        return &dst.tensor;
                    };
                    row = { store(res->t_sampled_logits[i0], 0), store(res->t_sampled_probs[i0], 1),
                            store(res->t_candidates[i0], 2), store(res->t_sampled_probs_by_token[i0], 3) };
                    if (allocation_failed) {
                        LLAMA_LOG_ERROR("%s: failed to allocate lazy sampling output\n", __func__);
                        return -2;
                    }
                    sampling.logits_count[i]     = row.logits     ? ggml_nelements(row.logits)     : 0;
                    sampling.probs_count[i]      = row.probs      ? ggml_nelements(row.probs)      : 0;
                    sampling.candidates_count[i] = row.candidates ? ggml_nelements(row.candidates) : 0;
                }
            } else {
                copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
                copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
                copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
            }
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
        if (graph_slot_live == 0) {
            n_tokens_prev_main += ubatch.n_tokens;
        }
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    verify_streams_decode_end(true);

    return 0;
}

bool llama_context::store_embeddings_nextn_backend(ggml_tensor * src, size_t row_offset, size_t n_rows) {
    GGML_ASSERT(src && src->type == GGML_TYPE_F32 && src->ne[0] == model.hparams.n_embd_out());
    GGML_ASSERT(n_rows > 0 && n_rows <= (size_t) ggml_nrows(src));
    const size_t n_rows_max = cparams.embeddings_nextn_masked ? cparams.n_outputs_max : cparams.n_batch;
    GGML_ASSERT(row_offset + n_rows <= n_rows_max);

    ggml_backend_t backend_src = ggml_backend_sched_get_tensor_backend(sched.get(), src);
    GGML_ASSERT(backend_src);

    if (!embd_nextn_backend) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        embd_nextn_backend_ctx.reset(ggml_init(params));
        if (!embd_nextn_backend_ctx) {
            return false;
        }

        embd_nextn_backend = ggml_new_tensor_2d(embd_nextn_backend_ctx.get(), GGML_TYPE_F32,
                model.hparams.n_embd_out(), n_rows_max);
        embd_nextn_backend_buf.reset(ggml_backend_alloc_ctx_tensors(embd_nextn_backend_ctx.get(), backend_src));
        if (!embd_nextn_backend_buf) {
            embd_nextn_backend = nullptr;
            embd_nextn_backend_ctx.reset();
            return false;
        }
    }

    ggml_tensor src_view = *src;
    src_view.ne[1] = n_rows;
    src_view.ne[2] = src_view.ne[3] = 1;
    src_view.nb[2] = src_view.nb[1] * n_rows;
    src_view.nb[3] = src_view.nb[2];
    src_view.view_src = nullptr;
    src_view.view_offs = 0;

    ggml_tensor dst_view = *embd_nextn_backend;
    dst_view.ne[1] = n_rows;
    dst_view.ne[2] = dst_view.ne[3] = 1;
    dst_view.nb[2] = dst_view.nb[1] * n_rows;
    dst_view.nb[3] = dst_view.nb[2];
    dst_view.data = static_cast<char *>(embd_nextn_backend->data) + row_offset * embd_nextn_backend->nb[1];
    dst_view.view_src = nullptr;
    dst_view.view_offs = 0;

    ggml_backend_synchronize(backend_src);
    ggml_backend_tensor_copy(&src_view, &dst_view);

    embd_nextn_backend_rows = std::max(embd_nextn_backend_rows, row_offset + n_rows);
    embd_nextn_backend_view = *embd_nextn_backend;
    embd_nextn_backend_view.ne[1] = embd_nextn_backend_rows;
    embd_nextn_backend_view.ne[2] = embd_nextn_backend_view.ne[3] = 1;
    embd_nextn_backend_view.nb[2] = embd_nextn_backend_view.nb[1] * embd_nextn_backend_rows;
    embd_nextn_backend_view.nb[3] = embd_nextn_backend_view.nb[2];
    embd_nextn_backend_view.view_src = nullptr;
    embd_nextn_backend_view.view_offs = 0;

    return true;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());
    sampling.lazy_rows.assign(n_outputs_max, {});

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled && !cparams.embeddings_layer_inp_dev) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il] && !cparams.embeddings_layer_inp_dev) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res, const llama_ubatch & ubatch, size_t token_offset) {
    if (cparams.embeddings_layer_inp_dev) {
        return;
    }
    const size_t n_tokens = ubatch.n_tokens;
    // pinned hand-off buffers can take side-stream downloads; feature_handoff_take joins them
    const bool side_ok = handoff_enabled && handoff_slots[handoff_cur].pinned;
    // The rows go to their batch index: the readers (speculative drafts) index the features by batch row, and a split
    // can reorder the tokens (equal splits interleave sequences, the DSV4 cache puts a long prompt first). Without the
    // indices, the ubatches are in batch order.
    const std::vector<int32_t> * idx = ubatch.data && ubatch.data->idx.size() == n_tokens ? &ubatch.data->idx : nullptr;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        auto * get_side = side_ok ? feature_side_get(backend) : nullptr;
        auto get = [&](void * dst, size_t off, size_t size) {
            if (get_side) {
                get_side(backend, t, dst, off, size);
            } else {
                ggml_backend_tensor_get_async(backend, t, dst, off, size);
            }
        };
        if (!idx) {
            get(embd_layer_inp[il].data + dst_offset, 0, nbytes);
            continue;
        }
        const size_t row_bytes = row_floats*sizeof(float);
        for (size_t i0 = 0; i0 < n_tokens;) {
            size_t i1 = i0 + 1;
            while (i1 < n_tokens && (*idx)[i1] == (*idx)[i1 - 1] + 1) {
                i1++;
            }
            const size_t dst = (size_t) (*idx)[i0]*row_floats;
            GGML_ASSERT(dst + (i1 - i0)*row_floats <= embd_layer_inp[il].size);
            get(embd_layer_inp[il].data + dst, i0*row_bytes, (i1 - i0)*row_bytes);
            i0 = i1;
        }
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (embd_layer_inp.size() > 0) {
            for (int lid = 0; lid < (int) embd_layer_inp.size(); ++lid) {
                if (embd_layer_inp[lid].size > 0) {
                    for (uint64_t k = 0; k < n_embd; ++k) {
                        std::swap(embd_layer_inp[lid].data[i0*n_embd + k], embd_layer_inp[lid].data[i1*n_embd + k]);
                    }
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.lazy_rows[i0],        sampling.lazy_rows[i1]);
            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5NEXT) {
        // the n_tokens*40 budget below runs out by ubatch 3840: KDA costs 182 nodes + ~16/token
        // per layer, so 34 KDA layers alone need 6.2k + 31.9*n_tokens before DSA or the MoE
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_DEEPSEEK41 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }
    // The expert cache adds a CPU branch and its input per layer, and per tier the map gather, three expert multiplies, SwiGLU
    // and a weighted sum over the expert slots.
    if (expert_cache) {
        res += (64u + llama_expert_cache::max_tiers * (16u + 2u * model.hparams.n_expert_used())) * model.layers.size();
    }

    uint32_t n_sampling_nodes = 0;
    uint32_t n_sampling_nodes_max = 0;
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler) + (cparams.sampler_lazy ? 8 : 0);
        n_sampling_nodes += n_nodes;
        if (cparams.n_outputs_max_per_seq > 1) {
            n_sampling_nodes_max = std::max(n_sampling_nodes_max, n_nodes);
        }
    }

    const uint32_t n_sampling_outputs_max = std::min<uint64_t>(
            std::min(n_tokens, cparams.n_outputs_max),
            (uint64_t) cparams.n_seq_max * cparams.n_outputs_max_per_seq);

    res += n_sampling_nodes;
    if (n_sampling_outputs_max > 1) {
        res += (n_sampling_outputs_max - 1) * n_sampling_nodes_max;
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// the last allowed CPU (ggml-cpuset.h) of a "0-7,16-23" list, -1 if the list has none
static int last_allowed_cpu(const std::string & list) {
    int last = -1;
    for (size_t pos = 0; pos < list.size();) {
        const size_t end = std::min(list.find(',', pos), list.size());
        const std::string r = list.substr(pos, end - pos);
        const size_t dash = r.find('-');
        const int first = atoi(r.c_str());
        const int hi    = dash == std::string::npos ? first : atoi(r.c_str() + dash + 1);
        for (int c = std::max(first, 0); c <= hi && c < 4096; ++c) {
            if (ggml_cpuset_allowed(c)) {
                last = c;
            }
        }
        pos = end + 1;
    }
    return last;
}

// delphi: a CPU on the NUMA node of a GPU for its scheduler worker: LLAMA_MOE_CACHE_WORKER_CPU, else the last allowed CPU of
// the node (--numa tensors gives the CPU threads the first CPUs of each node), else the last allowed CPU, else -1 (not pinned).
// The CPU is always inside the allowed set (ggml-cpuset.h): a restricted CPU set can leave a node without CPUs.
static int share_worker_cpu(ggml_backend_dev_t dev) {
    if (const char * env = getenv("LLAMA_MOE_CACHE_WORKER_CPU")) {
        if (ggml_cpuset_allowed(atoi(env))) {
            return atoi(env);
        }
        LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_WORKER_CPU=%s is outside the allowed CPU set, ignored\n", __func__, env);
    }
    int cpu = -1;
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (props.device_id != nullptr) {
        std::string id = props.device_id;
        for (auto & c : id) {
            c = (char) tolower((unsigned char) c);
        }
        int node = -1;
        std::ifstream("/sys/bus/pci/devices/" + id + "/numa_node") >> node;
        if (node >= 0) {
            std::string list;
            std::ifstream("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist") >> list;
            cpu = last_allowed_cpu(list);
        }
    }
    if (cpu < 0) {
        std::vector<int> all(4096);
        all.resize(ggml_cpuset_list(all.data(), (int) all.size()));
        cpu = all.empty() ? -1 : all.back();
    }
    return cpu;
}

// delphi: the splits of an expert cache tier on another GPU are submitted by a scheduler worker, so that their host time
// overlaps the CPU expert part (LLAMA_MOE_CACHE_WORKER=0 turns this off)
void llama_context::sched_set_workers(ggml_backend_sched_t s) {
    const bool enabled = [] { const char * e = getenv("LLAMA_MOE_CACHE_WORKER"); return e == nullptr || atoi(e) != 0; }();
    if (s == nullptr) {
        return;
    }
    for (auto & b : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(b.get());
        if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        bool remote = false;
        for (int t = 0; enabled && expert_cache && t < expert_cache->n_tiers(); ++t) {
            remote |= expert_cache->tier_backend(t) == b.get() && expert_cache->serves_remote(b.get());
        }
        const bool prefill = expert_cache && expert_cache->prefill_overlap() && b.get() == expert_cache->backend();
        const bool worker = remote || (enabled && prefill);
        ggml_backend_sched_set_worker(s, sched_backend_of(s, b.get()), worker, worker ? share_worker_cpu(dev) : -1);
        ggml_backend_sched_set_worker_ready(s, sched_backend_of(s, b.get()), prefill ? llama_expert_cache::prefill_worker_ready : nullptr,
                prefill ? expert_cache.get() : nullptr);
    }
}

void llama_context::sched_set_workers_all() {
    sched_set_workers(sched.get());
    for (auto & slot : graph_slots) {
        if (slot.sched) {
            sched_set_workers(slot.sched.get());
        }
    }
}

void llama_context::expert_cache_auto(llama_context * draft) {
    // At the first decode every context of the process (draft, MTP) holds its memory, so what is free can go to the cache.
    // The margin is for buffers that grow later (graph reallocations, other processes). The decode-time refresh scratch is
    // allocated later on each tier GPU, so it is kept out of the tier too.
    // before the tiers take the free memory
    llama_dsv41_prefill_window::reserve(*this);
    if (!expert_cache || !expert_cache->auto_size) {
        return;
    }
    expert_cache->auto_size = false;
    synchronize();
    std::map<ggml_backend_dev_t, size_t> graph_budget;
    for (const auto & d : model.devices) {
        if (expert_cache->auto_device(d.dev)) {
            graph_budget[d.dev] = 0;
        }
    }
    const char * graph_env = getenv("LLAMA_MOE_CACHE_GRAPH_RESERVE");
    if (graph_env == nullptr || atoi(graph_env) != 0) {
        auto account_graphs = [&](llama_context & ctx) {
            ctx.synchronize();
            for (auto & b : ctx.backends) {
                auto * dev = ggml_backend_get_device(b.get());
                // A draft can use a GPU with no auto tier. Keep that device's executable policy unchanged.
                if (dev == nullptr || graph_budget.find(dev) == graph_budget.end()) {
                    continue;
                }
                auto * reg = ggml_backend_dev_backend_reg(dev);
                auto * reserve = reg ? (size_t (*)(ggml_backend_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_reserve") : nullptr;
                if (reserve == nullptr) {
                    continue;
                }
                size_t layers = 0;
                for (uint32_t il = 0; il < ctx.model.hparams.n_layer(); ++il) {
                    layers += ctx.model.dev_layer(il) == dev;
                }
                // Budget four device splits per layer and alternate scheduler, with at least 64 executables.
                // The backend bounds retained executables, so new request shapes cannot exceed this reserve.
                const size_t limit = std::max<size_t>(64, 4*layers*std::max<uint32_t>(1, ctx.graph_alt_slots));
                graph_budget[dev] += reserve(b.get(), limit);
            }
        };
        account_graphs(*this);
        if (draft && draft != this) {
            account_graphs(*draft);
        }
    }
    // staging slots of their own go back, so that the rings borrow from the tiers instead
    ggml_backend_sched_stage_release(sched.get());
    for (auto & slot : graph_slots) {
        if (slot.sched) {
            ggml_backend_sched_stage_release(slot.sched.get());
        }
    }
    // A staging slot holds one host expert tensor, and on a cache GPU the tier lends it. A GPU whose tier can not lend one
    // slot (a long context or a draft model left little memory) does not stage, else the first long prompt that the
    // scheduler sends there aborts.
    size_t stage_slot = 0;
    for (const auto & [tname, t] : model.tensors_by_name) {
        if (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer) && tname.find("_exps.") != std::string::npos) {
            stage_slot = std::max(stage_slot, ggml_nbytes(t));
        }
    }
    auto stage_check = [&](ggml_backend_dev_t dev, size_t tier_bytes) {
        if (stage_slot > 0 && tier_bytes < stage_slot) {
            LLAMA_LOG_WARN("%s: %s does not stage experts: its expert cache holds %.0f MiB, less than one %.0f MiB staging slot\n",
                    __func__, ggml_backend_dev_name(dev), tier_bytes / 1048576.0, stage_slot / 1048576.0);
            share_stage_allow(dev, false);
        }
    };
    // tiers on the GPUs of the cached layers only, unless LLAMA_MOE_CACHE_AUTO_REMOTE=1 (a tier on another GPU costs a copy to it
    // and back per layer)
    for (const auto & d : model.devices) {
        const std::string name = ggml_backend_dev_name(d.dev);
        if (!expert_cache->auto_device(d.dev)) {
            continue;
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(d.dev, &free, &total);
        const size_t scratch = expert_cache->async_scratch_bytes();
        size_t budget = expert_device_policy.auto_budget(name, free);
        budget = budget > scratch ? budget - scratch : 0;
        const size_t graphs = graph_budget[d.dev];
        budget = budget > graphs ? budget - graphs : 0;
        if (!budget) {
            LLAMA_LOG_INFO("%s: expert cache auto: %s has %.0f MiB free, no tier\n", __func__, ggml_backend_dev_name(d.dev),
                    free / 1048576.0);
            stage_check(d.dev, 0);
            continue;
        }
        const int id = share_add_tier(d.dev, budget);
        LLAMA_LOG_INFO("%s: expert cache auto: %s %.0f MiB free, %.0f MiB reserve, %.0f MiB refresh scratch, %.0f MiB graph reserve, tier %d of up to %.0f MiB\n",
                __func__, name.c_str(), free / 1048576.0, expert_device_policy.margin(name) / 1048576.0, scratch / 1048576.0, graphs / 1048576.0, id,
                budget / 1048576.0);
        stage_check(d.dev, id >= 0 ? expert_cache->tier_bytes(id) : 0);
    }
}

void llama_context::sched_set_stage_lender() {
    ggml_backend_sched_set_input_async(sched.get(), verify_input_async && (graph_slot_live != 0 || model.arch == LLM_ARCH_DFLASH));
    const char * prompt_input_env = getenv("LLAMA_VERIFY_STREAMS_PROMPT_INPUT_ASYNC");
    ggml_backend_sched_set_input_staging(sched.get(), verify_streams && verify_input_async && graph_slot_live == 0 &&
            (prompt_input_env == nullptr || atoi(prompt_input_env) != 0));
    sched_set_workers(sched.get());
    if (expert_cache) {
        ggml_backend_sched_set_stage_lender(sched.get(), llama_expert_cache::lend, expert_cache.get());
    }
    for (auto & b : backends) {
        auto * dev = ggml_backend_get_device(b.get());
        if (dev && (!expert_device_policy.stage_allowed(ggml_backend_dev_name(dev)) ||
                std::find(stage_blocked.begin(), stage_blocked.end(), dev) != stage_blocked.end())) {
            ggml_backend_sched_stage_allow(sched.get(), sched_backend_of(sched.get(), b.get()), false);
        }
    }
    if (stage_min_tokens >= 0) {
        ggml_backend_sched_stage_set_min_tokens(sched.get(), stage_min_tokens);
    }
}

void llama_context::graph_slot_activate(int id) {
    if (id == graph_slot_live) {
        return;
    }
    const auto exchange = [this](graph_slot & slot) {
        std::swap(sched,               slot.sched);
        std::swap(gf_res_prev,         slot.gf_res_prev);
        std::swap(gf_res_prev_active,  slot.gf_res_prev_active);
        std::swap(gf_res_prev_threads, slot.gf_res_prev_threads);
    };
    exchange(graph_slots[graph_slot_live]); // store the live slot
    exchange(graph_slots[id]);
    graph_slot_live = id;
    if (!sched) {
        sched.reset(ggml_backend_sched_new(sched_backend_ptrs(id), backend_buft.data(), backend_ptrs.size(),
                gf_res_reserve->get_max_nodes(), cparams.pipeline_parallel, cparams.op_offload));
        sched_set_stage_lender();
        gf_res_prev_active = nullptr;
    }
}

ggml_backend_sched_t llama_context::graph_sched_main() const {
    return graph_slot_live == 0 ? sched.get() : graph_slots[0].sched.get();
}

void llama_context::graph_alt_reset() {
    graph_slot_activate(0);
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    for (auto & slot : graph_slots) {
        for (auto & res : slot.gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        slot.gf_res_prev_active = nullptr;
    }
    gf_res_prev_active = nullptr;
}

//
// delphi: VRAM sharing between servers
//

// Forget every graph and scheduler; the next decode reserves again. Graphs hold the addresses of compute buffers, KV tensors and
// expert cache banks.
void llama_context::share_reset_scheds() {
    synchronize();
    graph_alt_reset();
    for (auto & slot : graph_slots) {
        slot.sched.reset();
        for (auto & res : slot.gf_res_prev) {
            res.reset();
        }
    }
    max_prefill_sched = nullptr;
    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_prev_active = nullptr;
    // a scheduler without buffers: other calls may still use it before the next decode reserves
    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
            gf_res_reserve->get_max_nodes(), cparams.pipeline_parallel, cparams.op_offload));
    sched_set_stage_lender();
    sched_need_reserve = true;
    share_rereserve = true;
    if (expert_cache) {
        // the old schedulers dropped their staging rings, lent slots included
        expert_cache->reclaimed();
    }
}

int llama_context::share_add_tier(ggml_backend_dev_t dev, size_t bytes) {
    if (!expert_cache) {
        return -1;
    }
    ggml_backend_t backend = nullptr;
    for (auto & b : backends) {
        if (ggml_backend_get_device(b.get()) == dev) {
            backend = b.get();
        }
    }
    if (backend == nullptr) {
        return -1;
    }
    synchronize();
    const int id = expert_cache->add_tier(backend, bytes);
    if (id >= 0) {
        share_gen++;
        sched_set_workers_all();
    }
    return id;
}

size_t llama_context::share_drop_tier(int id) {
    if (!expert_cache) {
        return 0;
    }
    synchronize();
    // staging rings may hold slots inside the tier, and graphs its banks
    ggml_backend_sched_stage_reclaim(sched.get());
    for (auto & slot : graph_slots) {
        if (slot.sched) {
            ggml_backend_sched_stage_reclaim(slot.sched.get());
        }
    }
    expert_cache->reclaimed();
    graph_alt_reset();
    const size_t bytes = expert_cache->drop_tier(id);
    share_gen++;
    sched_set_workers_all();
    return bytes;
}

size_t llama_context::share_release_compute() {
    const int64_t t0 = ggml_time_us();
    synchronize();
    size_t freed = 0;
    auto count = [&](ggml_backend_sched_t s) {
        if (s == nullptr) {
            return;
        }
        freed += ggml_backend_sched_stage_release(s);
        for (auto & b : backends) {
            if (ggml_backend_dev_type(ggml_backend_get_device(b.get())) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                freed += ggml_backend_sched_get_buffer_size(s, sched_backend_of(s, b.get()));
            }
        }
    };
    count(sched.get());
    for (auto & slot : graph_slots) {
        count(slot.sched.get());
    }
    share_reset_scheds();
    if (expert_cache) {
        freed += expert_cache->release_scratch();
    }
    // the backends' memory pools, captured graphs and scratch buffers
    for (auto & b : backends) {
        auto * dev = ggml_backend_get_device(b.get());
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * trim = reg ? (void (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_trim") : nullptr;
        if (trim != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            trim(b.get());
        }
    }
    LLAMA_LOG_WARN("%s: released %.1f MiB of compute buffers and staging slots in %.1f ms\n", __func__, freed / 1048576.0,
            (ggml_time_us() - t0) / 1000.0);
    return freed;
}

size_t llama_context::share_release_memory() {
    if (!memory || memory_released || model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return 0;
    }
    synchronize();
    size_t bytes = 0;
    for (const auto & [buft, size] : memory->memory_breakdown()) {
        if (!ggml_backend_buft_is_host(buft)) {
            bytes += size;
        }
    }
    share_reset_scheds();
    memory.reset();
    memory_released = true;
    LLAMA_LOG_WARN("%s: released %.1f MiB of device memory module (KV) buffers\n", __func__, bytes / 1048576.0);
    return bytes;
}

bool llama_context::share_restore_memory() {
    if (!memory_released) {
        return true;
    }
    const int64_t t0 = ggml_time_us();
    memory.reset(model.create_memory(mem_params, cparams));
    staggered_verify_init();
    memory_released = false;
    sched_need_reserve = true;
    LLAMA_LOG_WARN("%s: memory module created again in %.1f ms\n", __func__, (ggml_time_us() - t0) / 1000.0);
    return memory != nullptr;
}

void llama_context::share_stage_allow(ggml_backend_dev_t dev, bool allow) {
    const auto it = std::find(stage_blocked.begin(), stage_blocked.end(), dev);
    if (allow == (it == stage_blocked.end())) {
        return;
    }
    if (allow) {
        stage_blocked.erase(it);
    } else {
        stage_blocked.push_back(dev);
    }
    synchronize();
    for (auto & b : backends) {
        if (ggml_backend_get_device(b.get()) != dev) {
            continue;
        }
        const bool permitted = allow && expert_device_policy.stage_allowed(ggml_backend_dev_name(dev));
        ggml_backend_sched_stage_allow(sched.get(), sched_backend_of(sched.get(), b.get()), permitted);
        for (auto & slot : graph_slots) {
            if (slot.sched) {
                ggml_backend_sched_stage_allow(slot.sched.get(), sched_backend_of(slot.sched.get(), b.get()), permitted);
            }
        }
    }
    share_gen++;
}

// host-to-device rate of a GPU in bytes/us, from a 64 MiB copy out of pinned memory (the staging rings copy through pinned
// buffers); 0 if the buffers cannot be allocated
static double llama_stage_link_rate(ggml_backend_dev_t dev) {
    const size_t size = 64u << 20;
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_buffer_t host = host_buft ? ggml_backend_buft_alloc_buffer(host_buft, size) : nullptr;
    ggml_backend_buffer_t gpu  = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), size);
    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    double rate = 0;
    if (host != nullptr && gpu != nullptr && ctx != nullptr) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, size);
        if (ggml_backend_tensor_alloc(gpu, t, ggml_backend_buffer_get_base(gpu)) == GGML_STATUS_SUCCESS) {
            void * src = ggml_backend_buffer_get_base(host);
            memset(src, 0, size);
            for (int i = 0; i < 3; i++) {
                const int64_t t0 = ggml_time_us();
                ggml_backend_tensor_set(t, src, 0, size);
                rate = std::max(rate, size / (double) std::max<int64_t>(1, ggml_time_us() - t0));
            }
        }
    }
    ggml_free(ctx);
    ggml_backend_buffer_free(gpu);
    ggml_backend_buffer_free(host);
    return rate;
}

// The staging threshold: prompts of n/2 and n random tokens with the routed experts on the CPU, then streamed to the GPUs.
// The time of the CPU splits is the expert time e(t) of t tokens; the rest of a CPU run, r(t) = run(t) - e(t), is the same when
// streaming, so streaming the experts costs s(t) = run_streamed(t) - r(t). e(t) is taken as affine in t, and the threshold is
// where it reaches s(t). Every expert crosses the link whatever the batch, so s(t) is flat at the link time for short batches
// and rises once the GPU's expert compute takes longer (Qwen 3.8 on an R9700, -ub 4096: streamed 2.5 s for 1024 tokens, 9.5 s
// for 4096, so with 56 threads the CPU stays ahead): s(t) is affine above n/2 and s(n/2) below it. When
// the link rate says that streaming takes 3 times e(n_ubatch) or more, the streamed runs are skipped and the experts stay on
// the CPU (CMP 170HX links: about 1 GB/s). The calibration adds a few seconds to the warmup at -ub 512, 15 s at 2048.
int32_t llama_context::share_stage_calibrate() {
    if (!sched || ggml_backend_sched_stage_get_min_tokens(sched.get()) < 0) {
        return -1;
    }
    if (stage_calibrated) {
        return ggml_backend_sched_stage_get_min_tokens(sched.get());
    }
    stage_calibrated = true;
    std::vector<ggml_backend_sched_t> scheds = { sched.get() };
    for (auto & slot : graph_slots) {
        if (slot.sched) {
            scheds.push_back(slot.sched.get());
        }
    }
    const int prev = ggml_backend_sched_stage_get_min_tokens(sched.get());
    double expert_bytes = 0;
    for (const auto & l : model.layers) {
        for (const ggml_tensor * w : { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps }) {
            if (w != nullptr && w->buffer != nullptr && ggml_backend_buffer_is_host(w->buffer)) {
                expert_bytes += ggml_nbytes(w);
            }
        }
    }
    // up to 2048: the streamed cost is flat while the link bounds it and rises once the GPU's expert compute does (Qwen 3.8,
    // R9700, 56 threads: from 1024 tokens on), and points below that rise miss it
    const int n = (int) std::min<uint32_t>({ cparams.n_ubatch, n_ctx_seq(), 2048 });
    if (getenv("GGML_SCHED_STAGE_MIN_TOKENS") != nullptr || expert_bytes == 0 || n < 64) {
        return expert_bytes == 0 ? -1 : prev;
    }
    auto set_min_tokens = [&](int t) {
        synchronize();
        for (auto * s : scheds) {
            ggml_backend_sched_stage_set_min_tokens(s, t);
        }
        share_gen++; // split the graphs again
    };
    std::vector<llama_token> tokens(n);
    const uint32_t n_vocab = model.vocab.n_tokens();
    for (int i = 0; i < n; i++) {
        tokens[i] = (llama_token) ((i * 2654435761u + 977u) % n_vocab);
    }
    struct run_t { double us, cpu_us; uint64_t jobs; bool ok; };
    auto run = [&](int n_tokens, int min_tokens) {
        set_min_tokens(min_tokens);
        auto stats = [&](int64_t & cpu_us, uint64_t & jobs) {
            cpu_us = 0;
            jobs = 0;
            for (auto * s : scheds) {
                int64_t c; uint64_t j;
                ggml_backend_sched_stage_stats(s, &c, &j);
                cpu_us += c;
                jobs += j;
            }
        };
        int64_t c0, c1;
        uint64_t j0, j1;
        stats(c0, j0);
        const int64_t t0 = ggml_time_us();
        const bool ok = decode(llama_batch_get_one(tokens.data(), n_tokens)) == 0;
        synchronize();
        const int64_t t1 = ggml_time_us();
        stats(c1, j1);
        llama_memory_clear(get_memory(), true);
        return run_t { double(t1 - t0), double(c1 - c0), j1 - j0, ok };
    };
    // the first prompt after the warmup pays one-time costs (Qwen 3.8: 2 s), so it is not measured
    run(std::min(256, n/2), std::numeric_limits<int>::max());
    const run_t half = run(n/2, std::numeric_limits<int>::max());
    const run_t cpu  = run(n, std::numeric_limits<int>::max());
    const int n_max = (int) cparams.n_ubatch + 1; // never streams
    double slope = (cpu.cpu_us - half.cpu_us) / (n - n/2);
    if (slope <= 0) {
        slope = cpu.cpu_us / n;
    }
    const double e0 = cpu.cpu_us - slope*n;
    double rate = 0;
    int n_gpu = 0;
    for (auto & b : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(b.get());
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                std::find(stage_blocked.begin(), stage_blocked.end(), dev) == stage_blocked.end()) {
            rate = std::max(rate, llama_stage_link_rate(dev));
            n_gpu++;
        }
    }
    // optimistic: every GPU streams at the best rate
    const double s_link = rate > 0 ? expert_bytes / (rate * n_gpu) : 0;
    int threshold = n_max;
    const char * how = "link";
    run_t gpu = { 0, 0, 0, true };
    run_t gpu_half = gpu;
    if (!half.ok || !cpu.ok || cpu.cpu_us <= 0) {
        threshold = prev;
        how = "failed";
    } else if (s_link < 3 * (e0 + slope*cparams.n_ubatch)) {
        // n first: the first streamed run also allocates the ring, which then counts against streaming
        gpu      = run(n, 1);
        gpu_half = run(n/2, 1);
        if (!gpu.ok || !gpu_half.ok || gpu.jobs == 0) {
            threshold = prev;
            how = gpu.ok && gpu_half.ok ? "nothing streams" : "failed";
        } else {
            const double s      = gpu.us - (cpu.us - cpu.cpu_us);
            const double s_half = gpu_half.us - (half.us - half.cpu_us);
            const double s_slope = (s - s_half) / (n - n/2);
            const double s0 = s - s_slope*n;
            if (e0 + slope*(n/2) >= s_half) {
                // the CPU falls behind on the flat part
                threshold = std::max(1, (int) std::ceil((s_half - e0) / slope));
            } else if (slope > s_slope) {
                // e(t) - s(t) = (e0 - s0) + (slope - s_slope) t, negative at n/2: the CPU falls behind from its root on
                threshold = std::min(n_max, (int) std::ceil((s0 - e0) / (slope - s_slope)));
            }
            how = "measured";
        }
    }
    set_min_tokens(threshold);
    stage_min_tokens = threshold;
    LLAMA_LOG_INFO("%s: staging threshold %d tokens (%s, %d = never): %d and %d tokens: experts on the CPU %.0f and %.0f ms "
            "(runs %.0f and %.0f ms), streamed runs %.0f and %.0f ms; link %.1f GB/s for %.1f GiB of experts\n", __func__,
            threshold, how, n_max, n/2, n, half.cpu_us/1e3, cpu.cpu_us/1e3, half.us/1e3, cpu.us/1e3, gpu_half.us/1e3, gpu.us/1e3,
            rate/1e3, expert_bytes/(1u << 30));
    return threshold;
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes, bool main_only) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    // reserve on the main scheduler
    if (main_only) {
        graph_slot_activate(0);
    } else {
        graph_alt_reset();
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    llm_graph_params params = {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.mtp_device_draft_cache =*/ &mtp_device_draft_cache,
        /*.expert_cache =*/ expert_cache.get(),
        /*.engram_reader =*/ engram_reader.get(),
        /*.layer_inp_dev =*/ layer_inp_dev,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
    params.share_gen = share_gen;
    params.samplers_gen = samplers_gen;
    params.layer_inp_dev_row_base = &layer_inp_row_base;
    params.dsv41_prefill = dsv41_prefill_graph;
    if (dsv41_prefill_graph && dsv41_prefill_graph->last <= (int) cparams.n_layer_prefill) {
        std::fill(params.cparams.embeddings_layer_inp.begin(), params.cparams.embeddings_layer_inp.end(), false);
        params.cparams.embeddings_layer_inp_dev = false;
        params.layer_inp_dev = nullptr;
    }
    return params;
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    set_backend_n_threads(n_threads);

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

void llama_context::set_backend_n_threads(int n_threads) {
    if (backend_n_threads == n_threads) {
        return;
    }

    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }
    backend_n_threads = n_threads;

    // A graph assigned with a different CPU thread count has stale placement
    // costs: it is only reused under the thread count it was built with
    // (gf_res_prev_threads).
}

// copy the split device's slices of the head weights once, so that it does not take them from the layer's device every ubatch
void llama_context::init_attn_split_weights() {
    const auto & hparams = model.hparams;
    const int64_t n_layer     = hparams.n_layer();
    const int64_t n_groups    = hparams.dsv4_o_group_count;
    const int64_t ng          = cparams.attn_split_groups;
    const int64_t n_head_grp  = hparams.n_head()/n_groups;
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t g0          = n_groups - ng;
    const int64_t h0          = g0*n_head_grp;
    const int64_t nh          = ng*n_head_grp;

    ggml_init_params ip = { (size_t) (3*n_layer + 1)*ggml_tensor_overhead(), nullptr, true };
    attn_split_ctx.reset(ggml_init(ip));
    ggml_context * ctx = attn_split_ctx.get();

    attn_split_w.wq_b.assign(n_layer, nullptr);
    attn_split_w.wo_a.assign(n_layer, nullptr);
    attn_split_w.sinks.assign(n_layer, nullptr);
    for (int64_t il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        if (layer.wq_b == nullptr || layer.wo_a == nullptr || layer.attn_sinks == nullptr || layer.wo_a->ne[2] != n_groups ||
                layer.wq_b->data == nullptr || layer.wo_a->data == nullptr || layer.attn_sinks->data == nullptr ||
                layer.attn_sinks->type != GGML_TYPE_F32) {
            continue;
        }
        attn_split_w.wq_b[il]  = ggml_new_tensor_2d(ctx, layer.wq_b->type, layer.wq_b->ne[0], nh*n_embd_head);
        attn_split_w.wo_a[il]  = ggml_new_tensor_3d(ctx, layer.wo_a->type, layer.wo_a->ne[0], layer.wo_a->ne[1], ng);
        attn_split_w.sinks[il] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nh);
        ggml_format_name(attn_split_w.wq_b[il],  "blk.%d.attn_q_b.split",      (int) il);
        ggml_format_name(attn_split_w.wo_a[il],  "blk.%d.attn_output_a.split", (int) il);
        ggml_format_name(attn_split_w.sinks[il], "blk.%d.attn_sinks.split",    (int) il);
    }
    if (ggml_get_first_tensor(ctx) == nullptr) {
        return; // no weights loaded (a context that only measures memory)
    }
    attn_split_buf.reset(ggml_backend_alloc_ctx_tensors(ctx, backend_moe_split));
    if (!attn_split_buf) {
        throw std::runtime_error("LLAMA_ATTN_SPLIT: failed to allocate the head weights on the split device");
    }
    ggml_backend_buffer_set_usage(attn_split_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<uint8_t> tmp;
    auto copy = [&](ggml_tensor * dst, const ggml_tensor * src, size_t offset) {
        tmp.resize(ggml_nbytes(dst));
        ggml_backend_tensor_get(src, tmp.data(), offset, tmp.size());
        ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size());
    };
    for (int64_t il = 0; il < n_layer; ++il) {
        if (attn_split_w.wq_b[il] == nullptr) {
            continue;
        }
        const auto & layer = model.layers[il];
        copy(attn_split_w.wq_b[il],  layer.wq_b,       h0*n_embd_head*layer.wq_b->nb[1]);
        copy(attn_split_w.wo_a[il],  layer.wo_a,       g0*layer.wo_a->nb[2]);
        copy(attn_split_w.sinks[il], layer.attn_sinks, h0*sizeof(float));
    }
    cparams.attn_split_w = &attn_split_w;
    LLAMA_LOG_INFO("%s: %.1f MiB of head weights on %s\n", __func__,
            ggml_backend_buffer_get_size(attn_split_buf.get())/1048576.0, ggml_backend_name(backend_moe_split));
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        static const bool cpu_experts = [] {
            const char * p = getenv("LLAMA_MOE_CPU_PREFILL");
            return p && atoi(p) != 0;
        }();
        const bool cache_cpu = strncmp(name, "ffn_cache_cpu_", 14) == 0;
        const bool cache_gpu = strncmp(name, "ffn_cache_gpu_", 14) == 0;
        const bool routed = strcmp(name, "ffn_moe_up") == 0 || strcmp(name, "ffn_moe_gate") == 0 ||
                strcmp(name, "ffn_moe_down") == 0 || strcmp(name, "ffn_moe_swiglu") == 0 ||
                strcmp(name, "ffn_moe_weighted") == 0 || strcmp(name, "ffn_moe_out") == 0;
        // Model builders can rename the returned MoE tensor. Keep an explicitly placed cache merge on its GPU.
        const bool cache_merge = expert_cache && strcmp(name, "ffn_moe_out") == 0 &&
                ggml_backend_sched_get_tensor_backend(sched.get(), cur) == expert_cache->backend(il);
        if (cache_cpu || (cpu_experts && ubatch.n_tokens > 16 && routed && !cache_merge)) {
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
        } else if (cache_gpu && expert_cache) {
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, expert_cache->backend(il));
        } else if (expert_cache && strncmp(name, "ffn_cache_x", 11) == 0) {
            // an expert cache tier on another GPU
            const int tier = atoi(name + 11);
            if (tier >= 0 && tier < expert_cache->n_tiers()) {
                ggml_backend_sched_set_tensor_backend(sched.get(), cur, expert_cache->tier_backend(tier));
            }
        }

        const char * split_tag = strncmp(name, "ffn_moe_x", 9) == 0 ? name + 9 : strncmp(name, "attn_x", 6) == 0 ? name + 6 : nullptr;
        if (backend_moe_split != nullptr && split_tag != nullptr && il >= 0) {
            // expert and attention split: x?s_* on the split device, x?a_* on the layer's device (views of host weights are
            // not weights to the scheduler, so the xa_* nodes would otherwise follow the xs_* nodes)
            ggml_backend_t target = split_tag[0] == 's' ? backend_moe_split : nullptr;
            for (const auto & backend : backends) {
                if (target == nullptr && ggml_backend_get_device(backend.get()) == model.dev_layer(il)) {
                    target = backend.get();
                }
            }
            if (target != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched.get(), cur, target);
            }
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, sched_backend_of(sched.get(), backend.get()));
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

// LLAMA_STATE_SEQ_FLAGS_ASYNC: tensor data on a device is copied into a pinned staging buffer on the stream of that
// device, behind the work already submitted, and reaches dst at the next synchronize() of the context. Host data, and
// device data without a staging buffer or a known stream, is read after a synchronization, as llama_io_write_host does.
class llama_io_write_host_async : public llama_io_write_i {
public:
    llama_io_write_host_async(uint8_t * p, size_t len, llama_context & lctx) : ptr(p), buf_size(len), lctx(lctx) {}

    ~llama_io_write_host_async() {
        size_t n_dev = 0;
        for (const auto & winfo : winfos) {
            n_dev += winfo.backend ? winfo.size : 0;
        }

        uint8_t * staging = n_dev > 0 ? lctx.state_async_staging(n_dev) : nullptr;

        bool synced = false;
        size_t off = 0;
        for (const auto & winfo : winfos) {
            if (winfo.backend && staging) {
                ggml_backend_tensor_get_async(winfo.backend, winfo.tensor, staging + off, winfo.offset, winfo.size);
                lctx.state_async_pending.push_back({ staging + off, winfo.ptr, winfo.size });
                off += winfo.size;
                continue;
            }
            if (!synced && !ggml_backend_buffer_is_host(winfo.tensor->buffer)) {
                lctx.synchronize();
                synced = true;
            }
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        ggml_backend_t backend = nullptr;
        if (buf && !ggml_backend_buffer_is_host(buf)) {
            backend = lctx.state_async_backend(ggml_backend_buffer_get_type(buf));
        }
        winfos.push_back({tensor, backend, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        ggml_backend_t backend;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_context & lctx;
};

uint8_t * llama_context::state_async_staging(size_t size) {
    for (size_t i = 0; i < state_async_bufs.size(); ++i) {
        if (!state_async_busy[i] && ggml_backend_buffer_get_size(state_async_bufs[i].get()) >= size) {
            state_async_busy[i] = true;
            return (uint8_t *) ggml_backend_buffer_get_base(state_async_bufs[i].get());
        }
    }

    ggml_backend_buffer_type_t buft = nullptr;
    for (auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            buft = ggml_backend_dev_host_buffer_type(dev);
            if (buft) {
                break;
            }
        }
    }
    if (!buft) {
        return nullptr;
    }

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buf) {
        return nullptr;
    }
    LLAMA_LOG_INFO("%s: allocated %.2f MiB of '%s' staging for asynchronous state copies\n", __func__,
            size/1024.0/1024.0, ggml_backend_buft_name(buft));
    state_async_bufs.emplace_back(buf);
    state_async_busy.push_back(true);
    return (uint8_t *) ggml_backend_buffer_get_base(buf);
}

ggml_backend_t llama_context::state_async_backend(ggml_backend_buffer_type_t buft) const {
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU && ggml_backend_dev_buffer_type(dev) == buft) {
            return backend.get();
        }
    }
    return nullptr;
}

void llama_context::state_async_finish() {
    for (const auto & cpy : state_async_pending) {
        memcpy(cpy.dst, cpy.src, cpy.size);
    }
    state_async_pending.clear();
    std::fill(state_async_busy.begin(), state_async_busy.end(), false);
}

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else if (flags & LLAMA_STATE_SEQ_FLAGS_ASYNC) {
        io = std::make_unique<llama_io_write_host_async>(dst, size, *this);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

//
// delphi: VRAM sharing (llama-ext.h)
//

static ggml_backend_dev_t llama_share_find_dev(const char * name) {
    for (size_t i = 0; name != nullptr && i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(dev), name) == 0) {
            return dev;
        }
    }
    return nullptr;
}

void llama_expert_cache_auto(llama_context * ctx) {
    ctx->expert_cache_auto();
}

void llama_expert_cache_auto(llama_context * ctx, llama_context * draft) {
    ctx->expert_cache_auto(draft);
}

int32_t llama_share_add_tier(llama_context * ctx, const char * dev, size_t bytes) {
    ggml_backend_dev_t d = llama_share_find_dev(dev);
    return d ? ctx->share_add_tier(d, bytes) : -1;
}

size_t llama_share_drop_tier(llama_context * ctx, int32_t id) {
    return ctx->share_drop_tier(id);
}

size_t llama_share_release_compute(llama_context * ctx) {
    return ctx->share_release_compute();
}

size_t llama_share_release_memory(llama_context * ctx) {
    return ctx->share_release_memory();
}

bool llama_share_restore_memory(llama_context * ctx) {
    return ctx->share_restore_memory();
}

bool llama_share_memory_released(const llama_context * ctx) {
    return ctx->share_memory_released();
}

void llama_share_stage_allow(llama_context * ctx, const char * dev, bool allow) {
    if (ggml_backend_dev_t d = llama_share_find_dev(dev)) {
        ctx->share_stage_allow(d, allow);
    }
}

int32_t llama_share_stage_calibrate(llama_context * ctx) {
    return ctx->share_stage_calibrate();
}

bool llama_share_has_expert_cache(const llama_context * ctx) {
    return ctx->share_expert_cache() != nullptr;
}

bool llama_share_has_cache_on_device(const llama_context * ctx, const char * dev) {
    const auto * cache = ctx->share_expert_cache();
    const auto d = llama_share_find_dev(dev);
    if (!cache || !d) {
        return false;
    }
    if (cache->auto_size && cache->auto_device(d)) {
        for (const auto & device : ctx->get_model().devices) {
            if (device.dev == d) {
                return true;
            }
        }
    }
    for (const auto & tier : cache->tier_list()) {
        if (ggml_backend_get_device(tier.second) == d) {
            return true;
        }
    }
    return false;
}

const char * llama_share_home_device(const llama_context * ctx) {
    const auto * cache = ctx->share_expert_cache();
    return cache ? ggml_backend_dev_name(ggml_backend_get_device(cache->backend())) : nullptr;
}

llama_share_usage llama_share_device_usage(const llama_context * ctx, const char * dev) {
    llama_share_usage res = {};
    ggml_backend_dev_t d = llama_share_find_dev(dev);
    if (d == nullptr) {
        return res;
    }
    for (const auto & [buft, mb] : ctx->memory_breakdown()) {
        if (buft != nullptr && ggml_backend_buft_get_device(buft) == d && !ggml_backend_buft_is_host(buft)) {
            res.model   += mb.model;
            res.context += mb.context;
            res.compute += mb.compute;
        }
    }
    if (const auto * cache = ctx->share_expert_cache()) {
        for (const auto & [id, backend] : cache->tier_list()) {
            if (ggml_backend_get_device(backend) == d) {
                res.cache += cache->tier_bytes(id);
            }
        }
    }
    return res;
}

size_t llama_share_stage_bytes(const llama_context * ctx) {
    const char * stage = getenv("GGML_SCHED_STAGE");
    if (stage != nullptr && atoi(stage) == 0) {
        return 0;
    }
    const char * slots_env = getenv("GGML_SCHED_STAGE_SLOTS");
    const size_t n_slots = slots_env ? std::max(1, atoi(slots_env)) : 2;
    size_t largest = 0;
    for (const auto & l : ctx->get_model().layers) {
        for (const ggml_tensor * w : { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps }) {
            if (w != nullptr && w->buffer != nullptr && ggml_backend_buffer_is_host(w->buffer)) {
                largest = std::max(largest, ggml_nbytes(w));
            }
        }
    }
    // a ring slot may hold the padded tensor
    return largest ? n_slots * (largest + (4u << 20)) : 0;
}

int32_t llama_share_tiers(const llama_context * ctx, int32_t * ids, size_t * bytes, const char ** devs, int32_t n_max) {
    const auto * cache = ctx->share_expert_cache();
    if (!cache) {
        return 0;
    }
    int32_t n = 0;
    for (const auto & [id, backend] : cache->tier_list()) {
        if (n < n_max) {
            ids[n] = id;
            bytes[n] = cache->tier_bytes(id);
            devs[n] = ggml_backend_dev_name(ggml_backend_get_device(backend));
        }
        n++;
    }
    return n;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (mtp_device_draft_buf) {
        auto * buft = ggml_backend_buffer_get_type(mtp_device_draft_buf.get());
        ret[buft].context += ggml_backend_buffer_get_size(mtp_device_draft_buf.get());
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), sched_backend_of(sched.get(), backend));
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), sched_backend_of(sched.get(), backend));
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), sched_backend_of(sched.get(), backend));
            for (const auto & slot : graph_slots) {
                if (slot.sched) {
                    ret[buft].compute += ggml_backend_sched_get_buffer_size(slot.sched.get(), sched_backend_of(slot.sched.get(), backend));
                }
            }
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;

        // the graph changes without flash attention, need to reserve again
        sched_need_reserve = true;
        sched_reserve();
    }

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), model.hparams.n_embd_out(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            graph_alt_reset();

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4 || model->arch == LLM_ARCH_DEEPSEEK41) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize_outputs();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize_outputs();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_nextn_host(llama_context * ctx, bool value) {
    ctx->set_embeddings_nextn_host(value);
}

void llama_set_embeddings_layer_inp_dev(llama_context * ctx, bool value) {
    ctx->set_embeddings_layer_inp_dev(value);
}

ggml_tensor * llama_get_embeddings_layer_inp_tensor(llama_context * ctx) {
    return ctx->get_layer_inp_dev();
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

void llama_set_mtp_index_reuse(llama_context * ctx, bool reuse) {
    ctx->set_mtp_index_reuse(reuse);
}

bool llama_mtp_device_draft_begin(llama_context * ctx, uint32_t n_steps) {
    return ctx->mtp_device_draft_begin(n_steps);
}

void llama_mtp_device_draft_advance(llama_context * ctx) {
    ctx->mtp_device_draft_advance();
}

bool llama_mtp_device_draft_finish(
        llama_context * ctx, llama_token * tokens, float * probs, size_t n_results) {
    return ctx->mtp_device_draft_finish(tokens, probs, n_results);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_nextn_ith(i);
}

ggml_tensor * llama_get_embeddings_nextn_tensor(llama_context * ctx) {
    return ctx->get_embeddings_nextn_tensor();
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize_outputs();

    return ctx->get_embeddings_layer_inp(lid);
}

void llama_set_feature_handoff(llama_context * ctx, bool enable) {
    ctx->set_feature_handoff(enable);
}

int32_t llama_feature_handoff_take(llama_context * ctx) {
    return ctx->feature_handoff_take();
}

const float * llama_feature_handoff_nextn(llama_context * ctx, int32_t ticket) {
    return ctx->feature_handoff_nextn(ticket);
}

const float * llama_feature_handoff_layer_inp(llama_context * ctx, int32_t ticket, uint32_t lid) {
    return ctx->feature_handoff_layer_inp(ticket, lid);
}

uint64_t llama_graph_n_built(const llama_context * ctx) {
    return ctx->n_graph_built;
}

void llama_graph_alt_slots_reserve(llama_context * ctx, uint32_t n_slots, uint32_t n_max_tokens) {
    ctx->graph_alt_slots_reserve(n_slots, n_max_tokens);
}

void llama_context::staggered_verify_init() {
    const char * env = getenv("LLAMA_STAGGERED_VERIFY");
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    const uint32_t n_split = hybrid ? hybrid->get_n_seq_split_tokens() : 0;

    // The gain is the overlap of sequences over the devices of a layer split. Without pipeline parallelism the ubatches
    // of the sequences run one after the other, and without an alternate graph slot per sequence they rebuild graphs.
    staggered_verify = (env == nullptr || atoi(env) != 0) && cparams.pipeline_parallel && !cparams.kv_unified &&
        cparams.n_seq_max > 1 && n_split > 0 && graph_alt_max_tokens >= n_split && graph_alt_slots >= cparams.n_seq_max;

    if (hybrid) {
        hybrid->set_seq_staggered(staggered_verify);
    }
    if (staggered_verify) {
        LLAMA_LOG_INFO("%s: staggered verify enabled\n", __func__);
    }
}

void llama_context::graph_alt_slots_reserve(uint32_t n_slots, uint32_t n_max_tokens) {
    if (graph_reuse_disable) {
        return;
    }
    if (!getenv("LLAMA_GRAPH_ALT_MAX_TOKENS")) {
        graph_alt_max_tokens = std::max(graph_alt_max_tokens, n_max_tokens);
    }
    if (!getenv("LLAMA_GRAPH_ALT_SLOTS")) {
        graph_alt_slots = std::max(graph_alt_slots, n_slots);
    }
    if (graph_alt_max_tokens > 0 && graph_alt_slots > 0) {
        // new slots are empty; their schedulers are created on first use (graph_slot_activate)
        const size_t n = 1 + (size_t) graph_alt_slots;
        if (graph_slots.size() < n) {
            graph_slots.resize(n);
            LLAMA_LOG_INFO("%s: %u alternate graph slots of up to %u tokens\n", __func__, graph_alt_slots, graph_alt_max_tokens);
        }
    }
}

void llama_feature_handoff_release(llama_context * ctx, int32_t ticket) {
    ctx->feature_handoff_release(ticket);
}

void llama_set_sampler_lazy(llama_context * ctx, bool enable) {
    ctx->set_sampler_lazy(enable);
}

bool llama_get_sampled_prob_ith(llama_context * ctx, int32_t i, llama_token token, float * prob) {
    return ctx->get_sampled_prob_ith(i, token, prob);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize_outputs();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

void llama_context_sched_reserve(struct llama_context * ctx) {
    ctx->sched_reserve();
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (!(flags & LLAMA_STATE_SEQ_FLAGS_ASYNC) || (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        ctx->synchronize();
    }

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    uint32_t trace_execution_id = 0;
    if (ggml_exec_trace_enabled()) {
        trace_execution_id = ggml_exec_trace_execution_begin(
                GGML_EXEC_TRACE_EXECUTION_ENCODE,
                batch.n_tokens,
                ctx->n_batch(),
                ctx->n_ubatch(),
                ctx->n_threads(),
                ctx->n_threads_batch());
    }
    const int ret = ctx->encode(batch);
    if (trace_execution_id) {
        ggml_exec_trace_execution_end(trace_execution_id, ret);
    }
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    uint32_t trace_execution_id = 0;
    if (ggml_exec_trace_enabled()) {
        const enum ggml_exec_trace_execution_kind trace_kind = batch.n_tokens > 1 ?
                GGML_EXEC_TRACE_EXECUTION_PREFILL : GGML_EXEC_TRACE_EXECUTION_DECODE;
        trace_execution_id = ggml_exec_trace_execution_begin(
                trace_kind,
                batch.n_tokens,
                ctx->n_batch(),
                ctx->n_ubatch(),
                ctx->n_threads(),
                ctx->n_threads_batch());
    }
    const int ret = ctx->decode(batch);
    if (trace_execution_id) {
        ggml_exec_trace_execution_end(trace_execution_id, ret);
    }
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
