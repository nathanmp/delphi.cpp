#include "../src/llama-ext.h"
#include "ggml-cpp.h"
#include "llama.h"

#include <cstdio>

static llama_model * make_model(const char * architecture) {
    const llama_quant_model_desc desc = {
        /*.architecture  =*/ architecture,
        /*.n_embd        =*/ 5120,
        /*.n_ff          =*/ 2304,
        /*.n_layer       =*/ 40,
        /*.n_head        =*/ 64,
        /*.n_head_kv     =*/ 1,
        /*.n_expert      =*/ 384,
        /*.n_embd_head_k =*/ 512,
        /*.n_embd_head_v =*/ 512,
    };
    return llama_quant_model_from_metadata(&desc);
}

static ggml_type selected_type(
        quantize_state_impl * qs,
        const char * name,
        ggml_type source_type = GGML_TYPE_BF16,
        int64_t ne0 = 5120,
        int64_t ne1 = 512) {
    const size_t ctx_size = ggml_tensor_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx.get(), source_type, ne0, ne1);
    ggml_set_name(tensor, name);

    GGML_ASSERT(llama_quant_tensor_allows_quantization(qs, tensor));

    ggml_type result = GGML_TYPE_COUNT;
    llama_quant_compute_types(qs, LLAMA_FTYPE_MOSTLY_Q1_0, &tensor, &result, 1);
    return result;
}

static void test_deepseek41_protection() {
    llama_model_quantize_params params = llama_model_quantize_default_params();
    llama_model * model = make_model("deepseek41");
    quantize_state_impl * qs = llama_quant_init(model, &params);

    const char * protected_bf16[] = {
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_q_a.weight",
        "blk.0.attn_q_b.weight",
        "blk.0.attn_kv.weight",
        "blk.0.attn_output_a.weight",
        "blk.0.attn_output_b.weight",
        "blk.0.attn_compressor_kv.weight",
        "blk.0.attn_compressor_gate.weight",
        "blk.0.indexer.proj.weight",
        "blk.0.indexer.attn_k.weight",
        "blk.0.indexer.attn_q_b.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.engram_wkv.weight",
    };
    for (const char * name : protected_bf16) {
        GGML_ASSERT(selected_type(qs, name) == GGML_TYPE_BF16);
    }
    GGML_ASSERT(selected_type(qs, "blk.0.hc_attn_fn.weight", GGML_TYPE_F32, 20480, 24) == GGML_TYPE_F32);
    GGML_ASSERT(selected_type(qs, "blk.1.engram_embd.weight", GGML_TYPE_Q8_0, 256, 1024) == GGML_TYPE_Q8_0);

    GGML_ASSERT(selected_type(qs, "blk.0.ffn_gate_exps.weight") == GGML_TYPE_Q1_0);
    GGML_ASSERT(selected_type(qs, "blk.0.ffn_down_exps.weight") == GGML_TYPE_Q1_0);
    GGML_ASSERT(selected_type(qs, "blk.0.ffn_up_exps.weight") == GGML_TYPE_Q1_0);

    llama_quant_free(qs);
    llama_model_free(model);
}

static void test_unrelated_architectures_unchanged() {
    llama_model_quantize_params params = llama_model_quantize_default_params();

    const char * architectures[] = { "deepseek2", "deepseek4", "llama" };
    for (const char * architecture : architectures) {
        llama_model * model = make_model(architecture);
        quantize_state_impl * qs = llama_quant_init(model, &params);

        GGML_ASSERT(selected_type(qs, "blk.0.attn_q_a.weight") == GGML_TYPE_Q1_0);
        GGML_ASSERT(selected_type(qs, "blk.0.ffn_gate_shexp.weight") == GGML_TYPE_Q1_0);
        GGML_ASSERT(selected_type(qs, "blk.0.engram_wkv.weight") == GGML_TYPE_Q1_0);

        llama_quant_free(qs);
        llama_model_free(model);
    }
}

int main() {
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    test_deepseek41_protection();
    test_unrelated_architectures_unchanged();

    std::puts("DeepSeek-V4.1 quantization protection tests passed");
    return 0;
}
