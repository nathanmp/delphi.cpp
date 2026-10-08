#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama-cpp.h"
#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct hidden_check {
    std::vector<float> expected;
    int seen = 0;
};

static bool check_hidden(ggml_tensor * tensor, bool ask, void * opaque) {
    const bool selected = tensor->op == GGML_OP_RMS_NORM && tensor->src[0] &&
            strcmp(tensor->src[0]->name, "mtp_h_input") == 0;
    if (ask && selected) {
        auto & check = *static_cast<hidden_check *>(opaque);
        // Inspect before compute: the scheduler may reuse this input's storage afterwards.
        auto * input = tensor->src[0];
        std::vector<float> actual(ggml_nelements(input));
        GGML_ASSERT(actual.size() == check.expected.size());
        ggml_backend_tensor_get(input, actual.data(), 0, actual.size()*sizeof(float));
        GGML_ASSERT(actual == check.expected);
        check.seen++;
    }
    return selected;
}

static void fill_weights(ggml_tensor * tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> values(ggml_nelements(tensor));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = strstr(tensor->name, "norm") ? 1.0f : 0.05f*sinf(float(i + 1));
    }
    memcpy(tensor->data, values.data(), ggml_nbytes(tensor));
}

int main() {
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    gguf_context_ptr metadata(gguf_init_empty());
    llama_model_saver saver(LLM_ARCH_MIMO2, metadata.get());
    saver.add_kv(LLM_KV_GENERAL_ARCHITECTURE, "mimo2");
    saver.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
    saver.add_kv(LLM_KV_VOCAB_SIZE, uint32_t(32));
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(64));
    saver.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(32));
    saver.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(4));
    saver.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(3));
    saver.add_kv(LLM_KV_FEED_FORWARD_LENGTH, uint32_t(64));
    saver.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(32));
    saver.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(2));
    saver.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(1));
    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, uint32_t(2));
    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(1));
    saver.add_kv(LLM_KV_ATTENTION_KEY_LENGTH, uint32_t(16));
    saver.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, uint32_t(16));
    saver.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-6f);
    saver.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW, uint32_t(16));
    saver.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>({0, 1, 1, 1}));
    saver.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, uint32_t(8));
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 10000.0f);

    // Load a small fused-QKV checkpoint, including the three separate draft heads.
    ggml_context_ptr weights(ggml_init({2*1024*1024, nullptr, false}));
    auto add_weight = [&](const char * name, int64_t n0, int64_t n1 = 1) {
        auto * tensor = n1 == 1 ? ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, n0)
                               : ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, n0, n1);
        ggml_set_name(tensor, name);
        fill_weights(tensor);
        gguf_add_tensor(metadata.get(), tensor);
    };
    add_weight("token_embd.weight", 32, 32);
    add_weight("output.weight", 32, 32);
    add_weight("output_norm.weight", 32);
    for (int layer = 0; layer < 4; ++layer) {
        char name[128];
        auto add_layer = [&](const char * suffix, int64_t n0, int64_t n1 = 1) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", layer, suffix);
            add_weight(name, n0, n1);
        };
        add_layer("attn_qkv", 32, 64);
        add_layer("attn_output", 32, 32);
        add_layer("attn_norm", 32);
        add_layer("ffn_norm", 32);
        add_layer("ffn_gate", 32, 64);
        add_layer("ffn_up", 32, 64);
        add_layer("ffn_down", 64, 32);
        if (layer > 0) {
            add_layer("nextn.eh_proj", 64, 32);
            add_layer("nextn.enorm", 32);
            add_layer("nextn.hnorm", 32);
            add_layer("layer_output_norm", 32);
        }
    }

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    FILE * file = tmpfile();
    GGML_ASSERT(file && gguf_write_to_file_ptr(metadata.get(), file, false));
    rewind(file);
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model_ptr model(llama_model_load_from_file_ptr(file, mp));
    fclose(file);
    GGML_ASSERT(model);
    hidden_check check;
    auto cp = llama_context_default_params();
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx = 64;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = check_hidden;
    cp.cb_eval_user_data = &check;
    llama_context_ptr context(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(context);
    common_params params;
    params.numa = GGML_NUMA_STRATEGY_TENSORS;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 4;
    common_threadpools pools;
    pools.init(context.get(), params);

    llama_batch batch = llama_batch_init(8, 0, 1);
    for (int head = 0; head < 3; ++head) {
        llama_set_nextn_layer_offset(context.get(), head);
        for (int rows : {1, 3, 1}) {
            llama_memory_clear(llama_get_memory(context.get()), true);
            common_batch_clear(batch);
            check.expected.resize(rows*32);
            for (size_t i = 0; i < check.expected.size(); ++i) {
                check.expected[i] = sinf(float(i + rows + head));
            }
            for (int row = 0; row < rows; ++row) {
                common_batch_add(batch, row + 1, row, {0}, true);
            }
            batch.embd_h = check.expected.data();
            GGML_ASSERT(batch.embd == nullptr);
            const int before = check.seen;
            GGML_ASSERT(llama_decode(context.get(), batch) == 0);
            GGML_ASSERT(check.seen == before + 1);
            const float * logits = llama_get_logits_ith(context.get(), rows - 1);
            for (int token = 0; token < 32; ++token) {
                GGML_ASSERT(std::isfinite(logits[token]));
            }
        }
    }
    batch.embd_h = nullptr;
    llama_batch_free(batch);
    printf("MiMo MTP hidden input: all three heads and changing batch widths pass\n");
    return 0;
}
