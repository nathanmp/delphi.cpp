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

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    gguf_context_ptr metadata(gguf_init_empty());
    llama_model_saver saver(LLM_ARCH_LLAMA, metadata.get());
    saver.add_kv(LLM_KV_GENERAL_ARCHITECTURE, "llama");
    saver.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
    saver.add_kv(LLM_KV_VOCAB_SIZE, uint32_t(32));
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(64));
    saver.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(32));
    saver.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(3));
    saver.add_kv(LLM_KV_FEED_FORWARD_LENGTH, uint32_t(64));
    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, uint32_t(2));
    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(1));
    saver.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-6f);
    saver.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, uint32_t(16));
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 10000.0f);

    ggml_context_ptr weights(ggml_init({2*1024*1024, nullptr, false}));
    auto add = [&](const char * name, int64_t n0, int64_t n1 = 1) {
        auto * tensor = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, n0, n1);
        ggml_set_name(tensor, name);
        auto * data = (float *) tensor->data;
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            data[i] = strstr(name, "norm") ? 1.0f : 0.05f*sinf(float(i + 1));
        }
        gguf_add_tensor(metadata.get(), tensor);
    };
    add("token_embd.weight", 32, 32);
    add("output.weight", 32, 32);
    add("output_norm.weight", 32);
    for (int layer = 0; layer < 3; ++layer) {
        char name[128];
        auto add_layer = [&](const char * suffix, int64_t n0, int64_t n1 = 1) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", layer, suffix);
            add(name, n0, n1);
        };
        add_layer("attn_norm", 32);
        add_layer("attn_q", 32, 32);
        add_layer("attn_k", 32, 16);
        add_layer("attn_v", 32, 16);
        add_layer("attn_output", 32, 32);
        add_layer("ffn_norm", 32);
        add_layer("ffn_gate", 32, 64);
        add_layer("ffn_up", 32, 64);
        add_layer("ffn_down", 64, 32);
    }
    ggml_backend_dev_t devices[] = { nullptr, nullptr };
    auto mp = llama_model_default_params();
    if (argc > 1) {
        devices[0] = ggml_backend_dev_by_name(argv[1]);
        GGML_ASSERT(devices[0]);
        mp.devices = devices;
        mp.n_gpu_layers = 99;
    } else {
        mp.n_gpu_layers = 0;
    }
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    FILE * file = tmpfile();
    GGML_ASSERT(file && gguf_write_to_file_ptr(metadata.get(), file, false));
    rewind(file);
    llama_model_ptr model(llama_model_load_from_file_ptr(file, mp));
    fclose(file);
    GGML_ASSERT(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 8;
    cp.n_ubatch = 4;
    cp.n_seq_max = 2;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(ctx);
    common_params params;
    params.numa = GGML_NUMA_STRATEGY_TENSORS;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 4;
    common_threadpools pools;
    pools.init(ctx.get(), params);
    // Enable out of order: the fused buffer must still use ascending layer IDs.
    llama_set_embeddings_layer_inp(ctx.get(), 2, true);
    llama_set_embeddings_layer_inp(ctx.get(), 0, true);
    std::vector<std::vector<float>> expected;
    llama_batch batch = llama_batch_init(8, 32, 1);
    for (bool device : {false, true, false, true}) {
        llama_set_embeddings_layer_inp_dev(ctx.get(), device);
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        for (int round = 0; round < 2; ++round) {
            const int per_seq = round == 0 ? 4 : 2;
            batch.n_tokens = 2 * per_seq;
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.pos[i] = (round == 0 ? 0 : 4) + i % per_seq;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = round == 0 ? 1 - i / per_seq : i / per_seq;
                batch.logits[i] = false;
                for (int k = 0; k < 32; ++k) {
                    batch.embd[i*32 + k] = sinf(float(1 + i*32 + k + round*17));
                }
            }
            GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
            llama_synchronize(ctx.get());
            std::vector<float> actual(batch.n_tokens * 64);
            auto * tensor = llama_get_embeddings_layer_inp_tensor(ctx.get());
            if (device) {
                GGML_ASSERT(tensor && tensor->ne[0] == 64 && tensor->ne[1] == 8);
                ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size()*sizeof(float));
            } else {
                GGML_ASSERT(!tensor);
                for (int layer = 0; layer < 2; ++layer) {
                    const float * src = llama_get_embeddings_layer_inp(ctx.get(), layer * 2);
                    for (int i = 0; i < batch.n_tokens; ++i) {
                        memcpy(actual.data() + i*64 + layer*32, src + i*32, 32*sizeof(float));
                    }
                }
            }
            if (expected.size() <= (size_t) round) {
                expected.push_back(actual);
            } else {
                for (size_t i = 0; i < actual.size(); ++i) {
                    GGML_ASSERT(std::isfinite(actual[i]));
                    GGML_ASSERT(fabsf(actual[i] - expected[round][i]) < 1e-5f);
                }
            }
        }
    }
    llama_batch_free(batch);
    fprintf(stderr, "layer input device extraction: OK\n");
}
