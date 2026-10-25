#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"
#include "llama-cpp.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

static void env(const char * name, const std::string & value) {
    setenv(name, value.c_str(), 1);
}

static void fill(ggml_tensor * tensor, void *) {
    std::mt19937 rng(17);
    std::normal_distribution<float> normal(0.0f, 0.01f);
    std::vector<float> values(ggml_nelements(tensor));
    for (auto & x : values) { x = normal(rng); }
    std::vector<uint8_t> data(ggml_nbytes(tensor));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, data.size());
    } else {
        assert(ggml_quantize_chunk(tensor->type, values.data(), data.data(), 0,
                    ggml_nelements(tensor) / tensor->ne[0], tensor->ne[0], nullptr) == data.size());
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
    }
}

static llama_model_ptr make_model(const std::vector<ggml_backend_dev_t> & devices, bool layer_split = false) {
    gguf_context_ptr metadata(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN2MOE, metadata.get());
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, "qwen2moe");
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
    ms.add_kv(LLM_KV_VOCAB_SIZE, uint32_t(128));
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(2048));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(128));
    ms.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(3));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, uint32_t(256));
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(256));
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, uint32_t(128));
    ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(64));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    for (int il = 0; il < 3; ++il) {
        for (const char * projection : {"gate", "up", "down"}) {
            ggml_tensor tensor = {};
            tensor.type = GGML_TYPE_Q4_0;
            ggml_format_name(&tensor, "blk.%d.ffn_%s_exps.weight", il, projection);
            gguf_add_tensor(metadata.get(), &tensor);
        }
    }
    auto params = llama_model_default_params();
    const float split_single[] = {0, 0, 0, 1};
    const float split_layers[] = {1, 1, 1, 1};
    const llama_model_tensor_buft_override overrides[] = {
        {"ffn_(gate|up|down)_exps", ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))},
        {nullptr, nullptr},
    };
    auto selected = devices;
    selected.push_back(nullptr);
    params.devices = selected.data();
    params.tensor_split = layer_split ? split_layers : split_single;
    params.tensor_buft_overrides = overrides;
    params.n_gpu_layers = 999;
    params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    return llama_model_ptr(llama_model_init_from_user(metadata.get(), fill, nullptr, params));
}

static void record(llama_expert_cache & cache, ggml_backend_t cpu, int first) {
    ggml_context_ptr ctx(ggml_init({size_t(1) << 20, nullptr, true}));
    auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, 256);
    auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    for (int il = 0; il < 3; ++il) { ggml_build_forward_expand(graph, cache.build_record(ctx.get(), ids, il)); }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu));
    assert(buffer);
    std::vector<int32_t> route(512);
    for (size_t i = 0; i < route.size(); ++i) { route[i] = first + i % 32; }
    ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
    assert(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    cache.needs_refresh(256, false);
    assert(cache.needs_refresh(1, false));
    cache.refresh();
}

static std::vector<int32_t> routes(const llama_expert_cache & cache, int tier) {
    std::vector<int32_t> result(3 * 64);
    for (int il = 0; il < 3; ++il) {
        auto * ids = cache.hot_ids(tier, il);
        assert(ids);
        ggml_backend_tensor_get(ids, result.data() + il * 64, 0, 64 * sizeof(int32_t));
    }
    return result;
}

static std::vector<uint8_t> bank_bytes(const llama_expert_cache & cache, int tier) {
    std::vector<uint8_t> result;
    for (int p = 0; p < 3; ++p) {
        auto * bank = cache.bank(tier, 0, p);
        const size_t off = result.size();
        result.resize(off + ggml_nbytes(bank));
        ggml_backend_tensor_get(bank, result.data() + off, 0, ggml_nbytes(bank));
    }
    return result;
}

static void check_partition(const llama_expert_cache & cache) {
    for (int il = 0; il < 3; ++il) {
        std::vector<int> owners(64, 0);
        for (int t = 0; t < cache.n_tiers(); ++t) {
            auto * ids = cache.hot_ids(t, il);
            if (!ids) { continue; }
            std::vector<int32_t> map(64);
            ggml_backend_tensor_get(ids, map.data(), 0, map.size() * sizeof(int32_t));
            for (int e = 0; e < 64; ++e) { owners[e] += map[e] >= 0; }
        }
        for (int e = 0; e < 64; ++e) {
            const int cold = cache.layer(il)->cold[e];
            assert(cold == -1 || cold == e);
            assert(owners[e] == (cold < 0 ? 1 : 0));
        }
    }
}

static void decode(llama_context * ctx, int n) {
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<llama_token> tokens(n);
    for (int i = 0; i < n; ++i) { tokens[i] = i % 128; }
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    std::vector<int8_t> outputs(n, 1);
    batch.logits = outputs.data();
    assert(llama_decode(ctx, batch) == 0);
    const auto * logits = llama_get_logits_ith(ctx, -1);
    for (int i = 0; i < 128; ++i) { assert(std::isfinite(logits[i])); }
}

int main(int argc, char ** argv) {
    if (argc != 8 || std::string(argv[1]) != "--numa" || std::string(argv[2]) != "tensors") {
        fprintf(stderr, "usage: %s --numa tensors <GPU0> <GPU1> <GPU2> <pinned GPU> <stage trace>\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    llama_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    std::vector<ggml_backend_dev_t> devices;
    std::vector<std::string> names;
    for (int i = 3; i < 7; ++i) {
        auto * dev = ggml_backend_dev_by_name(argv[i]);
        assert(dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU);
        devices.push_back(dev);
        names.emplace_back(argv[i]);
    }
    env("LLAMA_MOE_CACHE_MIB", "1");
    env("LLAMA_MOE_CACHE_ROLES", names[0] + ":rotate," + names[1] + ":static," + names[2] + ":off," + names[3] + ":pinned");
    env("LLAMA_MOE_CACHE_STATIC", names[0]); // explicit rotate must override the legacy static list
    env("LLAMA_MOE_STAGE_DEVICES", names[0] + "," + names[1] + "," + names[2]);
    env("LLAMA_MOE_CACHE_AUTO_MARGIN_MIB", "1536," + names[2] + ":6144");
    env("LLAMA_MOE_CACHE_WINDOW", "256");
    env("LLAMA_MOE_CACHE_HYSTERESIS", "0");
    env("LLAMA_MOE_CACHE_STATIC_SWAPS", "64");
    env("LLAMA_MOE_CACHE_EVERY", "1");
    env("LLAMA_MOE_CACHE_TIERS", names[0] + ":1," + names[1] + ":1," + names[2] + ":1");
    env("GGML_SCHED_STAGE", "1");
    env("GGML_SCHED_STAGE_SPREAD", "1");
    env("GGML_SCHED_STAGE_SLOTS", "4");
    env("GGML_SCHED_STAGE_TRACE", argv[7]);
    auto model = make_model(devices);
    assert(model);
    auto params = llama_context_default_params();
    params.n_ctx = 2048;
    params.n_batch = params.n_ubatch = 1024;
    params.n_threads = params.n_threads_batch = 4;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr ctx(llama_init_from_model(model.get(), params));
    assert(ctx);
    auto * cache = ctx->share_expert_cache();
    assert(cache && cache->n_tiers() == 3);
    assert(ggml_backend_get_device(cache->tier_backend(0)) == devices[3]);
    assert(!llama_share_has_cache_on_device(ctx.get(), names[2].c_str()));
    ggml_backend_ptr cpu(ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr));
    record(*cache, cpu.get(), 0);
    check_partition(*cache);
    const auto pinned = routes(*cache, 0);
    const auto weights = bank_bytes(*cache, 0);
    const auto rotating = routes(*cache, 1);
    assert(std::count_if(pinned.begin(), pinned.end(), [](int32_t s) { return s >= 0; }) > 0);
    assert(std::count_if(rotating.begin(), rotating.end(), [](int32_t s) { return s >= 0; }) > 0);
    void * slot = nullptr;
    assert(!llama_expert_cache::lend(cache, cache->tier_backend(0), 1, 64, &slot));
    assert(routes(*cache, 0) == pinned && bank_bytes(*cache, 0) == weights);
    assert(llama_expert_cache::lend(cache, cache->tier_backend(1), 1, 64, &slot));
    cache->reclaimed();
    record(*cache, cpu.get(), 32);
    check_partition(*cache);
    assert(routes(*cache, 0) == pinned && bank_bytes(*cache, 0) == weights);
    assert(routes(*cache, 1) != rotating);

    int excluded = -1;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(ctx->get_sched()); ++i) {
        if (ggml_backend_get_device(ggml_backend_sched_get_backend(ctx->get_sched(), i)) == devices[3]) { excluded = i; }
    }
    assert(excluded >= 0);
    decode(ctx.get(), 1024);
    decode(ctx.get(), 4);
    assert(routes(*cache, 0) == pinned && bank_bytes(*cache, 0) == weights);
    llama_share_stage_allow(ctx.get(), names[3].c_str(), true); // a lease must not override the launch policy
    llama_share_release_compute(ctx.get());
    decode(ctx.get(), 1024);
    decode(ctx.get(), 4);
    assert(routes(*cache, 0) == pinned && bank_bytes(*cache, 0) == weights);
    std::ifstream trace(argv[7]);
    std::set<int> staged;
    int backend, index;
    size_t bytes;
    int64_t a, b, c, d;
    while (trace >> backend >> index >> bytes >> a >> b >> c >> d) { staged.insert(backend); }
    assert(staged.size() == 3 && !staged.count(excluded));
    ctx.reset();

    // Keep the real auto-sizing test bounded. Sample free memory after the first context has released its buffers.
    std::string margins = "999999";
    for (int i : {0, 3}) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(devices[i], &free, &total);
        const size_t free_mib = free >> 20;
        assert(free_mib > 256);
        margins += "," + names[i] + ":" + std::to_string(free_mib - 128);
    }
    env("LLAMA_MOE_CACHE_MIB", "auto");
    env("LLAMA_MOE_CACHE_AUTO_MARGIN_MIB", margins);
    unsetenv("LLAMA_MOE_CACHE_TIERS");
    llama_context_ptr automatic(llama_init_from_model(model.get(), params));
    assert(automatic);
    assert(llama_share_has_cache_on_device(automatic.get(), names[0].c_str()));
    assert(!llama_share_has_cache_on_device(automatic.get(), names[2].c_str()));
    llama_expert_cache_auto(automatic.get());
    auto * auto_cache = automatic->share_expert_cache();
    assert(auto_cache && auto_cache->n_tiers() == 2);
    for (const auto & tier : auto_cache->tier_list()) {
        auto * dev = ggml_backend_get_device(tier.second);
        assert(dev == devices[0] || dev == devices[3]);
        assert(auto_cache->tier_bytes(tier.first) <= (size_t(129) << 20));
    }
    assert(!llama_share_has_cache_on_device(automatic.get(), names[1].c_str()));
    llama_expert_cache_auto(automatic.get());
    assert(auto_cache->n_tiers() == 2);
    decode(automatic.get(), 4);
    automatic.reset();
    model.reset();

    // A pinned tier must also serve the hottest experts from another dense stage, without widening rotating domains.
    env("LLAMA_MOE_CACHE_MIB", "1");
    env("LLAMA_MOE_CACHE_TIERS", names[1] + ":1," + names[3] + ":1");
    auto split_model = make_model(devices, true);
    assert(split_model);
    llama_context_ptr split_ctx(llama_init_from_model(split_model.get(), params));
    assert(split_ctx);
    auto * split_cache = split_ctx->share_expert_cache();
    assert(split_cache && split_cache->n_tiers() == 3);
    int pinned_tier = -1;
    for (int t = 0; t < split_cache->n_tiers(); ++t) {
        if (ggml_backend_get_device(split_cache->tier_backend(t)) == devices[3]) { pinned_tier = t; }
    }
    assert(pinned_tier >= 0);
    assert(ggml_backend_get_device(split_cache->backend(0)) != devices[3]);
    record(*split_cache, cpu.get(), 0);
    check_partition(*split_cache);
    const auto global_pins = routes(*split_cache, pinned_tier);
    assert(std::any_of(global_pins.begin(), global_pins.begin() + 64, [](int32_t slot) { return slot >= 0; }));
    assert(split_cache->serves_remote(split_cache->tier_backend(pinned_tier)));
    for (int t = 0; t < split_cache->n_tiers(); ++t) {
        if (t == pinned_tier) { continue; }
        for (int il = 0; il < 3; ++il) {
            if (split_cache->tier_backend(t) != split_cache->backend(il)) { assert(!split_cache->hot_ids(t, il)); }
        }
    }
    decode(split_ctx.get(), 4);
    record(*split_cache, cpu.get(), 32);
    check_partition(*split_cache);
    assert(routes(*split_cache, pinned_tier) == global_pins);
    decode(split_ctx.get(), 4);
    puts("expert device GPU lifecycle: pinned weights/maps stable; rotating cache changed; three staging devices; recreated scheduler excluded pinned GPU");
}
