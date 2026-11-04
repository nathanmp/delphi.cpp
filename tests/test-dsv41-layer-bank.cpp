#include "common.h"
#include "arg.h"
#include "../src/llama-ext.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using logits = std::vector<std::vector<float>>;

static void compare(const logits & expected, const logits & actual, const char * name, bool strict = true) {
    assert(expected.size() == actual.size());
    double worst_kl = 0, worst_nmse = 0;
    for (size_t row = 0; row < expected.size(); ++row) {
        const auto & a = expected[row];
        const auto & b = actual[row];
        assert(a.size() == b.size());
        const double ma = *std::max_element(a.begin(), a.end());
        const double mb = *std::max_element(b.begin(), b.end());
        double sa = 0, sb = 0, error = 0, energy = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            assert(std::isfinite(a[i]) && std::isfinite(b[i]));
            sa += std::exp(a[i] - ma);
            sb += std::exp(b[i] - mb);
            error += std::pow(double(a[i]) - b[i], 2);
            energy += double(a[i]) * a[i];
        }
        double kl = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            kl += std::exp(a[i] - ma) / sa * (a[i] - ma - std::log(sa) - b[i] + mb + std::log(sb));
        }
        worst_kl = std::max(worst_kl, kl);
        worst_nmse = std::max(worst_nmse, error / std::max(energy, 1e-30));
        const bool same_top = std::max_element(a.begin(), a.end()) - a.begin() == std::max_element(b.begin(), b.end()) - b.begin();
        if (kl >= 1e-5 || error / std::max(energy, 1e-30) >= 1e-6 || !same_top) {
            double mean_delta = 0;
            for (size_t i = 0; i < a.size(); ++i) { mean_delta += b[i] - a[i]; }
            mean_delta /= a.size();
            double centered_error = 0;
            for (size_t i = 0; i < a.size(); ++i) { centered_error += std::pow(double(b[i]) - a[i] - mean_delta, 2); }
            std::fprintf(stderr, "%s row %zu: KL %.9g, NMSE %.9g, same top %d, strict %d\n",
                    name, row, kl, error / std::max(energy, 1e-30), same_top, strict);
            std::fprintf(stderr, "max %.9g / %.9g, top probability %.9g, mean delta %.9g, centered NMSE %.9g\n",
                    ma, mb, 1.0 / sa, mean_delta, centered_error / std::max(energy, 1e-30));
        }
        if (strict) {
            assert(kl < 1e-5 && error / std::max(energy, 1e-30) < 1e-6);
            assert(same_top);
        }
    }
    std::printf("%s: %zu vectors, max KL %.9g, max NMSE %.9g\n", name, expected.size(), worst_kl, worst_nmse);
    std::fflush(stdout);
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) { return 1; }
    assert(params.numa == GGML_NUMA_STRATEGY_TENSORS && params.n_batch > params.n_ubatch + 17);
    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params, true);
    assert(init && init->model());
    auto cp = common_context_params_to_llama(params);
    cp.n_rs_seq = 4;
    llama_context_ptr ctx(llama_init_from_model(init->model(), cp));
    assert(ctx);
    auto * cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    auto new_pool = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new");
    auto free_pool = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free");
    assert(new_pool && free_pool);
    auto pool_params = ggml_threadpool_params_default(cp.n_threads);
    auto pool = std::unique_ptr<ggml_threadpool, decltype(free_pool)>(new_pool(&pool_params), free_pool);
    assert(pool);
    llama_attach_threadpool(ctx.get(), pool.get(), pool.get());
    llama_set_embeddings_layer_inp(ctx.get(), 37, true);

    const int prompt = params.n_batch - 17;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));
    const int embd = llama_model_n_embd(init->model());
    const auto corpus = common_tokenize(ctx.get(),
            "Preserve causal state across chunks. A layer reads previous tokens before it updates the next state. "
            "def update(previous, item): return previous + item\n", true);
    assert(!corpus.empty());
    std::vector<llama_token> tokens(prompt + 32);
    for (size_t i = 0; i < tokens.size(); ++i) { tokens[i] = corpus[i % corpus.size()]; }
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    const auto decode = [&](int pos, int count, bool all) {
        common_batch_clear(batch);
        for (int i = 0; i < count; ++i) {
            common_batch_add(batch, tokens[pos + i], pos + i, {0}, all || i + 1 == count);
        }
        return llama_decode(ctx.get(), batch);
    };
    const auto copy_logits = [&](int row) {
        const float * data = llama_get_logits_ith(ctx.get(), row);
        assert(data);
        return std::vector<float>(data, data + vocab);
    };
    const auto prefill = [&](int mode) {
        setenv("LLAMA_DSV41_LAYER_BANK", std::to_string(mode).c_str(), 1);
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        std::printf("prefill mode %d, %d tokens\n", mode, prompt);
        std::fflush(stdout);
        assert(decode(0, prompt, false) == 0);
        const float * data = llama_get_embeddings_layer_inp(ctx.get(), 37);
        assert(data);
        logits features;
        for (int row : {params.n_ubatch - 1, prompt - 128, prompt - 2, prompt - 1}) {
            features.emplace_back(data + size_t(row) * embd, data + size_t(row + 1) * embd);
        }
        return features;
    };
    const auto score = [&](bool prefix) {
        logits result;
        if (prefix) { result.push_back(copy_logits(-1)); }
        for (int step = 0; step < 4; ++step) {
            assert(decode(prompt + step, 1, true) == 0);
            result.push_back(copy_logits(-1));
        }
        assert(decode(prompt + 4, 5, true) == 0);
        for (int i = 0; i < 5; ++i) { result.push_back(copy_logits(i)); }
        assert(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, prompt + 7, -1));
        assert(decode(prompt + 7, 1, true) == 0);
        result.push_back(copy_logits(-1));
        return result;
    };
    const auto save = [&] {
        std::vector<uint8_t> state(llama_state_seq_get_size(ctx.get(), 0));
        const size_t size = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), 0);
        assert(size && size <= state.size());
        state.resize(size);
        return state;
    };
    const auto restore = [&](const std::vector<uint8_t> & state) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        assert(llama_state_seq_set_data(ctx.get(), state.data(), state.size(), 0) == state.size());
    };
    const auto dump = [](const std::vector<uint8_t> & state, int mode) {
        if (const char * path = std::getenv("LLAMA_LAYER_BANK_TEST_DUMP")) {
            std::ofstream out(std::string(path) + "/state-" + std::to_string(mode) + ".bin", std::ios::binary);
            out.write(reinterpret_cast<const char *>(state.data()), state.size());
            assert(out);
        }
    };
    const auto dump_vectors = [](const logits & values, const char * name) {
        const char * path = std::getenv("LLAMA_LAYER_BANK_TEST_DUMP");
        assert(path && !values.empty());
        std::ofstream out(std::string(path) + "/" + name + ".bin", std::ios::binary);
        const uint32_t shape[] = { uint32_t(values.size()), uint32_t(values[0].size()) };
        out.write(reinterpret_cast<const char *>(shape), sizeof(shape));
        for (const auto & row : values) {
            assert(row.size() == shape[1]);
            out.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
        }
        assert(out);
    };

    if (const char * env = std::getenv("LLAMA_LAYER_BANK_TEST_MODE")) {
        const int mode = std::atoi(env);
        assert(mode >= 0 && mode <= 2);
        const auto fixed_features = prefill(mode);
        const auto fixed_state = save();
        const auto fixed_logits = score(true);
        dump_vectors(fixed_features, "features");
        dump(fixed_state, mode);
        dump_vectors(fixed_logits, "logits");
        if (mode == 1) {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            int calls = 0;
            llama_set_abort_callback(ctx.get(), [](void * data) { return ++*static_cast<int *>(data) > 5; }, &calls);
            assert(decode(0, prompt, false) == 2);
            llama_set_abort_callback(ctx.get(), nullptr, nullptr);
            assert(llama_memory_seq_pos_min(llama_get_memory(ctx.get()), 0) == -1);
        }
        compare(fixed_features, prefill(mode), "fixed-mode repeat features");
        assert(save() == fixed_state);
        const auto repeated = score(true);
        dump_vectors(repeated, "logits-repeated");
        compare(fixed_logits, repeated, "fixed-mode repeat control", false);
        llama_batch_free(batch);
        llama_detach_threadpool(ctx.get());
        ctx.reset();
        std::puts("fixed-mode capture complete");
        return 0;
    }

    const auto features = prefill(0);
    const auto baseline_state = save();
    dump(baseline_state, 0);
    const auto baseline = score(true);
    const logits suffix(baseline.begin() + 1, baseline.end());
    restore(baseline_state);
    const auto restored_baseline = score(false);
    // Ordinary byte-state restoration can change the physical cache layout. Compare the new path through the same operation.
    compare(suffix, restored_baseline, "ordinary restore control", false);
    for (int mode : {2, 1}) {
        compare(features, prefill(mode), mode == 2 ? "schedule features" : "retained features");
        const auto state = save();
        dump(state, mode);
        size_t different = 0;
        for (size_t i = 0; i < std::min(state.size(), baseline_state.size()); ++i) {
            different += state[i] != baseline_state[i];
        }
        std::printf("mode %d state image: %zu / %zu bytes, %zu differing bytes\n",
                mode, state.size(), baseline_state.size(), different);
        std::fflush(stdout);
        assert(state == baseline_state);
        // The continuation from the same state is not bit-reproducible between two ordinary runs either (KL up to ~3e-5 on
        // 2 R9700), so the logits are reported; the state bytes above are the strict check.
        compare(baseline, score(true), mode == 2 ? "schedule logits" : "retained logits", false);
        setenv("LLAMA_DSV41_LAYER_BANK", "0", 1);
        restore(state);
        // Byte identity above checks the window's serialized state. Restore numerics also vary in the ordinary control.
        compare(restored_baseline, score(false), "window state -> ordinary restored continuation", false);
    }

    setenv("LLAMA_DSV41_LAYER_BANK", "1", 1);
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    int abort_calls = 0;
    llama_set_abort_callback(ctx.get(), [](void * data) { return ++*static_cast<int *>(data) > 5; }, &abort_calls);
    assert(decode(0, prompt, false) == 2);
    llama_set_abort_callback(ctx.get(), nullptr, nullptr);
    assert(llama_memory_seq_pos_min(llama_get_memory(ctx.get()), 0) == -1);
    compare(features, prefill(1), "abort recovery features");
    compare(baseline, score(true), "abort recovery logits", false);
    llama_batch_free(batch);
    llama_detach_threadpool(ctx.get());
    ctx.reset();
    std::puts("layer bank: logits, decoder features, state restore, verify rollback and abort recovery passed");
}
