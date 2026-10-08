#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "gguf.h"
#include "speculative.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Internal model map is used only by this diagnostic executable, before its first decode.
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

static void replace_probe_weights(llama_model * model, const char * path) {
    ggml_context * context = nullptr;
    auto * file = gguf_init_from_file(path, {false, &context});
    GGML_ASSERT(file && context);
    int replaced = 0;
    for (const auto & entry : llama_internal_get_tensor_map(model)) {
        auto * source = ggml_get_tensor(context, entry.first.c_str());
        if (!source) { continue; }
        auto * target = entry.second;
        GGML_ASSERT(source->type == GGML_TYPE_Q4_G128 && ggml_are_same_shape(source, target) && ggml_nbytes(source) <= ggml_nbytes(target));
        target->type = source->type;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) { target->nb[i] = source->nb[i]; }
        ggml_backend_tensor_set(target, source->data, 0, ggml_nbytes(source));
        replaced++;
    }
    GGML_ASSERT(replaced == 3);
    gguf_free(file); ggml_free(context);
    fprintf(stderr, "probe: replaced three resident expert tensors; original allocation remains (no VRAM-saving claim)\n");
}

// Diagnostic export only. This callback synchronizes and is excluded from performance measurements.
struct capture_state { std::string directory; int width; bool enabled = false, done = false; };

static void save(const std::string & path, ggml_tensor * t) {
    std::vector<char> data(ggml_nelements(t) * ggml_type_size(t->type));
    if (ggml_is_contiguous(t)) { ggml_backend_tensor_get(t, data.data(), 0, data.size()); }
    else {
        GGML_ASSERT(t->type == GGML_TYPE_I32 && t->ne[2] == 1 && t->ne[3] == 1);
        const size_t row = t->ne[0] * sizeof(int32_t);
        for (int i = 0; i < t->ne[1]; ++i) { ggml_backend_tensor_get(t, data.data() + i * row, i * t->nb[1], row); }
    }
    std::ofstream file(path, std::ios::binary);
    file.write(data.data(), data.size());
    GGML_ASSERT(file.good());
}

static bool capture(ggml_tensor * t, bool ask, void * opaque) {
    auto & s = *static_cast<capture_state *>(opaque);
    if (!s.enabled || s.done || std::string(t->name) != "ffn_moe_weighted-0" || t->op != GGML_OP_MUL) { return false; }
    if (ask) { return true; }
    auto * down = t->src[0];
    auto * glu = down->src[1];
    GGML_ASSERT(down->op == GGML_OP_MUL_MAT_ID && glu->op == GGML_OP_GLU);
    auto * input = glu->src[0]->src[1];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && down->src[2]->type == GGML_TYPE_I32);
    save(s.directory + "/input.f32", input);
    save(s.directory + "/ids.i32", down->src[2]);
    save(s.directory + "/router.f32", t->src[1]);
    std::ofstream meta(s.directory + "/fixture.json");
    meta << "{\"layer\":0,\"h\":" << input->ne[0] << ",\"width\":" << input->ne[2]
         << ",\"topk\":" << down->src[2]->ne[0] << ",\"kind\":\"teacher-forced coding continuation\"}\n";
    s.done = true;
    return true;
}

int main(int argc, char ** argv) {
    const char * backends = getenv("DELPHI_COMPARE_BINDIR"), * directory = getenv("DELPHI_COMPARE_CAPTURE");
    GGML_ASSERT(backends && directory);
    ggml_backend_load_all_from_path(backends);
    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) { return 1; }
    params.warmup = false;
    const char * bank = getenv("DELPHI_COMPARE_OVERRIDE_BANK");
    capture_state state{directory, 17};
    std::filesystem::create_directories(directory);
    params.cb_eval = bank ? nullptr : capture;
    params.cb_eval_user_data = &state;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    if (!init || !init->context()) { return 1; }
    int replaced_layers = 0;
    if (bank) {
        std::string paths = bank;
        size_t start = 0;
        do {
            const size_t end = paths.find(',', start);
            replace_probe_weights(init->model(), paths.substr(start, end == std::string::npos ? end : end - start).c_str());
            replaced_layers++;
            if (end == std::string::npos) { break; }
            start = end + 1;
        } while (start < paths.size());
    }
    common_threadpools pools;
    pools.init(init->context(), params);
    common_speculative_init_result_ptr draft_init;
    common_speculative_ptr spec;
    if (bank && !params.speculative.types.empty()) {
        GGML_ASSERT(!params.speculative.has_synth());
        auto dparams = common_base_params_to_speculative(params);
        draft_init = common_speculative_init_from_params(dparams, init->model(), init->context());
        GGML_ASSERT(draft_init && draft_init->context());
        params.speculative.draft.ctx_tgt = init->context();
        params.speculative.draft.ctx_dft = draft_init->context();
        spec.reset(common_speculative_init(params.speculative, 1));
    }
    auto * vocab = llama_model_get_vocab(init->model());
    auto tokens = common_tokenize(vocab, "Write a Python function that merges two sorted lists without modifying its inputs.\n```python\ndef merge_sorted(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n", true);
    auto batch = llama_batch_init(std::max(tokens.size(), size_t(state.width)), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) { common_batch_add(batch, tokens[i], i, {0}, i == tokens.size() - 1); }
    if (llama_decode(init->context(), batch)) { return 1; }
    auto next = common_tokenize(vocab, "        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n", false);
    GGML_ASSERT(next.size() >= size_t(state.width));
    common_batch_clear(batch);
    if (spec) {
        // Ingest the prompt before resetting the target batch; MTP pairs its hidden rows with the next input tokens.
        for (size_t i = 0; i < tokens.size(); ++i) { common_batch_add(batch, tokens[i], i, {0}, i == tokens.size() - 1); }
        GGML_ASSERT(common_speculative_process(spec.get(), batch));
        common_speculative_begin(spec.get(), 0, tokens);
        const float * logits = llama_get_logits_ith(init->context(), -1);
        const llama_token last = std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits;
        llama_tokens drafted;
        common_speculative_get_draft_params(spec.get(), 0) = {true, params.speculative.draft.n_max, int32_t(tokens.size()), last, &tokens, &drafted};
        common_speculative_draft(spec.get());
        GGML_ASSERT(!drafted.empty());
        common_batch_clear(batch);
        common_batch_add(batch, last, tokens.size(), {0}, true);
        for (size_t i = 0; i < drafted.size(); ++i) { common_batch_add(batch, drafted[i], tokens.size() + 1 + i, {0}, true); }
        fprintf(stderr, "probe: normal MTP drafted %zu tokens, target verification width %d\n", drafted.size(), batch.n_tokens);
    } else {
        for (int i = 0; i < state.width; ++i) { common_batch_add(batch, next[i], tokens.size() + i, {0}, true); }
    }
    state.enabled = true;
    const int status = llama_decode(init->context(), batch);
    if (!status && spec) {
        int accepted = 0;
        for (int i = 0; i + 1 < batch.n_tokens; ++i) {
            const float * logits = llama_get_logits_ith(init->context(), i);
            const llama_token token = std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits;
            if (token != batch.token[i + 1]) { break; }
            accepted++;
        }
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft_init->context()), 0, tokens.size(), -1));
        GGML_ASSERT(common_speculative_process(spec.get(), batch));
        common_speculative_accept(spec.get(), 0, accepted);
        fprintf(stderr, "probe: normal greedy acceptance %d/%d (one cycle, no quality claim)\n", accepted, batch.n_tokens - 1);
    }
    llama_batch_free(batch);
    if (bank) {
        const float * logits = llama_get_logits_ith(init->context(), -1);
        GGML_ASSERT(logits);
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) { GGML_ASSERT(std::isfinite(logits[i])); }
        auto * reg = ggml_backend_reg_by_name("ROCm");
        GGML_ASSERT(reg);
        auto stats = reinterpret_cast<void (*)(int64_t *)>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_get_stats"));
        int64_t values[16] = {};
        GGML_ASSERT(stats); stats(values);
        GGML_ASSERT(values[0] == 3 * replaced_layers && values[4] > 0 && values[6] == 0);
        fprintf(stderr, "probe: graph/fusion enabled, finite logits, packed=%lld regions=%lld fallback=%lld chunked=%lld\n",
            (long long) values[0], (long long) values[4], (long long) values[6], (long long) values[15]);
        return status;
    }
    return status || !state.done;
}
