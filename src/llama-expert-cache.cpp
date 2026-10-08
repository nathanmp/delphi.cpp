#include "llama-expert-cache.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <tuple>

const llama_expert_cache::entry * llama_expert_cache::layer(int il) const {
    return il >= 0 && il < (int) layers.size() && layers[il].gate ? &layers[il] : nullptr;
}

std::unique_ptr<llama_expert_cache> llama_expert_cache::create(
        const llama_model & model, const std::vector<ggml_backend_ptr> & backends) {
    const char * budget_env = getenv("LLAMA_MOE_CACHE_MIB");
    if (!budget_env || atoll(budget_env) <= 0) { return nullptr; }
    if (model.hparams.no_alloc) { return nullptr; }
    const char * path = getenv("LLAMA_MOE_CACHE_PROFILE");
    if (!path || !*path) { throw std::runtime_error("LLAMA_MOE_CACHE_PROFILE is required with LLAMA_MOE_CACHE_MIB"); }
    if (model.arch != LLM_ARCH_MIMO2) { throw std::runtime_error("static expert-cache prototype currently supports MiMo2 only"); }
    auto result = std::unique_ptr<llama_expert_cache>(new llama_expert_cache);
    result->layers.resize(model.layers.size());
    std::ifstream file(path);
    std::string magic;
    int h, f, experts, n_layers;
    if (!(file >> magic >> h >> f >> experts >> n_layers) || magic != "DELPHI_MOE_PROFILE_V1" ||
            n_layers != (int) model.layers.size()) {
        throw std::runtime_error("invalid expert-cache profile header");
    }
    struct candidate { int il, e; double count; size_t bytes; };
    std::vector<candidate> candidates;
    std::set<std::pair<int, int>> seen;
    int il, e;
    double count;
    while (file >> il >> e >> count) {
        if (il < 0 || il >= n_layers || e < 0 || e >= experts || !std::isfinite(count) || count <= 0 || !seen.emplace(il, e).second) {
            throw std::runtime_error("invalid or repeated expert-cache profile entry");
        }
        const auto & l = model.layers[il];
        const ggml_tensor * ws[] = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
        size_t bytes = 0;
        for (const auto * w : ws) {
            if (!w || w->type != GGML_TYPE_Q4_G128 || !ggml_is_contiguous(w) || w->ne[2] != experts ||
                    w->view_src || !w->buffer || !ggml_backend_buffer_is_host(w->buffer)) {
                throw std::runtime_error("expert-cache requires original host G128 gate/up/down tensors");
            }
            bytes += w->nb[2];
        }
        if (ws[0]->ne[0] != h || ws[0]->ne[1] != f || ws[1]->ne[0] != h || ws[1]->ne[1] != f ||
                ws[2]->ne[0] != f || ws[2]->ne[1] != h || h % 128 || f % 128) {
            throw std::runtime_error("expert-cache profile geometry differs from model");
        }
        for (const auto & b : backends) {
            if (ggml_backend_get_device(b.get()) == model.dev_layer(il)) {
                if (result->gpu && result->gpu != b.get()) { throw std::runtime_error("expert-cache requires one GPU"); }
                result->gpu = b.get();
            }
        }
        candidates.push_back({il, e, count, bytes});
    }
    if (!file.eof() || !result->gpu || candidates.empty() ||
            ggml_backend_dev_type(ggml_backend_get_device(result->gpu)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        throw std::runtime_error("invalid expert-cache profile or GPU placement");
    }
    // Frequency per byte. Cold misses never enter the resident set.
    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        const double av = a.count / a.bytes, bv = b.count / b.bytes;
        return av != bv ? av > bv : std::tie(a.il, a.e) < std::tie(b.il, b.e);
    });
    const size_t budget = size_t(atoll(budget_env)) * 1024 * 1024;
    size_t used = 0;
    std::vector<std::vector<int>> selected(n_layers);
    for (const auto & c : candidates) {
        if (c.bytes <= budget - used) { selected[c.il].push_back(c.e); used += c.bytes; }
    }
    if (!used) { throw std::runtime_error("expert-cache budget cannot fit an expert"); }
    result->ctx.reset(ggml_init({size_t(n_layers) * 5 * ggml_tensor_overhead(), nullptr, true}));
    const auto buft = ggml_backend_get_default_buffer_type(result->gpu);
    for (int i = 0; i < n_layers; ++i) {
        if (selected[i].empty()) { continue; }
        std::sort(selected[i].begin(), selected[i].end());
        const auto & l = model.layers[i];
        auto & dst = result->layers[i];
        ggml_tensor ** banks[] = { &dst.gate, &dst.up, &dst.down };
        const ggml_tensor * src[] = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
        for (int p = 0; p < 3; ++p) {
            *banks[p] = ggml_new_tensor_3d(result->ctx.get(), src[p]->type, src[p]->ne[0], src[p]->ne[1], selected[i].size());
            ggml_format_name(*banks[p], "cache_bank_%d_%d", i, p);
        }
        dst.hot_ids = ggml_new_tensor_2d(result->ctx.get(), GGML_TYPE_I32, 1, experts);
        dst.cold_ids = ggml_new_tensor_2d(result->ctx.get(), GGML_TYPE_I32, 1, experts);
    }
    result->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result->ctx.get(), buft));
    if (!result->buffer) { throw std::runtime_error("cannot allocate expert-cache banks"); }
    ggml_backend_buffer_set_usage(result->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    size_t n_resident = 0;
    std::vector<char> witness;
    for (int i = 0; i < n_layers; ++i) {
        auto & dst = result->layers[i];
        if (!dst.gate) { continue; }
        const auto & l = model.layers[i];
        const ggml_tensor * src[] = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
        ggml_tensor * banks[] = { dst.gate, dst.up, dst.down };
        std::vector<int32_t> hot(experts, -1), cold(experts);
        for (int j = 0; j < experts; ++j) { cold[j] = j; }
        for (size_t j = 0; j < selected[i].size(); ++j) {
            const int expert = selected[i][j];
            hot[expert] = j; cold[expert] = -1;
            for (int p = 0; p < 3; ++p) {
                ggml_backend_tensor_set(banks[p], static_cast<const char *>(src[p]->data) + expert * src[p]->nb[2],
                        j * banks[p]->nb[2], src[p]->nb[2]);
            }
        }
        // Check one slab in every bank before the backend's lossless in-place packing.
        for (int p = 0; p < 3; ++p) {
            witness.resize(src[p]->nb[2]);
            ggml_backend_tensor_get(banks[p], witness.data(), 0, witness.size());
            if (memcmp(witness.data(), static_cast<const char *>(src[p]->data) + selected[i][0] * src[p]->nb[2], witness.size())) {
                throw std::runtime_error("expert-cache uploaded weight mismatch");
            }
        }
        ggml_backend_tensor_set(dst.hot_ids, hot.data(), 0, hot.size() * sizeof(int32_t));
        ggml_backend_tensor_set(dst.cold_ids, cold.data(), 0, cold.size() * sizeof(int32_t));
        n_resident += selected[i].size();
    }
    LLAMA_LOG_INFO("%s: %zu static experts, %.2f MiB payload, %.2f MiB allocated on %s; no decode fills\n", __func__,
            n_resident, used / 1048576.0, ggml_backend_buffer_get_size(result->buffer.get()) / 1048576.0,
            ggml_backend_name(result->gpu));
    return result;
}
