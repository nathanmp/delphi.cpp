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
    return il >= 0 && il < (int) layers.size() && layers[il].hot_ids ? &layers[il] : nullptr;
}

std::unique_ptr<llama_expert_cache> llama_expert_cache::create(
        const llama_model & model, const std::vector<ggml_backend_ptr> & backends) {
    const char * budget_env = getenv("LLAMA_MOE_CACHE_MIB");
    if (!budget_env || atoll(budget_env) <= 0) { return nullptr; }
    if (model.hparams.no_alloc) { return nullptr; }
    if (model.arch != LLM_ARCH_MIMO2) { throw std::runtime_error("expert-cache prototype currently supports MiMo2 only"); }
    auto result = std::unique_ptr<llama_expert_cache>(new llama_expert_cache);
    result->model = &model;
    if (const char * env = getenv("LLAMA_MOE_CACHE_WINDOW")) { result->window = atoll(env); }
    if (const char * env = getenv("LLAMA_MOE_CACHE_PREFILL")) { result->record_prompts = atoi(env) != 0; }
    if (const char * env = getenv("LLAMA_MOE_CACHE_HYSTERESIS")) { result->hysteresis = std::max(0.0, atof(env)); }

    // Every routed layer must be original host G128 with one geometry, because all layers share the slot arena.
    const int n_layers = model.layers.size();
    const ggml_tensor * shape[3] = {};
    size_t expert_bytes = 0;
    int n_expert = 0;
    result->layers.resize(n_layers);
    for (int il = 0; il < n_layers; ++il) {
        const auto & l = model.layers[il];
        const ggml_tensor * ws[] = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
        if (!ws[0] && !ws[1] && !ws[2]) { continue; }
        size_t bytes = 0;
        for (int p = 0; p < 3; ++p) {
            const auto * w = ws[p];
            if (!w || w->type != GGML_TYPE_Q4_G128 || !ggml_is_contiguous(w) || w->view_src || !w->buffer ||
                    !ggml_backend_buffer_is_host(w->buffer) || w->ne[2] > 65535 ||
                    (shape[p] && (w->ne[0] != shape[p]->ne[0] || w->ne[1] != shape[p]->ne[1] || w->ne[2] != shape[p]->ne[2]))) {
                throw std::runtime_error("expert-cache requires original host G128 gate/up/down tensors of one geometry");
            }
            shape[p] = w;
            bytes += w->nb[2];
        }
        if (ws[0]->ne[0] % 128 || ws[0]->ne[1] % 128 || ws[0]->ne[0] != ws[1]->ne[0] || ws[0]->ne[1] != ws[1]->ne[1] ||
                ws[2]->ne[0] != ws[0]->ne[1] || ws[2]->ne[1] != ws[0]->ne[0]) {
            throw std::runtime_error("expert-cache gate/up/down geometry mismatch");
        }
        for (const auto & b : backends) {
            if (ggml_backend_get_device(b.get()) == model.dev_layer(il)) {
                if (result->gpu && result->gpu != b.get()) { throw std::runtime_error("expert-cache requires one GPU"); }
                result->gpu = b.get();
            }
        }
        expert_bytes = bytes;
        n_expert = ws[0]->ne[2];
        auto & dst = result->layers[il];
        dst.owner = result.get();
        dst.il = il;
        dst.cold.resize(n_expert);
        for (int e = 0; e < n_expert; ++e) { dst.cold[e] = e; }
        dst.slot.assign(n_expert, -1);
        result->stamp.assign(n_expert, 0);
        dst.count.assign(n_expert, 0);
        dst.prior.assign(n_expert, 0.0f);
        if (result->window > 0) { dst.ring.assign(size_t(result->window) * model.hparams.n_expert_used(il), 0); }
    }
    if (!expert_bytes || !result->gpu || ggml_backend_dev_type(ggml_backend_get_device(result->gpu)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        throw std::runtime_error("expert-cache requires routed experts and one GPU layer device");
    }
    const size_t budget = size_t(atoll(budget_env)) * 1024 * 1024;
    const int64_t n_slots = budget / expert_bytes;
    if (n_slots <= 0) { throw std::runtime_error("expert-cache budget cannot fit an expert"); }

    // Optional profile: the prior for rows that the window does not hold yet.
    if (const char * path = getenv("LLAMA_MOE_CACHE_PROFILE"); path && *path) {
        std::ifstream file(path);
        std::string magic;
        int h, f, experts, profile_layers;
        if (!(file >> magic >> h >> f >> experts >> profile_layers) || magic != "DELPHI_MOE_PROFILE_V1" ||
                profile_layers != n_layers || experts != n_expert || h != shape[0]->ne[0] || f != shape[0]->ne[1]) {
            throw std::runtime_error("invalid expert-cache profile header or geometry");
        }
        std::set<std::pair<int, int>> seen;
        int il, e;
        double count;
        while (file >> il >> e >> count) {
            if (il < 0 || il >= n_layers || !result->layers[il].owner || e < 0 || e >= experts || !std::isfinite(count) ||
                    count <= 0 || !seen.emplace(il, e).second) {
                throw std::runtime_error("invalid or repeated expert-cache profile entry");
            }
            result->layers[il].prior[e] = count;
        }
        if (!file.eof() || seen.empty()) { throw std::runtime_error("invalid expert-cache profile"); }
        for (int i = 0; i < n_layers; ++i) {
            auto & l = result->layers[i];
            double total = 0;
            for (float c : l.prior) { total += c; }
            const double rows = total / model.hparams.n_expert_used(i);
            for (float & c : l.prior) { c = rows > 0 ? c / rows : 0.0f; }
        }
    }

    result->ctx.reset(ggml_init({size_t(n_layers + 3) * ggml_tensor_overhead(), nullptr, true}));
    for (int p = 0; p < 3; ++p) {
        result->banks[p] = ggml_new_tensor_3d(result->ctx.get(), shape[p]->type, shape[p]->ne[0], shape[p]->ne[1], n_slots);
        ggml_format_name(result->banks[p], "cache_bank_%d", p);
    }
    for (auto & l : result->layers) {
        if (l.owner) { l.hot_ids = ggml_new_tensor_2d(result->ctx.get(), GGML_TYPE_I32, 1, n_expert); }
    }
    result->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result->ctx.get(), ggml_backend_get_default_buffer_type(result->gpu)));
    if (!result->buffer) { throw std::runtime_error("cannot allocate expert-cache arena"); }
    ggml_backend_buffer_set_usage(result->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    result->owner_of.assign(n_slots, {-1, -1});
    std::vector<int32_t> none(n_expert, -1);
    for (auto & l : result->layers) {
        if (l.owner) { ggml_backend_tensor_set(l.hot_ids, none.data(), 0, none.size() * sizeof(int32_t)); }
    }
    if (auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(result->gpu))) {
        result->set_expert = (decltype(result->set_expert)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_set_expert");
    }

    result->refresh();
    // Check one slab in every bank before the backend's lossless in-place packing.
    if (result->n_uploaded) {
        const auto [il, e] = result->owner_of[0];
        const auto & l = model.layers[il];
        const ggml_tensor * src[] = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
        std::vector<char> witness;
        for (int p = 0; p < 3; ++p) {
            witness.resize(src[p]->nb[2]);
            ggml_backend_tensor_get(result->banks[p], witness.data(), 0, witness.size());
            if (memcmp(witness.data(), static_cast<const char *>(src[p]->data) + e * src[p]->nb[2], witness.size())) {
                throw std::runtime_error("expert-cache uploaded weight mismatch");
            }
        }
    }
    LLAMA_LOG_WARN("%s: %lld slots, %.2f MiB allocated on %s; window %lld rows, prompts %s, hysteresis %.2f\n", __func__,
            (long long) n_slots, ggml_backend_buffer_get_size(result->buffer.get()) / 1048576.0, ggml_backend_name(result->gpu),
            (long long) result->window, result->record_prompts ? "on" : "off", result->hysteresis);
    return result;
}

llama_expert_cache::~llama_expert_cache() {
    if (n_refresh) {
        LLAMA_LOG_WARN("expert-cache: %lld refreshes, %lld experts uploaded, %.1f ms; decode hits since last refresh: %.1f%% selections, %.1f%% distinct\n",
                (long long) n_refresh, (long long) n_uploaded, refresh_ms, all_sel ? 100.0 * hit_sel / all_sel : 0.0,
                all_distinct ? 100.0 * hit_distinct / all_distinct : 0.0);
    }
}

void llama_expert_cache::record(entry & l, const char * data, size_t nb0, size_t nb1, int64_t k, int64_t rows, bool is_f32) {
    const int n_expert = l.count.size();
    const bool keep = window > 0;
    for (int64_t r = 0; r < rows; ++r) {
        if (keep && l.rows == window) {
            for (int64_t j = 0; j < k; ++j) { l.count[l.ring[l.head * k + j]]--; }
        } else {
            l.rows++;
        }
        for (int64_t j = 0; j < k; ++j) {
            const char * p = data + r * nb1 + j * nb0;
            const int32_t e = is_f32 ? (int32_t) *reinterpret_cast<const float *>(p) : *reinterpret_cast<const int32_t *>(p);
            GGML_ASSERT(e >= 0 && e < n_expert);
            l.count[e]++;
            if (keep) { l.ring[l.head * k + j] = e; }
        }
        if (keep) { l.head = (l.head + 1) % window; }
    }
}

void llama_expert_cache::op_cold_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) { return; }
    auto & l = *static_cast<entry *>(userdata);
    auto & c = *l.owner;
    const ggml_tensor * ids = dst->src[0];
    const int64_t k = ids->ne[0], rows = ids->ne[1];
    const bool warmup = c.warmup;
    if (!warmup && c.window >= 0) {
        c.record(l, (const char *) ids->data, ids->nb[0], ids->nb[1], k, rows, true);
    }
    // Hit statistics: selections, and distinct experts, because the CPU reads each expert once per ubatch.
    c.stamp_gen++;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < k; ++j) {
            const int32_t e = (int32_t) *reinterpret_cast<const float *>((const char *) ids->data + r * ids->nb[1] + j * ids->nb[0]);
            GGML_ASSERT(e >= 0 && e < (int32_t) l.cold.size());
            const int32_t v = l.cold[e];
            *reinterpret_cast<int32_t *>((char *) dst->data + r * dst->nb[1] + j * dst->nb[0]) = v;
            if (warmup) { continue; }
            c.all_sel++;
            c.hit_sel += v < 0;
            if (c.stamp[e] != c.stamp_gen) {
                c.stamp[e] = c.stamp_gen;
                c.all_distinct++;
                c.hit_distinct += v < 0;
            }
        }
    }
}

void llama_expert_cache::op_record(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) { return; }
    auto & l = *static_cast<entry *>(userdata);
    if (l.owner->warmup) { return; }
    const ggml_tensor * ids = dst->src[0];
    l.owner->record(l, (const char *) ids->data, ids->nb[0], ids->nb[1], ids->ne[0], ids->ne[1], false);
}

ggml_tensor * llama_expert_cache::build_cold_ids(ggml_context * ctx, ggml_tensor * ids, int il) {
    GGML_ASSERT(ids->type == GGML_TYPE_F32 && layer(il));
    ggml_tensor * args[] = { ids };
    return ggml_custom_4d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1], 1, 1, args, 1, op_cold_ids, 1, &layers[il]);
}

ggml_tensor * llama_expert_cache::build_record(ggml_context * ctx, ggml_tensor * ids, int il) {
    if (window < 0 || !record_prompts || !layer(il)) { return nullptr; }
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1);
    ggml_tensor * args[] = { ids };
    return ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, args, 1, op_record, 1, &layers[il]);
}

void llama_expert_cache::before_batch(ggml_backend_sched_t sched, int64_t n_tokens, bool is_warmup) {
    // Graph reuse ignores warmup, so the ops read this flag at execution time.
    warmup = is_warmup;
    if (warmup) { return; }
    if (n_tokens > max_rows) { pending = true; return; }
    if (!pending) { return; }
    pending = false;
    if (window < 0) {
        LLAMA_LOG_WARN("expert-cache: static; previous decode hits %.1f%% selections, %.1f%% distinct\n",
                all_sel ? 100.0 * hit_sel / all_sel : 0.0, all_distinct ? 100.0 * hit_distinct / all_distinct : 0.0);
        hit_sel = all_sel = hit_distinct = all_distinct = 0;
        return;
    }
    // No queued GPU work may read a slot that this refresh overwrites.
    ggml_backend_sched_synchronize(sched);
    refresh();
}

void llama_expert_cache::upload(int il, int e, int s) {
    const auto & l = model->layers[il];
    const ggml_tensor * src[] = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
    for (int p = 0; p < 3; ++p) {
        const char * data = static_cast<const char *>(src[p]->data) + e * src[p]->nb[2];
        if (!set_expert || !set_expert(banks[p], s, data)) {
            ggml_backend_tensor_set(banks[p], data, s * banks[p]->nb[2], banks[p]->nb[2]);
        }
    }
}

// Global top-N of (layer, expert) by selections in the most recent window rows. Rows that the window does not hold yet are
// filled from the profile prior. Resident experts keep a hysteresis bonus, because a replacement costs a full PCIe upload.
void llama_expert_cache::refresh() {
    const int64_t t0 = ggml_time_us();
    const int64_t prior_rows = window > 0 ? window : 2048;
    struct candidate { double score; int il, e; };
    std::vector<candidate> candidates;
    double total = 0;
    for (auto & l : layers) {
        if (!l.owner) { continue; }
        const double fill = std::max<int64_t>(0, prior_rows - l.rows);
        for (int e = 0; e < (int) l.count.size(); ++e) {
            const double score = l.count[e] + fill * l.prior[e];
            total += score;
            if (score > 0) { candidates.push_back({score * (l.slot[e] >= 0 ? 1.0 + hysteresis : 1.0), l.il, e}); }
        }
    }
    const size_t n_slots = owner_of.size();
    const size_t n_keep = std::min(n_slots, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + n_keep, candidates.end(), [](const candidate & a, const candidate & b) {
        return a.score != b.score ? a.score > b.score : std::tie(a.il, a.e) < std::tie(b.il, b.e);
    });
    std::vector<std::vector<bool>> keep(layers.size());
    double covered = 0;
    for (size_t i = 0; i < n_keep; ++i) {
        const auto & c = candidates[i];
        auto & k = keep[c.il];
        if (k.empty()) { k.assign(layers[c.il].count.size(), false); }
        k[c.e] = true;
        covered += c.score / (layers[c.il].slot[c.e] >= 0 ? 1.0 + hysteresis : 1.0);
    }
    std::vector<int> free_slots;
    std::vector<bool> changed(layers.size(), false);
    for (size_t s = 0; s < n_slots; ++s) {
        const auto [il, e] = owner_of[s];
        if (il >= 0 && (keep[il].empty() || !keep[il][e])) {
            layers[il].slot[e] = -1;
            layers[il].cold[e] = e;
            owner_of[s] = {-1, -1};
            changed[il] = true;
        }
        if (owner_of[s].first < 0) { free_slots.push_back(s); }
    }
    std::reverse(free_slots.begin(), free_slots.end());
    int64_t uploaded = 0;
    for (size_t i = 0; i < n_keep; ++i) {
        const auto & c = candidates[i];
        auto & l = layers[c.il];
        if (l.slot[c.e] >= 0) { continue; }
        const int s = free_slots.back();
        free_slots.pop_back();
        upload(c.il, c.e, s);
        l.slot[c.e] = s;
        l.cold[c.e] = -1;
        owner_of[s] = {c.il, c.e};
        changed[c.il] = true;
        uploaded++;
    }
    for (auto & l : layers) {
        if (l.owner && changed[l.il]) { ggml_backend_tensor_set(l.hot_ids, l.slot.data(), 0, l.slot.size() * sizeof(int32_t)); }
    }
    const double ms = (ggml_time_us() - t0) / 1000.0;
    LLAMA_LOG_WARN("expert-cache: refresh %lld: %zu resident, %lld uploaded in %.1f ms, window covers %.1f%%; previous decode hits %.1f%% selections, %.1f%% distinct\n",
            (long long) n_refresh, n_keep, (long long) uploaded, ms, total > 0 ? 100.0 * covered / total : 0.0,
            all_sel ? 100.0 * hit_sel / all_sel : 0.0, all_distinct ? 100.0 * hit_distinct / all_distinct : 0.0);
    n_refresh++;
    n_uploaded += uploaded;
    refresh_ms += ms;
    hit_sel = all_sel = hit_distinct = all_distinct = 0;
}
