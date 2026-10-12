#include "llama-expert-cache.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <tuple>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

// gate, up and down slabs may differ in size (types per projection and group); buffers that hold slabs of any bank use the
// largest as their stride
size_t llama_expert_cache::max_slab() const {
    size_t m = 0;
    for (const auto & types : group_types) {
        for (int p = 0; p < 3; ++p) { m = std::max(m, ggml_row_size(types[p], shape[p]->ne[0]) * shape[p]->ne[1]); }
    }
    return m;
}

bool llama_expert_cache::any_lent() const {
    for (const auto & t : tiers) {
        for (const auto & g : t.groups) { if (g.lent_lo != g.n) { return true; } }
    }
    return false;
}

const llama_expert_cache::entry * llama_expert_cache::layer(int il) const {
    return il >= 0 && il < (int) layers.size() && layers[il].owner ? &layers[il] : nullptr;
}

ggml_tensor * llama_expert_cache::hot_ids(int t, int il) const {
    const auto & l = layers[il];
    return l.owner && tiers[t].groups[l.group].n > 0 ? tiers[t].hot_ids[il] : nullptr;
}

// Global slots run over the tiers in order, then over their groups; refresh() fills the lowest free slots first, so the best
// experts go to the first tiers.
void llama_expert_cache::renumber() {
    int64_t n = 0;
    where.clear();
    for (size_t t = 0; t < tiers.size(); ++t) {
        for (size_t g = 0; g < tiers[t].groups.size(); ++g) {
            auto & grp = tiers[t].groups[g];
            grp.base = n;
            n += grp.n;
            where.resize(n, { int(t), int(g) });
        }
    }
    owner_of.resize(n, { -1, -1 });
}

bool llama_expert_cache::make_tier(tier & t, ggml_backend_t backend, size_t bytes) {
    // the hot experts run as MUL_MAT_ID on the GPU banks
    ggml_context_ptr probe(ggml_init({12 * group_types.size() * ggml_tensor_overhead(), nullptr, true}));
    for (const auto & types : group_types) {
        for (int p = 0; p < 3; ++p) {
            ggml_tensor * a   = ggml_new_tensor_3d(probe.get(), types[p], shape[p]->ne[0], shape[p]->ne[1], 2);
            ggml_tensor * b   = ggml_new_tensor_3d(probe.get(), GGML_TYPE_F32, shape[p]->ne[0], 1, 1);
            ggml_tensor * ids = ggml_new_tensor_2d(probe.get(), GGML_TYPE_I32, 1, 1);
            if (!ggml_backend_supports_op(backend, ggml_mul_mat_id(probe.get(), a, b, ids))) {
                LLAMA_LOG_WARN("expert-cache: %s cannot run %s experts\n", ggml_backend_name(backend), ggml_type_name(types[p]));
                return false;
            }
        }
    }
    // The same number of slots for every layer: a group's share of the budget follows its layers and its expert size.
    const int64_t per_layer = bytes / all_layer_bytes;
    if (per_layer <= 0) { return false; }
    // the budget left after per_layer slots in every layer goes to the groups in order (all of it to a single group)
    size_t left = bytes - per_layer * all_layer_bytes;
    t.backend = backend;
    t.groups.assign(group_types.size(), {});
    t.ctx.reset(ggml_init({size_t(layers.size() + 3 * group_types.size()) * ggml_tensor_overhead(), nullptr, true}));
    for (size_t g = 0; g < group_types.size(); ++g) {
        auto & grp = t.groups[g];
        grp.n = per_layer * group_layers[g] + left / group_bytes[g];
        left -= (left / group_bytes[g]) * group_bytes[g];
        grp.lent_lo = grp.n;
        for (int p = 0; p < 3; ++p) {
            grp.banks[p] = ggml_new_tensor_3d(t.ctx.get(), group_types[g][p], shape[p]->ne[0], shape[p]->ne[1], grp.n);
            if (t.id == 0) {
                ggml_format_name(grp.banks[p], "cache_bank_%zu_%d", g, p);
            } else {
                ggml_format_name(grp.banks[p], "cache_t%d_bank_%zu_%d", t.id, g, p);
            }
        }
    }
    t.hot_ids.assign(layers.size(), nullptr);
    for (auto & l : layers) {
        if (l.owner) { t.hot_ids[l.il] = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_I32, 1, n_expert); }
    }
    t.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(t.ctx.get(), ggml_backend_get_default_buffer_type(backend)));
    if (!t.buffer) { return false; }
    ggml_backend_buffer_set_usage(t.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<int32_t> none(n_expert, -1);
    for (auto * h : t.hot_ids) {
        if (h) { ggml_backend_tensor_set(h, none.data(), 0, none.size() * sizeof(int32_t)); }
    }
    return true;
}

std::unique_ptr<llama_expert_cache> llama_expert_cache::create(
        const llama_model & model, const std::vector<ggml_backend_ptr> & backends) {
    // LLAMA_MOE_CACHE_SHARED=1: a cache without a static tier, for tiers leased at run time (VRAM sharing)
    const char * budget_env = getenv("LLAMA_MOE_CACHE_MIB");
    const char * shared_env = getenv("LLAMA_MOE_CACHE_SHARED");
    const long long budget_mib = budget_env ? atoll(budget_env) : 0;
    const bool shared = shared_env && atoi(shared_env) != 0;
    if (budget_mib <= 0 && !shared) { return nullptr; }
    if (model.hparams.no_alloc) { return nullptr; }
    auto result = std::unique_ptr<llama_expert_cache>(new llama_expert_cache);
    result->model = &model;
    if (const char * env = getenv("LLAMA_MOE_CACHE_WINDOW")) { result->window = atoll(env); }
    if (const char * env = getenv("LLAMA_MOE_CACHE_PREFILL")) { result->record_prompts = atoi(env) != 0; }
    if (const char * env = getenv("LLAMA_MOE_CACHE_PROMPT_ROWS")) { result->prompt_rows = std::max(0LL, atoll(env)); }
    if (const char * env = getenv("LLAMA_MOE_CACHE_HYSTERESIS")) { result->hysteresis = std::max(0.0, atof(env)); }
    if (const char * env = getenv("LLAMA_MOE_CACHE_EVERY")) { result->every = std::max(0LL, atoll(env)); }
    if (const char * env = getenv("LLAMA_MOE_CACHE_ASYNC_EXPERTS")) { result->max_async = std::max(1LL, atoll(env)); }
    if (const char * path = getenv("LLAMA_MOE_CACHE_DUMP"); path && *path) {
        result->dump = fopen(path, "wb");
        if (!result->dump) { throw std::runtime_error("cannot open LLAMA_MOE_CACHE_DUMP"); }
        fwrite("DELPHI_MOE_DUMP1", 1, 16, result->dump);
    }

    // Every routed layer must have original host gate/up/down experts of one geometry per projection. Layers with the same expert
    // types share a slot arena (group), so a per-layer quantization mix gets one arena per combination of types. The graph uses
    // the cache only for a SiLU FFN without expert biases or scales (build_moe_ffn).
    const int n_layers = model.layers.size();
    auto & c = *result;
    c.layers.resize(n_layers);
    for (int il = 0; il < n_layers; ++il) {
        const auto & l = model.layers[il];
        const ggml_tensor * ws[] = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
        if (!ws[0] && !ws[1] && !ws[2]) { continue; }
        if (l.ffn_gate_up_exps || l.ffn_gate_exps_b || l.ffn_up_exps_b || l.ffn_down_exps_b || l.ffn_gate_up_exps_b ||
                l.ffn_gate_exps_s || l.ffn_up_exps_s || l.ffn_down_exps_s) {
            throw std::runtime_error("expert-cache requires separate gate/up/down experts without biases or scales");
        }
        size_t bytes = 0;
        for (int p = 0; p < 3; ++p) {
            const auto * w = ws[p];
            if (!w || !ggml_is_contiguous(w) || w->view_src || !w->buffer || !ggml_backend_buffer_is_host(w->buffer) || w->ne[2] > 65535 ||
                    (c.shape[p] && (w->ne[0] != c.shape[p]->ne[0] || w->ne[1] != c.shape[p]->ne[1] || w->ne[2] != c.shape[p]->ne[2]))) {
                throw std::runtime_error("expert-cache requires original host gate/up/down tensors of one geometry per projection");
            }
            c.shape[p] = w;
            bytes += w->nb[2];
        }
        if ((ws[0]->type == GGML_TYPE_Q4_G128 && (ws[0]->ne[0] % 128 || ws[0]->ne[1] % 128)) ||
                ws[0]->ne[0] != ws[1]->ne[0] || ws[0]->ne[1] != ws[1]->ne[1] ||
                ws[2]->ne[0] != ws[0]->ne[1] || ws[2]->ne[1] != ws[0]->ne[0]) {
            throw std::runtime_error("expert-cache gate/up/down geometry mismatch");
        }
        for (const auto & b : backends) {
            if (ggml_backend_get_device(b.get()) == model.dev_layer(il)) {
                if (c.gpu && c.gpu != b.get()) { throw std::runtime_error("expert-cache requires one GPU"); }
                c.gpu = b.get();
            }
        }
        const std::array<ggml_type, 3> types = { ws[0]->type, ws[1]->type, ws[2]->type };
        int g = std::find(c.group_types.begin(), c.group_types.end(), types) - c.group_types.begin();
        if (g == (int) c.group_types.size()) {
            c.group_types.push_back(types);
            c.group_layers.push_back(0);
            c.group_bytes.push_back(bytes);
        }
        c.group_layers[g]++;
        c.all_layer_bytes += bytes;
        c.n_expert = ws[0]->ne[2];
        auto & dst = c.layers[il];
        dst.owner = result.get();
        dst.il = il;
        dst.group = g;
        dst.cold.resize(c.n_expert);
        for (int e = 0; e < c.n_expert; ++e) { dst.cold[e] = e; }
        dst.slot.assign(c.n_expert, -1);
        c.stamp.assign(c.n_expert, 0);
        dst.count.assign(c.n_expert, 0);
        dst.prior.assign(c.n_expert, 0.0f);
        if (c.window > 0) { dst.ring.assign(size_t(c.window) * model.hparams.n_expert_used(il), 0); }
    }
    if (!c.all_layer_bytes || !c.gpu || ggml_backend_dev_type(ggml_backend_get_device(c.gpu)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        throw std::runtime_error("expert-cache requires routed experts and one GPU layer device");
    }

    // Optional profile: the prior for rows that the window does not hold yet.
    if (const char * path = getenv("LLAMA_MOE_CACHE_PROFILE"); path && *path) {
        std::ifstream file(path);
        std::string magic;
        int h, f, experts, profile_layers;
        if (!(file >> magic >> h >> f >> experts >> profile_layers) || magic != "DELPHI_MOE_PROFILE_V1" ||
                profile_layers != n_layers || experts != c.n_expert || h != c.shape[0]->ne[0] || f != c.shape[0]->ne[1]) {
            throw std::runtime_error("invalid expert-cache profile header or geometry");
        }
        std::set<std::pair<int, int>> seen;
        int il, e;
        double count;
        while (file >> il >> e >> count) {
            if (il < 0 || il >= n_layers || !c.layers[il].owner || e < 0 || e >= experts || !std::isfinite(count) ||
                    count <= 0 || !seen.emplace(il, e).second) {
                throw std::runtime_error("invalid or repeated expert-cache profile entry");
            }
            c.layers[il].prior[e] = count;
        }
        if (!file.eof() || seen.empty()) { throw std::runtime_error("invalid expert-cache profile"); }
        for (int i = 0; i < n_layers; ++i) {
            auto & l = c.layers[i];
            double total = 0;
            for (float x : l.prior) { total += x; }
            const double rows = total / model.hparams.n_expert_used(i);
            for (float & x : l.prior) { x = rows > 0 ? x / rows : 0.0f; }
        }
    }

    if (auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(c.gpu))) {
        c.set_expert = (decltype(c.set_expert)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_set_expert");
        c.forget_inside = (decltype(c.forget_inside)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_forget_inside");
    }

    std::string desc;
    for (size_t g = 0; g < c.group_types.size(); ++g) {
        desc += format("%s%s/%s/%s x %lld layers", g ? ", " : "", ggml_type_name(c.group_types[g][0]), ggml_type_name(c.group_types[g][1]),
                ggml_type_name(c.group_types[g][2]), (long long) c.group_layers[g]);
    }
    if (budget_mib <= 0) {
        LLAMA_LOG_WARN("%s: no static tier (LLAMA_MOE_CACHE_SHARED); %s; window %lld rows, prompts %s, hysteresis %.2f, decode-time refresh every %lld small batches (up to %lld experts)\n", __func__,
                desc.c_str(), (long long) c.window, c.record_prompts ? "on" : "off", c.hysteresis, (long long) c.every, (long long) c.max_async);
        return result;
    }

    const size_t budget = size_t(budget_mib) * 1024 * 1024;
    if (budget / c.all_layer_bytes <= 0) { throw std::runtime_error("expert-cache budget cannot fit an expert of every layer"); }
    auto & t0 = c.tiers.emplace_back();
    t0.id = c.next_tier_id++;
    if (!c.make_tier(t0, c.gpu, budget)) { throw std::runtime_error("cannot allocate expert-cache arena"); }
    c.renumber();
    c.gen++;

    c.refresh();
    // Check one slab in every bank before the backend's lossless in-place packing.
    for (const auto & grp : t0.groups) {
        const auto [il, e] = c.owner_of[grp.base];
        if (il < 0) { continue; }
        const auto & l = model.layers[il];
        const ggml_tensor * src[] = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
        std::vector<char> witness;
        for (int p = 0; p < 3; ++p) {
            witness.resize(src[p]->nb[2]);
            ggml_backend_tensor_get(grp.banks[p], witness.data(), 0, witness.size());
            if (memcmp(witness.data(), static_cast<const char *>(src[p]->data) + e * src[p]->nb[2], witness.size())) {
                throw std::runtime_error("expert-cache uploaded weight mismatch");
            }
        }
    }
    int64_t n_slots = 0;
    for (const auto & grp : t0.groups) { n_slots += grp.n; }
    LLAMA_LOG_WARN("%s: %lld slots (%lld per layer; %s), %.2f MiB allocated on %s; window %lld rows, prompts %s, hysteresis %.2f, decode-time refresh every %lld small batches (up to %lld experts)\n", __func__,
            (long long) n_slots, (long long) (budget / c.all_layer_bytes), desc.c_str(), ggml_backend_buffer_get_size(t0.buffer.get()) / 1048576.0,
            ggml_backend_name(c.gpu), (long long) c.window, c.record_prompts ? "on" : "off", c.hysteresis,
            (long long) c.every, (long long) c.max_async);
    return result;
}

int llama_expert_cache::add_tier(ggml_backend_t backend, size_t bytes) {
    if ((int) tiers.size() >= max_tiers) {
        LLAMA_LOG_WARN("expert-cache: cannot add a tier: %d tiers already\n", max_tiers);
        return -1;
    }
    tier t;
    t.id = next_tier_id;
    if (!make_tier(t, backend, bytes)) {
        LLAMA_LOG_WARN("expert-cache: cannot add a tier of %.1f MiB on %s\n", bytes / 1048576.0, ggml_backend_name(backend));
        if (t.buffer) { forget(t); }
        return -1;
    }
    next_tier_id++;
    const size_t size = ggml_backend_buffer_get_size(t.buffer.get());
    int64_t n = 0;
    for (const auto & g : t.groups) { n += g.n; }
    tiers.push_back(std::move(t));
    renumber(); // appended: the slots of the other tiers keep their numbers
    gen++;
    pending = true; // filled at the next small batch
    LLAMA_LOG_WARN("expert-cache: tier %d: %lld slots, %.1f MiB on %s (%zu tiers)\n", tiers.back().id, (long long) n,
            size / 1048576.0, ggml_backend_name(backend), tiers.size());
    return tiers.back().id;
}

void llama_expert_cache::forget(const tier & t) const {
    if (!forget_inside) { return; }
    for (const auto & g : t.groups) {
        for (auto * b : g.banks) { if (b && b->data) { forget_inside(t.backend, b); } }
    }
}

size_t llama_expert_cache::drop_tier(int id) {
    const auto it = std::find_if(tiers.begin(), tiers.end(), [&](const tier & t) { return t.id == id; });
    if (it == tiers.end()) { return 0; }
    // a decode-time refresh may target its slots
    cancel_async();
    ggml_backend_synchronize(gpu);
    ggml_backend_synchronize(it->backend);
    int64_t evicted = 0;
    for (const auto & g : it->groups) {
        for (int64_t s = g.base; s < g.base + g.n; ++s) {
            const auto [il, e] = owner_of[s];
            if (il < 0) { continue; }
            layers[il].slot[e] = -1;
            layers[il].cold[e] = e;
            evicted++;
        }
    }
    forget(*it);
    // The later tiers move down; their maps hold local slots and do not change.
    std::vector<std::vector<int64_t>> old_base(tiers.size());
    for (size_t t = 0; t < tiers.size(); ++t) {
        for (const auto & g : tiers[t].groups) { old_base[t].push_back(g.base); }
    }
    const size_t ti = it - tiers.begin();
    const size_t bytes = ggml_backend_buffer_get_size(it->buffer.get());
    const auto old_owner = owner_of;
    tiers.erase(it);
    old_base.erase(old_base.begin() + ti);
    owner_of.clear();
    renumber();
    for (size_t t = 0; t < tiers.size(); ++t) {
        for (size_t g = 0; g < tiers[t].groups.size(); ++g) {
            const auto & grp = tiers[t].groups[g];
            for (int64_t k = 0; k < grp.n; ++k) {
                const auto owner = old_owner[old_base[t][g] + k];
                owner_of[grp.base + k] = owner;
                if (owner.first >= 0) { layers[owner.first].slot[owner.second] = grp.base + k; }
            }
        }
    }
    gen++;
    LLAMA_LOG_WARN("expert-cache: dropped tier %d: %.1f MiB, %lld experts evicted (%zu tiers left)\n", id, bytes / 1048576.0,
            (long long) evicted, tiers.size());
    return bytes;
}

std::vector<std::pair<int, ggml_backend_t>> llama_expert_cache::tier_list() const {
    std::vector<std::pair<int, ggml_backend_t>> res;
    for (const auto & t : tiers) { res.push_back({ t.id, t.backend }); }
    return res;
}

size_t llama_expert_cache::tier_bytes(int id) const {
    for (const auto & t : tiers) {
        if (t.id == id) { return ggml_backend_buffer_get_size(t.buffer.get()); }
    }
    return 0;
}

// Decode-time refresh: a worker copies the incoming experts' slabs through a pinned buffer to a device scratch on a backend
// of its own (a second stream on each tier's GPU), so the copy overlaps decoding; tick() packs them into arena slots once done.
struct llama_expert_cache::async_refresh {
    struct device {
        ggml_backend_t backend = nullptr; // the tier backend, which packs from the scratch
        ggml_backend_ptr copy;
        ggml_context_ptr ctx;
        ggml_tensor * scratch = nullptr;
        ggml_backend_buffer_ptr scratch_buf;
    };
    std::vector<device> devices;
    ggml_backend_buffer_ptr host;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<const char *> srcs;          // slab sources, 3 per incoming expert (gate, up, down)
    std::vector<size_t> sizes;               // their sizes
    std::vector<upload_job> jobs;            // incoming experts and the slots they take, contiguous per device
    std::vector<int> dev;                    // per job, its device
    std::vector<int> pos;                    // per job, its index in the device scratch
    std::vector<std::pair<int, int>> evict;  // per job, the expert leaving that slot, or (-1, -1)
    bool queued = false, done = false, stop = false;
    int64_t copy_us = 0;

    ~async_refresh() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) { worker.join(); }
    }
};

llama_expert_cache::~llama_expert_cache() {
    async.reset();
    if (dump) { fclose(dump); }
    if (n_refresh) {
        LLAMA_LOG_WARN("expert-cache: %lld refreshes, %lld experts uploaded, %.1f ms; %lld decode-time refreshes, %lld experts, %.1f ms applying; decode hits since last refresh: %.1f%% selections, %.1f%% distinct\n",
                (long long) n_refresh, (long long) n_uploaded, refresh_ms, (long long) n_async, (long long) n_async_uploaded,
                async_apply_ms, all_sel ? 100.0 * hit_sel / all_sel : 0.0, all_distinct ? 100.0 * hit_distinct / all_distinct : 0.0);
    }
    if (prompt_all_sel) {
        LLAMA_LOG_WARN("expert-cache: prompt batches of %lld+ rows: %.1f%% of %lld selections on the GPU\n",
                (long long) max_rows + 1, 100.0 * prompt_hit_sel / prompt_all_sel, (long long) prompt_all_sel);
    }
}

size_t llama_expert_cache::release_scratch() {
    cancel_async();
    size_t bytes = 0;
    if (async) {
        for (const auto & d : async->devices) { bytes += d.scratch_buf ? ggml_backend_buffer_get_size(d.scratch_buf.get()) : 0; }
    }
    async.reset();
    pinned.reset();
    return bytes;
}

void llama_expert_cache::reclaimed() {
    for (auto & t : tiers) {
        for (auto & g : t.groups) {
            if (g.lent_lo == g.n) { continue; }
            // Their pointers inside the banks had their own weight layouts.
            for (auto * b : g.banks) { if (forget_inside) { forget_inside(t.backend, b); } }
            g.lent_lo = g.n;
        }
    }
}

void llama_expert_cache::record(entry & l, const char * data, size_t nb0, size_t nb1, int64_t k, int64_t rows, bool is_f32) {
    const int n_exp = l.count.size();
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
            GGML_ASSERT(e >= 0 && e < n_exp);
            l.count[e]++;
            if (keep) { l.ring[l.head * k + j] = e; }
        }
        if (keep) { l.head = (l.head + 1) % window; }
    }
}

// Record: u32 batch, u16 layer, u16 rows, u8 top_k, u8 kind (0 small batch, 1 prompt batch, 2 refresh, 3 batch start with
// rows = min(n_tokens, 65535)), u16 0, then rows * top_k u8 expert ids.
void llama_expert_cache::dump_record(int il, int kind, int64_t rows, int64_t k, const char * data, size_t nb0, size_t nb1, bool is_f32) {
    const uint16_t head[5] = { uint16_t(il), uint16_t(std::min<int64_t>(rows, 65535)), uint16_t(k | (kind << 8)), 0, 0 };
    fwrite(&batch_seq, sizeof(batch_seq), 1, dump);
    fwrite(head, sizeof(uint16_t), 4, dump);
    for (int64_t r = 0; data && r < rows; ++r) {
        uint8_t ids[256];
        for (int64_t j = 0; j < k; ++j) {
            const char * p = data + r * nb1 + j * nb0;
            ids[j] = uint8_t(is_f32 ? (int32_t) *reinterpret_cast<const float *>(p) : *reinterpret_cast<const int32_t *>(p));
        }
        fwrite(ids, 1, k, dump);
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
    // a prompt batch (graph_rows()) is recorded as op_record does, and its hits are counted apart from the decode hits
    const bool prompt = rows > max_rows;
    if (!warmup && c.window >= 0 && (!prompt || c.record_prompts)) {
        c.record(l, (const char *) ids->data, ids->nb[0], ids->nb[1], k, rows, true);
    }
    if (!warmup && c.dump) {
        c.dump_record(l.il, prompt ? 1 : 0, rows, k, (const char *) ids->data, ids->nb[0], ids->nb[1], true);
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
            if (prompt) {
                c.prompt_all_sel++;
                c.prompt_hit_sel += v < 0;
                continue;
            }
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
    if (l.owner->window >= 0 && l.owner->record_prompts) {
        l.owner->record(l, (const char *) ids->data, ids->nb[0], ids->nb[1], ids->ne[0], ids->ne[1], false);
    }
    if (l.owner->dump) {
        l.owner->dump_record(l.il, 1, ids->ne[1], ids->ne[0], (const char *) ids->data, ids->nb[0], ids->nb[1], false);
    }
}

ggml_tensor * llama_expert_cache::build_cold_ids(ggml_context * ctx, ggml_tensor * ids, int il) {
    GGML_ASSERT(ids->type == GGML_TYPE_F32 && layer(il));
    ggml_tensor * args[] = { ids };
    return ggml_custom_4d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1], 1, 1, args, 1, op_cold_ids, 1, &layers[il]);
}

ggml_tensor * llama_expert_cache::build_record(ggml_context * ctx, ggml_tensor * ids, int il) {
    if (((window < 0 || !record_prompts) && !dump) || !layer(il)) { return nullptr; }
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1);
    ggml_tensor * args[] = { ids };
    return ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, args, 1, op_record, 1, &layers[il]);
}

bool llama_expert_cache::needs_refresh(int64_t n_tokens, bool is_warmup) {
    // Graph reuse ignores warmup, so the ops read this flag at execution time.
    warmup = is_warmup;
    if (warmup) { return false; }
    if (dump) {
        batch_seq++;
        dump_record(0, 3, n_tokens, 0, nullptr, 0, 0, false);
    }
    if (n_tokens > max_rows) { pending = true; return false; }
    if (!active()) { return false; }
    if (!pending) {
        tick();
        return false;
    }
    pending = false;
    if (window < 0 && !any_lent() && n_refresh > 0) {
        LLAMA_LOG_WARN("expert-cache: static; previous decode hits %.1f%% selections, %.1f%% distinct\n",
                all_sel ? 100.0 * hit_sel / all_sel : 0.0, all_distinct ? 100.0 * hit_distinct / all_distinct : 0.0);
        hit_sel = all_sel = hit_distinct = all_distinct = 0;
        return false;
    }
    return true;
}

bool llama_expert_cache::lend(void * user_data, ggml_backend_t backend, int n_slots, size_t slot_size, void ** slots) {
    auto & c = *static_cast<llama_expert_cache *>(user_data);
    if (n_slots <= 0) { return false; }
    // A group of the tiers on this backend lends the top of its banks: the one that evicts the fewest experts (a tier leased for
    // this prompt is still empty), then the largest. Ring slot j uses the top of bank j % 3, so lending evicts the same slots in
    // all three banks; a ring slot spans the slots of the smallest of them.
    const int64_t per_bank = (n_slots + 2) / 3;
    auto top = [&](const group & g) {
        const size_t slab = std::min({ g.banks[0]->nb[2], g.banks[1]->nb[2], g.banks[2]->nb[2] });
        return g.n - per_bank * int64_t((slot_size + slab - 1) / slab);
    };
    auto resident_above = [&](const group & g, int64_t lo) {
        int64_t n = 0;
        for (int64_t s = g.base + std::max<int64_t>(lo, 0); s < g.base + g.lent_lo; ++s) { n += c.owner_of[s].first >= 0; }
        return n;
    };
    tier * lt = nullptr;
    group * lg = nullptr;
    int64_t best_evict = 0;
    for (auto & t : c.tiers) {
        if (t.backend != backend) { continue; }
        for (auto & g : t.groups) {
            if (top(g) <= 0) { continue; }
            const int64_t evict = resident_above(g, top(g));
            if (!lg || evict < best_evict || (evict == best_evict && g.n * g.banks[0]->nb[2] > lg->n * lg->banks[0]->nb[2])) {
                lt = &t; lg = &g; best_evict = evict;
            }
        }
    }
    if (!lg) {
        bool any = false;
        for (const auto & t : c.tiers) { any |= t.backend == backend; }
        if (any) {
            LLAMA_LOG_WARN("expert-cache: cannot lend %d staging slots of %.1f MiB: the tiers on %s are too small\n", n_slots,
                    slot_size / 1048576.0, ggml_backend_name(backend));
        }
        return false;
    }
    c.cancel_async();
    auto & g = *lg;
    const size_t slab = std::min({ g.banks[0]->nb[2], g.banks[1]->nb[2], g.banks[2]->nb[2] });
    const int64_t per_slot = (slot_size + slab - 1) / slab;
    const int64_t lo = top(g); // > 0: a lent slot at the bank base would share the bank's layout state
    int64_t evicted = 0;
    if (lo < g.lent_lo) {
        // Decode graphs read the arena; no queued work may read a slot that the ring now writes.
        ggml_backend_synchronize(c.gpu);
        ggml_backend_synchronize(lt->backend);
        std::vector<bool> changed(c.layers.size(), false);
        for (int64_t s = g.base + lo; s < g.base + g.lent_lo; ++s) {
            const auto [il, e] = c.owner_of[s];
            if (il < 0) { continue; }
            c.layers[il].slot[e] = -1;
            c.layers[il].cold[e] = e;
            c.owner_of[s] = {-1, -1};
            changed[il] = true;
            evicted++;
        }
        c.push_maps(changed);
        g.lent_lo = lo;
    }
    for (int j = 0; j < n_slots; ++j) {
        slots[j] = static_cast<char *>(g.banks[j % 3]->data) + (g.n - (j / 3 + 1) * per_slot) * g.banks[j % 3]->nb[2];
    }
    LLAMA_LOG_WARN("expert-cache: lent %lld of %lld slots of tier %d on %s as %d staging slots of %.1f MiB, evicted %lld experts\n",
            (long long) (g.n - lo), (long long) g.n, lt->id, ggml_backend_name(backend), n_slots, slot_size / 1048576.0, (long long) evicted);
    return true;
}

void llama_expert_cache::push_maps(const std::vector<bool> & changed) {
    std::vector<int32_t> local;
    for (const auto & t : tiers) {
        for (auto & l : layers) {
            if (!l.owner || !changed[l.il]) { continue; }
            const auto & grp = t.groups[l.group];
            local.resize(l.slot.size());
            for (size_t e = 0; e < l.slot.size(); ++e) {
                const int64_t s = l.slot[e];
                local[e] = s >= grp.base && s < grp.base + grp.n ? int32_t(s - grp.base) : -1;
            }
            ggml_backend_tensor_set(t.hot_ids[l.il], local.data(), 0, local.size() * sizeof(int32_t));
        }
    }
}

// A sysfs list such as "0-7,16-23"; empty if the file cannot be read.
static std::vector<int> read_cpulist(const std::string & path) {
    std::vector<int> cpus;
#ifdef __linux__
    std::ifstream f(path);
    std::string list;
    if (!std::getline(f, list)) { return cpus; }
    size_t pos = 0;
    while (pos < list.size()) {
        const size_t end = std::min(list.find(',', pos), list.size());
        const std::string r = list.substr(pos, end - pos);
        const size_t dash = r.find('-');
        const int first = atoi(r.c_str()), last = dash == std::string::npos ? first : atoi(r.c_str() + dash + 1);
        for (int c = first; c <= last && c < CPU_SETSIZE; ++c) { cpus.push_back(c); }
        pos = end + 1;
    }
#else
    GGML_UNUSED(path);
#endif
    return cpus;
}

// The CPUs of each NUMA node that has CPUs, in node order, from sysfs; empty if unknown.
static std::vector<std::vector<int>> numa_node_cpus() {
    std::vector<std::vector<int>> nodes;
    for (int n : read_cpulist("/sys/devices/system/node/online")) {
        auto cpus = read_cpulist("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (!cpus.empty()) { nodes.push_back(std::move(cpus)); }
    }
    return nodes;
}

// The CPUs next to the GPU, from sysfs; empty if unknown. Copies from them reach ~30 GB/s instead of ~18 GB/s unpinned.
static std::vector<int> local_cpus(ggml_backend_dev_t dev) {
    std::vector<int> cpus;
#ifdef __linux__
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    if (!props.device_id) { return cpus; }
    cpus = read_cpulist(std::string("/sys/bus/pci/devices/") + props.device_id + "/local_cpulist");
#else
    GGML_UNUSED(dev);
#endif
    return cpus;
}

static void pin_thread(const std::vector<int> & cpus) {
#ifdef __linux__
    if (cpus.empty()) { return; }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) { CPU_SET(c, &set); }
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    GGML_UNUSED(cpus);
#endif
}

// Pipelined refill: threads next to the GPU copy expert slabs into one pinned half while the GPU uploads the other half to
// device scratch and packs each slab into its arena slot, all on the stream of the tier's backend.
void llama_expert_cache::upload(const std::vector<upload_job> & jobs) {
    if (jobs.empty()) { return; }
    struct slab { const char * src; ggml_tensor * bank; int64_t slot; };
    std::map<ggml_backend_t, std::vector<slab>> by_backend;
    for (const auto & j : jobs) {
        const auto & l = model->layers[j.il];
        const ggml_tensor * src[] = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
        const auto [ti, gi] = locate(j.s);
        const auto & g = tiers[ti].groups[gi];
        for (int p = 0; p < 3; ++p) {
            by_backend[tiers[ti].backend].push_back({ static_cast<const char *>(src[p]->data) + j.e * src[p]->nb[2], g.banks[p], j.s - g.base });
        }
    }
    const size_t bytes = max_slab();
    if (!set_expert) {
        for (const auto & [b, slabs] : by_backend) {
            for (const auto & x : slabs) { ggml_backend_tensor_set(x.bank, x.src, x.slot * x.bank->nb[2], x.bank->nb[2]); }
        }
        return;
    }
    static const int per_half = [] { const char * e = getenv("LLAMA_MOE_CACHE_REFILL_SLABS"); return e ? std::max(1, atoi(e)) : 32; }();
    static const char * env_threads = getenv("LLAMA_MOE_CACHE_REFILL_THREADS");
    // Two options for --numa tensors (experts in CPU_NUMA buffers), off by default; for NPS4 set both (MiMo V2.6 Flash, NPS4:
    // refresh 210 -> ~150 ms per prompt; NPS1 is bound by the PCIe upload and not measured with them).
    // The pinned staging buffer next to the GPU is one node's memory, which at NPS4 is one quadrant: its 2 channels take all
    // copy writes and all upload reads. LLAMA_MOE_CACHE_REFILL_INTERLEAVE=1 interleaves it over the nodes instead.
    // An expert stores its rows node by node, in node order. With LLAMA_MOE_CACHE_REFILL_NUMA=1, thread t copies part t of
    // each slab and runs on the node that holds that part (at least 2 threads per node), so that it reads local memory. By
    // default all threads run next to the GPU.
    static const std::vector<std::vector<int>> node_cpus = numa_node_cpus();
    static const bool stage_interleave = [] { const char * e = getenv("LLAMA_MOE_CACHE_REFILL_INTERLEAVE"); return e && atoi(e) != 0; }();
    static const bool pin_nodes = [] { const char * e = getenv("LLAMA_MOE_CACHE_REFILL_NUMA"); return e && atoi(e) != 0; }();
    const ggml_tensor * src0 = model->layers[jobs[0].il].ffn_gate_exps;
    const int n_nodes = (int) node_cpus.size();
    const bool numa = n_nodes > 1 && src0->buffer && strcmp(ggml_backend_buffer_name(src0->buffer), "CPU_NUMA") == 0;
    const bool by_node = numa && pin_nodes;
    int n_threads = env_threads ? std::max(1, atoi(env_threads)) : 8;
    if (by_node && !env_threads) { n_threads = std::max(n_threads, 2 * n_nodes); }
    if (by_node) { n_threads = (n_threads + n_nodes - 1) / n_nodes * n_nodes; }
    ggml_backend_dev_t dev0 = ggml_backend_get_device(gpu);
    if (cpus.empty()) { cpus = local_cpus(dev0); }
    if (!pinned) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev0);
        if (buft && numa && stage_interleave) {
            auto * numa_buft_fn = (ggml_backend_buffer_type_t (*)(void))
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev0), "ggml_backend_host_numa_buffer_type");
            if (numa_buft_fn) { buft = numa_buft_fn(); }
        }
        // allocate (and first-touch) the pinned buffer next to the GPU, unless its pages are interleaved
        std::thread([&] {
            pin_thread(cpus);
            if (buft) { pinned.reset(ggml_backend_buft_alloc_buffer(buft, 2 * per_half * bytes)); }
            if (pinned) { memset(ggml_backend_buffer_get_base(pinned.get()), 0, 2 * per_half * bytes); }
        }).join();
        if (!pinned) { throw std::runtime_error("expert-cache: cannot allocate pinned refill buffers"); }
    }
    char * host = static_cast<char *>(ggml_backend_buffer_get_base(pinned.get()));
    for (const auto & [backend, slabs] : by_backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_context_ptr sctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        ggml_tensor * scratch = ggml_new_tensor_1d(sctx.get(), GGML_TYPE_I8, 2 * per_half * bytes);
        ggml_backend_buffer_ptr sbuf(ggml_backend_alloc_ctx_tensors_from_buft(sctx.get(), ggml_backend_get_default_buffer_type(backend)));
        if (!sbuf) { throw std::runtime_error("expert-cache: cannot allocate refill scratch"); }
        ggml_backend_event_t ev[2] = { ggml_backend_event_new(dev), ggml_backend_event_new(dev) };

        // workers copy batch `batch` when it changes; each takes an interleaved share of every slab
        std::mutex mutex;
        std::condition_variable cv_start, cv_done;
        int64_t batch = -1;
        int done = 0;
        bool stop = false;
        size_t first = 0, n = 0;
        char * dst = nullptr;
        std::vector<std::thread> workers;
        for (int t = 0; t < n_threads; ++t) {
            workers.emplace_back([&, t] {
                pin_thread(by_node ? node_cpus[t * n_nodes / n_threads] : cpus);
                int64_t seen = -1;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv_start.wait(lock, [&] { return stop || batch != seen; });
                        if (stop) { return; }
                        seen = batch;
                    }
                    for (size_t i = 0; i < n; ++i) {
                        // parts of each slab's own size, so that part t matches the rows of the same node in every slab
                        const size_t size = slabs[first + i].bank->nb[2];
                        const size_t part = ((size + n_threads - 1) / n_threads + 63) & ~size_t(63);
                        const size_t off = std::min(size, t * part), len = std::min(size, off + part) - off;
                        if (len > 0) { memcpy(dst + i * bytes + off, slabs[first + i].src + off, len); }
                    }
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        done++;
                    }
                    cv_done.notify_one();
                }
            });
        }
        const size_t n_batches = (slabs.size() + per_half - 1) / per_half;
        for (size_t b = 0; b < n_batches; ++b) {
            const int half = b % 2;
            int64_t t = ggml_time_us();
            if (b >= 2) { ggml_backend_event_synchronize(ev[half]); } // the upload that read this half is done
            upload_wait_us += ggml_time_us() - t;
            t = ggml_time_us();
            {
                std::lock_guard<std::mutex> lock(mutex);
                first = b * per_half;
                n = std::min<size_t>(per_half, slabs.size() - first);
                dst = host + half * per_half * bytes;
                done = 0;
                batch = b;
            }
            cv_start.notify_all();
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_done.wait(lock, [&] { return done == n_threads; });
            }
            upload_copy_us += ggml_time_us() - t;
            ggml_backend_tensor_set_async(backend, scratch, dst, half * per_half * bytes, n * bytes);
            for (size_t i = 0; i < n; ++i) {
                const auto & x = slabs[first + i];
                set_expert(backend, x.bank, x.slot, static_cast<const char *>(scratch->data) + (half * per_half + i) * bytes);
            }
            ggml_backend_event_record(ev[half], backend);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv_start.notify_all();
        for (auto & th : workers) { th.join(); }
        // the host halves are reused by the next backend
        ggml_backend_synchronize(backend);
        ggml_backend_event_free(ev[0]);
        ggml_backend_event_free(ev[1]);
    }
}

// Global top-N of (layer, expert) by selections in the most recent window rows. Rows that the window does not hold yet are
// filled from the profile prior. Resident experts keep a hysteresis bonus, because a replacement costs a full PCIe upload.
std::vector<llama_expert_cache::candidate> llama_expert_cache::select() const {
    const int64_t prior_rows = window > 0 ? window : 2048;
    std::vector<candidate> candidates;
    for (auto & l : layers) {
        if (!l.owner) { continue; }
        const double fill = std::max<int64_t>(0, prior_rows - l.rows);
        for (int e = 0; e < (int) l.count.size(); ++e) {
            const double score = l.count[e] + fill * l.prior[e];
            if (score > 0) { candidates.push_back({score * (l.slot[e] >= 0 ? 1.0 + hysteresis : 1.0), l.il, e}); }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        return a.score != b.score ? a.score > b.score : std::tie(a.il, a.e) < std::tie(b.il, b.e);
    });
    // best first, up to each group's slots in all tiers
    std::vector<int64_t> left(group_types.size(), 0);
    for (const auto & t : tiers) {
        for (size_t g = 0; g < t.groups.size(); ++g) { left[g] += t.groups[g].n; }
    }
    std::vector<candidate> keep;
    for (const auto & c : candidates) {
        if (left[layers[c.il].group]-- > 0) { keep.push_back(c); }
    }
    return keep;
}

void llama_expert_cache::cancel_async() {
    if (!async) { return; }
    std::unique_lock<std::mutex> lock(async->mutex);
    async->cv.wait(lock, [&] { return !async->queued; });
    async->done = false;
}

void llama_expert_cache::tick() {
    if (every <= 0 || window < 0 || !set_expert) { return; }
    if (async) {
        bool ready;
        {
            std::lock_guard<std::mutex> lock(async->mutex);
            ready = async->done;
        }
        if (ready) {
            // Swap the copied experts in. The decode that just returned left the GPUs idle; the packs read the scratch, so
            // they finish before the worker writes it again, and the maps change before the next graph runs.
            const int64_t t0 = ggml_time_us();
            ggml_backend_synchronize(gpu);
            for (const auto & d : async->devices) { ggml_backend_synchronize(d.backend); }
            const size_t bytes = max_slab();
            std::vector<bool> changed(layers.size(), false);
            for (size_t i = 0; i < async->jobs.size(); ++i) {
                const auto & j = async->jobs[i];
                const auto [il, e] = async->evict[i];
                if (il >= 0) {
                    layers[il].slot[e] = -1;
                    layers[il].cold[e] = e;
                    changed[il] = true;
                }
                layers[j.il].slot[j.e] = j.s;
                layers[j.il].cold[j.e] = -1;
                owner_of[j.s] = {j.il, j.e};
                changed[j.il] = true;
                const auto [ti, gi] = locate(j.s);
                const auto & g = tiers[ti].groups[gi];
                const auto & d = async->devices[async->dev[i]];
                for (int p = 0; p < 3; ++p) {
                    set_expert(d.backend, g.banks[p], j.s - g.base, static_cast<const char *>(d.scratch->data) + (3 * async->pos[i] + p) * bytes);
                }
            }
            for (const auto & d : async->devices) { ggml_backend_synchronize(d.backend); }
            push_maps(changed);
            {
                std::lock_guard<std::mutex> lock(async->mutex);
                async->done = false;
            }
            n_async++;
            n_async_uploaded += async->jobs.size();
            async_apply_ms += (ggml_time_us() - t0) / 1000.0;
        }
    }
    if (++n_small % every != 0 || any_lent()) { return; }
    if (async) {
        std::lock_guard<std::mutex> lock(async->mutex);
        if (async->queued || async->done) { return; }
    }

    // Experts that entered the top set, best first, take the free slots of their group and then the slots of the group's
    // resident experts that left it, weakest first.
    const auto keep = select();
    std::vector<std::vector<bool>> in(layers.size());
    for (auto & l : layers) { if (l.owner) { in[l.il].assign(l.count.size(), false); } }
    for (const auto & c : keep) { in[c.il][c.e] = true; }
    std::vector<candidate> add;
    for (const auto & c : keep) {
        if (layers[c.il].slot[c.e] < 0) { add.push_back(c); }
    }
    if (add.empty()) { return; }
    const int64_t prior_rows = window > 0 ? window : 2048;
    const size_t n_groups = group_types.size();
    std::vector<std::vector<candidate>> out(n_groups);
    std::vector<std::vector<int>> free_slots(n_groups);
    for (int64_t s = (int64_t) owner_of.size() - 1; s >= 0; --s) { // lowest free slot at the back
        const auto [il, e] = owner_of[s];
        const int g = locate(s).second;
        if (il < 0) {
            free_slots[g].push_back(s);
        } else if (!in[il][e]) {
            const auto & l = layers[il];
            out[g].push_back({l.count[e] + std::max<int64_t>(0, prior_rows - l.rows) * l.prior[e], il, e});
        }
    }
    for (auto & o : out) {
        std::sort(o.begin(), o.end(), [](const candidate & a, const candidate & b) {
            return a.score != b.score ? a.score < b.score : std::tie(a.il, a.e) < std::tie(b.il, b.e);
        });
    }
    std::vector<upload_job> jobs;
    std::vector<std::pair<int, int>> evict;
    std::vector<size_t> next_out(n_groups, 0);
    for (const auto & c : add) {
        if ((int64_t) jobs.size() == max_async) { break; }
        const int g = layers[c.il].group;
        int s;
        if (!free_slots[g].empty()) {
            s = free_slots[g].back();
            free_slots[g].pop_back();
            evict.push_back({-1, -1});
        } else if (next_out[g] < out[g].size()) {
            const auto & o = out[g][next_out[g]++];
            s = layers[o.il].slot[o.e];
            evict.push_back({o.il, o.e});
        } else {
            continue;
        }
        jobs.push_back({c.il, c.e, s});
    }
    if (jobs.empty()) { return; }

    const size_t bytes = max_slab(), size = 3 * max_async * bytes;
    if (!async) {
        async = std::make_unique<async_refresh>();
        ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
        if (auto * buft = ggml_backend_dev_host_buffer_type(dev)) { async->host.reset(ggml_backend_buft_alloc_buffer(buft, size)); }
        if (!async->host) {
            LLAMA_LOG_WARN("expert-cache: cannot allocate %.1f MiB of pinned memory for decode-time refresh; refreshing after prompts only\n", size / 1048576.0);
            async.reset();
            every = 0;
            return;
        }
        if (cpus.empty()) { cpus = local_cpus(dev); }
        auto * a = async.get();
        a->worker = std::thread([a, bytes, cpus = cpus] {
            pin_thread(cpus);
            char * host = static_cast<char *>(ggml_backend_buffer_get_base(a->host.get()));
            for (;;) {
                {
                    std::unique_lock<std::mutex> lock(a->mutex);
                    a->cv.wait(lock, [&] { return a->stop || a->queued; });
                    if (a->stop) { return; }
                }
                const int64_t t0 = ggml_time_us();
                for (size_t i = 0; i < a->srcs.size(); ++i) { memcpy(host + i * bytes, a->srcs[i], a->sizes[i]); }
                // the jobs of a device are contiguous
                for (size_t i = 0; i < a->jobs.size();) {
                    size_t k = i;
                    while (k < a->jobs.size() && a->dev[k] == a->dev[i]) { k++; }
                    auto & d = a->devices[a->dev[i]];
                    ggml_backend_tensor_set_async(d.copy.get(), d.scratch, host + 3 * i * bytes, 0, 3 * (k - i) * bytes);
                    i = k;
                }
                for (auto & d : a->devices) { ggml_backend_synchronize(d.copy.get()); }
                {
                    std::lock_guard<std::mutex> lock(a->mutex);
                    a->copy_us += ggml_time_us() - t0;
                    a->queued = false;
                    a->done = true;
                }
                a->cv.notify_all();
            }
        });
    }
    // one copy stream and scratch per tier GPU; the worker is idle here
    std::vector<int> dev_of(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) {
        ggml_backend_t b = tiers[locate(jobs[i].s).first].backend;
        int d = 0;
        while (d < (int) async->devices.size() && async->devices[d].backend != b) { d++; }
        if (d == (int) async->devices.size()) {
            async_refresh::device nd;
            nd.backend = b;
            ggml_backend_dev_t dev = ggml_backend_get_device(b);
            nd.copy.reset(ggml_backend_dev_init(dev, nullptr));
            nd.ctx.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
            nd.scratch = ggml_new_tensor_1d(nd.ctx.get(), GGML_TYPE_I8, size);
            nd.scratch_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(nd.ctx.get(), ggml_backend_get_default_buffer_type(b)));
            if (!nd.copy || !nd.scratch_buf) {
                LLAMA_LOG_WARN("expert-cache: cannot allocate %.1f MiB on %s for decode-time refresh; refreshing after prompts only\n",
                        size / 1048576.0, ggml_backend_name(b));
                async.reset();
                every = 0;
                return;
            }
            async->devices.push_back(std::move(nd));
        }
        dev_of[i] = d;
    }
    std::vector<size_t> order(jobs.size());
    for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return dev_of[a] < dev_of[b]; });

    std::lock_guard<std::mutex> lock(async->mutex);
    async->jobs.clear();
    async->evict.clear();
    async->dev.clear();
    async->pos.clear();
    async->srcs.clear();
    async->sizes.clear();
    std::vector<int> n_dev(async->devices.size(), 0);
    for (size_t i : order) {
        const auto & j = jobs[i];
        async->jobs.push_back(j);
        async->evict.push_back(evict[i]);
        async->dev.push_back(dev_of[i]);
        async->pos.push_back(n_dev[dev_of[i]]++);
        const auto & l = model->layers[j.il];
        for (const ggml_tensor * w : { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps }) {
            async->srcs.push_back(static_cast<const char *>(w->data) + j.e * w->nb[2]);
            async->sizes.push_back(w->nb[2]);
        }
    }
    async->queued = true;
    async->cv.notify_all();
}

void llama_expert_cache::refresh() {
    const int64_t t0 = ggml_time_us();
    cancel_async();
    if (dump) {
        dump_record(0, 2, 0, 0, nullptr, 0, 0, false);
        fflush(dump);
    }
    int64_t reclaimed_slots = 0;
    for (const auto & t : tiers) {
        for (const auto & g : t.groups) { reclaimed_slots += g.n - g.lent_lo; }
    }
    // the caller reclaimed the staging slots
    reclaimed();
    const int64_t prior_rows = window > 0 ? window : 2048;
    double total = 0;
    for (auto & l : layers) {
        if (!l.owner) { continue; }
        const double fill = std::max<int64_t>(0, prior_rows - l.rows);
        for (int e = 0; e < (int) l.count.size(); ++e) { total += l.count[e] + fill * l.prior[e]; }
    }
    const auto candidates = select();
    const size_t n_keep = candidates.size();
    std::vector<std::vector<bool>> keep(layers.size());
    double covered = 0;
    for (size_t i = 0; i < n_keep; ++i) {
        const auto & c = candidates[i];
        auto & k = keep[c.il];
        if (k.empty()) { k.assign(layers[c.il].count.size(), false); }
        k[c.e] = true;
        covered += c.score / (layers[c.il].slot[c.e] >= 0 ? 1.0 + hysteresis : 1.0);
    }
    std::vector<std::vector<int>> free_slots(group_types.size());
    std::vector<bool> changed(layers.size(), false);
    for (int64_t s = (int64_t) owner_of.size() - 1; s >= 0; --s) { // lowest free slot at the back
        const auto [il, e] = owner_of[s];
        if (il >= 0 && (keep[il].empty() || !keep[il][e])) {
            layers[il].slot[e] = -1;
            layers[il].cold[e] = e;
            owner_of[s] = {-1, -1};
            changed[il] = true;
        }
        if (owner_of[s].first < 0) { free_slots[locate(s).second].push_back(s); }
    }
    std::vector<upload_job> jobs;
    for (size_t i = 0; i < n_keep; ++i) {
        const auto & c = candidates[i];
        auto & l = layers[c.il];
        if (l.slot[c.e] >= 0) { continue; }
        auto & fs = free_slots[l.group];
        const int s = fs.back(); // select() keeps at most the group's slots in all tiers
        fs.pop_back();
        jobs.push_back({c.il, c.e, s});
        l.slot[c.e] = s;
        l.cold[c.e] = -1;
        owner_of[s] = {c.il, c.e};
        changed[c.il] = true;
    }
    const int64_t uploaded = jobs.size();
    upload_copy_us = upload_wait_us = 0;
    upload(jobs);
    push_maps(changed);
    const double ms = (ggml_time_us() - t0) / 1000.0;
    LLAMA_LOG_WARN("expert-cache: refresh %lld: %zu resident, %lld uploaded (%lld slots reclaimed) in %.1f ms (memcpy %.1f, gpu wait %.1f), window covers %.1f%%; previous decode hits %.1f%% selections, %.1f%% distinct\n",
            (long long) n_refresh, n_keep, (long long) uploaded, (long long) reclaimed_slots, ms, upload_copy_us / 1e3, upload_wait_us / 1e3, total > 0 ? 100.0 * covered / total : 0.0,
            all_sel ? 100.0 * hit_sel / all_sel : 0.0, all_distinct ? 100.0 * hit_distinct / all_distinct : 0.0);
    n_refresh++;
    n_uploaded += uploaded;
    refresh_ms += ms;
    hit_sel = all_sel = hit_distinct = all_distinct = 0;
}
