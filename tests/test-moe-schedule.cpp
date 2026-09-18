// delphi.cpp: tests for the grouped routed-MoE schedule (ggml/src/ggml-moe-schedule.h).
//
// default: the host builder and checker on fixed and synthetic routing, the CUDA/HIP device builders against the host builder,
//          and a grouped reference executor (gather -> per-expert MUL_MAT -> gather back -> weighted sum) against the MUL_MAT_ID
//          path on every backend
// --bench: schedule construction cost and routing statistics

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../ggml/src/ggml-moe-schedule.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using set_mode_fn   = void (*)(int);
using build_host_fn = bool (*)(ggml_backend_t, int, const int32_t *, int64_t, int, int, int32_t *, int32_t *, int32_t *, int, float *);
using counts_fn     = bool (*)(int, int64_t *);

struct routing {
    std::string          name;
    int                  n_experts = 0;
    int                  top_k     = 0;
    int64_t              n_tokens  = 0;
    int64_t              ids_s1    = 0;
    std::vector<int32_t> ids;
    bool                 valid_distinct = true; // every id in range and distinct inside each token
};

struct schedule {
    std::vector<int32_t> offsets;
    std::vector<int32_t> slot;
    std::vector<int32_t> inverse;
};

static schedule build_ref(const routing & r) {
    schedule s;
    s.offsets.resize(r.n_experts + 1);
    s.slot.assign(r.n_tokens*r.top_k, -1);
    s.inverse.resize(r.n_tokens*r.top_k);
    ggml_moe_schedule_build_ref(r.ids.data(), r.ids_s1, r.n_tokens, r.top_k, r.n_experts, s.offsets.data(), s.slot.data(), s.inverse.data());
    return s;
}

// Routes expert e to exactly rows[e] distinct tokens (rows[e] <= n_tokens, sum = n_tokens*top_k): the assignments are laid out
// expert by expert and dealt to a shuffled token order, so the rows[e] consecutive ones of expert e land on different tokens.
static void fill_from_rows(routing & r, const std::vector<int64_t> & rows, std::mt19937 & rng) {
    std::vector<int64_t> tokens(r.n_tokens);
    std::iota(tokens.begin(), tokens.end(), 0);
    std::shuffle(tokens.begin(), tokens.end(), rng);
    int64_t p = 0;
    for (int e = 0; e < r.n_experts; ++e) {
        for (int64_t q = 0; q < rows[e]; ++q, ++p) {
            r.ids[tokens[p % r.n_tokens]*r.ids_s1 + p / r.n_tokens] = e;
        }
    }
    GGML_ASSERT(p == r.n_tokens*r.top_k);
}

static routing make_routing(const std::string & name, int n_experts, int top_k, int64_t n_tokens, std::mt19937 & rng) {
    routing r;
    r.name      = name;
    r.n_experts = n_experts;
    r.top_k     = top_k;
    r.n_tokens  = n_tokens;
    r.ids_s1    = top_k;
    r.ids.resize(n_tokens*top_k);

    auto at = [&](int64_t t, int k) -> int32_t & { return r.ids[t*r.ids_s1 + k]; };

    // distinct draws from pool for one token
    auto fill_distinct = [&](int64_t t, const std::vector<int32_t> & pool, auto draw) {
        for (int k = 0; k < top_k; ++k) {
            int32_t e;
            do {
                e = pool[draw()];
            } while (std::find(&at(t, 0), &at(t, 0) + k, e) != &at(t, 0) + k);
            at(t, k) = e;
        }
    };

    std::vector<int32_t> all(n_experts);
    std::iota(all.begin(), all.end(), 0);

    if (name == "uniform" || name == "duplicate" || name == "invalid" || name == "strided") {
        if (name == "strided") {
            r.ids_s1 = top_k + 3;
            r.ids.assign(n_tokens*r.ids_s1, -7); // padding outside the top-k columns must be ignored
        }
        std::uniform_int_distribution<int> pick(0, n_experts - 1);
        for (int64_t t = 0; t < n_tokens; ++t) {
            fill_distinct(t, all, [&] { return pick(rng); });
        }
        if (name == "duplicate" && top_k > 1) {
            for (int64_t t = 0; t < n_tokens; t += 2) {
                at(t, top_k - 1) = at(t, 0);
            }
            r.valid_distinct = false;
        }
        if (name == "invalid") {
            for (int64_t t = 0; t < n_tokens; t += 3) {
                at(t, (int) (t % top_k)) = t % 2 ? -1 : n_experts + (int32_t) (t % 5);
            }
            r.valid_distinct = false;
        }
    } else if (name == "zipf") {
        // hot experts at random indices, so the skew does not always sit at the start of the expert range
        std::vector<int32_t> perm = all;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<double> w(n_experts);
        for (int i = 0; i < n_experts; ++i) {
            w[i] = 1.0/std::pow(i + 1.0, 1.2);
        }
        std::discrete_distribution<int> pick(w.begin(), w.end());
        for (int64_t t = 0; t < n_tokens; ++t) {
            fill_distinct(t, perm, [&] { return pick(rng); });
        }
    } else if (name == "dominant") {
        // one expert in every token, at a random top-k position
        const int32_t hot = n_experts/3;
        std::uniform_int_distribution<int> pick(0, n_experts - 1);
        std::uniform_int_distribution<int> pos(0, top_k - 1);
        for (int64_t t = 0; t < n_tokens; ++t) {
            at(t, 0) = hot;
            for (int k = 1; k < top_k; ++k) {
                int32_t e;
                do {
                    e = pick(rng);
                } while (std::find(&at(t, 0), &at(t, 0) + k, e) != &at(t, 0) + k);
                at(t, k) = e;
            }
            std::swap(at(t, 0), at(t, pos(rng)));
        }
    } else if (name == "sparse") {
        // a few distant experts, the last one included, with empty experts between them
        const int n_pool = std::max(top_k, n_experts/8);
        const int stride = n_experts/n_pool;
        std::vector<int32_t> pool;
        for (int i = 0; i < n_pool; ++i) {
            pool.push_back((i + 1)*stride - 1);
        }
        std::uniform_int_distribution<int> pick(0, n_pool - 1);
        for (int64_t t = 0; t < n_tokens; ++t) {
            fill_distinct(t, pool, [&] { return pick(rng); });
        }
    } else if (name == "roundrobin") {
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int k = 0; k < top_k; ++k) {
                at(t, k) = (int32_t) ((t*top_k + k) % n_experts);
            }
        }
    } else if (name == "single") {
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int k = 0; k < top_k; ++k) {
                at(t, k) = k;
            }
        }
    } else if (name == "last") {
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int k = 0; k < top_k; ++k) {
                at(t, k) = n_experts - top_k + k;
            }
        }
    } else if (name == "hottail") {
        // the measured GLM-5.3-Flash shape: one hot expert in about 60% of the tokens, a Zipf tail, about a fifth of the experts empty
        std::vector<int32_t> perm = all;
        std::shuffle(perm.begin(), perm.end(), rng);
        const int n_pool = std::min(n_experts, std::max(top_k + 1, (int) (0.8*n_experts)));
        std::vector<double> w(n_pool - 1);
        for (int i = 0; i < n_pool - 1; ++i) {
            w[i] = 1.0/(i + 1.0);
        }
        std::discrete_distribution<int> pick(w.begin(), w.end());
        std::bernoulli_distribution     use_hot(0.6);
        std::uniform_int_distribution<int> pos(0, top_k - 1);
        for (int64_t t = 0; t < n_tokens; ++t) {
            const int k0 = use_hot(rng) ? 1 : 0;
            if (k0) {
                at(t, 0) = perm[0];
            }
            for (int k = k0; k < top_k; ++k) {
                int32_t e;
                do {
                    e = perm[1 + pick(rng)];
                } while (std::find(&at(t, 0), &at(t, 0) + k, e) != &at(t, 0) + k);
                at(t, k) = e;
            }
            std::swap(at(t, 0), at(t, pos(rng)));
        }
    } else if (name == "tiles") {
        // rows per expert on both sides of the tile widths and class limits of the grouped executor, the rest spread thinly
        std::vector<int64_t> rows(n_experts, 0);
        int64_t left = n_tokens*top_k;
        int     e    = 0;
        for (int64_t J : { 1, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024 }) {
            for (int64_t d : { -1, 0, 1 }) {
                const int64_t n = J + d;
                if (n > 0 && n <= n_tokens && n <= left && e < n_experts/2) {
                    rows[e++] = n;
                    left -= n;
                }
            }
        }
        for (int i = e; left > 0; i = i + 1 < n_experts ? i + 1 : e) {
            const int64_t n = std::min<int64_t>({ left, 3, n_tokens - rows[i] });
            rows[i] += n;
            left    -= n;
        }
        std::shuffle(rows.begin(), rows.end(), rng);
        fill_from_rows(r, rows, rng);
    } else {
        fprintf(stderr, "unknown routing %s\n", name.c_str());
        exit(1);
    }
    return r;
}

static const char * compare_schedule(const schedule & a, const schedule & b, int32_t n_valid) {
    if (a.offsets != b.offsets) {
        return "offsets differ";
    }
    if (!std::equal(a.slot.begin(), a.slot.begin() + n_valid, b.slot.begin())) {
        return "slot differs";
    }
    if (a.inverse != b.inverse) {
        return "inverse differs";
    }
    return nullptr;
}

struct test_counts {
    int passed = 0;
    int failed = 0;

    void add(bool ok) {
        (ok ? passed : failed)++;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Host builder and checker

static void test_fixed(test_counts & counts) {
    struct fixed_case {
        const char *         name;
        int                  n_experts;
        int                  top_k;
        std::vector<int32_t> ids;
        std::vector<int32_t> offsets;
        std::vector<int32_t> slot;
        std::vector<int32_t> inverse;
    };
    const std::vector<fixed_case> cases = {
        // token 0 -> {2, 0}, token 1 -> {0, 3}, token 2 -> {2, 1}; expert 0 gets a1, a2; expert 1 a5; expert 2 a0, a4; expert 3 a3
        { "hand",      4, 2, { 2, 0,  0, 3,  2, 1 }, { 0, 2, 3, 5, 6 }, { 1, 2, 5, 0, 4, 3 }, { 3, 0, 1, 5, 4, 2 } },
        // expert 1 twice in token 0: two rows for one token
        { "duplicate", 2, 2, { 1, 1,  0, 1 },        { 0, 1, 4 },       { 2, 0, 1, 3 },       { 1, 2, 0, 3 } },
        // out-of-range ids are dropped
        { "invalid",   2, 2, { 0, 5, -1, 1 },        { 0, 1, 2 },       { 0, 3 },             { 0, -1, -1, 1 } },
        // one token, top-1
        { "single",    3, 1, { 2 },                  { 0, 0, 0, 1 },    { 0 },                { 0 } },
    };

    for (const fixed_case & c : cases) {
        routing r;
        r.name      = c.name;
        r.n_experts = c.n_experts;
        r.top_k     = c.top_k;
        r.n_tokens  = (int64_t) c.ids.size()/c.top_k;
        r.ids_s1    = c.top_k;
        r.ids       = c.ids;

        const schedule s = build_ref(r);
        const bool ok = s.offsets == c.offsets && s.inverse == c.inverse &&
            std::equal(c.slot.begin(), c.slot.end(), s.slot.begin()) &&
            ggml_moe_schedule_check(r.ids.data(), r.ids_s1, r.n_tokens, r.top_k, r.n_experts, s.offsets.data(), s.slot.data(), s.inverse.data()) == nullptr;
        if (!ok) {
            printf("  FAIL fixed %s\n", c.name);
        }
        counts.add(ok);
    }

    // the checker must reject broken schedules
    routing r;
    r.n_experts = 4;
    r.top_k     = 2;
    r.n_tokens  = 3;
    r.ids_s1    = 2;
    r.ids       = { 2, 0, 0, 3, 2, 1 };
    const schedule good = build_ref(r);
    int rejected = 0;
    for (int mutation = 0; mutation < 4; ++mutation) {
        schedule bad = good;
        switch (mutation) {
            case 0: std::swap(bad.slot[0], bad.slot[1]); break;         // unstable order inside expert 0
            case 1: std::swap(bad.slot[2], bad.slot[3]); break;         // row moved to another expert
            case 2: bad.offsets[2]++;                    break;         // wrong range
            case 3: bad.inverse[0] = bad.inverse[1];     break;         // inverse not a bijection
        }
        rejected += ggml_moe_schedule_check(r.ids.data(), r.ids_s1, r.n_tokens, r.top_k, r.n_experts,
            bad.offsets.data(), bad.slot.data(), bad.inverse.data()) != nullptr;
    }
    if (rejected != 4) {
        printf("  FAIL checker accepted %d of 4 broken schedules\n", 4 - rejected);
    }
    counts.add(rejected == 4);
}

struct shape {
    int n_experts;
    int top_k;
};

static const std::vector<shape>       k_shapes   = { { 288, 8 }, { 4, 2 }, { 8, 1 }, { 16, 4 }, { 64, 6 }, { 256, 16 } };
static const std::vector<int64_t>     k_tokens   = { 1, 2, 3, 8, 16, 33, 128, 512, 1024, 2048, 4096 };
static const std::vector<std::string> k_routings = {
    "uniform", "zipf", "dominant", "sparse", "roundrobin", "single", "last", "duplicate", "invalid", "strided"
};

// The routing weight is a unique function of (token, expert), as get_rows(probs, ids) makes it in build_moe_ffn.
static float probs_at(int64_t t, int32_t e) {
    return (float) (t*1000 + e);
}

static const char * check_weights(const routing & r, const schedule & s) {
    std::vector<float> weights(r.n_tokens*r.top_k);
    for (int64_t t = 0; t < r.n_tokens; ++t) {
        for (int k = 0; k < r.top_k; ++k) {
            weights[t*r.top_k + k] = probs_at(t, r.ids[t*r.ids_s1 + k]);
        }
    }
    for (int e = 0; e < r.n_experts; ++e) {
        for (int32_t row = s.offsets[e]; row < s.offsets[e + 1]; ++row) {
            if (weights[s.slot[row]] != probs_at(s.slot[row]/r.top_k, e)) {
                return "routing weight does not follow its (token, expert) pair";
            }
        }
    }
    return nullptr;
}

static void test_host_matrix(test_counts & counts, std::vector<routing> & device_cases) {
    std::mt19937 rng(1234);
    for (const shape & sh : k_shapes) {
        for (int64_t n_tokens : k_tokens) {
            for (const std::string & name : k_routings) {
                const routing r = make_routing(name, sh.n_experts, sh.top_k, n_tokens, rng);
                const schedule s = build_ref(r);
                const char * err = ggml_moe_schedule_check(r.ids.data(), r.ids_s1, r.n_tokens, r.top_k, r.n_experts,
                    s.offsets.data(), s.slot.data(), s.inverse.data());
                if (!err) {
                    err = check_weights(r, s);
                }
                if (err) {
                    printf("  FAIL host E=%d K=%d T=%" PRId64 " %s: %s\n", sh.n_experts, sh.top_k, n_tokens, name.c_str(), err);
                }
                counts.add(err == nullptr);
                if (r.ids_s1 == r.top_k) {
                    device_cases.push_back(r);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Device builders

static void test_device_builders(test_counts & counts, ggml_backend_t backend, build_host_fn build_host, const std::vector<routing> & cases) {
    int n_helper = 0;
    for (const routing & r : cases) {
        const schedule ref = build_ref(r);
        const int32_t  n_valid = ref.offsets[r.n_experts];

        for (int arm = 0; arm < 2; ++arm) {
            // mm_ids_helper keeps one row per (token, expert) and has no range check
            if (arm == 1 && !r.valid_distinct) {
                continue;
            }
            schedule dev;
            dev.offsets.resize(r.n_experts + 1);
            dev.slot.assign(r.n_tokens*r.top_k, -1);
            dev.inverse.assign(r.n_tokens*r.top_k, -2);
            if (!build_host(backend, arm, r.ids.data(), r.n_tokens, r.top_k, r.n_experts,
                    dev.offsets.data(), dev.slot.data(), dev.inverse.data(), 0, nullptr)) {
                if (arm == 0) {
                    printf("  FAIL device arm 0 refused E=%d K=%d T=%" PRId64 " %s\n", r.n_experts, r.top_k, r.n_tokens, r.name.c_str());
                    counts.add(false);
                }
                continue;
            }
            const char * err = compare_schedule(ref, dev, n_valid);
            if (err) {
                printf("  FAIL device arm %d E=%d K=%d T=%" PRId64 " %s: %s\n", arm, r.n_experts, r.top_k, r.n_tokens, r.name.c_str(), err);
            }
            counts.add(err == nullptr);
            n_helper += arm == 1;
        }
    }
    printf("  %zu routings, mm_ids_helper compared on %d of them\n", cases.size(), n_helper);
}

// ---------------------------------------------------------------------------------------------------------------------
// Reference executor against MUL_MAT_ID

static const int64_t k_n_embd = 256;
static const int64_t k_n_ff   = 256;

struct expert_weights {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *         gate   = nullptr; // [n_embd, n_ff, E]
    ggml_tensor *         up     = nullptr;
    ggml_tensor *         down   = nullptr; // [n_ff, n_embd, E]

    ~expert_weights() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

static std::vector<uint8_t> quantize_host(ggml_type type, int64_t n_per_row, int64_t nrows, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> src(n_per_row*nrows);
    for (float & v : src) {
        v = dist(rng);
    }
    std::vector<uint8_t> dst(ggml_row_size(type, n_per_row)*nrows);
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) dst.data(), n_per_row*nrows);
    } else {
        // dummy importance matrix, required by IQ2_XXS/IQ2_XS/IQ1_S
        const std::vector<float> imatrix(n_per_row, 1.0f);
        ggml_quantize_chunk(type, src.data(), dst.data(), 0, nrows, n_per_row,
            ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr);
    }
    return dst;
}

// Host copies of the quantized weights, shared by every backend so that all of them compute the same model.
static const std::vector<uint8_t> & cached_weights(ggml_type type, int n_experts, int which) {
    static std::map<std::tuple<int, int, int>, std::vector<uint8_t>> cache;
    const auto key = std::make_tuple((int) type, n_experts, which);
    auto it = cache.find(key);
    if (it == cache.end()) {
        std::mt19937 rng(100 + 10*which + n_experts);
        const int64_t n_per_row = which == 2 ? k_n_ff : k_n_embd;
        const int64_t nrows     = (which == 2 ? k_n_embd : k_n_ff)*n_experts;
        it = cache.emplace(key, quantize_host(type, n_per_row, nrows, rng)).first;
    }
    return it->second;
}

static void make_weights(expert_weights & w, ggml_backend_t backend, ggml_type type, int n_experts) {
    ggml_init_params params = { ggml_tensor_overhead()*4, nullptr, true };
    w.ctx  = ggml_init(params);
    w.gate = ggml_new_tensor_3d(w.ctx, type, k_n_embd, k_n_ff, n_experts);
    w.up   = ggml_new_tensor_3d(w.ctx, type, k_n_embd, k_n_ff, n_experts);
    w.down = ggml_new_tensor_3d(w.ctx, type, k_n_ff, k_n_embd, n_experts);
    w.buffer = ggml_backend_alloc_ctx_tensors(w.ctx, backend);
    ggml_tensor * tensors[3] = { w.gate, w.up, w.down };
    for (int which = 0; which < 3; ++which) {
        const std::vector<uint8_t> & data = cached_weights(type, n_experts, which);
        ggml_backend_tensor_set(tensors[which], data.data(), 0, data.size());
    }
}

static ggml_tensor * expert_view(ggml_context * ctx, ggml_tensor * w, int e) {
    return ggml_view_2d(ctx, w, w->ne[0], w->ne[1], w->nb[1], e*w->nb[2]);
}

static ggml_tensor * sum_top_k(ggml_context * ctx, ggml_tensor * y, ggml_cgraph * gf) {
    // [n_embd, top_k, n_tokens] -> [n_embd, n_tokens], in top-k order as build_moe_ffn does it
    std::vector<ggml_tensor *> views(y->ne[1]);
    for (int64_t k = 0; k < y->ne[1]; ++k) {
        views[k] = ggml_view_2d(ctx, y, y->ne[0], y->ne[2], y->nb[2], k*y->nb[1]);
        ggml_build_forward_expand(gf, views[k]);
    }
    ggml_tensor * out = views[0];
    for (int64_t k = 1; k < y->ne[1]; ++k) {
        out = ggml_add(ctx, out, views[k]);
        ggml_build_forward_expand(gf, out);
    }
    return y->ne[1] == 1 ? ggml_cont(ctx, out) : out;
}

static ggml_tensor * concat_rows(ggml_context * ctx, std::vector<ggml_tensor *> parts) {
    while (parts.size() > 1) {
        std::vector<ggml_tensor *> next;
        for (size_t i = 0; i + 1 < parts.size(); i += 2) {
            next.push_back(ggml_concat(ctx, parts[i], parts[i + 1], 1));
        }
        if (parts.size() % 2) {
            next.push_back(parts.back());
        }
        parts.swap(next);
    }
    return parts[0];
}

static std::vector<float> compute(ggml_backend_t backend, ggml_cgraph * gf, ggml_tensor * out) {
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<float> result;
    if (ggml_gallocr_alloc_graph(galloc, gf) && ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        result.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    }
    ggml_gallocr_free(galloc);
    return result;
}

struct moe_inputs {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *         x      = nullptr; // [n_embd, n_tokens]
    ggml_tensor *         ids    = nullptr; // [top_k, n_tokens], a view of [n_experts, n_tokens] as the router's argsort output
    ggml_tensor *         w      = nullptr; // [1, top_k, n_tokens]
    ggml_tensor *         tok    = nullptr; // [n_valid]  grouped row -> token
    ggml_tensor *         slot   = nullptr; // [n_valid]
    ggml_tensor *         inv    = nullptr; // [n_assign]

    ~moe_inputs() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

static void make_inputs(moe_inputs & in, ggml_backend_t backend, const routing & r, const schedule & s, std::mt19937 & rng) {
    const int64_t n_assign = r.n_tokens*r.top_k;
    const int32_t n_valid  = s.offsets[r.n_experts];

    ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
    in.ctx = ggml_init(params);
    in.x   = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, k_n_embd, r.n_tokens);
    ggml_tensor * ids_full = ggml_new_tensor_2d(in.ctx, GGML_TYPE_I32, r.n_experts, r.n_tokens);
    in.ids  = r.top_k == r.n_experts ? ids_full : ggml_view_2d(in.ctx, ids_full, r.top_k, r.n_tokens, ids_full->nb[1], 0);
    in.w    = ggml_new_tensor_3d(in.ctx, GGML_TYPE_F32, 1, r.top_k, r.n_tokens);
    in.tok  = ggml_new_tensor_1d(in.ctx, GGML_TYPE_I32, n_valid);
    in.slot = ggml_new_tensor_1d(in.ctx, GGML_TYPE_I32, n_valid);
    in.inv  = ggml_new_tensor_1d(in.ctx, GGML_TYPE_I32, n_assign);
    in.buffer = ggml_backend_alloc_ctx_tensors(in.ctx, backend);

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(k_n_embd*r.n_tokens);
    for (float & v : x) {
        v = dist(rng);
    }
    ggml_backend_tensor_set(in.x, x.data(), 0, ggml_nbytes(in.x));

    // the columns after top_k hold other expert ids, as in the argsort output
    std::vector<int32_t> ids_full_data(r.n_experts*r.n_tokens);
    for (int64_t t = 0; t < r.n_tokens; ++t) {
        for (int i = 0; i < r.n_experts; ++i) {
            ids_full_data[t*r.n_experts + i] = i < r.top_k ? r.ids[t*r.ids_s1 + i] : (int32_t) ((t + i) % r.n_experts);
        }
    }
    ggml_backend_tensor_set(ids_full, ids_full_data.data(), 0, ggml_nbytes(ids_full));

    std::uniform_real_distribution<float> wdist(0.1f, 1.0f);
    std::vector<float> w(n_assign);
    for (float & v : w) {
        v = wdist(rng);
    }
    ggml_backend_tensor_set(in.w, w.data(), 0, ggml_nbytes(in.w));

    std::vector<int32_t> tok(n_valid);
    for (int32_t i = 0; i < n_valid; ++i) {
        tok[i] = s.slot[i]/r.top_k;
    }
    ggml_backend_tensor_set(in.tok,  tok.data(),       0, ggml_nbytes(in.tok));
    ggml_backend_tensor_set(in.slot, s.slot.data(),    0, ggml_nbytes(in.slot));
    ggml_backend_tensor_set(in.inv,  s.inverse.data(), 0, ggml_nbytes(in.inv));
}

struct graph_ctx {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;

    graph_ctx() {
        const size_t n_nodes = 16384;
        ggml_init_params params = { ggml_tensor_overhead()*n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
        ctx = ggml_init(params);
        gf  = ggml_new_graph_custom(ctx, n_nodes, false);
    }
    ~graph_ctx() {
        ggml_free(ctx);
    }
};

// build_moe_ffn: MUL_MAT_ID gate and up, SwiGLU, MUL_MAT_ID down, routing weights, sum over top-k.
static std::vector<float> run_mul_mat_id(ggml_backend_t backend, const expert_weights & wt, const moe_inputs & in) {
    graph_ctx g;
    ggml_tensor * x3   = ggml_reshape_3d(g.ctx, in.x, in.x->ne[0], 1, in.x->ne[1]);
    ggml_tensor * gate = ggml_mul_mat_id(g.ctx, wt.gate, x3, in.ids);
    ggml_tensor * up   = ggml_mul_mat_id(g.ctx, wt.up,   x3, in.ids);
    ggml_tensor * h    = ggml_swiglu_split(g.ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(g.ctx, wt.down, h, in.ids);
    ggml_tensor * out  = sum_top_k(g.ctx, ggml_mul(g.ctx, down, in.w), g.gf);
    ggml_set_output(out);
    ggml_build_forward_expand(g.gf, out);
    return compute(backend, g.gf, out);
}

// The same FFN from the schedule: gather tokens into grouped rows, one MUL_MAT per active expert, weight each grouped row by
// weights[slot], gather back to assignment order with inverse, sum over top-k.
static std::vector<float> run_grouped(ggml_backend_t backend, const expert_weights & wt, const moe_inputs & in, const routing & r, const schedule & s) {
    graph_ctx g;
    ggml_tensor * xg = ggml_get_rows(g.ctx, in.x, in.tok);
    std::vector<ggml_tensor *> parts;
    for (int e = 0; e < r.n_experts; ++e) {
        const int32_t row0 = s.offsets[e];
        const int32_t rows = s.offsets[e + 1] - row0;
        if (rows == 0) {
            continue;
        }
        ggml_tensor * xe   = ggml_view_2d(g.ctx, xg, xg->ne[0], rows, xg->nb[1], row0*xg->nb[1]);
        ggml_tensor * gate = ggml_mul_mat(g.ctx, expert_view(g.ctx, wt.gate, e), xe);
        ggml_tensor * up   = ggml_mul_mat(g.ctx, expert_view(g.ctx, wt.up,   e), xe);
        ggml_tensor * h    = ggml_swiglu_split(g.ctx, gate, up);
        parts.push_back(ggml_mul_mat(g.ctx, expert_view(g.ctx, wt.down, e), h));
    }
    ggml_tensor * yg  = concat_rows(g.ctx, parts);
    ggml_tensor * wg  = ggml_get_rows(g.ctx, ggml_reshape_2d(g.ctx, in.w, 1, r.n_tokens*r.top_k), in.slot);
    ggml_tensor * ya  = ggml_get_rows(g.ctx, ggml_mul(g.ctx, yg, wg), in.inv);
    ggml_tensor * out = sum_top_k(g.ctx, ggml_reshape_3d(g.ctx, ya, ya->ne[0], r.top_k, r.n_tokens), g.gf);
    ggml_set_output(out);
    ggml_build_forward_expand(g.gf, out);
    return compute(backend, g.gf, out);
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & out) {
    double err = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        err += (double) (out[i] - ref[i])*(out[i] - ref[i]);
        sum += (double) ref[i]*ref[i];
    }
    return sum > 0.0 ? err/sum : err;
}

struct numeric_case {
    shape                    sh;
    std::vector<int64_t>     tokens;
    std::vector<std::string> routings;
};

static const std::vector<ggml_type> k_types = {
    GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S
};

static const std::vector<numeric_case> k_numeric = {
    { { 288, 8 }, { 1, 4, 64, 512, 2048 }, { "uniform", "zipf", "dominant", "sparse", "hottail" } },
    { {  16, 4 }, { 1, 7, 128 },           { "uniform", "dominant", "sparse", "duplicate" } },
    { {   4, 2 }, { 3, 33 },               { "uniform", "single", "duplicate" } },
};

// Stream-k splits a tile between blocks at points that depend on the tile list, so the grouped MMQ path sums some outputs in
// another order than the old path. The quantized operands and dot products are the same.
static const double k_grouped_tol = 1e-9;

static void test_numeric(test_counts & counts, ggml_backend_dev_t dev, set_mode_fn set_mode, set_mode_fn set_grouped, double tol) {
    const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;

    std::mt19937 rng(42);
    double worst = 0.0;
    double worst_grouped = 0.0;
    int    n_exact = 0;
    int    n_cases = 0;
    int    n_dup = 0;

    for (const numeric_case & nc : k_numeric) {
        for (ggml_type type : k_types) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            expert_weights wt;
            make_weights(wt, backend, type, nc.sh.n_experts);
            for (int64_t n_tokens : nc.tokens) {
                for (const std::string & name : nc.routings) {
                    const routing r = make_routing(name, nc.sh.n_experts, nc.sh.top_k, n_tokens, rng);
                    // On a GPU, only MMVQ and MMQ with GGML_CUDA_MOE_SCHEDULE give each duplicate id its own row; MMF and the
                    // default MMQ grouping (mm_ids_helper) keep one row per (token, expert).
                    if (!r.valid_distinct && !is_cpu && !(set_mode && ggml_is_quantized(type))) {
                        continue;
                    }
                    const schedule s = build_ref(r);
                    moe_inputs in;
                    make_inputs(in, backend, r, s, rng);

                    const std::vector<float> grouped = run_grouped(backend, wt, in, r, s);

                    std::vector<float> ref;
                    bool exact_ab = true;
                    if (set_mode) {
                        // the default grouping, then GGML_CUDA_MOE_SCHEDULE on a fresh backend so no CUDA graph is reused.
                        // With duplicate ids, mm_ids_helper does not write the inverse entry of the second assignment, so
                        // the default MMQ path scatters to an uninitialized row index: run the schedule mode only.
                        std::vector<float> ab[2];
                        for (int mode = r.valid_distinct ? 0 : 1; mode < 2; ++mode) {
                            set_mode(mode);
                            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
                            ab[mode] = run_mul_mat_id(b, wt, in);
                            ggml_backend_free(b);
                        }
                        set_mode(-1);
                        if (r.valid_distinct) {
                            exact_ab = ab[0].size() == ab[1].size() && memcmp(ab[0].data(), ab[1].data(), ab[0].size()*sizeof(float)) == 0;
                            n_exact += exact_ab;
                        } else {
                            n_dup++;
                        }
                        ref = ab[1];
                    } else {
                        ref = run_mul_mat_id(backend, wt, in);
                    }

                    // GGML_CUDA_MOE_GROUPED=grouped against the schedule mode, which has the same rows and handles duplicate ids
                    double err_grouped = 0.0;
                    if (set_grouped && ggml_is_quantized(type)) {
                        set_grouped(2);
                        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
                        const std::vector<float> out = run_mul_mat_id(b, wt, in);
                        ggml_backend_free(b);
                        set_grouped(-1);
                        err_grouped = out.size() == ref.size() ? nmse(ref, out) : INFINITY;
                        worst_grouped = std::max(worst_grouped, err_grouped);
                    }

                    const bool   ran = !ref.empty() && ref.size() == grouped.size();
                    const double err = ran ? nmse(ref, grouped) : INFINITY;
                    const bool   ok  = ran && err <= tol && exact_ab && err_grouped <= k_grouped_tol;
                    worst = std::max(worst, err);
                    n_cases++;
                    if (!ok) {
                        printf("  FAIL %s %s E=%d K=%d T=%" PRId64 " %s: nmse %.3e%s, grouped nmse %.3e\n", ggml_backend_dev_name(dev), ggml_type_name(type),
                            nc.sh.n_experts, nc.sh.top_k, n_tokens, name.c_str(), err, exact_ab ? "" : ", GGML_CUDA_MOE_SCHEDULE output differs", err_grouped);
                    }
                    counts.add(ok);
                }
            }
            ggml_backend_free(backend);
        }
    }
    printf("  %s: %d cases, worst nmse %.3e (limit %.1e)", ggml_backend_dev_name(dev), n_cases, worst, tol);
    if (set_mode) {
        printf(", GGML_CUDA_MOE_SCHEDULE bit-exact on %d, %d duplicate-id cases in schedule mode only", n_exact, n_dup);
    }
    if (set_grouped) {
        printf(", GGML_CUDA_MOE_GROUPED worst nmse %.3e", worst_grouped);
    }
    printf("\n");
}

// ---------------------------------------------------------------------------------------------------------------------
// Grouped MMQ executor against the old MMQ path, one MUL_MAT_ID

struct mmid_shape {
    const char * name;
    int          n_experts;
    int          top_k;
    int64_t      k;     // src0->ne[0]
    int64_t      n;     // src0->ne[1]
    bool         bcast; // src1 [k, 1, T] as gate/up, else [k, top_k, T] as down
};

// K spans several stream-k iterations, so tiles are split between blocks and the fixup kernel runs. n = 192 needs the fallback kernels.
static const std::vector<mmid_shape>  k_mmid_shapes   = {
    { "gate",     288, 8, 1024,  256, true  },
    { "down",     288, 8,  512, 1024, false },
    { "fallback",  64, 6, 1024,  192, true  },
    { "small",     16, 4,  512,  128, false },
};
static const std::vector<ggml_type>   k_mmid_types    = { GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q5_K, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1,
    GGML_TYPE_Q3_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_K, GGML_TYPE_IQ3_K, GGML_TYPE_IQ4_K,
    GGML_TYPE_IQ5_K };
static const std::vector<int64_t>     k_mmid_tokens   = { 16, 128, 512, 2048 };
static const std::vector<std::string> k_mmid_routings = { "uniform", "zipf", "dominant", "hottail", "sparse", "tiles", "duplicate" };

// A fresh backend per run: a CUDA graph captured in one mode would replay that mode.
static std::vector<float> run_one_mmid(ggml_backend_dev_t dev, ggml_type type, const mmid_shape & sh, const std::vector<uint8_t> & w,
        const std::vector<float> & x, const routing & r) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    ggml_init_params params = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * wt       = ggml_new_tensor_3d(ctx, type, sh.k, sh.n, sh.n_experts);
    ggml_tensor * xt       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.k, sh.bcast ? 1 : sh.top_k, r.n_tokens);
    ggml_tensor * ids_full = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, sh.n_experts, r.n_tokens);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<int32_t> ids(sh.n_experts*r.n_tokens);
    for (int64_t t = 0; t < r.n_tokens; ++t) {
        for (int i = 0; i < sh.n_experts; ++i) {
            ids[t*sh.n_experts + i] = i < r.top_k ? r.ids[t*r.ids_s1 + i] : (int32_t) ((t + i) % sh.n_experts);
        }
    }
    ggml_backend_tensor_set(wt, w.data(), 0, ggml_nbytes(wt));
    ggml_backend_tensor_set(xt, x.data(), 0, ggml_nbytes(xt));
    ggml_backend_tensor_set(ids_full, ids.data(), 0, ggml_nbytes(ids_full));

    graph_ctx g;
    ggml_tensor * idv = ggml_view_2d(g.ctx, ids_full, r.top_k, r.n_tokens, ids_full->nb[1], 0);
    ggml_tensor * out = ggml_mul_mat_id(g.ctx, wt, xt, idv);
    ggml_set_output(out);
    ggml_build_forward_expand(g.gf, out);
    std::vector<float> result = compute(backend, g.gf, out);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}

static int64_t grouped_calls(counts_fn get_counts) {
    int64_t sum = 0;
    for (int d = 0; d < 16; ++d) {
        int64_t c[2];
        if (get_counts(d, c)) {
            sum += c[0];
        }
    }
    return sum;
}

static void test_grouped_mmid(test_counts & counts, ggml_backend_dev_t dev, set_mode_fn set_schedule, set_mode_fn set_grouped, counts_fn get_counts) {
    std::mt19937 rng(77);
    double worst_old   = 0.0;
    double worst_sched = 0.0;
    int    n_cases     = 0;
    int    n_exact     = 0;
    int    n_grouped   = 0;

    for (const mmid_shape & sh : k_mmid_shapes) {
        for (ggml_type type : k_mmid_types) {
            // MXFP4 is covered by the packed tests below; regular grouped MMQ has no MXFP4 implementation on SM80.
            // The IQ*_K types are not in moe_grouped_type_supported() either, so the grouped path never runs for
            // them: this test would compare against the default path, which keeps one row per (token, expert).
            if (type == GGML_TYPE_MXFP4 ||
                type == GGML_TYPE_IQ2_K || type == GGML_TYPE_IQ3_K ||
                type == GGML_TYPE_IQ4_K || type == GGML_TYPE_IQ5_K) {
                continue;
            }
            const std::vector<uint8_t> w = quantize_host(type, sh.k, sh.n*sh.n_experts, rng);
            for (int64_t n_tokens : k_mmid_tokens) {
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                std::vector<float> x(sh.k*(sh.bcast ? 1 : sh.top_k)*n_tokens);
                for (float & v : x) {
                    v = dist(rng);
                }
                for (const std::string & name : k_mmid_routings) {
                    const routing r = make_routing(name, sh.n_experts, sh.top_k, n_tokens, rng);

                    set_schedule(1);
                    const std::vector<float> sched = run_one_mmid(dev, type, sh, w, x, r);
                    set_schedule(-1);

                    const int64_t before = grouped_calls(get_counts);
                    set_grouped(2);
                    const std::vector<float> grouped = run_one_mmid(dev, type, sh, w, x, r);
                    set_grouped(-1);
                    const bool ran_grouped = grouped_calls(get_counts) > before;

                    // the default MMQ grouping keeps one row per (token, expert), see test_numeric
                    std::vector<float> old;
                    if (r.valid_distinct) {
                        old = run_one_mmid(dev, type, sh, w, x, r);
                    }

                    const bool   sizes_ok  = !grouped.empty() && sched.size() == grouped.size() && (old.empty() || old.size() == grouped.size());
                    const double err_sched = sizes_ok ? nmse(sched, grouped) : INFINITY;
                    const double err_old   = sizes_ok && !old.empty() ? nmse(old, grouped) : 0.0;
                    // below 128 tokens MMVQ may take the op, so the grouped path is optional there
                    const bool   ok        = sizes_ok && err_sched <= k_grouped_tol && err_old <= k_grouped_tol && (ran_grouped || n_tokens < 128);

                    worst_sched = std::max(worst_sched, err_sched);
                    worst_old   = std::max(worst_old, err_old);
                    n_exact    += sizes_ok && memcmp(sched.data(), grouped.data(), grouped.size()*sizeof(float)) == 0;
                    n_grouped  += ran_grouped;
                    n_cases++;
                    if (!ok) {
                        printf("  FAIL %s %s %s T=%" PRId64 " %s: nmse vs schedule %.3e, vs old %.3e%s\n", ggml_backend_dev_name(dev), sh.name,
                            ggml_type_name(type), n_tokens, name.c_str(), err_sched, err_old, ran_grouped ? "" : ", grouped path did not run");
                    }
                    counts.add(ok);
                }
            }
        }
    }
    printf("  %s: %d cases, %d on the grouped path, bit-exact against the schedule mode on %d, worst nmse vs schedule %.3e, vs old %.3e (limit %.1e)\n",
        ggml_backend_dev_name(dev), n_cases, n_grouped, n_exact, worst_sched, worst_old, k_grouped_tol);
}

// ---------------------------------------------------------------------------------------------------------------------
// Weights packed in place (GGML_CUDA_MOE_GROUPED=packed, GGML_CUDA_MOE_PACKED_INPLACE=1): one model, one pack, a mixed sequence of
// MMVQ batches and grouped batches, against the GGUF layout on the old path (MMVQ, MMQ) and on grouped MMQ.

static const int64_t k_pk_embd    = 2048;
static const int64_t k_pk_ff      = 1024;
static const int     k_pk_experts = 16;
static const int     k_pk_top_k   = 8;

// The first graphs have single tokens, so the pack happens before any batch reaches the grouped path.
static const std::vector<int64_t> k_pk_sequence = { 1, 1, 2, 4, 512, 1, 6, 8, 64, 3, 1, 16, 2, 128, 1, 5, 7 };

// The down projection quantizes the SwiGLU output to q8_1 again, so a changed fp32 summation order in gate/up moves a few values to
// another q8_1 step: grouped MMQ and the old MMQ path differ by about 1e-9 on these random weights. A layout error gives >= 1e-2.
static const double k_pk_ffn_tol = 1e-6;

struct pk_types {
    ggml_type gate_up;
    ggml_type down;
};

static const std::vector<pk_types> k_pk_types = {
    { GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0   },
    { GGML_TYPE_Q5_0,   GGML_TYPE_Q5_0   },
    { GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL },
    { GGML_TYPE_MXFP4,  GGML_TYPE_MXFP4  },
    { GGML_TYPE_Q5_K,   GGML_TYPE_Q5_K   },
    { GGML_TYPE_IQ3_S,  GGML_TYPE_IQ3_S  },
    { GGML_TYPE_IQ3_S,  GGML_TYPE_Q4_K   },
    { GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS },
    { GGML_TYPE_IQ3_S,  GGML_TYPE_IQ4_XS },
    { GGML_TYPE_Q4_K,   GGML_TYPE_Q4_K   },
    { GGML_TYPE_Q6_K,   GGML_TYPE_Q6_K   },
    { GGML_TYPE_IQ3_S,  GGML_TYPE_Q6_K   },
    { GGML_TYPE_Q4_1,   GGML_TYPE_Q4_1   },
    { GGML_TYPE_Q5_1,   GGML_TYPE_Q5_1   },
    { GGML_TYPE_Q3_K,   GGML_TYPE_Q3_K   },
    { GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_XXS },
    { GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS },
    { GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS  },
    { GGML_TYPE_IQ2_S,  GGML_TYPE_IQ2_S   },
    { GGML_TYPE_IQ2_K,  GGML_TYPE_IQ2_K   },
    { GGML_TYPE_IQ3_K,  GGML_TYPE_IQ3_K   },
    { GGML_TYPE_IQ4_K,  GGML_TYPE_IQ4_K   },
    { GGML_TYPE_IQ5_K,  GGML_TYPE_IQ5_K   },
    { GGML_TYPE_IQ2_KS, GGML_TYPE_IQ2_KS  },
    { GGML_TYPE_IQ3_KS, GGML_TYPE_IQ3_KS  },
    { GGML_TYPE_IQ4_KS, GGML_TYPE_IQ4_KS  },
    { GGML_TYPE_IQ5_KS, GGML_TYPE_IQ5_KS  },
};

static bool pk_packable(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q5_K ||
        type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K ||
        type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ3_XXS ||
        type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_K ||
        type == GGML_TYPE_IQ3_K || type == GGML_TYPE_IQ4_K || type == GGML_TYPE_IQ5_K || type == GGML_TYPE_IQ2_KS ||
        type == GGML_TYPE_IQ3_KS || type == GGML_TYPE_IQ4_KS || type == GGML_TYPE_IQ5_KS;
}

// Up to this batch every projection goes to MMVQ on all packed-capable GPUs (get_mmvq_mmid_max_batch_turing_plus: IQ3_S 6, IQ3_XXS and MXFP4 7;
// IQ4_XS, Q4_K, Q5_K and Q6_K 8), and the packed MMVQ kernels must give the bits of MMVQ on the GGUF layout. At T = 1 an IQ4_XS,
// MXFP4, Q4_K, Q5_K or Q6_K down projection runs the fused MoE down reduction. Larger batches run the grouped packed kernel against MMQ.
static int64_t pk_mmvq_max(const pk_types & types) {
    auto max_batch = [](ggml_type type) {
        if (type == GGML_TYPE_IQ3_S) {
            return 6;
        }
        if (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_MXFP4) {
            return 7;
        }
        if (type == GGML_TYPE_IQ2_S) {
            return 7;
        }
        if (type == GGML_TYPE_Q3_K) {
            return 5;
        }
        return 8;
    };
    return std::min(max_batch(types.gate_up), max_batch(types.down));
}

// Random bits in every block with finite scales: a lossless layout must carry all non-scale byte patterns. KS formats also need a finite row scale.
static std::vector<uint8_t> random_blocks(ggml_type type, int64_t n_per_row, int64_t nrows, std::mt19937 & rng) {
    std::vector<uint8_t> data(ggml_row_size(type, n_per_row)*nrows);
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t & b : data) {
        b = (uint8_t) byte(rng);
    }
    if (type == GGML_TYPE_IQ2_KS || type == GGML_TYPE_IQ3_KS || type == GGML_TYPE_IQ4_KS || type == GGML_TYPE_IQ5_KS) {
        const bool f16 = type == GGML_TYPE_IQ2_KS || type == GGML_TYPE_IQ3_KS;
        std::uniform_real_distribution<float> row_scale(0.5e-3f, 2e-3f);
        const size_t row_size = ggml_row_size(type, n_per_row);
        for (int64_t r = 0; r < nrows; ++r) {
            const float d = row_scale(rng);
            if (f16) {
                const ggml_fp16_t dh = ggml_fp32_to_fp16(d);
                memcpy(data.data() + r*row_size, &dh, sizeof(dh));
            } else {
                memcpy(data.data() + r*row_size, &d, sizeof(d));
            }
        }
        return data;
    }
    // Q5_K has unsigned 5-bit codes and therefore a larger positive mean than the other DM test format, Q4_K. Use smaller
    // super-block scales for its three-stage gate/up/down chain so random codes do not overflow the nonlinear intermediate.
    const float scale_min = type == GGML_TYPE_Q5_K ? 0.05e-3f : 0.5e-3f;
    const float scale_max = type == GGML_TYPE_Q5_K ? 0.20e-3f : 2e-3f;
    std::uniform_real_distribution<float> dist(scale_min, scale_max);
    // The byte offsets of f16 scales in the block: d and dmin first (Q4_K and Q5_K), d last (Q6_K), else d first.
    std::vector<size_t> f16_offsets = type == GGML_TYPE_MXFP4 ? std::vector<size_t>{} : std::vector<size_t>{ 0 };
    if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        f16_offsets = { 0, sizeof(ggml_fp16_t) };
    } else if (type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1) {
        f16_offsets = { 0, sizeof(ggml_fp16_t) };
    } else if (type == GGML_TYPE_Q6_K) {
        f16_offsets = { ggml_type_size(type) - sizeof(ggml_fp16_t) };
    } else if (type == GGML_TYPE_Q3_K) {
        f16_offsets = { ggml_type_size(type) - sizeof(ggml_fp16_t) };
    }
    // Keep MXFP4 scales finite and within a useful range for the FFN chain. The byte round-trip check still covers arbitrary source
    // bytes; this bound keeps the numerical comparison from turning into an overflow test at the largest batches.
    std::uniform_int_distribution<int> e8m0(120, 135);
    for (size_t off = 0; off < data.size(); off += ggml_type_size(type)) {
        if (type == GGML_TYPE_MXFP4) {
            data[off] = (uint8_t) e8m0(rng);
        }
        for (size_t f : f16_offsets) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(dist(rng));
            memcpy(data.data() + off + f, &d, sizeof(d));
        }
    }
    return data;
}

struct pk_ffn {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *         w[3]   = {}; // gate, up [n_embd, n_ff, E], down [n_ff, n_embd, E]

    pk_ffn(ggml_backend_t backend, const pk_types & types, const std::vector<uint8_t> (&data)[3]) {
        ggml_init_params params = { ggml_tensor_overhead()*4, nullptr, true };
        ctx  = ggml_init(params);
        w[0] = ggml_new_tensor_3d(ctx, types.gate_up, k_pk_embd, k_pk_ff, k_pk_experts);
        w[1] = ggml_new_tensor_3d(ctx, types.gate_up, k_pk_embd, k_pk_ff, k_pk_experts);
        w[2] = ggml_new_tensor_3d(ctx, types.down, k_pk_ff, k_pk_embd, k_pk_experts);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        for (int i = 0; i < 3; ++i) {
            ggml_backend_tensor_set(w[i], data[i].data(), 0, data[i].size());
        }
    }
    ~pk_ffn() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

struct pk_input {
    std::vector<float>   x;   // [n_embd, T]
    std::vector<int32_t> ids; // [E, T], top-k first
    std::vector<float>   w;   // [top_k, T]
};

static pk_input make_pk_input(int64_t n_tokens, std::mt19937 & rng) {
    pk_input in;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    in.x.resize(k_pk_embd*n_tokens);
    for (float & v : in.x) {
        v = dist(rng);
    }
    const routing r = make_routing("uniform", k_pk_experts, k_pk_top_k, n_tokens, rng);
    in.ids.resize(k_pk_experts*n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < k_pk_experts; ++i) {
            in.ids[t*k_pk_experts + i] = i < k_pk_top_k ? r.ids[t*r.ids_s1 + i] : (int32_t) ((t + i) % k_pk_experts);
        }
    }
    std::uniform_real_distribution<float> wdist(0.1f, 1.0f);
    in.w.resize(k_pk_top_k*n_tokens);
    for (float & v : in.w) {
        v = wdist(rng);
    }
    return in;
}

// build_moe_ffn as in run_mul_mat_id, computed reps times with one graph and one allocation, so that the CUDA backend runs the
// graph eagerly, then captures it, then replays it. stable: every repetition gave the same bits.
static std::vector<float> run_pk_ffn(ggml_backend_t backend, const pk_ffn & f, const pk_input & in, int reps, bool & stable) {
    const int64_t n_tokens = (int64_t) in.x.size()/k_pk_embd;

    ggml_init_params params = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_pk_embd, n_tokens);
    ggml_tensor * ids_full = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k_pk_experts, n_tokens);
    ggml_tensor * w        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k_pk_top_k, n_tokens);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(x, in.x.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids_full, in.ids.data(), 0, ggml_nbytes(ids_full));
    ggml_backend_tensor_set(w, in.w.data(), 0, ggml_nbytes(w));

    graph_ctx g;
    ggml_tensor * ids  = ggml_view_2d(g.ctx, ids_full, k_pk_top_k, n_tokens, ids_full->nb[1], 0);
    ggml_tensor * x3   = ggml_reshape_3d(g.ctx, x, k_pk_embd, 1, n_tokens);
    ggml_tensor * gate = ggml_mul_mat_id(g.ctx, f.w[0], x3, ids);
    ggml_tensor * up   = ggml_mul_mat_id(g.ctx, f.w[1], x3, ids);
    ggml_tensor * h    = ggml_swiglu_split(g.ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(g.ctx, f.w[2], h, ids);
    ggml_tensor * out  = sum_top_k(g.ctx, ggml_mul(g.ctx, down, w), g.gf);
    ggml_set_output(out);
    ggml_build_forward_expand(g.gf, out);

    std::vector<float> result;
    stable = true;
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (ggml_gallocr_alloc_graph(galloc, g.gf)) {
        std::vector<float> rep(ggml_nelements(out));
        for (int i = 0; i < reps; ++i) {
            if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
                result.clear();
                break;
            }
            ggml_backend_tensor_get(out, rep.data(), 0, ggml_nbytes(out));
            if (i == 0) {
                result = rep;
            } else {
                stable = stable && memcmp(rep.data(), result.data(), rep.size()*sizeof(float)) == 0;
            }
        }
    }
    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static std::vector<uint8_t> tensor_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, data.size());
    return data;
}

static void test_packed_mixed(test_counts & counts, ggml_backend_dev_t dev, set_mode_fn set_grouped, set_mode_fn set_inplace) {
    std::mt19937 rng(91);
    for (const pk_types & types : k_pk_types) {
        const std::vector<uint8_t> data[3] = {
            random_blocks(types.gate_up, k_pk_embd, k_pk_ff*k_pk_experts, rng),
            random_blocks(types.gate_up, k_pk_embd, k_pk_ff*k_pk_experts, rng),
            random_blocks(types.down,    k_pk_ff,   k_pk_embd*k_pk_experts, rng),
        };
        std::map<int64_t, pk_input> inputs;
        for (int64_t n_tokens : k_pk_sequence) {
            if (!inputs.count(n_tokens)) {
                inputs.emplace(n_tokens, make_pk_input(n_tokens, rng));
            }
        }

        // references on the GGUF layout: the old path (MMVQ, then MMQ) and grouped MMQ
        std::map<int64_t, std::vector<float>> ref[2];
        const int ref_modes[2] = { 0, 2 };
        for (int a = 0; a < 2; ++a) {
            set_grouped(ref_modes[a]);
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            {
                pk_ffn f(backend, types, data);
                for (const auto & it : inputs) {
                    bool stable;
                    ref[a][it.first] = run_pk_ffn(backend, f, it.second, 1, stable);
                }
            }
            ggml_backend_free(backend);
        }

        set_grouped(5);
        set_inplace(1);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        {
            pk_ffn f(backend, types, data);
            const int64_t mmvq_max = pk_mmvq_max(types);
            std::vector<uint8_t> packed[3];
            int n_bitwise = 0;
            double worst = 0.0;
            double worst_ref = 0.0; // old MMQ against grouped MMQ, both on the GGUF layout
            bool ok_all = true;
            for (size_t i = 0; i < k_pk_sequence.size(); ++i) {
                const int64_t n_tokens = k_pk_sequence[i];
                bool stable;
                const std::vector<float> out = run_pk_ffn(backend, f, inputs.at(n_tokens), 3, stable);
                const std::vector<float> & r_old = ref[0][n_tokens];
                const std::vector<float> & r_grp = ref[1][n_tokens];
                const bool sizes_ok = !out.empty() && out.size() == r_old.size() && out.size() == r_grp.size();
                const bool bitwise  = sizes_ok && memcmp(out.data(), r_old.data(), out.size()*sizeof(float)) == 0;
                const double e_old  = sizes_ok ? nmse(r_old, out) : INFINITY;
                const double e_grp  = sizes_ok ? nmse(r_grp, out) : INFINITY;
                const double e_ref  = sizes_ok ? nmse(r_old, r_grp) : INFINITY;
                bool ok = sizes_ok && stable && (n_tokens <= mmvq_max ? bitwise : e_old <= k_pk_ffn_tol && e_grp <= k_pk_ffn_tol);

                // the weights are packed once, in the first graph, and stay packed
                for (int p = 0; p < 3; ++p) {
                    const std::vector<uint8_t> now = tensor_bytes(f.w[p]);
                    if (i == 0) {
                        packed[p] = now;
                        const bool changed = now != data[p];
                        ok = ok && changed == pk_packable(p < 2 ? types.gate_up : types.down);
                    } else {
                        ok = ok && now == packed[p];
                    }
                }

                n_bitwise += bitwise;
                if (n_tokens > mmvq_max) {
                    worst     = std::max(worst, std::max(e_old, e_grp));
                    worst_ref = std::max(worst_ref, e_ref);
                }
                ok_all = ok_all && ok;
                if (!ok) {
                    printf("  FAIL %s %s/%s step %zu T=%" PRId64 ": %s, nmse vs old %.3e, vs grouped %.3e (old vs grouped %.3e)%s\n",
                        ggml_backend_dev_name(dev), ggml_type_name(types.gate_up), ggml_type_name(types.down), i, n_tokens,
                        bitwise ? "bitwise" : "not bitwise", e_old, e_grp, e_ref, stable ? "" : ", repetitions differ");
                }
                counts.add(ok);
            }
            printf("  %s %s/%s: %zu graphs x 3 repetitions, bitwise against the GGUF layout on %d (required up to T=%" PRId64 "), "
                "above: worst nmse %.3e (limit %.1e, old vs grouped MMQ %.3e)%s\n",
                ggml_backend_dev_name(dev), ggml_type_name(types.gate_up), ggml_type_name(types.down), k_pk_sequence.size(), n_bitwise,
                mmvq_max, worst, k_pk_ffn_tol, worst_ref, ok_all ? "" : ", FAILED");
        }
        ggml_backend_free(backend);
        set_inplace(-1);
        set_grouped(-1);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Construction cost

static void bench(ggml_backend_t backend, build_host_fn build_host, int n_experts, int top_k, const std::vector<int64_t> & tokens,
        const std::vector<std::string> & routings, int reps) {
    printf("| T | routing | active | p50 | p90 | p99 | max | max/mean | n_eff | KiB | host us | device us | mm_ids_helper us |\n");
    printf("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    std::mt19937 rng(7);
    for (int64_t n_tokens : tokens) {
        for (const std::string & name : routings) {
            const routing r = make_routing(name, n_experts, top_k, n_tokens, rng);
            schedule s = build_ref(r);

            std::vector<double> host_us;
            for (int i = 0; i < reps; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                ggml_moe_schedule_build_ref(r.ids.data(), r.ids_s1, r.n_tokens, r.top_k, r.n_experts, s.offsets.data(), s.slot.data(), s.inverse.data());
                host_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(host_us.begin(), host_us.end());

            ggml_moe_schedule_stats st;
            std::vector<int32_t> scratch(n_experts);
            ggml_moe_schedule_get_stats(s.offsets.data(), n_experts, n_tokens*top_k, scratch.data(), &st);

            float dev_us[2] = { NAN, NAN };
            if (backend) {
                schedule d = s;
                for (int arm = 0; arm < 2; ++arm) {
                    if (!build_host(backend, arm, r.ids.data(), r.n_tokens, r.top_k, r.n_experts,
                            d.offsets.data(), d.slot.data(), d.inverse.data(), reps, &dev_us[arm])) {
                        dev_us[arm] = NAN;
                    }
                }
            }

            printf("| %" PRId64 " | %s | %d | %.1f | %.1f | %.1f | %d | %.2f | %.1f | %.1f | %.1f | %.1f | %.1f |\n",
                n_tokens, name.c_str(), st.n_active, st.rows_p50, st.rows_p90, st.rows_p99, st.rows_max, st.max_over_mean, st.n_eff,
                st.nbytes/1024.0, host_us[host_us.size()/2], dev_us[0], dev_us[1]);
        }
    }
}

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t next = s.find(',', pos);
        out.push_back(s.substr(pos, next == std::string::npos ? std::string::npos : next - pos));
        if (next == std::string::npos) {
            break;
        }
        pos = next + 1;
    }
    return out;
}

int main(int argc, char ** argv) {
    bool        do_bench   = false;
    bool        no_numeric = false; // e.g. for compute-sanitizer runs of the builders only
    bool        only_packed = false;
    std::string only_dev;
    int         n_experts = 288;
    int         top_k     = 8;
    int         reps      = 100;
    std::vector<int64_t>     bench_tokens   = { 1, 8, 32, 128, 512, 1024, 2048, 4096, 8192 };
    std::vector<std::string> bench_routings = { "uniform", "zipf", "dominant", "single" };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", arg.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (arg == "--bench") {
            do_bench = true;
        } else if (arg == "--no-numeric") {
            no_numeric = true;
        } else if (arg == "--packed") {
            only_packed = true;
        } else if (arg == "-b" || arg == "--device") {
            only_dev = next();
        } else if (arg == "--experts") {
            n_experts = std::stoi(next());
        } else if (arg == "--top-k") {
            top_k = std::stoi(next());
        } else if (arg == "--reps") {
            reps = std::stoi(next());
        } else if (arg == "--m") {
            bench_tokens.clear();
            for (const std::string & v : split(next())) {
                bench_tokens.push_back(std::stoll(v));
            }
        } else if (arg == "--dist") {
            bench_routings = split(next());
        } else {
            fprintf(stderr, "usage: %s [-b DEVICE] [--no-numeric] [--packed] [--bench [--experts N] [--top-k N] [--m LIST] [--dist LIST] [--reps N]]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        if (!only_dev.empty() && only_dev != ggml_backend_dev_name(dev)) {
            continue;
        }
        devs.push_back(dev);
    }

    auto proc = [](ggml_backend_dev_t dev, const char * name) {
        return ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), name);
    };

    if (do_bench) {
        ggml_backend_t backend    = nullptr;
        build_host_fn  build_host = nullptr;
        for (ggml_backend_dev_t dev : devs) {
            build_host = (build_host_fn) proc(dev, "ggml_backend_cuda_moe_schedule_build_host");
            if (build_host) {
                backend = ggml_backend_dev_init(dev, nullptr);
                printf("device %s, E=%d, top-k %d, %d reps\n\n", ggml_backend_dev_name(dev), n_experts, top_k, reps);
                break;
            }
        }
        bench(backend, build_host, n_experts, top_k, bench_tokens, bench_routings, reps);
        if (backend) {
            ggml_backend_free(backend);
        }
        return 0;
    }

    test_counts counts;
    std::vector<routing> device_cases;

    printf("host builder, fixed cases\n");
    test_fixed(counts);
    printf("host builder, synthetic routing matrix\n");
    test_host_matrix(counts, device_cases);

    for (ggml_backend_dev_t dev : devs) {
        const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        build_host_fn build_host  = (build_host_fn) proc(dev, "ggml_backend_cuda_moe_schedule_build_host");
        set_mode_fn   set_mode    = (set_mode_fn)   proc(dev, "ggml_backend_cuda_moe_schedule_set_mode");
        set_mode_fn   set_grouped = (set_mode_fn)   proc(dev, "ggml_backend_cuda_moe_grouped_set_mode");
        counts_fn     get_counts  = (counts_fn)     proc(dev, "ggml_backend_cuda_moe_grouped_get_counts");

        // the grouped path exists only on NVIDIA GPUs; elsewhere the mode falls back to the old path
        if (set_grouped && get_counts && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            const int64_t before = grouped_calls(get_counts);
            std::mt19937 rng(5);
            const mmid_shape & sh = k_mmid_shapes[0];
            set_grouped(2);
            run_one_mmid(dev, GGML_TYPE_Q8_0, sh, quantize_host(GGML_TYPE_Q8_0, sh.k, sh.n*sh.n_experts, rng),
                std::vector<float>(sh.k*512, 0.5f), make_routing("uniform", sh.n_experts, sh.top_k, 512, rng));
            set_grouped(-1);
            if (grouped_calls(get_counts) == before) {
                printf("grouped MMQ is not available on %s\n", ggml_backend_dev_name(dev));
                set_grouped = nullptr;
            }
        } else {
            set_grouped = nullptr;
        }

        if (build_host) {
            printf("device builders on %s\n", ggml_backend_dev_name(dev));
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            test_device_builders(counts, backend, build_host, device_cases);
            ggml_backend_free(backend);
        }

        if (no_numeric) {
            continue;
        }
        if (!only_packed) {
            printf("grouped reference vs MUL_MAT_ID on %s\n", ggml_backend_dev_name(dev));
            // The CPU computes both paths with the same dot products. A GPU picks another kernel for a few rows per expert (MMVQ
            // instead of MMQ, cuBLAS instead of MMF). The FFN chains two matrix multiplications, so allow twice the MUL_MAT_ID limit
            // of test-backend-ops (5e-4). Grouping errors give >= 1e-2.
            test_numeric(counts, dev, set_mode, set_grouped, is_cpu ? 1e-8 : 1e-3);
        }

        if (set_grouped && set_mode && !only_packed) {
            printf("grouped MMQ vs old MMQ, one MUL_MAT_ID on %s\n", ggml_backend_dev_name(dev));
            test_grouped_mmid(counts, dev, set_mode, set_grouped, get_counts);
        }

        set_mode_fn set_inplace = (set_mode_fn) proc(dev, "ggml_backend_cuda_moe_packed_set_inplace");
        if (set_grouped && set_inplace) {
            printf("packed in place, mixed MMVQ and grouped batches on %s\n", ggml_backend_dev_name(dev));
            test_packed_mixed(counts, dev, set_grouped, set_inplace);
        }
    }

    printf("\n%d passed, %d failed\n", counts.passed, counts.failed);
    return counts.failed == 0 ? 0 : 1;
}
