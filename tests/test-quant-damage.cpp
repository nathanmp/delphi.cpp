// Unit and integration tests for the sampled quantization damage estimator (tools/quant-damage).

#include "quant-damage.h"

#include "gguf.h"
#include "imatrix-loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond); n_failed++; } } while (0)

static bool near(double a, double b, double rel) {
    return std::fabs(a - b) <= rel * std::max(std::fabs(a), std::fabs(b)) + 1e-300;
}

// heavy-tailed rows with per-row scales and a few outlier rows, like real weights
static std::vector<float> make_weights(int64_t n_rows, int64_t k, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::student_t_distribution<float> td(4.0f);
    std::vector<float> w(n_rows*k);
    for (int64_t r = 0; r < n_rows; ++r) {
        const float scale = 0.01f * std::exp(0.5f * nd(rng));
        const bool outlier = r % 97 == 13;
        for (int64_t j = 0; j < k; ++j) {
            float v = scale * td(rng);
            if (outlier && j % 61 == 0) {
                v *= 20.0f;
            }
            w[r*k + j] = v;
        }
    }
    return w;
}

static qd_tensor_info make_info(const std::string & name, int64_t k, int64_t rows, int64_t n_expert) {
    qd_tensor_info t;
    t.name   = name;
    t.type   = GGML_TYPE_F32;
    t.n_dims = n_expert > 1 ? 3 : 2;
    t.ne[0] = k;
    t.ne[1] = rows;
    t.ne[2] = n_expert;
    return t;
}

static common_imatrix make_imatrix(const qd_tensor_info & t, uint64_t seed, int64_t missing_expert = -1) {
    std::mt19937_64 rng(seed);
    std::lognormal_distribution<float> ld(0.0f, 1.0f);
    common_imatrix im;
    im.has_metadata = true;
    auto & e = im.entries[t.name];
    const int64_t nex = t.n_expert();
    for (int64_t x = 0; x < nex; ++x) {
        const int64_t count = x == missing_expert ? 0 : 100 + 50*x;
        e.counts.push_back(count);
        for (int64_t j = 0; j < t.n_per_row(); ++j) {
            e.sums.push_back(count > 0 ? count * ld(rng) : 0.0f);
        }
    }
    return im;
}

static void test_estimators() {
    // two strata with known values; census gives zero variance
    std::vector<qd_stratum_sample> s(2);
    s[0].N = 4; s[0].y = {1, 2, 3, 4}; s[0].x = {1, 1, 1, 1}; s[0].X = 4;
    s[1].N = 2; s[1].y = {10, 20};     s[1].x = {2, 2};       s[1].X = 4;
    auto tot = qd_estimate_total(s);
    CHECK(near(tot.value, 40.0, 1e-12));
    CHECK(tot.se == 0.0);
    auto rat = qd_estimate_ratio(s);
    CHECK(near(rat.value, 5.0, 1e-12));
    CHECK(rat.se == 0.0);

    // stratum weighting: stratum means are expanded by N_h, not pooled
    s[0].N = 400; s[0].X = 400;
    s[1].N = 2;
    tot = qd_estimate_total(s);
    CHECK(near(tot.value, 400*2.5 + 30.0, 1e-12));
    // Var = N^2 (1 - n/N) s^2 / n with s^2 = 5/3
    CHECK(near(tot.se, std::sqrt(400.0*400.0*(1 - 4.0/400)*(5.0/3)/4), 1e-12));

    auto rt = qd_estimate_total_ratio(s);
    CHECK(near(rt.value, (400*2.5 + 30.0) / (400 + 4) * 404, 1e-12));

    // singleton stratum borrows the pooled relative variance and does not return NaN
    std::vector<qd_stratum_sample> s2(2);
    s2[0].N = 10; s2[0].y = {1, 3}; s2[0].x = {1, 1};
    s2[1].N = 10; s2[1].y = {5};    s2[1].x = {1};
    tot = qd_estimate_total(s2);
    CHECK(std::isfinite(tot.se) && tot.se > 0);

    // unsampled stratum -> NaN, never a silent zero
    s2[1].y.clear(); s2[1].x.clear();
    tot = qd_estimate_total(s2);
    CHECK(std::isnan(tot.value));
}

static void test_permutation() {
    for (int64_t n : { 1, 2, 3, 7, 64, 1000, 4097 }) {
        std::vector<uint8_t> seen(n, 0);
        bool ok = true;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t v = qd_feistel(i, n, 12345);
            ok = ok && v >= 0 && v < n && !seen[v];
            if (v >= 0 && v < n) seen[v] = 1;
        }
        CHECK(ok);
    }
    // different keys give different orders
    int same = 0;
    for (int64_t i = 0; i < 100; ++i) {
        same += qd_feistel(i, 1000, 1) == qd_feistel(i, 1000, 2);
    }
    CHECK(same < 10);
}

static void test_types() {
    CHECK(qd_type_from_name("Q4_K") == GGML_TYPE_Q4_K);
    CHECK(qd_type_from_name("iq4_ks") == GGML_TYPE_IQ4_KS);
    CHECK(qd_type_from_name("iq4_xxs") == GGML_TYPE_COUNT);
    CHECK(!qd_type_is_candidate(GGML_TYPE_Q8_K));
    CHECK(!qd_type_is_candidate(GGML_TYPE_Q8_1));
    CHECK(qd_type_is_candidate(GGML_TYPE_IQ2_KS));
    CHECK(qd_type_is_candidate(GGML_TYPE_F32));
}

// every candidate type: real quantizer round trip, row-local, and F32/F16 (near) identity
static void test_roundtrip_and_locality() {
    const int64_t k = 512, rows = 64;
    const qd_tensor_info t = make_info("rt.weight", k, rows, 1);
    qd_memory_reader src(t, make_weights(rows, k, 1));
    const common_imatrix im = make_imatrix(t, 2);
    const qd_importance imp = qd_importance_build(&im, t, 0);

    std::vector<ggml_type> types;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        if (qd_type_is_candidate((ggml_type) i)) {
            types.push_back((ggml_type) i);
        }
    }
    types.push_back(GGML_TYPE_Q8_K);

    qd_params p;
    p.qtypes = types;
    p.n_threads = 4;
    const auto ex = qd_exact_tensor(src, imp, p);
    for (size_t q = 0; q < types.size(); ++q) {
        const auto & r = ex.types[q];
        const ggml_type ty = types[q];
        if (ty == GGML_TYPE_Q8_K) {
            CHECK(r.status == QD_STATUS_UNSUPPORTED);
            continue;
        }
        if (r.status != QD_STATUS_OK) {
            fprintf(stderr, "  %s: %s\n", ggml_type_name(ty), qd_status_name(r.status));
            CHECK(r.status == QD_STATUS_INCOMPATIBLE);
            continue;
        }
        const double rel = r.unweighted.rel.value;
        if (ty == GGML_TYPE_F32) {
            CHECK(rel == 0.0);
        } else if (ty == GGML_TYPE_F16 || ty == GGML_TYPE_BF16) {
            CHECK(rel < 1e-4);
        } else {
            // even 1-bit types must reconstruct something; 8-bit types must be close
            if (!(rel > 0.0 && rel < 1.0)) {
                fprintf(stderr, "  %s: relative error %g\n", ggml_type_name(ty), rel);
            }
            CHECK(rel > 0.0 && rel < 1.0);
            if (ty == GGML_TYPE_Q8_0) {
                CHECK(rel < 1e-3);
            }
        }
    }

    for (const auto & l : qd_check_locality(src, imp, types, 4, 3)) {
        if (l.status == QD_STATUS_OK && !l.row_local) {
            fprintf(stderr, "  %s is not row-local\n", ggml_type_name(l.type));
        }
        CHECK(l.status != QD_STATUS_OK || l.row_local);
    }
}

static void test_status() {
    const qd_tensor_info t = make_info("odd.weight", 200, 8, 1);
    qd_memory_reader src(t, make_weights(8, 200, 3));
    qd_params p;
    p.qtypes = { GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ2_XXS, GGML_TYPE_F16 };
    const qd_importance imp = qd_importance_build(nullptr, t, 0);
    CHECK(!imp.has_imatrix);
    const auto r = qd_exact_tensor(src, imp, p);
    CHECK(r.types[0].status == QD_STATUS_INCOMPATIBLE); // 200 % 256
    CHECK(r.types[1].status == QD_STATUS_INCOMPATIBLE); // 200 % 32
    CHECK(r.types[2].status == QD_STATUS_INCOMPATIBLE);
    CHECK(r.types[3].status == QD_STATUS_OK);

    const qd_tensor_info t2 = make_info("even.weight", 256, 8, 1);
    qd_memory_reader src2(t2, make_weights(8, 256, 4));
    const auto r2 = qd_exact_tensor(src2, qd_importance_build(nullptr, t2, 0), p);
    CHECK(r2.types[2].status == QD_STATUS_REQUIRES_IMATRIX);
    CHECK(r2.types[0].status == QD_STATUS_OK);
    // without an imatrix the weighted metric is the unweighted one
    CHECK(near(r2.types[0].weighted[QD_WEIGHTING_MASS].rel.value, r2.types[0].unweighted.rel.value, 1e-9));
}

static void test_imatrix_indexing() {
    // dense: quantizer weights are sums / count
    {
        const qd_tensor_info t = make_info("d.weight", 64, 4, 1);
        const common_imatrix im = make_imatrix(t, 5);
        const auto imp = qd_importance_build(&im, t, 0);
        CHECK(imp.has_imatrix && imp.has_counts);
        const auto & e = im.entries.at("d.weight");
        for (int j = 0; j < 64; ++j) {
            CHECK(near(imp.quant_w[j], e.sums[j] / e.counts[0], 1e-6));
        }
        CHECK(imp.w[QD_WEIGHTING_MASS][0] == 1.0);
    }
    // merged experts: per-expert slices, routing weights and a missing expert
    {
        const int64_t k = 32, nex = 4, used = 2;
        const qd_tensor_info t = make_info("m.weight", k, 8, nex);
        const common_imatrix im = make_imatrix(t, 6, 2);
        const auto imp = qd_importance_build(&im, t, used);
        const auto & e = im.entries.at("m.weight");
        CHECK(imp.n_missing == 1 && imp.expert_missing[2] == 1);
        for (int64_t x = 0; x < nex; ++x) {
            for (int64_t j = 0; j < k; ++j) {
                const float want = x == 2 ? 1.0f : e.sums[x*k + j] / e.counts[x];
                CHECK(near(imp.quant_w[x*k + j], want, 1e-6));
            }
        }
        double sum_count = 0;
        for (auto c : e.counts) sum_count += c;
        const double n_tok = sum_count / used;
        CHECK(near(imp.w[QD_WEIGHTING_MASS][1], e.counts[1] / n_tok, 1e-12));
        // the missing expert gets the smallest observed count, never zero
        CHECK(near(imp.w[QD_WEIGHTING_MASS][2], e.counts[0] / n_tok, 1e-12));
        CHECK(near(imp.w[QD_WEIGHTING_UNIFORM][0], (double) used / nex, 1e-12));
        double act_missing = 0;
        for (int64_t j = 0; j < k; ++j) act_missing += imp.act[2*k + j];
        CHECK(act_missing > 0);
    }
    // wrong size -> treated as missing, with a note
    {
        const qd_tensor_info t = make_info("s.weight", 64, 4, 1);
        common_imatrix im = make_imatrix(t, 7);
        im.entries["s.weight"].sums.resize(32);
        const auto imp = qd_importance_build(&im, t, 0);
        CHECK(!imp.has_imatrix && !imp.note.empty());
    }
}

// sampled estimates converge to the exact values, the ranking stabilizes, and sampling is deterministic
static void test_convergence(bool moe) {
    const int64_t k = 512, nex = moe ? 8 : 1, rpe = moe ? 256 : 2048;
    const qd_tensor_info t = make_info(moe ? "moe.weight" : "dense.weight", k, rpe, nex);
    qd_memory_reader src(t, make_weights(rpe*nex, k, moe ? 8 : 9));
    const common_imatrix im = make_imatrix(t, 10);
    const qd_importance imp = qd_importance_build(&im, t, moe ? 2 : 0);

    qd_params p;
    p.qtypes    = { GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_KS, GGML_TYPE_Q8_0 };
    p.n_threads = 4;
    p.adaptive  = false;
    p.n_max     = 1024;
    const auto ex = qd_exact_tensor(src, imp, p);

    const int n_seeds = 8;
    std::vector<double> err_first(n_seeds), err_last(n_seeds);
    int covered = 0, total = 0;
    for (int s = 0; s < n_seeds; ++s) {
        p.seed = 100 + s;
        const auto r = qd_estimate_tensor(src, imp, p);
        CHECK(r.units_sampled <= p.n_max + (int64_t) r.n_strata);
        CHECK(r.rounds.size() >= 3);
        // scan and sample byte accounting
        CHECK(r.bytes_scan == (uint64_t) t.n_rows() * k * sizeof(float));
        CHECK(r.bytes_sample == (uint64_t) r.units_sampled * k * sizeof(float));
        if (moe) {
            CHECK(r.experts_sampled == nex);
        }
        double e0 = 0, e1 = 0;
        for (size_t q = 0; q < p.qtypes.size(); ++q) {
            const double x = ex.types[q].weighted[QD_WEIGHTING_MASS].rel.value;
            e0 = std::max(e0, std::fabs(r.rounds.front().rel[q] - x) / x);
            e1 = std::max(e1, std::fabs(r.rounds.back().rel[q] - x) / x);
            const auto & m = r.types[q].weighted[QD_WEIGHTING_MASS];
            covered += std::fabs(m.rel.value - x) <= 3.0 * m.rel.se;
            total++;
            // the ranking of well-separated types is exact at the end
            CHECK(r.types[q].rank == ex.types[q].rank);
            // total D and R agree with the known signal
            CHECK(near(m.total.value, m.rel.value * r.signal[QD_WEIGHTING_MASS], 1e-9));
        }
        err_first[s] = e0;
        err_last[s]  = e1;
    }
    std::sort(err_first.begin(), err_first.end());
    std::sort(err_last.begin(), err_last.end());
    fprintf(stderr, "  %s: median max rel err first round %.4f, last round %.4f, 3-SE coverage %d/%d\n", t.name.c_str(),
            err_first[n_seeds/2], err_last[n_seeds/2], covered, total);
    CHECK(err_last[n_seeds/2] < err_first[n_seeds/2]);
    CHECK(err_last[n_seeds/2] < 0.02);
    CHECK(covered >= total * 8 / 10);

    // determinism
    p.seed = 42;
    const auto a = qd_estimate_tensor(src, imp, p);
    const auto b = qd_estimate_tensor(src, imp, p);
    p.seed = 43;
    const auto c = qd_estimate_tensor(src, imp, p);
    bool same = a.units_sampled == b.units_sampled, differ = false;
    for (size_t q = 0; q < p.qtypes.size(); ++q) {
        same   = same && a.types[q].weighted[0].rel.value == b.types[q].weighted[0].rel.value;
        differ = differ || a.types[q].weighted[0].rel.value != c.types[q].weighted[0].rel.value;
    }
    CHECK(same);
    CHECK(differ);

    // budget and early stopping
    p.adaptive = true;
    p.n_max    = 64;
    p.rse_target = 1e-9;
    p.rank_z   = 0;
    const auto d = qd_estimate_tensor(src, imp, p);
    CHECK(d.units_sampled <= std::max<int64_t>(64, p.min_per_expert * nex) + d.n_strata);
    p.n_max    = 4096;
    p.rse_target = 0.5;
    const auto e = qd_estimate_tensor(src, imp, p);
    CHECK(e.rounds.size() == 1);

    // without row statistics the expansion estimator still converges
    p.adaptive  = false;
    p.n_max     = 1024;
    p.row_stats = false;
    p.seed      = 7;
    const auto f = qd_estimate_tensor(src, imp, p);
    CHECK(f.estimator == "expansion");
    for (size_t q = 0; q < p.qtypes.size(); ++q) {
        const double x = ex.types[q].weighted[QD_WEIGHTING_MASS].rel.value;
        // the expansion estimator is noisier; check it against its own standard error
        const auto & m = f.types[q].weighted[QD_WEIGHTING_MASS].rel;
        CHECK(std::fabs(m.value - x) <= 4.0 * m.se);
        CHECK(std::fabs(m.value - x) / x < 0.25);
    }
}

// GGUF reader: byte offsets, F16 conversion and row extraction of a merged expert tensor, identity fields
static void test_gguf_reader() {
    const char * path = "test-quant-damage.gguf";
    const char * path2 = "test-quant-damage-2.gguf";
    const int64_t k = 64, rpe = 3, nex = 2;
    std::vector<float> data(k*rpe*nex);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = 0.001f * (float) i - 0.1f;
    }

    ggml_init_params ip = { 1 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 5);
    ggml_set_name(a, "pad.weight");
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, rpe, nex);
    ggml_set_name(b, "blk.0.ffn_down_exps.weight");
    memset(a->data, 0, ggml_nbytes(a));
    ggml_fp32_to_fp16_row(data.data(), (ggml_fp16_t *) b->data, (int64_t) data.size());

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "test");
    gguf_set_val_u32(g, "test.expert_used_count", 1);
    gguf_add_tensor(g, a);
    gguf_add_tensor(g, b);
    CHECK(gguf_write_to_file(g, path, false));
    gguf_set_val_u32(g, "test.expert_used_count", 2);
    CHECK(gguf_write_to_file(g, path2, false));
    gguf_free(g);
    ggml_free(ctx);

    qd_gguf_model model;
    std::string err;
    CHECK(model.open(path, err));
    CHECK(model.arch == "test" && model.n_expert_used == 1);
    CHECK(model.n_files == 1 && model.total_file_size > 0);
    CHECK(model.metadata_hash != 0);
    for (const auto & t : model.tensors()) {
        CHECK(t.split == 0);
        CHECK(t.data_bytes == ggml_row_size(t.type, t.ne[0]) * t.ne[1] * t.ne[2] * t.ne[3]);
    }
    qd_gguf_model model2;
    CHECK(model2.open(path, err));
    CHECK(model2.metadata_hash == model.metadata_hash);
    qd_gguf_model model3;
    CHECK(model3.open(path2, err));
    CHECK(model3.metadata_hash != model.metadata_hash);
    CHECK(model3.n_expert_used == 2);

    auto r = model.reader("blk.0.ffn_down_exps.weight");
    CHECK(r != nullptr);
    if (r) {
        CHECK(r->info().n_expert() == nex && r->info().rows_per_expert() == rpe);
        std::vector<float> row(k);
        const int64_t g_row = 1*rpe + 2; // expert 1, row 2
        CHECK(r->read_rows(g_row, 1, row.data()));
        for (int64_t j = 0; j < k; ++j) {
            CHECK(near(row[j], data[g_row*k + j], 1e-3));
        }
        CHECK(!r->read_rows(rpe*nex, 1, row.data()));
    }
    CHECK(model.reader("missing") == nullptr);
    std::remove(path);
    std::remove(path2);
}

// file fingerprint: stable, content-sensitive
static void test_imatrix_means() {
    const char * path = "test-quant-damage-imatrix.gguf";
    for (int mode = 0; mode < 4; ++mode) {
        ggml_context * ctx = ggml_init({ 1 << 20, nullptr, false });
        gguf_context * g = gguf_init_empty();
        ggml_tensor * sums = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        ggml_tensor * counts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
        ggml_set_name(sums, "test.weight.in_sum2");
        ggml_set_name(counts, "test.weight.counts");
        for (int j = 0; j < 8; ++j) { ((float *) sums->data)[j] = 10.0f + j; }
        ((float *) counts->data)[0] = 2.0f;
        ((float *) counts->data)[1] = 3.0f;
        gguf_add_tensor(g, sums);
        gguf_add_tensor(g, counts);
        if (mode != 0) {
            ggml_tensor * means = ggml_new_tensor_2d(ctx, mode == 3 ? GGML_TYPE_F16 : GGML_TYPE_F32, mode == 2 ? 3 : 4, 2);
            ggml_set_name(means, "test.weight.in_sum");
            memset(means->data, 0, ggml_nbytes(means));
            if (mode == 1) {
                for (int j = 0; j < 8; ++j) { ((float *) means->data)[j] = -2.0f + j; }
            }
            gguf_add_tensor(g, means);
        }
        CHECK(gguf_write_to_file(g, path, false));
        gguf_free(g);
        ggml_free(ctx);
        common_imatrix im;
        const bool loaded = common_imatrix_load(path, im);
        CHECK(loaded == (mode < 2));
        if (loaded) {
            const auto & e = im.entries.at("test.weight");
            CHECK(e.counts == std::vector<int64_t>({ 2, 3 }));
            CHECK(e.sums.size() == 8 && e.sums[7] == 17.0f);
            CHECK(e.activations.size() == (mode == 1 ? 8 : 0));
            if (mode == 1) { CHECK(e.activations[0] == -2.0f && e.activations[7] == 5.0f); }
        }
    }
    std::remove(path);
}

static void test_hash_file() {
    const char * path = "test-quant-damage-hash.bin";
    {
        FILE * f = fopen(path, "wb");
        CHECK(f && fwrite("abc", 1, 3, f) == 3);
        fclose(f);
    }
    uint64_t h1 = 0, b1 = 0, h2 = 0, b2 = 0;
    CHECK(qd_hash_file(path, h1, b1));
    CHECK(qd_hash_file(path, h2, b2));
    CHECK(h1 == h2 && b1 == 3);
    CHECK(h1 == qd_fnv1a("abc", 3));
    {
        FILE * f = fopen(path, "wb");
        CHECK(f && fwrite("abd", 1, 3, f) == 3);
        fclose(f);
    }
    CHECK(qd_hash_file(path, h2, b2));
    CHECK(h2 != h1 && b2 == 3);
    uint64_t tmp;
    CHECK(!qd_hash_file("test-quant-damage-missing.bin", tmp, b2));
    std::remove(path);
}

// a row with one extreme weight dominates the damage of block-scale types (MXFP4: err ~ blockmax^2);
// the certainty rows and the per-expert minimum must always measure it
static void test_certainty_rows() {
    const int64_t k = 512, rows = 4096;
    std::vector<float> w = make_weights(rows, k, 11);
    const int64_t hot0 = 7, hot1 = 3001;
    w[hot0*k + 5]  *= 2000.0f;
    w[hot1*k + 77] *= 2000.0f;

    const qd_tensor_info t = make_info("hot.weight", k, rows, 1);
    qd_memory_reader src(t, w);
    const common_imatrix im = make_imatrix(t, 12);
    const qd_importance imp = qd_importance_build(&im, t, 0);

    qd_params p;
    p.qtypes    = { GGML_TYPE_MXFP4, GGML_TYPE_Q4_K };
    p.n_threads = 4;
    p.adaptive  = false;
    p.n_max     = 64;
    const auto ex = qd_exact_tensor(src, imp, p);
    const double x = ex.types[0].weighted[QD_WEIGHTING_MASS].rel.value;

    // premise: the two hot rows carry a large share of the Q4_0 damage
    const double Dtot = ex.types[0].weighted[QD_WEIGHTING_MASS].total.value;
    double Dhot = 0.0;
    for (int64_t r : { hot0, hot1 }) {
        std::vector<float> one(w.begin() + r*k, w.begin() + (r + 1)*k);
        qd_tensor_info t1 = t; t1.ne[1] = 1;
        qd_memory_reader s1(t1, one);
        qd_params p1 = p; p1.qtypes = { GGML_TYPE_MXFP4 };
        Dhot += qd_exact_tensor(s1, imp, p1).types[0].weighted[QD_WEIGHTING_MASS].total.value;
    }
    fprintf(stderr, "  hot rows hold %.1f%% of D\n", 100.0 * Dhot / Dtot);
    CHECK(Dhot > 0.15 * Dtot);

    // the remaining error is the sampling error of the ordinary rows; the imatrix-aware MXFP4 exponent search makes their
    // damage more variable, so check it against the reported standard error as well as a loose absolute bound
    double max_err_on = 0.0, max_err_off = 0.0, max_z_on = 0.0;
    for (int s = 0; s < 8; ++s) {
        p.seed = 200 + s;
        p.certainty_factor = 8.0;
        const auto on = qd_estimate_tensor(src, imp, p);
        p.certainty_factor = 0.0;
        const auto off = qd_estimate_tensor(src, imp, p);
        const double eon = std::fabs(on.types[0].weighted[0].rel.value - x) / x;
        const double eoff = std::fabs(off.types[0].weighted[0].rel.value - x) / x;
        max_err_on = std::max(max_err_on, eon);
        max_err_off = std::max(max_err_off, eoff);
        max_z_on = std::max(max_z_on, std::fabs(on.types[0].weighted[0].rel.value - x) / on.types[0].weighted[0].rel.se);
    }
    fprintf(stderr, "  certainty rows: worst error %.3f%% on (%.2f s.e.), %.1f%% off\n", max_err_on*100.0, max_z_on, max_err_off*100.0);
    CHECK(max_err_on < 0.06);
    CHECK(max_z_on < 4.0);
    CHECK(max_err_off > 0.10);
}

// tie group aggregation: sums, uncertainty propagation and status reporting
static void test_tie_combine() {
    const int64_t k = 512, rows = 512;
    qd_params p;
    p.qtypes    = { GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XXS };
    p.n_threads = 4;
    p.adaptive  = false;
    p.n_max     = 256;
    p.seed      = 31;

    const qd_tensor_info tg = make_info("blk.0.ffn_gate.weight", k, rows, 1);
    const qd_tensor_info tu = make_info("blk.0.ffn_up.weight", k, rows, 1);
    std::vector<float> wg = make_weights(rows, k, 21);
    std::vector<float> wu = make_weights(rows, k, 23);
    for (auto & v : wu) {
        v *= 0.5f;
    }
    qd_memory_reader sg(tg, wg);
    qd_memory_reader su(tu, wu);
    const common_imatrix img = make_imatrix(tg, 22);
    const common_imatrix imu = make_imatrix(tu, 24);

    const auto rg = qd_estimate_tensor(sg, qd_importance_build(&img, tg, 0), p);
    const auto ru = qd_estimate_tensor(su, qd_importance_build(&imu, tu, 0), p);

    qd_tie_group grp{ "gateup", { tg.name, tu.name } };
    const auto comb = qd_tie_combine(grp, { rg, ru }, p.ci_z);

    CHECK(comb.types.size() == 2);
    const auto & q = comb.types[0];
    CHECK(q.status == QD_STATUS_OK);
    CHECK(q.bytes == rg.types[0].bytes + ru.types[0].bytes);
    for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
        CHECK(near(q.signal[w], rg.signal[w] + ru.signal[w], 1e-9));
        CHECK(near(q.weighted[w].total.value, rg.types[0].weighted[w].total.value + ru.types[0].weighted[w].total.value, 1e-9));
        const double se = std::sqrt(std::pow(rg.types[0].weighted[w].total.se, 2) + std::pow(ru.types[0].weighted[w].total.se, 2));
        CHECK(near(q.weighted[w].total.se, se, 1e-9));
        CHECK(near(q.weighted[w].rel.value, q.weighted[w].total.value / q.signal[w], 1e-9));
        CHECK(q.weighted[w].ci_lo <= q.weighted[w].rel.value && q.weighted[w].rel.value <= q.weighted[w].ci_hi);
    }
    // the members keep separate damages
    CHECK(comb.members[0].types[0].weighted[0].rel.value != comb.members[1].types[0].weighted[0].rel.value);
    CHECK(q.bpw == 8.0 * q.bytes / (rg.info.n_elements() + ru.info.n_elements()));

    // unsupported in one member: the group reports the member's rejection reason
    const auto rg2 = qd_estimate_tensor(sg, qd_importance_build(nullptr, tg, 0), p);
    const auto ru2 = qd_estimate_tensor(su, qd_importance_build(nullptr, tu, 0), p);
    const auto c2 = qd_tie_combine(grp, { rg2, ru2 }, p.ci_z);
    CHECK(c2.types[0].status == QD_STATUS_OK);
    CHECK(c2.types[1].status == QD_STATUS_REQUIRES_IMATRIX);
    CHECK(c2.types[1].note.find(tg.name) != std::string::npos);
}

int main() {
    test_estimators();
    test_permutation();
    test_types();
    test_status();
    test_imatrix_indexing();
    test_roundtrip_and_locality();
    test_gguf_reader();
    test_imatrix_means();
    test_hash_file();
    test_tie_combine();
    test_certainty_rows();
    test_convergence(false);
    test_convergence(true);

    if (n_failed) {
        fprintf(stderr, "%d checks failed\n", n_failed);
        return 1;
    }
    fprintf(stderr, "all checks passed\n");
    return 0;
}
