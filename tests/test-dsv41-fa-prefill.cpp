// DeepSeek V4.1 prefill attention shapes: 512-wide heads, one KV head, sinks, a raw sliding window plus a top-k of
// compressed rows per query. Compares the dense masked launch (sparse on RDNA4 with n_kv_max), the indexed launch and the
// CPU reference.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int64_t D      = 512;
constexpr int64_t H      = 64;
constexpr int64_t W      = 128;  // sliding window
constexpr int64_t S      = 512;  // selected compressed rows
constexpr int64_t N_COMP = 4096; // with n_raw = n_tok + 256: n_kv a multiple of 256, above the sparse threshold

struct problem {
    int64_t n_tok;
    int64_t n_raw;
    std::vector<float> q, sinks;
    std::vector<ggml_fp16_t> k;            // [D, n_raw + N_COMP]: raw rows, then compressed rows
    std::vector<std::vector<int32_t>> sel; // per query: compressed rows (0..N_COMP)
    int64_t n_kv() const { return n_raw + N_COMP; }
    int64_t qrow(int64_t t) const { return n_raw - n_tok + t; }
};

problem make_problem(int64_t n_tok, uint32_t seed) {
    problem p;
    p.n_tok = n_tok;
    p.n_raw = n_tok + 2*W;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    p.q.resize(D*n_tok*H);
    for (auto & x : p.q) { x = nd(rng); }
    p.k.resize(D*p.n_kv());
    for (auto & x : p.k) { x = ggml_fp32_to_fp16(nd(rng)); }
    p.sinks.resize(H);
    for (auto & x : p.sinks) { x = nd(rng); }
    std::vector<int32_t> all(N_COMP);
    std::iota(all.begin(), all.end(), 0);
    p.sel.resize(n_tok);
    for (int64_t t = 0; t < n_tok; ++t) {
        // neighbouring queries share most of their selection, as in the model
        if (t % 8 == 0) {
            std::shuffle(all.begin(), all.end(), rng);
        } else {
            std::swap(all[rng() % S], all[S + rng() % (N_COMP - S)]);
        }
        p.sel[t].assign(all.begin(), all.begin() + S);
    }
    return p;
}

// mode 0: dense mask, no n_kv_max; 1: dense mask with n_kv_max = W + S; 2: indexed
std::vector<float> run(ggml_backend_t backend, const problem & p, int mode) {
    ggml_init_params params = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);

    const int64_t n_cols = mode == 2 ? S + W : p.n_kv();
    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, p.n_tok, H, 1);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, p.n_kv(), 1, 1);
    ggml_tensor * mask  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_cols, p.n_tok, 1, 1);
    ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * idx   = mode == 2 ? ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_cols, p.n_tok, 1) : nullptr;

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, k, mask, 1.0f/std::sqrt((float) D), 0.0f, 0.0f);
    if (idx) {
        ggml_flash_attn_ext_set_indices(out, idx);
    }
    ggml_flash_attn_ext_add_sinks(out, sinks);
    ggml_flash_attn_ext_set_n_kv_max(out, mode == 0 ? 0 : (int32_t) (S + W));
    ggml_prec_set_acc(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    std::vector<ggml_fp16_t> m(n_cols*p.n_tok, ggml_fp32_to_fp16(-INFINITY));
    std::vector<int32_t> ix(idx ? n_cols*p.n_tok : 0);
    for (int64_t t = 0; t < p.n_tok; ++t) {
        ggml_fp16_t * row = m.data() + t*n_cols;
        if (mode == 2) {
            for (int64_t j = 0; j < S; ++j) {
                ix[t*n_cols + j] = (int32_t) (p.n_raw + p.sel[t][j]);
                row[j] = ggml_fp32_to_fp16(0.0f);
            }
            for (int64_t j = 0; j < W; ++j) {
                ix[t*n_cols + S + j] = (int32_t) (p.qrow(t) - W + 1 + j);
                row[S + j] = ggml_fp32_to_fp16(0.0f);
            }
        } else {
            for (int64_t r = p.qrow(t) - W + 1; r <= p.qrow(t); ++r) {
                row[r] = ggml_fp32_to_fp16(0.0f);
            }
            for (int32_t c : p.sel[t]) {
                row[p.n_raw + c] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
    ggml_backend_tensor_set(q, p.q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, p.k.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(sinks, p.sinks.data(), 0, ggml_nbytes(sinks));
    if (idx) {
        ggml_backend_tensor_set(idx, ix.data(), 0, ggml_nbytes(idx));
    }

    std::vector<float> res;
    if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        res.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

bool check(const char * name, const std::vector<float> & ref, const std::vector<float> & got) {
    if (got.size() != ref.size()) {
        printf("  %-24s FAILED to compute\n", name);
        return false;
    }
    double err = 0.0, norm = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        err += d*d;
        norm += (double) ref[i]*ref[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    const double nmse = err / std::max(norm, 1e-30);
    const bool ok = nmse < 1e-5;
    printf("  %-24s NMSE %.3e max abs %.3e %s\n", name, nmse, max_abs, ok ? "ok" : "FAILED");
    return ok;
}

}

int main(int argc, char ** argv) {
    llama_backend_init();
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t gpu = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (!cpu || !gpu) {
        fprintf(stderr, "needs a CPU and a GPU backend\n");
        return 1;
    }
    bool passed = true;
    for (int64_t n_tok : { 512, 1024 }) {
        for (uint32_t seed : { 1u, 2u }) {
            const problem p = make_problem(n_tok, seed);
            printf("%s: n_tok %" PRId64 " n_kv %" PRId64 " seed %u\n", ggml_backend_name(gpu), n_tok, p.n_kv(), seed);
            const auto ref = run(cpu, p, 0);
            passed &= check("gpu dense", ref, run(gpu, p, 0));
            passed &= check("gpu dense n_kv_max", ref, run(gpu, p, 1));
            passed &= check("gpu indexed", ref, run(gpu, p, 2));
        }
    }
    GGML_UNUSED(argc); GGML_UNUSED(argv);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    llama_backend_free();
    printf("%s\n", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}
