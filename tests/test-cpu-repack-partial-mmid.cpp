// test-cpu-repack-partial-mmid: partial MUL_MAT_ID on the CPU repack extra buffer type.
//
// test-backend-ops allocates the weights in the default CPU buffer, so the repack buffer is never
// exercised there. This test uploads the same quantized weights to the plain CPU buffer and to the
// CPU_REPACK buffer and checks that:
//   - rows of the out-of-range ids of a partial MUL_MAT_ID are exactly zero in both outputs
//   - the other rows of the repack output match the plain CPU output within a small NMSE
//     (the repack kernels round differently)
//   - a non-partial MUL_MAT_ID through the repack buffer still matches (control)
// Both outputs are also compared against a float64 reference built from the dequantized weights.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

static constexpr int64_t K        = 512;
static constexpr int64_t N        = 256;
static constexpr int64_t N_EXPERT = 8;
static constexpr int64_t N_USED   = 4;
static constexpr double  NMSE_MAX = 1e-6;

struct cpu_ctx {
    ggml_backend_dev_t         dev         = nullptr;
    ggml_backend_t             backend     = nullptr;
    ggml_backend_buffer_type_t repack_buft = nullptr;
};

struct mmid_case {
    ggml_type type;
    bool      partial;
    int64_t   n_tokens;
    int64_t   b_ne1;             // 1 (broadcast) or N_USED
    std::vector<uint8_t> q;      // quantized weights [K, N, N_EXPERT]
    std::vector<int32_t> ids;    // [N_USED, n_tokens]
    std::vector<float>   b;      // [K, b_ne1, n_tokens]
};

// Runs one MUL_MAT_ID with the weights in the given buffer type. Returns false when the device refuses
// the op, or when the repack buffer does not take the tensor (then repack_taken is false).
static bool run_mmid(const cpu_ctx & c, const mmid_case & mc, bool repack, std::vector<float> & out,
                     bool & repack_taken) {
    repack_taken = false;

    ggml_init_params ip_w = { /*.mem_size =*/ ggml_tensor_overhead() * 4, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_init_params ip_g = { /*.mem_size =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx_w = ggml_init(ip_w);
    ggml_context * ctx_g = ggml_init(ip_g);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx_w, mc.type, K, N, N_EXPERT);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, K, mc.b_ne1, mc.n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx_g, GGML_TYPE_I32, N_USED, mc.n_tokens);
    ggml_tensor * dst = mc.partial ? ggml_mul_mat_id_partial(ctx_g, as, b, ids) : ggml_mul_mat_id(ctx_g, as, b, ids);

    ggml_backend_buffer_type_t w_buft = repack ? c.repack_buft : ggml_backend_dev_buffer_type(c.dev);
    ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, w_buft);
    ggml_backend_buffer_t g_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_g, ggml_backend_dev_buffer_type(c.dev));

    bool ok = false;
    // the repack buffer leaves tensor->extra NULL for a type it does not repack, and its set_tensor
    // dereferences it, so check before the upload
    if (w_buf && (!repack || as->extra != nullptr)) {
        repack_taken = true;
        ggml_backend_tensor_set(as, mc.q.data(), 0, mc.q.size());
        ggml_backend_tensor_set(b, mc.b.data(), 0, mc.b.size() * sizeof(float));
        ggml_backend_tensor_set(ids, mc.ids.data(), 0, mc.ids.size() * sizeof(int32_t));

        ok = ggml_backend_dev_supports_op(c.dev, dst);
        if (ok) {
            ggml_cgraph * gf = ggml_new_graph(ctx_g);
            ggml_build_forward_expand(gf, dst);
            ok = ggml_backend_graph_compute(c.backend, gf) == GGML_STATUS_SUCCESS;
        }
        if (ok) {
            out.resize(ggml_nelements(dst));
            ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));
        }
    }

    if (g_buf) ggml_backend_buffer_free(g_buf);
    if (w_buf) ggml_backend_buffer_free(w_buf);
    ggml_free(ctx_w);
    ggml_free(ctx_g);
    return ok;
}

// dst layout: [N, N_USED, n_tokens]; the row of slot s of token t starts at (t*N_USED + s)*N
static bool is_oor(int32_t id) {
    return id < 0 || id >= N_EXPERT;
}

// NMSE of got vs ref over the rows that are not skipped; counts the non-zero values in the skipped rows
static double nmse_rows(const std::vector<float> & ref, const std::vector<float> & got, const mmid_case & mc,
                        bool skip_oor, int64_t & n_zero_rows, int64_t & n_nonzero_skipped) {
    double num = 0.0, den = 0.0;
    n_zero_rows = 0;
    n_nonzero_skipped = 0;
    for (int64_t t = 0; t < mc.n_tokens; ++t) {
        for (int64_t s = 0; s < N_USED; ++s) {
            const float * r = got.data() + (t*N_USED + s)*N;
            const float * e = ref.data() + (t*N_USED + s)*N;
            if (skip_oor && is_oor(mc.ids[t*N_USED + s])) {
                n_zero_rows++;
                for (int64_t i = 0; i < N; ++i) {
                    n_nonzero_skipped += r[i] != 0.0f;
                }
                continue;
            }
            for (int64_t i = 0; i < N; ++i) {
                const double d = (double) r[i] - (double) e[i];
                num += d*d;
                den += (double) e[i]*e[i];
            }
        }
    }
    return den > 0.0 ? num/den : num;
}

// float64 reference from the dequantized weights; out-of-range slots are zero
static std::vector<float> reference(const mmid_case & mc, const std::vector<float> & w_deq) {
    std::vector<float> out(N*N_USED*mc.n_tokens, 0.0f);
    for (int64_t t = 0; t < mc.n_tokens; ++t) {
        for (int64_t s = 0; s < N_USED; ++s) {
            const int32_t id = mc.ids[t*N_USED + s];
            if (is_oor(id)) {
                continue;
            }
            const int64_t i11 = s % mc.b_ne1;
            const float * bv = mc.b.data() + (t*mc.b_ne1 + i11)*K;
            for (int64_t n = 0; n < N; ++n) {
                const float * wv = w_deq.data() + (id*N + n)*K;
                double acc = 0.0;
                for (int64_t k = 0; k < K; ++k) {
                    acc += (double) wv[k] * (double) bv[k];
                }
                out[(t*N_USED + s)*N + n] = (float) acc;
            }
        }
    }
    return out;
}

int main() {
    ggml_backend_load_all();

    cpu_ctx c;
    c.dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!c.dev) {
        printf("SKIP: no CPU device\n");
        return 0;
    }
    c.backend = ggml_backend_dev_init(c.dev, nullptr);
    if (!c.backend) {
        printf("FAIL: cannot init CPU backend\n");
        return 1;
    }

    // find the repack extra buffer type through the CPU reg proc address (as llama-model.cpp does)
    {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(c.dev);
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        if (get_extra) {
            ggml_backend_buffer_type_t * bufts = get_extra(c.dev);
            while (bufts && *bufts) {
                if (strcmp(ggml_backend_buft_name(*bufts), "CPU_REPACK") == 0) {
                    c.repack_buft = *bufts;
                }
                ++bufts;
            }
        }
    }
    if (!c.repack_buft) {
        printf("SKIP: no CPU_REPACK extra buffer type on this build\n");
        return 0;
    }

    const ggml_type types[] = {
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4,
    };
    const int64_t n_tokens_list[] = { 1, 4, 33 };

    int n_fail = 0, n_pass = 0, n_skip = 0;

    for (ggml_type type : types) {
        const char * tname = ggml_type_name(type);
        std::mt19937 rng(1234 + (int) type);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> w(K * N * N_EXPERT);
        for (auto & x : w) x = dist(rng);

        const size_t row_size = ggml_row_size(type, K);
        std::vector<uint8_t> q(row_size * N * N_EXPERT);
        ggml_quantize_chunk(type, w.data(), q.data(), 0, N * N_EXPERT, K, nullptr);
        std::vector<float> w_deq(K * N * N_EXPERT);
        ggml_get_type_traits(type)->to_float(q.data(), w_deq.data(), (int64_t) w_deq.size());

        for (int64_t n_tokens : n_tokens_list) {
            for (int partial_i = 0; partial_i < 2; ++partial_i) {
                for (int64_t b_ne1 : { (int64_t) 1, N_USED }) {
                    mmid_case mc;
                    mc.type     = type;
                    mc.partial  = partial_i == 1;
                    mc.n_tokens = n_tokens;
                    mc.b_ne1    = b_ne1;
                    mc.q        = q;

                    // ids: each token routes to distinct experts (the row map of MUL_MAT_ID holds at most one
                    // row per token and expert); the partial case sets the first slot of every token to -1 and the last one to n_expert
                    mc.ids.resize(N_USED * n_tokens);
                    for (int64_t t = 0; t < n_tokens; ++t) {
                        std::vector<int32_t> perm(N_EXPERT);
                        for (int32_t e = 0; e < N_EXPERT; ++e) {
                            perm[e] = e;
                        }
                        std::shuffle(perm.begin(), perm.end(), rng);
                        for (int64_t s = 0; s < N_USED; ++s) {
                            mc.ids[t*N_USED + s] = perm[s];
                        }
                    }
                    if (mc.partial) {
                        for (int64_t t = 0; t < n_tokens; ++t) {
                            mc.ids[t*N_USED + 0]          = -1;
                            mc.ids[t*N_USED + N_USED - 1] = (int32_t) N_EXPERT;
                        }
                    }

                    mc.b.resize(K * b_ne1 * n_tokens);
                    for (auto & x : mc.b) x = dist(rng);

                    printf("%-8s %-7s tokens=%-3lld b_ne1=%lld: ", tname, mc.partial ? "partial" : "full",
                           (long long) n_tokens, (long long) b_ne1);
                    fflush(stdout);

                    std::vector<float> ref_plain, got, ref_f64;
                    bool taken_plain = false, taken_repack = false;
                    const bool ok_plain = run_mmid(c, mc, false, ref_plain, taken_plain);
                    const bool ok_repack = run_mmid(c, mc, true, got, taken_repack);

                    if (!ok_plain) {
                        printf("SKIP (plain CPU refused the op)\n");
                        n_skip++;
                        continue;
                    }
                    if (!taken_repack) {
                        printf("SKIP (CPU_REPACK does not take %s on this CPU)\n", tname);
                        n_skip++;
                        continue;
                    }
                    if (!ok_repack) {
                        printf("FAIL (repack device refuses the partial op)\n");
                        n_fail++;
                        continue;
                    }

                    ref_f64 = reference(mc, w_deq);
                    int64_t n_zero_rows = 0, n_bad = 0, dummy = 0;
                    const double nmse_rp   = nmse_rows(ref_plain, got, mc, mc.partial, n_zero_rows, n_bad);
                    const double nmse_pf   = nmse_rows(ref_f64, ref_plain, mc, mc.partial, dummy, dummy);
                    const double nmse_rf   = nmse_rows(ref_f64, got, mc, mc.partial, dummy, dummy);
                    const bool pass = n_bad == 0 && nmse_rp < NMSE_MAX;
                    printf("%s repack-vs-cpu NMSE=%.3e zero_rows=%lld nonzero_in_zero_rows=%lld "
                           "(cpu-vs-f64 %.2e, repack-vs-f64 %.2e)\n",
                           pass ? "PASS" : "FAIL", nmse_rp, (long long) n_zero_rows, (long long) n_bad,
                           nmse_pf, nmse_rf);
                    pass ? n_pass++ : n_fail++;
                }
            }
        }
    }

    ggml_backend_free(c.backend);

    printf("\n%d passed, %d failed, %d skipped\n", n_pass, n_fail, n_skip);
    return n_fail == 0 ? 0 : 1;
}
