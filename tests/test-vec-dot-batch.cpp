// Batched expert dots must preserve the single-row result, including strided rows and tails.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const ggml_type types[] = {
    GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4,
};

static bool check_rows(ggml_type type, int n, const void * weights, size_t stride,
                       const void * const * acts, const char * pattern) {
    const auto batch = ggml_get_vec_dot_batch_rows(type);
    const auto dot = ggml_get_type_traits_cpu(type)->vec_dot;
    constexpr int rows = 3;
    constexpr float guard = -1234567.0f;
    for (int nr = 1; nr <= 4; ++nr) {
        float got[4][rows + 2], ref[4][rows + 2];
        float * dst[4];
        for (int r = 0; r < 4; ++r) {
            std::fill_n(got[r], rows + 2, guard);
            std::fill_n(ref[r], rows + 2, guard);
            dst[r] = got[r] + 1;
            if (r < nr) {
                for (int row = 0; row < rows; ++row) {
                    dot(n, &ref[r][row + 1], 0, (const char *) weights + row * stride, 0, acts[r], 0, 1);
                }
            }
        }
        batch(n, dst, weights, stride, acts, nr, rows);
        if (memcmp(got, ref, sizeof(got)) != 0) {
            for (int r = 0; r < 4; ++r) {
                for (int row = 0; row < rows + 2; ++row) {
                    if (memcmp(&got[r][row], &ref[r][row], sizeof(float)) != 0) {
                        fprintf(stderr, "%s n=%d nr=%d %s: row=%d act=%d ref=%a got=%a\n",
                                ggml_type_name(type), n, nr, pattern, row - 1, r, ref[r][row], got[r][row]);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static bool test_type(ggml_type type) {
    std::mt19937 rng(1234 + type);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const auto act_type = ggml_get_type_traits_cpu(type)->vec_dot_type;
    const auto quantize_act = ggml_get_type_traits_cpu(act_type)->from_float;
    const int block = ggml_blck_size(type);
    for (int n : {block, 3*block, 2560, 4096}) {
        const size_t stride = ggml_row_size(type, n) + 8;
        std::vector<uint8_t> weights(3*stride + 8);
        std::vector<uint8_t> activations[4];
        const void * acts[4];
        std::vector<float> data(n), imatrix(n, 1.0f);
        for (int pattern = 0; pattern < 3; ++pattern) {
            for (int row = 0; row < 3; ++row) {
                for (int j = 0; j < n; ++j) {
                    data[j] = pattern == 1 ? 0.0f : std::ldexp(dist(rng), (j / block + row) % 9 - 4);
                }
                // IQ2_XXS and IQ2_XS need an importance matrix; they have no CPU from_float function.
                ggml_quantize_chunk(type, data.data(), weights.data() + 8 + row*stride, 0, 1, n, imatrix.data());
            }
            for (int r = 0; r < 4; ++r) {
                activations[r].resize(ggml_row_size(act_type, n) + 8);
                for (int j = 0; j < n; ++j) {
                    data[j] = pattern == 1 ? 0.0f : dist(rng) * (r + 1);
                }
                // Q8_K is an activation format. ggml_quantize_chunk does not support it.
                quantize_act(data.data(), activations[r].data() + 8, n);
                if (pattern == 2) {
                    if (act_type == GGML_TYPE_Q8_0) {
                        auto * y = reinterpret_cast<block_q8_0 *>(activations[r].data() + 8);
                        for (int b = 0; b < n / QK8_0; ++b) {
                            for (int j = 0; j < QK8_0; ++j) {
                                y[b].qs[j] = (j + r) % 3 == 0 ? -128 : (j + r) % 3 == 1 ? 127 : 0;
                            }
                        }
                    } else {
                        auto * y = reinterpret_cast<block_q8_K *>(activations[r].data() + 8);
                        for (int b = 0; b < n / QK_K; ++b) {
                            for (int j = 0; j < QK_K; ++j) {
                                y[b].qs[j] = (j + r) % 3 == 0 ? -128 : (j + r) % 3 == 1 ? 127 : 0;
                            }
                            for (int j = 0; j < QK_K / 16; ++j) {
                                int sum = 0;
                                for (int k = 0; k < 16; ++k) { sum += y[b].qs[16*j + k]; }
                                y[b].bsums[j] = sum;
                            }
                        }
                    }
                }
            }
            // Activation pointers need not follow token order or share a contiguous allocation.
            const int order[] = {3, 0, 2, 1};
            for (int r = 0; r < 4; ++r) { acts[r] = activations[order[r]].data() + 8; }
            const char * name = pattern == 0 ? "random" : pattern == 1 ? "zero" : "signed-byte limits";
            if (!check_rows(type, n, weights.data() + 8, stride, acts, name)) { return false; }
        }
    }
    printf("%s: exact parity for 1-4 activation rows, padded weight rows and output guards\n", ggml_type_name(type));
    return true;
}

static bool test_mul_mat_id(ggml_type type, ggml_threadpool_t pool, int rows = 65) {
    constexpr int n = 256;
    constexpr int experts = 3;
    constexpr int top_k = 2;
    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int tokens : {1, 2, 5, 7}) {
        for (int mode = 0; mode < 4; ++mode) {
            const bool partial = mode % 2 != 0;
            const auto act_type = mode < 2 ? GGML_TYPE_F32 : ggml_get_type_traits_cpu(type)->vec_dot_type;
            const int channels = mode == 0 ? 1 : top_k;
            auto * ctx = ggml_init({8*1024*1024, nullptr, false});
            GGML_ASSERT(ctx);
            auto * w = ggml_new_tensor_3d(ctx, type, n, rows, experts);
            // Mode 0 uses ordinary chunks. The other modes also test the NUMA column split.
            if (mode != 0) { w->flags |= GGML_TENSOR_FLAG_NUMA_EXPERT; }
            auto * backing = ggml_new_tensor_3d(ctx, act_type, mode == 3 ? 2*n : n, channels, tokens);
            auto * a = mode == 3 ? ggml_view_3d(ctx, backing, n, channels, tokens, backing->nb[1], backing->nb[2], 0) : backing;
            auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, tokens);
            auto * out = partial ? ggml_mul_mat_id_partial(ctx, w, a, ids) : ggml_mul_mat_id(ctx, w, a, ids);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            std::vector<float> data(n), imatrix(n, 1.0f);
            for (int row = 0; row < rows*experts; ++row) {
                for (float & v : data) { v = dist(rng); }
                ggml_quantize_chunk(type, data.data(), (char *) w->data + row*w->nb[1], 0, 1, n, imatrix.data());
            }
            for (int t = 0; t < tokens; ++t) {
                for (int c = 0; c < channels; ++c) {
                    for (float & v : data) { v = dist(rng); }
                    auto * dst = (char *) a->data + c*a->nb[1] + t*a->nb[2];
                    ggml_get_type_traits_cpu(act_type)->from_float(data.data(), dst, n);
                }
                auto * id = (int32_t *) ((char *) ids->data + t*ids->nb[1]);
                id[0] = 0;
                id[1] = t == 1 ? 2 : 1;
                if (partial && t % 3 != 0) { id[1] = t % 3 == 1 ? -1 : experts; }
            }
            auto plan = ggml_graph_plan(graph, 4, pool);
            std::vector<uint8_t> scratch(plan.work_size);
            plan.work_data = scratch.data();
            plan.use_ref = true;
            GGML_ASSERT(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
            std::vector<uint8_t> ref(ggml_nbytes(out));
            memcpy(ref.data(), out->data, ref.size());
            memset(out->data, 0xa5, ref.size());
            plan.use_ref = false;
            GGML_ASSERT(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
            const bool equal = memcmp(ref.data(), out->data, ref.size()) == 0;
            ggml_free(ctx);
            if (!equal) {
                fprintf(stderr, "%s MUL_MAT_ID tokens=%d mode=%d: differs from per-row reference\n", ggml_type_name(type), tokens, mode);
                return false;
            }
        }
    }
    printf("%s: exact MUL_MAT_ID parity for broadcast, per-expert, strided and partial cases\n", ggml_type_name(type));
    return true;
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (argc == 3 && strcmp(argv[1], "--numa") == 0 && strcmp(argv[2], "tensors") == 0) {
        ggml_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--numa tensors]\n", argv[0]);
        return 1;
    }
    int failed = 0;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const auto type = (ggml_type) i;
        const bool covered = std::find(std::begin(types), std::end(types), type) != std::end(types);
        const bool available = ggml_get_vec_dot_batch_rows(type) != nullptr;
        if (available != (covered && ggml_cpu_has_avx2())) {
            fprintf(stderr, "Unexpected batched dispatch for type %d\n", i);
            ++failed;
        } else if (available && !test_type(type)) {
            ++failed;
        }
    }
    auto params = ggml_threadpool_params_default(4);
    auto * pool = ggml_threadpool_new(&params);
    GGML_ASSERT(pool);
    for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_IQ3_S, GGML_TYPE_Q4_0}) {
        if (!test_mul_mat_id(type, pool)) { ++failed; }
    }
    // More workers than small output tiles can produce empty ranges in the ordinary chunk path.
    if (!test_mul_mat_id(GGML_TYPE_Q4_K, pool, 5)) { ++failed; }
    ggml_threadpool_free(pool);
    ggml_quantize_free();
    if (!ggml_cpu_has_avx2()) { printf("No AVX2 batch kernels: all types use the existing fallback\n"); }
    return failed ? 1 : 0;
}
