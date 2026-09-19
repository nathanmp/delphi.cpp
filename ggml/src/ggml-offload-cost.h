#pragma once

// delphi.cpp: backend-neutral cost model for operations whose weights and compute can use different backends.

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_offload_cost_input {
    uint64_t operations;
    uint64_t weight_h2d_bytes;
    uint64_t cpu_input_d2h_bytes;
    uint64_t cpu_output_h2d_bytes;

    double cpu_operations_per_second;
    double gpu_operations_per_second;
    double h2d_bytes_per_second;
    double d2h_bytes_per_second;

    uint64_t cpu_fixed_ns;
    uint64_t gpu_fixed_ns;
};

struct ggml_offload_cost_result {
    uint64_t cpu_compute_ns;
    uint64_t cpu_transfer_ns;
    uint64_t gpu_compute_ns;
    uint64_t gpu_transfer_ns;
    uint64_t cpu_total_ns;
    uint64_t gpu_total_ns;
    int valid;
    int offload;
};

static inline uint64_t ggml_offload_cost_add_sat(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static inline uint64_t ggml_offload_cost_duration_ns(uint64_t amount, double rate) {
    if (amount == 0) {
        return 0;
    }
    if (!(rate > 0.0) || !isfinite(rate)) {
        return UINT64_MAX;
    }
    const double ns = 1.0e9*(double) amount/rate;
    return !isfinite(ns) || ns >= (double) UINT64_MAX ? UINT64_MAX : (uint64_t) ceil(ns);
}

static inline struct ggml_offload_cost_result ggml_offload_cost(const struct ggml_offload_cost_input * in) {
    struct ggml_offload_cost_result out = { 0, 0, 0, 0, 0, 0, 0, 0 };
    out.cpu_compute_ns = ggml_offload_cost_duration_ns(in->operations, in->cpu_operations_per_second);
    out.gpu_compute_ns = ggml_offload_cost_duration_ns(in->operations, in->gpu_operations_per_second);

    const uint64_t cpu_d2h_ns = ggml_offload_cost_duration_ns(in->cpu_input_d2h_bytes, in->d2h_bytes_per_second);
    const uint64_t cpu_h2d_ns = ggml_offload_cost_duration_ns(in->cpu_output_h2d_bytes, in->h2d_bytes_per_second);
    out.cpu_transfer_ns = ggml_offload_cost_add_sat(cpu_d2h_ns, cpu_h2d_ns);
    out.gpu_transfer_ns = ggml_offload_cost_duration_ns(in->weight_h2d_bytes, in->h2d_bytes_per_second);

    out.cpu_total_ns = ggml_offload_cost_add_sat(in->cpu_fixed_ns,
            ggml_offload_cost_add_sat(out.cpu_compute_ns, out.cpu_transfer_ns));
    out.gpu_total_ns = ggml_offload_cost_add_sat(in->gpu_fixed_ns,
            ggml_offload_cost_add_sat(out.gpu_compute_ns, out.gpu_transfer_ns));
    out.valid = out.cpu_total_ns != UINT64_MAX && out.gpu_total_ns != UINT64_MAX;
    out.offload = out.valid && out.gpu_total_ns < out.cpu_total_ns;
    return out;
}

// MUL_MAT_ID uses src1->ne[2] as its token count. src1->ne[1] is not the batch size.
static inline double ggml_offload_cost_expected_active_experts(int64_t n_experts, int64_t n_tokens, int64_t experts_per_token) {
    if (n_experts <= 0 || n_tokens <= 0 || experts_per_token <= 0) {
        return 0.0;
    }
    if (experts_per_token >= n_experts) {
        return (double) n_experts;
    }

    // Routing chooses experts without replacement inside one token. Tokens are treated as independent.
    const double inactive = exp((double) n_tokens*log1p(-(double) experts_per_token/n_experts));
    return n_experts*(1.0 - inactive);
}

static inline uint64_t ggml_offload_cost_moe_weight_bytes(
        uint64_t all_expert_bytes, int64_t n_experts, int64_t n_tokens, int64_t experts_per_token) {
    if (all_expert_bytes == 0 || n_experts <= 0) {
        return 0;
    }
    double n_active = ggml_offload_cost_expected_active_experts(n_experts, n_tokens, experts_per_token);
    if (n_active <= 0.0) {
        return 0;
    }
    if (n_active >= n_experts) {
        return all_expert_bytes;
    }
    return (uint64_t) ceil((double) all_expert_bytes*n_active/n_experts);
}

static inline uint64_t ggml_offload_cost_matmul_operations(
        int64_t m, int64_t n, int64_t k, int64_t experts_per_token) {
    if (m <= 0 || n <= 0 || k <= 0 || experts_per_token <= 0) {
        return 0;
    }
    const long double operations = 2.0L*m*n*k*experts_per_token;
    return operations >= UINT64_MAX ? UINT64_MAX : (uint64_t) operations;
}

#ifdef __cplusplus
}
#endif
