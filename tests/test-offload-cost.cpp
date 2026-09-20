// delphi.cpp: offline tests for the transfer-aware offload cost model (ggml/src/ggml-offload-cost.h).

#include "../ggml/src/ggml-offload-cost.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static ggml_offload_cost_result trace_cost(
        double cpu_ms, double gpu_compute_ms, double weight_gb, double h2d_gbps,
        double cpu_input_gb = 0.0, double cpu_output_gb = 0.0, double d2h_gbps = 20.0) {
    const uint64_t operations = 1000000000000ULL;
    const ggml_offload_cost_input input = {
        /* .operations                 = */ operations,
        /* .weight_h2d_bytes           = */ (uint64_t) std::llround(weight_gb*1.0e9),
        /* .cpu_input_d2h_bytes        = */ (uint64_t) std::llround(cpu_input_gb*1.0e9),
        /* .cpu_output_h2d_bytes       = */ (uint64_t) std::llround(cpu_output_gb*1.0e9),
        /* .cpu_operations_per_second  = */ operations/(cpu_ms*1.0e-3),
        /* .gpu_operations_per_second  = */ operations/(gpu_compute_ms*1.0e-3),
        /* .h2d_bytes_per_second       = */ h2d_gbps*1.0e9,
        /* .d2h_bytes_per_second       = */ d2h_gbps*1.0e9,
        /* .cpu_fixed_ns               = */ 0,
        /* .gpu_fixed_ns               = */ 0,
    };
    return ggml_offload_cost(&input);
}

static void test_moe_shape_accounting() {
    require(std::fabs(ggml_offload_cost_expected_active_experts(256, 1, 8) - 8.0) < 1.0e-9,
            "one token selects exactly top-k active experts in expectation");
    require(ggml_offload_cost_expected_active_experts(256, 512, 8) > 255.999,
            "Qwen prefill reaches essentially all experts");
    require(ggml_offload_cost_expected_active_experts(256, 512, 6) > 255.998,
            "DeepSeek prefill reaches essentially all experts");

    require(ggml_offload_cost_matmul_operations(512, 512, 2048, 8) == 8589934592ULL,
            "MUL_MAT_ID work uses tokens from ne[2] and includes top-k");
    require(ggml_offload_cost_moe_weight_bytes(21304000000ULL, 256, 512, 8) > 21303900000ULL,
            "Qwen prefill estimates the recorded full expert transfer");
}

static void test_archived_trace_retrodiction() {
    // Qwen3.6 pp512 moved 21.304 GB for Q8_0 and 16.428 GB for Q6_K. Their measured GPU paths
    // are the only arms retained in that archive. The CPU times below are explicit counterfactual
    // bounds, not measurements; live A/B runs are required before treating these choices as proof.
    const auto q8 = trace_cost(1550.0, 40.0, 21.304, 18.23, 0.1887, 1.346);
    const auto q6 = trace_cost(1600.0, 65.0, 16.428, 15.38, 0.1887, 1.346);
    require(q8.valid && q8.offload, "Qwen Q8_0 pp512 selects GPU");
    require(q6.valid && q6.offload, "Qwen Q6_K pp512 selects GPU");

    // MXFP4 and Q4_K used CPU_REPACK buffers that did not present a GPU candidate in the recorded
    // run. A missing candidate throughput keeps that recorded CPU placement.
    ggml_offload_cost_input unsupported = {
        /* .operations                 = */ 1000000000000ULL,
        /* .weight_h2d_bytes           = */ 10000000000ULL,
        /* .cpu_input_d2h_bytes        = */ 188700000ULL,
        /* .cpu_output_h2d_bytes       = */ 1346000000ULL,
        /* .cpu_operations_per_second  = */ 1.0e12/0.7882,
        /* .gpu_operations_per_second  = */ 0.0,
        /* .h2d_bytes_per_second       = */ 20.0e9,
        /* .d2h_bytes_per_second       = */ 20.0e9,
        /* .cpu_fixed_ns               = */ 0,
        /* .gpu_fixed_ns               = */ 0,
    };
    auto mxfp4 = ggml_offload_cost(&unsupported);
    require(!mxfp4.valid && !mxfp4.offload, "Qwen MXFP4 pp512 keeps CPU when GPU has no implementation");
    unsupported.cpu_operations_per_second = 1.0e12/0.9122;
    auto q4 = ggml_offload_cost(&unsupported);
    require(!q4.valid && !q4.offload, "Qwen Q4_K pp512 keeps CPU when GPU has no implementation");

    // DeepSeek V4 pp512 reached almost every expert. The CPU path took 3.405 s, while the recorded
    // single-socket GPU path moved 98.535 GB at 19.70 GB/s before GPU compute was included.
    const auto deepseek = trace_cost(3405.0, 250.0, 98.535, 19.70);
    require(deepseek.valid && !deepseek.offload, "DeepSeek MUL_MAT_ID pp512 selects CPU");
    require(deepseek.gpu_transfer_ns > deepseek.cpu_total_ns,
            "DeepSeek transfer alone decides the placement");
}

static void test_measured_qwen_q4_crossover() {
    // A separate Qwen3.6 Q4_K_M A/B sweep at commit 98318b126 measured CPU faster through ub256
    // and GPU faster from ub512. These are machine-scoped rates, not a model-keyed lookup.
    const uint64_t all_weight_bytes = 21000000000ULL;
    auto cost = [&](int64_t tokens) {
        const uint64_t operations = 120*ggml_offload_cost_matmul_operations(512, tokens, 2048, 8);
        const uint64_t weight_bytes = 120*ggml_offload_cost_moe_weight_bytes(
                all_weight_bytes/120, 256, tokens, 8);
        const ggml_offload_cost_input input = {
            /* .operations                 = */ operations,
            /* .weight_h2d_bytes           = */ weight_bytes,
            /* .cpu_input_d2h_bytes        = */ 0,
            /* .cpu_output_h2d_bytes       = */ 0,
            /* .cpu_operations_per_second  = */ 0.8e12,
            /* .gpu_operations_per_second  = */ 5.0e12,
            /* .h2d_bytes_per_second       = */ 30.0e9,
            /* .d2h_bytes_per_second       = */ 30.0e9,
            /* .cpu_fixed_ns               = */ 0,
            /* .gpu_fixed_ns               = */ 0,
        };
        return ggml_offload_cost(&input);
    };

    require(!cost(64).offload,  "Qwen Q4_K ub64 selects measured CPU winner");
    require(!cost(128).offload, "Qwen Q4_K ub128 selects measured CPU winner");
    require(!cost(256).offload, "Qwen Q4_K ub256 selects measured CPU winner");
    require(cost(512).offload,  "Qwen Q4_K ub512 selects measured GPU winner");
    require(cost(1024).offload, "Qwen Q4_K ub1024 selects measured GPU winner");
}

int main() {
    test_moe_shape_accounting();
    test_archived_trace_retrodiction();
    test_measured_qwen_q4_crossover();
    std::puts("test-offload-cost: all tests passed");
    return 0;
}
