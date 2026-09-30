#include "../src/llama-quant-refine.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

static double damage(const std::vector<float> & src, const std::vector<float> & decoded,
                     const std::vector<float> & second, const std::vector<float> & mean) {
    double diagonal = 0.0;
    double rank1 = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        const double error = decoded[i] - src[i];
        const double variance = std::max((double) second[i] - (double) mean[i]*mean[i],
                                         1e-8*(double) std::max(second[i], 1.0f));
        diagonal += variance*error*error;
        rank1 += mean[i]*error;
    }
    return diagonal + rank1*rank1;
}

static void check_type(ggml_type type, int64_t n) {
    std::vector<float> src(n), second(n), mean(n), decoded(n), baseline_decoded(n);
    for (int64_t i = 0; i < n; ++i) {
        src[i] = 0.09f*std::sin(0.21f*i) + 0.03f*std::cos(0.07f*i);
    }
    const size_t bytes = ggml_row_size(type, n);
    std::vector<uint8_t> baseline(bytes), feedback(bytes), gsq(bytes), combined(bytes), repeat(bytes);
    ggml_quantize_chunk(type, src.data(), baseline.data(), 0, 1, n, second.data());
    const auto * traits = ggml_get_type_traits(type);
    traits->to_float(baseline.data(), baseline_decoded.data(), n);
    for (int64_t i = 0; i < n; ++i) {
        mean[i] = 5.0f*(baseline_decoded[i] - src[i]);
        second[i] = mean[i]*mean[i] + 0.0001f;
    }
    ggml_quantize_chunk(type, src.data(), baseline.data(), 0, 1, n, second.data());
    traits->to_float(baseline.data(), baseline_decoded.data(), n);
    feedback = baseline;
    gsq = baseline;
    combined = baseline;
    repeat = baseline;

    assert(llama_quant_refine_row(type, src.data(), feedback.data(), n, second.data(), mean.data(), true, 0, 17));
    assert(llama_quant_refine_row(type, src.data(), gsq.data(), n, second.data(), mean.data(), false, 8, 17));
    assert(llama_quant_refine_row(type, src.data(), repeat.data(), n, second.data(), mean.data(), false, 8, 17));
    assert(gsq == repeat);
    assert(gsq != baseline);
    if (type == GGML_TYPE_Q2_0) { assert(feedback != baseline); }
    assert(llama_quant_refine_row(type, src.data(), combined.data(), n, second.data(), mean.data(), true, 8, 17));

    const double base_damage = damage(src, baseline_decoded, second, mean);
    traits->to_float(feedback.data(), decoded.data(), n);
    assert(damage(src, decoded, second, mean) <= base_damage + 1e-10);
    traits->to_float(gsq.data(), decoded.data(), n);
    assert(damage(src, decoded, second, mean) <= base_damage + 1e-10);
    traits->to_float(combined.data(), decoded.data(), n);
    assert(damage(src, decoded, second, mean) <= base_damage + 1e-10);

    std::fill(mean.begin(), mean.end(), 0.0f);
    repeat = baseline;
    assert(llama_quant_refine_row(type, src.data(), repeat.data(), n, second.data(), mean.data(), true, 0, 17));
    assert(repeat == baseline);
}

static bool check_k_type(ggml_type type) {
    constexpr int64_t n = 512;
    std::vector<float> src(n), second(n, 1.0f), mean(n), decoded(n), baseline_decoded(n);
    for (int64_t i = 0; i < n; ++i) {
        src[i] = 0.11f*std::sin(0.17f*i) + 0.02f*std::cos(0.43f*i);
    }
    const size_t bytes = ggml_row_size(type, n);
    std::vector<uint8_t> baseline(bytes), feedback(bytes), repeat(bytes);
    const auto * traits = ggml_get_type_traits(type);
    ggml_quantize_chunk(type, src.data(), baseline.data(), 0, 1, n, second.data());
    traits->to_float(baseline.data(), baseline_decoded.data(), n);
    for (int64_t i = 0; i < n; ++i) {
        mean[i] = 5.0f*(baseline_decoded[i] - src[i]);
        second[i] = mean[i]*mean[i] + 0.0001f;
    }
    ggml_quantize_chunk(type, src.data(), baseline.data(), 0, 1, n, second.data());
    traits->to_float(baseline.data(), baseline_decoded.data(), n);
    feedback = baseline;
    repeat = baseline;

    assert(llama_quant_refine_row(type, src.data(), feedback.data(), n, second.data(), mean.data(), true, 0, 17));
    assert(llama_quant_refine_row(type, src.data(), repeat.data(), n, second.data(), mean.data(), true, 0, 17));
    assert(feedback == repeat);
    const double base_damage = damage(src, baseline_decoded, second, mean);
    traits->to_float(feedback.data(), decoded.data(), n);
    assert(damage(src, decoded, second, mean) <= base_damage + 1e-10);

    std::fill(mean.begin(), mean.end(), 0.0f);
    repeat = baseline;
    assert(llama_quant_refine_row(type, src.data(), repeat.data(), n, second.data(), mean.data(), true, 0, 17));
    assert(repeat == baseline);
    return feedback != baseline;
}

int main() {
    check_type(GGML_TYPE_Q2_0, 128);
    check_type(GGML_TYPE_Q4_0, 96);
    bool changed = false;
    changed |= check_k_type(GGML_TYPE_Q2_K);
    changed |= check_k_type(GGML_TYPE_Q3_K);
    changed |= check_k_type(GGML_TYPE_Q4_K);
    changed |= check_k_type(GGML_TYPE_Q5_K);
    changed |= check_k_type(GGML_TYPE_Q6_K);
    assert(changed);
    return 0;
}
