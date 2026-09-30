#include "llama-quant-refine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

struct scalar_format {
    int block;
    int bits;
    int zero;
    int levels;
};

static scalar_format format(ggml_type type) {
    if (type == GGML_TYPE_Q2_0) { return {64, 2, 1, 4}; }
    if (type == GGML_TYPE_Q4_0) { return {32, 4, 8, 16}; }
    return {0, 0, 0, 0};
}

static bool is_k_type(ggml_type type) {
    return type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

static uint64_t random_next(uint64_t & state) {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t x = state;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static float gumbel(uint64_t & state) {
    const double u = ((random_next(state) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return (float) -std::log(-std::log(u));
}

static int get_code(const uint8_t * block, const scalar_format & f, int i) {
    const uint8_t * qs = block + 2;
    if (f.bits == 2) { return (qs[i/4] >> (2*(i%4))) & 3; }
    return i < 16 ? (qs[i] & 15) : (qs[i-16] >> 4);
}

static void set_code(uint8_t * block, const scalar_format & f, int i, int code) {
    uint8_t * qs = block + 2;
    if (f.bits == 2) {
        const int shift = 2*(i%4);
        qs[i/4] = (qs[i/4] & ~(3 << shift)) | (code << shift);
    } else if (i < 16) {
        qs[i] = (qs[i] & 0xf0) | code;
    } else {
        qs[i-16] = (qs[i-16] & 0x0f) | (code << 4);
    }
}

static float scale_of(const uint8_t * block) {
    ggml_fp16_t half;
    std::memcpy(&half, block, sizeof(half));
    return ggml_fp16_to_fp32(half);
}

static void set_scale(uint8_t * block, float scale) {
    const ggml_fp16_t half = ggml_fp32_to_fp16(scale);
    std::memcpy(block, &half, sizeof(half));
}

static double damage(const float * src, const float * second, const float * mean,
                     const float * decoded, int64_t n) {
    double diag = 0.0;
    double rank1 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double e = decoded[i] - src[i];
        const double m = mean[i];
        const double d = std::max((double) second[i] - m*m, 1e-8*(double) std::max(second[i], 1.0f));
        diag += d*e*e;
        rank1 += m*e;
    }
    return diag + rank1*rank1;
}

static void refine_block(uint8_t * block, const scalar_format & f, const float * target,
                         const float * second, const float * mean, int steps, uint64_t & state) {
    if (steps <= 0) { return; }
    const float base_scale = scale_of(block);
    if (!std::isfinite(base_scale) || base_scale == 0.0f) { return; }

    std::array<std::array<float, 5>, 64> logits = {};
    std::array<std::array<int, 5>, 64> choices = {};
    std::array<int, 64> nchoices = {};
    for (int i = 0; i < f.block; ++i) {
        const int base = get_code(block, f, i);
        const int low = f.bits == 2 ? 0 : std::max(0, base - 2);
        const int high = f.bits == 2 ? 3 : std::min(15, base + 2);
        nchoices[i] = high - low + 1;
        for (int k = 0; k < nchoices[i]; ++k) {
            choices[i][k] = low + k;
            logits[i][k] = choices[i][k] == base ? 0.5f : 0.0f;
        }
    }

    float scale = base_scale;
    std::array<std::array<float, 5>, 64> prob = {};
    std::array<float, 64> output = {};
    for (int step = 0; step < steps; ++step) {
        const float temp = std::max(0.2f, 1.5f*(1.0f - (float) step / steps));
        double rank1 = 0.0;
        for (int i = 0; i < f.block; ++i) {
            float zmax = -1e30f;
            for (int k = 0; k < nchoices[i]; ++k) {
                prob[i][k] = (logits[i][k] + gumbel(state)) / temp;
                zmax = std::max(zmax, prob[i][k]);
            }
            float sum = 0.0f;
            float code = 0.0f;
            for (int k = 0; k < nchoices[i]; ++k) {
                prob[i][k] = std::exp(prob[i][k] - zmax);
                sum += prob[i][k];
            }
            for (int k = 0; k < nchoices[i]; ++k) {
                prob[i][k] /= sum;
                code += prob[i][k] * (choices[i][k] - f.zero);
            }
            output[i] = scale * code;
            rank1 += mean[i] * (output[i] - target[i]);
        }
        double scale_grad = 0.0;
        double scale_norm = 1e-12;
        for (int i = 0; i < f.block; ++i) {
            const double m = mean[i];
            const double d = std::max((double) second[i] - m*m, 1e-8*(double) std::max(second[i], 1.0f));
            const double grad = 2.0*(d*(output[i] - target[i]) + m*rank1);
            const double expected = output[i] / scale;
            scale_grad += grad*expected;
            scale_norm += d*expected*expected;
            for (int k = 0; k < nchoices[i]; ++k) {
                const float signed_grad = grad*scale > 0.0 ? 1.0f : -1.0f;
                const float delta = signed_grad * prob[i][k] * ((choices[i][k] - f.zero) - expected) / temp;
                logits[i][k] = std::clamp(logits[i][k] - 0.4f*delta, -12.0f, 12.0f);
            }
        }
        scale = std::clamp((float) (scale - 0.05*scale_grad/scale_norm),
                           std::min(0.5f*base_scale, 1.5f*base_scale),
                           std::max(0.5f*base_scale, 1.5f*base_scale));
    }

    std::array<int, 64> hard = {};
    double num = 0.0, den = 0.0, mc = 0.0, mw = 0.0;
    for (int i = 0; i < f.block; ++i) {
        int best = 0;
        for (int k = 1; k < nchoices[i]; ++k) {
            if (logits[i][k] > logits[i][best]) { best = k; }
        }
        hard[i] = choices[i][best];
        const double c = hard[i] - f.zero;
        const double m = mean[i];
        const double d = std::max((double) second[i] - m*m, 1e-8*(double) std::max(second[i], 1.0f));
        num += d*c*target[i];
        den += d*c*c;
        mc += m*c;
        mw += m*target[i];
    }
    const double fitted = (num + mc*mw) / std::max(den + mc*mc, 1e-12);
    if (!std::isfinite(fitted) || fitted*base_scale <= 0.0) { return; }
    set_scale(block, (float) std::clamp(fitted, (double) std::min(0.5f*base_scale, 1.5f*base_scale),
                                                 (double) std::max(0.5f*base_scale, 1.5f*base_scale)));
    for (int i = 0; i < f.block; ++i) { set_code(block, f, i, hard[i]); }
}

} // namespace

bool llama_quant_refine_row(ggml_type type, const float * src, void * dst, int64_t n,
                            const float * second, const float * mean,
                            bool feedback, int gsq_steps, uint64_t seed) {
    const scalar_format f = format(type);
    const bool k_type = is_k_type(type);
    const int block_size = f.block > 0 ? f.block : k_type ? ggml_blck_size(type) : 0;
    if (block_size == 0 || (gsq_steps > 0 && f.block == 0) || n <= 0 || n % block_size != 0 || !second || !mean) { return false; }
    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) { return false; }

    const size_t row_bytes = ggml_row_size(type, n);
    const size_t block_bytes = ggml_row_size(type, block_size);
    std::vector<uint8_t> original(row_bytes);
    std::memcpy(original.data(), dst, row_bytes);
    std::vector<float> adjusted(n);
    std::vector<float> decoded(n);
    double past = 0.0;
    double future = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = std::max((double) second[i] - (double) mean[i]*mean[i], 1e-8*(double) std::max(second[i], 1.0f));
        future += (double) mean[i]*mean[i] / d;
    }

    uint8_t * bytes = (uint8_t *) dst;
    for (int64_t start = 0; start < n; start += block_size) {
        uint8_t * block = bytes + (start/block_size)*block_bytes;
        if (feedback) {
            for (int i = 0; i < block_size; ++i) {
                const int64_t col = start + i;
                const double d = std::max((double) second[col] - (double) mean[col]*mean[col],
                                          1e-8*(double) std::max(second[col], 1.0f));
                future -= (double) mean[col]*mean[col] / d;
            }
        }
        float max_correction = 0.0f;
        for (int i = 0; i < block_size; ++i) {
            const int64_t col = start + i;
            const double d = std::max((double) second[col] - (double) mean[col]*mean[col], 1e-8*(double) std::max(second[col], 1.0f));
            adjusted[col] = feedback ? (float) (src[col] + (mean[col]/d)*past/(1.0 + std::max(0.0, future))) : src[col];
            max_correction = std::max(max_correction, std::abs(adjusted[col] - src[col]));
        }
        if (feedback && max_correction > 0.0f) {
            float limit = k_type ? 0.0f : 0.5f*std::abs(scale_of(block));
            if (k_type) {
                traits->to_float(block, decoded.data() + start, block_size);
                limit = 0.0f;
                for (int i = 0; i < block_size; ++i) {
                    limit = std::max(limit, std::abs(src[start + i] - decoded[start + i]));
                }
            }
            const float gain = std::min(1.0f, limit/max_correction);
            for (int i = 0; i < block_size; ++i) {
                const int64_t col = start + i;
                adjusted[col] = src[col] + gain*(adjusted[col] - src[col]);
            }
        }
        if (feedback) {
            ggml_quantize_chunk(type, adjusted.data() + start, block, 0, 1, block_size, second + start);
        }
        if (gsq_steps > 0) {
            refine_block(block, f, adjusted.data() + start, second + start, mean + start, gsq_steps, seed);
        }
        if (feedback) {
            traits->to_float(block, decoded.data() + start, block_size);
            for (int i = 0; i < block_size; ++i) {
                const int64_t col = start + i;
                past += mean[col] * (src[col] - decoded[col]);
            }
        }
    }

    traits->to_float(dst, decoded.data(), n);
    const double candidate = damage(src, second, mean, decoded.data(), n);
    traits->to_float(original.data(), decoded.data(), n);
    const double baseline = damage(src, second, mean, decoded.data(), n);
    if (!std::isfinite(candidate) || candidate >= baseline) {
        std::memcpy(dst, original.data(), row_bytes);
    }
    return true;
}
