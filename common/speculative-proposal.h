#pragma once

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// A separate stream keeps proposal draws independent of target rejection draws.
struct common_speculative_proposal {
    std::mt19937 rng;
    int64_t seed = -1;

    void begin() { seed = -1; }

    llama_token sample(const llama_token_data_array & candidates, float temp, uint32_t request_seed,
            std::vector<llama_token_data> & q) {
        GGML_ASSERT(candidates.size > 0 && temp > 0.0f && std::isfinite(temp));
        if (seed != request_seed) {
            seed = request_seed;
            rng.seed(request_seed == LLAMA_DEFAULT_SEED ? std::random_device{}() : request_seed ^ 0x85ebca6bu);
        }

        q.assign(candidates.data, candidates.data + candidates.size);
        // Backend top-k need not order ties or candidates consistently.
        std::sort(q.begin(), q.end(), [](const llama_token_data & a, const llama_token_data & b) {
            return a.id < b.id;
        });
        const auto best = std::max_element(q.begin(), q.end(), [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit < b.logit;
        });
        const float max_logit = best->logit;
        GGML_ASSERT(std::isfinite(max_logit));
        double sum = 0.0;
        for (auto & e : q) {
            e.p = std::exp((double(e.logit) - max_logit) / temp);
            sum += e.p;
        }
        for (auto & e : q) {
            e.p /= sum;
        }

        double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        for (const auto & e : q) {
            u -= e.p;
            if (u < 0.0) {
                return e.id;
            }
        }
        // A rounding overshoot must still return a token with nonzero q.
        return best->id;
    }
};
