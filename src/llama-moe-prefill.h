#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

struct llama_moe_prefill_plan {
    std::vector<int32_t> experts;
    std::vector<int32_t> slots;
    bool capacity_rejected = false;
};

// A distinct cold expert has one destination for all three projections and all of its token slots.
inline llama_moe_prefill_plan llama_moe_prefill_partition(
        const std::vector<uint32_t> & count, const std::vector<int32_t> & cold,
        uint32_t min_reuse, size_t capacity, bool gpu_only = false) {
    llama_moe_prefill_plan plan;
    plan.slots.assign(count.size(), -1);
    for (size_t e = 0; e < count.size(); ++e) {
        if (cold[e] >= 0 && count[e] > 0 && count[e] >= min_reuse) {
            plan.experts.push_back(e);
        }
    }
    if (gpu_only && plan.experts.size() > capacity) {
        plan.experts.clear();
        plan.capacity_rejected = true;
        return plan;
    }
    std::sort(plan.experts.begin(), plan.experts.end(), [&](int32_t a, int32_t b) {
        return count[a] != count[b] ? count[a] > count[b] : a < b;
    });
    plan.experts.resize(std::min(capacity, plan.experts.size()));
    for (size_t s = 0; s < plan.experts.size(); ++s) {
        plan.slots[plan.experts[s]] = s;
    }
    return plan;
}
