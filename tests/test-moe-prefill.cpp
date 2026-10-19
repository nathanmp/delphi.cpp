#include "../src/llama-moe-prefill.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

static void check(bool value) {
    if (!value) { fprintf(stderr, "prefill partition check failed\n"); exit(1); }
}

static void check_ownership(const llama_moe_prefill_plan & plan, const std::vector<int32_t> & cold,
        const std::vector<int32_t> & routes) {
    for (int e : routes) {
        const int resident = cold[e] < 0;
        const int cpu = cold[e] >= 0 && plan.slots[e] < 0;
        const int transient = plan.slots[e] >= 0;
        check(resident + cpu + transient == 1);
        if (transient) { check(plan.experts[plan.slots[e]] == e); }
    }
}

int main() {
    const std::vector<uint32_t> count = {2, 8, 8, 1, 0, 32};
    const std::vector<int32_t> cold = {0, 1, -1, 3, 4, 5};
    const std::vector<int32_t> routes = {5, 1, 5, 2, 0, 1, 3, 5, 2};
    auto plan = llama_moe_prefill_partition(count, cold, 4, 2);
    check(plan.experts == std::vector<int32_t>({5, 1}));
    check(plan.slots[2] == -1 && plan.slots[0] == -1 && plan.slots[4] == -1);
    check_ownership(plan, cold, routes);

    auto cpu = llama_moe_prefill_partition(count, cold, std::numeric_limits<uint32_t>::max(), 6);
    check(cpu.experts.empty());
    check_ownership(cpu, cold, routes);

    auto gpu = llama_moe_prefill_partition(count, cold, 1, 6, true);
    check(gpu.experts.size() == 4 && !gpu.capacity_rejected);
    for (int e : routes) { check(cold[e] < 0 || gpu.slots[e] >= 0); }
    check_ownership(gpu, cold, routes);

    auto rejected = llama_moe_prefill_partition(count, cold, 1, 3, true);
    check(rejected.capacity_rejected && rejected.experts.empty());
    check_ownership(rejected, cold, routes);
    check(llama_moe_prefill_partition(count, cold, 1, 0).experts.empty());

    auto tied = llama_moe_prefill_partition({7, 7, 7}, {0, 1, 2}, 7, 2);
    check(tied.experts == std::vector<int32_t>({0, 1}));
    // Repeated calls use unrelated masks, including an empty transfer set and an entirely resident batch.
    auto empty = llama_moe_prefill_partition({0, 0, 0}, {0, 1, 2}, 1, 3, true);
    check(empty.experts.empty() && !empty.capacity_rejected);
    auto hot = llama_moe_prefill_partition({7, 7, 7}, {-1, -1, -1}, 1, 3, true);
    check(hot.experts.empty());
    auto tail = llama_moe_prefill_partition({0, 0, 1}, {0, 1, 2}, 1, 3);
    check(tail.experts == std::vector<int32_t>({2}) && tail.slots[0] == -1 && tail.slots[2] == 0);
    puts("prefill partition: passed");
}
