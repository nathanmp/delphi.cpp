#include "speculative-block.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

static common_speculative_cost_table table() {
    common_speculative_cost_table tab;
    tab.t_verify = {0, .040, .060, .080, .079, .100, .098};
    return tab;
}

static void test_prefix_and_costs() {
    auto tab = table();
    common_speculative_block ctl(5);
    ctl.begin(1.0f);
    // Warm all depths with complete cycle costs. Adjacent widths need not be monotonic.
    for (int n = 1; n <= 5; ++n) {
        for (int i = 0; i < 3; ++i) {
            ctl.times.emplace_back(n, tab.verify(n + 1) + .010);
        }
    }
    assert(ctl.cost(tab, 3) < ctl.cost(tab, 2));
    assert(ctl.cost(tab, 5) < ctl.cost(tab, 4));
    assert(ctl.choose(tab, {1, 1, 1, 1, 1}, 1, 0) == 5);
    assert(ctl.choose(tab, {0, 0, 0, 0, 0}, 1, 0) == 1);
    assert(ctl.choose(tab, {1, 1, 1, 0, 0}, 1, 0) == 3);
    assert(ctl.choose(tab, {0, 0, 0, 0, 0}, 3, 0) == 3);
    // A caller's bound cannot be exceeded by the floor or exploration.
    assert(ctl.choose(tab, {1, 1}, 3, 0) <= 2);
}

static void test_feedback_and_reset() {
    auto tab = table();
    common_speculative_block ctl(5);
    ctl.begin(1.0f);
    ctl.round(1000000, 0);
    assert(ctl.choose(tab, {.9f, .8f, .7f, .6f, .5f}, 1, 0) == 1);
    ctl.observe(1, 0);
    const int b = common_speculative_conf::bin(.9f);
    assert(ctl.calibration.trials[0][b] == 1);
    assert(ctl.calibration.trials[1][common_speculative_conf::bin(.8f)] == 0);
    ctl.observe(1, 0); // checkpoint replay must not count twice
    assert(ctl.calibration.trials[0][b] == 1);
    ctl.round(1100000, 0);
    assert(ctl.samples(1) == 1);
    ctl.choose(tab, {.9f, .8f, .7f, .6f, .5f}, 1, 0);
    ctl.observe(1, 1);
    ctl.round(2000000, 1); // graph build is excluded from the steady estimate
    assert(ctl.samples(1) == 1);
    ctl.begin(1.0f);
    ctl.round(100000000, 1); // idle/prefill time is not a cycle
    assert(ctl.samples(1) == 1);
    assert(ctl.calibration.trials[0][b] > 1);
    ctl.begin(0.0f); // sampling regime changes invalidate acceptance calibration
    assert(ctl.calibration.trials[0][b] == 0);
    assert(ctl.samples(1) == 1);
}

static void test_probes_and_fallback() {
    common_speculative_block ctl(5);
    common_speculative_cost_table empty;
    assert(ctl.choose(empty, {1, 1, 1, 1, 1}, 1, 0) == 3);
    assert(ctl.choose(empty, {1, 1, 1, 1, 1}, 1, 4) == 4);
    auto tab = table();
    // Even when no steady samples can be collected, exploration has a finite startup budget.
    for (int i = 0; i < 160; ++i) {
        const int n = ctl.choose(tab, {1, 1, 1, 1, 1}, 1, 0);
        assert(n >= 1 && n <= 5);
    }
    for (int n = 1; n <= 5; ++n) {
        assert(ctl.attempts[n] >= 8);
    }
    puts("OK");
}

static void test_empty_prefix() {
    auto tab = table();
    common_speculative_block ctl(5);
    ctl.begin(1.0f);
    ctl.round(1000000, 0);
    assert(ctl.choose(tab, {.1f, .1f, .1f, .1f, .1f}, 0, 0) == 0);
    assert(ctl.calibration.seqs[0].p.empty());
    // No accept callback is made by ordinary decoding. Count its token and full block cost on the next round.
    ctl.round(1050000, 0);
    assert(ctl.samples(0) == 1);
    assert(ctl.tokens == 1);
    assert(std::fabs(ctl.seconds - .05) < 1e-9);
    assert(ctl.calibration.trials[0][0] == 0); // skipping a proposal does not reject it
    assert(ctl.choose(tab, {.1f, .1f, .1f, .1f, .1f}, 0, 0) == 0);
    ctl.begin(1.0f);
    ctl.round(100000000, 0);
    assert(ctl.samples(0) == 1); // cancellation/idle time is not an emitted token
}

static void test_parallel_batch() {
    common_speculative_cost_table tab;
    for (int w = 0; w <= 12; ++w) { tab.t_verify.push_back(.02 + .01*w); }
    std::vector<common_speculative_block> slots(2, common_speculative_block(5));
    for (auto & c : slots) { c.begin(1); c.round(1000000, 0, false); }
    common_speculative_block_batch batch;
    batch.rounds = 1001; // outside startup/periodic exploration
    batch.round(1000000, 0, {0, 1}, slots);
    const std::vector<std::vector<float>> confidence = {{1, 1, 1, 1, 1}, {0, 0, 0, 0, 0}};
    batch.choose(tab, slots, confidence, 1, 3);
    assert(slots[0].depth == 5 && slots[1].depth == 1);
    slots[0].observe(5, 3);
    slots[1].observe(1, 0);
    assert(slots[0].calibration.trials[0][common_speculative_conf::bin(1)] == 1);
    assert(slots[1].calibration.trials[0][common_speculative_conf::bin(1)] == 0);
    batch.round(1100000, 0, {0, 1}, slots);
    assert(batch.times.size() == 1 && batch.times.back().width == 8);
    assert(std::abs(batch.cost(tab, 2, 8) - .1) < 1e-9);
    for (auto & c : slots) { c.round(1100000, 0, false); assert(c.times.empty()); }
    slots[1].begin(0); // a new request/temperature must not clear its peer's calibration
    assert(slots[0].calibration.trials[0][common_speculative_conf::bin(1)] == 1);
    batch.round(1200000, 0, {0}, slots); // a departure/incomplete batch supplies no cost sample
    assert(batch.times.size() == 1 && batch.tokens == 0);
    batch.start = 0; // prefill/new request invalidates an open timing interval
    batch.round(9000000, 0, {0}, slots);
    assert(batch.times.size() == 1);
    assert(batch.cost(tab, 2, 50) > batch.cost(tab, 2, 12)); // no zero-cost tail beyond startup widths
    batch.active = {0, 1};
    batch.choose(tab, slots, {{}, {.5f}}, 3, 5); // remaining output budget bounds each slot independently
    assert(slots[0].depth == 0 && slots[1].depth == 1);
}

int main() {
    test_parallel_batch();
    test_empty_prefix();
    test_prefix_and_costs();
    test_feedback_and_reset();
    test_probes_and_fallback();
}
