#include "speculative-cost.h"

#undef NDEBUG

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

// A synthetic machine and workload for the cost-aware depth (speculative-cost.h): the round time of a draft of n is
// t0 + slope*n, and each draft position is accepted with probability a when the previous ones were.

struct sim {
    double t0;
    double slope;
    double step; // draft time per token (part of the round time)

    double round(int n) const { return t0 + slope*n + step*n; }
};

// expected tokens per second of a fixed depth
static double rate(const sim & m, double a, int n) {
    double e = 1.0, p = 1.0;
    for (int k = 1; k <= n; ++k) {
        p *= a;
        e += p;
    }
    return e/m.round(n);
}

static int best_depth(const sim & m, double a, int cap) {
    int best = 1;
    for (int n = 2; n <= cap; ++n) {
        if (rate(m, a, n) > rate(m, a, best)) {
            best = n;
        }
    }
    return best;
}

// state of the simulated sequence across run() calls; fresh() starts a new one
static common_speculative_cost_round g_st;
static int    g_prev     = 0;
static int    g_since    = 1 << 20;
static int    g_switches = 0;
static double g_tokens   = 0.0;
static double g_time     = 0.0;

static void fresh(void) {
    g_st       = {};
    g_prev     = 0;
    g_since    = 1 << 20;
    g_switches = 0;
    g_tokens   = 0.0;
    g_time     = 0.0;
}

// run the controller for n_rounds on one sequence; returns the depth it ends at. A change of depth costs t_switch,
// spread over the next two rounds (the build and the uncaptured device graph run)
static int run(common_speculative_cost_table & tab, common_speculative_cost_ctrl & ctrl, const sim & m, double a,
               int cap, int n_rounds, std::mt19937 & rng, bool calibrated, double t_switch = 0.0) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (calibrated) {
        tab.t_verify.assign(cap + 2, 0.0);
        for (int w = 1; w <= cap + 1; ++w) {
            tab.t_verify[w] = m.t0 + m.slope*(w - 1);
        }
    }
    for (int r = 0; r < n_rounds; ++r) {
        const int n = ctrl.n_next;
        const bool shadow = ctrl.shadow;
        if (g_prev > 0 && n != g_prev) {
            g_since = 0;
            g_switches++;
        }
        g_prev = n;
        const double t = m.round(n) + (g_since == 0 ? 0.7*t_switch : g_since == 1 ? 0.3*t_switch : 0.0);
        g_since++;
        int acc = 0;
        while (acc < n && u(rng) < a) {
            acc++;
        }
        int shadow_hit = -1;
        if (shadow && acc == n) {
            shadow_hit = u(rng) < a ? 1 : 0;
        }
        tab.observe_draft(n + (shadow ? 1 : 0), m.step*(n + (shadow ? 1 : 0)));
        tab.observe_round(g_st, n, t);
        g_tokens += 1 + acc;
        g_time   += t;
        ctrl.observe(n, acc, shadow_hit, tab);
        ctrl.choose(1, cap, tab);
    }
    return ctrl.n_cur;
}

static void test_settles_at_the_optimum(void) {
    std::mt19937 rng(1234);
    const int cap = 10;

    // a cheap verify width: long drafts pay
    // an expensive verify width (CPU-resident experts): short drafts pay at the same acceptance
    const sim cheap  = { 20e-3, 0.3e-3, 0.5e-3 };
    const sim costly = { 20e-3, 6.0e-3, 0.5e-3 };

    for (const double a : { 0.5, 0.75, 0.9 }) {
        for (const sim * m : { &cheap, &costly }) {
            for (const bool calibrated : { false, true }) {
                common_speculative_cost_table tab;
                tab.resize(cap);
                common_speculative_cost_ctrl ctrl;
                ctrl.reset(cap, 1, 3);
                ctrl.choose(1, cap, tab);
                fresh();

                run(tab, ctrl, *m, a, cap, 1500, rng, calibrated);

                // the depth wanders with the decayed acceptance; judge the throughput of the depths it visits
                double got = 0.0;
                const int n_tail = 500;
                for (int r = 0; r < n_tail; ++r) {
                    run(tab, ctrl, *m, a, cap, 1, rng, calibrated);
                    got += rate(*m, a, ctrl.n_cur);
                }
                got /= n_tail;

                const int    opt  = best_depth(*m, a, cap);
                const double best = rate(*m, a, opt);
                printf("a = %.2f, slope = %.1f ms, calibrated = %d: optimum %2d, end depth %2d, throughput %.3f of best\n",
                       a, 1e3*m->slope, calibrated, opt, ctrl.n_cur, got/best);
                assert(got/best > 0.95);
            }
        }
    }
}

// a switch from hard to easy content within one request: the depth follows within a few tens of rounds
static void test_follows_a_content_switch(void) {
    std::mt19937 rng(42);
    const int cap = 10;
    const sim m = { 20e-3, 1.0e-3, 0.5e-3 };

    common_speculative_cost_table tab;
    tab.resize(cap);
    common_speculative_cost_ctrl ctrl;
    ctrl.reset(cap, 1, 3);
    ctrl.choose(1, cap, tab);
    fresh();

    const int d_hard = run(tab, ctrl, m, 0.4, cap, 300, rng, true);
    assert(d_hard <= best_depth(m, 0.4, cap) + 1);

    int rounds = 0;
    while (ctrl.n_cur < best_depth(m, 0.95, cap) - 1 && rounds < 400) {
        run(tab, ctrl, m, 0.95, cap, 1, rng, true);
        rounds++;
    }
    printf("hard -> easy: depth %d -> %d in %d rounds (optimum %d -> %d)\n",
           d_hard, ctrl.n_cur, rounds, best_depth(m, 0.4, cap), best_depth(m, 0.95, cap));
    assert(rounds < 150);

    rounds = 0;
    while (ctrl.n_cur > best_depth(m, 0.4, cap) + 1 && rounds < 400) {
        run(tab, ctrl, m, 0.4, cap, 1, rng, true);
        rounds++;
    }
    printf("easy -> hard: depth %d in %d rounds\n", ctrl.n_cur, rounds);
    assert(rounds < 60);
}

// an expensive change of depth (Qwen 3.8 Flash Next on R9700: about two rounds): the depth settles instead of
// wandering with the decayed acceptance, and the throughput with the switch costs stays near the best fixed depth
static void test_expensive_switch(void) {
    std::mt19937 rng(7);
    const int cap = 10;
    const sim m = { 30e-3, 2.5e-3, 2.0e-3 };
    for (const double a : { 0.6, 0.8 }) {
        common_speculative_cost_table tab;
        tab.resize(cap);
        tab.t_verify.assign(cap + 2, 0.0);
        for (int w = 1; w <= cap + 1; ++w) {
            tab.t_verify[w] = m.t0 + m.slope*(w - 1);
        }
        common_speculative_cost_ctrl ctrl;
        ctrl.reset(cap, 1, 3);
        ctrl.choose(1, cap, tab);
        fresh();
        const double t_switch = 2.0*m.round(3);
        run(tab, ctrl, m, a, cap, 2000, rng, true, t_switch);
        const int    opt  = best_depth(m, a, cap);
        const double got  = g_tokens/g_time;
        const double best = rate(m, a, opt);
        printf("expensive switch, a = %.2f: optimum %d, end depth %d, %d switches in 2000 rounds, t_switch %.1f ms, "
               "throughput %.3f of best\n", a, opt, ctrl.n_cur, g_switches, 1e3*tab.t_switch, got/best);
        assert(g_switches <= 40);
        assert(got/best > 0.95);
    }
}

static void test_bounds(void) {
    common_speculative_cost_table tab;
    tab.resize(4);
    common_speculative_cost_ctrl ctrl;

    // the start is 3 by default, bounded by the floor and the cap
    ctrl.reset(10, 1, 0);
    assert(ctrl.n_cur == 3);
    ctrl.reset(2, 1, 0);
    assert(ctrl.n_cur == 2);
    ctrl.reset(10, 5, 0);
    assert(ctrl.n_cur == 5);
    ctrl.reset(10, 1, 7);
    assert(ctrl.n_cur == 7);

    // no shadow at the cap
    ctrl.reset(4, 1, 4);
    ctrl.choose(1, 4, tab);
    assert(!ctrl.shadow);

    // a shadow hit counts at position n + 1, a shadow after a rejection does not count
    ctrl.reset(4, 1, 2);
    ctrl.observe(2, 2, 1, tab);
    assert(ctrl.trials[2] > 0.99 && ctrl.hits[2] > 0.99);
    ctrl.reset(4, 1, 2);
    ctrl.observe(2, 1, 1, tab);
    assert(ctrl.trials[2] == 0.0);
    // a miss at position 2 is a trial there; position 3 is not observed
    assert(ctrl.trials[1] > 0.99 && ctrl.hits[1] == 0.0);

    // the first rounds of a length (graph builds) and a stray slow round do not move the estimate
    common_speculative_cost_table t2;
    t2.resize(4);
    common_speculative_cost_round s2;
    t2.t_verify = { 0.0, 0.020, 0.020, 0.020, 0.020, 0.020 };
    t2.observe_round(s2, 2, 0.200);
    t2.observe_round(s2, 2, 0.150);
    for (int i = 0; i < 5; ++i) {
        t2.observe_round(s2, 2, 0.020);
    }
    t2.observe_round(s2, 2, 1.0);
    assert(t2.round_time(2) < 0.025);
    // a round with a graph build and the rounds after it are not settled; a build is not a change of depth
    t2.observe_round(s2, 2, 0.5, true);
    t2.observe_round(s2, 2, 0.3);
    assert(t2.round_time(2) < 0.025 && t2.t_switch == 0.0);
    // a change of depth: the extra time of the settle rounds is its cost
    for (int i = 0; i < common_speculative_cost_table::SETTLE_ROUNDS + 3; ++i) {
        t2.observe_round(s2, 2, 0.020);
    }
    t2.observe_round(s2, 3, 0.020 + 0.090);
    t2.observe_round(s2, 3, 0.020 + 0.030);
    t2.observe_round(s2, 3, 0.020);
    assert(std::fabs(t2.t_switch - 0.120) < 0.005);

    // slow rounds of a length that is not drafted again leave the window: the length is then estimated from the
    // calibration and the residual of the lengths in use
    common_speculative_cost_table t3;
    common_speculative_cost_round s3;
    t3.resize(6);
    t3.t_verify = { 0.0, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032 };
    for (int i = 0; i < 4; ++i) {
        t3.observe_round(s3, 2, 0.200);
    }
    for (size_t i = 0; i < common_speculative_cost_table::LIVE + common_speculative_cost_table::SETTLE_ROUNDS; ++i) {
        t3.observe_round(s3, 4, 0.028 + 0.003);
    }
    assert(std::fabs(t3.round_time(4) - 0.031) < 1e-9);
    assert(std::fabs(t3.round_time(2) - 0.027) < 1e-9);
}

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    test_bounds();
    test_settles_at_the_optimum();
    test_follows_a_content_switch();
    test_expensive_switch();
    printf("OK\n");
    return 0;
}
