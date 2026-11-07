#pragma once

#include "speculative-cost.h"

#include <cmath>
#include <cstdint>

// DSpark generates the full block before this policy selects a verification prefix. Costs are complete rounds,
// measured from one draft start to the next, including feature ingest and checkpoint replay. There is no per-token
// draft cost: shortening the prefix does not undo block generation. Initially restricted to one sequence/proposer.
struct common_speculative_block {
    static constexpr size_t WINDOW = 96;
    static constexpr int MIN_SAMPLES = 3;
    common_speculative_conf calibration;
    std::deque<std::pair<int, double>> times;
    std::vector<int> attempts;
    std::vector<int> chosen;
    int cap = 0;
    int depth = 0;
    int rounds = 0;
    int probe = 0;
    int64_t start = 0;
    uint64_t builds = 0;
    bool feedback = false;
    bool selected = false;
    float temperature = NAN;
    double tokens = 0.0;
    double seconds = 0.0;
    int emitted = 0;

    explicit common_speculative_block(int n = 0) : attempts(n + 1), chosen(n + 1), cap(n) {
        calibration.seqs.resize(1);
    }

    int samples(int n) const {
        return (int) std::count_if(times.begin(), times.end(), [n](const auto & v) { return v.first == n; });
    }

    double cost(const common_speculative_cost_table & tab, int n) const {
        std::vector<double> own, residual;
        for (const auto & v : times) {
            if (v.first == n) {
                own.push_back(v.second);
            }
            residual.push_back(v.second - tab.verify(v.first + 1));
        }
        if (own.size() >= MIN_SAMPLES) {
            return common_speculative_cost_table::median(own);
        }
        // The startup target-only calibration is a prior, corrected by complete live rounds. Probe every depth
        // before trusting this extrapolation; host experts and the live expert cache can change the width curve.
        return std::max(1e-6, tab.verify(n + 1) + common_speculative_cost_table::median(residual));
    }

    void begin(float temp) {
        start = 0;
        feedback = false;
        selected = false;
        if (temp != temperature) {
            calibration = {};
            calibration.seqs.resize(1);
            temperature = temp;
            tokens = seconds = 0.0;
        }
    }

    void round(int64_t now, uint64_t n_builds) {
        // Empty drafts use the server's ordinary sampling path, which has no speculative accept callback. Reaching
        // the next draft means that path emitted one token. begin() discards an unfinished or cancelled request.
        if (selected && depth == 0) {
            feedback = true;
            emitted = 1;
        }
        if (start > 0 && feedback && now > start) {
            tokens = 0.98*tokens + emitted;
            seconds = 0.98*seconds + 1e-6*(now - start);
        }
        if (start > 0 && feedback && n_builds == builds && now > start) {
            times.emplace_back(depth, 1e-6*(now - start));
            if (times.size() > WINDOW) {
                times.pop_front();
            }
        }
        start = now;
        builds = n_builds;
        feedback = false;
        selected = false;
    }

    int choose(const common_speculative_cost_table & tab, const std::vector<float> & confidence, int floor, int start_depth) {
        const int n = std::min(cap, (int) confidence.size());
        floor = std::min(floor, n);
        int best = std::clamp(start_depth > 0 ? start_depth : 3, floor, n);
        if (tab.calibrated()) {
            double q = 1.0, expected = 1.0;
            const double zero_cost = cost(tab, 0);
            double best_value = floor == 0 ? (seconds > 0 ? expected - (tokens/seconds)*zero_cost : expected/zero_cost) : -INFINITY;
            if (floor == 0) {
                best = 0;
            }
            for (int k = 0; k < n; ++k) {
                q *= calibration.cal(k, confidence[k]);
                expected += q;
                if (k + 1 >= floor) {
                    // Average output/time across changing content is a ratio of sums. Once a live rate exists,
                    // maximize expected output minus rate times cost, rather than each state's individual ratio.
                    const double value = seconds > 0 ? expected - (tokens/seconds)*cost(tab, k + 1) :
                                                       expected/cost(tab, k + 1);
                    if (value > best_value) {
                        best_value = value;
                        best = k + 1;
                    }
                }
            }
            // Bound startup exploration even if graph rebuilds prevent steady samples. Three observations per
            // depth are enough to replace the startup prior; later probes keep unused depths from going stale.
            for (int d = floor; d <= n; ++d) {
                if (samples(d) < MIN_SAMPLES && attempts[d] < 8) {
                    best = d;
                    break;
                }
            }
            if (++rounds % 32 == 0) {
                best = floor + probe++ % (n - floor + 1);
            }
        }
        selected = true;
        depth = best;
        attempts[best]++;
        chosen[best]++;
        calibration.seqs[0].p.assign(confidence.begin(), confidence.begin() + best);
        return best;
    }

    void observe(int drafted, int accepted) {
        if (!selected || feedback || start == 0) {
            return;
        }
        // A caller can shorten a draft for the remaining output budget. Positions after that bound are censored.
        depth = std::min(depth, drafted);
        calibration.observe(0, depth, std::min(accepted, depth));
        emitted = std::min(accepted, depth) + 1;
        feedback = true;
    }
};
