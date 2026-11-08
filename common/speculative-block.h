#pragma once

#include "speculative-cost.h"

#include <cmath>
#include <cstdint>

// DSpark generates the full block before this policy selects a verification prefix. Costs are complete rounds,
// measured from one draft start to the next, including feature ingest and checkpoint replay. There is no per-token
// draft cost: shortening the prefix does not undo block generation. One acceptance controller per sequence; batched timing is accounted for separately.
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

    void round(int64_t now, uint64_t n_builds, bool learn = true) {
        // Empty drafts use the server's ordinary sampling path, which has no speculative accept callback. Reaching
        // the next draft means that path emitted one token. begin() discards an unfinished or cancelled request.
        if (selected && depth == 0) {
            feedback = true;
            emitted = 1;
        }
        if (learn && start > 0 && feedback && now > start) {
            tokens = 0.98*tokens + emitted;
            seconds = 0.98*seconds + 1e-6*(now - start);
        }
        if (learn && start > 0 && feedback && n_builds == builds && now > start) {
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

// Batched rounds have one wall time. Never assign that time to each sequence's prefix.
struct common_speculative_block_batch {
    struct sample { int slots, width; double seconds; };
    std::deque<sample> times;
    std::vector<int> active;
    int64_t start = 0;
    uint64_t builds = 0;
    double tokens = 0, seconds = 0;
    int rounds = 0;

    void round(int64_t now, uint64_t graph_builds, const std::vector<int> & next,
            const std::vector<common_speculative_block> & controllers) {
        bool complete = start > 0 && active == next && now > start && builds == graph_builds;
        int width = 0, emitted = 0;
        for (int s : active) {
            const auto & c = controllers[s];
            complete &= c.selected && (c.feedback || c.depth == 0);
            width += c.depth + 1;
            emitted += c.feedback ? c.emitted : 1;
        }
        if (complete) {
            const double elapsed = 1e-6*(now - start);
            tokens = .98*tokens + emitted;
            seconds = .98*seconds + elapsed;
            times.push_back({(int) active.size(), width, elapsed});
            if (times.size() > common_speculative_block::WINDOW) { times.pop_front(); }
        }
        if (active != next) { tokens = seconds = 0; }
        active = next;
        start = now;
        builds = graph_builds;
    }

    double cost(const common_speculative_cost_table & tab, int slots, int width) const {
        auto prior = [&](int w) {
            const int last = (int) tab.t_verify.size() - 1;
            if (last < 1) { return double(w); }
            if (w <= last) { return tab.verify(w); }
            // Startup measures one sequence only. Extrapolate until this total batch width has live observations.
            const double slope = last > 1 ? std::max(1e-6, (tab.verify(last) - tab.verify(1))/(last - 1)) : tab.verify(1);
            return tab.verify(last) + (w - last)*slope;
        };
        std::vector<double> own, residual;
        for (const auto & v : times) {
            if (v.slots != slots) { continue; }
            residual.push_back(v.seconds - prior(v.width));
            if (v.width == width) { own.push_back(v.seconds); }
        }
        return std::max(1e-6, own.size() >= common_speculative_block::MIN_SAMPLES ?
                common_speculative_cost_table::median(own) : prior(width) + common_speculative_cost_table::median(residual));
    }

    void choose(const common_speculative_cost_table & tab, std::vector<common_speculative_block> & controllers,
            const std::vector<std::vector<float>> & confidence, int floor, int initial) {
        if (active.empty()) { return; }
        std::vector<int> depths(controllers.size());
        auto expected = [&](int s, int n) {
            double value = 1, survival = 1;
            for (int k = 0; k < n; ++k) {
                survival *= controllers[s].calibration.cal(k, confidence[s][k]);
                value += survival;
            }
            return value;
        };
        for (int s : active) {
            depths[s] = std::clamp(initial > 0 ? initial : 3, std::min(floor, (int) confidence[s].size()), (int) confidence[s].size());
        }
        // Coordinate updates price a row against the whole batch and the total expected output.
        for (int pass = 0; pass < 2; ++pass) {
            for (int s : active) {
                int others = 0;
                double output = 0;
                for (int peer : active) {
                    if (peer != s) { others += depths[peer] + 1; output += expected(peer, depths[peer]); }
                }
                double best_value = -INFINITY;
                for (int n = std::min(floor, (int) confidence[s].size()); n <= (int) confidence[s].size(); ++n) {
                    const double total = output + expected(s, n);
                    const double t = cost(tab, (int) active.size(), others + n + 1);
                    const double value = seconds > 0 ? total - (tokens/seconds)*t : total/t;
                    if (value > best_value) { best_value = value; depths[s] = n; }
                }
            }
        }
        // Probe one slot at a time, rather than force every slot through the same startup widths.
        const int s_probe = active[rounds % active.size()];
        if (rounds < (int) active.size()*(controllers[s_probe].cap + 1)*3 || rounds % 32 == 0) {
            const int lo = std::min(floor, (int) confidence[s_probe].size());
            depths[s_probe] = lo + (rounds/(int) active.size()) % ((int) confidence[s_probe].size() - lo + 1);
        }
        ++rounds;
        for (int s : active) {
            auto & c = controllers[s];
            c.selected = true;
            c.depth = depths[s];
            c.attempts[c.depth]++;
            c.chosen[c.depth]++;
            c.calibration.seqs[0].p.assign(confidence[s].begin(), confidence[s].begin() + c.depth);
        }
    }
};
