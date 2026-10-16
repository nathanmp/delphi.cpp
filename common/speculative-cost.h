#pragma once

#include <algorithm>
#include <deque>
#include <vector>

// Cost-aware draft depth for draft-model speculative decoding (--spec-draft-adaptive-mode cost).
//
// The depth n of the next draft maximizes the expected tokens per second of a round, E(n) / C(n):
//
//   E(n) = 1 + sum_{k=1..n} prod_{i=1..k} a_i   a_i: chance that draft position i is accepted when 1..i-1 were
//   C(n) = wall time of a round that drafts n tokens: the draft, the verify of n + 1 rows, the draft ingest, the
//          acceptance and a checkpoint replay, from the start of one draft of the sequence to the start of the next
//
// The bucket controller (speculative-adaptive.h) sees only the acceptance, so the depth it settles at is fixed by its
// constants. This one also sees what a wider verify costs on the hardware that runs it.
//
// Cost (common_speculative_cost_table, one per implementation, shared by the sequences and kept across requests):
//   * t_verify[w]: verify time of a target batch of w rows, measured at load (common_speculative_calibrate)
//   * live rounds: the last LIVE settled rounds (see below), over all draft lengths. A round time is the
//                  calibrated part (verify of n + 1 rows, n draft steps) plus a residual (sampling, acceptance, the
//                  draft ingest): the median residual of its own length when the window has MIN_SAMPLES of it, else
//                  interpolated between the nearest such lengths, else the median of all. Old rounds leave the
//                  window, so a slow start (weights paged in on first use) does not stay in the table of a length
//                  that is not drafted again
//   * t_switch:    the extra time of the SETTLE_ROUNDS rounds after a change of depth. A new verify width builds a
//                  graph when its graph is stale (the KV view grew since its last use) or evicted, and the device
//                  graph runs uncaptured until the second identical call captures it: on Qwen 3.8 Flash Next (three
//                  R9700) about 90 ms in the first round and more in the next one, against a round of 45 ms. The
//                  rounds after a change of depth or a graph build (llama_graph_n_built) are not settled and are
//                  kept out of the live rounds
//
// Acceptance (common_speculative_cost_ctrl, one per sequence, reset by each request):
//   * decayed hits and trials per position: a draft of n with m accepted is a hit at 1..m and a miss at m + 1
//   * shrunk towards the rate of the implementation over all requests (decays slowly), so a new request starts from
//     what the previous ones showed and moves to its own content within a few tens of rounds
//   * position n + 1 is learned without verifying it: the draft makes one extra (shadow) token that is not verified;
//     when all n are accepted, the target's next token shows whether the shadow token would have been accepted
//
// The depth moves down to any length and up one step at a time, so a climb rests on an observed position. A change
// of depth must win back t_switch within SWITCH_ROUNDS rounds, and stay worth it for DEBOUNCE rounds. A request starts
// at the best depth for the acceptance over all requests, kept at the previous depth unless that is worth a change. Without a
// calibration, every PROBE_EVERY rounds a neighbouring length without live round times is drafted once (a cost
// probe); with one, the calibrated verify difference extrapolates from the length in use.

// rounds of one sequence after a disturbance (a change of depth or a graph build)
struct common_speculative_cost_round {
    int    n_prev = 0;     // draft length of the previous round
    int    left   = 0;     // rounds left in the open window
    double extra  = 0.0;   // extra time of the window so far, seconds
    bool   sw     = false; // a change of depth opened the window
};

struct common_speculative_cost_table {
    static constexpr size_t LIVE         = 48;    // live round times kept, over all draft lengths
    static constexpr size_t WINDOW       = 9;     // switch costs kept (and x2 draft steps)
    static constexpr int    SETTLE_ROUNDS = 3;    // rounds after a disturbance: the build, the uncaptured device graph
                                                  // runs, the capture
    static constexpr double MIN_SAMPLES  = 3.0;   // live rounds before a draft length has its own residual
    static constexpr double GAMMA_GLOBAL = 0.995; // decay of the acceptance over all requests (per round)
    static constexpr double PRIOR_RATE   = 0.5;   // acceptance of a position never observed
    static constexpr double PRIOR_WEIGHT = 1.0;

    std::vector<double> t_verify; // [w] calibrated verify time of w rows, seconds; 0 = not measured
    std::vector<double> n_round;  // [n] live rounds of a draft of n tokens in the window
    std::vector<double> res;      // [n] median residual of those rounds, seconds (valid from MIN_SAMPLES)
    double res_all = 0.0;         // median residual of all live rounds, seconds
    double t_live  = 0.0;         // median of all live round times, seconds
    double t_step  = 0.0;         // live draft time per drafted token (median), seconds
    double t_switch = 0.0;        // extra time of a change of depth (median), seconds; 0 = not observed

    std::deque<std::pair<int, double>> s_live;  // last LIVE rounds without a build: (draft length, time)
    std::deque<double>                 s_step;  // last draft times per token
    std::deque<double>                 s_switch; // last extra times of changes of depth

    std::vector<double> hits;     // [k - 1] acceptance over all requests
    std::vector<double> trials;

    void resize(int cap) {
        const size_t n = (size_t) std::max(1, cap) + 2;
        if (n_round.size() < n) {
            n_round.resize(n, 0.0);
            res.resize(n, 0.0);
            hits.resize(n, 0.0);
            trials.resize(n, 0.0);
        }
    }

    static double median(std::vector<double> v) {
        if (v.empty()) {
            return 0.0;
        }
        std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
        return v[v.size()/2];
    }

    // push x into the window and return the median of the window
    static double push_median(std::deque<double> & win, double x, size_t window) {
        win.push_back(x);
        while (win.size() > window) {
            win.pop_front();
        }
        return median(std::vector<double>(win.begin(), win.end()));
    }

    bool calibrated() const {
        return t_verify.size() > 1;
    }

    double verify(int w) const {
        return w >= 1 && w < (int) t_verify.size() ? t_verify[w] : 0.0;
    }

    // the part of a round time that changes with the draft length: the calibrated verify and the draft; without a
    // calibration a token of verify width costs 2 % of a round
    double model(int n) const {
        const double vn = verify(n + 1);
        return vn > 0.0 ? vn + n*t_step : n*(t_step + 0.02*t_live);
    }

    // recompute the residuals (live time - model) from the window; the model moves with t_step
    void refresh() {
        std::vector<double> t_all;
        for (const auto & [n, t] : s_live) {
            t_all.push_back(t);
        }
        t_live = median(t_all);

        std::vector<double>              r_all;
        std::vector<std::vector<double>> r_n(n_round.size());
        for (const auto & [n, t] : s_live) {
            const double r = t - model(n);
            r_all.push_back(r);
            if (n < (int) r_n.size()) {
                r_n[n].push_back(r);
            }
        }
        res_all = median(r_all);
        for (size_t n = 0; n < r_n.size(); ++n) {
            n_round[n] = (double) r_n[n].size();
            res[n]     = median(r_n[n]);
        }
    }

    // one closed round of a sequence; built: the round built a graph
    void observe_round(common_speculative_cost_round & st, int n, double t, bool built = false) {
        if (n < 1 || t <= 0.0) {
            return;
        }
        resize(n);
        // without a calibration the cost probes (one round at a neighbouring length) are the only times of other
        // lengths, so a change of depth cannot be set aside
        const bool switched = calibrated() && st.n_prev > 0 && n != st.n_prev;
        st.n_prev = n;
        if (switched || (built && st.left == 0)) {
            if (st.left > 0 && st.sw) {
                // cut short by the next change: what it has is still its cost
                t_switch = push_median(s_switch, st.extra, WINDOW);
            }
            st.left  = SETTLE_ROUNDS;
            st.extra = 0.0;
            st.sw    = switched;
        }
        if (st.left > 0) {
            st.extra += s_live.empty() ? 0.0 : std::max(0.0, t - round_time(n));
            if (--st.left == 0 && st.sw) {
                t_switch = push_median(s_switch, st.extra, WINDOW);
            }
            return;
        }
        s_live.emplace_back(n, t);
        while (s_live.size() > LIVE) {
            s_live.pop_front();
        }
        refresh();
    }

    void observe_draft(int n, double t, bool built = false) {
        if (n < 1 || t <= 0.0 || built) {
            return;
        }
        t_step = push_median(s_step, t/n, 2*WINDOW);
        refresh();
    }

    bool trusted(int n) const {
        return n >= 1 && n < (int) n_round.size() && n_round[n] >= MIN_SAMPLES;
    }

    // residual of a draft length: its own, else interpolated between the nearest lengths with their own, else that
    // of all live rounds
    double residual(int n) const {
        if (trusted(n)) {
            return res[n];
        }
        int lo = -1;
        int hi = -1;
        for (int m = n - 1; m >= 1 && lo < 0; --m) {
            lo = trusted(m) ? m : -1;
        }
        for (int m = n + 1; m < (int) n_round.size() && hi < 0; ++m) {
            hi = trusted(m) ? m : -1;
        }
        if (lo > 0 && hi > 0) {
            return res[lo] + (res[hi] - res[lo])*(n - lo)/(hi - lo);
        }
        return lo > 0 ? res[lo] : hi > 0 ? res[hi] : res_all;
    }

    // estimated wall time of a round that drafts n tokens; only ratios between lengths matter
    double round_time(int n) const {
        if (s_live.empty()) {
            // nothing live yet: the calibrated verify, or a flat cost
            return calibrated() ? model(n) : 1.0;
        }
        return std::max(0.25*t_live, model(n) + residual(n));
    }

    double global_rate(int k) const {
        // the last observed position stands in for the deeper ones
        double rate = PRIOR_RATE;
        for (int i = 1; i <= k && i - 1 < (int) trials.size(); ++i) {
            if (trials[i - 1] >= 0.5) {
                rate = (hits[i - 1] + PRIOR_WEIGHT*PRIOR_RATE)/(trials[i - 1] + PRIOR_WEIGHT);
            }
        }
        return rate;
    }
};

struct common_speculative_cost_ctrl {
    static constexpr double GAMMA     = 0.95;  // decay per round: an effective window of 20 rounds
    static constexpr double W_GLOBAL  = 3.0;   // weight of the global rate, in rounds
    static constexpr double HYST      = 0.01;  // a new depth must be this much better
    static constexpr double SHADOW_N  = 4.0;   // shadow while position n + 1 has fewer effective trials
    static constexpr int    PROBE_EVERY = 8;   // rounds between cost probes (without a calibration)
    static constexpr double SWITCH_ROUNDS = 64.0; // rounds in which a change of depth must win back its cost
    static constexpr int    DEBOUNCE   = 8;    // rounds a change must stay worth its cost (in one direction) before it is made

    std::vector<double> hits;   // [k - 1]
    std::vector<double> trials;

    int  n_cur    = 0;     // chosen depth
    int  n_next   = 0;     // depth of the next draft: n_cur, or a neighbour on a cost probe
    bool shadow   = false; // the next draft carries one shadow token
    int  n_rounds = 0;
    int  n_want   = 0;     // rounds in a row that a change in the direction dir was worth its cost
    int  dir      = 0;     // +1 deeper, -1 shallower

    void reset(int cap, int floor, int n_start) {
        const int c = std::max(1, cap);
        const int f = std::min(c, std::max(1, floor));
        hits.assign((size_t) c + 2, 0.0);
        trials.assign((size_t) c + 2, 0.0);
        // without a set start, the depth of the previous request (its graph is the most likely to be warm), else 3
        n_cur    = std::min(c, std::max(f, n_start > 0 ? n_start : n_cur > 0 ? n_cur : 3));
        n_next   = n_cur;
        shadow   = false;
        n_rounds = 0;
        n_want   = 0;
        dir      = 0;
    }

    double rate(int k, const common_speculative_cost_table & tab) const {
        const double g = tab.global_rate(k);
        if (k < 1 || k - 1 >= (int) trials.size()) {
            return g;
        }
        return (hits[k - 1] + W_GLOBAL*g)/(trials[k - 1] + W_GLOBAL);
    }

    // expected tokens of a round that drafts n; a_last replaces the rate of position n when >= 0
    double expected(int n, const common_speculative_cost_table & tab, double a_last = -1.0) const {
        double e = 1.0;
        double p = 1.0;
        for (int k = 1; k <= n; ++k) {
            p *= (k == n && a_last >= 0.0) ? a_last : rate(k, tab);
            e += p;
        }
        return e;
    }

    double score(int n, const common_speculative_cost_table & tab, double a_last = -1.0) const {
        return expected(n, tab, a_last)/tab.round_time(n);
    }

    // one verified round: n_draft verified tokens, n_accepted of them accepted; shadow_hit is -1 without a shadow
    // token (or when a verified token was rejected), else whether the shadow token matched the target's next token
    void observe(int n_draft, int n_accepted, int shadow_hit, common_speculative_cost_table & tab) {
        const int n_pos = n_draft + (shadow_hit >= 0 ? 1 : 0);
        if (n_pos < 1) {
            return;
        }
        tab.resize(n_pos);
        if ((int) trials.size() < n_pos + 1) {
            hits.resize((size_t) n_pos + 1, 0.0);
            trials.resize((size_t) n_pos + 1, 0.0);
        }
        for (size_t i = 0; i < trials.size(); ++i) {
            hits[i]   *= GAMMA;
            trials[i] *= GAMMA;
        }
        for (size_t i = 0; i < tab.trials.size(); ++i) {
            tab.hits[i]   *= common_speculative_cost_table::GAMMA_GLOBAL;
            tab.trials[i] *= common_speculative_cost_table::GAMMA_GLOBAL;
        }

        auto add = [&](int k, bool hit) {
            trials[k - 1]     += 1.0;
            tab.trials[k - 1] += 1.0;
            if (hit) {
                hits[k - 1]     += 1.0;
                tab.hits[k - 1] += 1.0;
            }
        };
        for (int k = 1; k <= std::min(n_accepted + 1, n_draft); ++k) {
            add(k, k <= n_accepted);
        }
        if (shadow_hit >= 0 && n_accepted >= n_draft) {
            add(n_draft + 1, shadow_hit > 0);
        }
    }

    // choose the depth of the next draft within [floor, cap] and whether it carries a shadow token
    // start: the first choice of a request, from the acceptance over all requests: any length, without the debounce
    // (the depth the previous content left is no evidence for the new one, and a new prompt often builds graphs)
    void choose(int floor, int cap, const common_speculative_cost_table & tab, bool start = false) {
        const int c = std::max(1, cap);
        const int f = std::min(c, std::max(1, floor));
        n_cur = std::min(c, std::max(f, n_cur));

        const double s_cur = score(n_cur, tab);
        int    best   = n_cur;
        double s_best = s_cur;
        for (int n = f; n <= (start ? c : std::min(c, n_cur + 1)); ++n) {
            const double s = score(n, tab);
            if (s > s_best) {
                best   = n;
                s_best = s;
            }
        }
        // a change of depth must win back its cost within SWITCH_ROUNDS rounds; until one is observed, assume two
        // rounds (the build and the uncaptured run). Without a calibration the cost is not measured (see observe_round)
        const double t_cur = tab.round_time(n_cur);
        const double t_sw  = !tab.calibrated() ? 0.0 : tab.s_switch.empty() ? 2.0*t_cur : tab.t_switch;
        const double hyst  = std::max(HYST, t_cur > 0.0 ? t_sw/(SWITCH_ROUNDS*t_cur) : 0.0);
        // the first rounds of a response are often easy (a formulaic opening), and the decayed acceptance follows a
        // few lucky rounds: a change is made only when it stays worth its cost for DEBOUNCE rounds. Waiting costs a
        // fraction of a round per round; a needless change costs t_switch twice
        const int d = best == n_cur || s_best <= s_cur*(1.0 + hyst) ? 0 : best > n_cur ? 1 : -1;
        if (d == 0 || d != dir) {
            n_want = d == 0 ? 0 : 1;
            dir    = d;
        } else {
            n_want++;
        }
        if (d != 0 && (n_want >= DEBOUNCE || t_sw == 0.0 || start)) {
            n_cur  = best;
            n_want = 0;
            dir    = 0;
        }

        // a shadow token pays when position n + 1 could win if it were always accepted and is not known well
        shadow = false;
        if (n_cur < c) {
            const int k = n_cur + 1;
            const double t_k = k - 1 < (int) trials.size() ? trials[k - 1] : 0.0;
            shadow = t_k < SHADOW_N && score(k, tab, 1.0) > score(n_cur, tab)*(1.0 + hyst);
        }

        n_next = n_cur;
        if (++n_rounds % PROBE_EVERY == 0 && !tab.calibrated()) {
            for (const int m : { n_cur - 1, n_cur + 1 }) {
                if (m >= f && m <= c && (m >= (int) tab.n_round.size() || tab.n_round[m] < tab.MIN_SAMPLES)) {
                    n_next = m;
                    shadow = false;
                    break;
                }
            }
        }
    }
};

// Confidence stop (--spec-draft-adaptive-mode conf): the depth is chosen inside each round, one draft step at a time.
//
// The draft keeps token k (0-based) while the chance that tokens 0..k are all accepted pays for its verify row:
//
//   q_k = prod_{i<=k} cal(i, p_i) >= lam * (t_verify[k + 2] - t_verify[k + 1] + t_step)
//
//   p_i: top-1 probability of draft step i; cal(i, p): observed acceptance of draft tokens at position i with that
//   probability (the draft probabilities are overconfident), decayed over all requests, with p itself as prior
//   lam: tokens per second of the recent rounds, so a row must give more tokens per time than the average round
//
// On GLM 5.3 Flash (3x CMP 170HX) an offline replay of a fixed-7 log gave +11.5 % over the best fixed depth with this
// rule, against +1 % for the best fixed depth per prompt. It needs cheap changes of verify width: every width keeps its
// own graph (LLAMA_GRAPH_ALT_SLOTS >= n_max + 1) and the KV view grows seldom (LLAMA_KV_N_PAD); it ignores t_switch.
struct common_speculative_conf {
    static constexpr int    N_POS      = 3;     // position classes: 1st, 2nd, 3rd and later draft token
    static constexpr int    N_BIN      = 8;     // classes of the top-1 probability
    static constexpr double GAMMA_CAL  = 0.999; // decay of the calibration per verified round
    static constexpr double PRIOR_W    = 4.0;   // weight of p itself as the prior of cal(i, p), in trials
    static constexpr double GAMMA_RATE = 0.98;  // decay of the throughput per round

    struct seq_state {
        double             q      = 1.0; // chance that the kept tokens of the open draft are all accepted
        std::vector<float> p;            // top-1 probabilities of the kept tokens
        int64_t            t_last = 0;   // start of the open round (us), 0: none
        int                tok    = 1;   // tokens of the open round (1 until its verify is fed back)
    };

    double hits[N_POS][N_BIN]   = {};
    double trials[N_POS][N_BIN] = {};
    double e_tok  = 0.0;
    double e_time = 0.0; // seconds
    std::vector<seq_state> seqs;

    static int bin(float p) {
        static const float edges[N_BIN - 1] = { 0.3f, 0.5f, 0.7f, 0.8f, 0.9f, 0.95f, 0.98f };
        int b = 0;
        while (b < N_BIN - 1 && p >= edges[b]) {
            b++;
        }
        return b;
    }

    double cal(int k, float p) const {
        const int i = std::min(k, N_POS - 1);
        const int b = bin(p);
        return (hits[i][b] + PRIOR_W*p)/(trials[i][b] + PRIOR_W);
    }

    double rate() const {
        return e_time > 0.0 ? e_tok/e_time : 0.0;
    }

    // keep draft token k of sequence s, whose top-1 probability is p?
    bool keep(const common_speculative_cost_table & tab, int s, int k, float p) {
        auto & st = seqs[s];
        if (k == 0) {
            st.q = 1.0;
            st.p.clear();
        }
        st.q *= cal(k, p);
        const double dv = tab.verify(k + 2) - tab.verify(k + 1);
        if (tab.calibrated() && dv > 0.0 && st.q < rate()*(dv + tab.t_step)) {
            return false;
        }
        st.p.push_back(p);
        return true;
    }

    // the verify of the last draft of s: n_draft tokens, n_accepted of them accepted
    void observe(int s, int n_draft, int n_accepted) {
        auto & st = seqs[s];
        for (auto & h : hits) for (double & x : h) x *= GAMMA_CAL;
        for (auto & t : trials) for (double & x : t) x *= GAMMA_CAL;
        const int n = std::min({ n_accepted + 1, n_draft, (int) st.p.size() });
        for (int k = 0; k < n; ++k) {
            const int i = std::min(k, N_POS - 1);
            const int b = bin(st.p[k]);
            trials[i][b] += 1.0;
            hits[i][b]   += k < n_accepted ? 1.0 : 0.0;
        }
        st.tok = n_accepted + 1;
    }

    // a new draft of s starts at t_us: the previous one is a closed round
    void round(int s, int64_t t_us) {
        auto & st = seqs[s];
        if (st.t_last > 0) {
            e_tok  = GAMMA_RATE*e_tok  + st.tok;
            e_time = GAMMA_RATE*e_time + 1e-6*(t_us - st.t_last);
        }
        st.t_last = t_us;
        st.tok    = 1;
    }

    // a new request: the prompt in between is not a round
    void begin(int s) {
        seqs[s].t_last = 0;
        seqs[s].tok    = 1;
    }
};
