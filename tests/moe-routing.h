#pragma once

// Expert-routing distributions for MoE benchmarks.
//
// For a routed MUL_MAT_ID the distribution of tokens over the experts changes the cost as much as the tensor
// shape does: a batch whose tokens concentrate on a few experts gives each expert enough rows to reach the
// batched GEMM and repack paths, while a batch spread evenly over hundreds of experts never does. A benchmark
// that only ever uses an even spread therefore reports a cost that no real model pays.
//
// Each generator fills the id tensor of one (n_experts, top_k, n_tokens) workload. The synthetic
// distributions approximate shapes that were measured on real models; "replay" reproduces a distribution
// recorded from an actual run with llama-moe-schedule-stats and is the accurate option when a capture exists.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace moe_routing {

struct params {
    int     n_experts = 0;
    int     top_k     = 0;
    int64_t n_tokens  = 0;

    double  alpha      = 1.0; // zipf exponent
    int64_t tpe_target = 0;   // concentration: assignments per active expert

    std::string replay_file;
    int replay_layer  = -1; // replay: -1 accepts the first record that matches the workload
    int replay_ubatch = -1;

    uint32_t seed = 1234;
};

struct routing {
    std::string          name;      // distribution, plus the parameter that identifies this case
    std::vector<int32_t> ids;       // top_k ids for each token, token major
    std::vector<int64_t> histogram; // assignments held by each expert

    int     n_active = 0;
    int64_t tpe_min  = 0;
    int64_t tpe_max  = 0;
    double  tpe_mean = 0.0;
};

static inline const char * dist_names() {
    return "uniform,single,zipf,skewed,concentration,hottail,replay";
}

// Every token must get top_k distinct experts: a token that holds the same expert twice is rejected by the
// routed MUL_MAT_ID paths that assign one slot per expert.
static inline void pick_distinct(
        std::vector<int32_t> & chosen, int top_k, std::mt19937 & rng, std::discrete_distribution<int> & pick,
        const std::vector<int32_t> & index) {
    chosen.clear();
    while ((int) chosen.size() < top_k) {
        const int32_t expert = index[pick(rng)];
        if (std::find(chosen.begin(), chosen.end(), expert) == chosen.end()) {
            chosen.push_back(expert);
        }
    }
}

static inline void routing_uniform(const params & p, std::mt19937 & rng, routing & out) {
    out.name = "uniform";
    std::vector<int32_t> experts(p.n_experts);
    std::iota(experts.begin(), experts.end(), 0);
    for (int64_t t = 0; t < p.n_tokens; ++t) {
        std::shuffle(experts.begin(), experts.end(), rng);
        std::copy_n(experts.begin(), p.top_k, out.ids.begin() + (size_t) t * p.top_k);
    }
}

// One group of top_k experts takes every token; the rest stay empty.
static inline void routing_single(const params & p, routing & out) {
    out.name = "single";
    for (int64_t t = 0; t < p.n_tokens; ++t) {
        for (int k = 0; k < p.top_k; ++k) {
            out.ids[(size_t) t * p.top_k + k] = k;
        }
    }
}

static inline void routing_zipf(const params & p, std::mt19937 & rng, routing & out) {
    out.name = "zipf:" + std::to_string(p.alpha);
    std::vector<double> weights(p.n_experts);
    for (int i = 0; i < p.n_experts; ++i) {
        weights[i] = 1.0 / std::pow((double) i + 1.0, p.alpha);
    }
    std::vector<int32_t> index(p.n_experts);
    std::iota(index.begin(), index.end(), 0);
    std::discrete_distribution<int> pick(weights.begin(), weights.end());
    std::vector<int32_t> chosen;
    chosen.reserve(p.top_k);
    for (int64_t t = 0; t < p.n_tokens; ++t) {
        pick_distinct(chosen, p.top_k, rng, pick, index);
        std::copy(chosen.begin(), chosen.end(), out.ids.begin() + (size_t) t * p.top_k);
    }
}

// Approximate the observed DeepSeek skew: about 3% of the experts receive 30% of the assignments.
static inline void routing_skewed(const params & p, std::mt19937 & rng, routing & out) {
    out.name = "skewed-3pct-30pct";
    const int n_hot = std::max(1, (int) std::lround(0.03 * p.n_experts));
    std::uniform_int_distribution<int> hot_pick(0, n_hot - 1);
    std::uniform_int_distribution<int> cold_pick(n_hot, p.n_experts - 1);
    std::bernoulli_distribution choose_hot(0.30);
    std::vector<int32_t> chosen;
    chosen.reserve(p.top_k);
    for (int64_t t = 0; t < p.n_tokens; ++t) {
        chosen.clear();
        while ((int) chosen.size() < p.top_k) {
            const int32_t expert = choose_hot(rng) ? hot_pick(rng) : cold_pick(rng);
            if (std::find(chosen.begin(), chosen.end(), expert) == chosen.end()) {
                chosen.push_back(expert);
            }
        }
        std::copy(chosen.begin(), chosen.end(), out.ids.begin() + (size_t) t * p.top_k);
    }
}

// Spread the batch evenly over exactly `active` experts, so that each one holds tpe_target assignments.
// This is the control used to find the row count at which a backend changes its kernel choice.
static inline void routing_concentration(const params & p, int active, std::mt19937 & rng, routing & out) {
    out.name = "concentration:" + std::to_string(p.tpe_target);
    std::vector<int32_t> experts(active);
    std::iota(experts.begin(), experts.end(), 0);
    for (int64_t t = 0; t < p.n_tokens; ++t) {
        std::shuffle(experts.begin(), experts.end(), rng);
        std::copy_n(experts.begin(), p.top_k, out.ids.begin() + (size_t) t * p.top_k);
    }
}

// The measured GLM-5.3-Flash shape: one hot expert in about 60% of the tokens, a Zipf tail over the others,
// about a fifth of the experts empty. The hot expert sits at a random index because it moves between layers.
static inline void routing_hottail(const params & p, std::mt19937 & rng, routing & out) {
    out.name = "hottail";
    std::vector<int32_t> perm(p.n_experts);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    const int n_pool = std::min(p.n_experts, std::max(p.top_k + 1, (int) (0.8 * p.n_experts)));
    std::vector<double> weights(n_pool - 1);
    for (int i = 0; i < n_pool - 1; ++i) {
        weights[i] = 1.0 / (i + 1.0);
    }
    std::discrete_distribution<int> pick(weights.begin(), weights.end());
    std::bernoulli_distribution use_hot(0.6);
    std::uniform_int_distribution<int> pos(0, p.top_k - 1);
    for (int64_t t = 0; t < p.n_tokens; ++t) {
        int32_t * chosen = out.ids.data() + (size_t) t * p.top_k;
        const int k0 = use_hot(rng) ? 1 : 0;
        if (k0) {
            chosen[0] = perm[0];
        }
        for (int k = k0; k < p.top_k; ++k) {
            int32_t expert;
            do {
                expert = perm[1 + pick(rng)];
            } while (std::find(chosen, chosen + k, expert) != chosen + k);
            chosen[k] = expert;
        }
        std::swap(chosen[0], chosen[pos(rng)]);
    }
}

// One (layer, ubatch) record of a llama-moe-schedule-stats CSV: the rows held by every expert.
struct replay_record {
    int     layer    = 0;
    int     ubatch   = 0;
    int64_t n_tokens = 0;
    int     top_k    = 0;
    int     n_experts = 0;
    std::vector<int64_t> rows;
};

static inline std::vector<std::string> split_fields(const std::string & line, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    for (std::string field; std::getline(ss, field, sep);) {
        out.push_back(field);
    }
    return out;
}

static inline bool read_replay_file(const std::string & path, std::vector<replay_record> & out, std::string & error) {
    std::ifstream in(path);
    std::string line;
    if (!in.is_open() || !std::getline(in, line)) {
        error = "cannot read replay file " + path;
        return false;
    }
    const std::vector<std::string> header = split_fields(line, ',');
    auto column = [&](const char * name) {
        const auto it = std::find(header.begin(), header.end(), name);
        return it == header.end() ? -1 : (int) (it - header.begin());
    };
    const int c_layer = column("layer"), c_ubatch = column("ubatch"), c_tokens = column("n_tokens");
    const int c_top_k = column("top_k"), c_experts = column("n_experts"), c_rows = column("rows");
    if (c_layer < 0 || c_ubatch < 0 || c_tokens < 0 || c_top_k < 0 || c_experts < 0 || c_rows < 0) {
        error = path + " is missing one of the layer,ubatch,n_tokens,top_k,n_experts,rows columns";
        return false;
    }
    while (std::getline(in, line)) {
        const std::vector<std::string> f = split_fields(line, ',');
        if ((int) f.size() != (int) header.size()) {
            continue;
        }
        replay_record rec;
        rec.layer     = std::stoi(f[c_layer]);
        rec.ubatch    = std::stoi(f[c_ubatch]);
        rec.n_tokens  = std::stoll(f[c_tokens]);
        rec.top_k     = std::stoi(f[c_top_k]);
        rec.n_experts = std::stoi(f[c_experts]);
        for (const std::string & v : split_fields(f[c_rows], ';')) {
            rec.rows.push_back(std::stoll(v));
        }
        if ((int) rec.rows.size() == rec.n_experts) {
            out.push_back(std::move(rec));
        }
    }
    if (out.empty()) {
        error = path + " holds no usable record";
        return false;
    }
    return true;
}

// Expert e gets exactly rows[e] distinct tokens: the assignments are laid out expert by expert and dealt to a
// shuffled token order, so the rows[e] <= n_tokens consecutive assignments of expert e land on different tokens.
static inline void routing_replay(const params & p, const replay_record & rec, std::mt19937 & rng, routing & out) {
    out.name = "replay:L" + std::to_string(rec.layer) + ":U" + std::to_string(rec.ubatch);
    std::vector<int64_t> tokens(rec.n_tokens);
    std::iota(tokens.begin(), tokens.end(), 0);
    std::shuffle(tokens.begin(), tokens.end(), rng);
    int64_t pos = 0;
    for (int e = 0; e < p.n_experts; ++e) {
        for (int64_t q = 0; q < rec.rows[e]; ++q, ++pos) {
            out.ids[(size_t) tokens[pos % rec.n_tokens] * p.top_k + pos / rec.n_tokens] = e;
        }
    }
}

static inline void route_stats(routing & out, int n_experts) {
    out.histogram.assign(n_experts, 0);
    for (int32_t expert : out.ids) {
        ++out.histogram[expert];
    }
    int64_t sum = 0;
    out.n_active = 0;
    out.tpe_min = std::numeric_limits<int64_t>::max();
    out.tpe_max = 0;
    for (int64_t count : out.histogram) {
        if (!count) {
            continue;
        }
        ++out.n_active;
        sum += count;
        out.tpe_min = std::min(out.tpe_min, count);
        out.tpe_max = std::max(out.tpe_max, count);
    }
    if (out.n_active == 0) {
        out.tpe_min = 0;
        return;
    }
    out.tpe_mean = (double) sum / out.n_active;
}

static inline bool build(const std::string & dist, const params & p, routing & out, std::string & error) {
    if (p.n_experts <= 0 || p.top_k <= 0 || p.top_k > p.n_experts || p.n_tokens <= 0) {
        error = "invalid routing shape: need 0 < top_k <= n_experts and n_tokens > 0";
        return false;
    }

    out = routing();
    out.ids.assign((size_t) p.top_k * p.n_tokens, 0);
    std::mt19937 rng(p.seed);

    if (dist == "uniform") {
        routing_uniform(p, rng, out);
    } else if (dist == "single") {
        routing_single(p, out);
    } else if (dist == "zipf") {
        routing_zipf(p, rng, out);
    } else if (dist == "skewed") {
        routing_skewed(p, rng, out);
    } else if (dist == "hottail") {
        // The tail is drawn from the experts after the hot one, so a token cannot fill top_k distinct slots
        // when every expert is used.
        if (p.top_k >= p.n_experts) {
            error = "hottail needs top_k < n_experts";
            return false;
        }
        routing_hottail(p, rng, out);
    } else if (dist == "concentration") {
        const int64_t total = p.n_tokens * p.top_k;
        if (p.tpe_target <= 0 || total % p.tpe_target != 0) {
            error = "concentration needs tpe to divide n_tokens*top_k = " + std::to_string(total);
            return false;
        }
        const int64_t active = total / p.tpe_target;
        if (active < p.top_k || active > p.n_experts) {
            error = "concentration tpe " + std::to_string(p.tpe_target) + " needs " + std::to_string(active) +
                    " active experts, outside [" + std::to_string(p.top_k) + ", " + std::to_string(p.n_experts) + "]";
            return false;
        }
        routing_concentration(p, (int) active, rng, out);
    } else if (dist == "replay") {
        if (p.replay_file.empty()) {
            error = "replay needs a replay=FILE argument";
            return false;
        }
        std::vector<replay_record> records;
        if (!read_replay_file(p.replay_file, records, error)) {
            return false;
        }
        const replay_record * chosen = nullptr;
        for (const replay_record & rec : records) {
            if (rec.n_tokens != p.n_tokens || rec.top_k != p.top_k || rec.n_experts != p.n_experts) {
                continue;
            }
            if (p.replay_layer >= 0 && rec.layer != p.replay_layer) {
                continue;
            }
            if (p.replay_ubatch >= 0 && rec.ubatch != p.replay_ubatch) {
                continue;
            }
            // The record must account for every assignment, otherwise routing_replay leaves ids unwritten or
            // runs past the end of the token order.
            const int64_t total = std::accumulate(rec.rows.begin(), rec.rows.end(), (int64_t) 0);
            if (total != rec.n_tokens * rec.top_k) {
                continue;
            }
            chosen = &rec;
            break;
        }
        if (!chosen) {
            error = "no replay record for n_tokens=" + std::to_string(p.n_tokens) + " top_k=" +
                    std::to_string(p.top_k) + " n_experts=" + std::to_string(p.n_experts);
            return false;
        }
        routing_replay(p, *chosen, rng, out);
    } else {
        error = "unknown routing distribution '" + dist + "', valid: " + dist_names();
        return false;
    }

    route_stats(out, p.n_experts);
    return true;
}

} // namespace moe_routing
