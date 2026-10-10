#include "quant-damage.h"

#include "build-info.h"
#include "common.h"
#include "imatrix-loader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static void print_usage(const char * exe) {
    printf("usage: %s -m MODEL.gguf --qtypes T1,T2,... [options]\n\n", exe);
    printf("Estimates the activation-weighted quantization damage of each candidate type from a sample of tensor rows.\n\n");
    printf("  -m, --model FILE          source GGUF (first file of a split model)\n");
    printf("  --imatrix FILE            importance matrix (GGUF or legacy .dat)\n");
    printf("  --qtypes LIST             comma-separated ggml types, e.g. q3_K,q4_K,iq4_ks\n");
    printf("  --tensor REGEX            include tensors that match (repeatable, default: all)\n");
    printf("  --exclude REGEX           exclude tensors that match (repeatable)\n");
    printf("  --min-elements N          skip smaller tensors (default: 1048576)\n");
    printf("  --samples N               rows in the first round (default: 64)\n");
    printf("  --max-samples N           row budget (default: 4096)\n");
    printf("  --max-sample-mib N        budget of sampled F32 source data in MiB (default: 1024)\n");
    printf("  --min-per-expert N        rows measured in every expert: its most important rows, or with --expert-strata the rows per expert stratum (default: 1)\n");
    printf("  --strata N                importance levels over the tensor, or per expert with --expert-strata (default: 8)\n");
    printf("  --no-row-stats            do not read the full tensor for per-row statistics\n");
    printf("  --no-outlier-stratum      do not keep outlier rows in a stratum of their own\n");
    printf("  --expert-strata           stratify merged expert tensors by expert first (many small strata; weaker variance estimates)\n");
    printf("  --certainty-factor F      always measure a row whose importance is >= 1/F of the mean per sampled row, 0 = off (default: 8)\n");
    printf("  --weighting W             primary expert weighting: mass, count or uniform (default: mass)\n");
    printf("  --rse X                   stop when every candidate has relative SE <= X (default: 0.03)\n");
    printf("  --rank-z X                or when every adjacent pair is ordered at z >= X, 0 = off (default: 0)\n");
    printf("  --dominance-z X           z for quality/size dominance (default: 3)\n");
    printf("  --ci-z X                  z of the reported confidence interval (default: 1.96)\n");
    printf("  --fixed                   run every round up to the budget\n");
    printf("  --allocation A            later rounds: proxy (Neyman under the noise proxy), observed (Neyman on observed residuals, biased with many small strata) or first (default: proxy)\n");
    printf("  --seed N                  sampling seed (default: 1234)\n");
    printf("  --repeat N                independent runs with seeds seed..seed+N-1 (default: 1)\n");
    printf("  --exact                   also quantize every row for exact values (validation)\n");
    printf("  --no-sampled              with --exact: skip the sampled estimate\n");
    printf("  --dump-rows FILE          with --exact: write the per-row values of the last tensor as CSV\n");
    printf("  --check-locality          test that rows and sub-row spans quantize independently\n");
    printf("  --per-expert              write per-expert ratios to the JSON output\n");
    printf("  --tie NAME:REGEX          group every tensor matching REGEX with its fused partner (the first\n");
    printf("                            'gate' in the name replaced by 'up'); the group must share one qtype,\n");
    printf("                            member damages stay separate (repeatable, JSON only)\n");
    printf("  --list                    list matching tensors and exit\n");
    printf("  -t, --threads N           threads (default: %d)\n", common_cpu_get_num_math());
    printf("  --json FILE               write JSON\n");
    printf("  --csv FILE                write CSV, one record per (tensor, run, qtype)\n");
}

static std::string shape_str(const qd_tensor_info & t) {
    std::string s = "[";
    for (int d = 0; d < std::max(1, t.n_dims); ++d) {
        s += (d ? ", " : "") + std::to_string(t.ne[d]);
    }
    return s + "]";
}

static double num(double v) {
    return std::isfinite(v) ? v : 0.0;
}

static json jnum(double v) {
    return std::isfinite(v) ? json(v) : json(nullptr);
}

static double imatrix_coverage(const qd_importance & imp, int64_t n_expert) {
    if (!imp.has_imatrix) {
        return 0.0;
    }
    return 1.0 - (double) imp.n_missing / (double) n_expert;
}

static std::string hex64(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016" PRIx64, v);
    return buf;
}

// routing weight provenance of one tensor: where the expert weights came from and what they are
static json routing_json(const qd_importance & imp, int64_t n_expert, int64_t n_expert_used) {
    json j;
    j["counts_source"]  = imp.counts_source;
    j["n_expert_used"]  = n_expert_used;
    j["coverage"]       = imatrix_coverage(imp, n_expert);
    j["experts_missing"] = imp.n_missing;
    if (!imp.note.empty()) {
        j["note"] = imp.note;
    }
    if (n_expert > 1) {
        json counts = json::array(), missing = json::array(), weights = json::object();
        for (int64_t e = 0; e < n_expert; ++e) {
            counts.push_back(jnum(imp.counts[e]));
            missing.push_back(imp.expert_missing[e] != 0);
        }
        for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
            json wj = json::array();
            for (int64_t e = 0; e < n_expert; ++e) {
                wj.push_back(jnum(imp.w[w][e]));
            }
            weights[qd_weighting_name((qd_weighting) w)] = wj;
        }
        j["expert_counts"]  = counts;
        j["expert_weights"] = weights;
        j["expert_missing"] = missing;
    }
    return j;
}

static json run_json(const qd_tensor_result & r, qd_weighting primary, bool per_expert) {
    json j;
    j["estimator"]       = r.estimator;
    j["encoder"]         = QD_ENCODER_GGML;
    j["seed"]            = r.seed;
    j["units_sampled"]   = r.units_sampled;
    j["units_total"]     = r.units_total;
    j["sample_fraction"] = (double) r.units_sampled / (double) r.units_total;
    j["sample_weights"]  = r.units_sampled * r.info.n_per_row();
    j["n_strata"]        = r.n_strata;
    j["experts_sampled"] = r.experts_sampled;
    j["seconds_scan"]    = r.seconds_scan;
    j["seconds_eval"]    = r.seconds_eval;
    j["bytes_scan"]      = r.bytes_scan;
    j["bytes_sample"]    = r.bytes_sample;
    json sig;
    for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
        sig[qd_weighting_name((qd_weighting) w)] = jnum(r.signal[w]);
    }
    j["signal"] = sig;
    j["energy"] = jnum(r.energy);

    json types = json::array();
    for (const auto & t : r.types) {
        json jt;
        jt["qtype"]  = ggml_type_name(t.type);
        jt["status"] = qd_status_name(t.status);
        if (!t.note.empty()) {
            jt["note"] = t.note;
        }
        jt["bytes"] = t.bytes;
        jt["bpw"]   = t.bpw;
        if (t.status == QD_STATUS_OK) {
            const auto & m = t.weighted[primary];
            jt["D"]       = jnum(m.total.value);
            jt["D_se"]    = jnum(m.total.se);
            jt["R"]       = jnum(m.rel.value);
            jt["R_se"]    = jnum(m.rel.se);
            jt["R_ci_lo"] = jnum(m.ci_lo);
            jt["R_ci_hi"] = jnum(m.ci_hi);
            json wj;
            for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
                const auto & mw = t.weighted[w];
                wj[qd_weighting_name((qd_weighting) w)] = { {"D", jnum(mw.total.value)}, {"D_se", jnum(mw.total.se)}, {"R", jnum(mw.rel.value)}, {"R_se", jnum(mw.rel.se)} };
            }
            jt["weightings"]  = wj;
            jt["sse"]         = jnum(t.unweighted.total.value);
            jt["sse_se"]      = jnum(t.unweighted.total.se);
            jt["rel_mse"]     = jnum(t.unweighted.rel.value);
            jt["rel_mse_se"]  = jnum(t.unweighted.rel.se);
            jt["mse"]         = jnum(t.mse);
            if (t.share_top1_signal >= 0.0) {
                jt["share_top1_signal"]  = t.share_top1_signal;
                jt["share_top10_signal"] = t.share_top10_signal;
                jt["share_top1_damage"]  = t.share_top1_damage;
            }
            jt["rank"]        = t.rank;
            jt["dominated_by"] = t.dominated_by >= 0 ? json(ggml_type_name(r.types[t.dominated_by].type)) : json(nullptr);
        }
        types.push_back(jt);
    }
    j["types"] = types;

    json rounds = json::array();
    for (const auto & rd : r.rounds) {
        json jr;
        jr["n_units"] = rd.n_units;
        json R = json::array(), se = json::array();
        for (size_t q = 0; q < rd.rel.size(); ++q) {
            R.push_back(jnum(rd.rel[q]));
            se.push_back(jnum(rd.rel_se[q]));
        }
        jr["R"]    = R;
        jr["R_se"] = se;
        rounds.push_back(jr);
    }
    j["rounds"] = rounds;

    if (per_expert) {
        json ex = json::array();
        for (const auto & e : r.experts) {
            json je;
            je["n"] = e.n_sampled;
            json R = json::array();
            for (double v : e.rel) {
                R.push_back(jnum(v));
            }
            je["R"] = R;
            ex.push_back(je);
        }
        j["experts"] = ex;
    }
    return j;
}

static void print_result(const qd_tensor_result & r, qd_weighting primary) {
    const auto & t = r.info;
    printf("\n%s %s %s", t.name.c_str(), shape_str(t).c_str(), ggml_type_name(t.type));
    if (t.n_expert() > 1) {
        printf(", %" PRId64 " experts (%" PRId64 " sampled, %" PRId64 " without imatrix data)", t.n_expert(), r.experts_sampled, r.imp.n_missing);
    }
    printf("\n  %s, seed %" PRIu64 ": %" PRId64 "/%" PRId64 " rows (%.3f%%), %" PRId64 " strata, %zu rounds, scan %.2f s, eval %.2f s\n",
           r.estimator.c_str(), r.seed, r.units_sampled, r.units_total, 100.0 * r.units_sampled / r.units_total, r.n_strata, r.rounds.size(), r.seconds_scan, r.seconds_eval);
    if (!r.imp.has_imatrix) {
        printf("  NO IMATRIX (%s): weighted values are unweighted\n", r.imp.note.c_str());
    } else if (!r.imp.note.empty()) {
        printf("  imatrix: %s\n", r.imp.note.c_str());
    }

    std::vector<size_t> order(r.types.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const bool oka = r.types[a].status == QD_STATUS_OK, okb = r.types[b].status == QD_STATUS_OK;
        if (oka != okb) {
            return oka;
        }
        return oka && r.types[a].weighted[primary].rel.value < r.types[b].weighted[primary].rel.value;
    });

    printf("  %-8s %6s  %-24s %-23s %-11s %s\n", "qtype", "bpw", "R (rel. weighted)", "95% CI", "rel. MSE", "note");
    for (size_t q : order) {
        const auto & ty = r.types[q];
        if (ty.status != QD_STATUS_OK) {
            printf("  %-8s %6.3f  %s%s%s\n", ggml_type_name(ty.type), ty.bpw, qd_status_name(ty.status), ty.note.empty() ? "" : ": ", ty.note.c_str());
            continue;
        }
        const auto & m = ty.weighted[primary];
        char val[64], ci[64], note[128] = "";
        snprintf(val, sizeof(val), "%.4e +- %.2e", m.rel.value, num(m.rel.se));
        snprintf(ci, sizeof(ci), "[%.3e, %.3e]", m.ci_lo, m.ci_hi);
        if (ty.dominated_by >= 0) {
            snprintf(note, sizeof(note), "dominated by %s", ggml_type_name(r.types[ty.dominated_by].type));
        }
        printf("  %-8s %6.3f  %-24s %-23s %-11.4e %s\n", ggml_type_name(ty.type), ty.bpw, val, ci, ty.unweighted.rel.value, note);
    }
}

static void write_csv_header(FILE * f) {
    fprintf(f, "tensor,shape,source_dtype,qtype,estimator,seed,status,estimated_weighted_error,weighted_error_se,estimated_relative_weighted_error,standard_error,confidence_low,confidence_high,"
               "relative_weighted_error_count,relative_weighted_error_uniform,estimated_unweighted_error,estimated_relative_unweighted_error,relative_unweighted_error_se,"
               "sample_spans,sample_weights,sample_fraction,imatrix_coverage,experts_total,experts_sampled,bytes,bpw,rank,dominated_by,notes\n");
}

static void write_csv(FILE * f, const qd_tensor_result & r, qd_weighting primary) {
    const auto & t = r.info;
    std::string shape = shape_str(t);
    std::replace(shape.begin(), shape.end(), ',', 'x');
    shape.erase(std::remove(shape.begin(), shape.end(), ' '), shape.end());
    for (const auto & ty : r.types) {
        const auto & m = ty.weighted[primary];
        std::string note = ty.note;
        if (!r.imp.has_imatrix) {
            note += std::string(note.empty() ? "" : "; ") + "no imatrix";
        }
        std::replace(note.begin(), note.end(), ',', ';');
        const bool ok = ty.status == QD_STATUS_OK;
        fprintf(f, "%s,%s,%s,%s,%s,%" PRIu64 ",%s,", t.name.c_str(), shape.c_str(), ggml_type_name(t.type), ggml_type_name(ty.type), r.estimator.c_str(), r.seed, qd_status_name(ty.status));
        if (ok) {
            fprintf(f, "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,", m.total.value, m.total.se, m.rel.value, m.rel.se, m.ci_lo, m.ci_hi,
                    ty.weighted[QD_WEIGHTING_COUNT].rel.value, ty.weighted[QD_WEIGHTING_UNIFORM].rel.value, ty.mse, ty.unweighted.rel.value, ty.unweighted.rel.se);
        } else {
            fprintf(f, ",,,,,,,,,,,");
        }
        fprintf(f, "%" PRId64 ",%" PRId64 ",%.6e,%.4f,%" PRId64 ",%" PRId64 ",%zu,%.4f,", r.units_sampled, r.units_sampled * t.n_per_row(), (double) r.units_sampled / r.units_total,
                imatrix_coverage(r.imp, t.n_expert()), t.n_expert(), r.experts_sampled, ty.bytes, ty.bpw);
        if (ok) {
            fprintf(f, "%d,%s,", ty.rank, ty.dominated_by >= 0 ? ggml_type_name(r.types[ty.dominated_by].type) : "");
        } else {
            fprintf(f, ",,");
        }
        fprintf(f, "%s\n", note.c_str());
    }
}

int main(int argc, char ** argv) {
    std::string model_path, imatrix_path, json_path, csv_path;
    std::vector<std::string> include, exclude, tie_specs;
    std::string qtypes_arg;
    int64_t min_elements = 1 << 20;
    int     repeat = 1;
    bool    exact = false, sampled = true, check_locality = false, per_expert = false, list = false;
    qd_params params;
    params.n_threads = common_cpu_get_num_math();

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s needs a value\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "-m" || a == "--model")  { model_path = next(); }
        else if (a == "--imatrix")             { imatrix_path = next(); }
        else if (a == "--qtypes")              { qtypes_arg = next(); }
        else if (a == "--tensor")              { include.push_back(next()); }
        else if (a == "--exclude")             { exclude.push_back(next()); }
        else if (a == "--min-elements")        { min_elements = std::stoll(next()); }
        else if (a == "--samples")             { params.n_initial = std::stoll(next()); }
        else if (a == "--max-samples")         { params.n_max = std::stoll(next()); }
        else if (a == "--max-sample-mib")      { params.max_sample_mib = std::stoll(next()); }
        else if (a == "--min-per-expert")      { params.min_per_expert = std::stoll(next()); }
        else if (a == "--strata")              { params.n_levels = std::stoi(next()); }
        else if (a == "--no-row-stats")        { params.row_stats = false; }
        else if (a == "--no-outlier-stratum")  { params.outlier_stratum = false; }
        else if (a == "--expert-strata")       { params.expert_strata = true; }
        else if (a == "--certainty-factor")    { params.certainty_factor = std::stod(next()); }
        else if (a == "--rse")                 { params.rse_target = std::stod(next()); }
        else if (a == "--rank-z")              { params.rank_z = std::stod(next()); }
        else if (a == "--dominance-z")         { params.dominance_z = std::stod(next()); }
        else if (a == "--ci-z")                { params.ci_z = std::stod(next()); }
        else if (a == "--fixed")               { params.adaptive = false; }
        else if (a == "--allocation") {
            const std::string v = next();
            if (v == "proxy") params.allocation = QD_ALLOCATION_PROXY;
            else if (v == "observed") params.allocation = QD_ALLOCATION_OBSERVED;
            else if (v == "first") params.allocation = QD_ALLOCATION_FIRST;
            else { fprintf(stderr, "error: unknown allocation %s\n", v.c_str()); return 1; }
        }
        else if (a == "--seed")                { params.seed = std::stoull(next()); }
        else if (a == "--repeat")              { repeat = std::stoi(next()); }
        else if (a == "--exact")               { exact = true; }
        else if (a == "--no-sampled")          { sampled = false; }
        else if (a == "--dump-rows")           { params.dump_rows = next(); }
        else if (a == "--check-locality")      { check_locality = true; }
        else if (a == "--per-expert")          { per_expert = true; }
        else if (a == "--tie")                 { tie_specs.push_back(next()); }
        else if (a == "--list")                { list = true; }
        else if (a == "-t" || a == "--threads") { params.n_threads = std::stoi(next()); }
        else if (a == "--json")                { json_path = next(); }
        else if (a == "--csv")                 { csv_path = next(); }
        else if (a == "--weighting") {
            const std::string w = next();
            if (w == "mass") params.weighting = QD_WEIGHTING_MASS;
            else if (w == "count") params.weighting = QD_WEIGHTING_COUNT;
            else if (w == "uniform") params.weighting = QD_WEIGHTING_UNIFORM;
            else { fprintf(stderr, "error: unknown weighting %s\n", w.c_str()); return 1; }
        } else {
            fprintf(stderr, "error: unknown argument %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || (qtypes_arg.empty() && !list)) {
        print_usage(argv[0]);
        return 1;
    }

    for (const auto & name : string_split<std::string>(qtypes_arg, ',')) {
        if (name.empty()) {
            continue;
        }
        const ggml_type t = qd_type_from_name(name);
        if (t == GGML_TYPE_COUNT) {
            fprintf(stderr, "error: unknown ggml type '%s'\n", name.c_str());
            return 1;
        }
        params.qtypes.push_back(t);
    }

    qd_gguf_model model;
    std::string err;
    if (!model.open(model_path, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    common_imatrix imatrix;
    const bool have_imatrix = !imatrix_path.empty();
    if (have_imatrix && !common_imatrix_load(imatrix_path, imatrix)) {
        fprintf(stderr, "error: failed to load imatrix %s\n", imatrix_path.c_str());
        return 1;
    }

    std::vector<std::regex> inc, exc;
    for (const auto & r : include) inc.emplace_back(r);
    for (const auto & r : exclude) exc.emplace_back(r);

    std::vector<qd_tensor_info> selected;
    for (const auto & t : model.tensors()) {
        if (t.n_dims < 2 || t.n_elements() < min_elements) {
            continue;
        }
        bool ok = inc.empty();
        for (const auto & r : inc) ok = ok || std::regex_search(t.name, r);
        for (const auto & r : exc) ok = ok && !std::regex_search(t.name, r);
        if (ok) {
            selected.push_back(t);
        }
    }

    // tie groups: every tensor matching a --tie regex is measured together with its fused partner,
    // the first "gate" in the name replaced by "up" (the llama.cpp FFN naming: ffn_gate / ffn_up,
    // ffn_gate_exps / ffn_up_exps, ffn_gate_shexp / ffn_up_shexp). The partner joins the selection
    // even if the regexes excluded it, because a tied qtype can only be chosen with both measured.
    struct tie_group {
        std::string name;
        std::vector<std::string> tensors; // the matched tensor first, its partner second when found
    };
    std::vector<tie_group> tie_groups;
    if (!tie_specs.empty()) {
        std::vector<std::pair<std::string, std::regex>> ties;
        for (const auto & spec : tie_specs) {
            const size_t colon = spec.find(':');
            if (colon == std::string::npos || colon + 1 == spec.size()) {
                fprintf(stderr, "error: bad --tie '%s' (expected NAME:REGEX)\n", spec.c_str());
                return 1;
            }
            ties.emplace_back(spec.substr(0, colon), std::regex(spec.substr(colon + 1)));
        }
        std::map<std::string, const qd_tensor_info *> by_name;
        for (const auto & t : model.tensors()) {
            by_name[t.name] = &t;
        }
        std::set<std::string> grouped;
        std::vector<std::string> names;
        for (const auto & t : selected) {
            names.push_back(t.name);
        }
        for (const auto & name : names) {
            for (const auto & tie : ties) {
                if (grouped.count(name) || !std::regex_search(name, tie.second)) {
                    continue;
                }
                grouped.insert(name);
                tie_group g;
                g.name = tie.first;
                g.tensors.push_back(name);
                const size_t p = name.find("gate");
                if (p == std::string::npos) {
                    printf("warning: --tie %s matches %s, which has no gate/up partner\n", g.name.c_str(), name.c_str());
                } else {
                    const std::string partner = name.substr(0, p) + "up" + name.substr(p + 4);
                    const auto it = by_name.find(partner);
                    if (it == by_name.end()) {
                        printf("warning: --tie %s: %s has no %s in the model\n", g.name.c_str(), name.c_str(), partner.c_str());
                    } else if (!grouped.count(partner)) {
                        grouped.insert(partner);
                        g.tensors.push_back(partner);
                        const bool selected_partner = std::any_of(selected.begin(), selected.end(),
                                                                  [&](const qd_tensor_info & s) { return s.name == partner; });
                        if (!selected_partner) {
                            selected.push_back(*it->second);
                        }
                    }
                }
                tie_groups.push_back(std::move(g));
                break;
            }
        }
        printf("%s: %zu tie groups\n", __func__, tie_groups.size());
    }

    printf("%s: %s, arch %s, expert_used_count %" PRId64 ", metadata %s, %zu tensors selected, %d threads\n", __func__, model_path.c_str(), model.arch.c_str(), model.n_expert_used, hex64(model.metadata_hash).c_str(), selected.size(), params.n_threads);
    if (list) {
        for (const auto & t : selected) {
            printf("  %-40s %-24s %s\n", t.name.c_str(), shape_str(t).c_str(), ggml_type_name(t.type));
        }
        return 0;
    }

    json out;
    out["schema"] = QD_RESULT_SCHEMA;
    json calib = { {"path", imatrix_path} };
    if (have_imatrix) {
        uint64_t calib_hash = 0, calib_bytes = 0;
        if (!qd_hash_file(imatrix_path, calib_hash, calib_bytes)) {
            fprintf(stderr, "error: failed to hash imatrix %s\n", imatrix_path.c_str());
            return 1;
        }
        calib["kind"]        = imatrix.is_legacy ? "legacy-dat" : "gguf";
        calib["data_hash"]   = hex64(calib_hash);
        calib["file_bytes"]  = calib_bytes;
        calib["n_entries"]   = (int64_t) imatrix.entries.size();
        calib["datasets"]    = imatrix.datasets;
        calib["chunk_count"] = imatrix.chunk_count;
        calib["chunk_size"]  = imatrix.chunk_size;
    } else {
        calib["kind"] = "none";
    }
    out["identity"] = {
        {"model", {
             {"path", model_path}, {"n_files", model.n_files}, {"file_bytes", model.total_file_size},
             {"metadata_hash", hex64(model.metadata_hash)}, {"arch", model.arch}, {"n_expert_used", model.n_expert_used},
         }},
        {"calibration", calib},
        {"encoder", QD_ENCODER_GGML},
        {"build", { {"commit", llama_commit()}, {"number", llama_build_number()} }},
    };
    out["model"]         = model_path;
    out["imatrix"]       = imatrix_path;
    out["arch"]          = model.arch;
    out["n_expert_used"] = model.n_expert_used;
    out["weighting"]     = qd_weighting_name(params.weighting);
    out["params"] = {
        {"samples", params.n_initial}, {"max_samples", params.n_max}, {"max_sample_mib", params.max_sample_mib}, {"min_per_expert", params.min_per_expert},
        {"strata", params.n_levels}, {"row_stats", params.row_stats}, {"outlier_stratum", params.outlier_stratum}, {"expert_strata", params.expert_strata}, {"certainty_factor", params.certainty_factor}, {"rse", params.rse_target},
        {"rank_z", params.rank_z}, {"dominance_z", params.dominance_z}, {"ci_z", params.ci_z}, {"adaptive", params.adaptive}, {"allocation", params.allocation == QD_ALLOCATION_PROXY ? "proxy" : params.allocation == QD_ALLOCATION_OBSERVED ? "observed" : "first"}, {"seed", params.seed}, {"repeat", repeat},
    };
    json qt = json::array();
    for (auto t : params.qtypes) qt.push_back(ggml_type_name(t));
    out["qtypes"] = qt;
    json tensors = json::array();

    FILE * csv = nullptr;
    if (!csv_path.empty()) {
        csv = fopen(csv_path.c_str(), "w");
        if (!csv) {
            fprintf(stderr, "error: failed to open %s\n", csv_path.c_str());
            return 1;
        }
        write_csv_header(csv);
    }

    std::map<std::string, std::vector<qd_tensor_result>> results; // per tensor, runs in order (tie groups only)

    for (const auto & t : selected) {
        auto reader = model.reader(t.name);
        json jt;
        jt["name"]         = t.name;
        jt["shape"]        = std::vector<int64_t>(t.ne, t.ne + std::max(1, t.n_dims));
        jt["source_dtype"] = ggml_type_name(t.type);
        jt["n_expert"]     = t.n_expert();
        jt["identity"] = {
            {"name", t.name}, {"shape", jt["shape"]}, {"source_dtype", jt["source_dtype"]},
            {"split", t.split}, {"data_offset", t.data_offset}, {"data_bytes", t.data_bytes},
        };
        if (!reader) {
            printf("\n%s: source type %s cannot be read as F32, skipped\n", t.name.c_str(), ggml_type_name(t.type));
            jt["status"] = "unreadable_source";
            tensors.push_back(jt);
            continue;
        }

        const qd_importance imp = qd_importance_build(have_imatrix ? &imatrix : nullptr, t, model.n_expert_used);
        jt["imatrix"] = {
            {"present", imp.has_imatrix}, {"has_counts", imp.has_counts}, {"coverage", imatrix_coverage(imp, t.n_expert())},
            {"experts_missing", imp.n_missing}, {"note", imp.note},
        };
        jt["routing"] = routing_json(imp, t.n_expert(), model.n_expert_used);
        json runs = json::array();

        auto store = [&](qd_tensor_result & r) {
            if (tie_groups.empty()) {
                return;
            }
            // keep only what qd_tie_combine needs; the importance and per-unit data can be large
            r.imp = qd_importance{};
            r.experts.clear();
            r.rounds.clear();
            results[t.name].push_back(std::move(r));
        };

        if (check_locality) {
            json jl = json::array();
            printf("\n%s: locality check\n", t.name.c_str());
            for (const auto & l : qd_check_locality(*reader, imp, params.qtypes, 8, params.seed)) {
                printf("  %-8s %-22s rows %3d  row-local %-3s  span %5" PRId64 " %s\n", ggml_type_name(l.type), qd_status_name(l.status), l.rows_tried,
                       l.row_local ? "yes" : "NO", l.span, l.block_local == 1 ? "block-local" : l.block_local == 0 ? "NOT block-local" : "(row metadata or no sub-span)");
                jl.push_back({ {"qtype", ggml_type_name(l.type)}, {"status", qd_status_name(l.status)}, {"rows", l.rows_tried}, {"row_local", l.row_local},
                               {"span", l.span}, {"block_local", l.block_local} });
            }
            jt["locality"] = jl;
        }

        if (exact) {
            auto r = qd_exact_tensor(*reader, imp, params);
            print_result(r, params.weighting);
            runs.push_back(run_json(r, params.weighting, per_expert));
            if (csv) write_csv(csv, r, params.weighting);
            store(r);
        }
        if (sampled) {
            qd_row_stats stats;
            double scan_s = 0.0;
            if (params.row_stats) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!qd_scan_rows(*reader, imp, params.n_threads, stats)) {
                    fprintf(stderr, "error: failed to read %s\n", t.name.c_str());
                    return 1;
                }
                scan_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
            for (int rep = 0; rep < repeat; ++rep) {
                qd_params p = params;
                p.seed = params.seed + rep;
                auto r = qd_estimate_tensor(*reader, imp, p, params.row_stats ? &stats : nullptr);
                r.seconds_scan = scan_s;
                if (rep == 0 || repeat <= 4) {
                    print_result(r, params.weighting);
                }
                runs.push_back(run_json(r, params.weighting, per_expert));
                if (csv) write_csv(csv, r, params.weighting);
                store(r);
            }
        }
        jt["runs"] = runs;
        tensors.push_back(jt);
        if (csv) fflush(csv);
    }
    out["tensors"] = tensors;
    out["bytes_read"] = model.bytes_read();

    // tie groups: one aggregate record per member run, aligned by position in the runs lists
    if (!tie_groups.empty()) {
        json garr = json::array();
        for (const auto & g : tie_groups) {
            json jg;
            jg["name"]    = g.name;
            jg["members"] = g.tensors;
            json gruns  = json::array();
            size_t n_runs = SIZE_MAX;
            bool   complete = true;
            for (const auto & m : g.tensors) {
                const auto it = results.find(m);
                if (it == results.end()) {
                    complete = false;
                    break;
                }
                n_runs = std::min(n_runs, it->second.size());
            }
            if (complete) {
                for (size_t i = 0; i < n_runs; ++i) {
                    std::vector<qd_tensor_result> mem;
                    for (const auto & m : g.tensors) {
                        const auto & r = results[m][i];
                        const bool aligned = mem.empty() || (r.seed == mem[0].seed && r.estimator == mem[0].estimator);
                        if (!aligned) {
                            mem.clear();
                            break;
                        }
                        mem.push_back(r);
                    }
                    if (mem.empty()) {
                        continue;
                    }
                    const auto tr = qd_tie_combine(qd_tie_group{ g.name, g.tensors }, mem, params.ci_z);
                    json jr;
                    jr["estimator"] = tr.members[0].estimator;
                    jr["encoder"]   = QD_ENCODER_GGML;
                    jr["seed"]      = tr.members[0].seed;
                    json typesj = json::array();
                    for (size_t q = 0; q < tr.types.size(); ++q) {
                        const auto & tt = tr.types[q];
                        json jt2;
                        jt2["qtype"]  = ggml_type_name(tt.type);
                        jt2["status"] = qd_status_name(tt.status);
                        if (!tt.note.empty()) {
                            jt2["note"] = tt.note;
                        }
                        if (tt.status == QD_STATUS_OK) {
                            const auto & m = tt.weighted[params.weighting];
                            jt2["bytes"]   = tt.bytes;
                            jt2["bpw"]     = tt.bpw;
                            jt2["D"]       = jnum(m.total.value);
                            jt2["D_se"]    = jnum(m.total.se);
                            jt2["R"]       = jnum(m.rel.value);
                            jt2["R_se"]    = jnum(m.rel.se);
                            jt2["R_ci_lo"] = jnum(m.ci_lo);
                            jt2["R_ci_hi"] = jnum(m.ci_hi);
                            json memj = json::array();
                            for (const auto & r : tr.members) {
                                const auto & rt = r.types[q];
                                memj.push_back({
                                    {"tensor", r.info.name}, {"bytes", rt.bytes},
                                    {"R", jnum(rt.weighted[params.weighting].rel.value)},
                                    {"R_se", jnum(rt.weighted[params.weighting].rel.se)},
                                    {"status", qd_status_name(rt.status)},
                                });
                            }
                            jt2["members"] = memj;
                        }
                        typesj.push_back(jt2);
                    }
                    jr["types"] = typesj;
                    gruns.push_back(jr);
                }
            }
            if (!complete) {
                jg["note"] = "a member was skipped or unreadable; no group records";
            }
            jg["runs"] = gruns;
            garr.push_back(jg);
        }
        out["tie_groups"] = garr;
    }

    if (csv) {
        fclose(csv);
    }
    if (!json_path.empty()) {
        std::ofstream f(json_path);
        f << out.dump(1) << "\n";
    }
    printf("\n%s: read %.2f GiB from the model\n", __func__, model.bytes_read() / 1073741824.0);
    return 0;
}
