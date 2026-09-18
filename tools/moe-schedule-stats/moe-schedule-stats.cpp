// delphi.cpp: routing statistics and grouped schedule construction cost of a real MoE model.
//
// The eval callback copies each ffn_moe_topk-<layer> tensor (the router's top-k expert ids) to the host; the model's own
// computation is not changed. For each (layer, ubatch) the tool builds the grouped schedule of ggml-moe-schedule.h, records the
// rows per expert, and times the host reference builder and, when a CUDA/HIP device exposes them, the device builders.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include "../../ggml/src/ggml-moe-schedule.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using build_host_fn = bool (*)(ggml_backend_t, int, const int32_t *, int64_t, int, int, int32_t *, int32_t *, int32_t *, int, float *);

struct routing_record {
    int                     layer     = 0;
    int                     ubatch    = 0;
    int64_t                 n_tokens  = 0;
    int                     top_k     = 0;
    int                     n_experts = 0;
    std::vector<int32_t>    ids;       // [n_tokens][top_k]
    std::vector<int32_t>    offsets;
    ggml_moe_schedule_stats st        = {};
    double                  host_us   = 0.0;
    float                   device_us = NAN;
    float                   helper_us = NAN;
};

struct collector {
    std::vector<routing_record> records;
    std::map<int, int>          calls; // per layer, so the n-th call of a layer is ubatch n
    std::vector<int32_t>        buf;
};

static const char k_prefix[] = "ffn_moe_topk-";

static bool collect_routing(ggml_tensor * t, bool ask, void * user_data) {
    const bool match = strncmp(t->name, k_prefix, sizeof(k_prefix) - 1) == 0;
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }

    auto * c = (collector *) user_data;
    GGML_ASSERT(t->type == GGML_TYPE_I32 && t->nb[0] == sizeof(int32_t) && t->ne[2] == 1 && t->ne[3] == 1);

    routing_record r;
    r.layer     = atoi(t->name + sizeof(k_prefix) - 1);
    r.ubatch    = c->calls[r.layer]++;
    r.top_k     = (int) t->ne[0];
    r.n_tokens  = t->ne[1];
    // the top-k tensor is a view of the argsort over all experts
    r.n_experts = (int) (t->src[0] ? t->src[0]->ne[0] : t->nb[1]/sizeof(int32_t));

    const int64_t s1 = t->nb[1]/sizeof(int32_t);
    c->buf.resize(ggml_nbytes(t)/sizeof(int32_t));
    ggml_backend_tensor_get(t, c->buf.data(), 0, ggml_nbytes(t));
    r.ids.resize(r.n_tokens*r.top_k);
    for (int64_t i = 0; i < r.n_tokens; ++i) {
        std::copy_n(c->buf.data() + i*s1, r.top_k, r.ids.data() + i*r.top_k);
    }
    c->records.push_back(std::move(r));
    return true;
}

static double quantile(std::vector<int32_t> & v, double q) {
    return ggml_moe_schedule_quantile(v.data(), (int) v.size(), q);
}

static double median(std::vector<double> v) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

static void analyze(std::vector<routing_record> & records, int reps, ggml_backend_t backend, build_host_fn build_host) {
    int n_device_mismatch = 0;
    for (routing_record & r : records) {
        const int64_t n_assign = r.n_tokens*r.top_k;
        r.offsets.resize(r.n_experts + 1);
        std::vector<int32_t> slot(n_assign);
        std::vector<int32_t> inverse(n_assign);

        std::vector<double> host_us;
        for (int i = 0; i < std::max(reps, 1); ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_moe_schedule_build_ref(r.ids.data(), r.top_k, r.n_tokens, r.top_k, r.n_experts, r.offsets.data(), slot.data(), inverse.data());
            host_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        r.host_us = median(host_us);

        std::vector<int32_t> scratch(r.n_experts);
        ggml_moe_schedule_get_stats(r.offsets.data(), r.n_experts, n_assign, scratch.data(), &r.st);

        if (build_host) {
            std::vector<int32_t> d_offsets(r.n_experts + 1);
            std::vector<int32_t> d_slot(n_assign);
            std::vector<int32_t> d_inverse(n_assign);
            if (build_host(backend, 0, r.ids.data(), r.n_tokens, r.top_k, r.n_experts, d_offsets.data(), d_slot.data(), d_inverse.data(), reps, &r.device_us)) {
                n_device_mismatch += d_offsets != r.offsets || d_slot != slot || d_inverse != inverse;
            }
            if (!build_host(backend, 1, r.ids.data(), r.n_tokens, r.top_k, r.n_experts, d_offsets.data(), d_slot.data(), d_inverse.data(), reps, &r.helper_us)) {
                r.helper_us = NAN;
            }
        }
    }
    if (n_device_mismatch) {
        LOG_ERR("device schedule differs from the host reference on %d of %zu records\n", n_device_mismatch, records.size());
    }
}

static void print_summary(const std::vector<routing_record> & records) {
    if (records.empty()) {
        LOG_ERR("no ffn_moe_topk tensors were observed\n");
        return;
    }

    std::map<int64_t, std::vector<const routing_record *>> by_tokens;
    for (const routing_record & r : records) {
        by_tokens[r.n_tokens].push_back(&r);
    }

    for (const auto & it : by_tokens) {
        const auto & recs = it.second;
        const routing_record & r0 = *recs[0];
        std::map<int, bool> layers;
        std::vector<double> active, p50, p90, p99, maxr, mom, neff, host, dev, helper;
        std::vector<int32_t> pooled;
        int64_t n_empty = 0;
        for (const routing_record * r : recs) {
            layers[r->layer] = true;
            active.push_back(r->st.n_active);
            p50.push_back(r->st.rows_p50);
            p90.push_back(r->st.rows_p90);
            p99.push_back(r->st.rows_p99);
            maxr.push_back(r->st.rows_max);
            mom.push_back(r->st.max_over_mean);
            neff.push_back(r->st.n_eff);
            host.push_back(r->host_us);
            dev.push_back(r->device_us);
            helper.push_back(r->helper_us);
            for (int e = 0; e < r->n_experts; ++e) {
                const int32_t rows = r->offsets[e + 1] - r->offsets[e];
                if (rows > 0) {
                    pooled.push_back(rows);
                } else {
                    n_empty++;
                }
            }
        }
        std::sort(pooled.begin(), pooled.end());
        auto mean = [](const std::vector<double> & v) {
            double s = 0.0;
            for (double x : v) {
                s += x;
            }
            return v.empty() ? NAN : s/v.size();
        };
        auto frac_le = [&](int32_t j) {
            return 100.0*(std::upper_bound(pooled.begin(), pooled.end(), j) - pooled.begin())/pooled.size();
        };

        printf("\n%zu records: %zu layers, %" PRId64 " tokens per ubatch, top-%d of %d experts, uniform mean %.1f rows per expert\n",
            recs.size(), layers.size(), r0.n_tokens, r0.top_k, r0.n_experts, (double) r0.n_tokens*r0.top_k/r0.n_experts);
        printf("\nmean over (layer, ubatch) records; construction times are medians\n\n");
        printf("| active | rows p50 | p90 | p99 | max | max/mean | n_eff | KiB | host us | device us | mm_ids_helper us |\n");
        printf("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
        printf("| %.1f | %.1f | %.1f | %.1f | %.1f | %.2f | %.1f | %.1f | %.1f | %.1f | %.1f |\n",
            mean(active), mean(p50), mean(p90), mean(p99), mean(maxr), mean(mom), mean(neff), r0.st.nbytes/1024.0,
            median(host), median(dev), median(helper));
        printf("\nrows per active expert, pooled over all records; empty experts %.1f%%\n\n", 100.0*n_empty/(n_empty + pooled.size()));
        printf("| p10 | p25 | p50 | p75 | p90 | p99 | max | <=8 | <=16 | <=32 | <=64 |\n");
        printf("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
        printf("| %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %d | %.1f%% | %.1f%% | %.1f%% | %.1f%% |\n",
            quantile(pooled, 0.10), quantile(pooled, 0.25), quantile(pooled, 0.50), quantile(pooled, 0.75), quantile(pooled, 0.90),
            quantile(pooled, 0.99), pooled.back(), frac_le(8), frac_le(16), frac_le(32), frac_le(64));
    }
}

static bool write_csv(const std::string & path, const std::vector<routing_record> & records) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        return false;
    }
    fprintf(f, "layer,ubatch,n_tokens,top_k,n_experts,n_active,rows_min,rows_p50,rows_p90,rows_p99,rows_max,max_over_mean,n_eff,nbytes,host_us,device_us,helper_us,rows\n");
    for (const routing_record & r : records) {
        fprintf(f, "%d,%d,%" PRId64 ",%d,%d,%d,%d,%.1f,%.1f,%.1f,%d,%.3f,%.2f,%zu,%.2f,%.2f,%.2f,",
            r.layer, r.ubatch, r.n_tokens, r.top_k, r.n_experts, r.st.n_active, r.st.rows_min, r.st.rows_p50, r.st.rows_p90,
            r.st.rows_p99, r.st.rows_max, r.st.max_over_mean, r.st.n_eff, r.st.nbytes, r.host_us, r.device_us, r.helper_us);
        for (int e = 0; e < r.n_experts; ++e) {
            fprintf(f, "%s%d", e ? ";" : "", r.offsets[e + 1] - r.offsets[e]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return true;
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -ngl 999 -fa on -c 4096 -b 2048 -ub 512 --moe-random --moe-tokens 2048 [--moe-csv out.csv] [--moe-reps 20]\n", argv[0]);
    LOG("    %s -m model.gguf -ngl 999 -f prompt.txt --moe-tokens 512\n\n", argv[0]);
}

int main(int argc, char ** argv) {
    int64_t     n_prompt      = 512;
    bool        random_tokens = false;
    std::string csv_path;
    int         reps          = 20;

    // own options first; the rest goes to the common parser
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--moe-tokens" && i + 1 < argc) {
            n_prompt = std::stoll(argv[++i]);
        } else if (arg == "--moe-random") {
            random_tokens = true;
        } else if (arg == "--moe-csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--moe-reps" && i + 1 < argc) {
            reps = std::stoi(argv[++i]);
        } else {
            rest.push_back(argv[i]);
        }
    }

    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }
    common_init();

    collector c;
    params.cb_eval           = collect_routing;
    params.cb_eval_user_data = &c;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens;
    if (random_tokens) {
        // as llama-bench builds its prompt
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        for (int64_t i = 0; i < n_prompt; ++i) {
            tokens.push_back(i == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab);
        }
    } else {
        tokens = common_tokenize(ctx, params.prompt, true, true);
        if ((int64_t) tokens.size() < n_prompt) {
            LOG_ERR("%s: the prompt has %zu tokens, %" PRId64 " requested\n", __func__, tokens.size(), n_prompt);
            return 1;
        }
        tokens.resize(n_prompt);
    }

    const int n_batch = (int) llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int64_t i0 = 0; i0 < n_prompt; i0 += n_batch) {
        common_batch_clear(batch);
        const int64_t n = std::min<int64_t>(n_batch, n_prompt - i0);
        for (int64_t i = 0; i < n; ++i) {
            common_batch_add(batch, tokens[i0 + i], (llama_pos) (i0 + i), { 0 }, i0 + i == n_prompt - 1);
        }
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s: llama_decode failed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
    }
    llama_synchronize(ctx);
    llama_batch_free(batch);

    ggml_backend_t backend    = nullptr;
    build_host_fn  build_host = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && !build_host; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        build_host = (build_host_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_moe_schedule_build_host");
        if (build_host) {
            backend = ggml_backend_dev_init(dev, nullptr);
            LOG_INF("%s: device builders timed on %s\n", __func__, ggml_backend_dev_name(dev));
        }
    }

    analyze(c.records, reps, backend, build_host);
    print_summary(c.records);
    if (!csv_path.empty() && !write_csv(csv_path, c.records)) {
        LOG_ERR("%s: cannot write %s\n", __func__, csv_path.c_str());
    }

    if (backend) {
        ggml_backend_free(backend);
    }
    llama_backend_free();
    return 0;
}
