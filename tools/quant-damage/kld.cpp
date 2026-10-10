// Measures the model-level damage of quantizing one tensor at a time, to calibrate llama-quant-damage.
// The model is loaded once. For each (tensor, type) job, the tensor is replaced by its quantize -> dequantize round trip,
// the KL divergence to the reference logits is measured on a text, and the original data is restored.

#include "quant-damage.h"

#include "common.h"
#include "imatrix-loader.h"
#include "llama.h"
#include "llama-model.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static void print_usage(const char * exe) {
    printf("usage: %s -m MODEL.gguf -f TEXT --qtypes T1,... --tensor REGEX [options]\n\n", exe);
    printf("  -m FILE        model; its tensors are also the quantization source unless --source is given\n");
    printf("  --source FILE  source GGUF with the same tensors\n");
    printf("  --imatrix FILE importance matrix given to the quantizer\n");
    printf("  -f FILE        evaluation text\n");
    printf("  -c N           context per chunk (default: 512); the second half of each chunk is scored\n");
    printf("  --chunks N     chunks (default: 16)\n");
    printf("  --tensor REGEX tensors to perturb (repeatable)\n");
    printf("  --qtypes LIST  types; 'none' re-uploads the original data as a control, 'noise:R' adds Gaussian noise of relative power R\n");
    printf("  --recipe SPEC  one job that applies several assignments together: NAME:REGEX=TYPE;... (NAME without ':')\n");
    printf("  --router       also measure the change of the MoE router logits and expert selection per layer\n");
    printf("  --shard I/N    run every N-th job, starting at I\n");
    printf("  -ngl N         GPU layers (default: 999)\n");
    printf("  --device LIST  comma-separated backend device names\n");
    printf("  --numa TYPE    tensors, distribute, isolate or numactl\n");
    printf("  -cmoe          keep routed experts on the CPU\n");
    printf("  -t N           CPU threads for quantization (default: %d)\n", common_cpu_get_num_math());
    printf("  -o FILE        CSV output (appended)\n");
}

// one or more (tensor, type) assignments that are applied together
struct job {
    std::string label;
    std::string qtype;
    std::vector<std::pair<std::string, std::string>> parts; // (tensor, type): "none", "noise:R" or a ggml type name
};

// the router logits and the selected experts of every MoE layer, for all tokens of one pass over the text
struct router_capture {
    bool enabled = false;
    int  chunk   = 0;
    std::map<int, std::vector<float>>   logits; // layer -> n_expert x tokens
    std::map<int, std::vector<int32_t>> topk;   // layer -> n_used x tokens
    std::map<int, int> n_expert, n_used;
};

// copies the rows of a possibly non-contiguous 2D tensor into dst at the position of the current chunk
template <typename T>
static void router_store(const ggml_tensor * t, const router_capture & rc, std::vector<T> & dst) {
    const int64_t n = t->ne[0], rows = t->ne[1];
    std::vector<uint8_t> buf((rows - 1) * t->nb[1] + n * t->nb[0]);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    // the last layer may only compute the output rows, so the stride is the row count of the tensor
    const size_t off = (size_t) rc.chunk * rows * n;
    if (dst.size() < off + rows * n) {
        dst.resize(off + rows * n);
    }
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < n; ++j) {
            memcpy(&dst[off + r*n + j], buf.data() + r*t->nb[1] + j*t->nb[0], sizeof(T));
        }
    }
}

static bool router_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * rc = (router_capture *) user_data;
    const bool is_logits = strncmp(t->name, "ffn_moe_logits-", 15) == 0 && t->type == GGML_TYPE_F32;
    const bool is_topk   = strncmp(t->name, "ffn_moe_topk-", 13) == 0 && t->type == GGML_TYPE_I32;
    if (ask) {
        return rc->enabled && (is_logits || is_topk);
    }
    if (!rc->enabled || (!is_logits && !is_topk)) {
        return true;
    }
    const int il = atoi(strrchr(t->name, '-') + 1);
    if (is_logits) {
        rc->n_expert[il] = (int) t->ne[0];
        router_store(t, *rc, rc->logits[il]);
    } else {
        rc->n_used[il] = (int) t->ne[0];
        router_store(t, *rc, rc->topk[il]);
    }
    return true;
}

static void log_softmax(const float * logits, int n, std::vector<float> & out) {
    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = std::max(mx, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::exp((double) logits[i] - mx);
    const float lse = mx + (float) std::log(sum);
    out.resize(n);
    for (int i = 0; i < n; ++i) out[i] = logits[i] - lse;
}

int main(int argc, char ** argv) {
    std::string model_path, source_path, imatrix_path, text_path, out_path = "quant-damage-kld.csv", qtypes_arg;
    std::vector<std::string> tensor_re, recipes;
    int n_ctx = 512, n_chunks = 16, ngl = 999, shard_i = 0, shard_n = 1;
    bool router = false;
    bool cpu_moe = false;
    std::string device_list;
    common_params runtime;
    int n_threads = common_cpu_get_num_math();

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "-m")          { model_path = next(); }
        else if (a == "--source")    { source_path = next(); }
        else if (a == "--imatrix")   { imatrix_path = next(); }
        else if (a == "-f")          { text_path = next(); }
        else if (a == "-c")          { n_ctx = std::stoi(next()); }
        else if (a == "--chunks")    { n_chunks = std::stoi(next()); }
        else if (a == "--tensor")    { tensor_re.push_back(next()); }
        else if (a == "--recipe")    { recipes.push_back(next()); }
        else if (a == "--qtypes")    { qtypes_arg = next(); }
        else if (a == "--router")    { router = true; }
        else if (a == "--device")    { device_list = next(); }
        else if (a == "-cmoe" || a == "--cpu-moe") { cpu_moe = true; }
        else if (a == "--numa") {
            const std::string value = next();
            if      (value == "tensors")    { runtime.numa = GGML_NUMA_STRATEGY_TENSORS; }
            else if (value == "distribute") { runtime.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (value == "isolate")    { runtime.numa = GGML_NUMA_STRATEGY_ISOLATE; }
            else if (value == "numactl")    { runtime.numa = GGML_NUMA_STRATEGY_NUMACTL; }
            else { fprintf(stderr, "error: invalid NUMA strategy %s\n", value.c_str()); return 1; }
        }
        else if (a == "-ngl")        { ngl = std::stoi(next()); }
        else if (a == "-t")          { n_threads = std::stoi(next()); }
        else if (a == "-o")          { out_path = next(); }
        else if (a == "--shard") {
            const std::string s = next();
            if (sscanf(s.c_str(), "%d/%d", &shard_i, &shard_n) != 2 || shard_n < 1) { fprintf(stderr, "error: bad --shard\n"); return 1; }
        } else { fprintf(stderr, "error: unknown argument %s\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (model_path.empty() || text_path.empty() || ((tensor_re.empty() || qtypes_arg.empty()) && recipes.empty())) {
        print_usage(argv[0]);
        return 1;
    }
    if (source_path.empty()) {
        source_path = model_path;
    }

    qd_gguf_model source;
    std::string err;
    if (!source.open(source_path, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    common_imatrix imatrix;
    if (!imatrix_path.empty() && !common_imatrix_load(imatrix_path, imatrix)) {
        fprintf(stderr, "error: failed to load %s\n", imatrix_path.c_str());
        return 1;
    }

    std::vector<job> jobs;
    if (!qtypes_arg.empty()) {
        const auto qtypes = string_split<std::string>(qtypes_arg, ',');
        for (const auto & t : source.tensors()) {
            bool match = false;
            for (const auto & r : tensor_re) match = match || std::regex_search(t.name, std::regex(r));
            if (!match) continue;
            for (const auto & q : qtypes) {
                jobs.push_back({ t.name, q, { { t.name, q } } });
            }
        }
    }
    for (const auto & spec : recipes) {
        const size_t colon = spec.find(':');
        job jb;
        jb.label = colon == std::string::npos ? spec : spec.substr(0, colon);
        jb.qtype = "recipe";
        std::vector<std::pair<std::regex, std::string>> rules;
        for (const auto & part : string_split<std::string>(colon == std::string::npos ? spec : spec.substr(colon + 1), ';')) {
            const size_t eq = part.rfind('=');
            if (eq == std::string::npos) {
                fprintf(stderr, "error: bad recipe part '%s'\n", part.c_str());
                return 1;
            }
            rules.emplace_back(std::regex(part.substr(0, eq)), part.substr(eq + 1));
        }
        for (const auto & t : source.tensors()) {
            for (const auto & [re, q] : rules) {
                if (std::regex_search(t.name, re)) {
                    jb.parts.emplace_back(t.name, q);
                    break;
                }
            }
        }
        jobs.push_back(jb);
    }
    std::vector<job> mine;
    for (size_t i = 0; i < jobs.size(); ++i) {
        if ((int) (i % shard_n) == shard_i) mine.push_back(jobs[i]);
    }
    fprintf(stderr, "%s: %zu jobs, %zu in this shard\n", __func__, jobs.size(), mine.size());

    common_init();
    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(runtime.numa);
    if (!device_list.empty()) {
        for (const auto & name : string_split<std::string>(device_list, ',')) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
            if (!dev) {
                fprintf(stderr, "error: device %s is unavailable\n", name.c_str());
                return 1;
            }
            runtime.devices.push_back(dev);
        }
        runtime.devices.push_back(nullptr);
    }
    if (cpu_moe) {
        runtime.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        runtime.tensor_buft_overrides.push_back({nullptr, nullptr});
    }
    runtime.n_gpu_layers = ngl;
    runtime.load_mode = LLAMA_LOAD_MODE_NONE; // tensors on the CPU must be writable
    // A cached repack can retain the previous job's weights after a tensor update.
    runtime.no_extra_bufts = true;
    llama_model_params mp = common_model_params_to_llama(runtime);
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "error: failed to load %s\n", model_path.c_str());
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = n_ctx;
    cp.n_ubatch  = n_ctx;
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    router_capture rc;
    if (router) {
        cp.cb_eval           = router_cb;
        cp.cb_eval_user_data = &rc;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::ifstream tf(text_path);
    std::stringstream ss;
    ss << tf.rdbuf();
    std::vector<llama_token> tokens = common_tokenize(ctx, ss.str(), true);
    n_chunks = std::min<int>(n_chunks, (int) tokens.size() / n_ctx);
    const int first = n_ctx / 2;
    const int n_scored = n_chunks * (n_ctx - 1 - first);
    fprintf(stderr, "%s: %zu tokens, %d chunks of %d, %d scored tokens\n", __func__, tokens.size(), n_chunks, n_ctx, n_scored);

    // runs every chunk; fn(scored index, log-probs, target token)
    auto run = [&](const std::function<void(int, const std::vector<float> &, llama_token)> & fn) -> bool {
        llama_batch batch = llama_batch_init(n_ctx, 0, 1);
        int idx = 0;
        for (int c = 0; c < n_chunks; ++c) {
            llama_memory_clear(llama_get_memory(ctx), true);
            common_batch_clear(batch);
            rc.chunk = c;
            for (int i = 0; i < n_ctx; ++i) {
                llama_token tok = tokens[c*n_ctx + i];
                if (i == 0 && llama_vocab_get_add_bos(vocab)) {
                    tok = llama_vocab_bos(vocab);
                }
                common_batch_add(batch, tok, i, { 0 }, i >= first);
            }
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                return false;
            }
            // fn must be safe to call from several threads for different scored indices
            std::vector<std::thread> workers;
            for (int th = 0; th < n_threads; ++th) {
                workers.emplace_back([&, th]() {
                    std::vector<float> lp;
                    for (int i = first + th; i < n_ctx - 1; i += n_threads) {
                        log_softmax(llama_get_logits_ith(ctx, i), n_vocab, lp);
                        fn(idx + i - first, lp, tokens[c*n_ctx + i + 1]);
                    }
                });
            }
            for (auto & w : workers) w.join();
            idx += n_ctx - 1 - first;
        }
        llama_batch_free(batch);
        return true;
    };

    rc.enabled = router;
    std::vector<float> ref((size_t) n_scored * n_vocab);
    std::vector<double> ref_nlls(n_scored, 0.0);
    std::vector<llama_token> ref_top(n_scored);
    if (!run([&](int i, const std::vector<float> & lp, llama_token tgt) {
            std::copy(lp.begin(), lp.end(), ref.begin() + (size_t) i * n_vocab);
            ref_nlls[i] = -lp[tgt];
            ref_top[i] = (llama_token) (std::max_element(lp.begin(), lp.end()) - lp.begin());
        })) {
        fprintf(stderr, "error: decode failed\n");
        return 1;
    }
    double ref_nll = 0.0;
    for (double v : ref_nlls) ref_nll += v;
    const double ref_ppl = std::exp(ref_nll / n_scored);
    const auto ref_logits = rc.logits;
    const auto ref_topk   = rc.topk;
    if (router) {
        fprintf(stderr, "%s: captured the router of %zu layers\n", __func__, ref_logits.size());
        if (ref_logits.empty() || ref_topk.size() != ref_logits.size()) {
            fprintf(stderr, "error: no router logits or expert selection found\n");
            return 1;
        }
    }

    // per layer: mean squared change of the router logits per token, and fraction of tokens whose expert set changed
    auto router_diff = [&](std::vector<double> & e, std::vector<double> & flip) {
        e.clear();
        flip.clear();
        for (const auto & [il, rl] : ref_logits) {
            const auto & cl = rc.logits.at(il);
            const size_t n_tok = rl.size() / rc.n_expert.at(il);
            double sum = 0.0;
            for (size_t i = 0; i < rl.size(); ++i) {
                const double d = (double) cl[i] - rl[i];
                sum += d * d;
            }
            e.push_back(sum / n_tok);
            const int k = rc.n_used.at(il);
            const auto & rt = ref_topk.at(il);
            const auto & ct = rc.topk.at(il);
            size_t n_flip = 0;
            std::vector<int32_t> a(k), b(k);
            for (size_t i = 0; i < n_tok; ++i) {
                std::copy(rt.begin() + i*k, rt.begin() + (i + 1)*k, a.begin());
                std::copy(ct.begin() + i*k, ct.begin() + (i + 1)*k, b.begin());
                std::sort(a.begin(), a.end());
                std::sort(b.begin(), b.end());
                n_flip += a != b;
            }
            flip.push_back((double) n_flip / n_tok);
        }
    };
    auto join = [](const std::vector<double> & v) {
        std::string r;
        char buf[32];
        for (size_t i = 0; i < v.size(); ++i) {
            snprintf(buf, sizeof(buf), "%s%.4e", i ? ";" : "", v[i]);
            r += buf;
        }
        return r;
    };
    fprintf(stderr, "%s: reference PPL %.4f\n", __func__, ref_ppl);

    const auto & tmap = llama_internal_get_tensor_map(model);
    auto find_tensor = [&](const std::string & name) -> ggml_tensor * {
        for (const auto & [n, t] : tmap) {
            if (n == name) return t;
        }
        return nullptr;
    };

    FILE * out = fopen(out_path.c_str(), "a");
    if (!out) {
        fprintf(stderr, "error: failed to open %s\n", out_path.c_str());
        return 1;
    }
    fseek(out, 0, SEEK_END);
    if (ftell(out) == 0) {
        fprintf(out, "tensor,qtype,shape,bpw,D,R,rel_mse,kld_mean,kld_p99,kld_max,ppl,ref_ppl,top1,n_scored,seconds%s\n",
                router ? ",router_e,router_flip,router_e_layers,router_flip_layers" : "");
    }

    // when the model file is the source, the original bytes are rebuilt from it instead of downloaded from the device
    const bool restore_from_source = source_path == model_path;
    struct applied {
        ggml_tensor *        mt;
        std::string          name;
        std::vector<uint8_t> backup;
    };

    // replaces one model tensor by its round trip; adds the exact local damage of the values written
    auto perturb = [&](const std::string & name, const std::string & qtype, std::vector<applied> & undo,
                       double & D, double & S, double & U, double & E, double & bits, double & n_el, std::string & shape) -> bool {
        ggml_tensor * mt = find_tensor(name);
        auto reader = source.reader(name);
        if (!mt || !reader) {
            fprintf(stderr, "%s: skip %s (not found)\n", __func__, name.c_str());
            return false;
        }
        const qd_tensor_info & ti = reader->info();
        if (ggml_nelements(mt) != ti.n_elements() || (mt->type != GGML_TYPE_F32 && mt->type != GGML_TYPE_F16 && mt->type != GGML_TYPE_BF16)) {
            fprintf(stderr, "%s: skip %s (model tensor type %s)\n", __func__, name.c_str(), ggml_type_name(mt->type));
            return false;
        }
        const bool control = qtype == "none";
        // noise:<R> adds Gaussian noise of relative power R instead of quantizing, to trace KLD against the error magnitude
        const bool noise = qtype.rfind("noise:", 0) == 0;
        const double noise_r = noise ? std::stod(qtype.substr(6)) : 0.0;
        const ggml_type qt = control || noise ? GGML_TYPE_F32 : qd_type_from_name(qtype);
        if (qt == GGML_TYPE_COUNT || !qd_type_is_candidate(qt) || ti.n_per_row() % ggml_blck_size(qt) != 0) {
            fprintf(stderr, "%s: skip %s %s (type)\n", __func__, name.c_str(), qtype.c_str());
            return false;
        }
        const qd_importance imp = qd_importance_build(imatrix_path.empty() ? nullptr : &imatrix, ti, source.n_expert_used);
        if (ggml_quantize_requires_imatrix(qt) && !imp.has_imatrix) {
            fprintf(stderr, "%s: skip %s %s (needs imatrix)\n", __func__, name.c_str(), qtype.c_str());
            return false;
        }

        const int64_t k = ti.n_per_row(), n_rows = ti.n_rows(), rpe = ti.rows_per_expert();
        const size_t nbytes = ggml_nbytes(mt);
        applied a { mt, name, {} };
        if (!restore_from_source) {
            a.backup.resize(nbytes);
            ggml_backend_tensor_get(mt, a.backup.data(), 0, nbytes);
        }
        std::vector<uint8_t> pert(nbytes);

        std::vector<float> src(n_rows * k);
        if (!reader->read_rows(0, n_rows, src.data())) {
            fprintf(stderr, "%s: read error %s\n", __func__, name.c_str());
            return false;
        }
        std::vector<double> part_d(n_threads, 0.0), part_s(n_threads, 0.0), part_u(n_threads, 0.0), part_e(n_threads, 0.0);
        std::vector<std::thread> workers;
        for (int th = 0; th < n_threads; ++th) {
            workers.emplace_back([&, th]() {
                std::vector<uint8_t> q(ggml_row_size(qt, k));
                std::vector<float> deq(k);
                for (int64_t r = th; r < n_rows; r += n_threads) {
                    const int64_t e = r / rpe;
                    const float * x = src.data() + r*k;
                    const float * act = imp.act.data() + e*k;
                    if (noise) {
                        double e2 = 0.0;
                        for (int64_t j = 0; j < k; ++j) e2 += (double) x[j] * x[j];
                        const float sigma = (float) std::sqrt(noise_r * e2 / k);
                        std::mt19937_64 rng(qd_hash((uint64_t) r, name + qtype));
                        std::normal_distribution<float> nd(0.0f, sigma);
                        for (int64_t j = 0; j < k; ++j) deq[j] = x[j] + nd(rng);
                    } else if (control || qt == GGML_TYPE_F32) {
                        memcpy(deq.data(), x, k*sizeof(float));
                    } else {
                        ggml_quantize_chunk(qt, x, q.data(), 0, 1, k, imp.has_imatrix ? imp.quant_w.data() + e*k : nullptr);
                        ggml_get_type_traits(qt)->to_float(q.data(), deq.data(), k);
                    }
                    uint8_t * dst = pert.data() + r * ggml_row_size(mt->type, k);
                    if (mt->type == GGML_TYPE_F32) {
                        memcpy(dst, deq.data(), k*sizeof(float));
                    } else if (mt->type == GGML_TYPE_F16) {
                        ggml_fp32_to_fp16_row(deq.data(), (ggml_fp16_t *) dst, k);
                        ggml_fp16_to_fp32_row((const ggml_fp16_t *) dst, deq.data(), k);
                    } else {
                        ggml_fp32_to_bf16_row_ref(deq.data(), (ggml_bf16_t *) dst, k);
                        ggml_bf16_to_fp32_row((const ggml_bf16_t *) dst, deq.data(), k);
                    }
                    // the damage of the values the model actually receives, after rounding to its tensor type
                    double d = 0, s = 0, u = 0, en = 0;
                    for (int64_t j = 0; j < k; ++j) {
                        const double err = (double) x[j] - deq[j];
                        d  += act[j] * err * err;
                        s  += act[j] * (double) x[j] * x[j];
                        u  += err * err;
                        en += (double) x[j] * x[j];
                    }
                    const double w = imp.w[QD_WEIGHTING_MASS][e];
                    part_d[th] += w * d;
                    part_s[th] += w * s;
                    part_u[th] += u;
                    part_e[th] += en;
                }
            });
        }
        for (auto & w : workers) w.join();
        for (int th = 0; th < n_threads; ++th) { D += part_d[th]; S += part_s[th]; U += part_u[th]; E += part_e[th]; }

        // the control re-uploads the untouched bytes; rebuilt from the source, they are the same as the original ones
        ggml_backend_tensor_set(mt, control && !restore_from_source ? a.backup.data() : pert.data(), 0, nbytes);
        undo.push_back(std::move(a));

        const double bpw = control || noise ? 8.0 * ggml_type_size(mt->type) / ggml_blck_size(mt->type) : 8.0 * ggml_row_size(qt, k) / k;
        bits += bpw * ti.n_elements();
        n_el += ti.n_elements();
        shape = std::to_string(ti.ne[0]) + "x" + std::to_string(ti.ne[1]) + "x" + std::to_string(ti.ne[2]);
        return true;
    };

    auto restore_source = [&](const std::string & name) {
        double d0 = 0, s0 = 0, u0 = 0, e0 = 0, b0 = 0, n0 = 0;
        std::string sh0;
        std::vector<applied> ignore;
        perturb(name, "none", ignore, d0, s0, u0, e0, b0, n0, sh0);
    };
    // tensors of the previous job that are not restored yet: with restore_from_source, a tensor that the next job
    // overwrites is not restored first, which saves one upload per tensor between recipes over the same tensors
    std::vector<applied> pending;
    auto restore_pending = [&](const std::vector<std::pair<std::string, std::string>> & keep) {
        for (auto & a : pending) {
            const bool kept = std::any_of(keep.begin(), keep.end(), [&](const auto & p) { return p.first == a.name; });
            if (!kept) {
                restore_source(a.name);
            }
        }
        pending.clear();
    };

    for (const auto & jb : mine) {
        const auto t0 = std::chrono::steady_clock::now();
        restore_pending(jb.parts);
        std::vector<applied> undo;
        double D = 0, S = 0, U = 0, E = 0, bits = 0, n_el = 0, R_sum = 0;
        std::string shape;
        bool all = true;
        for (const auto & [name, qtype] : jb.parts) {
            double d = 0, s_ = 0;
            if (!perturb(name, qtype, undo, d, s_, U, E, bits, n_el, shape)) {
                all = false;
                break;
            }
            D += d;
            S += s_;
            R_sum += s_ > 0 ? d / s_ : 0.0;
        }
        auto restore = [&](bool lazy) {
            if (restore_from_source && lazy) {
                pending = std::move(undo);
                return;
            }
            for (auto & a : undo) {
                if (restore_from_source) {
                    restore_source(a.name);
                } else {
                    ggml_backend_tensor_set(a.mt, a.backup.data(), 0, a.backup.size());
                }
            }
        };
        if (!all || undo.empty()) {
            // tensors kept from the previous job may not have been reached
            restore(false);
            if (restore_from_source) {
                for (const auto & p : jb.parts) {
                    restore_source(p.first);
                }
            }
            continue;
        }

        std::vector<double> kl(n_scored, 0.0), nlls(n_scored, 0.0);
        std::vector<char> hits(n_scored, 0);
        const bool ok = run([&](int i, const std::vector<float> & lp, llama_token tgt) {
            const float * rl = ref.data() + (size_t) i * n_vocab;
            double v = 0.0;
            for (int j = 0; j < n_vocab; ++j) {
                v += std::exp((double) rl[j]) * ((double) rl[j] - lp[j]);
            }
            kl[i] = std::max(0.0, v);
            nlls[i] = -lp[tgt];
            hits[i] = (std::max_element(lp.begin(), lp.end()) - lp.begin()) == ref_top[i];
        });
        double nll = 0.0;
        int top1 = 0;
        for (int i = 0; i < n_scored; ++i) { nll += nlls[i]; top1 += hits[i]; }
        restore(true);
        if (!ok) {
            fprintf(stderr, "error: decode failed\n");
            return 1;
        }

        double kl_mean = 0.0, kl_max = 0.0;
        for (double v : kl) { kl_mean += v; kl_max = std::max(kl_max, v); }
        kl_mean /= n_scored;
        std::vector<double> sorted = kl;
        std::sort(sorted.begin(), sorted.end());
        const double kl_p99 = sorted[std::min<size_t>(sorted.size() - 1, (size_t) (0.99 * sorted.size()))];
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const bool single = jb.parts.size() == 1;
        // a recipe reports the sum of D and the sum of the per-tensor R
        const double R_out = single ? (S > 0 ? D / S : 0.0) : R_sum;

        fprintf(out, "%s,%s,%s,%.4f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%d,%.2f",
                jb.label.c_str(), jb.qtype.c_str(), single ? shape.c_str() : std::to_string(jb.parts.size()).c_str(), bits / n_el, D, R_out, E > 0 ? U / E : 0.0,
                kl_mean, kl_p99, kl_max, std::exp(nll / n_scored), ref_ppl, (double) top1 / n_scored, n_scored, secs);
        if (router) {
            std::vector<double> re, rf;
            router_diff(re, rf);
            double e_sum = 0.0, f_mean = 0.0;
            for (double v : re) e_sum += v;
            for (double v : rf) f_mean += v / rf.size();
            fprintf(out, ",%.6e,%.6e,%s,%s", e_sum, f_mean, join(re).c_str(), join(rf).c_str());
        }
        fprintf(out, "\n");
        fflush(out);
        fprintf(stderr, "%s: %-32s %-7s R %.3e  KLD %.3e  PPL %.4f (ref %.4f)  %.1f s\n", __func__, jb.label.c_str(), jb.qtype.c_str(),
                R_out, kl_mean, std::exp(nll / n_scored), ref_ppl, secs);
    }
    restore_pending({});
    fclose(out);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
