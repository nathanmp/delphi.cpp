// delphi.cpp: the logits of one model over a mixed sequence of prefill and decode steps in one process, for A/B checks of backends
// that change how weights are read between prefill and decode (GGML_CUDA_MOE_PACKED_INPLACE). The tokens come from a text file
// (teacher forcing), so two runs see the same tokens at the same positions whatever their logits are.
//
// A plan is a list of requests separated by '|'. Each request starts with an empty KV cache and runs its steps P:T in order: a
// prefill of P tokens in one llama_decode (split into ubatches by -ub), then T decode steps of one token each. The logits of the
// last --keep-pp tokens of each prefill and of every decode step are written to a file (--save) or compared with a file (--compare):
// per position the KL divergence from the file, the top-1 token, the NLL of the next text token, and bitwise identity.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct mixed_step {
    int n_pp;
    int n_tg;
};

struct record_head {
    int32_t phase;   // 0: prefill, 1: decode
    int32_t request;
    int32_t pos;
    int32_t next;    // the next text token
};

static const char k_magic[8] = { 'D', 'M', 'X', 'L', 'O', 'G', '0', '1' };

static const char * k_default_plan = "512:32,37:16,2048:32,5:8,7:8|1:48|300:24,128:24,1:8,64:16";

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t b = 0;
    while (true) {
        const size_t e = s.find(sep, b);
        out.push_back(s.substr(b, e == std::string::npos ? std::string::npos : e - b));
        if (e == std::string::npos) {
            return out;
        }
        b = e + 1;
    }
}

static bool parse_plan(const std::string & plan, std::vector<std::vector<mixed_step>> & out) {
    for (const std::string & req : split(plan, '|')) {
        std::vector<mixed_step> steps;
        for (const std::string & st : split(req, ',')) {
            const std::vector<std::string> pt = split(st, ':');
            if (pt.size() != 2) {
                return false;
            }
            const mixed_step s = { atoi(pt[0].c_str()), atoi(pt[1].c_str()) };
            if (s.n_pp < 1 || s.n_tg < 0) {
                return false;
            }
            steps.push_back(s);
        }
        out.push_back(steps);
    }
    return !out.empty();
}

static bool g_verbose = false;

// Only warnings, errors and the packed-weight lines of the CUDA backend unless -v.
static void log_filter(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (g_verbose || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || strstr(text, "moe_packed") != nullptr) {
        fputs(text, stderr);
    }
}

struct phase_stats {
    int    n      = 0;
    int    n_bit  = 0;
    int    n_top  = 0;
    double kld    = 0.0;
    double kld_max = 0.0;
    double nll_base = 0.0;
    double nll_cur  = 0.0;

    void print(const char * name) const {
        if (n == 0) {
            return;
        }
        printf("| %s | %d | %d | %.6f | %.6f | %.2f%% | %.6f |\n", name, n, n_bit, kld/n, kld_max, 100.0*n_top/n, exp((nll_cur - nll_base)/n));
    }
};

static double log_sum_exp(const float * x, int n, int & argmax) {
    float mx = x[0];
    argmax = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > mx) {
            mx = x[i];
            argmax = i;
        }
    }
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += exp((double) x[i] - mx);
    }
    return mx + log(s);
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf -f text [--save FILE] [--compare FILE] [-ngl N] [-t N] [-fa on|off|auto] [-ub N] [--plan PLAN]\n"
        "       [--keep-pp N] [--expect-bitwise R,...] [-v]\n"
        "  --plan            requests separated by '|', each a list of P:T steps (default: %s)\n"
        "  --keep-pp         stored logits per prefill, its last tokens (default: 8)\n"
        "  --expect-bitwise  exit 1 unless every stored position of these requests is bitwise identical to --compare\n",
        argv0, k_default_plan);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string text_path;
    std::string save_path;
    std::string compare_path;
    std::string plan_str = k_default_plan;
    std::string fa       = "auto";
    std::vector<int> expect_bitwise;
    int ngl     = 999;
    int threads = 8;
    int n_ub    = 512;
    int keep_pp = 8;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0]);
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "-m")               { model_path   = next(); }
        else if (a == "-f")               { text_path    = next(); }
        else if (a == "--save")           { save_path    = next(); }
        else if (a == "--compare")        { compare_path = next(); }
        else if (a == "--plan")           { plan_str     = next(); }
        else if (a == "-fa")              { fa           = next(); }
        else if (a == "-ngl")             { ngl     = atoi(next().c_str()); }
        else if (a == "-t")               { threads = atoi(next().c_str()); }
        else if (a == "-ub")              { n_ub    = atoi(next().c_str()); }
        else if (a == "--keep-pp")        { keep_pp = atoi(next().c_str()); }
        else if (a == "--expect-bitwise") {
            for (const std::string & r : split(next(), ',')) {
                expect_bitwise.push_back(atoi(r.c_str()));
            }
        }
        else if (a == "-v")               { g_verbose = true; }
        else                              { usage(argv[0]); return 1; }
    }
    std::vector<std::vector<mixed_step>> plan;
    if (model_path.empty() || text_path.empty() || (save_path.empty() && compare_path.empty()) || !parse_plan(plan_str, plan) ||
            keep_pp < 1 || n_ub < 1) {
        usage(argv[0]);
        return 1;
    }

    llama_log_set(log_filter, nullptr);
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "error: cannot load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // enough text for any plan: about 4 bytes per token
    std::string text;
    {
        std::ifstream f(text_path, std::ios::binary);
        text.resize(1 << 20);
        f.read(&text[0], text.size());
        text.resize(f.gcount());
    }
    std::vector<llama_token> toks(text.size() + 16);
    const int n_toks = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (n_toks < 0) {
        fprintf(stderr, "error: tokenization failed\n");
        return 1;
    }
    toks.resize(n_toks);

    int64_t n_needed = 1;
    int max_pp  = 1;
    int max_req = 1;
    for (const auto & req : plan) {
        int n_req = 0;
        for (const mixed_step & s : req) {
            n_req += s.n_pp + s.n_tg;
            max_pp = std::max(max_pp, s.n_pp);
        }
        n_needed += n_req;
        max_req = std::max(max_req, n_req);
    }
    if (n_needed > (int64_t) toks.size()) {
        fprintf(stderr, "error: the plan needs %" PRId64 " tokens, the text has %zu\n", n_needed, toks.size());
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (max_req + 255)/256*256;
    cparams.n_batch         = std::max(max_pp, n_ub);
    cparams.n_ubatch        = n_ub;
    cparams.n_threads       = threads;
    cparams.n_threads_batch = threads;
    cparams.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: cannot create the context\n");
        return 1;
    }

    std::vector<record_head> heads;
    std::vector<float>       logits;
    auto store = [&](int phase, int request, int pos, int next, const float * l) {
        heads.push_back({ phase, request, pos, next });
        logits.insert(logits.end(), l, l + n_vocab);
    };

    llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
    size_t off = 0;
    printf("plan %s, ubatch %d, n_ctx %u\n\n| request | prefill tokens | prefill s | decode tokens | decode s | decode tok/s |\n"
        "| --- | --- | --- | --- | --- | --- |\n", plan_str.c_str(), n_ub, cparams.n_ctx);
    for (size_t r = 0; r < plan.size(); ++r) {
        llama_memory_clear(llama_get_memory(ctx), true);
        int    pos = 0;
        int    n_pp = 0, n_tg = 0;
        double t_pp = 0.0, t_tg = 0.0;
        for (const mixed_step & s : plan[r]) {
            batch.n_tokens = s.n_pp;
            for (int i = 0; i < s.n_pp; ++i) {
                batch.token[i]     = toks[off + i];
                batch.pos[i]       = pos + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = i >= s.n_pp - keep_pp;
            }
            auto t0 = std::chrono::steady_clock::now();
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "error: prefill of %d tokens failed\n", s.n_pp);
                return 1;
            }
            for (int i = std::max(0, s.n_pp - keep_pp); i < s.n_pp; ++i) {
                store(0, (int) r, pos + i, toks[off + i + 1], llama_get_logits_ith(ctx, i));
            }
            t_pp += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            off  += s.n_pp;
            pos  += s.n_pp;
            n_pp += s.n_pp;

            t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < s.n_tg; ++k) {
                batch.n_tokens     = 1;
                batch.token[0]     = toks[off];
                batch.pos[0]       = pos;
                batch.n_seq_id[0]  = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0]    = 1;
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "error: decode step failed\n");
                    return 1;
                }
                store(1, (int) r, pos, toks[off + 1], llama_get_logits_ith(ctx, 0));
                ++off;
                ++pos;
            }
            t_tg += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            n_tg += s.n_tg;
        }
        printf("| %zu | %d | %.3f | %d | %.3f | %.2f |\n", r, n_pp, t_pp, n_tg, t_tg, n_tg > 0 ? n_tg/t_tg : 0.0);
    }
    llama_batch_free(batch);
    const int32_t n_rec = (int32_t) heads.size();

    int ret = 0;
    if (!save_path.empty()) {
        FILE * f = fopen(save_path.c_str(), "wb");
        const int32_t plan_len = (int32_t) plan_str.size();
        bool ok = f != nullptr;
        ok = ok && fwrite(k_magic, sizeof(k_magic), 1, f) == 1 && fwrite(&n_vocab, sizeof(n_vocab), 1, f) == 1 &&
            fwrite(&n_rec, sizeof(n_rec), 1, f) == 1 && fwrite(&plan_len, sizeof(plan_len), 1, f) == 1 &&
            fwrite(plan_str.data(), 1, plan_len, f) == (size_t) plan_len && fwrite(heads.data(), sizeof(record_head), n_rec, f) == (size_t) n_rec &&
            fwrite(logits.data(), sizeof(float), logits.size(), f) == logits.size();
        if (f) {
            ok = fclose(f) == 0 && ok;
        }
        if (!ok) {
            fprintf(stderr, "error: cannot write %s\n", save_path.c_str());
            return 1;
        }
        printf("\nsaved %d positions to %s\n", n_rec, save_path.c_str());
    }
    if (!compare_path.empty()) {
        FILE * f = fopen(compare_path.c_str(), "rb");
        char    magic[8];
        int32_t b_vocab = 0, b_rec = 0, plan_len = 0;
        bool ok = f != nullptr && fread(magic, sizeof(magic), 1, f) == 1 && memcmp(magic, k_magic, sizeof(magic)) == 0 &&
            fread(&b_vocab, sizeof(b_vocab), 1, f) == 1 && fread(&b_rec, sizeof(b_rec), 1, f) == 1 &&
            fread(&plan_len, sizeof(plan_len), 1, f) == 1 && b_vocab == n_vocab && b_rec == n_rec;
        std::string b_plan(ok ? plan_len : 0, ' ');
        std::vector<record_head> b_heads(ok ? n_rec : 0);
        std::vector<float>       b_logits(ok ? logits.size() : 0);
        ok = ok && fread(&b_plan[0], 1, plan_len, f) == (size_t) plan_len && b_plan == plan_str &&
            fread(b_heads.data(), sizeof(record_head), n_rec, f) == (size_t) n_rec &&
            fread(b_logits.data(), sizeof(float), b_logits.size(), f) == b_logits.size() &&
            memcmp(b_heads.data(), heads.data(), n_rec*sizeof(record_head)) == 0;
        if (f) {
            fclose(f);
        }
        if (!ok) {
            fprintf(stderr, "error: %s is not a run of the same model, plan and --keep-pp\n", compare_path.c_str());
            return 1;
        }

        std::vector<phase_stats> st(plan.size()*2);
        phase_stats total[2];
        for (int i = 0; i < n_rec; ++i) {
            const float * a = b_logits.data() + (size_t) i*n_vocab;
            const float * b = logits.data()   + (size_t) i*n_vocab;
            int am, bm;
            const double la = log_sum_exp(a, n_vocab, am);
            const double lb = log_sum_exp(b, n_vocab, bm);
            double kld = 0.0;
            for (int j = 0; j < n_vocab; ++j) {
                const double pa = (double) a[j] - la;
                kld += exp(pa)*(pa - ((double) b[j] - lb));
            }
            const record_head & h = heads[i];
            for (phase_stats * s : { &st[h.request*2 + h.phase], &total[h.phase] }) {
                s->n        += 1;
                s->n_bit    += memcmp(a, b, n_vocab*sizeof(float)) == 0;
                s->n_top    += am == bm;
                s->kld      += kld;
                s->kld_max   = std::max(s->kld_max, kld);
                s->nll_base += la - a[h.next];
                s->nll_cur  += lb - b[h.next];
            }
        }
        printf("\nagainst %s:\n\n| request, phase | positions | bitwise | mean KLD | max KLD | same top-1 | PPL ratio |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n", compare_path.c_str());
        for (size_t r = 0; r < plan.size(); ++r) {
            for (int p = 0; p < 2; ++p) {
                const std::string name = std::to_string(r) + (p == 0 ? ", prefill" : ", decode");
                st[r*2 + p].print(name.c_str());
            }
        }
        total[0].print("all, prefill");
        total[1].print("all, decode");
        for (int r : expect_bitwise) {
            if (r < 0 || r >= (int) plan.size()) {
                fprintf(stderr, "error: no request %d\n", r);
                ret = 1;
                continue;
            }
            const int n   = st[r*2].n + st[r*2 + 1].n;
            const int bit = st[r*2].n_bit + st[r*2 + 1].n_bit;
            printf("request %d: %d of %d positions bitwise identical%s\n", r, bit, n, bit == n ? "" : ", FAILED");
            ret |= bit != n;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    return ret;
}
