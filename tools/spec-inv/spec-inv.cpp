// Checks that a token's logits, graph tensors and committed sequence state do not depend on how many
// tokens are decoded beside it (the verify widths used by speculative decoding).
#include "llama.h"
#include "ggml-backend.h"
#include "../../src/llama-ext.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct tensor_rec {
    std::string          key;
    ggml_type            type;
    std::vector<uint8_t> data;
};

// per tail token: tensors in graph order
struct trace_state {
    bool active = false;
    int  base   = 0; // tail index of the first token in the current decode
    int  ntok   = 0;
    std::map<std::string, int> occ;
    std::vector<std::vector<tensor_rec>> tok;
};

static bool gather(const ggml_tensor * t, std::vector<uint8_t> & out) {
    if (ggml_blck_size(t->type) != 1 || t->nb[0] != ggml_type_size(t->type)) {
        return false;
    }
    const size_t row = t->ne[0] * t->nb[0];
    out.resize(row * ggml_nrows(t));
    size_t o = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
        ggml_backend_tensor_get(t, out.data() + o, i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], row);
        o += row;
    }
    return true;
}

static bool trace_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * s = (trace_state *) ud;
    if (ask) {
        return s->active;
    }
    std::vector<uint8_t> all;
    if (!gather(t, all)) {
        return true;
    }
    const std::string key = std::string(t->name) + "#" + std::to_string(s->occ[t->name]++) + " " + ggml_op_desc(t);
    if (s->ntok == 1) {
        s->tok[s->base].push_back({ key, t->type, std::move(all) });
        return true;
    }
    int d = 3;
    while (d > 0 && t->ne[d] == 1) {
        d--;
    }
    if (t->ne[d] != s->ntok) {
        return true;
    }
    const size_t per = all.size() / s->ntok;
    for (int i = 0; i < s->ntok; i++) {
        s->tok[s->base + i].push_back({ key, t->type, std::vector<uint8_t>(all.begin() + i*per, all.begin() + (i+1)*per) });
    }
    return true;
}

struct run_result {
    std::vector<std::vector<float>> logits; // per tail token
    std::vector<uint8_t>             state;
    trace_state                      trace;
};

static const char * k_text =
    "The committee met on a grey Tuesday morning to review the proposal for the new railway line. "
    "Several members argued that the projected costs were optimistic, citing the tunnel under the river "
    "and the price of steel, which had risen sharply over the previous two years. Others pointed out that "
    "the existing road network was congested and that freight traffic would double within a decade. "
    "After three hours of debate, the chair proposed a compromise: a phased build, starting with the "
    "section between the port and the industrial district, with a review of costs before the second phase. "
    "The vote was close. In the weeks that followed, engineers surveyed the route, farmers objected to the "
    "loss of pasture, and a local historian discovered that the planned depot sat on the site of a medieval "
    "market. The plans were redrawn twice before the first contract was signed.";

int main(int argc, char ** argv) {
    std::string model_path, dev = "none", widths_s = "1,2,3,4,5,6,7,8";
    std::vector<std::string> ot; // tensor-name regexes kept on the CPU
    int  n_prefix = 64, n_tail = 12, n_threads = 32;
    bool trace = false, fa_off = false, nextn = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-dev" && i + 1 < argc) dev = argv[++i];
        else if (a == "-w" && i + 1 < argc) widths_s = argv[++i];
        else if (a == "-np" && i + 1 < argc) n_prefix = atoi(argv[++i]);
        else if (a == "-nt" && i + 1 < argc) n_tail = atoi(argv[++i]);
        else if (a == "--trace") trace = true;
        else if (a == "--fa-off") fa_off = true;
        else if (a == "--nextn") nextn = true; // target graph as in draft-mtp
        else if (a == "-ot" && i + 1 < argc) ot.push_back(argv[++i]);
        else if (a == "-t" && i + 1 < argc) n_threads = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s -m model [-dev ROCm1,ROCm2|none] [-ot regex]... [-t 32] [-w 1,2,...,8] [-np 64] [-nt 12] [--trace] [--fa-off] [--nextn]\n", argv[0]); return 1; }
    }
    std::vector<int> widths;
    for (size_t p = 0; p < widths_s.size();) {
        widths.push_back(atoi(widths_s.c_str() + p));
        p = widths_s.find(',', p);
        p = p == std::string::npos ? widths_s.size() : p + 1;
    }

    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs;
    if (dev != "none") {
        for (size_t p = 0; p < dev.size();) {
            size_t e = dev.find(',', p);
            e = e == std::string::npos ? dev.size() : e;
            const std::string name = dev.substr(p, e - p);
            devs.push_back(ggml_backend_dev_by_name(name.c_str()));
            if (!devs.back()) { fprintf(stderr, "no device %s\n", name.c_str()); return 1; }
            p = e + 1;
        }
        mp.n_gpu_layers = 999;
    } else {
        mp.n_gpu_layers = 0;
    }
    devs.push_back(nullptr);
    mp.devices = devs.data();
    std::vector<llama_model_tensor_buft_override> ovr;
    for (auto & o : ot) ovr.push_back({ o.c_str(), ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) });
    ovr.push_back({ nullptr, nullptr });
    if (!ot.empty()) mp.tensor_buft_overrides = ovr.data();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(n_prefix + n_tail + 4096);
    std::string text;
    while ((int) text.size() < 8 * (n_prefix + n_tail)) text += k_text;
    const int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false);
    if (n < n_prefix + n_tail) { fprintf(stderr, "only %d tokens\n", n); return 1; }

    std::vector<run_result> res(widths.size());
    for (size_t wi = 0; wi < widths.size(); wi++) {
        const int w = widths[wi];
        run_result & r = res[wi];
        r.trace.tok.resize(n_tail);
        auto cp = llama_context_default_params();
        cp.n_ctx = std::max(1024, n_prefix + n_tail + 256); cp.n_batch = std::max(512, n_prefix); cp.n_ubatch = 512; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = n_threads;
        if (fa_off) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if (trace) { cp.cb_eval = trace_cb; cp.cb_eval_user_data = &r.trace; }
        llama_context * ctx = llama_init_from_model(model, cp);
        if (ctx && nextn) llama_set_embeddings_nextn(ctx, true, false);
        if (!ctx) { return 1; }

        llama_batch b = llama_batch_init(std::max(512, n_prefix), 0, 1);
        auto add = [&](int pos, bool logits) {
            b.token[b.n_tokens] = toks[pos]; b.pos[b.n_tokens] = pos; b.n_seq_id[b.n_tokens] = 1;
            b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = logits; b.n_tokens++;
        };
        b.n_tokens = 0;
        for (int i = 0; i < n_prefix; i++) add(i, i == n_prefix - 1);
        if (llama_decode(ctx, b)) { fprintf(stderr, "prefill failed\n"); return 1; }

        r.trace.active = trace;
        for (int t0 = 0; t0 < n_tail; t0 += w) {
            const int c = std::min(w, n_tail - t0);
            b.n_tokens = 0;
            for (int i = 0; i < c; i++) add(n_prefix + t0 + i, true);
            r.trace.base = t0; r.trace.ntok = c; r.trace.occ.clear();
            if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed (w=%d)\n", w); return 1; }
            for (int i = 0; i < c; i++) {
                const float * l = llama_get_logits_ith(ctx, i);
                r.logits.emplace_back(l, l + n_vocab);
            }
        }
        r.trace.active = false;
        r.state.resize(llama_state_seq_get_size(ctx, 0));
        llama_state_seq_get_data(ctx, r.state.data(), r.state.size(), 0);
        llama_batch_free(b);
        llama_free(ctx);
    }

    // compare every width against the first
    const run_result & ref = res[0];
    bool all_ok = true;
    for (size_t wi = 1; wi < widths.size(); wi++) {
        const run_result & r = res[wi];
        int n_exact = 0; double worst = 0;
        for (int t = 0; t < n_tail; t++) {
            if (memcmp(ref.logits[t].data(), r.logits[t].data(), n_vocab * sizeof(float)) == 0) { n_exact++; continue; }
            for (int v = 0; v < n_vocab; v++) worst = std::max(worst, (double) std::fabs(ref.logits[t][v] - r.logits[t][v]));
        }
        size_t nd = 0, first = SIZE_MAX;
        const bool same_size = ref.state.size() == r.state.size();
        if (same_size) {
            for (size_t i = 0; i < r.state.size(); i++) if (ref.state[i] != r.state[i]) { nd++; if (first == SIZE_MAX) first = i; }
        }
        const bool ok = n_exact == n_tail && same_size && nd == 0;
        all_ok &= ok;
        printf("w=%d vs w=%d: logits exact %d/%d (max |d| %.3g), state %s", widths[wi], widths[0], n_exact, n_tail, worst,
               !same_size ? "SIZE DIFFERS" : nd == 0 ? "identical" : "differs");
        if (same_size && nd) printf(" (%zu of %zu bytes, first at %zu)", nd, r.state.size(), first);
        printf("  %s\n", ok ? "PASS" : "FAIL");

        if (!trace) continue;
        for (int t = 0; t < n_tail; t++) {
            std::map<std::string, const tensor_rec *> idx;
            for (auto & x : r.trace.tok[t]) idx[x.key] = &x;
            int n_cmp = 0, n_i32_diff = 0; const tensor_rec * fd = nullptr; double fd_rel = 0; std::string first_i32;
            for (auto & x : ref.trace.tok[t]) {
                auto it = idx.find(x.key);
                if (it == idx.end() || it->second->data.size() != x.data.size() || it->second->type != x.type) continue;
                n_cmp++;
                if (memcmp(x.data.data(), it->second->data.data(), x.data.size()) == 0) continue;
                if (x.type == GGML_TYPE_I32) { if (!n_i32_diff++) first_i32 = x.key; continue; }
                if (!fd) {
                    fd = &x;
                    if (x.type == GGML_TYPE_F32) {
                        const float * a = (const float *) x.data.data(), * bb = (const float *) it->second->data.data();
                        double num = 0, den = 0;
                        for (size_t i = 0; i < x.data.size() / 4; i++) { num += (a[i]-bb[i])*(a[i]-bb[i]); den += a[i]*a[i]; }
                        fd_rel = std::sqrt(num / (den + 1e-30));
                    }
                }
            }
            if (!fd && !n_i32_diff) continue;
            printf("  tail %2d: %d tensors compared, first diff %s", t, n_cmp, fd ? fd->key.c_str() : "-");
            if (fd && fd->type == GGML_TYPE_F32) printf(" (rel %.2e)", fd_rel);
            if (n_i32_diff) printf(", %d I32 tensors differ (first %s)", n_i32_diff, first_i32.c_str());
            printf("\n");
        }
    }
    printf("%s\n", all_ok ? "ALL PASS" : "SOME FAIL");
    llama_model_free(model);
    return all_ok ? 0 : 2;
}
