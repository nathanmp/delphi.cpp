#pragma once

// KL divergence of a live context against a base file written by `llama-perplexity --kl-divergence-base`.
// The base file carries the evaluation tokens, so no text is needed.

#include <cstddef>
#include <string>

struct llama_context;

struct llama_kld_stats {
    bool        ok = false;
    std::string error;
    int         n_chunk = 0;
    int         n_ctx   = 0;
    size_t      count   = 0;           // evaluated tokens
    double      kld = 0, kld_se = 0;   // mean KL divergence
    double      ppl = 0, ppl_se = 0;   // PPL of the context
    double      ppl_base = 0;          // PPL of the base
    double      log_ppl_ratio = 0, log_ppl_ratio_se = 0;
    double      same_top = 0, same_top_se = 0; // fraction, 0..1
    double      p_diff_rms = 0;        // fraction, 0..1
};

// n_batch: tokens per llama_decode call (small batches run the expert cache path as in decode).
// Clears the memory of the context. Returns stats.ok.
bool llama_kld_compute(llama_context * ctx, const std::string & base_file, int n_batch, llama_kld_stats & stats);
