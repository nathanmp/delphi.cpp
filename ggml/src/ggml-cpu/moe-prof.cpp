#include "moe-prof.h"

#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

// 16 buckets covering 1..32k routed rows per expert, doubling each step
constexpr int k_row_buckets = 16;

struct moe_prof_counters {
    int64_t us[GGML_MOE_PROF_COUNT] = {};
    int64_t calls[GGML_MOE_PROF_COUNT] = {};
    int64_t rows_hist[k_row_buckets] = {};
    int64_t rows_sum[k_row_buckets] = {};
    int64_t rows_total = 0;
    int64_t rows_samples = 0;
};

struct moe_prof_state {
    std::mutex mutex;
    std::vector<moe_prof_counters *> threads;

    ~moe_prof_state() { report(); }

    void report() {
        std::lock_guard<std::mutex> lock(mutex);
        if (threads.empty()) {
            return;
        }

        moe_prof_counters total;
        for (const auto * t : threads) {
            for (int p = 0; p < GGML_MOE_PROF_COUNT; ++p) {
                total.us[p]    += t->us[p];
                total.calls[p] += t->calls[p];
            }
            for (int b = 0; b < k_row_buckets; ++b) {
                total.rows_hist[b] += t->rows_hist[b];
                total.rows_sum[b]  += t->rows_sum[b];
            }
            total.rows_total   += t->rows_total;
            total.rows_samples += t->rows_samples;
        }

        static const char * const names[GGML_MOE_PROF_COUNT] = {
            "quant_src1", "row_map", "barrier", "repack", "gemm", "mmid_total",
        };

        const int64_t total_us = total.us[GGML_MOE_PROF_TOTAL];
        if (total_us == 0) {
            return;
        }

        int64_t accounted_us = 0;
        for (int p = 0; p < GGML_MOE_PROF_TOTAL; ++p) {
            accounted_us += total.us[p];
        }

        fprintf(stderr, "\nmoe phase profile: %zu worker threads, %.3f s of thread time in MUL_MAT_ID\n",
                threads.size(), total_us / 1e6);
        fprintf(stderr, "  %-11s %12s %8s %14s %12s\n", "phase", "thread_s", "share", "calls", "us/call");
        for (int p = 0; p < GGML_MOE_PROF_COUNT; ++p) {
            fprintf(stderr, "  %-11s %12.3f %7.2f%% %14lld %12.2f\n",
                    names[p], total.us[p] / 1e6, 100.0 * (double) total.us[p] / (double) total_us,
                    (long long) total.calls[p],
                    total.calls[p] ? (double) total.us[p] / (double) total.calls[p] : 0.0);
        }
        fprintf(stderr, "  %-11s %12.3f %7.2f%%\n", "other",
                (total_us - accounted_us) / 1e6,
                100.0 * (double) (total_us - accounted_us) / (double) total_us);

        // The expert loop ends on the graph-wide barrier outside this op, so its
        // imbalance shows up as spread between threads rather than as barrier time.
        int64_t min_us = total.us[GGML_MOE_PROF_TOTAL];
        int64_t max_us = 0;
        for (const auto * t : threads) {
            const int64_t v = t->us[GGML_MOE_PROF_TOTAL];
            min_us = v < min_us ? v : min_us;
            max_us = v > max_us ? v : max_us;
        }
        const double mean_us = (double) total_us / (double) threads.size();
        fprintf(stderr, "  per thread mmid_total: min %.3f s, mean %.3f s, max %.3f s (max/mean %.3f)\n",
                min_us / 1e6, mean_us / 1e6, max_us / 1e6, mean_us > 0 ? max_us / mean_us : 0.0);

        if (total.rows_samples > 0) {
            fprintf(stderr, "  routed rows per active expert: mean %.1f over %lld samples\n",
                    (double) total.rows_total / (double) total.rows_samples,
                    (long long) total.rows_samples);
            fprintf(stderr, "  histogram:");
            for (int b = 0; b < k_row_buckets; ++b) {
                if (total.rows_hist[b] > 0) {
                    fprintf(stderr, " [%d,%d)=%.1f%%", 1 << b, 1 << (b + 1),
                            100.0 * (double) total.rows_hist[b] / (double) total.rows_samples);
                }
            }
            fprintf(stderr, "\n");
            // Tagged block for tools/moe-schedule-stats/moe-rows-hist.py. One line
            // per non-empty power-of-two bucket: lo is inclusive, hi is exclusive.
            // The rows share is the script's MMID-time proxy: expert GEMM FLOPs
            // scale with rows, while fixed per-expert overhead is not attributed.
            fprintf(stderr, "moe_prof_begin samples=%lld rows=%lld mmid_thread_s=%.3f\n",
                    (long long) total.rows_samples, (long long) total.rows_total,
                    total_us / 1e6);
            for (int b = 0; b < k_row_buckets; ++b) {
                if (total.rows_hist[b] > 0) {
                    fprintf(stderr, "moe_prof_bucket lo=%d hi=%d count=%lld rows=%lld\n",
                            1 << b, 1 << (b + 1),
                            (long long) total.rows_hist[b], (long long) total.rows_sum[b]);
                }
            }
            fprintf(stderr, "moe_prof_end\n");
        }
    }
};

moe_prof_state & state() {
    static moe_prof_state s;
    return s;
}

moe_prof_counters * thread_counters() {
    thread_local moe_prof_counters * counters = [] {
        auto * c = new moe_prof_counters();
        auto & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.threads.push_back(c);
        return c;
    }();
    return counters;
}

} // namespace

bool ggml_moe_prof_enabled(void) {
    static const bool enabled = [] {
        const char * value = getenv("GGML_MOE_PROF");
        return value && value[0] != '\0' && strcmp(value, "0") != 0;
    }();
    return enabled;
}

void ggml_moe_prof_add(enum ggml_moe_prof_phase phase, int64_t us) {
    auto * c = thread_counters();
    c->us[phase]    += us;
    c->calls[phase] += 1;
}

void ggml_moe_prof_add_expert_rows(int64_t rows) {
    auto * c = thread_counters();
    int bucket = 0;
    while (bucket + 1 < k_row_buckets && (rows >> (bucket + 1)) != 0) {
        ++bucket;
    }
    c->rows_hist[bucket] += 1;
    c->rows_sum[bucket]  += rows;
    c->rows_total        += rows;
    c->rows_samples      += 1;
}
