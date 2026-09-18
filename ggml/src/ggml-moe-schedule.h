#pragma once

// delphi.cpp: grouped routed-MoE schedule. Backend-neutral contract, host reference builder, checker and statistics.
//
// Input is the ids operand of MUL_MAT_ID: top_k int32 expert ids per token, ids[t*ids_s1 + k] for t < n_tokens, k < top_k.
// An assignment is one (token, top-k position) pair, numbered a = t*top_k + k. It is also the row that the pair owns in the
// [n, top_k, n_tokens] MUL_MAT_ID output and in the [1, top_k, n_tokens] routing weights.
//
// The schedule is three int32 arrays, n_assign = n_tokens*top_k:
//   offsets[n_experts + 1]: grouped rows [offsets[e], offsets[e + 1]) belong to expert e, offsets[0] = 0
//   slot[n_assign]:          grouped row -> assignment, ascending inside each expert (a stable sort by expert)
//   inverse[n_assign]:       assignment -> grouped row, -1 if the expert id of the assignment is out of range
//
// For grouped row r: token = slot[r] / top_k, top-k position = slot[r] % top_k, routing weight = weights[slot[r]].
// Duplicate ids inside one token are allowed: each assignment is its own row.
// Assignments with an out-of-range id are dropped, so then offsets[n_experts] < n_assign and the tail of slot is unused.
// The schedule depends on ids only, so one schedule serves gate, up and down of a layer.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline size_t ggml_moe_schedule_nbytes(int n_experts, int64_t n_assign) {
    return ((size_t) n_experts + 1 + 2*(size_t) n_assign)*sizeof(int32_t);
}

static inline int32_t ggml_moe_schedule_id(const int32_t * ids, int64_t ids_s1, int top_k, int64_t a) {
    return ids[(a / top_k)*ids_s1 + a % top_k];
}

// Counting sort over the assignments. O(n_assign + n_experts) time, no scratch memory.
static inline void ggml_moe_schedule_build_ref(
        const int32_t * ids, int64_t ids_s1, int64_t n_tokens, int top_k, int n_experts,
        int32_t * offsets, int32_t * slot, int32_t * inverse) {
    memset(offsets, 0, ((size_t) n_experts + 1)*sizeof(int32_t));
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            const int32_t e = ids[t*ids_s1 + k];
            if (e >= 0 && e < n_experts) {
                offsets[e + 1]++;
            }
        }
    }
    for (int e = 0; e < n_experts; ++e) {
        offsets[e + 1] += offsets[e];
    }

    // offsets[e] is the write cursor of expert e here; afterwards it holds the end of e, so shift by one
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            const int64_t a = t*top_k + k;
            const int32_t e = ids[t*ids_s1 + k];
            if (e < 0 || e >= n_experts) {
                inverse[a] = -1;
                continue;
            }
            const int32_t r = offsets[e]++;
            slot[r]    = (int32_t) a;
            inverse[a] = r;
        }
    }
    for (int e = n_experts; e > 0; --e) {
        offsets[e] = offsets[e - 1];
    }
    offsets[0] = 0;
}

// Checks a schedule against ids without the reference builder. Returns NULL if it satisfies the contract, the stable order
// included, else the first violated property.
static inline const char * ggml_moe_schedule_check(
        const int32_t * ids, int64_t ids_s1, int64_t n_tokens, int top_k, int n_experts,
        const int32_t * offsets, const int32_t * slot, const int32_t * inverse) {
    const int64_t n_assign = n_tokens*top_k;

    int64_t n_valid = 0;
    for (int64_t a = 0; a < n_assign; ++a) {
        const int32_t e = ggml_moe_schedule_id(ids, ids_s1, top_k, a);
        n_valid += e >= 0 && e < n_experts;
    }

    if (offsets[0] != 0) {
        return "offsets[0] != 0";
    }
    for (int e = 0; e < n_experts; ++e) {
        if (offsets[e + 1] < offsets[e]) {
            return "offsets not monotonic";
        }
    }
    if (offsets[n_experts] != n_valid) {
        return "offsets[n_experts] != number of valid assignments";
    }

    for (int e = 0; e < n_experts; ++e) {
        for (int32_t r = offsets[e]; r < offsets[e + 1]; ++r) {
            const int32_t a = slot[r];
            if (a < 0 || a >= n_assign) {
                return "slot out of range";
            }
            if (ggml_moe_schedule_id(ids, ids_s1, top_k, a) != e) {
                return "grouped row is not routed to the expert of its range";
            }
            if (r > offsets[e] && slot[r - 1] >= a) {
                return "rows of an expert are not in ascending assignment order";
            }
            if (inverse[a] != r) {
                return "inverse[slot[r]] != r";
            }
        }
    }

    // with inverse[slot[r]] == r for every row, this makes slot a bijection onto the valid assignments
    for (int64_t a = 0; a < n_assign; ++a) {
        const int32_t e = ggml_moe_schedule_id(ids, ids_s1, top_k, a);
        if (e < 0 || e >= n_experts) {
            if (inverse[a] != -1) {
                return "inverse of an invalid assignment is not -1";
            }
        } else if (inverse[a] < 0 || inverse[a] >= n_valid || slot[inverse[a]] != a) {
            return "slot[inverse[a]] != a";
        }
    }

    return NULL;
}

struct ggml_moe_schedule_stats {
    int64_t n_assign;        // grouped rows
    int32_t n_experts;
    int32_t n_active;        // experts with at least one row
    int32_t rows_min;        // rows per active expert
    int32_t rows_max;
    double  rows_mean;
    double  rows_p50;
    double  rows_p90;
    double  rows_p99;
    double  max_over_mean;   // routing skew: rows_max / rows_mean
    double  n_eff;           // n_assign^2 / sum(rows^2): the number of equally loaded experts with the same concentration
    size_t  nbytes;          // offsets + slot + inverse
};

static inline int ggml_moe_schedule_cmp_i32(const void * a, const void * b) {
    const int32_t x = *(const int32_t *) a;
    const int32_t y = *(const int32_t *) b;
    return (x > y) - (x < y);
}

// Percentiles interpolate linearly between the sorted rows of the active experts.
static inline double ggml_moe_schedule_quantile(const int32_t * sorted, int n, double q) {
    if (n <= 0) {
        return 0.0;
    }
    const double pos = q*(n - 1);
    const int    lo  = (int) pos;
    const int    hi  = lo + 1 < n ? lo + 1 : n - 1;
    return sorted[lo] + (pos - lo)*(sorted[hi] - sorted[lo]);
}

// scratch holds n_experts int32.
static inline void ggml_moe_schedule_get_stats(
        const int32_t * offsets, int n_experts, int64_t n_assign_total, int32_t * scratch, struct ggml_moe_schedule_stats * st) {
    memset(st, 0, sizeof(*st));
    st->n_experts = n_experts;
    st->n_assign  = offsets[n_experts];
    st->nbytes    = ggml_moe_schedule_nbytes(n_experts, n_assign_total);

    double sum_sq = 0.0;
    for (int e = 0; e < n_experts; ++e) {
        const int32_t rows = offsets[e + 1] - offsets[e];
        if (rows > 0) {
            scratch[st->n_active++] = rows;
            sum_sq += (double) rows*rows;
        }
    }
    if (st->n_active == 0) {
        return;
    }

    qsort(scratch, st->n_active, sizeof(int32_t), ggml_moe_schedule_cmp_i32);
    st->rows_min      = scratch[0];
    st->rows_max      = scratch[st->n_active - 1];
    st->rows_mean     = (double) st->n_assign/st->n_active;
    st->rows_p50      = ggml_moe_schedule_quantile(scratch, st->n_active, 0.50);
    st->rows_p90      = ggml_moe_schedule_quantile(scratch, st->n_active, 0.90);
    st->rows_p99      = ggml_moe_schedule_quantile(scratch, st->n_active, 0.99);
    st->max_over_mean = st->rows_max/st->rows_mean;
    st->n_eff         = (double) st->n_assign*st->n_assign/sum_sq;
}

#ifdef __cplusplus
}
#endif
