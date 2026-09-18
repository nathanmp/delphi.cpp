#pragma once

#include "common.cuh"
#include "moe-packed.cuh"

struct mmq_args;

// delphi.cpp: grouped routed-MoE execution of MUL_MAT_ID MMQ (mmq-grouped.cuh), selected by GGML_CUDA_MOE_GROUPED:
//   old     (default) the MMQ path with mm_ids_helper and one tile width for all experts
//   auto    grouped for supported types from GGML_CUDA_MOE_GROUPED_MIN_TOKENS tokens on (default 128)
//   grouped grouped for every supported MMQ MUL_MAT_ID
//   packed  as grouped, but IQ3_S and IQ4_XS run with the packed-weight kernel of moe-packed.cuh (packed-f16 and packed-i8: Phase 3A0
//           experiments)
// GGML_CUDA_MOE_GROUPED_CLASSES sets the tile-width classes as "J:rows,...,J": an expert with rows <= the limit of the first
// class that fits takes that class, the last class takes the rest. The routing decides the tile count of a class, not its launch.
// GGML_CUDA_MOE_GROUPED_TIMING=1 times the tile list and each class kernel and fixup with events and logs them at backend free
// (measurement only: one synchronization per call, not while a stream is captured).

enum ggml_cuda_moe_grouped_mode {
    GGML_CUDA_MOE_GROUPED_OLD        = 0,
    GGML_CUDA_MOE_GROUPED_AUTO       = 1,
    GGML_CUDA_MOE_GROUPED_GROUPED    = 2,
    GGML_CUDA_MOE_GROUPED_PACKED_F16 = 3, // grouped, with the packed-weight kernel of moe-packed.cuh (experiment)
    GGML_CUDA_MOE_GROUPED_PACKED_I8  = 4,
    GGML_CUDA_MOE_GROUPED_PACKED     = 5, // grouped, with the packed kernel for IQ3_S and IQ4_XS; other types use grouped MMQ
};

// The current mode if it is a packed mode (GGML_CUDA_MOE_GROUPED_PACKED_F16, _I8 or GGML_CUDA_MOE_GROUPED_PACKED), else
// GGML_CUDA_MOE_GROUPED_OLD. ggml_cuda_moe_packed_select gives the packed format of one tensor.
int ggml_cuda_moe_grouped_packed_mode(void);

struct moe_packed_weights;

// Runs the grouped tile lists of one MUL_MAT_ID with the packed-weight kernel. y is in grouped order (fp16 rows for f16, q8_1 for i8).
void ggml_cuda_mul_mat_id_packed(ggml_backend_cuda_context & ctx, int fmt, const moe_packed_weights & w, const char * y, int64_t ncols_y,
        const int32_t * ids_dst, const int32_t * expert_bounds, float * dst, int64_t nrows, int64_t ncols_k, int n_experts,
        int64_t n_assign, int64_t stride_col_dst, cudaStream_t stream);

// Decides whether one MMQ MUL_MAT_ID runs grouped and counts the decision. shape_ok: src1 is broadcast over the top-k
// ([ne10, 1, n_tokens]) or packed as [ne10, top_k, n_tokens].
bool ggml_cuda_moe_grouped_select(int device, ggml_type type, int cc, int64_t n_tokens, bool shape_ok);

// Runs the MMQ of args from the schedule in args.expert_bounds and args.ids_dst. args.y holds the q8_1 rows in grouped
// order with at least GGML_CUDA_MOE_GROUPED_Y_PAD blocks of padding.
void ggml_cuda_mul_mat_q_grouped(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);

// Extra q8_1 blocks after the last grouped row: a tile of the widest class reads up to J - 1 rows past it.
#define GGML_CUDA_MOE_GROUPED_Y_PAD 128

// Logs the per-device decision counts if the grouped path was requested.
void ggml_cuda_moe_grouped_log_stats(int device);

// Entry points for tests and benchmarks, returned by the CUDA backend registry's get_proc_address.

// -1: use GGML_CUDA_MOE_GROUPED, else a ggml_cuda_moe_grouped_mode.
void ggml_backend_cuda_moe_grouped_set_mode(int mode);

// counts[0]: grouped calls, counts[1]: calls on the old path although the grouped path was requested. Returns false for a bad device.
bool ggml_backend_cuda_moe_grouped_get_counts(int device, int64_t * counts);
