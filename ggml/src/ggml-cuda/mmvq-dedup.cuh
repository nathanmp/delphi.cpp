#pragma once

#include "common.cuh"

// delphi: multi-token MUL_MAT_ID on packed IQ4_KS experts with each distinct expert decoded once for all tokens that route
// to it. mul_mat_vec_q_packed_moe gives every (token, slot) pair its own thread block, so an expert that two tokens of a
// verify batch share is read and decoded twice. Here every warp groups the ids itself (__match_any_sync, first-occurrence order),
// a thread block takes every 8th distinct expert, and each decoded weight word is dotted with the activations of all its tokens.
// The partial sums and their reduction are those of mul_mat_vec_q_packed_moe, so the results are bitwise the same.
// Used for the fused gate/up product of 3 or 4 tokens only: on CMP 170HX (GLM shapes, 64 of 288 experts so that ~27 of 32 pairs
// are distinct, as in real routing) gate/up n=4 takes 264 instead of 303 us, while n=2 (+2 %) and down (+5..+24 %) were slower.
//   GGML_CUDA_MMVQ_DEDUP  0: off (mul_mat_vec_q_packed_moe), 1: on (default)
// Returns false when the call is not one it handles (another format, no gate, fewer than 3 or more than 4 tokens, more than 32 pairs, biases).
bool ggml_cuda_mmvq_dedup_launch(int fmt, const char * vx, const void * vy, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device & fusion, float * dst, int ncols_x, int nrows_x, int ncols_dst, int stride_col_y,
        int stride_col_dst, int nchannels_y, int nchannels_dst, int64_t stride_channel_x, int stride_channel_y,
        int stride_channel_dst, int ids_stride, cudaStream_t stream);
