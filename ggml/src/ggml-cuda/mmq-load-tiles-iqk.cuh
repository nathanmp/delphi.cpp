//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "iqk_cuda_common.h"

// IQ*_K MMQ tile loaders adapted from ik_llama.cpp commit 3e416d7f5a9d4cc3195e8171dbf891541ca59c6a.

template <typename block_t>
static __device__ __forceinline__ const char * iqk_mmq_row_ptr(
        const char * x, const int logical_kbx, const int blocks_per_row, const int row_meta_size) {
    return x + (logical_kbx / blocks_per_row) * (row_meta_size + blocks_per_row * sizeof(block_t));
}

template <typename block_t>
static __device__ __forceinline__ const block_t * iqk_mmq_block_ptr(
        const char * x, const int logical_kbx, const int blocks_per_row, const int row_meta_size) {
    const char * row = iqk_mmq_row_ptr<block_t>(x, logical_kbx, blocks_per_row, row_meta_size);
    return (const block_t *) (row + row_meta_size) + logical_kbx % blocks_per_row;
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int  kbx0, const int  i_max, const int  stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x%16;

#ifdef __CUDA_ARCH__
    #pragma unroll
    for (int i0 = 0; i0 < I; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = iqk_mmq_block_ptr<block_iq2_ks>(x, kbx0 + i*stride, stride, sizeof(half));

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

        uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;
        uint32_t val1 = ((q2 >> 0) & 0x33333333) | ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
        uint32_t val2 = ((q2 >> 2) & 0x33333333) | ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
        int2 v1 = get_int_from_table_8(val1, iq2nl_values);
        int2 v2 = get_int_from_table_8(val2, iq2nl_values);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*sram_stride + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*sram_stride + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*sram_stride + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#else // __CUDA_ARCH__

    const int * all_values = (const int *)iq2k_table;
    #pragma unroll
    for (int i0 = 0; i0 < I; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = iqk_mmq_block_ptr<block_iq2_ks>(x, kbx0 + i*stride, stride, sizeof(half));

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*sram_stride + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*sram_stride + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*sram_stride + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
#endif // __CUDA_ARCH__

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * 8) {
        int i = i0 + threadIdx.y * 8 + threadIdx.x / 4;

        if (fallback) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *) iqk_mmq_row_ptr<block_iq2_ks>(x, kbx0 + i*stride, stride, sizeof(half));
        const float d = dptr[0];
        const block_iq2_ks * bxi = iqk_mmq_block_ptr<block_iq2_ks>(x, kbx0 + i*stride, stride, sizeof(half));
        const int ls1 = ((bxi->scales[threadIdx.x % 4] >> 0) & 0xf) | ((bxi->extra >> (4 + 2*(threadIdx.x % 4))) & 0x10);
        const int ls2 = ((bxi->scales[threadIdx.x % 4] >> 4) & 0xf) | ((bxi->extra >> (5 + 2*(threadIdx.x % 4))) & 0x10);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*sram_stride + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4   + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*(MMQ_TILE_NE_K/4) + i/4   + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int  kbx0, const int  i_max, const int  stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    #pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * MMQ_TILE_NE_K/qstep) {
        int i = i0 + threadIdx.y*(MMQ_TILE_NE_K/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_k * bxi = (const block_iq2_k *) x + i*stride + kbx0;

        const float d = bxi->d;
        uint16_t extra = bxi->extra >> (kqsx/4);

#ifdef __CUDA_ARCH__

        uint32_t extra32[2] = { uint32_t(extra & 0xff) * 0x01010101, uint32_t(extra >> 8) * 0x01010101 };
        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {
            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((extra32[l] << 2) & 0x44444444);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((extra32[l] << 0) & 0x44444444);
            int2 v1 = get_int_from_table_8(val1, iq2nl_values);
            int2 v2 = get_int_from_table_8(val2, iq2nl_values);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + kqsx + 32*l +  0] = v1.x;
            x_qs[i*sram_stride + kqsx + 32*l +  8] = v2.x;
            x_qs[i*sram_stride + kqsx + 32*l + 16] = v1.y;
            x_qs[i*sram_stride + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#else

        auto all_values = (const int *)iq2k_table;

        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + kqsx + 32*l +  0] = int_from_table_4((ql >> 0) & 0x03030303, all_values + ((extra & 0x01) << 8));
            x_qs[i*sram_stride + kqsx + 32*l +  8] = int_from_table_4((ql >> 2) & 0x03030303, all_values + ((extra & 0x04) << 6));
            x_qs[i*sram_stride + kqsx + 32*l + 16] = int_from_table_4((ql >> 4) & 0x03030303, all_values + ((extra & 0x10) << 4));
            x_qs[i*sram_stride + kqsx + 32*l + 24] = int_from_table_4((ql >> 6) & 0x03030303, all_values + ((extra & 0x40) << 2));
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l +  0] = int_from_table_4((ql >> 0) & 0x03030303, all_values + ((extra & 0x01) << 8));
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l +  8] = int_from_table_4((ql >> 2) & 0x03030303, all_values + ((extra & 0x04) << 6));
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l + 16] = int_from_table_4((ql >> 4) & 0x03030303, all_values + ((extra & 0x10) << 4));
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l + 24] = int_from_table_4((ql >> 6) & 0x03030303, all_values + ((extra & 0x40) << 2));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

            extra >>= 8;
        }
#endif // __CUDA_ARCH__

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride               + 2*kqsx+0] = d * (((bxi->scales[kqsx] >> 0) & 0xf) - 8);
        x_df[i*sram_stride               + 2*kqsx+1] = d * (((bxi->scales[kqsx] >> 4) & 0xf) - 8);
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * (((bxi->scales[kqsx] >> 0) & 0xf) - 8);
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * (((bxi->scales[kqsx] >> 4) & 0xf) - 8);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int  kbx0, const int  i_max, const int  stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * MMQ_TILE_NE_K/qstep) {
        int i = i0 + threadIdx.y*(MMQ_TILE_NE_K/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_k * bxi = (const block_iq3_k *) x + i*stride + kbx0;

        const float d = bxi->d;

        uint16_t extra = bxi->extra >> (kqsx/4);
        uint32_t extra32[2] = { uint32_t(extra & 0xff) * 0x01010101, uint32_t(extra >> 8) * 0x01010101 };
        int qh = get_int_b2(bxi->qh, kqsx);

    #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            //extra << 3, extra << 1, extra >> 1, extra >> 3
            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((extra32[l] << 3) & 0x88888888)
                          | ((qh << 2) & 0x04040404) | ((qh << 4) & 0x40404040);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((extra32[l] << 1) & 0x88888888)
                          | ((qh << 1) & 0x04040404) | ((qh << 3) & 0x40404040);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            qh >>= 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + kqsx + 32*l +  0] = v1.x;
            x_qs[i*sram_stride + kqsx + 32*l +  8] = v2.x;
            x_qs[i*sram_stride + kqsx + 32*l + 16] = v1.y;
            x_qs[i*sram_stride + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        uint16_t sh = bxi->scales_h >> 2*kqsx;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride               + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*sram_stride               + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int  kbx0, const int  i_max, const int  stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * MMQ_TILE_NE_K/qstep) {
        int i = i0 + threadIdx.y*(MMQ_TILE_NE_K/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *) iqk_mmq_row_ptr<block_iq3_ks>(x, kbx0 + i*stride, stride, sizeof(half));
        const float d = __half2float(dptr[0]);
        const block_iq3_ks * bxi = iqk_mmq_block_ptr<block_iq3_ks>(x, kbx0 + i*stride, stride, sizeof(half));

        //uint16_t extra = bxi->extra >> 8;
        int qh = get_int_b2(bxi->qh, kqsx);

        uint32_t extra32 = uint32_t(bxi->extra >> 8) * 0x01010101;

        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((qh << 2) & 0x04040404) | ((extra32 << 3) & 0x08080808)
                                                     | ((qh << 4) & 0x40404040) | ((extra32 << 5) & 0x80808080);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((qh << 1) & 0x04040404) | ((extra32 << 2) & 0x08080808)
                                                     | ((qh << 3) & 0x40404040) | ((extra32 << 4) & 0x80808080);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            extra32 >>= 4;
            qh      >>= 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + kqsx + 32*l +  0] = v1.x;
            x_qs[i*sram_stride + kqsx + 32*l +  8] = v2.x;
            x_qs[i*sram_stride + kqsx + 32*l + 16] = v1.y;
            x_qs[i*sram_stride + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4   + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int  kbx0, const int  i_max, const int  stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x / 4;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (fallback) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *) iqk_mmq_row_ptr<block_iq4_ks>(x, kbx0 + i*stride, stride, sizeof(float));
        const block_iq4_ks * bxi = iqk_mmq_block_ptr<block_iq4_ks>(x, kbx0 + i*stride, stride, sizeof(float));
        const int ls = (bxi->scales[kqsx] & 254) - 127;

        auto values = iq4k_values + ((bxi->scales[kqsx] & 1) << 4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q4 = get_int_b4(bxi->qs, 4*kqsx+j);
            const int2 v = get_int_from_table_16(q4, values);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + j + 0] = v.x;
            x_qs[i*sram_stride + 8*kqsx + j + 4] = v.y;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + 8*kqsx + j + 0] = v.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + 8*kqsx + j + 4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride + kqsx] = dptr[0] * ls;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4   + kqsx] = dptr[0] * ls;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int  kbx0, const int  i_max, const int  stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * MMQ_TILE_NE_K/qstep) {
        int i = i0 + threadIdx.y*(MMQ_TILE_NE_K/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_k * bxi = (const block_iq4_k *) x + i*stride + kbx0;
        const uint16_t extra = bxi->extra >> 2*kqsx;

        // the second iq4k_values table is the first plus 4 in each byte (no byte carries)
        const int shift_l = (extra & 1)*0x04040404;
        const int shift_h = ((extra >> 1) & 1)*0x04040404;

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int q4 = get_int_b4(bxi->qs, (qstep/2)*kqsx + l);
            const int2 v = get_int_from_table_16(q4, iq4k_values);

            int val0 = v.x + shift_l;
            int val1 = v.y + shift_h;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + l + 0] = val0;
            x_qs[i*sram_stride + 8*kqsx + l + 4] = val1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + 8*kqsx + l + 0] = val0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + 8*kqsx + l + 4] = val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const uint8_t sh = bxi->scales_h[kqsx/2] >> 4*(kqsx%2);
        const int ls1 = ((bxi->scales_l[kqsx] & 0xf) | ((sh << 4) & 0x30)) - 32;
        const int ls2 = ((bxi->scales_l[kqsx] >>  4) | ((sh << 2) & 0x30)) - 32;

        const float d = bxi->d;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride               + 2*kqsx+0] = d * ls1;
        x_df[i*sram_stride               + 2*kqsx+1] = d * ls2;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ls1;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ls2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq5_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int  kbx0, const int  i_max, const int  stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq5nl_values;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * MMQ_TILE_NE_K/qstep) {
        int i = i0 + threadIdx.y*(MMQ_TILE_NE_K/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq5_k * bxi = (const block_iq5_k *) x + i*stride + kbx0;

        int qh = get_int_b4(bxi->qh, kqsx);
        uint16_t extra = bxi->extra >> (kqsx/4);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((extra & 1) * 0x20202020); // this is very slightly faster
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((extra & 4) * 0x08080808); // then the version below
            //aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((extra & 1) ? 0x20202020 : 0);
            //aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((extra & 4) ? 0x20202020 : 0);
            qh    >>= 2;
            extra >>= 4;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*sram_stride + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const uint8_t sh = bxi->scales_h[kqsx/2] >> 4*(kqsx%2);
        const int ls1 = ((bxi->scales_l[kqsx] & 0xf) | ((sh << 4) & 0x30)) - 32;
        const int ls2 = ((bxi->scales_l[kqsx] >>  4) | ((sh << 2) & 0x30)) - 32;

        const float d = bxi->d;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride               + 2*kqsx+0] = d * ls1;
        x_df[i*sram_stride               + 2*kqsx+1] = d * ls2;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ls1;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ls2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq5_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int  kbx0, const int  i_max, const int  stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ5_KS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq5nl_values;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * MMQ_TILE_NE_K/qstep) {
        int i = i0 + threadIdx.y*(MMQ_TILE_NE_K/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *) iqk_mmq_row_ptr<block_iq5_ks>(x, kbx0 + i*stride, stride, sizeof(float));
        const float d = dptr[0];
        const block_iq5_ks * bxi = iqk_mmq_block_ptr<block_iq5_ks>(x, kbx0 + i*stride, stride, sizeof(float));

        int qh = get_int_b4(bxi->qh, kqsx);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((bxi->scales[2*l+0] & 1) * 0x20202020);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((bxi->scales[2*l+1] & 1) * 0x20202020);
            qh    >>= 2;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*sram_stride + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride               + kqsx] = d * ((bxi->scales[kqsx] & 254) - 127);
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + kqsx] = d * ((bxi->scales[kqsx] & 254) - 127);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq6_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int  kbx0, const int  i_max, const int  stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq6nl_values;
    int qh[2];

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * MMQ_TILE_NE_K/qstep) {
        int i = i0 + threadIdx.y*(MMQ_TILE_NE_K/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq6_k * bxi = (const block_iq6_k *) x + i*stride + kbx0;

        const float d = bxi->d;
        uint16_t extra = bxi->extra >> (kqsx/4);

        qh[0] = get_int_b4(bxi->qh, kqsx+0);
        qh[1] = get_int_b4(bxi->qh, kqsx+8);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh[l/2] & 0x03030303) << 4) | ((extra & 1) * 0x40404040);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh[l/2] & 0x0c0c0c0c) << 2) | ((extra & 4) * 0x10101010);
            qh[l/2] >>= 4;
            extra   >>= 4;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*sram_stride + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }


#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride               + 2*kqsx+0] = d * bxi->scales[2*kqsx+0];
        x_df[i*sram_stride               + 2*kqsx+1] = d * bxi->scales[2*kqsx+1];
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * bxi->scales[2*kqsx+0];
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * bxi->scales[2*kqsx+1];
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}
