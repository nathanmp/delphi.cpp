//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

// IQ*_K dequantization kernels ported from ik_llama.cpp commit
// 3e416d7f5a9d4cc3195e8171dbf891541ca59c6a.

template<typename dst_t>
static __global__ void dequantize_block_iq4_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq4_ks * x = (const block_iq4_ks *)(cx + sizeof(float));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + ii*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = scale * ((x[i].scales[ib] & 254) - 127);
    const int8_t * values = iq4k_values + ((x[i].scales[ib] & 1) << 4);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = __float2bfloat16(d * values[q4[j] & 0xf]);
            y[j+16] = __float2bfloat16(d * values[q4[j] >>  4]);
        }
    } else {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = d * values[q4[j] & 0xf];
            y[j+16] = d * values[q4[j] >>  4];
        }
    }
}
template<typename dst_t>
static __global__ void dequantize_block_iq4_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_k * x = (const block_iq4_k *)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d;
    const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
    const float d1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
    const float d2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
    const int8_t * values1 = iq4k_values + 16*((x[i].extra >> (2*ib+0)) & 1);
    const int8_t * values2 = iq4k_values + 16*((x[i].extra >> (2*ib+1)) & 1);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = __float2bfloat16(d1 * values1[q4[j] & 0xf]);
            y[j+16] = __float2bfloat16(d2 * values2[q4[j] >>  4]);
        }
    } else {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = d1 * values1[q4[j] & 0xf];
            y[j+16] = d2 * values2[q4[j] >>  4];
        }
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq5_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_iq5_k * x = (const block_iq5_k *) vx;

    const int tid = threadIdx.x;
    int ib64 = tid/8; // 0...3
    int il   = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 64*ib64 + 2*il;
    const float d = (float)x[i].d;
    const float dl1 = d * (((x[i].scales_l[2*ib64+0] & 0xf) | ((x[i].scales_h[ib64] << 4) & 0x30)) - 32);
    const float dl2 = d * (((x[i].scales_l[2*ib64+0] >>  4) | ((x[i].scales_h[ib64] << 2) & 0x30)) - 32);
    const float dl3 = d * (((x[i].scales_l[2*ib64+1] & 0xf) | ((x[i].scales_h[ib64] >> 0) & 0x30)) - 32);
    const float dl4 = d * (((x[i].scales_l[2*ib64+1] >>  4) | ((x[i].scales_h[ib64] >> 2) & 0x30)) - 32);
    const uint8_t * qs = x[i].qs + 32*ib64 + 2*il;
    const uint8_t * qh = x[i].qh + 2*il;
    const uint8_t extra = x[i].extra >> 4*(ib64%4);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = __float2bfloat16(dl1 * iq5nl_values[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4) | ((extra << 5) & 0x20)]);
            y[j+16] = __float2bfloat16(dl2 * iq5nl_values[(qs[j+16] & 0xf) | ((h2 & 1) << 4) | ((extra << 4) & 0x20)]);
            y[j+32] = __float2bfloat16(dl3 * iq5nl_values[(qs[j+ 0] >>  4) | ((h1 & 2) << 3) | ((extra << 3) & 0x20)]);
            y[j+48] = __float2bfloat16(dl4 * iq5nl_values[(qs[j+16] >>  4) | ((h2 & 2) << 3) | ((extra << 2) & 0x20)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = dl1 * iq5nl_values[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4) | ((extra << 5) & 0x20)];
            y[j+16] = dl2 * iq5nl_values[(qs[j+16] & 0xf) | ((h2 & 1) << 4) | ((extra << 4) & 0x20)];
            y[j+32] = dl3 * iq5nl_values[(qs[j+ 0] >>  4) | ((h1 & 2) << 3) | ((extra << 3) & 0x20)];
            y[j+48] = dl4 * iq5nl_values[(qs[j+16] >>  4) | ((h2 & 2) << 3) | ((extra << 2) & 0x20)];
        }
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq5_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float d = *(const float *)cx;
    const block_iq5_ks * x = (const block_iq5_ks *)(cx + sizeof(float));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int tid = threadIdx.x;
    int ib64 = tid/8; // 0...3
    int il   = tid%8; // 0...7
    dst_t * y = yy + ii*QK_K + 64*ib64 + 2*il;
    const float dl1 = d * ((int)(x[i].scales[2*ib64+0] & 254) - 127);
    const float dl2 = d * ((int)(x[i].scales[2*ib64+1] & 254) - 127);
    const uint8_t * qs = x[i].qs + 32*ib64 + 2*il;
    const uint8_t * qh = x[i].qh + 2*il;
    auto values1 = iq5nl_values + ((x[i].scales[2*ib64+0] & 1) << 5);
    auto values2 = iq5nl_values + ((x[i].scales[2*ib64+1] & 1) << 5);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = __float2bfloat16(dl1 * values1[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4)]);
            y[j+16] = __float2bfloat16(dl1 * values1[(qs[j+16] & 0xf) | ((h2 & 1) << 4)]);
            y[j+32] = __float2bfloat16(dl2 * values2[(qs[j+ 0] >>  4) | ((h1 & 2) << 3)]);
            y[j+48] = __float2bfloat16(dl2 * values2[(qs[j+16] >>  4) | ((h2 & 2) << 3)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = dl1 * values1[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4)];
            y[j+16] = dl1 * values1[(qs[j+16] & 0xf) | ((h2 & 1) << 4)];
            y[j+32] = dl2 * values2[(qs[j+ 0] >>  4) | ((h1 & 2) << 3)];
            y[j+48] = dl2 * values2[(qs[j+16] >>  4) | ((h2 & 2) << 3)];
        }
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq6_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_iq6_k * x = (const block_iq6_k *) vx;

    const int tid = threadIdx.x;
    int ib64 = tid/8; // 0...3
    int il   = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 64*ib64 + 2*il;
    const float d = (float)x[i].d;
    const float dl1 = d * x[i].scales[4*ib64+0];
    const float dl2 = d * x[i].scales[4*ib64+1];
    const float dl3 = d * x[i].scales[4*ib64+2];
    const float dl4 = d * x[i].scales[4*ib64+3];
    const uint8_t * qs = x[i].qs + 32*ib64 + 2*il;
    const uint8_t * qh = x[i].qh + 32*(ib64/2) + 2*il;
    const uint8_t extra = x[i].extra >> 4*(ib64%4);
    for (int j = 0; j < 2; ++j) {
        const uint8_t h1 = qh[j] >> 4*(ib64%2), h2 = qh[j+16] >> 4*(ib64%2);
        uint8_t q1 = (qs[j+ 0] & 0xf) | ((h1 & 0x03) << 4);
        uint8_t q2 = (qs[j+16] & 0xf) | ((h2 & 0x03) << 4);
        uint8_t q3 = (qs[j+ 0] >>  4) | ((h1 & 0x0c) << 2);
        uint8_t q4 = (qs[j+16] >>  4) | ((h2 & 0x0c) << 2);
        if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
            y[j+ 0] = __float2bfloat16(dl1 * (iq6nl_values[q1] + (extra & 1 ? 1 : 0)));
            y[j+16] = __float2bfloat16(dl2 * (iq6nl_values[q2] + (extra & 2 ? 1 : 0)));
            y[j+32] = __float2bfloat16(dl3 * (iq6nl_values[q3] + (extra & 4 ? 1 : 0)));
            y[j+48] = __float2bfloat16(dl4 * (iq6nl_values[q4] + (extra & 8 ? 1 : 0)));
        } else {
            y[j+ 0] = dl1 * (iq6nl_values[q1] + (extra & 1 ? 1 : 0));
            y[j+16] = dl2 * (iq6nl_values[q2] + (extra & 2 ? 1 : 0));
            y[j+32] = dl3 * (iq6nl_values[q3] + (extra & 4 ? 1 : 0));
            y[j+48] = dl4 * (iq6nl_values[q4] + (extra & 8 ? 1 : 0));
        }
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_iq2_k * x = (const block_iq2_k *) vx;

    const int tid = threadIdx.x;
    int ib128 = tid/16; // 0 or 1
    int il    = tid%16; // 0...15
    dst_t * y = yy + i*QK_K + 128*ib128 + 2*il;
    const float d = (float)x[i].d;
    const float dl1 = d * (((x[i].scales[4*ib128+0] >> 4*(il/8)) & 0xf) - 8);
    const float dl2 = d * (((x[i].scales[4*ib128+1] >> 4*(il/8)) & 0xf) - 8);
    const float dl3 = d * (((x[i].scales[4*ib128+2] >> 4*(il/8)) & 0xf) - 8);
    const float dl4 = d * (((x[i].scales[4*ib128+3] >> 4*(il/8)) & 0xf) - 8);
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    const int16_t extra = x[i].extra >> (8*ib128 + (il/8));
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = __float2bfloat16(dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)]);
            y[j+32] = __float2bfloat16(dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 0) & 4)]);
            y[j+64] = __float2bfloat16(dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 2) & 4)]);
            y[j+96] = __float2bfloat16(dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 4) & 4)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)];
            y[j+32] = dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 0) & 4)];
            y[j+64] = dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 2) & 4)];
            y[j+96] = dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 4) & 4)];
        }
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float d = (float)*(const half *)cx;
    const block_iq2_ks * x = (const block_iq2_ks *)(cx + sizeof(half));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int tid = threadIdx.x;
    int ib128 = tid/16; // 0 or 1
    int il    = tid%16; // 0...15
    dst_t * y = yy + ii*QK_K + 128*ib128 + 2*il;
    const int16_t extra = x[i].extra >> 4*ib128;
    const float dl1 = d * (((x[i].scales[2*ib128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16);
    const float dl2 = d * (((x[i].scales[2*ib128+0] >>  4) | ((extra >> 5) & 0x10)) - 16);
    const float dl3 = d * (((x[i].scales[2*ib128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16);
    const float dl4 = d * (((x[i].scales[2*ib128+1] >>  4) | ((extra >> 7) & 0x10)) - 16);
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = __float2bfloat16(dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)]);
            y[j+32] = __float2bfloat16(dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 1) & 4)]);
            y[j+64] = __float2bfloat16(dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 0) & 4)]);
            y[j+96] = __float2bfloat16(dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 1) & 4)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)];
            y[j+32] = dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 1) & 4)];
            y[j+64] = dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 0) & 4)];
            y[j+96] = dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 1) & 4)];
        }
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_iq3_k * x = (const block_iq3_k *) vx;

    const int tid = threadIdx.x;
    int ib128 = tid/16; // 0 or 1
    int il    = tid%16; // 0...15
    dst_t * y = yy + i*QK_K + 128*ib128 + 2*il;
    const float d = (float)x[i].d;
    const uint16_t sh = x[i].scales_h >> (8*ib128 + (il/8));
    const float dl1 = d * ((2*((x[i].scales_l[4*ib128+0] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x01) ? -1 : 1));
    const float dl2 = d * ((2*((x[i].scales_l[4*ib128+1] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x04) ? -1 : 1));
    const float dl3 = d * ((2*((x[i].scales_l[4*ib128+2] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x10) ? -1 : 1));
    const float dl4 = d * ((2*((x[i].scales_l[4*ib128+3] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x40) ? -1 : 1));
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    const uint8_t * qh = x[i].qh + 2*il;
    const int16_t extra = x[i].extra >> (8*ib128 + (il/8));
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h = qh[j] >> (4*(ib128%2));
            y[j+ 0] = __float2bfloat16(dl1 * iq3nl_values[(((qs[j] >> 0) & 0x03) | ((h & 0x01) << 2)) + ((extra << 3) & 8)]);
            y[j+32] = __float2bfloat16(dl2 * iq3nl_values[(((qs[j] >> 2) & 0x03) | ((h & 0x02) << 1)) + ((extra << 1) & 8)]);
            y[j+64] = __float2bfloat16(dl3 * iq3nl_values[(((qs[j] >> 4) & 0x03) | ((h & 0x04) >> 0)) + ((extra >> 1) & 8)]);
            y[j+96] = __float2bfloat16(dl4 * iq3nl_values[(((qs[j] >> 6) & 0x03) | ((h & 0x08) >> 1)) + ((extra >> 3) & 8)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h = qh[j] >> (4*(ib128%2));
            y[j+ 0] = dl1 * iq3nl_values[(((qs[j] >> 0) & 0x03) | ((h & 0x01) << 2)) + ((extra << 3) & 8)];
            y[j+32] = dl2 * iq3nl_values[(((qs[j] >> 2) & 0x03) | ((h & 0x02) << 1)) + ((extra << 1) & 8)];
            y[j+64] = dl3 * iq3nl_values[(((qs[j] >> 4) & 0x03) | ((h & 0x04) >> 0)) + ((extra >> 1) & 8)];
            y[j+96] = dl4 * iq3nl_values[(((qs[j] >> 6) & 0x03) | ((h & 0x08) >> 1)) + ((extra >> 3) & 8)];
        }
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const ggml_half *)cx;
    const block_iq3_ks * x = (const block_iq3_ks *)(cx + sizeof(ggml_half));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t is = tid/16;
    const int64_t il = tid%16;
    dst_t * y = yy + ii*QK_K + 128*is + 2*il;
    const uint8_t  * qs = x[i].qs + 32*is + 2*il;
    const uint8_t  * qh = x[i].qh + 2*il;
    uint16_t extra = x[i].extra >> 4*is;
    const float d0 = scale * (int(((x[i].scales[0] >> 4*is) & 0xf) | ((extra << 4) & 0x10)) - 16);
    const float d1 = scale * (int(((x[i].scales[1] >> 4*is) & 0xf) | ((extra << 3) & 0x10)) - 16);
    const float d2 = scale * (int(((x[i].scales[2] >> 4*is) & 0xf) | ((extra << 2) & 0x10)) - 16);
    const float d3 = scale * (int(((x[i].scales[3] >> 4*is) & 0xf) | ((extra << 1) & 0x10)) - 16);
    extra >>= 8;
    const int8_t * values0 = iq3nl_values + ((extra & 1) << 3);
    const int8_t * values1 = iq3nl_values + ((extra & 2) << 2);
    const int8_t * values2 = iq3nl_values + ((extra & 4) << 1);
    const int8_t * values3 = iq3nl_values + ((extra & 8) << 0);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 4*is;
            y[j+ 0] = __float2bfloat16(d0 * values0[((qs[j] >> 0) & 3) | ((h << 2) & 4)]);
            y[j+32] = __float2bfloat16(d1 * values1[((qs[j] >> 2) & 3) | ((h << 1) & 4)]);
            y[j+64] = __float2bfloat16(d2 * values2[((qs[j] >> 4) & 3) | ((h >> 0) & 4)]);
            y[j+96] = __float2bfloat16(d3 * values3[((qs[j] >> 6) & 3) | ((h >> 1) & 4)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 4*is;
            y[j+ 0] = d0 * values0[((qs[j] >> 0) & 3) | ((h << 2) & 4)];
            y[j+32] = d1 * values1[((qs[j] >> 2) & 3) | ((h << 1) & 4)];
            y[j+64] = d2 * values2[((qs[j] >> 4) & 3) | ((h >> 0) & 4)];
            y[j+96] = d3 * values3[((qs[j] >> 6) & 3) | ((h >> 1) & 4)];
        }
    }
}

template<typename dst_t>
static void dequantize_row_iq4_ks_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_ks<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

template<typename dst_t>
static void dequantize_row_iq5_ks_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq5_ks<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

template<typename dst_t>
static void dequantize_row_iq2_ks_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ2_KS, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq2_ks<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

template<typename dst_t>
static void dequantize_row_iq2_k_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq2_k<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_k_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq3_k<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_ks_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ3_KS, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq3_ks<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

template<typename dst_t>
static void dequantize_row_iq4_k_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_k<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq5_k_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq5_k<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq6_k_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq6_k<<<nb, 32, 0, stream>>>(vx, y);
}
