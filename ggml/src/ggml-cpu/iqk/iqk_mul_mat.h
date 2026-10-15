// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool iqk_mul_mat(long nx, long ny, long ne00,
        int type_a, const void * a, long stride_a,
        int type_b, const void * b, long stride_b,
        float * c, long stride_c, int ith, int nth);

bool iqk_mul_mat_4d(long nx, long ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        long nb02, long nb03, long nb12, long nb13, long nb2, long nb3,
        int type_a, const void * a, long stride_a,
        int type_b, const void * b, long stride_b,
        float * c, long stride_c, int ith, int nth);

// `rot` rotates which thread takes the remainder of the 8-row weight groups in the
// repacked path. Callers that loop over experts should pass the expert index.
bool iqk_mul_mat_moe(long nx, long ny, long ne00, int ne11,
        int type_a, const void * a, long stride_a,
        int type_b, const void * b, long stride_b,
        float * c, long nb1, long nb2, const void * row_mapping,
        int rot, int ith, int nth);

// Fused MoE up+gate: computes SwiGLU(gate*B, up*B), optionally clamped, into `c`.
// Layout requirements and `rot` are the same as iqk_mul_mat_moe. The row map must
// contain only valid routes; the caller clears inactive partial outputs.
bool iqk_moe_fused_up_gate(long nx, long ny, long ne00, int ne11,
        int type_a, const void * a_gate, const void * a_up, long stride_a,
        int type_b, const void * b, long stride_b,
        float * c, long nb1, long nb2, const void * row_mapping,
        bool clamp, float limit, int rot, int ith, int nth);

// Whether iqk_mul_mat_moe has kernels for this type pair. Diagnostics only: the
// caller must still meet the layout conditions checked in iqk_mul_mat_moe.
bool iqk_mul_mat_moe_supported(long ne00, int type_a, int type_b);

#ifdef __cplusplus
}
#endif
