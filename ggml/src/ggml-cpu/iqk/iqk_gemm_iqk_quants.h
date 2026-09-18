// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT

#pragma once

#include "iqk_common.h"

#if defined(IQK_IMPLEMENT)

bool iqk_set_kernels_iqk_quants(
        int ne00, int type_a, int type_b,
        std::array<mul_mat_t, IQK_MAX_NY> & kernels,
        std::array<mul_mat_t, IQK_MAX_NY_HI - IQK_MAX_NY> & kernels_hi);

bool iqk_convert_iqk_quants_q80_r8(
        int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

#endif
