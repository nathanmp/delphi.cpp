// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT

#include "iqk_config.h"
#include "iqk_mul_mat.h"

#if defined(IQK_IMPLEMENT)

#include "iqk_gemm_iqk_quants.h"
#include "../moe-prof.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static bool is_iqk_type(int type) {
    switch ((ggml_type) type) {
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ5_KS:
            return true;
        default:
            return false;
    }
}

// SWIGLU_CLAMP combine, kept identical to ggml_compute_forward_swiglu_clamp_f32 so that the
// fused path is bit-for-bit equal to running the two matmuls and the GLU separately.
static void iqk_swiglu_clamp(int n, float * up_dst, const float * gate, float limit) {
    for (int j = 0; j < n; ++j) {
        const float g = std::min(gate[j], limit);
        const float u = std::min(std::max(up_dst[j], -limit), limit);
        up_dst[j] = g / (1.0f + expf(-g)) * u;
    }
}

struct MulMat {
    std::array<mul_mat_t, IQK_MAX_NY> funcs = {};
    // Kernels for IQK_MAX_NY+1 .. IQK_MAX_NY_HI activation rows, where a type has them. A kernel
    // unpacks each weight row once and reuses it for every activation row it holds, so a call that
    // exceeds the widest kernel has to walk and unpack the same weight rows again. The wider
    // kernels halve that pass count for the routed-expert row counts that sit above 8.
    std::array<mul_mat_t, IQK_MAX_NY_HI - IQK_MAX_NY> funcs_hi = {};

    int max_ny() const {
        int ny = IQK_MAX_NY_HI;
        while (ny > 0 && !kernel(ny)) {
            --ny;
        }
        return ny;
    }

    mul_mat_t kernel(int ny) const {
        return ny <= IQK_MAX_NY ? funcs[ny - 1] : funcs_hi[ny - IQK_MAX_NY - 1];
    }

    void mul_mat_nxm(
            int n, const void * vx, size_t bx, DataInfo & info,
            int nrc_x, int nrc_y) const {
        constexpr int x_step = 64;

        int ny = max_ny();
        GGML_ASSERT(ny > 0);

        int n_left = nrc_y - info.cur_y;
        int n_step = n_left / ny;
        if (n_step > 0) {
            if (n_step * ny != n_left) {
                ++n_step;
                const int ny1 = n_left / n_step;
                const int ny2 = ny1 + 1;
                const int my1 = n_step * ny2 - n_left;
                const int my2 = n_step - my1;
                for (int ix = 0; ix < nrc_x; ix += x_step) {
                    DataInfo this_info = info;
                    this_info.s += ix;
                    const int this_nrc_x = std::min(x_step, nrc_x - ix);
                    for (int iy = 0; iy < my1; ++iy) {
                        kernel(ny1)(n, (const char *) vx + ix * bx, bx, this_info, this_nrc_x);
                        this_info.cur_y += ny1;
                    }
                    for (int iy = 0; iy < my2; ++iy) {
                        kernel(ny2)(n, (const char *) vx + ix * bx, bx, this_info, this_nrc_x);
                        this_info.cur_y += ny2;
                    }
                }
                info.cur_y += n_left;
            } else {
                for (int ix = 0; ix < nrc_x; ix += x_step) {
                    DataInfo this_info = info;
                    this_info.s += ix;
                    const int this_nrc_x = std::min(x_step, nrc_x - ix);
                    for (int iy = 0; iy < n_step; ++iy) {
                        kernel(ny)(n, (const char *) vx + ix * bx, bx, this_info, this_nrc_x);
                        this_info.cur_y += ny;
                    }
                }
                info.cur_y += ny * n_step;
            }
        }

        n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            kernel(n_left)(n, vx, bx, info, nrc_x);
        }
    }

    // Fused up/gate variant of mul_mat_nxm. The gate GEMM writes the destination rows, the
    // result is copied into the tile buffer, then the up GEMM overwrites the same rows and the
    // two are combined in place. Unlike mul_mat_nxm this tiles x unconditionally, because the
    // tile buffer bounds how many columns can be held between the two GEMMs.
    void mul_mat_up_gate_nxm(
            int n, const void * vx_gate, const void * vx_up, size_t bx,
            DataInfo & info, int nrc_x, int nrc_y, float limit) const {
        constexpr int x_step = 64;
        float tmp[x_step * IQK_MAX_NY_HI];

        auto process = [&](mul_mat_t func, DataInfo & this_info, int ix, int this_nrc_x, int ny) {
            func(n, (const char *) vx_gate + ix * bx, bx, this_info, this_nrc_x);
            for (int ky = 0; ky < ny; ++ky) {
                memcpy(tmp + ky * x_step, this_info.dst_row(ky), this_nrc_x * sizeof(float));
            }
            func(n, (const char *) vx_up + ix * bx, bx, this_info, this_nrc_x);
            for (int ky = 0; ky < ny; ++ky) {
                iqk_swiglu_clamp(this_nrc_x, this_info.dst_row(ky), tmp + ky * x_step, limit);
            }
        };

        int ny = max_ny();
        GGML_ASSERT(ny > 0 && ny <= IQK_MAX_NY_HI);

        int n_left = nrc_y - info.cur_y;
        int n_step = n_left / ny;
        if (n_step > 0) {
            if (n_step * ny != n_left) {
                ++n_step;
                const int ny1 = n_left / n_step;
                const int ny2 = ny1 + 1;
                const int my1 = n_step * ny2 - n_left;
                const int my2 = n_step - my1;
                for (int ix = 0; ix < nrc_x; ix += x_step) {
                    DataInfo this_info = info;
                    this_info.s += ix;
                    const int this_nrc_x = std::min(x_step, nrc_x - ix);
                    for (int iy = 0; iy < my1; ++iy) {
                        process(kernel(ny1), this_info, ix, this_nrc_x, ny1);
                        this_info.cur_y += ny1;
                    }
                    for (int iy = 0; iy < my2; ++iy) {
                        process(kernel(ny2), this_info, ix, this_nrc_x, ny2);
                        this_info.cur_y += ny2;
                    }
                }
                info.cur_y += n_left;
            } else {
                for (int ix = 0; ix < nrc_x; ix += x_step) {
                    DataInfo this_info = info;
                    this_info.s += ix;
                    const int this_nrc_x = std::min(x_step, nrc_x - ix);
                    for (int iy = 0; iy < n_step; ++iy) {
                        process(kernel(ny), this_info, ix, this_nrc_x, ny);
                        this_info.cur_y += ny;
                    }
                }
                info.cur_y += ny * n_step;
            }
        }

        n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            for (int ix = 0; ix < nrc_x; ix += x_step) {
                DataInfo this_info = info;
                this_info.s += ix;
                const int this_nrc_x = std::min(x_step, nrc_x - ix);
                process(kernel(n_left), this_info, ix, this_nrc_x, n_left);
            }
            info.cur_y += n_left;
        }
    }

    static bool prepare(int type_a, int type_b, int ne00, MulMat & mm) {
        if (type_a != IQK_TYPE_Q8_K_R8 && !is_iqk_type(type_a)) {
            return false;
        }
        return iqk_set_kernels_iqk_quants(ne00, type_a, type_b, mm.funcs, mm.funcs_hi);
    }
};

static std::vector<char> & thread_local_work_buffer() {
    thread_local std::vector<char> buffer;
    return buffer;
}

// Converting a weight row group to Q8_K_R8 costs the same whatever `ny` is, so it
// only pays once enough activation columns share the decode. ik_llama.cpp uses 32
// for every IQ*_K type on AVX2; GGML_IQK_REPACK_MIN_NY exists to A/B that against
// the routed-row distribution of a real MoE, where most experts are far below it.
static long repack_min_ny(void) {
    static const long min_ny = [] {
        const char * value = getenv("GGML_IQK_REPACK_MIN_NY");
        const long parsed = value ? strtol(value, nullptr, 10) : 0;
        return parsed > 0 ? parsed : 32;
    }();
    return min_ny;
}

static bool should_repack(int type_a, long ny) {
    return is_iqk_type(type_a) && ny >= repack_min_ny();
}

static size_t repacked_row_size(long ne00) {
    GGML_ASSERT(ne00 % QK_K == 0);
    return sizeof(block_q8_k_r8) * (size_t) ne00 / QK_K / 8;
}

static uint32_t simple_gcd(uint32_t a, uint32_t b) {
    while (a != b) {
        if (a > b) {
            a -= b;
        } else {
            b -= a;
        }
    }
    return a;
}

} // namespace

extern "C" bool iqk_mul_mat(
        long nx, long ny, long ne00,
        int type_a, const void * a, long stride_a,
        int type_b, const void * b, long stride_b,
        float * c, long stride_c, int ith, int nth) {
    if (!is_iqk_type(type_a) || (ggml_type) type_b != GGML_TYPE_Q8_K ||
        nx <= 0 || ny <= 0 || ne00 % QK_K != 0 || nth <= 0) {
        return false;
    }

    constexpr int min_thread_step = 32;
    MulMat mm;

    if (nx / nth < min_thread_step) {
        if (!MulMat::prepare(type_a, type_b, (int) ne00, mm)) {
            return false;
        }
        const int min_step = ny <= 16 ? 16 : 32;
        const int ntile_x = (nx + min_step - 1) / min_step;
        const int ntile_y = (ny + min_step - 1) / min_step;
        const int ntile = ntile_x * ntile_y;
        for (int itile = ith; itile < ntile; itile += nth) {
            const int iy = (itile / ntile_x) * min_step;
            const int ix = (itile % ntile_x) * min_step;
            const int nrc_x = std::min<long>(min_step, nx - ix);
            const int nrc_y = std::min<long>(min_step, ny - iy);
            DataInfo info{c + ix, (const char *) b, (size_t) stride_c,
                (size_t) stride_b, iy, 1, nullptr, 0};
            mm.mul_mat_nxm((int) ne00, (const char *) a + ix * stride_a,
                (size_t) stride_a, info, nrc_x, iy + nrc_y);
        }
        return true;
    }

    const int npt = (nx + nth - 1) / nth;
    if (should_repack(type_a, ny) && npt >= 16 && nx % 8 == 0 &&
        MulMat::prepare(IQK_TYPE_Q8_K_R8, type_b, (int) ne00, mm)) {
        int nrc_x = 8 * ((nx / 8 + nth - 1) / nth);
        int first_x = ith * nrc_x;
        nrc_x = (int) std::max<long>(0, std::min<long>(nrc_x, nx - first_x));

        const size_t row_size = repacked_row_size(ne00);
        DataInfo info{c + first_x, (const char *) b, (size_t) stride_c,
            (size_t) stride_b, 0, 1, nullptr, 0};
        auto & work = thread_local_work_buffer();

        constexpr int x_step = 32;
        for (int ix = 0; ix < nrc_x; ix += x_step) {
            DataInfo this_info = info;
            this_info.s += ix;
            const int this_nrc_x = std::min(x_step, nrc_x - ix);
            work.resize(row_size * this_nrc_x);
            if (!iqk_convert_iqk_quants_q80_r8(type_a, (int) ne00,
                    (const char *) a + (first_x + ix) * stride_a,
                    (size_t) stride_a, work.data(), this_nrc_x)) {
                return false;
            }
            mm.mul_mat_nxm((int) ne00, work.data(), row_size,
                this_info, this_nrc_x, (int) ny);
        }
        return true;
    }

    if (!MulMat::prepare(type_a, type_b, (int) ne00, mm)) {
        return false;
    }

    int nrc_x = (nx + nth - 1) / nth;
    const int first_x = ith * nrc_x;
    nrc_x = (int) std::max<long>(0, std::min<long>(nrc_x, nx - first_x));
    DataInfo info{c + first_x, (const char *) b, (size_t) stride_c,
        (size_t) stride_b, 0, 1, nullptr, 0};
    mm.mul_mat_nxm((int) ne00, (const char *) a + first_x * stride_a,
        (size_t) stride_a, info, nrc_x, (int) ny);
    return true;
}

extern "C" bool iqk_mul_mat_4d(
        long nx, long ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        long nb02, long nb03, long nb12, long nb13, long nb2, long nb3,
        int type_a, const void * a, long stride_a,
        int type_b, const void * b, long stride_b,
        float * c, long stride_c, int ith, int nth) {
    if (ne02 <= 0 || ne03 <= 0 || ne12 % ne02 != 0 || ne13 % ne03 != 0) {
        return false;
    }

    const long r2 = ne12 / ne02;
    const long r3 = ne13 / ne03;
    const int gcd = (int) simple_gcd((uint32_t) (ne12 * ne13), (uint32_t) nth);
    int counter = 0;
    for (long i13 = 0; i13 < ne13; ++i13) {
        for (long i12 = 0; i12 < ne12; ++i12) {
            if ((counter++ % gcd) != (ith % gcd)) {
                continue;
            }
            if (!iqk_mul_mat(nx, ny, ne00,
                    type_a, (const char *) a + (i12 / r2) * nb02 + (i13 / r3) * nb03, stride_a,
                    type_b, (const char *) b + i12 * nb12 + i13 * nb13, stride_b,
                    c + i12 * nb2 + i13 * nb3, stride_c,
                    ith / gcd, nth / gcd)) {
                return false;
            }
        }
    }
    return true;
}

extern "C" bool iqk_mul_mat_moe_supported(long ne00, int type_a, int type_b) {
    if (!is_iqk_type(type_a) || (ggml_type) type_b != GGML_TYPE_Q8_K || ne00 % QK_K != 0) {
        return false;
    }
    MulMat mm;
    return MulMat::prepare(type_a, type_b, (int) ne00, mm);
}

extern "C" bool iqk_mul_mat_moe(
        long nx, long ny, long ne00, int ne11,
        int type_a, const void * a, long stride_a,
        int type_b, const void * b, long stride_b,
        float * c, long nb1, long nb2, const void * vrow_mapping,
        int rot, int ith, int nth) {
    if (!is_iqk_type(type_a) || (ggml_type) type_b != GGML_TYPE_Q8_K ||
        !vrow_mapping || nx <= 0 || ny <= 0 || ne00 % QK_K != 0 || nth <= 0) {
        return false;
    }

    const auto * row_mapping = (const mmid_row_mapping *) vrow_mapping;
    MulMat mm;

    if (should_repack(type_a, ny) && nx % 8 == 0 &&
        MulMat::prepare(IQK_TYPE_Q8_K_R8, type_b, (int) ne00, mm)) {
        // The repack works on 8 row groups, and nx/8 groups seldom divide evenly over
        // the thread group. Splitting by a fixed `ith * ceil(groups/nth)` hands the
        // same threads one extra group on every expert and leaves the last threads
        // with nothing, so the slowest thread sets the critical path for the whole
        // expert loop. Rotating the slot by the expert index spreads the remainder.
        const long n_groups = nx / 8;
        const int  slot     = (int) (((long) ith + rot) % nth);
        const long first_g  = ((long) slot * n_groups) / nth;
        const long last_g   = ((long) (slot + 1) * n_groups) / nth;
        const int  first_x  = (int) (first_g * 8);
        const int  nrc_x    = (int) ((last_g - first_g) * 8);
        const size_t row_size = repacked_row_size(ne00);
        DataInfo info{c + first_x, (const char *) b, (size_t) nb1 / sizeof(float),
            (size_t) stride_b, 0, ne11, row_mapping, (size_t) nb2 / sizeof(float)};
        auto & work = thread_local_work_buffer();

        const bool prof = ggml_moe_prof_enabled();

        constexpr int x_step = 32;
        for (int ix = 0; ix < nrc_x; ix += x_step) {
            DataInfo this_info = info;
            this_info.s += ix;
            const int this_nrc_x = std::min(x_step, nrc_x - ix);
            work.resize(row_size * this_nrc_x);
            const int64_t t0 = prof ? ggml_time_us() : 0;
            if (!iqk_convert_iqk_quants_q80_r8(type_a, (int) ne00,
                    (const char *) a + (first_x + ix) * stride_a,
                    (size_t) stride_a, work.data(), this_nrc_x)) {
                return false;
            }
            const int64_t t1 = prof ? ggml_time_us() : 0;
            mm.mul_mat_nxm((int) ne00, work.data(), row_size,
                this_info, this_nrc_x, (int) ny);
            if (prof) {
                ggml_moe_prof_add(GGML_MOE_PROF_REPACK, t1 - t0);
                ggml_moe_prof_add(GGML_MOE_PROF_GEMM, ggml_time_us() - t1);
            }
        }
        return true;
    }

    if (!MulMat::prepare(type_a, type_b, (int) ne00, mm)) {
        return false;
    }

    int nrc_x = (nx + nth - 1) / nth;
    const int first_x = ith * nrc_x;
    nrc_x = (int) std::max<long>(0, std::min<long>(nrc_x, nx - first_x));
    DataInfo info{c + first_x, (const char *) b, (size_t) nb1 / sizeof(float),
        (size_t) stride_b, 0, ne11, row_mapping, (size_t) nb2 / sizeof(float)};
    const int64_t t0 = ggml_moe_prof_enabled() ? ggml_time_us() : 0;
    mm.mul_mat_nxm((int) ne00, (const char *) a + first_x * stride_a,
        (size_t) stride_a, info, nrc_x, (int) ny);
    if (t0) {
        ggml_moe_prof_add(GGML_MOE_PROF_GEMM, ggml_time_us() - t0);
    }
    return true;
}

extern "C" bool iqk_moe_fused_up_gate(
        long nx, long ny, long ne00, int ne11,
        int type_a, const void * a_gate, const void * a_up, long stride_a,
        int type_b, const void * b, long stride_b,
        float * c, long nb1, long nb2, const void * vrow_mapping,
        float limit, int rot, int ith, int nth) {
    if (!is_iqk_type(type_a) || (ggml_type) type_b != GGML_TYPE_Q8_K ||
        !vrow_mapping || nx <= 0 || ny <= 0 || ne00 % QK_K != 0 || nth <= 0) {
        return false;
    }

    const auto * row_mapping = (const mmid_row_mapping *) vrow_mapping;
    MulMat mm;

    if (should_repack(type_a, ny) && nx % 8 == 0 &&
        MulMat::prepare(IQK_TYPE_Q8_K_R8, type_b, (int) ne00, mm)) {
        // Same rotated row split as iqk_mul_mat_moe. Fusion does not halve the repack:
        // gate and up are different matrices, so both row tiles must be converted.
        const long n_groups = nx / 8;
        const int  slot     = (int) (((long) ith + rot) % nth);
        const long first_g  = ((long) slot * n_groups) / nth;
        const long last_g   = ((long) (slot + 1) * n_groups) / nth;
        const int  first_x  = (int) (first_g * 8);
        const int  nrc_x    = (int) ((last_g - first_g) * 8);
        const size_t row_size = repacked_row_size(ne00);
        DataInfo info{c + first_x, (const char *) b, (size_t) nb1 / sizeof(float),
            (size_t) stride_b, 0, ne11, row_mapping, (size_t) nb2 / sizeof(float)};
        auto & work = thread_local_work_buffer();

        const bool prof = ggml_moe_prof_enabled();

        constexpr int x_step = 32;
        for (int ix = 0; ix < nrc_x; ix += x_step) {
            DataInfo this_info = info;
            this_info.s += ix;
            const int this_nrc_x = std::min(x_step, nrc_x - ix);
            const size_t tile_size = row_size * this_nrc_x;
            work.resize(2 * tile_size);
            const int64_t t0 = prof ? ggml_time_us() : 0;
            if (!iqk_convert_iqk_quants_q80_r8(type_a, (int) ne00,
                    (const char *) a_gate + (first_x + ix) * stride_a,
                    (size_t) stride_a, work.data(), this_nrc_x) ||
                !iqk_convert_iqk_quants_q80_r8(type_a, (int) ne00,
                    (const char *) a_up + (first_x + ix) * stride_a,
                    (size_t) stride_a, work.data() + tile_size, this_nrc_x)) {
                return false;
            }
            const int64_t t1 = prof ? ggml_time_us() : 0;
            mm.mul_mat_up_gate_nxm((int) ne00, work.data(), work.data() + tile_size,
                row_size, this_info, this_nrc_x, (int) ny, limit);
            if (prof) {
                ggml_moe_prof_add(GGML_MOE_PROF_REPACK, t1 - t0);
                ggml_moe_prof_add(GGML_MOE_PROF_GEMM, ggml_time_us() - t1);
            }
        }
        return true;
    }

    if (!MulMat::prepare(type_a, type_b, (int) ne00, mm)) {
        return false;
    }

    int nrc_x = (nx + nth - 1) / nth;
    const int first_x = ith * nrc_x;
    nrc_x = (int) std::max<long>(0, std::min<long>(nrc_x, nx - first_x));
    DataInfo info{c + first_x, (const char *) b, (size_t) nb1 / sizeof(float),
        (size_t) stride_b, 0, ne11, row_mapping, (size_t) nb2 / sizeof(float)};
    const int64_t t0 = ggml_moe_prof_enabled() ? ggml_time_us() : 0;
    mm.mul_mat_up_gate_nxm((int) ne00,
        (const char *) a_gate + first_x * stride_a,
        (const char *) a_up   + first_x * stride_a,
        (size_t) stride_a, info, nrc_x, (int) ny, limit);
    if (t0) {
        ggml_moe_prof_add(GGML_MOE_PROF_GEMM, ggml_time_us() - t0);
    }
    return true;
}

#else

extern "C" bool iqk_mul_mat(long, long, long, int, const void *, long,
        int, const void *, long, float *, long, int, int) { return false; }
extern "C" bool iqk_mul_mat_4d(long, long, long, long, long, long, long,
        long, long, long, long, long, long, int, const void *, long,
        int, const void *, long, float *, long, int, int) { return false; }
extern "C" bool iqk_mul_mat_moe(long, long, long, int, int, const void *, long,
        int, const void *, long, float *, long, long, const void *, int, int, int) { return false; }
extern "C" bool iqk_moe_fused_up_gate(long, long, long, int, int, const void *, const void *, long,
        int, const void *, long, float *, long, long, const void *, float, int, int, int) { return false; }
extern "C" bool iqk_mul_mat_moe_supported(long, int, int) { return false; }

#endif
