// Unit tests for the generic per-row metadata concept (ggml_row_meta_size()).
//
// A type with nonzero row metadata stores a few bytes once per row, ahead of the
// row's blocks, so a row is *not* just `ne[0]/blck_size` blocks. These tests pin
// the invariants that the rest of ggml relies on, and check that the metadata
// really is per row by round-tripping rows whose magnitudes differ by orders of
// magnitude - a single shared scale could not represent both.

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        printf("  FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// Every type ggml can construct a tensor of.
static std::vector<ggml_type> all_constructible_types() {
    std::vector<ggml_type> types;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type type = (ggml_type) i;
        if (ggml_type_name(type) == nullptr || ggml_blck_size(type) == 0) {
            continue; // hole in the enum
        }
        types.push_back(type);
    }
    return types;
}

// ggml_row_size() is exactly the metadata plus the blocks, for every type.
static void test_row_size_identity() {
    printf("test_row_size_identity\n");
    for (const ggml_type type : all_constructible_types()) {
        const int64_t blck = ggml_blck_size(type);
        for (const int64_t nblocks : {1, 2, 7, 64}) {
            const int64_t ne = nblocks * blck;
            const size_t expected = ggml_row_meta_size(type) + ggml_type_size(type) * (size_t) nblocks;
            check(ggml_row_size(type, ne) == expected,
                std::string("ggml_row_size(") + ggml_type_name(type) + ", " + std::to_string(ne) + ")");
        }
    }
}

// A freshly created tensor's strides and byte count must agree with ggml_row_size().
static void test_tensor_layout() {
    printf("test_tensor_layout\n");

    ggml_init_params params = { 32u*1024u*1024u, nullptr, /*no_alloc =*/ true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx);

    for (const ggml_type type : all_constructible_types()) {
        const int64_t blck = ggml_blck_size(type);
        const int64_t ne0  = 4 * blck;
        const int64_t ne1  = 3;
        const int64_t ne2  = 2;

        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        const std::string tag = ggml_type_name(type);

        const size_t row_size = ggml_row_size(type, ne0);
        check(t->nb[0] == ggml_type_size(type), tag + ": nb[0]");
        check(t->nb[1] == row_size,             tag + ": nb[1] == ggml_row_size()");
        check(t->nb[2] == row_size * ne1,       tag + ": nb[2]");
        check(t->nb[3] == row_size * ne1 * ne2, tag + ": nb[3]");
        check(ggml_nbytes(t) == row_size * ne1 * ne2, tag + ": ggml_nbytes()");
        check(ggml_is_contiguous(t),                  tag + ": ggml_is_contiguous()");
        check(ggml_is_contiguously_allocated(t),      tag + ": ggml_is_contiguously_allocated()");

        // a whole-row view keeps the row length and lands on a row boundary
        ggml_tensor * v = ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], t->nb[2]);
        check(v->nb[1] == row_size,        tag + ": view nb[1]");
        check(ggml_nbytes(v) == row_size * ne1, tag + ": view ggml_nbytes()");
        check(ggml_is_contiguous(v),       tag + ": view ggml_is_contiguous()");
    }

    ggml_free(ctx);
}

// The point of per-row metadata: each row carries its own scale, and rows are
// independent. Quantize rows whose magnitudes differ by orders of magnitude,
// then check that each row's bytes are identical to quantizing that row on its
// own - which can only hold if the metadata is written per row and the row
// stride accounts for it. Also check that every row still round-trips, which a
// single shared scale could not manage across this magnitude range.
static void test_per_row_scale_is_per_row() {
    printf("test_per_row_scale_is_per_row\n");

    bool saw_row_meta_type = false;

    for (const ggml_type type : all_constructible_types()) {
        if (ggml_row_meta_size(type) == 0 || !ggml_is_quantized(type)) {
            continue;
        }
        if (ggml_get_type_traits(type)->to_float == nullptr ||
            ggml_quantize_requires_imatrix(type)) {
            continue;
        }
        saw_row_meta_type = true;

        const std::string tag = ggml_type_name(type);
        const int64_t n_per_row = 4 * ggml_blck_size(type);
        const int64_t nrows     = 4;

        // Every row holds the same waveform scaled by a different power of two:
        // 2^-8 .. 2^16. IQ2_KS and IQ3_KS store their row scale as a half, and it
        // is roughly magnitude/16, so this stays well clear of both fp16 ends.
        // Powers of two keep the scaling exact, so a correct
        // per-row scale must produce *identical* block payloads for every row and
        // differing row metadata - which is precisely the layout under test.
        std::vector<float> src((size_t) n_per_row * nrows);
        for (int64_t r = 0; r < nrows; ++r) {
            const float mag = std::ldexp(1.0f, (int) (8 * r - 8));
            for (int64_t i = 0; i < n_per_row; ++i) {
                src[(size_t) r * n_per_row + i] = mag * std::sin(0.1f * (float) (i + 1));
            }
        }

        const size_t row_size = ggml_row_size(type, n_per_row);
        std::vector<uint8_t> q(row_size * nrows);

        ggml_quantize_init(type);
        const size_t written = ggml_quantize_chunk(type, src.data(), q.data(), 0, nrows, n_per_row, nullptr);
        check(written == row_size * nrows, tag + ": ggml_quantize_chunk() byte count");

        std::vector<uint8_t> q_one(row_size);
        std::vector<float>   dst((size_t) n_per_row * nrows);
        const ggml_to_float_t to_float = ggml_get_type_traits(type)->to_float;

        const size_t meta_size   = ggml_row_meta_size(type);
        const size_t blocks_size = row_size - meta_size;

        for (int64_t r = 0; r < nrows; ++r) {
            const float * row_src = src.data() + (size_t) r * n_per_row;
            const uint8_t * row_q = q.data() + (size_t) r * row_size;

            // quantizing the row on its own must produce exactly the same bytes
            ggml_quantize_chunk(type, row_src, q_one.data(), 0, 1, n_per_row, nullptr);
            check(memcmp(row_q, q_one.data(), row_size) == 0,
                tag + ": row " + std::to_string(r) + " differs from the same row quantized alone");

            if (r > 0) {
                const uint8_t * prev_q = q.data() + (size_t) (r - 1) * row_size;
                // same waveform, exactly scaled: the blocks must be identical...
                check(memcmp(row_q + meta_size, prev_q + meta_size, blocks_size) == 0,
                    tag + ": row " + std::to_string(r) + " block payload differs from row " +
                    std::to_string(r - 1) + ", so the scale is not purely per row");
                // ...and the whole magnitude difference must sit in the row metadata
                check(memcmp(row_q, prev_q, meta_size) != 0,
                    tag + ": row " + std::to_string(r) + " has the same row metadata as row " +
                    std::to_string(r - 1));
            }

            // and each row must decode from a pointer to the start of the row
            to_float(row_q, dst.data() + (size_t) r * n_per_row, n_per_row);

            double num = 0, den = 0;
            for (int64_t i = 0; i < n_per_row; ++i) {
                const double a = row_src[i];
                const double b = dst[(size_t) r * n_per_row + i];
                num += (a - b) * (a - b);
                den += a * a;
            }
            const double rel_rmse = std::sqrt(num / den);
            // A scale shared across rows would leave the small rows at O(1) here.
            check(rel_rmse < 0.5, tag + ": row " + std::to_string(r) +
                " relative rmse " + std::to_string(rel_rmse));
        }
    }

    check(saw_row_meta_type, "at least one type with nonzero row metadata exists");
}

int main() {
    test_row_size_identity();
    test_tensor_layout();
    test_per_row_scale_is_per_row();

    printf("%d checks failed\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
