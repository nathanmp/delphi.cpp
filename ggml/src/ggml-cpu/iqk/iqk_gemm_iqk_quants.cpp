// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT

#include "iqk_gemm_iqk_quants.h"

#include <cstdlib>
#include <cstring>

#if defined(IQK_IMPLEMENT)

namespace {

inline void prepare_scales_16(const __m256i& all_scales, __m256i * scales) {
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    scales[0] = MM256_SET1_M128I(l_scales);
    scales[1] = MM256_SET1_M128I(h_scales);
}

struct IQXKScales {
    IQXKScales(int8_t shift, int8_t min_val) : min(_mm256_set1_epi16(min_val)), eshift(_mm_set1_epi8(shift)) {}
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m128i scales8, const Q8& q8, __m256 * accm, __m256i * scales) const {
        auto scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, hshuff));
        process(i, d, extra, scales16, q8, accm, scales);
    }
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m256i scales16, const Q8& q8, __m256 * accm, __m256i * scales) const {
        auto extra128 = _mm_set1_epi16(extra);
        extra128 = _mm_cmpeq_epi8(_mm_and_si128(extra128, emask), emask);
        extra128 = _mm_and_si128(extra128, eshift);
        extra128 = _mm_shuffle_epi8(extra128, eshuffle);
        auto scales_s = _mm256_mullo_epi16(scales16, _mm256_add_epi16(min, _mm256_cvtepi8_epi16(extra128)));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i prod  = _mm256_madd_epi16(scales_s, q8.load_bsums(iy, i));
            accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
        }
        prepare_scales_16(scales16, scales);
    }

    const __m256i min;
    const __m128i eshift;
    const __m128i hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    const __m128i emask    = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
    const __m128i eshuffle = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
};

struct DequantizerIQ2K final : public BaseDequantizer<block_iq2_k> {
    DequantizerIQ2K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(5, -32), values(load_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales), q8, accm, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m256i load_values() {
        static const uint8_t kvalues_iq2nl[16] = {1, 19, 33, 49, 0, 0, 0, 0,  6, 24, 38, 54, 0, 0, 0, 0};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq2nl);
        return MM256_SET1_M128I(val128);
    }
    inline __m128i make_scales(const uint8_t * scales_l) const {
        uint64_t aux64; std::memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), maskl);
        return _mm_add_epi8(scl, m8);
    }

    Q2Bits bits;
    const IQXKScales iqxk;
    const __m256i values;
    const __m128i m8       = _mm_set1_epi8(-8);
    const __m128i maskl    = _mm_set1_epi8(0xf);
};

struct DequantizerIQ3K final : public BaseDequantizer<block_iq3_k> {
    DequantizerIQ3K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(4, -64), values(load_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales_h, x[i].scales_l), q8, accm, scales);
        hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        auto h256 = j == 0 ? hbits : _mm256_srli_epi16(hbits, 4);
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(h256, 2), hmask));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(h256, 1), hmask));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(h256, hmask));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(h256, 1), hmask));
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m256i load_values() {
        static const uint8_t kvalues_iq3nl[16] = {1, 24, 41, 54, 65, 77, 92, 111, 5, 28, 45, 58, 69, 81, 96, 115};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq3nl);
        return MM256_SET1_M128I(val128);
    }
    inline __m128i make_scales(uint16_t signs, const uint8_t * scales_l) const {
        uint64_t aux64; std::memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
        scl = _mm_add_epi8(_mm_slli_epi16(scl, 1), m1);
        const __m128i sc_signs = _mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi16(signs), sign_mask), sign_mask);
        const __m128i sch = _mm_shuffle_epi8(_mm_or_si128(sc_signs, _mm_set1_epi8(1)), hshuff);
        return _mm_sign_epi8(scl, sch);
    }

    Q2Bits bits;
    const IQXKScales iqxk;
    const __m256i values;
    __m256i hbits;
    const __m256i hmask  = _mm256_set1_epi8(4);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m128i sign_mask = _mm_set_epi64x(0x8080404020201010, 0x0808040402020101);
    const __m128i hshuff = _mm_loadu_si128((const __m128i*)k_shuff);
    constexpr static uint8_t k_shuff[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
};

struct DequantizerIQ4K final : public BaseDequantizer<block_iq4_k> {
    DequantizerIQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(); }
    template <typename Q8>
    inline void new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        auto scales8 = make_scales(x[i].scales_l, (const uint16_t *)x[i].scales_h);
        auto scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, hshuff));
        prepare_scales_16(scales16, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs, j);
        auto extra = x[i].extra >> 8*j;
        bits.values[0] = _mm256_shuffle_epi8(values[extra & 3], bits.values[0]); extra >>= 2;
        bits.values[1] = _mm256_shuffle_epi8(values[extra & 3], bits.values[1]); extra >>= 2;
        bits.values[2] = _mm256_shuffle_epi8(values[extra & 3], bits.values[2]); extra >>= 2;
        bits.values[3] = _mm256_shuffle_epi8(values[extra & 3], bits.values[3]);
    }
    __m128i make_scales(const uint8_t * scales_l, const uint16_t * scales_h) const {
        uint64_t aux64;
        memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), maskl);
        const uint32_t aux32 = scales_h[0] | (scales_h[1] << 16);
        auto aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), maskh);
        auto sch = _mm_shuffle_epi8(aux, hshuff);
        return _mm_add_epi8(_mm_or_si128(scl, sch), m32);
    }
    void load_values() {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET1_M128I(v1);
        values[1] = MM256_SET_M128I(v1, v2);
        values[2] = MM256_SET_M128I(v2, v1);
        values[3] = MM256_SET1_M128I(v2);
    }

    Q4Bits bits;
    const __m128i maskl    = _mm_set1_epi8(0xf);
    const __m128i maskh    = _mm_set1_epi8(0x30);
    const __m128i m32      = _mm_set1_epi8(-32);
    const __m128i hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    __m256i values[4];
};

struct DequantizerIQ5K final : public BaseDequantizer<block_iq5_k> {
    DequantizerIQ5K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(2, 0) { load_values(values); }
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales_l, (const uint16_t *)x[i].scales_h), q8, accm, scales);
        hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        auto h = j == 0 ? hbits : _mm256_srli_epi16(hbits, 4);
        for (int k = 0; k < 4; ++k) {
            auto qh = _mm256_and_si256(_mm256_slli_epi16(h, 7-k), mh);
            auto q5vl = _mm256_or_si256(bits.values[k], qh);
            auto q5vh = _mm256_or_si256(bits.values[k], _mm256_xor_si256(qh, mh));
            bits.values[k] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
        }
    }
    __m128i make_scales(const uint8_t * scales_l, const uint16_t * scales_h) const {
        uint64_t aux64;
        memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), maskl);
        const uint32_t aux32 = scales_h[0] | (scales_h[1] << 16);
        auto aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), maskh);
        auto sch = _mm_shuffle_epi8(aux, iqxk.hshuff);
        return _mm_add_epi8(_mm_or_si128(scl, sch), m32);
    }
    static void load_values(__m256i * values) {
        auto values128_1 = _mm_loadu_si128((const __m128i *)iq5nl_values + 0);
        auto values128_2 = _mm_loadu_si128((const __m128i *)iq5nl_values + 1);
        values[0] = MM256_SET1_M128I(values128_1);
        values[1] = MM256_SET1_M128I(values128_2);
    }

    Q4Bits bits;
    const IQXKScales iqxk;
    __m256i hbits;
    __m256i values[2];
    const __m128i maskl    = _mm_set1_epi8(0xf);
    const __m128i maskh    = _mm_set1_epi8(0x30);
    const __m128i m32      = _mm_set1_epi8(-32);
    const __m256i mh       = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing
};

struct DequantizerIQ6K final : public BaseDequantizer<block_iq6_k> {
    DequantizerIQ6K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(1, 0) { load_values(values); }
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        auto scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
        auto scales16 = _mm256_cvtepi8_epi16(scales8);
        iqxk.process(i, d, x[i].extra, scales16, q8, accm, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        auto hbits = _mm256_loadu_si256((const __m256i *)x[i].qh + j);
        for (int k = 0; k < 4; ++k) {
            bits.values[k] = make_one(bits.values[k], hbits);
            hbits = _mm256_srli_epi16(hbits, 2);
        }
    }
    inline __m256i make_one(__m256i l, __m256i hbits) const {
        auto mask4 = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, mh3), mh3);
        auto h1 = _mm256_andnot_si256(mask4, hbits);
        auto mask2 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh1), mh1);
        auto mask3 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh2), mh2);
        auto mask1 = _mm256_andnot_si256(_mm256_or_si256(mask4, _mm256_or_si256(mask2, mask3)), _mm256_set1_epi8(-1)); // 0xff;
        return _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(mask1, _mm256_shuffle_epi8(values[0], l)),
                                               _mm256_and_si256(mask2, _mm256_shuffle_epi8(values[1], l))),
                               _mm256_or_si256(_mm256_and_si256(mask3, _mm256_shuffle_epi8(values[2], l)),
                                               _mm256_and_si256(mask4, _mm256_shuffle_epi8(values[3], l))));
    }
    static void load_values(__m256i * values) {
        for (int k = 0; k < 4; ++k) {
            auto values128 = _mm_loadu_si128((const __m128i *)iq6nl_values + k);
            values[k] = MM256_SET1_M128I(values128);
        }
    }

    Q4Bits bits;
    const IQXKScales iqxk;
    __m256i values[4];
    const __m256i mh1 = _mm256_set1_epi8(1);
    const __m256i mh2 = _mm256_set1_epi8(2);
    const __m256i mh3 = _mm256_set1_epi8(3);
    const __m256i mh       = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing
};

//
// IQ*_KS. These types keep their scale once per row rather than once per block,
// which BaseDequantizer's per_row_scale/is_f16 template arguments handle: new_row()
// reads the row scale and points `x` at the first block. They also use the
// `mul_mat_qX_K_q8_K_T` schedule (an 8-scale block returned by new_block) rather
// than IQ*_K's 16-scale one.
//

struct DequantizerIQ2KS final : public BaseDequantizer<block_iq2_ks, true, true> {
    DequantizerIQ2KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accm) {
        auto scales128 = make_scales(x[i].scales, x[i].extra >> 8);
        auto shifts = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi8(x[i].extra), hmask), hmask), m5);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_cvtepi8_epi16(_mm_add_epi8(m32, shifts)));
        s8k.accum_mins(scales_s, q8, i, d, accm);
        return MM256_SET1_M128I(scales128);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m256i load_values() {
        static const uint8_t kvalues_iq2nl[16] = {1, 19, 33, 49, 0, 0, 0, 0,  6, 24, 38, 54, 0, 0, 0, 0};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq2nl);
        return MM256_SET1_M128I(val128);
    }
    inline __m128i make_scales(const uint8_t * scales_l, uint8_t scales_h) const {
        const uint16_t * scales = (const uint16_t *)scales_l;
        uint32_t aux32 = scales[0] | (uint32_t(scales[1]) << 16);
        auto scl = _mm_srlv_epi32(_mm_set1_epi32(aux32), shift);
        scl = _mm_and_si128(_mm_shuffle_epi8(scl, shuffle), _mm_set1_epi8(0xf));
        auto sch = _mm_set1_epi8(scales_h);
        sch = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(sch, hmask), _mm_setzero_si128()), m16);
        return _mm_cvtepi8_epi16(_mm_add_epi8(scl, sch));
    }
    Q2Bits bits;
    Scales8KBase s8k;

    const __m256i values;
    const __m128i m16 = _mm_set1_epi8(-16);
    const __m128i m5  = _mm_set1_epi8(5);
    const __m128i m32 = _mm_set1_epi8(-32);
    const __m128i hmask = _mm_set1_epi64x(0x8040201008040201);
    const __m128i shuffle = _mm_set1_epi64x(0x0703060205010400);
    const __m128i shift = _mm_set_epi32(0, 0, 4, 0);
};

struct DequantizerIQ3KS final : public BaseDequantizer<block_iq3_ks, true, true> {
    DequantizerIQ3KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        GGML_UNUSED(q8);
        GGML_UNUSED(accd);
        uint32_t aux32; std::memcpy(&aux32, x[i].scales, 4);
        auto scl = _mm_cvtepi8_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(aux32), _mm_set_epi32(0, 0, 4, 0)), _mm_set1_epi8(0xf)));
        auto sch = _mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(x[i].extra), mask), mask);
        auto scales128 = _mm_add_epi16(scl, _mm_and_si128(sch, _mm_set1_epi16(16)));
        scales128 = _mm_sub_epi16(scales128, _mm_set1_epi16(16));
        return MM256_SET1_M128I(scales128);
    }
    inline void prepare(int i, int j) {
        uint8_t extra = x[i].extra >> (8 + 4*j);
        hbits = j == 0 ? _mm256_loadu_si256((const __m256i *)x[i].qh) : _mm256_srli_epi16(hbits, 4);
        bits.prepare(x[i].qs, j);
        bits.values[0] = _mm256_add_epi8(_mm256_set1_epi8((extra << 3) & 8), _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh)));
        bits.values[1] = _mm256_add_epi8(_mm256_set1_epi8((extra << 2) & 8), _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), mh)));
        bits.values[2] = _mm256_add_epi8(_mm256_set1_epi8((extra << 1) & 8), _mm256_or_si256(bits.values[2], _mm256_and_si256(hbits, mh)));
        bits.values[3] = _mm256_add_epi8(_mm256_set1_epi8((extra << 0) & 8), _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(hbits, 1), mh)));
        for (int k = 0; k < 4; ++k) bits.values[k] = _mm256_shuffle_epi8(values, bits.values[k]);
    }
    static inline __m256i load_values() {
        auto v = _mm_loadu_si128((const __m128i *)iq3nl_values);
        return MM256_SET1_M128I(v);
    }

    Q2Bits bits;
    __m256i hbits;
    const __m256i values;
    const __m256i mh   = _mm256_set1_epi8(4);
    const __m128i mask = _mm_setr_epi16(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
};

struct DequantizerIQ4KS final : public BaseDequantizer<block_iq4_ks, true> {
    DequantizerIQ4KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(); }
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        GGML_UNUSED(q8);
        GGML_UNUSED(accd);
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        return MM256_SET1_M128I(scales128);
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values[x[i].scales[4*j+0] & 1], bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values[x[i].scales[4*j+1] & 1], bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values[x[i].scales[4*j+2] & 1], bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values[x[i].scales[4*j+3] & 1], bits.values[3]);
    }
    void load_values() {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET1_M128I(v1);
        values[1] = MM256_SET1_M128I(v2);
    }

    Q4Bits bits;
    __m256i values[2];
    const __m128i mask = _mm_set1_epi16(254);
    const __m128i m127 = _mm_set1_epi16(-127);
};

// Deviation from ik_llama.cpp: its AVX2 IQ5_KS dequantizer decodes through a table
// biased by +128 and multiplies with _mm256_maddubs_epi16, whose int16 pairwise sum
// saturates for 5-bit values that large (up to 2*249*127, against a 32767 limit).
// That costs several percent on random data. We decode through the signed
// iq5nl_values table and use the sign-trick multiply (as DequantizerIQ5K does), so
// the products stay inside int16. The +2 "shifted values" offset is still folded
// into a bsums term, only without the -128 debiasing ik needs.
struct DequantizerIQ5KS final : public BaseDequantizer<block_iq5_ks, true> {
    DequantizerIQ5KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(values); }
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        auto shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, m1), m1), m2);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        auto scales_s = _mm_mullo_epi16(scales128, shifts);
        s8k.accum_mins(scales_s, q8, i, d, accd);
        return MM256_SET1_M128I(scales128);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        auto h = j == 0 ? hbits : _mm256_srli_epi16(hbits, 4);
        for (int k = 0; k < 4; ++k) {
            auto qh = _mm256_and_si256(_mm256_slli_epi16(h, 7-k), mh);
            auto q5vl = _mm256_or_si256(bits.values[k], qh);
            auto q5vh = _mm256_or_si256(bits.values[k], _mm256_xor_si256(qh, mh));
            bits.values[k] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
        }
    }
    static void load_values(__m256i * values) {
        auto values128_1 = _mm_loadu_si128((const __m128i *)iq5nl_values + 0);
        auto values128_2 = _mm_loadu_si128((const __m128i *)iq5nl_values + 1);
        values[0] = MM256_SET1_M128I(values128_1);
        values[1] = MM256_SET1_M128I(values128_2);
    }

    Q4Bits bits;
    Scales8KBase s8k;
    __m256i hbits;
    __m256i values[2];
    const __m256i mh   = _mm256_set1_epi8(-128); // 0x80, spelled so it does not overflow int8
    const __m128i mask = _mm_set1_epi16(254);
    const __m128i m127 = _mm_set1_epi16(-127);
    const __m128i m1   = _mm_set1_epi16(1);
    const __m128i m2   = _mm_set1_epi16(2);
};

inline __m256i get_scale_shuffle_16(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

inline void set_scales_16(const __m256i& all_scales, __m256i * scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(0));
    scales[1] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(1));
    scales[2] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(2));
    scales[3] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(3));
}

inline __m256i get_scale_shuffle_8(int i) {
    return _mm256_set1_epi16((2*i) | ((2*i+1) << 8));
}

inline void set_scales_8(const __m256i& all_scales, int j, __m256i * scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+0));
    scales[1] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+1));
    scales[2] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+2));
    scales[3] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+3));
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qY_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Q8<nrc_y> q8(info);

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    Dequantizer deq(vx, bx);

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {

            deq.new_block(i, q8, accd, all_scales);

            __m256i sumi[nrc_y];

            for (int j = 0; j < QK_K/128; ++j) {
                deq.prepare(i, j);
                set_scales_16(all_scales[j], scales);
                if constexpr (std::is_same_v<Dequantizer, DequantizerIQ4K> ||
                              std::is_same_v<Dequantizer, DequantizerIQ5K> ||
                              std::is_same_v<Dequantizer, DequantizerIQ6K>) {
                    multiply_add_avx2(deq.bits, scales, j, i, q8, sumi);
                } else {
                    multiply_add(deq.bits, scales, j, i, q8, sumi);
                }
            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8.scale(iy, i)), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }

    }

}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(info);

    Dequantizer deq(vx, bx);

    __m256  accd[nrc_y];
    __m256i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            auto all_scales = deq.new_block(i, q8, accd);

            __m256i sumi[nrc_y];

            for (int j = 0; j < QK_K/128; ++j) {

                deq.prepare(i, j);

                set_scales_8(all_scales, j, scales);

                if constexpr (std::is_same_v<Dequantizer, DequantizerIQ3KS> ||
                              std::is_same_v<Dequantizer, DequantizerIQ4KS> ||
                              std::is_same_v<Dequantizer, DequantizerIQ5KS>) {
                    multiply_add_avx2(deq.bits, scales, j, i, q8, sumi);
                } else {
                    multiply_add(deq.bits, scales, j, i, q8, sumi);
                }

            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8.scale(iy, i));
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }

    }
}

// Two decoded groups leave more registers for the activation rows at small speculative widths.
template <int nrc_y>
static void mul_mat_iq4_ks_q8_K_pairs(int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;
    Q8<nrc_y> q8(info);
    DequantizerIQ4KS deq(vx, bx);

    for (int ix = 0; ix < nrc_x; ++ix) {
        deq.new_row(ix);
        __m256 accd[nrc_y];
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const auto all_scales = deq.new_block(i, q8, accd);
            __m256i sumi[nrc_y];

            for (int j = 0; j < QK_K/64; ++j) {
                __m256i values[2], scales[2];
                for (int k = 0; k < 2; ++k) {
                    const int group = 2*j + k;
                    const auto bits = deq.bits.dequant16(deq.x[i].qs + 16*group);
                    values[k] = _mm256_shuffle_epi8(deq.values[deq.x[i].scales[group] & 1], bits);
                    scales[k] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(group));
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    __m256i prod[2];
                    for (int k = 0; k < 2; ++k) {
                        prod[k] = _mm256_madd_epi16(scales[k], _mm256_maddubs_epi16(
                            _mm256_abs_epi8(values[k]), _mm256_sign_epi8(q8.load_quants(iy, i, 2*j + k), values[k])));
                    }
                    const auto sum = _mm256_add_epi32(prod[0], prod[1]);
                    sumi[iy] = j == 0 ? sum : _mm256_add_epi32(sumi[iy], sum);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8.scale(iy, i)), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) info.store(ix, iy, hsum_float_8(accd[iy]));
    }
}

void iqk_convert_iq2_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    constexpr int k_nr = 16;
    using block_q8_k_r = block_q8_k_r16;
#else
    constexpr int k_nr = 8;
    using block_q8_k_r = block_q8_k_r8;
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%k_nr == 0);

    int nb = n/QK_K;

    const block_iq2_k * x8[k_nr];

    block_q8_k_r * y = (block_q8_k_r *)vy;

    __m256i values;
    {
        auto v = _mm_loadl_epi64((const __m128i *)iq2nl_values);
        values = MM256_SET1_M128I(v);
    }

    __m256i  xv[8];
    uint32_t block[8];

    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);

    union { __m256i vec; int16_t val[16]; } helper;

    auto ml = _mm256_set1_epi8(0x03);

    for (int ix = 0; ix < nrc_x; ix += k_nr) {
        for (int k = 0; k < k_nr; ++k) x8[k] = (const block_iq2_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < k_nr; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                uint64_t aux64; std::memcpy(&aux64, x8[k][i].scales, 8);
                auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
                scl = _mm_add_epi8(scl, _mm_set1_epi8(-8));
                helper.vec = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scl, scale_shuffle));
                auto extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[4*i128+0] = _mm256_and_si256(bits, ml);
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    auto shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x02) << 1), _mm_set1_epi8((extra & 0x01) << 2));
                    auto shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x08) >> 1), _mm_set1_epi8((extra & 0x04) >> 0));
                    auto shift3 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x20) >> 3), _mm_set1_epi8((extra & 0x10) >> 2));
                    auto shift4 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x80) >> 5), _mm_set1_epi8((extra & 0x40) >> 4));
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], shift1);
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], shift2);
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], shift3);
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], shift4);
                    xv[4*i128+0] = _mm256_shuffle_epi8(values, xv[4*i128+0]);
                    xv[4*i128+1] = _mm256_shuffle_epi8(values, xv[4*i128+1]);
                    xv[4*i128+2] = _mm256_shuffle_epi8(values, xv[4*i128+2]);
                    xv[4*i128+3] = _mm256_shuffle_epi8(values, xv[4*i128+3]);
                    extra >>= 8;
                }
                float dnew = convert_to_q8_k_r8<k_nr>(k, 1.f/120, xv, helper.val, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
#ifdef HAVE_FANCY_SIMD
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[i].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[i].qs + l, v);
            }
#endif
        }
        y += nb;
    }
}

void iqk_convert_iq3_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    constexpr int k_nr = 16;
    using block_q8_k_r = block_q8_k_r16;
#else
    constexpr int k_nr = 8;
    using block_q8_k_r = block_q8_k_r8;
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%k_nr == 0);

    int nb = n/QK_K;

    const block_iq3_k * x8[k_nr];

    block_q8_k_r * y = (block_q8_k_r *)vy;

    __m256i values;
    {
        auto v = _mm_loadu_si128((const __m128i *)iq3nl_values);
        values = MM256_SET1_M128I(v);
    }

    __m256i  xv[8];
    uint32_t block[8];

    constexpr static uint8_t k_shuff[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    const __m128i sign_mask = _mm_set_epi64x(0x8080404020201010, 0x0808040402020101);
    const __m128i hshuff = _mm_loadu_si128((const __m128i*)k_shuff);
    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);

    union { __m256i vec; int16_t val[16]; } helper;

    auto ml = _mm256_set1_epi8(0x03);
    auto hmask  = _mm256_set1_epi8(4);

    for (int ix = 0; ix < nrc_x; ix += k_nr) {
        for (int k = 0; k < k_nr; ++k) x8[k] = (const block_iq3_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < k_nr; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                uint64_t aux64; std::memcpy(&aux64, x8[k][i].scales_l, 8);
                auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
                scl = _mm_add_epi8(_mm_slli_epi16(scl, 1), _mm_set1_epi8(1));
                auto sc_signs = _mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi16(x8[k][i].scales_h), sign_mask), sign_mask);
                auto sch   = _mm_shuffle_epi8(_mm_or_si128(sc_signs, _mm_set1_epi8(1)), hshuff);
                helper.vec = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(_mm_sign_epi8(scl, sch), scale_shuffle));
                auto extra = x8[k][i].extra;
                auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int i128 = 0; i128 < 2; ++i128) {
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[4*i128+0] = _mm256_and_si256(bits, ml);
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    xv[4*i128+0] = _mm256_or_si256(xv[4*i128+0], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), hmask));
                    xv[4*i128+1] = _mm256_or_si256(xv[4*i128+1], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), hmask));
                    xv[4*i128+2] = _mm256_or_si256(xv[4*i128+2], _mm256_and_si256(hbits, hmask));
                    xv[4*i128+3] = _mm256_or_si256(xv[4*i128+3], _mm256_and_si256(_mm256_srli_epi16(hbits, 1), hmask));
                    auto shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x02) << 2), _mm_set1_epi8((extra & 0x01) << 3));
                    auto shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x08) << 0), _mm_set1_epi8((extra & 0x04) << 1));
                    auto shift3 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x20) >> 2), _mm_set1_epi8((extra & 0x10) >> 1));
                    auto shift4 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x80) >> 4), _mm_set1_epi8((extra & 0x40) >> 3));
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], shift1);
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], shift2);
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], shift3);
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], shift4);
                    xv[4*i128+0] = _mm256_shuffle_epi8(values, xv[4*i128+0]);
                    xv[4*i128+1] = _mm256_shuffle_epi8(values, xv[4*i128+1]);
                    xv[4*i128+2] = _mm256_shuffle_epi8(values, xv[4*i128+2]);
                    xv[4*i128+3] = _mm256_shuffle_epi8(values, xv[4*i128+3]);
                    hbits = _mm256_srli_epi16(hbits, 4);
                    extra >>= 8;
                }
                float dnew = convert_to_q8_k_r8<k_nr>(k, 1.f/127, xv, helper.val, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
#ifdef HAVE_FANCY_SIMD
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[i].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[i].qs + l, v);
            }
#endif
        }
        y += nb;
    }
}

void iqk_convert_iq4_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    constexpr int k_nr = 16;
    using block_q8_k_r = block_q8_k_r16;
#else
    constexpr int k_nr = 8;
    using block_q8_k_r = block_q8_k_r8;
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%k_nr == 0);

    int nb = n/QK_K;

    const block_iq4_k * x8[k_nr];

    block_q8_k_r * y = (block_q8_k_r *)vy;

    __m256i values[4];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET1_M128I(v1);
        values[1] = MM256_SET_M128I(v1, v2);
        values[2] = MM256_SET_M128I(v2, v1);
        values[3] = MM256_SET1_M128I(v2);
    }

    __m256i  xv[8];
    uint32_t block[8];
    int16_t  ls[16];

    //auto hshuff = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);

    //union { __m256i vec; int16_t val[16]; } helper;

    for (int ix = 0; ix < nrc_x; ix += k_nr) {
        for (int k = 0; k < k_nr; ++k) x8[k] = (const block_iq4_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < k_nr; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                //uint64_t aux64;
                //memcpy(&aux64, x8[k][i].scales_l, 8);
                //auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
                //const uint32_t aux32 = *(const uint32_t *)x8[k][i].scales_h;
                //auto aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), _mm_set1_epi8(0x30));
                //auto sch = _mm_shuffle_epi8(aux, hshuff);
                //aux = _mm_add_epi8(_mm_or_si128(scl, sch), _mm_set1_epi8(-32));
                //helper.vec = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(aux, hshuff));
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint8_t sh = x8[k][i].scales_h[ib32/2] >> 4*(ib32%2);
                    ls[2*ib32+0] = ((x8[k][i].scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
                    ls[2*ib32+1] = ((x8[k][i].scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
                    auto bits = _mm_loadu_si128((const __m128i *)x8[k][i].qs+ib32);
                    xv[ib32]  = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(bits, 4), bits), _mm256_set1_epi8(0xf));
                    xv[ib32]  = _mm256_shuffle_epi8(values[extra & 3], xv[ib32]); extra >>= 2;
                }
                //float dnew = convert_to_q8_k_r8(k, 1.f/127, xv, helper.val, block, y[i].qs);
                float dnew = convert_to_q8_k_r8<k_nr>(k, 1.f/127, xv, ls, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
#ifdef HAVE_FANCY_SIMD
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[i].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[i].qs + l, v);
            }
#endif
        }
        y += nb;
    }
}

void iqk_convert_iq5_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    constexpr int k_nr = 16;
    using block_q8_k_r = block_q8_k_r16;
#else
    constexpr int k_nr = 8;
    using block_q8_k_r = block_q8_k_r8;
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%k_nr == 0);

    int nb = n/QK_K;

    const block_iq5_k * x8[k_nr];

    block_q8_k_r * y = (block_q8_k_r *)vy;

    __m256i values[2];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq5nl_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq5nl_values+1);
        values[0] = MM256_SET1_M128I(v1);
        values[1] = MM256_SET1_M128I(v2);
    }

    __m256i  xv[8];
    uint32_t block[8];
    int16_t  ls[16];

    auto mh = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing

    for (int ix = 0; ix < nrc_x; ix += k_nr) {
        for (int k = 0; k < k_nr; ++k) x8[k] = (const block_iq5_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < k_nr; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    ls[4*ib64+0] = ((x8[k][i].scales_l[2*ib64+0] & 0xf) | ((x8[k][i].scales_h[ib64] << 4) & 0x30)) - 32;
                    ls[4*ib64+1] = ((x8[k][i].scales_l[2*ib64+0] >>  4) | ((x8[k][i].scales_h[ib64] << 2) & 0x30)) - 32;
                    ls[4*ib64+2] = ((x8[k][i].scales_l[2*ib64+1] & 0xf) | ((x8[k][i].scales_h[ib64] >> 0) & 0x30)) - 32;
                    ls[4*ib64+3] = ((x8[k][i].scales_l[2*ib64+1] >>  4) | ((x8[k][i].scales_h[ib64] >> 2) & 0x30)) - 32;
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+ib64);
                    xv[2*ib64+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[2*ib64+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    auto qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 7), mh);
                    auto q5vl = _mm256_or_si256(xv[2*ib64+0], qh);
                    auto q5vh = _mm256_or_si256(xv[2*ib64+0], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+0] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 6), mh);
                    q5vl = _mm256_or_si256(xv[2*ib64+1], qh);
                    q5vh = _mm256_or_si256(xv[2*ib64+1], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+1] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    auto shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 2) << 0), _mm_set1_epi8((extra & 1) << 1));
                    auto shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 8) >> 2), _mm_set1_epi8((extra & 4) >> 1));
                    xv[2*ib64+0] = _mm256_add_epi8(xv[2*ib64+0], shift1);
                    xv[2*ib64+1] = _mm256_add_epi8(xv[2*ib64+1], shift2);
                    hbits = _mm256_srli_epi16(hbits, 2);
                    extra >>= 4;
                }
                float dnew = convert_to_q8_k_r8<k_nr>(k, 1.f/127, xv, ls, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
#ifdef HAVE_FANCY_SIMD
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[i].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[i].qs + l, v);
            }
#endif
        }
        y += nb;
    }
}

void iqk_convert_iq6_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    constexpr int k_nr = 16;
    using block_q8_k_r = block_q8_k_r16;
#else
    constexpr int k_nr = 8;
    using block_q8_k_r = block_q8_k_r8;
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%k_nr == 0);

    int nb = n/QK_K;

    const block_iq6_k * x8[k_nr];

    block_q8_k_r * y = (block_q8_k_r *)vy;

    __m256i values[4];
    for (int k = 0; k < 4; ++k) {
        auto values128 = _mm_loadu_si128((const __m128i *)iq6nl_values + k);
        values[k] = MM256_SET1_M128I(values128);
    }

    __m256i  xv[8];
    uint32_t block[8];

    union { __m256i vec; int16_t val[16]; } helper;

    auto mh1 = _mm256_set1_epi8(1);
    auto mh2 = _mm256_set1_epi8(2);
    auto mh3 = _mm256_set1_epi8(3);

    auto make_one = [&values, &mh1, &mh2, &mh3] (__m256i l, __m256i hbits) {
        auto mask4 = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, mh3), mh3);
        auto h1 = _mm256_andnot_si256(mask4, hbits);
        auto mask2 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh1), mh1);
        auto mask3 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh2), mh2);
        auto mask1 = _mm256_andnot_si256(_mm256_or_si256(mask4, _mm256_or_si256(mask2, mask3)), _mm256_set1_epi8(-1)); // 0xff;
        return _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(mask1, _mm256_shuffle_epi8(values[0], l)),
                                               _mm256_and_si256(mask2, _mm256_shuffle_epi8(values[1], l))),
                               _mm256_or_si256(_mm256_and_si256(mask3, _mm256_shuffle_epi8(values[2], l)),
                                               _mm256_and_si256(mask4, _mm256_shuffle_epi8(values[3], l))));
    };

    for (int ix = 0; ix < nrc_x; ix += k_nr) {
        for (int k = 0; k < k_nr; ++k) x8[k] = (const block_iq6_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < k_nr; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                helper.vec = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)x8[k][i].scales));
                auto extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh+i128);
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+2*i128+0);
                    xv[4*i128+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+2*i128+1);
                    xv[4*i128+2] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    for (int k = 0; k < 4; ++k) {
                        xv[4*i128+k] = make_one(xv[4*i128+k], hbits);
                        hbits = _mm256_srli_epi16(hbits, 2);
                    }
                    auto shift1 = MM256_SET_M128I(_mm_set1_epi8((extra >> 1) & 1), _mm_set1_epi8((extra >> 0) & 1));
                    auto shift2 = MM256_SET_M128I(_mm_set1_epi8((extra >> 3) & 1), _mm_set1_epi8((extra >> 2) & 1));
                    auto shift3 = MM256_SET_M128I(_mm_set1_epi8((extra >> 5) & 1), _mm_set1_epi8((extra >> 4) & 1));
                    auto shift4 = MM256_SET_M128I(_mm_set1_epi8((extra >> 7) & 1), _mm_set1_epi8((extra >> 6) & 1));
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], shift1);
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], shift2);
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], shift3);
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], shift4);
                    extra >>= 8;
                }
                float dnew = convert_to_q8_k_r8<k_nr>(k, 1.f/127, xv, helper.val, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
#ifdef HAVE_FANCY_SIMD
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[i].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[i].qs + l, v);
            }
#endif
        }
        y += nb;
    }
}


void iqk_convert_iq2_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    constexpr int k_nr = 16;
    using block_q8_k_r = block_q8_k_r16;
#else
    constexpr int k_nr = 8;
    using block_q8_k_r = block_q8_k_r8;
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%k_nr == 0);

    int nb = n/QK_K;

    const block_iq2_ks * x8[k_nr];

    block_q8_k_r * y = (block_q8_k_r *)vy;

    __m256i values;
    {
        auto v = _mm_loadl_epi64((const __m128i *)iq2nl_values);
        values = MM256_SET1_M128I(v);
    }

    ggml_half dh[k_nr];
    float     dnew[k_nr];
    uint32_t  block[8];
    int16_t   ls[16];

    __m256i  xv[8];

    auto ml = _mm256_set1_epi8(0x03);

    for (int ix = 0; ix < nrc_x; ix += k_nr) {
        for (int k = 0; k < k_nr; ++k) {
            const ggml_half * dptr = (const ggml_half *)((const char *)vx + (ix+k)*bx);
            dh[k] = dptr[0];
            x8[k] = (const block_iq2_ks *)(dptr + 1);
        }
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < k_nr; ++k) {
                auto extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    ls[8*i128+0] = ls[8*i128+1] = ((x8[k][i].scales[2*i128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16;
                    ls[8*i128+2] = ls[8*i128+3] = ((x8[k][i].scales[2*i128+0] >>  4) | ((extra >> 5) & 0x10)) - 16;
                    ls[8*i128+4] = ls[8*i128+5] = ((x8[k][i].scales[2*i128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16;
                    ls[8*i128+6] = ls[8*i128+7] = ((x8[k][i].scales[2*i128+1] >>  4) | ((extra >> 7) & 0x10)) - 16;
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[4*i128+0] = _mm256_and_si256(bits, ml);
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], _mm256_set1_epi8((extra << 2) & 0x04));
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], _mm256_set1_epi8((extra << 1) & 0x04));
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], _mm256_set1_epi8((extra >> 0) & 0x04));
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], _mm256_set1_epi8((extra >> 1) & 0x04));
                    xv[4*i128+0] = _mm256_shuffle_epi8(values, xv[4*i128+0]);
                    xv[4*i128+1] = _mm256_shuffle_epi8(values, xv[4*i128+1]);
                    xv[4*i128+2] = _mm256_shuffle_epi8(values, xv[4*i128+2]);
                    xv[4*i128+3] = _mm256_shuffle_epi8(values, xv[4*i128+3]);
                    extra >>= 4;
                }
                dnew[k] = convert_to_q8_k_r8<k_nr>(k, 1.f/125, xv, ls, block, y[i].qs);
            }
#ifdef HAVE_FANCY_SIMD
            auto vd = _mm512_mul_ps(_mm512_loadu_ps(dnew), _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)dh)));
            _mm256_storeu_si256((__m256i *)y[i].d, _mm512_cvtps_ph(vd, _MM_ROUND_NEAREST));
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[i].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[i].qs + l, v);
            }
#else
            auto vd = _mm256_mul_ps(_mm256_loadu_ps(dnew), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh)));
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(vd, _MM_ROUND_NEAREST));
#endif
        }
        y += nb;
    }
}

void iqk_convert_iq3_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    constexpr int k_nr = 16;
    using block_q8_k_r = block_q8_k_r16;
#else
    constexpr int k_nr = 8;
    using block_q8_k_r = block_q8_k_r8;
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%k_nr == 0);

    int nb = n/QK_K;

    const block_iq3_ks * x8[k_nr];

    block_q8_k_r * y = (block_q8_k_r *)vy;

    __m256i values;
    {
        auto v = _mm_loadu_si128((const __m128i *)iq3nl_values);
        values = MM256_SET1_M128I(v);
    }

    ggml_half drow[k_nr];
    float dnew[k_nr];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += k_nr) {
        for (int k = 0; k < k_nr; ++k) {
            const ggml_half * dptr = (const ggml_half *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq3_ks *)(dptr + 1);
        }
#ifdef HAVE_FANCY_SIMD
        auto vd = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)drow));
#else
        auto vd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)drow));
#endif
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < k_nr; ++k) {
                auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                auto extra = x8[k][i].extra;
                uint8_t extra_v = extra >> 8;
                for (int j = 0; j < 4; ++j) {
                    ls[2*j+0] = ls[2*j+1] = ((x8[k][i].scales[j] & 0xf) | ((extra << 4) & 0x10)) - 16;
                    ls[2*j+8] = ls[2*j+9] = ((x8[k][i].scales[j] >>  4) | ((extra << 0) & 0x10)) - 16;
                    extra >>= 1;
                }
                for (int i128 = 0; i128 < QK_K/128; ++i128) {
                    auto lbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs + i128);
                    for (int j = 0; j < 4; ++j) {
                        xv[4*i128+j] = _mm256_or_si256(_mm256_and_si256(lbits, _mm256_set1_epi8(3)), _mm256_and_si256(_mm256_slli_epi16(hbits, 2), _mm256_set1_epi8(4)));
                        xv[4*i128+j] = _mm256_add_epi8(xv[4*i128+j], _mm256_set1_epi8((extra_v & 1) << 3));
                        xv[4*i128+j] = _mm256_shuffle_epi8(values, xv[4*i128+j]);
                        extra_v >>= 1;
                        lbits = _mm256_srli_epi16(lbits, 2);
                        hbits = _mm256_srli_epi16(hbits, 1);
                    }
                }
                dnew[k] = convert_to_q8_k_r8<k_nr>(k, 1.f/127, xv, ls, block, y[i].qs);
            }
#ifdef HAVE_FANCY_SIMD
            _mm256_storeu_si256((__m256i *)y[i].d, _mm512_cvtps_ph(_mm512_mul_ps(vd, _mm512_loadu_ps(dnew)), _MM_ROUND_NEAREST));
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[i].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[i].qs + l, v);
            }
#else
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
#endif
        }
        y += nb;
    }
}

void iqk_convert_iq4_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    constexpr int k_nr = 16;
    using block_q8_k_r = block_q8_k_r16;
#else
    constexpr int k_nr = 8;
    using block_q8_k_r = block_q8_k_r8;
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%k_nr == 0);

    int nb = n/QK_K;

    const block_iq4_ks * x8[k_nr];

    block_q8_k_r * y = (block_q8_k_r *)vy;

    __m256i values[2];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET1_M128I(v1);
        values[1] = MM256_SET1_M128I(v2);
    }

    float drow[k_nr];
    float dnew[k_nr];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += k_nr) {
        for (int k = 0; k < k_nr; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq4_ks *)(dptr + 1);
        }
#ifdef HAVE_FANCY_SIMD
        auto vd = _mm512_loadu_ps(drow);
#else
        auto vd = _mm256_loadu_ps(drow);
#endif
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < k_nr; ++k) {
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    ls[2*ib32+0] = (x8[k][i].scales[ib32] & 254) - 127;
                    ls[2*ib32+1] = ls[2*ib32+0];
                    auto aux128 = _mm_loadu_si128((const __m128i *)x8[k][i].qs+ib32);
                    xv[ib32] = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128), _mm256_set1_epi8(0xf));
                    xv[ib32] = _mm256_shuffle_epi8(values[x8[k][i].scales[ib32] & 1], xv[ib32]);
                }
                dnew[k] = convert_to_q8_k_r8<k_nr>(k, 1.f/127, xv, ls, block, y[i].qs);
            }
#ifdef HAVE_FANCY_SIMD
            _mm256_storeu_si256((__m256i *)y[i].d, _mm512_cvtps_ph(_mm512_mul_ps(vd, _mm512_loadu_ps(dnew)), _MM_ROUND_NEAREST));
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[i].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[i].qs + l, v);
            }
#else
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
#endif
        }
        y += nb;
    }
}

void iqk_convert_iq5_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    constexpr int k_nr = 16;
    using block_q8_k_r = block_q8_k_r16;
#else
    constexpr int k_nr = 8;
    using block_q8_k_r = block_q8_k_r8;
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%k_nr == 0);

    int nb = n/QK_K;

    const block_iq5_ks * x8[k_nr];

    block_q8_k_r * y = (block_q8_k_r *)vy;

    __m256i values[2];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq5nl_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq5nl_values+1);
        values[0] = MM256_SET1_M128I(v1);
        values[1] = MM256_SET1_M128I(v2);
    }

    float drow[k_nr];
    float dnew[k_nr];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    auto mh = _mm256_set1_epi8(-128); // 0x80, spelled so it does not overflow int8

    for (int ix = 0; ix < nrc_x; ix += k_nr) {
        for (int k = 0; k < k_nr; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq5_ks *)(dptr + 1);
        }
#ifdef HAVE_FANCY_SIMD
        auto vd = _mm512_loadu_ps(drow);
#else
        auto vd = _mm256_loadu_ps(drow);
#endif
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < k_nr; ++k) {
                auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    ls[4*ib64+0] = (x8[k][i].scales[2*ib64+0] & 254) - 127;
                    ls[4*ib64+1] = ls[4*ib64+0];
                    ls[4*ib64+2] = (x8[k][i].scales[2*ib64+1] & 254) - 127;
                    ls[4*ib64+3] = ls[4*ib64+2];
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+ib64);
                    xv[2*ib64+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[2*ib64+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    auto qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 7), mh);
                    auto q5vl = _mm256_or_si256(xv[2*ib64+0], qh);
                    auto q5vh = _mm256_or_si256(xv[2*ib64+0], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+0] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 6), mh);
                    q5vl = _mm256_or_si256(xv[2*ib64+1], qh);
                    q5vh = _mm256_or_si256(xv[2*ib64+1], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+1] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    auto shift1 = _mm256_set1_epi8((x8[k][i].scales[2*ib64+0] & 1) << 1);
                    auto shift2 = _mm256_set1_epi8((x8[k][i].scales[2*ib64+1] & 1) << 1);
                    xv[2*ib64+0] = _mm256_add_epi8(xv[2*ib64+0], shift1);
                    xv[2*ib64+1] = _mm256_add_epi8(xv[2*ib64+1], shift2);
                    hbits = _mm256_srli_epi16(hbits, 2);
                }
                dnew[k] = convert_to_q8_k_r8<k_nr>(k, 1.f/127, xv, ls, block, y[i].qs);
            }
#ifdef HAVE_FANCY_SIMD
            _mm256_storeu_si256((__m256i *)y[i].d, _mm512_cvtps_ph(_mm512_mul_ps(vd, _mm512_loadu_ps(dnew)), _MM_ROUND_NEAREST));
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[i].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[i].qs + l, v);
            }
#else
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
#endif
        }
        y += nb;
    }
}

// A Q8_K_R8 tile holds 8 output rows interleaved, so one 32 byte load feeds 8 rows at once and
// the 8 int32 lanes of the accumulator are the 8 row sums.
//
// This variant walks the activation rows in the innermost loop. The weight tile and its unsigned
// copy are therefore produced once per group of 16 columns and reused by every activation row,
// instead of being re-read and re-signed for each row. Ported from ik_llama.cpp
// (ggml/src/iqk/iqk_gemm_kquants.cpp, mul_mat_q8_k_r8_q8_k).
template <int nrc_y>
static void mul_mat_q8_k_r8_q8_k(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x) {
    assert(n % QK_K == 0);
    assert(nrc_x % 8 == 0);

    const int nb = n / QK_K;
    Q8<nrc_y> q8(info);
    const __m256i ones = _mm256_set1_epi16(1);

    __m256  acc [nrc_y];
    __m256i sumi[nrc_y];
    __m256i xv[4];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        const auto * x = (const block_q8_k_r8 *) ((const char *) vx + ix * bx);

        for (int iy = 0; iy < nrc_y; ++iy) {
            acc[iy] = _mm256_setzero_ps();
        }

        for (int ib = 0; ib < nb; ++ib) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                sumi[iy] = _mm256_setzero_si256();
            }

            // 16 columns per step: one __m128i of activations, four 8-row weight loads
            for (int ib16 = 0; ib16 < QK_K / 16; ++ib16) {
                xv[0] = _mm256_loadu_si256((const __m256i *) x[ib].qs + 4 * ib16 + 0);
                xv[1] = _mm256_loadu_si256((const __m256i *) x[ib].qs + 4 * ib16 + 1);
                xv[2] = _mm256_loadu_si256((const __m256i *) x[ib].qs + 4 * ib16 + 2);
                xv[3] = _mm256_loadu_si256((const __m256i *) x[ib].qs + 4 * ib16 + 3);

                // vpmaddubsw needs an unsigned left operand, so move the sign of the weight
                // onto the activation. Both stay within +-127, so the int16 sum cannot saturate.
                const __m256i ax0 = _mm256_sign_epi8(xv[0], xv[0]);
                const __m256i ax1 = _mm256_sign_epi8(xv[1], xv[1]);
                const __m256i ax2 = _mm256_sign_epi8(xv[2], xv[2]);
                const __m256i ax3 = _mm256_sign_epi8(xv[3], xv[3]);

                for (int iy = 0; iy < nrc_y; ++iy) {
                    const __m256i y = MM256_SET1_M128I(
                        _mm_loadu_si128((const __m128i *) q8.y[iy][ib].qs + ib16));
                    const __m256i p0 = _mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax0,
                        _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), xv[0])));
                    const __m256i p1 = _mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax1,
                        _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), xv[1])));
                    const __m256i p2 = _mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax2,
                        _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), xv[2])));
                    const __m256i p3 = _mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax3,
                        _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), xv[3])));
                    sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p0, p1));
                    sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p2, p3));
                }
            }

            const __m256 dx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) x[ib].d));
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 scale = _mm256_mul_ps(dx, _mm256_set1_ps(q8.scale(iy, ib)));
                acc[iy] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(sumi[iy]), acc[iy]);
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
        }
    }
}

// The pre-existing delphi.cpp variant, kept so that the reuse above can be A/B tested with
// GGML_IQK_R8_KERNEL=stream. It re-reads the whole weight tile for every activation row.
template <int nrc_y>
static void mul_mat_q8_k_r8_q8_k_stream_x(
        int n, const void * vx, size_t bx, const DataInfo & info, int nrc_x) {
    assert(n % QK_K == 0);
    assert(nrc_x % 8 == 0);

    const int nb = n / QK_K;
    Q8<nrc_y> q8(info);
    const __m256i ones = _mm256_set1_epi16(1);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        const auto * x = (const block_q8_k_r8 *) ((const char *) vx + ix * bx);

        for (int iy = 0; iy < nrc_y; ++iy) {
            __m256 acc = _mm256_setzero_ps();

            for (int ib = 0; ib < nb; ++ib) {
                __m256i sumi = _mm256_setzero_si256();

                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    for (int l = 0; l < 8; ++l) {
                        const auto * xq = (const __m256i *) (x[ib].qs + 4 * (64 * ib32 + 8 * l));
                        uint32_t y4;
                        std::memcpy(&y4, q8.y[iy][ib].qs + 32 * ib32 + 4 * l, sizeof(y4));
                        const __m256i yq = _mm256_set1_epi32((int) y4);
                        const __m256i xv = _mm256_loadu_si256(xq);
                        const __m256i ax = _mm256_sign_epi8(xv, xv);
                        const __m256i sy = _mm256_sign_epi8(yq, xv);
                        const __m256i p16 = _mm256_maddubs_epi16(ax, sy);
                        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, ones));
                    }
                }

                const __m256 dx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) x[ib].d));
                const __m256 scale = _mm256_mul_ps(dx, _mm256_set1_ps(q8.scale(iy, ib)));
                acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(sumi), acc);
            }

            info.store(ix, iy, acc);
        }
    }
}

template <typename Dequantizer>
static void set_functions(std::array<mul_mat_t, IQK_MAX_NY> & funcs) {
    IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qY_K_q8_K_T, Dequantizer, funcs);
}

// IQ*_KS: one scale per row and 8 (not 16) block scales, hence the other schedule
template <typename Dequantizer>
static void set_functions_ks(std::array<mul_mat_t, IQK_MAX_NY> & funcs) {
    IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, Dequantizer, funcs);
}

// Which paths get the 9..16 activation row kernels. Without them funcs[] stops at 8 rows, so a
// call above that unpacks the same weight rows again for every further group of 8. ik_llama.cpp
// only ever enabled a 16 row kernel under AVX-512, and dropped it when Q8_K_R16 replaced Q8_K_R8
// there, so AVX2 had no result to inherit; measured here it is worth 5 to 12% on the direct IQ4_K
// path between 9 and 31 routed rows, and about 1% on the repacked one. GGML_IQK_NY_MAX takes
// "off", "direct", "repack" or "both" (the default), to A/B the two paths.
static int iqk_ny_max_mode(void) {
    static const int mode = [] {
        const char * value = getenv("GGML_IQK_NY_MAX");
        if (!value) { return 3; }
        if (strcmp(value, "off") == 0) { return 0; }
        if (strcmp(value, "direct") == 0) { return 1; }
        if (strcmp(value, "repack") == 0) { return 2; }
        return 3;
    }();
    return mode;
}

static void set_repacked_functions(std::array<mul_mat_t, IQK_MAX_NY> & funcs) {
    static const bool stream_x = [] {
        const char * value = getenv("GGML_IQK_R8_KERNEL");
        return value && strcmp(value, "stream") == 0;
    }();
    if (stream_x) {
        IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_k_r8_q8_k_stream_x, funcs);
    } else {
        IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_k_r8_q8_k, funcs);
    }
}

} // namespace

bool iqk_convert_iqk_quants_q80_r8(
        int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    if (n % QK_K != 0 || nrc_x % 8 != 0) {
        return false;
    }

    switch ((ggml_type) type) {
        case GGML_TYPE_IQ2_K: iqk_convert_iq2_k_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_K: iqk_convert_iq3_k_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_K: iqk_convert_iq4_k_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_K: iqk_convert_iq5_k_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ6_K: iqk_convert_iq6_k_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ2_KS: iqk_convert_iq2_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_KS: iqk_convert_iq3_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_KS: iqk_convert_iq4_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_KS: iqk_convert_iq5_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        default: return false;
    }

    return true;
}

bool iqk_set_kernels_iqk_quants(
        int ne00, int type_a, int type_b,
        std::array<mul_mat_t, IQK_MAX_NY> & kernels,
        std::array<mul_mat_t, IQK_MAX_NY_HI - IQK_MAX_NY> & kernels_hi) {
    if (ne00 % QK_K != 0 || (ggml_type) type_b != GGML_TYPE_Q8_K) {
        return false;
    }

    kernels_hi = {};

    if (type_a == IQK_TYPE_Q8_K_R8) {
        set_repacked_functions(kernels);
        if (iqk_ny_max_mode() & 2) {
            IQK_SET_MUL_MAT_FUNCTIONS_HI(mul_mat_q8_k_r8_q8_k, kernels_hi);
        }
        return true;
    }

    switch ((ggml_type) type_a) {
        case GGML_TYPE_IQ2_K: set_functions<DequantizerIQ2K>(kernels); break;
        case GGML_TYPE_IQ3_K: set_functions<DequantizerIQ3K>(kernels); break;
        case GGML_TYPE_IQ4_K:
            set_functions<DequantizerIQ4K>(kernels);
            if (iqk_ny_max_mode() & 1) {
                IQK_SET_MUL_MAT_FUNCTIONS_HI_T(mul_mat_qY_K_q8_K_T, DequantizerIQ4K, kernels_hi);
            }
            break;
        case GGML_TYPE_IQ5_K: set_functions<DequantizerIQ5K>(kernels); break;
        case GGML_TYPE_IQ6_K: set_functions<DequantizerIQ6K>(kernels); break;
        case GGML_TYPE_IQ2_KS: set_functions_ks<DequantizerIQ2KS>(kernels); break;
        case GGML_TYPE_IQ3_KS: set_functions_ks<DequantizerIQ3KS>(kernels); break;
        case GGML_TYPE_IQ4_KS: {
            set_functions_ks<DequantizerIQ4KS>(kernels);
            static const bool pairs = [] {
                const char * value = getenv("GGML_IQ4_KS_KERNEL");
                return !value || strcmp(value, "base") != 0;
            }();
            if (pairs) {
                kernels[1] = mul_mat_iq4_ks_q8_K_pairs<2>;
                kernels[2] = mul_mat_iq4_ks_q8_K_pairs<3>;
                kernels[3] = mul_mat_iq4_ks_q8_K_pairs<4>;
            }
            break;
        }
        case GGML_TYPE_IQ5_KS: set_functions_ks<DequantizerIQ5KS>(kernels); break;
        default: return false;
    }

    return true;
}

#endif // IQK_IMPLEMENT
