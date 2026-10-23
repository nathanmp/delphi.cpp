#pragma once

#include "common.cuh"
#include "iqk_cuda_common.h"
#include "moe-packed.cuh"
#include "vecdotq.cuh"

// The IQ*_K and IQ*_KS decoders implement the block formats of ik_llama.cpp (Copyright (C) 2024 Iwan Kawrakow, MIT license) and
// use its lookup tables from iqk_cuda_common.h.
//
// delphi.cpp: device side of the lossless packed layouts of moe-packed.cuh, shared by the pack kernels and the grouped packed kernel
// (moe-packed.cu) and by MMVQ (mmvq.cu). A superblock holds one row tile of MOE_PACKED_I rows and one QK_K block, and has exactly the
// size of its MOE_PACKED_I GGUF blocks. The superblocks of an expert are ordered [row tile][QK_K block]. A superblock is 2 k chunks
// of 128 values (the codes, in m16n8k32 A fragment order), then the per-row metadata of the QK_K block.
//
// 16-row group mg (0..7) of a row tile and lane (g = lane/4, t = lane%4) of an A fragment: register j of the lane holds row
// 16*mg + g + 8*(j%2) and the values 16*(j/2) + 4*t ... + 3 of a k step of 32 values.

#define MOE_PACKED_I            128   // rows of src0 per tile, the row tile of the grouped MMQ path

#define MOE_PACKED_IQ3S_CHUNK   6656  // Q 4096, S 2048, H 512
#define MOE_PACKED_IQ3S_S       4096  // offset of S in a chunk
#define MOE_PACKED_IQ3S_H       6144  // offset of H in a chunk
#define MOE_PACKED_IQ3S_BLOCK   14080 // 2 chunks, d [128] f16, scales [128] u32

#define MOE_PACKED_IQ4XS_CHUNK  8192  // qs words [mg][4 k steps][2 halves][32 lanes]
#define MOE_PACKED_IQ4XS_BLOCK  17408 // 2 chunks, d [128] f16, the 6-bit scales as 48 bits: [128] u32, [128] u16

#define MOE_PACKED_Q4K_CHUNK    8192  // qs words [mg][2 groups of 64 values][32 lanes] uint4
#define MOE_PACKED_Q4K_BLOCK    18432 // 2 chunks, dm [128] half2, the 6-bit scales and mins as 96 bits: [3][128] u32

#define MOE_PACKED_Q6K_CHUNK    12288 // L 8192, H 4096
#define MOE_PACKED_Q6K_H        8192  // offset of H in a chunk
#define MOE_PACKED_Q6K_BLOCK    26880 // 2 chunks, d [128] f16, scales [128][16] i8

#define MOE_PACKED_Q80_CHUNK    16384 // qs words [mg][4 k steps][lane] uint4
#define MOE_PACKED_Q80_BLOCK    34816 // 2 chunks, d [128][8] f16

#define MOE_PACKED_Q50_CHUNK    10240 // ql words [mg][4 k steps][lane] uint2, qh bytes [mg][4 k steps][lane] uint16
#define MOE_PACKED_Q50_H        8192  // offset of qh bytes in a chunk
#define MOE_PACKED_Q50_BLOCK    22528 // 2 chunks, d [128][8] f16

#define MOE_PACKED_IQ4NL_CHUNK  8192  // qs words [mg][4 k steps][2 halves][32 lanes]
#define MOE_PACKED_IQ4NL_BLOCK  18432 // 2 chunks, d [128] f16

#define MOE_PACKED_MXFP4_CHUNK  8192  // qs words [mg][4 k steps][2 halves][32 lanes]
#define MOE_PACKED_MXFP4_BLOCK  17408 // 2 chunks, e8m0 exponents [128][8]

#define MOE_PACKED_Q5K_CHUNK    10240 // qs words [mg][2 groups of 64 values][lane] uint4, qh bytes [mg][2 groups][lane] u32
#define MOE_PACKED_Q5K_H        8192  // offset of qh bytes in a chunk
#define MOE_PACKED_Q5K_BLOCK    22528 // 2 chunks, dm [128] half2, the 6-bit scales and mins as 96 bits: [3][128] u32

#define MOE_PACKED_Q41_CHUNK    8192  // qs bytes [row][4 source blocks]
#define MOE_PACKED_Q41_BLOCK    20480 // 2 chunks, dm [128][8] half2

#define MOE_PACKED_Q51_CHUNK    10240 // qs and qh bytes [row][4 source blocks]
#define MOE_PACKED_Q51_BLOCK    24576 // 2 chunks, dm [128][8] half2

#define MOE_PACKED_Q3K_CHUNK    4096  // qs bytes [row], one source 128-value half
#define MOE_PACKED_Q3K_BLOCK    14080 // 2 chunks, d [128] f16, scales [128][12], hmask [128][32]

#define MOE_PACKED_IQ3XXS_CHUNK 6144  // qs and scale/sign bytes [row], four source 32-value groups
#define MOE_PACKED_IQ3XXS_BLOCK 12544 // 2 chunks, d [128] f16

#define MOE_PACKED_IQ2XXS_CHUNK 4096  // qs bytes [row], four source 32-value groups
#define MOE_PACKED_IQ2XXS_BLOCK 8448  // 2 chunks, d [128] f16

#define MOE_PACKED_IQ2XS_CHUNK  4096  // qs bytes [row], four source 32-value groups
#define MOE_PACKED_IQ2XS_BLOCK  9472  // 2 chunks, d [128] f16, scales [128][8] u8

#define MOE_PACKED_IQ2S_CHUNK   4096  // qs and sign bytes [row], four source 32-value groups
#define MOE_PACKED_IQ2S_BLOCK   10496 // 2 chunks, d [128] f16, qh [128][8], scales [128][8]

#define MOE_PACKED_IQ2K_CHUNK   4096  // qs bytes [row], four source 32-value groups
#define MOE_PACKED_IQ2K_BLOCK   9728  // 2 chunks, d [128] f16, extra [128] u16, scales [128][8] u8

#define MOE_PACKED_IQ3K_CHUNK   4096  // qs bytes [row], one source 128-value half
#define MOE_PACKED_IQ3K_BLOCK   14080 // 2 chunks, d [128] f16, extra [128] u16, scales_h [128] u16, scales_l [128][8], qh [128][32]

#define MOE_PACKED_IQ4K_CHUNK   8192  // qs words [mg][4 k steps][2 halves][32 lanes], as IQ4_KS
#define MOE_PACKED_IQ4K_BLOCK   18432 // 2 chunks, d [128] f16, extra [128] u16, scales_h [128][4], scales_l [128][8]

#define MOE_PACKED_IQ5K_CHUNK   8192  // qs bytes [row], one source 128-value half
#define MOE_PACKED_IQ5K_BLOCK   22528 // 2 chunks, d [128] f16, extra [128] u16, scales_h [128][4], scales_l [128][8], qh [128][32]

// IQ*_KS stores one scale before every GGUF tensor row. The packed layout keeps that scale once in a per-expert row-metadata
// plane after the packed superblocks. A staging buffer holds the ordinary block metadata followed by the metadata of its 128 rows.
#define MOE_PACKED_IQ2KS_CHUNK  4096  // qs [row][128 values]
#define MOE_PACKED_IQ2KS_BLOCK  8960  // 2 chunks, extra [128] u16, scales [128][4]
#define MOE_PACKED_IQ3KS_CHUNK  6144  // qs and qh [row][128 values]
#define MOE_PACKED_IQ3KS_BLOCK 13056  // 2 chunks, extra [128] u16, scales [128][4]
#define MOE_PACKED_IQ4KS_CHUNK  8192  // qs words in m16n8k32 A-fragment order
#define MOE_PACKED_IQ4KS_BLOCK 17408  // 2 chunks, scales [128][8]
#define MOE_PACKED_IQ5KS_CHUNK 10240  // qs and qh [row][128 values]
#define MOE_PACKED_IQ5KS_BLOCK 21504  // 2 chunks, scales [128][8]

static_assert(MOE_PACKED_IQ3S_BLOCK == MOE_PACKED_I*sizeof(block_iq3_s), "the iq3s layout must keep the IQ3_S size");
static_assert(MOE_PACKED_IQ3S_BLOCK == 2*MOE_PACKED_IQ3S_CHUNK + MOE_PACKED_I*(sizeof(half) + sizeof(uint32_t)), "bad iq3s layout");
static_assert(MOE_PACKED_IQ4XS_BLOCK == MOE_PACKED_I*sizeof(block_iq4_xs), "the iq4xs layout must keep the IQ4_XS size");
static_assert(MOE_PACKED_IQ4XS_BLOCK == 2*MOE_PACKED_IQ4XS_CHUNK + MOE_PACKED_I*(sizeof(half) + sizeof(uint32_t) + sizeof(uint16_t)),
    "bad iq4xs layout");
static_assert(MOE_PACKED_Q4K_BLOCK == MOE_PACKED_I*sizeof(block_q4_K), "the q4k layout must keep the Q4_K size");
static_assert(MOE_PACKED_Q4K_BLOCK == 2*MOE_PACKED_Q4K_CHUNK + MOE_PACKED_I*(sizeof(half2) + 3*sizeof(uint32_t)), "bad q4k layout");
static_assert(MOE_PACKED_Q6K_BLOCK == MOE_PACKED_I*sizeof(block_q6_K), "the q6k layout must keep the Q6_K size");
static_assert(MOE_PACKED_Q6K_BLOCK == 2*MOE_PACKED_Q6K_CHUNK + MOE_PACKED_I*(sizeof(half) + QK_K/16), "bad q6k layout");
static_assert(MOE_PACKED_Q80_BLOCK == MOE_PACKED_I*(QK_K/QK8_0)*sizeof(block_q8_0), "the q80 layout must keep the Q8_0 size");
static_assert(MOE_PACKED_Q80_BLOCK == 2*MOE_PACKED_Q80_CHUNK + MOE_PACKED_I*(QK_K/QK8_0)*sizeof(half), "bad q80 layout");
static_assert(MOE_PACKED_Q50_BLOCK == MOE_PACKED_I*(QK_K/QK5_0)*sizeof(block_q5_0), "the q50 layout must keep the Q5_0 size");
static_assert(MOE_PACKED_Q50_BLOCK == 2*MOE_PACKED_Q50_CHUNK + MOE_PACKED_I*(QK_K/QK5_0)*sizeof(half), "bad q50 layout");
static_assert(MOE_PACKED_IQ4NL_BLOCK == MOE_PACKED_I*(QK_K/QK4_NL)*sizeof(block_iq4_nl), "the iq4nl layout must keep the IQ4_NL size");
static_assert(MOE_PACKED_IQ4NL_BLOCK == 2*MOE_PACKED_IQ4NL_CHUNK + MOE_PACKED_I*(QK_K/QK4_NL)*sizeof(half), "bad iq4nl layout");
static_assert(MOE_PACKED_MXFP4_BLOCK == MOE_PACKED_I*(QK_K/QK_MXFP4)*sizeof(block_mxfp4), "the mxfp4 layout must keep the MXFP4 size");
static_assert(MOE_PACKED_MXFP4_BLOCK == 2*MOE_PACKED_MXFP4_CHUNK + MOE_PACKED_I*(QK_K/QK_MXFP4), "bad mxfp4 layout");
static_assert(MOE_PACKED_Q5K_BLOCK == MOE_PACKED_I*sizeof(block_q5_K), "the q5k layout must keep the Q5_K size");
static_assert(MOE_PACKED_Q5K_BLOCK == 2*MOE_PACKED_Q5K_CHUNK + MOE_PACKED_I*(sizeof(half2) + 3*sizeof(uint32_t)), "bad q5k layout");
static_assert(MOE_PACKED_Q41_BLOCK == MOE_PACKED_I*(QK_K/QK4_1)*sizeof(block_q4_1), "the q41 layout must keep the Q4_1 size");
static_assert(MOE_PACKED_Q41_BLOCK == 2*MOE_PACKED_Q41_CHUNK + MOE_PACKED_I*(QK_K/QK4_1)*sizeof(half2), "bad q41 layout");
static_assert(MOE_PACKED_Q51_BLOCK == MOE_PACKED_I*(QK_K/QK5_1)*sizeof(block_q5_1), "the q51 layout must keep the Q5_1 size");
static_assert(MOE_PACKED_Q51_BLOCK == 2*MOE_PACKED_Q51_CHUNK + MOE_PACKED_I*(QK_K/QK5_1)*sizeof(half2), "bad q51 layout");
static_assert(MOE_PACKED_Q3K_BLOCK == MOE_PACKED_I*sizeof(block_q3_K), "the q3k layout must keep the Q3_K size");
static_assert(MOE_PACKED_Q3K_BLOCK == 2*MOE_PACKED_Q3K_CHUNK + MOE_PACKED_I*(sizeof(half) + 12 + QK_K/8), "bad q3k layout");
static_assert(MOE_PACKED_IQ3XXS_BLOCK == MOE_PACKED_I*sizeof(block_iq3_xxs), "the iq3xxs layout must keep the IQ3_XXS size");
static_assert(MOE_PACKED_IQ3XXS_BLOCK == 2*MOE_PACKED_IQ3XXS_CHUNK + MOE_PACKED_I*sizeof(half), "bad iq3xxs layout");
static_assert(MOE_PACKED_IQ2XXS_BLOCK == MOE_PACKED_I*sizeof(block_iq2_xxs), "the iq2xxs layout must keep the IQ2_XXS size");
static_assert(MOE_PACKED_IQ2XXS_BLOCK == 2*MOE_PACKED_IQ2XXS_CHUNK + MOE_PACKED_I*(sizeof(half)), "bad iq2xxs layout");
static_assert(MOE_PACKED_IQ2XS_BLOCK == MOE_PACKED_I*sizeof(block_iq2_xs), "the iq2xs layout must keep the IQ2_XS size");
static_assert(MOE_PACKED_IQ2XS_BLOCK == 2*MOE_PACKED_IQ2XS_CHUNK + MOE_PACKED_I*(sizeof(half) + 8), "bad iq2xs layout");
static_assert(MOE_PACKED_IQ2S_BLOCK == MOE_PACKED_I*sizeof(block_iq2_s), "the iq2s layout must keep the IQ2_S size");
static_assert(MOE_PACKED_IQ2S_BLOCK == 2*MOE_PACKED_IQ2S_CHUNK + MOE_PACKED_I*(sizeof(half) + 8 + 8), "bad iq2s layout");
static_assert(MOE_PACKED_IQ2K_BLOCK == MOE_PACKED_I*sizeof(block_iq2_k), "the iq2k layout must keep the IQ2_K size");
static_assert(MOE_PACKED_IQ2K_BLOCK == 2*MOE_PACKED_IQ2K_CHUNK + MOE_PACKED_I*(sizeof(half) + sizeof(uint16_t) + 8), "bad iq2k layout");
static_assert(MOE_PACKED_IQ3K_BLOCK == MOE_PACKED_I*sizeof(block_iq3_k), "the iq3k layout must keep the IQ3_K size");
static_assert(MOE_PACKED_IQ3K_BLOCK == 2*MOE_PACKED_IQ3K_CHUNK + MOE_PACKED_I*(sizeof(half) + 2*sizeof(uint16_t) + 8 + QK_K/8), "bad iq3k layout");
static_assert(MOE_PACKED_IQ4K_BLOCK == MOE_PACKED_I*sizeof(block_iq4_k), "the iq4k layout must keep the IQ4_K size");
static_assert(MOE_PACKED_IQ4K_BLOCK == 2*MOE_PACKED_IQ4K_CHUNK + MOE_PACKED_I*(sizeof(half) + sizeof(uint16_t) + QK_K/64 + QK_K/32), "bad iq4k layout");
static_assert(MOE_PACKED_IQ5K_BLOCK == MOE_PACKED_I*sizeof(block_iq5_k), "the iq5k layout must keep the IQ5_K size");
static_assert(MOE_PACKED_IQ5K_BLOCK == 2*MOE_PACKED_IQ5K_CHUNK + MOE_PACKED_I*(sizeof(half) + sizeof(uint16_t) + QK_K/64 + QK_K/32 + QK_K/8), "bad iq5k layout");
static_assert(MOE_PACKED_IQ2KS_BLOCK == MOE_PACKED_I*sizeof(block_iq2_ks), "the iq2ks payload layout must keep the IQ2_KS block size");
static_assert(MOE_PACKED_IQ2KS_BLOCK == 2*MOE_PACKED_IQ2KS_CHUNK + MOE_PACKED_I*(sizeof(uint16_t) + QK_K/64), "bad iq2ks payload layout");
static_assert(MOE_PACKED_IQ3KS_BLOCK == MOE_PACKED_I*sizeof(block_iq3_ks), "the iq3ks payload layout must keep the IQ3_KS block size");
static_assert(MOE_PACKED_IQ3KS_BLOCK == 2*MOE_PACKED_IQ3KS_CHUNK + MOE_PACKED_I*(sizeof(uint16_t) + QK_K/64), "bad iq3ks payload layout");
static_assert(MOE_PACKED_IQ4KS_BLOCK == MOE_PACKED_I*sizeof(block_iq4_ks), "the iq4ks payload layout must keep the IQ4_KS block size");
static_assert(MOE_PACKED_IQ4KS_BLOCK == 2*MOE_PACKED_IQ4KS_CHUNK + MOE_PACKED_I*(QK_K/32), "bad iq4ks payload layout");
static_assert(MOE_PACKED_IQ5KS_BLOCK == MOE_PACKED_I*sizeof(block_iq5_ks), "the iq5ks payload layout must keep the IQ5_KS block size");
static_assert(MOE_PACKED_IQ5KS_BLOCK == 2*MOE_PACKED_IQ5KS_CHUNK + MOE_PACKED_I*(QK_K/32), "bad iq5ks payload layout");

// Byte b of the result is 0xFF if bit 7 of byte b of x is set, else 0 (prmt sign replication).
static __device__ __forceinline__ uint32_t moe_iq3s_sign_mask(const uint32_t x) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(0), "r"(0xBA98));
    return r;
}

// vec_dot_iq3_s_q8_1 on the iq3s layout for rows g and g + 8 of 16-row group mg (0 .. 7), sub-block ib of superblock sb, bq8_1 the
// q8_1 block of the sub-block. Row 16*mg + 8*h + g holds its 8 grid groups l in the bytes h + 2*(l/4) of the Q words of lanes
// 4*g + l%4, so one 16 B load gives the codes of both rows. The integer codes, the scale and the float operations of each row are
// those of vec_dot_iq3_s_q8_1.
// grid: iq3s_grid, or (grid_m1) a shared memory copy of iq3s_grid - 0x01010101, whose 16 lookups per call hit fewer banks than
// L1 lines. The grid bytes g are odd, so (g - 1) | 1 = g and ~(g - 1) | 1 = -g: two instructions instead of the emulated __vsub4.
template <bool grid_m1 = false>
static __device__ __forceinline__ float2 vec_dot_iq3s_packed_q8_1(
        const char * __restrict__ sb, const int mg, const int g, const int ib, const block_q8_1 * __restrict__ bq8_1,
        const uint32_t * __restrict__ grid) {
    const int ks = ib % 4;

    const char * ch = sb + (ib/4)*MOE_PACKED_IQ3S_CHUNK;
    const uint4 q = *(const uint4 *) (ch + 4*((mg*4 + ks)*WARP_SIZE + 4*g));
    const uint4 s = *(const uint4 *) (ch + MOE_PACKED_IQ3S_S + 4*((mg*2 + ks/2)*WARP_SIZE + 4*g));
    const uint2 hh = *(const uint2 *) (ch + MOE_PACKED_IQ3S_H + 2*(mg*WARP_SIZE + 4*g));

    const uint32_t qw[4] = { q.x, q.y, q.z, q.w };
    const uint32_t sw[4] = { s.x, s.y, s.z, s.w };
    const uint32_t hw[4] = { hh.x & 0xFFFF, hh.x >> 16, hh.y & 0xFFFF, hh.y >> 16 };

    int sumi[2] = { 0, 0 };
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int t = l % 4;
        const int u = get_int_b4(bq8_1->qs, l);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int j    = h + 2*(l/4);
            const int idx  = __byte_perm(qw[t], 0, 0x4440 | j) | (((hw[t] >> (4*ks + j)) & 1) << 8);
            const int sgn  = moe_iq3s_sign_mask(sw[t] << (4*(ks % 2) + j));
            const int gv   = grid_m1 ? (grid[idx] ^ sgn) | 0x01010101 : __vsub4(grid[idx] ^ sgn, sgn);
            sumi[h] = ggml_cuda_dp4a(gv, u, sumi[h]);
        }
    }

    const char * meta = sb + 2*MOE_PACKED_IQ3S_CHUNK;
    const float  dy   = __low2float(bq8_1->ds);
    float res[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int      r  = 16*mg + g + 8*h;
        const uint32_t sc = ((const uint32_t *) (meta + MOE_PACKED_I*sizeof(half)))[r];
        sumi[h] *= 1 + 2*((sc >> (4*ib)) & 0x0F);
        const float d = __half2float(((const half *) meta)[r]) * dy;
        res[h] = d * sumi[h];
    }
    return make_float2(res[0], res[1]);
}

// prmt.b32 in the generic mode: a selector nibble n < 8 copies byte n, n >= 8 replicates the sign of byte n - 8.
static __device__ __forceinline__ uint32_t moe_prmt(const uint32_t a, const uint32_t b, const uint32_t s) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
}

// kvalues_iq4nl of the 8 nibbles of an iq4xs word: nibbles 0-3 give x, nibbles 4-7 give y (the values of get_int_from_table_16 on
// the GGUF word). A nibble n < 8 takes byte n of the low table half, n >= 8 byte n - 8 of the high half (selector n ^ 8), and the
// last prmt takes one of the two for each byte.
static __device__ __forceinline__ int2 moe_iq4_decode(const uint32_t w, const int8_t * values) {
    const uint32_t * tab = (const uint32_t *) values;
    const uint32_t x = w ^ 0x88888888;
    const uint32_t s = ((w >> 1) & 0x44444444) | 0x32103210;
    const uint32_t lo = moe_prmt(moe_prmt(tab[0], tab[1], w),       moe_prmt(tab[2], tab[3], x),       s);
    const uint32_t hi = moe_prmt(moe_prmt(tab[0], tab[1], w >> 16), moe_prmt(tab[2], tab[3], x >> 16), s >> 16);
    return make_int2((int) lo, (int) hi);
}

static __device__ __forceinline__ int2 moe_iq4xs_decode(const uint32_t w) {
    return moe_iq4_decode(w, kvalues_iq4nl);
}

// Register lookups for the iq2nl, iq3nl and iq5nl tables of the IQ*_K formats, as table words (byte i of a word is table entry i) so
// that the lookups need no loads.
static constexpr __host__ __device__ uint32_t moe_w4(const int a, const int b, const int c, const int d) {
    return (uint32_t) (uint8_t) a | (uint32_t) (uint8_t) b << 8 | (uint32_t) (uint8_t) c << 16 | (uint32_t) (uint8_t) d << 24;
}

// The values of the 8 nibbles of idx (each < 8) in a table of 8 entries (the words t0 and t1): the even nibbles in x, the odd nibbles
// in y, as get_int_from_table_8. A caller that needs only x or only y gets the other prmt removed.
static __device__ __forceinline__ int2 moe_lookup8(const uint32_t idx, const uint32_t t0, const uint32_t t1) {
    const uint32_t v0 = moe_prmt(t0, t1, idx);
    const uint32_t v1 = moe_prmt(t0, t1, idx >> 16);
    return make_int2((int) moe_prmt(v0, v1, 0x6420), (int) moe_prmt(v0, v1, 0x7531));
}

// The values of the 8 nibbles of q in a table of 16 entries (the words t0 .. t3): the low nibbles of the 4 bytes in x, the high
// nibbles in y, as get_int_from_table_16.
static __device__ __forceinline__ int2 moe_lookup16(const uint32_t q, const uint32_t t0, const uint32_t t1, const uint32_t t2,
        const uint32_t t3) {
    const uint32_t x  = q ^ 0x88888888;
    const uint32_t s  = ((q >> 1) & 0x44444444) | 0x32103210;
    const uint32_t lo = moe_prmt(moe_prmt(t0, t1, q),       moe_prmt(t2, t3, x),       s);
    const uint32_t hi = moe_prmt(moe_prmt(t0, t1, q >> 16), moe_prmt(t2, t3, x >> 16), s >> 16);
    return make_int2((int) moe_prmt(lo, hi, 0x6420), (int) moe_prmt(lo, hi, 0x7531));
}

// The iq2nl and iq3nl tables have a shifted second half (iq2nl_values + 5, iq3nl_values + 4) that no byte carry affects, so the callers
// add 0x05050505 or 0x04040404 for an extra bit instead of a table index bit.
#define MOE_IQ2NL_T0 moe_w4(-31, -13, 1, 17)
#define MOE_IQ2NL_T1 moe_w4(-26, -8, 6, 22)
#define MOE_IQ3NL_T0 moe_w4(-63, -40, -23, -10)
#define MOE_IQ3NL_T1 moe_w4(1, 13, 28, 47)

// iq5nl_values with 0x80 added to each byte (xor 0x80808080), in 8 words: the second half of iq5nl_values is the first plus 2, but
// the first holds -1, so the extra shift needs unsigned bytes. moe_iq5_value() adds the shift and removes the bias.
static __device__ __forceinline__ void moe_iq5_lookup(const uint32_t q, int2 & a, int2 & b) {
    a = moe_lookup16(q, moe_w4(-126, -114, -103, -92) ^ 0x80808080, moe_w4(-83, -74, -65, -57) ^ 0x80808080,
        moe_w4(-50, -43, -36, -30) ^ 0x80808080, moe_w4(-24, -18, -12, -6) ^ 0x80808080);
    b = moe_lookup16(q, moe_w4(-1, 5, 11, 17) ^ 0x80808080, moe_w4(23, 29, 36, 43) ^ 0x80808080,
        moe_w4(51, 59, 68, 77) ^ 0x80808080, moe_w4(87, 97, 109, 121) ^ 0x80808080);
}

static __device__ __forceinline__ uint32_t moe_iq5_value(const uint32_t biased, const uint32_t shift) {
    return (biased + shift) ^ 0x80808080;
}

// Per byte: a where mask is 0, b where mask is 0xFF.
static __device__ __forceinline__ uint32_t moe_select(const uint32_t mask, const uint32_t a, const uint32_t b) {
    return (a & ~mask) | (b & mask);
}

// The iq2nl values of the 2-bit codes of the 4 bytes of w at k step ks of a 128-value chunk: the codes of k steps ks and ks ^ 2 share
// the nibbles of one lookup (ks & 1 selects the shift), so the two calls of a pair share their prmt. e0 and e1 (0 or 1) are the extra
// bits of ks & 1 (even nibbles) and (ks & 1) + 2 (odd nibbles), the index bit 2.
static __device__ __forceinline__ uint32_t moe_iq2_reg(const uint32_t w, const uint32_t e0, const uint32_t e1, const int ks) {
    const int2 v = moe_lookup8(((w >> (2*(ks & 1))) & 0x33333333) | e0*0x04040404u | e1*0x40404040u, MOE_IQ2NL_T0, MOE_IQ2NL_T1);
    return ks < 2 ? v.x : v.y;
}

// The iq3nl values of the 3-bit codes of the 4 bytes of w at k step ks (without the extra shift). qh holds in byte j the high bits
// of byte j in its 4 low bits (k step k at bit k): the index bit 2 of the even nibbles is bit ks & 1 and of the odd nibbles bit (ks & 1) + 2.
static __device__ __forceinline__ uint32_t moe_iq3_reg(const uint32_t w, const uint32_t qh, const int ks) {
    const int a = ks & 1;
    const uint32_t m = ((qh << (2 - a)) & 0x04040404u) | ((qh << (4 - a)) & 0x40404040u);
    const int2 v = moe_lookup8(((w >> (2*a)) & 0x33333333) | m, MOE_IQ3NL_T0, MOE_IQ3NL_T1);
    return ks < 2 ? v.x : v.y;
}

// Byte j of the result is nibble j of the 16 bits of p.
static __device__ __forceinline__ uint32_t moe_nibbles_to_bytes(const uint32_t p) {
    const uint32_t x = (p | (p << 8)) & 0x00FF00FF;
    return (x | (x << 4)) & 0x0F0F0F0F;
}

// The packed q5 layouts keep the four high bits needed by one ql byte in a byte: low nibble for ql values 0..3 and high nibble for
// ql values 4..7. This is the form used by the code fragments and is expanded only when the upstream q5 MMVQ helper needs qh bytes.
static __device__ __forceinline__ uint8_t moe_q50_bits(const uint32_t qh, const int t) {
    uint8_t h = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h |= ((qh >> (4*t + i)) & 1) << i;
        h |= ((qh >> (16 + 4*t + i)) & 1) << (4 + i);
    }
    return h;
}

static __device__ __forceinline__ uint8_t moe_q5k_bits(const uint32_t qh) {
    uint8_t h = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h |= ((qh >> (8*i + 0)) & 1) << i;
        h |= ((qh >> (8*i + 1)) & 1) << (4 + i);
    }
    return h;
}

static __device__ __forceinline__ uint32_t moe_q5_code(const uint32_t ql, const uint8_t h, const int part) {
    uint32_t r = (ql >> (4*part)) & 0x0F0F0F0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r |= ((h >> (part ? 4 + i : i)) & 1) << (8*i + 4);
    }
    return r;
}

// Q5_0 uses the signed code q - 16 in the D4 MMQ fragment; Q5_K keeps the unsigned code for its DM fragment.
static __device__ __forceinline__ uint32_t moe_q5_0_code(const uint32_t ql, const uint8_t h, const int part) {
    return __vsubss4(moe_q5_code(ql, h, part), 0x10101010);
}

static __device__ __forceinline__ uint32_t moe_q50_vh(const uint8_t h) {
    return (h & 0x0F) | ((uint32_t) (h >> 4) << 16);
}

static __device__ __forceinline__ uint32_t moe_q5k_vh(const uint8_t h) {
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r |= ((h >> i) & 1) << (8*i + 0);
        r |= ((h >> (4 + i)) & 1) << (8*i + 1);
    }
    return r;
}

// GGUF qs word (byte k: value k in the low nibble, value 16 + k in the high nibble) <-> iq4xs word (nibble k: value k, nibble 4 + k:
// value 16 + k).
static __device__ __forceinline__ uint32_t moe_iq4xs_word(const uint32_t q) {
    uint32_t w = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        w |= ((q >> (8*k)) & 0xF) << (4*k) | ((q >> (8*k + 4)) & 0xF) << (4*k + 16);
    }
    return w;
}

static __device__ __forceinline__ uint32_t moe_iq4xs_qs(const uint32_t w) {
    uint32_t q = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        q |= ((w >> (4*k)) & 0xF) << (8*k) | ((w >> (4*k + 16)) & 0xF) << (8*k + 4);
    }
    return q;
}

// The 4 six-bit scales of k chunk c of a row, from the words A and B of the row.
static __device__ __forceinline__ uint32_t moe_iq4xs_scales(const uint32_t A, const uint32_t B, const int c) {
    return c == 0 ? A & 0xFFFFFF : A >> 24 | B << 8;
}

// vec_dot_iq4_xs_q8_1 on the iq4xs layout for rows g and g + 8 of 16-row group mg, sub-block ib of superblock sb, bq8_1 the q8_1
// block of the sub-block. The 16 qs bytes of the sub-block of row 16*mg + 8*h + g are the words of lanes 4*g ... 4*g + 3 of half h,
// so one 16 B load per row gives them. The int8 values, the scale and the operations of each row are those of vec_dot_iq4_xs_q8_1.
static __device__ __forceinline__ float2 vec_dot_iq4xs_packed_q8_1(
        const char * __restrict__ sb, const int mg, const int g, const int ib, const block_q8_1 * __restrict__ bq8_1) {
    const char * ch = sb + (ib/4)*MOE_PACKED_IQ4XS_CHUNK + 4*((mg*4 + ib % 4)*2*WARP_SIZE + 4*g);
    const uint4 q0 = *(const uint4 *) ch;
    const uint4 q1 = *(const uint4 *) (ch + 4*WARP_SIZE);
    const uint32_t qw[2][4] = { { q0.x, q0.y, q0.z, q0.w }, { q1.x, q1.y, q1.z, q1.w } };

    int sumi[2] = { 0, 0 };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int u0 = get_int_b4(bq8_1->qs, j + 0);
        const int u1 = get_int_b4(bq8_1->qs, j + 4);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int2 v = moe_iq4xs_decode(qw[h][j]);
            sumi[h] = ggml_cuda_dp4a(v.x, u0, sumi[h]);
            sumi[h] = ggml_cuda_dp4a(v.y, u1, sumi[h]);
        }
    }

    const char * meta = sb + 2*MOE_PACKED_IQ4XS_CHUNK;
    const float  dy   = __low2float(bq8_1->ds);
    float res[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int      r  = 16*mg + g + 8*h;
        const uint32_t A  = ((const uint32_t *) (meta + MOE_PACKED_I*sizeof(half)))[r];
        const uint32_t B  = ((const uint16_t *) (meta + MOE_PACKED_I*(sizeof(half) + sizeof(uint32_t))))[r];
        const int      ls = (moe_iq4xs_scales(A, B, ib/4) >> (6*(ib % 4))) & 0x3F;
        sumi[h] *= ls - 32;
        const float d = __half2float(((const half *) meta)[r]) * dy;
        res[h] = d * sumi[h];
    }
    return make_float2(res[0], res[1]);
}

// ---------------------------------------------------------------------------------------------------------------------
// Per-format interface. A new lossless format defines these members, adds its moe_packed_fmt value and its GGUF type to
// moe_packed_is_q and moe_packed_q_fmt, and adds a case to the format switches of moe-packed.cu and mmvq.cu.
//   block_t, type          the GGUF block and type
//   chunk_bytes, sb_bytes  the code bytes of one k chunk of 128 values, the superblock bytes (2 chunks, then the metadata)
//   family                 the MMQ arithmetic that the grouped kernel reproduces (moe_packed_family)
//   pack(b0, b1, out, mg, lane)
//                          writes what lane (g, t) of 16-row group mg owns in superblock out: rows 16*mg + g (b0) and + 8 (b1)
//   unpack(in, r, b[, ib]) rebuilds the GGUF block ib of row r (GGML_CUDA_MOE_PACKED_VERIFY; ib is used when QK_K spans blocks)
//   tab_words, tab(i)      a table that the grouped kernel keeps in shared memory (tab_words = 0: none)
//   frag_state, load_state(s, mg, lane, kc), fragment(s, tab, st, mg, ks, lane), scale(st, h, ks)
//                          the grouped kernel, on a stage s that holds k chunk kc and the metadata of its QK_K block: the metadata
//                          of rows g and g + 8 of group mg, the m16n8k32 A fragment of k step ks as the int8 codes that MMQ
//                          builds, and the scales of row g + 8*h at k step ks as MMQ computes them (float for D4, float2 for DM, D16)
//   vec_dot2(sb, mg, g, p, y)
//                          MMVQ: the partial sum of the upstream MMVQ thread at position p of QK_K block sb (p = 0 .. T - 1, with
//                          T = QK_K/qk * qi/vdr threads per QK_K block) for rows g and g + 8 of group mg, with the operations of
//                          vec_dot_*_q8_1 for each row. y is the first q8_1 block of the QK_K block. Lane (ks, g) of a warp computes
//                          position 4*q + ks of one quad q, so the data of one quad and 16-row group should be contiguous.

// The MMQ arithmetic of a format: how the grouped kernel applies the scales to the int32 sums of the mma.
enum moe_packed_family {
    MOE_PACKED_FAMILY_D4,  // vec_dot_q8_0_q8_1_mma, D4 activations: one scale per 32 values and row, acc += d_x*d_y*sum
    MOE_PACKED_FAMILY_DM,  // vec_dot_q8_1_q8_1_mma, DS4 activations: a scale and a min per 32 values, acc += d_x*d_y*sum + m_x*s_y
    MOE_PACKED_FAMILY_D16, // vec_dot_q8_0_16_q8_1_mma and Q6_K, D4 activations: one scale per 16 values, two m16n8k16 per k step;
                           // each scale must have at most 22 significant bits (the magic epilogue needs 1.5*2^23*scale exact)
};

template <int fmt> struct moe_packed_q {
    // F16 and I8 have no lossless layout
    static constexpr int sb_bytes  = 0;
    static constexpr int tab_words = 0;
    static constexpr int family    = MOE_PACKED_FAMILY_D4;
    struct frag_state {};
};

template <int fmt> static constexpr __host__ __device__ int moe_packed_row_meta_bytes() { return 0; }
template <> constexpr __host__ __device__ int moe_packed_row_meta_bytes<GGML_CUDA_MOE_PACKED_IQ2KS>() { return sizeof(half); }
template <> constexpr __host__ __device__ int moe_packed_row_meta_bytes<GGML_CUDA_MOE_PACKED_IQ3KS>() { return sizeof(half); }
template <> constexpr __host__ __device__ int moe_packed_row_meta_bytes<GGML_CUDA_MOE_PACKED_IQ4KS>() { return sizeof(float); }
template <> constexpr __host__ __device__ int moe_packed_row_meta_bytes<GGML_CUDA_MOE_PACKED_IQ5KS>() { return sizeof(float); }

static constexpr __host__ __device__ bool moe_packed_is_q(const int fmt) {
    return fmt == GGML_CUDA_MOE_PACKED_IQ3S || fmt == GGML_CUDA_MOE_PACKED_IQ4XS || fmt == GGML_CUDA_MOE_PACKED_Q4K ||
        fmt == GGML_CUDA_MOE_PACKED_Q6K || fmt == GGML_CUDA_MOE_PACKED_Q80 || fmt == GGML_CUDA_MOE_PACKED_Q50 ||
        fmt == GGML_CUDA_MOE_PACKED_IQ4NL || fmt == GGML_CUDA_MOE_PACKED_MXFP4 || fmt == GGML_CUDA_MOE_PACKED_Q5K ||
        fmt == GGML_CUDA_MOE_PACKED_Q41 || fmt == GGML_CUDA_MOE_PACKED_Q51 || fmt == GGML_CUDA_MOE_PACKED_Q3K ||
        fmt == GGML_CUDA_MOE_PACKED_IQ3XXS || fmt == GGML_CUDA_MOE_PACKED_IQ2XXS || fmt == GGML_CUDA_MOE_PACKED_IQ2XS ||
        fmt == GGML_CUDA_MOE_PACKED_IQ2S || fmt == GGML_CUDA_MOE_PACKED_IQ2K || fmt == GGML_CUDA_MOE_PACKED_IQ3K ||
        fmt == GGML_CUDA_MOE_PACKED_IQ4K || fmt == GGML_CUDA_MOE_PACKED_IQ5K || fmt == GGML_CUDA_MOE_PACKED_IQ2KS ||
        fmt == GGML_CUDA_MOE_PACKED_IQ3KS || fmt == GGML_CUDA_MOE_PACKED_IQ4KS || fmt == GGML_CUDA_MOE_PACKED_IQ5KS;
}

// Q8_0 chunk: [mg][4 q8 blocks][lane] uint4. Each word contains q8 words t and t + 4 of rows g and g + 8. Metadata is d [128][8] f16.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_Q80> {
    using block_t = block_q8_0;
    static constexpr ggml_type type        = GGML_TYPE_Q8_0;
    static constexpr int chunk_bytes       = MOE_PACKED_Q80_CHUNK;
    static constexpr int sb_bytes           = MOE_PACKED_Q80_BLOCK;
    static constexpr int tab_words          = 0;
    static constexpr int family             = MOE_PACKED_FAMILY_D4;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ib = 4*c + ks;
                const uint4 v = make_uint4(get_int_b2((b0 + ib)->qs, t), get_int_b2((b1 + ib)->qs, t),
                    get_int_b2((b0 + ib)->qs, t + 4), get_int_b2((b1 + ib)->qs, t + 4));
                ((uint4 *) (out + c*MOE_PACKED_Q80_CHUNK))[(mg*4 + ks)*WARP_SIZE + lane] = v;
            }
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_Q80_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
                for (int ib = 0; ib < QK_K/QK8_0; ++ib) {
                    ((half *) meta)[r*(QK_K/QK8_0) + ib] = (h ? b1 : b0)[ib].d;
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b, const int ib) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint4 v = ((const uint4 *) (in + (ib/4)*MOE_PACKED_Q80_CHUNK))[(mg*4 + ib%4)*WARP_SIZE + 4*g + t];
            const uint32_t lo = h ? v.y : v.x;
            const uint32_t hi = h ? v.w : v.z;
            for (int i = 0; i < 4; ++i) {
                b.qs[4*t + i]       = lo >> (8*i);
                b.qs[4*(t + 4) + i] = hi >> (8*i);
            }
        }
        const char * meta = in + 2*MOE_PACKED_Q80_CHUNK;
        b.d = ((const half *) meta)[r*(QK_K/QK8_0) + ib];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) {
        GGML_UNUSED(i);
        return 0;
    }

    struct frag_state {
        float d[2][4];
    };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const char * meta = s + MOE_PACKED_Q80_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                st.d[h][ks] = __half2float(((const half *) meta)[r*(QK_K/QK8_0) + 4*(kc % 2) + ks]);
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        return ((const uint4 *) s)[(mg*4 + ks)*WARP_SIZE + lane];
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) {
        return st.d[h][ks];
    }

    static __device__ __forceinline__ float2 vec_dot2(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int ib  = p / 4;
        const int iqs = 2*(p % 4);
        const block_q8_1 * bq8 = y + ib;
        int u[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = get_int_b4(bq8->qs, iqs + i);
        }
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int v[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int q = iqs + i;
                const int t = q % 4;
                const uint4 w = ((const uint4 *) (sb + (ib/4)*MOE_PACKED_Q80_CHUNK))[(mg*4 + ib%4)*WARP_SIZE + 4*g + t];
                const uint32_t row = h ? (q/4 ? w.w : w.y) : (q/4 ? w.z : w.x);
                v[i] = row;
            }
            const int r = 16*mg + g + 8*h;
            const float d = __half2float(((const half *) (sb + 2*MOE_PACKED_Q80_CHUNK))[r*(QK_K/QK8_0) + ib]);
            res[h] = vec_dot_q8_0_q8_1_impl<float, 2>(v, u, d, __low2half(bq8->ds));
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ2_KS keeps its row scale in the separate row-metadata plane.  The two-bit codes remain in their native row order; the
// per-QK_K extra and scale fields follow the two chunks.  This makes the 32-value D4 groups especially direct to decode.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ2KS> {
    using block_t = block_iq2_ks;
    static constexpr ggml_type type = GGML_TYPE_IQ2_KS;
    static constexpr int chunk_bytes = MOE_PACKED_IQ2KS_CHUNK;
    static constexpr int sb_bytes = MOE_PACKED_IQ2KS_BLOCK;
    static constexpr int tab_words = 0;
    static constexpr int family = MOE_PACKED_FAMILY_D4;
    static constexpr int local_meta_bytes = sb_bytes - 2*chunk_bytes;
    static constexpr int off_scales = MOE_PACKED_I*sizeof(uint16_t);

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*chunk_bytes;
            const int r0 = mg*16 + lane/4;
            const int r1 = r0 + 8;
            // KS rows can start at a two-byte offset, so qs is not necessarily aligned for a uint2 source load.
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                oc[r0*32 + 8*t + i] = b0->qs[c*32 + 8*t + i];
                oc[r1*32 + 8*t + i] = b1->qs[c*32 + 8*t + i];
            }
        }
        if (t == 0) {
            char * meta = out + 2*chunk_bytes;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const block_t & b = h ? b1[0] : b0[0];
                ((uint16_t *) meta)[r] = b.extra;
#pragma unroll
                for (int i = 0; i < 4; ++i) ((uint8_t *) (meta + off_scales))[r*4 + i] = b.scales[i];
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        for (int c = 0; c < 2; ++c) {
#pragma unroll
            for (int i = 0; i < 32; ++i) b.qs[c*32 + i] = ((const uint8_t *) (in + c*chunk_bytes))[r*32 + i];
        }
        const char * meta = in + 2*chunk_bytes;
        b.extra = ((const uint16_t *) meta)[r];
#pragma unroll
        for (int i = 0; i < 4; ++i) b.scales[i] = ((const uint8_t *) (meta + off_scales))[r*4 + i];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }
    struct frag_state { float d[2]; uint16_t extra[2]; uint8_t scales[2][4]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * local = s + chunk_bytes;
        const half * row_meta = (const half *) (local + local_meta_bytes);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            st.d[h] = __half2float(row_meta[r]);
            st.extra[h] = ((const uint16_t *) local)[r];
#pragma unroll
            for (int i = 0; i < 4; ++i) st.scales[h][i] = ((const uint8_t *) (local + off_scales))[r*4 + i];
        }
        return st;
    }

    static __device__ __forceinline__ int group_scale(const frag_state & st, const int h, const int ks) {
        const int ib64 = ks/2;
        // The extra bits are indexed by the superblock-global ib64 (0..3), not the chunk-local one: chunk 1 covers global
        // ib64 2 and 3, so its extra-bit position needs the +2*st.c term that the scale-byte index already has.
        const int gib64 = 2*st.c + ib64;
        const int high = ks & 1;
        const uint8_t sc = st.scales[h][gib64];
        const int lo = high ? sc >> 4 : sc & 0x0F;
        // Reference: ((e >> 4) & 0x10) / ((e >> 5) & 0x10) with e already shifted by 2*gib64, i.e. bit (8 + 2*gib64 + high).
        const int hi = (st.extra[h] >> (8 + 2*gib64 + high)) & 1;
        return lo | (hi << 4);
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        // the extra bit of k step k of the superblock is bit 4*st.c + k of extra
        const int g = lane/4, t = lane%4, a = ks & 1;
        uint32_t r[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t * q = (const uint32_t *) (s + (mg*16 + g + 8*h)*32);
            const uint32_t e = (uint32_t) st.extra[h] >> (4*st.c + a);
#pragma unroll
            for (int hv = 0; hv < 2; ++hv) {
                r[h][hv] = moe_iq2_reg(q[4*hv + t], e & 1, (e >> 2) & 1, ks);
            }
        }
        return make_uint4(r[0][0], r[1][0], r[0][1], r[1][1]);
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) { return st.d[h]*(group_scale(st, h, ks) - 16); }

    static __device__ __forceinline__ float2 vec_dot2_row_meta(const char * sb, const char * row_meta, const int mg, const int g,
            const int p, const block_q8_1 * y) {
        const int c = p/4, ks = p%4, ib64 = ks/2;
        const char * ch = sb + c*chunk_bytes;
        const char * local = sb + 2*chunk_bytes;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = 16*mg + g + 8*h;
            const uint32_t * q = (const uint32_t *) ((const uint8_t *) ch + r*32);
            const uint16_t extra = ((const uint16_t *) local)[r];
            const uint8_t sc = ((const uint8_t *) (local + off_scales))[r*4 + 2*c + ib64];
            const int ls = (ks & 1 ? sc >> 4 : sc & 0x0F) | (((extra >> (8 + 2*(p/2) + (p & 1))) & 1) << 4);
            const uint32_t tab_bit = (extra >> p) & 1;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t v = moe_iq2_reg(q[i], ks < 2 ? tab_bit : 0, ks < 2 ? 0 : tab_bit, ks);
                sumi = ggml_cuda_dp4a((int) v, get_int_b4((y + p)->qs, i), sumi);
            }
            res[h] = __half2float(((const half *) row_meta)[r]) * (ls - 16) * __low2float((y + p)->ds) * sumi;
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ3_KS has four 32-value D4 groups per 128-value chunk.  Its qh bytes are split into the low or high nibble needed by that
// chunk, so the two chunks plus local metadata still occupy exactly one ordinary IQ3_KS block per row.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ3KS> {
    using block_t = block_iq3_ks;
    static constexpr ggml_type type = GGML_TYPE_IQ3_KS;
    static constexpr int chunk_bytes = MOE_PACKED_IQ3KS_CHUNK;
    static constexpr int sb_bytes = MOE_PACKED_IQ3KS_BLOCK;
    static constexpr int tab_words = 0;
    static constexpr int family = MOE_PACKED_FAMILY_D4;
    static constexpr int local_meta_bytes = sb_bytes - 2*chunk_bytes;
    static constexpr int off_scales = MOE_PACKED_I*sizeof(uint16_t);

    static __device__ __forceinline__ uint32_t pack_qh(const uint8_t * qh, const int c, const int t) {
        uint32_t r = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) r |= ((qh[8*t + 2*i] >> (4*c)) & 0x0F) << (8*i) | ((qh[8*t + 2*i + 1] >> (4*c)) & 0x0F) << (8*i + 4);
        return r;
    }

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*chunk_bytes;
            const int r0 = mg*16 + lane/4, r1 = r0 + 8;
            // KS rows can start at a two-byte offset, so qs is not necessarily aligned for a uint2 source load.
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                oc[r0*48 + 8*t + i] = b0->qs[c*32 + 8*t + i];
                oc[r1*48 + 8*t + i] = b1->qs[c*32 + 8*t + i];
            }
            ((uint32_t *) (oc + r0*48 + 32 + 4*t))[0] = pack_qh(b0->qh, c, t);
            ((uint32_t *) (oc + r1*48 + 32 + 4*t))[0] = pack_qh(b1->qh, c, t);
        }
        if (t == 0) {
            char * meta = out + 2*chunk_bytes;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const block_t & b = h ? b1[0] : b0[0];
                ((uint16_t *) meta)[r] = b.extra;
#pragma unroll
                for (int i = 0; i < 4; ++i) ((uint8_t *) (meta + off_scales))[r*4 + i] = b.scales[i];
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
#pragma unroll
        for (int i = 0; i < QK_K/8; ++i) b.qh[i] = 0;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const uint8_t * ch = (const uint8_t *) in + c*chunk_bytes + r*48;
#pragma unroll
            for (int i = 0; i < 32; ++i) b.qs[c*32 + i] = ch[i];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t packed = ch[32 + i];
                b.qh[2*i]     |= (packed & 0x0F) << (4*c);
                b.qh[2*i + 1] |= (packed >> 4) << (4*c);
            }
        }
        const char * meta = in + 2*chunk_bytes;
        b.extra = ((const uint16_t *) meta)[r];
#pragma unroll
        for (int i = 0; i < 4; ++i) b.scales[i] = ((const uint8_t *) (meta + off_scales))[r*4 + i];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }
    struct frag_state { float d[2]; uint16_t extra[2]; uint8_t scales[2][4]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * local = s + chunk_bytes;
        const half * row_meta = (const half *) (local + local_meta_bytes);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            st.d[h] = __half2float(row_meta[r]);
            st.extra[h] = ((const uint16_t *) local)[r];
#pragma unroll
            for (int i = 0; i < 4; ++i) st.scales[h][i] = ((const uint8_t *) (local + off_scales))[r*4 + i];
        }
        return st;
    }

    // The second iq3nl_values table is the first plus 4 in each byte (no byte carries), so an extra bit adds 0x04040404.
    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        // the qh nibble of value v (k step k at bit k) is nibble v%2 of byte 32 + v/2 of the row
        const int g = lane/4, t = lane%4;
        uint32_t r[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const char * row = s + (mg*16 + g + 8*h)*48;
            const uint32_t e = (st.extra[h] >> (8 + 4*st.c + ks)) & 1;
#pragma unroll
            for (int hv = 0; hv < 2; ++hv) {
                const uint32_t qh = moe_nibbles_to_bytes(((const uint16_t *) (row + 32))[4*hv + t]);
                r[h][hv] = moe_iq3_reg(((const uint32_t *) row)[4*hv + t], qh, ks) + e*0x04040404u;
            }
        }
        return make_uint4(r[0][0], r[1][0], r[0][1], r[1][1]);
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) {
        const int ib = 4*st.c + ks;
        const int lo = st.c ? st.scales[h][ks] >> 4 : st.scales[h][ks] & 0x0F;
        return st.d[h]*((lo | (((st.extra[h] >> ib) & 1) << 4)) - 16);
    }

    static __device__ __forceinline__ float2 vec_dot2_row_meta(const char * sb, const char * row_meta, const int mg, const int g,
            const int p, const block_q8_1 * y) {
        const int c = p/4, ks = p%4;
        const char * ch = sb + c*chunk_bytes;
        const char * local = sb + 2*chunk_bytes;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = 16*mg + g + 8*h;
            const uint8_t * q = (const uint8_t *) ch + r*48;
            const uint16_t extra = ((const uint16_t *) local)[r];
            const uint8_t scale = ((const uint8_t *) (local + off_scales))[r*4 + (p < 4 ? p : p - 4)];
            const int ls = (p < 4 ? scale & 0x0F : scale >> 4) | (((extra >> p) & 1) << 4);
            const uint32_t shift = ((extra >> (8 + p)) & 1)*0x04040404u;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t qh = moe_nibbles_to_bytes(((const uint16_t *) (q + 32))[i]);
                const uint32_t v = moe_iq3_reg(((const uint32_t *) q)[i], qh, ks) + shift;
                sumi = ggml_cuda_dp4a((int) v, get_int_b4((y + p)->qs, i), sumi);
            }
            res[h] = __half2float(((const half *) row_meta)[r]) * (ls - 16) * __low2float((y + p)->ds) * sumi;
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ5_KS uses the same row-scale plane as IQ4_KS.  A chunk holds its 64 qs bytes and a packed 16-byte nibble plane of qh:
// the low four high-bit planes are in chunk zero and the high four are in chunk one.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ5KS> {
    using block_t = block_iq5_ks;
    static constexpr ggml_type type = GGML_TYPE_IQ5_KS;
    static constexpr int chunk_bytes = MOE_PACKED_IQ5KS_CHUNK;
    static constexpr int sb_bytes = MOE_PACKED_IQ5KS_BLOCK;
    static constexpr int tab_words = 0;
    static constexpr int family = MOE_PACKED_FAMILY_D4;
    static constexpr int local_meta_bytes = sb_bytes - 2*chunk_bytes;

    static __device__ __forceinline__ uint32_t pack_qh(const uint8_t * qh, const int c, const int t) {
        uint32_t r = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) r |= ((qh[8*t + 2*i] >> (4*c)) & 0x0F) << (8*i) | ((qh[8*t + 2*i + 1] >> (4*c)) & 0x0F) << (8*i + 4);
        return r;
    }

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*chunk_bytes;
            const int r0 = mg*16 + lane/4, r1 = r0 + 8;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ((uint32_t *) (oc + r0*80 + 16*t + 4*i))[0] = ((const uint32_t *) (b0->qs + c*64 + 16*t))[i];
                ((uint32_t *) (oc + r1*80 + 16*t + 4*i))[0] = ((const uint32_t *) (b1->qs + c*64 + 16*t))[i];
            }
            ((uint32_t *) (oc + r0*80 + 64 + 4*t))[0] = pack_qh(b0->qh, c, t);
            ((uint32_t *) (oc + r1*80 + 64 + 4*t))[0] = pack_qh(b1->qh, c, t);
        }
        if (t == 0) {
            uint8_t * scales = (uint8_t *) (out + 2*chunk_bytes);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const block_t & b = h ? b1[0] : b0[0];
#pragma unroll
                for (int i = 0; i < 8; ++i) scales[r*8 + i] = b.scales[i];
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
#pragma unroll
        for (int i = 0; i < QK_K/8; ++i) b.qh[i] = 0;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const uint8_t * ch = (const uint8_t *) in + c*chunk_bytes + r*80;
#pragma unroll
            for (int i = 0; i < 64; ++i) b.qs[c*64 + i] = ch[i];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t packed = ch[64 + i];
                b.qh[2*i]     |= (packed & 0x0F) << (4*c);
                b.qh[2*i + 1] |= (packed >> 4) << (4*c);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) b.scales[i] = ((const uint8_t *) (in + 2*chunk_bytes))[r*8 + i];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }
    struct frag_state { float d[2]; uint8_t scales[2][8]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const uint8_t * local = (const uint8_t *) s + chunk_bytes;
        const float * row_meta = (const float *) (local + local_meta_bytes);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            st.d[h] = row_meta[r];
#pragma unroll
            for (int i = 0; i < 8; ++i) st.scales[h][i] = local[r*8 + i];
        }
        return st;
    }

    // As IQ5_K: the qs words hold the indices of the k steps ks & ~1 (low nibbles) and ks | 1 (high nibbles). The fifth index bit of
    // value v at k step ks is bit ks of nibble v%2 of byte 64 + v/2, which is bit 2 of a prmt selector nibble that picks the byte
    // of the first or the second half of iq5nl_values; the shift for the second half is 2 (the extra bit is bit 0 of the scale).
    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        const int g = lane/4, t = lane%4;
        uint32_t r[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const char * row = s + (mg*16 + g + 8*h)*80;
            const uint32_t shift = (st.scales[h][4*st.c + ks] & 1)*0x02020202u;
#pragma unroll
            for (int hv = 0; hv < 2; ++hv) {
                int2 a, b;
                moe_iq5_lookup(((const uint32_t *) (row + 32*(ks/2)))[4*hv + t], a, b);
                const uint32_t sel = 0x3210 | (((((const uint16_t *) (row + 64))[4*hv + t] >> ks) & 0x1111) << 2);
                r[h][hv] = moe_iq5_value(moe_prmt(ks & 1 ? a.y : a.x, ks & 1 ? b.y : b.x, sel), shift);
            }
        }
        return make_uint4(r[0][0], r[1][0], r[0][1], r[1][1]);
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) {
        return st.d[h]*(int(st.scales[h][4*st.c + ks] & 254) - 127);
    }

    static __device__ __forceinline__ float2 vec_dot2_row_meta(const char * sb, const char * row_meta, const int mg, const int g,
            const int p, const block_q8_1 * y) {
        const int c = p/4, ks = p%4;
        const char * ch = sb + c*chunk_bytes;
        const uint8_t * local = (const uint8_t *) sb + 2*chunk_bytes;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = 16*mg + g + 8*h;
            const uint8_t * q = (const uint8_t *) ch + r*80;
            const uint8_t scale = local[r*8 + p];
            const uint32_t shift = (scale & 1)*0x02020202u;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int2 a, b;
                moe_iq5_lookup(((const uint32_t *) (q + 32*(ks/2)))[i], a, b);
                const uint32_t sel = 0x3210 | (((((const uint16_t *) (q + 64))[i] >> ks) & 0x1111) << 2);
                const uint32_t v = moe_iq5_value(moe_prmt(ks & 1 ? a.y : a.x, ks & 1 ? b.y : b.x, sel), shift);
                sumi = ggml_cuda_dp4a((int) v, get_int_b4((y + p)->qs, i), sumi);
            }
            res[h] = ((const float *) row_meta)[r] * (int(scale & 254) - 127) * __low2float((y + p)->ds) * sumi;
        }
        return make_float2(res[0], res[1]);
    }
};

// Q5_0 chunk: ql [mg][4 q5 blocks][lane] uint2, qh bytes [mg][4 q5 blocks][lane] uint16. Each ql word contains the low and high
// nibbles of rows g and g + 8; the corresponding packed qh byte uses its low and high nibbles for the two halves. Metadata is d [128][8] f16.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_Q50> {
    using block_t = block_q5_0;
    static constexpr ggml_type type        = GGML_TYPE_Q5_0;
    static constexpr int chunk_bytes       = MOE_PACKED_Q50_CHUNK;
    static constexpr int sb_bytes           = MOE_PACKED_Q50_BLOCK;
    static constexpr int tab_words          = 0;
    static constexpr int family             = MOE_PACKED_FAMILY_D4;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_Q50_CHUNK;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ib = 4*c + ks;
                ((uint2 *) oc)[(mg*4 + ks)*WARP_SIZE + lane] = make_uint2(get_int_b2((b0 + ib)->qs, t), get_int_b2((b1 + ib)->qs, t));
                ((uint16_t *) (oc + MOE_PACKED_Q50_H))[(mg*4 + ks)*WARP_SIZE + lane] =
                    (uint16_t) moe_q50_bits(get_int_b2((b0 + ib)->qh, 0), t) |
                    (uint16_t) moe_q50_bits(get_int_b2((b1 + ib)->qh, 0), t) << 8;
            }
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_Q50_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
                for (int ib = 0; ib < QK_K/QK5_0; ++ib) {
                    ((half *) meta)[r*(QK_K/QK5_0) + ib] = (h ? b1 : b0)[ib].d;
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b, const int ib) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
        uint32_t qh = 0;
        for (int t = 0; t < 4; ++t) {
            const uint2 q = ((const uint2 *) (in + (ib/4)*MOE_PACKED_Q50_CHUNK))[(mg*4 + ib%4)*WARP_SIZE + 4*g + t];
            const uint16_t H = ((const uint16_t *) (in + (ib/4)*MOE_PACKED_Q50_CHUNK + MOE_PACKED_Q50_H))[(mg*4 + ib%4)*WARP_SIZE + 4*g + t];
            const uint32_t qword = h ? q.y : q.x;
            const uint8_t hb = h ? H >> 8 : H & 0xFF;
            for (int i = 0; i < 4; ++i) {
                b.qs[4*t + i] = qword >> (8*i);
                qh |= ((hb >> i) & 1) << (4*t + i);
                qh |= ((hb >> (4 + i)) & 1) << (16 + 4*t + i);
            }
        }
        for (int i = 0; i < 4; ++i) {
            b.qh[i] = qh >> (8*i);
        }
        const char * meta = in + 2*MOE_PACKED_Q50_CHUNK;
        b.d = ((const half *) meta)[r*(QK_K/QK5_0) + ib];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) {
        GGML_UNUSED(i);
        return 0;
    }

    struct frag_state {
        float d[2][4];
    };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const char * meta = s + MOE_PACKED_Q50_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                st.d[h][ks] = __half2float(((const half *) meta)[r*(QK_K/QK5_0) + 4*(kc % 2) + ks]);
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const uint2 q = ((const uint2 *) s)[(mg*4 + ks)*WARP_SIZE + lane];
        const uint16_t H = ((const uint16_t *) (s + MOE_PACKED_Q50_H))[(mg*4 + ks)*WARP_SIZE + lane];
        const uint8_t h0 = H & 0xFF;
        const uint8_t h1 = H >> 8;
        return make_uint4(moe_q5_0_code(q.x, h0, 0), moe_q5_0_code(q.y, h1, 0), moe_q5_0_code(q.x, h0, 1), moe_q5_0_code(q.y, h1, 1));
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) {
        return st.d[h][ks];
    }

    static __device__ __forceinline__ float2 vec_dot2(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int ib  = p / 2;
        const int iqs = 2*(p % 2);
        int vl[2];
        int vh[2];
        int u[4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = iqs + i;
            u[2*i + 0] = get_int_b4((y + ib)->qs, q);
            u[2*i + 1] = get_int_b4((y + ib)->qs, q + QI5_0);
        }
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int q = iqs + i;
                const uint2 v = ((const uint2 *) (sb + (ib/4)*MOE_PACKED_Q50_CHUNK))[(mg*4 + ib%4)*WARP_SIZE + 4*g + q];
                const uint16_t H = ((const uint16_t *) (sb + (ib/4)*MOE_PACKED_Q50_CHUNK + MOE_PACKED_Q50_H))[(mg*4 + ib%4)*WARP_SIZE + 4*g + q];
                vl[i] = h ? (int) v.y : (int) v.x;
                vh[i] = moe_q50_vh(h ? H >> 8 : H & 0xFF);
            }
            const int r = 16*mg + g + 8*h;
            const float d = __half2float(((const half *) (sb + 2*MOE_PACKED_Q50_CHUNK))[r*(QK_K/QK5_0) + ib]);
            res[h] = vec_dot_q5_0_q8_1_impl<2>(vl, vh, u, d, (y + ib)->ds);
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ3S chunk: Q [mg][4 k steps][lane] u32, byte j is the qs byte of the grid group in register j; S [mg][2 k step pairs][lane] u32,
// bit 8*b + 7 - (4*(ks%2) + j) is the sign of value b of register j; H [mg][lane] u16, bit 4*ks + j is the qh bit (bit 8 of the
// grid index) of register j. Metadata: d [128] f16, the 4 scale bytes of each row as one u32 [128].
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ3S> {
    using block_t = block_iq3_s;
    static constexpr ggml_type type        = GGML_TYPE_IQ3_S;
    static constexpr int       chunk_bytes = MOE_PACKED_IQ3S_CHUNK;
    static constexpr int       sb_bytes    = MOE_PACKED_IQ3S_BLOCK;
    static constexpr int       tab_words   = 512;
    static constexpr int       family      = MOE_PACKED_FAMILY_D4;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const block_t * b[2] = { b0, b1 };
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_IQ3S_CHUNK;
            uint32_t S[2] = { 0, 0 };
            uint32_t H    = 0;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                uint32_t Q = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const block_t * bj = b[j % 2];
                    const int q  = 32*c + 8*ks + 4*(j/2) + t; // grid group of register j in the QK_K block
                    const int jj = 4*(ks % 2) + j;
                    Q |= (uint32_t) bj->qs[q] << (8*j);
                    H |= (uint32_t) ((bj->qh[q/8] >> (q % 8)) & 1) << (4*ks + j);
                    const int sg = bj->signs[q/2] >> (4*(q % 2));
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        S[ks/2] |= (uint32_t) ((sg >> v) & 1) << (8*v + 7 - jj);
                    }
                }
                ((uint32_t *) oc)[(mg*4 + ks)*WARP_SIZE + lane] = Q;
            }
            ((uint32_t *) (oc + MOE_PACKED_IQ3S_S))[(mg*2 + 0)*WARP_SIZE + lane] = S[0];
            ((uint32_t *) (oc + MOE_PACKED_IQ3S_S))[(mg*2 + 1)*WARP_SIZE + lane] = S[1];
            ((uint16_t *) (oc + MOE_PACKED_IQ3S_H))[mg*WARP_SIZE + lane] = H;
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_IQ3S_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const uint8_t * sc = b[h]->scales;
                ((half     *) meta)[r] = b[h]->d;
                ((uint32_t *) (meta + MOE_PACKED_I*sizeof(half)))[r] = sc[0] | sc[1] << 8 | sc[2] << 16 | (uint32_t) sc[3] << 24;
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
        for (int i = 0; i < QK_K/32; ++i) {
            b.qh[i] = 0;
        }
        for (int i = 0; i < QK_K/8; ++i) {
            b.signs[i] = 0;
        }
        for (int q = 0; q < QK_K/4; ++q) {
            const int c    = q / 32;
            const int ks   = (q % 32) / 8;
            const int j    = h + 2*((q % 8) / 4);
            const int lane = 4*g + q % 4;
            const int jj   = 4*(ks % 2) + j;
            const char * ic = in + c*MOE_PACKED_IQ3S_CHUNK;
            const uint32_t Q = ((const uint32_t *) ic)[(mg*4 + ks)*WARP_SIZE + lane];
            const uint32_t S = ((const uint32_t *) (ic + MOE_PACKED_IQ3S_S))[(mg*2 + ks/2)*WARP_SIZE + lane];
            const uint32_t H = ((const uint16_t *) (ic + MOE_PACKED_IQ3S_H))[mg*WARP_SIZE + lane];
            b.qs[q] = Q >> (8*j);
            b.qh[q/8] |= ((H >> (4*ks + j)) & 1) << (q % 8);
            for (int v = 0; v < 4; ++v) {
                b.signs[q/2] |= ((S >> (8*v + 7 - jj)) & 1) << (4*(q % 2) + v);
            }
        }
        const char * meta = in + 2*MOE_PACKED_IQ3S_CHUNK;
        b.d = ((const half *) meta)[r];
        const uint32_t sc = ((const uint32_t *) (meta + MOE_PACKED_I*sizeof(half)))[r];
        for (int i = 0; i < IQ3S_N_SCALE; ++i) {
            b.scales[i] = sc >> (8*i);
        }
    }

    // iq3s_grid - 0x01010101: the grid bytes g are odd, so (g - 1) | 1 = g and ~(g - 1) | 1 = -g
    static __device__ __forceinline__ uint32_t tab(const int i) {
        return iq3s_grid[i] - 0x01010101;
    }

    struct frag_state {
        uint32_t qh;    // the H word of the lane
        float    d[2];  // rows g and g + 8
        uint32_t sc[2]; // their scale nibbles of the chunk
    };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.qh = ((const uint16_t *) (s + MOE_PACKED_IQ3S_H))[mg*WARP_SIZE + lane];
        const char * meta = s + MOE_PACKED_IQ3S_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            st.d[h]  = __half2float(((const half *) meta)[r]);
            st.sc[h] = ((const uint32_t *) (meta + MOE_PACKED_I*sizeof(half)))[r] >> (16*(kc % 2));
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * grid, const frag_state & st, const int mg, const int ks,
            const int lane) {
        const uint32_t q  = ((const uint32_t *) s)[(mg*4 + ks)*WARP_SIZE + lane];
        const uint32_t sg = ((const uint32_t *) (s + MOE_PACKED_IQ3S_S))[(mg*2 + ks/2)*WARP_SIZE + lane];
        uint32_t v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int      b  = 4*ks + j;
            const uint32_t hi = (b <= 8 ? st.qh << (8 - b) : st.qh >> (b - 8)) & 0x100;
            const uint32_t w  = grid[__byte_perm(q, 0, 0x4440 | j) | hi];
            v[j] = (w ^ moe_iq3s_sign_mask(sg << (4*(ks % 2) + j))) | 0x01010101;
        }
        return make_uint4(v[0], v[1], v[2], v[3]);
    }

    // ls*d as in MMQ, exact in fp32; ls = 1 + 2*scale < 2^23 is converted by an OR into the mantissa of 2^23
    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) {
        const uint32_t ls = 1 + 2*((st.sc[h] >> (4*ks)) & 0xF);
        return st.d[h]*(__uint_as_float(0x4B000000 | ls) - 8388608.0f);
    }

    static __device__ __forceinline__ float2 vec_dot2(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        return vec_dot_iq3s_packed_q8_1(sb, mg, g, p, y + p, iq3s_grid);
    }

    static __device__ __forceinline__ float2 vec_dot2_grid(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y, const uint32_t * grid) {
        return vec_dot_iq3s_packed_q8_1<true>(sb, mg, g, p, y + p, grid);
    }
};

// IQ4XS chunk: words [mg][4 k steps][2 halves h][lane] u32. The word of lane (g, t) and half h at k step ks of chunk c is int t of the
// qs of sub-block 4*c + ks of row 16*mg + g + 8*h with the nibbles in value order (moe_iq4xs_word): its low 16 bits give register h,
// its high 16 bits register h + 2. Metadata: d [128] f16, then the eight 6-bit scales ls = scales_l nibble | scales_h bits << 4 of
// each row as 48 bits: A [128] u32 (scales 0-3 in bits 0-23, scales 4-7 in bits 24-31 and in B) and B [128] u16.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ4XS> {
    using block_t = block_iq4_xs;
    static constexpr ggml_type type        = GGML_TYPE_IQ4_XS;
    static constexpr int       chunk_bytes = MOE_PACKED_IQ4XS_CHUNK;
    static constexpr int       sb_bytes    = MOE_PACKED_IQ4XS_BLOCK;
    static constexpr int       tab_words   = 0;
    static constexpr int       family      = MOE_PACKED_FAMILY_D4;

    static constexpr int off_a = 2*MOE_PACKED_IQ4XS_CHUNK + MOE_PACKED_I*sizeof(half);
    static constexpr int off_b = off_a + MOE_PACKED_I*sizeof(uint32_t);

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const block_t * b[2] = { b0, b1 };
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    ((uint32_t *) (out + c*MOE_PACKED_IQ4XS_CHUNK))[((mg*4 + ks)*2 + h)*WARP_SIZE + lane] =
                        moe_iq4xs_word(get_int_b4(b[h]->qs, 4*(4*c + ks) + t));
                }
            }
        }
        if (t == 0) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                uint64_t ls = 0;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    const uint32_t l = ((b[h]->scales_l[ib/2] >> (4*(ib % 2))) & 0xF) | ((b[h]->scales_h >> (2*ib)) & 3) << 4;
                    ls |= (uint64_t) l << (6*ib);
                }
                ((half     *) (out + 2*MOE_PACKED_IQ4XS_CHUNK))[r] = b[h]->d;
                ((uint32_t *) (out + off_a))[r] = (uint32_t) ls;
                ((uint16_t *) (out + off_b))[r] = (uint16_t) (ls >> 32);
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            for (int t = 0; t < 4; ++t) {
                const uint32_t w = ((const uint32_t *) (in + (ib/4)*MOE_PACKED_IQ4XS_CHUNK))[((mg*4 + ib % 4)*2 + h)*WARP_SIZE + 4*g + t];
                const uint32_t q = moe_iq4xs_qs(w);
                for (int i = 0; i < 4; ++i) {
                    b.qs[16*ib + 4*t + i] = q >> (8*i);
                }
            }
        }
        b.d = ((const half *) (in + 2*MOE_PACKED_IQ4XS_CHUNK))[r];
        const uint64_t ls = ((const uint32_t *) (in + off_a))[r] | (uint64_t) ((const uint16_t *) (in + off_b))[r] << 32;
        uint32_t sh = 0;
        for (int i = 0; i < QK_K/64; ++i) {
            b.scales_l[i] = 0;
        }
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint32_t l = (ls >> (6*ib)) & 0x3F;
            b.scales_l[ib/2] |= (l & 0xF) << (4*(ib % 2));
            sh |= (l >> 4) << (2*ib);
        }
        b.scales_h = sh;
    }

    static __device__ __forceinline__ uint32_t tab(const int i) {
        GGML_UNUSED(i);
        return 0;
    }

    struct frag_state {
        float    d[2];  // rows g and g + 8
        uint32_t ls[2]; // their 4 six-bit scales of the chunk
    };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            const uint32_t A = ((const uint32_t *) (s + off_a - MOE_PACKED_IQ4XS_CHUNK))[r];
            const uint32_t B = ((const uint16_t *) (s + off_b - MOE_PACKED_IQ4XS_CHUNK))[r];
            st.d[h]  = __half2float(((const half *) (s + MOE_PACKED_IQ4XS_CHUNK))[r]);
            st.ls[h] = moe_iq4xs_scales(A, B, kc % 2);
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const uint32_t * w = (const uint32_t *) s + (mg*4 + ks)*2*WARP_SIZE + lane;
        const int2 v0 = moe_iq4xs_decode(w[0]);
        const int2 v1 = moe_iq4xs_decode(w[WARP_SIZE]);
        return make_uint4(v0.x, v1.x, v0.y, v1.y);
    }

    // d*(ls - 32) as in MMQ; ls - 32 is exact as 1.5*2^23 + ls - (1.5*2^23 + 32)
    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) {
        const uint32_t ls = (st.ls[h] >> (6*ks)) & 0x3F;
        return st.d[h]*(__uint_as_float(0x4B400000 | ls) - 12582944.0f);
    }

    static __device__ __forceinline__ float2 vec_dot2(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        return vec_dot_iq4xs_packed_q8_1(sb, mg, g, p, y + p);
    }
};

// IQ4_NL chunk: words [mg][4 blocks][2 rows][lane] u32, in the same value-order nibble layout as IQ4_XS. Metadata is d [128][8] f16.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ4NL> {
    using block_t = block_iq4_nl;
    static constexpr ggml_type type        = GGML_TYPE_IQ4_NL;
    static constexpr int chunk_bytes       = MOE_PACKED_IQ4NL_CHUNK;
    static constexpr int sb_bytes           = MOE_PACKED_IQ4NL_BLOCK;
    static constexpr int tab_words          = 0;
    static constexpr int family             = MOE_PACKED_FAMILY_D4;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const block_t * b[2] = { b0, b1 };
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    ((uint32_t *) (out + c*MOE_PACKED_IQ4NL_CHUNK))[((mg*4 + ks)*2 + h)*WARP_SIZE + lane] =
                        moe_iq4xs_word(get_int_b2((b[h] + 4*c + ks)->qs, t));
                }
            }
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_IQ4NL_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
                for (int ib = 0; ib < QK_K/QK4_NL; ++ib) {
                    ((half *) meta)[r*(QK_K/QK4_NL) + ib] = (b[h] + ib)->d;
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b, const int ib) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
        for (int t = 0; t < 4; ++t) {
            const uint32_t w = ((const uint32_t *) (in + (ib/4)*MOE_PACKED_IQ4NL_CHUNK))[((mg*4 + ib%4)*2 + h)*WARP_SIZE + 4*g + t];
            const uint32_t q = moe_iq4xs_qs(w);
            for (int i = 0; i < 4; ++i) {
                b.qs[4*t + i] = q >> (8*i);
            }
        }
        b.d = ((const half *) (in + 2*MOE_PACKED_IQ4NL_CHUNK))[r*(QK_K/QK4_NL) + ib];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) {
        GGML_UNUSED(i);
        return 0;
    }

    struct frag_state {
        float d[2][4];
    };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const char * meta = s + MOE_PACKED_IQ4NL_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                st.d[h][ks] = __half2float(((const half *) meta)[r*(QK_K/QK4_NL) + 4*(kc % 2) + ks]);
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const uint32_t * w = (const uint32_t *) s + (mg*4 + ks)*2*WARP_SIZE + lane;
        const int2 v0 = moe_iq4_decode(w[0], kvalues_iq4nl);
        const int2 v1 = moe_iq4_decode(w[WARP_SIZE], kvalues_iq4nl);
        return make_uint4(v0.x, v1.x, v0.y, v1.y);
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) {
        return st.d[h][ks];
    }

    static __device__ __forceinline__ float2 vec_dot2(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int ib  = p / 2;
        const int iqs = 2*(p % 2);
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int sumi = 0;
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int q = iqs + l;
                const uint32_t w = ((const uint32_t *) (sb + (ib/4)*MOE_PACKED_IQ4NL_CHUNK))[((mg*4 + ib%4)*2 + h)*WARP_SIZE + 4*g + q];
                const int2 v = moe_iq4_decode(w, kvalues_iq4nl);
                sumi = ggml_cuda_dp4a(v.x, get_int_b4((y + ib)->qs, q), sumi);
                sumi = ggml_cuda_dp4a(v.y, get_int_b4((y + ib)->qs, q + QI4_NL), sumi);
            }
            const int r = 16*mg + g + 8*h;
            const float d = __half2float(((const half *) (sb + 2*MOE_PACKED_IQ4NL_CHUNK))[r*(QK_K/QK4_NL) + ib]);
            res[h] = d * __low2float((y + ib)->ds) * sumi;
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ4_KS has the IQ4_NL-style 32-value decode, but its d is stored once for the complete GGUF row. The packed row metadata
// plane holds d[128] once; the eight scale bytes of each QK_K block stay next to the packed payload of that block.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ4KS> {
    using block_t = block_iq4_ks;
    static constexpr ggml_type type           = GGML_TYPE_IQ4_KS;
    static constexpr int chunk_bytes          = MOE_PACKED_IQ4KS_CHUNK;
    static constexpr int sb_bytes             = MOE_PACKED_IQ4KS_BLOCK;
    static constexpr int tab_words            = 0;
    static constexpr int family               = MOE_PACKED_FAMILY_D4;
    static constexpr int row_meta_bytes       = sizeof(float);
    static constexpr int local_meta_bytes     = sb_bytes - 2*chunk_bytes;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const block_t * b[2] = { b0, b1 };
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    ((uint32_t *) (out + c*chunk_bytes))[((mg*4 + ks)*2 + h)*WARP_SIZE + lane] =
                        moe_iq4xs_word(get_int_b4(b[h]->qs, 4*(4*c + ks) + t));
                }
            }
        }
        if (t == 0) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    ((uint8_t *) (out + 2*chunk_bytes))[r*(QK_K/32) + ib] = b[h]->scales[ib];
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
#pragma unroll
        for (int ib = 0; ib < QK_K/32; ++ib) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t w = ((const uint32_t *) (in + (ib/4)*chunk_bytes))[(mg*4 + ib%4)*2*WARP_SIZE + h*WARP_SIZE + 4*g + t];
                const uint32_t q = moe_iq4xs_qs(w);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    b.qs[16*ib + 4*t + i] = q >> (8*i);
                }
            }
            b.scales[ib] = ((const uint8_t *) (in + 2*chunk_bytes))[r*(QK_K/32) + ib];
        }
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { float d[2]; uint8_t scales[2][4]; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const uint8_t * local = (const uint8_t *) (s + chunk_bytes);
        const float * row_meta = (const float *) (local + local_meta_bytes);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            st.d[h] = row_meta[r];
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                st.scales[h][ks] = local[r*(QK_K/32) + 4*(kc % 2) + ks];
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const uint32_t * w = (const uint32_t *) s + (mg*4 + ks)*2*WARP_SIZE + lane;
        // the second iq4k_values table is the first plus 4 in each byte (no byte carries)
        const uint32_t sh0 = (st.scales[0][ks] & 1)*0x04040404u;
        const uint32_t sh1 = (st.scales[1][ks] & 1)*0x04040404u;
        const int2 v0 = moe_iq4_decode(w[0], iq4k_values);
        const int2 v1 = moe_iq4_decode(w[WARP_SIZE], iq4k_values);
        return make_uint4(v0.x + sh0, v1.x + sh1, v0.y + sh0, v1.y + sh1);
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) {
        return st.d[h]*(int(st.scales[h][ks] & 254) - 127);
    }

    static __device__ __forceinline__ float2 vec_dot2_row_meta(const char * sb, const char * row_meta, const int mg, const int g,
            const int p, const block_q8_1 * y) {
        const int ib = p;
        const int c = ib/4;
        const int ks = ib%4;
        const uint8_t * local = (const uint8_t *) (sb + 2*chunk_bytes);
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = 16*mg + g + 8*h;
            const uint32_t * q = (const uint32_t *) (sb + c*chunk_bytes) + (mg*4 + ks)*2*WARP_SIZE + h*WARP_SIZE + 4*g;
            const uint8_t scale = local[r*(QK_K/32) + ib];
            const uint32_t sh = (scale & 1)*0x04040404u;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int2 v = moe_iq4_decode(q[i], iq4k_values);
                sumi = ggml_cuda_dp4a(v.x + sh, get_int_b4((y + ib)->qs, i), sumi);
                sumi = ggml_cuda_dp4a(v.y + sh, get_int_b4((y + ib)->qs, i + 4), sumi);
            }
            const float d = ((const float *) row_meta)[r];
            res[h] = d*(int(scale & 254) - 127)*__low2float((y + ib)->ds)*sumi;
        }
        return make_float2(res[0], res[1]);
    }
};

// MXFP4 chunk: words [mg][4 blocks][2 rows][lane] u32, in the same value-order nibble layout as IQ4_XS. Metadata is e8m0 [128][8].
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_MXFP4> {
    using block_t = block_mxfp4;
    static constexpr ggml_type type        = GGML_TYPE_MXFP4;
    static constexpr int chunk_bytes       = MOE_PACKED_MXFP4_CHUNK;
    static constexpr int sb_bytes           = MOE_PACKED_MXFP4_BLOCK;
    static constexpr int tab_words          = 0;
    static constexpr int family             = MOE_PACKED_FAMILY_D4;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const block_t * b[2] = { b0, b1 };
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    ((uint32_t *) (out + c*MOE_PACKED_MXFP4_CHUNK))[((mg*4 + ks)*2 + h)*WARP_SIZE + lane] =
                        moe_iq4xs_word(get_int_b1((b[h] + 4*c + ks)->qs, t));
                }
            }
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_MXFP4_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
                for (int ib = 0; ib < QK_K/QK_MXFP4; ++ib) {
                    ((uint8_t *) meta)[r*(QK_K/QK_MXFP4) + ib] = (b[h] + ib)->e;
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b, const int ib) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
        for (int t = 0; t < 4; ++t) {
            const uint32_t w = ((const uint32_t *) (in + (ib/4)*MOE_PACKED_MXFP4_CHUNK))[((mg*4 + ib%4)*2 + h)*WARP_SIZE + 4*g + t];
            const uint32_t q = moe_iq4xs_qs(w);
            for (int i = 0; i < 4; ++i) {
                b.qs[4*t + i] = q >> (8*i);
            }
        }
        b.e = ((const uint8_t *) (in + 2*MOE_PACKED_MXFP4_CHUNK))[r*(QK_K/QK_MXFP4) + ib];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) {
        GGML_UNUSED(i);
        return 0;
    }

    struct frag_state {
        uint8_t e[2][4];
    };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const char * meta = s + MOE_PACKED_MXFP4_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                st.e[h][ks] = ((const uint8_t *) meta)[r*(QK_K/QK_MXFP4) + 4*(kc % 2) + ks];
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const uint32_t * w = (const uint32_t *) s + (mg*4 + ks)*2*WARP_SIZE + lane;
        const int2 v0 = moe_iq4_decode(w[0], kvalues_mxfp4);
        const int2 v1 = moe_iq4_decode(w[WARP_SIZE], kvalues_mxfp4);
        return make_uint4(v0.x, v1.x, v0.y, v1.y);
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) {
        return ggml_cuda_e8m0_to_fp32(st.e[h][ks]) * 0.5f;
    }

    static __device__ __forceinline__ float2 vec_dot2(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int ib  = p / 2;
        const int iqs = 2*(p % 2);
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int sumi = 0;
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int q = iqs + l;
                const uint32_t w = ((const uint32_t *) (sb + (ib/4)*MOE_PACKED_MXFP4_CHUNK))[((mg*4 + ib%4)*2 + h)*WARP_SIZE + 4*g + q];
                const int2 v = moe_iq4_decode(w, kvalues_mxfp4);
                sumi = ggml_cuda_dp4a(v.x, get_int_b4((y + ib)->qs, q), sumi);
                sumi = ggml_cuda_dp4a(v.y, get_int_b4((y + ib)->qs, q + QI_MXFP4), sumi);
            }
            const int r = 16*mg + g + 8*h;
            const uint8_t e = ((const uint8_t *) (sb + 2*MOE_PACKED_MXFP4_CHUNK))[r*(QK_K/QK_MXFP4) + ib];
            const float d = ggml_cuda_e8m0_to_fp32(e) * 0.5f;
            res[h] = d * __low2float((y + ib)->ds) * sumi;
        }
        return make_float2(res[0], res[1]);
    }
};

// Q4K chunk: [mg][2 groups of 64 values jj][lane] uint4. For group j = 2*c + jj of chunk c, lane (g, t) holds the qs ints 8*j + t and
// 8*j + t + 4 of rows 16*mg + g and + 8 as { row g, row g + 8, row g, row g + 8 }: the low nibbles are the A fragment of k step 2*jj,
// the high nibbles that of k step 2*jj + 1, and the lane holds the ints of MMVQ position 4*j + t of both rows. Metadata: dm [128]
// half2, then the 6-bit scales and mins of each row as 96 bits. For chunk c they are the 48 bits sc0-3 | m0-3 << 24 of its 4
// sub-blocks: bits 0-31 in W_c [128] u32 (c = 0, 1), bits 32-47 in half c of X [128] u32.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_Q4K> {
    using block_t = block_q4_K;
    static constexpr ggml_type type        = GGML_TYPE_Q4_K;
    static constexpr int       chunk_bytes = MOE_PACKED_Q4K_CHUNK;
    static constexpr int       sb_bytes    = MOE_PACKED_Q4K_BLOCK;
    static constexpr int       tab_words   = 0;
    static constexpr int       family      = MOE_PACKED_FAMILY_DM;

    // W_0, W_1 and X, from the start of the metadata
    static constexpr int off_w = MOE_PACKED_I*sizeof(half2);
    static constexpr int off_x = off_w + 2*MOE_PACKED_I*sizeof(uint32_t);

    // the min of sub-block k (0 .. 3) of a chunk from its 48 bits lo | hi << 32
    static __device__ __forceinline__ uint8_t min6(const uint32_t lo, const uint32_t hi, const int k) {
        return (((uint64_t) hi << 32 | lo) >> (24 + 6*k)) & 63;
    }

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const block_t * b[2] = { b0, b1 };
        const int t = lane % 4;
#pragma unroll
        for (int j = 0; j < QK_K/64; ++j) {
            const uint4 v = make_uint4(get_int_b4(b0->qs, 8*j + t), get_int_b4(b1->qs, 8*j + t),
                get_int_b4(b0->qs, 8*j + t + 4), get_int_b4(b1->qs, 8*j + t + 4));
            ((uint4 *) (out + (j/2)*MOE_PACKED_Q4K_CHUNK))[(mg*2 + j % 2)*WARP_SIZE + lane] = v;
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_Q4K_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const uint8_t * q = b[h]->scales;
                uint64_t v[2] = { 0, 0 };
                for (int k = 0; k < QK_K/32; ++k) {
                    // get_scale_min_k4
                    const uint64_t sc = k < 4 ? q[k] & 63     : (q[k + 4] & 0xF) | (q[k - 4] >> 6) << 4;
                    const uint64_t m  = k < 4 ? q[k + 4] & 63 : (q[k + 4] >>  4) | (q[k]     >> 6) << 4;
                    v[k/4] |= sc << (6*(k % 4)) | m << (24 + 6*(k % 4));
                }
                ((half2    *) meta)[r] = b[h]->dm;
                ((uint32_t *) (meta + off_w))[r]                                  = (uint32_t) v[0];
                ((uint32_t *) (meta + off_w + MOE_PACKED_I*sizeof(uint32_t)))[r] = (uint32_t) v[1];
                ((uint32_t *) (meta + off_x))[r] = (uint32_t) (v[0] >> 32) | (uint32_t) (v[1] >> 32) << 16;
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int t = 0; t < 4; ++t) {
                const uint4 v = ((const uint4 *) (in + (j/2)*MOE_PACKED_Q4K_CHUNK))[(mg*2 + j % 2)*WARP_SIZE + 4*g + t];
                const uint32_t lo = h ? v.y : v.x;
                const uint32_t hi = h ? v.w : v.z;
                for (int i = 0; i < 4; ++i) {
                    b.qs[4*(8*j + t)     + i] = lo >> (8*i);
                    b.qs[4*(8*j + t + 4) + i] = hi >> (8*i);
                }
            }
        }
        const char * meta = in + 2*MOE_PACKED_Q4K_CHUNK;
        b.dm = ((const half2 *) meta)[r];
        const uint32_t X = ((const uint32_t *) (meta + off_x))[r];
        uint8_t sc[QK_K/32];
        uint8_t m[QK_K/32];
        for (int c = 0; c < 2; ++c) {
            const uint64_t v = ((const uint32_t *) (meta + off_w + c*MOE_PACKED_I*sizeof(uint32_t)))[r] | (uint64_t) ((X >> (16*c)) & 0xFFFF) << 32;
            for (int k = 0; k < 4; ++k) {
                sc[4*c + k] = (v >> (6*k)) & 63;
                m[4*c + k]  = (v >> (24 + 6*k)) & 63;
            }
        }
        for (int k = 0; k < 4; ++k) {
            b.scales[k]     = sc[k] | (sc[k + 4] >> 4) << 6;
            b.scales[k + 4] = m[k]  | (m[k + 4]  >> 4) << 6;
            b.scales[k + 8] = (sc[k + 4] & 0xF) | (m[k + 4] & 0xF) << 4;
        }
    }

    static __device__ __forceinline__ uint32_t tab(const int i) {
        GGML_UNUSED(i);
        return 0;
    }

    struct frag_state {
        half2    dm[2]; // (d, -dmin) of rows g and g + 8
        uint32_t lo[2]; // bits 0-31 of their 48 scale and min bits of the chunk
        uint32_t hi[2]; // bits 32-47
    };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const char * meta = s + MOE_PACKED_Q4K_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            st.dm[h] = ((const half2 *) meta)[r] * make_half2(1.0f, -1.0f);
            st.lo[h] = ((const uint32_t *) (meta + off_w + (kc % 2)*MOE_PACKED_I*sizeof(uint32_t)))[r];
            st.hi[h] = ((const uint32_t *) (meta + off_x))[r] >> (16*(kc % 2));
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const uint4 v  = ((const uint4 *) s)[(mg*2 + ks/2)*WARP_SIZE + lane];
        const int   sh = 4*(ks % 2);
        return make_uint4((v.x >> sh) & 0x0F0F0F0F, (v.y >> sh) & 0x0F0F0F0F, (v.z >> sh) & 0x0F0F0F0F, (v.w >> sh) & 0x0F0F0F0F);
    }

    // (d*sc, -dmin*m) with the fp16 products of MMQ (load_tiles_q4_K)
    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        const uint8_t sc = (st.lo[h] >> (6*ks)) & 63;
        const uint8_t m  = min6(st.lo[h], st.hi[h], ks);
        return __half22float2(st.dm[h] * make_half2(sc, m));
    }

    // vec_dot_q4_K_q8_1 at position p (iqs = 2*p): the ints of the lane of position p, the scales and mins of sub-blocks 2*(p/4) and
    // 2*(p/4) + 1, and vec_dot_q4_K_q8_1_impl_vmmq for each row.
    static __device__ __forceinline__ float2 vec_dot2(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int j  = p / 4; // group of 64 values
        const int i4 = p % 4;
        const uint4 v = ((const uint4 *) (sb + (j/2)*MOE_PACKED_Q4K_CHUNK))[(mg*2 + j % 2)*WARP_SIZE + 4*g + i4];

        int   u[2*QR4_K];
        float d8[QR4_K];
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1 * bq8i = y + 2*j + i;
            d8[i] = __low2float(bq8i->ds);

            const int * q8 = (const int *) bq8i->qs + i4;
            u[2*i + 0] = q8[0];
            u[2*i + 1] = q8[4];
        }

        const char * meta = sb + 2*MOE_PACKED_Q4K_CHUNK;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int      r  = 16*mg + g + 8*h;
            const uint32_t lo = ((const uint32_t *) (meta + off_w + (j/2)*MOE_PACKED_I*sizeof(uint32_t)))[r];
            const uint32_t hi = ((const uint32_t *) (meta + off_x))[r] >> (16*(j/2));
            uint8_t sc[QR4_K];
            uint8_t m[QR4_K];
#pragma unroll
            for (int i = 0; i < QR4_K; ++i) {
                const int k = 2*(j % 2) + i; // sub-block in the chunk
                sc[i] = (lo >> (6*k)) & 63;
                m[i]  = min6(lo, hi, k);
            }
            const int vq[2] = { (int) (h ? v.y : v.x), (int) (h ? v.w : v.z) };
            res[h] = vec_dot_q4_K_q8_1_impl_vmmq(vq, u, sc, m, ((const half2 *) meta)[r], d8);
        }
        return make_float2(res[0], res[1]);
    }
};

// Q5K chunk: [mg][2 groups of 64 values][lane] uint4 ql words and uint32 qh bytes. H stores the two high bits of each ql byte in
// compressed bytes, in the same word order as Q4K. Metadata is the Q4K dm and 96-bit scale/minimum representation.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_Q5K> {
    using block_t = block_q5_K;
    static constexpr ggml_type type        = GGML_TYPE_Q5_K;
    static constexpr int chunk_bytes       = MOE_PACKED_Q5K_CHUNK;
    static constexpr int sb_bytes           = MOE_PACKED_Q5K_BLOCK;
    static constexpr int tab_words          = 0;
    static constexpr int family             = MOE_PACKED_FAMILY_DM;

    static constexpr int off_w = MOE_PACKED_I*sizeof(half2);
    static constexpr int off_x = off_w + 2*MOE_PACKED_I*sizeof(uint32_t);

    static __device__ __forceinline__ uint8_t min6(const uint32_t lo, const uint32_t hi, const int k) {
        return (((uint64_t) hi << 32 | lo) >> (24 + 6*k)) & 63;
    }

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_Q5K_CHUNK;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int group = 2*c + j;
                const uint4 v = make_uint4(get_int_b4(b0->qs, 8*group + t), get_int_b4(b1->qs, 8*group + t),
                    get_int_b4(b0->qs, 8*group + t + 4), get_int_b4(b1->qs, 8*group + t + 4));
                const uint32_t h0 = moe_q5k_bits(get_int_b4(b0->qh, t) >> (2*group));
                const uint32_t h1 = moe_q5k_bits(get_int_b4(b1->qh, t) >> (2*group));
                const uint32_t h2 = moe_q5k_bits(get_int_b4(b0->qh, t + 4) >> (2*group));
                const uint32_t h3 = moe_q5k_bits(get_int_b4(b1->qh, t + 4) >> (2*group));
                ((uint4 *) oc)[(mg*2 + j)*WARP_SIZE + lane] = v;
                ((uint32_t *) (oc + MOE_PACKED_Q5K_H))[(mg*2 + j)*WARP_SIZE + lane] =
                    h0 | h1 << 8 | h2 << 16 | h3 << 24;
            }
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_Q5K_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const uint8_t * q = (h ? b1 : b0)->scales;
                uint64_t v[2] = { 0, 0 };
#pragma unroll
                for (int k = 0; k < QK_K/32; ++k) {
                    const uint64_t sc = k < 4 ? q[k] & 63     : (q[k + 4] & 0xF) | (q[k - 4] >> 6) << 4;
                    const uint64_t m  = k < 4 ? q[k + 4] & 63 : (q[k + 4] >>  4) | (q[k]     >> 6) << 4;
                    v[k/4] |= sc << (6*(k % 4)) | m << (24 + 6*(k % 4));
                }
                ((half2    *) meta)[r] = (h ? b1 : b0)->dm;
                ((uint32_t *) (meta + off_w))[r]                                  = (uint32_t) v[0];
                ((uint32_t *) (meta + off_w + MOE_PACKED_I*sizeof(uint32_t)))[r] = (uint32_t) v[1];
                ((uint32_t *) (meta + off_x))[r] = (uint32_t) (v[0] >> 32) | (uint32_t) (v[1] >> 32) << 16;
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
        uint32_t qh[QK_K/32] = { 0 };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const char * ic = in + (j/2)*MOE_PACKED_Q5K_CHUNK;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint4 v = ((const uint4 *) ic)[(mg*2 + j%2)*WARP_SIZE + 4*g + t];
                const uint32_t H = ((const uint32_t *) (ic + MOE_PACKED_Q5K_H))[(mg*2 + j%2)*WARP_SIZE + 4*g + t];
                const uint32_t lo = h ? v.y : v.x;
                const uint32_t hi = h ? v.w : v.z;
                const uint8_t h0 = H >> (8*h);
                const uint8_t h1 = H >> (16 + 8*h);
                for (int i = 0; i < 4; ++i) {
                    b.qs[4*(8*j + t) + i]     = lo >> (8*i);
                    b.qs[4*(8*j + t + 4) + i] = hi >> (8*i);
                    qh[t]     |= ((h0 >> i) & 1) << (8*i + 2*j + 0);
                    qh[t]     |= ((h0 >> (4 + i)) & 1) << (8*i + 2*j + 1);
                    qh[t + 4] |= ((h1 >> i) & 1) << (8*i + 2*j + 0);
                    qh[t + 4] |= ((h1 >> (4 + i)) & 1) << (8*i + 2*j + 1);
                }
            }
        }
        for (int i = 0; i < QK_K/32; ++i) {
            ((uint32_t *) b.qh)[i] = qh[i];
        }
        const char * meta = in + 2*MOE_PACKED_Q5K_CHUNK;
        b.dm = ((const half2 *) meta)[r];
        const uint32_t X = ((const uint32_t *) (meta + off_x))[r];
        uint8_t sc[QK_K/32];
        uint8_t m[QK_K/32];
        for (int c = 0; c < 2; ++c) {
            const uint64_t v = ((const uint32_t *) (meta + off_w + c*MOE_PACKED_I*sizeof(uint32_t)))[r] |
                (uint64_t) ((X >> (16*c)) & 0xFFFF) << 32;
            for (int k = 0; k < 4; ++k) {
                sc[4*c + k] = (v >> (6*k)) & 63;
                m[4*c + k]  = (v >> (24 + 6*k)) & 63;
            }
        }
        for (int k = 0; k < 4; ++k) {
            b.scales[k]     = sc[k] | (sc[k + 4] >> 4) << 6;
            b.scales[k + 4] = m[k]  | (m[k + 4]  >> 4) << 6;
            b.scales[k + 8] = (sc[k + 4] & 0xF) | (m[k + 4] & 0xF) << 4;
        }
    }

    static __device__ __forceinline__ uint32_t tab(const int i) {
        GGML_UNUSED(i);
        return 0;
    }

    struct frag_state {
        half2    dm[2];
        uint32_t lo[2];
        uint32_t hi[2];
    };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const char * meta = s + MOE_PACKED_Q5K_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            st.dm[h] = ((const half2 *) meta)[r] * make_half2(1.0f, -1.0f);
            st.lo[h] = ((const uint32_t *) (meta + off_w + (kc % 2)*MOE_PACKED_I*sizeof(uint32_t)))[r];
            st.hi[h] = ((const uint32_t *) (meta + off_x))[r] >> (16*(kc % 2));
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const uint4 v = ((const uint4 *) s)[(mg*2 + ks/2)*WARP_SIZE + lane];
        const uint32_t H = ((const uint32_t *) (s + MOE_PACKED_Q5K_H))[(mg*2 + ks/2)*WARP_SIZE + lane];
        const int part = ks % 2;
        return make_uint4(moe_q5_code(v.x, H >> 0, part), moe_q5_code(v.y, H >> 8, part),
            moe_q5_code(v.z, H >> 16, part), moe_q5_code(v.w, H >> 24, part));
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        const uint8_t sc = (st.lo[h] >> (6*ks)) & 63;
        const uint8_t m  = min6(st.lo[h], st.hi[h], ks);
        return __half22float2(st.dm[h] * make_half2(sc, m));
    }

    static __device__ __forceinline__ float2 vec_dot2(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int j  = p / 4;
        const int i4 = p % 4;
        const uint4 v = ((const uint4 *) (sb + (j/2)*MOE_PACKED_Q5K_CHUNK))[(mg*2 + j%2)*WARP_SIZE + 4*g + i4];
        int vl[2];
        int vh[2];
        int u[2*QR5_K];
        float d8[QR5_K];
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1 * bq8i = y + 2*j + i;
            d8[i] = __low2float(bq8i->ds);
            const int * q8 = (const int *) bq8i->qs + i4;
            u[2*i + 0] = q8[0];
            u[2*i + 1] = q8[4];
        }
        const char * meta = sb + 2*MOE_PACKED_Q5K_CHUNK;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t H = ((const uint32_t *) (sb + (j/2)*MOE_PACKED_Q5K_CHUNK + MOE_PACKED_Q5K_H))[(mg*2 + j%2)*WARP_SIZE + 4*g + i4];
            vl[0] = h ? (int) v.y : (int) v.x;
            vl[1] = h ? (int) v.w : (int) v.z;
            vh[0] = moe_q5k_vh(h ? H >> 8 : H & 0xFF);
            vh[1] = moe_q5k_vh(h ? H >> 24 : H >> 16);
            const uint32_t lo = ((const uint32_t *) (meta + off_w + (j/2)*MOE_PACKED_I*sizeof(uint32_t)))[16*mg + g + 8*h];
            const uint32_t hi = ((const uint32_t *) (meta + off_x))[16*mg + g + 8*h] >> (16*(j/2));
            uint8_t sc[QR5_K];
            uint8_t m[QR5_K];
#pragma unroll
            for (int i = 0; i < QR5_K; ++i) {
                const int k = 2*(j % 2) + i;
                sc[i] = (lo >> (6*k)) & 63;
                m[i]  = min6(lo, hi, k);
            }
            res[h] = vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m,
                ((const half2 *) meta)[16*mg + g + 8*h], d8);
        }
        return make_float2(res[0], res[1]);
    }
};

// Q6K chunk: L [mg][4 quads q][lane] uint2, H [mg][4 quads q][lane] u32. Lane (g, t) of quad q of chunk n holds MMVQ position
// iqs = 16*n + 4*q + t of rows 16*mg + g and + 8: L = the ql int iqs of both rows, H = the bits 0x33333333 of the shifted qh int of
// vec_dot_q6_K_q8_1 for iqs, row g + 8 shifted left by 2. Position iqs has 4 values of sub-block 4*n + q/2 (low nibbles) and of
// sub-block 4*n + q/2 + 2 (high nibbles), from value 16*(q % 2) + 4*t: quads 2*s and 2*s + 1 give the A fragment of k step s (low
// nibbles) and s + 2 (high nibbles). Metadata: d [128] f16, the 16 int8 scales of each row [128][16].
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_Q6K> {
    using block_t = block_q6_K;
    static constexpr ggml_type type        = GGML_TYPE_Q6_K;
    static constexpr int       chunk_bytes = MOE_PACKED_Q6K_CHUNK;
    static constexpr int       sb_bytes    = MOE_PACKED_Q6K_BLOCK;
    static constexpr int       tab_words   = 0;
    static constexpr int       family      = MOE_PACKED_FAMILY_D16;

    static constexpr int off_sc = MOE_PACKED_I*sizeof(half); // scales, from the start of the metadata

    // the qh int of vec_dot_q6_K_q8_1 for position iqs after its shift, the bits that it uses
    static __device__ __forceinline__ uint32_t qh_bits(const block_t * b, const int iqs) {
        return (get_int_b2(b->qh, 8*(iqs/16) + iqs % 8) >> (2*((iqs % 16)/8))) & 0x33333333;
    }

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const block_t * b[2] = { b0, b1 };
        const int t = lane % 4;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            char * oc = out + n*MOE_PACKED_Q6K_CHUNK;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int iqs = 16*n + 4*q + t;
                ((uint2    *) oc)[(mg*4 + q)*WARP_SIZE + lane] = make_uint2(get_int_b2(b0->ql, iqs), get_int_b2(b1->ql, iqs));
                ((uint32_t *) (oc + MOE_PACKED_Q6K_H))[(mg*4 + q)*WARP_SIZE + lane] = qh_bits(b0, iqs) | qh_bits(b1, iqs) << 2;
            }
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_Q6K_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                ((half *) meta)[r] = b[h]->d;
                for (int i = 0; i < QK_K/16; ++i) {
                    ((int8_t *) (meta + off_sc))[16*r + i] = b[h]->scales[i];
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
        uint32_t qh[QK_K/16] = { 0 };
        for (int iqs = 0; iqs < QI6_K; ++iqs) {
            const int n = iqs / 16;
            const int q = (iqs % 16) / 4;
            const char * ic = in + n*MOE_PACKED_Q6K_CHUNK;
            const uint2    L = ((const uint2    *) ic)[(mg*4 + q)*WARP_SIZE + 4*g + iqs % 4];
            const uint32_t H = ((const uint32_t *) (ic + MOE_PACKED_Q6K_H))[(mg*4 + q)*WARP_SIZE + 4*g + iqs % 4];
            const uint32_t l = h ? L.y : L.x;
            for (int i = 0; i < 4; ++i) {
                b.ql[4*iqs + i] = l >> (8*i);
            }
            qh[8*n + iqs % 8] |= ((H >> (2*h)) & 0x33333333) << (2*((iqs % 16)/8));
        }
        for (int k = 0; k < QK_K/16; ++k) {
            for (int i = 0; i < 4; ++i) {
                b.qh[4*k + i] = qh[k] >> (8*i);
            }
        }
        const char * meta = in + 2*MOE_PACKED_Q6K_CHUNK;
        b.d = ((const half *) meta)[r];
        for (int i = 0; i < QK_K/16; ++i) {
            b.scales[i] = ((const int8_t *) (meta + off_sc))[16*r + i];
        }
    }

    static __device__ __forceinline__ uint32_t tab(const int i) {
        GGML_UNUSED(i);
        return 0;
    }

    struct frag_state {
        float    d[2];     // rows g and g + 8
        uint32_t sc[2][2]; // their 8 scales of the chunk
    };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const char * meta = s + MOE_PACKED_Q6K_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int   r = mg*16 + lane/4 + 8*h;
            const uint2 w = *(const uint2 *) (meta + off_sc + 16*r + 8*(kc % 2));
            st.d[h]     = __half2float(((const half *) meta)[r]);
            st.sc[h][0] = w.x;
            st.sc[h][1] = w.y;
        }
        return st;
    }

    // the int8 codes (ql | qh << 4) - 32 of load_tiles_q6_K for the 4 values of an L word, i = 0: low nibbles, 1: high nibbles.
    // For a byte b in [0, 63], (b + 0x60) ^ 0x80 is b - 32 as int8 with no carry to the next byte (2 instructions, __vsubss4 has 7).
    static __device__ __forceinline__ uint32_t code(const uint32_t l, const uint32_t hb, const int i) {
        return (((l >> (4*i)) & 0x0F0F0F0F) + (((hb >> (4*i)) << 4) & 0x30303030) + 0x60606060) ^ 0x80808080;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const int q = 2*(ks % 2);
        const int i = ks / 2;
        const uint2    L0 = ((const uint2    *) s)[(mg*4 + q + 0)*WARP_SIZE + lane];
        const uint2    L1 = ((const uint2    *) s)[(mg*4 + q + 1)*WARP_SIZE + lane];
        const uint32_t H0 = ((const uint32_t *) (s + MOE_PACKED_Q6K_H))[(mg*4 + q + 0)*WARP_SIZE + lane];
        const uint32_t H1 = ((const uint32_t *) (s + MOE_PACKED_Q6K_H))[(mg*4 + q + 1)*WARP_SIZE + lane];
        return make_uint4(code(L0.x, H0, i), code(L0.y, H0 >> 2, i), code(L1.x, H1, i), code(L1.y, H1 >> 2, i));
    }

    // d*sc of the values 0-15 and 16-31 of k step ks, exact in fp32
    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        const uint32_t w = st.sc[h][ks/2] >> (16*(ks % 2));
        return make_float2(st.d[h]*(int8_t) (w & 0xFF), st.d[h]*(int8_t) ((w >> 8) & 0xFF));
    }

    // vec_dot_q6_K_q8_1 at position p (iqs = p): the words of the lane of position p and vec_dot_q6_K_q8_1_impl_mmvq for each row.
    static __device__ __forceinline__ float2 vec_dot2(
            const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const char *   ch = sb + (p/16)*MOE_PACKED_Q6K_CHUNK;
        const int      e  = (mg*4 + (p % 16)/4)*WARP_SIZE + 4*g + p % 4;
        const uint2    L  = ((const uint2    *) ch)[e];
        const uint32_t H  = ((const uint32_t *) (ch + MOE_PACKED_Q6K_H))[e];

        const int bq8_offset   = 2*QR6_K*(p/(QI6_K/2)) + (p % (QI6_K/2))/(QI6_K/4);
        const int scale_offset = (QI6_K/4)*(p/(QI6_K/2)) + (p % (QI6_K/2))/(QI6_K/8);

        int   u[QR6_K];
        float d8[QR6_K];
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            u[i]  = get_int_b4(y[bq8_offset + 2*i].qs, p % QI8_1);
            d8[i] = __low2float(y[bq8_offset + 2*i].ds);
        }

        const char * meta = sb + 2*MOE_PACKED_Q6K_CHUNK;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int   r  = 16*mg + g + 8*h;
            const int   vl = h ? L.y : L.x;
            const int   vh = (H >> (2*h)) & 0x33333333;
            const float d  = __half2float(((const half *) meta)[r]);
            res[h] = vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, (const int8_t *) (meta + off_sc) + 16*r + scale_offset, d, d8);
        }
        return make_float2(res[0], res[1]);
    }
};

// Q4_1 chunk: the four source blocks of one 128-value half, with each row kept contiguous. Metadata is the original half2 d,m pair.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_Q41> {
    using block_t = block_q4_1;
    static constexpr ggml_type type      = GGML_TYPE_Q4_1;
    static constexpr int chunk_bytes     = MOE_PACKED_Q41_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_Q41_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_DM;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_Q41_CHUNK;
#pragma unroll
            for (int ib = 0; ib < 4; ++ib) {
                const int src = 4*c + ib;
                const int off = (mg*16 + lane/4)*64 + ib*16 + 4*t;
                *(uint32_t *) (oc + off) = get_int_b4((b0 + src)->qs, t);
                *(uint32_t *) (oc + off + 8*64) = get_int_b4((b1 + src)->qs, t);
            }
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_Q41_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    ((half2 *) meta)[r*8 + ib] = (h ? b1 : b0)[ib].dm;
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b, const int ib) {
        const int c = ib / 4;
        const int j = ib % 4;
        const uint32_t * q = (const uint32_t *) (in + c*MOE_PACKED_Q41_CHUNK) + r*16 + j*4;
        for (int i = 0; i < 4; ++i) ((uint32_t *) b.qs)[i] = q[i];
        b.dm = ((const half2 *) (in + 2*MOE_PACKED_Q41_CHUNK))[r*8 + ib];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { half2 dm[2][4]; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const half2 * meta = (const half2 *) (s + MOE_PACKED_Q41_CHUNK);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                st.dm[h][ks] = meta[r*8 + 4*(kc % 2) + ks];
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const uint32_t * q = (const uint32_t *) s;
        const uint32_t a = q[(mg*16 + lane/4)*16 + ks*4 + lane%4];
        const uint32_t b = q[(mg*16 + lane/4 + 8)*16 + ks*4 + lane%4];
        return make_uint4(a & 0x0F0F0F0F, b & 0x0F0F0F0F, (a >> 4) & 0x0F0F0F0F, (b >> 4) & 0x0F0F0F0F);
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        return __half22float2(st.dm[h][ks]);
    }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int ib = p / 2;
        const int iqs = 2*(p % 2);
        int u[4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[2*i + 0] = get_int_b4((y + ib)->qs, iqs + i);
            u[2*i + 1] = get_int_b4((y + ib)->qs, iqs + i + QI4_1);
        }
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int v[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int q = iqs + i;
                const uint32_t w = ((const uint32_t *) (sb + (ib/4)*MOE_PACKED_Q41_CHUNK))[(mg*16 + g + 8*h)*16 + (ib%4)*4 + q];
                v[i] = (int) w;
            }
            const int r = 16*mg + g + 8*h;
            const half2 dm = ((const half2 *) (sb + 2*MOE_PACKED_Q41_CHUNK))[r*8 + ib];
            res[h] = vec_dot_q4_1_q8_1_impl<2>(v, u, dm, (y + ib)->ds);
        }
        return make_float2(res[0], res[1]);
    }
};

// Q5_1 has the same four-block DM organization as Q4_1, with one qh byte per four qs bytes.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_Q51> {
    using block_t = block_q5_1;
    static constexpr ggml_type type      = GGML_TYPE_Q5_1;
    static constexpr int chunk_bytes     = MOE_PACKED_Q51_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_Q51_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_DM;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_Q51_CHUNK;
#pragma unroll
            for (int ib = 0; ib < 4; ++ib) {
                const int src = 4*c + ib;
                const int off0 = (mg*16 + lane/4)*80 + ib*16 + 4*t;
                const int off1 = (mg*16 + lane/4)*80 + 64 + ib*4 + t;
                *(uint32_t *) (oc + off0) = get_int_b4((b0 + src)->qs, t);
                ((uint8_t *) oc)[off1] = (b0 + src)->qh[t];
                *(uint32_t *) (oc + off0 + 8*80) = get_int_b4((b1 + src)->qs, t);
                ((uint8_t *) oc)[off1 + 8*80] = (b1 + src)->qh[t];
            }
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_Q51_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    ((half2 *) meta)[r*8 + ib] = (h ? b1 : b0)[ib].dm;
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b, const int ib) {
        const int c = ib / 4;
        const int j = ib % 4;
        const char * oc = in + c*MOE_PACKED_Q51_CHUNK;
        const uint32_t * row = (const uint32_t *) oc + r*20;
        const uint32_t * q = row + j*4;
        const uint32_t * h = row + 16 + j; // the qh region starts at byte 64 (word 16) of the row, one word per block
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ((uint32_t *) b.qs)[i] = q[i];
        }
        const uint32_t qh = h[0];
        for (int i = 0; i < 4; ++i) b.qh[i] = qh >> (8*i);
        b.dm = ((const half2 *) (in + 2*MOE_PACKED_Q51_CHUNK))[r*8 + ib];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { half2 dm[2][4]; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const half2 * meta = (const half2 *) (s + MOE_PACKED_Q51_CHUNK);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                st.dm[h][ks] = meta[r*8 + 4*(kc % 2) + ks];
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const char * oc = s;
        const int e0 = (mg*16 + lane/4)*20 + ks*4 + lane%4;
        const int e1 = (mg*16 + lane/4 + 8)*20 + ks*4 + lane%4;
        const uint32_t q0 = ((const uint32_t *) oc)[e0];
        const uint32_t q1 = ((const uint32_t *) oc)[e1];
        const uint32_t hw0 = ((const uint32_t *) (oc + (mg*16 + lane/4)*80 + 64))[ks];
        const uint32_t hw1 = ((const uint32_t *) (oc + (mg*16 + lane/4 + 8)*80 + 64))[ks];
        const uint8_t h0 = moe_q50_bits(hw0, lane%4);
        const uint8_t h1 = moe_q50_bits(hw1, lane%4);
        return make_uint4(moe_q5_code(q0, h0, 0), moe_q5_code(q1, h1, 0), moe_q5_code(q0, h0, 1), moe_q5_code(q1, h1, 1));
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        return __half22float2(st.dm[h][ks]);
    }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int ib = p / 2;
        const int iqs = 2*(p % 2);
        int u[4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[2*i + 0] = get_int_b4((y + ib)->qs, iqs + i);
            u[2*i + 1] = get_int_b4((y + ib)->qs, iqs + i + QI5_1);
        }
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int vl[2], vh[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int q = iqs + i;
                const char * oc = sb + (ib/4)*MOE_PACKED_Q51_CHUNK;
                const int row = 16*mg + g + 8*h;
                vl[i] = get_int_b4(oc + row*80, (ib%4)*4 + q);
                const uint32_t qh = ((const uint32_t *) (oc + row*80 + 64))[ib%4];
                vh[i] = (int) (qh >> (4*q));
            }
            const int r = 16*mg + g + 8*h;
            const half2 dm = ((const half2 *) (sb + 2*MOE_PACKED_Q51_CHUNK))[r*8 + ib];
            res[h] = vec_dot_q5_1_q8_1_impl<2>(vl, vh, u, dm, (y + ib)->ds);
        }
        return make_float2(res[0], res[1]);
    }
};

// Q3_K keeps the ql bytes in the chunks and the shared hmask/scales in metadata. The hmask is used by both 128-value halves, with
// the low four bits selecting the first half and the high four bits selecting the second.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_Q3K> {
    using block_t = block_q3_K;
    static constexpr ggml_type type      = GGML_TYPE_Q3_K;
    static constexpr int chunk_bytes     = MOE_PACKED_Q3K_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_Q3K_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_D16;
    static constexpr int off_sc          = MOE_PACKED_I*sizeof(half);
    static constexpr int off_hm          = off_sc + MOE_PACKED_I*12;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_Q3K_CHUNK;
            const int r0 = mg*16 + lane/4;
            const int r1 = r0 + 8;
            ((uint32_t *) (oc + r0*32 + 8*t))[0] = get_int_b1(b0->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + r0*32 + 8*t))[1] = get_int_b1(b0->qs + c*32, 2*t + 1);
            ((uint32_t *) (oc + r1*32 + 8*t))[0] = get_int_b1(b1->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + r1*32 + 8*t))[1] = get_int_b1(b1->qs + c*32, 2*t + 1);
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_Q3K_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const block_t & b = h ? b1[0] : b0[0];
                ((half *) meta)[r] = b.d;
#pragma unroll
                for (int i = 0; i < 12; ++i) ((uint8_t *) (meta + off_sc))[r*12 + i] = b.scales[i];
#pragma unroll
                for (int i = 0; i < QK_K/8; ++i) ((uint8_t *) (meta + off_hm))[r*(QK_K/8) + i] = b.hmask[i];
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        for (int c = 0; c < 2; ++c) {
            const uint8_t * q = (const uint8_t *) (in + c*MOE_PACKED_Q3K_CHUNK) + r*32;
            for (int i = 0; i < 32; ++i) b.qs[c*32 + i] = q[i];
        }
        const char * meta = in + 2*MOE_PACKED_Q3K_CHUNK;
        b.d = ((const half *) meta)[r];
        for (int i = 0; i < 12; ++i) b.scales[i] = ((const uint8_t *) (meta + off_sc))[r*12 + i];
        for (int i = 0; i < QK_K/8; ++i) b.hmask[i] = ((const uint8_t *) (meta + off_hm))[r*(QK_K/8) + i];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { float d; float sc[2][2][4]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        const char * meta = s + MOE_PACKED_Q3K_CHUNK;
        st.c = kc % 2;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            st.d = __half2float(((const half *) meta)[r]);
            uint32_t aux[4] = { 0, 0, 0, 0 };
#pragma unroll
            for (int i = 0; i < 3; ++i) aux[i] = ((const uint32_t *) (meta + off_sc))[3*r + i];
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
            aux[0] = (aux[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
            aux[1] = (aux[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
            const int8_t * sc = (const int8_t *) aux;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                st.sc[h][0][ks] = st.d * (sc[st.c*8 + 2*ks + 0] - 32);
                st.sc[h][1][ks] = st.d * (sc[st.c*8 + 2*ks + 1] - 32);
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint8_t code(const char * s, const frag_state & st, const int row, const int v, const int ks) {
        const uint8_t q = ((const uint8_t *) s)[row*32 + v];
        const uint8_t hm = ((const uint8_t *) (s + MOE_PACKED_Q3K_CHUNK + off_hm))[row*32 + v];
        return (uint8_t) (((q >> (2*ks)) & 3) - ((hm & (1 << (st.c*4 + ks))) ? 0 : 4));
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        const int g = lane/4;
        const int t = lane%4;
        const int r0 = mg*16 + g;
        const int r1 = r0 + 8;
        return make_uint4(code(s, st, r0, 4*t + 0, ks) | code(s, st, r0, 4*t + 1, ks) << 8 |
                code(s, st, r0, 4*t + 2, ks) << 16 | code(s, st, r0, 4*t + 3, ks) << 24,
            code(s, st, r1, 4*t + 0, ks) | code(s, st, r1, 4*t + 1, ks) << 8 |
                code(s, st, r1, 4*t + 2, ks) << 16 | code(s, st, r1, 4*t + 3, ks) << 24,
            code(s, st, r0, 16 + 4*t + 0, ks) | code(s, st, r0, 16 + 4*t + 1, ks) << 8 |
                code(s, st, r0, 16 + 4*t + 2, ks) << 16 | code(s, st, r0, 16 + 4*t + 3, ks) << 24,
            code(s, st, r1, 16 + 4*t + 0, ks) | code(s, st, r1, 16 + 4*t + 1, ks) << 8 |
                code(s, st, r1, 16 + 4*t + 2, ks) << 16 | code(s, st, r1, 16 + 4*t + 3, ks) << 24);
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        return make_float2(st.sc[h][0][ks], st.sc[h][1][ks]);
    }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int c = p / 8;
        const int iqs = p % 8;
        const char * ch = sb + c*MOE_PACKED_Q3K_CHUNK;
        const int r0 = 16*mg + g;
        const int r1 = r0 + 8;
        const int vl0 = get_int_b1(ch + r0*32, iqs);
        const int vl1 = get_int_b1(ch + r1*32, iqs);
        const char * meta = sb + 2*MOE_PACKED_Q3K_CHUNK;
        const uint32_t h0 = ((const uint32_t *) (meta + off_hm))[r0*8 + iqs];
        const uint32_t h1 = ((const uint32_t *) (meta + off_hm))[r1*8 + iqs];
        const int bq8_offset = QR3_K * (p / (QI3_K/2));
        const int scale_offset = p - p % QI8_1 + (p % QI8_1)/(QI8_1/2);
        int u[QR3_K];
        float d8[QR3_K];
#pragma unroll
        for (int i = 0; i < QR3_K; ++i) {
            u[i] = get_int_b4((y + bq8_offset + i)->qs, iqs % QI8_1);
            d8[i] = __low2float((y + bq8_offset + i)->ds);
        }
        const float d0 = __half2float(((const half *) meta)[r0]);
        const float d1 = __half2float(((const half *) meta)[r1]);
        const uint8_t * sc0 = (const uint8_t *) (meta + off_sc + 12*r0);
        const uint8_t * sc1 = (const uint8_t *) (meta + off_sc + 12*r1);
        const float a = vec_dot_q3_K_q8_1_impl_mmvq(vl0, ~h0 >> bq8_offset, u, sc0, scale_offset, d0, d8);
        const float b = vec_dot_q3_K_q8_1_impl_mmvq(vl1, ~h1 >> bq8_offset, u, sc1, scale_offset, d1, d8);
        return make_float2(a, b);
    }
};

// IQ2_XXS uses four 32-value code groups in each packed chunk. Its one scale is derived from the group sign/index word.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ2XXS> {
    using block_t = block_iq2_xxs;
    static constexpr ggml_type type      = GGML_TYPE_IQ2_XXS;
    static constexpr int chunk_bytes     = MOE_PACKED_IQ2XXS_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_IQ2XXS_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_D4;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_IQ2XXS_CHUNK;
            ((uint32_t *) (oc + (mg*16 + lane/4)*32 + 8*t))[0] = get_int_b1((const uint8_t *) b0->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + (mg*16 + lane/4)*32 + 8*t))[1] = get_int_b1((const uint8_t *) b0->qs + c*32, 2*t + 1);
            ((uint32_t *) (oc + (mg*16 + lane/4 + 8)*32 + 8*t))[0] = get_int_b1((const uint8_t *) b1->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + (mg*16 + lane/4 + 8)*32 + 8*t))[1] = get_int_b1((const uint8_t *) b1->qs + c*32, 2*t + 1);
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_IQ2XXS_CHUNK;
            ((half *) meta)[mg*16 + lane/4] = b0->d;
            ((half *) meta)[mg*16 + lane/4 + 8] = b1->d;
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        for (int c = 0; c < 2; ++c) {
            const uint8_t * q = (const uint8_t *) (in + c*MOE_PACKED_IQ2XXS_CHUNK) + r*32;
            for (int i = 0; i < 32; ++i) ((uint8_t *) b.qs)[c*32 + i] = q[i];
        }
        b.d = ((const half *) (in + 2*MOE_PACKED_IQ2XXS_CHUNK))[r];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { float d[2][4]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * meta = s + MOE_PACKED_IQ2XXS_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            const float d = __half2float(((const half *) meta)[r]);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const uint32_t aux = ((const uint32_t *) s)[(mg*16 + lane/4 + 8*h)*8 + 2*ks + 1];
                st.d[h][ks] = d * (0.5f + (aux >> 28)) * 0.25f;
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint8_t code(const char * s, const frag_state & st, const int row, const int ks, const int j) {
        const uint8_t * q = (const uint8_t *) s + row*32 + 8*ks;
        const uint8_t * aux8 = q;
        const uint32_t aux32 = ((const uint32_t *) q)[1];
        const uint8_t * grid = (const uint8_t *) (iq2xxs_grid + aux8[j/8]);
        const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*(j/8))) & 127];
        return (uint8_t) (signs & kmask_iq2xs[j%8] ? -(int) grid[j%8] : (int) grid[j%8]);
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        const int g = lane/4;
        const int t = lane%4;
        const int r0 = mg*16 + g;
        const int r1 = r0 + 8;
        return make_uint4(code(s, st, r0, ks, 4*t+0) | code(s, st, r0, ks, 4*t+1) << 8 |
                code(s, st, r0, ks, 4*t+2) << 16 | code(s, st, r0, ks, 4*t+3) << 24,
            code(s, st, r1, ks, 4*t+0) | code(s, st, r1, ks, 4*t+1) << 8 |
                code(s, st, r1, ks, 4*t+2) << 16 | code(s, st, r1, ks, 4*t+3) << 24,
            code(s, st, r0, ks, 16+4*t+0) | code(s, st, r0, ks, 16+4*t+1) << 8 |
                code(s, st, r0, ks, 16+4*t+2) << 16 | code(s, st, r0, ks, 16+4*t+3) << 24,
            code(s, st, r1, ks, 16+4*t+0) | code(s, st, r1, ks, 16+4*t+1) << 8 |
                code(s, st, r1, ks, 16+4*t+2) << 16 | code(s, st, r1, ks, 16+4*t+3) << 24);
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) { return st.d[h][ks]; }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int c = p / 4;
        const int lbase = p % 4;
        float res[2];
        const char * meta = sb + 2*MOE_PACKED_IQ2XXS_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16*mg + g + 8*h;
            const uint8_t * q = (const uint8_t *) (sb + c*MOE_PACKED_IQ2XXS_CHUNK) + row*32 + 8*lbase;
            const uint32_t q2 = ((const uint32_t *) q)[0];
            const uint32_t aux32 = ((const uint32_t *) q)[1];
            int sumi = 0;
#pragma unroll
            for (int k0 = 0; k0 < 8; k0 += 2) {
                const uint2 grid_pos = ((const uint2 *) iq2xxs_grid)[((const uint8_t *) &q2)[k0/2]];
                const uint32_t signs = unpack_ksigns(aux32 >> (7*k0/2));
                const int grid0 = __vsub4(grid_pos.x ^ __vcmpne4(signs & 0x08040201, 0), __vcmpne4(signs & 0x08040201, 0));
                const int grid1 = __vsub4(grid_pos.y ^ __vcmpne4(signs & 0x80402010, 0), __vcmpne4(signs & 0x80402010, 0));
                sumi = ggml_cuda_dp4a(grid0, get_int_b4((y + p)->qs, k0 + 0), sumi);
                sumi = ggml_cuda_dp4a(grid1, get_int_b4((y + p)->qs, k0 + 1), sumi);
            }
            const int ls = aux32 >> 27 | 1;
            const float d = __half2float(((const half *) meta)[row]) * __low2float((y + p)->ds);
            res[h] = d * (sumi * ls / 8);
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ3_XXS stores the q grid indices and their scale/sign word together in each chunk.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ3XXS> {
    using block_t = block_iq3_xxs;
    static constexpr ggml_type type      = GGML_TYPE_IQ3_XXS;
    static constexpr int chunk_bytes     = MOE_PACKED_IQ3XXS_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_IQ3XXS_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_D4;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_IQ3XXS_CHUNK;
            const int r0 = mg*16 + lane/4;
            const int r1 = r0 + 8;
            ((uint32_t *) (oc + r0*48 + 8*t))[0] = get_int_b1(b0->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + r0*48 + 8*t))[1] = get_int_b1(b0->qs + c*32, 2*t + 1);
            ((uint32_t *) (oc + r1*48 + 8*t))[0] = get_int_b1(b1->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + r1*48 + 8*t))[1] = get_int_b1(b1->qs + c*32, 2*t + 1);
            *(uint32_t *) (oc + r0*48 + 32 + 4*t) = get_int_b1(b0->qs + 64 + c*16, t);
            *(uint32_t *) (oc + r1*48 + 32 + 4*t) = get_int_b1(b1->qs + 64 + c*16, t);
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_IQ3XXS_CHUNK;
            ((half *) meta)[mg*16 + lane/4] = b0->d;
            ((half *) meta)[mg*16 + lane/4 + 8] = b1->d;
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        for (int c = 0; c < 2; ++c) {
            const char * oc = in + c*MOE_PACKED_IQ3XXS_CHUNK;
            for (int i = 0; i < 32; ++i) b.qs[c*32 + i] = ((const uint8_t *) (oc + r*48))[i];
            for (int i = 0; i < 16; ++i) b.qs[64 + c*16 + i] = ((const uint8_t *) (oc + r*48 + 32))[i];
        }
        b.d = ((const half *) (in + 2*MOE_PACKED_IQ3XXS_CHUNK))[r];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { float d[2][4]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * meta = s + MOE_PACKED_IQ3XXS_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = mg*16 + lane/4 + 8*h;
            const float d = __half2float(((const half *) meta)[row]);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const uint32_t aux = ((const uint32_t *) s)[row*12 + 8 + ks];
                st.d[h][ks] = d * (0.5f + (aux >> 28)) * 0.5f;
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint8_t code(const char * s, const frag_state & st, const int row, const int ks, const int j) {
        const uint8_t * q = (const uint8_t *) s + row*48 + 8*ks;
        const uint32_t aux = ((const uint32_t *) (s + row*48 + 32))[ks];
        const int l = j/8;
        const uint8_t * grid = (const uint8_t *) (iq3xxs_grid + q[2*l + (j/4)%2]);
        const uint8_t signs = ksigns_iq2xs[(aux >> (7*l)) & 127];
        return (uint8_t) (signs & kmask_iq2xs[j%8] ? -(int) grid[j%4] : (int) grid[j%4]);
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        const int g = lane/4;
        const int t = lane%4;
        const int r0 = mg*16 + g;
        const int r1 = r0 + 8;
        return make_uint4(code(s, st, r0, ks, 4*t+0) | code(s, st, r0, ks, 4*t+1) << 8 |
                code(s, st, r0, ks, 4*t+2) << 16 | code(s, st, r0, ks, 4*t+3) << 24,
            code(s, st, r1, ks, 4*t+0) | code(s, st, r1, ks, 4*t+1) << 8 |
                code(s, st, r1, ks, 4*t+2) << 16 | code(s, st, r1, ks, 4*t+3) << 24,
            code(s, st, r0, ks, 16+4*t+0) | code(s, st, r0, ks, 16+4*t+1) << 8 |
                code(s, st, r0, ks, 16+4*t+2) << 16 | code(s, st, r0, ks, 16+4*t+3) << 24,
            code(s, st, r1, ks, 16+4*t+0) | code(s, st, r1, ks, 16+4*t+1) << 8 |
                code(s, st, r1, ks, 16+4*t+2) << 16 | code(s, st, r1, ks, 16+4*t+3) << 24);
    }

    static __device__ __forceinline__ float scale(const frag_state & st, const int h, const int ks) { return st.d[h][ks]; }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int c = p / 4;
        const int lbase = p % 4;
        const char * ch = sb + c*MOE_PACKED_IQ3XXS_CHUNK;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16*mg + g + 8*h;
            const uint8_t * q = (const uint8_t *) (ch + row*48 + 8*lbase);
            const uint32_t aux = ((const uint32_t *) (ch + row*48 + 32))[lbase];
            int sumi = 0;
#pragma unroll
            for (int l = 0; l < 8; l += 2) {
                const int2 grid_pos = make_int2(iq3xxs_grid[q[l+0]], iq3xxs_grid[q[l+1]]);
                const uint32_t signs = unpack_ksigns(aux >> (7*l/2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                sumi = ggml_cuda_dp4a(__vsub4(grid_pos.x ^ signs0, signs0), get_int_b4((y + p)->qs, l+0), sumi);
                sumi = ggml_cuda_dp4a(__vsub4(grid_pos.y ^ signs1, signs1), get_int_b4((y + p)->qs, l+1), sumi);
            }
            const float d = __half2float(((const half *) (sb + 2*MOE_PACKED_IQ3XXS_CHUNK))[row]) * __low2float((y + p)->ds);
            const int sum = ((int) (aux >> 28)*sumi + sumi/2) / 2;
            res[h] = d * sum;
        }
        return make_float2(res[0], res[1]);
    }
};

template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ2XS> {
    using block_t = block_iq2_xs;
    static constexpr ggml_type type      = GGML_TYPE_IQ2_XS;
    static constexpr int chunk_bytes     = MOE_PACKED_IQ2XS_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_IQ2XS_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_D16;
    static constexpr int off_sc          = MOE_PACKED_I*sizeof(half);

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_IQ2XS_CHUNK;
            ((uint32_t *) (oc + (mg*16 + lane/4)*32 + 8*t))[0] = get_int_b1((const uint8_t *) b0->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + (mg*16 + lane/4)*32 + 8*t))[1] = get_int_b1((const uint8_t *) b0->qs + c*32, 2*t + 1);
            ((uint32_t *) (oc + (mg*16 + lane/4 + 8)*32 + 8*t))[0] = get_int_b1((const uint8_t *) b1->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + (mg*16 + lane/4 + 8)*32 + 8*t))[1] = get_int_b1((const uint8_t *) b1->qs + c*32, 2*t + 1);
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_IQ2XS_CHUNK;
            ((half *) meta)[mg*16 + lane/4] = b0->d;
            ((half *) meta)[mg*16 + lane/4 + 8] = b1->d;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                ((uint8_t *) (meta + off_sc))[(mg*16 + lane/4)*8 + i] = b0->scales[i];
                ((uint8_t *) (meta + off_sc))[(mg*16 + lane/4 + 8)*8 + i] = b1->scales[i];
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        for (int c = 0; c < 2; ++c) {
            const uint8_t * q = (const uint8_t *) (in + c*MOE_PACKED_IQ2XS_CHUNK) + r*32;
            for (int i = 0; i < 32; ++i) ((uint8_t *) b.qs)[c*32 + i] = q[i];
        }
        const char * meta = in + 2*MOE_PACKED_IQ2XS_CHUNK;
        b.d = ((const half *) meta)[r];
        for (int i = 0; i < 8; ++i) b.scales[i] = ((const uint8_t *) (meta + off_sc))[r*8 + i];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { float sc[2][2][4]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * meta = s + MOE_PACKED_IQ2XS_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            const float d = __half2float(((const half *) meta)[r]);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const uint8_t ls = ((const uint8_t *) (meta + off_sc))[r*8 + st.c*4 + ks];
                st.sc[h][0][ks] = d * ((ls & 0x0F) + 0.5f) / 4.0f;
                st.sc[h][1][ks] = d * ((ls >> 4) + 0.5f) / 4.0f;
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint8_t code(const char * s, const frag_state & st, const int row, const int ks, const int v) {
        const uint16_t q = ((const uint16_t *) (s + row*32))[4*ks + v/8];
        const uint8_t * grid = (const uint8_t *) (iq2xs_grid + (q & 511));
        const uint8_t signs = unpack_ksigns(q >> 9);
        return (uint8_t) (signs & kmask_iq2xs[v%8] ? -(int) grid[v%8] : (int) grid[v%8]);
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const int g = lane/4;
        const int t = lane%4;
        const int r0 = mg*16 + g;
        const int r1 = r0 + 8;
        return make_uint4(code(s, st, r0, ks, 4*t+0) | code(s, st, r0, ks, 4*t+1) << 8 |
                code(s, st, r0, ks, 4*t+2) << 16 | code(s, st, r0, ks, 4*t+3) << 24,
            code(s, st, r1, ks, 4*t+0) | code(s, st, r1, ks, 4*t+1) << 8 |
                code(s, st, r1, ks, 4*t+2) << 16 | code(s, st, r1, ks, 4*t+3) << 24,
            code(s, st, r0, ks, 16+4*t+0) | code(s, st, r0, ks, 16+4*t+1) << 8 |
                code(s, st, r0, ks, 16+4*t+2) << 16 | code(s, st, r0, ks, 16+4*t+3) << 24,
            code(s, st, r1, ks, 16+4*t+0) | code(s, st, r1, ks, 16+4*t+1) << 8 |
                code(s, st, r1, ks, 16+4*t+2) << 16 | code(s, st, r1, ks, 16+4*t+3) << 24);
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        return make_float2(st.sc[h][0][ks], st.sc[h][1][ks]);
    }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int c = p / 4;
        const int lbase = p % 4;
        float res[2];
        const char * meta = sb + 2*MOE_PACKED_IQ2XS_CHUNK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16*mg + g + 8*h;
            const uint16_t * q2 = (const uint16_t *) (sb + c*MOE_PACKED_IQ2XS_CHUNK + row*32 + 8*lbase);
            const int ls0 = ((const uint8_t *) (meta + off_sc))[row*8 + p] & 0x0F;
            const int ls1 = ((const uint8_t *) (meta + off_sc))[row*8 + p] >> 4;
            int sumi0 = 0;
            int sumi1 = 0;
#pragma unroll
            for (int l = 0; l < 8; l += 2) {
                const uint2 grid_pos = ((const uint2 *) iq2xs_grid)[q2[l/2] & 511];
                const uint32_t signs = unpack_ksigns(q2[l/2] >> 9);
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                const int u0 = get_int_b4((y + p)->qs, l+0);
                const int u1 = get_int_b4((y + p)->qs, l+1);
                if (l < 4) {
                    sumi0 = ggml_cuda_dp4a(__vsub4(grid_pos.x ^ signs0, signs0), u0, sumi0);
                    sumi0 = ggml_cuda_dp4a(__vsub4(grid_pos.y ^ signs1, signs1), u1, sumi0);
                } else {
                    sumi1 = ggml_cuda_dp4a(__vsub4(grid_pos.x ^ signs0, signs0), u0, sumi1);
                    sumi1 = ggml_cuda_dp4a(__vsub4(grid_pos.y ^ signs1, signs1), u1, sumi1);
                }
            }
            const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
            res[h] = __half2float(((const half *) meta)[row]) * __low2float((y + p)->ds) * sumi;
        }
        return make_float2(res[0], res[1]);
    }
};

template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ2S> {
    using block_t = block_iq2_s;
    static constexpr ggml_type type      = GGML_TYPE_IQ2_S;
    static constexpr int chunk_bytes     = MOE_PACKED_IQ2S_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_IQ2S_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_D16;
    static constexpr int off_qh          = MOE_PACKED_I*sizeof(half);
    static constexpr int off_sc          = off_qh + MOE_PACKED_I*8;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_IQ2S_CHUNK;
            const int r0 = mg*16 + lane/4;
            const int r1 = r0 + 8;
            ((uint32_t *) (oc + r0*32 + 8*t))[0] = get_int_b1(b0->qs + c*16, 2*t + 0);
            ((uint32_t *) (oc + r0*32 + 8*t))[1] = get_int_b1(b0->qs + c*16, 2*t + 1);
            ((uint32_t *) (oc + r1*32 + 8*t))[0] = get_int_b1(b1->qs + c*16, 2*t + 0);
            ((uint32_t *) (oc + r1*32 + 8*t))[1] = get_int_b1(b1->qs + c*16, 2*t + 1);
            *(uint32_t *) (oc + r0*32 + 16 + 4*t) = get_int_b1(b0->qs + QK_K/8 + c*16, t);
            *(uint32_t *) (oc + r1*32 + 16 + 4*t) = get_int_b1(b1->qs + QK_K/8 + c*16, t);
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_IQ2S_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const block_t & b = h ? b1[0] : b0[0];
                ((half *) meta)[r] = b.d;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    ((uint8_t *) (meta + off_qh))[r*8 + i] = b.qh[i];
                    ((uint8_t *) (meta + off_sc))[r*8 + i] = b.scales[i];
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        for (int c = 0; c < 2; ++c) {
            const char * oc = in + c*MOE_PACKED_IQ2S_CHUNK;
            for (int i = 0; i < 16; ++i) b.qs[c*16 + i] = ((const uint8_t *) (oc + r*32))[i];
            for (int i = 0; i < 16; ++i) b.qs[QK_K/8 + c*16 + i] = ((const uint8_t *) (oc + r*32 + 16))[i];
        }
        const char * meta = in + 2*MOE_PACKED_IQ2S_CHUNK;
        b.d = ((const half *) meta)[r];
        for (int i = 0; i < 8; ++i) {
            b.qh[i] = ((const uint8_t *) (meta + off_qh))[r*8 + i];
            b.scales[i] = ((const uint8_t *) (meta + off_sc))[r*8 + i];
        }
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }
    struct frag_state { float sc[2][2][4]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * meta = s + MOE_PACKED_IQ2S_CHUNK;
        const uint8_t * scales = (const uint8_t *) (meta + off_sc);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            const float d = __half2float(((const half *) meta)[r]);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const uint8_t ls = scales[r*8 + st.c*4 + ks];
                st.sc[h][0][ks] = d * ((ls & 0x0F) + 0.5f) / 4.0f;
                st.sc[h][1][ks] = d * ((ls >> 4) + 0.5f) / 4.0f;
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint8_t code(const char * s, const char * meta, const frag_state & st, const int row, const int ks, const int v) {
        const int l = v/8;
        const uint8_t q = ((const uint8_t *) s)[row*32 + 4*ks + l];
        const uint8_t qh = ((const uint8_t *) (meta + off_qh))[row*8 + st.c*4 + ks];
        const uint8_t sign = ((const uint8_t *) (s + row*32 + 16))[4*ks + l];
        const uint8_t * grid = (const uint8_t *) (iq2s_grid + (q | ((qh << (8-2*l)) & 0x300)));
        return (uint8_t) (sign & kmask_iq2xs[v%8] ? -(int) grid[v%8] : (int) grid[v%8]);
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED_VARS(tab, st);
        const char * meta = s + MOE_PACKED_IQ2S_CHUNK;
        const int g = lane/4, t = lane%4, r0 = mg*16 + g, r1 = r0 + 8;
        return make_uint4(code(s, meta, st, r0, ks, 4*t+0) | code(s, meta, st, r0, ks, 4*t+1) << 8 |
                code(s, meta, st, r0, ks, 4*t+2) << 16 | code(s, meta, st, r0, ks, 4*t+3) << 24,
            code(s, meta, st, r1, ks, 4*t+0) | code(s, meta, st, r1, ks, 4*t+1) << 8 |
                code(s, meta, st, r1, ks, 4*t+2) << 16 | code(s, meta, st, r1, ks, 4*t+3) << 24,
            code(s, meta, st, r0, ks, 16+4*t+0) | code(s, meta, st, r0, ks, 16+4*t+1) << 8 |
                code(s, meta, st, r0, ks, 16+4*t+2) << 16 | code(s, meta, st, r0, ks, 16+4*t+3) << 24,
            code(s, meta, st, r1, ks, 16+4*t+0) | code(s, meta, st, r1, ks, 16+4*t+1) << 8 |
                code(s, meta, st, r1, ks, 16+4*t+2) << 16 | code(s, meta, st, r1, ks, 16+4*t+3) << 24);
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) { return make_float2(st.sc[h][0][ks], st.sc[h][1][ks]); }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int c = p / 4, lbase = p % 4;
        const char * ch = sb + c*MOE_PACKED_IQ2S_CHUNK;
        const char * meta = sb + 2*MOE_PACKED_IQ2S_CHUNK;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16*mg + g + 8*h;
            const uint8_t * qs = (const uint8_t *) (ch + row*32);
            const uint8_t * signs = qs + 16;
            const uint8_t qh = ((const uint8_t *) (meta + off_qh))[row*8 + p];
            const uint8_t * sc = (const uint8_t *) (meta + off_sc);
            const int ls0 = sc[row*8 + p] & 0x0F;
            const int ls1 = sc[row*8 + p] >> 4;
            int sumi0 = 0, sumi1 = 0;
#pragma unroll
            for (int l = 0; l < 8; l += 2) {
                const int * grid_pos = (const int *) (iq2s_grid + (qs[4*lbase + l/2] | ((qh << (8-l)) & 0x300)));
                const uint8_t sign0 = signs[4*lbase + l/2];
                const int s0 = __vcmpne4(((sign0 & 0x03) << 7) | ((sign0 & 0x0C) << 21), 0);
                const int s1 = __vcmpne4(((sign0 & 0x30) << 3) | ((sign0 & 0xC0) << 17), 0);
                const int u0 = get_int_b4((y + p)->qs, l+0);
                const int u1 = get_int_b4((y + p)->qs, l+1);
                if (l < 4) {
                    sumi0 = ggml_cuda_dp4a(__vsub4(grid_pos[0] ^ s0, s0), u0, sumi0);
                    sumi0 = ggml_cuda_dp4a(__vsub4(grid_pos[1] ^ s1, s1), u1, sumi0);
                } else {
                    sumi1 = ggml_cuda_dp4a(__vsub4(grid_pos[0] ^ s0, s0), u0, sumi1);
                    sumi1 = ggml_cuda_dp4a(__vsub4(grid_pos[1] ^ s1, s1), u1, sumi1);
                }
            }
            const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
            res[h] = __half2float(((const half *) meta)[row]) * __low2float((y + p)->ds) * sumi;
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ2_K keeps the original two-bit qs bytes in each 128-value chunk. The extra bits and the two 16-value scales per 32-value group
// remain in row metadata; fragment() applies the same iq2nl lookup as the ordinary q8_0_16 MMQ loader.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ2K> {
    using block_t = block_iq2_k;
    static constexpr ggml_type type      = GGML_TYPE_IQ2_K;
    static constexpr int chunk_bytes     = MOE_PACKED_IQ2K_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_IQ2K_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_D16;
    static constexpr int off_extra       = MOE_PACKED_I*sizeof(half);
    static constexpr int off_sc          = off_extra + MOE_PACKED_I*sizeof(uint16_t);

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_IQ2K_CHUNK;
            const int r0 = mg*16 + lane/4;
            const int r1 = r0 + 8;
            ((uint32_t *) (oc + r0*32 + 8*t))[0] = get_int_b1(b0->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + r0*32 + 8*t))[1] = get_int_b1(b0->qs + c*32, 2*t + 1);
            ((uint32_t *) (oc + r1*32 + 8*t))[0] = get_int_b1(b1->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + r1*32 + 8*t))[1] = get_int_b1(b1->qs + c*32, 2*t + 1);
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_IQ2K_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const block_t & b = h ? b1[0] : b0[0];
                ((half *) meta)[r] = b.d;
                ((uint16_t *) (meta + off_extra))[r] = b.extra;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    ((uint8_t *) (meta + off_sc))[r*8 + i] = b.scales[i];
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        for (int c = 0; c < 2; ++c) {
            const uint8_t * q = (const uint8_t *) (in + c*MOE_PACKED_IQ2K_CHUNK) + r*32;
            for (int i = 0; i < 32; ++i) b.qs[c*32 + i] = q[i];
        }
        const char * meta = in + 2*MOE_PACKED_IQ2K_CHUNK;
        b.d = ((const half *) meta)[r];
        b.extra = ((const uint16_t *) (meta + off_extra))[r];
        for (int i = 0; i < 8; ++i) {
            b.scales[i] = ((const uint8_t *) (meta + off_sc))[r*8 + i];
        }
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { float sc[2][2][4]; uint16_t extra[2]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * meta = s + MOE_PACKED_IQ2K_CHUNK;
        const uint8_t * scales = (const uint8_t *) (meta + off_sc);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            const float d = __half2float(((const half *) meta)[r]);
            st.extra[h] = ((const uint16_t *) (meta + off_extra))[r];
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const uint8_t ls = scales[r*8 + st.c*4 + ks];
                st.sc[h][0][ks] = d * ((ls & 0x0F) - 8);
                st.sc[h][1][ks] = d * ((ls >> 4) - 8);
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        // the extra bit of k step k and 16-value half hv of the superblock is bit 8*st.c + 2*k + hv of extra
        const int g = lane/4, t = lane%4, a = ks & 1;
        uint32_t r[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t * q = (const uint32_t *) (s + (mg*16 + g + 8*h)*32);
#pragma unroll
            for (int hv = 0; hv < 2; ++hv) {
                const uint32_t e = (uint32_t) st.extra[h] >> (8*st.c + 2*a + hv);
                r[h][hv] = moe_iq2_reg(q[4*hv + t], e & 1, (e >> 4) & 1, ks);
            }
        }
        return make_uint4(r[0][0], r[1][0], r[0][1], r[1][1]);
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        return make_float2(st.sc[h][0][ks], st.sc[h][1][ks]);
    }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int c = p / 4;
        const int seg = p % 4;
        const char * ch = sb + c*MOE_PACKED_IQ2K_CHUNK;
        const char * meta = sb + 2*MOE_PACKED_IQ2K_CHUNK;
        const block_q8_1 * y0 = y + 4*c;
        const int32_t * q8_1 = (const int32_t *) y0[0].qs + 2*seg;
        const int32_t * q8_2 = (const int32_t *) y0[1].qs + 2*seg;
        const int32_t * q8_3 = (const int32_t *) y0[2].qs + 2*seg;
        const int32_t * q8_4 = (const int32_t *) y0[3].qs + 2*seg;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16*mg + g + 8*h;
            const uint8_t * q = (const uint8_t *) (ch + row*32 + 8*seg);
            const uint32_t q0 = ((const uint32_t *) q)[0];
            const uint32_t q1 = ((const uint32_t *) q)[1];
            const uint16_t extra = ((const uint16_t *) (meta + off_extra))[row] >> (8*c + seg/2);
            const uint8_t * scales = (const uint8_t *) (meta + off_sc) + row*8 + 4*c;
            const int shift = 4*((seg/2) % 2);
            const int s0 = ((scales[0] >> shift) & 0x0F) - 8;
            const int s1 = ((scales[1] >> shift) & 0x0F) - 8;
            const int s2 = ((scales[2] >> shift) & 0x0F) - 8;
            const int s3 = ((scales[3] >> shift) & 0x0F) - 8;
            int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;

#ifdef __CUDA_ARCH__
            const uint32_t extra32 = uint32_t(extra & 0xFF) * 0x01010101;
            const uint32_t extra32_1 = (extra32 << 2) & 0x44444444;
            const uint32_t extra32_2 = (extra32 << 0) & 0x44444444;
            uint32_t val1 = ((q0 >> 0) & 0x33333333) | extra32_1;
            uint32_t val2 = ((q1 >> 0) & 0x33333333) | extra32_1;
            int2 v1 = get_int_from_table_8(val1, iq2nl_values);
            int2 v2 = get_int_from_table_8(val2, iq2nl_values);
            sumi1 = ggml_cuda_dp4a(v2.x, q8_1[1], ggml_cuda_dp4a(v1.x, q8_1[0], 0)) * s0;
            sumi3 = ggml_cuda_dp4a(v2.y, q8_3[1], ggml_cuda_dp4a(v1.y, q8_3[0], 0)) * s2;
            val1 = ((q0 >> 2) & 0x33333333) | extra32_2;
            val2 = ((q1 >> 2) & 0x33333333) | extra32_2;
            v1 = get_int_from_table_8(val1, iq2nl_values);
            v2 = get_int_from_table_8(val2, iq2nl_values);
            sumi2 = ggml_cuda_dp4a(v2.x, q8_2[1], ggml_cuda_dp4a(v1.x, q8_2[0], 0)) * s1;
            sumi4 = ggml_cuda_dp4a(v2.y, q8_4[1], ggml_cuda_dp4a(v1.y, q8_4[0], 0)) * s3;
#else
            const int * all_values = (const int *) iq2k_table;
            uint32_t aux32[2];
            int v1, v2;
            aux32[0] = ((q0 >> 0) & 0x03030303); aux32[1] = ((q1 >> 0) & 0x03030303);
            const int * values = all_values + ((extra & 0x01) << 8);
            v1 = int_from_table_4(aux32[0], values);
            v2 = int_from_table_4(aux32[1], values);
            sumi1 = ggml_cuda_dp4a(v2, q8_1[1], ggml_cuda_dp4a(v1, q8_1[0], 0)) * s0;
            aux32[0] = ((q0 >> 2) & 0x03030303); aux32[1] = ((q1 >> 2) & 0x03030303);
            values = all_values + ((extra & 0x04) << 6);
            v1 = int_from_table_4(aux32[0], values);
            v2 = int_from_table_4(aux32[1], values);
            sumi2 = ggml_cuda_dp4a(v2, q8_2[1], ggml_cuda_dp4a(v1, q8_2[0], 0)) * s1;
            aux32[0] = ((q0 >> 4) & 0x03030303); aux32[1] = ((q1 >> 4) & 0x03030303);
            values = all_values + ((extra & 0x10) << 4);
            v1 = int_from_table_4(aux32[0], values);
            v2 = int_from_table_4(aux32[1], values);
            sumi3 = ggml_cuda_dp4a(v2, q8_3[1], ggml_cuda_dp4a(v1, q8_3[0], 0)) * s2;
            aux32[0] = ((q0 >> 6) & 0x03030303); aux32[1] = ((q1 >> 6) & 0x03030303);
            values = all_values + ((extra & 0x40) << 2);
            v1 = int_from_table_4(aux32[0], values);
            v2 = int_from_table_4(aux32[1], values);
            sumi4 = ggml_cuda_dp4a(v2, q8_4[1], ggml_cuda_dp4a(v1, q8_4[0], 0)) * s3;
#endif
            const float d = __half2float(((const half *) meta)[row]);
            res[h] = d * (__low2float(y0[0].ds) * sumi1 + __low2float(y0[1].ds) * sumi2 +
                __low2float(y0[2].ds) * sumi3 + __low2float(y0[3].ds) * sumi4);
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ3_K keeps the original two-bit qs bytes in each 128-value chunk. The qh, extra and scale fields stay in row metadata, with the
// same two 16-value scales per 32-value group that the q8_0_16 MMQ loader constructs.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ3K> {
    using block_t = block_iq3_k;
    static constexpr ggml_type type      = GGML_TYPE_IQ3_K;
    static constexpr int chunk_bytes     = MOE_PACKED_IQ3K_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_IQ3K_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_D16;
    static constexpr int off_extra       = MOE_PACKED_I*sizeof(half);
    static constexpr int off_sh          = off_extra + MOE_PACKED_I*sizeof(uint16_t);
    static constexpr int off_sl          = off_sh + MOE_PACKED_I*sizeof(uint16_t);
    static constexpr int off_qh          = off_sl + MOE_PACKED_I*8;

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_IQ3K_CHUNK;
            const int r0 = mg*16 + lane/4;
            const int r1 = r0 + 8;
            ((uint32_t *) (oc + r0*32 + 8*t))[0] = get_int_b1(b0->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + r0*32 + 8*t))[1] = get_int_b1(b0->qs + c*32, 2*t + 1);
            ((uint32_t *) (oc + r1*32 + 8*t))[0] = get_int_b1(b1->qs + c*32, 2*t + 0);
            ((uint32_t *) (oc + r1*32 + 8*t))[1] = get_int_b1(b1->qs + c*32, 2*t + 1);
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_IQ3K_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const block_t & b = h ? b1[0] : b0[0];
                ((half *) meta)[r] = b.d;
                ((uint16_t *) (meta + off_extra))[r] = b.extra;
                ((uint16_t *) (meta + off_sh))[r] = b.scales_h;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    ((uint8_t *) (meta + off_sl))[r*8 + i] = b.scales_l[i];
                }
#pragma unroll
                for (int i = 0; i < QK_K/8; ++i) {
                    ((uint8_t *) (meta + off_qh))[r*(QK_K/8) + i] = b.qh[i];
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        for (int c = 0; c < 2; ++c) {
            const uint8_t * q = (const uint8_t *) (in + c*MOE_PACKED_IQ3K_CHUNK) + r*32;
#pragma unroll
            for (int i = 0; i < 32; ++i) b.qs[c*32 + i] = q[i];
        }
        const char * meta = in + 2*MOE_PACKED_IQ3K_CHUNK;
        b.d = ((const half *) meta)[r];
        b.extra = ((const uint16_t *) (meta + off_extra))[r];
        b.scales_h = ((const uint16_t *) (meta + off_sh))[r];
#pragma unroll
        for (int i = 0; i < 8; ++i) b.scales_l[i] = ((const uint8_t *) (meta + off_sl))[r*8 + i];
#pragma unroll
        for (int i = 0; i < QK_K/8; ++i) b.qh[i] = ((const uint8_t *) (meta + off_qh))[r*(QK_K/8) + i];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { float sc[2][2][4]; uint16_t extra[2]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * meta = s + MOE_PACKED_IQ3K_CHUNK;
        const uint8_t * scales = (const uint8_t *) (meta + off_sl);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            const float d = __half2float(((const half *) meta)[r]);
            st.extra[h] = ((const uint16_t *) (meta + off_extra))[r];
            const uint16_t sh = ((const uint16_t *) (meta + off_sh))[r];
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ib = 4*st.c + ks;
                const uint8_t sl = scales[r*8 + ib];
                const uint16_t signs = sh >> 2*ib;
                st.sc[h][0][ks] = d * ((2*(sl & 0x0F) + 1) * (signs & 1 ? -1 : 1));
                st.sc[h][1][ks] = d * ((2*(sl >> 4) + 1) * (signs & 2 ? -1 : 1));
            }
        }
        return st;
    }

    // The second iq3nl_values table is the first plus 4 in each byte (no byte carries), so an extra bit adds 0x04040404.
    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        // the extra bit of k step k and 16-value half hv of the superblock is bit 8*st.c + 2*k + hv of extra, its high bit of
        // value v is bit 4*st.c + k of the qh byte v
        const int g = lane/4, t = lane%4;
        uint32_t r[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = mg*16 + g + 8*h;
            const uint32_t * q  = (const uint32_t *) (s + row*32);
            const uint32_t * qh = (const uint32_t *) (s + MOE_PACKED_IQ3K_CHUNK + off_qh + row*32);
#pragma unroll
            for (int hv = 0; hv < 2; ++hv) {
                const uint32_t e = (st.extra[h] >> (8*st.c + 2*ks + hv)) & 1;
                r[h][hv] = moe_iq3_reg(q[4*hv + t], qh[4*hv + t] >> (4*st.c), ks) + e*0x04040404u;
            }
        }
        return make_uint4(r[0][0], r[1][0], r[0][1], r[1][1]);
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        return make_float2(st.sc[h][0][ks], st.sc[h][1][ks]);
    }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int ib128 = p/4;
        const int il8 = p%4;
        const int shift = 4*(il8/2);
        const char * ch = sb + ib128*MOE_PACKED_IQ3K_CHUNK;
        const char * meta = sb + 2*MOE_PACKED_IQ3K_CHUNK;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16*mg + g + 8*h;
            const uint16_t * ql = (const uint16_t *) (ch + row*32) + 4*il8;
            const uint16_t * qh = (const uint16_t *) (meta + off_qh + row*32) + 4*il8;
            const int hshift = 4*(1-ib128);
            const uint16_t sh = ((const uint16_t *) (meta + off_sh))[row] >> (8*ib128 + il8/2);
            const uint8_t extra = ((const uint16_t *) (meta + off_extra))[row] >> (8*ib128 + il8/2);
            const uint32_t extra32 = uint32_t(extra) * 0x01010101;
            const uint32_t extra32_1 = (extra32 << 3) & 0x88888888;
            const uint32_t extra32_2 = (extra32 << 1) & 0x88888888;
            int sumi[4] = { 0, 0, 0, 0 };
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
                const uint32_t vh = (qh[2*i+0] | (qh[2*i+1] << 16)) << hshift;
                const uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh >> 2) & 0x04040404) | ((vh >> 0) & 0x40404040);
                const uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh >> 3) & 0x04040404) | ((vh >> 1) & 0x40404040);
                const int2 v1 = get_int_from_table_16(val1, iq3nl_values);
                const int2 v2 = get_int_from_table_16(val2, iq3nl_values);
                const int * q8 = (const int *) (y + 4*ib128 + 0)->qs + 2*il8;
                sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);
                q8 = (const int *) (y + 4*ib128 + 1)->qs + 2*il8;
                sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);
                q8 = (const int *) (y + 4*ib128 + 2)->qs + 2*il8;
                sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);
                q8 = (const int *) (y + 4*ib128 + 3)->qs + 2*il8;
                sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
            }
            const uint16_t * sl16 = (const uint16_t *) (meta + off_sl + row*8) + 2*ib128;
            uint32_t aux32 = ((((sl16[0] | (sl16[1] << 16)) >> shift) & 0x0f0f0f0f) << 1) | 0x01010101;
            const uint8_t * aux8 = (const uint8_t *) &aux32;
            const float d = __half2float(((const half *) meta)[row]);
            res[h] = d * (__low2float((y + 4*ib128 + 0)->ds) * aux8[0] * (sh & 0x01 ? -1 : 1) * sumi[0] +
                    __low2float((y + 4*ib128 + 1)->ds) * aux8[1] * (sh & 0x04 ? -1 : 1) * sumi[1] +
                    __low2float((y + 4*ib128 + 2)->ds) * aux8[2] * (sh & 0x10 ? -1 : 1) * sumi[2] +
                    __low2float((y + 4*ib128 + 3)->ds) * aux8[3] * (sh & 0x40 ? -1 : 1) * sumi[3]);
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ4_K uses the IQ4_KS payload layout: qs words in m16n8k32 A-fragment order with the nibbles in value order. The two extra bits
// per 32 values and the 6-bit scales remain in row metadata, matching the q8_0_16 MMQ loader. The second IQ4_K table is the first
// plus 4 in each byte (no byte carries), so an extra bit adds moe_iq4k_shift to the IQ4_NL-style decode of the first table.
static __device__ __forceinline__ uint32_t moe_iq4k_shift(const uint32_t extra, const int bit) {
    return ((extra >> bit) & 1) * 0x04040404u;
}

template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ4K> {
    using block_t = block_iq4_k;
    static constexpr ggml_type type      = GGML_TYPE_IQ4_K;
    static constexpr int chunk_bytes     = MOE_PACKED_IQ4K_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_IQ4K_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_D16;
    static constexpr int off_extra       = MOE_PACKED_I*sizeof(half);
    static constexpr int off_sh          = off_extra + MOE_PACKED_I*sizeof(uint16_t);
    static constexpr int off_sl          = off_sh + MOE_PACKED_I*(QK_K/64);

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const block_t * b[2] = { b0, b1 };
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    ((uint32_t *) (out + c*chunk_bytes))[((mg*4 + ks)*2 + h)*WARP_SIZE + lane] =
                        moe_iq4xs_word(get_int_b4(b[h]->qs, 4*(4*c + ks) + t));
                }
            }
        }
        if (t == 0) {
            char * meta = out + 2*chunk_bytes;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                ((half *) meta)[r] = b[h]->d;
                ((uint16_t *) (meta + off_extra))[r] = b[h]->extra;
#pragma unroll
                for (int i = 0; i < QK_K/64; ++i) {
                    ((uint8_t *) (meta + off_sh))[r*(QK_K/64) + i] = b[h]->scales_h[i];
                }
#pragma unroll
                for (int i = 0; i < QK_K/32; ++i) {
                    ((uint8_t *) (meta + off_sl))[r*(QK_K/32) + i] = b[h]->scales_l[i];
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        const int mg = r / 16;
        const int g  = r % 8;
        const int h  = (r % 16) / 8;
#pragma unroll
        for (int ib = 0; ib < QK_K/32; ++ib) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t w = ((const uint32_t *) (in + (ib/4)*chunk_bytes))[(mg*4 + ib%4)*2*WARP_SIZE + h*WARP_SIZE + 4*g + t];
                const uint32_t q = moe_iq4xs_qs(w);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    b.qs[16*ib + 4*t + i] = q >> (8*i);
                }
            }
        }
        const char * meta = in + 2*chunk_bytes;
        b.d = ((const half *) meta)[r];
        b.extra = ((const uint16_t *) (meta + off_extra))[r];
#pragma unroll
        for (int i = 0; i < QK_K/64; ++i) b.scales_h[i] = ((const uint8_t *) (meta + off_sh))[r*(QK_K/64) + i];
#pragma unroll
        for (int i = 0; i < QK_K/32; ++i) b.scales_l[i] = ((const uint8_t *) (meta + off_sl))[r*(QK_K/32) + i];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { float sc[2][2][4]; uint16_t extra[2]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * meta = s + chunk_bytes;
        const uint8_t * sh_all = (const uint8_t *) (meta + off_sh);
        const uint8_t * sl_all = (const uint8_t *) (meta + off_sl);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            const float d = __half2float(((const half *) meta)[r]);
            st.extra[h] = ((const uint16_t *) (meta + off_extra))[r];
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ib = 4*st.c + ks;
                const uint8_t sh = sh_all[r*4 + ib/2] >> 4*(ib%2);
                const uint8_t sl = sl_all[r*8 + ib];
                st.sc[h][0][ks] = d * (((sl & 0x0F) | ((sh << 4) & 0x30)) - 32);
                st.sc[h][1][ks] = d * (((sl >> 4) | ((sh << 2) & 0x30)) - 32);
            }
        }
        return st;
    }

    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        const int ib = 4*st.c + ks;
        const uint32_t * w = (const uint32_t *) s + (mg*4 + ks)*2*WARP_SIZE + lane;
        const int2 v0 = moe_iq4_decode(w[0], iq4k_values);
        const int2 v1 = moe_iq4_decode(w[WARP_SIZE], iq4k_values);
        return make_uint4(v0.x + moe_iq4k_shift(st.extra[0], 2*ib), v1.x + moe_iq4k_shift(st.extra[1], 2*ib),
            v0.y + moe_iq4k_shift(st.extra[0], 2*ib + 1), v1.y + moe_iq4k_shift(st.extra[1], 2*ib + 1));
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        return make_float2(st.sc[h][0][ks], st.sc[h][1][ks]);
    }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int ib = p;
        const int c  = ib/4;
        const int ks = ib%4;
        const char * meta = sb + 2*chunk_bytes;
        const int * q8 = (const int *) (y + ib)->qs;
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16*mg + g + 8*h;
            const uint32_t * q = (const uint32_t *) (sb + c*chunk_bytes) + (mg*4 + ks)*2*WARP_SIZE + h*WARP_SIZE + 4*g;
            const uint16_t extra = ((const uint16_t *) (meta + off_extra))[row];
            const uint32_t s1 = moe_iq4k_shift(extra, 2*ib);
            const uint32_t s2 = moe_iq4k_shift(extra, 2*ib + 1);
            int sumi1 = 0, sumi2 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int2 v = moe_iq4_decode(q[j], iq4k_values);
                sumi1 = ggml_cuda_dp4a((int) (v.x + s1), q8[j+0], sumi1);
                sumi2 = ggml_cuda_dp4a((int) (v.y + s2), q8[j+4], sumi2);
            }
            const uint8_t sh = ((const uint8_t *) (meta + off_sh))[row*4 + ib/2] >> 4*(ib%2);
            const uint8_t sl = ((const uint8_t *) (meta + off_sl))[row*8 + ib];
            const int ls1 = ((sl & 0x0F) | ((sh << 4) & 0x30)) - 32;
            const int ls2 = ((sl >> 4) | ((sh << 2) & 0x30)) - 32;
            res[h] = __half2float(((const half *) meta)[row]) * __low2float((y + ib)->ds) * (sumi1*ls1 + sumi2*ls2);
        }
        return make_float2(res[0], res[1]);
    }
};

// IQ5_K keeps the four-bit qs bytes in each 128-value chunk and the complete qh array in row metadata. This avoids inventing a
// transformed high-bit representation while retaining the exact 22528-byte GGUF block size.
template <> struct moe_packed_q<GGML_CUDA_MOE_PACKED_IQ5K> {
    using block_t = block_iq5_k;
    static constexpr ggml_type type      = GGML_TYPE_IQ5_K;
    static constexpr int chunk_bytes     = MOE_PACKED_IQ5K_CHUNK;
    static constexpr int sb_bytes        = MOE_PACKED_IQ5K_BLOCK;
    static constexpr int tab_words       = 0;
    static constexpr int family          = MOE_PACKED_FAMILY_D16;
    static constexpr int off_extra       = MOE_PACKED_I*sizeof(half);
    static constexpr int off_sh          = off_extra + MOE_PACKED_I*sizeof(uint16_t);
    static constexpr int off_sl          = off_sh + MOE_PACKED_I*(QK_K/64);
    static constexpr int off_qh          = off_sl + MOE_PACKED_I*(QK_K/32);

    static __device__ __forceinline__ void pack(const block_t * b0, const block_t * b1, char * out, const int mg, const int lane) {
        const int t = lane % 4;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            char * oc = out + c*MOE_PACKED_IQ5K_CHUNK;
            const int r0 = mg*16 + lane/4;
            const int r1 = r0 + 8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                ((uint32_t *) (oc + r0*64 + 16*j + 4*t))[0] = get_int_b1(b0->qs + c*64, 4*j + t);
                ((uint32_t *) (oc + r1*64 + 16*j + 4*t))[0] = get_int_b1(b1->qs + c*64, 4*j + t);
            }
        }
        if (t == 0) {
            char * meta = out + 2*MOE_PACKED_IQ5K_CHUNK;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mg*16 + lane/4 + 8*h;
                const block_t & b = h ? b1[0] : b0[0];
                ((half *) meta)[r] = b.d;
                ((uint16_t *) (meta + off_extra))[r] = b.extra;
#pragma unroll
                for (int i = 0; i < QK_K/64; ++i) {
                    ((uint8_t *) (meta + off_sh))[r*(QK_K/64) + i] = b.scales_h[i];
                }
#pragma unroll
                for (int i = 0; i < QK_K/32; ++i) {
                    ((uint8_t *) (meta + off_sl))[r*(QK_K/32) + i] = b.scales_l[i];
                }
#pragma unroll
                for (int i = 0; i < QK_K/8; ++i) {
                    ((uint8_t *) (meta + off_qh))[r*(QK_K/8) + i] = b.qh[i];
                }
            }
        }
    }

    static __device__ __forceinline__ void unpack(const char * in, const int r, block_t & b) {
        for (int c = 0; c < 2; ++c) {
            const uint8_t * q = (const uint8_t *) (in + c*MOE_PACKED_IQ5K_CHUNK) + r*64;
#pragma unroll
            for (int i = 0; i < 64; ++i) b.qs[c*64 + i] = q[i];
        }
        const char * meta = in + 2*MOE_PACKED_IQ5K_CHUNK;
        b.d = ((const half *) meta)[r];
        b.extra = ((const uint16_t *) (meta + off_extra))[r];
#pragma unroll
        for (int i = 0; i < QK_K/64; ++i) b.scales_h[i] = ((const uint8_t *) (meta + off_sh))[r*(QK_K/64) + i];
#pragma unroll
        for (int i = 0; i < QK_K/32; ++i) b.scales_l[i] = ((const uint8_t *) (meta + off_sl))[r*(QK_K/32) + i];
#pragma unroll
        for (int i = 0; i < QK_K/8; ++i) b.qh[i] = ((const uint8_t *) (meta + off_qh))[r*(QK_K/8) + i];
    }

    static __device__ __forceinline__ uint32_t tab(const int i) { GGML_UNUSED(i); return 0; }

    struct frag_state { float sc[2][2][4]; uint16_t extra[2]; int c; };

    static __device__ __forceinline__ frag_state load_state(const char * s, const int mg, const int lane, const int kc) {
        frag_state st;
        st.c = kc % 2;
        const char * meta = s + MOE_PACKED_IQ5K_CHUNK;
        const uint8_t * sh_all = (const uint8_t *) (meta + off_sh);
        const uint8_t * sl_all = (const uint8_t *) (meta + off_sl);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mg*16 + lane/4 + 8*h;
            const float d = __half2float(((const half *) meta)[r]);
            st.extra[h] = ((const uint16_t *) (meta + off_extra))[r];
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ib64 = 2*st.c + ks/2;
                const int part = ks%2;
                const uint8_t sh = sh_all[r*4 + ib64];
                const uint8_t sl = sl_all[r*8 + 2*ib64 + part];
                const int hi0 = part == 0 ? ((sh << 4) & 0x30) : (sh & 0x30);
                const int hi1 = part == 0 ? ((sh << 2) & 0x30) : ((sh >> 2) & 0x30);
                st.sc[h][0][ks] = d * (((sl & 0x0F) | hi0) - 32);
                st.sc[h][1][ks] = d * (((sl >> 4) | hi1) - 32);
            }
        }
        return st;
    }

    // The qs words of rows g and g + 8 hold the 4-bit indices of the k steps ks & ~1 (low nibbles) and ks | 1 (high nibbles), so the
    // two fragments of a pair share the two 16-entry lookups. The fifth index bit is bit 4*st.c + ks of the qh byte of the value; the
    // second iq5nl_values table is the first plus 2 (the extra bit of k step k and 16-value half hv is bit 8*st.c + 2*k + hv of extra).
    static __device__ __forceinline__ uint4 fragment(const char * s, const uint32_t * tab, const frag_state & st, const int mg, const int ks,
            const int lane) {
        GGML_UNUSED(tab);
        const int g = lane/4, t = lane%4;
        uint32_t r[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = mg*16 + g + 8*h;
            const uint32_t * q  = (const uint32_t *) (s + row*64 + 32*(ks/2));
            const uint32_t * qh = (const uint32_t *) (s + MOE_PACKED_IQ5K_CHUNK + off_qh + row*32);
#pragma unroll
            for (int hv = 0; hv < 2; ++hv) {
                int2 a, b;
                moe_iq5_lookup(q[4*hv + t], a, b);
                const uint32_t m = moe_iq3s_sign_mask(qh[4*hv + t] << (7 - 4*st.c - ks));
                const uint32_t e = (st.extra[h] >> (8*st.c + 2*ks + hv)) & 1;
                r[h][hv] = moe_iq5_value(moe_select(m, ks & 1 ? a.y : a.x, ks & 1 ? b.y : b.x), e*0x02020202u);
            }
        }
        return make_uint4(r[0][0], r[1][0], r[0][1], r[1][1]);
    }

    static __device__ __forceinline__ float2 scale(const frag_state & st, const int h, const int ks) {
        return make_float2(st.sc[h][0][ks], st.sc[h][1][ks]);
    }

    static __device__ __forceinline__ float2 vec_dot2(const char * sb, const int mg, const int g, const int p, const block_q8_1 * y) {
        const int i4 = p;
        const int c = i4/4;
        const char * ch = sb + c*MOE_PACKED_IQ5K_CHUNK;
        const char * meta = sb + 2*MOE_PACKED_IQ5K_CHUNK;
        const int32_t * q8_1 = (const int32_t *) (y + 2*(i4/2) + 0)->qs + 4*(i4%2);
        const int32_t * q8_2 = (const int32_t *) (y + 2*(i4/2) + 1)->qs + 4*(i4%2);
        float res[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16*mg + g + 8*h;
            const uint32_t * q4 = (const uint32_t *) (ch + row*64 + 16*(i4%4));
            const uint32_t * qh = (const uint32_t *) (meta + off_qh + row*32) + 4*(i4%2);
            const uint16_t extra = ((const uint16_t *) (meta + off_extra))[row] >> (4*(i4/2) + (i4%2));
            const int shift = 2*(i4/2);
            const uint32_t shift1 = (extra & 1)*0x02020202u;
            const uint32_t shift2 = ((extra >> 2) & 1)*0x02020202u;
            int sumi1 = 0, sumi2 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int2 a, b;
                moe_iq5_lookup(q4[j], a, b);
                const uint32_t hh = qh[j] >> shift;
                const uint32_t v1 = moe_iq5_value(moe_select(moe_iq3s_sign_mask(hh << 7), a.x, b.x), shift1);
                const uint32_t v2 = moe_iq5_value(moe_select(moe_iq3s_sign_mask(hh << 6), a.y, b.y), shift2);
                sumi1 = ggml_cuda_dp4a((int) v1, q8_1[j], sumi1);
                sumi2 = ggml_cuda_dp4a((int) v2, q8_2[j], sumi2);
            }
            const uint8_t sh = ((const uint8_t *) (meta + off_sh))[row*4 + i4/2] >> 2*(i4%2);
            const uint8_t * sl = (const uint8_t *) (meta + off_sl) + row*8;
            const int ls1 = (((sl[2*(i4/2)+0] >> 4*(i4%2)) & 0x0F) | ((sh << 4) & 0x30)) - 32;
            const int ls2 = (((sl[2*(i4/2)+1] >> 4*(i4%2)) & 0x0F) | ((sh << 0) & 0x30)) - 32;
            const float d = __half2float(((const half *) meta)[row]);
            res[h] = d * (__low2float((y + 2*(i4/2) + 0)->ds) * sumi1 * ls1 +
                __low2float((y + 2*(i4/2) + 1)->ds) * sumi2 * ls2);
        }
        return make_float2(res[0], res[1]);
    }
};
