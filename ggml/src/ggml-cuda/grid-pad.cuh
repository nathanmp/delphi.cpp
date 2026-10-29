#pragma once

#include "common.cuh"

// delphi: on RDNA4 (gfx1201, ROCm 7.2) a short kernel whose launch has a multiple of 64 Ki threads, or up to 3 blocks
// fewer, takes ~20 us instead of 2-5 us; a few blocks more do not (2026-10-29 probe: 253-256 blocks of 256 threads,
// 125-128 of 512, also with no stores). DeepSeek V4.1 4-token verification hits it in rope, quantize_q8_1, concat and
// the flash attention fixup; Qwen 3.8 decode in get_rows. Returns the block count along one grid dimension, with idle
// blocks added until the launch leaves that range. The kernel must tolerate the extra blocks (a bounds check or a grid
// stride loop). other_blocks is the product of the other grid dimensions. Launches of 1 Mi threads or more, and shapes
// where a few blocks more along this dimension cannot leave the range (a step of 64 Ki threads, e.g. get_rows over 5120
// columns), are left alone: the former run long enough not to notice, and padding the latter would never terminate.
static inline int64_t ggml_cuda_grid_pad(const int64_t blocks, const int64_t other_blocks, const int block_threads) {
#if defined(GGML_USE_HIP)
    if (GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        const int64_t step = other_blocks*block_threads;
        if (blocks*step >= (int64_t(1) << 20)) {
            return blocks;
        }
        for (int64_t padded = blocks; padded < blocks + 16; ++padded) {
            if ((65536 - (padded*step) % 65536) % 65536 >= 4*block_threads) {
                return padded;
            }
        }
        return blocks;
    }
#else
    GGML_UNUSED(other_blocks);
    GGML_UNUSED(block_threads);
#endif // defined(GGML_USE_HIP)
    return blocks;
}
