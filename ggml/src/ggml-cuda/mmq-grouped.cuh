#pragma once

// delphi.cpp: grouped routed-MoE MMQ. The expert grouping comes from the grouped schedule (ggml-moe-schedule.h). The work is a
// device-side list of column tiles (expert, first row inside the expert), one list per tile-width class J, instead of
// ceil(n_tokens/J) tiles for every expert. Each class is one stream-k launch whose tile count is read on the device, so the
// routing changes data only and never the launch topology. The tile arithmetic is mul_mat_q_process_tile, unchanged.

#include "mmq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_MMQ_GROUPED_AVAILABLE
#endif

struct mmq_grouped_args {
    const char * x; const int * y; const int32_t * ids_dst; const int32_t * expert_bounds; float * dst;
    const int2 * tiles; const int32_t * n_tiles; int64_t max_tiles; float * tmp_fixup;
    int64_t ncols_x; int64_t nrows_x; int64_t stride_row_x; int64_t ncols_y; int64_t stride_col_dst; int64_t stride_channel_x;
    int J;
    cudaEvent_t ev_fixup; // If not null, recorded between the kernel and the fixup kernel (GGML_CUDA_MOE_GROUPED_TIMING).
};

// Stream-k order: the row tile of src0 is the outer index, the entry of the tile list the inner index.
template <ggml_type type, int J, bool fallback, bool fixup>
static __device__ __forceinline__ void mul_mat_q_grouped_tile(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int2 * __restrict__ tiles, float * __restrict__ dst,
        float * __restrict__ tmp_fixup, const int n_tiles, const int tile, const int kb0_start, const int kb0_stop,
        const int nrows_x, const int stride_row_x, const int ncols_y, const int stride_col_dst, const int stride_channel_x) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.

    const int  it       = tile / n_tiles;
    const int2 t        = tiles[tile - it*n_tiles];
    const int  col_low  = expert_bounds[t.x] + t.y;
    const int  col_diff = expert_bounds[t.x + 1] - col_low;

    __syncthreads();
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }

        // The fixup buffer is contiguous. Columns after the expert's last row are computed but not written back.
        ids_dst_shared[j] = fixup ? j : (j < col_diff ? ids_dst[col_low + j] : 0);
    }
    __syncthreads();

    const int offset_x = t.x*stride_channel_x + it*I*stride_row_x;
    const int offset_y = col_low*(sizeof(block_q8_1_mmq)/sizeof(int));

    mul_mat_q_process_tile<type, J, fallback, fixup>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + it*I, tmp_fixup, nullptr,
         stride_row_x, ncols_y, stride_col_dst,
         nrows_x - it*I - 1, col_diff - 1, kb0_start, kb0_stop);
}

// The stream-k decomposition of mul_mat_q over n_tiles*nty tiles, with n_tiles read from device memory.
template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback), ggml_cuda_mmq_get_occupancy(type, J, fallback))
static __global__ void mul_mat_q_grouped(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int2 * __restrict__ tiles, const int32_t * __restrict__ n_tiles_ptr,
        float * __restrict__ dst, float * __restrict__ tmp_fixup, const uint3 blocks_per_ne00,
        const int nrows_x, const int stride_row_x, const int ncols_y, const int stride_col_dst, const int stride_channel_x) {

    // Skip unused template specializations for faster compilation:
    if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    const int n_tiles = *n_tiles_ptr;
    const int ntotal  = n_tiles*((nrows_x + I - 1) / I)*blocks_per_ne00.z;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc      = int64_t(blockIdx.x)    *ntotal / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1)*ntotal / gridDim.x;

    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        mul_mat_q_grouped_tile<type, J, fallback, false>
            (x, y, ids_dst, expert_bounds, tiles, dst, tmp_fixup, n_tiles, fastdiv(kbc, blocks_per_ne00), kb0_start, kb0_stop,
             nrows_x, stride_row_x, ncols_y, stride_col_dst, stride_channel_x);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_grouped_tile<type, J, fallback, true>
        (x, y, ids_dst, expert_bounds, tiles, dst, tmp_fixup, n_tiles, fastdiv(kbc, blocks_per_ne00), kb0_start, kb0_stop,
         nrows_x, stride_row_x, ncols_y, stride_col_dst, stride_channel_x);
}

// mul_mat_q_stream_k_fixup for the tile lists of mul_mat_q_grouped. It must use the same decomposition.
template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback)/2, 1)
static __global__ void mul_mat_q_grouped_fixup(
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, const int2 * __restrict__ tiles,
        const int32_t * __restrict__ n_tiles_ptr, float * __restrict__ dst, const float * __restrict__ tmp_last_tile,
        const uint3 blocks_per_ne00, const int nrows_x, const int stride_col_dst) {
    if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = (ggml_cuda_mmq_get_nthreads(type, J, fallback) / 2) / warp_size;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[J / nwarps] = {0.0f};
    const int i = blockIdx.y*warp_size + threadIdx.x;

    const int n_tiles = *n_tiles_ptr;
    const int ntotal  = n_tiles*((nrows_x + I - 1) / I)*blocks_per_ne00.z;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc0      = int64_t(bidx0)    *ntotal / gridDim.x;
    int kbc0_stop = int64_t(bidx0 + 1)*ntotal / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while (true) {
        int kbc = int64_t(bidx)*ntotal / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps] += tmp_last_tile[bidx*(J*I) + j*I + i];
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    const int  tile     = fastdiv(kbc0, blocks_per_ne00);
    const int  it       = tile / n_tiles;
    const int2 t        = tiles[tile - it*n_tiles];
    const int  col_low  = expert_bounds[t.x] + t.y;
    const int  col_diff = expert_bounds[t.x + 1] - col_low;

    __shared__ int ids_dst_shared[J];
    for (int j = threadIdx.y*warp_size + threadIdx.x; j < J; j += nwarps*warp_size) {
        ids_dst_shared[j] = j < col_diff ? ids_dst[col_low + j] : 0;
    }
    __syncthreads();

    dst += it*I;

    const int i_max = nrows_x  - it*I - 1;
    const int j_max = col_diff - 1;
    if (fallback && i > i_max) {
        return;
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

        dst[ids_dst_shared[j]*stride_col_dst + i] += sum[j0/nwarps];
    }
}

template <ggml_type type, int J, bool fallback>
static void launch_mul_mat_q_grouped(const mmq_grouped_args & args, cudaStream_t stream) {
    const int id        = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[id].cc;
    const int nsm       = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    GGML_ASSERT(config.type != GGML_TYPE_COUNT);
    GGML_ASSERT(config.nthreads % warp_size == 0);
    const int nwarps        = config.nthreads / warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_grouped<type, J, fallback>), nbytes_shared);

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
    const int64_t nty = (args.nrows_x + config.I - 1) / config.I;
    GGML_ASSERT(args.max_tiles*nty*blocks_per_ne00_fd.z < (1 << 30)); // Assert that variable kbc will not overflow.

    // The tile count is only known on the device, so the grid always has one block per SM and the fixup kernel always runs.
    mul_mat_q_grouped<type, J, fallback><<<nsm, dim3(warp_size, nwarps, 1), nbytes_shared, stream>>>
        (args.x, args.y, args.ids_dst, args.expert_bounds, args.tiles, args.n_tiles, args.dst, args.tmp_fixup,
         blocks_per_ne00_fd, args.nrows_x, args.stride_row_x, args.ncols_y, args.stride_col_dst, args.stride_channel_x);
    CUDA_CHECK(cudaGetLastError());
    if (args.ev_fixup) {
        CUDA_CHECK(cudaEventRecord(args.ev_fixup, stream));
    }

    mul_mat_q_grouped_fixup<type, J, fallback><<<dim3(nsm, config.I/warp_size, 1), dim3(warp_size, nwarps/2, 1), 0, stream>>>
        (args.ids_dst, args.expert_bounds, args.tiles, args.n_tiles, args.dst, args.tmp_fixup,
         blocks_per_ne00_fd, args.nrows_x, args.stride_col_dst);
}

// Tile widths with a grouped kernel instance. The host may only pick one with a valid MMQ config for the type and fallback.
#define MMQ_GROUPED_J_LIST 8, 16, 24, 32, 48, 64, 96, 128

template <ggml_type type, bool fallback>
static void mul_mat_q_grouped_switch_J(const mmq_grouped_args & args, cudaStream_t stream) {
    switch (args.J) {
        case   8: launch_mul_mat_q_grouped<type,   8, fallback>(args, stream); break;
        case  16: launch_mul_mat_q_grouped<type,  16, fallback>(args, stream); break;
        case  24: launch_mul_mat_q_grouped<type,  24, fallback>(args, stream); break;
        case  32: launch_mul_mat_q_grouped<type,  32, fallback>(args, stream); break;
        case  48: launch_mul_mat_q_grouped<type,  48, fallback>(args, stream); break;
        case  64: launch_mul_mat_q_grouped<type,  64, fallback>(args, stream); break;
        case  96: launch_mul_mat_q_grouped<type,  96, fallback>(args, stream); break;
        case 128: launch_mul_mat_q_grouped<type, 128, fallback>(args, stream); break;
        default:
            GGML_ABORT("grouped MMQ: no instance for J=%d", args.J);
    }
}

template <ggml_type type>
void mul_mat_q_grouped_case(const mmq_grouped_args & args, cudaStream_t stream) {
    if (args.nrows_x % 128 == 0) {
        mul_mat_q_grouped_switch_J<type, false>(args, stream);
    } else {
        mul_mat_q_grouped_switch_J<type, true>(args, stream);
    }
}

#ifdef GGML_CUDA_MMQ_GROUPED_AVAILABLE
#define DECL_MMQ_GROUPED_CASE(type) \
    template void mul_mat_q_grouped_case<type>(const mmq_grouped_args & args, cudaStream_t stream) \

extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q1_0);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q2_0);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q4_1);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q5_0);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q5_1);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q8_0);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q2_K);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q3_K);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q4_K);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q5_K);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_Q6_K);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_IQ1_S);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_IQ2_XXS);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_IQ2_XS);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_IQ2_S);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_IQ3_XXS);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_IQ3_S);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_IQ4_NL);
extern DECL_MMQ_GROUPED_CASE(GGML_TYPE_IQ4_XS);
#else
#define DECL_MMQ_GROUPED_CASE(type)
#endif // GGML_CUDA_MMQ_GROUPED_AVAILABLE
