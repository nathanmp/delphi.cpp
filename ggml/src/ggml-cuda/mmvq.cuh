#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// ne01: rows of the weights (0 if unknown), some architectures switch to MMQ sooner for large matrices
bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11, int64_t ne01 = 0);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// delphi: n (2 to GGML_CUDA_MMVQ_MAX_SIBLINGS) MUL_MATs whose weights sit back to back in one buffer and read the
// same src1, as one launch
#define GGML_CUDA_MMVQ_MAX_SIBLINGS 4
bool ggml_cuda_mul_mat_vec_q_siblings_supported(const ggml_tensor * const * mm, int n);
void ggml_cuda_mul_mat_vec_q_siblings(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mm, int n);

bool ggml_cuda_moe_down_q_reduction_supported(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    const ggml_tensor * weights, const ggml_tensor * dst);

void ggml_cuda_moe_down_q_reduction(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    const ggml_tensor * ids, const ggml_tensor * weights, ggml_tensor * dst);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
