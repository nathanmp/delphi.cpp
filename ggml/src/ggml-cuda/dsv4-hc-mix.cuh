#include "common.cuh"

// delphi: GGML_OP_DSV4_HC_MIX (ggml.h), the coefficients of a hyper-connection boundary and the reduced stream
bool ggml_cuda_dsv4_hc_mix_supported(const ggml_tensor * op);

void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
