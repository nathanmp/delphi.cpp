#include "dsv4-hc-mix.cuh"

// GGML_OP_DSV4_HC_MIX in two kernels. The norm of the stream is a scale per token, so the product with fn reads the stream
// as it is and the scale comes after the sum.
// 1. dsv4_hc_mix_part: block (slice, token) takes the values i of the slice in the hc streams. One warp per 6 rows of fn,
//    8 values per lane: the products with the 24 rows of fn. A fifth warp sums x[i, h]*x[i, g] (the Gram matrix of the
//    streams, 10 values): the sum of squares of the stream is its trace, and the sum of squares of
//    sum_h pre[h]*x[:, h] is pre^T G pre. 34 partial sums.
// 2. dsv4_hc_mix_final: block (token, 0) and blocks (token, 1..) add the partial sums in slice order (the result does
//    not change between runs). Block 0 computes post and comb like the separate ops; the other blocks pre, then
//    sum_h pre[h]*x[:, h] and, with a weight, its RMS norm, each for 4*HC_MIX_BLOCK values.
// This replaces rms_norm, quantize_q8_1, mul_vec_q, the two weight chains, hc_comb, hc_pre and the rms_norm of the layer
// input: 7 to 9 kernels.

static constexpr int HC_MIX_HC          = 4;
static constexpr int HC_MIX_DIM         = (2 + HC_MIX_HC)*HC_MIX_HC;    // 24 rows of fn
static constexpr int HC_MIX_NGRAM       = HC_MIX_HC*(HC_MIX_HC + 1)/2; // G[h, g], h <= g
static constexpr int HC_MIX_NPART       = HC_MIX_DIM + HC_MIX_NGRAM;
static constexpr int HC_MIX_PART_NWARPS = 4;
static constexpr int HC_MIX_PART_ROWS   = HC_MIX_DIM/HC_MIX_PART_NWARPS;
static constexpr int HC_MIX_MAX_SLICES  = 64;  // GLM-5.3: 64 slices of 64 values in each stream, one pass per warp
static constexpr int HC_MIX_BLOCK       = 256; // dsv4_hc_mix_final

static_assert(HC_MIX_DIM % HC_MIX_PART_NWARPS == 0, "bad rows per warp");

// dsv4_hc_mix_part: HC_MIX_PART_NWARPS warps for the rows of fn and one warp for the Gram matrix
static constexpr int HC_MIX_PART_BLOCK = (HC_MIX_PART_NWARPS + 1)*WARP_SIZE;

// 8 consecutive values of a row of fn, i % 8 == 0
template <ggml_type type>
static __device__ __forceinline__ void hc_mix_load8(const char * row, const int64_t i, float * w) {
    if constexpr (type == GGML_TYPE_F32) {
        const float4 a = *(const float4 *) ((const float *) row + i);
        const float4 b = *(const float4 *) ((const float *) row + i + 4);
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
        w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    } else if constexpr (type == GGML_TYPE_F16) {
        const uint4 u = *(const uint4 *) ((const half *) row + i);
        const half * p = (const half *) &u;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            w[k] = __half2float(p[k]);
        }
    } else if constexpr (type == GGML_TYPE_BF16) {
        const uint4 u = *(const uint4 *) ((const nv_bfloat16 *) row + i);
        const nv_bfloat16 * p = (const nv_bfloat16 *) &u;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            w[k] = __bfloat162float(p[k]);
        }
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "bad type");
        const block_q8_0 * b = (const block_q8_0 *) row + i/QK8_0;
        const float d = __half2float(b->d);
        const char2 * q = (const char2 *) (b->qs + i%QK8_0); // 2-byte aligned: blocks of 34 bytes, qs at byte 2
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const char2 c = q[k];
            w[2*k + 0] = d*c.x;
            w[2*k + 1] = d*c.y;
        }
    }
}

template <ggml_type type>
static __global__ void __launch_bounds__(HC_MIX_PART_BLOCK) dsv4_hc_mix_part(
        const float * __restrict__ x, const char * __restrict__ fn, float * __restrict__ part,
        const int64_t n_embd, const int64_t len, const int64_t stride_x, const int64_t stride_fn) {
    const int s    = blockIdx.x;
    const int it   = blockIdx.y;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    const float * xt = x + it*stride_x;
    float * pt = part + ((int64_t) it*gridDim.x + s)*HC_MIX_NPART;

    if (warp == HC_MIX_PART_NWARPS) {
        float g[HC_MIX_NGRAM];
#pragma unroll
        for (int k = 0; k < HC_MIX_NGRAM; ++k) {
            g[k] = 0.0f;
        }
        for (int64_t i = s*len + 2*lane; i < (s + 1)*len; i += 2*WARP_SIZE) {
            float2 a[HC_MIX_HC];
#pragma unroll
            for (int h = 0; h < HC_MIX_HC; ++h) {
                a[h] = *(const float2 *) (xt + h*n_embd + i);
            }
            int k = 0;
#pragma unroll
            for (int h = 0; h < HC_MIX_HC; ++h) {
#pragma unroll
                for (int h2 = h; h2 < HC_MIX_HC; ++h2) {
                    g[k++] += a[h].x*a[h2].x + a[h].y*a[h2].y;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < HC_MIX_NGRAM; ++k) {
            g[k] = warp_reduce_sum(g[k]);
        }
        if (lane == 0) {
#pragma unroll
            for (int k = 0; k < HC_MIX_NGRAM; ++k) {
                pt[HC_MIX_DIM + k] = g[k];
            }
        }
        return;
    }

    const char * fw = fn + (int64_t) warp*HC_MIX_PART_ROWS*stride_fn;

    float acc[HC_MIX_PART_ROWS];
#pragma unroll
    for (int r = 0; r < HC_MIX_PART_ROWS; ++r) {
        acc[r] = 0.0f;
    }

    // f: position in the hc*len values of the slice; len % 8 == 0, so 8 values are in one stream
    for (int64_t f = 8*lane; f < HC_MIX_HC*len; f += 8*WARP_SIZE) {
        const int64_t h = f / len;
        const int64_t i = h*n_embd + s*len + (f - h*len);
        const float4 v0 = *(const float4 *) (xt + i);
        const float4 v1 = *(const float4 *) (xt + i + 4);
        const float v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
        for (int r = 0; r < HC_MIX_PART_ROWS; ++r) {
            float w[8];
            hc_mix_load8<type>(fw + r*stride_fn, i, w);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc[r] += w[k]*v[k];
            }
        }
    }

#pragma unroll
    for (int r = 0; r < HC_MIX_PART_ROWS; ++r) {
        acc[r] = warp_reduce_sum(acc[r]);
    }

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < HC_MIX_PART_ROWS; ++r) {
            pt[warp*HC_MIX_PART_ROWS + r] = acc[r];
        }
    }
}

// G[h, h2] in the partial sums, h <= h2
static __device__ __forceinline__ int hc_mix_gram_idx(const int h, const int h2) {
    return HC_MIX_DIM + h*HC_MIX_HC - h*(h - 1)/2 + (h2 - h);
}

// block (token, 0): post and comb; blocks (token, 1..): pre, sum_h pre[h]*x[:, h] and its norm. All add the partial sums.
static __global__ void __launch_bounds__(HC_MIX_BLOCK) dsv4_hc_mix_final(
        const float * __restrict__ x, const float * __restrict__ part, const float * __restrict__ scale,
        const float * __restrict__ base, const float * __restrict__ norm_w, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_tokens, const int nslice, const int64_t stride_x, const int64_t stride_scale,
        const float norm_eps, const float eps, const int n_iter) {
    constexpr int hc     = HC_MIX_HC;

    const int it   = blockIdx.x;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    __shared__ float sp[HC_MIX_MAX_SLICES*HC_MIX_NPART];
    __shared__ float raw[HC_MIX_NPART];

    // the partial sums of the slices: all loads at once, then one thread per sum in slice order
    const float * pt = part + (int64_t) it*nslice*HC_MIX_NPART;
    constexpr int nload = (HC_MIX_MAX_SLICES*HC_MIX_NPART + HC_MIX_BLOCK - 1)/HC_MIX_BLOCK;
#pragma unroll
    for (int k = 0; k < nload; ++k) {
        const int e = k*HC_MIX_BLOCK + threadIdx.x;
        if (e < nslice*HC_MIX_NPART) {
            sp[e] = pt[e];
        }
    }
    __syncthreads();
    if (threadIdx.x < HC_MIX_NPART) {
        float sum = 0.0f;
        for (int s = 0; s < nslice; ++s) {
            sum += sp[s*HC_MIX_NPART + threadIdx.x];
        }
        raw[threadIdx.x] = sum;
    }
    __syncthreads();

    float ss = 0.0f; // the trace of G
#pragma unroll
    for (int h = 0; h < hc; ++h) {
        ss += raw[hc_mix_gram_idx(h, h)];
    }
    const float rscale = 1.0f / sqrtf(ss/(hc*n_embd) + norm_eps);

    if (blockIdx.y == 0) {
        if (warp != 0) {
            return;
        }
        // the post weight chain of dsv4_hc_affine (deepseek4.cpp)
        if (lane < hc) {
            const float tq = __fadd_rn(__fmul_rn(raw[hc + lane]*rscale, scale[stride_scale]), base[hc + lane]);
            dst[n_embd*n_tokens + it*hc + lane] = 2.0f*(1.0f/(1.0f + expf(-tq)));
        }

        // ggml_dsv4_hc_comb in the registers of each lane, c[idst + 4*isrc]: softmax over idst, then the column and row
        // normalizations of Sinkhorn. The reciprocals are approximate (__fdividef, 2 ulp): 41 of them are in series.
        float c[hc*hc];
#pragma unroll
        for (int idx = 0; idx < hc*hc; ++idx) {
            c[idx] = (raw[2*hc + idx]*rscale)*scale[2*stride_scale] + base[2*hc + idx];
        }
#pragma unroll
        for (int isrc = 0; isrc < hc; ++isrc) {
            const float max = fmaxf(fmaxf(c[hc*isrc + 0], c[hc*isrc + 1]), fmaxf(c[hc*isrc + 2], c[hc*isrc + 3]));
#pragma unroll
            for (int idst = 0; idst < hc; ++idst) {
                c[hc*isrc + idst] = expf(c[hc*isrc + idst] - max);
            }
            const float inv_sum = __fdividef(1.0f, (c[hc*isrc + 0] + c[hc*isrc + 1]) + (c[hc*isrc + 2] + c[hc*isrc + 3]));
#pragma unroll
            for (int idst = 0; idst < hc; ++idst) {
                c[hc*isrc + idst] = c[hc*isrc + idst]*inv_sum + eps;
            }
        }
        for (int i = 0; i < n_iter; ++i) {
            if (i > 0) {
#pragma unroll
                for (int isrc = 0; isrc < hc; ++isrc) {
                    const float inv = __fdividef(1.0f, eps + ((c[hc*isrc + 0] + c[hc*isrc + 1]) + (c[hc*isrc + 2] + c[hc*isrc + 3])));
#pragma unroll
                    for (int idst = 0; idst < hc; ++idst) {
                        c[hc*isrc + idst] *= inv;
                    }
                }
            }
#pragma unroll
            for (int idst = 0; idst < hc; ++idst) {
                const float inv = __fdividef(1.0f, eps + ((c[idst] + c[idst + hc]) + (c[idst + 2*hc] + c[idst + 3*hc])));
#pragma unroll
                for (int isrc = 0; isrc < hc; ++isrc) {
                    c[idst + hc*isrc] *= inv;
                }
            }
        }
        if (lane == 0) {
            float4 * out = (float4 *) (dst + (n_embd + hc)*n_tokens + it*hc*hc);
#pragma unroll
            for (int k = 0; k < hc; ++k) {
                out[k] = make_float4(c[4*k + 0], c[4*k + 1], c[4*k + 2], c[4*k + 3]);
            }
        }
        return;
    }

    // the pre weight chain of dsv4_hc_affine (deepseek4.cpp), in each thread
    float pre[hc];
#pragma unroll
    for (int h = 0; h < hc; ++h) {
        const float tp = __fadd_rn(__fmul_rn(raw[h]*rscale, scale[0]), base[h]);
        pre[h] = 1.0f/(1.0f + expf(-tp)) + eps;
    }

    // ggml_dsv4_hc_pre, 4 values per thread
    const int64_t i = 4*((int64_t) (blockIdx.y - 1)*HC_MIX_BLOCK + threadIdx.x);
    if (i >= n_embd) {
        return;
    }
    const float * xt = x + it*stride_x;
    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int h = 0; h < hc; ++h) {
        const float4 v = *(const float4 *) (xt + h*n_embd + i);
        o.x += v.x*pre[h];
        o.y += v.y*pre[h];
        o.z += v.z*pre[h];
        o.w += v.w*pre[h];
    }

    if (norm_w) {
        // ggml_rms_norm, then ggml_mul with the weight; the sum of squares of the result is pre^T G pre
        float sum = 0.0f;
#pragma unroll
        for (int h = 0; h < hc; ++h) {
#pragma unroll
            for (int h2 = 0; h2 < hc; ++h2) {
                sum += pre[h]*pre[h2]*raw[hc_mix_gram_idx(min(h, h2), max(h, h2))];
            }
        }
        const float sc_norm = rsqrtf(sum/n_embd + norm_eps);
        const float4 w = *(const float4 *) (norm_w + i);
        o.x = (o.x*sc_norm)*w.x;
        o.y = (o.y*sc_norm)*w.y;
        o.z = (o.z*sc_norm)*w.z;
        o.w = (o.w*sc_norm)*w.w;
    }
    *(float4 *) (dst + it*n_embd + i) = o;
}

// slices of the n_embd values per token: up to 64, each a multiple of 8 values
static int hc_mix_nslice(const int64_t n_embd) {
    int nslice = HC_MIX_MAX_SLICES;
    while (nslice > 1 && n_embd % (8*nslice) != 0) {
        nslice /= 2;
    }
    return nslice;
}

bool ggml_cuda_dsv4_hc_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * x      = op->src[0];
    const ggml_tensor * fn     = op->src[1];
    const ggml_tensor * norm_w = op->src[4];
    const bool type_ok = fn->type == GGML_TYPE_F32 || fn->type == GGML_TYPE_F16 || fn->type == GGML_TYPE_BF16 ||
        fn->type == GGML_TYPE_Q8_0;
    // slices of 8 values; 16-byte loads of x, fn, the output and norm_w: 16-byte rows; Q8_0: 8 values never cross a block
    return type_ok && x->type == GGML_TYPE_F32 && x->ne[1] == HC_MIX_HC && ggml_is_contiguous(x) && x->ne[0] % 8 == 0 &&
        fn->nb[0] == ggml_type_size(fn->type) && (fn->type == GGML_TYPE_Q8_0 || fn->nb[1] % 16 == 0) &&
        op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[3]) &&
        (norm_w == nullptr || (norm_w->type == GGML_TYPE_F32 && ggml_is_contiguous(norm_w))) &&
        op->type == GGML_TYPE_F32;
}

void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x      = dst->src[0];
    const ggml_tensor * fn     = dst->src[1];
    const ggml_tensor * scale  = dst->src[2];
    const ggml_tensor * base   = dst->src[3];
    const ggml_tensor * norm_w = dst->src[4];

    GGML_ASSERT(ggml_cuda_dsv4_hc_mix_supported(dst));

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[2];
    const int     nslice   = hc_mix_nslice(n_embd);

    const float   norm_eps = ggml_get_op_params_f32(dst, 0);
    const float   eps      = ggml_get_op_params_f32(dst, 1);
    const int32_t n_iter   = ggml_get_op_params_i32(dst, 2);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> part(ctx.pool(), n_tokens*nslice*HC_MIX_NPART);

    const float * x_d = (const float *) x->data;
    const int64_t stride_x = x->nb[2] / sizeof(float);
    const dim3 grid_part(nslice, n_tokens, 1);
    const dim3 block_part(HC_MIX_PART_BLOCK, 1, 1);
    switch (fn->type) {
        case GGML_TYPE_F32:
            dsv4_hc_mix_part<GGML_TYPE_F32><<<grid_part, block_part, 0, stream>>>(
                x_d, (const char *) fn->data, part.get(), n_embd, n_embd/nslice, stride_x, fn->nb[1]);
            break;
        case GGML_TYPE_F16:
            dsv4_hc_mix_part<GGML_TYPE_F16><<<grid_part, block_part, 0, stream>>>(
                x_d, (const char *) fn->data, part.get(), n_embd, n_embd/nslice, stride_x, fn->nb[1]);
            break;
        case GGML_TYPE_BF16:
            dsv4_hc_mix_part<GGML_TYPE_BF16><<<grid_part, block_part, 0, stream>>>(
                x_d, (const char *) fn->data, part.get(), n_embd, n_embd/nslice, stride_x, fn->nb[1]);
            break;
        case GGML_TYPE_Q8_0:
            dsv4_hc_mix_part<GGML_TYPE_Q8_0><<<grid_part, block_part, 0, stream>>>(
                x_d, (const char *) fn->data, part.get(), n_embd, n_embd/nslice, stride_x, fn->nb[1]);
            break;
        default:
            GGML_ABORT("unsupported type %s", ggml_type_name(fn->type));
    }
    CUDA_CHECK(cudaGetLastError());

    const int nblocks_pre = (n_embd + 4*HC_MIX_BLOCK - 1) / (4*HC_MIX_BLOCK);
    dsv4_hc_mix_final<<<dim3(n_tokens, 1 + nblocks_pre, 1), HC_MIX_BLOCK, 0, stream>>>(
        x_d, part.get(), (const float *) scale->data, (const float *) base->data,
        norm_w ? (const float *) norm_w->data : nullptr, (float *) dst->data,
        n_embd, n_tokens, nslice, stride_x, scale->nb[0] / sizeof(float), norm_eps, eps, n_iter);
    CUDA_CHECK(cudaGetLastError());
}
