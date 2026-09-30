#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;

// delphi: the chain that produces the pre/post weights, evaluated by the kernel that reads them:
//   DSV4_HC_WCHAIN_SCALE:   SCALE -> SIGMOID -> SCALE        t = s0*w + b0
//   DSV4_HC_WCHAIN_MUL_ADD: MUL -> ADD -> SIGMOID -> SCALE   t = w*mul[0] + add[ih]  (dsv4_hc_affine of deepseek4.cpp)
// then w = s1*sigmoid(t) + b1. Every step rounds like its separate kernel, so the result does not change.
enum dsv4_hc_wchain_mode {
    DSV4_HC_WCHAIN_NONE    = 0,
    DSV4_HC_WCHAIN_SCALE   = 1,
    DSV4_HC_WCHAIN_MUL_ADD = 2,
};

struct dsv4_hc_wchain {
    const float * mul;
    const float * add;
    int64_t       add_s;
    float         s0;
    float         b0;
    float         s1;
    float         b1;
};

template <int mode>
static __device__ __forceinline__ float dsv4_hc_wchain_apply(const dsv4_hc_wchain & c, const float w, const int64_t ih) {
    if constexpr (mode == DSV4_HC_WCHAIN_NONE) {
        return w;
    } else {
        float t;
        if constexpr (mode == DSV4_HC_WCHAIN_SCALE) {
            t = c.s0 * w + c.b0;
        } else {
            t = __fadd_rn(__fmul_rn(w, c.mul[0]), c.add[ih*c.add_s]);
        }
        return c.s1 * (1.0f / (1.0f + expf(-t))) + c.b1;
    }
}

// the chain on its own, for weights that the next node does not read
static __global__ void dsv4_hc_wchain_f32(const float * x, float * dst, const dsv4_hc_wchain c,
        const int64_t ne0, const int64_t n, const int64_t sx0, const int64_t sx1) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ggml_cuda_pdl_sync();
    const int64_t i0 = i % ne0;
    const int64_t i1 = i / ne0;
    dst[i] = dsv4_hc_wchain_apply<DSV4_HC_WCHAIN_MUL_ADD>(c, x[i0*sx0 + i1*sx1], i0);
}


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated, int wmode = DSV4_HC_WCHAIN_NONE>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale,
        const dsv4_hc_wchain wc) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = dsv4_hc_wchain_apply<wmode>(wc, weights[ih*sw0 + it*sw1], ih);
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

// wmode: post holds the raw input of the weight chain, which the kernel applies per weight
template <bool has_comb, int wmode = DSV4_HC_WCHAIN_NONE>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        const dsv4_hc_wchain wc) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    const float w = dsv4_hc_wchain_apply<wmode>(wc, post[idst*sp0 + it*sp1], idst);
    float sum = x[i0*sx0 + it*sx1] * w;
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

static void dsv4_hc_pre_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * weights,
        const int wmode, const dsv4_hc_wchain & wc) {
    const ggml_tensor * x = dst->src[0];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;
    GGML_ASSERT(!gated || wmode == DSV4_HC_WCHAIN_NONE);

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> :
        wmode == DSV4_HC_WCHAIN_SCALE   ? dsv4_hc_pre_f32<false, DSV4_HC_WCHAIN_SCALE> :
        wmode == DSV4_HC_WCHAIN_MUL_ADD ? dsv4_hc_pre_f32<false, DSV4_HC_WCHAIN_MUL_ADD> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale, wc);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    dsv4_hc_pre_launch(ctx, dst, dst->src[1], DSV4_HC_WCHAIN_NONE, dsv4_hc_wchain{});
}

static void dsv4_hc_post_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * post,
        const int wmode, const dsv4_hc_wchain & wc) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel =
        wmode == DSV4_HC_WCHAIN_SCALE   ? (comb ? dsv4_hc_post_f32<true, DSV4_HC_WCHAIN_SCALE>   : dsv4_hc_post_f32<false, DSV4_HC_WCHAIN_SCALE>) :
        wmode == DSV4_HC_WCHAIN_MUL_ADD ? (comb ? dsv4_hc_post_f32<true, DSV4_HC_WCHAIN_MUL_ADD> : dsv4_hc_post_f32<false, DSV4_HC_WCHAIN_MUL_ADD>) :
                                          (comb ? dsv4_hc_post_f32<true>                         : dsv4_hc_post_f32<false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            wc);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    dsv4_hc_post_launch(ctx, dst, dst->src[2], DSV4_HC_WCHAIN_NONE, dsv4_hc_wchain{});
}

// first is the SCALE or MUL that starts the chain, add the ADD of a MUL chain (nullptr for SCALE), last the final SCALE
static dsv4_hc_wchain dsv4_hc_wchain_make(const ggml_tensor * first, const ggml_tensor * add, const ggml_tensor * last, int & mode) {
    dsv4_hc_wchain c = {};
    if (first->op == GGML_OP_SCALE) {
        mode = DSV4_HC_WCHAIN_SCALE;
        c.s0 = ggml_get_op_params_f32(first, 0);
        c.b0 = ggml_get_op_params_f32(first, 1);
    } else {
        GGML_ASSERT(first->op == GGML_OP_MUL && add && add->op == GGML_OP_ADD);
        mode    = DSV4_HC_WCHAIN_MUL_ADD;
        c.mul   = (const float *) first->src[1]->data;
        c.add   = (const float *) add->src[1]->data;
        c.add_s = add->src[1]->nb[0] / sizeof(float);
    }
    c.s1 = ggml_get_op_params_f32(last, 0);
    c.b1 = ggml_get_op_params_f32(last, 1);
    return c;
}

void ggml_cuda_op_dsv4_hc_wchain(ggml_backend_cuda_context & ctx, const ggml_tensor * mul, const ggml_tensor * add,
        ggml_tensor * last) {
    int mode = DSV4_HC_WCHAIN_NONE;
    const dsv4_hc_wchain c = dsv4_hc_wchain_make(mul, add, last, mode);
    GGML_ASSERT(mode == DSV4_HC_WCHAIN_MUL_ADD);

    const ggml_tensor * x = mul->src[0];
    const int64_t n = ggml_nelements(last);
    const int block_size = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3((n + block_size - 1) / block_size, 1, 1), dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(dsv4_hc_wchain_f32, launch_params, (const float *) x->data, (float *) last->data, c,
            x->ne[0], n, (int64_t) (x->nb[0] / sizeof(float)), (int64_t) (x->nb[1] / sizeof(float)));
}

void ggml_cuda_op_dsv4_hc_pre_fused_w(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * add,
        const ggml_tensor * last, ggml_tensor * dst) {
    int mode = DSV4_HC_WCHAIN_NONE;
    const dsv4_hc_wchain c = dsv4_hc_wchain_make(first, add, last, mode);
    dsv4_hc_pre_launch(ctx, dst, first->src[0], mode, c);
}

void ggml_cuda_op_dsv4_hc_post_fused_w(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * add,
        const ggml_tensor * last, ggml_tensor * dst) {
    int mode = DSV4_HC_WCHAIN_NONE;
    const dsv4_hc_wchain c = dsv4_hc_wchain_make(first, add, last, mode);
    dsv4_hc_post_launch(ctx, dst, first->src[0], mode, c);
}
