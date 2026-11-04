#include "llama-dsv41-prefill.h"

#include "llama-context.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-model.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdlib>

namespace {

int window_mode() {
    const char * env = std::getenv("LLAMA_DSV41_LAYER_BANK");
    return env ? std::atoi(env) : 0;
}

// Bytes of the device state of `n_tokens` rows: the F32 HC stream and the F32 pending mix, each aligned.
size_t state_bytes(const llama_hparams & hparams, int64_t n_tokens, size_t align) {
    const int64_t hc = hparams.dsv4_hc_mult;
    return GGML_PAD(size_t(n_tokens * hparams.n_embd * hc) * sizeof(float), align) + GGML_PAD(size_t(n_tokens * hc) * sizeof(float), align);
}

class prefill_input : public llm_graph_input_i {
public:
    prefill_input(const llama_dsv41_prefill_tensor & src, ggml_tensor * dst) : src(src), dst(dst) {}

    void set_input(const llama_ubatch *) override {
        if (dst->buffer) {
            GGML_ASSERT(src.data.size() == ggml_nbytes(dst));
            ggml_backend_tensor_set(dst, src.data.data(), 0, src.data.size());
        }
    }

private:
    const llama_dsv41_prefill_tensor & src;
    ggml_tensor * dst;
};

}

ggml_tensor * llama_dsv41_prefill_tensor::input(ggml_context * ctx, llm_graph_result * res, const char * name) const {
    if (data.empty()) {
        return nullptr;
    }
    auto * tensor = ggml_new_tensor(ctx, type, 4, ne.data());
    ggml_set_name(tensor, name);
    ggml_set_input(tensor);
    res->add_input(std::make_unique<prefill_input>(*this, tensor));
    return tensor;
}

void llama_dsv41_prefill_tensor::store(ggml_tensor * tensor) {
    if (!tensor) {
        return;
    }
    GGML_ASSERT(ggml_is_contiguous(tensor));
    type = tensor->type;
    std::copy(tensor->ne, tensor->ne + 4, ne.begin());
    data.resize(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
}

bool llama_dsv41_prefill_window::context_eligible(const llama_context & ctx, int mode) {
    return (mode == 1 || mode == 2) && ctx.model.arch == LLM_ARCH_DEEPSEEK41 && ctx.cparams.ced_compact &&
            ctx.cparams.n_seq_max == 1 && !ctx.cparams.pipeline_parallel && !ctx.cparams.embeddings &&
            ctx.cparams.moe_split_n == 0 && ctx.cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT;
}

void llama_dsv41_prefill_window::reserve(llama_context & ctx) {
    const uint32_t n_chunks = std::min<uint32_t>(4, ctx.cparams.n_batch / std::max<uint32_t>(ctx.cparams.n_ubatch, 1));
    if (ctx.dsv41_prefill_state || n_chunks < 2 || !context_eligible(ctx, window_mode())) {
        return;
    }
    // The encoder layers read and write the state; on another GPU each graph copies it twice.
    ggml_backend_dev_t dev = ctx.model.dev_layer(0);
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev) : nullptr;
    if (!buft || ggml_backend_buft_is_host(buft)) {
        return;
    }
    const size_t size = n_chunks * state_bytes(ctx.model.hparams, ctx.cparams.n_ubatch, ggml_backend_buft_get_alignment(buft));
    ctx.dsv41_prefill_state.reset(ggml_backend_buft_alloc_buffer(buft, size));
    if (!ctx.dsv41_prefill_state) {
        LLAMA_LOG_WARN("dsv41-layer-bank: cannot allocate %.1f MiB of window state on %s; ordinary prefill\n", size / 1048576.0,
                ggml_backend_dev_name(dev));
        return;
    }
    LLAMA_LOG_INFO("dsv41-layer-bank: %.1f MiB of window state for %u chunks on %s\n", size / 1048576.0, n_chunks, ggml_backend_dev_name(dev));
}

std::unique_ptr<llama_dsv41_prefill_window> llama_dsv41_prefill_window::create(
        llama_context & ctx, llama_memory_context_i * memory) {
    const int mode = window_mode();
    if (!context_eligible(ctx, mode)) {
        return nullptr;
    }
    auto * mctx = dynamic_cast<llama_kv_cache_dsv4_context *>(memory);
    if (!mctx || mctx->get_ubatches().size() < 2 || mctx->get_ubatches().size() > 4) {
        return nullptr;
    }
    int min_tokens = mode == 1 ? ggml_backend_sched_stage_get_min_tokens(ctx.sched.get()) : 1;
    if (min_tokens <= 0) {
        return nullptr;
    }
    if (mode == 1 && ctx.cparams.cb_eval) {
        return nullptr;
    }
    if (mode == 1) {
        min_tokens = std::max<int>(min_tokens, ctx.graph_alt_max_tokens + 1);
    }
    const auto & ubatches = mctx->get_ubatches();
    const llama_seq_id seq = ubatches.front().seq_id_unq[0];
    llama_pos pos = ubatches.front().pos[0];
    for (size_t i = 0; i < ubatches.size(); ++i) {
        const auto & batch = ubatches[i];
        if (!batch.token || batch.embd || batch.embd_tensor || batch.embd_h || batch.embd_h_tensor ||
                batch.n_pos != 1 || batch.n_seqs_unq != 1 || batch.seq_id_unq[0] != seq ||
                batch.n_tokens < (uint32_t) min_tokens) {
            return nullptr;
        }
        for (uint32_t row = 0; row < batch.n_tokens; ++row) {
            if (batch.pos[row] != pos++ || (batch.output && batch.output[row] &&
                    (i + 1 != ubatches.size() || row + 1 != batch.n_tokens))) {
                return nullptr;
            }
        }
    }
    for (uint32_t il = 0; il < ctx.cparams.n_layer_prefill; ++il) {
        if (ctx.cparams.embeddings_layer_inp[il]) {
            return nullptr;
        }
    }
    reserve(ctx);
    if (!ctx.dsv41_prefill_state) {
        return nullptr;
    }
    size_t need = 0;
    for (const auto & batch : ubatches) {
        need += state_bytes(ctx.model.hparams, batch.n_tokens, ggml_backend_buffer_get_alignment(ctx.dsv41_prefill_state.get()));
    }
    if (need > ggml_backend_buffer_get_size(ctx.dsv41_prefill_state.get())) {
        return nullptr;
    }
    return std::unique_ptr<llama_dsv41_prefill_window>(new llama_dsv41_prefill_window(ctx, *mctx, mode == 1));
}

llama_dsv41_prefill_window::llama_dsv41_prefill_window(
        llama_context & ctx, llama_kv_cache_dsv4_context & mctx, bool retain) :
    ctx(ctx), mctx(mctx), retain(retain), chunks(mctx.get_ubatches().size()) {
    const auto & hparams = ctx.model.hparams;
    ggml_init_params params = { 2 * chunks.size() * ggml_tensor_overhead(), nullptr, true };
    state_ctx.reset(ggml_init(params));
    ggml_backend_buffer_t buf = ctx.dsv41_prefill_state.get();
    const size_t align = ggml_backend_buffer_get_alignment(buf);
    char * addr = (char *) ggml_backend_buffer_get_base(buf);
    const auto alloc = [&](ggml_tensor * t, const char * name, size_t i) {
        ggml_format_name(t, "%s-%zu", name, i);
        GGML_ASSERT(ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS);
        addr += GGML_PAD(ggml_nbytes(t), align);
        return t;
    };
    for (size_t i = 0; i < chunks.size(); ++i) {
        const int64_t n_tokens = mctx.get_ubatches()[i].n_tokens;
        chunks[i].hc  = alloc(ggml_new_tensor_3d(state_ctx.get(), GGML_TYPE_F32, hparams.n_embd, hparams.dsv4_hc_mult, n_tokens), "prefill_hc", i);
        chunks[i].mix = alloc(ggml_new_tensor_2d(state_ctx.get(), GGML_TYPE_F32, hparams.dsv4_hc_mult, n_tokens), "prefill_mix", i);
    }
}

llama_dsv41_prefill_window::~llama_dsv41_prefill_window() {
    ggml_backend_sched_synchronize(ctx.sched.get());
    ggml_backend_sched_stage_keep(ctx.sched.get(), false);
    ctx.dsv41_prefill_graph = nullptr;
}

ggml_status llama_dsv41_prefill_window::encode() {
    ggml_backend_sched_synchronize(ctx.sched.get());
    ctx.graph_slot_activate(0);
    ggml_backend_sched_synchronize(ctx.sched.get());
    if (!ctx.state_async_pending.empty()) {
        ctx.state_async_finish();
    }
    mctx.begin_window();
    // without GGML_SCHED_STAGE_KEEP_SPREAD a staged expert multiply runs where the layer's activations are
    const auto backend_of = [&](uint32_t il) -> ggml_backend_t {
        ggml_backend_dev_t dev = ctx.model.dev_layer(il);
        for (const auto & backend : ctx.backends) {
            if (ggml_backend_get_device(backend.get()) == dev) {
                return backend.get();
            }
        }
        return nullptr;
    };
    const int64_t start = ggml_time_us();
    size_t n_index = 0;
    ggml_backend_sched_stage_keep(ctx.sched.get(), retain);
    for (uint32_t il = 0; il < ctx.cparams.n_layer_prefill; ++il) {
        mctx.rewind_window();
        size_t chunk = 0;
        do {
            if (ctx.abort_callback && ctx.abort_callback(ctx.abort_callback_data)) {
                return GGML_STATUS_ABORTED;
            }
            graph = {};
            graph.first = il;
            graph.last = il + 1;
            graph.chunk = &chunks[chunk++];
            ctx.dsv41_prefill_graph = &graph;
            ctx.n_outputs = 0;
            ggml_status status;
            if (!ctx.process_ubatch(mctx.get_ubatch(), LLM_GRAPH_TYPE_DEFAULT, &mctx, status)) {
                return status;
            }
            if (retain && chunk == 1 && il + 1 < ctx.cparams.n_layer_prefill) {
                // the next layer's bank uploads while this layer reads its own bank for the other chunks
                const auto & next = ctx.model.layers[il + 1];
                ggml_tensor * weights[3] = { next.ffn_gate_exps, next.ffn_up_exps, next.ffn_down_exps };
                ggml_backend_sched_stage_prefetch(ctx.sched.get(), backend_of(il + 1), weights, 3);
            }
            if (graph.top_k || graph.cand) {
                // the next graph reuses the compute buffer
                ggml_backend_sched_synchronize(ctx.sched.get());
                graph.chunk->top_k.store(graph.top_k);
                graph.chunk->cand.store(graph.cand);
                n_index++;
            }
        } while (mctx.next());
    }
    ggml_backend_sched_stage_keep(ctx.sched.get(), false);
    mctx.rewind_window();
    ctx.dsv41_prefill_graph = nullptr;
    static const bool log = std::getenv("LLAMA_DSV41_LAYER_BANK_LOG") != nullptr;
    if (log) {
        ggml_backend_sched_synchronize(ctx.sched.get());
        LLAMA_LOG_WARN("dsv41-layer-bank: %zu chunks x %u encoder layers, %s, %zu index checkpoints, %.1f ms\n",
                chunks.size(), ctx.cparams.n_layer_prefill, retain ? "retained banks" : "schedule only", n_index,
                (ggml_time_us() - start) / 1000.0);
    }
    return GGML_STATUS_SUCCESS;
}

void llama_dsv41_prefill_window::decoder(size_t chunk) {
    graph = {};
    graph.first = ctx.cparams.n_layer_prefill;
    graph.last = ctx.model.hparams.n_layer();
    graph.chunk = &chunks.at(chunk);
    ctx.dsv41_prefill_graph = &graph;
}
