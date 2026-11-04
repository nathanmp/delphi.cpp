#include "llama-dsv41-prefill.h"

#include "llama-context.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-model.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdlib>

namespace {

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

std::unique_ptr<llama_dsv41_prefill_window> llama_dsv41_prefill_window::create(
        llama_context & ctx, llama_memory_context_i * memory) {
    const char * env = std::getenv("LLAMA_DSV41_LAYER_BANK");
    const int mode = env ? std::atoi(env) : 0;
    if ((mode != 1 && mode != 2) || ctx.model.arch != LLM_ARCH_DEEPSEEK41 ||
            !ctx.cparams.ced_compact || ctx.cparams.n_seq_max != 1 || ctx.cparams.pipeline_parallel ||
            ctx.cparams.embeddings || ctx.cparams.moe_split_n > 0 || ctx.cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
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
    return std::unique_ptr<llama_dsv41_prefill_window>(new llama_dsv41_prefill_window(ctx, *mctx, mode == 1));
}

llama_dsv41_prefill_window::llama_dsv41_prefill_window(
        llama_context & ctx, llama_kv_cache_dsv4_context & mctx, bool retain) :
    ctx(ctx), mctx(mctx), retain(retain), chunks(mctx.get_ubatches().size()) {
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
    const int64_t start = ggml_time_us();
    uint64_t spill = 0;
    for (uint32_t il = 0; il < ctx.cparams.n_layer_prefill; ++il) {
        mctx.rewind_window();
        ggml_backend_sched_stage_keep(ctx.sched.get(), retain);
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
            // Preserve the logical llama_decode timing transaction across internal graph boundaries.
            ggml_backend_sched_synchronize(ctx.sched.get());
            graph.chunk->hc.store(graph.hc);
            graph.chunk->mix.store(graph.mix);
            graph.chunk->top_k.store(graph.top_k);
            graph.chunk->cand.store(graph.cand);
            spill += ggml_nbytes(graph.hc) + ggml_nbytes(graph.mix);
        } while (mctx.next());
    }
    ggml_backend_sched_stage_keep(ctx.sched.get(), false);
    mctx.rewind_window();
    ctx.dsv41_prefill_graph = nullptr;
    LLAMA_LOG_INFO("dsv41-layer-bank: %zu chunks x %u encoder layers, %s, %.3f GiB HC/mix spills, %.1f ms\n",
            chunks.size(), ctx.cparams.n_layer_prefill, retain ? "retained banks" : "schedule only",
            spill / double(1ULL << 30), (ggml_time_us() - start) / 1000.0);
    return GGML_STATUS_SUCCESS;
}

void llama_dsv41_prefill_window::decoder(size_t chunk) {
    graph = {};
    graph.first = ctx.cparams.n_layer_prefill;
    graph.last = ctx.model.hparams.n_layer();
    graph.chunk = &chunks.at(chunk);
    ctx.dsv41_prefill_graph = &graph;
}
