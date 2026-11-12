// delphi: verify streams (LLAMA_VERIFY_STREAMS=1).
//
// With one context, all schedulers share one backend (one stream) per device. A verify graph that is queued behind prompt
// ubatches then waits until they have passed every device of a layer split, and llama_synchronize() waits for them too.
// Here the alternate graph slots (small ubatches, i.e. verify rows) get a second set of GPU backends whose streams have the
// highest priority, so the devices run the verify kernels next to and ahead of the prompt kernels. The server marks the
// prompt part of a batch as background work (llama_set_prompt_background): synchronize_outputs() does not wait for it,
// and each background decode gets a ticket that the server polls without a wait (llama_prompt_background_done).
//
// Invariants:
// - The two sets write disjoint KV cells only if every sequence has its own KV stream (no unified KV cache).
// - A sequence that moves from one set to the other waits on the host for the set it leaves (verify_streams_ubatch).
// - Background ubatches have no outputs. Their device feature rows (DFlash) go to a region of the feature buffer that
//   the next bg_regions - 1 background decodes do not overwrite; the server reads them there after the ticket is done.

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-kv-cache-kpool.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

static constexpr int32_t bg_regions = 4; // regions of the feature buffer for background decodes (region 0: foreground)

static bool verify_streams_env() {
    const char * env = getenv("LLAMA_VERIFY_STREAMS");
    return env != nullptr && atoi(env) != 0;
}

using set_stream_priority_t = bool (*)(ggml_backend_t, bool);

static set_stream_priority_t stream_priority_fn(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg ? (set_stream_priority_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_stream_priority") : nullptr;
}

static bool is_gpu(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    return dev && (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU);
}

void llama_context::verify_streams_init() {
    if (!verify_streams_env() || backends.empty()) {
        return;
    }

    const char * input_async_env = getenv("LLAMA_VERIFY_STREAMS_INPUT_ASYNC");
    const bool input_async = input_async_env == nullptr || atoi(input_async_env) != 0;

    // a draft model only runs verify-side work: give its streams the high priority, so its drafts also pass prompt work
    if (model.arch == LLM_ARCH_DFLASH) {
        verify_input_async = input_async;
        int n = 0;
        for (auto & b : backends) {
            if (is_gpu(b.get())) {
                auto fn = stream_priority_fn(b.get());
                n += fn && fn(b.get(), true) ? 1 : 0;
            }
        }
        LLAMA_LOG_WARN("%s: verify streams: draft context, %d high-priority streams\n", __func__, n);
        return;
    }

    if (!staggered_verify || expert_cache) {
        LLAMA_LOG_WARN("%s: verify streams: off (needs staggered verify and no expert cache)\n", __func__);
        return;
    }

    backend_ptrs_v = backend_ptrs;
    int n = 0;
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        if (!is_gpu(backend_ptrs[i])) {
            continue;
        }
        ggml_backend_t b = ggml_backend_dev_init(ggml_backend_get_device(backend_ptrs[i]), nullptr);
        if (b == nullptr) {
            throw std::runtime_error("verify streams: failed to initialize a second backend");
        }
        auto fn = stream_priority_fn(b);
        if (!fn || !fn(b, true)) {
            LLAMA_LOG_WARN("%s: verify streams: %s has no stream priorities\n", __func__, ggml_backend_name(b));
        }
        if (bg_event_query == nullptr) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(b));
            bg_event_query = (bool (*)(ggml_backend_event_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_event_query");
        }
        backends_v.emplace_back(b);
        backend_ptrs_v[i] = b;
        n++;
    }
    if (n == 0) {
        backend_ptrs_v.clear();
        return;
    }
    verify_streams = true;
    verify_input_async = input_async;
    seq_set.assign(LLAMA_MAX_SEQ, -1);
    if (const char * env = getenv("LLAMA_VERIFY_STREAMS_KV_STEP")) {
        bg_n_kv_step = (uint32_t) GGML_PAD(std::max(256, atoi(env)), 256);
    }
    LLAMA_LOG_WARN("%s: verify streams: on, %d GPU backends for the alternate graph slots, background regions of %d rows\n",
            __func__, n, bg_region_rows());
}

// set at the first background decode, so that the buffers sized at load (and by the fit) do not get the headroom
static void verify_streams_grow_env() {
    // a buffer that grows at run time frees the old one, and cudaFree waits for all work of the device, i.e. for the
    // background prompt: give such buffers headroom (ggml-alloc bounds it, GGML_GALLOC_GROW_MAX_MIB)
    if (getenv("GGML_GALLOC_GROW_PCT") == nullptr) {
        setenv("GGML_GALLOC_GROW_PCT", "25", 0);
    }
    // a prompt ubatch has a larger KV view than the last one, and a new layout waits for all background ubatches in
    // flight: tensors that grew get room for the next ubatches (bounded by GGML_GALLOC_GROW_NODE_MAX_MIB)
    if (getenv("GGML_GALLOC_GROW_NODE_PCT") == nullptr) {
        setenv("GGML_GALLOC_GROW_NODE_PCT", "400", 0);
    }
}

void llama_context::verify_streams_free() {
    if (!verify_streams) {
        return;
    }
    // the alternate schedulers use the second set: free them first
    graph_alt_reset();
    for (size_t id = 1; id < graph_slots.size(); ++id) {
        graph_slots[id].sched.reset();
    }
    for (auto & t : bg_tickets) {
        bg_recycle(t);
    }
    bg_tickets.clear();
    for (auto & pool : bg_event_pool) {
        for (auto * ev : pool) {
            ggml_backend_event_free(ev);
        }
    }
    bg_event_pool.clear();
}

ggml_backend_t * llama_context::sched_backend_ptrs(int slot_id) {
    return verify_streams && slot_id != 0 ? backend_ptrs_v.data() : backend_ptrs.data();
}

ggml_backend_t llama_context::sched_backend_of(ggml_backend_sched_t s, ggml_backend_t b) const {
    if (!verify_streams || s == nullptr || b == nullptr) {
        return b;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    for (int i = 0; i < ggml_backend_sched_get_n_backends(s); ++i) {
        ggml_backend_t sb = ggml_backend_sched_get_backend(s, i);
        if (sb == b || ggml_backend_get_device(sb) == dev) {
            return sb;
        }
    }
    return b;
}

int32_t llama_context::bg_region_rows() const {
    return (int32_t) cparams.n_batch / (bg_regions + 1);
}

void llama_context::verify_streams_decode_begin(uint32_t n_tokens_all) {
    decode_bg      = verify_streams && prompt_bg && (int32_t) n_tokens_all <= bg_region_rows();
    static bool grow_env = false;
    if (decode_bg && !grow_env) {
        verify_streams_grow_env();
        grow_env = true;
    }
    decode_bg_main = false;
    layer_inp_row_base = 0;
}

void llama_context::verify_streams_ubatch(const llama_ubatch & ubatch, int32_t n_outputs_ubatch) {
    if (!verify_streams) {
        return;
    }
    const int8_t set = graph_slot_live == 0 ? 0 : 1;
    bool other = false;
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq = ubatch.seq_id_unq[s];
        if (seq < 0 || seq >= (llama_seq_id) seq_set.size()) {
            continue;
        }
        other = other || (seq_set[seq] >= 0 && seq_set[seq] != set);
        seq_set[seq] = set;
    }
    if (other) {
        // the KV cells of the sequence were written on the other streams
        if (set == 0) {
            verify_streams_sync_alt();
        } else {
            verify_streams_sync_main();
        }
    }
    if (set == 0) {
        if (decode_bg) {
            layer_inp_row_base = (int64_t) (1 + bg_next % bg_regions) * bg_region_rows();
        }
        if (decode_bg && n_outputs_ubatch == 0) {
            decode_bg_main = true;
        } else {
            main_fg_queued = true;
        }
    } else {
        layer_inp_row_base = 0;
    }
}

void llama_context::verify_streams_decode_end(bool ok) {
    bg_last = -1;
    if (ok && decode_bg_main) {
        bg_ticket t;
        t.id       = bg_next++;
        t.row_base = (int32_t) layer_inp_row_base;
        // events[i] and bg_event_pool[i] belong to the device of backends[i]
        bg_event_pool.resize(backends.size());
        t.events.assign(backends.size(), nullptr);
        for (size_t i = 0; i < backends.size(); ++i) {
            if (!is_gpu(backends[i].get())) {
                continue;
            }
            auto & pool = bg_event_pool[i];
            ggml_backend_event_t ev = nullptr;
            if (!pool.empty()) {
                ev = pool.back();
                pool.pop_back();
            } else {
                ev = ggml_backend_event_new(ggml_backend_get_device(backends[i].get()));
            }
            ggml_backend_event_record(ev, backends[i].get());
            t.events[i] = ev;
        }
        bg_last = t.id;
        bg_last_row_base = t.row_base;
        bg_tickets.push_back(std::move(t));
    }
    decode_bg      = false;
    decode_bg_main = false;
    layer_inp_row_base = 0;
}

void llama_context::bg_recycle(bg_ticket & t) {
    for (size_t i = 0; i < t.events.size(); ++i) {
        if (t.events[i] != nullptr) {
            bg_event_pool[i].push_back(t.events[i]);
        }
    }
    t.events.clear();
}

void llama_context::verify_streams_sync_main() {
    ggml_backend_sched_t s = graph_sched_main();
    if (s) {
        ggml_backend_sched_synchronize(s);
    }
    for (auto & t : bg_tickets) {
        bg_recycle(t);
    }
    bg_tickets.clear();
    main_fg_queued = false;
}

void llama_context::verify_streams_sync_alt() {
    if (graph_slot_live != 0 && sched) {
        ggml_backend_sched_synchronize(sched.get());
    }
    for (size_t id = 1; id < graph_slots.size(); ++id) {
        if ((int) id != graph_slot_live && graph_slots[id].sched) {
            ggml_backend_sched_synchronize(graph_slots[id].sched.get());
        }
    }
}

void llama_context::synchronize_outputs() {
    if (!verify_streams) {
        synchronize();
        return;
    }
    sync_outputs_only = true;
    synchronize();
    sync_outputs_only = false;
}

int64_t llama_context::prompt_background_ticket(int32_t * row_base) const {
    if (row_base) {
        *row_base = bg_last >= 0 ? bg_last_row_base : 0;
    }
    return bg_last;
}

int32_t llama_context::prompt_background_pending() {
    if (bg_event_query == nullptr && !bg_tickets.empty()) {
        verify_streams_sync_main();
    }
    while (!bg_tickets.empty()) {
        auto & t = bg_tickets.front();
        bool done = true;
        for (auto * ev : t.events) {
            done = done && (ev == nullptr || bg_event_query(ev));
        }
        if (!done) {
            break;
        }
        bg_recycle(t);
        bg_tickets.pop_front();
    }
    return (int32_t) bg_tickets.size();
}

bool llama_context::prompt_background_done(int64_t ticket) {
    prompt_background_pending();
    return bg_tickets.empty() || bg_tickets.front().id > ticket;
}

// The same decisions as reserve_max_prefill() and the graph allocation, without their effects.
bool llama_context::prompt_background_new_layout(llama_seq_id seq_id, llama_pos pos_last, uint32_t n_tokens) {
    if (!verify_streams || n_tokens == 0 || pos_last < 0) {
        return false;
    }
    const auto * mem = dynamic_cast<const llama_memory_hybrid *>(memory.get());
    auto * cache = mem ? mem->get_kpool_cache() : nullptr;
    if (cache == nullptr || dynamic_cast<const llama_memory_hybrid_idx *>(memory.get()) != nullptr) {
        return false; // no reservation variants: the regular reservation covers the chunk
    }
    // a cold pool cache gives the first ubatch of the sequence its own graph shape
    if (seq_id >= 0 && (uint32_t) seq_id < cache->get_n_stream() && cache->needs_rebuild((uint32_t) seq_id, 1)) {
        return true;
    }
    const uint32_t n_kv    = (uint32_t) GGML_PAD((int64_t) pos_last + 1, 256);
    const int      variant = llama_kpool_indexed_attn_enabled(n_kv, n_tokens) ? 1 : 0;
    ggml_backend_sched_t sched_main = graph_sched_main();
    return !(sched_main == max_prefill_sched && ggml_backend_sched_get_n_reserves(sched_main) == max_prefill_reserves &&
            variant == max_prefill_variant && n_kv <= max_prefill_n_kv);
}

void llama_context::prompt_background_wait() {
    if (verify_streams) {
        verify_streams_sync_main();
    }
}

//
// API
//

bool llama_verify_streams(const llama_context * ctx) {
    return ctx->verify_streams_enabled();
}

void llama_set_prompt_background(llama_context * ctx, bool background) {
    ctx->set_prompt_background(background);
}

void llama_set_prompt_background_end(llama_context * ctx, llama_pos pos_end) {
    ctx->set_prompt_background_end(pos_end);
}

bool llama_prompt_background_new_layout(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_last, uint32_t n_tokens) {
    return ctx->prompt_background_new_layout(seq_id, pos_last, n_tokens);
}

int64_t llama_prompt_background_ticket(llama_context * ctx, int32_t * row_base) {
    return ctx->prompt_background_ticket(row_base);
}

int32_t llama_prompt_background_pending(llama_context * ctx) {
    return ctx->prompt_background_pending();
}

bool llama_prompt_background_done(llama_context * ctx, int64_t ticket) {
    return ctx->prompt_background_done(ticket);
}

void llama_prompt_background_wait(llama_context * ctx) {
    ctx->prompt_background_wait();
}
