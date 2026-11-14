#include "llama-kv-cache-kpool.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_seqs > 0);

    return n_kv/kpool + 2*n_seqs;
}

uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_pools > 0);
    GGML_ASSERT(indexer_top_k % kpool == 0 && "indexer_top_k must be a whole number of pools");

    return std::min(n_pools, indexer_top_k/kpool);
}

bool llama_kpool_indexed_attn_enabled(int64_t n_kv, int64_t n_tps) {
    if (n_tps <= 1) {
        return false;
    }

    const char * env = std::getenv("LLAMA_GLM5_INDEXED_ATTN");
    const int mode = env ? std::atoi(env) : 1;

    // A large prefill batch alone is not enough to amortize the indirect
    // reads and extra graph nodes of indexed attention while the KV cache is
    // still short.  Keep the contiguous dense FA path until the cache reaches
    // the long-context crossover; mode 2 remains the explicit force-on A/B
    // override.
    return mode >= 2 || (mode == 1 && n_kv >= 32768);
}

bool llama_kpool_device_masks_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_GLM5_DEVICE_MASKS");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

// larger than any position, and exact in f32 comparisons against positions
static constexpr float k_kpool_mask_big = 1e30f;

struct llama_kpool_cache::impl {
    struct pool_id_hash {
        size_t operator()(const pool_id & id) const {
            const uint64_t a = (uint32_t) id.seq;
            const uint64_t b = (uint64_t) id.block;
            return (size_t) (a*0x9e3779b97f4a7c15ULL ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
        }
    };

    struct entry {
        int32_t slot;
        bool cached;
        uint64_t gen = 0; // the prepare_stream call that last named this pool
    };

    struct layer {
        uint32_t il;
        ggml_tensor * keys;
        ggml_tensor * topk_hint;
        ggml_tensor * mtp_topk;
        ggml_tensor * mtp_mask;
        int32_t table = -1;
    };

    uint32_t max_slots;
    uint32_t n_stream;
    uint32_t n_embd;
    uint32_t kpool;
    std::vector<bool> rebuild;
    bool mtp_index_reuse = false;

    std::vector<std::unordered_map<pool_id, entry, pool_id_hash>> maps;
    uint64_t gen = 0;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
    std::vector<layer> layers;
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    // ring mode (one sequence per stream), see llama_kpool_cache::is_ring
    bool ring = false;
    std::vector<int64_t>              ring_org;
    std::vector<std::vector<int64_t>> ring_block;
    std::vector<std::vector<uint8_t>> ring_cache;
    std::vector<uint32_t>             ring_hi;

    std::vector<dev_table> tables;

    void ring_clear(uint32_t s) {
        std::fill(ring_block[s].begin(), ring_block[s].begin() + ring_hi[s], -1);
        std::fill(ring_cache[s].begin(), ring_cache[s].begin() + ring_hi[s], 0);
        ring_hi[s] = 0;
    }

    // the state of a fresh buffer: no member (cell 0), every slot invalid (BIG), row 1 = ordinal
    void init_tables() {
        for (auto & t : tables) {
            t.shadow_cells.assign((size_t) kpool*max_slots*n_stream, 0);
            t.shadow_valid.assign((size_t) max_slots*n_stream, 0);
            if (t.cells->buffer == nullptr || t.cells->data == nullptr || ggml_backend_buffer_get_size(t.cells->buffer) == 0) {
                continue; // no_alloc (fit)
            }
            ggml_backend_tensor_memset(t.cells, 0, 0, ggml_nbytes(t.cells));
            std::vector<float> iv((size_t) max_slots*n_stream*2);
            for (uint32_t st = 0; st < n_stream; ++st) {
                for (uint32_t p = 0; p < max_slots; ++p) {
                    iv[(size_t) st*max_slots + p] = 1e30f;
                    iv[(size_t) (n_stream + st)*max_slots + p] = (float) p;
                }
            }
            ggml_backend_tensor_set(t.iv, iv.data(), 0, ggml_nbytes(t.iv));
        }
    }
};

llama_kpool_cache::llama_kpool_cache(
        const llama_model & model,
        bool                offload,
        bool                unified,
        uint32_t            kv_size,
        uint32_t            n_seq_max,
        uint32_t            kpool,
        uint32_t            n_embd,
        const layer_filter_cb & filter) : pimpl(new impl) {
    GGML_ASSERT(kpool > 1);
    GGML_ASSERT(n_seq_max > 0);
    GGML_ASSERT(n_embd > 0);

    pimpl->n_stream = unified ? 1 : n_seq_max;
    const uint32_t n_ps_max = unified ? n_seq_max : 1;
    pimpl->max_slots = llama_kpool_n_pools(kv_size, kpool, n_ps_max);
    pimpl->n_embd = n_embd;
    pimpl->kpool = kpool;
    pimpl->maps.resize(pimpl->n_stream);
    pimpl->rebuild.resize(pimpl->n_stream, true);

    // n_stream == 1 also when n_seq_max == 1: a single sequence then owns the one stream, which is the ring case
    pimpl->ring = !unified || n_seq_max == 1;
    if (pimpl->ring) {
        pimpl->ring_block.assign(pimpl->n_stream, std::vector<int64_t>(pimpl->max_slots, -1));
        pimpl->ring_cache.assign(pimpl->n_stream, std::vector<uint8_t>(pimpl->max_slots, 0));
        pimpl->ring_hi.assign(pimpl->n_stream, 0);
        pimpl->ring_org.assign(pimpl->n_stream, 0);
    }
    std::map<ggml_backend_buffer_type_t, int32_t> table_of_buft;

    struct buft_comparator {
        bool operator()(ggml_backend_buffer_type_t lhs, ggml_backend_buffer_type_t rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it != ctx_map.end()) {
            return it->second.get();
        }

        ggml_init_params params = {
            /*.mem_size   =*/ size_t(4u*model.hparams.n_layer_all*ggml_tensor_overhead()),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            return nullptr;
        }
        ctx_map.emplace(buft, ctx);
        return ctx;
    };

    for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for GLM pool-key cache");
        }

        int32_t table = -1;
        if (pimpl->ring) {
            auto it = table_of_buft.find(buft);
            if (it == table_of_buft.end()) {
                dev_table t;
                t.cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, (int64_t) kpool*pimpl->max_slots, pimpl->n_stream);
                t.iv    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, pimpl->max_slots, pimpl->n_stream, 2);
                ggml_format_name(t.cells, "cache_kpool_tbl_cells_%zu", pimpl->tables.size());
                ggml_format_name(t.iv,    "cache_kpool_tbl_iv_%zu",    pimpl->tables.size());
                it = table_of_buft.emplace(buft, (int32_t) pimpl->tables.size()).first;
                pimpl->tables.push_back(std::move(t));
            }
            table = it->second;
        }

        // delphi: LLAMA_GLM5_POOL_KEYS_F16=1 stores the pool keys as F16 (opt-in precision trade): half the memory,
        // and the indexer reads them with tensor cores (its WMMA kernel) instead of FP32 FMA
        static const bool keys_f16 = [] {
            const char * env = std::getenv("LLAMA_GLM5_POOL_KEYS_F16");
            return env != nullptr && std::atoi(env) != 0;
        }();
        ggml_tensor * keys = ggml_new_tensor_3d(
                ctx, keys_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, n_embd, pimpl->max_slots, pimpl->n_stream);
        ggml_format_name(keys, "cache_kpool_l%d", il);

        ggml_tensor * topk_hint = ggml_new_tensor_2d(
                ctx, GGML_TYPE_I32, model.hparams.indexer_top_k/kpool, pimpl->n_stream);
        ggml_format_name(topk_hint, "cache_kpool_topk_l%d", il);

        ggml_tensor * mtp_topk = nullptr;
        ggml_tensor * mtp_mask = nullptr;
        if (il >= model.hparams.n_layer()) {
            mtp_topk = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_I32, model.hparams.indexer_top_k, pimpl->n_stream);
            mtp_mask = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_F16, model.hparams.indexer_top_k, pimpl->n_stream);
            ggml_format_name(mtp_topk, "cache_mtp_topk_l%d", il);
            ggml_format_name(mtp_mask, "cache_mtp_topk_mask_l%d", il);
        }

        pimpl->map_layer_ids[il] = pimpl->layers.size();
        pimpl->layers.push_back({ il, keys, topk_hint, mtp_topk, mtp_mask, table });
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error("failed to allocate GLM pool-key cache");
        }

        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s GLM pool-key cache buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        pimpl->ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    pimpl->init_tables();

    LLAMA_LOG_INFO("%s: slots = %u, width = %u, streams = %u, layers = %zu, ring = %d, device tables = %zu\n",
            __func__, pimpl->max_slots, n_embd, pimpl->n_stream, pimpl->layers.size(), pimpl->ring, pimpl->tables.size());
}

llama_kpool_cache::~llama_kpool_cache() = default;

void llama_kpool_cache::clear(bool data) {
    invalidate();
    if (data) {
        for (auto & [_, buf] : pimpl->ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
        pimpl->init_tables();
    }
}

void llama_kpool_cache::invalidate() {
    for (auto & map : pimpl->maps) {
        map.clear();
    }
    for (uint32_t s = 0; pimpl->ring && s < pimpl->n_stream; ++s) {
        pimpl->ring_clear(s);
    }
    std::fill(pimpl->rebuild.begin(), pimpl->rebuild.end(), true);
    pimpl->mtp_index_reuse = false;
}

void llama_kpool_cache::invalidate(llama_seq_id seq_id) {
    // without a unified cache, stream s holds sequence s only (llama_kv_cache::seq_to_stream)
    if (seq_id < 0 || pimpl->n_stream == 1 || (uint32_t) seq_id >= pimpl->n_stream) {
        invalidate();
        return;
    }
    pimpl->maps[seq_id].clear();
    if (pimpl->ring) {
        pimpl->ring_clear(seq_id);
    }
    pimpl->rebuild[seq_id] = true;
    pimpl->mtp_index_reuse = false;
}

void llama_kpool_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p0 >= 0 && p1 >= 0 && p0 >= p1) {
        return;
    }

    const int64_t block0 = p0 < 0 ? 0 : p0/pimpl->kpool;
    const int64_t block1 = p1 < 0 ? INT64_MAX : (p1 - 1)/pimpl->kpool;

    if (pimpl->ring) {
        // a stream holds one sequence: stream s is sequence s, or the only stream holds them all
        for (uint32_t s = 0; s < pimpl->n_stream; ++s) {
            if (seq_id >= 0 && pimpl->n_stream > 1 && (uint32_t) seq_id != s) {
                continue;
            }
            int64_t * blocks = pimpl->ring_block[s].data();
            uint8_t * cached = pimpl->ring_cache[s].data();
            for (uint32_t slot = 0; slot < pimpl->ring_hi[s]; ++slot) {
                if (blocks[slot] >= block0 && blocks[slot] <= block1) {
                    blocks[slot] = -1;
                    cached[slot] = 0;
                }
            }
        }
        return;
    }

    for (auto & map : pimpl->maps) {
        for (auto it = map.begin(); it != map.end();) {
            const pool_id & id = it->first;
            if ((seq_id < 0 || id.seq == seq_id) && id.block >= block0 && id.block <= block1) {
                it = map.erase(it);
            } else {
                ++it;
            }
        }
    }
}

bool llama_kpool_cache::needs_rebuild(uint32_t stream0, uint32_t n_stream) const {
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    return std::any_of(
            pimpl->rebuild.begin() + stream0,
            pimpl->rebuild.begin() + stream0 + n_stream,
            [](bool rebuild) { return rebuild; });
}

void llama_kpool_cache::finish_rebuild(uint32_t stream0, uint32_t n_stream) {
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    std::fill(
            pimpl->rebuild.begin() + stream0,
            pimpl->rebuild.begin() + stream0 + n_stream,
            false);
}

void llama_kpool_cache::set_mtp_index_reuse(bool reuse) {
    // Keep the diagnostic A/B switch coherent across graph construction and
    // host input preparation.  If graph reuse is disabled, the normal pool
    // cache update inputs must remain active as well.
    if (reuse) {
        const char * env = std::getenv("LLAMA_GLM5_MTP_TOPK_SHARE");
        reuse = env == nullptr || std::atoi(env) != 0;
    }

    pimpl->mtp_index_reuse = reuse;
}

bool llama_kpool_cache::get_mtp_index_reuse() const {
    return pimpl->mtp_index_reuse;
}

void llama_kpool_cache::mark_dirty(uint32_t stream, llama_seq_id seq_id, llama_pos pos) {
    GGML_ASSERT(stream < pimpl->n_stream);
    GGML_ASSERT(seq_id >= 0 && pos >= 0);

    if (pimpl->ring) {
        const int64_t block = pos/pimpl->kpool;
        const uint32_t slot = ring_slot(stream, block);
        if (pimpl->ring_block[stream][slot] == block) {
            pimpl->ring_cache[stream][slot] = 0;
        }
        return;
    }

    auto & map = pimpl->maps[stream];
    const auto it = map.find({ seq_id, pos/pimpl->kpool });
    if (it != map.end()) {
        it->second.cached = false;
    }
}

llama_kpool_cache::stream_plan llama_kpool_cache::prepare_stream(
        uint32_t stream,
        const std::vector<pool_id> & ids,
        size_t n_scratch,
        bool rebuild) {
    GGML_ASSERT(stream < pimpl->n_stream);
    GGML_ASSERT(ids.size() + n_scratch <= pimpl->max_slots);

    if (pimpl->ring) {
        // the map semantics: a pool keeps its key while it is named on every call; any other slot is dropped
        int64_t * blocks = pimpl->ring_block[stream].data();
        uint8_t * cached = pimpl->ring_cache[stream].data();
        if (rebuild) {
            pimpl->ring_clear(stream);
        }

        static thread_local std::vector<uint8_t> named;
        named.assign(pimpl->max_slots, 0);

        stream_plan result;
        result.slots.reserve(ids.size());
        result.cached.reserve(ids.size());
        uint32_t hi = 0;
        for (const auto & id : ids) {
            GGML_ASSERT(id.block >= 0);
            const uint32_t slot = ring_slot(stream, id.block);
            GGML_ASSERT(!named[slot] && "the pools of a stream must span at most max_slots blocks");
            named[slot] = 1;
            hi = std::max(hi, slot + 1);
            const bool hit = blocks[slot] == id.block && cached[slot];
            if (!hit) {
                blocks[slot] = id.block;
                cached[slot] = 0;
            }
            result.slots.push_back((int32_t) slot);
            result.cached.push_back(hit ? 1 : 0);
        }
        for (uint32_t slot = 0; slot < pimpl->ring_hi[stream]; ++slot) {
            if (!named[slot]) {
                blocks[slot] = -1;
                cached[slot] = 0;
            }
        }
        pimpl->ring_hi[stream] = hi;

        uint32_t next_free = 0;
        result.scratch.reserve(n_scratch);
        for (size_t i = 0; i < n_scratch; ++i) {
            while (next_free < pimpl->max_slots && named[next_free]) {
                ++next_free;
            }
            GGML_ASSERT(next_free < pimpl->max_slots);
            result.scratch.push_back((int32_t) next_free++);
        }
        return result;
    }

    auto & map = pimpl->maps[stream];
    if (rebuild) {
        map.clear();
    }

    // stamp the pools that are still named, then drop the rest: one lookup per pool, where a set
    // of the current pools would be rebuilt on every decode step
    // pointers, not iterators: a rehash in the emplace below keeps elements in place but not iterators
    static thread_local std::vector<impl::entry *> found;
    found.assign(ids.size(), nullptr);

    const uint64_t gen = ++pimpl->gen;
    for (size_t i = 0; i < ids.size(); ++i) {
        auto it = map.find(ids[i]);
        if (it != map.end()) {
            GGML_ASSERT(it->second.gen != gen && "pool identities within a stream must be unique");
            it->second.gen = gen;
            found[i] = &it->second;
        }
    }

    // Drop only the pools of the sequences that this ubatch names. Server slots share one stream, and each slot
    // decodes in its own ubatch: dropping the other slots' pools here made all of them new again on their next
    // ubatch (98 new pools for 4 update rows, which aborted).
    // erase leaves the other elements in place, so the pointers in found stay valid
    static thread_local std::vector<llama_seq_id> seqs;
    seqs.clear();
    for (const auto & id : ids) {
        if (std::find(seqs.begin(), seqs.end(), id.seq) == seqs.end()) {
            seqs.push_back(id.seq);
        }
    }
    for (auto it = map.begin(); it != map.end();) {
        if (it->second.gen != gen && std::find(seqs.begin(), seqs.end(), it->first.seq) != seqs.end()) {
            it = map.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<bool> used(pimpl->max_slots);
    for (const auto & [_, value] : map) {
        GGML_ASSERT(value.slot >= 0 && (uint32_t) value.slot < pimpl->max_slots);
        GGML_ASSERT(!used[value.slot]);
        used[value.slot] = true;
    }

    stream_plan result;
    result.slots.reserve(ids.size());
    result.cached.reserve(ids.size());

    uint32_t next_free = 0;
    auto take_free = [&]() {
        while (next_free < pimpl->max_slots && used[next_free]) {
            ++next_free;
        }
        GGML_ASSERT(next_free < pimpl->max_slots);
        used[next_free] = true;
        return (int32_t) next_free++;
    };

    for (size_t i = 0; i < ids.size(); ++i) {
        impl::entry * e = found[i];
        if (e == nullptr) {
            auto [nit, inserted] = map.emplace(ids[i], impl::entry { take_free(), false, gen });
            GGML_ASSERT(inserted && "pool identities within a stream must be unique");
            e = &nit->second;
        }
        result.slots.push_back(e->slot);
        result.cached.push_back(e->cached ? 1 : 0);
    }

    result.scratch.reserve(n_scratch);
    for (size_t i = 0; i < n_scratch; ++i) {
        result.scratch.push_back(take_free());
    }

    return result;
}

void llama_kpool_cache::mark_cached(uint32_t stream, const std::vector<int32_t> & slots) {
    GGML_ASSERT(stream < pimpl->n_stream);
    if (pimpl->ring) {
        for (int32_t slot : slots) {
            GGML_ASSERT(slot >= 0 && (uint32_t) slot < pimpl->max_slots && pimpl->ring_block[stream][slot] >= 0);
            pimpl->ring_cache[stream][slot] = 1;
        }
        return;
    }
    auto & map = pimpl->maps[stream];
    if (slots.size() <= 8) {
        for (int32_t slot : slots) {
            bool found = false;
            for (auto & [_, value] : map) {
                if (value.slot == slot) {
                    value.cached = true;
                    found = true;
                    break;
                }
            }
            GGML_ASSERT(found);
        }
        return;
    }
    // A scan of the map per slot is quadratic: a full pool rebuild marks every pool, and at 450K cells (112K pools)
    // that took ~27 s on the server thread before the first token of a cached prompt. Index the slots once.
    std::vector<impl::entry *> by_slot(pimpl->max_slots, nullptr);
    for (auto & [_, value] : map) {
        GGML_ASSERT(value.slot >= 0 && (uint32_t) value.slot < pimpl->max_slots);
        by_slot[value.slot] = &value;
    }
    for (int32_t slot : slots) {
        GGML_ASSERT(slot >= 0 && (uint32_t) slot < pimpl->max_slots && by_slot[slot] != nullptr);
        by_slot[slot]->cached = true;
    }
}

uint32_t llama_kpool_cache::get_max_slots() const {
    return pimpl->max_slots;
}

uint32_t llama_kpool_cache::get_n_stream() const {
    return pimpl->n_stream;
}

uint32_t llama_kpool_cache::get_max_uncached(uint32_t stream0, uint32_t n_stream) const {
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    uint32_t result = 0;
    for (uint32_t s = stream0; s < stream0 + n_stream; ++s) {
        uint32_t current = 0;
        if (pimpl->ring) {
            for (uint32_t slot = 0; slot < pimpl->ring_hi[s]; ++slot) {
                current += pimpl->ring_block[s][slot] >= 0 && !pimpl->ring_cache[s][slot];
            }
        } else {
            for (const auto & [_, entry] : pimpl->maps[s]) {
                current += !entry.cached;
            }
        }
        result = std::max(result, current);
    }
    return result;
}

bool llama_kpool_cache::is_ring() const {
    return pimpl->ring;
}

int64_t * llama_kpool_cache::ring_blocks(uint32_t stream) {
    GGML_ASSERT(pimpl->ring && stream < pimpl->n_stream);
    return pimpl->ring_block[stream].data();
}

uint8_t * llama_kpool_cache::ring_cached(uint32_t stream) {
    GGML_ASSERT(pimpl->ring && stream < pimpl->n_stream);
    return pimpl->ring_cache[stream].data();
}

uint32_t & llama_kpool_cache::ring_hi(uint32_t stream) {
    GGML_ASSERT(pimpl->ring && stream < pimpl->n_stream);
    return pimpl->ring_hi[stream];
}

int64_t llama_kpool_cache::ring_origin(uint32_t stream) const {
    GGML_ASSERT(pimpl->ring && stream < pimpl->n_stream);
    return pimpl->ring_org[stream];
}

uint32_t llama_kpool_cache::ring_slot(uint32_t stream, int64_t block) const {
    const int64_t m = pimpl->max_slots;
    return (uint32_t) (((block - ring_origin(stream)) % m + m) % m);
}

void llama_kpool_cache::ring_rebase(uint32_t stream, int64_t origin) {
    GGML_ASSERT(pimpl->ring && stream < pimpl->n_stream);
    pimpl->maps[stream].clear();
    pimpl->ring_clear(stream);
    pimpl->ring_org[stream] = origin;
    pimpl->rebuild[stream] = true;
    pimpl->mtp_index_reuse = false;
}

size_t llama_kpool_cache::n_tables() const {
    return pimpl->tables.size();
}

llama_kpool_cache::dev_table & llama_kpool_cache::get_table(size_t i) {
    GGML_ASSERT(i < pimpl->tables.size());
    return pimpl->tables[i];
}

int32_t llama_kpool_cache::table_of_layer(int32_t il) const {
    return pimpl->layers[pimpl->map_layer_ids.at(il)].table;
}

ggml_tensor * llama_kpool_cache::store_prefix(
        ggml_context * ctx,
        ggml_tensor * cur,
        int32_t il,
        uint32_t stream0,
        uint32_t n_stream) const {
    ggml_tensor * keys = get(ctx, il, stream0, n_stream);
    GGML_ASSERT(cur->type == GGML_TYPE_F32 && cur->ne[0] == keys->ne[0] && cur->ne[1] <= keys->ne[1] && cur->ne[2] == keys->ne[2]);
    ggml_tensor * dst = ggml_view_3d(ctx, keys, keys->ne[0], cur->ne[1], n_stream, keys->nb[1], keys->nb[2], 0);
    return ggml_cpy(ctx, cur, dst);
}

ggml_tensor * llama_kpool_cache::get(
        ggml_context * ctx,
        int32_t il,
        uint32_t stream0,
        uint32_t n_stream) const {
    const int32_t ic = pimpl->map_layer_ids.at(il);
    ggml_tensor * keys = pimpl->layers[ic].keys;
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    return ggml_view_3d(ctx, keys, pimpl->n_embd, pimpl->max_slots, n_stream,
            keys->nb[1], keys->nb[2], stream0*keys->nb[2]);
}

ggml_tensor * llama_kpool_cache::store(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * idxs,
        int32_t il,
        uint32_t stream0,
        uint32_t n_stream) const {
    ggml_tensor * dst = get(ctx, il, stream0, n_stream);
    GGML_ASSERT(cur->type == GGML_TYPE_F32);
    GGML_ASSERT(cur->ne[0] == dst->ne[0] && cur->ne[2] == dst->ne[2]);
    return ggml_set_rows(ctx, dst, cur, idxs);
}

ggml_tensor * llama_kpool_cache::get_mtp_selection(
        ggml_context * ctx,
        int32_t il,
        int64_t n_selected,
        uint32_t stream0,
        uint32_t n_stream,
        bool mask) const {
    const int32_t ic = pimpl->map_layer_ids.at(il);
    ggml_tensor * src = mask ? pimpl->layers[ic].mtp_mask : pimpl->layers[ic].mtp_topk;

    GGML_ASSERT(src != nullptr && "MTP selection storage exists only for NextN layers");
    GGML_ASSERT(n_selected > 0 && n_selected <= src->ne[0]);
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    return ggml_view_3d(ctx, src, n_selected, 1, n_stream,
            src->nb[1], src->nb[1], stream0*src->nb[1]);
}

ggml_tensor * llama_kpool_cache::store_mtp_selection(
        ggml_context * ctx,
        ggml_tensor * cur,
        int32_t il,
        uint32_t stream0,
        uint32_t n_stream,
        bool mask) const {
    GGML_ASSERT(cur->ne[1] == 1 && cur->ne[2] == n_stream &&
            "MTP Top-K sharing is defined for one query per active stream");

    ggml_tensor * dst = get_mtp_selection(ctx, il, cur->ne[0], stream0, n_stream, mask);
    GGML_ASSERT(cur->type == dst->type);
    return ggml_cpy(ctx, cur, dst);
}

ggml_tensor * llama_kpool_cache::get_topk_hint(
        ggml_context * ctx,
        int32_t il,
        int64_t n_pools,
        uint32_t stream0,
        uint32_t n_stream) const {
    const int32_t ic = pimpl->map_layer_ids.at(il);
    ggml_tensor * src = pimpl->layers[ic].topk_hint;

    GGML_ASSERT(n_pools > 0 && n_pools <= src->ne[0]);
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    return ggml_view_3d(ctx, src, n_pools, 1, n_stream, src->nb[1], src->nb[1], stream0*src->nb[1]);
}

ggml_tensor * llama_kpool_cache::store_topk_hint(
        ggml_context * ctx,
        ggml_tensor * cur,
        int32_t il,
        uint32_t stream0,
        uint32_t n_stream) const {
    GGML_ASSERT(cur->ne[1] == 1 && cur->ne[2] == n_stream);

    ggml_tensor * dst = get_topk_hint(ctx, il, cur->ne[0], stream0, n_stream);
    GGML_ASSERT(cur->type == dst->type);
    return ggml_cpy(ctx, cur, dst);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kpool_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & [ctx, buf] : pimpl->ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        // with no_alloc (fit-params) the buffer is a dummy of size 0: count the size the tensors will take
        result[buft] += ggml_backend_buffer_get_base(buf.get()) == nullptr && ggml_backend_buffer_get_size(buf.get()) == 0
                ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft)
                : ggml_backend_buffer_get_size(buf.get());
    }
    return result;
}

// sel_mask and cand_mask hold only 0.0f and -INFINITY, so f16 is exact here
template <typename T> struct kpool_mask_of;

template <> struct kpool_mask_of<float> {
    static float from(float v) { return v; }
};

template <> struct kpool_mask_of<ggml_fp16_t> {
    static ggml_fp16_t from(float v) { return ggml_fp32_to_fp16(v); }
};

template <typename T>
static void kpool_mask_fill(T * dst, int64_t n) {
    std::fill(dst, dst + n, kpool_mask_of<T>::from(-INFINITY));
}

template <typename T>
static void kpool_mask_row(
                T * cur_sel,
                T * cur_cand,
          int32_t * cur_tail_cells,
                T * cur_tail_mask,
        const llama_pos * pos_at,
        const int32_t   * pool_of,
          int64_t   n_kv,
          int64_t   n_tail_pad,
        llama_pos   q,
        llama_pos   tail_start,
          int64_t   bo_vis) {
    const T v_sel  = kpool_mask_of<T>::from(0.0f);
    const T v_mask = kpool_mask_of<T>::from(-INFINITY);

    for (int64_t j = 0; j < n_kv; ++j) {
        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
        const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;
        const bool tail   = pos_at[j] >= tail_start;

        if (cur_sel != nullptr) {
            cur_sel[j] = vis && tail ? v_sel : v_mask;
        }
        // the candidate set, which the top-k budget may overrun but must never escape
        if (cur_cand != nullptr) {
            cur_cand[j] = vis && (pooled || tail) ? v_sel : v_mask;
        }

        if (vis && tail && cur_tail_cells != nullptr) {
            const int64_t ti = pos_at[j] - tail_start;
            GGML_ASSERT(ti >= 0 && ti < n_tail_pad);
            cur_tail_cells[ti] = (int32_t) j;
            cur_tail_mask [ti] = v_sel;
        }
    }
}

void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
        const llama_ubatch   * ubatch,
              uint32_t         kpool,
              uint32_t         n_kv,
              ggml_tensor    * compact_tail_cells,
              ggml_tensor    * compact_tail_mask,
              llama_kpool_cache * pool_cache,
              ggml_tensor    * pool_cache_slots,
              ggml_tensor    * pool_store_src,
              ggml_tensor    * pool_store_dst,
              ggml_tensor    * pool_update_cells,
              ggml_tensor    * pool_update_dst,
              bool             rebuild_pool_cache,
              uint32_t         stream0,
              ggml_tensor    * mask_cells,
              ggml_tensor    * mask_queries,
              ggml_tensor    * pool_iv_cells,
              ggml_tensor    * pool_iv_queries) {
    GGML_ASSERT(kv != nullptr);
    GGML_ASSERT(kpool > 0);

    GGML_ASSERT(pool_cells->buffer && "pool_cells input was not allocated");
    // pool_bias is null when a state-only MTP graph prunes the scoring that reads it
    GGML_ASSERT(pool_bias == nullptr || (pool_bias->buffer && "pool_bias input was not allocated"));
    GGML_ASSERT((sel_mask == nullptr) == (cand_mask == nullptr));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(pool_bias == nullptr || ggml_backend_buffer_is_host(pool_bias ->buffer));
    if (sel_mask != nullptr) {
        GGML_ASSERT(sel_mask->buffer && cand_mask->buffer && "dense sparse masks were not allocated");
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask ->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(cand_mask->buffer));
    }

    GGML_ASSERT(pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(pool_bias == nullptr || pool_bias ->type == GGML_TYPE_F32);
    if (sel_mask != nullptr) {
        GGML_ASSERT((sel_mask->type == GGML_TYPE_F16 || sel_mask->type == GGML_TYPE_F32) &&
                "sel_mask must be f16 or f32");
        GGML_ASSERT(cand_mask->type == sel_mask->type && "both masks must have the KQ mask's type");
    }
    GGML_ASSERT((compact_tail_cells == nullptr) == (compact_tail_mask == nullptr));
    GGML_ASSERT((pool_cache == nullptr) == (pool_cache_slots == nullptr));
    if (pool_cache != nullptr) {
        GGML_ASSERT(rebuild_pool_cache ?
                (pool_store_src != nullptr && pool_store_dst != nullptr &&
                 pool_update_cells == nullptr && pool_update_dst == nullptr) :
                (pool_store_src == nullptr && pool_store_dst == nullptr &&
                 pool_update_cells != nullptr && pool_update_dst != nullptr));
    }

    GGML_ASSERT(ggml_is_contiguous(pool_cells));
    GGML_ASSERT(pool_bias == nullptr || ggml_is_contiguous(pool_bias));
    GGML_ASSERT(sel_mask == nullptr || ggml_is_contiguous(sel_mask));
    GGML_ASSERT(cand_mask == nullptr || ggml_is_contiguous(cand_mask));

    if (compact_tail_cells != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(compact_tail_cells->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(compact_tail_mask ->buffer));
        GGML_ASSERT(compact_tail_cells->type == GGML_TYPE_I32);
        GGML_ASSERT(compact_tail_mask->type == GGML_TYPE_F16 || compact_tail_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask == nullptr || compact_tail_mask->type == sel_mask->type);
        GGML_ASSERT(ggml_is_contiguous(compact_tail_cells));
        GGML_ASSERT(ggml_is_contiguous(compact_tail_mask));
    }

    if (pool_cache != nullptr) {
        GGML_ASSERT(pool_cache_slots->buffer && "pool_cache_slots input was not allocated");
        GGML_ASSERT(ggml_backend_buffer_is_host(pool_cache_slots->buffer));
        GGML_ASSERT(pool_cache_slots->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(pool_cache_slots));

        if (rebuild_pool_cache) {
            GGML_ASSERT(pool_store_src->buffer && "pool_store_src input was not allocated");
            GGML_ASSERT(pool_store_dst->buffer && "pool_store_dst input was not allocated");
            GGML_ASSERT(ggml_backend_buffer_is_host(pool_store_src->buffer));
            GGML_ASSERT(ggml_backend_buffer_is_host(pool_store_dst->buffer));
            GGML_ASSERT(pool_store_src->type == GGML_TYPE_I32 && pool_store_dst->type == GGML_TYPE_I32);
            GGML_ASSERT(ggml_is_contiguous(pool_store_src) && ggml_is_contiguous(pool_store_dst));
        } else {
            GGML_ASSERT(pool_update_cells->buffer && "pool_update_cells input was not allocated");
            GGML_ASSERT(pool_update_dst->buffer && "pool_update_dst input was not allocated");
            GGML_ASSERT(ggml_backend_buffer_is_host(pool_update_cells->buffer));
            GGML_ASSERT(ggml_backend_buffer_is_host(pool_update_dst->buffer));
            GGML_ASSERT(pool_update_cells->type == GGML_TYPE_I32 && pool_update_dst->type == GGML_TYPE_I32);
            GGML_ASSERT(ggml_is_contiguous(pool_update_cells) && ggml_is_contiguous(pool_update_dst));
        }
    }

    const int64_t n_ns     = pool_cells->ne[1];
    const int64_t r        = kpool;
    const int64_t n_tokens = ubatch->n_tokens;

    // [TAG_KPOOL_SEQ_PARTITION] positions are unambiguous only within one sequence, so
    // one pool map per SEQUENCE, not per stream
    GGML_ASSERT(n_ns == 1 || (int64_t) ubatch->n_seqs_unq == n_ns);

    const int64_t n_ps    = (int64_t) ubatch->n_seqs_unq/n_ns;
    const int64_t n_pools = pool_cells->ne[0]/r;

    GGML_ASSERT(n_ps > 0 && (int64_t) ubatch->n_seqs_unq == n_ns*n_ps);
    GGML_ASSERT(pool_cells->ne[0] % r == 0);
    GGML_ASSERT(n_pools >= 2*n_ps);
    GGML_ASSERT(pool_cells->ne[1] == n_ns);
    GGML_ASSERT(sel_mask == nullptr || sel_mask->ne[2] == 1);
    GGML_ASSERT(sel_mask == nullptr || ggml_are_same_shape(cand_mask, sel_mask));
    GGML_ASSERT(pool_bias == nullptr || (pool_bias->ne[0] == n_pools && pool_bias->ne[2] == n_ns));
    GGML_ASSERT(n_tokens % n_ns == 0);

    const int64_t n_tps      = n_tokens/n_ns;
    const int64_t n_padq     = sel_mask ? sel_mask->ne[1] : n_tps;
    const int64_t n_tail_pad = compact_tail_cells ? compact_tail_cells->ne[0] : 0;

    GGML_ASSERT(pool_bias == nullptr || pool_bias->ne[1] == n_tps);
    GGML_ASSERT(n_padq >= n_tps);

    if (compact_tail_cells != nullptr) {
        GGML_ASSERT(n_tail_pad >= r - 1);
        GGML_ASSERT(compact_tail_cells->ne[1] == n_tps && compact_tail_cells->ne[2] == n_ns);
        GGML_ASSERT(ggml_are_same_shape(compact_tail_mask, compact_tail_cells));
    }

    if (pool_cache != nullptr) {
        GGML_ASSERT(stream0 + n_ns <= pool_cache->get_n_stream());
        GGML_ASSERT(pool_cache_slots->ne[0] == n_pools && pool_cache_slots->ne[1] == n_ns);
        if (rebuild_pool_cache) {
            GGML_ASSERT(ggml_are_same_shape(pool_store_src, pool_cache_slots));
            GGML_ASSERT(ggml_are_same_shape(pool_store_dst, pool_cache_slots));
        } else {
            GGML_ASSERT(pool_update_cells->ne[0] % r == 0 &&
                    pool_update_cells->ne[1] == n_ns);
            GGML_ASSERT(pool_update_dst->ne[0] == pool_update_cells->ne[0]/r &&
                    pool_update_dst->ne[1] == n_ns);
        }
    }

    if (cell_pool) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cell_pool->buffer));
        GGML_ASSERT(cell_pool->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(cell_pool));
        GGML_ASSERT(cell_pool->ne[0] == n_kv && cell_pool->ne[1] == n_ns);

        // one row per stream, so a shared cell has nowhere to put its second pool
        GGML_ASSERT(n_ps == 1 && "the per-cell pool view needs one sequence per stream");
    }

    if (bias) {
        GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(bias->ne[0] == n_kv && bias->ne[1] == n_tps && bias->ne[2] == n_ns);
    }

    int32_t * dst_cell_pool  = cell_pool ? (int32_t *) cell_pool->data : nullptr;
    int32_t * dst_pool_cells = (int32_t *) pool_cells->data;
    float   * dst_bias       = bias ? (float *) bias->data : nullptr;
    float   * dst_pool_bias  = pool_bias ? (float *) pool_bias->data : nullptr;
    char    * dst_sel_mask   = sel_mask  ? (char *) sel_mask ->data : nullptr;
    char    * dst_cand_mask  = cand_mask ? (char *) cand_mask->data : nullptr;

    int32_t * dst_cache_slots  = pool_cache_slots  ? (int32_t *) pool_cache_slots ->data : nullptr;
    int32_t * dst_store_src    = pool_store_src    ? (int32_t *) pool_store_src   ->data : nullptr;
    int32_t * dst_store_dst    = pool_store_dst    ? (int32_t *) pool_store_dst   ->data : nullptr;
    int32_t * dst_update_cells = pool_update_cells ? (int32_t *) pool_update_cells->data : nullptr;
    int32_t * dst_update_dst   = pool_update_dst   ? (int32_t *) pool_update_dst  ->data : nullptr;

    const ggml_type mask_type = sel_mask ? sel_mask->type : compact_tail_mask ? compact_tail_mask->type : GGML_TYPE_F16;
    const bool   mask_f16 = mask_type == GGML_TYPE_F16;
    const size_t mask_ts  = ggml_type_size(mask_type);

    GGML_ASSERT((mask_cells == nullptr) == (mask_queries == nullptr));
    GGML_ASSERT(mask_cells == nullptr || sel_mask == nullptr);
    GGML_ASSERT(mask_cells == nullptr || (mask_cells->ne[0] == n_kv && mask_cells->ne[1] == 3 && ggml_is_contiguous(mask_cells)));
    GGML_ASSERT(mask_cells == nullptr || n_ps == 1);
    GGML_ASSERT(mask_cells == nullptr || (ggml_backend_buffer_is_host(mask_cells->buffer) && ggml_backend_buffer_is_host(mask_queries->buffer)));
    float * dst_mask_cells   = mask_cells   ? (float *) mask_cells  ->data : nullptr;
    float * dst_mask_queries = mask_queries ? (float *) mask_queries->data : nullptr;

    GGML_ASSERT((pool_iv_cells == nullptr) == (pool_iv_queries == nullptr));
    GGML_ASSERT(pool_iv_cells == nullptr || pool_bias == nullptr);
    GGML_ASSERT(pool_iv_cells == nullptr || (pool_iv_cells->ne[0] == n_pools && pool_iv_cells->ne[1] == 2 &&
            pool_iv_cells->ne[3] == n_ns && ggml_is_contiguous(pool_iv_cells)));
    GGML_ASSERT(pool_iv_cells == nullptr || (pool_iv_queries->ne[0] == n_tps && pool_iv_queries->ne[1] == 2 &&
            pool_iv_queries->ne[3] == n_ns && ggml_is_contiguous(pool_iv_queries)));
    GGML_ASSERT(pool_iv_cells == nullptr ||
            (ggml_backend_buffer_is_host(pool_iv_cells->buffer) && ggml_backend_buffer_is_host(pool_iv_queries->buffer)));
    float * dst_iv_cells   = pool_iv_cells   ? (float *) pool_iv_cells  ->data : nullptr;
    float * dst_iv_queries = pool_iv_queries ? (float *) pool_iv_queries->data : nullptr;

    int32_t * dst_tail_cells = compact_tail_cells ? (int32_t *) compact_tail_cells->data : nullptr;
    char    * dst_tail_mask  = compact_tail_mask  ? (char    *) compact_tail_mask ->data : nullptr;

    if (compact_tail_cells != nullptr) {
        std::fill(dst_tail_cells, dst_tail_cells + ggml_nelements(compact_tail_cells), 0);
        if (mask_f16) {
            kpool_mask_fill((ggml_fp16_t *) dst_tail_mask, ggml_nelements(compact_tail_mask));
        } else {
            kpool_mask_fill((float *) dst_tail_mask, ggml_nelements(compact_tail_mask));
        }
    }

    // -1 marks a cell with no usable pool; host side only, never copied into cell_pool
    static thread_local std::vector<int32_t>   pool_of;
    static thread_local std::vector<int32_t>   filled;
    static thread_local std::vector<llama_pos> pos_at;
    pool_of.resize(n_kv);
    filled.resize(n_pools);

    std::vector<int64_t> run_off(n_ps);
    std::vector<int64_t> run_len(n_ps);

    auto seq_of = [&](int64_t s, int64_t ps) {
        return n_ps == 1 ? ubatch->seq_id[s*n_tps][0] : ubatch->seq_id_unq[ps];
    };

    for (int64_t s = 0; s < n_ns; ++s) {
        int32_t * cur_pool_cells = dst_pool_cells + s*(r*n_pools);
        char    * cur_sel_mask   = dst_sel_mask  ? dst_sel_mask  + s*(n_padq*n_kv)*mask_ts : nullptr;
        char    * cur_cand_mask  = dst_cand_mask ? dst_cand_mask + s*(n_padq*n_kv)*mask_ts : nullptr;
        float   * cur_pool_bias  = dst_pool_bias ? dst_pool_bias + s*(n_tps*n_pools) : nullptr;

        // the members of a valid pool are its row of cur_pool_cells, so they need no copy
        struct valid_pool {
            llama_kpool_cache::pool_id id;
            int32_t packed;
        };
        // reused between calls: at long context a fresh table per decode step is a large
        // allocation, and the page faults of a fresh mapping cost more than the loop
        static thread_local std::vector<valid_pool> valid_pools;
        valid_pools.clear();

        std::fill(cur_pool_cells, cur_pool_cells + r*n_pools, 0);
        if (cur_pool_bias != nullptr) {
            std::fill(cur_pool_bias, cur_pool_bias + n_tps*n_pools, -INFINITY);
        }

        // every slot starts invalid: BIG fails the upper bound of every query
        float * cur_iv_cells = dst_iv_cells ? dst_iv_cells + s*2*n_pools : nullptr;
        for (int64_t g = 0; cur_iv_cells != nullptr && g < n_pools; ++g) {
            cur_iv_cells[g]           = k_kpool_mask_big;
            cur_iv_cells[n_pools + g] = (float) g;
        }

        // the token loop writes rows < n_tps in full; only the padding rows need clearing
        if (cur_sel_mask != nullptr && mask_f16) {
            kpool_mask_fill((ggml_fp16_t *) (cur_sel_mask  + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
            kpool_mask_fill((ggml_fp16_t *) (cur_cand_mask + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
        } else if (cur_sel_mask != nullptr) {
            kpool_mask_fill((float *) (cur_sel_mask  + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
            kpool_mask_fill((float *) (cur_cand_mask + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
        }

        // [TAG_KPOOL_PACK] one packed run per sequence, sized on the pool range it holds.
        // NOT one full-width table per sequence: the indexer scores every slot against
        // every query, so that multiplies the score tensor by n_seq_max.
        // llama_memory_seq_cp can ask for more slots than exist; then a sequence keeps its
        // newest pools, the same cut a large hole already forces.
        {
            int64_t n_want = 0;

            for (int64_t ps = 0; ps < n_ps; ++ps) {
                const llama_seq_id seq = seq_of(s, ps);
                const auto & cells = kv->get_cells(seq);

                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j) || !cells.seq_has(j, seq)) {
                        continue;
                    }
                    const int64_t b = cells.pos_get(j)/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                run_len[ps] = found ? b_max - b_min + 1 : 0;
                n_want += run_len[ps];
            }

            if (n_want > n_pools) {
                int64_t rem = n_pools;

                for (int64_t ps = 0; ps < n_ps; ++ps) {
                    run_len[ps] = std::min(run_len[ps], rem/(n_ps - ps));
                    rem -= run_len[ps];
                }
            }

            int64_t off = 0;
            for (int64_t ps = 0; ps < n_ps; ++ps) {
                run_off[ps] = off;
                off += run_len[ps];
            }

            GGML_ASSERT(off <= n_pools);
        }

        int64_t n_done = 0;

        for (int64_t ps = 0; ps < n_ps; ++ps) {
            const llama_seq_id seq_of_pool = seq_of(s, ps);
            const auto & cells = kv->get_cells(seq_of_pool);

            const int64_t n_run = run_len[ps];

            int32_t * cur_cell_pool   = dst_cell_pool ? dst_cell_pool + s*n_kv : nullptr;
            int32_t * part_pool_cells = cur_pool_cells + run_off[ps]*r;

            std::fill(pool_of.begin(), pool_of.end(), -1);
            std::fill(filled.begin(),  filled.end(),   0);

            pos_at.resize(n_kv);
            for (int64_t j = 0; j < n_kv; ++j) {
                pos_at[j] = cells.is_empty(j) || !cells.seq_has(j, seq_of_pool) ? -1 : cells.pos_get(j);
            }

            // anchoring at the absolute p/kpool follows vLLM and SGLang, not HF
            // (valid_keys.argmax(-1)): it is the only anchor that keeps a pool's identity
            // stable from the prefill that built it to the decodes that read it.
            int64_t b_base = 0;
            {
                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (pos_at[j] < 0) {
                        continue;
                    }
                    const int64_t b = pos_at[j]/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                b_base = std::max(b_min, b_max - (n_run - 1));
            }

            for (int64_t j = 0; j < n_kv; ++j) {
                if (pos_at[j] < 0) {
                    continue;
                }

                const llama_pos p  = pos_at[j];
                const int64_t   bo = p/r - b_base;

                if (bo < 0 || bo >= n_run) {
                    continue;
                }

                pool_of[j] = (int32_t) bo;
                part_pool_cells[bo*r + (p%r)] = (int32_t) j;
                filled[bo]++;
            }

            // pool_valid = grouped_valid_keys.all(-1): the compressor consumes all r keys
            for (int64_t j = 0; j < n_kv; ++j) {
                // != rather than <: two cells claiming one position overwrite each other
                if (pool_of[j] >= 0 && filled[pool_of[j]] != (int32_t) r) {
                    pool_of[j] = -1;
                }
                if (cur_cell_pool) {
                    cur_cell_pool[j] = pool_of[j] < 0 ? 0 : pool_of[j];
                }
            }

            if (dst_mask_cells) {
                float * cur = dst_mask_cells + s*3*n_kv;
                for (int64_t j = 0; j < n_kv; ++j) {
                    const float pv = pos_at[j] < 0 ? k_kpool_mask_big : (float) pos_at[j];
                    cur[j]          = pv;
                    cur[n_kv + j]   = pv;
                    cur[2*n_kv + j] = pool_of[j] >= 0 ? k_kpool_mask_big : pv;
                }
            }

            for (int64_t p = 0; cur_iv_cells != nullptr && p < n_run; ++p) {
                if (filled[p] == (int32_t) r) {
                    cur_iv_cells[run_off[ps] + p] = (float) (run_off[ps] + p);
                }
            }

            if (pool_cache != nullptr) {
                for (int64_t p = 0; p < n_run; ++p) {
                    if (filled[p] != (int32_t) r) {
                        continue;
                    }

                    valid_pool item;
                    item.id = { seq_of_pool, b_base + p };
                    item.packed = (int32_t) (run_off[ps] + p);
                    valid_pools.push_back(std::move(item));
                }
            }

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const int64_t   i = s*n_tps + ii;

                if (ubatch->seq_id[i][0] != seq_of_pool) {
                    continue;
                }

                const llama_pos q = ubatch->pos[i];

                // q >= 0 is what makes the unsigned range test below a range test
                GGML_ASSERT(q >= 0);

                n_done++;

                // index_kpool_always_select_tail, which lands selection on pool boundaries
                const llama_pos tail_start = (q + 1)/r*r;

                // the reference tests visibility at a pool's LAST member, so a pool the
                // query straddles is dropped whole
                const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

                float * cur_bias = dst_bias ? dst_bias + i*n_kv : nullptr;
                char  * cur_sel  = cur_sel_mask  ? cur_sel_mask  + ii*n_kv*mask_ts : nullptr;
                char  * cur_cand = cur_cand_mask ? cur_cand_mask + ii*n_kv*mask_ts : nullptr;
                int32_t * cur_tail_cells = dst_tail_cells ?
                        dst_tail_cells + (s*n_tps + ii)*n_tail_pad : nullptr;
                char * cur_tail_mask = dst_tail_mask ?
                        dst_tail_mask + (s*n_tps + ii)*n_tail_pad*mask_ts : nullptr;

                if (dst_mask_queries) {
                    float * cur = dst_mask_queries + s*2*n_tps;
                    cur[ii]         = (float) q;
                    cur[n_tps + ii] = (float) tail_start;
                }

                // the slots [run_off, run_off + min(n_run, bo_vis)) of the pool_bias loop below
                if (dst_iv_queries) {
                    float * cur = dst_iv_queries + s*2*n_tps;
                    cur[ii]         = (float) (run_off[ps] + std::min<int64_t>(n_run, bo_vis) - 1);
                    cur[n_tps + ii] = (float) run_off[ps];
                }

                const int64_t bo_tail = tail_start/r - b_base;

                if (!cur_sel && !cur_cand && !cur_tail_cells) {
                    // nothing per cell to write (device-built masks)
                } else if (!cur_sel && !cur_cand && bo_tail >= 0 && bo_tail < n_run) {
                    // only the tail cells: the pool table already holds them, so skip the
                    // scan of every cell. A slot is set when a cell of this sequence holds its position.
                    for (llama_pos p = tail_start; p <= q; ++p) {
                        const int32_t j = part_pool_cells[bo_tail*r + p%r];
                        if (pos_at[j] != p) {
                            continue;
                        }
                        const int64_t ti = p - tail_start;
                        GGML_ASSERT(ti >= 0 && ti < n_tail_pad);
                        cur_tail_cells[ti] = j;
                        if (mask_f16) {
                            ((ggml_fp16_t *) cur_tail_mask)[ti] = kpool_mask_of<ggml_fp16_t>::from(0.0f);
                        } else {
                            ((float *) cur_tail_mask)[ti] = 0.0f;
                        }
                    }
                } else if (mask_f16) {
                    kpool_mask_row((ggml_fp16_t *) cur_sel, (ggml_fp16_t *) cur_cand,
                            cur_tail_cells, (ggml_fp16_t *) cur_tail_mask,
                            pos_at.data(), pool_of.data(), n_kv, n_tail_pad, q, tail_start, bo_vis);
                } else {
                    kpool_mask_row((float *) cur_sel, (float *) cur_cand,
                            cur_tail_cells, (float *) cur_tail_mask,
                            pos_at.data(), pool_of.data(), n_kv, n_tail_pad, q, tail_start, bo_vis);
                }

                if (cur_bias) {
                    for (int64_t j = 0; j < n_kv; ++j) {
                        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
                        const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;

                        cur_bias[j] = vis && pooled ? 0.0f : -INFINITY;
                    }
                }

                // the query's own sequence run only; every other slot keeps the -INFINITY
                // of the fill above, which is what keeps a foreign pool out of the budget
                float * q_pool_bias = cur_pool_bias ? cur_pool_bias + ii*n_pools + run_off[ps] : nullptr;

                for (int64_t p = 0; q_pool_bias != nullptr && p < n_run; ++p) {
                    const bool valid   = filled[p] == (int32_t) r;
                    const bool visible = p < bo_vis;

                    q_pool_bias[p] = valid && visible ? 0.0f : -INFINITY;
                }
            }
        }

        // exactly one partition per row, or a query reads another sequence's pools
        GGML_ASSERT(n_done == n_tps && "every query must belong to a sequence of the ubatch");

        if (pool_cache != nullptr) {
            std::vector<llama_kpool_cache::pool_id> ids;
            ids.reserve(valid_pools.size());
            for (const valid_pool & pool : valid_pools) {
                ids.push_back(pool.id);
            }

            const size_t n_scratch = rebuild_pool_cache ? n_pools - valid_pools.size() : 0;
            auto plan = pool_cache->prepare_stream(stream0 + s, ids, n_scratch, rebuild_pool_cache);

            int32_t * cur_cache_slots = dst_cache_slots + s*n_pools;
            std::fill(cur_cache_slots, cur_cache_slots + n_pools, 0);

            for (size_t i = 0; i < valid_pools.size(); ++i) {
                cur_cache_slots[valid_pools[i].packed] = plan.slots[i];
            }

            std::vector<int32_t> cached_now;

            if (rebuild_pool_cache) {
                int32_t * cur_store_src = dst_store_src + s*n_pools;
                int32_t * cur_store_dst = dst_store_dst + s*n_pools;

                size_t i = 0;
                for (; i < valid_pools.size(); ++i) {
                    cur_store_src[i] = valid_pools[i].packed;
                    cur_store_dst[i] = plan.slots[i];
                    cached_now.push_back(plan.slots[i]);
                }
                for (size_t j = 0; i < (size_t) n_pools; ++i, ++j) {
                    cur_store_src[i] = 0;
                    cur_store_dst[i] = plan.scratch[j];
                }
            } else {
                const size_t n_update = pool_update_cells->ne[0]/r;
                std::vector<size_t> uncached;
                for (size_t i = 0; i < valid_pools.size(); ++i) {
                    if (!plan.cached[i]) {
                        uncached.push_back(i);
                    }
                }
                // More new pools than the graph has update rows for. Update the first n_update now. The others stay
                // uncached in the map, so the next graph reserves rows for them (n_pending); until then their slots
                // hold stale keys, for this ubatch only. This aborted before.
                if (uncached.size() > n_update) {
                    LLAMA_LOG_WARN("%s: %zu pools need their keys, the graph updates %zu (valid pools %zu, tokens %u, stream %u); "
                            "the rest follow in the next ubatch\n", __func__, uncached.size(), n_update, valid_pools.size(),
                            (unsigned) n_tps, (unsigned) (stream0 + s));
                    uncached.resize(n_update);
                }

                // Inputs are laid out [r, n_update, n_stream]. Pad unused
                // update rows by recomputing an existing valid pool; writing
                // the same value to the same slot is side-effect free.
                const size_t fallback = valid_pools.empty() ? SIZE_MAX : 0;
                int32_t * cur_update_cells = dst_update_cells + s*r*n_update;
                int32_t * cur_update_dst   = dst_update_dst   + s*n_update;

                for (size_t u = 0; u < n_update; ++u) {
                    const size_t chosen = u < uncached.size() ? uncached[u] : fallback;
                    int32_t * row = cur_update_cells + u*r;
                    if (chosen != SIZE_MAX) {
                        const int32_t * members = cur_pool_cells + (int64_t) valid_pools[chosen].packed*r;
                        std::copy(members, members + r, row);
                        cur_update_dst[u] = plan.slots[chosen];
                    } else {
                        std::fill(row, row + r, 0);
                        cur_update_dst[u] = 0;
                    }
                }

                for (size_t chosen : uncached) {
                    cached_now.push_back(plan.slots[chosen]);
                }
            }

            pool_cache->mark_cached(stream0 + s, cached_now);
        }
    }

    if (pool_cache != nullptr && rebuild_pool_cache) {
        pool_cache->finish_rebuild(stream0, n_ns);
    }
}

bool llama_kpool_device_tables_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_GLM5_DEVICE_TABLES");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

int64_t llm_graph_input_kpool::get_n_pools() const {
    return fast ? fast_n_pools : pool_cells->ne[0]/kpool;
}

// the packing of llama_kv_cache_set_input_kpool for one sequence per stream: the first block of the packed run
static void kpool_fast_run(const llama_kv_cells & cells, llama_seq_id seq, int64_t r, int64_t n_pools,
        int64_t & b_base, int64_t & n_run) {
    const llama_pos p_min = cells.seq_pos_min(seq);
    const llama_pos p_max = cells.seq_pos_max(seq);
    const int64_t b_min = p_min < 0 ? 0 : p_min/r;
    const int64_t b_max = p_min < 0 ? 0 : p_max/r;
    n_run  = p_min < 0 ? 0 : std::min<int64_t>(b_max - b_min + 1, n_pools);
    b_base = p_min < 0 ? 0 : std::max(b_min, b_max - (n_run - 1));
}

bool llama_kpool_fast_eligible(
        const llama_kpool_cache * cache,
        const llama_ubatch      & ubatch,
        int64_t                   n_stream,
        int64_t                   n_pools,
        bool                      attn_indexed_or_compact,
        bool                      mtp_reuse) {
    return llama_kpool_device_tables_enabled() && llama_kpool_device_masks_enabled() && cache != nullptr &&
        cache->is_ring() && cache->n_tables() > 0 && attn_indexed_or_compact && !mtp_reuse &&
        n_stream > 0 && (int64_t) ubatch.n_seqs_unq == n_stream && ubatch.n_tokens % n_stream == 0 &&
        n_pools <= (int64_t) cache->get_max_slots();
}

void llama_kpool_fast_rebase(
        llama_kpool_cache    * cache,
        const llama_kv_cache * kv,
        const llama_ubatch   & ubatch,
        uint32_t               stream0,
        int64_t                n_stream,
        int64_t                n_pools,
        uint32_t               kpool) {
    const int64_t max_slots = cache->get_max_slots();
    const int64_t n_tps = ubatch.n_tokens/n_stream;
    for (int64_t s = 0; s < n_stream; ++s) {
        const llama_seq_id seq = ubatch.seq_id[s*n_tps][0];
        int64_t b_base = 0;
        int64_t n_run  = 0;
        kpool_fast_run(kv->get_cells(seq), seq, kpool, n_pools, b_base, n_run);
        const uint32_t st = stream0 + (uint32_t) s;
        if (n_run > 0 && ((b_base - cache->ring_origin(st)) % max_slots + max_slots) % max_slots != 0) {
            LLAMA_LOG_DEBUG("%s: stream %u: the pool run starts at block %" PRId64 ", rebasing its pool keys\n", __func__, st, b_base);
            cache->ring_rebase(st, b_base);
        }
    }
}

bool llama_kpool_fast_prepare(
        llm_graph_input_kpool & inp,
        const llama_kv_cache  * kv,
        llama_kpool_cache     * cache,
        const llama_ubatch    & ubatch,
        uint32_t                stream0,
        int64_t                 n_kv,
        int64_t                 n_stream,
        int64_t                 n_pools,
        bool                    rebuild,
        int64_t               & n_update,
        int64_t               & n_delta) {
    const int64_t r         = inp.kpool;
    const int64_t max_slots = cache->get_max_slots();
    const int64_t n_tps     = ubatch.n_tokens/n_stream;

    inp.fast_streams.resize(n_stream);
    int64_t n_need = 0;

    for (int64_t s = 0; s < n_stream; ++s) {
        auto & fs = inp.fast_streams[s];
        fs.seq = ubatch.seq_id[s*n_tps][0];

        const auto & cells = kv->get_cells(fs.seq);
        kpool_fast_run(cells, fs.seq, r, n_pools, fs.b_base, fs.n_run);
        // the slot of ordinal p is p only when the run starts at the ring origin (llama_kpool_fast_rebase); a
        // reservation's ubatch may not, and its tables are never used
        fs.aligned = fs.n_run == 0 ||
            ((fs.b_base - cache->ring_origin(stream0 + s)) % max_slots + max_slots) % max_slots == 0;

        fs.cells.assign(r*n_pools, 0);
        fs.valid.assign(n_pools, 0);
        fs.filled.assign(n_pools, 0);

        const llama_pos * pos  = cells.pos_data();
        const auto      * seqs = cells.seq_data();
        const int64_t b_base = fs.b_base;
        const int64_t n_run  = fs.n_run;
        const int64_t n_scan = std::min<int64_t>(n_kv, cells.size());
        for (int64_t j = 0; j < n_scan; ++j) {
            const llama_pos p = pos[j];
            if (p < 0 || !seqs[j].test(fs.seq)) {
                continue;
            }
            const int64_t bo = p/r - b_base;
            if (bo < 0 || bo >= n_run) {
                continue;
            }
            fs.cells[bo*r + p%r] = (int32_t) j;
            fs.filled[bo]++;
        }
        // != rather than <: two cells claiming one position overwrite each other (legacy pool_valid)
        for (int64_t p = 0; p < n_run; ++p) {
            fs.valid[p] = fs.filled[p] == (int32_t) r;
        }

        if (!rebuild) {
            const int64_t * blocks = cache->ring_blocks(stream0 + s);
            const uint8_t * cached = cache->ring_cached(stream0 + s);
            int64_t cur = 0;
            for (int64_t p = 0; p < n_run; ++p) {
                cur += fs.valid[p] && !(blocks[p] == b_base + p && cached[p]);
            }
            n_need = std::max(n_need, cur);
        }
    }

    // the rows of each device table that differ from what the device holds
    constexpr int64_t chunk = 64;
    inp.fast_dirty.resize(cache->n_tables());
    size_t max_dirty = 0;
    for (size_t t = 0; t < cache->n_tables(); ++t) {
        const auto & tb = cache->get_table(t);
        auto & dirty = inp.fast_dirty[t];
        dirty.clear();
        for (int64_t s = 0; s < n_stream; ++s) {
            const auto & fs = inp.fast_streams[s];
            const int64_t st = stream0 + s;
            const int32_t * sh_c = tb.shadow_cells.data() + st*r*max_slots;
            const uint8_t * sh_v = tb.shadow_valid.data() + st*max_slots;
            for (int64_t p0 = 0; p0 < n_pools; p0 += chunk) {
                const int64_t np = std::min(chunk, n_pools - p0);
                if (memcmp(sh_c + p0*r, fs.cells.data() + p0*r, np*r*sizeof(int32_t)) == 0 &&
                    memcmp(sh_v + p0, fs.valid.data() + p0, np) == 0) {
                    continue;
                }
                for (int64_t p = p0; p < p0 + np; ++p) {
                    if (sh_v[p] != fs.valid[p] || memcmp(sh_c + p*r, fs.cells.data() + p*r, r*sizeof(int32_t)) != 0) {
                        dirty.push_back((int32_t) (st*max_slots + p));
                    }
                }
            }
        }
        max_dirty = std::max(max_dirty, dirty.size());
    }

    // the floor: the pools a ubatch completes plus the ones it changes at its ends, so that a prompt's ubatches keep
    // the shape of their reservation
    n_delta = 32;
    while (n_delta < std::max<int64_t>((int64_t) max_dirty, (n_tps + r - 1)/r + 2)) {
        n_delta *= 2;
    }
    n_update = rebuild ? 0 : std::min<int64_t>(n_pools, std::max<int64_t>((n_tps + r - 1)/r + 1, n_need));

    inp.fast_n_pools = n_pools;
    inp.fast_n_need  = n_need;
    return true;
}

// a view with another element type of the same size (no conversion: the bits are kept)
static ggml_tensor * kpool_bitcast(ggml_context * ctx, ggml_tensor * t, ggml_type type) {
    GGML_ASSERT(ggml_type_size(type) == ggml_type_size(t->type) && !ggml_is_quantized(type) && !ggml_is_quantized(t->type));
    ggml_tensor * v = ggml_view_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[1], t->nb[2], t->nb[3], 0);
    v->type = type;
    return v;
}

static llm_graph_input_kpool::dev_update & kpool_fast_update(
        llm_graph_input_kpool & inp, ggml_context * ctx, ggml_cgraph * gf, int32_t il) {
    GGML_ASSERT(inp.fast && inp.pool_cache != nullptr);
    const int32_t t = inp.pool_cache->table_of_layer(il);
    GGML_ASSERT(t >= 0 && "a layer with pool keys must have a device table in ring mode");
    for (auto & u : inp.dev_updates) {
        if (u.table == t) {
            return u;
        }
    }

    auto & tb = inp.pool_cache->get_table(t);
    const int64_t r         = inp.kpool;
    const int64_t max_slots = inp.pool_cache->get_max_slots();
    const int64_t n_st_all  = inp.pool_cache->get_n_stream();
    const int64_t n_stream  = (int64_t) inp.fast_streams.size();
    const int64_t n_pools   = inp.fast_n_pools;
    const int64_t n_delta   = inp.fast_n_delta;

    llm_graph_input_kpool::dev_update u;
    u.table = t;
    u.idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_delta);
    u.cells = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, n_delta);
    u.iv    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_delta);
    ggml_set_input(u.idx);
    ggml_set_input(u.cells);
    ggml_set_input(u.iv);
    ggml_format_name(u.idx,   "kpool_tbl_idx_%d",   t);
    ggml_format_name(u.cells, "kpool_tbl_cells_%d", t);
    ggml_format_name(u.iv,    "kpool_tbl_iv_%d",    t);

    // the members: the I32 table as F32 rows of r (set_rows takes F32 and copies the bits)
    ggml_tensor * dst_c = ggml_view_2d(ctx, tb.cells, r, max_slots*n_st_all, r*ggml_type_size(GGML_TYPE_I32), 0);
    dst_c->type = GGML_TYPE_F32;
    ggml_tensor * c = ggml_set_rows(ctx, dst_c, u.cells, u.idx);
    ggml_build_forward_expand(gf, c);
    ggml_tensor * ci = kpool_bitcast(ctx, c, GGML_TYPE_I32);
    u.cells_out = ggml_view_2d(ctx, ci, r*n_pools, n_stream, tb.cells->nb[1], inp.stream0*tb.cells->nb[1]);
    if (n_stream > 1) {
        u.cells_out = ggml_cont(ctx, u.cells_out); // ggml_kpool_expand reads contiguous members
    }

    // row 0 of the validity rows; row 1 (the ordinals) never changes
    ggml_tensor * dst_v = ggml_view_2d(ctx, tb.iv, 1, max_slots*n_st_all, tb.iv->nb[0], 0);
    ggml_tensor * v = ggml_set_rows(ctx, dst_v, u.iv, u.idx);
    ggml_build_forward_expand(gf, v);
    u.iv_out = ggml_view_4d(ctx, v, n_pools, 2, 1, n_stream, tb.iv->nb[2], 2*tb.iv->nb[2], tb.iv->nb[1],
            inp.stream0*tb.iv->nb[1]);

    inp.dev_updates.push_back(u);
    return inp.dev_updates.back();
}

ggml_tensor * llm_graph_input_kpool::get_pool_cells(ggml_context * ctx, ggml_cgraph * gf, int32_t il) {
    return fast ? kpool_fast_update(*this, ctx, gf, il).cells_out : pool_cells;
}

ggml_tensor * llm_graph_input_kpool::get_pool_iv_cells(ggml_context * ctx, ggml_cgraph * gf, int32_t il) {
    return fast ? kpool_fast_update(*this, ctx, gf, il).iv_out : pool_iv_cells;
}

static void kpool_fast_set_input(llm_graph_input_kpool & inp, const llama_ubatch * ubatch) {
    llama_kpool_cache * cache = inp.pool_cache;
    GGML_ASSERT(cache != nullptr && cache->is_ring());

    const llama_kv_cache * kv = inp.mctx_attn->get_kv();
    const int64_t n_kv      = inp.mctx_attn->get_n_kv();
    const int64_t r         = inp.kpool;
    const int64_t n_pools   = inp.fast_n_pools;
    const int64_t n_stream  = (int64_t) inp.fast_streams.size();
    const int64_t max_slots = cache->get_max_slots();
    const uint32_t stream0  = inp.stream0;
    GGML_ASSERT(n_stream > 0 && ubatch->n_tokens % n_stream == 0);
    const int64_t n_tps = ubatch->n_tokens/n_stream;
    const bool rebuild = inp.rebuild_pool_cache;

    ggml_tensor * tail_cells = inp.compact_tail_cells && inp.compact_tail_cells->buffer ? inp.compact_tail_cells : nullptr;
    ggml_tensor * tail_mask  = inp.compact_tail_mask  && inp.compact_tail_mask ->buffer ? inp.compact_tail_mask  : nullptr;
    ggml_tensor * iv_q       = inp.pool_iv_queries    && inp.pool_iv_queries  ->buffer ? inp.pool_iv_queries    : nullptr;
    GGML_ASSERT((tail_cells == nullptr) == (tail_mask == nullptr));

    const int64_t n_tail_pad = tail_cells ? tail_cells->ne[0] : 0;
    int32_t     * dst_tail_cells = tail_cells ? (int32_t     *) tail_cells->data : nullptr;
    ggml_fp16_t * dst_tail_mask  = tail_mask  ? (ggml_fp16_t *) tail_mask ->data : nullptr;
    float       * dst_iv_q       = iv_q       ? (float       *) iv_q      ->data : nullptr;
    if (tail_cells != nullptr) {
        GGML_ASSERT(tail_mask->type == GGML_TYPE_F16 && tail_cells->ne[1] == n_tps && tail_cells->ne[2] == n_stream);
        std::fill(dst_tail_cells, dst_tail_cells + ggml_nelements(tail_cells), 0);
        kpool_mask_fill(dst_tail_mask, ggml_nelements(tail_mask));
    }
    GGML_ASSERT(iv_q == nullptr || (iv_q->ne[0] == n_tps && iv_q->ne[1] == 2 && iv_q->ne[3] == n_stream));

    const bool update = !rebuild && inp.pool_update_cells != nullptr;
    const int64_t n_update = update ? inp.pool_update_dst->ne[0] : 0;
    if (update) {
        GGML_ASSERT(inp.pool_update_cells->buffer && inp.pool_update_dst->buffer);
        GGML_ASSERT(inp.pool_update_cells->ne[0] == r*n_update && inp.pool_update_cells->ne[1] == n_stream);
    }

    static thread_local std::vector<int32_t> uncached;

    for (int64_t s = 0; s < n_stream; ++s) {
        const auto & fs = inp.fast_streams[s];
        const uint32_t st = stream0 + (uint32_t) s;
        GGML_ASSERT(fs.aligned && "the pool run of a device-table ubatch must start at its ring origin");
        const auto & cells = kv->get_cells(fs.seq);
        const llama_pos * pos  = cells.pos_data();
        const auto      * seqs = cells.seq_data();

        int64_t  * blocks = cache->ring_blocks(st);
        uint8_t  * cached = cache->ring_cached(st);
        uint32_t & hi     = cache->ring_hi(st);

        // the ring state: complete pools keep their key while they stay complete, everything else is dropped
        uncached.clear();
        int32_t first_valid = -1;
        for (int64_t p = 0; p < fs.n_run; ++p) {
            if (!fs.valid[p]) {
                blocks[p] = -1;
                cached[p] = 0;
                continue;
            }
            if (first_valid < 0) {
                first_valid = (int32_t) p;
            }
            const int64_t block = fs.b_base + p;
            if (rebuild) {
                blocks[p] = block;
                cached[p] = 1; // this graph computes every pool
            } else if (!(blocks[p] == block && cached[p])) {
                blocks[p] = block;
                cached[p] = 0;
                uncached.push_back((int32_t) p);
            }
        }
        for (uint32_t p = (uint32_t) fs.n_run; p < hi; ++p) {
            blocks[p] = -1;
            cached[p] = 0;
        }
        hi = (uint32_t) fs.n_run;

        if (update) {
            GGML_ASSERT((int64_t) uncached.size() <= n_update && "the graph has too few pool update rows");
            int32_t * cur_cells = (int32_t *) inp.pool_update_cells->data + s*r*n_update;
            int32_t * cur_dst   = (int32_t *) inp.pool_update_dst  ->data + s*n_update;
            for (int64_t u = 0; u < n_update; ++u) {
                // pad by recomputing a valid pool: the same value to the same slot
                const int32_t p = u < (int64_t) uncached.size() ? uncached[u] : first_valid;
                if (p >= 0) {
                    std::copy(fs.cells.data() + p*r, fs.cells.data() + (p + 1)*r, cur_cells + u*r);
                    cur_dst[u] = p;
                } else {
                    std::fill(cur_cells + u*r, cur_cells + (u + 1)*r, 0);
                    cur_dst[u] = 0; // no valid pool: slot 0 is not valid either
                }
            }
            for (int32_t p : uncached) {
                cached[p] = 1;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = s*n_tps + ii;
            GGML_ASSERT(ubatch->seq_id[i][0] == fs.seq);
            const llama_pos q = ubatch->pos[i];
            GGML_ASSERT(q >= 0);

            const llama_pos tail_start = (q + 1)/r*r;
            const int64_t   bo_vis     = std::max<int64_t>(0, tail_start/r - fs.b_base);

            if (dst_iv_q != nullptr) {
                float * cur = dst_iv_q + s*2*n_tps;
                cur[ii]         = (float) (std::min<int64_t>(fs.n_run, bo_vis) - 1);
                cur[n_tps + ii] = 0.0f;
            }

            if (dst_tail_cells == nullptr) {
                continue;
            }
            int32_t     * cur_tail_cells = dst_tail_cells + i*n_tail_pad;
            ggml_fp16_t * cur_tail_mask  = dst_tail_mask  + i*n_tail_pad;
            const ggml_fp16_t v_sel = ggml_fp32_to_fp16(0.0f);
            const int64_t bo_tail = tail_start/r - fs.b_base;
            if (bo_tail >= 0 && bo_tail < fs.n_run) {
                for (llama_pos p = tail_start; p <= q; ++p) {
                    const int32_t j = fs.cells[bo_tail*r + p%r];
                    if (pos[j] != p || !seqs[j].test(fs.seq)) {
                        continue;
                    }
                    const int64_t ti = p - tail_start;
                    GGML_ASSERT(ti >= 0 && ti < n_tail_pad);
                    cur_tail_cells[ti] = j;
                    cur_tail_mask [ti] = v_sel;
                }
            } else {
                // the scan of kpool_mask_row: the cells of the sequence in [tail_start, q]
                for (int64_t j = 0; j < n_kv; ++j) {
                    const llama_pos p = pos[j];
                    if (p < tail_start || p > q || !seqs[j].test(fs.seq)) {
                        continue;
                    }
                    const int64_t ti = p - tail_start;
                    GGML_ASSERT(ti >= 0 && ti < n_tail_pad);
                    cur_tail_cells[ti] = (int32_t) j;
                    cur_tail_mask [ti] = v_sel;
                }
            }
        }
    }

    // the rows that changed, into each device table this graph updates; padding repeats a row with its new value
    for (auto & u : inp.dev_updates) {
        auto & tb = cache->get_table(u.table);
        const auto & dirty = inp.fast_dirty[u.table];
        const int64_t n_delta = u.idx->ne[0];
        GGML_ASSERT((int64_t) dirty.size() <= n_delta && "the graph has too few device table rows");
        GGML_ASSERT(u.idx->buffer && u.cells->buffer && u.iv->buffer);

        int32_t * idx = (int32_t *) u.idx->data;
        int32_t * cl  = (int32_t *) u.cells->data; // F32 tensor, I32 bits
        float   * iv0 = (float   *) u.iv->data;
        for (int64_t k = 0; k < n_delta; ++k) {
            const int32_t row = k < (int64_t) dirty.size() ? dirty[k] : dirty.empty() ? (int32_t) (stream0*max_slots) : dirty[0];
            const int64_t s = row/max_slots - stream0;
            const int64_t p = row%max_slots;
            GGML_ASSERT(s >= 0 && s < n_stream && p < n_pools);
            const auto & fs = inp.fast_streams[s];
            idx[k] = row;
            std::copy(fs.cells.data() + p*r, fs.cells.data() + (p + 1)*r, cl + k*r);
            iv0[k] = fs.valid[p] ? (float) p : k_kpool_mask_big;
        }
        for (int32_t row : dirty) {
            const int64_t s = row/max_slots - stream0;
            const int64_t p = row%max_slots;
            const auto & fs = inp.fast_streams[s];
            std::copy(fs.cells.data() + p*r, fs.cells.data() + (p + 1)*r, tb.shadow_cells.data() + (int64_t) row*r);
            tb.shadow_valid[row] = fs.valid[p];
        }
    }

    if (rebuild) {
        cache->finish_rebuild(stream0, (uint32_t) n_stream);
    }
}

void llm_graph_input_kpool::set_input(const llama_ubatch * ubatch) {
    // The K/G store runs on every normal graph, including dense attention.
    // Reused MTP iterations intentionally omit that store, so its sole input
    // can be pruned by the allocator and has no backing buffer.
    if (k_idxs->buffer != nullptr) {
        mctx_idx->set_input_k_idxs(k_idxs, ubatch);
    } else {
        GGML_ASSERT(pool_cache != nullptr && pool_cache->get_mtp_index_reuse());
    }

    if (fast) {
        kpool_fast_set_input(*this, ubatch);
        return;
    }

    if (pool_cells == nullptr) {
        return;
    }

    // The compact suffix is consumed by indexed prefill and decode. On a dense
    // compatibility graph it is pruned and intentionally has no allocator buffer.
    ggml_tensor * tail_cells = compact_tail_cells && compact_tail_cells->buffer ? compact_tail_cells : nullptr;
    ggml_tensor * tail_mask  = compact_tail_mask  && compact_tail_mask ->buffer ? compact_tail_mask  : nullptr;
    GGML_ASSERT((tail_cells == nullptr) == (tail_mask == nullptr));

    // Reuse steps need a fresh compact tail map, but deliberately do not
    // mutate completed pool keys. Mark only pools touched by speculative
    // positions dirty so catch-up recompresses them without rebuilding the
    // full long-context cache.
    if (pool_cache != nullptr && pool_cache->get_mtp_index_reuse()) {
        const int64_t n_stream = pool_cells->ne[1];
        GGML_ASSERT(n_stream > 0 && ubatch->n_tokens%n_stream == 0);
        const int64_t n_tps = ubatch->n_tokens/n_stream;

        for (int64_t s = 0; s < n_stream; ++s) {
            for (int64_t i = s*n_tps; i < (s + 1)*n_tps; ++i) {
                GGML_ASSERT(ubatch->n_seq_id[i] > 0);
                pool_cache->mark_dirty(stream0 + s, ubatch->seq_id[i][0], ubatch->pos[i]);
            }
        }
    }

    llama_kpool_cache * active_pool_cache =
            pool_cache != nullptr && !pool_cache->get_mtp_index_reuse() ? pool_cache : nullptr;

    llama_kv_cache_set_input_kpool(
            mctx_attn->get_kv(),
            /* cell_pool */ nullptr, pool_cells, /* bias */ nullptr, pool_bias && pool_bias->buffer ? pool_bias : nullptr,
            // a state-only MTP graph prunes the masks along with attention
            sel_mask && sel_mask->buffer ? sel_mask : nullptr, cand_mask && cand_mask->buffer ? cand_mask : nullptr,
            ubatch, kpool, mctx_attn->get_n_kv(), tail_cells, tail_mask,
            active_pool_cache,
            active_pool_cache ? pool_cache_slots  : nullptr,
            active_pool_cache ? pool_store_src    : nullptr,
            active_pool_cache ? pool_store_dst    : nullptr,
            active_pool_cache ? pool_update_cells : nullptr,
            active_pool_cache ? pool_update_dst   : nullptr,
            active_pool_cache ? rebuild_pool_cache : false,
            mctx_idx->get_stream_base(),
            mask_cells   && mask_cells  ->buffer ? mask_cells   : nullptr,
            mask_queries && mask_queries->buffer ? mask_queries : nullptr,
            pool_iv_cells   && pool_iv_cells  ->buffer ? pool_iv_cells   : nullptr,
            pool_iv_queries && pool_iv_queries->buffer ? pool_iv_queries : nullptr);
}

bool llm_graph_input_kpool::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);
    const auto * attn = mctx->get_attn();
    const auto * idx  = mctx->get_idx();
    auto * cache = mctx->get_kpool_cache();

    const char * pool_cache_env = std::getenv("LLAMA_GLM5_POOL_CACHE");
    if (pool_cache_env != nullptr && std::atoi(pool_cache_env) == 0) {
        cache = nullptr;
    }

    // the memory contexts are new objects for every ubatch: set_input reads the current ones (comparing their
    // addresses only matched when the allocator happened to return the same ones)
    mctx_attn = attn;
    mctx_idx  = idx;

    bool res = true;
    res &= cache == pool_cache;
    res &= idx->get_stream_base() == stream0;
    res &= k_idxs->ne[0] == params.ubatch.n_tokens;
    res &= params.hparams.indexer_kpool == kpool;

    if (pool_cells == nullptr && !fast) {
        return res;
    }

    const int64_t n_kv = attn->get_n_kv();
    const int64_t n_stream = params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq;
    if (n_stream <= 0 || params.ubatch.n_tokens%n_stream != 0 || params.ubatch.n_seqs_unq%n_stream != 0) {
        return false;
    }

    const int64_t n_tps = params.ubatch.n_tokens/n_stream;
    const int64_t n_ps = params.ubatch.n_seqs_unq/n_stream;
    const int64_t n_pools = llama_kpool_n_pools(n_kv, kpool, n_ps);
    const int64_t n_pool_select = llama_kpool_select_k(
            n_pools, params.hparams.indexer_top_k, params.hparams.indexer_kpool);
    const int64_t n_selected = kpool*n_pool_select;
    const int64_t n_compact = GGML_PAD(n_selected + kpool - 1, 256);
    const bool indexed_attn = llama_kpool_indexed_attn_enabled(n_kv, n_tps);
    const bool compact_decode = n_tps == 1 && n_kv >= n_compact;
    const bool dense_masks = !indexed_attn && !compact_decode;

    auto shape = [](const ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return t != nullptr && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2 && t->ne[3] == ne3;
    };

    {
        const bool want_fast = res && cache != nullptr && n_ps == 1 &&
                llama_kpool_fast_eligible(cache, params.ubatch, n_stream, n_pools, indexed_attn || compact_decode,
                        cache->get_mtp_index_reuse());
        if (want_fast) {
            llama_kpool_fast_rebase(cache, attn->get_kv(), params.ubatch, stream0, n_stream, n_pools, kpool);
        }
        const bool rebuild   = cache != nullptr && cache->needs_rebuild(stream0, n_stream);
        const bool mtp_reuse = cache != nullptr && cache->get_mtp_index_reuse();
        if (!res || rebuild != rebuild_pool_cache || mtp_reuse != mtp_index_reuse || want_fast != fast) {
            return false;
        }
        const int64_t n_pools_built = fast_n_pools;
        int64_t n_update_fast = 0;
        int64_t n_delta_fast  = 0;
        if (want_fast) {
            llama_kpool_fast_prepare(*this, attn->get_kv(), cache, params.ubatch, stream0, n_kv, n_stream, n_pools,
                    rebuild, n_update_fast, n_delta_fast);
        }
        if (fast) {
            res &= n_pools_built == n_pools;
            res &= fast_n_delta == n_delta_fast;
            res &= shape(pool_iv_queries, n_tps, 2, 1, n_stream);
            res &= shape(compact_tail_cells, n_compact - n_selected, n_tps, n_stream, 1);
            res &= shape(compact_tail_mask, n_compact - n_selected, n_tps, n_stream, 1);
            if (rebuild) {
                res &= pool_update_cells == nullptr && pool_update_dst == nullptr;
            } else {
                res &= shape(pool_update_cells, kpool*n_update_fast, n_stream, 1, 1);
                res &= shape(pool_update_dst, n_update_fast, n_stream, 1, 1);
            }
            return res;
        }
    }

    res &= shape(pool_cells, kpool*n_pools, n_stream, 1, 1);
    if (llama_kpool_device_masks_enabled()) {
        res &= shape(pool_iv_cells, n_pools, 2, 1, n_stream);
        res &= shape(pool_iv_queries, n_tps, 2, 1, n_stream);
    } else {
        res &= shape(pool_bias, n_pools, n_tps, n_stream, 1);
    }
    res &= shape(compact_tail_cells, n_compact - n_selected, n_tps, n_stream, 1);
    res &= shape(compact_tail_mask, n_compact - n_selected, n_tps, n_stream, 1);

    if (pool_bias_f16 != nullptr) {
        res &= shape(pool_bias_f16, n_pools, n_tps, 1, n_stream);
    }

    const bool device_masks = dense_masks && n_ps == 1 && llama_kpool_device_masks_enabled();
    res &= (dense_masks && !device_masks) == (sel_mask != nullptr && cand_mask != nullptr);
    res &= device_masks == (mask_cells != nullptr && mask_queries != nullptr);
    if (dense_masks && !device_masks) {
        res &= shape(sel_mask, n_kv, n_tps, 1, n_stream);
        res &= shape(cand_mask, n_kv, n_tps, 1, n_stream);
    }
    if (device_masks) {
        res &= shape(mask_cells, n_kv, 3, 1, n_stream);
        res &= shape(mask_queries, n_tps, 2, 1, n_stream);
    }

    const bool rebuild = cache != nullptr && cache->needs_rebuild(stream0, n_stream);
    const bool mtp_reuse = cache != nullptr && cache->get_mtp_index_reuse();
    res &= rebuild == rebuild_pool_cache;
    res &= mtp_reuse == mtp_index_reuse;

    if (cache != nullptr) {
        res &= shape(pool_cache_slots, n_pools, n_stream, 1, 1);
        if (rebuild) {
            res &= shape(pool_store_src, n_pools, n_stream, 1, 1);
            res &= shape(pool_store_dst, n_pools, n_stream, 1, 1);
            res &= pool_update_cells == nullptr && pool_update_dst == nullptr;
        } else {
            const int64_t n_pending = cache->get_max_uncached(stream0, n_stream);
            const int64_t n_update = std::min<int64_t>(
                    n_pools, (n_tps + kpool - 1)/kpool + n_ps + n_pending);
            res &= shape(pool_update_cells, kpool*n_update, n_stream, 1, 1);
            res &= shape(pool_update_dst, n_update, n_stream, 1, 1);
            res &= pool_store_src == nullptr && pool_store_dst == nullptr;
        }
    }

    return res;
}
