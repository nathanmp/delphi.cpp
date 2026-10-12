#include "hotswap.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama-model.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

bool perplexity_hotswap_write_status(const std::string & path, const char * state, int64_t iteration, int64_t sequence) {
    common_log_flush(common_log_main());
    const std::string tmp = path + ".tmp";
    for (int attempt = 0; attempt < 250; ++attempt) {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << state << ' ' << iteration << ' ' << sequence << '\n';
        out.close();
        if (out) {
            std::error_code error;
#ifdef _WIN32
            fs::remove(path, error);
#endif
            fs::rename(tmp, path, error);
            if (!error) { return true; }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    LOG_ERR("hotswap: cannot publish status to %s\n", path.c_str());
    return false;
}

bool perplexity_hotswap_read_command(const std::string & path, perplexity_hotswap_command & command) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { return false; }
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    if (content.find('\n') == std::string::npos) { return false; }
    std::istringstream parser(content);
    command = {};
    if (!(parser >> command.action)) { return false; }
    if (!(parser >> command.sequence)) {
        command.sequence = -1;
        parser.clear();
    }
    std::getline(parser, command.path);
    const auto first = command.path.find_first_not_of(" \t\r");
    command.path = first == std::string::npos ? "" : command.path.substr(first);
    if (!command.path.empty() && command.path.back() == '\r') { command.path.pop_back(); }
    std::error_code error;
    fs::remove(path, error);
    return !error;
}

void perplexity_hotswap_clear_packed() {
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        auto reg = ggml_backend_reg_get(i);
        auto inplace = (void (*)(int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_packed_set_inplace");
        auto int4 = (void (*)(int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_set_mode");
        auto clear = (void (*)()) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_packed_clear");
        if (inplace) { inplace(0); }
        if (int4) { int4(0); }
        if (clear) { clear(); }
    }
}

struct hot_file {
    struct closer { void operator()(FILE * file) const { fclose(file); } };
    std::unique_ptr<FILE, closer> input;
    ggml_context * tensors = nullptr;
    gguf_context * meta = nullptr;
    std::string path;
    uint64_t size = 0;

    explicit hot_file(const std::string & file) : path(file) {
        input.reset(fopen(file.c_str(), "rb"));
        if (!input) { throw std::runtime_error("cannot open GGUF: " + file); }
#ifdef _WIN32
        _fseeki64(input.get(), 0, SEEK_END);
        size = _ftelli64(input.get());
        _fseeki64(input.get(), 0, SEEK_SET);
#else
        fseeko(input.get(), 0, SEEK_END);
        size = ftello(input.get());
        fseeko(input.get(), 0, SEEK_SET);
#endif
        meta = gguf_init_from_file_ptr(input.get(), { true, &tensors });
        if (!meta) { throw std::runtime_error("cannot read GGUF metadata: " + file); }
    }
    ~hot_file() { gguf_free(meta); ggml_free(tensors); }

    bool read(uint64_t offset, std::vector<uint8_t> & bytes) const {
        if (offset > size || bytes.size() > size-offset || offset > (uint64_t) std::numeric_limits<int64_t>::max()) { return false; }
#ifdef _WIN32
        if (_fseeki64(input.get(), offset, SEEK_SET) != 0) { return false; }
#else
        if (fseeko(input.get(), offset, SEEK_SET) != 0) { return false; }
#endif
        return fread(bytes.data(), 1, bytes.size(), input.get()) == bytes.size();
    }
};

struct hot_identity {
    fs::file_time_type time;
    uintmax_t size;
    bool operator==(const hot_identity & rhs) const { return time == rhs.time && size == rhs.size; }
};

static hot_identity identity(const std::string & path) {
    return { fs::last_write_time(path), fs::file_size(path) };
}

static bool same_shape(const ggml_tensor & a, const ggml_tensor & b) {
    return std::equal(a.ne, a.ne + GGML_MAX_DIMS, b.ne);
}

struct perplexity_hotswap::impl {
    using buffer_ptr = std::shared_ptr<ggml_backend_buffer>;
    struct target {
        ggml_tensor * tensor;
        ggml_tensor original;
        buffer_ptr active;
        std::string refusal = "not a plain source tensor";
    };
    struct source {
        std::string path;
        hot_identity seen;
    };
    struct update {
        target * dest;
        ggml_tensor before;
        buffer_ptr before_buffer;
        ggml_tensor after;
        buffer_ptr after_buffer;
    };
    std::map<std::string, std::vector<target>> targets;
    std::vector<source> sources;
    std::vector<update> pending;
    std::map<std::string, hot_identity> pending_sources;

    void stage(const hot_file & file, bool explicit_patch) {
        for (int64_t i = 0; i < gguf_get_n_tensors(file.meta); ++i) {
            const std::string name = gguf_get_tensor_name(file.meta, i);
            auto found = targets.find(name);
            if (found == targets.end()) {
                if (explicit_patch) { throw std::runtime_error("unused or unknown tensor: " + name); }
                continue;
            }
            const ggml_tensor * disk = ggml_get_tensor(file.tensors, name.c_str());
            const size_t bytes = ggml_nbytes(disk);
            const size_t offset = gguf_get_data_offset(file.meta) + gguf_get_tensor_offset(file.meta, i);
            if (offset > file.size || bytes > file.size - offset) { throw std::runtime_error("truncated tensor: " + name); }
            std::vector<uint8_t> data(bytes);
            if (!file.read(offset, data)) { throw std::runtime_error("cannot read tensor: " + name); }
            if (!ggml_validate_row_data(disk->type, data.data(), bytes)) { throw std::runtime_error("invalid tensor data: " + name); }
            for (auto & dest : found->second) {
                const auto & original = dest.original;
                if (!dest.refusal.empty()) { throw std::runtime_error(dest.refusal + ": " + name); }
                if (!same_shape(original, *disk)) { throw std::runtime_error("shape mismatch: " + name); }
                if (original.view_src || !ggml_is_contiguous(&original) || original.extra || !original.buffer) {
                    throw std::runtime_error("view, transformed or unavailable tensor: " + name);
                }
                auto buft = ggml_backend_buffer_get_type(original.buffer);
                auto device = ggml_backend_buft_get_device(buft);
                if (!device) { device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); }
                auto reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
                auto numa_buft = reg ? (ggml_backend_buffer_type_t (*)())
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_buffer_type") : nullptr;
                const bool is_numa = numa_buft && buft == numa_buft();
                if (!device || (buft != ggml_backend_dev_buffer_type(device) &&
                    buft != ggml_backend_dev_host_buffer_type(device) && !is_numa)) {
                    throw std::runtime_error("split or repacked buffer unsupported: " + name);
                }
                ggml_context_ptr ctx(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
                ggml_tensor * replacement = ggml_new_tensor(ctx.get(), disk->type, GGML_MAX_DIMS, disk->ne);
                replacement->flags = original.flags;
                ggml_set_name(replacement, name.c_str());
                buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, replacement)),
                                  ggml_backend_buffer_free);
                if (!buffer) { throw std::runtime_error("allocation failed on original device: " + name); }
                ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                replacement->buffer = buffer.get();
                replacement->data = ggml_backend_buffer_get_base(buffer.get());
                if (is_numa && (original.flags & GGML_TENSOR_FLAG_NUMA_EXPERT)) {
                    auto load = (bool (*)(ggml_tensor *, const void *, size_t))
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_load_expert_tensor");
                    if (!load || !load(replacement, data.data(), bytes)) {
                        throw std::runtime_error("NUMA expert load failed: " + name);
                    }
                } else {
                    ggml_backend_buffer_clear(buffer.get(), 0);
                    ggml_backend_tensor_set(replacement, data.data(), 0, bytes);
                }
                ggml_tensor after = original;
                after.type = replacement->type;
                std::copy(replacement->nb, replacement->nb + GGML_MAX_DIMS, after.nb);
                after.buffer = replacement->buffer;
                after.data = replacement->data;
                after.extra = replacement->extra;
                pending.push_back({ &dest, *dest.tensor, dest.active, after, std::move(buffer) });
            }
        }
    }

    void apply() {
        for (auto & update : pending) {
            *update.dest->tensor = update.after;
            update.dest->active = update.after_buffer;
            // Backend metadata must refer to the persistent model descriptor, not the temporary GGUF descriptor.
            if (update.after_buffer && ggml_backend_buffer_init_tensor(update.after_buffer.get(), update.dest->tensor) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("tensor initialization failed: ") + update.dest->tensor->name);
            }
        }
    }
};

perplexity_hotswap::perplexity_hotswap() : pimpl(new impl) {}
perplexity_hotswap::~perplexity_hotswap() {
    rollback();
    for (auto & entry : pimpl->targets) {
        for (auto & target : entry.second) { *target.tensor = target.original; }
    }
}

bool perplexity_hotswap::init(llama_model * model, const std::string & path, std::string & error) {
    return init(model->tensors_by_name, path, error);
}

bool perplexity_hotswap::init(const std::vector<std::pair<std::string, ggml_tensor *>> & tensors,
                              const std::string & path, std::string & error) try {
    for (const auto & entry : tensors) {
        auto & list = pimpl->targets[entry.first];
        if (std::none_of(list.begin(), list.end(), [&](const impl::target & t) { return t.tensor == entry.second; })) {
            list.push_back({ entry.second, *entry.second, {}, "not a plain source tensor" });
        }
    }
    hot_file first(path);
    std::vector<std::string> files = { path };
    const int64_t key = gguf_find_key(first.meta, "split.count");
    if (key >= 0 && gguf_get_val_u16(first.meta, key) > 1) {
        const int splits = gguf_get_val_u16(first.meta, key);
        const int64_t number_key = gguf_find_key(first.meta, "split.no");
        if (number_key < 0) { throw std::runtime_error("missing split.no"); }
        const int split = gguf_get_val_u16(first.meta, number_key);
        std::vector<char> prefix(path.size() + 64), name(path.size() + 64);
        if (llama_split_prefix(prefix.data(), prefix.size(), path.c_str(), split, splits) <= 0) {
            throw std::runtime_error("invalid split filename");
        }
        files.clear();
        for (int i = 0; i < splits; ++i) {
            llama_split_path(name.data(), name.size(), prefix.data(), i, splits);
            files.emplace_back(name.data());
        }
    }
    for (const auto & file : files) {
        hot_file source(file);
        pimpl->sources.push_back({ file, identity(file) });
        for (int64_t i = 0; i < gguf_get_n_tensors(source.meta); ++i) {
            const std::string name = gguf_get_tensor_name(source.meta, i);
            const auto found = pimpl->targets.find(name);
            if (found == pimpl->targets.end()) { continue; }
            const auto disk = ggml_get_tensor(source.tensors, name.c_str());
            for (auto & target : found->second) {
                if (target.original.type == disk->type && same_shape(target.original, *disk)) {
                    target.refusal.clear();
                }
                if (name.find("attn_k_b") != std::string::npos || name.find("attn_v_b") != std::string::npos ||
                    name.find("attn_kv_b") != std::string::npos || name.find("param_sink") != std::string::npos ||
                    name.find("attn_kv_a_norm") != std::string::npos || name.find("ffn_gate_inp_s") != std::string::npos) {
                    target.refusal = "load-time-derived consumer unsupported";
                }
                if (name.size() >= 7 && name.compare(name.size()-7, 7, ".weight") == 0 &&
                    gguf_find_tensor(source.meta, (name.substr(0, name.size()-7) + ".scale").c_str()) >= 0) {
                    target.refusal = "external weight scales unsupported";
                }
            }
        }
    }
    return true;
} catch (const std::exception & e) { error = e.what(); return false; }

int perplexity_hotswap::reload(const std::string & patch_path, std::string & error) try {
    if (!pimpl->pending.empty()) { throw std::runtime_error("previous update is not committed"); }
    if (!patch_path.empty()) {
        hot_file patch(patch_path);
        if (gguf_get_n_tensors(patch.meta) == 0) { throw std::runtime_error("empty tensor patch"); }
        pimpl->stage(patch, true);
        // Each explicit patch is one isolated job relative to the loaded baseline.
        for (auto & entry : pimpl->targets) {
            for (auto & target : entry.second) {
                if (target.active && std::none_of(pimpl->pending.begin(), pimpl->pending.end(),
                    [&](const impl::update & update) { return update.dest == &target; })) {
                    pimpl->pending.push_back({ &target, *target.tensor, target.active, target.original, {} });
                }
            }
        }
    } else {
        for (const auto & source : pimpl->sources) {
            const auto now = identity(source.path);
            if (!(now == source.seen)) {
                hot_file file(source.path);
                pimpl->stage(file, false);
                if (!(identity(source.path) == now)) { throw std::runtime_error("source changed during reload: " + source.path); }
                pimpl->pending_sources.emplace(source.path, now);
            }
        }
    }
    const int count = pimpl->pending.size();
    if (count == 0) { pimpl->pending_sources.clear(); }
    pimpl->apply();
    return count;
} catch (const std::exception & e) {
    error = e.what();
    rollback();
    return -1;
}

int perplexity_hotswap::restore(std::string & error) {
    if (!pimpl->pending.empty()) { error = "previous update is not committed"; return -1; }
    for (auto & entry : pimpl->targets) {
        for (auto & target : entry.second) {
            if (target.active) { pimpl->pending.push_back({ &target, *target.tensor, target.active, target.original, {} }); }
        }
    }
    pimpl->apply();
    return pimpl->pending.size();
}

void perplexity_hotswap::commit() {
    for (const auto & update : pimpl->pending) {
        LOG_INF("hotswap: reloaded tensor '%s' %s -> %s (%zu bytes, %s)\n", update.dest->tensor->name,
                ggml_type_name(update.before.type), ggml_type_name(update.after.type), ggml_nbytes(&update.after),
                ggml_backend_buffer_name(update.after.buffer));
    }
    for (auto & source : pimpl->sources) {
        const auto found = pimpl->pending_sources.find(source.path);
        if (found != pimpl->pending_sources.end()) { source.seen = found->second; }
    }
    pimpl->pending.clear();
    pimpl->pending_sources.clear();
}

void perplexity_hotswap::rollback() {
    for (auto it = pimpl->pending.rbegin(); it != pimpl->pending.rend(); ++it) {
        *it->dest->tensor = it->before;
        it->dest->active = it->before_buffer;
    }
    pimpl->pending.clear();
    pimpl->pending_sources.clear();
}
