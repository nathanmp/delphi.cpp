#include "server-prefix-cache.h"

#include "server-ckpt-spill.h"

#include "ggml.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <list>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <errno.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

#define PFX_INF(fmt, ...) LOG_INF("prefix cache: " fmt, __VA_ARGS__)
#define PFX_WRN(fmt, ...) LOG_WRN("prefix cache: " fmt, __VA_ARGS__)
#define PFX_DBG(fmt, ...) LOG_DBG("prefix cache: " fmt, __VA_ARGS__)

// File format, version 1. All fields in the byte order of the machine.
//   pfx_header, the n_tokens tokens of the prefix, then the data of the target context, of the draft context and of the
//   speculative state, in this order.
// The version changes when the header or the meaning of a field changes. A file of another version is not read.
static constexpr uint32_t PFX_MAGIC   = 0x58465044; // "DPFX"
static constexpr uint32_t PFX_VERSION = 1;

struct pfx_header {
    uint32_t magic;
    uint32_t version;
    uint64_t model_key;   // model files and context setup
    uint64_t prefix_hash; // of the tokens
    int64_t  n_tokens;
    int32_t  pos_min;
    int32_t  pos_max;
    uint64_t size_tgt;
    uint64_t size_dft;
    uint64_t size_spec;
    uint64_t checksum;    // of the tokens and the three data blocks
};

static_assert(sizeof(pfx_header) == 72, "the header has padding");

// snapshots being written or waiting to be written
static constexpr int PFX_MAX_QUEUE = 2;

// read speed in bytes per second that a read is planned with before any speed is measured
static constexpr double PFX_DEFAULT_BPS = 100e6;

// Speed of a state restore (host to device) before one is measured. Measured 2026-11-08 on an R9700: 22 GB/s for 1.5 GB.
// It is better to guess high: a restore that is skipped is never measured.
static constexpr double PFX_DEFAULT_RESTORE_BPS = 10e9;

static constexpr uint64_t PFX_SEED = 0xcbf29ce484222325ULL;

static const char * const PFX_PREFIX = "dpfx-";

static int64_t pfx_pid() {
#ifdef _WIN32
    return (int64_t) _getpid();
#else
    return (int64_t) getpid();
#endif
}

static bool pfx_pid_dead(int64_t pid) {
#ifdef _WIN32
    (void) pid;
    return false;
#else
    return pid > 0 && kill((pid_t) pid, 0) == -1 && errno == ESRCH;
#endif
}

static uint64_t pfx_checksum(const llama_tokens & tokens, const common_prompt_checkpoint & c) {
    uint64_t h = PFX_SEED;
    h = server_ckpt_hash(h, tokens.data(), tokens.size() * sizeof(llama_token));
    h = server_ckpt_hash(h, c.data_tgt.data(),  c.data_tgt.size());
    h = server_ckpt_hash(h, c.data_dft.data(),  c.data_dft.size());
    h = server_ckpt_hash(h, c.data_spec.data(), c.data_spec.size());
    return h;
}

static uint64_t pfx_file_bytes(int64_t n_tokens, const common_prompt_checkpoint & c) {
    return sizeof(pfx_header) + (uint64_t) n_tokens * sizeof(llama_token) + c.data_tgt.size() + c.data_dft.size() + c.data_spec.size();
}

static int64_t pfx_lcp(const llama_tokens & a, const llama_tokens & b, int64_t n_max) {
    const int64_t n = std::min<int64_t>(n_max, (int64_t) std::min(a.size(), b.size()));
    int64_t i = 0;
    while (i < n && a[i] == b[i]) {
        i++;
    }
    return i;
}

uint64_t server_prefix_ram_budget(int32_t mib) {
    if (mib == 0) {
        return 0;
    }
    if (mib == -1) {
        return UINT64_MAX;
    }
    if (mib > 0) {
        return (uint64_t) mib << 20;
    }

    // auto: a small part of the RAM, so that a machine with little RAM is not starved
    uint64_t phys = 0;
#ifdef _WIN32
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        phys = (uint64_t) st.ullTotalPhys;
    }
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long psize = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psize > 0) {
        phys = (uint64_t) pages * (uint64_t) psize;
    }
#endif
    const uint64_t cap = 4096ull << 20;

    return phys > 0 ? std::min(cap, phys / 16) : cap / 4;
}

int64_t server_prefix_pick_point(const std::vector<int64_t> & lcps, const std::vector<int64_t> & sizes, int64_t n_prompt,
                                 int64_t limit, int64_t min_tokens) {
    // at least one token of the prompt has to be processed after the point
    int64_t cap = n_prompt - 1;
    if (limit >= 0) {
        cap = std::min(cap, limit);
    }

    int64_t best = 0;
    for (size_t i = 0; i < lcps.size() && i < sizes.size(); i++) {
        if (lcps[i] < sizes[i] && lcps[i] <= cap) {
            best = std::max(best, lcps[i]);
        }
    }

    return best >= min_tokens ? best : 0;
}

int64_t server_prefix_pick_victim(const std::vector<uint64_t> & hits, const std::vector<uint64_t> & last_use) {
    int64_t best = -1;
    for (size_t i = 0; i < hits.size() && i < last_use.size(); i++) {
        if (best < 0) {
            best = (int64_t) i;
            continue;
        }
        const bool cold_i = hits[i]    == 0;
        const bool cold_b = hits[best] == 0;
        if (cold_i != cold_b ? cold_i : last_use[i] < last_use[best]) {
            best = (int64_t) i;
        }
    }
    return best;
}

void server_prefix_limit_params(common_json & llama_params, const common_chat_templates * tmpls,
                                const common_chat_templates_inputs & inputs, const std::string & prompt) {
    size_t k = 0;
    while (k < inputs.messages.size() && inputs.messages[k].role != "assistant") {
        k++;
    }

    if (k == inputs.messages.size()) {
        llama_params["prefix_cache_limit"] = -1; // no assistant message yet: the first request of a conversation
        return;
    }
    if (k == 0) {
        return;
    }

    common_chat_templates_inputs sub = inputs;
    sub.messages.resize(k);
    sub.add_generation_prompt  = false;
    sub.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    sub.grammar.clear();
    sub.json_schema.clear();
    sub.render_only = true;

    std::string text;
    try {
        text = common_chat_templates_apply(tmpls, sub).prompt;
    } catch (const std::exception & e) {
        PFX_DBG("no render of the messages before the first assistant message: %s\n", e.what());
        return;
    }

    // The limit is a bound, not a cut: a template that changes the earlier messages later gives a render that is not the
    // start of the prompt, and its length is still close to the right place.
    const size_t off = std::min(text.size(), prompt.size());
    if (off > 0) {
        llama_params["prefix_cache_limit"] = off;
    }
}

int32_t server_prefix_limit_tokens(const common_json & data, const llama_vocab * vocab, const std::string & prompt) {
    if (!data.contains("prefix_cache_limit") || !data.at("prefix_cache_limit").is_number_integer()) {
        return 0;
    }

    const int64_t off = data.at("prefix_cache_limit").get<int64_t>();
    if (off < 0) {
        return -1;
    }
    if (off == 0) {
        return 0;
    }

    const llama_tokens head = common_tokenize(vocab, prompt.substr(0, std::min<size_t>((size_t) off, prompt.size())), true, true);

    return (int32_t) head.size();
}

struct server_prefix_cache::impl {
    struct entry {
        uint64_t     id   = 0;
        llama_tokens tokens;
        uint64_t     hash = 0;

        llama_pos pos_min = 0;
        llama_pos pos_max = 0;

        std::shared_ptr<const common_prompt_checkpoint> snap; // nullptr = only in the file
        uint64_t bytes = 0; // of the data

        std::string path;           // empty = no file
        uint64_t    file_bytes = 0;
        bool        writing    = false;

        uint64_t hits     = 0;
        uint64_t last_use = 0;
    };

    struct job {
        uint64_t     id = 0;
        std::string  path;
        llama_tokens tokens;
        uint64_t     hash = 0;
        std::shared_ptr<const common_prompt_checkpoint> snap;
        uint64_t     file_bytes = 0;
    };

    bool     on         = false;
    uint64_t ram_bytes  = 0;
    int32_t  min_tok    = 0;
    std::string dir;
    uint64_t disk_bytes = 0;
    uint64_t model_key  = 0;

    mutable std::mutex      mutex;
    std::condition_variable cv;
    std::thread             writer;
    bool                    stop = false;
    bool                    busy = false;

    std::list<entry> entries;
    std::deque<job>  queue;

    uint64_t next_id       = 1;
    uint64_t clock         = 0;
    uint64_t used_ram      = 0;
    uint64_t used_disk     = 0;
    uint64_t pending_disk  = 0;
    double   read_bps      = 0.0;
    double   write_bps     = 0.0;
    double   restore_bps   = 0.0;
    int      n_failed      = 0;

    bool disk_on() const {
        return !dir.empty();
    }

    std::string file_name(int64_t n_tokens, uint64_t hash) const {
        char name[128];
        snprintf(name, sizeof(name), "%s%016" PRIx64 "-%" PRId64 "-%016" PRIx64 ".bin", PFX_PREFIX, model_key, n_tokens, hash);
        return (fs::path(dir) / name).string();
    }

    std::string key_prefix() const {
        char name[64];
        snprintf(name, sizeof(name), "%s%016" PRIx64 "-", PFX_PREFIX, model_key);
        return name;
    }

    // the caller holds the mutex
    std::list<entry>::iterator find_id(uint64_t id) {
        return std::find_if(entries.begin(), entries.end(), [&](const entry & e) { return e.id == id; });
    }

    // the caller holds the mutex. An entry that has neither data nor a file goes.
    std::list<entry>::iterator gc(std::list<entry>::iterator it) {
        if (!it->snap && it->path.empty() && !it->writing) {
            return entries.erase(it);
        }
        return std::next(it);
    }

    // the caller holds the mutex
    void drop_ram(entry & e) {
        if (e.snap) {
            used_ram -= std::min(used_ram, e.bytes);
            e.snap.reset();
        }
    }

    // the caller holds the mutex
    void drop_file(entry & e) {
        if (!e.path.empty()) {
            std::remove(e.path.c_str());
            used_disk -= std::min(used_disk, e.file_bytes);
            e.path.clear();
            e.file_bytes = 0;
        }
    }

    // the caller holds the mutex. Drops data from RAM until need more bytes fit. keep is not dropped.
    bool make_room_ram(uint64_t need, uint64_t keep) {
        if (ram_bytes == UINT64_MAX) {
            return true;
        }
        if (need > ram_bytes) {
            return false;
        }
        while (used_ram + need > ram_bytes) {
            std::vector<std::list<entry>::iterator> cand;
            std::vector<uint64_t> hits;
            std::vector<uint64_t> last;
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                if (it->snap && it->id != keep) {
                    cand.push_back(it);
                    hits.push_back(it->hits);
                    last.push_back(it->last_use);
                }
            }
            const int64_t v = server_prefix_pick_victim(hits, last);
            if (v < 0) {
                return false;
            }
            PFX_DBG("dropping the snapshot of %zu tokens from RAM (%.1f MiB, %" PRIu64 " hits)\n",
                    cand[v]->tokens.size(), cand[v]->bytes / (1024.0 * 1024.0), cand[v]->hits);
            drop_ram(*cand[v]);
            gc(cand[v]);
        }
        return true;
    }

    // the caller holds the mutex. Drops the files that were used the longest ago until need more bytes fit.
    bool make_room_disk(uint64_t need) {
        if (need > disk_bytes) {
            return false;
        }
        while (used_disk + pending_disk + need > disk_bytes) {
            auto v = entries.end();
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                if (!it->path.empty() && (v == entries.end() || it->last_use < v->last_use)) {
                    v = it;
                }
            }
            if (v == entries.end()) {
                return false;
            }
            drop_file(*v);
            gc(v);
        }
        return true;
    }

    // the time of a file is its last use, so that a later server keeps the ones that were used recently
    static void touch(const std::string & path) {
        std::error_code ec;
        fs::last_write_time(path, fs::file_time_type::clock::now(), ec);
    }

    // reads the header and the tokens of a file, and checks that the size adds up
    static bool read_head(FILE * f, uint64_t file_size, uint64_t key, pfx_header & hdr, llama_tokens & tokens) {
        if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
            return false;
        }
        if (hdr.magic != PFX_MAGIC || hdr.version != PFX_VERSION || hdr.model_key != key || hdr.n_tokens <= 0 ||
            hdr.n_tokens > (int64_t) (file_size / sizeof(llama_token))) {
            return false;
        }
        const uint64_t size_tok = (uint64_t) hdr.n_tokens * sizeof(llama_token);
        if (hdr.size_tgt > file_size || hdr.size_dft > file_size || hdr.size_spec > file_size ||
            sizeof(hdr) + size_tok + hdr.size_tgt + hdr.size_dft + hdr.size_spec != file_size) {
            return false;
        }
        tokens.resize((size_t) hdr.n_tokens);
        if (fread(tokens.data(), 1, size_tok, f) != size_tok) {
            return false;
        }
        return server_ckpt_prefix_hash(tokens, tokens.size()) == hdr.prefix_hash;
    }

    // indexes the files of this model key, removes the temporary files of dead servers, leaves the rest alone
    void scan() {
        const std::string pre = key_prefix();

        struct found {
            fs::file_time_type mtime;
            entry e;
        };
        std::vector<found> files;

        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.compare(0, pre.size(), pre) != 0) {
                continue;
            }
            const std::string path = it->path().string();

            // <name>.bin.<pid>.tmp
            const size_t n = name.size();
            if (n > 4 && name.compare(n - 4, 4, ".tmp") == 0) {
                const size_t dot = name.rfind('.', n - 5);
                const long pid = dot == std::string::npos ? 0 : strtol(name.c_str() + dot + 1, nullptr, 10);
                if (pfx_pid_dead(pid)) {
                    std::error_code ec2;
                    fs::remove(it->path(), ec2);
                }
                continue;
            }
            if (n < 4 || name.compare(n - 4, 4, ".bin") != 0) {
                continue;
            }

            std::error_code ec2;
            const uint64_t size  = (uint64_t) fs::file_size(it->path(), ec2);
            const auto     mtime = fs::last_write_time(it->path(), ec2);
            if (ec2) {
                continue;
            }

            FILE * f = ggml_fopen(path.c_str(), "rb");
            if (f == nullptr) {
                continue;
            }
            pfx_header hdr = {};
            entry e;
            const bool ok = read_head(f, size, model_key, hdr, e.tokens);
            fclose(f);

            if (!ok || name != fs::path(file_name(hdr.n_tokens, hdr.prefix_hash)).filename().string()) {
                PFX_WRN("bad file %s, removed\n", path.c_str());
                std::remove(path.c_str());
                continue;
            }

            e.hash       = hdr.prefix_hash;
            e.pos_min    = hdr.pos_min;
            e.pos_max    = hdr.pos_max;
            e.bytes      = hdr.size_tgt + hdr.size_dft + hdr.size_spec;
            e.path       = path;
            e.file_bytes = size;
            files.push_back({ mtime, std::move(e) });
        }

        // the order of the last uses carries over from the file times
        std::sort(files.begin(), files.end(), [](const found & a, const found & b) { return a.mtime < b.mtime; });
        for (auto & f : files) {
            f.e.id       = next_id++;
            f.e.last_use = ++clock;
            used_disk   += f.e.file_bytes;
            entries.push_back(std::move(f.e));
        }

        make_room_disk(0);
    }

    void run() {
        for (;;) {
            job j;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stop || !queue.empty(); });
                if (queue.empty()) {
                    return; // stop, and nothing left to write
                }
                j = std::move(queue.front());
                queue.pop_front();
                busy = true;
            }

            const common_prompt_checkpoint & c = *j.snap;

            pfx_header hdr = {};
            hdr.magic       = PFX_MAGIC;
            hdr.version     = PFX_VERSION;
            hdr.model_key   = model_key;
            hdr.prefix_hash = j.hash;
            hdr.n_tokens    = (int64_t) j.tokens.size();
            hdr.pos_min     = c.pos_min;
            hdr.pos_max     = c.pos_max;
            hdr.size_tgt    = c.data_tgt.size();
            hdr.size_dft    = c.data_dft.size();
            hdr.size_spec   = c.data_spec.size();
            hdr.checksum    = pfx_checksum(j.tokens, c);

            // the file appears under its name only when it is complete. Another server of the same model can write the same
            // file at the same time, so the temporary name has the pid.
            const std::string tmp = j.path + "." + std::to_string(pfx_pid()) + ".tmp";

            const int64_t t_start = ggml_time_us();

            bool ok = false;
            FILE * f = ggml_fopen(tmp.c_str(), "wb");
            if (f != nullptr) {
                ok = fwrite(&hdr, sizeof(hdr), 1, f) == 1;
                ok = ok && fwrite(j.tokens.data(), sizeof(llama_token), j.tokens.size(), f) == j.tokens.size();
                const std::vector<uint8_t> * blocks[3] = { &c.data_tgt, &c.data_dft, &c.data_spec };
                for (const auto * block : blocks) {
                    ok = ok && (block->empty() || fwrite(block->data(), 1, block->size(), f) == block->size());
                }
                ok = (fclose(f) == 0) && ok;
            }
            if (ok) {
                std::remove(j.path.c_str()); // rename does not replace a file on Windows
                ok = std::rename(tmp.c_str(), j.path.c_str()) == 0;
            }
            if (!ok) {
                std::remove(tmp.c_str());
            }
            const double t_sec = (ggml_time_us() - t_start) / 1e6;

            bool warn = false;
            {
                std::lock_guard<std::mutex> lock(mutex);

                busy = false;
                pending_disk -= std::min(pending_disk, j.file_bytes);

                auto it = find_id(j.id);
                if (it == entries.end()) {
                    // dropped while it was written (a restore from it failed)
                    if (ok) {
                        std::remove(j.path.c_str());
                    }
                } else {
                    it->writing = false;
                    if (ok) {
                        it->path       = j.path;
                        it->file_bytes = j.file_bytes;
                        used_disk     += j.file_bytes;
                        if (t_sec > 0.0) {
                            const double bps = (double) j.file_bytes / t_sec;
                            write_bps = write_bps > 0.0 ? 0.7 * write_bps + 0.3 * bps : bps;
                        }
                    } else {
                        warn = n_failed++ < 3;
                    }
                    gc(it);
                }
            }

            if (warn) {
                PFX_WRN("failed to write %s (disk full, or the directory is gone?)\n", j.path.c_str());
            } else if (ok) {
                PFX_DBG("wrote %s (%.1f MiB, %.0f ms)\n", j.path.c_str(), j.file_bytes / (1024.0 * 1024.0), t_sec * 1e3);
            }
        }
    }
};

server_prefix_cache::server_prefix_cache() : d(new impl) {
}

server_prefix_cache::~server_prefix_cache() {
    shutdown();
}

bool server_prefix_cache::init(uint64_t ram_bytes, int32_t min_tokens, const std::string & dir, uint64_t disk_bytes, uint64_t model_key) {
    // the server makes its context again (POST /tune/context, VRAM lease): the snapshots stay valid for the same setup
    if (d->on && d->ram_bytes == ram_bytes && d->min_tok == std::max(1, min_tokens) && d->model_key == model_key &&
        d->disk_bytes == disk_bytes && d->dir == dir) {
        return true;
    }

    shutdown();

    if (ram_bytes == 0) {
        return false;
    }

    d->ram_bytes  = ram_bytes;
    d->min_tok    = std::max(1, min_tokens);
    d->model_key  = model_key;
    d->disk_bytes = disk_bytes;
    d->dir.clear();

    if (!dir.empty() && disk_bytes > 0) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        bool ok = !ec && fs::is_directory(dir, ec);

        if (ok) {
            // a probe file shows now that the directory is writable
            const std::string probe = (fs::path(dir) / (std::string(PFX_PREFIX) + "probe-" + std::to_string(pfx_pid()) + ".tmp")).string();
            FILE * f = ggml_fopen(probe.c_str(), "wb");
            ok = f != nullptr && fclose(f) == 0;
            std::remove(probe.c_str());
        }

        if (ok) {
            d->dir    = dir;
            d->scan();
            d->writer = std::thread(&impl::run, d.get());
        } else {
            PFX_WRN("cannot write to the directory '%s', the snapshots stay in RAM only\n", dir.c_str());
        }
    }

    d->on = true;

    const auto st = stats();
    if (ram_bytes == UINT64_MAX) {
        PFX_INF("on, no RAM limit, snapshots of shared prefixes of at least %d tokens\n", d->min_tok);
    } else {
        PFX_INF("on, at most %.0f MiB of RAM, snapshots of shared prefixes of at least %d tokens\n", ram_bytes / (1024.0 * 1024.0), d->min_tok);
    }
    if (d->disk_on()) {
        PFX_INF("files in '%s', at most %.0f MiB, %zu files (%.1f MiB) of this model found\n",
                d->dir.c_str(), disk_bytes / (1024.0 * 1024.0), st.n_disk, st.bytes_disk / (1024.0 * 1024.0));
    }

    return true;
}

void server_prefix_cache::shutdown() {
    {
        std::lock_guard<std::mutex> lock(d->mutex);
        d->stop = true;
    }
    d->cv.notify_all();
    if (d->writer.joinable()) {
        d->writer.join();
    }

    std::lock_guard<std::mutex> lock(d->mutex);
    d->entries.clear();
    d->queue.clear();
    d->used_ram     = 0;
    d->used_disk    = 0;
    d->pending_disk = 0;
    d->busy         = false;
    d->stop         = false;
    d->on           = false;
}

bool server_prefix_cache::enabled() const {
    return d->on;
}

int32_t server_prefix_cache::min_tokens() const {
    return d->min_tok;
}

int64_t server_prefix_cache::lcp_differs(const llama_tokens & tokens, int64_t n_max) const {
    std::lock_guard<std::mutex> lock(d->mutex);

    int64_t best = 0;
    for (const auto & e : d->entries) {
        const int64_t n = pfx_lcp(e.tokens, tokens, n_max);
        if (n < (int64_t) e.tokens.size()) {
            best = std::max(best, n);
        }
    }
    return best;
}

bool server_prefix_cache::has(const llama_tokens & tokens, int64_t n) const {
    std::lock_guard<std::mutex> lock(d->mutex);

    for (const auto & e : d->entries) {
        if ((int64_t) e.tokens.size() == n && pfx_lcp(e.tokens, tokens, n) == n) {
            return true;
        }
    }
    return false;
}

bool server_prefix_cache::find(const llama_tokens & tokens, int64_t n_min, int64_t n_max, server_prefix_match & out) const {
    if (!d->on) {
        return false;
    }

    std::lock_guard<std::mutex> lock(d->mutex);

    const impl::entry * best = nullptr;
    for (const auto & e : d->entries) {
        const int64_t n = (int64_t) e.tokens.size();
        if (n <= n_min || n > n_max || (!e.snap && e.path.empty())) {
            continue;
        }
        if (best != nullptr && n <= (int64_t) best->tokens.size()) {
            continue;
        }
        if (pfx_lcp(e.tokens, tokens, n) == n) {
            best = &e;
        }
    }

    if (best == nullptr) {
        return false;
    }

    out.id       = best->id;
    out.n_tokens = (int64_t) best->tokens.size();
    out.bytes    = best->snap ? best->bytes : best->file_bytes;
    out.in_ram   = (bool) best->snap;

    return true;
}

double server_prefix_cache::fetch_ms(const server_prefix_match & m) const {
    if (m.in_ram) {
        return 0.0;
    }

    std::lock_guard<std::mutex> lock(d->mutex);

    // writes are no faster than reads, so they stand in until a read is measured
    const double bps = d->read_bps > 0.0 ? d->read_bps : (d->write_bps > 0.0 ? d->write_bps : PFX_DEFAULT_BPS);

    return 5.0 + 1e3 * (double) m.bytes / bps;
}

double server_prefix_cache::restore_ms(uint64_t bytes) const {
    std::lock_guard<std::mutex> lock(d->mutex);

    const double bps = d->restore_bps > 0.0 ? d->restore_bps : PFX_DEFAULT_RESTORE_BPS;

    return 1.0 + 1e3 * (double) bytes / bps;
}

void server_prefix_cache::note_restore(uint64_t bytes, double ms) {
    if (ms <= 0.0) {
        return;
    }

    std::lock_guard<std::mutex> lock(d->mutex);

    const double bps = 1e3 * (double) bytes / ms;
    d->restore_bps = d->restore_bps > 0.0 ? 0.7 * d->restore_bps + 0.3 * bps : bps;
}

std::shared_ptr<const common_prompt_checkpoint> server_prefix_cache::fetch(const server_prefix_match & m, double & ms) {
    ms = 0.0;

    std::string  path;
    uint64_t     file_bytes = 0;
    int64_t      n_tokens   = 0;
    uint64_t     hash       = 0;
    {
        std::lock_guard<std::mutex> lock(d->mutex);

        auto it = d->find_id(m.id);
        if (it == d->entries.end()) {
            return nullptr;
        }

        it->hits++;
        it->last_use = ++d->clock;

        if (it->snap) {
            if (!it->path.empty()) {
                impl::touch(it->path);
            }
            return it->snap;
        }

        if (it->path.empty()) {
            return nullptr;
        }

        path       = it->path;
        file_bytes = it->file_bytes;
        n_tokens   = (int64_t) it->tokens.size();
        hash       = it->hash;
    }

    const int64_t t_start = ggml_time_us();

    auto snap = std::make_shared<common_prompt_checkpoint>();
    bool ok = false;

    FILE * f = ggml_fopen(path.c_str(), "rb");
    if (f != nullptr) {
        pfx_header   hdr = {};
        llama_tokens tokens;
        ok = impl::read_head(f, file_bytes, d->model_key, hdr, tokens) && hdr.n_tokens == n_tokens && hdr.prefix_hash == hash;

        if (ok) {
            try {
                snap->data_tgt.resize(hdr.size_tgt);
                snap->data_dft.resize(hdr.size_dft);
                snap->data_spec.resize(hdr.size_spec);
            } catch (const std::bad_alloc &) {
                PFX_WRN("no memory to read %" PRIu64 " bytes\n", file_bytes);
                fclose(f);
                return nullptr; // the file is fine, it stays
            }

            std::vector<uint8_t> * blocks[3] = { &snap->data_tgt, &snap->data_dft, &snap->data_spec };
            for (auto * block : blocks) {
                ok = ok && (block->empty() || fread(block->data(), 1, block->size(), f) == block->size());
            }
            ok = ok && pfx_checksum(tokens, *snap) == hdr.checksum;

            snap->n_tokens = hdr.n_tokens;
            snap->pos_min  = hdr.pos_min;
            snap->pos_max  = hdr.pos_max;
            snap->id_task  = -1;
        }
        fclose(f);
    }

    if (!ok) {
        PFX_WRN("bad or missing file %s, removed\n", path.c_str());
        drop(m);
        return nullptr;
    }

    const double t_sec = (ggml_time_us() - t_start) / 1e6;
    ms = t_sec * 1e3;

    std::lock_guard<std::mutex> lock(d->mutex);

    if (t_sec > 0.0) {
        const double bps = (double) file_bytes / t_sec;
        d->read_bps = d->read_bps > 0.0 ? 0.7 * d->read_bps + 0.3 * bps : bps;
    }

    impl::touch(path);

    // keep it in RAM too, if it fits
    auto it = d->find_id(m.id);
    if (it != d->entries.end() && !it->snap) {
        const uint64_t need = snap->size();
        if (need <= d->ram_bytes / 2 && d->make_room_ram(need, it->id)) {
            it->snap  = snap;
            it->bytes = need;
            d->used_ram += need;
        }
    }

    return snap;
}

bool server_prefix_cache::put(const llama_tokens & tokens, common_prompt_checkpoint && snap) {
    if (!d->on || snap.n_tokens <= 0 || snap.n_tokens > (int64_t) tokens.size() || snap.empty()) {
        return false;
    }

    const int64_t  n    = snap.n_tokens;
    const uint64_t need = snap.size();

    // one snapshot does not take the whole budget
    if (d->ram_bytes != UINT64_MAX && need > d->ram_bytes / 2) {
        PFX_WRN("the snapshot of %" PRId64 " tokens (%.1f MiB) is over half of the RAM budget, not kept\n", n, need / (1024.0 * 1024.0));
        return false;
    }

    if (has(tokens, n)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(d->mutex);

    if (!d->make_room_ram(need, 0)) {
        return false;
    }

    impl::entry e;
    e.id       = d->next_id++;
    e.tokens.assign(tokens.begin(), tokens.begin() + n);
    e.hash     = server_ckpt_prefix_hash(e.tokens, e.tokens.size());
    e.pos_min  = snap.pos_min;
    e.pos_max  = snap.pos_max;
    e.bytes    = need;
    e.snap     = std::make_shared<const common_prompt_checkpoint>(std::move(snap));
    e.last_use = ++d->clock;

    d->used_ram += need;

    if (d->disk_on()) {
        const uint64_t file_bytes = pfx_file_bytes(n, *e.snap);
        if ((int) d->queue.size() + (d->busy ? 1 : 0) < PFX_MAX_QUEUE && file_bytes <= d->disk_bytes / 2 && d->make_room_disk(file_bytes)) {
            impl::job j;
            j.id         = e.id;
            j.path       = d->file_name(n, e.hash);
            j.tokens     = e.tokens;
            j.hash       = e.hash;
            j.snap       = e.snap;
            j.file_bytes = file_bytes;

            e.writing        = true;
            d->pending_disk += file_bytes;
            d->queue.push_back(std::move(j));
            d->cv.notify_one();
        } else {
            PFX_DBG("the snapshot of %" PRId64 " tokens is not written (queue full or over the disk budget)\n", n);
        }
    }

    d->entries.push_back(std::move(e));

    return true;
}

void server_prefix_cache::drop(const server_prefix_match & m) {
    std::lock_guard<std::mutex> lock(d->mutex);

    auto it = d->find_id(m.id);
    if (it != d->entries.end()) {
        d->drop_ram(*it);
        d->drop_file(*it);
        d->entries.erase(it);
    }
}

server_prefix_stats server_prefix_cache::stats() const {
    std::lock_guard<std::mutex> lock(d->mutex);

    server_prefix_stats st;
    for (const auto & e : d->entries) {
        st.n_ram  += e.snap ? 1 : 0;
        st.n_disk += e.path.empty() ? 0 : 1;
    }
    st.bytes_ram  = d->used_ram;
    st.bytes_disk = d->used_disk;

    return st;
}
