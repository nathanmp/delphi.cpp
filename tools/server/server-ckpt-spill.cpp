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
#include <filesystem>
#include <list>
#include <map>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>

#ifdef _WIN32
#include <process.h>
#else
#include <errno.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

#define SPILL_INF(fmt, ...) LOG_INF("ckpt spill: " fmt, __VA_ARGS__)
#define SPILL_WRN(fmt, ...) LOG_WRN("ckpt spill: " fmt, __VA_ARGS__)

// File format, version 1. All fields in the byte order of the machine: the files are never moved to another one.
//   spill_header, then the data of the target context, of the draft context and of the speculative state, in this order.
// The version changes when the header or the meaning of a field changes.
static constexpr uint32_t SPILL_MAGIC   = 0x504b4344; // "DCKP"
static constexpr uint32_t SPILL_VERSION = 1;

struct spill_header {
    uint32_t magic;
    uint32_t version;
    uint64_t model_key;   // model files and context setup
    uint64_t prefix_hash; // of the tokens before the checkpoint
    int64_t  n_tokens;
    int32_t  pos_min;
    int32_t  pos_max;
    int32_t  slot_id;
    int32_t  reserved;    // zero
    uint64_t size_tgt;
    uint64_t size_dft;
    uint64_t size_spec;
    uint64_t checksum;    // of the three data blocks
};

static_assert(sizeof(spill_header) == 80, "the header has padding");

// checkpoints being written or waiting to be written, each holds its data in memory
static constexpr int SPILL_MAX_QUEUE = 2;

// read speed in bytes per second that a read is planned with before any speed is measured
static constexpr double SPILL_DEFAULT_BPS = 100e6;

static const char * const SPILL_PREFIX = "dckpt-";

static constexpr uint64_t SPILL_SEED = 0xcbf29ce484222325ULL;

// FNV-1a over words of 8 bytes, and over single bytes for the rest. It finds a damaged file, it is no protection against an attack.
static uint64_t spill_hash(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;

    while (n >= 8) {
        uint64_t w;
        memcpy(&w, p, 8);
        h = (h ^ w) * 0x100000001b3ULL;
        h ^= h >> 29;
        p += 8;
        n -= 8;
    }
    while (n > 0) {
        h = (h ^ *p) * 0x100000001b3ULL;
        p++;
        n--;
    }

    return h;
}

static uint64_t spill_checksum(const common_prompt_checkpoint & c) {
    uint64_t h = SPILL_SEED;
    h = spill_hash(h, c.data_tgt.data(),  c.data_tgt.size());
    h = spill_hash(h, c.data_dft.data(),  c.data_dft.size());
    h = spill_hash(h, c.data_spec.data(), c.data_spec.size());
    return h;
}

static int64_t spill_pid() {
#ifdef _WIN32
    return (int64_t) _getpid();
#else
    return (int64_t) getpid();
#endif
}

uint64_t server_ckpt_hash(uint64_t h, const void * data, size_t n) {
    return spill_hash(h, data, n);
}

uint64_t server_ckpt_prefix_hash(const llama_tokens & tokens, size_t n) {
    n = std::min(n, tokens.size());

    const uint64_t n64 = n;
    const uint64_t h   = spill_hash(SPILL_SEED, &n64, sizeof(n64));

    return spill_hash(h, tokens.data(), n * sizeof(llama_token));
}

static void spill_model_info(std::string & s, const llama_model * model) {
    if (model == nullptr) {
        s += "none;";
        return;
    }

    char desc[256] = { 0 };
    llama_model_desc(model, desc, sizeof(desc));

    s += desc;
    s += ";";
    s += std::to_string(llama_model_n_params(model));
    s += ";";
    s += std::to_string(llama_model_size(model));
    s += ";";
}

uint64_t server_ckpt_model_key(const common_params & params, const llama_model * model_tgt, const llama_model * model_dft, uint32_t n_ctx_seq, uint32_t n_seq_max) {
    std::string s = params.model.path + ";";

    // the model file may have been replaced under the same name
    std::error_code ec;
    const auto size = fs::file_size(params.model.path, ec);
    if (!ec) {
        s += std::to_string((uint64_t) size) + ";";
    }
    const auto mtime = fs::last_write_time(params.model.path, ec);
    if (!ec) {
        s += std::to_string((int64_t) mtime.time_since_epoch().count()) + ";";
    }

    spill_model_info(s, model_tgt);
    spill_model_info(s, model_dft);

    s += std::to_string(n_ctx_seq) + ";";
    s += std::to_string(n_seq_max) + ";";
    s += std::to_string((int) params.cache_type_k) + ";";
    s += std::to_string((int) params.cache_type_v) + ";";
    s += std::to_string((int) params.swa_full) + ";";
    s += std::to_string((int) params.kv_unified) + ";";
    s += std::to_string((int) params.flash_attn_type) + ";";
    s += std::to_string(model_tgt ? llama_model_n_swa(model_tgt) : 0) + ";";

    return spill_hash(SPILL_SEED, s.data(), s.size());
}

static bool spill_write(const std::string & path, const spill_header & hdr, const common_prompt_checkpoint & c) {
    FILE * f = ggml_fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }

    bool ok = fwrite(&hdr, sizeof(hdr), 1, f) == 1;

    const std::vector<uint8_t> * blocks[3] = { &c.data_tgt, &c.data_dft, &c.data_spec };
    for (const auto * block : blocks) {
        ok = ok && (block->empty() || fwrite(block->data(), 1, block->size(), f) == block->size());
    }

    // fclose can fail on the last buffered write
    ok = (fclose(f) == 0) && ok;

    return ok;
}

// removes the files that a server left behind when it died. Their name has the pid of the server.
static void spill_remove_stale(const std::string & dir) {
#ifndef _WIN32
    const size_t n_prefix = strlen(SPILL_PREFIX);

    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.compare(0, n_prefix, SPILL_PREFIX) != 0) {
            continue;
        }

        char * rest = nullptr;
        const long pid = strtol(name.c_str() + n_prefix, &rest, 10);
        if (rest == name.c_str() + n_prefix || *rest != '-' || pid <= 0 || (int64_t) pid == spill_pid()) {
            continue;
        }

        if (kill((pid_t) pid, 0) == -1 && errno == ESRCH) {
            std::error_code ec2;
            fs::remove(it->path(), ec2);
        }
    }
#else
    (void) dir;
#endif
}

struct server_ckpt_spill::impl {
    struct file {
        server_ckpt_spill_entry e;
        std::string             path;
        uint64_t                last_use = 0;
    };

    struct job {
        server_ckpt_spill_entry  e;
        uint64_t                 gen = 0;
        common_prompt_checkpoint ckpt;
    };

    std::string dir;
    uint64_t    max_bytes = 0;
    uint64_t    model_key = 0;
    bool        on        = false;

    mutable std::mutex      mutex;
    std::condition_variable cv;
    std::thread             writer;
    bool                    stop = false;
    bool                    busy = false; // the writer holds a job

    std::list<file>         files;
    std::deque<job>         queue;
    std::map<int, uint64_t> gen;           // per slot, changed by drop_slot() so that a write in progress is not kept
    uint64_t                bytes_disk    = 0;
    uint64_t                bytes_pending = 0; // size of the files of the jobs
    uint64_t                clock         = 0;
    double                  read_bps      = 0.0;
    double                  write_bps     = 0.0;
    int                     n_failed      = 0;

    std::string file_path(const server_ckpt_spill_entry & e) const {
        char name[192];
        snprintf(name, sizeof(name), "%s%" PRId64 "-%08" PRIx64 "-s%d-%" PRId64 "-%016" PRIx64 ".bin",
                SPILL_PREFIX, spill_pid(), (uint64_t) (model_key & 0xffffffffULL), e.slot_id, e.n_tokens, e.prefix_hash);
        return (fs::path(dir) / name).string();
    }

    // the caller holds the mutex
    std::list<file>::iterator find(const server_ckpt_spill_entry & e) {
        return std::find_if(files.begin(), files.end(), [&](const file & f) {
            return f.e.slot_id == e.slot_id && f.e.n_tokens == e.n_tokens && f.e.prefix_hash == e.prefix_hash;
        });
    }

    // the caller holds the mutex
    std::list<file>::iterator drop_file(std::list<file>::iterator it) {
        std::remove(it->path.c_str());
        bytes_disk -= std::min(bytes_disk, it->e.bytes);
        return files.erase(it);
    }

    // the caller holds the mutex. Drops the files that were used the longest ago, until need more bytes fit.
    void make_room(uint64_t need) {
        while (!files.empty() && bytes_disk + bytes_pending + need > max_bytes) {
            drop_file(std::min_element(files.begin(), files.end(), [](const file & a, const file & b) {
                return a.last_use < b.last_use;
            }));
        }
    }

    void run() {
        for (;;) {
            job j;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stop || !queue.empty(); });
                if (stop) {
                    return;
                }
                j = std::move(queue.front());
                queue.pop_front();
                busy = true;
            }

            spill_header hdr = {};
            hdr.magic       = SPILL_MAGIC;
            hdr.version     = SPILL_VERSION;
            hdr.model_key   = model_key;
            hdr.prefix_hash = j.e.prefix_hash;
            hdr.n_tokens    = j.e.n_tokens;
            hdr.pos_min     = j.e.pos_min;
            hdr.pos_max     = j.e.pos_max;
            hdr.slot_id     = j.e.slot_id;
            hdr.size_tgt    = j.ckpt.data_tgt.size();
            hdr.size_dft    = j.ckpt.data_dft.size();
            hdr.size_spec   = j.ckpt.data_spec.size();
            hdr.checksum    = spill_checksum(j.ckpt);

            // the file appears under its name only when it is complete
            const std::string path = file_path(j.e);
            const std::string tmp  = path + ".tmp";

            const int64_t t_start = ggml_time_us();
            bool ok = spill_write(tmp, hdr, j.ckpt);
            if (ok) {
                std::remove(path.c_str()); // rename does not replace a file on Windows
                ok = std::rename(tmp.c_str(), path.c_str()) == 0;
            }
            if (!ok) {
                std::remove(tmp.c_str());
            }
            const double t_sec = (ggml_time_us() - t_start) / 1e6;

            bool warn = false;
            {
                std::lock_guard<std::mutex> lock(mutex);

                busy = false;
                bytes_pending -= std::min(bytes_pending, j.e.bytes);

                if (!ok) {
                    warn = n_failed++ < 3;
                } else if (stop || gen[j.e.slot_id] != j.gen) {
                    // the server stops, or the slot dropped its prompt while the file was written
                    std::remove(path.c_str());
                } else {
                    auto old = find(j.e);
                    if (old != files.end()) {
                        drop_file(old);
                    }

                    files.push_back({ j.e, path, ++clock });
                    bytes_disk += j.e.bytes;

                    if (t_sec > 0.0) {
                        const double bps = (double) j.e.bytes / t_sec;
                        write_bps = write_bps > 0.0 ? 0.7 * write_bps + 0.3 * bps : bps;
                    }
                }
            }

            if (warn) {
                SPILL_WRN("failed to write %s (disk full, or the directory is gone?)\n", path.c_str());
            }
        }
    }
};

server_ckpt_spill::server_ckpt_spill() : d(new impl) {
}

server_ckpt_spill::~server_ckpt_spill() {
    shutdown();
}

bool server_ckpt_spill::init(const std::string & dir, uint64_t max_bytes, uint64_t model_key) {
    shutdown();

    if (dir.empty() || max_bytes == 0) {
        return false;
    }

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) {
        SPILL_WRN("cannot use the directory '%s': %s\n", dir.c_str(), ec ? ec.message().c_str() : "not a directory");
        return false;
    }

    // a probe file shows now that the directory is writable
    const std::string probe = (fs::path(dir) / (std::string(SPILL_PREFIX) + std::to_string(spill_pid()) + "-probe.tmp")).string();
    FILE * f = ggml_fopen(probe.c_str(), "wb");
    if (f == nullptr || fclose(f) != 0) {
        SPILL_WRN("cannot write to the directory '%s'\n", dir.c_str());
        return false;
    }
    std::remove(probe.c_str());

    spill_remove_stale(dir);

    d->dir       = dir;
    d->max_bytes = max_bytes;
    d->model_key = model_key;
    d->on        = true;
    d->writer    = std::thread(&impl::run, d.get());

    SPILL_INF("dropped context checkpoints go to '%s', at most %.0f MiB\n", dir.c_str(), max_bytes / (1024.0 * 1024.0));

    return true;
}

void server_ckpt_spill::shutdown() {
    {
        std::lock_guard<std::mutex> lock(d->mutex);
        d->stop = true;
    }
    d->cv.notify_all();
    if (d->writer.joinable()) {
        d->writer.join();
    }

    std::lock_guard<std::mutex> lock(d->mutex);
    for (const auto & f : d->files) {
        std::remove(f.path.c_str());
    }
    d->files.clear();
    d->queue.clear();
    d->gen.clear();
    d->bytes_disk    = 0;
    d->bytes_pending = 0;
    d->busy          = false;
    d->stop          = false;
    d->on            = false;
}

bool server_ckpt_spill::enabled() const {
    return d->on;
}

bool server_ckpt_spill::put(int slot_id, uint64_t prefix_hash, common_prompt_checkpoint && ckpt) {
    if (!d->on) {
        return false;
    }

    const uint64_t need = sizeof(spill_header) + ckpt.data_tgt.size() + ckpt.data_dft.size() + ckpt.data_spec.size();

    if (need > d->max_bytes) {
        return false; // a single file is over the bound, so there is no use in dropping the others
    }

    std::lock_guard<std::mutex> lock(d->mutex);

    if ((int) d->queue.size() + (d->busy ? 1 : 0) >= SPILL_MAX_QUEUE) {
        return false; // the disk is slower than the checkpoints come
    }

    d->make_room(need);
    if (d->bytes_disk + d->bytes_pending + need > d->max_bytes) {
        return false;
    }

    impl::job j;
    j.e.slot_id     = slot_id;
    j.e.n_tokens    = ckpt.n_tokens;
    j.e.pos_min     = ckpt.pos_min;
    j.e.pos_max     = ckpt.pos_max;
    j.e.prefix_hash = prefix_hash;
    j.e.bytes       = need;
    j.gen           = d->gen[slot_id];
    j.ckpt          = std::move(ckpt);

    d->bytes_pending += need;
    d->queue.push_back(std::move(j));
    d->cv.notify_one();

    return true;
}

std::vector<server_ckpt_spill_entry> server_ckpt_spill::list(int slot_id) const {
    std::vector<server_ckpt_spill_entry> res;

    std::lock_guard<std::mutex> lock(d->mutex);
    for (const auto & f : d->files) {
        if (f.e.slot_id == slot_id) {
            res.push_back(f.e);
        }
    }

    return res;
}

bool server_ckpt_spill::load(const server_ckpt_spill_entry & entry, common_prompt_checkpoint & out, double & ms) {
    std::string path;
    {
        std::lock_guard<std::mutex> lock(d->mutex);

        auto it = d->find(entry);
        if (it == d->files.end()) {
            return false;
        }
        path = it->path;
        it->last_use = ++d->clock;
    }

    const int64_t t_start = ggml_time_us();

    spill_header hdr = {};
    bool ok = false;

    FILE * f = ggml_fopen(path.c_str(), "rb");
    if (f != nullptr) {
        // the file has to be the one that the entry says: same key, same positions, and a size that adds up
        ok = fread(&hdr, sizeof(hdr), 1, f) == 1 &&
            hdr.magic == SPILL_MAGIC && hdr.version == SPILL_VERSION && hdr.model_key == d->model_key &&
            hdr.slot_id == entry.slot_id && hdr.n_tokens == entry.n_tokens && hdr.prefix_hash == entry.prefix_hash &&
            hdr.pos_min == entry.pos_min && hdr.pos_max == entry.pos_max &&
            hdr.size_tgt <= entry.bytes && hdr.size_dft <= entry.bytes && hdr.size_spec <= entry.bytes &&
            sizeof(hdr) + hdr.size_tgt + hdr.size_dft + hdr.size_spec == entry.bytes;

        if (ok) {
            try {
                out.data_tgt.resize(hdr.size_tgt);
                out.data_dft.resize(hdr.size_dft);
                out.data_spec.resize(hdr.size_spec);
            } catch (const std::bad_alloc &) {
                SPILL_WRN("no memory to read %" PRIu64 " bytes\n", entry.bytes);
                fclose(f);
                out.clear();
                return false; // the file is fine, it stays
            }

            std::vector<uint8_t> * blocks[3] = { &out.data_tgt, &out.data_dft, &out.data_spec };
            for (auto * block : blocks) {
                ok = ok && (block->empty() || fread(block->data(), 1, block->size(), f) == block->size());
            }
            ok = ok && spill_checksum(out) == hdr.checksum;
        }
        fclose(f);
    }

    if (!ok) {
        SPILL_WRN("bad file %s, removed\n", path.c_str());
        out.clear();
        erase(entry);
        return false;
    }

    out.n_tokens = hdr.n_tokens;
    out.pos_min  = hdr.pos_min;
    out.pos_max  = hdr.pos_max;
    out.id_task  = -1;

    const double t_sec = (ggml_time_us() - t_start) / 1e6;
    ms = t_sec * 1e3;

    if (t_sec > 0.0) {
        const double bps = (double) entry.bytes / t_sec;
        std::lock_guard<std::mutex> lock(d->mutex);
        d->read_bps = d->read_bps > 0.0 ? 0.7 * d->read_bps + 0.3 * bps : bps;
    }

    return true;
}

double server_ckpt_spill::read_ms(uint64_t bytes) const {
    std::lock_guard<std::mutex> lock(d->mutex);

    // writes are no faster than reads, so they stand in until a read is measured
    const double bps = d->read_bps > 0.0 ? d->read_bps : (d->write_bps > 0.0 ? d->write_bps : SPILL_DEFAULT_BPS);

    return 5.0 + 1e3 * (double) bytes / bps;
}

void server_ckpt_spill::erase(const server_ckpt_spill_entry & entry) {
    std::lock_guard<std::mutex> lock(d->mutex);

    auto it = d->find(entry);
    if (it != d->files.end()) {
        d->drop_file(it);
    }
}

void server_ckpt_spill::erase_beyond(int slot_id, llama_pos pos_max) {
    if (!d->on) {
        return;
    }

    std::lock_guard<std::mutex> lock(d->mutex);

    for (auto it = d->files.begin(); it != d->files.end();) {
        if (it->e.slot_id == slot_id && it->e.pos_max > pos_max) {
            it = d->drop_file(it);
        } else {
            ++it;
        }
    }
}

void server_ckpt_spill::drop_slot(int slot_id) {
    if (!d->on) {
        return;
    }

    std::lock_guard<std::mutex> lock(d->mutex);

    for (auto it = d->files.begin(); it != d->files.end();) {
        if (it->e.slot_id == slot_id) {
            it = d->drop_file(it);
        } else {
            ++it;
        }
    }

    // the writes that wait are not needed any more, the one in progress is dropped when it is done
    for (auto it = d->queue.begin(); it != d->queue.end();) {
        if (it->e.slot_id == slot_id) {
            d->bytes_pending -= std::min(d->bytes_pending, it->e.bytes);
            it = d->queue.erase(it);
        } else {
            ++it;
        }
    }
    d->gen[slot_id]++;
}

uint64_t server_ckpt_spill::bytes() const {
    std::lock_guard<std::mutex> lock(d->mutex);
    return d->bytes_disk;
}

size_t server_ckpt_spill::size() const {
    std::lock_guard<std::mutex> lock(d->mutex);
    return d->files.size();
}
