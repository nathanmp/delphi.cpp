#include "server-vram.h"

#include "json.h"
#include "log.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#define VRAM_INF(fmt, ...) LOG_INF("vram: " fmt, __VA_ARGS__)
#define VRAM_WRN(fmt, ...) LOG_WRN("vram: " fmt, __VA_ARGS__)

static int64_t vram_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double vram_env(const char * name, double def) {
    const char * e = getenv(name);
    return e && *e ? atof(e) : def;
}

struct server_vram::impl {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;
    std::vector<std::string> gpus;
    std::string home;
    std::function<void()> wake_loop;
    std::function<void()> save_slots;

    // daemon connection; the reader thread only fills the inbox
    int fd = -1;
    std::thread reader;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<common_json> inbox;
    bool closed = false;

    struct lease {
        std::string gpu, purpose;
        size_t bytes = 0;
        int tier = -1; // expert cache tier of a cache lease
    };
    std::map<int64_t, lease> leases;
    struct request {
        bool done = false, waiting = false;
        size_t granted = 0;
    };
    std::map<int64_t, request> requests;
    int64_t next_req = 1;
    bool welcomed = false;

    phase_t phase = PHASE_IDLE;
    int64_t idle_since_ms = 0;
    bool compute_released = false, kv_released = false, shrunk = false;
    std::map<std::string, size_t> scratch_bytes, kv_bytes; // what the released buffers took, to lease back
    double idle_compute_s = -1, idle_kv_s = -1, idle_shrink_s = -1;
    size_t chunk = 0, want = 0, stage_need = 0, min_lease = 0;
    int max_tiers_per_gpu = 2;

    bool connected() const { return fd >= 0 && !closed; }

    ~impl() { disconnect(); }

    void disconnect() {
        if (fd >= 0) {
#ifndef _WIN32
            shutdown(fd, SHUT_RDWR);
#endif
        }
        if (reader.joinable()) {
            reader.join();
        }
        if (fd >= 0) {
#ifndef _WIN32
            close(fd);
#endif
            fd = -1;
        }
    }

    bool connect_to(const std::string & path) {
#ifdef _WIN32
        GGML_UNUSED(path);
        return false;
#else
        fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            return false;
        }
        sockaddr_un addr = {};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
        if (connect(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
            close(fd);
            fd = -1;
            return false;
        }
        reader = std::thread([this] {
            std::string buf;
            char chunk_buf[4096];
            for (;;) {
                const ssize_t n = recv(fd, chunk_buf, sizeof(chunk_buf), 0);
                if (n <= 0) {
                    break;
                }
                buf.append(chunk_buf, n);
                size_t pos;
                while ((pos = buf.find('\n')) != std::string::npos) {
                    const std::string line = buf.substr(0, pos);
                    buf.erase(0, pos + 1);
                    common_json j = common_json::parse_no_throw(line);
                    if (j.is_discarded() || !j.is_object()) {
                        continue;
                    }
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        inbox.push_back(std::move(j));
                    }
                    cv.notify_all();
                    if (wake_loop) {
                        wake_loop();
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                closed = true;
            }
            cv.notify_all();
            if (wake_loop) {
                wake_loop();
            }
        });
        return true;
#endif
    }

    void send(const common_json & j) {
        if (!connected()) {
            return;
        }
#ifndef _WIN32
        const std::string line = j.dump() + "\n";
        size_t off = 0;
        while (off < line.size()) {
            const ssize_t n = ::send(fd, line.data() + off, line.size() - off, MSG_NOSIGNAL);
            if (n <= 0) {
                VRAM_WRN("%s", "lost the daemon connection; keeping the current memory\n");
                std::lock_guard<std::mutex> lock(mutex);
                closed = true;
                return;
            }
            off += n;
        }
#endif
    }

    bool has_tier_for_staging(const std::string & gpu) const {
        int32_t ids[16];
        size_t bytes[16];
        const char * devs[16];
        const int n = std::min(16, (int) llama_share_tiers(ctx_tgt, ids, bytes, devs, 16));
        for (int i = 0; i < n; ++i) {
            if (gpu == devs[i] && bytes[i] >= stage_need) {
                return true;
            }
        }
        return false;
    }

    int n_tiers(const std::string & gpu) const {
        int32_t ids[16];
        size_t bytes[16];
        const char * devs[16];
        const int n = std::min(16, (int) llama_share_tiers(ctx_tgt, ids, bytes, devs, 16));
        int k = 0;
        for (int i = 0; i < n; ++i) {
            k += gpu == devs[i];
        }
        return k;
    }

    size_t leased(const std::string & gpu, const std::string & purpose) const {
        size_t n = 0;
        for (const auto & [id, l] : leases) {
            if (l.gpu == gpu && l.purpose == purpose) {
                n += l.bytes;
            }
        }
        return n;
    }

    void handle(const common_json & j) {
        const std::string op = j.value("op", "");
        if (op == "lease") {
            const int64_t id = j.value<int64_t>("lease", -1);
            const int64_t req = j.value<int64_t>("req", -1);
            lease l;
            l.gpu = j.value("gpu", "");
            l.purpose = j.value("purpose", "");
            l.bytes = j.value<int64_t>("bytes", 0);
            if (l.purpose == "cache") {
                l.tier = llama_share_add_tier(ctx_tgt, l.gpu.c_str(), l.bytes);
                if (l.tier < 0) {
                    // the lease could not be used (allocation failed, or too many tiers)
                    common_json r = common_json::object();
                    r["op"] = "release";
                    r["lease"] = id;
                    send(r);
                    VRAM_WRN("lease %lld of %.1f MiB on %s not usable; released\n", (long long) id, l.bytes / 1048576.0, l.gpu.c_str());
                    return;
                }
                VRAM_INF("lease %lld: expert cache tier %d, %.1f MiB on %s\n", (long long) id, l.tier, l.bytes / 1048576.0, l.gpu.c_str());
            }
            if (requests.count(req)) {
                requests[req].granted += l.bytes;
            }
            leases[id] = l;
        } else if (op == "done") {
            const int64_t req = j.value<int64_t>("req", -1);
            requests[req].done = true;
        } else if (op == "wait") {
            const int64_t req = j.value<int64_t>("req", -1);
            requests[req].waiting = true;
        } else if (op == "welcome") {
            welcomed = true;
        } else if (op == "revoke") {
            const int64_t id = j.value<int64_t>("lease", -1);
            const int64_t t0 = vram_now_ms();
            auto it = leases.find(id);
            if (it != leases.end()) {
                const lease l = it->second;
                leases.erase(it);
                if (l.purpose == "cache" && l.tier >= 0) {
                    llama_share_drop_tier(ctx_tgt, l.tier);
                    if (!has_tier_for_staging(l.gpu)) {
                        llama_share_stage_allow(ctx_tgt, l.gpu.c_str(), false);
                    }
                }
                VRAM_INF("revoked lease %lld (%s, %.1f MiB on %s) in %lld ms\n", (long long) id, l.purpose.c_str(), l.bytes / 1048576.0,
                        l.gpu.c_str(), (long long) (vram_now_ms() - t0));
            }
            common_json r = common_json::object();
            r["op"] = "released";
            r["lease"] = id;
            send(r);
        }
    }

    void process() {
        for (;;) {
            common_json j;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (inbox.empty()) {
                    return;
                }
                j = std::move(inbox.front());
                inbox.pop_front();
            }
            handle(j);
        }
    }

    int64_t request_lease(const std::string & gpu, const std::string & purpose, size_t bytes, size_t min, bool wait, bool prefill) {
        const int64_t id = next_req++;
        requests[id] = {};
        common_json r = common_json::object();
        r["op"] = "request";
        r["req"] = id;
        r["gpu"] = gpu;
        r["purpose"] = purpose;
        r["bytes"] = (int64_t) bytes;
        r["min"] = (int64_t) min;
        r["chunk"] = (int64_t) std::max(chunk, min);
        // each tier adds a partial expert pass per layer: few, large tiers
        r["max_leases"] = (int64_t) std::max(1, max_tiers_per_gpu - n_tiers(gpu));
        r["wait"] = wait;
        r["prefill"] = prefill;
        send(r);
        return id;
    }

    // until every request is done (or, unless until_done, has an answer), the timeout passes, or the daemon is gone
    void wait_for(const std::vector<int64_t> & reqs, double timeout_s, bool until_done) {
        const int64_t deadline = vram_now_ms() + int64_t(timeout_s * 1000);
        for (;;) {
            process();
            bool ok = true;
            for (int64_t r : reqs) {
                const auto & q = requests[r];
                ok = ok && (q.done || (!until_done && q.waiting));
            }
            if (ok || !connected() || vram_now_ms() >= deadline) {
                if (!ok && connected()) {
                    VRAM_WRN("%s", "lease request timed out; continuing with the memory held\n");
                }
                for (int64_t r : reqs) {
                    if (requests[r].done) {
                        requests.erase(r);
                    }
                }
                return;
            }
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return !inbox.empty() || closed; });
        }
    }

    void release_purpose(const std::string & purpose) {
        for (auto it = leases.begin(); it != leases.end();) {
            if (it->second.purpose == purpose) {
                common_json r = common_json::object();
                r["op"] = "release";
                r["lease"] = it->first;
                send(r);
                it = leases.erase(it);
            } else {
                ++it;
            }
        }
    }

    void send_phase(const char * state) {
        common_json r = common_json::object();
        r["op"] = "phase";
        r["state"] = state;
        send(r);
    }

    llama_share_usage usage(const std::string & gpu) const {
        llama_share_usage u = llama_share_device_usage(ctx_tgt, gpu.c_str());
        if (ctx_dft) {
            const llama_share_usage v = llama_share_device_usage(ctx_dft, gpu.c_str());
            // an MTP context shares the target's weights
            if (llama_get_model(ctx_dft) != llama_get_model(ctx_tgt)) {
                u.model += v.model;
            }
            u.context += v.context;
            u.compute += v.compute;
        }
        return u;
    }

    void wake() {
        process();
        if (kv_released) {
            std::vector<int64_t> reqs;
            for (const auto & [gpu, bytes] : kv_bytes) {
                if (bytes && connected()) {
                    reqs.push_back(request_lease(gpu, "kv", bytes, bytes, true, false));
                }
            }
            wait_for(reqs, 300, true);
            llama_share_restore_memory(ctx_tgt);
            if (ctx_dft) {
                llama_share_restore_memory(ctx_dft);
            }
            kv_released = false;
        }
        if (compute_released) {
            // the next decode reserves the buffers again
            std::vector<int64_t> reqs;
            for (const auto & [gpu, bytes] : scratch_bytes) {
                if (bytes && connected()) {
                    reqs.push_back(request_lease(gpu, "scratch", bytes, bytes, true, false));
                }
            }
            wait_for(reqs, 300, true);
            compute_released = false;
        }
    }

    void before_prefill() {
        wake();
        if (!connected() || !llama_share_has_expert_cache(ctx_tgt)) {
            return;
        }
        // The home GPU first: the daemon makes this wait while another server is in a prompt, and revokes the cache that
        // others hold there. The other GPUs only get what is free now (or later, applied at a batch boundary).
        std::vector<std::string> order = { home };
        for (const auto & g : gpus) {
            if (g != home) {
                order.push_back(g);
            }
        }
        std::vector<int64_t> others;
        for (const auto & g : order) {
            const size_t have = leased(g, "cache");
            if (have >= want) {
                continue;
            }
            const size_t min = has_tier_for_staging(g) ? min_lease : std::max(stage_need, min_lease);
            const int64_t r = request_lease(g, "cache", want - have, min, g == home, g == home);
            if (g == home) {
                wait_for({ r }, 600, true);
            } else {
                others.push_back(r);
            }
        }
        wait_for(others, 2, false);
        for (const auto & g : gpus) {
            llama_share_stage_allow(ctx_tgt, g.c_str(), has_tier_for_staging(g));
        }
    }

    void release_compute() {
        for (const auto & g : gpus) {
            scratch_bytes[g] = usage(g).compute;
        }
        size_t freed = llama_share_release_compute(ctx_tgt);
        if (ctx_dft) {
            freed += llama_share_release_compute(ctx_dft);
        }
        release_purpose("scratch");
        compute_released = true;
        VRAM_INF("idle: released %.1f MiB of compute buffers\n", freed / 1048576.0);
    }

    // a long idle: give back the cache on other servers' GPUs
    void shrink() {
        for (auto it = leases.begin(); it != leases.end();) {
            if (it->second.purpose == "cache" && it->second.gpu != home) {
                if (it->second.tier >= 0) {
                    llama_share_drop_tier(ctx_tgt, it->second.tier);
                }
                common_json r = common_json::object();
                r["op"] = "release";
                r["lease"] = it->first;
                send(r);
                VRAM_INF("idle: released lease %lld (%.1f MiB on %s)\n", (long long) it->first, it->second.bytes / 1048576.0, it->second.gpu.c_str());
                it = leases.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto & g : gpus) {
            llama_share_stage_allow(ctx_tgt, g.c_str(), has_tier_for_staging(g));
        }
        shrunk = true;
    }

    void release_kv() {
        save_slots();
        for (const auto & g : gpus) {
            kv_bytes[g] = usage(g).context;
        }
        size_t freed = llama_share_release_memory(ctx_tgt);
        if (ctx_dft) {
            freed += llama_share_release_memory(ctx_dft);
        }
        release_purpose("kv");
        kv_released = true;
        VRAM_INF("idle: released %.1f MiB of KV cache\n", freed / 1048576.0);
    }
};

server_vram::server_vram() = default;
server_vram::~server_vram() = default;

void server_vram::free() {
    d.reset();
}

void server_vram::init(llama_context * ctx_tgt, llama_context * ctx_dft, const std::vector<ggml_backend_dev_t> & devices,
        const std::string & name, std::function<void()> wake_loop, std::function<void()> save_slots) {
    d = std::make_unique<impl>();
    d->ctx_tgt = ctx_tgt;
    d->ctx_dft = ctx_dft;
    d->wake_loop = std::move(wake_loop);
    d->save_slots = std::move(save_slots);
    // the list may end with a null device
    for (auto * dev : devices) {
        if (dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            d->gpus.push_back(ggml_backend_dev_name(dev));
        }
    }
    if (d->gpus.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                d->gpus.push_back(ggml_backend_dev_name(dev));
            }
        }
    }
    if (const char * h = llama_share_home_device(ctx_tgt)) {
        d->home = h;
    } else {
        for (const auto & g : d->gpus) {
            if (d->home.empty() && d->usage(g).model > 0) {
                d->home = g;
            }
        }
    }
    d->chunk = size_t(vram_env("LLAMA_VRAMD_CHUNK_MIB", 4096)) << 20;
    d->want = size_t(vram_env("LLAMA_VRAMD_WANT_MIB", 1 << 20)) << 20;
    d->stage_need = llama_share_stage_bytes(ctx_tgt);
    d->min_lease = size_t(vram_env("LLAMA_VRAMD_MIN_MIB", 1024)) << 20;
    d->max_tiers_per_gpu = std::max(1, (int) vram_env("LLAMA_VRAMD_MAX_TIERS_PER_GPU", 2));
    d->idle_since_ms = vram_now_ms();

    const char * path = getenv("LLAMA_VRAMD");
    if (path && *path) {
        if (!d->connect_to(path)) {
            VRAM_WRN("cannot connect to delphi-vramd at %s; static memory\n", path);
        } else {
            common_json hello = common_json::object();
            hello["op"] = "hello";
            hello["model"] = name;
            hello["pid"] = (int64_t) getpid();
            hello["home"] = d->home;
            d->send(hello);
            for (const auto & g : d->gpus) {
                const llama_share_usage u = d->usage(g);
                common_json m = common_json::object();
                m["op"] = "gpu";
                m["gpu"] = g;
                m["resident"] = (int64_t) (u.model + u.cache);
                m["scratch"] = (int64_t) u.compute;
                m["kv"] = (int64_t) u.context;
                d->send(m);
            }
            common_json ready = common_json::object();
            ready["op"] = "ready";
            d->send(ready);
            const int64_t deadline = vram_now_ms() + 10000;
            while (!d->welcomed && d->connected() && vram_now_ms() < deadline) {
                d->process();
                std::unique_lock<std::mutex> lock(d->mutex);
                d->cv.wait_for(lock, std::chrono::milliseconds(50), [&] { return !d->inbox.empty() || d->closed; });
            }
            if (!d->welcomed) {
                VRAM_WRN("no welcome from delphi-vramd at %s; static memory\n", path);
                d->disconnect();
            } else {
                VRAM_INF("registered with delphi-vramd at %s as %s (home %s, %zu GPUs, staging ring %.1f MiB)\n", path, name.c_str(),
                        d->home.c_str(), d->gpus.size(), d->stage_need / 1048576.0);
            }
        }
    }
    const bool daemon = d->connected();
    d->idle_compute_s = vram_env("LLAMA_SHARE_IDLE_COMPUTE_S", daemon ? 30 : -1);
    d->idle_kv_s      = vram_env("LLAMA_SHARE_IDLE_KV_S", daemon ? 120 : -1);
    d->idle_shrink_s  = vram_env("LLAMA_VRAMD_IDLE_SHRINK_S", 600);
    if (daemon) {
        // until a prompt leases them, staging uses only GPUs whose tiers can lend
        for (const auto & g : d->gpus) {
            llama_share_stage_allow(ctx_tgt, g.c_str(), d->has_tier_for_staging(g));
        }
    }
}

void server_vram::wake() {
    if (d) {
        d->wake();
    }
}

void server_vram::update(phase_t phase) {
    if (!d) {
        return;
    }
    d->process();
    if (phase == d->phase) {
        return;
    }
    if (phase == PHASE_PREFILL) {
        d->before_prefill();
        d->send_phase("prefill");
    } else if (phase == PHASE_DECODE) {
        d->send_phase("decode");
    } else {
        d->send_phase("idle");
        d->idle_since_ms = vram_now_ms();
        d->shrunk = false;
    }
    d->phase = phase;
}

void server_vram::idle_tick() {
    if (!d) {
        return;
    }
    d->process();
    if (d->phase != PHASE_IDLE) {
        return;
    }
    const double idle_s = (vram_now_ms() - d->idle_since_ms) / 1000.0;
    if (!d->compute_released && d->idle_compute_s >= 0 && idle_s >= d->idle_compute_s) {
        d->release_compute();
    }
    if (!d->kv_released && d->idle_kv_s >= 0 && idle_s >= d->idle_kv_s) {
        d->release_kv();
    }
    if (!d->shrunk && d->connected() && d->idle_shrink_s >= 0 && idle_s >= d->idle_shrink_s) {
        d->shrink();
    }
}
