#pragma once

#include "ggml.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

// Internal copy/pack workers shared by whole-weight staging and compact expert staging.
using ggml_backend_stage_pack_host_t = void (*)(const ggml_tensor *, void *, const void *, size_t, int, int);

struct ggml_backend_sched_stage_copier {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    uint64_t generation = 0;
    int pending = 0;
    bool stop = false;
    char * dst = nullptr;
    const char * src = nullptr;
    size_t size = 0;
    int n = 1;
    ggml_backend_stage_pack_host_t xform = nullptr; // pack instead of copy (see pack)
    const struct ggml_tensor * xform_weight = nullptr;

    void part(int i) {
        if (xform != nullptr) {
            xform(xform_weight, dst, src, size, i, n);
            return;
        }
        const size_t begin = (size/n*i) & ~(size_t) 63;
        const size_t end   = i == n - 1 ? size : (size/n*(i + 1)) & ~(size_t) 63;
        if (end > begin) {
            memcpy(dst + begin, src + begin, end - begin);
        }
    }

    void start(int n_threads, const std::vector<int> & cpus, void (*pin)(const std::vector<int> &)) {
        n = std::max(1, n_threads);
        for (int i = 1; i < n; i++) {
            threads.emplace_back([this, i, cpus, pin] {
                pin(cpus);
                uint64_t seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv_start.wait(lock, [&] { return stop || generation != seen; });
                        if (stop) {
                            return;
                        }
                        seen = generation;
                    }
                    part(i);
                    std::lock_guard<std::mutex> lock(mutex);
                    if (--pending == 0) {
                        cv_done.notify_one();
                    }
                }
            });
        }
    }

    void copy(void * d, const void * s, size_t sz) {
        run(d, s, sz, nullptr, nullptr);
    }

    // the threads convert whole experts of weight w into the layout of its device while they copy
    void pack(void * d, const void * s, size_t sz, ggml_backend_stage_pack_host_t fn, const struct ggml_tensor * w) {
        run(d, s, sz, fn, w);
    }

    void run(void * d, const void * s, size_t sz, ggml_backend_stage_pack_host_t fn, const struct ggml_tensor * w) {
        if (n == 1 || sz < (4u << 20)) {
            if (fn != nullptr) {
                fn(w, d, s, sz, 0, 1);
            } else {
                memcpy(d, s, sz);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            dst = (char *) d;
            src = (const char *) s;
            size = sz;
            xform = fn;
            xform_weight = w;
            pending = n - 1;
            generation++;
        }
        cv_start.notify_all();
        part(0);
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait(lock, [&] { return pending == 0; });
    }

    ~ggml_backend_sched_stage_copier() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv_start.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }
};

