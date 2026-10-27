#include "llama-engram.h"

#include "ggml-backend.h"
#include "ggml-cpuset.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

struct llama_engram_reader::job {
    const ggml_tensor * table;
    std::vector<int32_t> ids;
    std::vector<float> rows;
    std::atomic<size_t> next{0};
    std::mutex mutex;
    std::condition_variable done;
    int pending = 0;
};

struct llama_engram_reader::impl {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<job_ptr> queue;
    std::vector<std::thread> workers;
    bool stop = false;

    explicit impl(int n_threads) {
        GGML_ASSERT(n_threads > 0);
        try {
            for (int i = 0; i < n_threads; ++i) {
                workers.emplace_back([this] { run(); });
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~impl() { shutdown(); }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        ready.notify_all();
        for (auto & worker : workers) {
            worker.join();
        }
    }

    void run() {
#if defined(__linux__)
        // A context can be created by an already pinned compute thread. Use the helper CPU limit instead of inheriting that one CPU.
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (ggml_cpuset_allowed(cpu)) {
                CPU_SET(cpu, &cpus);
            }
        }
        if (CPU_COUNT(&cpus) > 0) {
            pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
        }
        pthread_setname_np(pthread_self(), "engram-reader");
#endif
        for (;;) {
            job_ptr work;
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [&] { return stop || !queue.empty(); });
                if (queue.empty()) {
                    return;
                }
                work = std::move(queue.front());
                queue.pop_front();
            }
            const auto * table = work->table;
            const auto to_float = ggml_get_type_traits(table->type)->to_float;
            size_t i;
            while ((i = work->next.fetch_add(1, std::memory_order_relaxed)) < work->ids.size()) {
                const char * src = (const char *) table->data + work->ids[i]*table->nb[1];
                float * dst = work->rows.data() + i*table->ne[0];
                if (table->type == GGML_TYPE_F32) {
                    memcpy(dst, src, table->ne[0]*sizeof(float));
                } else {
                    to_float(src, dst, table->ne[0]);
                }
            }
            {
                std::lock_guard<std::mutex> lock(work->mutex);
                if (--work->pending == 0) {
                    work->done.notify_all();
                }
            }
        }
    }
};

llama_engram_reader::llama_engram_reader(int n_threads) : pimpl(new impl(n_threads)) {}
llama_engram_reader::~llama_engram_reader() = default;

bool llama_engram_reader::supports(const ggml_tensor * table) {
    return table && table->buffer && table->data && ggml_backend_buffer_is_host(table->buffer) &&
        table->ne[2] == 1 && table->ne[3] == 1 && table->nb[0] == ggml_type_size(table->type) &&
        (table->type == GGML_TYPE_F32 || ggml_get_type_traits(table->type)->to_float != nullptr);
}

llama_engram_reader::job_ptr llama_engram_reader::submit(const ggml_tensor * table, std::vector<int32_t> ids) {
    GGML_ASSERT(supports(table));
    for (int32_t id : ids) {
        GGML_ASSERT(id >= 0 && id < table->ne[1]);
    }
    auto work = std::make_shared<job>();
    work->table = table;
    work->ids = std::move(ids);
    work->rows.resize(work->ids.size()*table->ne[0]);
    work->pending = (int) pimpl->workers.size();
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        for (size_t i = 0; i < pimpl->workers.size(); ++i) {
            pimpl->queue.push_back(work);
        }
    }
    pimpl->ready.notify_all();
    return work;
}

void llama_engram_reader::collect(const job_ptr & work, float * dst, size_t n_bytes) {
    GGML_ASSERT(work && n_bytes == work->rows.size()*sizeof(float));
    std::unique_lock<std::mutex> lock(work->mutex);
    work->done.wait(lock, [&] { return work->pending == 0; });
    memcpy(dst, work->rows.data(), n_bytes);
}
