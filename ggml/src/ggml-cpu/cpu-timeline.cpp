#include "cpu-timeline.h"

#include "ggml-impl.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace {

constexpr int k_max_threads = 256;

#pragma pack(push, 1)
struct call_rec  { int64_t t_call, t_ret; int32_t n_nodes, n_threads, name, pad; };
struct enter_rec { int32_t call, cpu; int64_t t_enter, t_exit; };
// experts / rows: distinct in-range expert ids and in-range (row, slot) selections of a MUL_MAT_ID (main thread only), else 0
struct node_rec  { int32_t call; int16_t node, n_fused; int32_t name; int16_t op, experts, rows, pad; int64_t t0, t1, t2; };
#pragma pack(pop)

struct thread_buf {
    enter_rec * enter = nullptr;
    node_rec  * nodes = nullptr;
    size_t n_enter = 0, n_nodes = 0;
};

struct timeline {
    std::string path;
    size_t cap_calls = 0, cap_nodes = 0;
    std::vector<call_rec> calls;
    thread_buf threads[k_max_threads];
    std::unordered_map<std::string, int32_t> names;
    std::vector<std::string> name_list;
    std::atomic<bool> busy{false};
    std::atomic<bool> full{false};
    bool recording = false, done = false;
    int64_t polls = 0;

    timeline() {
        const char * p = getenv("GGML_CPU_TIMELINE");
        if (!p || !*p) {
            done = true;
            return;
        }
        path = p;
        const char * c = getenv("GGML_CPU_TIMELINE_CALLS");
        const char * n = getenv("GGML_CPU_TIMELINE_NODES");
        cap_calls = c ? strtoull(c, nullptr, 10) : 400000;
        cap_nodes = n ? strtoull(n, nullptr, 10) : 1u << 21;
        calls.reserve(cap_calls);
    }

    bool exists(const char * suffix) const {
        return access((path + suffix).c_str(), F_OK) == 0;
    }

    int32_t intern(const char * name) {
        auto it = names.find(name);
        if (it != names.end()) {
            return it->second;
        }
        const int32_t id = (int32_t) name_list.size();
        name_list.emplace_back(name);
        names.emplace(name, id);
        return id;
    }

    void write() {
        FILE * f = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "cpu-timeline: cannot write %s\n", path.c_str());
            return;
        }
        const int32_t header[2] = { k_max_threads, (int32_t) calls.size() };
        fwrite("GCTL1\0\0\0", 1, 8, f);
        fwrite(header, sizeof(header), 1, f);
        fwrite(calls.data(), sizeof(call_rec), calls.size(), f);
        size_t total = 0;
        for (auto & t : threads) {
            const int64_t counts[2] = { (int64_t) t.n_enter, (int64_t) t.n_nodes };
            fwrite(counts, sizeof(counts), 1, f);
            if (t.n_enter) { fwrite(t.enter, sizeof(enter_rec), t.n_enter, f); }
            if (t.n_nodes) { fwrite(t.nodes, sizeof(node_rec), t.n_nodes, f); }
            total += t.n_nodes;
        }
        const int32_t n_names = (int32_t) name_list.size();
        fwrite(&n_names, sizeof(n_names), 1, f);
        for (const auto & s : name_list) {
            const int32_t len = (int32_t) s.size();
            fwrite(&len, sizeof(len), 1, f);
            fwrite(s.data(), 1, s.size(), f);
        }
        fclose(f);
        fprintf(stderr, "cpu-timeline: wrote %zu graph computes, %zu node records to %s\n", calls.size(), total, path.c_str());
    }
};

timeline & tl() {
    static timeline t;
    return t;
}

} // namespace

int64_t ggml_cpu_timeline_now(void) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000 + ts.tv_nsec;
}

int32_t ggml_cpu_timeline_begin(const struct ggml_cgraph * cgraph, int n_threads) {
    auto & t = tl();
    if (t.done) {
        return -1;
    }
    if (!t.recording) {
        if ((t.polls++ & 63) != 0 || !t.exists(".start")) {
            return -1;
        }
        t.recording = true;
        fprintf(stderr, "cpu-timeline: recording\n");
    }
    if (n_threads > k_max_threads || t.calls.size() >= t.cap_calls || t.full.load(std::memory_order_relaxed)) {
        return -1;
    }
    bool expected = false;
    if (!t.busy.compare_exchange_strong(expected, true)) {
        return -1; // another graph compute is running concurrently; record only one at a time
    }
    call_rec r = {};
    r.t_call = ggml_cpu_timeline_now();
    r.n_threads = n_threads;
    r.n_nodes = cgraph->n_nodes;
    r.name = -1;
    t.calls.push_back(r);
    return (int32_t) t.calls.size() - 1;
}

void ggml_cpu_timeline_end(int32_t call) {
    auto & t = tl();
    t.calls[call].t_ret = ggml_cpu_timeline_now();
    t.busy.store(false, std::memory_order_release);
    if (t.full.load(std::memory_order_relaxed) || t.calls.size() >= t.cap_calls || ((call & 63) == 0 && t.exists(".stop"))) {
        t.done = true;
        t.write();
    }
}

void ggml_cpu_timeline_enter(int32_t call, int ith) {
    auto & b = tl().threads[ith];
    if (!b.enter) {
        b.enter = (enter_rec *) calloc(tl().cap_calls, sizeof(enter_rec));
        b.nodes = (node_rec *) calloc(tl().cap_nodes, sizeof(node_rec));
    }
    if (b.n_enter < tl().cap_calls) {
        b.enter[b.n_enter++] = { call, sched_getcpu(), ggml_cpu_timeline_now(), 0 };
    }
}

void ggml_cpu_timeline_node(int32_t call, int ith, int node, int n_fused, const struct ggml_tensor * tensor,
                            int64_t t0, int64_t t1, int64_t t2) {
    auto & t = tl();
    auto & b = t.threads[ith];
    if (b.n_nodes >= t.cap_nodes) {
        t.full.store(true, std::memory_order_relaxed);
        return;
    }
    int32_t id = -1;
    int16_t op = -1, experts = 0, rows = 0;
    if (tensor) { // only the main thread passes the node
        id = t.intern(tensor->name);
        op = (int16_t) tensor->op;
        if (t.calls[call].name < 0) {
            t.calls[call].name = id;
        }
        const ggml_tensor * ids = tensor->src[2];
        if (tensor->op == GGML_OP_MUL_MAT_ID && ids && ids->type == GGML_TYPE_I32) {
            const int64_t n_as = tensor->src[0]->ne[2];
            std::vector<bool> seen(n_as, false);
            for (int64_t r = 0; r < ids->ne[1]; ++r) {
                for (int64_t k = 0; k < ids->ne[0]; ++k) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + r * ids->nb[1] + k * ids->nb[0]);
                    if (e >= 0 && e < n_as) {
                        ++rows;
                        if (!seen[e]) {
                            seen[e] = true;
                            ++experts;
                        }
                    }
                }
            }
        }
    }
    b.nodes[b.n_nodes++] = { call, (int16_t) node, (int16_t) n_fused, id, op, experts, rows, 0, t0, t1, t2 };
}

void ggml_cpu_timeline_exit(int32_t call, int ith) {
    auto & b = tl().threads[ith];
    if (b.n_enter > 0 && b.enter[b.n_enter - 1].call == call) {
        b.enter[b.n_enter - 1].t_exit = ggml_cpu_timeline_now();
    }
}
