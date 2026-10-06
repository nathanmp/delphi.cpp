// Standalone complete routed-FFN replay harness for the GLM expert shape. No inference backend changes are required to build it.
//
//   ggml-ffn GGUF FIXTURE_DIR LAYER WIDTH REPEATS ITERATIONS [--manifest]     timing, one JSON line per fixture (--manifest: FIXTURE_DIR is
//                                                                             a file of "WIDTH DIRECTORY" lines); DELPHI_COMPARE_PACE_US: paced bursts
//   ggml-ffn --lifecycle GGUF_A LAYER_A FIXTURES_A GGUF_B LAYER_B FIXTURES_B  INT4 path ownership and fallback tests
//   ggml-ffn --verify-pack GGUF LAYER REFERENCE_DIR EXPERT...                 INT4 packing against the offline packer
//
// The fixture directory holds input.f32 [M, 4096], ids.i32 [M, 8] and router.f32 [M, 8]. Optional: reference-output.f64 (ideal FP64
// execution of the weights), cmp-orig-output.f32 (the CMP "orig" output on the matched INT4 weights), ggml-output.f32 (the output
// of the Q4_0 path). Every run reports the error against the files that exist. DELPHI_COMPARE_OUTPUT_DIR: also write the output.
// The graph is the one of build_moe_ffn (llama-graph.cpp): up, gate, swiglu_clamp, down, router weights, ordered slot reduction.
// GGML_CUDA_MOE_INT4_DECODE=1 (or --lifecycle) lets the INT4 path take that graph when the weights satisfy its contract.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "gguf.h"
#include "llama.h"
#include "ggml-cuda/common.cuh"

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void load_bytes(const std::string & path, size_t offset, ggml_tensor * tensor) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "input file unavailable: " + path);
    file.seekg(offset);
    std::vector<char> chunk(16 * 1024 * 1024);
    for (size_t pos = 0; pos < ggml_nbytes(tensor);) {
        size_t n = std::min(chunk.size(), ggml_nbytes(tensor) - pos);
        file.read(chunk.data(), n);
        require(size_t(file.gcount()) == n, "short input read");
        ggml_backend_tensor_set(tensor, chunk.data(), pos, n);
        pos += n;
    }
}

template <typename T>
static bool read_file(const std::string & path, size_t count, std::vector<T> & out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { return false; }
    out.resize(count);
    file.read(reinterpret_cast<char *>(out.data()), count * sizeof(T));
    return size_t(file.gcount()) == count * sizeof(T);
}

// ---- weights of one layer ------------------------------------------------------------------------------------------------------

static const char * projections[] = {"gate", "up", "down"};
static constexpr int64_t k_hidden = 4096, k_ff = 2048, k_experts = 288, k_topk = 8;

struct bank {
    ggml_backend_t        backend = nullptr;
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_tensor *         weights[3] = {};
    ggml_tensor *         scales[3]  = {};
    int                   layer = 0;
};

static bank load_bank(ggml_backend_dev_t dev, const std::string & model, int layer) {
    bank b;
    b.layer   = layer;
    b.backend = ggml_backend_dev_init(dev, nullptr);
    require(b.backend != nullptr, "weight backend init failed");
    ggml_context * metadata = nullptr;
    auto uf = gguf_init_from_file(model.c_str(), {true, &metadata});
    require(uf != nullptr && metadata != nullptr, "GGUF metadata load failed");
    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    b.ctx = ggml_init(params);
    std::vector<std::pair<ggml_tensor *, size_t>> reads;
    for (int p = 0; p < 3; ++p) {
        std::string base = "blk." + std::to_string(layer) + ".ffn_" + projections[p] + "_exps";
        for (int s = 0; s < 2; ++s) {
            std::string name = base + (s ? ".scale" : ".weight");
            auto src = ggml_get_tensor(metadata, name.c_str());
            if (s && src == nullptr) { continue; }
            require(src != nullptr, "expert tensor missing: " + name);
            auto dst = ggml_dup_tensor(b.ctx, src);
            ggml_set_name(dst, name.c_str());
            (s ? b.scales[p] : b.weights[p]) = dst;
            reads.emplace_back(dst, gguf_get_data_offset(uf) + gguf_get_tensor_offset(uf, gguf_find_tensor(uf, name.c_str())));
        }
        require(ggml_is_quantized(b.weights[p]->type), "expected quantized expert weights");
    }
    require(b.weights[0]->ne[0] == k_hidden && b.weights[0]->ne[1] == k_ff && b.weights[0]->ne[2] == k_experts, "unexpected GLM dimensions");
    b.buf = ggml_backend_alloc_ctx_tensors(b.ctx, b.backend);
    require(b.buf != nullptr, "weight allocation failed");
    ggml_backend_buffer_set_usage(b.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto & read : reads) { load_bytes(model, read.second, read.first); }
    std::cerr << "weights loaded: " << ggml_backend_buffer_get_size(b.buf) << " bytes, gate at " << b.weights[0]->data << "\n";
    gguf_free(uf);
    ggml_free(metadata);
    return b;
}

static void free_bank(bank & b) {
    ggml_backend_buffer_free(b.buf);
    ggml_backend_free(b.backend);
    ggml_free(b.ctx);
    b = bank();
}

// ---- one graph, its scheduler, and the captured CUDA graph ---------------------------------------------------------------------------

struct session {
    ggml_backend_dev_t        dev = nullptr;
    bank *                    b   = nullptr;
    int                       width = 0;
    std::string               dir;
    bool                      own_backend = true;
    ggml_backend_t            backend = nullptr, cpu = nullptr;
    ggml_context *            ctx = nullptr, * inputs = nullptr;
    ggml_backend_buffer_t     ib = nullptr;
    ggml_backend_sched_t      sched = nullptr;
    ggml_cgraph *             graph = nullptr;
    ggml_tensor *             x = nullptr, * ids = nullptr, * rw = nullptr, * out = nullptr;
    ggml_backend_cuda_context * cuda_ctx = nullptr;
    cudaGraph_t               cuda_graph = nullptr;
    cudaGraphExec_t           exec = nullptr;
    cudaStream_t              stream = nullptr;
    size_t                    nodes = 0;
    std::vector<uint64_t>     known_keys;   // CUDA graph keys of the backend before this graph was built
    std::vector<std::string>  kernels;
    int                       n_kernel_nodes = 0, n_other_nodes = 0;
};

static std::vector<std::string> graph_kernels(cudaGraph_t g, int & n_kernel, int & n_other) {
    std::vector<std::string> names;
    size_t n = 0;
    CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &n));
    n_kernel = n_other = 0;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeKernel) { ++n_other; continue; }
        ++n_kernel;
        // the driver API: the kernels belong to the module of the backend library, not to the runtime of this program
        CUDA_KERNEL_NODE_PARAMS p = {};
        const char * name = nullptr;
        bool known = cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &p) == CUDA_SUCCESS;
        known = known && (p.func != nullptr ? cuFuncGetName(&name, p.func) : cuKernelGetName(&name, p.kern)) == CUDA_SUCCESS && name != nullptr;
        names.push_back(known ? name : "?");
    }
    return names;
}

// shared_backend: build the graph on an existing backend (one context, several graphs), else on a new backend.
static void session_create(session & s, ggml_backend_dev_t dev, bank & b, int width, const std::string & dir, ggml_backend_t shared_backend = nullptr) {
    s.dev = dev; s.b = &b; s.width = width; s.dir = dir;
    s.own_backend = shared_backend == nullptr;
    s.backend = shared_backend ? shared_backend : ggml_backend_dev_init(dev, nullptr);
    require(s.backend != nullptr, "fixture backend init failed");
    // The graph key depends on the shapes only: a second graph of the same widths on one backend shares the captured graph.
    // Scheduling a graph can already create its entry, so the keys are recorded before that.
    for (const auto & g : static_cast<ggml_backend_cuda_context *>(s.backend->context)->cuda_graphs) { s.known_keys.push_back(g.first); }
    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    s.ctx = ggml_init(params);
    s.graph = ggml_new_graph_custom(s.ctx, 256, false);
    // Persistent GPU inputs keep direct replay independent of scheduler copies.
    s.inputs = ggml_init(params);
    s.x   = ggml_new_tensor_3d(s.inputs, GGML_TYPE_F32, k_hidden, 1, width);
    s.ids = ggml_new_tensor_2d(s.inputs, GGML_TYPE_I32, k_topk, width);
    s.rw  = ggml_new_tensor_3d(s.inputs, GGML_TYPE_F32, 1, k_topk, width);
    ggml_set_name(s.x, "fixture-input"); ggml_set_input(s.x);
    ggml_set_name(s.ids, "fixture-ids"); ggml_set_input(s.ids);
    ggml_set_name(s.rw, "fixture-router-weights"); ggml_set_input(s.rw);
    s.ib = ggml_backend_alloc_ctx_tensors(s.inputs, s.backend);
    require(s.ib != nullptr, "input allocation failed");
    const int64_t e = k_experts;
    auto mm = [&](int p, ggml_tensor * in) {
        auto out = ggml_mul_mat_id(s.ctx, b.weights[p], in, s.ids);
        ggml_set_name(out, (std::string("ffn_moe_") + projections[p] + "-3").c_str());
        if (b.scales[p] == nullptr) { return out; }
        auto sc = ggml_reshape_3d(s.ctx, b.scales[p], 1, e, 1);
        sc = ggml_repeat_4d(s.ctx, sc, 1, e, width, 1);
        sc = ggml_get_rows(s.ctx, sc, s.ids);
        out = ggml_mul(s.ctx, out, sc);
        ggml_set_name(out, (std::string("ffn_moe_") + projections[p] + "_scaled-3").c_str());
        return out;
    };
    // Follow the model builder's up-before-gate and ordered slot reduction.
    auto up = mm(1, s.x);
    auto gate = mm(0, s.x);
    auto act = ggml_swiglu_clamp(s.ctx, gate, up, 10.0f);
    ggml_set_name(act, "ffn_moe_swiglu_limited-3");
    auto down = mm(2, act);
    auto weighted = ggml_mul(s.ctx, down, s.rw);
    ggml_set_name(weighted, "ffn_moe_weighted-3");
    ggml_build_forward_expand(s.graph, weighted);
    ggml_tensor * slots[8];
    for (int i = 0; i < k_topk; ++i) {
        slots[i] = ggml_view_2d(s.ctx, weighted, k_hidden, width, weighted->nb[2], i * weighted->nb[1]);
        ggml_build_forward_expand(s.graph, slots[i]);
    }
    auto out = slots[0];
    for (int i = 1; i < k_topk; ++i) {
        out = ggml_add(s.ctx, out, slots[i]);
        ggml_build_forward_expand(s.graph, out);
    }
    ggml_set_name(out, "ffn_moe_out-3"); ggml_set_output(out);
    s.out = out;
    // Use the scheduler so CUDA fusion can declare extended allocation lifetimes.
    s.cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    require(s.cpu != nullptr, "CPU backend init failed");
    ggml_backend_t backends[] = {s.backend, s.cpu};
    s.sched = ggml_backend_sched_new(backends, nullptr, 2, 256, false, true);
    require(ggml_backend_sched_alloc_graph(s.sched, s.graph), "graph allocation failed");
    load_bytes(dir + "/input.f32", 0, s.x);
    load_bytes(dir + "/ids.i32", 0, s.ids);
    load_bytes(dir + "/router.f32", 0, s.rw);
}

static std::vector<float> session_output(session & s) {
    std::vector<float> result(k_hidden * s.width);
    ggml_backend_tensor_get(s.out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// Eight backend executions (the last ones capture and replay), then direct access to the captured executable graph.
// This test-only access avoids comparing Python/C++ graph submission overhead. The private structure must match the backend
// binary's source and build flags.
static void session_warm(session & s) {
    s.cuda_ctx = static_cast<ggml_backend_cuda_context *>(s.backend->context);
    for (int i = 0; i < 8; ++i) {
        require(ggml_backend_sched_graph_compute(s.sched, s.graph) == GGML_STATUS_SUCCESS, "warmup failed");
    }
    ggml_cuda_graph * captured = nullptr;
    int added = 0;
    for (const auto & g : s.cuda_ctx->cuda_graphs) {
        if (std::find(s.known_keys.begin(), s.known_keys.end(), g.first) == s.known_keys.end()) { captured = g.second.get(); ++added; }
    }
    require(added == 1 && captured != nullptr, "expected one new captured FFN graph");
    require(captured->instance != nullptr, "FFN was not captured");
    s.exec = captured->instance;
    s.cuda_graph = captured->graph;
    s.stream = s.cuda_ctx->streams[s.cuda_ctx->device][0];
    CUDA_CHECK(cudaGraphGetNodes(s.cuda_graph, nullptr, &s.nodes));
    s.kernels = graph_kernels(s.cuda_graph, s.n_kernel_nodes, s.n_other_nodes);
    // the backend replay and a direct replay of the captured graph must agree
    std::vector<float> result = session_output(s);
    CUDA_CHECK(cudaGraphLaunch(s.exec, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    require(result == session_output(s), "direct replay differs from backend replay");
}

static void session_replay(session & s, int n) {
    for (int i = 0; i < n; ++i) { CUDA_CHECK(cudaGraphLaunch(s.exec, s.stream)); }
}

static bool session_uses(const session & s, const char * kernel) {
    for (const auto & k : s.kernels) { if (k.find(kernel) != std::string::npos) { return true; } }
    return false;
}

static void session_destroy(session & s) {
    ggml_backend_synchronize(s.backend);
    ggml_backend_sched_free(s.sched);
    if (s.own_backend) { ggml_backend_free(s.backend); }
    ggml_backend_free(s.cpu);
    ggml_backend_buffer_free(s.ib); ggml_free(s.inputs);
    ggml_free(s.ctx);
    s = session();
}

// ---- numerics ------------------------------------------------------------------------------------------------------------------------

struct error_stats {
    bool   valid = false;
    double rel_l2 = 0, nmse = 0, max_abs = 0, max_rel = 0, max_abs_over_peak = 0;
    size_t differing = 0;     // outputs that are not bitwise equal
    bool   finite = true;
};

// max_rel: largest |a - r| / |r| over the elements with |r| >= 1e-3 * max |r|
static error_stats compare_outputs(const std::vector<float> & actual, const std::vector<double> & ref) {
    error_stats e;
    if (actual.size() != ref.size() || ref.empty()) { return e; }
    double num = 0, den = 0, peak = 0;
    for (double r : ref) { peak = std::max(peak, std::fabs(r)); }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(actual[i])) { e.finite = false; }
        const double d = double(actual[i]) - ref[i];
        num += d * d; den += ref[i] * ref[i];
        e.max_abs = std::max(e.max_abs, std::fabs(d));
        if (std::fabs(ref[i]) >= 1e-3 * peak) { e.max_rel = std::max(e.max_rel, std::fabs(d) / std::fabs(ref[i])); }
        e.differing += float(ref[i]) != actual[i];
    }
    e.valid = true;
    e.rel_l2 = std::sqrt(num / den); e.nmse = num / den; e.max_abs_over_peak = e.max_abs / peak;
    return e;
}

static std::string error_json(const error_stats & e) {
    if (!e.valid) { return "null"; }
    std::ostringstream o;
    o << "{\"relative_l2\":" << e.rel_l2 << ",\"nmse\":" << e.nmse << ",\"max_abs\":" << e.max_abs << ",\"max_rel\":" << e.max_rel
      << ",\"max_abs_over_peak\":" << e.max_abs_over_peak << ",\"not_bitwise_equal\":" << e.differing << ",\"finite\":" << (e.finite ? "true" : "false") << "}";
    return o.str();
}

struct references {
    std::vector<double> ideal, golden, ggml;   // empty when the file is missing
};

static references load_references(const std::string & dir, int width) {
    references r;
    const size_t n = size_t(width) * k_hidden;
    read_file(dir + "/reference-output.f64", n, r.ideal);
    std::vector<float> f;
    if (read_file(dir + "/cmp-orig-output.f32", n, f)) { r.golden.assign(f.begin(), f.end()); }
    if (read_file(dir + "/ggml-output.f32", n, f)) { r.ggml.assign(f.begin(), f.end()); }
    return r;
}

typedef void (*set_mode_fn)(int);
typedef void (*stats_fn)(int64_t *);
typedef bool (*pack_fn)(const void *, int, const void **, size_t *);
typedef void (*clear_fn)(void);

struct int4_hooks {
    set_mode_fn set_mode = nullptr;
    stats_fn    stats    = nullptr;
    pack_fn     pack     = nullptr;
    clear_fn    clear    = nullptr;
};

static int4_hooks get_hooks(ggml_backend_dev_t dev) {
    int4_hooks h;
    auto reg = ggml_backend_dev_backend_reg(dev);
    h.set_mode = (set_mode_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_set_mode");
    h.stats    = (stats_fn)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_get_stats");
    h.pack     = (pack_fn)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_get_pack");
    h.clear    = (clear_fn)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_clear");
    return h;
}

static ggml_backend_dev_t init_device() {
    if (const char * bindir = std::getenv("DELPHI_COMPARE_BINDIR")) {
        ggml_backend_load_all_from_path(bindir);
    } else {
        ggml_backend_load_all();
    }
    llama_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    auto dev = ggml_backend_dev_by_name("CUDA0");
    require(dev != nullptr, "CUDA0 not found");
    return dev;
}

static size_t free_vram() {
    size_t free_b = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total));
    return free_b;
}

// ---- timing --------------------------------------------------------------------------------------------------------------------------

static int timing_main(int argc, char ** argv) {
    require(argc == 7 || argc == 8, "usage: ggml-ffn GGUF FIXTURE_DIR LAYER WIDTH REPEATS ITERATIONS [--manifest]");
    const bool manifest = argc == 8;
    require(!manifest || std::string(argv[7]) == "--manifest", "unknown option");
    const std::string model = argv[1];
    std::string dir = argv[2];
    const int layer = std::stoi(argv[3]);
    int width = std::stoi(argv[4]);
    const int repeats = std::stoi(argv[5]), iterations = std::stoi(argv[6]);
    const char * pace_value = std::getenv("DELPHI_COMPARE_PACE_US");
    const int pace_us = pace_value ? std::stoi(pace_value) : 0;
    const int burst_replays = pace_us > 0 ? 16 : 1;
    require(pace_us >= 0, "invalid replay pace");
    require(width > 0 && repeats > 0 && iterations > 0, "invalid count");
    const char * output_dir = std::getenv("DELPHI_COMPARE_OUTPUT_DIR");
    auto dev = init_device();
    const int4_hooks hooks = get_hooks(dev);
    bank b = load_bank(dev, model, layer);

    std::vector<std::pair<int, std::string>> fixtures;
    if (manifest) {
        std::ifstream list(dir);
        require(bool(list), "manifest unavailable");
        while (list >> width >> dir) { fixtures.emplace_back(width, dir); }
        require(!fixtures.empty(), "empty manifest");
    } else {
        fixtures.emplace_back(width, dir);
    }
    // Keep weights resident; each fixture has its own graph and scratch context.
    for (const auto & item : fixtures) {
        width = item.first;
        dir = item.second;
        require(width > 0, "invalid fixture width");
        session s;
        session_create(s, dev, b, width, dir);
        session_warm(s);
        std::cerr << "captured nodes: " << s.nodes << " (" << s.n_kernel_nodes << " kernels, " << s.n_other_nodes << " other)\n";
        const std::vector<float> result = session_output(s);
        if (output_dir != nullptr) {
            std::ofstream output(std::string(output_dir) + "/" + dir.substr(dir.find_last_of('/') + 1) + "-output.f32", std::ios::binary);
            output.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
        }
        const references ref = load_references(dir, width);
        if (pace_us > 0) {
            for (int i = 0; i < 16; ++i) {
                session_replay(s, burst_replays);
                CUDA_CHECK(cudaStreamSynchronize(s.stream));
                std::this_thread::sleep_for(std::chrono::microseconds(pace_us));
            }
        } else {
            session_replay(s, 4096);
        }
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
        const size_t vram_before = free_vram();
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        std::ostream & json = std::cout; // one JSON line per fixture; nothing is written into the fixture directories
        json << "{\"engine\":\"ggml\",\"fixture\":\"" << dir << "\",\"width\":" << width << ",\"nodes\":" << s.nodes << ",\"kernel_nodes\":" << s.n_kernel_nodes
             << ",\"int4_path\":" << (session_uses(s, "moe_dec_gemm") ? "true" : "false") << ",\"weight_types\":[";
        for (int p = 0; p < 3; ++p) {
            if (p) { json << ','; }
            json << '"' << ggml_type_name(b.weights[p]->type) << '"';
        }
        json << "],\"device_us\":[";
        for (int r = 0; r < repeats; ++r) {
            float ms = 0;
            for (int i = 0; i < (pace_us > 0 ? iterations : 1); ++i) {
                CUDA_CHECK(cudaEventRecord(start, s.stream));
                session_replay(s, pace_us > 0 ? burst_replays : iterations);
                CUDA_CHECK(cudaEventRecord(stop, s.stream)); CUDA_CHECK(cudaEventSynchronize(stop));
                float burst_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&burst_ms, start, stop));
                ms += burst_ms;
                if (pace_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(pace_us)); }
            }
            if (r) { json << ','; } json << 1000.0 * ms / (iterations * burst_replays);
        }
        const size_t vram_after = free_vram();
        json << "],\"iterations\":" << iterations << ",\"burst_replays\":" << burst_replays << ",\"pace_us\":" << pace_us
             << ",\"warmup_replays\":" << (pace_us > 0 ? 16 * burst_replays : 4096)
             << ",\"error_ideal\":" << error_json(compare_outputs(result, ref.ideal))
             << ",\"error_cmp_orig\":" << error_json(compare_outputs(result, ref.golden))
             << ",\"error_q4_0_path\":" << error_json(compare_outputs(result, ref.ggml))
             << ",\"free_vram_change_bytes\":" << (int64_t(vram_after) - int64_t(vram_before)) << "}\n";
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        session_destroy(s);
    }
    if (hooks.stats) {
        int64_t st[8] = {};
        hooks.stats(st);
        std::cerr << "int4 stats: entries " << st[0] << " usable " << st[1] << " incompatible " << st[2] << " builds " << st[3] << " build_us "
                  << st[4] << " int4_bytes " << st[5] << " region_runs " << st[6] << " scratch_bytes " << st[7] << "\n";
    }
    free_bank(b);
    return 0;
}

// ---- lifecycle tests -------------------------------------------------------------------------------------------------------------------

static int g_failures = 0;
static void check(bool ok, const std::string & what) {
    std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
    if (!ok) { ++g_failures; }
}

struct case_ref {
    std::string dir;
    int         width;
    references  ref;
};

static case_ref make_case(const std::string & root, const std::string & name) {
    case_ref c;
    c.dir = root + "/" + name;
    c.width = std::stoi(name.substr(name.find("-w") + 2));
    c.ref = load_references(c.dir, c.width);
    require(!c.ref.golden.empty() && !c.ref.ideal.empty(), "missing reference files in " + c.dir);
    return c;
}

static std::string describe(const error_stats & e) {
    std::ostringstream o;
    o << "nmse " << e.nmse << " rel_l2 " << e.rel_l2 << " max_abs " << e.max_abs << " max_rel " << e.max_rel << " not_bitwise " << e.differing;
    return o.str();
}

// Runs one case; the INT4 path must be taken (want_int4) or not, and the result must meet the gates.
static std::vector<float> run_case(const std::string & label, session & s, const case_ref & c, bool want_int4) {
    const std::vector<float> out = session_output(s);
    const bool int4 = session_uses(s, "moe_dec_gemm");
    check(int4 == want_int4, label + ": path " + (int4 ? "int4" : "ggml") + " (" + std::to_string(s.n_kernel_nodes) + " kernel, " + std::to_string(s.n_other_nodes) + " other nodes)");
    const error_stats ideal = compare_outputs(out, c.ref.ideal);
    check(ideal.finite && ideal.nmse <= 5e-4, label + ": ideal FP64 " + describe(ideal));
    if (want_int4) {
        const error_stats golden = compare_outputs(out, c.ref.golden);
        std::cout << "INFO " << label << ": vs CMP orig " << describe(golden) << std::endl;
        check(golden.finite && golden.nmse <= 1e-6, label + ": close to the CMP orig golden output");
    }
    return out;
}

static int lifecycle_main(int argc, char ** argv) {
    require(argc == 8, "usage: ggml-ffn --lifecycle GGUF_A LAYER_A FIXTURES_A GGUF_B LAYER_B FIXTURES_B");
    const std::string gguf_a = argv[2], gguf_b = argv[5], root_a = argv[4], root_b = argv[7];
    const int layer_a = std::stoi(argv[3]), layer_b = std::stoi(argv[6]);
    auto dev = init_device();
    const int4_hooks hooks = get_hooks(dev);
    require(hooks.set_mode && hooks.stats && hooks.pack, "INT4 test hooks missing from the CUDA backend");
    auto stats = [&]() { std::vector<int64_t> v(8); hooks.stats(v.data()); return v; };

    const case_ref a3 = make_case(root_a, "d00-w3"), a4 = make_case(root_a, "d00-w4"), a8 = make_case(root_a, "d00-w8"), a9 = make_case(root_a, "d00-w9"), as3 = make_case(root_a, "s00-w3");
    const case_ref b9 = make_case(root_b, "d00-w9");

    // A. first use: preparation, path, numerics
    hooks.set_mode(1);
    bank bk = load_bank(dev, gguf_a, layer_a);
    const void * address_a = bk.weights[0]->data;
    session s3, s4, s8, s9, ss3;
    const size_t vram_start = free_vram();
    session_create(s3, dev, bk, 3, a3.dir);
    session_warm(s3);
    const std::vector<float> first3 = run_case("A first width 3", s3, a3, true);
    auto st = stats();
    check(st[0] == 1 && st[1] == 1 && st[3] == 1 && st[2] == 0, "A one INT4 copy built, none incompatible (entries " + std::to_string(st[0]) + ", builds " + std::to_string(st[3]) + ")");
    check(s3.n_kernel_nodes == 5 && s3.n_other_nodes == 0, "A the captured FFN is five kernels: no memcpy, memset or host node");
    std::cout << "INFO A kernels of the INT4 path:"; for (const auto & k : s3.kernels) { std::cout << " " << k.substr(0, 80) << ";"; } std::cout << std::endl;
    std::cout << "INFO A int4 bytes " << st[5] << " scratch bytes " << st[7] << " build ms " << st[4] / 1000.0 << " free VRAM drop MiB " << (double(vram_start) - double(free_vram())) / 1048576.0 << std::endl;

    // B. repeated replay: same bits, no memory growth, no new builds
    const size_t vram_before = free_vram();
    session_replay(s3, 3000);
    CUDA_CHECK(cudaStreamSynchronize(s3.stream));
    check(session_output(s3) == first3, "B 3000 replays: output unchanged bitwise");
    check(free_vram() == vram_before, "B no device memory growth during replay");
    auto st2 = stats();
    check(st2[3] == st[3] && st2[6] == st[6], "B no new builds or captures during replay");

    // C. several widths on one backend context (distinct widths: same shapes share one captured graph) and the real shallow input on
    //    its own context; replay all graphs interleaved
    session_create(s4, dev, bk, 4, a4.dir, s3.backend); session_warm(s4);
    session_create(s8, dev, bk, 8, a8.dir, s3.backend); session_warm(s8);
    session_create(s9, dev, bk, 9, a9.dir, s3.backend); session_warm(s9);
    session_create(ss3, dev, bk, 3, as3.dir); session_warm(ss3); // own context: a width-3 graph on s3's context would share its key
    const std::vector<float> o4 = run_case("C width 4", s4, a4, true);
    const std::vector<float> o8 = run_case("C width 8", s8, a8, true);
    const std::vector<float> o9 = run_case("C width 9", s9, a9, true);
    const std::vector<float> os3 = run_case("C actual shallow width 3", ss3, as3, true);
    check(stats()[3] == 1, "C the widths reuse the one INT4 copy");
    session * all[] = {&s3, &s4, &s8, &s9, &ss3};
    for (int round = 0; round < 40; ++round) {
        for (int i = 0; i < 5; ++i) { session_replay(*all[(i + round) % 5], 1 + (round + i) % 3); }
    }
    for (session * s : all) { CUDA_CHECK(cudaStreamSynchronize(s->stream)); }
    check(session_output(s3) == first3 && session_output(s4) == o4 && session_output(s8) == o8 && session_output(s9) == o9 && session_output(ss3) == os3,
        "C interleaved replays of five graphs on one context: every output unchanged bitwise");
    check(std::equal(o4.begin(), o4.begin() + 3 * k_hidden, first3.begin()), "C width-3 rows equal the first rows of the width-4 run (slots are independent)");

    // D. two backend contexts on the device replay without synchronisation: scratch is per context
    session d9;
    session_create(d9, dev, bk, 9, a9.dir); session_warm(d9);
    check(d9.stream != s9.stream, "D the second context has its own stream");
    for (int i = 0; i < 400; ++i) { session_replay(d9, 1); session_replay(s9, 1); session_replay(s3, 1); }
    CUDA_CHECK(cudaDeviceSynchronize());
    check(session_output(d9) == o9 && session_output(s9) == o9 && session_output(s3) == first3, "D concurrent contexts and graphs: outputs unchanged bitwise");
    session_destroy(d9);

    // E. teardown and reload: the next buffer may take the address of the old one; the INT4 copy must not survive it
    for (int i = 4; i >= 0; --i) { session_destroy(*all[i]); } // s3 owns the context that s4, s8 and s9 use: it goes last
    free_bank(bk);
    check(stats()[0] == 0, "E freeing the weight buffer forgets its INT4 copy");
    bank bb = load_bank(dev, gguf_b, layer_b);
    std::cout << "INFO E second weights at " << bb.weights[0]->data << (bb.weights[0]->data == address_a ? " (the same address as the first)" : " (a different address)") << std::endl;
    session t9;
    session_create(t9, dev, bb, 9, b9.dir); session_warm(t9);
    const std::vector<float> ob9 = run_case("E reload, other layer, width 9", t9, b9, true);
    check(compare_outputs(ob9, a9.ref.golden).nmse > 1e-3, "E the reloaded weights give different output than the first weights (no stale INT4 copy)");
    check(stats()[3] == 2, "E the INT4 copy was rebuilt");
    session_destroy(t9);
    free_bank(bb);

    // F. A scale group that breaks the contract keeps the old path
    struct variant { const char * name; bool equal_but_inexact; };
    for (const variant & v : {variant{"unequal block scales", false}, variant{"equal but not BF16 scales", true}}) {
        bank bs = load_bank(dev, gguf_a, layer_a);
        // gate expert 7, row 3, group of 128 values 5 = Q4_0 blocks 20..23 of that row
        const ggml_tensor * g = bs.weights[0];
        const size_t row_off = size_t(7) * g->nb[2] + size_t(3) * g->nb[1];
        for (int blk = 20; blk < 24; ++blk) {
            uint16_t d = 0;
            ggml_backend_tensor_get(bs.weights[0], &d, row_off + blk * 18, 2);
            if (v.equal_but_inexact) { d = 0x3c01; }                // 1.0009765625 = 1 + 2^-10: not a BF16 value
            else if (blk == 22) { d = uint16_t(d ^ 0x0400); }      // one of the four differs
            ggml_backend_tensor_set(bs.weights[0], &d, row_off + blk * 18, 2);
        }
        const auto before = stats();
        session f3, ref3;
        hooks.set_mode(1);
        session_create(f3, dev, bs, 3, a3.dir); session_warm(f3);
        const std::vector<float> fo = session_output(f3);
        check(!session_uses(f3, "moe_dec_gemm"), std::string("F ") + v.name + ": the old path runs");
        auto after = stats();
        check(after[2] == before[2] + 1 && after[3] == before[3], std::string("F ") + v.name + ": counted as incompatible, no INT4 copy built");
        session_destroy(f3);
        hooks.set_mode(0);
        session_create(ref3, dev, bs, 3, a3.dir); session_warm(ref3);
        check(session_output(ref3) == fo, std::string("F ") + v.name + ": output equals the switch-off output bitwise");
        session_destroy(ref3);
        free_bank(bs);
        hooks.set_mode(1);
    }

    // G. the switch off restores the previous path: same kernels and bits as the archived Q4_0 output of this graph
    hooks.set_mode(0);
    bank bo = load_bank(dev, gguf_a, layer_a);
    session o3;
    session_create(o3, dev, bo, 3, a3.dir); session_warm(o3);
    const std::vector<float> oo3 = session_output(o3);
    check(!session_uses(o3, "moe_dec_gemm"), "G switch off: no INT4 kernel in the captured graph (" + std::to_string(o3.n_kernel_nodes) + " kernels)");
    std::cout << "INFO G kernels of the Q4_0 path:"; for (const auto & k : o3.kernels) { std::cout << " " << k.substr(0, 80) << ";"; } std::cout << std::endl;
    if (!a3.ref.ggml.empty()) {
        const error_stats q = compare_outputs(oo3, a3.ref.ggml);
        check(q.nmse <= 1e-10, "G switch off: output equals the archived Q4_0 output of the preceding experiment " + describe(q));
    }
    session_destroy(o3);
    free_bank(bo);
    hooks.set_mode(-1);
    std::cout << (g_failures == 0 ? "ALL PASS" : "FAILURES: " + std::to_string(g_failures)) << std::endl;
    return g_failures == 0 ? 0 : 1;
}

// ---- packing against the offline packer ------------------------------------------------------------------------------------------------

// REFERENCE_DIR holds w13_e<E>.bin, s13_e<E>.bin, w2_e<E>.bin and s2_e<E>.bin per expert, cut from the marlin-int4.pt of the same
// layer by tools/cmp-int4-compare/dump-reference-pack.py (the packer of the CMP experiments). DELPHI_COMPARE_FIXTURE: a width-3
// fixture directory; running it builds the INT4 copy.
static int verify_pack_main(int argc, char ** argv) {
    require(argc >= 6, "usage: ggml-ffn --verify-pack GGUF LAYER REFERENCE_DIR EXPERT...");
    const std::string model = argv[2], refdir = argv[4];
    const int layer = std::stoi(argv[3]);
    auto dev = init_device();
    const int4_hooks hooks = get_hooks(dev);
    require(hooks.set_mode && hooks.pack, "INT4 test hooks missing from the CUDA backend");
    hooks.set_mode(1);
    bank b = load_bank(dev, model, layer);
    const char * fixture = std::getenv("DELPHI_COMPARE_FIXTURE");
    require(fixture != nullptr, "set DELPHI_COMPARE_FIXTURE to a width-3 fixture directory");
    session s;
    session_create(s, dev, b, 3, fixture);
    session_warm(s);
    const char * names[] = {"w13", "s13", "w2", "s2"};
    size_t total_bytes[4] = {};
    const void * base[4] = {};
    for (int w = 0; w < 4; ++w) {
        require(hooks.pack(b.weights[0]->data, w, &base[w], &total_bytes[w]), "no INT4 copy was built");
    }
    for (int i = 5; i < argc; ++i) {
        const int expert = std::stoi(argv[i]);
        for (int w = 0; w < 4; ++w) {
            const size_t per_expert = total_bytes[w] / k_experts;
            std::vector<char> ref;
            require(read_file(refdir + "/" + names[w] + "_e" + std::to_string(expert) + ".bin", per_expert, ref), std::string("reference file missing or short: ") + names[w]);
            std::vector<char> mine(per_expert);
            CUDA_CHECK(cudaMemcpy(mine.data(), static_cast<const char *>(base[w]) + per_expert * expert, per_expert, cudaMemcpyDeviceToHost));
            size_t bad = 0, first_bad = 0;
            for (size_t j = 0; j < per_expert; ++j) { if (mine[j] != ref[j]) { if (bad++ == 0) { first_bad = j; } } }
            check(bad == 0, std::string("pack ") + names[w] + " expert " + std::to_string(expert) + ": " + std::to_string(per_expert) + " bytes, " + std::to_string(bad) + " differ" +
                (bad ? " (first at byte " + std::to_string(first_bad) + ")" : ""));
        }
    }
    session_destroy(s);
    free_bank(b);
    std::cout << (g_failures == 0 ? "ALL PASS" : "FAILURES: " + std::to_string(g_failures)) << std::endl;
    return g_failures == 0 ? 0 : 1;
}

int main(int argc, char ** argv) try {
    if (argc >= 2 && std::string(argv[1]) == "--lifecycle") { return lifecycle_main(argc, argv); }
    if (argc >= 2 && std::string(argv[1]) == "--verify-pack") { return verify_pack_main(argc, argv); }
    return timing_main(argc, argv);
} catch (const std::exception & ex) {
    std::cerr << ex.what() << '\n'; return 1;
}
