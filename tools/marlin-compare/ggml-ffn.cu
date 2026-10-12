// Standalone complete routed-FFN replay harness for the GLM expert shape. No inference backend changes are required to build it.
//
//   ggml-ffn GGUF FIXTURE_DIR LAYER WIDTH REPEATS ITERATIONS [--manifest]     timing, one JSON line per fixture (--manifest: FIXTURE_DIR is
//                                                                             a file of "WIDTH DIRECTORY" lines); DELPHI_COMPARE_PACE_US: paced bursts
//   ggml-ffn --lifecycle GGUF_A LAYER_A FIXTURES_A GGUF_B LAYER_B FIXTURES_B [Q4_0_GGUF_A]  INT4 path ownership and fallback tests; the
//                                                                             GGUFs hold Q4_G128 experts, Q4_0_GGUF_A the same values as Q4_0
//   ggml-ffn --verify-pack GGUF LAYER REFERENCE_DIR EXPERT...                 INT4 in-place packing against the offline packer (Q4_G128 GGUF)
//   ggml-ffn --prefill GGUF LAYER FIXTURE_DIR WIDTH ITERATIONS [OUTPUT.f32]   eager timing of the FFN graph at any width (not captured), the output optionally saved
//   ggml-ffn --prefill-verify GGUF LAYER FIXTURE_DIR WIDTH                    Q4_G128 GGUF: the output at WIDTH rows (prefill kernels from 64 rows) against the decode
//                                                                             kernels run on the same rows in chunks of 16, bitwise per row
//   ggml-ffn --shapes                                                         Q4_G128 MUL_MAT_ID of shapes the executor does not take (random weights): the ordinary
//                                                                             Q4_G128 kernels against a host reference
//
// The fixture directory holds input.f32 [M, 4096], ids.i32 [M, 8] and router.f32 [M, 8]. Optional: reference-output.f64 (ideal FP64
// execution of the weights), cmp-orig-output.f32 (the CMP "orig" output on the matched INT4 weights), ggml-output.f32 (the output
// of the Q4_0 path). Every run reports the error against the files that exist. DELPHI_COMPARE_OUTPUT_DIR: also write the output.
// The graph is the one of build_moe_ffn (llama-graph.cpp): up, gate, swiglu_clamp, down, router weights, ordered slot reduction.
// GGML_CUDA_MOE_INT4_DECODE=1 (or --lifecycle) lets the INT4 path take that graph when the weights are Q4_G128.
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
#include <cstdint>
#include <filesystem>
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
    // DELPHI_COMPARE_IDS_VIEW: the ids are the first 8 columns of rows of 288, as the view of an argsort result in build_moe_ffn
    const bool ids_view = std::getenv("DELPHI_COMPARE_IDS_VIEW") != nullptr;
    ggml_tensor * ids_full = ids_view ? ggml_new_tensor_2d(s.inputs, GGML_TYPE_I32, k_experts, width) : nullptr;
    s.ids = ids_view ? ggml_view_2d(s.inputs, ids_full, k_topk, width, ids_full->nb[1], 0) : ggml_new_tensor_2d(s.inputs, GGML_TYPE_I32, k_topk, width);
    s.rw  = ggml_new_tensor_3d(s.inputs, GGML_TYPE_F32, 1, k_topk, width);
    ggml_set_name(s.x, "fixture-input"); ggml_set_input(s.x);
    ggml_set_name(ids_view ? ids_full : s.ids, "fixture-ids"); ggml_set_input(ids_view ? ids_full : s.ids);
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
    if (ids_view) {
        std::vector<int32_t> rows;
        require(read_file(dir + "/ids.i32", size_t(k_topk) * width, rows), "ids.i32 missing or short");
        for (int t = 0; t < width; ++t) { ggml_backend_tensor_set(ids_full, rows.data() + t * k_topk, t * ids_full->nb[1], k_topk * sizeof(int32_t)); }
    } else {
        load_bytes(dir + "/ids.i32", 0, s.ids);
    }
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
    std::vector<float> eager;
    for (int i = 0; i < 8; ++i) {
        require(ggml_backend_sched_graph_compute(s.sched, s.graph) == GGML_STATUS_SUCCESS, "warmup failed");
        if (i == 0) { eager = session_output(s); } // the first execution is not captured
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
    require(result == eager, "captured replay differs from the eager execution");
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
typedef bool (*pack_fn)(const void *, int, const void **, size_t *, size_t *);

struct int4_hooks {
    set_mode_fn set_mode = nullptr;
    stats_fn    stats    = nullptr;
    pack_fn     pack     = nullptr;
};

// ggml_backend_cuda_moe_int4_get_stats
enum { ST_SETS, ST_PACKS, ST_PACK_US, ST_PACKED_BYTES, ST_RUNS, ST_SCRATCH, ST_SHADOWS, ST_SHADOWS_WIDE, ST_REFUSED_ROWS, ST_REFUSED_LAYOUT,
       ST_REFUSED_WEIGHTS, ST_REFUSED_STATE, ST_PREPARE_REFUSED, ST_PREFILL_RUNS, ST_PREFILL_ROWS, ST_CHUNKED_RUNS, ST_COUNT };

static int4_hooks get_hooks(ggml_backend_dev_t dev) {
    int4_hooks h;
    auto reg = ggml_backend_dev_backend_reg(dev);
    h.set_mode = (set_mode_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_set_mode");
    h.stats    = (stats_fn)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_get_stats");
    h.pack     = (pack_fn)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_get_pack");
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
        int64_t st[ST_COUNT] = {};
        hooks.stats(st);
        std::cerr << "int4 stats: packed sets " << st[ST_SETS] << " packs " << st[ST_PACKS] << " pack_us " << st[ST_PACK_US] << " packed_bytes "
                  << st[ST_PACKED_BYTES] << " region_runs " << st[ST_RUNS] << " scratch_bytes " << st[ST_SCRATCH] << " shadow_nodes " << st[ST_SHADOWS]
                  << " (wide " << st[ST_SHADOWS_WIDE] << ") refused rows/layout/weights/state " << st[ST_REFUSED_ROWS] << "/" << st[ST_REFUSED_LAYOUT]
                  << "/" << st[ST_REFUSED_WEIGHTS] << "/" << st[ST_REFUSED_STATE] << "\n";
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

// Re-uploads the expert weights of a bank from the GGUF into the same buffer (the tensors keep their addresses).
static void reload_bank(bank & b, const std::string & model, int layer) {
    auto uf = gguf_init_from_file(model.c_str(), {true, nullptr});
    require(uf != nullptr, "GGUF metadata load failed");
    for (int p = 0; p < 3; ++p) {
        const std::string name = "blk." + std::to_string(layer) + ".ffn_" + projections[p] + "_exps.weight";
        load_bytes(model, gguf_get_data_offset(uf) + gguf_get_tensor_offset(uf, gguf_find_tensor(uf, name.c_str())), b.weights[p]);
    }
    gguf_free(uf);
}

// A fixture made of the rows of other fixtures (dir, rows): input.f32, ids.i32 and router.f32 of the concatenation. Slots are independent,
// so the rows of its output equal the rows of the outputs of the parts, on a path that computes every slot on its own.
static std::string make_wide_fixture(const std::string & out, const std::vector<std::pair<std::string, int>> & parts) {
    std::vector<float> input, router;
    std::vector<int32_t> ids;
    for (const auto & part : parts) {
        std::vector<float> x, r;
        std::vector<int32_t> i;
        require(read_file(part.first + "/input.f32", size_t(part.second) * k_hidden, x) && read_file(part.first + "/router.f32", size_t(part.second) * k_topk, r) &&
            read_file(part.first + "/ids.i32", size_t(part.second) * k_topk, i), "fixture files missing in " + part.first);
        input.insert(input.end(), x.begin(), x.end());
        router.insert(router.end(), r.begin(), r.end());
        ids.insert(ids.end(), i.begin(), i.end());
    }
    std::filesystem::create_directories(out);
    auto write = [&](const char * name, const void * data, size_t n) { std::ofstream f(out + "/" + name, std::ios::binary); f.write(static_cast<const char *>(data), n); };
    write("input.f32", input.data(), input.size() * sizeof(float));
    write("router.f32", router.data(), router.size() * sizeof(float));
    write("ids.i32", ids.data(), ids.size() * sizeof(int32_t));
    return out;
}

static size_t count_diff(const std::vector<char> & a, const std::vector<char> & b) {
    size_t n = 0;
    for (size_t j = 0; j < a.size(); ++j) { n += a[j] != b[j]; }
    return n;
}

static bool bits_equal(const std::vector<float> & a, const std::vector<float> & b, size_t first = 0, size_t n = SIZE_MAX) {
    if (a.size() < first || b.size() < n) { return false; }
    n = std::min(n, a.size() - first);
    return std::memcmp(a.data() + first, b.data(), n * sizeof(float)) == 0;
}

static int lifecycle_main(int argc, char ** argv) {
    require(argc == 8 || argc == 9, "usage: ggml-ffn --lifecycle GGUF_A LAYER_A FIXTURES_A GGUF_B LAYER_B FIXTURES_B [Q4_0_GGUF_A]");
    const std::string gguf_a = argv[2], gguf_b = argv[5], root_a = argv[4], root_b = argv[7];
    const std::string q40_a = argc == 9 ? argv[8] : "";
    const int layer_a = std::stoi(argv[3]), layer_b = std::stoi(argv[6]);
    const char * scratch_env = std::getenv("DELPHI_COMPARE_SCRATCH");
    const std::string scratch = scratch_env ? scratch_env : "/tmp";
    auto dev = init_device();
    const int4_hooks hooks = get_hooks(dev);
    require(hooks.set_mode && hooks.stats && hooks.pack, "INT4 test hooks missing from the CUDA backend");
    auto stats = [&]() { std::vector<int64_t> v(ST_COUNT); hooks.stats(v.data()); return v; };

    const case_ref a3 = make_case(root_a, "d00-w3"), a4 = make_case(root_a, "d00-w4"), a7 = make_case(root_a, "d00-w7"), a8 = make_case(root_a, "d00-w8"),
        a9 = make_case(root_a, "d00-w9"), as3 = make_case(root_a, "s00-w3");
    const case_ref b9 = make_case(root_b, "d00-w9");
    require(!a3.ref.ggml.empty() && !a9.ref.ggml.empty() && !as3.ref.ggml.empty(), "the fixtures lack ggml-output.f32 (the Q4_0 path of the same values)");

    // A. first use: packing in place, path, numerics, no second copy of the weights
    hooks.set_mode(1);
    bank bk = load_bank(dev, gguf_a, layer_a);
    require(bk.weights[0]->type == GGML_TYPE_Q4_G128, "GGUF_A must hold Q4_G128 experts");
    const void * address_a = bk.weights[0]->data;
    session s3, s4, s7, s8, s9, ss3;
    const size_t vram_start = free_vram();
    session_create(s3, dev, bk, 3, a3.dir);
    session_warm(s3);
    const std::vector<float> first3 = run_case("A first width 3", s3, a3, true);
    auto st = stats();
    check(st[ST_SETS] == 1 && st[ST_PACKS] == 1 && st[ST_PREPARE_REFUSED] == 0, "A the weights were packed in place once (sets " + std::to_string(st[ST_SETS]) + ", packs " + std::to_string(st[ST_PACKS]) + ")");
    check(s3.n_kernel_nodes == 5 && s3.n_other_nodes == 0, "A the captured FFN is five kernels: no memcpy, memset or host node");
    std::cout << "INFO A kernels of the INT4 path:"; for (const auto & k : s3.kernels) { std::cout << " " << k.substr(0, 80) << ";"; } std::cout << std::endl;
    const double vram_drop_mib = (double(vram_start) - double(free_vram())) / 1048576.0;
    std::cout << "INFO A packed bytes " << st[ST_PACKED_BYTES] << " scratch bytes " << st[ST_SCRATCH] << " pack ms " << st[ST_PACK_US] / 1000.0 << " free VRAM drop MiB " << vram_drop_mib << std::endl;
    check(vram_drop_mib < 64, "A first use needs scratch memory only, no second copy of the weights");

    // B. repeated replay: same bits, no memory growth, no new packing or captures
    const size_t vram_before = free_vram();
    session_replay(s3, 3000);
    CUDA_CHECK(cudaStreamSynchronize(s3.stream));
    check(session_output(s3) == first3, "B 3000 replays: output unchanged bitwise");
    check(free_vram() == vram_before, "B no device memory growth during replay");
    auto st2 = stats();
    check(st2[ST_PACKS] == st[ST_PACKS] && st2[ST_RUNS] == st[ST_RUNS], "B no new packing or captures during replay");

    // C. several widths on one backend context (distinct widths: same shapes share one captured graph), the boundary width 16, and the real
    //    shallow input on its own context; replay all graphs interleaved
    session_create(s4, dev, bk, 4, a4.dir, s3.backend); session_warm(s4);
    session_create(s7, dev, bk, 7, a7.dir, s3.backend); session_warm(s7);
    session_create(s8, dev, bk, 8, a8.dir, s3.backend); session_warm(s8);
    session_create(s9, dev, bk, 9, a9.dir, s3.backend); session_warm(s9);
    session_create(ss3, dev, bk, 3, as3.dir); session_warm(ss3); // own context: a width-3 graph on s3's context would share its key
    const std::vector<float> o4 = run_case("C width 4", s4, a4, true);
    const std::vector<float> o7 = run_case("C width 7", s7, a7, true);
    const std::vector<float> o8 = run_case("C width 8", s8, a8, true);
    const std::vector<float> o9 = run_case("C width 9", s9, a9, true);
    const std::vector<float> os3 = run_case("C actual shallow width 3", ss3, as3, true);
    check(stats()[ST_PACKS] == 1, "C the widths reuse the packed weights");
    session * all[] = {&s3, &s4, &s7, &s8, &s9, &ss3};
    const int n_all = 6;
    for (int round = 0; round < 40; ++round) {
        for (int i = 0; i < n_all; ++i) { session_replay(*all[(i + round) % n_all], 1 + (round + i) % 3); }
    }
    for (session * s : all) { CUDA_CHECK(cudaStreamSynchronize(s->stream)); }
    check(session_output(s3) == first3 && session_output(s4) == o4 && session_output(s7) == o7 && session_output(s8) == o8 && session_output(s9) == o9 && session_output(ss3) == os3,
        "C interleaved replays of six graphs on one context: every output unchanged bitwise");
    check(std::equal(o4.begin(), o4.begin() + 3 * k_hidden, first3.begin()), "C width-3 rows equal the first rows of the width-4 run (slots are independent)");

    // D. two backend contexts on the device replay without synchronisation: scratch is per context
    session d9;
    session_create(d9, dev, bk, 9, a9.dir); session_warm(d9);
    check(d9.stream != s9.stream, "D the second context has its own stream");
    for (int i = 0; i < 400; ++i) { session_replay(d9, 1); session_replay(s9, 1); session_replay(s3, 1); }
    CUDA_CHECK(cudaDeviceSynchronize());
    check(session_output(d9) == o9 && session_output(s9) == o9 && session_output(s3) == first3, "D concurrent contexts and graphs: outputs unchanged bitwise");
    session_destroy(d9);

    // E. widths above 16: 17 rows run on the decode kernels in chunks of 16 and 1 rows, 64 rows on the prefill kernels; every row equals the row of
    //    the decode run that it was cut from (bitwise), whatever the other rows of the batch are
    {
        const std::string w16 = make_wide_fixture(scratch + "/ggml-ffn-wide16", {{a9.dir, 9}, {a7.dir, 7}});
        session t16;
        session_create(t16, dev, bk, 16, w16); session_warm(t16);
        const std::vector<float> o16 = session_output(t16);
        check(session_uses(t16, "moe_dec_gemm") && t16.n_kernel_nodes == 5, "E width 16 runs on the INT4 path (5 kernels)");
        check(bits_equal(o16, o9, 0, 9 * k_hidden) && std::equal(o16.begin() + 9 * k_hidden, o16.end(), o7.begin()), "E width 16: rows equal the rows of the width-9 and width-7 runs they were cut from (bitwise)");
        session_destroy(t16);
        const auto before = stats();
        const std::string w17 = make_wide_fixture(scratch + "/ggml-ffn-wide17", {{a9.dir, 9}, {a8.dir, 8}});
        session t17;
        session_create(t17, dev, bk, 17, w17); session_warm(t17);
        const std::vector<float> o17 = session_output(t17);
        const auto after = stats();
        check(session_uses(t17, "moe_dec_gemm") && !session_uses(t17, "pf_w13") && t17.n_kernel_nodes == 10, "E width 17: two decode chunks (16 and 1 rows) run on the INT4 path (10 kernels, " + std::to_string(t17.n_kernel_nodes) + " found)");
        check(after[ST_CHUNKED_RUNS] > before[ST_CHUNKED_RUNS] && after[ST_SHADOWS_WIDE] == before[ST_SHADOWS_WIDE], "E width 17: counted as chunked runs, no wide conversion");
        check(bits_equal(o17, o9, 0, 9 * k_hidden) && std::equal(o17.begin() + 9 * k_hidden, o17.end(), o8.begin()), "E width 17: rows equal the rows of the width-9 and width-8 runs they were cut from (bitwise)");
        session_destroy(t17);

        // 64 rows (7 x the 9 rows of a9, then row 0 of a9 again): the prefill kernels
        std::vector<std::pair<std::string, int>> parts;
        for (int i = 0; i < 7; ++i) { parts.push_back({a9.dir, 9}); }
        parts.push_back({a9.dir, 1});
        const std::string w64 = make_wide_fixture(scratch + "/ggml-ffn-wide64", parts);
        const auto before64 = stats();
        session t64;
        session_create(t64, dev, bk, 64, w64); session_warm(t64);
        const std::vector<float> o64 = session_output(t64);
        const auto after64 = stats();
        check(session_uses(t64, "pf_w13") && session_uses(t64, "pf_w2") && !session_uses(t64, "moe_dec_gemm"), "E width 64: the prefill kernels run (" + std::to_string(t64.n_kernel_nodes) + " kernels)");
        check(after64[ST_PREFILL_RUNS] > before64[ST_PREFILL_RUNS] && after64[ST_PREFILL_ROWS] >= before64[ST_PREFILL_ROWS] + 64, "E width 64: counted as prefill runs");
        bool rows_ok = true;
        for (int i = 0; i < 7; ++i) { rows_ok = rows_ok && bits_equal(o64, o9, (size_t) i * 9 * k_hidden, 9 * k_hidden); }
        check(rows_ok && std::equal(o64.begin() + 63 * k_hidden, o64.end(), o9.begin()), "E width 64: every row equals the row of the width-9 decode run it was cut from (bitwise)");
        session_destroy(t64);
    }

    // F. the INT4 executor off, weights still PACKED: every node converts the packed bytes to Q4_0; the results equal the Q4_0 path bitwise
    hooks.set_mode(0);
    for (const case_ref * c : {&a3, &a4, &a8, &a9, &as3}) {
        session f;
        session_create(f, dev, bk, c->width, c->dir); session_warm(f);
        const std::vector<float> fo = session_output(f);
        const error_stats q = compare_outputs(fo, c->ref.ggml);
        check(!session_uses(f, "moe_dec_gemm") && session_uses(f, "moe_int4_packed_to_q4_0") && q.nmse <= 1e-10,
            std::string("F executor off, packed weights, ") + c->dir.substr(c->dir.find_last_of('/') + 1) + ": Q4_0 conversion, equals the Q4_0 path " + describe(q));
        session_destroy(f);
    }
    hooks.set_mode(1);

    // G. a write into the weight buffer forgets the packed state: with the executor off the ordinary kernels read the row-major bytes;
    //    with it on the weights are packed again
    for (int i = n_all - 1; i >= 0; --i) { session_destroy(*all[i]); } // s3 owns the context that s4, s7, s8 and s9 use: it goes last
    reload_bank(bk, gguf_a, layer_a);
    check(stats()[ST_SETS] == 0, "G a write into the weight buffer forgets the packed set");
    {
        hooks.set_mode(0);
        session g;
        session_create(g, dev, bk, 3, a3.dir); session_warm(g);
        const error_stats q = compare_outputs(session_output(g), a3.ref.ggml);
        check(!session_uses(g, "moe_int4_packed_to_q4_0") && q.nmse <= 1e-10, "G row-major weights, executor off: the ordinary Q4_G128 kernels, equal the Q4_0 path " + describe(q));
        session_destroy(g);
        hooks.set_mode(1);
        const auto before = stats();
        session g2;
        session_create(g2, dev, bk, 3, a3.dir); session_warm(g2);
        run_case("G repacked width 3", g2, a3, true);
        check(stats()[ST_PACKS] == before[ST_PACKS] + 1, "G the weights were packed again");
        check(session_output(g2) == first3, "G the repacked weights give the first output bitwise");
        session_destroy(g2);
    }

    // H. teardown and reload: the next buffer may take the address of the old one; the packed state must not survive it
    free_bank(bk);
    check(stats()[ST_SETS] == 0, "H freeing the weight buffer forgets its packed set");
    bank bb = load_bank(dev, gguf_b, layer_b);
    std::cout << "INFO H second weights at " << bb.weights[0]->data << (bb.weights[0]->data == address_a ? " (the same address as the first)" : " (a different address)") << std::endl;
    session t9;
    const int64_t packs_before = stats()[ST_PACKS];
    session_create(t9, dev, bb, 9, b9.dir); session_warm(t9);
    const std::vector<float> ob9 = run_case("H reload, other layer, width 9", t9, b9, true);
    check(compare_outputs(ob9, a9.ref.golden).nmse > 1e-3, "H the reloaded weights give different output than the first weights (no stale packed state)");
    check(stats()[ST_PACKS] == packs_before + 1, "H the reloaded weights were packed");
    session_destroy(t9);
    free_bank(bb);
    hooks.set_mode(-1);
    std::cout << (g_failures == 0 ? "ALL PASS" : "FAILURES: " + std::to_string(g_failures)) << std::endl;
    return g_failures == 0 ? 0 : 1;
}

// ---- packing against the offline packer ------------------------------------------------------------------------------------------------

// REFERENCE_DIR holds w13_e<E>.bin, s13_e<E>.bin, w2_e<E>.bin and s2_e<E>.bin per expert, cut from the marlin-int4.pt of the same
// layer by tools/cmp-int4-compare/dump-reference-pack.py (the packer of the CMP experiments): W13 codes int32 [256][8192] (gate words
// 0..4095 of a row, up words 4096..8191), W13 scales bf16 [32][4096] (gate columns 0..2047, up 2048..4095), W2 codes [128][8192], W2 scales
// [16][4096]. Every tensor is packed on its own (one matrix per tensor, codes then scales inside its expert), so the reference is sliced.
// DELPHI_COMPARE_FIXTURE: a width-3 fixture directory; running it packs the weights.
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
    const void * codes[3] = {}, * scales[3] = {};
    size_t codes_bytes = 0, scales_bytes = 0, stride = 0;
    for (int p = 0; p < 3; ++p) {
        require(hooks.pack(b.weights[p]->data, 0, &codes[p], &codes_bytes, &stride) && hooks.pack(b.weights[p]->data, 1, &scales[p], &scales_bytes, &stride), "the weights were not packed");
    }
    require(codes_bytes == 4194304 && scales_bytes == 131072, "unexpected packed sizes");
    auto device_bytes = [&](const void * base, size_t expert, size_t n) {
        std::vector<char> v(n);
        CUDA_CHECK(cudaMemcpy(v.data(), static_cast<const char *>(base) + stride * expert, n, cudaMemcpyDeviceToHost));
        return v;
    };
    for (int i = 5; i < argc; ++i) {
        const int expert = std::stoi(argv[i]);
        std::vector<char> w13, s13, w2, s2;
        require(read_file(refdir + "/w13_e" + std::to_string(expert) + ".bin", 256 * 8192 * 4, w13) && read_file(refdir + "/s13_e" + std::to_string(expert) + ".bin", 32 * 4096 * 2, s13) &&
            read_file(refdir + "/w2_e" + std::to_string(expert) + ".bin", 128 * 8192 * 4, w2) && read_file(refdir + "/s2_e" + std::to_string(expert) + ".bin", 16 * 4096 * 2, s2),
            "reference files missing or short for expert " + std::to_string(expert));
        for (int p = 0; p < 2; ++p) { // gate, up: slices of W13
            std::vector<char> ec(codes_bytes), es(scales_bytes);
            for (int r = 0; r < 256; ++r) { std::memcpy(ec.data() + size_t(r) * 4096 * 4, w13.data() + (size_t(r) * 8192 + p * 4096) * 4, 4096 * 4); }
            for (int g = 0; g < 32; ++g) { std::memcpy(es.data() + size_t(g) * 2048 * 2, s13.data() + (size_t(g) * 4096 + p * 2048) * 2, 2048 * 2); }
            const std::vector<char> mc = device_bytes(codes[p], expert, codes_bytes), ms = device_bytes(scales[p], expert, scales_bytes);
            const size_t bad_c = count_diff(mc, ec), bad_s = count_diff(ms, es);
            check(bad_c == 0 && bad_s == 0, std::string("pack ") + projections[p] + " expert " + std::to_string(expert) + ": " + std::to_string(codes_bytes) + " code bytes " + std::to_string(bad_c) + " differ, " + std::to_string(scales_bytes) + " scale bytes " + std::to_string(bad_s) + " differ");
        }
        const std::vector<char> mc = device_bytes(codes[2], expert, codes_bytes), ms = device_bytes(scales[2], expert, scales_bytes);
        const size_t bad_c = count_diff(mc, w2), bad_s = count_diff(ms, s2);
        check(bad_c == 0 && bad_s == 0, "pack down expert " + std::to_string(expert) + ": " + std::to_string(codes_bytes) + " code bytes " + std::to_string(bad_c) + " differ, " + std::to_string(scales_bytes) + " scale bytes " + std::to_string(bad_s) + " differ");
    }
    session_destroy(s);
    free_bank(b);
    std::cout << (g_failures == 0 ? "ALL PASS" : "FAILURES: " + std::to_string(g_failures)) << std::endl;
    return g_failures == 0 ? 0 : 1;
}

// ---- shapes the executor does not take ---------------------------------------------------------------------------------------------------
// A lone MUL_MAT_ID with Q4_G128 weights (no matched region, other shapes, GLM shapes outside a region) runs on a Q4_0 conversion of the
// experts, from the row-major state the weights have, and must match the host product of the dequantized weights within the error of the
// Q8_1 activations of the ordinary kernels.
static int shapes_main() {
    auto dev = init_device();
    const int4_hooks hooks = get_hooks(dev);
    require(hooks.set_mode && hooks.stats, "INT4 test hooks missing from the CUDA backend");
    hooks.set_mode(1);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    require(backend != nullptr, "backend init failed");
    struct shape_case { int64_t k, n, e; int m, used; };
    const shape_case cases[] = {{256, 64, 16, 1, 4}, {384, 128, 8, 5, 2}, {512, 96, 12, 40, 6}, {4096, 2048, 32, 3, 8}, {2048, 4096, 32, 17, 8}, {640, 192, 24, 1, 8}};
    uint32_t seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (int) (seed >> 8) / 8388608.0f - 1.0f; };
    for (const auto & c : cases) {
        int64_t before[ST_COUNT] = {};
        hooks.stats(before);
        ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_G128, c.k, c.n, c.e);
        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, 1, c.m);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.used, c.m);
        ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, ids);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buf != nullptr, "allocation failed");
        // weights: random values, quantized row by row
        std::vector<float> wf((size_t) c.k * c.n * c.e), xf((size_t) c.k * c.m);
        for (auto & v : wf) { v = rnd(); }
        for (auto & v : xf) { v = rnd(); }
        std::vector<char> wq(ggml_nbytes(w));
        ggml_quantize_chunk(GGML_TYPE_Q4_G128, wf.data(), wq.data(), 0, c.n * c.e, c.k, nullptr);
        std::vector<float> wd(wf.size());
        ggml_get_type_traits(GGML_TYPE_Q4_G128)->to_float(wq.data(), wd.data(), (int64_t) wd.size());
        std::vector<int32_t> idv((size_t) c.used * c.m);
        for (int t = 0; t < c.m; ++t) {
            for (int u = 0; u < c.used; ++u) { idv[(size_t) t * c.used + u] = (int32_t) ((t * 5 + u * 3 + (t % 3)) % c.e); }
        }
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
        ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
        const bool supported = ggml_backend_supports_op(backend, out);
        check(supported, "k " + std::to_string(c.k) + " n " + std::to_string(c.n) + " e " + std::to_string(c.e) + " rows " + std::to_string(c.m) + ": MUL_MAT_ID on Q4_G128 weights is supported");
        if (supported) {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
            std::vector<float> got((size_t) c.n * c.used * c.m);
            ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
            double num = 0, den = 0;
            for (int t = 0; t < c.m; ++t) {
                for (int u = 0; u < c.used; ++u) {
                    const float * wp = wd.data() + (size_t) idv[(size_t) t * c.used + u] * c.n * c.k;
                    for (int64_t r = 0; r < c.n; ++r) {
                        double acc = 0;
                        for (int64_t kk = 0; kk < c.k; ++kk) { acc += (double) wp[r * c.k + kk] * xf[(size_t) t * c.k + kk]; }
                        const double d = got[((size_t) t * c.used + u) * c.n + r] - acc;
                        num += d * d; den += acc * acc;
                    }
                }
            }
            const double rel = std::sqrt(num / std::max(den, 1e-30));
            int64_t after[ST_COUNT] = {};
            hooks.stats(after);
            check(rel < 0.03, "  output matches the host product of the dequantized weights: relative L2 " + std::to_string(rel));
            check(after[ST_SHADOWS] == before[ST_SHADOWS] && after[ST_PACKS] == before[ST_PACKS] && after[ST_RUNS] == before[ST_RUNS], "  the ordinary Q4_G128 kernels: no conversion, no packing, no region");
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    std::cout << (g_failures == 0 ? "ALL PASS" : "FAILURES: " + std::to_string(g_failures)) << std::endl;
    return g_failures == 0 ? 0 : 1;
}

// ---- eager timing at prefill widths ------------------------------------------------------------------------------------------------------
static int prefill_main(int argc, char ** argv) {
    require(argc == 7 || argc == 8, "usage: ggml-ffn --prefill GGUF LAYER FIXTURE_DIR WIDTH ITERATIONS [OUTPUT.f32]");
    const std::string model = argv[2], dir = argv[4];
    const int layer = std::stoi(argv[3]), width = std::stoi(argv[5]), iterations = std::stoi(argv[6]);
    auto dev = init_device();
    const int4_hooks hooks = get_hooks(dev);
    bank b = load_bank(dev, model, layer);
    session s;
    session_create(s, dev, b, width, dir);
    s.cuda_ctx = static_cast<ggml_backend_cuda_context *>(s.backend->context);
    for (int i = 0; i < 3; ++i) { require(ggml_backend_sched_graph_compute(s.sched, s.graph) == GGML_STATUS_SUCCESS, "warmup failed"); }
    const std::vector<float> result = session_output(s);
    if (argc == 8) {
        std::ofstream output(argv[7], std::ios::binary);
        output.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
    }
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start, nullptr));
        require(ggml_backend_sched_graph_compute(s.sched, s.graph) == GGML_STATUS_SUCCESS, "compute failed");
        ggml_backend_synchronize(s.backend);
        CUDA_CHECK(cudaEventRecord(stop, nullptr)); CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    const references ref = load_references(dir, width);
    std::cout << "{\"width\":" << width << ",\"weight_type\":\"" << ggml_type_name(b.weights[0]->type) << "\",\"median_ms\":" << times[times.size() / 2]
              << ",\"min_ms\":" << times.front() << ",\"max_ms\":" << times.back() << ",\"error_ideal\":" << error_json(compare_outputs(result, ref.ideal))
              << ",\"error_q4_0_path\":" << error_json(compare_outputs(result, ref.ggml)) << "}\n";
    if (hooks.stats) {
        int64_t st[ST_COUNT] = {};
        hooks.stats(st);
        std::cerr << "int4 stats: packs " << st[ST_PACKS] << " region_runs " << st[ST_RUNS] << " shadow_nodes " << st[ST_SHADOWS] << " (wide " << st[ST_SHADOWS_WIDE]
                  << ") refused rows/layout/weights/state " << st[ST_REFUSED_ROWS] << "/" << st[ST_REFUSED_LAYOUT] << "/" << st[ST_REFUSED_WEIGHTS] << "/" << st[ST_REFUSED_STATE] << "\n";
    }
    session_destroy(s);
    free_bank(b);
    return 0;
}

static int prefill_verify_main(int argc, char ** argv) {
    require(argc == 6, "usage: ggml-ffn --prefill-verify GGUF LAYER FIXTURE_DIR WIDTH");
    const std::string model = argv[2], dir = argv[4];
    const int layer = std::stoi(argv[3]), width = std::stoi(argv[5]);
    auto dev = init_device();
    const int4_hooks hooks = get_hooks(dev);
    require(hooks.set_mode && hooks.stats, "INT4 test hooks missing from the CUDA backend");
    hooks.set_mode(1);
    bank b = load_bank(dev, model, layer);
    std::vector<float> input, router;
    std::vector<int32_t> ids;
    require(read_file(dir + "/input.f32", size_t(width) * k_hidden, input) && read_file(dir + "/router.f32", size_t(width) * k_topk, router) &&
        read_file(dir + "/ids.i32", size_t(width) * k_topk, ids), "fixture files missing or short in " + dir);
    int64_t st0[ST_COUNT] = {}, st1[ST_COUNT] = {};
    hooks.stats(st0);
    session full;
    session_create(full, dev, b, width, dir);
    for (int i = 0; i < 3; ++i) { require(ggml_backend_sched_graph_compute(full.sched, full.graph) == GGML_STATUS_SUCCESS, "full-width compute failed"); }
    const std::vector<float> out_full = session_output(full);
    hooks.stats(st1);
    std::cout << "width " << width << ": prefill runs " << st1[ST_PREFILL_RUNS] - st0[ST_PREFILL_RUNS] << ", chunked runs " << st1[ST_CHUNKED_RUNS] - st0[ST_CHUNKED_RUNS]
              << ", shadow nodes " << st1[ST_SHADOWS] - st0[ST_SHADOWS] << std::endl;

    std::vector<float> out_chunks(size_t(width) * k_hidden, 0.f);
    std::vector<session> sessions(17);
    for (int row0 = 0; row0 < width; row0 += 16) {
        const int n = std::min(16, width - row0);
        session & t = sessions[n];
        if (t.width == 0) { session_create(t, dev, b, n, dir); }
        ggml_backend_tensor_set(t.x, input.data() + size_t(row0) * k_hidden, 0, size_t(n) * k_hidden * sizeof(float));
        ggml_backend_tensor_set(t.ids, ids.data() + size_t(row0) * k_topk, 0, size_t(n) * k_topk * sizeof(int32_t));
        ggml_backend_tensor_set(t.rw, router.data() + size_t(row0) * k_topk, 0, size_t(n) * k_topk * sizeof(float));
        require(ggml_backend_sched_graph_compute(t.sched, t.graph) == GGML_STATUS_SUCCESS, "chunk compute failed");
        const std::vector<float> o = session_output(t);
        std::copy(o.begin(), o.end(), out_chunks.begin() + size_t(row0) * k_hidden);
    }
    int64_t st2[ST_COUNT] = {};
    hooks.stats(st2);
    std::cout << "chunks: prefill runs " << st2[ST_PREFILL_RUNS] - st1[ST_PREFILL_RUNS] << " (expected 0), region runs " << st2[ST_RUNS] - st1[ST_RUNS] << std::endl;
    size_t bad_rows = 0, bad_values = 0, first_bad = SIZE_MAX;
    for (int r = 0; r < width; ++r) {
        size_t bad = 0;
        for (int64_t i = 0; i < k_hidden; ++i) {
            if (std::memcmp(&out_full[size_t(r) * k_hidden + i], &out_chunks[size_t(r) * k_hidden + i], sizeof(float)) != 0) { ++bad; }
        }
        if (bad) { ++bad_rows; bad_values += bad; if (first_bad == SIZE_MAX) { first_bad = r; } }
    }
    if (bad_rows) {
        double max_rel = 0, sum_rel = 0; size_t n = 0;
        for (size_t i = 0; i < out_full.size(); ++i) { const double d = std::fabs(out_full[i] - out_chunks[i]), m = std::max(std::fabs(out_chunks[i]), 1e-6f); max_rel = std::max(max_rel, d / m); sum_rel += d / m; ++n; }
        std::cout << "row 0, first values: full";
        for (int i = 0; i < 6; ++i) { std::cout << " " << out_full[i]; }
        std::cout << " | chunks";
        for (int i = 0; i < 6; ++i) { std::cout << " " << out_chunks[i]; }
        std::cout << "\nmean relative difference " << sum_rel / n << ", max " << max_rel << std::endl;
    }
    std::cout << (bad_rows == 0 ? "PASS" : "FAIL") << ": " << bad_rows << " of " << width << " rows differ (" << bad_values << " values)";
    if (bad_rows) { std::cout << ", first row " << first_bad; }
    std::cout << std::endl;
    for (auto & t : sessions) { if (t.width != 0) { session_destroy(t); } }
    session_destroy(full);
    free_bank(b);
    return bad_rows == 0 ? 0 : 2;
}

int main(int argc, char ** argv) try {
    if (argc >= 2 && std::string(argv[1]) == "--lifecycle") { return lifecycle_main(argc, argv); }
    if (argc >= 2 && std::string(argv[1]) == "--verify-pack") { return verify_pack_main(argc, argv); }
    if (argc >= 2 && std::string(argv[1]) == "--shapes") { return shapes_main(); }
    if (argc >= 2 && std::string(argv[1]) == "--prefill") { return prefill_main(argc, argv); }
    if (argc >= 2 && std::string(argv[1]) == "--prefill-verify") { return prefill_verify_main(argc, argv); }
    return timing_main(argc, argv);
} catch (const std::exception & ex) {
    std::cerr << ex.what() << '\n'; return 1;
}
