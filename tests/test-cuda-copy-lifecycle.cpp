#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

struct copy_graph {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * source;
    ggml_tensor * output;
    ggml_tensor * prefix = nullptr;
    ggml_cgraph * graph;
    int experts, top_k, rows;
    copy_graph(ggml_backend_t backend, int experts, int top_k, int rows, bool stable_prefix = false) : experts(experts), top_k(top_k), rows(rows) {
        ctx.reset(ggml_init({64*ggml_tensor_overhead()+3*ggml_graph_overhead_custom(64, false), nullptr, true}));
        source = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, experts, rows);
        ggml_set_name(source, "copy_base");
        auto * view = ggml_view_2d(ctx.get(), source, top_k, rows, source->nb[1], 0);
        ggml_set_name(view, "copy_view");
        output = ggml_cont(ctx.get(), view);
        ggml_set_name(output, "copy_output");
        graph = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_tensor * prefix_input = nullptr;
        if (stable_prefix) {
            prefix_input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
            prefix = prefix_input;
            // Most model nodes retain their shape when the route copy changes between one and many rows. This lets
            // the graph cache consider executable lending, which a graph containing only VIEW/CONT never exercises.
            for (int i = 0; i < 32; ++i) { prefix = ggml_sqr(ctx.get(), prefix); }
            ggml_build_forward_expand(graph, prefix);
        }
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        check(bool(buffer), "copy fixture allocation failed");
        if (prefix_input) {
            const float zeros[32]{};
            ggml_backend_tensor_set(prefix_input, zeros, 0, sizeof(zeros));
        }
    }
    void rebind_rows(int next_rows) {
        // Storage was reserved at the maximum row count. Rebuild the graph UID with the same tensor addresses so
        // the test also reaches an anchored cache entry under GGML_CUDA_GRAPH_SHAPE_CACHE=0.
        check(next_rows <= 17 && prefix, "invalid copy rebind");
        rows = next_rows;
        source->ne[1] = output->ne[1] = output->src[0]->ne[1] = rows;
        graph = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_build_forward_expand(graph, prefix);
        ggml_build_forward_expand(graph, output);
    }
    void run(ggml_backend_t backend, int seed) {
        std::vector<int32_t> input(experts*rows), expected(top_k*rows), actual(top_k*rows), sentinel(top_k*rows, 0x3f800000);
        for (size_t i = 0; i < input.size(); ++i) { input[i] = (i+i/experts+seed)%experts; }
        for (int row = 0; row < rows; ++row) {
            for (int j = 0; j < top_k; ++j) { expected[top_k*row+j] = input[experts*row+j]; }
        }
        ggml_backend_tensor_set(source, input.data(), 0, input.size()*sizeof(int32_t));
        ggml_backend_tensor_set(output, sentinel.data(), 0, sentinel.size()*sizeof(int32_t));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "copy graph execution failed");
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(int32_t));
        if (actual != expected) {
            std::fprintf(stderr, "copy-lifecycle,uid=%llu,source=%p,dest=%p,first=%d/%d,expected=%d/%d\n",
                    (unsigned long long) graph->uid, source->data, output->data, actual[0], actual[1], expected[0], expected[1]);
        }
        check(actual == expected, "strided copy lost data after capture/rebinding");
    }
};

static void produced_rows(ggml_backend_t backend) {
    ggml_context_ptr ctx(ggml_init({32*ggml_tensor_overhead()+2*ggml_graph_overhead_custom(32, false), nullptr, true}));
    auto * gate = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 8);
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 17);
    auto * logits = ggml_mul_mat(ctx.get(), gate, input);
    auto * ids = ggml_argsort(ctx.get(), ggml_soft_max(ctx.get(), logits), GGML_SORT_ORDER_DESC);
    auto * view = ggml_view_2d(ctx.get(), ids, 2, 17, ids->nb[1], 0);
    auto * output = ggml_cont(ctx.get(), view);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    // The router logits are dead by CONT. Reusing this storage is valid and matches the model's allocator.
    output->data = logits->data;
    auto * host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(host_buft, ggml_nbytes(input)));
    auto * values = static_cast<float *>(ggml_backend_buffer_get_base(host.get()));
    std::vector<float> weights(256*8, 0);
    for (int e = 0; e < 8; ++e) { weights[256*e+e] = 1; }
    ggml_backend_tensor_set(gate, weights.data(), 0, weights.size()*sizeof(float));
    std::vector<int32_t> actual(2*17), expected(2*17), sentinel(2*17, 0x3f800000);
    for (int repeat = 0; repeat < 45; ++repeat) {
        std::fill(values, values+256*17, 0);
        for (int row = 0; row < 17; ++row) {
            for (int e = 0; e < 8; ++e) { values[256*row+e] = (e+row+repeat)%8; }
            expected[2*row] = (87-row-repeat)%8;
            expected[2*row+1] = (86-row-repeat)%8;
        }
        ggml_backend_tensor_set(output, sentinel.data(), 0, sentinel.size()*sizeof(int32_t));
        ggml_backend_tensor_set_async(backend, input, values, 0, ggml_nbytes(input));
        check(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS, "produced copy graph failed");
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(int32_t));
        if (actual != expected) { std::fprintf(stderr, "produced-copy,round=%d,first=%d/%d,expected=%d/%d\n",
                repeat, actual[0], actual[1], expected[0], expected[1]); }
        check(actual == expected, "GPU-produced strided copy lost its source ordering");
    }
    std::puts("copy-lifecycle,async-input/matmul/argsort/strided-copy/reused-logit-storage,45,ok");
}

int main(int argc, char ** argv) {
    try {
        const char * cpu = nullptr, * hip = nullptr;
        bool tensors = false;
        for (int a = 1; a < argc; ++a) {
            const std::string key = argv[a];
            if (key == "--backend" && a+1 < argc) { cpu = argv[++a]; }
            else if (key == "--hip-backend" && a+1 < argc) { hip = argv[++a]; }
            else if (key == "--numa" && a+1 < argc) { tensors = std::string(argv[++a]) == "tensors"; }
            else { throw std::runtime_error("unknown copy fixture option"); }
        }
        check(cpu && hip && tensors, "explicit CPU/HIP plugins and --numa tensors required");
        auto * cpu_reg = ggml_backend_load(cpu);
        check(cpu_reg && std::strcmp(ggml_backend_reg_name(cpu_reg), "CPU") == 0, "CPU plugin required");
        auto numa_init = reinterpret_cast<decltype(ggml_numa_init) *>(
                ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_numa_init"));
        check(numa_init != nullptr, "CPU NUMA initialization unavailable");
        numa_init(GGML_NUMA_STRATEGY_TENSORS);
        auto * reg = ggml_backend_load(hip);
        check(reg && ggml_backend_reg_dev_count(reg) == 1, "select one HIP device");
        ggml_backend_ptr backend(ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr));
        {
            copy_graph transition(backend.get(), 8, 2, 17, true);
            transition.rebind_rows(1);
            for (int i = 0; i < 3; ++i) { transition.run(backend.get(), i); }
            transition.rebind_rows(17);
            transition.run(backend.get(), 3);
            std::puts("copy-lifecycle,linear-to-pitched/executable-compatibility,ok");
        }
        for (const auto & shape : {std::array<int,3>{8,2,17}, {384,6,5}, {384,6,1024}}) {
            copy_graph graphs[] = {copy_graph(backend.get(), shape[0], shape[1], shape[2]),
                copy_graph(backend.get(), shape[0], shape[1], shape[2]), copy_graph(backend.get(), shape[0], shape[1], shape[2])};
            for (int round = 0; round < 5; ++round) {
                for (auto & graph : graphs) {
                    for (int replay = 0; replay < 3; ++replay) { graph.run(backend.get(), round*3+replay); }
                }
            }
        }
        std::puts("copy-lifecycle,toy/real-verify/real-prefill/three-buffers/new-UID/capture/update/replay,135,ok");
        produced_rows(backend.get());
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    return 0;
}
