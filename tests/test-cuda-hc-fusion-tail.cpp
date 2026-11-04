#include "ggml-cpp.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4 || std::strcmp(argv[1], "--numa") || std::strcmp(argv[2], "tensors")) {
        std::fprintf(stderr, "usage: test-cuda-hc-fusion-tail --numa tensors DEVICE\n");
        return 1;
    }
    ggml_backend_load_all();
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * dev = ggml_backend_dev_by_name(argv[3]);
    assert(cpu && dev);
    auto numa_init = reinterpret_cast<void (*)(ggml_numa_strategy)>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu), "ggml_backend_cpu_numa_init"));
    assert(numa_init);
    numa_init(GGML_NUMA_STRATEGY_TENSORS);
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    assert(backend);

    for (int length : {1, 2, 3, 5}) {
        ggml_context_ptr ctx(ggml_init({size_t(1) << 20, nullptr, true}));
        auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 32);
        auto * scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        auto * base = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        ggml_tensor * y;
        if (length == 5) {
            // The pending HC mix can end the graph before a PRE or POST consumer exists.
            y = ggml_mul(ctx.get(), x, scale);
            y = ggml_add(ctx.get(), y, ggml_view_1d(ctx.get(), base, 4, 0));
            y = ggml_scale(ctx.get(), ggml_sigmoid(ctx.get(), y), 2.0f);
        } else {
            y = ggml_scale(ctx.get(), x, 1.3f);
            if (length >= 2) { y = ggml_sigmoid(ctx.get(), y); }
            if (length >= 3) { y = ggml_scale(ctx.get(), y, 2.0f); }
        }
        ggml_set_output(y);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        assert(buffer);
        const float s = 1.3f;
        const float b[] = {-0.4f, -0.1f, 0.2f, 0.5f};
        std::vector<float> input(128), output(128);
        for (size_t i = 0; i < input.size(); ++i) { input[i] = std::sin(float(i)); }
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(scale, &s, 0, sizeof(s));
        ggml_backend_tensor_set(base, b, 0, sizeof(b));
        assert(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(y, output.data(), 0, ggml_nbytes(y));
        for (size_t i = 0; i < output.size(); ++i) {
            float expected = input[i] * s;
            if (length == 5) { expected += b[i % 4]; }
            if (length >= 2) { expected = 1.0f / (1.0f + std::exp(-expected)); }
            if (length >= 3) { expected *= 2.0f; }
            assert(std::isfinite(output[i]) && std::abs(expected - output[i]) < 1e-5f);
        }
        std::printf("HC fusion tail %d: passed\n", length);
        std::fflush(stdout);
    }
}
