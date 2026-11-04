#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--numa") || std::strcmp(argv[2], "tensors")) {
        std::fprintf(stderr, "usage: test-stage-keep --numa tensors\n");
        return 1;
    }
    ggml_backend_load_all();
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * gpu_dev = ggml_backend_dev_by_name("ROCm0");
    if (!cpu_dev || !gpu_dev) {
        std::fprintf(stderr, "CPU and ROCm0 backends are required\n");
        return 1;
    }
    auto numa_init = reinterpret_cast<void (*)(ggml_numa_strategy)>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_dev), "ggml_backend_cpu_numa_init"));
    assert(numa_init);
    numa_init(GGML_NUMA_STRATEGY_TENSORS);
    ggml_backend_ptr gpu(ggml_backend_dev_init(gpu_dev, nullptr));
    ggml_backend_ptr cpu(ggml_backend_dev_init(cpu_dev, nullptr));
    ggml_backend_t backends[] = { gpu.get(), cpu.get() };
    ggml_backend_sched_ptr staged(ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true));
    ggml_backend_sched_ptr oracle(ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true));
    ggml_backend_sched_stage_set_min_tokens(staged.get(), 1);

    constexpr int width = 256, experts = 4, tokens = 1024, used = 2;
    ggml_context_ptr host_meta(ggml_init({size_t(1) << 20, nullptr, true}));
    ggml_context_ptr gpu_meta(ggml_init({size_t(1) << 20, nullptr, true}));
    std::array<ggml_tensor *, 3> host_weights, gpu_weights;
    const char * names[] = { "gate", "up", "down" };
    for (int i = 0; i < 3; ++i) {
        host_weights[i] = ggml_new_tensor_3d(host_meta.get(), GGML_TYPE_F32, width, width, experts);
        ggml_format_name(host_weights[i], "blk.0.ffn_%s_exps.weight", names[i]);
        gpu_weights[i] = ggml_dup_tensor(gpu_meta.get(), host_weights[i]);
    }
    ggml_backend_buffer_ptr host_buffer(ggml_backend_alloc_ctx_tensors(host_meta.get(), cpu.get()));
    ggml_backend_buffer_ptr gpu_buffer(ggml_backend_alloc_ctx_tensors(gpu_meta.get(), gpu.get()));
    assert(host_buffer && gpu_buffer);
    ggml_backend_buffer_set_usage(host_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(gpu_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const auto fill_weights = [&](int generation) {
        std::vector<float> values(width * width * experts);
        for (int i = 0; i < 3; ++i) {
            for (size_t j = 0; j < values.size(); ++j) {
                values[j] = std::sin(float(j + 31*i + 113*generation)) / width;
            }
            ggml_backend_tensor_set(host_weights[i], values.data(), 0, values.size() * sizeof(float));
            ggml_backend_tensor_set(gpu_weights[i], values.data(), 0, values.size() * sizeof(float));
        }
    };

    ggml_context_ptr input_meta(ggml_init({size_t(1) << 20, nullptr, true}));
    auto * input = ggml_new_tensor_3d(input_meta.get(), GGML_TYPE_F32, width, 1, tokens);
    auto * ids = ggml_new_tensor_2d(input_meta.get(), GGML_TYPE_I32, used, tokens);
    ggml_set_input(input);
    ggml_set_input(ids);
    ggml_backend_buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(input_meta.get(), cpu.get()));
    assert(input_buffer);
    const auto run = [&](ggml_backend_sched_t sched, const std::array<ggml_tensor *, 3> & weights, int seed) {
        ggml_backend_sched_reset(sched);
        std::vector<float> x(width * tokens);
        std::vector<int32_t> route(used * tokens);
        for (size_t i = 0; i < x.size(); ++i) { x[i] = std::cos(float(i + seed)); }
        for (size_t i = 0; i < route.size(); ++i) { route[i] = (i + seed) % experts; }
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
        ggml_context_ptr meta(ggml_init({size_t(1) << 20, nullptr, true}));
        auto * graph = ggml_new_graph_custom(meta.get(), 128, false);
        ggml_tensor * output = nullptr;
        for (auto * weight : weights) {
            auto * mm = ggml_mul_mat_id(meta.get(), weight, input, ids);
            ggml_backend_sched_set_tensor_backend(sched, mm, gpu.get());
            output = output ? ggml_add(meta.get(), output, mm) : mm;
        }
        ggml_build_forward_expand(graph, output);
        assert(ggml_backend_sched_alloc_graph(sched, graph));
        assert(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> result(ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
        return result;
    };
    const auto jobs = [&] {
        int64_t cpu_us;
        uint64_t count;
        ggml_backend_sched_stage_stats(staged.get(), &cpu_us, &count);
        return count;
    };
    for (int generation = 0; generation < 2; ++generation) {
        fill_weights(generation);
        ggml_backend_sched_stage_keep(staged.get(), true);
        for (int rep = 0; rep < 4; ++rep) {
            const auto expected = run(oracle.get(), gpu_weights, rep);
            const auto actual = run(staged.get(), host_weights, rep);
            double error = 0, energy = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                error += std::pow(double(actual[i]) - expected[i], 2);
                energy += std::pow(double(expected[i]), 2);
            }
            assert(error / energy < 1e-10);
            std::printf("generation %d, graph %d: uploads %llu, NMSE %.3g\n",
                    generation, rep, (unsigned long long) jobs(), error / energy);
            assert(jobs() == uint64_t(3 * (generation + 1)));
        }
        ggml_backend_sched_stage_keep(staged.get(), false);
    }
    std::puts("stage keep: 8 changed-input graphs, two weight generations, six uploads, GPU parity passed");
}
