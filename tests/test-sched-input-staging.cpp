#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-sched-input.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct upload {
    const void * source;
    std::vector<uint8_t> bytes;
};

static std::vector<upload> uploads;
static decltype(ggml_backend_i::set_tensor_async) upload_impl;

static void observe_upload(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    uploads.push_back({ data, std::vector<uint8_t>((const uint8_t *) data, (const uint8_t *) data + size) });
    upload_impl(backend, tensor, data, offset, size);
}

int main(int argc, char ** argv) {
    if (argc != 4 || std::strcmp(argv[1], "--numa") || std::strcmp(argv[2], "tensors")) {
        std::fprintf(stderr, "usage: test-sched-input-staging --numa tensors DEVICE\n");
        return 1;
    }
    // exercise growth of the staging slots: no initial capacity
    setenv("GGML_SCHED_INPUT_STAGING_MIN_KIB", "0", 1);
    ggml_backend_load_all();
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * gpu_dev = ggml_backend_dev_by_name(argv[3]);
    assert(cpu_dev && gpu_dev);
    auto numa_init = reinterpret_cast<void (*)(ggml_numa_strategy)>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_dev), "ggml_backend_cpu_numa_init"));
    assert(numa_init);
    numa_init(GGML_NUMA_STRATEGY_TENSORS);
    ggml_backend_ptr gpu(ggml_backend_dev_init(gpu_dev, nullptr));
    ggml_backend_ptr cpu(ggml_backend_dev_init(cpu_dev, nullptr));
    ggml_backend_t backends[] = { gpu.get(), cpu.get() };

    for (bool parallel : { false, true }) {
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 128, parallel, true));
        ggml_backend_sched_set_input_staging(sched.get(), true);
        upload_impl = gpu->iface.set_tensor_async;
        gpu->iface.set_tensor_async = observe_upload;
        for (int round = 0; round < 12; ++round) {
            // Rebuild at different sizes, then reuse the same graph. Both submission paths must protect uploads.
            ggml_backend_sched_reset(sched.get());
            const int n = round < 4 ? 32 << round : 4096;
            ggml_context_ptr inputs(ggml_init({16*ggml_tensor_overhead(), nullptr, true}));
            auto * a = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_F32, n);
            auto * b = ggml_dup_tensor(inputs.get(), a);
            ggml_set_input(a);
            ggml_set_input(b);
            ggml_context_ptr meta(ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true}));
            auto * output = ggml_add(meta.get(), a, b);
            ggml_set_output(output);
            auto * graph = ggml_new_graph_custom(meta.get(), 16, false);
            ggml_build_forward_expand(graph, output);
            ggml_backend_sched_set_tensor_backend(sched.get(), a, cpu.get());
            ggml_backend_sched_set_tensor_backend(sched.get(), b, cpu.get());
            ggml_backend_sched_set_tensor_backend(sched.get(), output, gpu.get());
            assert(ggml_backend_sched_alloc_graph(sched.get(), graph));
            for (int replay = 0; replay < 3; ++replay) {
                std::vector<float> x(n, float(round + replay)), y(n, float(2*round + replay));
                ggml_backend_tensor_set(a, x.data(), 0, n*sizeof(float));
                ggml_backend_tensor_set(b, y.data(), 0, n*sizeof(float));
                uploads.clear();
                assert(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
                assert(uploads.size() == 2);
                // Inspect the actual upload source after the caller reuses its data. This is deterministic even if
                // the GPU completes before the overwrite, so a borrowed-pointer implementation cannot pass by timing.
                std::memset(a->data, 0xff, n*sizeof(float));
                std::memset(b->data, 0xff, n*sizeof(float));
                for (const auto & u : uploads) {
                    assert(std::memcmp(u.source, u.bytes.data(), u.bytes.size()) == 0);
                }
                ggml_backend_sched_synchronize(sched.get());
                std::vector<float> result(n);
                ggml_backend_tensor_get(output, result.data(), 0, n*sizeof(float));
                for (float value : result) {
                    assert(value == float(3*round + 2*replay));
                }
            }
            ggml_backend_sched_reset(sched.get());
        }
        gpu->iface.set_tensor_async = upload_impl;
        std::printf("input staging: parallel=%d, growth/rebuild/replay/ring reuse/source overwrite, passed\n", int(parallel));
    }
}
