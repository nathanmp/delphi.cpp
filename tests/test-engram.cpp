#include "llama-engram.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// Compare the independent reader against the ordinary GET_ROWS CPU graph, including
// duplicate/reordered IDs, overlapping jobs, caller-ID reuse and reader teardown.
static bool test_type(ggml_type type, int n_readers) {
    constexpr int width = 256;
    constexpr int height = 83;
    ggml_init_params params = { 1 << 20, nullptr, true };
    auto * ctx = ggml_init(params);
    auto * table = ggml_new_tensor_2d(ctx, type, width, height);
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 79);
    auto * out = ggml_get_rows(ctx, table, ids);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto * cpu = ggml_backend_cpu_init();
    auto pool_params = ggml_threadpool_params_default(4);
    auto * pool = ggml_threadpool_new(&pool_params);
    ggml_backend_cpu_set_threadpool(cpu, pool);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);

    std::vector<float> values(width*height);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = sinf((float) i*0.13f);
    }
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(table, values.data(), 0, values.size()*sizeof(float));
    } else {
        ggml_get_type_traits(type)->from_float_ref(values.data(), table->data, values.size());
    }
    bool ok = true;
    llama_engram_reader::job_ptr last;
    std::vector<float> actual(width*ids->ne[0]);
    {
        llama_engram_reader reader(n_readers);
        for (int rep = 0; rep < 8; ++rep) {
            std::vector<int32_t> rows(ids->ne[0]);
            for (size_t i = 0; i < rows.size(); ++i) {
                rows[i] = (i*19 + rep*7) % height;
            }
            rows[5] = rows[1];
            ggml_backend_tensor_set(ids, rows.data(), 0, rows.size()*sizeof(int32_t));
            auto first = reader.submit(table, rows);
            last = reader.submit(table, rows);
            std::fill(rows.begin(), rows.end(), -1);
            ok &= ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS;
            llama_engram_reader::collect(first, actual.data(), actual.size()*sizeof(float));
            ok &= memcmp(actual.data(), out->data, actual.size()*sizeof(float)) == 0;
            llama_engram_reader::collect(last, actual.data(), actual.size()*sizeof(float));
            ok &= memcmp(actual.data(), out->data, actual.size()*sizeof(float)) == 0;
        }
        // Teardown must drain even an unconsumed job, before the model table is freed.
        last = reader.submit(table, std::vector<int32_t>(ids->ne[0], 0));
    }
    llama_engram_reader::collect(last, actual.data(), actual.size()*sizeof(float));
    for (int i = width; i < (int) actual.size(); ++i) {
        ok &= actual[i] == actual[i % width];
    }
    printf("%s readers=%d: %s\n", ggml_type_name(type), n_readers, ok ? "OK" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_threadpool_free(pool);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    bool ok = true;
    for (int n : { 1, 4, 8 }) {
        for (auto type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 }) {
            ok &= test_type(type, n);
        }
    }
    return ok ? 0 : 1;
}
