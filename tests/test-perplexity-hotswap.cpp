#include "hotswap.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;
#define CHECK(x) do { if (!(x)) { throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #x); } } while (0)

static void write_tensor(gguf_context * file, ggml_context * ctx, const char * name,
                         ggml_type type, int rows, float offset) {
    const int cols = 256;
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, cols, rows);
    ggml_set_name(t, name);
    std::vector<float> data(cols * rows);
    for (size_t i = 0; i < data.size(); ++i) { data[i] = offset + 0.1f * (float) (i % 17) - 0.8f; }
    ggml_quantize_chunk(type, data.data(), t->data, 0, rows, cols, nullptr);
    gguf_add_tensor(file, t);
}

static void write_patch(const fs::path & path, ggml_type type, int rows, float offset, bool invalid = false) {
    ggml_context * ctx = ggml_init({ 1 << 20, nullptr, false });
    gguf_context * file = gguf_init_empty();
    write_tensor(file, ctx, "test.weight", type, rows, offset);
    if (invalid) { write_tensor(file, ctx, "unused.weight", type, rows, offset); }
    CHECK(gguf_write_to_file(file, path.string().c_str(), false));
    gguf_free(file);
    ggml_free(ctx);
}

static std::vector<uint8_t> contents(ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

int main() try {
    common_init();
    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    const auto dir = fs::temp_directory_path() / ("delphi-hotswap-unit-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    const auto base = dir / "base.gguf", patch = dir / "patch.gguf", ctrl = dir / "control", status = dir / "status";
    write_patch(base, GGML_TYPE_F32, 2, 0);

    ggml_context * ctx = ggml_init({ 4*ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 2);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 2);
    ggml_set_name(a, "test.weight");
    ggml_set_name(b, "test.weight");
    auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(cpu);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(cpu));
    CHECK(buffer);
    std::vector<float> original(512, 0.25f);
    ggml_backend_tensor_set(a, original.data(), 0, original.size()*sizeof(float));
    ggml_backend_tensor_set(b, original.data(), 0, original.size()*sizeof(float));
    const auto bytes = contents(a);
    void * baseline_data = a->data;
    {
        perplexity_hotswap hot;
        std::string error;
        CHECK(hot.init({ {"test.weight", a}, {"test.weight", b}, {"test.weight", a} }, base.string(), error));
        CHECK(hot.reload("", error) == 0);
        write_patch(patch, GGML_TYPE_Q4_0, 2, 0);
        CHECK(hot.reload(patch.string(), error) == 2);
        CHECK(a->type == GGML_TYPE_Q4_0 && b->type == GGML_TYPE_Q4_0);
        CHECK(contents(a) == contents(b));
        hot.commit();
        const auto first = contents(a);
        write_patch(patch, GGML_TYPE_Q4_0, 2, 1);
        CHECK(hot.reload(patch.string(), error) == 2);
        CHECK(contents(a) != first);
        hot.commit();
        const auto current = contents(a);

        write_patch(patch, GGML_TYPE_Q2_K, 3, 0);
        CHECK(hot.reload(patch.string(), error) == -1);
        CHECK(a->type == GGML_TYPE_Q4_0 && contents(a) == current);
        write_patch(patch, GGML_TYPE_Q2_K, 2, 0, true);
        CHECK(hot.reload(patch.string(), error) == -1);
        CHECK(a->type == GGML_TYPE_Q4_0 && contents(a) == current);
        write_patch(patch, GGML_TYPE_Q2_K, 2, 0);
        fs::resize_file(patch, fs::file_size(patch)-64);
        CHECK(hot.reload(patch.string(), error) == -1);
        CHECK(contents(a) == current);

        write_patch(patch, GGML_TYPE_Q2_K, 2, 0);
        CHECK(hot.reload(patch.string(), error) == 2);
        CHECK(a->type == GGML_TYPE_Q2_K);
        hot.rollback();
        CHECK(a->type == GGML_TYPE_Q4_0 && contents(a) == current);
        CHECK(hot.restore(error) == 2);
        CHECK(a->type == GGML_TYPE_F32 && a->data == baseline_data && contents(a) == bytes);
        hot.rollback();
        CHECK(a->type == GGML_TYPE_Q4_0 && contents(a) == current);
        CHECK(hot.restore(error) == 2);
        hot.commit();
        CHECK(hot.restore(error) == 0);
        CHECK(a->type == GGML_TYPE_F32 && a->data == baseline_data && contents(a) == bytes);
        write_patch(base, GGML_TYPE_F32, 2, 2);
        fs::last_write_time(base, fs::last_write_time(base) + std::chrono::seconds(1));
        CHECK(hot.reload("", error) == 2);
        CHECK(a->type == GGML_TYPE_F32 && contents(a) != bytes);
        hot.commit();
        CHECK(hot.reload("", error) == 0);
        gguf_context * empty = gguf_init_empty();
        CHECK(gguf_write_to_file(empty, base.string().c_str(), false));
        gguf_free(empty);
        fs::last_write_time(base, fs::last_write_time(base) + std::chrono::seconds(1));
        CHECK(hot.reload("", error) == 0);
        write_patch(base, GGML_TYPE_Q2_K, 2, 0);
        fs::last_write_time(base, fs::last_write_time(base) + std::chrono::seconds(2));
        CHECK(hot.reload("", error) == 2);
        hot.commit();
        CHECK(hot.reload("", error) == 0);
    }
    CHECK(a->type == GGML_TYPE_F32 && a->data == baseline_data && contents(a) == bytes);

    perplexity_hotswap_command command;
    { std::ofstream f(ctrl); f << "reload 17 /tmp/path with spaces.gguf"; }
    CHECK(!perplexity_hotswap_read_command(ctrl.string(), command));
    { std::ofstream f(ctrl, std::ios::app); f << '\n'; }
    CHECK(perplexity_hotswap_read_command(ctrl.string(), command));
    CHECK(command.action == "reload" && command.sequence == 17 && command.path == "/tmp/path with spaces.gguf");
    CHECK(!fs::exists(ctrl));
    { std::ofstream f(ctrl); f << "compute\n"; }
    CHECK(perplexity_hotswap_read_command(ctrl.string(), command));
    CHECK(command.action == "compute" && command.sequence == -1);
    CHECK(perplexity_hotswap_write_status(status.string(), "done", 3, 17));
    std::ifstream f(status);
    std::string state; int iteration, sequence;
    f >> state >> iteration >> sequence;
    CHECK(state == "done" && iteration == 3 && sequence == 17);
    f.close();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    fs::remove_all(dir);
    fprintf(stderr, "hotswap checks passed\n");
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "hotswap test failed: %s\n", error.what());
    return 1;
}
