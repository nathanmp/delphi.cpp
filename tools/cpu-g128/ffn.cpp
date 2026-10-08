// Complete CPU routed-FFN replay with production NUMA placement and a persistent thread pool.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void load_input(const std::string & path, ggml_tensor * tensor) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes(ggml_nbytes(tensor));
    file.read(bytes.data(), bytes.size());
    require(size_t(file.gcount()) == bytes.size(), "short fixture input");
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
}

int main(int argc, char ** argv) try {
    require(argc == 9, "usage: cpu-g128-ffn GGUF LAYER MANIFEST TOPK THREADS REPEATS ITERATIONS OUTPUT_DIR");
    const int layer = std::stoi(argv[2]), topk = std::stoi(argv[4]), threads = std::stoi(argv[5]);
    const int repeats = std::stoi(argv[6]), iterations = std::stoi(argv[7]);
    require(topk > 0 && topk <= 16 && threads > 0 && repeats > 0 && iterations > 0, "invalid counts");
    const char * bindir = std::getenv("DELPHI_COMPARE_BINDIR");
    require(bindir, "set DELPHI_COMPARE_BINDIR to the tested build/bin");
    ggml_backend_load_all_from_path(bindir);
    llama_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(dev, "CPU device missing");
    auto reg = ggml_backend_dev_backend_reg(dev);
    auto numa_buft = reinterpret_cast<ggml_backend_buffer_type_t (*)()>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_buffer_type"));
    auto numa_load = reinterpret_cast<bool (*)(ggml_tensor *, const void *, size_t)>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_load_expert_tensor"));
    auto set_threads = reinterpret_cast<void (*)(ggml_backend_t, int)>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    auto new_pool = reinterpret_cast<ggml_threadpool_t (*)(ggml_threadpool_params *)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
    auto free_pool = reinterpret_cast<void (*)(ggml_threadpool_t)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
    auto set_pool = reinterpret_cast<void (*)(ggml_backend_t, ggml_threadpool_t)>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
    require(numa_buft && numa_load && set_threads && new_pool && free_pool && set_pool, "CPU NUMA or thread-pool hooks missing");
    auto backend = ggml_backend_dev_init(dev, nullptr);
    require(backend, "CPU backend init failed");
    set_threads(backend, threads);
    auto pool_params = ggml_threadpool_params_default(threads);
    auto pool = new_pool(&pool_params);
    require(pool, "thread pool init failed");
    set_pool(backend, pool);

    ggml_context * metadata = nullptr;
    auto uf = gguf_init_from_file(argv[1], {true, &metadata});
    require(uf && metadata, "GGUF metadata missing");
    const int fd = open(argv[1], O_RDONLY);
    struct stat info {};
    require(fd >= 0 && fstat(fd, &info) == 0, "GGUF open failed");
    auto mapping = mmap(nullptr, info.st_size, PROT_READ, MAP_SHARED, fd, 0);
    require(mapping != MAP_FAILED, "GGUF mmap failed");
    auto wc = ggml_init({16 * 1024 * 1024, nullptr, true});
    ggml_tensor * weights[3] = {};
    const char * projections[] = {"gate", "up", "down"};
    size_t offsets[3] = {};
    for (int p = 0; p < 3; ++p) {
        const std::string name = "blk." + std::to_string(layer) + ".ffn_" + projections[p] + "_exps.weight";
        auto src = ggml_get_tensor(metadata, name.c_str());
        require(src && src->ne[3] == 1, "expert tensor missing");
        weights[p] = ggml_dup_tensor(wc, src);
        ggml_set_name(weights[p], name.c_str());
        weights[p]->flags |= GGML_TENSOR_FLAG_NUMA_EXPERT;
        offsets[p] = gguf_get_data_offset(uf) + gguf_get_tensor_offset(uf, gguf_find_tensor(uf, name.c_str()));
    }
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, numa_buft());
    require(wb, "NUMA weight allocation failed");
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int p = 0; p < 3; ++p) {
        require(offsets[p] <= size_t(info.st_size) && ggml_nbytes(weights[p]) <= size_t(info.st_size) - offsets[p], "GGUF tensor out of bounds");
        require(numa_load(weights[p], static_cast<const char *>(mapping) + offsets[p], ggml_nbytes(weights[p])), "NUMA first-touch load failed");
    }
    munmap(mapping, info.st_size);
    close(fd);
    const int64_t h = weights[0]->ne[0], experts = weights[0]->ne[2];
    require(experts >= topk && weights[1]->ne[0] == h && weights[1]->ne[1] == weights[0]->ne[1] &&
        weights[2]->ne[0] == weights[0]->ne[1] && weights[2]->ne[1] == h, "FFN shape mismatch");

    std::ifstream manifest(argv[3]);
    require(bool(manifest), "manifest unavailable");
    int width;
    std::string dir;
    int cases = 0;
    while (manifest >> width >> dir) {
        require(width > 0, "invalid fixture width");
        auto ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        auto graph = ggml_new_graph_custom(ctx, 256, false);
        auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, h, 1, width);
        auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, width);
        auto rw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, topk, width);
        ggml_set_input(x); ggml_set_input(ids); ggml_set_input(rw);
        auto up = ggml_mul_mat_id(ctx, weights[1], x, ids);
        auto gate = ggml_mul_mat_id(ctx, weights[0], x, ids);
        const char * clamp = std::getenv("DELPHI_COMPARE_CLAMP");
        auto hidden = clamp ? ggml_swiglu_clamp(ctx, gate, up, std::stof(clamp)) : ggml_swiglu_split(ctx, gate, up);
        auto down = ggml_mul_mat_id(ctx, weights[2], hidden, ids);
        auto weighted = ggml_mul(ctx, down, rw);
        auto out = ggml_view_2d(ctx, weighted, h, width, weighted->nb[2], 0);
        for (int slot = 1; slot < topk; ++slot) {
            out = ggml_add(ctx, out, ggml_view_2d(ctx, weighted, h, width, weighted->nb[2], slot * weighted->nb[1]));
        }
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer, "graph allocation failed");
        load_input(dir + "/input.f32", x);
        load_input(dir + "/ids.i32", ids);
        load_input(dir + "/router.f32", rw);
        const auto run = [&] { require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "FFN execution failed"); };
        for (int i = 0; i < 8; ++i) { run(); }
        std::vector<float> result(h * width);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        const std::string output = std::string(argv[8]) + "/w" + std::to_string(width) + ".f32";
        std::ofstream file(output, std::ios::binary);
        file.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
        require(bool(file), "output write failed");
        std::cout << "{\"width\":" << width << ",\"types\":[";
        for (int p = 0; p < 3; ++p) { std::cout << (p ? "," : "") << '"' << ggml_type_name(weights[p]->type) << '"'; }
        std::cout << "],\"threads\":" << threads << ",\"iterations\":" << iterations << ",\"wall_us\":[";
        for (int r = 0; r < repeats; ++r) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) { run(); }
            const auto stop = std::chrono::steady_clock::now();
            std::cout << (r ? "," : "") << std::chrono::duration<double, std::micro>(stop - start).count() / iterations;
        }
        std::cout << "]}" << std::endl;
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ++cases;
    }
    require(manifest.eof() && cases > 0, "invalid or empty manifest");
    ggml_backend_buffer_free(wb);
    set_pool(backend, nullptr);
    free_pool(pool);
    ggml_backend_free(backend);
    ggml_free(wc); ggml_free(metadata); gguf_free(uf);
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
