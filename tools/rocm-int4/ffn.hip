// HIP G128 integration harness adapted from tools/marlin-compare/ggml-ffn.cu at 03d5ccb3b. Test-only private graph access; no inference dispatch changes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "gguf.h"
#include "llama.h"
#include "ggml-cuda/common.cuh"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void load_bytes(const std::string & path, size_t offset, ggml_tensor * tensor) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "input file unavailable");
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

int main(int argc, char ** argv) try {
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
    if (const char * bindir = std::getenv("DELPHI_COMPARE_BINDIR")) {
        ggml_backend_load_all_from_path(bindir);
    } else {
        ggml_backend_load_all();
    }
    llama_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    const char * device = std::getenv("DELPHI_COMPARE_DEVICE");
    auto dev = ggml_backend_dev_by_name(device ? device : "ROCm1");
    require(dev != nullptr, "ROCm1 not found");
    auto weight_backend = ggml_backend_dev_init(dev, nullptr);
    require(weight_backend != nullptr, "weight backend init failed");
    ggml_context * metadata = nullptr;
    auto uf = gguf_init_from_file(model.c_str(), {true, &metadata});
    require(uf != nullptr && metadata != nullptr, "GGUF metadata load failed");
    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    auto wc = ggml_init(params);
    ggml_tensor * weights[3] = {}, * scales[3] = {};
    const char * projections[] = {"gate", "up", "down"};
    std::vector<std::pair<ggml_tensor *, size_t>> reads;
    for (int p = 0; p < 3; ++p) {
        std::string base = "blk." + std::to_string(layer) + ".ffn_" + projections[p] + "_exps";
        for (int s = 0; s < 2; ++s) {
            std::string name = base + (s ? ".scale" : ".weight");
            auto src = ggml_get_tensor(metadata, name.c_str());
            if (s && src == nullptr) { continue; }
            require(src != nullptr, "expert tensor missing");
            auto dst = ggml_dup_tensor(wc, src);
            ggml_set_name(dst, name.c_str());
            (s ? scales[p] : weights[p]) = dst;
            reads.emplace_back(dst, gguf_get_data_offset(uf) + gguf_get_tensor_offset(uf, gguf_find_tensor(uf, name.c_str())));
        }
        require(ggml_is_quantized(weights[p]->type), "expected quantized expert weights");
    }
    const int64_t k = weights[0]->ne[0], n = weights[0]->ne[1], e = weights[0]->ne[2], topk = std::getenv("DELPHI_COMPARE_TOPK") ? std::stoi(std::getenv("DELPHI_COMPARE_TOPK")) : 10;
    require(k % 128 == 0 && n % 128 == 0 && e >= topk && topk <= 16, "unsupported dimensions");
    auto wb = ggml_backend_alloc_ctx_tensors(wc, weight_backend);
    require(wb != nullptr, "weight allocation failed");
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto & read : reads) { load_bytes(model, read.second, read.first); }
    std::cerr << "weights loaded: " << ggml_backend_buffer_get_size(wb) << " bytes\n";

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
    const bool rotate = std::getenv("DELPHI_COMPARE_ROTATE") != nullptr;
    struct retained_fixture {
        ggml_backend_t backend, cpu;
        ggml_backend_sched_t sched;
        ggml_backend_buffer_t inputs_buffer;
        ggml_context * inputs, * ctx;
        hipGraphExec_t executable;
    };
    std::vector<retained_fixture> retained;
    for (const auto & item : fixtures) {
        width = item.first;
        dir = item.second;
        require(width > 0, "invalid fixture width");
        auto backend = ggml_backend_dev_init(dev, nullptr);
        require(backend != nullptr, "fixture backend init failed");
        auto ctx = ggml_init(params);
        auto graph = ggml_new_graph_custom(ctx, 256, false);
        // Persistent GPU inputs keep direct replay independent of scheduler copies.
        auto inputs = ggml_init(params);
        auto x = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, k, 1, width);
        auto ids = ggml_new_tensor_2d(inputs, GGML_TYPE_I32, topk, width);
        auto rw = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, 1, topk, width);
        ggml_set_name(x, "fixture-input"); ggml_set_input(x);
        ggml_set_name(ids, "fixture-ids"); ggml_set_input(ids);
        ggml_set_name(rw, "fixture-router-weights"); ggml_set_input(rw);
        auto ib = ggml_backend_alloc_ctx_tensors(inputs, backend);
        require(ib != nullptr, "input allocation failed");
        auto mm = [&](int p, ggml_tensor * in) {
            auto out = ggml_mul_mat_id(ctx, weights[p], in, ids);
            ggml_set_name(out, (std::string("ffn_moe_") + projections[p] + "-3").c_str());
            if (scales[p] == nullptr) { return out; }
            auto s = ggml_reshape_3d(ctx, scales[p], 1, e, 1);
            s = ggml_repeat_4d(ctx, s, 1, e, width, 1);
            s = ggml_get_rows(ctx, s, ids);
            out = ggml_mul(ctx, out, s);
            ggml_set_name(out, (std::string("ffn_moe_") + projections[p] + "_scaled-3").c_str());
            return out;
        };
        // Follow the model builder's up-before-gate and ordered slot reduction.
        auto up = mm(1, x);
        auto gate = mm(0, x);
        auto act = std::getenv("DELPHI_COMPARE_CLAMP") ? ggml_swiglu_clamp(ctx, gate, up, std::stof(std::getenv("DELPHI_COMPARE_CLAMP"))) : ggml_swiglu_split(ctx, gate, up);
        ggml_set_name(act, "ffn_moe_swiglu_limited-3");
        auto down = mm(2, act);
        auto weighted = ggml_mul(ctx, down, rw);
        ggml_set_name(weighted, "ffn_moe_weighted-3");
        ggml_build_forward_expand(graph, weighted);
        ggml_tensor * slots[16];
        for (int i = 0; i < topk; ++i) {
            slots[i] = ggml_view_2d(ctx, weighted, k, width, weighted->nb[2], i * weighted->nb[1]);
            ggml_build_forward_expand(graph, slots[i]);
        }
        auto out = slots[0];
        for (int i = 1; i < topk; ++i) {
            out = ggml_add(ctx, out, slots[i]);
            ggml_build_forward_expand(graph, out);
        }
        ggml_set_name(out, "ffn_moe_out-3"); ggml_set_output(out);
        // Use the scheduler so CUDA fusion can declare extended allocation lifetimes.
        auto cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        require(cpu != nullptr, "CPU backend init failed");
        ggml_backend_t backends[] = {backend, cpu};
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, 256, false, true);
        require(ggml_backend_sched_alloc_graph(sched, graph), "graph allocation failed");
        load_bytes(dir + "/input.f32", 0, x);
        load_bytes(dir + "/ids.i32", 0, ids);
        load_bytes(dir + "/router.f32", 0, rw);
        for (int i = 0; i < 8; ++i) {
            require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "warmup failed");
        }
        // This test-only access avoids comparing Python/C++ graph submission overhead.
        // The private structure must match the backend binary's source and build flags.
        auto cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
        require(cuda_ctx->cuda_graphs.size() == 1, "expected one captured FFN graph");
        auto executable = cuda_ctx->cuda_graphs.begin()->second->instance;
        require(executable != nullptr, "FFN was not captured");
        auto stream = cuda_ctx->streams[cuda_ctx->device][0];
        size_t node_count = 0;
        CUDA_CHECK(cudaGraphGetNodes(cuda_ctx->cuda_graphs.begin()->second->graph, nullptr, &node_count));
        std::cerr << "captured nodes: " << node_count << "\n";
        std::ofstream census(dir + "/ggml-kernels.tsv");
        std::vector<hipGraphNode_t> nodes(node_count);
        CUDA_CHECK(hipGraphGetNodes(cuda_ctx->cuda_graphs.begin()->second->graph, nodes.data(), &node_count));
        census << "name\tgrid\tblock\tshared_bytes\n";
        for (auto node : nodes) {
            hipGraphNodeType type;
            CUDA_CHECK(hipGraphNodeGetType(node, &type));
            if (type != hipGraphNodeTypeKernel) { continue; }
            hipKernelNodeParams p;
            CUDA_CHECK(hipGraphKernelNodeGetParams(node, &p));
            auto name = hipKernelNameRefByPtr(p.func, stream);
            census << (name ? name : "unknown") << '\t' << p.gridDim.x << ',' << p.gridDim.y << ',' << p.gridDim.z
                   << '\t' << p.blockDim.x << ',' << p.blockDim.y << ',' << p.blockDim.z << '\t' << p.sharedMemBytes << '\n';
        }

        std::vector<float> result(k * width), replay(k * width);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ggml_backend_tensor_get(out, replay.data(), 0, replay.size() * sizeof(float));
        require(result == replay, "direct replay differs from backend replay");
        if (std::getenv("DELPHI_COMPARE_LIFECYCLE") && weights[0]->type == GGML_TYPE_Q4_G128) {
            auto reg = ggml_backend_dev_backend_reg(dev);
            auto mode = reinterpret_cast<void (*)(int)>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_set_mode"));
            require(mode != nullptr, "mode hook missing");
            mode(0);
            auto fallback = ggml_backend_dev_init(dev, nullptr);
            auto fc = ggml_init(params);
            auto fg = ggml_new_graph_custom(fc, 8, false);
            auto fo = ggml_mul_mat_id(fc, weights[0], x, ids);
            ggml_build_forward_expand(fg, fo);
            auto fb = ggml_backend_alloc_ctx_tensors(fc, fallback);
            require(fb != nullptr, "fallback allocation failed");
            for (int i = 0; i < 4; ++i) { require(ggml_backend_graph_compute(fallback, fg) == GGML_STATUS_SUCCESS, "fallback failed"); }
            std::vector<float> fref(ggml_nelements(fo)), fvalue(fref.size());
            ggml_backend_tensor_get(fo, fref.data(), 0, fref.size() * sizeof(float));
            std::vector<char> block(66), original(66);
            std::ifstream source(model, std::ios::binary);
            // Readback restores only this matrix; the other two must remain usable in their packed state.
            const auto offset = gguf_get_data_offset(uf) + gguf_get_tensor_offset(uf, gguf_find_tensor(uf, weights[0]->name));
            source.seekg(offset); source.read(original.data(), original.size());
            ggml_backend_tensor_get(weights[0], block.data(), 0, block.size());
            require(block == original, "packed-to-canonical readback differs from file");
            require(ggml_backend_graph_compute(fallback, fg) == GGML_STATUS_SUCCESS, "fallback after restore failed");
            ggml_backend_tensor_get(fo, fvalue.data(), 0, fvalue.size() * sizeof(float));
            require(fvalue == fref, "captured fallback did not observe canonical layout");
            mode(1);
            require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "repack failed");
            ggml_backend_tensor_get(out, replay.data(), 0, replay.size() * sizeof(float));
            require(replay == result, "repack changed output");
            require(ggml_backend_graph_compute(fallback, fg) == GGML_STATUS_SUCCESS, "fallback after repack failed");
            ggml_backend_tensor_get(fo, fvalue.data(), 0, fvalue.size() * sizeof(float));
            require(fvalue == fref, "captured fallback did not observe packed layout");
            // A partial write must preserve every byte outside its range, including scales and other matrices.
            std::vector<char> patch(64, char(0x88));
            ggml_backend_tensor_set(weights[0], patch.data(), 2, patch.size());
            ggml_backend_tensor_get(weights[0], block.data(), 0, block.size());
            require(block[0] == original[0] && block[1] == original[1] &&
                std::equal(patch.begin(), patch.end(), block.begin() + 2), "partial write corrupted scale or codes");
            require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "graph after partial write failed");
            ggml_backend_tensor_set(weights[0], original.data(), 0, original.size());
            require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "graph after repair failed");
            ggml_backend_tensor_get(out, replay.data(), 0, replay.size() * sizeof(float));
            require(replay == result, "partial write/repair changed untouched weights");
            ggml_backend_buffer_free(fb); ggml_backend_free(fallback); ggml_free(fc);
            std::cerr << "lifecycle: canonical readback, captured fallback in both layouts, partial write/repack passed\n";
        }
        std::ofstream output(dir + "/ggml-output.f32", std::ios::binary);
        output.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
        output.close();
        if (rotate) {
            retained.push_back({backend, cpu, sched, ib, inputs, ctx, executable});
            continue;
        }
        auto reg = ggml_backend_dev_backend_reg(dev);
        auto stats_fn = reinterpret_cast<void (*)(int64_t *)>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_int4_get_stats"));
        int64_t stats[16] = {};
        if (stats_fn) { stats_fn(stats); }
        std::cerr << "int4 stats: packed=" << stats[0] << " packs=" << stats[1] << " bytes=" << stats[3]
                  << " regions=" << stats[4] << " scratch=" << stats[5] << " fallback=" << stats[6] << " chunked=" << stats[15] << "\n";
        if (pace_us > 0) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < burst_replays; ++j) { CUDA_CHECK(cudaGraphLaunch(executable, stream)); }
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::this_thread::sleep_for(std::chrono::microseconds(pace_us));
            }
        } else {
            for (int i = 0; i < 4096; ++i) { CUDA_CHECK(cudaGraphLaunch(executable, stream)); }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaEvent_t start, stop;
        CUDA_CHECK(hipEventCreate(&start)); CUDA_CHECK(hipEventCreate(&stop));
        std::ofstream json_file;
        if (manifest) { json_file.open(dir + "/ggml.json"); require(bool(json_file), "JSON output unavailable"); }
        std::ostream & json = manifest ? json_file : std::cout;
        json << "{\"engine\":\"ggml\",\"width\":" << width << ",\"nodes\":" << node_count << ",\"weight_types\":[";
        for (int p = 0; p < 3; ++p) {
            if (p) { json << ','; }
            json << '"' << ggml_type_name(weights[p]->type) << '"';
        }
        json << "],\"device_us\":[";
        for (int r = 0; r < repeats; ++r) {
            float ms = 0;
            for (int i = 0; i < (pace_us > 0 ? iterations : 1); ++i) {
                CUDA_CHECK(cudaEventRecord(start, stream));
                for (int j = 0; j < (pace_us > 0 ? burst_replays : iterations); ++j) {
                    CUDA_CHECK(cudaGraphLaunch(executable, stream));
                }
                CUDA_CHECK(cudaEventRecord(stop, stream)); CUDA_CHECK(cudaEventSynchronize(stop));
                float burst_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&burst_ms, start, stop));
                ms += burst_ms;
                if (pace_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(pace_us)); }
            }
            if (r) { json << ','; } json << 1000.0 * ms / (iterations * burst_replays);
        }
        json << "],\"iterations\":" << iterations << ",\"burst_replays\":" << burst_replays << ",\"pace_us\":" << pace_us << "}\n";
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        ggml_backend_synchronize(backend);
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_backend_free(cpu);
        ggml_backend_buffer_free(ib); ggml_free(inputs);
        ggml_free(ctx);
    }
    if (rotate) {
        require(retained.size() > 1, "rotation needs multiple fixtures");
        hipStream_t stream;
        CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        auto replay_all = [&]() { for (const auto & f : retained) { CUDA_CHECK(hipGraphLaunch(f.executable, stream)); } };
        for (int i = 0; i < 5; ++i) { replay_all(); }
        CUDA_CHECK(hipStreamSynchronize(stream));
        hipEvent_t start, stop;
        CUDA_CHECK(hipEventCreate(&start)); CUDA_CHECK(hipEventCreate(&stop));
        std::cout << "{\"engine\":\"ggml\",\"rotating_fixtures\":" << retained.size() << ",\"width\":1,\"device_us\":[";
        for (int r = 0; r < repeats; ++r) {
            float total = 0;
            for (int burst = 0; burst < iterations; ++burst) {
                CUDA_CHECK(hipEventRecord(start, stream));
                for (int i = 0; i < burst_replays; ++i) { replay_all(); }
                CUDA_CHECK(hipEventRecord(stop, stream)); CUDA_CHECK(hipEventSynchronize(stop));
                float ms; CUDA_CHECK(hipEventElapsedTime(&ms, start, stop)); total += ms;
                if (pace_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(pace_us)); }
            }
            std::cout << (r ? "," : "") << 1000.0 * total / (iterations * burst_replays * retained.size());
        }
        std::cout << "]}\n";
        CUDA_CHECK(hipEventDestroy(start)); CUDA_CHECK(hipEventDestroy(stop)); CUDA_CHECK(hipStreamDestroy(stream));
        for (const auto & f : retained) {
            ggml_backend_sched_free(f.sched); ggml_backend_free(f.backend); ggml_backend_free(f.cpu);
            ggml_backend_buffer_free(f.inputs_buffer); ggml_free(f.inputs); ggml_free(f.ctx);
        }
    }
    ggml_backend_buffer_free(wb);
    ggml_backend_free(weight_backend);
    ggml_free(wc); ggml_free(metadata); gguf_free(uf);
    return 0;
} catch (const std::exception & ex) {
    std::cerr << ex.what() << '\n'; return 1;
}
