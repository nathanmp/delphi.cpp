#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

using encoder_ptr = std::unique_ptr<clip_ctx, decltype(&clip_free)>;

static encoder_ptr load(const char * path, ggml_backend_dev_t device, bool dynamic) {
    clip_context_params params = {};
    params.use_gpu = device != nullptr;
    params.device = device;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    params.dynamic_requested = dynamic;
    auto res = clip_init(path, params);
    clip_free(res.ctx_a);
    clip_free(res.ctx_gen_a);
    if (!res.ctx_v) {
        throw std::runtime_error("failed to load vision projector");
    }
    return encoder_ptr(res.ctx_v, clip_free);
}

static std::vector<float> encode(clip_ctx * ctx, clip_image_size size) {
    clip_image_f32 image;
    image.set_size(size, false, false);
    std::vector<float> pixels(size_t(size.width) * size.height * 3);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = float(int(i % 251) - 125) / 125.0f;
    }
    image.cpy_buf(pixels);
    std::vector<float> result(size_t(clip_n_output_tokens(ctx, &image)) * clip_n_mmproj_embd(ctx));
    if (!clip_image_encode(ctx, 8, &image, result)) {
        throw std::runtime_error("image encoding failed");
    }
    return result;
}

static void compare(const std::vector<float> & ref, const std::vector<float> & actual) {
    if (ref.size() != actual.size() || ref.empty()) {
        throw std::runtime_error("embedding size mismatch");
    }
    double error = 0, norm = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            throw std::runtime_error("non-finite embedding");
        }
        const double delta = double(ref[i]) - actual[i];
        error += delta * delta;
        norm += double(ref[i]) * ref[i];
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-30));
    std::printf("embedding relative L2 error %.9g\n", relative);
    if (relative > 1e-5) {
        throw std::runtime_error("embedding parity failed");
    }
}

static void check_released(clip_ctx * ctx, ggml_backend_dev_t device) {
    if (clip_dynamic_device(ctx) != device || clip_get_mem_usage(ctx)[device] != 0) {
        throw std::runtime_error("dynamic encoder retained device allocations");
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s projector.gguf DEVICE\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    llama_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    const auto device = ggml_backend_dev_by_name(argv[2]);
    if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr, "an explicit GPU is required\n");
        return 1;
    }
    try {
        std::vector<clip_image_size> sizes;
        std::vector<std::vector<float>> gpu_ref, cpu_ref;
        set_env("LLAMA_MMPROJ_DYNAMIC_GPU_MIB", nullptr);
        {
            auto ctx = load(argv[1], device, false);
            if (clip_dynamic_device(ctx.get())) {
                throw std::runtime_error("resident encoder is dynamic");
            }
            const auto & hp = *clip_get_hparams(ctx.get());
            const int unit = hp.patch_size * hp.n_merge;
            // Include growth and a non-square grid, aligned to this encoder's patch/merge geometry.
            sizes = {{unit * 2, unit * 2}, {unit * 4, unit * 4}, {unit * 4, unit * 2}};
            for (auto size : sizes) {
                gpu_ref.push_back(encode(ctx.get(), size));
            }
        }
        {
            auto ctx = load(argv[1], nullptr, true);
            if (clip_dynamic_device(ctx.get())) {
                throw std::runtime_error("CPU encoder is dynamic");
            }
            for (auto size : sizes) {
                cpu_ref.push_back(encode(ctx.get(), size));
            }
        }
        auto ctx = load(argv[1], device, true);
        check_released(ctx.get(), device);
        size_t free_after_first = 0, total = 0;
        for (int repeat = 0; repeat < 3; ++repeat) {
            for (size_t i = 0; i < sizes.size(); ++i) {
                compare(gpu_ref[i], encode(ctx.get(), sizes[i]));
                check_released(ctx.get(), device);
            }
            size_t free = 0;
            ggml_backend_dev_memory(device, &free, &total);
            if (repeat == 0) {
                free_after_first = free;
            } else if (free + (64ULL << 20) < free_after_first) {
                throw std::runtime_error("device memory grew across repeated encodings");
            }
        }
        set_env("LLAMA_MMPROJ_DYNAMIC_GPU_MIB", "0");
        for (size_t i = 0; i < sizes.size(); ++i) {
            compare(cpu_ref[i], encode(ctx.get(), sizes[i]));
            check_released(ctx.get(), device);
        }
        set_env("LLAMA_MMPROJ_DYNAMIC_GPU_MIB", "invalid");
        bool failed = false;
        try {
            encode(ctx.get(), sizes[0]);
        } catch (const std::runtime_error &) {
            failed = true;
        }
        if (!failed) {
            throw std::runtime_error("invalid GPU budget was accepted");
        }
        check_released(ctx.get(), device);
        set_env("LLAMA_MMPROJ_DYNAMIC_GPU_MIB", nullptr);
        compare(gpu_ref[0], encode(ctx.get(), sizes[0]));
        check_released(ctx.get(), device);
        std::puts("dynamic vision: GPU/CPU parity, release, growth and error recovery passed");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
