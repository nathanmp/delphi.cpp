#include "llama.h"
#include "../src/llama-ext.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "mtmd.h"
#include "speculative.h"

#include <algorithm>
#include <cinttypes>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// satisfies -Wmissing-declarations
int llama_fit_params(int argc, char ** argv);

int llama_fit_params(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_PARAMS)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (!params.fit_params_print) {
        const common_params_fit_status status = common_fit_params(params.model.path.c_str(), &mparams, &cparams,
                params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
                nullptr,
                params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
        if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            LOG_ERR("%s: failed to fit CLI arguments to free memory, exiting...\n", __func__);
            exit(1);
        }

        LOG_INF("%s: printing fitted CLI arguments to stdout...\n", __func__);
        common_log_flush(common_log_main());
        printf("-c %" PRIu32 " -ngl %" PRIi32, cparams.n_ctx, mparams.n_gpu_layers);

        size_t nd = llama_max_devices();
        while (nd > 1 && mparams.tensor_split[nd - 1] == 0.0f) {
            nd--;
        }
        if (nd > 1) {
            for (size_t id = 0; id < nd; id++) {
                if (id == 0) {
                    printf(" -ts ");
                }
                printf("%s%" PRIu32, id > 0 ? "," : "", uint32_t(mparams.tensor_split[id]));
            }
        }

        const size_t ntbo = llama_max_tensor_buft_overrides();
        bool any_tbo = false;
        for (size_t itbo = 0; itbo < ntbo && mparams.tensor_buft_overrides[itbo].pattern != nullptr; itbo++) {
            if (itbo == 0) {
                printf(" -ot \"");
            }
            printf("%s%s=%s", itbo > 0 ? "," : "", mparams.tensor_buft_overrides[itbo].pattern, ggml_backend_buft_name(mparams.tensor_buft_overrides[itbo].buft));
            any_tbo = true;
        }
        printf("%s\n", any_tbo ? "\"" : "");
    } else {
        LOG_INF("%s: printing estimated memory in MiB to stdout (device, model, context, compute) ...\n", __func__);
        common_log_flush(common_log_main());

        common_fit_print(params.model.path.c_str(), &mparams, &cparams);

        // with --spec-type draft-mtp the lines above include the MTP weights and the recurrent states of the draft;
        // the MTP draft context of the server (its KV cache and compute buffers, on the weights of the main model)
        // follows as "mtp <device> <context> <compute>"
        const bool spec_mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(),
                COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
        if (spec_mtp && !params.speculative.has_dft()) {
            common_params params_dft = common_base_params_to_speculative(params);
            auto mparams_dft = common_model_params_to_llama(params_dft);
            auto cparams_dft = common_context_params_to_llama(params_dft);
            cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cparams_dft.n_ctx    = cparams.n_ctx;
            cparams_dft.n_rs_seq = 0;
            std::vector<ggml_backend_dev_t> devs;
            uint32_t ngl = 0, nct = 0, nex = 0;
            const auto dmd = common_get_device_memory_data(params.model.path.c_str(), &mparams_dft, &cparams_dft, devs,
                    ngl, nct, nex, GGML_LOG_LEVEL_ERROR);
            for (size_t id = 0; id < devs.size(); id++) {
                printf("mtp %s %zu %zu\n", ggml_backend_dev_name(devs[id]), dmd[id].context/1024/1024, dmd[id].compute/1024/1024);
            }
        }

        // the multimodal projector (weights and worst-case compute buffers) goes to the first device, as in the server; a
        // dynamic encoder ("mmproj-dynamic") needs that memory only while it encodes, when the language model has released
        // its compute buffers
        if (!params.mmproj.path.empty()) {
            mtmd_context_params mp = mtmd_context_params_default();
            mp.use_gpu = params.mmproj_use_gpu;
            mp.device  = params.mmproj_device;
            for (const auto & [dev, size] : mtmd_get_memory_usage(params.mmproj.path.c_str(), mp)) {
                const bool dynamic = params.mmproj_dynamic && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
                printf("%s %s %zu\n", dynamic ? "mmproj-dynamic" : "mmproj", ggml_backend_dev_name(dev), (size + (1 << 20) - 1) / (1 << 20));
            }
        }
    }

    return 0;
}
