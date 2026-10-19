#include "server-tune.h"

#include "arg.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include <dirent.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static std::string tune_scalar_str(const json & v) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "1" : "0";
    }
    return v.dump();
}

bool server_tune_parse(const json & body, server_tune_request & req, std::string & err) {
    if (!body.is_object()) {
        err = "the body must be an object";
        return false;
    }
    // JSON key -> CLI option
    static const std::map<std::string, std::string> keys = {
        { "ctx_size",           "-c"   },
        { "batch_size",         "-b"   },
        { "ubatch_size",        "-ub"  },
        { "cache_type_k",       "-ctk" },
        { "cache_type_v",       "-ctv" },
        { "flash_attn",         "-fa"  },
        { "threads",            "-t"   },
        { "threads_batch",      "-tb"  },
        { "cpu_mask",           "-C"   },
        { "cpu_range",          "-Cr"  },
        { "cpu_mask_batch",     "-Cb"  },
        { "cpu_range_batch",    "-Crb" },
        { "ubatch_size_draft",  "-ubd" },
        { "threads_draft",      "-td"  },
        { "threads_batch_draft","-tbd" },
    };
    for (const auto & [key, val] : body.items()) {
        if (key == "env") {
            if (!val.is_object()) {
                err = "env must be an object";
                return false;
            }
            for (const auto & [name, v] : val.items()) {
                if (name.empty() || name.find('=') != std::string::npos) {
                    err = "invalid environment name '" + name + "'";
                    return false;
                }
                if (v.is_null()) {
                    req.env[name] = std::nullopt;
                } else if (v.is_string() || v.is_number() || v.is_boolean()) {
                    req.env[name] = tune_scalar_str(v);
                } else {
                    err = "the value of env." + name + " must be a string, a number, a boolean or null";
                    return false;
                }
            }
        } else if (key == "n_outputs_max") {
            if (!val.is_number_integer() || val.get<int>() < 0) {
                err = "n_outputs_max must be a non-negative integer";
                return false;
            }
            req.n_outputs_max = val.get<int>();
        } else if (key == "spec_args") {
            if (!val.is_array()) {
                err = "spec_args must be an array";
                return false;
            }
            for (const auto & a : val) {
                req.spec_args.push_back(tune_scalar_str(a));
            }
        } else {
            auto it = keys.find(key);
            if (it == keys.end()) {
                err = "unknown key '" + key + "'";
                return false;
            }
            if (val.is_null() || val.is_object() || val.is_array()) {
                err = "the value of " + key + " must be a string, a number or a boolean";
                return false;
            }
            std::string s = tune_scalar_str(val);
            if (key == "flash_attn" && val.is_boolean()) {
                s = val.get<bool>() ? "on" : "off";
            }
            req.args.push_back(it->second);
            req.args.push_back(s);
            req.cpu_mask_given            |= key.rfind("cpu_", 0) == 0 && key.find("batch") == std::string::npos;
            req.threads_given             |= key == "threads" || key.rfind("cpu_", 0) == 0;
            req.threads_batch_given       |= key == "threads_batch" || key.rfind("cpu_mask_batch", 0) == 0 || key.rfind("cpu_range_batch", 0) == 0;
            req.draft_threads_given       |= key == "threads_draft";
            req.draft_threads_batch_given |= key == "threads_batch_draft";
            req.ctx_size_given            |= key == "ctx_size";
        }
    }
    return true;
}

static void tune_clear_mask(common_cpu_params & cp) {
    memset(cp.cpumask, 0, sizeof(cp.cpumask));
    cp.mask_valid = false;
}

bool server_tune_apply(const common_params & cur, int32_t n_ctx_cur, const server_tune_request & req,
                       common_params & out, std::string & err) {
    out = cur;
    // the context that the first load fitted is the ceiling: -c can only go down
    out.n_ctx = n_ctx_cur;

    // a CPU mask in the request replaces the old one
    if (req.cpu_mask_given) {
        tune_clear_mask(out.cpuparams);
    }
    for (const auto & a : req.args) {
        if (a == "-Cb" || a == "-Crb") {
            tune_clear_mask(out.cpuparams_batch);
            break;
        }
    }

    // draft threads that follow the main threads (as set at launch) keep following them
    const bool draft_follows       = cur.speculative.draft.cpuparams.n_threads       == cur.cpuparams.n_threads;
    const bool draft_batch_follows = cur.speculative.draft.cpuparams_batch.n_threads == cur.cpuparams_batch.n_threads;
    const bool batch_follows       = cur.cpuparams_batch.n_threads == cur.cpuparams.n_threads;

    try {
        common_params_apply_ctx_args(out, LLAMA_EXAMPLE_SERVER, req.args);
        if (!req.spec_args.empty()) {
            common_params_apply_spec_args(out, LLAMA_EXAMPLE_SERVER, req.spec_args);
            common_speculative_params_resolve(out.speculative);
        }
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }

    if (req.threads_given && batch_follows && !req.threads_batch_given) {
        out.cpuparams_batch = out.cpuparams;
    }
    if (draft_follows && !req.draft_threads_given) {
        out.speculative.draft.cpuparams = out.cpuparams;
    }
    if (draft_batch_follows && !req.draft_threads_batch_given) {
        out.speculative.draft.cpuparams_batch = out.cpuparams_batch;
    }

    if (req.ctx_size_given && out.n_ctx > n_ctx_cur) {
        err = string_format("ctx_size %d is larger than the context that the server loaded (%d)", out.n_ctx, n_ctx_cur);
        return false;
    }
    if (out.n_ctx <= 0) {
        err = "invalid ctx_size";
        return false;
    }
    if (out.n_ubatch <= 0 || out.n_batch <= 0 || out.cpuparams.n_threads <= 0 || out.cpuparams_batch.n_threads <= 0) {
        err = "invalid batch or thread value";
        return false;
    }
    return true;
}

std::map<std::string, std::optional<std::string>> server_tune_env_set(const std::map<std::string, std::optional<std::string>> & env) {
    std::map<std::string, std::optional<std::string>> prev;
    for (const auto & [name, val] : env) {
        const char * old = getenv(name.c_str());
        prev[name] = old ? std::optional<std::string>(old) : std::nullopt;
        if (val) {
            setenv(name.c_str(), val->c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
    return prev;
}

void server_tune_env_restore(const std::map<std::string, std::optional<std::string>> & prev) {
    for (const auto & [name, val] : prev) {
        if (val) {
            setenv(name.c_str(), val->c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
}

static json tune_cpu_list(const common_cpu_params & cp) {
    if (!cp.mask_valid) {
        return nullptr;
    }
    json l = json::array();
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cp.cpumask[i]) {
            l.push_back(i);
        }
    }
    return l;
}

json server_tune_state(const common_params & params, int32_t n_ctx, const std::map<std::string, std::string> & env) {
    json devs = json::object();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        devs[ggml_backend_dev_name(dev)] = {
            { "used_mib",  (int64_t) ((total - free) >> 20) },
            { "free_mib",  (int64_t) (free >> 20) },
            { "total_mib", (int64_t) (total >> 20) },
        };
    }

    // resident set size and number of threads of the process
    int64_t rss_mib = -1;
    if (FILE * f = fopen("/proc/self/statm", "r")) {
        long pages = 0, res = 0;
        if (fscanf(f, "%ld %ld", &pages, &res) == 2) {
            rss_mib = (int64_t) res * sysconf(_SC_PAGESIZE) >> 20;
        }
        fclose(f);
    }
    int n_task = 0;
    if (DIR * d = opendir("/proc/self/task")) {
        while (const dirent * e = readdir(d)) {
            n_task += e->d_name[0] != '.';
        }
        closedir(d);
    }

    json jenv = json::object();
    for (const auto & [k, v] : env) {
        jenv[k] = v;
    }

    return json {
        { "ctx_size",            n_ctx },
        { "batch_size",          params.n_batch },
        { "ubatch_size",         params.n_ubatch },
        { "cache_type_k",        ggml_type_name(params.cache_type_k) },
        { "cache_type_v",        ggml_type_name(params.cache_type_v) },
        { "flash_attn",          llama_flash_attn_type_name(params.flash_attn_type) },
        { "threads",             params.cpuparams.n_threads },
        { "threads_batch",       params.cpuparams_batch.n_threads },
        { "cpu_mask",            tune_cpu_list(params.cpuparams) },
        { "cpu_mask_batch",      tune_cpu_list(params.cpuparams_batch) },
        { "ubatch_size_draft",   params.speculative.draft.n_ubatch },
        { "threads_draft",       params.speculative.draft.cpuparams.n_threads },
        { "threads_batch_draft", params.speculative.draft.cpuparams_batch.n_threads },
        { "env",                 jenv },
        { "vram",                devs },
        { "rss_mib",             rss_mib },
        { "n_process_threads",   n_task },
    };
}
