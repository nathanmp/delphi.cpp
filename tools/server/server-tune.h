#pragma once

#include "common.h"
#include "server-common.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

// delphi: helpers of the auto-tuner endpoints (--tune-endpoints), see docs/delphi/usage/tune-endpoints.md.
// POST /tune/context changes the context parameters of a running server by making the contexts again on the same weights.

struct server_tune_request {
    // the CLI options to apply with common_params_apply_ctx_args()
    std::vector<std::string> args;
    // --spec-* options to apply with common_params_apply_spec_args()
    std::vector<std::string> spec_args;
    // environment: name -> value, nullopt unsets
    std::map<std::string, std::optional<std::string>> env;
    bool cpu_mask_given   = false;
    bool threads_given    = false;
    bool threads_batch_given = false;
    bool draft_threads_given = false;
    bool draft_threads_batch_given = false;
    bool ctx_size_given   = false;
    // outputs per ubatch that the context allows at least (POST /tune/kld needs one per token of a batch); -1 = unchanged
    int  n_outputs_max    = -1;
};

// parse the body of POST /tune/context; false and err for an unknown key or a wrong type
bool server_tune_parse(const json & body, server_tune_request & req, std::string & err);

// the new parameters: the current ones with the request applied; n_ctx_cur is the context size now (never exceeded)
bool server_tune_apply(const common_params & cur, int32_t n_ctx_cur, const server_tune_request & req,
                       common_params & out, std::string & err);

// set the environment of a request; returns the previous values for server_tune_env_restore()
std::map<std::string, std::optional<std::string>> server_tune_env_set(const std::map<std::string, std::optional<std::string>> & env);
void server_tune_env_restore(const std::map<std::string, std::optional<std::string>> & prev);

// the values of GET /tune/context and the response of POST /tune/context: parameters, the environment that requests set,
// the memory per device, and the process resources (RSS, number of threads)
json server_tune_state(const common_params & params, int32_t n_ctx, const std::map<std::string, std::string> & env);
