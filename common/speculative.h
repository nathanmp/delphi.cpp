#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

// Opt-in cycle ledger for observing speculative decoding (handoff 2 of the DFlash2 task). Off unless DELPHI_SPEC_LEDGER=<file> is
// set; then host spans and witnesses are buffered in memory (no printing, no allocation, no device synchronization) and written as
// JSON lines when the speculative state is freed. Timestamps use the execution trace's clock (CLOCK_MONOTONIC_RAW).
enum common_spec_ledger_stage : uint16_t {
    COMMON_SPEC_LEDGER_TGT_DECODE = 1,  // server: llama_decode(ctx_tgt) host call.              a = tokens
    COMMON_SPEC_LEDGER_TGT_SYNC,        // server: llama_synchronize(ctx_tgt) after the decode.  a = tokens
    COMMON_SPEC_LEDGER_PROCESS,         // common_speculative_process, all implementations.     a = tokens
    COMMON_SPEC_LEDGER_FEAT_GATHER,     // DFlash: target layer rows copied into the inject batch. a = rows, b = layers, x = bytes
    COMMON_SPEC_LEDGER_DFT_INGEST,      // DFlash: llama_decode(ctx_dft, injected features).    a = rows, b = first pos, c = last pos
    COMMON_SPEC_LEDGER_DRAFT,           // common_speculative_draft, all implementations.       a = candidates returned
    COMMON_SPEC_LEDGER_DFT_BLOCK,       // DFlash: llama_decode(ctx_dft, noise block).          a = rows, b = requested L
    COMMON_SPEC_LEDGER_DFT_LATTICE,     // DFlash2: llama_get_embeddings_nextn(ctx_dft) (synchronizes). a = rows, x = physical bytes, c = logical bytes
    COMMON_SPEC_LEDGER_DFT_WALK,        // DFlash2: selector lattice walk on the host.          a = candidates, b = block rows
    COMMON_SPEC_LEDGER_DFT_ROLLBACK,    // server: draft KV removal after the draft block.     a = first removed pos
    COMMON_SPEC_LEDGER_ACCEPT,          // server: target sampling and acceptance of a draft.  a = draft size, b = accepted drafts
    COMMON_SPEC_LEDGER_KV_ROLLBACK,     // server: KV removal after acceptance.                a = first removed pos
};

enum common_spec_ledger_witness_kind : uint16_t {
    COMMON_SPEC_WITNESS_CANDIDATES = 1, // ids = candidate tokens in order, a = rows of the block, b = cutoff position (0: none), c = p_min * 1e6
    COMMON_SPEC_WITNESS_KV,             // a = target pos_max, b = draft pos_max, c = point (1 after draft rollback, 2 after ingest, 3 after acceptance)
    COMMON_SPEC_WITNESS_ACCEPTED,       // ids = accepted tokens (draft + bonus), a = draft size, b = accepted drafts
};

bool     common_spec_ledger_on();
uint64_t common_spec_ledger_now();     // 0 when off
void     common_spec_ledger_span(common_spec_ledger_stage stage, uint64_t t0_ns,
                                 uint32_t a = 0, uint32_t b = 0, uint32_t c = 0, uint64_t x = 0); // the span ends now
void     common_spec_ledger_witness(common_spec_ledger_witness_kind kind, const int32_t * ids, uint32_t n,
                                    int64_t a = 0, int64_t b = 0, int64_t c = 0);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// like common_speculative_process, for a batch without outputs whose features nobody else reads (an intermediate
// prompt batch): the draft ingests it on a worker thread while the caller continues with the next target decode
bool common_speculative_process_async(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model; id_next is the target's
// token after the accepted ones (the cost-aware depth compares it with its unverified shadow token)
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted,
                               llama_token id_next = LLAMA_TOKEN_NULL);

// report a partial acceptance that the context could not apply (the server restores a checkpoint and replays the
// accepted tokens; the replay round's accept is not counted by the adaptive draft depth)
void common_speculative_accept_partial(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
// the same without waiting for the asynchronous ingest, for a job queued behind it (common_speculative_run_ctx)
bool common_speculative_get_state_now(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// cost-aware draft depth (--spec-draft-adaptive-mode cost): the widest verify batch it needs (0: not in use)
int32_t common_speculative_cost_width(const common_speculative * spec);

// measure the target's verify time of batches of 1..n_width rows (seconds, index = rows; [0] unused); the sequence
// must be unused and is cleared after each batch
std::vector<double> common_speculative_calibrate(llama_context * ctx_tgt, llama_seq_id seq_id, int32_t n_width);

// hand the calibrated verify times to the cost-aware implementations
void common_speculative_set_verify_times(common_speculative * spec, const std::vector<double> & t_verify);

// statistics and adaptive state of one implementation
struct common_speculative_impl_info {
    std::string type;
    int32_t     n_max          = 0;
    std::string adaptive;          // "", "bucket" or "cost"
    int32_t     n_min_adaptive = 0;

    size_t n_gen_drafts = 0;
    size_t n_acc_drafts = 0;
    size_t n_gen_tokens = 0;
    size_t n_acc_tokens = 0;
    size_t n_call_accept = 0;
    std::vector<double> acc_rate_per_pos; // accepted tokens per verified draft, per position

    std::vector<int32_t> depth;           // [n_seq] current adaptive depth

    // cost mode
    std::vector<double> t_verify_ms;      // [rows] calibrated verify time
    std::vector<double> t_round_ms;       // [n] round time estimate per draft length
    std::vector<double> n_round;          // [n] live samples per draft length
    double              t_step_ms = 0.0;  // draft time per token
    double              t_switch_ms = 0.0; // extra time of a change of depth
    std::vector<double> accept_rate;      // [k - 1] conditional acceptance over all requests
};

std::vector<common_speculative_impl_info> common_speculative_get_info(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
