#pragma once

// Sampled estimate of the quantization damage of a tensor for a set of candidate types.
// The sampling unit is one complete row of one expert: the ggml quantizers never read outside a row, but several of them use whole-row statistics.

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct common_imatrix;

//
// statistics
//

struct qd_estimate {
    double value = 0.0;
    double se    = 0.0; // NaN when the variance cannot be estimated
};

// the samples of one stratum; x is the auxiliary variable, X its population total (X < 0: unknown)
struct qd_stratum_sample {
    double              N = 0.0;
    double              X = -1.0;
    std::vector<double> y;
    std::vector<double> x;
};

// stratified expansion estimate of the population total of y
qd_estimate qd_estimate_total(const std::vector<qd_stratum_sample> & strata);

// combined ratio estimate of R = Y / X
qd_estimate qd_estimate_ratio(const std::vector<qd_stratum_sample> & strata);

// total of y through the combined ratio and the known X; all strata must have X >= 0
qd_estimate qd_estimate_total_ratio(const std::vector<qd_stratum_sample> & strata);

// deterministic 64-bit hash for per-tensor and per-stratum seeds
uint64_t qd_hash(uint64_t seed, const std::string & s);

// FNV-1a: a cheap stable fingerprint for result identities, not a security hash
uint64_t qd_fnv1a(const void * data, size_t n, uint64_t h = 0xcbf29ce484222325ull);

// streaming FNV-1a of a whole file; false if it cannot be read
bool qd_hash_file(const std::string & path, uint64_t & hash, uint64_t & bytes);

// element i of a keyed permutation of [0, n), computed without a table
int64_t qd_feistel(int64_t i, int64_t n, uint64_t key);

//
// quantization types
//

// types that ggml_quantize_chunk() can produce and whose rows can be dequantized
bool qd_type_is_candidate(ggml_type type);

// case-insensitive ggml type name -> type; GGML_TYPE_COUNT if unknown
ggml_type qd_type_from_name(const std::string & name);

//
// source tensors
//

struct qd_tensor_info {
    std::string name;
    ggml_type   type   = GGML_TYPE_F32;
    int         n_dims = 0;
    int64_t     ne[4]  = {1, 1, 1, 1};

    // location in the GGUF (filled by qd_gguf_model::open; -1 for in-memory tensors)
    int      split       = -1;
    uint64_t data_offset = 0;
    uint64_t data_bytes  = 0;

    int64_t n_per_row()       const { return ne[0]; }
    int64_t rows_per_expert() const { return ne[1]; }
    int64_t n_expert()        const { return n_dims >= 3 ? ne[2]*ne[3] : 1; }
    int64_t n_rows()          const { return ne[1]*ne[2]*ne[3]; }
    int64_t n_elements()      const { return ne[0]*ne[1]*ne[2]*ne[3]; }
};

// thread-safe access to the rows of one source tensor, converted to F32
class qd_row_reader {
public:
    virtual ~qd_row_reader() = default;
    virtual const qd_tensor_info & info() const = 0;
    virtual bool read_rows(int64_t row0, int64_t n, float * dst) const = 0;
};

// rows held in memory (tests and synthetic data)
class qd_memory_reader : public qd_row_reader {
public:
    qd_memory_reader(qd_tensor_info info, std::vector<float> data) : info_(std::move(info)), data_(std::move(data)) {}
    const qd_tensor_info & info() const override { return info_; }
    bool read_rows(int64_t row0, int64_t n, float * dst) const override;
private:
    qd_tensor_info     info_;
    std::vector<float> data_;
};

// read-only GGUF model, split-aware; tensor data is read with positioned reads, never loaded as a whole
class qd_gguf_model {
public:
    ~qd_gguf_model();

    bool open(const std::string & path, std::string & err);

    const std::vector<qd_tensor_info> & tensors() const { return tensors_; }
    std::unique_ptr<qd_row_reader> reader(const std::string & name) const;

    std::string arch;
    int64_t     n_expert_used = 0;

    // model identity: FNV-1a over the raw metadata bytes (header, KV and tensor-info sections) of every
    // file, chained in order, plus the total file size. Path-independent, so a moved or copied model
    // keeps its identity; any metadata change (including tensor layout) changes it.
    uint64_t metadata_hash   = 0;
    uint64_t total_file_size = 0;
    int      n_files         = 0;

    uint64_t bytes_read() const;

private:
    friend class qd_gguf_reader;

    struct file {
        FILE *     fp = nullptr;
        std::mutex mtx;
    };

    bool read_bytes(int split, uint64_t offset, size_t n, void * dst) const;

    std::vector<std::unique_ptr<file>> files_;
    std::vector<qd_tensor_info>        tensors_;
    std::vector<int>                   tensor_split_;
    std::vector<uint64_t>              tensor_offset_;
    mutable std::mutex                 stat_mtx_;
    mutable uint64_t                   bytes_read_ = 0;
};

//
// importance
//

enum qd_weighting {
    QD_WEIGHTING_MASS,    // expert weight = routing frequency; the imatrix activation magnitude is kept (the imatrix sums)
    QD_WEIGHTING_COUNT,   // expert weight = routing frequency; the per-expert activation magnitude is normalized away
    QD_WEIGHTING_UNIFORM, // every expert is routed equally often
    QD_WEIGHTING_COUNT_,
};

const char * qd_weighting_name(qd_weighting w);

struct qd_importance {
    bool        has_imatrix = false;
    bool        has_counts  = false; // per-expert token counts are available (GGUF imatrix)
    std::string note;

    // provenance of the routing counts: "none" (no usable imatrix), "imatrix-counts" (per-expert token
    // counts), "imatrix-shared" (one count shared by all experts) or "imatrix-legacy" (no counts stored)
    std::string counts_source = "none";

    // n_expert x n_per_row, the same values llama-quantize gives the quantizer; empty without an imatrix
    std::vector<float> quant_w;

    // n_expert x n_per_row, mean squared activation per input channel; experts without data get the mean profile of the others
    std::vector<float> act;

    std::vector<double>  counts;         // tokens routed to each expert
    std::vector<uint8_t> expert_missing; // no calibration tokens reached the expert
    int64_t              n_missing = 0;

    // per-expert weight for each weighting; the damage total is sum_e w_e * sum_rows sum_j act_ej * err^2
    std::vector<double> w[QD_WEIGHTING_COUNT_];
};

// builds the importance of a tensor from a loaded imatrix; without an entry, act is 1 and has_imatrix is false
qd_importance qd_importance_build(const common_imatrix * imatrix, const qd_tensor_info & t, int64_t n_expert_used);

//
// estimator
//

// how later rounds distribute new rows over the strata
enum qd_allocation {
    QD_ALLOCATION_PROXY,    // Neyman under the noise-proxy model: N_h x RMS of the proxy; known before sampling, so unbiased
    QD_ALLOCATION_OBSERVED, // Neyman on the observed residual spread; biased when there are many small strata
    QD_ALLOCATION_FIRST,    // first-round proportions
};

struct qd_params {
    std::vector<ggml_type> qtypes;

    uint64_t seed            = 1234;
    int64_t  n_initial       = 64;   // units in the first round (raised to cover the per-expert minimum)
    int64_t  n_max           = 4096; // unit budget
    int64_t  max_sample_mib  = 1024; // budget of sampled source data, as F32
    int64_t  min_per_expert  = 1;    // with row statistics: certainty rows per expert (the most important ones); with expert strata or without row statistics: rows per expert stratum
    int      n_levels        = 8;    // importance levels per stratum group (the tensor, or one expert)
    bool     row_stats       = true; // read the full tensor once for per-row statistics (strata and ratio auxiliaries)
    bool     outlier_stratum = true;
    bool     expert_strata   = false; // one stratum group per expert instead of importance levels over the whole tensor
    double   certainty_factor = 8.0; // always measure rows with a large importance share (0 = off)

    double   rse_target      = 0.03; // stop when every active type has rse(R) <= this
    double   rank_z          = 0.0;  // or, when > 0, when every adjacent pair of active types is ordered at this z
    double   dominance_z     = 3.0;
    double   ci_z            = 1.96;
    bool     adaptive        = true; // false: run every round up to n_max
    qd_allocation allocation = QD_ALLOCATION_PROXY;

    qd_weighting weighting   = QD_WEIGHTING_MASS;

    int n_threads = 4;

    std::string dump_rows; // exact runs: CSV of the per-row values, for analysis
};

enum qd_status {
    QD_STATUS_OK,
    QD_STATUS_INCOMPATIBLE,     // row length is not a multiple of the block size
    QD_STATUS_REQUIRES_IMATRIX, // the quantizer cannot run without an imatrix
    QD_STATUS_UNSUPPORTED,      // not a quantization target of ggml_quantize_chunk
    QD_STATUS_FAILED,           // non-finite output
};

const char * qd_status_name(qd_status s);

struct qd_metric {
    qd_estimate total; // absolute distortion
    qd_estimate rel;   // normalized by the signal energy
    double      ci_lo = 0.0;
    double      ci_hi = 0.0;
};

struct qd_expert_result {
    int64_t             n_sampled = 0;
    std::vector<double> rel; // per type: sampled sum(d) / sum(s) of the expert
};

struct qd_round {
    int64_t             n_units = 0;
    std::vector<double> rel;    // per type, primary weighting
    std::vector<double> rel_se;
};

struct qd_type_result {
    ggml_type   type;
    qd_status   status = QD_STATUS_OK;
    std::string note;

    size_t bytes = 0; // full tensor size in this type
    double bpw   = 0.0;

    qd_metric weighted[QD_WEIGHTING_COUNT_]; // activation-weighted distortion D and R
    qd_metric unweighted;                    // sum of squared error and relative squared error
    double    mse = 0.0;

    // exact runs: share of D in the top 1% / 10% of rows by weighted signal energy, and in the top 1% of rows by their own D
    double share_top1_signal  = -1.0;
    double share_top10_signal = -1.0;
    double share_top1_damage  = -1.0;

    int         dominated_by = -1; // index into types, quality/size dominance
    int         rank         = 0;  // by primary R, 1 = least damage
};

struct qd_tensor_result {
    qd_tensor_info info;
    qd_importance  imp;
    std::string    estimator;  // "exact", "ratio" (with row statistics) or "expansion"

    int64_t units_total   = 0;
    int64_t units_sampled = 0;
    int64_t n_strata      = 0;
    int64_t experts_sampled = 0;
    double  signal[QD_WEIGHTING_COUNT_] = {}; // exact with row statistics, estimated otherwise
    double  energy = 0.0;

    double  seconds_scan  = 0.0;
    double  seconds_eval  = 0.0;
    uint64_t bytes_scan   = 0; // source bytes read for the row statistics
    uint64_t bytes_sample = 0; // source bytes read for the sampled rows
    uint64_t seed = 0;

    std::vector<qd_type_result>   types;
    std::vector<qd_round>         rounds;
    std::vector<qd_expert_result> experts;
};

// sampled estimate
qd_tensor_result qd_estimate_tensor(const qd_row_reader & src, const qd_importance & imp, const qd_params & params);

// exact values over every row (validation reference)
qd_tensor_result qd_exact_tensor(const qd_row_reader & src, const qd_importance & imp, const qd_params & params);

// per-row statistics shared by the runs of one tensor (e.g. several seeds)
struct qd_row_stats {
    std::vector<float> s;       // weighted energy with act, no expert weight
    std::vector<float> e;       // sum of squares
    std::vector<float> ns;      // weighted noise proxy: sum_j act_j * max^2 of the 32-block of j
    std::vector<float> ne;      // unweighted noise proxy
    std::vector<float> outlier; // max|w| / rms
};

bool qd_scan_rows(const qd_row_reader & src, const qd_importance & imp, int n_threads, qd_row_stats & out);

qd_tensor_result qd_estimate_tensor(const qd_row_reader & src, const qd_importance & imp, const qd_params & params, const qd_row_stats * stats);

//
// locality check
//

struct qd_locality {
    ggml_type type;
    qd_status status     = QD_STATUS_OK;
    int       rows_tried = 0;
    bool      row_local  = true;  // a row quantized alone == the same row inside a multi-row chunk
    int64_t   span       = 0;     // tested sub-row span
    int       block_local = -1;   // 1: a span quantized alone == its blocks inside the row, 0: differs, -1: not testable (row metadata)
};

std::vector<qd_locality> qd_check_locality(const qd_row_reader & src, const qd_importance & imp, const std::vector<ggml_type> & types, int n_rows, uint64_t seed);

//
// result identity and versioning
//

// bump when the meaning of any JSON field changes; consumers key cached records by this string plus
// the model / calibration identity, the encoder variant, the qtype and the seed.
static constexpr const char * QD_RESULT_SCHEMA = "delphi.quant-damage/1";

// the encoder that produced the damage values. "ggml_quantize_chunk" is the ordinary llama.cpp encoder.
// An encoder variant that changes bytes (for example activation feedback) must report a different
// string so its measurements are never mixed with ordinary ones.
static constexpr const char * QD_ENCODER_GGML = "ggml_quantize_chunk";

//
// tie groups
//

// several tensors that must share one output qtype, e.g. the gate and the up projection of a fused FFN.
// A tie constrains the choice, not the measurement: member damages stay separate in the members.
struct qd_tie_group {
    std::string name;
    std::vector<std::string> tensors; // member names; the order matches the members of qd_tie_result
};

// one qtype aggregated over a group. The group R is the sum of member D over the sum of member signal:
// a sum of per-tensor proxies, not a joint measurement. Member interactions (e.g. gate and up inside
// a gated unit) are not modeled, and member damages must not be assumed equal.
struct qd_tie_type {
    ggml_type type;
    qd_status status = QD_STATUS_OK; // OK only when every member supports this type
    std::string note;

    size_t bytes = 0; // sum of member full-tensor bytes
    double bpw   = 0.0;

    double    signal[QD_WEIGHTING_COUNT_] = {}; // sum of member signals
    qd_metric weighted[QD_WEIGHTING_COUNT_];    // D total: sum of member totals, se: quadrature sum (independent samples)
    qd_metric unweighted;
};

struct qd_tie_result {
    qd_tie_group                  group;
    std::vector<qd_tensor_result> members;
    std::vector<qd_tie_type>      types; // one per qtype; members must share one qtype order (same qd_params)
};

// aggregate member results of one run into a group record; members.size() must equal group.tensors.size()
qd_tie_result qd_tie_combine(const qd_tie_group & group, const std::vector<qd_tensor_result> & members, double ci_z);
