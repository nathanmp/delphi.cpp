#include "quant-damage.h"

#include "gguf.h"
#include "imatrix-loader.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <thread>

//
// statistics
//

static const double qd_nan = std::numeric_limits<double>::quiet_NaN();

// sum_h N_h^2 (1 - f_h) s_h^2 / n_h for a per-sample value z; strata with a single sample borrow the pooled relative variance of the others
static double qd_strata_variance(const std::vector<qd_stratum_sample> & strata, const std::function<double(size_t, size_t)> & z) {
    std::vector<double> mean(strata.size(), 0.0);
    std::vector<double> var (strata.size(), qd_nan);
    std::vector<double> ymean(strata.size(), 0.0);

    double pool_var = 0.0;
    double pool_ref = 0.0;
    for (size_t h = 0; h < strata.size(); ++h) {
        const auto & s = strata[h];
        const size_t n = s.y.size();
        if (n == 0) {
            continue;
        }
        double m = 0.0, my = 0.0;
        for (size_t i = 0; i < n; ++i) {
            m  += z(h, i);
            my += s.y[i];
        }
        m  /= n;
        my /= n;
        mean[h]  = m;
        ymean[h] = my;
        if (n >= 2) {
            double v = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double d = z(h, i) - m;
                v += d*d;
            }
            var[h] = v / (n - 1);
            pool_var += var[h];
            pool_ref += my*my;
        }
    }

    const double kappa = pool_ref > 0.0 ? pool_var / pool_ref : (pool_var == 0.0 ? 0.0 : qd_nan);

    double total = 0.0;
    for (size_t h = 0; h < strata.size(); ++h) {
        const auto & s = strata[h];
        const double n = (double) s.y.size();
        if (n == 0 || s.N <= 0) {
            continue;
        }
        const double v = n >= 2 ? var[h] : kappa * ymean[h] * ymean[h];
        const double f = std::min(1.0, n / s.N);
        total += s.N * s.N * (1.0 - f) * v / n;
    }
    return total;
}

static bool qd_strata_complete(const std::vector<qd_stratum_sample> & strata) {
    for (const auto & s : strata) {
        if (s.N > 0 && s.y.empty()) {
            return false;
        }
    }
    return true;
}

qd_estimate qd_estimate_total(const std::vector<qd_stratum_sample> & strata) {
    qd_estimate r;
    if (!qd_strata_complete(strata)) {
        r.value = r.se = qd_nan;
        return r;
    }
    for (const auto & s : strata) {
        if (s.y.empty()) {
            continue;
        }
        r.value += s.N * std::accumulate(s.y.begin(), s.y.end(), 0.0) / s.y.size();
    }
    r.se = std::sqrt(qd_strata_variance(strata, [&](size_t h, size_t i) { return strata[h].y[i]; }));
    return r;
}

// combined ratio; returns R, the residual variance and the estimated X total
static void qd_ratio_parts(const std::vector<qd_stratum_sample> & strata, double & R, double & var_e, double & x_hat) {
    double y_hat = 0.0;
    x_hat = 0.0;
    for (const auto & s : strata) {
        if (s.y.empty()) {
            continue;
        }
        y_hat += s.N * std::accumulate(s.y.begin(), s.y.end(), 0.0) / s.y.size();
        x_hat += s.N * std::accumulate(s.x.begin(), s.x.end(), 0.0) / s.x.size();
    }
    R = x_hat != 0.0 ? y_hat / x_hat : qd_nan;
    const double Rv = R;
    var_e = qd_strata_variance(strata, [&](size_t h, size_t i) { return strata[h].y[i] - Rv * strata[h].x[i]; });
}

qd_estimate qd_estimate_ratio(const std::vector<qd_stratum_sample> & strata) {
    qd_estimate r;
    if (!qd_strata_complete(strata)) {
        r.value = r.se = qd_nan;
        return r;
    }
    double var_e, x_hat;
    qd_ratio_parts(strata, r.value, var_e, x_hat);

    double X = 0.0;
    bool known = true;
    for (const auto & s : strata) {
        known = known && s.X >= 0.0;
        X += s.X;
    }
    const double denom = known ? X : x_hat;
    r.se = denom != 0.0 ? std::sqrt(var_e) / std::fabs(denom) : qd_nan;
    return r;
}

qd_estimate qd_estimate_total_ratio(const std::vector<qd_stratum_sample> & strata) {
    qd_estimate r;
    double X = 0.0;
    for (const auto & s : strata) {
        GGML_ASSERT(s.X >= 0.0);
        X += s.X;
    }
    if (!qd_strata_complete(strata)) {
        r.value = r.se = qd_nan;
        return r;
    }
    double R, var_e, x_hat;
    qd_ratio_parts(strata, R, var_e, x_hat);
    r.value = X == 0.0 ? 0.0 : R * X;
    r.se    = std::sqrt(var_e);
    return r;
}

static uint64_t qd_splitmix64(uint64_t & state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

uint64_t qd_hash(uint64_t seed, const std::string & s) {
    uint64_t h = seed ^ 0x6a09e667f3bcc909ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    uint64_t st = h;
    return qd_splitmix64(st);
}

uint64_t qd_fnv1a(const void * data, size_t n, uint64_t h) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

bool qd_hash_file(const std::string & path, uint64_t & hash, uint64_t & bytes) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        return false;
    }
    uint64_t h = 0xcbf29ce484222325ull;
    std::vector<uint8_t> buf(1 << 20);
    size_t total = 0;
    while (true) {
        const size_t n = fread(buf.data(), 1, buf.size(), fp);
        h = qd_fnv1a(buf.data(), n, h);
        total += n;
        if (n < buf.size()) {
            break;
        }
    }
    const bool ok = !ferror(fp);
    fclose(fp);
    hash  = h;
    bytes = total;
    return ok;
}

// Fisher-Yates with a portable generator, so a seed gives the same sample on every platform
static void qd_permute(std::vector<int64_t> & v, uint64_t seed) {
    uint64_t st = seed;
    for (size_t i = v.size(); i > 1; --i) {
        const size_t j = (size_t) (qd_splitmix64(st) % i);
        std::swap(v[i - 1], v[j]);
    }
}

// keyed bijection of [0, n) without a table: a 4-round Feistel network on the next even power of two, with cycle walking
int64_t qd_feistel(int64_t i, int64_t n, uint64_t key) {
    if (n <= 1) {
        return 0;
    }
    int bits = 0;
    while (bits < 62 && (1ll << bits) < n) {
        bits++;
    }
    bits = std::max(2, bits + (bits & 1));
    const int      half = bits / 2;
    const uint64_t mask = (1ull << half) - 1;
    uint64_t x = (uint64_t) i;
    do {
        uint64_t l = x >> half, r = x & mask;
        for (uint64_t k = 0; k < 4; ++k) {
            uint64_t st = key ^ (k * 0x9e3779b97f4a7c15ull) ^ r;
            const uint64_t f = qd_splitmix64(st) & mask;
            const uint64_t nl = r;
            r = l ^ f;
            l = nl;
        }
        x = (l << half) | r;
    } while (x >= (uint64_t) n);
    return (int64_t) x;
}

static void qd_parallel_for(int64_t n, int n_threads, const std::function<void(int64_t, int)> & fn) {
    n_threads = (int) std::max<int64_t>(1, std::min<int64_t>(n_threads, n));
    if (n_threads == 1) {
        for (int64_t i = 0; i < n; ++i) {
            fn(i, 0);
        }
        return;
    }
    std::atomic<int64_t> next(0);
    std::vector<std::thread> workers;
    for (int t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t]() {
            for (int64_t i = next++; i < n; i = next++) {
                fn(i, t);
            }
        });
    }
    for (auto & w : workers) {
        w.join();
    }
}

//
// quantization types
//

bool qd_type_is_candidate(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return ggml_get_type_traits(type)->to_float != nullptr;
        case GGML_TYPE_F32:
            return true;
        default:
            return false;
    }
}

ggml_type qd_type_from_name(const std::string & name) {
    std::string want = name;
    std::transform(want.begin(), want.end(), want.begin(), ::tolower);
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * tn = ggml_type_name((ggml_type) t);
        if (tn == nullptr) {
            continue;
        }
        std::string have = tn;
        std::transform(have.begin(), have.end(), have.begin(), ::tolower);
        if (have == want) {
            return (ggml_type) t;
        }
    }
    return GGML_TYPE_COUNT;
}

const char * qd_status_name(qd_status s) {
    switch (s) {
        case QD_STATUS_OK:               return "ok";
        case QD_STATUS_INCOMPATIBLE:     return "incompatible_row_size";
        case QD_STATUS_REQUIRES_IMATRIX: return "requires_imatrix";
        case QD_STATUS_UNSUPPORTED:      return "unsupported";
        case QD_STATUS_FAILED:           return "failed";
    }
    return "?";
}

const char * qd_weighting_name(qd_weighting w) {
    switch (w) {
        case QD_WEIGHTING_MASS:    return "mass";
        case QD_WEIGHTING_COUNT:   return "count";
        case QD_WEIGHTING_UNIFORM: return "uniform";
        default:                   return "?";
    }
}

static qd_status qd_type_status(ggml_type type, const qd_tensor_info & t, bool has_imatrix) {
    if (!qd_type_is_candidate(type)) {
        return QD_STATUS_UNSUPPORTED;
    }
    if (t.n_per_row() % ggml_blck_size(type) != 0) {
        return QD_STATUS_INCOMPATIBLE;
    }
    if (ggml_quantize_requires_imatrix(type) && !has_imatrix) {
        return QD_STATUS_REQUIRES_IMATRIX;
    }
    return QD_STATUS_OK;
}

//
// source tensors
//

bool qd_memory_reader::read_rows(int64_t row0, int64_t n, float * dst) const {
    const int64_t k = info_.n_per_row();
    if (row0 < 0 || row0 + n > info_.n_rows()) {
        return false;
    }
    memcpy(dst, data_.data() + row0*k, n*k*sizeof(float));
    return true;
}

class qd_gguf_reader : public qd_row_reader {
public:
    qd_gguf_reader(const qd_gguf_model & model, size_t idx) : model_(model), idx_(idx) {}

    const qd_tensor_info & info() const override { return model_.tensors_[idx_]; }

    bool read_rows(int64_t row0, int64_t n, float * dst) const override {
        const qd_tensor_info & t = info();
        const int64_t k        = t.n_per_row();
        const size_t  row_size = ggml_row_size(t.type, k);
        if (row0 < 0 || row0 + n > t.n_rows()) {
            return false;
        }
        const uint64_t offset = model_.tensor_offset_[idx_] + (uint64_t) row0 * row_size;
        if (t.type == GGML_TYPE_F32) {
            return model_.read_bytes(model_.tensor_split_[idx_], offset, n*row_size, dst);
        }
        std::vector<uint8_t> buf(n*row_size);
        if (!model_.read_bytes(model_.tensor_split_[idx_], offset, buf.size(), buf.data())) {
            return false;
        }
        const auto * traits = ggml_get_type_traits(t.type);
        for (int64_t r = 0; r < n; ++r) {
            traits->to_float(buf.data() + r*row_size, dst + r*k, k);
        }
        return true;
    }

private:
    const qd_gguf_model & model_;
    size_t                idx_;
};

qd_gguf_model::~qd_gguf_model() {
    for (auto & f : files_) {
        if (f->fp) {
            fclose(f->fp);
        }
    }
}

static int64_t qd_gguf_get_int(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return -1;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8 (ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default:               return -1;
    }
}

bool qd_gguf_model::open(const std::string & path, std::string & err) {
    std::vector<std::string> paths = { path };

    for (size_t split = 0; split < paths.size(); ++split) {
        ggml_context * meta = nullptr;
        gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
        gguf_context * ctx = gguf_init_from_file(paths[split].c_str(), params);
        if (!ctx) {
            err = "failed to read GGUF metadata from " + paths[split];
            return false;
        }

        if (split == 0) {
            const int64_t n_split = qd_gguf_get_int(ctx, "split.count");
            if (n_split > 1) {
                char prefix[4096];
                if (!llama_split_prefix(prefix, sizeof(prefix), path.c_str(), 0, (int) n_split)) {
                    err = "a split model must be opened by its first file: " + path;
                    gguf_free(ctx);
                    ggml_free(meta);
                    return false;
                }
                for (int i = 1; i < n_split; ++i) {
                    char p[4096];
                    llama_split_path(p, sizeof(p), prefix, i, (int) n_split);
                    paths.push_back(p);
                }
            }
            const int64_t id = gguf_find_key(ctx, "general.architecture");
            if (id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING) {
                arch = gguf_get_val_str(ctx, id);
                n_expert_used = std::max<int64_t>(0, qd_gguf_get_int(ctx, arch + ".expert_used_count"));
            }
        }

        auto f = std::make_unique<file>();
        f->fp = fopen(paths[split].c_str(), "rb");
        if (!f->fp) {
            err = "failed to open " + paths[split];
            gguf_free(ctx);
            ggml_free(meta);
            return false;
        }

        const size_t data_offset = gguf_get_data_offset(ctx);

        // identity: hash the raw metadata bytes (header, KV and tensor-info) of every file, and add the
        // file size; reading them again is cheap next to a scan, and it keys machine-readable results
        {
            std::vector<uint8_t> buf(1 << 16);
            fseek(f->fp, 0, SEEK_SET);
            size_t left = data_offset;
            while (left > 0) {
                const size_t n = fread(buf.data(), 1, std::min(left, buf.size()), f->fp);
                if (n == 0) {
                    err = "failed to read the metadata of " + paths[split];
                    fclose(f->fp);
                    f->fp = nullptr;
                    files_.push_back(std::move(f));
                    gguf_free(ctx);
                    ggml_free(meta);
                    return false;
                }
                metadata_hash = qd_fnv1a(buf.data(), n, metadata_hash);
                left -= n;
            }
            if (fseek(f->fp, 0, SEEK_END) == 0) {
                total_file_size += (uint64_t) ftello(f->fp);
            }
        }
        n_files = (int) (split + 1);
        files_.push_back(std::move(f));

        for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            const ggml_tensor * t = ggml_get_tensor(meta, name);
            qd_tensor_info info;
            info.name   = name;
            info.type   = t->type;
            info.n_dims = ggml_n_dims(t);
            for (int d = 0; d < 4; ++d) {
                info.ne[d] = t->ne[d];
            }
            info.split       = (int) split;
            info.data_offset = data_offset + gguf_get_tensor_offset(ctx, i);
            info.data_bytes  = ggml_row_size(info.type, info.ne[0]) * info.ne[1] * info.ne[2] * info.ne[3];
            tensors_.push_back(info);
            tensor_split_.push_back((int) split);
            tensor_offset_.push_back(info.data_offset);
        }

        gguf_free(ctx);
        ggml_free(meta);
    }
    return true;
}

std::unique_ptr<qd_row_reader> qd_gguf_model::reader(const std::string & name) const {
    for (size_t i = 0; i < tensors_.size(); ++i) {
        if (tensors_[i].name == name) {
            if (tensors_[i].type != GGML_TYPE_F32 && ggml_get_type_traits(tensors_[i].type)->to_float == nullptr) {
                return nullptr;
            }
            return std::make_unique<qd_gguf_reader>(*this, i);
        }
    }
    return nullptr;
}

bool qd_gguf_model::read_bytes(int split, uint64_t offset, size_t n, void * dst) const {
    auto & f = *files_[split];
    std::lock_guard<std::mutex> lock(f.mtx);
#ifdef _WIN32
    if (_fseeki64(f.fp, (__int64) offset, SEEK_SET) != 0) {
#else
    if (fseeko(f.fp, (off_t) offset, SEEK_SET) != 0) {
#endif
        return false;
    }
    if (fread(dst, 1, n, f.fp) != n) {
        return false;
    }
    std::lock_guard<std::mutex> lock_stat(stat_mtx_);
    bytes_read_ += n;
    return true;
}

uint64_t qd_gguf_model::bytes_read() const {
    std::lock_guard<std::mutex> lock(stat_mtx_);
    return bytes_read_;
}

//
// importance
//

qd_importance qd_importance_build(const common_imatrix * imatrix, const qd_tensor_info & t, int64_t n_expert_used) {
    qd_importance imp;
    const int64_t k   = t.n_per_row();
    const int64_t nex = t.n_expert();

    imp.counts.assign(nex, 0.0);
    imp.expert_missing.assign(nex, 0);

    const common_imatrix_entry * entry = nullptr;
    if (imatrix) {
        auto it = imatrix->entries.find(t.name);
        if (it != imatrix->entries.end()) {
            entry = &it->second;
        } else {
            imp.note = "no imatrix entry";
        }
    } else {
        imp.note = "no imatrix";
    }
    // a stacked tensor that is not routed shares one imatrix row
    common_imatrix_entry broadcast;
    if (entry && nex > 1 && (int64_t) entry->sums.size() == k && entry->counts.size() <= 1) {
        broadcast.counts = entry->counts;
        for (int64_t e = 0; e < nex; ++e) {
            broadcast.sums.insert(broadcast.sums.end(), entry->sums.begin(), entry->sums.end());
        }
        entry = &broadcast;
    }
    if (entry && (int64_t) entry->sums.size() != k*nex) {
        imp.note = "imatrix size " + std::to_string(entry->sums.size()) + " does not match " + std::to_string(k*nex);
        entry = nullptr;
    }
    if (entry && !imatrix->is_legacy && entry->counts.size() != (size_t) nex && entry->counts.size() != 1) {
        imp.note = "imatrix has " + std::to_string(entry->counts.size()) + " counts for " + std::to_string(nex) + " experts";
        entry = nullptr;
    }

    if (!entry) {
        imp.act.assign(k*nex, 1.0f);
        for (auto & w : imp.w) {
            w.assign(nex, 1.0);
        }
        imp.counts.assign(nex, 1.0);
        return imp;
    }

    imp.has_imatrix = true;
    imp.quant_w.assign(k*nex, 1.0f);

    // same normalization as llama-quantize: mean squared activation per token, 1 for experts without tokens
    if (!imatrix->is_legacy) {
        imp.has_counts = true;
        const bool shared = entry->counts.size() == 1;
        for (int64_t e = 0; e < nex; ++e) {
            const double c = (double) entry->counts[shared ? 0 : e];
            imp.counts[e] = c;
            if (c > 0) {
                for (int64_t j = 0; j < k; ++j) {
                    imp.quant_w[e*k + j] = entry->sums[e*k + j] / (float) c;
                }
            } else {
                imp.expert_missing[e] = 1;
                imp.n_missing++;
            }
        }
        imp.counts_source = shared ? "imatrix-shared" : "imatrix-counts";
        if (shared && nex > 1) {
            imp.has_counts = false;
        }
    } else {
        const int64_t ncall = entry->counts.empty() ? 0 : entry->counts[0];
        for (int64_t i = 0; i < k*nex; ++i) {
            imp.quant_w[i] = ncall > 0 ? entry->sums[i] / ncall : entry->sums[i];
        }
        imp.counts.assign(nex, 1.0);
        imp.counts_source = "imatrix-legacy";
        imp.note = "legacy imatrix: no per-expert token counts";
    }

    imp.act = imp.quant_w;

    // per-expert activation magnitude; missing experts get the mean channel profile of the covered ones at their median magnitude
    std::vector<double> m(nex, 0.0);
    std::vector<double> profile(k, 0.0);
    std::vector<double> covered_m;
    for (int64_t e = 0; e < nex; ++e) {
        if (imp.expert_missing[e]) {
            continue;
        }
        double s = 0.0;
        for (int64_t j = 0; j < k; ++j) {
            s += imp.act[e*k + j];
        }
        m[e] = s / k;
        covered_m.push_back(m[e]);
        if (m[e] > 0) {
            for (int64_t j = 0; j < k; ++j) {
                profile[j] += imp.act[e*k + j] / m[e];
            }
        }
    }
    if (covered_m.empty()) {
        imp.note = "imatrix entry has no tokens";
        imp.has_imatrix = false;
        imp.act.assign(k*nex, 1.0f);
        for (auto & w : imp.w) {
            w.assign(nex, 1.0);
        }
        imp.counts.assign(nex, 1.0);
        return imp;
    }
    for (auto & p : profile) {
        p /= covered_m.size();
    }
    std::vector<double> sorted_m = covered_m;
    std::sort(sorted_m.begin(), sorted_m.end());
    const double median_m = sorted_m[sorted_m.size()/2];
    double mean_m = 0.0;
    for (double v : covered_m) {
        mean_m += v;
    }
    mean_m /= covered_m.size();

    double min_count = std::numeric_limits<double>::infinity();
    double sum_count = 0.0;
    for (int64_t e = 0; e < nex; ++e) {
        if (imp.expert_missing[e]) {
            for (int64_t j = 0; j < k; ++j) {
                imp.act[e*k + j] = (float) (profile[j] * median_m);
            }
            m[e] = median_m;
        } else {
            min_count = std::min(min_count, imp.counts[e]);
            sum_count += imp.counts[e];
        }
    }

    for (auto & w : imp.w) {
        w.assign(nex, 1.0);
    }
    if (nex > 1) {
        // D is then the expected squared output error per token, summed over the routed experts
        const double n_used = n_expert_used > 0 ? (double) n_expert_used : 1.0;
        const double n_tok  = imp.has_counts ? sum_count / n_used : 0.0;
        if (n_expert_used <= 0) {
            imp.note += std::string(imp.note.empty() ? "" : "; ") + "expert_used_count unknown, weights assume 1 routed expert";
        }
        for (int64_t e = 0; e < nex; ++e) {
            const double c = imp.expert_missing[e] ? min_count : imp.counts[e];
            const double f = imp.has_counts ? c / n_tok : n_used / nex;
            imp.w[QD_WEIGHTING_MASS][e]    = f;
            imp.w[QD_WEIGHTING_COUNT][e]   = m[e] > 0 ? f * mean_m / m[e] : f;
            imp.w[QD_WEIGHTING_UNIFORM][e] = n_used / nex;
        }
        if (imp.n_missing > 0) {
            imp.note += std::string(imp.note.empty() ? "" : "; ") + std::to_string(imp.n_missing) + " experts without tokens use the smallest observed count and the mean profile";
        }
    }
    return imp;
}

//
// row evaluation
//

struct qd_row_eval {
    double s  = 0.0; // sum act w^2
    double e  = 0.0; // sum w^2
    double ns = 0.0; // weighted noise proxy
    double ne = 0.0; // unweighted noise proxy
    std::vector<double> d; // per type: sum act err^2
    std::vector<double> u; // per type: sum err^2
    bool finite = true;
};

struct qd_scratch {
    std::vector<uint8_t> q;
    std::vector<float>   deq;
};

// Quantization error of a block quantizer scales with the square of the block maximum, not with w^2.
// The noise proxy sum_j act_j * max^2(block of j) therefore tracks the damage of every type far better than the signal,
// in particular when one input channel holds most of the activation mass. It is only an auxiliary variable.
static constexpr int64_t QD_PROXY_BLOCK = 32;

static void qd_row_aux(const float * x, int64_t k, const float * act, double & s, double & e, double & ns, double & ne, double & mx) {
    s = e = ns = ne = mx = 0.0;
    for (int64_t b0 = 0; b0 < k; b0 += QD_PROXY_BLOCK) {
        const int64_t b1 = std::min(k, b0 + QD_PROXY_BLOCK);
        double bmax = 0.0, bact = 0.0;
        for (int64_t j = b0; j < b1; ++j) {
            const double v2 = (double) x[j] * x[j];
            s    += act[j] * v2;
            e    += v2;
            bact += act[j];
            bmax  = std::max(bmax, v2);
        }
        ns += bact * bmax;
        ne += (b1 - b0) * bmax;
        mx  = std::max(mx, bmax);
    }
    mx = std::sqrt(mx);
}

static void qd_eval_row(const float * x, int64_t k, const float * act, const float * quant_w,
                        const std::vector<ggml_type> & types, const std::vector<qd_status> & status,
                        qd_scratch & scratch, qd_row_eval & out) {
    double mx;
    qd_row_aux(x, k, act, out.s, out.e, out.ns, out.ne, mx);
    out.d.assign(types.size(), 0.0);
    out.u.assign(types.size(), 0.0);
    scratch.deq.resize(k);
    for (size_t q = 0; q < types.size(); ++q) {
        if (status[q] != QD_STATUS_OK) {
            continue;
        }
        const ggml_type type = types[q];
        const float * deq = x;
        if (type != GGML_TYPE_F32) {
            scratch.q.resize(ggml_row_size(type, k));
            ggml_quantize_chunk(type, x, scratch.q.data(), 0, 1, k, quant_w);
            ggml_get_type_traits(type)->to_float(scratch.q.data(), scratch.deq.data(), k);
            deq = scratch.deq.data();
        }
        double d = 0.0, u = 0.0;
        for (int64_t j = 0; j < k; ++j) {
            const double err = (double) x[j] - (double) deq[j];
            d += act[j] * err * err;
            u += err * err;
        }
        if (!std::isfinite(d) || !std::isfinite(u)) {
            out.finite = false;
        }
        out.d[q] = d;
        out.u[q] = u;
    }
}

//
// row statistics
//

bool qd_scan_rows(const qd_row_reader & src, const qd_importance & imp, int n_threads, qd_row_stats & out) {
    const qd_tensor_info & t = src.info();
    const int64_t k      = t.n_per_row();
    const int64_t n_rows = t.n_rows();
    const int64_t rpe    = t.rows_per_expert();

    out.s.assign(n_rows, 0.0f);
    out.e.assign(n_rows, 0.0f);
    out.ns.assign(n_rows, 0.0f);
    out.ne.assign(n_rows, 0.0f);
    out.outlier.assign(n_rows, 0.0f);

    // blocks of rows are read and converted by the workers; only the file read itself is serialized
    const int64_t block    = std::max<int64_t>(1, (4ll << 20) / (k * (int64_t) sizeof(float)));
    const int64_t n_blocks = (n_rows + block - 1) / block;
    std::atomic<bool> ok(true);
    std::vector<std::vector<float>> bufs(std::max(1, n_threads));
    qd_parallel_for(n_blocks, n_threads, [&](int64_t b, int tid) {
        const int64_t r0 = b * block;
        const int64_t n  = std::min(block, n_rows - r0);
        auto & buf = bufs[tid];
        buf.resize(n*k);
        if (!src.read_rows(r0, n, buf.data())) {
            ok = false;
            return;
        }
        for (int64_t i = 0; i < n; ++i) {
            const float * x   = buf.data() + i*k;
            const float * act = imp.act.data() + ((r0 + i) / rpe) * k;
            double s, e, ns, ne, mx;
            qd_row_aux(x, k, act, s, e, ns, ne, mx);
            out.s [r0 + i] = (float) s;
            out.e [r0 + i] = (float) e;
            out.ns[r0 + i] = (float) ns;
            out.ne[r0 + i] = (float) ne;
            out.outlier[r0 + i] = e > 0 ? (float) (mx / std::sqrt(e / k)) : 0.0f;
        }
    });
    return ok;
}

//
// estimator
//

struct qd_stratum {
    // permuted rows, or, when empty, the range [range0, range0 + range_n) in the order of a keyed permutation
    std::vector<int64_t> units;
    int64_t              range0   = 0;
    int64_t              range_n  = 0;
    uint64_t             perm_key = 0;

    int64_t size() const { return units.empty() ? range_n : (int64_t) units.size(); }
    int64_t row(int64_t i) const { return units.empty() ? range0 + qd_feistel(i, range_n, perm_key) : units[i]; }

    int64_t              n_take  = 0;
    int64_t              n_first = 0;
    // totals over the stratum with row statistics: w_e * s, e, and the noise proxies w_e * ns, ne
    double               X_s [QD_WEIGHTING_COUNT_] = { -1.0, -1.0, -1.0 };
    double               X_ns[QD_WEIGHTING_COUNT_] = { -1.0, -1.0, -1.0 };
    double               X_e  = -1.0;
    double               X_ne = -1.0;
    double               prior = 0.0;
    double               sigma_prior = 0.0; // RMS of the row importance, for proxy allocation
};

struct qd_unit {
    int64_t     stratum;
    int64_t     row;
    qd_row_eval ev;
};

static void qd_init_result(qd_tensor_result & res, const qd_row_reader & src, const qd_importance & imp, const qd_params & params) {
    const qd_tensor_info & t = src.info();
    res.info = t;
    res.imp  = imp;
    res.units_total = t.n_rows();
    res.seed = params.seed;
    res.types.resize(params.qtypes.size());
    for (size_t q = 0; q < params.qtypes.size(); ++q) {
        auto & r = res.types[q];
        r.type   = params.qtypes[q];
        r.status = qd_type_status(r.type, t, imp.has_imatrix);
        if (r.status != QD_STATUS_UNSUPPORTED && t.n_per_row() % ggml_blck_size(r.type) == 0) {
            r.bytes = ggml_row_size(r.type, t.n_per_row()) * t.n_rows();
            r.bpw   = 8.0 * r.bytes / t.n_elements();
        }
    }
    res.experts.resize(t.n_expert());
    for (auto & e : res.experts) {
        e.rel.assign(params.qtypes.size(), 0.0);
    }
}

static std::vector<qd_status> qd_statuses(const qd_tensor_result & res) {
    std::vector<qd_status> s;
    for (const auto & r : res.types) {
        s.push_back(r.status);
    }
    return s;
}

static void qd_fill_metric(qd_metric & m, const qd_estimate & total, const qd_estimate & rel, double z) {
    m.total = total;
    m.rel   = rel;
    m.ci_lo = std::max(0.0, rel.value - z * rel.se);
    m.ci_hi = rel.value + z * rel.se;
}

// estimates of every metric from the current samples
struct qd_estimator_state {
    const qd_importance *            imp;
    const std::vector<qd_stratum> *  strata;
    const std::vector<qd_unit> *     units;
    std::vector<std::vector<size_t>> by_stratum;
    bool                             known_x;
    int64_t                          rpe;
    double                           S[QD_WEIGHTING_COUNT_] = {}; // exact totals with row statistics
    double                           E = 0.0;

    void init_totals() {
        if (!known_x) {
            return;
        }
        for (const auto & st : *strata) {
            for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
                S[w] += st.X_s[w];
            }
            E += st.X_e;
        }
    }

    // auxiliary of a unit: the noise proxy with row statistics (its total is known), the signal otherwise
    double aux(const qd_unit & u, double w, bool energy) const {
        if (known_x) {
            return energy ? u.ev.ne : w * u.ev.ns;
        }
        return energy ? u.ev.e : w * u.ev.s;
    }

    std::vector<qd_stratum_sample> samples(const std::function<double(const qd_unit &)> & y, qd_weighting wt, bool energy) const {
        std::vector<qd_stratum_sample> out(strata->size());
        for (size_t h = 0; h < strata->size(); ++h) {
            const auto & st = (*strata)[h];
            out[h].N = (double) st.size();
            out[h].X = known_x ? (energy ? st.X_ne : st.X_ns[wt]) : -1.0;
            for (size_t i : by_stratum[h]) {
                const qd_unit & u = (*units)[i];
                const double w = imp->w[wt][u.row / rpe];
                out[h].y.push_back((energy ? 1.0 : w) * y(u));
                out[h].x.push_back(aux(u, w, energy));
            }
        }
        return out;
    }

    // total and relative estimate of a per-unit value; energy selects the unweighted metric
    void estimate(const std::function<double(const qd_unit &)> & y, qd_weighting wt, bool energy, qd_estimate & total, qd_estimate & rel) const {
        const auto s = samples(y, wt, energy);
        if (known_x) {
            total = qd_estimate_total_ratio(s);
            const double denom = energy ? E : S[wt];
            rel.value = denom > 0 ? total.value / denom : qd_nan;
            rel.se    = denom > 0 ? total.se    / denom : qd_nan;
        } else {
            total = qd_estimate_total(s);
            rel   = qd_estimate_ratio(s);
        }
    }
};

static void qd_compute_estimates(qd_tensor_result & res, const qd_estimator_state & st, const qd_params & params) {
    const size_t nq = res.types.size();
    for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
        qd_estimate tot, rel;
        st.estimate([](const qd_unit & u) { return u.ev.s; }, (qd_weighting) w, false, tot, rel);
        res.signal[w] = st.known_x ? st.S[w] : tot.value;
    }
    {
        qd_estimate tot, rel;
        st.estimate([](const qd_unit & u) { return u.ev.e; }, QD_WEIGHTING_MASS, true, tot, rel);
        res.energy = st.known_x ? st.E : tot.value;
    }
    for (size_t q = 0; q < nq; ++q) {
        auto & r = res.types[q];
        if (r.status != QD_STATUS_OK) {
            continue;
        }
        for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
            qd_estimate tot, rel;
            st.estimate([q](const qd_unit & u) { return u.ev.d[q]; }, (qd_weighting) w, false, tot, rel);
            qd_fill_metric(r.weighted[w], tot, rel, params.ci_z);
        }
        qd_estimate tot, rel;
        st.estimate([q](const qd_unit & u) { return u.ev.u[q]; }, QD_WEIGHTING_MASS, true, tot, rel);
        qd_fill_metric(r.unweighted, tot, rel, params.ci_z);
        r.mse = tot.value / (double) res.info.n_elements();
    }
}

// z of D_a - D_b under the primary weighting, from paired per-unit differences
static double qd_paired_z(const qd_estimator_state & st, size_t a, size_t b, qd_weighting wt) {
    qd_estimate tot, rel;
    st.estimate([a, b](const qd_unit & u) { return u.ev.d[a] - u.ev.d[b]; }, wt, false, tot, rel);
    const double diff = st.known_x ? tot.value : rel.value;
    const double se   = st.known_x ? tot.se    : rel.se;
    if (se == 0.0) {
        return diff == 0.0 ? 0.0 : std::copysign(std::numeric_limits<double>::infinity(), diff);
    }
    return diff / se;
}

static void qd_rank_and_dominance(qd_tensor_result & res, const qd_estimator_state * st, const qd_params & params) {
    const qd_weighting wt = params.weighting;
    std::vector<size_t> ok;
    for (size_t q = 0; q < res.types.size(); ++q) {
        res.types[q].dominated_by = -1;
        res.types[q].rank = 0;
        if (res.types[q].status == QD_STATUS_OK) {
            ok.push_back(q);
        }
    }
    std::sort(ok.begin(), ok.end(), [&](size_t a, size_t b) {
        return res.types[a].weighted[wt].rel.value < res.types[b].weighted[wt].rel.value;
    });
    for (size_t i = 0; i < ok.size(); ++i) {
        res.types[ok[i]].rank = (int) i + 1;
    }
    // a type is dominated when another type is no larger and confidently less damaging
    for (size_t a : ok) {
        for (size_t b : ok) {
            if (a == b || res.types[b].bytes > res.types[a].bytes) {
                continue;
            }
            const double da = res.types[a].weighted[wt].total.value;
            const double db = res.types[b].weighted[wt].total.value;
            if (!(da > db)) {
                continue;
            }
            const double z = st ? qd_paired_z(*st, a, b, wt) : std::numeric_limits<double>::infinity();
            if (z > params.dominance_z) {
                res.types[a].dominated_by = (int) b;
                break;
            }
        }
    }
}

// rounding-level damage (e.g. BF16 -> F16) is noise; it must not drive allocation or stopping
static bool qd_is_lossless(double rel) {
    return rel < 1e-9;
}

static bool qd_converged(const qd_tensor_result & res, const qd_estimator_state & st, const qd_params & params) {
    const qd_weighting wt = params.weighting;
    std::vector<size_t> active;
    for (size_t q = 0; q < res.types.size(); ++q) {
        if (res.types[q].status == QD_STATUS_OK && res.types[q].dominated_by < 0) {
            active.push_back(q);
        }
    }
    if (active.empty()) {
        return true;
    }
    bool rse_ok = true;
    for (size_t q : active) {
        const auto & m = res.types[q].weighted[wt].rel;
        if (qd_is_lossless(m.value)) {
            continue;
        }
        if (!(m.se / m.value <= params.rse_target)) {
            rse_ok = false;
        }
    }
    if (rse_ok) {
        return true;
    }
    active.erase(std::remove_if(active.begin(), active.end(), [&](size_t q) { return qd_is_lossless(res.types[q].weighted[wt].rel.value); }), active.end());
    if (params.rank_z <= 0.0 || active.size() < 2) {
        return false;
    }
    std::sort(active.begin(), active.end(), [&](size_t a, size_t b) {
        return res.types[a].weighted[wt].rel.value < res.types[b].weighted[wt].rel.value;
    });
    for (size_t i = 0; i + 1 < active.size(); ++i) {
        if (!(std::fabs(qd_paired_z(st, active[i + 1], active[i], wt)) >= params.rank_z)) {
            return false;
        }
    }
    return true;
}

// prior, proxy RMS and auxiliary totals of a stratum with an explicit row list
template <typename F>
static void qd_stratum_totals(qd_stratum & st, const qd_importance & imp, const qd_row_stats * stats, int64_t rpe, const F & importance) {
    double sq = 0.0;
    st.prior = 0.0;
    for (int64_t r : st.units) {
        const double v = importance(r);
        st.prior += v;
        sq += v * v;
    }
    st.sigma_prior = std::sqrt(sq / st.units.size());
    st.X_e = st.X_ne = 0.0;
    for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
        st.X_s[w] = st.X_ns[w] = 0.0;
    }
    for (int64_t r : st.units) {
        st.X_e  += stats->e[r];
        st.X_ne += stats->ne[r];
        for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
            st.X_s [w] += imp.w[w][r / rpe] * stats->s[r];
            st.X_ns[w] += imp.w[w][r / rpe] * stats->ns[r];
        }
    }
}

static void qd_build_strata(const qd_tensor_info & t, const qd_importance & imp, const qd_params & params, const qd_row_stats * stats,
                            int64_t n0, std::vector<qd_stratum> & strata) {
    const int64_t nex = t.n_expert();
    const int64_t rpe = t.rows_per_expert();
    const int64_t k   = t.n_per_row();
    const auto & wm = imp.w[QD_WEIGHTING_MASS];

    // importance of a row: its weighted signal energy; without row statistics, the activation mass of its expert
    std::vector<double> m(nex, 0.0);
    for (int64_t e = 0; e < nex; ++e) {
        for (int64_t j = 0; j < k; ++j) {
            m[e] += imp.act[e*k + j];
        }
    }
    auto importance = [&](int64_t r) {
        const int64_t e = r / rpe;
        return wm[e] * (double) stats->ns[r];
    };

    strata.clear();

    // without row statistics every stratum is a whole expert and needs no row list; nothing is known that would rank rows
    if (!stats) {
        const bool per_expert = nex == 1 || params.expert_strata || params.min_per_expert > 0;
        const int64_t n_groups = per_expert ? nex : 1;
        std::vector<double> prior(n_groups, 0.0);
        for (int64_t e = 0; e < nex; ++e) {
            prior[per_expert ? e : 0] += wm[e] * m[e] * rpe;
        }
        double prior_sum = std::accumulate(prior.begin(), prior.end(), 0.0);
        if (!(prior_sum > 0.0)) {
            std::fill(prior.begin(), prior.end(), 1.0);
            prior_sum = (double) n_groups;
        }
        for (int64_t g = 0; g < n_groups; ++g) {
            qd_stratum st;
            st.range0   = per_expert ? g*rpe : 0;
            st.range_n  = per_expert ? rpe : t.n_rows();
            st.perm_key = qd_hash(params.seed, t.name + ":" + std::to_string(g));
            st.prior    = prior[g];
            st.sigma_prior = prior[g] / st.range_n;
            const int64_t min_g = n_groups > 1 ? std::max<int64_t>(2, params.min_per_expert) : 1;
            st.n_take   = std::min<int64_t>(st.range_n, std::max<int64_t>(min_g, (int64_t) std::llround(n0 * prior[g] / prior_sum)));
            st.n_first  = st.n_take;
            strata.push_back(std::move(st));
        }
        return;
    }

    // groups: one per expert, or one for the whole tensor
    std::vector<std::vector<int64_t>> groups;
    if (params.expert_strata || nex == 1) {
        for (int64_t e = 0; e < nex; ++e) {
            groups.emplace_back(rpe);
            std::iota(groups.back().begin(), groups.back().end(), e*rpe);
        }
    } else {
        // every expert is measured: its most important rows are certainty rows, the rest share one group
        std::vector<int64_t> forced;
        std::vector<uint8_t> is_forced(t.n_rows(), 0);
        const int64_t m_e = std::min<int64_t>(rpe, params.min_per_expert);
        std::vector<int64_t> rows(rpe);
        for (int64_t e = 0; e < nex && m_e > 0; ++e) {
            std::iota(rows.begin(), rows.end(), e*rpe);
            std::partial_sort(rows.begin(), rows.begin() + m_e, rows.end(), [&](int64_t a, int64_t b) {
                const double ia = importance(a), ib = importance(b);
                return ia != ib ? ia > ib : a < b;
            });
            for (int64_t i = 0; i < m_e; ++i) {
                forced.push_back(rows[i]);
                is_forced[rows[i]] = 1;
            }
        }
        if (!forced.empty()) {
            qd_stratum st;
            st.units = std::move(forced);
            std::sort(st.units.begin(), st.units.end());
            qd_stratum_totals(st, imp, stats, rpe, importance);
            st.n_take = st.n_first = (int64_t) st.units.size();
            strata.push_back(std::move(st));
        }
        groups.emplace_back();
        groups.back().reserve(t.n_rows() - (int64_t) strata.size());
        for (int64_t r = 0; r < t.n_rows(); ++r) {
            if (!is_forced[r]) {
                groups.back().push_back(r);
            }
        }
    }

    std::vector<double> prior(groups.size(), 0.0);
    for (size_t g = 0; g < groups.size(); ++g) {
        for (int64_t r : groups[g]) {
            prior[g] += importance(r);
        }
    }
    double prior_sum = std::accumulate(prior.begin(), prior.end(), 0.0);
    if (!(prior_sum > 0.0)) {
        std::fill(prior.begin(), prior.end(), 1.0);
        prior_sum = (double) groups.size();
    }

    const int64_t min_per_group = groups.size() > 1 ? params.min_per_expert : 1;

    for (size_t g = 0; g < groups.size(); ++g) {
        auto & rows = groups[g];
        const int64_t n_rows_g = (int64_t) rows.size();
        const int64_t n_g = std::min<int64_t>(n_rows_g, std::max<int64_t>(min_per_group, (int64_t) std::llround(n0 * prior[g] / prior_sum)));

        std::vector<std::vector<int64_t>> parts;
        std::vector<uint8_t> take_all_part;

        // certainty rows: a row whose importance is at least 1/F of the mean importance per remaining sampled row is always measured.
        // A few such rows can hold most of the damage of a type whose error on one extreme weight is erratic (e.g. MXFP4 next to a massive activation).
        int64_t n_certain = 0;
        if (stats && params.certainty_factor > 0.0 && n_g >= 2) {
            std::sort(rows.begin(), rows.end(), [&](int64_t a, int64_t b) {
                const double ia = importance(a), ib = importance(b);
                return ia != ib ? ia > ib : a < b;
            });
            double rest = 0.0;
            for (int64_t r : rows) {
                rest += importance(r);
            }
            const int64_t cap = n_g / 2;
            while (n_certain < cap && rest > 0.0 && importance(rows[n_certain]) * params.certainty_factor * (n_g - n_certain) >= rest) {
                rest -= importance(rows[n_certain]);
                n_certain++;
            }
            if (n_certain > 0) {
                parts.emplace_back(rows.begin(), rows.begin() + n_certain);
                take_all_part.push_back(1);
                rows.erase(rows.begin(), rows.begin() + n_certain);
            }
        }
        const int64_t n_sampled_g = n_g - n_certain;
        const int levels = stats ? (int) std::max<int64_t>(1, std::min<int64_t>(params.n_levels, n_sampled_g / 2)) : 1;

        if (stats && params.outlier_stratum && levels >= 4 && n_rows_g >= 100) {
            const int64_t n_out = std::max<int64_t>(1, n_rows_g / 100);
            std::sort(rows.begin(), rows.end(), [&](int64_t a, int64_t b) {
                return stats->outlier[a] != stats->outlier[b] ? stats->outlier[a] > stats->outlier[b] : a < b;
            });
            parts.emplace_back(rows.begin(), rows.begin() + n_out);
            take_all_part.push_back(0);
            rows.erase(rows.begin(), rows.begin() + n_out);
        }
        if (levels > 1) {
            // equal-mass levels over the row importance, so high-importance rows are over-sampled
            std::sort(rows.begin(), rows.end(), [&](int64_t a, int64_t b) {
                const double ia = importance(a), ib = importance(b);
                return ia != ib ? ia < ib : a < b;
            });
            double total = 0.0;
            for (int64_t r : rows) {
                total += importance(r);
            }
            std::vector<std::vector<int64_t>> lv(levels);
            double cum = 0.0;
            for (size_t i = 0; i < rows.size(); ++i) {
                const double v = importance(rows[i]);
                int l = total > 0.0 ? (int) ((cum + 0.5*v) / total * levels) : (int) (i * levels / rows.size());
                l = std::min(levels - 1, std::max(0, l));
                lv[l].push_back(rows[i]);
                cum += v;
            }
            for (auto & l : lv) {
                if (!l.empty()) {
                    parts.push_back(std::move(l));
                    take_all_part.push_back(0);
                }
            }
        } else if (!rows.empty()) {
            parts.push_back(rows);
            take_all_part.push_back(0);
        }

        const int64_t n_parts   = (int64_t) parts.size();
        const int64_t n_shared  = n_parts - (n_certain > 0 ? 1 : 0);
        int64_t       i_shared  = 0;
        for (int64_t p = 0; p < n_parts; ++p) {
            qd_stratum st;
            st.units = std::move(parts[p]);
            std::sort(st.units.begin(), st.units.end());
            qd_stratum_totals(st, imp, stats, rpe, importance);
            if (take_all_part[p]) {
                st.n_take = (int64_t) st.size();
            } else {
                const int64_t share = n_shared > 0 ? n_sampled_g / n_shared + (i_shared < n_sampled_g % n_shared ? 1 : 0) : 0;
                st.n_take = std::min<int64_t>((int64_t) st.size(), std::max<int64_t>(1, share));
                i_shared++;
            }
            st.n_first = st.n_take;
            qd_permute(st.units, qd_hash(params.seed, t.name + ":" + std::to_string(strata.size())));
            strata.push_back(std::move(st));
        }
    }
}

// grows the stratum allocations towards target with Neyman weights from the current residuals
static void qd_allocate(std::vector<qd_stratum> & strata, const qd_tensor_result & res, const qd_estimator_state & st, const qd_params & params, int64_t target) {
    const qd_weighting wt = params.weighting;
    const size_t H = strata.size();
    std::vector<double> sigma(H, qd_nan);

    for (size_t q = 0; q < res.types.size(); ++q) {
        const auto & r = res.types[q];
        if (r.status != QD_STATUS_OK || r.dominated_by >= 0) {
            continue;
        }
        const double R = r.weighted[wt].rel.value;
        const double D = r.weighted[wt].total.value;
        if (!(D > 0.0) || qd_is_lossless(R)) {
            continue;
        }
        double sy = 0.0, sx = 0.0;
        for (const auto & u : *st.units) {
            const double w = st.imp->w[wt][u.row / st.rpe];
            sy += w * u.ev.d[q];
            sx += st.aux(u, w, false);
        }
        const double rho = sx > 0 ? sy / sx : 0.0;
        for (size_t h = 0; h < H; ++h) {
            const auto & idx = st.by_stratum[h];
            if (idx.size() < 2) {
                continue;
            }
            double m = 0.0;
            std::vector<double> e(idx.size());
            for (size_t i = 0; i < idx.size(); ++i) {
                const auto & u = (*st.units)[idx[i]];
                const double w = st.imp->w[wt][u.row / st.rpe];
                e[i] = w * u.ev.d[q] - rho * st.aux(u, w, false);
                m += e[i];
            }
            m /= idx.size();
            double v = 0.0;
            for (double x : e) {
                v += (x - m) * (x - m);
            }
            const double sd = std::sqrt(v / (idx.size() - 1)) / D;
            sigma[h] = std::isnan(sigma[h]) ? sd : std::max(sigma[h], sd);
        }
    }

    // strata without a variance estimate use the median relative sd per unit of prior
    std::vector<double> ratio;
    for (size_t h = 0; h < H; ++h) {
        if (!std::isnan(sigma[h]) && strata[h].prior > 0) {
            ratio.push_back(sigma[h] * strata[h].size() / strata[h].prior);
        }
    }
    std::sort(ratio.begin(), ratio.end());
    const double fill = ratio.empty() ? 1.0 : ratio[ratio.size()/2];

    std::vector<double> weight(H);
    double wsum = 0.0;
    for (size_t h = 0; h < H; ++h) {
        const double N = (double) strata[h].size();
        double s = sigma[h];
        if (std::isnan(s)) {
            s = strata[h].prior > 0 ? fill * strata[h].prior / N : 0.0;
        }
        switch (params.allocation) {
            case QD_ALLOCATION_PROXY:    weight[h] = N * strata[h].sigma_prior;     break;
            case QD_ALLOCATION_OBSERVED: weight[h] = N * s;                         break;
            case QD_ALLOCATION_FIRST:    weight[h] = (double) strata[h].n_first;     break;
        }
        wsum += weight[h];
    }
    if (!(wsum > 0.0)) {
        for (size_t h = 0; h < H; ++h) {
            weight[h] = (double) strata[h].size();
        }
        wsum = std::accumulate(weight.begin(), weight.end(), 0.0);
    }

    int64_t have = 0;
    for (const auto & s : strata) {
        have += s.n_take;
    }
    // hand out the new units in a few passes so capped strata return their share
    for (int pass = 0; pass < 4 && have < target; ++pass) {
        double wopen = 0.0;
        for (size_t h = 0; h < H; ++h) {
            if (strata[h].n_take < (int64_t) strata[h].size()) {
                wopen += weight[h];
            }
        }
        if (!(wopen > 0.0)) {
            break;
        }
        const int64_t extra = target - have;
        for (size_t h = 0; h < H && have < target; ++h) {
            auto & s = strata[h];
            const int64_t cap = (int64_t) s.size();
            if (s.n_take >= cap) {
                continue;
            }
            const int64_t add = std::min<int64_t>(cap - s.n_take, (int64_t) std::ceil(extra * weight[h] / wopen));
            s.n_take += add;
            have += add;
        }
    }
}

qd_tensor_result qd_estimate_tensor(const qd_row_reader & src, const qd_importance & imp, const qd_params & params) {
    if (!params.row_stats) {
        return qd_estimate_tensor(src, imp, params, nullptr);
    }
    qd_row_stats stats;
    const auto t0 = std::chrono::steady_clock::now();
    if (!qd_scan_rows(src, imp, params.n_threads, stats)) {
        qd_tensor_result res;
        qd_init_result(res, src, imp, params);
        for (auto & r : res.types) {
            r.status = QD_STATUS_FAILED;
            r.note   = "read error";
        }
        return res;
    }
    auto res = qd_estimate_tensor(src, imp, params, &stats);
    res.seconds_scan = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - res.seconds_eval;
    return res;
}

qd_tensor_result qd_estimate_tensor(const qd_row_reader & src, const qd_importance & imp, const qd_params & params, const qd_row_stats * stats) {
    const auto t0 = std::chrono::steady_clock::now();
    const qd_tensor_info & t = src.info();
    const int64_t k   = t.n_per_row();
    const int64_t rpe = t.rows_per_expert();

    qd_tensor_result res;
    qd_init_result(res, src, imp, params);
    res.estimator = stats ? "ratio" : "expansion";
    const auto status = qd_statuses(res);

    const int64_t budget_rows = std::max<int64_t>(1, (params.max_sample_mib << 20) / (k * (int64_t) sizeof(float)));
    const int64_t n_max = std::min<int64_t>({ params.n_max, budget_rows, t.n_rows() });
    // per-expert floors come on top: certainty rows, or the minimum of each expert stratum
    const int64_t n0    = std::min<int64_t>(n_max, params.n_initial);

    std::vector<qd_stratum> strata;
    qd_build_strata(t, imp, params, stats, n0, strata);
    res.n_strata = (int64_t) strata.size();

    std::vector<qd_unit> units;
    std::vector<int64_t> taken(strata.size(), 0);

    qd_estimator_state st;
    st.imp     = &res.imp;
    st.strata  = &strata;
    st.units   = &units;
    st.known_x = stats != nullptr;
    st.rpe     = rpe;
    st.by_stratum.resize(strata.size());
    st.init_totals();

    const int64_t batch_rows = std::max<int64_t>(1, (256ll << 20) / (k * (int64_t) sizeof(float)));
    std::vector<qd_scratch> scratch(std::max(1, params.n_threads));
    bool failed = false;

    for (int round = 0; ; ++round) {
        // new units of this round, in file order
        std::vector<std::pair<int64_t, int64_t>> todo; // (row, stratum)
        for (size_t h = 0; h < strata.size(); ++h) {
            for (int64_t i = taken[h]; i < strata[h].n_take; ++i) {
                todo.emplace_back(strata[h].row(i), (int64_t) h);
            }
            taken[h] = strata[h].n_take;
        }
        std::sort(todo.begin(), todo.end());

        std::vector<float> buf;
        for (size_t b0 = 0; b0 < todo.size(); b0 += batch_rows) {
            const size_t nb = std::min<size_t>(batch_rows, todo.size() - b0);
            buf.resize(nb*k);
            for (size_t i = 0; i < nb; ++i) {
                if (!src.read_rows(todo[b0 + i].first, 1, buf.data() + i*k)) {
                    failed = true;
                }
            }
            const size_t base = units.size();
            units.resize(base + nb);
            qd_parallel_for((int64_t) nb, params.n_threads, [&](int64_t i, int tid) {
                const int64_t row = todo[b0 + i].first;
                const int64_t e   = row / rpe;
                qd_unit & u = units[base + i];
                u.row     = row;
                u.stratum = todo[b0 + i].second;
                qd_eval_row(buf.data() + i*k, k, imp.act.data() + e*k, imp.has_imatrix ? imp.quant_w.data() + e*k : nullptr,
                            params.qtypes, status, scratch[tid], u.ev);
            });
            for (size_t i = 0; i < nb; ++i) {
                st.by_stratum[units[base + i].stratum].push_back(base + i);
                failed = failed || !units[base + i].ev.finite;
            }
        }

        if (failed) {
            for (auto & r : res.types) {
                if (r.status == QD_STATUS_OK) {
                    r.status = QD_STATUS_FAILED;
                    r.note   = "read error or non-finite result";
                }
            }
            break;
        }

        qd_compute_estimates(res, st, params);
        qd_rank_and_dominance(res, &st, params);

        qd_round rd;
        rd.n_units = (int64_t) units.size();
        for (const auto & r : res.types) {
            rd.rel.push_back(r.weighted[params.weighting].rel.value);
            rd.rel_se.push_back(r.weighted[params.weighting].rel.se);
        }
        res.rounds.push_back(rd);

        if (params.adaptive && qd_converged(res, st, params)) {
            break;
        }
        const int64_t have = (int64_t) units.size();
        if (have >= n_max) {
            break;
        }
        qd_allocate(strata, res, st, params, std::min<int64_t>(n_max, 2*have));
        int64_t now = 0;
        for (const auto & s : strata) {
            now += s.n_take;
        }
        if (now == have) {
            break;
        }
    }

    res.units_sampled = (int64_t) units.size();

    // raw per-expert ratios
    std::vector<double> es(t.n_expert(), 0.0);
    for (const auto & u : units) {
        const int64_t e = u.row / rpe;
        res.experts[e].n_sampled++;
        es[e] += u.ev.s;
        for (size_t q = 0; q < res.types.size(); ++q) {
            if (res.types[q].status == QD_STATUS_OK) {
                res.experts[e].rel[q] += u.ev.d[q];
            }
        }
    }
    for (int64_t e = 0; e < t.n_expert(); ++e) {
        if (res.experts[e].n_sampled > 0) {
            res.experts_sampled++;
        }
        for (auto & v : res.experts[e].rel) {
            v = es[e] > 0 ? v / es[e] : qd_nan;
        }
    }

    const uint64_t row_bytes = ggml_row_size(t.type, k);
    res.bytes_sample = (uint64_t) units.size() * row_bytes;
    if (stats) {
        res.bytes_scan = (uint64_t) t.n_rows() * row_bytes;
    }

    res.seconds_eval = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

qd_tensor_result qd_exact_tensor(const qd_row_reader & src, const qd_importance & imp, const qd_params & params) {
    const auto t0 = std::chrono::steady_clock::now();
    const qd_tensor_info & t = src.info();
    const int64_t k   = t.n_per_row();
    const int64_t rpe = t.rows_per_expert();
    const int64_t nex = t.n_expert();
    const size_t  nq  = params.qtypes.size();

    qd_tensor_result res;
    qd_init_result(res, src, imp, params);
    res.estimator = "exact";
    res.units_sampled = t.n_rows();
    const auto status = qd_statuses(res);

    // per-expert sums: s, e, and d, u per type
    std::vector<double> es(nex, 0.0), ee(nex, 0.0);
    std::vector<std::vector<double>> ed(nex, std::vector<double>(nq, 0.0)), eu(nex, std::vector<double>(nq, 0.0));
    bool failed = false;

    FILE * dump = params.dump_rows.empty() ? nullptr : fopen(params.dump_rows.c_str(), "w");
    if (dump) {
        fprintf(dump, "row,expert,s,ns,e,ne");
        for (auto ty : params.qtypes) {
            fprintf(dump, ",d_%s", ggml_type_name(ty));
        }
        fprintf(dump, "\n");
    }

    // per-row primary-weighted signal and damage for the concentration statistics
    const auto & wp = imp.w[params.weighting];
    std::vector<float> row_s(t.n_rows());
    std::vector<std::vector<float>> row_d(nq, std::vector<float>(t.n_rows()));

    const int64_t batch_rows = std::max<int64_t>(1, (256ll << 20) / (k * (int64_t) sizeof(float)));
    std::vector<qd_scratch> scratch(std::max(1, params.n_threads));
    std::vector<float> buf;
    for (int64_t r0 = 0; r0 < t.n_rows(); r0 += batch_rows) {
        const int64_t nb = std::min(batch_rows, t.n_rows() - r0);
        buf.resize(nb*k);
        if (!src.read_rows(r0, nb, buf.data())) {
            failed = true;
            break;
        }
        std::vector<qd_row_eval> ev(nb);
        qd_parallel_for(nb, params.n_threads, [&](int64_t i, int tid) {
            const int64_t e = (r0 + i) / rpe;
            qd_eval_row(buf.data() + i*k, k, imp.act.data() + e*k, imp.has_imatrix ? imp.quant_w.data() + e*k : nullptr,
                        params.qtypes, status, scratch[tid], ev[i]);
        });
        if (dump) {
            for (int64_t i = 0; i < nb; ++i) {
                fprintf(dump, "%" PRId64 ",%" PRId64 ",%.6e,%.6e,%.6e,%.6e", r0 + i, (r0 + i) / rpe, ev[i].s, ev[i].ns, ev[i].e, ev[i].ne);
                for (size_t q = 0; q < nq; ++q) {
                    fprintf(dump, ",%.6e", ev[i].d[q]);
                }
                fprintf(dump, "\n");
            }
        }
        for (int64_t i = 0; i < nb; ++i) {
            const int64_t e = (r0 + i) / rpe;
            es[e] += ev[i].s;
            ee[e] += ev[i].e;
            row_s[r0 + i] = (float) (wp[e] * ev[i].s);
            for (size_t q = 0; q < nq; ++q) {
                ed[e][q] += ev[i].d[q];
                eu[e][q] += ev[i].u[q];
                row_d[q][r0 + i] = (float) (wp[e] * ev[i].d[q]);
            }
            failed = failed || !ev[i].finite;
        }
    }

    if (dump) {
        fclose(dump);
    }

    for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
        double S = 0.0;
        for (int64_t e = 0; e < nex; ++e) {
            S += imp.w[w][e] * es[e];
        }
        res.signal[w] = S;
    }
    res.energy = std::accumulate(ee.begin(), ee.end(), 0.0);

    for (size_t q = 0; q < nq; ++q) {
        auto & r = res.types[q];
        if (r.status != QD_STATUS_OK) {
            continue;
        }
        if (failed) {
            r.status = QD_STATUS_FAILED;
            r.note   = "read error or non-finite result";
            continue;
        }
        for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
            double D = 0.0;
            for (int64_t e = 0; e < nex; ++e) {
                D += imp.w[w][e] * ed[e][q];
            }
            qd_estimate tot { D, 0.0 };
            qd_estimate rel { res.signal[w] > 0 ? D / res.signal[w] : qd_nan, 0.0 };
            qd_fill_metric(r.weighted[w], tot, rel, params.ci_z);
        }
        double U = 0.0;
        for (int64_t e = 0; e < nex; ++e) {
            U += eu[e][q];
        }
        qd_fill_metric(r.unweighted, { U, 0.0 }, { res.energy > 0 ? U / res.energy : qd_nan, 0.0 }, params.ci_z);
        r.mse = U / (double) t.n_elements();
    }
    for (int64_t e = 0; e < nex; ++e) {
        res.experts[e].n_sampled = rpe;
        for (size_t q = 0; q < nq; ++q) {
            res.experts[e].rel[q] = es[e] > 0 ? ed[e][q] / es[e] : qd_nan;
        }
    }
    res.experts_sampled = nex;

    std::vector<int64_t> by_signal(t.n_rows());
    std::iota(by_signal.begin(), by_signal.end(), 0);
    std::sort(by_signal.begin(), by_signal.end(), [&](int64_t a, int64_t b) { return row_s[a] > row_s[b]; });
    const int64_t n1  = std::max<int64_t>(1, t.n_rows() / 100);
    const int64_t n10 = std::max<int64_t>(1, t.n_rows() / 10);
    for (size_t q = 0; q < nq; ++q) {
        auto & r = res.types[q];
        if (r.status != QD_STATUS_OK) {
            continue;
        }
        const auto & d = row_d[q];
        const double tot = std::accumulate(d.begin(), d.end(), 0.0);
        if (!(tot > 0.0)) {
            continue;
        }
        double s1 = 0.0, s10 = 0.0;
        for (int64_t i = 0; i < n10; ++i) {
            s10 += d[by_signal[i]];
            if (i < n1) {
                s1 += d[by_signal[i]];
            }
        }
        std::vector<float> sorted_d = d;
        std::nth_element(sorted_d.begin(), sorted_d.begin() + n1, sorted_d.end(), std::greater<float>());
        const double d1 = std::accumulate(sorted_d.begin(), sorted_d.begin() + n1, 0.0);
        r.share_top1_signal  = s1 / tot;
        r.share_top10_signal = s10 / tot;
        r.share_top1_damage  = d1 / tot;
    }

    qd_rank_and_dominance(res, nullptr, params);
    res.bytes_scan = (uint64_t) t.n_rows() * ggml_row_size(t.type, k);
    res.seconds_eval = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

//
// locality check
//

std::vector<qd_locality> qd_check_locality(const qd_row_reader & src, const qd_importance & imp, const std::vector<ggml_type> & types, int n_rows, uint64_t seed) {
    const qd_tensor_info & t = src.info();
    const int64_t k   = t.n_per_row();
    const int64_t rpe = t.rows_per_expert();
    const int64_t n_chunk = std::min<int64_t>(4, rpe);

    std::vector<qd_locality> out;
    uint64_t rng = qd_hash(seed, t.name + ":locality");

    // groups of consecutive rows inside one expert, so a chunk shares one imatrix slice as in llama-quantize
    std::vector<int64_t> starts;
    for (int i = 0; i < n_rows; ++i) {
        const int64_t e = (int64_t) (qd_splitmix64(rng) % t.n_expert());
        const int64_t r = (int64_t) (qd_splitmix64(rng) % (rpe - n_chunk + 1));
        starts.push_back(e*rpe + r);
    }

    std::vector<float> x(n_chunk*k);
    for (ggml_type type : types) {
        qd_locality loc;
        loc.type   = type;
        loc.status = qd_type_status(type, t, imp.has_imatrix);
        if (loc.status != QD_STATUS_OK) {
            out.push_back(loc);
            continue;
        }
        const size_t row_size = ggml_row_size(type, k);
        const int64_t blck = ggml_blck_size(type);
        int64_t span = std::max<int64_t>(256, blck);
        span = (span + blck - 1) / blck * blck;
        const bool span_ok = ggml_row_meta_size(type) == 0 && span < k && k % span == 0;
        loc.span = span_ok ? span : 0;
        loc.block_local = span_ok ? 1 : -1;

        std::vector<uint8_t> chunk(n_chunk*row_size), single(row_size), part(ggml_row_size(type, span_ok ? span : blck));
        for (int64_t r0 : starts) {
            if (!src.read_rows(r0, n_chunk, x.data())) {
                loc.status = QD_STATUS_FAILED;
                break;
            }
            const int64_t e = r0 / rpe;
            const float * qw = imp.has_imatrix ? imp.quant_w.data() + e*k : nullptr;
            ggml_quantize_chunk(type, x.data(), chunk.data(), 0, n_chunk, k, qw);
            for (int64_t i = 0; i < n_chunk; ++i) {
                ggml_quantize_chunk(type, x.data() + i*k, single.data(), 0, 1, k, qw);
                if (memcmp(single.data(), chunk.data() + i*row_size, row_size) != 0) {
                    loc.row_local = false;
                }
            }
            if (span_ok) {
                const int64_t off = (k / span / 2) * span;
                ggml_quantize_chunk(type, x.data() + off, part.data(), 0, 1, span, qw ? qw + off : nullptr);
                const size_t boff = (size_t) (off / blck) * ggml_type_size(type);
                if (memcmp(part.data(), chunk.data() + boff, part.size()) != 0) {
                    loc.block_local = 0;
                }
            }
            loc.rows_tried += (int) n_chunk;
        }
        out.push_back(loc);
    }
    return out;
}

//
// tie groups
//

qd_tie_result qd_tie_combine(const qd_tie_group & group, const std::vector<qd_tensor_result> & members, double ci_z) {
    GGML_ASSERT(group.tensors.size() == members.size());
    GGML_ASSERT(!members.empty());

    qd_tie_result out;
    out.group   = group;
    out.members = members;

    const size_t nq = members[0].types.size();
    for (const auto & m : members) {
        GGML_ASSERT(m.types.size() == nq);
    }

    uint64_t elements = 0;
    for (const auto & m : members) {
        elements += m.info.n_elements();
    }

    out.types.resize(nq);
    for (size_t q = 0; q < nq; ++q) {
        auto & tt = out.types[q];
        tt.type   = members[0].types[q].type;
        tt.status = QD_STATUS_OK;

        double var[QD_WEIGHTING_COUNT_] = {};
        double var_u = 0.0;
        for (const auto & m : members) {
            const auto & r = m.types[q];
            tt.bytes += r.bytes;
            if (r.status != QD_STATUS_OK) {
                if (tt.status == QD_STATUS_OK) {
                    tt.status = r.status;
                }
                if (!tt.note.empty()) {
                    tt.note += "; ";
                }
                tt.note += m.info.name + ": " + qd_status_name(r.status);
                if (!r.note.empty()) {
                    tt.note += " (" + r.note + ")";
                }
                continue;
            }
            for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
                tt.signal[w]                  += m.signal[w];
                tt.weighted[w].total.value    += r.weighted[w].total.value;
                // members are sampled independently, so the variances of their totals add
                var[w] += r.weighted[w].total.se * r.weighted[w].total.se;
            }
            tt.unweighted.total.value += r.unweighted.total.value;
            var_u += r.unweighted.total.se * r.unweighted.total.se;
        }
        if (tt.status != QD_STATUS_OK) {
            continue;
        }
        tt.bpw = elements > 0 ? 8.0 * tt.bytes / elements : 0.0;
        for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
            tt.weighted[w].total.se = std::sqrt(var[w]);
            const double S = tt.signal[w];
            tt.weighted[w].rel.value = S > 0 ? tt.weighted[w].total.value / S : qd_nan;
            tt.weighted[w].rel.se    = S > 0 ? tt.weighted[w].total.se       / S : qd_nan;
        }
        const double E = std::accumulate(members.begin(), members.end(), 0.0,
                                         [](double e, const qd_tensor_result & m) { return e + m.energy; });
        tt.unweighted.total.se = std::sqrt(var_u);
        tt.unweighted.rel.value = E > 0 ? tt.unweighted.total.value / E : qd_nan;
        tt.unweighted.rel.se    = E > 0 ? tt.unweighted.total.se    / E : qd_nan;
        for (int w = 0; w < QD_WEIGHTING_COUNT_; ++w) {
            tt.weighted[w].ci_lo = std::max(0.0, tt.weighted[w].rel.value - ci_z * tt.weighted[w].rel.se);
            tt.weighted[w].ci_hi = tt.weighted[w].rel.value + ci_z * tt.weighted[w].rel.se;
        }
        tt.unweighted.ci_lo = std::max(0.0, tt.unweighted.rel.value - ci_z * tt.unweighted.rel.se);
        tt.unweighted.ci_hi = tt.unweighted.rel.value + ci_z * tt.unweighted.rel.se;
    }
    return out;
}
