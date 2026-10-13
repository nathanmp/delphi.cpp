#include "server-ckpt.h"

#include "ggml.h"

#include <algorithm>
#include <iterator>

size_t server_ckpt_victim_index(const std::vector<int64_t> & pos, int64_t pos_new, common_ckpt_evict policy, bool pin_first) {
    const size_t n = pos.size();

    GGML_ASSERT(n > 0);

    // with a single checkpoint there is no choice
    if (n < 2) {
        return 0;
    }

    if (policy == COMMON_CKPT_EVICT_FIFO) {
        return pin_first ? 1 : 0;
    }

    // gap[j] = distance between point j and point j + 1, the new checkpoint is point n
    std::vector<double> gap(n);
    double sum   = 0.0;
    double sumsq = 0.0;
    for (size_t j = 0; j < n; j++) {
        const int64_t next = j + 1 < n ? pos[j + 1] : std::max(pos_new, pos[n - 1]);
        gap[j] = (double) (next - pos[j]);
        sum   += gap[j];
        sumsq += gap[j] * gap[j];
    }

    // Removing interior point i merges gap[i - 1] and gap[i]. The number of gaps and their sum stay the same, so the variance is
    // smallest where the sum of squares is smallest. This changes by 2 * gap[i - 1] * gap[i].
    size_t best_i    = 1;
    double best_prod = gap[0] * gap[1];
    for (size_t i = 2; i < n; i++) {
        const double prod = gap[i - 1] * gap[i];
        if (prod < best_prod) {
            best_prod = prod;
            best_i    = i;
        }
    }

    if (!pin_first) {
        // Removing the first point drops gap[0], which changes the sum. Compare the variance of the two results.
        const double cnt = (double) (n - 1);
        const double var_interior = (sumsq + 2.0 * best_prod) / cnt - (sum / cnt) * (sum / cnt);
        const double var_first    = (sumsq - gap[0] * gap[0]) / cnt - ((sum - gap[0]) / cnt) * ((sum - gap[0]) / cnt);
        if (var_first <= var_interior) {
            return 0;
        }
    }

    return best_i;
}

std::list<common_prompt_checkpoint>::iterator server_ckpt_pick_victim(
        std::list<common_prompt_checkpoint> & ckpts, int64_t pos_new, common_ckpt_evict policy, bool pin_first) {
    std::vector<int64_t> pos;
    pos.reserve(ckpts.size());
    for (const auto & ckpt : ckpts) {
        pos.push_back(ckpt.n_tokens);
    }

    auto it = ckpts.begin();
    if (!pos.empty()) {
        std::advance(it, server_ckpt_victim_index(pos, pos_new, policy, pin_first));
    }
    return it;
}
