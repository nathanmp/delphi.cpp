#include "server-ckpt.h"

#include "ggml.h"

#include <algorithm>
#include <iterator>

size_t server_ckpt_victim_index(const std::vector<int64_t> & pos, int64_t pos_new, common_ckpt_evict policy) {
    const size_t n = pos.size();

    GGML_ASSERT(n > 0);

    // the first checkpoint has one neighbour only, so it is not interior: with a single checkpoint there is no choice
    if (policy == COMMON_CKPT_EVICT_FIFO || n < 2) {
        return 0;
    }

    // gap[j] = distance between point j and point j + 1, the new checkpoint is point n
    std::vector<double> gap(n);
    for (size_t j = 0; j < n; j++) {
        const int64_t next = j + 1 < n ? pos[j + 1] : std::max(pos_new, pos[n - 1]);
        gap[j] = (double) (next - pos[j]);
    }

    // Removing interior point i merges gap[i - 1] and gap[i]. The number of gaps and their sum stay the same, so the variance is
    // smallest where the sum of squares is smallest. This changes by 2 * gap[i - 1] * gap[i].
    size_t best_i   = 1;
    double best_dev = gap[0] * gap[1];
    for (size_t i = 2; i < n; i++) {
        const double dev = gap[i - 1] * gap[i];
        if (dev < best_dev) {
            best_dev = dev;
            best_i   = i;
        }
    }

    return best_i;
}

std::list<common_prompt_checkpoint>::iterator server_ckpt_pick_victim(
        std::list<common_prompt_checkpoint> & ckpts, int64_t pos_new, common_ckpt_evict policy) {
    std::vector<int64_t> pos;
    pos.reserve(ckpts.size());
    for (const auto & ckpt : ckpts) {
        pos.push_back(ckpt.n_tokens);
    }

    auto it = ckpts.begin();
    if (!pos.empty()) {
        std::advance(it, server_ckpt_victim_index(pos, pos_new, policy));
    }
    return it;
}
