#include "speculative-proposal.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

int main() {
    // Deliberately stale probabilities: only logits define the proposal law.
    std::vector<llama_token_data> data = {{9, std::log(3.0f), 0}, {2, 0, 1}, {5, -INFINITY, 0}};
    llama_token_data_array candidates = {data.data(), data.size(), -1, false};
    common_speculative_proposal sampler;
    std::vector<llama_token_data> q;
    std::vector<llama_token> draws;
    int count = 0;
    for (int i = 0; i < 100000; ++i) {
        const auto id = sampler.sample(candidates, 1, 42, q);
        assert(q.size() == 3 && q[0].id == 2 && q[1].id == 5 && q[2].id == 9);
        assert(std::abs(q[0].p - .25f) < 1e-6f && q[1].p == 0 && std::abs(q[2].p - .75f) < 1e-6f);
        assert(id == 2 || id == 9);
        count += id == 9;
        if (i < 128) { draws.push_back(id); }
    }
    assert(count > 74200 && count < 75800);
    // Reset only at request boundaries. Reordered backend candidates must reproduce a seeded request.
    sampler.begin();
    std::reverse(data.begin(), data.end());
    for (auto id : draws) { assert(sampler.sample(candidates, 1, 42, q) == id); }
    sampler.sample(candidates, .5f, 43, q);
    assert(std::abs(q[0].p - .1f) < 1e-6f && std::abs(q[2].p - .9f) < 1e-6f);
    common_speculative_proposal other;
    for (int i = 0; i < 128; ++i) {
        assert(sampler.sample(candidates, 1, 44, q) == other.sample(candidates, 1, 44, q));
    }
    // Underflow at low temperature must never produce a zero-probability token.
    for (int i = 0; i < 100; ++i) { assert(sampler.sample(candidates, 1e-6f, 44, q) == 9); }
    puts("OK: sparse proposal law, temperature, stream continuation and request reset");
}
