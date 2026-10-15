// Exact parity for the runtime Q8_K activation converter.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

static bool check_row(const std::vector<float> & data, const char * name, ggml_from_float_t quantize) {
    const size_t n = data.size();
    // Offset input and output to test loads/stores without vector alignment. Guard both output ends.
    std::vector<float> x(n + 1);
    std::copy(data.begin(), data.end(), x.begin() + 1);
    std::vector<block_q8_K> ref(n / QK_K + 2), got(ref.size());
    for (int fill : {0xa5, 0x5a}) {
        memset(ref.data(), fill, ref.size() * sizeof(block_q8_K));
        memset(got.data(), fill, got.size() * sizeof(block_q8_K));
        quantize_row_q8_K_ref(x.data() + 1, ref.data() + 1, n);
        quantize(x.data() + 1, got.data() + 1, n);
        // Equal initialization also checks that zero blocks leave bsums untouched.
        if (memcmp(ref.data(), got.data(), ref.size() * sizeof(block_q8_K)) != 0) {
            const auto * a = reinterpret_cast<const unsigned char *>(ref.data());
            const auto * b = reinterpret_cast<const unsigned char *>(got.data());
            size_t at = 0;
            while (a[at] == b[at]) { ++at; }
            fprintf(stderr, "Q8_K %s n=%zu fill=%x: mismatch at byte %zu (ref=%02x got=%02x)\n", name, n, fill, at, a[at], b[at]);
            return false;
        }
    }
    return true;
}

static bool test_parity(ggml_from_float_t quantize) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int checked = 0;
    auto check = [&](const std::vector<float> & x, const char * name) {
        ++checked;
        return check_row(x, name, quantize);
    };
    for (int n : {0, QK_K, 2*QK_K, 10*QK_K, 28*QK_K}) {
        std::vector<float> x(n, 0.0f);
        if (!check(x, "zero")) { return false; }
        for (int j = 0; j < n; ++j) { x[j] = j % 2 ? -0.0f : 0.0f; }
        if (!check(x, "signed zero")) { return false; }
        for (float scale : {1.0f, -1.0f, 1e-36f, -1e-36f, 1e36f, -1e36f}) {
            std::fill(x.begin(), x.end(), scale);
            if (!check(x, "constant")) { return false; }
            std::fill(x.begin(), x.end(), 0.0f);
            for (int j = 0; j < n; j += QK_K) { x[j + (j / QK_K * 37) % QK_K] = scale; }
            if (!check(x, "sparse")) { return false; }
        }
        for (int rep = 0; rep < 200; ++rep) {
            for (int b = 0; b < n / QK_K; ++b) {
                const float scale = std::ldexp(1.0f, int(rng() % 240) - 119);
                for (int j = 0; j < QK_K; ++j) {
                    x[b*QK_K + j] = b % 3 == 1 ? 0.0f : scale * dist(rng);
                }
            }
            if (!check(x, "random / alternating zero blocks")) { return false; }
        }
    }
    // Every position and both signs; all opposite-sign tie pairs, including cross-lane and cross-vector ties.
    std::vector<float> x(QK_K, 0.0f);
    for (int first = 0; first < QK_K; ++first) {
        for (float sign : {-1.0f, 1.0f}) {
            x[first] = sign;
            if (!check(x, "single maximum")) { return false; }
            for (int second = first + 1; second < QK_K; ++second) {
                x[second] = -sign;
                if (!check(x, "opposite-sign maximum ties")) { return false; }
                x[second] = 0.0f;
            }
            x[first] = 0.0f;
        }
    }
    // Neighbours of each half-integer, with exact and inexact inverse scales. The latter detect FMA contraction.
    for (float max : {127.0f, -127.0f, 3.1415927f, -3.1415927f, 1e-36f, -1e36f}) {
        const float iscale = -127.0f / max;
        for (int v = -127; v < 127; ++v) {
            const float tie = (v + 0.5f) / iscale;
            for (int j = 0; j < QK_K - 1; ++j) {
                x[j] = j % 3 == 0 ? tie : std::nextafter(tie, j % 3 == 1 ? -INFINITY : INFINITY);
            }
            x.back() = max;
            if (!check(x, "rounding boundaries")) { return false; }
        }
    }
    // Tiny members beside a finite maximum are valid even when an all-tiny block would overflow iscale.
    for (float max : {std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), 1.0f, -1.0f}) {
        for (int j = 0; j < QK_K; ++j) {
            x[j] = j % 2 ? std::numeric_limits<float>::denorm_min() : -std::numeric_limits<float>::min();
        }
        x.back() = max;
        if (!check(x, "finite extremes")) { return false; }
    }
    printf("Q8_K: %d rows match the reference exactly (two initial output patterns)\n", checked);
    return true;
}

static void benchmark(ggml_from_float_t quantize) {
#if defined(__linux__)
    // Discover both tensor NUMA nodes before pinning this single-thread converter benchmark.
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { perror("sched_getaffinity"); exit(1); }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) { continue; }
        cpu_set_t pinned;
        CPU_ZERO(&pinned);
        CPU_SET(cpu, &pinned);
        if (sched_setaffinity(0, sizeof(pinned), &pinned) != 0) { perror("sched_setaffinity"); exit(1); }
        printf("Q8_K bench CPU=%d\n", cpu);
        break;
    }
#endif
    std::mt19937 rng(5678);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    using clock = std::chrono::steady_clock;
    for (int n : {256, 2048, 7168}) {
        constexpr int rows = 256;
        std::vector<float> x(n * rows);
        for (auto & v : x) { v = dist(rng); }
        std::vector<block_q8_K> y(n / QK_K * rows);
        for (int rep = 0; rep < 5; ++rep) {
            for (int arm = 0; arm < 2; ++arm) {
                const bool reference = (arm + rep) % 2 == 0;
                auto run = [&] {
                    for (int r = 0; r < rows; ++r) {
                        if (reference) { quantize_row_q8_K_ref(x.data() + r*n, y.data() + r*n/QK_K, n); }
                        else { quantize(x.data() + r*n, y.data() + r*n/QK_K, n); }
                    }
                };
                for (int warm = 0; warm < 5; ++warm) { run(); }
                int runs = 0;
                const auto start = clock::now();
                double ns;
                do {
                    run();
                    ++runs;
                    ns = std::chrono::duration<double, std::nano>(clock::now() - start).count();
                } while (ns < 100000000.0);
                printf("Q8_K bench n=%d rep=%d reference=%d ns/row=%.3f\n", n, rep, reference, ns / (runs * rows));
            }
        }
    }
}

int main(int argc, char ** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") { bench = true; }
        else if (std::string(argv[i]) == "--numa" && i + 1 < argc && std::string(argv[i + 1]) == "tensors") { ++i; }
        else { fprintf(stderr, "usage: %s [--bench] [--numa tensors]\n", argv[0]); return 1; }
    }
    ggml_cpu_init();
    ggml_numa_init(GGML_NUMA_STRATEGY_TENSORS);
    const auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    if (!quantize || !test_parity(quantize)) { return 1; }
    if (bench) { benchmark(quantize); }
    return 0;
}
