// delphi: tests of the prefix snapshot cache of the server (tools/server/server-prefix-cache.h), without a model

#include "server-prefix-cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static llama_tokens make_tokens(int seed, size_t n) {
    llama_tokens t(n);
    for (size_t i = 0; i < n; i++) {
        t[i] = (llama_token) ((seed * 7919 + (int) i * 31) % 32000);
    }
    return t;
}

static common_prompt_checkpoint make_snap(int64_t n, size_t bytes, uint8_t fill) {
    common_prompt_checkpoint c;
    c.n_tokens = n;
    c.pos_min  = 0;
    c.pos_max  = (llama_pos) n - 1;
    c.data_tgt.assign(bytes, fill);
    c.data_dft.assign(16, (uint8_t) (fill + 1));
    c.data_spec.assign(8, (uint8_t) (fill + 2));
    return c;
}

static void test_pick_point() {
    // a prompt of 1000 tokens, min 100
    // diverges at 300 with a prompt of 900, and is a continuation of a prompt of 500 (lcp = size, does not count)
    CHECK(server_prefix_pick_point({ 300, 500 }, { 900, 500 }, 1000, -1, 100) == 300);
    // the limit cuts the candidates that are past it
    CHECK(server_prefix_pick_point({ 300, 700 }, { 900, 800 }, 1000, 400, 100) == 300);
    CHECK(server_prefix_pick_point({ 700 }, { 800 }, 1000, 400, 100) == 0);
    // below the minimum
    CHECK(server_prefix_pick_point({ 50 }, { 900 }, 1000, -1, 100) == 0);
    // the whole new prompt matches: at least one token has to be processed after the point
    CHECK(server_prefix_pick_point({ 1000 }, { 1200 }, 1000, -1, 100) == 0);
    CHECK(server_prefix_pick_point({ 999 }, { 1200 }, 1000, -1, 100) == 999);
    // nothing held
    CHECK(server_prefix_pick_point({}, {}, 1000, -1, 100) == 0);
}

static void test_pick_victim() {
    // entries without a hit go first, the oldest of them
    CHECK(server_prefix_pick_victim({ 3, 0, 0 }, { 1, 5, 4 }) == 2);
    // all with hits: the oldest
    CHECK(server_prefix_pick_victim({ 3, 1, 2 }, { 7, 5, 6 }) == 1);
    CHECK(server_prefix_pick_victim({}, {}) == -1);
}

static void test_ram(void) {
    server_prefix_cache c;
    CHECK(c.init(1000, 10, "", 0, 42));
    CHECK(c.enabled());

    const llama_tokens a = make_tokens(1, 200);
    llama_tokens b = a;
    b[150] = 7; // shares 150 tokens with a

    CHECK(c.put(a, make_snap(100, 300, 1)));
    CHECK(!c.put(a, make_snap(100, 300, 1))); // a copy exists
    CHECK(c.has(a, 100));
    CHECK(c.has(b, 100));
    CHECK(!c.has(a, 99));

    server_prefix_match m;
    CHECK(c.find(b, 0, 199, m) && m.n_tokens == 100 && m.in_ram);
    CHECK(!c.find(b, 100, 199, m)); // not past n_past = 100
    CHECK(!c.find(b, 0, 99, m));    // over n_max

    // a stored prefix of 160 tokens of a: b differs inside it at 150
    CHECK(c.put(a, make_snap(160, 300, 2)));
    CHECK(c.lcp_differs(b, 200) == 150);
    CHECK(c.find(a, 0, 199, m) && m.n_tokens == 160);
    CHECK(c.find(b, 0, 199, m) && m.n_tokens == 100);

    double ms = 0.0;
    auto snap = c.fetch(m, ms);
    CHECK(snap && snap->n_tokens == 100 && snap->data_tgt.size() == 300 && snap->data_tgt[0] == 1);

    // over half the budget
    CHECK(!c.put(a, make_snap(50, 600, 3)));

    // a third entry needs room: the one without a hit (160) goes, the one with a hit (100) stays
    CHECK(c.put(a, make_snap(120, 300, 4)));
    CHECK(c.put(a, make_snap(130, 300, 5)));
    CHECK(c.has(a, 100));
    CHECK(!c.has(a, 160));

    c.drop(m);
    CHECK(!c.has(a, 100));

    c.shutdown();
    CHECK(!c.enabled());

    CHECK(!c.init(0, 10, "", 0, 42)); // a budget of 0 is off
}

static void test_disk(void) {
#ifdef _WIN32
    const std::string dir = (fs::temp_directory_path() / "test-server-prefix-cache").string();
#else
    const std::string dir = (fs::temp_directory_path() / ("test-server-prefix-cache-" + std::to_string(getpid()))).string();
#endif
    std::error_code ec;
    fs::remove_all(dir, ec);

    const llama_tokens a = make_tokens(2, 300);
    const llama_tokens b = make_tokens(3, 300);

    {
        server_prefix_cache c;
        CHECK(c.init(1 << 20, 10, dir, 1 << 20, 42));
        CHECK(c.put(a, make_snap(200, 1000, 9)));
        CHECK(c.put(b, make_snap(250, 1000, 11)));
        c.shutdown(); // waits for the writes
    }

    // a file of another model stays where it is
    const std::string other = (fs::path(dir) / "dpfx-00000000000000aa-5-0000000000000001.bin").string();
    {
        FILE * f = fopen(other.c_str(), "wb");
        fputs("not ours", f);
        fclose(f);
    }

    size_t n_files = 0;
    for (const auto & e : fs::directory_iterator(dir)) {
        n_files += e.path().extension() == ".bin" ? 1 : 0;
    }
    CHECK(n_files == 3);

    {
        // a later server of the same model finds the files
        server_prefix_cache c;
        CHECK(c.init(1 << 20, 10, dir, 1 << 20, 42));
        CHECK(c.stats().n_disk == 2 && c.stats().n_ram == 0);

        server_prefix_match m;
        CHECK(c.find(a, 0, 299, m) && m.n_tokens == 200 && !m.in_ram);
        CHECK(c.fetch_ms(m) > 0.0);

        double ms = 0.0;
        auto snap = c.fetch(m, ms);
        CHECK(snap && snap->n_tokens == 200 && snap->data_tgt.size() == 1000 && snap->data_tgt[999] == 9);
        CHECK(snap && snap->data_dft.size() == 16 && snap->data_spec.size() == 8 && snap->data_spec[0] == 11);
        CHECK(c.stats().n_ram == 1);

        // a damaged file is found by the checksum and removed
        CHECK(c.find(b, 0, 299, m) && m.n_tokens == 250);
        for (const auto & e : fs::directory_iterator(dir)) {
            const std::string name = e.path().filename().string();
            if (name.find("-250-") != std::string::npos) {
                FILE * f = fopen(e.path().string().c_str(), "r+b");
                fseek(f, -10, SEEK_END);
                fputc(0x55, f);
                fclose(f);
            }
        }
        CHECK(!c.fetch(m, ms));
        CHECK(!c.has(b, 250));
        c.shutdown();
    }

    {
        // another model does not see the files
        server_prefix_cache c;
        CHECK(c.init(1 << 20, 10, dir, 1 << 20, 43));
        CHECK(c.stats().n_disk == 0);
        c.shutdown();
    }

    CHECK(fs::exists(other));

    {
        // a disk budget for two files: the third one drops the one that was used the longest ago
        const llama_tokens t3 = make_tokens(4, 300);
        server_prefix_cache c;
        const llama_tokens * all[3] = { &a, &b, &t3 };
        for (int i = 0; i < 3; i++) {
            CHECK(c.init(1 << 20, 10, dir, 2500, 44));
            CHECK(c.put(*all[i], make_snap(100, 500, (uint8_t) i)));
            c.shutdown();
        }
        CHECK(c.init(1 << 20, 10, dir, 2500, 44));
        CHECK(c.stats().n_disk == 2);
        CHECK(!c.has(a, 100) && c.has(b, 100) && c.has(t3, 100));
        c.shutdown();
    }

    fs::remove_all(dir, ec);
}

int main(void) {
    test_pick_point();
    test_pick_victim();
    test_ram();
    test_disk();

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
