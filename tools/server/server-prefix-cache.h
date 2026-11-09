#pragma once

#include "chat.h"
#include "common.h"
#include "json.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// delphi: snapshots of prompt prefixes that several conversations share (docs/delphi/usage/server-prefix-cache.md).
//
// The host prompt cache (--cache-ram) keeps whole prompts, and does not use a long cached prompt for a new prompt that shares
// only its start (the system prompt and the tools of an agent harness). This cache keeps the full sequence state of a slot at
// the point where two prompts start to differ, apart from the other caches:
//   - The server finds the point: the longest common prefix of a new prompt with a prompt that it holds, where the two then
//     differ, before the first assistant message of the new prompt. One conversation alone never makes a snapshot.
//   - The slot stops its prompt batch there and the server stores the state of the sequence (target, draft, speculative state).
//   - A later prompt that starts with the same tokens gets the state back, and processes only the tokens after it.
// A snapshot is a full state, so it restores models with SWA or recurrent memory without a checkpoint.
//
// With a directory, the snapshots are also written to files, which a later server of the same model and context setup uses.
// The files of other models in the directory are left alone.
//
// All calls come from the thread of the slots. A thread of its own writes the files.

// a stored prefix that a prompt starts with
struct server_prefix_match {
    uint64_t id       = 0;
    int64_t  n_tokens = 0;
    uint64_t bytes    = 0;
    bool     in_ram   = false;
};

struct server_prefix_stats {
    size_t   n_ram      = 0;
    size_t   n_disk     = 0;
    uint64_t bytes_ram  = 0;
    uint64_t bytes_disk = 0;
};

// the RAM budget in bytes for --cache-prefix-mib (-2 = auto: min(4096 MiB, physical RAM / 16), -1 = no limit, 0 = off)
uint64_t server_prefix_ram_budget(int32_t mib);

// Pick the snapshot point for a new prompt. lcps has the common prefix length of the prompt with each prompt that the
// server holds, and sizes the size of each of those prompts. A prompt counts only where it differs after the common prefix.
// The point is the largest of those that is at most limit (the start of the first assistant message, -1 = no limit) and less
// than n_prompt. Returns 0 if it is less than min_tokens.
int64_t server_prefix_pick_point(const std::vector<int64_t> & lcps, const std::vector<int64_t> & sizes, int64_t n_prompt,
                                 int64_t limit, int64_t min_tokens);

// Pick the entry to drop from RAM: the entries without a hit first, then the one that was used the longest ago.
// hits and last_use have one entry per candidate. Returns an index, or -1 if there is no candidate.
int64_t server_prefix_pick_victim(const std::vector<uint64_t> & hits, const std::vector<uint64_t> & last_use);

// HTTP thread: adds prefix_cache_limit to llama_params, the byte offset in the prompt text where the first assistant message
// starts, or -1 if there is none. Without it, the request makes no snapshot.
void server_prefix_limit_params(common_json & llama_params, const common_chat_templates * tmpls,
                                const common_chat_templates_inputs & inputs, const std::string & prompt);

// HTTP thread: the limit in tokens from what server_prefix_limit_params() left in data. 0 = no snapshot, -1 = no limit.
int32_t server_prefix_limit_tokens(const common_json & data, const llama_vocab * vocab, const std::string & prompt);

struct server_prefix_cache {
    server_prefix_cache();
    ~server_prefix_cache();

    // ram_bytes = 0 turns the cache off. dir can be empty (no files). The files of this server have model_key in their name.
    bool init(uint64_t ram_bytes, int32_t min_tokens, const std::string & dir, uint64_t disk_bytes, uint64_t model_key);

    // stops the writer thread, after the writes in the queue. The files stay.
    void shutdown();

    bool    enabled()    const;
    int32_t min_tokens() const;

    // the largest common prefix of tokens with a stored prefix that differs from tokens after it (a stored prefix that tokens
    // starts with does not count)
    int64_t lcp_differs(const llama_tokens & tokens, int64_t n_max) const;

    // a snapshot of the first n tokens exists, in RAM or in a file
    bool has(const llama_tokens & tokens, int64_t n) const;

    // the longest stored prefix that tokens starts with, with n_min < n_tokens <= n_max
    bool find(const llama_tokens & tokens, int64_t n_min, int64_t n_max, server_prefix_match & out) const;

    // estimated time in ms to get the data of m (0 if it is in RAM)
    double fetch_ms(const server_prefix_match & m) const;

    // estimated time in ms to set the state of a slot from bytes of data, from the earlier restores
    double restore_ms(uint64_t bytes) const;
    void   note_restore(uint64_t bytes, double ms);

    // The snapshot of m, read from its file if it is not in RAM. nullptr if it is gone or the file is bad (then removed).
    // ms gets the time of the read.
    std::shared_ptr<const common_prompt_checkpoint> fetch(const server_prefix_match & m, double & ms);

    // Stores snap, the state after tokens[0, snap.n_tokens). Returns false if it is too large or a copy exists.
    bool put(const llama_tokens & tokens, common_prompt_checkpoint && snap);

    // the restore of m failed: removes it from RAM and disk
    void drop(const server_prefix_match & m);

    server_prefix_stats stats() const;

private:
    struct impl;
    std::unique_ptr<impl> d;
};
