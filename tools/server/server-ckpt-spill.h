#pragma once

#include "common.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// delphi: context checkpoints on disk (docs/delphi/usage/server-checkpoints.md).
//
// A slot that is full drops a checkpoint to make room for a new one. With --ctx-checkpoints-spill-dir, the dropped checkpoint is
// written to a file first, and read back when a later prompt of the slot needs a checkpoint that reaches further than the ones in
// memory.
//
// A checkpoint holds only the part of the memory that cannot be rolled back (the SWA window, the recurrent state). It restores the
// slot only together with the KV cells of the prompt prefix, which the slot still has in memory. So a file is good for as long as the
// slot holds the same prefix, in the same process. The key of a file is the model and context setup, the slot, the position, and a
// hash of the tokens before the position. A file is never used for another prompt, and no file outlives the server.
//
// The files are written by a thread of their own, from at most two checkpoints at a time, and a full queue drops the checkpoint
// as before. A read blocks the slot loop, so the server reads only when the estimated time is well below the time it saves.

// a checkpoint on disk
struct server_ckpt_spill_entry {
    int       slot_id     = -1;
    int64_t   n_tokens    = 0;
    llama_pos pos_min     = 0;
    llama_pos pos_max     = 0;
    uint64_t  prefix_hash = 0; // of the first n_tokens tokens of the prompt
    uint64_t  bytes       = 0; // size of the file
};

// the hash of the first n tokens of a prompt (n is capped at the number of tokens)
uint64_t server_ckpt_prefix_hash(const llama_tokens & tokens, size_t n);

// identifies the model files and the context setup that the checkpoints depend on. model_dft can be null.
uint64_t server_ckpt_model_key(const common_params & params, const llama_model * model_tgt, const llama_model * model_dft, uint32_t n_ctx_seq);

struct server_ckpt_spill {
    server_ckpt_spill();
    ~server_ckpt_spill();

    // Starts spilling to dir (made if it does not exist), with at most max_bytes on disk. Files that a dead server left in the
    // directory are removed (Linux and macOS). Returns false, and stays off, if the directory cannot be written.
    bool init(const std::string & dir, uint64_t max_bytes, uint64_t model_key);

    // stops the writer thread and removes all files of this server
    void shutdown();

    bool enabled() const;

    // Queues ckpt for writing and takes its data. Returns false, with ckpt as it was, if it is dropped: the queue is full, or it
    // does not fit in max_bytes.
    bool put(int slot_id, uint64_t prefix_hash, common_prompt_checkpoint && ckpt);

    // the files of a slot
    std::vector<server_ckpt_spill_entry> list(int slot_id) const;

    // Reads a file back and checks it against the entry and the key. Returns false if it is gone or bad, and then removes it.
    // out gets n_tokens, pos_min, pos_max and the data. ms gets the time of the read.
    bool load(const server_ckpt_spill_entry & entry, common_prompt_checkpoint & out, double & ms);

    // estimated time in ms to read a file of this size, from the speed of the earlier reads and writes
    double read_ms(uint64_t bytes) const;

    void erase(const server_ckpt_spill_entry & entry);
    void erase_beyond(int slot_id, llama_pos pos_max); // the files with a larger pos_max
    void drop_slot(int slot_id);                       // the files, and the writes in progress

    uint64_t bytes() const; // on disk
    size_t   size()  const; // files

private:
    struct impl;
    std::unique_ptr<impl> d;
};
