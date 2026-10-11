#pragma once

#include "llama.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

// delphi: VRAM sharing between servers (tools/vramd/README.md).
//
// With LLAMA_VRAMD=<socket>, the server registers with delphi-vramd and leases device memory: expert cache tiers on its GPUs
// before each prompt (the staging rings borrow them), revoked at a batch boundary when another server needs the memory. While
// idle, it frees its compute buffers (LLAMA_SHARE_IDLE_COMPUTE_S) and then its KV cache (LLAMA_SHARE_IDLE_KV_S, saved to the
// host prompt cache first) and gives both back; the next task leases them again. Without a daemon, only the idle timers apply
// (off unless set). All calls come from the server's main thread.
struct server_vram {
    enum phase_t { PHASE_IDLE, PHASE_PREFILL, PHASE_DECODE };

    server_vram();
    ~server_vram();

    // devices: the model's devices (empty: all GPUs). wake_loop makes the main loop call update() soon; save_slots saves the
    // idle slots to the prompt cache (if any) and clears them before the KV cache is freed.
    void init(llama_context * ctx_tgt, llama_context * ctx_dft, const std::vector<ggml_backend_dev_t> & devices,
              const std::string & name, std::function<void()> wake_loop, std::function<void()> save_slots);
    void free();

    // before a task that uses the model or its memory: lease and restore what idle released
    void wake();
    // at the start of every slot update: daemon messages and phase changes
    void update(phase_t phase);
    // about once a second while the server waits for tasks
    void idle_tick();

private:
    struct impl;
    std::unique_ptr<impl> d;
};
