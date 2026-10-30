#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_model;

struct perplexity_hotswap_command {
    std::string action;
    int64_t sequence = -1;
    std::string path;
};

// Protocol adapted from ik_llama.cpp PR #2131 (6d78a87c), Thireus.
bool perplexity_hotswap_read_command(const std::string & path, perplexity_hotswap_command & command);
bool perplexity_hotswap_write_status(const std::string & path, const char * state, int64_t iteration, int64_t sequence);
void perplexity_hotswap_clear_packed();

// The caller must stop inference before an update, recreate its context after it,
// and then commit. A failed context recreation can roll back the whole update.
class perplexity_hotswap {
public:
    perplexity_hotswap();
    ~perplexity_hotswap();

    bool init(llama_model * model, const std::string & path, std::string & error);
    bool init(const std::vector<std::pair<std::string, ggml_tensor *>> & tensors,
              const std::string & path, std::string & error);
    int reload(const std::string & patch_path, std::string & error);
    int restore(std::string & error);
    // Applies a patch on top of the current tensors (no return to the floor first); returns the changed count.
    int add(const std::string & patch_path, std::string & error);
    void commit();
    void rollback();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
