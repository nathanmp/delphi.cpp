#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <vector>

// Bounded host row service. Jobs own their IDs and decoded rows, never scheduler buffers.
// The model's immutable table must outlive the reader and all submitted jobs.
class llama_engram_reader {
public:
    struct job;
    using job_ptr = std::shared_ptr<job>;

    explicit llama_engram_reader(int n_threads);
    ~llama_engram_reader();

    static bool supports(const ggml_tensor * table);
    job_ptr submit(const ggml_tensor * table, std::vector<int32_t> ids);
    static void collect(const job_ptr & work, float * dst, size_t n_bytes);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
