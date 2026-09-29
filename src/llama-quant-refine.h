#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// Experimental, ordinary-format refinement for Q2_0 and Q4_0 rows.
// second and mean are per-input activation moments from the same calibration run.
bool llama_quant_refine_row(ggml_type type, const float * src, void * dst, int64_t n,
                            const float * second, const float * mean,
                            bool feedback, int gsq_steps, uint64_t seed);
