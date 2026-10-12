#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// Experimental feedback for Q2_0, Q4_0, Q2_K-Q6_K, Q4_G128 and MXFP4 rows; local GSQ is Q2_0/Q4_0 only.
// second and mean are per-input activation moments from the same calibration run.
bool llama_quant_refine_row(ggml_type type, const float * src, void * dst, int64_t n,
                            const float * second, const float * mean,
                            bool feedback, int gsq_steps, uint64_t seed);
