# Delphi migration status

The clean bootstrap of 2026-09-17 rebuilt Delphi on current upstream llama.cpp. It ports only the core subsystems of the legacy tree (`../delphi.cpp-archive`, remote `legacy`). The legacy history rewrote the upstream commits, so legacy hashes do not map to this tree: legacy `next` sits on upstream `465e49b9c` here, found by tree hash.

Base: upstream `972d2313b`.

## Commits

| subsystem | commits | source |
| --- | --- | --- |
| `--numa tensors` | `ggml-cpu : add --numa tensors, ...` | `references/numa-tensors` patch (feature commit `62b8b6398`), `git am` unchanged |
| IQ*_K / IQ*_KS, CPU | `ggml : add the IQ*_K and IQ*_KS quantization types with CPU support` | legacy `next`; types from ik_llama.cpp |
| IQ*_K / IQ*_KS, CUDA/HIP | `ggml-cuda : add IQ*_K and IQ*_KS support` | legacy `05e884e52` + sentinel fix `016b6412b` |
| execution trace profiler | `ggml : add the execution trace profiler` | legacy `next` |
| tensor profiler | `tests : benchmark an explicit op, shape and type with test-backend-ops --custom` | legacy `next`, without PerfDB |
| GLM-5.3-Flash | `llama : add GLM-5.3-Flash (glm5next)` | smalinin `my_glm53_flash` (`271235bb6`), plus legacy fixes |
| MoE routing | `ggml-cuda : group MoE routing on the device ...` | legacy `675f3b083..dd83c8137`, from TheTom/llama-cpp-turboquant PR #322 |
| GLU fusion | `ggml-cuda : fuse the MMVQ GLU when its output reuses src1` | legacy `a2308d822` |
| SM80 grouped/packed MoE | `ggml-cuda : add the grouped MoE schedule and the SM80 grouped and packed MUL_MAT_ID executors` | legacy `feat/sm80-packed-family` (`86790c518`) |
| DeepSeek V4.1 | 23 commits, `convert : add DeepSeek V4.1 ...` to `quantize: protect DeepSeek V4.1 weights` | legacy `port/deepseek41` + `20960568f`, cherry-picked with authorship |

## Build

Release (`-O3 -DNDEBUG`) everywhere:

    cmake -S . -B build-cpu  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON
    cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DGGML_CUDA=ON \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80
    HIPCXX="$(/opt/rocm/bin/hipconfig -l)/clang" HIP_PATH="$(/opt/rocm/bin/hipconfig -R)" \
        cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DGGML_HIP=ON -DGPU_TARGETS=gfx1201
    cmake --build <dir> -j 48

`moe-packed.cu` takes about 13 minutes in ptxas for sm80.

## Validation

CPU tests ran under `numactl --cpunodebind=0 --membind=0` (except NUMA). GPU tests ran on a CMP 170HX (sm80, CUDA0) and an R9700 (gfx1201, ROCm1).

- IQK: test-quantize-fns, test-row-meta pass. test-backend-ops MUL_MAT, MUL_MAT_ID, CPY, GET_ROWS, SET_ROWS pass on CPU, CUDA0 and ROCm1.
- NUMA: perplexity is bit-identical with and without `--numa tensors` for IQ4_KS, and for Q4_0 with `--no-repack` (with repack on, the two arms use different kernels). TG on gemma-4-26B-A4B Q8_0: 12.77 t/s (one socket bound) -> 15.38 t/s (`--numa tensors`, 60 threads), +20%. Ling-3.0-tiny: +42% over `--numa distribute`, but below one bound socket (too little work per token).
- Profiler: 13 exec-trace ctest tests pass. Builds on CPU, CUDA and HIP. A llama-bench trace with `GGML_EXEC_TRACE_GPU_TIMING=op` on CUDA0 and ROCm1 records CPU and GPU dispatches, and all 4153 GPU spans resolve. exec-trace-dump.py and exec-trace-perfetto.py work on both traces.
- Tensor profiler: `--custom` perf, test and CSV modes work (iq4_k MUL_MAT_ID, zipf routing).
- GLM-5.3: test-glm5next-sparse passes on CPU. test-llama-archs passes on CPU and ROCm1. KPOOL_EXPAND, TOP_K, LIGHTNING_INDEXER and MUL_MAT_ID pass on CPU, CUDA0 and ROCm1, and FLASH_ATTN_EXT on CPU and CUDA0. Full model GLM-5.3-Flash-UD-IQ4_XS on 3x CMP 170HX, `-ngl 999 -fa 1`:

  | path | pp512 | tg32 |
  | --- | ---: | ---: |
  | `GGML_CUDA_MOE_GROUPED=old` (the default at the time) | 361.4 | 32.6 |
  | `GGML_CUDA_MOE_GROUPED=packed GGML_CUDA_MOE_PACKED_INPLACE=1` (now the default; see `usage/environment-variables.md`) | 571.9 | 33.6 |

  test-mixed-logits (mixed prefill and decode, 3 requests) with packed in place against the default: mean KLD 0.015 to 0.030, same top-1 95 to 96%, the fp32 summation-order floor; the 1-token request is bitwise.
- SM80 MoE: test-moe-schedule passes on CPU and ROCm1 (2243/0). On CUDA0 it gives 4231 passed, 39 failed. The 39 are the known legacy gap: IQ2_KS, IQ3_KS and IQ5_KS packed-in-place MMVQ is not bitwise equal to the GGUF-layout MMVQ at T <= 8 (the row scale is applied before the sum instead of after it; NMSE ~1e-14). test-backend-ops routing cases (moe-devsort) pass.
- DeepSeek V4.1: test-llama-archs (CPU, ROCm1), test-quant-rules and test-convert-deepseek41.py pass. DeepSeek-V4.1-Flash-Q1_0 on ROCm1 (`-ngl 999 -cmoe`) loads, prefills and decodes (13.1 t/s tg). The output is "KasipakKasipak..." and PPL is ~2e7, the same as the legacy port and the vcruz305 donor gave on this file. Almost every tensor in the file is 1-bit, so a sane-precision file is still needed to confirm text.

## Not ported

SQLite PerfDB and weight cache, benchmark-moe-mmq, TurboQuant/expert cache, heat-aware eviction, cache self-calibration, MMID expert-group experiments, the other NUMA branches (EXPERTS mode, gang diagnostics, mmid_sched autotuner, NUMA repack), old Qwen ports, KDA/mHC experiments, ROCprofiler-only infrastructure, speculative decoding research (GLM adaptive and GPU-direct MTP drafting in common/speculative), vision (glm5next mtmd), loss predictor/autoquant, the legacy compact sparse CSA decode for DeepSeek V4 (`LLAMA_DSV4_SPARSE_ATTN`), test-numa.

## Open items

- GPU hardware identity (PCI bus, NUMA node, gfx arch) in trace metadata: in legacy this lived in benchmark-moe-mmq/PerfDB, which was not ported. The trace records the CUDA cc only.
- The IQ*_KS packed MMVQ bitwise gap above.
- The packed and grouped MoE executors are CUDA only (NVIDIA PTX). On HIP and MUSA the packed entry points are stubs.
- Tracing-disabled overhead is covered by test-exec-trace-disabled only, with no timed A/B against upstream.
- DeepSeek V4.1 conversion was tested with the unit test only, not a full HF checkpoint. No V4.1 file on hand produces text (see above). Published V4.1 GGUFs lack the candidate-indexer keys and diverge above 16384 tokens until reconverted.
