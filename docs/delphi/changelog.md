# Features and improvements integrated into main

Last updated: 2026-11-13

## Most important features

- **Host-routed expert execution across both sockets (`--numa tensors`)** - the core of the DeepSeek deployment.
  - Splits host routed-expert output rows across sockets, places each node's tensor ranges locally, and combines the results without copying whole expert weights to one socket; rows are dynamically claimed pieces per worker.
  - Each node keeps its converted activations, G128 panels, row maps and piece counters in node-local memory, with cross-node piece stealing when a node runs empty; both bitwise identical, default on (`GGML_NUMA_NODE_LOCAL=0` / `GGML_NUMA_STEAL=0` restore the old path).
  - The CPU compute buffer is pinned and page-interleaved over all NUMA nodes instead of sitting on the first GPU's node (`LLAMA_NUMA_HOST_INTERLEAVE=0` restores); MiMo NPS4 decode 34.64 -> 37.63 t/s.
  - Batched AVX2 vec_dot ports decode each weight block once for up to four activation rows across 15 quant types (~1.31x isolated MUL_MAT_ID; `GGML_CPU_BATCHED_VEC_DOT=0` for A/B), and runtime Q8_K activation quantization is vectorized.
  - IQK gate/up/SwiGLU fusion now covers partial expert operations and plain SwiGLU (`GGML_CPU_MOE_FUSE_UP_GATE=0` disables); isolated fused FFN +4-10%.
  - Refill threads read the node that holds their part and the pinned refill buffer can interleave (opt-in, for NPS4); the main thread stays pinned between graphs; helper threads respect the allowed CPU set.
  - CPU expert prefetch reads the predicted experts of the next CPU split during the GPU wait (`GGML_CPU_EXPERT_PREFETCH`, default 6 under `--numa tensors`); the router pre-gate computes the next layer's top-k on the current router input and sends cold ids with the expert-cache packet (`LLAMA_MOE_PREGATE`, default on). DeepSeek V4.1, 2 R9700s + RAM, DSpark: cold read-ahead 22% -> 44.5%, verify 72.79 -> 67.84 ms, 37.05 -> 39.48 t/s.
  - DeepSeek Engram lookups run on eight host readers (`LLAMA_ENGRAM_PREFETCH`, default 8) and the shared GPU FFN overlaps cold CPU experts (`LLAMA_MOE_SHARED_OVERLAP`, default on); Qwen PLE rows use the same reader. Combined decode is about +4% on DeepSeek V4.1 DSpark at 1.5 R9700s.
  - Per-graph NUMA worker split, balanced Q8_K conversion, IQ4_KS two-group decode and repacked partial `MUL_MAT_ID` improve utilization and remove schedule faults.

  Details: CPU execution and NUMA; Hybrid expert placement and CPU-to-GPU streaming.

- **Streaming routed experts from host RAM into GPU compute, with prefill overlap** - the alternative to CPU expert execution for large prefill and decode.
  - Asynchronous host-weight prefetch into a per-device staging ring (`GGML_SCHED_STAGE*`), packed on upload, consumed in place, with pinned bounce buffers, per-device workers, multi-GPU spread (`GGML_SCHED_STAGE_SPREAD`) and an explicit CPU-versus-streamed batch threshold.
  - Selected-expert uploads now overlap cold CPU prefill on a scheduler worker; default on (`LLAMA_MOE_PREFILL_OVERLAP=0` disables), extended beyond G128 banks to ordinary expert bytes and mixed projections; MiMo 1024-token TTFT 3.679 -> 2.795 s.
  - Short-prefill sharing (S1) splits each layer's experts between GPU residents, CPU rows and a bounded transient streamed bank; opt-in via `LLAMA_MOE_PREFILL_SHARE`; 1024-token first token 4.572 -> 3.667 s.
  - Staging is default-on for long prompts (`GGML_SCHED_STAGE=0` off), with 2 slots per spread GPU and no allocation without routed experts in host RAM; DeepSeek V4.1 on 2 R9700s: 4K prefill 444 -> 637 t/s, 1K 258 -> 388 t/s, decode unchanged.
  - When `GGML_SCHED_STAGE_MIN_TOKENS` is unset, warmup measures the CPU-versus-streamed crossover; only batches accepted by the backend offload hook are sent to the GPU, and a failed dot calibration is cached.

  Details: Hybrid expert placement and CPU-to-GPU streaming.

- **GPU hot-expert cache** - keeps selected experts resident in VRAM while cold experts compute on the NUMA host weights.
  - Frequency-selected resident banks with background refresh at safe batch boundaries, multi-tier caches on arbitrary GPUs, any compatible MoE architecture and GPU-supported expert type.
  - Auto mode is now the default: enabled automatically, one tier per GPU sized to free VRAM minus a 1536 MiB margin, later lowered to 1024 MiB.
  - Decode batches compute the CPU part's weighted sum on the CPU and transfer only [n_embd, n_tokens] per layer instead of every expert slot (`LLAMA_MOE_CACHE_CPU_REDUCE=0` restores the GPU merge).
  - Uses clipped SwiGLU where the architecture requires it (DeepSeek correctness fix), and staging/refill scratch shrinks to available memory instead of crashing.
  - Partial expert placement, per-GPU tiers, static tiers, explicit roles/reserves and pinned tiers that span layer devices; staging slots lend from a rotating tier before a static one and across banks/groups.
  - Decode-time refresh defaults to 64 experts per cadence with its scratch outside the tiers; NVIDIA writes cache slots as a stream-ordered device copy.
  - A tier that cannot lend a staging slot turns staging off on that GPU with a warning instead of aborting the first long prompt.

  Details: Hot-expert cache.

- **ik_llama.cpp qtypes plus Q4_G128, executed on CPU and GPU** - the routed-expert formats for the target quantizations.
  - Ported ik_llama.cpp's nine formats (IQ2_K..IQ6_K, IQ2_KS..IQ5_KS) with CPU quantization/dots/GEMM, IQK routed-expert execution, and CUDA/HIP dequantization, MMVQ and MMQ.
  - Added Q4_G128 (symmetric INT4, one BF16 scale per 128 values): CPU AVX2 integer dots, panels and GEMM; ordinary CUDA/HIP MMVQ/MMQ/grouped kernels; Marlin-style INT4 regions on CUDA and compact W4A8 kernels on RDNA4; fused HIP G128 FFN. Q4_G128 matches Q4_0 speed at 4.125 bpw, or is 2-7% faster at 1-512 tokens on the CMP170HXs.
  - CUDA INT4 MoE FFN kernels are on by default for Q4_G128 experts.
  - IQK packed MoE and dense kernels now decode in registers with prmt lookups: IQ2_KS/IQ3_KS/IQ4_KS/IQ5_K/IQ5_KS packed `MUL_MAT_ID` improved 1.5-3.4x at width 1, and IQ4_K uses byte permutes instead of a table. Packed IQ4_K is removed from the fused MoE down reduction because it is slower there.

  Details: Quantization formats and quantization tools; Complete INT4 routed-FFN execution; CUDA/HIP kernels and small fusions.

- **Automatic quantizer and quantization planning** - replaces manual per-model quantization choices.
  - Auto-quantizer V2 predicts each tensor's KL against the BF16 teacher from sampled per-role damage with per-band sensitivity, then solves an exact multiple-choice knapsack per device under byte and loss budgets; calibration cost is independent of candidate qtypes and layers.
  - End-to-end `auto-quantize` CLI (`setup`/`plan`/`run`): measured device capacities, reserves from llama-fit-params, resumable encode, post-encode verification; GPU layer-split calibration sessions when the floor fits.
  - Weighted multi-corpus imatrix collection/merging (default calibration corpus is the bundled ubergarm set); persistent tensor hotswap in llama-perplexity for KL studies without reloads (Q4_G128, NVFP4), with opt-in per-token KL output.
  - Validated: predicted KL within ~6% on 5 of 6 held-out witnesses; plans beat the local-metric plan by -3.4%/-13.1% KL at +2.2/+4.2 GB.
  - Calibration and encode speed: quantize threads (allowed CPUs less 8) and a chosen measurement GPU cut the DeepSeek V4.1 projection from about 11-15 h to about 6.8 h; `--keep-calibration`, retained interrupted passes, archived replacements and a patch cache preserve expensive work across rebuilds and probes.
  - Deployment realism: MTP reserves from `llama-fit-params --spec-type draft-mtp`, a measured expert-cache hit curve from routing replay, `--runtime` type filtering, a lazy-table base shard with `--split-tensors`, loss-only timing skip, and verify hardening for long prompts and full GPUs.

  Details: Automatic quantizer and quantization planning.

- **DeepSeek V4.1 causal encoder-decoder (CED) compact prefill, default-on** - realizes the architecture's intended prefill cost.
  - Encoder layers run on every row; decoder layers run only on a bounded suffix retained at the first decoder layer, with the full normalized encoder output kept for global KV/index-key projection and separate encoder/decoder sliding-window caches.
  - Prefill 1.73x/1.46x/1.90x at 1K/3.9K/65K tokens; 2048-token teacher-forced KL 0.00278 vs the full graph, 98.73% top-1 agreement, DSpark acceptance parity. `LLAMA_DSV41_TRIM=0` restores full prefill, `=2` is a bounded-mask reference.
  - A compacted prompt ubatch that a later ubatch clears keeps one decoder row instead of the full decoder window, removing about 1.4 s per 1024-token ubatch with decoder experts on the CPU.
  - Provenance: adapted from DeepInfer (774ce244b8, Apache-2.0).

  Details: DeepSeek V4.1 CED prefill.

- **DeepSeek V4.1 Flash long-context deployment (300K)** - removes the context-dependent cost that blocked the target context.
  - Speculative verification gathers each query's selected rows instead of copying the compressed stream: about 9 GB of copies per verification at 300K are gone, and decode stays 30.9-36.0 t/s from 128K to 250K.
  - Full-width compressed prefill masks leave the graph; the indexer scores 512 query rows at a time and prefill switches to indexed attention from 131072 context rows. At 300K the expert cache holds 1235/513 slots (was 740/30), both GPUs stage, and short-prompt decode is 37.7 t/s.
  - Compute buffers that grew with context depth are bounded: indexer score chunks expand top-k as they complete, indexer and attention rows are chunked to `LLAMA_CHUNK_MIB` (default 256), and the candidate source layer keeps I32 kept-block indices instead of a dense F32 block mask. At 300K the old 8989 MiB ROCm1 buffer becomes 3610 MiB and the expert cache grows from 492 to 792 slots.
  - With several parallel sequences, indexer and dense prefill attention are chunked per stream instead of in one multi-stream call; chunked dense attention is the default (`LLAMA_DSV41_ATTN_ROWS=1024`), and indexed prefill from 128K applies only when it is 0.
  - Reader GPUs mirror reused compressed KV once instead of copying their view every graph (3 R9700s at 15-18K: 118 -> 80.5 ms/verify); the top-k scratch no longer grows with the stream, and contexts recapture after a pool flush.
  - Engram gate weights stay on the GPU through BF16 GET_ROWS (3 CMP170HXs, 4096-token prefill 334 -> 576 t/s), pipeline parallelism survives tensor overrides (`LLAMA_PP_ALLOW_OVERRIDES=1`, 576 -> 619 t/s), and the tail-ubatch prompt checkpoint is off by default.
  - Decode and verify read only the raw-window cells that some query sees: for one-sequence ubatches of at most 16 tokens, the host builds the visible-cell list and a 256-row mask, and the graph gathers those rows before the compact and verify gathers. This cuts the first reply after a fresh prompt from 20.6 to about 15.1 ms of ROCm0 dense time per verify at 10K context; KLD 0.00031 against a 0.00010 floor, top-1 100%; `LLAMA_DSV41_RAW_WINDOW=0` disables.
  - Compressor block counts stay stable per sequence independently of the overlap-window layout, with dummy writes outside visible rows, and graph reuse checks the raw-mask shape when the window reads it. The shape check fixes a decode degeneration after a long prompt (KL rose from 0.005 to 0.4-0.6 at position 6936; now 0.0026 to 9035, equal to `LLAMA_DSV41_RAW_WINDOW=0`), with graph builds per verify unchanged (about 7%).

  Details: DeepSeek V4.1 long context; Graph reuse, scheduling, transfers and loading.

- **RDNA4 sparse FlashAttention and indexed sparse attention**
  - Mask-compacted WMMA FlashAttention on RDNA4 (DeepSeek 512-wide, Qwen 256-wide heads), direct-index decode without mask conversion, and Qwen's indexer top-k selections passed directly into FlashAttention by default.
  - Qwen QSA gained a pooled-key cache and O(1) incremental block grouping: CMP decode at 100K context 970 -> 18 us/token, and 2 R9700s at 0/50K/100K decode 44.4/40.7/38.5 t/s; 256-wide sparse tiles and multi-block radix top-k are ported from upstream #28770.
  - DeepSeek V4.1 selected-row physical compaction for decode and speculative verification, plus opt-in indexed prefill; GLM sparse masks built on device; the CUDA/HIP decode KV_max masked-tail scan is on by default.
  - DeepSeek V4.1 prefill replaces full-width compressed masks with per-row visible counts and switches to indexed attention from 131072 context rows; GLM k-pool host inputs got a tail fast path.
  - The RDNA4 lightning indexer uses WMMA for F16 keys: DeepSeek V4.1 Flash on 2 R9700s, decode at 100K context 31.1 -> 35.5 t/s and 40K prefill +13%; decode KLD 0.0016 against the vector kernel, top-1 100%; `GGML_HIP_LID_WMMA=0` selects the vector kernel.

  Details: Sparse attention and attention-mask work; CUDA/HIP kernels and small fusions.

- **GLM KV window: long contexts with the KV cache mostly in host memory**
  - `LLAMA_KV_RESIDENT_CELLS` keeps the first cells of each attention KV stream of the hybrid memory in VRAM and maps the rest to pinned host memory at the same addresses (CUDA VMM); wide batches copy the layer K view to VRAM, narrow batches gather only the selected cells, and a verify gathers each selected cell once (`LLAMA_KV_WINDOW_UNION=0` disables). 16-bit rows gather as I32 pairs.
  - `auto` uses the `--fit` dry run to move each device's deficit to host memory, with per-device `LLAMA_KV_WINDOW_MARGIN_MIB` (default 1024); `LLAMA_KV_IDX_RESIDENT_CELLS` windows the indexer cache (4096 cells with auto). Only the pooled sparse attention and indexer of glm5next use the window; other models warn and ignore it, and state read/copy/add/div invalidate only their sequence's pools.
  - GLM-5.3 Flash on 3 CMP170HXs loads `-c 1048576` and runs a 300K prompt; 100K prefill is 1613 t/s (1627 at 655K without the window), DFlash2 decode 28.7 t/s. Auto at `-c 1572864 -np 3` moves 0/4832/5524 MiB to host, and three concurrent 300K prompts complete. Staged output is bitwise equal to no-window; the gather path stays below the ubatch-change KLD floor.

  Details: KV window and GLM long context.

- **Shared prompt prefix snapshots and checkpoint reuse**
  - `--cache-prefix-mib` (default min(4096 MiB, RAM/16)), `--cache-prefix-min-tokens` (1024), `--cache-prefix-dir` and `--cache-prefix-disk-mib` keep the full sequence state at the point where two prompts diverge, before the first assistant message, apart from `--cache-ram`. A restore is used only when its measured cost is below the prefill it saves. Files are versioned, checksummed and keyed by model, context setup and sequence count, so several servers can share a directory.
  - Prompts with media work while an mmproj is loaded: the snapshot ends before the first media chunk, and the checkpoint spill is keyed on the text tokens before it. Spilled files survive a request that keeps the slot prompt.
  - A memory that reports a single position (DeepSeek V4 compressed KV, hybrid SWA plus recurrent) uses its checkpoint without the SWA window check: a new DeepSeek V4.1 conversation with a 15K system prompt processes 10 tokens in 2.0 s instead of 15022 in 19 s.
  - Validated bitwise on plain, hybrid, SWA and recurrent models and after a restart; GLM 15K prefix snapshot 428 MiB, new-session TTFT 4.4 s vs about 16 s; DeepSeek V4.1 79.6 MiB, TTFT 4.0 s vs 20.3 s.

  Details: KV, context and server state management.

- **Parallel prompt processing with per-sequence KV streams**
  - DeepSeek V4.1 gives a sequence with more than `n_keep_dec` tokens in the batch its own prompt ubatches (`split_seq_id`), reserves the one-sequence prompt shape and multi-sequence ubatches at `n_seq_max * n_keep_dec` tokens, and skips the one-sequence max-prefill reservation for multi-stream full contexts. Extracted layer inputs are scattered to their batch rows so drafts read the right features.
  - Indexer and dense prefill attention are chunked per stream when several sequences are active; at `-np 3 -c 900000` the ROCm0/ROCm1 compute buffers fall from 13198/18066 MiB to 3701/11576 MiB, and three concurrent 16K prompts finish in 45 s (62-86 s before).
  - GLM hybrid memory uses the same one-sequence split (`LLAMA_HYBRID_SEQ_SPLIT_TOKENS`, default 16), reserves 16 tokens per sequence for multi-sequence ubatches, and compares tokens per stream against the staging threshold. With KV and indexer windows plus DFlash2 at `-c 1572864 -np 3`, the draft starts and a 16K prompt runs at 1266-1704 t/s.
  - With `-np auto`, DeepSeek V4 keeps 4 slots but turns unified KV off, so compute buffers are per stream; without `-c` the slots share the training context, `-c 0` gives each the full context, and a unified DeepSeek V4 cache with more than one sequence warns.

  Details: Parallel prompt processing and per-sequence KV streams; DeepSeek V4.1 long context.

- **Pipelined multi-slot decode and background prompt processing** - keeps generating slots moving while other slots process prompts.
  - Staggered verify is on by default where it applies (pipeline parallelism over a layer split, a KV stream per sequence, and an alternate graph slot per sequence): each sequence gets its own ubatches and scheduler, so one sequence's verify runs on one device while the next runs on another, and a reused graph waits only for its own scheduler. GLM-5.3 Flash, 3 CMP170HXs, DFlash2, `-np 3`: round 70 -> 61 ms at 2 slots and 91 -> 79.5 ms at 3 slots, aggregate 97-102 -> 117-128 t/s at 3 slots. `LLAMA_STAGGERED_VERIFY=0` disables.
  - Verify streams (default on where they apply, `LLAMA_VERIFY_STREAMS=0` opts out) give the alternate graph slots a second set of GPU backends with high-priority streams. Prompt chunks marked as background do not block output synchronization and complete by ticket; with generating slots the server adds at most one prompt chunk (1024 tokens by default) while fewer than 3 chunks are in flight, and the last 16 prompt tokens run alone. `LLAMA_SERVER_PREFILL_CAP` caps prompt tokens in a mixed batch and `LLAMA_STAGGERED_MIXED_UBATCH` cuts mixed prompts into pipelined ubatches: GLM-5.3 Flash, 3 CMP, 12.8K prompt while one slot decodes, longest decode gap 4.4-5.0 -> 1.2-1.4 s and prompt 1415-1623 -> 899-921 t/s.
  - Prompt inputs stage through a bounded pinned scheduler ring and upload on the destination stream; verify inputs upload asynchronously; growth room and CUDA buffer retirement avoid allocation stalls next to the high-priority streams; prompt checkpoints copy asynchronously by default (`LLAMA_CKPT_ASYNC=0` restores synchronous).
  - The mode is off with a unified KV cache, without alternate graph slots, or with an expert cache, and it requires staggered verify; behavior without `LLAMA_VERIFY_STREAMS` is unchanged.

  Details: Background prompt work, staggered verify and verify streams; Graph reuse, scheduling, transfers and loading; KV, context and server state management.

- **DeepSeek V4.1 layer-bank encoder windows, default-on**
  - Each prompt chunk is processed layer-major: an encoder layer's expert bank stays resident across its chunk graphs while the next layer's bank uploads; keep mode stages on the activation GPU, prefetches from the ring that spread will use, and evicts only slots the graph does not read. HC stream and pending mix live in device state, and attention inputs and index selections are shared per chunk. The CED decoder boundary is unchanged.
  - `LLAMA_DSV41_LAYER_BANK` defaults to 1 (2 = same schedule without retained banks, 0 = ordinary prefill); retained state is byte-identical to ordinary prefill. 2 R9700 at `-c 300000`, 8K/16K/32K prompt latency: 15.16/29.44/59.38 -> 8.67/14.60/30.33 s.
  - Chunked dense prefill attention is default at `LLAMA_DSV41_ATTN_ROWS=1024`; indexed prefill from 128K is selected only when it is 0.

  Details: DeepSeek V4.1 CED prefill.

- **Bottleneck profiler and execution traces**
  - Recipe-driven `tools/bottlenecks/profile.py` owns the llama-bench / llama-server + SPEED-Bench workflow, trace collection, build/model identity and Amdahl-oriented reports; binary traces (`GGML_EXEC_TRACE`) carry execution, ubatch, node, split, device and copy identity with GPU event timing and an integrity harness.
  - Device-interval placement fix: GPU epochs are attributed to the synchronize that completed them; the earlier "every device idle ~60%" decode findings were placement artifacts (now 6-15%).

  Details: Profiling, diagnostics and reproducible comparisons.

- **Speculative decoding stack**
  - Draft heads: GLM/Qwen MTP, DeepSeek DSpark, DFlash2, EAGLE3; asynchronous and state-only prompt ingest; adaptive draft-depth controller.
  - Per-round confidence draft depth is on by default for draft-mtp: each round stops where calibrated acceptance no longer pays for its verify row; conf vs fixed-3: GLM +2.5%/+1.6%, Qwen +4.0%/-2.1%.
  - `POST /speculative` reconfigures speculative options at run time; probabilistic draft sampling (`--spec-draft-sampling`) is ported for draft-simple/MTP and DFlash2 with an exact output distribution; the server reuses samplers between chat requests and reserves backend-sampler buffers only when chains change (DeepSeek first-token 2174 -> 680 ms).
  - DSpark drafts the trained block by default and verifies only the first n_max proposals (`--spec-draft-block`, 0 = trained size): DeepSeek V4.1 Flash on 1.5 R9700s at n_max 3, 25.47 -> 26.45 t/s.
  - Adaptive DSpark verification prefix (`--spec-draft-adaptive-mode block`): the complete trained block is still drafted, and per-slot confidence plus live complete-round costs choose how much of it to verify, including an empty prefix when `--spec-draft-n-min-adaptive 0`. Recommended floor 1: +4.0% aggregate decode on a four-task mixed workload (single sequence); the parallel controller keeps per-sequence state and prices shared verification rounds against the cohort.
  - Probabilistic DSpark proposals reuse the server rejection sampler with top-k proposal probabilities and an independent per-sequence stream; greedy stays the default. At temperature 1, adaptive probabilistic gained about +4% against greedy in two comparisons.
  - Draft features download on a side stream and DFlash reserves both contexts at load; the speculative prompt ends with a call of at most one ubatch so the draft ingests the rest in parallel (`LLAMA_SPEC_TAIL_SPLIT=0` off).
  - DFlash target features stay in a backend buffer and are gathered directly with GET_ROWS (port of upstream #31506, am17an, revision 806ae1a), avoiding the host embedding round trip; `LLAMA_DFLASH_DEV=0` keeps the host path.
  - Probabilistic verification reads only the drafted token probability on accepted rows and fetches the original candidate order only for a rejected row's residual; it no longer sorts full-vocabulary candidates, which cost about 6 ms per rejected row. GLM-5.3 Flash, 3 CMPs, DFlash2 probabilistic: top-p 1 52.7 -> 44.3 ms/verify, top-p 0.95 46.1 -> 45.0, greedy unchanged; `LLAMA_SPEC_REJECTION_HOST=1` keeps the old path.
  - Cached backend-sampler graphs keep their own input bindings, so replaying another cached graph no longer reuses stale RNG or penalty history.
  - Per-verify cache cleanup now scans only the used cell range (see KV, context and server state management): GLM-5.3 Flash, 3 CMPs, DFlash2 probabilistic short chat, 45.0 -> 43.4 ms/verify at top-p 0.95 and 43.4 -> 42.0 greedy.
  - A V4.1-format DSpark draft without `hc_head` weights takes V4.1 semantics with a warning instead of failing.

  Details: Speculative decoding, draft ingest and numerical controls; KV, context and server state management.

- **Model support for the target deployments**
  - DeepSeek V4.1 (`deepseek41`) with engrams, hyper-connections, sparse/compressed attention, variable compression ratios and the DSpark draft head; GLM-5.3 Flash (`glm5next`) with KDA, mHC, clamped SwiGLU, vision tower and MTP; Qwen 3.8 Flash Next with MTP and tensor-split; Muse Spark Open (text, vision and audio); MiMo.
  - MTP drafters read media (mtmd) batches with correctly sized embeddings; Qwen vision+MTP image acceptance went from 1/1750 to 171/355.
  - DeepSeek V4.1 vision: converter `deepseek41v` projector, mtmd reading-order grid, V4.1 resize/pad rules, and engram disabled for image spans.
  - DeepSeek V4.1 chat encoding: Jinja `dict()`, native 1017 template with spaced DSML parser/grammar, official fixtures byte-identical, server tool round trip 10/10. Existing GGUFs need `--chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1.jinja`.
  - Dynamic vision is the default (`--mmproj-memory dynamic`): encoder weights stay in RAM and load into released language compute memory per encoding; GLM's indexer pool bias is built on device, cutting 500K-context compute buffers from 26.3 to 8.2 GiB at -ub 2048.

  Details: Model support, conversion and model-specific correctness; DeepSeek V4.1 chat encoding.

- **v1 command line, defaults and usage guides** - the 2026-11-01 user-facing integration pass.
  - Command-line options proxy the delphi performance settings, each setting the engine variable that the code reads, so existing environment-based scripts keep working and an option value takes precedence.
  - New defaults: staging for long prompts (2 slots per GPU, spread), CPU expert prefetch (6) and router pre-gating under `--numa tensors`; the tail-ubatch prompt checkpoint is off. Nothing is allocated without routed experts in host RAM.
  - The 2026-11-12 delta makes staggered verify default-on where it applies, asynchronous prompt checkpoints default-on, GLM stable prompt-tail indexer partitions default-on, DeepSeek V4.1 raw-window reads default-on, and the RDNA4 lightning indexer default to WMMA for F16 keys.
  - Public usage guides under `docs/delphi/usage/` for auto-tune, auto-quantize, the expert cache, speculative decoding, VRAM sharing, vision and performance/placement, with an index and an updated environment reference.

  Details: v1 command line and defaults; Public usage documentation; Hybrid expert placement and CPU-to-GPU streaming; CPU execution and NUMA.

- **Auto-tune v1** - integrated launch-parameter tuner, replacing the development M0-M3 tool.
  - `tools/auto-tune` drives a plain `llama-server` command after `--` and a YAML prompt bank (default: 48 prompts, 8 domains, 18 checked answers). Phase 1 tunes context shaping (-ub, -b, -ubd, -t, -tb, expert-cache margin) through `/tune/context` at 256 and 2048 tokens; phase 2 picks the drafter (MTP, draft model, n-gram-simple/map-k/mod) and its depth/acceptance knobs, then confirms the top candidates with fresh launches.
  - Qwen 3.8 G128 validation: the hand-tuned start was within the tuner's noise; the confirm run showed +0.3% score against a 3.1% spread, so no reliable gain. A run with both phases takes about 70 minutes.

  Details: Automatic parameter tuner (auto-tune).

- **GLM multi-GPU prefill stability and kernel throughput** - removes regime changes that synchronized a multi-GPU prefill and widens the dense and packed MoE paths.
  - Single-ubatch prompt calls take the max-prefill reservation, so a smaller checkpoint call no longer records graph sizes a later 2048-token call outgrows: GLM-5.3 Flash on 3 CMP170HXs, -ub 2048, 4K prompt 639-669 -> 861-938 t/s.
  - Large prefill splits keep lending their executables and reclaim through cudaGraphExecUpdate instead of re-instantiating about 77 ms per 2000-node split: 16K 1427 -> 1492 t/s, 8K 1312 -> 1358.
  - A graph with a node that may call cuBLAS runs eagerly while the BLAS handle does not exist, so hipBLASLt no longer exits inside capture on 32-token or longer n-gram drafts.
  - `GGML_GALLOC_REALLOC_LOG=1` names the first node or count that forces a graph reallocation.
  - Wide dense Q8_0 products dequantize 64 MiB row chunks to F16 and run one cuBLAS GEMM per chunk on SM80 from 128 columns (`GGML_CUDA_DENSE_CHUNKED_*`): GLM-5.3 Flash prefill +8-9% (10K 1649 -> 1787 t/s). Packed MoE prefill uses 32 x J warp tiles and per-stage KS row scales: GLM prefill +4-5%.

  Details: Graph reuse, scheduling, transfers and loading; CUDA/HIP kernels and small fusions.

## Model support, conversion and model-specific correctness

- Ported GLM-5.3 Flash (`glm5next`), including its GGUF converter and runtime architecture.
- Added GLM-5.3's KDA recurrent/linear attention, mHC residual streams, clamped-SwiGLU routed experts, pooled indexer and kpool KV storage.
- Integrated GLM-5.3's library-side MTP graph and memory support, including draft graph-cache reset and recurrent reservation for multiple sequences.
- Ported GLM-5.3's vision tower and integrated its multimodal preprocessing, current clamp metadata key and 0826 image-preprocessor behavior.
- Ported DeepSeek V4.1 (`deepseek41`), including converter metadata, engram computation, hyper-connections and sparse/compressed attention.
- Added and validated DeepSeek V4.1's variable compression ratios and derived per-layer shared KV/indexer stream roles.
- Generalized the DSV4 compressor ratio and shared compressed-stream builders so V4 and V4.1 reuse the same implementation.
- Allowed filtered layers to reuse another layer's KV storage and preserved shared iSWA streams correctly.
- Fixed empty-cache K rotation construction, avoiding an infinite loop.
- Fixed DeepSeek V4.1 conversion of engram hash constants, tensor names, layer IDs and pad-token metadata, including model save/load round trips.
- Used each compressed cache's actual row ratio when trimming and serializing sequence state, rather than V4's fixed ratios.
- Reused the preceding index-source layer's top-k selection on DeepSeek V4.1 reader layers without their own indexer.
- Added DeepSeek V4.1's first-level candidate-block selector, newest-block pinning, metadata checks and fallback warning for older files.
- Protected DeepSeek V4.1's always-used and state-dependent tensors from aggressive quantization while retaining routed-expert eligibility.
- Rejected unsupported DeepSeek V4.1 tensor-split configurations rather than allowing a meta-backend assertion.
- Ported the DeepSeek V4.1 DSpark draft-head converter and runtime.
- Ported Qwen 3.8 Flash Next's NextN/MTP head, GGUF tensor registration and draft-head export.
- Allowed a draft-only Qwen MTP export to borrow the target's embeddings and output head instead of duplicating them; declared the converter's shared-embedding flag and rejected draft-only loading without its required target.
- Fixed Qwen/GLM MTP hidden-state transport through `batch.embd_h`, including the full hyper-connection width, and supplied hidden rows to the sequence-removal capability probe.
- Corrected the MTP head's hyper-connection normalization shape.
- Fixed MiMo MTP to read the hidden-input channel on every head instead of the token-embedding channel.
- Corrected DFlash export of partial rotary dimensions and value scaling.
- Kept DeepSeek V4's mHC mix normalization on its layer's device, avoiding unintended CPU placement.
- Added Muse Spark Open (`muse-spark`, port of upstream #30616): text, vision and audio, with chat-parser adaptation, self-expanded logits graph and the moe_latent_size saver key; cherry-picked the upstream Model-Saver SWA-pattern and architecture roundtrip work (#29042) to cover it.
- MTP drafters read media (mtmd) batches with media embeddings and shifted target hidden states, and take embedding inputs where the architecture requires it; media batch rows use the input embedding width.
- Added the DeepSeek V4.1 vision projector and mtmd path: the converter writes `deepseek41v`, with the V4.1 image-token cap (1024), optional width/height ratio limit and V4.1 resize/pad rules; V4 files without the new keys keep V4 limits.
- Disabled the DeepSeek V4.1 engram for image spans, matching the reference gate.
- Built the GLM-5.3 indexer pool bias on each device from small scalar/query inputs instead of uploading a full-width host bias; total compute buffers at 500K context fall from 26.3 to 8.2 GiB at -ub 2048 (5.9 -> 3.7 GiB at 256, 14.6 -> 5.5 GiB at 1024), KLD unchanged.
- Accepted the `glm5v` projector name (same tower and clamp key as `glm5next`) for Unsloth GLM mmproj files, and made a too-small incremental k-pool update degrade gracefully instead of aborting.
- A V4.1-format DSpark draft without `hc_head` weights (the Unsloth 1017 Q8_0 file) takes V4.1 semantics with a warning instead of failing on the missing V4 `output_hc_fn.weight`.
- Added dynamic vision (`--mmproj-memory dynamic|resident`, dynamic by default): encoder weights stay in host RAM and each encoding loads them into compute memory that the language model releases (CUDA and HIP; other backends stay resident). GLM 5.3 on 3 CMP170HXs: identical output, 2.6 GiB less on CUDA0, +1.2-1.9 s per image.

## DeepSeek V4.1 chat encoding

- Adapted the released 1017 chat template (attribution and DeepSeek MIT license preserved), making the template selectable per model in the converter while V4 and V4-0731 keep their defaults; run-time template overrides still work. Existing GGUFs need `--chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1.jinja`; weights need no re-conversion.
- Deviations from Unsloth kept deliberately: explicit `thinking` overrides, native numeric effort mapping (minimal/low -> 50, medium/high/xhigh -> 75, max -> 100), object arguments enforced, and stable tool-result sorting with the original position in the sort key.
- Added a Jinja `dict()` builtin with shallow-copy and keyword-override semantics and explicit errors on unsupported inputs.
- The parser shares the existing V4 parser behind a V4.1 entry point that detects spaced DSML tags first, derives reasoning stops from the generation prefix, supports mixed string/JSON union types and schema-constrained JSON responses, and sorts tool results itself.
- The sampler reads grammar triggers only for a lazy grammar, so a required tool call is not cut before the grammar runs.
- The shared chat boundary converts integer strings to integer template values, so numeric `reasoning_effort` works for embedded and external templates and command-line defaults, not only model-specific paths; named levels and range validation stay, and the server accepts JSON integers in its `reasoning_effort` field.
- Validation: five official 1017 fixture pairs render byte-identical, `test-jinja` and `test-chat` pass, the converter test passes, and a server tool round trip passes 10 of 10 with thinking off and on, plain and streamed.

## DeepSeek V4.1 CED prefill

- Realized DeepSeek V4.1 Flash's causal encoder-decoder semantics in deployment. Before this, prefill ran all 40 blocks over every token, paying roughly the 16B active cost instead of the intended ~8B; the gap was established by a CED audit.
- Encoder layers run on every row. At the first decoder layer the graph retains a bounded suffix of the last `n_swa + rollback_depth` rows plus the full normalized encoder output for global-KV and index-key projection; decoder attention and MoE run only on that suffix. Separate encoder/decoder sliding-window caches avoid restoring garbage keys.
- Compact prefill is enabled by default (`LLAMA_DSV41_TRIM=1`, read at context creation; `0` restores the full graph, `2` is a bounded-mask numerical reference). Multi-sequence, embedding, prefix-logits and unsupported layouts fall back to the full graph.
- Validated: prefill 1.73x/1.46x/1.90x at 1K/3.9K/65K tokens; 2048-token teacher-forced KL 0.00278 vs the full graph, 98.73% top-1 agreement, PPL ratio 1.009, 129-token control bitwise identical; DSpark task accuracy and draft acceptance parity; microbatch boundary tests.
- Provenance: adapted from DeepInfer (774ce244b8, Apache-2.0) as a C++/ggml re-implementation; deviations are recorded in the public documentation. The pre-snapshot `LLAMA_DSV41_TRIM_SPIKE` was replaced by the production implementation.
- Caveats: saved KV state blobs reject cross-mode restore; perplexity harnesses that scan all rows use the full graph.
- A compacted prompt ubatch that a later ubatch clears keeps one decoder row instead of the full decoder window; previously each 1024-token ubatch read nearly all decoder experts for 128 retired rows (about 1.4 s on 3 CMP170HXs).
- Added layer-major encoder windows for prompts with two to four eligible chunks: each chunk runs its encoder layers one at a time, with each layer's expert bank retained across the chunk graphs and the next layer's bank prefetched. The window rewinds host cache metadata per layer, keeps per-layer tensor state and chronological compressor updates, and resumes the existing decoder graph at the CED boundary. Unsupported inputs and all-row output requests use ordinary execution; a failed or aborted window clears its sequence because encoder layers may already have overwritten the old prefix.
- Staging keep mode records slot residency across graphs, keeps the bank on the GPU that holds the activations and prefetches from the ring that `GGML_SCHED_STAGE_SPREAD` would pick; `GGML_SCHED_STAGE_KEEP_SPREAD=1` restores spreading. The HC stream and pending mix of each chunk live in a device buffer on the first layer's GPU (1280 MiB for 4 x 4096 rows), and later layer graphs read and overwrite it in place.
- The attention inputs of a chunk (masks, cache indices) are computed once by the first layer graph and copied by later graphs; top-k and candidate indices persist in device state; the raw indices of indexed prefill are built by the GPU instead of a CPU REPEAT split. Only graphs that produce shared selections synchronize.
- Layer-bank windows are the default (`LLAMA_DSV41_LAYER_BANK=1`; `2` keeps the schedule without retained banks, `0` restores ordinary prefill). A later chunk shorter than the staging threshold stages on the bank that the first chunk made resident, and the window state counts a partial last ubatch. Retained state is byte-identical to ordinary prefill, and sampled layer features match. 2 R9700 at `-c 300000`, 8K/16K/32K prompt latency: 15.16/29.44/59.38 -> 8.67/14.60/30.33 s.
- Encoder checkpoints stay in pageable host memory. A pinned backend-host trial preserved state and output but increased median request latency by about 33% at 16K and 32K, so the smaller pageable prototype is retained.

## DeepSeek V4.1 long context

- DSpark verification (2-16 query rows) used the dense attention fallback: every layer copied the whole compressed stream and filled a top-k mask as wide as the stream, about 9 GB of copies per verification at 300K, and from 128K it ran out of memory. The gather path gives each query its own sequence and one attention call over the raw window plus its 512 selected rows; `LLAMA_DSV41_VERIFY_GATHER_MIN` (default 4096 compressed rows, 0 disables) selects the smallest stream that takes this path. Decode stays 30.9-36.0 t/s from 128K to 250K.
- Removed full-width compressed masks from the prefill graph: the lightning indexer op takes per-row visible counts (CPU and CUDA/HIP; Vulkan/SYCL/Metal reject it), the indexer scores 512 query rows per call (`LLAMA_DSV41_INDEXER_ROWS`), row validity comes from the counts in attention, and prefill from 131072 context rows uses indexed attention. At 300K the expert cache holds 1235/513 slots (was 740/30), both GPUs stage again, and short-prompt decode is 37.7 t/s.
- Chunked dense prefill attention is the default (`LLAMA_DSV41_ATTN_ROWS=1024`), and indexed prefill from 128K is selected only when that value is 0. The intermittent HSA fault seen with one early rows=512 run did not recur in 10 fresh fault-shape runs after the HIP copy-graph separation, nor in the layer-bank fault soak.
- Bounded the compute buffers that grew with context depth: each indexer score chunk expands top-k and the candidate output before the next chunk, indexer and attention chunk rows follow `LLAMA_CHUNK_MIB` (default 256 MiB; 0 disables), and the candidate source layer keeps I32 kept-block indices and builds the block mask per reader chunk instead of holding a dense F32 mask for all rows. At `-c 300000 -ub 4096` the ROCm0/ROCm1 compute buffers fall from 3664/8989 MiB to 3195/3610 MiB, and the ROCm1 expert cache grows from 492 to 792 slots.
- Compressed KV and indexer keys that reader layers reuse from a source layer are mirrored on the reader's device once, written in graph order right after the source and resynced after state reads and sequence clears (`LLAMA_KV_MIRROR=0` off); 3 R9700s at 15-18K context 118 -> 80.5 ms/verify, 2 R9700s with -ts 1,1 110.7 -> 92.5.
- The radix top-k scratch is sized for the most blocks per row instead of growing with the stream, and every context recaptures its graphs after a pool flush, removing the HIP memory fault near 90K tokens.
- The automatic expert cache disables staging on a GPU whose tier cannot lend any full staging slot (539 MiB tier vs 2302 MiB slot at 300K context), with a warning.
- DeepSeek V4.1 engram gate weights (`engram_q`/`engram_k`) stay on the GPU through BF16 GET_ROWS instead of riding a CPU round trip: 3 CMP170HXs, 4096-token prefill 334 -> 576 t/s.
- `LLAMA_PP_ALLOW_OVERRIDES=1` keeps pipeline parallelism with non-embedding tensor overrides, such as layers' experts held on the CPU: 3 CMP170HXs, 4096-token prefill 576 -> 619 t/s.
- The server prompt checkpoint one ubatch before the end of a prompt is off by default (`LLAMA_SERVER_CKPT_TAIL_UBATCH=1` restores it); with CED each batch that ended at a full ubatch re-ran the decoder window, costing prefill 4.50 -> 3.13 s at 2048 tokens and 18.5 -> 16.2 s at 16K.

## KV window and GLM long context

- Added a hybrid KV window: `LLAMA_KV_RESIDENT_CELLS=<cells>` or `<dev>=<cells>,...` keeps the first cells of each attention KV stream in VRAM and maps the cells past them to pinned host memory at the same virtual addresses (CUDA VMM, `kv-region.cu`). The indexer cache stays in VRAM unless `LLAMA_KV_IDX_RESIDENT_CELLS` windows it too. Without `LLAMA_KV_RESIDENT_CELLS` the attention cache stays all in VRAM.
- Readers never read the host cells in place: a ubatch with at least `LLAMA_KV_WINDOW_STAGE_MIN_TOKENS` (default 17) tokens per stream copies the layer's K view to a compute tensor, and a narrower ubatch gathers the selected cells of each query row with `get_rows` before indexed attention. Both paths run on every ubatch of a windowed layer, not only past the resident cells, so the graph shape does not change with depth and the full-depth reservation stays valid.
- A verify batch gathers each selected cell once for all its query rows: a per-cell map records the winning position, losing rows gather resident cell 0, and attention indices point at the winner (`LLAMA_KV_WINDOW_UNION=0` restores the per-row gather). F16 rows are gathered as I32 pairs, so there is no F32 intermediate; GLM-5.3 Flash at 100K, DFlash2, 23.1 -> 25.7 t/s.
- `LLAMA_KV_RESIDENT_CELLS=auto` uses the `--fit` dry run: the model is measured with the window graphs and the cache in VRAM, and each device's deficit plus `LLAMA_KV_WINDOW_MARGIN_MIB` (default 1024, or a per-device list such as `1024,CUDA2=2560`) is written to `LLAMA_KV_WINDOW_HOST_MIB`; the window moves that much of the cache to host memory. The indexer cache keeps 4096 resident cells with auto unless `LLAMA_KV_IDX_RESIDENT_CELLS` is set, the fit measures a draft that needs a main-model context (DFlash, EAGLE3) with a dry-run main context, and the memory breakdown does not count host cells as device memory.
- The window applies only to the pooled sparse attention and indexer of glm5next. Other models ignore the variables and log a warning. State read, sequence copy, add and divide invalidate only the pools of that sequence (non-unified), so a prompt-cache restore into one slot no longer rebuilds the other slots' pools.
- The context reserves the widest gather graph at load. `GGML_CUDA_GRAPH_CACHE_LARGE` bounds the large CUDA graphs (default 64), and the post-load VRAM growth is bounded and mostly depth-independent.
- GLM indexer buffers that grew with depth are chunked: pool scores and the device pool mask select a chunk of query rows at a time, and the pool rebuild compresses a chunk of pools at a time. The chunk count comes from the full cache depth so every graph matches the reservation. GLM defaults to 512 MiB; `LLAMA_CHUNK_MIB` overrides the limit for all models. At 512 MiB, `-c 300000 -ub 1024` stays on the unchunked graph (1210/1290/1328 MiB) and `-c 1048576 -ub 4096` falls from 8636/8957/9108 MiB to 2945/3898/3837 MiB. With `LLAMA_CHUNK_MIB=256`, 300K falls to 737/988/935 MiB and 1M to 2997/4179/3861 MiB. Outputs are identical from 2K to 250K tokens; the measured 2-3% prefill cost at 100K-250K comes from that 256 MiB limit, where the 300K graph chunks.
- Validated on GLM-5.3 Flash, 3 CMP170HXs: `-c 1048576 LLAMA_KV_RESIDENT_CELLS=CUDA1=32768,CUDA2=131072` loads and runs a 300K prompt; 100K prefill 1613 t/s against 1627 at `-c 655360` without the window; DFlash2 decode 28.7 t/s (34.1 without). `auto` at `-c 1572864 -np 3` moves 0/4832/5524 MiB to host memory, three concurrent 300K prompts complete, and fresh outputs are bitwise equal to the run without the window. The gather path has KLD 0.0113 (staging 0.0082) against the 0.019 GLM summation floor.

## Parallel prompt processing and per-sequence KV streams

- DeepSeek V4.1 with `-np > 1` and a split KV cache gave a sequence with more than `n_keep_dec` tokens in the batch its own prompt ubatches, ahead of the other sequences, through `llama_batch_allocr::split_seq_id`; the remaining multi-sequence ubatches hold at most `n_keep_dec` tokens of each sequence. This keeps CED compact prefill and layer-bank windows for prompts that share a server batch.
- The DeepSeek V4 cache reserves the one-sequence prompt shape over one stream and the largest multi-sequence ubatch (`n_seq_max * n_keep_dec` tokens) after a memory update, instead of a full ubatch over all streams with all decoder rows. The one-sequence max-prefill reservation is skipped when the memory exposes multiple streams, which had split attention rows and aborted the output reshape.
- Extracted target layer inputs are copied to their batch rows instead of being written in processing order. A split that reorders tokens (an equal split of several sequences, or the DSV4 long-prompt-first cache) previously gave the draft the features of other tokens.
- Indexer and dense prefill attention are chunked per stream when several sequences are active; each stream uses its own q block, K concat and top-k views, and indexer rows use strided views. At `-np 3 -c 900000` the ROCm0/ROCm1 compute buffers fall from 13198/18066 MiB to 3701/11576 MiB and host compute from 5602 to 1008 MiB; decode and verify calls of 64 queries or fewer keep the multi-stream call. Perplexity with 2 sequences per batch equals 1 sequence per batch (mean KLD 0.000000).
- Results: three concurrent 16K prompts finish in 45 s (62-86 s before); one 16K prompt alone runs at 1056-1108 t/s, equal to `-np 1`. Expert-cache slots at `-np 3 -c 900000` change from 1065/246 to 990/390 because the one-sequence prompt shape is now reserved.
- Ported the same policy to the GLM hybrid memory (pooled indexer, more than one stream): a sequence with more than `LLAMA_HYBRID_SEQ_SPLIT_TOKENS` (default 16) tokens takes its own ubatches, reservations use 16 tokens per sequence plus the one-sequence prompt shapes, and `stage_k` compares tokens per stream against the staging threshold so a multi-sequence verify gathers instead of staging every stream. At `-c 1572864 -np 3` with KV and indexer windows, DFlash2 now starts, a 16K prompt runs at 1266-1704 t/s, and three concurrent 300K prompts complete in 10.6 min (about 1400 t/s aggregate). Perplexity with 2 sequences per batch matches 1 (mean KLD 0.000000).
- Server `-np auto` on `deepseek4`/`deepseek41` keeps 4 slots but disables unified KV, so the per-stream compute-buffer limit applies. Without `-c` the 4 slots share the training context (256K each of 1M), otherwise the fit gives each slot the full training context, and `-c 0` also gives each the full context. The context warns when DeepSeek V4 runs unified with more than one sequence. The llama-swap DeepSeek command (no `-np`, no `-c`) loads with 938/662 cache slots and completes four parallel requests.
- `ggml-alloc` now records each node's source positions, view source and input/output flags at reserve and makes a new layout when the edges differ. Without this, draft graphs with the same sizes but different sampler edges reused a layout that freed candidate tensors early; DSpark draft acceptance fell from 45/51 to 0-7/165-183. Perplexity and mixed-batch outputs are unchanged after the fix.

## Background prompt work, staggered verify and verify streams

- Staggered verify (`LLAMA_STAGGERED_VERIFY`, default on where it applies) gives each sequence of a hybrid-memory batch its own ubatches, including short verify sequences, and gives each sequence an alternate graph slot and scheduler. Pipeline parallelism then overlaps sequence A's verify on one device with sequence B's on another; a reused graph waits only for the scheduler it reuses (`ggml_backend_sched_wait_graph`), not all backends, and lazy backend-sampler outputs are kept per ubatch. It applies only with pipeline parallelism, per-sequence KV streams (no unified cache), more than one sequence, a hybrid-memory split, and enough alternate graph slots and tokens; `LLAMA_STAGGERED_VERIFY=0` disables it. GLM-5.3 Flash, 3 CMP170HXs, DFlash2, `-np 3`: round 70 -> 61 ms at 2 slots and 91 -> 79.5 ms at 3 slots; aggregate 97-102 -> 117-128 t/s at 3 slots.
- Verify streams (default on where they apply since 2026-11-13, `LLAMA_VERIFY_STREAMS=0` opts out) give the alternate graph slots a second set of GPU backends whose streams have the highest priority. A prompt decode marked with `llama_set_prompt_background` does not block `llama_synchronize` or the output getters; each background decode gets a ticket with per-device events, and its device feature rows go to one of four background regions of the feature buffer for DFlash. A sequence that moves between the two stream sets waits on the host for the set it leaves, and short sequences go first in a hybrid-memory batch. The mode is off with a unified KV cache, without alternate graph slots, or with an expert cache, and it requires staggered verify. With verify streams the server defaults `CUDA_MODULE_LOADING` to `EAGER` unless it is set.
- With verify streams, a server round that has generating slots adds at most one prompt chunk (`LLAMA_SERVER_PREFILL_CAP` or 1024 tokens) while fewer than `LLAMA_VERIFY_STREAMS_DEPTH` (default 3) chunks are in flight. The DFlash draft ingests the feature rows of a chunk when its ticket completes (`common_speculative_process_rows`), and the last `LLAMA_VERIFY_STREAMS_TAIL` (default 16) prompt tokens run alone after the background work.
- `LLAMA_SERVER_PREFILL_CAP=N` caps the prompt tokens admitted to a batch that also has generating tokens, so a decoding slot does not wait for all prompt ubatches of the batch. `LLAMA_STAGGERED_MIXED_UBATCH=N` cuts the prompt of such a mixed batch into at most N-token ubatches so they pipeline over the devices. GLM-5.3 Flash, 3 CMP, 12.8K prompt while one slot decodes: longest decode gap 4.4-5.0 -> 1.2-1.4 s (cap 1024, mixed ubatch 512), prompt 1415-1623 -> 899-921 t/s.
- Prompt input staging keeps pinned snapshots of host inputs in a bounded scheduler ring and uploads them on the destination stream; events protect buffer reuse, including repeated graphs, and callers can reuse their input arrays after submission. It is opt-in and falls back to synchronous copies without pinned storage or upload events. The main prompt scheduler enables it under verify streams (`LLAMA_VERIFY_STREAMS_PROMPT_INPUT_ASYNC=0` restores synchronous copies). Each slot starts at 4 MiB (`GGML_SCHED_INPUT_STAGING_MIN_KIB`): growing a slot frees pinned memory, which waits for all work of the device, and the inputs grew with the KV view in sub-MiB steps, so the first prompt to reach a new length drained the prefill pipeline about 60 times (GLM-5.3 Flash first 60K prompt 1169 -> 1604 t/s, equal to verify streams off).
- Verify and DFlash schedulers opt into asynchronous host input uploads while background prompt work runs on separate streams; input buffers stay valid through graph completion, including single-copy graphs, and the owning scheduler is waited for before its inputs are reused or rebuilt. `LLAMA_VERIFY_STREAMS_INPUT_ASYNC=0` restores the synchronous path.
- Background prefill reservations are bounded by the prompt end: the first main-scheduler ubatch is checked even when verify ubatches precede it, the warm pool-cache graph is reserved for the dense crossover or the background prompt end, and chunks that need a new layout wait until pending background tickets finish so layout synchronization does not drain queued chunks on the server thread. `LLAMA_VERIFY_STREAMS_KV_STEP` (default 16384, padded to 256) sets the indexed reservation step for background prompts.
- GLM prompt-tail indexer partitions stay stable: the indexer query partition count for single-sequence prompt batches uses the main scheduler's reserved token width, and live query rows are balanced across those partitions. A 1020-token tail no longer replaces the allocation plan and drains all pipeline stages at a 512K-token slot context; `LLAMA_GLM5_STABLE_PREFILL_CHUNKS=0` restores the previous behavior.
- Prompt checkpoints no longer drain each decode round; see KV, context and server state management.

## Quantization formats and quantization tools

- Ported ik_llama.cpp's nine formats: IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K, IQ2_KS, IQ3_KS, IQ4_KS and IQ5_KS, retaining the donor's type/file IDs.
- Added CPU quantization, dequantization, dot products and IQK GEMM for the imported types.
- Added IQK execution for ordinary matrix products and routed-expert products, including a fused gate/up/SwiGLU CPU pass.
- Added per-row metadata support for the KS formats, including row-aware copies, dequantization, requantization and backend tests.
- Added CUDA/HIP dequantization, copy-to-F32, vector dots, MMVQ and MMQ support for the imported types.
- Fixed the MMQ proxy-configuration sentinel so unsupported imported-type configurations do not launch placeholder kernels.
- Fixed CUDA/HIP MMQ activation-buffer padding for the largest column tile and restored usable IQ_K/IQ_KS operation support on ROCm.
- Added load-time backend capability checks for IQ_K/IQ_KS and Q4_G128 instead of discovering unsupported weights during execution. The temporary HIP IQK veto was subsequently removed after the padding fix.
- Added Q4_G128: symmetric INT4 weights with one BF16 scale per 128 values, including GGUF/type registration and quantize/dequantize support.
- Added Q4_G128 to the ordinary CUDA/HIP matmul kernels: MMVQ, MMQ, grouped MMQ and dequantization, so dense MUL_MAT on NVIDIA and HIP and routed MUL_MAT_ID on NVIDIA run it natively; equal to Q4_0 speed at 4.125 bpw, or 2-7% faster at 1-512 tokens on the CMP170HXs.
- Fixed HIP staging of Q4_G128 experts: the layout flag is cleared on the compute stream (NaN with pack-on-upload) and staging slots take the newest G128 tensor's shape.
- Added a converter from NVFP4 routed experts to Q4_G128 and conversion support for compatible Q4_0 experts. (`156c988ba`; `tools/moe-int4-convert/`)
- Added a streaming MiMo expert converter from MXFP4 to G128 with Q8 dense weights, bounded parallel conversion and an eight-worker limit.
- Added optional activation-mean collection alongside imatrix second moments (`LLAMA_IMATRIX_COLLECT_MEANS`) and storage/loading of those moments.
- Added validation when combining imatrices so missing or incompatible activation means do not silently feed refinement.
- Added opt-in activation-mean error feedback for Q2_0 and Q4_0 quantization (`--gptq-feedback`), using a bounded rank-one covariance approximation. This is an experimental refinement method, not a full GPTQ implementation.
- Extended that feedback to Q2_K, Q3_K, Q4_K, Q5_K and Q6_K without changing their stored block formats.
- Extended activation feedback to Q4_G128 and MXFP4.
- Added opt-in bounded local Gumbel-softmax refinement for Q2_0/Q4_0 (`--gsq-steps`) and tensor-name filtering (`--refine-tensor`).
- Added correction bounds, refinement-input checks and a row-level damage guard that retains the baseline quantization when the proxy objective does not improve.
- Added imatrix-aware MXFP4 block-exponent search over the baseline exponent and its two neighbors; ordinary no-imatrix encoding stays unchanged.
- Builds DeepSeek V4's MXFP4 expert tensors lazily during conversion instead of holding all repacked experts in memory; DeepSeek V4.1 used about 240 GB and failed at layer 35 before this.
- `llama-quantize --split-tensors` writes chosen tensors to a second split file, and with `--only-tensors` those tensors alone form a split; the loader reads splits by tensor name. This lets large lazy tables live on another disk.
- The DeepSeek V4.1 converter stages engram tables in the scratch directory and releases each staging file after mapping it, so a conversion no longer fills the root file system or leaves about 104 GB files behind.

## Automatic quantizer and quantization planning

- Added the automatic quantizer (auto-quantizer V2): predicts each tensor's KL against the BF16 teacher (teacher loaded without VRAM via `-cmoe`) from llama-quant-damage's sampled relative error, with per-role sensitivity over layer bands, a smoothness penalty and measured super-additive composition. Calibration cost (2 x bands x roles + recipes passes; 48 passes and 24.5 min on Qwen3.6-35B-A3B) is independent of the candidate qtypes and layers, and requires an imatrix. Includes auto-quantizer V1's baseline-anchored advisor and profiler-trace workloads.
- Solved the per-device plan as an exact multiple-choice knapsack under byte budgets, a loss ceiling and speed/precision importance; MTP blocks are found by `blk.N.nextn.*` tensor names and `nextn_predict_layers` and pinned at the ceiling.
- Added the end-to-end `auto-quantize` CLI (`tools/auto-quantize/`): `setup`/`plan`/`run`/`capacity`/`show`, measured device capacities (`llama-bench --list-devices` minus `--keep-free`), reserves from llama-fit-params including `--mmproj`, pooled byte caps, `--max-size`/`--minimize`, disk-space preflight, progress lines and a resumable run state.
- Calibrates in one GPU layer-split session (`-sm layer`, about 10x faster than `-cmoe`) when the floor fits several GPUs, with `-cmoe` fallback for unloadable probes; sessions restart after process end and on SIGTERM; encode no longer stops after 30 minutes and `run` caps one operation measurement at one hour.
- Added weighted multi-corpus imatrix collection and merging with preserved activation sums; the default calibration text is the bundled ubergarm corpus. Unweighted configurations keep the previous exact behavior.
- Added persistent native tensor hotswap for llama-perplexity: patch/restore individual tensors in a loaded model without reloading (Q4_G128 and NVFP4 patches, parallel sessions via an extra environment, opt-in per-token KL output through `LLAMA_KLD_TOKENS_FILE`); patched KL equals direct load and patch reloads take 4-5 s.
- Added `llama-quantize --only-tensors`, which writes a partial GGUF with the listed tensors; auto-quantize calibration quantizes per-pass patches from the source instead of probe models.
- Added server `/tune/context` and `/tune/kld` endpoints behind `--tune-endpoints` (default off), context recreation on the loaded model and per-context cache-worker/refill env knobs, for tuning against live weights. The Python launch-parameter tuner from that period (tools/auto-tune, M0-M3) was later replaced by auto-tune v1; see Automatic parameter tuner (auto-tune).
- Validated: Qwen3.6-35B-A3B plans beat the local-metric plan by -3.4%+-1.5 and -13.1%+-2.6 KL at +2.2/+4.2 GB; predictions within about 6% for 5 of 6 held-out witnesses.
- Calibration and encode speed: `quantize_threads` (default: allowed CPUs less 8) gives 1.5-1.65x on the compute-bound kernels, and the first `calibration.devices` entry is the measurement GPU for the teacher and `-cmoe` sessions; the DeepSeek V4.1 estimate drops from about 11-15 h to about 6.8 h. A backend-library identity in the operation cache re-measures operations after a rebuild under `GGML_BACKEND_DL`.
- Calibration lifetime: `--keep-calibration` keeps a finished calibration across rebuilds; a stage chains only on prepare inputs that change what a pass measures; interrupted passes continue; replaced runs move to `SCRATCH/calibration.replaced` with their record, so a later `--keep-calibration` can move them back; band patches stay in a bounded patch cache (32 GiB free gate) across probes.
- Session robustness: devices that cannot hold the floor with `SPLIT_MARGIN` to spare are skipped with the reason, and the calibration probe model is sized at the top probe qtype under the calibrate space check.
- Deployment realism in plans and verify: reserves come from `llama-fit-params --spec-type draft-mtp` (MTP weights, recurrent states, per-device draft KV and compute; within 22 MiB on GLM 5.3 at 200K), verify runs with the MTP draft, MTP blocks stay in the tensor split at zero bytes, and each GPU reserves measured non-GGML runtime memory (defaults 768 MiB, 1 GiB with MTP, 1.75 GiB on the MTP block's GPU).
- The speed model replaces the uniform slots/experts prior with a measured expert-cache hit curve: one routing trace per model (draft and projector included) replayed offline per slot size; server hit shares match within 0.04 pp at 456 slots and 0.22 pp at 913. `--cache-mib auto` sizes the cache from free memory after the draft model, projector and compute buffers, and another `--cache-mib` re-scores the plan without recalibration.
- `--runtime` (delphi, llama, ik_llama) filters plan types to what the target runtime loads, with errors for named types the runtime cannot load; prepare and calibrate stay valid across runtime changes.
- One base stage rewrites the source once: lazy tables at `--lazy-type` in their own shard (positionally last, so it can sit on another disk), all other tensors bit-exact copies; calibration, teacher logits and encode use the base, and encoded models share the lazy shard by link.
- Loss-only weights (no speed weight or bound) skip the timing measurement and baseline profile: the frontier is loss against bytes, and a later timed run re-runs prepare while keeping calibration.
- Verify hardening: a long prefill (up to 40000 tokens, past GLM's indexed-attention crossover) runs before the free-memory read, a full GPU is reported instead of stopping, a converter-incompatible projector only warns, and a server exit during a request is an error.
- Removed prose fields from the JSON outputs because the same facts are data fields and no code read them.

## CPU execution and NUMA

- Added `--numa tensors`: split host routed-expert output rows across sockets, place the corresponding tensor ranges locally and combine the computation without copying whole expert weights to one socket.
- Integrated the NUMA row split with imported IQK `MUL_MAT_ID` kernels and their activation-conversion barrier.
- Enabled the existing Q2_K repacked GEMV/GEMM layout on AVX2 CPUs, rather than unnecessarily requiring AVX-512.
- Added mapped-file read-ahead for tensors copied once into NUMA storage at load time.
- Parallelized CPU `GET_ROWS` from 256 rows, helping wider embedding/engram gathers.
- Added Q4_G128 CPU inference using Q8 activations and AVX2 integer dots.
- Added G128 multi-row execution that reuses decoded weights across an expert's verification tokens, with four rows scheduled as two pairs.
- Added configurable singleton G128 weight prefetch.
- Added precomputed G128 activation panels containing scales and zero-point corrections.
- Added a register-tiled G128 CPU GEMM/panel path for experts with four or more rows, with an ordinary rows-kernel fallback for small groups.
- Fused CPU G128 gate, up and SwiGLU work for eligible ordinary and partial expert products.
- Reduced CPU split overhead by running small data-movement graphs on the calling thread.
- Parallelized zeroing of rows excluded by partial routed-expert execution instead of making one worker zero them while others wait.
- Added dynamically claimed NUMA-local expert-row pieces, with two pieces per worker by default and a configurable chunk count.
- Added node-local MUL_MAT_ID state under `--numa tensors`: each node keeps its converted activations, G128 panels, row map and piece counters in its own memory and waits only for its own workers after conversion, with cross-node piece stealing, nearest node first (both bitwise identical; `GGML_NUMA_NODE_LOCAL=0` / `GGML_NUMA_STEAL=0` restore the old path).
- Under `--numa tensors`, the CPU compute buffer is pinned host memory with pages interleaved over all NUMA nodes, instead of the first GPU's pinned memory (`LLAMA_NUMA_HOST_INTERLEAVE=0` restores); MiMo NPS4 decode 34.64 -> 37.63 t/s.
- Kept the main thread pinned between CPU graphs instead of pin/unpin per split (`GGML_NUMA_UNPIN_MAIN=1` restores), and confined scheduler, staging, refill, loader and CPU-backend threads to the allowed CPU set from `--cpu-mask`/`--cpu-range`.
- Added batched AVX2 vec_dot for expert dots in `MUL_MAT_ID`, ported from buun-llama-cpp: each weight block is decoded once and dotted against up to four activation rows for 15 types (Q2_K..Q6_K, IQ2_XXS..IQ4_NL, Q4_0, Q8_0, MXFP4), bitwise identical to the per-row dots (`GGML_CPU_BATCHED_VEC_DOT=0` for A/B); isolated MUL_MAT_ID median 183 -> 140 us.
- Vectorized runtime Q8_K activation quantization with AVX2, exactly reference-equivalent (Strata provenance, MIT header kept): converter 3.9-4x faster, complete IQ3_K FFN +1.69%.
- Extended IQK gate/up/SwiGLU fusion to partial expert operations (valid-route row maps, inactive-output zeroing) and plain SwiGLU, exact in match (`GGML_CPU_MOE_FUSE_UP_GATE=0` disables); isolated fused FFN +4.0-6.3% plain, +4.3-10.1% clipped.
- Added an opt-in per-thread CPU graph/node timeline (`GGML_CPU_TIMELINE`).
- Added CPU routed-MoE phase profiling (`GGML_MOE_PROF`) and tools to inspect routed-row multiplicities.
- `--numa tensors` graphs now split their workers over the NUMA nodes per graph (`ggml_numa_prepare_workers`), so a graph planned with fewer threads than the pool no longer leaves later nodes without row owners; a node count larger than the worker count keeps the ordinary schedule instead of aborting, and mlock runs after the node loaders touch their ranges. Qwen 3.8 on 2 R9700s + RAM at -t 56: tg 26.2, pp 702 (single-socket numactl -N0 -m0 -t 28: 21.5/647).
- NUMA Q8_K activation conversion balances row and block ranges across each node's workers, so short expert activation rows no longer leave a worker idle and shared output lines split between nodes (`GGML_NUMA_QUANT_FLAT=0` restores the old schedule).
- The CPU repack buffer skips out-of-range expert ids that partial `MUL_MAT_ID` uses for GPU-resident experts, so repacked weights (MXFP4/CPU_REPACK) support partial routed execution.
- IQ4_KS decodes two groups per step at 2-4 activation rows, lowering the live vector set (`GGML_IQ4_KS_KERNEL=base` for comparison).
- CPU routed-expert prefetch: `GGML_CPU_EXPERT_PREFETCH` (default 6, active under `--numa tensors`) reads the predicted experts of the next CPU split during the routing-split wait; `GGML_CPU_EXPERT_PREFETCH_STATS=1` prints hit rate, read time and remaining wait, and `GGML_CPU_EXPERT_PREFETCH_DRY=1` predicts without reads.
- Router pre-gating: `LLAMA_MOE_PREGATE` (default on) computes the next layer's top-k on the current layer's router input during its routing split and sends cold ids, most selected first, with the expert-cache packet; output is unchanged and it applies only to decode batches with both layers on one GPU. On DeepSeek V4.1 (2 R9700s + RAM, 48 threads, DSpark, prefetch 6): cold experts read ahead 22% -> 44.5%, CPU expert time -19%, verify 72.79 -> 67.84 ms, 37.05 -> 39.48 t/s.
- Engram readers restore the application helper CPU mask after their caller is pinned by a compute pool, instead of inheriting that CPU confinement; the Linux reader threads are named for placement inspection.
- DeepSeek Engram lookups prefetch on eight host readers (`LLAMA_ENGRAM_PREFETCH`, default 8); Qwen PLE host rows are prefetched before their layer and their collection stays in the input split.

## Hybrid expert placement and CPU-to-GPU streaming

- Added a transfer-aware offload cost equation that compares CPU computation plus activation/result transfers with GPU computation plus weight uploads.
- Calibrated CPU execution and device transfer rates, including PCI/NUMA transfer identity, before choosing host-weight placement.
- Configured CPU threads before graph placement and invalidated reusable placement when thread counts change.
- Calibrated routed-expert cost directly, keyed by shape and power-of-two token buckets.
- Accounted for the expected distinct selected-expert union and avoided reading routing IDs back when wide prefill is expected to use the complete expert set.
- Fixed the transfer-aware placement crash for DeepSeek Q2_K.
- Added persistent calibration files (`GGML_OFFLOAD_CALIBRATION`) and fastest-of-three calibration sampling for reproducible placement decisions.
- Added asynchronous host-weight prefetch into a GPU staging ring (`GGML_SCHED_STAGE`) so uploads of later splits overlap earlier GPU computation. CUDA and HIP hooks exist; the main deployment validation is on ROCm.
- Added pinned, chunked host bounce buffers and parallel memcpy workers for pageable expert weights.
- Added optional direct registration of host weight ranges for DMA without bounce copies (`GGML_SCHED_STAGE_REGISTER`).
- Protected staging-slot reuse with stream/event ordering and host-side reuse waits.
- Bound/pinned staging workers to CPUs near their GPU and exposed worker/chunk/bounce/slot controls.
- Added one independent staging worker/ring per device.
- Added in-place consumption of staged expert weights instead of copying them again into split input buffers.
- Kept each staged layer's expert projections together so the fused HIP G128 FFN can execute them as one region.
- Added stream-ordered packing on upload and host-side packing so the compute stream no longer repacks every staged G128 layer.
- Added AVX2 host packing for staged G128 experts.
- Added `GGML_SCHED_STAGE_SPREAD` to distribute streamed MoE layers across multiple GPUs and use multiple host links.
- Added an explicit CPU-versus-streamed-GPU batch threshold (`GGML_SCHED_STAGE_MIN_TOKENS`, default 512), keeping narrow routed batches on CPU; later, an unset threshold is measured at warmup (see below).
- Deferred prompt route recording and made its IDs contiguous so recording does not disrupt streamed prefill.
- Added staging statistics and a per-job staging timeline (`GGML_SCHED_STAGE_STATS`, `GGML_SCHED_STAGE_TRACE`).
- Added partial routed-expert products and an optional two-device prefill expert split (`LLAMA_MOE_SPLIT`) while retaining host-resident authoritative weights.
- Added an optional DeepSeek V4.1 attention-head split over the auxiliary device, including resident head-weight slices and an experimental indexer-query split.
- Added selected-expert upload overlap with cold CPU prefill (S2): a scheduler worker submits local transient-bank GPU splits while the caller runs cold CPU experts; default on (`LLAMA_MOE_PREFILL_OVERLAP=0` disables); MiMo 1024-token TTFT 3.679 -> 2.795 s.
- Generalized the upload overlap beyond G128 banks: the uploader picks packed G128 or ordinary bytes per bank, validated bitwise on mixed projections and multiple type groups.
- Shared the staging copy and pack workers between overlapping submissions instead of duplicating them per split.
- Added short-prefill expert sharing (S1, opt-in `LLAMA_MOE_PREFILL_SHARE`): per layer, GPU residents stay resident, low-reuse cold experts run on CPU NUMA RAM and selected high-reuse cold experts stream into a bounded transient bank, with one merge per FFN; 1024-token first token 4.572 -> 3.667 s against the one-GPU CPU control.
- Staging is default-on (`GGML_SCHED_STAGE=0` restores off) with 2 slots per spread GPU; nothing is allocated without routed experts in host RAM, and the "no backend supports staging" warning requires an explicit request. DeepSeek V4.1 on 2 R9700s: 4K prefill 444 -> 637 t/s, 1K prefill 258 -> 388 t/s, decode unchanged.
- The CPU-versus-streamed threshold is measured at warmup once per context when `GGML_SCHED_STAGE_MIN_TOKENS` is unset: throwaway warm prompt, CPU expert timing at two sizes, one 64 MiB pinned copy for the link rate, streamed timing at two sizes, then the first crossover. On Qwen 3.8 G128 (R9700, -cmoe) it measured 552 where the sweep crosses near 512 at 16 threads and never streams at 56 threads; startup grows a few seconds at -ub 1024 and up to ~27 s at -ub 4096/16 threads.
- Only batches that `ggml_backend_dev_offload_op` accepts are sent to the GPU, and a failed dot calibration is cached; 1-token `MUL_MAT_ID` of host experts no longer stalls decode (Qwen 3.8 decode with CPU experts 13 -> 26 t/s).
- A GPU whose expert-cache tier cannot lend any full staging slot turns staging off with a warning instead of aborting the first long prompt.
- The shared GPU FFN overlaps cold CPU experts (`LLAMA_MOE_SHARED_OVERLAP`, default on); combined with Engram prefetch, DeepSeek V4.1 DSpark at 1.5 R9700s gains about 4% decode.

## Hot-expert cache

- Added GPU-hot/CPU-cold expert execution using frequency-selected resident banks while keeping original NUMA host weights for cold computation.
- Combined CPU and GPU expert outputs on the GPU, preserving the slot ordering before the weighted reduction.
- Transferred cold-branch activation and router inputs in one packet instead of separate CPU handoffs.
- Reserved graph metadata for both expert branches, kept renamed model outputs on the GPU merge path and limited the experimental CPU-prefill control to prompt processing.
- Added safe updates of individual experts in packed HIP G128 banks without unpacking/repacking the entire bank.
- Added observed-routing selection from a bounded recent-row window, optional offline profile priors and a resident hysteresis bonus.
- Refreshed membership at prompt boundaries while excluding warmup from policy evidence.
- Added decode-time background refresh through pinned host buffers and device scratch, followed by bounded replacement at a safe batch boundary.
- Added configurable refresh cadence and newcomer limits, plus static/profile-only modes for comparisons.
- Added selection-hit and distinct-expert-hit statistics, routing dumps and offline cache-policy replay.
- Allowed the prefill staging ring to borrow cache memory, invalidate the borrowed residents and restore the hot set after prefill.
- Prevented lending a staging slot at a packed bank's base address, preserving the packed-bank ownership/lookup contract.
- Generalized the cache beyond MiMo/G128 to compatible MoE architectures and any expert type supported by the selected GPU backend. Eligibility still requires the supported separate gate/up/down SiLU graph form.
- Supported different slab sizes for gate/up/down and per-layer quantization mixes through banks grouped by expert-type combination.
- Added multiple cache tiers on arbitrary GPUs, a common ranking over those tiers, and generation-based graph invalidation when tiers change.
- Reduced remote GPU expert results before transferring them to the layer GPU.
- Built remote tier inputs from one packet copy and ordered local/remote gathers to avoid an early remote synchronization.
- Added NUMA-pinned scheduler workers to submit remote GPU tiers while the main thread computes CPU experts, with stream-specific cross-device copies and landing buffers.
- Added opt-in use of cached GPU experts for intermediate prompt batches (`LLAMA_MOE_CACHE_PROMPT_ROWS`), reducing each CPU/GPU part where it runs before transfer.
- Added auto mode, now the default: `LLAMA_MOE_CACHE_MIB` unset or `auto` enables the cache automatically for models with routed experts in host RAM, one tier per `--device` GPU sized to free VRAM minus a 1536 MiB margin (later 1024 MiB); tiers are allocated after draft contexts load; unservable models skip the cache instead of erroring. DeepSeek V4.1 Flash DSpark 36.3 vs 34.3 t/s without the cache, 90.5% selection hits.
- Used clipped SwiGLU in the expert-cache FFN path where the architecture requires it, instead of plain SwiGLU (wrong numbers on DeepSeek V4/V4.1).
- Reduced the CPU part of cached decode batches on the CPU: the CPU partition computes its own weighted sum in one fused op and transfers [n_embd, n_tokens] per layer (80 KiB at 4 tokens) instead of every expert slot (480 KiB/layer on DeepSeek V4); `LLAMA_MOE_CACHE_CPU_REDUCE=0` restores the GPU merge; default on.
- Added opt-in NUMA options for the refill copy: interleaved pinned staging buffer (`LLAMA_MOE_CACHE_REFILL_INTERLEAVE=1`) and per-node copy threads (`LLAMA_MOE_CACHE_REFILL_NUMA=1`) for NPS4; MiMo NPS4 refresh per prompt 210 -> ~150 ms.
- Staging rings and the expert-cache refill scratch shrink to available memory instead of crashing, fixing the DeepSeek + DSpark 65K prefill crash with the auto cache.
- Cached layers may keep part of their experts in VRAM and the rest in host RAM; each layer uses the tiers of its own GPU by default (local-first), and auto makes tiers only there (`LLAMA_MOE_CACHE_AUTO_REMOTE=1` restores the earlier behavior); a remote tier costs a per-layer copy pair.
- Static tiers: `LLAMA_MOE_CACHE_STATIC=<dev>[,...]` holds the session's most-selected experts with at most `LLAMA_MOE_CACHE_STATIC_SWAPS` (default 8) members changed per prompt; other tiers keep the window-score behavior; `LLAMA_MOE_CACHE_REFRESH_MAX` caps the post-prompt refresh and remaining slots fill in the background.
- Cache roles and explicit per-device auto reserves override legacy static membership and the scalar reserve; pinned tiers span layer devices, keep single ownership through moves/refreshes/staging transitions, and feed from the global heat ranking before domain-local pools.
- Staging-slot lending: `lend()` takes from a rotating tier on the same backend before a static one, and from the top of a tier's bank memory across banks and type groups when no single bank fits a slot; experts with slabs in borrowed memory leave their group for that period. (Qwen Q4_K_M on a 1 GiB tier: 280 of 563 slots became lendable.)
- The NVIDIA backend writes cache slots as a stream-ordered device copy (packed-in-place banks abort rather than corrupt); before this, NVIDIA caches never refreshed during decode.
- Decode-time refresh defaults to 64 experts per cadence (`LLAMA_MOE_CACHE_ASYNC_EXPERTS`), with the refresh scratch kept outside the tiers and the default margin lowered to 1024 MiB: DeepSeek V4.1 on 2 R9700s 36.8/37.2/37.9/37.2/36.7 t/s at 32/64/96/128/192.
- A GPU whose tier cannot lend any staging slot turns staging off with a warning instead of aborting the first long prompt.

## On-demand VRAM and sharing between servers

- Added runtime cache-tier creation/removal so expert residency can expand or shrink without reloading authoritative host weights.
- Added staging-ring release and per-backend staging enable/disable hooks, allowing ring capacity to be allocated when needed and freed for other work.
- Added CUDA/HIP idle-backend trimming of pools, graph executables, G128 scratch and cached Q8 activations.
- Added context hooks to release compute buffers, release/recreate memory modules and report device usage.
- Added the experimental `delphi-vramd` daemon and a server lease client (`LLAMA_VRAMD`) for shared GPUs.
- Registered persistent weights, scratch and KV usage per GPU and tracked revocable cache leases separately.
- Added home-GPU-first requests, opportunistic remote-GPU leases and cache-tier revocation at batch boundaries.
- Added prompt exclusivity between participating servers so simultaneous prefill does not divide the shared staging bandwidth.
- Added bounded lease sizing, per-server overhead accounting and explicit denial when a request cannot be satisfied.
- Added disconnect/crash lease cleanup and idle shrinking of remote cache leases.
- Preferred empty cache tiers when lending staging space, reducing unnecessary hot-expert eviction.
- Added idle compute-buffer release, saving idle server slots to host prompt cache before freeing KV, and re-reservation on the next request.
- Added server-queue wakeups, idle ticks and hooks to enable staging only on GPUs with an adequate leased arena.

## Sparse attention and attention-mask work

- Used the fused lightning-indexer operation on DeepSeek V4.1 instead of its unfused score graph.
- Added physical compaction of DeepSeek V4.1 selected compressed K/V rows for single-query decode, with the raw sliding window kept in cache order.
- Extended DeepSeek V4.1 compaction to speculative verification by building a separate selected K/V set per query; this graph path also helps CPU attention.
- Added opt-in indexed DeepSeek V4.1 prefill, passing per-query KV selections to FlashAttention rather than running one attention call per query.
- Added attention-sink handling to the indexed FlashAttention kernel.
- Enabled RDNA4 WMMA FlashAttention for wider heads up to 576, with required shared-memory barriers and head-size tuning.
- Enabled mask-compacted sparse FlashAttention on RDNA4 for 512-wide heads, including HIP ballot compaction and a query/head layout suited to WMMA.
- Enabled RDNA4 sparse WMMA attention for Qwen's 256-wide heads and GQA ratio 12 at supported decode/verification widths.
- Added direct-index RDNA4 Qwen sparse decode, avoiding conversion of the selected-row mask back into an index list.
- Passed Qwen's indexer top-k selections directly from the model graph into FlashAttention by default when device, cache and shape guards permit it (`LLAMA_QWEN4EXP_SPARSE_ATTN=2`).
- Added accurate Qwen selected-width hints and support for a full cache-width causal/visibility mask with indexed attention.
- Retained selected-row mask and dense-kernel fallbacks for unsupported Qwen devices, formats, shapes or disabled FlashAttention.
- Added `GGML_OP_INTERVAL_MASK` with CPU and CUDA/HIP implementations, building full attention masks from compact interval inputs.
- Built GLM-5.3 sparse-attention masks on the device instead of uploading the complete dense mask.
- Added fused GLM pooled-index expansion, indexed attention support and tiled pooled-indexer scoring as part of the GLM port.
- Added a lower KV tile bound to MMA FlashAttention so sliding-window layers skip fully masked older KV tiles.
- Qwen QSA gained a pooled-key cache (a row holds the raw and pooled key of its block, so a ubatch pools only touched blocks) and O(1) incremental block grouping; block-level top-k fixes the two-R9700 crash above 60K and the CMP prefill cliff. CMP decode at 100K context 970 -> 18 us/token; 2 R9700s at 0/50K/100K tokens 41.1/34.3/30.1 -> 44.4/40.7/38.5 t/s; outputs bitwise identical (KLD 0.0106 against mainline on CUDA).
- Sparse FlashAttention tiles now cover 256-wide Qwen heads (port of upstream #28770), with one compacted selection list per (sequence, query tile), union lists for tile queries and tiles past the list end skipped; mask compaction is templated per queries per tile (decode at 100K 170 -> 65 us). RDNA4 gets a 256-wide sparse prefill tile (Qwen pp2048 at 100K 705 -> 1082 t/s), and the CUDA/HIP radix top-k runs multi-block for up to 16 rows of 12288+ columns (one 25600-row call 54 -> ~13 us).
- GLM k-pool host inputs gained a tail fast path and generation stamps: GLM 5.3 on 3 CMP170HXs decode at 100K context 28.8 -> 33.9 t/s.
- DeepSeek V4.1 compressed-stream masks are replaced with per-row visible counts in the prefill graph (CPU and CUDA/HIP; 54 new test-backend-ops cases), with indexed attention selected from 131072 context rows.
- GLM's indexer pool bias is built on device from small inputs (`LLAMA_GLM5_DEVICE_MASKS=0` restores the host form).
- The RDNA4 lightning indexer uses WMMA for F16 keys instead of the vector kernel: DeepSeek V4.1 Flash on 2 R9700s, decode at 100K context 31.1 -> 35.5 t/s and 40K prefill +13%; decode KLD 0.0016 against the vector kernel, top-1 100%; `GGML_HIP_LID_WMMA=0` selects the vector kernel.

## CUDA/HIP kernels and small fusions

- Replaced host grouping in the sorted routed-MoE fallback with a stable GPU counting sort, retaining host/device A/B controls and verification.
- Added a graph-capturable grouped MoE schedule with histogram/scan/scatter, expert work tiles and a host reference builder.
- Added NVIDIA grouped MMQ execution with tile widths selected from routed-row distributions.
- Added lossless in-place packed MoE weights and SM80+ `cp.async`/integer tensor-core grouped executors across supported standard and imported quant types.
- Added packed-aware decode MMVQ and fused routed-expert down/weighted-reduction kernels, with a safe fallback for unsupported packed types.
- Enabled the packed in-place MoE path by default on NVIDIA Ampere and newer.
- Added MXFP4 support to fused single-token MoE down reduction.
- Allowed MMVQ gate/up/GLU fusion when the allocator reuses the original activation memory, since MMVQ has already quantized that input.
- Fused `SCALE -> SIGMOID/SILU [-> SCALE]` into one elementwise kernel.
- Fused `RMS_NORM -> SCALE`, preserving the separate operations' rounding order.
- Fused Qwen's scaled-sigmoid hyper-connection injection weights into the post operation.
- Fused GLM/DeepSeek affine hyper-connection weight chains into their pre/post consumers, including a standalone fused chain when the consumer is separated.
- Added the fused `DSV4_HC_MIX` boundary operation, including GLM-5.3 layer norms, and enabled eligible use by default (`LLAMA_FUSED_HC_MIX`).
- Folded a GLM sublayer's `hc_post` into the next eligible HC mix boundary on the same device.
- Used adjacent GLM KDA q/k/v weights, convolution weights and f_a/g_a/beta weights as combined products/views when their storage permits it.
- Combined adjacent GLM dense/shared-FFN gate and up weights into one product for multi-token batches, retaining LoRA/nonadjacent fallbacks.
- Removed unnecessary copies of contiguous GLM KDA tensors.
- Reused a quantized Q8_1 activation across eligible MMVQ operations within a graph.
- Widened MMVF eligibility for small F32/BF16 products.
- Added an Ampere F32 MMVF path up to eight columns for matrices with few output rows.
- Added short-row concat execution with one thread per element for noncontiguous inputs.
- Added aligned 32-bit Q8_0 weight loads for single-column MMVQ.
- Added four-row MMVQ blocks on sufficiently large NVIDIA matrices to reuse activation loads while retaining per-row reductions.
- Moved IQ3_S grid lookups into shared memory for packed MMVQ.
- Increased block residency for multi-token packed IQ3_S/IQ4_XS MoE kernels.
- Added an int8 tensor-core Q8_0/F32 product for eligible speculative-width NVIDIA matrices, reading weights once across columns.
- Added split-K MMVQ for long matrices with few output rows and deterministic partial-result reduction.
- Added a specialized short-K Q8_0 kernel for K=64..256.
- Tuned GA100/CMP K-quant MMVQ/MMQ crossover for verification widths, then made crossover selection depend on matrix row count on GA100 and RDNA4.
- Padded RDNA4 MMVQ grids around the measured 2048-wave performance resonance.
- Enabled the RDNA4 small-K rows-per-block path and changed short-K MMVQ to one warp per row.
- Capped large RDNA4 MMVF grids at 96 threads and added aligned 16-byte F32 loads.
- Kept RDNA4 Q4_K/Q5_0 routed matvecs on MMVQ through eight tokens where appropriate.
- Folded Q2_K sub-block scales/min corrections into RDNA integer tensor-core tiles to reduce per-product floating-point work.
- Moved Q6_K MMQ sub-block scale work to integer accumulation on AMD.
- Fused sibling MUL_MATs that share src1 (back-to-back weight products) into one MMVQ launch with a per-row destination split, and used 4 warps per row on RDNA4 for few rows with long K; Qwen 3.8 Flash verify round 41.0 -> 38.8 ms (`GGML_CUDA_FUSE_SIBLING_MM=0` / `GGML_CUDA_MMVQ_WIDE_K=0` disable).
- Added partial `MUL_MAT_ID` execution on MMVQ for small batches, so narrow routed batches no longer fall back to the host-grouped path.
- Added a 384-expert instantiation of the fused routed router for DeepSeek V4.1's 384 experts, and reworked the fused norm so weight lane k stays in lane k through the divide, matching the unfused chain's rounding; greedy DeepSeek text is byte-identical to the unfused chain.
- Removed the single-token exclusion from the standalone MoE weighted-sum fusion and added an exact kernel form at n_tokens == 1 that reproduces the chain's rounding bit-for-bit; DeepSeek decode dropped 39 kernels/token and paired TG gained 1.5-2.6%.
- Used radix top-k across all supported row counts and CCCL versions, with deterministic ties and live router logits retained through fusion.
- Split the HIP radix top-k gather across blocks, including full-vocabulary/many-row correctness cases.
- IQK packed MoE kernels of IQ2_KS, IQ3_KS, IQ4_KS, IQ5_K and IQ5_KS decode value tables with prmt register lookups instead of scalar global loads; the shifted second table is an add, with a biased table for the -1 entry. CMP170HX `MUL_MAT_ID` 2048x4096x64, top-8, 1/8/1024 tokens: IQ2_KS 99.6/797/4663 -> 42.3/312/3287 us, IQ3_KS 135/1331/5201 -> 65.4/536/3486, IQ5_K 123/829/5508 -> 75.1/570/4877, IQ5_KS 160/1434/4678 -> 100/759/3835; IQ4_KS 47.2/334/3316 -> 44.9/312/3229.
- IQ4_K decodes with IQ4_NL byte permutes plus a bias add and uses the IQ4_KS word layout (MUL_MAT_ID 1 token 98.8 -> 49.1 us; dense 4096x12288 1 token 85.1 -> 42.8 us).
- Dense MMVQ and MMQ tile loaders of IQ4_KS, IQ5_K and IQ5_KS use the same register lookups (IQ4_KS 38.4/110/919 -> 34.1/93.9/919 us, IQ5_K 84.0/185/1522 -> 56.0/168/1491, IQ5_KS 78.4/181/1059 -> 55.6/128/1016); an IQ6_K attempt was reverted because it was 13% slower at 8 tokens.
- Packed IQ4_K is removed from the fused MoE down reduction, which accepts no IQK type; when the gate accepted it the reduction was 5% slower for IQ4_K.
- RDNA4 launches are padded out of the 64 Ki-thread stall (~20 us instead of 2-5 us) by adding idle blocks along one grid dimension for rope, get_rows, concat, quantize_q8_1, the stream-k fixup and combine_results; 512-wide heads use 4 x 4 query/head tiles at 3-4 queries. DeepSeek V4.1 DSpark verification saves 2.8 ms of GPU kernel time and 2.3 ms per ~77 ms verify.
- The radix top-k scratch is sized for the most blocks per row, not the current stream width, and contexts recapture their graphs after a pool flush.
- Packed MoE prefill uses 32 x J warp tiles for the lossless formats that decode their A fragments in every warp that reads them (IQ4_XS/KS/K, MXFP4 and others), avoiding the double decode of 64 x J/2 at J = 32; Q4_K and the other scale+min formats keep 64 x J/2 at J = 64. The IQ*_KS row-scale plane loads through `cp.async` into the tail of every stage instead of a per-unit cache with an extra barrier per k iteration. CMP170HX, 64 experts, n = 228: IQ4_XS -11%, IQ4_KS -13%, IQ4_K -17%, MXFP4 -15%, Q4_K -4%. GLM-5.3 Flash 3-CMP prefill +4-5%, greedy output bitwise identical. `GGML_CUDA_MOE_PACKED_VARIANT=2` restores the old tiling, `4` forces 32 x J.
- IQ*_KS MMQ tile loaders divide the row pointer once per thread instead of once per row: R9700, 64 experts, n = 228 IQ2_KS -36%, IQ4_KS -12%, IQ3_KS/IQ5_KS -2-3%; dense n = 1024 -1 to -4%.
- Wide dense Q8_0 products run through 64 MiB chunked F16 dequantization and one cuBLAS GEMM per chunk into its destination rows, keeping extra VRAM to the chunk buffer plus the F16 copy of src1. On by default from 128 columns on cc 8.0 only; `GGML_CUDA_DENSE_CHUNKED_MIN_TOKENS` (-1 off), `GGML_CUDA_DENSE_CHUNKED_MIB` and `GGML_CUDA_DENSE_CHUNKED_TYPE` (f16, bf16, f32 reference) control it. GLM-5.3 Flash 3 CMP: prefill +8-9% (10K 1649 -> 1787 t/s), 62-138 MiB more VRAM per GPU; KLD 0.0101 versus 0.0108 for MMQ (run-to-run floor), DFlash2 decode unchanged.
- `GGML_CUDA_MMQ_MAX_DENSE_TOKENS` is an A/B knob that sends dense quantized products of more than n columns through dequantize plus cuBLAS instead of MMQ; unset, nothing changes.
- The default build compiles only the packed MoE kernels that the default dispatch launches: 268 instead of about 2000, 2.8 minutes instead of about 26 on sm_80. Selected kernels, configurations and prefill speed are unchanged; `-DGGML_CUDA_MOE_PACKED_TUNING=ON` restores the comparison variants, and without it the tiling and i8-magic environment variables are ignored with a warning.
- HIP keeps linear and pitched copy graph executables separate, including anchored graph entries, because a one-row CONT uses a linear memcpy and a multi-row prompt a pitched memcpy, and older HIP runtimes can accept an update between those native node classes without updating the copy.
- Narrow CUDA GET_ROWS rows pack into thread blocks: 1-32 lanes per row, and each 256-thread block gathers several indices when rows are small. Batch strides, type conversion and RDNA4 launch padding are preserved, and the one-block-per-index overhead in backend top-p sampling is gone; `GGML_CUDA_GET_ROWS_SMALL=0` selects the original dispatch.

## Complete INT4 routed-FFN execution

- Added a CUDA whole-region routed FFN executor spanning gate/up, clipped SwiGLU, down projection, router weighting and slot reduction, adapted from the SM80 Marlin-style donor.
- Added Q4_G128 in-place Marlin-layout packing with per-buffer ownership, lifecycle invalidation and packed-reader guards.
- Added ordinary-path fallbacks for unsupported CUDA regions and shapes, including safe temporary conversion of packed weights when needed.
- Added CUDA INT4 prefill expert routing into tiles, pipelined W13/W2 kernels and SwiGLU in the W13 epilogue.
- Extended CUDA INT4 execution to arbitrary row counts using decode chunks below the prefill threshold and bounded prefill passes above it.
- Added compact RDNA4 G128 routed-FFN execution using W4A8 vector/WMMA arithmetic and fused slot reduction.
- Kept G128 bank views outside the packed-layout ownership scope so views cannot be mistaken for independently packed root banks.
- Tuned wide loads and WMMA dispatch for Qwen's narrow expert geometry.
- Used vector waves for singleton Qwen expert groups and WMMA for reused groups within the same launch.
- Added grouped RDNA4 G128 prefill routing across token batches, reusing the GPU MoE schedule and compact packed banks.
- Added register-tiled RDNA4 G128 batch kernels that reuse weight/activation fragments, fuse intermediate quantization into the gate/up epilogue and avoid the F32 hidden buffer.
- Added packed-scale layouts and adjacent scale loads for those RDNA4 kernels.
- Bounded/cached graph-safe INT4 scratch, corrected documented prefill scratch requirements, separated batch/chunk/decode counters and added refusal diagnostics.
- Enabled eligible RDNA4 G128 fusion by default, while keeping shape/layout guards and explicit per-node/chunk comparators.
- Enabled the CUDA INT4 MoE FFN kernels by default for Q4_G128 experts (BF16 tensor-core MMA); GLM-5.3 Flash 79.9 -> 90.9 t/s without an env knob.
- Kept the fused INT4 FFN region available also under `GGML_CUDA_DISABLE_FUSION`, which had disabled it and left free memory near zero during profiling (CUDA2 free 10 -> 2366 MiB; 65K prefill 559 -> 739 t/s).
- Made the packed-expert prepare step skip partial `MUL_MAT_ID` nodes instead of aborting the run.
- Extended fused HIP G128 execution to partial batches of any size, including zeroed output for excluded expert slots.

## Speculative decoding, draft ingest and numerical controls

- Corrected recurrent-state rollback for every speculative draft type rather than only ordinary model drafts.
- Added a separate draft-context ubatch option (`-ubd`) and a default of 128.
- Added asynchronous per-token feature handoff through two pinned output sets and per-backend completion events.
- Added a helper reporting the share of routed-expert bytes residing on CPU, used when selecting an ingest policy.
- Overlapped intermediate prompt ingest into MTP/EAGLE3/DFlash/DSpark with the next target prompt batch on a worker thread.
- Added safe waiting/ordering for draft memory operations, checkpoint restore and speculative entry points around asynchronous ingest.
- Disabled asynchronous ingest by default for heavily CPU-offloaded targets where streamed prefill and draft GPU work interfere; retained `LLAMA_SPEC_ASYNC_INGEST` overrides.
- Added state-only GLM MTP prompt ingest: write persistent indexer/pooled/latent-KV state and omit unused queries, attention, MoE and head work.
- Added Qwen MTP state-only prompt ingest: store the required K/V without unused query/attention/FFN/head computation.
- Kept an optional copy of GLM token embeddings on the MTP head's device to remove its CPU lookup split.
- Added an opt-in adaptive draft-depth controller for draft-simple, EAGLE3, DFlash, DSpark and MTP, including per-generation reset and partial-acceptance/checkpoint accounting.
- Made per-round confidence draft depth the default for draft-mtp (conf mode): each round stops at the first draft token whose calibrated acceptance no longer pays for its verify row, up to 7 tokens; `--no-spec-draft-adaptive` restores fixed depth; a cost mode exists but loses to fixed depth on GLM. Conf vs fixed-3: GLM +2.5%/+1.6% (2048/512-token prompts), Qwen +4.0%/-2.1%.
- Added `POST /speculative` run-time reconfiguration of speculative options and `GET /speculative` reporting of active settings, graph-build counts and the conf calibration table.
- Ported upstream probabilistic draft sampling (#27694) for draft-simple/MTP and added DFlash2 probabilistic drafts (#30663): `--spec-draft-sampling greedy|probabilistic` (default greedy) with an exact output distribution and run-time switching; at T=1.0, fixed-3 greedy 70.6 -> 73.7 t/s.
- Reused the server sampler between chat requests with equal parameters, so graphs with backend sampling stay reusable, and reserved backend-sampler buffers again only when sampler chains change; DeepSeek first-token 2174/2820/4210 -> 680/1295/2654 ms, Qwen 653/714/828 -> 175/207/361 ms, identical outputs.
- Added `llama_kv_cache_n_pad_reserve`: the reserved KV view grows in bounded steps (2048 with per-round draft depth, applied only to ubatches up to 64 tokens) instead of padding the whole context.
- Enabled the CUDA/HIP FlashAttention decode KV_max masked-tail scan by default (`GGML_CUDA_FA_KV_MAX_DECODE`), also for few query rows, removing dense-attention padding cost in per-round-depth decode.
- Added `DELPHI_SPEC_PLOG` logging of MTP draft probabilities and the accepted count per round.
- Added `GGML_SPEC_INVARIANT` for eligible CPU and CUDA/HIP widths up to eight tokens, favoring single-token arithmetic across ordinary matmuls, routed products and FlashAttention.
- Kept invariant-mode float matvecs on MMVF and disabled width-dependent weighted-reduction/attention paths that change the reduction order.
- Added `llama-spec-inv` to compare logits and committed KV/recurrent state across verification widths, with first-differing-tensor tracing and target-NextN/reference controls.
- Added an opt-in speculative host-span ledger (`DELPHI_SPEC_LEDGER`) for target/draft synchronization, feature movement, lattice work, rollback and acceptance witnesses.
- Added repeated-request and ledger-analysis tools, including loading prompts from JSON.
- DSpark drafts the trained block by default: `--spec-draft-block N` (0 = the trained block size read from the draft GGUF) keeps the block's non-causal attention intact while verification only processes the first n_max proposals; DeepSeek V4.1 Flash on 1.5 R9700s at n_max 3: acceptance 0.533-0.553 -> 0.576-0.590, tokens per verify 2.61-2.67 -> 2.74-2.78, 25.47 -> 26.45 t/s. n_max 2, 4 and 5 are slower with either block length.
- Added an opt-in DSpark block-prefix controller (`--spec-draft-adaptive-mode block`): the complete trained block is still drafted, and per-slot calibrated confidence plus live complete-round cycle costs choose how much of the prefix to verify, including an empty prefix (the mode default floor is 0; `--spec-draft-n-min-adaptive 1` disables empty prefixes for A/B). Startup exploration is bounded, graph-build rounds are excluded from steady width costs but included in the measured output rate, and the expected-output-minus-live-rate comparison uses the aggregate output rate because per-round throughput ratios favor another policy when confidence varies. With the recommended floor 1, a four-task mixed workload gained +4.0% aggregate decode.
- Parallel DSpark block controllers keep acceptance calibration and request state per sequence; concurrent prefix decisions are priced against total verification width and total expected emitted tokens, with one complete-round cost history per active-slot count and no attribution of peer work to the single-slot history. Incomplete cohorts, prefill, request transitions and graph builds are excluded from timing samples. The one-slot restriction was removed without a separate parallel throughput claim; the retained controller is the accepted single-slot policy.
- Added exact sparse DSpark block proposals: the server rejection sampler is reused with top-k proposal probabilities and a separate per-sequence random stream; sampled tokens do not condition later rows, and confidence-based prefix selection is independent of the draws. Greedy remains the default; at temperature 1, adaptive probabilistic gained about +4% against greedy in two comparisons.
- Draft features download on a side stream: a target feature read of at least 1 MiB into a pinned hand-off buffer copies into a device slot on the compute stream, then downloads the slot on a side stream that is joined before the hand-off events are recorded (`GGML_CUDA_SIDE_DOWNLOAD=0` off). GLM-5.3 Flash with DFlash2 on 3 CMP170HXs: prefill +1-2% at 2K, +3-5% at 10K, +4-6% at 40K; decode and acceptance unchanged.
- DFlash reserves both contexts before it enables target-layer extraction and the unmasked NextN output; previously the first decode reserved larger buffers (+112 MiB on CUDA0/CUDA2) and `-c 786432` failed every request on 3 CMP170HXs. If the reservation fails, extraction is turned off and the server continues without speculative decoding. A side-download allocation failure now downloads on the compute stream instead of aborting.
- The server ends a speculative prompt with a call of at most one ubatch so the draft ingests the rest synchronously after the target finishes that call; `LLAMA_SPEC_TAIL_SPLIT=0` restores the old behavior. GLM-5.3 Flash with DFlash2 at 2K: about +1%.
- A V4.1-format DSpark draft without `hc_head` weights takes the V4.1 semantics with a warning instead of failing.

## Automatic parameter tuner (auto-tune)

- Added the first integrated auto-tuner (`tools/auto-tune`): `tune.py` takes a plain `llama-server` command and a YAML prompt bank (default: Haiku, 48 prompts, 8 domains, 18 checked answers). Phase 1 tunes context shaping (-ub, -b, -ubd, -t, -tb, expert-cache margin) through `POST /tune/context` at 256 and 2048 tokens, scored 80% decode without speculation and 20% prefill. Phase 2 decides the drafter once (draft model, else MTP from `nextn_predict_layers`, else none), tunes its n-max/p-min/adaptive settings and the n-gram-simple/map-k/mod modes, then confirms the top 3 plus the start on the full bank with fresh launches, greedy and 512 output tokens, and reports a verdict against the start spread. The tuner relaunches a dead server.
- The v1 tuner removed the development M0-M3 replay predictor, observation capture, KLD gate, placement search and recipe arms; `test-auto-tune.py` covers the v1 tuner, and the observation test of the replaced tuner was removed.
- The M0-M3 development history entered through the feature merge: device-memory budgets, CPU affinity and placement policy, split refinement, retained context search, MTP acceptance/pass-cost policies, trace/observation integrity checks and validation evidence.
- Qwen 3.8 G128 validation: the hand-tuned start was within the tuner's noise; the confirm run showed +0.3% score against a 3.1% spread, so no reliable gain. A run with both phases takes about 70 minutes.

## KV, context and server state management

- Added a KV checkpoint policy: variance-based eviction of interior checkpoints (ik_llama PR 2020 rule), first-checkpoint pinning on by default, checkpoints at ends of chat messages (`--ctx-checkpoints-turn-step`), and dropped-checkpoint spill to checksummed DCKP files on disk (`--ctx-checkpoints-spill-dir`).
- Prompts with media skip the checkpoint spill and use text tokens at turn boundaries.
- Added the context-recreation hooks the tuner needs: create a new `llama_context` on the loaded model and read the cache-worker/refill env knobs per context.
- The server ends a prompt batch 4 tokens before the end for a checkpoint; the one-ubatch-earlier break (which cost DeepSeek V4.1 CED a decoder pass per prompt) is now opt-in (`LLAMA_SERVER_CKPT_TAIL_UBATCH=1`); it is off by default.
- kpool keeps the pools of sequences that a ubatch does not name, so shared-stream slots stop rebuilding GLM kpool state between their own ubatches (98 new pools for 4 update rows before the fix); a pool-key update larger than the incremental capacity now degrades gracefully (fits what it can, logs, and the next graph reserves rows for the rest).
- fit-params includes the GLM pool-key cache in the context memory breakdown, counting the size its tensors will take when no_alloc leaves a zero-size dummy buffer.
- Added shared prompt-prefix snapshots, separate from `--cache-ram`: the server finds the longest common prefix between a new prompt and a prompt it holds (slots, host cache, stored prefixes), where the two then differ before the first assistant message, stops the batch there and reads the whole target/draft/speculative sequence state with `llama_state_seq_*_ext`. One conversation alone never makes a snapshot, and divergences inside a conversation stay with the host cache. A restore is used only when the measured read and restore speeds say it is faster than the prefill it saves.
- Snapshot storage: `--cache-prefix-mib` (default min(4096 MiB, RAM/16), -1 no limit, 0 off), `--cache-prefix-min-tokens` (1024), eviction of entries without a hit first. `--cache-prefix-dir` adds a writer thread and versioned, checksummed files keyed by the model, context setup and sequence count, with `--cache-prefix-disk-mib` (16384) of LRU by file time. Files survive restarts; other model keys are left alone, so several llama-swap entries can share a directory; a bad file is removed. The sequence count is part of the key because the state records the number of KV streams (`--parallel 1` and `--parallel 3` files do not mix).
- Prefix snapshots and checkpoint spill work with an mmproj and with media prompts: the snapshot and restored prefix end before the first media chunk, where the text tokens still have their prompt indices, so a shared system prompt followed by images is covered. A prefix that holds an image is not stored because the media ids are not in the key. Checkpoint spill keys on the text tokens and is skipped for checkpoints after the first media chunk.
- A request that saved its prompt to `--cache-ram` and found no better entry keeps its checkpoint files now; previously `prompt_load()` returned true and the files were dropped, so editing an early turn in a long conversation found no file. Gemma 4 26B with an mmproj: the edit restores a 7588-token checkpoint from disk (200 MiB, 143 ms) instead of processing 5066 tokens.
- Memories that report a single position (`pos_min == pos_max`: DeepSeek V4 compressed KV and hybrid SWA plus recurrent) now use their checkpoints without the SWA window check, like recurrent memories. DeepSeek V4.1, 15K-token system prompt, new conversation in the slot: 10 tokens processed in 2.0 s instead of 15022 in 19 s.
- Validation: restore versus in-slot reuse is bitwise on plain, hybrid, SWA and recurrent models, and a disk restore after restart is bitwise to the RAM restore. GLM-5.3 Flash 15K system prompt: snapshot 428 MiB, D2H 540 ms, restore 409 ms, new-session TTFT 4.4 s vs about 16 s; with the deployment flags (`-np 3 -c 1572864 LLAMA_KV_RESIDENT_CELLS=auto`), a restart restores from file in 773 ms and TTFT is 3.7 s vs 13.9 s. DeepSeek V4.1 with DSpark block, `-np 2 -c 1048576`, mmproj and CPU experts: snapshot 79.6 MiB, restore 91-94 ms from file, TTFT 4.0 s vs 20.3 s. Restore H2D runs at about 22 GB/s on an R9700.
- Prompt checkpoints copy asynchronously by default (`LLAMA_CKPT_ASYNC=0` restores synchronous): target state is copied on the sequence's owning stream set, draft state on the speculative worker, and completion events are recorded before pinned snapshots are published to host storage. Output-only synchronization does not publish unfinished background downloads, and NEXT_RESPONSE, metrics and slot polling do not drain checkpoints. Replacing a checkpoint keeps up to four discarded destinations alive until the existing synchronization completes when verify streams are active without checkpoint spilling; a checkpoint is taken only when the next prompt batch is admitted, so a generating-only round does not supersede a live destination.
- `seq_rm` now limits its scan to the used cell range instead of every cell of the stream. With `-c 1572864 --parallel 3` that was 524288 cells per slot, and speculative decoding calls it after each verify for the target attention and indexer caches and the draft context; unused cells have no position, so the result is unchanged. GLM-5.3 Flash, 3 CMPs, DFlash2 probabilistic short chat: 45.0 -> 43.4 ms/verify (top-p 0.95) and 43.4 -> 42.0 (greedy); seeded outputs are byte-identical.
- DeepSeek V4.1 reused compressed KV is mirrored per reader device (see DeepSeek V4.1 long context).

## Graph reuse, scheduling, transfers and loading

- Kept bounded GPU graph-cache entries alive until capacity eviction rather than expiring them solely by elapsed time.
- Stabilized shape/topology graph-cache reuse and skipped redundant rehashing of known graphs.
- Captured large MoE prefill graphs on first eligible use.
- Increased default captured-graph capacity to 2048 per device.
- Kept multiple small-ubatch graph/scheduler slots with LRU replacement to serve varying speculative widths.
- Refreshed kpool memory-context pointers when reusing graphs instead of rebuilding based on transient object addresses.
- Preserved tensor-split graphs across prompts, rebuilds and KV growth, including per-device state for several graphs.
- Added meta-backend events and pipelined tensor-split input copies, including asynchronous copies to an idle meta backend.
- Ran tensor-split AllReduce inside device graphs and added a small-tensor HIP P2P push AllReduce.
- Allowed tensor split for Qwen's `qwen4exp` architecture.
- Preserved pipeline parallelism when tensor overrides affect only embedding tables.
- Added maximum-prefill graph reservation ahead of growing prompts, generalized beyond GLM, with correct attention variant and output-count handling.
- Skipped repeated maximum-prefill reservation while the reservation remains valid.
- Added one scheduler/graph slot per speculative verify width, so per-round draft depth does not thrash graph caches.
- Avoided executable updates after a fresh graph instantiation and re-instantiated large graphs on HIP.
- Added `LLAMA_GRAPH_BUILD_LOG` entries for each graph build with its host time and the reason reuse failed.
- Ran scheduler copies between host memory and a GPU on the GPU stream.
- Disabled asynchronous host-to-GPU input copies when pipeline parallelism would queue them behind the previous graph, which had stalled host-read fences and cost 40-58% on multi-GPU prefill.
- Added asynchronous sequence-state export (`LLAMA_STATE_SEQ_FLAGS_ASYNC`).
- Created server prompt checkpoints without draining the prefill pipeline, ordering draft checkpoint capture behind asynchronous ingest.
- Reset only scheduler-copy slots actually written and used zero-initialized copy tables to avoid a full-table clear.
- Copied full-size cross-backend views through one copied source tensor rather than duplicating the source and view transfer.
- Excluded in-place-written tensors from that aliasing optimization.
- Kept a view of a computed tensor on the backend of its source rather than choosing a backend from the apparent host buffer.
- Limited previous-split waits to backends that can actually share memory.
- Consolidated copy-slot reuse waits and recorded the reuse proof before graph events are overwritten.
- Added per-split reuse events before overwriting inputs, avoiding a full destination synchronization for each copied input.
- Ordered split input copies by their producers' graph order so early-ready transfers do not queue behind later work.
- Added pinned-host staging for cross-device CUDA copies when peer access is unavailable.
- Uploaded mapped weights concurrently on one loading thread per device, with completion before unmapping/cancellation.
- A fused node without its own weights stays on a device that supports it; the earlier misread of a previous-layer device as missing support had disabled DeepSeek's fused HC pre on every multi-GPU split.
- Prompt-sized single-ubatch calls take the max-prefill reservation, so a smaller final call (checkpoint boundary) no longer records graph sizes that the next full call outgrows and reallocates; GLM 4K prompt 639-669 -> 861-938 t/s at -ub 2048.
- Large MoE prefill splits keep lending their executables and take over an oldest sibling's executable with cudaGraphExecUpdate, instead of marking themselves no-lend after the first repeated shape and re-instantiating ~77 ms per 2000-node split (16K prompt 1427 -> 1492 t/s, 8K 1312 -> 1358).
- A graph whose nodes may call cuBLAS runs eagerly while the BLAS handle is missing, instead of creating the handle inside capture (hipBLASLt `exit(1)` on the first eager shortcut with n-gram drafts of 32 tokens or longer).
- `GGML_GALLOC_REALLOC_LOG=1` names the first node or count that forces a graph allocation growth.
- The radix top-k scratch no longer grows per ubatch, and captured contexts recapture their graphs after a HIP legacy pool flush.
- `LLAMA_PP_ALLOW_OVERRIDES=1` keeps pipeline parallelism with non-embedding tensor overrides, such as layers' experts held on the CPU (3 CMP170HXs, 4096-token prefill 576 -> 619 t/s).
- Only batches accepted by the backend's `offload_op` are offloaded, and a failed dot calibration is cached rather than retried.
- A staging ring that could not get all its slots keeps going: keep mode counts only later inputs when deciding whether a staged weight is still read, and uploads into the slot whose weight is read last when every free slot is busy; the redirect ring has at least the most staged inputs of one split; and a ring that got fewer slots retries only after its slots are dropped. DeepSeek V4.1, `-np 4 -c 4194304` (1 of 6 ring slots, cache 252 + 170): the first 16K prompt aborted before, now 16K prompts run at 424 t/s, four parallel requests complete, and outputs match; `--parallel 1` 16K runs at 1112-1124 t/s (main 1101-1113).
- Keep mode stages on the GPU that holds the activations, with 6 slots for the current and next layer, instead of spreading layers across staging GPUs; `GGML_SCHED_STAGE_KEEP_SPREAD=1` restores spreading with 3 slots per ring. On the 2-R9700 DeepSeek V4.1 deployment both give the same 16K prefill time (26.1-27.1 s vs 26.6 s), and the ROCm1 tier keeps its slots.
- Added a staging free-slot wait that HIP polls under `ggml_cuda_lock` only when no capture is active, so a query never overlaps a capture on runtimes before ROCm 7.2.1 (which invalidated the capture and aborted long staged prefills); `GGML_SCHED_STAGE_WAIT_FREE=0` restores the plain host wait. A relay through a non-capturing stream also fixed the crash but made each free signal about 1 ms later and prefill 6-8% slower.
- Added internal retention of immutable staged weights across a graph group, with upload readiness reuse and invalidation before weight changes; inactive by default.
- `ggml_backend_sched_alloc_splits` returns an error when `ggml_gallocr_reserve_n` fails instead of allocating the graph in the missing buffer; an out-of-memory during reallocation had crashed the server (GLM-5.3 Flash at `-c 1048576`, first decode after a 1M prompt).
- Custom HC matchers reject fusion candidates that extend past a partial graph before dereferencing their nodes; the old HIP plugin crashed on a one-node SCALE tail.
- HIP executable accounting: probe the runtime cost (about 2 MiB per captured executable), bound retained executables by the context's layer and alternate-scheduler count, include copy-only graphs and the draft context in the budget, keep the full kernel-capacity reserve, and synchronize before eviction because HIP does not document deferred destruction. The graph cache could otherwise grow by more than 1 GiB after load and exhaust the auto expert-cache margin.
- Auto expert-cache devices keep their graph limits when tier creation cannot use their memory, so a draft backend on another GPU retains its policy.
- DSpark reserves the feature-extraction and draft-sampling layouts when it is initialized; with warmup enabled, each active-sequence count is exercised through target verification, feature injection and draft blocks before auto expert-cache sizing, creating the batched BLAS handle and compute buffers while VRAM is still available. KV state and timing counters are cleared before serving.
- Added `ggml_backend_set_stream_priority`, which creates the main stream of a CUDA backend with the highest (or lowest) priority, and `ggml_backend_event_query`, which reports whether an event is done without a wait.
- `GGML_GALLOC_GROW_PCT=N` gives a buffer that grows N% more room than the graph needs, bounded by `GGML_GALLOC_GROW_MAX_MIB` (default 16). `GGML_GALLOC_GROW_NODE_PCT=N` gives tensors that outgrow the current layout N% more room (at least `GGML_GALLOC_GROW_NODE_MIN_KIB`, default 64, at most `GGML_GALLOC_GROW_NODE_MAX_MIB`, default 32 each) while keeping the room of the others. Verify streams set these at the first background decode (25% and 400%), so load-time and fit buffers keep their size (about 450 MiB per card less before at the production context).
- `GGML_CUDA_RETIRE_MAX_MIB` bounds the bytes of freed CUDA buffers that are retired instead of freed immediately. `cudaFree`/`cudaFreeHost` wait for all work of the device, so with verify streams a buffer freed next to a high-priority stream could make the server thread wait 1.0-1.4 s for background prompt work; retired buffers are freed when a backend without a high-priority stream synchronizes or an allocation fails. Verify streams set 512 MiB at the first background decode. NUMA host buffers registered with `cudaHostRegister` retire their pointer and mapping size through the same queue, so growth does not block concurrent prefill in `cudaHostUnregister`.
- Eligible NVIDIA MoE prefill graphs are captured immediately after pointer changes with BLAS initialized before capture, so a warmed graph does not run eagerly because its input copy moved. The server defaults `CUDA_MODULE_LOADING=EAGER` before device discovery when verify streams are requested, and `GGML_CUDA_GRAPH_FALLBACK_LOG` exposes fallback boundaries.
- An absent optional decoder cache now reports `NO_UPDATE` instead of `SUCCESS`, so DeepSeek V4.1 keeps its idle graphs instead of resetting and reserving on every decode, including single-token steps.
- The maximum-prefill reservation and the background-prompt layout check share one predicate, so the server holds or releases a chunk against the same answer.

## Profiling, diagnostics and reproducible comparisons

- Added the binary execution-trace profiler (`GGML_EXEC_TRACE`) with execution, ubatch, graph, node, split, device and copy identity.
- Recorded CPU/CUDA/HIP matmul implementation choices, fusion variants and graph capture/update/replay/fallback reasons.
- Added optional device-event GPU timing around eager operations, submissions and captured graph launches without synchronizing the measured execution.
- Added explicit host/device/vendor timing provenance and production/full detail levels.
- Added structural fingerprints, truncation/unresolved-event handling and a paired-run integrity harness for production-representative tracing.
- Added trace inspection, GPU-activity correlation, ROCprofiler import and Perfetto timeline export tools.
- Added trace mapping from tensor-split/meta-backend device graph nodes back to the original traced graph nodes.
- Recorded offload-model decisions, transfer byte estimates and calibration results in traces.
- Recorded scheduler pipeline-parallelism state in traces.
- Added the recipe-driven bottleneck profiler (`tools/bottlenecks/profile.py`) for llama-bench and llama-server/SPEED-Bench workflows.
- Added build-directory selection, controlled command/environment variations, warmup and normalized results with raw artifacts.
- Fixed recipe imports to share the running profiler module and its class identities, and removed the duplicate alias introduced during integration.
- Recorded exact commands, Git/build/model identity, arm caveats and binary versions, displaying caveats in terminal and report summaries.
- Based bottleneck/Amdahl ceilings on end-to-end workload wall time rather than a partial traced span.
- Added separate long-prefill measurements and an exact-length prefill client, retaining complete trace sequences and their tails.
- Aggregated target/draft contexts correctly and isolated request windows so verification batches are not mislabeled as prefill.
- Added automatic per-operation and recurring-node-role bottleneck discovery from complete GPU and CPU operation traces.
- Reported unaccounted host time, copies/waits by path, per-device utilization, host/device overlap, idle time and recurring work.
- Flagged undersampled/partial trace breakdowns and withheld device-overlap claims when timestamp placement is too uncertain.
- Fixed device-interval placement in traces and profiler reports: GPU timing epochs are attributed to the synchronize that completed them (upper clock bound when the host blocked); the earlier "every device idle ~60%" multi-GPU decode findings were placement artifacts (now 6-15%), and the resource table unions busy intervals across devices.
- Added speculative-cycle reports: verify/draft widths, accepted tokens, per-cycle wall/device/CPU time and cost per output token.
- Added prefill pipeline summaries showing ubatch flow and stage stalls.
- Added GPU clock, memory-clock, utilization, power/limit and active-profile sampling per benchmark arm, including busy-before-run and changed-setting flags.
- Added a resumed long-prompt decode suffix client for separate context-length/decode studies.
- Added routed-row/expert histograms and an Amdahl-style CPU/device wavefront-overlap bound tool.
- Added dispatch summaries grouped by ubatch width and complete single-layer FFN timing/replay tools.
- Added CPU FFN replay and streamed-decode CPU-counter capture helpers. (`tools/cpu-g128/`, `ca36b4e80`)
- Added capture/replay helpers for Qwen G128 HIP backend probes and fixed FFN replay environment declarations.
- Added explicit operation/shape/type backend cases through `test-backend-ops --custom` and `--custom-file`, configurable threads and time_us CSV output.
- Added controlled MoE routing distributions for those cases: uniform, single, Zipf, skewed, concentration, hot-tail and recorded replay.
- Exposed profiler trace capacity for large MoEs so auto-quantize calibration traces do not drop events.
- Added the `worker_ready` trace span so exposed worker-join time is visible in overlap measurements.
- Parsed llama-bench JSON arrays in the bottleneck profiler.
- Added public reference/usage documentation for tracing, report interpretation, A/B environment switches, staging and deployment-independent recipes.
- Documented staging link floors, ubatch/slot/worker choices and the observed MiMo three-slot NaN caveat.
- Documented `ROCBLAS_USE_HIPBLASLT=0` as a repeatability control and its measured cost, rather than treating library-dependent output variation as guaranteed bitwise behavior.
- Recorded bootstrap subsystem provenance/limitations and repository development instructions.
- `GGML_GALLOC_REALLOC_LOG=1` also logs every compute buffer that grows, with the graph that caused it; `GGML_GALLOCR_PEAK_LOG` lines are warnings so servers show them at default verbosity; `GGML_CUDA_POOL_LOG=1` logs each CUDA pool growth with its op and the free device memory at every eighth graph instantiation.
- `GGML_GALLOCR_PEAK_LOG=1` prints, per buffer at reserve time, the live bytes at the peak and the 16 largest live tensors, to identify which tensors set a compute-buffer size.
- `LLAMA_GRAPH_BUILD_LOG` reports memory apply and input times as well as graph-build time.

## v1 command line and defaults

- Added command-line options for the delphi performance settings: `--moe-cache`, `--moe-cache-margin`, `--moe-cache-roles`, `--moe-cache-async`, `--moe-cache-refresh-max`, `--moe-cache-window`, `--[no-]moe-stage`, `--moe-stage-min-tokens`, `--moe-stage-slots`, `--[no-]moe-stage-spread`, `--moe-stage-devices`, `--cpu-expert-prefetch`, `--[no-]moe-pregate`, `--[no-]moe-shared-overlap`, `--engram-readers`, `--[no-]kv-mirror`, `--graph-alt-slots` and the server `--[no-]ckpt-tail-ubatch`. Each option sets the engine environment variable that the code reads (the variable name is also the environment name), so existing environment-based scripts keep working and an option value takes precedence.
- Added the negative environment variant only for `LLAMA_ARG_` names, so the delphi boolean options that use engine variables do not create duplicate names that fail `test-arg-parser`.
- Added server options `--cache-prefix-mib` (default -2 = min(4096 MiB, RAM/16), -1 no limit, 0 off), `--cache-prefix-min-tokens` (1024), `--cache-prefix-dir` (empty = RAM only) and `--cache-prefix-disk-mib` (16384), with `LLAMA_ARG_CACHE_PREFIX_*` environment variants.
- New defaults: staging for long prompts is on (`GGML_SCHED_STAGE=0` disables), spread over staging GPUs with 2 slots per GPU; CPU expert prefetch is 6 and active only under `--numa tensors`; router pre-gating is on but disabled when prefetch is off or NUMA tensor mode is inactive; nothing is allocated without host-routed experts.
- The server's prompt checkpoint one ubatch before the end is off by default (`LLAMA_SERVER_CKPT_TAIL_UBATCH=1` enables it), because it cost a decoder pass per DeepSeek V4.1 prompt and gained nothing on GLM.
- DeepSeek V4.1 layer-bank windows default to on (`LLAMA_DSV41_LAYER_BANK=1`, `2` same schedule without retained banks, `0` off); chunked dense prefill attention defaults to `LLAMA_DSV41_ATTN_ROWS=1024`, and indexed prefill from 128K applies only when it is 0.
- Server `-np auto` on `deepseek4`/`deepseek41` selects 4 slots and disables unified KV (per-stream compute buffers); without `-c` the slots share the training context, and `-c 0` gives each slot the full context.
- The 2026-11-12 delta adds defaults: staggered verify is on where it applies (`LLAMA_STAGGERED_VERIFY=0` disables), prompt checkpoints are asynchronous (`LLAMA_CKPT_ASYNC=0` disables), GLM prompt-tail indexer partitions are stable (`LLAMA_GLM5_STABLE_PREFILL_CHUNKS=0` restores), DeepSeek V4.1 reads only visible raw-window cells (`LLAMA_DSV41_RAW_WINDOW=0` disables), and the RDNA4 lightning indexer uses WMMA for F16 keys (`GGML_HIP_LID_WMMA=0` selects the vector kernel). Verify streams remain opt-in (`LLAMA_VERIFY_STREAMS=1`).
- The 2026-11-13 delta makes verify streams default on where they apply (`LLAMA_VERIFY_STREAMS=0` opts out) and adds `GGML_SCHED_INPUT_STAGING_MIN_KIB`. GLM-5.3 Flash, 3 CMP, 3x512K: single-session prefill and decode equal to off after the first prompt (one-time ~0.9 s on the first prompt after load), three-session decode 1-2 % slower, decode during a concurrent prompt much smoother.
- New environment variables: `LLAMA_KV_RESIDENT_CELLS`, `LLAMA_KV_IDX_RESIDENT_CELLS`, `LLAMA_KV_WINDOW_HOST_MIB`, `LLAMA_KV_WINDOW_MARGIN_MIB`, `LLAMA_KV_WINDOW_STAGE_MIN_TOKENS`, `LLAMA_KV_WINDOW_UNION`, `LLAMA_HYBRID_SEQ_SPLIT_TOKENS`, `LLAMA_CHUNK_MIB`, `LLAMA_DSV41_LAYER_BANK`, `LLAMA_DSV41_ATTN_ROWS`, `LLAMA_SPEC_TAIL_SPLIT`, `LLAMA_ARG_SPEC_DRAFT_ADAPTIVE_MODE` and the CUDA/HIP switches cited above.
- The 2026-11-12 delta adds `LLAMA_STAGGERED_VERIFY`, `LLAMA_VERIFY_STREAMS`, `LLAMA_VERIFY_STREAMS_DEPTH`, `LLAMA_VERIFY_STREAMS_TAIL`, `LLAMA_VERIFY_STREAMS_INPUT_ASYNC`, `LLAMA_VERIFY_STREAMS_PROMPT_INPUT_ASYNC`, `LLAMA_VERIFY_STREAMS_KV_STEP`, `LLAMA_VERIFY_STREAMS_SPLIT_ALONE` (test only), `LLAMA_SERVER_PREFILL_CAP`, `LLAMA_STAGGERED_MIXED_UBATCH`, `LLAMA_CKPT_ASYNC`, `LLAMA_GLM5_STABLE_PREFILL_CHUNKS`, `LLAMA_DSV41_RAW_WINDOW`, `LLAMA_DFLASH_DEV`, `LLAMA_SPEC_REJECTION_HOST`, `GGML_CUDA_RETIRE_MAX_MIB`, `GGML_GALLOC_GROW_PCT`, `GGML_GALLOC_GROW_MAX_MIB`, `GGML_GALLOC_GROW_NODE_PCT`, `GGML_GALLOC_GROW_NODE_MIN_KIB`, `GGML_GALLOC_GROW_NODE_MAX_MIB`, `GGML_CUDA_GRAPH_FALLBACK_LOG`, `GGML_CUDA_GET_ROWS_SMALL` and `GGML_HIP_LID_WMMA`.

## Public usage documentation

- Added `docs/delphi/usage/` guides for auto-tune, auto-quantize, the expert cache, speculative decoding, spec-depth, VRAM sharing, vision (dynamic/projector memory) and performance/placement, with an index, and updated the environment reference with the new defaults, per-variable options and the newly command-line-covered variables. The guides give working v1 GLM and DeepSeek command lines.
- Added the server prefix-cache guide and index entry, including media prompts, mmproj, disk files, restart behavior and the single-position checkpoint fix; updated the server-checkpoint guide for media and prefix snapshots.
- Added `docs/delphi/reference/server-flags.md` and documented the adaptive DSpark verification controls, complete-round cost model, single-slot scope and measured gain; updated the environment reference for the KV window variables.
- Clarified the README usage instructions, project description and credits.
- The bottleneck profiler, exec-trace family and tune endpoint documents remain from before this delta.
- No public documentation changed in the 2026-11-12 delta. The verify-stream, staggered-verify, checkpoint-async, GLM stable-chunk, DSV41 raw-window and CUDA growth/retirement variables are not yet in the public environment reference; the README feature list was updated with this documentation pass.

## Correctness and regression coverage added with these features

- Added grouped-MoE schedule/reference tests, schedule statistics and mixed-prefill/decode logits comparisons.
- Added imported-quantization and KS row-metadata tests, including routed and copy/gather operations.
- Added DeepSeek V4.1 converter, quantization-rule, architecture, shared-stream and save/load-state fixtures.
- Added GLM pooled-cache/sparse-attention and multimodal preprocessor coverage; fixed sparse tests to find the CPU backend by type in dynamically loaded builds.
- Added Qwen/DeepSeek sparse-attention cases covering causal prefixes, awkward selection widths, invalid/invisible selected rows and dense fallback.
- Added full-vocabulary/many-row top-k cases and sliding-window FlashAttention cases.
- Added fusion-specific backend cases for norm/scale and hyper-connection weight chains.
- Added CUDA INT4 ownership/lifetime/fallback, eager-versus-captured, strided-route and prefill-versus-decode comparison tools.
- Added HIP G128 batch-boundary, duplicate-route, strided-ID, partial/all-miss and packed-bank lifecycle checks.
- Added normal MiMo MTP hidden-input regression coverage.
- Kept the test NUMA threadpool attached to the tested CPU backend and fixed speculative-object construction after integration changes.
- Added quantization-refinement bounds and baseline-retention tests.
- Added offline trace/profiler integrity, decoding, vendor import, visualization, speculative-cycle and pipeline tests.
- Added VRAM lease, prompt-exclusivity, tier add/drop and release/re-reservation smoke hooks.
- Added DeepSeek V4.1 CED prefill microbatch boundary tests.
- Added Q4_G128 backend-operation coverage on CUDA and ROCm and a grouped-MMQ regression case.
- Added a batched-vec-dot exactness test (5400 dot results plus graph comparisons) that passes with the feature on and off.
- Added Muse Spark Open architecture roundtrip and chat cases.
- Added mixed-logits checks with NUMA host experts and an attached CPU pool.
- Added 1-token k=2..8 weighted-sum fusion cases and 384-expert router fusion cases, with byte-identical greedy-text comparisons against the unfused chains.
- Made `test-llama-archs` load the backends.
- Added Qwen QSA pooled-key cache and block-grouping tests with bitwise-identity comparisons; DeepSeek visible-count-mask cases (54 new test-backend-ops cases; MUL_MAT 200/200 on ROCm0 and CUDA0); IQ4_KS speculative-width and changing-mask FFN coverage.
- Added Engram row-reader coverage (GET_ROWS comparisons and mixed-logits placement cases) and loaded the fixture CPU backend dynamically in backend-DL builds.
- Added pinned expert-tier lifecycle tests with tensor NUMA (asynchronous replacement, release, promotion, scheduler recreation, per-device reserves, dense layer-split agreement), and initialized generic operation guards so recycled bytes cannot poison unchanged guards.
- Added DeepSeek V4.1 chat/template/converter fixture tests (official 1017 fixtures byte-identical, tool round trip 10 of 10) and the repacked-partial-`MUL_MAT_ID` test for out-of-range ids.
- Added RDNA4 padding launch-shape skip/range cases and DeepSeek-shaped ROPE/FLASH_ATTN_EXT and GET_ROWS perf cases; IQK register-decode paths pass `test-backend-ops` for the imported types and `MUL_MAT_ID` widths.
- Added `test-server-prefix-cache` for snapshot selection, eviction and file handling.
- Added adaptive DSpark block-controller and exact sparse-proposal tests, plus the explicit floor through the argument parser.
- Added a DeepSeek V4.1 prefill attention launch test against the CPU (512-wide heads, one KV head, sinks, 128-row sliding window plus 512 compressed rows per query; dense masked, dense `n_kv_max` and indexed launches; NMSE <= 2.3e-6 on gfx1201).
- Added layer-bank parity tests in independent processes (fixed-mode logit and decoder-feature captures, repeated prefill state, abort recovery) and byte-identical encoder-window cache-state checks with separate ordinary-restore controls and optional state dumps.
- Added staged-weight retention, HIP copy-graph lifecycle and HC fusion-tail tests.
- `test-backend-ops` quantizes row-meta weights on several threads, and `test-alloc` covers graph-edge layout changes.
- Added numeric reasoning-effort cases (strings, integers, request overrides, named levels, thinking off, invalid values).
- `test-auto-tune` covers the v1 tuner (13 tests); the observation test of the replaced M0-M3 tuner was removed.
- Added DeepSeek compressor-padding coverage on CPU, HIP and CUDA (graph reuse, logits, different sequence phases, rollback, checkpoint restore, padding slot becoming live) and updated the CED intermediate chunk-count expectation for the one-decoder-row compacted chunk.
- Extended `test-layer-input-device` with a small DFlash decoder: host and backend feature injection must produce identical serialized KV states across source offsets and repeated input-mode switches, which catches stale source addresses under CUDA graph replay.
- Added backend-sampler coverage for transaction rollback, changing penalty history, custom mutable inputs, scalar probability and rejection stream parity, and lazy probabilistic outputs.
- Added CUDA GET_ROWS coverage for partial blocks, strided views, batch-grid limits and full-vocabulary gathers.
- Added a scheduler input-staging ownership test for growth, graph rebuilds, replay and immediate source overwrites with single-copy and pipelined schedulers.
- Forced-window checkpoint tests compare asynchronous and synchronous snapshot bytes and check exact logits after checkpoint restore and replay.
- Added backend coverage for the RDNA4 WMMA lightning indexer.