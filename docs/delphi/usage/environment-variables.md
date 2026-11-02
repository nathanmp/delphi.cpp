# Runtime environment variables

delphi.cpp reads the environment variables below at run time. Each is an A/B switch: the default is the tested behavior, and the other value restores the previous behavior or enables a diagnostic. Set them in the environment of `llama-server`, `llama-cli`, `llama-bench` and the other tools, for example `GGML_SCHED_INPUT_ORDER=0 llama-server ...`. A value is read once per process.

Most of the variables below are read once, into a function-local `static`, and need a new process. The ones that a context, an expert cache or a scheduler reads when it is made (the `LLAMA_MOE_CACHE_*` sizes and prompt rows, `LLAMA_MOE_PREFILL_*`, `GGML_SCHED_*`) see a new value when the server makes the context again (`POST /tune/context` with `--tune-endpoints`). `tools/auto-tune/env-reload.json` lists the class of each tunable variable (`context` or `process`); see `tune-endpoints.md`.

Execution tracing has its own variables (`GGML_EXEC_TRACE*`); see `exec-trace-record.md`.

Since 2026-11-01 the main performance settings are also command-line options (`--moe-cache*`, `--moe-stage*`, `--cpu-expert-prefetch`, `--[no-]moe-pregate`, `--[no-]moe-shared-overlap`, `--engram-readers`, `--[no-]kv-mirror`, `--graph-alt-slots`, `--[no-]ckpt-tail-ubatch`). Each option sets the variable named in its `--help` entry, and a command-line value takes precedence over the environment. The option is given in the tables below where one exists.

## Model loading

| variable | default | effect |
| --- | --- | --- |
| `LLAMA_LOAD_PARALLEL` | `1` | Upload mapped weights to each device on its own thread. `0` uploads one device after another. |

With mmap (the default load mode), each tensor in a plain device buffer is copied by a worker thread of its device, so the host links of all devices are busy at once. The loader waits for the uploads before it unmaps the file, returns, or reports a cancellation. Tensors in split buffers (`-sm tensor`, `-sm row`), host buffers, and the no-mmap load path (`--load-mode none`, direct I/O) are uploaded on the loading thread as before.

The gain depends on how evenly the weights are spread and how slow the device links are. With a model split by layers over three GPUs on narrow PCIe links, loading from the page cache took half the time (249 s to 127 s). Models on one device, or with split buffers, load in the same time as before. Outputs are identical.

## DeepSeek V4.1 CED prefill

| variable | default | effect |
| --- | --- | --- |
| `LLAMA_DSV41_TRIM` | `1` | Compact decoder prefill. `0` uses full prefill; `2` keeps all queries with the bounded decoder mask as a numerical reference. Read when the context is made. |

For a supported Causal Encoder Decoder (CED) layout, the encoder processes every input row. At the first decoder layer, the graph keeps the last `sliding_window + rollback_depth` query rows. Global KV and index keys still use every encoder row. Decoder attention and FFNs process the retained suffix; subsequent decode runs all layers. Decoder local KV has its own positions, so discarded rows cannot expose stale cache slots. DSpark receives valid features for the retained suffix.

Compaction applies per microbatch. Short batches, batches with multiple sequences, embedding inputs, and requests for logits in the discarded prefix use the full graph. Unsupported source layouts disable CED. This includes decoder engrams or later decoder layers that own global KV or index keys. Ordinary all-row perplexity evaluation therefore uses the full graph; quality tests must prefill with only the final output requested, then score a continuation.

CED and full-prefill state blobs use different layouts and reject restoration across modes. Rebuild saved prompt states after changing the mode or upgrading from full prefill. The server's `/tune/context` endpoint can change this variable without reloading the model weights, but clears the prompt state.

CPU-offloaded DeepSeek V4.1 Flash tests on two R9700s showed about 1.7x faster 1K prefill and 1.9x faster 65K prefill. On 2,048 teacher-forced continuation tokens from prose, documentation and C++, compact versus full prefill had mean KL 0.00278, 98.73% top-token agreement and 0.93% higher aggregate perplexity. Four graded tasks had the same answer accuracy, including an arithmetic failure in both modes; normal DSpark acceptance remained close. These are regression checks on one quantization, not a broad quality evaluation. Outputs can change. A separate full-prefill microbatch-size control also changed distributions, so the measured difference cannot all be assigned to the bounded decoder history.

Adapted from [DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01), Apache-2.0: the consumer boundary in `decoder_layer.py` and `v4_modules/yoco.py`, full global-KV projection in `v4_modules/attn.py`, and bounded SWA visibility in `prepare_tensor_input_v4.cpp`. This implementation uses ggml graphs, separate local cache maps, and a suffix per microbatch rather than DeepInfer's request metadata. No source or kernels were copied.

## DeepSeek V4.1 small batches

| variable | default | effect |
| --- | --- | --- |
| `LLAMA_ENGRAM_PREFETCH` (`--engram-readers`) | `8` | Host reader count for Engram row lookup, capped at 32. `0` uses ordinary `GET_ROWS`. Read when the context is made. |
| `LLAMA_MOE_SHARED_OVERLAP` (`--[no-]moe-shared-overlap`) | `1` | Submit the GPU shared FFN before cold CPU expert arithmetic. `0` uses the previous graph order. Read once per process. |
| `LLAMA_KV_MIRROR` (`--[no-]kv-mirror`) | `1` | Keep a copy of the reused compressed KV rows (layers 20-39 read layer 20) on each GPU that reads them. Needed for long contexts on 3 or more GPUs. |
| `LLAMA_DSV41_VERIFY_GATHER_MIN` | `4096` | From this many compressed rows, each verify query attends to the raw window plus its own selected rows, so the cost does not grow with the context. `0` keeps the dense path, which runs out of memory after about 128K tokens at `-c 300000`. |
| `LLAMA_SERVER_CKPT_TAIL_UBATCH` (`--[no-]ckpt-tail-ubatch`) | `0` | `1` also saves a server prompt checkpoint one ubatch before the end of the prompt. Off since 2026-11-01: it costs a decoder pass per DeepSeek V4.1 prompt and gave GLM nothing. |

The Engram readers and the shared overlap apply to batches of at most 16 tokens. Engram readers gather known token-history rows from supported host tables into owned buffers; the graph consumes the result at the original layer. They use a separate pool from the CPU experts and respect the application CPU mask. Larger batches, unsupported tables and eval callbacks use ordinary lookup. A context without a supported host Engram table creates no readers.

Shared overlap requires the expert-cache path and shared weights on the layer GPU. The routing packet is copied to the CPU before the independent cached/shared GPU work is submitted. Routed expert selection, FFN arithmetic and the hyper-connection injection point are unchanged.

Teacher-forced checks at widths 1/2/3/4/5/8/16 and the width-17 fallback matched all 85 checked logit vectors bit for bit. Normal DSpark checks had equal acceptance counts. Both optimizations together improved repeated fixed-synthetic DSpark decode by about 4% on a tested hybrid CPU/GPU configuration; Engram benefit depends on page/cache state. Synthetic acceptance is performance-only evidence.

## Backend sampling (`-bs`)

| variable | default | set to | effect of the non-default value |
| --- | --- | --- | --- |
| `LLAMA_SAMPLER_RESERVE_ALWAYS` | off | `1` | Reserve the compute buffers again at each sampler change (upstream behavior) |

The server keeps the sampler object of a slot between requests when the sampling parameters are equal and the chain has no grammar or reasoning-budget sampler (it is reset, not made again); otherwise it sets a new sampler object. A reservation recreates the scheduler and all its compute buffers, and with staging the staging slots: 0.6 s per request on GLM-5.3 and Qwen 3.8, 1.7 s on DeepSeek V4.1 (measured from the execution trace of the first prompt batch). By default, the context reserves again only when the sampler chains of the sequences (the sampler names in each chain) are not the same as at the last reservation. A new sampler object always causes a new graph, not the reuse of a graph that was built with the old object. A sampler parameter that changes a tensor size (for example a larger top-k) can make the compute buffer grow on the next batch.

## Multi-device scheduler

These switches control how `ggml_backend_sched` copies and synchronizes split inputs between backends. They matter for models split by layers over several devices (`-sm layer`, pipeline parallelism). They change only copies and waits, never the computation, so outputs are identical with any setting.

| variable | default | set to | effect of the non-default value |
| --- | --- | --- | --- |
| `GGML_SCHED_VIEW_ALIAS` | on | `0` | Copy a full-size view input (for example a reshape of a hidden state) separately from its source |
| `GGML_SCHED_PREV_SPLIT_WAIT` | off | `1` | Before an input-less split, always wait for the previous split's backend |
| `GGML_SCHED_COPY_REUSE_WAIT` | off | `1` | Wait for the reuse of the input copy slot before every host input copy |
| `GGML_SCHED_INPUT_ORDER` | on | `0` | Copy split inputs in first-use order instead of the order of their producers |
| `GGML_SCHED_XDEV_EVENTS` | off (on with `LLAMA_MOE_SPLIT`) | `1` | Without pipeline parallelism, order a device-to-device input copy with an event on the devices instead of a host wait for the destination |
| `GGML_SCHED_REUSE_EVENTS` | `1` | `0` | Without pipeline parallelism, the host waits once per split, on an event recorded before its inputs are queued, before overwriting the inputs; `0` synchronizes the destination device before every device-to-device input copy, which also waits for the copies and staged uploads queued for the earlier inputs |

What each default does:

- **View alias.** When a split needs a view that covers all of a tensor from another backend, the scheduler copies the source once and gives the consumer a view of that copy. Previously the view and its source were each copied. This is skipped for weights, graph inputs, and any tensor that the graph writes in place, so every read sees the data it would have seen with two copies.
- **Previous-split wait.** The wait exists because the allocator may give a later split memory that the previous split still reads. It is now kept only when the two backends can share memory: the same buffer type, or one backend can use the other's buffer type (host buffers on integrated GPUs, CPU-like backends). Separate discrete GPUs never share compute memory, so they no longer wait on each other here.
- **Copy-slot reuse wait.** Before input copies into a slot, the host waits for the previous graph that used that slot. Once one host wait on a backend in the current graph has proven the slot free, later copies into the same backend skip it.
- **Device-to-device events.** Without pipeline parallelism there is one copy of each split input, and before overwriting it the host waits until the destination device is idle. With `GGML_SCHED_XDEV_EVENTS=1` the scheduler records an event on the destination at the start of the split and makes the source device's stream wait for it before the copy, so the host keeps queueing work. A node that reads a result of another device also starts a new split, so that the work before it does not wait for that device.
- **Input order.** Inputs are copied in the graph order of their producers. On systems without peer-to-peer access, the driver stages peer copies into a device one at a time in submission order. Copying the earliest-ready tensor first keeps a small copy from queueing behind one whose source is still being computed.

Validation used a layer split over three NVIDIA GPUs on narrow PCIe links without peer access, with multi-token prediction. Prompt processing rose by 15-29% at `-ub 2048` and by up to 52% with several ubatches per batch. Generation speed and outputs were unchanged. Most of the ub2048 gain comes from the view alias, because each device hand-off moves half as many bytes. The two wait changes and the input order matter when a server batch holds several ubatches: switching all three back cost 25% of prompt throughput at `-ub 512`.

## Qwen selected-row attention

`LLAMA_QWEN4EXP_SPARSE_ATTN` selects how Qwen QSA attends to the indexer's selected cache rows:

| value | effect |
| --- | --- |
| `0` | dense attention with the selection encoded in the mask |
| `1` | mask-backed attention with a selected-row count hint |
| `2` (default) | direct KV indices when FlashAttention and the layer device support them; otherwise the mask-backed path |

The RDNA4 direct path supports one sequence stream, F16 K/V, 256-wide heads, GQA ratio 12 and up to eight query rows. It requires a cache of at least 4096 rows and at least four times the selected width. Unsupported devices, cache formats and shapes keep masked attention; disabling FlashAttention also retains the selected-row mask.

Passing indices avoids building and compacting a full-width selection mask. A repeated 32K-context decode suffix improved by about 5% on the tested Qwen deployment, with identical greedy tokens. The real-MTP input was repetitive and had full acceptance, so that result does not establish the gain on arbitrary requests. Small contexts and prefill widths keep the existing path. Summation order can change floating-point results.

## ROCm libraries

| variable | default | effect |
| --- | --- | --- |
| `ROCBLAS_USE_HIPBLASLT` | rocBLAS default (hipBLASLt on the R9700 here) | read by rocBLAS, not by delphi.cpp. `0` makes rocBLAS use its own Tensile kernels instead of hipBLASLt |

hipBLASLt can pick a different kernel for the same GEMM in different processes, so results are not bitwise repeatable across runs. For MiMo V2.6 Flash, the F32 router SGEMM came out in two variants (largest difference 1.9e-5); top-k routing amplified this, and perplexity alternated between two values (5.3658 and 5.4084 on the first 8192-token chunk). With `ROCBLAS_USE_HIPBLASLT=0` every run gave the same result, so set it for A/B comparisons that expect bitwise equal outputs. It costs speed on F16 weights: Ling 3.0 tiny F16 pp4096 on an R9700 ran at 2780 instead of 5880 t/s. Qwen3.6-35B-A3B Q4_K_M, Gemma 4 31B Q6_K and MiMo two-GPU prefill ran at the same speed, and MiMo's first 16K request was faster (1280 instead of 1090 t/s), as hipBLASLt's first calls are slow.

## GPU graphs (CUDA and HIP)

| variable | default | effect |
| --- | --- | --- |
| `GGML_CUDA_GRAPH_CACHE_SIZE` | `2048` | number of captured graphs a device keeps; graphs with more than 256 nodes stay bounded at 64 |

Every distinct graph (each verify width and each draft step of speculative decoding, on every device) takes an entry. With 256 entries, Qwen 3.8 Flash Next with MTP on three R9700s (layer split) sometimes evicted its decode graphs: a whole request at draft depth 3 ran at 18 instead of 58 t/s, and at depth 5 most requests ran at 7-17 t/s. With 2048 entries every request ran at full speed, at under 0.1 GiB more memory per device.

## Speculative decoding: graphs of the verify widths

A per-round draft depth (the default conf mode of `draft-mtp`, see `speculative-draft-depth.md`) verifies a different number of tokens in most rounds, so the target and the draft context keep one graph per width. These switches keep those graphs valid.

| variable | default | effect |
| --- | --- | --- |
| `LLAMA_KV_N_PAD` | unset: 256, 2048 with a per-round draft depth | the KV view grows in steps of this many cells (multiple of 256). A step makes the graphs of all widths stale. Set, it wins over the reservation of the conf mode: `256` gives the upstream step |
| `GGML_CUDA_FA_KV_MAX_DECODE` | `1` | flash attention scans the mask for a fully masked tail also for fewer than 1024 query rows, so that the masked cells of a padded KV view cost nothing. `0`: upstream behavior. Outputs are identical |
| `LLAMA_GRAPH_ALT_SLOTS` (`--graph-alt-slots`) | `4`, n_max + 2 with a per-round draft depth | graphs kept for small batches (up to `LLAMA_GRAPH_ALT_MAX_TOKENS`, default 32, tokens); set, it wins |
| `LLAMA_GRAPH_BUILD_LOG` | off | `1` logs each graph build with the reuse check that failed, and its host times (build, scheduler allocation, first compute) |
| `DELPHI_SPEC_PLOG` | off | a file name: the MTP draft writes the top-1 probability of each draft step and the accepted count per round (JSON lines), for offline replays of stop rules |

On GLM-5.3 Flash (3x CMP 170HX) a graph build of a target width costs about 2 ms (build) + 10 ms (allocation) + 20 ms (first run); with the step 256 the conf mode built 68 graphs per 2048-token request and lost to a fixed depth, with 2048 it was the fastest mode. The masked cells cost GLM's sparse attention about 0.5 ms per round at the step 2048. On Qwen 3.8 Flash Next (3x R9700) they cost 4.4 ms per round (44.0 instead of 39.6 ms) until the flash-attention scan skipped them.

## Weight streaming (staging ring)

With routed experts in host memory (`-cmoe`, `-ncmoe`, `--numa tensors`), a wide prompt batch can run the expert matrix multiplications on the GPU: the scheduler copies each expert tensor to the device when it reaches the split that uses it (op offload). That copy runs in turn with the device's computation, and a copy from ordinary (pageable) host memory also blocks the scheduler thread at well below the PCIe rate.

| variable | default | effect |
| --- | --- | --- |
| `GGML_SCHED_STAGE` (`--[no-]moe-stage`) | `1` | uploads offloaded host weights ahead of the splits that use them. `0` turns staging off. It allocates nothing without routed experts in host memory. A GPU whose expert-cache tier is smaller than one slot does not stage (logged at the first decode). Default on since 2026-11-01: DeepSeek V4.1 4K prefill 444 -> 637 t/s, decode unchanged |
| `LLAMA_MOE_STAGE_DEVICES` (`--moe-stage-devices`) | the expert-cache GPUs | GPUs that stage, `DEV,...`, or `none` |
| `GGML_SCHED_STAGE_REDIRECT` | `1` | kernels read staged inputs in their ring slots instead of a copy of them; the slot is freed once the split's work is queued. `0` copies each staged input from its slot first (about 3 ms per expert tensor) |
| `GGML_SCHED_STAGE_SLOTS` (`--moe-stage-slots`) | `2` (`4` before 2026-11-01) | device slots, each the size of the largest staged tensor; more slots run further ahead and use more VRAM. MiMo V2.6 Flash with 3 slots gave NaN perplexity in 2 of 9 runs (cause unknown, later runs finite); 6 slots were always finite |
| `GGML_SCHED_STAGE_BOUNCE` | `4` | pinned host bounce buffers; `0` uploads from the weights directly (pageable) |
| `GGML_SCHED_STAGE_CHUNK_MB` | `64` | size of each bounce buffer |
| `GGML_SCHED_STAGE_THREADS` | 8 / staging devices, at least 2 | memcpy threads per staging device that fill its bounce buffers |
| `GGML_SCHED_STAGE_PIN` | `1` | run each device's staging threads on the CPUs next to that device (Linux) |
| `GGML_SCHED_STAGE_REGISTER` | `0` | `1` page-locks the staged host weight buffers when the ring first stages from them, so uploads go straight from the weights to the device without the bounce buffers |
| `GGML_SCHED_STAGE_STATS` | `0` | `1` prints where each staging thread spent its time when the scheduler is freed |
| `GGML_SCHED_STAGE_SPREAD` (`--[no-]moe-stage-spread`) | `1` (`0` before 2026-11-01) | with several staging devices: layer N's staged expert multiplies run on staging device N mod n, so each device's link uploads only its own layers' experts. MiMo V2.6 Flash 16K prefill on two R9700s (attention on ROCm1): 1150-1300 tokens/s, against 1060-1120 on one |
| `GGML_SCHED_STAGE_MIN_TOKENS` (`--moe-stage-min-tokens`) | measured at warmup (`512` with `--no-warmup`) | routed-expert multiplies on host weights run on the CPU for batches of fewer tokens and on a staging GPU, which streams all experts, for wider ones (every expert crosses the link whatever the batch size). Without the variable, the server and cli measure it at warmup, after the expert cache is allocated: prompts of random tokens of n/2 and n tokens with the experts on the CPU and streamed (n = `-ub`, at most 2048), and the threshold is where the CPU expert time reaches the streamed cost (flat at the link time for short batches, rising with the GPU compute for long ones); when a 64 MiB copy to the GPUs shows that streaming takes 3 times the CPU time of a full ubatch, the experts stay on the CPU without the streamed prompts. This adds a few seconds to the startup at `-ub 1024` with 56 threads, about 25 s at `-ub 4096` with 16 (Qwen 3.8, R9700). `-lv 4` prints `staging threshold N tokens`; `-ub` + 1 means never. Setting the variable (also `0`) skips the measurement. MiMo V2.6 Flash, R9700s: 300 tokens 81-94 t/s with CPU experts; 2000 tokens 284-296 t/s streamed to one GPU, 427-462 with `GGML_SCHED_STAGE_SPREAD=1` on two. `0` leaves the choice to the offload cost model |

At the start of each graph, the scheduler lists the host weights that its splits will copy whole, in split order. Each staging device has its own background thread, which copies that device's weights in chunks into pinned host buffers, and the device copies each chunk into a slot on a separate stream, so the uploads of later layers overlap the computation of earlier ones. When the scheduler reaches the split, the device copies the slot into the split input. Expert tensors are staged only when the batch is wide enough that nearly every expert is used; for narrower batches the scheduler keeps copying only the used experts, and decode is unaffected. Supported on CUDA and ROCm devices.

Each slot takes VRAM equal to the largest staged tensor (1.4 GiB for DeepSeek V4.1 Flash at Q2_K). If the slots cannot be allocated, staging turns itself off with a warning. Staging does not change what is computed, so outputs are identical with it on or off.

The offload cost model (or `GGML_OP_OFFLOAD_MIN_BATCH`) still decides whether the experts are offloaded at all, so staging matters only on a fast host link. Measured with DeepSeek V4.1 Flash (routed experts at Q2_K in host memory) on one Radeon AI PRO R9700 on PCIe 4.0 x16 at `-ub 2048`: a real-text prompt went from 84.5 to 134 tokens/s, random tokens (llama-bench pp4096) from 92 to 125 tokens/s, and decode was unchanged. On a PCIe 4.0 x2 link, the cost model keeps the experts on the CPU.

Every ubatch uploads all staged weights once, so the link sets a floor per token of (staged bytes) / (link rate × ubatch size): for DeepSeek V4.1 Flash, about 3.1 ms/token at `-ub 2048` and 1.6 ms/token at `-ub 4096`. A larger ubatch therefore pays off for long prompts, as long as its compute buffer fits. With the later kernel work on the same card, real-text prefill reached 255 tokens/s at `-ub 2048` and 484-489 tokens/s at `-ub 4096`. At `-ub 4096`, three slots ran as fast as four and used 1.4 GiB less VRAM (two slots lost 25%), so `GGML_SCHED_STAGE_SLOTS=3` is a good choice when VRAM is tight, for example next to a draft model. On a dual-socket EPYC 7532 host, pinning the staging threads next to the GPU raised the bounce-buffer fill rate from 26 to 35 GB/s and prefill from 468 to 494 tokens/s at `-ub 4096` (11.4k-token prompt); `GGML_SCHED_STAGE_THREADS=4` added a little more (508 tokens/s).

With a layer split over two GPUs (DeepSeek V4.1 Flash, two R9700s on PCIe 4.0 x16, `-ub 4096`), staging does not speed up prefill: 476 vs 530 tokens/s (3.9k-token prompt) and 417 vs 448 tokens/s (30.7k tokens) on one GPU. Within a ubatch the second GPU waits for the first, and its ring refills only as the scheduler reaches its splits, so the two links take turns rather than running at once. The per-device staging threads are for layouts where devices consume staged weights at the same time.

### Expert split over two devices

| variable | default | effect |
| --- | --- | --- |
| `LLAMA_MOE_SPLIT` | unset | `<device>[:<n>]`: compute `n` of the routed experts (default half) of each MoE layer whose experts sit in host memory on a second GPU, for example `ROCm2` |
| `LLAMA_MOE_SPLIT_MIN_TOKENS` | `1024` | smallest ubatch that splits the experts; smaller ubatches (decode) run as before |
| `LLAMA_ATTN_SPLIT` | unset | `<groups>`: with `LLAMA_MOE_SPLIT`, the same device also computes the attention heads of that many output groups (DeepSeek V4.1, 8 groups of 8 heads; use an even number, since the FA kernel takes 16 heads at a time) |
| `LLAMA_ATTN_SPLIT_COPY` | `1` | with `LLAMA_ATTN_SPLIT`, keep the split device's slices of `wq_b`, `wo_a` and the sinks in its memory (copied once at context creation, 1.5 GiB for DeepSeek V4.1 Flash) instead of copying them from the first device every ubatch; `0` turns it off |
| `LLAMA_ATTN_SPLIT_INDEXER` | `0` | `1`: with `LLAMA_ATTN_SPLIT`, the split device also scores the second half of the query rows in the indexer. Experimental: faster on long prompts, but it changes the greedy output although each row should score the same |

With the experts in host memory (`-cmoe`) and the staging ring on, each device streams and computes its own part of every MoE layer: the second device takes the top `n` experts, the first device the rest, together with everything else. Both parts run the same routing through `ggml_mul_mat_id_partial`, which skips the ids of the other part, and the first device adds the two partial sums. Per MoE layer the devices exchange the normed activations and the partial sum (64 MiB each at 4096 tokens). The second device needs no model layers: it is added to the context as an extra backend. Outputs change only by summation order: on DeepSeek V4.1 Flash the KL divergence to the one-device run is 0.0077 (same top token 98.1%), below the 0.0086 between `-ub 4096` and `-ub 512` on one device.

Measured with DeepSeek V4.1 Flash (Q2_K experts, `--numa tensors`) on two R9700s on PCIe 4.0 x16 next to one socket, server, `-ub 4096`, 3.9k- and 30.7k-token prompts:

| setup | 3.9k tokens | 30.7k tokens |
| --- | ---: | ---: |
| one device | 503 t/s | 440 t/s |
| split | 514 t/s | 459 t/s |
| split + `GGML_SCHED_STAGE_REGISTER=1` | 546 t/s | 481 t/s |

Without page-locked weights the two staging threads share the host's memcpy bandwidth (about 30 GB/s together, against 50 GB/s for the two links). With them, the first device becomes compute-bound: it keeps attention and the dense layers, about 120 ms per layer against 43 ms for the second device's experts, and the heavier attention layers take 170 ms. Giving the second device more experts does not help (224 of 384: 556 / 478 t/s; 256: 542 / 465). Page-locking 166 GiB of experts adds 142 s before the first staged prompt, which is why it is off by default.

With `LLAMA_ATTN_SPLIT=4` the second device also takes half the attention heads. Attention in DeepSeek V4.1 has one K head, and each query's top-k selection covers all its heads, so the second device reads the same K and top-k as the first and builds its own mask. It computes the queries, sparse FlashAttention and `wo_a` of its groups; the first device adds its half of `wo_b` to the product of the second half. Per layer, `qr` (F16, 10 MiB at 4096 tokens), K (6-36 MiB, growing with the context) and the top-k go to the second device, and the `wo_a` output (F16, 32 MiB) comes back. The second device takes all of these, and the slices of `wq_b`, `wo_a` and the sinks, before its first kernel, since with `GGML_SCHED_XDEV_EVENTS` a copy in the middle of its work would wait for everything the first device queued before it. The indexer stays on the first device. The KL divergence to the one-device run is 0.0080 (same top token 97.7%), against 0.0086 (97.7%) for `-ub 4096` against `-ub 512` on one device.

Same host, server, `GGML_SCHED_STAGE_SLOTS=8`, several requests per server (3.9k and 12.8k tokens), then one 30.7k-token request per server from an earlier build:

| setup | 3.9k tokens | 12.8k tokens | 30.7k tokens |
| --- | ---: | ---: | ---: |
| one device | 516-534 t/s | 453 t/s | |
| expert split | 508-575 t/s | 436-483 t/s | 492 t/s |
| expert and attention split | 541-617 t/s (mostly 595-617) | 501-507 t/s | 517 t/s |

With the attention split, a layer without heavy attention takes about 112 ms instead of 141 ms. With three staging slots, though, the expert stream can no longer run a layer ahead and every other layer waits for its experts. Use more slots: four or more (the default is four), and five to eight measured best. With eight slots, page-locking the experts no longer helps. On long prompts the indexer, which stays on the first device, takes about as long as FlashAttention. Reading staged inputs in their slots, and keeping the split device's head weights in its memory, raised a 3.9k-token prompt from 600 to 667 t/s and a 12.8k-token prompt from 506 to 535 t/s (`llama-completion`, eight slots), with identical output. With the experimental indexer split as well, the 12.8k-token prompt ran at 553 t/s.

## CUDA routed MoE (NVIDIA Ampere and newer)

These variables choose how routed-expert matrix multiplications (`MUL_MAT_ID`) run on NVIDIA GPUs of compute capability 8.0 and newer. Other devices, including ROCm, always use the standard path and ignore them.

| variable | default | values |
| --- | --- | --- |
| `GGML_CUDA_MOE_GROUPED` | `packed` | `old` or `0`: standard MMQ path; `auto` or `1`: grouped MMQ from `GGML_CUDA_MOE_GROUPED_MIN_TOKENS` tokens; `grouped` or `2`: grouped MMQ for every supported call; `packed`: grouped, with packed weights for the supported types |
| `GGML_CUDA_MOE_PACKED_INPLACE` | `1` | `0` keeps the GGUF weights and allocates a packed copy (twice the expert memory) |
| `GGML_CUDA_MOE_PACKED_TYPES` | all | comma-separated list of the weight types to pack, for example `iq4_xs,q4_K`; other types use grouped MMQ |
| `GGML_CUDA_MOE_GROUPED_MIN_TOKENS` | `128` | token threshold of `auto` |
| `GGML_CUDA_MOE_PACKED_VERIFY` | `0` | `1` unpacks every newly packed tensor on the device and aborts if a byte differs from the GGUF data |
| `GGML_CUDA_MOE_PACKED_NOCACHE` | `0` | `1` rebuilds the packed weights on every call (for tests that reuse buffer addresses; slow; disables in-place packing) |
| `GGML_CUDA_MOE_INT4_DECODE` | `1` | `0` disables the fused INT4 FFN kernels for `Q4_G128` routed experts (GLM INT4 models). They are on by default only when a `Q4_G128` tensor is in an NVIDIA GPU buffer. Without them, GLM-5.3 Flash INT4 adaptive MTP decode is about 12% slower (79.9 vs 90.7 t/s). `GGML_CUDA_DISABLE_FUSION=1` does not disable them: without them the packed experts would be converted to `Q4_0` for each operation. |

**Packed mode.** Before a graph first uses an expert weight, the weight is rearranged into a layout that the tensor cores can read directly. The rearrangement is lossless and keeps the tensor size, so by default it is done in place. Prompt processing then uses a packed grouped kernel, and single-token and speculative verification batches use packed-aware MMVQ kernels. Packed types: Q8_0, Q5_0, Q4_1, Q5_1, IQ4_NL, MXFP4, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ2_KS, IQ3_KS, IQ4_KS and IQ5_KS. Other types use grouped MMQ. A weight is packed only if its activations have a layout that the grouped kernels serve (broadcast gate/up inputs or gathered down inputs, as produced by the standard MoE graph); otherwise it keeps the GGUF layout.

Packed and grouped kernels add the same products in a different order from the old path. Measured on GLM-5.3 Flash (IQ4_XS), perplexity changed by a ratio of 1.001 ± 0.003 with KL divergence at the floor of any summation-order change. Decode logits were identical, and repeated runs gave byte-identical logits. Prompt processing was 22-26% faster than `old`, and speculative-decoding generation was about 1% slower.

**Limitations of in-place packing:**
- After packing, the device copy of an expert weight is no longer in GGUF layout. Reading it back with `ggml_backend_tensor_get` returns packed bytes, and writing new data to a packed tensor is not detected.
- A code path that can read only the GGUF layout aborts with a message naming the tensor, rather than computing a wrong result.
- Use `GGML_CUDA_MOE_PACKED_INPLACE=0` (if the memory allows it) or `GGML_CUDA_MOE_GROUPED=old` when you need the original layout on the device.

`GGML_CUDA_MOE_GROUPED` also accepts `packed-f16` and `packed-i8`, which are measurement-only formats 2.1-4.7x the GGUF size.

## CPU routed experts (`--numa tensors`)

| variable | default | effect |
| --- | --- | --- |
| `GGML_NUMA_NODE_LOCAL` | `1` | MUL_MAT_ID keeps each NUMA node's copy of the activations, its row map and its piece counters in that node's memory, and the workers of a node wait only for each other after they convert the activations. `0` uses the shared work buffer and waits for all nodes (previous behavior); it also turns off `GGML_NUMA_STEAL`. |
| `GGML_NUMA_STEAL` | `1` | A worker that has no more pieces of its own node takes spare pieces of the experts that another node has not started, nearest node first. `0` keeps each node on its own rows. |
| `GGML_CPU_NUMA_CHUNKS` | `2` | Pieces per worker of a node's rows of an expert. `0` gives each worker one static slice. |
| `GGML_CPU_EXPERT_PREFETCH` (`--cpu-expert-prefetch`) | `6` | Only with `--numa tensors`: while a GPU runs attention, the CPU reads this many experts per layer that it will likely need next into its caches. `0` turns it off. DeepSeek V4.1 on 2 R9700: verify -6.8 %, decode +6.5 %. `GGML_CPU_EXPERT_PREFETCH_STATS=1` prints at exit how much of the GPU wait each read left; near 0 means the budget is too large |
| `LLAMA_MOE_PREGATE` (`--[no-]moe-pregate`) | `1` | DeepSeek V4.1: run the next layer's router early to predict the experts for the prefetch. Off when the prefetch is off or not in `--numa tensors` mode |
| `GGML_NUMA_TP_SHARES` | equal | Relative share of each node in the column split, for example `1,3`. It also sets where the weight rows are placed at load. |
| `LLAMA_NUMA_HOST_INTERLEAVE` | `1` | With a GPU (CUDA or ROCm), the CPU compute buffer (activations and expert outputs of the CPU part) is pinned host memory with its pages interleaved over all NUMA nodes. `0` uses the GPU's plain pinned host memory, which is on the GPU's node only. |
| `LLAMA_MOE_CACHE_REFILL_INTERLEAVE` | `0` | `1`: the pinned staging buffer of expert-cache refreshes is interleaved over all NUMA nodes instead of lying on the GPU's node. For NPS4. |
| `LLAMA_MOE_CACHE_REFILL_NUMA` | `0` | `1`: each refresh copy thread runs on the node that holds its part of the expert rows (at least 2 threads per node). For NPS4, with the previous switch; together they cut a MiMo refresh from ~210 to ~150 ms per prompt. |

The first two change only where data is kept and which thread computes a piece, so outputs are bitwise identical with any setting (checked on a MiMo V2.6 Flash layer at 1 to 512 tokens, with equal and 1:3 shares). Stealing applies to experts on the per-row and G128 paths (for example Q4_G128 and MXFP4), not to the IQK paths, which split statically. A stolen piece reads its weight rows from the other node's memory. `LLAMA_NUMA_HOST_INTERLEAVE` changes only where the buffer's pages are.

Measured on MiMo V2.6 Flash (2x EPYC 7532, production deployment, -t 48, MTP decode and 128/300-token CPU prompts):
node-local state gave prompts -5% at NPS1 and decode +2.7% at NPS4; stealing was neutral (no node was slowed by other
load); the interleaved compute buffer gave decode +8.6% and prompts -13% / -25% at NPS4.

## GPU expert cache (routed experts in host memory)

When the routed experts stay in host memory (`-cmoe`, `-ncmoe` with all layers, `-ot exps=CPU`) and the other layers are on
one GPU, the most-used experts are copied to GPU memory and run there; the others run on the CPU. Which experts are hot comes
from the routing of the last `LLAMA_MOE_CACHE_WINDOW` rows. The cache refreshes after each prompt and every
`LLAMA_MOE_CACHE_EVERY` small batches while decoding. Batches of 16 rows or fewer use it; with `LLAMA_MOE_CACHE_PROMPT_ROWS`,
CPU prompt batches up to that size use it too.

Requirements: SiLU (or the clipped SiLU of DeepSeek V4) routed FFN with separate gate, up and down experts, no expert biases or
scales, one geometry per projection, all non-expert layers on one GPU. Other models get no cache in auto mode, and an error
with an explicit size.

The weight-streaming ring (above) takes its slots from the cache tier on the same GPU while a long prompt runs, and the cache
takes them back at the next refresh. So the cache and the staging slots do not need separate memory.

| variable | default | effect |
| --- | --- | --- |
| `LLAMA_MOE_CACHE_MIB` (`--moe-cache`) | `auto` | `auto` (or unset): when the model allows it, one tier per GPU of `--device`, of the GPU's free memory less `LLAMA_MOE_CACHE_AUTO_MARGIN_MIB`. The tiers are allocated when the server has loaded every model (draft and MTP included); other programs allocate them after the warmup. A number: one tier of that many MiB on the GPU of the layers. `0`: no cache. |
| `LLAMA_MOE_CACHE_AUTO_MARGIN_MIB` (`--moe-cache-margin`) | `1024` | memory left free on each GPU in auto mode, for buffers that grow later. The decode-time refresh scratch (3 slabs per `LLAMA_MOE_CACHE_ASYNC_EXPERTS` expert) is kept out of the tier in addition. |
| `LLAMA_MOE_CACHE_TIERS` | none | more tiers, `<device>:<MiB>[,...]`, for example `ROCm0:16384` |
| `LLAMA_MOE_CACHE_PROFILE` | none | expert frequencies (`tools/moe-static-cache/profile.py`), used until the window has rows |
| `LLAMA_MOE_CACHE_WINDOW` (`--moe-cache-window`) | `256` | rows of routing that choose the hot experts; `0` keeps every row, negative never refreshes |
| `LLAMA_MOE_CACHE_EVERY` | `8` | small batches between decode-time refreshes (at most `LLAMA_MOE_CACHE_ASYNC_EXPERTS` experts each); `0` refreshes only after prompts |
| `LLAMA_MOE_CACHE_ASYNC_EXPERTS` (`--moe-cache-async`) | `64` | experts per decode-time refresh. Each one needs 3 slabs of device scratch on every tier GPU and of pinned memory (DeepSeek V4.1: 18 MiB per expert). DeepSeek on 2 R9700: 32 36.8, 64 37.2, 96 37.9, 128 37.2, 192 36.7 t/s. |
| `LLAMA_MOE_CACHE_REFRESH_MAX` (`--moe-cache-refresh-max`) | `-1` (all) | experts uploaded in the refresh after a prompt; the decode-time refreshes upload the rest. DeepSeek V4.1 uses 512 |
| `LLAMA_MOE_CACHE_ROLES` (`--moe-cache-roles`) | `rotate` | role of each GPU's tier, `DEV:ROLE,...`: `rotate` follows the routing, `static` keeps the hottest experts, `pinned` is static and never lends slots to staging, `off` gives no tier |
| `LLAMA_MOE_CACHE_PROMPT_ROWS` | `0` | prompt batches up to this many rows also run their hot experts on the GPU |
| `LLAMA_MOE_CACHE_SHARED` | `0` | `1`: no tier of its own; tiers are leased at run time by the VRAM-sharing daemon. Turns auto mode off. |
| `LLAMA_MMPROJ_DYNAMIC_GPU_MIB` | none | Optional limit on the device memory that each dynamic encoding may use (`--mmproj-memory dynamic`). `0` forces CPU encoding. See [vision memory](vision-memory.md). |

DeepSeek V4.1 Flash (Q2_K experts, DSpark, `-ts 1,0` on ROCm1 with ROCm0 for staging, 256-token decode after short
prompts): auto mode put 7.1 GiB on ROCm1 and 29.6 GiB on ROCm0. Decode was 36.3 t/s against 34.3 without the cache, with 90%
of the expert selections on the GPU. KL divergence against the CPU path was 0.015 (8 x 512 tokens, 16-token batches), the
order of a change of summation order.
