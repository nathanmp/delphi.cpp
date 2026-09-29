# Runtime environment variables

delphi.cpp reads the environment variables below at run time. Each is an A/B switch: the default is the tested behavior, and the other value restores the previous behavior or enables a diagnostic. Set them in the environment of `llama-server`, `llama-cli`, `llama-bench` and the other tools, for example `GGML_SCHED_INPUT_ORDER=0 llama-server ...`. A value is read once per process.

Execution tracing has its own variables (`GGML_EXEC_TRACE*`); see `exec-trace-record.md`.

## Model loading

| variable | default | effect |
| --- | --- | --- |
| `LLAMA_LOAD_PARALLEL` | `1` | Upload mapped weights to each device on its own thread. `0` uploads one device after another. |

With mmap (the default load mode), each tensor in a plain device buffer is copied by a worker thread of its device, so the host links of all devices are busy at once. The loader waits for the uploads before it unmaps the file, returns, or reports a cancellation. Tensors in split buffers (`-sm tensor`, `-sm row`), host buffers, and the no-mmap load path (`--load-mode none`, direct I/O) are uploaded on the loading thread as before.

The gain depends on how evenly the weights are spread and how slow the device links are. With a model split by layers over three GPUs on narrow PCIe links, loading from the page cache took half the time (249 s to 127 s). Models on one device, or with split buffers, load in the same time as before. Outputs are identical.

## Multi-device scheduler

These switches control how `ggml_backend_sched` copies and synchronizes split inputs between backends. They matter for models split by layers over several devices (`-sm layer`, pipeline parallelism). They change only copies and waits, never the computation, so outputs are identical with any setting.

| variable | default | set to | effect of the non-default value |
| --- | --- | --- | --- |
| `GGML_SCHED_VIEW_ALIAS` | on | `0` | Copy a full-size view input (for example a reshape of a hidden state) separately from its source |
| `GGML_SCHED_PREV_SPLIT_WAIT` | off | `1` | Before an input-less split, always wait for the previous split's backend |
| `GGML_SCHED_COPY_REUSE_WAIT` | off | `1` | Wait for the reuse of the input copy slot before every host input copy |
| `GGML_SCHED_INPUT_ORDER` | on | `0` | Copy split inputs in first-use order instead of the order of their producers |
| `GGML_SCHED_XDEV_EVENTS` | off (on with `LLAMA_MOE_SPLIT`) | `1` | Without pipeline parallelism, order a device-to-device input copy with an event on the devices instead of a host wait for the destination |

What each default does:

- **View alias.** When a split needs a view that covers all of a tensor from another backend, the scheduler copies the source once and gives the consumer a view of that copy. Previously the view and its source were each copied. This is skipped for weights, graph inputs, and any tensor that the graph writes in place, so every read sees the data it would have seen with two copies.
- **Previous-split wait.** The wait exists because the allocator may give a later split memory that the previous split still reads. It is now kept only when the two backends can share memory: the same buffer type, or one backend can use the other's buffer type (host buffers on integrated GPUs, CPU-like backends). Separate discrete GPUs never share compute memory, so they no longer wait on each other here.
- **Copy-slot reuse wait.** Before input copies into a slot, the host waits for the previous graph that used that slot. Once one host wait on a backend in the current graph has proven the slot free, later copies into the same backend skip it.
- **Device-to-device events.** Without pipeline parallelism there is one copy of each split input, and before overwriting it the host waits until the destination device is idle. With `GGML_SCHED_XDEV_EVENTS=1` the scheduler records an event on the destination at the start of the split and makes the source device's stream wait for it before the copy, so the host keeps queueing work. A node that reads a result of another device also starts a new split, so that the work before it does not wait for that device.
- **Input order.** Inputs are copied in the graph order of their producers. On systems without peer-to-peer access, the driver stages peer copies into a device one at a time in submission order. Copying the earliest-ready tensor first keeps a small copy from queueing behind one whose source is still being computed.

Validation used a layer split over three NVIDIA GPUs on narrow PCIe links without peer access, with multi-token prediction. Prompt processing rose by 15-29% at `-ub 2048` and by up to 52% with several ubatches per batch. Generation speed and outputs were unchanged. Most of the ub2048 gain comes from the view alias, because each device hand-off moves half as many bytes. The two wait changes and the input order matter when a server batch holds several ubatches: switching all three back cost 25% of prompt throughput at `-ub 512`.

## Weight streaming (staging ring)

With routed experts in host memory (`-cmoe`, `-ncmoe`, `--numa tensors`), a wide prompt batch can run the expert matrix multiplications on the GPU: the scheduler copies each expert tensor to the device when it reaches the split that uses it (op offload). That copy runs in turn with the device's computation, and a copy from ordinary (pageable) host memory also blocks the scheduler thread at well below the PCIe rate.

| variable | default | effect |
| --- | --- | --- |
| `GGML_SCHED_STAGE` | `0` | `1` uploads offloaded host weights ahead of the splits that use them |
| `GGML_SCHED_STAGE_REDIRECT` | `1` | kernels read staged inputs in their ring slots instead of a copy of them; the slot is freed once the split's work is queued. `0` copies each staged input from its slot first (about 3 ms per expert tensor) |
| `GGML_SCHED_STAGE_SLOTS` | `4` | device slots, each the size of the largest staged tensor; more slots run further ahead and use more VRAM |
| `GGML_SCHED_STAGE_BOUNCE` | `4` | pinned host bounce buffers; `0` uploads from the weights directly (pageable) |
| `GGML_SCHED_STAGE_CHUNK_MB` | `64` | size of each bounce buffer |
| `GGML_SCHED_STAGE_THREADS` | 8 / staging devices, at least 2 | memcpy threads per staging device that fill its bounce buffers |
| `GGML_SCHED_STAGE_PIN` | `1` | run each device's staging threads on the CPUs next to that device (Linux) |
| `GGML_SCHED_STAGE_REGISTER` | `0` | `1` page-locks the staged host weight buffers when the ring first stages from them, so uploads go straight from the weights to the device without the bounce buffers |
| `GGML_SCHED_STAGE_STATS` | `0` | `1` prints where each staging thread spent its time when the scheduler is freed |

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

**Packed mode.** Before a graph first uses an expert weight, the weight is rearranged into a layout that the tensor cores can read directly. The rearrangement is lossless and keeps the tensor size, so by default it is done in place. Prompt processing then uses a packed grouped kernel, and single-token and speculative verification batches use packed-aware MMVQ kernels. Packed types: Q8_0, Q5_0, Q4_1, Q5_1, IQ4_NL, MXFP4, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ2_KS, IQ3_KS, IQ4_KS and IQ5_KS. Other types use grouped MMQ. A weight is packed only if its activations have a layout that the grouped kernels serve (broadcast gate/up inputs or gathered down inputs, as produced by the standard MoE graph); otherwise it keeps the GGUF layout.

Packed and grouped kernels add the same products in a different order from the old path. Measured on GLM-5.3 Flash (IQ4_XS), perplexity changed by a ratio of 1.001 ± 0.003 with KL divergence at the floor of any summation-order change. Decode logits were identical, and repeated runs gave byte-identical logits. Prompt processing was 22-26% faster than `old`, and speculative-decoding generation was about 1% slower.

**Limitations of in-place packing:**
- After packing, the device copy of an expert weight is no longer in GGUF layout. Reading it back with `ggml_backend_tensor_get` returns packed bytes, and writing new data to a packed tensor is not detected.
- A code path that can read only the GGUF layout aborts with a message naming the tensor, rather than computing a wrong result.
- Use `GGML_CUDA_MOE_PACKED_INPLACE=0` (if the memory allows it) or `GGML_CUDA_MOE_GROUPED=old` when you need the original layout on the device.

`GGML_CUDA_MOE_GROUPED` also accepts `packed-f16` and `packed-i8`, which are measurement-only formats 2.1-4.7x the GGUF size.
