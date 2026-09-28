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

What each default does:

- **View alias.** When a split needs a view that covers all of a tensor from another backend, the scheduler copies the source once and gives the consumer a view of that copy. Previously the view and its source were each copied. This is skipped for weights, graph inputs, and any tensor that the graph writes in place, so every read sees the data it would have seen with two copies.
- **Previous-split wait.** The wait exists because the allocator may give a later split memory that the previous split still reads. It is now kept only when the two backends can share memory: the same buffer type, or one backend can use the other's buffer type (host buffers on integrated GPUs, CPU-like backends). Separate discrete GPUs never share compute memory, so they no longer wait on each other here.
- **Copy-slot reuse wait.** Before input copies into a slot, the host waits for the previous graph that used that slot. Once one host wait on a backend in the current graph has proven the slot free, later copies into the same backend skip it.
- **Input order.** Inputs are copied in the graph order of their producers. On systems without peer-to-peer access, the driver stages peer copies into a device one at a time in submission order. Copying the earliest-ready tensor first keeps a small copy from queueing behind one whose source is still being computed.

Validation used a layer split over three NVIDIA GPUs on narrow PCIe links without peer access, with multi-token prediction. Prompt processing rose by 15-29% at `-ub 2048` and by up to 52% with several ubatches per batch. Generation speed and outputs were unchanged. Most of the ub2048 gain comes from the view alias, because each device hand-off moves half as many bytes. The two wait changes and the input order matter when a server batch holds several ubatches: switching all three back cost 25% of prompt throughput at `-ub 512`.

## Weight streaming (staging ring)

With routed experts in host memory (`-cmoe`, `-ncmoe`, `--numa tensors`), a wide prompt batch can run the expert matrix multiplications on the GPU: the scheduler copies each expert tensor to the device when it reaches the split that uses it (op offload). That copy runs in turn with the device's computation, and a copy from ordinary (pageable) host memory also blocks the scheduler thread at well below the PCIe rate.

| variable | default | effect |
| --- | --- | --- |
| `GGML_SCHED_STAGE` | `0` | `1` uploads offloaded host weights ahead of the splits that use them |
| `GGML_SCHED_STAGE_SLOTS` | `4` | device slots, each the size of the largest staged tensor; more slots run further ahead and use more VRAM |
| `GGML_SCHED_STAGE_BOUNCE` | `4` | pinned host bounce buffers; `0` uploads from the weights directly (pageable) |
| `GGML_SCHED_STAGE_CHUNK_MB` | `64` | size of each bounce buffer |
| `GGML_SCHED_STAGE_THREADS` | `8` | memcpy threads that fill the bounce buffers |

At the start of each graph, the scheduler lists the host weights that its splits will copy whole, in split order. A background thread copies them in chunks into pinned host buffers, and the device copies each chunk into a slot on a separate stream, so the uploads of later layers overlap the computation of earlier ones. When the scheduler reaches the split, the device copies the slot into the split input. Expert tensors are staged only when the batch is wide enough that nearly every expert is used; for narrower batches the scheduler keeps copying only the used experts, and decode is unaffected. Supported on CUDA and ROCm devices.

Each slot takes VRAM equal to the largest staged tensor (1.4 GiB for DeepSeek V4.1 Flash at Q2_K). If the slots cannot be allocated, staging turns itself off with a warning. Staging does not change what is computed, so outputs are identical with it on or off.

The offload cost model (or `GGML_OP_OFFLOAD_MIN_BATCH`) still decides whether the experts are offloaded at all, so staging matters only on a fast host link. Measured with DeepSeek V4.1 Flash (routed experts at Q2_K in host memory) on one Radeon AI PRO R9700 on PCIe 4.0 x16 at `-ub 2048`: a real-text prompt went from 84.5 to 134 tokens/s, random tokens (llama-bench pp4096) from 92 to 125 tokens/s, and decode was unchanged. On a PCIe 4.0 x2 link, the cost model keeps the experts on the CPU.

Every ubatch uploads all staged weights once, so the link sets a floor per token of (staged bytes) / (link rate × ubatch size): for DeepSeek V4.1 Flash, about 3.1 ms/token at `-ub 2048` and 1.6 ms/token at `-ub 4096`. A larger ubatch therefore pays off for long prompts, as long as its compute buffer fits. With the later kernel work on the same card, real-text prefill reached 255 tokens/s at `-ub 2048` and 484-489 tokens/s at `-ub 4096`. At `-ub 4096`, three slots ran as fast as four and used 1.4 GiB less VRAM (two slots lost 25%), so `GGML_SCHED_STAGE_SLOTS=3` is a good choice when VRAM is tight, for example next to a draft model. On a dual-socket EPYC 7532 host, `GGML_SCHED_STAGE_THREADS=4` was 7-8% faster than the default 8 at `-ub 4096` (504-508 against 469 tokens/s on an 11.4k-token prompt, two runs each); 2, 3 and 6 threads were no better than 8, so measure before changing it on other hosts.

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
