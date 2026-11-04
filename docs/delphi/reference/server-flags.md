# llama-server flag reference

This page lists the command-line flags that `llama-server` accepts, grouped by topic. Each entry gives the flag name, its environment variable if it has one, its default, what it does, and when you would change it. Flags marked (delphi.cpp) either exist only in delphi.cpp or behave differently from upstream llama.cpp, and each of those entries says what differs. Values are taken from the code on the `next` branch.

Most delphi.cpp features are on by default. You don't need to set most of the flags below. Read the defaults column first, and change a flag only when the table says a different value fits your case.

## Model loading and files

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `-m`, `--model` FNAME | `LLAMA_ARG_MODEL` | none | Loads the GGUF file at this path. | You serve a local model file. |
| `-mu`, `--model-url` MODEL_URL | `LLAMA_ARG_MODEL_URL` | unused | Downloads the model from this URL before loading it. | The model is not on disk yet. |
| `-dr`, `--docker-repo` [repo/]model[:quant] | `LLAMA_ARG_DOCKER_REPO` | unused | Downloads the model from Docker Hub. The repo defaults to `ai/` and the tag to `latest`. | You want a model from a Docker Hub repository. |
| `-hf`, `-hfr`, `--hf-repo` user/model[:quant] | `LLAMA_ARG_HF_REPO` | unused | Downloads the model from Hugging Face. The quant defaults to Q4_K_M, or to the first file in the repo if Q4_K_M is missing. The matching mmproj file is also downloaded when one exists. | You want a model from Hugging Face without managing the file yourself. |
| `-hff`, `--hf-file` FILE | `LLAMA_ARG_HF_FILE` | unused | Picks this file from the `--hf-repo` repository. It overrides the quant in the repo string. | The repo has several files and you want a specific one. |
| `-hft`, `--hf-token` TOKEN | `HF_TOKEN` | the `HF_TOKEN` value | Authenticates Hugging Face downloads. | The repo is private or gated. |
| `--offline` | `LLAMA_ARG_OFFLINE` | off | Uses only files already in the cache and makes no network requests. | The machine has no network access, or you want to be sure nothing is downloaded. |
| `-a`, `--alias` STRING | `LLAMA_ARG_ALIAS` | unset | Sets the model name or names that the API reports. Separate several names with commas. | Clients select the model by a name other than the file name. |
| `--tags` STRING | `LLAMA_ARG_TAGS` | none | Sets comma-separated tags that the server reports. The server does not use them for routing. | You want to label the model for your own tools. |
| `-lm`, `--load-mode` MODE | `LLAMA_ARG_LOAD_MODE` | `auto` | Sets how the model file is loaded. `auto` memory-maps the file unless a device cannot use it. `none` uses no special mode. `mmap` memory-maps the file. `mlock` keeps the model in RAM. `mmap+mlock` does both. `dio` uses direct I/O when the system supports it. | You need the model locked in RAM (`mlock`), or your storage is slow and you want direct I/O (`dio`). |
| `-lzm`, `--lazy-mode` MODE | `LLAMA_ARG_LAZY_MODE` | `auto` | Controls on-demand reading of large per-layer tensors. `on` reads their rows from disk when needed. `auto` does this only for tensors larger than 4 GiB. `off` keeps every tensor in memory. | Memory is tight and you accept slower steps, or you want all tensors resident (`off`). |
| `--repack`, `-nr`, `--no-repack` | `LLAMA_ARG_REPACK` | enabled | Rearranges weights into a layout that some backends run faster. | You want to skip the repacking step, for example to compare results. |
| `--no-host` | `LLAMA_ARG_NO_HOST` | off | Lets extra buffer types be used instead of the host buffer. | A backend needs this to load the model. |
| `--check-tensors` | none | off | Checks the model tensor data for invalid values while loading. | You suspect a damaged model file. Loading is slower with this on. |
| `--override-kv` KEY=TYPE:VALUE,... | none | none | Overrides model metadata keys at load time. Types are `int`, `float`, `bool` and `str`. | A model's metadata is wrong for your use, such as a missing or incorrect BOS setting. |
| `--lora` FNAME | none | none | Loads a LoRA adapter. Separate several adapters with commas. | You want to run the model with an adapter applied. |
| `--lora-scaled` FNAME:SCALE,... | none | none | Loads LoRA adapters with a scale factor for each. | You want an adapter applied at a strength other than 1.0. |
| `--lora-init-without-apply` | none | disabled | Loads the LoRA adapters but does not apply them. You apply them later with `POST /lora-adapters`. | You want to switch adapters at runtime. |
| `--control-vector` FNAME | none | none | Adds a control vector. Separate several with commas. | You use a control vector to steer the model. |
| `--control-vector-scaled` FNAME:SCALE,... | none | none | Adds control vectors with a scale factor for each. | You want a control vector at a scale other than 1.0. |
| `--control-vector-layer-range` START END | none | none | Sets the first and last layer, inclusive, to which the control vectors apply. | The control vector should affect only some layers. |
| `--op-offload`, `--no-op-offload` | none | enabled | Lets host tensor operations run on the device. | You want to keep all host operations on the CPU, for debugging. |

## GPU offload and devices

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `-dev`, `--device` LIST | `LLAMA_ARG_DEVICE` | not set | Lists the devices used for offloading. `none` turns offloading off. Use `--list-devices` to see the names. | You want to limit the model to certain GPUs, or keep it on the CPU with `none`. |
| `--list-devices` | none | none | Prints the available devices and exits. | You need the device names for `--device`, `--tensor-split` or `--main-gpu`. |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers` N | `LLAMA_ARG_N_GPU_LAYERS` | `auto` | Sets how many layers go to VRAM. The value is a number, `auto`, or `all`. | You want to fix the number of GPU layers, or force all of them onto the GPU. |
| `-sm`, `--split-mode` MODE | `LLAMA_ARG_SPLIT_MODE` | `layer` | Splits the model across GPUs. `none` uses one GPU. `layer` splits layers and KV cache across GPUs in a pipeline. `row` splits weights by rows. `tensor` splits weights and KV cache and is experimental. | You have several GPUs and want a different split than the layer split. |
| `-ts`, `--tensor-split` N0,N1,... | `LLAMA_ARG_TENSOR_SPLIT` | not set | Sets the share of the model that goes to each GPU, for example `3,1`. | The GPUs have different amounts of memory, or you want to balance them by hand. |
| `-mg`, `--main-gpu` INDEX | `LLAMA_ARG_MAIN_GPU` | 0 | Picks the GPU that holds the whole model with `--split-mode none`, or the intermediate results and KV cache with `--split-mode row`. | The main GPU is not the first device. |
| `-fit`, `--fit` [on, off] | `LLAMA_ARG_FIT` | on | Adjusts the arguments you did not set so that the model fits in device memory. | You set the memory layout yourself and want no automatic changes, so use `off`. |
| `-fitt`, `--fit-target` MiB0,MiB1,... | `LLAMA_ARG_FIT_TARGET` | 1024 | Sets the memory margin to leave free on each device when `--fit` is on. One value applies to all devices. | Other programs share the GPU and need more free memory. |
| `-fitc`, `--fit-ctx` N | `LLAMA_ARG_FIT_CTX` | 4096 | Sets the smallest context size that `--fit` may choose. | You need a larger minimum context. |
| `-ot`, `--override-tensor` PATTERN=BUFFER,... | `LLAMA_ARG_OVERRIDE_TENSOR` | none | Puts tensors whose names match a pattern into a chosen buffer type. | You want specific tensors on a particular memory type. |
| `-cmoe`, `--cpu-moe` (delphi.cpp) | `LLAMA_ARG_CPU_MOE` | off | Keeps all Mixture of Experts weights on the CPU. Unlike upstream llama.cpp, delphi.cpp then also copies the most-used experts into free GPU memory and runs them there, for models that support it; see the GPU expert cache section of [environment-variables.md](../usage/environment-variables.md). | The experts do not fit in VRAM. |
| `-ncmoe`, `--n-cpu-moe` N | `LLAMA_ARG_N_CPU_MOE` | off | Keeps the expert weights of the first N layers on the CPU. | You need to move part of the expert weights off the GPU. |
| `-ncffn`, `--n-cpu-ffn` N | `LLAMA_ARG_N_CPU_FFN` | off | Keeps the dense feed-forward weights of the first N layers on the CPU. This is for dense models. | A dense model does not fit, and you want to move some of its feed-forward weights to the CPU. |
| `--rpc` SERVERS | none | none | Adds remote RPC servers as devices. This flag appears only in builds that support RPC. | You use an RPC backend on another machine. |

> REVIEW: The default for `--device` when it is not set (all available devices) - could not verify beyond `common/common.cpp` (line 1748 checks `params.devices` but does not state the default) in `next`.
>
> REVIEW: `--rpc` is listed in `common/arg.cpp` only when `llama_supports_rpc()` is true, and the local build's `--help` does not show it - could not verify in `build/bin/llama-server --help`.

## CPU, threads and NUMA

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `-t`, `--threads` N | `LLAMA_ARG_THREADS` | -1 | Sets the number of CPU threads used during generation. | You want a fixed thread count. Do not set it above the number of cores you can spare. |
| `-tb`, `--threads-batch` N | none | same as `--threads` | Sets the number of threads used for prompt processing. | Prompt processing should use a different count than generation. |
| `-C`, `--cpu-mask` M | none | empty | Sets the CPU affinity mask as a hex string. It complements `--cpu-range`. | You need to pin the threads to specific cores. |
| `-Cr`, `--cpu-range` lo-hi | none | none | Sets a range of CPUs for affinity. | You want the threads kept within a range of cores. |
| `-Cb`, `--cpu-mask-batch` M | none | same as `--cpu-mask` | Sets the affinity mask for batch and prompt processing. | Prompt processing should run on other cores. |
| `-Crb`, `--cpu-range-batch` lo-hi | none | none | Sets a CPU range for batch and prompt processing. | As for `--cpu-range`, but for prompt processing. |
| `--cpu-strict` <0\|1> | none | 0 | Requires strict CPU placement when set to 1. | The threads must stay on the cores in the mask. |
| `--cpu-strict-batch` <0\|1> | none | same as `--cpu-strict` | Sets strict placement for batch and prompt processing. | As for `--cpu-strict`, but for prompt processing. |
| `--prio` N | none | 0 | Sets the process or thread priority. Values run from -1 (low) to 3 (realtime). | The server must compete for CPU time with other programs. |
| `--prio-batch` N | none | 0 | Sets the priority for batch and prompt processing. | As for `--prio`, but for prompt processing. |
| `--poll` <0...100> | none | 50 | Sets how long idle threads busy-wait for work. 0 disables polling. | You want lower latency at the cost of CPU use, or the reverse. |
| `--poll-batch` <0\|1> | none | same as `--poll` | Sets polling for batch and prompt processing. | As for `--poll`, but for prompt processing. |
| `--numa` TYPE (delphi.cpp) | `LLAMA_ARG_NUMA` | disabled | Turns on NUMA optimizations. `distribute` spreads work over all nodes, `isolate` keeps threads on the starting node, `numactl` uses the CPU map from numactl, and `tensors` splits each CPU expert's weights across the NUMA nodes. Unlike upstream llama.cpp, delphi.cpp adds the `tensors` value. Set `GGML_NUMA_TP_SHARES` to weight the nodes unevenly. | The host has several NUMA nodes and CPU experts are part of the model. Read the help text for the cache warning before you use it. |

## Context and KV cache

Context checkpoints and the prompt cache are in the first group. The other flags in this section set the context size, batch sizes and KV cache types.

### Context checkpoints and prompt cache

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `--ctx-checkpoints-eviction` {fifo,variance} (delphi.cpp) | `LLAMA_ARG_CTX_CHECKPOINTS_EVICTION` | `variance` | Chooses which checkpoint a full slot drops to make room. `fifo` drops the oldest. `variance` drops the one whose removal leaves the most even spacing. | You want to drop the oldest checkpoint instead (`fifo`). |
| `--ctx-checkpoints-pin-first`, `--no-ctx-checkpoints-pin-first` (delphi.cpp) | `LLAMA_ARG_CTX_CHECKPOINTS_PIN_FIRST` | enabled | Keeps the oldest checkpoint of a full slot, which usually marks the end of the system prompt. | You want every checkpoint to be evictable. |
| `--ctx-checkpoints-turn-step` N (delphi.cpp) | `LLAMA_ARG_CTX_CHECKPOINTS_TURN_STEP` | 4096 | For chat requests, creates checkpoints at the ends of messages, at least N tokens apart. 0 turns this off. It applies only while Jinja templates are on. | Long chats need more or fewer message-end checkpoints. |
| `--ctx-checkpoints-spill-dir` DIR (delphi.cpp) | `LLAMA_ARG_CTX_CHECKPOINTS_SPILL_DIR` | off | Writes the checkpoints that a full slot drops to files in this directory, and reads them back when a prompt needs one. The files stay until the server stops. Reads are skipped when they would be slower than processing the tokens again. | Checkpoints are evicted often and you have spare disk space on a fast disk. |
| `--ctx-checkpoints-spill-max-mib` N (delphi.cpp) | `LLAMA_ARG_CTX_CHECKPOINTS_SPILL_MAX_MIB` | 8192 | Limits the disk space in MiB that spilled checkpoints may use. | You need to cap disk use for spilled checkpoints. |
| `-ctxcp`, `--ctx-checkpoints`, `--swa-checkpoints` N | `LLAMA_ARG_CTX_CHECKPOINTS` | 32 | Sets the maximum number of context checkpoints per slot. | Long prompts reuse context poorly, or memory is tight. |
| `-cms`, `--checkpoint-min-step` N | `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT` | 8192 | Sets the minimum spacing in tokens between checkpoints. 0 means no minimum. | You want checkpoints closer together, or fewer of them. |
| `-cram`, `--cache-ram` N | `LLAMA_ARG_CACHE_RAM` | 8192 | Sets the maximum prompt cache size in MiB. -1 means no limit and 0 disables the cache. | Memory is tight (lower it), or you want more prompt reuse (raise it or set -1). |
| `--cache-prompt`, `--no-cache-prompt` | `LLAMA_ARG_CACHE_PROMPT` | enabled | Reuses the cached prompt of a previous request when the start matches. | You want every request processed from scratch, for testing. |
| `--cache-reuse` N | `LLAMA_ARG_CACHE_REUSE` | 0 | Sets the minimum chunk size that may be reused from the cache by shifting the KV cache. It needs prompt caching on. | Prompts often share a middle section that is not at the start. |
| `--cache-idle-slots`, `--no-cache-idle-slots` | `LLAMA_ARG_CACHE_IDLE_SLOTS` | enabled | Saves an idle slot to the prompt cache when a new task starts, and clears it when unified KV is on. It needs `--cache-ram` above 0. | You need a slot's state kept in place, or you want to free it. |

### Context size, batches and KV cache

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `-c`, `--ctx-size` N | `LLAMA_ARG_CTX_SIZE` | 0 | Sets the prompt context size in tokens. 0 uses the size from the model. | You need a context smaller or larger than the model default. |
| `-n`, `--predict`, `--n-predict` N | `LLAMA_ARG_N_PREDICT` | -1 | Sets the number of tokens to generate. -1 means no limit. | You want a hard cap on output length. |
| `-b`, `--batch-size` N | `LLAMA_ARG_BATCH` | 2048 | Sets the logical maximum batch size. | You want to change how many tokens go into one batch. |
| `-ub`, `--ubatch-size` N | `LLAMA_ARG_UBATCH` | 512 | Sets the physical maximum batch size. | You run out of memory during prompt processing, so lower it. |
| `--keep` N | none | 0 | Sets how many tokens from the start of the prompt are kept. -1 keeps all of them. | You want part of the start of the prompt to stay when the context is trimmed. |
| `-fa`, `--flash-attn` [on, off, auto] | `LLAMA_ARG_FLASH_ATTN` | `auto` | Sets Flash Attention use. | You want to force it on or off, for example to isolate a problem. |
| `-kvo`, `--kv-offload`, `-nkvo`, `--no-kv-offload` | `LLAMA_ARG_KV_OFFLOAD` | enabled | Stores the KV cache on the device. | The KV cache must stay in host memory. |
| `-ctk`, `--cache-type-k` TYPE | `LLAMA_ARG_CACHE_TYPE_K` | `f16` | Sets the data type of the K cache. Values are f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0 and q5_1. | You need a smaller KV cache, at some cost in quality. |
| `-ctv`, `--cache-type-v` TYPE | `LLAMA_ARG_CACHE_TYPE_V` | `f16` | Sets the data type of the V cache. The values are the same as for `-ctk`. | As for `-ctk`. |
| `-kvu`, `--kv-unified`, `-no-kvu`, `--no-kv-unified` | `LLAMA_ARG_KV_UNIFIED` | enabled when slots are automatic | Uses one KV buffer shared by all sequences. | You want a separate buffer per slot. |
| `--kv-unified-per-slot` N | `LLAMA_ARG_KV_UNIFIED_PER_SLOT` | unset | Sets the context limit per slot. When `-c` is not set, the shared KV pool is sized to the slot count times N. | You want each slot to have a fixed share of one shared pool. |
| `--swa-full` | `LLAMA_ARG_SWA_FULL` | false | Uses a full-size cache for sliding-window attention. | A model with sliding-window attention needs to see its whole context. |
| `--context-shift`, `--no-context-shift` | `LLAMA_ARG_CONTEXT_SHIFT` | disabled | Shifts the context when generation runs past the window. | You generate endless text and accept dropped older tokens. |
| `--perf`, `--no-perf` | `LLAMA_ARG_PERF` | false | Turns on the internal library performance timings. | You need timing data from the library. |
| `--rope-scaling` {none,linear,yarn} | `LLAMA_ARG_ROPE_SCALING_TYPE` | model default | Sets the RoPE scaling method. | You extend the context beyond the model's trained length. |
| `--rope-scale` N | `LLAMA_ARG_ROPE_SCALE` | model default | Sets the RoPE context scaling factor. | As for `--rope-scaling`. |
| `--rope-freq-base` N | `LLAMA_ARG_ROPE_FREQ_BASE` | model default | Sets the RoPE base frequency. | You use NTK-aware scaling. |
| `--rope-freq-scale` N | `LLAMA_ARG_ROPE_FREQ_SCALE` | model default | Sets the RoPE frequency scaling factor. | As for `--rope-scaling`. |
| `--yarn-orig-ctx` N | `LLAMA_ARG_YARN_ORIG_CTX` | 0 | Sets the original context size for YaRN. 0 uses the model's training context. | You use YaRN and the model's metadata is wrong. |
| `--yarn-ext-factor` N | `LLAMA_ARG_YARN_EXT_FACTOR` | -1.00 | Sets the YaRN extrapolation mix factor. 0.0 means full interpolation. | You tune YaRN scaling by hand. |
| `--yarn-attn-factor` N | `LLAMA_ARG_YARN_ATTN_FACTOR` | -1.00 | Sets the YaRN attention scale. | As for `--yarn-ext-factor`. |
| `--yarn-beta-slow` N | `LLAMA_ARG_YARN_BETA_SLOW` | -1.00 | Sets the YaRN high correction dimension. | As for `--yarn-ext-factor`. |
| `--yarn-beta-fast` N | `LLAMA_ARG_YARN_BETA_FAST` | -1.00 | Sets the YaRN low correction dimension. | As for `--yarn-ext-factor`. |

## Batching and slots

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `-np`, `--parallel` N | `LLAMA_ARG_N_PARALLEL` | -1 (automatic) | Sets the number of server slots, which is the number of requests handled at once. | You serve several clients at once, or you want one slot only. |
| `-cb`, `--cont-batching`, `-nocb`, `--no-cont-batching` | `LLAMA_ARG_CONT_BATCHING` | enabled | Mixes new requests into batches while others are still generating. | You want requests processed one batch at a time, for debugging. |
| `-sps`, `--slot-prompt-similarity` N | none | 0.10 | Sets how closely a request's prompt must match a slot's prompt before that slot is used. 0.0 disables the check. | Requests reuse the wrong slot, or you want more reuse. |
| `--warmup`, `--no-warmup` | none | enabled | Runs an empty batch at start-up. | You want faster start-up and accept a slower first request. |
| `--sleep-idle-seconds` SECONDS | none | -1 | Puts the server to sleep after this many idle seconds. -1 disables sleep. | The server is idle for long periods and you want to save resources. |

## Sampling

These flags set the defaults for generated text.

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `--samplers` LIST | none | `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature` | Sets the sampler chain, separated by semicolons. | You want a different order or set of samplers. |
| `-s`, `--seed` SEED | none | -1 | Sets the random seed. -1 picks a random seed. | You need repeatable output. |
| `--sampler-seq`, `--sampling-seq` SEQUENCE | none | `edskypmxt` | Sets the sampler chain with a short code. | You prefer the short form. |
| `--ignore-eos` | none | off | Keeps generating past the end-of-stream token. | You want generation to continue after the model would stop. |
| `--temp`, `--temperature` N | none | 0.80 | Sets the sampling temperature. | Lower values make output more predictable, and higher values make it more varied. |
| `--top-k` N | `LLAMA_ARG_TOP_K` | 40 | Keeps only the N most likely tokens. 0 disables it. | You want to narrow or widen the candidates. |
| `--top-p` N | none | 0.95 | Keeps the smallest set of tokens whose probabilities add up to P. 1.0 disables it. | As for `--top-k`. |
| `--min-p` N | none | 0.05 | Drops tokens less likely than this fraction of the top token. 0.0 disables it. | You want to cut low-probability tokens. |
| `--top-nsigma`, `--top-n-sigma` N | none | -1.00 | Keeps tokens within N standard deviations of the top logit. -1 disables it. | You use this sampler on purpose. |
| `--xtc-probability` N | none | 0.00 | Sets the chance that XTC removes the top tokens. 0.0 disables it. | You want XTC sampling. |
| `--xtc-threshold` N | none | 0.10 | Sets the probability threshold for XTC. 1.0 disables it. | As for `--xtc-probability`. |
| `--typical`, `--typical-p` N | none | 1.00 | Sets locally typical sampling. 1.0 disables it. | You want typical sampling. |
| `--repeat-last-n` N | none | 64 | Sets how many recent tokens the penalties look at. 0 disables them. | Repetition is a problem, so raise it, or lower it to allow more repeats. |
| `--repeat-penalty` N | none | 1.00 | Penalizes tokens that repeat. 1.0 disables the penalty. | Output repeats itself. |
| `--presence-penalty` N | none | 0.00 | Penalizes any token that has already appeared. | You want fewer reused words. |
| `--frequency-penalty` N | none | 0.00 | Penalizes tokens in proportion to how often they appear. | As for `--presence-penalty`. |
| `--dry-multiplier` N | none | 0.00 | Sets the DRY penalty strength. 0.0 disables DRY. | You want DRY to reduce repeated sequences. |
| `--dry-base` N | none | 1.75 | Sets the base of the DRY penalty. | You tune DRY. |
| `--dry-allowed-length` N | none | 2 | Sets how long a repeated sequence can be before DRY applies. | As for `--dry-base`. |
| `--dry-penalty-last-n` N | none | 64 | Sets how many recent tokens DRY checks. 0 disables it. | As for `--dry-base`. |
| `--dry-sequence-breaker` STRING | none | `\n`, `:`, `"`, `*` | Adds a sequence breaker for DRY, which replaces the default breakers. Use `none` to have no breakers. | Your output contains punctuation that DRY should treat as a break. |
| `--adaptive-target` N | none | -1.00 | Sets the target probability for adaptive-p. A negative value disables it. | You use adaptive-p. |
| `--adaptive-decay` N | none | 0.90 | Sets how fast the adaptive-p target changes. Lower values react faster. | As for `--adaptive-target`. |
| `--dynatemp-range` N | none | 0.00 | Sets the dynamic temperature range. 0.0 disables it. | You want the temperature to vary with the model's uncertainty. |
| `--dynatemp-exp` N | none | 1.00 | Sets the exponent for dynamic temperature. | As for `--dynatemp-range`. |
| `--mirostat` N | none | 0 | Turns on Mirostat sampling. 1 is Mirostat and 2 is Mirostat 2.0. Top-k, top-p and typical sampling are ignored when it is on. | You want Mirostat to hold perplexity near a target. |
| `--mirostat-lr` N | none | 0.10 | Sets the Mirostat learning rate. | You tune Mirostat. |
| `--mirostat-ent` N | none | 5.00 | Sets the Mirostat target entropy. | As for `--mirostat-lr`. |
| `-l`, `--logit-bias` TOKEN_ID(+/-)BIAS | none | none | Raises or lowers the likelihood of one token. | You want to push the model away from or toward a token. |
| `--grammar` GRAMMAR | none | none | Constrains output with a BNF-style grammar. | You need output in a fixed format. |
| `--grammar-file` FNAME | none | none | Reads the grammar from a file. | The grammar is long, or you keep it in a file. |
| `-j`, `--json-schema` SCHEMA | none | none | Constrains output to a JSON schema. | You need JSON output that follows a schema. |
| `-jf`, `--json-schema-file` FILE | none | none | Reads the JSON schema from a file. | As for `--json-schema`. |
| `-bs`, `--backend-sampling` | `LLAMA_ARG_BACKEND_SAMPLING` | disabled | Runs sampling on the backend. This is experimental. | You want to test backend sampling. |

## Speculative decoding

Speculative decoding drafts several tokens and checks them in one step. Set `--spec-type` to choose a method. The flags below tune the method you pick. Adaptive drafting is on by default for `draft-mtp` when no synthetic acceptance is set.

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `--spec-draft-n-max` N (delphi.cpp) | `LLAMA_ARG_SPEC_DRAFT_N_MAX` | 3, or 7 in the default conf mode of `draft-mtp` | Sets the largest number of tokens drafted on each step. Unlike upstream llama.cpp, the default is 7 when `draft-mtp` runs in its conf mode and you do not set this flag. | You want a deeper or shallower draft. |
| `--spec-draft-adaptive`, `--no-spec-draft-adaptive` (delphi.cpp) | `LLAMA_ARG_SPEC_DRAFT_ADAPTIVE` | on for `draft-mtp` in conf mode without synthetic acceptance, otherwise off | Changes the draft depth with the acceptance rate, between `--spec-draft-n-min-adaptive` and `--spec-draft-n-max`. `--no-spec-draft-adaptive` keeps a fixed depth. | You want a fixed draft depth, so use `--no-spec-draft-adaptive`. |
| `--spec-draft-adaptive-mode` {bucket,cost,conf} (delphi.cpp) | `LLAMA_ARG_SPEC_DRAFT_ADAPTIVE_MODE` | `conf` for `draft-mtp`, otherwise `bucket` | Picks the controller for adaptive drafting. `bucket` follows the acceptance rate. `cost` picks the depth with the most tokens per second. `conf` stops each draft when the draft's confidence no longer pays for the check. Setting this flag also turns adaptive drafting on. | You want `bucket` or `cost` instead of the default. |
| `--spec-draft-n-min-adaptive` N (delphi.cpp) | `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE` | 0, which means 3 for `bucket` and 1 for `cost` and `conf` | Sets the smallest depth that adaptive drafting may use. | You want adaptive drafting to stay at least at some depth. |
| `--spec-draft-n-start` N (delphi.cpp) | `LLAMA_ARG_SPEC_DRAFT_N_START` | 0 | Sets the adaptive draft depth at the start of each generation. 0 uses a mode-specific start. | You know a better starting depth for your prompts. |
| `--spec-draft-block` N (delphi.cpp) | `LLAMA_ARG_SPEC_DRAFT_BLOCK` | 0, which means the trained block size | Sets the number of tokens in one drafted block for `draft-dspark`. The first `--spec-draft-n-max` tokens are checked, and the rest only condition the draft. | You use `draft-dspark` and want a different block size. |
| `--spec-draft-sampling` {greedy,probabilistic} (delphi.cpp) | `LLAMA_ARG_SPEC_DRAFT_SAMPLING` | `greedy` | Sets how the draft picks its tokens. `greedy` takes the most likely token. `probabilistic` samples the token and has the target check it by rejection sampling. | You want the draft to sample instead of taking the top token. |
| `--spec-draft-ubatch-size`, `-ubd` N (delphi.cpp) | `LLAMA_ARG_SPEC_DRAFT_UBATCH_SIZE` | 128 | Sets the physical batch size of the draft context. 0 uses the same value as `-ub`. A draft seldom gains from a large batch, and a large one uses more memory. | The draft context uses too much memory, or you want to tune its batch. |
| `--spec-draft-backend-sampling`, `--no-spec-draft-backend-sampling` | `LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING` | enabled | Runs draft sampling on the backend. | You need draft sampling on the CPU. |
| `--spec-type` LIST | `LLAMA_ARG_SPEC_TYPE` | none | Chooses the speculative methods, as a comma-separated list. Values are `none`, `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod` and `ngram-cache`. | You want speculative decoding at all. Choose a method your model supports. |
| `--spec-default` | none | off | Turns on the n-gram method `ngram-mod` with its default settings. | You want a quick speculative setup with no draft model. |
| `--spec-synth-len` L | `LLAMA_ARG_SPEC_SYNTH_LEN` | -1, which is off | Sets a synthetic target mean acceptance length. It is for benchmarking only. | You measure speculative behavior, not for normal serving. |
| `--spec-synth-rates` P0,P1,... | `LLAMA_ARG_SPEC_SYNTH_RATES` | not set | Sets synthetic per-position acceptance probabilities. It is for benchmarking only. | As for `--spec-synth-len`. |
| `--spec-draft-p-split`, `--draft-p-split` P | `LLAMA_ARG_SPEC_DRAFT_P_SPLIT` | 0.10 | Sets the split probability for speculative decoding. | You tune the split. |
| `--spec-draft-p-min`, `--draft-p-min` P | `LLAMA_ARG_SPEC_DRAFT_P_MIN` | 0.00 | Sets the minimum probability for a drafted token in greedy mode. | You want to drop low-probability draft tokens. |
| `--spec-draft-n-min` N | `LLAMA_ARG_SPEC_DRAFT_N_MIN` | 0 | Sets the minimum number of draft tokens per step. | You want every step to draft at least some tokens. |
| `--spec-draft-model`, `-md`, `--model-draft` FNAME | `LLAMA_ARG_SPEC_DRAFT_MODEL` | unused | Loads the draft model from this file. | You use a draft-model method such as `draft-simple`. |
| `--spec-draft-hf`, `-hfd`, `-hfrd`, `--hf-repo-draft` user/model[:quant] | `LLAMA_ARG_SPEC_DRAFT_HF_REPO` | unused | Downloads the draft model from Hugging Face, like `--hf-repo`. | As for `--spec-draft-model`, when you want the download. |
| `--spec-draft-ngl`, `-ngld`, `--gpu-layers-draft` N | `LLAMA_ARG_N_GPU_LAYERS_DRAFT` | `auto` | Sets how many draft model layers go to VRAM. | The draft model should use the GPU or the CPU differently from the main model. |
| `--spec-draft-device`, `-devd`, `--device-draft` LIST | none | follows `--device` | Sets the devices for the draft model. `none` keeps it on the CPU. | The draft model should run on other devices. |
| `--spec-draft-override-tensor`, `-otd`, `--override-tensor-draft` PATTERN=BUFFER,... | none | none | Places draft model tensors in buffer types, like `--override-tensor`. | As for `--override-tensor`, for the draft model. |
| `--spec-draft-cpu-moe`, `-cmoed`, `--cpu-moe-draft` | `LLAMA_ARG_SPEC_DRAFT_CPU_MOE` | off | Keeps all draft model expert weights on the CPU. | The draft model's experts do not fit in VRAM. |
| `--spec-draft-n-cpu-moe`, `--spec-draft-ncmoe`, `-ncmoed`, `--n-cpu-moe-draft` N | `LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE` | off | Keeps the expert weights of the first N draft layers on the CPU. | As for `--spec-draft-cpu-moe`, for part of the model. |
| `-td`, `--spec-draft-threads`, `--threads-draft` N | none | same as `--threads` | Sets the CPU threads for draft generation. | The draft runs on the CPU and needs its own thread count. |
| `-tbd`, `--spec-draft-threads-batch`, `--threads-batch-draft` N | none | same as `--threads-draft` | Sets the draft threads for prompt processing. | As for `--spec-draft-threads`, for prompt processing. |
| `-Cd`, `--spec-draft-cpu-mask`, `--cpu-mask-draft` M | none | same as `--cpu-mask` | Sets the affinity mask for the draft model. | The draft needs its own cores. |
| `-Crd`, `--spec-draft-cpu-range`, `--cpu-range-draft` lo-hi | none | none | Sets a CPU range for the draft model. | As for `--spec-draft-cpu-mask`, as a range. |
| `--spec-draft-cpu-strict`, `--cpu-strict-draft` <0\|1> | none | same as `--cpu-strict` | Sets strict CPU placement for the draft model. | As for `--cpu-strict`, for the draft. |
| `--spec-draft-prio`, `--prio-draft` N | none | 0 | Sets the draft process or thread priority. | As for `--prio`, for the draft. |
| `--spec-draft-poll`, `--poll-draft` <0\|1> | none | same as `--poll` | Sets polling for draft work. | As for `--poll`, for the draft. |
| `-Cbd`, `--spec-draft-cpu-mask-batch`, `--cpu-mask-batch-draft` M | none | same as `--cpu-mask` | Sets the batch affinity mask for the draft model. | As for `--spec-draft-cpu-mask`, for prompt processing. |
| `--spec-draft-cpu-strict-batch`, `--cpu-strict-batch-draft` <0\|1> | none | same as `--cpu-strict-draft` | Sets strict placement for draft prompt processing. | As for `--cpu-strict-draft`, for prompt processing. |
| `--spec-draft-prio-batch`, `--prio-batch-draft` N | none | 0 | Sets the draft priority for prompt processing. | As for `--prio-draft`, for prompt processing. |
| `--spec-draft-poll-batch`, `--poll-batch-draft` <0\|1> | none | same as `--poll-draft` | Sets draft polling for prompt processing. | As for `--poll-draft`, for prompt processing. |
| `--spec-draft-type-k`, `-ctkd`, `--cache-type-k-draft` TYPE | `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K` | `f16` | Sets the K cache data type for the draft model. | You need a smaller draft KV cache. |
| `--spec-draft-type-v`, `-ctvd`, `--cache-type-v-draft` TYPE | `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V` | `f16` | Sets the V cache data type for the draft model. | As for `--spec-draft-type-k`. |
| `--spec-ngram-mod-n-min` N | none | 48 | Sets the minimum n-gram length for `ngram-mod`. | You tune `ngram-mod`. |
| `--spec-ngram-mod-n-max` N | none | 64 | Sets the maximum n-gram length for `ngram-mod`. | As for `--spec-ngram-mod-n-min`. |
| `--spec-ngram-mod-n-match` N | none | 24 | Sets the lookup length for `ngram-mod`. | As for `--spec-ngram-mod-n-min`. |
| `--spec-ngram-simple-size-n` N | none | 12 | Sets the lookup n-gram length for `ngram-simple`. | You tune `ngram-simple`. |
| `--spec-ngram-simple-size-m` N | none | 48 | Sets the draft m-gram length for `ngram-simple`. | As for `--spec-ngram-simple-size-n`. |
| `--spec-ngram-simple-min-hits` N | none | 1 | Sets the minimum hits for `ngram-simple`. | As for `--spec-ngram-simple-size-n`. |
| `--spec-ngram-map-k-size-n` N | none | 12 | Sets the lookup n-gram length for `ngram-map-k`. | You tune `ngram-map-k`. |
| `--spec-ngram-map-k-size-m` N | none | 48 | Sets the draft m-gram length for `ngram-map-k`. | As for `--spec-ngram-map-k-size-n`. |
| `--spec-ngram-map-k-min-hits` N | none | 1 | Sets the minimum hits for `ngram-map-k`. | As for `--spec-ngram-map-k-size-n`. |
| `--spec-ngram-map-k4v-size-n` N | none | 12 | Sets the lookup n-gram length for `ngram-map-k4v`. | You tune `ngram-map-k4v`. |
| `--spec-ngram-map-k4v-size-m` N | none | 48 | Sets the draft m-gram length for `ngram-map-k4v`. | As for `--spec-ngram-map-k4v-size-n`. |
| `--spec-ngram-map-k4v-min-hits` N | none | 1 | Sets the minimum hits for `ngram-map-k4v`. | As for `--spec-ngram-map-k4v-size-n`. |

## Chat templates and reasoning

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `--jinja`, `--no-jinja` | `LLAMA_ARG_JINJA` | enabled | Uses the Jinja engine to format chat messages. | You want to turn Jinja off with `--no-jinja`. |
| `--chat-template` NAME_OR_TEMPLATE | `LLAMA_ARG_CHAT_TEMPLATE` | the model's template | Sets a custom chat template. Only the built-in names are accepted unless `--jinja` comes first. The built-in names include `chatml`, `llama3`, `gemma`, `deepseek3` and `gpt-oss`. | The model's template is wrong or missing. |
| `--chat-template-file` FILE | `LLAMA_ARG_CHAT_TEMPLATE_FILE` | the model's template | Sets a custom chat template from a file. The same name rules apply. | As for `--chat-template`, with the template in a file. |
| `--chat-template-kwargs` JSON | `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` | none | Passes extra values to the template parser. The value must be a JSON object. | A template needs a setting, such as enabling thinking. |
| `--reasoning-format` FORMAT | `LLAMA_ARG_THINK` | `auto` | Sets how thinking tags are returned. `none` leaves them in the content. `deepseek` moves them to `reasoning_content`. `deepseek-legacy` keeps the tags and also fills `reasoning_content`. `auto` behaves like `deepseek`. | Clients expect thinking in the content, so use `none` or `deepseek-legacy`. |
| `-rea`, `--reasoning` [on, off, auto] | `LLAMA_ARG_REASONING` | `auto` | Turns thinking on or off. `auto` follows the chat template. | You want to force thinking on or off for every request. |
| `--reasoning-effort` LEVEL | `LLAMA_ARG_REASONING_EFFORT` | `default` | Passes a reasoning effort level to the template. Use `default` to keep the template's own level. | The template supports effort levels and you want one set for all requests. |
| `--reasoning-budget` N | `LLAMA_ARG_THINK_BUDGET` | -1 | Sets the token budget for thinking. -1 is unlimited, and 0 ends thinking at once. | You want to cap how long the model thinks. |
| `--reasoning-budget-message` MESSAGE | `LLAMA_ARG_THINK_BUDGET_MESSAGE` | none | Inserts this message before the end-of-thinking tag when the budget runs out. | You want the model to be told that its budget ended. |
| `--reasoning-preserve`, `--no-reasoning-preserve` | `LLAMA_ARG_REASONING_PRESERVE` | enabled | Keeps the thinking traces in the full history, not only in the last assistant message. This works with templates that support it. | Your template does not support it, or you want history without thinking. |
| `--skip-chat-parsing`, `--no-skip-chat-parsing` | `LLAMA_ARG_SKIP_CHAT_PARSING` | disabled | Uses a plain content parser even when a Jinja template is set. Reasoning and tool calls then stay in the content. | Parsing the template output causes errors and you want raw content. |
| `--prefill-assistant`, `--no-prefill-assistant` | `LLAMA_ARG_PREFILL_ASSISTANT` | enabled | Continues a final assistant message as a prefill. With `--no-prefill-assistant`, the final assistant message is treated as complete. | You send a complete assistant message and want it closed. |

## Multimodal (vision, audio and video)

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `-mm`, `--mmproj` FILE | `LLAMA_ARG_MMPROJ` | none | Loads the multimodal projector file. With `-hf`, the file is found automatically. | You serve a vision or multimodal model from a local file. |
| `-mmu`, `--mmproj-url` URL | `LLAMA_ARG_MMPROJ_URL` | unused | Downloads the projector from this URL. | The projector is not on disk yet. |
| `--mmproj-auto`, `--no-mmproj`, `--no-mmproj-auto` | `LLAMA_ARG_MMPROJ_AUTO` | enabled | Uses the projector when one is available, such as with `-hf`. `--no-mmproj` turns it off. | You want text-only serving from a model that has a projector. |
| `--mmproj-offload`, `--no-mmproj-offload` | `LLAMA_ARG_MMPROJ_OFFLOAD` | enabled | Runs the projector on the GPU. | The projector should stay on the CPU. |
| `-mmdev`, `--mmproj-device` DEVICE | `MTMD_BACKEND_DEVICE` | follows `--device` | Sets the device for the projector. `none` keeps it on the CPU. | The projector needs a different device than the model. |
| `--image-min-tokens` N | `LLAMA_ARG_IMAGE_MIN_TOKENS` | model value | Sets the minimum tokens per image for models with dynamic resolution. | Small images are not seen clearly. |
| `--image-max-tokens` N | `LLAMA_ARG_IMAGE_MAX_TOKENS` | model value | Sets the maximum tokens per image for models with dynamic resolution. | Large images use too much context. |
| `--mtmd-batch-max-tokens` N | `LLAMA_ARG_MTMD_BATCH_MAX_TOKENS` | 1024 | Sets the maximum image tokens per batch when images are encoded. | Image encoding runs out of memory, so lower it. |
| `--video-fps` N | `LLAMA_ARG_VIDEO_FPS` | 4.0 | Sets the frame rate for video input. | You need more or fewer frames per second. |
| `--video-timestamp-interval` N | `LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL` | 5000 | Sets the interval in milliseconds between text timestamps in video. | You want timestamps more or less often. |
| `--video-ffmpeg-dir` DIR | `LLAMA_ARG_VIDEO_FFMPEG_DIR` | search in PATH | Sets the folder that holds `ffmpeg` and `ffprobe`. | The tools are not on PATH. |

## Server and HTTP

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `--host` HOST | `LLAMA_ARG_HOST` | 127.0.0.1 | Sets the address to listen on. A value that ends in `.sock` opens a Unix socket. | You want the server reachable from other machines, so use `0.0.0.0`. |
| `--port` PORT | `LLAMA_ARG_PORT` | 8080 | Sets the port to listen on. | The port is in use or you need a different one. |
| `--reuse-port` | `LLAMA_ARG_REUSE_PORT` | disabled | Lets several sockets bind to the same port. | You run several processes behind one port. |
| `--path` PATH | `LLAMA_ARG_STATIC_PATH` | empty | Serves static files from this folder. | You serve your own web front end. |
| `--api-prefix` PREFIX | `LLAMA_ARG_API_PREFIX` | empty | Sets a path prefix for all routes, without a trailing slash. | The server runs behind a reverse proxy at a sub-path. |
| `--timeout`, `-to` N | `LLAMA_ARG_TIMEOUT` | 3600 | Sets the read and write timeout in seconds. | Long generations are cut off, so raise it. |
| `--sse-ping-interval` N | `LLAMA_ARG_SSE_PING_INTERVAL` | 30 | Sets the interval in seconds for keep-alive pings on streams. -1 turns them off. | A proxy closes idle streams. |
| `--threads-http` N | `LLAMA_ARG_THREADS_HTTP` | -1 | Sets the number of threads that process HTTP requests. | Many clients connect at once. |
| `--cors-origins` ORIGINS | `LLAMA_ARG_CORS_ORIGINS` | `*` | Sets the origins allowed by CORS. `localhost` allows only localhost origins. | A browser page on another origin must call the server. |
| `--cors-methods` METHODS | `LLAMA_ARG_CORS_METHODS` | GET, POST, DELETE, OPTIONS | Sets the HTTP methods allowed by CORS. | As for `--cors-origins`. |
| `--cors-headers` HEADERS | `LLAMA_ARG_CORS_HEADERS` | `*` | Sets the headers allowed by CORS. | As for `--cors-origins`. |
| `--cors-credentials`, `--no-cors-credentials` | `LLAMA_ARG_CORS_CREDENTIALS` | enabled | Allows credentials in CORS requests. | A browser must not send cookies or auth headers cross-origin. |
| `--api-key` KEY | `LLAMA_API_KEY` | none | Requires this key on requests. Separate several keys with commas. | The server is reachable by others. |
| `--api-key-file` FNAME | `LLAMA_ARG_API_KEY_FILE` | none | Reads keys from a file, one per line. Lines starting with `#` are comments. | You manage several keys in a file. |
| `--ssl-key-file` FNAME | `LLAMA_ARG_SSL_KEY_FILE` | none | Sets the PEM private key for HTTPS. | You serve HTTPS. |
| `--ssl-cert-file` FNAME | `LLAMA_ARG_SSL_CERT_FILE` | none | Sets the PEM certificate for HTTPS. | As for `--ssl-key-file`. |
| `--metrics` | `LLAMA_ARG_ENDPOINT_METRICS` | disabled | Enables the Prometheus-compatible metrics endpoint. | You monitor the server. |
| `--props` | `LLAMA_ARG_ENDPOINT_PROPS` | disabled | Allows changing global properties with `POST /props`. | You change settings at runtime from a trusted client. |
| `--tune-endpoints` (delphi.cpp) | `LLAMA_ARG_TUNE_ENDPOINTS` | disabled | Enables the auto-tuner endpoints `POST` and `GET /tune/context` and `POST /tune/kld`. | You use the auto-tuner. |
| `--slots`, `--no-slots` | `LLAMA_ARG_ENDPOINT_SLOTS` | enabled | Exposes the slot monitoring endpoint. | You do not want slot details visible. |
| `--slot-save-path` PATH | none | disabled | Sets the folder where slot KV caches are saved. | You save and restore slot state. |
| `--media-path` PATH | none | disabled | Sets a folder for local media files. Files are then read by `file://` URL with paths relative to it. | Clients send media that is already on the server. |
| `--embedding`, `--embeddings` | `LLAMA_ARG_EMBEDDINGS` | disabled | Restricts the server to embeddings. Use it only with an embedding model. | You serve an embedding model. |
| `--embd-normalize` N | none | 2 | Sets the normalization for embeddings. -1 is none, 0 is max absolute int16, 1 is taxicab, 2 is Euclidean, and larger values are p-norms. | Your retrieval system expects a different normalization. |
| `--pooling` {none,mean,cls,last,rank} | `LLAMA_ARG_POOLING` | model default | Sets the pooling type for embeddings. | The model's pooling is wrong for your use. |
| `--rerank`, `--reranking` | `LLAMA_ARG_RERANKING` | disabled | Enables the reranking endpoint. | You serve a reranker model. |
| `--ui`, `--webui`, `--no-ui`, `--no-webui` | `LLAMA_ARG_UI` | enabled | Turns the built-in web UI on or off. | You want an API-only server. |
| `--ui-config`, `--webui-config` JSON | `LLAMA_ARG_UI_CONFIG` | none | Sets default UI settings as a JSON object. | You want different UI defaults. |
| `--ui-config-file`, `--webui-config-file` PATH | `LLAMA_ARG_UI_CONFIG_FILE` | none | Sets default UI settings from a JSON file. | As for `--ui-config`, from a file. |
| `--ui-mcp-proxy`, `--webui-mcp-proxy`, `--no-ui-mcp-proxy`, `--no-webui-mcp-proxy` | `LLAMA_ARG_UI_MCP_PROXY` | disabled | Enables a CORS proxy for MCP in the UI. This is experimental. Do not enable it in untrusted environments. | You use MCP servers from the web UI. |
| `--tools` LIST | `LLAMA_ARG_TOOLS` | none | Enables built-in tools for agents. The names are `read_file`, `file_glob_search`, `grep_search`, `exec_shell_command`, `write_file`, `edit_file` and `get_info`. `all` enables every tool. This is experimental. | You run an agent that needs local tools. Do not enable it in untrusted environments. |
| `--tools-runtime` OPTION | `LLAMA_ARG_TOOLS_RUNTIME` | none, which uses the host | Runs tools in a container or on a remote host. The options are `docker:<image>`, `podman:<image>`, `docker-container:<id>`, `podman-container:<id>`, and `ssh:<target>`. This is experimental. | You want tools isolated from the host. |
| `--mcp-servers-config` PATH | `LLAMA_ARG_MCP_SERVERS_CONFIG` | none | Loads MCP server definitions from a JSON file in Cursor's format. This is experimental. | You use MCP servers with the server's tools. |
| `--mcp-servers-json` JSON | `LLAMA_ARG_MCP_SERVERS_JSON` | none | Gives MCP server definitions inline as JSON. This is experimental. | As for `--mcp-servers-config`, inline. |
| `-ag`, `--agent`, `-no-ag`, `--no-agent` | `LLAMA_ARG_AGENT` | disabled | Enables the CORS proxy and all built-in tools together. This is experimental. | You want the full agent setup. Do not enable it in untrusted environments. |

### Router mode

These flags apply when the server runs as a router that loads several models.

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `--models-dir` PATH | `LLAMA_ARG_MODELS_DIR` | disabled | Sets the folder with the models the router can serve. | You serve several models from one router. |
| `--models-preset` PATH | `LLAMA_ARG_MODELS_PRESET` | disabled | Reads model presets from an INI file. | You define presets for the router's models. |
| `--models-max` N | `LLAMA_ARG_MODELS_MAX` | 4 | Sets how many models may be loaded at once. 0 means no limit. | Memory allows more or fewer models at once. |
| `--models-autoload`, `--no-models-autoload` | `LLAMA_ARG_MODELS_AUTOLOAD` | enabled | Loads a model automatically when a request asks for it. | You want to load models only on command. |

## Logging

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `--log-disable` | none | logging on | Turns off logging. | You need silent output. |
| `--log-file` FNAME | `LLAMA_ARG_LOG_FILE` | none | Writes the log to this file. | You keep logs on disk. |
| `--log-jsonl`, `--no-log-jsonl` | `LLAMA_ARG_LOG_JSONL` | disabled | Writes the log as JSON lines to stdout and turns off colors. | A log collector reads JSON. |
| `--log-colors` [on, off, auto] | `LLAMA_ARG_LOG_COLORS` | `auto` | Colors the log. `auto` colors it only when the output is a terminal. | You pipe the output somewhere, so use `off`. |
| `-v`, `--verbose`, `--log-verbose` | none | off | Sets the verbosity to the highest level. | You debug and need every message. |
| `-lv`, `--verbosity`, `--log-verbosity` N | `LLAMA_ARG_LOG_VERBOSITY` | 3 | Sets the verbosity threshold. 0 is generic output, 1 is error, 2 is warning, 3 is info, 4 is trace and 5 is debug. Messages above the threshold are dropped. | You need more or fewer messages. |
| `--log-prefix`, `--no-log-prefix` | `LLAMA_ARG_LOG_PREFIX` | off | Adds a prefix to each log message. | You want the source of each message in the log. |
| `--log-timestamps`, `--no-log-timestamps` | `LLAMA_ARG_LOG_TIMESTAMPS` | off | Adds a timestamp to each log message. | You correlate the log with other events. |
| `--log-prompts-dir` PATH | none | disabled | Writes each prompt to files in this folder. Use it for debugging only. | You debug a prompt. Prompts can contain sensitive text. |

> REVIEW: The defaults for `--log-prefix` and `--log-timestamps` are shown as off because the `--help` text gives none. Could not verify in `common/log.cpp` or `common/common.h`.

## Text and prompt handling

| Flag | Environment variable | Default | What it does | Change it when |
|---|---|---|---|---|
| `-e`, `--escape`, `--no-escape` | none | enabled | Processes escape sequences such as `\n` in the prompt. | Your prompt has backslashes that must stay as typed. |
| `-r`, `--reverse-prompt` PROMPT | none | none | Stops generation when this text appears, and gives control back to the user. | You use an interactive reverse prompt. |
| `-sp`, `--special` | none | off | Shows special tokens in the output. | You debug the token output. |
| `--spm-infill` | none | off | Uses the suffix-prefix-middle order for infill. | A model expects that order. |

## Presets

These flags download a known model and set its options. Each one can download weights from the internet.

| Flag | Default | What it does |
|---|---|---|
| `--embd-gemma-default` | none | Uses the default EmbeddingGemma model. |
| `--fim-qwen-1.5b-default` | none | Uses Qwen 2.5 Coder 1.5B. |
| `--fim-qwen-3b-default` | none | Uses Qwen 2.5 Coder 3B. |
| `--fim-qwen-7b-default` | none | Uses Qwen 2.5 Coder 7B. |
| `--fim-qwen-7b-spec` | none | Uses Qwen 2.5 Coder 7B with a 0.5B draft for speculative decoding. |
| `--fim-qwen-14b-spec` | none | Uses Qwen 2.5 Coder 14B with a 0.5B draft for speculative decoding. |
| `--fim-qwen-30b-default` | none | Uses Qwen 3 Coder 30B A3B Instruct. |
| `--gpt-oss-20b-default` | none | Uses gpt-oss-20b. |
| `--gpt-oss-120b-default` | none | Uses gpt-oss-120b. |
| `--vision-gemma-4b-default` | none | Uses Gemma 3 4B QAT. |
| `--vision-gemma-12b-default` | none | Uses Gemma 3 12B QAT. |

## Informational flags

These flags print information and exit, or print help.

| Flag | What it does |
|---|---|
| `-h`, `--help`, `--usage` | Prints the usage text and exits. |
| `--version` | Prints the version and build information and exits. |
| `-cl`, `--cache-list` | Lists the models in the local cache and exits. |
| `--completion-bash` | Prints a bash completion script and exits. |
| `--list-devices` | Prints the available devices and exits. |

## Removed and deprecated flags

Removed flags stop the server with an error that names the replacement. Deprecated flags still work but do nothing useful, or they log a warning.

| Flag | Status | Replacement |
|---|---|---|
| `--draft`, `--draft-n`, `--draft-max` | Removed. | `--spec-draft-n-max` or `--spec-ngram-mod-n-max`. |
| `--draft-min`, `--draft-n-min` | Removed. | `--spec-draft-n-min` or `--spec-ngram-mod-n-min`. |
| `--spec-ngram-size-n` | Removed. | The matching `--spec-ngram-*-size-n` flag. |
| `--spec-ngram-size-m` | Removed. | The matching `--spec-ngram-*-size-m` flag. |
| `--spec-ngram-min-hits` | Removed. | The matching `--spec-ngram-*-min-hits` flag. |
| `-dt`, `--defrag-thold` | Deprecated. It logs a warning and is no longer needed. | None needed. |
| `enable_thinking` in `--chat-template-kwargs` | Deprecated. The server logs a warning. | `--reasoning on` or `--reasoning off`. |
| `preserve_reasoning` in `--chat-template-kwargs` | Deprecated. The server logs a warning. | `--reasoning-preserve` or `--no-reasoning-preserve`. |
