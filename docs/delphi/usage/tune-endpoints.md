# Tune endpoints

`llama-server --tune-endpoints` (env `LLAMA_ARG_TUNE_ENDPOINTS`) adds three routes for the auto-tuner. They are off by default: without the option the routes do not exist (404). They are not part of the router server.

## POST /tune/context

Makes the `llama_context` again on the loaded weights. A launch takes minutes (weights, repack, cache fill, warmup); this takes seconds.

The body is a JSON object. Unknown keys are errors.

| key | CLI option | note |
| --- | --- | --- |
| `ctx_size` | `-c` | Only down: the context that the first load fitted is the ceiling. |
| `batch_size`, `ubatch_size` | `-b`, `-ub` | |
| `cache_type_k`, `cache_type_v` | `-ctk`, `-ctv` | |
| `flash_attn` | `-fa` | `"on"`, `"off"`, `"auto"` or a boolean. |
| `threads`, `threads_batch` | `-t`, `-tb` | A draft thread count that followed the main count at launch keeps following it. |
| `cpu_mask`, `cpu_range`, `cpu_mask_batch`, `cpu_range_batch` | `-C`, `-Cr`, `-Cb`, `-Crb` | A mask in the request replaces the old one. |
| `ubatch_size_draft`, `threads_draft`, `threads_batch_draft` | `-ubd`, `-td`, `-tbd` | |
| `n_outputs_max` | | Smallest output limit per ubatch. `POST /tune/kld` needs one output for each token of its batch. |
| `spec_args` | | Array of `--spec-*` options, applied as for `POST /speculative`. |
| `env` | | Object `name -> value`; `null` unsets. Set before the contexts are made, and kept for later requests. |

Steps: new requests wait; running slots finish; the slots, the speculative state (draft, DSpark and MTP contexts), the expert cache with its tiers, staging, pinned buffers and worker threads, the VRAM-sharing leases, the scheduler worker, the prompt cache and the `llama_context` are freed; they are made again from the stored parameters with the changes applied; the normal warmup runs; new requests continue. The response is the same as `GET /tune/context` plus `ok`, `load_ms` and `warmup_ms`.

On a failure (for example out of memory) the old parameters and env come back and the response is an error. If the old context cannot be made again, the server exits with code 1.

What stays: model weights, mmap, repacked CPU buffers and the draft model weights. What cannot change without a new launch: the model files, `--device`, `-ts`, `-ngl`, `-ot`, `--n-cpu-moe`, `--numa`, mmap and mlock options, the draft model, and `-c` above the launch value. A re-created context is not byte-identical in its timing to a fresh one; confirm results with a fresh launch.

## GET /tune/context

Current values (the keys above, plus `env` set by requests), `vram` (used, free and total MiB per device, from the backends), `rss_mib` and `n_process_threads`.

## POST /tune/kld

Body: `{"base_file": "<path>", "n_ctx": <int>, "batch": <int, default 4>}`.

Runs the KL divergence of the live context against a base file written by `llama-perplexity --kl-divergence-base` (same code and file format as `llama-perplexity --kl-divergence`). The base file holds the evaluation tokens, so no text is sent. `n_ctx` is optional and checked against the file. The batch is the number of tokens of each `llama_decode`; with the expert cache on, use small batches (4) so that the cache path runs as in decode. A batch above the output limit of the context needs `n_outputs_max` in a `POST /tune/context` first.

The response has `kld` and `kld_se`, `ppl`, `ppl_se`, `ppl_base`, `log_ppl_ratio` with `log_ppl_ratio_se`, `same_top` and `same_top_se` (fractions), `p_diff_rms`, `tokens`, `chunks` and `ms`. The request blocks the server; the slot caches are cleared afterwards.

## Speculative settings

`POST /speculative` changes without a new context: `--spec-type` (also the `ngram-*` types, and draft types when the launch made the draft context), `--spec-draft-n-max` (not above the launch value, which sized the output buffers), `--spec-draft-n-min`, `--spec-draft-p-min`, `--spec-draft-p-split`, the adaptive options, `--spec-draft-sampling`, `--spec-ngram-*`, `--spec-synth-*`. They need `POST /tune/context` (`spec_args`, `ubatch_size_draft`, `threads_draft`, `threads_batch_draft`): the draft ubatch and threads, and a draft n-max above the launch value. A different draft model, `--spec-draft-device` and `--spec-draft-ngl` need a new launch.

## Environment classes

`tools/auto-tune/env-reload.json` maps each tunable variable to `context` (read when a context, an expert cache or a scheduler is made, so a re-created context sees a new value) or `process` (read once into a `static`; needs a new launch). A name that is not in the file is `process`. The `LLAMA_MOE_CACHE_*` sizes, prompt rows and refill options, `LLAMA_MOE_PREFILL_*`, `GGML_SCHED_*` and `GGML_SCHED_STAGE*` are `context`.

## CPU set

Every thread that delphi.cpp pins stays inside the CPU mask of the process (`sched_getaffinity` at start) and inside `--cpu-mask` or `--cpu-range`: the scheduler worker, the staging threads (`GGML_SCHED_STAGE_PIN`), the expert cache refill threads (`LLAMA_MOE_CACHE_REFILL_NUMA`), the NUMA loader threads and the CPU backend threads (`--numa tensors`). The old choice stays where it is allowed; otherwise a thread takes an allowed CPU of the wanted node, then any allowed CPU.
