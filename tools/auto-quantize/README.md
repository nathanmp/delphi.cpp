# Offline auto-quantizer V1

`auto_quantize.py` chooses tensor qtypes under independent usable GPU/RAM budgets. It reuses `llama-quant-damage`, `test-backend-ops` and the bottleneck profiler. The damage objective is a sum of local source-relative errors; it is not predicted KLD or accuracy. Speed starts from a measured full-model baseline and measured operation signatures. Predictions remain marked unqualified until held-out validation passes.

Run with a Python environment that has the repository's `gguf-py` package installed. Build `llama-quant-damage`, `llama-quantize`, `llama-bench` and `test-backend-ops`, plus the required device backend. Use a dedicated build directory.

## Configuration

Paths are relative to the configuration file. Capacities in a string use decimal MB unless a unit is present; integer capacities in an object are bytes. Budgets are usable allocations after the user subtracts runtime/KV/vision allowances. Optional `reserve` subtracts additional fixed allowances. Tensor weights and cache copies are always counted by the advisor.

```json
{
  "schema": "delphi.auto-quantize/1",
  "model": "/models/model-bf16.gguf",
  "devices": "ROCm1:16000,RAM:32000",
  "imatrix": ["/models/imatrix.gguf"],
  "bindir": "/path/to/build-auto-quantizer/bin",
  "primary_qtypes": ["Q2_K", "Q4_0", "Q4_K", "Q6_K"],
  "other_qtypes": ["Q6_K"],
  "preserve_below_mib": 1,
  "tie_gate_up": true,
  "importance": {"precision": 1, "decode": 1, "short": 1, "long": 1},
  "runtime": {
    "threads": 32,
    "short_tokens": 128,
    "long_tokens": 4096,
    "decode_tokens": 256,
    "repetitions": 3,
    "env": {"GGML_SCHED_STAGE": "1"}
  }
}
```

For MoE, primary tensors are routed experts; for dense models they are ordinary FFN matrices. `primary_regex` can override this classification. Qtype pools accept actual GGML types, not presets such as Q4_K_M. `pins` maps exact tensor names to qtypes; `preserve_regex` adds preservation patterns. A preserved or pinned gate/up partner constrains the whole tied group. The tool verifies the final encoder output against the plan.

Multiple GGUF imatrices merge raw sums/counts. A single legacy matrix is accepted; merging legacy matrices is not implemented. A quantized source requires `allow_requantize: true`, and its pre-existing error is excluded from the source-relative objective.

`sampling` accepts `samples`, `max_samples`, `seed` and `row_stats`. The default estimator reads each selected tensor once for row statistics. Set `row_stats: false` to sample without that pass, with the estimator's documented accuracy limitations.

`search` accepts `beam`, `evaluations` and `display`. `bounds` accepts `max_damage`, `min_decode_t_s`, `min_short_t_s` and `min_long_t_s`. Speed bounds constrain predictions. The JSON retains all nondominated evaluated recipes even when the console display is capped. Search is bounded and does not guarantee the global frontier.

## Experimental repair quality

Set `"quality": {"table": "repair-table.json"}` to replace local damage with the sum of measured conditional repair KL values. Use `"bounds": {"max_loss": 0.09}` for a ceiling in those proxy units. `max_damage` and `max_loss` apply to different metrics and cannot be interchanged. Repair loss is not a bound on original-relative KLD or a task-accuracy percentage.

The table uses `delphi.floor-quality/2` and `method: sum_conditional_repair_kl`. It contains content identities for every source shard and imatrix, a complete floor tensor manifest, the quality execution policy, and non-overlapping measured groups. Each option records its qtype, exact bytes, mean and per-chunk losses, and the SHA256 of every member tensor payload. Source/imatrix records use `bytes` and `sha256`; the floor also supplies its GGUF `path` and a `tensors` map with `shape`, `qtype`, `bytes` and `sha256`. `execution` records the first `gpu`, `ram_routed` placement, `numa: tensors`, and the calibration command/environment. Draft `/1` tables lack these checks and require export to `/2`.

Prepare reuses the supplied floor and skips local-error sampling. All source, floor and imatrix identities must match. The floor qtype must be an allowed minimum-byte option for each measured group. Unmeasured tensors stay at their recorded floor types; the tool does not combine uncalibrated local scores with repair scores. The category pools, preservation rules and pins must permit that fixed floor. Search changes whole measured groups, while timing keeps the per-layer gate/up pairs. Final encoding verifies every tensor payload against the calibration identities, including unchanged tensors.

CPU-expert/GPU-dense placement is supported. A table collected with GPU-resident experts cannot be used for RAM-routed planning, or the reverse. Scores apply to the recorded quality execution policy; different prefill offload, kernel or cache paths need quality validation. The independent speed profiler retains its configured execution policy. Repair tables do not yet support an expert cache. The table provider remains opt-in and experimental; the default local metric is unchanged.

## Workflow

```sh
python tools/auto-quantize/auto_quantize.py inspect --config config.json
python tools/auto-quantize/auto_quantize.py prepare --config config.json --output /tmp/autoq-prepared
python tools/auto-quantize/auto_quantize.py plan --config config.json --prepared /tmp/autoq-prepared/prepared.json --output /tmp/autoq-frontier
python tools/auto-quantize/auto_quantize.py encode --frontier /tmp/autoq-frontier/frontier.json --recipe recipe-000 --output /models/selected.gguf
```

Prepare samples candidate tensors, measures legal operation signatures, creates one minimum-byte baseline with the permitted other-tensor types, and profiles short/long prefill and decode. The profiler owns the benchmark, warmup, repetitions, identity recording and trace artifacts; all three workloads share one loaded `llama-bench` process. Operation calibration has a 600-second cap, adjustable with `--measurement-seconds`. Completed signatures are saved if the cap is reached. The final GGUF is created only by an explicit encode command. Existing output files are refused.

Prepared inputs include source file size/mtime/metadata, imatrix content hashes, binary identities, runtime and sampling settings. A changed input requires a new prepare directory. Preference weights, budgets and speedup factors can be changed for planning when they do not change the measured layout. Timing caches describe their recorded machine state; remeasure after clock, topology or host-load changes. Source identity uses size/mtime/metadata rather than a full weight-payload hash.

The built-in timing prior measures each qtype/shape on CPU and the first GPU. `timing.link_bandwidth_gb_s` optionally supplies effective host-to-device bandwidth; the default 12 GB/s is an unmeasured prior. Fixed handoff latency, uniform routes, interpolated tail batches and a single positive scale per regime are V1 approximations. CPU/GPU compute is measured, not derived from one global qtype coefficient. Matched expert gate/up uses `test-backend-ops --custom ...,pair=1` to include SwiGLU and fusion eligibility.

A RAM-resident expert qtype can remain on CPU when its GPU operation is unsupported. Long prefill then uses its CPU estimate. GPU support is required for dense/GPU-resident weights and expert caches. Missing measurements still require calibration; an explicitly unsupported GPU signature does not exclude a supported CPU expert format.

Repeated isolated probes can keep active expert weights in CPU cache. The predictor also bounds CPU time by the bytes fetched from RAM. Its default effective byte rate is derived from the baseline's routed-expert bytes and traced decode CPU span; this incorporates CPU overhead and is not a measurement of peak RAM bandwidth. `timing.cpu_bandwidth_gb_s` can supply an explicit rate instead. The fitted scale and held-out checks remain necessary. Long-prefill transfer predictions use the actual configured `GGML_SCHED_STAGE_MIN_TOKENS` threshold.

With staging enabled and a positive threshold, the runtime selects the GPU above that threshold and transfers the whole expert bank. The predictor follows that policy even if a CPU estimate would be faster. It estimates compute plus transfer before the fitted workload scale; copy/compute overlap remains an approximation. Timing-model revisions invalidate old prepared calibrations.

CPU expert probes also set `numa_expert=1`. `--numa tensors` alone pins workers but does not select the model loader's flagged expert kernel path. Both the flag and the NUMA mode are required to reproduce that execution contract.

`runtime.depth_tokens` sets the benchmark's existing context depth before each workload; default zero. KV types can use the benchmark's `-ctk`/`-ctv` options in `extra_args`. The benchmark derives its allocation from depth and workload lengths. Models with large lazy engram/per-layer embedding tables are explicitly unsupported until resident-byte accounting is added.

`runtime.trace_max_events` sets the profiler trace capacity; default 1048576. Increase it for larger models or workloads if the profiler reports dropped events. A truncated trace is rejected rather than used for CPU calibration.

## Cache and multiple GPUs

With RAM supplied, routed experts remain authoritative in RAM and dense weights stay on the first GPU. `cache_mib` enables the existing first-GPU expert cache; default zero. Cache group rounding, maps, device scratch and pinned refill buffers are counted. Backend allocator/repack overhead and unprovided runtime allowances still need runtime validation. Cache hit predictions currently use a uniform residency prior, not a measured temporal-policy replay; validate cache-enabled mixes separately. Partial cache paths have different fusion eligibility.

Multiple GPUs do not pool VRAM or create a multi-GPU cache command. The first GPU's budget still governs its weights/cache. The other budgets are reported as unassigned. Supply throughput assumptions for the requested multi-GPU deployment:

```json
"multi_gpu_speedup_percent": {"decode": 30, "short": 10, "long": 50}
```

The predicted times divide by 1.30, 1.10 and 1.50 respectively. These are assumed whole-workload throughput gains, not percentages of saved wall time. Validation executes on the first GPU and does not certify these assumptions. GPU-only multiple-device placement is not implemented.

## Bounded validation

Use a small model and supply two fit mixes and at least two distinct held-out mixes:

```json
[
  {"name": "fit-q4k", "role": "fit", "primary": "Q4_K"},
  {"name": "fit-q6k", "role": "fit", "primary": "Q6_K"},
  {"name": "hold-q4zero", "role": "holdout", "primary": "Q4_0"},
  {"name": "hold-mixed", "role": "holdout", "primary": "Q4_K", "overrides": {"ffn_down_exps": "Q6_K"}}
]
```

```sh
python tools/auto-quantize/auto_quantize.py validate --config config.json --prepared /tmp/autoq-prepared/prepared.json --mixes mixes.json --output /tmp/autoq-validation
```

The command encodes and profiles only those mixes, fits one scale per regime from fit mixes, then records held-out relative error and material rank reversals. The initial gate requires at most 15% relative error in every held-out regime and no reversed ordering for measured differences of at least 5%. A failed fit or gate keeps predictions unqualified. The report saves exact profiler commands, binary/model identities and raw artifacts. This is plain `llama-bench` execution prediction at the configured depth; server overhead and speculative acceptance are not calibrated. Trace integrity and a final output-quality/memory witness remain separate release checks.

Run offline invariants with `python tests/test-auto-quantize.py` and estimator checks with `test-quant-damage`.
