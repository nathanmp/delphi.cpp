# Offline auto-quantizer

`auto_quantize.py` chooses tensor qtypes under independent usable GPU/RAM budgets. It starts from the minimum-byte floor recipe and spends the budget on upgrades. It reuses `llama-quant-damage`, `test-backend-ops`, `llama-perplexity` (with tensor hotswap) and the bottleneck profiler. With a calibrated quality table, the loss is the predicted KL against teacher logits of the source model. Without a table, the loss is a sum of local source-relative errors, which is a proxy only. Speed starts from a measured full-model baseline and measured operation signatures. Predictions remain marked unqualified until held-out validation passes.

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

Multiple GGUF imatrices merge raw sums/counts: a file weighs in proportion to its tokens, or by an explicit weight (see Weighted calibration). A single legacy matrix is accepted; merging legacy matrices is not implemented. A quantized source requires `allow_requantize: true`, and its pre-existing error is excluded from the source-relative objective.

`sampling` accepts `samples`, `max_samples`, `seed` and `row_stats`. The default estimator reads each selected tensor once for row statistics. Set `row_stats: false` to sample without that pass, with the estimator's documented accuracy limitations.

`search` accepts `display`. `bounds` accepts `max_loss` (with a quality table), `max_damage` (without one), `min_decode_t_s`, `min_short_t_s` and `min_long_t_s`. Speed bounds constrain predictions. The JSON retains all nondominated evaluated recipes even when the console display is capped.

The search uses the fact that loss and time are sums over groups (the loss at a fixed composition point, the time without an expert cache). For a loss multiplier theta, an exact multiple-choice knapsack per device minimizes `theta * loss / loss_floor + sum(importance * time / time_baseline)` within the byte budget of that device. A sweep of theta gives the supported points of the trade-off. Walks between neighbouring sweep solutions add recipes between them, and bisection finds the edges of the bounds. Every candidate is then evaluated with the full memory and timing models. The cost does not depend on the number of candidate qtypes beyond the knapsack table size.

## Calibrated quality

The quality model predicts the KL of a recipe against teacher logits of the source model, relative to the floor recipe. Per decision group g with role r (routed or dense gate/up, down, shared experts, attention, other kinds) at depth x:

```text
gain_g(q) = s_r(x) * (R_g(floor)^gamma_r - R_g(q)^gamma_r),   s_r(x) = exp(a0 + a1 x + a2 x^2)
G = sum_g gain_g(q_g),   u = G / G_full,   KL = KL_floor * (1 - kappa(u) G),   kappa(u) = 1 + (kinf - 1)(1 - exp(-u / u0))
```

R is the sampled local relative error of `llama-quant-damage`, summed over the group members. Qtypes enter only through R. A qtype that the calibration did not probe needs no extra passes: prepare samples its R and the table predicts its KL. kappa describes the measured super-additive composition of upgrades. MTP blocks are the blocks with `blk.N.nextn.*` tensors and the last `nextn_predict_layers` blocks of the GGUF metadata (a warning reports a difference). The KL measurement does not run them, so they take the ceiling: the qtype of the pool with the most bytes, in the floor and in every plan, unless `pins` set them.

Calibrate after prepare. Calibration requires an imatrix: without activation statistics, R is a weight-space error, and it misranks layers inside a band (OLMoE-1B-7B without an imatrix: chi2/dof 8.6). `calibration.text` can also name several corpora with weights (see Weighted calibration).

```json
"calibration": {"text": "/data/calibration.txt", "chunks": 64, "context": 512}
```

```sh
python tools/auto-quantize/auto_quantize.py calibrate --config config.json --prepared /tmp/autoq-prepared/prepared.json --output /tmp/autoq-quality
```

The command:

1. Computes teacher logits from the source model with `llama-perplexity --kl-divergence-base`, unless `calibration.teacher` names an existing file. With RAM-routed experts, it keeps experts in RAM (`-cmoe`) and uses a batch of 16 chunks (`calibration.batch` overrides), so the source does not need to fit in VRAM. Set `GGML_SCHED_STAGE=1` in `runtime.env` to stream the experts to the GPU once per batch.
2. Selects two probe qtypes from the sampled damage: top (smallest local error) and mid (error nearest the geometric mean of the floor and top errors). `calibration.probe_qtypes` overrides them as `[mid, top]`. No probe model is written: each pass quantizes only the tensors it changes from the source (`llama-quantize --only-tensors`, same imatrix and options as `encode`).
3. Measures the probes in persistent hotswap sessions on the floor model, paired with the floor on the same tokens: per role, `bands` contiguous layer bands (default 10) upgraded to top and to mid; `recipes - 2` random mixed recipes (default 8 in total); and the two uniform recipes (every modelled group at mid or at top). This is `2 x bands x roles + recipes` passes, independent of the number of candidate qtypes and of the layer count. Each session has recompute and restore controls that must reproduce the floor bitwise.
4. Fits the parameters (weighted least squares on the paired per-chunk gains) and writes `quality.json` (`delphi.floor-quality/3`).

By default the sessions use the deployment placement (first GPU, `-cmoe` with RAM-routed experts, `--numa tensors`). `calibration.devices` can list other GPUs for faster full-offload sessions, one session per device. The execution path changes the numerics slightly; the table records the command. The hotswap session keeps the previous patch until it commits, so it returns to the floor (`reset`, no evaluation) before a large patch. A patch holds at most `calibration.patch_mib` (default 8192) of tensors; a larger recipe (the uniform and dense mixed recipes) returns to the floor, adds its chunks one by one (`add`) and then evaluates, so the temporary disk space is one chunk. The session holds the floor and the patched tensors at the same time: with routed experts in RAM, a uniform recipe whose expert copy does not fit in the available RAM (less 16 GiB) runs as a separate probe model after the sessions, one at a time, deleted after its run (`calibration.uniform`: `auto`, `session` or `direct`). A recipe that fails to apply in the session (out of memory) also runs that way.

Set `"quality": {"table": "/tmp/autoq-quality/quality.json"}` for plan and validate. The table must match the prepared source, imatrices, sampled damage and floor recipe. `bounds.max_loss` is a ceiling on the predicted KL. It is a model estimate on the calibration text with the recorded teacher, not a bound on other text or on task accuracy. Expert caches need separate calibration and are not supported with a table.

Results on Qwen3.6-35B-A3B (floor Q2_K experts, Q8_0 rest), deployment ROCm2 + RAM with CPU experts: calibration took 24.5 minutes (48 passes of 16K scored tokens). On separate text, predicted KL was within 0.98-1.06 of measured KL for five of six witnesses, including qtypes that were not probed (1.14 for one witness). The plan picks had 3.4 % +- 1.5 and 13.1 % +- 2.6 lower measured KL than picks by the summed local error at +2.2 and +4.2 GB of expert budget over the floor. At +9.2 GB both picks were equal within the noise (+3.0 % +- 5.3); both were 18-20 % better than uniform Q4_K. OLMoE-1B-7B with an imatrix fitted all 21 probes within one standard error.

## Weighted calibration

Calibration data can come from several domains with explicit weights. A weight is relative and independent of the size of the source: `code:50,math:30,roleplay:25` gives code 50/105, math 30/105 and roleplay 25/105 of the influence. Weights are normalized to sum 1. A weight is a positive number. An entry is `PATH`, `PATH:WEIGHT` or `{"path": PATH, "weight": WEIGHT}`, in a JSON list or in one comma-separated string. A plain path in a weighted list counts 1. A path that ends in `:` and digits, or contains a comma inside a string, needs the object form. Weighted files must be distinct.

| key | value |
| --- | --- |
| `imatrix` | entries of raw GGUF imatrices; weights change the merge |
| `calibration.text` | one path, or entries of text corpora; objects also accept `bytes_per_token` |
| `calibration.text_seed` | integer, default 1: selects the pieces of weighted corpora |
| `calibration.bytes_per_token` | positive number, default 4.0: sizes a piece in bytes |
| `imatrix_collect` | `text` (entries, required), `context` (512), `chunks` (0 = whole corpus), `means` (true), `threads` (`runtime.threads`), `extra_args` |

### Imatrices

```json
"imatrix": ["/data/code.imatrix.gguf:50", "/data/math.imatrix.gguf:30", "/data/roleplay.imatrix.gguf:25"]
```

```sh
python tools/auto-quantize/auto_quantize.py merge-imatrix --imatrix code.imatrix.gguf:50 math.imatrix.gguf:30 roleplay.imatrix.gguf:25 --output mixed.imatrix.gguf
```

`prepare` runs the same merge; `merge-imatrix` writes it alone, for example for `llama-quantize --imatrix`. Without any weight the sums and counts are added, as before: each file weighs in proportion to its tokens. To give every file equal influence, write `:1` on each entry. With weights, file k (weight `w_k`, normalized) is scaled by one factor before the addition:

```text
T_k = largest count of file k (the tokens it saw)      T = sum_k T_k      c_k = w_k * T / T_k
sums_e = sum_k c_k * sums_k,e      counts_e = sum_k c_k * counts_k,e      (activation sums likewise)
mean_e = sums_e / counts_e = sum_k w_k (n_k,e / T_k) m_k,e / sum_k w_k (n_k,e / T_k),   m_k,e = sums_k,e / counts_k,e
```

The merged mean of every tensor (and every expert of a routed tensor) is the mean of a token mixture in which file k supplies the fraction `w_k` of the tokens; an expert that a domain uses rarely gets little of that domain's weight, as in the real mixture. For dense tensors the mean is the weighted mean `sum_k w_k m_k`. The counts are those of `T` tokens (for dense tensors, the total of all inputs, the same size as a plain merge). They are rounded to whole numbers because the quantizer rounds them on load, and the sums are scaled by the same ratio, so no mean changes. `imatrix.datasets` and `imatrix.chunk_count` are combined as in a plain merge. A tensor that is missing from some files is merged from the files that have it; its weights act relative to each other. Activation sums stay only for tensors that every input has them for. Weighted merging needs F32 raw GGUF files.

The weights enter the input fingerprint, the damage cache key and the quality table check. Unweighted lists keep their old fingerprints, so existing prepared directories stay valid. Changing a weight needs a new prepare directory and a new calibration.

### Calibration text

```json
"calibration": {"text": ["/data/code.txt:50", "/data/math.txt:30", "/data/roleplay.txt:25"], "chunks": 64, "context": 512}
```

`calibrate` composes one text from the corpora and writes it with a sidecar record to the prepared directory (`calibration-text-<digest>.txt`). The name holds a digest of the corpus contents, the weights and the parameters, so a different mix never overwrites an earlier text. The quality table records the corpora and the text hash.

1. The text holds `ceil(1.2 * chunks)` pieces. The extra 20 % cover tokenization differences, because `llama-perplexity` scores only the first `chunks` windows.
2. The pieces are divided among the corpora by weight (largest remainder). A corpus that would get no piece is an error.
3. A piece is a span of about `context` tokens. The tool has no tokenizer, so it is `context * bytes_per_token` bytes, moved to line starts. This is the documented approximation: the token share follows the weights only as far as `bytes_per_token` holds for each corpus (about 4 for English prose, less for code, math and non-Latin text). Set it per corpus with the object form after measuring tokens per byte, for example with `llama-tokenize -m MODEL -f CORPUS --show-count` (bytes divided by tokens). If `llama-perplexity` runs fewer windows than `chunks`, raise it.
4. The pieces of a corpus are the non-overlapping slots with the smallest `sha256(text_seed:corpus:slot)`, in file order. Nothing is random at run time.
5. The pieces of all corpora are interleaved by `(k + 0.5) / n` for the k-th of n pieces, so every prefix of the text keeps the proportions.
6. A corpus with fewer slots than its pieces is an error that names the file, the bytes needed and the bytes available.

Windows and pieces are not aligned exactly, so a window can span two corpora. A single path is used as it is. Teacher logits are tied to the text hash: `calibrate` stops when an output directory holds logits from another text. With `calibration.teacher` set, the logits must come from the composed text.

### Collecting imatrices per corpus

```json
"imatrix_collect": {"text": ["/data/code.txt:50", "/data/math.txt:30", "/data/roleplay.txt:25"], "context": 512, "chunks": 0}
```

```sh
python tools/auto-quantize/auto_quantize.py collect-imatrix --config config.json --output /tmp/autoq-imatrix
```

This runs `llama-imatrix` once per corpus. It is expensive and runs only on this command, never inside `prepare`. Each run uses the calibration placement: the first GPU, `-cmoe` when `RAM` is a device, `--numa tensors` and `runtime.env`. `means: true` sets `DELPHI_IMATRIX_MEANS=1`, which adds the activation sums. Outputs are `NN-<corpus>.imatrix.gguf` with `imatrix-collect.json`. A finished output is reused when its corpus hash, model, binary and command match; otherwise the command stops. The command prints the `"imatrix"` entries with the weights; paste them into the configuration. The weights do not affect a run, so a new mix needs a new merge, not a new collection. Use text that is different from `calibration.text` for the imatrix. The KL measurement on the text that trained the imatrix favors the quantized models.

Rejected inputs: weights that are zero, negative or not finite; duplicate files in a weighted list; unknown keys in an entry object; a corpus smaller than its share; a share of zero pieces; `extra_args` that set managed options; an output directory with logits from another text; a quality table made for other weights.

## Command-line wrapper

`quant_cli.py` wraps the workflow below for ordinary use. `setup` runs prepare and calibrate once into a work directory; `plan` turns a query into planning-only changes (weights, budgets, bounds), runs the existing planner on the prepared inputs and shows at most `--rows` recipes thinned from the whole frontier (best score, the extreme of each metric, then farthest-point samples in log space). A pick (`--pick`, or a prompt on a terminal when `--output` is set) can be encoded at once.

```sh
python tools/auto-quantize/quant_cli.py setup --config config.json --work /tmp/autoq
python tools/auto-quantize/quant_cli.py plan --config config.json --work /tmp/autoq --weights loss=.4,decode=.3,short=.2,long=.1
python tools/auto-quantize/quant_cli.py plan --config config.json --work /tmp/autoq --minimize kld --budget RAM=300GiB --pick min-kld --output /models/out.gguf
python tools/auto-quantize/quant_cli.py show --frontier /tmp/autoq/plans/plan-001/frontier.json --detail best
```

`--max-size 55GiB` caps the summed tensor bytes over all devices (`bounds.max_total_bytes`: one pooled knapsack, device capacities still checked). `--minimize size --max-kld 0.05` finds the smallest recipe whose predicted KL is within the ceiling (`search.minimize: size`): it bisects the pooled byte cap on the exact minimum-loss knapsack, then plans under that cap. Speed bounds and device capacities are checked on the result only. With a GPU-only configuration, the chosen recipe is followed by a `Run with:` line holding its `--device`, `-sm` and `-ts` arguments, and sizes count the host-resident input embedding. Run `python tests/test-quant-cli.py` for the offline checks.

## Workflow

```sh
python tools/auto-quantize/auto_quantize.py inspect --config config.json
python tools/auto-quantize/auto_quantize.py collect-imatrix --config config.json --output /tmp/autoq-imatrix   # optional, slow
python tools/auto-quantize/auto_quantize.py prepare --config config.json --output /tmp/autoq-prepared
python tools/auto-quantize/auto_quantize.py calibrate --config config.json --prepared /tmp/autoq-prepared/prepared.json --output /tmp/autoq-quality
python tools/auto-quantize/auto_quantize.py plan --config config.json --prepared /tmp/autoq-prepared/prepared.json --output /tmp/autoq-frontier
python tools/auto-quantize/auto_quantize.py encode --frontier /tmp/autoq-frontier/frontier.json --recipe recipe-000 --output /models/selected.gguf
```

Prepare samples candidate tensors, measures legal operation signatures, creates one minimum-byte baseline with the permitted other-tensor types, and profiles short/long prefill and decode. The profiler owns the benchmark, warmup, repetitions, identity recording and trace artifacts; all three workloads share one loaded `llama-bench` process. Operation calibration has a 600-second cap, adjustable with `--measurement-seconds`. Completed signatures are saved if the cap is reached. The final GGUF is created only by an explicit encode command. Existing output files are refused.

Prepared inputs include source file size/mtime/metadata, imatrix content hashes and weights, binary identities, runtime and sampling settings. A changed input requires a new prepare directory. Preference weights, budgets and speedup factors can be changed for planning when they do not change the measured layout. Timing caches describe their recorded machine state; remeasure after clock, topology or host-load changes. Source identity uses size/mtime/metadata rather than a full weight-payload hash.

The built-in timing prior measures each qtype/shape on CPU and the first GPU. `timing.link_bandwidth_gb_s` optionally supplies effective host-to-device bandwidth; the default 12 GB/s is an unmeasured prior. Fixed handoff latency, uniform routes, interpolated tail batches and a single positive scale per regime are V1 approximations. CPU/GPU compute is measured, not derived from one global qtype coefficient. Matched expert gate/up uses `test-backend-ops --custom ...,pair=1` to include SwiGLU and fusion eligibility.

A RAM-resident expert qtype can remain on CPU when its GPU operation is unsupported. Long prefill then uses its CPU estimate. GPU support is required for dense/GPU-resident weights and expert caches. Missing measurements still require calibration; an explicitly unsupported GPU signature does not exclude a supported CPU expert format.

Repeated isolated decode probes can keep their few active expert weights in CPU cache. For batches of at most 16 tokens, the predictor also bounds CPU time by the bytes fetched from RAM; larger batched probes read most of the bank and already include that traffic. Its default effective byte rate is derived from the baseline's routed-expert bytes and traced decode CPU span; this incorporates CPU overhead and is not a measurement of peak RAM bandwidth. `timing.cpu_bandwidth_gb_s` can supply an explicit rate instead. The fitted scale and held-out checks remain necessary. Long-prefill transfer predictions use the actual configured `GGML_SCHED_STAGE_MIN_TOKENS` threshold.

With staging enabled and a positive threshold, the runtime selects the GPU above that threshold and transfers the whole expert bank. The predictor follows that policy even if a CPU estimate would be faster. It estimates compute plus transfer before the fitted workload scale; copy/compute overlap remains an approximation. Timing-model revisions invalidate old prepared calibrations.

CPU expert probes also set `numa_expert=1`. `--numa tensors` alone pins workers but does not select the model loader's flagged expert kernel path. Both the flag and the NUMA mode are required to reproduce that execution contract.

`runtime.depth_tokens` sets the benchmark's existing context depth before each workload; default zero. KV types can use the benchmark's `-ctk`/`-ctv` options in `extra_args`. The benchmark derives its allocation from depth and workload lengths.

Lazy tables (`engram_embd`, `per_layer_token_embd`) are read row by row from disk. They stay at the source type, or at the type of a `pins` entry (a BF16 table can be about 100 GB in every encoded model; the production Qwen 3.8 quant pins `per_layer_token_embd.weight` to Q4_0), are not part of any qtype pool and do not count in device budgets, `bounds.max_total_bytes` or the reported sizes; plans report them as `lazy_bytes`. Every run of the tool on such a model (`llama-bench`, `llama-perplexity`, `llama-imatrix`) adds `-lm mmap -lzm on -ot <tables>=CPU`, so the tables are not loaded into a device. The measured speeds include the lazy lookups of the baseline's page-cache state (the same for every recipe), and the page cache decides how much RAM they use: size the RAM budget for the page cache you want to leave to them. `encode` copies the tables unchanged and verifies the copy.

`runtime.trace_max_events` sets the profiler trace capacity; default 1048576. Increase it for larger models or workloads if the profiler reports dropped events. A truncated trace is rejected rather than used for CPU calibration.

## Cache and multiple GPUs

With RAM supplied, routed experts remain authoritative in RAM and dense weights stay on the first GPU. `cache_mib` enables the existing first-GPU expert cache; default zero. Cache group rounding, maps, device scratch and pinned refill buffers are counted. Backend allocator/repack overhead and unprovided runtime allowances still need runtime validation. Cache hit predictions currently use a uniform residency prior, not a measured temporal-policy replay; validate cache-enabled mixes separately. Partial cache paths have different fusion eligibility.

With RAM supplied, several GPUs do not pool VRAM or create a multi-GPU cache command. The first GPU's budget still governs its weights/cache. The other budgets are reported as unassigned. Supply throughput assumptions for the requested multi-GPU deployment:

```json
"multi_gpu_speedup_percent": {"decode": 30, "short": 10, "long": 50}
```

The predicted times divide by 1.30, 1.10 and 1.50 respectively. These are assumed whole-workload throughput gains, not percentages of saved wall time. Validation executes on the first GPU and does not certify these assumptions.

## GPU-only placement over several GPUs

`"devices": {"ROCm0": "32GiB", "ROCm1": "32GiB", "ROCm2": "32GiB"}` (several GPUs, no `RAM`) plans a layer split like `-sm layer -ts`: blocks `0..n_layer-1` go to the GPUs in the configured order in contiguous runs, the output (and a tied embedding copy) to the last GPU, and `token_embd` stays on the host (`host_bytes`, outside every budget). The split minimizes the largest `(reserve + bytes) / capacity`; all search constraints use the per-GPU headroom, and the byte pool is the summed `capacity - reserve`. Each recipe records `layer_split` (the last count includes the output) and `runtime_args` (`--device ROCm0,ROCm1,ROCm2 -sm layer -ts 20,21,22`, the llama-server form; llama-bench takes `/` inside a value). The baseline and the validation witnesses are profiled with their real split. The GPUs must be identical models, since operation timings are measured on the first GPU; the timing model sums the layer costs over the devices, so pipeline overlap in multi-ubatch prefill is not modelled and the fitted slopes absorb a uniform scale. `multi_gpu_speedup_percent` must stay zero here. `cache_mib` needs RAM-routed experts. Calibration and imatrix runs use the first GPU with `-cmoe`, because a big model does not fit on one GPU and quality does not depend on the placement.

## Bounded validation

Use a small model and supply two fit mixes and at least two distinct held-out mixes:

```json
[
  {"name": "fit-q4k", "role": "fit", "primary": "Q4_K"},
  {"name": "fit-q6k", "role": "fit", "primary": "Q6_K"},
  {"name": "hold-q4zero", "role": "holdout", "primary": "Q4_0"},
  {"name": "hold-mixed", "role": "holdout", "primary": "Q4_K", "overrides": {"ffn_down_exps": "Q6_K"}},
  {"name": "hold-planned", "role": "holdout", "recipe": "/tmp/autoq-frontier/recipe-000.json"}
]
```

```sh
python tools/auto-quantize/auto_quantize.py validate --config config.json --prepared /tmp/autoq-prepared/prepared.json --mixes mixes.json --output /tmp/autoq-validation
```

A `recipe` entry uses the choices of a plan recipe file. The command encodes and profiles only those mixes, fits one scale per regime from fit mixes, then records held-out relative error and material rank reversals. With a quality table, it also measures the KL of every mix against the table's teacher and reports predicted against measured KL. The quality gate requires at most 15 % relative error and no reversed ordering for measured differences of at least 5 %. The initial gate requires at most 15% relative error in every held-out regime and no reversed ordering for measured differences of at least 5%. A failed fit or gate keeps predictions unqualified. The report saves exact profiler commands, binary/model identities and raw artifacts. This is plain `llama-bench` execution prediction at the configured depth; server overhead and speculative acceptance are not calibrated. Trace integrity and a final memory witness remain separate release checks.

Run offline invariants with `python tests/test-auto-quantize.py` and estimator checks with `test-quant-damage`.
