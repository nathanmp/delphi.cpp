# Launch-parameter auto-tuner

`auto_tune.py` finds better launch flags and environment knobs for one model on a fixed set of devices. It starts from a `tools/bottlenecks` recipe arm, changes one registered knob at a time, keeps a change when it beats the current best by the noise margin, and then confirms the best config against the arm with interleaved runs (best, base, best). It never changes a deployment; it writes a report and `best.json`.

Status: V1 milestone 3, with server limitations below. Loop 1 searches placement and process-time knobs. For each launch candidate, loop 2 searches context knobs through feature-detected `/tune/context`; a 404 uses full launches within the same budget. Opt-in loop 3 calculates MTP request policies from acceptance logs and pass costs. Precision gates are a later milestone and precision levels stay fixed, even when `kld.limit` is set.

Each context point uses three full workload repeats (`context_search.reps`, at least two). A complete repeat contains the recipe's decode client and every weighted prefill length. After each repeat, racing can stop a losing point if its log score is below the incumbent by more than three standard deviations of the incumbent's repeat scores (with `noise_margin` as the minimum gap). Partial points cannot win or be exported. Context points use wall time but no launch slots; three fresh launches stay reserved for confirmation. Server and recipe warmup stay enabled, and prompt caching is disabled.

Context search uses coordinate descent, with batch x auto-cache margin searched jointly while cache precision stays fixed. Explicit NUMA overrides are process knobs and each gets its own thread search. More than eight surviving context knobs trigger two randomized finite-domain Morris screening trajectories; the batch/cache and thread coordinates remain eligible for refinement. The target, batch and following draft thread counts stay inside the CPU budget; custom levels outside it and the prohibited count 64 are removed. Per-launch traces use the fresh-launch fallback. Loop 3 also needs that fallback because the server flushes its trace only at process exit.

## Commands

```bash
python3 tools/auto-tune/auto_tune.py plan   tune.json --max-cores 48  # loads nothing
python3 tools/auto-tune/auto_tune.py run tune.json --cpus 0-23,32-55 [--max-launches N] [--max-hours H]
python3 tools/auto-tune/auto_tune.py report tune.json   # rewrite report.md from the stored results
```

`run` takes the representative and device benchmark locks itself; do not wrap it in `flock`. No builds or other model runs may compete with it. Give exactly one CPU budget in the file or on the command line: `cpus` uses the exact logical CPU list, and `max_cores` selects at most that many physical cores from the process affinity, balanced across NUMA nodes. The server inherits this affinity, its target/draft masks are restricted to it, and every observed server thread is audited. `best.json` includes the `taskset` command needed to enforce the same budget.

`run` resumes: measurements already in `results.jsonl` with the same binary, model, command, env and measurement settings are reused. A launch that started but never ended (the machine went down) is recorded as `crashed` and is not tried again; delete `journal.jsonl` to retry it.

`devices` is ordered by priority. Each device takes `fill: "fill"` (use available memory) or `fill: "min"` (only the overflow), plus the existing memory caps. The default margin is 1536 MiB; ROCm2 defaults to 2048 MiB. `placement/plan.json` records no-allocation fit measurements, extra draft/MTP reservations and candidates. The runner verifies candidates with fresh profiler launches at the context floor. Host experts use auto cache tiers, sized after context and draft allocation. The auto cache has one margin for all devices, so protecting a `min` device can leave space unused on `fill` devices. MTP and coupled draft contexts (including DSpark/DFlash) need conservative buffer estimates because the fit CLI cannot create them. A GPU mmproj needs `placement.reserve_mib: {"DEVICE": MiB}` from a server memory report; check that it loads before auto cache sizing.

## Configuration

```json
{
  "schema": "delphi.auto-tune/1",
  "recipe": "/path/recipes.py:qwen_38_flash_next",
  "arm": "mtp-real",
  "bindir": "/path/build/bin",
  "env": {"LD_LIBRARY_PATH": "/path/build/bin:/opt/rocm/lib"},
  "output": "/dev/shm/tune-qwen",
  "weights": {"prefill": 30, "decode": 70},
  "prefill": {"text": "/path/real-text.txt", "lengths": {"300": 2, "2048": 1}, "reps": 3},
  "decode": {"osl": 256},
  "context_search": {"reps": 3, "screen_threshold": 8},
  "speculation": {"enabled": false, "max_draft": 6, "osl": 64, "prompts": 3},
  "context": 32768,
  "kld": {"limit": 0.02, "text": "/path/eval.txt", "chunks": 8, "ctx": 512, "batch": 4},
  "devices": {"ROCm2": {"min_free_mib": 2048}, "ROCm0": {"max_used_mib": 16384}},
  "clock_mode": "auto",
  "budget": {"max_launches": 60, "max_hours": 6},
  "knobs": {"only": [], "include": [], "exclude": ["-t"], "levels": {"-ub": [512, 1024, 2048]}},
  "class_order": ["fit", "precision", "accept", "speed"],
  "noise_margin": 0.02,
  "warmup_launch": true,
  "trace": false
}
```

| key | meaning |
| --- | --- |
| `recipe`, `arm` | the start point and the reference of every score; the arm must be a server arm |
| `weights` | relative weights of prefill and decode, normalized to sum 1 |
| `prefill.lengths` | prompt lengths in tokens with relative weights; each is `reps` real-text requests from `text`, the median counts |
| `decode.osl` | tokens generated by the arm's decode client (real acceptance) |
| `context` | `-c` of every launch: a constraint, not a knob |
| `kld.limit` | precision knobs are tuned only when it is set; a config with a non-exact precision level must have a mean KLD under it |
| `kld.batch` | `-b`/`-ub` of `llama-perplexity`; 4 runs the decode path (and lets the GPU expert cache refresh) |
| `devices` | per llama device: `min_free_mib` (free VRAM at the peak, other processes included) and `max_used_mib` (this launch's own peak use) |
| `clock_mode` | the forced performance level that every used amdgpu card must have at the start; `null` skips the check |
| `budget` | at most `max_launches` full launches and `max_hours`; re-creates use time only, and 3 launches stay reserved for confirmation |
| `context_search` | full workload repeats per context point, and the knob count above which screening runs |
| `speculation` | opt-in MTP replay, live snapshots and matched validation; see below |
| `knobs` | `only` limits the registry, `include` adds opt-in knobs, `exclude` removes knobs, `levels` replaces the levels of a knob |
| `class_order` | the search order of the knob classes; the budget can end the search before the last ones |
| `noise_margin` | a change must beat the current best score by this fraction |
| `warmup_launch` | one unscored launch first (page cache) |
| `trace` | record the execution trace of every launch (adds the speculative cycle to the results) |

## Objective

score = product over the metrics of (metric / base metric) ^ weight. The metrics are decode t/s of the client (real acceptance) and, for each prefill length, 1 / time to first token. The weights are the decode weight and the prefill weight split over the lengths. The base scores 1.00; 1.05 is a 5 % better weighted geometric mean. A launch that fails, breaks a VRAM limit or has a KLD over the limit has no score.

## Knobs

`knobs.py` holds the registry. Classes:

| class | examples | how they are judged |
| --- | --- | --- |
| fit | `-ub`, `-b`, `-ubd`, staging slots, cache margin | score; a level that does not load is a recorded failure |
| speed | `-t`, staging threshold, cache prompt rows and refresh interval | score |
| accept | draft n-max/n-min/p-min, adaptive depth and floor | MTP replay estimates, then fresh real-acceptance confirmation |
| precision | KV type, GPU expert cache, `-sm tensor` (opt-in), indexer split (opt-in) | KLD gate, then score |

A knob is offered only where it applies (for example staging knobs only with `GGML_SCHED_STAGE=1`, draft batch knobs only with `--spec-type`). Threads start at physical cores less two per GPU and can move up to the physical-core budget or `max_threads`. Unknown env knobs require a process launch. Context requests contain the complete target settings and env changes, including unsets, so rejected points and returning to an earlier point cannot leave stale settings behind. The server restores a rejected context; the tuner records it as infeasible and continues. Transport failures close the owned server and require a new launch.

## Output

`output/` holds `results.jsonl`, `journal.jsonl`, `state.json`, `report.md`, `best.json`, the launch artifacts and the `llama-perplexity` logs. The reference logits are in `/dev/shm` during the run and are deleted at its end.

The report gives the recommendation (the confirmed best, or the base when the confirm did not hold the gain) with a paste-ready command and environment, the best measured config for decode shares 0, 0.25, 0.5, 0.75 and 1 of the weight (single runs), and every measurement with TTFT and prompt t/s per length, acceptance, KLD and VRAM use.

## Speculative replay (loop 3)

Set `speculation.enabled: true` to collect greedy MTP acceptance and T(w) at each context point. The tuner enables `DELPHI_SPEC_TRACE`, detects `POST /tune/snapshot`, and reads `delphi-spec-pass-v1` observations while the model stays resident. Cumulative snapshots never reset the trace buffer. A point with dropped events, failed logging or unmatched trace markers is rejected. The next point restarts safely when estimated remaining trace capacity is too small. If the snapshot endpoint returns 404, the tuner uses the existing four-decimal `DELPHI_SPEC_PLOG` and fresh launches per context point. Launch/time limits and the confirmation reserve still apply.

`speculation.text` defaults to `prefill.text`. Defaults are three distinct 128-token text prompts, 64 output tokens, maximum draft 6 (clipped to launch capacity), three initial cycles omitted and at least three settled cycles per width/request. `trace_max_events` defaults to 33554432. Widths run in a fixed shuffled order, with a warmup request after each policy change. Fixed synthetic acceptance supplies performance costs only; its output is invalid. T(w) uses complete logical-pass wall intervals, including draft ingest, target verification, checkpoint replay and server overhead. Device times are not added. Width 1 keeps MTP enabled and measures actual empty drafts at p_min=1; passes that still return a token are omitted. A confidence cut pays for its discarded step; n_min can discard the whole draft.

The real capture uses fixed maximum depth, p_min=0, n_min=0 and adaptation off. Request/sequence/pass IDs select only the current point's requests from cumulative logs and correlate their timings with binary trace markers. Full float probabilities are preserved. Initial unsettled and final boundary passes are omitted. Replay clips accepted prefixes and simulates fixed or bucket depth, resetting bucket state per request. Candidate domains respect the registry filters and launch capacity. Cost/conf controllers and DSpark/block-drafter rankings remain unavailable: DSpark confidence comes from its learned head, and its physical block cost cannot be inferred from MTP width costs. Structured DSpark observations can be parsed without treating that confidence as a vocabulary probability.

With live snapshots, `validation_reps` (default 3) independently measures the launch policy, source policy and the top `validation_candidates` alternatives (default 2), reversing policy order on alternate repetitions. The measured gains against the launch policy select the confirmation candidate; the gain is applied to the recipe baseline as an estimate, without substituting raw-text rates for recipe rates. The prompts, output lengths, normal acceptance, warmup and context match the capture. Validation checks identical greedy output tokens, reports settled pass throughput and full-request decode throughput separately, and records per-prompt prediction errors, including timing error with observed acceptance and pass boundaries held fixed. These are fresh requests on the resident process. The deployment confirmation stage still uses fresh processes and the recipe workload; replay estimates alone cannot recommend a deployment. Legacy servers have only the original capture diagnostic.

Shorter drafts change future pass boundaries and draft distributions, so replay remains an empirical estimate. Settled width medians omit graph rebuild and depth-switch costs. The report distinguishes capture rates, matched validation rates and recipe throughput. MTP runs on the last active layer-split GPU; preserve device order and cache state when comparing results. The launch policy is restored after collection, including on errors, before normal recipe and prefill measurements.

Artifacts include `source.json`, request settings/restoration, cumulative observation JSONL and binary trace under the process launch directory, per-point `snapshot.json`, `capture.json`, synthetic request windows, `pass-cost.json`, `validation.json` and `replay.json`. Capture and cost files carry the same point identity. Recalculate policies without a server:

```bash
python3 tools/auto-tune/speculation.py /path/capture.json /path/pass-cost.json \
  --policy '{"n_max":4,"p_min":0.5}' \
  --policy '{"n_max":4,"adaptive":"bucket","n_min_adaptive":1}' \
  --output /tmp/replay.json
```
