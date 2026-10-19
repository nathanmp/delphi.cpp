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
| `speculation` | opt-in MTP replay; see below for measurements and current server requirements |
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

Set `speculation.enabled: true` to collect MTP acceptance and T(w) at each context point. The existing `DELPHI_SPEC_PLOG` hook is reused. One process measures every width through `POST /speculative`; replaying policies adds no launches or context re-creates. Because the trace is written at process exit, this mode currently uses a fresh process for each loop-2 point. It is opt-in until a live trace-flush hook is available. Launch/time limits and the confirmation reserve still apply.

`speculation.text` defaults to `prefill.text`. Defaults are three distinct real-text prompts of 128 tokens, 64 output tokens, a maximum draft of 6 (clipped to the launch output capacity), three initial cycles omitted per request, and at least three measured cycles per width/request. `trace_max_events` defaults to 8388608; a truncated trace or dropped events rejects the point. The width order is shuffled with a fixed seed. Server and recipe warmup stay enabled. Each width uses fixed synthetic acceptance; its generated output is invalid and is never quality evidence. T(w) is the median verify-to-verify host wall interval within a request, including draft work and synchronization. Device times are not added. A confidence cut also pays for the discarded draft step; n_min can discard the whole draft.

The capture uses greedy MTP, fixed maximum depth, p_min=0, n_min=0, and normal acceptance. Separate requests supply boundaries absent from the log. Short final passes are omitted. Replay clips the accepted prefix and simulates fixed or bucket depth, with bucket state reset for every request. Candidate domains come from the request knobs in the registry and respect `only`, `exclude`, and `levels`. Cost/conf controllers are not replayed: they need timing, shadow-token or calibration state that the current log does not contain. The best decode policy for a context also dominates the other policies at every positive decode weight because their prefill measurements are identical.

These are empirical pass estimates. Shorter drafts change future pass boundaries, the log rounds probabilities to four decimals, and settled T(w) omits graph rebuild/depth-switch costs. The log does not contain tokens, so this cannot be an exact token replay. Predicted rows are labeled `replay-estimate`; `best.json` requires fresh normal-acceptance confirmation before it can recommend them. The original policy is restored after synthetic collection, including on errors, before the recipe's real client and prefill measurements run.

DSpark and other implementations have no per-pass probability log. Their normal measurements continue with an explicit unavailable reason, and their request policy stays fixed. The report requests a shared log with request/sequence/pass IDs, tokens, full-precision probabilities, accepted length and checkpoint-replay markers, plus a live trace-flush hook. No C++ changes are made by this milestone.

Artifacts include `source.json` (point identity and tokenized prompts), request settings and restoration, one raw acceptance JSONL per prompt, `capture.json`, synthetic request windows, `loop3.trace`, `pass-cost.json`, the existing host execution summary, and `replay.json`. Capture and cost files carry the same point identity. Recalculate policies without a server:

```bash
python3 tools/auto-tune/speculation.py /path/capture.json /path/pass-cost.json \
  --policy '{"n_max":4,"p_min":0.5}' \
  --policy '{"n_max":4,"adaptive":"bucket","n_min_adaptive":1}' \
  --output /tmp/replay.json
```
