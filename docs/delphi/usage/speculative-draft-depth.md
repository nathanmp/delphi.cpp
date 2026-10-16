# Draft depth of MTP speculative decoding

## Default: confidence stop (conf mode)

With `--spec-type draft-mtp`, the draft depth is chosen in each round: the MTP head drafts token after token and stops at the first token whose calibrated chance of acceptance no longer pays for its verify row. The rule keeps draft token k while

```
q_k = prod_{i <= k} cal(i, p_i)  >=  rate * (t_verify(k + 2) - t_verify(k + 1) + t_step)
```

- `p_i` is the top-1 probability of the draft at step i; `cal(i, p)` is the observed acceptance of draft position i (1st, 2nd, 3rd and later) in bins of `p`, learned online with `p` itself as the prior.
- `rate` is the measured throughput (tokens per second of the recent rounds), `t_verify(n)` the verify time of n tokens (calibrated when the server starts), `t_step` the measured time of one draft step.
- The depth is at most `--spec-draft-n-max` (default 7 in this mode) and at least 1.

The mode also reserves a graph per verify width (`LLAMA_GRAPH_ALT_SLOTS`) and grows the KV views in steps of 2048 cells, so that the graphs of idle widths stay valid (see `environment-variables.md`, section "Speculative decoding").

| option | effect |
| --- | --- |
| (none) | conf mode, up to 7 draft tokens |
| `--spec-draft-n-max N` | conf mode, up to N draft tokens |
| `--no-spec-draft-adaptive` | fixed depth `--spec-draft-n-max` (default 3), upstream behavior |
| `--spec-draft-adaptive-mode bucket` | the acceptance-bucket controller (`--spec-draft-n-min-adaptive` sets its floor) |
| `--spec-synth-len`, `--spec-synth-rates` | synthetic acceptance: fixed depth, the conf mode is not used |

The conf mode supports only `draft-mtp`. `cost` (per-request depth from a cost model) is also available but did not beat a fixed depth on GLM or Qwen.

## Changes at run time

`POST /speculative` with `{"args": ["--spec-draft-n-max", "5", "--no-spec-draft-adaptive"]}` applies speculative options over those of the launch without a reload (the draft cannot be larger than at the launch). `GET /speculative` returns the active settings, the graph build counts and, for the conf mode, the verify times, the step time, the throughput and the calibration table.

## Probabilistic draft sampling

`--spec-draft-sampling probabilistic` makes `draft-simple` and `draft-mtp` sample each draft token from the draft distribution q, instead of taking its top-1 token (`greedy`, the default). The target then accepts a draft token x with probability min(1, p(x)/q(x)). After a rejection it samples from norm(max(0, p - q)). The output distribution of the target stays exact. At temperature above 0 this accepts more draft tokens than greedy drafting. At temperature 0 both modes give the same text, because the option has no effect there. Requests with a grammar or a JSON schema also work. The option can be changed at run time with `POST /speculative` (`{"args": ["--spec-draft-sampling", "probabilistic"]}`), and `GET /speculative` shows it as `draft_sampling`.

The rule that stops a draft must not depend on the value of the token that it drops, or the output is not exact. For this reason the conf mode and `--spec-draft-p-min` use the top-1 probability of the draft, also with probabilistic sampling. For the same reason `--spec-draft-n-min` does not cut a probabilistic draft (its default is 0).

Measured on GLM-5.3 Flash INT4 (3x CMP 170HX), temperature 1.0, seeded, 8 prompts, 512 tokens, 2 rounds:

| arm | tokens per second | accepted / drafted |
| --- | --- | --- |
| greedy, fixed-3 | 70.6 | 0.515 |
| probabilistic, fixed-3 | 73.7 | 0.552 |
| greedy, conf | 72.4 | 0.687 |
| probabilistic, conf | 71.9 | 0.674 |

With a fixed depth, probabilistic drafting gives more accepted tokens. In the conf mode it gave no gain in this test: the calibration is by the top-1 probability, and a sampled token is accepted less often than the top-1 token.

## Measurements

Greedy, 8 prompts (prose, math, code, JSON, explanation, C++, travel, SQL), 2 rounds, total tokens per second; fixed-3 is the upstream default depth.

| model and devices | output tokens | conf (default) | fixed-3 | bucket, n_max 6 |
| --- | --- | --- | --- | --- |
| GLM-5.3 Flash INT4, 3x CMP 170HX | 2048 | 73.4 | 71.6 (+2.5 %) | 73.0 |
| GLM-5.3 Flash INT4, 3x CMP 170HX | 512 | 74.9 | 73.8 (+1.6 %) | 73.8 |
| Qwen 3.8 Flash Next, 3x R9700 | 2048 | 69.7 | 67.0 (+4.0 %) | |
| Qwen 3.8 Flash Next, 3x R9700 | 512 | 67.5 | 68.9 (-2.1 %) | |

An earlier GLM launch pair gave +4.3 % (2048) and +2.7 % (512). At temperature 1.0 on GLM, conf was +2.8 % over fixed-3 (512 tokens).

## Limitations

- The depth depends on measured times and the verify width changes the numerics, so outputs with the conf mode are not repeatable bit for bit, also with greedy sampling. Use `--no-spec-draft-adaptive` for reproducible runs.
- On Qwen 3.8 (HIP) a round after a change of depth costs about 3 ms more than a round at the same depth (about 0.7 ms on GLM with CUDA); the rule does not include this, and short outputs lose about 2 %. A hysteresis term for it locked the depth at the maximum on long outputs and was removed.
- The calibration only sees tokens that were verified; tokens below the stop threshold keep the prior. A replay with all positions observed gained 0.1 %.
- The KV step 2048 applies to every KV cache of the process.
