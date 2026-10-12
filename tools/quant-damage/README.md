# llama-quant-damage

Estimates how much precision a tensor loses when it is quantized to each of a set of candidate types. It does not quantize the full tensor, does not run the model and does not load the model into memory.

The tool samples complete rows of the source tensor, runs the real ggml quantizer and dequantizer of every candidate type on each sampled row, and expands the row errors to the full tensor with a stratified survey estimator. It reports each estimate with a standard error.

The result is a proxy for downstream loss, not the loss itself. It is intended as the quality input of a future mixed-precision recipe optimizer.

## Metrics

For a tensor `W` with rows `i` and input channels `j`, a candidate type `q` and imatrix activations `a_j` (mean squared input activation per token):

- `D = sum_j a_j sum_i (W_ij - Q_q(W)_ij)^2`: activation-weighted distortion. Under a diagonal activation covariance, this is the expected squared error of the matrix output per token.
- `R = D / sum_j a_j sum_i W_ij^2`: `D` normalized by the weighted signal energy. Use it to compare tensors with different scales.
- `sse` and `rel_mse`: the same quantities with `a_j = 1`.

For merged expert tensors, each expert `e` has its own `a_ej` and a weight `w_e`. Three weightings are reported:

| weighting | `w_e` | meaning |
| --- | --- | --- |
| `mass` (default) | `count_e / n_tokens` | the imatrix sums: routing frequency and per-expert activation magnitude |
| `count` | `mass x mean_e' m_e' / m_e` | routing frequency only; `m_e` is the mean activation of expert `e` |
| `uniform` | `n_expert_used / n_expert` | every expert routed equally often |

With `mass`, `D` is the expected squared output error per token, summed over the routed experts.

An expert that received no calibration tokens is not given zero weight. It gets the smallest observed token count of the tensor and the mean channel profile of the other experts, and the output reports it under `experts_missing`. The quantizer still receives the same imatrix values as in `llama-quantize` (1.0 for an expert without tokens).

A tensor without an imatrix entry is reported with the note `no imatrix`, and its weighted values are the unweighted ones.

## Sampling

- **Sampling unit.** One complete row of one expert. Every ggml quantizer is row-local: a row quantized alone gives the same bytes as the same row inside the full tensor. Several quantizers are not block-local, however: Q4_0, Q4_1, Q5_0 and Q5_1 use whole-row statistics when an imatrix is supplied, and IQ*_KS store a per-row scale. `--check-locality` verifies both properties on real rows.
- **Row statistics.** By default the tool reads the tensor once without quantizing it. For every row it computes the weighted signal energy and a noise proxy, `sum_j a_j * max^2(|W| in the 32-wide block containing j)`. The error of a block quantizer scales with the square of the block maximum, so the proxy tracks the damage of every type. The signal energy does not: when one input channel holds most of the activation mass (common in early and late `ffn_down`), it follows the weight in that one column. The exact proxy totals serve as auxiliary totals for a ratio estimator of `D`, and `R` divides that estimate by the exact signal. `--no-row-stats` skips the read: strata are then per expert only, `D` uses the plain expansion estimator, and `R` is the ratio of the two estimates.
- **Strata.** With row statistics:
  - The `--min-per-expert` most important rows of every expert (default 1) are always measured, so no expert goes unmeasured, however rarely it is routed.
  - Rows whose importance is at least 1/`--certainty-factor` of the mean per sampled row are always measured. A row's importance is its noise proxy times its expert's routing weight. A few such rows can hold most of the damage of a type whose error on one extreme weight is erratic, such as MXFP4, IQ2_XXS or IQ3_S next to a massive activation.
  - The 1% of rows with the largest max/RMS ratio form a stratum of their own.
  - The remaining rows of the whole tensor are split into up to `--strata` levels of equal importance mass, which over-samples the rows expected to carry the most damage.

  `--expert-strata` stratifies by expert first instead. That gives many small strata, which weakens the variance estimate without reducing bias. Without row statistics, every expert is one stratum.
- **Rounds.** The first round takes `--samples` rows plus the certainty rows. Each later round doubles the sample. By default, new rows go to the strata by Neyman allocation under the noise-proxy model (`N_h x` RMS of the importance). That allocation is known before sampling, so it adds no bias. `--allocation observed` uses the observed residual spread instead, which biases the estimate when there are many small strata. Samples are nested: each stratum has a fixed seeded permutation (a keyed Feistel permutation when the stratum is a plain row range, so no row index is stored), and a round extends its prefix.
- **Stopping.** The tool stops when every candidate that is not dominated has a relative standard error <= `--rse`, or when the row budget (`--max-samples`, `--max-sample-mib`) runs out. `--rank-z` adds a second criterion: every adjacent pair of candidates is ordered at `|z| >= X`, by a paired difference on the same rows. `--fixed` disables early stopping. Types with rounding-level damage (R < 1e-9, e.g. BF16 -> F16) do not take part in allocation or stopping.
- **Dominance.** A candidate that is no smaller than another and more damaging at `z > --dominance-z` (paired) is marked `dominated_by`.

Each sampled row is read once, and all candidate types are evaluated on it before it is discarded. Memory is bounded by the per-row statistics and row lists (about 30 bytes per row, none without row statistics) and a 256 MiB batch buffer.

## Usage

```
llama-quant-damage -m model-bf16.gguf --imatrix imatrix.gguf \
    --tensor 'blk\.32\.ffn_down_exps' --qtypes q3_K,iq3_k,q4_K,iq4_k,q5_K \
    --json out.json --csv out.csv
```

- `--tensor` and `--exclude` take regexes. Tensors with fewer than `--min-elements` elements (default 1M) and 1D tensors are skipped.
- A split model is opened by its first file.
- The source may be any type with a dequantizer. A quantized source is used as the reference, so its own error is not included.
- `--exact` also quantizes every row for exact values, and `--repeat N` runs N seeds that share one row-statistics scan. `quant-damage-validate.py` compares the two and prints the error, CI coverage and ranking tables.
- The CSV has one record per (tensor, run, qtype). The JSON also contains the per-round convergence (`rounds`) and, with `--per-expert`, the sampled per-expert `R`.

## Machine-readable results

The `--json` output is the maintained interface for the auto-quantizer. It is versioned by `schema` (`delphi.quant-damage/1`) and every estimate record is keyed by:

| key | JSON path | meaning |
| --- | --- | --- |
| model identity | `identity.model.metadata_hash` | FNV-1a over the raw GGUF metadata bytes of every file, plus `file_bytes` and `n_files`. Path-independent; changes if any metadata or tensor layout changes. |
| tensor identity | `tensors[].identity` | name, shape, source dtype and the GGUF location (`split`, `data_offset`, `data_bytes`). |
| encoder variant | `identity.encoder` and each run's `encoder` | `ggml_quantize_chunk` today. An encoder variant that changes bytes (for example activation feedback) must report a different string so its records are never mixed with ordinary ones. |
| qtype | `types[].qtype` | candidate type; `status` and `note` give the support or rejection reason (`requires_imatrix`, `incompatible_row_size`, `unsupported`). |
| calibration identity | `identity.calibration` | imatrix file hash and size, `kind` (gguf or legacy dat), the datasets and chunk metadata it stores, and the entry count. |
| seed | `runs[].seed` | the sampling seed. |

`identity.build` records the llama.cpp commit and build number.

Each run record carries the estimate (`R`, `R_se`, `R_ci_lo`, `R_ci_hi`, absolute `D` and its `D_se`, all weightings, `bytes` and `bpw`), the routing weight provenance (`tensors[].routing`: `counts_source` of the expert weights, per-expert counts, the three expert weight vectors, missing experts and imatrix coverage), and the cost (`bytes_scan`, `bytes_sample`, `seconds_scan`, `seconds_eval`, sampled units and strata). Memory and I/O are bounded: the identities hash the metadata region and the imatrix file, never tensor data, and no candidate model is written.

### Tied groups

The optimizer fixes gate and up to one qtype for fusion while their damages differ. `--tie NAME:REGEX` groups every selected tensor matching REGEX with its fused partner (the first `gate` in the name replaced by `up`; this follows the llama.cpp FFN naming `ffn_gate`/`ffn_up`, `ffn_gate_exps`/`ffn_up_exps`, `ffn_gate_shexp`/`ffn_up_shexp`). The partner joins the selection even when the regexes excluded it, because a tied qtype can only be chosen with both measured. `tie_groups[]` then holds, per member run (aligned by seed and estimator), one record per qtype: summed `bytes`, the summed `D` with the quadrature-summed `D_se` (members are sampled independently), the group `R = sum D / sum signal` with its interval, and every member's own `R`. The group `R` is a sum of per-tensor proxies, not a joint measurement, and member damages are never assumed equal. A qtype that any member rejects is rejected for the group, with the member named in `note`. `qd_tie_combine()` exposes the same aggregation to C++ callers.
