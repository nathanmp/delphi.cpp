"""Objective of the auto-tuner.

score = product over metrics of (metric / reference metric) ^ weight, with the weights normalized to sum 1. Decode is client
t/s at real acceptance; prefill is time to first token (TTFT, smaller is better) per prompt length. The score is 1 at the
reference; 1.05 means a 5 % better weighted geometric mean. A config that broke a constraint has no score.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

TRADEOFF_DECODE_SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)


def aggregate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Use the median of full workload repeats; keep the samples for racing and reports."""
    out = dict(samples[-1])
    def median(values):
        return statistics.median(values) if all(v is not None and v > 0 for v in values) else None
    out['decode_t_s'] = median([s.get('decode_t_s') for s in samples])
    out['prefill'] = {length: {key: median([s.get('prefill', {}).get(length, {}).get(key) for s in samples])
                               for key in ('ttft_s', 't_s', 'prompt_ms')}
                      for length in samples[-1].get('prefill', {})}
    out['samples'] = list(samples)
    return out


def race(samples: Sequence[Mapping[str, Any]], incumbent: Mapping[str, Any], weights: Mapping[str, float], margin: float) -> dict[str, Any]:
    """Stop a point below the incumbent by more than three repeat standard deviations in log score."""
    repeats = incumbent.get('samples', [])
    values = [score(s, incumbent, weights) for s in repeats]
    logs = [math.log(v) for v in values if v is not None and v > 0]
    noise = statistics.stdev(logs) if len(logs) > 1 else None
    relative = score(aggregate(samples), incumbent, weights)
    threshold = max(3 * noise, math.log1p(margin)) if noise is not None else None
    return {'stopped': relative is not None and threshold is not None and math.log(relative) < -threshold,
            'relative_score': relative, 'noise_log': noise, 'threshold_log': threshold, 'repeats': len(samples)}


def normalize(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in weights.values())
    if total <= 0 or any(float(v) < 0 for v in weights.values()):
        raise ValueError(f"weights must be non-negative with a positive sum: {dict(weights)}")
    return {k: float(v) / total for k, v in weights.items()}


def metric_weights(weights: Mapping[str, float], lengths: Mapping[str, float]) -> dict[str, float]:
    """Weights of the individual metrics: 'decode' and 'ttft:<length>'."""
    top = normalize({"prefill": weights.get("prefill", 0.0), "decode": weights.get("decode", 0.0)})
    out = {"decode": top["decode"]}
    if top["prefill"] > 0:
        for length, share in normalize(lengths).items():
            out[f"ttft:{length}"] = top["prefill"] * share
    return {k: v for k, v in out.items() if v > 0}


def ratios(metrics: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    """Each metric relative to the reference, oriented so that larger is better."""
    out = {}
    decode, ref_decode = metrics.get("decode_t_s"), reference.get("decode_t_s")
    if decode and ref_decode:
        out["decode"] = decode / ref_decode
    for length, item in (metrics.get("prefill") or {}).items():
        ref = (reference.get("prefill") or {}).get(length) or {}
        if item.get("ttft_s") and ref.get("ttft_s"):
            out[f"ttft:{length}"] = ref["ttft_s"] / item["ttft_s"]
    return out


def score(metrics: Mapping[str, Any], reference: Mapping[str, Any], weights: Mapping[str, float]) -> float | None:
    """Weighted geometric mean of the ratios; None when a weighted metric is missing."""
    r = ratios(metrics, reference)
    total = 0.0
    for name, weight in weights.items():
        if name not in r or r[name] <= 0:
            return None
        total += weight * math.log(r[name])
    return math.exp(total)


def tradeoff(rows: Sequence[Mapping[str, Any]], reference: Mapping[str, Any], lengths: Mapping[str, float], shares: Sequence[float] = TRADEOFF_DECODE_SHARES) -> list[dict[str, Any]]:
    """The best feasible row for each decode share of the weight (the rest goes to prefill): options along the trade-off."""
    out = []
    for share in shares:
        w = metric_weights({"prefill": 1.0 - share, "decode": share}, lengths)
        best = None
        for row in rows:
            s = score(row["metrics"], reference, w)
            if s is not None and (best is None or s > best[0]):
                best = (s, row)
        if best:
            out.append({"decode_share": share, "score": best[0], "row": best[1]})
    return out
