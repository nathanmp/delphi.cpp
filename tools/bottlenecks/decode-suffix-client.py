#!/usr/bin/env python3
"""Measure decode speed after a long prompt on llama-server.

This is a small client for profile.py recipes. It sends exact-length token IDs
(as prefill-client.py does) and then repeats a fixed-length greedy decode on
top of that prompt. The server's own timing fields separate the prompt phase
from the decode suffix, so only `predicted_*` describes the suffix.

With --reuse the setup request is sent once with prompt caching and every
measured request extends that prompt by a few tokens, so the server only has to
evaluate the suffix when it holds restorable state for the prefix. Without
--reuse every request evaluates the whole prompt (use --no-cache-prompt).

The result uses the SPEED-Bench result schema, so profile.py reads it as a
decode workload.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_spec = importlib.util.spec_from_file_location("prefill_client", Path(__file__).with_name("prefill-client.py"))
_prefill = importlib.util.module_from_spec(_spec)
sys.modules["prefill_client"] = _prefill
_spec.loader.exec_module(_prefill)


def _complete(url: str, tokens: list[int], n_predict: int, cache_prompt: bool, timeout_s: float) -> dict[str, Any]:
    start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    response = _prefill._post(url, "/completion", {
        "prompt": tokens,
        "n_predict": n_predict,
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": cache_prompt,
        "stream": False,
        "return_tokens": True,
    }, timeout_s)
    end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return {"response": response, "window": [start_ns, end_ns]}


def run(url: str, prompt_tokens: int, n_predict: int, reps: int, reuse: bool, suffix_tokens: int, timeout_s: float) -> dict[str, Any]:
    tokens = _prefill._tokens(url, prompt_tokens + suffix_tokens, timeout_s)
    base = tokens[:prompt_tokens]
    results = []
    if reuse:
        # one setup request evaluates the prompt; it is not a measured sample
        _complete(url, base, 1, True, timeout_s)
    for rep in range(reps + 1):  # the first measured request warms the graph shapes of this depth and is dropped
        prompt = tokens[:prompt_tokens + suffix_tokens] if reuse else base
        done = _complete(url, prompt, n_predict, reuse, timeout_s)
        timings = done["response"].get("timings", {})
        generated = done["response"].get("tokens") or []
        results.append({
            "ok": True,
            "warmup": rep == 0,
            "prompt_tokens": timings.get("prompt_n"),
            "cache_tokens": timings.get("cache_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "prompt_t_s": timings.get("prompt_per_second"),
            "completion_tokens": timings.get("predicted_n"),
            "predicted_ms": timings.get("predicted_ms"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "draft_n": timings.get("draft_n") or 0,
            "draft_n_accepted": timings.get("draft_n_accepted") or 0,
            "request_windows_raw_ns": [done["window"]],
            "generated_tokens": generated,
        })
    measured = results[1:]
    speeds = [r["predicted_per_second"] for r in measured if r["predicted_per_second"]]
    draft_n = sum(r["draft_n"] for r in measured)
    accepted = sum(r["draft_n_accepted"] for r in measured)
    return {
        "schema": "delphi-decode-suffix-client-v1",
        "prompt_tokens": prompt_tokens,
        "suffix_tokens": suffix_tokens if reuse else 0,
        "n_predict": n_predict,
        "reuse": reuse,
        "completed_samples": len(measured),
        "failed_samples": 0,
        "summary": {"overall": {
            "category": "overall",
            "avg_prompt_t_s": statistics.mean([r["prompt_t_s"] for r in measured if r["prompt_t_s"]] or [0.0]),
            "avg_pred_t_s": statistics.mean(speeds) if speeds else None,
            "min_pred_t_s": min(speeds) if speeds else None,
            "max_pred_t_s": max(speeds) if speeds else None,
            "avg_latency": statistics.mean((r["request_windows_raw_ns"][0][1] - r["request_windows_raw_ns"][0][0]) / 1e9 for r in measured),
            "draft_n": draft_n,
            "accept_rate": (accepted / draft_n) if draft_n > 0 else None,
        }},
        "results": measured,
        "warmup_result": results[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=32768)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3, help="measured repetitions after one dropped warm-up request")
    parser.add_argument("--reuse", action="store_true", help="prefill once, then extend the cached prompt by --suffix-tokens per request")
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.url, args.prompt_tokens, args.n_predict, args.reps, args.reuse, args.suffix_tokens, args.timeout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"decode suffix client: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
