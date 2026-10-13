#!/usr/bin/env python3
"""Real-text prefill client: REPS requests of exactly TOKENS token IDs, each from a different slice of a text file, with one
generated token. Writes the delphi-prefill-client-v1 fields of the median request (latency_s is the time to first token
seen by the client) and every request."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path


def post(url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0, help="first token of the first slice")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ids = post(args.url, "/tokenize", {"content": args.text.read_text(errors="replace"), "add_special": False}, args.timeout)["tokens"]
    need = args.tokens * args.reps
    if len(ids) < need:
        print(f"prefill client: {args.text} has {len(ids)} tokens, need {need}", file=sys.stderr)
        return 1
    start = args.offset % max(1, len(ids) - need + 1)
    requests = []
    begin_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    for rep in range(args.reps):
        prompt = ids[start + rep * args.tokens:start + (rep + 1) * args.tokens]
        started = time.monotonic()
        response = post(args.url, "/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0.0, "ignore_eos": True,
                                                  "cache_prompt": False, "stream": False}, args.timeout)
        latency = time.monotonic() - started
        timings = response.get("timings", {})
        requests.append({"prompt_n": timings.get("prompt_n"), "prompt_ms": timings.get("prompt_ms"), "latency_s": latency})
    end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    prompt_ms = statistics.median(r["prompt_ms"] for r in requests)
    result = {
        "schema": "delphi-prefill-client-v1", "target_tokens": args.tokens, "input_tokens": args.tokens, "prompt_tokens": args.tokens,
        "prompt_ms": prompt_ms, "prompt_t_s": args.tokens / (prompt_ms / 1000.0),
        "latency_s": statistics.median(r["latency_s"] for r in requests),
        "request_start_raw_ns": begin_ns, "request_end_raw_ns": end_ns, "predicted_tokens": 1, "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
