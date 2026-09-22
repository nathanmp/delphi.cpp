#!/usr/bin/env python3
"""Send one deterministic, exact-token prefill request to llama-server.

This is a small client for profile.py recipes. It obtains token IDs from the
running server, then supplies exactly the requested number of IDs to the
native /completion endpoint. The server's timing fields therefore describe a
long prompt without depending on a local tokenizer or an approximate word
count.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


CORPUS = """\
The following source fragment is deterministic benchmark context. It models a
mixed technical prompt with prose, identifiers, numbers, punctuation, and
short code blocks. The benchmark measures execution cost only; it does not
judge the generated answer.

def update_state(previous, item):
    total = previous.total + item.value
    return {"count": previous.count + 1, "total": total, "mean": total / (previous.count + 1)}

For each request, preserve data dependencies, validate bounds, and report a
concise result. A production implementation must handle retries, timeouts,
and input that crosses a logical batch boundary.
"""


def _post(url: str, path: str, payload: Any, timeout_s: float) -> Any:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def _tokens(url: str, target_tokens: int, timeout_s: float) -> list[int]:
    # Start near the likely number of tokens and grow geometrically. Token IDs
    # are used for the final request, so a tokenizer boundary cannot change its
    # requested length.
    repeats = max(1, target_tokens // 80)
    while True:
        response = _post(url, "/tokenize", {"content": CORPUS * repeats, "add_special": False}, timeout_s)
        tokens = response.get("tokens") if isinstance(response, dict) else None
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise RuntimeError("/tokenize did not return integer token IDs")
        if len(tokens) >= target_tokens:
            return tokens[:target_tokens]
        repeats *= 2


def run(url: str, target_tokens: int, timeout_s: float) -> dict[str, Any]:
    input_tokens = _tokens(url, target_tokens, timeout_s)
    started = time.monotonic()
    response = _post(url, "/completion", {
        "prompt": input_tokens,
        "n_predict": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }, timeout_s)
    latency_s = time.monotonic() - started
    timings = response.get("timings", {}) if isinstance(response, dict) else {}
    if not isinstance(timings, dict):
        timings = {}
    prompt_ms = timings.get("prompt_ms")
    prompt_tokens = timings.get("prompt_n")
    prompt_t_s = timings.get("prompt_per_second")
    if not isinstance(prompt_t_s, (int, float)) and isinstance(prompt_ms, (int, float)) and prompt_ms > 0 and isinstance(prompt_tokens, int):
        prompt_t_s = prompt_tokens / (prompt_ms / 1000.0)
    return {
        "schema": "delphi-prefill-client-v1",
        "target_tokens": target_tokens,
        "input_tokens": len(input_tokens),
        "prompt_tokens": prompt_tokens,
        "prompt_ms": prompt_ms,
        "prompt_t_s": prompt_t_s,
        "latency_s": latency_s,
        "predicted_tokens": timings.get("predicted_n"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--tokens", type=int, default=4096, help="exact number of input token IDs (default: 4096)")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    try:
        result = run(args.url, args.tokens, args.timeout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"prefill client: {error}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
