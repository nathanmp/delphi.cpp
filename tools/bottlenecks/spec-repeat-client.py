#!/usr/bin/env python3
"""Repeat one SPEED-Bench prompt on a loaded llama-server and record per-request speculative counters.

A client for profile.py recipes (SPEED-Bench result schema). It reuses speed_bench.py to load the sample and send the
request, so the prompt and the request fields are the ones of every other SPEED-Bench arm. For each of --reps
sequential requests it adds what speed_bench.py does not keep:

- a SHA-256 of the output text and the completion token count;
- the deltas of the server's speculative counters across the request (start the server with --metrics): verify steps,
  draft tokens, accepted draft tokens and the accepted tokens by draft position;
- tokens per verify step.

The counters are server-wide, so the deltas are exact only for one request at a time (--concurrency 1, which this
client enforces by sending sequentially).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

_spec = importlib.util.spec_from_file_location(
    "speed_bench", Path(__file__).resolve().parents[1] / "server" / "bench" / "speed-bench" / "speed_bench.py")
sb = importlib.util.module_from_spec(_spec)
sys.modules["speed_bench"] = sb
_spec.loader.exec_module(sb)

_COUNTERS = {
    "verify_steps": "llamacpp:spec_decode_num_drafts_total",
    "draft_tokens": "llamacpp:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "llamacpp:spec_decode_num_accepted_tokens_total",
}
_POS = re.compile(r'^llamacpp:spec_decode_num_accepted_tokens_per_pos_total\{position="(\d+)"\}\s+(\S+)$')


def scrape(base_url: str) -> dict[str, Any]:
    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    with urllib.request.urlopen(root + "/metrics", timeout=30) as response:
        text = response.read().decode()
    values: dict[str, Any] = {}
    per_pos: dict[int, float] = {}
    for line in text.splitlines():
        match = _POS.match(line)
        if match:
            per_pos[int(match.group(1))] = float(match.group(2))
            continue
        for key, name in _COUNTERS.items():
            if line.startswith(name + " "):
                values[key] = float(line.split()[1])
    values["accepted_by_position"] = [per_pos[i] for i in sorted(per_pos)]
    return values


def delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {key: int(after.get(key, 0) - before.get(key, 0)) for key in _COUNTERS}
    a, b = after.get("accepted_by_position", []), before.get("accepted_by_position", [])
    out["accepted_by_position"] = [int(a[i] - (b[i] if i < len(b) else 0)) for i in range(len(a))]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--bench", default="qualitative")
    parser.add_argument("--category", default="coding")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--osl", type=int, default=512)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--extra-inputs", default='{"temperature":0,"ignore_eos":true}')
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    base_url = sb.normalize_base_url(args.url)
    endpoint = base_url + "/chat/completions"
    extra = sb.parse_extra_inputs(args.extra_inputs)
    args.extra_inputs = extra
    args.concurrency = 1
    samples = sb.load_samples(args)
    sample = samples[0]
    if len(sample.turns) != 1:
        print(f"spec-repeat-client: {sample.id} has {len(sample.turns)} turns; one expected", file=sys.stderr)
        return 2

    results, extras = [], []
    for rep in range(args.reps):
        before = scrape(base_url)
        messages = [{"role": "user", "content": sample.turns[0]}]
        try:
            data, latency_s, start_ns, end_ns = sb.run_request(endpoint, args.model, messages, args.osl, extra, args.timeout)
        except Exception as exc:  # recorded like speed_bench does; the recipe sees ok=False
            results.append(sb.RequestResult(sample.id, sample.category, False, 1, 0.0, 0, 0, 0, None, 0, 0, None, None, None, None, [], str(exc)))
            extras.append({"rep": rep, "error": str(exc)})
            break
        after = scrape(base_url)
        usage, timings, finish, text = sb.parse_completion_response(data)
        # GLM answers in reasoning_content first: hash both, in that order, or an all-reasoning output hashes as empty
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        text = (message.get("reasoning_content") or "") + "\x00" + text
        n_completion = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
        n_prompt = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
        counters = delta(before, after)
        steps = counters["verify_steps"]
        results.append(sb.RequestResult(
            id=sample.id, category=sample.category, ok=True, turns=1, latency_s=latency_s, prompt_tokens=n_prompt,
            completion_tokens=n_completion, total_tokens=int(usage.get("total_tokens") or n_prompt + n_completion),
            finish_reason=finish, draft_n=int(timings.get("draft_n") or 0), draft_n_accepted=int(timings.get("draft_n_accepted") or 0),
            prompt_ms=float(timings["prompt_ms"]) if timings.get("prompt_ms") else None,
            predicted_ms=float(timings["predicted_ms"]) if timings.get("predicted_ms") else None,
            prompt_per_second=timings.get("prompt_per_second"), predicted_per_second=timings.get("predicted_per_second"),
            request_windows_raw_ns=[[start_ns, end_ns]], error=None))
        extras.append({
            "rep": rep,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "output_chars": len(text) - 1,  # without the separator
            "completion_tokens": n_completion,
            "wall_latency_s": latency_s,
            "client_tokens_per_s": n_completion / latency_s if latency_s > 0 else None,
            "spec": counters,
            # tokens per verify step = emitted tokens / verify steps (steps without a draft are not counted by the server)
            "tokens_per_verify_step": (counters["accepted_tokens"] + steps) / steps if steps else None,
        })
        print(f"rep {rep}: {n_completion} tokens, {latency_s:.2f} s, sha {extras[-1]['output_sha256'][:12]}, "
              f"{counters['verify_steps']} verify steps, accepted by position {counters['accepted_by_position']}", flush=True)

    summary = [sb.summarize_group(sample.category, results), sb.summarize_group("overall", results)]
    payload = {
        "config": {"url": args.url, "bench": args.bench, "category": args.category, "osl": args.osl, "reps": args.reps,
                   "extra_inputs": extra, "sample_id": sample.id},
        "selected_samples": 1,
        "completed_samples": sum(1 for r in results if r.ok),
        "failed_samples": sum(1 for r in results if not r.ok),
        "summary": summary,
        "results": [asdict(r) for r in results],
        "repeat": extras,
        "output_hashes_equal": len({e.get("output_sha256") for e in extras if "output_sha256" in e}) == 1,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(r.ok for r in results) and len(results) == args.reps else 1


if __name__ == "__main__":
    raise SystemExit(main())
