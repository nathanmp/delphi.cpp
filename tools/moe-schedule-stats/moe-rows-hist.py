#!/usr/bin/env python3
"""Histogram of routed activation rows handled by each CPU MoE expert.

Runs any llama.cpp binary with GGML_MOE_PROF=1 (or parses a saved stderr log)
and renders one row per rows-per-expert bucket:

    rows/expert    fraction of active experts    fraction of MMID time

Each profiler sample is one (expert, MUL_MAT_ID node) with rows > 0, so every
layer/ubatch contributes its active experts. "Fraction of MMID time" is the
share of routed rows in the bucket: expert GEMM FLOPs scale with rows, while
fixed per-expert overhead (repack, dispatch) is not attributed, so small
buckets take somewhat more time than their rows share.

A decode histogram wants a decode-heavy workload: a tiny prompt with a long
generation, so prefill ubatches do not dominate the rows share. Example for a
CPU-offloaded MoE in its production placement:

    python3 tools/moe-schedule-stats/moe-rows-hist.py -- build/bin/llama-bench \
        --numa tensors -ngl 999 -ncmoe 999 --device ROCm3 \
        -m model.gguf -t 48 -p 32 -n 128

To re-render a saved run without re-executing the model:

    python3 tools/moe-schedule-stats/moe-rows-hist.py --input /tmp/run.stderr.log
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


BUCKET_RE = re.compile(r"^moe_prof_bucket lo=(\d+) hi=(\d+) count=(\d+) rows=(\d+)\s*$")
BEGIN_RE = re.compile(
    r"^moe_prof_begin samples=(\d+) rows=(\d+) mmid_thread_s=([0-9.]+)\s*$")
LEGACY_HIST_RE = re.compile(r"\[(\d+),(\d+)\)=([0-9.]+)%")

# Requested display buckets: (label, lo inclusive, hi exclusive or None).
DISPLAY_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 2),
    ("2-3", 2, 4),
    ("4-7", 4, 8),
    ("8-15", 8, 16),
    ("16-31", 16, 32),
    ("32+", 32, None),
)


def parse_new_format(text: str) -> tuple[list[tuple[int, int, int, int]], dict[str, str] | None]:
    buckets: list[tuple[int, int, int, int]] = []
    summary: dict[str, str] | None = None
    for line in text.splitlines():
        bucket = BUCKET_RE.match(line)
        if bucket:
            buckets.append(tuple(int(part) for part in bucket.groups()))  # type: ignore[assignment]
            continue
        begin = BEGIN_RE.match(line)
        if begin:
            summary = {"samples": begin.group(1), "rows": begin.group(2),
                       "mmid_thread_s": begin.group(3)}
    return buckets, summary


def parse_legacy_format(text: str) -> list[tuple[int, int, float]]:
    for line in text.splitlines():
        if "histogram:" in line:
            return [(int(lo), int(hi), float(share))
                    for lo, hi, share in LEGACY_HIST_RE.findall(line)]
    return []


def render(buckets: list[tuple[int, int, int, int]], summary: dict[str, str] | None) -> str:
    merged: dict[str, list[int]] = {label: [0, 0] for label, _, _ in DISPLAY_BUCKETS}
    for lo, _hi, count, rows in buckets:
        for label, dlo, dhi in DISPLAY_BUCKETS:
            if lo >= dlo and (dhi is None or lo < dhi):
                merged[label][0] += count
                merged[label][1] += rows
                break
    total_count = sum(count for count, _ in merged.values())
    total_rows = sum(rows for _, rows in merged.values())
    lines = ["rows/expert    fraction of active experts    fraction of MMID time"]
    for label, _, _ in DISPLAY_BUCKETS:
        count, rows = merged[label]
        count_share = 100.0 * count / total_count if total_count else 0.0
        rows_share = 100.0 * rows / total_rows if total_rows else 0.0
        lines.append(f"{label:<13}{count_share:5.1f}% ({count})"
                     f"               {rows_share:5.1f}% ({rows} rows)")
    if summary is not None:
        mean = float(summary["rows"]) / float(summary["samples"]) if int(summary["samples"]) else 0.0
        lines.append(f"# {summary['samples']} active-expert samples, "
                     f"{summary['rows']} routed rows, mean {mean:.1f} rows/expert, "
                     f"{summary['mmid_thread_s']} thread-s in MUL_MAT_ID")
        lines.append("# MMID time is the rows share: GEMM FLOPs scale with rows; "
                     "fixed per-expert overhead is not attributed.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        help="parse a saved stderr log instead of running a command")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command to run after -- (inherits stdout; stderr is parsed)")
    args = parser.parse_args()
    if args.input is not None:
        text = args.input.read_text(errors="replace")
    else:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("give a command after -- or --input PATH")
        env = dict(os.environ)
        env.setdefault("GGML_MOE_PROF", "1")
        try:
            completed = subprocess.run(command, env=env, stderr=subprocess.PIPE, text=True)
        except OSError as error:
            print(f"moe-rows-hist: cannot run {command[0]}: {error}", file=sys.stderr)
            return 1
        text = completed.stderr
        if completed.returncode != 0:
            print(f"moe-rows-hist: command exited with status {completed.returncode}; "
                  "parsing its stderr anyway", file=sys.stderr)
    buckets, summary = parse_new_format(text)
    if buckets:
        sys.stdout.write(render(buckets, summary))
        return 0
    legacy = parse_legacy_format(text)
    if legacy:
        print("moe-rows-hist: no moe_prof_bucket lines; rows shares need a binary with "
              "the per-bucket rows extension. Showing expert shares only.", file=sys.stderr)
        # Legacy buckets use the same power-of-two scheme, so los align with the
        # display buckets; los >= 32 merge into "32+".
        by_lo = {lo: share for lo, _hi, share in legacy}
        merged = {label: 0.0 for label, _, _ in DISPLAY_BUCKETS}
        for lo, share in by_lo.items():
            for label, dlo, dhi in DISPLAY_BUCKETS:
                if lo >= dlo and (dhi is None or lo < dhi):
                    merged[label] += share
                    break
        lines = ["rows/expert    fraction of active experts    fraction of MMID time"]
        for label, _, _ in DISPLAY_BUCKETS:
            lines.append(f"{label:<13}{merged[label]:5.1f}%"
                         "               n/a (rebuild for rows)")
        sys.stdout.write("\n".join(lines) + "\n")
        return 0
    print("moe-rows-hist: no GGML_MOE_PROF histogram found in the log. "
          "The run must execute a CPU MUL_MAT_ID path with GGML_MOE_PROF=1.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
