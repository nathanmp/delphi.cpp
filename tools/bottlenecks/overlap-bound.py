#!/usr/bin/env python3
"""Amdahl-style CPU/device overlap bound from a GGML execution trace.

For each execution kind/phase in each given trace, aggregates over executions:
  W = end-to-end host wall, C = CPU-busy union, D = device-busy union,
  U = wall minus union(all host spans), O = current C/D overlap.
Upper bound for ANY scheme that overlaps CPU and device work (in-ubatch or
cross-ubatch wavefront): the smaller side can at most hide fully inside the
larger, so W' >= W - min(C, D) and speedup <= W / (W - min(C, D)).

It also decomposes neither-time (wall covered by neither C nor D) by the
host-span kinds covering it: that is what survives even perfect C/D overlap,
modulo waits collapsing once the work they wait for overlaps.

Span classification and device-interval placement reuse profile.py, so the
numbers match the bottleneck report rows. Example:

    python3 tools/bottlenecks/overlap-bound.py trace1.trace trace2.trace
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bottlenecks.profile import (  # noqa: E402
    ProfileError,
    _load_trace_dump,
    _selected_execution_ids,
)

TRACE = _load_trace_dump()

HOST_KINDS = {
    "submit": (TRACE.K_BACKEND_SUBMIT,),
    "cpu": (TRACE.K_CPU_COMPUTE,),
    "wait": (TRACE.K_BACKEND_WAIT,),
    "copy_handling": (TRACE.K_COPY_HANDLING,),
    "copy_waits": (TRACE.K_COPY_WAIT_SRC, TRACE.K_COPY_WAIT_DST),
    "copy_body": (TRACE.K_COPY_BODY,),
}
ALL_HOST = tuple(kind for kinds in HOST_KINDS.values() for kind in kinds)


def union_ns(intervals):
    total = 0
    pos = None
    end = None
    for start, finish in sorted(intervals):
        if pos is None or start > end:
            if pos is not None:
                total += end - pos
            pos, end = start, finish
        else:
            end = max(end, finish)
    if pos is not None:
        total += end - pos
    return total


def subtract_ns(intervals, cut):
    """Remove cut intervals from intervals (lists of [start, end))."""
    out = []
    cuts = sorted(cut)
    for start, end in sorted(intervals):
        pos = start
        for cstart, cend in cuts:
            if cend <= pos:
                continue
            if cstart >= end:
                break
            if cstart > pos:
                out.append([pos, min(cstart, end)])
            pos = max(pos, cend)
            if pos >= end:
                break
        if pos < end:
            out.append([pos, end])
    return out


def analyze(path: Path):
    trace = TRACE.read_trace(path)
    if trace["flags"] & 1 or trace["trace_stats"].get("dropped_event_count", 0):
        raise ProfileError(f"trace is truncated or dropped events: {path}")
    events = trace["events"]
    selected, info = _selected_execution_ids(events)
    host: dict[int, list] = defaultdict(list)
    cpu: dict[int, list] = defaultdict(list)
    by_kind: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        eid = event["execution_id"]
        if eid not in selected:
            continue
        kind = event["kind"]
        if kind in ALL_HOST:
            interval = [event["timestamp_ns"], event["args"][0]]
            host[eid].append(interval)
            if kind == TRACE.K_CPU_COMPUTE:
                cpu[eid].append(interval)
            for name, kinds in HOST_KINDS.items():
                if kind in kinds:
                    by_kind[name][eid].append(interval)
    spans, _epochs = TRACE.gpu_timing_model(events)
    device: dict[int, list] = defaultdict(list)
    for span in spans:
        event = span["event"]
        eid = event["execution_id"]
        if eid not in selected or span["status"] != "resolved":
            continue
        if span["scope"] in ("eager_submit", "graph_launch"):
            device[eid].append([span["device_start_ns"], span["device_end_ns"]])
    by_exec_kind: dict[str, list[int]] = defaultdict(list)
    for eid in selected:
        by_exec_kind[f'{info[eid]["kind"]}/{info[eid]["phase"]}'].append(eid)
    result = {}
    for kind, ids in by_exec_kind.items():
        wall = cpu_ns = dev_ns = unaccounted = overlap = 0
        neither_by_host: dict[str, int] = defaultdict(int)
        tokens = 0
        for eid in ids:
            bounds = [info[eid]["start_ns"],
                      info[eid]["start_ns"] + int(info[eid]["wall_ms"] * 1e6)]
            w = info[eid]["wall_ms"] * 1e6
            c = union_ns(cpu[eid])
            d = union_ns(device[eid])
            neither = subtract_ns([bounds], sorted(cpu[eid] + device[eid]))
            for name in HOST_KINDS:
                for nstart, nend in neither:
                    for cstart, cend in sorted(by_kind[name][eid]):
                        if cend <= nstart or cstart >= nend:
                            continue
                        neither_by_host[name] += min(cend, nend) - max(cstart, nstart)
            wall += w
            cpu_ns += c
            dev_ns += d
            unaccounted += max(0, w - union_ns(host[eid]))
            overlap += max(0, c + d - union_ns(sorted(cpu[eid] + device[eid])))
            tokens += info[eid]["tokens"]
        result[kind] = {
            "executions": len(ids), "tokens": tokens,
            "W": wall / 1e6, "C": cpu_ns / 1e6, "D": dev_ns / 1e6,
            "U": unaccounted / 1e6, "O": overlap / 1e6,
            "neither": {k: v / 1e6 for k, v in neither_by_host.items()},
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", type=Path, help="execution trace paths")
    args = parser.parse_args()
    for path in args.traces:
        print(f"== {path}")
        for kind, r in analyze(path).items():
            bound = r["W"] / (r["W"] - min(r["C"], r["D"]))
            print(f"kind={kind} execs={r['executions']} tokens={r['tokens']} "
                  f"W={r['W']:.1f}ms C={r['C']:.1f}ms ({100*r['C']/r['W']:.1f}%) "
                  f"D={r['D']:.1f}ms ({100*r['D']/r['W']:.1f}%) U={r['U']:.1f}ms "
                  f"O={r['O']:.1f}ms -> overlap speedup <={bound:.2f}x")
            total_neither = sum(r["neither"].values())
            for name, ms in sorted(r["neither"].items(), key=lambda kv: -kv[1]):
                share = 100 * ms / total_neither if total_neither else 0.0
                print(f"    neither covered by {name}: {ms:.1f}ms ({share:.1f}% of neither)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
