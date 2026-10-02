#!/usr/bin/env python3
"""Show how the ubatches of large executions (prefill) flow through the devices in a GGML execution trace.

For every execution with at least --min-tokens tokens, the summary lists each ubatch on each device (first device
start, last device end and busy time, in ms from the start of the execution), the host waits by reason and device, and
how long 0, 1, 2, ... devices were busy at the same time with the work of any execution. With pipeline parallelism, a well-fed pipeline keeps every
stage busy back to back; drains show as gaps on the first stage and as long waits. Needs GPU timing
(GGML_EXEC_TRACE_GPU_TIMING=submit or op).
"""

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path


def load_trace_module():
    path = Path(__file__).with_name("exec-trace-dump.py")
    spec = importlib.util.spec_from_file_location("exec_trace_dump", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


T = load_trace_module()


def concurrency(intervals_by_device, start, end):
    """Time with exactly k devices busy, for k = 0..n, within [start, end]."""
    edges = []
    for intervals in intervals_by_device.values():
        merged = []
        for lo, hi in sorted(intervals):
            lo, hi = max(lo, start), min(hi, end)
            if hi <= lo:
                continue
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        for lo, hi in merged:
            edges += [(lo, 1), (hi, -1)]
    busy = defaultdict(int)
    level, last = 0, start
    for time, delta in sorted(edges):
        busy[level] += time - last
        level += delta
        last = time
    busy[level] += end - last
    return busy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--min-tokens", type=int, default=512, help="smallest execution to show (default: 512)")
    parser.add_argument("--last", type=int, default=0, help="show only the last N matching executions")
    args = parser.parse_args()
    try:
        trace = T.read_trace(args.trace)
    except (OSError, T.TraceError) as error:
        print(f"exec-trace-pipeline: {error}", file=sys.stderr)
        return 1
    events = trace["events"]
    backend_name = T.build_backend_names(events, trace["strings"])

    begin = {e["execution_id"]: e for e in events if e["kind"] == T.K_EXECUTION_BEGIN and e["args"][0] >= args.min_tokens}
    end = {e["execution_id"]: e["timestamp_ns"] for e in events if e["kind"] == T.K_EXECUTION_END and e["execution_id"] in begin}
    # per (execution, device): submit / graph spans, or single operations when only those were recorded
    collected = defaultdict(lambda: {"submit": [], "op": []})
    spans, _ = T.gpu_timing_model(events)
    for span in spans:
        event = span["event"]
        if "device_start_ns" not in span:
            continue
        kind = "op" if span["scope"] == "eager_op" else "submit"
        collected[(event["execution_id"], backend_name(event))][kind].append(
            (event["ubatch_id"], span["device_start_ns"], span["device_end_ns"]))
    ubatch = defaultdict(lambda: [None, None, 0])   # (execution, ubatch, device) -> first start, last end, busy
    device_intervals = defaultdict(list)   # device -> intervals of all executions, for the overlap
    for (execution_id, name), kinds in collected.items():
        for ub, start, stop in kinds["submit"] or kinds["op"]:
            device_intervals[name].append((start, stop))
            if execution_id not in begin:
                continue
            entry = ubatch[(execution_id, ub, name)]
            entry[0] = start if entry[0] is None else min(entry[0], start)
            entry[1] = stop if entry[1] is None else max(entry[1], stop)
            entry[2] += stop - start
    waits = defaultdict(lambda: defaultdict(float))
    for event in events:
        if event["execution_id"] in begin and event["kind"] in T.WAIT_KINDS:
            reason = T.WAIT_REASONS.get(event["flags"], str(event["flags"]))
            waits[event["execution_id"]][(reason, backend_name(event))] += T.span_duration_ns(event) / 1e6

    ids = sorted(begin)
    if args.last:
        ids = ids[-args.last:]
    if not ids:
        print(f"no executions of at least {args.min_tokens} tokens")
    window = None
    for execution_id in ids:
        t0 = begin[execution_id]["timestamp_ns"]
        rows = sorted(((k[1], k[2], v) for k, v in ubatch.items() if k[0] == execution_id), key=lambda row: row[2][0])
        t_end = max([end.get(execution_id, t0)] + [row[2][1] for row in rows])
        host = (end[execution_id] - t0) / 1e6 if execution_id in end else float("nan")
        print(f"== execution {execution_id}: {begin[execution_id]['args'][0]} tokens, phase {T.execution_phase(begin[execution_id])}, "
              f"host {host:.1f} ms, last device end {(t_end - t0) / 1e6:.1f} ms")
        for ub, name, (start, stop, busy) in rows:
            print(f"  ubatch {ub:5d} {name:8s} start {(start - t0) / 1e6:9.1f} end {(stop - t0) / 1e6:9.1f} busy {busy / 1e6:8.1f}")
        for (reason, name), ms in sorted(waits[execution_id].items(), key=lambda item: -item[1]):
            if ms >= 1.0:
                print(f"  wait {reason} {name}: {ms:.1f} ms")
        print_overlap("  devices busy at once (all work)", device_intervals, t0, t_end)
        window = (min(window[0], t0), max(window[1], t_end)) if window else (t0, t_end)
    if len(ids) > 1:
        print_overlap(f"shown executions, {(window[1] - window[0]) / 1e6:.1f} ms: devices busy at once", device_intervals, *window)
    return 0


def print_overlap(label, device_intervals, start, end):
    busy = concurrency(device_intervals, start, end)
    total = sum(busy.values()) or 1
    print(f"{label}: " + ", ".join(f"{k}: {busy[k] / 1e6:.1f} ms ({100 * busy[k] / total:.0f}%)" for k in sorted(busy)))


if __name__ == "__main__":
    sys.exit(main())
