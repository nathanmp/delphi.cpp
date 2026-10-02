#!/usr/bin/env python3
"""Summarize speculative decoding in a GGML execution trace.

Speculative decode executions are grouped by phase (generation_target: verify passes; generation_draft: draft steps, including the
draft context's catch-up decodes) and token count. For each group the summary gives the number of executions, the host
time in llama_decode per execution (llama_decode can return before the device work ends), the busy time of each device
and the CPU compute time. One speculative cycle (the draft executions of a verify pass plus the pass) is measured from
one verify pass to the next, with the device busy time per cycle and the share of the cycle when every device is idle. With operation timing (GGML_EXEC_TRACE_GPU_TIMING=op, or
CPU node timing), it also ranks the operations of each group.

Acceptance is not in the trace. Divide the cycle time by the tokens accepted per verify pass (plus one), from the
server's draft_n_accepted and the number of verify passes, for the time per output token.
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_trace_module():
    path = Path(__file__).with_name("exec-trace-dump.py")
    spec = importlib.util.spec_from_file_location("exec_trace_dump", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


T = load_trace_module()

PHASES = ("generation_draft", "generation_target")


def union_ns(intervals):
    total = 0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def median(values):
    return T.median(values) if values else None


def summarize(trace, top, node_pattern, windows_ns=()):
    """windows_ns: optional (start, end) raw-clock windows, e.g. the client's request windows; executions that begin
    outside every window are left out."""
    events = trace["events"]
    strings = trace["strings"]
    backend_name = T.build_backend_names(events, strings)
    node_metadata = T.build_node_metadata(events, strings)

    executions = {}
    for event in events:
        # executions of more than one token are recorded as prefill; the phase tells speculative work apart
        if event["kind"] == T.K_EXECUTION_BEGIN:
            phase = T.execution_phase(event)
            if windows_ns and not any(lo <= event["timestamp_ns"] <= hi for lo, hi in windows_ns):
                continue
            if phase in PHASES:
                executions[event["execution_id"]] = {"phase": phase, "tokens": event["args"][0],
                                                     "begin_ns": event["timestamp_ns"], "end_ns": None}
    for event in events:
        if event["kind"] == T.K_EXECUTION_END and event["execution_id"] in executions:
            executions[event["execution_id"]]["end_ns"] = event["timestamp_ns"]

    uncertainty = defaultdict(list)                    # execution -> placement uncertainty of its device spans (ns)
    submit = defaultdict(lambda: defaultdict(list))   # execution -> backend -> device intervals of submits / graphs
    ops = defaultdict(lambda: defaultdict(list))      # execution -> backend -> device intervals of single operations
    op_ns = defaultdict(Counter)                       # execution -> "backend / op" -> device ns
    cpu_ns = Counter()                                 # execution -> CPU compute span ns
    cpu_op_ns = defaultdict(Counter)                   # execution -> "CPU / op" -> ns
    spans, _ = T.gpu_timing_model(events)
    for span in spans:
        event = span["event"]
        execution_id = event["execution_id"]
        if execution_id not in executions or "device_start_ns" not in span:
            continue
        interval = (span["device_start_ns"], span["device_end_ns"])
        name = backend_name(event)
        uncertainty[execution_id].append(span.get("placement_uncertainty_ns") or 0)
        if span["scope"] in ("eager_submit", "graph_launch"):
            submit[execution_id][name].append(interval)
        elif span["scope"] == "eager_op":
            ops[execution_id][name].append(interval)
            node = node_metadata(event) or {}
            label = node.get("op", "unknown")
            if node_pattern and node_pattern.search(node.get("name", "")):
                label = f"[{node_pattern.pattern}]"
            op_ns[execution_id][f"{name} / {label}"] += span["device_duration_ns"]
    for event in events:
        execution_id = event["execution_id"]
        if execution_id not in executions:
            continue
        if event["kind"] == T.K_CPU_COMPUTE:
            cpu_ns[execution_id] += T.span_duration_ns(event)
        elif event["kind"] == T.K_CPU_NODE_TIMING:
            node = node_metadata(event) or {}
            label = node.get("op", "unknown")
            if node_pattern and node_pattern.search(node.get("name", "")):
                label = f"[{node_pattern.pattern}]"
            cpu_op_ns[execution_id][f"CPU / {label}"] += T.span_duration_ns(event)

    groups = defaultdict(list)
    for execution_id, info in executions.items():
        if info["end_ns"] is not None:
            groups[(info["phase"], info["tokens"])].append(execution_id)

    result = {"groups": [], "cycle": None}
    for (phase, tokens), ids in sorted(groups.items()):
        n = len(ids)
        backends = sorted({name for i in ids for name in (*submit[i], *ops[i])})
        device = {}
        for name in backends:
            per_exec = [union_ns(submit[i][name] or ops[i][name]) for i in ids]
            device[name] = sum(per_exec) / n / 1e6
        any_device = [union_ns([iv for name in backends for iv in (submit[i][name] or ops[i][name])]) for i in ids]
        ranked = Counter()
        for i in ids:
            ranked.update(op_ns[i])
            ranked.update(cpu_op_ns[i])
        result["groups"].append({
            "phase": phase,
            "tokens": tokens,
            "executions": n,
            "host_wall_ms_median": median([(executions[i]["end_ns"] - executions[i]["begin_ns"]) / 1e6 for i in ids]),
            "host_wall_ms_mean": sum((executions[i]["end_ns"] - executions[i]["begin_ns"]) for i in ids) / n / 1e6,
            "cpu_compute_ms_mean": sum(cpu_ns[i] for i in ids) / n / 1e6,
            "device_busy_ms_mean": device,
            "any_device_busy_ms_mean": sum(any_device) / n / 1e6,
            "ops_ms_mean": [{"op": op, "ms": ns / n / 1e6} for op, ns in ranked.most_common(top)],
        })

    verify_ids = [i for (phase, _), ids in groups.items() if phase == "generation_target" for i in ids]
    if len(verify_ids) >= 2:
        # llama_decode returns before the device work ends, so the cycle is measured from one verify pass to the next,
        # within one request window (or the whole trace without windows)
        starts = defaultdict(list)
        for i in verify_ids:
            begin_ns = executions[i]["begin_ns"]
            window = next((w for w, (lo, hi) in enumerate(windows_ns) if lo <= begin_ns <= hi), 0)
            starts[window].append(begin_ns)
        deltas = [b - a for values in starts.values() for a, b in zip(sorted(values), sorted(values)[1:])]
        spans_ns = sum(max(values) - min(values) for values in starts.values())
        n_cycles = len(deltas)
        all_ids = [i for ids in groups.values() for i in ids]
        backends = sorted({name for i in all_ids for name in (*submit[i], *ops[i])})
        device = {name: sum(union_ns(submit[i][name] or ops[i][name]) for i in all_ids) / len(verify_ids) / 1e6 for name in backends}
        any_device = union_ns([iv for i in all_ids for name in backends for iv in (submit[i][name] or ops[i][name])])
        # Device spans are placed at the earliest start the host clock allows. A stage that the host submits before the
        # previous stage finishes (layer split) can start much later than placed, so the overlap between devices, and
        # with it the any-device time, is only known when that uncertainty is small against the cycle.
        placement_ms = (median([u for i in all_ids for u in uncertainty[i]]) or 0) / 1e6
        host = {phase: sum(executions[i]["end_ns"] - executions[i]["begin_ns"]
                           for (p, _), ids in groups.items() if p == phase for i in ids) / len(verify_ids) / 1e6 for phase in PHASES}
        n_draft = sum(len(ids) for (phase, _), ids in groups.items() if phase == "generation_draft")
        result["cycle"] = {
            "verify_passes": len(verify_ids),
            "draft_executions_per_verify": n_draft / len(verify_ids),
            "cycle_wall_ms": spans_ns / n_cycles / 1e6 if n_cycles else None,
            "cycle_wall_ms_median": median(deltas) / 1e6 if deltas else None,
            "device_busy_ms_per_cycle": device,
            "device_busy_sum_ms_per_cycle": sum(device.values()),
            "placement_uncertainty_ms_median": placement_ms,
            "any_device_busy_ms_per_cycle": (any_device / len(verify_ids) / 1e6
                                             if n_cycles and placement_ms <= 0.05 * spans_ns / n_cycles / 1e6 else None),
            "cpu_compute_ms_per_cycle": sum(cpu_ns[i] for i in all_ids) / len(verify_ids) / 1e6,
            "host_ms_in_decode_per_cycle": {"draft": host["generation_draft"], "verify": host["generation_target"]},
        }
    return result


def render(result, accepted):
    lines = []
    for group in result["groups"]:
        device = ", ".join(f"{name} {ms:.2f}" for name, ms in group["device_busy_ms_mean"].items()) or "no device timing"
        lines.append(f"{group['phase']:17s} {group['tokens']:5d} tokens x {group['executions']:5d}: host in llama_decode "
                     f"{group['host_wall_ms_mean']:.2f} ms (median {group['host_wall_ms_median']:.2f}), CPU compute "
                     f"{group['cpu_compute_ms_mean']:.2f} ms, device busy {device} (any {group['any_device_busy_ms_mean']:.2f}) ms")
        for item in group["ops_ms_mean"]:
            lines.append(f"    {item['ms']:8.3f} ms  {item['op']}")
    cycle = result["cycle"]
    if cycle and cycle["cycle_wall_ms"]:
        device = ", ".join(f"{name} {ms:.2f}" for name, ms in cycle["device_busy_ms_per_cycle"].items()) or "no device timing"
        any_device = cycle["any_device_busy_ms_per_cycle"]
        if any_device is not None:
            overlap = f"any {any_device:.2f} ms, all devices idle {100.0 * (1.0 - any_device / cycle['cycle_wall_ms']):.0f}% of the cycle"
        else:
            overlap = (f"sum {cycle['device_busy_sum_ms_per_cycle']:.2f} ms, the critical path if the devices run one after another; "
                       f"overlap unknown, median placement uncertainty {cycle['placement_uncertainty_ms_median']:.2f} ms")
        lines.append(f"cycle: {cycle['verify_passes']} verify passes, {cycle['draft_executions_per_verify']:.2f} draft executions per verify; "
                     f"{cycle['cycle_wall_ms']:.2f} ms verify to verify (median {cycle['cycle_wall_ms_median']:.2f})")
        lines.append(f"  per cycle: device busy {device} ({overlap})"
                     + f"; CPU compute {cycle['cpu_compute_ms_per_cycle']:.2f} ms"
                     + f"; host time in llama_decode: draft {cycle['host_ms_in_decode_per_cycle']['draft']:.2f} ms, "
                     f"verify {cycle['host_ms_in_decode_per_cycle']['verify']:.2f} ms")
        if accepted is not None:
            lines.append(f"with {accepted:.2f} accepted draft tokens per verify: {cycle['cycle_wall_ms'] / (accepted + 1):.2f} ms per "
                         f"output token ({1000 * (accepted + 1) / cycle['cycle_wall_ms']:.1f} t/s)")
        else:
            lines.append("time per output token = cycle / (accepted draft tokens per verify + 1); see --accepted-per-verify")
    else:
        lines.append("fewer than two verify passes (generation_target executions) in this trace")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=8, help="operations listed per group with operation timing (default: 8)")
    parser.add_argument("--node-regex", help="report operations whose node name matches this regex as one group, e.g. 'attn|kq'")
    parser.add_argument("--accepted-per-verify", type=float, help="accepted draft tokens per verify pass (server draft_n_accepted / verify passes), for the time per output token")
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    args = parser.parse_args()
    try:
        trace = T.read_trace(args.trace)
    except (OSError, T.TraceError) as error:
        print(f"exec-trace-spec: {error}", file=sys.stderr)
        return 1
    result = summarize(trace, args.top, re.compile(args.node_regex) if args.node_regex else None)
    print(render(result, args.accepted_per_verify))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
