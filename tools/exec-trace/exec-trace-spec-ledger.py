#!/usr/bin/env python3
"""Per-cycle ledger of speculative decoding: host spans (DELPHI_SPEC_LEDGER) against device intervals (execution trace).

The host thread of the server is sequential, so the top-level ledger spans partition each cycle; the waits (the target
synchronization and the draft's lattice getter) are then split by the device intervals of the execution trace that overlap them,
so a kernel interval and the host waiting for it are one cost.

Cycle: from the end of one target verification (llama_synchronize of ctx_tgt returns) to the end of the next. It contains the draft
ingest of the previous verification's features, its acceptance, the next draft block, the draft KV removal and the next
verification. The tokens emitted by a verification (accepted drafts + 1) are assigned to the cycle that ends with it.

Stages (top-level, disjoint; ms per cycle):
  ingest      PROCESS: gather target features, llama_decode(ctx_dft) of the injected rows (host call; the device runs on)
  accept      ACCEPT: target sampling and acceptance of the draft
  draft block DRAFT minus lattice and walk: noise-batch construction and llama_decode(ctx_dft) host call
  lattice     DFT_LATTICE: llama_get_embeddings_nextn(ctx_dft), which synchronizes the draft context (waits for ingest and block)
  walk        DFT_WALK: selector lattice walk on the host
  dft rollback DFT_ROLLBACK, kv rollback KV_ROLLBACK
  verify call TGT_DECODE: llama_decode(ctx_tgt) host call;  verify wait TGT_SYNC: llama_synchronize(ctx_tgt)
  other       the remainder: sampling, response handling, batch assembly, locks (unclassified host work)

Usage: exec-trace-spec-ledger.py LEDGER.jsonl [--trace TRACE] [--windows CLIENT.json] [--json OUT]
"""

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

STAGES = {1: "TGT_DECODE", 2: "TGT_SYNC", 3: "PROCESS", 4: "FEAT_GATHER", 5: "DFT_INGEST", 6: "DRAFT", 7: "DFT_BLOCK",
          8: "DFT_LATTICE", 9: "DFT_WALK", 10: "DFT_ROLLBACK", 11: "ACCEPT", 12: "KV_ROLLBACK"}
WITNESS = {1: "candidates", 2: "kv", 3: "accepted"}


def load_trace_module():
    path = Path(__file__).with_name("exec-trace-dump.py")
    spec = importlib.util.spec_from_file_location("exec_trace_dump", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_ledger(path):
    spans, witnesses = [], []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if "dropped" in row:
            print(f"warning: ledger dropped {row['dropped']} records", file=sys.stderr)
        elif "w" in row:
            witnesses.append({"kind": WITNESS.get(row["w"], row["w"]), **{k: row[k] for k in ("t", "a", "b", "c", "ids")}})
        else:
            spans.append({"stage": STAGES.get(row["s"], row["s"]), "t0": row["t0"], "t1": row["t1"],
                          "a": row["a"], "b": row["b"], "c": row["c"], "x": row["x"]})
    spans.sort(key=lambda s: s["t0"])
    witnesses.sort(key=lambda w: w["t"])
    return spans, witnesses


def union(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def overlap_ns(window, intervals):
    lo, hi = window
    return sum(max(0, min(hi, b) - max(lo, a)) for a, b in union(intervals))


def device_intervals(trace, T, windows):
    """Executions of the speculative phases with their device intervals: {execution_id: {phase, tokens, begin, end, devices: {name: [(s, e)]}}}."""
    events = trace["events"]
    strings = trace["strings"]
    backend_name = T.build_backend_names(events, strings)
    executions = {}
    for event in events:
        if event["kind"] == T.K_EXECUTION_BEGIN:
            phase = T.execution_phase(event)
            if phase in ("generation_target", "generation_draft"):
                executions[event["execution_id"]] = {"phase": phase, "tokens": event["args"][0], "begin": event["timestamp_ns"],
                                                     "end": None, "devices": defaultdict(list), "uncertainty": []}
    for event in events:
        if event["kind"] == T.K_EXECUTION_END and event["execution_id"] in executions:
            executions[event["execution_id"]]["end"] = event["timestamp_ns"]
    spans, _ = T.gpu_timing_model(events)
    for span in spans:
        execution = executions.get(span["event"]["execution_id"])
        if execution is None or "device_start_ns" not in span or span["scope"] not in ("eager_submit", "graph_launch"):
            continue
        execution["devices"][backend_name(span["event"])].append((span["device_start_ns"], span["device_end_ns"]))
        execution["uncertainty"].append(span.get("placement_uncertainty_ns") or 0)
    return [e for e in executions.values() if e["end"] is not None]


def in_windows(t, windows):
    return not windows or any(lo <= t <= hi for lo, hi in windows)


def build_cycles(spans, windows):
    syncs = [s for s in spans if s["stage"] == "TGT_SYNC" and in_windows(s["t0"], windows)]
    cycles = []
    for prev, cur in zip(syncs, syncs[1:]):
        # consecutive verifications of one request only
        if windows and next((i for i, (lo, hi) in enumerate(windows) if lo <= prev["t1"] <= hi), None) != \
                next((i for i, (lo, hi) in enumerate(windows) if lo <= cur["t1"] <= hi), None):
            continue
        lo, hi = prev["t1"], cur["t1"]
        inside = [s for s in spans if lo <= s["t0"] and s["t1"] <= hi]
        cycles.append({"lo": lo, "hi": hi, "spans": inside, "verify_tokens": cur["a"]})
    return cycles


def stage_ms(cycle, name):
    return sum(s["t1"] - s["t0"] for s in cycle["spans"] if s["stage"] == name) / 1e6


def summarize(spans, witnesses, executions, windows):
    cycles = build_cycles(spans, windows)
    accepts = sorted((s for s in spans if s["stage"] == "ACCEPT"), key=lambda s: s["t0"])
    rows = []
    for cycle in cycles:
        wall = (cycle["hi"] - cycle["lo"]) / 1e6
        # the acceptance of the verification that ends this cycle is recorded right after the cycle's end
        accept_after = next((a for a in accepts if a["t0"] >= cycle["hi"]), None)
        row = {"wall_ms": wall, "verify_rows": cycle["verify_tokens"],
               "draft_size": accept_after["a"] if accept_after else None,
               "accepted_drafts": accept_after["b"] if accept_after else None}
        row["ingest"] = stage_ms(cycle, "PROCESS")
        row["ingest_gather"] = stage_ms(cycle, "FEAT_GATHER")
        row["ingest_decode_call"] = stage_ms(cycle, "DFT_INGEST")
        row["accept"] = stage_ms(cycle, "ACCEPT")
        row["draft_total"] = stage_ms(cycle, "DRAFT")
        row["lattice_wait"] = stage_ms(cycle, "DFT_LATTICE")
        row["walk"] = stage_ms(cycle, "DFT_WALK")
        row["block_call"] = row["draft_total"] - row["lattice_wait"] - row["walk"]
        row["block_decode_call"] = stage_ms(cycle, "DFT_BLOCK")
        row["dft_rollback"] = stage_ms(cycle, "DFT_ROLLBACK")
        row["kv_rollback"] = stage_ms(cycle, "KV_ROLLBACK")
        row["verify_call"] = stage_ms(cycle, "TGT_DECODE")
        row["verify_wait"] = stage_ms(cycle, "TGT_SYNC")
        classified = (row["ingest"] + row["accept"] + row["draft_total"] + row["dft_rollback"] + row["kv_rollback"]
                      + row["verify_call"] + row["verify_wait"])
        row["other"] = wall - classified
        # device intervals inside the cycle: draft executions on the draft device, target executions on all devices
        window = (cycle["lo"], cycle["hi"])
        drafts = sorted((e for e in executions if e["phase"] == "generation_draft" and window[0] <= e["begin"] <= window[1]),
                        key=lambda e: e["begin"])
        targets = [e for e in executions if e["phase"] == "generation_target" and window[0] <= e["begin"] <= window[1]]
        if drafts or targets:
            lattice = [(s["t0"], s["t1"]) for s in cycle["spans"] if s["stage"] == "DFT_LATTICE"]
            draft_dev = [iv for e in drafts for ivs in e["devices"].values() for iv in ivs]
            target_dev = [iv for e in targets for ivs in e["devices"].values() for iv in ivs]
            wait = [(s["t0"], s["t1"]) for s in cycle["spans"] if s["stage"] == "TGT_SYNC"]
            row["draft_executions"] = [(e["tokens"], sum(hi - lo for lo, hi in union([iv for ivs in e["devices"].values() for iv in ivs])) / 1e6)
                                       for e in drafts]
            row["lattice_wait_draft_device_busy"] = overlap_ns(lattice[0], draft_dev) / 1e6 if lattice else None
            row["draft_device_busy_total"] = sum(hi - lo for lo, hi in union(draft_dev)) / 1e6
            row["draft_device_busy_hidden"] = row["draft_device_busy_total"] - (row["lattice_wait_draft_device_busy"] or 0)
            row["verify_wait_target_device_busy"] = overlap_ns(wait[0], target_dev) / 1e6 if wait else None
            row["target_device_busy_total"] = sum(hi - lo for lo, hi in union(target_dev)) / 1e6
            row["placement_uncertainty_ms"] = max([u for e in drafts + targets for u in e["uncertainty"]] or [0]) / 1e6
        rows.append(row)
    return rows


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def render(rows, witnesses):
    if not rows:
        return "no complete cycle in the ledger"
    n = len(rows)
    wall = mean(rows, "wall_ms")
    lines = [f"{n} cycles, {wall:.2f} ms mean (verify end to verify end)"]
    emitted = [r["accepted_drafts"] + 1 for r in rows if r["accepted_drafts"] is not None]
    if emitted:
        mean_tokens = sum(emitted) / len(emitted)
        lines.append(f"tokens per cycle {mean_tokens:.3f}, {wall / mean_tokens:.2f} ms per token, {1000 * mean_tokens / wall:.1f} t/s; "
                     f"verify rows {mean(rows, 'verify_rows'):.2f}, draft size {mean(rows, 'draft_size'):.2f}")
    lines.append("stage (ms per cycle; indented rows are inside the row above and not added again)")
    table = [("ingest (gather + draft decode call)", "ingest"), ("  feature gather", "ingest_gather"), ("  draft ingest decode call", "ingest_decode_call"),
             ("accept (sampling, acceptance)", "accept"),
             ("draft block (noise batch + decode call)", "block_call"), ("  draft block decode call", "block_decode_call"),
             ("lattice getter (waits for the draft device)", "lattice_wait"), ("lattice walk", "walk"),
             ("draft KV rollback", "dft_rollback"), ("KV rollback after acceptance", "kv_rollback"),
             ("verify decode call (host)", "verify_call"), ("verify wait (device)", "verify_wait"), ("other host work (unclassified)", "other")]
    top = {"ingest", "accept", "block_call", "lattice_wait", "walk", "dft_rollback", "kv_rollback", "verify_call", "verify_wait", "other"}
    total = 0.0
    for label, key in table:
        value = mean(rows, key)
        if value is None:
            continue
        if key in top:
            total += value
        lines.append(f"  {label:48s} {value:8.3f}  {100 * value / wall:5.1f}%")
    lines.append(f"  {'sum of the top-level rows':48s} {total:8.3f}  {100 * total / wall:5.1f}%")
    if any("draft_device_busy_total" in r for r in rows):
        lines.append("device intervals (execution trace) against the host waits:")
        for label, key in (("draft device busy per cycle (CUDA2)", "draft_device_busy_total"),
                           ("  of it inside the lattice wait (critical path)", "lattice_wait_draft_device_busy"),
                           ("  of it hidden behind host work", "draft_device_busy_hidden"),
                           ("target devices busy per cycle (union)", "target_device_busy_total"),
                           ("  of it inside the verify wait", "verify_wait_target_device_busy"),
                           ("median-of-max placement uncertainty (ms)", "placement_uncertainty_ms")):
            value = mean(rows, key)
            if value is not None:
                lines.append(f"  {label:48s} {value:8.3f}")
        lattice, busy = mean(rows, "lattice_wait"), mean(rows, "lattice_wait_draft_device_busy")
        if lattice is not None and busy is not None:
            lines.append(f"  lattice wait not covered by a draft device interval: {lattice - busy:.3f} ms "
                         f"(readback, synchronization and placement error)")
    cand = [w for w in witnesses if w["kind"] == "candidates"]
    if cand:
        cut = sum(1 for w in cand if w["b"])
        lines.append(f"witnesses: {len(cand)} candidate lists, mean length {sum(len(w['ids']) for w in cand) / len(cand):.2f}, "
                     f"{cut} cut by the confidence threshold")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--trace", type=Path, help="execution trace of the same run (GGML_EXEC_TRACE_GPU_TIMING=submit)")
    parser.add_argument("--windows", type=Path, help="client JSON with results[].request_windows_raw_ns: only cycles inside them")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    spans, witnesses = read_ledger(args.ledger)
    windows = []
    if args.windows:
        windows = [tuple(w) for r in json.loads(args.windows.read_text())["results"] for w in r["request_windows_raw_ns"]]
    executions = []
    if args.trace:
        T = load_trace_module()
        executions = device_intervals(T.read_trace(args.trace), T, windows)
    rows = summarize(spans, witnesses, executions, windows)
    print(render(rows, witnesses))
    if args.json:
        args.json.write_text(json.dumps({"cycles": rows, "witnesses": witnesses}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
