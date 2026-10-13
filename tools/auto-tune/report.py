"""Markdown report and best.json of an auto-tuner run."""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Mapping

import score as S
from knobs import remove_option
from store import Store


def _f(value: float | None, fmt: str = "{:.2f}") -> str:
    return "-" if value is None else fmt.format(value)


def _changes(changes: Mapping[str, Any] | None) -> str:
    if not changes:
        return "(base)"
    return ", ".join(f"{name} {a if a is not None else 'unset'} -> {b if b is not None else 'unset'}" for name, (a, b) in changes.items())


def _prefill_cells(metrics: Mapping[str, Any], lengths) -> list[str]:
    cells = []
    for length in lengths:
        item = (metrics.get("prefill") or {}).get(length) or {}
        ttft = item.get("ttft_s")
        cells.append("-" if ttft is None else f"{1000 * ttft:.0f} ms ({_f(item.get('t_s'), '{:.0f}')} t/s)")
    return cells


def _acceptance(metrics: Mapping[str, Any]) -> str:
    n, a = metrics.get("draft_n"), metrics.get("draft_n_accepted")
    return f"{a}/{n}" if n else "-"


def _vram(metrics: Mapping[str, Any]) -> str:
    return ", ".join(f"{pci} {v['used_mib']}" for pci, v in sorted((metrics.get("vram") or {}).items()) if v.get("used_mib", 0) > 256) or "-"


def write(cfg: Mapping[str, Any], state: Mapping[str, Any], store: Store, out: Path) -> None:
    lengths = list(cfg["prefill"]["lengths"]) if cfg["weights"].get("prefill", 0) > 0 else []
    weights = S.metric_weights(cfg["weights"], cfg["prefill"]["lengths"])
    by_key = {}
    for record in store.records:
        if record["kind"] == "launch" and record.get("phase") == "search":
            by_key.setdefault(record["key"], record)
    base = by_key.get(state.get("base"))
    if base is None:
        raise ValueError("no base measurement in the store")
    confirm = state.get("confirm")
    # the confirm runs of the base come after warm launches; the first base launch can be slower (page cache, lazy loading)
    reference = confirm["base"] if confirm and confirm.get("base") else base["metrics"]
    rows = [r for r in by_key.values() if r["status"] == "ok"]
    incumbent = by_key.get(state.get("incumbent")) or base
    recommended = incumbent if (confirm and confirm.get("kept")) else base

    w = cfg["weights"]
    lines = [f"# Auto-tune report: {Path(cfg['recipe'].rsplit(':', 1)[0]).name}:{cfg['recipe'].rsplit(':', 1)[1]} arm {cfg['arm']}", ""]
    budget = state.get("budget") or {}
    lines.append(f"Weights: prefill {w.get('prefill', 0)}, decode {w.get('decode', 0)}; prefill lengths {dict(cfg['prefill']['lengths'])} "
                 f"(TTFT, {cfg['prefill']['reps']} requests each, median); decode osl {cfg['decode']['osl']}; -c {cfg['context']}; "
                 f"KLD limit {cfg['kld']['limit']}; budget used {budget.get('launches', '?')}/{budget.get('max_launches', '?')} launches, "
                 f"{_f(budget.get('hours'))}/{budget.get('max_hours', '?')} h.")
    lines.append("Score: weighted geometric mean of decode t/s and 1/TTFT per length, relative to the base (1.00"
                 + (", the mean of its confirm runs)." if reference is not base["metrics"] else ")."))
    lines += ["", "## Recommendation", ""]
    if recommended is base:
        why = "no change beat the base by the noise margin" if not confirm else f"the confirm run did not hold the gain (score {_f(confirm.get('gain'), '{:.3f}')})"
        lines.append(f"Keep the base configuration: {why}.")
    else:
        lines.append(f"Changes: {_changes(recommended.get('changes'))}.")
        lines.append(f"Confirm (best, base, best interleaved): score {_f(confirm.get('gain'), '{:.3f}')} against the base"
                     f"{'' if confirm.get('complete') else ' (incomplete: the budget ended the confirm runs)'}.")
        lines += ["", "| | decode t/s | " + " | ".join(f"TTFT {length}" for length in lengths) + " |",
                  "| --- | ---: |" + " ---: |" * len(lengths)]
        for role in ("base", "best"):
            lines.append(f"| {role} | {_f(confirm[role].get('decode_t_s'))} | " + " | ".join(_prefill_cells(confirm[role], lengths)) + " |")
    if recommended.get("kld"):
        lines.append(f"KLD against the exact-precision levels: {recommended['kld']:.4f}.")
    command = remove_option(recommended["command"], "--port")
    lines += ["", "```", " ".join(shlex.quote(item) for item in command), "```", "", "Environment:", "", "```"]
    lines += [f"{k}={shlex.quote(str(v))}" for k, v in sorted(recommended["env"].items())] or ["(none)"]
    lines.append("```")

    lines += ["", "## Options along the trade-off", "",
              "The best measured config for each share of the weight on decode (the rest on prefill). Single runs, not confirmed.", "",
              "| decode share | score | decode t/s | " + " | ".join(f"TTFT {length}" for length in lengths) + " | KLD | changes |",
              "| ---: | ---: | ---: |" + " ---: |" * len(lengths) + " ---: | --- |"]
    for option in S.tradeoff(rows, reference, cfg["prefill"]["lengths"]):
        row = option["row"]
        lines.append(f"| {option['decode_share']:.2f} | {option['score']:.3f} | {_f(row['metrics'].get('decode_t_s'))} | "
                     + " | ".join(_prefill_cells(row["metrics"], lengths)) + f" | {_f(row.get('kld'), '{:.4f}')} | {_changes(row.get('changes'))} |")

    lines += ["", "## Measurements", "", "| config | status | score | decode t/s | accepted | " + " | ".join(f"TTFT {length}" for length in lengths)
              + " | KLD | VRAM MiB used | note |", "| --- | --- | ---: | ---: | ---: |" + " ---: |" * len(lengths) + " ---: | --- | --- |"]
    for record in sorted(store.records, key=lambda r: r.get("utc", "")):
        if record["kind"] != "launch":
            continue
        m = record.get("metrics") or {}
        s = S.score(m, reference, weights) if record["status"] == "ok" and record.get("phase") != "warmup" else None
        note = (record.get("error") or "").replace("|", "/")[:160]
        lines.append(f"| {record['name']} | {record['status']} | {_f(s, '{:.3f}')} | {_f(m.get('decode_t_s'))} | {_acceptance(m)} | "
                     + " | ".join(_prefill_cells(m, lengths)) + f" | {_f(record.get('kld'), '{:.4f}')} | {_vram(m)} | {note} |")

    if state.get("untested"):
        lines += ["", f"Not tested (budget): {', '.join(state['untested'])}."]
    skipped = [step["knob"] for step in state.get("steps", ()) if step.get("skipped")]
    if skipped:
        lines += ["", f"Skipped (no longer applied after an earlier change): {', '.join(skipped)}."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "best.json").write_text(json.dumps({"command": recommended["command"], "env": recommended["env"],
                                               "changes": recommended.get("changes") or {}, "confirm": confirm}, indent=2) + "\n")
