"""Frontier presentation for the quantization CLI: rows, thinning to a readable set, table text and pick specifications.

Pure functions over the recipes of a `frontier.json`; no inference, GPU or numpy is needed."""
from __future__ import annotations

import math
import re
from collections import Counter

GIB = 1 << 30
# Metrics a pick specification can name: (row key, True when smaller is better)
METRICS = {"min-kld": ("kld", True), "min-size": ("size", True), "fastest-decode": ("decode", False),
           "fastest-short": ("short", False), "fastest-long": ("long", False)}


def rows(payload):
    """One display row per recipe: size, predicted KL (or the local proxy), speeds in t/s and the expert qtype mix."""
    cfg = payload["config"]
    runtime = cfg["runtime"]
    reserve = sum(cfg.get("reserve", {}).values())
    result = []
    for r in payload["recipes"]:
        ms = r["predicted_ms"]
        result.append({"id": r["id"], "size": r["memory"].get("total_bytes", sum(r["memory"]["used_bytes"].values()) - reserve), "kld": r["loss"],
                       "decode": 1000 / ms["decode"], "short": 1000 * runtime["short_tokens"] / ms["short"],
                       "long": 1000 * runtime["long_tokens"] / ms["long"], "score": r["score"],
                       "experts": qtype_mix(r["tensor_types"]), "recipe": r})
    return result


def qtype_mix(tensor_types):
    """Share of routed-expert tensors per qtype, largest first, as 'IQ4_XS 61% Q3_K 39%'."""
    counts = Counter(q for name, q in tensor_types.items() if "_exps." in name)
    total = sum(counts.values())
    if not total:
        counts, total = Counter(tensor_types.values()), len(tensor_types)
    return " ".join(f"{q} {100 * n / total:.0f}%" for q, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3])


def features(row):
    # log scale: a 10 % change counts the same at any magnitude. Speeds enter as times (smaller is better)
    return [math.log(max(row["kld"], 1e-12)), -math.log(row["decode"]), -math.log(row["short"]), -math.log(row["long"]),
            math.log(row["size"])]


def thin(all_rows, n):
    """At most n rows: the best-score recipe, the extreme of every metric, then farthest-point samples in log space."""
    if len(all_rows) <= n:
        return list(all_rows)
    chosen = [min(all_rows, key=lambda r: r["score"])]
    for key, small in (("kld", True), ("size", True), ("decode", False), ("short", False), ("long", False)):
        best = (min if small else max)(all_rows, key=lambda r: r[key])
        if best not in chosen and len(chosen) < n:
            chosen.append(best)
    feats = {r["id"]: features(r) for r in all_rows}
    spread = [max(f[i] for f in feats.values()) - min(f[i] for f in feats.values()) or 1.0 for i in range(5)]

    def distance(a, b):
        return math.sqrt(sum(((x - y) / s) ** 2 for x, y, s in zip(feats[a["id"]], feats[b["id"]], spread)))

    while len(chosen) < n:
        far = max((r for r in all_rows if r not in chosen), key=lambda r: min(distance(r, c) for c in chosen))
        chosen.append(far)
    return sorted(chosen, key=lambda r: (r["size"], r["kld"]))


def table(shown, loss_label="pred KL", mark=None):
    """Numbered text table; deltas are against the best-score recipe (marked with *)."""
    best = min(shown, key=lambda r: r["score"])
    lines = [f"{'#':>3s}  {'id':<10s} {'size GiB':>9s} {loss_label:>9s} {'dKL':>7s} {'dec t/s':>8s} {'short':>8s} {'long':>8s}  experts"]
    for i, r in enumerate(shown, 1):
        dk = f"{100 * (r['kld'] / best['kld'] - 1):+.0f}%" if r is not best else "-"
        flag = "*" if r is best else " "
        lines.append(f"{i:>3d}{flag} {r['id']:<10s} {r['size'] / GIB:9.2f} {r['kld']:9.4g} {dk:>7s} "
                     f"{r['decode']:8.2f} {r['short']:8.1f} {r['long']:8.1f}  {r['experts']}")
    return "\n".join(lines)


def pick(shown, all_rows, spec):
    """Resolve a pick specification to a row: a 1-based number of the table, a recipe id, 'best' or a METRICS name."""
    spec = spec.strip()
    if spec == "best":
        return min(all_rows, key=lambda r: r["score"])
    if spec in METRICS:
        key, small = METRICS[spec]
        return (min if small else max)(all_rows, key=lambda r: r[key])
    if re.fullmatch(r"[0-9]+", spec):
        index = int(spec)
        if not 1 <= index <= len(shown):
            raise ValueError(f"row {index} is outside 1..{len(shown)}")
        return shown[index - 1]
    for r in all_rows:
        if r["id"] == spec:
            return r
    raise ValueError(f"unknown pick {spec!r}; use a row number, a recipe id, best, or one of {', '.join(METRICS)}")


def detail(recipe):
    """Qtype histogram per tensor kind ('ffn_down_exps': 'IQ4_XS x30, Q3_K x14') for a recipe's tensor_types."""
    kinds = {}
    for name, q in recipe["tensor_types"].items():
        kinds.setdefault(re.sub(r"^blk\.[0-9]+\.", "", name).removesuffix(".weight"), Counter())[q] += 1
    return {kind: ", ".join(f"{q} x{n}" for q, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))) for kind, counts in sorted(kinds.items())}


DIVERSE_KEYS = ("kld", "size", "decode", "short", "long")


# The predicted KL is uncertain by more than 2 % (verify measured 0.9x to 1.6x), so rows closer than this are one choice
KLD_TOLERANCE = 0.10


def near(a, b, tolerance, kld_tolerance=KLD_TOLERANCE):
    """Whether every predicted metric of two rows is within the relative tolerance (kld_tolerance for the KL)."""
    return all(abs(a[k] - b[k]) <= (kld_tolerance if k == "kld" else tolerance) * max(abs(b[k]), 1e-12) for k in DIVERSE_KEYS)


def diverse(all_rows, n, tolerance=0.02):
    """At most n rows for a choice: the best-score row and the min-KL row, then the best scores that are not within
    `tolerance` (KL: KLD_TOLERANCE) on every predicted metric of a row already chosen. Sorted by size."""
    ordered = sorted(all_rows, key=lambda r: r["score"])
    chosen = [ordered[0]]
    lowest = min(all_rows, key=lambda r: r["kld"])
    if lowest not in chosen:
        chosen.append(lowest)
    for r in ordered:
        if len(chosen) >= n:
            break
        if r not in chosen and not any(near(r, c, tolerance) for c in chosen):
            chosen.append(r)
    return sorted(chosen, key=lambda r: (r["size"], r["kld"]))


def run_table(shown, loss_label="pred KL"):
    """Numbered table of the choice of `run`: row, size, predicted KL, decode/short/long t/s and score (* = best score)."""
    best = min(shown, key=lambda r: r["score"])
    lines = [f"{'row':>3s}  {'id':<10s} {'size GiB':>9s} {loss_label:>9s} {'decode':>8s} {'short':>8s} {'long':>8s} {'score':>8s}"]
    for i, r in enumerate(shown, 1):
        lines.append(f"{i:>3d}{'*' if r is best else ' '} {r['id']:<10s} {r['size'] / GIB:9.2f} {r['kld']:9.4g} {r['decode']:8.2f} "
                     f"{r['short']:8.1f} {r['long']:8.1f} {r['score']:8.4f}")
    return "\n".join(lines)
