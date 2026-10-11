"""Independent memory constraints and deterministic bounded Pareto search."""
from __future__ import annotations

import math
from collections import defaultdict

from config import AXES, QuantError
from model import tensor_bytes
from timing import gpu


def memory(info, cfg, choices):
    used = {name: cfg["reserve"].get(name, 0) for name in cfg["devices"]}
    types = {name: choices[group["name"]] for group in info["groups"] for name in group["members"]}
    experts = defaultdict(dict)
    for t in info["tensors"]:
        dtype = types[t["name"]]
        size = tensor_bytes(t["shape"], dtype)
        device = "RAM" if t["expert"] and "RAM" in used else gpu(cfg)
        used[device] += size
        if t["expert"]:
            projection = next((p for p in ("gate_up", "gate", "up", "down") if f"ffn_{p}_exps" in t["name"]), None)
            experts[t["layer"]][projection] = (dtype, size // t["shape"][2], t["shape"])
    groups = {}
    cache_bytes, scratch, slots = 0, 0, 0
    if cfg["cache_mib"]:
        if "RAM" not in used or not experts:
            raise QuantError("cache_mib requires a RAM-routed MoE")
        geometry = None
        for layer, ws in sorted(experts.items()):
            if set(ws) != {"gate", "up", "down"}:
                raise QuantError("cache requires separate gate/up/down tensors")
            shape = tuple(tuple(ws[p][2]) for p in ("gate", "up", "down"))
            if geometry and shape != geometry:
                raise QuantError("cache geometry differs between layers")
            geometry = shape
            key = tuple(ws[p][0] for p in ("gate", "up", "down"))
            if key not in groups:
                groups[key] = {"layers": [], "expert_bytes": sum(ws[p][1] for p in ("gate", "up", "down")), "slots": 0}
            groups[key]["layers"].append(layer)
        budget = int(cfg["cache_mib"] * (1 << 20))
        layer_bytes = sum(g["expert_bytes"] * len(g["layers"]) for g in groups.values())
        per_layer = budget // layer_bytes
        if per_layer == 0:
            raise QuantError("cache allowance cannot hold an expert per layer")
        left = budget - per_layer * layer_bytes
        for g in groups.values():
            g["slots"] = per_layer * len(g["layers"]) + left // g["expert_bytes"]
            left %= g["expert_bytes"]
            cache_bytes += g["slots"] * g["expert_bytes"]
            slots += g["slots"]
        largest_slab = max(ws[p][1] for ws in experts.values() for p in ws)
        # Device scratch and pinned refill buffer each hold three slabs per async replacement.
        async_experts = int(cfg["runtime"]["env"].get("LLAMA_MOE_CACHE_ASYNC_EXPERTS", 32))
        scratch = 3 * largest_slab * max(1, async_experts)
        used[gpu(cfg)] += cache_bytes + scratch + sum(ws["gate"][2][2] * 4 for ws in experts.values())
        used["RAM"] += scratch
    return {"used_bytes": used, "headroom_bytes": {name: cfg["devices"][name] - value for name, value in used.items()},
            "cache_bytes": cache_bytes, "cache_slots": slots,
            "cache_groups": [{"types": list(key), **value} for key, value in groups.items()],
            "cache_hit_prior": min(1.0, slots / max(1, info["n_expert"] * len(experts))),
            "allocation_scope": "tensor payloads, cache banks/maps/scratch and supplied reserves; allocator/repack overhead needs witness validation",
            "secondary_gpus": "unassigned; percentages change speed predictions only"}


def make_options(info, damage, operations, cfg):
    records = {}
    for artifact in damage:
        if artifact.get("schema") != "delphi.quant-damage/1":
            raise QuantError("unsupported damage schema")
        for t in artifact["tensors"]:
            if not t.get("runs"):
                continue
            for q in t["runs"][0]["types"]:
                if q["status"] == "ok" and q.get("R") is not None:
                    records[t["name"], q["qtype"].upper()] = q
    by_name = {t["name"]: t for t in info["tensors"]}
    excluded = []
    for group in info["groups"]:
        options = []
        for dtype in group["pool"]:
            error, variance, reason, uncertainty_known = 0.0, 0.0, None, True
            for name in group["members"]:
                t = by_name[name]
                if t["preserved"] or dtype == t["source_dtype"]:
                    continue
                record = records.get((name, dtype))
                if record is None:
                    reason = f"missing or rejected sampled damage: {name} {dtype}"
                    break
                error += record["R"]
                if record.get("R_se") is None:
                    uncertainty_known = False
                else:
                    variance += record["R_se"] ** 2
            # An unsupported measured backend signature excludes the qtype only for its matching region.
            for value in operations.values():
                sig = value["signature"]
                if sig["dtype"] == dtype and any(by_name[n]["shape"] == sig["shape"] and by_name[n]["expert"] == sig["expert"]
                                                 for n in group["members"]) and value["ms"] is None:
                    reason = f"backend rejected {dtype} on {sig['device']}"
            if reason:
                excluded.append({"group": group["name"], "qtype": dtype, "reason": reason})
            else:
                options.append({"qtype": dtype, "damage": error, "sampling_se": math.sqrt(variance) if uncertainty_known else None,
                                "bytes": sum(tensor_bytes(by_name[n]["shape"], dtype) for n in group["members"])})
        if not options:
            raise QuantError(f"no measured legal candidates for {group['name']}")
        group["options"] = options
        group["pool"] = [option["qtype"] for option in options]
    return excluded


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto(states):
    result = []
    for state in states:
        if any(dominates(other["axes"], state["axes"]) for other in result):
            continue
        result = [other for other in result if not dominates(state["axes"], other["axes"])]
        result.append(state)
    return result


def search(info, cfg, predictor, baseline_choices):
    groups = info["groups"]
    options = [{o["qtype"]: o for o in group["options"]} for group in groups]
    baseline = tuple(baseline_choices[group["name"]] for group in groups)
    minimum = tuple(min(o.values(), key=lambda x: (x["bytes"], x["damage"], x["qtype"]))["qtype"] for o in options)
    quality = tuple(min(o.values(), key=lambda x: (x["damage"], x["bytes"], x["qtype"]))["qtype"] for o in options)
    baseline_damage = sum(o[q]["damage"] for o, q in zip(options, baseline))
    reference = [baseline_damage or sum(o[q]["damage"] for o, q in zip(options, minimum)) or 1.0,
                 *[predictor.measured[regime] for regime in AXES[1:]]]
    weights = [cfg["importance"][axis] for axis in AXES]
    evaluated, feasible, rejected, frontier = {}, [], defaultdict(int), []
    cap = cfg["search"]["evaluations"]

    def evaluate(key):
        nonlocal frontier
        if key in evaluated:
            return evaluated[key]
        if len(evaluated) >= cap:
            return None
        evaluated[key] = None
        choices = {group["name"]: q for group, q in zip(groups, key)}
        try:
            mem = memory(info, cfg, choices)
            if any(value < 0 for value in mem["headroom_bytes"].values()):
                rejected["memory"] += 1
                return None
            times = predictor.predict(choices, mem["cache_hit_prior"])
        except QuantError as exc:
            rejected[str(exc)] += 1
            return None
        error = sum(o[q]["damage"] for o, q in zip(options, key))
        axes = [error, *[times[regime] for regime in AXES[1:]]]
        bounds = cfg["bounds"]
        violation = 0.0
        loss_bound = "max_loss" if "quality" in cfg else "max_damage"
        if error > bounds.get(loss_bound, math.inf):
            rejected["loss_bound" if "quality" in cfg else "damage_bound"] += 1
            violation += error / max(bounds[loss_bound], 1e-12) - 1
        for regime in AXES[1:]:
            rate = 1000 / times[regime] * (1 if regime == "decode" else cfg["runtime"][f"{regime}_tokens"])
            if rate < bounds.get(f"min_{regime}_t_s", 0):
                rejected["speed_bound"] += 1
                violation += bounds[f"min_{regime}_t_s"] / rate - 1
        state = {"key": key, "choices": choices, "axes": axes, "damage": error,
                 "sampling_se": (math.sqrt(sum(o[q]["sampling_se"] ** 2 for o, q in zip(options, key)))
                                 if all(o[q]["sampling_se"] is not None for o, q in zip(options, key)) else None),
                 "predicted_ms": times, "memory": mem,
                 "score": sum(w * value / scale for w, value, scale in zip(weights, axes, reference)),
                 "constraint_violation": violation}
        if "quality" in cfg:
            state["loss"] = error
        evaluated[key] = state
        if violation == 0:
            feasible.append(state)
            frontier = pareto([*frontier, state])
        return state

    # Uniform category mixes keep a small search cap from returning only neighbors of the two extremes.
    by_name = {t["name"]: t for t in info["tensors"]}
    seeds = [minimum, baseline, quality]
    primary_types = sorted({q for g in groups if any(by_name[n]["category"] == "primary" for n in g["members"]) for q in g["pool"]})
    other_types = sorted({q for g in groups if any(by_name[n]["category"] == "other" and not by_name[n]["preserved"] for n in g["members"])
                          for q in g["pool"]})
    for primary in primary_types:
        for other in other_types or [None]:
            seed = list(minimum)
            for i, group in enumerate(groups):
                dtype = primary if any(by_name[n]["category"] == "primary" for n in group["members"]) else other
                if dtype in options[i]:
                    seed[i] = dtype
            seeds.append(tuple(seed))
    beam = [state for key in dict.fromkeys(seeds) if (state := evaluate(key)) is not None]
    expanded = set()
    while beam and len(evaluated) < cap:
        new = []
        for state in beam:
            key = state["key"]
            if key in expanded:
                continue
            expanded.add(key)
            moves = []
            for i, option in enumerate(options):
                for q in sorted(option):
                    if q != key[i]:
                        changed = list(key)
                        changed[i] = q
                        moves.append(tuple(changed))
            # A budget-neutral swap can reach a state behind an infeasible one-tensor upgrade.
            upgrades, downgrades = [], []
            for i, option in enumerate(options):
                for q, value in option.items():
                    delta = value["bytes"] - option[key[i]]["bytes"]
                    if delta > 0:
                        upgrades.append((i, q, delta))
                    elif delta < 0:
                        downgrades.append((i, q, delta))
            for i, q, delta in upgrades[:64]:
                for j, qj, dj in downgrades[:64]:
                    if i != j and delta + dj <= 0:
                        changed = list(key)
                        changed[i], changed[j] = q, qj
                        moves.append(tuple(changed))
            for key in moves:
                if len(evaluated) >= cap:
                    break
                if (value := evaluate(key)) is not None:
                    new.append(value)
        candidates = {state["key"]: state for state in [*new, *frontier] if state["key"] not in expanded}
        if not candidates:
            break
        candidates = list(candidates.values())
        winners = [min(candidates, key=lambda s: (s["constraint_violation"], s["axes"][axis], s["score"], s["key"])) for axis in range(4)]
        winners.extend(sorted(candidates, key=lambda s: (s["constraint_violation"], s["score"], s["key"]))[:cfg["search"]["beam"]])
        beam = list({state["key"]: state for state in winners}.values())[:cfg["search"]["beam"]]
    minimum_memory = memory(info, cfg, {g["name"]: q for g, q in zip(groups, minimum)})
    infeasible_minimum = any(value < 0 for value in minimum_memory["headroom_bytes"].values()) and not cfg["cache_mib"]
    return {"frontier": sorted(frontier, key=lambda s: (s["score"], s["key"])), "evaluated": len(evaluated),
            "feasible": len(feasible), "rejected": dict(rejected), "cap_reached": len(evaluated) >= cap,
            "reference_scales": reference, "status": "ok" if frontier else ("infeasible_minimum_bytes" if infeasible_minimum else "search_exhausted")}
