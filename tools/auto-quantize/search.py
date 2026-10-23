"""Independent memory constraints and a knapsack search over the precision/speed trade-off."""
from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict

from config import AXES, QuantError
from knapsack import solve
from model import is_lazy, tensor_bytes
from timing import gpu, gpu_only, gpus, projected, requires_backend_signature


def split_layers(layer_bytes, tail_bytes, reserves, capacities):
    """Layers per device of the contiguous split that minimizes the largest (reserve + bytes) / capacity.

    The last device also holds tail_bytes (the output). Bisection on the utilization u with a greedy fill: each device
    takes layers while it stays within u * capacity and the last takes the rest. O(devices * log(layers)) per step."""
    n, count = len(layer_bytes), len(capacities)
    prefix = [0]
    for b in layer_bytes:
        prefix.append(prefix[-1] + b)

    def fill(u):
        counts, start = [], 0
        for d in range(count - 1):
            end = bisect_right(prefix, prefix[start] + u * capacities[d] - reserves[d]) - 1
            end = min(max(end, start), n)
            counts.append(end - start)
            start = end
        counts.append(n - start)
        last = reserves[-1] + prefix[n] - prefix[start] + tail_bytes
        return counts, last <= u * capacities[-1]

    low, high = 0.0, (prefix[n] + tail_bytes + max(reserves)) / min(capacities)
    for _ in range(100):
        mid = (low + high) / 2
        if fill(mid)[1]:
            high = mid
        else:
            low = mid
        if high - low <= high * 1e-13:
            break
    return fill(high)[0]


def llama_layer_devices(counts):
    """Device index of every layer and then the output, as llama-model.cpp assigns them for -ts counts with -ngl 999.

    The cumulative split points are normalized in float32 and layer il goes to upper_bound(splits, float(il) / (n + 1))."""
    import numpy as np
    f32 = np.float32
    total = sum(counts)  # n_layer_all + 1: the output counts as a layer
    splits, running = [], f32(0)
    for c in counts:
        running = f32(running + f32(c))
        splits.append(running)
    splits = [f32(x / running) for x in splits]
    return [next((d for d, x in enumerate(splits) if f32(f32(il) / f32(total)) < x), len(splits)) for il in range(total)]


def memory(info, cfg, choices):
    """Per-device resident bytes, headroom and the cache accounting of a recipe.

    GPU-only (several GPUs, no RAM): the layers are split contiguously over the GPUs the way -sm layer -ts does it, the
    output (and a tied embedding copy) on the last GPU, balancing (reserve + bytes) / capacity. The input embedding
    stays on the host and is reported as host_bytes, outside every device budget."""
    only = gpu_only(cfg)
    used = {name: cfg["reserve"].get(name, 0) for name in cfg["devices"]}
    types = {name: choices[group["name"]] for group in info["groups"] for name in group["members"]}
    experts = defaultdict(dict)
    lazy_bytes, host_bytes, tail, tied = 0, 0, 0, 0
    layer_bytes = [0] * (info["n_layer"] if only else 0)
    tied_output = only and not any(t["name"] == "output.weight" for t in info["tensors"])
    if only and not cfg.get("mtp", True):
        # the server does not load MTP blocks without an MTP draft; they are the last blocks
        layer_bytes = layer_bytes[:info["n_layer"] - len(info.get("mtp_layers", ()))]
    for t in info["tensors"]:
        if t.get("mtp") and not cfg.get("mtp", True):
            continue
        dtype = types[t["name"]]
        size = tensor_bytes(t["shape"], dtype)
        if is_lazy(t["name"]):
            # read on demand from disk: not resident, so outside every device budget
            lazy_bytes += size
            continue
        if only:
            if t["name"].startswith("token_embd"):
                host_bytes += size
                if tied_output and t["name"] == "token_embd.weight":
                    tied += size  # llama.cpp duplicates the tied embedding onto the output device
            elif t["layer"] >= 0:
                if t["layer"] >= len(layer_bytes):
                    raise QuantError(f"{t['name']} is outside the model's {len(layer_bytes)} blocks")
                layer_bytes[t["layer"]] += size
            else:
                tail += size
        else:
            device = "RAM" if t["expert"] and "RAM" in used else gpu(cfg)
            used[device] += size
        if t["expert"]:
            projection = next((p for p in ("gate_up", "gate", "up", "down") if f"ffn_{p}_exps" in t["name"]), None)
            experts[t["layer"]][projection] = (dtype, size // t["shape"][2], t["shape"])
    split = None
    if only:
        names = gpus(cfg)
        counts = split_layers(layer_bytes, tail + tied, [used[d] for d in names], [cfg["devices"][d] for d in names])
        start = 0
        for d, c in zip(names, counts):
            used[d] += sum(layer_bytes[start:start + c])
            start += c
        used[names[-1]] += tail + tied
        split = [*counts[:-1], counts[-1] + 1]  # the output counts as the last layer of -ts
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
            "lazy_bytes": lazy_bytes, "host_bytes": host_bytes,
            "total_bytes": sum(used.values()) - sum(cfg["reserve"].get(name, 0) for name in cfg["devices"]) + host_bytes,
            "cache_bytes": cache_bytes, "cache_slots": slots,
            "cache_groups": [{"types": list(key), **value} for key, value in groups.items()],
            "cache_hit_prior": min(1.0, slots / max(1, info["n_expert"] * len(experts))),
            "allocation_scope": "tensor payloads without lazy tables, cache banks/maps/scratch and supplied reserves; allocator/repack overhead needs witness validation; lazy tables sit on disk and use page cache as the system allows",
            "secondary_gpus": "unassigned; percentages change speed predictions only" if not only else
            "layers split contiguously over the GPUs in order (layer_split, last count includes the output); the input embedding stays on the host",
            **({"layer_split": split, "runtime_args": ["--device", ",".join(gpus(cfg)), "-sm", "layer", "-ts", ",".join(map(str, split))]}
               if only else {})}


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
                if t["preserved"] or dtype == t["source_dtype"] or is_lazy(name):
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
                                                 for n in group["members"]) and value["ms"] is None and requires_backend_signature(sig, cfg):
                    reason = f"backend rejected {dtype} on {sig['device']}"
            if reason:
                excluded.append({"group": group["name"], "qtype": dtype, "reason": reason})
            else:
                options.append({"qtype": dtype, "damage": error, "sampling_se": math.sqrt(variance) if uncertainty_known else None,
                                "bytes": sum(tensor_bytes(by_name[n]["shape"], dtype) for n in group["members"] if not is_lazy(n))})
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


def device_of(info, cfg, group):
    by_name = {t["name"]: t for t in info["tensors"]}
    return "RAM" if by_name[group["members"][0]]["expert"] and "RAM" in cfg["devices"] else gpu(cfg)


def search(info, cfg, predictor, baseline_choices, model=None):
    """Recipes on the precision/speed trade-off within the device budgets.

    Loss is the predicted KL of the quality model, or the summed local error without one. Both are separable over
    groups (the model at a fixed composition point), and so are the anchored time predictions without an expert cache.
    For each loss multiplier theta, an exact multiple-choice knapsack per device minimizes
    theta * loss / loss_ref + sum_regime importance * time / time_ref; theta = the precision importance is the linearized
    score. A sweep of theta gives the supported points, walks between neighbouring solutions add the recipes between
    them, and bisection finds the edges of the bounds. Every candidate is then evaluated exactly."""
    groups = info["groups"]
    options = {g["name"]: {o["qtype"]: o for o in g["options"]} for g in groups}
    baseline_memory = memory(info, cfg, baseline_choices)
    placement = {g["name"]: device_of(info, cfg, g) for g in groups}
    only = gpu_only(cfg)
    if only:
        # the layers move between the GPUs with the recipe, so a device budget is its capacity less the reserve
        budgets = {d: cfg["devices"][d] - cfg["reserve"].get(d, 0) for d in gpus(cfg)}
    else:
        budgets = {d: baseline_memory["headroom_bytes"][d] + sum(options[n][baseline_choices[n]]["bytes"]
                                                                 for n, dev in placement.items() if dev == d)
                   for d in cfg["devices"]}
    costs = predictor.group_costs()
    scale = {r: predictor.slopes[r] / ((1 + cfg["multi_gpu_speedup_percent"][r] / 100) if projected(cfg) else 1) for r in AXES[1:]}
    weights = cfg["importance"]

    # MTP layers do not run in the main model; the local metric ignores them too, so they stay at the floor
    by_name = {t["name"]: t for t in info["tensors"]}
    mtp = {g["name"] for g in groups if any(by_name[n].get("mtp") for n in g["members"])}

    def local(n, q):
        return 0.0 if n in mtp else options[n][q]["damage"]

    def loss_of(choices):
        if model is not None:
            return model.predict(choices)["loss"]
        return sum(local(n, q) for n, q in choices.items())

    loss_ref = loss_of(baseline_choices) or 1.0
    reference = [loss_ref, *[predictor.measured[regime] for regime in AXES[1:]]]
    time_terms = {n: {q: sum(weights[r] * scale[r] * costs[n][q][r] / predictor.measured[r] for r in AXES[1:])
                      if n in costs else 0.0 for q in options[n]} for n in options}
    bounds = cfg["bounds"]
    loss_bound = "max_loss" if model is not None else "max_damage"
    # Pooled mode: one knapsack over all groups under a cap on the summed tensor bytes (without reserves), whatever the
    # device. The cap of the bounds is a hard limit; the knapsack cap moves during the search for the smallest recipe.
    hard_cap = bounds.get("max_total_bytes")
    device_cap = None
    if only:
        # GPU-only: layers are not fixed to a device, so one pooled knapsack over the summed GPU capacities. The input
        # embedding lives on the host but is in the option bytes (unless it is tied: then its copy is on the last GPU).
        by_name = {t["name"]: t for t in info["tensors"]}
        host = 0 if "output.weight" not in by_name else sum(
            max(o["bytes"] for o in options[g["name"]].values()) for g in groups if all(n.startswith("token_embd") for n in g["members"]))
        device_cap = sum(budgets.values()) + host
    pool = {"cap": device_cap if hard_cap is None else hard_cap if device_cap is None else min(device_cap, hard_cap)}
    minimize = cfg["search"].get("minimize")
    evaluated, feasible, rejected, frontier = {}, [], defaultdict(int), []

    def evaluate(choices):
        nonlocal frontier
        key = tuple(choices[g["name"]] for g in groups)
        if key in evaluated:
            return evaluated[key]
        evaluated[key] = None
        try:
            mem = memory(info, cfg, choices)
            if any(value < 0 for value in mem["headroom_bytes"].values()):
                rejected["memory"] += 1
                return None
            if hard_cap is not None and mem["total_bytes"] > hard_cap:
                rejected["total_bytes"] += 1
                return None
            times = predictor.predict(choices, mem["cache_hit_prior"])
        except QuantError as exc:
            rejected[str(exc)] += 1
            return None
        loss = loss_of(choices)
        axes = [loss, *[times[regime] for regime in AXES[1:]]]
        violation = 0.0
        if loss > bounds.get(loss_bound, math.inf):
            rejected["loss_bound" if model is not None else "damage_bound"] += 1
            violation += loss / max(bounds[loss_bound], 1e-12) - 1
        for regime in AXES[1:]:
            rate = 1000 / times[regime] * (1 if regime == "decode" else cfg["runtime"][f"{regime}_tokens"])
            if rate < bounds.get(f"min_{regime}_t_s", 0):
                rejected["speed_bound"] += 1
                violation += bounds[f"min_{regime}_t_s"] / rate - 1
        state = {"key": key, "choices": dict(choices), "axes": axes, "loss": loss,
                 "damage": sum(options[n][q]["damage"] for n, q in choices.items()),
                 "sampling_se": (math.sqrt(sum(options[n][q]["sampling_se"] ** 2 for n, q in choices.items()))
                                 if all(options[n][q]["sampling_se"] is not None for n, q in choices.items()) else None),
                 "predicted_ms": times, "memory": mem,
                 "score": sum(w * value / s for w, value, s in zip([weights[a] for a in AXES], axes, reference)),
                 "constraint_violation": violation}
        if model is not None:
            state["composition_u"] = model.predict(choices)["u"]
        evaluated[key] = state
        if violation == 0:
            feasible.append(state)
            frontier = pareto([*frontier, state])
        return state

    def knapsacks(cap):
        """(group names, byte budget) per knapsack: one per device, or one pooled knapsack under a total cap."""
        if cap is None:
            return {d: ([n for n, dev in placement.items() if dev == d], b) for d, b in budgets.items()}
        return {"pool": (list(placement), cap)}

    def solve_for(theta):
        """Knapsack recipe for a loss multiplier. GPU-only: contiguity can make a recipe whose total fits fail on a
        device, so the cap of this solve (not pool["cap"]) drops by the deficit and the solve repeats, up to 8 times."""
        cap = pool["cap"]
        for _ in range(9 if only else 1):
            picked = solve_at(theta, cap)
            if picked is None or not only:
                return picked
            deficit = -min(memory(info, cfg, picked)["headroom_bytes"].values())
            if deficit <= 0:
                return picked
            cap -= max(deficit, 1 << 20)
        return picked

    def solve_at(theta, cap):
        """Knapsack recipe under a cap; the model's composition point is iterated to a fixed point."""
        marginal, previous = 1.0, None
        for _ in range(8):
            picked = {}
            for device, (names, budget) in knapsacks(cap).items():
                if not names:
                    continue
                table = {}
                for n in names:
                    table[n] = []
                    for q, o in options[n].items():
                        if model is not None:
                            loss = -model.kl_floor * marginal * model.gain(n, q)
                        else:
                            loss = local(n, q)
                        # a byte tie-breaker keeps groups without a gain or time difference at the smaller type
                        table[n].append((o["bytes"], theta * loss / loss_ref + time_terms[n][q] + 1e-9 * o["bytes"] / max(1, budget), q))
                result = solve(table, budget)
                if result is None:
                    return None
                picked.update(result[0])
            if model is None or picked == previous:
                return picked
            previous = picked
            marginal = model.marginal(model.predict(picked)["u"])
        return picked

    def smallest_recipe():
        """Smallest total bytes whose minimum-loss recipe meets the loss bound; sets the pooled cap to it.

        The minimum-loss knapsack is exact for a byte cap and its loss does not increase with the cap, so the cap is
        bisected. The loss multiplier is large, so time terms only break ties. Speed bounds and device capacities
        are checked on the final recipe only."""
        if loss_bound not in bounds:
            raise QuantError("minimizing size needs a loss bound: max_loss with a quality table, otherwise max_damage")
        ceiling = bounds[loss_bound]
        theta = 2.0 ** 10
        low = sum(min(o["bytes"] for o in options[n].values()) for n in options)
        high = sum(max(o["bytes"] for o in options[n].values()) for n in options)
        if hard_cap is not None:
            high = min(high, hard_cap)
        if device_cap is not None:
            high = min(high, device_cap)
        if high < low:
            return None, "infeasible_minimum_bytes"

        def meets(cap):
            pool["cap"] = cap
            choices = solve_for(theta)
            return choices if choices is not None and loss_of(choices) <= ceiling else None

        best = meets(high)
        if best is None:
            return None, "loss_bound_unreachable"
        found = meets(low)
        if found is not None:
            best, high = found, low
        while high - low > max(1, high * 1e-4):
            mid = (low + high) // 2
            found = meets(mid)
            if found is None:
                low = mid
            else:
                best, high = found, mid
        pool["cap"] = high
        return best, "ok"

    def separable(n, q, marginal):
        if model is not None:
            return -model.kl_floor * marginal * model.gain(n, q) / loss_ref
        return local(n, q) / loss_ref

    smallest, smallest_status = None, None
    if minimize == "size":
        choices, smallest_status = smallest_recipe()
        smallest = evaluate(choices) if choices is not None else None
        if choices is not None and (smallest is None or smallest["constraint_violation"] > 0):
            smallest_status = "smallest_violates_device_or_speed_bounds"
        if smallest is not None:
            smallest["smallest"] = True
    thetas = sorted({0.0, weights["precision"], *(2.0 ** k for k in range(-10, 11))})
    states = []
    for theta in thetas:
        choices = solve_for(theta)
        states.append((theta, evaluate(choices) if choices is not None else None))
    # Sweep solutions are the supported points of the trade-off only. Walk from each solution to the next one group at
    # a time, most loss reduction per unit of time first, so recipes between them are also evaluated.
    solved = [s for _, s in states if s is not None]
    for a, b in zip(solved, solved[1:]):
        marginal = model.marginal(model.predict(a["choices"])["u"]) if model is not None else 1.0
        changes = []
        for n, q in b["choices"].items():
            p = a["choices"][n]
            if p != q:
                dl = separable(n, q, marginal) - separable(n, p, marginal)
                dt = time_terms[n][q] - time_terms[n][p]
                changes.append((dl / dt if dt > 1e-15 else -math.inf if dl < 0 else math.inf, n, q))
        current = dict(a["choices"])
        for _, n, q in sorted(changes)[:-1]:
            current[n] = q
            evaluate(current)
    # The loss decreases and the time increases with theta, so each bound has one edge on the sweep.
    for (t0, s0), (t1, s1) in zip(states, states[1:]):
        if s0 is None or s1 is None or (s0["constraint_violation"] == 0) == (s1["constraint_violation"] == 0):
            continue
        lo, hi, ok_hi = t0, t1, s1["constraint_violation"] == 0
        for _ in range(12):
            mid = (lo + hi) / 2 if lo > 0 else hi / 2
            choices = solve_for(mid)
            state = evaluate(choices) if choices is not None else None
            if state is None:
                break
            if (state["constraint_violation"] == 0) == ok_hi:
                hi = mid
            else:
                lo = mid
    minimum = {g["name"]: min(options[g["name"]].values(), key=lambda o: (o["bytes"], o["damage"], o["qtype"]))["qtype"] for g in groups}
    minimum_memory = memory(info, cfg, minimum)
    infeasible_minimum = any(value < 0 for value in minimum_memory["headroom_bytes"].values()) and not cfg["cache_mib"]
    if smallest is not None and smallest["constraint_violation"] == 0 and all(smallest is not f for f in frontier):
        # the smallest recipe can be dominated on loss and speed; it stays in the result because size was the objective
        frontier = [*frontier, smallest]
    return {"frontier": sorted(frontier, key=lambda s: (s["score"], s["key"])), "evaluated": len(evaluated),
            "feasible": len(feasible), "rejected": dict(rejected), "cap_reached": False,
            "budgets_bytes": budgets, "reference_scales": reference,
            "minimize": minimize, "pool_cap_bytes": pool["cap"],
            "status": smallest_status if smallest_status not in (None, "ok") else "ok" if frontier else ("infeasible_minimum_bytes" if infeasible_minimum else "no_feasible_recipe")}
