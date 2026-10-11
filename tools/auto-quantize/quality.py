"""Conditional repair scores with fixed floor weights and whole-group decisions."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from config import QuantError, file_identity, number
from model import qtype, read_model, tensor_bytes
from timing import regions, signature

SCHEMA = "delphi.floor-quality/2"
METHOD = "sum_conditional_repair_kl"


def shape4(shape):
    return list(shape) + [1] * (4 - len(shape))


def tensor_hash(tensor):
    h = hashlib.sha256()
    with open(tensor["file"], "rb") as stream:
        stream.seek(tensor["offset"])
        remaining = tensor["source_bytes"]
        while remaining:
            block = stream.read(min(remaining, 4 << 20))
            if not block:
                raise QuantError(f"truncated tensor payload: {tensor['name']}")
            h.update(block)
            remaining -= len(block)
    return h.hexdigest()


def check_hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise QuantError(f"missing or invalid SHA256: {name}")


def check_files(expected, paths, label):
    if len(expected) != len(paths):
        raise QuantError(f"repair calibration {label} count differs")
    for record, path in zip(expected, paths):
        check_hash(record.get("sha256"), label)
        actual = file_identity(path, content=True)
        if record.get("bytes") != actual["bytes"] or record["sha256"] != actual["sha256"]:
            raise QuantError(f"repair calibration {label} content changed: {path}")


def load_quality(info, cfg):
    if "quality" not in cfg:
        return None
    path = Path(cfg["quality"]["table"])
    table = json.loads(path.read_text())
    if table.get("schema") != SCHEMA or table.get("method") != METHOD:
        raise QuantError(f"repair table must use {SCHEMA} and {METHOD}; draft tables need a full floor manifest")
    check_files(table["source_files"], [f["path"] for f in info["files"]], "source")
    check_files(table["imatrices"], cfg["imatrix"], "imatrix")
    execution = table["execution"]
    leading = next(device for device in cfg["devices"] if device != "RAM")
    if execution.get("gpu") != leading or execution.get("ram_routed") != ("RAM" in cfg["devices"]):
        raise QuantError("repair calibration placement differs; collect scores with the selected GPU and CPU-expert policy")
    if execution.get("numa") != "tensors":
        raise QuantError("repair calibration must record --numa tensors")
    if cfg["cache_mib"]:
        raise QuantError("repair quality with an expert cache needs separate calibration and is not implemented")
    floor = table["floor"]
    check_hash(floor.get("sha256"), "floor")
    floor["path"] = str((path.parent / floor["path"]).resolve())
    by_name = {t["name"]: t for t in info["tensors"]}
    if set(floor["tensors"]) != set(by_name):
        raise QuantError("repair floor must describe every source tensor")
    for name, entry in floor["tensors"].items():
        t = by_name[name]
        dtype = qtype(entry["qtype"])
        if shape4(entry["shape"]) != shape4(t["shape"]) or entry["bytes"] != tensor_bytes(t["shape"], dtype):
            raise QuantError(f"repair floor shape/size differs: {name}")
        check_hash(entry.get("sha256"), name)
        if dtype not in t["pool"]:
            raise QuantError(f"repair floor type {dtype} is outside configured candidates/preservation: {name}")
    names, members = set(), set()
    if not table["groups"]:
        raise QuantError("repair table contains no measured groups")
    for group in table["groups"]:
        name = group["name"]
        if not isinstance(name, str) or not name or name in names or name in by_name:
            raise QuantError(f"duplicate or conflicting repair group name: {name}")
        names.add(name)
        selected = group["members"]
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(by_name) or members.intersection(selected):
            raise QuantError(f"missing, duplicate, overlapping or unknown repair members: {name}")
        members.update(selected)
        for atomic in info["groups"]:
            overlap = set(atomic["members"]).intersection(selected)
            if overlap and overlap != set(atomic["members"]):
                raise QuantError(f"repair group splits a configured gate/up tie: {name}")
        floor_types = {floor["tensors"][n]["qtype"] for n in selected}
        if len(floor_types) != 1:
            raise QuantError(f"repair group has mixed floor qtypes: {name}")
        seen = set()
        for option in group["options"]:
            dtype = qtype(option["qtype"])
            if dtype in seen:
                raise QuantError(f"duplicate repair qtype: {name} {dtype}")
            seen.add(dtype)
            number(option["loss"], f"repair loss {name} {dtype}")
            losses = option["chunk_losses"]
            if not losses:
                raise QuantError(f"repair option needs chunk losses: {name} {dtype}")
            for loss in losses:
                number(loss, "repair chunk loss")
            if not math.isclose(option["loss"], sum(losses) / len(losses), rel_tol=1e-8, abs_tol=1e-12):
                raise QuantError(f"repair mean differs from chunk losses: {name} {dtype}")
            if option["bytes"] != sum(tensor_bytes(by_name[n]["shape"], dtype) for n in selected):
                raise QuantError(f"repair option byte count differs: {name} {dtype}")
            if set(option["tensor_sha256"]) != set(selected):
                raise QuantError(f"repair option payload identities are incomplete: {name} {dtype}")
            for tensor_name, value in option["tensor_sha256"].items():
                check_hash(value, tensor_name)
            if dtype in floor_types and any(option["tensor_sha256"][n] != floor["tensors"][n]["sha256"] for n in selected):
                raise QuantError(f"repair floor option payload differs: {name}")
        if not floor_types.issubset(seen):
            raise QuantError(f"repair group lacks its floor option: {name}")
    table["identity"] = file_identity(path, content=True)
    return table


def apply_quality(info, table):
    """Keep execution pairs, but couple search decisions at the measured granularity."""
    if table is None:
        return info
    by_name = {t["name"]: t for t in info["tensors"]}
    decisions, owner = [], {}
    for measured in table["groups"]:
        members = measured["members"]
        pool = set.intersection(*(set(by_name[n]["pool"]) for n in members))
        options = [copy.deepcopy(o) for o in measured["options"] if o["qtype"] in pool]
        floor_type = table["floor"]["tensors"][members[0]]["qtype"]
        if not options or floor_type != min(options, key=lambda o: (o["bytes"], o["loss"], o["qtype"]))["qtype"]:
            raise QuantError(f"repair floor is not the minimum-byte legal option: {measured['name']}")
        decision = {"name": measured["name"], "members": list(members), "pool": [o["qtype"] for o in options],
                    "repair_options": options, "floor_qtype": floor_type}
        decisions.append(decision)
        owner.update({n: decision for n in members})
    execution = []
    for atomic in info["groups"]:
        group = copy.deepcopy(atomic)
        decision = owner.get(group["members"][0])
        if decision is None:
            fixed = {table["floor"]["tensors"][n]["qtype"] for n in group["members"]}
            if len(fixed) != 1:
                raise QuantError(f"repair floor conflicts with a gate/up tie: {group['name']}")
            dtype = fixed.pop()
            decision = {"name": group["name"], "members": group["members"], "pool": [dtype], "floor_qtype": dtype}
            decisions.append(decision)
            # Fixed work cancels in the baseline-anchored predictor. Repair mode does not use cache-hit changes.
            group["quality_fixed"] = True
        group["decision"] = decision["name"]
        group["pool"] = decision["pool"]
        execution.append(group)
        for n in group["members"]:
            by_name[n]["pool"] = decision["pool"]
    info["execution_groups"] = execution
    info["groups"] = decisions
    return info


def repair_options(info, table, operations):
    by_name = {t["name"]: t for t in info["tensors"]}
    measured_regions = regions(info)
    excluded = []
    for group in info["groups"]:
        options = []
        for dtype in group["pool"]:
            failed = next((v["signature"] for v in operations.values() if v["ms"] is None and v["signature"]["dtype"] == dtype
                           and any(region.get("decision", region["name"]) == group["name"]
                                   and signature(members, paired, dtype, v["signature"]["device"], v["signature"]["tokens"], info["top_k"]) == v["signature"]
                                   for region, members, paired in measured_regions)), None)
            if failed:
                excluded.append({"group": group["name"], "qtype": dtype, "reason": f"backend rejected {dtype} on {failed['device']}"})
                continue
            measured = next((o for o in group.get("repair_options", []) if o["qtype"] == dtype), None)
            options.append({"qtype": dtype, "damage": measured["loss"] if measured else 0.0, "sampling_se": None,
                            "bytes": sum(tensor_bytes(by_name[n]["shape"], dtype) for n in group["members"])})
        if group["floor_qtype"] not in {o["qtype"] for o in options}:
            raise QuantError(f"repair floor has an unsupported operation signature: {group['name']}")
        group["options"] = options
        group["pool"] = [o["qtype"] for o in options]
    for group in info["execution_groups"]:
        group["pool"] = next(g["pool"] for g in info["groups"] if g["name"] == group["decision"])
    return excluded


def verify_quality_output(info, table, types, output):
    if table is None:
        return
    expected = {n: entry["sha256"] for n, entry in table["floor"]["tensors"].items()}
    for group in table["groups"]:
        dtype = types[group["members"][0]]
        if any(types[n] != dtype for n in group["members"]):
            raise QuantError(f"encoded recipe splits repair group: {group['name']}")
        option = next((o for o in group["options"] if o["qtype"] == dtype), None)
        if option is None:
            raise QuantError(f"encoded recipe uses unmeasured repair type: {group['name']} {dtype}")
        expected.update(option["tensor_sha256"])
    for tensor in read_model(output)["tensors"]:
        if tensor_hash(tensor) != expected[tensor["name"]]:
            raise QuantError(f"encoded tensor differs from repair calibration payload: {tensor['name']}")


def quality_metadata(table):
    if table is None:
        return {"metric": "sum_local_R", "units": "local source-relative error", "qualification": "proxy"}
    return {"metric": METHOD, "units": "conditional repair KL sum", "qualification": "experimental",
            "table": table["identity"], "execution": table["execution"],
            "scope": "whole measured groups on the recorded floor and quality execution policy",
            "ceiling": "max_loss constrains the proxy; it is not an upper bound on original-relative KLD or task accuracy"}
