"""Teacher-relative quality model: predicted KL of a recipe from local damage and a calibrated floor-relative table.

Per decision group g with role r and layer l, relative to the floor recipe:

    gain_g(q) = s_r(l) * (M_g(floor)**gamma_r - M_g(q)**gamma_r),   log s_r piecewise linear in the layer l
    G = sum_g gain_g(q_g),  u = G / G_full,  gain = kappa(u) * G,  kappa(u) = 1 + (kinf - 1) * (1 - exp(-u / u0))
    KL(recipe) = KL_floor * (1 - gain)

M is the sampled local relative error R of llama-quant-damage, summed over the group members (zero at the source
type). log s_r is given at knots (the centres of the calibration bands of the role) and is constant outside them.
G_full is G with every modelled group at the source type. The calibrate command fits the parameters to measured
KL against teacher logits; qtypes enter only through M, so new candidate qtypes need no extra calibration passes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from config import QuantError, digest, file_identity, number

SCHEMA = "delphi.floor-quality/3"
METHOD = "composed_smooth_sensitivity"
METRIC = "R"


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


def role_of(name):
    """Sensitivity role of a tensor: routed or dense FFN gate/up ('gu') and down ('dn'), shared experts, attention,
    recurrent (ssm) projections, other per-layer kinds by name, and whole-model tensors by name."""
    m = re.match(r"blk\.[0-9]+\.(.+)\.weight$", name)
    if not m:
        return name.removesuffix(".weight")
    kind = m[1]
    if kind.endswith("_shexp"):
        return "shexp"
    if re.fullmatch(r"ffn_(gate|up|gate_up)(_exps)?", kind):
        return "gu"
    if re.fullmatch(r"ffn_down(_exps)?", kind):
        return "dn"
    if kind.startswith("attn_"):
        return "attn"
    if kind.startswith("ssm_"):
        return "ssm"
    return kind


def damage_records(damage):
    """{(tensor, qtype): sampled local relative error} from prepared llama-quant-damage artifacts."""
    records = {}
    for artifact in damage:
        if artifact.get("schema") != "delphi.quant-damage/1":
            raise QuantError("unsupported damage schema")
        for t in artifact["tensors"]:
            for q in (t["runs"][0]["types"] if t.get("runs") else []):
                if q["status"] == "ok" and q.get(METRIC) is not None:
                    records[t["name"], q["qtype"].upper()] = q[METRIC]
    return records


def damage_digest(damage):
    return digest(sorted([name, qtype, value] for (name, qtype), value in damage_records(damage).items()))


def sensitivity(knots, log_s, layer):
    """exp of the piecewise linear interpolation of log_s over knots, constant outside the knots."""
    if len(knots) == 1 or layer <= knots[0]:
        return math.exp(log_s[0])
    if layer >= knots[-1]:
        return math.exp(log_s[-1])
    i = next(k for k in range(1, len(knots)) if layer <= knots[k])
    t = (layer - knots[i - 1]) / (knots[i] - knots[i - 1])
    return math.exp((1 - t) * log_s[i - 1] + t * log_s[i])


def modelled_groups(info, floor):
    """Decision groups whose qtype can change, with role and layer; preserved-only and MTP groups are skipped."""
    by_name = {t["name"]: t for t in info["tensors"]}
    result = []
    for group in info["groups"]:
        members = [by_name[n] for n in group["members"]]
        # MTP layers do not affect the main-model KL; they stay at the floor unless pinned.
        if len(group["pool"]) < 2 or all(t["preserved"] for t in members) or any(t.get("mtp") for t in members):
            continue
        result.append({"name": group["name"], "members": group["members"], "role": role_of(group["members"][0]),
                       "layer": members[0]["layer"], "floor": floor[group["name"]], "pool": group["pool"]})
    return result


class LossModel:
    def __init__(self, info, table, damage):
        self.table = table
        self.records = damage_records(damage)
        self.by_name = {t["name"]: t for t in info["tensors"]}
        self.n_layer = table["n_layer"]
        self.roles = table["roles"]
        self.kinf, self.u0 = table["composition"]["kappa_inf"], table["composition"]["u0"]
        self.kl_floor = table["floor"]["kl"]
        self.groups = {g["name"]: g for g in modelled_groups(info, table["floor"]["choices"])}
        for g in self.groups.values():
            if g["role"] not in self.roles:
                raise QuantError(f"quality table has no sensitivity for role {g['role']}: {g['name']}")
            g["scale"] = self.sensitivity(g["role"], g["layer"])
        self.cache = {}
        self.full = sum(self.contribution(g, g["floor"]) - self.contribution(g, None) for g in self.groups.values())
        if not self.full > 0:
            raise QuantError("quality table predicts no gain for a full repair")

    def sensitivity(self, role, layer):
        return sensitivity(self.roles[role]["knots"], self.roles[role]["log_s"], layer)

    def metric(self, group, qtype):
        total = 0.0
        for name in group["members"]:
            t = self.by_name[name]
            if t["preserved"] or qtype is None or qtype == t["source_dtype"]:
                continue
            value = self.records.get((name, qtype))
            if value is None:
                raise QuantError(f"no sampled damage for {name} {qtype}")
            total += value
        return total

    def contribution(self, group, qtype):
        key = (group["name"], qtype)
        if key not in self.cache:
            gamma = self.roles[group["role"]]["gamma"]
            self.cache[key] = group["scale"] * self.metric(group, qtype) ** gamma
        return self.cache[key]

    def gain(self, name, qtype):
        g = self.groups.get(name)
        if g is None:
            return 0.0
        return self.contribution(g, g["floor"]) - self.contribution(g, qtype)

    def kappa(self, u):
        return 1 + (self.kinf - 1) * (1 - math.exp(-u / self.u0))

    def marginal(self, u):
        """d(kappa(u) G)/dG at u: the composed value of one unit of additive gain."""
        return self.kappa(u) + u * (self.kinf - 1) * math.exp(-u / self.u0) / self.u0

    def predict(self, choices):
        g = sum(self.gain(name, choices[name]) for name in self.groups)
        u = g / self.full
        return {"loss": self.kl_floor * (1 - self.kappa(u) * g), "gain": g, "u": u}


def check_files(expected, paths, label):
    if len(expected) != len(paths):
        raise QuantError(f"quality table {label} count differs")
    for record, path in zip(expected, paths):
        actual = file_identity(path, content=True)
        if record.get("bytes") != actual["bytes"] or record.get("sha256") != actual["sha256"]:
            raise QuantError(f"quality table {label} content changed: {path}")


def load_quality(info, cfg, prepared):
    """The calibrated table of cfg['quality'], checked against the prepared source, imatrices, floor and damage."""
    if "quality" not in cfg:
        return None
    path = Path(cfg["quality"]["table"])
    table = json.loads(path.read_text())
    if table.get("schema") != SCHEMA or table.get("method") != METHOD or table.get("metric") != METRIC:
        raise QuantError(f"quality table must use {SCHEMA}, {METHOD} and metric {METRIC}; recalibrate older tables")
    check_files(table["source_files"], [f["path"] for f in info["files"]], "source")
    check_files(table["imatrices"], cfg["imatrix"], "imatrix")
    if table["damage"] != damage_digest(prepared["damage"]):
        raise QuantError("quality table was fitted to different sampled damage; calibrate with this prepared directory")
    if table["floor"]["choices"] != prepared["baseline"]["choices"]:
        raise QuantError("quality table floor differs from the prepared minimum-byte baseline")
    if cfg["cache_mib"]:
        raise QuantError("quality tables with an expert cache need separate calibration and are not implemented")
    for role, entry in table["roles"].items():
        knots, values = entry["knots"], entry["log_s"]
        if not knots or len(knots) != len(values) or any(b <= a for a, b in zip(knots, knots[1:])):
            raise QuantError(f"quality role {role} needs increasing knots with one log sensitivity each")
        for value in [*knots, *values]:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise QuantError(f"quality role {role} knots and sensitivities must be finite")
        number(entry["gamma"], f"quality role {role} gamma", positive=True)
    number(table["composition"]["kappa_inf"], "kappa_inf", minimum=1)
    number(table["composition"]["u0"], "u0", positive=True)
    number(table["floor"]["kl"], "floor KL", positive=True)
    table["identity"] = file_identity(path, content=True)
    return table


def quality_metadata(table):
    if table is None:
        return {"metric": "sum_local_R", "units": "local source-relative error", "qualification": "proxy"}
    return {"metric": METHOD, "units": "predicted KL against the teacher on the calibration text",
            "table": table["identity"], "execution": table["execution"], "teacher": table["teacher"],
            "floor_kl": table["floor"]["kl"], "fit": {k: table["fit"][k] for k in ("chi2_dof", "passes")},
            "scope": "the recorded calibration text, teacher and execution policy",
            "ceiling": "max_loss constrains the predicted KL; it is a model estimate, not a bound"}
