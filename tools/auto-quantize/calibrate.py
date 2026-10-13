"""Teacher-relative quality calibration around the prepared floor recipe.

Teacher logits come from the source model on the calibration text (llama-perplexity --kl-divergence-base). With
RAM-routed experts the teacher keeps them in RAM (-cmoe) and uses a large batch, so expert weights stream to the GPU
once per batch of chunks; the source does not need to fit in VRAM.

Probes measure the KL of the floor recipe with some groups upgraded, paired with the floor on the same tokens. One
hotswap llama-perplexity session per device holds the floor and reloads isolated tensor patches. Two probe qtypes are
used: top (the smallest local error) and mid (local error near the geometric mean of the floor and top errors). Per
role, contiguous layer bands are upgraded to top and to mid; a few mixed recipes upgrade random groups, and the two
uniform recipes run the probe models directly. The cost is 2 x bands x roles + recipes passes, independent of the
number of candidate qtypes and of the layer count. The fit gives the table of quality.py.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import ROOT, QuantError, file_identity, imatrix_identities, write_json
from quality import METHOD, METRIC, SCHEMA, damage_digest, damage_records, modelled_groups, sensitivity
from timing import gpu


def hotswap_module():
    spec = importlib.util.spec_from_file_location("hotswap", ROOT / "tools/perplexity/hotswap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sources:
    """Lazy GGUF readers keyed by qtype label."""

    def __init__(self, paths):
        from gguf import GGUFReader
        self.reader_type = GGUFReader
        self.paths, self.index = dict(paths), {}

    def tensor(self, key, name):
        if key not in self.index:
            self.index[key] = {t.name: t for t in self.reader_type(self.paths[key]).tensors}
        return self.index[key][name]


def write_patch(path, sources, tensors):
    from gguf import GGUFWriter
    writer = GGUFWriter(str(path), None)
    for name, key in sorted(tensors.items()):
        t = sources.tensor(key, name)
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def per_chunk(path, chunks):
    values = np.fromfile(path, dtype=np.float32).astype(np.float64)
    if values.size == 0 or values.size % chunks:
        raise QuantError(f"per-token KL file is empty or has an incomplete chunk: {path}")
    return values.reshape(chunks, -1).mean(axis=1)


def probe_qtypes(groups, records, c):
    """(mid, top): top has the smallest mean log local error, mid the error nearest the floor/top geometric mean."""
    if c.get("probe_qtypes"):
        mid, top = (q.upper() for q in c["probe_qtypes"])
    else:
        common = set.intersection(*(set(g["pool"]) for g in groups)) - {g["floor"] for g in groups}

        def level(q):
            values = [records.get((n, q)) for g in groups for n in g["members"]]
            values = [v for v in values if v is not None and v > 0]
            return float(np.mean(np.log(values))) if values else None
        levels = {q: level(q) for q in common}
        levels = {q: v for q, v in levels.items() if v is not None}
        if len(levels) < 2:
            raise QuantError("calibration needs two quantized candidate qtypes above the floor in every modelled group")
        top = min(levels, key=levels.get)
        floor_level = float(np.mean([level(q) for q in {g["floor"] for g in groups}]))
        target = (floor_level + levels[top]) / 2
        mid = min((q for q in levels if q != top), key=lambda q: abs(levels[q] - target))
    for q in (mid, top):
        if any(q not in g["pool"] for g in groups):
            raise QuantError(f"probe qtype {q} is not a candidate of every modelled group")
    return mid, top


def band_layers(layers, count):
    """Contiguous bands of sorted layer indices."""
    layers = sorted(layers)
    count = min(count, len(layers))
    return [layers[b * len(layers) // count:(b + 1) * len(layers) // count] for b in range(count)]


def role_bands(groups, count):
    """{role: [layers of each band]} as used by the probe design and the fit."""
    by_role = defaultdict(set)
    for g in groups:
        by_role[g["role"]].add(g["layer"])
    return {role: band_layers(layers, count) for role, layers in sorted(by_role.items())}


def design(groups, mid, top, c):
    """Band jobs per role and mixed recipes; uniform recipes are direct runs of the probe models."""
    jobs = []
    for role, bands in role_bands(groups, c["bands"]).items():
        for b, selected in enumerate(bands):
            for q in (top, mid):
                jobs.append({"label": f"{role}-band{b:02d}-{q}",
                             "choices": {g["name"]: q for g in groups if g["role"] == role and g["layer"] in selected}})
    rng = random.Random(c["seed"])
    densities = (0.15, 0.35, 0.6)
    for i in range(max(0, c["recipes"] - 2)):
        p = densities[i % len(densities)]
        choices = {g["name"]: rng.choice((mid, top)) for g in groups if rng.random() < p}
        if choices:
            jobs.append({"label": f"mix{i:02d}-p{round(100 * p)}", "choices": choices})
    for q in (mid, top):
        jobs.append({"label": f"uniform-{q}", "choices": {g["name"]: q for g in groups}, "direct": q})
    return jobs


def perplexity_command(cfg, c, model, device, teacher, base=False):
    ram = "RAM" in cfg["devices"] and device == gpu(cfg)
    batch = c["batch"] or (16 * c["context"] if ram else c["context"])
    command = [str(Path(cfg["bindir"]) / "llama-perplexity"), "-m", str(model), "-f", c["text"], "--chunks", str(c["chunks"]),
               "--kl-divergence-base", str(teacher), "-c", str(c["context"]), "-b", str(batch), "-ub", str(batch),
               "-t", str(c["threads"]), "-tb", str(c["threads"]), "--numa", "tensors", "--device", device, "-ngl", "999"]
    if ram:
        command.append("-cmoe")
    if not base:
        command.append("--kl-divergence")
    return command


def run_direct(command, tokens, log, env, timeout):
    environment = os.environ | env | {"LLAMA_KLD_TOKENS_FILE": str(tokens)}
    with open(log, "w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, timeout=timeout)
    if completed.returncode:
        raise QuantError(f"llama-perplexity failed ({completed.returncode}); see {log}")


def run_session(command, jobs, sources, floor_choices, chunks, output, env, timeout, restore_bytes=16 << 30):
    """Measure the floor, a recompute control, every job and a restore control; return {label: per-chunk KL}.

    Completed jobs are kept in output/run.json and skipped when the session is run again."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokens = output / "latest.f32"
    record_path = output / "run.json"
    old = json.loads(record_path.read_text())["jobs"] if record_path.exists() else []
    record = {"command": command, "jobs": [j for j in old if j["label"] not in ("floor", "recompute", "restore")
                                           and (output / f"{j['label']}.f32").exists()]}
    done = {j["label"] for j in record["jobs"]}
    pending = [j for j in jobs if j["label"] not in done]
    previous_floor = (output / "floor.f32").read_bytes() if done and (output / "floor.f32").exists() else None
    results = {}
    if not pending and previous_floor is not None:
        return {job["label"]: per_chunk(output / f"{job['label']}.f32", chunks) for job in jobs}

    def save(label, result):
        tokens.replace(output / f"{label}.f32")
        results[label] = per_chunk(output / f"{label}.f32", chunks)
        record["jobs"].append({"label": label, "seconds": result["seconds"], "kl": float(results[label].mean())})
        write_json(record_path, record)

    session = hotswap_module().HotswapSession(command, output / f"session-{time.time_ns()}", timeout,
                                              env=env | {"LLAMA_KLD_TOKENS_FILE": str(tokens)})
    try:
        save("floor", session.start())
        if previous_floor is not None and (output / "floor.f32").read_bytes() != previous_floor:
            raise QuantError(f"floor differs from the earlier session in {output}; remove it to measure again")
        save("recompute", session.request("compute"))
        patch = output / "patch.gguf"
        last = 0
        for job in pending:
            tensors = {n: q for g, q in job["choices"].items() for n in sources.members[g] if q != floor_choices[g]}
            write_patch(patch, sources, tensors)
            size = patch.stat().st_size
            # A reload keeps the previous patch until it commits; free a large previous patch first.
            if last and last + size > restore_bytes:
                session.request("restore")
            last = size
            result = session.request("reload", patch)
            applied = set(re.findall(r"hotswap: reloaded tensor '([^']+)'", result["log"]))
            if set(tensors) - applied:
                raise QuantError(f"hotswap did not apply every tensor of {job['label']}")
            save(job["label"], result)
        if pending:
            save("restore", session.request("restore"))
        patch.unlink(missing_ok=True)
    finally:
        session.close()
    for label in ("recompute", "restore") if pending else ("recompute",):
        if not np.array_equal(np.fromfile(output / f"{label}.f32", dtype=np.float32),
                              np.fromfile(output / "floor.f32", dtype=np.float32)):
            raise QuantError(f"{label} control differs from the floor; the session is not reproducible")
    for job in jobs:
        if job["label"] not in results:
            results[job["label"]] = per_chunk(output / f"{job['label']}.f32", chunks)
    return results


def levenberg_marquardt(fun, x0, iterations=200):
    """Minimize |fun(x)|^2 from x0 with a forward-difference Jacobian; an overflowing trial step is rejected."""
    raw = fun

    def fun(v):
        try:
            out = raw(v)
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
        return out if np.all(np.isfinite(out)) else None
    x = np.array(x0, dtype=float)
    r = fun(x)
    if r is None:
        raise QuantError("quality fit start point is not finite")
    cost, damping = float(r @ r), 1e-3
    for _ in range(iterations):
        jac = np.empty((r.size, x.size))
        for i in range(x.size):
            h = 1e-6 * (1 + abs(x[i]))
            step = x.copy()
            step[i] += h
            value = fun(step)
            jac[:, i] = 0.0 if value is None else (value - r) / h
        a, g = jac.T @ jac, jac.T @ r
        improved = False
        while damping < 1e10:
            delta = np.linalg.solve(a + damping * np.diag(np.diag(a) + 1e-12), -g)
            rn = fun(x + delta)
            cn = math.inf if rn is None else float(rn @ rn)
            if cn < cost:
                improved = cost - cn > 1e-12 * cost
                x, r, cost, damping = x + delta, rn, cn, max(damping / 3, 1e-12)
                break
            damping *= 4
        if not improved:
            break
    return x, r


class Fit:
    """Parameters of quality.py. Per role: log sensitivities at the probe band centres (piecewise linear in the layer)
    and gamma; global: kappa_inf and u0. A penalty on second differences of the log sensitivities keeps the depth
    profile smooth unless the bands require a step. Bounded values use smooth transforms."""

    SMOOTH = 0.5  # log-sensitivity second-difference scale of the penalty

    def __init__(self, groups, records, n_layer, bands):
        self.groups, self.n_layer = groups, n_layer
        self.knots = {role: [float(np.mean(b)) for b in layers] for role, layers in role_bands(groups, bands).items()}
        self.roles = sorted(self.knots)
        self.metric = {}
        for g in groups:
            for q in g["pool"]:
                values = [records.get((n, q)) for n in g["members"]]
                if any(v is None for v in values) and q == g["floor"]:
                    raise QuantError(f"no sampled damage for the floor of {g['name']}")
                self.metric[g["name"], q] = sum(v or 0.0 for v in values)

    def unpack(self, z):
        params, i = {}, 0
        for r in self.roles:
            n = len(self.knots[r])
            params[r] = {"knots": self.knots[r], "log_s": [float(v) for v in z[i:i + n]],
                         "gamma": float(0.05 + 2.95 / (1 + math.exp(-z[i + n])))}
            i += n + 1
        kinf = float(1 + 9 / (1 + math.exp(-z[i])))
        u0 = float(0.01 + 4.99 / (1 + math.exp(-z[i + 1])))
        return params, kinf, u0

    def predict(self, params, kinf, u0, choices_list):
        floor, contribution = 0.0, {}
        for g in self.groups:
            p = params[g["role"]]
            s = sensitivity(p["knots"], p["log_s"], g["layer"])
            contribution[g["name"]] = (s, p["gamma"], self.metric[g["name"], g["floor"]] ** p["gamma"])
            floor += s * contribution[g["name"]][2]
        out = []
        for choices in choices_list:
            total = 0.0
            for name, q in choices.items():
                s, gamma, base = contribution[name]
                total += s * (base - self.metric[name, q] ** gamma)
            u = total / floor
            out.append((1 + (kinf - 1) * (1 - math.exp(-u / u0))) * total)
        return np.array(out)

    def penalty(self, z):
        out, i = [], 0
        for r in self.roles:
            n = len(self.knots[r])
            c = z[i:i + n]
            out += [(c[k - 1] - 2 * c[k] + c[k + 1]) / self.SMOOTH for k in range(1, n - 1)]
            i += n + 1
        return np.array(out)

    def fit(self, observations):
        """observations: [(choices, gain / floor KL, se)]; returns (params, kinf, u0, chi2/dof of the data)."""
        y = np.array([o[1] for o in observations])
        se = np.array([o[2] for o in observations])
        choices = [o[0] for o in observations]
        mean_metric = np.mean([self.metric[g["name"], g["floor"]] for g in self.groups])
        best = None
        for gamma0 in (0.7, 1.0, 1.5):
            zg = math.log((gamma0 - 0.05) / (3.0 - gamma0))
            c0 = math.log(1.0 / (len(self.groups) * max(mean_metric, 1e-12) ** gamma0))
            z0 = []
            for r in self.roles:
                z0 += [c0] * len(self.knots[r]) + [zg]
            z0 += [math.log(0.6 / 9.4), 0.0]

            def residuals(v):
                return np.concatenate([(self.predict(*self.unpack(v), choices) - y) / se, self.penalty(v)])
            z, r = levenberg_marquardt(residuals, z0)
            if best is None or float(r @ r) < best[1]:
                best = (z, float(r @ r))
        params, kinf, u0 = self.unpack(best[0])
        data = (self.predict(params, kinf, u0, choices) - y) / se
        free = sum(len(k) + 1 for k in self.knots.values()) + 2
        return params, kinf, u0, float(data @ data) / max(1, len(observations) - free)


def teacher_logits(cfg, c, output, timeout):
    if c.get("teacher"):
        return Path(c["teacher"])
    teacher = output / "teacher.kld"
    if teacher.exists():
        return teacher
    partial = output / "teacher.partial.kld"
    command = perplexity_command(cfg, c, cfg["model"], gpu(cfg), partial, base=True)
    print("Computing teacher logits from the source model", flush=True)
    run_direct(command, output / "teacher.tokens.f32", output / "teacher.log", cfg["runtime"]["env"], timeout)
    partial.replace(teacher)
    write_json(output / "teacher.json", {"command": command})
    return teacher


def calibrate(data, cfg, output, encode, verify, timeout=7200):
    """Measure the probes and write the quality table; encode/verify are the advisor's model writer and checker."""
    c = cfg["calibration"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = data["model"]
    floor_choices = data["baseline"]["choices"]
    groups = modelled_groups(info, floor_choices)
    if not groups:
        raise QuantError("no group has more than one candidate qtype; nothing to calibrate")
    records = damage_records(data["damage"])
    mid, top = probe_qtypes(groups, records, c)
    print(f"Probe qtypes: mid {mid}, top {top}", flush=True)
    paths = {}
    for q in (mid, top):
        choices = dict(floor_choices)
        choices.update({g["name"]: q for g in groups})
        path = output / f"probe-{q}.gguf"
        if path.exists():
            verify(info, {n: choices[g["name"]] for g in info["groups"] for n in g["members"]}, path)
        else:
            encode(info, cfg, choices, data["imatrix"], path)
        paths[q] = str(path)
    teacher = teacher_logits(cfg, c, output, timeout)
    jobs = design(groups, mid, top, c)
    devices = c["devices"] or [gpu(cfg)]
    sources = Sources(paths)
    sources.members = {g["name"]: g["members"] for g in info["groups"]}
    hot = [j for j in jobs if "direct" not in j]
    direct = [j for j in jobs if "direct" in j]
    shares = {d: hot[i::len(devices)] for i, d in enumerate(devices)}
    results, errors = {}, []
    env = dict(cfg["runtime"]["env"])
    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))

    def worker(device):
        try:
            command = perplexity_command(cfg, c, data["baseline"]["path"], device, teacher)
            out = run_session(command, shares[device], sources, floor_choices, c["chunks"], output / f"session-{device}", env, timeout)
            results[device] = out
        except Exception as exc:  # reported after every session ends
            errors.append(f"{device}: {exc}")
    print(f"Measuring {len(hot)} patch probes on {', '.join(devices)} and {len(direct)} uniform recipes", flush=True)
    try:
        threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise QuantError("calibration session failed: " + "; ".join(errors))
        direct_out = output / f"session-{devices[0]}"
        for job in direct:
            target = direct_out / f"{job['label']}.f32"
            if not target.exists():
                command = perplexity_command(cfg, c, paths[job["direct"]], devices[0], teacher)
                run_direct(command, direct_out / "latest.f32", direct_out / f"{job['label']}.log", env, timeout)
                (direct_out / "latest.f32").replace(target)
            results.setdefault(devices[0], {})[job["label"]] = per_chunk(target, c["chunks"])
    finally:
        signal.signal(signal.SIGTERM, previous)
    observations, probes, floors = [], [], {}
    for device, out in results.items():
        floor = per_chunk(output / f"session-{device}" / "floor.f32", c["chunks"])
        floors[device] = float(floor.mean())
        for job in jobs:
            if job["label"] in out and job["label"] not in ("floor", "recompute", "restore"):
                d = (floor - out[job["label"]]) / floor.mean()
                y, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
                observations.append((job["choices"], y, max(se, 1e-6)))
                probes.append({"label": job["label"], "device": device, "gain": y, "se": se,
                               "kl": float(out[job["label"]].mean()), "choices": job["choices"]})
    n_layer = 1 + max(t["layer"] for t in info["tensors"] if not t.get("mtp"))
    fit = Fit(groups, records, n_layer, c["bands"])
    params, kinf, u0, chi2 = fit.fit(observations)
    predicted = fit.predict(params, kinf, u0, [o[0] for o in observations])
    for probe, p in zip(probes, predicted):
        probe["predicted"] = float(p)
    teacher_identity = file_identity(teacher, content=True)
    table = {"schema": SCHEMA, "method": METHOD, "metric": METRIC,
             "source_files": [file_identity(f["path"], content=True) for f in info["files"]],
             "imatrices": imatrix_identities(cfg),
             "damage": damage_digest(data["damage"]),
             "floor": {"choices": floor_choices, "kl": floors[devices[0]], "kl_by_device": floors,
                       **file_identity(data["baseline"]["path"])},
             "teacher": {**teacher_identity, "text": file_identity(c["text"], content=True), "chunks": c["chunks"],
                         "context": c["context"]},
             "execution": {"devices": devices, "ram_routed": "RAM" in cfg["devices"] and devices == [gpu(cfg)],
                           "numa": "tensors", "command": perplexity_command(cfg, c, "FLOOR", devices[0], "TEACHER"),
                           "env": env},
             "probe_qtypes": {"mid": mid, "top": top}, "n_layer": n_layer, "roles": params,
             "composition": {"kappa_inf": kinf, "u0": u0},
             "fit": {"chi2_dof": chi2, "passes": len(jobs), "probes": probes}}
    path = output / "quality.json"
    write_json(path, table)
    print(f"Saved {path}: {len(jobs)} passes, chi2/dof {chi2:.2f}, floor KL {table['floor']['kl']:.5f}")
    for role, p in params.items():
        print(f"  {role}: gamma {p['gamma']:.2f}, log s at layers " + " ".join(f"{k:g}:{v:.2f}" for k, v in zip(p["knots"], p["log_s"])))
    print(f"  composition: kappa_inf {kinf:.2f}, u0 {u0:.2f}")
    return table
