"""Teacher-relative quality calibration around the prepared floor recipe.

Teacher logits come from the source model on the calibration text (llama-perplexity --kl-divergence-base). With
RAM-routed experts the teacher keeps them in RAM (-cmoe) and uses a large batch, so expert weights stream to the GPU
once per batch of chunks; the source does not need to fit in VRAM.

Probes measure the KL of the floor recipe with some groups upgraded, paired with the floor on the same tokens. One
hotswap llama-perplexity session per device holds the floor and reloads isolated tensor patches. Two probe qtypes are
used: top (the smallest local error) and mid (local error near the geometric mean of the floor and top errors). Per
role, contiguous layer bands are upgraded to top and to mid; a few mixed recipes upgrade random groups, and the two
uniform recipes upgrade every modelled group. The cost is 2 x bands x roles + recipes passes, independent of the
number of candidate qtypes and of the layer count. The fit gives the table of quality.py.

Patches are quantized from the source per job (llama-quantize --only-tensors) in chunks of at most patch_mib, so the
temporary disk space is one chunk, not two probe models. A recipe larger than one chunk is applied chunk by chunk in
the session. A uniform recipe whose RAM copy does not fit runs as a separate probe model, one at a time, deleted after
its run.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from config import ROOT, QuantError, file_identity, imatrix_identities, write_json
from corpus import calibration_text
from model import lazy_args, tensor_bytes
from quality import METHOD, METRIC, SCHEMA, damage_digest, damage_records, modelled_groups, sensitivity
from timing import gpu, gpu_only, measure_cmoe, measure_env, run

# Running hotswap sessions; SIGTERM ends them so that the session threads stop at once
LIVE_SESSIONS = set()


def hotswap_module():
    spec = importlib.util.spec_from_file_location("hotswap", ROOT / "tools/perplexity/hotswap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PatchWriter:
    """Writes isolated tensor patches straight from the source with llama-quantize --only-tensors.

    A patch holds only the tensors of one job chunk, quantized with the same source, imatrix and options as encode, so
    no full probe model is written. Patches stay at most patch_mib (a chunk with one larger group excepted)."""

    def __init__(self, cfg, info, imatrix, scratch, patch_mib):
        self.cfg, self.imatrix, self.scratch = cfg, imatrix, Path(scratch)
        self.shapes = {t["name"]: t["shape"] for t in info["tensors"]}
        self.patch_bytes = int(patch_mib) << 20
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.count, self.seconds, self.bytes = 0, 0.0, 0

    def bytes_of(self, tensors):
        return sum(tensor_bytes(self.shapes[n], q) for n, q in tensors.items())

    def chunks(self, tensors, members):
        """Split {tensor: qtype} into chunks of whole groups (members: group -> tensor names) of at most patch_bytes."""
        result, current, size = [], {}, 0
        for group in members:
            part = {n: tensors[n] for n in members[group] if n in tensors}
            if not part:
                continue
            b = self.bytes_of(part)
            if current and size + b > self.patch_bytes:
                result.append(current)
                current, size = {}, 0
            current.update(part)
            size += b
        if current:
            result.append(current)
        return result

    def write(self, tensors, path):
        from gguf import GGUFReader
        path = Path(path)
        path.unlink(missing_ok=True)
        types = path.with_suffix(".types.txt")
        names = path.with_suffix(".names.txt")
        types.write_text("".join(f"^{re.escape(n)}$={q.lower()}\n" for n, q in sorted(tensors.items())))
        names.write_text("".join(f"{n}\n" for n in sorted(tensors)))
        command = [str(Path(self.cfg["bindir"]) / "llama-quantize"), "--pure", "--tensor-type-file", str(types),
                   "--only-tensors", str(names)]
        if self.imatrix:
            command += ["--imatrix", str(self.imatrix)]
        if self.cfg.get("allow_requantize", False):
            command += ["--allow-requantize"]
        command += [self.cfg["model"], str(path), "Q2_K", str(self.cfg["runtime"]["threads"])]
        started = time.monotonic()
        run(command, path.with_suffix(".log"), self.cfg["runtime"]["env"])
        found = {t.name: t.tensor_type.name for t in GGUFReader(str(path)).tensors}
        if found != {n: q.upper() for n, q in tensors.items()}:
            raise QuantError(f"patch {path} does not hold the planned tensors and types")
        self.count += 1
        self.seconds += time.monotonic() - started
        self.bytes += path.stat().st_size
        return path


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
    # No --numa tensors: op offload runs the experts on the GPU, so the node-local copy of every CPU expert only costs
    # RAM (the KL is bit-identical without it, and a big source plus probes in /dev/shm left no room for the copy).
    ram = measure_cmoe(cfg) and (device == gpu(cfg) or gpu_only(cfg))
    batch = c["batch"] or (16 * c["context"] if ram else c["context"])
    command = [str(Path(cfg["bindir"]) / "llama-perplexity"), "-m", str(model), "-f", c["text"], "--chunks", str(c["chunks"]),
               "--kl-divergence-base", str(teacher), "-c", str(c["context"]), "-b", str(batch), "-ub", str(batch),
               "-t", str(c["threads"]), "-tb", str(c["threads"]), "--device", device, "-ngl", "999"]
    if ram:
        command.append("-cmoe")
    command += lazy_args(cfg)
    if not base:
        command.append("--kl-divergence")
    return command


def run_direct(command, tokens, log, env, timeout):
    environment = os.environ | env | {"LLAMA_KLD_TOKENS_FILE": str(tokens)}
    with open(log, "w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, timeout=timeout)
    if completed.returncode:
        raise QuantError(f"llama-perplexity failed ({completed.returncode}); see {log}")


def run_session(command, jobs, writer, members, floor_choices, chunks, output, env, timeout, restore_bytes=16 << 30):
    """Measure the floor, a recompute control, every job and a restore control; return {label: per-chunk KL}.

    A job of one patch chunk is an isolated reload. A larger job (a uniform or dense mixed recipe) goes back to the
    floor, adds its chunks one after another and then computes, so the patch on disk stays one chunk. A job that fails
    to apply in the session (memory) is skipped and named in the result key "failed". Completed jobs are kept in
    output/run.json and skipped when the session is run again."""
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

    started = time.monotonic()

    def save(label, result):
        tokens.replace(output / f"{label}.f32")
        results[label] = per_chunk(output / f"{label}.f32", chunks)
        record["jobs"].append({"label": label, "seconds": result["seconds"], "kl": float(results[label].mean())})
        write_json(record_path, record)
        k = sum(1 for j in pending if j["label"] in results)
        if label not in ("floor", "recompute", "restore") and k:
            elapsed = time.monotonic() - started
            left = elapsed / k * (len(pending) - k)
            print(f"[calibration {output.name}] pass {k}/{len(pending)} {label} KL {results[label].mean():.5f}, "
                  f"elapsed {elapsed / 60:.1f} min, about {left / 60:.1f} min left (finish {time.strftime('%H:%M', time.localtime(time.time() + left))})",
                  flush=True)

    session = hotswap_module().HotswapSession(command, output / f"session-{time.time_ns()}", timeout,
                                              env=env | {"LLAMA_KLD_TOKENS_FILE": str(tokens)})
    LIVE_SESSIONS.add(session)
    try:
        save("floor", session.start())
        if previous_floor is not None and (output / "floor.f32").read_bytes() != previous_floor:
            raise QuantError(f"floor differs from the earlier session in {output}; remove it to measure again")
        save("recompute", session.request("compute"))
        last, failed, active = 0, [], False

        def applied(result):
            return set(re.findall(r"hotswap: reloaded tensor '([^']+)'", result["log"]))
        # One item per patch chunk; the next chunk is written while the session evaluates the current one, so the
        # quantization overlaps the passes and at most two chunks are on disk.
        items = []
        for job in pending:
            tensors = {n: q for g, q in job["choices"].items() for n in members[g] if q != floor_choices[g]}
            parts = writer.chunks(tensors, {g: members[g] for g in job["choices"]})
            items += [(job, tensors, part, k, len(parts)) for k, part in enumerate(parts)]
        names = [writer.scratch / f"patch-{output.name}-{k}.gguf" for k in range(2)]
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(writer.write, items[0][2], names[0]) if items else None
        done, ok = set(), True
        try:
            for i, (job, tensors, part, k, count) in enumerate(items):
                patch = future.result()
                if i + 1 < len(items):
                    future = pool.submit(writer.write, items[i + 1][2], names[(i + 1) % 2])
                if count == 1:
                    size = writer.bytes_of(part)
                    # A reload keeps the previous patch until it commits; free a large previous patch first.
                    if last and last + size > restore_bytes:
                        session.request("reset")
                    last, active = size, True
                    result = session.request("reload", patch)
                    patch.unlink(missing_ok=True)
                    if set(tensors) - applied(result):
                        raise QuantError(f"hotswap did not apply every tensor of {job['label']}")
                    save(job["label"], result)
                    continue
                if k == 0:
                    session.request("reset")
                    done, ok = set(), True
                if ok:
                    try:
                        done |= applied(session.request("add", patch))
                    except RuntimeError as exc:
                        print(f"{job['label']}: cannot apply in the session ({exc}); it runs as a separate model", flush=True)
                        ok = False
                patch.unlink(missing_ok=True)
                if k + 1 < count:
                    continue
                if ok:
                    if set(tensors) - done:
                        raise QuantError(f"hotswap did not apply every tensor of {job['label']}")
                    save(job["label"], session.request("compute"))
                else:
                    failed.append(job["label"])
                session.request("reset")
                last, active = 0, False
        finally:
            pool.shutdown(wait=True)
            for name in names:
                name.unlink(missing_ok=True)
        if pending:
            # The control evaluates the floor again after the patches; a reset already went back to it.
            save("restore", session.request("restore" if active else "compute"))
        if failed:
            results["failed"] = failed
    finally:
        LIVE_SESSIONS.discard(session)
        session.close()
    for label in ("recompute", "restore") if pending else ("recompute",):
        if not np.array_equal(np.fromfile(output / f"{label}.f32", dtype=np.float32),
                              np.fromfile(output / "floor.f32", dtype=np.float32)):
            raise QuantError(f"{label} control differs from the floor; the session is not reproducible")
    for job in jobs:
        if job["label"] not in results and job["label"] not in results.get("failed", ()):
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
    text = file_identity(c["text"], content=True)
    if teacher.exists():
        record = output / "teacher.json"
        recorded = json.loads(record.read_text()).get("text", {}).get("sha256") if record.exists() else None
        if recorded not in (None, text["sha256"]):
            raise QuantError(f"teacher logits in {output} come from a different calibration text; use a new output directory")
        return teacher
    partial = output / "teacher.partial.kld"
    command = perplexity_command(cfg, c, cfg["model"], gpu(cfg), partial, base=True)
    print("Computing teacher logits from the source model", flush=True)
    run_direct(command, output / "teacher.tokens.f32", output / "teacher.log", measure_env(cfg), timeout)
    partial.replace(teacher)
    write_json(output / "teacher.json", {"command": command, "text": text})
    return teacher


def available_ram():
    """MemAvailable in bytes, or None where /proc/meminfo is absent."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) << 10
    except (OSError, ValueError, IndexError):
        pass
    return None


def calibrate(data, cfg, output, encode, verify, timeout=7200, text_dir=None, scratch=None):
    """Measure the probes and write the quality table; encode/verify are the advisor's model writer and checker.

    Weighted calibration corpora are composed into one text in text_dir (the prepared directory) first. Patches (and
    a probe model for a uniform recipe that runs as a separate model) go to scratch (default: output)."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    text, corpora = calibration_text(cfg["calibration"], text_dir or output)
    c = dict(cfg["calibration"], text=text)
    info = data["model"]
    floor_choices = data["baseline"]["choices"]
    groups = modelled_groups(info, floor_choices)
    if not groups:
        raise QuantError("no group has more than one candidate qtype; nothing to calibrate")
    records = damage_records(data["damage"])
    mid, top = probe_qtypes(groups, records, c)
    print(f"Probe qtypes: mid {mid}, top {top}", flush=True)
    jobs = design(groups, mid, top, c)
    devices = c["devices"] or [gpu(cfg)]
    members = {g["name"]: g["members"] for g in info["groups"]}
    writer = PatchWriter(cfg, info, data["imatrix"], Path(scratch or output), c["patch_mib"])
    # A uniform recipe holds every modelled tensor at the probe qtype next to the floor in the session. With routed
    # experts in RAM, it runs as a separate probe model instead when that copy does not fit in the available RAM.
    ram = measure_cmoe(cfg) and (devices == [gpu(cfg)] or gpu_only(cfg))
    free = available_ram()

    def separate(job):
        if c["uniform"] != "auto":
            return c["uniform"] == "direct"
        need = writer.bytes_of({n: job["direct"] for g in groups for n in g["members"] if "_exps" in n}) if ram else 0
        return free is not None and need > free - (16 << 30)
    dev0 = output / f"session-{devices[0]}"
    recorded = json.loads((dev0 / "run.json").read_text())["jobs"] if (dev0 / "run.json").exists() else []
    recorded = {j["label"] for j in recorded}
    # Uniform results from a separate run (older work directories) are kept.
    direct = [j for j in jobs if "direct" in j and (separate(j) or ((dev0 / f"{j['label']}.f32").exists()
                                                                   and j["label"] not in recorded))]
    hot = [j for j in jobs if j not in direct]
    shares = {d: hot[i::len(devices)] for i, d in enumerate(devices)}
    teacher = teacher_logits(cfg, c, output, timeout)
    results, errors = {}, []
    env = measure_env(cfg)

    def terminate(*_):
        # The session threads are not daemons: end their llama-perplexity processes so that they stop now.
        for session in list(LIVE_SESSIONS):
            if session.process and session.process.poll() is None:
                session.process.terminate()
        sys.exit(1)
    previous = signal.signal(signal.SIGTERM, terminate)

    def worker(device):
        try:
            command = perplexity_command(cfg, c, data["baseline"]["path"], device, teacher)
            out = run_session(command, shares[device], writer, members, floor_choices, c["chunks"],
                              output / f"session-{device}", env, timeout)
            results[device] = out
        except Exception as exc:  # reported after every session ends
            errors.append(f"{device}: {exc}")
    print(f"Measuring {len(hot)} patch probes on {', '.join(devices)}"
          + (f" and {len(direct)} uniform recipes as separate models" if direct else ""), flush=True)
    try:
        threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise QuantError("calibration session failed: " + "; ".join(errors))
        failed = {label for out in results.values() for label in out.pop("failed", ())}
        direct += [j for j in jobs if j["label"] in failed and j not in direct]
        for job in direct:
            target = dev0 / f"{job['label']}.f32"
            if not target.exists():
                # One probe model at a time, deleted after its run.
                choices = dict(floor_choices)
                choices.update(job["choices"])
                path = writer.scratch / f"probe-{job['label']}.gguf"
                path.unlink(missing_ok=True)
                encode(info, cfg, choices, data["imatrix"], path)
                try:
                    command = perplexity_command(cfg, c, path, devices[0], teacher)
                    run_direct(command, dev0 / "latest.f32", dev0 / f"{job['label']}.log", env, timeout)
                    (dev0 / "latest.f32").replace(target)
                finally:
                    for suffix in (".gguf", ".encode.json", ".encode.log", ".tensor-types.txt"):
                        path.with_suffix(suffix).unlink(missing_ok=True)
            results.setdefault(devices[0], {})[job["label"]] = per_chunk(target, c["chunks"])
    finally:
        signal.signal(signal.SIGTERM, previous)
    if writer.count:
        print(f"Patches: {writer.count} written from the source, {writer.bytes / 2**30:.1f} GiB in total, "
              f"{writer.seconds / 60:.1f} min", flush=True)
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
             "teacher": {**teacher_identity, "text": {**file_identity(c["text"], content=True), **({"corpora": corpora} if corpora else {})},
                         "chunks": c["chunks"], "context": c["context"]},
             "execution": {"devices": devices, "ram_routed": measure_cmoe(cfg) and (devices == [gpu(cfg)] or gpu_only(cfg)),
                           "numa": "off", "command": perplexity_command(cfg, c, "FLOOR", devices[0], "TEACHER"),
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
