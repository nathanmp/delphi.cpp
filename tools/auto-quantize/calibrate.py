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
temporary disk space is one chunk, not two probe models. A chunk stays in a cache while a later job needs one of its
tensors (see PatchWriter), when the scratch disk has the room. A recipe larger than one chunk is applied chunk by chunk in
the session. A uniform recipe whose RAM copy does not fit runs as a separate probe model, one at a time, deleted after
its run.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import os
import random
import re
import shutil
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
from timing import gpu, gpu_only, gpus, measure_cmoe, measure_env, run

# Running hotswap sessions; SIGTERM ends them so that the session threads stop at once
LIVE_SESSIONS = set()


def hotswap_module():
    spec = importlib.util.spec_from_file_location("hotswap", ROOT / "tools/perplexity/hotswap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Free space that the patch cache leaves on the scratch file system (besides the two patch chunks in flight)
CACHE_MARGIN = 32 << 30


class PatchWriter:
    """Writes isolated tensor patches straight from the source with llama-quantize --only-tensors.

    A patch holds only the tensors of one job chunk, quantized with the same source, imatrix and options as encode, so
    no full probe model is written. Patches stay at most patch_mib (a chunk with one larger group excepted).

    The quantized chunks also stay in a cache (scratch/patch-cache) while a later job of expect() needs one of their
    tensors at the same qtype: the band probes quantize every expert at the probe qtypes, and the uniform recipes need
    the same tensors again. A patch of cached tensors is a hard link to the chunk, or a copy of the tensors from the
    chunks when it takes part of them. A chunk is deleted when no expected job needs it, and a new one is cached only
    when CACHE_MARGIN stays free. The cache is valid for one source, imatrix, quantizer and set of libraries."""

    def __init__(self, cfg, info, imatrix, scratch, patch_mib):
        self.cfg, self.imatrix, self.scratch = cfg, imatrix, Path(scratch)
        self.shapes = {t["name"]: t["shape"] for t in info["tensors"]}
        self.patch_bytes = int(patch_mib) << 20
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.count, self.seconds, self.bytes = 0, 0.0, 0
        self.reused, self.reused_bytes = 0, 0
        self.lock = threading.Lock()
        self.uses = defaultdict(int)  # (tensor, qtype) -> patches of expected jobs that still need it
        self.entries = {}             # (tensor, qtype) -> cached chunk
        self.cache = self.scratch / "patch-cache"
        self.open_cache(info)

    def open_cache(self, info):
        key = {"model": info.get("identity"), "imatrix": file_identity(self.imatrix) if self.imatrix else None,
               "allow_requantize": self.cfg.get("allow_requantize", False),
               "quantize": file_identity(Path(self.cfg["bindir"]) / "llama-quantize") if (Path(self.cfg["bindir"]) / "llama-quantize").exists() else None,
               "libraries": [file_identity(p) for p in sorted(Path(self.cfg["bindir"]).glob("lib*.so*")) if not p.is_symlink()]}
        record = self.cache / "key.json"
        if record.exists() and json.loads(record.read_text()) != key:
            print(f"{self.cache} holds patches of other inputs; removing it", flush=True)
            shutil.rmtree(self.cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        write_json(record, key)
        from gguf import GGUFReader
        for chunk in sorted(self.cache.glob("chunk-*.gguf")):
            try:
                for t in GGUFReader(str(chunk)).tensors:
                    self.entries[(t.name, t.tensor_type.name)] = chunk
            except (OSError, ValueError):
                chunk.unlink()

    def expect(self, jobs, floor_choices, members):
        """Count the patch tensors of jobs that will be written, so that their chunks stay cached until then."""
        with self.lock:
            for job in jobs:
                for g, q in job["choices"].items():
                    if q != floor_choices[g]:
                        for n in members[g]:
                            self.uses[(n, q.upper())] += 1

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

    def quantize(self, tensors, path):
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
        run(command, path.with_suffix(".log"), self.cfg["runtime"]["env"])

    def write(self, tensors, path):
        from gguf import GGUFReader
        path = Path(path)
        path.unlink(missing_ok=True)
        wanted = {(n, q.upper()) for n, q in tensors.items()}
        with self.lock:
            missing = {n: q for n, q in tensors.items() if (n, q.upper()) not in self.entries}
            cache = bool(missing) and any(self.uses[(n, q.upper())] > 1 for n, q in missing.items()) and (
                shutil.disk_usage(self.cache).free - self.bytes_of(missing) >= CACHE_MARGIN + 2 * self.patch_bytes)
        fresh = None
        if missing:
            started = time.monotonic()
            fresh = self.cache / f"chunk-{time.time_ns()}.gguf" if cache else path.with_suffix(".fresh.gguf")
            try:
                self.quantize(missing, fresh)
            finally:
                for suffix in (".types.txt", ".names.txt"):
                    fresh.with_suffix(suffix).unlink(missing_ok=True)
            self.count += 1
            self.seconds += time.monotonic() - started
            self.bytes += fresh.stat().st_size
        with self.lock:
            if cache:
                for n, q in missing.items():
                    self.entries[(n, q.upper())] = fresh
            sources = defaultdict(set)
            for key in wanted:
                sources[self.entries.get(key, fresh)].add(key[0])
            reused = {k for k in wanted if self.entries.get(k, fresh) != fresh}
            self.reused += len(reused)
            self.reused_bytes += self.bytes_of({n: q for n, q in reused})
            if len(sources) == 1:
                (source, names), = sources.items()
                if {t.name for t in GGUFReader(str(source)).tensors} == names:
                    if source == fresh and not cache:
                        fresh.replace(path)
                    else:
                        os.link(source, path)
                else:
                    copy_tensors(sources, path)
            else:
                copy_tensors(sources, path)
            if fresh is not None and not cache:
                fresh.unlink(missing_ok=True)
            for key in wanted:
                self.uses[key] -= 1
            self.evict()
        found = {t.name: t.tensor_type.name for t in GGUFReader(str(path)).tensors}
        if found != {n: q.upper() for n, q in tensors.items()}:
            raise QuantError(f"patch {path} does not hold the planned tensors and types")
        return path

    def evict(self):
        """Delete the cached chunks that no expected patch needs (callers hold the lock)."""
        live = defaultdict(bool)
        for key, chunk in self.entries.items():
            live[chunk] |= self.uses[key] > 0
        for chunk, needed in live.items():
            if not needed:
                chunk.unlink(missing_ok=True)
                chunk.with_suffix(".log").unlink(missing_ok=True)
                self.entries = {k: c for k, c in self.entries.items() if c != chunk}

    def close(self):
        shutil.rmtree(self.cache, ignore_errors=True)
        self.entries.clear()


def copy_tensors(sources, path):
    """Write the named tensors of the GGUF files of sources ({file: names}) to path, with the qtypes as they are."""
    from gguf import GGUFReader, GGUFWriter
    writer = GGUFWriter(str(path), "patch")
    for source, names in sources.items():
        for t in GGUFReader(str(source)).tensors:
            if t.name in names:
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


def qtype_levels(groups, records):
    """(mean log local error of every qtype above the floor in all modelled groups, mean log error of the floors)."""
    def level(q):
        values = [records.get((n, q)) for g in groups for n in g["members"]]
        values = [v for v in values if v is not None and v > 0]
        return float(np.mean(np.log(values))) if values else None
    common = set.intersection(*(set(g["pool"]) for g in groups)) - {g["floor"] for g in groups}
    levels = {q: v for q, v in ((q, level(q)) for q in common) if v is not None}
    floors = [v for v in (level(q) for q in {g["floor"] for g in groups}) if v is not None]
    return levels, float(np.mean(floors)) if floors else None


def probe_qtypes(groups, records, c):
    """(mid, top): top has the smallest mean log local error, mid the error nearest the floor/top geometric mean."""
    if c.get("probe_qtypes"):
        mid, top = (q.upper() for q in c["probe_qtypes"])
    else:
        levels, floor_level = qtype_levels(groups, records)
        if len(levels) < 2:
            raise QuantError("calibration needs two quantized candidate qtypes above the floor in every modelled group")
        top = min(levels, key=levels.get)
        target = (floor_level + levels[top]) / 2
        mid = min((q for q in levels if q != top), key=lambda q: abs(levels[q] - target))
    for q in (mid, top):
        if any(q not in g["pool"] for g in groups):
            raise QuantError(f"probe qtype {q} is not a candidate of every modelled group")
    return mid, top


def anchor_qtypes(groups, records, mid, top):
    """Qtypes for the extra uniform and mixed recipes: the errors nearest the geometric means of floor/mid and mid/top.

    The bands measure gains from the floor; a recipe far above the floor depends on the small remaining error of its
    qtypes. Uniform recipes between the probes measure that range directly (Qwen2.5 1.5B with an IQ2_XXS floor and
    probes Q5_0/Q8_0: the IQ4 picks measured 3x to 8x the KL that the band fit predicts)."""
    levels, floor_level = qtype_levels(groups, records)
    if mid not in levels or top not in levels or floor_level is None:
        return []
    anchors = []
    for high, low in ((floor_level, levels[mid]), (levels[mid], levels[top])):
        between = [q for q in levels if low < levels[q] < high]
        if between:
            anchors.append(min(between, key=lambda q: abs(levels[q] - (high + low) / 2)))
    return anchors


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


def design(groups, mid, top, c, anchors=()):
    """Band jobs per role (mid and top), mixed recipes and one uniform recipe per probe and anchor qtype."""
    jobs = []
    for role, bands in role_bands(groups, c["bands"]).items():
        for b, selected in enumerate(bands):
            for q in (top, mid):
                jobs.append({"label": f"{role}-band{b:02d}-{q}",
                             "choices": {g["name"]: q for g in groups if g["role"] == role and g["layer"] in selected}})
    rng = random.Random(c["seed"])
    densities = (0.15, 0.35, 0.6)
    qtypes = (*anchors, mid, top)
    tag = "" if not anchors else "-" + "-".join(anchors)
    for i in range(max(0, c["recipes"] - len(qtypes))):
        p = densities[i % len(densities)]
        choices = {g["name"]: rng.choice(qtypes) for g in groups if rng.random() < p}
        if choices:
            jobs.append({"label": f"mix{i:02d}-p{round(100 * p)}{tag}", "choices": choices})
    for q in qtypes:
        jobs.append({"label": f"uniform-{q}", "choices": {g["name"]: q for g in groups}, "direct": q})
    return jobs


def perplexity_command(cfg, c, model, device, teacher, base=False):
    # No --numa tensors: op offload runs the experts on the GPU, so the node-local copy of every CPU expert only costs
    # RAM (the KL is bit-identical without it, and a big source plus probes in /dev/shm left no room for the copy).
    # A device list ("CUDA0,CUDA1,CUDA2") is a layer-split session with every weight on the GPUs.
    split = "," in device
    ram = not split and measure_cmoe(cfg) and (device == gpu(cfg) or gpu_only(cfg))
    batch = c["batch"] or (min(16 * c["context"], OFFLOAD_BATCH) if ram else 4 * c["context"] if split else c["context"])
    command = [str(Path(cfg["bindir"]) / "llama-perplexity"), "-m", str(model), "-f", c["text"], "--chunks", str(c["chunks"]),
               "--kl-divergence-base", str(teacher), "-c", str(c["context"]), "-b", str(batch), "-ub", str(batch),
               "-t", str(c["threads"]), "-tb", str(c["threads"]), "--device", device, "-ngl", "999"]
    if split:
        command += ["-sm", "layer"]
    if ram:
        command.append("-cmoe")
    command += lazy_args(cfg)
    if not base:
        command.append("--kl-divergence")
    return command


# Op offload of CPU-resident IQ2_XXS experts gave NaN at -ub 4096 and garbage at 8192 (GLM 5.3 on a CMP 170HX,
# 2026-10-21; the same model is correct on the GPUs at 4096 and with op offload at 2048).
OFFLOAD_BATCH = 2048
# A floor this far from the teacher is a broken computation, not a coarse quantization (the GLM case above: 5.8)
FLOOR_KL_LIMIT = 4.0
# Per GPU in a layer-split session: compute buffers at -ub 4*context, the temporary buffers of the matrix products,
# the KV of the context, the backend context (8 GiB was not enough for GLM 5.3 with an IQ6_K band on one CMP 170HX)
SPLIT_MARGIN = 12 << 30
# A split session that ends this many times gives its remaining jobs to the other sessions
SPLIT_RESTARTS = 4


class SessionCrashed(RuntimeError):
    """The llama-perplexity process of a session ended during a job (for example out of memory in a pass)."""
    def __init__(self, label, message):
        super().__init__(f"{label}: {message}")
        self.label = label


def split_session(cfg, c, baseline):
    """(device list, free bytes per GPU) for one layer-split session over the GPUs of a GPU-only deployment when the
    floor fits on them with SPLIT_MARGIN per GPU; None otherwise (calibration.devices set, RAM routing, too large)."""
    if c["devices"] or not gpu_only(cfg):
        return None
    names = gpus(cfg)
    room = (sum(int(cfg["devices"][d]) for d in names) - Path(baseline).stat().st_size) / len(names) - SPLIT_MARGIN
    return (",".join(names), room) if room > 0 else None


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
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    if previous and (previous.get("command") != command or previous.get("env", env) != env):
        print(f"{output.name}: the command or environment changed; measuring every job again", flush=True)
        old = []
        for f in output.glob("*.f32"):
            f.unlink()
    record = {"command": command, "env": env, "jobs": [j for j in old if j["label"] not in ("floor", "recompute", "restore")
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
        if results["floor"].mean() > FLOOR_KL_LIMIT:
            raise QuantError(f"floor KL {results['floor'].mean():.2f} against the teacher: the floor model computes wrong "
                             f"results with this command (see {output}); stop before measuring noise")
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
                    try:
                        result = session.request("reload", patch)
                    except RuntimeError as exc:
                        if session.process.poll() is not None:
                            raise SessionCrashed(job["label"], str(exc)) from exc
                        # out of memory: the session keeps its tensors; the job goes to another session
                        print(f"{job['label']}: cannot apply in the session ({exc}); it runs elsewhere", flush=True)
                        patch.unlink(missing_ok=True)
                        failed.append(job["label"])
                        session.request("reset")
                        last, active = 0, False
                        continue
                    last, active = size, True
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
                        if session.process.poll() is not None:
                            raise SessionCrashed(job["label"], str(exc)) from exc
                        print(f"{job['label']}: cannot apply in the session ({exc}); it runs as a separate model", flush=True)
                        ok = False
                patch.unlink(missing_ok=True)
                if k + 1 < count:
                    continue
                if ok:
                    if set(tensors) - done:
                        raise QuantError(f"hotswap did not apply every tensor of {job['label']}")
                    try:
                        result = session.request("compute")
                    except RuntimeError as exc:
                        if session.process.poll() is not None:
                            raise SessionCrashed(job["label"], str(exc)) from exc
                        raise
                    save(job["label"], result)
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


# Lower bound of kappa_inf: above about 0.119, kappa(u) * u grows with u everywhere, so more gain never predicts more loss.
KAPPA_MIN = 0.12


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
        kinf = float(KAPPA_MIN + (10 - KAPPA_MIN) / (1 + math.exp(-z[i])))
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
        # kappa_inf starts super-additive (1.54) and sub-additive (0.6): with an IQ2 floor and Q8_0 upgrades the gains
        # saturate as the KL approaches zero (OLMoE: kappa_inf 0.12, held-out recipe 0.0254 predicted vs 0.0237 measured)
        for gamma0, kappa0 in itertools.product((0.7, 1.0, 1.5), (1.54, 0.6)):
            zg = math.log((gamma0 - 0.05) / (3.0 - gamma0))
            c0 = math.log(1.0 / (len(self.groups) * max(mean_metric, 1e-12) ** gamma0))
            z0 = []
            for r in self.roles:
                z0 += [c0] * len(self.knots[r]) + [zg]
            z0 += [math.log((kappa0 - KAPPA_MIN) / (10 - kappa0)), 0.0]

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
        recorded = json.loads(record.read_text()) if record.exists() else {}
        if recorded.get("text", {}).get("sha256") not in (None, text["sha256"]):
            raise QuantError(f"teacher logits in {output} come from a different calibration text; use a new output directory")
        if recorded.get("model") not in (None, file_identity(cfg["model"])):
            raise QuantError(f"teacher logits in {output} come from a different model; use a new output directory")
        return teacher
    partial = output / "teacher.partial.kld"
    command = perplexity_command(cfg, c, cfg["model"], gpu(cfg), partial, base=True)
    print("Computing teacher logits from the source model", flush=True)
    run_direct(command, output / "teacher.tokens.f32", output / "teacher.log", measure_env(cfg), timeout)
    partial.replace(teacher)
    write_json(output / "teacher.json", {"command": command, "text": text, "model": file_identity(cfg["model"])})
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
    anchors = anchor_qtypes(groups, records, mid, top)
    if anchors:
        print(f"Anchor qtypes: {', '.join(anchors)}", flush=True)
    jobs = design(groups, mid, top, c, anchors)
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
    # With the floor on the GPUs, a pass is about 10x faster than with the experts in RAM (no op-offload uploads).
    # The jobs whose upgrade fits next to it run there; the others (mixed and uniform recipes, and jobs that the split
    # session cannot apply) run in the sessions of `devices` as before.
    split = split_session(cfg, c, data["baseline"]["path"])
    split_jobs = []
    if split:
        def extra(job):
            return (writer.bytes_of({n: q for g, q in job["choices"].items() for n in members[g]})
                    - writer.bytes_of({n: floor_choices[g] for g in job["choices"] for n in members[g]}))
        split_jobs = [j for j in hot if extra(j) <= split[1]]
        hot = [j for j in hot if j not in split_jobs]
    shares = {d: hot[i::len(devices)] for i, d in enumerate(devices)}
    # the patch cache keeps a quantized chunk while a job that is not measured yet needs one of its tensors
    measured = {f.stem for f in output.glob("session-*/*.f32")}
    writer.expect([j for j in split_jobs + hot if j["label"] not in measured], floor_choices, members)
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
    def split_worker(device, share):
        """Run the split session; a job that ends the process goes to the other sessions and the session starts again
        (finished jobs are kept in run.json). Returns the labels of the jobs that the split session did not measure."""
        command = perplexity_command(cfg, c, data["baseline"]["path"], device, teacher)
        session_dir = output / f"session-{device}"
        crashed = []
        for _ in range(SPLIT_RESTARTS):
            try:
                # The split session has less room per GPU: it frees the previous patch before every reload
                out = run_session(command, [j for j in share if j["label"] not in crashed], writer, members, floor_choices,
                                  c["chunks"], session_dir, env, timeout, restore_bytes=0)
                results[device] = out
                return set(out.pop("failed", ())) | set(crashed)
            except SessionCrashed as exc:
                print(f"{exc.label}: the split session ended ({exc}); the job runs elsewhere, restarting the session", flush=True)
                crashed.append(exc.label)
        # too many restarts: keep the finished jobs, give the others to the -cmoe sessions
        finished = {j["label"] for j in share if (session_dir / f"{j['label']}.f32").exists()}
        print(f"The split session ended {SPLIT_RESTARTS} times; {len(share) - len(finished)} jobs run elsewhere", flush=True)
        if (session_dir / "floor.f32").exists() and finished:
            results[device] = {label: per_chunk(session_dir / f"{label}.f32", c["chunks"]) for label in finished}
        return {j["label"] for j in share} - (finished if device in results else set())

    try:
        failed = set()
        if split_jobs:
            print(f"Measuring {len(split_jobs)} patch probes on {split[0]} (layer split, every weight on the GPUs)", flush=True)
            failed = split_worker(split[0], split_jobs)
            for i, job in enumerate(j for j in split_jobs if j["label"] in failed):
                shares[devices[i % len(devices)]].append(job)
        sessions = [d for d in devices if shares[d] or direct]
        print(f"Measuring {sum(len(shares[d]) for d in sessions)} patch probes on {', '.join(sessions) or 'no device'}"
              + (f" and {len(direct)} uniform recipes as separate models" if direct else ""), flush=True)
        threads = [threading.Thread(target=worker, args=(d,)) for d in sessions]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise QuantError("calibration session failed: " + "; ".join(errors))
        failed = {label for d in sessions for label in results[d].pop("failed", ())}
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
    if writer.count or writer.reused:
        print(f"Patches: {writer.count} written from the source, {writer.bytes / 2**30:.1f} GiB in total, "
              f"{writer.seconds / 60:.1f} min; {writer.reused} tensors ({writer.reused_bytes / 2**30:.1f} GiB) "
              "taken from the patch cache", flush=True)
    writer.close()
    observations, probes, floors = [], [], {}
    for device, out in results.items():
        floor = per_chunk(output / f"session-{device}" / "floor.f32", c["chunks"])
        floors[device] = float(floor.mean())
        for job in jobs:
            if job["label"] in out and job["label"] not in ("floor", "recompute", "restore"):
                x = out[job["label"]]
                y = float(1 - x.mean() / floor.mean())
                # Ratio-estimator error of mean(x) / mean(floor) over paired chunks. The error of the paired
                # difference is the scatter of the floor itself when x is small, and it gave the uniform recipes no
                # weight (Qwen2.5 1.5B, floor KL 0.81: +-0.025 against +-0.00005 for uniform Q8_0).
                se = float((x - (1 - y) * floor).std(ddof=1) / floor.mean() / math.sqrt(len(x)))
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
             "floor": {"choices": floor_choices, "kl": floors.get(devices[0], next(iter(floors.values()))), "kl_by_device": floors,
                       **file_identity(data["baseline"]["path"])},
             "teacher": {**teacher_identity, "text": {**file_identity(c["text"], content=True), **({"corpora": corpora} if corpora else {})},
                         "chunks": c["chunks"], "context": c["context"]},
             "execution": {"devices": devices, "split_session": split[0] if split_jobs else None, "ram_routed": measure_cmoe(cfg) and (devices == [gpu(cfg)] or gpu_only(cfg)),
                           "numa": "off", "command": perplexity_command(cfg, c, "FLOOR", devices[0], "TEACHER"),
                           "env": env},
             "probe_qtypes": {"mid": mid, "top": top, "anchors": anchors}, "n_layer": n_layer, "roles": params,
             "composition": {"kappa_inf": kinf, "u0": u0},
             "fit": {"chi2_dof": chi2, "passes": len(jobs), "probes": probes}}
    path = output / "quality.json"
    write_json(path, table)
    print(f"Saved {path}: {len(jobs)} passes, chi2/dof {chi2:.2f}, floor KL {table['floor']['kl']:.5f}")
    for role, p in params.items():
        print(f"  {role}: gamma {p['gamma']:.2f}, log s at layers " + " ".join(f"{k:g}:{v:.2f}" for k, v in zip(p["knots"], p["log_s"])))
    print(f"  composition: kappa_inf {kinf:.2f}, u0 {u0:.2f}")
    return table
