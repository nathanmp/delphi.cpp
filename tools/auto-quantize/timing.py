"""Bounded backend measurements and baseline-anchored workload prediction."""
from __future__ import annotations

import csv
import io
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from config import ROOT, QuantError, digest, file_identity, write_json
from model import tensor_bytes

sys.path.insert(0, str(ROOT))
from tools.bottlenecks.profile import BenchArm, Recipe, GPU_SETTING_KEYS, _gpu_state, run_recipe


def machine_settings():
    keys = ("vendor", "pci", *GPU_SETTING_KEYS)
    return [{key: device[key] for key in keys if key in device} for device in _gpu_state()["gpus"]]


def run(command, log, env=None, timeout=1800):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    effective = os.environ.copy()
    effective.update(env or {})
    with log.open("w") as stream:
        completed = subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT,
                                   env=effective, timeout=timeout)
    if completed.returncode:
        raise QuantError(f"command failed ({completed.returncode}); see {log}")


def gpu(cfg):
    return next(name for name in cfg["devices"] if name != "RAM")


def profile_model(cfg, model, output, trace=True):
    r = cfg["runtime"]
    command = [str(Path(cfg["bindir"]) / "llama-bench"), "-m", str(model), "--numa", "tensors", "--device", gpu(cfg),
               "-ngl", "999", "-t", str(r["threads"]), "-b", str(r["batch"]), "-ub", str(r["ubatch"]),
               "-p", f"{r['short_tokens']},{r['long_tokens']}", "-n", str(r["decode_tokens"]),
               "-d", str(r["depth_tokens"]), "-r", str(r["repetitions"]), *r["extra_args"]]
    if "RAM" in cfg["devices"]:
        command += ["-ncmoe", "999"]
    environment = dict(r["env"])
    environment["LLAMA_MOE_CACHE_MIB"] = str(int(cfg["cache_mib"]))
    arm = BenchArm(name="plain", command=tuple(command), model_paths=(str(model),), env=environment,
                   trace_mode="submit" if trace else "off", trace_prefill_target_tokens=r["long_tokens"])
    recipe = Recipe(name="auto-quantize", description="Fixed V1 short/long prefill and decode with one model load.", arms=(arm,))
    result = run_recipe(recipe, Path(output).with_suffix(".artifacts"))
    write_json(output, result)
    return result


def profile_times(profile, cfg):
    tests = profile["arms"][0]["metrics"]["tests"]
    result = {}
    for item in tests:
        test = item["test"]
        prompt = int(test.get("n_prompt", 0))
        generation = int(test.get("n_gen", 0))
        rate = item["throughput_t_s"]["mean"]
        if not rate or rate <= 0:
            continue
        if generation and not prompt:
            result["decode"] = 1000 / rate
        elif prompt and not generation:
            for regime in ("short", "long"):
                if prompt == cfg["runtime"][f"{regime}_tokens"]:
                    result[regime] = prompt * 1000 / rate
    if set(result) != {"decode", "short", "long"}:
        raise QuantError("profile lacks the exact configured short/long/decode tests")
    return result


def signature(tensors, paired, dtype, device, tokens, top_k):
    t = tensors[0]
    return {"shape": t["shape"], "dtype": dtype, "device": device, "tokens": tokens,
            "expert": t["expert"], "top_k": top_k if t["expert"] else 1,
            "broadcast": "ffn_down" not in t["name"], "paired": paired,
            "numa_expert": t["expert"] and device == "CPU"}


def token_counts(cfg):
    return {"decode": 1, "short": min(cfg["runtime"]["short_tokens"], cfg["runtime"]["ubatch"]),
            "long": min(cfg["runtime"]["long_tokens"], cfg["runtime"]["ubatch"])}


def operation_times(text):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if {"op_name", "time_us", "test_mode"}.issubset(next(csv.reader([line]), []))), None)
    if start is None:
        raise QuantError("backend emitted no CSV header")
    rows = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    result = []
    for row in rows:
        if row.get("test_mode") == "perf" and row.get("time_us"):
            value = float(row["time_us"])
            if value > 0 and math.isfinite(value):
                result.append(value)
    return result


def regions(info):
    by_name = {t["name"]: t for t in info["tensors"]}
    result = []
    for group in info["groups"]:
        members = [by_name[name] for name in group["members"]]
        if all(t["preserved"] for t in members):
            continue
        # A separate embedding is a lookup. A tied output also performs a vocabulary projection when its type varies.
        if all(t["name"] == "token_embd.weight" for t in members) and ("output.weight" in by_name or len(group["pool"]) == 1):
            continue
        paired = len(members) == 2 and all(t["expert"] for t in members) and members[0]["shape"] == members[1]["shape"]
        if paired:
            result.append((group, members, True))
        else:
            for t in members:
                result.append((group, [t], False))
    return result


def measure_ops(info, cfg, directory, timeout=600):
    directory = Path(directory)
    binary = Path(cfg["bindir"]) / "test-backend-ops"
    if not binary.exists():
        raise QuantError(f"missing {binary}")
    sigs = {}
    for group, members, paired in regions(info):
        for dtype in group["pool"]:
            for tokens in sorted(set(token_counts(cfg).values())):
                devices = [gpu(cfg)]
                if members[0]["expert"] and "RAM" in cfg["devices"]:
                    devices.append("CPU")
                for device in devices:
                    sig = signature(members, paired, dtype, device, tokens, info["top_k"])
                    sigs[digest(sig)] = sig
    values = {}
    environment = cfg["runtime"]["env"]
    identity = {"binary": file_identity(binary),
                "devices": cfg["devices"], "threads": cfg["runtime"]["threads"], "env": environment, "numa": "tensors",
                "machine_settings": machine_settings()}
    cache_path = directory / "operations.json"
    if cache_path.exists():
        import json
        old = json.loads(cache_path.read_text())
        if old.get("identity") == identity:
            values = old["values"]
    import time
    deadline = time.monotonic() + timeout
    for key, sig in sorted(sigs.items()):
        if key in values:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise QuantError("operation measurement time cap reached; completed signatures were cached")
        shape = "x".join(str(x) for x in sig["shape"])
        spec = f"op={'MUL_MAT_ID' if sig['expert'] else 'MUL_MAT'},type={sig['dtype'].lower()},ne={shape},n={sig['tokens']},weights=zero"
        if sig["expert"]:
            spec += f",top_k={sig['top_k']},broadcast={int(sig['broadcast'])},numa_expert={int(sig['numa_expert'])},dist=uniform,seed=1234"
        if sig["paired"]:
            spec += ",pair=1"
        command = [str(binary), "perf", "-b", sig["device"], "--numa", "tensors", "--threads", str(cfg["runtime"]["threads"]),
                   "--output", "csv", "--custom", spec]
        log = directory / f"op-{key[:16]}.csv"
        run(command, log, environment, timeout=max(1, remaining))
        measured = operation_times(log.read_text())
        values[key] = {"signature": sig, "ms": statistics.median(measured) / 1000 if measured else None,
                       "command": command, "log": str(log), "reason": None if measured else "backend did not run this signature"}
        write_json(cache_path, {"schema": "delphi.auto-quantize.operations/1", "identity": identity, "values": values})
    return values


def active_experts(count, top_k, tokens):
    if top_k >= count:
        return count
    return count * (-math.expm1(tokens * math.log1p(-top_k / count)))


class Predictor:
    def __init__(self, info, cfg, operations, baseline_choices, measured, slopes=None, baseline_cpu_ms=None):
        self.info, self.cfg, self.operations = info, cfg, operations
        self.baseline_choices, self.measured = baseline_choices, measured
        self.slopes = slopes or {regime: 1.0 for regime in measured}
        self.regions = regions(info)
        self.counts = token_counts(cfg)
        self.cpu_bytes_per_ms = None
        if "cpu_bandwidth_gb_s" in cfg["timing"]:
            self.cpu_bytes_per_ms = cfg["timing"]["cpu_bandwidth_gb_s"] * 1e6
        elif baseline_cpu_ms and baseline_cpu_ms > 0 and "RAM" in cfg["devices"]:
            types = {name: baseline_choices[g["name"]] for g in info["groups"] for name in g["members"]}
            active_bytes = sum(tensor_bytes(t["shape"], types[t["name"]]) * min(info["top_k"], t["shape"][2]) / t["shape"][2]
                               for t in info["tensors"] if t["expert"])
            if active_bytes:
                self.cpu_bytes_per_ms = active_bytes / baseline_cpu_ms
        self.baseline_cost = self.cost(baseline_choices, 0)

    def lookup(self, members, paired, dtype, device, tokens):
        key = digest(signature(members, paired, dtype, device, tokens, self.info["top_k"]))
        value = self.operations.get(key)
        if value is None or value["ms"] is None:
            raise QuantError(f"unmeasured/unsupported signature: {members[0]['name']} {dtype} {device} n={tokens}")
        return value["ms"]

    def cost(self, choices, hit_fraction=0):
        result = {regime: 0.0 for regime in self.counts}
        link = self.cfg["timing"].get("link_bandwidth_gb_s", 12) * 1e6  # bytes/ms; prior is explicitly unmeasured
        streaming = str(self.cfg["runtime"]["env"].get("GGML_SCHED_STAGE", "0")) == "1"
        for group, members, paired in self.regions:
            dtype = choices[group["name"]]
            for regime, tokens in self.counts.items():
                device_time = self.lookup(members, paired, dtype, gpu(self.cfg), tokens)
                local = device_time
                t = members[0]
                if t["expert"] and "RAM" in self.cfg["devices"]:
                    cpu_time = self.lookup(members, paired, dtype, "CPU", tokens)
                    unique = active_experts(t["shape"][2], self.info["top_k"], tokens)
                    weight_bytes = sum(tensor_bytes(m["shape"], dtype) for m in members) * unique / t["shape"][2]
                    # Repeated isolated probes keep active weights in cache. A full model also pays for RAM traffic.
                    if self.cpu_bytes_per_ms:
                        cpu_time = max(cpu_time, weight_bytes / self.cpu_bytes_per_ms)
                    handoff = 0.02 + 4 * (t["shape"][0] + t["shape"][1] * self.info["top_k"]) * tokens / link
                    local = cpu_time + handoff
                    if tokens <= 16 and hit_fraction > 0:
                        local = (1 - hit_fraction) * cpu_time + hit_fraction * device_time + handoff
                    min_tokens = int(self.cfg["runtime"]["env"].get("GGML_SCHED_STAGE_MIN_TOKENS", 512))
                    if streaming and min_tokens > 0 and tokens >= min_tokens:
                        local = min(local, device_time + weight_bytes / link + 0.02)
                n_work = self.cfg["runtime"][f"{regime}_tokens"] if regime != "decode" else 1
                batches, tail = divmod(n_work, tokens)
                # Tail interpolation is flagged in the output; it does not change the fixed ubatch used for calibration.
                result[regime] += local * (batches + tail / tokens)
        return result

    def predict(self, choices, hit_fraction=0):
        cost = self.cost(choices, hit_fraction)
        result = {r: self.measured[r] + self.slopes[r] * (cost[r] - self.baseline_cost[r]) for r in cost}
        if any(not math.isfinite(value) or value <= 0 for value in result.values()):
            raise QuantError("anchored time estimate is nonpositive; recipe exceeds the calibrated model")
        factors = self.cfg["multi_gpu_speedup_percent"]
        if sum(name != "RAM" for name in self.cfg["devices"]) > 1:
            result = {r: value / (1 + factors[r] / 100) for r, value in result.items()}
        return result


def fit_slopes(predictor, witnesses):
    slopes = {}
    for regime in predictor.measured:
        pairs = [(predictor.cost(item["choices"], item.get("hit_fraction", 0))[regime] - predictor.baseline_cost[regime],
                  item["measured"][regime] - predictor.measured[regime]) for item in witnesses]
        denominator = sum(x * x for x, _ in pairs)
        slope = sum(x * y for x, y in pairs) / denominator if denominator > 1e-12 else 1.0
        if not math.isfinite(slope) or slope <= 0:
            raise QuantError(f"{regime} variation contradicts the timing prior; a positive scaling coefficient cannot fit it")
        slopes[regime] = slope
    return slopes
