"""Validated inputs shared by the offline quantization advisor."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "delphi.auto-quantize/1"
AXES = ("precision", "decode", "short", "long")
ROOT = Path(__file__).resolve().parents[2]


class QuantError(ValueError):
    pass


def number(value, name, minimum=0, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise QuantError(f"{name} must be finite")
    if value < minimum or (positive and value == 0):
        raise QuantError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def capacity(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return int(number(value, "capacity"))
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(MB|MiB|GB|GiB|B)?", str(value))
    if not match:
        raise QuantError(f"invalid capacity {value!r}; use MB or MiB")
    units = {None: 1_000_000, "MB": 1_000_000, "MiB": 1 << 20, "GB": 1_000_000_000, "GiB": 1 << 30, "B": 1}
    return int(float(match[1]) * units[match[2]])


def devices(value):
    entries = value.items() if isinstance(value, dict) else (item.split(":", 1) for item in value.split(","))
    result = {}
    try:
        for name, amount in entries:
            if name in result or not re.fullmatch(r"RAM|(?:CUDA|ROCm)[0-9]+", name):
                raise QuantError(f"duplicate or invalid device {name}")
            result[name] = capacity(amount)
    except (ValueError, TypeError) as exc:
        raise QuantError("devices must be CUDA0:16000,RAM:32000 or a byte-valued object") from exc
    if not result or not any(name != "RAM" for name in result):
        raise QuantError("at least one GPU is required")
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_identity(path, content=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        result["sha256"] = h.hexdigest()
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if cfg.get("schema") != SCHEMA:
        raise QuantError(f"configuration schema must be {SCHEMA}")
    allowed = {"schema", "model", "devices", "reserve", "imatrix", "primary_qtypes", "other_qtypes", "importance",
               "preserve_below_mib", "tie_gate_up", "primary_regex", "pins", "preserve_regex", "allow_requantize",
               "bindir", "runtime", "sampling", "search", "bounds", "cache_mib", "multi_gpu_speedup_percent", "timing", "quality"}
    unknown = set(cfg) - allowed
    if unknown:
        raise QuantError(f"unknown configuration keys: {sorted(unknown)}")
    def absolute(value):
        p = Path(value).expanduser()
        return str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    cfg["model"] = absolute(cfg["model"])
    cfg["bindir"] = absolute(cfg.get("bindir", str(ROOT / "build-auto-quantizer/bin")))
    cfg["imatrix"] = [absolute(p) for p in cfg.get("imatrix", [])]
    if "quality" in cfg:
        if not isinstance(cfg["quality"], dict) or set(cfg["quality"]) != {"table"}:
            raise QuantError("quality must contain only a repair calibration table path")
        cfg["quality"]["table"] = absolute(cfg["quality"]["table"])
    cfg["devices"] = devices(cfg["devices"])
    cfg["reserve"] = {name: capacity(value) for name, value in cfg.get("reserve", {}).items()}
    if set(cfg["reserve"]) - set(cfg["devices"]):
        raise QuantError("reserve names must occur in devices")
    for name, amount in cfg["reserve"].items():
        if amount > cfg["devices"][name]:
            raise QuantError(f"reserve exceeds {name} budget")
    priorities = cfg.get("importance", {"precision": 1, "decode": 1, "short": 1, "long": 1})
    if set(priorities) - set(AXES):
        raise QuantError("importance keys are precision, decode, short, long")
    priorities = {axis: number(priorities.get(axis, 0), f"importance.{axis}") for axis in AXES}
    total = sum(priorities.values())
    if total <= 0:
        raise QuantError("at least one importance must be positive")
    cfg["importance"] = {axis: value / total for axis, value in priorities.items()}
    cfg["preserve_below_mib"] = number(cfg.get("preserve_below_mib", 1), "preserve_below_mib")
    cfg.setdefault("tie_gate_up", True)
    for field in ("tie_gate_up", "allow_requantize"):
        if field in cfg and not isinstance(cfg[field], bool):
            raise QuantError(f"{field} must be boolean")
    cfg.setdefault("pins", {})
    cfg.setdefault("preserve_regex", [])
    for regex in cfg["preserve_regex"] + ([cfg["primary_regex"]] if cfg.get("primary_regex") else []):
        re.compile(regex)
    cfg["cache_mib"] = number(cfg.get("cache_mib", 0), "cache_mib")
    speedups = cfg.get("multi_gpu_speedup_percent", {})
    if set(speedups) - {"decode", "short", "long"}:
        raise QuantError("multi_gpu_speedup_percent keys are decode, short, long")
    cfg["multi_gpu_speedup_percent"] = {axis: number(speedups.get(axis, 0), f"speedup.{axis}") for axis in AXES[1:]}
    cfg.setdefault("runtime", {})
    runtime = cfg["runtime"]
    defaults = {"threads": 32, "depth_tokens": 0, "batch": 2048, "ubatch": 512, "short_tokens": 128,
                "long_tokens": 4096, "decode_tokens": 256, "repetitions": 3,
                "env": {}, "extra_args": []}
    if set(runtime) - set(defaults):
        raise QuantError(f"unknown runtime keys: {sorted(set(runtime) - set(defaults))}")
    for name, default in defaults.items():
        runtime.setdefault(name, default)
        if isinstance(default, int):
            number(runtime[name], name, positive=name != "depth_tokens")
            if not isinstance(runtime[name], int):
                raise QuantError(f"{name} must be an integer")
    if runtime["ubatch"] > runtime["batch"]:
        raise QuantError("ubatch must not exceed batch")
    protected = {"-m", "--model", "--device", "--numa", "-ngl", "--gpu-layers", "-cmoe", "--cpu-moe", "--port",
                 "-c", "--ctx-size", "-b", "--batch-size", "-ub", "--ubatch-size", "-t", "--threads", "--parallel",
                 "-p", "--n-prompt", "-n", "--n-gen", "-pg", "-r", "--repetitions", "-d", "--n-depth",
                 "-ncmoe", "--n-cpu-moe", "-ts", "--tensor-split", "-sm", "--split-mode"}
    if any(str(arg).split("=", 1)[0] in protected for arg in runtime["extra_args"]):
        raise QuantError("extra_args cannot override managed placement or workload options")
    if "LLAMA_MOE_CACHE_MIB" in runtime["env"]:
        raise QuantError("use cache_mib rather than runtime.env.LLAMA_MOE_CACHE_MIB")
    runtime["env"] = {str(name): str(value) for name, value in runtime["env"].items()}
    cfg.setdefault("sampling", {})
    for name, default in {"samples": 64, "max_samples": 512, "seed": 1234, "row_stats": True}.items():
        cfg["sampling"].setdefault(name, default)
    if set(cfg["sampling"]) - {"samples", "max_samples", "seed", "row_stats"}:
        raise QuantError("unknown sampling options")
    for name in ("samples", "max_samples", "seed"):
        value = cfg["sampling"][name]
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == "seed" else 1):
            raise QuantError(f"sampling.{name} must be a valid integer")
    if cfg["sampling"]["max_samples"] < cfg["sampling"]["samples"] or not isinstance(cfg["sampling"]["row_stats"], bool):
        raise QuantError("invalid sampling range or row_stats flag")
    cfg.setdefault("search", {})
    for name, default in {"beam": 16, "evaluations": 4000, "display": 8}.items():
        cfg["search"].setdefault(name, default)
        value = cfg["search"][name]
        if not isinstance(value, int) or value <= 0:
            raise QuantError(f"search.{name} must be a positive integer")
    cfg.setdefault("bounds", {})
    for name, value in cfg["bounds"].items():
        if name not in {"max_damage", "max_loss", "min_decode_t_s", "min_short_t_s", "min_long_t_s"}:
            raise QuantError(f"unknown bound {name}")
        number(value, f"bounds.{name}")
    if "max_loss" in cfg["bounds"] and "quality" not in cfg:
        raise QuantError("max_loss requires a repair calibration table")
    if "max_damage" in cfg["bounds"] and "quality" in cfg:
        raise QuantError("use max_loss for repair-score units; max_damage is for the local error metric")
    cfg.setdefault("timing", {})
    if set(cfg["timing"]) - {"link_bandwidth_gb_s", "cpu_bandwidth_gb_s"}:
        raise QuantError("unknown timing inputs")
    for name in ("link_bandwidth_gb_s", "cpu_bandwidth_gb_s"):
        if name in cfg["timing"]:
            number(cfg["timing"][name], f"timing.{name}", positive=True)
    return cfg
