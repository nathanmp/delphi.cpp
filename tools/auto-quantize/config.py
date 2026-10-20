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


NUMBER = r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"


def weighted_entries(value, name, extra=()):
    """Entries of PATH, PATH:WEIGHT or {"path", "weight", *extra}, given as a list, one entry or a comma-separated string."""
    items = [item for item in value.split(",") if item.strip()] if isinstance(value, str) else value
    if not isinstance(items, list) or any(not isinstance(item, (str, dict)) for item in items):
        raise QuantError(f"{name} must be a list of PATH, PATH:WEIGHT or objects with path and weight")
    result = []
    for item in items:
        if isinstance(item, dict):
            if set(item) - {"path", "weight", *extra}:
                raise QuantError(f"{name} entries accept only {sorted({'path', 'weight', *extra})}")
            path, weight = item.get("path"), item.get("weight")
        else:
            head, colon, tail = item.strip().rpartition(":")
            numeric = bool(colon and head and re.fullmatch(NUMBER, tail.strip()))
            path, weight = (head, float(tail)) if numeric else (item.strip(), None)
        if not isinstance(path, str) or not path:
            raise QuantError(f"{name} entry needs a path")
        if weight is not None:
            number(weight, f"{name} weight of {path}", positive=True)
        result.append({"path": path, "weight": weight, **{key: item[key] for key in extra if isinstance(item, dict) and key in item}})
    return result


def normalized(weights):
    """Relative weights scaled to sum 1, rounded so that equivalent spellings give equal values."""
    total = sum(weights)
    return [float(f"{weight / total:.12g}") for weight in weights]


def corpora(value, absolute, name):
    """[{"path", "weight", optional "bytes_per_token"}] with normalized weights; a plain path counts 1."""
    entries = weighted_entries(value, name, extra=("bytes_per_token",))
    if not entries:
        raise QuantError(f"{name} must name at least one corpus")
    for entry in entries:
        entry["path"] = absolute(entry["path"])
        if "bytes_per_token" in entry:
            number(entry["bytes_per_token"], f"{name} bytes_per_token of {entry['path']}", positive=True)
    if len({entry["path"] for entry in entries}) != len(entries):
        raise QuantError(f"{name} corpora must be distinct files")
    weights = normalized([1.0 if entry["weight"] is None else entry["weight"] for entry in entries])
    return [{**entry, "weight": weight} for entry, weight in zip(entries, weights)]


def imatrix_inputs(value, absolute):
    """(paths, weights) of the imatrix entries. Weights are normalized (plain paths count 1) and None when no entry has one."""
    entries = weighted_entries(value, "imatrix") if value else []
    paths = [absolute(entry["path"]) for entry in entries]
    if all(entry["weight"] is None for entry in entries):
        return paths, None
    if len(set(paths)) != len(paths):
        raise QuantError("weighted imatrix entries must be distinct files")
    return paths, normalized([1.0 if entry["weight"] is None else entry["weight"] for entry in entries])


def imatrix_identities(cfg):
    """Content identities of the imatrix inputs; weighted inputs add their normalized weight."""
    weights = cfg.get("imatrix_weights")
    return [{**file_identity(path, content=True), **({"weight": weights[i]} if weights else {})} for i, path in enumerate(cfg["imatrix"])]


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
               "bindir", "runtime", "sampling", "search", "bounds", "cache_mib", "multi_gpu_speedup_percent", "timing", "quality",
               "calibration", "imatrix_collect"}
    unknown = set(cfg) - allowed
    if unknown:
        raise QuantError(f"unknown configuration keys: {sorted(unknown)}")
    def absolute(value):
        p = Path(value).expanduser()
        return str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    cfg["model"] = absolute(cfg["model"])
    cfg["bindir"] = absolute(cfg.get("bindir", str(ROOT / "build-auto-quantizer/bin")))
    cfg["imatrix"], cfg["imatrix_weights"] = imatrix_inputs(cfg.get("imatrix", []), absolute)
    if "quality" in cfg:
        if not isinstance(cfg["quality"], dict) or set(cfg["quality"]) != {"table"}:
            raise QuantError("quality must contain only a calibrated table path")
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
    # timing imports this module, so the GPU-only condition of timing.gpu_only is repeated here
    if sum(name != "RAM" for name in cfg["devices"]) > 1 and "RAM" not in cfg["devices"] and any(cfg["multi_gpu_speedup_percent"].values()):
        raise QuantError("GPU-only placement profiles the real layer split; multi_gpu_speedup_percent applies only to RAM-routed projections")
    cfg.setdefault("runtime", {})
    runtime = cfg["runtime"]
    defaults = {"threads": 32, "depth_tokens": 0, "batch": 2048, "ubatch": 512, "short_tokens": 128,
                "long_tokens": 4096, "decode_tokens": 256, "repetitions": 3,
                "trace_max_events": 1_048_576, "env": {}, "extra_args": []}
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
    if runtime["trace_max_events"] > 128 * 1024 * 1024:
        raise QuantError("trace_max_events exceeds the native tracer limit")
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
    if set(cfg["search"]) - {"display", "minimize"}:
        raise QuantError("search accepts only display and minimize; the knapsack search has no beam or evaluation cap")
    if cfg["search"].get("minimize", "size") != "size":
        raise QuantError("search.minimize can only be size")
    cfg["search"].setdefault("display", 8)
    if not isinstance(cfg["search"]["display"], int) or cfg["search"]["display"] <= 0:
        raise QuantError("search.display must be a positive integer")
    cfg["calibration"] = calibration(cfg.get("calibration", {}), absolute, cfg)
    if "imatrix_collect" in cfg:
        cfg["imatrix_collect"] = imatrix_collect(cfg["imatrix_collect"], absolute, cfg)
    cfg.setdefault("bounds", {})
    for name, value in cfg["bounds"].items():
        if name not in {"max_damage", "max_loss", "max_total_bytes", "min_decode_t_s", "min_short_t_s", "min_long_t_s"}:
            raise QuantError(f"unknown bound {name}")
        number(value, f"bounds.{name}", positive=name == "max_total_bytes")
    if "max_loss" in cfg["bounds"] and "quality" not in cfg:
        raise QuantError("max_loss requires a calibrated quality table")
    if "max_damage" in cfg["bounds"] and "quality" in cfg:
        raise QuantError("use max_loss for predicted KL; max_damage is for the local error metric")
    cfg.setdefault("timing", {})
    if set(cfg["timing"]) - {"link_bandwidth_gb_s", "cpu_bandwidth_gb_s"}:
        raise QuantError("unknown timing inputs")
    for name in ("link_bandwidth_gb_s", "cpu_bandwidth_gb_s"):
        if name in cfg["timing"]:
            number(cfg["timing"][name], f"timing.{name}", positive=True)
    return cfg


def calibration(c, absolute, cfg):
    """Calibration inputs: text (a path or weighted corpora), teacher, probe design and the devices of the probe sessions."""
    if not isinstance(c, dict):
        raise QuantError("calibration must be an object")
    defaults = {"text": None, "teacher": None, "chunks": 64, "context": 512, "batch": 0, "threads": None,
                "probe_qtypes": None, "bands": 10, "recipes": 8, "seed": 1, "devices": None, "text_seed": 1,
                "bytes_per_token": 4.0, "patch_mib": 8192, "uniform": "auto"}
    if set(c) - set(defaults):
        raise QuantError(f"unknown calibration keys: {sorted(set(c) - set(defaults))}")
    result = {name: c.get(name, default) for name, default in defaults.items()}
    if result["teacher"] is not None:
        result["teacher"] = absolute(result["teacher"])
    number(result["bytes_per_token"], "calibration.bytes_per_token", positive=True)
    if result["text"] is not None:
        entries = corpora(result["text"], absolute, "calibration.text")
        result["text"] = entries[0]["path"] if len(entries) == 1 else entries
    if result["uniform"] not in ("auto", "session", "direct"):
        raise QuantError("calibration.uniform must be auto, session or direct")
    for name in ("chunks", "context", "bands", "recipes", "seed", "text_seed", "batch", "patch_mib"):
        value = result[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name in ("seed", "text_seed", "batch") else 1):
            raise QuantError(f"calibration.{name} must be a valid integer")
    if result["recipes"] < 2:
        raise QuantError("calibration.recipes counts the two uniform recipes and must be at least 2")
    if result["batch"] and result["batch"] % result["context"]:
        raise QuantError("calibration.batch must be a multiple of calibration.context")
    result["threads"] = result["threads"] or cfg["runtime"]["threads"]
    if result["probe_qtypes"] is not None and (not isinstance(result["probe_qtypes"], list) or len(result["probe_qtypes"]) != 2):
        raise QuantError("calibration.probe_qtypes must be [mid, top]")
    if result["devices"] is not None:
        if not isinstance(result["devices"], list) or not result["devices"] or any(
                not isinstance(d, str) or not re.fullmatch(r"(?:CUDA|ROCm)[0-9]+", d) for d in result["devices"]):
            raise QuantError("calibration.devices must list GPU names")
    return result


def imatrix_collect(c, absolute, cfg):
    """Settings of the collect-imatrix command: the corpora, one llama-imatrix run each, and the run options."""
    if not isinstance(c, dict):
        raise QuantError("imatrix_collect must be an object")
    defaults = {"text": None, "context": 512, "chunks": 0, "means": True, "threads": None, "extra_args": []}
    if set(c) - set(defaults):
        raise QuantError(f"unknown imatrix_collect keys: {sorted(set(c) - set(defaults))}")
    result = {name: c.get(name, default) for name, default in defaults.items()}
    if result["text"] is None:
        raise QuantError("imatrix_collect.text must list the corpora, as PATH or PATH:WEIGHT")
    result["text"] = corpora(result["text"], absolute, "imatrix_collect.text")
    result["threads"] = result["threads"] or cfg["runtime"]["threads"]
    for name in ("context", "chunks", "threads"):
        value = result[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == "chunks" else 1):
            raise QuantError(f"imatrix_collect.{name} must be a valid integer")
    if not isinstance(result["means"], bool):
        raise QuantError("imatrix_collect.means must be boolean")
    protected = {"-m", "--model", "-f", "--file", "-o", "--output", "--output-file", "--output-format", "--in-file", "--device",
                 "--numa", "-ngl", "--gpu-layers", "-cmoe", "--cpu-moe", "-ncmoe", "--n-cpu-moe", "-c", "--ctx-size", "-t",
                 "--threads", "--chunks", "--no-ppl", "--ppl"}
    if not isinstance(result["extra_args"], list) or any(str(arg).split("=", 1)[0] in protected for arg in result["extra_args"]):
        raise QuantError("imatrix_collect.extra_args must be a list that does not override managed placement or input options")
    return result
