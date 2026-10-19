"""Configuration of an auto-tuner run (JSON, schema delphi.auto-tune/1). Paths are relative to the configuration file."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "delphi.auto-tune/1"

DEFAULTS: dict[str, Any] = {
    "schema": SCHEMA,
    "recipe": None,  # /path/recipes.py:factory (tools/bottlenecks recipe)
    "arm": None,  # the recipe arm that is the start point and the reference
    "bindir": None,
    "env": {},  # added to every launch (for example LD_LIBRARY_PATH)
    "output": None,  # directory: results, journal, artifacts, report
    "weights": {"prefill": 1, "decode": 1},
    "prefill": {"text": None, "lengths": {"300": 1}, "reps": 3},
    "decode": {"osl": 256},
    "context": 8192,  # -c of every launch (a constraint, not a knob)
    "kld": {"limit": None, "text": None, "chunks": 8, "ctx": 512, "batch": 4},
    "devices": {},  # {"ROCm2": {"min_free_mib": 2048}, "ROCm0": {"max_used_mib": 16384}}
    "clock_mode": "auto",  # amdgpu power_dpm_force_performance_level that every used R9700 must have; null: no check
    "budget": {"max_launches": 60, "max_hours": 6.0},
    "knobs": {"only": [], "include": [], "exclude": [], "levels": {}},
    "noise_margin": 0.02,  # a change must beat the incumbent by this fraction of score to be kept
    "warmup_launch": True,  # one unscored launch first (page cache, compile caches)
    "trace": False,  # record the execution trace of every launch (slower; needed for pass-cost analysis)
    "max_threads": None,  # default: physical cores - 8
    "cpus": None,  # exact Linux CPU list, for example 0-23,32-55
    "max_cores": None,  # at most this many physical cores from the caller's affinity
    "placement": {"reserve_mib": {}},  # extra device buffers not measured by llama-fit-params (for example mmproj)
    "class_order": ["fit", "precision", "accept", "speed"],  # the order in which the knob classes are searched
}


class TuneError(RuntimeError):
    pass


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in ("env", "devices", "levels", "lengths"):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    raw = json.loads(path.read_text())
    if raw.get("schema", SCHEMA) != SCHEMA:
        raise TuneError(f"{path}: schema {raw.get('schema')} is not {SCHEMA}")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise TuneError(f"{path}: unknown key(s): {', '.join(unknown)}")
    cfg = _merge(DEFAULTS, raw)
    base = path.parent

    def resolve(value: str | None) -> str | None:
        if value is None:
            return None
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else (base / p).resolve())

    if not cfg["recipe"] or ":" not in cfg["recipe"]:
        raise TuneError("recipe must be /path/recipes.py:factory")
    recipe_path, factory = cfg["recipe"].rsplit(":", 1)
    cfg["recipe"] = f"{resolve(recipe_path)}:{factory}"
    for key in ("bindir", "output"):
        cfg[key] = resolve(cfg[key])
    cfg["prefill"]["text"] = resolve(cfg["prefill"]["text"])
    cfg["kld"]["text"] = resolve(cfg["kld"]["text"])
    if not cfg["arm"]:
        raise TuneError("arm is required")
    if not cfg["output"]:
        raise TuneError("output is required")
    if cfg["weights"].get("prefill", 0) > 0 and not cfg["prefill"]["text"]:
        raise TuneError("prefill.text is required when prefill has a weight")
    if set(cfg["weights"]) - {"prefill", "decode"}:
        raise TuneError("weights takes prefill and decode")
    for name, cap in cfg["devices"].items():
        if set(cap) - {"min_free_mib", "max_used_mib", "fill"}:
            raise TuneError(f"devices.{name}: takes fill, min_free_mib and max_used_mib")
        if cap.get("fill", "fill") not in ("fill", "min"):
            raise TuneError(f"devices.{name}.fill must be fill or min")
        for key in ("min_free_mib", "max_used_mib"):
            if key in cap and (not isinstance(cap[key], (int, float)) or cap[key] < 0):
                raise TuneError(f"devices.{name}.{key} must be non-negative")
    if cfg["context"] < 1:
        raise TuneError("context must be positive")
    if cfg["max_threads"] is not None and cfg["max_threads"] < 1:
        raise TuneError("max_threads must be positive")
    for name, value in cfg["placement"]["reserve_mib"].items():
        if not isinstance(value, (int, float)) or value < 0:
            raise TuneError(f"placement.reserve_mib.{name} must be non-negative")
    cfg["prefill"]["lengths"] = {str(int(k)): float(v) for k, v in cfg["prefill"]["lengths"].items()}
    cfg["config_path"] = str(path)
    return cfg


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_identity(path: Path) -> dict[str, Any] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
