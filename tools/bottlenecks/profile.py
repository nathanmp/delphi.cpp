#!/usr/bin/env python3
"""Run a recipe-defined DELPHI benchmark and summarize its execution trace.

Recipes deliberately describe commands as argument vectors. The runner owns
environment recording, trace collection, result parsing, and the normalized
JSON schema. This keeps private deployment policy separate from the public
measurement mechanism.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRACE_DUMP_PATH = ROOT / "tools" / "exec-trace" / "exec-trace-dump.py"
ENV_PREFIXES = ("GGML_", "LLAMA_", "CUDA_", "HIP_", "HSA_", "ROCR_", "OMP_")


@dataclass(frozen=True)
class FamilyRule:
    """Optional user-defined tag for an eager GPU operation span."""

    name: str
    operations: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    node_contains: tuple[str, ...] = ()
    source_contains: tuple[str, ...] = ()

    def matches(self, node: Mapping[str, Any]) -> bool:
        if self.operations and node.get("op") not in self.operations:
            return False
        node_text = node.get("name", "").lower()
        source_text = " ".join(x.get("name", "") for x in node.get("sources", {}).values()).lower()
        all_text = f"{node_text} {source_text}"
        return (all(value.lower() in all_text for value in self.contains)
                and all(value.lower() in node_text for value in self.node_contains)
                and all(value.lower() in source_text for value in self.source_contains))


@dataclass(frozen=True)
class BenchArm:
    """One llama-bench process. Use -r for repeated samples without reloads."""

    name: str
    command: tuple[str, ...]
    model_paths: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    trace_mode: str = "submit"  # off, submit, or op
    trace_max_events: int = 1_048_576
    trace_event_capacity: int = 16_384
    family_rules: tuple[FamilyRule, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServerArm:
    """One loaded llama-server and one measured SPEED-Bench client command."""

    name: str
    server_command: tuple[str, ...]
    client_command: tuple[str, ...]
    model_paths: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    warmup_command: tuple[str, ...] = ()
    port: int = 18081
    ready_timeout_s: float = 180.0
    request_timeout_s: float = 1800.0
    trace_mode: str = "submit"
    trace_max_events: int = 1_048_576
    trace_event_capacity: int = 16_384
    family_rules: tuple[FamilyRule, ...] = ()
    notes: tuple[str, ...] = ()
    extra_measurements: tuple["ServerMeasurement", ...] = ()
    trace_prefill_min_tokens: int = 0
    trace_prefill_target_tokens: int = 0


@dataclass(frozen=True)
class ServerMeasurement:
    """An additional client measurement made while a ServerArm is loaded.

    The command can use the same placeholders as ServerArm.client_command.
    Its {output} placeholder is an arm-local JSON file named after this
    measurement. result_kind selects the small normalized metric parser.
    """

    name: str
    command: tuple[str, ...]
    result_kind: str = "speed-bench"


Arm = BenchArm | ServerArm


@dataclass(frozen=True)
class Recipe:
    """A private or local deployment description consumed by this public runner."""

    name: str
    description: str
    arms: tuple[Arm, ...]
    notes: tuple[str, ...] = ()


class ProfileError(RuntimeError):
    pass


def _load_trace_dump():
    spec = importlib.util.spec_from_file_location("delphi_exec_trace_dump", TRACE_DUMP_PATH)
    if not spec or not spec.loader:
        raise ProfileError(f"cannot load execution-trace decoder: {TRACE_DUMP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRACE = _load_trace_dump()


def _stable_node_name(name: str) -> str:
    name = re.sub(r"\bblk\.\d+\.", "blk.*.", name)
    name = re.sub(r"\b(?:layer|layers)\.\d+\.", lambda match: match.group(0).split(".")[0] + ".*.", name)
    name = re.sub(r"([_-])\d+(?=$|[\s(])", r"\1*", name)
    return name


def _node_role(node: Mapping[str, Any]) -> str:
    """Group repeated layer nodes by their trace names and source tensors."""
    op = node.get("op") or "unknown op"
    name = node.get("name") or ""
    if re.fullmatch(r"node_\d+", name) or not name.strip() or name.startswith(" (view)"):
        sources = [re.sub(r"#\d+$", "", re.sub(r"^[^#]+#", "", source.get("name", ""))) for source in node.get("sources", {}).values()]
        named = next((source for source in sources if re.search(r"\bblk\.\d+\..*(?:weight|bias)", source)), None)
        if named is None:
            named = next((source for source in sources if source.strip() and not source.lstrip().startswith("(") and not re.fullmatch(r"node_\d+", source)), None)
        if named is not None:
            return f"{op} / src {_stable_node_name(named)}"
        dims = node.get("dims") or ()
        shape = "x".join(str(size) for size in dims if size != 1) or "scalar"
        source_kinds = "+".join("view" if "view" in source else "generic" for source in sources) or "none"
        return f"{op} / unnamed {node.get('type', '?')} {shape} src:{source_kinds}"
    return f"{op} / {_stable_node_name(name)}"


def _stats(values: Iterable[float]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"count": 0, "median": None, "mean": None, "min": None, "max": None, "p10": None, "p90": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": ordered[int((len(ordered) - 1) * 0.10)],
        "p90": ordered[int((len(ordered) - 1) * 0.90)],
    }


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(command)


def _parse_assignment(text: str, option: str) -> tuple[str, str]:
    if "=" not in text:
        raise ProfileError(f"{option} must be NAME=VALUE: {text}")
    name, value = text.split("=", 1)
    if not name:
        raise ProfileError(f"{option} has an empty name")
    return name, value


def _filtered_environment(extra: Mapping[str, str]) -> dict[str, str]:
    result = {key: value for key, value in os.environ.items() if key.startswith(ENV_PREFIXES)}
    result.update({key: str(value) for key, value in extra.items()})
    return dict(sorted(result.items()))


def _run(command: Sequence[str], env: Mapping[str, str], *, timeout: float | None = None, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    complete_env = os.environ.copy()
    complete_env.update(env)
    return subprocess.run(command, cwd=cwd, env=complete_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def _version(binary: str, env: Mapping[str, str]) -> str | None:
    version_env = dict(env)
    for key in ("GGML_EXEC_TRACE", "GGML_EXEC_TRACE_MAX_EVENTS", "GGML_EXEC_TRACE_GPU_TIMING", "GGML_EXEC_TRACE_GPU_TIMING_EVENTS"):
        version_env.pop(key, None)
    try:
        completed = _run((binary, "--version"), version_env, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode:
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _model_identity(paths: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    for raw_path in paths:
        path = Path(raw_path)
        item: dict[str, Any] = {"path": str(path)}
        try:
            stat = path.stat()
        except OSError as error:
            item["error"] = str(error)
        else:
            item.update({"bytes": stat.st_size, "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime))})
        result.append(item)
    return result


def _repository_identity() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        completed = _run(("git", *args), {})
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "tree": "clean" if status == "" else "dirty"}


def _trace_env(path: Path, mode: str, max_events: int, event_capacity: int) -> dict[str, str]:
    if mode not in ("off", "submit", "op"):
        raise ProfileError(f"unknown trace mode: {mode}")
    if mode == "off":
        return {}
    env = {"GGML_EXEC_TRACE": str(path), "GGML_EXEC_TRACE_MAX_EVENTS": str(max_events)}
    if mode in ("submit", "op"):
        env["GGML_EXEC_TRACE_GPU_TIMING"] = mode
        env["GGML_EXEC_TRACE_GPU_TIMING_EVENTS"] = str(event_capacity)
    return env


def _ensure_llama_bench_json(command: Sequence[str]) -> tuple[str, ...]:
    result = list(command)
    if "-o" not in result and "--output" not in result:
        result.extend(("-o", "json"))
    if "-oe" not in result and "--output-err" not in result:
        result.extend(("-oe", "none"))
    return tuple(result)


def _parse_json_output(text: str, description: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ProfileError(f"{description} did not write JSON: {error}") from error


def _bench_metrics(payload: Any) -> dict[str, Any]:
    tests = payload.get("tests", []) if isinstance(payload, dict) else []
    results = []
    for test in tests:
        if not isinstance(test, dict):
            continue
        samples = test.get("samples_ts", [])
        if not samples:
            continue
        results.append({"test": {key: value for key, value in test.items() if key not in ("samples_ns", "samples_ts")}, "throughput_t_s": _stats(samples)})
    return {"kind": "llama-bench", "tests": results}


def _speed_bench_metrics(payload: Any) -> dict[str, Any]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
    if not overall and isinstance(summary, list):
        overall = next((item for item in summary if item.get("category") == "overall"), {})
    return {
        "kind": "speed-bench",
        "completed_samples": payload.get("completed_samples"),
        "failed_samples": payload.get("failed_samples"),
        "prompt_t_s": overall.get("avg_prompt_t_s"),
        "predicted_t_s": overall.get("avg_pred_t_s"),
        "latency_s": overall.get("avg_latency"),
        "accept_rate": overall.get("accept_rate"),
        "overall": overall,
    }


def _prefill_metrics(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProfileError("prefill client result is not a JSON object")
    return {
        "kind": "prefill",
        "target_tokens": payload.get("target_tokens"),
        "input_tokens": payload.get("input_tokens"),
        "prompt_tokens": payload.get("prompt_tokens"),
        "prompt_t_s": payload.get("prompt_t_s"),
        "prompt_ms": payload.get("prompt_ms"),
        "latency_s": payload.get("latency_s"),
        "request_start_raw_ns": payload.get("request_start_raw_ns"),
        "request_end_raw_ns": payload.get("request_end_raw_ns"),
        "predicted_tokens": payload.get("predicted_tokens"),
    }


def _server_metrics(kind: str, payload: Any) -> dict[str, Any]:
    if kind == "speed-bench":
        return _speed_bench_metrics(payload)
    if kind == "prefill":
        return _prefill_metrics(payload)
    raise ProfileError(f"unknown server measurement kind: {kind}")


def _selected_execution_ids(events: Sequence[Mapping[str, Any]]) -> tuple[set[int], dict[int, dict[str, Any]]]:
    begins: dict[int, Mapping[str, Any]] = {}
    ends: dict[int, Mapping[str, Any]] = {}
    for event in events:
        if event["kind"] == TRACE.K_EXECUTION_BEGIN:
            begins[event["execution_id"]] = event
        elif event["kind"] == TRACE.K_EXECUTION_END:
            ends[event["execution_id"]] = event

    by_kind: dict[str, list[int]] = defaultdict(list)
    info: dict[int, dict[str, Any]] = {}
    for execution_id, begin in begins.items():
        kind = TRACE.EXECUTION_NAMES.get(begin["flags"], str(begin["flags"]))
        end = ends.get(execution_id)
        if end is None:
            continue
        by_kind[kind].append(execution_id)
        info[execution_id] = {
            "kind": kind,
            "tokens": begin["args"][0],
            "start_ns": begin["timestamp_ns"],
            "wall_ms": (end["timestamp_ns"] - begin["timestamp_ns"]) / 1e6,
        }
    selected = set()
    for ids in by_kind.values():
        selected.update(sorted(ids)[1:])  # first execution of each kind is warmup
    return selected, info


def _prefill_execution_ids(selected: set[int], execution_info: Mapping[int, Mapping[str, Any]], min_tokens: int, target_tokens: int, window_ns: tuple[int, int] | None = None) -> set[int]:
    pool = set(execution_info) if target_tokens else selected
    candidates = sorted(
        execution_id for execution_id in pool
        if execution_info[execution_id]["kind"] == "prefill"
    )
    if not target_tokens:
        return {execution_id for execution_id in candidates if execution_info[execution_id]["tokens"] >= min_tokens}
    if window_ns:
        return {execution_id for execution_id in candidates if window_ns[0] <= execution_info[execution_id]["start_ns"] <= window_ns[1]}

    # The exact-ID request is the final client action in an arm. Its logical
    # batches may include a small tail, so do not discard that tail merely
    # because it is below min_tokens. MTP/DSpark can evaluate the same prompt
    # in more than one context; retain the complete final run and record its
    # trace-token total rather than silently choosing one context.
    large = [execution_id for execution_id in candidates if execution_info[execution_id]["tokens"] >= max(min_tokens, target_tokens // 2)]
    if not large:
        return set()
    start = large[-1]
    while start - 1 in large:
        start -= 1
    return {execution_id for execution_id in candidates if execution_id >= start}


def _trace_summary(path: Path, family_rules: Sequence[FamilyRule], *, prefill_min_tokens: int = 0, prefill_target_tokens: int = 0, prefill_window_ns: tuple[int, int] | None = None) -> dict[str, Any]:
    trace = TRACE.read_trace(path)
    if trace["flags"] & 1 or trace["trace_stats"].get("dropped_event_count", 0):
        raise ProfileError(f"trace is truncated or dropped events: {path}")
    events, strings = trace["events"], trace["strings"]
    selected, execution_info = _selected_execution_ids(events)
    decode_ids = {execution_id for execution_id in selected if execution_info[execution_id]["kind"] == "decode"}
    prefill_ids = _prefill_execution_ids(selected, execution_info, prefill_min_tokens, prefill_target_tokens, prefill_window_ns)
    measured_ids = decode_ids | prefill_ids
    per_execution = defaultdict(lambda: Counter())
    host_detail: dict[int, Counter[str]] = defaultdict(Counter)
    copy_bytes: dict[int, Counter[str]] = defaultdict(Counter)
    backend_name = TRACE.build_backend_names(events, strings)
    span_names = {
        TRACE.K_BACKEND_SUBMIT: "host_submit_ms",
        TRACE.K_CPU_COMPUTE: "cpu_compute_ms",
        TRACE.K_COPY_HANDLING: "copy_handling_ms",
    }
    for event in events:
        label = span_names.get(event["kind"])
        if label and event["execution_id"] in measured_ids:
            per_execution[event["execution_id"]][label] += TRACE.span_duration_ns(event) / 1e6
        execution_id = event["execution_id"]
        if execution_id not in measured_ids:
            continue
        kind = event["kind"]
        if kind == TRACE.K_BACKEND_SUBMIT:
            detail = f"GPU submission on {backend_name(event)}"
        elif kind == TRACE.K_CPU_COMPUTE:
            detail = f"CPU split on {backend_name(event)}"
        elif kind == TRACE.K_BACKEND_WAIT:
            reason = TRACE.WAIT_REASONS.get(event["flags"], str(event["flags"]))
            detail = f"wait {reason} on {backend_name(event)}"
        elif kind in (TRACE.K_COPY_HANDLING, TRACE.K_COPY_BODY):
            source_id = event["args"][3] & 0xffffffff
            source = backend_name(event, source_id)
            destination = backend_name(event)
            path = TRACE.COPY_PATHS.get(event["flags"], str(event["flags"]))
            phase = "handling" if kind == TRACE.K_COPY_HANDLING else "body"
            detail = f"copy {phase} {path} {source} -> {destination}"
            if kind == TRACE.K_COPY_HANDLING:
                copy_bytes[execution_id][detail] += event["args"][2]
        elif kind in (TRACE.K_COPY_WAIT_SRC, TRACE.K_COPY_WAIT_DST):
            other_id = event["args"][2] & 0xffffffff
            other = backend_name(event, other_id)
            current = backend_name(event)
            source, destination = (current, other) if kind == TRACE.K_COPY_WAIT_SRC else (other, current)
            path = TRACE.COPY_PATHS.get(event["flags"], str(event["flags"]))
            phase = "source wait" if kind == TRACE.K_COPY_WAIT_SRC else "destination wait"
            detail = f"copy {phase} {path} {source} -> {destination}"
        else:
            continue
        host_detail[execution_id][detail] += TRACE.span_duration_ns(event) / 1e6

    spans, _ = TRACE.gpu_timing_model(events)
    gpu_status_counts = Counter(span["status"] for span in spans)
    submit_per_execution = Counter()
    submit_by_backend: dict[int, Counter[str]] = defaultdict(Counter)
    family_per_execution: dict[int, Counter[str]] = defaultdict(Counter)
    op_per_execution: dict[int, Counter[str]] = defaultdict(Counter)
    role_per_execution: dict[int, Counter[str]] = defaultdict(Counter)
    backend_op_per_execution: dict[int, Counter[str]] = defaultdict(Counter)
    node_metadata = TRACE.build_node_metadata(events, strings)
    for span in spans:
        event = span["event"]
        execution_id = event["execution_id"]
        if execution_id not in measured_ids or span["status"] != "resolved":
            continue
        duration_ms = span["device_duration_ns"] / 1e6
        if span["scope"] in ("eager_submit", "graph_launch"):
            submit_per_execution[execution_id] += duration_ms
            submit_by_backend[execution_id][backend_name(event)] += duration_ms
        if span["scope"] == "eager_op":
            node = node_metadata(event) or {}
            op = node.get("op") or "unknown op"
            op_per_execution[execution_id][op] += duration_ms
            role_per_execution[execution_id][_node_role(node)] += duration_ms
            backend_op_per_execution[execution_id][f"{backend_name(event)} / {op}"] += duration_ms
            family = next((rule.name for rule in family_rules if rule.matches(node)), None)
            if family:
                family_per_execution[execution_id][family] += duration_ms

    def summarize(ids: set[int], client_tokens: int = 0) -> dict[str, Any]:
        summary = {
            "executions": len(ids),
            "tokens": {
                "total": sum(execution_info[item]["tokens"] for item in ids),
                "per_execution": _stats(execution_info[item]["tokens"] for item in ids),
            },
            "wall_ms": _stats(execution_info[item]["wall_ms"] for item in ids),
            "wall_ms_per_token": _stats(execution_info[item]["wall_ms"] / max(1, execution_info[item]["tokens"]) for item in ids),
        }
        for label in span_names.values():
            summary[label] = _stats(per_execution[item][label] / max(1, execution_info[item]["tokens"]) for item in ids)
        summary["device_submit_ms"] = _stats(submit_per_execution[item] / max(1, execution_info[item]["tokens"]) for item in ids)
        detail_names = sorted({name for item in ids for name in host_detail[item]})
        summary["host_detail_ms"] = {
            name: _stats(host_detail[item][name] / max(1, execution_info[item]["tokens"]) for item in ids)
            for name in detail_names
        }
        summary["copy_bytes_per_token"] = {
            name: sum(copy_bytes[item][name] for item in ids) / max(1, sum(execution_info[item]["tokens"] for item in ids))
            for name in detail_names if any(copy_bytes[item][name] for item in ids)
        }
        backend_names = sorted({name for item in ids for name in submit_by_backend[item]})
        summary["device_submit_by_backend_ms"] = {
            name: _stats(submit_by_backend[item][name] / max(1, execution_info[item]["tokens"]) for item in ids)
            for name in backend_names
        }
        for key, per_execution_data in (
            ("diagnostic_by_op_ms", op_per_execution),
            ("diagnostic_by_node_role_ms", role_per_execution),
            ("diagnostic_by_backend_op_ms", backend_op_per_execution),
        ):
            names = sorted({name for item in ids for name in per_execution_data[item]})
            if names:
                summary[key] = {
                    name: _stats(per_execution_data[item][name] / max(1, execution_info[item]["tokens"]) for item in ids)
                    for name in names
                }
        family_names = sorted({name for item in ids for name in family_per_execution[item]})
        if family_names:
            summary["diagnostic_op_family_ms"] = {
                name: _stats(family_per_execution[item][name] / max(1, execution_info[item]["tokens"]) for item in ids)
                for name in family_names
            }
        if client_tokens:
            aggregate = {
                "wall_ms": sum(execution_info[item]["wall_ms"] for item in ids) / client_tokens,
                **{label: sum(per_execution[item][label] for item in ids) / client_tokens for label in span_names.values()},
                "device_submit_ms": sum(submit_per_execution[item] for item in ids) / client_tokens,
            }
            if family_names:
                aggregate["diagnostic_op_family_ms"] = {
                    name: sum(family_per_execution[item][name] for item in ids) / client_tokens
                    for name in family_names
                }
            for key, per_execution_data in (
                ("diagnostic_by_op_ms", op_per_execution),
                ("diagnostic_by_node_role_ms", role_per_execution),
                ("diagnostic_by_backend_op_ms", backend_op_per_execution),
            ):
                if key in summary:
                    aggregate[key] = {
                        name: sum(per_execution_data[item][name] for item in ids) / client_tokens
                        for name in summary[key]
                    }
            aggregate["host_detail_ms"] = {
                name: sum(host_detail[item][name] for item in ids) / client_tokens
                for name in detail_names
            }
            aggregate["copy_bytes_per_token"] = {
                name: sum(copy_bytes[item][name] for item in ids) / client_tokens
                for name in summary["copy_bytes_per_token"]
            }
            aggregate["device_submit_by_backend_ms"] = {
                name: sum(submit_by_backend[item][name] for item in ids) / client_tokens
                for name in backend_names
            }
            summary["aggregate_per_client_token_ms"] = aggregate
        return summary

    decode = summarize(decode_ids)
    prefill = summarize(prefill_ids, prefill_target_tokens)

    graph_states = Counter()
    for event in events:
        if event["kind"] == TRACE.K_GPU_GRAPH:
            graph_states[TRACE.GPU_GRAPH_STATES.get(event["flags"], str(event["flags"]))] += 1
    return {
        "schema_version": trace["version"],
        "event_count": len(events),
        "trace_stats": trace["trace_stats"],
        "gpu_timing_status_counts": dict(sorted(gpu_status_counts.items())),
        "warmups_excluded": sorted(set(execution_info) - selected),
        "decode": decode,
        "prefill": prefill,
        "prefill_min_tokens": prefill_min_tokens,
        "prefill_target_tokens": prefill_target_tokens,
        "prefill_excluded_below_min_tokens": sorted(
            execution_id for execution_id in selected
            if execution_info[execution_id]["kind"] == "prefill"
            and execution_info[execution_id]["tokens"] < prefill_min_tokens
        ),
        "gpu_graph_lifecycle": dict(sorted(graph_states.items())),
    }


def _arm_trace_path(artifacts: Path, arm: Arm) -> Path:
    return artifacts / arm.name / "execution.trace"


def _run_bench_arm(arm: BenchArm, artifacts: Path, common_env: Mapping[str, str]) -> dict[str, Any]:
    arm_dir = artifacts / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    trace_path = _arm_trace_path(artifacts, arm)
    env = dict(common_env)
    env.update(arm.env)
    env.update(_trace_env(trace_path, arm.trace_mode, arm.trace_max_events, arm.trace_event_capacity))
    command = _ensure_llama_bench_json(arm.command)
    completed = _run(command, env)
    (arm_dir / "stdout.json").write_text(completed.stdout)
    (arm_dir / "stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise ProfileError(f"{arm.name} failed with status {completed.returncode}; see {arm_dir / 'stderr.log'}")
    result = {
        "name": arm.name,
        "type": "llama-bench",
        "command": list(command),
        "command_text": _command_text(command),
        "environment": _filtered_environment(env),
        "binary_version": _version(command[0], env),
        "models": _model_identity(arm.model_paths),
        "metrics": _bench_metrics(_parse_json_output(completed.stdout, arm.name)),
        "notes": list(arm.notes),
    }
    if arm.trace_mode != "off":
        if not trace_path.exists():
            raise ProfileError(f"{arm.name} did not produce {trace_path}")
        result["trace"] = {
            "path": str(trace_path),
            "mode": arm.trace_mode,
            "integrity": "not-run",
            "analysis": _trace_summary(trace_path, arm.family_rules, prefill_min_tokens=arm.trace_prefill_min_tokens, prefill_target_tokens=arm.trace_prefill_target_tokens),
        }
    return result


def _render(command: Sequence[str], values: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(item.format(**values) for item in command)


def _wait_for_server(url: str, process: subprocess.Popen[str], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ProfileError(f"llama-server exited with status {process.returncode} before it became ready")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    raise ProfileError(f"llama-server did not become ready within {timeout_s:.0f}s: {url}")


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_server_arm(arm: ServerArm, artifacts: Path, common_env: Mapping[str, str]) -> dict[str, Any]:
    arm_dir = artifacts / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    trace_path = _arm_trace_path(artifacts, arm)
    output_path = arm_dir / "client.json"
    url = f"http://127.0.0.1:{arm.port}"
    values = {"port": str(arm.port), "url": url, "output": str(output_path), "artifact_dir": str(arm_dir)}
    env = dict(common_env)
    env.update(arm.env)
    env.update(_trace_env(trace_path, arm.trace_mode, arm.trace_max_events, arm.trace_event_capacity))
    server_command = _render(arm.server_command, values)
    complete_env = os.environ.copy()
    complete_env.update(env)
    with (arm_dir / "server.stdout.log").open("w") as stdout, (arm_dir / "server.stderr.log").open("w") as stderr:
        process = subprocess.Popen(server_command, cwd=ROOT, env=complete_env, text=True, stdout=stdout, stderr=stderr)
        try:
            _wait_for_server(url, process, arm.ready_timeout_s)
            if arm.warmup_command:
                warmup = _run(_render(arm.warmup_command, values), env, timeout=arm.request_timeout_s)
                (arm_dir / "warmup.stdout.log").write_text(warmup.stdout)
                (arm_dir / "warmup.stderr.log").write_text(warmup.stderr)
                if warmup.returncode:
                    raise ProfileError(f"{arm.name} warmup failed with status {warmup.returncode}")
            client_command = _render(arm.client_command, values)
            client = _run(client_command, env, timeout=arm.request_timeout_s)
            (arm_dir / "client.stdout.log").write_text(client.stdout)
            (arm_dir / "client.stderr.log").write_text(client.stderr)
            if client.returncode:
                raise ProfileError(f"{arm.name} client failed with status {client.returncode}; see {arm_dir / 'client.stderr.log'}")
            extra_measurements = []
            for measurement in arm.extra_measurements:
                measurement_output = arm_dir / f"{measurement.name}.json"
                measurement_values = dict(values)
                measurement_values["output"] = str(measurement_output)
                command = _render(measurement.command, measurement_values)
                completed = _run(command, env, timeout=arm.request_timeout_s)
                (arm_dir / f"{measurement.name}.stdout.log").write_text(completed.stdout)
                (arm_dir / f"{measurement.name}.stderr.log").write_text(completed.stderr)
                if completed.returncode:
                    raise ProfileError(f"{arm.name} {measurement.name} failed with status {completed.returncode}; see {arm_dir / f'{measurement.name}.stderr.log'}")
                if not measurement_output.exists():
                    raise ProfileError(f"{arm.name} {measurement.name} did not create {measurement_output}")
                extra_measurements.append({
                    "name": measurement.name,
                    "command": list(command),
                    "command_text": _command_text(command),
                    "metrics": _server_metrics(measurement.result_kind, _parse_json_output(measurement_output.read_text(), f"{arm.name} {measurement.name}")),
                })
        finally:
            _stop_server(process)
    if not output_path.exists():
        raise ProfileError(f"{arm.name} client did not create {output_path}")
    result = {
        "name": arm.name,
        "type": "llama-server + speed-bench",
        "server_command": list(server_command),
        "server_command_text": _command_text(server_command),
        "client_command": list(client_command),
        "client_command_text": _command_text(client_command),
        "environment": _filtered_environment(env),
        "binary_version": _version(server_command[0], env),
        "models": _model_identity(arm.model_paths),
        "metrics": _speed_bench_metrics(_parse_json_output(output_path.read_text(), arm.name)),
        "extra_measurements": extra_measurements,
        "notes": list(arm.notes),
    }
    if arm.trace_mode != "off":
        if not trace_path.exists():
            raise ProfileError(f"{arm.name} did not produce {trace_path}")
        prefill_metric = next((item["metrics"] for item in extra_measurements if item["metrics"]["kind"] == "prefill"), {})
        start_ns = prefill_metric.get("request_start_raw_ns")
        end_ns = prefill_metric.get("request_end_raw_ns")
        prefill_window_ns = (start_ns, end_ns) if isinstance(start_ns, int) and isinstance(end_ns, int) and start_ns < end_ns else None
        result["trace"] = {
            "path": str(trace_path),
            "mode": arm.trace_mode,
            "integrity": "not-run",
            "analysis": _trace_summary(trace_path, arm.family_rules, prefill_min_tokens=arm.trace_prefill_min_tokens, prefill_target_tokens=arm.trace_prefill_target_tokens, prefill_window_ns=prefill_window_ns),
        }
    return result


def run_recipe(recipe: Recipe, artifacts: Path, *, arm_names: Sequence[str] = (), environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run selected recipe arms and return JSON-safe normalized measurement data."""
    artifacts.mkdir(parents=True, exist_ok=True)
    chosen = [arm for arm in recipe.arms if not arm_names or arm.name in arm_names]
    unknown = sorted(set(arm_names) - {arm.name for arm in recipe.arms})
    if unknown:
        raise ProfileError(f"recipe {recipe.name} has no arm(s): {', '.join(unknown)}")
    if not chosen:
        raise ProfileError(f"recipe {recipe.name} selected no arms")
    common_env = dict(environment or {})
    started = time.time()
    arms = []
    for arm in chosen:
        if isinstance(arm, BenchArm):
            arms.append(_run_bench_arm(arm, artifacts, common_env))
        elif isinstance(arm, ServerArm):
            arms.append(_run_server_arm(arm, artifacts, common_env))
        else:
            raise ProfileError(f"unsupported arm type: {type(arm).__name__}")
    return {
        "schema": "delphi-bottleneck-v1",
        "recipe": recipe.name,
        "description": recipe.description,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "repository": _repository_identity(),
        "artifacts": str(artifacts),
        "notes": list(recipe.notes),
        "arms": arms,
    }


def _replace_option(command: Sequence[str], option: str, value: str) -> tuple[str, ...]:
    result = list(command)
    for index, item in enumerate(result):
        if item == option:
            if index + 1 == len(result):
                raise ProfileError(f"option {option} has no value in recipe command")
            result[index + 1] = value
            return tuple(result)
        if option.startswith("--") and item.startswith(option + "="):
            result[index] = f"{option}={value}"
            return tuple(result)
    result.extend((option, value))
    return tuple(result)


def apply_overrides(recipe: Recipe, settings: Sequence[tuple[str, str, str]], environment: Mapping[str, str]) -> Recipe:
    """Apply --set ARM.OPTION=VALUE and --env overrides without mutating a recipe."""
    arms: list[Arm] = []
    names = {arm.name for arm in recipe.arms}
    for arm in recipe.arms:
        command = arm.command if isinstance(arm, BenchArm) else arm.server_command
        for arm_name, option, value in settings:
            if arm_name not in ("all", arm.name):
                continue
            command = _replace_option(command, option, value)
        env = dict(arm.env)
        env.update(environment)
        if isinstance(arm, BenchArm):
            arms.append(replace(arm, command=command, env=env))
        else:
            arms.append(replace(arm, server_command=command, env=env))
    unknown = {name for name, _, _ in settings if name != "all"} - names
    if unknown:
        raise ProfileError(f"recipe {recipe.name} has no arm(s): {', '.join(sorted(unknown))}")
    return replace(recipe, arms=tuple(arms))


def apply_bindir(recipe: Recipe, bindir: Path | None) -> Recipe:
    """Replace a recipe arm's direct llama executable with one from *bindir*.

    Recipes use direct binary commands. Client helpers such as Python and
    SPEED-Bench remain unchanged. A recipe that needs a launcher should make
    that launcher part of a wrapper script, not rely on this replacement.
    """
    if bindir is None:
        return recipe
    bindir = bindir.resolve()
    arms: list[Arm] = []
    for arm in recipe.arms:
        if isinstance(arm, BenchArm):
            command = (str(bindir / Path(arm.command[0]).name), *arm.command[1:])
            arms.append(replace(arm, command=command))
        else:
            command = (str(bindir / Path(arm.server_command[0]).name), *arm.server_command[1:])
            arms.append(replace(arm, server_command=command))
    return replace(recipe, arms=tuple(arms))


def load_recipe(reference: str) -> Recipe:
    """Load /path/to/recipes.py:factory. The factory must return a Recipe."""
    if ":" not in reference:
        raise ProfileError("--recipe must be /path/to/recipes.py:factory")
    raw_path, factory_name = reference.rsplit(":", 1)
    path = Path(raw_path).resolve()
    spec = importlib.util.spec_from_file_location(f"delphi_recipe_{path.stem}", path)
    if not spec or not spec.loader:
        raise ProfileError(f"cannot load recipe file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ProfileError(f"recipe factory not found: {reference}")
    recipe = factory()
    if not isinstance(recipe, Recipe):
        raise ProfileError(f"recipe factory {reference} returned {type(recipe).__name__}, not Recipe")
    return recipe


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def _serial_gain_ceiling(component_ms: float | None, wall_ms: float | None) -> str:
    """Return the optimistic gain if a measured interval is serial and removed."""
    if component_ms is None or wall_ms is None or component_ms < 0 or wall_ms <= 0:
        return "n/a"
    share = component_ms / wall_ms
    if share >= 1:
        return "not bounded (<1x wall needed)"
    return f"<={100.0 * (1.0 / (1.0 - share) - 1.0):.1f}%"


def _render_detail_rows(lines: list[str], values: Mapping[str, float], wall_ms: float | None, *, unit: str = "ms/token", limit: int = 10) -> None:
    ordered = sorted(((name, duration) for name, duration in values.items() if duration >= 0.0005), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return
    lines.extend(("", f"| measured detail | {unit} | optimistic gain ceiling | evidence |", "| --- | ---: | ---: | --- |"))
    for name, duration in ordered[:limit]:
        evidence = "DEVICE_EVENT sum, non-additive" if name.startswith("device ") else "HOST_SAFE interval, non-additive"
        lines.append(f"| {name} | {_fmt(duration)} | {_serial_gain_ceiling(duration, wall_ms)} | {evidence} |")


def _render_diagnostic_groups(lines: list[str], title: str, values: Mapping[str, float], wall_ms: float | None, complete: bool, *, limit: int) -> None:
    if not values:
        return
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    lines.extend(("", f"{title}: {len(ordered)} observed groups (largest {min(limit, len(ordered))} shown; all groups in JSON).", "", "| observed group | diagnostic ms/token | conditional max throughput gain |", "| --- | ---: | ---: |"))
    for name, duration in ordered[:limit]:
        ceiling = _serial_gain_ceiling(duration, wall_ms) if complete else "n/a (incomplete events)"
        lines.append(f"| {name} | {_fmt(duration)} | {ceiling} |")
    if len(ordered) > limit:
        next_duration = ordered[limit][1]
        ceiling = _serial_gain_ceiling(next_duration, wall_ms) if complete else "n/a (incomplete events)"
        omitted = len(ordered) - limit
        subject = "The 1 omitted group" if omitted == 1 else f"Each of the {omitted} omitted groups"
        lines.append(f"{subject} is <= {_fmt(next_duration)} diagnostic ms/token ({ceiling} conditional gain); see JSON for names.")


def render_summary(result: Mapping[str, Any]) -> str:
    """Render the short, prompt-facing view of a normalized result."""
    lines = [f"# {result['recipe']} bottleneck measurement", "", f"Measured: {result['created_utc']}"]
    repository = result["repository"]
    lines.append(f"Checkout: {repository.get('commit') or 'unknown'} ({repository.get('tree', 'unknown')} tree); arm-level Binary lines identify the tested build.")
    lines.extend(("", "This is measured on this machine and configuration. It overrides prior assumptions. Rows can overlap; do not add their times or ceilings. The ceiling assumes removing only that measured interval without changing other work."))
    for arm in result["arms"]:
        if arm["name"].endswith("-op-diagnostic"):
            continue
        lines.extend(("", f"## {arm['name']} ({arm['type']})"))
        for note in arm.get("notes", ()):
            lines.append(f"Note: {note}")
        binary_version = arm.get("binary_version")
        if binary_version:
            lines.append(f"Binary: {binary_version.splitlines()[0]}")
        metrics = arm["metrics"]
        end_to_end_wall_ms = None
        if metrics["kind"] == "speed-bench":
            predicted_t_s = metrics.get("predicted_t_s")
            if predicted_t_s is not None and predicted_t_s > 0:
                end_to_end_wall_ms = 1000.0 / predicted_t_s
            lines.append(f"End to end: PP {_fmt(metrics.get('prompt_t_s'), ' t/s')}; TG {_fmt(predicted_t_s, ' t/s')}; TG wall {_fmt(end_to_end_wall_ms, ' ms/token')}; latency {_fmt(metrics.get('latency_s'), ' s')}; acceptance {metrics.get('accept_rate', 'n/a')}")
        else:
            for test in metrics["tests"]:
                name = test["test"].get("model_type", "benchmark")
                lines.append(f"{name}: {_fmt(test['throughput_t_s']['median'], ' t/s')} median ({test['throughput_t_s']['count']} samples)")
        for measurement in arm.get("extra_measurements", ()):
            extra_metrics = measurement["metrics"]
            if extra_metrics["kind"] == "prefill":
                prefill_wall_ms = None
                if extra_metrics.get("prompt_t_s"):
                    prefill_wall_ms = 1000.0 / extra_metrics["prompt_t_s"]
                lines.append(
                    f"Prefill ({measurement['name']}): target {extra_metrics.get('target_tokens', 'n/a')} tokens; "
                    f"evaluated {extra_metrics.get('prompt_tokens', 'n/a')}; PP {_fmt(extra_metrics.get('prompt_t_s'), ' t/s')}; "
                    f"PP wall {_fmt(prefill_wall_ms, ' ms/token')}; latency {_fmt(extra_metrics.get('latency_s'), ' s')}"
                )
        trace = arm.get("trace")
        if not trace:
            lines.append("Trace: not collected")
            continue
        decode = trace["analysis"]["decode"]
        lines.append(f"Trace ({trace['mode']}, integrity {trace['integrity']}): {decode['executions']} measured decode executions; instrumented window {_fmt(decode['wall_ms_per_token']['median'], ' ms/token')}")
        if decode["executions"] < 10:
            lines.append("Trace breakdown: insufficient steady decode executions; use the end-to-end measurement above.")
        elif "--spec-type" in arm.get("server_command", ()) and end_to_end_wall_ms and sum(decode[key]["median"] for key in ("cpu_compute_ms", "copy_handling_ms", "host_submit_ms", "device_submit_ms")) < 0.5 * end_to_end_wall_ms:
            lines.append("Trace breakdown: speculative decode spans cover too little of end-to-end wall time for an Amdahl screen; likely only a subset of target/draft contexts. Use the end-to-end measurement above.")
        else:
            lines.extend(("", "| measured span | median ms/token | optimistic gain ceiling | evidence |", "| --- | ---: | ---: | --- |"))
            for key, label in (("cpu_compute_ms", "CPU compute"), ("copy_handling_ms", "copy handling"), ("host_submit_ms", "host submit"), ("device_submit_ms", "device submit sum")):
                evidence = "HOST interval, upper-bound only" if key != "device_submit_ms" else "DEVICE_EVENT, non-additive"
                ceiling = _serial_gain_ceiling(decode[key]["median"], end_to_end_wall_ms or decode["wall_ms_per_token"]["median"])
                lines.append(f"| {label} | {_fmt(decode[key]['median'])} | {ceiling} | {evidence} |")
            detail = {name: stats["median"] for name, stats in decode.get("host_detail_ms", {}).items()}
            detail.update({f"device interval on {name}": stats["median"] for name, stats in decode.get("device_submit_by_backend_ms", {}).items()})
            _render_detail_rows(lines, detail, end_to_end_wall_ms or decode["wall_ms_per_token"]["median"])
        prefill = trace["analysis"].get("prefill")
        if prefill and prefill["executions"]:
            prefill_metric = next((item["metrics"] for item in arm.get("extra_measurements", ()) if item["metrics"].get("kind") == "prefill"), None)
            prefill_wall_ms = None
            if prefill_metric and prefill_metric.get("prompt_t_s"):
                prefill_wall_ms = 1000.0 / prefill_metric["prompt_t_s"]
            aggregate = prefill.get("aggregate_per_client_token_ms")
            span_values = aggregate or {key: prefill[key]["median"] for key in ("cpu_compute_ms", "copy_handling_ms", "host_submit_ms", "device_submit_ms")}
            wall_ms = span_values["wall_ms"] if aggregate else prefill["wall_ms_per_token"]["median"]
            span_label = "aggregate ms/client token" if aggregate else "median ms/token"
            evidence_suffix = " across trace contexts" if aggregate else ""
            lines.extend(("", f"Prefill trace: {prefill['executions']} selected executions, {prefill['tokens']['total']} trace-token executions; instrumented window {_fmt(wall_ms, ' ms/client token' if aggregate else ' ms/token')}"))
            lines.extend(("", f"| measured prefill span | {span_label} | optimistic gain ceiling | evidence |", "| --- | ---: | ---: | --- |"))
            for key, label in (("cpu_compute_ms", "CPU compute"), ("copy_handling_ms", "copy handling"), ("host_submit_ms", "host submit"), ("device_submit_ms", "device submit sum")):
                evidence = f"HOST interval, upper-bound only{evidence_suffix}" if key != "device_submit_ms" else "DEVICE_EVENT, non-additive"
                ceiling = _serial_gain_ceiling(span_values[key], prefill_wall_ms or wall_ms)
                lines.append(f"| {label} | {_fmt(span_values[key])} | {ceiling} | {evidence} |")
            if aggregate:
                detail = dict(aggregate.get("host_detail_ms", {}))
                detail.update({f"device interval on {name}": value for name, value in aggregate.get("device_submit_by_backend_ms", {}).items()})
            else:
                detail = {name: stats["median"] for name, stats in prefill.get("host_detail_ms", {}).items()}
                detail.update({f"device interval on {name}": stats["median"] for name, stats in prefill.get("device_submit_by_backend_ms", {}).items()})
            _render_detail_rows(lines, detail, prefill_wall_ms or wall_ms, unit="ms/client token" if aggregate else "ms/token")
    production = {arm["name"]: arm for arm in result["arms"] if not arm["name"].endswith("-op-diagnostic")}
    for arm in result["arms"]:
        if not arm["name"].endswith("-op-diagnostic"):
            continue
        base_name = arm["name"].removesuffix("-op-diagnostic")
        base = production.get(base_name)
        lines.extend(("", f"## {arm['name']} operation diagnostic"))
        lines.append("Graphs and fusion were disabled for per-op GPU timing. CPU split work is not decomposed by operation. The conditional gains below use production wall time as the denominator; they are screening estimates, not proven production ceilings.")
        binary_version = arm.get("binary_version")
        if binary_version:
            lines.append(f"Binary: {binary_version.splitlines()[0]}")
        if not base or not arm.get("trace"):
            lines.append("No matched production arm or trace was collected.")
            continue
        base_decode_t_s = base["metrics"].get("predicted_t_s")
        decode_wall_ms = 1000.0 / base_decode_t_s if base_decode_t_s and base_decode_t_s > 0 else None
        base_prefill = next((item["metrics"] for item in base.get("extra_measurements", ()) if item["metrics"].get("kind") == "prefill"), {})
        prefill_t_s = base_prefill.get("prompt_t_s")
        prefill_wall_ms = 1000.0 / prefill_t_s if prefill_t_s and prefill_t_s > 0 else None
        analysis = arm["trace"]["analysis"]
        statuses = analysis.get("gpu_timing_status_counts", {})
        if any(count for status, count in statuses.items() if status != "resolved"):
            lines.append("Some device events were unresolved; per-op times are incomplete, so conditional ceilings are suppressed.")
        complete = not any(count for status, count in statuses.items() if status != "resolved")
        for workload, groups, wall_ms, count in (
            ("decode", analysis["decode"], decode_wall_ms, analysis["decode"]["executions"]),
            ("prefill", analysis.get("prefill", {}).get("aggregate_per_client_token_ms", {}), prefill_wall_ms, analysis.get("prefill", {}).get("executions", 0)),
        ):
            lines.extend(("", f"{workload.capitalize()} per-op timing: {count} trace executions; production wall {_fmt(wall_ms, ' ms/token')}."))
            if not groups.get("diagnostic_by_op_ms"):
                lines.append("No resolved per-op device events for this workload.")
                continue
            breakdowns = [("diagnostic_by_op_ms", "By GGML operation", 12), ("diagnostic_by_node_role_ms", "By recurring node role", 20)]
            backend_ops = groups.get("diagnostic_by_backend_op_ms", {})
            if len({name.split(" / ", 1)[0] for name in backend_ops}) > 1:
                breakdowns.append(("diagnostic_by_backend_op_ms", "By backend and GGML operation", 12))
            for key, title, limit in breakdowns:
                raw = groups.get(key, {})
                values = {name: value["median"] for name, value in raw.items()} if workload == "decode" else raw
                _render_diagnostic_groups(lines, title, values, wall_ms, complete, limit=limit)
    return "\n".join(lines) + "\n"


def _parse_set(value: str) -> tuple[str, str, str]:
    if "=" not in value or "." not in value.split("=", 1)[0]:
        raise ProfileError("--set must be ARM.OPTION=VALUE, for example all.-ub=2048")
    target, replacement = value.split("=", 1)
    arm, option = target.split(".", 1)
    return arm, option, replacement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="/path/to/recipes.py:factory")
    parser.add_argument("--output", required=True, type=Path, help="normalized result JSON")
    parser.add_argument("--artifacts", type=Path, help="raw logs and execution traces (default: beside --output)")
    parser.add_argument("--bindir", type=Path, help="directory containing the llama executables for every selected arm")
    parser.add_argument("--arm", action="append", default=[], help="run one arm; repeat to select several")
    parser.add_argument("--set", action="append", default=[], metavar="ARM.OPTION=VALUE", help="replace or add a server/bench option")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE", help="set a runtime environment variable for every arm")
    parser.add_argument("--print-summary", action="store_true", help="also print a terse Markdown result")
    args = parser.parse_args()
    try:
        recipe = load_recipe(args.recipe)
        recipe = apply_bindir(recipe, args.bindir)
        settings = [_parse_set(value) for value in args.set]
        environment = dict(_parse_assignment(value, "--env") for value in args.env)
        recipe = apply_overrides(recipe, settings, environment)
        artifacts = args.artifacts or args.output.with_name(args.output.stem + "-artifacts")
        result = run_recipe(recipe, artifacts, arm_names=args.arm)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.print_summary:
            print(render_summary(result), end="")
        else:
            print(f"wrote {args.output}")
    except (OSError, ProfileError, subprocess.TimeoutExpired) as error:
        print(f"bottleneck profile: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
