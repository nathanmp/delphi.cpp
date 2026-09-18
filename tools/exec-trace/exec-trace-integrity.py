#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import shlex
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


INTEGRITY_HEADER = struct.Struct("<8sII25Q")
TRACE_HEADER = struct.Struct("<8sIIIIQQQQ")
TRACE_V3_STATS = struct.Struct("<5Q")

INTEGRITY_FIELDS = (
    "graph_count", "node_count", "split_count",
    "graph_sum", "graph_xor", "assignment_sum", "assignment_xor", "split_sum", "split_xor",
    "copy_count", "copy_sum", "copy_xor",
    "dispatch_count", "dispatch_sum", "dispatch_xor",
    "gpu_graph_event_count", "gpu_graph_sum", "gpu_graph_xor",
    "gpu_graph_capture_count", "gpu_graph_update_count", "gpu_graph_replay_count", "gpu_graph_fallback_count",
    "eval_callback_count", "build_flags", "runtime_flags",
)

BUILD_FLAGS = {
    1 << 0: "asserts_enabled",
    1 << 1: "optimization_disabled",
    1 << 2: "address_sanitizer",
    1 << 3: "thread_sanitizer",
    1 << 4: "undefined_behavior_sanitizer",
}
RUNTIME_FLAGS = {
    1 << 0: "GGML_CUDA_DISABLE_GRAPHS",
    1 << 1: "GGML_HIP_DISABLE_GRAPHS",
    1 << 2: "GGML_CUDA_OP_TRACE",
    1 << 3: "GGML_HIP_OP_TRACE",
    1 << 4: "GGML_EXEC_TRACE_DETAIL=full",
    1 << 5: "GGML_EXEC_TRACE_GPU_TIMING",
}
GPU_TIMING_RUNTIME_FLAG = 1 << 5
GPU_TIMING_ENV = ("GGML_EXEC_TRACE_GPU_TIMING", "GGML_EXEC_TRACE_GPU_TIMING_EVENTS")

STRUCTURE_FIELDS = {
    "graph fingerprint": ("graph_count", "node_count", "graph_sum", "graph_xor"),
    "split fingerprint": ("split_count", "split_sum", "split_xor"),
    "backend fingerprint": ("assignment_sum", "assignment_xor"),
    "copy fingerprint": ("copy_count", "copy_sum", "copy_xor"),
    "dispatch fingerprint": ("dispatch_count", "dispatch_sum", "dispatch_xor"),
    "GPU graph behavior": (
        "gpu_graph_event_count", "gpu_graph_sum", "gpu_graph_xor",
        "gpu_graph_capture_count", "gpu_graph_update_count",
        "gpu_graph_replay_count", "gpu_graph_fallback_count",
    ),
    "scheduler eval callback": ("eval_callback_count",),
}

ENV_PREFIXES = ("GGML_", "CUDA_", "HIP_", "HSA_", "ROCR_", "ROCPROFILER_", "NSYS_", "NVIDIA_")
ENV_NAMES = {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "LD_LIBRARY_PATH", "LD_PRELOAD"}
TEST_IDENTITY_FIELDS = (
    "model_filename", "model_type", "n_batch", "n_ubatch", "n_threads", "cpu_mask", "cpu_strict", "poll",
    "type_k", "type_v", "n_gpu_layers", "n_cpu_moe", "split_mode", "main_gpu", "no_kv_offload",
    "flash_attn", "devices", "tensor_split", "tensor_buft_overrides", "load_mode", "lazy_mode", "embeddings",
    "no_op_offload", "no_host", "fit_target", "fit_min_ctx", "n_prompt", "n_gen", "n_depth",
)


class IntegrityError(Exception):
    pass


def load_trace_reader():
    path = Path(__file__).with_name("exec-trace-dump.py")
    spec = importlib.util.spec_from_file_location("exec_trace_dump", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACE = load_trace_reader()


def read_integrity(path):
    data = path.read_bytes()
    if len(data) < INTEGRITY_HEADER.size:
        raise IntegrityError(f"integrity summary is incomplete: {path}")
    magic, version, header_size, *values = INTEGRITY_HEADER.unpack_from(data)
    if magic.rstrip(b"\0") != b"GEXINT":
        raise IntegrityError(f"invalid integrity summary magic: {path}")
    if version != 1 or header_size != INTEGRITY_HEADER.size:
        raise IntegrityError(f"unsupported integrity summary schema {version}/{header_size}")
    return dict(zip(INTEGRITY_FIELDS, values))


def read_trace_stats(path):
    data = path.read_bytes()
    if len(data) < TRACE_HEADER.size:
        raise IntegrityError(f"raw trace is incomplete: {path}")
    magic, version, header_size, event_size, flags, _, event_count, string_count, _ = TRACE_HEADER.unpack_from(data)
    if magic.rstrip(b"\0") != b"GEXECTR":
        raise IntegrityError(f"invalid raw trace magic: {path}")
    if version < 3 or header_size < TRACE_HEADER.size + TRACE_V3_STATS.size:
        raise IntegrityError("raw trace does not contain schema v3 self-overhead statistics")
    capacity, attempted, dropped, growths, serialization_ns = TRACE_V3_STATS.unpack_from(data, TRACE_HEADER.size)
    return {
        "schema_version": version,
        "truncated": bool(flags & 1),
        "event_count": event_count,
        "attempted_event_count": attempted,
        "dropped_event_count": dropped,
        "interned_string_count": string_count,
        "event_buffer_bytes": capacity * event_size,
        "buffer_growth_count": growths,
        "serialization_duration_ns": serialization_ns,
        "file_bytes": len(data),
    }


def relevant_environment(environment):
    return {
        key: value for key, value in sorted(environment.items())
        if key in ENV_NAMES or key.startswith(ENV_PREFIXES)
    }


def command_text(command, environment):
    assignments = [f"{key}={shlex.quote(value)}" for key, value in relevant_environment(environment).items()]
    return " ".join(assignments + [shlex.join(command)])


def run_benchmark(command, environment):
    start_ns = time.monotonic_ns()
    completed = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_wall_ns = time.monotonic_ns() - start_ns
    if completed.returncode != 0:
        raise IntegrityError(
            f"benchmark exited with status {completed.returncode}: {shlex.join(command)}\n{completed.stderr.strip()}")
    try:
        results = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise IntegrityError(f"benchmark did not produce JSON: {error}") from error
    if not isinstance(results, list) or not results:
        raise IntegrityError("benchmark JSON contains no tests")
    return {
        "command": command,
        "command_text": command_text(command, environment),
        "environment": relevant_environment(environment),
        "process_wall_ns": process_wall_ns,
        "stderr": completed.stderr,
        "tests": results,
    }


def combine_benchmark_runs(runs):
    if not runs:
        raise IntegrityError("cannot combine an empty benchmark run list")
    identities = [[test_identity(test) for test in run["tests"]] for run in runs]
    if any(identity != identities[0] for identity in identities[1:]):
        raise IntegrityError("repeated benchmark process parameters differ")
    tests = []
    for test_index in range(len(runs[0]["tests"])):
        combined = dict(runs[0]["tests"][test_index])
        combined["samples_ns"] = [sample for run in runs for sample in run["tests"][test_index]["samples_ns"]]
        combined["samples_ts"] = [sample for run in runs for sample in run["tests"][test_index]["samples_ts"]]
        tests.append(combined)
    return {
        "command": [run["command"] for run in runs],
        "command_text": [run["command_text"] for run in runs],
        "environment": runs[0]["environment"],
        "process_wall_ns": sum(run["process_wall_ns"] for run in runs),
        "stderr": "".join(run["stderr"] for run in runs),
        "tests": tests,
        "processes": runs,
    }


def combine_trace_stats(stats):
    if not stats:
        raise IntegrityError("cannot combine an empty trace statistics list")
    combined = dict(stats[0])
    for field in (
            "event_count", "attempted_event_count", "dropped_event_count",
            "buffer_growth_count", "serialization_duration_ns", "file_bytes"):
        combined[field] = sum(item[field] for item in stats)
    combined["truncated"] = any(item["truncated"] for item in stats)
    combined["interned_string_count"] = max(item["interned_string_count"] for item in stats)
    combined["event_buffer_bytes"] = max(item["event_buffer_bytes"] for item in stats)
    combined["run_stats"] = stats
    return combined


def test_identity(test):
    return {field: test.get(field) for field in TEST_IDENTITY_FIELDS}


def sample_summary(test):
    samples_ns = test.get("samples_ns", [])
    samples_ts = test.get("samples_ts", [])
    if not samples_ns or len(samples_ns) != len(samples_ts):
        raise IntegrityError("benchmark JSON has missing or inconsistent samples")
    median_ts = statistics.median(samples_ts)
    return {
        "samples_ns": samples_ns,
        "samples_ts": samples_ts,
        "median_wall_ns": statistics.median(samples_ns),
        "min_wall_ns": min(samples_ns),
        "max_wall_ns": max(samples_ns),
        "median_ts": median_ts,
        "min_ts": min(samples_ts),
        "max_ts": max(samples_ts),
        "dispersion_percent": 0.0 if median_ts == 0 else 100.0 * (max(samples_ts) - min(samples_ts)) / median_ts,
    }


def compare_performance(baseline_tests, trace_tests):
    if len(baseline_tests) != len(trace_tests):
        raise IntegrityError("baseline and trace runs produced different test counts")
    comparisons = []
    for index, (baseline, traced) in enumerate(zip(baseline_tests, trace_tests)):
        if test_identity(baseline) != test_identity(traced):
            raise IntegrityError(f"baseline and trace benchmark {index} parameters differ")
        baseline_summary = sample_summary(baseline)
        trace_summary = sample_summary(traced)
        if baseline_summary["median_ts"] <= 0:
            raise IntegrityError(f"baseline benchmark {index} has non-positive throughput")
        delta_percent = 100.0 * (trace_summary["median_ts"] / baseline_summary["median_ts"] - 1.0)
        slowdown_percent = -delta_percent
        status = "preferred" if slowdown_percent <= 1.0 else "acceptable" if slowdown_percent <= 2.0 else "investigate"
        comparisons.append({
            "index": index,
            "identity": test_identity(baseline),
            "baseline": baseline_summary,
            "trace": trace_summary,
            "throughput_delta_percent": delta_percent,
            "slowdown_percent": slowdown_percent,
            "classification": status,
        })
    return comparisons


def compare_structure(baseline, traced):
    comparisons = {}
    for label, fields in STRUCTURE_FIELDS.items():
        baseline_values = {field: baseline[field] for field in fields}
        trace_values = {field: traced[field] for field in fields}
        comparisons[label] = {
            "match": baseline_values == trace_values,
            "baseline": baseline_values,
            "trace": trace_values,
        }
    return comparisons


def active_flags(value, definitions):
    return [name for bit, name in definitions.items() if value & bit]


def read_cmake_cache(binary):
    cache_path = binary.resolve().parent.parent / "CMakeCache.txt"
    if not cache_path.is_file():
        return {"path": str(cache_path), "found": False, "values": {}}
    interesting = {}
    for line in cache_path.read_text(errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key_type, value = line.split("=", 1)
        key, _ = key_type.split(":", 1)
        if key == "CMAKE_BUILD_TYPE" or "SANITIZE" in key or "OP_TRACE" in key or key in ("GGML_HIP", "GGML_CUDA"):
            interesting[key] = value
    return {"path": str(cache_path), "found": True, "values": interesting}


def linked_profiler_libraries(binary):
    markers = ("roctx", "rocprof", "cupti", "nvperf", "nsys")
    candidates = [binary]
    candidates.extend(binary.parent.glob("libggml*.so*"))
    dependencies = []
    for candidate in candidates:
        try:
            result = subprocess.run(
                ["readelf", "-d", str(candidate)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            return []
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            if "(NEEDED)" in line and any(marker in line.lower() for marker in markers):
                dependencies.append(f"{candidate.name}: {line.strip()}")
    return sorted(set(dependencies))


def contamination_reasons(integrity, environment, cache, linked_libraries, launcher):
    reasons = []
    reasons.extend(f"instrumented build: {name}" for name in active_flags(integrity["build_flags"], BUILD_FLAGS))
    reasons.extend(f"runtime setting: {name}" for name in active_flags(integrity["runtime_flags"], RUNTIME_FLAGS))

    profiler_env = [
        key for key, value in environment.items()
        if (key.startswith(("ROCPROFILER_", "NSYS_")) or key in ("CUDA_INJECTION64_PATH", "NVPROF_ID")) and value
    ]
    if environment.get("LD_PRELOAD") and any(
            marker in environment["LD_PRELOAD"].lower() for marker in ("roctx", "rocprof", "cupti", "profiler")):
        profiler_env.append("LD_PRELOAD")
    reasons.extend(f"external profiler environment: {key}" for key in sorted(set(profiler_env)))

    if launcher and any(marker in Path(token).name.lower() for token in launcher for marker in ("rocprof", "nsys", "ncu", "profiler")):
        reasons.append(f"harness launched external profiler: {shlex.join(launcher)}")
    reasons.extend(f"profiler library directly linked: {line}" for line in linked_libraries)

    cache_values = cache["values"]
    build_type = cache_values.get("CMAKE_BUILD_TYPE", "").lower()
    if build_type and build_type not in ("release", "relwithdebinfo", "minsizerel"):
        reasons.append(f"unexpected CMake build type: {cache_values['CMAKE_BUILD_TYPE']}")
    for key, value in cache_values.items():
        if "SANITIZE" in key and value.upper() in ("ON", "TRUE", "1"):
            reasons.append(f"sanitizer enabled in CMake cache: {key}")
        if "OP_TRACE" in key and value.upper() not in ("", "OFF", "FALSE", "0"):
            reasons.append(f"operation tracing enabled in CMake cache: {key}={value}")
    return sorted(set(reasons))


def compare_structure_with_repeat(baseline, traced, traced_repeat):
    structure = compare_structure(baseline, traced)
    repeat_structure = compare_structure(traced, traced_repeat)
    for label, comparison in structure.items():
        comparison["trace_repeat"] = {field: traced_repeat[field] for field in STRUCTURE_FIELDS[label]}
        comparison["match"] = comparison["match"] and repeat_structure[label]["match"]
    return structure


def gpu_timing_completeness(summaries):
    status = Counter()
    placement = Counter()
    unreleased = 0
    recorded = 0
    for summary in summaries:
        status.update(summary["status_counts"])
        placement.update(summary["epoch_placement_counts"])
        unreleased += summary["spans_unreleased"] or 0
        recorded += summary["spans_recorded"]
    uncertainties = [summary["placement_uncertainty_max_ns"] for summary in summaries
                     if summary["placement_uncertainty_max_ns"] is not None]
    return {
        "spans_recorded": recorded,
        "spans_unreleased": unreleased,
        "status_counts": dict(status),
        "epoch_placement_counts": dict(placement),
        "resolved_percent": 100.0 * status["resolved"] / recorded if recorded else 0.0,
        "placement_uncertainty_max_ns": max(uncertainties) if uncertainties else None,
    }


def make_gpu_timing_report(baseline_run, trace_run, baseline_integrity, timing, cache, linked, launcher):
    integrity = timing["integrity"]
    performance = compare_performance(baseline_run["tests"], timing["run"]["tests"])
    versus_trace = compare_performance(trace_run["tests"], timing["run"]["tests"])
    structure = compare_structure_with_repeat(baseline_integrity, integrity, timing["integrity_repeat"])
    structural_mismatches = [label for label, comparison in structure.items() if not comparison["match"]]
    # The timing flag is the setting under test; every other contamination source still applies.
    expected_flags = dict(integrity, runtime_flags=integrity["runtime_flags"] & ~GPU_TIMING_RUNTIME_FLAG)
    contamination = contamination_reasons(expected_flags, timing["run"]["environment"], cache, linked, launcher)
    completeness = gpu_timing_completeness(timing["summaries"])
    status = completeness["status_counts"]

    reasons = []
    if not integrity["runtime_flags"] & GPU_TIMING_RUNTIME_FLAG:
        reasons.append("GPU timing was not active in the timing run")
    reasons.extend(f"structural mismatch: {label}" for label in structural_mismatches)
    reasons.extend(contamination)
    if timing["stats"]["truncated"] or timing["stats"]["dropped_event_count"]:
        reasons.append(f"timing trace dropped {timing['stats']['dropped_event_count']} events")
    if not status.get("resolved"):
        reasons.append("no GPU timing span was resolved")
    if status.get("failed"):
        reasons.append(f"{status['failed']} GPU timing spans failed")
    if completeness["epoch_placement_counts"].get("inconsistent"):
        reasons.append(f"{completeness['epoch_placement_counts']['inconsistent']} GPU timing epochs have inconsistent host placement")
    if completeness["spans_unreleased"]:
        reasons.append(f"{completeness['spans_unreleased']} GPU timing spans were never released")

    gaps = []
    if status.get("not_recorded"):
        gaps.append(f"{status['not_recorded']} spans were not recorded because the event pool was full")
    if status.get("unresolved"):
        gaps.append(f"{status['unresolved']} spans were released before completion was known")

    perturbative = [item for item in performance if item["classification"] == "investigate"]
    if reasons:
        classification = "invalid"
    elif perturbative:
        classification = "perturbative"
    elif gaps:
        classification = "low-perturbation, incomplete coverage"
    else:
        classification = "low-perturbation"
    reasons.extend(
        f"benchmark {item['index']} timing slowdown is {item['slowdown_percent']:.2f}%" for item in perturbative)

    return {
        "mode": timing["mode"],
        "provenance": "DEVICE_EVENT",
        "classification": classification,
        "trustworthy": classification.startswith("low-perturbation"),
        "reasons": reasons,
        "coverage_gaps": gaps,
        "performance": performance,
        "performance_versus_trace": versus_trace,
        "structure": structure,
        "completeness": completeness,
        "trace_overhead": timing["stats"],
        "contamination_detection": {"detected_reasons": contamination, "detection_exhaustive": False},
        "run": timing["run"],
        "integrity": {"timing": integrity, "timing_repeat": timing["integrity_repeat"]},
        "summaries": timing["summaries"],
    }


def make_report(
        baseline_run, baseline_structure_run, trace_run,
        baseline_integrity, trace_integrity, trace_integrity_repeat,
        trace_stats, cache, linked, launcher, gpu_timing=None):
    if [test_identity(test) for test in baseline_run["tests"]] != [
            test_identity(test) for test in baseline_structure_run["tests"]]:
        raise IntegrityError("clean baseline and structural witness benchmark parameters differ")
    performance = compare_performance(baseline_run["tests"], trace_run["tests"])
    structure = compare_structure_with_repeat(baseline_integrity, trace_integrity, trace_integrity_repeat)
    structural_mismatches = [label for label, comparison in structure.items() if not comparison["match"]]
    contamination = contamination_reasons(trace_integrity, trace_run["environment"], cache, linked, launcher)

    reasons = []
    reasons.extend(f"structural mismatch: {label}" for label in structural_mismatches)
    reasons.extend(contamination)
    if trace_stats["truncated"] or trace_stats["dropped_event_count"]:
        reasons.append(f"trace dropped {trace_stats['dropped_event_count']} events")
    if trace_integrity["eval_callback_count"]:
        reasons.append("scheduler evaluation callback was active")
    investigate = [item for item in performance if item["classification"] == "investigate"]
    reasons.extend(
        f"benchmark {item['index']} trace slowdown is {item['slowdown_percent']:.2f}%" for item in investigate)

    production_representative = not reasons
    if not production_representative:
        result = "invalid for production-representative timing"
    elif any(item["classification"] == "acceptable" for item in performance):
        result = "acceptable for Milestone 1"
    else:
        result = "production-safe"

    report = {
        "schema_version": 1,
        "production_representative": production_representative,
        "result": result,
        "reasons": reasons,
        "timing_provenance": {"host_spans": "HOST_SAFE"},
        "performance": performance,
        "structure": structure,
        "trace_overhead": trace_stats,
        "contamination_detection": {
            "detected_reasons": contamination,
            "detection_exhaustive": False,
            "cmake_cache": cache,
            "direct_profiler_dependencies": linked,
        },
        "runs": {"baseline": baseline_run, "baseline_structure": baseline_structure_run, "trace": trace_run},
        "integrity": {
            "baseline": baseline_integrity,
            "trace": trace_integrity,
            "trace_repeat": trace_integrity_repeat,
        },
    }
    if gpu_timing:
        report["gpu_timing"] = make_gpu_timing_report(
            baseline_run, trace_run, baseline_integrity, gpu_timing, cache, linked, launcher)
        report["timing_provenance"]["gpu_timing"] = "DEVICE_EVENT"
    return report


def test_name(identity):
    if identity["n_prompt"] and not identity["n_gen"]:
        return f"PP {identity['n_prompt']}"
    if identity["n_gen"] and not identity["n_prompt"]:
        return f"TG {identity['n_gen']}"
    return f"PP {identity['n_prompt']} + TG {identity['n_gen']}"


def print_gpu_timing_report(timing):
    print()
    print(f"GPU timing ({timing['mode']}, {timing['provenance']})")
    for item, versus_trace in zip(timing["performance"], timing["performance_versus_trace"]):
        name = test_name(item["identity"])
        print(f"{name} timing:   {item['trace']['median_ts']:.2f} tok/s")
        print(f"{name} vs clean: {item['throughput_delta_percent']:+.2f}% ({item['classification']})")
        print(f"{name} vs trace: {versus_trace['throughput_delta_percent']:+.2f}%")
        print(f"{name} timing samples: {item['trace']['samples_ts']}")
    for label, comparison in timing["structure"].items():
        print(f"{label + ':':28} {'MATCH' if comparison['match'] else 'MISMATCH'}")
    completeness = timing["completeness"]
    statuses = " ".join(f"{name}={count}" for name, count in sorted(completeness["status_counts"].items()))
    print(f"timing spans:   {completeness['spans_recorded']} ({statuses})")
    print(f"epoch placement: {completeness['epoch_placement_counts']} max uncertainty "
          f"{TRACE.microseconds(completeness['placement_uncertainty_max_ns'])} us")
    for gap in timing["coverage_gaps"]:
        print(f"coverage gap:   {gap}")
    for reason in timing["reasons"]:
        print(f"  - {reason}")
    print(f"GPU TIMING: {timing['classification'].upper()}")


def print_report(report):
    print("Profiler integrity")
    for item in report["performance"]:
        name = test_name(item["identity"])
        print(f"{name} baseline: {item['baseline']['median_ts']:.2f} tok/s")
        print(f"{name} trace:    {item['trace']['median_ts']:.2f} tok/s")
        print(f"{name} delta:    {item['throughput_delta_percent']:+.2f}% ({item['classification']})")
        print(f"{name} baseline samples: {item['baseline']['samples_ts']}")
        print(f"{name} trace samples:    {item['trace']['samples_ts']}")
    print()
    for label, comparison in report["structure"].items():
        print(f"{label + ':':28} {'MATCH' if comparison['match'] else 'MISMATCH'}")
    overhead = report["trace_overhead"]
    print()
    print(f"trace events:   {overhead['event_count']}")
    print(f"dropped events: {overhead['dropped_event_count']}")
    print(f"trace buffer:   {overhead['event_buffer_bytes']} bytes per process")
    print(f"serialization:  {overhead['serialization_duration_ns'] / 1e6:.3f} ms total")
    if report["reasons"]:
        print("reasons:")
        for reason in report["reasons"]:
            print(f"  - {reason}")
    print()
    print(f"production_representative: {str(report['production_representative']).lower()}")
    print(f"RESULT: {report['result'].upper()}")
    if "gpu_timing" in report:
        print_gpu_timing_report(report["gpu_timing"])


def main():
    parser = argparse.ArgumentParser(description="Compare a clean llama-bench run with an equivalent execution-traced run")
    parser.add_argument("--binary", type=Path, required=True, help="llama-bench executable")
    parser.add_argument("--report", type=Path, required=True, help="machine-readable JSON report")
    parser.add_argument("--trace", type=Path, help="raw trace path (default: beside the report)")
    parser.add_argument("--repetitions", type=int, default=5, help="llama-bench repetitions per process (default: 5)")
    parser.add_argument("--max-events", type=int, help="set GGML_EXEC_TRACE_MAX_EVENTS for the trace run")
    parser.add_argument("--launcher", default="", help="command prefix, such as a profiler, parsed with shell quoting")
    parser.add_argument("--gpu-timing", choices=("submit", "op"),
                        help="also run a safe trace with GGML_EXEC_TRACE_GPU_TIMING set to this mode")
    parser.add_argument("--gpu-timing-events", type=int, help="set GGML_EXEC_TRACE_GPU_TIMING_EVENTS for the timing runs")
    parser.add_argument("bench_args", nargs=argparse.REMAINDER, help="llama-bench arguments after --")
    args = parser.parse_args()

    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    bench_args = args.bench_args[1:] if args.bench_args[:1] == ["--"] else args.bench_args
    launcher = shlex.split(args.launcher)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.trace or args.report.with_suffix(".trace")
    trace_repeat_path = trace_path.with_name(trace_path.stem + ".repeat" + trace_path.suffix)
    baseline_integrity_path = args.report.with_suffix(".baseline.integrity")
    trace_integrity_path = args.report.with_suffix(".trace.integrity")
    trace_repeat_integrity_path = args.report.with_suffix(".trace.repeat.integrity")
    timing_path = trace_path.with_name(trace_path.stem + ".gpu-timing" + trace_path.suffix)
    timing_repeat_path = trace_path.with_name(trace_path.stem + ".gpu-timing.repeat" + trace_path.suffix)
    timing_integrity_path = args.report.with_suffix(".gpu-timing.integrity")
    timing_repeat_integrity_path = args.report.with_suffix(".gpu-timing.repeat.integrity")
    for path in (trace_path, trace_repeat_path, baseline_integrity_path, trace_integrity_path, trace_repeat_integrity_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    common_command = launcher + [str(binary)] + bench_args + ["-r", str(args.repetitions), "-o", "json", "-oe", "none"]
    baseline_env = os.environ.copy()
    baseline_env.pop("GGML_EXEC_TRACE", None)
    baseline_env.pop("GGML_EXEC_TRACE_INTEGRITY", None)
    for name in GPU_TIMING_ENV:
        baseline_env.pop(name, None)
    baseline_structure_env = baseline_env.copy()
    baseline_structure_env["GGML_EXEC_TRACE_INTEGRITY"] = str(baseline_integrity_path.resolve())
    trace_env = baseline_env.copy()
    trace_env["GGML_EXEC_TRACE"] = str(trace_path.resolve())
    trace_env["GGML_EXEC_TRACE_INTEGRITY"] = str(trace_integrity_path.resolve())
    trace_repeat_env = baseline_env.copy()
    trace_repeat_env["GGML_EXEC_TRACE"] = str(trace_repeat_path.resolve())
    trace_repeat_env["GGML_EXEC_TRACE_INTEGRITY"] = str(trace_repeat_integrity_path.resolve())
    if args.max_events is not None:
        trace_env["GGML_EXEC_TRACE_MAX_EVENTS"] = str(args.max_events)
        trace_repeat_env["GGML_EXEC_TRACE_MAX_EVENTS"] = str(args.max_events)
    timing_env = dict(trace_env, GGML_EXEC_TRACE=str(timing_path.resolve()),
                      GGML_EXEC_TRACE_INTEGRITY=str(timing_integrity_path.resolve()))
    if args.gpu_timing:
        timing_env["GGML_EXEC_TRACE_GPU_TIMING"] = args.gpu_timing
    if args.gpu_timing_events is not None:
        timing_env["GGML_EXEC_TRACE_GPU_TIMING_EVENTS"] = str(args.gpu_timing_events)
    timing_repeat_env = dict(timing_env, GGML_EXEC_TRACE=str(timing_repeat_path.resolve()),
                             GGML_EXEC_TRACE_INTEGRITY=str(timing_repeat_integrity_path.resolve()))

    try:
        # Symmetric order (clean, trace, timing, timing, trace, clean) keeps slow drift from favoring one configuration.
        gpu_timing = None
        baseline_structure_run = run_benchmark(common_command, baseline_structure_env)
        baseline_first = run_benchmark(common_command, baseline_env)
        trace_first = run_benchmark(common_command, trace_env)
        if args.gpu_timing:
            timing_first = run_benchmark(common_command, timing_env)
            timing_second = run_benchmark(common_command, timing_repeat_env)
        trace_second = run_benchmark(common_command, trace_repeat_env)
        baseline_second = run_benchmark(common_command, baseline_env)
        if args.gpu_timing:
            gpu_timing = {
                "mode": args.gpu_timing,
                "run": combine_benchmark_runs([timing_first, timing_second]),
                "integrity": read_integrity(timing_integrity_path),
                "integrity_repeat": read_integrity(timing_repeat_integrity_path),
                "stats": combine_trace_stats([read_trace_stats(timing_path), read_trace_stats(timing_repeat_path)]),
                "summaries": [TRACE.gpu_timing_summary(TRACE.read_trace(path)) for path in (timing_path, timing_repeat_path)],
            }
        baseline_run = combine_benchmark_runs([baseline_first, baseline_second])
        trace_run = combine_benchmark_runs([trace_first, trace_second])
        baseline_integrity = read_integrity(baseline_integrity_path)
        trace_integrity = read_integrity(trace_integrity_path)
        trace_integrity_repeat = read_integrity(trace_repeat_integrity_path)
        trace_stats = combine_trace_stats([read_trace_stats(trace_path), read_trace_stats(trace_repeat_path)])
        cache = read_cmake_cache(binary)
        linked = linked_profiler_libraries(binary)
        report = make_report(
            baseline_run, baseline_structure_run, trace_run,
            baseline_integrity, trace_integrity, trace_integrity_repeat,
            trace_stats, cache, linked, launcher, gpu_timing)
    except (OSError, IntegrityError, TRACE.TraceError) as error:
        print(f"exec-trace-integrity: {error}", file=sys.stderr)
        return 1

    report["artifacts"] = {
        "report": str(args.report.resolve()),
        "raw_trace": str(trace_path.resolve()),
        "raw_trace_repeat": str(trace_repeat_path.resolve()),
        "baseline_integrity": str(baseline_integrity_path.resolve()),
        "trace_integrity": str(trace_integrity_path.resolve()),
        "trace_integrity_repeat": str(trace_repeat_integrity_path.resolve()),
    }
    if args.gpu_timing:
        report["artifacts"].update({
            "gpu_timing_trace": str(timing_path.resolve()),
            "gpu_timing_trace_repeat": str(timing_repeat_path.resolve()),
            "gpu_timing_integrity": str(timing_integrity_path.resolve()),
            "gpu_timing_integrity_repeat": str(timing_repeat_integrity_path.resolve()),
        })
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print_report(report)
    timing_ok = "gpu_timing" not in report or report["gpu_timing"]["trustworthy"]
    return 0 if report["production_representative"] and timing_ok else 2


if __name__ == "__main__":
    sys.exit(main())
