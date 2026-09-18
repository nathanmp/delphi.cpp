#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from pathlib import Path


# ROCprofiler-SDK adapter. It converts rocprofv3 JSON into the provider-neutral delphi-gpu-activity-v1 document and
# merges it with exec-trace-gpu-activity.py. rocprofv3 tracing perturbs the run it observes, so all of its timing is
# VENDOR_EXTERNAL diagnostic data.
PROVIDER = "rocprofiler-sdk"
ROCPROFILER_GPU_AGENT_TYPE = 2
ROCPROFILER_CPU_AGENT_TYPE = 1
ROCPROFILER_CLOCK = "CLOCK_MONOTONIC"
GRAPH_LAUNCH_APIS = ("hipGraphLaunch", "hipGraphLaunch_spt")


def load_module(name, file_name):
    path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACTIVITY = load_module("exec_trace_gpu_activity", "exec-trace-gpu-activity.py")
TRACE = ACTIVITY.TRACE
MergeError = ACTIVITY.ActivityError


def parse_rocprofiler_json(data):
    try:
        text = data.decode("utf-8")
        replacements = 0
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        replacements = text.count("\ufffd")
    return json.loads(text), replacements


def handle(value):
    if isinstance(value, dict):
        value = value.get("handle", 0)
    return int(value or 0)


def correlation_ids(record):
    value = record.get("correlation_id", {})
    return {
        key: int(value.get(key, 0))
        for key in ("internal", "external", "ancestor")
        if key in value or key != "ancestor"
    }


def timestamp_pair(record, context):
    try:
        start_ns = int(record["start_timestamp"])
        end_ns = int(record["end_timestamp"])
    except (KeyError, TypeError, ValueError) as error:
        raise MergeError(f"{context} has invalid timestamps") from error
    if end_ns < start_ns:
        raise MergeError(f"{context} ends before it starts")
    return start_ns, end_ns


def uuid_string(agent):
    values = agent.get("uuid", {}).get("bytes", {})
    data = bytes(int(values.get(f"value{index}", 0)) for index in range(16))
    return data.hex() if any(data) else ""


def operation_tables(process):
    tables = {}
    for kind, entry in enumerate(process.get("strings", {}).get("buffer_records", [])):
        tables[kind] = entry.get("operations", [])
    return tables


def operation_name(tables, record):
    operations = tables.get(int(record.get("kind", -1)), [])
    operation = int(record.get("operation", -1))
    return operations[operation] if 0 <= operation < len(operations) else f"operation-{operation}"


def select_process(document, pid):
    processes = document.get("rocprofiler-sdk-tool")
    if not isinstance(processes, list) or not processes:
        raise MergeError("ROCprofiler JSON has no rocprofiler-sdk-tool process records")
    if pid is not None:
        matches = [process for process in processes if int(process.get("metadata", {}).get("pid", -1)) == pid]
        if len(matches) != 1:
            raise MergeError(f"ROCprofiler JSON does not contain exactly one process with PID {pid}")
        return matches[0]
    if len(processes) != 1:
        raise MergeError("ROCprofiler JSON contains multiple processes; select one with --pid")
    return processes[0]


def graph_fields(raw):
    return {
        "graph_exec_id": int(raw.get("graph_exec_id", 0)),
        "graph_node_id": int(raw.get("graph_node_id", 0)),
        "graph_identity_present": "graph_exec_id" in raw or "graph_node_id" in raw,
    }


def normalize_rocprofiler(process):
    """Returns a delphi-gpu-activity-v1 document for one rocprofv3 process record."""
    tables = operation_tables(process)
    devices = {}
    for raw in process.get("agents", []):
        agent_id = handle(raw.get("id"))
        if not agent_id:
            continue
        agent_type = int(raw.get("type", 0))
        devices[agent_id] = {
            "device_ref": agent_id,
            "type": "gpu" if agent_type == ROCPROFILER_GPU_AGENT_TYPE else "cpu" if agent_type == ROCPROFILER_CPU_AGENT_TYPE else "other",
            "gpu_index": int(raw.get("gpu_index", raw.get("logical_node_type_id", -1))),
            "node_id": int(raw.get("node_id", -1)),
            "logical_node_id": int(raw.get("logical_node_id", -1)),
            "name": raw.get("name", ""),
            "product_name": raw.get("product_name", ""),
            "uuid": uuid_string(raw),
        }

    symbols = {}
    for symbol in process.get("kernel_symbols", []):
        kernel_id = int(symbol.get("kernel_id", 0))
        names = [
            symbol.get("formatted_kernel_name", ""),
            symbol.get("demangled_kernel_name", ""),
            symbol.get("kernel_name", ""),
        ]
        symbols[kernel_id] = next((name for name in names if name and name != str(kernel_id)), f"kernel-{kernel_id}")

    records = process.get("buffer_records", {})
    api_calls = []
    for index, raw in enumerate(records.get("hip_api", []), 1):
        start_ns, end_ns = timestamp_pair(raw, f"HIP API record {index}")
        name = operation_name(tables, raw)
        api_calls.append({
            "id": index,
            "name": name,
            "api_domain": "HIP",
            "thread_id": int(raw.get("thread_id", 0)),
            "stream_id": handle(raw.get("stream_id")),
            "correlation_ids": correlation_ids(raw),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "graph_launch": name in GRAPH_LAUNCH_APIS,
        })

    activities = []
    for raw in records.get("kernel_dispatch", []):
        index = len(activities) + 1
        start_ns, end_ns = timestamp_pair(raw, f"kernel dispatch record {index}")
        info = raw.get("dispatch_info", {})
        agent_id = handle(info.get("agent_id"))
        kernel_id = int(info.get("kernel_id", 0))
        activities.append(dict({
            "id": index,
            "type": "kernel",
            "name": symbols.get(kernel_id, f"kernel-{kernel_id}"),
            "device_ref": agent_id,
            "device_refs": [agent_id] if agent_id else [],
            "stream_id": handle(raw.get("stream_id")),
            "queue_id": handle(info.get("queue_id")),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "correlation_ids": correlation_ids(raw),
            "dispatch_id": int(info.get("dispatch_id", 0)),
            "kernel_id": kernel_id,
        }, **graph_fields(raw)))

    for raw in records.get("memory_copy", []):
        index = len(activities) + 1
        start_ns, end_ns = timestamp_pair(raw, f"memory copy record {index}")
        src_agent_id = handle(raw.get("src_agent_id"))
        dst_agent_id = handle(raw.get("dst_agent_id"))
        gpu_agents = [
            agent_id for agent_id in (src_agent_id, dst_agent_id)
            if devices.get(agent_id, {}).get("type") == "gpu"
        ]
        direction = operation_name(tables, raw)
        activities.append(dict({
            "id": index,
            "type": "copy",
            "name": direction,
            "device_ref": gpu_agents[-1] if gpu_agents else 0,
            "device_refs": list(dict.fromkeys(gpu_agents)),
            "stream_id": handle(raw.get("stream_id")),
            "queue_id": handle(raw.get("queue_id")),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "correlation_ids": correlation_ids(raw),
            "bytes": int(raw.get("bytes", 0)),
            "copy_direction": direction,
            "src_device_ref": src_agent_id,
            "dst_device_ref": dst_agent_id,
        }, **graph_fields(raw)))

    activities.sort(key=lambda item: (item["start_ns"], item["end_ns"], item["id"]))
    for index, activity in enumerate(activities, 1):
        activity["id"] = index

    metadata = process.get("metadata", {})
    return {
        "schema": ACTIVITY.SCHEMA,
        "provider": {
            "name": PROVIDER,
            "provenance": "VENDOR_EXTERNAL",
            "production_representative": False,
            "clock": ROCPROFILER_CLOCK,
            # rocprofv3 kernel records carry no graph node identity. Report IDs only when a record has them.
            "graph_node_identity": "provider_reported" if any(item["graph_identity_present"] for item in activities) else "none",
            "process": {
                "pid": int(metadata.get("pid", 0)),
                "command": metadata.get("command", []),
                "init_time_ns": int(metadata.get("init_time", 0)),
                "fini_time_ns": int(metadata.get("fini_time", 0)),
            },
        },
        "devices": list(devices.values()),
        "api_calls": api_calls,
        "activities": activities,
    }


def merge(trace, process, requested_maps=None, manual_offset_ns=None,
          correlation_tolerance_ns=ACTIVITY.DEFAULT_CORRELATION_TOLERANCE_NS):
    return ACTIVITY.merge(trace, normalize_rocprofiler(process), requested_maps, manual_offset_ns, correlation_tolerance_ns)


def main():
    parser = argparse.ArgumentParser(description="Merge a DELPHI execution trace with ROCprofiler-SDK JSON activity")
    parser.add_argument("trace", type=Path, help="DELPHI raw execution trace")
    parser.add_argument("rocprofiler", type=Path, help="rocprofv3 JSON output")
    parser.add_argument("-o", "--output", type=Path, required=True, help=f"{ACTIVITY.MERGED_SCHEMA} JSON output")
    parser.add_argument("--activity-output", type=Path, help=f"also write the {ACTIVITY.SCHEMA} provider document")
    parser.add_argument("--pid", type=int, help="ROCprofiler process PID when the JSON contains multiple processes")
    parser.add_argument("--device-map", action="append", default=[], metavar="DEVICE=GPU",
                        help="map a DELPHI device ID to a ROCprofiler GPU index; may be repeated")
    parser.add_argument("--clock-offset-ns", type=int,
                        help="manual value added to ROCprofiler timestamps; for traces without CLOCK_SYNC")
    parser.add_argument("--correlation-tolerance-ns", type=int, default=ACTIVITY.DEFAULT_CORRELATION_TOLERANCE_NS,
                        help="extra clock-boundary tolerance for matching HIP APIs to host spans (default: 10000)")
    args = parser.parse_args()

    try:
        trace = TRACE.read_trace(args.trace)
        document, utf8_replacements = parse_rocprofiler_json(args.rocprofiler.read_bytes())
        activity = normalize_rocprofiler(select_process(document, args.pid))
        if args.correlation_tolerance_ns < 0:
            raise MergeError("--correlation-tolerance-ns must not be negative")
        result = ACTIVITY.merge(trace, activity, ACTIVITY.parse_device_maps(args.device_map), args.clock_offset_ns,
                                args.correlation_tolerance_ns)
        result["source"]["delphi_trace"] = str(args.trace)
        result["source"]["rocprofiler_json"] = str(args.rocprofiler)
        result["diagnostics"]["rocprofiler_utf8_replacements"] = utf8_replacements
        if args.activity_output:
            args.activity_output.write_text(json.dumps(activity, indent=2) + "\n")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        ACTIVITY.print_diagnostics(result, sys.stdout)
        if utf8_replacements:
            print(f"warning: replaced {utf8_replacements} invalid UTF-8 sequence(s) in ROCprofiler JSON strings")
    except (OSError, json.JSONDecodeError, MergeError, TRACE.TraceError) as error:
        print(f"exec-trace-rocm: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
