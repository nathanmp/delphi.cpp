#!/usr/bin/env python3

import argparse
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path


HEADER = struct.Struct("<8sIIIIQQQQ")
HEADER_V3_STATS = struct.Struct("<5Q")
EVENT = struct.Struct("<Q4Q10I")
STRING_HEADER = struct.Struct("<II")

K_RUN_METADATA = 1
K_EXECUTION_BEGIN = 2
K_EXECUTION_END = 3
K_UBATCH_BEGIN = 4
K_UBATCH_END = 5
K_BACKEND = 6
K_GRAPH = 7
K_NODE = 8
K_NODE_DIMS = 9
K_NODE_SOURCE = 10
K_NODE_SOURCE_DIMS = 11
K_NODE_VIEW = 12
K_NODE_MATMUL = 13
K_SCHED_SPLIT = 14
K_SCHED_SPLIT_INPUT = 15
K_FINGERPRINT = 16
K_BACKEND_SUBMIT = 17
K_CPU_COMPUTE = 18
K_BACKEND_WAIT = 19
K_COPY_HANDLING = 20
K_COPY_WAIT_SRC = 21
K_COPY_WAIT_DST = 22
K_COPY_BODY = 23
K_DISPATCH = 24
K_GPU_GRAPH = 25
K_CLOCK_SYNC = 26
K_GPU_TIMING = 27
K_GPU_TIMING_EPOCH = 28

DISPATCH_NAMES = {
    0: "unknown",
    1: "GPU_CUBLAS",
    2: "GPU_HIPBLAS",
    3: "GPU_MMVF",
    4: "GPU_MMVQ",
    5: "GPU_MMQ",
    6: "GPU_MMF",
    7: "GPU_FWHT",
    8: "GPU_MUL_MAT_ID_FALLBACK",
    9: "CPU_LLAMAFILE",
    10: "CPU_IQP",
    11: "CPU_VEC_DOT",
    12: "CPU_REPACK_GEMV",
    13: "CPU_REPACK_GEMM",
    14: "CPU_REPACK_GEMM_GEMV",
    15: "CPU_FWHT",
    16: "CPU_IQP_VEC_DOT",
    17: "CPU_IQK",
}
DISPATCH_VARIANTS = {
    0: "",
    1: "default",
    2: "bad_padding_or_type",
    3: "transposed",
    4: "fused",
    5: "sorted_fallback",
    6: "per_expert",
}
GPU_GRAPH_STATES = {1: "capture", 2: "update", 3: "replay", 4: "fallback"}
GPU_GRAPH_REASONS = {
    0: "none",
    1: "disabled",
    2: "incompatible_mul_mat_id",
    3: "warmup",
    4: "properties_changed",
    5: "update_reinstantiated",
}

EXECUTION_NAMES = {1: "decode", 2: "prefill", 3: "encode"}
SPAN_NAMES = {
    K_BACKEND_SUBMIT: "BACKEND_SUBMIT",
    K_CPU_COMPUTE: "CPU_COMPUTE",
    K_BACKEND_WAIT: "BACKEND_WAIT",
    K_COPY_HANDLING: "COPY_HANDLING",
    K_COPY_WAIT_SRC: "COPY_WAIT_SRC",
    K_COPY_WAIT_DST: "COPY_WAIT_DST",
    K_COPY_BODY: "COPY_BODY",
}
WAIT_REASONS = {
    1: "previous_split",
    2: "moe_ids",
    3: "eval_callback",
    4: "scheduler_synchronize",
    5: "graph_reallocation",
    6: "copy_dst_reuse",
}
WAIT_APIS = {1: "backend_synchronize", 2: "event_synchronize"}
COPY_PATHS = {
    1: "user_input_sync",
    2: "moe_expert_async",
    3: "backend_async",
    4: "fallback_sync",
}
WAIT_KINDS = {K_BACKEND_WAIT, K_COPY_WAIT_SRC, K_COPY_WAIT_DST}

# Timing provenance classes. Host spans are HOST_SAFE, backend timing events are DEVICE_EVENT, and offline profiler data is VENDOR_EXTERNAL.
TIMING_PROVENANCE = {1: "HOST_SAFE", 2: "DEVICE_EVENT", 3: "VENDOR_EXTERNAL"}
GPU_TIMING_SCOPES = {1: "eager_submit", 2: "graph_launch", 3: "eager_op"}
GPU_TIMING_STATUSES = {1: "resolved", 2: "unresolved", 3: "not_recorded", 4: "failed"}
GPU_TIMING_COMPLETIONS = {1: "synchronize", 2: "query", 3: "none"}


class TraceError(Exception):
    pass


def read_trace(path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise TraceError("file is shorter than the trace header")

    magic, version, header_size, event_size, flags, start_ns, event_count, string_count, string_bytes = HEADER.unpack_from(data)
    if magic.rstrip(b"\0") != b"GEXECTR":
        raise TraceError("invalid trace magic")
    if version not in (1, 2, 3, 4):
        raise TraceError(f"unsupported schema version {version}")
    if header_size < HEADER.size or event_size < EVENT.size:
        raise TraceError("invalid record sizes")

    event_end = header_size + event_count * event_size
    string_end = event_end + string_bytes
    if event_end > len(data) or string_end > len(data):
        raise TraceError("trace is incomplete")

    events = []
    for index in range(event_count):
        offset = header_size + index * event_size
        values = EVENT.unpack_from(data, offset)
        events.append({
            "timestamp_ns": values[0],
            "args": values[1:5],
            "execution_id": values[5],
            "ubatch_id": values[6],
            "graph_id": values[7],
            "node_id": values[8],
            "kind": values[9],
            "flags": values[10],
            "split_id": values[11],
            "backend_id": values[12],
            "device_id": values[13],
            "thread_id": values[14],
        })

    strings = {}
    offset = event_end
    for _ in range(string_count):
        if offset + STRING_HEADER.size > string_end:
            raise TraceError("incomplete string header")
        string_id, size = STRING_HEADER.unpack_from(data, offset)
        offset += STRING_HEADER.size
        if offset + size > string_end:
            raise TraceError("incomplete string data")
        if string_id == 0 or string_id in strings:
            raise TraceError(f"invalid or duplicate string ID {string_id}")
        try:
            strings[string_id] = data[offset:offset + size].decode("utf-8")
        except UnicodeDecodeError as error:
            raise TraceError(f"invalid UTF-8 in string {string_id}") from error
        offset += size
    if offset != string_end:
        raise TraceError("string table size does not match its records")

    trace_stats = {}
    if version >= 3:
        stats_offset = HEADER.size
        if header_size < stats_offset + HEADER_V3_STATS.size:
            raise TraceError("schema version 3 header has no trace statistics")
        capacity, attempted, dropped, growths, serialization_ns = HEADER_V3_STATS.unpack_from(data, stats_offset)
        trace_stats = {
            "event_capacity": capacity,
            "attempted_event_count": attempted,
            "dropped_event_count": dropped,
            "buffer_growth_count": growths,
            "serialization_duration_ns": serialization_ns,
        }

    return {
        "version": version,
        "flags": flags,
        "start_ns": start_ns,
        "events": events,
        "strings": strings,
        "trace_stats": trace_stats,
    }


def string(strings, string_id):
    return strings.get(string_id, "") if string_id else ""


def signed_u32(value):
    return value if value < 0x80000000 else value - 0x100000000


def dispatch_fields(event, strings):
    implementation_id = event["args"][0]
    implementation = string(strings, event["args"][1]) or DISPATCH_NAMES.get(
        implementation_id, f"implementation-{implementation_id}")
    variant_id = event["args"][2] & 0xffffffff
    variant_name_id = event["args"][2] >> 32
    variant = string(strings, variant_name_id) or DISPATCH_VARIANTS.get(variant_id, f"variant-{variant_id}")
    config_0 = event["args"][3] & 0xffffffff
    config_1 = event["args"][3] >> 32
    return implementation_id, implementation, variant_id, variant, config_0, config_1


def fnv1a_u64(hash_value, value):
    for byte in value.to_bytes(8, "little"):
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xffffffffffffffff
    return hash_value


def fnv1a_text(hash_value, value):
    data = value.encode("utf-8")
    hash_value = fnv1a_u64(hash_value, len(data))
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xffffffffffffffff
    return hash_value


def build_node_metadata(events, strings):
    graph_topologies = {}
    nodes = {}
    for event in events:
        if event["kind"] == K_GRAPH:
            graph_topologies[event["graph_id"]] = event["args"][0] or event["graph_id"]
        elif event["kind"] == K_NODE:
            nodes[(event["graph_id"], event["node_id"])] = {
                "index": event["args"][0],
                "name": string(strings, event["args"][1]),
                "op": string(strings, event["args"][2]) or str(event["flags"]),
                "op_id": event["flags"],
                "type": string(strings, event["args"][3]),
                "dims": (),
                "matmul": (),
                "sources": {},
            }
        elif event["kind"] == K_NODE_DIMS:
            node = nodes.get((event["graph_id"], event["node_id"]))
            if node is not None:
                node["dims"] = event["args"]
        elif event["kind"] == K_NODE_MATMUL:
            node = nodes.get((event["graph_id"], event["node_id"]))
            if node is not None:
                node["matmul"] = event["args"]
        elif event["kind"] == K_NODE_SOURCE:
            node = nodes.get((event["graph_id"], event["node_id"]))
            if node is not None:
                source_index = event["args"][0]
                node["sources"][source_index] = {
                    "type_id": event["args"][2],
                    "type": string(strings, event["flags"]) or f"type-{event['args'][2]}",
                    "name": string(strings, event["args"][3]),
                    "dims": (),
                }
        elif event["kind"] == K_NODE_SOURCE_DIMS:
            node = nodes.get((event["graph_id"], event["node_id"]))
            source = node["sources"].get(event["flags"]) if node is not None else None
            if source is not None:
                source["dims"] = event["args"]

    def node_metadata(event):
        topology_id = graph_topologies.get(event["graph_id"], event["graph_id"])
        return nodes.get((topology_id, event["node_id"]))

    return node_metadata


def dispatch_fingerprints(events, strings):
    node_metadata = build_node_metadata(events, strings)
    hashes = {}
    for event in events:
        if event["kind"] != K_DISPATCH:
            continue
        hash_value = hashes.setdefault(event["graph_id"], 14695981039346656037)
        node = node_metadata(event) or {}
        implementation_id, implementation, variant_id, variant, config_0, config_1 = dispatch_fields(event, strings)
        for value in (node.get("index", 0), node.get("op_id", event["flags"]), *node.get("dims", ())):
            hash_value = fnv1a_u64(hash_value, value)
        for value in node.get("matmul", ()):
            hash_value = fnv1a_u64(hash_value, value)
        hash_value = fnv1a_text(hash_value, node.get("name", ""))
        hash_value = fnv1a_text(hash_value, node.get("type", ""))
        for source_index, source in sorted(node.get("sources", {}).items()):
            hash_value = fnv1a_u64(hash_value, source_index)
            hash_value = fnv1a_u64(hash_value, source["type_id"])
            hash_value = fnv1a_text(hash_value, source["type"])
            hash_value = fnv1a_text(hash_value, source["name"])
            for value in source["dims"]:
                hash_value = fnv1a_u64(hash_value, value)
        hash_value = fnv1a_u64(hash_value, implementation_id)
        hash_value = fnv1a_text(hash_value, implementation)
        hash_value = fnv1a_u64(hash_value, variant_id)
        hash_value = fnv1a_text(hash_value, variant)
        hash_value = fnv1a_u64(hash_value, config_0)
        hash_value = fnv1a_u64(hash_value, config_1)
        hashes[event["graph_id"]] = hash_value
    return hashes


def dispatch_config(implementation, variant, config_0, config_1):
    if implementation == "GPU_MMQ":
        j = config_1 & 0xff
        tile_i = (config_1 >> 8) & 0xff
        threads = config_1 >> 16
        return f"arch=0x{config_0:08x} tile={tile_i}x{j} threads={threads}"
    if implementation.startswith("GPU_"):
        return f"arch=0x{config_0:08x} choice={config_1}"
    if implementation.startswith("CPU_REPACK_"):
        return f"threads={config_0} interleave={config_1 >> 16} cols={config_1 & 0xffff}"
    if variant == "per_expert":
        return f"threads={config_0} active_experts={config_1 & 0xffff} iqp_experts={config_1 >> 16}"
    if implementation == "CPU_VEC_DOT":
        return f"threads={config_0} chunks={config_1 & 0xffff}x{config_1 >> 16}"
    if implementation.startswith("CPU_"):
        return f"threads={config_0} choice={config_1}"
    return f"{config_0},{config_1}"


def dump_dispatches(events, strings, backend_name):
    node_metadata = build_node_metadata(events, strings)
    dispatches = [event for event in events if event["kind"] == K_DISPATCH]
    print(f"dispatch_count: {len(dispatches)}")
    print("dispatches:")
    for event in dispatches:
        node = node_metadata(event) or {}
        _, implementation, _, variant, config_0, config_1 = dispatch_fields(event, strings)
        print(f"  graph {event['graph_id']} node {event['node_id']} {node.get('op', event['flags'])} {node.get('name', '')}".rstrip())
        print(f"    backend: {backend_name(event)}")
        if node.get("matmul"):
            m, n, k, selected = node["matmul"]
            shape = f"M={m} N={n} K={k}"
            if selected:
                shape += f" selected_experts={selected}"
            print(f"    shape: {shape}")
        if node.get("sources"):
            inputs = " ".join(
                f"src{source_index}={source['type']}" for source_index, source in sorted(node["sources"].items()))
            print(f"    inputs: {inputs}")
        print(f"    dispatch: {implementation}")
        if variant:
            print(f"    variant: {variant}")
        print(f"    config: {dispatch_config(implementation, variant, config_0, config_1)}")


def dump_gpu_graphs(events, backend_name):
    lifecycle = [event for event in events if event["kind"] == K_GPU_GRAPH]
    groups = defaultdict(lambda: {"counts": Counter(), "reasons": Counter(), "nodes": 0, "event": None})
    for event in lifecycle:
        key = (event["args"][0], backend_name(event))
        group = groups[key]
        state = GPU_GRAPH_STATES.get(event["flags"], f"state-{event['flags']}")
        reason = GPU_GRAPH_REASONS.get(event["args"][2], f"reason-{event['args'][2]}")
        group["counts"][state] += 1
        if reason != "none":
            group["reasons"][reason] += 1
        group["nodes"] = event["args"][1]
        group["event"] = event

    print(f"gpu_graph_event_count: {len(lifecycle)}")
    print("gpu_graphs:")
    for (gpu_graph_id, name), group in sorted(groups.items()):
        identity = f"graph_exec {gpu_graph_id}" if gpu_graph_id else "uncaptured"
        counts = group["counts"]
        print(f"  {identity} / {name}: nodes={group['nodes']} capture={counts['capture']} update={counts['update']} replay={counts['replay']} fallback={counts['fallback']}")
        if group["reasons"]:
            reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(group["reasons"].items()))
            print(f"    reasons: {reasons}")


def span_duration_ns(event):
    end_ns = event["args"][0]
    if end_ns < event["timestamp_ns"]:
        raise TraceError(f"{SPAN_NAMES.get(event['kind'], 'span')} ends before it starts")
    return end_ns - event["timestamp_ns"]


def build_backend_names(events, strings):
    graph_topologies = {}
    names = {}
    fallback_names = {}
    for event in events:
        if event["kind"] == K_GRAPH:
            graph_topologies[event["graph_id"]] = event["args"][0] or event["graph_id"]
        elif event["kind"] == K_BACKEND:
            name = string(strings, event["args"][0]) or string(strings, event["args"][2])
            names[(event["graph_id"], event["backend_id"])] = name
            fallback_names[event["backend_id"]] = name

    def backend_name(event, backend_id=None):
        backend_id = event["backend_id"] if backend_id is None else backend_id
        if not backend_id:
            return "unknown"
        topology_id = graph_topologies.get(event["graph_id"], event["graph_id"])
        return names.get((topology_id, backend_id)) or fallback_names.get(backend_id) or f"backend-{backend_id}"

    return backend_name


def dump_scheduler_summary(events, backend_name):
    cpu_compute = [0, 0]
    copy_handling = [0, 0]
    backend_submit = defaultdict(lambda: [0, 0])
    backend_wait = defaultdict(lambda: [0, 0])

    for event in events:
        kind = event["kind"]
        if kind not in SPAN_NAMES:
            continue
        duration_ns = span_duration_ns(event)
        if kind == K_CPU_COMPUTE:
            cpu_compute[0] += 1
            cpu_compute[1] += duration_ns
        elif kind == K_BACKEND_SUBMIT:
            stat = backend_submit[backend_name(event)]
            stat[0] += 1
            stat[1] += duration_ns
        elif kind in WAIT_KINDS:
            stat = backend_wait[backend_name(event)]
            stat[0] += 1
            stat[1] += duration_ns
        elif kind == K_COPY_HANDLING:
            copy_handling[0] += 1
            copy_handling[1] += duration_ns

    print("scheduler_summary:")
    print("  provenance: HOST_SAFE (host clock around host calls)")
    print(f"  cpu_compute: count={cpu_compute[0]} host_ms={cpu_compute[1] / 1e6:.3f}")
    for name, stat in sorted(backend_submit.items()):
        print(f"  backend_submit[{name}]: count={stat[0]} host_ms={stat[1] / 1e6:.3f}")
    for name, stat in sorted(backend_wait.items()):
        print(f"  backend_wait[{name}]: count={stat[0]} host_ms={stat[1] / 1e6:.3f}")
    print(f"  copy_handling: count={copy_handling[0]} host_ms={copy_handling[1] / 1e6:.3f}")
    print("  note: nested spans and asynchronous submissions are not additive latency or device execution time")


def timeline_detail(event, backend_name):
    kind = event["kind"]
    args = event["args"]
    if kind in (K_BACKEND_SUBMIT, K_CPU_COMPUTE):
        return f"status={signed_u32(args[1])} nodes=[{args[2]},{args[3]})"
    if kind == K_BACKEND_WAIT:
        reason = WAIT_REASONS.get(event["flags"], str(event["flags"]))
        wait_api = WAIT_APIS.get(args[1], str(args[1]))
        return f"reason={reason} api={wait_api}"
    if kind in (K_COPY_HANDLING, K_COPY_BODY):
        source_backend_id = args[3] & 0xffffffff
        source = backend_name(event, source_backend_id)
        path = COPY_PATHS.get(event["flags"], str(event["flags"]))
        return f"src={source} dst_node={args[1]} bytes={args[2]} path={path}"
    if kind in (K_COPY_WAIT_SRC, K_COPY_WAIT_DST):
        other_backend_id = args[2] & 0xffffffff
        other = backend_name(event, other_backend_id)
        wait_api = WAIT_APIS.get(args[1], str(args[1]))
        path = COPY_PATHS.get(event["flags"], str(event["flags"]))
        return f"other={other} bytes={args[3]} path={path} api={wait_api}"
    return ""


def gpu_timing_fields(event):
    flags = event["flags"]
    scope_id = flags & 0xff
    status_id = (flags >> 8) & 0xff
    provenance_id = (flags >> 16) & 0xff
    return {
        "scope": GPU_TIMING_SCOPES.get(scope_id, f"scope-{scope_id}"),
        "status": GPU_TIMING_STATUSES.get(status_id, f"status-{status_id}"),
        "provenance": TIMING_PROVENANCE.get(provenance_id, f"provenance-{provenance_id}"),
        "host_enqueue_start_ns": event["timestamp_ns"],
        "host_enqueue_end_ns": event["args"][0],
        "device_offset_ns": event["args"][1],
        "device_duration_ns": event["args"][2],
        "epoch_id": event["args"][3] >> 32,
        "detail": event["args"][3] & 0xffffffff,
    }


def gpu_timing_model(events):
    spans = []
    epochs = {}
    for event in events:
        if event["kind"] == K_GPU_TIMING:
            span = gpu_timing_fields(event)
            span["event"] = event
            spans.append(span)
        elif event["kind"] == K_GPU_TIMING_EPOCH:
            epochs[event["args"][1]] = {
                "epoch_id": event["args"][1],
                "anchor_enqueue_ns": event["timestamp_ns"],
                "complete_ns": event["args"][0],
                "span_count": event["args"][2],
                "completion": GPU_TIMING_COMPLETIONS.get(event["args"][3], f"completion-{event['args'][3]}"),
                "backend_id": event["backend_id"],
                "device_id": event["device_id"],
            }

    members = defaultdict(list)
    for span in spans:
        if span["status"] == "resolved":
            members[span["epoch_id"]].append(span)

    # All spans of an epoch share one device anchor A. The host reads its clock before each event record, and the device
    # reaches the event after that and before completion is known, so lower <= A <= upper. The lower bound is the placement.
    for epoch_id, epoch in epochs.items():
        epoch_spans = members.get(epoch_id, [])
        epoch["resolved_span_count"] = len(epoch_spans)
        if not epoch_spans:
            epoch["placement"] = "none"
            continue
        lower = max(max(span["host_enqueue_start_ns"] - span["device_offset_ns"],
                        span["host_enqueue_end_ns"] - span["device_offset_ns"] - span["device_duration_ns"])
                    for span in epoch_spans)
        upper = min(epoch["complete_ns"] - span["device_offset_ns"] - span["device_duration_ns"] for span in epoch_spans)
        epoch["anchor_lower_ns"] = lower
        epoch["anchor_upper_ns"] = upper
        epoch["placement"] = "consistent" if lower <= upper else "inconsistent"
        epoch["placement_uncertainty_ns"] = upper - lower if lower <= upper else None
        for span in epoch_spans:
            span["device_start_ns"] = lower + span["device_offset_ns"]
            span["device_end_ns"] = span["device_start_ns"] + span["device_duration_ns"]
            span["placement"] = epoch["placement"]
            span["placement_uncertainty_ns"] = epoch["placement_uncertainty_ns"]
    for span in spans:
        if span["epoch_id"] not in epochs and span["status"] == "resolved":
            span["placement"] = "missing_epoch"
    return spans, epochs


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def duration_stats(durations):
    return {
        "count": len(durations),
        "device_sum_ns": sum(durations),
        "device_mean_ns": sum(durations) / len(durations) if durations else None,
        "device_median_ns": median(durations),
        "device_min_ns": min(durations) if durations else None,
        "device_max_ns": max(durations) if durations else None,
    }


def gpu_timing_summary(trace):
    events = trace["events"]
    strings = trace["strings"]
    metadata = {}
    for event in events:
        if event["kind"] == K_RUN_METADATA:
            metadata[string(strings, event["args"][0])] = string(strings, event["args"][1])
    spans, epochs = gpu_timing_model(events)
    backend_name = build_backend_names(events, strings)
    node_metadata = build_node_metadata(events, strings)
    implementations = {}
    for event in events:
        if event["kind"] == K_DISPATCH:
            implementations[(event["graph_id"], event["node_id"])] = dispatch_fields(event, strings)[1]

    def int_metadata(key):
        try:
            return int(metadata[key])
        except (KeyError, ValueError):
            return None

    started = int_metadata("gpu_timing_spans_started")
    released = int_metadata("gpu_timing_spans_released")
    status_counts = Counter(span["status"] for span in spans)
    completion_counts = Counter(epoch["completion"] for epoch in epochs.values())
    placement_counts = Counter(epoch["placement"] for epoch in epochs.values())
    uncertainties = [epoch["placement_uncertainty_ns"] for epoch in epochs.values()
                     if epoch.get("placement_uncertainty_ns") is not None]

    resolved = [span for span in spans if span["status"] == "resolved"]
    by_scope = defaultdict(list)
    by_graph = defaultdict(list)
    by_op = defaultdict(list)
    for span in resolved:
        event = span["event"]
        by_scope[(span["scope"], backend_name(event))].append(span["device_duration_ns"])
        if span["scope"] == "graph_launch":
            by_graph[(span["detail"], backend_name(event))].append(span["device_duration_ns"])
        elif span["scope"] == "eager_op":
            node = node_metadata(event) or {}
            op = node.get("op", "unknown")
            if span["detail"] > 1:
                op += f" (fused group of {span['detail']})"
            implementation = implementations.get((event["graph_id"], event["node_id"]), "")
            by_op[(op, implementation, backend_name(event))].append(span["device_duration_ns"])

    return {
        "mode": metadata.get("gpu_timing", "off"),
        "provenance": "DEVICE_EVENT",
        "event_capacity": int_metadata("gpu_timing_event_capacity"),
        "spans_started": started,
        "spans_released": released,
        "spans_unreleased": started - released if started is not None and released is not None else None,
        "spans_recorded": len(spans),
        "status_counts": {name: status_counts.get(name, 0) for name in GPU_TIMING_STATUSES.values()},
        "epoch_count": len(epochs),
        "epoch_completion_counts": dict(sorted(completion_counts.items())),
        "epoch_placement_counts": dict(sorted(placement_counts.items())),
        "placement_uncertainty_median_ns": median(uncertainties),
        "placement_uncertainty_max_ns": max(uncertainties) if uncertainties else None,
        "by_scope": [dict(duration_stats(values), scope=scope, backend=name)
                     for (scope, name), values in sorted(by_scope.items())],
        "by_gpu_graph": [dict(duration_stats(values), gpu_graph_id=graph_id, backend=name)
                         for (graph_id, name), values in sorted(by_graph.items())],
        "by_op": sorted((dict(duration_stats(values), op=op, implementation=implementation, backend=name)
                         for (op, implementation, name), values in by_op.items()),
                        key=lambda item: (-item["device_sum_ns"], item["op"], item["implementation"])),
    }


def microseconds(value):
    return "n/a" if value is None else f"{value / 1e3:.1f}"


def dump_gpu_timing(summary, op_limit=20):
    print("gpu_timing:")
    print(f"  mode: {summary['mode']}")
    print("  provenance: DEVICE_EVENT (timing events on the backend stream; intervals include gaps between kernels in a span)")
    print(f"  spans: started={summary['spans_started']} released={summary['spans_released']} "
          f"unreleased={summary['spans_unreleased']} recorded={summary['spans_recorded']}")
    print("  status: " + " ".join(f"{name}={count}" for name, count in summary["status_counts"].items()))
    completions = " ".join(f"{name}={count}" for name, count in summary["epoch_completion_counts"].items())
    print(f"  epochs: {summary['epoch_count']} {completions}".rstrip())
    placements = " ".join(f"{name}={count}" for name, count in summary["epoch_placement_counts"].items())
    print(f"  host_placement: {placements} uncertainty_us median={microseconds(summary['placement_uncertainty_median_ns'])} "
          f"max={microseconds(summary['placement_uncertainty_max_ns'])}")
    for item in summary["by_scope"]:
        print(f"  scope[{item['scope']}/{item['backend']}]: count={item['count']} device_ms={item['device_sum_ns'] / 1e6:.3f} "
              f"median_us={microseconds(item['device_median_ns'])}")
    for item in summary["by_gpu_graph"]:
        print(f"  graph_launch[graph_exec {item['gpu_graph_id']}/{item['backend']}]: count={item['count']} "
              f"device_ms={item['device_sum_ns'] / 1e6:.3f} median_us={microseconds(item['device_median_ns'])} "
              f"min_us={microseconds(item['device_min_ns'])} max_us={microseconds(item['device_max_ns'])}")
    if summary["by_op"]:
        print(f"  eager_op (top {min(op_limit, len(summary['by_op']))} of {len(summary['by_op'])} by device time):")
        for item in summary["by_op"][:op_limit]:
            implementation = f" dispatch={item['implementation']}" if item["implementation"] else ""
            print(f"    {item['op']}{implementation} [{item['backend']}]: count={item['count']} "
                  f"device_ms={item['device_sum_ns'] / 1e6:.3f} median_us={microseconds(item['device_median_ns'])}")


def dump_timeline(trace, backend_name):
    endpoints = []
    for index, event in enumerate(trace["events"]):
        if event["kind"] not in SPAN_NAMES:
            continue
        span_duration_ns(event)
        endpoints.append((event["timestamp_ns"], 0, index, "begin", event))
        endpoints.append((event["args"][0], 1, index, "end", event))
    spans, _ = gpu_timing_model(trace["events"])
    device_spans = {}
    for span in spans:
        if "device_start_ns" in span:
            index = len(trace["events"]) + len(device_spans)
            device_spans[index] = span
            endpoints.append((span["device_start_ns"], 0, index, "begin", span["event"]))
            endpoints.append((span["device_end_ns"], 1, index, "end", span["event"]))

    print("timeline: (host spans are HOST_SAFE; GPU_TIMING spans are DEVICE_EVENT intervals placed on the host clock)")
    for timestamp_ns, _, index, phase, event in sorted(endpoints):
        fields = []
        if event["execution_id"]:
            fields.append(f"execution={event['execution_id']}")
        if event["ubatch_id"]:
            fields.append(f"ubatch={event['ubatch_id']}")
        if event["graph_id"]:
            fields.append(f"graph={event['graph_id']}")
        if event["split_id"]:
            fields.append(f"split={event['split_id']}")
        fields.append(backend_name(event))
        span = device_spans.get(index)
        if span:
            fields.extend(("GPU_TIMING", phase, f"scope={span['scope']}", f"placement={span['placement']}",
                           f"uncertainty_us={microseconds(span['placement_uncertainty_ns'])}"))
            offset_ms = (timestamp_ns - trace["start_ns"]) / 1e6
            print(f"  {offset_ms:12.3f} ms  {' '.join(fields)}")
            continue
        fields.append(SPAN_NAMES[event["kind"]])
        fields.append(phase)
        detail = timeline_detail(event, backend_name)
        if detail:
            fields.append(detail)
        offset_ms = (timestamp_ns - trace["start_ns"]) / 1e6
        print(f"  {offset_ms:12.3f} ms  {' '.join(fields)}")


def dump(trace, scheduler_summary=False, timeline=False, dispatches=False, gpu_timing=False):
    events = trace["events"]
    strings = trace["strings"]
    backend_name = build_backend_names(events, strings)
    metadata = {}
    backends = {}
    assignments = Counter()
    executions = []
    execution_ends = {}
    ubatches = []
    ubatch_ends = {}
    graphs = []
    fingerprints = []

    for event in events:
        args = event["args"]
        kind = event["kind"]
        if kind == K_RUN_METADATA:
            metadata[string(strings, args[0])] = string(strings, args[1])
        elif kind == K_EXECUTION_BEGIN:
            executions.append(event)
        elif kind == K_EXECUTION_END:
            execution_ends[event["execution_id"]] = event
        elif kind == K_UBATCH_BEGIN:
            ubatches.append(event)
        elif kind == K_UBATCH_END:
            ubatch_ends[event["ubatch_id"]] = event
        elif kind == K_BACKEND:
            backends[event["backend_id"]] = string(strings, args[0]) or string(strings, args[2])
        elif kind == K_GRAPH:
            graphs.append(event)
        elif kind == K_NODE:
            assignments[event["backend_id"]] += 1
        elif kind == K_FINGERPRINT:
            fingerprints.append(event)

    print(f"schema_version: {trace['version']}")
    print(f"truncated: {'yes' if trace['flags'] & 1 else 'no'}")
    print(f"event_count: {len(events)}")
    if trace["trace_stats"]:
        stats = trace["trace_stats"]
        print("trace_overhead:")
        print(f"  attempted_events: {stats['attempted_event_count']}")
        print(f"  dropped_events: {stats['dropped_event_count']}")
        print(f"  interned_strings: {len(strings)}")
        print(f"  event_buffer_bytes: {stats['event_capacity'] * EVENT.size}")
        print(f"  buffer_growths: {stats['buffer_growth_count']}")
        print(f"  serialization_ms: {stats['serialization_duration_ns'] / 1e6:.3f}")
    print("run_metadata:")
    for key in sorted(metadata):
        print(f"  {key}: {metadata[key]}")

    print(f"executions: {len(executions)}")
    for event in executions:
        args = event["args"]
        n_threads = args[3] & 0xffffffff
        n_threads_batch = args[3] >> 32
        end = execution_ends.get(event["execution_id"])
        status = signed_u32(end["args"][0]) if end else "missing"
        host_ms = f"{(end['timestamp_ns'] - event['timestamp_ns']) / 1e6:.3f}" if end else "missing"
        print(
            f"  execution {event['execution_id']}: "
            f"kind={EXECUTION_NAMES.get(event['flags'], event['flags'])} "
            f"tokens={args[0]} batch={args[1]} ubatch={args[2]} "
            f"threads={n_threads}/{n_threads_batch} status={status} host_ms={host_ms}"
        )

    print(f"ubatches: {len(ubatches)}")
    for event in ubatches:
        args = event["args"]
        end = ubatch_ends.get(event["ubatch_id"])
        status = signed_u32(end["args"][0]) if end else "missing"
        host_ms = f"{(end['timestamp_ns'] - event['timestamp_ns']) / 1e6:.3f}" if end else "missing"
        print(
            f"  ubatch {event['ubatch_id']}: execution={event['execution_id']} "
            f"tokens={args[0]} sequences={args[1]} unique_sequences={args[2]} "
            f"outputs={args[3]} status={status} host_ms={host_ms}"
        )

    topology_graphs = [event for event in graphs if not event["flags"] & 1]
    print(f"graph_count: {len(graphs)}")
    print(f"topology_count: {len(topology_graphs)}")
    print(f"node_count: {sum(event['args'][1] for event in topology_graphs)}")
    print(f"scheduler_split_count: {sum(event['args'][3] for event in topology_graphs)}")
    print("backend_assignments:")
    for backend_id, count in sorted(assignments.items()):
        print(f"  {backend_id} ({backends.get(backend_id, 'unknown')}): {count} nodes")

    print("structural_fingerprints:")
    for event in fingerprints:
        args = event["args"]
        print(f"  graph {event['graph_id']}: ops_shapes={args[0]:016x} assignments={args[1]:016x} split_sequence={args[2]:016x} cross_backend_edges={args[3]}")

    print("dispatch_fingerprints:")
    for graph_id, fingerprint in sorted(dispatch_fingerprints(events, strings).items()):
        print(f"  graph {graph_id}: {fingerprint:016x}")

    if dispatches:
        dump_dispatches(events, strings, backend_name)
    else:
        print(f"dispatch_count: {sum(event['kind'] == K_DISPATCH for event in events)}")
    dump_gpu_graphs(events, backend_name)

    gpu_timing_count = sum(event["kind"] == K_GPU_TIMING for event in events)
    if gpu_timing:
        dump_gpu_timing(gpu_timing_summary(trace))
    elif gpu_timing_count:
        print(f"gpu_timing_span_count: {gpu_timing_count}")

    if scheduler_summary:
        dump_scheduler_summary(events, backend_name)
    if timeline:
        dump_timeline(trace, backend_name)


def main():
    parser = argparse.ArgumentParser(description="Print a summary of a GGML execution trace")
    parser.add_argument("--dispatches", action="store_true", help="print every backend dispatch decision")
    parser.add_argument("--scheduler-summary", action="store_true", help="aggregate host scheduler spans")
    parser.add_argument("--timeline", action="store_true", help="print chronological host span endpoints")
    parser.add_argument("--gpu-timing", action="store_true", help="summarize DEVICE_EVENT GPU timing spans")
    parser.add_argument("--gpu-timing-json", type=Path, help="write the GPU timing summary as JSON")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    try:
        trace = read_trace(args.trace)
        dump(trace, scheduler_summary=args.scheduler_summary, timeline=args.timeline, dispatches=args.dispatches,
             gpu_timing=args.gpu_timing)
        if args.gpu_timing_json:
            args.gpu_timing_json.write_text(json.dumps(gpu_timing_summary(trace), indent=2) + "\n")
    except (OSError, TraceError) as error:
        print(f"exec-trace-dump: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
