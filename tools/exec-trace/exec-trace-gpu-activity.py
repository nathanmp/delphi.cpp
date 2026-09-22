#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path


# Provider-neutral GPU activity. Offline importers (ROCprofiler-SDK now, CUPTI or Nsight Systems later) write this
# document, and merge() aligns it with a DELPHI raw trace. Provider timing is VENDOR_EXTERNAL: the provider changes the
# run it observes, so its intervals are diagnostic and never production-representative.
SCHEMA = "delphi-gpu-activity-v1"
MERGED_SCHEMA = "delphi-gpu-activity-merged-v1"
LEGACY_MERGED_SCHEMAS = ("delphi-rocm-activity-v1",)

PROVENANCE = ("HOST_SAFE", "DEVICE_EVENT", "VENDOR_EXTERNAL")
DEVICE_TYPES = ("gpu", "cpu", "other")
ACTIVITY_TYPES = ("kernel", "copy", "memset")
# none: the provider reports no graph node identity.
# provider_reported: activities carry provider graph IDs; DELPHI does not map them to GGML nodes.
GRAPH_NODE_IDENTITY = ("none", "provider_reported")
CLOCK_SYNC_KINDS = {1: "CLOCK_MONOTONIC"}
DEFAULT_CORRELATION_TOLERANCE_NS = 10_000
GPU_DEVICE_TYPE = 1


class ActivityError(Exception):
    pass


def load_trace_reader():
    path = Path(__file__).with_name("exec-trace-dump.py")
    spec = importlib.util.spec_from_file_location("exec_trace_dump", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACE = load_trace_reader()


def require(condition, message):
    if not condition:
        raise ActivityError(message)


def validate_activity_document(document):
    require(isinstance(document, dict) and document.get("schema") == SCHEMA, f"GPU activity input is not a {SCHEMA} document")
    provider = document.get("provider")
    require(isinstance(provider, dict) and provider.get("name"), "GPU activity document has no provider name")
    require(provider.get("provenance") in PROVENANCE, f"invalid provider provenance {provider.get('provenance')!r}")
    if provider["provenance"] == "VENDOR_EXTERNAL":
        require(provider.get("production_representative") is False,
                "VENDOR_EXTERNAL activity must set production_representative to false")
    require(provider.get("graph_node_identity") in GRAPH_NODE_IDENTITY,
            f"invalid graph_node_identity {provider.get('graph_node_identity')!r}")
    require(isinstance(provider.get("clock"), str) and provider["clock"], "GPU activity document has no provider clock")

    device_refs = set()
    for device in document.get("devices", []):
        require(device.get("device_ref") and device["device_ref"] not in device_refs, "invalid or duplicate device_ref")
        require(device.get("type") in DEVICE_TYPES, f"device {device['device_ref']} has an invalid type")
        device_refs.add(device["device_ref"])

    def check_interval(item, context):
        try:
            start_ns = int(item["start_ns"])
            end_ns = int(item["end_ns"])
        except (KeyError, TypeError, ValueError) as error:
            raise ActivityError(f"{context} has invalid timestamps") from error
        require(end_ns >= start_ns, f"{context} ends before it starts")

    api_ids = set()
    for api in document.get("api_calls", []):
        require(api.get("id") and api["id"] not in api_ids, "invalid or duplicate API call ID")
        check_interval(api, f"API call {api['id']}")
        require(isinstance(api.get("graph_launch"), bool), f"API call {api['id']} has no graph_launch flag")
        api_ids.add(api["id"])

    activity_ids = set()
    for activity in document.get("activities", []):
        require(activity.get("id") and activity["id"] not in activity_ids, "invalid or duplicate activity ID")
        context = f"activity {activity['id']}"
        require(activity.get("type") in ACTIVITY_TYPES, f"{context} has an invalid type")
        check_interval(activity, context)
        for device_ref in activity.get("device_refs", []):
            require(device_ref in device_refs, f"{context} references unknown device {device_ref}")
        if activity.get("graph_identity_present"):
            require(provider["graph_node_identity"] != "none", f"{context} has graph IDs, but the provider declares none")
        activity_ids.add(activity["id"])
    return document


def trace_metadata(trace):
    result = {}
    for event in trace["events"]:
        if event["kind"] == TRACE.K_RUN_METADATA:
            result[TRACE.string(trace["strings"], event["args"][0])] = TRACE.string(trace["strings"], event["args"][1])
    return result


def trace_model(trace):
    strings = trace["strings"]
    node_metadata = TRACE.build_node_metadata(trace["events"], strings)
    devices = {}
    dispatches = []
    submissions = []
    copies = []
    waits = []
    graph_events = []
    execution_begins = {}
    executions = []

    for event in trace["events"]:
        kind = event["kind"]
        args = event["args"]
        if kind == TRACE.K_BACKEND and event["flags"] == GPU_DEVICE_TYPE:
            device = devices.setdefault(event["device_id"], {
                "device_id": event["device_id"],
                "backend_ids": [],
                "backend_names": [],
                "device_name": TRACE.string(strings, args[2]),
                "description": TRACE.string(strings, args[3]),
            })
            if event["backend_id"] not in device["backend_ids"]:
                device["backend_ids"].append(event["backend_id"])
            name = TRACE.string(strings, args[0])
            if name and name not in device["backend_names"]:
                device["backend_names"].append(name)
        elif kind == TRACE.K_DISPATCH:
            node = node_metadata(event) or {}
            _, implementation, _, variant, config_0, config_1 = TRACE.dispatch_fields(event, strings)
            dispatches.append({
                "id": len(dispatches) + 1,
                "timestamp_ns": event["timestamp_ns"],
                "execution_id": event["execution_id"],
                "ubatch_id": event["ubatch_id"],
                "graph_id": event["graph_id"],
                "split_id": event["split_id"],
                "backend_id": event["backend_id"],
                "device_id": event["device_id"],
                "node_id": event["node_id"],
                "node_index": int(node.get("index", 0)),
                "node_name": node.get("name", ""),
                "op": node.get("op", str(event["flags"])),
                "implementation": implementation,
                "variant": variant,
                "config_0": config_0,
                "config_1": config_1,
            })
        elif kind == TRACE.K_BACKEND_SUBMIT:
            submissions.append({
                "id": len(submissions) + 1,
                "start_ns": event["timestamp_ns"],
                "end_ns": args[0],
                "execution_id": event["execution_id"],
                "ubatch_id": event["ubatch_id"],
                "graph_id": event["graph_id"],
                "split_id": event["split_id"],
                "backend_id": event["backend_id"],
                "device_id": event["device_id"],
                "first_node": args[2],
                "last_node": args[3],
                "status": TRACE.signed_u32(args[1]),
                "dispatch_ids": [],
                "gpu_graph_event_ids": [],
            })
        elif kind == TRACE.K_COPY_HANDLING:
            src_ref = args[3]
            copies.append({
                "id": len(copies) + 1,
                "start_ns": event["timestamp_ns"],
                "end_ns": args[0],
                "execution_id": event["execution_id"],
                "ubatch_id": event["ubatch_id"],
                "graph_id": event["graph_id"],
                "split_id": event["split_id"],
                "path": TRACE.COPY_PATHS.get(event["flags"], str(event["flags"])),
                "src_backend_id": src_ref & 0xffffffff,
                "src_device_id": src_ref >> 32,
                "dst_backend_id": event["backend_id"],
                "dst_device_id": event["device_id"],
                "src_node_id": event["node_id"],
                "dst_node_id": args[1],
                "bytes": args[2],
            })
        elif kind in TRACE.WAIT_KINDS:
            waits.append({
                "id": len(waits) + 1,
                "kind": TRACE.SPAN_NAMES[kind],
                "start_ns": event["timestamp_ns"],
                "end_ns": args[0],
                "execution_id": event["execution_id"],
                "graph_id": event["graph_id"],
                "split_id": event["split_id"],
                "backend_id": event["backend_id"],
                "device_id": event["device_id"],
            })
        elif kind == TRACE.K_GPU_GRAPH:
            graph_events.append({
                "id": len(graph_events) + 1,
                "timestamp_ns": event["timestamp_ns"],
                "execution_id": event["execution_id"],
                "ubatch_id": event["ubatch_id"],
                "graph_id": event["graph_id"],
                "split_id": event["split_id"],
                "backend_id": event["backend_id"],
                "device_id": event["device_id"],
                "state": TRACE.GPU_GRAPH_STATES.get(event["flags"], str(event["flags"])),
                "gpu_graph_id": args[0],
                "node_count": args[1],
            })
        elif kind == TRACE.K_EXECUTION_BEGIN:
            execution_begins[event["execution_id"]] = event
        elif kind == TRACE.K_EXECUTION_END:
            begin = execution_begins.get(event["execution_id"])
            if begin:
                executions.append({
                    "execution_id": event["execution_id"],
                    "start_ns": begin["timestamp_ns"],
                    "end_ns": event["timestamp_ns"],
                    "kind": TRACE.execution_kind(begin),
                    "phase": TRACE.execution_phase(begin),
                })

    submissions.sort(key=lambda item: (item["start_ns"], item["end_ns"], item["id"]))
    for index, submission in enumerate(submissions, 1):
        submission["id"] = index

    def matching_submission(item):
        candidates = [submission for submission in submissions if
                      submission["device_id"] == item["device_id"] and
                      submission["graph_id"] == item["graph_id"] and
                      submission["split_id"] == item["split_id"] and
                      submission["backend_id"] == item["backend_id"] and
                      submission["start_ns"] <= item["timestamp_ns"] <= submission["end_ns"]]
        return candidates[0] if len(candidates) == 1 else None

    for dispatch in dispatches:
        submission = matching_submission(dispatch)
        if submission:
            submission["dispatch_ids"].append(dispatch["id"])
    for graph_event in graph_events:
        submission = matching_submission(graph_event)
        if submission:
            submission["gpu_graph_event_ids"].append(graph_event["id"])

    if executions:
        analysis_window = {
            "start_ns": min(item["start_ns"] for item in executions),
            "end_ns": max(item["end_ns"] for item in executions),
        }
    else:
        analysis_window = {"start_ns": trace["start_ns"], "end_ns": trace["start_ns"]}

    return {
        "metadata": trace_metadata(trace),
        "devices": devices,
        "dispatches": dispatches,
        "submissions": submissions,
        "copies": copies,
        "waits": waits,
        "gpu_graph_events": graph_events,
        "executions": executions,
        "analysis_window": analysis_window,
    }


def clock_alignment(trace, manual_offset_ns, external_clock="CLOCK_MONOTONIC"):
    if manual_offset_ns is not None:
        return {
            "method": "manual_offset",
            "trace_clock": trace_metadata(trace).get("clock", "unknown"),
            "external_clock": external_clock,
            "offset_ns": manual_offset_ns,
            "uncertainty_ns": None,
        }

    anchors = [event for event in trace["events"] if
               event["kind"] == TRACE.K_CLOCK_SYNC and CLOCK_SYNC_KINDS.get(event["flags"]) == external_clock]
    if anchors:
        candidates = []
        for event in anchors:
            raw_before = event["timestamp_ns"]
            secondary = event["args"][0]
            raw_after = event["args"][1]
            if raw_before <= raw_after and secondary:
                width = raw_after - raw_before
                candidates.append({
                    "external_ns": secondary,
                    "trace_ns": (raw_before + raw_after) // 2,
                    "uncertainty_ns": (width + 1) // 2,
                })
        candidates.sort(key=lambda item: item["external_ns"])
        if len(candidates) >= 2:
            first = candidates[0]
            last = candidates[-1]
            external_delta = last["external_ns"] - first["external_ns"]
            trace_delta = last["trace_ns"] - first["trace_ns"]
            if external_delta > 0 and trace_delta > 0:
                return {
                    "method": "clock_sync_affine",
                    "trace_clock": "CLOCK_MONOTONIC_RAW",
                    "external_clock": external_clock,
                    "offset_ns": first["trace_ns"] - first["external_ns"],
                    "uncertainty_ns": max(first["uncertainty_ns"], last["uncertainty_ns"]),
                    "external_anchor_ns": first["external_ns"],
                    "trace_anchor_ns": first["trace_ns"],
                    "rate_numerator": trace_delta,
                    "rate_denominator": external_delta,
                    "drift_ppm": (trace_delta / external_delta - 1.0) * 1_000_000.0,
                    "anchor_count": len(candidates),
                }
        if candidates:
            candidate = min(candidates, key=lambda item: item["uncertainty_ns"])
            return {
                "method": "clock_sync",
                "trace_clock": "CLOCK_MONOTONIC_RAW",
                "external_clock": external_clock,
                "offset_ns": candidate["trace_ns"] - candidate["external_ns"],
                "uncertainty_ns": candidate["uncertainty_ns"],
                "anchor_count": len(candidates),
            }

    metadata = trace_metadata(trace)
    if metadata.get("clock") == external_clock:
        return {
            "method": "shared_clock",
            "trace_clock": external_clock,
            "external_clock": external_clock,
            "offset_ns": 0,
            "uncertainty_ns": 0,
        }
    raise ActivityError(f"trace has no {external_clock} clock anchor; use a schema-v4 trace or --clock-offset-ns")


def divide_round_nearest(numerator, denominator):
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def external_to_trace_ns(alignment, timestamp_ns):
    if alignment["method"] == "clock_sync_affine":
        delta = timestamp_ns - alignment["external_anchor_ns"]
        scaled = divide_round_nearest(delta * alignment["rate_numerator"], alignment["rate_denominator"])
        return alignment["trace_anchor_ns"] + scaled
    return timestamp_ns + alignment["offset_ns"]


def parse_device_maps(values):
    result = {}
    for value in values:
        try:
            device_text, gpu_text = value.split("=", 1)
            device_id = int(device_text)
            gpu_index = int(gpu_text)
        except ValueError as error:
            raise ActivityError(f"invalid device map '{value}'; expected DELPHI_DEVICE_ID=PROVIDER_GPU_INDEX") from error
        if device_id <= 0 or gpu_index < 0 or device_id in result:
            raise ActivityError(f"invalid or duplicate device map '{value}'")
        result[device_id] = gpu_index
    return result


def map_devices(model, activity, requested):
    devices = {item["device_ref"]: item for item in activity["devices"]}
    active_refs = set()
    for item in activity["activities"]:
        active_refs.update(item["device_refs"])
    active_devices = [devices[ref] for ref in active_refs if devices.get(ref, {}).get("type") == "gpu"]
    active_devices.sort(key=lambda item: (item.get("gpu_index", -1), item["device_ref"]))
    trace_devices = [model["devices"][device_id] for device_id in sorted(model["devices"])]

    by_gpu_index = defaultdict(list)
    for device in active_devices:
        by_gpu_index[device.get("gpu_index", -1)].append(device)

    mappings = []
    used_refs = set()
    used_devices = set()
    for device_id, gpu_index in requested.items():
        matches = by_gpu_index.get(gpu_index, [])
        if device_id not in model["devices"]:
            raise ActivityError(f"DELPHI device {device_id} is not a traced GPU")
        if len(matches) != 1:
            raise ActivityError(f"provider GPU index {gpu_index} is not one active GPU")
        device = matches[0]
        if device["device_ref"] in used_refs:
            raise ActivityError(f"provider GPU index {gpu_index} is mapped more than once")
        mappings.append({"device_id": device_id, "device_ref": device["device_ref"], "method": "explicit_gpu_index"})
        used_devices.add(device_id)
        used_refs.add(device["device_ref"])

    remaining_devices = [item for item in trace_devices if item["device_id"] not in used_devices]
    remaining_refs = [item for item in active_devices if item["device_ref"] not in used_refs]
    if len(remaining_devices) == len(remaining_refs):
        method = "single_active_gpu" if len(remaining_devices) == 1 else "ordered_gpu_index"
        for trace_device, device in zip(remaining_devices, remaining_refs):
            mappings.append({"device_id": trace_device["device_id"], "device_ref": device["device_ref"], "method": method})

    mappings.sort(key=lambda item: item["device_id"])
    for mapping in mappings:
        mapping["gpu_index"] = devices[mapping["device_ref"]].get("gpu_index", -1)
    return mappings


class SpanIndex:
    def __init__(self, spans, span_devices):
        self.by_device = {}
        pending = defaultdict(list)
        for span in spans:
            for device_id in set(span_devices(span)):
                pending[device_id].append(span)

        for device_id, device_spans in pending.items():
            device_spans.sort(key=lambda item: (item["start_ns"], item["id"]))
            prefix_max_end = []
            maximum = 0
            for span in device_spans:
                maximum = max(maximum, span["end_ns"])
                prefix_max_end.append(maximum)
            self.by_device[device_id] = {
                "spans": device_spans,
                "starts": [span["start_ns"] for span in device_spans],
                "prefix_max_end": prefix_max_end,
            }

    def unique_containing(self, timestamp_ns, device_ids, uncertainty_ns):
        margin = uncertainty_ns or 0
        lower = timestamp_ns - margin
        upper = timestamp_ns + margin
        matches = {}
        for device_id in device_ids:
            indexed = self.by_device.get(device_id)
            if not indexed:
                continue
            index = bisect_right(indexed["starts"], upper) - 1
            while index >= 0 and indexed["prefix_max_end"][index] >= lower:
                span = indexed["spans"][index]
                if span["end_ns"] >= lower:
                    matches[span["id"]] = span
                    if len(matches) > 1:
                        return None
                index -= 1
        return next(iter(matches.values())) if matches else None

    def latest_before(self, timestamp_ns, device_ids):
        candidates = []
        for device_id in device_ids:
            indexed = self.by_device.get(device_id)
            if not indexed:
                continue
            index = bisect_right(indexed["starts"], timestamp_ns) - 1
            if index >= 0:
                candidates.append(indexed["spans"][index])
        return max(candidates, key=lambda item: (item["start_ns"], item["id"])) if candidates else None


def time_window_submission(submission_index, timestamp_ns, device_ids, analysis_window):
    if not (analysis_window["start_ns"] <= timestamp_ns <= analysis_window["end_ns"]):
        return None
    return submission_index.latest_before(timestamp_ns, device_ids)


def correlate(model, api_calls, activities, alignment, mappings, correlation_tolerance_ns):
    uncertainty_ns = alignment["uncertainty_ns"]
    matching_margin_ns = (uncertainty_ns or 0) + correlation_tolerance_ns
    ref_to_device = {item["device_ref"]: item["device_id"] for item in mappings}
    api_by_correlation = defaultdict(list)
    submission_index = SpanIndex(model["submissions"], lambda item: (item["device_id"],))
    copy_index = SpanIndex(model["copies"], lambda item: (item["src_device_id"], item["dst_device_id"]))

    for api in api_calls:
        api_by_correlation[api["correlation_ids"].get("internal", 0)].append(api)
    activities_by_correlation = defaultdict(list)
    for activity in activities:
        activities_by_correlation[activity["correlation_ids"].get("internal", 0)].append(activity)

    graph_events = {item["id"]: item for item in model["gpu_graph_events"]}
    for api in api_calls:
        related = activities_by_correlation.get(api["correlation_ids"].get("internal", 0), [])
        device_ids = {ref_to_device[ref] for activity in related for ref in activity["device_refs"] if ref in ref_to_device}
        submission = submission_index.unique_containing(api["start_ns"], device_ids, matching_margin_ns)
        copy = copy_index.unique_containing(api["start_ns"], device_ids, matching_margin_ns)
        api["host_submission_id"] = submission["id"] if submission else 0
        api["host_copy_id"] = copy["id"] if copy else 0
        api["delphi_gpu_graph_id"] = 0
        if submission and api["graph_launch"]:
            replays = [graph_events[event_id] for event_id in submission["gpu_graph_event_ids"] if
                       graph_events[event_id]["state"] == "replay"]
            if len(replays) == 1:
                api["delphi_gpu_graph_id"] = replays[0]["gpu_graph_id"]

    for activity in activities:
        device_ids = {ref_to_device[ref] for ref in activity["device_refs"] if ref in ref_to_device}
        apis = api_by_correlation.get(activity["correlation_ids"].get("internal", 0), [])
        api = apis[0] if len(apis) == 1 else None
        correlation = {
            "relationship": "unattributed",
            "confidence": "none",
            "api_call_id": api["id"] if api else 0,
            "host_submission_id": 0,
            "host_copy_id": 0,
            "delphi_gpu_graph_id": 0,
        }
        if api and api["delphi_gpu_graph_id"]:
            correlation["relationship"] = "ordered_graph_launch"
            correlation["confidence"] = "heuristic"
            correlation["host_submission_id"] = api["host_submission_id"]
            correlation["delphi_gpu_graph_id"] = api["delphi_gpu_graph_id"]
        elif api and api["host_copy_id"] and activity["type"] == "copy":
            correlation["relationship"] = "api_within_copy"
            correlation["confidence"] = "heuristic"
            correlation["host_copy_id"] = api["host_copy_id"]
            correlation["host_submission_id"] = api["host_submission_id"]
        elif api and api["host_submission_id"]:
            correlation["relationship"] = "ordered_within_dispatch"
            correlation["confidence"] = "heuristic"
            correlation["host_submission_id"] = api["host_submission_id"]
        else:
            submission = time_window_submission(submission_index, activity["start_ns"], device_ids, model["analysis_window"])
            if submission:
                correlation["relationship"] = "time_window_candidate"
                correlation["confidence"] = "heuristic"
                correlation["host_submission_id"] = submission["id"]
        activity["correlation"] = correlation


def clipped(interval, window):
    start_ns = max(interval[0], window["start_ns"])
    end_ns = min(interval[1], window["end_ns"])
    return (start_ns, end_ns) if end_ns > start_ns else None


def merge_intervals(intervals):
    merged = []
    for start_ns, end_ns in sorted(intervals):
        if not merged or start_ns > merged[-1][1]:
            merged.append([start_ns, end_ns])
        else:
            merged[-1][1] = max(merged[-1][1], end_ns)
    return merged


def interval_records(intervals):
    return [{"start_ns": start_ns, "end_ns": end_ns, "duration_ns": end_ns - start_ns} for start_ns, end_ns in intervals]


def complement(intervals, window):
    result = []
    cursor = window["start_ns"]
    for start_ns, end_ns in intervals:
        if start_ns > cursor:
            result.append([cursor, start_ns])
        cursor = max(cursor, end_ns)
    if cursor < window["end_ns"]:
        result.append([cursor, window["end_ns"]])
    return result


def overlap_duration(intervals, start_ns, end_ns):
    return sum(max(0, min(end_ns, right) - max(start_ns, left)) for left, right in intervals)


def build_intervals(model, devices, activities, mappings):
    window = model["analysis_window"]
    interval_sets = {}
    mapping_by_ref = {item["device_ref"]: item for item in mappings}
    refs = sorted({ref for activity in activities for ref in activity["device_refs"]})
    summaries = []
    for ref in refs:
        related = [item for item in activities if ref in item["device_refs"]]
        kernel = [value for item in related if item["type"] == "kernel" for value in [clipped((item["start_ns"], item["end_ns"]), window)] if value]
        dma = [value for item in related if item["type"] != "kernel" for value in [clipped((item["start_ns"], item["end_ns"]), window)] if value]
        kernel = merge_intervals(kernel)
        dma = merge_intervals(dma)
        active = merge_intervals(kernel + dma)
        interval_sets[ref] = {"kernel": kernel, "dma": dma, "active": active}
        mapping = mapping_by_ref.get(ref, {})
        summaries.append({
            "device_ref": ref,
            "gpu_index": devices.get(ref, {}).get("gpu_index", -1),
            "device_id": mapping.get("device_id", 0),
            "gpu_busy_intervals": interval_records(kernel),
            "dma_intervals": interval_records(dma),
            "gpu_active_intervals": interval_records(active),
            "gpu_idle_intervals": interval_records(complement(active, window)),
        })

    wait_overlaps = []
    ref_by_device = {item["device_id"]: item["device_ref"] for item in mappings}
    for wait in model["waits"]:
        ref = ref_by_device.get(wait["device_id"])
        if not ref:
            continue
        intervals = interval_sets.get(ref, {"kernel": [], "dma": [], "active": []})
        duration_ns = wait["end_ns"] - wait["start_ns"]
        active_ns = overlap_duration(intervals["active"], wait["start_ns"], wait["end_ns"])
        wait_overlaps.append({
            "wait_id": wait["id"],
            "device_ref": ref,
            "device_id": wait["device_id"],
            "duration_ns": duration_ns,
            "gpu_busy_overlap_ns": overlap_duration(intervals["kernel"], wait["start_ns"], wait["end_ns"]),
            "dma_overlap_ns": overlap_duration(intervals["dma"], wait["start_ns"], wait["end_ns"]),
            "gpu_active_overlap_ns": active_ns,
            "no_gpu_activity_ns": duration_ns - active_ns,
        })
    return summaries, wait_overlaps


def merge(trace, document, requested_maps=None, manual_offset_ns=None,
          correlation_tolerance_ns=DEFAULT_CORRELATION_TOLERANCE_NS):
    validate_activity_document(document)
    if trace["flags"] & 1:
        raise ActivityError("DELPHI trace is truncated")
    provider = document["provider"]
    model = trace_model(trace)
    alignment = clock_alignment(trace, manual_offset_ns, provider["clock"])

    # Keep the provider timestamps and add trace-clock copies; the input document is not changed.
    api_calls = []
    for api in document.get("api_calls", []):
        api_calls.append(dict(api, correlation_ids=dict(api.get("correlation_ids", {})),
                              external_start_ns=api["start_ns"], external_end_ns=api["end_ns"],
                              start_ns=external_to_trace_ns(alignment, api["start_ns"]),
                              end_ns=external_to_trace_ns(alignment, api["end_ns"])))
    activities = []
    for activity in document.get("activities", []):
        activities.append(dict(activity, correlation_ids=dict(activity.get("correlation_ids", {})),
                               device_refs=list(activity.get("device_refs", [])),
                               external_start_ns=activity["start_ns"], external_end_ns=activity["end_ns"],
                               start_ns=external_to_trace_ns(alignment, activity["start_ns"]),
                               end_ns=external_to_trace_ns(alignment, activity["end_ns"]),
                               provenance=provider["provenance"], timing_source=provider["name"]))

    devices = {item["device_ref"]: item for item in document.get("devices", [])}
    mappings = map_devices(model, {"devices": list(devices.values()), "activities": activities}, requested_maps or {})
    correlate(model, api_calls, activities, alignment, mappings, correlation_tolerance_ns)
    intervals, wait_overlaps = build_intervals(model, devices, activities, mappings)

    relationships = Counter(item["correlation"]["relationship"] for item in activities)
    exact = relationships["exact_graph_node"]
    unattributed = relationships["unattributed"]
    diagnostics = {
        "external_gpu_records": len(activities),
        "correlated_exact": exact,
        "correlated_heuristic": len(activities) - exact - unattributed,
        "unattributed": unattributed,
        "relationships": dict(sorted(relationships.items())),
        "external_graph_node_records": sum(bool(item.get("graph_identity_present")) for item in activities),
        "api_records": len(api_calls),
        "mapped_devices": len(mappings),
        "traced_gpu_devices": len(model["devices"]),
    }

    process = provider.get("process", {})
    source = {
        "delphi_schema_version": trace["version"],
        "delphi_build_commit": model["metadata"].get("build_commit", ""),
        "activity_schema": document["schema"],
    }
    if process:
        source["provider_process"] = dict(process)
        for key in ("init_time_ns", "fini_time_ns"):
            if process.get(key):
                source["provider_process"][key.replace("_ns", "_aligned_ns")] = external_to_trace_ns(alignment, process[key])

    return {
        "schema": MERGED_SCHEMA,
        "provider": provider,
        "source": source,
        "clock_alignment": alignment,
        "analysis_window": model["analysis_window"],
        "correlation_tolerance_ns": correlation_tolerance_ns,
        "device_mappings": mappings,
        "external_devices": list(devices.values()),
        "executions": model["executions"],
        "host_submissions": model["submissions"],
        "host_copies": model["copies"],
        "host_waits": model["waits"],
        "dispatches": model["dispatches"],
        "gpu_graph_events": model["gpu_graph_events"],
        "gpu_api_calls": api_calls,
        "gpu_activities": activities,
        "device_intervals": intervals,
        "host_wait_overlaps": wait_overlaps,
        "diagnostics": diagnostics,
    }


LEGACY_ACTIVITY_TYPES = {"GPU_KERNEL": "kernel", "GPU_MEMCPY": "copy"}


def upgrade_merged(merged):
    """Returns a merged document in the current schema. Legacy ROCm merges are converted field by field."""
    schema = merged.get("schema")
    if schema == MERGED_SCHEMA:
        return merged
    if schema not in LEGACY_MERGED_SCHEMAS:
        raise ActivityError(f"GPU activity input is not a {MERGED_SCHEMA} document")

    def device_ref(item, key="agent_id"):
        return dict((k, v) for k, v in item.items() if k != key) | {"device_ref": item.get(key, 0)}

    activities = []
    for item in merged.get("gpu_activities", []):
        activity = device_ref(item)
        activity.pop("device_agent_ids", None)
        activity["device_refs"] = list(item.get("device_agent_ids", []))
        activity.pop("kind", None)
        activity["type"] = LEGACY_ACTIVITY_TYPES.get(item.get("kind"), "kernel")
        for old, new in (("src_agent_id", "src_device_ref"), ("dst_agent_id", "dst_device_ref")):
            if old in activity:
                activity[new] = activity.pop(old)
        activity["provenance"] = "VENDOR_EXTERNAL"
        activity["timing_source"] = "rocprofiler-sdk"
        activities.append(activity)
    apis = []
    for item in merged.get("gpu_api_calls", []):
        api = dict(item)
        api["name"] = api.pop("operation", api.get("name", ""))
        api["graph_launch"] = api["name"] in ("hipGraphLaunch", "hipGraphLaunch_spt")
        apis.append(api)

    result = dict(merged)
    result.update({
        "schema": MERGED_SCHEMA,
        "provider": {"name": "rocprofiler-sdk", "provenance": "VENDOR_EXTERNAL", "production_representative": False,
                     "clock": "CLOCK_MONOTONIC", "graph_node_identity": "none"},
        "device_mappings": [device_ref(item) for item in merged.get("device_mappings", [])],
        "external_devices": [dict(device_ref(item), type="gpu" if item.get("type", 2) == 2 else "cpu")
                             for item in merged.get("external_agents", [])],
        "gpu_api_calls": apis,
        "gpu_activities": activities,
        "device_intervals": [device_ref(item) for item in merged.get("device_intervals", [])],
        "host_wait_overlaps": [device_ref(item) for item in merged.get("host_wait_overlaps", [])],
    })
    result.pop("external_agents", None)
    result["source"] = dict(merged.get("source", {}), activity_schema=schema)
    return result


def print_diagnostics(result, stream):
    diagnostics = result["diagnostics"]
    alignment = result["clock_alignment"]
    provider = result["provider"]
    uncertainty = alignment["uncertainty_ns"]
    uncertainty_text = "unknown" if uncertainty is None else str(uncertainty)
    print(f"provider: {provider['name']} ({provider['provenance']}, production_representative="
          f"{str(provider['production_representative']).lower()}, graph_node_identity={provider['graph_node_identity']})",
          file=stream)
    print(f"external GPU records: {diagnostics['external_gpu_records']}", file=stream)
    print(f"correlated exact: {diagnostics['correlated_exact']}", file=stream)
    print(f"correlated heuristic: {diagnostics['correlated_heuristic']}", file=stream)
    print(f"unattributed: {diagnostics['unattributed']}", file=stream)
    drift = f" drift_ppm={alignment['drift_ppm']:.3f}" if "drift_ppm" in alignment else ""
    print(f"clock alignment: {alignment['method']} offset_ns={alignment['offset_ns']} uncertainty_ns={uncertainty_text}{drift}", file=stream)


def main():
    parser = argparse.ArgumentParser(description=f"Validate or merge provider-neutral {SCHEMA} GPU activity")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="check a GPU activity document")
    validate_parser.add_argument("activity", type=Path)
    merge_parser = subparsers.add_parser("merge", help="align GPU activity with a DELPHI raw trace")
    merge_parser.add_argument("trace", type=Path, help="DELPHI raw execution trace")
    merge_parser.add_argument("activity", type=Path, help=f"{SCHEMA} JSON")
    merge_parser.add_argument("-o", "--output", type=Path, required=True, help="merged JSON output")
    merge_parser.add_argument("--device-map", action="append", default=[], metavar="DEVICE=GPU",
                              help="map a DELPHI device ID to a provider GPU index; may be repeated")
    merge_parser.add_argument("--clock-offset-ns", type=int, help="manual value added to provider timestamps")
    merge_parser.add_argument("--correlation-tolerance-ns", type=int, default=DEFAULT_CORRELATION_TOLERANCE_NS,
                              help="extra clock-boundary tolerance for matching APIs to host spans (default: 10000)")
    args = parser.parse_args()

    try:
        document = json.loads(args.activity.read_text())
        if args.command == "validate":
            validate_activity_document(document)
            print(f"{args.activity}: valid {SCHEMA} from {document['provider']['name']}")
            return 0
        if args.correlation_tolerance_ns < 0:
            raise ActivityError("--correlation-tolerance-ns must not be negative")
        result = merge(TRACE.read_trace(args.trace), document, parse_device_maps(args.device_map),
                       args.clock_offset_ns, args.correlation_tolerance_ns)
        result["source"]["delphi_trace"] = str(args.trace)
        result["source"]["activity"] = str(args.activity)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print_diagnostics(result, sys.stdout)
    except (OSError, json.JSONDecodeError, ActivityError, TRACE.TraceError) as error:
        print(f"exec-trace-gpu-activity: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
