#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = "delphi-unified-timeline-v1"
PERFETTO_SCHEMA = "delphi-perfetto-trace-v1"


class ExportError(Exception):
    pass


def load_module(name, file_name):
    path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACTIVITY = load_module("exec_trace_gpu_activity", "exec-trace-gpu-activity.py")
TRACE = ACTIVITY.TRACE


def overlap(interval, window):
    return interval[0] < window["end_ns"] and interval[1] > window["start_ns"]


def clipped(interval, window):
    start_ns = max(interval[0], window["start_ns"])
    end_ns = min(interval[1], window["end_ns"])
    return (start_ns, end_ns) if end_ns > start_ns else None


def merge_intervals(intervals):
    result = []
    for start_ns, end_ns in sorted(intervals):
        if end_ns <= start_ns:
            continue
        if not result or start_ns > result[-1][1]:
            result.append([start_ns, end_ns])
        else:
            result[-1][1] = max(result[-1][1], end_ns)
    return result


def interval_duration(intervals):
    return sum(end_ns - start_ns for start_ns, end_ns in intervals)


def interval_intersection(left, right):
    result = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start_ns = max(left[left_index][0], right[right_index][0])
        end_ns = min(left[left_index][1], right[right_index][1])
        if end_ns > start_ns:
            result.append([start_ns, end_ns])
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


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


def paired_spans(events, begin_kind, end_kind, id_name):
    begins = {event[id_name]: event for event in events if event["kind"] == begin_kind}
    result = []
    for event in events:
        if event["kind"] != end_kind:
            continue
        begin = begins.get(event[id_name])
        if begin is None:
            continue
        if event["timestamp_ns"] < begin["timestamp_ns"]:
            raise ExportError(f"{id_name} {event[id_name]} ends before it starts")
        result.append((begin, event))
    return result


def trace_catalog(trace):
    events = trace["events"]
    strings = trace["strings"]
    backend_name = TRACE.build_backend_names(events, strings)
    node_metadata = TRACE.build_node_metadata(events, strings)
    topology_by_graph = {}
    graph_events = []
    splits = defaultdict(list)
    split_inputs = defaultdict(list)
    devices = {}
    nodes = defaultdict(list)

    for event in events:
        kind = event["kind"]
        args = event["args"]
        if kind == TRACE.K_GRAPH:
            topology_id = args[0] or event["graph_id"]
            topology_by_graph[event["graph_id"]] = topology_id
            graph_events.append(event)
        elif kind == TRACE.K_BACKEND:
            devices[(event["backend_id"], event["device_id"])] = {
                "backend_name": backend_name(event),
                "device_name": TRACE.string(strings, args[2]),
                "device_description": TRACE.string(strings, args[3]),
                "device_type": event["flags"],
            }
        elif kind == TRACE.K_NODE:
            nodes[event["graph_id"]].append(event)
        elif kind == TRACE.K_SCHED_SPLIT:
            splits[event["graph_id"]].append(event)
        elif kind == TRACE.K_SCHED_SPLIT_INPUT:
            split_inputs[(event["graph_id"], event["split_id"])].append(event)

    return {
        "backend_name": backend_name,
        "node_metadata": node_metadata,
        "topology_by_graph": topology_by_graph,
        "graph_events": graph_events,
        "splits": splits,
        "split_inputs": split_inputs,
        "devices": devices,
        "nodes": nodes,
        "executions": paired_spans(events, TRACE.K_EXECUTION_BEGIN, TRACE.K_EXECUTION_END, "execution_id"),
        "ubatches": paired_spans(events, TRACE.K_UBATCH_BEGIN, TRACE.K_UBATCH_END, "ubatch_id"),
    }


def execution_window(catalog, execution_id):
    executions = catalog["executions"]
    if execution_id is not None:
        executions = [pair for pair in executions if pair[0]["execution_id"] == execution_id]
        if not executions:
            raise ExportError(f"execution {execution_id} is not present in the trace")
    if not executions:
        raise ExportError("trace has no complete execution")
    return {
        "start_ns": min(begin["timestamp_ns"] for begin, _ in executions),
        "end_ns": max(end["timestamp_ns"] for _, end in executions),
        "execution_ids": [begin["execution_id"] for begin, _ in executions],
    }


def upgrade_merged(merged):
    try:
        return ACTIVITY.upgrade_merged(merged)
    except ACTIVITY.ActivityError as error:
        raise ExportError(str(error)) from error


def validate_merged(trace, catalog, merged):
    source = merged.get("source", {})
    if source.get("delphi_schema_version") != trace["version"]:
        raise ExportError("GPU activity input was produced from a different DELPHI schema version")
    expected = sorted((begin["execution_id"], begin["timestamp_ns"], end["timestamp_ns"])
                      for begin, end in catalog["executions"])
    actual = sorted((item["execution_id"], item["start_ns"], item["end_ns"])
                    for item in merged.get("executions", []))
    if expected != actual:
        raise ExportError("GPU activity input was produced from a different DELPHI trace")


class Timeline:
    def __init__(self, window):
        self.window = window
        self.processes = {}
        self.tracks = {}
        self.events = []
        self.flows = []
        self.next_event_id = 1
        self.next_flow_id = 1

    def process(self, key, name, sort_index):
        if key not in self.processes:
            self.processes[key] = {
                "id": len(self.processes) + 1,
                "key": key,
                "name": name,
                "sort_index": sort_index,
            }
        return self.processes[key]

    def track(self, key, process_key, process_name, name, sort_index=0, process_sort_index=0):
        if key not in self.tracks:
            process = self.process(process_key, process_name, process_sort_index)
            process_tracks = [item for item in self.tracks.values() if item["process_id"] == process["id"]]
            self.tracks[key] = {
                "id": key,
                "process_id": process["id"],
                "thread_id": len(process_tracks) + 1,
                "name": name,
                "sort_index": sort_index,
            }
        return self.tracks[key]

    def add_event(self, name, category, track, start_ns, end_ns=None, args=None):
        event = {
            "id": self.next_event_id,
            "name": name,
            "category": category,
            "track_id": track["id"],
            "start_ns": start_ns,
            "args": args or {},
        }
        if end_ns is not None:
            if end_ns < start_ns:
                raise ExportError(f"{name} ends before it starts")
            event["end_ns"] = end_ns
            event["duration_ns"] = end_ns - start_ns
        self.events.append(event)
        self.next_event_id += 1
        return event

    def add_flow(self, name, source, destination, start_ns, end_ns, args=None):
        flow = {
            "id": self.next_flow_id,
            "name": name,
            "source_event_id": source["id"],
            "destination_event_id": destination["id"],
            "source_track_id": source["track_id"],
            "destination_track_id": destination["track_id"],
            "start_ns": start_ns,
            "end_ns": end_ns,
            "args": args or {},
        }
        self.flows.append(flow)
        self.next_flow_id += 1
        return flow


def common_args(event):
    return {
        "execution_id": event["execution_id"],
        "ubatch_id": event["ubatch_id"],
        "graph_id": event["graph_id"],
        "split_id": event["split_id"],
        "node_id": event["node_id"],
        "backend_id": event["backend_id"],
        "device_id": event["device_id"],
        "host_thread_id": event["thread_id"],
    }


def node_args(event, catalog):
    node = catalog["node_metadata"](event) or {}
    args = common_args(event)
    args.update({
        "node_index": node.get("index", 0),
        "node_name": node.get("name", ""),
        "ggml_op": node.get("op", str(event["flags"])),
        "output_type": node.get("type", ""),
        "output_shape": list(node.get("dims", ())),
        "backend": catalog["backend_name"](event),
    })
    if node.get("matmul"):
        m, n, k, selected = node["matmul"]
        args.update({"M": m, "N": n, "K": k})
        if selected:
            args["selected_experts"] = selected
        source = node.get("sources", {}).get(0)
        if source:
            args["weight_type"] = source["type"]
            args["weight_shape"] = list(source["dims"])
    return args


def event_selected(event, execution_ids):
    return event["execution_id"] in execution_ids


def add_raw_events(timeline, trace, catalog, merged):
    execution_ids = set(timeline.window["execution_ids"])
    backend_name = catalog["backend_name"]
    span_events = {}
    submission_lookup = {}
    copy_lookup = {}
    if merged:
        for item in merged.get("host_submissions", []):
            key = (item["start_ns"], item["end_ns"], item["graph_id"], item["split_id"], item["backend_id"])
            submission_lookup[key] = item["id"]
        for item in merged.get("host_copies", []):
            key = (item["start_ns"], item["end_ns"], item["graph_id"], item["split_id"], item["src_node_id"])
            copy_lookup[key] = item["id"]

    for begin, end in catalog["executions"]:
        if begin["execution_id"] not in execution_ids:
            continue
        track = timeline.track(f"host:{begin['thread_id']}", "host", "Host", f"Host thread {begin['thread_id']}", process_sort_index=0)
        args = common_args(begin)
        args.update({
            "kind": TRACE.execution_kind(begin),
            "phase": TRACE.execution_phase(begin),
            "token_count": begin["args"][0],
            "logical_batch_size": begin["args"][1],
            "ubatch_size": begin["args"][2],
            "status": TRACE.signed_u32(end["args"][0]),
        })
        timeline.add_event(f"Execution {begin['execution_id']} ({args['kind']})", "execution", track,
                           begin["timestamp_ns"], end["timestamp_ns"], args)

    for begin, end in catalog["ubatches"]:
        if begin["execution_id"] not in execution_ids:
            continue
        track = timeline.track(f"host:{begin['thread_id']}", "host", "Host", f"Host thread {begin['thread_id']}")
        args = common_args(begin)
        args.update({
            "token_count": begin["args"][0],
            "sequence_count": begin["args"][1],
            "unique_sequence_count": begin["args"][2],
            "output_count": begin["args"][3],
            "status": TRACE.signed_u32(end["args"][0]),
        })
        timeline.add_event(f"Ubatch {begin['ubatch_id']}", "ubatch", track,
                           begin["timestamp_ns"], end["timestamp_ns"], args)

    graph_track = timeline.track("scheduler:graph", "scheduler", "Scheduler", "Graph metadata", process_sort_index=10)
    split_track = timeline.track("scheduler:split", "scheduler", "Scheduler", "Split metadata", sort_index=1)
    node_track = timeline.track("scheduler:node", "scheduler", "Scheduler", "GGML node metadata", sort_index=2)
    dependency_track = timeline.track("scheduler:dependency", "scheduler", "Scheduler", "Cross-backend dependencies", sort_index=3)

    for event in catalog["graph_events"]:
        if not event_selected(event, execution_ids):
            continue
        args = common_args(event)
        args.update({
            "topology_graph_id": event["args"][0] or event["graph_id"],
            "node_count": event["args"][1],
            "leaf_count": event["args"][2],
            "split_count": event["args"][3],
            "reused_topology": bool(event["flags"] & 1),
        })
        timeline.add_event(f"Graph {event['graph_id']}", "graph", graph_track, event["timestamp_ns"], args=args)
        topology_id = args["topology_graph_id"]
        for split in catalog["splits"].get(topology_id, []):
            split_args = common_args(split)
            split_args.update({
                "graph_id": event["graph_id"],
                "topology_graph_id": topology_id,
                "first_node": split["args"][0],
                "last_node": split["args"][1],
                "node_count": split["args"][2],
                "input_count": split["args"][3],
                "backend": backend_name(split),
            })
            timeline.add_event(f"Split {split['split_id']} -> {backend_name(split)}", "scheduler_split",
                               split_track, event["timestamp_ns"], args=split_args)
            for split_input in catalog["split_inputs"].get((topology_id, split["split_id"]), []):
                input_args = common_args(split_input)
                input_args.update({
                    "execution_id": event["execution_id"],
                    "graph_id": event["graph_id"],
                    "topology_graph_id": topology_id,
                    "input_index": split_input["args"][0],
                    "source_node_id": split_input["args"][1],
                    "source_split_id": split_input["args"][2],
                    "source_name": TRACE.string(trace["strings"], split_input["args"][3]),
                    "source_backend_id": split_input["flags"],
                    "source_backend": backend_name(split_input, split_input["flags"]),
                    "destination_backend": backend_name(split_input),
                })
                timeline.add_event(f"Split {split['split_id']} input {split_input['args'][0]}",
                                   "scheduler_dependency", dependency_track, event["timestamp_ns"], args=input_args)
        for node_event in catalog["nodes"].get(topology_id, []):
            args = node_args(node_event, catalog)
            args.update({"execution_id": event["execution_id"], "graph_id": event["graph_id"],
                         "topology_graph_id": topology_id})
            name = f"{args['ggml_op']} {args['node_name']}".rstrip()
            timeline.add_event(name, "ggml_node", node_track, event["timestamp_ns"], args=args)

    split_activity = defaultdict(list)
    for event in trace["events"]:
        if event["kind"] not in TRACE.SPAN_NAMES or not event_selected(event, execution_ids):
            continue
        end_ns = event["args"][0]
        if not overlap((event["timestamp_ns"], end_ns), timeline.window):
            continue
        args = common_args(event)
        args["backend"] = backend_name(event)
        args["status"] = TRACE.signed_u32(event["args"][1]) if event["kind"] in (TRACE.K_BACKEND_SUBMIT, TRACE.K_CPU_COMPUTE) else 0
        name = TRACE.SPAN_NAMES[event["kind"]]
        if event["kind"] == TRACE.K_CPU_COMPUTE:
            track = timeline.track(f"cpu:{event['backend_id']}:{event['device_id']}", "cpu", "CPU backend execution",
                                   backend_name(event), process_sort_index=20)
            args.update({"first_node": event["args"][2], "last_node": event["args"][3]})
        else:
            track = timeline.track(f"host:{event['thread_id']}", "host", "Host", f"Host thread {event['thread_id']}")
            if event["kind"] == TRACE.K_BACKEND_SUBMIT:
                args.update({"first_node": event["args"][2], "last_node": event["args"][3]})
                key = (event["timestamp_ns"], end_ns, event["graph_id"], event["split_id"], event["backend_id"])
                args["host_submission_id"] = submission_lookup.get(key, 0)
            elif event["kind"] == TRACE.K_BACKEND_WAIT:
                args.update({
                    "wait_reason": TRACE.WAIT_REASONS.get(event["flags"], str(event["flags"])),
                    "wait_api": TRACE.WAIT_APIS.get(event["args"][1], str(event["args"][1])),
                })
            elif event["kind"] in (TRACE.K_COPY_HANDLING, TRACE.K_COPY_BODY):
                src_ref = event["args"][3]
                args.update({
                    "copy_path": TRACE.COPY_PATHS.get(event["flags"], str(event["flags"])),
                    "src_backend_id": src_ref & 0xffffffff,
                    "src_device_id": src_ref >> 32,
                    "dst_node_id": event["args"][1],
                    "bytes": event["args"][2],
                })
                key = (event["timestamp_ns"], end_ns, event["graph_id"], event["split_id"], event["node_id"])
                args["host_copy_id"] = copy_lookup.get(key, 0)
            elif event["kind"] in (TRACE.K_COPY_WAIT_SRC, TRACE.K_COPY_WAIT_DST):
                args.update({
                    "copy_path": TRACE.COPY_PATHS.get(event["flags"], str(event["flags"])),
                    "wait_api": TRACE.WAIT_APIS.get(event["args"][1], str(event["args"][1])),
                    "bytes": event["args"][3],
                })
        unified = timeline.add_event(name, "host", track, event["timestamp_ns"], end_ns, args)
        span_events[(event["kind"], event["timestamp_ns"], end_ns, event["graph_id"], event["split_id"], event["backend_id"])] = unified
        if event["split_id"]:
            split_activity[(event["execution_id"], event["graph_id"], event["split_id"])].append(event)

        if event["kind"] == TRACE.K_COPY_HANDLING:
            producer_args = dict(args, endpoint="producer", derived_marker=True)
            consumer_args = dict(args, endpoint="destination_tensor", derived_marker=True)
            producer = timeline.add_event(f"Producer node {event['node_id']}", "dependency", dependency_track,
                                          event["timestamp_ns"], args=producer_args)
            consumer = timeline.add_event(f"Destination node {event['args'][1]}", "dependency", dependency_track,
                                          end_ns, args=consumer_args)
            flow_args = {"relationship": "cross_backend_tensor", "confidence": "exact_trace_ids"}
            timeline.add_flow("producer -> transfer", producer, unified, event["timestamp_ns"], event["timestamp_ns"], flow_args)
            timeline.add_flow("transfer -> destination", unified, consumer, end_ns, end_ns, flow_args)

    lifecycle_track = timeline.track("scheduler:split-lifecycle", "scheduler", "Scheduler", "Split host activity", sort_index=4)
    for (execution_id, graph_id, split_id), events in sorted(split_activity.items()):
        start_ns = min(event["timestamp_ns"] for event in events)
        end_ns = max(event["args"][0] for event in events)
        backend = backend_name(events[0])
        timeline.add_event(f"Graph {graph_id} split {split_id} host activity", "scheduler_split_lifecycle",
                           lifecycle_track, start_ns, end_ns,
                           {"execution_id": execution_id, "graph_id": graph_id, "split_id": split_id,
                            "backend": backend, "derived_interval": True})

    for event in trace["events"]:
        if not event_selected(event, execution_ids):
            continue
        if event["kind"] == TRACE.K_DISPATCH:
            args = node_args(event, catalog)
            _, implementation, _, variant, config_0, config_1 = TRACE.dispatch_fields(event, trace["strings"])
            args.update({
                "dispatch": implementation,
                "dispatch_variant": variant,
                "dispatch_config": TRACE.dispatch_config(implementation, variant, config_0, config_1),
            })
            if implementation.startswith("CPU_"):
                track = timeline.track(f"cpu:{event['backend_id']}:{event['device_id']}", "cpu", "CPU backend execution",
                                       backend_name(event), process_sort_index=20)
            else:
                track = timeline.track(f"gpu-lifecycle:{event['device_id']}", f"gpu:{event['device_id']}",
                                       f"GPU device {event['device_id']}", "Dispatch and graph lifecycle",
                                       process_sort_index=100 + event["device_id"])
            timeline.add_event(f"{implementation}: {args['ggml_op']} {args['node_name']}".rstrip(),
                               "dispatch", track, event["timestamp_ns"], args=args)
        elif event["kind"] == TRACE.K_GPU_GRAPH:
            track = timeline.track(f"gpu-lifecycle:{event['device_id']}", f"gpu:{event['device_id']}",
                                   f"GPU device {event['device_id']}", "Dispatch and graph lifecycle",
                                   process_sort_index=100 + event["device_id"])
            state = TRACE.GPU_GRAPH_STATES.get(event["flags"], str(event["flags"]))
            args = common_args(event)
            args.update({
                "backend": backend_name(event),
                "state": state,
                "gpu_graph_id": event["args"][0],
                "node_count": event["args"][1],
                "reason": TRACE.GPU_GRAPH_REASONS.get(event["args"][2], str(event["args"][2])),
            })
            timeline.add_event(f"GPU graph {state}", "gpu_graph", track, event["timestamp_ns"], args=args)

    return span_events


def device_timing_spans(trace, window):
    execution_ids = set(window["execution_ids"])
    spans, _ = TRACE.gpu_timing_model(trace["events"])
    return [span for span in spans if span["event"]["execution_id"] in execution_ids]


def add_device_timing_events(timeline, trace, catalog):
    for span in device_timing_spans(trace, timeline.window):
        if "device_start_ns" not in span or not overlap((span["device_start_ns"], span["device_end_ns"]), timeline.window):
            continue
        event = span["event"]
        device_id = event["device_id"]
        lane = "operations" if span["scope"] == "eager_op" else "submissions"
        track = timeline.track(f"device-event:{device_id}:{lane}", f"gpu:{device_id}", f"GPU device {device_id}",
                               f"Device-event {lane} (DEVICE_EVENT)", sort_index=-3 if lane == "submissions" else -2,
                               process_sort_index=100 + device_id)
        args = node_args(event, catalog) if span["scope"] == "eager_op" else common_args(event)
        args.update({
            "backend": catalog["backend_name"](event),
            "provenance": span["provenance"],
            "timing_source": "backend_timing_events",
            "scope": span["scope"],
            "status": span["status"],
            "device_duration_ns": span["device_duration_ns"],
            "host_enqueue_start_ns": span["host_enqueue_start_ns"],
            "host_enqueue_end_ns": span["host_enqueue_end_ns"],
            "epoch_id": span["epoch_id"],
            "host_placement": span["placement"],
            "host_placement_uncertainty_ns": span["placement_uncertainty_ns"],
        })
        if span["scope"] == "graph_launch":
            args["gpu_graph_id"] = span["detail"]
            name = f"GPU graph {span['detail']} launch"
        elif span["scope"] == "eager_submit":
            args["node_count"] = span["detail"]
            name = "Eager submission"
        else:
            args["fused_node_count"] = span["detail"]
            name = f"{args['ggml_op']} {args['node_name']}".rstrip()
        timeline.add_event(name, "gpu_device_event", track, span["device_start_ns"], span["device_end_ns"], args)


def add_gpu_events(timeline, merged, span_events):
    if not merged:
        return
    provider = merged.get("provider", {})
    provider_name = provider.get("name", "external provider")
    provenance = provider.get("provenance", "VENDOR_EXTERNAL")
    mapping_by_ref = {item["device_ref"]: item for item in merged.get("device_mappings", [])}
    devices = {item["device_ref"]: item for item in merged.get("external_devices", [])}
    api_events = {}
    submission_events = {}
    copy_events = {}
    submissions_by_id = {item["id"]: item for item in merged.get("host_submissions", [])}
    copies_by_id = {item["id"]: item for item in merged.get("host_copies", [])}
    dispatches_by_id = {item["id"]: item for item in merged.get("dispatches", [])}

    def add_correlation_context(args, submission_id, copy_id):
        submission = submissions_by_id.get(submission_id)
        if submission:
            args.update({
                "execution_id": submission["execution_id"],
                "ubatch_id": submission["ubatch_id"],
                "graph_id": submission["graph_id"],
                "split_id": submission["split_id"],
                "backend_id": submission["backend_id"],
                "device_id": submission["device_id"],
            })
            candidate_ids = [dispatch_id for dispatch_id in submission.get("dispatch_ids", []) if dispatch_id in dispatches_by_id]
            args["candidate_dispatch_count"] = len(candidate_ids)
            args["dispatch_attribution_scope"] = "host_submission"
            if len(candidate_ids) == 1:
                dispatch = dispatches_by_id[candidate_ids[0]]
                args["candidate_dispatch"] = {
                    "dispatch_id": dispatch["id"],
                    "node_id": dispatch["node_id"],
                    "node_name": dispatch["node_name"],
                    "ggml_op": dispatch["op"],
                    "implementation": dispatch["implementation"],
                }
                args["node_id"] = dispatch["node_id"]
            return
        host_copy = copies_by_id.get(copy_id)
        if host_copy:
            args.update({
                "execution_id": host_copy["execution_id"],
                "ubatch_id": host_copy["ubatch_id"],
                "graph_id": host_copy["graph_id"],
                "split_id": host_copy["split_id"],
                "node_id": host_copy["src_node_id"],
                "backend_id": host_copy["dst_backend_id"],
                "device_id": host_copy["dst_device_id"],
            })
    for item in merged.get("host_submissions", []):
        key = (TRACE.K_BACKEND_SUBMIT, item["start_ns"], item["end_ns"], item["graph_id"], item["split_id"], item["backend_id"])
        if key in span_events:
            submission_events[item["id"]] = span_events[key]
    for item in merged.get("host_copies", []):
        key = (TRACE.K_COPY_HANDLING, item["start_ns"], item["end_ns"], item["graph_id"], item["split_id"], item["dst_backend_id"])
        if key in span_events:
            copy_events[item["id"]] = span_events[key]

    referenced_api_ids = {
        activity.get("correlation", {}).get("api_call_id", 0)
        for activity in merged.get("gpu_activities", [])
    }
    referenced_api_ids.discard(0)
    for api in merged.get("gpu_api_calls", []):
        if api["id"] not in referenced_api_ids:
            continue
        if not overlap((api["start_ns"], api["end_ns"]), timeline.window):
            continue
        track = timeline.track(f"external-api:{api.get('thread_id', 0)}", "external-api",
                               f"External profiler API ({provider_name}, {provenance})",
                               f"API thread {api.get('thread_id', 0)}", process_sort_index=90)
        args = {
            "api_call_id": api["id"],
            "stream_id": api.get("stream_id", 0),
            "correlation_ids": api.get("correlation_ids", {}),
            "host_submission_id": api.get("host_submission_id", 0),
            "host_copy_id": api.get("host_copy_id", 0),
            "delphi_gpu_graph_id": api.get("delphi_gpu_graph_id", 0),
            "timing_source": provider_name,
            "provenance": provenance,
        }
        add_correlation_context(args, api.get("host_submission_id", 0), api.get("host_copy_id", 0))
        api_events[api["id"]] = timeline.add_event(api.get("name", ""), "gpu_api", track,
                                                    api["start_ns"], api["end_ns"], args)

    for activity in merged.get("gpu_activities", []):
        if not overlap((activity["start_ns"], activity["end_ns"]), timeline.window):
            continue
        device_ref = activity["device_ref"]
        mapping = mapping_by_ref.get(device_ref, {})
        device_id = mapping.get("device_id", 0)
        device = devices.get(device_ref, {})
        device_label = f"GPU {device_id}" if device_id else f"{provider_name} device {device_ref}"
        if device.get("product_name"):
            device_label += f" ({device['product_name']})"
        activity_type = "kernels" if activity["type"] == "kernel" else "DMA"
        queue = activity.get("queue_id", 0)
        stream = activity.get("stream_id", 0)
        track_key = f"gpu-activity:{device_ref}:{activity_type}:{queue}:{stream}"
        track_name = f"Queue {queue} / stream {stream} {activity_type} ({provenance})"
        track = timeline.track(track_key, f"gpu:{device_id or device_ref}", device_label, track_name,
                               process_sort_index=100 + (device_id or device_ref))
        correlation = activity.get("correlation", {})
        args = {
            "activity_id": activity["id"],
            "activity_type": activity["type"],
            "device_ref": device_ref,
            "device_id": device_id,
            "gpu_index": mapping.get("gpu_index", device.get("gpu_index", -1)),
            "queue_id": queue,
            "stream_id": stream,
            "correlation_ids": activity.get("correlation_ids", {}),
            "correlation_relationship": correlation.get("relationship", "unattributed"),
            "correlation_confidence": correlation.get("confidence", "none"),
            "host_submission_id": correlation.get("host_submission_id", 0),
            "host_copy_id": correlation.get("host_copy_id", 0),
            "delphi_gpu_graph_id": correlation.get("delphi_gpu_graph_id", 0),
            "graph_exec_id": activity.get("graph_exec_id", 0),
            "graph_node_id": activity.get("graph_node_id", 0),
            "graph_node_identity": provider.get("graph_node_identity", "none"),
            "timing_source": provider_name,
            "provenance": activity.get("provenance", provenance),
        }
        add_correlation_context(args, correlation.get("host_submission_id", 0), correlation.get("host_copy_id", 0))
        if activity["type"] != "kernel":
            args.update({
                "bytes": activity.get("bytes", 0),
                "copy_direction": activity.get("copy_direction", activity["name"]),
                "src_device_ref": activity.get("src_device_ref", 0),
                "dst_device_ref": activity.get("dst_device_ref", 0),
            })
        unified = timeline.add_event(activity["name"], "gpu_dma" if activity_type == "DMA" else "gpu_kernel",
                                     track, activity["start_ns"], activity["end_ns"], args)
        source = api_events.get(correlation.get("api_call_id", 0))
        if source is None:
            source = copy_events.get(correlation.get("host_copy_id", 0))
        if source is None:
            source = submission_events.get(correlation.get("host_submission_id", 0))
        if source is not None:
            relationship = correlation.get("relationship", "unattributed")
            confidence = correlation.get("confidence", "none")
            timeline.add_flow("host -> GPU activity", source, unified, source["start_ns"], activity["start_ns"],
                              {"relationship": relationship, "confidence": confidence})

    for intervals in merged.get("device_intervals", []):
        device_ref = intervals["device_ref"]
        mapping = mapping_by_ref.get(device_ref, {})
        device_id = mapping.get("device_id", intervals.get("device_id", 0))
        device = devices.get(device_ref, {})
        label = f"GPU {device_id}" if device_id else f"{provider_name} device {device_ref}"
        if device.get("product_name"):
            label += f" ({device['product_name']})"
        track = timeline.track(f"gpu-union:{device_ref}", f"gpu:{device_id or device_ref}", label,
                               f"Observed activity/idle union ({provenance})", sort_index=-1,
                               process_sort_index=100 + (device_id or device_ref))
        for record in intervals.get("gpu_active_intervals", []):
            value = clipped((record["start_ns"], record["end_ns"]), timeline.window)
            if value:
                timeline.add_event("GPU active union", "gpu_active_union", track, value[0], value[1],
                                   {"device_ref": device_ref, "device_id": device_id, "interval_semantics": "union",
                                    "provenance": provenance})
        for record in intervals.get("gpu_idle_intervals", []):
            value = clipped((record["start_ns"], record["end_ns"]), timeline.window)
            if value:
                timeline.add_event("GPU observed idle", "gpu_idle", track, value[0], value[1],
                                   {"device_ref": device_ref, "device_id": device_id,
                                    "interval_semantics": "complement of observed kernel/DMA union",
                                    "provenance": provenance})


def build_summary(trace, timeline, merged):
    window = timeline.window
    raw_spans = [event for event in trace["events"] if event["kind"] in TRACE.SPAN_NAMES and
                 event["execution_id"] in window["execution_ids"]]

    def intervals_for(kinds):
        return [value for event in raw_spans if event["kind"] in kinds
                for value in [clipped((event["timestamp_ns"], event["args"][0]), window)] if value]

    cpu_spans = intervals_for({TRACE.K_CPU_COMPUTE})
    submit_spans = intervals_for({TRACE.K_BACKEND_SUBMIT})
    backend_wait_spans = intervals_for({TRACE.K_BACKEND_WAIT})
    copy_wait_spans = intervals_for({TRACE.K_COPY_WAIT_SRC, TRACE.K_COPY_WAIT_DST})
    copy_spans = intervals_for({TRACE.K_COPY_HANDLING})
    cpu_union = merge_intervals(cpu_spans)

    device_spans = device_timing_spans(trace, window)
    device_event = None
    if device_spans:
        by_device = defaultdict(list)
        for span in device_spans:
            if span["scope"] != "eager_op" and "device_start_ns" in span:
                value = clipped((span["device_start_ns"], span["device_end_ns"]), window)
                if value:
                    by_device[span["event"]["device_id"]].append(value)
        per_device = []
        all_intervals = []
        for device_id, intervals in sorted(by_device.items()):
            union = merge_intervals(intervals)
            per_device.append({"device_id": device_id, "span_count": len(intervals),
                               "submission_span_union_ns": interval_duration(union)})
            all_intervals.extend(union)
        union = merge_intervals(all_intervals)
        uncertainties = [span["placement_uncertainty_ns"] for span in device_spans
                         if span.get("placement_uncertainty_ns") is not None]
        device_event = {
            "provenance": "DEVICE_EVENT",
            "status_counts": dict(sorted(Counter(span["status"] for span in device_spans).items())),
            "submission_span_union_ns": interval_duration(union),
            "cpu_compute_overlap_ns": interval_duration(interval_intersection(cpu_union, union)),
            "placement_uncertainty_max_ns": max(uncertainties) if uncertainties else None,
            "per_device": per_device,
        }

    summary = {
        "schema": "delphi-execution-summary-v1",
        "execution_ids": window["execution_ids"],
        "window_start_ns": window["start_ns"],
        "window_end_ns": window["end_ns"],
        "execution_wall_ns": window["end_ns"] - window["start_ns"],
        "host": {
            "cpu_compute_span_sum_ns": interval_duration(cpu_spans),
            "cpu_compute_union_ns": interval_duration(cpu_union),
            "backend_submit_span_sum_ns": interval_duration(submit_spans),
            "backend_wait_span_sum_ns": interval_duration(backend_wait_spans),
            "copy_wait_span_sum_ns": interval_duration(copy_wait_spans),
            "copy_handling_span_sum_ns": interval_duration(copy_spans),
        },
        "gpu": None,
        "gpu_device_event": device_event,
        "timing_scope": "in_process_device_event" if device_event else "in_process_only",
        "timing_provenance": {
            "host": "HOST_SAFE",
            "gpu_device_event": "DEVICE_EVENT" if device_event else None,
            "gpu_external": None,
        },
        "value_semantics": {
            "execution_wall_ns": "wall-clock interval from the first selected execution begin to the last selected execution end",
            "span_sum_ns": "sum of matching host span durations; nested or concurrent spans can overlap",
            "union_ns": "duration of the interval union; overlap is counted once",
        },
    }
    if device_event:
        summary["value_semantics"]["submission_span_union_ns"] = (
            "union of eager submission and GPU graph launch intervals measured with timing events on the backend stream; "
            "it includes gaps between kernels inside a span, so it is an upper bound of kernel busy time")
    if not merged:
        return summary

    provider = merged.get("provider", {})
    mapping_by_ref = {item["device_ref"]: item for item in merged.get("device_mappings", [])}
    kernel_by_ref = defaultdict(list)
    dma_by_ref = defaultdict(list)
    for activity in merged.get("gpu_activities", []):
        value = clipped((activity["start_ns"], activity["end_ns"]), window)
        if not value:
            continue
        targets = activity.get("device_refs", []) or [activity.get("device_ref", 0)]
        for device_ref in set(targets):
            if activity["type"] == "kernel":
                kernel_by_ref[device_ref].append(value)
            else:
                dma_by_ref[device_ref].append(value)

    per_device = []
    all_kernel = []
    all_dma = []
    for device_ref in sorted(set(kernel_by_ref) | set(dma_by_ref)):
        kernel = merge_intervals(kernel_by_ref[device_ref])
        dma = merge_intervals(dma_by_ref[device_ref])
        active = merge_intervals(kernel + dma)
        mapping = mapping_by_ref.get(device_ref, {})
        per_device.append({
            "device_ref": device_ref,
            "device_id": mapping.get("device_id", 0),
            "gpu_index": mapping.get("gpu_index", -1),
            "kernel_union_ns": interval_duration(kernel),
            "dma_union_ns": interval_duration(dma),
            "active_union_ns": interval_duration(active),
            "observed_idle_ns": interval_duration(complement(active, window)),
        })
        all_kernel.extend(kernel)
        all_dma.extend(dma)

    kernel_union = merge_intervals(all_kernel)
    dma_union = merge_intervals(all_dma)
    active_union = merge_intervals(kernel_union + dma_union)
    cpu_gpu_overlap = interval_intersection(cpu_union, active_union)
    summary["gpu"] = {
        "kernel_union_ns": interval_duration(kernel_union),
        "dma_union_ns": interval_duration(dma_union),
        "active_union_ns": interval_duration(active_union),
        "observed_idle_ns": interval_duration(complement(active_union, window)),
        "cpu_compute_overlap_ns": interval_duration(cpu_gpu_overlap),
        "per_device": per_device,
        "correlation_diagnostics": merged.get("diagnostics", {}),
        "clock_alignment": merged.get("clock_alignment", {}),
        "provider": provider.get("name", ""),
        "provenance": provider.get("provenance", "VENDOR_EXTERNAL"),
        "production_representative": False,
        "graph_node_identity": provider.get("graph_node_identity", "none"),
    }
    summary["timing_scope"] = "external_profiler_diagnostic"
    summary["timing_provenance"]["gpu_external"] = provider.get("provenance", "VENDOR_EXTERNAL")
    summary["value_semantics"].update({
        "gpu_union_ns": "interval union across all mapped GPU devices; concurrent device activity is counted once",
        "observed_idle_ns": "selected execution window minus the observed kernel/DMA interval union",
        "cpu_compute_overlap_ns": "intersection of the CPU_COMPUTE union and observed GPU kernel/DMA union",
        "external_timing": "VENDOR_EXTERNAL provider timing is diagnostic and is not production-representative",
    })
    return summary


def build_unified(trace, merged=None, execution_id=None):
    if trace["flags"] & 1:
        raise ExportError("DELPHI trace is truncated")
    catalog = trace_catalog(trace)
    if merged:
        merged = upgrade_merged(merged)
        validate_merged(trace, catalog, merged)
    window = execution_window(catalog, execution_id)
    timeline = Timeline(window)
    span_events = add_raw_events(timeline, trace, catalog, merged)
    add_device_timing_events(timeline, trace, catalog)
    add_gpu_events(timeline, merged, span_events)
    summary = build_summary(trace, timeline, merged)
    timeline.events.sort(key=lambda item: (item["start_ns"], item.get("end_ns", item["start_ns"]), item["id"]))
    return {
        "schema": SCHEMA,
        "source": {
            "delphi_schema_version": trace["version"],
            "gpu_activity_schema": merged.get("schema") if merged else None,
            "gpu_activity_provider": merged.get("provider", {}).get("name") if merged else None,
        },
        "window": window,
        "processes": sorted(timeline.processes.values(), key=lambda item: (item["sort_index"], item["id"])),
        "tracks": sorted(timeline.tracks.values(), key=lambda item: (item["process_id"], item["sort_index"], item["thread_id"])),
        "events": timeline.events,
        "flows": timeline.flows,
        "summary": summary,
    }


def perfetto_document(unified):
    tracks = {item["id"]: item for item in unified["tracks"]}
    result = []
    for process in unified["processes"]:
        result.append({"ph": "M", "pid": process["id"], "tid": 0, "name": "process_name",
                       "args": {"name": process["name"]}})
        result.append({"ph": "M", "pid": process["id"], "tid": 0, "name": "process_sort_index",
                       "args": {"sort_index": process["sort_index"]}})
    for track in unified["tracks"]:
        result.append({"ph": "M", "pid": track["process_id"], "tid": track["thread_id"],
                       "name": "thread_name", "args": {"name": track["name"]}})
        result.append({"ph": "M", "pid": track["process_id"], "tid": track["thread_id"],
                       "name": "thread_sort_index", "args": {"sort_index": track["sort_index"]}})

    origin = unified["window"]["start_ns"]
    for event in unified["events"]:
        track = tracks[event["track_id"]]
        output = {
            "name": event["name"],
            "cat": event["category"],
            "pid": track["process_id"],
            "tid": track["thread_id"],
            "ts": (event["start_ns"] - origin) / 1000.0,
            "args": dict(event["args"], start_ns=event["start_ns"]),
        }
        if "end_ns" in event:
            output.update({"ph": "X", "dur": event["duration_ns"] / 1000.0})
            output["args"]["end_ns"] = event["end_ns"]
            output["args"]["duration_ns"] = event["duration_ns"]
        else:
            output.update({"ph": "I", "s": "t"})
        result.append(output)

    for flow in unified["flows"]:
        source = tracks[flow["source_track_id"]]
        destination = tracks[flow["destination_track_id"]]
        common = {"name": flow["name"], "cat": "flow", "id": flow["id"], "args": flow["args"]}
        result.append(dict(common, ph="s", pid=source["process_id"], tid=source["thread_id"],
                           ts=(flow["start_ns"] - origin) / 1000.0))
        result.append(dict(common, ph="f", bp="e", pid=destination["process_id"], tid=destination["thread_id"],
                           ts=(flow["end_ns"] - origin) / 1000.0))

    return {
        "schema": PERFETTO_SCHEMA,
        "traceEvents": result,
        "displayTimeUnit": "ns",
        "delphi": {
            "timestamp_origin_ns": origin,
            "window": unified["window"],
            "summary": unified["summary"],
        },
    }


def tree_text(trace, unified, merged=None):
    merged = upgrade_merged(merged) if merged else None
    catalog = trace_catalog(trace)
    execution_ids = set(unified["window"]["execution_ids"])
    dispatches = defaultdict(list)
    graph_lifecycle = defaultdict(list)
    copies = defaultdict(list)
    for event in trace["events"]:
        if event["execution_id"] not in execution_ids:
            continue
        key = (event["graph_id"], event["split_id"])
        if event["kind"] == TRACE.K_DISPATCH:
            dispatches[(event["graph_id"], event["node_id"])].append(event)
        elif event["kind"] == TRACE.K_GPU_GRAPH:
            graph_lifecycle[key].append(event)
        elif event["kind"] == TRACE.K_COPY_HANDLING:
            copies[key].append(event)

    device_timing = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for span in device_timing_spans(trace, unified["window"]):
        if span["status"] == "resolved":
            event = span["event"]
            stat = device_timing[(event["graph_id"], event["split_id"])][span["scope"]]
            stat[0] += 1
            stat[1] += span["device_duration_ns"]

    activities = defaultdict(list)
    submissions = {item["id"]: item for item in (merged or {}).get("host_submissions", [])}
    for activity in (merged or {}).get("gpu_activities", []):
        correlation = activity.get("correlation", {})
        submission = submissions.get(correlation.get("host_submission_id", 0))
        if submission and submission["execution_id"] in execution_ids:
            activities[(submission["graph_id"], submission["split_id"])].append(activity)

    lines = ["# Derived execution tree", "", "This browsing view is derived from the timeline. It does not represent overlap or a call tree.", ""]
    for execution_id in sorted(execution_ids):
        lines.append(f"execution {execution_id}")
        graphs = [event for event in catalog["graph_events"] if event["execution_id"] == execution_id]
        for graph in graphs:
            graph_id = graph["graph_id"]
            topology_id = graph["args"][0] or graph_id
            lines.append(f"  graph {graph_id} (topology {topology_id})")
            topology_nodes = catalog["nodes"].get(topology_id, [])
            for split in sorted(catalog["splits"].get(topology_id, []), key=lambda item: item["split_id"]):
                key = (graph_id, split["split_id"])
                backend = catalog["backend_name"](split)
                lines.append(f"    split {split['split_id']} -> {backend} nodes [{split['args'][0]},{split['args'][1]})")
                for node_event in topology_nodes:
                    node = catalog["node_metadata"](node_event) or {}
                    if not (split["args"][0] <= node.get("index", -1) < split["args"][1]):
                        continue
                    label = f"{node.get('op', node_event['flags'])} {node.get('name', '')}".rstrip()
                    lines.append(f"      node {node_event['node_id']}: {label} -> {catalog['backend_name'](node_event)}")
                    if node.get("matmul"):
                        m, n, k, selected = node["matmul"]
                        weight = node.get("sources", {}).get(0, {}).get("type", "unknown")
                        suffix = f" selected_experts={selected}" if selected else ""
                        lines.append(f"        matmul: weight={weight} M={m} N={n} K={k}{suffix}")
                    for dispatch in dispatches.get((graph_id, node_event["node_id"]), []):
                        _, implementation, _, variant, config_0, config_1 = TRACE.dispatch_fields(dispatch, trace["strings"])
                        config = TRACE.dispatch_config(implementation, variant, config_0, config_1)
                        lines.append(f"        dispatch: {implementation} variant={variant or 'none'} {config}")
                for event in graph_lifecycle.get(key, []):
                    state = TRACE.GPU_GRAPH_STATES.get(event["flags"], str(event["flags"]))
                    reason = TRACE.GPU_GRAPH_REASONS.get(event["args"][2], str(event["args"][2]))
                    lines.append(f"      GPU graph: {state} id={event['args'][0]} nodes={event['args'][1]} reason={reason}")
                for event in copies.get(key, []):
                    lines.append(f"      cross-backend transfer: src_node={event['node_id']} dst_node={event['args'][1]} bytes={event['args'][2]}")
                for scope, (count, duration_ns) in sorted(device_timing.get(key, {}).items()):
                    lines.append(f"      device-event {scope}: count={count} device_duration_sum_ns={duration_ns} [DEVICE_EVENT]")
                activity_groups = defaultdict(lambda: [0, 0])
                for activity in activities.get(key, []):
                    correlation = activity["correlation"]
                    activity_key = (activity["type"], activity["name"], correlation["relationship"], correlation["confidence"])
                    activity_groups[activity_key][0] += 1
                    activity_groups[activity_key][1] += activity["end_ns"] - activity["start_ns"]
                for (kind, name, relationship, confidence), (count, duration_ns) in sorted(activity_groups.items()):
                    lines.append(f"      GPU {kind}: {name} count={count} duration_sum_ns={duration_ns} [{relationship}, {confidence}]")
    return "\n".join(lines) + "\n"


def print_summary(summary, stream):
    host = summary["host"]
    print(f"execution wall: {summary['execution_wall_ns'] / 1e6:.3f} ms", file=stream)
    print(f"CPU compute: {host['cpu_compute_span_sum_ns'] / 1e6:.3f} ms span sum, {host['cpu_compute_union_ns'] / 1e6:.3f} ms union", file=stream)
    print(f"backend waits: {host['backend_wait_span_sum_ns'] / 1e6:.3f} ms span sum", file=stream)
    print(f"copy handling: {host['copy_handling_span_sum_ns'] / 1e6:.3f} ms span sum", file=stream)
    if summary["gpu_device_event"]:
        device_event = summary["gpu_device_event"]
        statuses = " ".join(f"{name}={count}" for name, count in device_event["status_counts"].items())
        print(f"GPU submissions (DEVICE_EVENT): {device_event['submission_span_union_ns'] / 1e6:.3f} ms union ({statuses})", file=stream)
    if summary["gpu"]:
        gpu = summary["gpu"]
        print(f"GPU kernels: {gpu['kernel_union_ns'] / 1e6:.3f} ms union", file=stream)
        print(f"GPU DMA: {gpu['dma_union_ns'] / 1e6:.3f} ms union", file=stream)
        print(f"GPU idle: {gpu['observed_idle_ns'] / 1e6:.3f} ms observed", file=stream)
        print(f"CPU/GPU overlap: {gpu['cpu_compute_overlap_ns'] / 1e6:.3f} ms union intersection", file=stream)
        print(f"{gpu['provider']} timing is {gpu['provenance']} diagnostic data, not production-representative", file=stream)


def main():
    parser = argparse.ArgumentParser(description="Export a DELPHI execution trace to Perfetto Trace Event JSON")
    parser.add_argument("trace", type=Path, help="DELPHI raw execution trace")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Perfetto-compatible JSON output")
    parser.add_argument("--gpu-activity", "--rocm-merged", dest="gpu_activity", type=Path,
                        help=f"{ACTIVITY.MERGED_SCHEMA} (or legacy delphi-rocm-activity-v1) JSON from exec-trace-rocm.py")
    parser.add_argument("--execution", type=int, help="export one execution ID instead of the complete execution window")
    parser.add_argument("--unified-output", type=Path, help="also write the canonical merged timeline/DAG JSON")
    parser.add_argument("--summary", type=Path, help="also write the host/GPU interval summary JSON")
    parser.add_argument("--tree", type=Path, help="also write a derived tree-style browsing view")
    args = parser.parse_args()

    try:
        trace = TRACE.read_trace(args.trace)
        merged = upgrade_merged(json.loads(args.gpu_activity.read_text())) if args.gpu_activity else None
        unified = build_unified(trace, merged, args.execution)
        unified["source"]["delphi_trace"] = str(args.trace)
        if args.gpu_activity:
            unified["source"]["gpu_activity"] = str(args.gpu_activity)
        args.output.write_text(json.dumps(perfetto_document(unified), separators=(",", ":")) + "\n")
        if args.unified_output:
            args.unified_output.write_text(json.dumps(unified, indent=2) + "\n")
        if args.summary:
            args.summary.write_text(json.dumps(unified["summary"], indent=2) + "\n")
        if args.tree:
            args.tree.write_text(tree_text(trace, unified, merged))
        print_summary(unified["summary"], sys.stdout)
    except (OSError, json.JSONDecodeError, ExportError, TRACE.TraceError) as error:
        print(f"exec-trace-perfetto: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
