#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "exec-trace" / "exec-trace-perfetto.py"
SPEC = importlib.util.spec_from_file_location("exec_trace_perfetto", MODULE_PATH)
exec_trace_perfetto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exec_trace_perfetto)
TRACE = exec_trace_perfetto.TRACE


def event(kind, timestamp_ns, args=(), flags=0, execution_id=1, ubatch_id=1, graph_id=1,
          node_id=0, split_id=0, backend_id=0, device_id=0, thread_id=7):
    values = list(args) + [0] * (4 - len(args))
    return {
        "timestamp_ns": timestamp_ns,
        "args": tuple(values),
        "execution_id": execution_id,
        "ubatch_id": ubatch_id,
        "graph_id": graph_id,
        "node_id": node_id,
        "kind": kind,
        "flags": flags,
        "split_id": split_id,
        "backend_id": backend_id,
        "device_id": device_id,
        "thread_id": thread_id,
    }


def fixture():
    strings = {
        1: "ROCm0", 2: "gfx-test", 3: "GPU A", 4: "CPU", 5: "blk.0.ffn_up",
        6: "MUL_MAT", 7: "F32", 8: "Q6_K", 9: "weights", 10: "activations",
    }
    events = [
        event(TRACE.K_EXECUTION_BEGIN, 1000, (1, 1, 1, 4), flags=1, graph_id=0),
        event(TRACE.K_UBATCH_BEGIN, 1010, (1, 1, 1, 1), graph_id=0),
        event(TRACE.K_GRAPH, 1020, (1, 1, 0, 1)),
        event(TRACE.K_BACKEND, 1020, (1, 0, 2, 3), flags=1, backend_id=1, device_id=1),
        event(TRACE.K_BACKEND, 1020, (4, 0, 4, 4), flags=0, backend_id=2, device_id=2),
        event(TRACE.K_NODE, 1020, (0, 5, 6, 7), flags=26, node_id=11, backend_id=1, device_id=1),
        event(TRACE.K_NODE_DIMS, 1020, (64, 1, 1, 1), node_id=11),
        event(TRACE.K_NODE_SOURCE, 1020, (0, 0, 14, 9), flags=8, node_id=11),
        event(TRACE.K_NODE_SOURCE_DIMS, 1020, (128, 64, 1, 1), flags=0, node_id=11),
        event(TRACE.K_NODE_MATMUL, 1020, (64, 1, 128, 0), flags=26, node_id=11),
        event(TRACE.K_SCHED_SPLIT, 1020, (0, 1, 1, 1), split_id=1, backend_id=1, device_id=1),
        event(TRACE.K_SCHED_SPLIT_INPUT, 1020, (0, 11, 0, 9), flags=2, split_id=1, backend_id=1, device_id=1),
        event(TRACE.K_BACKEND_SUBMIT, 1100, (1200, 0, 0, 1), split_id=1, backend_id=1, device_id=1),
        event(TRACE.K_DISPATCH, 1110, (4, 0, 0, 0), flags=26, node_id=11, split_id=1, backend_id=1, device_id=1),
        event(TRACE.K_COPY_HANDLING, 1210, (1260, 12, 4096, (2 << 32) | 2), flags=3,
              node_id=11, split_id=1, backend_id=1, device_id=1),
        event(TRACE.K_CPU_COMPUTE, 1300, (1700, 0, 0, 1), split_id=2, backend_id=2, device_id=2),
        event(TRACE.K_BACKEND_WAIT, 1710, (1810, 1), flags=1, split_id=1, backend_id=1, device_id=1),
        event(TRACE.K_GPU_GRAPH, 1150, (4, 1, 0), flags=3, split_id=1, backend_id=1, device_id=1),
        # A resolved GRAPH_LAUNCH span with DEVICE_EVENT provenance, and its epoch.
        event(TRACE.K_GPU_TIMING, 1150, (1152, 0, 300, (3 << 32) | 4), flags=2 | (1 << 8) | (2 << 16),
              split_id=1, backend_id=1, device_id=1),
        event(TRACE.K_GPU_TIMING_EPOCH, 1150, (1700, 3, 1, 1), flags=2 << 16, split_id=1, backend_id=1, device_id=1),
        event(TRACE.K_UBATCH_END, 1900, (0,), graph_id=0),
        event(TRACE.K_EXECUTION_END, 2000, (0,), graph_id=0),
    ]
    trace = {"version": 4, "flags": 0, "start_ns": 900, "events": events, "strings": strings, "trace_stats": {}}
    merged = {
        "schema": "delphi-rocm-activity-v1",
        "source": {"delphi_schema_version": 4},
        "clock_alignment": {"method": "clock_sync", "uncertainty_ns": 1},
        "executions": [{"execution_id": 1, "start_ns": 1000, "end_ns": 2000, "kind": "decode"}],
        "device_mappings": [{"device_id": 1, "agent_id": 100, "gpu_index": 0}],
        "external_agents": [{"agent_id": 100, "gpu_index": 0, "product_name": "GPU A"}],
        "host_submissions": [{
            "id": 1, "start_ns": 1100, "end_ns": 1200, "execution_id": 1, "ubatch_id": 1, "graph_id": 1,
            "split_id": 1, "backend_id": 1, "device_id": 1,
        }],
        "host_copies": [{
            "id": 1, "start_ns": 1210, "end_ns": 1260, "execution_id": 1, "ubatch_id": 1, "graph_id": 1,
            "split_id": 1, "src_node_id": 11, "dst_backend_id": 1, "dst_device_id": 1,
        }],
        "gpu_api_calls": [{
            "id": 1, "operation": "hipGraphLaunch", "thread_id": 7, "stream_id": 9,
            "start_ns": 1120, "end_ns": 1140, "correlation_ids": {"internal": 20},
            "host_submission_id": 1, "host_copy_id": 0, "delphi_gpu_graph_id": 4,
        }, {
            "id": 2, "operation": "hipGetDevice", "thread_id": 7, "stream_id": 0,
            "start_ns": 1150, "end_ns": 1151, "correlation_ids": {"internal": 0},
            "host_submission_id": 0, "host_copy_id": 0, "delphi_gpu_graph_id": 0,
        }],
        "gpu_activities": [
            {
                "id": 1, "kind": "GPU_KERNEL", "name": "test_kernel", "agent_id": 100,
                "device_agent_ids": [100], "stream_id": 9, "queue_id": 10,
                "start_ns": 1160, "end_ns": 1500, "correlation_ids": {"internal": 20},
                "correlation": {"relationship": "ordered_graph_launch", "confidence": "heuristic",
                                "api_call_id": 1, "host_submission_id": 1, "host_copy_id": 0,
                                "delphi_gpu_graph_id": 4},
            },
            {
                "id": 2, "kind": "GPU_MEMCPY", "name": "H2D", "agent_id": 100,
                "device_agent_ids": [100], "stream_id": 9, "queue_id": 10,
                "start_ns": 1510, "end_ns": 1610, "correlation_ids": {"internal": 21},
                "correlation": {"relationship": "api_within_copy", "confidence": "heuristic",
                                "api_call_id": 0, "host_submission_id": 0, "host_copy_id": 1,
                                "delphi_gpu_graph_id": 0},
                "bytes": 4096, "copy_direction": "H2D", "src_agent_id": 1, "dst_agent_id": 100,
            },
        ],
        "device_intervals": [{
            "agent_id": 100, "device_id": 1,
            "gpu_active_intervals": [{"start_ns": 1160, "end_ns": 1500}, {"start_ns": 1510, "end_ns": 1610}],
            "gpu_idle_intervals": [{"start_ns": 1000, "end_ns": 1160}, {"start_ns": 1500, "end_ns": 1510},
                                   {"start_ns": 1610, "end_ns": 2000}],
        }],
        "diagnostics": {"correlated_heuristic": 2, "unattributed": 0},
    }
    return trace, merged


class TestExecTracePerfetto(unittest.TestCase):
    def test_builds_tracks_annotations_flows_and_summary(self):
        trace, merged = fixture()
        unified = exec_trace_perfetto.build_unified(trace, merged, 1)

        categories = {item["category"] for item in unified["events"]}
        self.assertTrue({"host", "ggml_node", "dispatch", "gpu_graph", "gpu_kernel", "gpu_dma", "gpu_idle"} <= categories)
        dispatch = next(item for item in unified["events"] if item["category"] == "dispatch")
        self.assertEqual(dispatch["args"]["ggml_op"], "MUL_MAT")
        self.assertEqual(dispatch["args"]["weight_type"], "Q6_K")
        self.assertEqual(dispatch["args"]["M"], 64)
        kernel = next(item for item in unified["events"] if item["category"] == "gpu_kernel")
        self.assertEqual(kernel["args"]["correlation_confidence"], "heuristic")
        self.assertEqual(sum(item["category"] == "gpu_api" for item in unified["events"]), 1)
        self.assertGreaterEqual(len(unified["flows"]), 4)

        summary = unified["summary"]
        self.assertEqual(summary["execution_wall_ns"], 1000)
        self.assertEqual(summary["host"]["cpu_compute_span_sum_ns"], 400)
        self.assertEqual(summary["gpu"]["kernel_union_ns"], 340)
        self.assertEqual(summary["gpu"]["dma_union_ns"], 100)
        self.assertEqual(summary["gpu"]["active_union_ns"], 440)
        self.assertEqual(summary["gpu"]["observed_idle_ns"], 560)
        self.assertEqual(summary["gpu"]["cpu_compute_overlap_ns"], 300)
        self.assertEqual(summary["timing_scope"], "external_profiler_diagnostic")
        self.assertEqual(summary["timing_provenance"],
                         {"host": "HOST_SAFE", "gpu_device_event": "DEVICE_EVENT", "gpu_external": "VENDOR_EXTERNAL"})
        self.assertFalse(summary["gpu"]["production_representative"])
        self.assertEqual(kernel["args"]["provenance"], "VENDOR_EXTERNAL")

    def test_device_event_tracks_without_external_activity(self):
        trace, _ = fixture()
        unified = exec_trace_perfetto.build_unified(trace, None, 1)

        device_event = next(item for item in unified["events"] if item["category"] == "gpu_device_event")
        self.assertEqual(device_event["name"], "GPU graph 4 launch")
        self.assertEqual((device_event["start_ns"], device_event["end_ns"]), (1150, 1450))
        self.assertEqual(device_event["args"]["provenance"], "DEVICE_EVENT")
        self.assertEqual(device_event["args"]["host_placement_uncertainty_ns"], 250)
        summary = unified["summary"]
        self.assertEqual(summary["timing_scope"], "in_process_device_event")
        self.assertEqual(summary["gpu_device_event"]["submission_span_union_ns"], 300)
        self.assertEqual(summary["gpu_device_event"]["cpu_compute_overlap_ns"], 150)
        self.assertIsNone(summary["gpu"])
        self.assertIn("device-event graph_launch: count=1 device_duration_sum_ns=300 [DEVICE_EVENT]",
                      exec_trace_perfetto.tree_text(trace, unified))

    def test_accepts_provider_neutral_merge(self):
        trace, legacy = fixture()
        merged = exec_trace_perfetto.ACTIVITY.upgrade_merged(legacy)
        merged["provider"]["name"] = "future-provider"
        unified = exec_trace_perfetto.build_unified(trace, merged, 1)

        kernel = next(item for item in unified["events"] if item["category"] == "gpu_kernel")
        self.assertEqual(kernel["args"]["timing_source"], "future-provider")
        self.assertEqual(unified["source"]["gpu_activity_schema"], "delphi-gpu-activity-merged-v1")
        self.assertEqual(unified["summary"]["gpu"]["provider"], "future-provider")

    def test_perfetto_document_uses_relative_timestamps_and_flow_events(self):
        trace, merged = fixture()
        unified = exec_trace_perfetto.build_unified(trace, merged, 1)
        document = exec_trace_perfetto.perfetto_document(unified)

        self.assertEqual(document["schema"], exec_trace_perfetto.PERFETTO_SCHEMA)
        self.assertEqual(document["delphi"]["timestamp_origin_ns"], 1000)
        phases = {item["ph"] for item in document["traceEvents"]}
        self.assertTrue({"M", "X", "I", "s", "f"} <= phases)
        execution = next(item for item in document["traceEvents"] if item.get("cat") == "execution")
        self.assertEqual(execution["ts"], 0.0)
        self.assertEqual(execution["dur"], 1.0)

    def test_tree_is_explicitly_derived(self):
        trace, merged = fixture()
        unified = exec_trace_perfetto.build_unified(trace, merged, 1)
        text = exec_trace_perfetto.tree_text(trace, unified, merged)

        self.assertIn("derived from the timeline", text)
        self.assertIn("node 11: MUL_MAT blk.0.ffn_up", text)
        self.assertIn("dispatch: GPU_MMVQ", text)
        self.assertIn("GPU kernel: test_kernel count=1 duration_sum_ns=340 [ordered_graph_launch, heuristic]", text)

    def test_rejects_a_merge_from_another_trace(self):
        trace, merged = fixture()
        merged["executions"][0]["end_ns"] += 1
        with self.assertRaises(exec_trace_perfetto.ExportError):
            exec_trace_perfetto.build_unified(trace, merged)


if __name__ == "__main__":
    unittest.main()
