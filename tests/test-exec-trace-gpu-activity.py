#!/usr/bin/env python3

import copy
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "exec-trace" / "exec-trace-gpu-activity.py"
SPEC = importlib.util.spec_from_file_location("exec_trace_gpu_activity", MODULE_PATH)
exec_trace_gpu_activity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exec_trace_gpu_activity)
TRACE = exec_trace_gpu_activity.TRACE


def event(kind, timestamp_ns, args=(), flags=0, execution_id=1, graph_id=1, node_id=0,
          split_id=0, backend_id=0, device_id=0):
    values = list(args) + [0] * (4 - len(args))
    return {
        "timestamp_ns": timestamp_ns,
        "args": tuple(values),
        "execution_id": execution_id,
        "ubatch_id": execution_id,
        "graph_id": graph_id,
        "node_id": node_id,
        "kind": kind,
        "flags": flags,
        "split_id": split_id,
        "backend_id": backend_id,
        "device_id": device_id,
        "thread_id": 1,
    }


def fixture():
    trace = {
        "version": 4,
        "flags": 0,
        "start_ns": 900,
        "strings": {1: "CUDA0", 2: "sm-test"},
        "trace_stats": {},
        "events": [
            event(TRACE.K_CLOCK_SYNC, 1000, (900, 1002), flags=1, execution_id=0, graph_id=0),
            event(TRACE.K_EXECUTION_BEGIN, 1050, flags=1, graph_id=0),
            event(TRACE.K_BACKEND, 1055, (1, 0, 2, 2), flags=1, backend_id=1, device_id=1),
            event(TRACE.K_BACKEND_SUBMIT, 1100, (1150, 0, 0, 10), split_id=1, backend_id=1, device_id=1),
            event(TRACE.K_GPU_GRAPH, 1105, (7, 4, 0), flags=3, split_id=1, backend_id=1, device_id=1),
            event(TRACE.K_EXECUTION_END, 2300, (0,), graph_id=0),
        ],
    }
    # A future CUPTI importer would write this shape. The provider name and device references are provider-scoped.
    document = {
        "schema": exec_trace_gpu_activity.SCHEMA,
        "provider": {
            "name": "test-provider",
            "provenance": "VENDOR_EXTERNAL",
            "production_representative": False,
            "clock": "CLOCK_MONOTONIC",
            "graph_node_identity": "none",
        },
        "devices": [{"device_ref": 5, "type": "gpu", "gpu_index": 0, "name": "sm-test"}],
        "api_calls": [{
            "id": 1, "name": "cudaGraphLaunch", "thread_id": 3, "stream_id": 1,
            "correlation_ids": {"internal": 20}, "start_ns": 1005, "end_ns": 1006, "graph_launch": True,
        }],
        "activities": [{
            "id": 1, "type": "kernel", "name": "k", "device_ref": 5, "device_refs": [5],
            "queue_id": 0, "stream_id": 1, "start_ns": 1010, "end_ns": 1030,
            "correlation_ids": {"internal": 20},
        }],
    }
    return trace, document


class TestExecTraceGpuActivity(unittest.TestCase):
    def test_affine_clock_alignment(self):
        trace = {
            "strings": {},
            "events": [
                event(TRACE.K_CLOCK_SYNC, 1000, (900, 1002), flags=1),
                event(TRACE.K_CLOCK_SYNC, 2200, (1900, 2202), flags=1),
            ],
        }
        alignment = exec_trace_gpu_activity.clock_alignment(trace, None)

        self.assertEqual(alignment["method"], "clock_sync_affine")
        self.assertEqual(alignment["rate_numerator"], 1200)
        self.assertEqual(alignment["rate_denominator"], 1000)
        self.assertEqual(exec_trace_gpu_activity.external_to_trace_ns(alignment, 1400), 1601)
        self.assertEqual(exec_trace_gpu_activity.external_to_trace_ns(alignment, 800), 881)

    def test_provider_clock_without_anchor_needs_manual_offset(self):
        trace, _ = fixture()
        with self.assertRaises(exec_trace_gpu_activity.ActivityError):
            exec_trace_gpu_activity.clock_alignment(trace, None, "CUPTI_TIMESTAMP")
        alignment = exec_trace_gpu_activity.clock_alignment(trace, 50, "CUPTI_TIMESTAMP")
        self.assertEqual(alignment["method"], "manual_offset")
        self.assertEqual(alignment["external_clock"], "CUPTI_TIMESTAMP")

    def test_span_index_preserves_unique_match_semantics(self):
        spans = [
            {"id": 1, "start_ns": 100, "end_ns": 200, "devices": (1,)},
            {"id": 2, "start_ns": 190, "end_ns": 220, "devices": (1,)},
            {"id": 3, "start_ns": 300, "end_ns": 320, "devices": (1, 2)},
        ]
        index = exec_trace_gpu_activity.SpanIndex(spans, lambda item: item["devices"])

        self.assertEqual(index.unique_containing(150, {1}, 0)["id"], 1)
        self.assertIsNone(index.unique_containing(195, {1}, 0))
        self.assertEqual(index.unique_containing(295, {1}, 5)["id"], 3)
        self.assertEqual(index.unique_containing(310, {1, 2}, 0)["id"], 3)
        self.assertEqual(index.latest_before(250, {1})["id"], 2)

    def test_provider_neutral_merge(self):
        trace, document = fixture()
        original = copy.deepcopy(document)
        result = exec_trace_gpu_activity.merge(trace, document)

        self.assertEqual(document, original)
        self.assertEqual(result["schema"], exec_trace_gpu_activity.MERGED_SCHEMA)
        self.assertFalse(result["provider"]["production_representative"])
        self.assertEqual(result["device_mappings"], [
            {"device_id": 1, "device_ref": 5, "method": "single_active_gpu", "gpu_index": 0}])
        activity = result["gpu_activities"][0]
        self.assertEqual(activity["provenance"], "VENDOR_EXTERNAL")
        self.assertEqual(activity["timing_source"], "test-provider")
        self.assertEqual((activity["external_start_ns"], activity["start_ns"]), (1010, 1111))
        self.assertEqual(activity["correlation"]["relationship"], "ordered_graph_launch")
        self.assertEqual(activity["correlation"]["delphi_gpu_graph_id"], 7)
        self.assertEqual(result["diagnostics"]["correlated_exact"], 0)

    def test_validation_rejects_overclaims(self):
        _, document = fixture()
        document["provider"]["production_representative"] = True
        with self.assertRaises(exec_trace_gpu_activity.ActivityError):
            exec_trace_gpu_activity.validate_activity_document(document)

        _, document = fixture()
        document["activities"][0]["graph_identity_present"] = True
        with self.assertRaises(exec_trace_gpu_activity.ActivityError):
            exec_trace_gpu_activity.validate_activity_document(document)

        _, document = fixture()
        document["activities"][0]["end_ns"] = 1
        with self.assertRaises(exec_trace_gpu_activity.ActivityError):
            exec_trace_gpu_activity.validate_activity_document(document)

    def test_upgrades_legacy_rocm_merge(self):
        legacy = {
            "schema": "delphi-rocm-activity-v1",
            "source": {"delphi_schema_version": 4},
            "device_mappings": [{"device_id": 1, "agent_id": 100, "gpu_index": 0}],
            "external_agents": [{"agent_id": 100, "type": 2, "gpu_index": 0}],
            "gpu_api_calls": [{"id": 1, "operation": "hipGraphLaunch", "start_ns": 1, "end_ns": 2}],
            "gpu_activities": [{"id": 1, "kind": "GPU_MEMCPY", "agent_id": 100, "device_agent_ids": [100],
                                "src_agent_id": 1, "dst_agent_id": 100, "start_ns": 3, "end_ns": 4}],
            "device_intervals": [{"agent_id": 100, "device_id": 1}],
        }
        result = exec_trace_gpu_activity.upgrade_merged(legacy)
        self.assertEqual(result["schema"], exec_trace_gpu_activity.MERGED_SCHEMA)
        self.assertEqual(result["provider"]["provenance"], "VENDOR_EXTERNAL")
        self.assertEqual(result["device_mappings"][0]["device_ref"], 100)
        self.assertEqual(result["external_devices"][0]["type"], "gpu")
        self.assertTrue(result["gpu_api_calls"][0]["graph_launch"])
        activity = result["gpu_activities"][0]
        self.assertEqual((activity["type"], activity["device_refs"], activity["dst_device_ref"]), ("copy", [100], 100))
        self.assertEqual(result["device_intervals"][0]["device_ref"], 100)


if __name__ == "__main__":
    unittest.main()
