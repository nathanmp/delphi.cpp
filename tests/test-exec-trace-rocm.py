#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "exec-trace" / "exec-trace-rocm.py"
SPEC = importlib.util.spec_from_file_location("exec_trace_rocm", MODULE_PATH)
exec_trace_rocm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exec_trace_rocm)


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


def api(operation, correlation_id, start_ns, end_ns, stream_id=1):
    return {
        "kind": 1,
        "operation": operation,
        "correlation_id": {"internal": correlation_id, "external": 0, "ancestor": 0},
        "start_timestamp": start_ns,
        "end_timestamp": end_ns,
        "thread_id": 7,
        "stream_id": {"handle": stream_id},
    }


def kernel(correlation_id, agent_id, start_ns, end_ns, dispatch_id, graph_exec_id=0, graph_node_id=0):
    return {
        "kind": 2,
        "operation": 0,
        "correlation_id": {"internal": correlation_id, "external": 0},
        "start_timestamp": start_ns,
        "end_timestamp": end_ns,
        "stream_id": {"handle": 1},
        "graph_exec_id": graph_exec_id,
        "graph_node_id": graph_node_id,
        "dispatch_info": {
            "agent_id": {"handle": agent_id},
            "queue_id": {"handle": agent_id + 1},
            "kernel_id": 1,
            "dispatch_id": dispatch_id,
        },
    }


def fixture():
    trace = {
        "version": 4,
        "flags": 0,
        "start_ns": 900,
        "strings": {1: "ROCm0", 2: "gfx-test", 3: "GPU A", 4: "ROCm1", 5: "GPU B"},
        "trace_stats": {},
        "events": [
            event(exec_trace_rocm.TRACE.K_CLOCK_SYNC, 1000, (900, 1002), flags=1, execution_id=0, graph_id=0),
            event(exec_trace_rocm.TRACE.K_EXECUTION_BEGIN, 1050, flags=1, graph_id=0),
            event(exec_trace_rocm.TRACE.K_BACKEND, 1055, (1, 0, 2, 3), flags=1, backend_id=1, device_id=1),
            event(exec_trace_rocm.TRACE.K_BACKEND, 1055, (4, 0, 2, 5), flags=1, backend_id=2, device_id=2),
            event(exec_trace_rocm.TRACE.K_BACKEND_SUBMIT, 1100, (1150, 0, 0, 10),
                  split_id=1, backend_id=1, device_id=1),
            event(exec_trace_rocm.TRACE.K_DISPATCH, 1105, (3, 0, 0, 0), flags=26,
                  node_id=11, split_id=1, backend_id=1, device_id=1),
            event(exec_trace_rocm.TRACE.K_COPY_HANDLING, 1120, (1140, 12, 4096, (2 << 32) | 2), flags=3,
                  node_id=11, split_id=1, backend_id=1, device_id=1),
            event(exec_trace_rocm.TRACE.K_BACKEND_WAIT, 1160, (1180, 1), flags=1,
                  backend_id=1, device_id=1),
            event(exec_trace_rocm.TRACE.K_BACKEND_SUBMIT, 2100, (2150, 0, 10, 20),
                  split_id=2, backend_id=2, device_id=2),
            event(exec_trace_rocm.TRACE.K_DISPATCH, 2103, (3, 0, 0, 0), flags=26,
                  node_id=21, split_id=2, backend_id=2, device_id=2),
            event(exec_trace_rocm.TRACE.K_GPU_GRAPH, 2105, (7, 4, 0), flags=3,
                  split_id=2, backend_id=2, device_id=2),
            event(exec_trace_rocm.TRACE.K_EXECUTION_END, 2300, (0,), graph_id=0),
        ],
    }

    empty = {"kind": "NONE", "operations": []}
    process = {
        "metadata": {"pid": 42, "init_time": 800, "fini_time": 4000, "command": ["test"]},
        "agents": [
            {"id": {"handle": 10}, "type": 1, "gpu_index": -1, "name": "CPU"},
            {"id": {"handle": 100}, "type": 2, "gpu_index": 0, "node_id": 2,
             "name": "gfx-test", "product_name": "GPU A"},
            {"id": {"handle": 200}, "type": 2, "gpu_index": 1, "node_id": 3,
             "name": "gfx-test", "product_name": "GPU B"},
        ],
        "kernel_symbols": [
            {"kernel_id": 1, "formatted_kernel_name": "test_kernel"},
        ],
        "strings": {"buffer_records": [
            empty,
            {"kind": "HIP_RUNTIME_API", "operations": ["hipLaunchKernel", "hipGraphLaunch", "hipMemcpyAsync"]},
            {"kind": "KERNEL_DISPATCH", "operations": []},
            {"kind": "MEMORY_COPY", "operations": ["NONE", "MEMORY_COPY_HOST_TO_DEVICE"]},
        ]},
        "buffer_records": {
            "hip_api": [
                api(0, 10, 1007, 1008),
                api(2, 30, 1020, 1021),
                api(1, 20, 2007, 2008),
            ],
            "kernel_dispatch": [
                kernel(10, 100, 1010, 1020, 1),
                kernel(40, 100, 1065, 1070, 2),
                kernel(20, 200, 2010, 2020, 3, graph_exec_id=9, graph_node_id=3),
                kernel(99, 200, 3000, 3010, 4),
            ],
            "memory_copy": [{
                "kind": 3,
                "operation": 1,
                "correlation_id": {"internal": 30, "external": 0},
                "start_timestamp": 1022,
                "end_timestamp": 1030,
                "src_agent_id": {"handle": 10},
                "dst_agent_id": {"handle": 100},
                "bytes": 4096,
                "stream_id": {"handle": 1},
            }],
        },
    }
    return trace, process


class TestExecTraceRocm(unittest.TestCase):
    def test_repairs_invalid_utf8_in_rocprofiler_string_payloads(self):
        document, replacements = exec_trace_rocm.parse_rocprofiler_json(b'{"fname":"\xab\xcb"}')

        self.assertEqual(document["fname"], "\ufffd\ufffd")
        self.assertEqual(replacements, 2)

    def test_rocprofiler_is_a_vendor_external_provider(self):
        _, process = fixture()
        document = exec_trace_rocm.normalize_rocprofiler(process)
        exec_trace_rocm.ACTIVITY.validate_activity_document(document)

        provider = document["provider"]
        self.assertEqual(document["schema"], "delphi-gpu-activity-v1")
        self.assertEqual((provider["name"], provider["provenance"]), ("rocprofiler-sdk", "VENDOR_EXTERNAL"))
        self.assertFalse(provider["production_representative"])
        self.assertEqual(provider["graph_node_identity"], "provider_reported")
        self.assertEqual({item["device_ref"]: item["type"] for item in document["devices"]}, {10: "cpu", 100: "gpu", 200: "gpu"})
        self.assertEqual([item["name"] for item in document["api_calls"] if item["graph_launch"]], ["hipGraphLaunch"])
        copy = next(item for item in document["activities"] if item["type"] == "copy")
        self.assertEqual((copy["device_refs"], copy["src_device_ref"], copy["bytes"]), ([100], 10, 4096))

        del process["buffer_records"]["kernel_dispatch"][2]["graph_exec_id"]
        for record in process["buffer_records"]["kernel_dispatch"]:
            record.pop("graph_exec_id", None)
            record.pop("graph_node_id", None)
        self.assertEqual(exec_trace_rocm.normalize_rocprofiler(process)["provider"]["graph_node_identity"], "none")

    def test_clock_device_and_activity_correlation(self):
        trace, process = fixture()
        result = exec_trace_rocm.merge(trace, process)

        self.assertEqual(result["schema"], "delphi-gpu-activity-merged-v1")
        self.assertEqual(result["provider"]["provenance"], "VENDOR_EXTERNAL")
        self.assertEqual(result["clock_alignment"]["offset_ns"], 101)
        self.assertEqual(result["clock_alignment"]["uncertainty_ns"], 1)
        self.assertEqual([(item["device_id"], item["gpu_index"]) for item in result["device_mappings"]],
                         [(1, 0), (2, 1)])

        by_correlation = {item["correlation_ids"]["internal"]: item for item in result["gpu_activities"]}
        self.assertTrue(all(item["provenance"] == "VENDOR_EXTERNAL" for item in result["gpu_activities"]))
        self.assertEqual(by_correlation[10]["correlation"]["relationship"], "ordered_within_dispatch")
        self.assertEqual(by_correlation[30]["correlation"]["relationship"], "api_within_copy")
        self.assertEqual(by_correlation[40]["correlation"]["relationship"], "time_window_candidate")
        self.assertEqual(by_correlation[20]["correlation"]["relationship"], "ordered_graph_launch")
        self.assertEqual(by_correlation[20]["correlation"]["delphi_gpu_graph_id"], 7)
        self.assertEqual(by_correlation[20]["graph_exec_id"], 9)
        self.assertEqual(by_correlation[20]["graph_node_id"], 3)
        self.assertEqual(by_correlation[99]["correlation"]["relationship"], "unattributed")

        self.assertEqual(result["diagnostics"]["external_gpu_records"], 5)
        self.assertEqual(result["diagnostics"]["correlated_exact"], 0)
        self.assertEqual(result["diagnostics"]["correlated_heuristic"], 4)
        self.assertEqual(result["diagnostics"]["unattributed"], 1)
        self.assertGreater(result["host_wait_overlaps"][0]["gpu_busy_overlap_ns"], 0)

    def test_explicit_device_map_and_missing_clock_anchor(self):
        trace, process = fixture()
        trace["events"] = [item for item in trace["events"] if item["kind"] != exec_trace_rocm.TRACE.K_CLOCK_SYNC]
        with self.assertRaises(exec_trace_rocm.MergeError):
            exec_trace_rocm.merge(trace, process, {1: 1, 2: 0})

        result = exec_trace_rocm.merge(trace, process, {1: 1, 2: 0}, manual_offset_ns=100)
        self.assertEqual(result["clock_alignment"]["method"], "manual_offset")
        self.assertEqual([(item["device_id"], item["gpu_index"]) for item in result["device_mappings"]],
                         [(1, 1), (2, 0)])

    def test_rejects_inverted_external_activity(self):
        trace, process = fixture()
        process["buffer_records"]["kernel_dispatch"][0]["end_timestamp"] = 1
        with self.assertRaises(exec_trace_rocm.MergeError):
            exec_trace_rocm.merge(trace, process)


if __name__ == "__main__":
    unittest.main()
