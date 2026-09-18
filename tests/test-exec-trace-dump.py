#!/usr/bin/env python3

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "exec-trace" / "exec-trace-dump.py"
SPEC = importlib.util.spec_from_file_location("exec_trace_dump", MODULE_PATH)
exec_trace_dump = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exec_trace_dump)


def event(kind, start_ns=0, end_ns=0, graph_id=1, node_id=0, split_id=0, backend_id=0, device_id=0, flags=0, args=()):
    values = ([end_ns] + list(args)) if kind in exec_trace_dump.SPAN_NAMES else list(args)
    values.extend([0] * (4 - len(values)))
    return {
        "timestamp_ns": start_ns,
        "args": tuple(values),
        "execution_id": 1 if kind in exec_trace_dump.SPAN_NAMES else 0,
        "ubatch_id": 1 if kind in exec_trace_dump.SPAN_NAMES else 0,
        "graph_id": graph_id,
        "node_id": node_id,
        "kind": kind,
        "flags": flags,
        "split_id": split_id,
        "backend_id": backend_id,
        "device_id": device_id,
        "thread_id": 1,
    }


class TestExecTraceDump(unittest.TestCase):
    def test_scheduler_summary_and_chronological_timeline(self):
        start_ns = 1_000_000_000
        events = [
            event(exec_trace_dump.K_GRAPH, graph_id=1, args=(1, 0, 0)),
            event(exec_trace_dump.K_BACKEND, graph_id=1, backend_id=1, device_id=1, args=(1,)),
            event(exec_trace_dump.K_BACKEND, graph_id=1, backend_id=2, device_id=2, args=(2,)),
            event(exec_trace_dump.K_CPU_COMPUTE, start_ns + 8_000_000, start_ns + 12_000_000, split_id=2, backend_id=2, device_id=2, args=(0, 4, 8)),
            event(exec_trace_dump.K_BACKEND_SUBMIT, start_ns + 1_000_000, start_ns + 2_000_000, split_id=1, backend_id=1, device_id=1, args=(0, 0, 4)),
            event(exec_trace_dump.K_COPY_HANDLING, start_ns + 2_000_000, start_ns + 5_000_000, split_id=1, backend_id=1, device_id=1, flags=3, args=(11, 4096, (2 << 32) | 2)),
            event(exec_trace_dump.K_COPY_WAIT_SRC, start_ns + 2_500_000, start_ns + 3_500_000, split_id=1, backend_id=2, device_id=2, flags=4, args=(1, (1 << 32) | 1, 4096)),
            event(exec_trace_dump.K_BACKEND_WAIT, start_ns + 5_000_000, start_ns + 7_000_000, backend_id=1, device_id=1, flags=4, args=(1,)),
        ]
        trace = {"version": 1, "flags": 0, "start_ns": start_ns, "events": events, "strings": {1: "ROCm1", 2: "CPU"}, "trace_stats": {}}

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec_trace_dump.dump(trace, scheduler_summary=True, timeline=True)
        text = output.getvalue()

        self.assertIn("cpu_compute: count=1 host_ms=4.000", text)
        self.assertIn("backend_submit[ROCm1]: count=1 host_ms=1.000", text)
        self.assertIn("backend_wait[CPU]: count=1 host_ms=1.000", text)
        self.assertIn("backend_wait[ROCm1]: count=1 host_ms=2.000", text)
        self.assertIn("copy_handling: count=1 host_ms=3.000", text)
        self.assertLess(text.index("1.000 ms"), text.index("8.000 ms"))
        self.assertIn("BACKEND_SUBMIT begin", text)
        self.assertIn("BACKEND_SUBMIT end", text)

    def test_rejects_inverted_span(self):
        bad_event = event(exec_trace_dump.K_CPU_COMPUTE, 20, 10, backend_id=1, device_id=1)
        with self.assertRaises(exec_trace_dump.TraceError):
            exec_trace_dump.span_duration_ns(bad_event)

    def test_dispatch_fingerprint_and_unknown_implementation(self):
        strings = {
            1: "ROCm1", 2: "blk.0.ffn_up", 3: "MUL_MAT", 4: "F32", 5: "future_path",
            6: "Q4_K", 7: "weights", 8: "F32", 9: "activations",
        }
        events = [
            event(exec_trace_dump.K_GRAPH, graph_id=1, args=(1, 1, 0, 1)),
            event(exec_trace_dump.K_BACKEND, graph_id=1, backend_id=1, device_id=1, args=(1,)),
            event(exec_trace_dump.K_NODE, graph_id=1, node_id=71, backend_id=1, device_id=1, flags=26, args=(0, 2, 3, 4)),
            event(exec_trace_dump.K_NODE_DIMS, graph_id=1, node_id=71, args=(64, 1, 1, 1)),
            event(exec_trace_dump.K_NODE_SOURCE, graph_id=1, node_id=71, flags=6, args=(0, 0, 14, 7)),
            event(exec_trace_dump.K_NODE_SOURCE_DIMS, graph_id=1, node_id=71, flags=0, args=(128, 64, 1, 1)),
            event(exec_trace_dump.K_NODE_SOURCE, graph_id=1, node_id=71, flags=8, args=(1, 0, 0, 9)),
            event(exec_trace_dump.K_NODE_SOURCE_DIMS, graph_id=1, node_id=71, flags=1, args=(128, 1, 1, 1)),
            event(exec_trace_dump.K_NODE_MATMUL, graph_id=1, node_id=71, flags=26, args=(64, 1, 128, 0)),
            event(exec_trace_dump.K_DISPATCH, graph_id=1, node_id=71, backend_id=1, device_id=1,
                  flags=26, args=(999, 5, (123 << 32) | 77, (64 << 32) | 1201)),
        ]
        fingerprint_a = exec_trace_dump.dispatch_fingerprints(events, strings)
        fingerprint_b = exec_trace_dump.dispatch_fingerprints(events, strings)
        self.assertEqual(fingerprint_a, fingerprint_b)

        changed = [dict(item) for item in events]
        changed[-1] = dict(changed[-1], args=(999, 5, (123 << 32) | 77, (65 << 32) | 1201))
        self.assertNotEqual(fingerprint_a, exec_trace_dump.dispatch_fingerprints(changed, strings))

        output = io.StringIO()
        trace = {"version": 2, "flags": 0, "start_ns": 0, "events": events, "strings": strings, "trace_stats": {}}
        with contextlib.redirect_stdout(output):
            exec_trace_dump.dump(trace, dispatches=True)
        text = output.getvalue()
        self.assertIn("dispatch: future_path", text)
        self.assertIn("variant: variant-77", text)
        self.assertIn("shape: M=64 N=1 K=128", text)
        self.assertIn("inputs: src0=Q4_K src1=F32", text)

    def test_gpu_graph_replays_and_devices(self):
        strings = {1: "ROCm1", 2: "ROCm2"}
        events = [
            event(exec_trace_dump.K_BACKEND, graph_id=1, backend_id=1, device_id=1, args=(1,)),
            event(exec_trace_dump.K_BACKEND, graph_id=1, backend_id=2, device_id=2, args=(2,)),
            event(exec_trace_dump.K_GPU_GRAPH, graph_id=1, backend_id=1, device_id=1, flags=1, args=(4, 20, 0)),
            event(exec_trace_dump.K_GPU_GRAPH, graph_id=1, backend_id=1, device_id=1, flags=3, args=(4, 20, 0)),
            event(exec_trace_dump.K_GPU_GRAPH, graph_id=2, backend_id=1, device_id=1, flags=3, args=(4, 20, 0)),
            event(exec_trace_dump.K_GPU_GRAPH, graph_id=2, backend_id=2, device_id=2, flags=4, args=(0, 10, 1)),
        ]
        backend_name = exec_trace_dump.build_backend_names(events, strings)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec_trace_dump.dump_gpu_graphs(events, backend_name)
        text = output.getvalue()
        self.assertIn("graph_exec 4 / ROCm1: nodes=20 capture=1 update=0 replay=2 fallback=0", text)
        self.assertIn("uncaptured / ROCm2: nodes=10 capture=0 update=0 replay=0 fallback=1", text)

    def test_gpu_timing_placement_and_summary(self):
        strings = {1: "ROCm1", 2: "gpu_timing", 3: "op", 4: "blk.0.ffn_up", 5: "MUL_MAT", 6: "F32",
                   7: "gpu_timing_spans_started", 8: "3", 9: "gpu_timing_spans_released"}
        resolved = 1 << 8
        device_event = 2 << 16
        events = [
            event(exec_trace_dump.K_RUN_METADATA, args=(2, 3)),
            event(exec_trace_dump.K_RUN_METADATA, args=(7, 8)),
            event(exec_trace_dump.K_RUN_METADATA, args=(9, 8)),
            event(exec_trace_dump.K_GRAPH, graph_id=1, args=(1, 1, 0, 1)),
            event(exec_trace_dump.K_BACKEND, graph_id=1, backend_id=1, device_id=1, args=(1,)),
            event(exec_trace_dump.K_NODE, graph_id=1, node_id=5, backend_id=1, device_id=1, flags=26, args=(0, 4, 5, 6)),
            event(exec_trace_dump.K_DISPATCH, graph_id=1, node_id=5, backend_id=1, device_id=1, flags=26, args=(5, 0, 1, 0)),
            event(exec_trace_dump.K_GPU_TIMING, 1000, graph_id=1, backend_id=1, device_id=1,
                  flags=1 | resolved | device_event, args=(1100, 0, 5000, (7 << 32) | 3)),
            event(exec_trace_dump.K_GPU_TIMING, 1010, graph_id=1, node_id=5, backend_id=1, device_id=1,
                  flags=3 | resolved | device_event, args=(1020, 100, 2000, (7 << 32) | 1)),
            event(exec_trace_dump.K_GPU_TIMING, 1030, graph_id=1, node_id=5, backend_id=1, device_id=1,
                  flags=3 | (3 << 8) | device_event, args=(1030, 0, 0, 1)),
            event(exec_trace_dump.K_GPU_TIMING_EPOCH, 1000, graph_id=1, backend_id=1, device_id=1,
                  flags=device_event, args=(9000, 7, 2, 1)),
        ]
        spans, epochs = exec_trace_dump.gpu_timing_model(events)
        self.assertEqual(epochs[7]["anchor_lower_ns"], 1000)
        self.assertEqual(epochs[7]["anchor_upper_ns"], 4000)
        self.assertEqual(epochs[7]["placement"], "consistent")
        op_span = spans[1]
        self.assertEqual((op_span["device_start_ns"], op_span["device_end_ns"]), (1100, 3100))
        self.assertEqual(op_span["provenance"], "DEVICE_EVENT")
        self.assertNotIn("device_start_ns", spans[2])

        trace = {"version": 4, "flags": 0, "start_ns": 0, "events": events, "strings": strings, "trace_stats": {}}
        summary = exec_trace_dump.gpu_timing_summary(trace)
        self.assertEqual(summary["mode"], "op")
        self.assertEqual(summary["spans_unreleased"], 0)
        self.assertEqual(summary["status_counts"]["resolved"], 2)
        self.assertEqual(summary["status_counts"]["not_recorded"], 1)
        self.assertEqual(summary["by_op"][0]["op"], "MUL_MAT")
        self.assertEqual(summary["by_op"][0]["implementation"], "GPU_MMQ")
        self.assertEqual(summary["placement_uncertainty_max_ns"], 3000)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec_trace_dump.dump(trace, gpu_timing=True, timeline=True)
        text = output.getvalue()
        self.assertIn("status: resolved=2 unresolved=0 not_recorded=1 failed=0", text)
        self.assertIn("MUL_MAT dispatch=GPU_MMQ [ROCm1]: count=1 device_ms=0.002", text)
        self.assertIn("GPU_TIMING begin scope=eager_op placement=consistent", text)

        # A completion time before the measured device end cannot hold one shared anchor.
        late = [dict(item) for item in events]
        late[-1] = dict(late[-1], args=(2000, 7, 2, 1))
        _, epochs = exec_trace_dump.gpu_timing_model(late)
        self.assertEqual(epochs[7]["placement"], "inconsistent")
        self.assertIsNone(epochs[7]["placement_uncertainty_ns"])


if __name__ == "__main__":
    unittest.main()
