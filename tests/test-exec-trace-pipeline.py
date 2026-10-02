#!/usr/bin/env python3

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "exec-trace"
SPEC = importlib.util.spec_from_file_location("exec_trace_pipeline", TOOLS / "exec-trace-pipeline.py")
exec_trace_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exec_trace_pipeline)


class TestExecTracePipeline(unittest.TestCase):
    def test_concurrency_counts_overlapping_devices(self):
        busy = exec_trace_pipeline.concurrency({"A": [(0, 10), (5, 12)], "B": [(8, 20)]}, 0, 25)
        self.assertEqual(busy[0], 5)    # 20..25
        self.assertEqual(busy[1], 16)   # 0..8 on A, 12..20 on B
        self.assertEqual(busy[2], 4)    # 8..12
        self.assertEqual(sum(busy.values()), 25)

    def test_concurrency_clips_to_window(self):
        busy = exec_trace_pipeline.concurrency({"A": [(0, 100)]}, 40, 60)
        self.assertEqual(dict(busy), {0: 0, 1: 20})

    def test_main_reports_ubatches(self):
        T = exec_trace_pipeline.T
        spec = importlib.util.spec_from_file_location("test_exec_trace_spec", Path(__file__).with_name("test-exec-trace-spec.py"))
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        events = helpers.execution(1, 0, 10_000_000, 4096, helpers.PROMPT)
        for ub, (start, backend) in enumerate([(1_000_000, 1), (3_000_000, 2), (3_000_000, 1)]):
            span = helpers.gpu_span(1, ub + 1, start, 2_000_000, backend)
            for item in span:
                item["ubatch_id"] = 7 + (ub == 2)
            events += span
        trace = helpers.TestExecTraceSpec().trace(events)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.trace"
            write_trace(T, trace, path)
            output = io.StringIO()
            argv, sys.argv = sys.argv, ["exec-trace-pipeline.py", str(path), "--min-tokens", "1024"]
            try:
                with contextlib.redirect_stdout(output):
                    self.assertEqual(exec_trace_pipeline.main(), 0)
            finally:
                sys.argv = argv
        text = output.getvalue()
        self.assertIn("== execution 1: 4096 tokens, phase prompt", text)
        self.assertIn("ubatch     7 GPU0     start       1.0 end       3.0 busy      2.0", text)
        self.assertIn("ubatch     8 GPU0     start       3.0 end       5.0 busy      2.0", text)
        self.assertIn("2: 2.0 ms", text)


def write_trace(T, trace, path):
    strings = sorted(trace["strings"].items())
    string_bytes = sum(T.STRING_HEADER.size + len(text.encode()) for _, text in strings)
    header_size = T.HEADER.size + T.HEADER_V3_STATS.size
    data = bytearray(T.HEADER.pack(b"GEXECTR", 5, header_size, T.EVENT.size, 0, 0, len(trace["events"]), len(strings), string_bytes))
    data += T.HEADER_V3_STATS.pack(len(trace["events"]), len(trace["events"]), 0, 0, 0)
    for e in trace["events"]:
        data += T.EVENT.pack(e["timestamp_ns"], *e["args"], e["execution_id"], e["ubatch_id"], e["graph_id"], e["node_id"],
                             e["kind"], e["flags"], e["split_id"], e["backend_id"], e["device_id"], e["thread_id"])
    for string_id, text in strings:
        data += T.STRING_HEADER.pack(string_id, len(text.encode())) + text.encode()
    path.write_bytes(bytes(data))


if __name__ == "__main__":
    unittest.main()
