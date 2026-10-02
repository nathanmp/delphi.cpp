#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "exec-trace"
SPEC = importlib.util.spec_from_file_location("exec_trace_spec", TOOLS / "exec-trace-spec.py")
exec_trace_spec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exec_trace_spec)
T = exec_trace_spec.T

DECODE, PREFILL = 1, 2
PROMPT, TARGET, DRAFT = 1, 2, 3


def event(kind, ts, execution_id=0, args=(), flags=0, backend_id=0, node_id=0, graph_id=1):
    values = list(args) + [0] * (4 - len(args))
    return {"timestamp_ns": ts, "args": tuple(values), "execution_id": execution_id, "ubatch_id": execution_id,
            "graph_id": graph_id, "node_id": node_id, "kind": kind, "flags": flags, "split_id": 0,
            "backend_id": backend_id, "device_id": backend_id, "thread_id": 1}


def execution(execution_id, start, end, tokens, phase):
    kind = DECODE if tokens == 1 else PREFILL
    return [event(T.K_EXECUTION_BEGIN, start, execution_id, (tokens,), kind | (phase << 8)),
            event(T.K_EXECUTION_END, end, execution_id)]


def gpu_span(execution_id, epoch, start, duration, backend_id, scope=1, node_id=0):
    # resolved DEVICE_EVENT span placed at `start` by an epoch anchored at `start`
    flags = scope | (1 << 8) | (2 << 16)
    return [event(T.K_GPU_TIMING, start, execution_id, (start, 0, duration, epoch << 32), flags, backend_id, node_id),
            event(T.K_GPU_TIMING_EPOCH, start, execution_id, (start + duration, epoch, 1, 1), 0, backend_id)]


class TestExecTraceSpec(unittest.TestCase):
    def trace(self, events):
        strings = {1: "GPU0", 2: "GPU1", 3: "MUL_MAT", 4: "blk.0.attn_q"}
        events = [event(T.K_GRAPH, 0, args=(1,)),
                  event(T.K_BACKEND, 0, args=(1,), backend_id=1),
                  event(T.K_BACKEND, 0, args=(2,), backend_id=2),
                  event(T.K_NODE, 0, args=(0, 4, 3), node_id=7)] + events
        return {"version": 5, "flags": 0, "start_ns": 0, "events": events, "strings": strings, "trace_stats": {}}

    def test_groups_verify_and_draft_and_cycle(self):
        events = []
        events += execution(1, 0, 10_000_000, 4096, PROMPT)                  # prompt: ignored
        epoch = 1
        t = 20_000_000
        for cycle in range(2):
            for step in range(3):                                            # three 1-token draft steps of 1 ms
                xid = 10 + cycle * 10 + step
                events += execution(xid, t, t + 1_000_000, 1, DRAFT)
                events += gpu_span(xid, epoch, t, 500_000, 2)
                epoch += 1
                t += 1_000_000
            xid = 10 + cycle * 10 + 5                                        # 4-token verify of 6 ms over two devices
            events += execution(xid, t, t + 6_000_000, 4, TARGET)
            events += gpu_span(xid, epoch, t, 2_000_000, 1)
            events += gpu_span(xid, epoch + 1, t + 2_000_000, 3_000_000, 2)
            events += gpu_span(xid, epoch + 2, t, 1_000_000, 1, scope=3, node_id=7)
            events.append(event(T.K_CPU_COMPUTE, t + 5_000_000, xid, (t + 5_500_000,)))
            epoch += 3
            t += 6_000_000
        result = exec_trace_spec.summarize(self.trace(events), top=4, node_pattern=None)

        groups = {(g["phase"], g["tokens"]): g for g in result["groups"]}
        self.assertEqual(set(groups), {("generation_draft", 1), ("generation_target", 4)})
        draft, verify = groups[("generation_draft", 1)], groups[("generation_target", 4)]
        self.assertEqual(draft["executions"], 6)
        self.assertAlmostEqual(draft["host_wall_ms_mean"], 1.0)
        self.assertAlmostEqual(draft["device_busy_ms_mean"]["GPU1"], 0.5)
        self.assertEqual(verify["executions"], 2)
        self.assertAlmostEqual(verify["device_busy_ms_mean"]["GPU0"], 2.0)     # submit spans win over op spans
        self.assertAlmostEqual(verify["device_busy_ms_mean"]["GPU1"], 3.0)
        self.assertAlmostEqual(verify["any_device_busy_ms_mean"], 5.0)
        self.assertAlmostEqual(verify["cpu_compute_ms_mean"], 0.5)
        self.assertEqual(verify["ops_ms_mean"], [{"op": "GPU0 / MUL_MAT", "ms": 1.0}])

        cycle = result["cycle"]
        self.assertEqual(cycle["verify_passes"], 2)
        self.assertAlmostEqual(cycle["draft_executions_per_verify"], 3.0)
        self.assertAlmostEqual(cycle["cycle_wall_ms"], 9.0)                    # verify start to verify start
        self.assertAlmostEqual(cycle["device_busy_ms_per_cycle"]["GPU0"], 2.0)
        self.assertAlmostEqual(cycle["device_busy_ms_per_cycle"]["GPU1"], 4.5)
        self.assertAlmostEqual(cycle["any_device_busy_ms_per_cycle"], 6.5)
        self.assertAlmostEqual(cycle["host_ms_in_decode_per_cycle"]["draft"], 3.0)
        self.assertAlmostEqual(cycle["cpu_compute_ms_per_cycle"], 0.5)
        text = exec_trace_spec.render(result, accepted=2.0)
        self.assertIn("9.00 ms verify to verify", text)
        self.assertIn("3.00 ms per output token", text)

    def test_windows_select_executions(self):
        events = []
        for xid, start in ((1, 0), (2, 10_000_000), (3, 100_000_000)):
            events += execution(xid, start, start + 2_000_000, 3, TARGET)
        result = exec_trace_spec.summarize(self.trace(events), top=4, node_pattern=None, windows_ns=[(0, 50_000_000)])
        self.assertEqual(result["cycle"]["verify_passes"], 2)
        self.assertAlmostEqual(result["cycle"]["cycle_wall_ms"], 10.0)

    def test_node_pattern_groups_operations(self):
        import re
        events = execution(1, 0, 2_000_000, 3, TARGET) + gpu_span(1, 1, 0, 1_000_000, 1, scope=3, node_id=7)
        result = exec_trace_spec.summarize(self.trace(events), top=4, node_pattern=re.compile("attn"))
        self.assertEqual(result["groups"][0]["ops_ms_mean"], [{"op": "GPU0 / [attn]", "ms": 1.0}])

    def test_no_verify_passes(self):
        result = exec_trace_spec.summarize(self.trace(execution(1, 0, 1_000_000, 512, PROMPT)), top=4, node_pattern=None)
        self.assertIsNone(result["cycle"])
        self.assertIn("fewer than two verify passes", exec_trace_spec.render(result, accepted=None))


if __name__ == "__main__":
    unittest.main()
