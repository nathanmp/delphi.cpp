#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "exec-trace" / "exec-trace-integrity.py"
SPEC = importlib.util.spec_from_file_location("exec_trace_integrity", MODULE_PATH)
exec_trace_integrity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exec_trace_integrity)


def integrity(**changes):
    result = {field: 0 for field in exec_trace_integrity.INTEGRITY_FIELDS}
    result.update({
        "graph_count": 10,
        "node_count": 100,
        "split_count": 20,
        "graph_sum": 1,
        "assignment_sum": 2,
        "split_sum": 3,
        "copy_count": 4,
        "copy_sum": 5,
        "dispatch_count": 30,
        "dispatch_sum": 6,
        "gpu_graph_event_count": 8,
        "gpu_graph_sum": 7,
        "gpu_graph_replay_count": 6,
    })
    result.update(changes)
    return result


class TestExecTraceIntegrity(unittest.TestCase):
    def test_repeated_runs_are_combined_without_losing_process_records(self):
        identity = {field: 0 for field in exec_trace_integrity.TEST_IDENTITY_FIELDS}
        first = {
            "command": ["bench"], "command_text": "bench", "environment": {},
            "process_wall_ns": 10, "stderr": "first\n",
            "tests": [dict(identity, samples_ns=[1, 2], samples_ts=[10.0, 11.0])],
        }
        second = {
            "command": ["bench"], "command_text": "bench", "environment": {},
            "process_wall_ns": 20, "stderr": "second\n",
            "tests": [dict(identity, samples_ns=[3, 4], samples_ts=[12.0, 13.0])],
        }
        combined = exec_trace_integrity.combine_benchmark_runs([first, second])
        self.assertEqual(combined["tests"][0]["samples_ts"], [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(combined["process_wall_ns"], 30)
        self.assertEqual(combined["processes"], [first, second])

    def test_trace_statistics_are_combined(self):
        first = {
            "event_count": 10, "attempted_event_count": 11, "dropped_event_count": 1,
            "buffer_growth_count": 2, "serialization_duration_ns": 3, "file_bytes": 4,
            "truncated": False, "interned_string_count": 5, "event_buffer_bytes": 100,
        }
        second = {
            "event_count": 20, "attempted_event_count": 22, "dropped_event_count": 2,
            "buffer_growth_count": 4, "serialization_duration_ns": 6, "file_bytes": 8,
            "truncated": True, "interned_string_count": 7, "event_buffer_bytes": 200,
        }
        combined = exec_trace_integrity.combine_trace_stats([first, second])
        self.assertEqual(combined["event_count"], 30)
        self.assertEqual(combined["buffer_growth_count"], 6)
        self.assertEqual(combined["event_buffer_bytes"], 200)
        self.assertTrue(combined["truncated"])

    def test_structure_mismatch_overrides_performance(self):
        baseline = integrity()
        traced = integrity(dispatch_sum=99)
        comparison = exec_trace_integrity.compare_structure(baseline, traced)
        self.assertFalse(comparison["dispatch fingerprint"]["match"])
        self.assertTrue(comparison["graph fingerprint"]["match"])

    def test_performance_classification_uses_medians(self):
        identity = {field: 0 for field in exec_trace_integrity.TEST_IDENTITY_FIELDS}
        baseline = dict(identity, samples_ns=[100, 101, 102], samples_ts=[100.0, 99.0, 98.0])
        traced = dict(identity, samples_ns=[101, 102, 103], samples_ts=[99.0, 98.0, 97.0])
        result = exec_trace_integrity.compare_performance([baseline], [traced])[0]
        self.assertEqual(result["classification"], "acceptable")
        self.assertAlmostEqual(result["slowdown_percent"], 100.0 * (1.0 - 98.0 / 99.0))

    def test_contamination_flags_are_reported(self):
        summary = integrity(build_flags=1, runtime_flags=1 << 3)
        cache = {"values": {"CMAKE_BUILD_TYPE": "Debug", "GGML_SANITIZE_ADDRESS": "ON"}}
        reasons = exec_trace_integrity.contamination_reasons(
            summary, {"ROCPROFILER_LOG": "1"}, cache, ["libroctx64.so"], ["rocprof"])
        joined = "\n".join(reasons)
        self.assertIn("asserts_enabled", joined)
        self.assertIn("GGML_HIP_OP_TRACE", joined)
        self.assertIn("ROCPROFILER_LOG", joined)
        self.assertIn("rocprof", joined)
        self.assertIn("Debug", joined)
        self.assertIn("GGML_SANITIZE_ADDRESS", joined)

    def test_gpu_timing_report_classification(self):
        identity = {field: 0 for field in exec_trace_integrity.TEST_IDENTITY_FIELDS}

        def run(samples):
            return {"environment": {}, "tests": [dict(identity, samples_ns=[1] * len(samples), samples_ts=samples)]}

        def summary(**status):
            counts = {"resolved": 10, "unresolved": 0, "not_recorded": 0, "failed": 0}
            counts.update(status)
            return {"status_counts": counts, "epoch_placement_counts": {"consistent": 4},
                    "spans_unreleased": 0, "spans_recorded": sum(counts.values()), "placement_uncertainty_max_ns": 500}

        def timing(samples, runtime_flags=exec_trace_integrity.GPU_TIMING_RUNTIME_FLAG, **status):
            return {
                "mode": "submit",
                "run": run(samples),
                "integrity": integrity(runtime_flags=runtime_flags),
                "integrity_repeat": integrity(runtime_flags=runtime_flags),
                "stats": {"truncated": False, "dropped_event_count": 0},
                "summaries": [summary(**status)],
            }

        clean = run([100.0, 100.0])
        cache = {"values": {}}
        report = exec_trace_integrity.make_gpu_timing_report(
            clean, clean, integrity(), timing([99.5, 99.5]), cache, [], [])
        self.assertEqual(report["classification"], "low-perturbation")
        self.assertTrue(report["trustworthy"])

        report = exec_trace_integrity.make_gpu_timing_report(
            clean, clean, integrity(), timing([99.5, 99.5], not_recorded=3), cache, [], [])
        self.assertEqual(report["classification"], "low-perturbation, incomplete coverage")

        report = exec_trace_integrity.make_gpu_timing_report(
            clean, clean, integrity(), timing([95.0, 95.0]), cache, [], [])
        self.assertEqual(report["classification"], "perturbative")
        self.assertFalse(report["trustworthy"])

        report = exec_trace_integrity.make_gpu_timing_report(
            clean, clean, integrity(), timing([100.0, 100.0], runtime_flags=0), cache, [], [])
        self.assertEqual(report["classification"], "invalid")
        self.assertIn("GPU timing was not active in the timing run", report["reasons"])


if __name__ == "__main__":
    unittest.main()
