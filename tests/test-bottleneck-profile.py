#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "bottlenecks" / "profile.py"
SPEC = importlib.util.spec_from_file_location("bottleneck_profile", MODULE_PATH)
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


class TestBottleneckProfile(unittest.TestCase):
    def test_option_override_replaces_short_and_adds_missing_options(self):
        command = ("llama-bench", "-ub", "1024", "-n", "32")
        self.assertEqual(profile._replace_option(command, "-ub", "2048"), ("llama-bench", "-ub", "2048", "-n", "32"))
        self.assertEqual(profile._replace_option(command, "-fa", "on")[-2:], ("-fa", "on"))

    def test_override_does_not_mutate_recipe_and_applies_environment(self):
        recipe = profile.Recipe("example", "example", (profile.BenchArm("plain", ("llama-bench", "-ub", "512")),))
        changed = profile.apply_overrides(recipe, [("plain", "-ub", "2048")], {"GGML_TEST": "1"})
        self.assertEqual(recipe.arms[0].command, ("llama-bench", "-ub", "512"))
        self.assertEqual(changed.arms[0].command, ("llama-bench", "-ub", "2048"))
        self.assertEqual(changed.arms[0].env["GGML_TEST"], "1")

    def test_bindir_replaces_only_llama_executables(self):
        recipe = profile.Recipe("example", "example", (
            profile.BenchArm("bench", ("old/bin/llama-bench",)),
            profile.ServerArm("server", ("old/bin/llama-server",), ("python3", "speed_bench.py")),
        ))
        changed = profile.apply_bindir(recipe, Path("/tmp/new-build/bin"))
        self.assertEqual(changed.arms[0].command[0], "/tmp/new-build/bin/llama-bench")
        self.assertEqual(changed.arms[1].server_command[0], "/tmp/new-build/bin/llama-server")
        self.assertEqual(changed.arms[1].client_command[0], "python3")

    def test_family_rules_require_all_configured_matches(self):
        rule = profile.FamilyRule("routed", operations=("MUL_MAT_ID",), source_contains=("exps",))
        matching = {"op": "MUL_MAT_ID", "name": "out", "sources": {0: {"name": "ffn_exps"}}}
        self.assertTrue(rule.matches(matching))
        self.assertFalse(rule.matches({"op": "MUL_MAT", "name": "out", "sources": {0: {"name": "ffn_exps"}}}))

    def test_summary_marks_device_families_non_additive(self):
        result = {
            "recipe": "example", "created_utc": "2026-09-22T00:00:00Z", "repository": {"commit": "abc", "tree": "clean"}, "arms": [{
                "name": "plain", "type": "llama-bench", "metrics": {"kind": "llama-bench", "tests": []}, "trace": {"mode": "op", "integrity": "not-run", "analysis": {"decode": {
                    "executions": 10,
                    "wall_ms_per_token": {"median": 10.0},
                    "cpu_compute_ms": {"median": 1.0}, "copy_handling_ms": {"median": 2.0},
                    "host_submit_ms": {"median": 3.0}, "device_submit_ms": {"median": 4.0},
                    "diagnostic_op_family_ms": {"attention": {"median": 0.5}},
                }}}}]
        }
        summary = profile.render_summary(result)
        self.assertIn("diagnostic, non-additive", summary)
        self.assertIn("instrumented window 10.000 ms/token", summary)
        self.assertIn("<=25.0%", summary)

    def test_speed_bench_parser_uses_saved_json_field_names(self):
        metrics = profile._speed_bench_metrics({"completed_samples": 1, "failed_samples": 0, "summary": [{
            "category": "overall", "avg_prompt_t_s": 100.0, "avg_pred_t_s": 20.0, "avg_latency": 3.0, "accept_rate": 0.75,
        }]})
        self.assertEqual(metrics["prompt_t_s"], 100.0)
        self.assertEqual(metrics["predicted_t_s"], 20.0)
        self.assertEqual(metrics["accept_rate"], 0.75)

    def test_prefill_parser_keeps_requested_and_realized_lengths(self):
        metrics = profile._prefill_metrics({
            "target_tokens": 4096, "input_tokens": 4096, "prompt_tokens": 4097,
            "prompt_ms": 2000.0, "prompt_t_s": 2048.5, "latency_s": 2.1,
            "predicted_tokens": 1,
        })
        self.assertEqual(metrics["kind"], "prefill")
        self.assertEqual(metrics["target_tokens"], 4096)
        self.assertEqual(metrics["prompt_tokens"], 4097)
        self.assertEqual(metrics["prompt_t_s"], 2048.5)

    def test_speed_bench_summary_uses_end_to_end_wall_for_ceilings(self):
        result = {
            "recipe": "example", "created_utc": "2026-09-22T00:00:00Z", "repository": {"commit": "abc", "tree": "clean"}, "arms": [{
                "name": "plain", "type": "llama-server + speed-bench",
                "metrics": {"kind": "speed-bench", "prompt_t_s": 100.0, "predicted_t_s": 20.0, "latency_s": 3.0, "accept_rate": None},
                "trace": {"mode": "submit", "integrity": "not-run", "analysis": {"decode": {
                    "executions": 10,
                    "wall_ms_per_token": {"median": 10.0},
                    "cpu_compute_ms": {"median": 1.0}, "copy_handling_ms": {"median": 2.0},
                    "host_submit_ms": {"median": 3.0}, "device_submit_ms": {"median": 4.0},
                    "diagnostic_op_family_ms": {},
                }}}}]
        }
        summary = profile.render_summary(result)
        self.assertIn("TG wall 50.000 ms/token", summary)
        self.assertIn("<=2.0%", summary)

    def test_summary_includes_arm_notes(self):
        result = {
            "recipe": "example", "created_utc": "2026-09-22T00:00:00Z", "repository": {"commit": "abc", "tree": "clean"},
            "arms": [{"name": "diagnostic", "type": "llama-bench", "notes": ["graphs are disabled"], "metrics": {"kind": "llama-bench", "tests": []}}],
        }
        self.assertIn("Note: graphs are disabled", profile.render_summary(result))

    def test_summary_rejects_low_sample_trace_breakdown(self):
        result = {
            "recipe": "example", "created_utc": "2026-09-22T00:00:00Z", "repository": {"commit": "abc", "tree": "clean"}, "arms": [{
                "name": "plain", "type": "llama-bench", "metrics": {"kind": "llama-bench", "tests": []}, "trace": {"mode": "submit", "integrity": "not-run", "analysis": {"decode": {
                    "executions": 3, "wall_ms_per_token": {"median": 10.0},
                    "cpu_compute_ms": {"median": 1.0}, "copy_handling_ms": {"median": 2.0}, "host_submit_ms": {"median": 3.0}, "device_submit_ms": {"median": 4.0},
                }}}}]
        }
        summary = profile.render_summary(result)
        self.assertIn("insufficient steady decode executions", summary)
        self.assertNotIn("| measured span |", summary)

    def test_summary_renders_prefill_when_decode_trace_is_short(self):
        span = {
            "executions": 2, "wall_ms_per_token": {"median": 4.0},
            "cpu_compute_ms": {"median": 1.0}, "copy_handling_ms": {"median": 0.5},
            "host_submit_ms": {"median": 0.25}, "device_submit_ms": {"median": 2.0},
        }
        result = {
            "recipe": "example", "created_utc": "2026-09-22T00:00:00Z", "repository": {"commit": "abc", "tree": "clean"}, "arms": [{
                "name": "plain", "type": "llama-server + speed-bench",
                "metrics": {"kind": "speed-bench", "prompt_t_s": 10.0, "predicted_t_s": 20.0, "latency_s": 3.0, "accept_rate": None},
                "extra_measurements": [{"name": "prefill-4096", "metrics": {"kind": "prefill", "target_tokens": 4096, "prompt_tokens": 4097, "prompt_t_s": 250.0, "latency_s": 16.5}}],
                "trace": {"mode": "submit", "integrity": "not-run", "analysis": {"decode": span, "prefill": {**span, "tokens": {"total": 4096}}}},
            }],
        }
        summary = profile.render_summary(result)
        self.assertIn("Prefill (prefill-4096): target 4096 tokens", summary)
        self.assertIn("Prefill trace: 2 selected executions, 4096 trace tokens", summary)
        self.assertIn("measured prefill span", summary)


if __name__ == "__main__":
    unittest.main()
