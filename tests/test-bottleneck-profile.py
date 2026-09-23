#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
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
        self.assertIn("Do not add rows or ceilings", summary)
        self.assertIn("traced llama_decode host wall 10.000 median ms/trace token", summary)
        self.assertIn("<=25.0%", summary)

    def test_speed_bench_parser_uses_saved_json_field_names(self):
        metrics = profile._speed_bench_metrics({"completed_samples": 1, "failed_samples": 0, "summary": [{
            "category": "overall", "avg_prompt_t_s": 100.0, "avg_pred_t_s": 20.0, "avg_latency": 3.0, "accept_rate": 0.75,
        }], "results": [{"ok": True, "prompt_tokens": 56, "completion_tokens": 1024, "prompt_ms": 500.0, "predicted_ms": 2000.0, "request_windows_raw_ns": [[100, 200]]}]})
        self.assertEqual(metrics["prompt_t_s"], 100.0)
        self.assertEqual(metrics["predicted_t_s"], 20.0)
        self.assertEqual(metrics["accept_rate"], 0.75)
        self.assertEqual(metrics["prompt_tokens"], 56)
        self.assertEqual(metrics["completion_tokens"], 1024)
        self.assertEqual(metrics["request_windows_raw_ns"], [[100, 200]])

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

    def test_prefill_trace_selection_uses_final_complete_sequence(self):
        info = {
            1: {"kind": "prefill", "tokens": 2},
            3: {"kind": "prefill", "tokens": 2048},
            4: {"kind": "prefill", "tokens": 2048},
            5: {"kind": "prefill", "tokens": 4},
            2: {"kind": "decode", "tokens": 1},
            6: {"kind": "decode", "tokens": 1},
        }
        self.assertEqual(profile._prefill_execution_ids({3, 4, 5, 6}, info, 128, 4096), {3, 4, 5})

    def test_prefill_trace_selection_uses_request_clock_for_dspark(self):
        info = {
            1: {"kind": "prefill", "tokens": 5, "start_ns": 50},
            2: {"kind": "prefill", "tokens": 2048, "start_ns": 150},
            3: {"kind": "prefill", "tokens": 512, "start_ns": 200},
        }
        self.assertEqual(profile._prefill_execution_ids({1, 2, 3}, info, 128, 4096, (100, 300)), {2, 3})

    def test_decode_selection_includes_multitoken_target_and_draft_batches(self):
        info = {
            1: {"kind": "decode", "phase": "generation_target", "tokens": 1, "start_ns": 50},
            2: {"kind": "prefill", "phase": "generation_target", "tokens": 5, "start_ns": 150},
            3: {"kind": "prefill", "phase": "generation_draft", "tokens": 4, "start_ns": 175},
            4: {"kind": "prefill", "phase": "prompt", "tokens": 101, "start_ns": 180},
        }
        selected, phase_selected = profile._decode_execution_ids({1, 2, 3, 4}, info, ((100, 200),))
        self.assertTrue(phase_selected)
        self.assertEqual(selected, {2, 3})

    def test_diagnostic_discovers_groups_without_recipe_rules(self):
        base = {"name": "plain", "type": "llama-server + speed-bench", "metrics": {"kind": "speed-bench", "prompt_t_s": 10.0, "predicted_t_s": 100.0, "latency_s": 1.0, "accept_rate": None}}
        diagnostic = {
            "name": "plain-op-diagnostic", "type": "llama-server + speed-bench", "metrics": {"kind": "speed-bench"},
            "trace": {"mode": "op", "integrity": "not-run", "analysis": {
                "decode": {"executions": 20, "diagnostic_by_op_ms": {"NEW_OP": {"median": 1.0}}, "diagnostic_by_node_role_ms": {"NEW_OP / custom-*": {"median": 1.0}}},
                "prefill": {"executions": 2, "aggregate_per_client_token_ms": {"diagnostic_by_op_ms": {"NEW_OP": 2.0}, "diagnostic_by_node_role_ms": {"NEW_OP / custom-*": 2.0}}},
                "gpu_timing_status_counts": {"resolved": 20},
            }},
        }
        result = {"recipe": "example", "created_utc": "2026-09-22T00:00:00Z", "repository": {"commit": "abc", "tree": "clean"}, "arms": [base, diagnostic]}
        summary = profile.render_summary(result)
        self.assertIn("conditional max throughput gain", summary)
        self.assertIn("| NEW_OP | 1.000 | <=11.1% |", summary)
        self.assertIn("| NEW_OP / custom-* | 1.000 | <=11.1% |", summary)
        self.assertIn("screening estimates, not proven production ceilings", summary)

    def test_role_normalization_is_not_specific_to_known_ops(self):
        self.assertEqual(profile._node_role({"op": "FUTURE_OP", "name": "future_mix-19"}), "FUTURE_OP / future_mix-*")
        self.assertEqual(profile._node_role({"op": "MUL_MAT", "name": "node_7", "sources": {0: {"name": "blk.23.ffn_gate.weight"}}}), "MUL_MAT / src blk.*.ffn_gate.weight")
        self.assertEqual(profile._node_role({"op": "ADD", "name": "node_7", "type": "f32", "dims": (48, 2, 1, 1), "sources": {0: {"name": " (view)"}, 1: {"name": " (view)"}}}), "ADD / unnamed f32 48x2 src:view+view")

    def test_diagnostic_cutoff_bounds_each_omitted_group(self):
        lines = []
        profile._render_diagnostic_groups(lines, "By operation", {"A": 2.0, "B": 1.0, "C": 0.5}, 10.0, True, limit=2)
        self.assertIn("The 1 omitted group is <= 0.500 diagnostic ms/token (<=5.3% conditional gain)", lines[-1])

    def test_speculative_diagnostic_renders_aggregate_gpu_and_cpu_groups(self):
        base = {"name": "mtp-synthetic", "type": "llama-server + speed-bench", "metrics": {"kind": "speed-bench", "predicted_t_s": 100.0}}
        aggregate = {
            "diagnostic_by_op_ms": {"MUL_MAT": 2.0},
            "diagnostic_cpu_by_op_ms": {"MUL_MAT_ID": 4.0},
        }
        diagnostic = {
            "name": "mtp-synthetic-op-diagnostic", "type": "llama-server + speed-bench", "metrics": {"kind": "speed-bench"},
            "trace": {"analysis": {"decode": {"executions": 400, "aggregate_per_client_token_ms": aggregate}, "prefill": {"executions": 0}, "gpu_timing_status_counts": {"resolved": 1}}},
        }
        result = {"recipe": "example", "created_utc": "2026-09-22T00:00:00Z", "repository": {"commit": "abc", "tree": "clean"}, "arms": [base, diagnostic]}
        summary = profile.render_summary(result)
        self.assertIn("| MUL_MAT | 2.000 | <=25.0% |", summary)
        self.assertIn("CPU by GGML operation", summary)
        self.assertIn("| MUL_MAT_ID | 4.000 | <=66.7% |", summary)

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
        self.assertIn("TG wall 50.000 ms/output token", summary)
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
        self.assertIn("insufficient steady executions", summary)
        self.assertNotIn("| measured span |", summary)

    def test_summary_withholds_partial_speculative_decode_trace(self):
        result = {
            "recipe": "example", "created_utc": "2026-09-22T00:00:00Z", "repository": {"commit": "abc", "tree": "clean"}, "arms": [{
                "name": "mtp", "type": "llama-server + speed-bench", "server_command": ["llama-server", "--spec-type", "draft-mtp"],
                "metrics": {"kind": "speed-bench", "prompt_t_s": 100.0, "predicted_t_s": 40.0, "latency_s": 3.0},
                "trace": {"mode": "submit", "integrity": "not-run", "analysis": {"decode": {
                    "executions": 100, "wall_ms_per_token": {"median": 0.4},
                    "cpu_compute_ms": {"median": 0.1}, "copy_handling_ms": {"median": 0.2},
                    "host_submit_ms": {"median": 0.3}, "device_submit_ms": {"median": 1.0},
                }}},
            }],
        }
        summary = profile.render_summary(result)
        self.assertIn("legacy trace lacks phase tags", summary)
        self.assertNotIn("| measured span |", summary)

    def test_union_duration_counts_overlaps_once(self):
        self.assertEqual(profile._union_duration_ns([(0, 10), (5, 15), (20, 30)]), 25)
        self.assertEqual(profile._union_duration_ns([(0, 10), (10, 20)]), 20)
        self.assertEqual(profile._union_duration_ns([]), 0)

    def test_pipeline_parallelism_reads_enable_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.stderr.log"
            path.write_text("llama_context: pipeline parallelism enabled\n")
            self.assertEqual(profile._pipeline_parallelism(path)["state"], "enabled")
            path.write_text("pipeline parallelism enabled\ncompute buffer allocation failed, retrying without pipeline parallelism\n")
            self.assertEqual(profile._pipeline_parallelism(path)["state"], "enabled_then_disabled")
            path.write_text("nothing here\n")
            self.assertEqual(profile._pipeline_parallelism(path)["state"], "not_enabled")

    def test_summary_renders_unaccounted_resource_and_counters(self):
        result = {
            "recipe": "example", "created_utc": "2026-09-22T00:00:00Z", "repository": {"commit": "abc", "tree": "clean"}, "arms": [{
                "name": "plain", "type": "llama-server + speed-bench",
                "pipeline_parallelism": {"state": "enabled_then_disabled"},
                "metrics": {"kind": "speed-bench", "prompt_t_s": 100.0, "predicted_t_s": 20.0, "latency_s": 3.0, "accept_rate": None},
                "trace": {"mode": "submit", "integrity": "not-run", "analysis": {"decode": {
                    "executions": 100,
                    "wall_ms_per_token": {"median": 50.0},
                    "cpu_compute_ms": {"median": 1.0}, "copy_handling_ms": {"median": 2.0},
                    "host_submit_ms": {"median": 3.0}, "device_submit_ms": {"median": 4.0},
                    "aggregate_per_client_token_ms": {
                        "wall_ms": 50.0, "cpu_compute_ms": 10.0, "copy_handling_ms": 5.0,
                        "host_submit_ms": 2.0, "device_submit_ms": 30.0, "unaccounted_ms": 1.5,
                        "resource_busy_ms": {"cpu": 10.0, "device": 30.0, "device_union": 30.0, "combined": 35.0, "overlap": 5.0, "idle": 15.0},
                        "device_busy_by_backend_ms": {"ROCm3": 30.0},
                        "counters_per_client_token": {"offload_measurements": 0.0, "offload_decisions": 0.8, "copies": 45.0, "splits": 90.0, "graph_replay": 45.3},
                    },
                }}},
            }],
        }
        summary = profile.render_summary(result)
        self.assertIn("Pipeline parallelism: enabled, then disabled after a compute buffer allocation failure.", summary)
        self.assertIn("| unaccounted host time | 1.500 |", summary)
        self.assertIn("| resource | busy ms/output token | utilization |", summary)
        self.assertIn("| device ROCm3 | 30.000 | 60.000% |", summary)
        self.assertIn("Recurring work per output token:", summary)
        self.assertIn("copies 45.000", summary)

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
                "trace": {"mode": "submit", "integrity": "not-run", "analysis": {"decode": span, "prefill": {**span, "tokens": {"total": 4096}, "aggregate_per_client_token_ms": {"wall_ms": 4.5, "cpu_compute_ms": 1.1, "copy_handling_ms": 0.6, "host_submit_ms": 0.3, "device_submit_ms": 2.2}}}},
            }],
        }
        summary = profile.render_summary(result)
        self.assertIn("Bulk prefill (prefill-4096): target 4096 tokens", summary)
        self.assertIn("Prefill trace: 2 selected executions, 4096 trace-token executions", summary)
        self.assertIn("aggregate ms/client token", summary)


if __name__ == "__main__":
    unittest.main()
