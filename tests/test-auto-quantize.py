#!/usr/bin/env python3
"""Offline advisor invariants; no inference or GPU is needed."""
import copy
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/auto-quantize"))
from config import QuantError, capacity, devices, load_config
from model import classify, merge_imatrices, read_model, tensor_bytes
from search import dominates, memory, pareto, search
from timing import Predictor, digest, operation_times, signature
from auto_quantize import verify_output
import numpy as np
from gguf import GGUFReader, GGUFWriter


def cfg(**changes):
    result = {"devices": {"ROCm1": 1 << 30, "RAM": 1 << 30}, "reserve": {}, "cache_mib": 0,
              "primary_qtypes": ["Q2_K", "Q4_0"], "other_qtypes": ["Q6_K"], "pins": {}, "preserve_regex": [],
              "preserve_below_mib": 0, "tie_gate_up": True,
              "runtime": {"env": {}, "short_tokens": 8, "long_tokens": 128, "ubatch": 128},
              "importance": {"precision": .25, "decode": .25, "short": .25, "long": .25},
              "search": {"beam": 16, "evaluations": 1000}, "bounds": {}, "timing": {},
              "multi_gpu_speedup_percent": {"decode": 0, "short": 0, "long": 0}}
    result.update(changes)
    return result


def tensor(name, shape=(256, 256, 8)):
    return {"name": name, "shape": list(shape), "source_dtype": "BF16", "source_bytes": 2 * np.prod(shape).item()}


def info():
    return {"architecture": "olmoe", "n_expert": 8, "top_k": 2, "tensors": [
        tensor("blk.0.ffn_up_exps.weight"), tensor("blk.0.ffn_gate_exps.weight"), tensor("blk.0.ffn_down_exps.weight"),
        tensor("blk.0.ffn_gate_inp.weight", (256, 8)), tensor("output.weight", (256, 256))]}


class Inputs(unittest.TestCase):
    def test_units_duplicates_and_presets(self):
        self.assertEqual(capacity("16000"), 16_000_000_000)
        self.assertEqual(capacity("16MiB"), 16 << 20)
        self.assertEqual(capacity(123), 123)
        with self.assertRaises(QuantError):
            devices("CUDA0:1,CUDA0:2")
        with self.assertRaises(QuantError):
            tensor_bytes([256, 256], "Q4_K_M")

    def test_negative_priorities_and_override(self):
        base = {"schema": "delphi.auto-quantize/1", "model": "/tmp/a.gguf", "devices": "ROCm1:16000,RAM:16000",
                "primary_qtypes": ["Q2_K"], "other_qtypes": ["Q6_K"]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            for change in ({"importance": {"precision": -1}}, {"runtime": {"extra_args": ["--device=CUDA0"]}},
                           {"sampling": {"samples": 128, "max_samples": 32}}):
                path.write_text(json.dumps({**base, **change}))
                with self.assertRaises(QuantError):
                    load_config(path)

    def test_ties_work_when_up_precedes_gate(self):
        model = classify(info(), cfg())
        tied = [g for g in model["groups"] if len(g["members"]) == 2]
        self.assertEqual(len(tied), 1)
        self.assertEqual(len({n for g in model["groups"] for n in g["members"]}), len(model["tensors"]))
        self.assertTrue(next(t for t in model["tensors"] if "gate_inp" in t["name"])["preserved"])

    def test_preservation_constrains_partner_and_pin_conflicts(self):
        config = cfg(preserve_regex=["ffn_gate_exps"])
        model = classify(info(), config)
        self.assertEqual(next(g for g in model["groups"] if len(g["members"]) == 2)["pool"], ["BF16"])
        config["pins"] = {"blk.0.ffn_gate_exps.weight": "Q2_K"}
        with self.assertRaises(QuantError):
            classify(info(), config)

    def test_row_metadata(self):
        self.assertEqual(tensor_bytes([256, 3], "IQ4_KS"), 3 * (136 + 4))
        with self.assertRaises(QuantError):
            tensor_bytes([255, 3], "Q4_K")


class Memory(unittest.TestCase):
    def test_host_remains_authoritative(self):
        config = cfg(cache_mib=2, devices={"ROCm1": 1 << 30, "ROCm2": 1 << 30, "RAM": 1 << 30})
        model = classify(info(), config)
        choices = {g["name"]: g["pool"][0] for g in model["groups"]}
        result = memory(model, config, choices)
        weights = sum(tensor_bytes(t["shape"], choices[next(g["name"] for g in model["groups"] if t["name"] in g["members"])])
                      for t in model["tensors"] if t["expert"])
        self.assertGreater(result["used_bytes"]["RAM"], weights)
        self.assertGreater(result["used_bytes"]["ROCm1"], result["cache_bytes"])
        self.assertEqual(result["used_bytes"]["ROCm2"], 0)

    def test_dense_budget_is_independent(self):
        config = cfg(devices={"ROCm1": 1, "RAM": 1 << 30})
        model = classify(info(), config)
        choices = {g["name"]: g["pool"][0] for g in model["groups"]}
        result = memory(model, config, choices)
        self.assertLess(result["headroom_bytes"]["ROCm1"], 0)
        self.assertGreater(result["headroom_bytes"]["RAM"], 0)


class Frontier(unittest.TestCase):
    def fixture(self):
        model = classify(info(), cfg())
        for g in model["groups"]:
            g["options"] = [{"qtype": q, "damage": 2 if q == "Q2_K" else 1 if q == "Q4_0" else 0,
                             "sampling_se": 0, "bytes": sum(tensor_bytes(next(t["shape"] for t in model["tensors"] if t["name"] == n), q)
                                                            for n in g["members"])} for q in g["pool"]]
        class Fake:
            measured = {"decode": 3, "short": 4, "long": 5}
            def predict(self, choices, hit_fraction):
                n = list(choices.values()).count("Q4_0")
                return {"decode": 3 + n, "short": 4 + n, "long": 5 + n}
        baseline = {g["name"]: g["pool"][0] for g in model["groups"]}
        return model, Fake(), baseline

    def test_search_matches_exhaustive_tiny_frontier(self):
        model, predictor, baseline = self.fixture()
        result = search(model, cfg(), predictor, baseline)
        expected = []
        for combination in itertools.product(*(g["options"] for g in model["groups"])):
            choices = {g["name"]: q["qtype"] for g, q in zip(model["groups"], combination)}
            times = predictor.predict(choices, 0)
            expected.append([sum(q["damage"] for q in combination), *times.values()])
        nondominated = {tuple(x) for x in expected if not any(dominates(y, x) for y in expected)}
        self.assertEqual({tuple(s["axes"]) for s in result["frontier"]}, nondominated)
        again = search(model, cfg(), predictor, baseline)
        self.assertEqual(result, again)

    def test_all_returned_recipes_meet_hard_bounds(self):
        model, predictor, baseline = self.fixture()
        result = search(model, cfg(bounds={"max_damage": 3}), predictor, baseline)
        self.assertTrue(result["frontier"])
        self.assertTrue(all(s["damage"] <= 3 for s in result["frontier"]))

    def test_budget_failure_and_evaluation_cap(self):
        model, predictor, baseline = self.fixture()
        self.assertEqual(search(model, cfg(devices={"ROCm1": 1, "RAM": 1}), predictor, baseline)["status"], "infeasible_minimum_bytes")
        result = search(model, cfg(search={"beam": 2, "evaluations": 1}), predictor, baseline)
        self.assertEqual(result["evaluated"], 1)
        self.assertTrue(result["cap_reached"])


class Files(unittest.TestCase):
    def write_matrix(self, path, scale):
        writer = GGUFWriter(str(path), "imatrix")
        writer.add_array("imatrix.datasets", ["fixture"])
        writer.add_uint32("imatrix.chunk_count", 1)
        writer.add_uint32("imatrix.chunk_size", 8)
        writer.add_tensor("blk.0.ffn_up_exps.weight.in_sum2", np.ones((8, 256), dtype=np.float32) * scale)
        writer.add_tensor("blk.0.ffn_up_exps.weight.counts", np.ones(8, dtype=np.float32) * scale)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_merge_uses_sums_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, output = [Path(d) / name for name in ("a.gguf", "b.gguf", "merged.gguf")]
            self.write_matrix(a, 2)
            self.write_matrix(b, 5)
            merge_imatrices([str(a), str(b)], output)
            result = GGUFReader(str(output))
            self.assertTrue(all(np.all(t.data == 7) for t in result.tensors))

    def test_output_verification_detects_changed_preserved_data(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / name for name in ("source.gguf", "good.gguf", "bad.gguf")]
            for path, value in zip(paths, (1, 1, 2)):
                writer = GGUFWriter(str(path), "olmoe")
                writer.add_tensor("blk.0.attn_norm.weight", np.ones(256, dtype=np.float32) * value)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
            model = classify(read_model(str(paths[0])), cfg())
            types = {"blk.0.attn_norm.weight": "F32"}
            verify_output(model, types, paths[1])
            with self.assertRaises(QuantError):
                verify_output(model, types, paths[2])


class Timing(unittest.TestCase):
    def test_real_backend_csv_header_and_quoted_shape(self):
        text = 'backend initialization\n"backend_name","op_name","op_params","test_mode","time_us"\n"CPU","MUL_MAT_GATE_UP","ne=[256,256,8,1],pair=1","perf","120.5"\n'
        self.assertEqual(operation_times(text), [120.5])

    def test_extra_gpu_percentages_are_throughput_factors(self):
        config = cfg(devices={"ROCm1": 1 << 30, "ROCm2": 1 << 30, "RAM": 1 << 30},
                     multi_gpu_speedup_percent={"decode": 100, "short": 25, "long": 50})
        # All weights are fixed, so the measured anchor is the complete modeled work.
        model = {"tensors": [], "groups": [], "top_k": 1}
        predictor = Predictor(model, config, {}, {}, {"decode": 10, "short": 100, "long": 300})
        self.assertEqual(predictor.predict({}), {"decode": 5, "short": 80, "long": 200})
        config["devices"] = {"ROCm1": 1 << 30, "RAM": 1 << 30}
        self.assertEqual(predictor.predict({}), {"decode": 10, "short": 100, "long": 300})


if __name__ == "__main__":
    unittest.main()
