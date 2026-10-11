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
from config import QuantError, capacity, devices, file_identity, load_config, write_json
from model import classify, merge_imatrices, read_model, tensor_bytes
from search import dominates, make_options, memory, pareto, search
from quality import apply_quality, load_quality, repair_options, tensor_hash, verify_quality_output
from timing import Predictor, digest, operation_times, regions, signature, token_counts
from auto_quantize import tensor_plan, verify_output
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType


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
    def test_variable_tied_embedding_includes_the_output_projection(self):
        model = {"architecture": "olmoe", "n_expert": 8, "top_k": 2,
                 "tensors": [tensor("token_embd.weight", (256, 256))]}
        config = cfg(other_qtypes=["Q6_K", "Q8_0"])
        classify(model, config)
        self.assertEqual(len(regions(model)), 1)
        model["tensors"].append(tensor("output.weight", (256, 256)))
        classify(model, config)
        self.assertEqual([m[0]["name"] for _, m, _ in regions(model)], ["output.weight"])

    def test_ram_traffic_prevents_a_hot_probe_from_claiming_a_decode_win(self):
        config = cfg()
        model = classify(info(), config)
        operations = {}
        for group, members, paired in regions(model):
            for dtype in group["pool"]:
                for tokens in set(token_counts(config).values()):
                    for device in (["CPU", "ROCm1"] if members[0]["expert"] else ["ROCm1"]):
                        sig = signature(members, paired, dtype, device, tokens, model["top_k"])
                        operations[digest(sig)] = {"signature": sig, "ms": .01 if dtype == "Q4_0" else .02}
        baseline = {g["name"]: g["pool"][0] for g in model["groups"]}
        changed = {g["name"]: "Q4_0" if "Q4_0" in g["pool"] else baseline[g["name"]] for g in model["groups"]}
        predictor = Predictor(model, config, operations, baseline, {"decode": 12, "short": 200, "long": 1000}, baseline_cpu_ms=10)
        self.assertGreater(predictor.predict(changed)["decode"], predictor.predict(baseline)["decode"])

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


class RepairQuality(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source, self.floor, self.higher = [self.root / f"{name}.gguf" for name in ("source", "floor", "higher")]
        names = [f"blk.{layer}.ffn_{role}_exps.weight" for layer in range(2) for role in ("gate", "up", "down")]
        for path, dtype, width in ((self.source, "F32", 256), (self.floor, "Q2_K", 84), (self.higher, "Q4_K", 144)):
            writer = GGUFWriter(str(path), "olmoe")
            writer.add_uint32("olmoe.expert_count", 8)
            writer.add_uint32("olmoe.expert_used_count", 2)
            for name in names:
                data = np.zeros((8, 256, width), dtype=np.float32 if dtype == "F32" else np.uint8)
                writer.add_tensor(name, data, raw_dtype=GGMLQuantizationType[dtype])
            writer.add_tensor("output.weight", np.ones((256, 256), dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
        floors = {t["name"]: {"shape": t["shape"], "qtype": t["source_dtype"], "bytes": t["source_bytes"], "sha256": tensor_hash(t)}
                  for t in read_model(self.floor)["tensors"]}
        higher = {t["name"]: t for t in read_model(self.higher)["tensors"]}
        groups = []
        for role in ("gate-up", "down"):
            members = [n for n in names if ("down" in n) == (role == "down")]
            options = []
            for dtype, loss in (("Q2_K", .1), ("Q4_K", .02)):
                options.append({"qtype": dtype, "loss": loss, "chunk_losses": [loss * .9, loss * 1.1],
                                "bytes": sum(tensor_bytes(floors[n]["shape"], dtype) for n in members),
                                "tensor_sha256": {n: floors[n]["sha256"] if dtype == "Q2_K" else tensor_hash(higher[n]) for n in members}})
            groups.append({"name": role, "members": members, "options": options})
        self.table = {"schema": "delphi.floor-quality/2", "method": "sum_conditional_repair_kl",
                      "source_files": [file_identity(self.source, content=True)], "imatrices": [],
                      "floor": {**file_identity(self.floor, content=True), "tensors": floors},
                      "execution": {"gpu": "ROCm1", "ram_routed": True, "numa": "tensors"}, "groups": groups}
        self.path = self.root / "table.json"
        self.config = cfg(primary_qtypes=["Q2_K", "Q4_K"], other_qtypes=["F32", "Q6_K"],
                          quality={"table": str(self.path)}, imatrix=[])

    def load(self, table=None, config=None):
        write_json(self.path, self.table if table is None else table)
        config = self.config if config is None else config
        model = classify(read_model(self.source), config)
        table = load_quality(model, config)
        return apply_quality(model, table), table

    def test_groups_keep_execution_pairs_and_freeze_unmeasured_weights(self):
        model, table = self.load()
        self.assertEqual(len(model["groups"]), 3)
        self.assertEqual(next(g for g in model["groups"] if g["name"] == "output.weight")["pool"], ["F32"])
        pairs = [(g, m) for g, m, paired in regions(model) if paired]
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(g["decision"] == "gate-up" for g, _ in pairs))
        repair_options(model, table, {}, self.config)
        baseline = {g["name"]: g["floor_qtype"] for g in model["groups"]}
        self.assertAlmostEqual(sum(g["options"][0]["damage"] for g in model["groups"]), .2)
        verify_quality_output(model, table, tensor_plan(model, baseline), self.floor)
        ops = {}
        for group, members, paired in regions(model):
            for dtype in group["pool"]:
                for n in set(token_counts(self.config).values()):
                    for device in (["CPU", "ROCm1"] if members[0]["expert"] else ["ROCm1"]):
                        sig = signature(members, paired, dtype, device, n, model["top_k"])
                        ops[digest(sig)] = {"signature": sig, "ms": 1 if dtype == "Q2_K" else 2}
        predictor = Predictor(model, self.config, ops, baseline, {"decode": 20, "short": 30, "long": 50})
        self.config["bounds"] = {"max_loss": .13}
        result = search(model, self.config, predictor, baseline)
        self.assertTrue(result["frontier"])
        self.assertTrue(all(s["loss"] <= .13 and s["damage"] == s["loss"] for s in result["frontier"]))
        self.assertTrue(all(s["memory"]["used_bytes"]["RAM"] > 0 for s in result["frontier"]))
        self.assertEqual(set(result["frontier"][0]["choices"]), {"gate-up", "down", "output.weight"})

    def test_rejects_split_ties_overlap_bad_loss_and_wrong_bytes(self):
        changes = []
        bad = copy.deepcopy(self.table); bad["groups"][0]["members"].pop(); changes.append(bad)
        bad = copy.deepcopy(self.table); bad["groups"][1]["members"].append(bad["groups"][0]["members"][0]); changes.append(bad)
        bad = copy.deepcopy(self.table); bad["groups"][0]["options"][0]["loss"] = float("nan"); changes.append(bad)
        bad = copy.deepcopy(self.table); bad["groups"][0]["options"][0]["bytes"] += 1; changes.append(bad)
        bad = copy.deepcopy(self.table); bad["groups"][0]["options"].pop(0); changes.append(bad)
        for table in changes:
            self.path.write_text(json.dumps(table))
            model = classify(read_model(self.source), self.config)
            with self.assertRaises(QuantError):
                load_quality(model, self.config)

    def test_rejects_changed_source_and_cpu_gpu_policy(self):
        for change in ({"devices": {"ROCm1": 1 << 30}}, {"devices": {"CUDA0": 1 << 30, "RAM": 1 << 30}}, {"cache_mib": 1}):
            with self.assertRaises(QuantError):
                self.load(config={**self.config, **change})
        with self.source.open("r+b") as stream:
            stream.seek(read_model(self.source)["tensors"][0]["offset"])
            stream.write(b"\x01")
        with self.assertRaises(QuantError):
            self.load()

    def test_repair_configuration_uses_explicit_loss_units(self):
        base = {"schema": "delphi.auto-quantize/1", "model": str(self.source), "devices": "ROCm1:16000,RAM:16000",
                "primary_qtypes": ["Q2_K", "Q4_K"], "other_qtypes": ["F32"],
                "quality": {"table": "table.json"}, "bounds": {"max_loss": .09}}
        config_path = self.root / "config.json"
        write_json(config_path, base)
        parsed = load_config(config_path)
        self.assertEqual(parsed["quality"]["table"], str(self.path))
        self.assertEqual(parsed["bounds"]["max_loss"], .09)
        for changed in ({**base, "bounds": {"max_damage": .09}},
                        {k: v for k, v in base.items() if k != "quality"},
                        {**base, "quality": {"table": "table.json", "metric": "accuracy"}}):
            write_json(config_path, changed)
            with self.assertRaises(QuantError):
                load_config(config_path)

    def test_rejects_changed_imatrix_and_encoded_payload(self):
        imatrix = self.root / "imatrix.gguf"
        imatrix.write_bytes(b"original")
        self.table["imatrices"] = [file_identity(imatrix, content=True)]
        self.config["imatrix"] = [str(imatrix)]
        model, table = self.load()
        types = {n: t["qtype"] for n, t in table["floor"]["tensors"].items()}
        with self.floor.open("r+b") as stream:
            stream.seek(read_model(self.floor)["tensors"][0]["offset"])
            stream.write(b"\x01")
        with self.assertRaises(QuantError):
            verify_quality_output(model, table, types, self.floor)
        imatrix.write_bytes(b"modified")
        with self.assertRaises(QuantError):
            self.load()

    def test_rejects_unsupported_cpu_floor_and_excludes_only_upgrade(self):
        model, table = self.load()
        t = model["tensors"][0]
        sig = signature([t], True, "Q4_K", "CPU", 1, model["top_k"])
        excluded = repair_options(model, table, {"failed": {"signature": sig, "ms": None}}, self.config)
        self.assertTrue(excluded)
        self.assertEqual(next(g for g in model["groups"] if g["name"] == "gate-up")["pool"], ["Q2_K"])
        self.assertEqual(next(g for g in model["groups"] if g["name"] == "down")["pool"], ["Q2_K", "Q4_K"])
        sig["dtype"] = "Q2_K"
        model, table = self.load()
        with self.assertRaises(QuantError):
            repair_options(model, table, {"failed": {"signature": sig, "ms": None}}, self.config)

    def test_cpu_expert_format_survives_unsupported_gpu_and_stays_on_cpu(self):
        self.config["runtime"]["env"]["GGML_SCHED_STAGE"] = "1"
        self.config["runtime"]["env"]["GGML_SCHED_STAGE_MIN_TOKENS"] = "1"
        model, table = self.load()
        ops = {}
        for group, members, paired in regions(model):
            for dtype in group["pool"]:
                for n in set(token_counts(self.config).values()):
                    for device in ("CPU", "ROCm1"):
                        sig = signature(members, paired, dtype, device, n, model["top_k"])
                        ops[digest(sig)] = {"signature": sig, "ms": None if device == "ROCm1" else 1.0}
        self.assertEqual(repair_options(model, table, ops, self.config), [])
        baseline = {g["name"]: g["floor_qtype"] for g in model["groups"]}
        predictor = Predictor(model, self.config, ops, baseline, {"decode": 20, "short": 30, "long": 50})
        self.assertEqual(predictor.predict(baseline), predictor.measured)
        upgraded = {g["name"]: g["pool"][-1] for g in model["groups"]}
        self.assertTrue(all(v > 0 for v in predictor.predict(upgraded).values()))
        # The local metric uses the same placement rule, including a GPU cache's stricter requirement.
        atomic = classify(read_model(self.source), {**self.config, "other_qtypes": ["F32"]})
        damage = [{"schema": "delphi.quant-damage/1", "tensors": [
            {"name": t["name"], "runs": [{"types": [{"qtype": q, "status": "ok", "R": 1.0, "R_se": 0}
                                                       for q in t["pool"]]}]} for t in atomic["tensors"]]}]
        self.assertEqual(make_options(atomic, damage, ops, self.config), [])
        atomic = classify(read_model(self.source), {**self.config, "other_qtypes": ["F32"]})
        with self.assertRaises(QuantError):
            make_options(atomic, damage, ops, {**self.config, "cache_mib": 1})


if __name__ == "__main__":
    unittest.main()
