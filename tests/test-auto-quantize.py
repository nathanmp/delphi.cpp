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
from config import QuantError, capacity, devices, file_identity, imatrix_identities, imatrix_inputs, load_config, normalized, write_json
from corpus import calibration_text, collect_imatrices, compose_text, imatrix_command, shares
from model import classify, merge_imatrices, read_model, tensor_bytes
from search import dominates, make_options, memory, pareto, search
from quality import LossModel, damage_digest, load_quality, modelled_groups, role_of
from calibrate import Fit, design, probe_qtypes
from timing import Predictor, digest, operation_times, regions, signature, token_counts
from auto_quantize import tensor_plan, verify_output, witness_choices
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType


def cfg(**changes):
    result = {"devices": {"ROCm1": 1 << 30, "RAM": 1 << 30}, "reserve": {}, "cache_mib": 0,
              "primary_qtypes": ["Q2_K", "Q4_0"], "other_qtypes": ["Q6_K"], "pins": {}, "preserve_regex": [],
              "preserve_below_mib": 0, "tie_gate_up": True,
              "runtime": {"env": {}, "short_tokens": 8, "long_tokens": 128, "ubatch": 128},
              "importance": {"precision": .25, "decode": .25, "short": .25, "long": .25},
              "search": {"display": 8}, "bounds": {}, "timing": {},
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
            slopes = {"decode": 1, "short": 1, "long": 1}
            def predict(self, choices, hit_fraction):
                n = list(choices.values()).count("Q4_0")
                return {"decode": 3 + n, "short": 4 + n, "long": 5 + n}
            def group_costs(self):
                return {g["name"]: {q: {r: float(q == "Q4_0") for r in self.measured} for q in g["pool"]} for g in model["groups"]}
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

    def exhaustive(self, model):
        """(damage, total tensor bytes, choices) of every combination of group options."""
        result = []
        for combination in itertools.product(*(g["options"] for g in model["groups"])):
            choices = {g["name"]: q["qtype"] for g, q in zip(model["groups"], combination)}
            result.append((sum(q["damage"] for q in combination), sum(q["bytes"] for q in combination), choices))
        return result

    def test_smallest_recipe_matches_exhaustive_search(self):
        model, predictor, baseline = self.fixture()
        every = self.exhaustive(model)
        best = min(d for d, _, _ in every)
        for ceiling in (best, best + 1, best + 2, best + 3):
            feasible = [b for d, b, _ in every if d <= ceiling]
            result = search(model, cfg(bounds={"max_damage": ceiling}, search={"display": 8, "minimize": "size"}), predictor, baseline)
            smallest = [s for s in result["frontier"] if s.get("smallest")]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(smallest), 1)
            self.assertLessEqual(smallest[0]["damage"], ceiling)
            self.assertEqual(sum(o["bytes"] for o in (next(o for o in g["options"] if o["qtype"] == smallest[0]["choices"][g["name"]])
                                                     for g in model["groups"])), min(feasible))

    def test_smallest_recipe_unreachable_bound(self):
        model, predictor, baseline = self.fixture()
        floor = min(d for d, _, _ in self.exhaustive(model))
        result = search(model, cfg(bounds={"max_damage": floor - 0.5}, search={"display": 8, "minimize": "size"}), predictor, baseline)
        self.assertEqual(result["status"], "loss_bound_unreachable")
        self.assertFalse(result["frontier"])
        with self.assertRaises(QuantError):
            search(model, cfg(search={"display": 8, "minimize": "size"}), predictor, baseline)

    def test_total_byte_cap_pools_the_devices(self):
        model, predictor, baseline = self.fixture()
        every = self.exhaustive(model)
        sizes = sorted({b for _, b, _ in every})
        cap = sizes[len(sizes) // 2]
        result = search(model, cfg(bounds={"max_total_bytes": cap}), predictor, baseline)
        total = lambda s: sum(s["memory"]["used_bytes"].values())
        self.assertTrue(result["frontier"])
        self.assertTrue(all(total(s) <= cap for s in result["frontier"]))
        best = min(d for d, b, _ in every if b <= cap)
        self.assertEqual(min(s["damage"] for s in result["frontier"]), best)

    def test_budget_failure_and_byte_budget(self):
        model, predictor, baseline = self.fixture()
        self.assertEqual(search(model, cfg(devices={"ROCm1": 1, "RAM": 1}), predictor, baseline)["status"], "infeasible_minimum_bytes")
        # RAM for the floor experts plus one upgraded group only
        floor = memory(model, cfg(), baseline)["used_bytes"]["RAM"]
        step = min(sum(tensor_bytes(next(t["shape"] for t in model["tensors"] if t["name"] == n), "Q4_0") -
                       tensor_bytes(next(t["shape"] for t in model["tensors"] if t["name"] == n), "Q2_K") for n in g["members"])
                   for g in model["groups"] if "Q4_0" in g["pool"])
        result = search(model, cfg(devices={"ROCm1": 1 << 30, "RAM": floor + step}), predictor, baseline)
        self.assertTrue(result["frontier"])
        self.assertTrue(all(list(s["choices"].values()).count("Q4_0") <= 1 for s in result["frontier"]))


class Knapsack(unittest.TestCase):
    def test_exact_against_brute_force(self):
        from knapsack import solve
        rng = np.random.default_rng(5)
        for trial in range(40):
            groups = {f"g{i}": [(int(rng.integers(1, 40)), float(rng.normal()), f"q{k}") for k in range(3)] for i in range(5)}
            budget = int(rng.integers(20, 120))
            best = None
            for combo in itertools.product(*(groups[n] for n in sorted(groups))):
                if sum(o[0] for o in combo) <= budget and (best is None or sum(o[1] for o in combo) < best - 1e-12):
                    best = sum(o[1] for o in combo)
            result = solve(groups, budget, max_units=1 << 16)
            if best is None:
                self.assertIsNone(result)
                continue
            self.assertAlmostEqual(result[2], best)
            self.assertLessEqual(result[1], budget)


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

    def write_tokens(self, path, counts, means, activations=None, dense=None, only=None):
        """Raw imatrix with a 2-expert tensor (counts per expert, mean x^2 per expert), a dense tensor and optional extras."""
        writer = GGUFWriter(str(path), "imatrix")
        writer.add_array("imatrix.datasets", [Path(path).stem])
        writer.add_uint32("imatrix.chunk_count", int(max(counts)) // 8)
        writer.add_uint32("imatrix.chunk_size", 8)

        def add(name, n, mean, act=None):
            n, mean = np.asarray(n, dtype=np.float32), np.asarray(mean, dtype=np.float32)
            writer.add_tensor(name + ".in_sum2", (n * mean)[:, None] * np.ones((1, 4), dtype=np.float32))
            writer.add_tensor(name + ".counts", n.reshape(-1, 1))
            if act is not None:
                writer.add_tensor(name + ".in_sum", (n * act)[:, None] * np.ones((1, 4), dtype=np.float32))
        add("blk.0.ffn_up_exps.weight", counts, means)
        add("blk.0.attn_q.weight", [dense[0]], [dense[1]], None if activations is None else np.float32(activations))
        if only:
            add("blk.1.attn_q.weight", [only[0]], [only[1]])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def merged(self, path):
        return {t.name: np.array(t.data, copy=True) for t in GGUFReader(str(path)).tensors}

    def test_weighted_merge_is_the_token_mixture(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, output = [Path(d) / name for name in ("a.gguf", "b.gguf", "merged.gguf")]
            self.write_tokens(a, [100, 50], [2, 4], 0.5, (100, 1), only=(100, 7))
            self.write_tokens(b, [10, 10], [6, 8], 1.0, (10, 5))
            merge_imatrices([str(a), str(b)], output, [3, 1])
            result = self.merged(output)
            counts = result["blk.0.ffn_up_exps.weight.counts"].reshape(-1)
            sums = result["blk.0.ffn_up_exps.weight.in_sum2"]
            # expert 0: both files route all tokens to it; expert 1: file A routes half of its tokens
            np.testing.assert_allclose(counts, [110, 69])
            np.testing.assert_allclose(sums / counts[:, None], [[3.0] * 4, [5.6] * 4], rtol=1e-5)
            dense = result["blk.0.attn_q.weight.counts"].reshape(-1)
            np.testing.assert_allclose(dense, [110])
            np.testing.assert_allclose(result["blk.0.attn_q.weight.in_sum2"] / dense[:, None], [[2.0] * 4], rtol=1e-5)
            np.testing.assert_allclose(result["blk.0.attn_q.weight.in_sum"] / dense[:, None], [[0.625] * 4], rtol=1e-5)
            # a tensor of one file keeps its mean; the other file does not dilute it
            only = result["blk.1.attn_q.weight.counts"].reshape(-1)
            np.testing.assert_allclose(result["blk.1.attn_q.weight.in_sum2"] / only[:, None], [[7.0] * 4], rtol=1e-5)
            self.assertTrue(all(v.dtype == np.float32 for v in result.values()))
            self.assertTrue(np.all(counts == np.rint(counts)))

    def test_token_proportional_weights_reproduce_the_plain_merge(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, plain, weighted = [Path(d) / name for name in ("a.gguf", "b.gguf", "plain.gguf", "weighted.gguf")]
            self.write_tokens(a, [100, 50], [2, 4], None, (100, 1))
            self.write_tokens(b, [10, 10], [6, 8], None, (10, 5))
            merge_imatrices([str(a), str(b)], plain)
            merge_imatrices([str(a), str(b)], weighted, [100, 10])
            left, right = self.merged(plain), self.merged(weighted)
            self.assertEqual(set(left), set(right))
            np.testing.assert_allclose(left["blk.0.ffn_up_exps.weight.counts"], right["blk.0.ffn_up_exps.weight.counts"])
            np.testing.assert_allclose(left["blk.0.ffn_up_exps.weight.in_sum2"], right["blk.0.ffn_up_exps.weight.in_sum2"], rtol=1e-5)

    def test_weighted_entries(self):
        absolute = lambda p: "/x/" + p
        paths, weights = imatrix_inputs("code.gguf:50, math.gguf:30,roleplay.gguf:25", absolute)
        self.assertEqual(paths, ["/x/code.gguf", "/x/math.gguf", "/x/roleplay.gguf"])
        self.assertAlmostEqual(weights[0], 50 / 105)
        self.assertAlmostEqual(sum(weights), 1)
        self.assertEqual(weights, imatrix_inputs(["code.gguf:5", "math.gguf:3", "roleplay.gguf:2.5"], absolute)[1])
        self.assertEqual(imatrix_inputs(["a.gguf", "b.gguf"], absolute), (["/x/a.gguf", "/x/b.gguf"], None))
        self.assertEqual(imatrix_inputs([], absolute), ([], None))
        paths, weights = imatrix_inputs(["C:/m/a.gguf:2", "b.gguf", {"path": "c.gguf", "weight": 1}], absolute)
        self.assertEqual((paths[0], weights), ("/x/C:/m/a.gguf", normalized([2, 1, 1])))
        for bad in ("a.gguf:0,b.gguf:1", "a.gguf:-2,b.gguf:1", "a.gguf:1,a.gguf:2", [{"path": "a.gguf", "weight": True}],
                    [{"path": "a.gguf", "color": 1}], [3], "a.gguf:1e999,b.gguf"):
            with self.assertRaises(QuantError):
                imatrix_inputs(bad, absolute)

    def test_weights_enter_the_fingerprint_and_the_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            for name in ("a.gguf", "b.gguf"):
                (root / name).write_bytes(name.encode())
            base = {"schema": "delphi.auto-quantize/1", "model": "/tmp/a.gguf", "devices": "ROCm1:16000,RAM:16000",
                    "primary_qtypes": ["Q2_K"], "other_qtypes": ["Q6_K"]}
            parsed = {}
            for label, entries in (("plain", ["a.gguf", "b.gguf"]), ("weighted", "a.gguf:3,b.gguf:1"), ("other", "a.gguf:1,b.gguf:1")):
                write_json(root / "c.json", {**base, "imatrix": entries})
                parsed[label] = load_config(root / "c.json")
            self.assertIsNone(parsed["plain"]["imatrix_weights"])
            self.assertEqual(parsed["weighted"]["imatrix_weights"], [.75, .25])
            self.assertEqual(parsed["plain"]["imatrix"], [str(root / "a.gguf"), str(root / "b.gguf")])
            self.assertEqual(imatrix_identities(parsed["plain"])[0], file_identity(root / "a.gguf", content=True))
            self.assertEqual(imatrix_identities(parsed["weighted"])[0]["weight"], .75)
            self.assertEqual(len({digest(imatrix_identities(c)) for c in parsed.values()}), 3)

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


class Corpora(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.paths = {}
        for name in ("code", "math", "rp"):
            self.paths[name] = self.root / f"{name}.txt"
            self.paths[name].write_text("".join(f"{name} line {i:04d}\n" for i in range(400)))

    def corpora(self, weights=(50, 30, 25)):
        return [{"path": str(self.paths[n]), "weight": w} for n, w in zip(self.paths, normalized(weights))]

    def compose(self, name="out", seed=1, chunks=21, weights=(50, 30, 25)):
        # 16 tokens of 4 bytes: pieces of 64 bytes, about four lines
        return compose_text(self.corpora(weights), 16, chunks, seed, 4.0, self.root / name)

    def test_pieces_follow_the_weights_and_the_text_is_deterministic(self):
        path, record = self.compose()
        self.assertEqual([s["pieces"] for s in record["corpora"]], [12, 8, 6])
        self.assertEqual(shares([.5, .3, .2], 10), [5, 3, 2])
        body = Path(path).read_bytes()
        self.assertEqual(record["text"]["bytes"], len(body))
        self.assertTrue(body.endswith(b"\n"))
        again, same = self.compose("again")
        self.assertEqual(Path(again).read_bytes(), body)
        self.assertEqual(same["key"], record["key"])
        self.assertNotEqual(Path(self.compose("other", seed=2)[0]).read_bytes(), body)
        # the order interleaves the corpora, so a prefix keeps the proportions
        names = [line.split()[0] for line in body.decode().splitlines()]
        runs = [key for key, _ in itertools.groupby(names)]
        self.assertEqual(runs[:3], ["code", "math", "rp"])
        half = names[:len(names) // 2]
        for name, weight in zip(self.paths, normalized((50, 30, 25))):
            self.assertAlmostEqual(half.count(name) / len(half), weight, delta=.2)

    def test_reuse_and_digest_names(self):
        path, record = self.compose()
        stamp = Path(path).stat().st_mtime_ns
        self.assertEqual(self.compose()[0], path)
        self.assertEqual(Path(path).stat().st_mtime_ns, stamp)
        self.assertNotEqual(self.compose(weights=(50, 30, 30))[0], path)
        self.assertNotEqual(self.compose(chunks=30)[0], path)
        self.assertEqual(calibration_text({"text": "plain.txt"}, self.root), ("plain.txt", None))
        c = {"text": self.corpora(), "context": 16, "chunks": 21, "text_seed": 1, "bytes_per_token": 4.0}
        self.assertEqual(calibration_text(c, self.root / "out")[0], path)

    def test_small_corpus_and_empty_share_fail(self):
        self.paths["rp"].write_text("rp line 0000\nrp line 0001\n")
        with self.assertRaisesRegex(QuantError, "too small for its share"):
            self.compose()
        self.paths["rp"].write_text("".join(f"rp line {i:04d}\n" for i in range(400)))
        with self.assertRaisesRegex(QuantError, "receives no piece"):
            self.compose(chunks=1, weights=(1000, 1000, 1))

    def test_configuration(self):
        base = {"schema": "delphi.auto-quantize/1", "model": "/tmp/a.gguf", "devices": "ROCm1:16000,RAM:16000",
                "primary_qtypes": ["Q2_K"], "other_qtypes": ["Q6_K"]}
        path = self.root / "config.json"
        for text, expected in (("one.txt", str(self.root / "one.txt")),
                               ("one.txt:7", str(self.root / "one.txt")),
                               ("code.txt:50,math.txt:30,rp.txt:25", None),
                               (["code.txt:5", {"path": "math.txt", "weight": 3, "bytes_per_token": 3.2}, "rp.txt"], None)):
            write_json(path, {**base, "calibration": {"text": text, "text_seed": 5}})
            parsed = load_config(path)["calibration"]
            self.assertEqual(parsed["text_seed"], 5)
            if expected:
                self.assertEqual(parsed["text"], expected)
            else:
                self.assertEqual([e["path"] for e in parsed["text"]], [str(self.root / f"{n}.txt") for n in ("code", "math", "rp")])
                self.assertAlmostEqual(sum(e["weight"] for e in parsed["text"]), 1)
        self.assertEqual(parsed["text"][1]["bytes_per_token"], 3.2)
        self.assertAlmostEqual(parsed["text"][0]["weight"], 5 / 9)
        for bad in ({"text": "a.txt:1,a.txt:2"}, {"text": "a.txt:0,b.txt"}, {"text": []}, {"text_seed": -1},
                    {"bytes_per_token": 0}, {"text": [{"path": "a.txt", "color": 1}, "b.txt"]}):
            write_json(path, {**base, "calibration": bad})
            with self.assertRaises(QuantError):
                load_config(path)


class Collect(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        (self.root / "bin").mkdir()
        script = self.root / "bin/llama-imatrix"
        script.write_text('#!/bin/sh\necho "$@" >> "$FAKE_LOG"\necho "means=$DELPHI_IMATRIX_MEANS" >> "$FAKE_LOG"\n'
                          'while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\ncp "$FAKE_IMATRIX" "$out"\n')
        script.chmod(0o755)
        writer = GGUFWriter(str(self.root / "model.gguf"), "olmoe")
        writer.add_tensor("blk.0.attn_norm.weight", np.ones(256, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = GGUFWriter(str(self.root / "fake.gguf"), "imatrix")
        writer.add_array("imatrix.datasets", ["fake"])
        writer.add_uint32("imatrix.chunk_count", 1)
        writer.add_uint32("imatrix.chunk_size", 8)
        for suffix, value in ((".in_sum2", np.ones((1, 4))), (".counts", np.ones((1, 1))), (".in_sum", np.ones((1, 4)))):
            writer.add_tensor("blk.0.attn_q.weight" + suffix, value.astype(np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        for name in ("code", "math", "rp"):
            (self.root / f"{name}.txt").write_text(f"{name}\n" * 50)
        self.log = self.root / "log.txt"
        self.base = {"schema": "delphi.auto-quantize/1", "model": str(self.root / "model.gguf"), "devices": "ROCm1:16000,RAM:16000",
                     "bindir": str(self.root / "bin"), "primary_qtypes": ["Q2_K"], "other_qtypes": ["Q6_K"],
                     "runtime": {"env": {"FAKE_LOG": str(self.log), "FAKE_IMATRIX": str(self.root / "fake.gguf")}}}

    def config(self, **collect):
        write_json(self.root / "config.json", {**self.base, "imatrix_collect": collect})
        return load_config(self.root / "config.json")

    def test_one_run_per_corpus_then_reuse(self):
        cfg = self.config(text="code.txt:50,math.txt:30,rp.txt:25", context=64, chunks=4, threads=3, extra_args=["-b", "256"])
        output = self.root / "collected"
        entries = collect_imatrices(cfg, output)
        log = self.log.read_text().splitlines()
        self.assertEqual(len(log), 6)
        for part in (f"-m {self.root / 'model.gguf'}", f"-f {self.root / 'code.txt'}", "-c 64", "-t 3", "--device ROCm1",
                     "-ngl 999", "-cmoe", "--chunks 4", "--no-ppl", "-b 256"):
            self.assertIn(part, log[0])
        self.assertEqual(log[1], "means=1")
        self.assertEqual([Path(e.rpartition(":")[0]).name for e in entries],
                         ["00-code.imatrix.gguf", "01-math.imatrix.gguf", "02-rp.imatrix.gguf"])
        paths, weights = imatrix_inputs(entries, lambda p: p)
        for weight, expected in zip(weights, normalized((50, 30, 25))):
            self.assertAlmostEqual(weight, expected, places=8)
        self.assertTrue(all(Path(path).exists() for path in paths))
        self.assertEqual(collect_imatrices(cfg, output), entries)
        self.assertEqual(len(self.log.read_text().splitlines()), 6)
        (self.root / "code.txt").write_text("changed\n")
        with self.assertRaises(QuantError):
            collect_imatrices(cfg, output)

    def test_command_and_configuration(self):
        c = {"context": 512, "chunks": 0, "threads": 4, "extra_args": []}
        command = imatrix_command({"bindir": "/b", "model": "/m.gguf", "devices": {"ROCm1": 1}}, c, "/t.txt", "/o.gguf")
        self.assertEqual(command[0], "/b/llama-imatrix")
        self.assertNotIn("-cmoe", command)
        self.assertNotIn("--chunks", command)
        self.assertEqual(self.config(text="a.txt")["imatrix_collect"]["means"], True)
        with self.assertRaises(QuantError):
            collect_imatrices(load_config(self.root / "config.json") | {"imatrix_collect": None}, self.root / "none")
        for bad in ({}, {"text": "a.txt", "extra_args": ["--device", "CUDA0"]}, {"text": "a.txt", "chunks": -1},
                    {"text": "a.txt", "means": 1}, {"text": "a.txt", "context": 0}, {"text": "a.txt", "color": 1}):
            with self.assertRaises(QuantError):
                self.config(**bad)


class Timing(unittest.TestCase):
    def test_forced_staging_counts_the_whole_bank_even_when_cpu_is_faster(self):
        config = cfg(primary_qtypes=["Q2_K"], runtime={"env": {"GGML_SCHED_STAGE": "1", "GGML_SCHED_STAGE_MIN_TOKENS": "128"},
                                                      "short_tokens": 8, "long_tokens": 128, "ubatch": 128},
                     timing={"link_bandwidth_gb_s": .001})
        raw = info()
        raw["n_expert"] = 64
        for t in raw["tensors"]:
            if len(t["shape"]) == 3:
                t["shape"][2] = 64
                t["source_bytes"] = 2 * np.prod(t["shape"]).item()
        model = classify(raw, config)
        operations = {}
        for group, members, paired in regions(model):
            for dtype in group["pool"]:
                for n in set(token_counts(config).values()):
                    for device in (["CPU", "ROCm1"] if members[0]["expert"] else ["ROCm1"]):
                        sig = signature(members, paired, dtype, device, n, model["top_k"])
                        operations[digest(sig)] = {"signature": sig, "ms": .01}
        baseline = {g["name"]: g["pool"][0] for g in model["groups"]}
        predictor = Predictor(model, config, operations, baseline, {"decode": 12, "short": 200, "long": 1000})
        staged = predictor.cost(baseline)["long"]
        banks = sum(tensor_bytes(t["shape"], "Q2_K") for t in model["tensors"] if t["expert"])
        self.assertAlmostEqual(staged, banks / 1000 + 2 * (.01 + .02) + .01)
        config["runtime"]["env"]["GGML_SCHED_STAGE"] = "0"
        self.assertLess(predictor.cost(baseline)["long"], staged)

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


class Quality(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.gguf"
        self.write_source(self.source)
        self.config = cfg(primary_qtypes=["Q2_K", "Q4_K", "Q6_K"], other_qtypes=["F32"], imatrix=[],
                          quality={"table": str(self.root / "quality.json")})
        self.model = classify(read_model(str(self.source)), self.config)
        levels = {"Q2_K": .1, "Q4_K": .02, "Q6_K": .004}
        self.damage = [{"schema": "delphi.quant-damage/1", "tensors": [
            {"name": t["name"], "runs": [{"types": [{"qtype": q, "status": "ok", "R": levels[q] * (1 + t["layer"] / 4), "R_se": 0}
                                                     for q in levels]}]} for t in self.model["tensors"] if t["expert"]]}]
        self.floor = {g["name"]: "Q2_K" if len(g["pool"]) > 1 else g["pool"][0] for g in self.model["groups"]}
        self.prepared = {"damage": self.damage, "baseline": {"choices": self.floor}}
        self.table = {"schema": "delphi.floor-quality/3", "method": "composed_smooth_sensitivity", "metric": "R",
                      "source_files": [file_identity(self.source, content=True)], "imatrices": [],
                      "damage": damage_digest(self.damage), "floor": {"choices": self.floor, "kl": .1},
                      "teacher": {}, "execution": {"devices": ["ROCm1"]}, "n_layer": 4,
                      "roles": {"gu": {"knots": [0.5, 2.5], "log_s": [-2.0, -1.8], "gamma": 1.0},
                                "dn": {"knots": [0.5, 2.5], "log_s": [-1.6, -1.9], "gamma": 0.8}},
                      "composition": {"kappa_inf": 1.5, "u0": 0.3}, "fit": {"chi2_dof": 1.0, "passes": 10}}

    def load(self, table=None, config=None, prepared=None):
        write_json(self.root / "quality.json", self.table if table is None else table)
        return load_quality(self.model, self.config if config is None else config, self.prepared if prepared is None else prepared)

    def test_roles_and_modelled_groups(self):
        self.assertEqual([role_of(n) for n in ("blk.3.ffn_gate_exps.weight", "blk.3.ffn_down_exps.weight", "blk.1.ffn_up_shexp.weight",
                                                "blk.0.attn_q.weight", "blk.2.ffn_gate.weight", "output.weight")],
                         ["gu", "dn", "shexp", "attn", "gu", "output"])
        groups = modelled_groups(self.model, self.floor)
        self.assertEqual(len(groups), 8)
        self.assertEqual({g["role"] for g in groups}, {"gu", "dn"})

    def write_source(self, path, extra=()):
        writer = GGUFWriter(str(path), "olmoe")
        writer.add_uint32("olmoe.expert_count", 8)
        writer.add_uint32("olmoe.expert_used_count", 2)
        for layer in range(4):
            for role in ("gate", "up", "down"):
                writer.add_tensor(f"blk.{layer}.ffn_{role}_exps.weight", np.zeros((8, 256, 256), dtype=np.float32))
        for name in extra:
            writer.add_tensor(name, np.zeros((256, 512), dtype=np.float32))
        writer.add_tensor("output.weight", np.ones((256, 256), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_mtp_layers_are_found_pinned_at_the_ceiling_and_recipes_are_witnesses(self):
        raw = read_model(str(self.source))
        raw["n_layer"], raw["nextn"] = 4, 1
        self.assertEqual(classify(raw, self.config)["mtp_layers"], [3])
        source = self.root / "mtp.gguf"
        self.write_source(source, ["blk.3.nextn.eh_proj.weight"])
        model = classify(read_model(str(source)), self.config)
        self.assertEqual(model["mtp_layers"], [3])
        floor = {g["name"]: "Q2_K" if len(g["pool"]) > 1 else g["pool"][0] for g in model["groups"]}
        self.assertEqual({floor[f"blk.3.ffn_{r}_exps.weight"] for r in ("gate", "down")}, {"Q6_K"})
        pinned = classify(read_model(str(source)), dict(self.config, pins={"blk.3.ffn_down_exps.weight": "Q4_K"}))
        self.assertEqual(next(g["pool"] for g in pinned["groups"] if g["name"] == "blk.3.ffn_down_exps.weight"), ["Q4_K"])
        groups = modelled_groups(model, floor)
        self.assertEqual(len(groups), 6)
        self.assertTrue(all(g["layer"] < 3 for g in groups))
        recipe = self.root / "recipe.json"
        planned = dict(floor, **{g["name"]: "Q6_K" for g in groups})
        write_json(recipe, {"choices": planned})
        self.assertEqual(witness_choices(model, floor, {"recipe": str(recipe), "overrides": {"blk.0.ffn_down": "Q4_K"}}),
                         dict(planned, **{"blk.0.ffn_down_exps.weight": "Q4_K"}))
        write_json(recipe, {"choices": dict(planned, **{"output.weight": "Q4_K"})})
        with self.assertRaises(QuantError):
            witness_choices(model, floor, {"recipe": str(recipe)})

    def test_floor_prediction_composition_and_bounds(self):
        loss = LossModel(self.model, self.load(), self.damage)
        self.assertAlmostEqual(loss.predict(self.floor)["loss"], .1)
        upgraded = [dict(self.floor, **{g: q for g in loss.groups}) for q in ("Q4_K", "Q6_K")]
        values = [loss.predict(c) for c in upgraded]
        self.assertGreater(.1, values[0]["loss"])
        self.assertGreater(values[0]["loss"], values[1]["loss"])
        self.assertTrue(0 < values[0]["u"] < values[1]["u"] < 1)
        # the composed gain grows faster than the additive gain
        self.assertGreater(1 - values[1]["loss"] / .1, values[1]["gain"])
        # numerical derivative of the composed gain
        u = .4
        h = 1e-6
        composed = lambda v: loss.kappa(v) * v
        self.assertAlmostEqual(loss.marginal(u), (composed(u + h) - composed(u - h)) / (2 * h), places=5)

    def test_rejects_changed_damage_floor_source_and_cache(self):
        self.load()
        bad_knots = {**self.table["roles"], "gu": {"knots": [2.5, 0.5], "log_s": [-2.0, -1.8], "gamma": 1.0}}
        for table in (dict(self.table, roles=bad_knots), dict(self.table, damage="0" * 64), dict(self.table, floor={"choices": {**self.floor, "blk.0.ffn_down_exps.weight": "Q4_K"}, "kl": .1}),
                      dict(self.table, schema="delphi.floor-quality/2"), dict(self.table, composition={"kappa_inf": .5, "u0": .3})):
            with self.assertRaises(QuantError):
                self.load(table)
        with self.assertRaises(QuantError):
            self.load(config={**self.config, "cache_mib": 1})
        with self.source.open("r+b") as stream:
            stream.seek(read_model(str(self.source))["tensors"][0]["offset"])
            stream.write(b"\x01")
        with self.assertRaises(QuantError):
            self.load()

    def test_knapsack_search_respects_loss_ceiling_and_budget(self):
        config = dict(self.config, bounds={"max_loss": .06})
        loss = LossModel(self.model, self.load(), self.damage)
        damage = self.damage
        make_options(self.model, damage, {}, config)
        ops = {}
        for group, members, paired in regions(self.model):
            for dtype in group["pool"]:
                for n in set(token_counts(config).values()):
                    for device in (["CPU", "ROCm1"] if members[0]["expert"] else ["ROCm1"]):
                        sig = signature(members, paired, dtype, device, n, self.model["top_k"])
                        ops[digest(sig)] = {"signature": sig, "ms": {"Q2_K": 1, "Q4_K": 2, "Q6_K": 3}.get(dtype, 1)}
        predictor = Predictor(self.model, config, ops, self.floor, {"decode": 20, "short": 30, "long": 50})
        result = search(self.model, config, predictor, self.floor, loss)
        self.assertTrue(result["frontier"])
        self.assertTrue(all(s["loss"] <= .06 for s in result["frontier"]))
        self.assertTrue(all(abs(s["loss"] - loss.predict(s["choices"])["loss"]) < 1e-12 for s in result["frontier"]))
        # the fastest recipe that meets the ceiling is found at the edge of the bound
        fastest = min(result["frontier"], key=lambda s: s["axes"][1])
        self.assertGreater(fastest["loss"], .04)

    def test_smallest_recipe_with_the_quality_model(self):
        config = dict(self.config, bounds={"max_loss": .06}, search={"display": 8, "minimize": "size"})
        loss = LossModel(self.model, self.load(), self.damage)
        make_options(self.model, self.damage, {}, config)
        ops = {}
        for group, members, paired in regions(self.model):
            for dtype in group["pool"]:
                for n in set(token_counts(config).values()):
                    for device in (["CPU", "ROCm1"] if members[0]["expert"] else ["ROCm1"]):
                        sig = signature(members, paired, dtype, device, n, self.model["top_k"])
                        ops[digest(sig)] = {"signature": sig, "ms": {"Q2_K": 1, "Q4_K": 2, "Q6_K": 3}.get(dtype, 1)}
        predictor = Predictor(self.model, config, ops, self.floor, {"decode": 20, "short": 30, "long": 50})
        plain = search(self.model, dict(config, search={"display": 8}), predictor, self.floor, loss)
        result = search(self.model, config, predictor, self.floor, loss)
        total = lambda s: sum(s["memory"]["used_bytes"].values())
        smallest = [s for s in result["frontier"] if s.get("smallest")]
        self.assertEqual(len(smallest), 1)
        self.assertLessEqual(smallest[0]["loss"], .06)
        self.assertLessEqual(total(smallest[0]), min(total(s) for s in plain["frontier"]))
        self.assertLess(total(smallest[0]), max(total(s) for s in plain["frontier"]))

    def test_fit_recovers_a_known_model(self):
        groups = modelled_groups(self.model, self.floor)
        records = {(t["name"], q["qtype"]): q["R"] for t in self.damage[0]["tensors"] for q in t["runs"][0]["types"]}
        fit = Fit(groups, records, 4, 4)
        knots = [0.0, 1.0, 2.0, 3.0]
        truth = {"gu": {"knots": knots, "log_s": [-2.0, -1.9, -1.7, -1.4], "gamma": 1.1},
                 "dn": {"knots": knots, "log_s": [-1.0, -1.8, -1.9, -2.0], "gamma": 0.7}}
        jobs = design(groups, "Q4_K", "Q6_K", {"bands": 4, "recipes": 8, "seed": 3})
        observations = [(j["choices"], float(y), .002) for j, y in zip(jobs, fit.predict(truth, 1.6, .4, [j["choices"] for j in jobs]))]
        params, kinf, u0, chi2 = fit.fit(observations)
        predicted = fit.predict(params, kinf, u0, [o[0] for o in observations])
        self.assertLess(np.max(np.abs(predicted - np.array([o[1] for o in observations]))), .004)
        self.assertAlmostEqual(params["gu"]["gamma"], 1.1, delta=.15)
        # a step at the first down band survives the smoothness penalty
        self.assertGreater(params["dn"]["log_s"][0] - params["dn"]["log_s"][1], .4)

    def test_design_cost_is_independent_of_the_candidate_count(self):
        groups = modelled_groups(self.model, self.floor)
        jobs = design(groups, "Q4_K", "Q6_K", {"bands": 2, "recipes": 5, "seed": 1})
        self.assertEqual(sum(1 for j in jobs if "band" in j["label"]), 2 * 2 * 2)
        self.assertEqual(sum(1 for j in jobs if "direct" in j), 2)
        more = [dict(g, pool=g["pool"] + ["Q5_K", "Q8_0"]) for g in groups]
        self.assertEqual(len(design(more, "Q4_K", "Q6_K", {"bands": 2, "recipes": 5, "seed": 1})), len(jobs))
        records = {(t["name"], q["qtype"]): q["R"] for t in self.damage[0]["tensors"] for q in t["runs"][0]["types"]}
        self.assertEqual(probe_qtypes(groups, records, {"probe_qtypes": None}), ("Q4_K", "Q6_K"))

    def test_configuration(self):
        base = {"schema": "delphi.auto-quantize/1", "model": str(self.source), "devices": "ROCm1:16000,RAM:16000",
                "primary_qtypes": ["Q2_K", "Q4_K"], "other_qtypes": ["F32"], "quality": {"table": "quality.json"},
                "bounds": {"max_loss": .09}, "calibration": {"text": "eval.txt", "devices": ["CUDA0", "CUDA1"]}}
        path = self.root / "config.json"
        write_json(path, base)
        parsed = load_config(path)
        self.assertEqual(parsed["calibration"]["text"], str(self.root / "eval.txt"))
        self.assertEqual(parsed["calibration"]["bands"], 10)
        for changed in ({**base, "bounds": {"max_damage": .09}}, {k: v for k, v in base.items() if k != "quality"},
                        {**base, "calibration": {"recipes": 1}}, {**base, "calibration": {"devices": ["RAM"]}},
                        {**base, "calibration": {"batch": 1000}}, {**base, "search": {"beam": 4}}):
            write_json(path, changed)
            with self.assertRaises(QuantError):
                load_config(path)

    def test_cpu_expert_format_survives_unsupported_gpu(self):
        config = dict(self.config, runtime=dict(self.config["runtime"], env={"GGML_SCHED_STAGE": "1", "GGML_SCHED_STAGE_MIN_TOKENS": "1"}))
        ops = {}
        for group, members, paired in regions(self.model):
            for dtype in group["pool"]:
                for n in set(token_counts(config).values()):
                    for device in (["CPU", "ROCm1"] if members[0]["expert"] else ["ROCm1"]):
                        sig = signature(members, paired, dtype, device, n, self.model["top_k"])
                        ops[digest(sig)] = {"signature": sig, "ms": None if device == "ROCm1" and members[0]["expert"] else 1.0}
        self.assertEqual(make_options(self.model, self.damage, ops, config), [])
        predictor = Predictor(self.model, config, ops, self.floor, {"decode": 20, "short": 30, "long": 50})
        self.assertEqual(predictor.predict(self.floor), predictor.measured)
        upgraded = {g["name"]: g["pool"][-1] for g in self.model["groups"]}
        self.assertTrue(all(v > 0 for v in predictor.predict(upgraded).values()))
        model = classify(read_model(str(self.source)), config)
        with self.assertRaises(QuantError):
            make_options(model, self.damage, ops, {**config, "cache_mib": 1})


if __name__ == "__main__":
    unittest.main()
