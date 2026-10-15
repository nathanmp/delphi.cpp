#!/usr/bin/env python3
"""Offline checks of the quantization CLI wrapper: query parsing, frontier thinning, picks. No inference or numpy."""
import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/auto-quantize"))
import frontier as fr
from config import QuantError
from quant_cli import apply_query, build_parser, parse_weights

GIB = 1 << 30


def payload(n=40, quality=True):
    """Synthetic plan: more bytes -> lower KL; decode and prefill times vary independently of size."""
    recipes = []
    for i in range(n):
        ms = {"decode": 20 + (i * 7) % 11, "short": 300 + (i * 13) % 17 * 5, "long": 4000 + (i * 29) % 23 * 40}
        recipes.append({"id": f"recipe-{i:03d}", "loss": 0.1 / (1 + i), "predicted_ms": ms, "score": abs(i - 12) + 1,
                        "memory": {"used_bytes": {"ROCm1": 5 * GIB, "RAM": (20 + i) * GIB}},
                        "tensor_types": {"blk.0.ffn_down_exps.weight": "Q3_K" if i % 2 else "IQ4_XS",
                                         "blk.1.ffn_down_exps.weight": "IQ4_XS", "blk.0.attn_q.weight": "Q8_0"}})
    return {"config": {"runtime": {"short_tokens": 128, "long_tokens": 4096}, "reserve": {"RAM": 1 * GIB}},
            "recipes": recipes, "quality": {} if quality else None}


def cfg():
    return {"search": {"display": 8}, "devices": {"ROCm1": 24 * GIB, "RAM": 100 * GIB}, "reserve": {"RAM": GIB}, "quality": {"table": "q.json"},
            "importance": {"precision": .25, "decode": .25, "short": .25, "long": .25}, "bounds": {}}


def args(*argv):
    return build_parser().parse_args(["plan", "--config", "c.json", "--work", "w", *argv])


class Query(unittest.TestCase):
    def test_weights_normalize_and_alias(self):
        self.assertEqual(parse_weights("loss=.4,decode=.3,short=.2,long=.1"),
                         {"precision": .4, "decode": .3, "short": .2, "long": .1})
        out = apply_query(cfg(), args("--weights", "loss=2,decode=2"))
        self.assertEqual(out["importance"], {"precision": .5, "decode": .5, "short": 0, "long": 0})

    def test_bad_weights(self):
        for text in ("speed=1", "loss=0", "loss=-1,decode=2", "loss", "loss=1,loss=2", "loss=x"):
            with self.assertRaises(QuantError, msg=text):
                parse_weights(text)

    def test_minimize_kld_is_precision_only(self):
        out = apply_query(cfg(), args("--minimize", "kld"))
        self.assertEqual(out["importance"], {"precision": 1.0, "decode": 0.0, "short": 0.0, "long": 0.0})
        with self.assertRaises(QuantError):
            apply_query(cfg(), args("--minimize", "kld", "--weights", "loss=1"))

    def test_size_mode(self):
        out = apply_query(cfg(), args("--minimize", "size", "--max-kld", "0.05", "--max-size", "400GiB"))
        self.assertEqual(out["search"]["minimize"], "size")
        self.assertEqual(out["importance"]["precision"], 1.0)
        self.assertEqual(out["bounds"], {"max_loss": 0.05, "max_total_bytes": 400 * GIB})
        with self.assertRaises(QuantError):
            apply_query(cfg(), args("--minimize", "size"))
        with self.assertRaises(QuantError):
            apply_query(cfg(), args("--minimize", "size", "--max-kld", "0.05", "--weights", "loss=1"))

    def test_budget_and_bounds(self):
        out = apply_query(cfg(), args("--budget", "ROCm1=20GiB,RAM=400GiB", "--max-kld", "0.05", "--min-decode", "30"))
        self.assertEqual(out["devices"], {"ROCm1": 20 * GIB, "RAM": 400 * GIB})
        self.assertEqual(out["bounds"], {"max_loss": 0.05, "min_decode_t_s": 30.0})
        with self.assertRaises(QuantError):
            apply_query(cfg(), args("--budget", "CUDA0=1GiB"))
        with self.assertRaises(QuantError):
            apply_query(cfg(), args("--budget", "RAM=0.5GiB"))  # below the 1 GiB reserve

    def test_max_kld_needs_a_table(self):
        c = cfg()
        del c["quality"]
        with self.assertRaises(QuantError):
            apply_query(c, args("--max-kld", "0.05"))

    def test_original_is_untouched(self):
        c = cfg()
        apply_query(c, args("--budget", "RAM=50GiB", "--max-kld", "0.1"))
        self.assertEqual(c["devices"]["RAM"], 100 * GIB)
        self.assertEqual(c["bounds"], {})


class Frontier(unittest.TestCase):
    def setUp(self):
        self.rows = fr.rows(payload())

    def test_rows(self):
        r = self.rows[0]
        self.assertAlmostEqual(r["size"] / GIB, 24.0)  # 5 + 20 GiB minus the 1 GiB reserve
        self.assertAlmostEqual(r["decode"], 1000 / 20)
        self.assertIn("IQ4_XS", r["experts"])

    def test_thin_keeps_extremes_and_best(self):
        shown = fr.thin(self.rows, 8)
        self.assertEqual(len(shown), 8)
        self.assertEqual(len({r["id"] for r in shown}), 8)
        ids = {r["id"] for r in shown}
        for key, small in (("kld", True), ("size", True), ("decode", False), ("short", False), ("long", False)):
            self.assertIn((min if small else max)(self.rows, key=lambda r: r[key])["id"], ids, key)
        self.assertIn(min(self.rows, key=lambda r: r["score"])["id"], ids)
        self.assertEqual([r["size"] for r in shown], sorted(r["size"] for r in shown))

    def test_thin_small_inputs(self):
        self.assertEqual(len(fr.thin(self.rows[:5], 8)), 5)
        self.assertEqual(len(fr.thin(self.rows, 3)), 3)

    def test_pick(self):
        shown = fr.thin(self.rows, 6)
        self.assertEqual(fr.pick(shown, self.rows, "2"), shown[1])
        self.assertEqual(fr.pick(shown, self.rows, "recipe-005")["id"], "recipe-005")
        self.assertEqual(fr.pick(shown, self.rows, "min-kld")["id"], "recipe-039")
        self.assertEqual(fr.pick(shown, self.rows, "min-size")["id"], "recipe-000")
        self.assertEqual(fr.pick(shown, self.rows, "best")["id"], "recipe-012")
        for bad in ("0", "7", "nope"):
            with self.assertRaises(ValueError, msg=bad):
                fr.pick(shown, self.rows, bad)

    def test_table_and_detail(self):
        shown = fr.thin(self.rows, 6)
        text = fr.table(shown)
        self.assertEqual(len(text.splitlines()), 7)
        self.assertEqual(text.count("*"), 1)
        d = fr.detail(self.rows[1]["recipe"])
        self.assertEqual(d["ffn_down_exps"], "IQ4_XS x1, Q3_K x1")
        self.assertEqual(d["attn_q"], "Q8_0 x1")


if __name__ == "__main__":
    unittest.main()
