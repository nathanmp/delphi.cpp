#!/usr/bin/env python3
"""Auto-tuner invariants with fake measurements; no server or GPU is needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/auto-tune"))
import knobs as K
import score as S
from auto_tune import Budget, Tuner, plan_text
from config import DEFAULTS, _merge
from report import write
from store import Store

BASE = ("llama-server", "-m", "model.gguf", "-c", "8192", "-b", "2048", "-ub", "512", "-fa", "on", "-t", "32",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--device", "ROCm0,ROCm1")


def ctx(command=BASE, env=None):
    return K.Context(tuple(command), dict(env or {}), 64, 56)


def cfg(out, **changes):
    c = _merge(DEFAULTS, {"recipe": "r.py:f", "arm": "a", "output": str(out), "warmup_launch": False,
                          "prefill": {"text": "t.txt", "lengths": {"300": 1}, "reps": 3},
                          "weights": {"prefill": 1, "decode": 1}, "noise_margin": 0.01})
    return _merge(c, changes)


def fake_metrics(command, env):
    """decode is best at -ub 1024 and n-max 5 and gains 10 % from q8_0 KV; prefill is best at -ub 2048."""
    ub = int(K.option_value(command, "-ub"))
    nmax = int(K.option_value(command, "--spec-draft-n-max"))
    q8 = K.option_value(command, "-ctk") == "q8_0"
    decode = 50.0 * (1.0 - 0.05 * abs(ub - 1024) / 512) * (1.0 - 0.04 * abs(nmax - 5)) * (1.1 if q8 else 1.0)
    ttft = 0.30 * (1.0 + 0.1 * abs(ub - 2048) / 1024)
    return {"decode_t_s": decode, "prefill": {"300": {"ttft_s": ttft, "t_s": 300 / ttft}}, "vram": {}}


class Knobs(unittest.TestCase):
    def test_apply(self):
        ub = K.BY_NAME["-ub"]
        cmd, env = ub.apply(BASE, {}, "4096")
        self.assertEqual(K.option_value(cmd, "-ub"), "4096")
        self.assertEqual(K.option_value(cmd, "-b"), "4096")  # -b is raised to -ub
        switch = K.BY_NAME["--spec-draft-adaptive"]
        cmd, _ = switch.apply(BASE, {}, K.ON)
        self.assertEqual(switch.current(cmd, {}), K.ON)
        cmd, _ = switch.apply(cmd, {}, K.OFF)
        self.assertEqual(cmd, BASE)
        kv = K.BY_NAME["kv-type"]
        cmd, _ = kv.apply(BASE, {}, "q8_0")
        self.assertEqual((K.option_value(cmd, "-ctk"), K.option_value(cmd, "-ctv")), ("q8_0", "q8_0"))
        env_knob = K.BY_NAME["GGML_SCHED_STAGE_SLOTS"]
        _, env = env_knob.apply(BASE, {"GGML_SCHED_STAGE": "1"}, "6")
        self.assertEqual(env["GGML_SCHED_STAGE_SLOTS"], "6")
        _, env = env_knob.apply(BASE, env, None)
        self.assertNotIn("GGML_SCHED_STAGE_SLOTS", env)

    def test_select(self):
        chosen = dict((k.name, levels) for k, levels in K.select(ctx()))
        self.assertEqual(chosen["-ub"][0], "512")  # the current level first
        self.assertEqual(chosen["-t"][0], "32")
        self.assertTrue(all(int(t) <= 56 for t in chosen["-t"]))  # never all cores
        self.assertNotIn("-sm", chosen)  # opt-in
        self.assertIn("-sm", dict((k.name, v) for k, v in K.select(ctx(), include=["-sm"])))
        self.assertNotIn("GGML_SCHED_STAGE_SLOTS", chosen)  # needs staging
        self.assertEqual(set(dict((k.name, v) for k, v in K.select(ctx(), only=["-ub"]))), {"-ub"})
        synthetic = (*BASE, "--spec-synth-rates", "1,1,0")
        self.assertNotIn("--spec-draft-n-max", dict((k.name, v) for k, v in K.select(ctx(synthetic))))
        with self.assertRaises(ValueError):
            K.select(ctx(), only=["nonsense"])

    def test_exact(self):
        self.assertTrue(K.is_exact(ctx()))
        cmd, _ = K.BY_NAME["kv-type"].apply(BASE, {}, "q8_0")
        self.assertFalse(K.is_exact(ctx(cmd)))


class Score(unittest.TestCase):
    def test_score(self):
        ref = {"decode_t_s": 50.0, "prefill": {"300": {"ttft_s": 0.3}}}
        w = S.metric_weights({"prefill": 30, "decode": 70}, {"300": 1})
        self.assertAlmostEqual(w["decode"], 0.7)
        self.assertAlmostEqual(S.score(ref, ref, w), 1.0)
        faster = {"decode_t_s": 55.0, "prefill": {"300": {"ttft_s": 0.3}}}
        self.assertAlmostEqual(S.score(faster, ref, w), 1.1 ** 0.7)
        shorter = {"decode_t_s": 50.0, "prefill": {"300": {"ttft_s": 0.15}}}
        self.assertAlmostEqual(S.score(shorter, ref, w), 2.0 ** 0.3)  # TTFT: smaller is better
        self.assertIsNone(S.score({"decode_t_s": 50.0}, ref, w))


class Search(unittest.TestCase):
    def tuner(self, out, limit=None, max_launches=60, launcher=None, **changes):
        c = cfg(out, kld={"limit": limit, "text": "k.txt", "chunks": 8, "ctx": 512, "batch": 4},
                knobs={"only": ["-ub", "--spec-draft-n-max", "kv-type"], "include": [], "exclude": [], "levels": {}}, **changes)
        calls = []

        def launch(command, env, name):
            calls.append(name)
            return fake_metrics(command, env)

        def kld(command, env, key, compare):
            return 0.01 if K.option_value(command, "-ctk") == "q8_0" else 0.0

        t = Tuner(c, BASE, {}, {"test": 1}, launcher or launch, kld, budget=Budget(max_launches, 6.0), log=lambda s: None)
        return t, calls

    def test_coordinate_search_and_confirm(self):
        with tempfile.TemporaryDirectory() as d:
            t, calls = self.tuner(Path(d), weights={"prefill": 0, "decode": 1})
            state = t.run()
            best = t.store.find(state["incumbent"], "launch", "search")
            self.assertEqual(K.option_value(best["command"], "-ub"), "1024")
            self.assertEqual(K.option_value(best["command"], "--spec-draft-n-max"), "5")
            self.assertNotIn("-ctk", best["command"])  # precision knobs need a KLD limit
            self.assertTrue(state["confirm"]["kept"])
            self.assertEqual([n for n in calls if n.startswith("confirm")], ["confirm-1 best", "confirm-2 base", "confirm-3 best"])
            write(t.cfg, state, t.store, Path(d))
            self.assertIn("## Options along the trade-off", (Path(d) / "report.md").read_text())
            self.assertEqual(json.loads((Path(d) / "best.json").read_text())["changes"]["-ub"], ["512", "1024"])

    def test_precision_gate(self):
        with tempfile.TemporaryDirectory() as d:
            t, _ = self.tuner(Path(d), limit=0.02, weights={"prefill": 0, "decode": 1})
            state = t.run()
            best = t.store.find(state["incumbent"], "launch", "search")
            self.assertEqual(K.option_value(best["command"], "-ctk"), "q8_0")
            self.assertAlmostEqual(best["kld"], 0.01)
        with tempfile.TemporaryDirectory() as d:
            t, _ = self.tuner(Path(d), limit=0.005, weights={"prefill": 0, "decode": 1})
            state = t.run()
            best = t.store.find(state["incumbent"], "launch", "search")
            self.assertNotIn("-ctk", best["command"])
            self.assertTrue(any(r.get("status") == "kld-over" for r in t.store.records))

    def test_weights_change_the_winner(self):
        with tempfile.TemporaryDirectory() as d:
            t, _ = self.tuner(Path(d), weights={"prefill": 1, "decode": 0})
            state = t.run()
            best = t.store.find(state["incumbent"], "launch", "search")
            self.assertEqual(K.option_value(best["command"], "-ub"), "2048")

    def test_budget(self):
        with tempfile.TemporaryDirectory() as d:
            t, calls = self.tuner(Path(d), max_launches=6)
            state = t.run()
            self.assertLessEqual(len(calls), 6)
            self.assertTrue(state["untested"])
            self.assertIn("--spec-draft-n-max", state["untested"])

    def test_crash_journal_and_failures(self):
        with tempfile.TemporaryDirectory() as d:
            def launch(command, env, name):
                if K.option_value(command, "-ub") == "4096":
                    raise RuntimeError("out of memory")
                return fake_metrics(command, env)
            t, _ = self.tuner(Path(d), launcher=launch)
            key = t.key(*K.BY_NAME["-ub"].apply(BASE, {}, "256"))
            t.store.start(key, "crash")  # a launch that never ended
            t.store.crashed = Store(Path(d)).crashed
            t.run()
            status = {r["name"]: r["status"] for r in t.store.records if r["kind"] == "launch"}
            self.assertEqual(status["01 -ub=256"], "crashed")
            self.assertEqual(status["04 -ub=4096"], "failed")
            # a second run reuses every search measurement
            t2, calls2 = self.tuner(Path(d), launcher=launch)
            t2.run()
            self.assertTrue(all(name.startswith("confirm") for name in calls2))

    def test_plan(self):
        with tempfile.TemporaryDirectory() as d:
            t, _ = self.tuner(Path(d), limit=0.02)
            text = plan_text(t)
            self.assertIn("kv-type", text)
            self.assertIn("-ub", text)


if __name__ == "__main__":
    unittest.main()
