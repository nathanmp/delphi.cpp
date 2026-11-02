#!/usr/bin/env python3
"""Tests of tools/auto-tune that need no server: command editing, prompt banks, drafter choice, scoring."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "auto-tune"))

import cmdline as C  # noqa: E402
import prompts as PB  # noqa: E402
import tune as T  # noqa: E402


class CommandTest(unittest.TestCase):
    def test_set_and_remove(self):
        cmd = ("llama-server", "-m", "a.gguf", "-ub", "512", "--port=8080")
        self.assertEqual(C.set_option(cmd, "-ub", "1024")[4], "1024")
        self.assertEqual(C.set_option(cmd, "--port", "1")[-1], "--port=1")
        self.assertEqual(C.set_option(cmd, "-b", "4096")[-2:], ("-b", "4096"))
        self.assertNotIn("-ub", C.set_option(cmd, "-ub", None))

    def test_strip_spec_keeps_draft_model(self):
        cmd = ("s", "-md", "d.gguf", "--spec-draft-device", "ROCm0", "--spec-type", "draft-simple,ngram-mod",
               "--spec-draft-n-max", "4", "--spec-draft-adaptive", "--spec-ngram-mod-n-match=16", "-c", "8192")
        self.assertEqual(C.strip_spec(cmd), ("s", "-md", "d.gguf", "--spec-draft-device", "ROCm0", "-c", "8192"))

    def test_spec_options(self):
        cmd = ("s", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--no-spec-draft-adaptive", "--spec-draft-p-min=0.5")
        self.assertEqual(T.spec_options(cmd), {"--spec-draft-n-max": "3", "--no-spec-draft-adaptive": None,
                                               "--spec-draft-p-min": "0.5"})
        self.assertEqual(C.spec_args({"--spec-draft-n-max": "3", "--no-spec-draft-adaptive": None}),
                         ["--spec-draft-n-max", "3", "--no-spec-draft-adaptive"])


class DrafterTest(unittest.TestCase):
    def test_draft_model_wins(self):
        self.assertEqual(T.find_drafter(("s", "-m", "x.gguf", "-md", "d.gguf", "--spec-type", "draft-dflash")), "draft-dflash")
        self.assertEqual(T.find_drafter(("s", "-m", "x.gguf", "--spec-draft-model", "d.gguf")), "draft-simple")

    def test_mtp_from_command(self):
        self.assertEqual(T.find_drafter(("s", "-m", "x.gguf", "--spec-type", "ngram-mod,draft-mtp")), "draft-mtp")

    def test_none_without_model_file(self):
        self.assertIsNone(T.find_drafter(("s", "-m", "/nonexistent.gguf")))


class PromptTest(unittest.TestCase):
    def test_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.yaml"
            p.write_text("- domain: math\n  id: a\n  prompt: |\n    1+1?\n  answer: 2\n- domain: chat\n  id: b\n  prompt: hi\n")
            bank = PB.load(p)
            self.assertEqual([(x.domain, x.id, x.answer) for x in bank], [("math", "a", "2"), ("chat", "b", None)])

    def test_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.txt"
            p.write_text("one\nline two\n---\n\n---\nthree\n")
            self.assertEqual([x.prompt for x in PB.load(p)], ["one\nline two", "three"])

    def test_subset_and_domains(self):
        bank = [PB.Prompt(dm, f"{dm}{i}", "x") for dm in ("a", "b") for i in range(3)]
        self.assertEqual([p.id for p in PB.subset(bank, 2)], ["a0", "a1", "b0", "b1"])
        self.assertEqual(PB.domains(bank), ["a", "b"])

    def test_default_bank_loads(self):
        bank = PB.load(Path(__file__).resolve().parents[1] / "tools" / "auto-tune" / "prompts" / "default.yaml")
        self.assertGreaterEqual(len(PB.domains(bank)), 8)


class ScoreTest(unittest.TestCase):
    def test_decode_summary(self):
        rows = [{"domain": "a", "n": 100, "ms": 1000, "draft_n": 50, "draft_accepted": 25},
                {"domain": "a", "n": 100, "ms": 1000, "draft_n": 50, "draft_accepted": 25},
                {"domain": "b", "n": 400, "ms": 1000, "draft_n": 0, "draft_accepted": 0}]
        s = T.decode_summary(rows)
        self.assertAlmostEqual(s["domains"]["a"]["tps"], 100.0)
        self.assertAlmostEqual(s["domains"]["a"]["accept"], 0.5)
        self.assertIsNone(s["domains"]["b"]["accept"])
        self.assertAlmostEqual(s["tps"], 200.0)  # geometric mean of 100 and 400

    def test_merge_spec_switches(self):
        m = T.merge_spec({"--spec-draft-n-max": "7", "--spec-draft-adaptive": None}, {"--no-spec-draft-adaptive": None})
        self.assertEqual(m, {"--spec-draft-n-max": "7", "--no-spec-draft-adaptive": None})

    def test_thread_levels_capped(self):
        levels = T.thread_levels(None, 2, 10)
        self.assertTrue(levels and max(levels) <= 10)


if __name__ == "__main__":
    unittest.main()
