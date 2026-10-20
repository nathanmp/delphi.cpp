#!/usr/bin/env python3
"""Offline checks of the quantization CLI conveniences: progress lines, scratch estimates, capacities, state. No inference."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/auto-quantize"))
import progress
import space
from config import QuantError

GIB = 1 << 30


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class ProgressLines(unittest.TestCase):
    def test_text_helpers(self):
        self.assertEqual(progress.duration_text(3725), "01:02:05")
        self.assertEqual(progress.duration_text(90000), "1d 01:00:00")
        self.assertEqual(progress.size_text(1536 * 1024), "1.5 MiB")
        self.assertEqual(progress.size_text(5), "5 B")

    def test_eta_is_linear_and_needs_progress(self):
        self.assertEqual(progress.eta_seconds(25, 100, 50), 150)
        self.assertIsNone(progress.eta_seconds(0, 100, 50))
        self.assertIsNone(progress.eta_seconds(5, None, 50))
        self.assertEqual(progress.eta_seconds(120, 100, 50), 0)

    def test_line_has_stage_count_elapsed_and_finish(self):
        now = 1_000_000.0
        text = progress.format_line("prepare", 3, 10, "", 60, now, 140)
        self.assertIn("[prepare] step 3/10 (30%)", text)
        self.assertIn("elapsed 00:01:00", text)
        self.assertIn("(in 00:02:20)", text)
        self.assertNotIn("\r", text)
        text = progress.format_line("encode", 5 * GIB, 20 * GIB, "bytes", 10, now, None)
        self.assertIn("5.0 GiB/20.0 GiB (25%)", text)
        self.assertIn("finish unknown", text)

    def test_updates_are_throttled_and_the_last_one_prints(self):
        lines, clock = [], Clock()
        p = progress.Progress("s", total=100, interval=30, out=lines.append, clock=clock)
        self.assertEqual(len(lines), 1)  # start line
        for step in range(1, 100):
            clock.now += 1
            p.update(step)
        # step 1 prints at once, then one line per 30 s: steps 31, 61, 91
        self.assertEqual(len(lines) - 1, 4)
        p.update(100)
        self.assertIn("(100%)", lines[-1])
        self.assertIn("(in 00:00:00)", lines[-1])
        p.finish()
        self.assertIn("finished in 00:01:39", lines[-1])

    def test_explicit_eta_wins(self):
        lines, clock = [], Clock()
        p = progress.Progress("s", total=10, interval=0, out=lines.append, clock=clock, quiet_start=True)
        clock.now += 100
        p.update(5, eta=7)
        self.assertIn("(in 00:00:07)", lines[0])

    def test_stage_reports_a_stop(self):
        lines = []
        with self.assertRaises(RuntimeError):
            with progress.stage("x", total=3, out=lines.append):
                raise RuntimeError("boom")
        self.assertIn("stopped after", lines[-1])

    def test_watcher_heartbeat_without_counts(self):
        lines, clock = [], Clock()
        w = progress.Watcher("quiet", lambda: None, interval=30, out=lines.append, clock=clock)
        w.tick()
        clock.now += 10
        w.tick()
        clock.now += 25
        w.tick()
        self.assertEqual(sum("running, elapsed" in x for x in lines), 2)


class CalibrationFiles(unittest.TestCase):
    def record(self, directory, device, jobs):
        d = Path(directory) / f"session-{device}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(json.dumps({"jobs": [{"label": l, "seconds": s} for l, s in jobs]}))
        return d

    def test_counts_skip_controls_and_add_uniform_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.record(tmp, "ROCm0", [("floor", 30), ("recompute", 5), ("a", 4), ("b", 6)])
            d = self.record(tmp, "ROCm1", [("floor", 28), ("c", 8)])
            (d / "uniform-Q4_0.f32").write_bytes(b"x")
            done, mean, floor = progress.calibration_counts(tmp)
            self.assertEqual(done, 4)
            self.assertAlmostEqual(mean, 6.0)
            self.assertEqual(floor, 28)

    def test_watcher_phases_and_passes_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines, clock = [], Clock()
            w = progress.CalibrationWatcher(tmp, teacher_bytes=1000, interval=0, out=lines.append, clock=clock)
            self.assertIsNone(w.probe())
            (Path(tmp) / "teacher.partial.kld").write_bytes(b"x" * 250)
            self.assertEqual(w.probe(), (250, 1000, None, ""))
            self.assertEqual(w.watcher.progress.name, "calibration: teacher logits")
            (Path(tmp) / "teacher.partial.kld").unlink()
            w.on_line("Measuring 10 patch probes on ROCm0, ROCm1 and 2 uniform recipes")
            self.record(tmp, "ROCm0", [("floor", 30), ("a", 10), ("b", 10)])
            done, total, eta, _ = w.probe()
            self.assertEqual((done, total), (2, 12))
            # 10 left: 2 uniform recipes at the floor time, 8 probes at 10 s over 2 sessions
            self.assertAlmostEqual(eta, 8 * 10 / 2 + 2 * 30)
            self.assertEqual(w.watcher.progress.name, "calibration: passes")

    def test_passes_left_to_calibrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines, clock = [], Clock()
            w = progress.CalibrationWatcher(tmp, passes=False, interval=0, out=lines.append, clock=clock)
            self.record(tmp, "ROCm0", [("floor", 30), ("a", 10)])
            self.assertIs(w.probe(), False)
            w.watcher.tick()
            self.assertEqual(len(lines), 1)  # the start line only

    def test_stdout_sniffer_forwards_text(self):
        seen, out = [], io.StringIO()
        saved = sys.stdout
        sys.stdout = out
        try:
            with progress.sniff_stdout(seen.append):
                print("hello")
                print("Measuring 3 patch probes on ROCm0 and 1 uniform recipes")
        finally:
            sys.stdout = saved
        self.assertIn("hello", out.getvalue())
        self.assertEqual(len(seen), 2)

    def test_count_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "x.log"
            self.assertEqual(progress.count_matches(log, "a"), 0)
            log.write_text("blk.0.w [1, 2] q8_0, 4 experts\nnoise\nblk.1.w [1, 2] q8_0, 4 experts\n")
            self.assertEqual(progress.count_matches(log, r"^\S+ \[[0-9, ]+\] \S+,"), 2)


def info():
    """Two decision groups (an MoE expert pair and a dense matrix), a preserved norm and a lazy table."""
    return {"tensors": [
        {"name": "blk.0.ffn_gate_exps.weight", "shape": [2048, 1024, 8]},
        {"name": "blk.0.ffn_up_exps.weight", "shape": [2048, 1024, 8]},
        {"name": "blk.0.attn_q.weight", "shape": [2048, 2048]},
        {"name": "blk.0.attn_norm.weight", "shape": [2048]},
        {"name": "engram_embd.weight", "shape": [4096, 1000]}],
        "groups": [{"name": "blk.0.ffn_gate_exps.weight", "members": ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"],
                    "pool": ["Q2_K", "Q4_0", "Q8_0"]},
                   {"name": "blk.0.attn_q.weight", "members": ["blk.0.attn_q.weight"], "pool": ["Q8_0"]},
                   {"name": "blk.0.attn_norm.weight", "members": ["blk.0.attn_norm.weight"], "pool": ["F32"]},
                   {"name": "engram_embd.weight", "members": ["engram_embd.weight"], "pool": ["Q4_0"]}]}


class Space(unittest.TestCase):
    def test_teacher_size_matches_a_real_file(self):
        # llama-perplexity wrote this file: 64 chunks of 512 tokens, vocabulary 248320 (measured 2026-10-19)
        self.assertEqual(space.teacher_bytes(248320, 64, 512), 8105426452)

    def test_teacher_size_grows_with_chunks_and_vocab(self):
        base = space.teacher_bytes(1000, 8, 128)
        self.assertGreater(space.teacher_bytes(1000, 16, 128), base)
        self.assertGreater(space.teacher_bytes(2000, 8, 128), base)
        self.assertEqual(space.teacher_bytes(1001, 8, 128), space.teacher_bytes(1002, 8, 128))  # nv rounds up to even

    def test_model_sizes_include_lazy_tables(self):
        low, high, lazy = space.model_sizes(info())
        block = 2048 * 1024 * 8
        # Q2_K 84/256 B per weight, Q4_0 18/32, Q8_0 34/32; two experts tensors per group
        self.assertEqual(lazy, 4096 * 1000 // 32 * 18)
        self.assertEqual(low, 2 * block // 256 * 84 + 2048 * 2048 // 32 * 34 + 2048 * 4 + lazy)
        self.assertEqual(high, 2 * block // 32 * 34 + 2048 * 2048 // 32 * 34 + 2048 * 4 + lazy)

    def test_rows_have_one_entry_per_artifact(self):
        sizes = space.model_sizes(info())
        c = {"chunks": 64, "context": 512, "patch_mib": 100}
        rows = space.artifacts("calibrate", sizes, c, vocab=1000, ram=10 * sizes[1])
        self.assertEqual([n for n, _, _ in rows], ["patch files", "teacher logits", "records and logs"])
        self.assertTrue(all(key == "scratch" for _, key, _ in rows))
        by_name = {n: b for n, _, b in rows}
        self.assertEqual(by_name["teacher logits"], space.teacher_bytes(1000, 64, 512))
        self.assertEqual(by_name["patch files"], 200 << 20)  # the previous patch stays until the reload commits
        # a uniform recipe that does not fit in RAM adds a probe model of the largest qtypes
        rows = space.artifacts("calibrate", sizes, c, vocab=1000, ram=sizes[1] // 2)
        self.assertEqual(rows[0], ("probe model (uniform recipe)", "scratch", sizes[1]))
        self.assertEqual(space.artifacts("prepare", sizes)[0], ("baseline GGUF", "work", sizes[0]))
        self.assertEqual(space.artifacts("encode", sizes)[0][1:], ("output", sizes[1]))
        self.assertEqual(space.artifacts("encode", sizes, output_bytes=5)[0][2], 5)
        self.assertEqual(space.artifacts("convert", sizes, source_bytes=7, mmproj_bytes=3)[0][2:], (7,))
        with self.assertRaises(ValueError):
            space.artifacts("nope", sizes)

    def test_check_names_directory_bytes_needed_and_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "scratch" / "deep"  # does not exist yet: the check uses the nearest parent
            rows = [("probe model (mid)", "scratch", 30 * GIB), ("teacher logits", "scratch", 8 * GIB)]
            usage = lambda path: type("U", (), {"free": 20 * GIB})()
            with self.assertRaises(QuantError) as raised:
                space.check("calibrate", rows, {"scratch": scratch}, True, usage)
            text = str(raised.exception)
            for part in ("calibrate", str(scratch), "38.0 GiB needed", "20.0 GiB free", "probe model (mid) 30.0 GiB", "--no-space-check"):
                self.assertIn(part, text)
            space.check("calibrate", rows, {"scratch": scratch}, False, usage)  # --no-space-check
            space.check("calibrate", rows[1:], {"scratch": scratch}, True, usage)  # 8 GiB fits

    def test_directories_on_one_file_system_add_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = {"work": Path(tmp) / "work", "scratch": Path(tmp) / "scratch"}
            usage = lambda path: type("U", (), {"free": 20 * GIB})()
            rows = [("baseline GGUF", "work", 12 * GIB), ("probe", "scratch", 12 * GIB)]
            with self.assertRaises(QuantError) as raised:
                space.check("calibrate", rows, dirs, True, usage)
            self.assertIn("24.0 GiB needed", str(raised.exception))
            space.check("calibrate", [rows[0]], dirs, True, usage)

    def test_zero_rows_are_ignored(self):
        space.check("convert", [("mmproj GGUF", "scratch", 0)], {"scratch": Path("/nonexistent/x")}, True)

    def test_recipe_bytes(self):
        types = {t["name"]: "Q8_0" for t in info()["tensors"]}
        types["engram_embd.weight"] = "Q4_0"
        types["blk.0.attn_norm.weight"] = "F32"
        self.assertEqual(space.recipe_bytes(info(), types), space.model_sizes(info())[1])


if __name__ == "__main__":
    unittest.main()
