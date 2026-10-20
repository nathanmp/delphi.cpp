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


if __name__ == "__main__":
    unittest.main()
