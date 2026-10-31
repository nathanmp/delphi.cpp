#!/usr/bin/env python3
"""Offline checks of the quantization CLI conveniences: progress lines, scratch estimates, capacities, state. No inference."""
import argparse
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/auto-quantize"))
import capacity as cap
import convert as cv
import frontier as fr
import progress
import quant_cli as qc
import runstate
import space
import verify as vf
from config import QuantError, write_json

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

    def test_the_last_step_prints_once(self):
        lines, clock = [], Clock()
        p = progress.Progress("x", 4, interval=0, out=lines.append, clock=clock, quiet_start=True)
        for _ in range(3):
            p.update(4)
        self.assertEqual(len(lines), 1)

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

    def test_foreign_pass_lines_silence_the_watcher_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = []
            w = progress.CalibrationWatcher(tmp, interval=0, out=lines.append, clock=Clock())
            w.on_line("Measuring 4 patch probes on ROCm0 and 2 uniform recipes")
            self.record(tmp, "ROCm0", [("floor", 30), ("a", 10)])
            self.assertEqual(w.probe()[:2], (1, 6))
            w.on_line("[calibration passes] pass 2/6 elapsed 00:01:00")
            self.assertIs(w.probe(), False)
            own = lines[0]
            w.on_line(own)  # its own lines never count as foreign
            self.assertTrue(w.foreign)

    def test_bytes_phases_print_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = []
            w = progress.CalibrationWatcher(tmp, teacher_bytes=2 << 20, interval=0, out=lines.append, clock=Clock())
            (Path(tmp) / "teacher.partial.kld").write_bytes(b"x" * (1 << 20))
            w.watcher.tick()
            self.assertIn("1.0 MiB/2.0 MiB (50%)", lines[-1])

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
        # with the top probe qtype known, the row is that recipe and appears only when its experts do not fit in RAM
        probe = space.probe_sizes(info(), "Q4_0")
        block = 2048 * 1024 * 8
        self.assertEqual(probe[1], 2 * block // 32 * 18)
        self.assertEqual(probe[0], probe[1] + 2048 * 2048 // 32 * 34 + 2048 * 4 + 4096 * 1000 // 32 * 18)
        roomy = probe[1] + space.UNIFORM_RAM_MARGIN
        self.assertNotIn("probe model (uniform recipe)",
                         [n for n, _, _ in space.artifacts("calibrate", sizes, c, vocab=1000, ram=roomy, probe=probe)])
        rows = space.artifacts("calibrate", sizes, c, vocab=1000, ram=roomy - 1, probe=probe)
        self.assertEqual(rows[0], ("probe model (uniform recipe)", "scratch", probe[0]))
        rows = space.artifacts("calibrate", sizes, c, vocab=1000, ram=roomy - 1, probe=probe, linked_lazy=True)
        self.assertEqual(rows[0][2], probe[0] - sizes[2])
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


LISTED = """Available devices:
  ROCm0: AMD Radeon AI PRO R9700 (32624 MiB, 32526 MiB free)
  ROCm1: AMD Radeon AI PRO R9700 (32624 MiB, 30000 MiB free)
  ROCm2: AMD Radeon AI PRO R9700 (32624 MiB, 31000 MiB free)
"""


class Capacity(unittest.TestCase):
    def rows(self, devices, keep=None, mode="free", ram=None):
        names = [n for n, _ in cap.parse_devices(devices)]
        return cap.capacity_rows(cap.parse_devices(devices), cap.parse_listed(LISTED), cap.parse_keep_free(keep, names), mode, ram)

    def test_listing_is_parsed_to_bytes(self):
        listed = cap.parse_listed(LISTED)
        self.assertEqual(sorted(listed), ["ROCm0", "ROCm1", "ROCm2"])
        self.assertEqual(listed["ROCm1"], {"description": "AMD Radeon AI PRO R9700", "total": 32624 << 20, "free": 30000 << 20})

    def test_capacity_is_free_minus_default_keep_free(self):
        rows = self.rows("ROCm1,ROCm0")
        self.assertEqual([r["device"] for r in rows], ["ROCm1", "ROCm0"])  # the order of --devices is the layer order
        self.assertEqual(rows[0]["capacity"], (30000 - 512) << 20)
        self.assertEqual(rows[1]["capacity"], (32526 - 512) << 20)

    def test_keep_free_global_and_per_device(self):
        self.assertEqual({r["capacity"] for r in self.rows("ROCm0", "1GiB")}, {(32526 - 1024) << 20})
        rows = self.rows("ROCm2,ROCm0", "ROCm2=2GiB")
        self.assertEqual(rows[0]["capacity"], (31000 - 2048) << 20)
        self.assertEqual(rows[1]["capacity"], (32526 - 512) << 20)  # not named: the default
        with self.assertRaises(QuantError):
            cap.parse_keep_free("ROCm9=1GiB", ["ROCm0"])

    def test_total_mode_ignores_what_other_jobs_hold(self):
        rows = self.rows("ROCm1", mode="total")
        self.assertEqual(rows[0]["capacity"], (32624 - 512) << 20)

    def test_explicit_capacity_wins(self):
        rows = self.rows("ROCm0=24GiB,ROCm1", "4GiB")
        self.assertEqual((rows[0]["capacity"], rows[0]["source"]), (24 << 30, "explicit"))
        self.assertEqual(rows[1]["capacity"], (30000 << 20) - (4 << 30))

    def test_host_memory_and_errors(self):
        rows = self.rows("ROCm0,RAM", ram=(500 << 30, 400 << 30))
        self.assertEqual(rows[1]["capacity"], (400 << 30) - (512 << 20))
        with self.assertRaises(QuantError):
            self.rows("ROCm0,RAM")  # host memory unknown and no size given
        with self.assertRaises(QuantError):
            self.rows("ROCm7")
        for bad in ("RAM", "ROCm0,ROCm0", "gpu0", "ROCm0=x"):
            with self.assertRaises(QuantError, msg=bad):
                cap.parse_devices(bad)

    def test_table_text_has_one_row_per_device(self):
        text = cap.table_text(self.rows("ROCm0,ROCm1", "ROCm1=2GiB"))
        self.assertEqual(len(text.splitlines()), 3)
        for part in ("device", "capacity", "ROCm1", "2.0 GiB"):
            self.assertIn(part, text)

    def test_initial_split_follows_capacity_and_sums_to_layers_plus_output(self):
        split = cap.initial_split(47, [30 << 30, 30 << 30, 15 << 30])
        self.assertEqual(sum(split), 48)
        self.assertEqual(split, [19, 19, 10])
        self.assertEqual(cap.initial_split(10, [1, 1]), [6, 5])

    def test_fit_print_and_serving_args(self):
        text = "ROCm3 29125 2351 2973 \nROCm0 30619 2926 3097 \nHost 28110 0 1563 \nmmproj ROCm3 1114\nmmproj CPU 25\n"
        devices, mmproj, mtp = cap.parse_fit_print(text + "mtp ROCm3 0 0\nmtp ROCm0 317 454\n")
        self.assertEqual(devices["ROCm0"], (30619 << 20, 2926 << 20, 3097 << 20))
        self.assertEqual(mmproj, {"ROCm3": 1114 << 20, "CPU": 25 << 20})
        self.assertEqual(mtp, {"ROCm3": 0, "ROCm0": (317 + 454) << 20})
        self.assertEqual(set(devices), {"ROCm3", "ROCm0", "Host"})
        args = cap.serving_args({"ctx": 8192, "ub": 512, "parallel": 2, "ctk": "q8_0", "ctv": None})
        self.assertEqual(args, ["-c", "8192", "-ub", "512", "-np", "2", "-ctk", "q8_0"])

    def test_dynamic_projector_reserves_only_what_exceeds_the_compute_buffers(self):
        _, mmproj, _ = cap.parse_fit_print("mmproj-dynamic CUDA0 1500\nmmproj CPU 25\n")
        self.assertEqual(mmproj, {"CUDA0": ("dynamic", 1500 << 20), "CPU": 25 << 20})
        self.assertEqual(cap.projector_reserve(mmproj["CUDA0"], 6600 << 20), 0)
        self.assertEqual(cap.projector_reserve(mmproj["CUDA0"], 1000 << 20), (1500 + 256 - 1000) << 20)
        self.assertEqual(cap.projector_reserve(1114 << 20, 6600 << 20), 1114 << 20)


class KldCorrection(unittest.TestCase):
    def cfg(self):
        return {"search": {}, "bounds": {}, "importance": {}, "devices": {"ROCm0": 10 * GIB}, "reserve": {"ROCm0": GIB},
                "quality": {"table": "q.json"}}

    def args(self, **kw):
        values = dict(weights=None, minimize="score", max_size=None, max_kld=0.05, budget=None, kld_correction="ratio",
                      min_decode=None, min_short=None, min_long=None)
        return argparse.Namespace(**{**values, **kw})

    def test_ceiling_is_divided_by_the_ratio(self):
        self.assertAlmostEqual(vf.corrected_ceiling(0.06, 1.2), 0.05)
        self.assertEqual(vf.corrected_ceiling(0.06, None), 0.06)
        out = io.StringIO()
        with unittest.mock.patch("sys.stdout", out):
            cfg = qc.apply_query(self.cfg(), self.args(), 1.25)
        self.assertAlmostEqual(cfg["bounds"]["max_loss"], 0.04)
        self.assertIn("ratio 1.250", out.getvalue())

    def test_none_turns_the_correction_off(self):
        cfg = qc.apply_query(self.cfg(), self.args(kld_correction="none"), 1.25)
        self.assertEqual(cfg["bounds"]["max_loss"], 0.05)
        self.assertEqual(qc.apply_query(self.cfg(), self.args(), None)["bounds"]["max_loss"], 0.05)

    def test_ratio_is_stored_and_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(vf.load_ratio(d))
            vf.store_ratio(d, {"ratio": vf.kl_ratio(0.06, 0.05), "measured": 0.06})
            self.assertAlmostEqual(vf.load_ratio(d), 1.2)
            (Path(d) / vf.RATIO_FILE).write_text('{"ratio": -1}')
            self.assertIsNone(vf.load_ratio(d))
        with self.assertRaises(QuantError):
            vf.kl_ratio(0.0, 0.05)


@unittest.mock.patch("verify.lazy_args", lambda cfg: ["-lm", "mmap", "-ot", "engram=CPU"])
class Verify(unittest.TestCase):
    def cfg(self):
        return {"bindir": "/bin", "model": "/m.gguf", "runtime": {"threads": 16, "env": {"HIP_VISIBLE_DEVICES": "0"}},
                "devices": {"ROCm0": 1, "ROCm1": 1}, "cache_mib": 0}

    def serve(self, **kw):
        return {"ctx": 8192, "ub": 512, "parallel": 1, "ctk": None, "ctv": None, "mmproj": None, "draft": None, **kw}

    @unittest.mock.patch("capacity.has_mtp", lambda model: False)
    def test_command_has_layout_context_and_mmproj(self):
        command = vf.server_command(self.cfg(), "/o.gguf", ["--device", "ROCm0,ROCm1", "-sm", "layer", "-ts", "9,8"],
                                    self.serve(mmproj="/p.gguf", ctk="q8_0"), 9000)
        text = " ".join(command)
        for part in ("-m /o.gguf", "--port 9000", "--fit off", "-c 8192", "-ctk q8_0", "-ts 9,8", "--mmproj /p.gguf --mmproj-memory dynamic",
                     "-t 16", "-lm mmap -ot engram=CPU"):
            self.assertIn(part, text)
        self.assertNotIn("-cmoe", command)

    def test_ram_device_routes_experts_and_draft_is_passed(self):
        cfg = self.cfg()
        cfg["devices"]["RAM"] = 1
        command = vf.server_command(cfg, "/o.gguf", None, self.serve(draft="/d.gguf"), 1)
        self.assertIn("-cmoe", command)
        self.assertEqual(command[command.index("--device") + 1], "ROCm0")
        self.assertEqual(command[command.index("--spec-draft-model") + 1], "/d.gguf")
        self.assertEqual(command[command.index("--spec-type") + 1], "draft-mtp")

    def test_mtp_is_served_unless_turned_off(self):
        with unittest.mock.patch.object(cap, "has_mtp", lambda model: True):
            command = vf.server_command(self.cfg(), "/o.gguf", None, self.serve(), 1)
            self.assertEqual(command[command.index("--spec-type") + 1], "draft-mtp")
            self.assertNotIn("--spec-type", vf.server_command(self.cfg(), "/o.gguf", None, self.serve(mtp=False), 1))
        with unittest.mock.patch.object(cap, "has_mtp", lambda model: False):
            self.assertNotIn("--spec-type", vf.server_command(self.cfg(), "/o.gguf", None, self.serve(), 1))

    def test_runtime_memory_is_measured_and_read_back(self):
        cfg = {**self.cfg(), "devices": {"ROCm0": 10 * GIB, "ROCm1": 10 * GIB}, "runtime_reserve": {"ROCm0": 256 << 20, "ROCm1": 256 << 20}}
        row = {"recipe": {"memory": {"used_bytes": {"ROCm0": 9 * GIB, "ROCm1": 10 * GIB}}}}
        keep = {"ROCm0": 512 << 20, "ROCm1": 512 << 20}
        memory = [{"device": "ROCm0", "free": 1536 << 20}, {"device": "ROCm1", "free": 100 << 20}]
        # ROCm0 used exactly its plan; ROCm1 used 412 MiB more than planned
        measured = vf.measured_runtime(cfg, row, keep, memory)
        self.assertEqual(measured, {"ROCm0": 256 << 20, "ROCm1": 704 << 20})
        self.assertIsNone(vf.measured_runtime(cfg, row, keep, [{"device": "ROCm0", "free": None}]))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cap.runtime_reserves(d, ["ROCm0", "ROCm1"], True, 1024),
                             {"ROCm0": cap.RUNTIME_RESERVE_MTP, "ROCm1": cap.RUNTIME_RESERVE_MTP_BLOCK})
            write_json(Path(d) / cap.RUNTIME_FILE, {"mtp": True, "ub": 1024, "per_device": measured})
            self.assertEqual(cap.runtime_reserves(d, ["ROCm0", "ROCm1"], True, 1024), measured)
            self.assertEqual(cap.runtime_reserves(d, ["ROCm1"], False, 1024), {"ROCm1": cap.RUNTIME_RESERVE})
            self.assertEqual(cap.runtime_reserves(d, ["ROCm1"], True, 512), {"ROCm1": cap.RUNTIME_RESERVE_MTP_BLOCK})

    def test_command_text_can_be_pasted(self):
        text = vf.command_text(["a", "b c"], {"X": "1"})
        self.assertEqual(text, "X=1 a 'b c'")

    def test_keep_free_report(self):
        rows = [{"device": "ROCm0", "keep_free": 512 << 20}, {"device": "ROCm1", "keep_free": 512 << 20}, {"device": "RAM", "keep_free": 0}]
        report = vf.keep_free_report(rows, {"ROCm0": {"free": 600 << 20}, "ROCm1": {"free": 100 << 20}})
        self.assertEqual([m["ok"] for m in report], [True, False])
        self.assertIn("BELOW keep-free", vf.memory_text(report))
        # the device list fails when a GPU has no room for the context of another process
        report = vf.keep_free_report(rows, None)
        self.assertEqual([m["ok"] for m in report], [False, False])
        self.assertIn("unknown", vf.memory_text(report))

    def test_long_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            text = Path(d) / "text.txt"
            text.write_text("word " * 1000)
            seen = []

            def ask(url, payload=None, timeout=600):
                seen.append(len(payload["content"]))
                return {"tokens": list(range(len(payload["content"]) // 5))}
            with unittest.mock.patch.object(vf, "ask", ask):
                tokens = vf.long_prompt_tokens({"calibration": {"text": [{"path": str(text)}]}}, "http://x",
                                               {"ctx": 256 + 100, "parallel": 1})
                self.assertEqual(len(tokens), 100)
                self.assertEqual(seen, [800])
                self.assertIsNone(vf.long_prompt_tokens({"calibration": {"text": None}}, "http://x", {"ctx": 4096}))
                self.assertIsNone(vf.long_prompt_tokens({"calibration": {"text": str(text)}}, "http://x", {"ctx": 200}))


class Diversity(unittest.TestCase):
    def row(self, i, kld, size, decode, score):
        return {"id": f"r{i}", "kld": kld, "size": size, "decode": decode, "short": decode * 10, "long": decode * 5, "score": score}

    def test_best_and_min_kld_rows_are_always_shown(self):
        rows = [self.row(1, 0.05, 10, 30, 0.5), self.row(2, 0.01, 20, 20, 0.9), self.row(3, 0.03, 15, 25, 0.7)]
        shown = fr.diverse(rows, 2)
        self.assertEqual({r["id"] for r in shown}, {"r1", "r2"})
        self.assertEqual([r["id"] for r in shown], ["r1", "r2"])  # sorted by size

    def test_near_duplicates_are_skipped(self):
        base = self.row(1, 0.05, 100, 30, 0.5)
        twin = self.row(2, 0.0505, 101, 30.2, 0.51)
        far = self.row(3, 0.08, 80, 35, 0.6)
        shown = fr.diverse([base, twin, far, self.row(4, 0.01, 200, 10, 0.9)], 10)
        ids = {r["id"] for r in shown}
        self.assertIn("r1", ids)
        self.assertNotIn("r2", ids)
        self.assertIn("r3", ids)
        self.assertTrue(fr.near(base, twin, 0.02))
        self.assertFalse(fr.near(base, far, 0.02))

    def test_count_is_capped_and_table_marks_the_best(self):
        rows = [self.row(i, 0.01 * i, 10 * i, 50 - i * 3, i / 10) for i in range(1, 13)]
        shown = fr.diverse(rows, 5)
        self.assertEqual(len(shown), 5)
        lines = fr.run_table(shown).splitlines()
        self.assertEqual(len(lines), 6)
        self.assertEqual(sum("*" in line for line in lines[1:]), 1)


class ResumeState(unittest.TestCase):
    def test_new_stage_runs_then_skips_with_same_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            self.assertEqual(state.decide("prepare", {"a": 1}, "x"), ("run", []))
            state.mark("prepare", {"a": 1}, "x", 3.2, [])
            self.assertEqual(runstate.State(d).decide("prepare", {"a": 1}, "x"), ("skip", []))

    def test_changed_input_reruns_and_names_the_key(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            state.mark("prepare", {"a": 1, "b": 2}, "x", 1)
            self.assertEqual(state.decide("prepare", {"a": 1, "b": 3}, "x"), ("rerun", ["b"]))
            self.assertEqual(state.decide("prepare", {"a": 1, "b": 2}, "y"), ("rerun", ["earlier stage"]))

    def test_missing_output_runs_the_stage_again(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            state.mark("encode", {}, None, 1, [Path(d) / "out.gguf"])
            decision, why = state.decide("encode", {}, None)
            self.assertEqual(decision, "run")
            self.assertIn("missing output", why[0])
            (Path(d) / "out.gguf").write_text("x")
            self.assertEqual(state.decide("encode", {}, None)[0], "skip")

    def test_forget_drops_the_stage_and_marks_later_ones_stale(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            for name in runstate.STAGES:
                state.mark(name, {"k": 1}, None, 1)
            state.forget_from("calibrate")
            stages = runstate.State(d).stages
            self.assertEqual(sorted(stages), ["convert", "encode", "plan", "prepare", "verify"])
            self.assertTrue(all(stages[n].get("stale") for n in ("plan", "encode", "verify")))
            self.assertEqual(state.decide("plan", {"k": 2}, None), ("rerun", ["k"]))
            # the same fingerprint as before: the outputs of the stage are still the ones of its inputs
            self.assertEqual(state.decide("plan", {"k": 1}, None), ("skip", []))

    def test_started_stage_resumes_with_same_inputs_only(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            state.start("calibrate", {"t": 1}, "x")
            self.assertEqual(runstate.State(d).decide("calibrate", {"t": 1}, "x"), ("resume", []))
            self.assertEqual(runstate.State(d).decide("calibrate", {"t": 2}, "x"), ("rerun", ["t"]))

    def test_later_stage_removes_its_files_after_an_earlier_change(self):
        # A new source in the same work directory: the files of prepare and calibrate are from the old model
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            resets = []

            def go(name, inputs, previous, foreign=None):
                with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                    fp, _ = qc.run_stage(state, name, inputs, previous, lambda: None, reset=lambda: resets.append(name),
                                         foreign=foreign)
                return fp, out.getvalue()
            fp0, _ = go("convert", {"src": "a"}, None)
            fp1, _ = go("prepare", {"m": "a"}, fp0)
            go("calibrate", {"t": 1}, fp1)
            self.assertEqual(resets, [])
            fp0, _ = go("convert", {"src": "b"}, None)
            fp1, _ = go("prepare", {"m": "b"}, fp0)
            _, out = go("calibrate", {"t": 1}, fp1)
            self.assertEqual(resets, ["convert", "prepare", "calibrate"])
            self.assertIn("earlier stage", out)

    def test_foreign_files_without_a_record_are_reset(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            resets = []
            with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                qc.run_stage(state, "prepare", {"m": 1}, None, lambda: None, reset=lambda: resets.append(1),
                             foreign=lambda: "prepared.json is for other inputs")
            self.assertEqual(resets, [1])
            self.assertIn("prepared.json is for other inputs", out.getvalue())
            with unittest.mock.patch("sys.stdout", io.StringIO()):
                qc.run_stage(runstate.State(d), "calibrate", {"t": 1}, None, lambda: None, reset=lambda: resets.append(2),
                             foreign=lambda: None)
            self.assertEqual(resets, [1])

    def test_run_stage_chain_skips_finished_and_reruns_after_a_change(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            ran = []

            def go(name, inputs, previous):
                with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                    fp, _ = qc.run_stage(state, name, inputs, previous, lambda: ran.append(name))
                return fp, out.getvalue()
            fp1, _ = go("prepare", {"m": 1}, None)
            fp2, _ = go("calibrate", {"t": 1}, fp1)
            self.assertEqual(ran, ["prepare", "calibrate"])
            fp1, out = go("prepare", {"m": 1}, None)
            fp2, out2 = go("calibrate", {"t": 1}, fp1)
            self.assertEqual(ran, ["prepare", "calibrate"])
            self.assertIn("skipping", out2)
            fp1, out = go("prepare", {"m": 2}, None)
            self.assertIn("input changed (m)", out)
            self.assertTrue(state.stages["calibrate"]["stale"])
            go("calibrate", {"t": 1}, fp1)
            self.assertEqual(ran, ["prepare", "calibrate", "prepare", "calibrate"])


    def test_stale_stage_with_the_same_fingerprint_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            ran = []

            def go(name, inputs, previous):
                with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                    fp, _ = qc.run_stage(state, name, inputs, previous, lambda: ran.append(name))
                return fp, out.getvalue()
            go("prepare", {"ub": 512}, None)
            go("calibrate", {"t": 1}, "converted")
            go("prepare", {"ub": 256}, None)
            self.assertTrue(state.stages["calibrate"]["stale"])
            _, out = go("calibrate", {"t": 1}, "converted")
            self.assertIn("skipping", out)
            self.assertEqual(ran, ["prepare", "calibrate", "prepare"])
            self.assertNotIn("stale", runstate.State(d).stages["calibrate"])
            go("prepare", {"ub": 128}, None)
            _, out = go("calibrate", {"t": 2}, "converted")
            self.assertIn("input changed (t)", out)

    def test_interrupted_stage_continues_after_an_earlier_stage_ran_again(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            state.mark("prepare", {"ub": 512}, None, 1)
            state.start("calibrate", {"t": 1}, "converted")
            state.forget_from("prepare")
            self.assertEqual(state.decide("calibrate", {"t": 1}, "converted"), ("resume", []))
            state.forget_from("prepare")
            self.assertEqual(state.decide("calibrate", {"t": 2}, "converted"), ("rerun", ["t"]))

    def test_keep_calibration_after_a_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            p = {"quality": Path(d) / "quality", "calibration": Path(d) / "calibration"}
            p["quality"].mkdir()
            table = p["quality"] / "quality.json"
            table.write_text("{}")
            state.mark("prepare", {"binaries": "old"}, None, 1)
            state.mark("calibrate", {"t": 1}, "before", 5, [table])
            state.forget_from("prepare")
            with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                qc.keep_calibration(state, {"t": 2}, "after", p, lambda directory=None: None)
            self.assertIn("calibrate runs as usual", out.getvalue())
            self.assertEqual(state.decide("calibrate", {"t": 1}, "after")[0], "rerun")
            with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                qc.keep_calibration(state, {"t": 1}, "after", p, lambda directory=None: "other model")
            self.assertIn("other model", out.getvalue())
            with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                qc.keep_calibration(state, {"t": 1}, "after", p, lambda directory=None: None)
            self.assertIn("keeping the finished calibration", out.getvalue())
            self.assertEqual(state.decide("calibrate", {"t": 1}, "after"), ("skip", []))

    def test_keep_calibration_moves_a_replaced_calibration_back(self):
        # a run without --keep-calibration moved the finished calibration aside and started a new one
        with tempfile.TemporaryDirectory() as d:
            state = runstate.State(d)
            p = {"quality": Path(d) / "quality", "calibration": Path(d) / "calibration"}
            kept = Path(d) / "calibration.replaced"
            kept.mkdir()
            (kept / "quality.json").write_text('{"old": 1}')
            (kept / "record.json").write_text(json.dumps({"inputs": {"t": 1}, "finished": "x", "seconds": 7,
                                                          "outputs": [str(p["quality"] / "quality.json")]}))
            p["calibration"].mkdir()
            (p["calibration"] / "teacher.log").write_text("new")
            state.start("calibrate", {"t": 1}, "after")
            foreign = lambda directory=None: None
            with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                qc.keep_calibration(state, {"t": 2}, "after", p, foreign)
            self.assertIn("calibrate runs as usual", out.getvalue())
            self.assertFalse((p["calibration"] / "quality.json").exists())
            with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                qc.keep_calibration(state, {"t": 1}, "after", p, lambda directory=None: "other model" if directory else None)
            self.assertIn("calibrate runs as usual", out.getvalue())
            with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                qc.keep_calibration(state, {"t": 1}, "after", p, foreign)
            self.assertIn("moved the finished calibration", out.getvalue())
            self.assertEqual(json.loads((p["quality"] / "quality.json").read_text()), {"old": 1})
            self.assertFalse((p["calibration"] / "record.json").exists())
            self.assertEqual((kept / "teacher.log").read_text(), "new")
            self.assertEqual(state.decide("calibrate", {"t": 1}, "after"), ("skip", []))
            self.assertEqual(state.stages["calibrate"]["seconds"], 7)

    def test_calibration_ignores_timing_inputs(self):
        cfg = {"runtime": {"env": {}}}
        prepare = {"model": "m", "binaries": "b", "runtime": "r1", "timing": "t1", "inherited_environment": "e1"}
        base = qc.calibration_previous(prepare, cfg, "c")
        with unittest.mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/elsewhere"}):
            self.assertEqual(qc.calibration_previous({**prepare, "runtime": "r2", "timing": "t2", "inherited_environment": "e2"},
                                                     cfg, "c"), base)
        with unittest.mock.patch.dict(os.environ, {"GGML_CUDA_MOE_GROUPED": "packed"}):
            self.assertNotEqual(qc.calibration_previous(prepare, cfg, "c"), base)
        self.assertNotEqual(qc.calibration_previous({**prepare, "binaries": "b2"}, cfg, "c"), base)
        self.assertNotEqual(qc.calibration_previous(prepare, {"runtime": {"env": {"X": "1"}}}, "c"), base)
    def test_loss_only_prepare_measures_timing_when_a_later_run_needs_it(self):
        weighted = {"importance": {"precision": .5, "decode": .5, "short": 0, "long": 0}, "bounds": {}}
        lossy = {"importance": {"precision": 1, "decode": 0, "short": 0, "long": 0}, "bounds": {}}
        bounded = {**lossy, "bounds": {"min_decode_t_s": 20}}
        base = {"model": "m", "binaries": "b", "runtime": "r"}
        # a time importance or a speed bound needs the timing that the loss-only prepare skipped
        for timed in (weighted, bounded):
            with tempfile.TemporaryDirectory() as d:
                state = runstate.State(d)
                ran = []

                def go(cfg):
                    inputs = qc.prepare_stage_inputs(state, cfg, base)
                    with unittest.mock.patch("sys.stdout", io.StringIO()) as out:
                        qc.run_stage(state, "prepare", inputs, None, lambda: ran.append(inputs.get("loss_only", False)))
                    return inputs, out.getvalue()
                inputs, _ = go(lossy)
                self.assertEqual(ran, [True])
                self.assertEqual(qc.calibration_previous(inputs, {"runtime": {"env": {}}}, "c"),
                                 qc.calibration_previous(base, {"runtime": {"env": {}}}, "c"))
                _, out = go(lossy)
                self.assertIn("skipping", out)
                _, out = go(timed)
                self.assertIn("input changed (loss_only)", out)
                self.assertEqual(ran, [True, False])
                # the prepare with timing serves loss-only weights as it is
                _, out = go(lossy)
                self.assertIn("skipping", out)
                self.assertEqual(ran, [True, False])


class Interrupt(unittest.TestCase):
    @unittest.skipUnless(Path("/proc/self/stat").exists(), "needs /proc")
    def test_children_are_terminated(self):
        import subprocess
        child = subprocess.Popen(["sleep", "30"])
        qc.terminate_children()
        self.assertEqual(child.wait(10), -15)


class Convert(unittest.TestCase):
    def write_safetensors(self, path, tensors):
        header = json.dumps({n: {"dtype": "BF16", "shape": s, "data_offsets": [0, 0]} for n, s in tensors.items()}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header)

    def test_q8_estimate_counts_matrices_and_f32_norms(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_safetensors(Path(d) / "a.safetensors", {"w": [64, 64], "norm": [64], "odd": [10, 10]})
            self.assertEqual(cv.q8_bytes(d), 64 * 64 // 32 * 34 + 64 * 4 + 100 * 4)

    def test_staging_counts_engram_tables(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_safetensors(Path(d) / "a.safetensors", {"layers.2.engram.embed.weight": [1000, 256], "w": [64, 64]})
            self.assertEqual(cv.staging_bytes(d), 1000 * 256 // 32 * 34)

    def test_unsupported_projector_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            source, out = Path(d) / "src", Path(d) / "out"
            source.mkdir()
            (source / "config.json").write_text('{"vision_config": {}}')
            self.write_safetensors(source / "a.safetensors", {"w": [64, 64]})
            out.mkdir()
            (out / "model-q8_0.gguf").write_bytes(b"GGUF")

            def converter(python, src, outfile, extra, log, env, expected, text="ERROR:hf-to-gguf:Model X is not supported\n"):
                Path(log).write_text(text)
                raise cv.QuantError("conversion failed (1)")
            with unittest.mock.patch.object(cv, "run_converter", converter), unittest.mock.patch("builtins.print"):
                self.assertEqual(cv.convert(source, out), (out / "model-q8_0.gguf", None))
            self.assertFalse((out / "mmproj.partial").exists())
            with unittest.mock.patch.object(cv, "run_converter", lambda *a: converter(*a, text="Traceback\n")):
                with self.assertRaises(cv.QuantError):
                    cv.convert(source, out)

    def test_gguf_detection_and_vision(self):
        with tempfile.TemporaryDirectory() as d:
            gguf = Path(d) / "m.gguf"
            gguf.write_bytes(b"GGUF\0\0\0\3")
            self.assertTrue(cv.is_gguf(gguf))
            self.assertFalse(cv.is_gguf(d))
            self.assertFalse(cv.has_vision(d))
            (Path(d) / "config.json").write_text('{"vision_config": {}}')
            self.assertTrue(cv.has_vision(d))
            (Path(d) / "config.json").write_text('{"hidden_size": 4}')
            self.assertFalse(cv.has_vision(d))

    def test_converter_python_without_a_module_is_refused(self):
        with unittest.mock.patch.object(cv, "CONVERTER_MODULES", ("json",)):
            cv.check_converter(sys.executable)
        with unittest.mock.patch.object(cv, "CONVERTER_MODULES", ("json", "no_such_module_x")):
            with self.assertRaisesRegex(QuantError, "no_such_module_x"):
                cv.check_converter(sys.executable)
        with self.assertRaisesRegex(QuantError, "does not start"):
            cv.check_converter("/nonexistent/python")

    def test_projector_is_found_where_the_converter_writes_it(self):
        # like convert_hf_to_gguf.py, the fake converter writes exactly the --outfile name
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake = d / "python"
            fake.write_text('#!/bin/sh\n[ "$1" = "-c" ] && exit 0\n'
                            'while [ $# -gt 0 ]; do [ "$1" = "--outfile" ] && printf GGUF > "$2"; shift; done\n')
            fake.chmod(0o755)
            source = d / "ckpt"
            source.mkdir()
            self.write_safetensors(source / "a.safetensors", {"w": [64, 64]})
            (source / "config.json").write_text('{"vision_config": {}}')
            (d / "out").mkdir()
            (d / "out" / "model-q8_0.gguf").write_bytes(b"GGUF")
            with unittest.mock.patch.object(progress, "watch_file", lambda *a, **k: contextlib.nullcontext()):
                gguf, projector = cv.convert(source, d / "out", str(fake))
            self.assertEqual(gguf, d / "out" / "model-q8_0.gguf")
            self.assertTrue(projector.name.startswith("mmproj-") and projector.is_file())
            self.assertFalse((d / "out" / "mmproj.partial").exists())

    def test_missing_safetensors_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(QuantError):
                cv.source_identity(d)


class RunConfig(unittest.TestCase):
    def args(self, **kw):
        values = dict(imatrix=[Path("/x/im.gguf")], bindir=Path("/bin"), calibration_text=["/c/text.txt"], ub=512, ctk=None, ctv="q8_0")
        return argparse.Namespace(**{**values, **kw})

    def measured(self):
        return {"per_device": {"ROCm0": {"total": 3 * GIB}, "ROCm1": {"total": 2 * GIB}}}

    def rows(self):
        return [{"device": "ROCm0", "capacity": 20 * GIB}, {"device": "ROCm1", "capacity": 10 * GIB}]

    def test_reserves_are_measured_plus_the_users(self):
        cfg = qc.build_config(self.args(), "/m.gguf", self.rows(), {"ROCm1": GIB}, self.measured(), {})
        self.assertEqual(cfg["reserve"], {"ROCm0": 3 * GIB, "ROCm1": 3 * GIB})
        self.assertEqual(cfg["devices"], {"ROCm0": 20 * GIB, "ROCm1": 10 * GIB})
        self.assertEqual(cfg["runtime"]["extra_args"], ["-ctv", "q8_0"])
        self.assertEqual(cfg["calibration"]["text"], "/c/text.txt")
        self.assertTrue(cfg["allow_requantize"])

    def test_override_wins_for_devices_and_merges_runtime(self):
        override = {"devices": {"ROCm1": "4GiB"}, "primary_qtypes": ["Q2_K"], "runtime": {"threads": 8, "extra_args": ["-fa", "on"]}}
        cfg = qc.build_config(self.args(), "/m.gguf", self.rows(), {}, self.measured(), override)
        self.assertEqual(cfg["devices"]["ROCm1"], 4 << 30)
        self.assertEqual(cfg["primary_qtypes"], ["Q2_K"])
        self.assertEqual(cfg["runtime"]["threads"], 8)
        self.assertEqual(cfg["runtime"]["extra_args"], ["-ctv", "q8_0", "-fa", "on"])

    def test_runtime_option_wins_over_the_file_and_reaches_the_config(self):
        cfg = qc.build_config(self.args(runtime="llama.cpp"), "/m.gguf", self.rows(), {}, self.measured(), {"target_runtime": "ik_llama"})
        self.assertEqual(cfg["target_runtime"], "llama")
        cfg = qc.build_config(self.args(), "/m.gguf", self.rows(), {}, self.measured(), {"target_runtime": "ik_llama"})
        self.assertEqual(cfg["target_runtime"], "ik_llama")
        self.assertNotIn("target_runtime", qc.build_config(self.args(), "/m.gguf", self.rows(), {}, self.measured(), {}))

    def test_a_runtime_switch_keeps_the_prepare_and_calibration_inputs(self):
        prepared = {"model": "m", "imatrix": [], "sampling": {}, "groups": [], "binaries": {}, "libraries": [], "gpu": "g",
                    "ram_routed": True, "runtime": {"env": {}}}
        fingerprints = {qc.calibration_previous(prepared, {"runtime": {"env": {}}, "target_runtime": runtime}, "converted")
                        for runtime in ("delphi", "llama", "ik_llama")}
        self.assertEqual(len(fingerprints), 1)

    def test_bad_overrides_are_refused(self):
        for override in ({"devices": {"ROCm5": "1GiB"}}, {"model": "/other.gguf"}):
            with self.assertRaises(QuantError):
                qc.build_config(self.args(), "/m.gguf", self.rows(), {}, self.measured(), override)
        with self.assertRaises(QuantError):
            qc.build_config(self.args(), "/m.gguf", self.rows(), {"ROCm7": 1}, self.measured(), {})

    def test_rounding_hides_small_drifts(self):
        self.assertEqual(qc.rounded({"a": 100 * 2**20 + 5}), qc.rounded({"a": 100 * 2**20 + 60}))
        self.assertNotEqual(qc.rounded({"a": 100 * 2**20}), qc.rounded({"a": 200 * 2**20}))

    def test_relative_paths_of_a_config_follow_the_file(self):
        raw = qc.absolutize_override({"bindir": "bin", "calibration": {"text": "t.txt"}}, "/cfg")
        self.assertEqual(raw["bindir"], "/cfg/bin")
        self.assertEqual(raw["calibration"]["text"], "/cfg/t.txt")


if __name__ == "__main__":
    unittest.main()
