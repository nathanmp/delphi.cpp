#!/usr/bin/env python3
"""Auto-tuner invariants with fake measurements; no server or GPU is needed."""
import json
import sys
import tempfile
import unittest
import os
import subprocess
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/auto-tune"))
import knobs as K
import score as S
from auto_tune import Budget, Tuner, plan_text, base_launch
from config import DEFAULTS, _merge
from report import write
from store import Store
from cpu import CpuBudget, parse_list
import placement as PL
from measure import vram_violations, P, CpuSampler
from verify_client import verify

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


class CpuPolicy(unittest.TestCase):
    facts = {i: (i // 4, i % 4, i // 4) for i in range(8)} | {8: (0, 0, 0)}

    def test_exact_list_and_smt(self):
        budget = CpuBudget.resolve('0-3,8', facts=self.facts)
        self.assertEqual(budget.cpus, (0, 1, 2, 3, 8))
        self.assertEqual(budget.cores, 4)
        self.assertEqual(budget.nodes, (0,))
        with self.assertRaises(RuntimeError):
            CpuBudget.resolve('0-9', facts=self.facts)
        for value in ('3-1', '-1', '1,,2', '1-2-3'):
            with self.assertRaises(RuntimeError):
                parse_list(value)

    def test_core_limit_is_balanced_and_uses_no_smt_sibling(self):
        budget = CpuBudget.resolve(max_cores=4, facts=self.facts)
        self.assertEqual(budget.cpus, (0, 1, 4, 5))
        self.assertEqual(budget.threads(1), 2)
        self.assertEqual(len(CpuBudget.resolve(max_cores=100, facts=self.facts).cpus), 8)
        with self.assertRaises(RuntimeError):
            CpuBudget.resolve('0', 2, self.facts)

    def test_command_replaces_all_masks_and_thread_counts(self):
        budget = CpuBudget.resolve('0-1,4-5', facts=self.facts)
        command = (*BASE, '--threads', '64', '--cpu-range', '0-63', '--cpu-mask-batch', 'ffff',
                   '--spec-draft-cpu-range', '0-63', '--spec-draft-threads', '64')
        command = budget.apply(command, 1)
        self.assertEqual(K.option_value(command, '-t'), '2')
        self.assertEqual(K.option_value(command, '-tb'), '2')
        self.assertEqual(K.option_value(command, '--spec-draft-threads'), '2')
        self.assertEqual(int(K.option_value(command, '--cpu-mask'), 16), 0x33)
        self.assertNotIn('--cpu-range', command)
        self.assertNotIn('--threads', command)
        one_node = CpuBudget.resolve('0-3', facts=self.facts)
        self.assertIsNone(K.option_value(one_node.apply(BASE, 1), '--numa'))
        self.assertEqual(K.option_value(command, '--numa'), 'tensors')
        self.assertEqual(K.option_value(budget.apply((*BASE, '--numa', 'numactl'), 1), '--numa'), 'numactl')

    def test_affinity_is_inherited_and_restored(self):
        original = os.sched_getaffinity(0)
        cpus = tuple(sorted(original)[:2])
        budget = CpuBudget(cpus, len(cpus), (0,))
        with budget.affinity():
            result = subprocess.check_output([sys.executable, '-c', 'import os; print(sorted(os.sched_getaffinity(0)))'], text=True)
            self.assertEqual(json.loads(result), list(cpus))
        self.assertEqual(os.sched_getaffinity(0), original)

    def test_audit_observes_child_threads(self):
        cpus = tuple(sorted(os.sched_getaffinity(0))[:2])
        budget = CpuBudget(cpus, len(cpus), (0,))
        audit = CpuSampler((sys.executable,), budget)
        with budget.affinity():
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
        try:
            audit.sample()
            self.assertGreater(audit.samples, 0)
            self.assertFalse(audit.violations)
        finally:
            child.terminate()
            child.wait()

    def test_context_floor_uses_server_props(self):
        self.assertEqual(verify({'default_generation_settings': {'n_ctx': 8192}}, 8192)['verified_context'], 8192)
        for props in ({}, {'default_generation_settings': {'n_ctx': 4096}}):
            with self.assertRaises(ValueError):
                verify(props, 8192)


class PlacementPolicy(unittest.TestCase):
    def rows(self, model, context=50, compute=50):
        return {'ROCm0': {'model': model, 'context': context, 'compute': compute},
                'Host': {'model': 10000, 'context': 0, 'compute': 0}}

    def options(self, full=500, dense=100, experts=8, free=1000):
        info = {'experts': experts, 'expert_layers': list(range(20))}
        return PL.candidates(BASE, {}, [PL.Device('ROCm0', free, 100)], self.rows(full), self.rows(dense), info)

    def test_fits(self):
        option = self.options()[0]
        self.assertEqual(option['name'], 'fits')
        self.assertEqual(K.option_value(option['command'], '-ngl'), '999')
        self.assertNotIn('-cmoe', option['command'])
        self.assertEqual(option['env']['LLAMA_MOE_CACHE_MIB'], '0')

    def test_dense_fits_moe(self):
        option = self.options(full=2000)[0]
        self.assertEqual(option['name'], 'host-experts')
        self.assertIn('-cmoe', option['command'])
        self.assertEqual(option['env']['LLAMA_MOE_CACHE_MIB'], 'auto')
        self.assertEqual(option['env']['GGML_SCHED_STAGE_MIN_TOKENS'], '1024')

    def test_dense_fits_dense_and_poor_fit(self):
        option = self.options(full=2000, dense=2000, experts=0)[0]
        self.assertEqual(option['name'], 'dense-partial')
        self.assertIsNone(K.option_value(option['command'], '-ngl'))
        option = self.options(full=2000, dense=1000)[0]
        self.assertEqual(option['name'], 'poor-fit')
        self.assertTrue(option['warnings'])

    def test_small_cpu_expert_fraction(self):
        options = self.options(full=900, dense=100)
        self.assertEqual([o['name'] for o in options], ['host-experts', 'cpu-expert-layers'])
        self.assertNotIn('-cmoe', options[1]['command'])
        self.assertEqual(K.option_value(options[1]['command'], '--n-cpu-moe'), '3')
        self.assertEqual(len(self.options(full=2000)), 1)

    def test_fill_rules_and_per_device_buffers(self):
        devices = [PL.Device('ROCm2', 1000, 100, 'min'), PL.Device('ROCm0', 1000, 100), PL.Device('ROCm3', 1000, 100)]
        full = {d.name: {'model': 600, 'context': 50, 'compute': 50} for d in devices}
        full['Host'] = {'model': 10000, 'context': 0, 'compute': 0}
        options = PL.candidates(BASE, {}, devices, full, full, {'experts': 0}, {'ROCm3': 100})
        split = [float(n) for n in K.option_value(options[0]['command'], '-ts').split(',')]
        self.assertEqual(split, [300, 800, 700])
        self.assertEqual(K.option_value(options[0]['command'], '--device'), 'ROCm2,ROCm0,ROCm3')

    def test_display_margin_caps_and_free_memory(self):
        devices = PL.devices_from_state(('--device', 'ROCm0,ROCm2'), {'ROCm2': {'fill': 'min'}, 'ROCm0': {'max_used_mib': 4000}},
                                       {'ROCm0': 'a', 'ROCm2': 'b'}, {'a': (2000, 10000), 'b': (3000, 10000)})
        self.assertEqual([d.name for d in devices], ['ROCm2', 'ROCm0'])
        self.assertEqual(devices[0].margin_mib, 2048)
        self.assertEqual(devices[1].capacity, 4000)
        self.assertTrue(vram_violations({'b': {'min_free_mib': 2000}}, {'ROCm2': {'min_free_mib': 2048}}, {'ROCm2': 'b'}))

    def test_fit_parser_and_server_only_options(self):
        rows = PL.parse_fit('ROCm0 123 45 67 \nHost 890 0 4\n')
        self.assertEqual(rows['ROCm0']['compute'], 67)
        with self.assertRaises(RuntimeError):
            PL.parse_fit('')
        cmd = PL.fit_command((*BASE, '--spec-draft-model', 'draft.gguf', '--parallel', '1'))
        self.assertNotIn('--spec-type', cmd)
        self.assertNotIn('--spec-draft-model', cmd)
        self.assertIn('--fit-print', cmd)

    def test_loop_one_uses_reload_table(self):
        with tempfile.TemporaryDirectory() as d:
            c = cfg(d, knobs={'only': ['-ub', '-t', '--spec-draft-n-max', '--no-repack'], 'include': [], 'exclude': [], 'levels': {}})
            t = Tuner(c, BASE, {}, {}, lambda *a: {}, None, launch_only=True)
            self.assertEqual([k.name for k, _ in t.selected()], ['--no-repack'])
            self.assertEqual(K.BY_NAME['-ub'].reload, 'context')
            self.assertEqual(K.BY_NAME['--spec-draft-n-max'].reload, 'request')
        command = (*BASE, '--numa', 'tensors', '-cmoe')
        self.assertEqual(K.BY_NAME['GGML_CPU_NUMA_CHUNKS'].reload, 'process')
        self.assertIn('GGML_CPU_NUMA_CHUNKS', dict((k.name, v) for k, v in K.select(ctx(command))))

    def test_requested_devices_reorder_reference_ratios(self):
        with tempfile.TemporaryDirectory() as d:
            arm = P.ServerArm('a', (*BASE, '-ts', '2,1'), ('client',))
            recipe = P.Recipe('test', '', (arm,))
            with patch('measure.P.load_recipe', return_value=recipe):
                c = cfg(d, devices={'ROCm1': {}, 'ROCm0': {}})
                _, command, _ = base_launch(c)
                self.assertEqual(K.option_value(command, '--device'), 'ROCm1,ROCm0')
                self.assertEqual(K.option_value(command, '-ts'), '1,2')
                c['devices'] = {'ROCm1': {}}
                _, command, _ = base_launch(c)
                self.assertEqual(K.option_value(command, '--device'), 'ROCm1')
                self.assertIsNone(K.option_value(command, '-ts'))

    def test_coupled_draft_reserves_weights_and_buffers_without_fit(self):
        with tempfile.TemporaryDirectory() as d:
            c = cfg(d, devices={'ROCm0': {'min_free_mib': 100}})
            command = K.set_option(K.set_option(BASE, '--spec-type', 'draft-dspark'), '--spec-draft-model', 'draft.gguf')
            command = K.set_option(command, '--spec-draft-device', 'ROCm0')
            facts = {'architecture': 'deepseek41', 'experts': 8, 'expert_layers': list(range(20)), 'layers': 20, 'nextn': 0,
                     'mtp_mib': 0, 'model_mib': 2100, 'files': [], 'max_layer_mib': 100}
            labels = []
            def probe(command, label):
                labels.append(label)
                return self.rows(2000 if label == 'full' else 100)
            with patch('placement.model_info', side_effect=[facts, {**facts, 'architecture': 'dflash', 'model_mib': 500}]):
                plan = PL.prepare(c, command, {}, {'ROCm0': 'a'}, {'a': (0, 1000)}, None, probe)
            self.assertEqual(labels, ['full', 'dense'])
            self.assertEqual(plan['extra_mib']['ROCm0'], 600)
            self.assertEqual(plan['candidates'][0]['name'], 'host-experts')
            self.assertIsNotNone(plan['draft_estimate'])

    def test_mtp_reservation_follows_last_active_device(self):
        with tempfile.TemporaryDirectory() as d:
            c = cfg(d, devices={'ROCm0': {'min_free_mib': 100}, 'ROCm1': {'fill': 'min', 'min_free_mib': 100}})
            facts = {'architecture': 'qwen4exp', 'experts': 0, 'expert_layers': [], 'layers': 10, 'nextn': 1,
                     'mtp_mib': 50, 'model_mib': 450, 'files': []}
            full = {**self.rows(200), 'ROCm1': self.rows(200)['ROCm0']}
            def probe(command, label):
                return full if label == 'full' else self.rows(400)
            with patch('placement.model_info', return_value=facts):
                plan = PL.prepare(c, BASE, {}, {'ROCm0': 'a', 'ROCm1': 'b'}, {'a': (0, 1000), 'b': (0, 1000)}, None, probe)
            self.assertEqual(K.option_value(plan['candidates'][0]['command'], '--device'), 'ROCm0')
            self.assertGreater(plan['extra_mib']['ROCm0'], 0)
            self.assertEqual(plan['extra_mib']['ROCm1'], 0)

    def test_split_rounding_is_checked_before_a_load(self):
        with tempfile.TemporaryDirectory() as d:
            c = cfg(d, devices={'ROCm0': {'min_free_mib': 100}, 'ROCm1': {'fill': 'min', 'min_free_mib': 100}})
            command = K.remove_option(BASE, '--spec-type')
            facts = {'architecture': 'dense', 'experts': 0, 'expert_layers': [], 'layers': 10, 'nextn': 0,
                     'mtp_mib': 0, 'model_mib': 1300, 'files': []}
            labels = []
            def probe(command, label):
                labels.append(label)
                if label == 'full':
                    return {**self.rows(650), 'ROCm1': self.rows(650)['ROCm0']}
                if label == 'split-0':
                    return {**self.rows(950), 'ROCm1': self.rows(350)['ROCm0']}
                return {**self.rows(649), 'ROCm1': self.rows(651)['ROCm0']}
            with patch('placement.model_info', return_value=facts):
                plan = PL.prepare(c, command, {}, {'ROCm0': 'a', 'ROCm1': 'b'}, {'a': (0, 1000), 'b': (0, 1000)}, None, probe)
            self.assertEqual(labels, ['full', 'split-0', 'split-1'])
            self.assertEqual(plan['split_reserve_mib']['ROCm0'], 151)
            self.assertIn('projected', plan['candidates'][0])

    def test_reference_footprint_calibrates_unmeasured_runtime_memory(self):
        with tempfile.TemporaryDirectory() as d:
            c = cfg(d, devices={'ROCm0': {'min_free_mib': 100}})
            command = K.remove_option(BASE, '--spec-type')
            facts = {'architecture': 'dense', 'experts': 0, 'expert_layers': [], 'layers': 10, 'nextn': 0,
                     'mtp_mib': 0, 'model_mib': 500, 'files': []}
            observed = {'command': command, 'metrics': {'vram': {'a': {'used_mib': 700}}}}
            with patch('placement.model_info', return_value=facts):
                plan = PL.prepare(c, command, {}, {'ROCm0': 'a'}, {'a': (0, 1000)}, None,
                                  lambda command, label: self.rows(500), observed)
            self.assertEqual(plan['runtime_overhead_mib']['ROCm0'], 164)
            self.assertEqual(plan['extra_mib']['ROCm0'], 164)

    def test_failed_reference_can_use_verified_policy(self):
        with tempfile.TemporaryDirectory() as d:
            c = cfg(d, knobs={'only': ['--no-repack'], 'include': [], 'exclude': ['--no-repack'], 'levels': {}})
            def launch(command, env, name):
                if K.option_value(command, '-ts') is None:
                    raise RuntimeError('out of memory')
                return fake_metrics(command, env)
            candidate = {'name': 'fits', 'command': K.set_option(BASE, '-ts', '1,1'), 'env': {}, 'warnings': []}
            t = Tuner(c, BASE, {}, {}, launch, None, launch_only=True,
                      placement_factory=lambda: {'dry_runs': [], 'candidates': [candidate]}, log=lambda s: None)
            state = t.run()
            self.assertIn('reference_failure', state)
            self.assertEqual(t.store.find(state['base'], 'launch')['status'], 'ok')
            write(c, state, t.store, Path(d))
            self.assertIn('reference was infeasible', (Path(d) / 'report.md').read_text())


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
