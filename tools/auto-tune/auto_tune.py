#!/usr/bin/env python3
"""Launch-parameter auto-tuner: plan, run and report. See README.md."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knobs as K  # noqa: E402
import score as S  # noqa: E402
from config import TuneError, digest, file_identity, load_config  # noqa: E402
from store import Store  # noqa: E402

CONFIRM_LAUNCHES = 3


class BudgetExhausted(Exception):
    pass


def physical_cores() -> int:
    cores = set()
    for topo in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology"):
        try:
            cores.add(((topo / "physical_package_id").read_text().strip(), (topo / "core_id").read_text().strip()))
        except OSError:
            continue
    return len(cores) or (os.cpu_count() or 1)


class Budget:
    def __init__(self, max_launches: int, max_hours: float, clock: Callable[[], float] = time.time) -> None:
        self.max_launches = max_launches
        self.max_s = max_hours * 3600.0
        self.clock = clock
        self.started = clock()
        self.used = 0
        self.durations: list[float] = []

    def estimate_s(self) -> float:
        return statistics.median(self.durations) if self.durations else 0.0

    def left(self) -> tuple[int, float]:
        return self.max_launches - self.used, self.max_s - (self.clock() - self.started)

    def allow(self, reserve: int = 0) -> bool:
        launches, seconds = self.left()
        return launches - reserve >= 1 and seconds - (reserve + 1) * self.estimate_s() >= 0

    def spend(self, seconds: float) -> None:
        self.used += 1
        self.durations.append(seconds)


class Tuner:
    """Coordinate search from the recipe arm over the registry knobs, then an interleaved confirm against the arm.

    *launcher(command, env, name) -> metrics* and *kld_runner(command, env, name, compare) -> float | None* are the
    measurements; tests replace them."""

    def __init__(self, cfg: Mapping[str, Any], base_command: Sequence[str], base_env: Mapping[str, str], identity: Mapping[str, Any],
                 launcher: Callable[..., dict[str, Any]], kld_runner: Callable[..., float | None] | None,
                 vram_check: Callable[[Mapping[str, Any], Sequence[str]], list[str]] = lambda metrics, command: [],
                 budget: Budget | None = None, log: Callable[[str], None] = print, *, launch_only: bool = False,
                 placement_factory=None, cpu_budget=None) -> None:
        self.cfg = cfg
        self.base = (tuple(base_command), dict(base_env))
        self.identity = dict(identity)
        self.launcher = launcher
        self.kld_runner = kld_runner
        self.vram_check = vram_check
        self.store = Store(Path(cfg["output"]))
        self.budget = budget or Budget(cfg["budget"]["max_launches"], cfg["budget"]["max_hours"])
        self.log = log
        self.launch_only = launch_only
        self.placement_factory = placement_factory
        self.cpu_budget = cpu_budget
        self.weights = S.metric_weights(cfg["weights"], cfg["prefill"]["lengths"])
        self.measure_spec = {"prefill": cfg["prefill"] if cfg["weights"].get("prefill", 0) > 0 else None,
                             "osl": cfg["decode"]["osl"] if cfg["weights"].get("decode", 0) > 0 else 16,
                             "trace": cfg["trace"], "env": cfg["env"]}
        self.kld_spec = {k: v for k, v in cfg["kld"].items() if k != "limit"}
        self.max_threads = cpu_budget.threads(len((K.option_value(base_command, '--device') or '').split(',')), cfg['max_threads']) if cpu_budget else cfg["max_threads"] or max(1, physical_cores() - 8)
        self.state: dict[str, Any] = {"steps": [], "untested": [], "confirm": None}
        self.logits_ready: set[str] = set()  # reference logits written by this process (they live in /dev/shm for one run)

    # helpers

    def context(self, command: Sequence[str], env: Mapping[str, str]) -> K.Context:
        return K.Context(tuple(command), dict(env), physical_cores(), self.max_threads)

    def key(self, command: Sequence[str], env: Mapping[str, str]) -> str:
        return digest({"identity": self.identity, "command": list(command), "env": dict(env), "measure": self.measure_spec})

    def changes(self, command: Sequence[str], env: Mapping[str, str]) -> dict[str, list[str | None]]:
        base = self.context(*self.base)
        ctx = self.context(command, env)
        out = {}
        for knob in K.REGISTRY:
            a, b = knob.current(base.command, base.env), knob.current(ctx.command, ctx.env)
            if a != b:
                out[knob.name] = [a, b]
        for name in ('--device', '-ts', '-ngl', '--n-cpu-moe', '--numa'):
            a, b = K.option_value(base.command, name), K.option_value(ctx.command, name)
            if a != b:
                out[name] = [a, b]
        for name in ('-cmoe',):
            if (name in base.command) != (name in ctx.command):
                out[name] = [K.ON if name in base.command else K.OFF, K.ON if name in ctx.command else K.OFF]
        return out

    def selected(self) -> list[tuple[K.Knob, list[str]]]:
        k = self.cfg["knobs"]
        chosen = K.select(self.context(*self.base), only=k["only"], include=k["include"], exclude=k["exclude"], levels=k["levels"])
        if self.cfg["kld"]["limit"] is None:
            chosen = [(knob, levels) for knob, levels in chosen if knob.cls != "precision"]
        if self.launch_only:
            chosen = [(knob, levels) for knob, levels in chosen if knob.reload == 'process']
        order = {name: i for i, name in enumerate(self.cfg.get("class_order") or ("fit", "precision", "accept", "speed"))}
        return sorted(chosen, key=lambda item: order[item[0].cls])

    def _spend(self, started: float) -> None:
        self.budget.spend(time.time() - started)

    # measurements

    def reference_logits(self, reserve: int = 0) -> str:
        """Name of the exact-precision reference logits; written on first use in this process."""
        ctx = self.context(*self.base)
        command, env = self.base
        for knob, level in K.exact_settings(ctx):
            command, env = knob.apply(command, env, level)
        key = digest({"identity": self.identity, "logits": K.precision_key(self.context(command, env)), "kld": self.kld_spec})
        if key not in self.logits_ready:
            if not self.budget.allow(reserve):
                raise BudgetExhausted()
            self.log("kld: reference logits (exact precision levels)")
            started = time.time()
            self.store.start(key, "logits")
            try:
                self.kld_runner(command, env, key, False)
            finally:
                self.store.end(key)
            self._spend(started)
            self.logits_ready.add(key)
        return key

    def kld(self, command: Sequence[str], env: Mapping[str, str], reserve: int = 0) -> float | None:
        ctx = self.context(command, env)
        if self.cfg["kld"]["limit"] is None or K.is_exact(ctx):
            return None if self.cfg["kld"]["limit"] is None else 0.0
        logits = self.reference_logits(reserve)
        # only the precision knobs change the logits that are compared; the other knobs stay at the base
        base_cmd, base_env = self.base
        for knob, _ in K.exact_settings(ctx):
            base_cmd, base_env = knob.apply(base_cmd, base_env, knob.current(ctx.command, ctx.env))
        key = digest({"identity": self.identity, "precision": K.precision_key(ctx), "kld": self.kld_spec})
        found = self.store.find(key, "kld")
        if found:
            return found.get("kld")
        if not self.budget.allow(reserve):
            raise BudgetExhausted()
        self.log(f"kld: {K.precision_key(ctx)}")
        started = time.time()
        self.store.start(key, "kld")
        try:
            value = self.kld_runner(base_cmd, base_env, logits, True)
        except Exception as error:  # noqa: BLE001 - a failed KLD run is a result
            self.store.end(key)
            self._spend(started)
            self.store.add({"key": key, "kind": "kld", "kld": None, "error": str(error)})
            return None
        self.store.end(key)
        self._spend(started)
        self.store.add({"key": key, "kind": "kld", "kld": value})
        return value

    def evaluate(self, command: Sequence[str], env: Mapping[str, str], name: str, *, phase: str = "search",
                 cached: bool = True, reserve: int = 0) -> dict[str, Any]:
        key = self.key(command, env)
        if cached:
            found = self.store.find(key, "launch", "search")
            if found:
                self.log(f"{name}: cached ({found['status']})")
                return found
        record: dict[str, Any] = {"key": key, "kind": "launch", "phase": phase, "name": name, "command": list(command), "env": dict(env),
                                  "changes": self.changes(command, env)}
        if key in self.store.crashed:
            return self.store.add({**record, "status": "crashed", "error": "an earlier launch of this config did not end (journal)"})
        kld = None
        if self.cfg["kld"]["limit"] is not None:
            kld = self.kld(command, env, reserve)
            record["kld"] = kld
            if kld is None and not K.is_exact(self.context(command, env)):
                return self.store.add({**record, "status": "failed", "error": "KLD run failed"})
            if kld is not None and kld > self.cfg["kld"]["limit"]:
                self.log(f"{name}: KLD {kld:.4f} over the limit")
                return self.store.add({**record, "status": "kld-over"})
        if not self.budget.allow(reserve):
            raise BudgetExhausted()
        self.log(f"{name}: launch {self.budget.used + 1}/{self.budget.max_launches}")
        started = time.time()
        self.store.start(key, name)
        try:
            metrics = self.launcher(command, env, name)
        except KeyboardInterrupt:
            self.store.end(key)
            raise
        except Exception as error:  # noqa: BLE001 - a failed launch is a measured boundary
            self.store.end(key)
            self._spend(started)
            self.log(f"{name}: failed: {error}")
            return self.store.add({**record, "status": "failed", "error": str(error)})
        self.store.end(key)
        self._spend(started)
        problems = self.vram_check(metrics, command)
        status = "vram-over" if problems else "ok"
        return self.store.add({**record, "status": status, "metrics": metrics, "error": "; ".join(problems) or None})

    # search

    def run(self) -> dict[str, Any]:
        command, env = self.base
        placement_options = []
        if self.placement_factory:
            self.log('placement: dry runs at the context floor')
            started = time.time()
            self.state['placement'] = self.placement_factory()
            self.state['stages'] = {'placement': {'dry_runs': len(self.state['placement']['dry_runs']), 'wall_s': time.time() - started}}
            placement_options = self.state['placement']['candidates']
        if self.cpu_budget:
            self.state['cpu_budget'] = self.cpu_budget.summary()
        if self.cfg["warmup_launch"]:
            try:
                self.evaluate(command, env, "warmup", phase="warmup", cached=False)
            except BudgetExhausted:
                raise TuneError("the budget does not cover the warm-up launch") from None
        try:
            base_row = self.evaluate(command, env, "base")
        except BudgetExhausted:
            raise TuneError("the budget does not cover the base launch") from None
        if base_row["status"] != "ok":
            self.state['reference_failure'] = base_row.get('error') or base_row['status']
            for option in placement_options:
                try:
                    row = self.evaluate(option['command'], option['env'], 'placement ' + option['name'])
                except BudgetExhausted:
                    break
                if row['status'] == 'ok':
                    command, env, base_row = tuple(option['command']), option['env'], row
                    self.base = (command, env)
                    break
            if base_row['status'] != 'ok':
                raise TuneError(f"no placement loaded within the constraints: {self.state['reference_failure']}")
        reference = base_row["metrics"]
        self.state["base"] = base_row["key"]
        incumbent = {"command": command, "env": env, "row": base_row, "score": 1.0}
        for option in placement_options:
            step = {'knob': 'placement', 'candidate': option['name']}
            try:
                row = self.evaluate(option['command'], option['env'], 'placement ' + option['name'], reserve=CONFIRM_LAUNCHES)
            except BudgetExhausted:
                self.state['untested'].append('placement ' + option['name'])
                continue
            value = S.score(row['metrics'], reference, self.weights) if row['status'] == 'ok' else None
            step.update(key=row['key'], status=row['status'], score=value)
            if value is not None and value > incumbent['score'] * (1 + self.cfg['noise_margin']):
                incumbent = {'command': tuple(option['command']), 'env': option['env'], 'row': row, 'score': value}
                step['kept'] = option['name']
            self.state['steps'].append(step)
        selected = self.selected()
        index = 0
        for position, (knob, levels) in enumerate(selected):
            ctx = self.context(incumbent["command"], incumbent["env"])
            if not knob.applies(ctx):
                self.state["steps"].append({"knob": knob.name, "skipped": "does not apply to the incumbent"})
                continue
            current = knob.current(ctx.command, ctx.env)
            step: dict[str, Any] = {"knob": knob.name, "class": knob.cls, "from": current, "tested": []}
            best = None
            try:
                for level in levels:
                    if level == current:
                        continue
                    index += 1
                    cmd2, env2 = knob.apply(incumbent["command"], incumbent["env"], level)
                    row = self.evaluate(cmd2, env2, f"{index:02d} {knob.name}={level}", reserve=CONFIRM_LAUNCHES)
                    s = S.score(row["metrics"], reference, self.weights) if row["status"] == "ok" else None
                    step["tested"].append({"level": level, "key": row["key"], "status": row["status"], "score": s})
                    if s is not None and (best is None or s > best[0]):
                        best = (s, level, cmd2, env2, row)
            except BudgetExhausted:
                self.state["steps"].append(step)
                self.state["untested"] = [k.name for k, _ in selected[position:]]
                self.log(f"budget: stopped at {knob.name}")
                break
            if best and best[0] > incumbent["score"] * (1.0 + self.cfg["noise_margin"]):
                step["kept"] = best[1]
                incumbent = {"command": best[2], "env": best[3], "row": best[4], "score": best[0]}
                self.log(f"keep {knob.name}={best[1]} (score {best[0]:.3f})")
            self.state["steps"].append(step)
        self.state["incumbent"] = incumbent["row"]["key"]
        self.state["confirm"] = self.confirm(incumbent)
        self.state["budget"] = {"launches": self.budget.used, "max_launches": self.budget.max_launches,
                                "hours": (time.time() - self.budget.started) / 3600.0, "max_hours": self.budget.max_s / 3600.0}
        self.state.setdefault('stages', {})['launch'] = {'launches': self.budget.used, 'recreates': 0,
                                                       'wall_s': sum(self.budget.durations)}
        return self.state

    def confirm(self, incumbent: Mapping[str, Any]) -> dict[str, Any] | None:
        """best, base, best interleaved; the gain is the score of the mean best against the mean base."""
        if incumbent["row"]["key"] == self.state["base"]:
            return None
        command, env = self.base
        order = [("best", incumbent["command"], incumbent["env"]), ("base", command, env), ("best", incumbent["command"], incumbent["env"])]
        runs: dict[str, list[dict[str, Any]]] = {"best": [], "base": []}
        for i, (role, cmd, e) in enumerate(order):
            try:
                row = self.evaluate(cmd, e, f"confirm-{i + 1} {role}", phase="confirm", cached=False)
            except BudgetExhausted:
                break
            if row["status"] == "ok":
                runs[role].append(row["metrics"])
        if not runs["best"] or not runs["base"]:
            return {"complete": False}
        best, base = mean_metrics(runs["best"]), mean_metrics(runs["base"])
        gain = S.score(best, base, self.weights)
        return {"complete": len(runs["best"]) == 2, "gain": gain, "best": best, "base": base,
                "kept": len(runs['best']) == 2 and gain is not None and gain > 1.0 + self.cfg["noise_margin"]}


def mean_metrics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Geometric mean of the scored metrics of several runs of one config."""
    def gmean(values: Sequence[float]) -> float | None:
        values = [v for v in values if v]
        return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None
    out: dict[str, Any] = {"decode_t_s": gmean([m.get("decode_t_s") for m in items]), "prefill": {}}
    for length in items[0].get("prefill", {}):
        out["prefill"][length] = {key: gmean([m["prefill"].get(length, {}).get(key) for m in items]) for key in ("ttft_s", "t_s")}
    return out


# CLI

def base_launch(cfg: Mapping[str, Any]):
    from measure import P
    from knobs import set_option
    recipe = P.apply_bindir(P.load_recipe(cfg["recipe"]), Path(cfg["bindir"]) if cfg["bindir"] else None)
    arm = next((a for a in recipe.arms if a.name == cfg["arm"]), None)
    if arm is None:
        raise TuneError(f"recipe {recipe.name} has no arm {cfg['arm']}")
    if not isinstance(arm, P.ServerArm):
        raise TuneError("the auto-tuner needs a server arm")
    command = set_option(arm.server_command, "-c", str(cfg["context"]))
    # Canonicalize aliases used by the placement and CPU policy before replacing their values.
    aliases = {'--ctx-size': '-c', '--threads': '-t', '--threads-batch': '-tb', '--tensor-split': '-ts',
               '--n-gpu-layers': '-ngl', '-dev': '--device', '--cpu-moe': '-cmoe', '-ncmoe': '--n-cpu-moe',
               '--override-tensor': '-ot', '--load-mode': '-lm', '-nr': '--no-repack', '-fit': '--fit', '-fitt': '--fit-target'}
    for old, new in aliases.items():
        if old in ('--cpu-moe', '-nr'):
            command = tuple(new if item == old else item for item in command)
        elif K.option_value(command, old) is not None:
            value = K.option_value(command, old)
            command = K.set_option(K.remove_option(command, old), new, value)
    command = K.set_option(command, '-c', str(cfg['context']))
    if cfg['devices']:
        previous_devices = (K.option_value(command, '--device') or '').split(',')
        requested = list(cfg['devices'])
        split = (K.option_value(command, '-ts') or '').split(',')
        if previous_devices != requested:
            command = K.set_option(command, '--device', ','.join(requested))
            if set(previous_devices) == set(requested) and len(split) == len(previous_devices):
                ratios = dict(zip(previous_devices, split))
                command = K.set_option(command, '-ts', ','.join(ratios[name] for name in requested))
            else:
                command = K.remove_option(command, '-ts')
        draft_devices = K.option_value(command, '--spec-draft-device')
        if draft_devices and not set(draft_devices.split(',')) <= set(requested):
            raise TuneError('the draft devices must be included in the device budget')
    inherited = {name: value for name, value in os.environ.items()
                 if name.startswith(('GGML_', 'LLAMA_')) or name in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')}
    return arm, command, {**inherited, **arm.env, **cfg['env']}


def identity(cfg: Mapping[str, Any], arm, command: Sequence[str]) -> dict[str, Any]:
    server = Path(command[0])
    libs = sorted(p for p in server.parent.glob("lib*.so*") if not p.is_symlink())
    return {"server": file_identity(server), "libraries": [file_identity(p) for p in libs],
            "models": [file_identity(Path(p)) for p in arm.model_paths]}


def make_tuner(cfg: Mapping[str, Any], log: Callable[[str], None] = print) -> Tuner:
    import measure as M
    from cpu import CpuBudget
    import placement
    arm, command, env = base_launch(cfg)
    cpu = CpuBudget.resolve(cfg['cpus'], cfg['max_cores'])
    # ROCr otherwise starts helper threads on all CPUs, even under taskset.
    env['HSA_OVERRIDE_CPU_AFFINITY_DEBUG'] = '0'
    command = cpu.apply(command, len(M.used_devices(command)), cfg['max_threads'])
    command = K.set_option(command, '-fa', 'on') if not (K.option_value(command, '-fa') or K.option_value(command, '--flash-attn')) else command
    out = Path(cfg["output"])
    pci_map = M.device_pci_map(env)
    clock_command = K.set_option(command, '--device', ','.join(set(cfg['devices']) | set(M.used_devices(command))))
    problems = M.clock_problems(clock_command, pci_map, cfg["clock_mode"])
    if problems:
        raise TuneError("GPU clock mode: " + "; ".join(problems))
    shm = Path("/dev/shm") / f"auto-tune-{digest(cfg['output'])}"

    def launcher(cmd, e, name):
        with cpu.affinity():
            return M.launch(cfg, arm, cmd, e, name.replace(" ", "_").replace("=", "-"), out / "artifacts", cpu_budget=cpu)

    def kld_runner(cmd, e, key, compare):
        shm.mkdir(exist_ok=True)
        with cpu.affinity():
            return M.kld(cfg, cmd, e, shm / f"{key}.kld", out / "kld" / f"{digest([cmd, e])}.log", compare=compare)

    def vram_check(metrics, cmd):
        caps = {d.name: {'min_free_mib': d.margin_mib, **({'max_used_mib': d.max_used_mib} if d.max_used_mib is not None else {})}
                for d in placement.devices_from_state(command, cfg['devices'], pci_map, M.vram_state(), int(env.get('LLAMA_MOE_CACHE_AUTO_MARGIN_MIB', '1536')))}
        return M.vram_violations(metrics.get("vram", {}), caps, pci_map)

    def placement_factory():
        return placement.prepare(cfg, command, env, pci_map, M.vram_state(), cpu)

    run_identity = {**identity(cfg, arm, command), 'cpu_budget': cpu.summary(), 'devices': cfg['devices'],
                    'clock_mode': cfg['clock_mode']}
    tuner = Tuner(cfg, command, env, run_identity, launcher, kld_runner, vram_check, log=log,
                  launch_only=True, placement_factory=placement_factory, cpu_budget=cpu)
    tuner.shm = shm
    return tuner


def plan_text(tuner: Tuner) -> str:
    cfg = tuner.cfg
    lines = [f"base: {cfg['recipe']} arm {cfg['arm']}", "command: " + " ".join(tuner.base[0]),
             "env: " + " ".join(f"{k}={v}" for k, v in sorted(tuner.base[1].items())), "", "knobs (current level first):"]
    selected = tuner.selected()
    if tuner.cpu_budget:
        lines += ['CPU budget: ' + json.dumps(tuner.cpu_budget.summary()),
                  'loop 1 only: placement dry runs and fresh-load verification run with the run command']
    launches = (1 if cfg["warmup_launch"] else 0) + 1 + CONFIRM_LAUNCHES
    if tuner.launch_only:
        launches += 2
    precision = 0
    for knob, levels in selected:
        lines.append(f"  {knob.cls:9s} {knob.name:34s} {' '.join(levels)}")
        launches += len(levels) - 1
        if knob.cls == "precision":
            precision += sum(1 for level in levels if level != knob.exact)
    if precision:
        launches += precision + 1
    skipped = [k.name for k in K.REGISTRY if k.cls == "precision" and cfg["kld"]["limit"] is None and k.applies(tuner.context(*tuner.base))
               and (not tuner.launch_only or k.reload == 'process')]
    if skipped:
        lines.append(f"  not tuned without kld.limit: {', '.join(skipped)}")
    optin = [k.name for k in K.REGISTRY if not k.default and k.applies(tuner.context(*tuner.base)) and k.name not in cfg["knobs"]["include"]
             and (not tuner.launch_only or k.reload == 'process')]
    if optin:
        lines.append(f"  opt-in knobs that apply (add to knobs.include): {', '.join(optin)}")
    budget = cfg["budget"]
    lines += ["", f"launches at most (KLD runs count as launches): {launches}; budget {budget['max_launches']} launches, {budget['max_hours']} h"]
    if launches > budget["max_launches"]:
        lines.append("the budget ends the search early: the knobs at the end of the list are then not tested")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-launches", type=int, help="override budget.max_launches")
    parser.add_argument("--max-hours", type=float, help="override budget.max_hours")
    cpu_options = parser.add_mutually_exclusive_group()
    cpu_options.add_argument('--cpus', help='use exactly this Linux CPU list')
    cpu_options.add_argument('--max-cores', type=int, help='use at most this many physical cores')
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        if args.cpus is not None:
            cfg.update(cpus=args.cpus, max_cores=None)
        if args.max_cores is not None:
            cfg.update(max_cores=args.max_cores, cpus=None)
        if args.max_launches is not None:
            cfg["budget"]["max_launches"] = args.max_launches
        if args.max_hours is not None:
            cfg["budget"]["max_hours"] = args.max_hours
        import report as R
        if args.command == "report":
            out = Path(cfg["output"])
            state = json.loads((out / "state.json").read_text())
            R.write(cfg, state, Store(out), out)
            print(out / "report.md")
            return 0
        tuner = make_tuner(cfg, log=lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True))
        if args.command == "plan":
            print(plan_text(tuner))
            return 0
        try:
            from measure import benchmark_locks
            with benchmark_locks(tuner.base[0], cfg['devices']):
                state = tuner.run()
        finally:
            shm = getattr(tuner, "shm", None)
            if shm is not None and shm.is_dir():
                for path in shm.glob("*.kld"):
                    path.unlink()
                shm.rmdir()
        state["base_command"] = list(tuner.base[0])
        state["base_env"] = tuner.base[1]
        out = Path(cfg["output"])
        (out / "state.json").write_text(json.dumps(state, indent=2) + "\n")
        R.write(cfg, state, tuner.store, out)
        print(out / "report.md")
    except (TuneError, ValueError, OSError) as error:
        print(f"auto-tune: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
