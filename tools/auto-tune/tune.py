#!/usr/bin/env python3
"""Tune llama-server launch settings for one model on fixed devices.

Phase 1 tunes the batch sizes, threads and (for models with CPU experts) the expert cache margin. Phase 2 tunes
speculative decoding: the draft model's settings and at most one n-gram method with its three parameters. Both phases
change one setting at a time in one running server (POST /tune/context, no reload). The best result is confirmed with
fresh launches against the starting command.

    python3 tools/auto-tune/tune.py --prompts tools/auto-tune/prompts/default.yaml --out /tmp/tune-qwen \\
        -- build/bin/llama-server -m model.gguf -ngl 999 -fa on -c 8192 --spec-type draft-mtp
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cmdline as C  # noqa: E402
import prompts as PB  # noqa: E402
from server import Rejected, Server, ServerError  # noqa: E402

# n-gram methods with their three parameters; the first level is the server default and the start of the search.
# ngram-cache (needs a static cache file) and ngram-map-k4v (experimental) are left out.
NGRAM: dict[str, list[tuple[str, list[str]]]] = {
    "ngram-simple": [("--spec-ngram-simple-size-n", ["12", "6", "24"]), ("--spec-ngram-simple-size-m", ["48", "16"]),
                     ("--spec-ngram-simple-min-hits", ["1", "2"])],
    "ngram-map-k":  [("--spec-ngram-map-k-size-n", ["12", "6", "24"]), ("--spec-ngram-map-k-size-m", ["48", "16"]),
                     ("--spec-ngram-map-k-min-hits", ["1", "2"])],
    "ngram-mod":    [("--spec-ngram-mod-n-match", ["24", "16", "32"]), ("--spec-ngram-mod-n-min", ["48", "16"]),
                     ("--spec-ngram-mod-n-max", ["64", "32"])],
}
DRAFT_N_MAX = ["2", "3", "4", "5", "6"]
DRAFT_P_MIN = ["0", "0.5", "0.8"]
DRAFT_MODEL_OPTIONS = ("-md", "--model-draft", "--spec-draft-model", "--spec-draft-hf")
HOST_EXPERT_OPTIONS = ("-cmoe", "--cpu-moe", "-ncmoe", "--n-cpu-moe")
MARGIN_ENV = "LLAMA_MOE_CACHE_AUTO_MARGIN_MIB"

# phase 1 settings: name -> (/tune/context key, CLI options, default when the command does not set it)
CONTEXT_KNOBS = {
    "ub":  ("ubatch_size", ("-ub", "--ubatch-size"), 512),
    "b":   ("batch_size", ("-b", "--batch-size"), 2048),
    "ubd": ("ubatch_size_draft", ("-ubd", "--spec-draft-ubatch-size"), None),
    "t":   ("threads", ("-t", "--threads"), None),
    "tb":  ("threads_batch", ("-tb", "--threads-batch"), None),
}


def physical_cores() -> int:
    cores = set()
    for cpu in os.sched_getaffinity(0):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            cores.add(((base / "physical_package_id").read_text().strip(), (base / "core_id").read_text().strip()))
        except OSError:
            cores.add(("cpu", str(cpu)))
    return len(cores)


def model_has_mtp(model: str) -> bool:
    """True when the GGUF says the model has MTP (next-n prediction) layers."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    try:
        from gguf import GGUFReader
        reader = GGUFReader(model)
    except Exception:  # an unreadable file or a missing gguf-py: assume no MTP
        return False
    for name, field in reader.fields.items():
        if name.endswith(".nextn_predict_layers"):
            return int(field.parts[field.data[0]][0]) > 0
    return False


def find_drafter(command: Sequence[str]) -> str | None:
    """The draft method, decided once: a draft model if the command has one, else MTP if the model has MTP layers."""
    types = [t for t in (C.option_value(command, "--spec-type") or "").split(",") if t.startswith("draft-")]
    if C.first_value(command, *DRAFT_MODEL_OPTIONS):
        return types[0] if types else "draft-simple"
    if "draft-mtp" in types:
        return "draft-mtp"
    model = C.first_value(command, "-m", "--model")
    return "draft-mtp" if model and model_has_mtp(model) else None


def spec_options(command: Sequence[str]) -> dict[str, str | None]:
    """The speculative strategy options of a command other than --spec-type, in order (None for a switch)."""
    out: dict[str, str | None] = {}
    for i, item in enumerate(command):
        name, eq, value = item.partition("=")
        if not C.is_spec_strategy(name) or name == "--spec-type":
            continue
        if name in C.SPEC_SWITCHES:
            out[name] = None
        else:
            out[name] = value if eq else (command[i + 1] if i + 1 < len(command) else None)
    return out


def merge_spec(base: dict[str, str | None], over: dict[str, str | None]) -> dict[str, str | None]:
    """*base* updated with *over*; an adaptive switch in *over* replaces the other one in *base*."""
    out = dict(base)
    if "--spec-draft-adaptive" in over or "--no-spec-draft-adaptive" in over:
        out.pop("--spec-draft-adaptive", None)
        out.pop("--no-spec-draft-adaptive", None)
    out.update(over)
    return out


def thread_levels(start: int | None, gpus: int, max_threads: int | None) -> list[int]:
    p = physical_cores()
    top = max_threads or max(1, p - 2 * gpus)
    levels = {max(1, p // 4), max(1, p // 2), max(1, 3 * p // 4), top}
    if start:
        levels.add(start)
    return sorted(level for level in levels if level <= top)


def gmean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else 0.0


def decode_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Tokens per second per domain (all tokens / all time of the domain), and the geometric mean over the domains."""
    by: dict[str, dict[str, float]] = {}
    for r in rows:
        d = by.setdefault(r["domain"], {"n": 0, "ms": 0, "draft_n": 0, "draft_accepted": 0})
        for k in d:
            d[k] += r[k]
    domains = {name: {"tps": d["n"] / (d["ms"] / 1000.0),
                      "accept": d["draft_accepted"] / d["draft_n"] if d["draft_n"] else None} for name, d in by.items()}
    return {"tps": gmean([d["tps"] for d in domains.values()]), "domains": domains}


class Tuner:
    def __init__(self, args: argparse.Namespace, command: Sequence[str]) -> None:
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.bank = PB.load(Path(args.prompts))
        self.sampling = json.loads(args.sampling)
        self.deadline = time.time() + args.max_hours * 3600
        self.drafter = find_drafter(command)
        command = tuple(command)
        if self.drafter and self.drafter not in (C.option_value(command, "--spec-type") or ""):
            # the draft context must exist at launch for phase 2
            command = C.set_option(command, "--spec-type", ",".join(
                [t for t in (C.option_value(command, "--spec-type") or "").split(",") if t and t != "none"] + [self.drafter]))
        self.start_command = command
        self.start_spec = ["--spec-type", C.option_value(command, "--spec-type") or "none",
                           *C.spec_args(spec_options(command))]
        self.gpus = len((C.option_value(command, "--device") or C.option_value(command, "-dev") or "x").split(","))
        self.host_experts = any(o in command for o in HOST_EXPERT_OPTIONS) or any(
            "exps" in v and "CPU" in v for v in (command[i + 1] for i, x in enumerate(command[:-1]) if x in ("-ot", "--override-tensor")))
        self.env: dict[str, str] = {}
        self.server: Server | None = None
        self.n_launch = 0
        self.results = open(self.out / "results.jsonl", "a")
        self.tokens: list[int] | None = None

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.out / "tune.log", "a") as f:
            f.write(line + "\n")

    def record(self, item: dict[str, Any]) -> None:
        self.results.write(json.dumps(item) + "\n")
        self.results.flush()

    # --- server ---------------------------------------------------------------------------------------------------

    def launch(self, command: Sequence[str], env: dict[str, str], name: str) -> Server:
        self.stop()
        self.n_launch += 1
        self.log(f"launch {name}: {shlex.join(command)}")
        self.server = Server(command, env, self.args.port, self.out / f"server-{self.n_launch}-{name}.log")
        self.log(f"  ready after {self.server.load_s:.0f} s")
        self.tokens = None
        self.warm()
        return self.server

    def stop(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None

    def warm(self) -> None:
        """One unmeasured short request and one prefill of each length: the first ones after a (re)start are slow."""
        assert self.server
        self.server.chat(self.bank[0].prompt, 16, self.sampling)
        self.prefill(reps=1)

    # --- measurement ----------------------------------------------------------------------------------------------

    def prefill(self, reps: int = 3) -> dict[str, float]:
        """Median prompt tokens/s at each prefill length over *reps* requests, slices of the prompt bank text."""
        assert self.server
        if self.tokens is None:
            text = "\n\n".join(p.prompt for p in self.bank)
            self.tokens = self.server.tokenize(text)
        out = {}
        for length in self.args.prefill:
            need = reps * length
            ids = (self.tokens * (need // len(self.tokens) + 1))[:need]
            out[str(length)] = statistics.median(self.server.prefill(ids[i * length:(i + 1) * length]) for i in range(reps))
        return out

    def decode(self, bank: Sequence[PB.Prompt], max_tokens: int | None = None) -> dict[str, Any]:
        assert self.server
        rows = [{"domain": p.domain, "id": p.id, **self.server.chat(p.prompt, max_tokens or self.args.max_tokens, self.sampling)}
                for p in bank]
        return {**decode_summary(rows), "rows": rows}

    def score(self, m: dict[str, Any], ref: dict[str, Any]) -> float:
        wd = self.args.decode_weight
        s = (m["decode"]["tps"] / ref["decode"]["tps"]) ** wd
        for length, tps in m["prefill"].items():
            s *= (tps / ref["prefill"][length]) ** ((1 - wd) / len(m["prefill"]))
        return s

    def time_left(self) -> bool:
        return time.time() < self.deadline

    # --- phase 1 --------------------------------------------------------------------------------------------------

    def phase1(self) -> dict[str, Any]:
        cmd = self.start_command
        settings: dict[str, Any] = {}
        for name, (key, options, default) in CONTEXT_KNOBS.items():
            value = C.first_value(cmd, *options)
            settings[key] = int(value) if value is not None else default
        if self.host_experts:
            settings["env"] = {MARGIN_ENV: os.environ.get(MARGIN_ENV, "1536")}
        bank = PB.subset(self.bank, 1)
        self.launch(cmd, self.env, "phase1")

        def measure(s: dict[str, Any], label: str) -> dict[str, Any] | None:
            assert self.server
            body = {k: v for k, v in s.items() if v is not None}
            try:
                secs = self.server.recreate(body)
                self.warm()
                # prefill with the starting speculative settings (a drafter ingests the prompt too); decode without
                # speculation, which has no acceptance noise and still shows the cost of threads and cache size
                pre = self.prefill(reps=5)
                spec = self.start_spec != ["--spec-type", "none"]
                if spec:
                    self.server.post("/speculative", {"args": ["--spec-type", "none"]})
                dec = self.decode(bank, min(128, self.args.max_tokens))
                if spec:
                    self.server.post("/speculative", {"args": self.start_spec})
            except Rejected as e:
                self.log(f"  {label}: rejected ({str(e)[:200]})")
                self.record({"phase": 1, "label": label, "settings": s, "rejected": str(e)})
                return None
            except ServerError as e:
                self.log(f"  {label}: server died ({e}); relaunching with the best settings")
                self.record({"phase": 1, "label": label, "settings": s, "failed": str(e)})
                self.launch(self.command_for(best), self.env_for(best), "phase1-relaunch")
                return None
            return {"prefill": pre, "decode": dec, "recreate_s": secs}

        best = dict(settings)
        ref = measure(best, "start")
        if ref is None:
            raise ServerError("the starting settings were rejected")
        best_score = 1.0
        self.log(f"  start: decode (no speculation) {ref['decode']['tps']:.1f} t/s, prefill {self.fmt_prefill(ref['prefill'])}")
        self.record({"phase": 1, "label": "start", "settings": best, "metrics": ref, "score": 1.0})
        threads = thread_levels(best["threads"], self.gpus, self.args.max_threads)
        plan: list[tuple[str, Callable[[dict[str, Any]], list[Any]]]] = [
            ("ubatch_size", lambda s: [256, 512, 1024, 2048, 4096]),
            ("batch_size", lambda s: [v for v in (s["ubatch_size"], 2048, 4096, 8192, 16384) if v >= s["ubatch_size"]]),
            ("threads", lambda s: threads),
            ("threads_batch", lambda s: threads),
        ]
        if self.drafter:
            plan.insert(2, ("ubatch_size_draft", lambda s: [128, 512, 2048]))
        if self.host_experts:
            plan.append((MARGIN_ENV, lambda s: ["1024", "1536", "3072", "6144"]))
        for key, levels in plan:
            for level in levels(best):
                if not self.time_left():
                    self.log("  time budget used up")
                    return best
                cand = json.loads(json.dumps(best))
                current = cand["env"][MARGIN_ENV] if key == MARGIN_ENV else cand.get(key)
                if level == current:
                    continue
                if key == MARGIN_ENV:
                    cand["env"][MARGIN_ENV] = level
                else:
                    cand[key] = level
                if key == "ubatch_size" and (cand["batch_size"] or 2048) < level:
                    cand["batch_size"] = level
                label = f"{key}={level}"
                m = measure(cand, label)
                if m is None:
                    continue
                s = self.score(m, ref)
                self.log(f"  {label}: score {s:.3f}, decode {m['decode']['tps']:.1f} t/s, prefill {self.fmt_prefill(m['prefill'])}")
                self.record({"phase": 1, "label": label, "settings": cand, "metrics": m, "score": s})
                if s > best_score * (1 + self.args.margin):
                    best, best_score = cand, s
                    self.log(f"  -> new best {label}")
        self.log(f"phase 1 best: {self.describe(best)} (score {best_score:.3f})")
        return best

    def command_for(self, settings: dict[str, Any], spec: list[str] | None = None) -> tuple[str, ...]:
        cmd = self.start_command
        for name, (key, options, default) in CONTEXT_KNOBS.items():
            value = settings.get(key)
            start = C.first_value(cmd, *options)
            if value is None or str(value) == (start if start is not None else str(default)):
                continue
            for o in options[1:]:
                cmd = C.remove_option(cmd, o)
            cmd = C.set_option(cmd, options[0], str(value))
        if spec is not None:
            cmd = (*C.strip_spec(cmd), *spec)
        return cmd

    def env_for(self, settings: dict[str, Any]) -> dict[str, str]:
        return {**self.env, **settings.get("env", {})}

    @staticmethod
    def describe(settings: dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in settings.items() if v is not None)

    @staticmethod
    def fmt_prefill(p: dict[str, float]) -> str:
        return " ".join(f"{k}:{v:.0f}" for k, v in p.items()) + " t/s"

    # --- phase 2 --------------------------------------------------------------------------------------------------

    def phase2(self, settings: dict[str, Any]) -> list[dict[str, Any]]:
        """Returns the measured strategies, best first: {"spec": [...args], "decode": ...}."""
        if self.server is None or not self.server.alive():
            self.launch(self.command_for(settings), self.env_for(settings), "phase2")
        bank = PB.subset(self.bank, self.args.phase2_prompts)
        # /tune/context applies spec_args on top of the current settings, so every point sends every drafter option
        drafter_opts = self.effective_draft_options()
        start_opts = merge_spec(drafter_opts, spec_options(self.start_command))
        start_spec = ["--spec-type", C.option_value(self.start_command, "--spec-type") or "none", *C.spec_args(start_opts)]
        self.log(f"  drafter settings at the start: {shlex.join(C.spec_args(drafter_opts)) or 'none'}")
        tried: dict[str, dict[str, Any]] = {}

        def args_of(draft: dict[str, str | None], ngram: str | None, ngram_opts: dict[str, str]) -> list[str]:
            types = ([self.drafter] if self.drafter else []) + ([ngram] if ngram else [])
            return ["--spec-type", ",".join(types) or "none", *C.spec_args(draft), *C.spec_args(dict(ngram_opts))]

        def measure(spec: list[str], label: str) -> float | None:
            key = shlex.join(spec)
            if key in tried:
                return tried[key]["decode"]["tps"]
            if not self.time_left():
                return None
            assert self.server
            body = {k: v for k, v in settings.items() if v is not None}
            try:
                self.server.recreate({**body, "spec_args": spec})
                self.warm()
                d = self.decode(bank)
            except Rejected as e:
                self.log(f"  {label}: rejected ({str(e)[:200]})")
                self.record({"phase": 2, "label": label, "spec": spec, "rejected": str(e)})
                tried[key] = {"spec": spec, "decode": {"tps": 0.0}}
                return None
            except ServerError as e:
                self.log(f"  {label}: server died ({e}); relaunching")
                self.record({"phase": 2, "label": label, "spec": spec, "failed": str(e)})
                tried[key] = {"spec": spec, "decode": {"tps": 0.0}}
                self.launch(self.command_for(settings), self.env_for(settings), "phase2-relaunch")
                return None
            tried[key] = {"spec": spec, "label": label, "decode": d}
            acc = [v["accept"] for v in d["domains"].values() if v["accept"] is not None]
            self.log(f"  {label}: decode {d['tps']:.1f} t/s" + (f", acceptance {statistics.mean(acc):.2f}" if acc else ""))
            self.record({"phase": 2, "label": label, "spec": spec, "decode": d})
            return d["tps"]

        measure(start_spec, "start")

        def descend(draft: dict[str, str | None], ngram: str | None, ngram_opts: dict[str, str],
                    knobs: list[tuple[str, list[str]]], into_draft: bool) -> tuple[dict, dict, float]:
            best_tps = measure(args_of(draft, ngram, ngram_opts), f"{ngram or 'draft'} start") or 0.0
            for option, levels in knobs:
                for level in levels:
                    cand_draft, cand_ngram = dict(draft), dict(ngram_opts)
                    target = cand_draft if into_draft else cand_ngram
                    if option == "adaptive":
                        on = level == "on"
                        if ("--spec-draft-adaptive" in target) == on and ("--no-spec-draft-adaptive" in target) != on:
                            continue
                        target.pop("--spec-draft-adaptive", None)
                        target.pop("--no-spec-draft-adaptive", None)
                        target["--spec-draft-adaptive" if on else "--no-spec-draft-adaptive"] = None
                    else:
                        if target.get(option) == level:
                            continue
                        target[option] = level
                    tps = measure(args_of(cand_draft, ngram, cand_ngram), f"{ngram or 'draft'} {option}={level}")
                    if tps and tps > best_tps * (1 + self.args.margin):
                        draft, ngram_opts, best_tps = cand_draft, cand_ngram, tps
            return draft, ngram_opts, best_tps

        draft = {k: v for k, v in start_opts.items() if not k.startswith("--spec-ngram-")}
        if self.drafter:
            draft, _, _ = descend(draft, None, {}, [("--spec-draft-n-max", DRAFT_N_MAX), ("--spec-draft-p-min", DRAFT_P_MIN),
                                                   ("adaptive", ["off", "on"])], True)
        for method, knobs in NGRAM.items():
            start = {option: levels[0] for option, levels in knobs}
            descend(draft, method, start, [(o, levels[1:]) for o, levels in knobs], False)

        ranked = sorted((t for t in tried.values() if t["decode"]["tps"] > 0), key=lambda t: -t["decode"]["tps"])
        # the top three and the start on the whole bank: the subset is noisy, the start is the fallback
        top = ranked[:3]
        start_key = shlex.join(start_spec)
        if start_key in tried and all(shlex.join(t["spec"]) != start_key for t in top):
            top.append(tried[start_key])
        finals = []
        for t in top:
            assert self.server
            try:
                self.server.recreate({**{k: v for k, v in settings.items() if v is not None}, "spec_args": t["spec"]})
                self.warm()
                d = self.decode(self.bank)
            except ServerError as e:  # Rejected included
                self.log(f"  full bank {shlex.join(t['spec'])}: failed ({str(e)[:200]})")
                if not self.server.alive():
                    self.launch(self.command_for(settings), self.env_for(settings), "phase2-relaunch")
                continue
            self.log(f"  full bank {shlex.join(t['spec'])}: {d['tps']:.1f} t/s")
            self.record({"phase": 2, "label": "full", "spec": t["spec"], "decode": d})
            finals.append({"spec": t["spec"], "decode": d})
        finals.sort(key=lambda t: -t["decode"]["tps"])
        return finals

    def effective_draft_options(self) -> dict[str, str | None]:
        """The drafter's n-max, p-min and adaptive switch as the running server has them (GET /speculative)."""
        if not self.drafter:
            return {}
        assert self.server
        info = self.server.get("/speculative")
        impls = [i for i in info.get("impls", []) if "draft" in str(i.get("type", "")) or "mtp" in str(i.get("type", "")).lower()]
        impl = impls[0] if impls else (info.get("impls") or [{}])[0]
        n_max = impl.get("n_max") or info.get("n_max")
        adaptive = str(impl.get("adaptive", "")).lower() not in ("", "none", "off", "false", "0")
        out: dict[str, str | None] = {}
        if n_max:
            out["--spec-draft-n-max"] = str(n_max)
        out["--spec-draft-p-min"] = "0"
        out["--spec-draft-adaptive" if adaptive else "--no-spec-draft-adaptive"] = None
        return out

    # --- confirm --------------------------------------------------------------------------------------------------

    def confirm(self, final_cmd: Sequence[str], final_env: dict[str, str]) -> list[dict[str, Any]]:
        runs = []
        for label, cmd, env in (("start", self.start_command, self.env), ("tuned", final_cmd, final_env),
                                ("start", self.start_command, self.env)):
            self.launch(cmd, env, f"confirm-{label}")
            m = {"label": label, "prefill": self.prefill(), "decode": self.decode(self.bank)}
            self.log(f"  confirm {label}: decode {m['decode']['tps']:.1f} t/s, prefill {self.fmt_prefill(m['prefill'])}")
            self.record({"phase": "confirm", **m})
            runs.append(m)
        self.stop()
        return runs

    def report(self, final_cmd: Sequence[str], final_env: dict[str, str], runs: list[dict[str, Any]]) -> None:
        lines = ["# Tune report", "", f"Prompts: {self.args.prompts} ({len(self.bank)} prompts, max {self.args.max_tokens} "
                 f"output tokens, sampling {json.dumps(self.sampling)}). Draft method: {self.drafter or 'none'}.", ""]
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in final_env.items())
        lines += ["## Tuned command", "", "```sh", (env_prefix + " " if env_prefix else "") + shlex.join(final_cmd), "```", ""]
        if runs:
            start = [r for r in runs if r["label"] == "start"]
            tuned = next(r for r in runs if r["label"] == "tuned")
            base = {"decode": {"tps": statistics.mean(r["decode"]["tps"] for r in start)},
                    "prefill": {k: statistics.mean(r["prefill"][k] for r in start) for k in tuned["prefill"]}}
            score = self.score(tuned, base)
            spread = max(self.score(r, base) for r in start) - min(self.score(r, base) for r in start)
            noise = max(self.args.margin, spread)
            verdict = (f"The tuned command is {score - 1:+.1%} by the score, more than the noise between the two start runs "
                       f"({spread:.1%}): use it." if score - 1 > noise else
                       f"The tuned command is {score - 1:+.1%} by the score, within the noise between the two start runs "
                       f"({spread:.1%}) or the {self.args.margin:.0%} margin: no reliable gain, keep your command.")
            lines += ["## Confirmation (fresh launches: start, tuned, start)", "", verdict, "",
                      "| | start (mean of 2) | tuned | change |", "| --- | ---: | ---: | ---: |",
                      f"| score ({self.args.decode_weight:.0%} decode) | 1.000 | {score:.3f} | |",
                      f"| decode t/s (all domains) | {base['decode']['tps']:.1f} | {tuned['decode']['tps']:.1f} | "
                      f"{tuned['decode']['tps'] / base['decode']['tps'] - 1:+.1%} |"]
            for k, v in tuned["prefill"].items():
                lines.append(f"| prefill {k} tokens, t/s | {base['prefill'][k]:.0f} | {v:.0f} | {v / base['prefill'][k] - 1:+.1%} |")
            lines += ["", "Decode by domain (tokens/s, draft acceptance):", "", "| domain | start | tuned | change |",
                      "| --- | ---: | ---: | ---: |"]
            for dom, d in tuned["decode"]["domains"].items():
                s = statistics.mean(r["decode"]["domains"][dom]["tps"] for r in start)
                sa = [r["decode"]["domains"][dom]["accept"] for r in start if r["decode"]["domains"][dom]["accept"] is not None]
                acc = lambda a: f" ({a:.2f})" if a is not None else ""
                lines.append(f"| {dom} | {s:.1f}{acc(statistics.mean(sa) if sa else None)} | {d['tps']:.1f}{acc(d['accept'])} | "
                             f"{d['tps'] / s - 1:+.1%} |")
        lines += ["", f"Every measured point is in {self.out / 'results.jsonl'}; the log is {self.out / 'tune.log'}.", ""]
        (self.out / "report.md").write_text("\n".join(lines))
        (self.out / "command.txt").write_text((env_prefix + " " if env_prefix else "") + shlex.join(final_cmd) + "\n")

    def run(self) -> int:
        t0 = time.time()
        self.log(f"draft method: {self.drafter or 'none'}; host experts: {self.host_experts}; {len(self.bank)} prompts")
        try:
            settings = self.phase1() if not self.args.skip_phase1 else {}
            final_spec = None
            if not self.args.skip_phase2:
                finals = self.phase2(settings)
                if finals:
                    final_spec = finals[0]["spec"]
                    self.log(f"phase 2 best: {shlex.join(final_spec)}")
            final_cmd = self.command_for(settings, final_spec)
            final_env = self.env_for(settings)
            runs = self.confirm(final_cmd, final_env) if not self.args.no_confirm else []
            self.report(final_cmd, final_env, runs)
        finally:
            self.stop()
        self.log(f"done in {(time.time() - t0) / 60:.0f} min; report: {self.out / 'report.md'}")
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "prompts" / "default.yaml"),
                    help="prompt bank: YAML list of {domain, id, prompt} or text with prompts separated by '---' lines")
    ap.add_argument("--out", required=True, help="output directory (report.md, command.txt, results.jsonl, logs)")
    ap.add_argument("--max-tokens", type=int, default=512, help="output tokens per prompt (default 512)")
    ap.add_argument("--prefill", type=lambda s: [int(x) for x in s.split(",")], default=[256, 2048],
                    help="prefill lengths in tokens (default 256,2048)")
    ap.add_argument("--decode-weight", type=float, default=0.8, help="weight of decode in the score; prefill gets the rest")
    ap.add_argument("--sampling", default='{"temperature": 0}', help="sampling fields of every request, as JSON (default greedy)")
    ap.add_argument("--phase2-prompts", type=int, default=2, help="prompts per domain for each phase-2 point (default 2)")
    ap.add_argument("--max-threads", type=int, help="most threads to try (default: physical cores less 2 per GPU)")
    ap.add_argument("--margin", type=float, default=0.02, help="a change must beat the best by this fraction (default 0.02)")
    ap.add_argument("--max-hours", type=float, default=4.0, help="search time budget; the confirmation runs after it")
    ap.add_argument("--port", type=int, default=18190)
    ap.add_argument("--skip-phase1", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true")
    ap.add_argument("--no-confirm", action="store_true", help="skip the fresh-launch confirmation")
    args = ap.parse_args(argv[:split])

    def stop(*_: Any) -> None:
        raise KeyboardInterrupt  # unwinds through Tuner.run, which stops the server

    signal.signal(signal.SIGTERM, stop)
    command = argv[split + 1:]
    if not command:
        ap.error("no llama-server command after --")
    return Tuner(args, command).run()


if __name__ == "__main__":
    sys.exit(main())
