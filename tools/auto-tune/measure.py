"""Measurements of one tuner config: a server launch through tools/bottlenecks/profile.py, VRAM use while it runs, and the
KLD of its precision settings (llama-perplexity)."""
from __future__ import annotations

import json
import contextlib
import fcntl
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.bottlenecks import profile as P  # noqa: E402

from knobs import option_value, set_option  # noqa: E402
from cpu import parse_list  # noqa: E402
from config import TuneError  # noqa: E402

PREFILL_CLIENT = Path(__file__).resolve().parent / "prefill_client.py"
DRM = Path("/sys/class/drm")

# llama-server options that llama-perplexity takes as well and that can change the numerics or the placement
PPL_VALUE_OPTIONS = {"-ngl", "--n-gpu-layers", "-dev", "--device", "-ts", "--tensor-split", "-sm", "--split-mode", "-ot",
                     "--override-tensor", "-ncmoe", "--n-cpu-moe", "--numa", "-t", "--threads", "-fa", "--flash-attn", "-lm",
                     "--load-mode", "-lzm", "--lazy-mode", "-ctk", "--cache-type-k", "-ctv", "--cache-type-v", "-mg", "--main-gpu"}
PPL_SWITCHES = {"-cmoe", "--cpu-moe", "--no-mmap", "--mlock"}

_PCI_SCRIPT = r"""
import ctypes, json, sys
out = {}
try:
    if sys.argv[1] == "hip":
        lib = ctypes.CDLL("libamdhip64.so")
        n = ctypes.c_int()
        if lib.hipGetDeviceCount(ctypes.byref(n)) == 0:
            for i in range(n.value):
                buf = ctypes.create_string_buffer(64)
                if lib.hipDeviceGetPCIBusId(buf, 64, i) == 0:
                    out[f"ROCm{i}"] = buf.value.decode()
    else:
        lib = ctypes.CDLL("libcuda.so.1")
        if lib.cuInit(0) == 0:
            n = ctypes.c_int()
            lib.cuDeviceGetCount(ctypes.byref(n))
            for i in range(n.value):
                dev = ctypes.c_int()
                buf = ctypes.create_string_buffer(64)
                if lib.cuDeviceGet(ctypes.byref(dev), i) == 0 and lib.cuDeviceGetPCIBusId(buf, 64, dev) == 0:
                    out[f"CUDA{i}"] = buf.value.decode()
except OSError:
    pass
print(json.dumps(out))
"""


class LaunchFailed(TuneError):
    pass


@contextlib.contextmanager
def benchmark_locks(command, devices):
    names = set(devices) | set(used_devices(command))
    names.update((option_value(command, '--spec-draft-device') or '').split(','))
    locks = ['/tmp/delphi-representative.lock']
    if any(n.startswith('ROCm') for n in names):
        locks.append('/tmp/delphi-rocm.lock')
    if any(n.startswith('CUDA') for n in names):
        locks.append('/tmp/delphi-cuda.lock')
    with contextlib.ExitStack() as stack:
        for path in locks:
            handle = stack.enter_context(open(path, 'a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LaunchFailed(f'benchmark lock is busy: {path}; run the tuner without an outer flock') from None
        yield


class CpuSampler:
    """Audit the affinity of every observed server thread, including helper and draft threads."""
    def __init__(self, command, budget):
        self.binary = Path(command[0]).resolve()
        self.allowed = set(budget.cpus)
        self.children = Path(f'/proc/{os.getpid()}/task/{threading.get_native_id()}/children')
        self.samples = 0
        self.max_threads = 0
        self.violations = set()
        self.stopped = set()
        self.stop = threading.Event()

    def sample(self):
        try:
            children = self.children.read_text().split()
        except OSError:
            return
        for pid in children:
            process = Path('/proc') / pid
            try:
                if (process / 'exe').resolve() != self.binary:
                    continue
                threads = list((process / 'task').iterdir())
                self.max_threads = max(self.max_threads, len(threads))
                for thread in threads:
                    try:
                        status = dict(line.split(':', 1) for line in (thread / 'status').read_text().splitlines() if ':' in line)
                        mask = parse_list(status['Cpus_allowed_list'].strip())
                    except (OSError, KeyError):
                        continue
                    self.samples += 1
                    if not mask <= self.allowed:
                        self.violations.add(f'{pid}/{thread.name}: CPUs outside the budget {sorted(mask - self.allowed)}')
                        if pid not in self.stopped:
                            try:
                                os.kill(int(pid), signal.SIGINT)
                                self.stopped.add(pid)
                            except ProcessLookupError:
                                pass
            except OSError:
                continue

    def loop(self):
        while not self.stop.wait(0.1):
            self.sample()

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, name='cpu-budget-audit', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()

    def summary(self):
        return {'thread_samples': self.samples, 'max_threads': self.max_threads, 'violations': sorted(self.violations)}


def short_pci(value: str) -> str:
    """'0000:43:00.0' or '00000000:43:00.0' -> '43:00.0'"""
    value = value.strip().lower()
    return value.split(":", 1)[1] if value.count(":") == 2 else value


def device_pci_map(env: Mapping[str, str]) -> dict[str, str]:
    """llama device name -> PCI slot. Runs in a child process, so this process holds no GPU context."""
    out: dict[str, str] = {}
    full_env = {**os.environ, **env}
    for kind in ("hip", "cuda"):
        try:
            completed = subprocess.run([sys.executable, "-c", _PCI_SCRIPT, kind], env=full_env, capture_output=True, text=True, timeout=120)
            out.update({name: short_pci(pci) for name, pci in json.loads(completed.stdout or "{}").items()})
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return out


def vram_state() -> dict[str, tuple[float, float]]:
    """PCI slot -> (used MiB, total MiB) for every GPU."""
    out: dict[str, tuple[float, float]] = {}
    for device in sorted(DRM.glob("card[0-9]*/device")):
        try:
            used = int((device / "mem_info_vram_used").read_text())
            total = int((device / "mem_info_vram_total").read_text())
            uevent = (device / "uevent").read_text()
        except OSError:
            continue
        slot = next((line.split("=", 1)[1] for line in uevent.splitlines() if line.startswith("PCI_SLOT_NAME=")), None)
        if slot:
            out[short_pci(slot)] = (used / 2**20, total / 2**20)
    if shutil.which("nvidia-smi"):
        try:
            text = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            text = ""
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                try:
                    out[short_pci(parts[0])] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
    return out


class VramSampler:
    """Peak use and least free VRAM per GPU while a launch runs, and the use before it (other processes)."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.baseline: dict[str, tuple[float, float]] = {}
        self.peak: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        for pci, (used, _) in vram_state().items():
            self.peak[pci] = max(self.peak.get(pci, 0.0), used)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> "VramSampler":
        self.baseline = vram_state()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="vram-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for pci, (used0, total) in self.baseline.items():
            peak = self.peak.get(pci, used0)
            out[pci] = {"baseline_mib": round(used0), "peak_mib": round(peak), "total_mib": round(total),
                        "used_mib": round(peak - used0), "min_free_mib": round(total - peak)}
        return out


def used_devices(command: Sequence[str]) -> list[str]:
    value = option_value(command, "--device") or option_value(command, "-dev") or ""
    return [d for d in value.split(",") if d and d.lower() != "none"]


def vram_violations(vram: Mapping[str, Mapping[str, float]], caps: Mapping[str, Mapping[str, float]], pci_map: Mapping[str, str]) -> list[str]:
    out = []
    for device, cap in caps.items():
        pci = pci_map.get(device)
        item = vram.get(pci) if pci else None
        if item is None:
            out.append(f"{device}: no VRAM reading")
            continue
        if "min_free_mib" in cap and item["min_free_mib"] < cap["min_free_mib"]:
            out.append(f"{device}: {item['min_free_mib']} MiB free < {cap['min_free_mib']}")
        if "max_used_mib" in cap and item["used_mib"] > cap["max_used_mib"]:
            out.append(f"{device}: {item['used_mib']} MiB used > {cap['max_used_mib']}")
    return out


def clock_problems(command: Sequence[str], pci_map: Mapping[str, str], mode: str | None) -> list[str]:
    """Used amdgpu cards whose forced performance level is not *mode*."""
    if not mode:
        return []
    levels = {short_pci(gpu["pci"]): gpu.get("performance_level") for gpu in P._gpu_state()["gpus"] if gpu.get("vendor") == "amd"}
    out = []
    for device in used_devices(command):
        pci = pci_map.get(device)
        if pci in levels and levels[pci] != mode:
            out.append(f"{device} ({pci}) is at {levels[pci]}, not {mode}")
    return out


def _client(base: Sequence[str], osl: int) -> tuple[str, ...]:
    return set_option(base, "--osl", str(osl)) if "--osl" in base else tuple(base)


def measurement_arm(cfg, base_arm, command, env, name, cpu_budget=None):
    """Use the same profiler clients for fresh launches and retained contexts."""
    prefill = cfg["prefill"]
    measures = ()
    if cfg["weights"].get("prefill", 0) > 0:
        measures = tuple(
            P.ServerMeasurement(f"prefill-{length}", (sys.executable, str(PREFILL_CLIENT), "--url", "{url}", "--text", prefill["text"],
                                                      "--tokens", length, "--reps", str(prefill["reps"]), "--offset", str(7919 * i),
                                                      "--output", "{output}"), "prefill")
            for i, length in enumerate(prefill["lengths"]))
    osl = cfg["decode"]["osl"] if cfg["weights"].get("decode", 0) > 0 else 16
    client = _client(base_arm.client_command, osl)
    if cpu_budget:
        client = (sys.executable, str(Path(__file__).parent / 'verify_client.py'), '--url', '{url}', '--floor', str(cfg['context']),
                  '--output', '{artifact_dir}/verification.json', '--', *client)
    return replace(base_arm, name=name, server_command=tuple(command), env=dict(env), client_command=client,
                  extra_measurements=measures, trace_mode="submit" if cfg["trace"] else "off",
                  trace_prefill_min_tokens=0, trace_prefill_target_tokens=0)


def placement_warnings(log):
    return [line.strip() for line in log.splitlines()
            if re.search(r'expert cache auto:.*no tier|staging.*(?:disabled|fail)|failed.*staging', line, re.I)]


def launch(cfg: Mapping[str, Any], base_arm: P.ServerArm, command: Sequence[str], env: Mapping[str, str], name: str,
           artifacts: Path, cpu_budget=None) -> dict[str, Any]:
    """One server launch: the decode client, then the prefill lengths. Returns metrics; raises LaunchFailed."""
    arm = measurement_arm(cfg, base_arm, command, env, name, cpu_budget)
    recipe = P.Recipe(name="auto-tune", description="auto-tuner launch", arms=(arm,))
    started = time.time()
    audit = CpuSampler(command, cpu_budget) if cpu_budget else None
    with VramSampler() as vram, audit if audit else contextlib.nullcontext():
        try:
            result = P.run_recipe(recipe, artifacts, environment=cfg["env"], gpu_sample_interval_s=0)
        except (P.ProfileError, subprocess.TimeoutExpired, OSError, ValueError) as error:
            if audit and audit.violations:
                raise LaunchFailed('CPU budget audit failed: ' + '; '.join(sorted(audit.violations))) from error
            raise LaunchFailed(f"{error}; {stderr_tail(artifacts / name)}") from error
    item = result["arms"][0]
    decode = item["metrics"]
    metrics: dict[str, Any] = {
        "decode_t_s": decode.get("predicted_t_s"),
        "draft_n": decode.get("draft_n"),
        "draft_n_accepted": decode.get("draft_n_accepted"),
        "completion_tokens": decode.get("completion_tokens"),
        "prefill": {},
        "vram": vram.summary(),
        "wall_s": time.time() - started,
    }
    if audit:
        metrics['cpu_budget'] = audit.summary()
        if audit.violations or not audit.samples:
            raise LaunchFailed('CPU budget audit failed: ' + ('; '.join(sorted(audit.violations)) or 'no server threads observed'))
        metrics.update(json.loads((artifacts / name / 'verification.json').read_text()))
        log = '\n'.join((artifacts / name / f'server.{stream}.log').read_text(errors='replace') for stream in ('stdout', 'stderr'))
        metrics['placement_warnings'] = placement_warnings(log)
    for extra in item["extra_measurements"]:
        m = extra["metrics"]
        metrics["prefill"][str(m["target_tokens"])] = {"ttft_s": m.get("latency_s"), "t_s": m.get("prompt_t_s"), "prompt_ms": m.get("prompt_ms")}
    if cfg["trace"] and item.get("trace", {}).get("speculative"):
        metrics["speculative_cycle"] = item["trace"]["speculative"].get("cycle")
    return metrics


def stderr_tail(arm_dir: Path, lines: int = 6) -> str:
    try:
        text = (arm_dir / "server.stderr.log").read_text(errors="replace").splitlines()
    except OSError:
        return "no server log"
    keep = [line for line in text if re.search(r"error|fail|abort|out of memory|exception", line, re.I)] or text
    return " | ".join(keep[-lines:])


def perplexity_command(cfg: Mapping[str, Any], command: Sequence[str], logits: Path, compare: bool) -> list[str]:
    model = option_value(command, "-m") or option_value(command, "--model")
    if not model:
        raise LaunchFailed("server command has no -m")
    out = [str(Path(cfg["bindir"] or Path(command[0]).parent) / "llama-perplexity"), "-m", model]
    items = list(command[1:])
    i = 0
    while i < len(items):
        item = items[i]
        if item in PPL_VALUE_OPTIONS and i + 1 < len(items):
            out += [item, items[i + 1]]
            i += 2
            continue
        if item in PPL_SWITCHES:
            out.append(item)
        i += 1
    k = cfg["kld"]
    out += ["-f", k["text"], "-c", str(k["ctx"]), "--chunks", str(k["chunks"]), "-b", str(k["batch"]), "-ub", str(k["batch"]),
            "--kl-divergence-base", str(logits)]
    if compare:
        out.append("--kl-divergence")
    return out


def kld(cfg: Mapping[str, Any], command: Sequence[str], env: Mapping[str, str], logits: Path, log: Path, *, compare: bool) -> float | None:
    """Write the reference logits (compare=False) or return the mean KLD against them."""
    cmd = perplexity_command(cfg, command, logits, compare)
    full_env = {**os.environ, **cfg["env"], **env}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        handle.write(P._command_text(cmd) + "\n")
        handle.flush()
        completed = subprocess.run(cmd, env=full_env, stdout=handle, stderr=subprocess.STDOUT, timeout=4 * 3600, cwd=ROOT)
    if completed.returncode:
        raise LaunchFailed(f"llama-perplexity exit {completed.returncode}; see {log}")
    if not compare:
        return None
    match = re.search(r"Mean\s+KLD:\s+([0-9.eE+-]+)", log.read_text(errors="replace"))
    if not match:
        raise LaunchFailed(f"no KLD in {log}")
    return float(match.group(1))
