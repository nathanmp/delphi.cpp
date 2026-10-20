"""Device capacities and reserves from measurements instead of arithmetic.

Capacity of a GPU = the free memory that ggml reports when the tool starts (`llama-bench --list-devices`, so it works on
any OS and backend) minus a keep-free amount. Reserves (KV cache, compute buffers, multimodal projector, draft model)
come from `llama-fit-params --fit-print on`, which sizes a model and context without loading the weights."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from config import QuantError, capacity
from model import lazy_args
from progress import size_text
from space import ram_available

MIB = 1 << 20
KEEP_FREE_DEFAULT = 512 * MIB
DEVICE_NAME = r"RAM|(?:CUDA|ROCm)[0-9]+"


def parse_devices(text):
    """[(name, explicit capacity or None)] from 'ROCm3,ROCm0=24GiB,RAM=200GiB'. An explicit size wins over the measurement."""
    result = []
    for item in text.split(","):
        name, sep, amount = item.strip().partition("=")
        if not re.fullmatch(DEVICE_NAME, name) or name in [n for n, _ in result]:
            raise QuantError(f"--devices must list distinct devices such as ROCm3,ROCm0=24GiB,RAM=200GiB: {item!r}")
        result.append((name, capacity(amount) if sep else None))
    if not any(name != "RAM" for name, _ in result):
        raise QuantError("--devices needs at least one GPU")
    return result


def parse_keep_free(text, names):
    """{device: bytes} from '1GiB' (every device) or 'ROCm2=2GiB,ROCm0=1GiB' (others keep the default 512 MiB)."""
    keep = {name: KEEP_FREE_DEFAULT for name in names}
    if text is None:
        return keep
    if "=" not in text:
        return {name: capacity(text) for name in names}
    for item in text.split(","):
        name, sep, amount = item.strip().partition("=")
        if not sep or name not in keep:
            raise QuantError(f"--keep-free names a device that is not in --devices, or is not DEVICE=SIZE: {item!r}")
        keep[name] = capacity(amount)
    return keep


def parse_listed(text):
    """{device: {"description", "total", "free"}} in bytes from the lines of `llama-bench --list-devices`:
    'ROCm0: AMD Radeon AI PRO R9700 (32624 MiB, 32526 MiB free)'."""
    found = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\S+):\s*(.*?)\s*\((\d+) MiB, (\d+) MiB free\)\s*$", line)
        if m:
            found[m[1]] = {"description": m[2], "total": int(m[3]) * MIB, "free": int(m[4]) * MIB}
    return found


def list_devices(bindir, env=None):
    """Memory of every device as ggml sees it right now."""
    command = [str(Path(bindir) / "llama-bench"), "--list-devices"]
    try:
        text = subprocess.check_output(command, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}).decode()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QuantError(f"cannot list devices with {command[0]}: {exc}") from exc
    return parse_listed(text)


def total_ram():
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def capacity_rows(devices, listed, keep_free, mode="free", ram=None):
    """One row per device: device, total, free, keep_free, capacity and where the capacity came from.

    devices is parse_devices() output; ram is (total, available) of the host or None. mode 'total' plans on the total
    memory (while other jobs hold the devices) instead of the free memory."""
    rows = []
    for name, explicit in devices:
        if name == "RAM":
            total, free = (ram or (None, None))
        else:
            if name not in listed:
                raise QuantError(f"device {name} is not in llama-bench --list-devices: {sorted(listed)}")
            total, free = listed[name]["total"], listed[name]["free"]
        base = total if mode == "total" else free
        if explicit is not None:
            amount, source = explicit, "explicit"
        elif base is None:
            raise QuantError(f"cannot measure {name} here; give it a size, for example {name}=200GiB")
        else:
            amount, source = max(base - keep_free[name], 0), f"{mode} - keep-free"
        rows.append({"device": name, "total": total, "free": free, "keep_free": keep_free[name], "capacity": amount, "source": source})
    return rows


def table_text(rows):
    def cell(n):
        return "-" if n is None else size_text(n)
    lines = [f"{'device':<8s} {'total':>10s} {'free':>10s} {'keep-free':>10s} {'capacity':>10s}  source"]
    for r in rows:
        lines.append(f"{r['device']:<8s} {cell(r['total']):>10s} {cell(r['free']):>10s} {cell(r['keep_free']):>10s} "
                     f"{cell(r['capacity']):>10s}  {r['source']}")
    return "\n".join(lines)


def host_memory():
    return (total_ram(), ram_available())


def initial_split(n_layer, capacities):
    """Layers per GPU, in proportion to the capacities, as -ts counts (n_layer + 1 slots: the last one is the output)."""
    total = n_layer + 1
    whole = sum(capacities)
    counts = [int(total * c / whole) for c in capacities]
    order = sorted(range(len(counts)), key=lambda i: -(total * capacities[i] / whole - counts[i]))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def serving_args(serve):
    """Options that change memory and that fit-params and llama-server take alike."""
    args = ["-c", str(serve["ctx"]), "-ub", str(serve["ub"]), "-np", str(serve["parallel"])]
    for flag, key in (("-ctk", "ctk"), ("-ctv", "ctv")):
        if serve.get(key):
            args += [flag, serve[key]]
    return args


def parse_fit_print(text):
    """({device: (model, context, compute)}, {device: mmproj}) in bytes from `llama-fit-params --fit-print on`."""
    devices, mmproj = {}, {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] != "mmproj" and all(p.isdigit() for p in parts[1:]):
            devices[parts[0]] = tuple(int(p) * MIB for p in parts[1:])
        elif len(parts) == 3 and parts[0] == "mmproj" and parts[2].isdigit():
            mmproj[parts[1]] = int(parts[2]) * MIB
    return devices, mmproj


def measure_reserves(cfg, serve, gpus, split, ram_routed):
    """Per-device reserve of the planned layout from llama-fit-params, without loading the weights.

    Returns {"command", "per_device": {device: {"context", "compute", "mmproj", "draft", "total"}}}. The multimodal
    projector goes to the first device, as the server does it. A draft file counts its size on its device (default the
    first); fit-params cannot size a draft model alone, so its KV and compute buffers are not included."""
    command = [str(Path(cfg["bindir"]) / "llama-fit-params"), "-m", cfg["model"], "-fitp", "on", "--device", ",".join(gpus),
               "-ngl", "999", *serving_args(serve), *lazy_args(cfg)]
    if len(gpus) > 1:
        command += ["-sm", "layer", "-ts", ",".join(map(str, split))]
    if ram_routed:
        command.append("-cmoe")
    if serve.get("mmproj"):
        command += ["--mmproj", serve["mmproj"]]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, **cfg["runtime"]["env"]})
    devices, mmproj = parse_fit_print(completed.stdout.decode())
    if completed.returncode or not set(gpus) <= set(devices):
        tail = completed.stderr.decode().strip().splitlines()[-3:]
        raise QuantError(f"llama-fit-params failed ({completed.returncode}) on {cfg['model']}: " + " | ".join(tail))
    result = {name: {"context": devices[name][1], "compute": devices[name][2], "mmproj": mmproj.get(name, 0), "draft": 0}
              for name in gpus}
    if serve.get("mmproj") and not any(mmproj.get(name) for name in gpus):
        raise QuantError("llama-fit-params printed no projector memory; build it from this tree (--mmproj support)")
    if serve.get("draft"):
        result[serve.get("draft_device") or gpus[0]]["draft"] = os.path.getsize(serve["draft"])
    for entry in result.values():
        entry["total"] = entry["context"] + entry["compute"] + entry["mmproj"] + entry["draft"]
    return {"command": command, "per_device": result}


def reserves_text(measured, split=None, gpus=None):
    lines = [f"{'device':<8s} {'layers':>6s} {'KV+state':>10s} {'compute':>10s} {'mmproj':>10s} {'draft':>10s} {'reserve':>10s}"]
    for i, (name, e) in enumerate(measured["per_device"].items()):
        lines.append(f"{name:<8s} {split[i] if split else '-':>6} {size_text(e['context']):>10s} {size_text(e['compute']):>10s} "
                     f"{size_text(e['mmproj']):>10s} {size_text(e['draft']):>10s} {size_text(e['total']):>10s}")
    return "\n".join(lines)
