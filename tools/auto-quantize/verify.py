"""Checks of an encoded model: it loads with the planned layout, every GPU keeps its free memory, one request works, and the
measured KL against the quality table's teacher is compared with the prediction.

The KL ratio (measured / predicted) is stored in the work directory. `plan --max-kld X` divides the ceiling by it, so
that the ceiling applies to the measured KL and not to the model's estimate."""
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import capacity as cap
import progress
from config import QuantError, write_json
from model import lazy_args

MIB = 1 << 20
RATIO_FILE = "kld-ratio.json"


def server_command(cfg, model, runtime_args, serve, port):
    """The llama-server command line of the planned deployment: placement, context, projector, draft, lazy tables.

    runtime_args is the recipe's --device/-sm/-ts (GPU-only plans); other plans use the first GPU with the routed experts
    on the CPU when RAM is a device. --fit off keeps the server from changing the planned layout."""
    gpus = [name for name in cfg["devices"] if name != "RAM"]
    command = [str(Path(cfg["bindir"]) / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
               "-t", str(cfg["runtime"]["threads"]), "-ngl", "999", "--fit", "off", *cap.serving_args(serve)]
    command += list(runtime_args) if runtime_args else ["--device", gpus[0]]
    if "RAM" in cfg["devices"]:
        command.append("-cmoe")
    command += lazy_args(cfg)
    if serve.get("mmproj"):
        command += ["--mmproj", serve["mmproj"]]
    if serve.get("draft"):
        command += ["--spec-draft-model", serve["draft"], "--spec-type", serve.get("spec_type") or "draft-mtp"]
    return command + list(serve.get("server_args") or [])


def server_env(cfg):
    env = dict(cfg["runtime"]["env"])
    env["LLAMA_MOE_CACHE_MIB"] = str(int(cfg["cache_mib"]))
    return env


def command_text(command, env):
    """The command with its environment, in a form that can be pasted into a shell."""
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
    return (prefix + " " if prefix else "") + " ".join(shlex.quote(str(c)) for c in command)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def keep_free_report(rows, listed):
    """[{device, free, keep_free, ok}] for the GPUs of the capacity table, from the memory ggml reports with the server up.
    listed None: the device list failed, because a GPU had too little free memory for the context of another process
    (about 300 MiB on CUDA); free is then None and every GPU fails, since it is not known which one is full."""
    if listed is None:
        return [{"device": r["device"], "free": None, "keep_free": r["keep_free"], "ok": False}
                for r in rows if r["device"] != "RAM"]
    return [{"device": r["device"], "free": listed[r["device"]]["free"], "keep_free": r["keep_free"],
             "ok": listed[r["device"]]["free"] >= r["keep_free"]} for r in rows if r["device"] != "RAM"]


def ask(url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def run_server_check(cfg, model, runtime_args, serve, rows, directory, timeout=3600):
    """Launch the server, wait until it is ready, read the free memory per device, send one request, stop it.

    rows is the capacity table (keep-free amounts). Returns the report; raises when the server exits or never gets ready."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    port = free_port()
    command = server_command(cfg, model, runtime_args, serve, port)
    env = server_env(cfg)
    print("Server command: " + command_text(command, env), flush=True)
    log = directory / "server.log"
    started = time.monotonic()
    with open(log, "w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, **env})
    try:
        base = f"http://127.0.0.1:{port}"
        with progress.Watcher("verify: server load", lambda: None):
            while True:
                if process.poll() is not None:
                    raise QuantError(f"llama-server exited with {process.returncode} while loading; see {log}")
                if time.monotonic() - started > timeout:
                    raise QuantError(f"llama-server was not ready after {timeout} s; see {log}")
                try:
                    if ask(base + "/health", timeout=5).get("status") == "ok":
                        break
                except (urllib.error.URLError, OSError, ValueError):
                    pass
                time.sleep(2)
        load_seconds = time.monotonic() - started
        answer = ask(base + "/completion", {"prompt": "The three primary colors are", "n_predict": 24, "temperature": 0})
        text = answer.get("content", "")
        # after the request, so that buffers allocated on first use are counted
        try:
            listed = cap.list_devices(cfg["bindir"], cfg["runtime"]["env"])
        except QuantError as exc:
            print(f"With the server up, {exc}", flush=True)
            listed = None
        memory = keep_free_report(rows, listed)
        timings = answer.get("timings", {})
    finally:
        process.terminate()
        try:
            process.wait(60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if not text.strip():
        raise QuantError(f"the server answered with no text; see {log}")
    return {"command": command_text(command, env), "load_seconds": load_seconds, "answer": text, "memory": memory,
            "decode_t_s": timings.get("predicted_per_second"), "log": str(log)}


def memory_text(memory):
    lines = [f"{'device':<8s} {'free MiB':>9s} {'keep-free MiB':>14s}  status"]
    for m in memory:
        free = f"{m['free'] // MIB:>9d}" if m["free"] is not None else f"{'?':>9s}"
        lines.append(f"{m['device']:<8s} {free} {m['keep_free'] // MIB:>14d}  "
                     + ("keeps its keep-free" if m["ok"] else "BELOW keep-free" if m["free"] is not None
                        else "unknown: no room for another process on some GPU"))
    return "\n".join(lines)


def kl_ratio(measured, predicted):
    if predicted <= 0 or measured <= 0:
        raise QuantError(f"cannot compare KL: measured {measured}, predicted {predicted}")
    return measured / predicted


def store_ratio(work, record):
    write_json(Path(work) / RATIO_FILE, record)


def load_ratio(work):
    """The stored measured/predicted KL ratio of the last verified encode, or None."""
    path = Path(work) / RATIO_FILE
    if not path.exists():
        return None
    ratio = json.loads(path.read_text()).get("ratio")
    return ratio if isinstance(ratio, (int, float)) and ratio > 0 else None


def corrected_ceiling(ceiling, ratio):
    """The model-estimate ceiling that gives `ceiling` on the measured KL when the model is off by `ratio`."""
    return ceiling / ratio if ratio else ceiling


def ensure_teacher(cfg, table, scratch):
    """The table's teacher logits file; recomputed into the scratch directory when it was deleted.

    Returns a copy of the table with the path of the file in use."""
    teacher = table["teacher"]
    if Path(teacher["path"]).exists():
        return table
    from calibrate import teacher_logits
    c = dict(cfg["calibration"], text=teacher["text"]["path"], chunks=teacher["chunks"], context=teacher["context"], teacher=None)
    path = teacher_logits(cfg, c, Path(scratch), 7200)
    return {**table, "teacher": {**teacher, "path": str(path)}}
