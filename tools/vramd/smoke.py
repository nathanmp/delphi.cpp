#!/usr/bin/env python3
"""Functional smoke test of delphi-vramd with two llama-servers sharing two GPUs.

Starts the daemon and servers A (home GPU 0) and B (home GPU 1), then: a prompt on A; a prompt on B while A decodes (B revokes
A's cache on B's GPU); idle release of compute buffers and KV; a follow-up on A that restores its KV from the host prompt
cache; B killed (its leases return). Prints the daemon state after each step and a summary. Not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def status(sock: str) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock)
    s.sendall(b'{"op":"status"}\n')
    buf = b""
    while not buf.endswith(b"\n"):
        buf += s.recv(65536)
    s.close()
    return json.loads(buf)


def show(sock: str, title: str) -> dict:
    st = status(sock)
    print(f"--- {title}")
    for g, v in st["gpus"].items():
        print(f"  {g}: used {v['used'] / 2**20:.0f} MiB, free {v['free'] / 2**20:.0f} MiB")
    for s in st["servers"]:
        if s["home"]:
            leases = [l for l in st["leases"] if l["server"] == s["id"]]
            desc = ", ".join(f"{l['purpose']} {l['bytes'] / 2**20:.0f}@{l['gpu']}" + (" (revoking)" if l["revoking"] else "") for l in leases)
            print(f"  server {s['id']} {s['model']} home {s['home']} {s['phase']} {s['phase_s']} s: {desc}")
    if st["pending"]:
        print(f"  pending: {st['pending']}")
    sys.stdout.flush()
    return st


def complete(port: int, prompt: str, n: int) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": True, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def wait_health(port: int, proc: subprocess.Popen, timeout: float = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server on port {port} exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"server on port {port} did not start")


def same_prefix(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", required=True, help="directory with llama-server")
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--gpus", default="ROCm0,ROCm1", help="two llama device names: A's home first")
    ap.add_argument("--capacity", default="6G", help="daemon capacity per GPU")
    ap.add_argument("--prompt-file", default="/dev/shm/doc.txt")
    ap.add_argument("--prompt-chars", type=int, default=6000)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--long-predict", type=int, default=256, help="A's generation while B starts")
    ap.add_argument("--ports", default="18291,18292")
    ap.add_argument("--out", default="/tmp/vramd-smoke")
    ap.add_argument("--server-args", default="--numa tensors -ngl 999 -cmoe -c 8192 -b 2048 -ub 2048 -np 1 -t 16 -fa on --cache-ram 4096")
    ap.add_argument("--env", action="append", default=[], help="extra NAME=VALUE for both servers")
    ap.add_argument("--idle-compute", type=float, default=5)
    ap.add_argument("--idle-kv", type=float, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    g0, g1 = args.gpus.split(",")
    pa, pb = (int(p) for p in args.ports.split(","))
    sock = os.path.join(args.out, "vramd.sock")
    prompt = open(args.prompt_file, errors="ignore").read()[: args.prompt_chars]
    procs = []

    daemon = subprocess.Popen([sys.executable, os.path.join(HERE, "delphi-vramd.py"), "--socket", sock, "--gpu", f"{g0}={args.capacity}",
                               "--gpu", f"{g1}={args.capacity}", "--grace", "30", "--log", os.path.join(args.out, "vramd.log")])
    procs.append(daemon)
    for _ in range(50):
        if os.path.exists(sock):
            break
        time.sleep(0.1)

    env = dict(os.environ)
    env.update({"LD_LIBRARY_PATH": f"{args.bin}:/opt/rocm/lib", "LLAMA_VRAMD": sock, "LLAMA_MOE_CACHE_SHARED": "1",
                "GGML_SCHED_STAGE": "1", "GGML_SCHED_STAGE_SPREAD": "1", "GGML_SCHED_STAGE_SLOTS": "6",
                "LLAMA_SHARE_IDLE_COMPUTE_S": str(args.idle_compute), "LLAMA_SHARE_IDLE_KV_S": str(args.idle_kv)})
    for item in args.env:
        k, _, v = item.partition("=")
        env[k] = v

    def start(tag: str, model: str, devs: str, port: int) -> subprocess.Popen:
        cmd = [os.path.join(args.bin, "llama-server"), "-m", model, "--device", devs, "-ts", "1,0", "--port", str(port), *args.server_args.split()]
        log = open(os.path.join(args.out, f"server-{tag}.log"), "w")
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        return p

    results = {}
    try:
        a = start("a", args.model_a, f"{g0},{g1}", pa)
        wait_health(pa, a)
        b = start("b", args.model_b, f"{g1},{g0}", pb)
        wait_health(pb, b)
        show(sock, "both registered")

        r = complete(pa, prompt, args.n_predict)
        results["a1"] = r
        print(f"A: {r['timings']['prompt_n']} prompt tokens {r['timings']['prompt_per_second']:.0f} t/s, "
              f"{r['timings']['predicted_per_second']:.1f} t/s decode")
        show(sock, "after A's first request")

        # B's prompt while A decodes
        ta = {}
        th = threading.Thread(target=lambda: ta.update(r=complete(pa, prompt + " Continue.", args.long_predict)))
        th.start()
        time.sleep(3)
        show(sock, "A decoding")
        t0 = time.time()
        rb = complete(pb, prompt, args.n_predict)
        results["b1"] = rb
        print(f"B: {rb['timings']['prompt_n']} prompt tokens {rb['timings']['prompt_per_second']:.0f} t/s, "
              f"{rb['timings']['predicted_per_second']:.1f} t/s decode, {time.time() - t0:.1f} s total")
        show(sock, "after B's request")
        th.join()
        results["a2"] = ta["r"]
        print(f"A (overlapped): {ta['r']['timings']['predicted_per_second']:.1f} t/s decode")
        print(f"A vs B first answer: {same_prefix(results['a1']['content'], rb['content'])} of {len(rb['content'])} chars equal")

        time.sleep(max(args.idle_compute, args.idle_kv) + 3)
        show(sock, "both idle past the timers")

        r = complete(pa, prompt, args.n_predict)
        results["a3"] = r
        print(f"A again: {r['timings']['prompt_n']} prompt tokens processed ({r['tokens_cached']} cached); "
              f"{same_prefix(results['a1']['content'], r['content'])} of {len(r['content'])} chars as the first answer")
        show(sock, "after A's follow-up")

        b.send_signal(signal.SIGKILL)
        b.wait()
        time.sleep(1)
        show(sock, "B killed")
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
        json.dump({k: {"content": v["content"], "timings": v["timings"]} for k, v in results.items()},
                  open(os.path.join(args.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
