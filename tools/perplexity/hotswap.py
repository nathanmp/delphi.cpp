#!/usr/bin/env python3
"""Drive a persistent llama-perplexity process with isolated partial GGUF patches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


class HotswapSession:
    def __init__(self, command, output, timeout=600):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.control = self.output / "control"
        self.status = self.output / "status"
        self.log = self.output / "session.log"
        if self.log.exists():
            raise FileExistsError(f"session log already exists: {self.log}; choose a new output directory")
        self.timeout = timeout
        self.sequence = 0
        self.iteration = 0
        self.process = None
        self.command = list(command)
        for path in (self.control, self.status):
            path.unlink(missing_ok=True)

    def state(self):
        try:
            fields = self.status.read_text().split()
            return fields[0], int(fields[1]), int(fields[2])
        except (OSError, ValueError, IndexError):
            return None

    def wait(self, sequence, initial=False, exiting=False):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            state = self.state()
            if state and state[2] == sequence:
                if state[0] == "exiting" and exiting:
                    return state
                if state[0] == "done" and (initial or state[1] > self.iteration):
                    self.iteration = state[1]
                    return state
                if state[0] in ("reload_failed", "command_failed", "error"):
                    raise RuntimeError(f"hotswap {state[0]}: see {self.log}")
            if self.process.poll() is not None:
                raise RuntimeError(f"perplexity exited with {self.process.returncode}: see {self.log}")
            time.sleep(0.05)
        raise TimeoutError(f"hotswap timed out at {state}: see {self.log}")

    def start(self):
        self.stream = self.log.open("wb", buffering=0)
        env = os.environ | {
            "LLAMA_HOTSWAP_ENABLED": "1",
            "LLAMA_HOTSWAP_CONTROL_FILE": str(self.control.resolve()),
            "LLAMA_HOTSWAP_STATUS_FILE": str(self.status.resolve()),
        }
        self.process = subprocess.Popen(self.command, env=env, stdout=self.stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            self.wait(0, initial=True)
        except Exception:
            self.close()
            raise
        return {"state": self.state(), "seconds": time.monotonic()-started, "log": self.log.read_text(errors="replace")}

    def request(self, action, patch=None):
        if action not in ("reload", "restore", "compute", "exit"):
            raise ValueError(action)
        if patch is not None:
            patch = Path(patch).resolve()
            if action != "reload" or not patch.is_file() or "\n" in str(patch):
                raise ValueError("reload requires an existing patch path without newlines")
        self.sequence += 1
        line = f"{action} {self.sequence}" + (f" {patch}" if patch else "") + "\n"
        offset = self.log.stat().st_size
        temporary = self.control.with_suffix(".tmp")
        temporary.write_text(line)
        temporary.replace(self.control)
        started = time.monotonic()
        self.wait(self.sequence, exiting=action == "exit")
        with self.log.open("rb") as stream:
            stream.seek(offset)
            text = stream.read().decode(errors="replace")
        return {"state": self.state(), "seconds": time.monotonic()-started, "log": text}

    def close(self):
        if self.process and self.process.poll() is None:
            try:
                self.request("exit")
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if hasattr(self, "stream"):
            self.stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- llama-perplexity ARGS")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a llama-perplexity command is required after --")
    session = HotswapSession(command, args.output, args.timeout)
    manifest = {"command": command, "jobs": []}
    try:
        first = session.start()
        (args.output/"baseline.log").write_text(first.pop("log"))
        manifest["baseline"] = first
        for index, patch in enumerate(args.patch, 1):
            job = {"index": index, "patch": str(patch.resolve()),
                   "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
            try:
                result = session.request("reload", patch)
                (args.output/f"job-{index}.log").write_text(result.pop("log"))
                job.update(result)
            except Exception as error:
                job["error"] = str(error)
                (args.output/f"job-{index}.failed.json").write_text(json.dumps(job, indent=2)+"\n")
                raise
            finally:
                manifest["jobs"].append(job)
        if args.patch:
            restored = session.request("restore")
            (args.output/"restore.log").write_text(restored.pop("log"))
            manifest["restore"] = restored
    finally:
        session.close()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"run.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
