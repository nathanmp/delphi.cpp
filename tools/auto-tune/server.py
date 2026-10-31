"""llama-server process and HTTP helpers for the tuner: launch with --tune-endpoints, context re-creation, prompt requests."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import cmdline as C


class ServerError(RuntimeError):
    pass


class Rejected(ServerError):
    """The server refused a change (for example out of memory); the old settings are still active."""


class Server:
    def __init__(self, command: Sequence[str], env: Mapping[str, str], port: int, log_path: Path,
                 load_timeout: float = 3600.0) -> None:
        cmd = C.set_option(C.set_option(command, "--port", str(port)), "--host", "127.0.0.1")
        if "--tune-endpoints" not in cmd:
            cmd = (*cmd, "--tune-endpoints")
        self.command = cmd
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "ab")
        start = time.time()
        self.proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        while True:
            if self.proc.poll() is not None:
                raise ServerError(f"server exited with code {self.proc.returncode} during load (log: {log_path})")
            try:
                if self._get("/health", timeout=5).get("status") == "ok":
                    break
            except (OSError, ServerError, ValueError):
                pass
            if time.time() - start > load_timeout:
                self.stop()
                raise ServerError(f"server not ready after {load_timeout:.0f} s (log: {log_path})")
            time.sleep(2)
        self.load_s = time.time() - start

    def _request(self, path: str, body: Any | None, timeout: float) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            if e.code in (400, 500):
                raise Rejected(f"{path}: HTTP {e.code}: {text[:500]}") from None
            raise ServerError(f"{path}: HTTP {e.code}: {text[:500]}") from None
        except OSError as e:  # connection refused or reset: the server is gone
            raise ServerError(f"{path}: {e} (log: {self.log_path})") from None

    def _get(self, path: str, timeout: float = 60) -> Any:
        return self._request(path, None, timeout)

    def get(self, path: str, timeout: float = 60) -> Any:
        return self._get(path, timeout)

    def post(self, path: str, body: Any, timeout: float = 1800) -> Any:
        if self.proc.poll() is not None:
            raise ServerError(f"server exited with code {self.proc.returncode} (log: {self.log_path})")
        return self._request(path, body, timeout)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def recreate(self, settings: Mapping[str, Any]) -> float:
        """POST /tune/context; returns the seconds it took. Rejected keeps the old context."""
        start = time.time()
        res = self.post("/tune/context", dict(settings))
        if not res.get("ok", True):
            raise Rejected(f"/tune/context: {res}")
        return time.time() - start

    def tokenize(self, text: str) -> list[int]:
        return self.post("/tokenize", {"content": text, "add_special": False})["tokens"]

    def prefill(self, tokens: Sequence[int]) -> float:
        """Prompt tokens per second of one request of exactly these tokens (one generated token, no prompt cache)."""
        res = self.post("/completion", {"prompt": list(tokens), "n_predict": 1, "cache_prompt": False,
                                        "temperature": 0})
        t = res["timings"]
        return t["prompt_n"] / (t["prompt_ms"] / 1000.0)

    def chat(self, prompt: str, max_tokens: int, sampling: Mapping[str, Any]) -> dict[str, float]:
        """One chat request; returns generated tokens, generation ms and draft counts from the server timings."""
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False,
                "cache_prompt": False, **sampling}
        t = self.post("/v1/chat/completions", body)["timings"]
        return {"n": t["predicted_n"], "ms": t["predicted_ms"], "draft_n": t.get("draft_n", 0),
                "draft_accepted": t.get("draft_n_accepted", 0)}

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.log.close()
