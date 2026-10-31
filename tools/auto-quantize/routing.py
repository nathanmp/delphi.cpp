"""Expert-cache hit curve of a model: a routing trace of a real llama-server run, and an offline replay of the cache policy.

The speed model needs the share of routed-expert selections that run on the GPU for a cache of N slots. A uniform prior
(N / all experts) is far too low, because the router prefers some experts (DeepSeek V4.1 Flash: about 3 % of the experts in
the cache give a hit share of about 25 %, and 953 MXFP4 slots about 45-50 %). So the tool records the router's top-k ids of
a few thousand generated tokens (`LLAMA_MOE_ROUTING_LOG`, the same server layout as the deployment, with the draft model) and
replays the policy of `src/llama-expert-cache.cpp` on them for a grid of slot counts:

* every row of every batch goes to a window of the last `window` rows per layer (prompts included);
* a prompt batch (more than 16 rows) asks for a refresh, which runs before the next small batch; it keeps the top `slots`
  (layer, expert) pairs by window count, residents with the bonus `1 + hysteresis`, ties by (layer, expert);
* every `every` small batches a decode-time refresh swaps in at most `max_async` experts (the best experts that are not
  resident, into free slots and then into the weakest residents that left the top set); the swap takes effect `delay`
  small batches later, and a prompt refresh drops a swap that has not taken effect;
* a small batch (at most 16 rows) counts its selections as hits when the expert is resident at its start.

The result is `{points: [[slots, hit fraction], ...]}`, stored in `routing-curve.json` of the prepared directory and used by
`search.memory` in place of the uniform prior. The replay is checked against the server's own `previous decode hits` lines
(`compare`)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from config import QuantError, digest, file_identity, write_json

SCHEMA = "delphi.routing-curve/1"
MAX_ROWS = 16          # llama_expert_cache::max_rows: larger batches are prompts
WINDOW = 256           # LLAMA_MOE_CACHE_WINDOW
EVERY = 8              # LLAMA_MOE_CACHE_EVERY
HYSTERESIS = 1.0       # LLAMA_MOE_CACHE_HYSTERESIS
MAX_ASYNC = 64         # LLAMA_MOE_CACHE_ASYNC_EXPERTS
DELAY = 1              # small batches between queuing a swap and its taking effect
GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)


# ---------------------------------------------------------------------------------------------------------------- trace

def read_trace(path):
    """(layers, batches): the sorted layer ids of the trace and the list of batches, each an int16 array
    [n_layers, rows, top_k] in the order the server ran them. A record is int32 layer, int32 rows, int32 top_k, then
    rows * top_k int32 ids; the records of one batch come in rising layer order."""
    data = np.fromfile(path, dtype=np.int32)
    records, at = [], 0
    while at < len(data):
        if at + 3 > len(data):
            raise QuantError(f"{path}: truncated record header")
        layer, rows, k = (int(x) for x in data[at:at + 3])
        end = at + 3 + rows * k
        if rows <= 0 or k <= 0 or layer < 0 or end > len(data):
            raise QuantError(f"{path}: invalid record at int {at}")
        records.append((layer, rows, k, at + 3))
        at = end
    if not records:
        raise QuantError(f"{path}: no routing records")
    # The graph of a batch that runs the cache logs a layer's top-k more than once (the same tensor again, and a reshaped
    # copy): keep the first record of each run of one layer.
    records = [r for i, r in enumerate(records) if i == 0 or r[0] != records[i - 1][0]]
    layers = sorted({r[0] for r in records})
    index = {layer: i for i, layer in enumerate(layers)}
    batches, current, last = [], [], -1
    for rec in records:
        if rec[0] <= last and current:
            batches.append(current)
            current = []
        current.append(rec)
        last = rec[0]
    batches.append(current)
    result = []
    for batch in batches:
        k = batch[0][2]
        if len(batch) != len(layers) or any(r[2] != k for r in batch):
            # a batch with a missing layer (the server stopped in it) cannot be replayed
            continue
        # The layers of a large prompt batch can run fewer rows than the batch: DeepSeek V4.1 drops the rows that only
        # the first layers need (n_layer_prefill). Such a batch is a list of [rows, top_k] arrays, one per layer.
        parts = [None] * len(layers)
        for layer, rows, _, offset in batch:
            parts[index[layer]] = data[offset:offset + rows * k].reshape(rows, k).astype(np.int16)
        if len({p.shape[0] for p in parts}) == 1:
            result.append(np.stack(parts))
        else:
            result.append(parts)
    if not result:
        raise QuantError(f"{path}: no complete batch")
    return layers, result


def batch_rows(batch):
    """Rows of a batch (the largest of its layers when the layers differ)."""
    return batch.shape[1] if isinstance(batch, np.ndarray) else max(p.shape[0] for p in batch)


def batch_top_k(batch):
    return int(batch.shape[2]) if isinstance(batch, np.ndarray) else int(batch[0].shape[1])


def trace_summary(layers, batches):
    rows = [batch_rows(b) for b in batches]
    small = sum(r for r in rows if r <= MAX_ROWS)
    return {"layers": len(layers), "batches": len(batches), "decode_rows": small, "prompt_rows": sum(rows) - small,
            "prompts": sum(1 for r in rows if r > MAX_ROWS), "top_k": batch_top_k(batches[0])}


# ---------------------------------------------------------------------------------------------------------------- replay

class Replay:
    """The expert cache of one tier group with `slots` slots over layers x experts."""

    def __init__(self, n_layers, n_expert, top_k, slots, window=WINDOW, every=EVERY, hysteresis=HYSTERESIS,
                 max_async=MAX_ASYNC, delay=DELAY):
        self.L, self.E, self.k, self.slots = n_layers, n_expert, top_k, int(slots)
        self.window, self.every, self.hysteresis, self.max_async, self.delay = window, every, hysteresis, max_async, delay
        self.count = np.zeros((n_layers, n_expert), dtype=np.int32)
        self.ring = np.zeros((n_layers, max(window, 1), top_k), dtype=np.int16)
        self.rows = np.zeros(n_layers, dtype=np.int64)  # rows in the window of each layer
        self.head = np.zeros(n_layers, dtype=np.int64)
        self.resident = np.zeros((n_layers, n_expert), dtype=bool)
        self.pending = False
        self.n_small = 0
        self.tick_no = 0
        self.queued = None       # (tick number it takes effect at, add flat ids, evict flat ids)
        self.refreshes = 0
        self.segments = [[0, 0]]  # [hits, selections] per span between prompt refreshes

    def record(self, batch):
        """Rows enter the window of each layer one by one; a full window drops its oldest row (llama_expert_cache::record).
        batch is an array [layers, rows, top_k] or a list of [rows, top_k] arrays, one per layer."""
        if isinstance(batch, np.ndarray):
            self._record(np.arange(self.L), batch)
            return
        by_rows = {}
        for layer, part in enumerate(batch):
            by_rows.setdefault(part.shape[0], []).append(layer)
        for rows, members in by_rows.items():
            self._record(np.array(members), np.stack([batch[m] for m in members]))

    def _record(self, idx, ids):
        """The layers idx (an index array) take ids [len(idx), rows, top_k]."""
        rows = ids.shape[1]
        if rows >= self.window:
            # only the last `window` rows stay, oldest first from position 0
            last = ids[:, rows - self.window:]
            self.count[idx] = 0
            np.add.at(self.count, (np.broadcast_to(idx[:, None, None], last.shape), last.astype(np.int64)), 1)
            self.ring[idx, :self.window] = last
            self.rows[idx] = self.window
            self.head[idx] = 0
            return
        for r in range(rows):
            full = self.rows[idx] == self.window
            if full.any():
                old = self.ring[idx[full], self.head[idx[full]]].astype(np.int64)
                np.subtract.at(self.count, (np.broadcast_to(idx[full][:, None], old.shape), old), 1)
            self.rows[idx] = np.minimum(self.rows[idx] + 1, self.window)
            new = ids[:, r].astype(np.int64)
            np.add.at(self.count, (np.broadcast_to(idx[:, None], new.shape), new), 1)
            self.ring[idx, self.head[idx]] = ids[:, r]
            self.head[idx] = (self.head[idx] + 1) % self.window

    def ranked(self, bonus=True):
        """Flat ids of the (layer, expert) pairs with a positive score, best first (ties: lowest layer, then expert)."""
        score = self.count.astype(np.float64)
        if bonus:
            score = np.where(self.resident, score * (1.0 + self.hysteresis), score)
        flat = score.reshape(-1)
        order = np.argsort(-flat, kind="stable")
        return order[flat[order] > 0], flat

    def refresh(self):
        """After a prompt: the top set becomes resident, everything else leaves."""
        self.queued = None  # cancel_async() drops a swap that has not taken effect
        keep, _ = self.ranked()
        self.resident[:] = False
        self.resident.reshape(-1)[keep[:self.slots]] = True
        self.refreshes += 1
        self.segments.append([0, 0])

    def tick(self):
        self.tick_no += 1
        if self.queued is not None and self.tick_no >= self.queued[0]:
            _, add, evict = self.queued
            flat = self.resident.reshape(-1)
            flat[evict] = False
            flat[add] = True
            self.queued = None
        self.n_small += 1
        if self.n_small % self.every != 0 or self.queued is not None:
            return
        keep, _ = self.ranked()
        keep = keep[:self.slots]
        flat = self.resident.reshape(-1)
        add = keep[~flat[keep]]
        if len(add) == 0:
            return
        free = self.slots - int(flat.sum())
        count = self.count.reshape(-1)
        leaving = np.flatnonzero(flat)
        leaving = leaving[~np.isin(leaving, keep)]
        leaving = leaving[np.lexsort((leaving, count[leaving]))]  # weakest first, ties by layer and expert
        jobs = min(len(add), self.max_async, max(free, 0) + len(leaving))
        if jobs <= 0:
            return
        evict = leaving[:max(0, jobs - max(free, 0))]
        self.queued = (self.tick_no + self.delay, add[:jobs], evict)

    def run(self, batches):
        """Replay the batches; returns {hit_fraction, steady_hit_fraction, segments}."""
        hits = total = 0
        layer_base = np.arange(self.L)[:, None, None] * self.E
        flat_of = lambda x: x.astype(np.int64)
        for ids in batches:
            rows = batch_rows(ids)
            if rows > MAX_ROWS:
                self.pending = True
            elif self.pending:
                self.pending = False
                self.refresh()
            else:
                self.tick()
            if isinstance(ids, np.ndarray):
                parts = [(slice(None), ids)] if ids.shape[1] <= MAX_ROWS else []
            else:
                # the server counts the hits of a layer that runs a small batch, whatever the batch around it
                parts = [(layer, p[None]) for layer, p in enumerate(ids) if p.shape[0] <= MAX_ROWS]
            for layer, part in parts:
                if layer == slice(None):
                    got = int(self.resident.reshape(-1)[(layer_base + flat_of(part))].sum()) if self.slots > 0 else 0
                else:
                    got = int(self.resident[layer][flat_of(part)].sum()) if self.slots > 0 else 0
                self.segments[-1][0] += got
                self.segments[-1][1] += part.size
            self.record(ids)
        for h, n in self.segments:
            hits, total = hits + h, total + n
        # the span before the first refresh runs with an empty cache: leave it out of the steady-state share
        steady = self.segments[1:]
        steady_hits, steady_total = sum(s[0] for s in steady), sum(s[1] for s in steady)
        return {"hit_fraction": hits / total if total else 0.0,
                "steady_hit_fraction": steady_hits / steady_total if steady_total else 0.0,
                "segments": [list(s) for s in self.segments]}


def replay(layers, batches, n_expert, slots, **options):
    return Replay(len(layers), n_expert, batch_top_k(batches[0]), slots, **options).run(batches)


def curve(layers, batches, n_expert, grid=GRID, **options):
    """[[slots, hit fraction]] for the grid (fractions of all experts), from the steady-state share, monotone."""
    total = len(layers) * n_expert
    points, best = [[0, 0.0]], 0.0
    for fraction in grid:
        slots = max(1, int(round(fraction * total)))
        value = replay(layers, batches, n_expert, slots, **options)["steady_hit_fraction"]
        best = max(best, value)
        points.append([slots, round(best, 5)])
    points.append([total, 1.0])
    return points


def load_curve(path):
    """The routing curve file written by `routing_curve`, checked."""
    curve = json.loads(Path(path).read_text())
    if curve.get("schema") != SCHEMA:
        raise QuantError(f"{path}: not a routing curve ({SCHEMA})")
    points = curve.get("points")
    if (not points or any(len(p) != 2 or not all(isinstance(x, (int, float)) for x in p) for p in points)
            or any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(points, points[1:])) or not 0 <= points[0][1] <= points[-1][1] <= 1):
        raise QuantError(f"{path}: the points of the routing curve must rise in slots and hit fraction")
    curve["path"] = str(path)
    return curve


def hit_at(points, slots):
    """Hit fraction of a cache with `slots` slots: linear between the points of the curve."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(np.interp(slots, xs, ys))


# ----------------------------------------------------------------------------------------------------------- validation

REFRESH_LINE = re.compile(r"expert-cache: refresh (\d+): (\d+) resident, .*previous decode hits ([0-9.]+)% selections")
FINAL_LINE = re.compile(r"expert-cache: \d+ refreshes, .*decode hits since last refresh: ([0-9.]+)% selections")


def server_segments(log_text):
    """Hit percent of every span of the server log, in order. The server prints the span before refresh n at refresh n, and
    the last span when it exits; a startup refresh that has no rows yet (0 resident) closes an empty span and is dropped."""
    result = [(int(m[2]), float(m[3])) for m in REFRESH_LINE.finditer(log_text)]
    if result and result[0][0] == 0:
        result = result[1:]
    values = [percent for _, percent in result]
    final = FINAL_LINE.search(log_text)
    if final:
        values.append(float(final[1]))
    return values


def compare(log_text, layers, batches, n_expert, slots, **options):
    """The replay of a trace against the `previous decode hits` lines of the server run that wrote it.

    Returns per-span server and replay percentages, and the selection-weighted means over all compared spans (the weights
    are the replay's selection counts of each span)."""
    seen = server_segments(log_text)
    result = replay(layers, batches, n_expert, slots, **options)
    spans = result["segments"][:len(seen)]
    rows = []
    for i, (server, (hits, total)) in enumerate(zip(seen, spans)):
        rows.append({"span": i, "server_percent": server, "replay_percent": 100.0 * hits / total if total else 0.0, "selections": total})
    used = [r for r in rows if r["selections"]]
    weight = sum(r["selections"] for r in used)

    def mean(key):
        return sum(r[key] * r["selections"] for r in used) / weight if weight else 0.0

    return {"slots": slots, "spans": rows, "server_percent": mean("server_percent"), "replay_percent": mean("replay_percent"),
            "max_span_difference": max((abs(r["server_percent"] - r["replay_percent"]) for r in used), default=0.0),
            "replay_spans": len(result["segments"]), "server_spans": len(seen)}


# ------------------------------------------------------------------------------------------------------------ collection

PROMPTS = [
    "Write a Python function that merges overlapping intervals and explain its complexity.",
    "Explain how a transformer language model computes attention, step by step, for a beginner.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Translate into French and German: 'The meeting has been moved to Thursday afternoon because of the storm.'",
    "Solve: a train leaves at 9:15 at 80 km/h, a second train leaves the same station at 10:00 at 110 km/h. When does it catch up?",
    "Give me a recipe for a vegetarian lasagna and list the shopping items.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Write a SQL query that returns the three customers with the highest total order value per country.",
    "What is the difference between a mutex and a semaphore? Give a code example in C++.",
    "Compose a formal email that asks a landlord to repair a broken heater.",
    "Explain the proof that the square root of two is irrational.",
    "Write a haiku sequence of four verses about autumn rain.",
    "Describe the main steps of the Krebs cycle and where each step takes place.",
    "Write a bash script that finds the ten largest files in a directory tree and prints their sizes.",
    "Role-play as a medieval innkeeper greeting a tired traveler who has just arrived in a snowstorm.",
    "Return a JSON object that describes a book with title, authors, year and three tags, for a fictional novel.",
    "Compare the economic policies of two countries of your choice and keep it neutral.",
    "Implement binary search in Rust and write two unit tests.",
    "Why is the sky blue, and why are sunsets red? Keep it short.",
    "Give five tips for a first marathon and explain the reason for each tip.",
    "Write a limerick about a programmer who never commented the code.",
    "Explain what a hash map is and how it handles collisions, then show a tiny implementation in JavaScript.",
    "Derive the formula for the sum of the first n cubes by induction.",
    "Plan a three-day trip to Kyoto with a low budget and list the main costs.",
    "Write dialogue between a detective and a suspect who answers every question with another question.",
    "Describe how a refrigerator works, with the role of the refrigerant.",
    "What are the main risks of storing passwords in plain text, and what should be done instead?",
    "Rewrite this sentence in five different styles: 'The results were not what we had hoped for.'",
    "Explain gradient descent and write a short NumPy example that fits a line to noisy points.",
    "Tell me about the history of the printing press and its effect on science.",
]


def long_prompts(text_paths, count=3, characters=3000):
    """A few prompts that hold a slice of the calibration text, so that the trace has prompt batches of several ubatches."""
    pieces = []
    for path in text_paths or []:
        try:
            data = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(data) < characters * 2:
            continue
        for i in range(count):
            start = (len(data) - characters) * (i + 1) // (count + 1)
            pieces.append("Summarize the following text in three sentences.\n\n" + data[start:start + characters])
        break
    return pieces


def ordered_prompts(text_paths):
    """The short prompts with the long ones spread among them (after the 4th, 12th and 20th), so that a trace of a few
    thousand tokens holds prompts of several ubatches between decode spans."""
    longs = long_prompts(text_paths)
    result = list(PROMPTS)
    for position, prompt in reversed(list(zip((4, 12, 20), longs))):
        result.insert(position, prompt)
    return result


def ask(url, payload, timeout):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def collect(command, env, trace, tokens, log, prompts, per_request=160, ready_timeout=3600, request_timeout=3600, seed=1):
    """Run the server command with the routing log, send prompts one at a time until `tokens` tokens are generated.

    The server is stopped at the end. Returns {"generated", "requests": [{"prompt_tokens", "completion_tokens"}], "seconds"}."""
    trace = Path(trace)
    trace.unlink(missing_ok=True)
    started = time.monotonic()
    with open(log, "w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, **env, "LLAMA_MOE_ROUTING_LOG": str(trace), "LLAMA_MOE_ROUTING_LOG_MAX_TOKENS": "0"})
    sent, generated = [], 0
    try:
        port = command[command.index("--port") + 1]
        base = f"http://127.0.0.1:{port}"
        while True:
            if process.poll() is not None:
                raise QuantError(f"llama-server exited with {process.returncode} while loading; see {log}")
            if time.monotonic() - started > ready_timeout:
                raise QuantError(f"llama-server was not ready after {ready_timeout} s; see {log}")
            try:
                if ask(base + "/health", None, 5).get("status") == "ok":
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(2)
        for i, prompt in enumerate(prompts):
            if generated >= tokens:
                break
            reply = ask(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": per_request,
                                                          "temperature": 0.7, "seed": seed + i}, request_timeout)
            usage = reply["usage"]
            sent.append({"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"]})
            generated += usage["completion_tokens"]
    finally:
        process.terminate()
        try:
            process.wait(60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if not trace.exists() or trace.stat().st_size == 0:
        raise QuantError(f"the server wrote no routing records to {trace}; see {log}")
    return {"generated": generated, "requests": sent, "seconds": time.monotonic() - started}


# ------------------------------------------------------------------------------------------------------------- the stage

CURVE_FILE = "routing-curve.json"
TRACE_FILE = "routing-trace.bin"
TOKENS = 4000


def calibration_paths(cfg):
    text = cfg.get("calibration", {}).get("text")
    entries = text if isinstance(text, list) else [text] if text else []
    return [e["path"] if isinstance(e, dict) else str(e).split(":")[0] for e in entries]


def curve_key(cfg, model, serve, tokens, bindir_files):
    return digest({"model": file_identity(model), "serve": serve, "tokens": tokens, "prompts": PROMPTS, "binaries": bindir_files,
                   "devices": cfg["devices"], "threads": cfg["runtime"]["threads"],
                   "policy": [MAX_ROWS, WINDOW, EVERY, HYSTERESIS, MAX_ASYNC, DELAY, list(GRID)], "format": 1})


def ensure(cfg, info, model, serve, directory, tokens=TOKENS, runtime_args=None, ready_timeout=3600):
    """The hit curve of the model in `directory`: reused when the model file, the serving layout and the binaries are the
    same, else a trace is collected with the planned server (draft model and projector included, expert cache off) and replayed.

    Returns the curve (the dict of routing-curve.json), or None when tokens is 0."""
    from verify import free_port, server_command, server_env
    if tokens <= 0:
        return None
    directory = Path(directory)
    bindir = Path(cfg["bindir"])
    binaries = {name: file_identity(bindir / name) for name in ("llama-server",)}
    key = curve_key(cfg, model, serve, tokens, binaries)
    path = directory / CURVE_FILE
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("key") == key:
            print(f"Reusing {path}: {old['trace']['generated']} generated tokens, {old['trace']['summary']['decode_rows']} decode rows", flush=True)
            return load_curve(path)
    trace = directory / TRACE_FILE
    prompts = ordered_prompts(calibration_paths(cfg))
    command = server_command(cfg, model, runtime_args, serve, free_port())
    env = {**server_env(cfg), "LLAMA_MOE_CACHE_MIB": "0"}
    print(f"Collecting a routing trace of about {tokens} generated tokens with {model}", flush=True)
    import progress
    with progress.Watcher("prepare: routing trace", lambda: None):
        sent = collect(command, env, trace, tokens, directory / "routing-server.log", prompts, ready_timeout=ready_timeout)
    layers, batches = read_trace(trace)
    summary = trace_summary(layers, batches)
    points = curve(layers, batches, info["n_expert"])
    write_json(path, {"schema": SCHEMA, "key": key, "n_expert": info["n_expert"], "layers": len(layers), "top_k": summary["top_k"],
                      "points": points, "trace": {**sent, "summary": summary, "file": str(trace), "model": str(model),
                                                   "command": command},
                      "policy": {"max_rows": MAX_ROWS, "window": WINDOW, "every": EVERY, "hysteresis": HYSTERESIS,
                                 "max_async": MAX_ASYNC, "delay": DELAY}})
    print(f"Routing curve: {len(points)} points from {summary['decode_rows']} decode rows and {summary['prompts']} prompt batches; "
          f"saved in {path}", flush=True)
    return load_curve(path)
