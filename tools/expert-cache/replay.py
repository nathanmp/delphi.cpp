#!/usr/bin/env python3
"""Replay expert-cache residency policies over a routing dump (LLAMA_MOE_CACHE_DUMP).

The dump holds every routed row in order: small batches (decode / speculative verification, kind 0), prompt batches
(kind 1), batch starts (kind 3) and the server's refreshes (kind 2). A policy picks the resident set at each refresh point;
the replay scores the small batches against it. The CPU reads each cold expert once per ubatch and layer, so the main
metric is cold distinct experts per (small batch, layer).

  replay.py DUMP --slots 1323 [--profile mimo-cache.profile] [--per-response]
"""
import argparse
import collections
import numpy as np

KIND_SMALL, KIND_PROMPT, KIND_REFRESH, KIND_BATCH = 0, 1, 2, 3


def load(path):
    data = open(path, "rb").read()
    assert data[:16] == b"DELPHI_MOE_DUMP1", "not a routing dump"
    off, events = 16, []
    while off + 12 <= len(data):
        seq = int.from_bytes(data[off:off + 4], "little")
        il, rows, kk = np.frombuffer(data, np.uint16, 3, off + 4)
        off += 12
        k, kind = int(kk) & 0xFF, int(kk) >> 8
        ids = None
        if kind in (KIND_SMALL, KIND_PROMPT):
            n = int(rows) * k
            if off + n > len(data):
                break  # a dump of a running server ends mid-record
            ids = np.frombuffer(data, np.uint8, n, off).reshape(int(rows), k)
            off += n
        events.append((seq, kind, int(il), int(rows), ids))
    return events


def load_profile(path, n_expert):
    prior = collections.defaultdict(lambda: np.zeros(n_expert))
    with open(path) as f:
        head = f.readline().split()
        assert head[0] == "DELPHI_MOE_PROFILE_V1"
        for line in f:
            if line.strip():
                il, e, c = line.split()
                prior[int(il)][int(e)] = float(c)
    return prior


class Window:
    """Per-layer routing window: counts over the last `rows` routed rows (0 = unbounded), optionally decayed."""

    def __init__(self, n_expert, k, rows, decay=0.0):
        self.n_expert, self.k, self.cap, self.decay = n_expert, k, rows, decay
        self.count = np.zeros(n_expert)
        self.ring = collections.deque()
        self.rows = 0

    def add(self, ids):
        if self.decay > 0:
            self.count *= (1.0 - self.decay) ** len(ids)
            np.add.at(self.count, ids.ravel(), 1.0)
            self.rows += len(ids)
            return
        np.add.at(self.count, ids.ravel(), 1.0)
        if self.cap <= 0:
            self.rows += len(ids)
            return
        self.ring.append(ids)
        self.rows += len(ids)
        while self.rows > self.cap:
            old = self.ring[0]
            drop = min(len(old), self.rows - self.cap)
            np.subtract.at(self.count, old[:drop].ravel(), 1.0)
            if drop == len(old):
                self.ring.popleft()
            else:
                self.ring[0] = old[drop:]
            self.rows -= drop


class Policy:
    def __init__(self, name, window=2048, prompts=True, decode=True, prior_rows=2048, prior_scale=1.0, hysteresis=0.25,
                 decay=0.0, every=0, quota=None, lag=0, reserve=0, cap=0):
        self.name, self.window, self.prompts, self.decode = name, window, prompts, decode
        self.prior_rows, self.prior_scale, self.hysteresis = prior_rows, prior_scale, hysteresis
        self.decay, self.every, self.quota = decay, every, quota
        # lag: small batches before a periodic refresh takes effect (uploads in flight); reserve: slots kept free for them
        self.lag, self.reserve = lag, reserve
        self.cap = cap  # experts uploaded per periodic refresh (llama_expert_cache::max_async), 0 = unlimited


def choose(windows, prior, resident, slots, pol, n_expert):
    """Top `slots` (layer, expert) by window count + prior fill, with a bonus for resident experts (refresh() in C++)."""
    keys, scores = [], []
    for il, w in windows.items():
        fill = max(0, pol.prior_rows - w.rows) * pol.prior_scale
        s = w.count + (fill * prior[il] if prior is not None and il in prior else 0.0)
        s = s * np.where(resident.get(il, np.zeros(n_expert, bool)), 1.0 + pol.hysteresis, 1.0)
        keys.append(il)
        scores.append(s)
    scores = np.stack(scores)
    flat = scores.ravel()
    order = np.argsort(-flat, kind="stable")
    take = order[:slots]
    take = take[flat[take] > 0]
    out = {il: np.zeros(n_expert, bool) for il in keys}
    for idx in take:
        out[keys[idx // n_expert]][idx % n_expert] = True
    return out


def replay(events, pol, slots, prior, n_expert, k, oracle=None):
    windows = collections.defaultdict(lambda: Window(n_expert, k, pol.window, pol.decay))
    for il in (prior or {}):
        windows[il]
    resident, uploads, refreshes = {}, 0, 0
    small_batches = 0
    stats = collections.Counter()
    per_response = []
    cur = collections.Counter()

    pending_sets = collections.deque()  # (small batch at which it takes effect, set)

    def capped(new):
        """Admit only the best `cap` newcomers, in place of the weakest residents that left the top set (tick() in C++)."""
        scores = {il: w.count + (max(0, pol.prior_rows - w.rows) * pol.prior_scale * prior[il]
                                 if prior is not None and il in prior else 0.0) for il, w in windows.items()}
        adds, outs, n_res = [], [], 0
        for il, m in new.items():
            old = resident.get(il, np.zeros(n_expert, bool))
            n_res += int(old.sum())
            for e in np.nonzero(m & ~old)[0]:
                adds.append((-scores[il][e] * 1.0, il, e))
            for e in np.nonzero(old & ~m)[0]:
                outs.append((scores[il][e], il, e))
        adds.sort()
        outs.sort()
        free = slots - pol.reserve - n_res
        out = {il: resident.get(il, np.zeros(n_expert, bool)).copy() for il in new}
        for i, (_, il, e) in enumerate(adds[:pol.cap]):
            if free > 0:
                free -= 1
            elif i - max(0, slots - pol.reserve - n_res) < len(outs):
                _, oil, oe = outs[i - max(0, slots - pol.reserve - n_res)]
                out[oil][oe] = False
            else:
                break
            out[il][e] = True
        return out

    def refresh(tag=None, delay=0):
        nonlocal resident, uploads, refreshes
        if oracle is not None:
            new = oracle(tag)
        else:
            new = choose(windows, prior, resident, slots - pol.reserve, pol, n_expert)
            if pol.cap and delay is not None and tag is None:
                new = capped(new)
        if delay:
            pending_sets.append((small_batches + delay, new))
            return
        for il, m in new.items():
            old = resident.get(il)
            uploads += int(m.sum() if old is None else (m & ~old).sum())
        resident = new
        refreshes += 1

    response = -1
    for seq, kind, il, rows, ids in events:
        if kind == KIND_REFRESH:
            if cur["all_sel"]:
                per_response.append(cur)
            cur = collections.Counter()
            response += 1
            refresh(response)
            continue
        if kind == KIND_BATCH:
            if rows <= 16:
                small_batches += 1
                while pending_sets and pending_sets[0][0] <= small_batches:
                    new = pending_sets.popleft()[1]
                    for il2, m2 in new.items():
                        old2 = resident.get(il2)
                        uploads += int(m2.sum() if old2 is None else (m2 & ~old2).sum())
                    resident = new
                    refreshes += 1
                if pol.every and small_batches % pol.every == 0 and oracle is None:
                    refresh(delay=pol.lag)
            continue
        if kind == KIND_SMALL:
            m = resident.get(il)
            hit = m[ids] if m is not None else np.zeros(ids.shape, bool)
            stats["all_sel"] += ids.size
            stats["hit_sel"] += int(hit.sum())
            u = np.unique(ids)
            cold = int((~m[u]).sum()) if m is not None else len(u)
            stats["distinct"] += len(u)
            stats["cold_distinct"] += cold
            stats["layer_batches"] += 1
            for key, v in (("all_sel", ids.size), ("hit_sel", int(hit.sum())), ("distinct", len(u)), ("cold_distinct", cold),
                           ("layer_batches", 1)):
                cur[key] += v
            if pol.decode:
                windows[il].add(ids)
        elif kind == KIND_PROMPT and pol.prompts:
            windows[il].add(ids)
    if cur["all_sel"]:
        per_response.append(cur)
    return stats, uploads, refreshes, per_response


def response_oracle(events, slots, n_expert, lookahead):
    """Resident set = top `slots` of the routing of the next `lookahead` responses' small batches (1 = this response)."""
    spans, cur = [], collections.defaultdict(lambda: np.zeros(n_expert))
    for seq, kind, il, rows, ids in events:
        if kind == KIND_REFRESH:
            spans.append(cur)
            cur = collections.defaultdict(lambda: np.zeros(n_expert))
        elif kind == KIND_SMALL:
            np.add.at(cur[il], ids.ravel(), 1.0)
    spans.append(cur)
    spans = spans[1:]  # rows before the first refresh are warm-up; the set chosen at refresh i serves span i

    def pick(i):
        acc = collections.defaultdict(lambda: np.zeros(n_expert))
        for s in spans[i:i + lookahead]:
            for il, c in s.items():
                acc[il] += c
        if not acc:
            return {}
        keys = list(acc)
        flat = np.stack([acc[il] for il in keys]).ravel()
        take = np.argsort(-flat, kind="stable")[:slots]
        take = take[flat[take] > 0]
        out = {il: np.zeros(n_expert, bool) for il in keys}
        for idx in take:
            out[keys[idx // n_expert]][idx % n_expert] = True
        return out
    return lambda i: pick(i) if i is not None and i < len(spans) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--slots", type=int, default=1323)
    ap.add_argument("--profile")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--per-response", action="store_true")
    ap.add_argument("--policies", default="all")
    a = ap.parse_args()

    events = load(a.dump)
    k = next(ids.shape[1] for _, kind, _, _, ids in events if ids is not None)
    prior = None
    if a.profile:
        prior = load_profile(a.profile, a.experts)
        for il in prior:  # selections per row, as in llama_expert_cache::create
            prior[il] = prior[il] / (prior[il].sum() / k)
    n_small = sum(1 for e in events if e[1] == KIND_SMALL)
    n_prompt_rows = sum(e[3] for e in events if e[1] == KIND_PROMPT)
    n_small_rows = sum(e[3] for e in events if e[1] == KIND_SMALL)
    n_ref = sum(1 for e in events if e[1] == KIND_REFRESH)
    print(f"{len(events)} records: {n_small} small-batch layer records ({n_small_rows} rows), {n_prompt_rows} prompt layer-rows, "
          f"{n_ref} refreshes, top_k {k}, {a.slots} slots")

    P = Policy
    policies = [
        P("current (W2048, prompts, prior, hyst 0.25)"),
        P("no prior", prior_scale=0.0),
        P("hysteresis 0", hysteresis=0.0),
        P("hysteresis 1.0", hysteresis=1.0),
        P("decode rows only", prompts=False),
        P("prompt rows only", decode=False),
        P("W512", window=512, prior_rows=512),
        P("W1024", window=1024, prior_rows=1024),
        P("W4096", window=4096, prior_rows=4096),
        P("W8192", window=8192, prior_rows=8192),
        P("W unbounded", window=0, prior_rows=2048),
        P("refresh every 16 small batches", every=16),
        P("refresh every 64 small batches", every=64),
        P("W1024 + every 16", window=1024, prior_rows=1024, every=16),
        P("W512 + every 16, hyst 0.25", window=512, prior_rows=512, every=16),
    ]
    if a.policies != "all":
        sel = set(a.policies.split(","))
        policies = [p for i, p in enumerate(policies) if str(i) in sel]

    def report(name, stats, uploads, refreshes, per):
        lb = max(1, stats["layer_batches"])
        line = (f"{name:44s} hit {100 * stats['hit_sel'] / max(1, stats['all_sel']):5.1f}% sel  "
                f"{100 * (1 - stats['cold_distinct'] / max(1, stats['distinct'])):5.1f}% distinct  "
                f"cold distinct/layer-batch {stats['cold_distinct'] / lb:5.2f}  refreshes {refreshes:4d} uploads {uploads:6d}")
        print(line)
        if a.per_response:
            print("    per response hit% sel: " + " ".join(f"{100 * c['hit_sel'] / max(1, c['all_sel']):.0f}" for c in per))

    for pol in policies:
        report(pol.name, *replay(events, pol, a.slots, prior, a.experts, k))
    for look, name in ((1, "oracle: this response's own routing"), (3, "oracle: this + next 2 responses")):
        report(name, *replay(events, P(name), a.slots, prior, a.experts, k, oracle=response_oracle(events, a.slots, a.experts, look)))


if __name__ == "__main__":
    main()
