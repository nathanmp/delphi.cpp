#!/usr/bin/env python3
"""delphi-vramd: device memory leases for llama-servers that share GPUs.

Each llama-server started with LLAMA_VRAMD=<socket> connects, registers what it holds on each GPU, and leases more memory
for its expert cache before each prompt. The daemon grants free memory, revokes other servers' cache leases when the policy
allows it, and reclaims everything a server held when its connection closes. See README.md for the protocol and policy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field

MiB = 1 << 20

log = logging.getLogger("vramd")


@dataclass
class Lease:
    id: int
    server: "Server"
    gpu: str
    bytes: int
    purpose: str  # cache, scratch, kv
    revoking: float = 0.0  # time the revoke was sent, 0 if not revoking


@dataclass
class Request:
    server: "Server"
    req: int
    gpu: str
    purpose: str
    bytes: int
    min: int
    chunk: int
    max_leases: int
    wait: bool
    prefill: bool
    t: float
    waiting_sent: bool = False


@dataclass
class Server:
    id: int
    writer: asyncio.StreamWriter
    model: str = "?"
    pid: int = 0
    home: str = ""
    resident: dict[str, int] = field(default_factory=dict)
    phase: str = "idle"
    phase_t: float = field(default_factory=time.monotonic)
    ready: bool = False

    def name(self) -> str:
        return f"{self.model}#{self.id}"


class Daemon:
    def __init__(self, capacity: dict[str, int], grace: float, deadline: float, overhead: int):
        self.capacity = capacity
        self.overhead = overhead
        self.grace = grace
        self.deadline = deadline
        self.servers: dict[int, Server] = {}
        self.leases: dict[int, Lease] = {}
        self.pending: list[Request] = []
        self.next_server = 1
        self.next_lease = 1

    # accounting

    def used(self, gpu: str) -> int:
        # per server and GPU: what it registered, plus the runtime's memory that no lease covers (context, pools, scratch)
        resident = sum(s.resident.get(gpu, 0) + self.overhead for s in self.servers.values() if gpu in s.resident)
        return resident + sum(l.bytes for l in self.leases.values() if l.gpu == gpu)

    def free(self, gpu: str) -> int:
        return self.capacity.get(gpu, 0) - self.used(gpu)

    def send(self, s: Server, msg: dict) -> None:
        try:
            s.writer.write((json.dumps(msg) + "\n").encode())
        except Exception as e:  # the connection handler cleans up
            log.warning("send to %s failed: %s", s.name(), e)

    def new_lease(self, s: Server, gpu: str, nbytes: int, purpose: str, req: int) -> Lease:
        lease = Lease(self.next_lease, s, gpu, nbytes, purpose)
        self.next_lease += 1
        self.leases[lease.id] = lease
        self.send(s, {"op": "lease", "req": req, "lease": lease.id, "gpu": gpu, "purpose": purpose, "bytes": nbytes})
        return lease

    def drop_lease(self, lease_id: int, why: str) -> None:
        lease = self.leases.pop(lease_id, None)
        if lease is not None:
            log.info("lease %d (%s, %s, %.0f MiB on %s) %s%s", lease.id, lease.server.name(), lease.purpose, lease.bytes / MiB,
                     lease.gpu, why, f" after {time.monotonic() - lease.revoking:.3f} s" if lease.revoking else "")

    # policy

    def priority(self, r: Request) -> bool:
        return r.purpose in ("scratch", "kv") or r.gpu == r.server.home

    def victims(self, r: Request) -> list[Lease]:
        """Cache leases of other servers on the GPU that this request may take, best victim first."""
        now = time.monotonic()
        out = []
        for lease in self.leases.values():
            o = lease.server
            if lease.gpu != r.gpu or o is r.server or lease.purpose != "cache" or lease.revoking:
                continue
            idle_for = now - o.phase_t if o.phase == "idle" else -1.0
            if self.priority(r) and o.phase != "prefill":
                out.append((0 if idle_for >= 0 else 1, -idle_for, -lease.id, lease))
            elif idle_for >= self.grace:
                out.append((0, -idle_for, -lease.id, lease))
        return [x[-1] for x in sorted(out, key=lambda x: x[:3])]

    def finish(self, r: Request, granted: int) -> None:
        self.send(r.server, {"op": "done", "req": r.req, "granted": granted})
        if r in self.pending:
            self.pending.remove(r)

    def grant(self, r: Request, amount: int) -> None:
        chunk = max(r.chunk, r.min, math.ceil(amount / max(1, r.max_leases)))
        left, granted = amount, 0
        while left > 0:
            c = min(chunk, left)
            if 0 < left - c < r.min:
                c = left
            self.new_lease(r.server, r.gpu, c, r.purpose, r.req)
            left -= c
            granted += c
        log.info("granted %s %.0f MiB of %s on %s (%s, %d s waited)", r.server.name(), granted / MiB, r.purpose, r.gpu,
                 r.server.phase, time.monotonic() - r.t)
        self.finish(r, granted)

    def wait(self, r: Request) -> None:
        if not r.waiting_sent:
            r.waiting_sent = True
            self.send(r.server, {"op": "wait", "req": r.req})

    def process(self, r: Request) -> None:
        s = r.server
        if r.gpu not in self.capacity or r.min > self.capacity[r.gpu] - s.resident.get(r.gpu, 0) - self.overhead:
            log.info("denied %s %.0f MiB of %s on %s", s.name(), r.min / MiB, r.purpose, r.gpu)
            self.finish(r, 0)
            return
        # one prompt at a time: two would split the host's staging bandwidth
        if r.prefill and any(o.phase == "prefill" for o in self.servers.values() if o is not s):
            self.wait(r)
            return
        avail = self.free(r.gpu)
        if avail >= r.min and (avail >= r.bytes or not self.priority(r)):
            self.grant(r, min(avail, r.bytes))
            return
        # revoke until the request fits: all it asks for on its own GPU, at least its minimum elsewhere
        target = r.bytes if self.priority(r) else r.min
        coming = sum(l.bytes for l in self.leases.values() if l.gpu == r.gpu and l.revoking)
        for lease in self.victims(r):
            if avail + coming >= target:
                break
            lease.revoking = time.monotonic()
            coming += lease.bytes
            log.info("revoking lease %d (%s, %.0f MiB on %s) for %s", lease.id, lease.server.name(), lease.bytes / MiB, lease.gpu,
                     s.name())
            self.send(lease.server, {"op": "revoke", "lease": lease.id, "deadline_ms": int(self.deadline * 1000)})
        if coming > 0:
            self.wait(r)  # the revoked memory comes back soon
        elif avail >= r.min:
            self.grant(r, min(avail, r.bytes))  # nothing more to take
        elif self.priority(r) and any(l.gpu == r.gpu and l.purpose == "cache" and l.server is not s and l.server.phase == "prefill"
                                      for l in self.leases.values()):
            self.wait(r)  # revocable once that prompt is done
        elif r.purpose in ("scratch", "kv"):
            self.wait(r)  # the server cannot run without it; wait for others to release
        else:
            self.finish(r, 0)

    def evaluate(self) -> None:
        # requests on a server's own GPU first, then in arrival order
        for r in sorted(self.pending, key=lambda r: (not self.priority(r), r.t)):
            if r in self.pending:
                self.process(r)
        now = time.monotonic()
        for lease in self.leases.values():
            if lease.revoking and now - lease.revoking > self.deadline:
                log.warning("lease %d of %s not released %.1f s after the revoke", lease.id, lease.server.name(), now - lease.revoking)
                lease.revoking = now  # warn again after another deadline

    def status(self) -> dict:
        now = time.monotonic()
        return {
            "gpus": {g: {"capacity": c, "used": self.used(g), "free": self.free(g)} for g, c in self.capacity.items()},
            "servers": [{"id": s.id, "model": s.model, "pid": s.pid, "home": s.home, "phase": s.phase,
                         "phase_s": round(now - s.phase_t, 1), "resident": s.resident} for s in self.servers.values()],
            "leases": [{"lease": l.id, "server": l.server.id, "gpu": l.gpu, "purpose": l.purpose, "bytes": l.bytes,
                        "revoking": bool(l.revoking)} for l in self.leases.values()],
            "pending": [{"server": r.server.id, "req": r.req, "gpu": r.gpu, "purpose": r.purpose, "bytes": r.bytes, "min": r.min,
                         "prefill": r.prefill} for r in self.pending],
        }

    # connections

    def on_message(self, s: Server, m: dict) -> None:
        op = m.get("op")
        if op == "hello":
            s.model, s.pid, s.home = str(m.get("model", "?")), int(m.get("pid", 0)), str(m.get("home", ""))
        elif op == "gpu":
            gpu = str(m["gpu"])
            s.resident[gpu] = int(m.get("resident", 0))
            for purpose in ("scratch", "kv"):
                if int(m.get(purpose, 0)) > 0:
                    self.new_lease(s, gpu, int(m[purpose]), purpose, -1)
        elif op == "ready":
            s.ready = True
            log.info("server %s (pid %d, home %s) registered: %s", s.name(), s.pid, s.home,
                     ", ".join(f"{g} {b / MiB:.0f} MiB resident" for g, b in s.resident.items()))
            for g in s.resident:
                if g in self.capacity and self.free(g) < 0:
                    log.warning("%s is over capacity by %.0f MiB", g, -self.free(g) / MiB)
            self.send(s, {"op": "welcome", "id": s.id})
        elif op == "phase":
            state = str(m.get("state", "idle"))
            if state != s.phase:
                log.info("%s: %s -> %s", s.name(), s.phase, state)
                s.phase, s.phase_t = state, time.monotonic()
            if state == "idle":
                # an idle server does not need what it asked for opportunistically
                for r in [r for r in self.pending if r.server is s and not r.wait]:
                    self.finish(r, 0)
        elif op == "request":
            r = Request(s, int(m["req"]), str(m["gpu"]), str(m.get("purpose", "cache")), int(m["bytes"]), int(m.get("min", 0)),
                        int(m.get("chunk", 0)), int(m.get("max_leases", 4)), bool(m.get("wait", False)),
                        bool(m.get("prefill", False)), time.monotonic())
            self.pending.append(r)
        elif op == "cancel":
            for r in [r for r in self.pending if r.server is s and r.req == int(m.get("req", -1))]:
                self.finish(r, 0)
        elif op in ("release", "released"):
            lease = self.leases.get(int(m.get("lease", -1)))
            if lease is not None and lease.server is s:
                self.drop_lease(lease.id, "released" if op == "release" else "given back")
        elif op == "status":
            self.send(s, {"op": "status", **self.status()})
        else:
            log.warning("unknown message from %s: %s", s.name(), m)
        self.evaluate()

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        s = Server(self.next_server, writer)
        self.next_server += 1
        self.servers[s.id] = s
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("bad message from %s: %r", s.name(), line[:200])
                    continue
                self.on_message(s, m)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            # crash cleanup: the connection held the leases
            del self.servers[s.id]
            for lease_id in [l.id for l in self.leases.values() if l.server is s]:
                self.drop_lease(lease_id, "freed on disconnect")
            self.pending = [r for r in self.pending if r.server is not s]
            if s.ready:
                log.info("server %s disconnected", s.name())
            writer.close()
            self.evaluate()

    async def tick(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self.evaluate()


def parse_size(text: str) -> int:
    text = text.strip().upper()
    for suffix, mult in (("G", 1 << 30), ("M", MiB)):
        if text.endswith(suffix):
            return int(float(text[:-1]) * mult)
    return int(text) * MiB


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", default="/tmp/delphi-vramd.sock", help="Unix socket path (llama-server: LLAMA_VRAMD=<path>)")
    ap.add_argument("--gpu", action="append", required=True, metavar="NAME=SIZE",
                    help="memory the servers may use on a GPU in total, e.g. ROCm0=30.5G (MiB without a suffix); repeat per GPU")
    ap.add_argument("--grace", type=float, default=120.0, help="seconds an idle server keeps its cache on other servers' GPUs")
    ap.add_argument("--deadline", type=float, default=10.0, help="seconds before warning about an unanswered revoke")
    ap.add_argument("--overhead", default="1G", help="memory per server and GPU that leases do not cover (driver context, pools)")
    ap.add_argument("--log", default=None, help="log file (default: stderr)")
    args = ap.parse_args()
    logging.basicConfig(filename=args.log, level=logging.INFO, format="%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S")
    capacity = {}
    for item in args.gpu:
        name, _, size = item.partition("=")
        capacity[name] = parse_size(size)
    daemon = Daemon(capacity, args.grace, args.deadline, parse_size(args.overhead))
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = await asyncio.start_unix_server(daemon.connection, path=args.socket)
    log.info("delphi-vramd on %s: %s; grace %.0f s, overhead %.0f MiB per server and GPU", args.socket,
             ", ".join(f"{g} {c / MiB:.0f} MiB" for g, c in capacity.items()), args.grace, daemon.overhead / MiB)
    asyncio.create_task(daemon.tick())
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
