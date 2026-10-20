"""Progress lines on stdout for the long stages of the advisor.

One plain line per step, and at most one every INTERVAL seconds, so a log file stays readable (no carriage returns).
A line holds the stage name, the step or byte count, the elapsed time and the estimated finish (clock time and time left).
Watchers poll files or counters from a thread, so stages that only run a subprocess still report."""
from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import time
from pathlib import Path

INTERVAL = float(os.environ.get("DELPHI_QUANT_PROGRESS_SECONDS", "30"))
_print_lock = threading.Lock()


def emit(text):
    with _print_lock:
        print(text, flush=True)


def size_text(n):
    """12.3 GiB style text of a byte count."""
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def duration_text(seconds):
    seconds = int(max(0, round(seconds)))
    days, rest = divmod(seconds, 86400)
    h, rest = divmod(rest, 3600)
    m, s = divmod(rest, 60)
    return (f"{days}d " if days else "") + f"{h:02d}:{m:02d}:{s:02d}"


def clock_text(when, now):
    """Wall-clock time of day; the date is added when it is not today."""
    local = time.localtime(when)
    same_day = time.strftime("%Y%m%d", local) == time.strftime("%Y%m%d", time.localtime(now))
    return time.strftime("%H:%M:%S" if same_day else "%Y-%m-%d %H:%M:%S", local)


def eta_seconds(done, total, elapsed):
    """Linear estimate of the time left, None while there is no basis for one."""
    if not total or done <= 0 or elapsed <= 0:
        return None
    return elapsed * max(total - done, 0) / done


def format_line(name, done, total, unit, elapsed, now, eta, note=""):
    """'[name] step 3/10 (30%) elapsed 00:01:00 finish 12:30:00 (in 00:02:20)'."""
    if unit == "bytes":
        count = f"{size_text(done)}/{size_text(total)}" if total else size_text(done)
    else:
        count = f"step {done:g}/{total:g}" if total else f"step {done:g}"
    part = f" ({100 * min(done / total, 1):.0f}%)" if total else ""
    tail = f"finish {clock_text(now + eta, now)} (in {duration_text(eta)})" if eta is not None else "finish unknown"
    return f"[{name}] {count}{part} elapsed {duration_text(elapsed)} {tail}" + (f" {note}" if note else "")


class Progress:
    """One stage. update() is cheap; it prints when INTERVAL seconds have passed since the last line or on the last step."""

    def __init__(self, name, total=None, unit="", interval=None, out=emit, clock=time.time, quiet_start=False):
        self.name, self.total, self.unit = name, total, unit
        self.interval = INTERVAL if interval is None else interval
        self.out, self.clock = out, clock
        self.start = clock()
        self.done, self.last = 0, None
        if not quiet_start:
            self.out(f"[{name}] started at {clock_text(self.start, self.start)}"
                     + (f", total {size_text(total)}" if total and unit == "bytes" else f", {total:g} steps" if total else ""))

    def update(self, done, total=None, eta=None, note=""):
        if total is not None:
            self.total = total
        self.done = done
        now = self.clock()
        final = bool(self.total) and done >= self.total
        if self.last is not None and now - self.last < self.interval and not final:
            return
        self.last = now
        elapsed = now - self.start
        if eta is None:
            eta = eta_seconds(done, self.total, elapsed)
        self.out(format_line(self.name, done, self.total, self.unit, elapsed, now, 0.0 if final else eta, note))

    def finish(self, note=""):
        now = self.clock()
        self.out(f"[{self.name}] finished in {duration_text(now - self.start)}" + (f" {note}" if note else ""))


@contextlib.contextmanager
def stage(name, total=None, unit="", **kwargs):
    p = Progress(name, total, unit, **kwargs)
    try:
        yield p
    except BaseException:
        p.out(f"[{name}] stopped after {duration_text(p.clock() - p.start)}")
        raise
    p.finish()


class Watcher:
    """Polls probe() from a thread: it returns (done, total, eta, note), or None while there is nothing to show yet.

    Without a measurable step the line is a heartbeat (elapsed only), so a silent subprocess still shows life."""

    def __init__(self, name, probe, unit="", total=None, poll=2.0, **kwargs):
        self.progress = Progress(name, total, unit, **kwargs)
        self.probe, self.poll = probe, poll
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def tick(self):
        try:
            seen = self.probe()
        except OSError:
            seen = None
        p = self.progress
        if seen is None:
            if p.last is None or p.clock() - p.last >= p.interval:
                p.last = p.clock()
                p.out(f"[{p.name}] running, elapsed {duration_text(p.clock() - p.start)} (no step count yet)")
            return
        done, total, eta, note = seen
        p.update(done, total, eta, note)

    def run(self):
        while not self.stop.wait(self.poll):
            self.tick()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, kind, value, trace):
        self.stop.set()
        self.thread.join()
        if kind is None:
            self.tick()
            self.progress.finish()
        else:
            self.progress.out(f"[{self.progress.name}] stopped after {duration_text(self.progress.clock() - self.progress.start)}")


def file_probe(path, total):
    """Bytes written to path (or to its .partial sibling) against the expected total."""
    path = Path(path)

    def probe():
        for candidate in (path, path.with_suffix(".partial" + path.suffix)):
            if candidate.exists():
                return candidate.stat().st_size, total, None, ""
        return None
    return probe


def watch_file(name, path, total, **kwargs):
    return Watcher(name, file_probe(path, total), unit="bytes", total=total, **kwargs)


def count_matches(path, pattern):
    """Lines of a log file that match a pattern, 0 when the file does not exist yet."""
    try:
        regex = re.compile(pattern)
        with open(path, errors="replace") as stream:
            return sum(1 for line in stream if regex.search(line))
    except OSError:
        return 0


class Tee:
    """File-like that forwards writes and calls on_line(text) for each finished line; used to read counts from the
    printed lines of code that cannot report progress itself."""

    def __init__(self, inner, on_line):
        self.inner, self.on_line, self.buffer = inner, on_line, ""

    def write(self, text):
        self.buffer += text
        *lines, self.buffer = self.buffer.split("\n")
        for line in lines:
            self.on_line(line)
        return self.inner.write(text)

    def flush(self):
        self.inner.flush()

    def __getattr__(self, name):
        return getattr(self.inner, name)


@contextlib.contextmanager
def sniff_stdout(on_line):
    saved = sys.stdout
    sys.stdout = Tee(saved, on_line)
    try:
        yield
    finally:
        sys.stdout = saved


PASS_LINE = re.compile(r"Measuring (\d+) patch probes on (.+) and (\d+) uniform recipes")
CONTROL_LABELS = ("floor", "recompute", "restore")


def calibration_counts(output):
    """(passes done, mean seconds of a recorded probe pass, seconds of the floor pass) from the calibration directory.

    The sessions record every finished probe in session-DEVICE/run.json, and the uniform recipes leave uniform-*.f32."""
    import json
    done, seconds, floor = 0, [], None
    for record in sorted(Path(output).glob("session-*/run.json")):
        try:
            jobs = json.loads(record.read_text())["jobs"]
        except (OSError, ValueError, KeyError):
            continue
        for job in jobs:
            if job["label"] == "floor":
                floor = job.get("seconds", floor)
            elif job["label"] not in CONTROL_LABELS:
                done += 1
                seconds.append(job.get("seconds", 0.0))
    done += len(list(Path(output).glob("session-*/uniform-*.f32")))
    return done, (sum(seconds) / len(seconds) if seconds else None), floor


class CalibrationWatcher:
    """Progress of calibrate() without hooks in it: probe model encoding and teacher logits by file size, then the
    passes (done / total, estimated from the mean pass time and the number of sessions).

    The total comes from the line that calibrate prints before the passes; without it the lines show the count only."""

    def __init__(self, output, teacher_bytes=None, probe_bytes=None, poll=2.0, **kwargs):
        self.output, self.teacher_bytes, self.probe_bytes = Path(output), teacher_bytes, probe_bytes
        self.total, self.devices, self.direct = None, 1, 0
        self.watcher = Watcher("calibration", self.probe, poll=poll, **kwargs)
        self.sniffer = None

    def on_line(self, line):
        m = PASS_LINE.search(line)
        if m:
            self.direct = int(m[3])
            self.total = int(m[1]) + self.direct
            self.devices = len(m[2].split(","))

    def phase(self, name):
        """A new phase restarts the clock of the line, so elapsed and the estimate belong to it."""
        p = self.watcher.progress
        if p.name != name:
            p.name, p.start, p.last = name, p.clock(), None
            p.out(f"[{name}] started at {clock_text(p.start, p.start)}")
        return p

    def probe(self):
        partial = sorted(self.output.glob("*.partial.gguf"))
        if partial:
            self.phase("calibration: probe model")
            return sum(f.stat().st_size for f in partial), self.probe_bytes, None, ""
        teacher = self.output / "teacher.partial.kld"
        if teacher.exists():
            self.phase("calibration: teacher logits")
            return teacher.stat().st_size, self.teacher_bytes, None, ""
        done, mean, floor = calibration_counts(self.output)
        if not done and not self.total:
            return None
        self.phase("calibration: passes")
        if not self.total:
            return done, None, None, ""
        left = max(self.total - done, 0)
        eta = None
        if mean is not None:
            hot_left = max(left - max(self.direct - len(list(self.output.glob("session-*/uniform-*.f32"))), 0), 0)
            eta = hot_left * mean / self.devices + (left - hot_left) * (floor or mean)
        return done, self.total, eta, ""

    def __enter__(self):
        self.sniffer = sniff_stdout(self.on_line)
        self.sniffer.__enter__()
        self.watcher.__enter__()
        return self

    def __exit__(self, kind, value, trace):
        self.watcher.__exit__(kind, value, trace)
        self.sniffer.__exit__(kind, value, trace)
