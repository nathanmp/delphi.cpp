"""Resume state of `quant_cli.py run`: each finished stage with the fingerprint of its inputs.

A stage is finished when its recorded fingerprint equals the current one and its output files still exist. The
fingerprint of a stage includes the fingerprint of the stage before it, so a changed input re-runs its stage and every
stage after it. The record also keeps the inputs, so the message can name what changed.

A stage that starts gets a record without "finished": a run with the same inputs continues it, a run with other inputs
first removes its files. A re-run stage marks the later stages stale (fingerprint None), so that they also remove their
files from the earlier inputs when they run again. A stale stage whose fingerprint comes out the same as before (it does
not chain on everything of the stage before it) is finished again."""
from __future__ import annotations

import json
import time
from pathlib import Path

from config import digest, write_json

STAGES = ("convert", "prepare", "calibrate", "plan", "encode", "verify")


def fingerprint(inputs, previous):
    return digest({"inputs": inputs, "previous": previous})


def changed_keys(old, new):
    """Top-level keys whose values differ between two input records, sorted."""
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


class State:
    def __init__(self, work):
        self.path = Path(work) / "state.json"
        self.stages = json.loads(self.path.read_text())["stages"] if self.path.exists() else {}
        self.skipped = set()  # stages that this run found finished

    def decide(self, name, inputs, previous):
        """('skip', []) for a finished stage; ('run', []) for a new one; ('resume', []) for a started one with the same
        inputs; ('rerun', changed keys) when inputs changed or the stage is stale.

        previous is the fingerprint of the stage before; when only it changed the key list is ['earlier stage']."""
        record = self.stages.get(name)
        fp = fingerprint(inputs, previous)
        if record is not None and record.get("stale") and record.get("was") == fp:
            record = self.stages[name] = {k: v for k, v in record.items() if k not in ("stale", "was")} | {"fingerprint": fp}
            self.save()
        if record is None:
            return "run", []
        if record["fingerprint"] == fp and "finished" not in record:
            return "resume", []
        if record["fingerprint"] == fp:
            missing = [o for o in record.get("outputs", []) if not Path(o).exists()]
            return ("run", [f"missing output {missing[0]}"]) if missing else ("skip", [])
        keys = changed_keys(record["inputs"], inputs)
        return "rerun", keys or ["earlier stage"]

    def mark(self, name, inputs, previous, seconds, outputs=(), **extra):
        self.stages[name] = {"fingerprint": fingerprint(inputs, previous), "inputs": inputs, "seconds": round(seconds, 1),
                             "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "outputs": [str(o) for o in outputs], **extra}
        self.save()

    def start(self, name, inputs, previous):
        self.stages[name] = {"fingerprint": fingerprint(inputs, previous), "inputs": inputs,
                             "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.save()

    def forget_from(self, name):
        """Drop the record of a stage and mark every later recorded one stale."""
        self.stages.pop(name, None)
        for later in STAGES[STAGES.index(name) + 1:]:
            if later in self.stages:
                record = self.stages[later]
                self.stages[later] = {**record, "fingerprint": None, "stale": True,
                                      **({"was": record["fingerprint"]} if "finished" in record and record["fingerprint"] else {})}
        self.save()

    def save(self):
        write_json(self.path, {"stages": self.stages})
