"""Resume state of `quant_cli.py run`: each finished stage with the fingerprint of its inputs.

A stage is finished when its recorded fingerprint equals the current one and its output files still exist. The
fingerprint of a stage includes the fingerprint of the stage before it, so a changed input re-runs its stage and every
stage after it. The record also keeps the inputs, so the message can name what changed."""
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

    def decide(self, name, inputs, previous):
        """('skip', []) for a finished stage; ('run', []) for a new one; ('rerun', changed keys) when inputs changed.

        previous is the fingerprint of the stage before; when only it changed the key list is ['earlier stage']."""
        record = self.stages.get(name)
        fp = fingerprint(inputs, previous)
        if record is None:
            return "run", []
        if record["fingerprint"] == fp:
            missing = [o for o in record.get("outputs", []) if not Path(o).exists()]
            return ("run", [f"missing output {missing[0]}"]) if missing else ("skip", [])
        keys = changed_keys(record["inputs"], inputs)
        return "rerun", keys or ["earlier stage"]

    def mark(self, name, inputs, previous, seconds, outputs=(), **extra):
        self.stages[name] = {"fingerprint": fingerprint(inputs, previous), "inputs": inputs, "seconds": round(seconds, 1),
                             "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "outputs": [str(o) for o in outputs], **extra}
        self.save()

    def forget_from(self, name):
        """Drop the record of a stage and of every later one."""
        for later in STAGES[STAGES.index(name):]:
            self.stages.pop(later, None)
        self.save()

    def save(self):
        write_json(self.path, {"stages": self.stages})
