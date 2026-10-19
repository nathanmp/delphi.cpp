"""Results store and crash journal of an auto-tuner output directory.

results.jsonl holds one record per measurement. journal.jsonl gets a 'start' line (fsync) before a launch and an 'end' line
after it, so a launch that took the machine down is known on resume and is not tried again.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator


class Store:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.results_path = directory / "results.jsonl"
        self.journal_path = directory / "journal.jsonl"
        self.records = list(self._read(self.results_path))
        started: dict[str, int] = {}
        for item in self._read(self.journal_path):
            started[item["key"]] = started.get(item["key"], 0) + (1 if item["event"] == "start" else -1)
        self.crashed = {key for key, open_count in started.items() if open_count > 0}

    @staticmethod
    def _read(path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError:
                    continue  # a line cut by a crash

    @staticmethod
    def _append(path: Path, item: dict[str, Any]) -> None:
        with path.open("a") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def start(self, key: str, name: str) -> None:
        self._append(self.journal_path, {"key": key, "event": "start", "name": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def end(self, key: str) -> None:
        self._append(self.journal_path, {"key": key, "event": "end"})

    def add(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {**record, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self._append(self.results_path, record)
        self.records.append(record)
        return record

    def find(self, key: str, kind: str, phase: str | None = None) -> dict[str, Any] | None:
        """The latest record of *key*; incomplete racing runs can be measured again."""
        for record in reversed(self.records):
            if record["key"] == key and record["kind"] == kind and (phase is None or record.get("phase") == phase):
                return record
        return None
