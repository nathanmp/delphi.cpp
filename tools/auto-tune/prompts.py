"""Prompt banks: a YAML list of {domain, id, prompt, answer?} or plain text with prompts separated by lines of '---'."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    domain: str
    id: str
    prompt: str
    answer: str | None = None


def load(path: Path) -> list[Prompt]:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # only YAML banks need PyYAML
        items = yaml.safe_load(text)
        if not isinstance(items, list):
            raise ValueError(f"{path}: the top level must be a list")
        out = []
        for i, item in enumerate(items):
            if not isinstance(item, dict) or not str(item.get("prompt", "")).strip():
                raise ValueError(f"{path}: item {i} needs a prompt")
            answer = item.get("answer")
            out.append(Prompt(str(item.get("domain", "default")), str(item.get("id", f"p{i}")), str(item["prompt"]),
                              None if answer is None else str(answer)))
    else:
        parts = [p.strip() for p in "\n".join(l if l.strip() != "---" else "\x00" for l in text.splitlines()).split("\x00")]
        out = [Prompt("default", f"p{i}", p) for i, p in enumerate(p for p in parts if p)]
    if not out:
        raise ValueError(f"{path}: no prompts")
    ids = [p.id for p in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    return out


def domains(bank: list[Prompt]) -> list[str]:
    seen: list[str] = []
    for p in bank:
        if p.domain not in seen:
            seen.append(p.domain)
    return seen


def subset(bank: list[Prompt], per_domain: int) -> list[Prompt]:
    """The first *per_domain* prompts of every domain, in bank order."""
    count: dict[str, int] = {}
    out = []
    for p in bank:
        if count.get(p.domain, 0) < per_domain:
            out.append(p)
            count[p.domain] = count.get(p.domain, 0) + 1
    return out
