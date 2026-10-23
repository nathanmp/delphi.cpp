"""Disk space of the stages: one estimate function with one row per artifact, and the free-space check.

Every full GGUF that the tool writes holds the lazy tables (engram, per-layer embeddings) at their stored size, so the
rows include them. When the source keeps them in a shard of their own (a base, base.py), the models that the tool writes
link that shard, so the rows leave them out. The probe model row is an upper bound (the largest qtype of every group)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from config import QuantError
from model import is_lazy, tensor_bytes
from progress import size_text

# Calibration writes one patch chunk (calibration.patch_mib, default 8192) at a time; the previous one stays until the
# reload commits, so two chunks are on disk
PATCH_MIB_DEFAULT = 8192
# Small files of every stage: JSON records, logs, per-chunk KL values
RECORDS = 256 << 20


def teacher_bytes(n_vocab, chunks, context):
    """Size of the `llama-perplexity --kl-divergence-base` file (tools/perplexity/perplexity.cpp).

    Header (magic 8, n_ctx, n_vocab, n_chunk 4 each), the tokens of all chunks, then for the scored second half of every
    chunk (n_ctx - 1 - n_ctx / 2 positions) one row of nv uint16 log probabilities, nv = 2 * ceil(n_vocab / 2) + 4."""
    nv = 2 * ((n_vocab + 1) // 2) + 4
    return 20 + chunks * context * 4 + chunks * (context - 1 - context // 2) * nv * 2


def vocab_size(path):
    """Number of tokens in the vocabulary of a GGUF (the length of tokenizer.ggml.tokens)."""
    import sys
    from config import ROOT
    sys.path.insert(0, str(ROOT / "gguf-py"))
    from gguf import GGUFReader
    field = GGUFReader(str(path)).fields.get("tokenizer.ggml.tokens")
    if field is None:
        raise QuantError(f"{path} has no tokenizer.ggml.tokens; cannot size the teacher logits")
    return len(field.data)


def model_sizes(info):
    """(smallest, largest, lazy) bytes of a full GGUF of the model: every group at its smallest or largest pool qtype."""
    by_name = {t["name"]: t for t in info["tensors"]}
    low = high = lazy = 0
    for group in info["groups"]:
        members = [by_name[n] for n in group["members"]]
        sizes = [sum(tensor_bytes(m["shape"], q) for m in members) for q in group["pool"]]
        low += min(sizes)
        high += max(sizes)
        if all(is_lazy(m["name"]) for m in members):
            lazy += min(sizes)
    return low, high, lazy


def recipe_bytes(info, tensor_types):
    """Bytes of the tensor payloads of the GGUF that encode() writes for the given type per tensor name: the lazy tables
    are included, except in the model with a lazy shard of its own, where the shard is a link."""
    return sum(tensor_bytes(t["shape"], tensor_types[t["name"]]) for t in info["tensors"]
               if not (info.get("lazy_separate") and is_lazy(t["name"])))


def artifacts(stage, sizes, calibration=None, vocab=None, source_bytes=0, mmproj_bytes=0, output_bytes=None, ram=None,
              linked_lazy=False):
    """[(artifact, directory key, bytes)] that a stage holds at its peak.

    sizes is model_sizes(info); calibration is the calibration section of the configuration. Directory keys: work
    (durable results), scratch (temporary files), output (the directory of the final GGUF)."""
    low, high, lazy = sizes
    if linked_lazy:
        low, high = low - lazy, high - lazy
    if stage == "convert":
        return [("Q8_0 source GGUF", "scratch", source_bytes), ("mmproj GGUF", "scratch", mmproj_bytes)]
    if stage == "prepare":
        return [("baseline GGUF", "work", low), ("records and logs", "work", RECORDS)]
    if stage == "calibrate":
        teacher = teacher_bytes(vocab, calibration["chunks"], calibration["context"]) if vocab else 0
        patch = 2 * calibration.get("patch_mib", PATCH_MIB_DEFAULT) * (1 << 20)
        rows = [("patch files", "scratch", patch), ("teacher logits", "scratch", teacher), ("records and logs", "scratch", RECORDS)]
        # a uniform recipe that does not fit in RAM runs as one separate probe model at the top probe qtype
        if calibration.get("uniform") == "direct" or (ram is not None and high > ram):
            rows.insert(0, ("probe model (uniform recipe)", "scratch", high))
        return rows
    if stage == "encode":
        return [("encoded GGUF", "output", high if output_bytes is None else output_bytes)]
    if stage == "verify":
        return [("KL token files and logs", "scratch", RECORDS)]
    raise ValueError(f"unknown stage {stage}")


def existing_parent(path):
    path = Path(path).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def check(stage, rows, directories, enabled=True, disk_usage=shutil.disk_usage):
    """Stop when a file system cannot hold the rows that go to it. Directories on one file system add up.

    The message names each directory with the bytes needed and free; --no-space-check turns the check off."""
    if not enabled:
        return
    groups = {}
    for name, key, size in rows:
        if size <= 0:
            continue
        where = existing_parent(directories[key])
        groups.setdefault(os.stat(where).st_dev, {"where": where, "rows": []})["rows"].append((name, key, size))
    for group in groups.values():
        need = sum(size for _, _, size in group["rows"])
        free = disk_usage(group["where"]).free
        if need > free:
            parts = ", ".join(f"{name} {size_text(size)} in {directories[key]}" for name, key, size in group["rows"])
            raise QuantError(f"not enough disk space for stage {stage} on the file system of {group['where']}: "
                             f"{size_text(need)} needed ({parts}), {size_text(free)} free. "
                             "Use --scratch DIR on a larger disk, free space, or pass --no-space-check")


def ram_available():
    """Bytes of memory available to programs (MemAvailable of /proc/meminfo), None where that file does not exist."""
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def preflight(stage, info, cfg, directories, enabled=True, **extra):
    """Estimate and check one stage for a classified model (config.py configuration cfg)."""
    rows = artifacts(stage, model_sizes(info), cfg.get("calibration"), ram=ram_available(),
                     linked_lazy=bool(info.get("lazy_separate")), **extra)
    check(stage, rows, directories, enabled)
    return rows
