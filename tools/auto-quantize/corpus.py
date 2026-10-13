"""Weighted calibration corpora: one deterministic KL text composed from several corpora.

The tool has no tokenizer. A piece is a span of about `context` tokens, sized in bytes with bytes_per_token, so the share of
tokens follows the weights only as well as that ratio holds for each corpus. Pieces of one corpus are chosen by a seeded hash,
and the pieces of all corpora are interleaved in proportion to the weights. llama-perplexity scores the first `chunks` windows
of the text: the text holds MARGIN times that many pieces, and any prefix of it keeps the proportions.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path

from config import QuantError, digest, file_identity, write_json

MARGIN = 1.2
FORMAT = 1


def shares(weights, total):
    """Split total pieces by the weights with the largest-remainder rule."""
    raw = [weight * total for weight in weights]
    counts = [int(value) for value in raw]
    for i in sorted(range(len(raw)), key=lambda i: (counts[i] - raw[i], i))[:total - sum(counts)]:
        counts[i] += 1
    return counts


def line_start(stream, position, size, limit):
    """First line start at or after position. Without a newline within limit bytes, a raw cut that keeps UTF-8 sequences whole."""
    if position <= 0 or position >= size:
        return min(max(position, 0), size)
    stream.seek(position - 1)
    block = stream.read(limit + 1)
    found = block.find(b"\n")
    if found >= 0:
        return position + found
    cut = position
    while cut < size and cut - position < 4 and (block[cut - position + 1] & 0xC0) == 0x80:
        cut += 1
    return cut


def read_piece(stream, slot, length, size):
    """Slot `slot` of a corpus cut into slots of `length` bytes, moved to line starts; neighbouring slots do not overlap."""
    limit = max(16, length // 4)
    start = line_start(stream, slot * length, size, limit)
    end = line_start(stream, (slot + 1) * length, size, limit)
    stream.seek(start)
    data = stream.read(end - start)
    return data if data.endswith(b"\n") else data + b"\n"


def compose_text(corpora, context, chunks, seed, bytes_per_token, directory):
    """(path, record) of one text from weighted corpora, written to directory under a name that holds the input digest."""
    total = math.ceil(chunks * MARGIN)
    counts = shares([c["weight"] for c in corpora], total)
    sources = []
    for c, count in zip(corpora, counts):
        identity = file_identity(c["path"], content=True)
        identity.pop("mtime_ns")
        ratio = c.get("bytes_per_token", bytes_per_token)
        length = max(64, round(context * ratio))
        slots = identity["bytes"] // length
        if count < 1:
            raise QuantError(f"calibration.chunks {chunks} is too small for the weight of {c['path']}: it receives no piece")
        if slots < count:
            raise QuantError(f"calibration corpus {c['path']} is too small for its share: {count} pieces of about {length} bytes "
                             f"need {count * length} bytes and the file has {identity['bytes']}; lower its weight or "
                             "calibration.chunks, add text, or set bytes_per_token")
        sources.append({**identity, "weight": c["weight"], "bytes_per_token": ratio, "piece_bytes": length, "slots": slots,
                        "pieces": count})
    record = {"format": FORMAT, "seed": seed, "context": context, "chunks": chunks, "pieces": total, "margin": MARGIN,
              "corpora": sources}
    record["key"] = digest(record)[:16]
    directory = Path(directory)
    path = directory / f"calibration-text-{record['key']}.txt"
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        old = json.loads(sidecar.read_text())
        if old.get("key") == record["key"] and hashlib.sha256(path.read_bytes()).hexdigest() == old["text"]["sha256"]:
            return str(path), old
    queue = []
    for index, source in enumerate(sources):
        slots = heapq.nsmallest(source["pieces"], range(source["slots"]),
                                key=lambda j: hashlib.sha256(f"{seed}:{index}:{j}".encode()).digest())
        with open(source["path"], "rb") as stream:
            pieces = [read_piece(stream, j, source["piece_bytes"], source["bytes"]) for j in sorted(slots)]
        queue += [((k + 0.5) / len(pieces), index, piece) for k, piece in enumerate(pieces)]
    queue.sort(key=lambda item: item[:2])
    body = b"".join(piece for _, _, piece in queue)
    record["text"] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    directory.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".txt.partial")
    partial.write_bytes(body)
    partial.replace(path)
    write_json(sidecar, record)
    names = ", ".join(f"{Path(s['path']).name} {s['pieces']}" for s in sources)
    print(f"Composed calibration text {path.name}: {len(queue)} pieces ({names}), {len(body)} bytes", flush=True)
    return str(path), record


def calibration_text(c, directory):
    """(path, record) of the calibration text. Weighted corpora become one text in directory; a single path stays as it is."""
    if isinstance(c["text"], str):
        return c["text"], None
    return compose_text(c["text"], c["context"], c["chunks"], c["text_seed"], c["bytes_per_token"], directory)
