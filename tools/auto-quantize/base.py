"""Base model: the source with its lazy tables (engram, per-layer embeddings) rewritten at a chosen type, as a split GGUF.

Shard 1 holds every other tensor with its source type and bytes (a copy: a tensor that keeps its type is not quantized
again, so MXFP4 experts and Q8_0 dense weights stay bit-exact). The last shard holds only the lazy tables. The lazy shard can
sit on a fast disk of its own (`lazy_dir`; the name next to shard 1 is a link to it), and every model that the tool then
encodes from the base shares that shard (auto_quantize.encode links it instead of writing it again).

Calibration, the teacher logits and the encode of a run use the base as their source. The KL values are then relative to
the base, so they leave out the error of the lazy tables. `measure_lazy_kl` measures that error once, on a few chunks."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import progress
from config import QuantError, file_identity, write_json
from model import is_lazy, qtype, read_model, shard_name, tensor_bytes, unlink_model, unsharded
from timing import run

# bytes compared at the head, middle and tail of every large tensor of the copy (the small tensors are compared in full)
SAMPLE = 1 << 20
LARGE = 64 << 20


def base_types(info, lazy_type):
    """{tensor: type name} of the base: the source type everywhere, lazy_type for the lazy weight tables."""
    wanted = qtype(lazy_type)
    return {t["name"]: wanted if is_lazy(t["name"]) and len(t["shape"]) >= 2 and t["name"].endswith("weight") else t["source_dtype"]
            for t in info["tensors"]}


def base_rows(info, lazy_type, output, lazy_dir=None):
    """[(artifact, directory key, bytes)] of the base for space.check; keys: output (shard 1) and lazy (the lazy shard)."""
    types = base_types(info, lazy_type)
    lazy = sum(tensor_bytes(t["shape"], types[t["name"]]) for t in info["tensors"] if is_lazy(t["name"]))
    main = sum(tensor_bytes(t["shape"], types[t["name"]]) for t in info["tensors"] if not is_lazy(t["name"]))
    return [("base shard 1 (all tensors but the lazy tables)", "output", main), ("base lazy shard", "lazy", lazy)]


def already_base(info, lazy_type):
    """Whether the source is a base of this lazy type already (a split with the lazy tables alone in the last shard)."""
    return bool(info.get("lazy_separate")) and all(
        t["source_dtype"] == qtype(lazy_type) for t in info["tensors"] if is_lazy(t["name"]) and len(t["shape"]) >= 2)


def link(source, name):
    """A hard link of source at name when both are on one file system, else a symbolic link."""
    name = Path(name)
    name.unlink(missing_ok=True)
    try:
        os.link(source, name)
        return "hard link"
    except OSError:
        os.symlink(Path(source).resolve(), name)
        return "symbolic link"


def compare_copy(source_info, found, lazy_names):
    """The copied tensors equal their source: names, types, shapes and bytes (large tensors at three places)."""
    expected = {t["name"]: t for t in source_info["tensors"]}
    actual = {t["name"]: t for t in found["tensors"]}
    if set(expected) != set(actual):
        raise QuantError("the base has other tensors than the source")
    handles = {}
    try:
        def open_once(path):
            if path not in handles:
                handles[path] = open(path, "rb")
            return handles[path]
        for name, t in actual.items():
            if name in lazy_names:
                continue
            s = expected[name]
            if (t["source_dtype"], t["shape"], t["source_bytes"]) != (s["source_dtype"], s["shape"], s["source_bytes"]):
                raise QuantError(f"the base changed the type, shape or size of {name}")
            size = t["source_bytes"]
            spots = [0] if size <= LARGE else [0, (size - SAMPLE) // 2, size - SAMPLE]
            length = size if size <= LARGE else SAMPLE
            for spot in spots:
                a, b = open_once(s["file"]), open_once(t["file"])
                a.seek(s["offset"] + spot)
                b.seek(t["offset"] + spot)
                if a.read(length) != b.read(length):
                    raise QuantError(f"the base differs from the source in {name}")
    finally:
        for stream in handles.values():
            stream.close()


def make_base(source, lazy_type, output, bindir, threads=32, lazy_dir=None, env=None, check=None):
    """Write the base of the source GGUF to output (X.gguf gives X-00001-of-00002.gguf and X-00002-of-00002.gguf).

    The lazy shard goes to lazy_dir when given, with a link next to shard 1. check(rows) stops when a disk is full.
    Returns the path of shard 1; a record X.base.json names the files, the types and the times."""
    source = Path(source).resolve()
    stem = unsharded(Path(output).resolve())
    info = read_model(source)
    lazy_names = [t["name"] for t in info["tensors"] if is_lazy(t["name"]) and len(t["shape"]) >= 2 and t["name"].endswith("weight")]
    if not lazy_names:
        raise QuantError(f"{source} has no lazy tables (engram_embd, per_layer_token_embd); there is nothing to put in a base")
    first, last = shard_name(stem, 0, 2), shard_name(stem, 1, 2)
    if first.exists() or last.exists():
        raise QuantError(f"the base exists: {first}")
    types = base_types(info, lazy_type)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if check:
        check(base_rows(info, lazy_type, stem, lazy_dir))
    partial = stem.with_name(stem.stem + ".partial.gguf")
    partial_first, partial_last = shard_name(partial, 0, 2), shard_name(partial, 1, 2)
    unlink_model(partial)
    plan_file = stem.with_suffix(".tensor-types.txt")
    lazy_file = stem.with_suffix(".lazy-tensors.txt")
    plan_file.write_text("".join(f"^{re.escape(n)}$={q.lower()}\n" for n, q in types.items()))
    lazy_file.write_text("".join(f"{n}\n" for n in lazy_names))
    target = None
    if lazy_dir:
        target = Path(lazy_dir).resolve() / last.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise QuantError(f"the lazy shard exists: {target}")
        partial_last.unlink(missing_ok=True)
        os.symlink(target, partial_last)
    # no --imatrix: the lazy tables are looked up, not multiplied, and every other tensor keeps its type
    command = [str(Path(bindir) / "llama-quantize"), "--allow-requantize", "--pure", "--tensor-type-file", str(plan_file),
               "--split-tensors", str(lazy_file), str(source), str(partial), "Q2_K", str(threads)]
    rows = base_rows(info, lazy_type, stem, lazy_dir)
    started = time.monotonic()
    print(f"Writing the base {first.name}: {len(lazy_names)} lazy tables to {qtype(lazy_type)}, the other tensors copied", flush=True)
    with progress.watch_file(f"base {first.name}", partial_first, rows[0][2]):
        run(command, stem.with_suffix(".base.log"), env, timeout=None)
    found = read_model(partial_first)
    if found.get("lazy_separate") is not True:
        raise QuantError("the written base has no separate lazy shard")
    wrong = [n for n in lazy_names if next(t for t in found["tensors"] if t["name"] == n)["source_dtype"] != qtype(lazy_type)]
    if wrong:
        raise QuantError(f"the lazy tables {wrong[:3]} do not have the type {qtype(lazy_type)}")
    compare_copy(info, found, set(lazy_names))
    partial_first.replace(first)
    if target:
        partial_last.unlink()
        os.symlink(target, last)
    else:
        partial_last.replace(last)
    seconds = time.monotonic() - started
    write_json(stem.with_suffix(".base.json"), {
        "source": file_identity(source), "lazy_type": qtype(lazy_type), "lazy_tensors": lazy_names, "files": [str(first), str(last)],
        "lazy_target": str(target) if target else None, "seconds": seconds, "command": command,
        "bytes": {"shard 1": rows[0][2], "lazy shard": rows[1][2]}})
    plan_file.unlink(missing_ok=True)
    lazy_file.unlink(missing_ok=True)
    print(f"Base written in {progress.duration_text(seconds)}: {first} and {last}", flush=True)
    return first


def measure_lazy_kl(bindir, source, base, text, chunks, context, device, output, threads=32, env=None, timeout=None):
    """KL of the base against its source on the first chunks of text: the error of the lazy tables alone (every other tensor
    is the same). Two llama-perplexity runs with the routed experts on the CPU; the source run stores its logits, the
    base run compares with them. The result goes to output (JSON) and is returned."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kld = output.with_suffix(".teacher.kld")
    tokens = output.with_suffix(".tokens.f32")
    lazy = [part for part in ("engram_embd", "per_layer_token_embd") if any(part in t["name"] for t in read_model(base)["tensors"])]
    common = ["-f", str(text), "--chunks", str(chunks), "-c", str(context), "-b", "2048", "-ub", "2048", "-t", str(threads),
              "-tb", str(threads), "--device", device, "-ngl", "999", "-cmoe", "-lm", "mmap", "-lzm", "on",
              "-ot", "|".join(lazy) + "=CPU"]
    env = dict(env or {}, LLAMA_MOE_CACHE_MIB="0", LLAMA_KLD_TOKENS_FILE=str(tokens))
    perplexity = str(Path(bindir) / "llama-perplexity")
    started = time.monotonic()
    run([perplexity, "-m", str(source), *common, "--kl-divergence-base", str(kld)], output.with_suffix(".teacher.log"), env, timeout=timeout)
    middle = time.monotonic()
    run([perplexity, "-m", str(base), *common, "--kl-divergence-base", str(kld), "--kl-divergence"], output.with_suffix(".base.log"), env,
        timeout=timeout)
    log = output.with_suffix(".base.log").read_text()
    kl = re.search(r"Mean\s+KLD:\s+([0-9.eE+-]+)\s+±\s+([0-9.eE+-]+)", log)
    same = re.search(r"Same top p:\s+([0-9.]+)", log)
    if not kl:
        raise QuantError(f"no KL value in {output.with_suffix('.base.log')}")
    result = {"kl": float(kl[1]), "se": float(kl[2]), "same_top_p_percent": float(same[1]) if same else None, "chunks": chunks,
              "context": context, "text": str(text), "source": str(source), "base": str(base),
              "seconds_source": middle - started, "seconds_base": time.monotonic() - middle}
    write_json(output, result)
    kld.unlink(missing_ok=True)
    tokens.unlink(missing_ok=True)
    return result
