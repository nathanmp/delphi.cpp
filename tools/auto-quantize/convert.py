"""Conversion of a safetensors checkpoint to a Q8_0 GGUF (and a projector GGUF for a vision tower) in the scratch directory.

Q8_0 as the source of the advisor is within noise of BF16 (measured 2026-10-15) and needs half the disk."""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import progress
from config import ROOT, QuantError, file_identity

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


def is_gguf(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"GGUF"


def safetensors_files(directory):
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise QuantError(f"{directory} holds no .safetensors files and is not a GGUF")
    return files


def header(path):
    with open(path, "rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(size))


def q8_bytes(directory):
    """Estimated size of the Q8_0 GGUF: matrices with a row length of 32 multiples take 34 bytes per 32 weights, the
    rest (norms, small tensors) is stored as F32. Scale tensors of FP8 checkpoints are not subtracted, so it is an upper bound."""
    total = 0
    for f in safetensors_files(directory):
        for name, t in header(f).items():
            if name == "__metadata__":
                continue
            shape = t["shape"]
            n = math.prod(shape) if shape else 1
            total += n // 32 * 34 if len(shape) >= 2 and shape[-1] % 32 == 0 else n * 4
    return total


def staging_bytes(directory):
    """Scratch space that the converter needs while it runs: DeepSeek V4.1 engram tables are staged as Q8_0 in files
    next to the output (V41_ENGRAM_TMPDIR) until the GGUF is written."""
    total = 0
    for f in safetensors_files(directory):
        for name, t in header(f).items():
            if name.endswith("engram.embed.weight"):
                total += math.prod(t["shape"]) // 32 * 34
    return total


def source_bytes(directory):
    return sum(f.stat().st_size for f in safetensors_files(directory))


def has_vision(directory):
    """Whether config.json describes a vision tower (a vision_config at the top level or inside the text config)."""
    config = Path(directory) / "config.json"
    if not config.exists():
        return False
    text = json.loads(config.read_text())
    return any(key in text for key in ("vision_config", "visual", "vision_tower"))


def source_identity(source):
    """Inputs of the convert stage: the GGUF itself, or the checkpoint files and the converter script."""
    source = Path(source)
    if is_gguf(source):
        return {"gguf": file_identity(source)}
    return {"files": [file_identity(f) for f in safetensors_files(source)],
            "config": file_identity(source / "config.json") if (source / "config.json").exists() else None,
            "converter": file_identity(ROOT / "convert_hf_to_gguf.py")}


# modules that convert_hf_to_gguf.py imports at the start or for the tokenizer
CONVERTER_MODULES = ("torch", "numpy", "sentencepiece", "transformers")


def check_converter(python, env=None):
    """Stop before a long conversion when `python` lacks a module of the converter."""
    code = "import importlib.util, sys; print(' '.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))"
    try:
        completed = subprocess.run([python, "-c", code, *CONVERTER_MODULES], capture_output=True, text=True, env={**os.environ, **(env or {})})
        missing = completed.stdout.split() if completed.returncode == 0 else ["(the interpreter failed)"]
    except OSError as e:
        missing = [f"(the interpreter does not start: {e.strerror})"]
    if missing:
        raise QuantError(f"{python} cannot run convert_hf_to_gguf.py, missing: {' '.join(missing)}. Install "
                         f"{ROOT / 'requirements/requirements-convert_hf_to_gguf.txt'} into it or give another python with --converter-python")


def run_converter(python, source, outfile, extra, log, env, expected):
    check_converter(python, env)
    command = [python, str(ROOT / "convert_hf_to_gguf.py"), str(source), "--outfile", str(outfile), *extra]
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", ""), **env}
    with open(log, "w") as stream, progress.watch_file(f"convert {Path(outfile).name}", outfile, expected, poll=5.0):
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    if completed.returncode:
        raise QuantError(f"conversion failed ({completed.returncode}); see {log}")
    return command


def convert(source, directory, python=None, mmproj=None, env=None, check=lambda rows: None):
    """Q8_0 GGUF of the checkpoint in `directory`, plus a projector GGUF when the checkpoint has a vision tower and mmproj
    is None. Returns (gguf, mmproj path or None). check(rows) is called with the space rows before each conversion."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable
    final = directory / "model-q8_0.gguf"
    expected = q8_bytes(source)
    if not final.exists():
        partial = final.with_suffix(".partial.gguf")
        partial.unlink(missing_ok=True)
        staging = staging_bytes(source)
        check([("Q8_0 source GGUF", "scratch", expected)] + ([("converter staging files", "scratch", staging)] if staging else []))
        # the staging files go to the scratch disk, not to /tmp
        env = {"V41_ENGRAM_TMPDIR": str(directory), **(env or {})}
        run_converter(python, source, partial, ["--outtype", "q8_0"], directory / "convert.log", env, expected)
        partial.replace(final)
    projector = Path(mmproj) if mmproj else None
    if projector is None and has_vision(source):
        folder = directory / "mmproj"
        found = sorted(folder.glob("mmproj-*.gguf"))
        if not found:
            # written to a partial directory first, so that a partial file cannot be taken for a finished one. The converter
            # adds the "mmproj-" prefix only when --outfile is a directory; a file name it keeps as it is
            partial = directory / "mmproj.partial"
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir()
            log = directory / "convert-mmproj.log"
            try:
                run_converter(python, source, partial / "mmproj-model.gguf", ["--mmproj", "--outtype", "f16"], log, env or {}, None)
            except QuantError:
                if "is not supported" not in log.read_text(errors="replace"):
                    raise
                # the converter has no projector for this architecture (DeepSeek V4.1): the text model is still usable
                shutil.rmtree(partial, ignore_errors=True)
                print(f"warning: the checkpoint has a vision tower, but convert_hf_to_gguf.py cannot convert its projector "
                      f"(see {log}); continuing without one", flush=True)
                return final, None
            if not list(partial.glob("mmproj-*.gguf")):
                raise QuantError(f"the converter wrote no mmproj file in {partial}; see {directory / 'convert-mmproj.log'}")
            shutil.rmtree(folder, ignore_errors=True)
            partial.rename(folder)
            found = sorted(folder.glob("mmproj-*.gguf"))
        projector = found[0]
    return final, projector
