"""GGUF metadata, exact encoded sizes, classification and raw imatrix merging."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from config import ROOT, RUNTIME_NAMES, QuantError, digest, file_identity, normalized, runtime_allows

sys.path.insert(0, str(ROOT / "gguf-py"))
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGML_QUANT_SIZES, GGML_ROW_META_SIZES


def qtype(value):
    try:
        result = GGMLQuantizationType[str(value).upper()]
        if result not in GGML_QUANT_SIZES:
            raise KeyError(value)
        return result.name
    except KeyError as exc:
        raise QuantError(f"unknown tensor qtype {value}; model presets such as Q4_K_M are not tensor types") from exc


def tensor_bytes(shape, dtype):
    dtype = GGMLQuantizationType[qtype(dtype)]
    block, size = GGML_QUANT_SIZES[dtype]
    if shape[0] % block:
        raise QuantError(f"row length {shape[0]} is incompatible with {dtype.name}")
    return math.prod(shape[1:]) * (shape[0] // block * size + GGML_ROW_META_SIZES.get(dtype, 0))


SHARD = re.compile(r"(.*)-([0-9]{5})-of-([0-9]{5})\.gguf")


def unsharded(path):
    """X.gguf of X-00001-of-00002.gguf (a path that is not a shard name stays as it is)."""
    match = SHARD.fullmatch(str(path))
    return Path(match[1] + ".gguf") if match else Path(path)


def shard_name(path, index, count):
    """The name of shard INDEX (0 based) of COUNT shards of the model that X.gguf (or one of its shards) names."""
    return Path(f"{str(unsharded(path))[:-5]}-{index + 1:05d}-of-{count:05d}.gguf")


def model_files(path):
    """Every file of a model by its first file: the shards of a split GGUF, else the file itself."""
    match = SHARD.fullmatch(str(path))
    if not match:
        return [Path(path)]
    return [shard_name(path, i, int(match[3])) for i in range(int(match[3]))]


def unlink_model(path):
    """Delete the files of a model, written as X.gguf or as shards, and the links among them. A missing file is skipped."""
    stem = unsharded(path)
    for candidate in [stem, *sorted(stem.parent.glob(glob_escape(stem.stem) + "-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf"))]:
        candidate.unlink(missing_ok=True)


def glob_escape(text):
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def read_model(path):
    first = GGUFReader(path)
    def field(name, default=None):
        value = first.fields.get(name)
        return value.contents() if value else default
    count = int(field("split.count", 1))
    paths = [Path(path)]
    if count > 1:
        match = re.fullmatch(r"(.*)-[0-9]{5}-of-[0-9]{5}\.gguf", str(path))
        if not match:
            raise QuantError("split GGUF requires its numbered first shard")
        paths = [Path(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    arch = field("general.architecture")
    info = {"architecture": arch, "n_expert": int(field(f"{arch}.expert_count", 0)),
            "top_k": int(field(f"{arch}.expert_used_count", 1)), "n_layer": int(field(f"{arch}.block_count", 0)),
            "nextn": int(field(f"{arch}.nextn_predict_layers", 0)), "tensors": [], "files": []}
    seen = set()
    for index, p in enumerate(paths):
        reader = first if index == 0 else GGUFReader(p)
        info["files"].append(file_identity(p))
        for t in reader.tensors:
            if t.name in seen:
                raise QuantError(f"duplicate tensor {t.name} across shards")
            seen.add(t.name)
            info["tensors"].append({"name": t.name, "shape": [int(x) for x in t.shape], "source_dtype": t.tensor_type.name,
                                    "source_bytes": t.n_bytes, "offset": t.data_offset, "file": str(p), "split": index})
    info["identity"] = digest({"files": info["files"], "tensors": info["tensors"]})
    # The last shard holds all lazy tables and nothing else: a file that can sit on a fast disk of its own, and that every
    # model written from this one shares (see auto_quantize.encode)
    last = len(paths) - 1
    lazy = [t for t in info["tensors"] if is_lazy(t["name"])]
    info["lazy_separate"] = bool(last and lazy and all((t["split"] == last) == is_lazy(t["name"]) for t in info["tensors"]))
    return info


def mtp_layers(info):
    """Multi-token-prediction blocks: the blocks with blk.N.nextn.* tensors and the last nextn_predict_layers blocks."""
    found = {int(m[1]) for t in info["tensors"] if (m := re.match(r"blk\.([0-9]+)\.nextn\.", t["name"]))}
    declared = set(range(info["n_layer"] - info["nextn"], info["n_layer"])) if info.get("nextn", 0) > 0 else set()
    if found != declared:
        print(f"warning: MTP blocks by tensor names {sorted(found)} and by nextn_predict_layers {sorted(declared)} differ; "
              "both are treated as MTP blocks", file=sys.stderr)
    return sorted(found | declared)


# Large lookup tables that the runtime reads row by row from disk (lazy mode); they stay fixed (at the source type or a pin)
# and do not count as device bytes.
LAZY = ("engram_embd", "per_layer_token_embd")


def is_lazy(name):
    return any(part in name for part in LAZY)


_lazy_args = {}


def lazy_args(cfg):
    """Load options for models with lazy tables: memory-mapped, lazy reads on, tables kept on the host (empty without tables).

    The model's tensor names are read once per path."""
    path = cfg["model"]
    if path not in _lazy_args:
        found = [part for part in LAZY if any(part in t["name"] for t in read_model(path)["tensors"])]
        _lazy_args[path] = ["-lm", "mmap", "-lzm", "on", "-ot", "|".join(found) + "=CPU"] if found else []
    return list(_lazy_args[path])


# Conservative preservation mirrors the ordinary quantizer exclusions. Encoding is independently checked against the plan.
PRESERVE = ("_norm.weight", "ffn_gate_inp.weight", "position_embd", "token_types", "ssm_conv1d", "shortconv.conv.weight",
            "indexer.k_proj.weight", "indexer.q_proj.weight", "time_mix_", "attn_rel_b.weight", ".patch_embd",
            ".patch_merger", ".rel_pos", "sam.pos_embd", "sam.neck.", "sam.net_", "a.rvq.codebook", "mm.a.code_embd")
GLM_PRESERVE = ("hc_", "indexer_compressor_ape", "indexer_compressor_gate", "indexer.proj", "indexer.attn_k",
                "indexer.attn_q_b", "ssm_f_a", "ssm_f_b", "ssm_g_a", "ssm_g_b", "ssm_beta", "attn_kv_a_mqa",
                "attn_k_b", "attn_v_b")


def lazy_pins(info, cfg):
    """The pins of the configuration, and the type lazy_type (config key; --lazy-type) for every lazy table that has
    another type and no pin of its own."""
    pins = dict(cfg["pins"])
    if cfg.get("lazy_type"):
        wanted = qtype(cfg["lazy_type"])
        for t in info["tensors"]:
            if is_lazy(t["name"]) and len(t["shape"]) >= 2 and t["name"].endswith("weight") and t["source_dtype"] != wanted:
                pins.setdefault(t["name"], wanted)
    return pins


def classify(info, cfg):
    pools = {category: [qtype(q) for q in cfg[f"{category}_qtypes"]] for category in ("primary", "other")}
    if any(not pool for pool in pools.values()):
        raise QuantError("both qtype pools must be nonempty")
    info["mtp_layers"] = mtp_layers(info)
    names = {t["name"] for t in info["tensors"]}
    if set(cfg["pins"]) - names:
        raise QuantError(f"pins name missing tensors: {sorted(set(cfg['pins']) - names)}")
    pins = lazy_pins(info, cfg)
    for t in info["tensors"]:
        name, shape = t["name"], t["shape"]
        primary = bool(re.search(cfg["primary_regex"], name)) if cfg.get("primary_regex") else (
            "_exps.weight" in name if info["n_expert"] else bool(re.search(r"\.ffn_(gate|up|down|gate_up)\.weight$", name)))
        t["category"] = "primary" if primary else "other"
        t["expert"] = bool(re.search(r"\.ffn_(gate|up|down|gate_up)_exps\.weight$", name))
        t["layer"] = int(re.match(r"blk\.([0-9]+)\.", name)[1]) if name.startswith("blk.") else -1
        # Perplexity and the main decode path do not run the MTP blocks; their weights stay at the ceiling (see below).
        t["mtp"] = t["layer"] in info["mtp_layers"]
        preserve = (len(shape) < 2 or not name.endswith("weight") or t["source_bytes"] < cfg["preserve_below_mib"] * (1 << 20)
                    or any(part in name for part in PRESERVE)
                    # a pin can store a lazy table below the source precision (a BF16 table can be ~100 GB)
                    or (is_lazy(name) and name not in pins)
                    or (info["architecture"] == "glm5next" and any(part in name for part in GLM_PRESERVE))
                    or any(re.search(pattern, name) for pattern in cfg["preserve_regex"])
                    or t["source_dtype"].startswith("I") and t["source_dtype"] in ("I8", "I16", "I32", "I64"))
        if preserve and name in pins and qtype(pins[name]) != t["source_dtype"]:
            raise QuantError(f"pin conflicts with preservation: {name}")
        if not preserve and t["source_dtype"] not in ("F16", "BF16", "F32") and not cfg.get("allow_requantize", False):
            raise QuantError("source is quantized; set allow_requantize for source-relative damage")
        t["preserved"] = preserve
        t["pool"] = [t["source_dtype"]] if preserve else ([qtype(pins[name])] if name in pins else pools[t["category"]])
        if t["category"] + "_qtypes" in cfg.get("default_pools", ()) and not preserve and name not in pins:
            # default pools: only the types whose block fits the row length of this tensor
            t["pool"] = [q for q in t["pool"] if shape[0] % GGML_QUANT_SIZES[GGMLQuantizationType[q]][0] == 0]
        t["pool"] = sorted(set(t["pool"]))
    groups, assigned = [], set()
    by_name = {t["name"]: t for t in info["tensors"]}
    for t in info["tensors"]:
        if t["name"] in assigned:
            continue
        members = [t]
        partner = ""
        if "ffn_gate_up" not in t["name"]:
            if "ffn_gate" in t["name"]:
                partner = t["name"].replace("ffn_gate", "ffn_up", 1)
            elif "ffn_up" in t["name"]:
                partner = t["name"].replace("ffn_up", "ffn_gate", 1)
        if cfg["tie_gate_up"] and partner != t["name"] and partner in by_name:
            members.append(by_name[partner])
        pool = set(members[0]["pool"])
        for member in members[1:]:
            pool &= set(member["pool"])
        if not pool:
            # A preserved/pinned member constrains its partner even if the type is outside the category allowlist.
            fixed = [m for m in members if m["preserved"] or m["name"] in pins]
            if fixed and len({m["pool"][0] for m in fixed}) == 1:
                pool = {fixed[0]["pool"][0]}
            else:
                raise QuantError(f"unsatisfiable tied group {t['name']}")
        legal = []
        for dtype in sorted(pool):
            try:
                for m in members:
                    tensor_bytes(m["shape"], dtype)
                legal.append(dtype)
            except QuantError:
                pass
        if not legal:
            raise QuantError(f"no row-compatible qtype for {t['name']}")
        if any(m["mtp"] for m in members) and not any(m["preserved"] or m["name"] in pins for m in members):
            # The KL measurement cannot see the draft quality, so MTP groups take the ceiling: the most bytes of the pool.
            legal = [max(legal, key=lambda q: (sum(tensor_bytes(m["shape"], q) for m in members), q))]
        for member in members:
            member["pool"] = legal
            assigned.add(member["name"])
        groups.append({"name": t["name"], "members": [m["name"] for m in members], "pool": legal})
    info["groups"] = groups
    return info


def check_runtime_source(info, cfg):
    """Stop when a tensor that the encoder copies unchanged has a type that the target runtime cannot load. Such a tensor
    is never quantized again (the encoder checks its payload), so the source of another type is the remedy."""
    runtime = cfg.get("target_runtime", "delphi")
    bad = [t for t in info["tensors"] if t["preserved"] and not runtime_allows(runtime, t["source_dtype"])]
    if bad:
        raise QuantError(f"the source has {len(bad)} preserved tensors with types that runtime {RUNTIME_NAMES[runtime]} cannot load "
                         f"(first: {bad[0]['name']} is {bad[0]['source_dtype']}); use a BF16, F16 or Q8_0 source or another --runtime")


def imatrix_tokens(path):
    """Tokens that a raw imatrix saw: its largest count, which is how llama-imatrix defines the chunk count."""
    counts = [float(np.max(t.data)) for t in GGUFReader(path).tensors if t.name.endswith(".counts") and t.data.size]
    if not counts or max(counts) <= 0:
        raise QuantError(f"imatrix has no token counts: {path}")
    return max(counts)


def whole_counts(tensors):
    """F32 tensors with whole counts. The quantizer rounds counts on load, so round here and scale the sums to keep every mean."""
    for name in [n for n in tensors if n.endswith(".counts")]:
        base, counts = name[:-7], tensors[name].reshape(-1)
        rounded = np.where(counts > 0, np.maximum(np.rint(counts), 1), 0)
        ratio = np.divide(rounded, counts, out=np.ones_like(counts), where=counts > 0)
        for suffix in (".in_sum2", ".in_sum"):
            sums = tensors.get(base + suffix)
            if sums is not None:
                if sums.size % counts.size:
                    raise QuantError(f"imatrix sums and counts disagree: {base}")
                tensors[base + suffix] = (sums.reshape(counts.size, -1) * ratio[:, None]).reshape(sums.shape)
        tensors[name] = rounded.reshape(tensors[name].shape)
    result = {name: value.astype(np.float32) for name, value in tensors.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise QuantError("weighted imatrix sums exceed the F32 range")
    return result


def merge_imatrices(paths, output, weights=None):
    """Merge raw GGUF imatrices. Without weights the sums and counts are added, so a file weighs in proportion to its tokens.

    With weights (relative, one per file) file k is scaled by w_k * T / T_k before the addition, where T_k is the token count
    of the file and T the sum over all files. The merged mean of every tensor and expert is then the mean of the mixture
    in which file k supplies the fraction w_k of the tokens, and the counts are those of T tokens."""
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    for path in paths:
        with open(path, "rb") as stream:
            if stream.read(4) != b"GGUF":
                raise QuantError("multiple imatrices require raw GGUF sums/counts; legacy merging is not supported")
    scales = None
    if weights is not None:
        if len(weights) != len(paths) or any(w <= 0 for w in weights):
            raise QuantError("imatrix weights must be positive, one per file")
        weights = normalized(weights)
        tokens = [imatrix_tokens(path) for path in paths]
        scales = [w * sum(tokens) / t for w, t in zip(weights, tokens)]
        print("Weighted imatrix merge: " + ", ".join(f"{Path(p).name} weight {100 * w:.1f} % (own tokens {100 * t / sum(tokens):.1f} %)"
                                                      for p, w, t in zip(paths, weights, tokens)), flush=True)

    def scaled(value, index):
        if scales is None:
            return value
        if value.dtype != np.float32:
            raise QuantError("weighted merging needs F32 sums and counts")
        return value.astype(np.float64) * scales[index]
    tensors, means, datasets, chunks, chunk_size = {}, {}, [], 0, None
    for index, path in enumerate(paths):
        reader = GGUFReader(path)
        entries = {t.name: t for t in reader.tensors}
        for name, t in entries.items():
            if name.endswith((".in_sum", ".in_sum1")):
                # activation sums (".in_sum1" is the older delphi name); summed like the second moments
                base = name[:name.rindex(".in_sum")]
                if base + ".in_sum2" not in entries:
                    raise QuantError(f"imatrix activation sums have no second moments: {name}")
                means.setdefault(base + ".in_sum", []).append(scaled(np.array(t.data, copy=True), index))
                continue
            if not name.endswith((".in_sum2", ".counts")):
                continue
            partner = name[:-8] + ".counts" if name.endswith(".in_sum2") else name[:-7] + ".in_sum2"
            if partner not in entries:
                raise QuantError(f"imatrix entry has no raw counterpart: {name}")
            value = scaled(np.array(t.data, copy=True), index)
            if name in tensors:
                if tensors[name].shape != value.shape or tensors[name].dtype != value.dtype:
                    raise QuantError(f"incompatible imatrix shape/type: {name}")
                limit = np.iinfo(value.dtype).max if value.dtype.kind in "iu" else np.finfo(value.dtype).max
                if name.endswith(".counts") and np.any(value > limit - tensors[name]):
                    raise QuantError(f"imatrix count overflow: {name}")
                tensors[name] += value
            else:
                tensors[name] = value
            if not np.isfinite(tensors[name]).all() or np.any(tensors[name] < 0):
                raise QuantError(f"invalid imatrix sums/counts: {name}")
        for key, field in reader.fields.items():
            if key == "imatrix.datasets":
                datasets.extend(field.contents())
            elif key == "imatrix.chunk_count":
                chunks += int(field.contents())
            elif key == "imatrix.chunk_size":
                size = int(field.contents())
                if chunk_size is not None and size != chunk_size:
                    raise QuantError("imatrix chunk sizes differ")
                chunk_size = size
    if not tensors:
        raise QuantError("imatrices contain no raw sums/counts")
    for name, values in means.items():
        # means need the same tokens as the second moments: keep them only when every input has them
        if len(values) == len(paths):
            if any(v.shape != values[0].shape for v in values):
                raise QuantError(f"incompatible imatrix activation sums: {name}")
            tensors[name] = np.sum(values, axis=0)
            if not np.isfinite(tensors[name]).all():
                raise QuantError(f"invalid imatrix activation sums: {name}")
    if scales is not None:
        tensors = whole_counts(tensors)
    writer = GGUFWriter(str(output), "imatrix")
    writer.add_array("imatrix.datasets", datasets)
    writer.add_uint32("imatrix.chunk_count", chunks)
    writer.add_uint32("imatrix.chunk_size", chunk_size or 0)
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(output)
