#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Requantize the routed experts of a native NVFP4 GGUF to symmetric INT4 with one BF16 scale per 128 values.

The output is an ordinary GGUF that keeps every other tensor and all metadata of the template. The routed expert tensors
(blk.N.ffn_{gate,up,down}_exps.weight) are replaced by tensors of the group-128 INT4 contract of the decode path
(ggml/src/ggml-cuda/moe-int4-decode.cuh), in one of two layouts (--format):
  g128   GGML_TYPE_Q4_G128, 66 bytes per 128 values (4.125 bpw): what the deployment stores.
  q4_0   GGML_TYPE_Q4_0 in which four successive blocks of 32 values share one scale (4.5 bpw): the same values for the generic
         kernels, so it is the baseline of an executor A/B on identical weights (and the format of the matched pilot banks).
The scale is exactly representable in BF16 and FP16. The per-expert global scales (ffn_*_exps.scale) of the NVFP4 source are
folded into the requantized weights, so the output has no scale tensors; ffn_*_exps.input_scale (activation scale, unused by
W4A16) is dropped.

Rule (the one of the matched banks of tools/cmp-int4-compare/compare.py, so the results stay comparable):
  w   = fp4_value * e4m3_block_scale * global_scale[expert]          (the effective source weight, FP32)
  s   = bf16(max(absmax(group of 128) / 7, 2^MIN_SCALE_EXP))         (RNE; the floor keeps s an exact FP16 value)
  q   = clamp(round(w / s), -8, 7), stored as q + 8 in the nibble
  w'  = q * s                                                         (what the kernels compute)
No calibration, no activation statistics. 4.125 bit per weight in the packed form of the kernels, 4.5 in Q4_0.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

EXPERT_WEIGHT = "blk.{}.ffn_{}_exps.weight"
PROJECTIONS = ("gate", "up", "down")
GROUP = 128


def open_readers(paths):
    readers = [gguf.GGUFReader(p, "r") for p in paths]
    tensors = {}
    order = []
    for r in readers:
        for t in r.tensors:
            tensors[t.name] = t
            order.append(t.name)
    return readers, tensors, order


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        left = limit
        while True:
            n = 1 << 24 if left is None else min(1 << 24, left)
            b = f.read(n)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
                if left <= 0:
                    break
    return h.hexdigest()


def nvfp4_tables(device):
    scale = torch.from_numpy(gguf.quants.NVFP4.ue4m3_to_fp32(np.arange(256, dtype=np.uint8)).astype(np.float32)).to(device)
    value = torch.tensor(gguf.quants.NVFP4.kvalues, dtype=torch.float32, device=device)
    return scale, value


def decode_nvfp4(raw, scale_table, value_table, n, k):
    """raw: uint8 [b, n, k/64*36] of the ggml NVFP4 layout -> FP32 [b, n, k] before the global scale."""
    b = raw.shape[0]
    blocks = raw.reshape(b, n, k // 64, 36)
    d = scale_table[blocks[..., :4].long()].unsqueeze(-1)
    qs = blocks[..., 4:].reshape(b, n, k // 64, 4, 8)
    codes = torch.cat((qs & 15, qs >> 4), dim=-1)
    return (value_table[codes.long()] * d).reshape(b, n, k)


def quantize_group128(w, min_scale_exp):
    """w: FP32 [b, n, k] -> (q int32 in [-8, 7] [b, n, k], s FP32 [b, n, k/128] holding BF16 values)."""
    b, n, k = w.shape
    g = w.reshape(b, n, k // GROUP, GROUP)
    s = (g.abs().amax(dim=-1) / 7).clamp_min(2.0 ** min_scale_exp).bfloat16().float()
    q = (g / s[..., None]).round().clamp(-8, 7).to(torch.int32)
    return q.reshape(b, n, k), s


def encode_q4_0(q, s):
    """Q4_0 blocks (FP16 scale, 16 bytes) with the group scale repeated over four blocks of 32. -> uint8 [b, n, k/32*18]."""
    b, n, k = q.shape
    s16 = s.half()
    if not torch.equal(s16.float(), s):
        raise RuntimeError("a BF16 group scale is not exact in FP16 (raise --min-scale-exp)")
    sb = s16.repeat_interleave(4, dim=-1).contiguous().view(torch.uint8).reshape(b, n, k // 32, 2)
    q32 = (q + 8).to(torch.uint8).reshape(b, n, k // 32, 32)
    qs = q32[..., :16] | (q32[..., 16:] << 4)
    return torch.cat((sb, qs), dim=-1).reshape(b, n, k // 32 * 18)


def decode_q4_0(blocks, n, k):
    """Independent decode of the bytes that encode_q4_0 wrote: uint8 [b, n, k/32*18] -> FP32 [b, n, k]."""
    b = blocks.shape[0]
    x = blocks.reshape(b, n, k // 32, 18)
    d = x[..., :2].contiguous().view(torch.float16).float()
    qs = x[..., 2:]
    lo = (qs & 15).float() - 8
    hi = (qs >> 4).float() - 8
    return (torch.cat((lo, hi), dim=-1) * d).reshape(b, n, k)


def encode_g128(q, s):
    """Q4_G128 blocks (BF16 scale bits, four groups of 32 codes in the Q4_0 layout). -> uint8 [b, n, k/128*66]."""
    b, n, k = q.shape
    if not torch.equal(s.half().float(), s):
        raise RuntimeError("a BF16 group scale is not exact in FP16 (raise --min-scale-exp)")
    bits = (s.contiguous().view(torch.int32) >> 16).to(torch.int16)                      # BF16 bits: the low 16 bits of the float are zero
    sb = bits.contiguous().view(torch.uint8).reshape(b, n, k // 128, 2)
    q32 = (q + 8).to(torch.uint8).reshape(b, n, k // 32, 32)
    qs = (q32[..., :16] | (q32[..., 16:] << 4)).reshape(b, n, k // 128, 64)
    return torch.cat((sb, qs), dim=-1).reshape(b, n, k // 128 * 66)


def decode_g128(blocks, n, k):
    """Independent decode of the bytes that encode_g128 wrote: uint8 [b, n, k/128*66] -> FP32 [b, n, k]."""
    b = blocks.shape[0]
    x = blocks.reshape(b, n, k // 128, 66)
    d = ((x[..., :2].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF) << 16).view(torch.float32)
    qs = x[..., 2:].reshape(b, n, k // 32, 16)
    lo = (qs & 15).float() - 8
    hi = (qs >> 4).float() - 8
    v = torch.cat((lo, hi), dim=-1).reshape(b, n, k // 128, 128)
    return (v * d).reshape(b, n, k)


FORMATS = {"q4_0": (encode_q4_0, decode_q4_0, gguf.GGMLQuantizationType.Q4_0, 32, 18),
           "g128": (encode_g128, decode_g128, gguf.GGMLQuantizationType.Q4_G128, 128, 66)}


def convert_tensor(src, glob, args, device, tables, stats):
    k, n, experts = (int(x) for x in src.shape)
    assert k % GROUP == 0 and n % 64 == 0, (k, n)
    encode, decode, _, qk, qbytes = FORMATS[args.format]
    raw = src.data.reshape(experts, n, -1)
    out = np.empty((experts, n, k // qk * qbytes), dtype=np.uint8)
    sq_err = sq_w = 0.0
    n_floor = n_groups = 0
    min_s = float("inf")
    for e0 in range(0, experts, args.batch_experts):
        e1 = min(e0 + args.batch_experts, experts)
        r = torch.from_numpy(np.array(raw[e0:e1])).to(device)
        w = decode_nvfp4(r, *tables, n, k)
        w *= glob[e0:e1].to(device).reshape(-1, 1, 1)
        q, s = quantize_group128(w, args.min_scale_exp)
        blocks = encode(q, s)
        # the decode of the written bytes must give exactly q * s (checked here for every expert, not sampled)
        eff = q.float() * s.repeat_interleave(GROUP, dim=-1)
        if not torch.equal(decode(blocks, n, k), eff):
            raise RuntimeError("encoded blocks do not decode to the quantized values")
        sq_err += ((eff - w) ** 2).sum().item()
        sq_w += (w ** 2).sum().item()
        n_floor += (s <= 2.0 ** args.min_scale_exp).sum().item()
        n_groups += s.numel()
        min_s = min(min_s, s.min().item())
        out[e0:e1] = blocks.cpu().numpy()
    stats.update(nmse=sq_err / sq_w, groups=n_groups, groups_at_floor=n_floor, min_scale=min_s,
                 global_scale_min=float(glob.min()), global_scale_max=float(glob.max()), bytes=int(out.nbytes))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", required=True, help="NVFP4 GGUF file(s) that hold the routed experts")
    p.add_argument("--template", nargs="+", help="GGUF file(s) that provide the metadata and every other tensor (default: the source)")
    p.add_argument("--output", required=True)
    p.add_argument("--format", choices=sorted(FORMATS), default="q4_0")
    p.add_argument("--layers", help="comma-separated block indices to convert (default: all that the source has); the others are copied unchanged")
    p.add_argument("--min-scale-exp", type=int, default=-14, help="the group scale floor is 2^N; -14 is the smallest normal FP16 value")
    p.add_argument("--batch-experts", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--manifest", help="write the provenance and statistics to this JSON file (default: OUTPUT.manifest.json)")
    p.add_argument("--hash", action="store_true", help="also record the SHA-256 of the output")
    p.add_argument("--experts-only", action="store_true", help="write only the converted expert tensors (for checks)")
    args = p.parse_args()

    device = torch.device(args.device)
    template = args.template or args.source
    t_readers, t_tensors, t_order = open_readers(template)
    s_readers, s_tensors, _ = open_readers(args.source)
    only = None if not args.layers else {int(x) for x in args.layers.split(",")}
    tables = nvfp4_tables(device)

    # plan: which template tensors are replaced, dropped or copied
    plan = []
    for name in t_order:
        t = t_tensors[name]
        layer = None
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "blk" and parts[2].startswith("ffn_") and "_exps" in parts[2]:
            layer = int(parts[1])
            proj = parts[2][len("ffn_"):-len("_exps")]
            kind = parts[3]
            selected = (only is None or layer in only) and EXPERT_WEIGHT.format(layer, proj) in s_tensors and \
                s_tensors[EXPERT_WEIGHT.format(layer, proj)].tensor_type == gguf.GGMLQuantizationType.NVFP4
            if selected and kind == "weight":
                plan.append(("convert", name, layer, proj))
                continue
            if selected and kind in ("scale", "input_scale"):
                plan.append(("drop", name, layer, proj))
                continue
        plan.append(("copy", name, layer, None))

    if args.experts_only:
        plan = [x for x in plan if x[0] == "convert"]
    arch = t_readers[0].fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(args.output, arch)
    for field in t_readers[0].fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name.startswith("split."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub)

    encode, decode, qtype, qk, qbytes = FORMATS[args.format]
    total = 0
    for kind, name, layer, proj in plan:
        t = t_tensors[name]
        if kind == "drop":
            continue
        if kind == "convert":
            k, n, e = (int(x) for x in t.shape)
            shape, nbytes = (e, n, k // qk * qbytes), e * n * (k // qk * qbytes)
            writer.add_tensor_info(name, shape, np.uint8, nbytes, qtype)
        else:
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        total += nbytes if kind == "convert" else t.data.nbytes
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    manifest = dict(tool="tools/moe-int4-convert/convert.py", format=args.format, rule="absmax/7 RNE-BF16 group scale, clamp(round(w/s),-8,7)",
                    min_scale_exp=args.min_scale_exp, source=[dict(path=str(x), size=os.path.getsize(x)) for x in args.source],
                    template=[dict(path=str(x), size=os.path.getsize(x)) for x in template], tensors=[], dropped=[])
    t0 = time.time()
    done = 0
    for kind, name, layer, proj in plan:
        t = t_tensors[name]
        if kind == "drop":
            manifest["dropped"].append(name)
            continue
        if kind == "convert":
            src = s_tensors[EXPERT_WEIGHT.format(layer, proj)]
            glob = torch.from_numpy(np.array(s_tensors[f"blk.{layer}.ffn_{proj}_exps.scale"].data).reshape(-1))
            stats = dict(name=name, source_type="NVFP4", shape=[int(x) for x in src.shape])
            data = convert_tensor(src, glob, args, device, tables, stats)
            writer.write_tensor_data(data)
            manifest["tensors"].append(stats)
            print(f"{name}: nmse {stats['nmse']:.3e} min scale {stats['min_scale']:.3e} floor {stats['groups_at_floor']}/{stats['groups']} "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        else:
            writer.write_tensor_data(t.data)
        done += 1
    writer.close()

    mse_w = sum(s["nmse"] * s["bytes"] for s in manifest["tensors"]) / max(1, sum(s["bytes"] for s in manifest["tensors"]))
    manifest.update(converted=len(manifest["tensors"]), dropped_count=len(manifest["dropped"]), mean_nmse=mse_w,
                    output=dict(path=str(args.output), size=os.path.getsize(args.output)), seconds=round(time.time() - t0, 1))
    if args.hash:
        manifest["output"]["sha256"] = sha256_file(args.output)
    Path(args.manifest or (str(args.output) + ".manifest.json")).write_text(json.dumps(manifest, indent=1))
    print(f"converted {manifest['converted']} tensors, dropped {manifest['dropped_count']} scale tensors, "
          f"{manifest['output']['size'] / 2**30:.2f} GiB in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
