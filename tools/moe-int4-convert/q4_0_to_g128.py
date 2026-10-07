#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rewrite the Q4_0 tensors that satisfy the group-128 contract of a GGUF as Q4_G128 (a byte shuffle, no requantization).

The contract: the four successive block scales of every group of 128 values are equal and exactly BF16 (so also exact in FP16 above
2^-14). Used to turn the matched Q4_0 banks of the replay experiments (tools/cmp-int4-compare/compare.py) into the type that the
INT4 decode path stores. Tensors of other types are copied. Fails if a tensor breaks the contract.

  q4_0_to_g128.py INPUT.gguf OUTPUT.gguf [--only-experts]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402


def to_g128(raw):
    """raw: uint8 [..., K/32*18] of Q4_0 rows -> uint8 [..., K/128*66]."""
    lead = raw.shape[:-1]
    b = raw.reshape(*lead, -1, 4, 18)
    d = np.ascontiguousarray(b[..., :2]).view(np.float16).astype(np.float32)[..., 0]      # [..., groups, 4]
    if not (d == d[..., :1]).all():
        raise SystemExit("the four block scales of a group of 128 differ")
    bits = d[..., 0].view(np.uint32)
    if (bits & 0xFFFF).any() or not np.isfinite(d).all():
        raise SystemExit("a group scale is not exactly BF16")
    scale = (bits >> 16).astype(np.uint16).view(np.uint8).reshape(*lead, -1, 2)
    codes = b[..., 2:].reshape(*lead, -1, 64)
    return np.concatenate((scale, codes), axis=-1).reshape(*lead, -1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--only-experts", action="store_true", help="write only the converted tensors")
    args = p.parse_args()
    reader = gguf.GGUFReader(args.input, "r")
    writer = gguf.GGUFWriter(args.output, reader.fields["general.architecture"].contents())
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name.startswith("split."):
            continue
        vtype = field.types[0]
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None)
    plan = []
    for t in reader.tensors:
        convert = t.tensor_type == gguf.GGMLQuantizationType.Q4_0 and "_exps" in t.name
        if args.only_experts and not convert:
            continue
        plan.append((t, convert))
        if convert:
            k = int(t.shape[0])
            assert k % 128 == 0
            lead = tuple(int(x) for x in t.data.shape[:-1])
            writer.add_tensor_info(t.name, lead + (k // 128 * 66,), np.uint8, int(np.prod(lead)) * (k // 128 * 66), gguf.GGMLQuantizationType.Q4_G128)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t, convert in plan:
        writer.write_tensor_data(to_g128(np.asarray(t.data)) if convert else t.data)
        print(("converted " if convert else "copied ") + t.name, flush=True)
    writer.close()


if __name__ == "__main__":
    main()
