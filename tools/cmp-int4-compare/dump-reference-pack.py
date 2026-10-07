#!/usr/bin/env python3
"""Cut experts of a CMP-packed layer (marlin-int4.pt) into raw files for ggml-ffn --verify-pack.

The .pt file is the output of the offline packer (tools/cmp-int4-compare/compare.py prepare on the experiment branch).
This tool needs torch only to read it; inference does not.
"""
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", help="marlin-int4.pt")
    parser.add_argument("output", help="output directory")
    parser.add_argument("experts", type=int, nargs="+")
    args = parser.parse_args()
    packed = torch.load(args.bank, weights_only=True, map_location="cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # W13 joins the already packed gate and up columns, as the CMP comparison does
    w13 = torch.cat((packed["gate"]["weight"], packed["up"]["weight"]), dim=2)
    s13 = torch.cat((packed["gate"]["scale"], packed["up"]["scale"]), dim=2)
    parts = dict(w13=w13, s13=s13, w2=packed["down"]["weight"], s2=packed["down"]["scale"])
    for expert in args.experts:
        for name, tensor in parts.items():
            data = tensor[expert].contiguous()
            raw = data.view(torch.int16) if data.dtype == torch.bfloat16 else data
            (out / f"{name}_e{expert}.bin").write_bytes(raw.numpy().tobytes())
            print(name, expert, tuple(data.shape), data.dtype)


if __name__ == "__main__":
    main()
