#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic fixture of ggml-ffn for prefill widths: random activations and routes of a chosen expert-popularity shape.

  make-prefill-fixture.py OUTPUT_DIR ROWS [--skew S] [--seed N]

input.f32 [ROWS, 4096] (normal), ids.i32 [ROWS, 8] (eight distinct experts per row, expert e drawn with weight 1/(e+1)^S in a seeded
permutation of the 288 experts; S = 0 is uniform), router.f32 [ROWS, 8] (softmax of normal logits, normalized, times 2.5).
No reference outputs: compare outputs of two executors with --prefill ... OUTPUT.f32.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("rows", type=int)
    p.add_argument("--skew", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    experts = 288
    perm = rng.permutation(experts)
    w = 1.0 / (np.arange(experts) + 1.0) ** a.skew
    w = w[np.argsort(perm)]
    w /= w.sum()
    ids = np.stack([rng.choice(experts, size=8, replace=False, p=w) for _ in range(a.rows)]).astype(np.int32)
    x = rng.standard_normal((a.rows, 4096)).astype(np.float32)
    logits = rng.standard_normal((a.rows, 8))
    r = np.exp(logits - logits.max(axis=1, keepdims=True))
    r = (r / r.sum(axis=1, keepdims=True) * 2.5).astype(np.float32)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    x.tofile(out / "input.f32")
    ids.tofile(out / "ids.i32")
    r.tofile(out / "router.f32")
    counts = np.bincount(ids.ravel(), minlength=experts)
    print(f"{a.rows} rows: experts used {int((counts > 0).sum())}, rows per used expert mean {counts[counts > 0].mean():.1f} max {counts.max()}")


if __name__ == "__main__":
    main()
