#!/usr/bin/env python3
"""Create static expert frequencies from LLAMA_MOE_ROUTING_LOG records."""
import argparse
from collections import Counter
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
from gguf import GGUFReader

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', required=True)
p.add_argument('--routes', required=True)
p.add_argument('--start-offset', type=int, default=0, help='Byte offset after warmup/prompt records')
p.add_argument('--output', required=True)
a = p.parse_args()
r = GGUFReader(a.model)
weights = {t.name: t for t in r.tensors}
n_layers = max(int(t.name.split('.')[1]) for t in r.tensors if t.name.startswith('blk.')) + 1
sample = next(t for t in r.tensors if t.name.endswith('ffn_gate_exps.weight'))
h, f, experts = map(int, sample.shape)
counts = Counter()
records = 0
with open(a.routes, 'rb') as stream:
    stream.seek(a.start_offset)
    while hdr := stream.read(12):
        if len(hdr) != 12:
            raise ValueError('truncated route header')
        il, rows, topk = struct.unpack('<3i', hdr)
        assert 0 <= il < n_layers and 0 < rows <= 16 and 0 < topk <= experts
        values = stream.read(rows * topk * 4)
        assert len(values) == rows * topk * 4
        ids = struct.unpack('<' + str(rows * topk) + 'i', values)
        assert all(0 <= e < experts for e in ids)
        assert f'blk.{il}.ffn_gate_exps.weight' in weights
        counts.update((il, e) for e in ids)
        records += 1
assert counts
with open(a.output, 'w') as out:
    out.write(f'DELPHI_MOE_PROFILE_V1 {h} {f} {experts} {n_layers}\n')
    for (il, e), count in sorted(counts.items()):
        out.write(f'{il} {e} {count}\n')
size = 3 * h * f * 66 // 128
ranked = sorted(counts.values(), reverse=True)
for gib in (4, 8):
    slots = gib * 2**30 // size
    covered = sum(ranked[:slots]) / sum(ranked)
    print(f'{gib} GiB: {slots} experts, training selection coverage {covered:.1%}')
print(f'{records} layer records, {sum(counts.values())} selections; test on separate inputs')
