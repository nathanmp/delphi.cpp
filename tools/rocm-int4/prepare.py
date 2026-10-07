#!/usr/bin/env python3
"""Build one uncalibrated expert-bank probe, slice saved routes, and check W4A8 arithmetic independently."""
import argparse
import ctypes
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf


def q8(x):
    x = np.asarray(x, np.float32)
    groups = x.reshape(*x.shape[:-1], -1, 128)
    scale = np.max(np.abs(groups), axis=-1) / np.float32(127)
    inv = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
    return (np.clip(np.rint(groups * inv[..., None]), -127, 127) * scale[..., None]).reshape(x.shape)


def decode(t, expert):
    k, n, e = map(int, t.shape)
    raw = t.data.reshape(e, n, k // 128, 66)[expert]
    scale = (raw[..., :2].copy().view('<u2').reshape(n, -1).astype(np.uint32) << 16).view(np.float32)
    b = raw[..., 2:].reshape(n, k // 128, 4, 16)
    codes = np.concatenate((b & 15, b >> 4), axis=-1).reshape(n, k)
    return (codes.astype(np.float32) - 8) * np.repeat(scale, 128, axis=-1)


def error(x, ref):
    x, ref = x.astype(np.float64), ref.astype(np.float64)
    return dict(nmse=float(np.sum((x-ref)**2) / max(np.sum(ref**2), 1e-30)), max_abs=float(np.max(np.abs(x-ref))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['bank', 'fixtures', 'reference', 'validate'])
    p.add_argument('--model', required=True)
    p.add_argument('--bindir')
    p.add_argument('--fixture')
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--output', required=True)
    p.add_argument('--widths', default='1,5,8,17')
    p.add_argument('--actual', help='validation output path, with optional {width} placeholder')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reader = gguf.GGUFReader(args.model)
    ts = {t.name: t for t in reader.tensors}
    tensors = [ts[f'blk.{args.layer}.ffn_{proj}_exps.weight'] for proj in ('gate', 'up', 'down')]
    h, ff, experts = map(int, tensors[0].shape)
    if args.mode == 'bank':
        assert args.bindir
        lib = ctypes.CDLL(str(Path(args.bindir).resolve() / 'libggml-base.so'))
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        gtype = gguf.GGMLQuantizationType.Q4_G128
        writers = [gguf.GGUFWriter(str(out / name), 'qwen4exp') for name in ('g128.gguf', 'matched-q4_0.gguf')]
        banks = []
        for t in tensors:
            k, n, e = map(int, t.shape)
            raw = np.memmap(out / (t.name + '.g128'), mode='w+', dtype=np.uint8, shape=(e, n, k//128*66))
            matched = np.memmap(out / (t.name + '.q4_0'), mode='w+', dtype=np.uint8, shape=(e, n, k//32*18))
            for expert in range(e):
                source = np.ascontiguousarray(gguf.quants.dequantize(t.data.reshape(e, n, -1)[expert], t.tensor_type), dtype=np.float32)
                dst = raw[expert]
                size = lib.ggml_quantize_chunk(int(gtype), source.ctypes.data, dst.ctypes.data, 0, n, k, None)
                assert size == dst.nbytes
                groups = dst.reshape(n, k//128, 66)
                scale = (groups[..., :2].copy().view('<u2').reshape(n, -1).astype(np.uint32) << 16).view(np.float32)
                fp16 = scale.astype(np.float16)
                assert np.array_equal(fp16.astype(np.float32), scale), 'matched Q4_0 scale is not exact'
                q4 = matched[expert].reshape(n, k//128, 4, 18)
                q4[..., :2] = np.repeat(fp16[..., None], 4, axis=-1).copy().view(np.uint8).reshape(n, k//128, 4, 2)
                q4[..., 2:] = groups[..., 2:].reshape(n, k//128, 4, 16)
            raw.flush(); matched.flush()
            banks.append((raw, matched))
            for writer, data, dtype in zip(writers, (raw, matched), (gtype, gguf.GGMLQuantizationType.Q4_0)):
                writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, dtype)
            print('converted one layer:', t.name, flush=True)
        for writer, which in zip(writers, (0, 1)):
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
            for pair in banks: writer.write_tensor_data(pair[which])
            writer.close()
        (out / 'bank.json').write_text(json.dumps(dict(source=args.model, layer=args.layer, h=h, f=ff, experts=experts,
            policy='Uncalibrated requantization of existing GGUF for execution testing only; other model tensors are not written.'), indent=2)+'\n')
        return
    assert args.fixture
    fixture = Path(args.fixture)
    meta = json.loads((fixture / 'fixture.json').read_text())
    topk = meta['topk']
    x = np.fromfile(fixture / 'input.f32', np.float32).reshape(-1, h)
    ids = np.fromfile(fixture / 'ids.i32', np.int32).reshape(-1, topk)
    rw = np.fromfile(fixture / 'router.f32', np.float32).reshape(-1, topk)
    for m in map(int, args.widths.split(',')):
        d = out / f'w{m}'; d.mkdir(exist_ok=True)
        xx, ii, ww = x[:m], ids[:m], rw[:m]
        assert len(xx) == m
        if args.mode == 'fixtures':
            for name, a in [('input.f32', xx), ('ids.i32', ii), ('router.f32', ww)]: a.tofile(d / name)
            counts = np.bincount(ii.reshape(-1), minlength=experts)
            (d/'fixture.json').write_text(json.dumps(dict(width=m, topk=topk, unique_experts=int(np.count_nonzero(counts)),
                rows_per_expert={str(c): int(np.sum(counts == c)) for c in np.unique(counts) if c},
                unique_weight_bytes=int(np.count_nonzero(counts)*3*h*ff*66//128)), indent=2)+'\n')
        elif args.mode == 'reference':
            slots = np.zeros((m, topk, h), np.float64)
            for expert in np.unique(ii):
                rows, which = np.where(ii == expert)
                gate, up, down = [decode(t, expert).astype(np.float64) for t in tensors]
                inp = q8(xx[rows]).astype(np.float64)
                g = (inp @ gate.T).astype(np.float32); u = (inp @ up.T).astype(np.float32)
                hidden = ((g / (1 + np.exp(-g))) * u).astype(np.float32)
                slots[rows, which] = q8(hidden).astype(np.float64) @ down.T
            np.sum(slots * ww[..., None], axis=1).astype(np.float32).tofile(d/'reference.f32')
        else:
            actual_path = Path(args.actual.format(width=m)) if args.actual else d/'ggml-output.f32'
            actual = np.fromfile(actual_path, np.float32)
            ref = np.fromfile(d/'reference.f32', np.float32)
            assert actual.shape == ref.shape and np.all(np.isfinite(actual))
            result = dict(actual=str(actual_path), **error(actual, ref))
            assert result['nmse'] < 1e-7, (m, result)
            (d/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
            print(m, result)
    if args.mode == 'fixtures':
        manifest = []
        for i in range(len(x)):
            d = out / f'r{i:02d}'; d.mkdir(exist_ok=True)
            for name, a in [('input.f32', x[i:i+1]), ('ids.i32', ids[i:i+1]), ('router.f32', rw[i:i+1])]: a.tofile(d / name)
            manifest.append(f'1 {d.resolve()}')
        (out / 'rotation.manifest').write_text('\n'.join(manifest)+'\n')


if __name__ == '__main__': main()
