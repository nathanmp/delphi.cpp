#!/usr/bin/env python3
"""Check G128 batch boundaries, graph replay, duplicate routes and strided IDs against independent saved/reference arithmetic."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gguf-py'))
import gguf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--fixture', required=True, help='Saved input/ids/router and independent reference.f32 for the same bank')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='ROCm1')
    args = p.parse_args()
    build, bank, source, output = [Path(s).resolve() for s in (args.build, args.bank, args.fixture, args.output)]
    output.mkdir(parents=True, exist_ok=True)
    reader = gguf.GGUFReader(bank)
    weights = [next(t for t in reader.tensors if t.name == f'blk.0.ffn_{proj}_exps.weight') for proj in ('gate', 'up', 'down')]
    assert all(t.tensor_type == gguf.GGMLQuantizationType.Q4_G128 for t in weights)
    h, _, experts = map(int, weights[0].shape)
    x = np.fromfile(source / 'input.f32', np.float32).reshape(-1, h)
    ids = np.fromfile(source / 'ids.i32', np.int32).reshape(len(x), -1)
    rw = np.fromfile(source / 'router.f32', np.float32).reshape(ids.shape)
    ref = np.fromfile(source / 'reference.f32', np.float32).reshape(x.shape)
    topk = ids.shape[1]
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=f'{build}/bin:/opt/rocm/lib', DELPHI_COMPARE_BINDIR=str(build / 'bin'),
        DELPHI_COMPARE_DEVICE=args.device, DELPHI_COMPARE_TOPK=str(topk), DELPHI_COMPARE_PACE_US='20000',
        GGML_HIP_MOE_INT4_DECODE='1', GGML_HIP_MOE_INT4_BATCH='1')
    for key in ('GGML_HIP_MOE_INT4_PATH', 'DELPHI_COMPARE_CLAMP', 'DELPHI_COMPARE_STRIDED_IDS', 'DELPHI_COMPARE_LIFECYCLE'):
        env.pop(key, None)
    records = []
    with open('/tmp/delphi-representative.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = [(f'w{m}', m, False, False) for m in (5, 17, 63, 64, 65, 1024, 1031)]
        cases += [('strided-w65', 65, True, False), ('duplicate-w65', 65, True, True)]
        for name, m, strided, duplicate in cases:
            d = output / name; d.mkdir(exist_ok=True)
            select = np.arange(m) % len(x)
            xx, ii, ww, rr = x[select], ids[select], rw[select], ref[select]
            if duplicate:
                ii = np.tile(np.where(np.arange(topk) % 2, experts - 1, 0).astype(np.int32), (m, 1))
                spec = importlib.util.spec_from_file_location('g128_reference', ROOT / 'tools/rocm-int4/prepare.py')
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                result = np.zeros((m, topk, h), np.float64)
                for e in np.unique(ii):
                    rows, slots = np.where(ii == e)
                    gate, up, down = [mod.decode(t, e).astype(np.float64) for t in weights]
                    qx = mod.q8(xx[rows]).astype(np.float64)
                    g = (qx @ gate.T).astype(np.float32); u = (qx @ up.T).astype(np.float32)
                    hidden = (g / (1 + np.exp(-g))) * u
                    result[rows, slots] = mod.q8(hidden).astype(np.float64) @ down.T
                rr = np.sum(result * ww[..., None], axis=1).astype(np.float32)
            for filename, a in [('input.f32', xx), ('ids.i32', ii), ('router.f32', ww), ('reference.f32', rr)]:
                a.tofile(d / filename)
            case_env = dict(env)
            if strided: case_env['DELPHI_COMPARE_STRIDED_IDS'] = '1'
            if duplicate: case_env['DELPHI_COMPARE_LIFECYCLE'] = '1'
            command = [str(build / 'rocm-int4-ffn'), str(bank), str(d), '0', str(m), '1', '1']
            with (d / 'run.json').open('w') as out, (d / 'run.log').open('w') as err:
                subprocess.run(command, env=case_env, cwd=ROOT, stdout=out, stderr=err, check=True)
            actual = np.fromfile(d / 'ggml-output.f32', np.float32).reshape(rr.shape)
            a, b = actual.astype(np.float64), rr.astype(np.float64)
            nmse = float(np.sum((a - b)**2) / max(np.sum(b**2), 1e-30))
            record = dict(name=name, nmse=nmse, max_abs=float(np.max(np.abs(a - b))), command=command,
                environment={k: v for k, v in case_env.items() if k.startswith(('GGML_', 'DELPHI_')) or k == 'LD_LIBRARY_PATH'})
            records.append(record)
            (output / 'validation.json').write_text(json.dumps(records, indent=2) + '\n')
            assert np.all(np.isfinite(actual)) and nmse < 1e-7, record
            print(name, nmse, flush=True)


if __name__ == '__main__':
    main()
