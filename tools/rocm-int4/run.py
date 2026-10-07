#!/usr/bin/env python3
"""Measure saved routed FFN graphs under the project lock, with backend and clock provenance."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--fixtures', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--modes', nargs='+', choices=['q4_0', 'vector', 'wide', 'wmma', 'mixed', 'auto'], default=['q4_0', 'vector', 'wmma', 'mixed'])
    p.add_argument('--plain-modes', nargs='+', choices=['q4_0', 'vector', 'wide', 'wmma', 'mixed', 'auto'], default=['vector', 'q4_0'])
    p.add_argument('--widths', nargs='+', type=int, default=[5, 8])
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('delphi_profile', root/'tools/bottlenecks/profile.py')
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    build, bank, fixtures, output = [Path(s).resolve() for s in (args.build, args.bank, args.fixtures, args.output)]
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=f'{build}/bin:/opt/rocm/lib', DELPHI_COMPARE_BINDIR=str(build/'bin'), DELPHI_COMPARE_PACE_US='20000')
    git = lambda *s: subprocess.check_output(['git', *s], cwd=root, text=True).strip()
    paths = [build/'rocm-int4-ffn', build/'bin/libggml-hip.so', build/'bin/libggml-base.so']
    meta = dict(branch=git('branch', '--show-current'), commit=git('rev-parse', 'HEAD'), status=git('status', '--short'),
        binaries={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, arms={})
    arms = []
    for width in args.widths:
        for mode in args.modes:
            arms.append((f'{mode}-w{width}', mode, width, False))
    for mode in args.plain_modes:
        arms.append((f'{mode}-w1-cached', mode, 1, False))
        arms.append((f'{mode}-w1-rotation', mode, 1, True))
    with open('/tmp/delphi-representative.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, mode, width, rotate in arms:
            arm_env = env.copy()
            arm_env['GGML_HIP_MOE_INT4_DECODE'] = '0' if mode == 'q4_0' else '1'
            arm_env['GGML_HIP_MOE_INT4_PATH'] = mode
            model = bank/('matched-q4_0.gguf' if mode == 'q4_0' else 'g128.gguf')
            fixture = fixtures/'rotation.manifest' if rotate else fixtures/f'w{width}'
            cmd = [str(build/'rocm-int4-ffn'), str(model), str(fixture), '0', str(width), '5', '4']
            if rotate: arm_env['DELPHI_COMPARE_ROTATE'] = '1'; cmd.append('--manifest')
            print('running', name, flush=True)
            with (output/f'{name}.jsonl').open('w') as out, (output/f'{name}.log').open('w') as err:
                with mod.GpuClockSampler(0.1) as clocks:
                    result = subprocess.run(cmd, env=arm_env, stdout=out, stderr=err, cwd=root)
            record = dict(command=cmd, env={k: v for k, v in arm_env.items() if k.startswith(('GGML_', 'DELPHI_')) or k == 'LD_LIBRARY_PATH'},
                returncode=result.returncode, clocks=clocks.summary())
            meta['arms'][name] = record
            (output/'run.json').write_text(json.dumps(meta, indent=2)+'\n')
            if result.returncode: raise RuntimeError(f'{name} failed; see {output}/{name}.log')
            if rotate:
                saved = output/name; saved.mkdir(exist_ok=True)
                for line in fixture.read_text().splitlines():
                    _, path = line.split(maxsplit=1)
                    shutil.copyfile(Path(path)/'ggml-output.f32', saved/(Path(path).name+'.f32'))
            else:
                shutil.copyfile(fixture/'ggml-output.f32', output/(name+'.f32'))
            timings = json.loads((output/f'{name}.jsonl').read_text())
            record['median_us'] = statistics.median(timings['device_us'])
            print(name, round(record['median_us'], 3), 'us', flush=True)
            (output/'run.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__': main()
