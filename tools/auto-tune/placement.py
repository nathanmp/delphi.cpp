"""Placement candidates from free VRAM and no-allocation llama-fit-params measurements."""
from __future__ import annotations

import math
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from config import TuneError
import knobs as K

ROOT = Path(__file__).resolve().parents[2]
MiB = 2**20


@dataclass(frozen=True)
class Device:
    name: str
    free_mib: float
    margin_mib: float = 1536
    fill: str = 'fill'
    max_used_mib: float | None = None

    @property
    def capacity(self):
        return max(0, min(self.free_mib - self.margin_mib, self.max_used_mib if self.max_used_mib is not None else math.inf))


def devices_from_state(command, policies, pci_map, state, margin=1536):
    names = list(policies) or (K.option_value(command, '--device') or K.option_value(command, '-dev') or '').split(',')
    out = []
    for name in names:
        if name not in pci_map or pci_map[name] not in state:
            raise TuneError(f'{name}: no free VRAM reading')
        used, total = state[pci_map[name]]
        policy = policies.get(name, {})
        out.append(Device(name, total - used, policy.get('min_free_mib', max(margin, 2048 if name == 'ROCm2' else 0)),
                          policy.get('fill', 'fill'), policy.get('max_used_mib')))
    return out


def model_info(path):
    # Use the project reader. It maps tensor data but only the metadata is read here.
    sys.path.insert(0, str(ROOT / 'gguf-py'))
    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise TuneError('placement needs the gguf-py dependencies (including numpy) in the Python environment') from error
    reader = GGUFReader(path)

    def field(name, default=None):
        item = reader.fields.get(name)
        return item.contents() if item else default

    arch = field('general.architecture')
    layers = int(field(f'{arch}.block_count', 0))
    nextn = int(field(f'{arch}.nextn_predict_layers', 0))
    count = int(field('split.count', 1))
    paths = [Path(path)]
    if count > 1:
        match = re.fullmatch(r'(.*)-[0-9]{5}-of-[0-9]{5}\.gguf', str(path))
        if not match:
            raise TuneError('split GGUF needs a numbered shard path')
        paths = [Path(f'{match[1]}-{i:05d}-of-{count:05d}.gguf') for i in range(1, count + 1)]
    mtp_bytes = 0
    total_bytes = 0
    expert_layers = set()
    for index, p in enumerate(paths):
        part = reader if index == 0 else GGUFReader(p)
        for tensor in part.tensors:
            total_bytes += tensor.n_bytes
            match = re.match(r'blk\.(\d+)\.', tensor.name)
            if match and nextn and int(match[1]) >= layers - nextn:
                mtp_bytes += tensor.n_bytes
            elif match and re.search(r'\.ffn_(?:gate|up|down|gate_up)_(?:ch)?exps\.', tensor.name):
                expert_layers.add(int(match[1]))
    return {'architecture': arch, 'experts': int(field(f'{arch}.expert_count', 0)), 'layers': layers,
            'expert_layers': sorted(expert_layers), 'mtp_mib': mtp_bytes / MiB, 'nextn': nextn,
            'files': [str(p) for p in paths], 'model_mib': total_bytes / MiB}


# Spec and HTTP options are server-only. The standalone fit tool does not fit an extra draft model.
FIT_VALUES = {'-m', '--model', '-c', '--ctx-size', '-b', '--batch-size', '-ub', '--ubatch-size', '-t', '--threads',
              '-tb', '--threads-batch', '--device', '-dev', '-ngl', '--n-gpu-layers', '-ts', '--tensor-split',
              '-sm', '--split-mode', '-ot', '--override-tensor', '-ncmoe', '--n-cpu-moe', '--numa', '-fa', '--flash-attn',
              '-lm', '--load-mode', '-lzm', '--lazy-mode', '-ctk', '--cache-type-k', '-ctv', '--cache-type-v',
              '--cpu-mask', '--cpu-mask-batch', '-mg', '--main-gpu'}
FIT_SWITCHES = {'-cmoe', '--cpu-moe', '-nr', '--no-repack', '--repack'}


def fit_command(command):
    out = [str(Path(command[0]).parent / 'llama-fit-params')]
    i = 1
    while i < len(command):
        key = command[i].split('=', 1)[0]
        if key in FIT_VALUES:
            if '=' in command[i]:
                out.extend(command[i].split('=', 1))
            else:
                out.extend(command[i:i+2])
                i += 1
        elif key in FIT_SWITCHES:
            out.append(key)
        i += 1
    return [*out, '--fit-print', 'on']


def parse_fit(text):
    rows = {}
    for line in text.splitlines():
        match = re.fullmatch(r'\s*(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*', line)
        if match:
            rows[match[1]] = dict(zip(('model', 'context', 'compute'), map(int, match.groups()[1:])))
    if 'Host' not in rows:
        raise TuneError('llama-fit-params returned no memory table')
    return rows


def strip_placement(command):
    cmd = tuple(command)
    for name in ('-ngl', '--n-gpu-layers', '-ts', '--tensor-split', '-ncmoe', '--n-cpu-moe'):
        cmd = K.remove_option(cmd, name)
    for name in ('-cmoe', '--cpu-moe'):
        cmd = K.remove_option(cmd, name, has_value=False)
    return cmd


def distribute(weights, capacities, devices):
    """Use fill devices in priority order, then use only the residual on min devices."""
    allocations = {d.name: 0.0 for d in devices}
    remaining = weights
    for rule in ('fill', 'min'):
        for device in devices:
            if device.fill == rule:
                amount = min(max(0, remaining), max(0, capacities[device.name]))
                allocations[device.name] = amount
                remaining -= amount
    return allocations, max(0, remaining)


def candidates(command, env, devices, full, dense, info, extra=None):
    extra = extra or {}
    capacities = {d.name: d.capacity - full[d.name]['context'] - full[d.name]['compute'] - extra.get(d.name, 0) for d in devices}
    model_mib = sum(full[d.name]['model'] for d in devices)
    dense_mib = sum(dense[d.name]['model'] for d in devices)
    full_alloc, deficit = distribute(model_mib, capacities, devices)
    dense_capacities = {d.name: d.capacity - dense[d.name]['context'] - dense[d.name]['compute'] - extra.get(d.name, 0) for d in devices}
    dense_alloc, dense_deficit = distribute(dense_mib, dense_capacities, devices)
    cmd = strip_placement(command)
    warnings = []
    if env.get('GGML_SCHED_STAGE_REGISTER') == '1':
        warnings.append('GGML_SCHED_STAGE_REGISTER=1 uses memory outside the fit estimate; increase the device margins and verify the load')
    moe = bool(info['experts'])
    if deficit <= 0:
        mode, allocation = 'fits', full_alloc
    elif moe and dense_deficit <= 0:
        mode, allocation = 'host-experts', dense_alloc
    else:
        mode, allocation = ('dense-partial' if not moe else 'poor-fit'), full_alloc
        if mode == 'poor-fit':
            warnings.append('dense weights and buffers do not fit well; use mainline partial offload')
    host = mode == 'host-experts'
    active = [d for d in devices if allocation[d.name] > 0 or (host and d.fill == 'fill')]
    if not active:
        raise TuneError('no device has space for the model and its buffers at the context floor')
    cmd = K.set_option(cmd, '--device', ','.join(d.name for d in active))
    cmd = K.remove_option(cmd, '-dev')
    # Split ratios are model bytes, after compute/KV/draft buffers have been reserved.
    if len(active) > 1 and mode in ('fits', 'host-experts'):
        cmd = K.set_option(cmd, '-ts', ','.join(f'{allocation[d.name]:.3f}' for d in active))
    cmd = K.set_option(cmd, '--fit', 'on')
    cmd = K.set_option(cmd, '--fit-target', ','.join(str(math.ceil(d.margin_mib)) for d in active))
    if mode in ('fits', 'host-experts'):
        cmd = K.set_option(cmd, '-ngl', '999')
    e = dict(env)
    if host:
        cmd = (*cmd, '-cmoe')
        e.pop('LLAMA_MOE_CACHE_TIERS', None)
        e['LLAMA_MOE_CACHE_MIB'] = 'auto'
        # Auto sizing reads one margin for all GPUs, after KV, compute and draft allocation.
        # Use the strictest device bound; a min device is not allowed an extra cache tier.
        margins = [max(d.margin_mib, d.free_mib - d.max_used_mib if d.max_used_mib is not None else 0) for d in active]
        margins += [max(0, d.free_mib - allocation[d.name] - dense[d.name]['context'] - dense[d.name]['compute'] - extra.get(d.name, 0))
                    for d in active if d.fill == 'min']
        e['LLAMA_MOE_CACHE_AUTO_MARGIN_MIB'] = str(math.ceil(max(margins)))
        e.setdefault('GGML_SCHED_STAGE', '1')
        e.setdefault('GGML_SCHED_STAGE_MIN_TOKENS', '1024')
        e.setdefault('LLAMA_MOE_CACHE_PROMPT_ROWS', '1023')
        if any(d.fill == 'min' for d in active):
            warnings.append('the common auto-cache margin protects min devices; this can leave fill-device memory unused')
    else:
        e['LLAMA_MOE_CACHE_MIB'] = '0'
        e.pop('LLAMA_MOE_CACHE_TIERS', None)
    result = [{'name': mode, 'command': list(cmd), 'env': e, 'warnings': warnings}]
    expert_mib = model_mib - dense_mib
    if host and expert_mib > 0:
        fraction = deficit / expert_mib
        layers = info['expert_layers']
        # --n-cpu-moe applies to the first N transformer layers, including any initial dense layers.
        count = math.ceil(len(layers) * fraction)
        if 0 < fraction < 0.25 and 0 < count <= len(layers):
            cpu_cmd = K.remove_option(cmd, '-cmoe', has_value=False)
            cpu_cmd = K.set_option(cpu_cmd, '--n-cpu-moe', str(layers[count - 1] + 1))
            resident = model_mib - expert_mib * count / len(layers)
            alloc, rest = distribute(resident, capacities, devices)
            if rest <= 0:
                cpu_cmd = K.set_option(cpu_cmd, '--device', ','.join(d.name for d in devices))
                if len(devices) > 1:
                    cpu_cmd = K.set_option(cpu_cmd, '-ts', ','.join(f'{alloc[d.name]:.3f}' for d in devices))
                result.append({'name': 'cpu-expert-layers', 'command': list(cpu_cmd), 'env': dict(e),
                               'warnings': ['layer sizes are estimated; the load must verify this candidate']})
    return result


def prepare(cfg, command, env, pci_map, state, cpu, probe=None):
    """No-allocation measurements first; candidates are verified by the normal profiler launches."""
    import os
    devices = devices_from_state(command, cfg['devices'], pci_map, state, int(env.get('LLAMA_MOE_CACHE_AUTO_MARGIN_MIB', '1536')))
    info = model_info(K.option_value(command, '-m') or K.option_value(command, '--model'))
    out = Path(cfg['output']) / 'placement'
    out.mkdir(parents=True, exist_ok=True)
    commands = []

    def measure(cmd, label):
        fit = fit_command(cmd)
        commands.append({'name': label, 'command': fit})
        if probe:
            return probe(fit, label)
        with (out / f'{label}.stderr.log').open('w') as log, cpu.affinity():
            run = subprocess.run(fit, env={**os.environ, **env}, cwd=ROOT, stdout=subprocess.PIPE, stderr=log, text=True, timeout=600)
        (out / f'{label}.stdout.log').write_text(run.stdout)
        if run.returncode:
            raise TuneError(f'placement dry run {label} failed ({run.returncode}); see {out / (label + ".stderr.log")}')
        return parse_fit(run.stdout)

    cmd = K.set_option(strip_placement(command), '--device', ','.join(d.name for d in devices))
    cmd = K.remove_option(cmd, '-dev')
    cmd = K.set_option(cmd, '-ngl', '999')
    full = measure(cmd, 'full')
    dense = measure((*cmd, '-cmoe'), 'dense') if info['experts'] else full
    extra = dict(cfg['placement']['reserve_mib'])
    spec = K.option_value(command, '--spec-type') or ''
    mtp_reserve = 0
    if 'draft-mtp' in spec:
        # The fit CLI cannot create an MTP context. Reserve its weights and a conservative scratch-buffer bound.
        mtp_device = devices[-1].name
        mtp_reserve = info['mtp_mib'] + max(r['compute'] for n, r in full.items() if n != 'Host')
        mtp_reserve += sum(r['context'] for n, r in full.items() if n != 'Host') * max(1, info['nextn']) / max(1, info['layers'])
        extra[mtp_device] = extra.get(mtp_device, 0) + mtp_reserve
    draft = K.option_value(command, '--spec-draft-model')
    draft_estimate = None
    if draft:
        dcmd = K.set_option(strip_placement(command), '-m', draft)
        dcmd = K.remove_option(dcmd, '--model')
        dcmd = K.set_option(dcmd, '--device', K.option_value(command, '--spec-draft-device') or devices[0].name)
        dcmd = K.remove_option(dcmd, '-dev')
        dcmd = K.set_option(dcmd, '-ngl', K.option_value(command, '--spec-draft-ngl') or '999')
        dcmd = K.set_option(dcmd, '-ub', K.option_value(command, '-ubd') or '512')
        # Target-only lazy embedding overrides do not describe the draft model.
        dcmd = K.remove_option(K.remove_option(dcmd, '-ot'), '--override-tensor')
        draft_info = model_info(draft)
        if draft_info['architecture'] in ('dflash', 'eagle3', 'gemma4-assistant'):
            # These contexts require ctx_other, which the standalone fit CLI cannot supply.
            draft_devices = K.option_value(dcmd, '--device').split(',')
            if len(draft_devices) != 1:
                raise TuneError('coupled draft estimation currently needs one draft device')
            device = draft_devices[0]
            draft_rows = {device: {'model': draft_info['model_mib'],
                                  'context': sum(r['context'] for n, r in dense.items() if n != 'Host'),
                                  'compute': max(r['compute'] for n, r in dense.items() if n != 'Host')}}
            draft_estimate = {'architecture': draft_info['architecture'], 'memory': draft_rows,
                              'reason': 'the draft context requires a loaded target; reserve its weights and a target-buffer bound'}
        else:
            draft_rows = measure(dcmd, 'draft')
        for name, row in draft_rows.items():
            if name != 'Host':
                extra[name] = extra.get(name, 0) + sum(row.values())
    if K.option_value(command, '--mmproj') and not cfg['placement']['reserve_mib']:
        raise TuneError('mmproj needs placement.reserve_mib from a server memory report; the fit CLI cannot measure it')
    options = candidates(command, env, devices, full, dense, info, extra)
    if mtp_reserve:
        for _ in range(len(devices)):
            last = K.option_value(options[0]['command'], '--device').split(',')[-1]
            if last == mtp_device:
                break
            extra[mtp_device] -= mtp_reserve
            extra[last] = extra.get(last, 0) + mtp_reserve
            mtp_device = last
            options = candidates(command, env, devices, full, dense, info, extra)
    # Layer boundaries round a byte ratio. Measure the derived split before paying for a full model load.
    corrections = {}
    for attempt in range(3):
        first = options[0]
        if first['name'] != 'fits' or len(devices) < 2:
            break
        projected = measure(first['command'], f'split-{attempt}')
        first['projected'] = projected
        overflow = {d.name: sum(projected.get(d.name, {}).values()) + extra.get(d.name, 0) - d.capacity for d in devices}
        overflow = {name: value for name, value in overflow.items() if value > 0}
        if not overflow:
            break
        if attempt == 2:
            first['warnings'].append('the split estimate still exceeds a device budget; fresh-load verification can reject it')
            break
        for name, value in overflow.items():
            corrections[name] = corrections.get(name, 0) + value + 1
        options = candidates(command, env, devices, full, dense, info,
                             {name: extra.get(name, 0) + corrections.get(name, 0) for name in set(extra) | set(corrections)})
    for option in options:
        if 'draft-mtp' in spec:
            option['warnings'].append('MTP scratch memory is a conservative estimate; verify the fresh launch')
        if draft_estimate:
            option['warnings'].append('coupled draft buffers use a conservative target-buffer estimate; verify the fresh launch')
        if K.option_value(command, '--mmproj'):
            option['warnings'].append('check mmproj load order: auto cache sizing must run after mmproj allocation')
    result = {'devices': [asdict(d) for d in devices], 'model': info, 'full': full, 'dense': dense,
              'extra_mib': extra, 'draft_estimate': draft_estimate, 'split_reserve_mib': corrections, 'dry_runs': commands, 'candidates': options}
    (out / 'plan.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
