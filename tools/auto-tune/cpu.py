"""CPU budgets for the tuner and all descendants of a server launch."""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

from config import TuneError
import knobs as K


def parse_list(value: str) -> set[int]:
    cpus = set()
    try:
        for part in value.split(','):
            bounds = [int(n) for n in part.split('-')]
            if not 1 <= len(bounds) <= 2 or min(bounds) < 0 or bounds[0] > bounds[-1]:
                raise ValueError()
            cpus.update(range(bounds[0], bounds[-1] + 1))
    except ValueError:
        raise TuneError(f'invalid CPU list: {value}') from None
    return cpus


def cpu_list(cpus) -> str:
    return ','.join(str(n) for n in sorted(cpus))


def topology(allowed=None, root=Path('/sys/devices/system/cpu')) -> dict[int, tuple[int, int, int]]:
    out = {}
    for cpu in sorted(os.sched_getaffinity(0) if allowed is None else allowed):
        path = root / f'cpu{cpu}'
        try:
            package = int((path / 'topology/physical_package_id').read_text())
            core = int((path / 'topology/core_id').read_text())
            nodes = sorted(path.glob('node[0-9]*'))
            node = int(nodes[0].name[4:]) if nodes else package
        except OSError as error:
            raise TuneError(f'cannot read CPU {cpu} topology: {error}') from error
        out[cpu] = (package, core, node)
    return out


@dataclass(frozen=True)
class CpuBudget:
    cpus: tuple[int, ...]
    cores: int
    nodes: tuple[int, ...]

    @classmethod
    def resolve(cls, cpus=None, max_cores=None, facts=None):
        facts = topology() if facts is None else facts
        if (cpus is None) == (max_cores is None):
            raise TuneError('give exactly one of --cpus / cpus or --max-cores / max_cores')
        if cpus is not None:
            chosen = parse_list(cpus)
            if not chosen or not chosen <= facts.keys():
                raise TuneError(f'CPU budget is outside the process affinity: {cpu_list(chosen - facts.keys())}')
        else:
            if not isinstance(max_cores, int) or isinstance(max_cores, bool) or max_cores < 1:
                raise TuneError('max_cores must be a positive integer')
            # One logical CPU per physical core; alternate NUMA nodes before taking a second core from a node.
            groups = {}
            seen = set()
            for cpu, (package, core, node) in sorted(facts.items()):
                if (package, core) not in seen:
                    groups.setdefault(node, []).append(cpu)
                    seen.add((package, core))
            ordered = [group[i] for i in range(max(map(len, groups.values())))
                       for _, group in sorted(groups.items()) if i < len(group)]
            chosen = set(ordered[:max_cores])
        return cls(tuple(sorted(chosen)), len({facts[c][:2] for c in chosen}), tuple(sorted({facts[c][2] for c in chosen})))

    def threads(self, devices: int, limit=None) -> int:
        count = min(self.cores, limit or self.cores, max(1, self.cores - 2 * devices))
        return 63 if count == 64 else count

    def apply(self, command, devices, limit=None):
        cmd = tuple(command)
        # Remove all aliases so an old mask or thread count cannot override the budget later in the command.
        aliases = ('-t', '--threads', '-tb', '--threads-batch', '-td', '--threads-draft', '--spec-draft-threads',
                   '-tbd', '--threads-batch-draft', '--spec-draft-threads-batch', '-C', '--cpu-mask', '-Cr', '--cpu-range',
                   '-Cb', '--cpu-mask-batch', '-Crb', '--cpu-range-batch', '-Cd', '--cpu-mask-draft', '--spec-draft-cpu-mask',
                   '-Crd', '--cpu-range-draft', '--spec-draft-cpu-range', '-Cbd', '--cpu-mask-batch-draft',
                   '--spec-draft-cpu-mask-batch', '-Crbd', '--cpu-range-batch-draft', '--spec-draft-cpu-range-batch')
        for alias in aliases:
            cmd = K.remove_option(cmd, alias)
        threads = str(self.threads(devices, limit))
        mask = hex(sum(1 << c for c in self.cpus))
        for name, value in (('-t', threads), ('-tb', threads), ('--cpu-mask', mask), ('--cpu-mask-batch', mask)):
            cmd = K.set_option(cmd, name, value)
        if K.option_value(cmd, '--spec-type'):
            for name, value in (('--spec-draft-threads', threads), ('--spec-draft-threads-batch', threads),
                                ('--spec-draft-cpu-mask', mask), ('--spec-draft-cpu-mask-batch', mask)):
                cmd = K.set_option(cmd, name, value)
        if K.option_value(cmd, '--numa') is None and len(self.nodes) > 1:
            cmd = K.set_option(cmd, '--numa', 'tensors')
        return cmd

    @contextlib.contextmanager
    def affinity(self):
        previous = os.sched_getaffinity(0)
        if not set(self.cpus) <= previous:
            raise TuneError('the CPU budget is no longer inside the process affinity')
        os.sched_setaffinity(0, self.cpus)
        try:
            yield
        finally:
            os.sched_setaffinity(0, previous)

    def summary(self):
        return {'cpus': cpu_list(self.cpus), 'physical_cores': self.cores, 'numa_nodes': list(self.nodes)}
