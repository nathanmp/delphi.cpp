"""Acceptance replay and pass costs. Replay estimates need a fresh real-acceptance confirmation."""
from __future__ import annotations

import itertools
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import knobs as K


class SpecError(ValueError):
    pass


class Unavailable(SpecError):
    """A server capability cannot supply replay inputs. Keep the normal measurements."""


@dataclass(frozen=True)
class Policy:
    n_max: int
    n_min: int = 0
    p_min: float = 0.0
    adaptive: str = 'off'
    n_min_adaptive: int = 1
    n_start: int = 0

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.n_max, self.n_min, self.n_min_adaptive, self.n_start))
                or self.n_max < 1 or not 0 <= self.n_min <= self.n_max or not 1 <= self.n_min_adaptive <= self.n_max
                or self.n_start < 0 or type(self.p_min) not in (int, float) or not math.isfinite(self.p_min) or not 0 <= self.p_min <= 1
                or self.adaptive not in ('off', 'bucket')):
            raise SpecError('invalid fixed/bucket replay policy')

    def args(self):
        out = ['--spec-draft-n-max', str(self.n_max), '--spec-draft-n-min', str(self.n_min),
               '--spec-draft-p-min', str(self.p_min), '--spec-draft-n-start', str(self.n_start),
               '--spec-draft-n-min-adaptive', str(self.n_min_adaptive)]
        return out + (['--spec-draft-adaptive-mode', 'bucket'] if self.adaptive == 'bucket' else ['--no-spec-draft-adaptive'])

    def command(self, command):
        cmd = tuple(command)
        for flag in ('--spec-draft-adaptive', '--no-spec-draft-adaptive'):
            cmd = K.remove_option(cmd, flag, has_value=False)
        cmd = K.remove_option(cmd, '--spec-draft-adaptive-mode')
        args = self.args()
        for i in range(0, 10, 2):
            cmd = K.set_option(cmd, args[i], args[i + 1])
        return (*cmd, *args[10:])


class Bucket:
    """common/speculative-adaptive.h; state starts again at each request."""
    def __init__(self, policy):
        self.p = policy
        self.depth = min(policy.n_max, max(policy.n_min_adaptive, policy.n_start or policy.n_max - 3))
        self.credit = self.pressure()

    def pressure(self):
        return max(60, self.depth * 10)

    def cap(self):
        return self.pressure() + 20 + 6 * (self.depth - 1)

    def update(self, accepted):
        self.credit += max(1, accepted - 1) if accepted >= self.depth else accepted - self.depth
        if self.credit >= self.cap():
            if self.depth < self.p.n_max:
                self.credit = min(self.credit, self.cap() + self.depth) - self.cap()
                self.depth += 1
                self.credit += self.pressure()
            else:
                self.credit = self.cap()
        elif self.credit <= 0:
            if self.depth > self.p.n_min_adaptive:
                self.credit -= self.pressure()
                self.depth -= 1
                self.credit += self.cap()
            else:
                self.credit = 0


def read_plog(text, cap):
    """Read the existing MTP JSONL. The caller supplies one request boundary and an uncensored capture cap."""
    rows, tails = [], 0
    for line_no, line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(line)
            n, acc, probs = row['n'], row['acc'], row['p']
            if (type(n) is not int or type(acc) is not int or not 0 <= acc <= n <= cap or not isinstance(probs, list)
                    or len(probs) not in (n, n + 1) or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs)):
                raise ValueError('invalid counts or probabilities')
        except (ValueError, TypeError, KeyError) as error:
            raise SpecError(f'probability log line {line_no}: {error}') from error
        # With p_min=0 and adaptation disabled, only the output/context boundary can shorten a draft.
        if n != cap:
            tails += 1
            continue
        if len(probs) != cap:
            raise SpecError(f'probability log line {line_no}: capture was confidence-censored')
        rows.append({'n': n, 'acc': acc, 'p': probs})
    if not rows:
        raise SpecError('no full-depth passes in the probability log')
    return {'passes': rows, 'boundary_passes_omitted': tails}


def cost_table(trace, windows, settle=3, minimum=3):
    """Median verify-to-verify wall time, within each fixed-width request. Do not sum device times."""
    from tools.bottlenecks import profile as P
    T = P.TRACE_SPEC.T
    if trace['flags'] & 1 or trace.get('trace_stats', {}).get('dropped_event_count', 0):
        raise SpecError('execution trace is truncated or dropped events')
    executions = {}
    for e in trace['events']:
        if e['kind'] == T.K_EXECUTION_BEGIN:
            phase = T.execution_phase(e)
            if phase in ('generation_target', 'generation_draft'):
                executions[e['execution_id']] = {'phase': phase, 'tokens': e['args'][0], 'begin': e['timestamp_ns'], 'end': None}
        elif e['kind'] == T.K_EXECUTION_END and e['execution_id'] in executions:
            executions[e['execution_id']]['end'] = e['timestamp_ns']
    by_width, by_prompt = {}, {}
    for window in windows:
        width, prompt = window['width'], str(window['prompt'])
        lo, hi = window['start_ns'], window['end_ns']
        # A partial accept on a recurrent model can replay narrower target batches after checkpoint restoration.
        # Bracket the full verify passes; their interval also owns those replay calls.
        target = sorted((e for e in executions.values() if e['phase'] == 'generation_target' and e['tokens'] == width
                         and lo <= e['begin'] <= hi), key=lambda e: e['begin'])
        pairs = list(zip(target, target[1:]))[settle:]
        samples = []
        for a, b in pairs:
            if a['end'] is None or b['end'] is None:
                raise SpecError('incomplete execution in a measured request')
            draft = [e for e in executions.values() if e['phase'] == 'generation_draft' and a['begin'] <= e['begin'] < b['begin']]
            if any(e['end'] is None for e in draft):
                raise SpecError('incomplete draft execution')
            # These are host spans, including draft ingest. Keep them diagnostic; the complete wall interval owns T(w).
            draft_ms = P.TRACE_SPEC.union_ns([(e['begin'], min(e['end'], b['begin'])) for e in draft]) / 1e6
            samples.append({'cycle_ms': (b['begin'] - a['begin']) / 1e6, 'draft_ms': draft_ms,
                            'verify_ms': (a['end'] - a['begin']) / 1e6})
        if len(samples) < minimum:
            raise SpecError(f'width {width}, prompt {prompt}: only {len(samples)} settled cycles, need {minimum}')
        key = str(width)
        by_width.setdefault(key, []).extend(samples)
        by_prompt.setdefault(prompt, {}).setdefault(key, []).extend(samples)

    def summarize(groups):
        return {w: {**{k: statistics.median(s[k] for s in samples) for k in ('cycle_ms', 'draft_ms', 'verify_ms')},
                    'cycles': len(samples), 'cycle_ms_samples': [s['cycle_ms'] for s in samples]} for w, samples in groups.items()}
    return {'schema': 'delphi.auto-tune/pass-cost/1', 'widths': summarize(by_width),
            'by_prompt': {p: summarize(groups) for p, groups in by_prompt.items()}, 'settle_cycles': settle}


def replay(requests, policy, widths):
    """Empirical pass replay. Clipping changes future pass boundaries, so this is a prediction, not a token replay."""
    tokens, elapsed, accepted_total, passes = 0, 0.0, 0, 0
    depths = {}
    for request in requests:
        bucket = Bucket(policy)
        for row in request['passes']:
            if policy.n_max > row['n']:
                raise SpecError('policy exceeds the uncensored capture depth')
            depth = bucket.depth if policy.adaptive == 'bucket' else policy.n_max
            n = depth
            for i, p in enumerate(row['p'][:depth]):
                if p < policy.p_min:
                    n = i
                    break
            # The low-confidence step ran before its token was cut. n_min can then discard the whole draft.
            attempted = min(depth, n + 1) if n < depth else depth
            verified = n if n >= policy.n_min else 0
            accepted = min(row['acc'], verified)
            try:
                verify_cost, draft_cost = widths[str(verified + 1)], widths[str(attempted + 1)]
                cost = verify_cost['cycle_ms'] - verify_cost['draft_ms'] + draft_cost['draft_ms']
            except KeyError as error:
                raise SpecError(f'missing measured cost for width {error.args[0]}') from error
            if not math.isfinite(cost) or cost <= 0:
                raise SpecError('non-positive replay pass cost')
            tokens += accepted + 1
            accepted_total += accepted
            elapsed += cost
            passes += 1
            depths[str(depth)] = depths.get(str(depth), 0) + 1
            # The server does not feed a discarded/empty draft to this implementation's controller.
            if verified:
                bucket.update(accepted)
    if not passes:
        raise SpecError('acceptance corpus is empty')
    return {'policy': asdict(policy), 'decode_t_s': 1000 * tokens / elapsed, 'output_tokens': tokens, 'wall_ms': elapsed,
            'accepted_per_pass': accepted_total / passes, 'passes': passes, 'depths': depths, 'estimate': True}


def policies(command, env, cfg, cap, info=None):
    """Use the request registry domains, clipped to the output capacity of the launch."""
    info = info or {}
    mode = info.get('adaptive', 'bucket' if '--spec-draft-adaptive' in command else 'off')
    def current(suffix, default):
        return K.option_value(command, '--spec-draft-' + suffix) or env.get('LLAMA_ARG_SPEC_DRAFT_' + suffix.upper().replace('-', '_'), str(default))
    domains = {
        '--spec-draft-n-max': [current('n-max', min(info.get('n_max', cap), cap))],
        '--spec-draft-n-min': [current('n-min', 0)],
        '--spec-draft-p-min': [str(info.get('p_min', current('p-min', 0)))],
        '--spec-draft-adaptive': [mode],
        '--spec-draft-n-min-adaptive': [current('n-min-adaptive', (info.get('impls') or [{}])[0].get('n_min_adaptive') or 3)],
    }
    # Expose effective env/default values to the registry. Enable the floor dependency for candidate bucket policies.
    active = tuple(command)
    for flag, values in domains.items():
        if flag != '--spec-draft-adaptive':
            active = K.set_option(active, flag, values[0])
    ctx = K.Context((*active, '--spec-draft-adaptive'), env, 1, 1)
    selected = {k.name: values for k, values in K.select(ctx, **cfg['knobs']) if k.reload == 'request'}
    if not selected:
        return []
    flag = '--spec-draft-adaptive'
    if flag in selected:
        values = cfg['knobs']['levels'].get(flag) or K.BY_NAME[flag].level_list(ctx)
        # The synthetic dependency above is not an observed current bucket policy.
        selected[flag] = list(dict.fromkeys([str(v) for v in values] + [mode]))
    domains.update(selected)
    result = set()
    for n, minimum, prob, adaptive, floor in itertools.product(*domains.values()):
        if adaptive in ('cost', 'conf'):
            continue
        try:
            adaptive = 'bucket' if adaptive in (K.ON, 'bucket') else 'off'
            policy = Policy(int(n), int(minimum), float(prob), adaptive,
                            int(floor) if adaptive == 'bucket' else 1, int(current('n-start', 0)))
        except SpecError:
            continue
        if policy.n_max <= cap:
            result.add(policy)
    return sorted(result, key=lambda p: (p.n_max, p.n_min, p.p_min, p.adaptive, p.n_min_adaptive))


def rank(requests, candidates, table):
    results, skipped = [], []
    for policy in candidates:
        try:
            result = replay(requests, policy, table['widths'])
            result['repeat_decode_t_s'] = [replay(requests, policy, widths)['decode_t_s'] for widths in table['by_prompt'].values()]
            results.append(result)
        except SpecError as error:
            skipped.append({'policy': asdict(policy), 'reason': str(error)})
    return {'candidates': sorted(results, key=lambda r: r['decode_t_s'], reverse=True), 'skipped': skipped,
            'limitations': ['Empirical pass replay changes pass boundaries; fresh real-acceptance confirmation is required.',
                            'p_min uses the MTP log rounded to four decimals; cut-step draft cost uses host-span estimates.',
                            'Cost/conf controllers require live timing, shadow tokens or calibration and are not replayed.',
                            'Settled width medians omit graph rebuild and depth-switch costs.']}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path, help='loop-3 capture.json with request boundaries')
    parser.add_argument('costs', type=Path, help='pass-cost.json from the same context point')
    parser.add_argument('--policy', action='append', required=True, help='JSON Policy fields; repeat to compare')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        capture = json.loads(args.capture.read_text())
        table = json.loads(args.costs.read_text())
        if capture['point'] != table['point']:
            raise SpecError('capture and costs belong to different context points')
        result = rank(capture['requests'], [Policy(**json.loads(p)) for p in args.policy], table)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'speculation: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
