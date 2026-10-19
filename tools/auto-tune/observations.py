"""Read request-tagged observations and correlate them with cumulative trace snapshots."""
from __future__ import annotations

import json
import math
import statistics

from speculation import SpecError, cost_table

SCHEMA = 'delphi-spec-pass-v1'
K_SPECULATIVE = 33


def snapshot_ok(value):
    spec = value.get('speculative', {})
    if (not value.get('ok') or not value.get('cumulative') or value.get('dropped_event_count') != 0
            or not spec.get('enabled') or not spec.get('ok') or spec.get('failed') or spec.get('dropped') != 0):
        raise SpecError('incomplete speculative observation or execution trace snapshot')
    if not 0 <= value['event_count'] <= value['event_capacity']:
        raise SpecError('invalid snapshot event counts')
    return value


def read(text):
    """Keep physical verifications together; a replay never supplies a second logical acceptance."""
    groups, owners = {}, {}
    try:
        for line in text.splitlines():
            r = json.loads(line)
            if r['schema'] != SCHEMA or r['kind'] == 'incomplete':
                raise ValueError('unknown schema or incomplete observation')
            for key in ('request_id', 'sequence_id', 'pass_id', 'verification', 'time_ns'):
                if type(r[key]) is not int or r[key] < 0:
                    raise ValueError('invalid observation identity or timestamp')
            owner = (r['request_id'], r['sequence_id'])
            if owners.setdefault(r['pass_id'], owner) != owner:
                raise ValueError('pass ID belongs to multiple requests/sequences')
            groups.setdefault((*owner, r['pass_id']), []).append(r)
    except (ValueError, KeyError, TypeError) as error:
        raise SpecError(f'invalid speculative observations: {error}') from error
    requests = {}
    for (request, sequence, pass_id), rows in groups.items():
        if any(r['kind'] == 'abandoned' for r in rows):
            continue
        try:
            drafts = [r for r in rows if r['kind'] == 'draft']
            targets = [r for r in rows if r['kind'] == 'target']
            verifies = [r for r in rows if r['kind'] == 'verify']
            if any(r['kind'] not in ('draft', 'target', 'verify') for r in rows):
                raise ValueError('unknown observation kind')
            if not drafts or not targets or not verifies or verifies[-1]['checkpoint_replay_required']:
                raise ValueError('unfinished pass')
            draft, verify = drafts[-1], verifies[0]
            if (len(drafts) != 1 or draft['probabilistic'] or draft['verification'] != 0
                    or draft['selected'] != bool(draft['draft_tokens'])):
                raise ValueError('replay needs a single greedy draft implementation')
            tokens, probs, attempts = draft['draft_tokens'], draft['draft_probabilities'], draft['attempts']
            if (len(tokens) != len(probs) or len(attempts) != draft['attempted_steps']
                    or any(type(t) is not int for t in tokens) or draft['time_ns'] < draft['draft_start_ns']):
                raise ValueError('invalid draft tokens, steps or timing')
            if any(p is not None and (type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1) for p in probs):
                raise ValueError('invalid draft probability')
            for i, v in enumerate(verifies):
                n = v['accepted_length']
                if (v['verification'] != i or v['replay'] != (i > 0)
                        or (i < len(verifies) - 1 and not v['checkpoint_replay_required'])
                        or type(n) is not int or not 0 <= n <= len(v['draft_tokens'])
                        or len(v['accepted_tokens']) != n + 1
                        or v['accepted_tokens'][:n] != v['draft_tokens'][:n]):
                    raise ValueError('invalid physical verification chain')
                if not any(t['verification'] == i and t['status'] == 0 for t in targets):
                    raise ValueError('missing successful target execution')
            if verify['draft_tokens'] != tokens:
                raise ValueError('draft/verification token mismatch')
            if any(t['target_start_ns'] > t['time_ns'] for t in targets):
                raise ValueError('invalid target timing')
            row = {'request_id': request, 'sequence_id': sequence, 'pass_id': pass_id,
                   'n': len(tokens), 'acc': verify['accepted_length'], 'p': probs,
                   'draft': draft, 'targets': targets, 'verifies': verifies}
            requests.setdefault(request, []).append(row)
        except (ValueError, TypeError, KeyError) as error:
            raise SpecError(f'pass {pass_id}: {error}') from error
    for rows in requests.values():
        rows.sort(key=lambda r: r['draft']['draft_start_ns'])
        for a, b in zip(rows, rows[1:]):
            a['cycle_ms'] = (b['draft']['draft_start_ns'] - a['draft']['draft_start_ns']) / 1e6
            if a['cycle_ms'] <= 0 or a['verifies'][-1]['time_ns'] > b['draft']['draft_start_ns']:
                raise SpecError('overlapping logical passes in a request')
    return requests


def capture(rows, cap, settle=0):
    full, omitted = [], 0
    for row in rows[settle:]:
        d = row['draft']
        if row['n'] != cap or d['effective_n_max'] != cap or 'cycle_ms' not in row:
            omitted += 1
            continue
        if (d['implementation'] != 'draft-mtp' or d['physical_draft_rows'] != 0
                or d['attempted_steps'] != cap or len(row['p']) != cap
                or any(p is None for p in row['p'])
                or any(s['confidence_kind'] != 'vocab_top1' or s['token'] != token
                       or s['confidence'] != p for s, token, p in zip(d['attempts'], d['draft_tokens'], row['p']))):
            raise SpecError('capture is censored or lacks greedy MTP confidence probabilities')
        full.append(row)
    if not full:
        raise SpecError('no settled uncensored passes in the observation request')
    return {'passes': full, 'boundary_passes_omitted': omitted, 'settle_cycles': settle,
            'aligned_decode_t_s': rate(full)}


def rate(rows):
    return 1000 * sum(r['acc'] + 1 for r in rows) / sum(r['cycle_ms'] for r in rows)


def correlate(trace, rows):
    if trace['flags'] & 1 or trace.get('trace_stats', {}).get('dropped_event_count', 0):
        raise SpecError('execution trace is truncated or dropped events')
    markers = {(tuple(e['args'][:4]), e['flags'], e['timestamp_ns']) for e in trace['events'] if e['kind'] == K_SPECULATIVE}
    for row in rows:
        identity = (row['request_id'], row['sequence_id'], row['pass_id'])
        d = row['draft']
        expected = [(0, 1, d['draft_start_ns']), (0, 2, d['time_ns'])]
        for t in row['targets']:
            expected.extend([(t['verification'], 3, t['target_start_ns']), (t['verification'], 4, t['time_ns'])])
        for v in row['verifies']:
            expected.append((v['verification'], 5 if v['checkpoint_replay_required'] else 6, v['time_ns']))
        if any(((*identity, verification), flag, ns) not in markers for verification, flag, ns in expected):
            raise SpecError(f'pass {row["pass_id"]}: missing matching binary trace markers')


def costs(trace, requests, windows, settle=3, minimum=3):
    """Full logical cycles include checkpoint replay and server overhead; draft spans include ingest."""
    plain = [w for w in windows if w['width'] == 1]
    table = cost_table(trace, plain, settle, minimum) if plain else {'widths': {}, 'by_prompt': {}}
    by_width, by_prompt = {}, {}
    for w in windows:
        if w['width'] == 1:
            continue
        rows = requests.get(w['request_id'], [])
        correlate(trace, rows)
        samples = []
        for r in rows[settle:]:
            d = r['draft']
            if r['n'] + 1 != w['width'] or 'cycle_ms' not in r:
                continue
            if not w['start_ns'] <= d['draft_start_ns'] <= r['verifies'][-1]['time_ns'] <= w['end_ns']:
                raise SpecError('observation escaped its request window')
            samples.append({'cycle_ms': r['cycle_ms'], 'draft_ms': (d['time_ns'] - d['draft_start_ns']) / 1e6,
                            'verify_ms': sum(t['time_ns'] - t['target_start_ns'] for t in r['targets']) / 1e6})
        if len(samples) < minimum:
            raise SpecError(f'width {w["width"]}, prompt {w["prompt"]}: insufficient settled observation cycles')
        key, prompt = str(w['width']), str(w['prompt'])
        by_width.setdefault(key, []).extend(samples)
        by_prompt.setdefault(prompt, {}).setdefault(key, []).extend(samples)
    def summary(samples):
        return {**{k: statistics.median(s[k] for s in samples) for k in ('cycle_ms', 'draft_ms', 'verify_ms')},
                'cycles': len(samples), 'cycle_ms_samples': [s['cycle_ms'] for s in samples]}
    table['widths'].update({k: summary(v) for k, v in by_width.items()})
    for prompt, groups in by_prompt.items():
        table['by_prompt'].setdefault(prompt, {}).update({k: summary(v) for k, v in groups.items()})
    table.update(schema='delphi.auto-tune/pass-cost/2', settle_cycles=settle, observations=SCHEMA)
    return table
