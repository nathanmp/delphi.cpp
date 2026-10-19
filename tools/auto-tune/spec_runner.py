"""Collect loop-3 inputs through /speculative on one owned server process."""
from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import knobs as K
import measure as M
from config import digest
from prefill_client import post
from speculation import Policy, SpecError, Unavailable, cost_table, policies, rank, read_plog, replay
import observations as O


def unavailable(command, env=None):
    types = set((K.option_value(command, '--spec-type') or '').split(',')) - {'none'}
    if types != {'draft-mtp'}:
        return 'Replay costs currently support MTP only; DSpark and other block drafters need physical-block cost calibration.'
    env = env or {}
    if (K.option_value(command, '--spec-synth-rates') or K.option_value(command, '--spec-synth-len')
            or env.get('LLAMA_ARG_SPEC_SYNTH_RATES') or env.get('LLAMA_ARG_SPEC_SYNTH_LEN')):
        return 'The reference must use real acceptance.'
    return None


class Collector:
    def __init__(self, runner, command, env, directory, continue_measure):
        self.runner, self.command, self.env, self.directory = runner, command, env, directory
        self.cfg = runner.cfg['speculation']
        self.continue_measure = continue_measure
        self.windows, self.requests, self.operations = [], [], []
        launch = directory / 'launch.json'
        self.launch = json.loads(launch.read_text()) if launch.exists() else {}
        self.point = digest([command, env, runner.cfg['speculation'], self.launch, time.time_ns()])
        self.started = time.monotonic()
        self.request_count = 0

    def save(self, name, value):
        (self.directory / name).write_text(json.dumps(value, indent=2) + '\n')

    def check_budget(self):
        if not self.continue_measure():
            raise SpecError('budget ended before loop-3 measurements completed')

    def configure(self, args):
        self.check_budget()
        result = self.runner.request({'args': args, 'calibrate': False}, path='/speculative')
        self.operations.append({'args': args, 'response': result})
        self.save('request-settings.json', self.operations)
        return result

    def completion(self, prompt, *, window=None):
        self.check_budget()
        lo = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        result = post(self.runner.url, '/completion', {'prompt': prompt, 'n_predict': self.cfg['osl'], 'temperature': 0,
                     'seed': 0, 'ignore_eos': True, 'cache_prompt': False, 'stream': False, 'return_tokens': True}, self.runner.arm.request_timeout_s)
        self.request_count += 1
        hi = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        if result.get('tokens_predicted') != self.cfg['osl'] or result.get('truncated'):
            raise SpecError('loop-3 completion was short or context-truncated')
        if window is not None:
            self.windows.append({**window, 'start_ns': lo, 'end_ns': hi, 'request_id': result.get('speculative_request_id'),
                                 'timings': result.get('timings', {})})
            self.save('windows.json', self.windows)
        return result

    def collect(self):
        if getattr(self.runner, 'snapshot_supported', False) is True:
            return self.collect_observed()
        return self.collect_legacy()

    def collect_legacy(self):
        info = self.runner.request(path='/speculative')
        self.save('speculative-before.json', info)
        types = set(info['types'].split(',')) - {'none'}
        if types != {'draft-mtp'} or info['draft_sampling'] != 'greedy' or not info['draft_context']:
            raise Unavailable('replay needs one greedy MTP implementation and its draft context')
        props = self.runner.request(path='/props')
        if props.get('total_slots') != 1:
            raise Unavailable('the untagged MTP log needs exactly one server slot')
        self.cap = min(info['n_max_launch'], self.cfg['max_draft'])
        self.candidates = policies(self.command, self.env, self.runner.cfg, self.cap, info)
        if not self.candidates:
            raise Unavailable('no supported request policies fit the launch capacity and registry filter')
        ids = post(self.runner.url, '/tokenize', {'content': Path(self.cfg['text']).read_text(), 'add_special': False},
                   self.runner.arm.request_timeout_s)['tokens']
        count = self.cfg['prompt_tokens'] * self.cfg['prompts']
        if len(ids) < count:
            raise SpecError(f'loop-3 text has {len(ids)} tokens, need {count}')
        if self.cfg['prompt_tokens'] + self.cfg['osl'] + self.cap + 1 > self.runner.cfg['context']:
            raise SpecError('loop-3 workload exceeds the verified context floor')
        prompts = [ids[i:i + self.cfg['prompt_tokens']] for i in range(0, count, self.cfg['prompt_tokens'])]
        self.save('source.json', {'point': self.point, 'launch': self.launch, 'speculative': info, 'capture_policy': asdict(Policy(self.cap)),
                                 'command': list(self.command), 'env': self.env, 'prompt_tokens': prompts})
        try:
            # Each update starts from the real-acceptance launch defaults. The CLI rejects an explicit synth-len=-1.
            clean = ['--spec-draft-sampling', 'greedy']
            captured = self.configure([*Policy(self.cap).args(), *clean])
            if captured['adaptive'] != 'off' or captured['p_min'] != 0 or captured['n_max'] != self.cap:
                raise SpecError('server did not apply the uncensored capture policy')
            for i, prompt in enumerate(prompts):
                before = self.runner.plog_path.stat().st_size if self.runner.plog_path.exists() else 0
                response = self.completion(prompt)
                if not self.runner.plog_path.exists():
                    raise Unavailable('server did not write DELPHI_SPEC_PLOG; request the per-pass server log')
                with self.runner.plog_path.open('rb') as handle:
                    handle.seek(before)
                    text = handle.read().decode()
                (self.directory / f'acceptance-{i}.jsonl').write_text(text)
                request = {**read_plog(text, self.cap), 'prompt': i, 'timings': response.get('timings', {})}
                self.requests.append(request)
            self.save('capture.json', {'point': self.point, 'cap': self.cap, 'requests': self.requests})
            # Width 1 prices a discarded draft. Fixed widths use deterministic synthetic acceptance; output is invalid.
            widths = list(range(1, self.cap + 2))
            random.Random(0).shuffle(widths)
            for width in widths:
                n = width - 1
                args = ([*Policy(n).args(), '--spec-synth-rates', ','.join(['1'] * (n - 1) + ['0'])]
                        if n else ['--spec-type', 'none'])
                state = self.configure(args)
                if n and (state['n_max'] != n or state['adaptive'] != 'off'):
                    raise SpecError('server did not apply the fixed synthetic width')
                for i, prompt in enumerate(prompts):
                    self.completion(prompt, window={'width': width, 'prompt': i})
        finally:
            # /speculative applies args over the launch state. An empty list restores that complete state.
            restored = self.runner.request({'args': [], 'calibrate': False}, path='/speculative')
            self.save('restored.json', restored)
            for key in ('types', 'n_max', 'adaptive', 'p_min', 'draft_sampling'):
                if restored.get(key) != info.get(key):
                    raise SpecError(f'failed to restore speculative {key}')
        self.collection_wall_s = time.monotonic() - self.started

    def live_inputs(self):
        shot = O.snapshot_ok(self.runner.request({}, path='/tune/snapshot'))
        self.runner.snapshot_state = shot
        self.save('snapshot.json', shot)
        if Path(shot['trace_path']) != self.runner.trace_path or Path(shot['speculative']['path']) != self.runner.observation_path:
            raise SpecError('snapshot paths differ from this owned process')
        requests = O.read(self.runner.observation_path.read_text())
        trace = M.P.TRACE_SPEC.T.read_trace(self.runner.trace_path,
                kinds={M.P.TRACE.K_EXECUTION_BEGIN, M.P.TRACE.K_EXECUTION_END, O.K_SPECULATIVE})
        return requests, trace

    def collect_observed(self):
        info = self.runner.request(path='/speculative')
        self.save('speculative-before.json', info)
        if set(info['types'].split(',')) - {'none'} != {'draft-mtp'} or info['draft_sampling'] != 'greedy' or not info['draft_context']:
            raise Unavailable('counterfactual costs currently require greedy MTP; block drafters need physical-block cost calibration')
        if self.runner.request(path='/props').get('total_slots') != 1:
            raise Unavailable('pass wall costs currently require exactly one server slot')
        self.cap = min(info['n_max_launch'], self.cfg['max_draft'])
        self.candidates = policies(self.command, self.env, self.runner.cfg, self.cap, info)
        if not self.candidates:
            raise Unavailable('no supported request policies fit the launch capacity and registry filter')
        ids = post(self.runner.url, '/tokenize', {'content': Path(self.cfg['text']).read_text(), 'add_special': False},
                   self.runner.arm.request_timeout_s)['tokens']
        count = self.cfg['prompt_tokens'] * self.cfg['prompts']
        if len(ids) < count or self.cfg['prompt_tokens'] + self.cfg['osl'] + self.cap + 1 > self.runner.cfg['context']:
            raise SpecError('loop-3 text or context is too short for the requested workload')
        prompts = [ids[i:i + self.cfg['prompt_tokens']] for i in range(0, count, self.cfg['prompt_tokens'])]
        self.save('source.json', {'point': self.point, 'launch': self.launch, 'speculative': info,
                                 'capture_policy': asdict(Policy(self.cap)), 'command': list(self.command),
                                 'env': self.env, 'prompt_tokens': prompts, 'observations': O.SCHEMA})
        try:
            widths = list(range(1, self.cap + 2))
            random.Random(0).shuffle(widths)
            for width in widths:
                n = width - 1
                args = ([*Policy(n).args(), '--spec-synth-rates', ','.join(['1'] * (n - 1) + ['0'])]
                        if n else ['--spec-type', 'none'])
                self.configure(args)
                self.completion(prompts[0])  # Warm the new graph width; never use synthetic output as quality evidence.
                for i, prompt in enumerate(prompts):
                    self.completion(prompt, window={'width': width, 'prompt': i})
            captured = self.configure([*Policy(self.cap).args(), '--spec-draft-sampling', 'greedy'])
            if captured['adaptive'] != 'off' or captured['p_min'] != 0 or captured['n_max'] != self.cap:
                raise SpecError('server did not apply the uncensored capture policy')
            for prompt in prompts:
                self.completion(prompt)
            source = [self.completion(prompt) for prompt in prompts]
            inputs, trace = self.live_inputs()
            for i, response in enumerate(source):
                request_id = response.get('speculative_request_id')
                rows = inputs.get(request_id, [])
                O.correlate(trace, rows)
                self.requests.append({**O.capture(rows, self.cap, self.cfg['settle_cycles']), 'prompt': i,
                                      'request_id': request_id, 'timings': response.get('timings', {})})
            self.save('capture.json', {'point': self.point, 'cap': self.cap, 'requests': self.requests, 'observations': O.SCHEMA})
            table = O.costs(trace, inputs, self.windows, self.cfg['settle_cycles'], self.cfg['min_cycles'])
            table['point'] = self.point
            self.save('pass-cost.json', table)
            self.result = {**rank(self.requests, self.candidates, table), 'status': 'ok', 'point': self.point,
                           'cap': self.cap, 'path': str(self.directory), 'observations': O.SCHEMA,
                           'capture_prediction': replay(self.requests, Policy(self.cap), table['widths']),
                           'capture_observed_t_s': [r['timings'].get('predicted_per_second') for r in self.requests],
                           'capture_aligned_t_s': [r['aligned_decode_t_s'] for r in self.requests],
                           'trace_fallback': 'Cumulative live snapshots; model weights remain resident between context points.'}
            self.validate(prompts, source, table)
        finally:
            restored = self.runner.request({'args': [], 'calibrate': False}, path='/speculative')
            self.save('restored.json', restored)
            for key in ('types', 'n_max', 'adaptive', 'p_min', 'draft_sampling'):
                if restored.get(key) != info.get(key):
                    raise SpecError(f'failed to restore speculative {key}')
        self.collection_wall_s = time.monotonic() - self.started

    def validate(self, prompts, source, table):
        """Independent normal-acceptance requests on the same workload; report both pass and full decode rates."""
        selected = [Policy(self.cap)]
        for candidate in self.result['candidates']:
            policy = Policy(**candidate['policy'])
            if policy not in selected:
                selected.append(policy)
            if len(selected) >= 1 + self.cfg['validation_candidates']:
                break
        pending = []
        for rep in range(self.cfg['validation_reps']):
            order = list(selected)
            if rep % 2:
                order.reverse()
            for policy in order:
                self.configure([*policy.args(), '--spec-draft-sampling', 'greedy'])
                self.completion(prompts[0])
                for i, prompt in enumerate(prompts):
                    response = self.completion(prompt)
                    if not response.get('tokens') or response['tokens'] != source[i].get('tokens'):
                        raise SpecError('normal-acceptance validation changed the greedy token sequence')
                    pending.append({'policy': asdict(policy), 'repeat': rep, 'prompt': i,
                                    'request_id': response.get('speculative_request_id'), 'timings': response['timings']})
        inputs, trace = self.live_inputs()
        results = []
        for policy in selected:
            samples = []
            for item in pending:
                if item['policy'] != asdict(policy):
                    continue
                rows = inputs.get(item['request_id'], [])
                O.correlate(trace, rows)
                rows = [r for r in rows[self.cfg['settle_cycles']:] if 'cycle_ms' in r]
                if len(rows) < self.cfg['min_cycles']:
                    raise SpecError('too few settled normal-acceptance validation cycles')
                prediction = replay([self.requests[item['prompt']]], policy, table['by_prompt'][str(item['prompt'])])
                actual = O.rate(rows)
                samples.append({**item, 'aligned_decode_t_s': actual, 'prediction_t_s': prediction['decode_t_s'],
                                'error_fraction': prediction['decode_t_s'] / actual - 1,
                                'accepted_per_pass': statistics.mean(r['acc'] for r in rows),
                                'passes': len(rows), 'tokens_identical': True})
            predicted = statistics.median(s['prediction_t_s'] for s in samples)
            actual = statistics.median(s['aligned_decode_t_s'] for s in samples)
            results.append({'policy': asdict(policy), 'prediction_t_s': predicted, 'aligned_decode_t_s': actual,
                            'decode_t_s': statistics.median(s['timings']['predicted_per_second'] for s in samples),
                            'error_fraction': predicted / actual - 1, 'samples': samples})
        self.result['validation'] = {'schema': 'delphi.auto-tune/replay-validation/1', 'normal_acceptance': True,
                                     'same_workload': True, 'fresh_requests': True, 'fresh_process': False,
                                     'policies': results, 'repetitions': self.cfg['validation_reps'],
                                     'note': 'Matched-workload diagnostics; deployment recommendations still require fresh-process confirmation.'}
        self.save('validation.json', self.result['validation'])

    def finish(self):
        if hasattr(self, 'result'):
            self.result.update(wall_s=self.collection_wall_s, collection_wall_s=self.collection_wall_s,
                               requests=self.request_count, policy_updates=len(self.operations) + 1)
            self.save('replay.json', self.result)
            return self.result
        # Older servers have no live flush endpoint. The caller stops that process before reading the trace.
        started = time.monotonic()
        trace = M.P.TRACE_SPEC.T.read_trace(self.runner.trace_path, kinds={M.P.TRACE.K_EXECUTION_BEGIN, M.P.TRACE.K_EXECUTION_END})
        table = cost_table(trace, self.windows, self.cfg['settle_cycles'], self.cfg['min_cycles'])
        table['point'] = self.point
        self.save('pass-cost.json', table)
        # Also retain the existing diagnostic summary; T(w) is based on full cycle wall intervals above.
        self.save('exec-trace-spec.json', M.P.TRACE_SPEC.summarize(trace, 0, None,
                  [(w['start_ns'], w['end_ns']) for w in self.windows]))
        source_prediction = replay(self.requests, Policy(self.cap), table['widths'])
        observed = [r['timings'].get('predicted_per_second') for r in self.requests]
        result = {**rank(self.requests, self.candidates, table), 'status': 'ok', 'point': self.point,
                  'cap': self.cap, 'path': str(self.directory), 'wall_s': self.collection_wall_s + time.monotonic() - started,
                  'collection_wall_s': self.collection_wall_s,
                  'capture_prediction': source_prediction, 'capture_observed_t_s': observed,
                  'requests': len(self.windows) + len(self.requests), 'policy_updates': len(self.operations) + 1,
                  'trace_fallback': 'Trace flush requires process exit; loop-2 points use fresh launches for this opt-in mode.'}
        self.save('replay.json', result)
        return result
