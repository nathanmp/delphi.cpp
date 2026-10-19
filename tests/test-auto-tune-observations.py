#!/usr/bin/env python3
"""Observation integrity and retained-process fallbacks, without a model or GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/auto-tune'))
import observations as O
from speculation import SpecError, Policy, replay, validated_estimates
from context_runner import ContextRunner, process_key
from config import DEFAULTS, _merge


def records(request=10, count=5, checkpoint=False, confidence_kind='vocab_top1'):
    rows = []
    for i in range(count):
        base = {'schema': O.SCHEMA, 'request_id': request, 'sequence_id': 0,
                'pass_id': request * 100 + i, 'verification': 0}
        start = i * 20_000_000 + request * 1_000_000_000
        draft = {**base, 'kind': 'draft', 'time_ns': start + 4_000_000, 'draft_start_ns': start,
                 'implementation': 'draft-mtp', 'selected': True, 'probabilistic': False,
                 'draft_tokens': [42, 43], 'draft_probabilities': [.9876543283462524, .7],
                 'attempted_steps': 2, 'effective_n_max': 2, 'physical_draft_rows': 0,
                 'attempts': [{'token': t, 'confidence_kind': confidence_kind, 'confidence': p}
                              for t, p in zip([42, 43], [.9876543283462524, .7])]}
        target = {**base, 'kind': 'target', 'time_ns': start + 10_000_000,
                  'target_start_ns': start + 5_000_000, 'batch_tokens': 3, 'status': 0}
        verify = {**base, 'kind': 'verify', 'time_ns': start + 11_000_000, 'draft_tokens': [42, 43],
                  'accepted_tokens': [42, 99], 'accepted_length': 1, 'replay': False,
                  'checkpoint_replay_required': checkpoint}
        rows += [draft, target, verify]
        if checkpoint:
            rows += [{**target, 'verification': 1, 'target_start_ns': start + 12_000_000, 'time_ns': start + 14_000_000},
                     {**verify, 'verification': 1, 'time_ns': start + 15_000_000, 'replay': True,
                      'checkpoint_replay_required': False}]
    return rows


def parse(rows):
    return O.read('\n'.join(json.dumps(r) for r in rows))


def trace(rows):
    events = []
    for r in rows:
        identity = [r['request_id'], r['sequence_id'], r['pass_id'], r['verification']]
        def marker(flag, time):
            events.append({'kind': 33, 'args': identity, 'flags': flag, 'timestamp_ns': time})
        if r['kind'] == 'draft':
            marker(1, r['draft_start_ns'])
            marker(2, r['time_ns'])
        elif r['kind'] == 'target':
            marker(3, r['target_start_ns'])
            marker(4, r['time_ns'])
        else:
            marker(5 if r['checkpoint_replay_required'] else 6, r['time_ns'])
    return {'flags': 0, 'trace_stats': {}, 'events': events}


class Observations(unittest.TestCase):
    def test_request_identity_and_float_precision(self):
        requests = parse(records(10) + records(20))
        capture = O.capture(requests[10], 2, settle=1)
        self.assertEqual(len(capture['passes']), 3)
        self.assertEqual(capture['passes'][0]['p'][0], .9876543283462524)
        self.assertEqual(capture['aligned_decode_t_s'], 100)
        self.assertTrue(all(r['request_id'] == 10 for r in capture['passes']))

    def test_checkpoint_keeps_original_acceptance_and_complete_wall_cost(self):
        raw = records(checkpoint=True)
        requests = parse(raw)
        rows = requests[10]
        self.assertEqual(rows[0]['acc'], 1)
        self.assertEqual(O.rate(rows[:-1]), 100)
        window = {'width': 3, 'prompt': 0, 'request_id': 10, 'start_ns': 0, 'end_ns': 100_000_000_000}
        table = O.costs(trace(raw), requests, [window], settle=1, minimum=3)
        self.assertEqual(table['widths']['3']['cycle_ms'], 20)
        self.assertEqual(table['widths']['3']['verify_ms'], 7)
        self.assertEqual(table['widths']['3']['draft_ms'], 4)

    def test_corrupted_or_partial_capture_is_rejected(self):
        cases = []
        rows = records()
        cases.append(rows[:-1])
        cases.append(rows + [{'schema': O.SCHEMA, 'kind': 'incomplete', 'dropped': 1, 'failed': False}])
        for field, value in [('accepted_length', 3), ('accepted_tokens', [99, 99]), ('verification', 2)]:
            changed = copy.deepcopy(rows)
            changed[2][field] = value
            cases.append(changed)
        changed = copy.deepcopy(rows)
        changed[0]['draft_probabilities'] = [float('nan'), .7]
        cases.append(changed)
        changed = copy.deepcopy(rows)
        changed[1]['request_id'] = 90
        cases.append(changed)
        for rows in cases:
            with self.subTest(rows=rows[-1]), self.assertRaises(SpecError):
                parse(rows)

    def test_abandoned_pass_does_not_supply_acceptance(self):
        rows = records()
        rows[2] = {**rows[2], 'kind': 'abandoned'}
        self.assertEqual(len(parse(rows)[10]), 4)

    def test_block_head_confidence_cannot_be_used_as_mtp_probability(self):
        requests = parse(records(confidence_kind='dspark_head'))
        with self.assertRaisesRegex(SpecError, 'confidence'):
            O.capture(requests[10], 2)
        raw = records()
        raw[0]['draft_probabilities'][0] = None
        with self.assertRaises(SpecError):
            O.capture(parse(raw)[10], 2)

    def test_missing_or_mismatched_markers_are_rejected(self):
        raw = records()
        rows = parse(raw)[10]
        t = trace(raw)
        O.correlate(t, rows)
        t['events'][0]['args'] = [99, 0, 0, 0]
        with self.assertRaises(SpecError):
            O.correlate(t, rows)

    def test_request_windows_do_not_accept_previous_context(self):
        raw = records(10) + records(20)
        requests = parse(raw)
        w = {'width': 3, 'prompt': 0, 'request_id': 20, 'start_ns': 19_000_000_000, 'end_ns': 21_000_000_000}
        result = O.costs(trace(raw), requests, [w], settle=1, minimum=3)
        self.assertEqual(result['widths']['3']['cycles'], 3)
        w['request_id'] = 10
        with self.assertRaises(SpecError):
            O.costs(trace(raw), requests, [w], settle=1, minimum=3)

    def test_empty_draft_cost_keeps_the_confidence_cut_step(self):
        raw = records()
        for r in raw:
            if r['kind'] == 'draft':
                r.update(draft_tokens=[], draft_probabilities=[], selected=False)
            elif r['kind'] == 'target':
                r['batch_tokens'] = 1
            else:
                r.update(draft_tokens=[], accepted_tokens=[99], accepted_length=0)
        requests = parse(raw)
        window = {'width': 1, 'prompt': 0, 'request_id': 10, 'start_ns': 0,
                  'end_ns': 100_000_000_000, 'observed_empty_draft': True}
        table = O.costs(trace(raw), requests, [window], settle=1, minimum=3)
        self.assertEqual(table['widths']['1']['draft_ms'], 4)
        self.assertEqual(table['widths']['1']['cycle_ms'], 20)

    def test_replay_uses_the_servers_float_threshold(self):
        widths = {str(w): {'cycle_ms': 10, 'draft_ms': 0} for w in range(1, 4)}
        result = replay([{'passes': [{'n': 2, 'acc': 2, 'p': [.800000011920929, .9]}]}],
                        Policy(2, p_min=.80000002), widths)
        self.assertEqual(result['output_tokens'], 3)

    def test_measured_validation_overrules_a_wrong_replay_ranking(self):
        def measured(policy, speeds):
            return {'policy': policy, 'samples': [{'repeat': i, 'timings': {'predicted_per_second': value}}
                                                 for i, value in enumerate(speeds)]}
        a, b = {'n_max': 4}, {'n_max': 2}
        spec = {'candidates': [{'policy': a, 'decode_t_s': 200}, {'policy': b, 'decode_t_s': 150}],
                'validation': {'repetitions': 3, 'launch': measured(None, [50, 51, 49]),
                               'policies': [measured(a, [40, 40, 40]), measured(b, [60, 60, 60])]}}
        candidates = validated_estimates(spec, reference_t_s=100)
        self.assertEqual(candidates[0]['policy'], b)
        self.assertEqual(candidates[0]['decode_source'], 'matched-validation-estimate')
        self.assertAlmostEqual(candidates[0]['decode_t_s'], 120)
        self.assertAlmostEqual(candidates[1]['decode_t_s'], 80)

    def test_snapshot_rejects_either_drop_counter_and_io_failure(self):
        shot = {'ok': True, 'cumulative': True, 'event_count': 10, 'event_capacity': 100, 'dropped_event_count': 0,
                'speculative': {'enabled': True, 'ok': True, 'failed': False, 'dropped': 0}}
        O.snapshot_ok(shot)
        for owner, key, value in [('root', 'dropped_event_count', 1), ('root', 'cumulative', False),
                                  ('spec', 'dropped', 1), ('spec', 'failed', True), ('spec', 'enabled', False)]:
            changed = copy.deepcopy(shot)
            (changed if owner == 'root' else changed['speculative'])[key] = value
            with self.assertRaises(SpecError):
                O.snapshot_ok(changed)


class RetainedSnapshots(unittest.TestCase):
    def test_endpoint_404_falls_back_but_snapshot_failure_does_not(self):
        import measure as M
        for status in (404, 500):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                cfg = _merge(DEFAULTS, {'speculation': {'enabled': True}})
                cmd = ('server', '--spec-type', 'draft-mtp')
                runner = ContextRunner(cfg, M.P.ServerArm('test', cmd, ('client',)), None, log=lambda _: None)
                error = urllib.error.HTTPError('http://server/tune/snapshot', status, 'unavailable', {}, None)
                try:
                    with patch('context_runner.subprocess.run', return_value=Mock(stdout='--tune-endpoints', stderr='')), \
                            patch('context_runner.subprocess.Popen') as process, patch('context_runner.M.vram_state', return_value={}), \
                            patch.object(M.P, '_version', return_value='test'), patch.object(M.P, '_wait_for_server'), \
                            patch.object(M.P, '_stop_server'), patch.object(runner, 'request', side_effect=[{}, error]):
                        process.return_value.pid = 123
                        if status == 404:
                            runner._start(cmd, {}, Path(directory))
                            self.assertIs(runner.snapshot_supported, False)
                            self.assertTrue(runner.supported)
                            self.assertIn('DELPHI_SPEC_PLOG', process.call_args.kwargs['env'])
                            runner.close()
                        else:
                            with self.assertRaises(urllib.error.HTTPError):
                                runner._start(cmd, {}, Path(directory))
                            self.assertIsNone(runner.process)
                finally:
                    runner.process = None
                    runner.close()

    def runner(self):
        cfg = _merge(DEFAULTS, {'speculation': {'enabled': True}})
        r = ContextRunner(cfg, Mock(port=18081), None)
        cmd = ('server', '--spec-type', 'draft-mtp', '-ub', '512')
        r.process = Mock()
        r.process.poll.return_value = None
        r.signature = process_key(cmd, {})
        r.launch_command = cmd
        r.supported = True
        return r, cmd

    def test_live_snapshot_preserves_model_load_and_legacy_relaunches(self):
        r, cmd = self.runner()
        r.snapshot_supported = True
        self.assertEqual(r.operation((*cmd[:-1], '256'), {}), 'context')
        r.snapshot_supported = False
        self.assertEqual(r.operation(cmd, {}), 'launch')

    def test_cumulative_capacity_uses_safe_restart(self):
        r, cmd = self.runner()
        r.snapshot_supported = True
        r.snapshot_state = {'event_count': 90, 'event_capacity': 100}
        r.point_events = 30
        self.assertEqual(r.operation(cmd, {}), 'launch')
        r.snapshot_state['event_count'] = 40
        self.assertEqual(r.operation(cmd, {}), 'context')


if __name__ == '__main__':
    unittest.main()
