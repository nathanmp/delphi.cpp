"""Retain model weights between loop-2 points, using the profiler's lifecycle and clients."""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import knobs as K
import measure as M
import score as S
from config import digest
from verify_client import verify

# Everything else in the command is part of the process identity. In particular, NUMA and device placement are fixed.
FIELDS = {
    '-c': ('ctx_size', int), '-b': ('batch_size', int), '-ub': ('ubatch_size', int),
    '-ctk': ('cache_type_k', str), '-ctv': ('cache_type_v', str), '-fa': ('flash_attn', str),
    '-t': ('threads', int), '-tb': ('threads_batch', int), '-ubd': ('ubatch_size_draft', int),
    '--spec-draft-threads': ('threads_draft', int), '--spec-draft-threads-batch': ('threads_batch_draft', int),
}


def process_key(command, env):
    cmd = tuple(command)
    for flag in FIELDS:
        cmd = K.remove_option(cmd, flag)
    e = {k: v for k, v in env.items() if K.ENV_RELOAD.get(k, 'process') != 'context'}
    return digest([cmd, e])


def payload(command, env, defaults, active_env):
    out = {field: convert(K.option_value(command, flag)) if K.option_value(command, flag) is not None else defaults[field]
           for flag, (field, convert) in FIELDS.items() if field in defaults or K.option_value(command, flag) is not None}
    out['env'] = {name: env.get(name) for name in set(active_env) | set(env)
                  if K.ENV_RELOAD.get(name, 'process') == 'context' and active_env.get(name) != env.get(name)}
    if out.get('flash_attn') in ('enabled', 'disabled'):
        out['flash_attn'] = 'on' if out['flash_attn'] == 'enabled' else 'off'
    return out


class ContextRejected(M.LaunchFailed):
    """The hook restored the preceding context; the failed point is infeasible."""


class ContextRunner:
    def __init__(self, cfg, arm, cpu_budget, log=print):
        self.cfg, self.arm, self.cpu, self.log = cfg, arm, cpu_budget, log
        self.process = None
        self.handles = contextlib.ExitStack()
        self.supported = None
        self.help_supported = None
        self.signature = None
        self.active = None
        self.baseline = None
        self.audit = None
        self.stats = {'launches': 0, 'recreates': 0, 'launch_wall_s': 0.0, 'recreate_wall_s': 0.0, 'measurement_wall_s': 0.0}
        self.url = f'http://127.0.0.1:{arm.port}'

    def spec_enabled(self, command):
        from spec_runner import unavailable
        return self.cfg['speculation']['enabled'] and unavailable(command) is None

    def request(self, body=None, *, path='/tune/context'):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.arm.request_timeout_s) as response:
            return json.loads(response.read())

    def operation(self, command, env):
        if self.cfg['trace'] or self.spec_enabled(command) or self.supported is False or self.process is None or self.process.poll() is not None:
            return 'launch'
        # A resumed search can first load an explicit level. Its GET value is not the CLI default of an omitted flag.
        if any(K.option_value(command, flag) is None and K.option_value(self.launch_command, flag) is not None for flag in FIELDS):
            return 'launch'
        return 'context' if self.signature == process_key(command, env) else 'launch'

    def close(self):
        try:
            if self.process is not None:
                M.P._stop_server(self.process)
        finally:
            self.process = None
            self.handles.close()
            self.handles = contextlib.ExitStack()

    def _start(self, command, env, directory):
        self.close()
        started = time.monotonic()
        self.stats['launches'] += 1
        try:
            full_env = {**os.environ, **self.cfg['env'], **env}
            if self.help_supported is None:
                help_result = subprocess.run([command[0], '--help'], env=full_env, capture_output=True, text=True, timeout=60)
                self.help_supported = '--tune-endpoints' in help_result.stdout + help_result.stderr
            cmd = tuple(command)
            if self.help_supported and '--tune-endpoints' not in cmd:
                cmd = (*cmd, '--tune-endpoints')
            self.baseline = M.vram_state()
            stdout = self.handles.enter_context((directory / 'server.stdout.log').open('w'))
            stderr = self.handles.enter_context((directory / 'server.stderr.log').open('w'))
            self.server_logs = [directory / f'server.{stream}.log' for stream in ('stdout', 'stderr')]
            self.audit = self.handles.enter_context(M.CpuSampler(command, self.cpu)) if self.cpu else None
            values = {'port': str(self.arm.port), 'url': self.url, 'artifact_dir': str(directory), 'output': str(directory / 'client.json')}
            # Context measurements cannot use a trace spanning other contexts. Traced runs use the fresh-launch path.
            full_env.update(M.P._trace_env(directory / 'unused.trace', 'off', 0, 0))
            if self.spec_enabled(command):
                self.trace_path = directory / 'loop3.trace'
                self.plog_path = directory / 'mtp-probabilities.jsonl'
                # The log appends and opens its path once per process. Never append to a previous attempt.
                self.plog_path.unlink(missing_ok=True)
                for key in list(full_env):
                    if key.startswith('GGML_EXEC_TRACE'):
                        del full_env[key]
                full_env.update(GGML_EXEC_TRACE=str(self.trace_path), GGML_EXEC_TRACE_MAX_EVENTS=str(self.cfg['speculation']['trace_max_events']),
                                DELPHI_SPEC_PLOG=str(self.plog_path))
            self.process = subprocess.Popen(M.P._render(cmd, values), cwd=M.ROOT, env=full_env, text=True, stdout=stdout, stderr=stderr)
            (directory / 'launch.json').write_text(json.dumps({'command': list(cmd), 'env': dict(env), 'pid': self.process.pid,
                'binary_version': M.P._version(command[0], env), 'models': M.P._model_identity(self.arm.model_paths)}, indent=2) + '\n')
            M.P._wait_for_server(self.url, self.process, self.arm.ready_timeout_s)
            self.signature = process_key(command, env)
            self.launch_command = tuple(command)
            self.active = (tuple(command), dict(env))
            try:
                self.defaults = self.request()
                self.current_state = self.defaults
                self.supported = True
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                self.supported = False
                self.defaults = {}
                self.log('context: endpoints absent (404); use full launches')
        except BaseException:
            self.close()
            raise
        finally:
            self.stats['launch_wall_s'] += time.monotonic() - started

    def _recreate(self, command, env, directory):
        if self.active == (tuple(command), dict(env)):
            return
        body = payload(command, env, self.defaults, self.active[1])
        (directory / 'context-request.json').write_text(json.dumps(body, indent=2) + '\n')
        self.stats['recreates'] += 1
        started = time.monotonic()
        try:
            response = self.request(body)
            (directory / 'context-response.json').write_text(json.dumps(response, indent=2) + '\n')
            if not response.get('ok'):
                raise M.LaunchFailed('context hook did not return ok')
            self.active = (tuple(command), dict(env))
            self.current_state = response
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors='replace')
            (directory / 'context-error.txt').write_text(detail)
            if error.code == 404:
                self.supported = False
                self.close()
                raise M.LaunchFailed('context endpoint disappeared; subsequent points use full launches') from error
            # GET succeeds only after restoration. A dead or unavailable server must be launched again.
            restored = self.request()
            (directory / 'context-restored.json').write_text(json.dumps(restored, indent=2) + '\n')
            previous = getattr(self, 'current_state', self.defaults)
            fields = {field for field, _ in FIELDS.values()}
            if (any(restored.get(k) != v for k, v in previous.items() if k in fields)
                    or restored.get('env', {}) != getattr(self, 'current_state', {}).get('env', {})):
                self.close()
                raise M.LaunchFailed('the context hook did not restore the previous parameters') from error
            raise ContextRejected(f'context rejected (HTTP {error.code}); old parameters restored: {detail}') from error
        finally:
            self.stats['recreate_wall_s'] += time.monotonic() - started

    def _client(self, command, env, values, directory, name):
        completed = M.P._run(M.P._render(command, values), env, timeout=self.arm.request_timeout_s)
        (directory / f'{name}.stdout.log').write_text(completed.stdout)
        (directory / f'{name}.stderr.log').write_text(completed.stderr)
        if completed.returncode:
            raise M.LaunchFailed(f'{name} failed ({completed.returncode}); see {directory}')

    def warnings(self, offsets):
        chunks = []
        for path in getattr(self, 'server_logs', []):
            with path.open('rb') as handle:
                handle.seek(offsets.get(path, 0))
                chunks.append(handle.read().decode(errors='replace'))
        return M.placement_warnings('\n'.join(chunks))

    def _measure(self, command, env, directory):
        arm = M.measurement_arm(self.cfg, self.arm, command, env, 'context', self.cpu)
        values = {'url': self.url, 'port': str(arm.port), 'output': str(directory / 'client.json'), 'artifact_dir': str(directory)}
        self._client(arm.client_command, env, values, directory, 'client')
        decode = M.P._speed_bench_metrics(json.loads((directory / 'client.json').read_text()))
        out = {'decode_t_s': decode.get('predicted_t_s'), 'draft_n': decode.get('draft_n'),
               'draft_n_accepted': decode.get('draft_n_accepted'), 'completion_tokens': decode.get('completion_tokens'), 'prefill': {}}
        if self.cpu:
            out.update(json.loads((directory / 'verification.json').read_text()))
        else:
            with urllib.request.urlopen(self.url + '/props', timeout=30) as response:
                out.update(verify(json.load(response), self.cfg['context']))
        for measurement in arm.extra_measurements:
            path = directory / f'{measurement.name}.json'
            self._client(measurement.command, env, {**values, 'output': str(path)}, directory, measurement.name)
            m = M.P._server_metrics(measurement.result_kind, json.loads(path.read_text()))
            out['prefill'][str(m['target_tokens'])] = {'ttft_s': m.get('latency_s'), 't_s': m.get('prompt_t_s'), 'prompt_ms': m.get('prompt_ms')}
        return out

    def measure(self, command, env, name, incumbent=None, continue_measure=lambda: True):
        """Measure complete repeats, racing only against an incumbent with a measured repeat spread."""
        directory = Path(self.cfg['output']) / 'artifacts' / (name.replace(' ', '_').replace('=', '-') + '-' + digest([command, env]))
        directory.mkdir(parents=True, exist_ok=True)
        samples = []
        started = time.monotonic()
        with self.cpu.affinity() if self.cpu else contextlib.nullcontext():
            if self.cfg['trace'] and not self.spec_enabled(command):
                # Keep trace windows and fresh-launch validation identical to M0.
                self.close()
                self.stats['launches'] += 1
                try:
                    return M.launch(self.cfg, self.arm, command, env, directory.name, directory.parent, self.cpu)
                finally:
                    self.stats['launch_wall_s'] += time.monotonic() - started
            try:
                offsets = {p: p.stat().st_size for p in getattr(self, 'server_logs', [])} if self.operation(command, env) == 'context' else {}
                if self.operation(command, env) == 'launch':
                    self._start(command, env, directory)
                else:
                    self._recreate(command, env, directory)
                if self.arm.warmup_command:
                    values = {'url': self.url, 'port': str(self.arm.port), 'artifact_dir': str(directory), 'output': str(directory / 'warmup.json')}
                    self._client(self.arm.warmup_command, env, values, directory, 'warmup')
                with M.VramSampler() as vram:
                    vram.baseline = self.baseline
                    collector = None
                    spec_status = None
                    if self.spec_enabled(command):
                        from spec_runner import Collector
                        from speculation import Unavailable
                        collector = Collector(self, command, env, directory, continue_measure)
                        try:
                            collector.collect()
                        except Unavailable as error:
                            collector = None
                            spec_status = {'status': 'unavailable', 'reason': str(error)}
                        except urllib.error.HTTPError as error:
                            if error.code != 404:
                                raise
                            collector = None
                            spec_status = {'status': 'unavailable', 'reason': '/speculative is absent (404); request the server control hook.'}
                        # Collection restores the launch policy. Warm it again after the synthetic widths.
                        if self.arm.warmup_command:
                            self._client(self.arm.warmup_command, env, values, directory, 'restored-warmup')
                    elif self.cfg['speculation']['enabled']:
                        from spec_runner import unavailable
                        spec_status = {'status': 'unavailable', 'reason': unavailable(command)}
                    for rep in range(self.cfg['context_search']['reps']):
                        rep_dir = directory / f'repeat-{rep + 1}'
                        rep_dir.mkdir(exist_ok=True)
                        sample_started = time.monotonic()
                        sample = self._measure(command, env, rep_dir)
                        self.stats['measurement_wall_s'] += time.monotonic() - sample_started
                        samples.append(sample)
                        racing = S.race(samples, incumbent, S.metric_weights(self.cfg['weights'], self.cfg['prefill']['lengths']),
                                        self.cfg['noise_margin']) if incumbent else {'stopped': False}
                        if racing['stopped'] or not continue_measure():
                            break
                    vram._sample()
                result = S.aggregate(samples)
                result.update(vram=vram.summary(), racing=racing, wall_s=time.monotonic() - started,
                              context_hooks=self.supported, measurement_complete=len(samples) == self.cfg['context_search']['reps'])
                if self.audit:
                    self.audit.sample()
                    result['cpu_budget'] = self.audit.summary()
                    if self.audit.violations or not self.audit.samples:
                        raise M.LaunchFailed('CPU budget audit failed: ' + str(result['cpu_budget']))
                result['placement_warnings'] = self.warnings(offsets)
                if collector:
                    self.close()
                    result['speculation'] = collector.finish()
                elif spec_status:
                    result['speculation'] = spec_status
                (directory / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
                if not self.supported:
                    self.close()
                return result
            except ContextRejected:
                raise
            except BaseException:
                self.close()
                raise
