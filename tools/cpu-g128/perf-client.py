#!/usr/bin/env python3
"""Collect process CPU counters/samples between the first visible generated chunk and stream completion.

Use as a profile.py extra measurement. The standard SPEED-Bench client remains the primary timing workload.
Counter collection excludes prompt evaluation and the initial generated chunk, and includes all server threads.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('speed_bench', ROOT / 'tools/server/bench/speed-bench/speed_bench.py')
sb = importlib.util.module_from_spec(spec); sys.modules[spec.name] = sb; spec.loader.exec_module(sb)


class Collector:
    def __init__(self, output, name, command):
        self.control = output / (name + '.control')
        self.ack = output / (name + '.ack')
        for p in (self.control, self.ack): os.mkfifo(p)
        self.cfd = os.open(self.control, os.O_RDWR | os.O_NONBLOCK)
        self.afd = os.open(self.ack, os.O_RDWR | os.O_NONBLOCK)
        self.log = (output / (name + '.log')).open('w')
        self.command = [*command, '--delay=-1', '--control=fifo:' + str(self.control) + ',' + str(self.ack)]
        self.proc = subprocess.Popen(self.command, stdout=self.log, stderr=self.log)

    def send(self, command):
        if self.proc.poll() is not None: raise RuntimeError('perf exited: ' + str(self.command))
        os.write(self.cfd, (command + '\n').encode())
        if not select.select([self.afd], [], [], 10)[0]: raise RuntimeError('perf control acknowledgement timed out')
        if b'ack' not in os.read(self.afd, 1024): raise RuntimeError('invalid perf acknowledgement')

    def close(self):
        if self.proc.poll() is None: self.proc.send_signal(signal.SIGINT)
        try: rc = self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill(); rc = self.proc.wait()
        os.close(self.cfd); os.close(self.afd); self.log.close()
        self.control.unlink(); self.ack.unlink()
        return rc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--model-path', required=True, help='Exact target path in the server command; avoids attaching to another server')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--mode', choices=('core', 'memory'), default='core')
    p.add_argument('--osl', type=int, default=128)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = args.output.parent / (args.output.stem + '-perf'); out.mkdir(exist_ok=True)
    matches = []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit(): continue
        try:
            cmd = (d / 'cmdline').read_bytes().split(b'\0')
            if args.model_path.encode() in cmd and b'--port' in cmd and str(args.port).encode() in cmd and Path(os.readlink(d / 'exe')).name == 'llama-server':
                matches.append(int(d.name))
        except (OSError, PermissionError): pass
    if len(matches) != 1: raise RuntimeError('Expected one matching server, found ' + str(matches))
    pid = matches[0]
    events = ['cycles:u', 'instructions:u', 'ex_ret_cops:u', 'ls_dispatch.ld_dispatch:u'] if args.mode == 'core' else [
        'cycles:u', 'instructions:u', 'ls_refills_from_sys.ls_mabresp_lcl_dram:u',
        'ls_refills_from_sys.ls_mabresp_rmt_dram:u', 'ls_hw_pf_dc_fill.ls_mabresp_lcl_dram:u',
        'ls_hw_pf_dc_fill.ls_mabresp_rmt_dram:u']
    collectors = []
    try:
        collectors.append(Collector(out, 'stat', ['perf', 'stat', '-x', ',', '-o', str(out / 'stat.csv'), '-e', ','.join(events), '-p', str(pid)]))
        if args.mode == 'core':
            collectors.append(Collector(out, 'record', ['perf', 'record', '--no-buildid-cache', '-o', str(out / 'perf.data'),
                '-e', 'cycles:u', '-F', '199', '-p', str(pid)]))
        sample_args = argparse.Namespace(bench='qualitative', category='coding', limit=1)
        sample = sb.load_samples(sample_args)[0]
        body = {'messages': [{'role': 'user', 'content': sample.turns[0]}], 'max_tokens': args.osl, 'temperature': 0,
                'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True}}
        url = sb.normalize_base_url(args.url) + '/chat/completions'
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        begin = None; timing = {}; usage = {}; chunks = 0; finish = None
        with urllib.request.urlopen(request, timeout=1800) as response:
            for line in response:
                if not line.startswith(b'data: '): continue
                data = line[6:].strip()
                if data == b'[DONE]': break
                row = json.loads(data)
                if row.get('error'): raise RuntimeError(str(row['error']))
                timing.update(row.get('timings') or {}); usage.update(row.get('usage') or {})
                for choice in row.get('choices') or []:
                    delta = choice.get('delta') or {}
                    if delta.get('content') or delta.get('reasoning_content'):
                        chunks += 1
                        if begin is None:
                            for c in collectors: c.send('enable')
                            begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    finish = choice.get('finish_reason') or finish
        end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        if begin is None: raise RuntimeError('No visible generated chunk; perf was never enabled')
        for c in collectors: c.send('disable')
        commands = [c.command for c in collectors]
        status = [c.close() for c in collectors]; collectors.clear()
        if any(rc not in (0, -signal.SIGINT, 128 + signal.SIGINT) for rc in status): raise RuntimeError('perf failed: ' + str(status))
        n = int(usage.get('completion_tokens') or timing.get('predicted_n') or args.osl)
        result = sb.RequestResult(id=sample.id, category=sample.category, ok=True, turns=1, latency_s=(end-start)/1e9,
            prompt_tokens=int(usage.get('prompt_tokens') or timing.get('prompt_n') or 0), completion_tokens=n,
            total_tokens=int(usage.get('total_tokens') or 0), finish_reason=finish,
            draft_n=int(timing.get('draft_n') or 0), draft_n_accepted=int(timing.get('draft_n_accepted') or 0),
            prompt_ms=timing.get('prompt_ms'), predicted_ms=timing.get('predicted_ms'),
            prompt_per_second=timing.get('prompt_per_second'), predicted_per_second=timing.get('predicted_per_second'),
            request_windows_raw_ns=[[start,end]], error=None)
        payload = {'results': [asdict(result)], 'summary': [sb.summarize_group('overall', [result])],
            'cpu_perf': {'pid': pid, 'commands': commands, 'events': events, 'mode': args.mode,
                'window_raw_ns': [begin,end], 'window_s': (end-begin)/1e9, 'visible_chunks': chunks, 'artifacts': str(out),
                'scope': 'All server threads after first visible generated chunk; excludes prefill and initial generation, includes host work.'}}
        args.output.write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps(payload['cpu_perf']))
    finally:
        for c in collectors: c.close()


if __name__ == '__main__': main()
