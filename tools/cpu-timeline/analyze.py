#!/usr/bin/env python3
"""Summarize a GGML_CPU_TIMELINE recording (ggml/src/ggml-cpu/cpu-timeline.cpp).

Each CPU graph compute (one scheduler split) is broken into, per call:
  start   main-thread call -> last worker entering the parallel region (fork + wake + pinning)
  work    per node: the slowest thread's compute time
  skew    per node: slowest thread end - median thread end (load imbalance / stragglers)
  sync    per node: barrier exit after the slowest thread finished (barrier propagation)
  finish  last worker exit -> main-thread return (join + affinity reset)
plus the gap from one call's return to the next call (GPU work, copies, host code outside the CPU backend).
Times are summed per call class (first node name with the layer number stripped) and reported per cycle when --per is
given (e.g. the number of verify passes), otherwise as totals.
"""
import argparse, re, struct, sys
from collections import defaultdict
import numpy as np

CALL = np.dtype([('t_call', '<i8'), ('t_ret', '<i8'), ('n_nodes', '<i4'), ('n_threads', '<i4'), ('name', '<i4'), ('pad', '<i4')])
ENTER = np.dtype([('call', '<i4'), ('cpu', '<i4'), ('t_enter', '<i8'), ('t_exit', '<i8')])
NODE = np.dtype([('call', '<i4'), ('node', '<i2'), ('n_fused', '<i2'), ('name', '<i4'), ('op', '<i2'), ('experts', '<i2'), ('rows', '<i2'), ('pad', '<i2'), ('t0', '<i8'), ('t1', '<i8'), ('t2', '<i8')])


def load(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:5] == b'GCTL1', 'not a cpu timeline'
    max_threads, n_calls = struct.unpack_from('<ii', data, 8)
    off = 16
    calls = np.frombuffer(data, CALL, n_calls, off); off += n_calls * CALL.itemsize
    enters, nodes = [], []
    for t in range(max_threads):
        n_enter, n_node = struct.unpack_from('<qq', data, off); off += 16
        e = np.frombuffer(data, ENTER, n_enter, off); off += n_enter * ENTER.itemsize
        n = np.frombuffer(data, NODE, n_node, off); off += n_node * NODE.itemsize
        if n_enter:
            enters.append((t, e)); nodes.append((t, n))
    (n_names,) = struct.unpack_from('<i', data, off); off += 4
    names = []
    for _ in range(n_names):
        (ln,) = struct.unpack_from('<i', data, off); off += 4
        names.append(data[off:off + ln].decode(errors='replace')); off += ln
    return calls, enters, nodes, names


def klass(name):
    return re.sub(r'-\d+$', '', name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--per', type=float, default=0, help='divide totals by this count (e.g. verify passes or tokens)')
    ap.add_argument('--threads', action='store_true', help='per-thread table')
    a = ap.parse_args()
    calls, enters, nodes, names = load(a.file)
    n_calls = len(calls)
    threads = [t for t, _ in enters]
    nt = len(threads)
    # dense [thread, call] arrays for entry/exit; nodes grouped by (call, node)
    t_enter = np.full((nt, n_calls), np.nan); t_exit = np.full((nt, n_calls), np.nan); cpu = np.full((nt, n_calls), -1)
    for i, (_, e) in enumerate(enters):
        t_enter[i, e['call']] = e['t_enter']; t_exit[i, e['call']] = e['t_exit']; cpu[i, e['call']] = e['cpu']
    all_nodes = np.concatenate([n for _, n in nodes])
    thread_of = np.concatenate([np.full(len(n), i) for i, (_, n) in enumerate(nodes)])
    order = np.lexsort((thread_of, all_nodes['node'], all_nodes['call']))
    all_nodes = all_nodes[order]; thread_of = thread_of[order]
    key = all_nodes['call'].astype(np.int64) * 65536 + all_nodes['node']
    bounds = np.flatnonzero(np.diff(key)) + 1
    groups = np.split(np.arange(len(key)), bounds)

    cls = [klass(names[c['name']]) if c['name'] >= 0 else '(no nodes)' for c in calls]
    acc = defaultdict(lambda: defaultdict(float))
    node_acc = defaultdict(lambda: defaultdict(float))
    per_thread_work = np.zeros(nt); per_thread_wait = np.zeros(nt)
    last_end = np.full(n_calls, np.nan)
    for g in groups:
        r = all_nodes[g]; c = int(r['call'][0]); k = cls[c]
        work = (r['t1'] - r['t0']) / 1e6
        ends = r['t1']; slow = ends.max()
        nm = next((names[x] for x in r['name'] if x >= 0), '?')
        a_ = acc[k]
        a_['work'] += work.max(); a_['skew'] += (slow - np.median(ends)) / 1e6
        a_['sync'] += (r['t2'].max() - slow) / 1e6
        b = node_acc[(k, klass(nm))]
        b['n'] += 1; b['experts'] += int(r['experts'].max()); b['rows'] += int(r['rows'].max()); b['work_max'] += work.max(); b['work_med'] += np.median(work); b['skew'] += (slow - np.median(ends)) / 1e6
        per_thread_work[thread_of[g]] += work
        per_thread_wait[thread_of[g]] += (r['t2'] - r['t1']) / 1e6
        last_end[c] = np.nanmax([last_end[c], r['t2'].max()])
    gaps = defaultdict(float)
    for c in range(n_calls):
        k = cls[c]; a_ = acc[k]
        a_['calls'] += 1
        a_['wall'] += (calls['t_ret'][c] - calls['t_call'][c]) / 1e6
        ent = t_enter[:, c]; ex = t_exit[:, c]
        if np.all(np.isnan(ent)):
            continue
        a_['start'] += (np.nanmax(ent) - calls['t_call'][c]) / 1e6
        a_['start_main'] += (ent[threads.index(0)] - calls['t_call'][c]) / 1e6 if 0 in threads and not np.isnan(ent[threads.index(0)]) else 0
        a_['finish'] += (calls['t_ret'][c] - np.nanmax(ex)) / 1e6
        if c + 1 < n_calls:
            gaps[k] += (calls['t_call'][c + 1] - calls['t_ret'][c]) / 1e6
    div = a.per if a.per > 0 else 1.0
    unit = 'ms/cycle' if a.per > 0 else 'ms total'
    span = (calls['t_ret'][-1] - calls['t_call'][0]) / 1e6
    print(f'{n_calls} CPU graph computes over {span:.0f} ms, {nt} threads; {unit}')
    cols = ['calls', 'wall', 'start', 'start_main', 'work', 'skew', 'sync', 'finish']
    print(f"{'class':38s} " + ' '.join(f'{c:>9s}' for c in cols) + f" {'gap after':>9s}")
    tot = defaultdict(float)
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]['wall']):
        print(f'{k[:38]:38s} ' + ' '.join(f'{v[c] / div:9.3f}' for c in cols) + f' {gaps[k] / div:9.3f}')
        for c in cols:
            tot[c] += v[c]
    tot_gap = sum(gaps.values())
    print(f"{'TOTAL':38s} " + ' '.join(f'{tot[c] / div:9.3f}' for c in cols) + f' {tot_gap / div:9.3f}')
    print('\nper node class (ms per ' + ('cycle' if a.per > 0 else 'run') + '): slowest-thread work, median-thread work, skew;'
          ' MUL_MAT_ID: distinct experts per call and slowest-thread us per expert')
    for (k, nm), v in sorted(node_acc.items(), key=lambda kv: -kv[1]['work_max'])[:16]:
        ex = f' experts/call {v["experts"] / v["n"]:5.2f} rows/expert {v["rows"] / v["experts"]:4.2f} us/expert {1000 * v["work_max"] / v["experts"]:6.2f}' if v['experts'] else ''
        print(f'  {k[:30]:30s} {nm[:34]:34s} n={v["n"] / div:7.1f} max {v["work_max"] / div:8.3f} med {v["work_med"] / div:8.3f} skew {v["skew"] / div:7.3f}{ex}')
    if a.threads:
        print('\nthread  cpus                work_ms   barrier_wait_ms')
        for i, t in enumerate(threads):
            cs = sorted(set(cpu[i][cpu[i] >= 0].tolist()))
            print(f'{t:6d}  {str(cs[:4])[:18]:18s} {per_thread_work[i] / div:9.3f} {per_thread_wait[i] / div:9.3f}')


if __name__ == '__main__':
    main()
