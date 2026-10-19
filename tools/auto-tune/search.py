"""Nested process/context coordinate search, with coupled batch/cache points and screening."""
from __future__ import annotations

import itertools
import math
import random
import statistics

import knobs as K
import score as S


class NestedSearch:
    def __init__(self, tuner, reference, reserve):
        self.t = tuner
        self.reference = reference
        self.reserve = reserve
        self.index = 0

    def point(self, command, env, name, incumbent):
        self.index += 1
        row = self.t.evaluate(command, env, f'{self.index:03d} {name}', reserve=self.reserve,
                              racing=incumbent['row']['metrics'] if incumbent else None)
        value = S.score(row['metrics'], self.reference, self.t.weights) if row['status'] == 'ok' else None
        point = {'command': tuple(command), 'env': dict(env), 'row': row, 'score': value}
        if self.better(point, self.best):
            self.best = point
        return point

    def better(self, candidate, incumbent):
        return candidate['score'] is not None and candidate['score'] > incumbent['score'] * (1 + self.t.cfg['noise_margin'])

    def screen(self, selected, incumbent):
        if len(selected) <= self.t.cfg['context_search']['screen_threshold']:
            return selected
        # Two finite-domain Morris trajectories. Each step changes one coordinate to the opposite endpoint.
        # Failed boundaries stay in the refinement set; incomplete/raced points cannot supply an elementary effect.
        effects = {k.name: [] for k, _ in selected}
        boundaries = set()
        rng = random.Random(0)
        for trajectory in range(2):
            path = list(selected)
            rng.shuffle(path)
            previous = incumbent
            for knob, levels in path:
                ctx = self.t.context(previous['command'], previous['env'])
                if not knob.applies(ctx):
                    continue
                alternatives = [v for v in levels if v != knob.current(ctx.command, ctx.env)]
                if not alternatives:
                    continue
                try:
                    grid = {v: float(v) for v in levels}
                except ValueError:
                    grid = {v: float(i) for i, v in enumerate(levels)}
                alternatives.sort(key=grid.get)
                level = alternatives[-1] if trajectory == 0 else alternatives[0]
                current = knob.current(ctx.command, ctx.env)
                delta = abs(grid[level] - grid[current]) / (max(grid.values()) - min(grid.values())) if current in grid else 1.0
                command, env = knob.apply(ctx.command, ctx.env, level)
                point = self.point(command, env, 'screen ' + knob.name, None)
                if point['score'] is None:
                    boundaries.add(knob.name)
                    continue
                effects[knob.name].append(abs(math.log(point['score'] / previous['score'])) / max(delta, 1e-12))
                previous = point
        ranked = sorted(selected, key=lambda item: statistics.mean(effects[item[0].name]) if effects[item[0].name] else 0, reverse=True)
        noise = S.race([incumbent['row']['metrics']], incumbent['row']['metrics'], self.t.weights, self.t.cfg['noise_margin'])['threshold_log']
        cutoff = noise if noise is not None else math.log1p(self.t.cfg['noise_margin'])
        survivors = [(k, v) for k, v in ranked if k.name in boundaries or effects[k.name] and statistics.mean(effects[k.name]) > cutoff]
        # Preserve known interactions even when their marginal effects cancel.
        paired = {'-ub', 'LLAMA_MOE_CACHE_AUTO_MARGIN_MIB', '-t', '-tb'}
        names = {k.name for k, _ in survivors} | paired
        self.t.state.setdefault('screening', []).append({'effects_log': effects, 'boundaries': sorted(boundaries), 'survivors': sorted(names & effects.keys())})
        return [(k, v) for k, v in selected if k.name in names]

    def refine(self, start):
        incumbent = start
        selected = self.t.selected(start['command'], start['env'])
        selected = [(k, v) for k, v in selected if k.reload == 'context']
        selected = self.screen(selected, incumbent)
        by_name = {k.name: (k, levels) for k, levels in selected}
        groups = []
        pair = ('-ub', 'LLAMA_MOE_CACHE_AUTO_MARGIN_MIB')
        if all(n in by_name for n in pair):
            groups.append([by_name.pop(n) for n in pair])
        groups += [[by_name[k.name]] for k, _ in selected if k.name in by_name]
        # Revisit coordinates once after an improvement, so a later change can expose an earlier one.
        for sweep in range(2):
            improved = False
            for group in groups:
                ctx = self.t.context(incumbent['command'], incumbent['env'])
                if not all(k.applies(ctx) for k, _ in group):
                    continue
                step = {'loop': 2, 'knob': ' x '.join(k.name for k, _ in group), 'sweep': sweep, 'tested': []}
                best = incumbent
                for values in itertools.product(*(levels for _, levels in group)):
                    command, env = incumbent['command'], incumbent['env']
                    for (knob, _), level in zip(group, values):
                        command, env = knob.apply(command, env, level)
                    if tuple(command) == tuple(incumbent['command']) and env == incumbent['env']:
                        continue
                    candidate = self.point(command, env, ','.join(k.name + '=' + v for (k, _), v in zip(group, values)), incumbent)
                    step['tested'].append({'levels': list(values), 'key': candidate['row']['key'], 'status': candidate['row']['status'], 'score': candidate['score']})
                    if self.better(candidate, best):
                        best = candidate
                if self.better(best, incumbent):
                    step['kept'] = best['row']['key']
                    incumbent = best
                    improved = True
                self.t.state['steps'].append(step)
            if not improved:
                break
        return incumbent

    def run(self, incumbent, placements):
        best = self.refine(incumbent)
        # Each placement gets its own context search, including placements whose starting point loses.
        for option in placements:
            start = self.point(option['command'], option['env'], 'placement ' + option['name'], None)
            self.t.state['steps'].append({'loop': 1, 'knob': 'placement', 'candidate': option['name'], 'key': start['row']['key'], 'status': start['row']['status']})
            if start['score'] is not None:
                candidate = self.refine(start)
                if self.better(candidate, best):
                    best = candidate
        process = [(k, v) for k, v in self.t.selected(best['command'], best['env']) if k.reload == 'process']
        for knob, levels in process:
            origin = best
            for level in levels:
                if level == knob.current(origin['command'], origin['env']):
                    continue
                command, env = knob.apply(origin['command'], origin['env'], level)
                start = self.point(command, env, 'process ' + knob.name + '=' + level, None)
                if start['score'] is not None:
                    candidate = self.refine(start)
                    if self.better(candidate, best):
                        best = candidate
        return self.best
