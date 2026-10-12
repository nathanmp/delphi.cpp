"""Multiple-choice knapsack for separable recipes: every group picks one option, cost and bytes add over groups.

solve() is an exact dynamic program over bytes in fixed units. Option bytes round up to whole units, so a returned
recipe always fits the budget; the rounding can only exclude recipes within one unit per group of the budget.
"""
from __future__ import annotations

import numpy as np


def solve(groups, budget, max_units=262144):
    """groups: {name: [(bytes, cost, key), ...]}. Minimize the summed cost with summed bytes <= budget.

    Returns ({name: key}, bytes, cost) or None when even the cheapest options exceed the budget."""
    if budget < 0:
        return None
    unit = max(1, -(-int(budget) // max_units))
    cap = int(budget) // unit
    names = sorted(groups)
    inf = np.inf
    dp = np.zeros(cap + 1)  # dp[b] = best cost of the groups so far within b units
    picks = []
    for name in names:
        options = groups[name]
        new = np.full(cap + 1, inf)
        arg = np.full(cap + 1, -1, dtype=np.int16)
        for i, (nbytes, cost, _) in enumerate(options):
            w = -(-int(nbytes) // unit)
            if w > cap:
                continue
            cand = np.full(cap + 1, inf)
            cand[w:] = dp[:cap + 1 - w] + cost
            better = cand < new
            new[better] = cand[better]
            arg[better] = i
        dp = new
        picks.append(arg)
        if not np.isfinite(dp[cap]):
            return None
    choice, b = {}, cap
    for name, arg in zip(reversed(names), reversed(picks)):
        i = int(arg[b])
        option = groups[name][i]
        choice[name] = option
        b -= -(-int(option[0]) // unit)
    return ({n: o[2] for n, o in choice.items()}, sum(o[0] for o in choice.values()), sum(o[1] for o in choice.values()))
