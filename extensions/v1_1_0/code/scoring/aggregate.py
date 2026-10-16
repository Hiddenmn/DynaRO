"""Selected original definitions; numerical bodies and decorators unchanged."""
from __future__ import annotations
import math
import statistics
from types import SimpleNamespace


study = SimpleNamespace(BLOCKS=('z128_y384_x256', 'z256_y0_x128', 'z384_y0_x128', 'z384_y384_x0', 'z512_y256_x256'))


def mean_tree(values):
    if isinstance(values[0], dict):
        if any(set(v) != set(values[0]) for v in values):
            raise ValueError('Metric keys differ')
        return {k: mean_tree([v[k] for v in values]) for k in values[0]}
    if isinstance(values[0], list):
        if any(len(v) != len(values[0]) for v in values):
            raise ValueError('Metric lengths differ')
        return [mean_tree([v[i] for v in values]) for i in range(len(values[0]))]
    present = [v for v in values if v is not None]
    if not present:
        return None
    if any(not math.isfinite(v) for v in present):
        raise ValueError('Nonfinite scores are not accepted')
    return statistics.fmean(present)


def summarize(rows, contract):
    if ([r['pair_id'] for r in rows] != [r['pair_id'] for r in contract['panel']]
            or len(rows) != 600):
        raise ValueError('Only the full frozen600-pair panel may be summarized')
    per_block = {}
    for block in study.BLOCKS:
        selected = [r['scores'] for r in rows if r['block_id'] == block]
        if len(selected) != 120:
            raise ValueError('Require exactly120 pairs per TEST block')
        per_block[block] = mean_tree(selected)
    return {'overall_equal_block_mean': mean_tree(list(per_block.values())),
            'per_block': per_block, 'pairs_per_horizon': 600, 'spatial_blocks': 5}


def prefix50(pair):
    labels, inlet, connected, aa, ab, times, ta, tb = pair
    if (len(aa) != 100 or len(ab) != 100 or len(times) != 100
            or len(ta) != 101 or len(tb) != 101):
        raise ValueError('Expected a complete100-step target including initial state')
    return labels, inlet, connected, aa[:50], ab[:50], times[:50], ta[:51], tb[:51]

