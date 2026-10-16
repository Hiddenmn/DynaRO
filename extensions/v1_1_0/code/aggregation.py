from __future__ import annotations
import math,statistics

def require(ok,message):
    if not ok: raise ValueError(message)

def mean_tree(values):
    if not values:
        raise ValueError("Empty aggregate")
    first = values[0]
    if isinstance(first, dict):
        if any(set(v) != set(first) for v in values):
            raise ValueError("Metric key mismatch")
        return {k: mean_tree([v[k] for v in values]) for k in first}
    if isinstance(first, list):
        if any(len(v) != len(first) for v in values):
            raise ValueError("Curve length mismatch")
        return [mean_tree([v[i] for v in values]) for i in range(len(first))]
    present = [v for v in values if v is not None]
    if not present:
        return None
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in present):
        raise ValueError("Nonfinite/invalid metric")
    return statistics.fmean(present)

def summarize(rows, panel):
    records, blocks = panel["pairs"], panel["blocks"]
    if [r["pair_id"] for r in rows] != [r["pair_id"] for r in records]:
        raise ValueError("Only complete ordered frozen panels can be summarized")
    per_block = {}
    expected = len(records) // len(blocks)
    for block in blocks:
        selected = [r["scores"] for r in rows if r["block_id"] == block]
        if len(selected) != expected:
            raise ValueError("Unbalanced frozen block panel")
        per_block[block] = mean_tree(selected)
    return {"overall_equal_block_mean": mean_tree(list(per_block.values())),
            "per_block": per_block, "pairs_per_horizon": len(records),
            "spatial_blocks": len(blocks)}
def numeric_mean(values):
    """Recursive descriptive means; absent class values remain null, not zero."""
    nonempty = [v for v in values if v is not None]
    if not nonempty:
        return None
    first = nonempty[0]
    if isinstance(first, dict):
        require(all(set(v) == set(first) for v in nonempty), 'Metric key mismatch')
        return {key: numeric_mean([v[key] for v in nonempty]) for key in first}
    if isinstance(first, list):
        require(all(len(v) == len(first) for v in nonempty), 'Metric length mismatch')
        return [numeric_mean([v[i] for v in nonempty]) for i in range(len(first))]
    return statistics.fmean(nonempty)

def summarize_seed(rows, panel):
    expected = {r['pair_id']: r for r in panel}
    require(len(rows) == len(expected) and {r['pair_id'] for r in rows} == set(expected),
            'Missing/duplicate/unexpected pairs; no complete summary')
    by_block = {}
    for row in rows:
        meta = expected[row['pair_id']]
        require(row['core_id'] == meta['core_id'] and row['block_id'] == meta['block_id'], 'Changed core/block identity')
        by_block.setdefault((row['core_id'], row['block_id']), []).append(row)
    core_ids = sorted({r['core_id'] for r in panel})
    core_means, block_means, coverage = {}, {}, {}
    for core in core_ids:
        block_means[core] = {}
        coverage[core] = {}
        for (c, block), records in by_block.items():
            if c != core:
                continue
            valid = [r['scores'] for r in records if r['scores'] is not None]
            block_means[core][block] = numeric_mean(valid)
            coverage[core][block] = {'pairs': len(records), 'scored': len(valid), 'no_initial_oil': len(records)-len(valid)}
        core_means[core] = numeric_mean(list(block_means[core].values()))
    return {'per_core_equal_block_mean': core_means, 'per_block': block_means, 'coverage': coverage,
            'overall_equal_core_mean': numeric_mean(list(core_means.values()))}
