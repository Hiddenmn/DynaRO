"""Selected original definitions; numerical bodies and decorators unchanged."""
from __future__ import annotations
import hashlib
import math
import statistics
import re
from pathlib import Path
import numpy as np
import torch
from regional_budget_objective import PairWindowTargets, prepare_pair_targets, score_full_horizon_pair
from collections import defaultdict
import json
from patent22.actions import build_action_protocol
from .util import RegionalPair, _gpu_inputs, _cpu_output


@torch.inference_mode()
def score_pair(model: RegionalActionSurrogate, pair: RegionalPair,
               device: torch.device) -> dict[str, dict[str, float]]:
    labels, inlet, connected, actions_a, actions_b, times = _gpu_inputs(pair, device)
    output_a = _cpu_output(model(labels, inlet, connected, actions_a, times,
                                 compact_output=True))
    output_b = _cpu_output(model(labels, inlet, connected, actions_b, times,
                                 compact_output=True))
    targets = prepare_pair_targets(pair.initial_labels, pair.target_a, pair.target_b,
                                   output_a, start=0, stop=HORIZON)
    return score_full_horizon_pair(output_a, output_b, targets)


STREAMS = ("P1-22-TEST-A", "P1-22-TEST-B", "P1-22-TEST-C")


SEEDS = (20260811, 20260812, 20260813)


ARMS = ("response0", "response50", "du_gradcos", "stateguard")


PAIR_COUNT = 600


BLOCK_COUNT = 5


PAIRS_PER_BLOCK = 120


HORIZON = 25


INTERVALS = ("all_25", "steps_1_5", "steps_6_15", "steps_16_25")


MEASURES = ("site", "region", "response", "zero_response",
            "unchanged_site", "unchanged_region")


def test_groups(records: list[dict]) -> list[tuple[str, str, list[tuple[int, int, str]]]]:
    """Pair by frozen metadata, without reading outcomes."""
    sequences = build_action_protocol(HORIZON)["groups"]["paired_order"]
    orders = defaultdict(list)
    for sequence_id in sequences:
        orders[sequence_id.split("__", 1)[0]].append(sequence_id)
    if len(orders) != 10 or any(len(ids) != 2 for ids in orders.values()):
        raise ValueError("Canonical action protocol lacks ten reversal pairs")
    lookup = {}
    stages = defaultdict(set)
    for index, row in enumerate(records):
        if (row.get("split") != "locked_test" or int(row.get("n_steps", -1)) != HORIZON
                or row.get("sequence_id") not in sequences
                or row.get("stream_id") not in STREAMS):
            raise ValueError("Noncanonical TEST metadata")
        block, stage = str(row["block_id"]), str(row["stage"])
        key = (block, stage, str(row["sequence_id"]), str(row["stream_id"]))
        if key in lookup:
            raise ValueError("Duplicate TEST metadata cell")
        lookup[key] = index
        stages[block].add(stage)
    if (len(records) != 1200 or len(stages) != BLOCK_COUNT
            or any(len(values) != 4 for values in stages.values())):
        raise ValueError("Expected five TEST blocks with four stages each")
    groups = []
    used = []
    for block in sorted(stages):
        for stage in sorted(stages[block]):
            members = []
            for pair_id in sorted(orders):
                first, second = sorted(orders[pair_id])
                for stream in STREAMS:
                    a = lookup[(block, stage, first, stream)]
                    b = lookup[(block, stage, second, stream)]
                    if records[a]["noise_seed_sha256"] != records[b]["noise_seed_sha256"]:
                        raise ValueError("Reversed TEST trajectories must share their noise stream")
                    members.append((a, b, f"{block}|{stage}|{pair_id}|{stream}"))
                    used.extend((a, b))
            if len(members) != 30:
                raise ValueError("Incomplete TEST block/stage pair grid")
            groups.append((block, stage, members))
    if len(groups) != 20 or len(used) != 1200 or len(set(used)) != 1200:
        raise ValueError("600 TEST pairs must exhaust all 1200 trajectories")
    return groups


def panel_keys() -> tuple[str, ...]:
    return tuple(f"{arm}/{seed}" for seed in SEEDS for arm in ARMS) + ("pcgrad/20260811",)


def summarize_arm(rows: list[dict]) -> dict:
    blocks = sorted({row["block_id"] for row in rows})
    if (len(rows) != PAIR_COUNT or len({row["pair_id"] for row in rows}) != PAIR_COUNT
            or len(blocks) != BLOCK_COUNT):
        raise ValueError("Final TEST summary requires 600 unique pairs and five blocks")

    def mean_scores(selected: list[dict]) -> dict:
        return {interval: {measure: statistics.mean(
            row["scores"][interval][measure] for row in selected)
            for measure in MEASURES} for interval in INTERVALS}

    per_block = {}
    for block in blocks:
        selected = [row for row in rows if row["block_id"] == block]
        if len(selected) != PAIRS_PER_BLOCK:
            raise ValueError("Each TEST block must contain 120 pairs")
        per_block[block] = mean_scores(selected)
    overall = {interval: {measure: statistics.mean(
        per_block[block][interval][measure] for block in blocks)
        for measure in MEASURES} for interval in INTERVALS}
    return {"overall_equal_block_mean": overall, "per_block": per_block,
            "pair_count": PAIR_COUNT, "spatial_block_count": BLOCK_COUNT}


def _relative_gain(reference: float, candidate: float) -> float:
    if reference <= 0:
        raise ValueError("Relative test comparison has a nonpositive reference")
    return (reference - candidate) / reference


def decide(summaries: dict[str, dict]) -> dict:
    primary = {key: value["overall_equal_block_mean"]["all_25"]
               for key, value in summaries.items()}
    mean = {arm: {metric: statistics.mean(
        primary[f"{arm}/{seed}"][metric] for seed in SEEDS)
        for metric in ("site", "response")} for arm in ARMS}
    state, zero, fixed, du = (mean[arm] for arm in
                              ("stateguard", "response0", "response50", "du_gradcos"))
    zero_seed = [primary[f"stateguard/{seed}"]["response"]
                 < primary[f"response0/{seed}"]["response"] for seed in SEEDS]
    fixed_seed = [
        _relative_gain(primary[f"response50/{seed}"]["site"],
                       primary[f"stateguard/{seed}"]["site"]) >= 0.01
        and _relative_gain(primary[f"response50/{seed}"]["response"],
                           primary[f"stateguard/{seed}"]["response"]) >= -0.01
        for seed in SEEDS
    ]
    du_seed = [
        primary[f"stateguard/{seed}"]["site"] < primary[f"du_gradcos/{seed}"]["site"]
        and primary[f"stateguard/{seed}"]["response"]
        < primary[f"du_gradcos/{seed}"]["response"] for seed in SEEDS
    ]
    reduction_vs_zero = _relative_gain(zero["response"], state["response"])
    site_change_vs_zero = -_relative_gain(zero["site"], state["site"])
    site_gain_vs_fixed = _relative_gain(fixed["site"], state["site"])
    response_change_vs_fixed = -_relative_gain(fixed["response"], state["response"])
    return {
        "three_seed_equal_block_then_equal_seed_mean": mean,
        "vs_response0": {
            "response_relative_reduction": reduction_vs_zero,
            "site_relative_degradation": site_change_vs_zero,
            "favourable_response_seeds": sum(zero_seed),
            "pass": reduction_vs_zero >= 0.10 and sum(zero_seed) >= 2
                    and site_change_vs_zero <= 0.01,
        },
        "vs_response50": {
            "site_relative_reduction": site_gain_vs_fixed,
            "response_relative_degradation": response_change_vs_fixed,
            "seeds_meeting_both_conditions": sum(fixed_seed),
            "pass": site_gain_vs_fixed >= 0.01 and response_change_vs_fixed <= 0.01
                    and sum(fixed_seed) >= 2,
        },
        "vs_du_gradcos": {
            "site_relative_reduction": _relative_gain(du["site"], state["site"]),
            "response_relative_reduction": _relative_gain(du["response"], state["response"]),
            "seeds_better_on_both": sum(du_seed),
            "pass": state["site"] < du["site"] and state["response"] < du["response"]
                    and sum(du_seed) >= 2,
        },
        "pcgrad_seed20260811_descriptive_only": {
            metric: primary["pcgrad/20260811"][metric]
                    - primary["stateguard/20260811"][metric]
            for metric in ("site", "response")
        },
    }


def software_self_test() -> None:
    sequences = build_action_protocol(25)["groups"]["paired_order"]
    synthetic = []
    for block in range(5):
        for stage in range(4):
            for stream in STREAMS:
                for sequence_id in sequences:
                    synthetic.append({
                        "split": "locked_test", "block_id": f"B{block}",
                        "stage": str(stage), "stream_id": stream,
                        "sequence_id": sequence_id, "n_steps": 25,
                        "noise_seed_sha256": f"B{block}|{stage}|{stream}",
                    })
    groups = test_groups(synthetic)
    if sum(len(members) for _, _, members in groups) != PAIR_COUNT:
        raise AssertionError("Synthetic TEST pairing did not produce 600 pairs")
    scores = {interval: {measure: 0.1 for measure in MEASURES}
              for interval in INTERVALS}
    rows = [{"pair_id": pair_id, "block_id": block, "stage": stage,
             "scores": scores} for block, stage, members in groups
            for _, _, pair_id in members]
    if summarize_arm(rows)["overall_equal_block_mean"]["all_25"]["site"] != 0.1:
        raise AssertionError("Synthetic equal-block aggregation changed")
    if tuple(panel_keys()) != tuple(f"{arm}/{seed}" for seed in SEEDS for arm in ARMS) + (
            "pcgrad/20260811",):
        raise AssertionError("Frozen TEST checkpoint panel changed")
    fixture = {
        "response0": (0.020, 0.0020),
        "response50": (0.021, 0.0011),
        "du_gradcos": (0.020, 0.0012),
        "stateguard": (0.019, 0.0010),
        "pcgrad": (0.020, 0.0012),
    }
    fake = {key: {"overall_equal_block_mean": {"all_25": {
        "site": fixture[key.split("/")[0]][0],
        "response": fixture[key.split("/")[0]][1],
    }}} for key in panel_keys()}
    verdict = decide(fake)
    if not all(verdict[name]["pass"] for name in
               ("vs_response0", "vs_response50", "vs_du_gradcos")):
        raise AssertionError("Synthetic favourable decision did not pass")
    fake["du_gradcos/20260813"]["overall_equal_block_mean"]["all_25"]["site"] = 0.018
    fake["du_gradcos/20260812"]["overall_equal_block_mean"]["all_25"]["site"] = 0.018
    if decide(fake)["vs_du_gradcos"]["pass"]:
        raise AssertionError("Two unfavourable seed comparisons must fail")

