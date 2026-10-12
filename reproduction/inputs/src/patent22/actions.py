"""Result-blind action library and predeclared action-sequence families."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    fluid_type: str
    concentration: float
    rate_factor: float
    wettability_shift: float
    cost_factor: float
    description: str

    def source_strategy(self) -> Dict[str, object]:
        d = asdict(self)
        d["strategy_id"] = d.pop("action_id")
        return d


ACTION_LIBRARY: Dict[str, ActionSpec] = {
    "S1": ActionSpec("S1", "water", 0.00, 0.65, 0.00, 1.00, "low-rate water"),
    "S2": ActionSpec("S2", "water", 0.00, 1.00, 0.00, 1.20, "medium-rate water"),
    "S3": ActionSpec("S3", "water", 0.00, 1.45, 0.00, 1.60, "high-rate water"),
    "S4": ActionSpec("S4", "nanofluid_proxy", 0.06, 1.00, 0.16, 1.55, "0.06 wt% proxy"),
    "S5": ActionSpec("S5", "nanofluid_proxy", 0.12, 1.00, 0.28, 1.90, "0.12 wt% proxy"),
}

ACTION_ORDER = tuple(ACTION_LIBRARY)
ACTION_VECTOR_FIELDS = (
    "rate_factor_over_1p45",
    "concentration_over_0p12",
    "wettability_shift_over_0p28",
    "is_nanofluid_proxy",
)


def action_vector(action_id: str) -> np.ndarray:
    a = ACTION_LIBRARY[action_id]
    return np.asarray(
        [
            a.rate_factor / 1.45,
            a.concentration / 0.12,
            a.wettability_shift / 0.28,
            float(a.fluid_type == "nanofluid_proxy"),
        ],
        dtype=np.float32,
    )


def build_action_protocol(n_steps: int = 25) -> Dict[str, object]:
    """Build result-blind, dose-matched order pairs.

    For every unordered action pair {A, B}, the protocol freezes A*10 -> B*15
    and its exact count-matched reversal B*15 -> A*10. Thus any within-pair
    contrast changes order only; it never changes the total exposure to either
    action. The same templates can be evaluated on disjoint spatial splits.
    """

    if n_steps != 25:
        raise ValueError("The canonical paired-order protocol is fixed at 25 steps")

    paired_order: Dict[str, List[str]] = {}
    for a, b in combinations(ACTION_ORDER, 2):
        pair = f"{a}_{b}"
        paired_order[f"pair_{pair}__{a}10_then_{b}15"] = [a] * 10 + [b] * 15
        paired_order[f"pair_{pair}__{b}15_then_{a}10"] = [b] * 15 + [a] * 10

    groups = {"paired_order": paired_order}

    if len(paired_order) != 20 or len({tuple(v) for v in paired_order.values()}) != 20:
        raise RuntimeError("Expected 20 unique paired-order sequences")
    for a, b in combinations(ACTION_ORDER, 2):
        pair_records = [v for k, v in paired_order.items() if k.startswith(f"pair_{a}_{b}__")]
        if len(pair_records) != 2:
            raise RuntimeError(f"Missing order reversal for {a}, {b}")
        for sequence in pair_records:
            if sequence.count(a) != 10 or sequence.count(b) != 15:
                raise RuntimeError(f"Dose mismatch for {a}, {b}")

    return {
        "protocol_id": "P1-PATENT22-ACTIONS-V1-20260811",
        "n_steps": n_steps,
        "action_vector_fields": ACTION_VECTOR_FIELDS,
        "action_library": {k: asdict(v) for k, v in ACTION_LIBRARY.items()},
        "groups": groups,
        "split_sequence_policy": {
            "train": ["paired_order"],
            "dev": ["paired_order"],
            "locked_test": ["paired_order"],
            "existing_fixed_action_fields": (
                "AUXILIARY_REGRESSION_ONLY_NOT_NEWLY_GENERATED_OR_IN_CANONICAL_TRAIN_PRIMARY"
            ),
        },
        "paired_order_rule": (
            "Ten unordered action pairs x two exact count-matched reversals. "
            "Each pair has A*10 then B*15 and B*15 then A*10; switch time moves "
            "from t=10 to t=15 while per-action counts remain unchanged."
        ),
        "history_rule": (
            "The complete sequence_id and action list are fixed before simulation. "
            "Transitions from one trajectory remain contiguous; the recurrent hidden "
            "state is never initialized from a future transition."
        ),
        "noise_rule": (
            "For stream_id P1-22-DEV-A/B and canonical action-independent "
            "case_id=f'{block_id}|{stage}|{time_step}', seed=uint64_big_endian(SHA256(stream_id+'|'"
            "+case_id)[0:8]) mod 2^32. Split, crop, sequence_id and action are excluded; "
            "paired reversals therefore share all 25 seed values."
        ),
    }


def sequence_vectors(action_ids: Iterable[str], horizon_steps: int = 25) -> np.ndarray:
    del horizon_steps  # kept only for backward-compatible generator calls
    return np.stack([action_vector(a) for a in action_ids], axis=0)
