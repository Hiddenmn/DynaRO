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
from dataclasses import dataclass


@dataclass(frozen=True)
class RegionalPair:
    initial_labels: torch.Tensor
    inlet: torch.Tensor
    connected: torch.Tensor
    actions_a: torch.Tensor
    actions_b: torch.Tensor
    times: torch.Tensor
    target_a: torch.Tensor
    target_b: torch.Tensor
    pair_id: str


def _gpu_inputs(pair: RegionalPair, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in
                 (pair.initial_labels, pair.inlet, pair.connected,
                  pair.actions_a, pair.actions_b, pair.times))


def _cpu_output(output: dict) -> dict:
    return {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for name, value in output.items()}

