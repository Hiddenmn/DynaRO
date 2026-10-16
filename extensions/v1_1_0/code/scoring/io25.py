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


ACTION_VECTORS = np.asarray([
    [0.65 / 1.45, 0, 0, 0], [1 / 1.45, 0, 0, 0], [1.45 / 1.45, 0, 0, 0],
    [1 / 1.45, 0.06 / 0.12, 0.16 / 0.28, 1],
    [1 / 1.45, 0.12 / 0.12, 0.28 / 0.28, 1],
], dtype=np.float32)


def read_trajectory(path, record, horizon):
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"]
        connected = data["connected_pore"].astype(bool, copy=True)
        indexes = data["action_index"].astype(np.int64, copy=True)
        if (labels.dtype != np.uint8 or labels.shape != (horizon + 1, 128, 128, 128)
                or connected.shape != (128, 128, 128) or indexes.shape != (horizon,)
                or labels.max() > 7 or indexes.min() < 0 or indexes.max() > 4):
            raise ValueError(f"Invalid TRAIN trajectory: {path.name}")
        initial = labels[0].copy()
        target = torch.from_numpy((labels >= 3).astype(np.uint8))
        pore = (initial != 0) & (initial != 1)
        if np.any(connected & ~pore) or np.any(((labels != 0) & (labels != 1)) != pore[None]):
            raise ValueError("Pore support changed")
        if horizon == 25:
            if int(data["inlet_axis"]) != 0:
                raise ValueError("Matched training uses the x inlet")
            if not np.array_equal(data["actions"], ACTION_VECTORS[indexes]):
                raise ValueError("Stored four-dimensional actions differ")
            if str(data["stream_id"].item()) != record["stream_id"]:
                raise ValueError("Stored TRAIN stream differs")
    if horizon == 25:
        sequence = record["sequence_id"].split("__")[1]
    else:
        sequence = re.search(r"_(S[1-5](?:20|30)_then_S[1-5](?:20|30))\.npz$", path.name).group(1)
    first, n_first, second, n_second = re.fullmatch(r"S([1-5])(10|15|20|30)_then_S([1-5])(10|15|20|30)", sequence).groups()
    expected = np.asarray([int(first) - 1] * int(n_first) + [int(second) - 1] * int(n_second))
    if not np.array_equal(indexes, expected):
        raise ValueError("Stored actions do not match the frozen schedule")
    return initial, connected, indexes, target


def load_pair(row, horizon, device):
    first, second = [read_trajectory(path, metadata, horizon) for path, metadata in row[horizon]]
    if (not np.array_equal(first[0], second[0]) or not np.array_equal(first[1], second[1])
            or not np.array_equal(np.bincount(first[2], minlength=5), np.bincount(second[2], minlength=5))):
        raise ValueError("Counterfactual pair is not count/geometry matched")
    labels = torch.from_numpy(first[0]).to(device)
    connected = torch.from_numpy(first[1]).to(device)
    inlet = torch.zeros_like(connected)
    inlet[:, :, 0] = connected[:, :, 0]
    actions = [torch.from_numpy(ACTION_VECTORS[values[2]].copy()).to(device) for values in (first, second)]
    steps = torch.arange(1, horizon + 1, device=device, dtype=torch.float32)
    return labels, inlet, connected, *actions, (steps / (steps + 25)).unsqueeze(1), first[3], second[3]

