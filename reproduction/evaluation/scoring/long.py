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


INTERVALS = {
    75: {"all": (0, 75), "unseen": (50, 75), "tail": (56, 75), "last": (74, 75)},
    100: {"all": (0, 100), "unseen": (50, 100), "tail": (75, 100), "last": (99, 100)},
}


MEASURES = ("site", "region", "response", "zero_response", "unchanged_site",
            "unchanged_region", "frozen50_site", "frozen50_region")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pair(record: dict, device: torch.device) -> tuple[tuple, dict]:
    from patent22.actions import ACTION_ORDER, action_vector
    initial, connected, targets, indexes, displaced, hashes = [], [], [], [], [], []
    for path, schedule in zip(record["files"], record["schedules"]):
        with np.load(path, allow_pickle=False) as data:
            labels = data["labels"]
            mask = data["connected_pore"].copy()
            controls = data["action_index"].copy()
            counts = data["displaced_by_class"].copy()
        expected = np.asarray([ACTION_ORDER.index(s) for s in schedule], dtype=np.int8)
        if (labels.shape != (101, 128, 128, 128) or labels.dtype != np.uint8
                or int(labels.max()) > 7 or mask.shape != (128, 128, 128)
                or not np.array_equal(controls, expected) or counts.shape != (100, 5)
                or np.any(counts < 0) or np.any(mask > 1)):
            raise ValueError(f"Invalid TRAIN extension: {path}")
        occupancy = np.asarray(labels >= 3, dtype=np.uint8)
        oil_counts = np.count_nonzero(occupancy, axis=(1, 2, 3))
        if not np.array_equal(oil_counts[:-1] - oil_counts[1:], counts.sum(1)):
            raise ValueError(f"Oil count identity differs: {path}")
        initial.append(labels[0].copy())
        connected.append(mask)
        targets.append(torch.from_numpy(occupancy))
        indexes.append(controls)
        displaced.append(counts)
        hashes.append(file_hash(path))
        del labels
    if (not np.array_equal(initial[0], initial[1])
            or not np.array_equal(connected[0], connected[1])):
        raise ValueError("Two schedules must share the initial TRAIN state")
    labels = torch.from_numpy(initial[0]).to(device)
    connected_gpu = torch.from_numpy(connected[0].astype(bool)).to(device)
    inlet = torch.zeros_like(connected_gpu)
    inlet[:, :, 0] = connected_gpu[:, :, 0]
    controls_gpu = [torch.from_numpy(np.stack([
        action_vector(ACTION_ORDER[int(i)]) for i in control
    ])).to(device) for control in indexes]
    steps = torch.arange(1, 101, dtype=torch.float32, device=device)
    times = (steps / (steps + 25)).unsqueeze(1)
    signal = {name: [int(counts[start:stop].sum()) for counts in displaced]
              for name, (start, stop) in {"51_75": (50, 75), "51_100": (50, 100),
                                        "76_100": (75, 100)}.items()}
    return (labels, inlet, connected_gpu, *controls_gpu, times, *targets), {
        "input_files": [str(p) for p in record["files"]],
        "input_sha256": hashes, "target_mobilized_counts": signal,
        "initial_oil_sites": int(np.count_nonzero(initial[0] >= 3)),
    }


def score_outputs(a: dict, b: dict, targets: PairWindowTargets) -> dict:
    if (a["step_offset"] != 0 or b["step_offset"] != 0
            or a["remaining_sites"].shape[0] != 100
            or not torch.equal(a["site_index"], b["site_index"])
            or not torch.equal(a["group_id"], b["group_id"])
            or not torch.equal(a["initial_group_inventory"], b["initial_group_inventory"])):
        raise ValueError("Paired prediction supports differ")
    inventory = a["initial_group_inventory"]
    mask = targets.nonempty_group
    sa, sb = a["remaining_sites"], b["remaining_sites"]
    ra, rb = (o["group_remaining"] / inventory.clamp_min(1) for o in (a, b))
    ta, tb = targets.site_a, targets.site_b
    ua, ub = targets.region_a[:, mask], targets.region_b[:, mask]
    ga, gb = ra[:, mask], rb[:, mask]
    fa, fb, fga, fgb = sa.clone(), sb.clone(), ga.clone(), gb.clone()
    fa[50:], fb[50:] = sa[49], sb[49]
    fga[50:], fgb[50:] = ga[49], gb[49]
    curves = {
        "site": 0.5 * ((sa - ta).square() + (sb - tb).square()).mean(1),
        "region": 0.5 * ((ga - ua).square() + (gb - ub).square()).mean(1),
        "response": ((ga - gb) - (ua - ub)).square().mean(1),
        "zero_response": (ua - ub).square().mean(1),
        "unchanged_site": 0.5 * ((1 - ta).square() + (1 - tb).square()).mean(1),
        "unchanged_region": 0.5 * ((1 - ua).square() + (1 - ub).square()).mean(1),
        "frozen50_site": 0.5 * ((fa - ta).square() + (fb - tb).square()).mean(1),
        "frozen50_region": 0.5 * ((fga - ua).square() + (fgb - ub).square()).mean(1),
    }
    curves = {key: value.tolist() for key, value in curves.items()}
    if any(not math.isfinite(v) or v < 0 for curve in curves.values() for v in curve):
        raise FloatingPointError("Non-finite or negative score")
    class_initial = inventory.reshape(-1, 5).sum(0)
    present = class_initial > 0
    scores = {}
    for horizon, intervals in INTERVALS.items():
        values = {name: {key: statistics.fmean(curves[key][start:stop])
                         for key in MEASURES} for name, (start, stop) in intervals.items()}
        actual = [(r[horizon - 1] * inventory).reshape(-1, 5).sum(0)
                  for r in (targets.region_a, targets.region_b)]
        predicted = [o["group_remaining"][horizon - 1].reshape(-1, 5).sum(0)
                     for o in (a, b)]
        errors = 0.5 * sum((p - t).abs() / class_initial.clamp_min(1)
                           for p, t in zip(predicted, actual))
        values["last_class_inventory_fraction_mae"] = float(errors[present].mean())
        values["last_class_inventory_fraction_mae_by_class"] = [
            float(v) if bool(keep) else None for v, keep in zip(errors, present)]
        values.update({f"{key}_by_step": curves[key][:horizon] for key in MEASURES})
        scores[str(horizon)] = values
    return scores


@torch.inference_mode()
def score_pair(models: dict, pair: tuple) -> dict:
    labels, inlet, connected, aa, ab, times, ta, tb = pair
    scores = {"75": {}, "100": {}}
    for arm, model in models.items():
        cpu = lambda out: {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                           for k, v in out.items()}
        a = cpu(model(labels, inlet, connected, aa, times, compact_output=True))
        b = cpu(model(labels, inlet, connected, ab, times, compact_output=True))
        targets = prepare_pair_targets(labels, ta, tb, a, start=0, stop=100)
        arm_scores = score_outputs(a, b, targets)
        for horizon in scores:
            scores[horizon][arm] = arm_scores[horizon]
    return scores

