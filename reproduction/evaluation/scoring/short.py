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
from .util import _cpu_output
from . import old50


INTERVALS = {
    25: {"all": (0, 25), "early": (0, 12), "tail": (12, 25),
         "last": (24, 25)},
    50: {"all": (0, 50), "early": (0, 25), "tail": (37, 50),
         "last": (49, 50)},
}


MEASURES = ("site", "region", "response", "zero_response",
            "unchanged_site", "unchanged_region")


def _score_outputs(output_a: dict, output_b: dict, targets,
                   horizon: int) -> dict:
    if (output_a["step_offset"] != 0 or output_b["step_offset"] != 0
            or output_a["remaining_sites"].shape[0] != horizon
            or output_b["remaining_sites"].shape != output_a["remaining_sites"].shape
            or not torch.equal(output_a["site_index"], output_b["site_index"])
            or not torch.equal(output_a["group_id"], output_b["group_id"])):
        raise ValueError("DEV paired outputs differ")
    inventory = output_a["initial_group_inventory"]
    mask = targets.nonempty_group
    sa, sb = output_a["remaining_sites"], output_b["remaining_sites"]
    ra = output_a["group_remaining"] / inventory.clamp_min(1)
    rb = output_b["group_remaining"] / inventory.clamp_min(1)
    scores = {"site_by_step": [float(v) for v in
              (0.5 * ((sa - targets.site_a).square()
                      + (sb - targets.site_b).square()).mean(1)).tolist()],
              "response_by_step": [float(v) for v in
              (((ra[:, mask] - rb[:, mask])
                - (targets.region_a[:, mask] - targets.region_b[:, mask]))
               .square().mean(1)).tolist()]}
    for name, (start, stop) in INTERVALS[horizon].items():
        a, b = sa[start:stop], sb[start:stop]
        ta, tb = targets.site_a[start:stop], targets.site_b[start:stop]
        ga, gb = ra[start:stop, mask], rb[start:stop, mask]
        ua = targets.region_a[start:stop, mask]
        ub = targets.region_b[start:stop, mask]
        values = {
            "site": 0.5 * ((a - ta).square() + (b - tb).square()).mean(),
            "region": 0.5 * ((ga - ua).square() + (gb - ub).square()).mean(),
            "response": ((ga - gb) - (ua - ub)).square().mean(),
            "zero_response": (ua - ub).square().mean(),
            "unchanged_site": 0.5 * ((1 - ta).square() + (1 - tb).square()).mean(),
            "unchanged_region": 0.5 * ((1 - ua).square() + (1 - ub).square()).mean(),
        }
        scores[name] = {key: float(value.item()) for key, value in values.items()}
    class_initial = inventory.reshape(-1, 5).sum(0)
    present = class_initial > 0
    denominator = class_initial.clamp_min(1)
    actual_a = (targets.region_a[-1] * inventory).reshape(-1, 5).sum(0)
    actual_b = (targets.region_b[-1] * inventory).reshape(-1, 5).sum(0)
    predicted_a = output_a["group_remaining"][-1].reshape(-1, 5).sum(0)
    predicted_b = output_b["group_remaining"][-1].reshape(-1, 5).sum(0)
    scores["last_class_inventory_fraction_mae"] = float((
        0.5 * (((predicted_a - actual_a).abs() / denominator)[present].mean()
               + ((predicted_b - actual_b).abs() / denominator)[present].mean())
    ).item())
    values = (scores["site_by_step"] + scores["response_by_step"]
              + [scores["last_class_inventory_fraction_mae"]]
              + [value for interval in INTERVALS[horizon]
                 for value in scores[interval].values()])
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise FloatingPointError("Invalid DEV score")
    return scores


@torch.inference_mode()
def _score_pair(models: dict, pair: tuple, horizon: int) -> dict:
    labels, inlet, connected, aa, ab, times, ta, tb = pair
    scores = {}
    for arm, model in models.items():
        output_a = _cpu_output(model(labels, inlet, connected, aa, times,
                                     compact_output=True))
        output_b = _cpu_output(model(labels, inlet, connected, ab, times,
                                     compact_output=True))
        targets = prepare_pair_targets(labels, ta, tb, output_a,
                                       start=0, stop=horizon)
        scores[arm] = _score_outputs(output_a, output_b, targets, horizon)
    return scores


def synthetic_self_test() -> None:
    """Check generic interval scoring against the established 50-step scorer."""
    for horizon in (25, 50):
        steps = torch.arange(1, horizon + 1, dtype=torch.float32)
        site_a = torch.stack((1 - steps / (horizon + 1),
                              1 - 0.5 * steps / (horizon + 1)))
        site_a = site_a.transpose(0, 1).contiguous()
        site_b = torch.stack((1 - 0.8 * steps / (horizon + 1),
                              1 - 0.6 * steps / (horizon + 1)))
        site_b = site_b.transpose(0, 1).contiguous()
        inventory = torch.tensor([1., 1., 0., 0., 0.])
        def output(site: torch.Tensor) -> dict:
            groups = torch.cat((site, torch.zeros(horizon, 3)), dim=1)
            return {"step_offset": 0, "site_index": torch.tensor([0, 1]),
                    "group_id": torch.tensor([0, 1]),
                    "initial_group_inventory": inventory,
                    "remaining_sites": site, "group_remaining": groups}
        targets = PairWindowTargets(
            site_a * 0.9, site_b * 0.85,
            output(site_a * 0.9)["group_remaining"],
            output(site_b * 0.85)["group_remaining"], inventory > 0,
        )
        ours = _score_outputs(output(site_a), output(site_b), targets, horizon)
        assert len(ours["site_by_step"]) == horizon
        assert all(math.isfinite(value) for value in ours["all"].values())
        if horizon == 50:
            previous = old50.score_outputs(output(site_a), output(site_b), targets)
            for measure in MEASURES:
                assert math.isclose(ours["all"][measure],
                                    previous["all_50"][measure],
                                    rel_tol=1e-7, abs_tol=1e-8)
            assert math.isclose(ours["last_class_inventory_fraction_mae"],
                                previous["final_class_inventory_fraction_mae"],
                                rel_tol=1e-7, abs_tol=1e-8)

