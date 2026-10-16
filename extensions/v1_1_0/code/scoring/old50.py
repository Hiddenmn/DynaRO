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


INTERVALS = {"all_50": (0, 50), "steps_1_25": (0, 25),
             "steps_26_37": (25, 37), "steps_38_50": (37, 50),
             "step_50": (49, 50)}


def score_outputs(output_a: dict, output_b: dict, targets) -> dict:
    inventory = output_a["initial_group_inventory"]
    mask = targets.nonempty_group
    if (output_a["step_offset"] != 0 or output_b["step_offset"] != 0
            or output_a["remaining_sites"].shape[0] != 50
            or output_b["remaining_sites"].shape != output_a["remaining_sites"].shape
            or not torch.equal(output_a["site_index"], output_b["site_index"])
            or not torch.equal(output_a["group_id"], output_b["group_id"])
            or not torch.equal(inventory, output_b["initial_group_inventory"])
            or not torch.equal(mask, inventory > 0)):
        raise ValueError("Paired 50-step outputs do not align")
    site_a, site_b = output_a["remaining_sites"], output_b["remaining_sites"]
    region_a = output_a["group_remaining"] / inventory.clamp_min(1)
    region_b = output_b["group_remaining"] / inventory.clamp_min(1)
    scores = {}
    scores["site_by_step"] = [float(value) for value in
                              (0.5 * ((site_a - targets.site_a).square()
                                      + (site_b - targets.site_b).square())
                               .mean(dim=1)).tolist()]
    scores["response_by_step"] = [float(value) for value in
                                  (((region_a[:, mask] - region_b[:, mask])
                                    - (targets.region_a[:, mask]
                                       - targets.region_b[:, mask]))
                                   .square().mean(dim=1)).tolist()]
    for name, (start, stop) in INTERVALS.items():
        sa, sb = site_a[start:stop], site_b[start:stop]
        ta, tb = targets.site_a[start:stop], targets.site_b[start:stop]
        ra, rb = region_a[start:stop, mask], region_b[start:stop, mask]
        ua, ub = targets.region_a[start:stop, mask], targets.region_b[start:stop, mask]
        values = {
            "site": 0.5 * ((sa - ta).square() + (sb - tb).square()).mean(),
            "region": 0.5 * ((ra - ua).square() + (rb - ub).square()).mean(),
            "response": ((ra - rb) - (ua - ub)).square().mean(),
            "zero_response": (ua - ub).square().mean(),
            "unchanged_site": 0.5 * ((1 - ta).square() + (1 - tb).square()).mean(),
            "unchanged_region": 0.5 * ((1 - ua).square() + (1 - ub).square()).mean(),
        }
        scores[name] = {key: float(value.item()) for key, value in values.items()}
    initial_class = inventory.reshape(-1, 5).sum(0)
    present = initial_class > 0
    actual_a = (targets.region_a[-1] * inventory).reshape(-1, 5).sum(0)
    actual_b = (targets.region_b[-1] * inventory).reshape(-1, 5).sum(0)
    pred_a = output_a["group_remaining"][-1].reshape(-1, 5).sum(0)
    pred_b = output_b["group_remaining"][-1].reshape(-1, 5).sum(0)
    denom = initial_class.clamp_min(1)
    class_mae = 0.5 * (
        ((pred_a - actual_a).abs() / denom)[present].mean()
        + ((pred_b - actual_b).abs() / denom)[present].mean()
    )
    scores["final_class_inventory_fraction_mae"] = float(class_mae.item())
    checked = ([value for interval in INTERVALS
                for value in scores[interval].values()]
               + scores["site_by_step"] + scores["response_by_step"]
               + [scores["final_class_inventory_fraction_mae"]])
    if any(not math.isfinite(value) or value < 0 for value in checked):
        raise FloatingPointError("Non-finite or negative DEV50 score")
    return scores

