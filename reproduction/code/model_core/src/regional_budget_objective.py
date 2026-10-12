"""Matched paired supervision and sequential TBPTT replay; synthetic checks only.

All three regional readout modes use this same objective.  Future oil occupancy
is a target only: it never enters the model's forward rollout.  Sequential
replay gives the same gradient as a joint two-arm graph for the *same* detached
prefix/window and unchanged parameters.  It is not a full-horizon memory or
runtime measurement, training entry point, or model-performance result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from regional_budget_model import RegionalActionSurrogate, RegionalConfig


@dataclass(frozen=True)
class LossWeights:
    site: float
    region: float
    response: float

    def __post_init__(self) -> None:
        values = (self.site, self.region, self.response)
        if (any(not math.isfinite(value) or value < 0 for value in values)
                or not any(value > 0 for value in values)):
            raise ValueError("Loss weights must be finite, nonnegative, and nonzero")


@dataclass(frozen=True)
class PairWindowTargets:
    site_a: torch.Tensor
    site_b: torch.Tensor
    region_a: torch.Tensor
    region_b: torch.Tensor
    nonempty_group: torch.Tensor


def _check_common_outputs(output_a: dict, output_b: dict) -> None:
    for name in ("site_index", "group_id", "initial_group_inventory"):
        if not torch.equal(output_a[name], output_b[name]):
            raise ValueError("Paired arms must use the same initial oil groups")
    for name in ("remaining_sites", "group_remaining"):
        if output_a[name].shape != output_b[name].shape:
            raise ValueError("Paired output shapes must match")
    if output_a["step_offset"] != output_b["step_offset"]:
        raise ValueError("Paired outputs must cover the same window")


def _site_targets(
    initial_labels: torch.Tensor, target: torch.Tensor, site_index: torch.Tensor,
    *, start: int, stop: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    if (target.ndim != 4 or target.shape[1:] != initial_labels.shape
            or target.shape[0] < stop + 1):
        raise ValueError("Occupancy targets must be [T+1,Z,Y,X]")
    if target.dtype not in (torch.bool, torch.uint8):
        raise ValueError("Occupancy targets must be binary bool or uint8")
    initial_oil = (initial_labels >= 3).to(target.device)
    if not torch.equal(target[0].bool(), initial_oil):
        raise ValueError("Target initial oil occupancy differs from the model input")
    selected = target[start + 1:stop + 1]
    if bool(torch.any((selected != 0) & (selected != 1))):
        raise ValueError("Occupancy targets must be binary")
    if bool(torch.any(selected.bool() & ~initial_oil)):
        raise ValueError("Targets create oil outside initial oil support")
    indices = site_index.to(target.device)
    return selected.reshape(stop - start, -1)[:, indices].to(
        device=device, dtype=dtype
    )


def prepare_pair_targets(
    initial_labels: torch.Tensor, target_a: torch.Tensor, target_b: torch.Tensor,
    output: dict, *, start: int, stop: int,
) -> PairWindowTargets:
    """Project binary future So labels to fixed site and cell/class targets.

    Target tensors can stay on CPU; only initial-oil site values enter device
    memory.  All grouping comes from initial labels, not from future targets.
    """
    if (type(start) is not int or type(stop) is not int or not 0 <= start < stop
            or target_a.shape != target_b.shape
            or output["step_offset"] != start
            or output["remaining_sites"].shape[0] != stop - start):
        raise ValueError("Invalid paired target window")
    inventory = output["initial_group_inventory"]
    if not bool(torch.any(inventory > 0)):
        raise ValueError("No initial oil for paired supervision")
    site_index, group_id = output["site_index"], output["group_id"]
    if (output["remaining_sites"].shape[1] != site_index.numel()
            or output["group_remaining"].shape[1] != inventory.numel()):
        raise ValueError("Inconsistent model site or group dimensions")
    device, dtype = inventory.device, inventory.dtype
    sites = [
        _site_targets(initial_labels, target, site_index, start=start, stop=stop,
                      device=device, dtype=dtype)
        for target in (target_a, target_b)
    ]
    index = group_id.unsqueeze(0).expand(stop - start, -1)
    groups = [
        site.new_zeros(stop - start, inventory.numel()).scatter_add(1, index, site)
        / inventory.clamp_min(1)
        for site in sites
    ]
    return PairWindowTargets(sites[0], sites[1], groups[0], groups[1],
                             inventory > 0)


def paired_window_loss(
    output_a: dict, output_b: dict, targets: PairWindowTargets,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    """Equal-weight state and signed action-response Brier on one window."""
    _check_common_outputs(output_a, output_b)
    site_a, site_b = (output["remaining_sites"] for output in (output_a, output_b))
    inventory = output_a["initial_group_inventory"].clamp_min(1)
    region_a, region_b = (
        output["group_remaining"] / inventory
        for output in (output_a, output_b)
    )
    mask = targets.nonempty_group
    if (mask.shape != inventory.shape or mask.dtype != torch.bool
            or mask.device != inventory.device
            or not torch.equal(mask, output_a["initial_group_inventory"] > 0)
            or not bool(torch.any(mask))
            or site_a.shape != targets.site_a.shape
            or site_b.shape != targets.site_b.shape
            or region_a.shape != targets.region_a.shape
            or region_b.shape != targets.region_b.shape):
        raise ValueError("Loss targets and model outputs must match")
    site = ((site_a - targets.site_a).square()
            + (site_b - targets.site_b).square()).mean() * 0.5
    region = ((region_a[:, mask] - targets.region_a[:, mask]).square()
              + (region_b[:, mask] - targets.region_b[:, mask]).square()).mean() * 0.5
    residual = ((region_a[:, mask] - region_b[:, mask])
                - (targets.region_a[:, mask] - targets.region_b[:, mask]))
    response = residual.square().mean()
    return {
        "site": site,
        "region": region,
        "response": response,
        "total": weights.site * site + weights.region * region
                 + weights.response * response,
    }


def score_full_horizon_pair(
    output_a: dict, output_b: dict, targets: PairWindowTargets,
) -> dict[str, dict[str, float]]:
    """Score every prespecified 25-step interval; no training or selection.

    The zero-response reference predicts no difference between the two action
    paths.  The unchanged-oil references predict initial oil occupancy at
    every site/group.  All scores use the same initial-oil support and exclude
    empty cell/class groups; no future label is passed to the model.
    """
    _check_common_outputs(output_a, output_b)
    if output_a["step_offset"] != 0:
        raise ValueError("Full-horizon scoring must begin at transition 1")
    site_a, site_b = (output["remaining_sites"] for output in (output_a, output_b))
    inventory = output_a["initial_group_inventory"]
    mask = targets.nonempty_group
    if (site_a.shape[0] != 25 or site_a.shape != site_b.shape
            or site_a.shape != targets.site_a.shape
            or site_b.shape != targets.site_b.shape
            or output_a["group_remaining"].shape != (25, inventory.numel())
            or output_b["group_remaining"].shape != (25, inventory.numel())
            or targets.region_a.shape != (25, inventory.numel())
            or targets.region_b.shape != (25, inventory.numel())
            or mask.shape != inventory.shape or mask.dtype != torch.bool
            or mask.device != inventory.device
            or not bool(torch.any(mask))
            or not torch.equal(mask, inventory > 0)):
        raise ValueError("Expected aligned full-25-step paired outputs and targets")
    intervals = {"all_25": (0, 25), "steps_1_5": (0, 5),
                 "steps_6_15": (5, 15), "steps_16_25": (15, 25)}
    with torch.no_grad():
        region_a = output_a["group_remaining"] / inventory.clamp_min(1)
        region_b = output_b["group_remaining"] / inventory.clamp_min(1)
        scores = {}
        for name, (start, stop) in intervals.items():
            sa, sb = site_a[start:stop], site_b[start:stop]
            ta, tb = targets.site_a[start:stop], targets.site_b[start:stop]
            ra, rb = region_a[start:stop, mask], region_b[start:stop, mask]
            ua = targets.region_a[start:stop, mask]
            ub = targets.region_b[start:stop, mask]
            measures = {
                "site": 0.5 * ((sa - ta).square() + (sb - tb).square()).mean(),
                "region": 0.5 * ((ra - ua).square() + (rb - ub).square()).mean(),
                "response": ((ra - rb) - (ua - ub)).square().mean(),
                "zero_response": (ua - ub).square().mean(),
                "unchanged_site": 0.5 * ((1 - ta).square()
                                         + (1 - tb).square()).mean(),
                "unchanged_region": 0.5 * ((1 - ua).square()
                                           + (1 - ub).square()).mean(),
            }
            scores[name] = {key: float(value.item())
                            for key, value in measures.items()}
    return scores


def passes_prespecified_train_screen(
    scores_by_mode: dict[str, dict[str, dict[str, float]]],
    *, min_relative_gain: float = 0.01,
) -> bool:
    """In-sample screen only; a True result never authorizes DEV or a claim."""
    if (set(scores_by_mode) != set(RegionalActionSurrogate.MODES)
            or not math.isfinite(min_relative_gain)
            or not 0 < min_relative_gain < 1):
        raise ValueError("Expected exactly three modes and a valid frozen margin")
    required_intervals = ("all_25", "steps_1_5", "steps_6_15", "steps_16_25")
    required_measures = ("site", "region", "response", "zero_response",
                         "unchanged_site", "unchanged_region")
    for scores in scores_by_mode.values():
        if set(scores) != set(required_intervals):
            raise ValueError("Every mode must report all prespecified intervals")
        for interval in scores.values():
            if (set(interval) != set(required_measures)
                    or any(not math.isfinite(value) or value < 0
                           for value in interval.values())):
                raise ValueError("Missing, non-finite or negative screen metric")
    baseline = scores_by_mode["budget"]["all_25"]["zero_response"]
    if any(not math.isclose(scores["all_25"]["zero_response"], baseline,
                            rel_tol=1e-6, abs_tol=1e-12)
           for scores in scores_by_mode.values()):
        raise ValueError("Three modes must share exactly the same pair targets")
    budget = scores_by_mode["budget"]["all_25"]["response"]
    references = (baseline,
                  scores_by_mode["uncentered"]["all_25"]["response"],
                  scores_by_mode["direct"]["all_25"]["response"])
    return all(reference > 0 and budget <= (1 - min_relative_gain) * reference
               for reference in references)


def sequential_paired_window_backward(
    model: RegionalActionSurrogate,
    initial_labels: torch.Tensor, inlet: torch.Tensor, connected: torch.Tensor,
    actions_a: torch.Tensor, actions_b: torch.Tensor, times: torch.Tensor,
    target_a: torch.Tensor, target_b: torch.Tensor,
    *, start: int, stop: int, weights: LossWeights, gradient_scale: float = 1.0,
    memory_efficient: bool = False,
) -> dict[str, float]:
    """Accumulate one window's exact paired gradient without two live graphs.

    No optimizer operation occurs here.  The caller must keep parameters fixed
    through preview and both replays.  A suffix window truncates recurrent
    gradients at ``start``; no full-25-step gradient equivalence is implied.
    ``gradient_scale`` lets a future caller weight windows by their length.
    The optional compact/checkpoint path is shared by every readout mode.
    """
    if (type(start) is not int or type(stop) is not int
            or actions_a.ndim != 2 or actions_a.shape[1] != 4
            or actions_a.shape != actions_b.shape
            or times.shape != (actions_a.shape[0], 1)
            or target_a.shape[0] != actions_a.shape[0] + 1
            or target_b.shape[0] != actions_a.shape[0] + 1
            or not 0 <= start < stop <= actions_a.shape[0]
            or not math.isfinite(gradient_scale) or gradient_scale <= 0
            or type(memory_efficient) is not bool):
        raise ValueError("Invalid paired window or gradient scale")
    model_flags = {"compact_output": memory_efficient,
                   "checkpoint_local": memory_efficient}

    with torch.no_grad():
        preview_a = model(initial_labels, inlet, connected, actions_a[:stop],
                          times[:stop], start_grad_step=start, **model_flags)
        preview_b = model(initial_labels, inlet, connected, actions_b[:stop],
                          times[:stop], start_grad_step=start, **model_flags)
        _check_common_outputs(preview_a, preview_b)
        targets = prepare_pair_targets(initial_labels, target_a, target_b,
                                       preview_a, start=start, stop=stop)
        measures = paired_window_loss(preview_a, preview_b, targets, weights)
        divisor = preview_a["initial_group_inventory"].clamp_min(1)
        residual = (
            (preview_a["group_remaining"] - preview_b["group_remaining"]) / divisor
            - (targets.region_a - targets.region_b)
        )[:, targets.nonempty_group].detach()
        result = {name: float(value.item()) for name, value in measures.items()}
    del preview_a, preview_b, measures, divisor

    def backward_arm(actions: torch.Tensor, site_target: torch.Tensor,
                     region_target: torch.Tensor, sign: int) -> None:
        output = model(initial_labels, inlet, connected, actions[:stop],
                       times[:stop], start_grad_step=start, **model_flags)
        site = output["remaining_sites"]
        region = (output["group_remaining"]
                  / output["initial_group_inventory"].clamp_min(1))
        active = region[:, targets.nonempty_group]
        surrogate = (
            weights.site * 0.5 * (site - site_target).square().mean()
            + weights.region * 0.5
            * (active - region_target[:, targets.nonempty_group]).square().mean()
            + weights.response * (2 * sign * residual * active).mean()
        )
        (gradient_scale * surrogate).backward()

    backward_arm(actions_a, targets.site_a, targets.region_a, 1)
    backward_arm(actions_b, targets.site_b, targets.region_b, -1)
    return result


def _synthetic_self_test() -> None:
    torch.set_num_threads(2)
    labels = torch.empty((8, 8, 8), dtype=torch.uint8)
    for z in range(8):
        labels[z] = 3 + z % 5
    labels[:, :, 0] = 2
    connected = torch.ones_like(labels, dtype=torch.bool)
    inlet = torch.zeros_like(connected)
    inlet[:, :, 0] = True
    actions_a = torch.tensor([[0.2, 0.5, 0.0, 0.0],
                              [0.7, 0.1, 0.4, 1.0],
                              [0.5, 0.6, 0.2, 0.0]])
    actions_b = torch.tensor([[0.8, 0.1, 0.3, 1.0],
                              [0.3, 0.8, 0.0, 0.0],
                              [0.5, 0.6, 0.2, 0.0]])
    times = torch.tensor([[1 / 25], [2 / 25], [3 / 25]])
    initial = (labels >= 3).to(torch.uint8)
    target_a = initial.repeat(4, 1, 1, 1)
    target_b = target_a.clone()
    for step in range(1, 4):
        target_a[step:, :, :, 1:step + 1] = 0
        target_b[step:, :step, :, 1:step + 1] = 0
    config = RegionalConfig(cell_width=4, local_width=4, hidden_width=8)
    weights = LossWeights(site=1.0, region=0.7, response=1.3)
    for mode in RegionalActionSurrogate.MODES:
        for start, stop in ((0, 3), (1, 3)):
            torch.manual_seed(20260920)
            joint_model = RegionalActionSurrogate(config, mode=mode)
            replay_model = RegionalActionSurrogate(config, mode=mode)
            replay_model.load_state_dict(joint_model.state_dict())
            efficient_model = RegionalActionSurrogate(config, mode=mode)
            efficient_model.load_state_dict(joint_model.state_dict())
            outputs = [
                joint_model(labels, inlet, connected, actions[:stop],
                            times[:stop], start_grad_step=start)
                for actions in (actions_a, actions_b)
            ]
            targets = prepare_pair_targets(labels, target_a, target_b,
                                           outputs[0], start=start, stop=stop)
            assert bool(torch.any(targets.region_a != targets.region_b))
            terms = paired_window_loss(*outputs, targets, weights)
            terms["total"].backward()
            measured = sequential_paired_window_backward(
                replay_model, labels, inlet, connected, actions_a, actions_b,
                times, target_a, target_b, start=start, stop=stop,
                weights=weights,
            )
            efficient = sequential_paired_window_backward(
                efficient_model, labels, inlet, connected, actions_a, actions_b,
                times, target_a, target_b, start=start, stop=stop,
                weights=weights, memory_efficient=True,
            )
            for name, value in measured.items():
                assert math.isclose(value, float(terms[name].item()),
                                    rel_tol=1e-6, abs_tol=1e-7)
                assert math.isclose(efficient[name], value,
                                    rel_tol=1e-6, abs_tol=1e-7)
            for name, parameter in joint_model.named_parameters():
                other = dict(replay_model.named_parameters())[name]
                efficient_parameter = dict(efficient_model.named_parameters())[name]
                if parameter.grad is None or other.grad is None:
                    assert (parameter.grad is None and other.grad is None
                            and efficient_parameter.grad is None), name
                    continue
                assert torch.isfinite(parameter.grad).all()
                assert torch.isfinite(other.grad).all()
                assert torch.isfinite(efficient_parameter.grad).all()
                assert torch.allclose(parameter.grad, other.grad,
                                      atol=3e-5, rtol=2e-4), name
                assert torch.allclose(parameter.grad, efficient_parameter.grad,
                                      atol=3e-5, rtol=2e-4), name
    torch.manual_seed(20260920)
    long_model = RegionalActionSurrogate(config, mode="budget")
    long_a = actions_a[:1].expand(25, -1).clone()
    long_b = actions_b[:1].expand(25, -1).clone()
    long_times = torch.arange(1, 26, dtype=actions_a.dtype).unsqueeze(1) / 25
    long_target = initial.repeat(26, 1, 1, 1)
    long_measures = sequential_paired_window_backward(
        long_model, labels, inlet, connected, long_a, long_b, long_times,
        long_target, long_target, start=23, stop=25, weights=weights,
    )
    assert all(math.isfinite(value) for value in long_measures.values())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in long_model.parameters())
    long_target_b = long_target.clone()
    long_target_b[1:, 0, :, 1] = 0
    with torch.no_grad():
        full_a = long_model(labels, inlet, connected, long_a, long_times)
        full_b = long_model(labels, inlet, connected, long_b, long_times)
        full_targets = prepare_pair_targets(
            labels, long_target, long_target_b, full_a, start=0, stop=25
        )
        full_scores = score_full_horizon_pair(full_a, full_b, full_targets)
        direct_full = paired_window_loss(full_a, full_b, full_targets, weights)
    assert full_scores["all_25"]["zero_response"] > 0
    for name in ("site", "region", "response"):
        assert math.isclose(full_scores["all_25"][name],
                            float(direct_full[name].item()),
                            rel_tol=1e-6, abs_tol=1e-8)
    for name in full_scores["all_25"]:
        interval_mean = (5 * full_scores["steps_1_5"][name]
                         + 10 * full_scores["steps_6_15"][name]
                         + 10 * full_scores["steps_16_25"][name]) / 25
        assert math.isclose(full_scores["all_25"][name], interval_mean,
                            rel_tol=1e-6, abs_tol=1e-8)
    fictitious = {
        mode: {interval: dict(values) for interval, values in full_scores.items()}
        for mode in RegionalActionSurrogate.MODES
    }
    for scores in fictitious.values():
        scores["all_25"]["zero_response"] = 1.0
    fictitious["budget"]["all_25"]["response"] = 0.8
    fictitious["uncentered"]["all_25"]["response"] = 0.9
    fictitious["direct"]["all_25"]["response"] = 1.1
    assert passes_prespecified_train_screen(fictitious)
    fictitious["budget"]["all_25"]["response"] = 0.895
    assert not passes_prespecified_train_screen(fictitious)
    stagnant_connected = torch.zeros_like(connected)
    stagnant_inlet = torch.zeros_like(inlet)
    stagnant_target = initial.repeat(4, 1, 1, 1)
    for mode in RegionalActionSurrogate.MODES:
        stagnant_model = RegionalActionSurrogate(config, mode=mode)
        stagnant_measures = sequential_paired_window_backward(
            stagnant_model, labels, stagnant_inlet, stagnant_connected,
            actions_a, actions_b, times, stagnant_target, stagnant_target,
            start=0, stop=3, weights=weights, memory_efficient=True,
        )
        assert all(value == 0 for value in stagnant_measures.values())
        assert all(parameter.grad is None
                   or torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
                   for parameter in stagnant_model.parameters())
    bad_target = target_a.clone()
    bad_target[1, :, :, 0] = 1
    with torch.no_grad():
        sample = joint_model(labels, inlet, connected, actions_a, times)
    try:
        prepare_pair_targets(labels, bad_target, target_b, sample,
                             start=0, stop=3)
    except ValueError:
        pass
    else:
        raise AssertionError("Future oil outside initial support was accepted")


if __name__ == "__main__":
    _synthetic_self_test()
    print("regional_budget_objective synthetic self-test: PASS")
