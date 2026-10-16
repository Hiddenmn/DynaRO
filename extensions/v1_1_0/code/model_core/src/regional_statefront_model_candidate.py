"""Untrained variable-length 3-D surrogate with a predicted-state front.

This candidate keeps the frozen StateGuard forecasting backbone and parameter
names. Only the per-step front/readout changes: an oil site must face predicted
water, and each cell/class spends an exact fraction of that exposed inventory.
No checkpoint, training, simulator target, DEV, or TEST is loaded here.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from regional_budget_model import RegionalActionSurrogate, RegionalConfig
from regional_budget_readout import full_so_from_sites, initial_oil_groups
from softfront_budget_candidate import (
    SparseFrontGeometry, front_weighted_budget_step, front_weighted_direct_step,
)
from upt3d_readout import iter_reachable_oil_support


class RegionalStateFrontSurrogate(RegionalActionSurrogate):
    """Same learnable layers as StateGuard, with a causal soft-front readout."""

    def __init__(self, config: RegionalConfig = RegionalConfig(), *,
                 readout_mode: str = "budget"):
        if readout_mode not in ("budget", "direct"):
            raise ValueError("State-front readout must be budget or direct")
        super().__init__(config, mode=readout_mode)
        self.readout_mode = readout_mode

    def forward(
        self, initial_labels: torch.Tensor, inlet: torch.Tensor,
        connected: torch.Tensor, actions: torch.Tensor, times: torch.Tensor,
        *, return_fields: bool = False, start_grad_step: int = 0,
        compact_output: bool = False, checkpoint_local: bool = False,
    ) -> dict[str, torch.Tensor | int]:
        self._validate_input(initial_labels, inlet, connected, actions, times)
        if any(type(flag) is not bool for flag in
               (return_fields, compact_output, checkpoint_local)):
            raise ValueError("Output and checkpoint flags must be boolean")
        if return_fields and compact_output:
            raise ValueError("Compact output cannot include full 3-D fields")
        if (type(start_grad_step) is not int
                or not 0 <= start_grad_step < actions.shape[0]):
            raise ValueError("start_grad_step must be within the action prefix")

        width = self.config.cell_width
        site_index, group_id, n_groups = initial_oil_groups(
            initial_labels, cell_width=width
        )
        pore = initial_labels >= 2
        oil_classes = torch.stack([initial_labels == code for code in range(3, 8)], 0)
        geometry = torch.cat((pore.unsqueeze(0), oil_classes), 0).unsqueeze(0)
        channels = torch.stack(
            [initial_labels == code for code in range(8)] + [inlet, connected], 0
        ).unsqueeze(0).to(dtype=actions.dtype)
        local = (checkpoint(self.local_encoder, channels, use_reentrant=False)
                 if checkpoint_local and torch.is_grad_enabled()
                 else self.local_encoder(channels))
        cell_static = F.avg_pool3d(local, width, stride=width)
        hidden = self.cell_initial(cell_static).tanh()
        static_site_logits = self.local_logit(local).reshape(-1)[site_index]
        cell_id = group_id // 5
        initial_inventory = actions.new_zeros(n_groups).scatter_add(
            0, group_id, actions.new_ones(site_index.numel())
        )
        previous = actions.new_ones(site_index.numel())
        previous_inventory = initial_inventory
        front_geometry = SparseFrontGeometry.from_initial(
            initial_labels, site_index, inlet, connected
        )
        states, inventories, fractions, fields = [], [], [], []
        supports = iter_reachable_oil_support(
            geometry, inlet=inlet[None, None],
            connected=connected[None, None], steps=actions.shape[0]
        )
        cell_shape = cell_static.shape[2:]

        for step, reachable in enumerate(supports):
            with (torch.no_grad() if step < start_grad_step else nullcontext()):
                cell_inventory_fraction = (
                    previous_inventory / initial_inventory.clamp_min(1)
                ).reshape(*cell_shape, 5).permute(3, 0, 1, 2).unsqueeze(0)
                action_map = actions[step].reshape(1, 4, 1, 1, 1).expand(
                    1, 4, *cell_shape
                )
                time_map = times[step].reshape(1, 1, 1, 1, 1).expand(
                    1, 1, *cell_shape
                )
                context = torch.cat((cell_static, cell_inventory_fraction,
                                     action_map, time_map), 1)
                hidden = self.cell_step(hidden, context)
                group_logits = self.class_logits(hidden).permute(
                    0, 2, 3, 4, 1
                ).reshape(-1)
                site_logits = (static_site_logits
                               + self.cell_local_bias(hidden).reshape(-1)[cell_id])
                front_weight = (
                    checkpoint(front_geometry.weight, previous, use_reentrant=False)
                    if checkpoint_local and torch.is_grad_enabled()
                    else front_geometry.weight(previous)
                )
                static_upper_bound = reachable.reshape(-1)[site_index]
                front_weight = front_weight * static_upper_bound.to(front_weight.dtype)
                update = (front_weighted_direct_step if self.readout_mode == "direct"
                          else front_weighted_budget_step)
                previous, previous_inventory = update(
                    previous, front_weight, site_logits, group_id, group_logits
                )
            if step < start_grad_step:
                continue
            states.append(previous)
            inventories.append(previous_inventory)
            if not compact_output:
                fractions.append(torch.sigmoid(group_logits))
            if return_fields:
                fields.append(full_so_from_sites(initial_labels, site_index, previous))

        remaining_sites = torch.stack(states)
        group_remaining = torch.stack(inventories)
        output = {
            "step_offset": start_grad_step,
            "site_index": site_index,
            "group_id": group_id,
            "initial_group_inventory": initial_inventory,
            "remaining_sites": remaining_sites,
            "group_remaining": group_remaining,
        }
        if compact_output:
            return output
        initial_class = initial_inventory.reshape(-1, 5).sum(0)
        class_remaining = group_remaining.reshape(len(states), -1, 5).sum(1)
        class_mobilized = initial_class.unsqueeze(0) - class_remaining
        total_initial = initial_class.sum()
        total_remaining = class_remaining.sum(1)
        pore_count = pore.sum().to(actions.dtype)
        output.update({
            "mobilization_fraction": torch.stack(fractions),
            "class_remaining": class_remaining,
            "class_mobilized": class_mobilized,
            "So_final": full_so_from_sites(
                initial_labels, site_index, remaining_sites[-1]
            ),
            "Sor": total_remaining / pore_count,
            "RF": (total_initial - total_remaining) / total_initial.clamp_min(1),
        })
        if return_fields:
            output["So_sequence"] = torch.stack(fields)
        return output


def _synthetic_self_test() -> None:
    torch.set_num_threads(2)
    config = RegionalConfig(cell_width=4, local_width=4, hidden_width=8)
    torch.manual_seed(20260929)
    baseline = RegionalActionSurrogate(config, mode="budget")
    candidate = RegionalStateFrontSurrogate(config)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )
    labels = torch.full((8, 8, 8), 3, dtype=torch.uint8)
    labels[:, :, 0] = 2
    for z in range(8):
        labels[z, :, 1:] = 3 + z % 5
    connected = torch.ones_like(labels, dtype=torch.bool)
    inlet = torch.zeros_like(connected)
    inlet[:, :, 0] = True
    actions = torch.tensor([[0.5, 0.0, 0.0, 0.0],
                            [0.7, 0.3, 0.1, 1.0],
                            [0.5, 0.0, 0.0, 0.0]])
    times = (torch.arange(1, 4, dtype=torch.float32) /
             (torch.arange(1, 4, dtype=torch.float32) + 25)).unsqueeze(1)
    output = candidate(labels, inlet, connected, actions, times,
                       return_fields=True)
    assert output["remaining_sites"].shape == (3, int((labels >= 3).sum()))
    assert output["So_sequence"].shape == (3, *labels.shape)
    assert torch.all(output["remaining_sites"][1:] <= output["remaining_sites"][:-1])
    assert torch.all(output["group_remaining"][1:] <= output["group_remaining"][:-1] + 1e-6)
    assert torch.all(output["remaining_sites"] >= 0)
    assert torch.all(output["class_remaining"] >= 0)
    loss = output["remaining_sites"][-1].mean() + output["group_remaining"][-1].mean()
    loss.backward()
    assert candidate.class_logits.weight.grad is not None
    assert torch.isfinite(candidate.class_logits.weight.grad).all()

    direct = RegionalStateFrontSurrogate(config, readout_mode="direct")
    direct.load_state_dict(baseline.state_dict(), strict=True)
    direct_output = direct(labels, inlet, connected, actions, times,
                           compact_output=True)
    assert direct_output["remaining_sites"].shape == output["remaining_sites"].shape
    assert torch.all(direct_output["remaining_sites"] >= 0)
    direct_output["remaining_sites"][-1].mean().backward()
    assert direct.class_logits.weight.grad is not None
    assert torch.isfinite(direct.class_logits.weight.grad).all()
    old_direct = RegionalActionSurrogate(config, mode="direct")
    old_direct.load_state_dict(direct.state_dict(), strict=True)
    first_old = old_direct(labels, inlet, connected, actions[:1], times[:1],
                           compact_output=True)
    first_new = direct(labels, inlet, connected, actions[:1], times[:1],
                       compact_output=True)
    assert torch.allclose(first_old["remaining_sites"],
                          first_new["remaining_sites"], atol=1e-7, rtol=1e-7)

    with torch.no_grad():
        long_actions = actions[:1].repeat(60, 1)
        steps = torch.arange(1, 61, dtype=torch.float32)
        long_times = (steps / (steps + 25)).unsqueeze(1)
        long_output = candidate(labels, inlet, connected, long_actions,
                                long_times, compact_output=True)
    assert long_output["remaining_sites"].shape[0] == 60

    # The planned 32^3 crop and five-class 16^3 cells accept a detached
    # prefix/compact gradient replay without loading any research data.
    canonical = RegionalStateFrontSurrogate()
    crop = torch.full((32, 32, 32), 3, dtype=torch.uint8)
    crop[:, :, 0] = 2
    crop_connected = torch.ones_like(crop, dtype=torch.bool)
    crop_inlet = torch.zeros_like(crop_connected)
    crop_inlet[:, :, 0] = True
    short_actions = actions[:2]
    short_times = times[:2]
    compact = canonical(crop, crop_inlet, crop_connected,
                        short_actions, short_times, start_grad_step=1,
                        compact_output=True, checkpoint_local=True)
    assert compact["remaining_sites"].shape == (1, int((crop >= 3).sum()))
    compact["group_remaining"].mean().backward()
    assert canonical.class_logits.weight.grad is not None
    assert torch.isfinite(canonical.class_logits.weight.grad).all()

    # Recomputing front activations must preserve both long-rollout states and
    # the joint paired-loss gradient, including a detached-prefix interface.
    from regional_budget_objective import (
        LossWeights, prepare_pair_targets, paired_window_loss,
        sequential_paired_window_backward,
    )
    weights = LossWeights(site=1.0, region=1.0, response=50.0)
    for horizon in (25, 50):
        steps = torch.arange(1, horizon + 1, dtype=torch.float32)
        long_times = (steps / (steps + 25)).unsqueeze(1)
        action_a = actions.repeat((horizon + 2) // 3, 1)[:horizon]
        action_b = action_a.flip(0).contiguous()
        target_a = (labels >= 3).repeat(horizon + 1, 1, 1, 1).to(torch.uint8)
        target_b = target_a.clone()
        for step in range(1, horizon + 1):
            target_a[step, :, :, :min(1 + step // 4, 8)] = 0
            target_b[step, :min(step // 6, 8), :, :min(1 + step // 5, 8)] = 0
        for start in (0, horizon // 3):
            torch.manual_seed(20260930)
            reference = RegionalStateFrontSurrogate(config, readout_mode="direct")
            efficient = RegionalStateFrontSurrogate(config, readout_mode="direct")
            efficient.load_state_dict(reference.state_dict())
            outputs = [reference(labels, inlet, connected, action, long_times,
                                 start_grad_step=start, compact_output=True)
                       for action in (action_a, action_b)]
            checked = efficient(labels, inlet, connected, action_a, long_times,
                                start_grad_step=start, compact_output=True,
                                checkpoint_local=True)
            for name in ("remaining_sites", "group_remaining"):
                torch.testing.assert_close(outputs[0][name], checked[name],
                                           rtol=0, atol=0)
            del checked
            targets = prepare_pair_targets(labels, target_a, target_b, outputs[0],
                                           start=start, stop=horizon)
            paired_window_loss(*outputs, targets, weights)["total"].backward()
            sequential_paired_window_backward(
                efficient, labels, inlet, connected, action_a, action_b,
                long_times, target_a, target_b, start=start, stop=horizon,
                weights=weights, memory_efficient=True,
            )
            for (name, parameter), (_, other) in zip(
                    reference.named_parameters(), efficient.named_parameters()):
                if parameter.grad is None:
                    assert other.grad is None, name
                else:
                    torch.testing.assert_close(parameter.grad, other.grad,
                                               atol=3e-5, rtol=2e-4)


if __name__ == "__main__":
    _synthetic_self_test()
    print("regional_statefront_model_candidate synthetic checks: PASS")
