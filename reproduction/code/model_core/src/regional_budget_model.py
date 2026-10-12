"""Untrained action-conditioned 3-D remaining-oil candidate and matched arms.

The shared network predicts one logit per 16^3 cell/oil class and one local
propensity logit per initial oil site.  Only the final algebraic readout mode
changes across arms.  This module has no data loader, optimizer, checkpoint,
evaluation entry point, or claim of simulator/physical fidelity.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from regional_budget_readout import (
    full_so_from_sites,
    initial_oil_groups,
    readout_from_shared_logits,
)
from upt3d_readout import iter_reachable_oil_support


def initial_labels_from_canonical_channels(
    core: torch.Tensor, pore: torch.Tensor, initial_oil: torch.Tensor,
) -> torch.Tensor:
    """Recover exact 0..7 initial labels from existing canonical item channels.

    ``core`` and ``pore`` have shape [1,Z,Y,X], and ``initial_oil`` has
    shape [5,Z,Y,X].  Both core and pore are required: the old compact
    six-channel geometry alone cannot distinguish exterior code 0 from
    solid-core code 1.  This consumes no future target or learned feature.
    """
    if (core.ndim != 4 or core.shape[0] != 1
            or pore.shape != core.shape
            or initial_oil.shape != (5, *core.shape[1:])
            or min(core.shape[1:]) < 1):
        raise ValueError("Expected canonical core/pore [1,Z,Y,X] and oil [5,Z,Y,X]")
    channels = (core, pore, initial_oil)
    if (any(value.device != core.device for value in channels)
            or any(value.dtype not in (torch.bool, torch.uint8) for value in channels)
            or any(bool(torch.any((value != 0) & (value != 1))) for value in channels)):
        raise ValueError("Canonical initial channels must be same-device binary masks")
    core_mask, pore_mask, oil_masks = core[0].bool(), pore[0].bool(), initial_oil.bool()
    if (bool(torch.any(pore_mask & ~core_mask))
            or bool(torch.any(oil_masks & ~pore_mask.unsqueeze(0)))
            or bool(torch.any(oil_masks.sum(0) > 1))):
        raise ValueError("Canonical initial classes violate core/pore/oil nesting")
    labels = torch.zeros_like(core[0], dtype=torch.uint8)
    labels[core_mask] = 1
    labels[pore_mask] = 2
    for offset in range(5):
        labels[oil_masks[offset]] = offset + 3
    return labels


@dataclass(frozen=True)
class RegionalConfig:
    cell_width: int = 16
    local_width: int = 8
    hidden_width: int = 24

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (
            self.cell_width, self.local_width, self.hidden_width
        )):
            raise ValueError("Model widths must be positive integers")


class CellGRU(nn.Module):
    """Conventional 3-D convolutional GRU; not the algorithmic contribution."""

    def __init__(self, hidden_width: int, context_width: int):
        super().__init__()
        channels = hidden_width + context_width
        self.gates = nn.Conv3d(channels, 2 * hidden_width, 3, padding=1)
        self.candidate = nn.Conv3d(channels, hidden_width, 3, padding=1)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((hidden, context), 1)).sigmoid().chunk(2, 1)
        proposed = self.candidate(torch.cat((reset * hidden, context), 1)).tanh()
        return (1 - update) * hidden + update * proposed


class RegionalActionSurrogate(nn.Module):
    """One-sample causal rollout with a selectable, parameter-free readout.

    Inputs: initial class volume [Z,Y,X] with oil codes 3..7; inlet and
    connected-pore boolean masks of the same shape; current/previous controls
    [T,4]; increasing normalized transition times [T,1].  No future true
    state, stream identifier, or opposite-arm target is accepted.

    ``start_grad_step`` replays earlier controls without gradients, stores only
    the suffix and truncates recurrent gradients at its start.  It recomputes
    the initial static encoder per call.  This is a code interface, not a
    measured full-128^3 training memory/time guarantee.

    Compact output omits fields unused by paired training. Local activation
    checkpointing recomputes encoder intermediates during backward. These
    engineering options apply equally to all three readout modes.
    """

    MODES = ("budget", "uncentered", "direct")

    def __init__(self, config: RegionalConfig = RegionalConfig(), *, mode: str = "budget"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unknown readout mode")
        self.config = config
        self.mode = mode
        local, hidden = config.local_width, config.hidden_width
        self.local_encoder = nn.Sequential(
            nn.Conv3d(10, local, 3, padding=1), nn.GELU(),
            nn.Conv3d(local, local, 3, padding=1), nn.GELU(),
        )
        self.local_logit = nn.Conv3d(local, 1, 1)
        self.cell_initial = nn.Conv3d(local, hidden, 3, padding=1)
        context_width = local + 5 + 4 + 1
        self.cell_step = CellGRU(hidden, context_width)
        self.class_logits = nn.Conv3d(hidden, 5, 1)
        self.cell_local_bias = nn.Conv3d(hidden, 1, 1)

    def _validate_input(
        self, initial_labels: torch.Tensor, inlet: torch.Tensor,
        connected: torch.Tensor, actions: torch.Tensor, times: torch.Tensor,
    ) -> None:
        if initial_labels.ndim != 3 or min(initial_labels.shape) < 1:
            raise ValueError("Expected one nonempty initial 3-D class volume")
        if inlet.shape != initial_labels.shape or connected.shape != initial_labels.shape:
            raise ValueError("Inlet and connected masks must match the 3-D volume")
        if inlet.dtype != torch.bool or connected.dtype != torch.bool:
            raise ValueError("Inlet and connected masks must be boolean")
        if (initial_labels.device != inlet.device
                or initial_labels.device != connected.device
                or initial_labels.device != actions.device
                or initial_labels.device != times.device):
            raise ValueError("All inputs must share a device")
        if actions.ndim != 2 or actions.shape[1] != 4 or actions.shape[0] < 1:
            raise ValueError("Expected a nonempty action sequence [T,4]")
        if times.shape != (actions.shape[0], 1):
            raise ValueError("Expected normalized transition times [T,1]")
        parameter = next(self.parameters())
        if actions.dtype != parameter.dtype or times.dtype != parameter.dtype:
            raise ValueError("Controls and model parameters must share a dtype")
        if actions.device != parameter.device:
            raise ValueError("Model and inputs must share a device")
        if not bool(torch.isfinite(actions).all() and torch.isfinite(times).all()):
            raise ValueError("Non-finite controls")
        if bool(torch.any((times < 0) | (times > 1))
                or torch.any(times[1:] <= times[:-1])):
            raise ValueError("Transition times must increase within [0,1]")
        pore = initial_labels >= 2
        if bool(torch.any(connected & ~pore) or torch.any(inlet & ~connected)):
            raise ValueError("Require inlet subset of connected subset of pore")
        if not bool(torch.any(pore)):
            raise ValueError("Initial volume has no pores")

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
                eligible = reachable.reshape(-1)[site_index]
                previous, previous_inventory = readout_from_shared_logits(
                    previous, site_logits, group_id, group_logits,
                    mode=self.mode, eligible=eligible
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
    source = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8).reshape(2, 2, 2)
    source_core = (source != 0).unsqueeze(0).to(torch.uint8)
    source_pore = (source >= 2).unsqueeze(0).to(torch.uint8)
    source_oil = torch.stack([source == code for code in range(3, 8)], 0).to(torch.uint8)
    assert torch.equal(initial_labels_from_canonical_channels(
        source_core, source_pore, source_oil
    ), source)
    labels = torch.full((8, 8, 8), 3, dtype=torch.uint8)
    labels[:, :, 0] = 2  # Initial water plane, never an oil output site.
    for z in range(8):
        labels[z, :, 1:] = 3 + z % 5
    connected = torch.ones_like(labels, dtype=torch.bool)
    inlet = torch.zeros_like(connected)
    inlet[:, :, 0] = True
    actions = torch.tensor([
        [0.25, 0.4, 0.0, 0.0],
        [0.70, 0.2, 0.3, 1.0],
        [0.55, 0.6, 0.1, 0.0],
    ])
    times = torch.tensor([[1 / 25], [2 / 25], [3 / 25]])
    config = RegionalConfig(cell_width=4, local_width=4, hidden_width=8)
    models = {}
    for mode in RegionalActionSurrogate.MODES:
        torch.manual_seed(20260920)
        models[mode] = RegionalActionSurrogate(config, mode=mode)
    reference = models["budget"].state_dict()
    for model in models.values():
        assert all(torch.equal(reference[name], model.state_dict()[name])
                   for name in reference)
    outputs = {
        mode: model(labels, inlet, connected, actions, times, return_fields=True)
        for mode, model in models.items()
    }
    assert torch.max(torch.abs(outputs["budget"]["So_final"]
                               - outputs["uncentered"]["So_final"])) > 1e-8
    assert torch.max(torch.abs(outputs["budget"]["So_final"]
                               - outputs["direct"]["So_final"])) > 1e-8
    for result in outputs.values():
        so = result["So_sequence"]
        assert so.shape == (3, 8, 8, 8)
        assert torch.equal(result["So_final"], so[-1])
        assert torch.isfinite(so).all()
        assert bool(torch.all((so >= 0) & (so <= (labels >= 3))))
        assert bool(torch.all(so[1:] <= so[:-1]))
        assert torch.equal(so[:, :, :, 0], torch.zeros_like(so[:, :, :, 0]))
        assert torch.equal(so[0, :, :, 2:], torch.ones_like(so[0, :, :, 2:]))
        assert torch.allclose(result["class_remaining"].sum(1),
                              so.flatten(1).sum(1), atol=2e-5)
        initial_by_class = torch.stack([
            (labels == code).sum() for code in range(3, 8)
        ]).to(so.dtype)
        assert torch.allclose(result["class_mobilized"] + result["class_remaining"],
                              initial_by_class.expand(3, -1), atol=2e-5)
        assert bool(torch.all(result["RF"][1:] >= result["RF"][:-1]))

    result = outputs["budget"]
    sites, group_id = result["site_index"], result["group_id"]
    previous = torch.ones_like(result["remaining_sites"][0])
    for step in range(3):
        eligible = (sites % 8) <= step + 1
        old_total = previous.new_zeros(40).scatter_add(0, group_id, previous)
        active = previous.new_zeros(40).scatter_add(
            0, group_id, previous * eligible
        )
        expected = old_total - active * result["mobilization_fraction"][step]
        assert torch.allclose(result["group_remaining"][step], expected,
                              atol=2e-5, rtol=1e-6)
        previous = result["remaining_sites"][step]

    changed_future = actions.clone()
    changed_future[1:] = 1 - changed_future[1:]
    alternative = models["budget"](labels, inlet, connected, changed_future, times,
                                   return_fields=True)
    assert torch.equal(result["So_sequence"][0], alternative["So_sequence"][0])
    assert torch.max(torch.abs(result["So_sequence"][1:]
                               - alternative["So_sequence"][1:])) > 1e-8

    spatial_weights = torch.linspace(0.1, 1.0, 8).reshape(1, 1, 1, 8)
    loss = (result["So_sequence"] * spatial_weights).sum() + result["RF"].sum()
    loss.backward()
    for parameter in (models["budget"].local_encoder[0].weight,
                      models["budget"].cell_step.gates.weight,
                      models["budget"].class_logits.weight,
                      models["budget"].local_logit.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    with torch.no_grad():
        long_actions = actions[:1].expand(25, -1).clone()
        long_times = torch.arange(1, 26, dtype=actions.dtype).unsqueeze(1) / 25
        long_result = models["budget"](
            labels, inlet, connected, long_actions, long_times
        )
    assert long_result["remaining_sites"].shape == (25, sites.numel())
    assert torch.isfinite(long_result["remaining_sites"]).all()
    assert bool(torch.all(long_result["remaining_sites"][1:]
                          <= long_result["remaining_sites"][:-1]))
    assert torch.isfinite(long_result["RF"]).all()
    tail = models["budget"](labels, inlet, connected, actions, times,
                            start_grad_step=1, return_fields=True)
    assert tail["step_offset"] == 1
    assert torch.equal(tail["So_sequence"], result["So_sequence"][1:])
    assert torch.equal(tail["group_remaining"], result["group_remaining"][1:])

    blocked_connected = connected.clone()
    blocked_connected[:, :, 4:] = False
    with torch.no_grad():
        blocked = models["budget"](
            labels, inlet, blocked_connected, actions, times, return_fields=True
        )
    assert torch.equal(blocked["So_sequence"][:, :, :, 4:],
                       torch.ones_like(blocked["So_sequence"][:, :, :, 4:]))


if __name__ == "__main__":
    _synthetic_self_test()
    print("regional_budget_model synthetic self-test: PASS")
