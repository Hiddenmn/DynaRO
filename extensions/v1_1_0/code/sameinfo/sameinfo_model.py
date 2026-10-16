"""Untrained matched-backbone controls; no automatic data or checkpoint access.

The forward pass follows the preserved public state-front predictor, with a
shared local residual MLP in both arms. This is a new controlled variant, not
a rename or replacement of the original DynaRO checkpoint.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SOURCE = Path(__file__).resolve().parents[1] / "model_core" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from regional_budget_model import RegionalActionSurrogate, RegionalConfig
from regional_budget_readout import full_so_from_sites, initial_oil_groups
from softfront_budget_candidate import SparseFrontGeometry
from upt3d_readout import iter_reachable_oil_support

from matched_contact_readout import MatchedContactReadout


class SameInformationContactSurrogate(RegionalActionSurrogate):
    def __init__(self, control_mode: str,
                 config: RegionalConfig = RegionalConfig(), *, head_width: int = 16):
        super().__init__(config, mode="direct")
        self.contact_readout = MatchedContactReadout(
            control_mode, local_width=config.local_width,
            cell_width=config.hidden_width, head_width=head_width,
        )

    def forward(self, initial_labels: torch.Tensor, inlet: torch.Tensor,
                connected: torch.Tensor, actions: torch.Tensor, times: torch.Tensor,
                *, return_fields: bool = False, start_grad_step: int = 0,
                compact_output: bool = False, checkpoint_local: bool = False) -> dict:
        self._validate_input(initial_labels, inlet, connected, actions, times)
        if any(type(flag) is not bool for flag in
               (return_fields, compact_output, checkpoint_local)):
            raise ValueError("Output and checkpoint flags must be boolean")
        if return_fields and compact_output:
            raise ValueError("Compact output cannot include full 3-D fields")
        if type(start_grad_step) is not int or not 0 <= start_grad_step < len(actions):
            raise ValueError("Invalid gradient start step")
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
        use_checkpoint = checkpoint_local and torch.is_grad_enabled()
        local = (checkpoint(self.local_encoder, channels, use_reentrant=False)
                 if use_checkpoint else self.local_encoder(channels))
        local_sites = local[0].flatten(1).transpose(0, 1)[site_index]
        cell_static = F.avg_pool3d(local, width, stride=width)
        hidden = self.cell_initial(cell_static).tanh()
        static_logits = self.local_logit(local).reshape(-1)[site_index]
        cell_id = group_id // 5
        initial_inventory = actions.new_zeros(n_groups).scatter_add(
            0, group_id, actions.new_ones(site_index.numel())
        )
        previous = actions.new_ones(site_index.numel())
        previous_inventory = initial_inventory
        front_geometry = SparseFrontGeometry.from_initial(
            initial_labels, site_index, inlet, connected
        )
        supports = iter_reachable_oil_support(
            geometry, inlet=inlet[None, None], connected=connected[None, None],
            steps=len(actions),
        )
        cell_shape = cell_static.shape[2:]
        states, inventories, fields = [], [], []
        for step, reachable in enumerate(supports):
            with (torch.no_grad() if step < start_grad_step else nullcontext()):
                fraction = (previous_inventory / initial_inventory.clamp_min(1))
                fraction = fraction.reshape(*cell_shape, 5).permute(3, 0, 1, 2)[None]
                action_map = actions[step].reshape(1, 4, 1, 1, 1).expand(
                    1, 4, *cell_shape
                )
                time_map = times[step].reshape(1, 1, 1, 1, 1).expand(
                    1, 1, *cell_shape
                )
                hidden = self.cell_step(hidden, torch.cat(
                    (cell_static, fraction, action_map, time_map), dim=1
                ))
                group_logits = self.class_logits(hidden).permute(
                    0, 2, 3, 4, 1
                ).reshape(-1)
                base_logits = (static_logits
                               + self.cell_local_bias(hidden).reshape(-1)[cell_id]
                               + group_logits[group_id])
                recompute = checkpoint_local and torch.is_grad_enabled()
                contact = (checkpoint(front_geometry.weight, previous,
                                      use_reentrant=False) if recompute
                           else front_geometry.weight(previous))
                cell_sites = hidden[0].flatten(1).transpose(0, 1)[cell_id]
                arguments = (
                    previous, contact, base_logits, local_sites, cell_sites,
                    actions[step], times[step],
                    reachable.reshape(-1)[site_index], group_id, n_groups,
                )
                previous, previous_inventory = (
                    checkpoint(self.contact_readout, *arguments, use_reentrant=False)
                    if recompute else self.contact_readout(*arguments)
                )
            if step < start_grad_step:
                continue
            states.append(previous)
            inventories.append(previous_inventory)
            if return_fields:
                fields.append(full_so_from_sites(initial_labels, site_index, previous))
        remaining_sites = torch.stack(states)
        group_remaining = torch.stack(inventories)
        output = {
            "step_offset": start_grad_step, "site_index": site_index,
            "group_id": group_id, "initial_group_inventory": initial_inventory,
            "remaining_sites": remaining_sites, "group_remaining": group_remaining,
        }
        if compact_output:
            return output
        initial_class = initial_inventory.reshape(-1, 5).sum(0)
        class_remaining = group_remaining.reshape(len(states), -1, 5).sum(1)
        total_initial = initial_class.sum()
        total_remaining = class_remaining.sum(1)
        output.update({
            "class_remaining": class_remaining,
            "class_mobilized": initial_class[None] - class_remaining,
            "So_final": full_so_from_sites(initial_labels, site_index, remaining_sites[-1]),
            "Sor": total_remaining / pore.sum().to(actions.dtype),
            "RF": (total_initial - total_remaining) / total_initial.clamp_min(1),
        })
        if return_fields:
            output["So_sequence"] = torch.stack(fields)
        return output
