"""Untrained matched readout prototype; no data, checkpoints or optimizer.

Both arms consume the same kinds of information.  The only arm-dependent
operation is multiplication by predicted contact.  In an actual autoregressive
rollout the arms will have different self-predicted states; equal information
access must not be misreported as identical state values at every time step.

This is a software prototype, not a trained model or a scientific result.
"""

from __future__ import annotations

import torch
from torch import nn


class MatchedContactReadout(nn.Module):
    """A common residual MLP followed by explicit or learned-only contact use.

    Inputs, per initial oil site: previous occupancy, own-state contact,
    original additive logit, static local features, and current cell hidden
    features. Current action and time are shared across sites. The geometric
    reachability mask is applied identically outside the learned head.

    The residual head is zero at initialization, so the explicit arm recovers
    the original direct algebra for a given input. The learned-only arm does
    NOT have identical initial predictions; that is a structural consequence
    of removing the multiplier, not an initialization-weight mismatch.
    """

    MODES = ("explicit_contact", "learned_contact")

    def __init__(self, mode: str, *, local_width: int = 8,
                 cell_width: int = 24, head_width: int = 16):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unknown contact-control mode")
        if any(type(value) is not int or value < 1
               for value in (local_width, cell_width, head_width)):
            raise ValueError("Feature widths must be positive integers")
        self.mode = mode
        self.local_width = local_width
        self.cell_width = cell_width
        # p, contact, original logit, local/cell features, action[4], time[1].
        input_width = 3 + local_width + cell_width + 4 + 1
        self.residual = nn.Sequential(
            nn.Linear(input_width, head_width), nn.SiLU(),
            nn.Linear(head_width, head_width), nn.SiLU(),
            nn.Linear(head_width, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, previous: torch.Tensor, contact: torch.Tensor,
                base_logits: torch.Tensor, local_features: torch.Tensor,
                cell_features: torch.Tensor, action: torch.Tensor,
                time: torch.Tensor, reachable: torch.Tensor,
                group_id: torch.Tensor, n_groups: int,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous.ndim != 1 or previous.numel() == 0:
            raise ValueError("Expected a nonempty initial oil-site vector")
        n = previous.numel()
        if (contact.shape != (n,) or base_logits.shape != (n,)
                or local_features.shape != (n, self.local_width)
                or cell_features.shape != (n, self.cell_width)
                or action.shape != (4,) or time.shape != (1,)
                or reachable.shape != (n,) or reachable.dtype != torch.bool
                or group_id.shape != (n,) or group_id.dtype != torch.long
                or type(n_groups) is not int or n_groups < 1):
            raise ValueError("Input shapes, masks, or group specification differ")
        floating = (previous, contact, base_logits, local_features,
                    cell_features, action, time)
        parameter = next(self.parameters())
        if (not previous.is_floating_point()
                or any(value.dtype != parameter.dtype for value in floating)
                or any(value.device != parameter.device for value in
                       (*floating, reachable, group_id))):
            raise ValueError("All tensors and parameters must share dtype/device")
        if (any(not bool(torch.isfinite(value).all()) for value in floating)
                or bool(torch.any((previous < 0) | (previous > 1)))
                or bool(torch.any((contact < 0) | (contact > 1)))
                or bool(torch.any((time < 0) | (time > 1)))
                or bool(torch.any((group_id < 0) | (group_id >= n_groups)))):
            raise ValueError("Non-finite or out-of-domain readout inputs")
        features = torch.cat((
            previous[:, None], contact[:, None], base_logits[:, None],
            local_features, cell_features,
            action.expand(n, -1), time.expand(n, -1),
        ), dim=1)
        propensity = torch.sigmoid(
            base_logits + self.residual(features).squeeze(1)
        )
        # Keep these two operations explicit: this is the isolated factor.
        hazard = reachable.to(previous.dtype) * propensity
        if self.mode == "explicit_contact":
            hazard = hazard * contact
        remaining = previous * (1 - hazard)
        inventory = previous.new_zeros(n_groups).scatter_add(
            0, group_id, remaining
        )
        return remaining, inventory
