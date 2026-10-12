"""Causal remaining-oil adaptations of FNO and recurrent transformer U-Net.

Paper-guided implementations, NOT downloaded authors' models/checkpoints:
  FNO: https://arxiv.org/abs/2010.08895
  RTU: https://arxiv.org/html/2508.16631v1 (Section 3 / Table 2)

The shared survival head and action inputs are remaining-oil task adaptations.
Neither external backbone receives the explicit predicted-state front. It can
learn exposure from its own previous full-resolution oil field. No trajectory,
checkpoint, optimizer, DEV/TEST reader, or performance selection exists here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from regional_budget_model import RegionalActionSurrogate, RegionalConfig
from regional_budget_readout import full_so_from_sites, initial_oil_groups
from softfront_budget_candidate import SparseFrontGeometry, front_weighted_direct_step
from upt3d_readout import iter_reachable_oil_support


def _recompute(function, *args, enabled: bool):
    if enabled and torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


def _fields_and_counts(labels, site_index, group_id, initial, sites, groups,
                       *, offset: int, compact: bool, return_fields: bool) -> dict:
    output = {
        "step_offset": offset, "site_index": site_index, "group_id": group_id,
        "initial_group_inventory": initial, "remaining_sites": sites,
        "group_remaining": groups,
    }
    if compact:
        return output
    initial_class = initial.reshape(-1, 5).sum(0)
    class_remaining = groups.reshape(groups.shape[0], -1, 5).sum(1)
    initial_total = initial_class.sum()
    output.update({
        "class_remaining": class_remaining,
        "class_mobilized": initial_class.unsqueeze(0) - class_remaining,
        "So_final": full_so_from_sites(labels, site_index, sites[-1]),
        "Sor": class_remaining.sum(1) / (labels >= 2).sum(),
        "RF": (initial_total - class_remaining.sum(1)) / initial_total.clamp_min(1),
    })
    if return_fields:
        output["So_sequence"] = torch.stack([
            full_so_from_sites(labels, site_index, value) for value in sites
        ])
    return output


@dataclass(frozen=True)
class FNOConfig:
    cell_width: int = 16
    width: int = 24
    modes: tuple[int, int, int] = (8, 8, 8)
    layers: int = 4
    padding: int = 8

    def __post_init__(self):
        if (any(type(value) is not int or value < 1 for value in
                (self.cell_width, self.width, self.layers, *self.modes))
                or len(self.modes) != 3 or type(self.padding) is not int
                or self.padding < 0):
            raise ValueError("Positive FNO widths/modes and nonnegative padding required")


class SpectralConv3D(nn.Module):
    """Four low-frequency corners of a three-spatial-dimensional real FFT.

    Real/imaginary coordinates are real Parameters, so real gradient norms and
    the existing StateGuard act on all real degrees of freedom correctly.
    """

    def __init__(self, width: int, modes: tuple[int, int, int]):
        super().__init__()
        self.width, self.modes = width, modes
        self.weight_ri = nn.Parameter(
            torch.rand(4, width, width, *modes, 2) / (width * width)
        )

    def forward(self, value):
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError("FNO FFT uses float32/float64, not implicit mixed precision")
        z, y, x = value.shape[-3:]
        mz, my, mx = self.modes
        if 2 * mz > z or 2 * my > y or mx > x // 2 + 1:
            raise ValueError("Spectral corners overlap or exceed the grid")
        spectrum = torch.fft.rfftn(value, dim=(-3, -2, -1), norm="ortho")
        weights = torch.view_as_complex(self.weight_ri.contiguous())
        corners = ((slice(0, mz), slice(0, my)),
                   (slice(-mz, None), slice(0, my)),
                   (slice(0, mz), slice(-my, None)),
                   (slice(-mz, None), slice(-my, None)))
        # Clone only the selected coefficients: parameter gradients must not
        # retain a view whose storage is the entire dense input spectrum.
        selected = torch.stack([
            spectrum[:, :, zs, ys, :mx].clone() for zs, ys in corners
        ])
        del spectrum
        transformed = torch.einsum("kbizyx,kiozyx->kbozyx", selected, weights)
        x_bins = x // 2 + 1
        indices = torch.stack([
            (torch.arange(z, device=value.device)[zs, None, None] * (y * x_bins)
             + torch.arange(y, device=value.device)[None, ys, None] * x_bins
             + torch.arange(mx, device=value.device)[None, None, :]).reshape(-1)
            for zs, ys in corners
        ]).reshape(-1)
        coefficients = transformed.permute(1, 2, 0, 3, 4, 5).reshape(
            value.shape[0], self.width, -1
        )
        # One scatter has the same four-corner projection as slice assignment,
        # without a chain of dense CopySlices backward intermediates.
        projected = coefficients.new_zeros(value.shape[0], self.width, z * y * x_bins)
        projected.scatter_(2, indices.reshape(1, 1, -1).expand_as(coefficients), coefficients)
        projected = projected.reshape(value.shape[0], self.width, z, y, x_bins)
        return torch.fft.irfftn(projected, s=(z, y, x), dim=(-3, -2, -1),
                               norm="ortho")


class FourierBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv3D(width, modes)
        self.local = nn.Conv3d(width, width, 1)

    def forward(self, value):
        return F.gelu(self.spectral(value) + self.local(value))


class _CausalOilRollout(nn.Module):
    """Same grouping/targets API as the current regional surrogate.

    Whole-transition checkpointing takes the sparse previous state, NOT an
    already materialized dense input. It recomputes spatial activations without
    detaching the temporal state; full-chain gradients remain available.
    """

    _validate_input = RegionalActionSurrogate._validate_input

    def _initial_memory(self, labels, reference):
        return reference.new_empty(0), reference.new_empty(0)

    def _static_channels(self, labels, inlet, connected, reference):
        return torch.stack([labels == code for code in range(8)] + [inlet, connected],
                           0).unsqueeze(0).to(reference.dtype)

    def _spatial_context(self, labels, reference):
        return reference.new_empty(0)

    def forward(self, initial_labels, inlet, connected, actions, times, *,
                return_fields=False, start_grad_step=0, compact_output=False,
                checkpoint_local=False):
        self._validate_input(initial_labels, inlet, connected, actions, times)
        if (any(type(flag) is not bool for flag in
                (return_fields, compact_output, checkpoint_local))
                or (return_fields and compact_output)
                or type(start_grad_step) is not int
                or not 0 <= start_grad_step < actions.shape[0]):
            raise ValueError("Invalid output/checkpoint/window flags")
        site_index, group_id, n_groups = initial_oil_groups(
            initial_labels, cell_width=self.config.cell_width
        )
        initial = actions.new_zeros(n_groups).scatter_add(
            0, group_id, actions.new_ones(site_index.numel())
        )
        static = self._static_channels(initial_labels, inlet, connected, actions)
        context = self._spatial_context(initial_labels, actions)
        pore = initial_labels >= 2
        geometry = torch.cat((pore[None], torch.stack([
            initial_labels == code for code in range(3, 8)
        ])), 0).unsqueeze(0)
        supports = iter_reachable_oil_support(
            geometry, inlet=inlet[None, None], connected=connected[None, None],
            steps=actions.shape[0]
        )
        previous = actions.new_ones(site_index.numel())
        hidden, cell = self._initial_memory(initial_labels, actions)
        states, inventories = [], []

        def transition(previous, hidden, cell, action, time, reachable):
            logits, next_hidden, next_cell = self._network_step(
                previous, hidden, cell, action, time, static, context, site_index,
                checkpoint_local=checkpoint_local,
            )
            site_logits = logits.reshape(-1)[site_index]
            hazard = reachable.reshape(-1)[site_index].to(actions.dtype) * site_logits.sigmoid()
            return previous * (1 - hazard), next_hidden, next_cell

        for step, reachable in enumerate(supports):
            with (torch.no_grad() if step < start_grad_step else nullcontext()):
                previous, hidden, cell = _recompute(
                    transition, previous, hidden, cell, actions[step], times[step],
                    reachable, enabled=checkpoint_local,
                )
                inventory = actions.new_zeros(n_groups).scatter_add(0, group_id, previous)
            if step >= start_grad_step:
                states.append(previous)
                inventories.append(inventory)
        return _fields_and_counts(
            initial_labels, site_index, group_id, initial, torch.stack(states),
            torch.stack(inventories), offset=start_grad_step, compact=compact_output,
            return_fields=return_fields,
        )


class ARFNO3DRemainingOil(_CausalOilRollout):
    """Action-conditioned spatial FNO, autoregressive in predicted oil only."""

    def __init__(self, config=FNOConfig()):
        super().__init__()
        self.config = config
        # Static 10 + previous oil 1 + action/time 5 + coordinates 3.
        self.lift = nn.Conv3d(19, config.width, 1)
        self.blocks = nn.ModuleList([
            FourierBlock(config.width, config.modes) for _ in range(config.layers)
        ])
        self.project = nn.Sequential(
            nn.Conv3d(config.width, 2 * config.width, 1), nn.GELU(),
            nn.Conv3d(2 * config.width, 1, 1),
        )
        with torch.no_grad():
            self.project[-1].bias.fill_(-5.0)

    def _spatial_context(self, labels, reference):
        coordinates = torch.meshgrid(*[
            torch.linspace(0, 1, count, device=labels.device, dtype=reference.dtype)
            for count in labels.shape
        ], indexing="ij")
        return torch.stack(coordinates, 0).unsqueeze(0)

    def _network_step(self, previous, hidden, cell, action, time, static,
                      coordinates, site_index, *, checkpoint_local):
        oil = previous.new_zeros(static.shape[2:]).reshape(-1).scatter(
            0, site_index, previous
        ).reshape(1, 1, *static.shape[2:])
        control = torch.cat((action, time)).reshape(1, 5, 1, 1, 1).expand(
            1, 5, *static.shape[2:]
        )
        value = self.lift(torch.cat((static, oil, control, coordinates), 1))
        padding = self.config.padding
        if padding:
            value = F.pad(value, (padding,) * 6)
        for block in self.blocks:
            value = _recompute(block, value, enabled=checkpoint_local)
        if padding:
            value = value[:, :, padding:-padding, padding:-padding, padding:-padding]
        return _recompute(self.project, value, enabled=checkpoint_local), hidden, cell


@dataclass(frozen=True)
class RTUConfig:
    cell_width: int = 16
    base_width: int = 16
    heads: int = 4
    transformer_layers: int = 2
    positional_shape: tuple[int, int, int] = (16, 16, 16)

    def __post_init__(self):
        if (any(type(value) is not int or value < 1 for value in
                (self.cell_width, self.base_width, self.heads,
                 self.transformer_layers, *self.positional_shape))
                or len(self.positional_shape) != 3
                or 4 * self.base_width % self.heads):
            raise ValueError("Invalid RTU widths/heads/position shape")


class Residual3D(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        self.conv1 = nn.Conv3d(input_width, output_width, 3, padding=1)
        self.conv2 = nn.Conv3d(output_width, output_width, 3, padding=1)
        self.skip = (nn.Identity() if input_width == output_width else
                     nn.Conv3d(input_width, output_width, 1))

    def forward(self, value):
        return F.relu(self.conv2(F.relu(self.conv1(value))) + self.skip(value))


class AttentionGate3D(nn.Module):
    def __init__(self, skip_width, gate_width, *, stride):
        super().__init__()
        intermediate = max(1, min(skip_width, gate_width) // 2)
        self.theta = nn.Conv3d(skip_width, intermediate, stride, stride=stride)
        self.phi = nn.Conv3d(gate_width, intermediate, 1)
        self.psi = nn.Conv3d(intermediate, 1, 1)

    def forward(self, skip, gate):
        alpha = self.psi(F.relu(self.theta(skip) + self.phi(gate))).sigmoid()
        alpha = F.interpolate(alpha, size=skip.shape[2:], mode="trilinear",
                              align_corners=False)
        return skip * alpha


class ConvLSTM3D(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gates = nn.Conv3d(2 * width + 5, 4 * width, 3, padding=1)

    def forward(self, features, hidden, cell, control):
        inputs = torch.cat((features, hidden, control), 1)
        input_gate, forget, output_gate, proposed = self.gates(inputs).chunk(4, 1)
        next_cell = forget.sigmoid() * cell + input_gate.sigmoid() * proposed.tanh()
        return output_gate.sigmoid() * next_cell.tanh(), next_cell


class RTU3DRemainingOil(_CausalOilRollout):
    """Residual/attention-gated U-Net + spatial transformer + ConvLSTM.

    Re-encodes previous predicted oil alongside the static initial state. Action
    and common time features enter the ConvLSTM bottleneck. ReLU residual blocks,
    pre-norm GELU transformer, zero dropout, trilinear upsampling and the common
    survival head are explicit implementation/task choices, not claimed exact
    matches to an unavailable authors' code release.
    """

    def __init__(self, config=RTUConfig()):
        super().__init__()
        self.config = config
        a, b, c = config.base_width, 2 * config.base_width, 4 * config.base_width
        self.encoder = nn.ModuleList([
            Residual3D(11, a), Residual3D(a, b), Residual3D(b, b), Residual3D(b, c),
        ])
        self.position = nn.Parameter(torch.empty(1, c, *config.positional_shape))
        nn.init.normal_(self.position, std=0.02)
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                c, config.heads, dim_feedforward=c, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(config.transformer_layers)
        ])
        self.temporal = ConvLSTM3D(c)
        self.attention = nn.ModuleList([
            AttentionGate3D(c, c, stride=2), AttentionGate3D(b, c, stride=2),
            AttentionGate3D(b, b, stride=1), AttentionGate3D(a, b, stride=2),
        ])
        self.decoder = nn.ModuleList([
            Residual3D(2 * c, c), Residual3D(c + b, b),
            Residual3D(2 * b, b), Residual3D(b + a, a),
        ])
        self.head = nn.Conv3d(a, 1, 3, padding=1)
        with torch.no_grad():
            self.head.bias.fill_(-5.0)

    def _initial_memory(self, labels, reference):
        if min(labels.shape) < 8 or any(count % 8 for count in labels.shape):
            raise ValueError("RTU requires each dimension to be a multiple of eight")
        shape = (1, 4 * self.config.base_width, *[count // 8 for count in labels.shape])
        return reference.new_zeros(shape), reference.new_zeros(shape)

    def _network_step(self, previous, hidden, cell, action, time, static,
                      context, site_index, *, checkpoint_local):
        oil = previous.new_zeros(static.shape[2:]).reshape(-1).scatter(
            0, site_index, previous
        ).reshape(1, 1, *static.shape[2:])
        f1 = _recompute(self.encoder[0], torch.cat((static, oil), 1),
                        enabled=checkpoint_local)
        f2 = _recompute(self.encoder[1], F.max_pool3d(f1, 2), enabled=checkpoint_local)
        f3 = _recompute(self.encoder[2], f2, enabled=checkpoint_local)
        f4 = _recompute(self.encoder[3], F.max_pool3d(f3, 2), enabled=checkpoint_local)
        bottleneck = F.max_pool3d(f4, 2)
        position = F.interpolate(self.position, size=bottleneck.shape[2:],
                                 mode="trilinear", align_corners=False)
        tokens = (bottleneck + position).flatten(2).transpose(1, 2)
        for layer in self.transformer:
            tokens = _recompute(layer, tokens, enabled=checkpoint_local)
        features = tokens.transpose(1, 2).reshape_as(bottleneck)
        control = torch.cat((action, time)).reshape(1, 5, 1, 1, 1).expand(
            1, 5, *bottleneck.shape[2:]
        )
        next_hidden, next_cell = self.temporal(features, hidden, cell, control)
        decoded = next_hidden
        for skip, gate, block in zip((f4, f3, f2, f1), self.attention, self.decoder):
            gated = _recompute(gate, skip, decoded, enabled=checkpoint_local)
            upsampled = F.interpolate(decoded, size=skip.shape[2:], mode="trilinear",
                                      align_corners=False)
            decoded = _recompute(block, torch.cat((upsampled, gated), 1),
                                 enabled=checkpoint_local)
        return self.head(decoded), next_hidden, next_cell


class CoarseStateFrontRemainingOil(RegionalActionSurrogate):
    """Same 90,991-parameter model: only front sees region/class mean oil.

    Fine site survival, learned logits, recurrent memory and regional feedback
    remain unchanged. Parameter keys match the already trained fine-front arm.
    """

    def __init__(self, config=RegionalConfig()):
        super().__init__(config, mode="direct")

    def forward(self, initial_labels, inlet, connected, actions, times, *,
                return_fields=False, start_grad_step=0, compact_output=False,
                checkpoint_local=False):
        self._validate_input(initial_labels, inlet, connected, actions, times)
        if (any(type(flag) is not bool for flag in
                (return_fields, compact_output, checkpoint_local))
                or (return_fields and compact_output)
                or type(start_grad_step) is not int
                or not 0 <= start_grad_step < actions.shape[0]):
            raise ValueError("Invalid output/checkpoint/window flags")
        site_index, group_id, n_groups = initial_oil_groups(
            initial_labels, cell_width=self.config.cell_width
        )
        pore = initial_labels >= 2
        geometry = torch.cat((pore[None], torch.stack([
            initial_labels == code for code in range(3, 8)
        ])), 0).unsqueeze(0)
        channels = torch.stack([initial_labels == code for code in range(8)]
                               + [inlet, connected], 0).unsqueeze(0).to(actions.dtype)
        local = _recompute(self.local_encoder, channels, enabled=checkpoint_local)
        cell_static = F.avg_pool3d(local, self.config.cell_width,
                                  stride=self.config.cell_width)
        hidden = self.cell_initial(cell_static).tanh()
        static_logits = self.local_logit(local).reshape(-1)[site_index]
        initial = actions.new_zeros(n_groups).scatter_add(
            0, group_id, actions.new_ones(site_index.numel())
        )
        previous, inventory = actions.new_ones(site_index.numel()), initial
        front = SparseFrontGeometry.from_initial(initial_labels, site_index, inlet, connected)
        supports = iter_reachable_oil_support(
            geometry, inlet=inlet[None, None], connected=connected[None, None],
            steps=actions.shape[0]
        )
        states, inventories = [], []
        shape = cell_static.shape[2:]

        def coarse_weight(inventory):
            return front.weight((inventory / initial.clamp_min(1))[group_id])

        for step, reachable in enumerate(supports):
            with (torch.no_grad() if step < start_grad_step else nullcontext()):
                fractions = (inventory / initial.clamp_min(1)).reshape(
                    *shape, 5
                ).permute(3, 0, 1, 2).unsqueeze(0)
                action_map = actions[step].reshape(1, 4, 1, 1, 1).expand(1, 4, *shape)
                time_map = times[step].reshape(1, 1, 1, 1, 1).expand(1, 1, *shape)
                hidden = self.cell_step(hidden, torch.cat(
                    (cell_static, fractions, action_map, time_map), 1
                ))
                group_logits = self.class_logits(hidden).permute(0, 2, 3, 4, 1).reshape(-1)
                site_logits = static_logits + self.cell_local_bias(hidden).reshape(-1)[group_id // 5]
                weight = _recompute(coarse_weight, inventory, enabled=checkpoint_local)
                weight = weight * reachable.reshape(-1)[site_index].to(weight.dtype)
                previous, inventory = front_weighted_direct_step(
                    previous, weight, site_logits, group_id, group_logits
                )
            if step >= start_grad_step:
                states.append(previous)
                inventories.append(inventory)
        return _fields_and_counts(
            initial_labels, site_index, group_id, initial, torch.stack(states),
            torch.stack(inventories), offset=start_grad_step, compact=compact_output,
            return_fields=return_fields,
        )


def make_model(arm: str, *, small=False):
    """Default resource candidates; not frozen formal TRAIN configurations."""
    if arm == "fno":
        config = (FNOConfig(cell_width=4, width=4, modes=(2, 2, 2), padding=2)
                  if small else FNOConfig())
        return ARFNO3DRemainingOil(config)
    if arm == "rtu":
        config = (RTUConfig(cell_width=4, base_width=4, positional_shape=(2, 2, 2))
                  if small else RTUConfig())
        return RTU3DRemainingOil(config)
    if arm == "coarse_front":
        config = RegionalConfig(cell_width=4, local_width=4, hidden_width=8) if small else RegionalConfig()
        model = CoarseStateFrontRemainingOil(config)
        with torch.no_grad():
            model.class_logits.bias.fill_(-5.0)
        return model
    raise ValueError("Unknown TNNLS comparison arm")
