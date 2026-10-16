"""Untrained state-driven front and exact class-cell inventory update.

The front is recomputed from predicted oil at every step, rather than from a
pre-expanded time cone. This is an algorithm candidate, not a validated model,
physical flow solver, or authorization to train/evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from full3d_order_recurrence import six_neighbour_max


def predicted_front_weight(
    initial_labels: torch.Tensor,
    site_index: torch.Tensor,
    previous_sites: torch.Tensor,
    inlet: torch.Tensor,
    connected: torch.Tensor,
) -> torch.Tensor:
    """Six-face front weight on initial oil sites from the predicted state.

    Water probability is one minus predicted oil occupancy inside connected
    pores. Initial-water sites therefore have unit source strength. No future
    simulator state or target is used.
    """
    if initial_labels.ndim != 3 or inlet.shape != initial_labels.shape or connected.shape != initial_labels.shape:
        raise ValueError("Expected matching 3-D labels and masks")
    if inlet.dtype != torch.bool or connected.dtype != torch.bool:
        raise ValueError("Inlet and connected masks must be boolean")
    if site_index.ndim != 1 or site_index.dtype != torch.long or previous_sites.shape != site_index.shape:
        raise ValueError("Expected one probability per initial oil site")
    if not previous_sites.is_floating_point() or not torch.isfinite(previous_sites).all():
        raise ValueError("Predicted oil must be finite and floating point")
    if len({v.device for v in (initial_labels, site_index, previous_sites, inlet, connected)}) != 1:
        raise ValueError("All inputs must share a device")
    flat = initial_labels.reshape(-1)
    if (torch.any((site_index < 0) | (site_index >= flat.numel()))
            or torch.any(flat[site_index] < 3)
            or torch.any((previous_sites < 0) | (previous_sites > 1))):
        raise ValueError("Initial oil sites or predicted occupancy are invalid")
    pore = initial_labels >= 2
    if torch.any(connected & ~pore) or torch.any(inlet & ~connected):
        raise ValueError("Require inlet subset of connected subset of pore")
    oil = previous_sites.new_zeros(flat.numel()).scatter(0, site_index, previous_sites)
    oil = oil.reshape(initial_labels.shape)
    water = connected.to(previous_sites.dtype) - oil
    source = torch.maximum(water, inlet.to(previous_sites.dtype))
    neighbour_water = six_neighbour_max(source[None, None])[0, 0]
    return neighbour_water.reshape(-1)[site_index]


@dataclass(frozen=True)
class SparseFrontGeometry:
    """Cache six-face gather indices once for a variable-length rollout."""

    site_index: torch.Tensor
    neighbour_index: torch.Tensor
    connected_flat: torch.Tensor
    inlet_flat: torch.Tensor
    voxel_count: int

    @classmethod
    def from_initial(
        cls, initial_labels: torch.Tensor, site_index: torch.Tensor,
        inlet: torch.Tensor, connected: torch.Tensor,
    ) -> "SparseFrontGeometry":
        if initial_labels.ndim != 3 or inlet.shape != initial_labels.shape or connected.shape != initial_labels.shape:
            raise ValueError("Expected matching 3-D labels and masks")
        if inlet.dtype != torch.bool or connected.dtype != torch.bool:
            raise ValueError("Inlet and connected masks must be boolean")
        if site_index.ndim != 1 or site_index.dtype != torch.long:
            raise ValueError("Expected flat initial oil indices")
        if len({v.device for v in (initial_labels, site_index, inlet, connected)}) != 1:
            raise ValueError("All geometry inputs must share a device")
        flat = initial_labels.reshape(-1)
        if (torch.any((site_index < 0) | (site_index >= flat.numel()))
                or torch.any(flat[site_index] < 3)
                or torch.any(connected & ~(initial_labels >= 2))
                or torch.any(inlet & ~connected)):
            raise ValueError("Invalid oil sites or inlet-connected geometry")
        z_size, y_size, x_size = initial_labels.shape
        plane = y_size * x_size
        z = site_index // plane
        y = (site_index // x_size) % y_size
        x = site_index % x_size
        sentinel = flat.numel()
        neighbours = torch.stack((
            torch.where(z > 0, site_index - plane, sentinel),
            torch.where(z + 1 < z_size, site_index + plane, sentinel),
            torch.where(y > 0, site_index - x_size, sentinel),
            torch.where(y + 1 < y_size, site_index + x_size, sentinel),
            torch.where(x > 0, site_index - 1, sentinel),
            torch.where(x + 1 < x_size, site_index + 1, sentinel),
        ), dim=1)
        return cls(site_index, neighbours, connected.reshape(-1),
                   inlet.reshape(-1), sentinel)

    def weight(self, previous_sites: torch.Tensor) -> torch.Tensor:
        if (previous_sites.shape != self.site_index.shape
                or not previous_sites.is_floating_point()
                or previous_sites.device != self.site_index.device
                or not torch.isfinite(previous_sites).all()
                or torch.any((previous_sites < 0) | (previous_sites > 1))):
            raise ValueError("Predicted site occupancy is invalid")
        oil = previous_sites.new_zeros(self.voxel_count).scatter(
            0, self.site_index, previous_sites
        )
        water = self.connected_flat.to(previous_sites.dtype) - oil
        source = torch.maximum(water, self.inlet_flat.to(previous_sites.dtype))
        source = torch.cat((source, source.new_zeros(1)))
        return source[self.neighbour_index].amax(dim=1)


def front_weighted_budget_step(
    previous_sites: torch.Tensor,
    front_weight: torch.Tensor,
    site_logits: torch.Tensor,
    group_id: torch.Tensor,
    group_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spend each class-cell fraction of front-exposed oil, exactly by group.

    With I_g=sum_x p_x w_x, f_g=sigmoid(group_logit_g), and a centered local
    propensity, the updated group inventory is old_inventory_g - f_g I_g.
    This identity holds for soft front weights, including partial occupancy.
    """
    if (previous_sites.ndim != 1 or front_weight.shape != previous_sites.shape
            or site_logits.shape != previous_sites.shape
            or group_id.shape != previous_sites.shape
            or group_id.dtype != torch.long or group_logits.ndim != 1):
        raise ValueError("Site and group tensor shapes do not match")
    if (not previous_sites.is_floating_point()
            or previous_sites.dtype != front_weight.dtype
            or previous_sites.dtype != site_logits.dtype
            or previous_sites.dtype != group_logits.dtype):
        raise ValueError("All floating tensors must share a dtype")
    if len({v.device for v in (previous_sites, front_weight, site_logits, group_id, group_logits)}) != 1:
        raise ValueError("All inputs must share a device")
    if (not all(torch.isfinite(v).all() for v in
                (previous_sites, front_weight, site_logits, group_logits))
            or torch.any((previous_sites < 0) | (previous_sites > 1))
            or torch.any((front_weight < 0) | (front_weight > 1))
            or torch.any((group_id < 0) | (group_id >= group_logits.numel()))):
        raise ValueError("Non-finite or out-of-range site or group values")
    n_groups = group_logits.numel()
    exposed = previous_sites * front_weight
    exposed_inventory = previous_sites.new_zeros(n_groups).scatter_add(0, group_id, exposed)
    propensity = torch.sigmoid(site_logits)
    weighted_propensity = previous_sites.new_zeros(n_groups).scatter_add(
        0, group_id, exposed * propensity
    )
    mean_propensity = torch.where(
        exposed_inventory > 0,
        weighted_propensity / exposed_inventory.clamp_min(torch.finfo(previous_sites.dtype).tiny),
        torch.zeros_like(exposed_inventory),
    )
    fraction = torch.sigmoid(group_logits)[group_id]
    centred = propensity - mean_propensity[group_id]
    hazard = front_weight * (fraction + fraction * (1 - fraction) * centred)
    next_sites = previous_sites * (1 - hazard)
    next_inventory = previous_sites.new_zeros(n_groups).scatter_add(0, group_id, next_sites)
    return next_sites, next_inventory


def front_weighted_direct_step(
    previous_sites: torch.Tensor,
    front_weight: torch.Tensor,
    site_logits: torch.Tensor,
    group_id: torch.Tensor,
    group_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """State-dependent exposure with the frozen StateGuard direct hazard.

    Only the eligibility factor changes: the original direct readout uses a
    binary, geometry/time-derived mask; this uses a predicted-state weight.
    The five-class/cell inventory is still the exact sum of updated sites.
    """
    if (previous_sites.ndim != 1 or front_weight.shape != previous_sites.shape
            or site_logits.shape != previous_sites.shape
            or group_id.shape != previous_sites.shape
            or group_id.dtype != torch.long or group_logits.ndim != 1):
        raise ValueError("Site and group tensor shapes do not match")
    if (previous_sites.dtype != front_weight.dtype
            or previous_sites.dtype != site_logits.dtype
            or previous_sites.dtype != group_logits.dtype
            or not previous_sites.is_floating_point()
            or len({value.device for value in (
                previous_sites, front_weight, site_logits, group_id, group_logits
            )}) != 1):
        raise ValueError("Site and group tensor types do not match")
    if (not torch.isfinite(previous_sites).all()
            or not torch.isfinite(front_weight).all()
            or not torch.isfinite(site_logits).all()
            or not torch.isfinite(group_logits).all()
            or torch.any((previous_sites < 0) | (previous_sites > 1))
            or torch.any((front_weight < 0) | (front_weight > 1))
            or torch.any((group_id < 0) | (group_id >= group_logits.numel()))):
        raise ValueError("Non-finite or out-of-range site or group values")
    hazard = front_weight * torch.sigmoid(site_logits + group_logits[group_id])
    next_sites = previous_sites * (1 - hazard)
    next_inventory = previous_sites.new_zeros(group_logits.numel()).scatter_add(
        0, group_id, next_sites
    )
    return next_sites, next_inventory


def _synthetic_self_test() -> None:
    # One water site, followed by three connected oil sites: no two-edge leap.
    labels = torch.tensor([[[2, 3, 3, 3]]], dtype=torch.uint8)
    sites = torch.tensor([1, 2, 3], dtype=torch.long)
    inlet = torch.tensor([[[True, False, False, False]]])
    connected = torch.ones_like(inlet)
    previous = torch.ones(3, dtype=torch.float64, requires_grad=True)
    front = predicted_front_weight(labels, sites, previous, inlet, connected)
    assert torch.equal(front, torch.tensor([1., 0., 0.], dtype=front.dtype))
    site_logits = torch.tensor([-1., 0., 1.], dtype=torch.float64, requires_grad=True)
    group_logits = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
    groups = torch.zeros(3, dtype=torch.long)
    updated, inventory = front_weighted_budget_step(
        previous, front, site_logits, groups, group_logits
    )
    from regional_budget_readout import readout_from_shared_logits
    direct_sites, direct_inventory = front_weighted_direct_step(
        previous, front, site_logits, groups, group_logits
    )
    old_direct, old_inventory = readout_from_shared_logits(
        previous, site_logits, groups, group_logits, mode="direct",
        eligible=front.bool(),
    )
    assert torch.allclose(direct_sites, old_direct, atol=1e-12)
    assert torch.allclose(direct_inventory, old_inventory, atol=1e-12)
    expected = previous.sum() - torch.sigmoid(group_logits)[0] * (previous * front).sum()
    assert torch.allclose(inventory[0], expected, atol=1e-12)
    assert torch.equal(updated[1:], previous[1:])
    next_front = predicted_front_weight(labels, sites, updated, inlet, connected)
    assert next_front[1] > 0 and next_front[2] == 0
    sparse = SparseFrontGeometry.from_initial(labels, sites, inlet, connected)
    assert torch.equal(sparse.weight(previous), front)
    assert torch.equal(sparse.weight(updated), next_front)
    assert torch.all((updated >= 0) & (updated <= previous))
    updated.sum().backward()
    assert previous.grad is not None and torch.isfinite(previous.grad).all()
    assert group_logits.grad is not None and torch.isfinite(group_logits.grad).all()

    partial_front = torch.tensor([0.2, 0.7, 0.0], dtype=torch.float64,
                                 requires_grad=True)
    partial, partial_inventory = front_weighted_budget_step(
        previous.detach(), partial_front, site_logits, groups,
        group_logits.detach()
    )
    expected_partial = (previous.detach().sum()
                        - torch.sigmoid(group_logits.detach())[0]
                        * (previous.detach() * partial_front).sum())
    assert torch.allclose(partial_inventory[0], expected_partial, atol=1e-12)
    assert torch.all((partial >= 0) & (partial <= previous.detach()))
    partial[0].backward()
    assert partial_front.grad is not None and torch.isfinite(partial_front.grad).all()

    # Full exposure recovers the existing centred-budget identity for unequal
    # local propensities; local gradients are then nonzero.
    front2 = torch.ones(3, dtype=torch.float64, requires_grad=True)
    later, later_inventory = front_weighted_budget_step(
        previous.detach(), front2, site_logits, groups, group_logits.detach()
    )
    assert torch.allclose(later_inventory[0],
                          previous.detach().sum() * (1 - torch.sigmoid(group_logits.detach())[0]),
                          atol=1e-12)
    later[0].backward()
    assert site_logits.grad is not None and torch.isfinite(site_logits.grad).all()
    assert front2.grad is not None and torch.isfinite(front2.grad).all()

    # Boundary handling and connected-pore masking agree with the dense
    # six-neighbour reference on a non-cubic volume.
    varied = torch.tensor([
        [[2, 3, 1, 3], [3, 3, 2, 3], [3, 3, 3, 3]],
        [[3, 3, 3, 3], [3, 2, 3, 3], [3, 3, 3, 3]],
    ], dtype=torch.uint8)
    varied_connected = varied >= 2
    varied_inlet = torch.zeros_like(varied_connected)
    varied_inlet[:, :, 0] = varied_connected[:, :, 0]
    varied_sites = torch.nonzero(varied.reshape(-1) >= 3).flatten()
    varied_previous = torch.linspace(0.1, 0.9, varied_sites.numel(),
                                     dtype=torch.float64)
    dense_weight = predicted_front_weight(
        varied, varied_sites, varied_previous, varied_inlet, varied_connected
    )
    sparse_weight = SparseFrontGeometry.from_initial(
        varied, varied_sites, varied_inlet, varied_connected
    ).weight(varied_previous)
    assert torch.allclose(sparse_weight, dense_weight, atol=0, rtol=0)


if __name__ == "__main__":
    _synthetic_self_test()
    print("softfront_budget_candidate synthetic checks: PASS")
