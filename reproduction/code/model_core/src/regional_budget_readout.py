"""Matched regional-budget and direct voxel readouts; synthetic checks only.

This is a candidate algebraic component, not a fitted remaining-oil model.
Each group can represent one fixed 3-D cell and one initial oil class.  A
separate causal network would predict the group's mobilization fraction from
the initial geometry and available action prefix.  The voxel propensities may
use initial geometry and the network state, but never future true labels.
"""

from __future__ import annotations

import torch


def initial_oil_groups(
    initial_labels: torch.Tensor, *, cell_width: int = 16
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Map initial 3-D oil codes 3..7 to fixed cell/class group IDs.

    Returns flat oil-site indices, corresponding group IDs and total group
    count.  Codes 0/1 are solid, 2 is initial water.  This function consumes
    only the initial state; target or future labels are never an input.
    """
    if initial_labels.ndim != 3 or initial_labels.dtype not in (
        torch.uint8, torch.int16, torch.int32, torch.int64
    ):
        raise ValueError("Expected one 3-D integer initial-label volume")
    if type(cell_width) is not int or cell_width < 1:
        raise ValueError("cell_width must be a positive integer")
    z_size, y_size, x_size = initial_labels.shape
    if min(initial_labels.shape) < 1 or any(
        length % cell_width for length in initial_labels.shape
    ):
        raise ValueError("Cell width must divide all three spatial axes")
    if bool(torch.any((initial_labels < 0) | (initial_labels > 7))):
        raise ValueError("Unexpected initial-label code")
    flat = initial_labels.reshape(-1)
    site_index = torch.nonzero(flat >= 3, as_tuple=False).flatten()
    z = site_index // (y_size * x_size)
    y = (site_index // x_size) % y_size
    x = site_index % x_size
    cells_y, cells_x = y_size // cell_width, x_size // cell_width
    cell_id = ((z // cell_width) * cells_y + y // cell_width) * cells_x + x // cell_width
    group_id = 5 * cell_id + flat[site_index].long() - 3
    n_groups = (z_size // cell_width) * cells_y * cells_x * 5
    return site_index, group_id, n_groups


def full_so_from_sites(
    initial_labels: torch.Tensor, site_index: torch.Tensor, remaining: torch.Tensor
) -> torch.Tensor:
    """Assemble a complete 3-D expected So field on initial oil support."""
    if initial_labels.ndim != 3 or site_index.ndim != 1:
        raise ValueError("Expected 3-D labels and flat site indices")
    if (remaining.ndim != 1 or remaining.numel() != site_index.numel()
            or remaining.device != initial_labels.device
            or site_index.device != initial_labels.device
            or site_index.dtype != torch.long
            or not remaining.is_floating_point()):
        raise ValueError("Remaining oil and site indices must match labels")
    if not bool(torch.isfinite(remaining).all()
                and torch.all((remaining >= 0) & (remaining <= 1))):
        raise ValueError("Remaining-oil site values must be finite and in [0, 1]")
    if bool(torch.any((site_index < 0) | (site_index >= initial_labels.numel()))):
        raise ValueError("Oil-site index outside the initial volume")
    if bool(torch.any((initial_labels.reshape(-1)[site_index] < 3))):
        raise ValueError("Oil-site index includes noninitial oil")
    if site_index.numel() != torch.unique(site_index).numel():
        raise ValueError("Oil-site indices must be unique")
    return remaining.new_zeros(initial_labels.numel()).scatter(
        0, site_index, remaining
    ).reshape(initial_labels.shape)


def _validate_sites(
    previous: torch.Tensor,
    propensity_logits: torch.Tensor,
    group_id: torch.Tensor,
    group_values: torch.Tensor,
) -> None:
    if previous.ndim != 1 or propensity_logits.shape != previous.shape:
        raise ValueError("Site tensors must have one matching flat shape")
    if group_id.shape != previous.shape or group_id.dtype != torch.long:
        raise ValueError("group_id must be a matching int64 vector")
    if group_values.ndim != 1 or group_values.numel() == 0:
        raise ValueError("Group values must be a nonempty vector")
    if (previous.device != propensity_logits.device
            or previous.device != group_id.device
            or previous.device != group_values.device):
        raise ValueError("All tensors must share a device")
    if (previous.dtype != propensity_logits.dtype
            or previous.dtype != group_values.dtype
            or not previous.is_floating_point()):
        raise ValueError("Floating tensors must share a floating dtype")
    if not bool(torch.isfinite(previous).all()
                and torch.isfinite(propensity_logits).all()
                and torch.isfinite(group_values).all()):
        raise ValueError("All floating tensors must be finite")
    if bool(torch.any((previous < 0) | (previous > 1))):
        raise ValueError("Previous oil occupancy must lie in [0, 1]")
    if bool(torch.any((group_id < 0) | (group_id >= group_values.numel()))):
        raise ValueError("group_id outside the group range")


def _inventory(remaining: torch.Tensor, group_id: torch.Tensor, n_groups: int) -> torch.Tensor:
    return remaining.new_zeros(n_groups).scatter_add(0, group_id, remaining)


def _eligible_sites(previous: torch.Tensor, eligible: torch.Tensor | None) -> torch.Tensor:
    if eligible is None:
        return torch.ones_like(previous, dtype=torch.bool)
    if eligible.shape != previous.shape or eligible.device != previous.device:
        raise ValueError("Eligible sites must match previous oil occupancy")
    if eligible.dtype != torch.bool:
        raise ValueError("Eligible sites must be a boolean initial-geometry mask")
    return eligible


def allocate_remaining_oil(
    previous: torch.Tensor,
    propensity_logits: torch.Tensor,
    group_id: torch.Tensor,
    mobilization_fraction: torch.Tensor,
    *,
    eligible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded voxel oil and exact per-group remaining-oil inventory.

    All site tensors are flat and have the same length.  ``group_id`` maps an
    initial-oil site to one of ``len(mobilization_fraction)`` cell/class
    groups. ``previous`` is the previous expected oil occupancy in [0, 1],
    and each mobilization fraction is in [0, 1].  If provided, ``eligible``
    must be computed from the initial geometry and elapsed step, not a true
    future front.  The fraction then applies only to *eligible* inventory;
    ineligible oil stays unchanged.  The group ledger closes up to floating-
    point summation error, with no cross-group oil transfer.
    """
    _validate_sites(previous, propensity_logits, group_id, mobilization_fraction)
    if bool(torch.any((mobilization_fraction < 0) | (mobilization_fraction > 1))):
        raise ValueError("Mobilization fractions must lie in [0, 1]")
    eligible = _eligible_sites(previous, eligible)

    n_groups = mobilization_fraction.numel()
    eligible_previous = previous * eligible
    inventory = _inventory(eligible_previous, group_id, n_groups)
    propensity = torch.sigmoid(propensity_logits)
    weighted_sum = _inventory(eligible_previous * propensity, group_id, n_groups)
    mean_propensity = torch.where(
        inventory > 0,
        weighted_sum / inventory.clamp_min(torch.finfo(previous.dtype).tiny),
        torch.zeros_like(inventory),
    )

    fraction = mobilization_fraction[group_id]
    centered = propensity - mean_propensity[group_id]
    # Its inventory-weighted mean is exactly zero.  Since both propensity and
    # mean_propensity lie in [0, 1], h = s + s(1-s) * centered lies in [0, 1].
    hazard = eligible * (fraction + fraction * (1 - fraction) * centered)
    remaining = previous * (1 - hazard)
    remaining_inventory = _inventory(remaining, group_id, n_groups)
    return remaining, remaining_inventory


def readout_from_shared_logits(
    previous: torch.Tensor,
    propensity_logits: torch.Tensor,
    group_id: torch.Tensor,
    mobilization_logits: torch.Tensor,
    *,
    mode: str,
    eligible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Three parameter-free readouts of the same site/group neural outputs.

    ``budget`` enforces each cell/class predicted inventory. ``uncentered``
    changes only the inventory-weighted centering term to the fixed 0.5; it
    isolates the budget constraint. ``direct`` is an additive-logit voxel
    readout. None is a trained model or the historical DirectState-3D model.
    A future fair comparison must share the entire upstream network, losses,
    available action prefix, data and optimization budget across these modes.
    """
    if mode not in ("budget", "uncentered", "direct"):
        raise ValueError("Unknown readout mode")
    _validate_sites(previous, propensity_logits, group_id, mobilization_logits)
    eligible = _eligible_sites(previous, eligible)
    fraction = torch.sigmoid(mobilization_logits)
    if mode == "budget":
        return allocate_remaining_oil(
            previous, propensity_logits, group_id, fraction, eligible=eligible
        )
    if mode == "uncentered":
        propensity = torch.sigmoid(propensity_logits)
        site_fraction = fraction[group_id]
        hazard = eligible * (site_fraction + site_fraction * (1 - site_fraction)
                             * (propensity - 0.5))
    else:
        hazard = eligible * torch.sigmoid(
            propensity_logits + mobilization_logits[group_id]
        )
    remaining = previous * (1 - hazard)
    return remaining, _inventory(remaining, group_id, mobilization_logits.numel())


def _synthetic_self_test() -> None:
    dtype = torch.float64
    previous = torch.tensor([1.0, 0.5, 0.25, 1.0, 0.0, 0.8], dtype=dtype)
    group_id = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    logits = torch.tensor([-2.0, 1.0, 2.0, 0.3, 4.0, -1.0],
                          dtype=dtype, requires_grad=True)
    fraction = torch.tensor([0.3, 0.8, 0.4, 0.7],
                            dtype=dtype, requires_grad=True)
    remaining, totals = allocate_remaining_oil(previous, logits, group_id, fraction)
    old_totals = previous.new_zeros(4).scatter_add(0, group_id, previous)
    assert torch.allclose(totals, old_totals * (1 - fraction), atol=1e-14)
    assert bool(torch.all((remaining >= 0) & (remaining <= previous)))
    assert totals[3] == 0  # Empty cell/class group cannot create oil.

    changed_logits = logits.detach().clone()
    changed_logits[0], changed_logits[1] = logits.detach()[1], logits.detach()[0]
    changed, changed_totals = allocate_remaining_oil(
        previous, changed_logits, group_id, fraction.detach()
    )
    assert not torch.allclose(remaining[:3].detach(), changed[:3])
    assert torch.allclose(totals.detach(), changed_totals, atol=1e-14)

    weighted_loss = (remaining * torch.arange(1, 7, dtype=dtype)).sum()
    weighted_loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert fraction.grad is not None and torch.isfinite(fraction.grad).all()
    assert logits.grad[:3].abs().sum() > 0
    assert fraction.grad[:3].abs().sum() > 0

    # Same upstream tensors; the centering ablation and direct readout are
    # bounded but do not silently inherit the regional inventory identity.
    group_logits = torch.tensor([-0.7, 0.8, -0.2, 0.5], dtype=dtype,
                                requires_grad=True)
    common_logits = logits.detach().clone().requires_grad_()
    paired = {
        mode: readout_from_shared_logits(previous, common_logits, group_id,
                                        group_logits, mode=mode)
        for mode in ("budget", "uncentered", "direct")
    }
    target = old_totals * (1 - torch.sigmoid(group_logits))
    assert torch.allclose(paired["budget"][1], target, atol=1e-14)
    assert not torch.allclose(paired["uncentered"][1], target, atol=1e-5)
    assert not torch.allclose(paired["direct"][1], target, atol=1e-5)
    for state, inventory in paired.values():
        assert bool(torch.all((state >= 0) & (state <= previous)))
        assert inventory[3] == 0
    sum((state * torch.arange(1, 7, dtype=dtype)).sum()
        for state, _ in paired.values()).backward()
    assert torch.isfinite(common_logits.grad).all()
    assert torch.isfinite(group_logits.grad).all()
    assert common_logits.grad.abs().sum() > 0
    assert group_logits.grad.abs().sum() > 0

    reachable = torch.tensor([True, False, True, True, False, False])
    for mode in ("budget", "uncentered", "direct"):
        restricted, restricted_totals = readout_from_shared_logits(
            previous, logits.detach(), group_id, group_logits.detach(),
            mode=mode, eligible=reachable
        )
        assert torch.equal(restricted[~reachable], previous[~reachable])
        assert bool(torch.all((restricted >= 0) & (restricted <= previous)))
        if mode == "budget":
            active = _inventory(previous * reachable, group_id, 4)
            expected_restricted = old_totals - active * torch.sigmoid(group_logits.detach())
            assert torch.allclose(restricted_totals, expected_restricted, atol=1e-14)

    label_volume = torch.zeros((4, 4, 4), dtype=torch.uint8)
    label_volume[0, 0, 0] = 3
    label_volume[0, 0, 1] = 4
    label_volume[3, 3, 3] = 7
    label_volume[1, 1, 1] = 2
    sites, groups, count = initial_oil_groups(label_volume, cell_width=2)
    assert count == 8 * 5
    assert torch.equal(groups, torch.tensor([0, 1, 39]))
    values = torch.tensor([0.9, 0.8, 0.4], dtype=dtype, requires_grad=True)
    so = full_so_from_sites(label_volume, sites, values)
    assert so.shape == (4, 4, 4)
    assert so[0, 0, 0] == 0.9 and so[0, 0, 1] == 0.8
    assert so[3, 3, 3] == 0.4 and so[1, 1, 1] == 0
    so.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))

    # A larger, deterministic float32 case catches group reduction and
    # permutation mistakes that the hand-built three-site example may miss.
    generator = torch.Generator().manual_seed(20260920)
    many_groups = torch.randint(0, 5 * 16, (4096,), generator=generator)
    many_previous = torch.rand(4096, generator=generator)
    many_local = torch.randn(4096, generator=generator, requires_grad=True)
    many_budget = torch.randn(5 * 16, generator=generator, requires_grad=True)
    many_remaining, many_totals = readout_from_shared_logits(
        many_previous, many_local, many_groups, many_budget, mode="budget"
    )
    expected = _inventory(many_previous, many_groups, 5 * 16) * (
        1 - torch.sigmoid(many_budget)
    )
    assert torch.allclose(many_totals, expected, atol=5e-6, rtol=1e-6)
    assert bool(torch.all((many_remaining >= 0) & (many_remaining <= many_previous)))
    permutation = torch.randperm(4096, generator=generator)
    permuted, permuted_totals = readout_from_shared_logits(
        many_previous[permutation], many_local[permutation],
        many_groups[permutation], many_budget, mode="budget"
    )
    assert torch.allclose(permuted_totals, many_totals, atol=5e-6, rtol=1e-6)
    assert torch.allclose(permuted, many_remaining[permutation], atol=5e-7, rtol=1e-6)
    (many_remaining * torch.arange(4096, dtype=many_remaining.dtype)).sum().backward()
    assert torch.isfinite(many_local.grad).all()
    assert torch.isfinite(many_budget.grad).all()


if __name__ == "__main__":
    _synthetic_self_test()
    print("regional_budget_readout synthetic self-test: PASS")
