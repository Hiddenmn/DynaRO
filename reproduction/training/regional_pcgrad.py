"""Two-task PCGrad comparator for remaining-oil surrogate training.

The primary state loss and the weighted paired-response loss are treated as
two tasks.  When their global gradients conflict, each task gradient is
projected against the other task's original gradient before the two projected
vectors are summed.  This is a matched comparator for StateGuard-RO: it is
symmetric and has no auxiliary-to-primary norm cap.

This module performs no data access, optimizer step, training, evaluation, or
checkpoint selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import torch


GradientBuffer = list[torch.Tensor | None]


@dataclass(frozen=True)
class PCGradStats:
    """Scalar diagnostics for one global two-task projection."""

    primary_norm: float
    weighted_auxiliary_norm: float
    projected_primary_norm: float
    projected_auxiliary_norm: float
    combined_norm: float
    cosine_before: float
    dot_before: float
    conflict_projection_applied: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def new_gradient_buffer(
    parameters: Iterable[torch.nn.Parameter],
) -> GradientBuffer:
    """Create an empty detached accumulator aligned with ``parameters``."""
    return [None for _ in parameters]


def accumulate_current_gradients(
    parameters: Sequence[torch.nn.Parameter], buffer: GradientBuffer
) -> None:
    """Add current parameter gradients to one detached task buffer."""
    if len(parameters) != len(buffer):
        raise ValueError("Parameter and gradient-buffer lengths differ")
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        if gradient is None:
            continue
        if (
            gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or gradient.dtype != parameter.dtype
            or not bool(torch.isfinite(gradient).all())
        ):
            raise FloatingPointError("Invalid parameter gradient")
        value = gradient.detach()
        if buffer[index] is None:
            buffer[index] = value.clone()
        else:
            existing = buffer[index]
            if existing is None:  # pragma: no cover
                raise AssertionError("Unreachable empty gradient buffer")
            existing.add_(value)


def _validate_buffers(
    parameters: Sequence[torch.nn.Parameter],
    primary: GradientBuffer,
    auxiliary: GradientBuffer,
) -> None:
    if (
        not parameters
        or len(parameters) != len(primary)
        or len(primary) != len(auxiliary)
        or not any(value is not None for value in primary)
        or not any(value is not None for value in auxiliary)
    ):
        raise ValueError("Two nonempty task-gradient buffers are required")
    for parameter, first, second in zip(parameters, primary, auxiliary):
        for gradient in (first, second):
            if gradient is None:
                continue
            if (
                gradient.shape != parameter.shape
                or gradient.device != parameter.device
                or gradient.dtype != parameter.dtype
                or not bool(torch.isfinite(gradient).all())
            ):
                raise FloatingPointError("Invalid buffered gradient")


def assign_pcgrad_gradient(
    parameters: Sequence[torch.nn.Parameter],
    primary: GradientBuffer,
    auxiliary: GradientBuffer,
    *,
    auxiliary_weight: float,
    epsilon: float = 1e-12,
) -> PCGradStats:
    """Assign the standard symmetric two-task PCGrad sum.

    The auxiliary task is multiplied by ``auxiliary_weight`` before conflict
    detection.  If the global dot product is negative, the primary gradient is
    projected against the weighted auxiliary gradient and the weighted
    auxiliary gradient is projected against the original primary gradient.
    Both projections use the original task vectors, as required for a matched
    two-task PCGrad comparator.
    """
    parameters = tuple(parameters)
    _validate_buffers(parameters, primary, auxiliary)
    if (
        not math.isfinite(auxiliary_weight)
        or auxiliary_weight <= 0
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("Projection constants must be finite and positive")

    reference = next(
        value for value in (*primary, *auxiliary) if value is not None
    )
    zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
    primary_sq = zero.clone()
    auxiliary_sq = zero.clone()
    dot = zero.clone()
    for first, second in zip(primary, auxiliary):
        if first is not None:
            primary_sq = primary_sq + first.square().sum()
        if second is not None:
            weighted = auxiliary_weight * second
            auxiliary_sq = auxiliary_sq + weighted.square().sum()
            if first is not None:
                dot = dot + (first * weighted).sum()

    conflict = bool(
        dot < 0 and primary_sq > epsilon and auxiliary_sq > epsilon
    )
    primary_coefficient = (
        dot / auxiliary_sq.clamp_min(epsilon) if conflict else zero
    )
    auxiliary_coefficient = (
        dot / primary_sq.clamp_min(epsilon) if conflict else zero
    )
    projected_primary_sq = zero.clone()
    projected_auxiliary_sq = zero.clone()
    combined_sq = zero.clone()
    for parameter, first, second in zip(parameters, primary, auxiliary):
        if first is None and second is None:
            parameter.grad = None
            continue
        primary_value = (
            torch.zeros_like(parameter) if first is None else first
        )
        auxiliary_value = (
            torch.zeros_like(parameter)
            if second is None
            else auxiliary_weight * second
        )
        if conflict:
            projected_primary = (
                primary_value - primary_coefficient * auxiliary_value
            )
            projected_auxiliary = (
                auxiliary_value - auxiliary_coefficient * primary_value
            )
        else:
            projected_primary = primary_value
            projected_auxiliary = auxiliary_value
        combined = projected_primary + projected_auxiliary
        if not bool(torch.isfinite(combined).all()):
            raise FloatingPointError("PCGrad produced a non-finite gradient")
        parameter.grad = combined.detach().clone()
        projected_primary_sq += projected_primary.square().sum()
        projected_auxiliary_sq += projected_auxiliary.square().sum()
        combined_sq += combined.square().sum()

    primary_norm = float(primary_sq.sqrt().detach().cpu())
    auxiliary_norm = float(auxiliary_sq.sqrt().detach().cpu())
    denominator = primary_norm * auxiliary_norm
    cosine = float(dot.detach().cpu()) / denominator if denominator > 0 else 0.0
    stats = PCGradStats(
        primary_norm=primary_norm,
        weighted_auxiliary_norm=auxiliary_norm,
        projected_primary_norm=float(
            projected_primary_sq.sqrt().detach().cpu()
        ),
        projected_auxiliary_norm=float(
            projected_auxiliary_sq.sqrt().detach().cpu()
        ),
        combined_norm=float(combined_sq.sqrt().detach().cpu()),
        cosine_before=max(-1.0, min(1.0, cosine)),
        dot_before=float(dot.detach().cpu()),
        conflict_projection_applied=conflict,
    )
    if any(
        not math.isfinite(value)
        for key, value in stats.to_dict().items()
        if key != "conflict_projection_applied"
    ):
        raise FloatingPointError("Non-finite PCGrad diagnostics")
    return stats


def _synthetic_self_test() -> None:
    # Both original task vectors are changed under a conflict.
    parameter = torch.nn.Parameter(torch.zeros(2))
    stats = assign_pcgrad_gradient(
        [parameter],
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([-1.0, 1.0])],
        auxiliary_weight=1.0,
    )
    assert stats.conflict_projection_applied
    assert torch.allclose(parameter.grad, torch.tensor([0.5, 1.5]))

    # Aligned gradients are summed without modification.
    stats = assign_pcgrad_gradient(
        [parameter],
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.5, 0.0])],
        auxiliary_weight=1.0,
    )
    assert not stats.conflict_projection_applied
    assert torch.allclose(parameter.grad, torch.tensor([1.5, 0.0]))

    # Unlike StateGuard-RO, orthogonal auxiliary magnitude is not capped.
    stats = assign_pcgrad_gradient(
        [parameter],
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, 10.0])],
        auxiliary_weight=1.0,
    )
    assert not stats.conflict_projection_applied
    assert torch.allclose(parameter.grad, torch.tensor([1.0, 10.0]))
    assert math.isclose(stats.projected_auxiliary_norm, 10.0)

    # Accumulation is detached and additive.
    parameter.grad = torch.tensor([0.25, -0.5])
    buffer = new_gradient_buffer([parameter])
    accumulate_current_gradients([parameter], buffer)
    parameter.grad.add_(torch.tensor([0.75, 0.5]))
    accumulate_current_gradients([parameter], buffer)
    assert buffer[0] is not None
    assert torch.allclose(buffer[0], torch.tensor([1.25, -0.5]))


if __name__ == "__main__":
    _synthetic_self_test()
    print("regional_pcgrad synthetic self-test: PASS")
