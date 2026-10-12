"""Primary-state-safe auxiliary-gradient projection for remaining-oil learning.

The three-dimensional remaining-oil state objective is the protected primary
objective.  The paired signed action-response objective is auxiliary.  Before
an optimizer update, the weighted auxiliary gradient is projected into

    C(g_s) = {u : <u, g_s> >= 0 and ||u|| <= rho ||g_s||},

where ``g_s`` is the accumulated state gradient.  The raw gradient passed to
the optimizer is ``g_s + u``.  This module performs no data access, optimizer
step, training, evaluation, or checkpoint selection.

The projection is a domain adaptation of established gradient-surgery ideas;
it is not presented as a new general-purpose multi-task optimizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import torch


GradientBuffer = list[torch.Tensor | None]


@dataclass(frozen=True)
class StateGuardStats:
    """Scalar diagnostics for one accumulated optimizer update."""

    primary_norm: float
    weighted_auxiliary_norm: float
    guarded_auxiliary_norm: float
    combined_norm: float
    cosine_before: float
    primary_auxiliary_dot_before: float
    primary_auxiliary_dot_after: float
    primary_combined_dot: float
    projection_applied: bool
    norm_cap_applied: bool
    auxiliary_scale_after_projection: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def new_gradient_buffer(parameters: Iterable[torch.nn.Parameter]) -> GradientBuffer:
    """Create an empty detached accumulator aligned with ``parameters``."""
    return [None for _ in parameters]


def accumulate_current_gradients(
    parameters: Sequence[torch.nn.Parameter], buffer: GradientBuffer,
) -> None:
    """Add current ``.grad`` tensors to a detached accumulation buffer."""
    if len(parameters) != len(buffer):
        raise ValueError("Parameter and gradient-buffer lengths differ")
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        if gradient is None:
            continue
        if (gradient.shape != parameter.shape or gradient.device != parameter.device
                or gradient.dtype != parameter.dtype
                or not bool(torch.isfinite(gradient).all())):
            raise FloatingPointError("Invalid parameter gradient")
        detached = gradient.detach()
        if buffer[index] is None:
            buffer[index] = detached.clone()
        else:
            existing = buffer[index]
            if existing is None:  # pragma: no cover - narrows the static type.
                raise AssertionError("Unreachable empty gradient buffer")
            existing.add_(detached)


def _validate_buffers(
    parameters: Sequence[torch.nn.Parameter],
    primary: GradientBuffer,
    auxiliary: GradientBuffer,
) -> None:
    if not parameters or len(parameters) != len(primary) or len(primary) != len(auxiliary):
        raise ValueError("Gradient buffers must align with nonempty parameters")
    if not any(value is not None for value in primary):
        raise ValueError("Primary gradient buffer is empty")
    for parameter, first, second in zip(parameters, primary, auxiliary):
        for gradient in (first, second):
            if gradient is None:
                continue
            if (gradient.shape != parameter.shape
                    or gradient.device != parameter.device
                    or gradient.dtype != parameter.dtype
                    or not bool(torch.isfinite(gradient).all())):
                raise FloatingPointError("Invalid buffered gradient")


def assign_state_guarded_gradient(
    parameters: Sequence[torch.nn.Parameter],
    primary: GradientBuffer,
    auxiliary: GradientBuffer,
    *,
    auxiliary_weight: float,
    max_auxiliary_to_primary_norm: float = 1.0,
    epsilon: float = 1e-12,
) -> StateGuardStats:
    """Project and assign one state-protected combined gradient.

    ``auxiliary_weight`` is applied before the projection.  A negative global
    dot product removes only the component of the auxiliary gradient opposing
    the primary gradient.  The remaining auxiliary vector is then capped to a
    fixed fraction of the primary norm.  The operation is global across all
    trainable parameters, not layer-wise.
    """
    parameters = tuple(parameters)
    _validate_buffers(parameters, primary, auxiliary)
    if (not math.isfinite(auxiliary_weight) or auxiliary_weight <= 0
            or not math.isfinite(max_auxiliary_to_primary_norm)
            or max_auxiliary_to_primary_norm <= 0
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("Projection constants must be finite and positive")

    reference = next(
        value for value in (*primary, *auxiliary) if value is not None
    )
    scalar_zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
    primary_sq = scalar_zero.clone()
    auxiliary_sq = scalar_zero.clone()
    dot_before = scalar_zero.clone()
    for first, second in zip(primary, auxiliary):
        if first is not None:
            primary_sq = primary_sq + first.square().sum()
        if second is not None:
            weighted = auxiliary_weight * second
            auxiliary_sq = auxiliary_sq + weighted.square().sum()
            if first is not None:
                dot_before = dot_before + (first * weighted).sum()

    primary_norm_tensor = primary_sq.sqrt()
    auxiliary_norm_tensor = auxiliary_sq.sqrt()
    projection_applied = bool(dot_before < 0 and primary_sq > epsilon)
    coefficient = (
        dot_before / primary_sq.clamp_min(epsilon)
        if projection_applied else scalar_zero
    )

    guarded: GradientBuffer = []
    guarded_sq = scalar_zero.clone()
    for first, second in zip(primary, auxiliary):
        if second is None:
            guarded.append(None)
            continue
        value = auxiliary_weight * second
        if projection_applied and first is not None:
            value = value - coefficient * first
        guarded.append(value)
        guarded_sq = guarded_sq + value.square().sum()

    guarded_norm_before_cap = guarded_sq.sqrt()
    maximum_norm = max_auxiliary_to_primary_norm * primary_norm_tensor
    if bool(guarded_norm_before_cap > maximum_norm) and bool(
        guarded_norm_before_cap > epsilon
    ):
        scale_tensor = maximum_norm / guarded_norm_before_cap
        norm_cap_applied = True
    else:
        scale_tensor = torch.ones_like(guarded_norm_before_cap)
        norm_cap_applied = False

    dot_after = scalar_zero.clone()
    combined_sq = scalar_zero.clone()
    primary_combined_dot = scalar_zero.clone()
    for parameter, first, value in zip(parameters, primary, guarded):
        guarded_value = None if value is None else scale_tensor * value
        if first is not None and guarded_value is not None:
            dot_after = dot_after + (first * guarded_value).sum()
        if first is None and guarded_value is None:
            parameter.grad = None
            continue
        if first is None:
            combined = guarded_value
        elif guarded_value is None:
            combined = first
        else:
            combined = first + guarded_value
        if combined is None:  # pragma: no cover - all cases handled above.
            raise AssertionError("Unreachable empty combined gradient")
        parameter.grad = combined.detach().clone()
        combined_sq = combined_sq + combined.square().sum()
        if first is not None:
            primary_combined_dot = primary_combined_dot + (first * combined).sum()

    tolerance = 5e-6 * max(1.0, float(primary_sq.detach().cpu()))
    dot_after_value = float(dot_after.detach().cpu())
    primary_combined_value = float(primary_combined_dot.detach().cpu())
    primary_sq_value = float(primary_sq.detach().cpu())
    if dot_after_value < -tolerance:
        raise FloatingPointError("Projected auxiliary gradient still conflicts")
    if primary_combined_value + tolerance < primary_sq_value:
        raise FloatingPointError("Combined raw gradient weakens primary descent")

    primary_norm = float(primary_norm_tensor.detach().cpu())
    auxiliary_norm = float(auxiliary_norm_tensor.detach().cpu())
    denominator = primary_norm * auxiliary_norm
    cosine = (
        float(dot_before.detach().cpu()) / denominator
        if denominator > 0 else 0.0
    )
    stats = StateGuardStats(
        primary_norm=primary_norm,
        weighted_auxiliary_norm=auxiliary_norm,
        guarded_auxiliary_norm=float(
            (scale_tensor * guarded_norm_before_cap).detach().cpu()
        ),
        combined_norm=float(combined_sq.sqrt().detach().cpu()),
        cosine_before=max(-1.0, min(1.0, cosine)),
        primary_auxiliary_dot_before=float(dot_before.detach().cpu()),
        primary_auxiliary_dot_after=dot_after_value,
        primary_combined_dot=primary_combined_value,
        projection_applied=projection_applied,
        norm_cap_applied=norm_cap_applied,
        auxiliary_scale_after_projection=float(scale_tensor.detach().cpu()),
    )
    if any(
        not math.isfinite(value)
        for key, value in stats.to_dict().items()
        if key not in ("projection_applied", "norm_cap_applied")
    ):
        raise FloatingPointError("Non-finite state-guard diagnostics")
    return stats


def _synthetic_self_test() -> None:
    # Conflicting component is removed; the orthogonal component is retained.
    parameter = torch.nn.Parameter(torch.zeros(2))
    primary = [torch.tensor([1.0, 0.0])]
    auxiliary = [torch.tensor([-1.0, 1.0])]
    stats = assign_state_guarded_gradient(
        [parameter], primary, auxiliary, auxiliary_weight=1.0,
    )
    assert stats.projection_applied and not stats.norm_cap_applied
    assert torch.allclose(parameter.grad, torch.tensor([1.0, 1.0]))
    assert abs(stats.primary_auxiliary_dot_after) < 1e-7

    # A large non-conflicting orthogonal auxiliary update is norm-capped.
    stats = assign_state_guarded_gradient(
        [parameter], primary, [torch.tensor([0.0, 10.0])],
        auxiliary_weight=1.0,
    )
    assert not stats.projection_applied and stats.norm_cap_applied
    assert torch.allclose(parameter.grad, torch.tensor([1.0, 1.0]))
    assert math.isclose(stats.guarded_auxiliary_norm, 1.0, rel_tol=1e-6)

    # An aligned, sub-radius auxiliary gradient is unchanged.
    stats = assign_state_guarded_gradient(
        [parameter], primary, [torch.tensor([0.5, 0.0])],
        auxiliary_weight=1.0,
    )
    assert not stats.projection_applied and not stats.norm_cap_applied
    assert torch.allclose(parameter.grad, torch.tensor([1.5, 0.0]))

    # Detached accumulation does not retain the source graph or mutate values.
    parameter.grad = torch.tensor([0.25, -0.5])
    buffer = new_gradient_buffer([parameter])
    accumulate_current_gradients([parameter], buffer)
    parameter.grad.add_(torch.tensor([0.75, 0.5]))
    accumulate_current_gradients([parameter], buffer)
    assert buffer[0] is not None
    assert torch.allclose(buffer[0], torch.tensor([1.25, -0.5]))


if __name__ == "__main__":
    _synthetic_self_test()
    print("regional_state_guard synthetic self-test: PASS")
