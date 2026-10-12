"""Cosine-gated auxiliary-gradient comparator for the remaining-oil task.

This implements the weighted cosine variant of Du et al., *Adapting Auxiliary
Losses Using Gradient Similarity* (arXiv:1812.02224): the state gradient is
primary and the weighted paired-response gradient is admitted with coefficient
max(0, cosine(primary, auxiliary)).  Unlike StateGuard-RO, a conflicting
auxiliary gradient is suppressed in full and there is no auxiliary norm cap.
This module reads no data and performs no optimizer step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch

from regional_state_guard import GradientBuffer


@dataclass(frozen=True)
class DuGradCosStats:
    primary_norm: float
    weighted_auxiliary_norm: float
    cosine_before: float
    cosine_gate: float
    gated_auxiliary_norm: float
    combined_norm: float
    primary_combined_dot: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def assign_du_gradcos_gradient(
    parameters: Sequence[torch.nn.Parameter],
    primary: GradientBuffer,
    auxiliary: GradientBuffer,
    *,
    auxiliary_weight: float,
    epsilon: float = 1e-12,
) -> DuGradCosStats:
    """Assign one global primary + cosine-gated auxiliary gradient."""
    parameters = tuple(parameters)
    if not parameters or len(parameters) != len(primary) or len(primary) != len(auxiliary):
        raise ValueError("Gradient buffers must align with nonempty parameters")
    if not any(value is not None for value in primary):
        raise ValueError("Primary gradient buffer is empty")
    if not math.isfinite(auxiliary_weight) or auxiliary_weight <= 0:
        raise ValueError("Auxiliary weight must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Epsilon must be finite and positive")

    reference = next(value for value in (*primary, *auxiliary) if value is not None)
    zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
    primary_sq = zero.clone()
    auxiliary_sq = zero.clone()
    dot = zero.clone()
    for parameter, state, response in zip(parameters, primary, auxiliary):
        for value in (state, response):
            if value is None:
                continue
            if (value.shape != parameter.shape or value.device != parameter.device
                    or value.dtype != parameter.dtype
                    or not bool(torch.isfinite(value).all())):
                raise FloatingPointError("Invalid buffered gradient")
        if state is not None:
            primary_sq = primary_sq + state.square().sum()
        if response is not None:
            weighted = auxiliary_weight * response
            auxiliary_sq = auxiliary_sq + weighted.square().sum()
            if state is not None:
                dot = dot + (state * weighted).sum()

    primary_norm = primary_sq.sqrt()
    auxiliary_norm = auxiliary_sq.sqrt()
    denominator = primary_norm * auxiliary_norm
    cosine = (
        (dot / denominator).clamp(-1.0, 1.0)
        if bool(denominator > epsilon)
        else zero
    )
    gate = cosine.clamp_min(0.0)
    combined_sq = zero.clone()
    primary_combined_dot = zero.clone()
    for parameter, state, response in zip(parameters, primary, auxiliary):
        gated = None if response is None else gate * auxiliary_weight * response
        if state is None and gated is None:
            parameter.grad = None
            continue
        combined = gated if state is None else state if gated is None else state + gated
        if combined is None:  # pragma: no cover - all cases handled above.
            raise AssertionError("Unreachable empty gradient")
        parameter.grad = combined.detach().clone()
        combined_sq = combined_sq + combined.square().sum()
        if state is not None:
            primary_combined_dot = primary_combined_dot + (state * combined).sum()

    tolerance = 5e-6 * max(1.0, float(primary_sq.detach().cpu()))
    if float(primary_combined_dot.detach().cpu()) + tolerance < float(primary_sq.detach().cpu()):
        raise FloatingPointError("Cosine-gated update weakens raw primary descent")
    return DuGradCosStats(
        primary_norm=float(primary_norm.detach().cpu()),
        weighted_auxiliary_norm=float(auxiliary_norm.detach().cpu()),
        cosine_before=float(cosine.detach().cpu()),
        cosine_gate=float(gate.detach().cpu()),
        gated_auxiliary_norm=float((gate * auxiliary_norm).detach().cpu()),
        combined_norm=float(combined_sq.sqrt().detach().cpu()),
        primary_combined_dot=float(primary_combined_dot.detach().cpu()),
    )


def synthetic_self_test() -> None:
    """Check the published cosine gate with in-memory vectors only."""
    parameter = torch.nn.Parameter(torch.zeros(2))

    stats = assign_du_gradcos_gradient(
        (parameter,), [torch.tensor([1.0, 0.0])],
        [torch.tensor([-1.0, 2.0])], auxiliary_weight=1.0,
    )
    assert stats.cosine_gate == 0.0
    assert torch.allclose(parameter.grad, torch.tensor([1.0, 0.0]))

    stats = assign_du_gradcos_gradient(
        (parameter,), [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, 2.0])], auxiliary_weight=1.0,
    )
    assert stats.cosine_gate == 0.0
    assert torch.allclose(parameter.grad, torch.tensor([1.0, 0.0]))

    stats = assign_du_gradcos_gradient(
        (parameter,), [torch.tensor([1.0, 0.0])],
        [torch.tensor([1.0, 1.0])], auxiliary_weight=1.0,
    )
    expected_gate = 1.0 / math.sqrt(2.0)
    assert math.isclose(stats.cosine_gate, expected_gate, rel_tol=1e-6)
    assert torch.allclose(
        parameter.grad,
        torch.tensor([1.0 + expected_gate, expected_gate]),
        atol=1e-6,
    )

    stats = assign_du_gradcos_gradient(
        (parameter,), [torch.tensor([1.0, 0.0])],
        [torch.tensor([2.0, 0.0])], auxiliary_weight=1.0,
    )
    assert stats.cosine_gate == 1.0
    assert torch.allclose(parameter.grad, torch.tensor([3.0, 0.0]))


if __name__ == "__main__":
    synthetic_self_test()
    print("DU_GRADCOS_SYNTHETIC_SELF_TEST_PASS_NO_TRAINING")
