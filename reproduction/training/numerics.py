from __future__ import annotations
import hashlib
import math
import torch
from regional_budget_model import RegionalActionSurrogate
from regional_budget_objective import (LossWeights, sequential_paired_window_backward,
    paired_window_loss, prepare_pair_targets)
from regional_state_guard import accumulate_current_gradients
from tnnls_remaining_oil_models import (ARFNO3DRemainingOil, FNOConfig,
    RTU3DRemainingOil, RTUConfig)


ACCUMULATION = 4  # two paired schedules at each of 25 and 50 steps


STATE_WEIGHTS = LossWeights(site=1.0, region=1.0, response=0.0)


RESPONSE_WEIGHTS = LossWeights(site=0.0, region=0.0, response=1.0)


def _accumulate(model: RegionalActionSurrogate, parameters: tuple,
                pair: tuple, primary: list, auxiliary: list) -> dict:
    labels, inlet, connected, action_a, action_b, times, target_a, target_b = pair
    horizon = action_a.shape[0]
    model.zero_grad(set_to_none=True)
    state = sequential_paired_window_backward(
        model, labels, inlet, connected, action_a, action_b, times,
        target_a, target_b, start=0, stop=horizon,
        weights=STATE_WEIGHTS, gradient_scale=1 / ACCUMULATION,
        memory_efficient=True,
    )
    accumulate_current_gradients(parameters, primary)
    model.zero_grad(set_to_none=True)
    response = sequential_paired_window_backward(
        model, labels, inlet, connected, action_a, action_b, times,
        target_a, target_b, start=0, stop=horizon,
        weights=RESPONSE_WEIGHTS, gradient_scale=1 / ACCUMULATION,
        memory_efficient=True,
    )
    accumulate_current_gradients(parameters, auxiliary)
    model.zero_grad(set_to_none=True)
    return {"site": state["site"], "region": state["region"],
            "response": response["response"]}


def joint_pair_gradients(
    model, initial_labels, inlet, connected, actions_a, actions_b, times,
    target_a, target_b, *, gradient_scale=1.0,
):
    """Return detached primary/auxiliary full-chain gradients for one pair.

    Callers accumulate these buffers across the SAME pairs and horizons before
    invoking the existing StateGuard projection once per optimizer update.
    No batch enlargement, temporal truncation or parameter mutation occurs.
    Whole-transition checkpointing remains enabled on both action paths.
    """
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("Gradient scale must be finite and positive")
    horizon = actions_a.shape[0]
    if (actions_a.ndim != 2 or actions_a.shape[1] != 4
            or actions_b.shape != actions_a.shape or horizon <= 0
            or times.shape != (horizon, 1)
            or target_a.shape[0] != horizon + 1
            or target_b.shape != target_a.shape):
        raise ValueError("Expected aligned complete action paths and targets")
    if not torch.is_grad_enabled():
        raise RuntimeError("Joint gradient computation requires autograd")
    outputs = [
        model(initial_labels, inlet, connected, actions, times,
              start_grad_step=0, compact_output=True, checkpoint_local=True)
        for actions in (actions_a, actions_b)
    ]
    targets = prepare_pair_targets(
        initial_labels, target_a, target_b, outputs[0], start=0, stop=horizon,
    )
    terms = paired_window_loss(
        outputs[0], outputs[1], targets, LossWeights(site=1, region=1, response=1),
    )
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    if not parameters:
        raise ValueError("No trainable parameters")
    primary = torch.autograd.grad(
        gradient_scale * (terms["site"] + terms["region"]),
        parameters, retain_graph=True, allow_unused=True,
    )
    auxiliary = torch.autograd.grad(
        gradient_scale * terms["response"], parameters, allow_unused=True,
    )
    gradients = []
    for values in (primary, auxiliary):
        detached = [None if value is None else value.detach() for value in values]
        if any(value is not None and not bool(torch.isfinite(value).all())
               for value in detached):
            raise FloatingPointError("Non-finite joint gradient")
        gradients.append(detached)
    return {
        "site": float(terms["site"].detach()),
        "region": float(terms["region"].detach()),
        "response": float(terms["response"].detach()),
        "primary": gradients[0], "auxiliary": gradients[1],
    }


def _add_buffer(destination, source):
    if len(destination) != len(source):
        raise ValueError("Gradient buffer lengths differ")
    for index, value in enumerate(source):
        if value is not None:
            if destination[index] is None:
                destination[index] = value.clone()
            else:
                destination[index].add_(value)


class ServerFNO(ARFNO3DRemainingOil):
    def _network_step(self, *args, checkpoint_local):
        return super()._network_step(*args, checkpoint_local=False)


class ServerRTU(RTU3DRemainingOil):
    def _network_step(self, *args, checkpoint_local):
        return super()._network_step(*args, checkpoint_local=False)


def server_model(arm, *, small=False):
    if arm == "fno":
        config = (FNOConfig(cell_width=4, width=4, modes=(2, 2, 2), padding=2)
                  if small else FNOConfig(width=16))
        return ServerFNO(config)
    if arm == "rtu":
        config = (RTUConfig(cell_width=4, base_width=4, positional_shape=(2, 2, 2))
                  if small else RTUConfig())
        return ServerRTU(config)
    raise ValueError("Only the two approved external network speed probes")


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

