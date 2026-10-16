"""Untrained full-field recurrence with an explicit limited GPU-check mode.

Default: CPU hand tensors, sides <=16, batch <=2, horizon <=25.
Opt-in full128_gpu_check: exactly 128^3, batch 1, CUDA:0 FP32, <=5 steps.
Separate full128_paired_pilot profile: <=25 steps; its guarded entry requires
fresh approval. Adding this profile does not execute it or change the old one.
The separate runtime entry enforces the one-TRAIN/resource/no-update scope.
No simulator, checkpoint I/O or optimizer. Both arms share these equations.
Optional activation_recompute changes backward storage, not the recurrence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MAX_SIDE = 16
MAX_BATCH = 2
MAX_STEPS = 25
CPU_SOFTWARE = "cpu_software"
FULL128_CHECK = "full128_gpu_check"
FULL128_PAIRED = "full128_paired_pilot"
MENU = (
    (0.65 / 1.45, 0.0, 0.0, 0.0),
    (1.0 / 1.45, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0 / 1.45, 0.5, 0.16 / 0.28, 1.0),
    (1.0 / 1.45, 1.0, 1.0, 1.0),
)


def action_tensor(name: str, *, dtype=torch.float64, batch=1):
    if name not in ("S1", "S2", "S3", "S4", "S5"):
        raise ValueError("Only canonical S1..S5 are supported")
    return torch.tensor(MENU[int(name[1]) - 1], dtype=dtype).repeat(batch, 1)


def _checked_float(tensor, name, runtime=CPU_SOFTWARE):
    if runtime == CPU_SOFTWARE:
        if tensor.device.type != "cpu" or tensor.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name}: CPU float32/float64 only")
    elif runtime in (FULL128_CHECK, FULL128_PAIRED):
        # CPU is allowed for initial-state validation before explicit transfer.
        if (tensor.dtype != torch.float32 or tensor.device.type not in ("cpu", "cuda")
                or (tensor.device.type == "cuda" and tensor.device.index != 0)):
            raise ValueError(f"{name}: full128 check requires FP32, CPU adapter or CUDA:0")
    else:
        raise ValueError("Unknown runtime mode")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name}: nonfinite input")


def _unit_interval(tensor, name):
    if bool(((tensor < 0) | (tensor > 1)).any()):
        raise ValueError(f"{name}: values must lie in [0,1]")


def validate_action(action, reference, runtime=CPU_SOFTWARE):
    _checked_float(action, "action", runtime)
    if (action.shape != (reference.shape[0], 4) or action.dtype != reference.dtype
            or action.device != reference.device):
        raise ValueError("Expected [B,4] action with matching state dtype/device")
    menu = action.new_tensor(MENU)
    matches = (action[:, None] - menu[None]).abs().amax(dim=2) <= 2e-7
    if not bool(matches.any(dim=1).all()):
        raise ValueError("Only the fixed five-action menu is enabled")


@dataclass(frozen=True)
class Geometry:
    pore: torch.Tensor
    core: torch.Tensor
    oil0: torch.Tensor
    water0: torch.Tensor
    connected: torch.Tensor
    inlet: torch.Tensor
    axis_gradient: torch.Tensor

    def validate(self, runtime=CPU_SOFTWARE):
        shape = self.pore.shape
        if runtime in (FULL128_CHECK, FULL128_PAIRED):
            if shape != (1, 1, 128, 128, 128):
                raise ValueError("Full128 check requires exactly [1,1,128,128,128]")
        elif runtime == CPU_SOFTWARE:
            if (len(shape) != 5 or shape[1] != 1 or not 1 <= shape[0] <= MAX_BATCH
                or any(n < 4 or n > MAX_SIDE or n % 4 for n in shape[2:])):
                raise ValueError("Prototype accepts [B,1,D,H,W], B<=2, sides 4..16 divisible by 4")
        else:
            raise ValueError("Unknown runtime mode")
        for name, tensor in vars(self).items():
            expected = (shape[0], 5, *shape[2:]) if name == "oil0" else shape
            _checked_float(tensor, name, runtime)
            if (tensor.shape != expected or tensor.dtype != self.pore.dtype
                    or tensor.device != self.pore.device):
                raise ValueError(f"{name}: incompatible shape/dtype/device")
            if name == "axis_gradient":
                _unit_interval(tensor, name)
            elif not bool(((tensor == 0) | (tensor == 1)).all()):
                raise ValueError(f"{name}: initial masks must be binary")
        if not torch.equal(self.oil0.sum(dim=1, keepdim=True) + self.water0, self.pore):
            raise ValueError("Five disjoint initial oil classes plus water must equal pore")
        if bool((self.pore > self.core).any()):
            raise ValueError("Pore extends outside core")
        if bool((self.connected > self.pore).any()) or bool((self.inlet > self.connected).any()):
            raise ValueError("Inlet/connectivity extends outside its allowed support")
        canonical_inlet = (self.axis_gradient >= 0.999999).to(self.pore.dtype) * self.pore
        if not torch.equal(self.inlet, canonical_inlet):
            raise ValueError("Inlet must follow the unchanged global-gradient convention")


@dataclass(frozen=True)
class Prepared:
    geometry: Geometry
    z0: torch.Tensor
    z_bias: torch.Tensor
    h_bias: torch.Tensor
    h_memory_weights: torch.Tensor
    h_action_weights: torch.Tensor


@dataclass(frozen=True)
class State:
    u: torch.Tensor
    z: torch.Tensor


def six_neighbour_max(field):
    # Zero padding, exactly six faces; no wrap, centre, or diagonal neighbours.
    neighbours = [
        F.pad(field[:, :, :-1], (0, 0, 0, 0, 1, 0)),
        F.pad(field[:, :, 1:], (0, 0, 0, 0, 0, 1)),
        F.pad(field[:, :, :, :-1], (0, 0, 1, 0, 0, 0)),
        F.pad(field[:, :, :, 1:], (0, 0, 0, 1, 0, 0)),
        F.pad(field[:, :, :, :, :-1], (1, 0, 0, 0, 0, 0)),
        F.pad(field[:, :, :, :, 1:], (0, 1, 0, 0, 0, 0)),
    ]
    return torch.stack(neighbours).amax(dim=0)


def front_gate(geometry, u):
    water = geometry.water0 + geometry.oil0.sum(dim=1, keepdim=True) * u
    source = torch.maximum(water, geometry.inlet) * geometry.connected
    return six_neighbour_max(source) * geometry.connected


class OrderRecurrence3D(nn.Module):
    """Static 3D encoder + isotone (or signed) low-resolution recurrence.

    Geometry features are prepared once per rollout. Re-prepare after any
    future parameter update: cached features must not cross optimizer steps.
    With activation_recompute, internal activations are recomputed during
    backward. Do not mutate parameters, prepared fields or states between
    forward and backward. This is activation checkpointing, not file I/O.
    No dynamic normalization, FiLM, attention, or pressure feedback.
    """

    def __init__(self, *, order_preserving=True, base_channels=8, latent_channels=16,
                 runtime=CPU_SOFTWARE, activation_recompute=False):
        super().__init__()
        if not isinstance(activation_recompute, bool):
            raise ValueError("activation_recompute must be an explicit bool")
        self.activation_recompute = activation_recompute
        if runtime not in (CPU_SOFTWARE, FULL128_CHECK, FULL128_PAIRED):
            raise ValueError("Unknown runtime mode")
        self.runtime = runtime
        self.max_steps = 5 if runtime == FULL128_CHECK else MAX_STEPS
        if not 1 <= base_channels <= 8 or not 1 <= latent_channels <= 16:
            raise ValueError("Prototype channel limits: base<=8, latent<=16")
        self.order_preserving = order_preserving
        self.latent_channels = latent_channels
        b, d = base_channels, latent_channels
        self.encoder = nn.Sequential(nn.Conv3d(11, b, 3, padding=1), nn.SiLU(),
                                     nn.Conv3d(b, b, 3, padding=1), nn.SiLU())
        self.z_initial = nn.Conv3d(b, d, 1)
        self.z_bias = nn.Conv3d(b, d, 1)
        self.h_bias = nn.Conv3d(b, 1, 1)
        self.h_memory = nn.Conv3d(b, d, 1)
        self.h_action = nn.Conv3d(b, 4, 1)
        self.kz_raw = nn.Parameter(torch.empty(d, d, 3, 3, 3))
        self.ku_raw = nn.Parameter(torch.empty(d, 1, 3, 3, 3))
        self.ba_raw = nn.Parameter(torch.empty(d, 4))
        with torch.no_grad():
            # Fan-in scaling below prevents a sum of hundreds of softplus(0)'s.
            for raw in (self.kz_raw, self.ku_raw, self.ba_raw):
                nn.init.normal_(raw, mean=math.log(math.expm1(0.25)), std=0.05)
            nn.init.normal_(self.h_bias.weight, mean=0.0, std=0.03)
            self.h_bias.bias.fill_(math.log(0.015 / 0.985))
            # Constant coefficient fields at initialization permit exact
            # all-geometry function matching with the signed control.
            for head in (self.h_memory, self.h_action):
                head.weight.zero_()
                head.bias.fill_(math.log(math.expm1(0.25)))

    def coefficient(self, raw, fan_in):
        return (F.softplus(raw) if self.order_preserving else raw) / fan_in

    def prepare(self, geometry):
        geometry.validate(self.runtime)
        parameter = next(self.parameters())
        if parameter.device != geometry.pore.device or parameter.dtype != geometry.pore.dtype:
            raise ValueError("Model must match geometry dtype/device")
        expected_device = "cpu" if self.runtime == CPU_SOFTWARE else "cuda"
        if parameter.device.type != expected_device:
            raise ValueError(f"{self.runtime} model must execute on {expected_device}")
        # Snapshot immutable source masks; do not let caller-side changes affect a rollout.
        geometry = Geometry(**{k: v.clone() for k, v in vars(geometry).items()})
        inputs = (
            geometry.pore, geometry.core, geometry.oil0, geometry.water0,
            geometry.connected, geometry.inlet, geometry.axis_gradient,
        )
        if self.activation_recompute and torch.is_grad_enabled():
            fields = checkpoint(self._prepare_fields, *inputs, use_reentrant=False,
                                preserve_rng_state=True)
        else:
            fields = self._prepare_fields(*inputs)
        return Prepared(geometry, *fields)

    def _prepare_fields(self, *inputs):
        # Identical operation order for eager and recomputing execution.
        features = self.encoder(torch.cat(inputs, dim=1))
        coarse = F.avg_pool3d(features, 4, 4)
        return (torch.sigmoid(self.z_initial(coarse)), self.z_bias(coarse),
                self.h_bias(features),
                self.coefficient(self.h_memory(features), self.latent_channels),
                self.coefficient(self.h_action(features), 4))

    @staticmethod
    def initial_state(prepared):
        return State(torch.zeros_like(prepared.geometry.pore), prepared.z0)

    def validate_state(self, prepared, state):
        g = prepared.geometry
        for name, value, shape in (("u", state.u, g.pore.shape),
                                   ("z", state.z, prepared.z0.shape)):
            _checked_float(value, name, self.runtime)
            if value.shape != shape or value.dtype != g.pore.dtype or value.device != g.pore.device:
                raise ValueError(f"{name}: incompatible state")
            _unit_interval(value, name)
        allowed = g.oil0.sum(dim=1, keepdim=True) * g.connected
        if bool((state.u * (1 - allowed) != 0).any()):
            raise ValueError("Mobilization outside connected initial oil")

    def step(self, prepared, state, action):
        self.validate_state(prepared, state)
        validate_action(action, state.u, self.runtime)
        if self.activation_recompute and torch.is_grad_enabled():
            # Explicit tensor inputs preserve the full temporal and static-head
            # gradient paths. Geometry is the existing immutable snapshot.
            def advance(u, z, act, z_bias, h_bias, memory_weights, action_weights):
                local = Prepared(prepared.geometry, prepared.z0, z_bias, h_bias,
                                 memory_weights, action_weights)
                next_state, aux = self._step_fields(local, State(u, z), act)
                return (next_state.u, next_state.z, aux["old_front_gate"],
                        aux["effective_hazard"])

            u, z, gate, hazard = checkpoint(
                advance, state.u, state.z, action, prepared.z_bias, prepared.h_bias,
                prepared.h_memory_weights, prepared.h_action_weights,
                use_reentrant=False, preserve_rng_state=True)
            return State(u, z), {"old_front_gate": gate, "effective_hazard": hazard}
        return self._step_fields(prepared, state, action)

    def _step_fields(self, prepared, state, action):
        g, d = prepared.geometry, self.latent_channels
        gate = front_gate(g, state.u)  # Always the OLD water state.
        z = torch.sigmoid(
            prepared.z_bias
            + F.conv3d(state.z, self.coefficient(self.kz_raw, 27 * d), padding=1)
            + F.conv3d(F.avg_pool3d(state.u, 4, 4),
                       self.coefficient(self.ku_raw, 27), padding=1)
            + F.linear(action, self.coefficient(self.ba_raw, 4))[:, :, None, None, None]
        )
        lifted = F.interpolate(z, size=state.u.shape[2:], mode="trilinear", align_corners=False)
        h = torch.sigmoid(
            prepared.h_bias
            + (prepared.h_memory_weights * lifted).sum(dim=1, keepdim=True)
            + (prepared.h_action_weights * action[:, :, None, None, None]).sum(dim=1, keepdim=True)
        )
        u = g.oil0.sum(dim=1, keepdim=True) * (state.u + (1 - state.u) * gate * h)
        return State(u, z), {"old_front_gate": gate, "effective_hazard": h}

    def rollout(self, prepared, actions):
        if (actions.ndim != 3 or actions.shape[1:] != (prepared.geometry.pore.shape[0], 4)
                or not 1 <= len(actions) <= self.max_steps):
            raise ValueError(f"Expected 1..{self.max_steps} actions in [T,B,4]")
        states, auxiliary = [self.initial_state(prepared)], []
        for action in actions:
            state, aux = self.step(prepared, states[-1], action)
            states.append(state)
            auxiliary.append(aux)
        return states, auxiliary

    def readout(self, prepared, state, *, action=None, previous=None):
        self.validate_state(prepared, state)
        if previous is not None:
            self.validate_state(prepared, previous)
            if bool((previous.u > state.u).any()):
                raise ValueError("Previous state cannot have more cumulative mobilization")
        if action is not None:
            validate_action(action, state.u, self.runtime)
        return field_readout(prepared.geometry, state.u, action=action,
                             previous_u=None if previous is None else previous.u)


def field_readout(g, u, *, action=None, previous_u=None):
    """Unchanged analytic readout shared by neural and fitted mean-field arms.

    Callers validate their geometry/state/runtime before entering this kernel.
    No network, fitted statistics or future labels enter the readout.
    """
    oil = g.oil0 * (1 - u)
    water = g.water0 + g.oil0.sum(dim=1, keepdim=True) * u
    initial = g.oil0.double().sum(dim=(2, 3, 4))
    remaining = oil.double().sum(dim=(2, 3, 4))
    mobilized = initial - remaining
    pore_count = g.pore.double().sum(dim=(1, 2, 3, 4)).clamp_min(1)
    initial_total = initial.sum(dim=1)
    soft_front = oil.sum(dim=1, keepdim=True).clamp(0, 1) * front_gate(g, u)
    result = {
        "oil": oil, "water": water, "So": oil.sum(dim=1, keepdim=True),
        "Sor": remaining.sum(dim=1) / pore_count,
        "RF": mobilized.sum(dim=1) / initial_total.clamp_min(1),
        "class_remaining": remaining, "class_mobilized": mobilized,
        "class_mobilized_fraction": mobilized / initial.clamp_min(1),
        "front": soft_front, "front_hard": soft_front >= 0.5,
        "initial_oil_present": initial_total > 0,
        "class_present": initial > 0,
    }
    if previous_u is not None:
        result["class_step_mobilized"] = (
            g.oil0 * (u - previous_u)
        ).double().sum(dim=(2, 3, 4))
    if action is not None:
        # Same full-field analytic closure as patent22/model.py, not a neural input.
        pc = g.pore.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1)
        cc = g.core.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1)
        phi = (pc / cc).clamp_min(1e-4)
        sor = oil.sum(dim=(1, 2, 3, 4), keepdim=True) / pc
        rate = action[:, 0:1, None, None, None] * 1.45
        permeability = phi.pow(3) / ((1 - phi).pow(2) + 1e-6)
        drop = rate * (1 + 1.8 * sor) / (permeability + 1e-6)
        physical = g.axis_gradient * g.pore * drop
        result["pressure_proxy"] = torch.log1p(physical.clamp_min(0)) / torch.log(
            g.pore.new_tensor(2049.0)
        )
    return result


def make_matched_pair(*, seed=20260908, dtype=torch.float64,
                      base_channels=8, latent_channels=16, runtime=CPU_SOFTWARE,
                      activation_recompute=False):
    """Fresh models, identical effective initial functions for EVERY geometry.

    Not checkpoint conversion or a trained-model ablation. Both arms would
    require independent training under any future approved comparison.
    """
    if runtime in (FULL128_CHECK, FULL128_PAIRED) and dtype != torch.float32:
        raise ValueError("Full128 GPU check is FP32 only")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        candidate = OrderRecurrence3D(base_channels=base_channels,
                                      latent_channels=latent_channels, runtime=runtime,
                                      activation_recompute=activation_recompute).to(dtype=dtype)
    control = deepcopy(candidate)
    control.order_preserving = False
    with torch.no_grad():
        for name in ("kz_raw", "ku_raw", "ba_raw"):
            getattr(control, name).copy_(F.softplus(getattr(candidate, name)))
        for name in ("h_memory", "h_action"):
            source, target = getattr(candidate, name), getattr(control, name)
            if bool((source.weight != 0).any()):
                raise ValueError("Only fresh constant coefficient heads can be matched exactly")
            target.bias.copy_(F.softplus(source.bias))
    return candidate, control
