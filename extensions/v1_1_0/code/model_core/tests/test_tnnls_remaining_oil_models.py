"""Synthetic-only model/gradient tests; no data/checkpoint/optimizer access."""

from __future__ import annotations

import copy
import json
import math
import time
import unittest

import torch

from regional_budget_objective import (
    LossWeights, paired_window_loss, prepare_pair_targets,
    sequential_paired_window_backward,
)
from regional_budget_model import RegionalConfig
from regional_state_guard import assign_state_guarded_gradient
from regional_statefront_model_candidate import RegionalStateFrontSurrogate
from softfront_budget_candidate import SparseFrontGeometry
from tnnls_remaining_oil_models import (
    CoarseStateFrontRemainingOil, SpectralConv3D, make_model,
)


def synthetic_inputs(size=16, horizon=4, *, device="cpu", dense_oil=False):
    """Independent in-memory fixture, not a generated research/test dataset."""
    z, y, x = torch.meshgrid(*[torch.arange(size)] * 3, indexing="ij")
    oil = x > 0 if dense_oil else ((x > 0) & ((3 * z + 5 * y + x) % 5 == 0))
    labels = torch.full((size, size, size), 2, dtype=torch.uint8)
    labels[oil] = (3 + z[oil] % 5).to(torch.uint8)
    connected = torch.ones_like(oil)
    inlet = x == 0
    action_a = torch.tensor([[0.25, 0.4, 0.0, 0.0], [0.7, 0.2, 0.3, 1.0]])
    action_b = action_a.flip(0)
    action_a = action_a.repeat((horizon + 1) // 2, 1)[:horizon].clone()
    action_b = action_b.repeat((horizon + 1) // 2, 1)[:horizon].clone()
    steps = torch.arange(1, horizon + 1, dtype=torch.float32)
    times = (steps / (steps + 25)).unsqueeze(1)
    target_a = oil.repeat(horizon + 1, 1, 1, 1).to(torch.uint8)
    target_b = target_a.clone()
    for step in range(1, horizon + 1):
        target_a[step, :, :, :min(step + 1, size)] = 0
        target_b[step, :min(step, size), :, :min(step + 1, size)] = 0
    inputs = [value.to(device) for value in
              (labels, inlet, connected, action_a, action_b, times)]
    return (*inputs, target_a, target_b)


class TNNLSModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20260811)

    def test_spectral_dc_and_real_parameter_gradient(self):
        spectral = SpectralConv3D(1, (2, 2, 2))
        with torch.no_grad():
            spectral.weight_ri.zero_()
            spectral.weight_ri[0, 0, 0, 0, 0, 0, 0] = 1
        value = torch.ones(1, 1, 8, 8, 8, requires_grad=True)
        output = spectral(value)
        torch.testing.assert_close(output, value)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(spectral.weight_ri.grad).all())
        self.assertFalse(spectral.weight_ri.is_complex())

        random = SpectralConv3D(2, (2, 2, 2)).double()
        value = torch.randn(1, 2, 4, 4, 6, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(random, (value,), fast_mode=True))
        random(value).square().mean().backward()
        self.assertGreater(float(random.weight_ri.grad[..., 1].abs().max()), 0)

    def test_coarse_front_has_same_weights_and_same_first_step(self):
        config = RegionalConfig(cell_width=4, local_width=4, hidden_width=8)
        fine = RegionalStateFrontSurrogate(config, readout_mode="direct")
        coarse = CoarseStateFrontRemainingOil(config)
        coarse.load_state_dict(fine.state_dict(), strict=True)
        self.assertEqual(list(fine.state_dict()), list(coarse.state_dict()))
        self.assertEqual(sum(p.numel() for p in fine.parameters()),
                         sum(p.numel() for p in coarse.parameters()))
        labels, inlet, connected, a, _, times, *_ = synthetic_inputs()
        with torch.no_grad():
            fa = fine(labels, inlet, connected, a[:1], times[:1], compact_output=True)
            ca = coarse(labels, inlet, connected, a[:1], times[:1], compact_output=True)
        for name in ("remaining_sites", "group_remaining"):
            torch.testing.assert_close(fa[name], ca[name], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in make_model("coarse_front").parameters()),
                         90991)

    def test_sparse_spectrum_matches_dense_four_corner_reference(self):
        layer = SpectralConv3D(2, (2, 2, 2)).double()
        reference = copy.deepcopy(layer)
        value = torch.randn(1, 2, 8, 8, 8, dtype=torch.float64, requires_grad=True)
        other = value.detach().clone().requires_grad_(True)
        actual = layer(value)
        spectrum = torch.fft.rfftn(other, dim=(-3, -2, -1), norm="ortho")
        projected = torch.zeros_like(spectrum)
        weights = torch.view_as_complex(reference.weight_ri.contiguous())
        for index, (zs, ys) in enumerate(((slice(0, 2), slice(0, 2)),
                                         (slice(-2, None), slice(0, 2)),
                                         (slice(0, 2), slice(-2, None)),
                                         (slice(-2, None), slice(-2, None)))):
            projected[:, :, zs, ys, :2] = torch.einsum(
                "bizyx,iozyx->bozyx", spectrum[:, :, zs, ys, :2], weights[index])
        expected = torch.fft.irfftn(projected, s=(8, 8, 8),
                                   dim=(-3, -2, -1), norm="ortho")
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(value.grad, other.grad, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(layer.weight_ri.grad, reference.weight_ri.grad,
                                   atol=1e-12, rtol=1e-12)

    def test_equal_inventory_can_have_different_fine_fronts(self):
        labels = torch.full((1, 1, 9), 2, dtype=torch.uint8)
        labels[0, 0, 1:8] = 3
        indices = torch.arange(1, 8)
        inlet = torch.zeros_like(labels, dtype=torch.bool)
        connected = torch.ones_like(inlet)
        front = SparseFrontGeometry.from_initial(labels, indices, inlet, connected)
        pa = torch.tensor([0.2, 0.2, 1, 1, 1, 1, 1])
        pb = torch.tensor([0.2, 1, 1, 1, 1, 1, 0.2])
        torch.testing.assert_close(pa.sum(), pb.sum())
        self.assertAlmostEqual(float(front.weight(pa)[2]), 0.8, places=6)
        self.assertEqual(float(front.weight(pb)[2]), 0)
        torch.testing.assert_close(front.weight(pa.mean().expand(7)),
                                   front.weight(pb.mean().expand(7)))

    def test_fields_counts_causality_and_no_oil_creation(self):
        labels, inlet, connected, a, _, times, *_ = synthetic_inputs(horizon=4)
        for arm in ("fno", "rtu", "coarse_front"):
            with self.subTest(arm=arm):
                model = make_model(arm, small=True)
                with torch.no_grad():
                    output = model(labels, inlet, connected, a, times, return_fields=True)
                    prefix = model(labels, inlet, connected, a[:2], times[:2], compact_output=True)
                    changed = a.clone()
                    changed[2:] = 1 - changed[2:]
                    future_changed = model(labels, inlet, connected, changed, times,
                                           compact_output=True)
                sites = output["remaining_sites"]
                self.assertTrue(bool(torch.all((sites >= 0) & (sites <= 1))))
                self.assertTrue(bool(torch.all(sites[1:] <= sites[:-1])))
                torch.testing.assert_close(sites[:2], prefix["remaining_sites"], rtol=0, atol=0)
                torch.testing.assert_close(sites[:2], future_changed["remaining_sites"][:2],
                                           rtol=0, atol=0)
                self.assertEqual(output["So_sequence"].shape, (4, *labels.shape))
                self.assertTrue(bool(torch.all(output["So_sequence"][:, labels < 3] == 0)))
                torch.testing.assert_close(output["So_sequence"].flatten(1).sum(1),
                                           output["class_remaining"].sum(1))
                initial = output["initial_group_inventory"].reshape(-1, 5).sum(0)
                torch.testing.assert_close(output["class_remaining"]
                                           + output["class_mobilized"], initial.expand(4, -1))
                with torch.no_grad():
                    stagnant = model(labels, torch.zeros_like(inlet),
                                     torch.zeros_like(connected), a, times, compact_output=True)
                self.assertTrue(bool(torch.all(stagnant["remaining_sites"] == 1)))

    def test_full_chain_checkpoint_and_paired_replay_gradients(self):
        labels, inlet, connected, a, b, times, ta, tb = synthetic_inputs(horizon=3)
        weights = LossWeights(site=1, region=1, response=5)
        for arm in ("fno", "rtu", "coarse_front"):
            for start in (0, 1):
                with self.subTest(arm=arm, start=start):
                    reference = make_model(arm, small=True)
                    checked = copy.deepcopy(reference)
                    replay = copy.deepcopy(reference)
                    outputs = [reference(labels, inlet, connected, act, times,
                                         start_grad_step=start, compact_output=True)
                               for act in (a, b)]
                    targets = prepare_pair_targets(labels, ta, tb, outputs[0],
                                                   start=start, stop=3)
                    losses = paired_window_loss(*outputs, targets, weights)
                    losses["total"].backward()
                    checked_outputs = [checked(labels, inlet, connected, act, times,
                                               start_grad_step=start, compact_output=True,
                                               checkpoint_local=True)
                                       for act in (a, b)]
                    for expected, actual in zip(outputs, checked_outputs):
                        for name in ("remaining_sites", "group_remaining"):
                            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
                    paired_window_loss(*checked_outputs, targets, weights)["total"].backward()
                    measured = sequential_paired_window_backward(
                        replay, labels, inlet, connected, a, b, times, ta, tb,
                        start=start, stop=3, weights=weights, memory_efficient=True,
                    )
                    for name, value in losses.items():
                        self.assertTrue(math.isclose(float(value.detach()), measured[name],
                                                     rel_tol=2e-5, abs_tol=2e-7))
                    for (name, parameter), (_, check), (_, sequential) in zip(
                            reference.named_parameters(), checked.named_parameters(),
                            replay.named_parameters()):
                        if parameter.grad is None:
                            # Only the explicitly detached suffix interface can
                            # make an initial-hidden-state parameter unused.
                            self.assertGreater(start, 0, name)
                            self.assertIsNone(check.grad, name)
                            self.assertIsNone(sequential.grad, name)
                            continue
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                        torch.testing.assert_close(parameter.grad, check.grad, atol=3e-5, rtol=3e-4)
                        torch.testing.assert_close(parameter.grad, sequential.grad, atol=3e-5, rtol=3e-4)
                    parameters = tuple(replay.parameters())
                    primary = [None if p.grad is None else p.grad.clone() for p in parameters]
                    auxiliary = [None if value is None else -0.1 * value for value in primary]
                    stats = assign_state_guarded_gradient(parameters, primary, auxiliary,
                                                          auxiliary_weight=50)
                    self.assertGreaterEqual(stats.primary_auxiliary_dot_after, -1e-5)

    def test_hundred_steps_and_early_action_gradient(self):
        labels, inlet, connected, a, _, times, *_ = synthetic_inputs(size=8, horizon=100)
        for arm in ("fno", "rtu", "coarse_front"):
            with self.subTest(arm=arm):
                model = make_model(arm, small=True)
                with torch.no_grad():
                    output = model(labels, inlet, connected, a, times, compact_output=True)
                self.assertEqual(output["remaining_sites"].shape[0], 100)
                early = a[:3].clone().requires_grad_(True)
                end = model(labels, inlet, connected, early, times[:3],
                            compact_output=True, checkpoint_local=True)
                end["remaining_sites"][-1].mean().backward()
                self.assertTrue(bool(torch.isfinite(early.grad).all()))
                self.assertGreater(float(early.grad[0].abs().max()), 0)


if __name__ == "__main__":
    began = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TNNLSModelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "status": "SYNTHETIC_UNIT_PASS" if result.wasSuccessful() else "SYNTHETIC_UNIT_FAIL",
        "seconds": time.perf_counter() - began, "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "optimizer_updates": 0, "research_trajectories_read": 0,
        "arms": ["fno", "rtu", "coarse_front"],
    }))
    raise SystemExit(0 if result.wasSuccessful() else 1)
