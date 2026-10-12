"""New control loop retaining the recorded numerical operations and ordering."""
from __future__ import annotations
import hashlib
import math
import torch
from regional_budget_objective import LossWeights, sequential_paired_window_backward
from regional_state_guard import (new_gradient_buffer, accumulate_current_gradients,
                                  assign_state_guarded_gradient)
from regional_du_gradcos import assign_du_gradcos_gradient
from regional_pcgrad import assign_pcgrad_gradient
from numerics import _accumulate, joint_pair_gradients, _add_buffer


def shuffled_order(panel, config):
    order = torch.randperm(len(panel), generator=torch.Generator().manual_seed(config["seed"])).tolist()
    sha = hashlib.sha256("\n".join(panel[i]["pair_id"] for i in order).encode()).hexdigest()
    if sha != config["pair_order_sha256"]: raise ValueError("Recorded shuffled TRAIN order differs")
    return order


def assign_rule(parameters, primary, auxiliary, rule):
    if rule == "state_prioritized":
        return assign_state_guarded_gradient(parameters, primary, auxiliary, auxiliary_weight=50.0,
                                            max_auxiliary_to_primary_norm=1.0)
    if rule == "du_cosine_gate":
        return assign_du_gradcos_gradient(parameters, primary, auxiliary, auxiliary_weight=50.0)
    if rule == "pcgrad":
        return assign_pcgrad_gradient(parameters, primary, auxiliary, auxiliary_weight=50.0)
    raise ValueError("Unknown adaptive learning rule")


def one_update(model, optimizer, config, batch, load_pair, check_memory=lambda: None):
    """One original batch: two mixed-horizon IDs or four five-window IDs."""
    if len(batch) != config["pair_ids_per_update"]: raise ValueError("Incomplete optimizer batch")
    parameters = tuple(model.parameters())
    primary, auxiliary = new_gradient_buffer(parameters), new_gradient_buffer(parameters)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    measures = []
    learning = config["study"] == "learning_rule_25"
    fixed = config["learning_rule"] in ("response_weight_0", "fixed_weight_50")
    for record in batch:
        initial = None
        for horizon in config["horizons"]:
            pair = load_pair(record, horizon)
            if initial is not None and any(not torch.equal(a, b.detach().cpu()) for a, b in zip(initial, pair[:3])):
                raise ValueError("Mixed-horizon initial geometry differs")
            initial = tuple(v.detach().cpu().clone() for v in pair[:3])
            if not learning:
                if config["gradient_path"] == "joint_full_chain":
                    result = joint_pair_gradients(model, *pair, gradient_scale=0.25)
                    _add_buffer(primary, result.pop("primary"))
                    _add_buffer(auxiliary, result.pop("auxiliary"))
                else: result = _accumulate(model, parameters, pair, primary, auxiliary)
                measures.append(result)
                check_memory()
            else:
                for start in range(0, 25, 5):
                    if fixed:
                        result = sequential_paired_window_backward(model, *pair, start=start, stop=start+5,
                            weights=LossWeights(site=1, region=1, response=config["response_weight"]),
                            gradient_scale=0.05, memory_efficient=True)
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        result = sequential_paired_window_backward(model, *pair, start=start, stop=start+5,
                            weights=LossWeights(site=1, region=1, response=0), gradient_scale=0.05, memory_efficient=True)
                        accumulate_current_gradients(parameters, primary)
                        optimizer.zero_grad(set_to_none=True)
                        other = sequential_paired_window_backward(model, *pair, start=start, stop=start+5,
                            weights=LossWeights(site=0, region=0, response=1), gradient_scale=0.05, memory_efficient=True)
                        for name in ("site", "region", "response"):
                            if not math.isclose(result[name], other[name], rel_tol=1e-4, abs_tol=1e-8):
                                raise ValueError("Primary/auxiliary replay differs")
                        accumulate_current_gradients(parameters, auxiliary)
                        optimizer.zero_grad(set_to_none=True)
                    measures.append(result)
                    check_memory()
            del pair
    diagnostics = None if fixed else assign_rule(parameters, primary, auxiliary, config["learning_rule"])
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not math.isfinite(float(norm)): raise FloatingPointError("Non-finite gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    result = {"loss": {name: sum(r[name] for r in measures) / len(measures) for name in ("site", "region", "response")},
              "preclip_gradient_norm": float(norm), "gradient_contributions": len(measures)}
    if not all(math.isfinite(v) for v in result["loss"].values()): raise FloatingPointError("Non-finite loss")
    if diagnostics is not None: result["gradient_rule"] = diagnostics.to_dict()
    return result
