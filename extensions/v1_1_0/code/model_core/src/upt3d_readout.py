"""Initial-support oil readout and inference-only streaming.

The decoder score is interpreted as remaining fraction on initial oil sites.
This is a shared engineering output constraint, not conservation dynamics:
successive predictions can increase oil. No front-probability or pressure claim.
Formula reference: full3d_order_recurrence.field_readout.
"""
import math
import torch


def iter_reachable_oil_support(geometry, *, inlet, connected, steps):
    """Yield conservative oil support after each synchronous six-face step.

    Existing water and the effective inlet are step-zero sources. A conversion
    can traverse at most one connected pore edge per action. This is an upper
    bound, not a predicted front or a claim that every marked oil site moves.
    Geometry is fixed; no future states, actions, random fields or targets enter.
    """
    validate_geometry(geometry)
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    pore = geometry[:, :1].bool()
    oil = geometry[:, 1:].sum(1, keepdim=True).bool()
    for name, mask in (("inlet", inlet), ("connected", connected)):
        if mask.shape != pore.shape or mask.device != pore.device:
            raise ValueError(name + " must match geometry shape/device")
        if not torch.isfinite(mask).all() or ((mask != 0) & (mask != 1)).any():
            raise ValueError(name + " must be binary")
    inlet, connected = inlet.bool(), connected.bool()
    if (connected & ~pore).any() or (inlet & ~connected).any():
        raise ValueError("Require inlet subset of connected subset of pore")
    reached = ((pore & ~oil) | inlet) & connected
    for _ in range(steps):
        expanded = reached.clone()
        expanded[:, :, 1:] |= reached[:, :, :-1]
        expanded[:, :, :-1] |= reached[:, :, 1:]
        expanded[:, :, :, 1:] |= reached[:, :, :, :-1]
        expanded[:, :, :, :-1] |= reached[:, :, :, 1:]
        expanded[:, :, :, :, 1:] |= reached[:, :, :, :, :-1]
        expanded[:, :, :, :, :-1] |= reached[:, :, :, :, 1:]
        reached = expanded & connected
        yield reached & oil


def deterministic_front_proxy(so, pore, *, inlet, connected):
    """Existing point-state front operator, NOT a joint front probability.

    Full-volume [B,1,Z,Y,X] inputs on one device. Caller supplies authoritative
    static masks; this function validates containment, not graph connectivity.
    No threshold tuning, learned head, target input or default fabricated masks.
    Diagnostic output only; not an authorization to run the formal evaluator.
    """
    from full3d_order_recurrence import six_neighbour_max

    if so.ndim != 5 or so.shape[1] != 1 or min(so.shape) < 1:
        raise ValueError("Expected a complete [B,1,Z,Y,X] oil-state field")
    if not so.is_floating_point() or not torch.isfinite(so).all():
        raise ValueError("Oil state must be finite and floating point")
    for name, mask in (("pore", pore), ("inlet", inlet), ("connected", connected)):
        if mask.shape != so.shape or mask.device != so.device:
            raise ValueError(name + " must match state shape/device")
        if not torch.isfinite(mask).all() or ((mask != 0) & (mask != 1)).any():
            raise ValueError(name + " must be binary")
    if (connected > pore).any() or (inlet > connected).any():
        raise ValueError("Require inlet subset of connected subset of pore")
    if ((so < 0) | (so > pore)).any():
        raise ValueError("Oil state must lie in [0,pore]; no clipping allowed")
    water = pore.to(so.dtype) - so
    source = torch.maximum(water, inlet.to(so.dtype)) * connected
    return so * six_neighbour_max(source) * connected


def validate_geometry(geometry):
    if geometry.ndim != 5 or geometry.shape[1] != 6:
        raise ValueError("Expected [B,6,Z,Y,X] geometry")
    if min(geometry.shape) < 1:
        raise ValueError("Empty geometry")
    if not torch.isfinite(geometry).all() or ((geometry != 0) & (geometry != 1)).any():
        raise ValueError("Full-resolution geometry must contain binary indicators")
    if (geometry[:, 1:].sum(1) > geometry[:, 0]).any():
        raise ValueError("Initial oil classes must be disjoint within pores")
    if (geometry[:, 0].flatten(1).sum(1) == 0).any():
        raise ValueError("No pores in sample")


def validate_reachable_oil(reachable_oil, geometry):
    if reachable_oil.shape != geometry[:, :1].shape or reachable_oil.device != geometry.device:
        raise ValueError("Reachable oil support must match geometry shape/device")
    if not torch.isfinite(reachable_oil).all() or ((reachable_oil != 0) & (reachable_oil != 1)).any():
        raise ValueError("Reachable oil support must be binary")
    if (reachable_oil > geometry[:, 1:].sum(1, keepdim=True)).any():
        raise ValueError("Reachable support must be within initial oil")
    return reachable_oil.bool()


def inventory_readout(initial, remaining, pore_count):
    """Counts in voxel equivalents, fractions dimensionless; no clipping."""
    mobilized = initial - remaining
    total = initial.sum(1)
    return {
        "Sor": remaining.sum(1) / pore_count,
        "RF": mobilized.sum(1) / total.clamp_min(1),
        "class_remaining": remaining,
        "class_mobilized": mobilized,
        "class_mobilized_fraction": mobilized / initial.clamp_min(1),
        "initial_oil_present": total > 0,
        "class_present": initial > 0,
    }


def remaining_oil_readout(survival, geometry, *, reachable_oil=None):
    """Differentiable dense reference. Not for memory-bounded training.

    The support is fixed by initial classes, not by thresholding predictions.
    Optional reachability support forces unchanged oil outside the k-step cone;
    it leaves the decoder and any inside-cone scores untouched.
    RF is zero with a false presence flag when there is no initial oil,
    matching the project's existing denominator convention.
    """
    validate_geometry(geometry)
    if survival.shape != geometry[:, :1].shape:
        raise ValueError("Expected [B,1,Z,Y,X] remaining-fraction scores")
    if not torch.isfinite(survival).all() or ((survival < 0) | (survival > 1)).any():
        raise ValueError("Scores must be finite in [0,1]; do not silently clip")
    if reachable_oil is not None:
        support = validate_reachable_oil(reachable_oil, geometry)
        survival = torch.where(support, survival, torch.ones_like(survival))
    oil = geometry[:, 1:] * survival
    so = oil.sum(1, keepdim=True)
    result = inventory_readout(
        geometry[:, 1:].double().flatten(2).sum(2),
        oil.double().flatten(2).sum(2),
        geometry[:, :1].double().flatten(1).sum(1))
    result.update(oil=oil, So=so, water=geometry[:, :1] - so)
    return result


def iter_remaining_oil_rollout(model, geometry, actions, times, *,
                               chunk_size=None, include_fields=False,
                               inlet=None, connected=None):
    """Lazy initial-state rollout yielding (zero-based action index, CPU readout).

    Initial state comes from the same initial-class geometry as the inventory.
    Times are caller-supplied normalized transition times; no horizon-dependent
    renormalization. No teacher forcing, target input or GPU history list.
    Consume/discard each result to bound host memory too; list(iterator) retains
    all requested CPU fields. Closing the iterator stops further computation.
    Supplying both inlet and connected enables conservative stepwise sparse
    decoding; otherwise the original full-query behavior is unchanged.
    Do not mutate model/inputs while consuming. This is inference, not training.
    """
    if model.training:
        raise ValueError("Rollout requires model.eval()")
    validate_geometry(geometry)
    if (actions.ndim != 3 or actions.shape[0] != geometry.shape[0]
            or actions.shape[2] != 4 or actions.shape[1] == 0):
        raise ValueError("Expected nonempty actions [B,T,4]")
    if times.shape != (*actions.shape[:2], 1):
        raise ValueError("Expected times [B,T,1]")
    if actions.device != geometry.device or times.device != geometry.device:
        raise ValueError("Geometry and controls must share a device")
    if not torch.isfinite(actions).all() or not torch.isfinite(times).all():
        raise ValueError("Non-finite controls")
    if ((times < 0) | (times > 1)).any() or (times[:, 1:] <= times[:, :-1]).any():
        raise ValueError("Transition times must be increasing and within [0,1]")
    if type(include_fields) is not bool:
        raise ValueError("include_fields must be boolean")
    if (inlet is None) != (connected is None):
        raise ValueError("Supply both inlet and connected, or neither")
    chunk = model.config.query_chunk if chunk_size is None else chunk_size
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk_size must be a positive integer")
    supports = (iter_reachable_oil_support(geometry, inlet=inlet, connected=connected,
                                           steps=actions.shape[1]) if inlet is not None else None)
    with torch.no_grad():
        initial_state = geometry[:, 1:].sum(1, keepdim=True)
        z = model.encode(initial_state, geometry)
    del initial_state
    previous_mobilized = None
    for step_index in range(actions.shape[1]):
        if model.training:
            raise ValueError("Model mode changed during rollout")
        with torch.no_grad():
            z = model.step(z, actions[:, step_index], times[:, step_index])
            result = stream_remaining_oil(model, z, geometry, actions[:, step_index],
                                          times[:, step_index], chunk_size=chunk,
                                          include_fields=include_fields,
                                          reachable_oil=next(supports) if supports is not None else None)
        moved = result["class_mobilized"]
        result["class_step_mobilized"] = (moved.clone() if previous_mobilized is None
                                          else moved - previous_mobilized)
        previous_mobilized = moved.clone()
        # Never yield from inside no_grad: caller's thread grad mode must survive.
        yield step_index, result
        del result, moved


def stream_remaining_oil(model, z, geometry, action, time, *,
                         chunk_size=None, include_fields=False, reachable_oil=None):
    """Decode complete volume in bounded query chunks, returning CPU tensors.

    Keeps only latent/input geometry, one query block and scalar accumulators
    on the model device. Optional So/water volumes are allocated on CPU only.
    With reachable_oil, only marked initial-oil sites are decoded; all other
    oil sites retain their initial inventory exactly. This is optional and
    valid only for the frozen monotone one-hop generator semantics.
    Inference only: no autograd, no hidden training truncation, no data IO.
    The input volume/latent still consume memory; not a measured GPU guarantee.
    """
    if model.training:
        raise ValueError("Streaming readout requires model.eval()")
    validate_geometry(geometry)
    if geometry.device != z.device or geometry.shape[0] != z.shape[0]:
        raise ValueError("Latent and geometry batch/device must match")
    chunk = model.config.query_chunk if chunk_size is None else chunk_size
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk_size must be a positive integer")
    shape = geometry.shape[2:]
    count = math.prod(shape)
    flat = geometry.flatten(2)
    batch = geometry.shape[0]
    support = (validate_reachable_oil(reachable_oil, geometry).flatten(2)
               if reachable_oil is not None else None)
    with torch.no_grad():
        initial = flat[:, 1:].sum(2, dtype=torch.float64)
        pore_count = flat[:, 0].sum(1, dtype=torch.float64)
        if support is None:
            remaining = torch.zeros_like(initial)
            so_cpu = torch.empty((batch, count), dtype=z.dtype, device="cpu") if include_fields else None
            water_cpu = torch.empty_like(so_cpu) if include_fields else None
            for start in range(0, count, chunk):
                end = min(start + chunk, count)
                i = torch.arange(start, end, device=z.device)
                xyz = torch.stack((i // (shape[1] * shape[2]),
                                   (i // shape[2]) % shape[1], i % shape[2]), -1).to(z.dtype)
                xyz = (xyz + .5) * 2 / z.new_tensor(shape) - 1
                score = model.decode(z, xyz.expand(batch, -1, -1),
                                     action, time, chunk).transpose(1, 2)
                if not torch.isfinite(score).all() or ((score < 0) | (score > 1)).any():
                    raise ValueError("Invalid decoder score")
                local_oil = score * flat[:, 1:, start:end]
                remaining += local_oil.sum(2, dtype=torch.float64)
                if include_fields:
                    so = local_oil.sum(1)
                    so_cpu[:, start:end] = so.cpu()
                    water_cpu[:, start:end] = (flat[:, 0, start:end] - so).cpu()
        else:
            # Outside the conservative reachability cone, initial oil is fixed.
            # Query only active oil sites; no full-volume decoder output is made.
            remaining = initial.clone()
            initial_so = flat[:, 1:].sum(1).to(z.dtype) if include_fields else None
            so_cpu = initial_so.cpu() if include_fields else None
            water_cpu = (flat[:, 0].to(z.dtype) - initial_so).cpu() if include_fields else None
            for b in range(batch):
                active = support[b, 0].nonzero(as_tuple=False).flatten()
                if active.numel() == 0:
                    continue
                for i in active.split(chunk):
                    xyz = torch.stack((i // (shape[1] * shape[2]),
                                       (i // shape[2]) % shape[1], i % shape[2]), -1).to(z.dtype)
                    xyz = ((xyz + .5) * 2 / z.new_tensor(shape) - 1).unsqueeze(0)
                    score = model.decode(z[b:b + 1], xyz, action[b:b + 1],
                                         time[b:b + 1], chunk)[0, :, 0]
                    if not torch.isfinite(score).all() or ((score < 0) | (score > 1)).any():
                        raise ValueError("Invalid decoder score")
                    moved = (1 - score) * flat[b, 1:][:, i]
                    remaining[b] -= moved.sum(1, dtype=torch.float64)
                    if include_fields:
                        index_cpu = i.cpu()
                        so_cpu[b, index_cpu] = score.cpu()
                        water_cpu[b, index_cpu] = (1 - score).cpu()
        result = {k: v.cpu() for k, v in inventory_readout(initial, remaining, pore_count).items()}
    if include_fields:
        result.update(So=so_cpu.reshape(batch, 1, *shape),
                      water=water_cpu.reshape(batch, 1, *shape))
    return result
