"""Construct the five manuscript architectures, without loading learned weights.

The numerical model definitions are preserved in the adjacent modules. This
entry selects the recorded comparison widths and direct readout explicitly;
older module defaults also contain exploratory configurations.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
from regional_budget_model import RegionalActionSurrogate, RegionalConfig
from regional_statefront_model_candidate import RegionalStateFrontSurrogate
from tnnls_remaining_oil_models import (
    ARFNO3DRemainingOil, FNOConfig, RTU3DRemainingOil, RTUConfig,
    CoarseStateFrontRemainingOil,
)


def configurations():
    """Architecture parameters only; not a replacement for TRAIN protocols."""
    return json.loads((Path(__file__).resolve().parents[1] / "model_configs.json").read_text(encoding="utf-8"))


def make_model(name="dynaro"):
    """Return a fresh, untrained CPU model at the published comparison width.

    RNG seeding is deliberately left to the caller. CPU initial parameter bytes
    need not match a historical CUDA run. This function never reads research
    arrays/checkpoints and never trains or selects a model.
    """
    configs = configurations()
    if name not in configs:
        raise ValueError("Choose dynaro, static_front, coarse_front, fno or rtu")
    record = configs[name]
    values = dict(record["architecture"])
    if name in ("dynaro", "static_front", "coarse_front"):
        config = RegionalConfig(**values)
        if name == "dynaro":
            model = RegionalStateFrontSurrogate(config, readout_mode="direct")
        elif name == "static_front":
            model = RegionalActionSurrogate(config, mode="direct")
        else:
            model = CoarseStateFrontRemainingOil(config)
        with torch.no_grad():
            model.class_logits.bias.fill_(-5.0)
    elif name == "fno":
        values["modes"] = tuple(values["modes"])
        model = ARFNO3DRemainingOil(FNOConfig(**values))
    else:
        values["positional_shape"] = tuple(values["positional_shape"])
        model = RTU3DRemainingOil(RTUConfig(**values))
    if sum(p.numel() for p in model.parameters()) != record["parameters"]:
        raise ValueError("Model parameter count differs from the manuscript configuration")
    return model
