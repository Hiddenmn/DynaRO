"""Read-only adapter around the legacy Step-2 lightweight PNM proxy."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Mapping, Tuple

import numpy as np

from .actions import ACTION_LIBRARY


DEVELOPMENT_STREAM_IDS = ("P1-22-DEV-A", "P1-22-DEV-B")
OIL_CLASSES = (3, 4, 5, 6, 7)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def noise_case_id(block_id: str, stage: str, time_step: int) -> str:
    """Canonical action/order-independent full-field case serialization."""

    if time_step < 1 or time_step > 25:
        raise ValueError("time_step must be in [1,25]")
    return f"{block_id}|{stage}|{time_step}"


def stable_step_seed(stream_id: str, case_id: str) -> int:
    """Frozen CRN seed: uint64_be(SHA256(stream_id+'|'+case_id)[:8]) mod 2^32."""

    token = f"{stream_id}|{case_id}"
    first_uint64 = int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:8], "big", signed=False
    )
    return first_uint64 % (2**32)


def load_legacy_source(path: Path) -> ModuleType:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("legacy_pnm_step2_readonly", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import legacy source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "displacement_step",
        "axis_gradient",
        "estimate_pressure_drop",
        "neighbor_any",
        "connected_to_inlet",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Legacy source missing functions: {missing}")
    return module


def crop_offsets(crop_index: int) -> Tuple[int, int, int]:
    if crop_index not in range(8):
        raise ValueError("crop_index must be in [0, 7]")
    z = 64 if crop_index & 4 else 0
    y = 64 if crop_index & 2 else 0
    x = 64 if crop_index & 1 else 0
    return z, y, x


def exact_64_crop(label: np.ndarray, crop_index: int) -> np.ndarray:
    if label.shape == (64, 64, 64):
        if crop_index != 0:
            raise ValueError("A native 64^3 volume only permits crop_index=0")
        return label.astype(np.uint8, copy=True)
    if label.shape != (128, 128, 128):
        raise ValueError(f"Expected a 128^3 source or native 64^3 input, got {label.shape}")
    z, y, x = crop_offsets(crop_index)
    return label[z : z + 64, y : y + 64, x : x + 64].astype(np.uint8, copy=True)


def resolve_label_path(row: Mapping[str, str], semantic_root: Path) -> Path:
    """Resolve a label only after the caller has authorized its split."""

    candidates = [
        semantic_root / "subcores" / str(row["sample"]) / f"{row['subcore_id']}_label_uint8.npy",
        Path(str(row.get("label_path", ""))),
        Path(str(row.get("label_path_resolved", ""))),
    ]
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve read-only label for {row['subcore_id']}")


@dataclass
class StepOutput:
    label_next: np.ndarray
    pressure_drop: float
    front: np.ndarray
    displacement: np.ndarray
    noise_seed: int


class ReadOnlyStep2Adapter:
    def __init__(self, source_path: Path, inlet_axis: str = "x", base_probability: float = 0.06):
        if inlet_axis not in {"x", "y", "z"}:
            raise ValueError("inlet_axis must be x, y or z")
        self.source_path = source_path.resolve()
        self.source_sha256 = sha256_file(self.source_path)
        self.source = load_legacy_source(self.source_path)
        self.inlet_axis = inlet_axis
        self.base_probability = float(base_probability)

    def pressure_field(self, label: np.ndarray, pressure_drop: float) -> np.ndarray:
        pore = ((label != 0) & (label != 1)).astype(np.float32)
        return self.source.axis_gradient(label.shape, self.inlet_axis).astype(np.float32) * pore * pressure_drop

    def step(
        self,
        label: np.ndarray,
        action_id: str,
        block_id: str,
        stage: str,
        stream_id: str,
        time_step: int,
    ) -> StepOutput:
        pore = (label != 0) & (label != 1)
        oil = label >= 3
        water = label == 2
        inlet = np.zeros_like(water, dtype=bool)
        if self.inlet_axis == "x":
            inlet[:, :, 0] = pore[:, :, 0]
        elif self.inlet_axis == "y":
            inlet[:, 0, :] = pore[:, 0, :]
        else:
            inlet[0, :, :] = pore[0, :, :]
        connected = self.source.connected_to_inlet(pore, inlet_axis=self.inlet_axis)
        front = self.source.neighbor_any((water | inlet) & connected) & oil & connected

        case_id = noise_case_id(block_id, stage, time_step)
        seed = stable_step_seed(stream_id, case_id)
        rng = np.random.default_rng(seed)
        strategy: Dict[str, object] = ACTION_LIBRARY[action_id].source_strategy()
        label_next, _ = self.source.displacement_step(
            label=label,
            strategy=strategy,
            rng=rng,
            pressure_weight=self.source.axis_gradient(label.shape, self.inlet_axis),
            inlet_axis=self.inlet_axis,
            base_probability=self.base_probability,
        )
        displacement = np.stack(
            [((label == cls) & (label_next == 2)) for cls in OIL_CLASSES], axis=0
        ).astype(np.uint8)
        pressure_drop = float(self.source.estimate_pressure_drop(label_next, strategy))
        return StepOutput(
            label_next=label_next.astype(np.uint8),
            pressure_drop=pressure_drop,
            front=front.astype(np.uint8),
            displacement=displacement,
            noise_seed=seed,
        )
