"""Switched-action trajectory generation, confined to this development package."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .actions import sequence_vectors
from .source_adapter import (
    ReadOnlyStep2Adapter,
    noise_case_id,
    sha256_file,
)
from .splits import block_id_from_coord, coordinate_key


def assert_inside(child: Path, parent: Path) -> None:
    child = child.resolve()
    parent = parent.resolve()
    if not child.is_relative_to(parent):
        raise PermissionError(f"Output must stay inside package root: {parent}")


def trajectory_id(
    block_id: str,
    stage: str,
    subcore_id: str,
    sequence_id: str,
    stream_id: str,
) -> str:
    return f"{block_id}__{stage}__{subcore_id}__{sequence_id}__{stream_id}"


def short_trajectory_filename(canonical_trajectory_id: str) -> str:
    tag = hashlib.sha256(canonical_trajectory_id.encode("utf-8")).hexdigest()[:16]
    return f"traj_{tag}.npz"


def generate_trajectory(
    adapter: ReadOnlyStep2Adapter,
    label_path: Path,
    row: Mapping[str, str],
    split: str,
    sequence_id: str,
    action_ids: Sequence[str],
    stream_id: str,
    output_dir: Path,
    canonical_horizon: int = 25,
) -> Dict[str, object]:
    if split == "locked_test":
        raise PermissionError("Locked-test labels and trajectories are unavailable in development")
    if split not in {"train", "dev"}:
        raise ValueError(f"Unsupported development split: {split}")
    return _generate_trajectory_impl(
        adapter, label_path, row, split, sequence_id, action_ids,
        stream_id, output_dir, canonical_horizon,
    )


def generate_locked_test_trajectory(
    adapter: ReadOnlyStep2Adapter,
    label_path: Path,
    row: Mapping[str, str],
    sequence_id: str,
    action_ids: Sequence[str],
    stream_id: str,
    output_dir: Path,
    canonical_horizon: int = 25,
) -> Dict[str, object]:
    """Dedicated final-test caller; the development entry remains closed."""
    if stream_id not in {"P1-22-TEST-A", "P1-22-TEST-B", "P1-22-TEST-C"}:
        raise PermissionError("Only the three frozen independent TEST streams are eligible")
    return _generate_trajectory_impl(
        adapter, label_path, row, "locked_test", sequence_id, action_ids,
        stream_id, output_dir, canonical_horizon,
    )


def _generate_trajectory_impl(
    adapter: ReadOnlyStep2Adapter,
    label_path: Path,
    row: Mapping[str, str],
    split: str,
    sequence_id: str,
    action_ids: Sequence[str],
    stream_id: str,
    output_dir: Path,
    canonical_horizon: int,
) -> Dict[str, object]:
    if not action_ids:
        raise ValueError("Action sequence is empty")

    coord = coordinate_key(row)
    block_id = block_id_from_coord(coord)
    stage = str(row["sample"])
    label0 = np.load(label_path, mmap_mode="r").astype(np.uint8, copy=True)
    if label0.shape != (128, 128, 128):
        raise ValueError(f"Full-core generation requires a 128^3 source, got {label0.shape}")
    pore0 = (label0 != 0) & (label0 != 1)
    connected_pore = adapter.source.connected_to_inlet(
        pore0, inlet_axis=adapter.inlet_axis
    ).astype(np.uint8)
    labels: List[np.ndarray] = [label0]
    pressure_drop = [0.0]
    front_counts: List[int] = []
    displaced_by_class: List[List[int]] = []
    seeds: List[int] = []

    current = label0
    for t, action_id in enumerate(action_ids, start=1):
        result = adapter.step(
            label=current,
            action_id=action_id,
            block_id=block_id,
            stage=stage,
            stream_id=stream_id,
            time_step=t,
        )
        labels.append(result.label_next)
        pressure_drop.append(result.pressure_drop)
        front_counts.append(int(result.front.sum()))
        displaced_by_class.append([int(v) for v in result.displacement.reshape(5, -1).sum(axis=1)])
        seeds.append(result.noise_seed)
        current = result.label_next

    case_ids = [noise_case_id(block_id, stage, t) for t in range(1, len(action_ids) + 1)]
    tid = trajectory_id(block_id, stage, str(row["subcore_id"]), sequence_id, stream_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / short_trajectory_filename(tid)
    if npz_path.exists():
        raise FileExistsError(f"Refusing to overwrite committed trajectory: {npz_path}")
    temporary = output_dir / f".{npz_path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(
                handle,
                labels=np.stack(labels, axis=0).astype(np.uint8),
                pressure_drop=np.asarray(pressure_drop, dtype=np.float32),
                actions=sequence_vectors(action_ids, horizon_steps=canonical_horizon),
                action_index=np.asarray([int(a[1:]) - 1 for a in action_ids], dtype=np.int8),
                front_count=np.asarray(front_counts, dtype=np.int32),
                displaced_by_class=np.asarray(displaced_by_class, dtype=np.int32),
                noise_seed=np.asarray(seeds, dtype=np.uint32),
                noise_case_id=np.asarray(case_ids, dtype="U96"),
                stream_id=np.asarray(stream_id, dtype="U32"),
                pore_voxels=np.asarray(int(pore0.sum()), dtype=np.int32),
                core_voxels=np.asarray(int((label0 != 0).sum()), dtype=np.int32),
                full_shape=np.asarray(label0.shape, dtype=np.int16),
                connected_pore=connected_pore,
                inlet_axis=np.asarray(
                    {"x": 0, "y": 1, "z": 2}[adapter.inlet_axis], dtype=np.int8
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, npz_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    initial_oil = int((label0 >= 3).sum())
    final_oil = int((current >= 3).sum())
    return {
        "trajectory_id": tid,
        "split": split,
        "block_id": block_id,
        "coord": coord,
        "stage": stage,
        "subcore_id": str(row["subcore_id"]),
        "sequence_id": sequence_id,
        "stream_id": stream_id,
        "n_steps": len(action_ids),
        "canonical_horizon": canonical_horizon,
        "label_source": str(label_path.resolve()),
        "label_source_sha256": sha256_file(label_path),
        "npz_path": str(npz_path.resolve()),
        "npz_filename": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "initial_oil_voxels": initial_oil,
        "final_oil_voxels": final_oil,
        "final_rf": (initial_oil - final_oil) / initial_oil if initial_oil else None,
        "source_adapter_sha256": adapter.source_sha256,
        "noise_seed_sha256": hashlib.sha256(
            np.asarray(seeds, dtype=np.uint32).tobytes()
        ).hexdigest(),
        "noise_case_id_sha256": hashlib.sha256(
            "\n".join(case_ids).encode("utf-8")
        ).hexdigest(),
        "noise_case_id_serialization": "{block_id}|{stage}|{time_step}",
        "simulated_shape": [128, 128, 128],
        "training_crop_applied_during_generation": False,
        "counter_noise_excludes_action_sequence_split_and_crop": True,
        "case_seed_definition": (
            "uint64_big_endian(SHA256(stream_id + '|' + case_id)[0:8]) mod 2^32"
        ),
    }


def select_rows(
    metadata_rows: Iterable[Mapping[str, str]],
    split_by_block: Mapping[str, str],
    requested_split: str,
) -> List[Mapping[str, str]]:
    if requested_split == "locked_test":
        raise PermissionError("Development generator must not select locked-test rows")
    selected = []
    for row in metadata_rows:
        block_id = block_id_from_coord(coordinate_key(row))
        if split_by_block.get(block_id) == requested_split:
            selected.append(row)
    return sorted(selected, key=lambda r: (str(r["subcore_id"]), str(r["sample"])))


def write_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
