"""TRAIN-only portable loader copied from the verified 2026-10-02 server loader.

No original data/index is modified. Only seed 20260811 is enabled here.
Panel, action, metadata, trajectory and pairing checks are retained verbatim.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch

RUNS = Path("workflow_full_audit_20260717/paper1_workspace/runs")
TRAIN25 = RUNS / "patent22_conservative_dynamic_surrogate_v1_20260811/data/canonical_train_full128"
TRAIN50 = RUNS / "stateguard_horizon50_train_data_20260928"
PARAMETERS = {"fno": 4196289, "rtu": 2103565}
SEEDS = (20260811,)
UPDATES = 400
PAIRS = 800
INDEX25_SHA = "0997bd47e5b94a2ad7c16cd5c48a57c63502b2b2cb764257fbfdbce88f44ff99"
# The generator's files_sha256 is a hash of sorted FILENAMES, not file contents.
NAMES50_SHA = "2e89dc112cae2edc1d0b8d0a1685dbb69c5d9598cef153c3fbd4cc1f88601a3f"
ORDER11_SHA = "5532780c999982289c36163361b851ad2f52017bef4a46560cf092445751e742"
STREAMS = {"P1-22-DEV-A", "P1-22-DEV-B"}  # identifiers retained from canonical TRAIN
ACTION_VECTORS = np.asarray([
    [0.65 / 1.45, 0, 0, 0], [1 / 1.45, 0, 0, 0], [1.45 / 1.45, 0, 0, 0],
    [1 / 1.45, 0.06 / 0.12, 0.16 / 0.28, 1],
    [1 / 1.45, 0.12 / 0.12, 0.28 / 0.28, 1],
], dtype=np.float32)

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_path(directory, filename):
    if Path(filename).name != filename or not filename.endswith(".npz"):
        raise ValueError("Expected a plain trajectory filename")
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    return path


def panel(root):
    """Metadata-only reconstruction of the existing paired_panel sorted keys."""
    index = root / TRAIN25 / "trajectory_index.jsonl"
    if file_hash(index) != INDEX25_SHA:
        raise ValueError("The frozen TRAIN index differs")
    manifest25 = json.loads((root / TRAIN25 / "run_manifest.json").read_text(encoding="utf-8"))
    if (manifest25.get("split") != "train"
            or manifest25.get("canonical_complete") is not True
            or manifest25.get("storage_kind") != "full_trajectory_128"
            or manifest25.get("locked_test_accessed") is not False):
        raise ValueError("Require the completed canonical TRAIN manifest")
    records = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest50 = json.loads((root / TRAIN50 / "manifest.json").read_text(encoding="utf-8"))
    if (len(records) != 1600
            or manifest50.get("scope") != "complete_50_step_canonical_TRAIN_extension_only"
            or manifest50.get("pairs") != PAIRS
            or manifest50.get("trajectories") != 1600
            or manifest50.get("horizon_steps") != 50):
        raise ValueError("Require the existing complete 25/50-step TRAIN sets")
    grouped = {25: {}, 50: {}}
    for record in records:
        if (record.get("split") != "train" or record.get("n_steps") != 25
                or record.get("stream_id") not in STREAMS):
            raise ValueError("Index contains a non-canonical TRAIN record")
        match = re.fullmatch(r"pair_(S[1-5])_(S[1-5])__((?:S[1-5])(?:10|15)_then_(?:S[1-5])(?:10|15))", record["sequence_id"])
        if not match:
            raise ValueError("Unknown canonical action protocol")
        a, b, schedule = match.groups()
        if a >= b:
            raise ValueError("Canonical action-pair ordering differs")
        forward = f"{a}10_then_{b}15"
        reverse = f"{b}15_then_{a}10"
        if schedule not in (forward, reverse):
            raise ValueError("TRAIN25 is not the count-matched reversal")
        key = f"{record['block_id']}|{record['stage']}|{record['stream_id']}|{a}~{b}"
        slot = grouped[25].setdefault(key, {})
        side = 0 if schedule == forward else 1
        if side in slot:
            raise ValueError("Duplicate TRAIN25 schedule")
        slot[side] = record
    names50 = []
    for record in manifest50["entries"]:
        match = re.search(r"_(S[1-5])(20|30)_then_(S[1-5])(20|30)\.npz$", record["file"])
        if not match or record["stream_id"] not in STREAMS:
            raise ValueError("Unknown TRAIN50 schedule")
        first, n_first, second, n_second = match.groups()
        a, b = sorted((first, second))
        if a == b or {n_first, n_second} != {"20", "30"}:
            raise ValueError("Invalid paired action counts")
        side = 0 if first == a else 1
        if (n_first, n_second) != (("20", "30") if side == 0 else ("30", "20")):
            raise ValueError("TRAIN50 action doses differ")
        key = f"{record['block_id']}|{record['stage']}|{record['stream_id']}|{a}~{b}"
        slot = grouped[50].setdefault(key, {})
        if side in slot:
            raise ValueError("Duplicate TRAIN50 schedule")
        slot[side] = record
        names50.append(record["file"])
    actual_names_hash = hashlib.sha256("\n".join(sorted(names50)).encode()).hexdigest()
    if actual_names_hash != NAMES50_SHA or manifest50["files_sha256"] != NAMES50_SHA:
        raise ValueError("TRAIN50 filename panel differs")
    if (len(grouped[25]) != PAIRS or set(grouped[25]) != set(grouped[50])
            or any(set(sides) != {0, 1} for groups in grouped.values() for sides in groups.values())):
        raise ValueError("The TRAIN25/50 pair keys do not match")
    rows = []
    for key in sorted(grouped[25]):
        row = {"key": key}
        for horizon in (25, 50):
            sides = grouped[horizon][key]
            directory = root / (TRAIN25 if horizon == 25 else TRAIN50 / "data")
            field = "npz_filename" if horizon == 25 else "file"
            row[horizon] = tuple((data_path(directory, sides[side][field]), sides[side]) for side in (0, 1))
        if row[25][0][1]["noise_seed_sha256"] != row[25][1][1]["noise_seed_sha256"]:
            raise ValueError("TRAIN25 reversal does not share the canonical random field")
        rows.append(row)
    if len({row["key"].split("|")[0] for row in rows}) != 10:
        raise ValueError("Require exactly ten TRAIN spatial blocks")
    return rows


def pair_order(rows, seed):
    order = torch.randperm(PAIRS, generator=torch.Generator().manual_seed(seed)).tolist()
    digest = hashlib.sha256("\n".join(rows[index]["key"] for index in order).encode()).hexdigest()
    if seed == SEEDS[0] and digest != ORDER11_SHA:
        raise ValueError("TRAIN pair order differs from the matched experiment")
    return order, digest


def read_trajectory(path, record, horizon):
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"]
        connected = data["connected_pore"].astype(bool, copy=True)
        indexes = data["action_index"].astype(np.int64, copy=True)
        if (labels.dtype != np.uint8 or labels.shape != (horizon + 1, 128, 128, 128)
                or connected.shape != (128, 128, 128) or indexes.shape != (horizon,)
                or labels.max() > 7 or indexes.min() < 0 or indexes.max() > 4):
            raise ValueError(f"Invalid TRAIN trajectory: {path.name}")
        initial = labels[0].copy()
        target = torch.from_numpy((labels >= 3).astype(np.uint8))
        pore = (initial != 0) & (initial != 1)
        if np.any(connected & ~pore) or np.any(((labels != 0) & (labels != 1)) != pore[None]):
            raise ValueError("Pore support changed")
        if horizon == 25:
            if int(data["inlet_axis"]) != 0:
                raise ValueError("Matched training uses the x inlet")
            if not np.array_equal(data["actions"], ACTION_VECTORS[indexes]):
                raise ValueError("Stored four-dimensional actions differ")
            if str(data["stream_id"].item()) != record["stream_id"]:
                raise ValueError("Stored TRAIN stream differs")
    if horizon == 25:
        sequence = record["sequence_id"].split("__")[1]
    else:
        sequence = re.search(r"_(S[1-5](?:20|30)_then_S[1-5](?:20|30))\.npz$", path.name).group(1)
    first, n_first, second, n_second = re.fullmatch(r"S([1-5])(10|15|20|30)_then_S([1-5])(10|15|20|30)", sequence).groups()
    expected = np.asarray([int(first) - 1] * int(n_first) + [int(second) - 1] * int(n_second))
    if not np.array_equal(indexes, expected):
        raise ValueError("Stored actions do not match the frozen schedule")
    return initial, connected, indexes, target


def load_pair(row, horizon, device):
    first, second = [read_trajectory(path, metadata, horizon) for path, metadata in row[horizon]]
    if (not np.array_equal(first[0], second[0]) or not np.array_equal(first[1], second[1])
            or not np.array_equal(np.bincount(first[2], minlength=5), np.bincount(second[2], minlength=5))):
        raise ValueError("Counterfactual pair is not count/geometry matched")
    labels = torch.from_numpy(first[0]).to(device)
    connected = torch.from_numpy(first[1]).to(device)
    inlet = torch.zeros_like(connected)
    inlet[:, :, 0] = connected[:, :, 0]
    actions = [torch.from_numpy(ACTION_VECTORS[values[2]].copy()).to(device) for values in (first, second)]
    steps = torch.arange(1, horizon + 1, device=device, dtype=torch.float32)
    return labels, inlet, connected, *actions, (steps / (steps + 25)).unsqueeze(1), first[3], second[3]


