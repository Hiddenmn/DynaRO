"""Verify the prepared label/manifest component without running a simulator."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_at(root, name):
    require(isinstance(name, str) and name and not name.startswith("/") and ":" not in name
            and "\\" not in name and all(p not in ("", ".", "..") for p in name.split("/")),
            "Unsafe relative filename")
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*PurePosixPath(name).parts)
    require(path.is_file() and path.resolve().is_relative_to(root) and not path.is_symlink()
            and not any(p.is_symlink() or p.is_junction() for p in path.parents if p != root),
            "Missing or redirected input file")
    return path


def private_text(text):
    patterns = (r"(?i)(?<![a-z])[a-z]:[\\/]", r"(?i)/(?:Users|home|root|autodl-tmp)/",
                r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+)",
                r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
    require(not any(re.search(p, text) for p in patterns), "Private path/contact/credential-like content")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify(root, arrays=True):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(file_at(root, "manifest.json").read_text(encoding="utf-8"))
    require(manifest["format"] == "dynaro-labels-simulator-candidate-v1"
            and manifest["public_release"] is False and manifest["licence"] is None,
            "Unsupported component format/status")
    private_text(json.dumps(manifest))
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(manifest["files"]) | {"manifest.json"}, "Missing or unexpected component file")
    for name, row in manifest["files"].items():
        path = file_at(root, name)
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "File differs: " + name)
        if path.suffix != ".npy":
            private_text(path.read_text(encoding="utf-8-sig"))
    labels = json.loads(file_at(root, "label_manifest.json").read_text(encoding="utf-8"))
    require(len(labels) == 80 and len({r["file"] for r in labels}) == 80, "Expected 80 distinct label files")
    require(Counter(r["split"] for r in labels) == {"train": 40, "dev": 20, "locked_test": 20},
            "Stage-block split sizes differ")
    lookup = {(r["block_id"], r["stage"]): r for r in labels}
    require(len(lookup) == 80, "Duplicate stage/block label identity")
    with file_at(root, "spatial_block_split.csv").open(encoding="utf-8-sig", newline="") as f:
        split = {r["block_id"]: r for r in csv.DictReader(f)}
    require(len(split) == 20, "Expected 20 spatial blocks")
    for block, row in split.items():
        for stage in ("Oil", "WI", "NPI-006", "NPI-012"):
            label = lookup[block, stage]
            require(label["split"] == row["split"] and label["origin_zyx"] == [int(row[k]) for k in ("z0", "y0", "x0")],
                    "Stage versions cross partitions or coordinates differ")
    if arrays:
        import numpy as np
        for row in labels:
            label = np.load(file_at(root, row["file"]), mmap_mode="r", allow_pickle=False)
            require(label.shape == (128, 128, 128) and label.dtype == np.uint8
                    and int(label.max()) <= 7, "Unexpected label shape/dtype/code")
            require(np.bincount(label.ravel(), minlength=8).tolist() == row["class_voxel_counts"],
                    "Initial label counts differ")
            require(hashlib.sha256(label.tobytes()).hexdigest() == row["array_sha256"], "Label array bytes differ")
            del label
    planned = {}
    for name, (expected, horizon, partition) in {
            "train25": (1600, 25, "train"), "dev25": (800, 25, "dev"),
            "test25": (1200, 25, "locked_test"), "train50": (1600, 50, "train"),
            "test100": (1200, 100, "locked_test")}.items():
        rows = read_rows(file_at(root, f"trajectory_plans/{name}.jsonl"))
        require(len(rows) == expected and len({r["file"] for r in rows}) == expected, "Trajectory plan count differs")
        for row in rows:
            label = lookup[row["block_id"], row["stage"]]
            require(row["split"] == label["split"] == partition and row["label_file"] == label["file"]
                    and row["label_source_sha256"] == manifest["files"][label["file"]]["sha256"]
                    and row["steps"] == horizon and len(row["action_ids"]) == horizon,
                    "Plan label/split/horizon differs")
            allowed_streams = {"P1-22-TEST-A", "P1-22-TEST-B", "P1-22-TEST-C"} if partition == "locked_test" else {"P1-22-DEV-A", "P1-22-DEV-B"}
            require(row["stream_id"] in allowed_streams and set(row["action_ids"]) <= {"S1", "S2", "S3", "S4", "S5"},
                    "Unknown action or simulator stream")
        planned[name] = expected
    return {"status": "labels_and_plans_verified", "labels": 80, "spatial_blocks": 20,
            "partitions": {"train": 40, "dev": 20, "locked_test": 20},
            "trajectory_plans": planned, "label_arrays_checked": arrays,
            "simulator_steps_executed": 0, "model_or_test_evaluation": "not_run", "public_release": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--metadata-only", action="store_true", help="Verify byte hashes and metadata, but do not deserialize labels")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, arrays=not args.metadata_only), indent=2))
