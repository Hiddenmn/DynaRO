"""Spatial-block split construction for the 80 stage-specific subcores."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


SPLIT_SALT = "P1-PATENT22-ACCF3D-SPLIT-20260811-v1"

# Exact input-side roster sealed by the result-blind Paper-1 strategy decision.
# Keeping the hashes here makes a count-preserving but role-changing split drift
# fail loudly before a label or trajectory is opened.
FROZEN_SPLIT_ROSTER = {
    "z0_y384_x384": (1, "train", "433d16d0c0ce4e138d19576d1ee14fd8d63986bfe24affbe93f2595d75ae903d"),
    "z384_y0_x384": (1, "train", "96b6a0c4ebb071f0158346dfaa93456b0122477d9d9a3a9905ee3d94ca045777"),
    "z384_y384_x384": (1, "dev", "ba5e0f6f3d05e43b489d735838593051dbcb22b8c54e2c16d4053641f6bb154a"),
    "z384_y384_x0": (1, "locked_test", "cb256ad486a18605ff9d0ee37d422cb7855a316772899d2fd3f74352621526a5"),
    "z256_y256_x128": (2, "train", "86a4e621a992fdebafc0c8f94ba99dffc5eda6a7dde356190af8c27b26aaa214"),
    "z384_y256_x384": (2, "train", "a554d2c64c72c89e0934bde9a1fbc08bc513ec3893dfdaffb752084517a90221"),
    "z512_y0_x128": (2, "dev", "b771ab0ee1edf241a94ed76eded020899440948297335617169be9415d659957"),
    "z256_y0_x128": (2, "locked_test", "dda88cd6f44797496723ccfc735b09465faf14aff3805137d4c743cb17100142"),
    "z0_y0_x128": (3, "train", "272f1df9da3118c73d49a7ac22f9e8789e26adaa5d8302f6f48b11dddc35eeb2"),
    "z128_y256_x256": (3, "train", "22e94095602c45c2bc1aab73e8bf814ab0c0a4bdb7ebc1092031035acc9b9ee5"),
    "z0_y256_x256": (3, "dev", "4660d6eddd03cb6ef954d0195a27b7ced7886ee8550eaad9d54232455757a4ac"),
    "z384_y0_x128": (3, "locked_test", "fec5729ca062fe6509f087150f0b951a54ade5cc3187165a770059d26d38dfb8"),
    "z384_y384_x256": (4, "train", "219524a8a727a5646018ec59a43819962b2039a2b44c41c2536cfd0b45379496"),
    "z640_y128_x0": (4, "train", "271570b7fbe47c563121471161ce6b215f4bcd658342f4ccec8b1d5a1a3c184a"),
    "z512_y128_x0": (4, "dev", "3c7f3a20bdf0ee65ac6ef0f43ef53333c75a54ade52183057671076b60b38761"),
    "z128_y384_x256": (4, "locked_test", "6e54108f6fbe1c52724dc736bf4c0a52d3d721161ca179270cdb3496b253e2ab"),
    "z0_y256_x128": (5, "train", "226e96248e9492f54ef2b0b365c751f59a5578dd2fcef7647d33747785830eba"),
    "z384_y128_x0": (5, "train", "0ecd30846b1f9fdfd72e221ce0e908a5c99051c56eb6012890aad4dbf91278d6"),
    "z640_y256_x128": (5, "dev", "ad7805187da9ea7bf6a2390bf4db363b3c6315784f35d34cc6d76c1942e5c78d"),
    "z512_y256_x256": (5, "locked_test", "cef72be3c5aa82ff5077a246572dae49bc3e1f3cc6b2543e2b4074755e8b223f"),
}


def coordinate_key(row: Mapping[str, str]) -> str:
    return f"{int(float(row['z0']))},{int(float(row['y0']))},{int(float(row['x0']))}"


def block_id_from_coord(coord: str) -> str:
    z, y, x = (int(v) for v in coord.split(","))
    return f"z{z}_y{y}_x{x}"


def read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"sample", "subcore_id", "z0", "y0", "x0", "oil_saturation_in_pore"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")
    return rows


def build_spatial_split(rows: Iterable[Mapping[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[coordinate_key(row)].append(row)

    blocks: List[Dict[str, object]] = []
    for coord, group in grouped.items():
        samples = sorted({str(r["sample"]) for r in group})
        mean_sor = sum(float(r["oil_saturation_in_pore"]) for r in group) / len(group)
        blocks.append(
            {
                "block_id": block_id_from_coord(coord),
                "coord": coord,
                "z0": int(coord.split(",")[0]),
                "y0": int(coord.split(",")[1]),
                "x0": int(coord.split(",")[2]),
                "mean_static_sor": mean_sor,
                "n_stages": len(group),
                "stages": ";".join(samples),
            }
        )

    if len(blocks) != 20:
        raise ValueError(f"Canonical split requires 20 spatial blocks, found {len(blocks)}")
    if any(int(b["n_stages"]) != 4 for b in blocks):
        raise ValueError("Each canonical spatial block must contain all four stages")

    blocks.sort(key=lambda b: (float(b["mean_static_sor"]), str(b["block_id"])))
    for rank, block in enumerate(blocks):
        block["sor_quintile"] = rank // 4 + 1

    allocated: List[Dict[str, object]] = []
    for quintile in range(1, 6):
        bucket = [b for b in blocks if b["sor_quintile"] == quintile]
        for b in bucket:
            token = f"{SPLIT_SALT}|{b['z0']}|{b['y0']}|{b['x0']}"
            b["salted_sha256"] = hashlib.sha256(token.encode("utf-8")).hexdigest()
        bucket.sort(key=lambda b: str(b["salted_sha256"]))
        if len(bucket) != 4:
            raise RuntimeError("Every static mean-Sor quintile must contain four blocks")
        for position, b in enumerate(bucket):
            b["within_quintile_position"] = position
            b["split"] = "train" if position < 2 else ("dev" if position == 2 else "locked_test")
            allocated.append(b)

    counts = {s: sum(b["split"] == s for b in allocated) for s in ("train", "dev", "locked_test")}
    if counts != {"train": 10, "dev": 5, "locked_test": 5}:
        raise RuntimeError(f"Unexpected split counts: {counts}")
    observed = {
        str(block["block_id"]): (
            int(block["sor_quintile"]),
            str(block["split"]),
            str(block["salted_sha256"]),
        )
        for block in allocated
    }
    if observed != FROZEN_SPLIT_ROSTER:
        missing = sorted(set(FROZEN_SPLIT_ROSTER).difference(observed))
        extra = sorted(set(observed).difference(FROZEN_SPLIT_ROSTER))
        drift = sorted(
            block
            for block in set(observed).intersection(FROZEN_SPLIT_ROSTER)
            if observed[block] != FROZEN_SPLIT_ROSTER[block]
        )
        raise RuntimeError(
            f"Frozen 20-block split parity failed: missing={missing}, extra={extra}, drift={drift}"
        )
    return sorted(allocated, key=lambda b: str(b["block_id"]))


def split_map(blocks: Iterable[Mapping[str, object]]) -> Dict[str, str]:
    return {str(b["block_id"]): str(b["split"]) for b in blocks}
