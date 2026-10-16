"""Load only the original predeclared, untrained CPU random states.

Raw float32 buffers avoid pickle/torch.load and are bound to the frozen seed
hashes. No research data, learned checkpoint, optimizer or forward pass.
"""
import hashlib
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
SEEDS = (20260811, 20260812, 20260813)


def _read_locked(root, relative, expected, size):
    path = root / relative
    if (Path(relative).name != relative or path.is_symlink()
            or path.resolve().parent != root.resolve() or path.stat().st_size != size):
        raise PermissionError("Invalid frozen initialization artifact")
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected:
        raise PermissionError("Frozen initialization artifact hash differs")
    return value


def restore_initialization(model, seed, expected, *, directory=HERE):
    from source_helpers import model_hash
    if seed not in SEEDS or sys.byteorder != "little":
        raise PermissionError("Unsupported frozen seed/platform")
    root = Path(directory)
    spec = json.loads((root / "frozen_initial_states.json").read_text(encoding="utf-8"))
    if spec["kind"] != "untrained_random_initial_states" or spec["optimizer_updates"] != 0:
        raise PermissionError("Not a from-scratch initialization")
    entry = spec["seeds"][str(seed)]
    if entry["parameter_sha256"] != expected:
        raise PermissionError("Initialization must retain original plan hash")
    raw = _read_locked(root, entry["file"], entry["file_sha256"], entry["bytes"])
    rng = _read_locked(root, entry["rng_file"], entry["rng_sha256"], entry["rng_bytes"])
    state = model.state_dict()
    tensors = entry["tensors"]
    if list(state) != [e["name"] for e in tensors]:
        raise PermissionError("Initialization tensor names/order differ")
    offset = 0
    with torch.no_grad():
        for item in tensors:
            dest = state[item["name"]]
            size = dest.numel() * 4
            if (dest.device.type != "cpu" or dest.dtype != torch.float32
                    or list(dest.shape) != item["shape"] or item["bytes"] != size):
                raise PermissionError("Initialization shape/dtype/device differs")
            if offset + size > len(raw):
                raise PermissionError("Initialization buffer is truncated")
            source = torch.frombuffer(bytearray(raw[offset:offset+size]), dtype=torch.float32)
            dest.copy_(source.reshape(dest.shape))
            offset += size
    if offset != len(raw) or model_hash(model) != expected:
        raise PermissionError("Loaded parameters differ from original frozen state")
    torch.set_rng_state(torch.frombuffer(bytearray(rng), dtype=torch.uint8))
    return {"source": "original_frozen_random_state", "seed": seed,
            "parameter_sha256": expected, "artifact_sha256": entry["file_sha256"]}
