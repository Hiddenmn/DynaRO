"""Verify the local training component without importing a numerical runtime."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def safe_file(root, name):
    root = Path(root).resolve(strict=True)
    rel = Path(name)
    require(not rel.is_absolute() and rel.parts and ".." not in rel.parts, "Unsafe relative path")
    p = root
    for part in rel.parts:
        p = p / part
        require(not p.is_symlink() and not p.is_junction(), "Redirected input")
    p = p.resolve(strict=True)
    require(p.is_relative_to(root) and p.is_file(), "Input outside component")
    return p


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(safe_file(root, "manifest.json").read_text(encoding="utf-8"))
    require(manifest["format"] == "dynaro-training-candidate-v1", "Unknown training format")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    require(actual == set(manifest["files"]) | {"manifest.json"}, "Unexpected or missing training file")
    for name, row in manifest["files"].items():
        p = safe_file(root, name)
        require(p.stat().st_size == row["bytes"] and digest(p) == row["sha256"], "Changed component file: " + name)
    protocol = json.loads(safe_file(root, "protocol.json").read_text(encoding="utf-8"))
    require(protocol["format"] == "dynaro-training-protocol-v1" and len(protocol["configurations"]) == 28,
            "Incomplete training protocol")
    return {"status": "training_component_files_match", "files": len(manifest["files"]),
            "training_configurations": 28, "optimizer_updates": 0, "manifest_sha256": digest(root / "manifest.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(p.parse_args().root), indent=2))
