"""Standard-library check of the explicit retrained-final evaluation route."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def safe_file(root, name):
    root = Path(root).resolve(strict=True)
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            "Expected a safe relative filename")
    parts = name.split("/")
    require(all(p not in ("", ".", "..") for p in parts), "Unsafe relative path")
    path = root
    for part in parts:
        path /= part
        require(not path.is_symlink() and not path.is_junction(), "Redirected input")
    path = path.resolve(strict=True)
    require(path.is_file() and path.is_relative_to(root), "Input outside component")
    return path


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(safe_file(root, "manifest.json").read_text(encoding="utf-8"))
    require(manifest["format"] == "dynaro-retrained-evaluation-candidate-v1", "Unknown bridge format")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    require(actual == set(manifest["files"]) | {"manifest.json"}, "Unexpected or missing bridge file")
    for name, row in manifest["files"].items():
        path = safe_file(root, name)
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "Changed bridge file")
    return {"status": "retrained_evaluation_component_files_match", "files": len(manifest["files"]),
            "manifest_sha256": digest(root / "manifest.json"), "pairs_scored": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(p.parse_args().root), indent=2))
