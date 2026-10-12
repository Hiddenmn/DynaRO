"""Standard-library integrity and structural checks for this local candidate."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import runpy
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def private_text(text):
    patterns = (r"(?i)(?<![a-z])[a-z]:[\\/]", r"(?i)/(?:Users|home|root|autodl-tmp)/",
                r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+)",
                r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
    require(not any(re.search(p, text) for p in patterns), "Private path/contact/credential-like content")


def relative(value):
    p = PurePosixPath(value)
    require(isinstance(value, str) and value and not p.is_absolute() and "\\" not in value
            and ":" not in value and not any(part in ("", ".", "..") for part in value.split("/")),
            "Unsafe candidate filename")
    return Path(*p.parts)


def import_closure(files):
    modules = {PurePosixPath(n).stem for n in files if n.startswith("model_core/src/") and n.endswith(".py")}
    dependencies = set(sys.stdlib_module_names) | modules | {"torch", "numpy"}
    for name, raw in files.items():
        if not name.endswith(".py"):
            continue
        tree = ast.parse(raw.decode("utf-8-sig"), filename=name)
        compile(tree, name, "exec")
        if name.startswith("model_core/src/"):
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = {v.name.split(".")[0] for v in node.names}
                elif isinstance(node, ast.ImportFrom):
                    require(node.level == 0 and bool(node.module), "Unexpected relative core import")
                    imported = {node.module.split(".")[0]}
                else:
                    continue
                require(imported <= dependencies, "Unbundled scientific dependency: " + name)
    return len(modules)


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["format"] == "dynaro-local-release-candidate-v1"
            and manifest["public_release"] is False and manifest["complete_reproduction_release"] is False
            and manifest["licence"] is None, "Unsupported candidate status")
    listed = manifest["files"]
    require(isinstance(listed, dict) and "manifest.json" not in listed, "Invalid file manifest")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(listed) | {"manifest.json"}, "Missing or unexpected candidate files")
    files = {}
    for name, record in listed.items():
        path = root / relative(name)
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Escaping/redirection path")
        raw = path.read_bytes()
        require(len(raw) == record["bytes"] and sha256(raw) == record["sha256"], "Changed file: " + name)
        private_text(raw.decode("utf-8-sig"))
        files[name] = raw
    private_text((root / "manifest.json").read_text(encoding="utf-8"))
    core_modules = import_closure(files)
    checker = runpy.run_path(str(root / "source_data/verify_source_data.py"))
    data = checker["verify"](root / "source_data")
    return {"status": "local_candidate_verified", "files": len(actual),
            "core_modules": core_modules, "source_data": data,
            "scientific_modules_imported": 0, "research_arrays_or_weights_loaded": 0,
            "model_execution": "not_run", "public_release": False,
            "complete_reproduction_release": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
