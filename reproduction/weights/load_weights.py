"""Load an explicitly selected frozen model; never run a prediction or train."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_file(root, relative):
    require(isinstance(relative, str), "Expected a relative filename")
    parts = PurePosixPath(relative).parts
    require(relative and not relative.startswith("/") and "\\" not in relative and ":" not in relative
            and all(p not in ("", ".", "..") for p in relative.split("/")), "Unsafe relative filename")
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*parts)
    require(path.resolve(strict=True).is_relative_to(root) and path.is_file()
            and not any(p.is_symlink() or p.is_junction() for p in (path, *path.parents) if p != root),
            "Missing file or redirected path")
    return path


def private_text(value):
    patterns = (r"(?i)(?<![a-z])[a-z]:[\\/]", r"(?i)/(?:Users|home|root|autodl-tmp)/",
                r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+)",
                r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
    require(not any(re.search(p, value) for p in patterns), "Private path/contact/credential-like text")


def tensor_records(state):
    import torch
    require(type(state) is dict and state, "Expected a plain nonempty tensor dictionary")
    records = {}
    for name, tensor in state.items():
        require(isinstance(name, str) and type(tensor) is torch.Tensor
                and tensor.device.type == "cpu" and tensor.layout == torch.strided,
                "Weights must contain only named dense CPU tensors")
        private_text(name)
        require(bool(torch.isfinite(tensor).all()), "Nonfinite model tensor: " + name)
        records[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                         "sha256": hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()}
    return records


def historical_state_hash(state):
    """Original experiment definition: state-dict order, name then tensor bytes."""
    result = hashlib.sha256()
    for name, tensor in state.items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def read_manifest(root):
    root = Path(root).resolve(strict=True)
    text = safe_file(root, "manifest.json").read_text(encoding="utf-8")
    private_text(text)
    manifest = json.loads(text)
    require(manifest["format"] == "dynaro-frozen-weights-candidate-v1"
            and manifest["public_release"] is False and manifest["licence"] is None,
            "Unsupported candidate format/status")
    require(len(manifest["models"]) == 28, "Expected all 28 frozen model identities")
    expected = set(manifest["support_files"]) | {"manifest.json"} | {
        row["file"] for row in manifest["models"].values()}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == expected, "Missing or unexpected candidate file")
    for name, record in manifest["support_files"].items():
        path = safe_file(root, name)
        require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                "Changed support file: " + name)
        private_text(path.read_text(encoding="utf-8"))
    return manifest


def checked_core(core, manifest):
    core = Path(core).resolve(strict=True)
    for name, sha in manifest["model_core_files"].items():
        require(digest(safe_file(core, name)) == sha, "Model-core file differs: " + name)
    require({p.name for p in (core / "src").iterdir() if p.is_file()} == {
        PurePosixPath(name).name for name in manifest["model_core_files"] if name.startswith("src/")},
        "Unexpected model-core source file")
    source = str(core / "src")
    # Do not reuse modules imported earlier from another project checkout.
    for name in (Path(p).stem for p in manifest["model_core_files"] if p.startswith("src/")):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == core / "src" / (name + ".py"),
                    "Model module already imported from another location")
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("dynaro")


def load_frozen(core, key, root=None):
    """Return (eval-mode model, metadata), with strict CPU weight verification.

    core is the sibling model_core directory. This is an inference-state load,
    not a training-resume operation: optimizer and RNG states are not included.
    """
    import torch
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = read_manifest(root)
    require(key in manifest["models"], "Unknown model key; inspect manifest.json")
    row = manifest["models"][key]
    path = safe_file(root, row["file"])
    require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "Weight file differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(tensor_records(state) == row["tensors"], "Tensor names, shapes, dtypes or bytes differ")
    factory = checked_core(core, manifest)
    # Creating a model consumes random numbers; preserve the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        model = factory.make_model(row["model"])
    model.load_state_dict(state, strict=True)
    require(historical_state_hash(model.state_dict()) == row["original_parameter_sha256"],
            "Loaded state differs from the frozen experiment")
    require(sum(p.numel() for p in model.parameters()) == row["parameters"], "Parameter count differs")
    return model.eval().requires_grad_(False), row


def verify_all(core, root=None):
    import torch
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = read_manifest(root)
    for key in manifest["models"]:
        model, _ = load_frozen(core, key, root)
        require(not model.training and all(not p.requires_grad for p in model.parameters()), "Model not frozen")
        del model
    return {"status": "frozen_weights_verified", "models": len(manifest["models"]),
            "tensor_bytes_match_original": True, "strict_cpu_load": True,
            "forward_calls": 0, "optimizer_updates": 0, "test_pairs_scored": 0,
            "public_release": False, "torch": str(torch.__version__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True, help="Directory containing model_configs.json and src/")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    print(json.dumps(verify_all(args.core), indent=2))
