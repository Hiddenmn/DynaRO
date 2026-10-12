"""Rebuild the fixed main Figure 5 from saved DEV states and predictions."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from reporting import read_json, digest

STEM = "figure12_central64_displacement_v2"
INPUT_NAMES = {"dev_parent_prediction.npz", "dev_parent_prediction.source.json",
               "dev_parent_simulator.npz", "expected_central64_time.csv",
               "expected_central64_depth.csv", "expected_central64_contact.csv"}

def verify_inputs(inputs):
    manifest = read_json(inputs / "manifest.json")
    if manifest["schema_version"] != 1 or set(manifest["files"]) != INPUT_NAMES:
        raise ValueError("Unexpected saved 3D input manifest")
    if (manifest["stem"] != STEM
            or manifest["pair_id"] != "z0_y256_x256|NPI-012|P1-22-DEV-A|S3~S5"):
        raise ValueError("Wrong case or figure")
    for name, record in manifest["files"].items():
        path = inputs / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Saved 3D input changed: {name}")
    return manifest

def verify_versions():
    versions = {}
    for filename in ("requirements.txt", "requirements-3d.txt"):
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, wanted = line.split("==")
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed = "not installed"
            if installed != wanted:
                raise RuntimeError(f"{name}: expected {wanted}, found {installed}; "
                                   "use the recorded 3D reporting environment")
            versions[name] = installed
    return versions

def compare_quantification(inputs, output):
    counts = {}
    for kind in ("time", "depth", "contact"):
        with (inputs / f"expected_central64_{kind}.csv").open(encoding="utf-8", newline="") as f:
            expected = list(csv.reader(f))
        with (output / f"{STEM}.{kind}.csv").open(encoding="utf-8", newline="") as f:
            actual = list(csv.reader(f))
        if actual != expected:
            raise ValueError(f"The {kind} source data disagree with the saved manuscript figure")
        counts[kind] = len(actual) - 1
    return counts

def run(inputs, output):
    started = time.perf_counter()
    if sys.flags.optimize:
        raise ValueError("Do not use -O: preserved scientific checks use assertions")
    inputs, output = inputs.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing results are not overwritten")
    if output == ROOT or output.is_relative_to(inputs) or inputs.is_relative_to(output):
        raise ValueError("Output must be separate from the input directory")
    manifest = verify_inputs(inputs)
    versions = verify_versions()
    output.mkdir(parents=True)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    # The original renderer uses two CPU threads, with no CUDA/model dependency.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "2"
    import plot_central64_displacement as figure
    figure.render(output, inputs / "dev_parent_prediction.npz", inputs / "dev_parent_simulator.npz")
    counts = compare_quantification(inputs, output)
    from matplotlib import font_manager, ft2font
    report = {
        "scope": "Fixed saved-case 3D reporting only; no inference, simulation or TEST evaluation",
        "figure": "main Figure 5", "input_files": len(manifest["files"]),
        "input_manifest_sha256": digest(inputs / "manifest.json"),
        "source_csv_rows_match": counts, "versions": versions,
        "python": platform.python_version(), "platform": platform.system(),
        "font": Path(font_manager.findfont(font_manager.FontProperties(
            family=["Arial", "DejaVu Sans"]))).name, "freetype": ft2font.__freetype_version__,
        "elapsed_seconds": round(time.perf_counter()-started, 3),
        "new_inference": False, "new_simulation": False,
    }
    (output / "reproduction_3d_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "inputs_3d")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)
