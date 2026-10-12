"""Rebuild supplementary Figure S1 from its fixed saved TRAIN trajectory."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reproduce_3d import verify_versions
from reporting import read_json, digest

STEM = "figure2_train50_interior_v8"
DATA = "train50_NPI-012_z384_y384_x256_streamA_S320_then_S530.npz"
SELECTION = "train_selection.json"
EXPECTED_META = "expected_train_source.json"
EXPECTED_COUNTS = "expected_train_counts.csv"
INPUT_NAMES = {DATA, SELECTION, EXPECTED_META, EXPECTED_COUNTS}

def verify_inputs(inputs):
    manifest = read_json(inputs / "manifest.json")
    if (manifest["schema_version"] != 1 or manifest["stem"] != STEM
            or set(manifest["files"]) != INPUT_NAMES):
        raise ValueError("Unexpected training-figure input manifest")
    for name, record in manifest["files"].items():
        path = inputs / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Saved training input changed: {name}")
    selection, expected = read_json(inputs / SELECTION), read_json(inputs / EXPECTED_META)
    chosen = selection["selected"]
    if (chosen["source"] != DATA or (chosen["z"], chosen["y"], chosen["x"]) != (40, 16, 16)
            or chosen["width"] != 64 or chosen["initial_oil"] != 18396
            or chosen["converted"] != 3516 or chosen["interior_converted"] != 1616):
        raise ValueError("The fixed training case or crop differs")
    if (selection["selected_source_sha256"] != manifest["files"][DATA]["sha256"]
            or expected["source_sha256"] != selection["selected_source_sha256"]
            or expected["selection"] != selection
            or expected["schedule"] != "S3 x20 then S5 x30"
            or expected["display_steps"] != [0, 20, 50]):
        raise ValueError("Training selection and source metadata disagree")
    return manifest

def compare_quantification(inputs, output):
    with (inputs / EXPECTED_COUNTS).open(encoding="utf-8", newline="") as handle:
        expected_rows = list(csv.reader(handle))
    with (output / f"{STEM}.source.csv").open(encoding="utf-8", newline="") as handle:
        actual_rows = list(csv.reader(handle))
    if actual_rows != expected_rows or len(actual_rows) != 52:
        raise ValueError("Training source counts disagree with the original 51 states")
    expected = read_json(inputs / EXPECTED_META)
    actual = read_json(output / f"{STEM}.source.json")
    # A relocated source has a portable filename rather than the original path.
    if {k: v for k, v in actual.items() if k != "source"} != {
            k: v for k, v in expected.items() if k != "source"}:
        raise ValueError("Training figure metadata or display settings differ")
    return len(actual_rows) - 1

def run(inputs, output):
    started = time.perf_counter()
    if sys.flags.optimize:
        raise ValueError("Do not use -O: preserved scientific checks use assertions")
    inputs, output = inputs.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing results are not overwritten")
    if output == ROOT or output.is_relative_to(inputs) or inputs.is_relative_to(output):
        raise ValueError("Output must be separate from inputs")
    manifest = verify_inputs(inputs)
    versions = verify_versions()
    output.mkdir(parents=True)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "2"
    import plot_train50_displacement as figure
    figure.render(output, inputs / SELECTION)
    rows = compare_quantification(inputs, output)
    from matplotlib import font_manager, ft2font
    report = {
        "scope": "Fixed saved TRAIN illustration only; no training, inference, simulation or TEST access",
        "figure": "supplementary Figure S1", "source_csv_rows_match": rows,
        "all_stable_source_metadata_match": True, "input_files": len(manifest["files"]),
        "input_manifest_sha256": digest(inputs / "manifest.json"),
        "versions": versions, "python": platform.python_version(), "platform": platform.system(),
        "font": Path(font_manager.findfont(font_manager.FontProperties(
            family=["Arial", "DejaVu Sans"]))).name, "freetype": ft2font.__freetype_version__,
        "elapsed_seconds": round(time.perf_counter()-started, 3),
        "new_training": False, "new_inference": False, "new_simulation": False,
    }
    (output / "reproduction_train3d_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "inputs_train3d")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)
