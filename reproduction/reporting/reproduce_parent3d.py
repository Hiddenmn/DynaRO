"""Rebuild supplementary Figure S2 from the same saved parent as main Figure 5."""
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
from reproduce_3d import verify_inputs as verify_shared_inputs, verify_versions
from reporting import read_json, digest

STEM = "figure8_dev_max_mobilization_prediction_v1"
EXPECTED = "expected_parent_display.json"
# File names and elapsed time do not change any scientific or display setting.
VOLATILE_KEYS = {"source", "elapsed_seconds"}

def verify_inputs(shared, reference):
    base = verify_shared_inputs(shared)
    manifest = read_json(reference / "manifest.json")
    if (manifest["schema_version"] != 1 or manifest["stem"] != STEM
            or manifest["figure"] != "supplementary Figure S2"
            or set(manifest["files"]) != {EXPECTED}):
        raise ValueError("Unexpected full-parent reference manifest")
    record = manifest["files"][EXPECTED]
    path = reference / EXPECTED
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError("Full-parent expected data changed")
    for name, record in manifest["shared_inputs"].items():
        if base["files"].get(name) != record:
            raise ValueError("Full-parent and central-crop inputs differ")
    if set(manifest["shared_inputs"]) != {
            "dev_parent_prediction.npz", "dev_parent_prediction.source.json"}:
        raise ValueError("Wrong full-parent input dependencies")
    expected = read_json(path)
    if (expected["pair_id"] != base["pair_id"]
            or expected["source_sha256"] != base["files"]["dev_parent_prediction.npz"]["sha256"]):
        raise ValueError("The parent and crop no longer refer to the same saved case")
    return manifest, expected

def compare_quantification(actual, expected):
    keys = set(expected) - VOLATILE_KEYS
    if set(actual) - VOLATILE_KEYS != keys:
        raise ValueError("Full-parent display metadata fields differ")
    for key in sorted(keys):
        if actual[key] != expected[key]:
            raise ValueError(f"Full-parent data or display setting differs: {key}")
    if (len(actual["curve"]["steps"]) != 101
            or len(actual["curve"]["site_brier_steps1_100"]) != 100
            or len(actual["depth"]["initial_oil"]) != 8):
        raise ValueError("Incomplete full-parent quantitative panels")
    return {"time": 101, "brier": 100, "depth": 8}

def write_source_data(metadata, output):
    # Values come only from the freshly computed metadata, never the reference file.
    curve, depth = metadata["curve"], metadata["depth"]
    tables = {
        "time": (
            ["step", "simulator_mobilized_percent", "prediction_expected_mobilized_percent"],
            zip(curve["steps"], curve["simulator_percent"], curve["prediction_percent"])),
        "brier": (
            ["step", "site_brier_raw", "site_brier_display_x1000"],
            ((i, value, 1000 * value)
             for i, value in enumerate(curve["site_brier_steps1_100"], 1))),
        "depth": (
            ["x_start_inclusive", "x_stop_exclusive", "x_center", "initial_oil_voxels",
             "simulator_mobilized_percent", "prediction_expected_mobilized_percent"],
            ((a, b, (a+b-1)/2, n, t, p)
             for a, b, n, t, p in zip(depth["edges"][:-1], depth["edges"][1:],
                 depth["initial_oil"], depth["simulator_percent"], depth["prediction_percent"]))),
    }
    for name, (header, rows) in tables.items():
        with (output / f"{STEM}.{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

def run(shared, reference, output):
    started = time.perf_counter()
    if sys.flags.optimize:
        raise ValueError("Do not use -O: preserved scientific checks use assertions")
    shared, reference, output = shared.resolve(), reference.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing results are not overwritten")
    if output == ROOT or any(output.is_relative_to(p) or p.is_relative_to(output)
                             for p in (shared, reference)):
        raise ValueError("Output must be separate from the input directories")
    manifest, expected = verify_inputs(shared, reference)
    versions = verify_versions()
    output.mkdir(parents=True)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "2"
    import plot_parent128_displacement as figure
    actual = figure.draw(output / STEM, shared / "dev_parent_prediction.npz")
    rows = compare_quantification(actual, expected)
    write_source_data(actual, output)
    from matplotlib import font_manager, ft2font
    report = {
        "scope": "Fixed saved full-parent reporting only; no new inference, simulation or TEST access",
        "figure": manifest["figure"], "shared_case_with": "main Figure 5",
        "shared_input_manifest_sha256": digest(shared / "manifest.json"),
        "reference_manifest_sha256": digest(reference / "manifest.json"),
        "matched_source_rows": rows, "all_stable_display_metadata_match": True,
        "versions": versions, "python": platform.python_version(), "platform": platform.system(),
        "font": Path(font_manager.findfont(font_manager.FontProperties(
            family=["Arial", "DejaVu Sans"]))).name, "freetype": ft2font.__freetype_version__,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "new_inference": False, "new_simulation": False,
    }
    (output / "reproduction_parent3d_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "inputs_3d")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "inputs_parent3d")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.reference_dir, args.output_dir)
