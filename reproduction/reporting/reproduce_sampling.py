"""Rebuild main Figure 1 from the unchanged semantic stack and saved DEV input."""
from __future__ import annotations
import argparse
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

STEM = "figure13_sampling_hierarchy_v3"
FULL = "NPI-012_label_uint8.npy"
SLICE = "NPI-012-Reconstructed0000.filtered000_semantic_8bit_palette.tif"
EXPECTED = "expected_sampling_source.json"
CASE = "dev_parent_prediction.npz"
INPUT_NAMES = {FULL, SLICE, EXPECTED}
FULL_HASH = "964932bad9743649955ce9e1cd29056da40985c587185a6752a3469dd425023b"
CASE_HASH = "a938cd5e8f4b5a36182a7829b41804ac8881d5c69a4447abcf8c3727b7870a4f"

def verify_inputs(inputs, saved_3d):
    manifest = read_json(inputs / "manifest.json")
    if (manifest["schema_version"] != 1 or manifest["stem"] != STEM
            or set(manifest["files"]) != INPUT_NAMES
            or set(manifest["shared_files"]) != {CASE}):
        raise ValueError("Unexpected sampling-figure manifest")
    for base, records in ((inputs, manifest["files"]), (saved_3d, manifest["shared_files"])):
        for name, record in records.items():
            path = base / name
            if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError(f"Saved sampling input changed: {name}")
    expected = read_json(inputs / EXPECTED)
    if (manifest["files"][FULL]["sha256"] != FULL_HASH
            or manifest["shared_files"][CASE]["sha256"] != CASE_HASH
            or expected["full_source_sha256"] != FULL_HASH
            or expected["case_sha256"] != CASE_HASH
            or expected["parent_origin_zyx"] != [0, 256, 256]
            or expected["interior_parent_bounds_zyx"] != [[32, 96]]*3
            or expected["interior_stack_bounds_zyx"] != [[32, 96], [288, 352], [288, 352]]
            or expected["panels"]["b"]["native_oil_voxels"] != 295851
            or expected["panels"]["c"]["native_oil_voxels"] != 35740):
        raise ValueError("The fixed sampling identity or crop differs")
    return manifest

def compare_metadata(inputs, output):
    expected = read_json(inputs / EXPECTED)
    actual = read_json(output / f"{STEM}.source.json")
    excluded = {"full_source", "cached_case"}
    if {k: v for k, v in actual.items() if k not in excluded} != {
            k: v for k, v in expected.items() if k not in excluded}:
        raise ValueError("Sampling geometry or display metadata differ from the original figure")
    return True

def run(inputs, saved_3d, output):
    started = time.perf_counter()
    if sys.flags.optimize:
        raise ValueError("Do not use -O: preserved scientific checks use assertions")
    inputs, saved_3d, output = inputs.resolve(), saved_3d.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing results are not overwritten")
    if output == ROOT or any(output.is_relative_to(p) or p.is_relative_to(output)
                             for p in (inputs, saved_3d)):
        raise ValueError("Output must be separate from inputs")
    manifest = verify_inputs(inputs, saved_3d)
    versions = verify_versions()
    output.mkdir(parents=True)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "2"
    import plot_sampling_hierarchy as figure
    figure.render(output, inputs / FULL, saved_3d / CASE, inputs / SLICE)
    compare_metadata(inputs, output)
    from matplotlib import font_manager, ft2font
    report = {
        "scope": "Fixed semantic-stack sampling illustration; all panels newly rendered from saved labels",
        "figure": "main Figure 1", "all_stable_source_metadata_match": True,
        "input_files": len(manifest["files"]), "shared_input_files": len(manifest["shared_files"]),
        "input_manifest_sha256": digest(inputs / "manifest.json"),
        "versions": versions, "python": platform.python_version(), "platform": platform.system(),
        "font": Path(font_manager.findfont(font_manager.FontProperties(
            family=["Arial", "DejaVu Sans"]))).name, "freetype": ft2font.__freetype_version__,
        "elapsed_seconds": round(time.perf_counter()-started, 3),
        "new_training": False, "new_inference": False, "new_simulation": False,
        "new_selection": False, "reused_render_cache": False,
    }
    (output / "reproduction_sampling_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "inputs_sampling")
    parser.add_argument("--saved-3d-dir", type=Path, default=ROOT / "inputs_3d")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.saved_3d_dir, args.output_dir)
