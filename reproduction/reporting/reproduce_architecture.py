"""Rebuild main Figure 2 from fixed, previously exported DEV features."""
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

STEM = "figure1_statefront_dataflow_v9_unified"
INSET_STEM = "figure1_statefront_phase_insets_v8_unified"
FEATURES = "figure1_features_fixed_dev_v5.npz"
FEATURE_META = "figure1_features_fixed_dev_v5.source.json"
EXPECTED_META = "expected_architecture_source.json"
EXPECTED_INSET = "expected_architecture_insets.json"
INPUT_NAMES = {FEATURES, FEATURE_META, EXPECTED_META, EXPECTED_INSET,
               "expected_objects.csv", "expected_edges.csv"}
FEATURE_HASH = "4a962796641bb002cea492806d74dbb5ecb434eb5886b25929113155e99ace2a"
PAIR = "z0_y256_x256|Oil|P1-22-DEV-B|S2~S3"

def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def verify_inputs(inputs):
    manifest = read_json(inputs / "manifest.json")
    if (manifest["schema_version"] != 1 or manifest["stem"] != STEM
            or set(manifest["files"]) != INPUT_NAMES):
        raise ValueError("Unexpected architecture input manifest")
    for name, record in manifest["files"].items():
        path = inputs / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Saved architecture input changed: {name}")
    meta = read_json(inputs / FEATURE_META)
    expected = read_json(inputs / EXPECTED_META)
    inset = read_json(inputs / EXPECTED_INSET)
    if (manifest["files"][FEATURES]["sha256"] != FEATURE_HASH
            or expected["feature_cache_sha256"] != FEATURE_HASH
            or inset["source_sha256"] != FEATURE_HASH
            or meta["split"] != "DEV" or meta["pair_id"] != PAIR
            or expected["same_fixed_case"] != PAIR
            or meta["seed"] != 20260811 or meta["step"] != 100
            or meta["schedule"] != "S2x20 -> S3x80"
            or meta["model"] != "RegionalStateFrontSurrogate(readout_mode=direct)"
            or meta["parameters"] != 90991
            or meta["max_difference_from_existing_prediction"] != 0
            or meta["max_readout_reconstruction_error"] != 0
            or meta["test_accessed"] or not meta["no_future_target_loaded"]):
        raise ValueError("The fixed architecture case or feature provenance differs")
    if (len(csv_rows(inputs / "expected_objects.csv")) != 73
            or len(csv_rows(inputs / "expected_edges.csv")) != 24):
        raise ValueError("Architecture object/edge reference is incomplete")
    return manifest

def current_caption(caption):
    # The original figure metadata predates the manuscript's accepted rename.
    # This changes only prose outside the exported scientific diagram.
    return caption.replace("StateGuard-RO recurrent", "DynaRO recurrent").replace(
        "StateGuard acts during training.", "State-prioritized learning acts during training.")

def compare_outputs(inputs, output):
    counts = {}
    for kind in ("objects", "edges"):
        before = csv_rows(inputs / f"expected_{kind}.csv")
        after = csv_rows(output / f"{STEM}.{kind}.csv")
        if before != after:
            raise ValueError(f"Architecture {kind} or their positions changed")
        counts[kind] = len(after)
    expected = read_json(inputs / EXPECTED_META)
    actual = read_json(output / f"{STEM}.source.json")
    excluded = {"caption", "qa_scope"}
    if {k: v for k, v in actual.items() if k not in excluded} != {
            k: v for k, v in expected.items() if k not in excluded}:
        raise ValueError("Architecture fields, scales, display or export hashes changed")
    if actual["caption"] != current_caption(expected["caption"]):
        raise ValueError("Unexpected architecture caption difference")
    if actual["qa_scope"] != (
            "Saved-output reproduction; legacy caption proof omitted; manuscript embedding unchanged"):
        raise ValueError("Unexpected rendering scope")
    if read_json(output / f"{INSET_STEM}.source.json") != read_json(inputs / EXPECTED_INSET):
        raise ValueError("Architecture volume geometry or display values changed")
    return counts

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
    import plot_architecture as figure
    figure.render(output, inputs)
    counts = compare_outputs(inputs, output)
    from matplotlib import font_manager, ft2font
    report = {
        "scope": "Fixed saved-feature illustration only; no new model forward, training or TEST evaluation",
        "figure": "main Figure 2", "same_objects_and_edges": counts,
        "all_feature_limits_and_export_hashes_match": True,
        "all_volume_metadata_match": True, "input_files": len(manifest["files"]),
        "input_manifest_sha256": digest(inputs / "manifest.json"),
        "metadata_prose_changes": ["Accepted method rename in external caption",
                                   "Legacy caption proof omitted; canonical manuscript caption is authoritative"],
        "versions": versions, "python": platform.python_version(), "platform": platform.system(),
        "font": Path(font_manager.findfont(font_manager.FontProperties(
            family=["DejaVu Sans", "Arial"]))).name, "freetype": ft2font.__freetype_version__,
        "elapsed_seconds": round(time.perf_counter()-started, 3),
        "new_training": False, "new_inference": False, "new_selection": False,
        "copied_rendered_insets": False, "model_library_loaded": "torch" in sys.modules,
    }
    if report["model_library_loaded"]:
        raise RuntimeError("Reporting must not load the model library")
    (output / "reproduction_architecture_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "inputs_architecture")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)
