"""Unify two architecture volume insets; retain every cached tensor and edge.

This is a local adaptation of the existing 180 x 128 mm dataflow composition.
The real DEV input and cached prediction stay fixed. Only their surface display
uses the native-grid lighting, phase colours and camera of the interior plates.
No model is loaded and no inference, training or TEST access is performed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "2"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

import parent_volume_renderer as surface
import architecture_drawing as architecture
import numpy as np
import psutil
from matplotlib.colors import to_hex

STEM = "figure1_statefront_dataflow_v9_unified"
INSET_STEM = "figure1_statefront_phase_insets_v8_unified"
SOURCE_SHA256 = "4a962796641bb002cea492806d74dbb5ecb434eb5886b25929113155e99ace2a"
WATER_OPACITY = 0.12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene(oil, water):
    # Shared native mesh vertices, interpolated normals, depth buffer and lights.
    # Water is a subdued display context, not a change in phase labels.
    result = surface.render(oil)
    camera = result["camera"]
    wd, wn, wc = surface.geometry_buffer(surface.mesh(water, 2), camera)
    visible = (wc == 2) & (wd >= result["depth"])
    shade = .74 + .26 * np.maximum(wn @ surface.LIGHT, 0)
    blue = surface.PALETTE[2][None, None, :] * shade[..., None]
    rgb = result["rgb"]
    rgb[visible] = (1 - WATER_OPACITY) * rgb[visible] + WATER_OPACITY * blue[visible]
    lines = architecture.visible_wire(camera, result["depth"])
    record = {"oil_voxels": int(oil.sum()), "water_voxels": int(water.sum()),
              "oil_triangles": result["triangles"],
              "image_shape_yxc": list(rgb.shape), "camera_center_xyz": camera["center"].tolist(),
              "camera_span": np.asarray(camera["span"]).tolist()}
    del wd, wn, wc, blue, result
    gc.collect()
    return rgb, lines, record


def build_insets(out):
    features = architecture.FEATURES
    assert digest(features) == SOURCE_SHA256, "The fixed feature cache has changed"
    with np.load(features, allow_pickle=False) as source:
        labels = source["initial_labels"]
        occupancy = source["occupancy"]
        step = int(source["display_step"])
    assert labels.shape == occupancy.shape == (128, 128, 128)
    assert not np.any(labels == 0) and step == 100
    assert np.isfinite(occupancy).all() and occupancy.min() >= 0 and occupancy.max() <= 1
    oil0 = labels >= 3
    final_oil = oil0 & (occupancy >= .5)
    initial_water = labels == 2
    final_water = initial_water | (oil0 & ~final_oil)
    assert int(oil0.sum()) == 460312 and int(final_oil.sum()) == 457150
    surface.PIXELS = 1200
    print("Rendering fixed architecture input on the native grid", flush=True)
    initial, initial_wire, initial_record = scene(oil0, initial_water)
    print("Rendering its cached step-100 prediction on the same grid and camera", flush=True)
    output, output_wire, output_record = scene(final_oil, final_water)
    assert initial.shape == output.shape
    assert initial_record["camera_span"] == output_record["camera_span"]
    path = out / (INSET_STEM + ".npz")
    np.savez_compressed(path, initial=initial, output=output,
                        initial_wire=initial_wire, output_wire=output_wire)
    metadata = {
        "source": features.name, "source_sha256": SOURCE_SHA256,
        "source_shape_zyx": list(labels.shape), "display_step": step,
        "view": "Complete unchanged 128-cubed architecture DEV example",
        "camera": surface.CAMERA,
        "palette": {k: to_hex(colour).upper() for k, colour in enumerate(surface.PALETTE)},
        "phase_opacity": {"oil": 1.0, "water": WATER_OPACITY, "rock": 0.0},
        "surface": "Native level-0.5 marching cubes; smooth-normal lighting, self-shadow and local occlusion",
        "geometry_smoothing": False, "geometry_decimation": False, "display_stride": 1,
        "surface_points_overlay": False, "output_display_threshold": .5,
        "output_water": "Initial water plus initial oil sites with cached predicted occupancy below 0.5",
        "quantification": "Continuous cached probabilities remain unaltered; thresholds are for geometry only",
        "initial_class_counts": np.bincount(labels.ravel(), minlength=8).tolist(),
        "continuous_output_oil": float(occupancy[oil0].sum(dtype=np.float64)),
        "records": {"initial": initial_record, "output": output_record},
        # Upstream renderer sources; portable module names are recorded by the runner.
        "renderer_source": ["render_dev_interior_displacement.py", "render_multiphase_depth_v5.py",
                            "render_architecture_dataflow_v9_unified.py"],
        "new_inference": False, "test_accessed": False,
    }
    path.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    assert digest(features) == SOURCE_SHA256
    return path


def render(out, inputs):
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    architecture.FEATURES = inputs / "figure1_features_fixed_dev_v5.npz"
    architecture.SOURCE = inputs / "figure1_features_fixed_dev_v5.source.json"
    architecture.PREFIX = out / STEM
    architecture.SURFACES = build_insets(out)
    old = ("Oil isosurfaces use a display-only stride of two, water a stride of four; sparse oil "
           "points use a stride-two grid and a fixed coordinate hash. ")
    new = ("Native-grid level-0.5 surfaces use smooth-normal lighting, self-shadow and local occlusion, "
           "without mesh smoothing, decimation or point overlays. The orthographic camera "
           "(azimuth -125 degrees, elevation 24 degrees) and phase colours match the displacement plates. ")
    assert old in architecture.CAPTION
    architecture.CAPTION = architecture.CAPTION.replace(old, new)
    architecture.COMPOSITION_DESCRIPTION = (
        "Existing compact two-row composition, all feature maps, dimensions, scientific nodes and edges retained. "
        "Only the two full-domain volume insets use native-grid 0.5 surfaces with the shared "
        "smooth-normal lighting and orthographic camera. Purple oil is opaque; pale-blue water has opacity 0.12; "
        "rock is hidden. No new concept pass: local inset style adaptation, not a composition redesign."
    )
    architecture.mpl.rcParams["svg.hashsalt"] = STEM
    architecture.render()
    # An inset display change must not silently move or rewire the architecture.
    unchanged = {}
    for kind in ("objects", "edges"):
        with (inputs / f"expected_{kind}.csv").open(newline="", encoding="utf-8") as handle:
            before = list(csv.DictReader(handle))
        with (out / f"{STEM}.{kind}.csv").open(newline="", encoding="utf-8") as handle:
            after = list(csv.DictReader(handle))
        assert before == after, f"Unexpected architecture {kind} change"
        unchanged[kind] = len(after)
        if kind == "objects":
            nodes = {row["object_id"]: row for row in after}
            assert all(nodes["X"][key] == nodes["OUTPUT"][key] for key in ("width", "height"))
    record_path = out / f"{STEM}.source.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["style_revision_checks"] = {"unchanged_objects_and_edges": unchanged,
        "volume_view_sizes_equal": True, "feature_cache_unchanged": digest(architecture.FEATURES) == SOURCE_SHA256,
        "composition_preserved": True}
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ARCHITECTURE_STYLE_COMPLETE", "prefix": str(out / STEM),
                      "unchanged": unchanged}), flush=True)


if __name__ == "__main__":
    raise SystemExit("Use reproduce_architecture.py with validated inputs and a fresh output directory")
