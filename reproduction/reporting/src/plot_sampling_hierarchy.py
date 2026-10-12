"""Full-stack classes and pooled-oil nested subvolumes from one fixed case.

Question: where do the 128-cubed input and central 64-cubed display region
come from? Panel a preserves the user's full-stack surface-plus-points style;
b/c use native masks and the manuscript's depth-buffered oil/water style.
All views have the same orthographic orientation. Colour pooling changes only
display, not labels, model inputs or measurements. No inference or TEST access.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "2"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import sampling_overview as overview
import surface_renderer as native
from figure_layout import check_layout as require_matplotlib_panel_alignment
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import psutil
from PIL import Image

STEM = "figure13_sampling_hierarchy_v3"
WIDTH_MM, HEIGHT_MM = 180., 73.
CAMERA = {"azimuth": -55., "elevation": 20., "projection": "orthographic"}
BACKGROUND = overview.BACKGROUND
LOCATOR = overview.LOCATOR


def pooled_scene(initial, central=False, pixels=1600):
    oil, water = initial >= 3, initial == 2
    if central:
        keep = np.zeros_like(oil)
        keep[overview.CROP] = True
        oil &= keep
        water &= keep
    camera = native.make_camera(pixels, az=CAMERA["azimuth"], el=CAMERA["elevation"])
    if central:
        camera["span"] *= .5
    surface = native.mesh(oil, 7)
    depth, normals, categories = native.geometry_buffer(surface, camera)
    light_camera = native.make_camera(700, direction=native.LIGHT)
    if central:
        light_camera["span"] *= .5
    shadow, _, _ = native.geometry_buffer(surface, light_camera)
    rgb, _ = native.light_pixels(
        depth, normals, categories, oil, native.PALETTE,
        camera["center"], camera["right"], camera["up"], camera["view"], camera["span"],
        native.LIGHT, shadow, light_camera["center"], light_camera["right"],
        light_camera["up"], light_camera["view"], light_camera["span"])
    rgb[categories == 0] = mpl.colors.to_rgb(BACKGROUND)
    wd, wn, wc = native.geometry_buffer(native.mesh(water, 2), camera)
    visible = (wc == 2) & (wd >= depth)
    shade = .74 + .26 * np.maximum(wn @ native.LIGHT, 0)
    water_rgb = native.PALETTE[2][None, None, :] * shade[..., None]
    rgb[visible] = .88 * rgb[visible] + .12 * water_rgb[visible]
    low, high = ([31.5]*3, [95.5]*3) if central else ([-.5]*3, [127.5]*3)
    outer = native.project(overview.corners(low, high), camera)[:, :2]
    roi = None if central else native.project(
        overview.corners([31.5]*3, [95.5]*3), camera)[:, :2]
    scene = {"image": np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8),
             "outer": outer, "roi": roi, "depth": depth, "camera": camera}
    record = {
        "source_shape_zyx": [128]*3, "display_shape_zyx": [64 if central else 128]*3,
        "native_oil_voxels": int(oil.sum()), "native_oil_triangles": len(surface["faces"]),
        "oil_classes_pooled_for_display": True, "oil_opacity": 1., "water_opacity": .12,
        "surface_grid_stride": 1, "isosurface": .5, "geometry_smoothing": False,
        "lighting": "Smooth-normal depth-buffered light and contact shading on the native mask",
        "coordinate_spacing_zyx": [1, 1, 1]}
    del surface, normals, categories, shadow, wd, wn, wc
    gc.collect()
    return scene, record


def render(out, full_path, case_path, original_slice):
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": STEM, "image.composite_image": False, "legend.frameon": False})
    full = np.load(full_path, mmap_mode="r")
    with np.load(case_path, allow_pickle=False) as data:
        initial = data["initial_labels"]
    with Image.open(original_slice) as image:
        assert image.size == (1024, 1024)
    assert full.shape == (845, 512, 512) and initial.shape == (128,)*3
    assert np.array_equal(full[:128, 256:384, 256:384], initial)
    assert int((initial >= 3).sum()) == 295851
    assert int((initial[overview.CROP] >= 3).sum()) == 35740
    print("Confirmed exact full-stack / input / interior mapping", flush=True)
    hashes = {"full_sha256": overview.digest(full_path),
              "case_sha256": overview.digest(case_path)}
    scenes, records = [], []
    out.mkdir(parents=True, exist_ok=True)
    print("Rendering reference-style full stack from saved labels", flush=True)
    scene, record = overview.draw_scene(
        full, (8,4,4), (1,2,2), surface_coarsen=2, water_alpha=.045,
        roi_zyx=[[0,128],[256,384],[256,384]])
    scenes.append(scene)
    records.append(record)
    for central in (False, True):
        print("Rendering pooled native "+("64-cubed interior" if central else "128-cubed input"), flush=True)
        scene, record = pooled_scene(initial, central)
        scenes.append(scene)
        records.append(record)
    payload = {**hashes, "records": json.dumps(records)}
    for letter, scene in zip("abc", scenes):
        for key in ("image", "outer", "roi"):
            if scene[key] is not None:
                payload[letter+"_"+key] = scene[key]
    np.savez_compressed(out/(STEM+".render_cache.npz"), **payload)
    # Equal-span panels, with local legends distinguishing classification from
    # phase-only display. No subtitles or explanatory prose inside the figure.
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), facecolor=BACKGROUND)
    width = .305
    height = width*WIDTH_MM/(HEIGHT_MM*1.06)
    axes = [overview.image_panel(fig, [x, .195, width, height], scene, letter)
            for x, scene, letter in zip((.01, .3475, .685), scenes, "abc")]
    for ax, scene in zip(axes[1:], scenes[1:]):
        outer = scene["outer"]
        for i in range(8):
            for j in range(i):
                if (i ^ j) in (1,2,4):
                    xy = outer[[i,j]]
                    ax.plot(xy[:,0], xy[:,1], color="#BFC9D2", alpha=.70,
                            lw=.40, zorder=3)
    overview.link(fig, axes[0], axes[1], scenes[0], scenes[1])
    overview.link(fig, axes[1], axes[2], scenes[1], scenes[2])
    ids = [2,3,4,5,6,7]
    phase_keys = [Patch(facecolor=overview.reference.PALETTE[k],
                        label=overview.reference.NAMES[k]) for k in ids]
    fig.legend(handles=phase_keys, ncol=3, loc="lower center",
               bbox_to_anchor=(.170,.012), fontsize=7, columnspacing=.65,
               handlelength=.7, handletextpad=.35, labelspacing=.45)
    pooled_keys = [Patch(facecolor=native.PALETTE[7], label="Oil"),
                   Patch(facecolor="#D6E7F0", label="Water"),
                   Patch(facecolor="none", edgecolor=LOCATOR, label="Selected region")]
    fig.legend(handles=pooled_keys, ncol=3, loc="lower center",
               bbox_to_anchor=(.673,.038), fontsize=7.5, columnspacing=1.5,
               handlelength=.85, handletextpad=.4)
    prefix = out/STEM
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=list("abc"), row_groups=[list("abc")],
        require_panel_labels=True, tolerance_pt=1.5,
        json_out=str(prefix)+".alignment.json", strict=True)
    fig.savefig(str(prefix)+".pdf", dpi=600, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(str(prefix)+".svg", dpi=600, metadata={"Date": None})
    fig.savefig(str(prefix)+".png", dpi=400)
    fig.savefig(str(prefix)+".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    metadata = {
        "scientific_question": "How do the complete semantic volume, model input and interior region correspond?",
        "full_source": full_path.name, "full_source_sha256": hashes["full_sha256"],
        "cached_case": case_path.name, "case_sha256": hashes["case_sha256"],
        "exact_parent_match": True, "parent_origin_zyx": [0,256,256],
        "parent_shape_zyx": [128]*3, "interior_parent_bounds_zyx": [[32,96]]*3,
        "interior_stack_bounds_zyx": [[32,96],[288,352],[288,352]],
        "original_slice_size_xy": [1024,1024],
        "coordinate_note": "Overview display spacing z:y:x=1:2:2 accounts for verified in-plane resizing. Subvolumes use isotropic model indices. Separate magnification; no physical-unit calibration.",
        "camera": CAMERA, "panels": dict(zip("abc", records)),
        "figure_size_mm": [WIDTH_MM, HEIGHT_MM], "rock_hidden_display_only": True,
        "source_labels_modified": False, "new_inference": False, "test_accessed": False,
        "full_reference_renderer_sha256": overview.reference.REFERENCE_SHA256}
    prefix.with_suffix(".source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Wrote "+str(prefix), flush=True)
    return metadata


if __name__ == "__main__":
    raise SystemExit("Use reproduce_sampling.py with validated inputs and a fresh output directory")
