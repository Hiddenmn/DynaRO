"""Render an existing TRAIN displacement example with the paper's volume style.

Image plate + quantification. The three states, converted support and complete
inventory curve describe one existing trajectory, not a model-performance test.
The fixed selection records the simulator-content-selected internal crop.
All 51 exact counts are retained. Python only; no legacy-crop fallback.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "2"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

from surface_renderer import (
    PALETTE, LIGHT, mesh, geometry_buffer, make_camera, light_pixels, project,
)
from train_counts import count_rows
from figure_layout import check_layout as require_matplotlib_panel_alignment
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import psutil

SOURCE = HERE.parent / "inputs_train3d/train50_NPI-012_z384_y384_x256_streamA_S320_then_S530.npz"
START, WIDTH, TIMES = (40, 16, 16), 64, (0, 20, 50)
STEM = "figure2_train50_interior_v8"
FIG_W, FIG_H = 180, 130
CAMERA = {"azimuth": -125.0, "elevation": 24.0, "projection": "orthographic"}
WATER_OPACITY = 0.12
GREEN, INK, MUTED = "#428F7D", "#293745", "#52677A"
REGION = tuple(slice(s, s + WIDTH) for s in START)
CENTER = (np.asarray(START[::-1]) + (WIDTH - 1) / 2).astype(np.float32)


def camera(pixels, direction=None):
    c = make_camera(pixels, az=CAMERA["azimuth"], el=CAMERA["elevation"], direction=direction)
    c["span"] *= WIDTH / 128
    c["center"] = CENTER.copy()
    return c


def embed(cropped):
    mask = np.zeros((128,) * 3, dtype=bool)
    mask[REGION] = cropped
    return mask


def scene(mask, water=None, *, changed=False, ghost=None, pixels=1400):
    retained = embed(mask)
    c = camera(pixels)
    m = mesh(retained, 6 if changed else 7)
    depth, normals, categories = geometry_buffer(m, c)
    lc = camera(700, direction=LIGHT)
    shadow, _, _ = geometry_buffer(m, lc)
    rgb, _ = light_pixels(
        depth, normals, categories, retained, PALETTE,
        c["center"], c["right"], c["up"], c["view"], c["span"], LIGHT,
        shadow, lc["center"], lc["right"], lc["up"], lc["view"], lc["span"],
    )
    oil = rgb.copy()
    if water is not None:
        wd, wn, wc = geometry_buffer(mesh(embed(water), 2), c)
        visible = (wc == 2) & (wd >= depth)
        shade = 0.74 + 0.26 * np.maximum(wn @ LIGHT, 0)
        wrgb = PALETTE[2][None, None, :] * shade[..., None]
        rgb[visible] = (1 - WATER_OPACITY) * rgb[visible] + WATER_OPACITY * wrgb[visible]
    if ghost is not None:
        gray = ghost @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        faint = 1 - 0.17 * (1 - gray)
        background = categories == 0
        rgb[background] = faint[background, None]
    result = {"rgb": np.clip(rgb, 0, 1), "oil": oil, "camera": c, "depth": depth}
    del m, normals, categories, shadow
    gc.collect()
    return result


def frame(ax, c, depth):
    corners = np.array(list(itertools.product(*[(s - 0.5, s + WIDTH - 0.5) for s in START[::-1]])))
    for i in range(8):
        for j in range(i):
            if np.count_nonzero(corners[i] != corners[j]) != 1:
                continue
            xyz = corners[i] + np.linspace(0, 1, 240)[:, None] * (corners[j] - corners[i])
            q = project(xyz, c)
            ix = np.clip(q[:, 0].astype(int), 0, c["width"] - 1)
            iy = np.clip(q[:, 1].astype(int), 0, c["height"] - 1)
            visible = q[:, 2] >= depth[iy, ix] - 0.25
            xy = q[:, :2].copy()
            xy[~visible] = np.nan
            ax.plot(xy[:, 0], xy[:, 1], lw=0.4, color="#CED6DE")


def heading(ax, letter, title):
    ax.text(0, 1.03, letter, transform=ax.transAxes, fontsize=10, weight="bold",
            color=INK, ha="left", va="bottom")


def selected_crop(selection_path):
    global SOURCE, START, WIDTH, REGION, CENTER, STEM
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection["selected"]
    SOURCE = selection_path.parent / chosen["source"]
    assert SOURCE.name.startswith("train50_") and SOURCE.name.endswith(".npz")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == selection["selected_source_sha256"]
    START = (chosen["z"], chosen["y"], chosen["x"])
    WIDTH = chosen["width"]
    REGION = tuple(slice(s, s + WIDTH) for s in START)
    CENTER = (np.asarray(START[::-1]) + (WIDTH - 1) / 2).astype(np.float32)
    STEM = "figure2_train50_interior_v8"
    with np.load(SOURCE, allow_pickle=False) as data:
        labels = data["labels"]
        assert labels.shape == (51, 128, 128, 128)
        cube = labels[(slice(None),) + REGION].copy()
        actions = data["action_index"]
        assert len(actions) == 50 and np.all(actions[:20] == actions[0])
        assert np.all(actions[20:] == actions[-1]) and actions[0] != actions[-1]
        measures = (f"S{int(actions[0]) + 1}", f"S{int(actions[-1]) + 1}")
    assert not np.any(cube[0] == 0)
    oil = cube >= 3
    assert not np.any(oil[1:] & ~oil[:-1])
    rows = count_rows(cube)
    assert rows[0]["remaining_voxels"] == chosen["initial_oil"]
    assert rows[-1]["converted_voxels"] == chosen["converted"]
    inner = cube[:, 16:48, 16:48, 16:48]
    assert np.count_nonzero((inner[0] >= 3) & (inner[-1] == 2)) == chosen["interior_converted"]
    record = {"role": "illustrative simulator-content-selected TRAIN targets", "horizon": 50,
        "source": SOURCE.name,
        "source_sha256": selection["selected_source_sha256"],
        "crop_start_zyx": list(START), "crop_shape": [WIDTH] * 3,
        "source_shape": [51, 128, 128, 128], "selection": selection,
        "initial_oil_sites": chosen["initial_oil"], "initial_water_sites": chosen["initial_water"],
        "schedule": f"{measures[0]} x20 then {measures[1]} x30", "display_steps": list(TIMES),
        "quantification": "Exact original voxel counts at all 51 states; no surface-derived measurements",
        "display_exclusions": "Rock hidden; five oil classes pooled; no exterior background in crop",
        "physical_scale": "Voxel coordinates only; no calibrated physical length or time",
        "statistical_unit": "One simulator-content-selected illustrative TRAIN trajectory, not an aggregate performance estimate",
        "model_inference": False, "test_access": False}
    return cube, rows, record, measures


def render(out, selection_path):
    if sys.platform == "win32":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cube, rows, old, measures = selected_crop(selection_path)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": STEM,
        "image.composite_image": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "legend.frameon": False,
    })
    n0, converted = rows[0]["remaining_voxels"], rows[-1]["converted_voxels"]
    scenes = []
    for t in TIMES:
        print(f"Rendering fixed TRAIN crop, step {t}", flush=True)
        scenes.append(scene(cube[t] >= 3, cube[t] == 2))
    moved = (cube[0] >= 3) & (cube[50] == 2)
    print("Rendering converted oil in the initial-oil context", flush=True)
    scenes.append(scene(moved, changed=True, ghost=scenes[0]["oil"]))

    fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4), facecolor="white")
    xs = [0.018, 0.348, 0.678]
    width = 0.307
    height = width * FIG_W / (FIG_H * 1.06)
    axes = []
    titles = ["Initial · step 0", f"After {measures[0]} · step 20", f"After {measures[1]} · step 50", "Mobilized oil · step 50"]
    for i, (result, title) in enumerate(zip(scenes, titles)):
        x, y = (xs[i], 0.550) if i < 3 else (xs[0], 0.060)
        ax = fig.add_axes([x, y, width, height], label="abcd"[i])
        ax.imshow(result["rgb"], interpolation="none")
        frame(ax, result["camera"], result["depth"])
        ax.set(xlim=(-0.5, result["camera"]["width"] - 0.5),
               ylim=(result["camera"]["height"] - 0.5, -0.5))
        ax.set_axis_off()
        heading(ax, "abcd"[i], title)
        axes.append(ax)

    curve = fig.add_axes([0.406, 0.159, 0.573, 0.286], label="e")
    title_y = 0.060 + height * 1.03
    fig.text(0.406, title_y, "e", fontsize=10, weight="bold", color=INK, va="bottom")
    step = np.array([r["step"] for r in rows])
    fraction = 100 * np.array([r["converted_voxels"] / n0 for r in rows])
    curve.axvspan(0, 20, color="#F3F5F6", zorder=0)
    curve.axvline(20, color="#99A7B0", lw=0.7, ls="--", zorder=1)
    curve.plot(step, fraction, color=GREEN, lw=1.6, zorder=3)
    curve.plot(step[list(TIMES)], fraction[list(TIMES)], "o", color=GREEN, ms=4, zorder=4)
    for t in TIMES[1:]:
        curve.annotate(f"{fraction[t]:.2f}%", xy=(step[t], fraction[t]),
                       xytext=(-2 if t == TIMES[-1] else 5, 7),
                       textcoords="offset points",
                       ha="right" if t == TIMES[-1] else "left", va="bottom",
                       fontsize=7, color=INK, zorder=5)
    top = np.ceil(fraction[-1] / 10) * 10 + 6
    ticks = np.arange(0, top, 10)
    curve.set(xlim=(-1, 51), ylim=(0, top), xticks=[0, 10, 20, 30, 40, 50],
              yticks=ticks, xlabel="Simulator step", ylabel="Initial oil mobilized (%)")
    curve.tick_params(length=3, pad=3, labelsize=7.5)
    curve.grid(axis="y", color="#E6EBEF", lw=0.5, zorder=0)
    fig.legend(handles=[Patch(facecolor=PALETTE[7], label="Remaining oil"),
                        Patch(facecolor="#CDE5F3", label="Water"),
                        Patch(facecolor=PALETTE[6], label="Mobilized oil")],
               loc="lower center", bbox_to_anchor=(0.5, 0.001), ncol=3,
               fontsize=8, handlelength=1, columnspacing=2)
    prefix = out / STEM
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=list("abcd"), row_groups=[list("abc")],
        column_groups=[list("ad")], json_out=str(prefix) + ".alignment.json",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
    )
    fig.savefig(str(prefix) + ".pdf", dpi=600, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(str(prefix) + ".svg", dpi=600, metadata={"Date": None})
    fig.savefig(str(prefix) + ".png", dpi=300)
    fig.savefig(str(prefix) + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with Path(str(prefix) + ".source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    record = {k: v for k, v in old.items() if not k.startswith("animation")}
    record.update({
        "image_plate_size_mm": [FIG_W, FIG_H], "volume_panel_render_pixels": [1484, 1400],
        "camera": {**CAMERA, "center_xyz": CENTER.tolist(), "crop_width_voxels": WIDTH},
        "surface": "Native-grid 0.5 marching cubes; smooth-normal lighting, depth testing, self-shadow and local occlusion. No geometric smoothing, decimation or resampling.",
        "opacity": {"remaining_oil": 1.0, "water": WATER_OPACITY, "mobilized": 1.0},
        "water_context": "Water at the corresponding simulated state",
        "mobilized_context": "Faint neutral-gray initial oil silhouette, using the same camera and depth projection.",
        "render_style": "Shared palette, light, depth-buffer renderer and orthographic camera direction with the central-interior figure.",
        "curve": "All 51 original exact counts; converted oil divided by initial oil, in percent; no smoothing or time interpolation.",
        "style_revision_only": False, "original_count_rows_verified": 51,
    })
    Path(str(prefix) + ".source.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "TRAIN_STYLE_COMPLETE", "prefix": str(prefix),
                      "remaining_counts": [rows[t]["remaining_voxels"] for t in TIMES],
                      "converted_count": converted}), flush=True)
    return record


if __name__ == "__main__":
    raise SystemExit("Use reproduce_train3d.py to validate the fixed selection and preserve outputs.")
