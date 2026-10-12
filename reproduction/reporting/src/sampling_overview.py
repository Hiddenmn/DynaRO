"""Unchanged projection and panel helpers for the fixed sampling hierarchy."""
from __future__ import annotations
import gc
import hashlib
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import proj3d
import sampling_reference as reference

BACKGROUND = reference.BACKGROUND
INK, LOCATOR = "#293745", "#385C70"
CROP = (slice(32, 96),) * 3

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def corners(low, high):
    return np.array(list(itertools.product(*zip(low, high))), float)


def draw_scene(labels, stride, spacing, *, surface_coarsen, water_alpha,
               roi_zyx=None, pixels=1600):
    stride, spacing = np.asarray(stride, int), np.asarray(spacing, float)
    displayed = np.array(labels[::stride[0], ::stride[1], ::stride[2]], copy=True)
    ns = reference.original_functions(stride*spacing, surface_coarsen)
    fig = plt.figure(figsize=(6.36, 6.), dpi=pixels/6, facecolor=BACKGROUND)
    ax = fig.add_axes([.04, .04, .92, .92], projection="3d")
    ax.set_facecolor(BACKGROUND)
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-55, roll=0)
    records = {}
    for cid in reversed(range(3, 8)):
        if np.any(displayed == cid):
            records[reference.NAMES[cid]] = ns["add_class_layer"](ax, displayed, cid)
    context = reference.original_functions(stride*spacing*2)
    water = displayed[::2, ::2, ::2] == 2
    if water.any():
        records["Water"] = context["add_surface_layer"](
            ax, water, reference.PALETTE[2], water_alpha)
    for collection in ax.collections:
        collection.set_clip_on(False)
    extent = np.asarray(labels.shape) * spacing
    for start, end in ns["box_edges_xyz"](extent):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color="#7F8B94", alpha=.26, lw=.45, solid_capstyle="round", clip_on=False)
    z, y, x = extent
    ax.set(xlim=(0, x), ylim=(0, y), zlim=(0, z))
    ax.set_box_aspect((x, y, z), zoom=1.20)
    ax.set_axis_off()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    outer = corners([0, 0, 0], extent[::-1])
    roi = None
    if roi_zyx is not None:
        bounds = np.array(roi_zyx) * spacing[:, None]
        roi = corners(bounds[::-1, 0], bounds[::-1, 1])
    def project_points(points):
        xx, yy, zz = proj3d.proj_transform(*points.T, ax.get_proj())
        p = ax.transData.transform(np.c_[xx, yy])
        p[:, 1] = image.shape[0] - p[:, 1]
        return p
    scene = {"image": image, "outer": project_points(outer),
             "roi": None if roi is None else project_points(roi)}
    record = {
        "source_shape_zyx": list(labels.shape),
        "display_grid_shape_zyx": list(displayed.shape),
        "display_stride_zyx": stride.tolist(),
        "surface_additional_stride": surface_coarsen,
        "water_additional_stride": 2, "coordinate_spacing_zyx": spacing.tolist(),
        "oil_alpha": reference.OIL_ALPHA, "oil_point_alpha": reference.OIL_POINT_ALPHA,
        "water_alpha": water_alpha, "phase_records": records}
    plt.close(fig); gc.collect()
    return scene, record


def image_panel(fig, rect, scene, letter):
    ax = fig.add_axes(rect, label=letter)
    ax.imshow(scene["image"], interpolation="none")
    ax.set_axis_off()
    h, w = scene["image"].shape[:2]
    ax.set(xlim=(-.5, w-.5), ylim=(h-.5, -.5))
    ax.text(.01, 1.01, letter, transform=ax.transAxes, fontsize=9,
            weight="bold", color=INK, ha="left", va="bottom")
    points = scene["roi"]
    if points is not None:
        for i in range(8):
            for j in range(i):
                # Cartesian corner ordering: an edge differs in one binary bit.
                if (i ^ j) not in (1, 2, 4):
                    continue
                xy = points[[i, j]]
                line, = ax.plot(xy[:, 0], xy[:, 1], lw=.75, color=LOCATOR, zorder=5)
                line.set_path_effects([pe.Stroke(linewidth=1.35, foreground=BACKGROUND), pe.Normal()])
    return ax


def link(fig, left, right, scene_left, scene_right):
    a, b = scene_left["roi"], scene_right["outer"]
    # Paired non-crossing leaders from right side of source ROI to left side
    # of magnified box. Exact projected coordinates, no decorative relocation.
    a_ids = np.argsort(a[:, 0])[-4:]
    b_ids = np.argsort(b[:, 0])[:4]
    for upper in (True, False):
        pick = np.argmin if upper else np.argmax
        ia, ib = a_ids[pick(a[a_ids, 1])], b_ids[pick(b[b_ids, 1])]
        fig.add_artist(ConnectionPatch(
            xyA=a[ia], coordsA="data", axesA=left,
            xyB=b[ib], coordsB="data", axesB=right,
            color=LOCATOR, alpha=.58, lw=.60, linestyle=(0, (3, 3)),
            zorder=3, clip_on=False))
