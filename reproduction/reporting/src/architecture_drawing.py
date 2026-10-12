"""Editable, real-tensor architecture candidate; no network inference.

Composition follows the inspected Image2 concept, not its pixels or arrows.
Input/output renders and every feature thumbnail derive from the same fixed
DEV example. Vector annotations and connections are drawn from actual code.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS"):
    os.environ[key] = "2"
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image
import psutil
from figure_layout import check_layout as require_matplotlib_panel_alignment
from surface_renderer import project

PREFIX = FEATURES = SOURCE = SURFACES = None  # supplied by the validated entry point
COMPOSITION_DESCRIPTION = (
    "Compact two-row composition and all five cached class-logit planes retained; no global/image titles or explanatory annotations. "
    "Input/output views reuse the user's original faceted surface-plus-points renderer, with full-domain water and pooled purple oil; "
    "rock is hidden only in the display. The two-phase legend describes the views, not the five-channel head. "
    "Class provenance and display sampling stay in the external caption."
)
BLUE = "#287aad"
TEAL = "#008c89"
PURPLE = "#7955a6"
ORANGE = "#c97132"
DARK = "#233649"
GRAY = "#637584"
LIGHT = "#bdcbd4"
W, H = 180.0, 128.0
PT = 72 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.8, "svg.fonttype": "none", "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans", "legend.frameon": False,
    "savefig.facecolor": "white", "image.interpolation": "nearest",
    "image.composite_image": False, "svg.hashsalt": "StateGuard-RO-dataflow-v8-exposure",
})

CAPTION = (
    "DynaRO recurrent 3D architecture with predicted-state fronts. Eight one-hot channels from stacked slice labels and "
    "the inlet mask I and connected-pore mask C form the 128-cubed input G. Two 3D convolutions encode local features E; "
    "pooling produces regional features S, which initialize recurrent memory and enter the ConvGRU "
    "context together with previous five-class fractions, the current action and bounded time. Two "
    "recurrent heads supply five oil-class logits and a cell bias. Gathering these by each initial oil site's "
    "fixed initial class k0 and cell c, adding a local-logit skip and applying sigmoid yields site propensity. "
    "Six-face exposure from the previous predicted state, bounded by inlet-connected reachability, "
    "modulates this propensity. A nonnegative multiplicative update produces oil occupancy; "
    "five-class remaining amounts N and RF follow by aggregation using the fixed initial oil classes. The five "
    "regional fractions feed back into the next step; the model does not reclassify oil morphology. "
    "Dashed paths carry the predicted state and "
    "hidden memory to the next step. State-prioritized learning acts during training. All windows use the same fixed "
    "background-free DEV case (seed 20260811; S2 for 20 steps then S3 for 80), with recurrent fields shown "
    "for transition 99 to 100. Encoder and hidden-state stacks show channels 0-2 at z=64 on the native "
    "grid or z=4 on the regional grid. All five class-logit planes are shown at z=4 in the fixed "
    "order film, corner, droplet, columnar and cluster. Feature intensities use channelwise min-max display normalization. Exposure is "
    "maximum-projected along z. A full 128-by-128 locator outlines the magnified 40-by-40 exposure "
    "window at x=0-39, y=0-39; this window contains all nonzero projected exposure in the cached "
    "transition. Exposure and occupancy views retain their 0-1 scales. Input and "
    "output volumes share the camera, model-grid coordinate scale and phase opacities. "
    "Water and oil are rendered over the complete 128-cubed domain, with rock hidden for clarity. "
    "Translucent blue water provides context for a single purple oil phase; this display pooling does not remove the model's "
    "five-class information. Oil isosurfaces use a display-only stride of two, water a stride of four; sparse oil "
    "points use a stride-two grid and a fixed coordinate hash. Output rendering uses a 0.5 occupancy threshold; numerical quantities "
    "use continuous probabilities. Initial classes come from the stacked two-dimensional labels, "
    "not an independent three-dimensional morphology classification."
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible_wire(camera, depth):
    corners = np.asarray([[x, y, z] for z in (-1., 128.)
                          for y in (-1., 128.) for x in (-1., 128.)])
    lines = []
    for first in range(8):
        for bit in (1, 2, 4):
            second = first ^ bit
            if first >= second:
                continue
            xyz = corners[first] + np.linspace(0, 1, 300)[:, None] * (corners[second] - corners[first])
            q = project(xyz, camera)
            ix = np.clip(q[:, 0].astype(int), 0, camera["width"]-1)
            iy = np.clip(q[:, 1].astype(int), 0, camera["height"]-1)
            xy = q[:, :2].copy()
            xy[q[:, 2] < depth[iy, ix]-.25] = np.nan
            lines.append(xy)
    return np.asarray(lines)


def get_surfaces(data):
    if not SURFACES.exists():
        raise FileNotFoundError("Run render_complete_multiphase_views.py to update data-only insets")
    saved = json.loads(SURFACES.with_suffix(".source.json").read_text(encoding="utf-8"))
    if saved["source_sha256"] != digest(FEATURES):
        raise ValueError("Volume inset cache no longer matches the fixed feature cache")
    if saved["source_shape_zyx"] != list(data["initial_labels"].shape):
        raise ValueError("Volume inset dimensions disagree")
    with np.load(SURFACES, allow_pickle=False) as cache:
        return (cache["initial"], cache["output"], cache["initial_wire"], cache["output_wire"],
                {int(key): value for key, value in saved["palette"].items()})


class Drawing:
    def __init__(self):
        self.fig = plt.figure(figsize=(W/25.4, H/25.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
        self.ax.axis("off")
        self.nodes = {}
        self.labels = []
        self.edges = []

    def obj(self, name, bounds, kind="tensor", label="", parent=""):
        x, y, w, h = bounds
        self.nodes[name] = {"object_id": name, "kind": kind, "label": label or name,
                           "x": x*PT, "y": y*PT, "width": w*PT, "height": h*PT,
                           "parent_id": parent, "allow_overlap_with": "",
                           "allow_line_crossing": False}

    def text(self, name, x, y, value, *, size=7.8, color=DARK, weight="normal",
             ha="center", va="center", parent=""):
        artist = self.ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                              ha=ha, va=va, zorder=20)
        self.labels.append((name, artist, parent))
        return artist

    def stack(self, name, values, cx, cy, side, *, cmap="Blues", color=BLUE,
              scale=None, parent="", channel_colors=None):
        if values.ndim != 3:
            raise ValueError("Expected selected actual channel/plane maps")
        n = len(values)
        dx, dy = .95, -1.10
        x, y = cx-side/2, cy-side/2
        vmin, vmax = (float(np.min(values)), float(np.max(values))) if scale is None else scale
        if vmax <= vmin:
            vmax = vmin+1e-8
        limits = []
        for k in range(n-1, -1, -1):
            xx, yy = x+k*dx, y+k*dy
            low, high = (float(values[k].min()), float(values[k].max())) if scale is None else scale
            if high <= low:
                high = low + 1e-8
            norm = Normalize(low, high)
            limits.append({"channel": k, "min": low, "max": high})
            self.ax.imshow(values[k], cmap=cmap, norm=norm,
                           extent=(xx, xx+side, yy+side, yy), zorder=6+n-k,
                           interpolation="nearest", aspect="equal")
            self.ax.add_patch(Rectangle((xx, yy), side, side, fill=False,
                                       edgecolor=(color if channel_colors is None
                                                  else channel_colors[k]),
                                       lw=.6, zorder=10+n-k))
        self.obj(name, (x, y+(n-1)*dy, side+(n-1)*dx, side-(n-1)*dy),
                 "real_tensor_view", parent=parent)
        return {"per_channel_limits": limits}

    def volume(self, name, image, wire_lines, cx, cy, side):
        # Preserve the renderer's 1.06 width/height aspect.
        width = side*image.shape[1]/image.shape[0]
        x, y = cx-width/2, cy-side/2
        self.ax.imshow(image, extent=(x, x+width, y+side, y),
                       aspect="equal", zorder=8, interpolation="none")
        for projected_line in wire_lines:
            xy = projected_line.copy()
            xy[:, 0] = x + xy[:, 0]/image.shape[1]*width
            xy[:, 1] = y + xy[:, 1]/image.shape[0]*side
            self.ax.plot(xy[:, 0], xy[:, 1], color="#c6d0d7", lw=.35, zorder=9)
        self.obj(name, (x, y, width, side), "real_full_volume")

    def box(self, name, x, y, w, h, *, color=BLUE, face="white"):
        self.ax.add_patch(Rectangle((x, y), w, h, edgecolor=color,
                                   facecolor=face, lw=.75, zorder=5))
        self.obj(name, (x, y, w, h), "operator")

    def circle(self, name, x, y, r, value, color):
        self.ax.add_patch(Circle((x, y), r, ec=color, fc="white", lw=.85, zorder=10))
        self.obj(name, (x-r, y-r, 2*r, 2*r), "operator")
        self.text(name+"_label", x, y-.05, value, size=8.8, color=color, parent=name)

    def edge(self, name, source, target, points, *, color=BLUE, dashed=False,
             meaning="", crossings=()):
        points = np.asarray(points, float)
        self.ax.plot(points[:, 0], points[:, 1], color=color, lw=.8,
                     ls=(0, (3.2, 2.2)) if dashed else "-", zorder=3,
                     solid_capstyle="butt")
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                         mutation_scale=7, lw=.8, color=color,
                                         linestyle="none", shrinkA=0, shrinkB=0,
                                         zorder=4))
        self.edges.append({"edge_id": name, "source": source, "target": target,
                           "meaning": meaning or name,
                           "style": "dashed arrow; one-step delay" if dashed else "solid arrow; forward",
                           "direction": "forward", "route": json.dumps((points*PT).tolist()),
                           "allow_crossing_with": ";".join(crossings)})

    def finish_geometry(self):
        self.fig.canvas.draw()
        inv = self.ax.transData.inverted()
        for name, artist, parent in self.labels:
            bounds = artist.get_window_extent(self.fig.canvas.get_renderer())
            (x0, y1), (x1, y0) = inv.transform([[bounds.x0, bounds.y0], [bounds.x1, bounds.y1]])
            self.obj(name, (x0, y0, x1-x0, y1-y0), "text", artist.get_text(), parent)
        for stem, rows in (("objects", list(self.nodes.values())), ("edges", self.edges)):
            with Path(str(PREFIX)+"."+stem+".csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


def render():
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    with np.load(FEATURES, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    metadata = json.loads(SOURCE.read_text(encoding="utf-8"))
    if metadata["max_difference_from_existing_prediction"] != 0:
        raise ValueError("Expected unchanged fixed-case prediction")
    initial_im, output_im, initial_wire, output_wire, volume_palette = get_surfaces(data)
    d = Drawing()
    scales = {}
    # One-time native-resolution encoder: actual channel planes, not resized labels.
    d.volume("X", initial_im, initial_wire, 20, 33, 30)
    d.text("X_dim", 20, 50.8, "G: 10 × 128³", color=BLUE)
    scales["conv1"] = d.stack("E1", data["conv1_gelu"][:3, 64], 52, 33, 14)
    d.text("E1_title", 52.7, 20.5, "Conv + GELU", size=7.5, color=BLUE)
    d.text("E1_dim", 52.7, 44, "8 × 128³", color=BLUE)
    scales["E"] = d.stack("E", data["encoder_E"][:3, 64], 76, 33, 14)
    d.text("E_title", 76.8, 48, "Conv + GELU", size=7.5, color=BLUE)
    d.text("E_dim", 76.8, 44, "E: 8 × 128³", color=BLUE)
    d.edge("X_E1", "X", "E1", [(35.9, 33), (45, 33)], meaning="3x3x3 convolution 10 to 8, then GELU")
    d.text("conv1_kernel", 40, 29.6, "3³", size=7.8, color=BLUE)
    d.edge("E1_E", "E1", "E", [(60.9, 33), (69, 33)], meaning="3x3x3 convolution 8 to 8, then GELU")
    d.text("conv2_kernel", 65, 29.6, "3³", size=7.8, color=BLUE)
    scales["S"] = d.stack("S", data["pooled_S"][:3, 4], 99, 33, 10)
    d.text("S_title", 99.8, 22.4, "AvgPool", size=7.5, color=BLUE)
    d.text("S_dim", 99.8, 44, "S: 8 × 8³", color=BLUE)
    d.edge("E_S", "E", "S", [(84.9, 33), (94, 33)], meaning="Average pool kernel=stride=16")
    d.text("pool_size", 89.2, 29.6, "16³", color=BLUE)

    # ConvGRU group, with an actual hidden-state feature stack.
    d.box("GRU", 116, 23, 21, 20, color=TEAL, face="#f5faf9")
    scales["H"] = d.stack("H", data["hidden"][:3, 4], 125.5, 32.5, 10,
                          cmap="viridis", color=TEAL, parent="GRU")
    d.text("GRU_title", 126.5, 18.8, "ConvGRU3D", weight="bold", color=TEAL)
    d.text("H_dim", 126.5, 40.9, "Hₜ: 24 × 8³", size=7.5, color=TEAL, parent="GRU")
    # Static regional features initialize memory exactly once.
    d.edge("S_H0", "S", "GRU", [(105.9, 33), (116, 33)],
           color=GRAY, meaning="Conv3D 3x3x3 and tanh initialize H0 once")
    d.text("H0_init", 111, 27.8, "H₀", size=7.5, color=GRAY)
    # Native-resolution logit skip, deliberately separate from H_t.
    scales["ell"] = d.stack("L", data["local_logit"][:, 64], 148, 9.5, 9,
                            color=BLUE)
    d.text("L_label", 148, 2.2, "1³ Conv", size=7.5, color=BLUE)
    d.edge("E_L", "E", "L", [(80, 23.8), (80, 15.5), (137, 15.5),
                             (137, 9.5), (143.5, 9.5)], meaning="E to static full-resolution local logit")
    d.text("L_dim", 147.8, 17.5, "ℓ: 1 × 128³", color=BLUE)
    # Two cell-resolution 1x1x1 heads, displayed as actual output planes.
    d.box("HEADS", 145, 24.5, 17.5, 19.5, color=BLUE, face="#f7fafc")
    scales["q"] = d.stack("Q", data["class_logits"][:, 4], 149.4, 32.1, 4.8,
                          parent="HEADS")
    scales["b"] = d.stack("B", data["cell_bias"][:, 4], 158, 39, 6,
                          parent="HEADS")
    d.text("Q_label", 158.4, 28.6, "qₜ", color=BLUE, parent="HEADS")
    d.text("B_label", 148.4, 41, "bₜ", color=BLUE, parent="HEADS")
    d.text("heads_title", 153.5, 21.6, "1³ Conv × 2", size=7.5, color=BLUE)
    d.text("heads_dims", 153.7, 47.6, "5 × 8³ + 1 × 8³", size=7.5, color=BLUE)
    d.edge("H_heads", "GRU", "HEADS", [(137, 31), (145, 31)],
           meaning="Hidden state to class logits and cell bias")
    # Gather comes before the sitewise sum.
    d.circle("PLUS", 172, 51, 2.3, "+", BLUE)
    d.edge("heads_gather_sum", "HEADS", "PLUS",
           [(162.5, 35), (166, 35), (166, 51), (169.7, 51)],
           meaning="Gather q_t by fixed cell and initial class; gather b_t by fixed cell")
    d.text("gather_label", 155.6, 55.2, "Gather (c, k₀)", size=7.2, color=BLUE)
    d.edge("L_sum", "L", "PLUS", [(152.5, 9.5), (172, 9.5), (172, 48.7)],
           meaning="Gather local logit at initial oil sites")
    d.circle("SIG", 172, 63, 2.6, "σ", BLUE)
    d.edge("sum_sigmoid", "PLUS", "SIG", [(172, 53.3), (172, 60.4)],
           meaning="Sigmoid of the sum gives one propensity per initial oil site")

    # Context contains S, previous class fractions, action and bounded time.
    d.box("CONTEXT", 112, 55, 30, 6, color=TEAL, face="#f5faf9")
    d.text("context_components", 127, 58, "8 + 5 + 4 + 1", color=TEAL,
           parent="CONTEXT")
    d.text("context_dims", 109.5, 63.7, "18 × 8³", size=7.5,
           color=TEAL, ha="right")
    d.edge("S_context", "S", "CONTEXT", [(105.9, 36), (111, 36),
                                        (111, 50), (111, 55), (112, 55)],
           meaning="Static eight-channel S enters recurrent context")
    d.edge("context_GRU", "CONTEXT", "GRU", [(121, 55), (121, 43)],
           color=TEAL, meaning="18-channel concatenated context")
    d.edge("H_memory", "GRU", "GRU", [(137, 39), (140, 39),
                                      (140, 49.5), (133, 49.5), (133, 43)],
           color=TEAL, dashed=True, meaning="One-step hidden-state memory")
    d.text("memory_label", 135, 52, "t+1", size=7.5, color=TEAL, ha="center")
    d.obj("ACTION", (115, 67, 29, 3.6), "conditioning")
    d.text("action_label", 129.5, 68.8, "aₜ (4) + τₜ (1)", color=ORANGE, parent="ACTION")
    d.edge("action_context", "ACTION", "CONTEXT", [(130, 67), (130, 61)],
           color=ORANGE, meaning="Current four-component action and t/(t+25)")
    # Derive previous regional fractions from p_{t-1}: no separate network.
    d.circle("REGIONAL", 76, 57.8, 2.6, "Σ", TEAL)
    d.text("regional_label", 76, 52.4, "rₜ₋₁: 5 × 8³", size=7.5, color=TEAL)
    d.edge("regional_context", "REGIONAL", "CONTEXT",
           [(78.6, 57.8), (112, 57.8)], color=TEAL,
           meaning="Five normalized regional inventories from previous state")

    # Keep the existing two-row composition; close the large lower-row gaps.
    # All original scientific nodes and edges are retained.
    scales["previous"] = d.stack("PREV", data["previous_occupancy"][None, 64],
                                 28, 88, 19, cmap="Purples", color=PURPLE, scale=(0, 1))
    d.text("prev_dim", 28, 102, "pₜ₋₁ · 128³", color=PURPLE)
    d.edge("previous_regional", "PREV", "REGIONAL",
           [(18.5, 82), (12, 82), (12, 57.8), (73.4, 57.8)], color=TEAL,
           meaning="Sum previous occupancy by fixed initial class and cell; divide by initial inventory",
           crossings=("geometry_front",))
    # Same full projection and absolute scale as v7, with a linked magnification.
    # This is a localized display adaptation, not a new inference or data filter.
    front_projection = data["front_weight"].max(axis=0)
    outside_window = front_projection.copy()
    outside_window[:40, :40] = 0
    if np.count_nonzero(outside_window):
        raise ValueError("The declared exposure window must contain every nonzero projected value")
    exposure_cmap = LinearSegmentedColormap.from_list(
        "absolute_exposure", ["#f3f6f6", "#85c9c2", "#006b67"])
    scales["front_locator"] = d.stack("FRONT_LOCATOR", front_projection[None],
                                      46, 78, 8, cmap=exposure_cmap,
                                      color=LIGHT, scale=(0, 1))
    locator_roi = 8 * 40 / 128
    d.ax.add_patch(Rectangle((42, 74), locator_roi, locator_roi,
                            fill=False, edgecolor=TEAL, lw=.6, zorder=16))
    d.obj("front_roi", (42, 74, locator_roi, locator_roi),
          "display_crop_locator", parent="FRONT_LOCATOR")
    d.ax.plot([42+locator_roi, 53, 55.5], [74+locator_roi, 76.5, 78.5],
              color=LIGHT, lw=.5, zorder=2)
    d.text("front_locator_dim", 46, 84.1, "128²", size=6.5, color=GRAY)
    scales["front"] = d.stack("FRONT", front_projection[None, :40, :40],
                              65, 88, 19, cmap=exposure_cmap, color=TEAL, scale=(0, 1))
    d.text("front_operator", 65, 74.3, "6-neighbour max", size=7.5, color=TEAL)
    d.text("front_dim", 65, 100.4, "wₜ · 40² view", size=7.5, color=TEAL)
    d.ax.imshow(np.linspace(0, 1, 128)[None], cmap=exposure_cmap,
                norm=Normalize(0, 1), extent=(60, 70, 104.9, 103.5),
                aspect="auto", interpolation="nearest", zorder=6)
    d.obj("front_scale", (60, 103.5, 10, 1.4), "absolute_colour_scale")
    d.text("front_scale_zero", 58.5, 104.2, "0", size=6.5, color=GRAY, ha="right")
    d.text("front_scale_one", 71.5, 104.2, "1", size=6.5, color=GRAY, ha="left")
    d.edge("previous_front", "PREV", "FRONT", [(37.5, 88), (55.5, 88)],
           color=TEAL, meaning="Six-face maximum predicted water occupancy at initial oil sites")
    d.edge("geometry_front", "X", "FRONT",
           [(4.1, 33), (2.5, 33), (2.5, 70), (82, 70), (82, 83), (74.5, 83)],
           color=GRAY, meaning="Fixed inlet and connected-pore geometry; reachable upper support m_t",
           crossings=("previous_regional",))
    # Distinguish the static/dynamic crossing from a graph junction.
    d.ax.plot([10.4, 13.6], [70, 70], color="white", lw=3.5, zorder=11)
    bridge = MplPath([(10.4, 70), (10.4, 67.8), (13.6, 67.8), (13.6, 70)],
                     [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    d.ax.add_patch(PathPatch(bridge, lw=.8, edgecolor=GRAY, facecolor="none", zorder=12))
    d.text("geometry_label", 40, 64, "I, C", size=7.5, color=GRAY)
    d.text("support_label", 67, 66.5, "mₜ", size=7.5, color=GRAY)
    d.circle("PRODUCT", 88, 88, 2.6, "×", TEAL)
    d.edge("front_hazard", "FRONT", "PRODUCT", [(74.5, 88), (85.4, 88)],
           color=TEAL, meaning="Multiply exposure and learned propensity")
    d.edge("propensity_hazard", "SIG", "PRODUCT",
           [(172, 65.6), (172, 71), (101, 71), (101, 78), (88, 78), (88, 85.4)],
           meaning="Learned site propensity modulated by causal spatial exposure")
    d.text("hazard_label", 96, 84, "hₜ", color=TEAL)
    d.box("UPDATE", 103, 82.5, 33, 11, color=PURPLE, face="#faf8fc")
    d.text("update_title", 119.5, 85.5, "State update", size=7.5, weight="bold",
           color=PURPLE, parent="UPDATE")
    d.text("update_equation", 119.5, 90.5, "pₜ = pₜ₋₁(1 − hₜ)", size=7.8,
           color=PURPLE, parent="UPDATE")
    d.edge("hazard_update", "PRODUCT", "UPDATE", [(90.6, 88), (103, 88)],
           color=TEAL, meaning="Nonnegative multiplicative site update")
    d.edge("previous_update", "PREV", "UPDATE",
           [(36, 97.5), (43, 97.5), (43, 107), (109, 107), (109, 93.5)],
           color=PURPLE, meaning="Previous occupancy multiplied by one minus hazard")
    d.volume("OUTPUT", output_im, output_wire, 163, 91, 30)
    d.text("output_dim", 163, 109, "pₜ · 128³", color=PURPLE)
    d.edge("update_scatter", "UPDATE", "OUTPUT", [(136, 88), (147.1, 88)],
           color=PURPLE, meaning="Scatter initial-oil site probabilities into the full 128-cubed volume")
    d.text("scatter_label", 141.5, 84, "scatter", size=6.5, color=PURPLE)
    d.edge("state_memory", "UPDATE", "PREV",
           [(122, 93.5), (122, 114), (11, 114), (11, 88), (18.5, 88)],
           color=PURPLE, dashed=True, meaning="One-step delayed predicted state; p0=1 at initial oil sites")
    d.text("rollout_label", 71, 117.8, "t = 1, …, T",
           size=7.8, color=PURPLE)
    d.circle("SUM", 138, 113.5, 2.6, "Σ", PURPLE)
    d.edge("update_totals", "UPDATE", "SUM",
           [(131, 93.5), (131, 113.5), (135.4, 113.5)],
           color=PURPLE, meaning="Aggregate continuous updated probabilities")
    d.text("totals_label", 143, 113.5, "Nₜ (5), RFₜ", size=7.5,
           color=PURPLE, ha="left")
    # Volume colour denotes phase; five initial classes remain in the tensors.
    labels, anchors, phase_ids = ["Water", "Oil"], [69, 101], [2, 7]
    for idx, (x, label, phase) in enumerate(zip(anchors, labels, phase_ids), 1):
        d.ax.add_patch(Rectangle((x, 123.2), 2.4, 2.4, fc=volume_palette[phase], ec="none", zorder=12))
        d.obj("legend_patch_"+str(idx), (x, 123.2, 2.4, 2.4), "legend")
        d.text("legend_text_"+str(idx), x+3.9, 124.4, label, size=7.5, ha="left")
    d.finish_geometry()
    require_matplotlib_panel_alignment(
        d.fig, axes=[d.ax], panel_ids=["continuous_architecture"],
        json_out=str(PREFIX)+".alignment.json", tolerance_pt=1.5, strict=True)
    d.fig.savefig(str(PREFIX)+".png", dpi=300)
    d.fig.savefig(str(PREFIX)+".pdf", dpi=600,
                  metadata={"CreationDate": None, "ModDate": None})
    d.fig.savefig(str(PREFIX)+".svg", dpi=600, metadata={"Date": None})
    plt.close(d.fig)
    # Legacy caption proofs are omitted; the manuscript owns its current caption.
    metadata_out = {
        "role": "current compact architecture for the ESWA manuscript; technical labels only, no title banner",
        "concept_reference": "figure1_statefront_dataflow_concept_only_20261005_v2.png",
        "AI_pixels_used": False, "feature_cache": FEATURES.name,
        "composition_references": [
            {"paper": "MeshGraphNets, ICLR 2021, Figure 1", "url": "https://arxiv.org/html/2010.03409v4", "adopted": "Real-state windows and clear iterative-state flow; no artwork or graph modules copied"},
            {"paper": "GraphCast, Science 2023, Figure 1, author manuscript", "url": "https://arxiv.org/html/2212.12794v2", "adopted": "Consistent 3D viewpoints, explicit representation changes and distinct rollout feedback; no artwork or architecture copied"},
        ],
        "volume_insets": SURFACES.with_suffix(".source.json").name,
        "composition_changes": COMPOSITION_DESCRIPTION,
        "volume_palette": volume_palette,
        "canvas_mm": [W, H], "prior_canvas_mm": [180, 150],
        "concept_pass": "Reuse completed v2 concept; localized data-inset and legend revision, no new topology or composition family",
        "feature_cache_sha256": digest(FEATURES),
        "same_fixed_case": metadata["pair_id"], "step": 100,
        "feature_selection": "Unchanged cached channels 0-2 for encoder/hidden-state windows, native z=64 and regional z=4. Class-logit window displays all five fixed-order class channels at z=4; no activation-based selection.",
        "class_logit_channel_order": ["Film", "Corner", "Droplet", "Columnar", "Cluster"],
        "feature_display_limits": scales,
        "front_display": "Full-domain maximum projection along z plus a linked 40x40 magnification at x=0:40,y=0:40, covering every nonzero projected value. Identical absolute 0-1 scale in both views; no rescaling or new inference.",
        "front_view_contract": {
            "reuse_level": "localized display adaptation of existing architecture",
            "source": "front_weight[z,y,x] in the unchanged cached transition 99 to 100",
            "full_projection_shape": list(front_projection.shape),
            "magnified_xy_bounds_half_open": [0, 40, 0, 40],
            "full_nonzero_pixels": int(np.count_nonzero(front_projection)),
            "magnified_nonzero_pixels": int(np.count_nonzero(front_projection[:40, :40])),
            "normalization": "linear absolute range [0,1] in full and magnified views",
            "scientific_edges_changed": False
        },
        "display_threshold_only": .5,
        "actual_network": metadata["model"], "new_inference": False,
        "class_identity": "Fixed initial slice-derived labels; not reclassification at each time or independently validated 3D morphology categories",
        "single_crossing_bridge": "geometry_front passes over previous_regional, no graph junction",
        "artifact_sha256": {suffix: digest(Path(str(PREFIX)+"."+suffix)) for suffix in ("png", "svg", "pdf")},
        "caption": CAPTION,
        "qa_scope": "Saved-output reproduction; legacy caption proof omitted; manuscript embedding unchanged",
    }
    Path(str(PREFIX)+".source.json").write_text(json.dumps(metadata_out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": "RENDERED_ARCHITECTURE_CANDIDATE", "prefix": str(PREFIX)}), flush=True)


if __name__ == "__main__":
    raise SystemExit("Use reproduce_architecture.py")
