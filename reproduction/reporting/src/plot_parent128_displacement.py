"""One frozen DEV case: matched 3D states, spatial change and continuous scores.

Real cached arrays only; no inference, smoothing, cropping, or sample selection.
The visual question is whether the model follows a substantial internal oil
displacement over time, including its spatial and inventory discrepancies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
from parent_volume_renderer import header_frame, plot_style, render, volume_panel, PALETTE, CAMERA
from figure_layout import check_layout as require_matplotlib_panel_alignment
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import psutil

SOURCE = HERE.parent / "inputs_3d/dev_parent_prediction.npz"
GREEN, PINK = "#559D7A", "#C878A2"
WIDTH_MM, HEIGHT_MM = 180, 180
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "pdf.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "max-mobilization-prediction-v1", "image.composite_image": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "legend.frameon": False,
})


def draw(prefix, SOURCE=SOURCE):
    prefix = prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began = time.perf_counter()
    provenance = json.loads(SOURCE.with_suffix(".source.json").read_text(encoding="utf-8"))
    if (provenance["pair_id"] != "z0_y256_x256|NPI-012|P1-22-DEV-A|S3~S5"
            or provenance["network_forward_calls"] != 1
            or provenance["schedules_evaluated"] != ["S5x30 -> S3x70"]):
        raise ValueError("The approved single-case cache differs")
    with np.load(SOURCE, allow_pickle=False) as archive:
        initial = archive["initial_labels"]
        truth = archive["truth_labels_a"]
        probability = archive["prediction_a"]
        true_counts = archive["true_class_counts_a"].sum(1)
        predicted_counts = archive["predicted_class_counts_a"].sum(1)
        brier = archive["site_brier_by_step_a"]
        steps = archive["steps"].tolist()
    if (initial.shape != (128, 128, 128) or np.any(initial == 0)
            or steps != [0, 25, 50, 75, 100]
            or not np.isfinite(probability).all()):
        raise ValueError("Expected complete, background-free cached 128-cubed arrays")
    oil0 = initial >= 3
    n0 = int(oil0.sum())
    if n0 != 295851 or int(true_counts[-1]) != 202570:
        raise ValueError("The pre-inference source selection changed")
    true_rf = 100 * (1 - true_counts / n0)
    predicted_rf = 100 * (1 - predicted_counts / n0)
    remaining = (probability >= .5) & oil0[None]
    true_changed = oil0 & (truth[-1] == 2)
    initial_scene = render(oil0, water=initial == 2)
    scenes = [initial_scene]
    for index, prediction in ((2, False), (2, True)):
        print(f"Rendering step {steps[index]} {'prediction' if prediction else 'simulator'}", flush=True)
        mask = remaining[index] if prediction else truth[index] >= 3
        water = ((initial == 2) | (oil0 & ~mask)) if prediction else truth[index] == 2
        scenes.append(render(mask, water=water))
    scenes.append(render(true_changed, mobilized=True, ghost=initial_scene["oil_rgb"]))
    for prediction in (False, True):
        print(f"Rendering step 100 {'prediction' if prediction else 'simulator'}", flush=True)
        mask = remaining[-1] if prediction else truth[-1] >= 3
        water = ((initial == 2) | (oil0 & ~mask)) if prediction else truth[-1] == 2
        scenes.append(render(mask, water=water))

    edges = np.arange(0, 129, 16)
    centers = (edges[:-1] + edges[1:] - 1) / 2
    denominators = np.asarray([oil0[:, :, a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    true_depth = np.asarray([true_changed[:, :, a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    expected_changed = oil0.astype(np.float64) - probability[-1].astype(np.float64)
    expected_depth = np.asarray([expected_changed[:, :, a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    if np.any(denominators <= 0) or abs(expected_depth.sum() - (n0 - predicted_counts[-1])) > .2:
        raise ValueError("Depth and inventory definitions disagree")
    true_depth_percent = 100 * true_depth / denominators
    predicted_depth_percent = 100 * expected_depth / denominators

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    columns, rows = [.012, .345, .678], [.666, .344, .022]
    w, h = .310, .310 / 1.06
    axes = []
    titles = ["Initial · step 0", "Simulator · step 50", "Prediction · step 50",
              "Mobilized oil · step 100", "Simulator · step 100", "Prediction · step 100"]
    for k, (scene, title) in enumerate(zip(scenes, titles)):
        row, col = divmod(k, 3)
        axes.append(volume_panel(fig, [columns[col], rows[row], w, h], scene, "abcdef"[k], ""))
    for col, (letter, title) in enumerate(zip("ghi", ("Cumulative mobilization", "Spatial profile · step 100", "Oil-state prediction error"))):
        axes.append(header_frame(fig, [columns[col], rows[2], w, h], letter, ""))
    plots = [fig.add_axes([x + .059, .080, .235, .205]) for x in columns]
    inventory, depth, error = plots
    inventory.plot(np.arange(101), true_rf, color=GREEN, lw=1.4, label="Simulator")
    inventory.plot(np.arange(101), predicted_rf, color=PINK, lw=1.4, ls="--", label="Prediction")
    inventory.axvline(30, color="#a4abb2", lw=.65, ls=":")
    inventory.set(xlim=(0, 132), ylim=(0, 43), xticks=(0, 50, 100), yticks=(0, 20, 40),
                  xlabel="Simulator step", ylabel="Initial oil mobilized (%)")
    for value in (true_rf[-1], predicted_rf[-1]):
        inventory.annotate(f"{value:.2f}%", xy=(100, value), xytext=(4, 0),
                           textcoords="offset points", ha="left", va="center",
                           fontsize=6.5, color="#293745", zorder=5)
    inventory.legend(loc="lower left", bbox_to_anchor=(0, 1.025), fontsize=6.5,
                     handlelength=1.1, ncol=2, columnspacing=.8, borderaxespad=0)
    depth.plot(centers, true_depth_percent, color=GREEN, lw=1.25, marker="o", markersize=2.6,
               label="Simulator")
    depth.plot(centers, predicted_depth_percent, color=PINK, lw=1.25, ls="--", marker="s", markersize=2.6,
               label="Prediction")
    depth.set(xlim=(0, 128), ylim=(0, 100), xticks=(0, 64, 128), yticks=(0, 50, 100),
              xlabel="Inlet-axis position (voxels)", ylabel="Local oil mobilized (%)")
    depth.legend(loc="lower left", bbox_to_anchor=(0, 1.025), fontsize=6.5,
                 handlelength=1.1, ncol=2, columnspacing=.8, borderaxespad=0)
    error.plot(np.arange(1, 101), 1000*brier, color=PINK, lw=1.4, label="Prediction vs simulator")
    error.axvline(30, color="#a4abb2", lw=.65, ls=":")
    error.set(xlim=(0, 128), ylim=(0, 105), xticks=(0, 50, 100), yticks=(0, 50, 100),
              xlabel="Simulator step", ylabel="Site-state Brier error (×10⁻³)")
    error.yaxis.label.set_fontfamily("DejaVu Sans")
    error.annotate(f"{1000*brier[-1]:.1f}", xy=(100, 1000*brier[-1]), xytext=(4, 0),
                   textcoords="offset points", ha="left", va="center",
                   fontsize=6.5, color="#293745", zorder=5)
    error.legend(loc="lower left", bbox_to_anchor=(0, 1.025), fontsize=6.5,
                 handlelength=1.1, borderaxespad=0)
    for ax in plots:
        plot_style(ax)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
    fig.legend(handles=[Patch(facecolor=PALETTE[7], label="Remaining oil"),
                        Patch(facecolor=PALETTE[2], label="Water"),
                        Patch(facecolor=PALETTE[6], label="Mobilized oil")],
               loc="lower center", bbox_to_anchor=(.5, .002), ncol=3,
               fontsize=7.2, columnspacing=2, handlelength=1)
    require_matplotlib_panel_alignment(fig, axes=axes[:6] + plots, panel_ids=list("abcdefghi"),
        row_groups=[list("abc"), list("def"), list("ghi")],
        column_groups=[list("ad"), list("be"), list("cf")],
        json_out=str(prefix) + ".alignment.json", strict=True)
    fig.savefig(str(prefix) + ".png", dpi=300, facecolor="white")
    fig.savefig(str(prefix) + ".pdf", dpi=600, facecolor="white")
    fig.savefig(str(prefix) + ".svg", dpi=600, facecolor="white")
    plt.close(fig)
    caption = (
        "A complete 128³ DEV volume (NPI-012, S5×30→S3×70) fixed before inference by the largest "
        "simulator mobilized-site count among 240 background-free trajectories. The block starts at "
        "z=0 of the acquired stack; it contains no rock-exterior background. (a) Initial oil and water. "
        "(b,c,e,f) Matched simulator and predicted states at steps 50 and 100. (d) Oil mobilized by "
        "step 100, with initial oil support in pale grey. All views share a native-grid orthographic "
        "camera, scale and illumination; no smoothing or additional crop is applied. Oil classes "
        "are pooled for display and rock is hidden. Predicted surfaces use a fixed 0.5 occupancy "
        "threshold. (g) Mobilized fraction from simulator inventories and unthresholded predicted "
        f"probabilities, reaching {true_rf[-1]:.2f}% and {predicted_rf[-1]:.2f}% at step 100. "
        "(h) Step-100 mobilization relative to initial oil in each of eight non-overlapping 16-voxel x slabs, "
        "using the same unthresholded probabilities. (i) Mean squared occupancy-probability error "
        "over initially oil-filled sites, displayed in units of 10^-3. Green solid lines denote the simulator; pink dashed lines "
        "denote predictions in (g,h), and the pink line in (i) denotes prediction error. "
        "Dotted lines mark the action change after step 30. "
        "Curves show this one sample and one frozen seed, not population means or uncertainty. "
        "The displacement-selected example is not representative of aggregate performance. "
        "Coordinates and simulator steps are not calibrated physical units.")
    metadata = {
        "source": SOURCE.name, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "pair_id": provenance["pair_id"], "seed": provenance["seed"],
        "selection": provenance["initial_geometry_selection"], "shape_zyx": [128, 128, 128],
        "camera": CAMERA, "native_grid": True, "smoothing": False, "additional_crop": False,
        "hidden_phase": "rock", "pooled_oil_display_only": True,
        "display_threshold": .5, "statistics_use_continuous_probability": True,
        "water_opacity": .2, "oil_opacity": 1.0, "phase_palette": PALETTE.tolist(),
        "statistics_colors": {"simulator": GREEN, "prediction": PINK},
        "site_brier_display_multiplier": 1000,
        "simulator_mobilized_percent_at100": float(true_rf[-1]),
        "predicted_expected_mobilized_percent_at100": float(predicted_rf[-1]),
        "curve": {"steps": list(range(101)), "simulator_percent": true_rf.tolist(),
                  "prediction_percent": predicted_rf.tolist(), "site_brier_steps1_100": brier.tolist()},
        "depth": {"edges": edges.tolist(), "initial_oil": denominators.tolist(),
                  "simulator_percent": true_depth_percent.tolist(),
                  "prediction_percent": predicted_depth_percent.tolist()},
        "caption": caption,
        "references": ["https://www.pnas.org/doi/10.1073/pnas.2305890120",
                       "https://proceedings.mlr.press/v235/wu24r.html"],
        "reference_adaptation": "Matched viewpoints, aligned truth/prediction panels, separate change and quantitative views; no borrowed images",
        "new_inference_by_renderer": False, "test_accessed": False,
        "elapsed_seconds": time.perf_counter() - began,
    }
    prefix.with_suffix(".display.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "SINGLE_CASE_FIGURE_COMPLETE", "prefix": str(prefix),
                      "truth_percent": float(true_rf[-1]), "prediction_percent": float(predicted_rf[-1]),
                      "seconds": metadata["elapsed_seconds"]}), flush=True)
    return metadata


if __name__ == "__main__":
    raise SystemExit("Use reproduce_parent3d.py to validate inputs and protect existing outputs.")
