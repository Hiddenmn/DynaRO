"""Replace redundant horizon means with pair-level TEST heterogeneity.

Question: how do aggregate gains vary across schedule pairs and initial classes?
Panels a/b show paired differences, not another plot of published mean errors.
Panel c retains the distinct five-class inventory decomposition from the prior
figure. The horizon-average curve is omitted because Table S2 reports it.
All 600 pairs, five blocks and three matched seeds are retained. Box plots
describe per-pair seed-mean gains; bars show equal-block means +/- seed SD.
Python only; 180 x 108 mm; editable PDF/SVG plus 600-dpi PNG/TIFF.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "inputs"
STEM = "figure11_test_heterogeneity_v2"
SEEDS = (20260811, 20260812, 20260813)
MODELS = ("state_front", "static_front", "rtu", "fno")
COMPARATORS = MODELS[1:]
NAMES = {"state_front": "DynaRO", "static_front": "Static front", "rtu": "RTU", "fno": "FNO"}
COLORS = {"state_front": "#428F7D", "static_front": "#C97C9D", "rtu": "#8A8F9E", "fno": "#AA9B83"}
MARKERS = {"state_front": "o", "static_front": "s", "rtu": "^", "fno": "D"}
CLASSES = ("Film", "Corner", "Droplet", "Columnar", "Cluster")
TOLERANCE = 1e-12


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_data(analysis):
    records, blocks, sources = {}, {}, []
    path = analysis / "paired_scores_51_100.csv"
    sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            seed, model, pair = int(row["seed"]), row["model"], row["pair_id"]
            assert seed in SEEDS and model in MODELS
            assert blocks.setdefault(pair, row["block_id"]) == row["block_id"]
            slot = records.setdefault((model, seed), {})
            assert pair not in slot
            slot[pair] = {key: float(row[key]) for key in ("site", "response")}
    ids = sorted(blocks)
    assert len(ids) == 600 and sorted(Counter(blocks.values()).values()) == [120] * 5
    assert set(records) == {(m, s) for m in MODELS for s in SEEDS}
    assert all(sorted(value) == ids for value in records.values())
    pairs, seed_rows, stats = [], [], []
    summary = json.loads((analysis / "long_horizon.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ALL_SAVED_AGGREGATES_REPRODUCED"
    assert summary["split"] == "TEST" and summary["pairs"] == 600
    for endpoint in ("site", "response"):
        arrays = {model: np.array([[records[model, seed][pair][endpoint] for pair in ids]
                                   for seed in SEEDS]) for model in MODELS}
        assert all(np.isfinite(a).all() and (a >= 0).all() for a in arrays.values())
        for model, values in arrays.items():
            saved = summary["models"][model]["intervals"]["100"][endpoint]["mean"]
            assert np.isclose(values.mean(), saved, rtol=0, atol=1e-12)
        for model in COMPARATORS:
            deltas = arrays[model] - arrays["state_front"]
            gains = deltas.mean(axis=0)
            for i, pair in enumerate(ids):
                pairs.append({"endpoint": endpoint, "comparator": model, "pair_id": pair,
                              "block_id": blocks[pair], "seed_count": 3,
                              "comparator_error": arrays[model][:, i].mean(),
                              "stateguard_error": arrays["state_front"][:, i].mean(),
                              "gain": gains[i]})
                for j, seed in enumerate(SEEDS):
                    seed_rows.append({"endpoint": endpoint, "comparator": model, "seed": seed,
                                      "pair_id": pair, "block_id": blocks[pair],
                                      "comparator_error": arrays[model][j, i],
                                      "stateguard_error": arrays["state_front"][j, i],
                                      "gain": deltas[j, i]})
            stats.append({"endpoint": endpoint, "comparator": model,
                          "improved": int((gains > TOLERANCE).sum()),
                          "tied": int((np.abs(gains) <= TOLERANCE).sum()),
                          "worse": int((gains < -TOLERANCE).sum()),
                          "quantiles_x1000": (1000 * np.quantile(gains, [0, .25, .5, .75, 1])).tolist(),
                          "tie_blocks": dict(Counter(blocks[pair] for pair, gain in zip(ids, gains)
                                                     if abs(gain) <= TOLERANCE))})
    class_rows = []
    for model in MODELS:
        values = np.asarray(summary["models"][model]["class_mae_100"]["seeds"])
        assert values.shape == (3, 5) and np.isfinite(values).all()
        for stat, expected in ((values.mean(0), "mean"), (values.std(0, ddof=1), "seed_sd")):
            assert np.allclose(stat, summary["models"][model]["class_mae_100"][expected], rtol=0, atol=1e-12)
        for i, seed in enumerate(SEEDS):
            for j, name in enumerate(CLASSES):
                class_rows.append({"model": model, "seed": seed, "initial_class": name,
                                   "inventory_mae_pp": 100 * values[i, j]})
    metadata = {"question": "How broadly do mean gains extend across pairs and initial classes?",
                "split": "TEST", "pairs": 600, "spatial_blocks": 5, "pairs_per_block": 120,
                "seeds": SEEDS, "interval": [51, 100], "class_endpoint": 100,
                "paired_gain": "Comparator minus DynaRO, averaged over matched seeds within pair",
                "box_summary": "Median, 25th-75th percentiles; whiskers are minimum-maximum, all pairs retained",
                "bar_summary": "Equal-block mean +/- sample SD over three seeds",
                "independence": "Pairs nested in five spatial blocks; seeds repeat the same 600 pairs",
                "inference": "Descriptive; no significance tests or confidence intervals",
                "exclusions": 0, "new_model_inference": False, "classification_tolerance": TOLERANCE,
                "paired_counts": stats, "sources": sources,
                "summary_sha256": hashlib.sha256((analysis / "long_horizon.json").read_bytes()).hexdigest(),
                "removed_panel": "Horizon inventory curve duplicated the values in Supplementary Table S2",
                "retained_panel": "Five-class error decomposition, unchanged data from previous Figure S5",
                "style_inheritance": "Established method colors and typography; new paired-distribution layout"}
    return pairs, seed_rows, class_rows, metadata


def panel_label(fig, ax, letter, title):
    transform = ax.transAxes + ScaledTranslation(-13/72, 6/72, fig.dpi_scale_trans)
    ax.text(0, 1, letter, transform=transform, fontsize=10, fontweight="bold", va="bottom")


def render(out, pairs, class_rows, metadata):
    from figure_layout import check_layout as require_matplotlib_panel_alignment
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7.5, "axes.linewidth": .65,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": STEM,
                         "hatch.linewidth": .28, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.0866141732, 4.2519685039))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], left=.105, right=.985,
                           bottom=.20, top=.91, wspace=.42, hspace=.55)
    a, b, c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    for ax, endpoint, letter, title, face, edge in (
        (a, "site", "a", "Pairwise spatial-state gain", "#B5D9CC", "#428F7D"),
        (b, "response", "b", "Pairwise schedule-response gain", "#E8BDD0", "#AC6083")):
        values = [1000 * np.array([float(row["gain"]) for row in pairs
                                  if row["endpoint"] == endpoint and row["comparator"] == model])
                  for model in COMPARATORS]
        assert all(v.shape == (600,) and np.isfinite(v).all() for v in values)
        ax.boxplot(values, orientation="horizontal", positions=range(3), widths=.44,
                   whis=(0, 100), showfliers=False, patch_artist=True,
                   boxprops={"facecolor": face, "edgecolor": edge, "linewidth": .8},
                   whiskerprops={"color": edge, "linewidth": .8},
                   capprops={"color": edge, "linewidth": .8},
                   medianprops={"color": "#263C35", "linewidth": 1.3})
        ax.axvline(0, color="#909995", linewidth=.65, linestyle=(0, (2, 2)), zorder=0)
        ax.set(yticks=range(3), yticklabels=[NAMES[m] for m in COMPARATORS], ylim=(2.6, -.6))
        ax.set_xlim((-2.5, 63) if endpoint == "site" else (-.95, 7.3))
        ax.set_xticks([0, 20, 40, 60] if endpoint == "site" else [0, 2, 4, 6])
        ax.set_xlabel(("Site Brier reduction" if endpoint == "site" else "Response MSE reduction")
                      + " (×10⁻³)", fontfamily="DejaVu Sans", labelpad=5)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.spines["left"].set_visible(False)
        panel_label(fig, ax, letter, title)
    width = .18
    max_error = 0
    for i, model in enumerate(MODELS):
        values = np.array([[float(row["inventory_mae_pp"]) for row in class_rows
                            if row["model"] == model and int(row["seed"]) == seed]
                           for seed in SEEDS])
        assert values.shape == (3, 5)
        mean, sd = values.mean(0), values.std(0, ddof=1)
        max_error = max(max_error, (mean + sd).max())
        c.bar(np.arange(5) + (i - 1.5) * width, mean, yerr=sd, width=width*.91,
              color=COLORS[model], edgecolor="#35443F", linewidth=.4,
              error_kw={"elinewidth": .75, "capsize": 2, "capthick": .75, "ecolor": "#35443F"}, zorder=3)
        c.plot(np.arange(5) + (i - 1.5) * width, mean, linestyle="none", marker=MARKERS[model],
               markersize=3, color="#35443F", markeredgewidth=.4, zorder=4)
        for x, value, spread in zip(np.arange(5) + (i - 1.5) * width, mean, sd):
            c.annotate(f"{value:.2f}", xy=(x, value + spread), xytext=(0, 4),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=6.5, color="#293745", zorder=5)
    ymax = np.ceil((max_error + .7) / 2) * 2
    c.set(xticks=range(5), xticklabels=CLASSES, xlim=(-.55, 4.55), ylim=(0, ymax),
          yticks=np.arange(0, ymax+.01, 2), ylabel="Inventory MAE (percentage points)")
    c.set_axisbelow(True)
    c.grid(False)
    panel_label(fig, c, "c", "Initial-class inventory error at step 100")
    fig.legend([Line2D([0], [0], color=COLORS[m], linewidth=2.5, marker=MARKERS[m],
                      markersize=3.5, markeredgewidth=.4, markeredgecolor="#35443F")
                for m in MODELS], [NAMES[m] for m in MODELS], loc="upper center",
               bbox_to_anchor=(.55, .12), ncol=4, frameon=False, fontsize=7.5,
               handlelength=1.5, columnspacing=1.8)
    for ax in (a, b, c):
        ax.tick_params(axis="x", width=.65, length=3)
    base = out / STEM
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[a, b, c], panel_ids=list("abc"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
        json_out=base.with_suffix(".alignment.json"), overlay_svg=base.with_suffix(".alignment.svg"))
    fig.savefig(base.with_suffix(".pdf"), metadata={"Title": "Test-pair and class-resolved forecasting errors",
                                                  "Author": "", "CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".preview.png"), dpi=300)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    metadata.update(size_mm=[180, 108], glyph_size_floor_pt=6.5,
                    archetype="Paired-distribution comparison with subordinate class decomposition")
    base.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs, seeds, classes, metadata = source_data(args.input_dir)
    write_csv(args.output_dir / f"{STEM}.pairs.csv", pairs)
    write_csv(args.output_dir / f"{STEM}.seeds.csv", seeds)
    write_csv(args.output_dir / f"{STEM}.classes.csv", classes)
    render(args.output_dir, pairs, classes, metadata)
    print(json.dumps({"pairs": 600, "comparisons": metadata["paired_counts"],
                      "output": str(args.output_dir / STEM)}, indent=2))


if __name__ == "__main__":
    main()
