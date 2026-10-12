"""Plot matched-seed state-prioritized learning gains from existing DEV and locked TEST summaries.

The plotted comparison is the fixed response-weight=50 checkpoint at the same
seed and split. No checkpoint, trajectory, or prediction files are opened.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEV_SUMMARY = HERE.parent / "inputs/learning_dev.json"
TEST_SUMMARY = HERE.parent / "inputs/learning_test.json"
STEM = "figure4_seedwise_bars_v2"
SEEDS = (20260811, 20260812, 20260813)


def load_rows(dev_path=DEV_SUMMARY, test_path=TEST_SUMMARY) -> list[dict[str, float | int | str]]:
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    if dev["status"] != "COMPLETE_STATEGUARD_THREE_SEED_ADAPTIVE_DEV_AGGREGATION":
        raise ValueError("DEV source is not the completed three-seed aggregate")
    if test["status"] != "COMPLETE_FROZEN_LOCKED_TEST" or test["pair_count"] != 600:
        raise ValueError("Locked TEST source is not the completed 600-pair evaluation")

    rows: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        dev_scores = dev["per_seed"][str(seed)]["equal_block_mean"]
        test_scores = {
            arm: test["per_checkpoint"][f"{arm}/{seed}"]["overall_equal_block_mean"]["all_25"]
            for arm in ("response50", "stateguard")
        }
        for split, pair_count, scores in (
            ("Development", 400, dev_scores),
            ("Locked test", 600, test_scores),
        ):
            fixed, state = scores["response50"], scores["stateguard"]
            if fixed["site"] <= 0 or fixed["response"] <= 0:
                raise ValueError("Comparator errors must be positive for relative reductions")
            rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "pairs": pair_count,
                    "fixed50_site": fixed["site"],
                    "stateguard_site": state["site"],
                    "fixed50_response": fixed["response"],
                    "stateguard_response": state["response"],
                    "site_reduction_pct": 100 * (fixed["site"] - state["site"]) / fixed["site"],
                    "response_reduction_pct": 100 * (fixed["response"] - state["response"]) / fixed["response"],
                }
            )
    return rows


def render(rows: list[dict[str, float | int | str]], outdir: Path) -> None:
    from figure_layout import check_layout as require_matplotlib_panel_alignment

    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / STEM
    with (base.with_suffix(".source.csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "savefig.facecolor": "white",
        }
    )

    # Individual-seed bars retain the paired gains and the negative DEV response.
    from matplotlib.patches import Patch
    from matplotlib.transforms import ScaledTranslation
    state_color, response_color = "#61AC9C", "#CE8CA5"
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 66 / 25.4), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.22, top=0.81, wspace=0.36)
    splits = (("Development", "Development · 400 pairs"),
              ("Locked test", "Test · 600 pairs"))
    for letter, ax, (split, title) in zip("ab", axes, splits):
        selected = [row for row in rows if row["split"] == split]
        if [int(row["seed"]) for row in selected] != list(SEEDS):
            raise ValueError("All three matched seeds must remain in order")
        for y, row in enumerate(selected):
            for key, shift, color, edge in (
                ("site_reduction_pct", -0.16, state_color, "#378575"),
                ("response_reduction_pct", 0.16, response_color, "#A66580"),
            ):
                value = float(row[key])
                ax.barh(y + shift, value, height=0.27, color=color,
                        edgecolor=edge, linewidth=0.45, zorder=3)
                ax.annotate(f"{value:.2f}%", xy=(value, y + shift),
                            xytext=(3 if value >= 0 else -3, 0),
                            textcoords="offset points",
                            ha="left" if value >= 0 else "right", va="center",
                            fontsize=7, color="#293745", zorder=4)
        ax.set_yticks(range(3), [str(seed) for seed in SEEDS])
        ax.set_ylim(2.55, -0.55)
        ax.set_xlim(-1.6, 5.8)
        ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
        ax.axvline(0, color="#7E858B", linewidth=0.7, zorder=2)
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", length=3)
        ax.set_xlabel("Error reduction vs fixed weight 50 (%)", labelpad=6)
        transform = ax.transAxes + ScaledTranslation(-13 / 72, 6 / 72, fig.dpi_scale_trans)
        ax.text(0, 1, letter, transform=transform, fontsize=10,
                fontweight="bold", va="bottom")
    fig.legend(handles=[
        Patch(facecolor=state_color, edgecolor="#378575", label="Site-state Brier"),
        Patch(facecolor=response_color, edgecolor="#A66580", label="Paired-response MSE"),
    ], loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=2,
       frameon=False, fontsize=8, handlelength=1.4, columnspacing=2.4)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base.with_suffix(".alignment.json")),
        overlay_svg=str(base.with_suffix(".alignment.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"Author": "", "CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    render(load_rows(), args.output_dir)


if __name__ == "__main__":
    main()


