"""Formal TEST, 600 paired schedules / 5 spatial blocks / 3 matched seeds.
Structural adaptation of plot_backbone_dynamics_v1.py; old DEV assets unchanged.
Figure contract: predicted-state exposure generalizes to held-out blocks and
sustains useful continuation beyond training. Four-panel quantitative grid:
a temporal state error; b temporal schedule-response error; c within-seed
continuation control; d block-resolved state error. Original scores, no fitting,
interpolation, inference or sample exclusion. Means +/- sample seed SD throughout.
Output: 180 x 112 mm, editable PDF/SVG, 600-dpi PNG/TIFF.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from plot_style import COLORS, STYLES, NAMES, MODELS, panel_label

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "inputs/long_horizon.json"
STEM = "figure10_frozen_test_dynamics_v1"


def endpoint_labels(ax, endpoints, decimals):
    """Keep endpoint values apart in physical points without moving the data."""
    ax.figure.canvas.draw()
    ordered = sorted(endpoints, key=lambda item: item[1])
    pixels_per_point = ax.figure.dpi / 72
    actual = np.array([ax.transData.transform((100, value))[1]
                       for _, value in ordered])
    placed = actual.copy()
    minimum_gap = 9 * pixels_per_point
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + minimum_gap)
    placed -= (placed - actual).mean()
    for (model, value), target, origin in zip(ordered, placed, actual):
        offset = (target - origin) / pixels_per_point
        leader = ({"arrowstyle": "-", "color": COLORS[model], "lw": .55,
                   "shrinkA": 1.5, "shrinkB": 1.5}
                  if abs(offset) > 1 else None)
        ax.annotate(f"{value:.{decimals}f}", xy=(100, value),
                    xytext=(4, offset), textcoords="offset points",
                    ha="left", va="center", fontsize=6.8, color="#263238",
                    arrowprops=leader, zorder=6)


def render(out, source=SOURCE):
    summary = json.loads(source.read_text(encoding="utf-8"))
    assert summary["status"] == "ALL_SAVED_AGGREGATES_REPRODUCED"
    assert summary["split"] == "TEST" and summary["pairs"] == 600
    assert len(summary["seeds"]) == 3 and len(summary["blocks"]) == 5
    data = summary["models"]
    from figure_layout import check_layout as require_matplotlib_panel_alignment
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.linewidth": .65,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, grid = plt.subplots(2, 2, figsize=(180/25.4, 112/25.4),
                             gridspec_kw={"height_ratios": [1.12, 1]})
    fig.subplots_adjust(left=.13, right=.985, top=.865, bottom=.17, wspace=.40, hspace=.48)
    a, b, c, d = grid.flat
    for ax, metric, scale, title, letter, ylabel, tick in (
        (a, "site", 1000, "Remaining-oil state", "a", "Site-state Brier error (×10⁻³)", 20),
        (b, "response", 1000, "Regional schedule response", "b", "Response MSE (×10⁻³)", 2)):
        arrays = {}
        for model in MODELS:
            arrays[model] = np.array([summary["seed_records"][model][str(s)]["horizons"]["100"][metric + "_by_step"]
                                      for s in summary["seeds"]]) * scale
            assert arrays[model].shape == (3, 100) and np.isfinite(arrays[model]).all()
        max_val = max((v.mean(0)+v.std(0, ddof=1)).max() for v in arrays.values())
        top = math.ceil(max_val * 1.16 / tick) * tick
        for model in reversed(MODELS):
            values = arrays[model]
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.fill_between(range(1,101), mean-sd, mean+sd, color=COLORS[model], alpha=.16, linewidth=0)
            ax.plot(range(1,101), mean, color=COLORS[model], linestyle=STYLES[model],
                    linewidth=1.7 if model == "state_front" else 1.25)
        ax.axvline(50, color="#8A9397", linewidth=.7, linestyle=(0,(2,2)))
        ax.set(xlim=(0,118), ylim=(0,top), xticks=(0,25,50,75,100),
               yticks=np.arange(0,top+tick/2,tick), xlabel="Simulator step", ylabel=ylabel)
        ax.yaxis.label.set_fontfamily("DejaVu Sans")
        endpoint_labels(ax, [(model, values.mean(0)[-1]) for model, values in arrays.items()],
                        1 if metric == "site" else 2)
        panel_label(fig, ax, letter, "")
    for index, model in enumerate(MODELS):
        mean = data[model]["continuation_gain_percent"]["mean"]
        sd = data[model]["continuation_gain_percent"]["seed_sd"]
        c.barh(index, mean, xerr=sd, height=.58, color=COLORS[model],
               edgecolor=COLORS[model], linewidth=.6,
               error_kw={"elinewidth":.8, "capsize":2, "capthick":.8, "ecolor":"#35433F"})
        c.annotate(f"{mean:.1f}%", xy=(mean + (sd if mean >= 0 else -sd), index),
                   xytext=(4 if mean >= 0 else -4, 0), textcoords="offset points",
                   ha="left" if mean >= 0 else "right", va="center",
                   fontsize=7, color="#263238")
    c.axvline(0, color="#8A9397", linewidth=.65)
    c.set(xlim=(-22,44), ylim=(3.65,-.65), xticks=(-10,0,10,20,30,40),
          yticks=range(4), yticklabels=[NAMES[m] for m in MODELS],
          xlabel="Error reduction vs holding step 50 (%)")
    c.tick_params(axis="y", length=0, pad=5)
    c.spines["left"].set_visible(False)
    panel_label(fig, c, "c", "")
    means = np.array([data[m]["blocks_site_51_100"]["mean"] for m in MODELS]) * 1000
    sds = np.array([data[m]["blocks_site_51_100"]["seed_sd"] for m in MODELS]) * 1000
    cmap = LinearSegmentedColormap.from_list("sage_error", ["#F8FBFA","#C2DBD2","#448C79"])
    upper = math.ceil(means.max()/10)*10
    im = d.imshow(means, cmap=cmap, norm=Normalize(0,upper), aspect="auto", interpolation="nearest")
    for row in range(means.shape[0]):
        for col in range(means.shape[1]):
            value = means[row, col]
            label = "0" if value == 0 else f"{value:.1f}"
            d.text(col, row, label, ha="center", va="center",
                   fontsize=7, color="#183B32")
    d.set_xticks(range(5), [f"B{i+1}" for i in range(5)])
    d.set_yticks(range(4), [NAMES[m] for m in MODELS])
    d.tick_params(axis="both", length=0, pad=5)
    for spine in d.spines.values():
        spine.set_visible(False)
    d.set_xticks(np.arange(-.5,5,1),minor=True)
    d.set_yticks(np.arange(-.5,4,1),minor=True)
    d.grid(which="minor",color="white",linewidth=1.2)
    d.tick_params(which="minor",length=0)
    pos = d.get_position()
    cbax = fig.add_axes([pos.x0,pos.y0-.082,pos.width,.012])
    cb = fig.colorbar(im,cax=cbax,orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7,length=2,width=.5,pad=2)
    cb.set_label("Site-state Brier error (×10⁻³)",fontsize=7,labelpad=2,fontfamily="DejaVu Sans")
    panel_label(fig,d,"d","")
    fig.legend([Line2D([0],[0],color=COLORS[m],linestyle=STYLES[m],lw=1.6) for m in MODELS],
               [NAMES[m] for m in MODELS], frameon=False,ncol=4,loc="upper center",
               bbox_to_anchor=(.54,.99),fontsize=7.5,handlelength=2.6,columnspacing=1.6)
    for ax in (a,b,c):
        ax.tick_params(width=.65,length=3)
    base = out / STEM
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=list(grid.flat),panel_ids=list("abcd"),
        exclude_axes=[cbax],tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        require_panel_labels=True,strict=True,json_out=base.with_suffix(".alignment.json"),
        overlay_svg=base.with_suffix(".alignment.svg"))
    fig.savefig(base.with_suffix(".pdf"),metadata={"Title":"Held-out long-horizon remaining-oil forecasting",
                                                  "Author":"","CreationDate":None,"ModDate":None})
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"),dpi=600)
    fig.savefig(base.with_suffix(".tiff"),dpi=600,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    metadata = {k:summary[k] for k in ("split","pairs","blocks","pairs_per_block","seeds","uncertainty","sources")}
    metadata.update(source_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    structure="quantitative grid",size_mm=[180,112],exclusions=0,new_inference=False,
                    mapping={"curves":"100-step pair/block means; site and response scaled by 1000; labels give step-100 means",
                             "gain":"100*(held50-continued)/held50 within seed, then mean/SD",
                             "heatmap":"per-block unseen site error, multiplied by 1000; colours and cell labels show seed means, seed SD remains in source data"},
                    adaptation="Structural adaptation of existing DEV plot; original assets preserved",
                    figure_guide="https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/")
    base.with_suffix(".source.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    render(args.output_dir)

