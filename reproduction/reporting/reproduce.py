"""Generate tables and quantitative figures using only the enclosed saved results."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from reporting import verify_inputs, export_tables, read_json, write_csv, digest

def export_figure3_data(inputs, out):
    import numpy as np
    saved = read_json(inputs / "long_horizon.json")
    curves, blocks, gains = [], [], []
    for model, values in saved["models"].items():
        if model not in ("state_front", "static_front", "rtu", "fno"):
            continue
        for metric in ("site", "response"):
            data = np.asarray([saved["seed_records"][model][str(s)]["horizons"]["100"][metric+"_by_step"]
                               for s in saved["seeds"]], dtype=float)
            if data.shape != (3, 100) or not np.isfinite(data).all():
                raise ValueError("Invalid saved per-step result")
            for step in range(100):
                curves.append({"model": model, "metric": metric, "step": step + 1,
                               "mean_x1000": 1000 * data[:, step].mean(),
                               "sample_sd_x1000": 1000 * data[:, step].std(ddof=1)})
        for block, mean, sd in zip(saved["blocks"], values["blocks_site_51_100"]["mean"],
                                   values["blocks_site_51_100"]["seed_sd"], strict=True):
            blocks.append({"model": model, "block": block, "site_mean_x1000": 1000 * mean,
                           "site_sample_sd_x1000": 1000 * sd})
        gains.append({"model": model, "mean_percent": values["continuation_gain_percent"]["mean"],
                      "sample_sd_percent": values["continuation_gain_percent"]["seed_sd"]})
    for name, rows in (("figure3_curves.csv", curves), ("figure3_blocks.csv", blocks),
                       ("figure3_continuation.csv", gains)):
        write_csv(out / name, rows)

def run(inputs, output, tables_only=False):
    started = time.perf_counter()
    if sys.flags.optimize:
        raise ValueError("Do not use -O: preserved figure input checks use assertions")
    if not tables_only:
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, wanted = line.split("==")
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed = "not installed"
            if installed != wanted:
                raise RuntimeError(f"{name}: expected {wanted}, found {installed}. "
                                   "Activate the matching environment or install requirements.lock.txt")
    manifest = verify_inputs(inputs)
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing results are never overwritten")
    if output == ROOT or output.is_relative_to(inputs) or inputs.is_relative_to(output):
        raise ValueError("Output must be separate from the input directory")
    output.mkdir(parents=True)
    # Cache belongs to this run; no skill directory or user-specific cache is required.
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    report = {"scope": "Saved-output reporting only; no training, model loading or evaluation",
              "input_manifest_sha256": digest(inputs / "manifest.json"),
              "input_files": len(manifest["files"]),
              "tables": export_tables(inputs, output / "tables"),
              "figures": [], "python": platform.python_version(), "platform": platform.system()}
    if not tables_only:
        import plot_frozen_test_dynamics_v1 as dynamics
        import plot_seedwise_bars_v2 as learning
        import plot_test_heterogeneity_v2 as heterogeneity
        figures = output / "figures"
        figures.mkdir()
        import matplotlib as mpl
        with mpl.rc_context():
            dynamics.render(figures, inputs / "long_horizon.json")
        with mpl.rc_context():
            learning.render(learning.load_rows(inputs / "learning_dev.json",
                                              inputs / "learning_test.json"), figures)
        pairs, seeds, classes, meta = heterogeneity.source_data(inputs)
        for suffix, rows in (("pairs", pairs), ("seeds", seeds), ("classes", classes)):
            write_csv(figures / f"{heterogeneity.STEM}.{suffix}.csv", rows)
        with mpl.rc_context():
            heterogeneity.render(figures, pairs, classes, meta)
        export_figure3_data(inputs, figures)
        report["figures"] = [dynamics.STEM, learning.STEM, heterogeneity.STEM]
        report["pair_panel"] = {"pairs": 600, "blocks": 5, "seeds": 3, "models": 4,
                                "excluded_pairs": 0, "saved_score_rows": 7200}
        report["versions"] = {name: importlib.metadata.version(name)
                              for name in ("numpy", "matplotlib", "pillow")}
        from matplotlib import font_manager
        report["font"] = Path(font_manager.findfont(
            font_manager.FontProperties(family=["Arial", "DejaVu Sans"]))).name
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (output / "reproduction_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "inputs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A new directory; existing directories are not overwritten")
    parser.add_argument("--tables-only", action="store_true",
                        help="Use only the Python standard library; skip figures")
    args = parser.parse_args()
    run(args.input_dir.resolve(), args.output_dir.resolve(), args.tables_only)

if __name__ == "__main__":
    main()
