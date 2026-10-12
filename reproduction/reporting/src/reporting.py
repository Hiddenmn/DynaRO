"""Reproduce all main and Supplement result tables from saved summaries."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

SEEDS = (20260811, 20260812, 20260813)
MODELS = ("state_front", "static_front", "rtu", "fno")
MODEL_NAMES = ("DynaRO", "Static front", "RTU", "FNO")
ARMS = ("response0", "response50", "du_gradcos", "stateguard")
ARM_NAMES = ("Response weight 0", "Fixed weight 50", "Du-style cosine gate", "State-prioritized")
INPUT_NAMES = {"long_horizon.json", "exposure_ablation.json", "learning_test.json",
               "learning_dev.json", "paired_scores_51_100.csv", "expected_table_displays.json",
               "dev_secondary_metrics.csv", "expected_supplement_displays.json",
               "du_dev_20260811.json", "du_dev_20260812.json", "du_dev_20260813.json",
               "pcgrad_dev_20260811.json"}

def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_inputs(root):
    manifest = read_json(root / "manifest.json")
    if manifest["schema_version"] != 2 or set(manifest["files"]) != INPUT_NAMES:
        raise ValueError("Unexpected saved-input manifest")
    for name, record in manifest["files"].items():
        path = root / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Saved input changed: {name}")
    return manifest

def reduction(comparator, proposed):
    if not math.isfinite(comparator) or comparator <= 0 or not math.isfinite(proposed):
        raise ValueError("A relative reduction needs a positive finite comparator")
    return 100 * (1 - proposed / comparator)

def score(record, metric, horizon):
    return (record["inventory_mae_100"] if metric == "inventory"
            else record["intervals"][str(horizon)][metric])

def spans():
    return [(metric, h, str(h) if metric == "inventory" else f"{1 if h < 75 else 51}-{h}")
            for metric in ("site", "response", "inventory")
            for h in ((100,) if metric == "inventory" else (25, 50, 75, 100))]

def make_main_tables(inputs):
    long = read_json(inputs / "long_horizon.json")
    coarse = read_json(inputs / "exposure_ablation.json")
    learning = read_json(inputs / "learning_test.json")
    if (long["status"] != "ALL_SAVED_AGGREGATES_REPRODUCED" or long["split"] != "TEST"
            or long["pairs"] != 600 or tuple(long["seeds"]) != SEEDS
            or len(long["blocks"]) != 5):
        raise ValueError("Unexpected long-horizon protocol")
    if (coarse["status"] != "THREE_SEED_MATCHED_TEST_ABLATION_VERIFIED"
            or coarse["pairs"] != 600 or tuple(coarse["seeds"]) != SEEDS):
        raise ValueError("Unexpected exposure comparison")
    if learning["status"] != "COMPLETE_FROZEN_LOCKED_TEST" or learning["pair_count"] != 600:
        raise ValueError("Unexpected 25-step learning comparison")
    tokens, rows = {key: [] for key in ("table1", "table2", "table3")}, {}
    records1, records2, records3 = [], [], []
    labels = {"site": "Site Brier", "response": "Response MSE", "inventory": "Inventory MAE (pp)"}
    wide1, wide2, wide3 = [], [], []
    for metric, horizon, span in spans():
        scale = 100 if metric == "inventory" else 1000
        line, display = [span, labels[metric]], []
        for model in MODELS:
            rec = score(long["models"][model], metric, horizon)
            mean, sd = [float(rec[k]) * scale for k in ("mean", "seed_sd")]
            display.extend((f"{mean:.3f}", f"{sd:.3f}"))
            line.append(f"{mean:.3f} +/- {sd:.3f}")
            records1.append({"steps": span, "metric": metric, "model": model, "n_seeds": 3,
                             "raw_mean": rec["mean"], "raw_sample_sd": rec["seed_sd"],
                             "display_multiplier": scale, "display_mean": f"{mean:.3f}",
                             "display_sample_sd": f"{sd:.3f}"})
        tokens["table1"].append(display)
        wide1.append(line)
        rcoarse = score(coarse["models"]["coarse_front"], metric, horizon)
        rfine = score(coarse["models"]["state_front"], metric, horizon)
        canonical = score(long["models"]["state_front"], metric, horizon)
        if any(not math.isclose(rfine[k], canonical[k], rel_tol=0, abs_tol=1e-13)
               for k in ("mean", "seed_sd")):
            raise ValueError("DynaRO scores disagree between the two saved summaries")
        gain = reduction(rcoarse["mean"], rfine["mean"])
        values = [f"{rcoarse[k] * scale:.3f}" for k in ("mean", "seed_sd")]
        tokens["table2"].append(values + [f"{gain:.2f}"])
        wide2.append([span, labels[metric], f"{values[0]} +/- {values[1]}", f"{gain:.2f}%"])
        records2.append({"steps": span, "metric": metric, "n_seeds": 3,
                         "coarse_raw_mean": rcoarse["mean"], "coarse_raw_sample_sd": rcoarse["seed_sd"],
                         "fine_raw_mean": rfine["mean"], "display_multiplier": scale,
                         "reduction_percent": gain, "display_reduction_percent": f"{gain:.2f}"})
    for arm, label in list(zip(ARMS, ARM_NAMES)) + [("pcgrad", "PCGrad-RO"), ("stateguard", "State-prioritized")]:
        single = len(wide3) >= 4
        seeds = SEEDS[:1] if single else SEEDS
        line = ["Seed 20260811" if single else "Three-seed mean +/- sample SD", label]
        display = []
        for metric in ("site", "response"):
            values = [float(learning["per_checkpoint"][f"{arm}/{seed}"]
                            ["overall_equal_block_mean"]["all_25"][metric]) for seed in seeds]
            mean = statistics.mean(values)
            sd = None if single else statistics.stdev(values)
            display.append(f"{1000 * mean:.3f}")
            cell = f"{1000 * mean:.3f}"
            if sd is not None:
                display.append(f"{1000 * sd:.3f}")
                cell += f" +/- {1000 * sd:.3f}"
            line.append(cell)
            records3.append({"group": "single_seed" if single else "three_seed", "arm": arm,
                             "metric": metric, "n_seeds": len(seeds), "seeds": ";".join(map(str, seeds)),
                             "raw_mean": mean, "raw_sample_sd": sd, "display_multiplier": 1000,
                             "seed_values": ";".join(map(repr, values))})
        tokens["table3"].append(display)
        wide3.append(line)
    expected = read_json(inputs / "expected_table_displays.json")
    if tokens != expected:
        raise ValueError("Recomputed table values do not match the manuscript snapshot")
    rows = {"table1": records1, "table2": records2, "table3": records3}
    wide = {"table1": (["Steps", "Metric", *MODEL_NAMES], wide1),
            "table2": (["Steps", "Metric", "Regional-mean", "DynaRO reduction"], wide2),
            "table3": (["Summary", "Training rule", "Site Brier", "Response MSE"], wide3)}
    return rows, wide, tokens

def make_tables(inputs):
    from supplement_reporting import make_supplement_tables
    main = make_main_tables(inputs)
    supplement = make_supplement_tables(inputs)
    return tuple(a | b for a, b in zip(main, supplement))

def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def tex_cell(value):
    return str(value).replace("%", r"\%").replace("+/-", r"\(\pm\)")

def export_tables(inputs, output):
    rows, wide, tokens = make_tables(inputs)
    output.mkdir()
    for table, records in rows.items():
        write_csv(output / f"{table}_values.csv", records)
        header, values = wide[table]
        with (output / f"{table}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(values)
        tex = [r"\begin{tabular}{" + "c" * len(header) + "}", r"\hline",
               " & ".join(map(tex_cell, header)) + r" \\", r"\hline"]
        tex.extend(" & ".join(map(tex_cell, row)) + r" \\" for row in values)
        tex.extend([r"\hline", r"\end{tabular}"])
        (output / f"{table}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    count = sum(len(row) for table in tokens.values() for row in table)
    return {"tables": len(tokens), "main_tables": 3, "supplement_tables": 4,
            "display_rows": sum(len(v) for v in tokens.values()),
            "numeric_values_match_manuscript": count,
            "numeric_values_by_table": {key: sum(map(len, values)) for key, values in tokens.items()},
            "learning_single_seed_rows": 4, "uncertainty": "Sample SD across matched seeds"}
