"""Rebuild all four Supplement tables from saved DEV and TEST scores."""
from __future__ import annotations
import csv
import math
import statistics

from reporting import ARMS, ARM_NAMES, MODELS, MODEL_NAMES, SEEDS, read_json, reduction

HORIZONS = (25, 50, 75, 100)

def interval(horizon):
    return f"{1 if horizon <= 50 else 51}-{horizon}"

def moments(values, saved=None):
    values = [float(v) for v in values]
    if len(values) != len(SEEDS) or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected three finite matched-seed scores")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    if saved is not None and (
        not math.isclose(mean, float(saved["mean"]), rel_tol=0, abs_tol=1e-13)
        or not math.isclose(sd, float(saved["seed_sd"]), rel_tol=0, abs_tol=1e-13)
    ):
        raise ValueError("Saved aggregate disagrees with its per-seed scores")
    return mean, sd

def display_pair(mean, sd, scale, places):
    if not math.isfinite(mean) or not math.isfinite(sd) or sd < 0:
        raise ValueError("Invalid mean or sample SD")
    return [f"{scale * value:.{places}f}" for value in (mean, sd)]

def load_dev_metrics(inputs):
    with (inputs / "dev_secondary_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    lookup = {}
    for row in source:
        model, horizon = row["model"], int(row["horizon"])
        key = model, horizon
        if key in lookup:
            raise ValueError("Duplicate DEV model/horizon row")
        if (int(row["n_seeds"]) != 3
                or tuple(map(int, row["seeds"].split(";"))) != SEEDS):
            raise ValueError("Unexpected DEV seed panel")
        if row["steps"] != interval(horizon):
            raise ValueError("DEV scored interval does not match its endpoint")
        lookup[key] = row
    if set(lookup) != {(model, h) for model in MODELS for h in HORIZONS}:
        raise ValueError("Incomplete DEV model/horizon panel")
    return lookup

def dev_comparator(inputs, arm, seed):
    prefix, mode = ("du_dev", "du_gradcos_ro") if arm == "du_gradcos" else ("pcgrad_dev", "pcgrad_ro")
    record = read_json(inputs / f"{prefix}_{seed}.json")
    protocol = record["protocol"]
    if (record["pair_count"] != 400 or record["spatial_block_count"] != 5
            or record["locked_test_accessed"] is not False
            or protocol["horizon_steps"] != 25
            or (arm == "du_gradcos" and protocol.get("mode") != mode)
            or (arm == "pcgrad" and record["status"] != "COMPLETE_PCGRAD_RO_FORMAL_DEV_ONLY")
            or f"{mode}_train800_e1_seed{seed}_" not in protocol["checkpoint"]):
        raise ValueError("Unexpected saved 25-step DEV comparator protocol")
    return record["overall_equal_block_mean"]["all_25"]

def make_supplement_tables(inputs):
    dev = load_dev_metrics(inputs)
    long = read_json(inputs / "long_horizon.json")
    coarse = read_json(inputs / "exposure_ablation.json")
    learning = read_json(inputs / "learning_dev.json")
    if (tuple(map(int, learning["per_seed"])) != SEEDS
            or learning["locked_test_accessed"] is not False):
        raise ValueError("Unexpected saved 25-step DEV learning panel")
    rows = {key: [] for key in ("tableS1", "tableS2", "tableS3", "tableS4")}
    tokens = {key: [] for key in rows}
    wide = {
        "tableS1": (["Horizon", "Scored interval", "Model", "Site R2", "Regional R2",
                     "Terminal class-inventory MAE (pp)"], []),
        "tableS2": (["Endpoint", "Scored interval", "Model", "Regional-state MSE (x1000)",
                     "Inventory MAE (pp)"], []),
        "tableS3": (["Interval", "Metric", "Static front", "DynaRO", "Reduction"], []),
        "tableS4": (["Summary", "Training rule", "Site Brier (x1000)", "Response MSE (x1000)"], []),
    }
    for horizon in HORIZONS:
        for model, label in zip(MODELS, MODEL_NAMES):
            record = dev[model, horizon]
            line, values = [horizon, interval(horizon), label], []
            for metric, scale, places in (("site_r2", 1, 4), ("region_r2", 1, 4),
                                           ("terminal_class_mae", 100, 3)):
                mean, sd = (float(record[metric + suffix]) for suffix in ("_mean", "_sample_sd"))
                pair = display_pair(mean, sd, scale, places)
                values.extend(pair)
                line.append(" +/- ".join(pair))
                rows["tableS1"].append({
                    "split": "DEV", "horizon": horizon, "steps": interval(horizon),
                    "model": model, "metric": metric, "n_seeds": 3,
                    "seeds": ";".join(map(str, SEEDS)), "raw_mean": mean, "raw_sample_sd": sd,
                    "display_multiplier": scale, "display_decimals": places})
            wide["tableS1"][1].append(line)
            tokens["tableS1"].append(values)

        test_models = ("state_front", "coarse_front", "static_front", "rtu", "fno")
        labels = dict(zip(MODELS, MODEL_NAMES)) | {"coarse_front": "Regional-mean exposure"}
        for model in test_models:
            line, values = [horizon, interval(horizon), labels[model]], []
            for metric, scale in (("region", 1000), ("inventory", 100)):
                if model == "coarse_front":
                    model_summary = coarse["models"][model]
                    aggregate = (model_summary["terminal_inventory"][str(horizon)] if metric == "inventory"
                                 else model_summary["intervals"][str(horizon)]["region"])
                    scores = aggregate["seeds"]
                    mean, sd = moments(scores, aggregate)
                else:
                    records = [long["seed_records"][model][str(seed)]["horizons"][str(horizon)]
                               for seed in SEEDS]
                    scores = ([r["last_class_inventory_fraction_mae"] for r in records]
                              if metric == "inventory"
                              else [r["all" if horizon <= 50 else "unseen"]["region"] for r in records])
                    aggregate = (long["models"][model]["inventory_mae_100"] if horizon == 100
                                 else None) if metric == "inventory" else (
                                     long["models"][model]["intervals"][str(horizon)]["region"])
                    mean, sd = moments(scores, aggregate)
                    if model in coarse["models"]:
                        saved = (coarse["models"][model]["terminal_inventory"][str(horizon)]
                                 if metric == "inventory"
                                 else coarse["models"][model]["intervals"][str(horizon)]["region"])
                        moments(scores, saved)
                pair = display_pair(mean, sd, scale, 3)
                line.append(" +/- ".join(pair))
                values.extend(pair)
                rows["tableS2"].append({
                    "split": "TEST", "horizon": horizon,
                    "steps": str(horizon) if metric == "inventory" else interval(horizon),
                    "model": model, "metric": metric, "n_seeds": 3,
                    "seeds": ";".join(map(str, SEEDS)), "seed_values": ";".join(map(repr, scores)),
                    "raw_mean": mean, "raw_sample_sd": sd, "display_multiplier": scale})
            wide["tableS2"][1].append(line)
            tokens["tableS2"].append(values)

    for metric, label in (("site", "Site-state Brier"), ("response", "Regional response MSE")):
        for horizon in (75, 100):
            line, values = [interval(horizon), label], []
            for model in ("static_front", "state_front"):
                record = dev[model, horizon]
                mean, sd = (float(record[metric + suffix]) for suffix in ("_mean", "_sample_sd"))
                pair = display_pair(mean, sd, 1000, 3)
                line.append(" +/- ".join(pair))
                values.extend(pair)
            static = float(dev["static_front", horizon][metric + "_mean"])
            proposed = float(dev["state_front", horizon][metric + "_mean"])
            gain = reduction(static, proposed)
            line.append(f"{gain:.2f}%")
            values.append(f"{gain:.2f}")
            rows["tableS3"].append({
                "split": "DEV", "horizon": horizon, "steps": interval(horizon),
                "metric": metric, "n_seeds": 3, "static_raw_mean": static,
                "static_raw_sample_sd": float(dev["static_front", horizon][metric + "_sample_sd"]),
                "dynaro_raw_mean": proposed,
                "dynaro_raw_sample_sd": float(dev["state_front", horizon][metric + "_sample_sd"]),
                "display_multiplier": 1000, "reduction_percent": gain})
            wide["tableS3"][1].append(line)
            tokens["tableS3"].append(values)

    arms = list(zip(ARMS, ARM_NAMES)) + [("pcgrad", "PCGrad-RO"), ("stateguard", "State-prioritized")]
    for index, (arm, label) in enumerate(arms):
        single = index >= 4
        seeds = SEEDS[:1] if single else SEEDS
        line = ["Seed 20260811" if single else "Three-seed mean +/- sample SD", label]
        values = []
        records = [(dev_comparator(inputs, arm, seed) if arm in ("du_gradcos", "pcgrad")
                    else learning["per_seed"][str(seed)]["equal_block_mean"][arm])
                   for seed in seeds]
        for metric in ("site", "response"):
            scores = [float(record[metric]) for record in records]
            if not all(math.isfinite(v) for v in scores):
                raise ValueError("Non-finite DEV learning score")
            mean = statistics.mean(scores)
            sd = None if single else statistics.stdev(scores)
            pair = [f"{mean * 1000:.3f}"] if single else display_pair(mean, sd, 1000, 3)
            values.extend(pair)
            line.append(" +/- ".join(pair))
            rows["tableS4"].append({
                "split": "DEV", "group": "single_seed" if single else "three_seed",
                "arm": arm, "metric": metric, "n_seeds": len(seeds),
                "seeds": ";".join(map(str, seeds)), "seed_values": ";".join(map(repr, scores)),
                "raw_mean": mean, "raw_sample_sd": sd, "display_multiplier": 1000})
        wide["tableS4"][1].append(line)
        tokens["tableS4"].append(values)
    expected = read_json(inputs / "expected_supplement_displays.json")
    if tokens != expected:
        mismatches = [key for key in tokens if tokens[key] != expected.get(key)]
        raise ValueError(f"Recomputed Supplement values do not match the manuscript: {mismatches}")
    return rows, wide, tokens
