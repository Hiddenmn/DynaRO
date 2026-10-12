"""Read-only checks for the portable CSV source-data candidate (standard library)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics

SEEDS = (20260811, 20260812, 20260813)
MODELS = {"state_front", "static_front", "rtu", "fno"}
BLOCKS = ("z128_y384_x256", "z256_y0_x128", "z384_y0_x128",
          "z384_y384_x0", "z512_y256_x256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, message, tolerance=1e-11):
    require(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance), message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_private_text(text):
    patterns = (r"(?i)(?<![a-z])[a-z]:[\\/]", r"(?i)/(?:Users|home|root|autodl-tmp)/",
                r"(?i)(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+)",
                r"(?i)seetacloud\.com", r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
    require(not any(re.search(pattern, text) for pattern in patterns),
            "Unexpected private path, account contact or credential-like content")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        require(all(None not in row and None not in row.values() for row in rows),
                f"Malformed CSV: {path.name}")
        return reader.fieldnames, rows


def validate_values(rows, definitions):
    for row in rows:
        for column, definition in definitions.items():
            value = row[column]
            if value == "":
                require(column == "raw_sample_sd" and row.get("n_seeds") == "1",
                        f"Unexpected missing value: {column}")
                continue
            kind = definition["type"]
            if kind == "text":
                continue
            values = value.split(";") if kind == "number_list" else [value]
            for number in values:
                parsed = float(number)
                require(math.isfinite(parsed), f"Non-finite value: {column}")
                if kind == "integer":
                    require(parsed == int(parsed) and parsed >= 0, f"Invalid count/index: {column}")


def semantic_checks(data):
    tables = {key: data[f"tables/table{key}_values.csv"]
              for key in ("1", "2", "3", "S1", "S2", "S3", "S4")}
    for rows in tables.values():
        for row in rows:
            n = int(row["n_seeds"])
            require(n in (1, 3), "Unexpected seed count")
            if "seeds" in row:
                require(tuple(map(int, row["seeds"].split(";"))) == SEEDS[:n], "Unexpected seed order")
            scale = 1 if row["metric"].endswith("_r2") else 100 if row["metric"] in ("inventory", "terminal_class_mae") else 1000
            near(row["display_multiplier"], scale, "Wrong display multiplier")
            if "raw_sample_sd" in row:
                require((row["raw_sample_sd"] == "") == (n == 1), "Single-seed SD must be empty, not zero")
            if "seed_values" in row:
                values = list(map(float, row["seed_values"].split(";")))
                require(len(values) == n, "Wrong number of seed scores")
                near(row["raw_mean"], statistics.mean(values), "Seed mean mismatch")
                if n > 1:
                    near(row["raw_sample_sd"], statistics.stdev(values), "Sample SD mismatch")
    for row in tables["1"]:
        require(row["display_mean"] == f'{float(row["raw_mean"])*float(row["display_multiplier"]):.3f}', "Table 1 rounded mean mismatch")
        require(row["display_sample_sd"] == f'{float(row["raw_sample_sd"])*float(row["display_multiplier"]):.3f}', "Table 1 rounded SD mismatch")
    for key, baseline, proposed in (("2", "coarse_raw_mean", "fine_raw_mean"),
                                    ("S3", "static_raw_mean", "dynaro_raw_mean")):
        for row in tables[key]:
            near(row["reduction_percent"], 100*(1-float(row[proposed])/float(row[baseline])), "Reduction mismatch")

    primary = {(r["model"], r["metric"], r["steps"]): r for r in tables["1"]}
    for key in ("3", "S4"):
        require(Counter(r["n_seeds"] for r in tables[key]) == {"3": 8, "1": 4}, "Learning comparison rows changed")
    curves = defaultdict(list)
    for row in data["figures/figure3_ab_curves.csv"]:
        curves[row["model"], row["metric"]].append(row)
    require(set(curves) == {(m, e) for m in MODELS for e in ("site", "response")}, "Incomplete curve panel")
    for (model, metric), rows in curves.items():
        require([int(r["step"]) for r in rows] == list(range(1, 101)), "Missing/duplicate curve steps")
        mean = statistics.mean(float(r["mean_x1000"]) for r in rows[50:]) / 1000
        near(mean, primary[model, metric, "51-100"]["raw_mean"], "Figure 3 / Table 1 mismatch")
    blocks = defaultdict(dict)
    for row in data["figures/figure3_d_blocks.csv"]:
        require(row["block"] not in blocks[row["model"]], "Duplicate block row")
        blocks[row["model"]][row["block"]] = float(row["site_mean_x1000"])
    require(set(blocks) == MODELS, "Incomplete block models")
    for model, values in blocks.items():
        require(set(values) == set(BLOCKS), "Incomplete spatial blocks")
        near(values[BLOCKS[3]], 0, "Zero-error block was changed")
        near(statistics.mean(values.values())/1000, primary[model, "site", "51-100"]["raw_mean"], "Block / Table 1 mismatch")

    pair_seeds = defaultdict(list)
    for row in data["figures/figureS3_ab_seeds.csv"]:
        key = row["endpoint"], row["comparator"], row["pair_id"]
        near(row["gain"], float(row["comparator_error"])-float(row["stateguard_error"]), "Within-seed paired gain mismatch")
        pair_seeds[key].append(row)
    pair_rows = {}
    groups = defaultdict(list)
    for row in data["figures/figureS3_ab_pairs.csv"]:
        key = row["endpoint"], row["comparator"], row["pair_id"]
        require(key not in pair_rows and row["seed_count"] == "3", "Duplicate pair or changed seed count")
        pair_rows[key] = row
        repeats = pair_seeds[key]
        require(tuple(sorted(int(r["seed"]) for r in repeats)) == SEEDS, "Unmatched paired seeds")
        require(all(r["block_id"] == row["block_id"] for r in repeats), "Pair block mismatch")
        require(row["pair_id"].split("|")[0] == row["block_id"], "Pair identifier block mismatch")
        for field in ("comparator_error", "stateguard_error", "gain"):
            near(row[field], statistics.mean(float(r[field]) for r in repeats), "Within-pair mean mismatch")
            if row["block_id"] == BLOCKS[3]:
                near(row[field], 0, "Zero-error pairs changed")
        groups[key[:2]].append(row)
    require(set(pair_rows) == set(pair_seeds), "Pair and seed panels differ")
    require(set(groups) == {(e, c) for e in ("site", "response") for c in MODELS-{"state_front"}}, "Incomplete paired comparisons")
    for (endpoint, comparator), rows in groups.items():
        require(Counter(r["block_id"] for r in rows) == {b: 120 for b in BLOCKS}, "Not all 600 pairs retained")
        near(statistics.mean(float(r["gain"]) for r in rows),
             float(primary[comparator, endpoint, "51-100"]["raw_mean"])-float(primary["state_front", endpoint, "51-100"]["raw_mean"]),
             "Paired differences / Table 1 mismatch")

    classes = defaultdict(list)
    for row in data["figures/figureS3_c_classes.csv"]:
        classes[row["model"], int(row["seed"])].append(row)
    require(set(classes) == {(m, s) for m in MODELS for s in SEEDS}, "Incomplete class-seed panel")
    for rows in classes.values():
        require({r["initial_class"] for r in rows} == {"Film", "Corner", "Droplet", "Columnar", "Cluster"} and len(rows) == 5, "Class panel changed")
    for model in MODELS:
        means = [statistics.mean(float(r["inventory_mae_pp"])/100 for r in classes[model, seed]) for seed in SEEDS]
        # The original evaluator saves errors[present].mean() in float32,
        # but exports each class separately before later Python aggregation.
        # These two saved reduction orders are not bit-identical. Preserve
        # both exports and compare within 1e-9 raw fraction (1e-7 pp).
        near(statistics.mean(means), primary[model, "inventory", "100"]["raw_mean"], "Class / Table 1 inventory mean mismatch", 1e-9)
        near(statistics.stdev(means), primary[model, "inventory", "100"]["raw_sample_sd"], "Class / Table 1 inventory SD mismatch", 1e-9)

    for row in data["figures/figure4_learning.csv"]:
        key = "3" if row["split"] == "Locked test" else "S4"
        require(row["split"] in ("Development", "Locked test"), "Unknown learning split")
        require(int(row["pairs"]) == (600 if key == "3" else 400), "Wrong learning pair count")
        seed_index = SEEDS.index(int(row["seed"]))
        for metric in ("site", "response"):
            for arm, field in (("response50", "fixed50_"), ("stateguard", "stateguard_")):
                match = [r for r in tables[key] if r["arm"] == arm and r["metric"] == metric and r["group"] == "three_seed"]
                require(len(match) == 1, "Ambiguous learning reference")
                near(row[field+metric], match[0]["seed_values"].split(";")[seed_index], "Figure 4 / learning table mismatch")
            near(row[metric+"_reduction_pct"], 100*(float(row["fixed50_"+metric])-float(row["stateguard_"+metric]))/float(row["fixed50_"+metric]), "Learning reduction mismatch")

    time = data["figures/figure5_e_time.csv"]
    require([int(r["step"]) for r in time] == [0, 25, 50, 75, 100], "Interior saved steps changed")
    for rows, denominator in ((time, "initial_oil"), (data["figures/figure5_auxiliary_depth.csv"], "initial_slab_oil")):
        for row in rows:
            require(float(row[denominator]) > 0, "Missing oil denominator")
            near(row["simulator_percent"], 100*float(row["simulator_mobilized_voxels"])/float(row[denominator]), "Simulator mobilization units mismatch")
            near(row["prediction_percent"], 100*float(row["predicted_expected_mobilized_voxels"])/float(row[denominator]), "Predicted mobilization units mismatch")
    contact = data["figures/figure5_f_contact.csv"]
    require({r["cohort"] for r in contact} == {"water_contact", "no_water_contact", "not_inlet_connected"}, "Cohort partition incomplete")
    for row in contact:
        require((row["baseline_step"], row["followup_step"]) == ("50", "75"), "Cohort interval changed")
        n = float(row["simulator_baseline_oil_sites"])
        near(sum(int(row[f"initial_class_{c}_sites"]) for c in range(3, 8)), n, "Initial class counts do not partition cohort")
        near(row["prediction_expected_converted_sites"], float(row["prediction_expected_oil_at_baseline"])-float(row["prediction_expected_oil_at_followup"]), "Prediction follow-up mismatch")
        for out, numerator in (("simulator_mobilized_percent", "simulator_converted_sites"),
                               ("prediction_mobilized_percent", "prediction_expected_converted_sites"),
                               ("prediction_baseline_remaining_percent", "prediction_expected_oil_at_baseline")):
            near(row[out], 100*float(row[numerator])/n, "Fixed cohort denominator mismatch")
    near(sum(int(r["simulator_baseline_oil_sites"]) for r in contact), int(time[2]["initial_oil"])-int(time[2]["simulator_mobilized_voxels"]), "Baseline partition mismatch")
    near(sum(int(r["simulator_converted_sites"]) for r in contact), int(time[3]["simulator_mobilized_voxels"])-int(time[2]["simulator_mobilized_voxels"]), "Simulator follow-up partition mismatch")
    train = data["figures/figureS1_e_conversion.csv"]
    require([int(r["step"]) for r in train] == list(range(51)), "TRAIN crop steps changed")
    for row in train:
        near(int(row["remaining_voxels"])+int(row["converted_voxels"]), 18396, "TRAIN oil count mismatch")
        near(row["remaining_fraction"], int(row["remaining_voxels"])/18396, "TRAIN remaining fraction mismatch")
    for name, expected in (("figureS2_g_time.csv", range(101)), ("figureS2_i_brier.csv", range(1, 101))):
        require([int(r["step"]) for r in data["figures/"+name]] == list(expected), "Parent step sequence changed")
    for row in data["figures/figureS2_i_brier.csv"]:
        near(row["site_brier_display_x1000"], 1000*float(row["site_brier_raw"]), "Brier display scale mismatch")
    for index, row in enumerate(data["figures/figureS2_h_depth.csv"]):
        require((int(row["x_start_inclusive"]), int(row["x_stop_exclusive"])) == (16*index, 16*(index+1)), "Parent slab bounds changed")
        near(row["x_center"], 16*index+7.5, "Parent slab center mismatch")


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1 and manifest["status"] == "local_candidate_not_published", "Unexpected manifest scope")
    require(len(manifest["files"]) == 28, "Expected 28 source CSVs")
    data, paths = {}, set()
    for record in manifest["files"]:
        relative = PurePosixPath(record["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in record["path"] and ":" not in record["path"], "Unsafe manifest path")
        require(record["path"] not in paths, "Duplicate manifest path")
        paths.add(record["path"])
        path = root.joinpath(*relative.parts)
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Escaping data path")
        require(path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"], f"Changed source CSV: {record['path']}")
        check_private_text(path.read_text(encoding="utf-8-sig"))
        header, rows = read_csv(path)
        require(header == record["columns"] and len(rows) == record["rows"], "CSV schema/row count mismatch")
        require(set(record["definitions"]) == set(header), "Incomplete column dictionary")
        validate_values(rows, record["definitions"])
        data[record["path"]] = rows
    require({str(p.relative_to(root)).replace("\\", "/") for p in root.glob("*/*.csv")} == paths, "Unlisted or missing data CSV")
    for name, record in manifest["support_files"].items():
        require(Path(name).name == name and name != "manifest.json", "Unsafe support filename")
        require(sha256(root/name) == record["sha256"], f"Changed support file: {name}")
    require(set(manifest["support_files"]) == {"README.txt", "data_dictionary.csv", "verify_source_data.py"}, "Unexpected support files")
    require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            == paths | set(manifest["support_files"]) | {"manifest.json"}, "Unexpected file in source-data candidate")
    for name in ("manifest.json", "README.txt", "data_dictionary.csv"):
        check_private_text((root/name).read_text(encoding="utf-8"))
    semantic_checks(data)
    return {"status": "VERIFIED", "csv_files": len(data), "data_rows": sum(map(len, data.values())),
            "tables": 7, "figures_with_quantitative_csvs": 6,
            "scope": "CSV integrity, schema and stated numerical consistency checks; no model execution, publication or full-volume reconstruction"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
