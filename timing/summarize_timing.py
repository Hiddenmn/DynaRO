"""Validate a complete timing panel and export its small manuscript table."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from benchmark_inference import MODELS, summary

DISPLAY = {"dynaro": "DynaRO", "static_front": "Static front", "rtu": "RTU", "fno": "FNO"}


def validate_report(report):
    assert report["status"] == "complete", "Do not publish an incomplete panel"
    assert report["plan"]["phase"] == "benchmark"
    assert report["plan"]["warmups"] == 2 and report["plan"]["repeats"] == 5
    assert report["optimizer_updates"] == report["test_pairs_scored"] == 0
    expected = {(m, h, c["block_id"]) for m in MODELS
                for h in [25, 50, 100] for c in report["plan"]["cases"]}
    actual = {(r["model"], r["horizon"], r["block_id"]) for r in report["records"]}
    assert expected == actual and len(report["records"]) == 60
    for row in report["records"]:
        assert len(row["repetitions"]) == 5
        assert row["output_check"]["finite_bounded_monotone"]
        assert row["output_check"]["inventory_identity"]
        assert row["latency_median_s"] == statistics.median(r["wall_s"] for r in row["repetitions"])
    calculated = summary(report["records"])
    assert calculated == report["summary"]
    return {(r["model"], r["horizon"]): r for r in calculated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    grouped = validate_report(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows["summary"][0]))
        writer.writeheader()
        writer.writerows(rows["summary"])
    with (args.output / "timed_repetitions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "case", "block", "horizon", "rep", "wall_s",
                         "peak_allocated_mib", "peak_reserved_mib"])
        for row in rows["records"]:
            for rep in row["repetitions"]:
                writer.writerow([row["model"], row["case"], row["block_id"], row["horizon"],
                                 rep["rep"], rep["wall_s"], rep["peak_allocated_mib"],
                                 rep["peak_reserved_mib"]])
    table = []
    for model in MODELS:
        durations = [grouped[model, h]["mean_case_median_s"] for h in [25, 50, 100]]
        memory = grouped[model, 100]["max_peak_allocated_mib"]
        table.append(DISPLAY[model] + " & " + " & ".join(f"{v:.3f}" for v in durations)
                     + f" & {memory:.0f}" + r" \\")
    (args.output / "table_rows.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    dynaro = grouped["dynaro", 100]["mean_case_median_s"]
    comparison = {
        "timing_source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "native_100_step_latency_ratio_comparator_over_dynaro": {
            model: grouped[model, 100]["mean_case_median_s"] / dynaro
            for model in MODELS if model != "dynaro"},
        "dynaro_over_static_latency_ratio": dynaro / grouped["static_front", 100]["mean_case_median_s"],
        "aggregation": "mean over five block-specific medians of five timed calls",
        "training_or_test_rerun": False,
    }
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("\n".join(table))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
