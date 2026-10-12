"""Matched, forward-only timing of byte-preserved DynaRO study checkpoints.

Default is a metadata-only plan. Use --run with a fresh --output directory for
local GPU inference; neither model code nor the prepared release is modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
MODELS = ("dynaro", "static_front", "rtu", "fno")
SEED = 20260811


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_cases(reproduction):
    rows = json.loads((reproduction / "inputs/label_manifest.json").read_text())
    cases = sorted((r for r in rows if r["split"] == "dev" and
                    r["stage"] == "NPI-012"), key=lambda r: r["block_id"])
    if len(cases) != 5 or len({r["block_id"] for r in cases}) != 5:
        raise ValueError("Expected exactly one NPI-012 volume per DEV block")
    if any(r["shape_zyx"] != [128, 128, 128] for r in cases):
        raise ValueError("Only the original 128-cubed grid is supported")
    return cases


def pore_masks(labels):
    import numpy as np
    from scipy.ndimage import binary_propagation, generate_binary_structure
    pore = labels >= 2
    inlet = np.zeros_like(pore)
    inlet[:, :, 0] = pore[:, :, 0]
    connected = binary_propagation(inlet, mask=pore,
                                   structure=generate_binary_structure(3, 1))
    return inlet, connected


def controls(horizon, device="cpu"):
    import torch
    actions = torch.zeros((horizon, 4), dtype=torch.float32, device=device)
    actions[:min(20, horizon), 0] = 1.0 / 1.45
    actions[20:, 0] = 1.0
    steps = torch.arange(1, horizon + 1, dtype=torch.float32, device=device)
    return actions, (steps / (steps + 25)).unsqueeze(1)


def checked_array(reproduction, record):
    import numpy as np
    path = (reproduction / "inputs" / record["file"]).resolve(strict=True)
    if not path.is_relative_to((reproduction / "inputs/labels").resolve()):
        raise ValueError("Label path escapes the approved input directory")
    labels = np.load(path, allow_pickle=False)
    if labels.dtype != np.uint8 or labels.shape != (128, 128, 128):
        raise ValueError("Unexpected label shape/dtype")
    if hashlib.sha256(labels.tobytes(order="C")).hexdigest() != record["array_sha256"]:
        raise ValueError("Input array differs from the frozen manifest")
    if np.bincount(labels.ravel(), minlength=8).tolist() != record["class_voxel_counts"]:
        raise ValueError("Input class counts differ")
    return labels


def validate_output(out, labels, horizon):
    import torch
    sites, inventories = out["remaining_sites"], out["group_remaining"]
    expected_index = torch.where(labels.reshape(-1) >= 3)[0]
    assert torch.equal(out["site_index"], expected_index)
    assert out["step_offset"] == 0
    assert sites.shape == (horizon, expected_index.numel())
    assert inventories.shape == (horizon, 8 * 8 * 8 * 5)
    assert bool(torch.isfinite(sites).all() and torch.isfinite(inventories).all())
    assert float(sites.min()) >= -1e-6 and float(sites.max()) <= 1 + 1e-6
    assert horizon == 1 or float((sites[1:] - sites[:-1]).max()) <= 1e-6
    assert torch.allclose(sites.sum(1), inventories.sum(1), rtol=2e-5, atol=.1)
    return {"initial_oil_sites": expected_index.numel(),
            "site_output_shape": list(sites.shape),
            "inventory_output_shape": list(inventories.shape),
            "finite_bounded_monotone": True, "inventory_identity": True}


def summary(records):
    groups = {}
    for row in records:
        groups.setdefault((row["model"], row["horizon"]), []).append(row)
    result = []
    for (model, horizon), rows in sorted(groups.items()):
        latencies = [r["latency_median_s"] for r in rows]
        result.append({
            "model": model, "horizon": horizon, "blocks": len(rows),
            "mean_case_median_s": statistics.mean(latencies),
            "min_case_median_s": min(latencies), "max_case_median_s": max(latencies),
            "max_peak_allocated_mib": max(r["peak_allocated_mib"] for r in rows),
            "max_peak_reserved_mib": max(r["peak_reserved_mib"] for r in rows),
        })
    return result


def run(args, plan, cases):
    import numpy as np
    import scipy
    import torch

    output = args.output.resolve()
    if output.exists():
        raise ValueError("Output must be a fresh directory; no timing records are overwritten")
    if output.is_relative_to(args.reproduction.resolve()):
        raise ValueError("Never write into the prepared reproduction release")
    if not torch.cuda.is_available():
        raise RuntimeError("The timing protocol requires a local CUDA GPU")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261012)
    total = torch.cuda.get_device_properties(device).total_memory
    cap = 9 * 1024 ** 3
    torch.cuda.set_per_process_memory_fraction(min(cap / total, .85), device)

    spec = importlib.util.spec_from_file_location(
        "frozen_timing_loader", args.reproduction / "weights/load_weights.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    environment = {
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "scipy": scipy.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device), "gpu_total_bytes": total,
        "allocation_cap_bytes": min(cap, int(total * .85)),
        "tf32": False, "cudnn_benchmark": False, "cpu_threads": 4,
        "inter_op_threads": 1, "batch_size": 1, "dtype": "float32",
        "timed_region": "native compact forward; CUDA-synchronized wall clock",
        "exclude": ["input I/O", "connectivity preprocessing", "host-to-device copy",
                    "checkpoint load", "disk export", "target scoring"],
    }
    try:
        environment["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
             "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        environment["nvidia_smi"] = "unavailable"
    output.mkdir(parents=True)
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "plan": plan, "environment": environment, "records": [],
              "script_sha256": sha256(__file__), "optimizer_updates": 0,
              "test_pairs_scored": 0}
    report_path = output / "timing.json"

    def save():
        report["summary"] = summary(report["records"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    started = time.perf_counter()
    save()
    try:
        with torch.inference_mode():
            for case_no, record in enumerate(cases):
                labels_cpu = checked_array(args.reproduction, record)
                inlet_cpu, connected_cpu = pore_masks(labels_cpu)
                for h_no, horizon in enumerate(plan["horizons"]):
                    rotation = (case_no + h_no) % len(MODELS)
                    order = MODELS[rotation:] + MODELS[:rotation]
                    for model_name in order:
                        if time.perf_counter() - started > args.max_seconds:
                            report["status"] = "incomplete_time_budget"
                            save()
                            print(report["status"], flush=True)
                            return
                        model, metadata = loader.load_frozen(
                            args.reproduction / "code/model_core",
                            f"mixed_horizon/{model_name}/{SEED}",
                            args.reproduction / "weights")
                        model.to(device)
                        labels = torch.from_numpy(labels_cpu).to(device)
                        inlet = torch.from_numpy(inlet_cpu).to(device)
                        connected = torch.from_numpy(connected_cpu).to(device)
                        actions, times = controls(horizon, device)

                        def forward():
                            return model(labels, inlet, connected, actions, times,
                                         compact_output=True, return_fields=False,
                                         checkpoint_local=False, start_grad_step=0)

                        check = None
                        for warm in range(plan["warmups"]):
                            out = forward()
                            if warm == 0:
                                check = validate_output(out, labels, horizon)
                            del out
                        torch.cuda.synchronize(device)
                        repetitions = []
                        for rep in range(plan["repeats"]):
                            torch.cuda.reset_peak_memory_stats(device)
                            torch.cuda.synchronize(device)
                            tick = time.perf_counter()
                            out = forward()
                            torch.cuda.synchronize(device)
                            elapsed = time.perf_counter() - tick
                            repetitions.append({
                                "rep": rep + 1, "wall_s": elapsed,
                                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                            })
                            del out
                        row = {"model": model_name, "seed": SEED,
                               "checkpoint_sha256": metadata["sha256"],
                               "parameters": metadata["parameters"],
                               "case": record["subcore_id"], "block_id": record["block_id"],
                               "input_array_sha256": record["array_sha256"],
                               "horizon": horizon, "output_check": check,
                               "repetitions": repetitions,
                               "latency_median_s": statistics.median(r["wall_s"] for r in repetitions),
                               "peak_allocated_mib": max(r["peak_allocated_mib"] for r in repetitions),
                               "peak_reserved_mib": max(r["peak_reserved_mib"] for r in repetitions)}
                        report["records"].append(row)
                        save()
                        print(json.dumps({k: row[k] for k in (
                            "model", "block_id", "horizon", "latency_median_s",
                            "peak_allocated_mib")}), flush=True)
                        del model, labels, inlet, connected, actions, times, forward
                        gc.collect()
                        torch.cuda.empty_cache()
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["wall_duration_s"] = time.perf_counter() - started
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction", type=Path, default=Path(__file__).resolve().parents[1] /
                        "DynaRO_release_prepared_20261009/reproduction")
    parser.add_argument("--phase", choices=("pilot", "benchmark"), default="pilot")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args()
    cases = choose_cases(args.reproduction)
    if args.phase == "pilot":
        cases = cases[:1]
    plan = {"phase": args.phase, "models": MODELS, "seed": SEED,
            "cases": cases, "horizons": [25] if args.phase == "pilot" else [25, 50, 100],
            "warmups": 1 if args.phase == "pilot" else 2,
            "repeats": 1 if args.phase == "pilot" else 5,
            "schedule": "S2 x min(20,T), then S3", "time_feature": "t/(t+25)",
            "grid": [128, 128, 128], "selection": "all five NPI-012 DEV blocks",
            "output_contract": "all-step initial-oil-site probabilities and group inventories"}
    print(json.dumps(plan, indent=2), flush=True)
    if args.run:
        if args.output is None:
            parser.error("--run requires a fresh --output directory")
        run(args, plan, cases)


if __name__ == "__main__":
    main()
