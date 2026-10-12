"""Describe a fixed input panel; generate only with --execute and a new output."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CHECK = runpy.run_path(str(HERE / "verify_inputs.py"))
require, file_at, digest = (CHECK[k] for k in ("require", "file_at", "digest"))
SETS = ("train25", "dev25", "test25", "train50", "test100")


def new_output(root, output):
    require(output is not None, "Generation requires --output pointing to a new directory")
    raw = Path(output).absolute()
    require(not raw.exists() and not raw.is_symlink() and not raw.is_junction(),
            "Refusing to reuse an existing or redirected output")
    target, root = raw.resolve(), Path(root).resolve(strict=True)
    require(target.parent.is_dir() and not target.is_relative_to(root)
            and not root.is_relative_to(target), "Output must be separate from the input component")
    return target


def read_plan(root, group, limit=None):
    require(group in SETS, "Unknown generation panel")
    root = Path(root).resolve(strict=True)
    CHECK["verify"](root, arrays=False)
    rows = CHECK["read_rows"](file_at(root, f"trajectory_plans/{group}.jsonl"))
    require(limit is None or type(limit) is int and 1 <= limit <= len(rows), "Invalid trajectory limit")
    selected = rows if limit is None else rows[:limit]
    labels = json.loads(file_at(root, "label_manifest.json").read_text(encoding="utf-8"))
    lookup = {(r["block_id"], r["stage"]): r for r in labels}
    report = {"status": "read_only_generation_plan", "set": group, "panel_trajectories": len(rows),
              "selected_trajectories": len(selected), "complete_panel": len(selected) == len(rows),
              "steps_per_trajectory": rows[0]["steps"],
              "generation_enabled": False, "simulator_steps_executed": 0,
              "training_or_model_evaluation": "not_run"}
    return selected, lookup, report


def compact_arrays(initial, row, source, actions, step_seed):
    """The recorded 50/100-step loop, using the unchanged numerical kernel."""
    import numpy as np
    current = initial.copy()
    connected = source.connected_to_inlet((initial != 0) & (initial != 1), inlet_axis="x").astype(np.uint8)
    pressure = source.axis_gradient(initial.shape, "x")
    labels, displaced_rows = [current], []
    for step, action in enumerate(row["action_ids"], 1):
        case = f"{row['block_id']}|{row['stage']}|{step}"
        rng = np.random.default_rng(step_seed(row["stream_id"], case))
        before = current
        current, info = source.displacement_step(
            label=current, strategy=actions[action].source_strategy(), rng=rng,
            pressure_weight=pressure, inlet_axis="x", base_probability=0.06)
        changes = (before >= 3) & (current == 2)
        require(np.all((before == current) | changes), "Unexpected simulator state transition")
        displaced = [int(info[f"displaced_class_{cls}_{source.CLASS_NAMES[cls]}"])
                     for cls in (3, 4, 5, 6, 7)]
        require(displaced == [int(((before == cls) & changes).sum()) for cls in (3, 4, 5, 6, 7)],
                "Per-class displacement differs from the state change")
        labels.append(current)
        displaced_rows.append(displaced)
    displaced = np.asarray(displaced_rows, dtype=np.int32)
    require(int((initial >= 3).sum()) - int((current >= 3).sum()) == int(displaced.sum()),
            "Oil inventory identity failed")
    return {"labels": np.stack(labels).astype(np.uint8), "connected_pore": connected,
            "action_index": np.asarray([int(a[1:])-1 for a in row["action_ids"]], dtype=np.int8),
            "displaced_by_class": displaced}


def write_compact(path, arrays):
    import numpy as np
    # A fresh batch directory is mandatory. Exclusive creation also rejects races.
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def generate25(adapter, path, row, label, output, generator):
    meta = {"sample": row["stage"], "subcore_id": row["subcore_id"],
            **dict(zip(("z0", "y0", "x0"), map(str, label["origin_zyx"])))}
    kwargs = {"adapter": adapter, "label_path": path, "row": meta,
              "sequence_id": row["sequence_id"], "action_ids": row["action_ids"],
              "stream_id": row["stream_id"], "output_dir": output, "canonical_horizon": 25}
    if row["split"] == "locked_test":
        result = generator.generate_locked_test_trajectory(**kwargs)
    else:
        result = generator.generate_trajectory(split=row["split"], **kwargs)
    require(result["trajectory_id"] == row["trajectory_id"] and result["npz_filename"] == row["file"]
            and result["noise_seed_sha256"] == row["noise_seed_sha256"], "Generated identity/noise differs")
    return result


def execute(root, group, output, limit=None):
    """Explicit local trajectory generation. This function never calls a model."""
    root = Path(root).resolve(strict=True)
    output = new_output(root, output)
    rows, lookup, plan = read_plan(root, group, limit)
    CHECK["verify"](root, arrays=True)
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import scipy
    from patent22 import generator
    from patent22.actions import ACTION_LIBRARY
    from patent22.source_adapter import ReadOnlyStep2Adapter, stable_step_seed
    adapter = ReadOnlyStep2Adapter(file_at(root, "src/simulator_core.py"), inlet_axis="x", base_probability=0.06)
    output.mkdir(exist_ok=False)
    data = output / "data"
    data.mkdir()
    run = {**plan, "status": "generation_started", "generation_enabled": True,
           "input_manifest_sha256": digest(root / "manifest.json"),
           "simulator_core_sha256": adapter.source_sha256,
           "environment": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "system": platform.system()}}
    with (output / "generation_plan.json").open("x", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
        f.write("\n")
    with (output / "generated_trajectories.jsonl").open("x", encoding="utf-8") as ledger:
        for index, row in enumerate(rows, 1):
            label = lookup[row["block_id"], row["stage"]]
            label_path = file_at(root, row["label_file"])
            require(digest(label_path) == row["label_source_sha256"], "Initial label changed")
            if row["steps"] == 25:
                generate25(adapter, label_path, row, label, data, generator)
            else:
                initial = np.load(label_path, allow_pickle=False)
                arrays = compact_arrays(initial, row, adapter.source, ACTION_LIBRARY, stable_step_seed)
                write_compact(data / row["file"], arrays)
                del arrays, initial
            path = data / row["file"]
            record = {**row, "file": "data/" + path.name, "bytes": path.stat().st_size,
                      "generated_npz_sha256": digest(path)}
            with np.load(path, allow_pickle=False) as saved:
                record["generated_arrays"] = {}
                for name in saved.files:
                    array = saved[name]
                    record["generated_arrays"][name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                                                       "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            # Original payload hashes remain references, never assertions of a new match.
            CHECK["private_text"](json.dumps(record))
            ledger.write(json.dumps(record, allow_nan=False) + "\n")
            ledger.flush()
            print(json.dumps({"generated": index, "selected": len(rows), "file": path.name}), flush=True)
    final = {**run, "status": "selected_trajectories_generated", "generated_trajectories": len(rows),
             "simulator_steps_executed": sum(r["steps"] for r in rows),
             "historical_numerical_identity": "not_established_by_generation_alone"}
    with (output / "generation_summary.json").open("x", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
        f.write("\n")
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--set", choices=SETS, required=True)
    parser.add_argument("--limit", type=int, help="Generate a fixed prefix only; marked incomplete")
    parser.add_argument("--execute", action="store_true", help="Explicitly run the simulator, not a model")
    parser.add_argument("--output", type=Path, help="New separate batch directory; never overwrite/resume")
    args = parser.parse_args(argv)
    if not args.execute:
        require(args.output is None, "--output requires --execute; read-only plans write nothing")
        report = read_plan(args.root, args.set, args.limit)[2]
    else:
        report = execute(args.root, args.set, args.output, args.limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
