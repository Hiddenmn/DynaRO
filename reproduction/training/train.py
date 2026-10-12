"""Recorded TRAIN configurations; default invocation only inspects metadata."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
API = runpy.run_path(str(HERE / "verify_training.py"))
require, digest, safe_file = (API[k] for k in ("require", "digest", "safe_file"))


def prepare(core, inputs, evaluation, key, train25=None, train50=None, kind="original"):
    require(kind in ("original", "regenerated"), "Unknown trajectory source")
    API["verify"](HERE)
    proto = json.loads(safe_file(HERE, "protocol.json").read_text(encoding="utf-8"))
    require(key in proto["configurations"], "Unknown training configuration")
    cfg = proto["configurations"][key]
    core, inputs, evaluation = map(lambda p: Path(p).resolve(strict=True), (core, inputs, evaluation))
    for root, field in ((inputs, "inputs_manifest_sha256"), (evaluation, "evaluation_manifest_sha256")):
        require(digest(safe_file(root, "manifest.json")) == proto["components"][field], "Different prepared component")
    for root, verifier in ((inputs, "verify_inputs.py"), (evaluation, "verify_evaluation.py")):
        manifest = json.loads(safe_file(root, "manifest.json").read_text(encoding="utf-8"))
        require(digest(safe_file(root, verifier)) == manifest["files"][verifier]["sha256"], "Different component verifier")
        api = runpy.run_path(str(root / verifier))
        api["verify"](root, **({"arrays": False} if root == inputs else {}))
    for name, sha in proto["components"]["model_core_files"].items():
        require(digest(safe_file(core, name)) == sha, "Changed model source")
    ev = runpy.run_path(str(evaluation / "evaluate.py"))
    learning = cfg["study"] == "learning_rule_25"
    require(not learning or train50 is None, "The learning-rule experiment uses TRAIN25 only")
    data, ready = {}, []
    for horizon in cfg["horizons"]:
        records = ev["rows"](inputs / f"trajectory_plans/train{horizon}.jsonl")
        require(len(records) == 1600 and all(r["split"] == "train" and r["steps"] == horizon for r in records), "TRAIN plan differs")
        plans = {r["file"]: dict(r) for r in records}
        require(len(plans) == 1600, "Duplicate TRAIN trajectory")
        if kind == "original":
            for name, row in plans.items(): row["original_npz_sha256"] = proto["original_train_payload_sha256"][name]
        data[horizon], present = ev["dataset_sources"](train25 if horizon == 25 else train50, plans, kind,
            f"train{horizon}", proto["components"]["inputs_manifest_sha256"])
        ready.append(present)
    panel = proto["panels"][cfg["study"]]
    require(len(panel) == len({r["pair_id"] for r in panel}) == 800, "Incomplete TRAIN pair panel")
    context = {"format": "dynaro-retraining-run-v1", "key": key, "configuration": cfg,
        "training_manifest_sha256": digest(HERE / "manifest.json"), "components": proto["components"],
        "data_kind": kind, "data_sha256": {str(h): {n: r["sha256"] for n, r in d.items()} for h, d in data.items()},
        "split_access": "TRAIN_only", "dev_or_test_evaluated": False}
    report = {"status": "read_only_training_plan", "key": key, "pairs": 800, "spatial_blocks": 10,
        "horizons": cfg["horizons"], "window_steps": cfg["window_steps"], "time_encoding": cfg["time_encoding"],
        "planned_optimizer_updates": cfg["optimizer_updates"], "checkpoint_selection": "final_update_only",
        "payload_files_present": all(ready), "trajectory_payloads_opened": 0, "initial_states_loaded": 0,
        "optimizer_updates": 0, "execution_enabled": False}
    return {"config": cfg, "context": context, "report": report, "panel": panel, "data": data,
            "core": core, "inputs": inputs, "evaluation": evaluation, "root": HERE}


def output_location(prepared, output, resume):
    require(output is not None, "Execution requires a separate --output directory")
    output = Path(output).absolute()
    require(not output.is_symlink() and not output.is_junction(), "Redirected output")
    output = output.resolve()
    require(output.parent.is_dir(), "Output parent must exist")
    roots = [HERE, prepared["core"], prepared["inputs"], prepared["evaluation"]]
    roots += list({r["path"].parent for d in prepared["data"].values() for r in d.values()})
    for root in roots:
        require(not output.is_relative_to(root) and not root.is_relative_to(output), "Output overlaps an input")
    require(not (output / "RUNNING.lock").exists(), "Existing execution lock; check its process first")
    if resume:
        require(output.is_dir() and not (output / "summary.json").exists(), "Resume requires incomplete own output")
        require(not any(p.is_symlink() or p.is_junction() for p in output.rglob("*")), "Redirected resume file")
        require(json.loads(safe_file(output, "context.json").read_text(encoding="utf-8")) == prepared["context"], "Resume context differs")
        safe_file(output, "checkpoint_latest.pt")
    else: require(not output.exists(), "Refusing to reuse existing output")
    return output


def execute(prepared, output, *, resume, max_seconds, session):
    output = output_location(prepared, output, resume)
    if not resume:
        output.mkdir()
        with (output / "context.json").open("x", encoding="utf-8") as f:
            json.dump(prepared["context"], f, indent=2); f.write("\n")
    lock = output / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as f: json.dump({"pid": os.getpid()}, f)
    began = time.monotonic()
    try:
        if resume: session.restore(output / "checkpoint_latest.pt")
        else: session.save(output / "checkpoint_latest.pt")
        target = prepared["config"]["optimizer_updates"]
        while session.completed < target:
            if time.monotonic() - began >= max_seconds or (output / "STOP_REQUESTED").exists():
                return {"status": "paused_at_update_boundary", "optimizer_updates": session.completed, "complete": False}
            result = session.step()
            # Checkpoint is the sole recovery source; progress is informative only.
            session.save(output / "checkpoint_latest.pt")
            with (output / "progress.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, allow_nan=False) + "\n")
            print(json.dumps({"optimizer_updates": session.completed, "target_updates": target}), flush=True)
        final = session.finish(output)
        with (output / "summary.json").open("x", encoding="utf-8") as f:
            json.dump(final, f, indent=2, allow_nan=False); f.write("\n")
        return {"status": final["status"], "optimizer_updates": session.completed, "complete": True}
    finally:
        lock.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "inputs", "evaluation"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--train25", type=Path)
    p.add_argument("--train50", type=Path)
    p.add_argument("--data-kind", choices=("original", "regenerated"), default="original")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-seconds", type=float, help="Soft limit checked after completed optimizer updates")
    a = p.parse_args(argv)
    prepared = prepare(a.core, a.inputs, a.evaluation, a.model_key, a.train25, a.train50, a.data_kind)
    if not a.execute:
        require(a.output is None and not a.resume and a.max_seconds is None, "Execution controls require --execute")
        print(json.dumps(prepared["report"], indent=2)); return
    require(a.max_seconds is not None and math.isfinite(a.max_seconds) and a.max_seconds > 0, "A positive duration is required")
    require(prepared["report"]["payload_files_present"], "Complete TRAIN files required")
    output_location(prepared, a.output, a.resume)
    sys.path.insert(0, str(HERE))
    import training_runtime
    session = training_runtime.bind(prepared)
    print(json.dumps(execute(prepared, a.output, resume=a.resume, max_seconds=a.max_seconds, session=session), indent=2))


if __name__ == "__main__": main()
