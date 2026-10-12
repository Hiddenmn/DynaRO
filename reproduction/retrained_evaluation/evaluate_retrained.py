"""Evaluate an explicitly identified NEW final training output, never replace it."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
API = runpy.run_path(str(HERE / "verify_bridge.py"))
require, digest, safe_file = (API[k] for k in ("require", "digest", "safe_file"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def context_hash(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def completed_training(run, training, tp, key):
    """Metadata and file-byte check only; no checkpoint deserialization."""
    run = Path(run).resolve(strict=True)
    require(not (run / "RUNNING.lock").exists(), "Training output is marked active")
    context_file, summary_file = safe_file(run, "context.json"), safe_file(run, "summary.json")
    context, summary = load_json(context_file), load_json(summary_file)
    cfg = tp["configurations"][key]
    require(context["format"] == "dynaro-retraining-run-v1" and context["key"] == key
            and context["configuration"] == cfg and context["training_manifest_sha256"] == digest(training / "manifest.json")
            and context["components"] == tp["components"] and context["split_access"] == "TRAIN_only"
            and context["dev_or_test_evaluated"] is False, "Different training context")
    require(summary["status"] == "complete_retraining_not_original_result" and summary["configuration"] == cfg
            and summary["optimizer_updates"] == cfg["optimizer_updates"] and summary["selection"] == "final_update_only"
            and summary["split_access"] == "TRAIN_only" and summary["dev_or_test_evaluated"] is False
            and summary["context_sha256"] == context_hash(context), "Not a completed matching final-update training run")
    require(context["data_kind"] in ("original", "regenerated"), "Unknown training data provenance")
    require(set(context["data_sha256"]) == {str(h) for h in cfg["horizons"]}, "Training horizons differ")
    panel = tp["panels"][cfg["study"]]
    for h in cfg["horizons"]:
        hashes = context["data_sha256"][str(h)]
        names = {n for p in panel for n in p[str(h)]}
        require(set(hashes) == names and len(hashes) == 1600
                and all(isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v) for v in hashes.values()), "Incomplete TRAIN provenance")
        if context["data_kind"] == "original":
            require(all(sha == tp["original_train_payload_sha256"][name] for name, sha in hashes.items()), "Original TRAIN payload identities differ")
    require(summary["checkpoint"] == "retrained_final.pt", "Recovery checkpoints are not selection eligible")
    checkpoint = safe_file(run, summary["checkpoint"])
    require(checkpoint.stat().st_size > 0 and digest(checkpoint) == summary["checkpoint_sha256"], "Retrained final file differs")
    require(isinstance(summary["parameter_sha256"], str) and re.fullmatch(r"[a-f0-9]{64}", summary["parameter_sha256"]),
            "Missing final parameter identity")
    losses = summary["mean_training_losses"]
    require(set(losses) == {"site", "region", "response"}
            and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in losses.values()), "Invalid training summary")
    return {"root": run, "context": context, "summary": summary, "checkpoint": checkpoint,
            "checkpoint_bytes": checkpoint.stat().st_size, "summary_sha256": digest(summary_file),
            "context_file_sha256": digest(context_file)}


def prepare(core, weights, inputs, evaluation, training, key, run=None, test25=None, test100=None, kind="original"):
    API["verify"](HERE)
    bp = load_json(safe_file(HERE, "protocol.json"))
    training, evaluation = map(lambda p: Path(p).resolve(strict=True), (training, evaluation))
    for root, field, verifier in ((training, "training_manifest_sha256", "verify_training.py"),
                                  (evaluation, "evaluation_manifest_sha256", "verify_evaluation.py")):
        require(digest(safe_file(root, "manifest.json")) == bp[field], "Different prepared component")
        manifest = load_json(root / "manifest.json")
        require(digest(safe_file(root, verifier)) == manifest["files"][verifier]["sha256"], "Changed component verifier")
        runpy.run_path(str(root / verifier))["verify"](root)
    tp = load_json(safe_file(training, "protocol.json"))
    require(key in bp["model_keys"] and key in tp["configurations"], "Unknown training/evaluation configuration")
    ev = runpy.run_path(str(evaluation / "evaluate.py"))
    prepared = ev["prepare"](core, weights, inputs, key, test25, test100, kind)
    cfg = tp["configurations"][key]
    require(cfg["configuration"] == prepared["identity"]["configuration"]
            and cfg["study"] == prepared["identity"]["study"]
            and cfg["optimizer_updates"] == prepared["identity"]["final_optimizer_update"], "Architecture/study mismatch")
    source = None if run is None else completed_training(run, training, tp, key)
    context = prepared["context"]
    reference_sha = context.pop("weights_sha256")
    context.update(format="dynaro-retrained-evaluation-run-v1", checkpoint_origin="new_retraining_final",
        reference_frozen_weights_sha256=reference_sha,
        weights_sha256=None if source is None else source["summary"]["checkpoint_sha256"],
        bridge_manifest_sha256=digest(HERE / "manifest.json"), training_manifest_sha256=bp["training_manifest_sha256"],
        training_context_sha256=None if source is None else source["summary"]["context_sha256"],
        training_summary_sha256=None if source is None else source["summary_sha256"],
        training_data_kind=None if source is None else source["context"]["data_kind"],
        completed_training_updates=None if source is None else cfg["optimizer_updates"])
    prepared.update(training=training, evaluation=evaluation, retrained=source, bridge=HERE, evaluation_api=ev)
    prepared["report"].update(status="read_only_retrained_evaluation_plan", checkpoint_origin="new_retraining_final",
        completed_training_output_supplied=source is not None, weights_loaded=0, pairs_scored=0,
        executable_inputs_ready=source is not None and prepared["report"]["payload_files_present"])
    return prepared


def load_retrained(prepared):
    """Strict CPU tensor-only load; no model forward call or TEST reader."""
    import torch
    source = prepared["retrained"]
    require(source is not None, "A completed training output is required")
    for name, sha in (("summary.json", source["summary_sha256"]), ("context.json", source["context_file_sha256"])):
        require(digest(safe_file(source["root"], name)) == sha, "Training record changed after planning")
    path = source["checkpoint"]
    require(path.stat().st_size == source["checkpoint_bytes"]
            and digest(path) == source["summary"]["checkpoint_sha256"], "Retrained checkpoint changed after planning")
    loader = runpy.run_path(str(prepared["weights"] / "load_weights.py"))
    wm = loader["read_manifest"](prepared["weights"])
    original = wm["models"][prepared["context"]["key"]]
    state = torch.load(path, map_location="cpu", weights_only=True)
    actual = loader["tensor_records"](state)
    require(set(actual) == set(original["tensors"]) and all(
        actual[n]["shape"] == original["tensors"][n]["shape"] and actual[n]["dtype"] == original["tensors"][n]["dtype"] for n in actual),
        "Retrained tensor architecture or dtype differs")
    factory = loader["checked_core"](prepared["core"], wm)
    with torch.random.fork_rng(devices=[]): model = factory.make_model(original["model"])
    model.load_state_dict(state, strict=True)
    require(loader["historical_state_hash"](model.state_dict()) == source["summary"]["parameter_sha256"]
            and sum(v.numel() for v in model.parameters()) == original["parameters"], "Loaded final state differs from training summary")
    identity = {k: original[k] for k in ("study", "model", "learning_rule", "seed", "configuration", "parameters")}
    identity.update(checkpoint_origin="new_retraining_final", sha256=source["summary"]["checkpoint_sha256"],
                    parameter_sha256=source["summary"]["parameter_sha256"], selection="final_update_only")
    return model.eval().requires_grad_(False), identity


def output_location(prepared, output, resume):
    require(prepared["retrained"] is not None, "A completed training output is required")
    path = prepared["evaluation_api"]["output_location"](prepared, output, resume)
    for root in (HERE, prepared["training"], prepared["retrained"]["root"]):
        require(not path.is_relative_to(root) and not root.is_relative_to(path), "Evaluation output overlaps the retrained source/component")
    return path


def run_evaluation(prepared, output, *, resume, max_seconds, score, summarize):
    output_location(prepared, output, resume)
    def separate_summary(rows):
        return {**summarize(rows), "status": "complete_retrained_checkpoint_evaluation",
                "checkpoint_origin": "new_retraining_final"}
    return prepared["evaluation_api"]["run_loop"](prepared, output, resume=resume, max_seconds=max_seconds,
                                                      score=score, summarize=separate_summary)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "weights", "inputs", "evaluation", "training"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--run", type=Path, help="Completed output of the new training driver")
    p.add_argument("--test25", type=Path)
    p.add_argument("--test100", type=Path)
    p.add_argument("--data-kind", choices=("original", "regenerated"), default="original")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-seconds", type=float)
    a = p.parse_args(argv)
    prepared = prepare(a.core, a.weights, a.inputs, a.evaluation, a.training, a.model_key, a.run, a.test25, a.test100, a.data_kind)
    if not a.execute:
        require(a.output is None and not a.resume and a.max_seconds is None, "Execution controls require --execute")
        print(json.dumps(prepared["report"], indent=2)); return
    require(prepared["report"]["executable_inputs_ready"], "Complete training output and TEST files required")
    require(a.max_seconds is not None and math.isfinite(a.max_seconds) and a.max_seconds > 0, "Positive duration required")
    output_location(prepared, a.output, a.resume)
    sys.path[:0] = [str(HERE), str(prepared["evaluation"])]
    import retrained_runtime
    score, summarize = retrained_runtime.bind(prepared)
    print(json.dumps(run_evaluation(prepared, a.output, resume=a.resume, max_seconds=a.max_seconds,
                                    score=score, summarize=summarize), indent=2))


if __name__ == "__main__": main()
