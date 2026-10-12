"""CPU synthetic software checks; no research data or trained checkpoints."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def run(include_unit_tests=True):
    import numpy as np
    import torch
    if torch.version.cuda is not None:
        raise RuntimeError("Use the CPU-only environment for these software checks")
    torch.set_num_threads(2)

    def forbid(*args, **kwargs):
        raise RuntimeError("This checker must not access research binaries, weights or CUDA")

    torch.load = torch.save = forbid
    torch.cuda._lazy_init = forbid
    np.load = np.save = np.savez = np.savez_compressed = forbid

    def restrict(event, args):
        if event != "open":
            return
        path, mode, flags = args
        if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).suffix.lower() \
                in (".npz", ".npy", ".pt", ".pth"):
            forbid()
        if isinstance(mode, str) and any(v in mode for v in ("w", "a", "x", "+")):
            forbid()
        if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            forbid()

    sys.addaudithook(restrict)
    import dynaro
    from test_tnnls_remaining_oil_models import synthetic_inputs, TNNLSModelTests
    labels, inlet, connected, actions, _, times, *_ = synthetic_inputs(size=32, horizon=4)
    records = []
    for name, config in dynaro.configurations().items():
        torch.manual_seed(20261009)
        model = dynaro.make_model(name).eval()
        with torch.no_grad():
            output = model(labels, inlet, connected, actions, times, return_fields=True)
            prefix = model(labels, inlet, connected, actions[:2], times[:2], compact_output=True)
        sites = output["remaining_sites"]
        if (not bool(torch.isfinite(sites).all()) or not bool(((sites >= 0) & (sites <= 1)).all())
                or not bool((sites[1:] <= sites[:-1]).all())):
            raise ValueError("Invalid remaining-oil state")
        torch.testing.assert_close(sites[:2], prefix["remaining_sites"], atol=0, rtol=0)
        fields = output["So_sequence"]
        if tuple(fields.shape) != (4, 32, 32, 32) or not bool((fields[:, labels < 3] == 0).all()):
            raise ValueError("Invalid field support")
        torch.testing.assert_close(fields.flatten(1).sum(1), output["class_remaining"].sum(1))
        if json.loads(json.dumps(asdict(model.config))) != config["architecture"]:
            raise ValueError("Architecture configuration differs")
        records.append({"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                        "shape": list(fields.shape), "bounded_monotone_state": True, "prefix_causality": True})
        del model, output, prefix, fields, sites
        gc.collect()
    tests = None
    if include_unit_tests:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(TNNLSModelTests))
        tests = {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}
        if not result.wasSuccessful():
            raise RuntimeError("Original synthetic model/gradient tests failed")
    return {"status": "synthetic_cpu_pass", "models": records, "original_unit_tests": tests,
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "research_binaries_opened": 0, "checkpoints_loaded": 0, "optimizer_updates": 0,
            "test_dataset_evaluated": False, "numerical_experiment_reproduction": "not_run"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(not args.forward_only), indent=2))
