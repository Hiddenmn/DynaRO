"""Explicit CUDA execution; never imported by the metadata-only planning path."""
from __future__ import annotations
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def require(ok, message):
    if not ok: raise ValueError(message)


def import_components(prepared):
    paths = [prepared["core"] / "src", prepared["root"], prepared["evaluation"]]
    for p in reversed(paths):
        if str(p) not in sys.path: sys.path.insert(0, str(p))
    import dynaro
    import numerics
    import training_engine
    from scoring import io25
    for mod, root in ((dynaro, paths[0]), (numerics, paths[1]), (training_engine, paths[1]), (io25, paths[2])):
        require(Path(mod.__file__).resolve().is_relative_to(root), "Conflicting imported module")
    return dynaro, numerics, training_engine, io25


def initial_model(prepared, device):
    """Construct the recorded starting model; no forward pass or data reader."""
    import torch
    dynaro, numeric, _, _ = import_components(prepared)
    cfg = prepared["config"]
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda": torch.cuda.manual_seed_all(cfg["seed"])
    model = numeric.server_model(cfg["model"]) if cfg["model"] in ("fno", "rtu") else dynaro.make_model(cfg["model"])
    if "initialization_file" in cfg:
        row = cfg["initialization_file"]
        api = runpy.run_path(str(prepared["root"] / "verify_training.py"))
        path = api["safe_file"](prepared["root"], row["file"])
        require(api["digest"](path) == row["sha256"], "Pre-update state file differs")
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(set(state) == {"seed", "optimizer_updates", "model", "torch_rng_state", "initial_parameter_sha256"}
                and state["seed"] == cfg["seed"] and state["optimizer_updates"] == 0
                and state["initial_parameter_sha256"] == cfg["initial_parameter_sha256"], "Not the recorded pre-update state")
        require(all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in state["model"].values()), "Invalid initial tensor")
        model.load_state_dict(state["model"], strict=True)
        torch.set_rng_state(state["torch_rng_state"])
    model = model.to(device)
    require(sum(p.numel() for p in model.parameters()) == cfg["parameters"], "Parameter count differs")
    require(numeric.model_hash(model) == cfg["initial_parameter_sha256"],
            "Initial parameters differ from the recorded run; stop before reading TRAIN data. Do not replace the expected hash.")
    return model


def bind(prepared):
    require(sys.platform.startswith("linux"), "Explicit execution uses Linux/CUDA; no CPU training fallback")
    cfg, rt = prepared["config"], prepared["config"]["runtime"]
    allocator = rt["allocator_configuration"]
    if allocator:
        require("torch" not in sys.modules, "Set the recorded allocator before importing torch; use a fresh process")
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            require(name not in os.environ or os.environ[name] == allocator, "Conflicting allocator setting")
        os.environ["PYTORCH_ALLOC_CONF"] = allocator
    import torch
    import numpy as np
    require(torch.cuda.is_available(), "CUDA required; no silent CPU fallback")
    require(torch.cuda.device_count() == 1, "Use exactly one visible GPU for the recorded single-device route")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    require("4090" in props.name, "The recorded execution route uses an RTX 4090")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                                  text=True, timeout=15).strip().splitlines()
    require(len(raw) == 1, "Ambiguous physical GPU; check the selected server")
    free, utilization = [int(v.strip()) for v in raw[0].split(",")]
    require(free >= rt["minimum_free_mib"] and utilization <= rt["maximum_utilization_percent"], "GPU is busy or lacks required free memory")
    if rt["recorded_server_environment"]:
        require(sys.version_info[:2] == (3, 12) and str(torch.__version__).split("+")[0] == "2.8.0"
                and torch.version.cuda == "12.8" and np.__version__ == "2.3.2", "Recorded server runtime differs")
    torch.set_num_threads(rt["cpu_threads"])
    if rt["tf32"] is not None:
        torch.backends.cuda.matmul.allow_tf32 = rt["tf32"]
        torch.backends.cudnn.allow_tf32 = rt["tf32"]
    if rt["cudnn_benchmark"] is not None: torch.backends.cudnn.benchmark = rt["cudnn_benchmark"]
    fraction = rt["memory_fraction"]
    if rt["allocator_cap_mib"] is not None: fraction = rt["allocator_cap_mib"] * 2**20 / props.total_memory
    require(fraction is not None and 0 < fraction <= 1, "Missing allocator limit")
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    model = initial_model(prepared, device)
    runtime = {"python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "numpy": np.__version__, "gpu": props.name, "cpu_threads": rt["cpu_threads"], "precision": "float32",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "allocator_fraction": fraction}
    return Session(prepared, model, device, runtime)


class Session:
    def __init__(self, prepared, model, device, runtime):
        import torch
        self.prepared, self.model, self.device, self.runtime = prepared, model, device, runtime
        self.cfg = prepared["config"]
        _, self.numeric, self.engine, self.io = import_components(prepared)
        self.api = runpy.run_path(str(prepared["root"] / "verify_training.py"))
        self.order = self.engine.shuffled_order(prepared["panel"], self.cfg)
        opt = self.cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(tuple(model.parameters()), lr=opt["lr"], weight_decay=opt["weight_decay"])
        self.completed = 0
        self.loss_sums = {name: 0.0 for name in ("site", "region", "response")}
        self.context_sha256 = __import__("hashlib").sha256(json.dumps(prepared["context"], sort_keys=True).encode()).hexdigest()

    def check_memory(self):
        import torch
        require(torch.cuda.max_memory_reserved(self.device) <= self.cfg["runtime"]["maximum_reserved_mib"] * 2**20,
                "Recorded memory ceiling exceeded")

    def load_pair(self, record, horizon):
        import torch
        import numpy as np
        entries = []
        for name in record[str(horizon)]:
            r = self.prepared["data"][horizon][name]
            require(r["path"].stat().st_size == r["bytes"] and self.api["digest"](r["path"]) == r["sha256"], "TRAIN payload differs")
            entries.append((r["path"], r["record"]))
        pair = self.io.load_pair({horizon: entries}, horizon, self.device)
        expected = np.load(self.api["safe_file"](self.prepared["inputs"], entries[0][1]["label_file"]), allow_pickle=False)
        require(np.array_equal(pair[0].cpu().numpy(), expected), "Initial labels differ from the reviewed TRAIN input")
        if self.cfg["study"] == "learning_rule_25":
            times = (torch.arange(1, 26, dtype=torch.float32).unsqueeze(1) / 25).to(self.device)
            pair = (*pair[:5], times, *pair[6:])
        return pair

    def step(self):
        import torch
        width = self.cfg["pair_ids_per_update"]
        start = self.completed * width
        batch = [self.prepared["panel"][i] for i in self.order[start:start+width]]
        torch.cuda.reset_peak_memory_stats(self.device)
        result = self.engine.one_update(self.model, self.optimizer, self.cfg, batch, self.load_pair, self.check_memory)
        torch.cuda.synchronize(self.device)
        self.check_memory()
        self.completed += 1
        for name, value in result["loss"].items(): self.loss_sums[name] += value
        return {"optimizer_updates": self.completed, "pair_ids": [r["pair_id"] for r in batch], **result}

    def save(self, path):
        import torch
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        require(not tmp.exists(), "Interrupted checkpoint exists; inspect before resuming")
        state = {"format": "dynaro-retraining-checkpoint-v1", "context_sha256": self.context_sha256,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "optimizer_updates": self.completed, "loss_sums": self.loss_sums,
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "runtime": self.runtime}
        with tmp.open("xb") as f:
            torch.save(state, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)

    def restore(self, path):
        import torch
        import math
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(state["format"] == "dynaro-retraining-checkpoint-v1" and state["context_sha256"] == self.context_sha256,
                "Not a matching new-driver recovery checkpoint")
        require(state["runtime"] == self.runtime, "Resume runtime differs")
        completed = state["optimizer_updates"]
        require(type(completed) is int and 0 <= completed <= self.cfg["optimizer_updates"], "Invalid recovery position")
        require(set(state["loss_sums"]) == set(self.loss_sums)
                and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in state["loss_sums"].values()), "Invalid saved loss")
        def finite(value):
            if isinstance(value, torch.Tensor): require(bool(torch.isfinite(value).all()), "Non-finite saved tensor")
            elif isinstance(value, float): require(math.isfinite(value), "Non-finite saved scalar")
            elif isinstance(value, dict):
                for v in value.values(): finite(v)
            elif isinstance(value, (list, tuple)):
                for v in value: finite(v)
        finite(state["model"]); finite(state["optimizer"])
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        expected = self.cfg["optimizer"]
        require(all(g["lr"] == expected["lr"] and g["weight_decay"] == expected["weight_decay"]
                    and tuple(g["betas"]) == (0.9, 0.999) and g["eps"] == 1e-8
                    and not g["amsgrad"] and not g["maximize"] and not g["capturable"]
                    for g in self.optimizer.param_groups), "Saved optimizer settings differ")
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        self.completed, self.loss_sums = completed, state["loss_sums"]

    def finish(self, output):
        import torch
        require(self.completed == self.cfg["optimizer_updates"], "Only the final update is selection eligible")
        state = {name: value.detach().cpu().contiguous().clone() for name, value in self.model.state_dict().items()}
        require(all(bool(torch.isfinite(v).all()) for v in state.values()), "Non-finite final state")
        path = Path(output) / "retrained_final.pt"
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            with tmp.open("xb") as f: torch.save(state, f)
            os.replace(tmp, path)
        else:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            require(list(saved) == list(state) and all(torch.equal(saved[n], state[n]) for n in state), "Existing final artifact differs")
        return {"status": "complete_retraining_not_original_result", "configuration": self.cfg,
            "runtime": self.runtime, "optimizer_updates": self.completed,
            "mean_training_losses": {n: v / self.completed for n, v in self.loss_sums.items()},
            "checkpoint": path.name, "checkpoint_sha256": self.api["digest"](path),
            "parameter_sha256": self.numeric.model_hash(self.model), "selection": "final_update_only",
            "split_access": "TRAIN_only", "dev_or_test_evaluated": False,
            "context_sha256": self.context_sha256}
