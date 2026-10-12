"""Explicit CUDA route for new retrained weights and unchanged score functions."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import runpy
import sys


def bind(prepared):
    # The command's normal read-only branch never imports this module.
    api=runpy.run_path(str(prepared["evaluation"]/"verify_evaluation.py"))
    require,digest=api["require"],api["digest"]
    require(sys.platform.startswith("linux"),"Numerical reproduction uses the recorded Linux/CUDA route")
    import torch
    import numpy as np
    require(torch.cuda.is_available(),"CUDA required; no silent CPU TEST fallback")
    device=torch.device("cuda:0")
    props=torch.cuda.get_device_properties(device)
    require("4090" in props.name,"This route retains the recorded RTX 4090 requirement")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    learning=prepared["identity"]["study"]=="learning_rule_25"
    cap=1100 if learning else (3000 if prepared["identity"]["model"]=="coarse_front" else 6000)
    require(props.total_memory>cap*2**20,"Insufficient GPU memory")
    torch.cuda.set_per_process_memory_fraction(cap*2**20/props.total_memory,device)
    from evaluate_retrained import load_retrained
    model,identity=load_retrained(prepared)
    model=model.to(device)
    sys.path.insert(0,str(prepared["inputs"]/"src"))
    for name in ("patent22","patent22.actions"):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve().is_relative_to(prepared["inputs"]),"Conflicting action module")
    from scoring import io25,short,long,learning as rule,aggregate
    from scoring.util import RegionalPair
    arm=identity["configuration"]["historical_arm"]
    prepared["runtime"]={"python":sys.version,"torch":str(torch.__version__),"numpy":np.__version__,
                         "cuda":torch.version.cuda,"gpu":props.name,"precision":"float32","tf32":False,"cpu_threads":2}

    def checked(h,name):
        value=prepared["data"][h][name]
        require(value["path"].stat().st_size==value["bytes"] and digest(value["path"])==value["sha256"],"Trajectory payload differs")
        return value["path"],value["record"]

    @torch.inference_mode()
    def score(record):
        torch.cuda.reset_peak_memory_stats(device)
        entries25=tuple(checked(25,n) for n in record["25"])
        pair25=io25.load_pair({25:entries25},25,device)
        initial=pair25[0].cpu().clone()
        expected=np.load(api["safe_file"](prepared["inputs"],entries25[0][1]["label_file"]),allow_pickle=False)
        require(np.array_equal(initial.numpy(),expected),"Initial label differs from the reviewed input")
        del expected
        if learning:
            labels,inlet,connected,aa,ab,_,ta,tb=pair25
            # The separate original study uses t/25, not mixed-horizon time encoding.
            times=torch.arange(1,26,dtype=torch.float32).unsqueeze(1)/25
            pair=RegionalPair(labels.cpu(),inlet.cpu(),connected.cpu(),aa.cpu(),ab.cpu(),times,ta,tb,record["pair_id"])
            scores=rule.score_pair(model,pair,device)
        else:
            scores={"25":short._score_pair({arm:model},pair25,25)}
            del pair25
            entries=[checked(100,n) for n in record["100"]]
            pair,_=long.read_pair({"files":[e[0] for e in entries],"schedules":[e[1]["action_ids"] for e in entries]},device)
            require(torch.equal(initial,pair[0].cpu()),"Short/long initial geometry differs")
            scores["50"]=short._score_pair({arm:model},aggregate.prefix50(pair),50)
            scores.update(long.score_pair({arm:model},pair))
        torch.cuda.synchronize(device)
        require(torch.cuda.max_memory_reserved(device)<=cap*2**20,"Recorded memory cap exceeded")
        return scores

    def summarize(values):
        return rule.summarize_arm(values) if learning else aggregate.summarize(values,{"panel":prepared["panel"]})
    return score,summarize
