"""Unified frozen-checkpoint evaluation entry; default is a read-only plan."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
API=runpy.run_path(str(HERE/"verify_evaluation.py"))
require,digest,safe_file=(API[k] for k in ("require","digest","safe_file"))


def rows(path):
    raw=Path(path).read_bytes()
    require(not raw or raw.endswith(b"\n"),"Incomplete JSON line; preserve the file")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def dataset_sources(directory, plans, kind, group, input_sha):
    if directory is None:return {},False
    directory=Path(directory).resolve(strict=True)
    ledger=None
    if kind=="regenerated":
        summary=json.loads(safe_file(directory,"generation_summary.json").read_text(encoding="utf-8"))
        require(summary["status"]=="selected_trajectories_generated" and summary["complete_panel"] is True
                and summary["set"]==group and summary["generated_trajectories"]==len(plans)
                and summary["input_manifest_sha256"]==input_sha,"Incomplete/different regenerated batch")
        records=rows(safe_file(directory,"generated_trajectories.jsonl"))
        ledger={r["file"]:r for r in records}
        require(len(records)==len(ledger)==len(plans),"Regenerated batch count differs")
    result={}
    for name,row in plans.items():
        relative="data/"+name if ledger is not None else name
        path=safe_file(directory,relative)
        require(path.stat().st_size>0,"Empty trajectory file")
        sha=row["original_npz_sha256"]
        if ledger is not None:
            generated=ledger[relative]
            for field in ("block_id","stage","stream_id","label_source_sha256","action_ids","steps"):
                require(generated[field]==row[field],"Regenerated identity differs: "+field)
            require(generated["original_npz_sha256"]==sha and generated["bytes"]==path.stat().st_size,
                    "Regenerated provenance/size differs")
            sha=generated["generated_npz_sha256"]
        require(isinstance(sha,str) and len(sha)==64,"Missing trajectory hash")
        result[name]={"path":path,"sha256":sha,"bytes":path.stat().st_size,"record":row}
    return result,True


def prepare(core,weights,inputs,key,test25=None,test100=None,kind="original"):
    require(kind in ("original","regenerated"),"Unknown trajectory source")
    API["verify"](HERE)
    proto=json.loads(safe_file(HERE,"protocol.json").read_text(encoding="utf-8"))
    require(key in proto["model_keys"],"Unknown model key; inspect protocol.json")
    core,weights,inputs=map(lambda p:Path(p).resolve(strict=True),(core,weights,inputs))
    for root,field in ((inputs,"inputs_manifest_sha256"),(weights,"weights_manifest_sha256")):
        require(digest(safe_file(root,"manifest.json"))==proto["components"][field],"Different prepared component")
    im=json.loads(safe_file(inputs,"manifest.json").read_text(encoding="utf-8"))
    require(digest(safe_file(inputs,"verify_inputs.py"))==im["files"]["verify_inputs.py"]["sha256"],"Changed input verifier")
    runpy.run_path(str(inputs/"verify_inputs.py"))["verify"](inputs,arrays=False)
    wm=json.loads(safe_file(weights,"manifest.json").read_text(encoding="utf-8"))
    require(digest(safe_file(weights,"load_weights.py"))==wm["support_files"]["load_weights.py"]["sha256"],"Changed weight loader")
    runpy.run_path(str(weights/"load_weights.py"))["read_manifest"](weights)
    for name,sha in proto["components"]["model_core_files"].items():
        require(digest(safe_file(core,name))==sha,"Different model source: "+name)
    identity=wm["models"][key]
    learned=identity["study"]=="learning_rule_25"
    require(not learned or test100 is None,"The 25-step learning-rule study does not use TEST100")
    plans={h:{r["file"]:r for r in rows(inputs/f"trajectory_plans/test{h}.jsonl")} for h in ((25,) if learned else (25,100))}
    data,ready={},[]
    for h in plans:
        data[h],found=dataset_sources(test25 if h==25 else test100,plans[h],kind,f"test{h}",proto["components"]["inputs_manifest_sha256"])
        ready.append(found)
    main={r["pair_id"]:r for r in proto["pairs"]}
    panel=([{**main[r["main_pair_id"]],"pair_id":r["pair_id"],"main_pair_id":r["main_pair_id"]}
            for r in proto["learning_order"]] if learned else proto["pairs"])
    require(len(panel)==600 and len({p["pair_id"] for p in panel})==600,"Incomplete pair panel")
    context={"format":"dynaro-evaluation-run-v1","key":key,"study":identity["study"],"seed":identity["seed"],
        "weights_sha256":identity["sha256"],"evaluation_manifest_sha256":digest(HERE/"manifest.json"),
        "input_manifest_sha256":proto["components"]["inputs_manifest_sha256"],"data_kind":kind,
        "time_encoding":"t/25" if learned else "t/(t+25)","horizons":[25] if learned else [25,50,75,100],
        "pairs":600,"spatial_blocks":5,"optimizer_updates":0,
        "data_sha256":{str(h):{n:r["sha256"] for n,r in values.items()} for h,values in data.items()}}
    report={"status":"read_only_evaluation_plan","key":key,"study":identity["study"],"pairs":600,
        "spatial_blocks":5,"horizons":context["horizons"],"time_encoding":context["time_encoding"],
        "trajectory_source":kind,"payload_files_present":all(ready),"trajectory_payloads_opened":0,
        "weights_loaded":0,"pairs_scored":0,"execution_enabled":False}
    return {"report":report,"context":context,"core":core,"weights":weights,"inputs":inputs,
            "data":data,"panel":panel,"identity":identity,"protocol":proto}


def score_schema(scores,study,arm):
    def numeric(v):return type(v) in (int,float) and math.isfinite(v) and v>=0
    measures={"site","region","response","zero_response","unchanged_site","unchanged_region"}
    if study=="learning_rule_25":
        require(set(scores)=={"all_25","steps_1_5","steps_6_15","steps_16_25"},"Different learning-rule intervals")
        for values in scores.values():
            require(set(values)==measures and all(numeric(v) for v in values.values()),"Invalid learning-rule metric")
        return
    require(set(scores)=={"25","50","75","100"},"Different mixed-horizon endpoints")
    for h,models in scores.items():
        require(set(models)=={arm},"Different checkpoint in score record")
        values=models[arm]
        long=int(h)>50
        metrics=measures|({"frozen50_site","frozen50_region"} if long else set())
        curves=metrics if long else {"site","response"}
        intervals={"all","tail","last","unseen" if long else "early"}
        extra={"last_class_inventory_fraction_mae"}|({"last_class_inventory_fraction_mae_by_class"} if long else set())
        require(set(values)==intervals|extra|{m+"_by_step" for m in curves},"Different metric schema")
        for name in intervals:
            require(set(values[name])==metrics and all(numeric(v) for v in values[name].values()),"Invalid interval metric")
        for name in curves:
            v=values[name+"_by_step"]
            require(len(v)==int(h) and all(numeric(x) for x in v),"Invalid per-step curve")
        require(numeric(values["last_class_inventory_fraction_mae"]),"Invalid class inventory error")
        if long:
            v=values["last_class_inventory_fraction_mae_by_class"]
            require(len(v)==5 and all(x is None or numeric(x) for x in v),"Invalid per-class inventory error")


def output_location(prepared,output,resume):
    require(output is not None,"Execution requires a separate --output directory")
    output=Path(output).absolute()
    require(not output.is_symlink() and not output.is_junction(),"Redirected output")
    output=output.resolve()
    require(output.parent.is_dir(),"Output parent must exist")
    for root in (HERE,prepared["core"],prepared["weights"],prepared["inputs"],
                 *(r["path"].parent for d in prepared["data"].values() for r in d.values())):
        require(not output.is_relative_to(root) and not root.is_relative_to(output),"Output overlaps an input")
    if resume:
        require(output.is_dir() and not (output/"summary.json").exists(),"Resume requires incomplete own output")
        require(not any(p.is_symlink() or p.is_junction() for p in output.rglob("*")),"Redirected resume file")
        require(json.loads(safe_file(output,"context.json").read_text(encoding="utf-8"))==prepared["context"],"Resume context differs")
    else:require(not output.exists(),"Refusing to reuse existing output")
    require(not (output/"RUNNING.lock").exists(),"Existing execution lock: check its process first")
    return output


def saved_prefix(prepared,output):
    path=output/"per_pair.jsonl"
    result=rows(path) if path.exists() else []
    require(len(result)<=len(prepared["panel"]),"Too many saved pairs")
    identity=prepared["identity"]
    arm=identity["configuration"]["historical_arm"]
    for r,p in zip(result,prepared["panel"]):
        require(all(r[k]==p[k] for k in ("pair_id","block_id","stage","stream_id")),"Saved prefix differs")
        require(r["key"]==prepared["context"]["key"],"Saved checkpoint differs")
        score_schema(r["scores"],identity["study"],arm)
    return result


def run_loop(prepared,output,*,resume,max_seconds,score,summarize):
    output=output_location(prepared,output,resume)
    result=saved_prefix(prepared,output) if resume else []
    if not resume:
        output.mkdir()
        with (output/"context.json").open("x",encoding="utf-8") as f:json.dump(prepared["context"],f,indent=2);f.write("\n")
    lock=output/"RUNNING.lock"
    with lock.open("x",encoding="utf-8") as f:json.dump({"pid":os.getpid()},f)
    started=time.monotonic()
    try:
        with (output/"per_pair.jsonl").open("a",encoding="utf-8") as f:
            for pair in prepared["panel"][len(result):]:
                if time.monotonic()-started>=max_seconds or (output/"STOP_REQUESTED").exists():
                    return {"status":"paused_at_pair_boundary","pairs_scored":len(result),"complete":False}
                scores=score(pair)
                score_schema(scores,prepared["identity"]["study"],prepared["identity"]["configuration"]["historical_arm"])
                row={**{k:pair[k] for k in ("pair_id","block_id","stage","stream_id")},
                     "key":prepared["context"]["key"],"scores":scores}
                f.write(json.dumps(row,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
                result.append(row)
                print(json.dumps({"completed_pairs":len(result),"target_pairs":len(prepared["panel"])}),flush=True)
        final={"status":"complete_frozen_checkpoint_evaluation","context":prepared["context"],
               "runtime":prepared.get("runtime",{}),"pairs_scored":len(result),**summarize(result)}
        with (output/"summary.json").open("x",encoding="utf-8") as f:json.dump(final,f,indent=2,allow_nan=False);f.write("\n")
        return {"status":final["status"],"pairs_scored":len(result),"complete":True}
    finally:
        lock.unlink()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("core","weights","inputs"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--model-key",required=True,help="Exact key from the weights manifest")
    p.add_argument("--test25",type=Path,help="Original NPZ directory, or regenerated batch with --data-kind regenerated")
    p.add_argument("--test100",type=Path)
    p.add_argument("--data-kind",choices=("original","regenerated"),default="original")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--max-seconds",type=float,help="Required soft limit; finish the current pair before stopping")
    a=p.parse_args(argv)
    prepared=prepare(a.core,a.weights,a.inputs,a.model_key,a.test25,a.test100,a.data_kind)
    if not a.execute:
        require(a.output is None and not a.resume and a.max_seconds is None,"Run controls require --execute")
        print(json.dumps(prepared["report"],indent=2));return
    require(a.max_seconds is not None and math.isfinite(a.max_seconds) and a.max_seconds>0,"Execution requires positive --max-seconds")
    require(prepared["report"]["payload_files_present"],"Complete trajectory files required")
    output_location(prepared,a.output,a.resume)
    sys.path.insert(0,str(HERE))
    import runtime
    score,summarize=runtime.bind(prepared)
    print(json.dumps(run_loop(prepared,a.output,resume=a.resume,max_seconds=a.max_seconds,score=score,summarize=summarize),indent=2))


if __name__=="__main__":main()
