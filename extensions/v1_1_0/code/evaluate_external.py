"""Evaluate an explicitly selected frozen original model on the two-core panel.

Default is a metadata-only plan. --execute performs model inference and writes
to a new directory. The release already contains all original saved results.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/model_core/src')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('dynaro','static_front','coarse_front','rtu','fno'),required=True)
    p.add_argument('--seed',type=int,choices=(20260811,20260812,20260813),required=True)
    p.add_argument('--base-release',type=Path,help='Extracted v1.0.0 root containing reproduction/')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--pair-id',help='A fixed identity from external/targets/pairs.jsonl; omit for all 240')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    panel=[json.loads(x) for x in (ROOT/'external/targets/pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=[r for r in panel if a.pair_id is None or r['pair_id']==a.pair_id]
    if not selected:raise ValueError('Unknown pair identity')
    weight=next(r for r in json.loads((ROOT/'external/weights.json').read_text(encoding='utf-8')) if r['model']==a.model and r['seed']==a.seed)
    if not a.execute:
        print(json.dumps({'model':a.model,'seed':a.seed,'pairs':len(selected),'weight_sha256':weight['sha256'],'inference_started':False}));return
    if a.output is None or a.output.exists() or a.base_release is None:raise ValueError('Provide base release and a new output directory')
    import numpy as np
    import torch
    import dynaro,actions
    import regional_budget_objective as objective
    import regional_budget_readout as readout
    from scoring import long,short
    from external_numerics import model_inputs,cpu_output,score_compact_pair
    from aggregation import summarize_seed
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    path=a.base_release/weight['path'].removeprefix('v1.0.0/')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=weight['sha256']:raise ValueError('Frozen model differs')
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=dynaro.make_model(a.model);model.load_state_dict(state,strict=True)
    model=model.eval().requires_grad_(False).to(a.device)
    rt=SimpleNamespace(actions=actions,objective=objective,readout=readout,long=long,short=short)
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for row in selected:
        initial=np.load(ROOT/'external'/row['label_file'],allow_pickle=False)
        if hashlib.sha256(initial.tobytes()).hexdigest()!=row['initial_sha256']:raise ValueError('Initial labels differ')
        targets=[]
        for artifact in row['artifacts']:
            file=ROOT/'external/targets'/artifact['file']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=artifact['sha256']:raise ValueError('Target differs')
            with np.load(file,allow_pickle=False) as d:targets.append({k:d[k] for k in d.files})
        outputs=[]
        for sequence,target in zip(row['schedules'],targets):
            inputs=model_inputs(initial,target['connected_pore'],sequence,rt,a.device)
            with torch.inference_mode():outputs.append(cpu_output(model(*inputs,compact_output=True)))
        scores=score_compact_pair(initial,outputs,[t['first_removal_step'] for t in targets],rt)
        record={k:row[k] for k in ('pair_id','core_id','block_id')};record['scores']=scores
        rows.append(record)
        with (a.output/'pairs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps({'completed':len(rows),'pairs':len(selected)}),flush=True)
    if a.pair_id is None:
        (a.output/'summary.json').write_text(json.dumps(summarize_seed(rows,panel),indent=2),encoding='utf-8')

if __name__=='__main__':main()
