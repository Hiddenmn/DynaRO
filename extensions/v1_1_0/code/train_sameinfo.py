"""Portable fixed-protocol training entry. Does not run unless --execute is set.

The scientific update follows the archived 400-update experiment. The published
final states are the original results, not outputs of this portability adapter.
Regenerate the original TRAIN trajectories with the v1.0.0 reproduction tools.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/model_core/src'),str(ROOT/'code/sameinfo')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True,
        help='Root containing the historical workflow_full_audit_20260717/paper1_workspace/runs tree')
    p.add_argument('--arm',choices=('explicit_contact','learned_contact'),required=True)
    p.add_argument('--seed',type=int,choices=(20260811,20260812,20260813),required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    import torch
    import portable_train_data as data
    from sameinfo_model import SameInformationContactSurrogate
    from frozen_initialization import restore_initialization
    from training_numerics import accumulate
    from regional_state_guard import assign_state_guarded_gradient,new_gradient_buffer
    from source_helpers import model_hash
    torch.set_num_threads(2)
    plan=json.loads((ROOT/'sameinfo/protocol.json').read_text(encoding='utf-8'))
    rows=data.panel(a.data_root);order,digest=data.pair_order(rows,a.seed)
    expected=plan['seeds'][str(a.seed)]
    if digest!=expected['pair_order_sha256']:raise ValueError('Pair order differs')
    if not a.execute:
        print(json.dumps({'pairs':len(rows),'seed':a.seed,'updates':400,'pair_order_sha256':digest,'training_started':False}))
        return
    if a.output is None or a.output.exists():raise ValueError('Provide a new output directory')
    lock=json.loads((ROOT/'sameinfo/train_input_lock.json').read_text(encoding='utf-8'))
    for rel,entry in lock['payloads'].items():
        path=a.data_root/rel
        if path.stat().st_size!=entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('TRAIN payload differs: '+rel)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for full 128-cubed training')
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=SameInformationContactSurrogate(a.arm)
    restore_initialization(model,a.seed,expected['initial_sha256'],directory=ROOT/'sameinfo/initialization')
    model=model.cuda().train();parameters=tuple(model.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    a.output.mkdir(parents=True,exist_ok=False)
    for update in range(400):
        torch.cuda.empty_cache()
        primary,auxiliary=new_gradient_buffer(parameters),new_gradient_buffer(parameters)
        for index in order[2*update:2*update+2]:
            p25,p50=(data.load_pair(rows[index],h,'cuda:0') for h in (25,50))
            if any(not torch.equal(p25[i],p50[i]) for i in range(3)):raise ValueError('Geometry differs')
            for pair in (p25,p50):accumulate(model,parameters,pair,primary,auxiliary)
            del pair,p25,p50
        assign_state_guarded_gradient(parameters,primary,auxiliary,
            auxiliary_weight=50.,max_auxiliary_to_primary_norm=1.)
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        print(json.dumps({'update':update+1}),flush=True)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},a.output/'final_state.pt')
    (a.output/'summary.json').write_text(json.dumps({'arm':a.arm,'seed':a.seed,'updates':400,
        'parameter_sha256':model_hash(model),'checkpoint_selection':'final_update_only'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
