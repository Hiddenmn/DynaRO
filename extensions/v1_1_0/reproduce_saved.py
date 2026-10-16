"""Reaggregate all saved pair records, without loading a model or scoring TEST."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from aggregation import mean_tree, numeric_mean, summarize, summarize_seed

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

def deviation(a,b):
    if isinstance(a,dict):
        if set(a) != set(b): raise ValueError('Aggregate key mismatch')
        return max((deviation(a[k],b[k]) for k in a), default=0)
    if isinstance(a,list):
        if len(a)!=len(b): raise ValueError('Aggregate length mismatch')
        return max((deviation(x,y) for x,y in zip(a,b)),default=0)
    if a is None or b is None:
        if a is not b: raise ValueError('Missing-value mismatch')
        return 0
    return abs(a-b)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=read(ROOT/'manifest.json')
    for rel,entry in manifest['files'].items():
        path=ROOT/rel
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('File differs: '+rel)
    plan=read(ROOT/'sameinfo/protocol.json'); endpoint_rows=[];errors=[]
    for split in ('dev','test'):
        for arm in ('explicit_contact','learned_contact'):
            for seed in (20260811,20260812,20260813):
                base=ROOT/f'sameinfo/results/{split}/{arm}/{seed}'
                saved=read(base/'summary.json')
                rebuilt=summarize(rows(base/'pairs.jsonl'),plan['evaluation'][split])
                errors.append(deviation(rebuilt,saved))
                metric=rebuilt['overall_equal_block_mean']['100'][arm]
                endpoint_rows.append({'study':'sameinfo','split':split,'model':arm,'seed':seed,
                    'core':'source','brier_51_100':metric['unseen']['site']})
    panel=rows(ROOT/'external/targets/pairs.jsonl')
    external={}
    for model in ('dynaro','static_front','coarse_front','rtu','fno'):
        external[model]={}
        for seed in (20260811,20260812,20260813):
            base=ROOT/f'external/results/{model}/{seed}'
            rebuilt=summarize_seed(rows(base/'pairs.jsonl'),panel)
            errors.append(deviation(rebuilt,read(base/'summary.json')))
            external[model][str(seed)]=rebuilt
            for core,metric in rebuilt['per_core_equal_block_mean'].items():
                endpoint_rows.append({'study':'external','split':'external','model':model,'seed':seed,
                    'core':core,'brier_51_100':metric['100']['unseen']['site']})
    if max(errors)>1e-12:
        raise ValueError('Saved aggregation differs: '+str(max(errors)))
    with (a.output/'primary_endpoints.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(endpoint_rows[0]));writer.writeheader();writer.writerows(endpoint_rows)
    (a.output/'external_summary.json').write_text(json.dumps(external,indent=2),encoding='utf-8')
    report={'verified_files':len(manifest['files']),'sameinfo_panels':12,'external_model_seed_panels':15,
        'maximum_aggregate_difference':max(errors),'model_forward_calls':0,'new_TEST_scores':0}
    (a.output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
