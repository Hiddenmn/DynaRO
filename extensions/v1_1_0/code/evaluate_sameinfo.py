"""Evaluate a frozen contact-control state on the fixed DEV or TEST panel.

The default only reports panel metadata. Model inference requires --execute,
the original trajectory tree and a new output directory. Published results
are already supplied; this entry never writes to their directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'code/model_core/src'),
               str(ROOT/'code/sameinfo')]

def prefix50(pair):
    labels, inlet, connected, aa, ab, times, ta, tb = pair
    if len(aa) != 100 or ta.shape[0] != 101:
        raise ValueError("Expected a full existing100-step trajectory")
    return labels, inlet, connected, aa[:50], ab[:50], times[:50], ta[:51], tb[:51]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('explicit_contact','learned_contact'), required=True)
    parser.add_argument('--seed', type=int, choices=(20260811,20260812,20260813), required=True)
    parser.add_argument('--split', choices=('dev','test'), required=True)
    parser.add_argument('--data-root', type=Path,
        help='Root of the original workflow_full_audit_20260717 trajectory tree')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pair-id', help='An exact identity; omit for the complete panel')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    protocol = json.loads((ROOT/'sameinfo/protocol.json').read_text(encoding='utf-8'))
    panel = protocol['evaluation'][args.split]
    records = [r for r in panel['pairs'] if args.pair_id is None or r['pair_id']==args.pair_id]
    if not records:
        raise ValueError('Unknown pair identity')
    if not args.execute:
        print(json.dumps({'split':args.split, 'arm':args.arm, 'seed':args.seed,
            'pairs':len(records), 'horizons':[25,50,75,100], 'inference_started':False}))
        return
    if args.data_root is None or args.output is None or args.output.exists():
        raise ValueError('Provide the original data root and a new output directory')
    base = args.data_root.resolve(strict=True)
    allowed = {}
    for record in records:
        for horizon in ('25','50','100'):
            for entry in record.get(horizon, []):
                path = (base/entry['path']).resolve(strict=True)
                if not path.is_relative_to(base) or path.is_symlink():
                    raise ValueError('Trajectory lies outside the data root')
                if path in allowed:
                    if allowed[path]!=entry:
                        raise ValueError('Conflicting trajectory identity')
                    continue
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024), b''):
                        digest.update(chunk)
                if path.stat().st_size!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:
                    raise ValueError('Trajectory differs: '+entry['path'])
                allowed[path] = entry
    import torch
    import portable_train_data as data
    from sameinfo_runtime import load_model
    from scoring import short, long
    from aggregation import summarize
    package = ModuleType('patent22')
    package.__path__ = [str(ROOT/'code')]
    sys.modules['patent22'] = package
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model = load_model(args.arm, args.seed).to(device)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for record in records:
        initial, scores = None, {}
        # DEV50 is a separate equal-dose trajectory; TEST50 is the prefix of TEST100.
        for horizon in ((25,50) if args.split=='dev' else (25,)):
            short_record = {horizon:tuple((base/e['path'],e['record'])
                                         for e in record[str(horizon)])}
            pair = data.load_pair(short_record, horizon, device)
            if initial is not None and not torch.equal(initial,pair[0].cpu()):
                raise ValueError('Short-horizon initial states differ')
            initial = pair[0].cpu()
            scores[str(horizon)] = short._score_pair({args.arm:model},pair,horizon)
            del pair
        long_record = {**record, 'files':[base/e['path'] for e in record['100']]}
        pair, info = long.read_pair(long_record, device)
        if not torch.equal(initial,pair[0].cpu()) or info['input_sha256']!=[e['sha256'] for e in record['100']]:
            raise ValueError('Long-horizon inputs differ')
        if args.split=='test':
            scores['50'] = short._score_pair({args.arm:model},prefix50(pair),50)
        scores.update(long.score_pair({args.arm:model},pair))
        row = {k:record[k] for k in ('pair_id','block_id','stage','stream_id')}
        row.update(scores=scores, split=args.split, seed=args.seed, optimizer_updates=0)
        rows.append(row)
        with (args.output/'pairs.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'completed':len(rows),'pairs':len(records)}),flush=True)
        del pair, initial
        if device.type=='cuda':
            torch.cuda.empty_cache()
    if args.pair_id is None:
        (args.output/'summary.json').write_text(json.dumps(summarize(rows,panel),indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
