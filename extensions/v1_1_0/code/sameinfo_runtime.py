"""Load the six frozen contact-control models; optional explicit inference CLI."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/model_core/src'))
sys.path.insert(0, str(ROOT / 'code/sameinfo'))

def state_hash(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_model(arm, seed):
    import torch
    from sameinfo_model import SameInformationContactSurrogate
    manifest = json.loads((ROOT / 'sameinfo/weights/manifest.json').read_text(encoding='utf-8'))
    entry = manifest[f'{arm}/{seed}']
    path = ROOT / entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Weight checksum differs')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state_hash(state) != entry['original_parameter_sha256']:
        raise ValueError('Tensor values differ from the experiment')
    with torch.random.fork_rng(devices=[]):
        model = SameInformationContactSurrogate(arm)
    model.load_state_dict(state, strict=True)
    if sum(p.numel() for p in model.parameters()) != 91936:
        raise ValueError('Unexpected model capacity')
    return model.eval().requires_grad_(False)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--arm', choices=('explicit_contact', 'learned_contact'))
    p.add_argument('--seed', type=int, choices=(20260811, 20260812, 20260813))
    p.add_argument('--input', type=Path, help='128-cubed uint8 eight-class .npy')
    p.add_argument('--actions', help='Comma-separated actions, one S1-S5 token per step')
    p.add_argument('--output', type=Path, help='A new .npz file, never overwritten')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    if a.verify_only:
        for arm in ('explicit_contact', 'learned_contact'):
            for seed in (20260811, 20260812, 20260813):
                load_model(arm, seed)
        print(json.dumps({'models_loaded': 6, 'forward_calls': 0, 'tensor_hashes_match': True}))
        return
    if None in (a.arm, a.seed, a.input, a.actions, a.output):
        p.error('Inference requires arm, seed, input, actions and output')
    if a.output.exists():
        raise FileExistsError(a.output)
    import numpy as np
    from scipy.ndimage import label
    from portable_train_data import ACTION_VECTORS
    initial = np.load(a.input, allow_pickle=False)
    if initial.shape != (128,)*3 or initial.dtype != np.uint8 or initial.max() > 7:
        raise ValueError('Expected a 128-cubed eight-class uint8 volume')
    sequence = a.actions.split(',')
    if not 1 <= len(sequence) <= 100 or any(x not in ('S1','S2','S3','S4','S5') for x in sequence):
        raise ValueError('Expected 1-100 S1-S5 actions')
    # Six-neighbour connectivity from the native x=0 inlet, as in the generator.
    groups, _ = label(initial >= 2)
    touched = np.unique(groups[:, :, 0]); touched = touched[touched > 0]
    connected = np.isin(groups, touched)
    device = torch.device(a.device)
    labels = torch.from_numpy(initial).to(device)
    connection = torch.from_numpy(connected).to(device)
    inlet = torch.zeros_like(connection); inlet[:, :, 0] = connection[:, :, 0]
    controls = torch.from_numpy(ACTION_VECTORS[[int(x[1:])-1 for x in sequence]].copy()).to(device)
    steps = torch.arange(1,len(sequence)+1,device=device,dtype=torch.float32)
    model = load_model(a.arm,a.seed).to(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        output = model(labels,inlet,connection,controls,(steps/(steps+25)).unsqueeze(1),compact_output=True)
    arrays = {k:v.cpu().numpy() for k,v in output.items() if isinstance(v,torch.Tensor)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    print(json.dumps({'steps':len(sequence),'arm':a.arm,'seed':a.seed,'output':str(a.output)}))

if __name__ == '__main__':
    main()
