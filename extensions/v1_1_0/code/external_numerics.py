from __future__ import annotations
import hashlib,time
import numpy as np
from aggregation import require

def removal_to_occupancy(first_removal, step):
    """Lossless target decoding on initial oil sites; 0 denotes no removal."""
    require(first_removal.dtype == np.uint8 and first_removal.ndim == 1
            and (not first_removal.size or int(first_removal.max()) <= 100),
            'Invalid first-removal vector')
    require(type(step) is int and 0 <= step <= 100, 'Invalid simulator step')
    return (first_removal == 0) | (first_removal > step)
def step_seed(stream, core_id, block_id, step):
    token = f'{stream}|{core_id}|{block_id}|{step}'
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big') % (2**32)

class RemovalRecorder:
    """Lossless irreversible label history on fixed initial oil support."""
    def __init__(self, initial, horizon=100):
        require(initial.ndim == 3 and initial.dtype == np.uint8 and initial.size > 0
                and initial.max() <= 7, 'Invalid initial eight-class grid')
        require(type(horizon) is int and 1 <= horizon <= 100, 'Unsupported horizon')
        self.initial = initial.copy()
        self.previous = initial.copy()
        self.indices = np.flatnonzero(initial.ravel() >= 3)
        self.classes = initial.ravel()[self.indices]
        self.removal = np.zeros(self.indices.size, np.uint8)
        self.displaced = np.zeros((horizon, 5), np.int32)
        self.horizon, self.step = horizon, 0

    def append(self, current, reported_counts=None):
        require(self.step < self.horizon and current.shape == self.initial.shape
                and current.dtype == np.uint8, 'Invalid state shape/type or extra step')
        before = self.previous
        changed = before != current
        require(np.all(~changed | ((before >= 3) & (current == 2))),
                'Only initial-oil-to-water transitions are permitted')
        counts = np.bincount(before[changed], minlength=8)[3:8].astype(np.int32)
        if reported_counts is not None:
            require(np.array_equal(counts, reported_counts), 'Simulator class counts disagree with state change')
        index = self.step + 1
        moved = (current.ravel()[self.indices] == 2) & (self.removal == 0)
        self.removal[moved] = index
        self.displaced[self.step] = counts
        decoded = self.initial.copy()
        decoded.ravel()[self.indices[self.removal != 0]] = 2
        require(np.array_equal(decoded, current), 'First-removal encoding is not lossless')
        self.previous = current.copy()
        self.step = index

    def finish(self):
        require(self.step == self.horizon, 'Partial trajectory cannot be marked complete')
        require(np.count_nonzero(self.removal) == int(self.displaced.sum()), 'Total removal count differs')
        return self.removal.copy(), self.displaced.copy()

def generate_compact(initial, schedule, stream, core_id, block_id, runtime, *, deadline=None, observe=None):
    require(1 <= len(schedule) <= 100 and all(a in runtime.actions.ACTION_LIBRARY for a in schedule),
            'Unsupported action sequence')
    source = runtime.simulator
    connected = source.connected_to_inlet(initial >= 2, inlet_axis='x').astype(np.uint8)
    pressure = source.axis_gradient(initial.shape, 'x')
    recorder = RemovalRecorder(initial, len(schedule))
    current = initial.copy()
    for step, action in enumerate(schedule, 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Generation deadline reached; no automatic retry')
        current, info = source.displacement_step(
            label=current, strategy=runtime.actions.ACTION_LIBRARY[action].source_strategy(),
            rng=np.random.default_rng(step_seed(stream, core_id, block_id, step)),
            pressure_weight=pressure, inlet_axis='x', base_probability=0.06)
        counts = [info[f'displaced_class_{cls}_{source.CLASS_NAMES[cls]}'] for cls in range(3, 8)]
        recorder.append(current, counts)
        if observe is not None:
            observe(step, current)  # Synthetic consistency tests only; not a model input.
    removal, counts = recorder.finish()
    return {'first_removal_step': removal, 'displaced_by_class': counts, 'connected_pore': connected,
            'action_index': np.asarray([runtime.actions.ACTION_ORDER.index(a) for a in schedule], np.int8)}

def validate_compact(initial, arrays, schedule):
    n = int(np.count_nonzero(initial >= 3))
    r, c, mask, actions = (arrays[k] for k in
                          ('first_removal_step', 'displaced_by_class', 'connected_pore', 'action_index'))
    horizon = len(schedule)
    require(r.shape == (n,) and r.dtype == np.uint8 and (not n or r.max() <= horizon), 'Invalid removal vector')
    require(c.shape == (horizon, 5) and c.dtype == np.int32 and np.all(c >= 0), 'Invalid count trajectory')
    require(mask.shape == initial.shape and mask.dtype == np.uint8 and np.all(mask <= 1)
            and not np.any(mask.astype(bool) & (initial < 2)), 'Invalid connected-pore mask')
    expected = np.asarray([int(a[1:]) - 1 for a in schedule], np.int8)
    require(actions.dtype == np.int8 and np.array_equal(actions, expected), 'Action indices differ')
    initial_classes = initial.ravel()[np.flatnonzero(initial.ravel() >= 3)]
    for step in range(1, horizon + 1):
        actual = np.bincount(initial_classes[r == step], minlength=8)[3:8]
        require(np.array_equal(actual, c[step - 1]), 'Removal vector/class count mismatch')

def model_inputs(initial, connected, schedule, runtime, device):
    import torch
    labels = torch.from_numpy(initial).to(device)
    connection = torch.from_numpy(connected.astype(bool)).to(device)
    inlet = torch.zeros_like(connection)
    inlet[:, :, 0] = connection[:, :, 0]
    actions = torch.from_numpy(np.stack([runtime.actions.action_vector(a) for a in schedule])).to(device)
    steps = torch.arange(1, len(schedule) + 1, dtype=torch.float32, device=device)
    return labels, inlet, connection, actions, (steps / (steps + 25)).unsqueeze(1)

def cpu_output(output):
    import torch
    return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}

def compact_targets(initial, removal_a, removal_b, output, runtime, horizon=100):
    import torch
    labels = torch.from_numpy(initial)
    sites, groups, count = runtime.readout.initial_oil_groups(labels, cell_width=16)
    inventory = torch.bincount(groups, minlength=count).float()
    require(torch.equal(output['site_index'], sites) and torch.equal(output['group_id'], groups)
            and torch.equal(output['initial_group_inventory'], inventory), 'Predicted support differs from initial semantics')
    require(sites.numel() > 0, 'No initial oil: report N/A, do not manufacture a score')
    target_sites, fractions = [], []
    for removal in (removal_a, removal_b):
        require(removal.shape == (sites.numel(),), 'Removal support differs')
        # Only initial-oil sites are materialized, never the full T x volume.
        array = np.stack([removal_to_occupancy(removal, t) for t in range(1, horizon + 1)])
        target = torch.from_numpy(array).float()
        values = torch.zeros((horizon, count), dtype=torch.float32)
        values.scatter_add_(1, groups.unsqueeze(0).expand(horizon, -1), target)
        target_sites.append(target)
        fractions.append(values / inventory.clamp_min(1))
    return runtime.objective.PairWindowTargets(*target_sites, *fractions, inventory > 0)

def prefix_output(output, horizon):
    return {k: v[:horizon] if k in ('remaining_sites', 'group_remaining') else v for k, v in output.items()}

def check_output(output, initial, horizon):
    import torch
    values = output['remaining_sites']
    require(output['step_offset'] == 0 and values.shape == (horizon, int(np.count_nonzero(initial >= 3)))
            and values.dtype == torch.float32, 'Invalid prediction size/precision')
    require(bool(torch.isfinite(values).all()) and bool(((values >= -1e-7) & (values <= 1 + 1e-7)).all()),
            'Nonfinite or out-of-bounds predicted occupancy')
    require(not bool((values[1:] - values[:-1] > 1e-6).any()), 'Oil occupancy increased unexpectedly')
    require(bool(torch.isfinite(output['group_remaining']).all()), 'Nonfinite predicted inventories')

def score_compact_pair(initial, outputs, removals, runtime):
    a, b = outputs
    for out in outputs:
        check_output(out, initial, 100)
    targets = compact_targets(initial, *removals, a, runtime)
    scores = runtime.long.score_outputs(a, b, targets)
    short_targets = runtime.objective.PairWindowTargets(
        targets.site_a[:50], targets.site_b[:50], targets.region_a[:50], targets.region_b[:50], targets.nonempty_group)
    scores['50'] = runtime.short._score_outputs(prefix_output(a, 50), prefix_output(b, 50), short_targets, 50)
    return scores

def pair_panel(plan):
    rows = []
    for crop in plan['crops']:
        for stream in plan['noise']['streams']:
            for action in plan['actions']:
                pair_id = f"{crop['core_id']}|{crop['block_id']}|{stream}|{action['a']}~{action['b']}"
                token = hashlib.sha256(pair_id.encode()).hexdigest()[:24]
                rows.append({'pair_id': pair_id, 'core_id': crop['core_id'], 'block_id': crop['block_id'],
                             'label_file': crop['file'], 'stream_id': stream,
                             'schedules': action['sequences'], 'target_files': [f'{token}_{i}.npz' for i in range(2)]})
    require(len(rows) == plan['counts']['pairs'] and len({r['pair_id'] for r in rows}) == len(rows),
            'External pair count/identity mismatch')
    return rows
