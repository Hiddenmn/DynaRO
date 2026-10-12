"""Exact voxel-count readout, with no surface-derived statistics."""
import numpy as np

def count_rows(cube: np.ndarray) -> list[dict]:
    initial = cube[0] >= 3
    n = int(initial.sum())
    return [dict(step=t, remaining_voxels=int((initial & (cube[t] >= 3)).sum()),
                 converted_voxels=int((initial & (cube[t] == 2)).sum()),
                 remaining_fraction=float((initial & (cube[t] >= 3)).sum() / n))
            for t in range(51)]

