"""Numerical definitions extracted unchanged from the frozen displacement source."""
from typing import Any, Dict, Tuple
import numpy as np
from scipy import ndimage as ndi
SCIPY_AVAILABLE = True

CLASS_NAMES = {
    0: "background",
    1: "rock",
    2: "water",
    3: "film_oil",
    4: "corner_oil",
    5: "droplet_oil",
    6: "columnar_oil",
    7: "cluster_oil",
}


OIL_CLASSES = [3, 4, 5, 6, 7]


BASE_CLASS_DIFFICULTY = {
    3: 1.55,
    4: 1.40,
    5: 1.05,
    6: 0.95,
    7: 0.85,
}


def neighbor_any(mask: np.ndarray) -> np.ndarray:
    """6邻域膨胀，不依赖scipy。"""
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def connected_to_inlet(pore_mask: np.ndarray, inlet_axis: str = "x") -> np.ndarray:
    inlet = np.zeros_like(pore_mask, dtype=bool)
    if inlet_axis == "x":
        inlet[:, :, 0] = pore_mask[:, :, 0]
    elif inlet_axis == "y":
        inlet[:, 0, :] = pore_mask[:, 0, :]
    elif inlet_axis == "z":
        inlet[0, :, :] = pore_mask[0, :, :]
    else:
        raise ValueError("inlet_axis必须是x/y/z")

    if SCIPY_AVAILABLE:
        structure = np.zeros((3, 3, 3), dtype=np.uint8)
        structure[1, 1, 0] = structure[1, 1, 2] = 1
        structure[1, 0, 1] = structure[1, 2, 1] = 1
        structure[0, 1, 1] = structure[2, 1, 1] = 1
        structure[1, 1, 1] = 1
        labels, _ = ndi.label(pore_mask, structure=structure)
        inlet_labels = np.unique(labels[inlet])
        inlet_labels = inlet_labels[inlet_labels > 0]
        if len(inlet_labels) == 0:
            return np.zeros_like(pore_mask, dtype=bool)
        return np.isin(labels, inlet_labels)

    reached = inlet.copy()
    for _ in range(sum(pore_mask.shape) + 5):
        new_reached = (neighbor_any(reached) | reached) & pore_mask
        if np.array_equal(new_reached, reached):
            break
        reached = new_reached
    return reached


def axis_gradient(shape: Tuple[int, int, int], axis: str) -> np.ndarray:
    z, y, x = shape
    if axis == "x":
        coord = np.linspace(1.0, 0.2, x, dtype=np.float32)[None, None, :]
        return np.broadcast_to(coord, shape)
    if axis == "y":
        coord = np.linspace(1.0, 0.2, y, dtype=np.float32)[None, :, None]
        return np.broadcast_to(coord, shape)
    if axis == "z":
        coord = np.linspace(1.0, 0.2, z, dtype=np.float32)[:, None, None]
        return np.broadcast_to(coord, shape)
    raise ValueError("axis must be x/y/z")


def compute_basic_metrics(label_vol: np.ndarray) -> Dict[str, Any]:
    counts = np.bincount(np.clip(label_vol.astype(np.int64), 0, 7).ravel(), minlength=8)
    total = int(counts.sum())
    background = int(counts[0])
    rock = int(counts[1])
    water = int(counts[2])
    oil_total = int(counts[3:8].sum())
    valid_core = total - background
    pore = water + oil_total

    porosity = pore / valid_core if valid_core > 0 else np.nan
    oil_sat = oil_total / pore if pore > 0 else np.nan
    water_sat = water / pore if pore > 0 else np.nan

    m = {
        "total_voxels": total,
        "background_voxels": background,
        "valid_core_voxels": valid_core,
        "rock_voxels": rock,
        "pore_voxels": int(pore),
        "water_voxels": water,
        "oil_voxels": oil_total,
        "porosity_excluding_background": float(porosity),
        "oil_saturation_in_pore": float(oil_sat),
        "water_saturation_in_pore": float(water_sat),
    }
    for k in range(8):
        m[f"class_{k}_{CLASS_NAMES[k]}_voxels"] = int(counts[k])
    return m


def strategy_adjusted_difficulty(cls: int, strategy: Dict[str, Any]) -> float:
    base = BASE_CLASS_DIFFICULTY[cls]
    wet = float(strategy.get("wettability_shift", 0.0))
    if cls == 3:
        base *= (1.0 - 0.55 * wet)
    elif cls == 4:
        base *= (1.0 - 0.45 * wet)
    elif cls == 5:
        base *= (1.0 - 0.20 * wet)
    elif cls == 6:
        base *= (1.0 - 0.15 * wet)
    elif cls == 7:
        base *= (1.0 - 0.10 * wet)
    return max(base, 0.35)


def estimate_pressure_drop(label: np.ndarray, strategy: Dict[str, Any]) -> float:
    m = compute_basic_metrics(label)
    phi = max(float(m["porosity_excluding_background"]), 1e-4)
    sor = max(float(m["oil_saturation_in_pore"]), 0.0)
    rate = float(strategy["rate_factor"])
    k_proxy = (phi ** 3) / ((1.0 - phi) ** 2 + 1e-6)
    mobility_penalty = 1.0 + 1.8 * sor
    return float(rate * mobility_penalty / (k_proxy + 1e-6))


def displacement_step(
    label: np.ndarray,
    strategy: Dict[str, Any],
    rng: np.random.Generator,
    pressure_weight: np.ndarray,
    inlet_axis: str,
    base_probability: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    new_label = label.copy()

    pore_mask = (label != 0) & (label != 1)
    oil_mask = label >= 3
    water_mask = label == 2

    inlet_water = np.zeros_like(water_mask, dtype=bool)
    if inlet_axis == "x":
        inlet_water[:, :, 0] = pore_mask[:, :, 0]
    elif inlet_axis == "y":
        inlet_water[:, 0, :] = pore_mask[:, 0, :]
    else:
        inlet_water[0, :, :] = pore_mask[0, :, :]

    water_mask = water_mask | inlet_water

    conn = connected_to_inlet(pore_mask, inlet_axis=inlet_axis)
    water_mask = water_mask & conn

    front = neighbor_any(water_mask) & oil_mask & conn

    rate = float(strategy["rate_factor"])
    conc = float(strategy.get("concentration", 0.0))
    wet = float(strategy.get("wettability_shift", 0.0))

    displaced_total = 0
    displaced_by_class = {k: 0 for k in OIL_CLASSES}
    front_by_class = {k: int(np.sum(front & (label == k))) for k in OIL_CLASSES}

    random_field = rng.random(label.shape)

    for cls in OIL_CLASSES:
        cls_front = front & (label == cls)
        if int(cls_front.sum()) == 0:
            continue

        difficulty = strategy_adjusted_difficulty(cls, strategy)
        fluid_boost = 1.0 + 0.6 * conc + 0.7 * wet
        p = base_probability * rate * fluid_boost * pressure_weight / difficulty

        if strategy["fluid_type"] == "water" and rate > 1.2 and cls in [6, 7]:
            p = p * 1.15
        if strategy["fluid_type"] == "water" and rate > 1.2 and cls in [3, 4]:
            p = p * 0.90

        p = np.clip(p, 0.0, 0.55)
        move = cls_front & (random_field < p)
        moved = int(move.sum())
        if moved:
            new_label[move] = 2
            displaced_total += moved
            displaced_by_class[cls] += moved

    info = {
        "front_voxels": int(front.sum()),
        "displaced_voxels": int(displaced_total),
    }
    for cls in OIL_CLASSES:
        info[f"front_class_{cls}_{CLASS_NAMES[cls]}"] = int(front_by_class[cls])
        info[f"displaced_class_{cls}_{CLASS_NAMES[cls]}"] = int(displaced_by_class[cls])
    return new_label, info
