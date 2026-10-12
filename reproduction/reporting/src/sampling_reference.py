"""Preserved surface/point functions used only for the full-stack overview.
No original loaders or project paths are executed. Display coordinates are not
calibrated physical units. See provenance/sampling_functions.json.
"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

REFERENCE = Path(__file__).with_name("sampling_surface_functions.py")
REFERENCE_SHA256 = "e61d0129189601a560835173d97be888c154be5f5dd30be51555cb97810f3653"
FUNCTIONS_SHA256 = "f91589f207260c11381887136acda86ef1bba99969515d0426be349b1d9ba81a"
PALETTE = {1: "#969696", 2: "#56B4E9", 3: "#E6AD15", 4: "#009E73",
           5: "#CC79A7", 6: "#D55E00", 7: "#7560A8"}
NAMES = {1: "Rock", 2: "Water", 3: "Film oil", 4: "Corner oil",
         5: "Droplet oil", 6: "Columnar oil", 7: "Cluster oil"}
MARKERS = {1: "s", 2: "X", 3: "o", 4: "^", 5: "s", 6: "D", 7: "P"}
BACKGROUND = "#F5F6F8"
OIL_ALPHA = .94
OIL_POINT_ALPHA = .35

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def original_functions(spacing, surface_coarsen=2):
    if digest(REFERENCE) != FUNCTIONS_SHA256:
        raise ValueError("The preserved source differs from the user-provided renderer")
    wanted = {"deterministic_points", "add_class_layer", "add_surface_layer", "box_edges_xyz"}
    parsed = ast.parse(REFERENCE.read_text(encoding="utf-8"))
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise ValueError("Missing original drawing function")
    ns = {"np": np, "mpl": mpl, "plt": plt, "Poly3DCollection": Poly3DCollection,
          "marching_cubes": marching_cubes, "CLASS_COLORS": PALETTE,
          "CLASS_MARKERS": MARKERS, "POINT_HASH_MODULUS": 3,
          "SURFACE_GRID_COARSEN": surface_coarsen, "DISPLAY_SPACING_MM_ZYX": np.asarray(spacing, float),
          "SURFACE_ALPHA": OIL_ALPHA, "POINT_ALPHA": OIL_POINT_ALPHA, "POINT_SIZE": .50,
          "LIGHT_AZIMUTH": 315, "LIGHT_ALTITUDE": 40}
    # The original variable name contains "MM"; these replacement values
    # are image/model-index display units, not a physical calibration.
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "original_user_render_volume_functions", "exec"), ns)
    return ns
