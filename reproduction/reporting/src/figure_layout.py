"""Project-specific layout checks for the fixed DynaRO figures.

This small checker measures the existing axes without moving or redrawing any
scientific element. It is independent of the external figure-tooling checker.
An optional trusted QA module can perform an additional pre-export inspection.
"""
from __future__ import annotations
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys


def check_layout(fig, *, axes=None, panel_ids=None, exclude_axes=(),
                 row_groups=None, column_groups=None, tolerance_pt=1.5,
                 gutter_tolerance_pt=1.5, require_panel_labels=False,
                 strict=True, json_out=None, overlay_svg=None):
    """Check fixed figure groups in points; raise before export on a mismatch."""
    if tolerance_pt <= 0 or gutter_tolerance_pt <= 0:
        raise ValueError("Layout tolerances must be positive")
    fig.canvas.draw()
    selected = [a for a in (fig.axes if axes is None else axes)
                if a not in exclude_axes and a.get_visible()]
    ids = list(panel_ids) if panel_ids is not None else [
        chr(ord('a') + i) for i in range(len(selected))]
    if not selected or len(ids) != len(selected) or len(set(ids)) != len(ids):
        raise ValueError("Invalid layout panel identities")
    scale = 72.0 / fig.dpi
    rects, labels, spans = {}, {}, {}
    for name, ax in zip(ids, selected):
        box = ax.get_window_extent()
        rect = [float(v) * scale for v in (box.x0, box.y0, box.x1, box.y1)]
        if not all(math.isfinite(v) for v in rect) or rect[2] <= rect[0] or rect[3] <= rect[1]:
            raise ValueError("Invalid rendered plot rectangle")
        rects[name] = rect
        candidates = [t for t in list(ax.texts) + list(fig.texts)
                      if t.get_visible() and t.get_text().strip() == name]
        if require_panel_labels and len(candidates) != 1:
            raise ValueError(f"Panel {name} must have one unambiguous label")
        if len(candidates) == 1:
            t = candidates[0]
            labels[name] = [float(v) * scale for v in t.get_transform().transform(t.get_position())]
        spec = ax.get_subplotspec() if hasattr(ax, 'get_subplotspec') else None
        if spec is not None:
            spans[name] = (id(spec.get_gridspec()), spec.rowspan.start, spec.rowspan.stop,
                           spec.colspan.start, spec.colspan.stop)

    def infer_groups(start, stop):
        groups = {}
        for name, span in spans.items():
            groups.setdefault((span[0], span[start], span[stop]), []).append(name)
        return [v for v in groups.values() if len(v) > 1]

    rows = infer_groups(1, 2) if row_groups is None else list(row_groups)
    columns = infer_groups(3, 4) if column_groups is None else list(column_groups)
    checks, failures = [], []

    def spread(names, values, kind, tolerance=tolerance_pt):
        if len(values) < 2:
            return
        error = float(max(values) - min(values))
        record = {"panels": list(names), "check": kind, "spread_pt": error,
                  "tolerance_pt": float(tolerance), "pass": bool(error <= tolerance)}
        checks.append(record)
        if not record['pass']:
            failures.append(record)

    for direction, groups in (("row", rows), ("column", columns)):
        for group in groups:
            if len(group) < 2 or len(set(group)) != len(group) or any(n not in rects for n in group):
                raise ValueError("Invalid comparable panel group")
            axis = 0 if direction == 'row' else 1
            order = sorted(group, key=lambda n: rects[n][axis])
            edges = (1, 3) if direction == 'row' else (0, 2)
            for edge in edges:
                spread(order, [rects[n][edge] for n in order], f'{direction}-edge-{edge}')
            if direction == 'row' and len(order) >= 3:
                spread(order, [rects[n][2] - rects[n][0] for n in order], 'row-width')
            if len(order) >= 3:
                gaps = [rects[b][axis] - rects[a][axis + 2] for a, b in zip(order, order[1:])]
                spread(order, gaps, f'{direction}-gutters', gutter_tolerance_pt)
            if all(n in labels for n in order):
                label_axis = 1 if direction == 'row' else 0
                spread(order, [labels[n][label_axis] for n in order], f'{direction}-label-anchor')

    # Match shared outer edges in unequal-span grids, without equating a hero
    # panel's height or width to the smaller panels that border it.
    for a, b in itertools.combinations(spans, 2):
        sa, sb = spans[a], spans[b]
        if sa[0] != sb[0]:
            continue
        for slot, edge in ((1, 3), (2, 1), (3, 0), (4, 2)):
            if sa[slot] == sb[slot]:
                spread([a, b], [rects[a][edge], rects[b][edge]], f'grid-boundary-{edge}')
    for a, b in itertools.combinations(ids, 2):
        ra, rb = rects[a], rects[b]
        dx = min(ra[2], rb[2]) - max(ra[0], rb[0])
        dy = min(ra[3], rb[3]) - max(ra[1], rb[1])
        if dx > 0.01 and dy > 0.01:
            failures.append({"panels": [a, b], "check": "plot-area-overlap"})
    if len(ids) > 1 and not checks:
        raise ValueError("Comparable groups are required for this fixed figure")
    report = {"schema": "dynaro-fixed-figure-layout-v1", "size_pt": [
        float(v) * 72 for v in fig.get_size_inches()], "rectangles_pt": rects,
        "label_anchors_pt": labels, "checks": checks, "failures": failures,
        "status": "FAIL" if failures else ("NOT_APPLICABLE" if len(ids) == 1 else "PASS")}
    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if failures:
        raise ValueError(f"Figure layout has {len(failures)} failed checks")

    # This optional independent QA tool is not bundled or required to render.
    # Only execute a module explicitly selected by the person running the command.
    extra = os.environ.get('DYNARO_FIGURE_QA')
    if extra:
        path = Path(extra).resolve(strict=True)
        spec = importlib.util.spec_from_file_location('_dynaro_external_figure_qa', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        qa_json = Path(json_out).with_suffix('.external.json') if json_out else None
        qa_svg = Path(overlay_svg).with_suffix('.external.svg') if overlay_svg else None
        module.require_matplotlib_panel_alignment(
            fig, axes=selected, panel_ids=ids, row_groups=row_groups, column_groups=column_groups,
            tolerance_pt=tolerance_pt, gutter_tolerance_pt=gutter_tolerance_pt,
            require_panel_labels=require_panel_labels, strict=strict,
            json_out=qa_json, overlay_svg=qa_svg)
    return report
