"""Regression checks for the unchanged full-parent figure and source exports."""
import ast
import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reproduce_parent3d import (
    STEM, EXPECTED, verify_inputs, compare_quantification, write_source_data, run)
from reporting import read_json

class Parent3DTests(unittest.TestCase):
    def test_shared_case_matches_central_crop(self):
        manifest, expected = verify_inputs(ROOT / "inputs_3d", ROOT / "inputs_parent3d")
        self.assertEqual(manifest["figure"], "supplementary Figure S2")
        self.assertEqual(expected["shape_zyx"], [128]*3)
        self.assertEqual(expected["camera"], {
            "azimuth": -125., "elevation": 24., "projection": "orthographic"})
        self.assertEqual(expected["display_threshold"], 0.5)
        self.assertFalse(expected["additional_crop"])
        self.assertFalse(expected["selection"]["truth_only_selection"]["case_is_representative"])

    def test_helpers_are_exact_source_copies(self):
        source = (ROOT / "src/parent_volume_renderer.py").read_text(encoding="utf-8")
        expected = read_json(ROOT / "provenance/parent_surface_functions.json")
        found = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                found[node.name] = "\n".join(source.splitlines()[start-1:node.end_lineno])
        self.assertEqual(found, expected)

    def test_saved_voxels_counts_and_brier_agree(self):
        import numpy as np
        with np.load(ROOT / "inputs_3d/dev_parent_prediction.npz", allow_pickle=False) as data:
            initial = data["initial_labels"]
            truth, probability = data["truth_labels_a"], data["prediction_a"]
            true_counts, predicted_counts = data["true_class_counts_a"], data["predicted_class_counts_a"]
            brier, steps = data["site_brier_by_step_a"], data["steps"]
        oil = initial >= 3
        self.assertEqual(int(oil.sum()), 295851)
        self.assertEqual(int(((truth[-1] == 2) & oil).sum()), 93281)
        self.assertEqual(int(oil[32:96, 32:96, 32:96].sum()), 35740)
        self.assertFalse(np.any(probability[:, ~oil]))
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
        for i, step in enumerate(steps):
            np.testing.assert_array_equal(
                [int((truth[i] == cls).sum()) for cls in range(3, 8)], true_counts[step])
            # The saved exporter sums float32 regional inventories; this check
            # instead sums saved site probabilities in float64. Allow their
            # reduction-roundoff difference, while the source hash and figure
            # metadata comparisons continue to require exact equality.
            np.testing.assert_allclose(
                [probability[i][initial == cls].sum(dtype=np.float64) for cls in range(3, 8)],
                predicted_counts[step], rtol=1e-6, atol=1e-5)
            if step:
                rebuilt = np.mean(
                    (probability[i][oil].astype(np.float64) - (truth[i][oil] >= 3))**2)
                self.assertAlmostEqual(float(rebuilt), float(brier[step-1]), places=7)

    def test_changed_curve_is_rejected(self):
        expected = read_json(ROOT / "inputs_parent3d" / EXPECTED)
        actual = copy.deepcopy(expected)
        actual["curve"]["simulator_percent"][50] += 0.01
        with self.assertRaisesRegex(ValueError, "differs: curve"):
            compare_quantification(actual, expected)

    def test_source_exports_keep_all_points_and_units(self):
        data = read_json(ROOT / "inputs_parent3d" / EXPECTED)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_source_data(data, output)
            for kind, n in (("time", 101), ("depth", 8), ("brier", 100)):
                with (output / f"{STEM}.{kind}.csv").open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), n)
                if kind == "brier":
                    for i, row in enumerate(rows):
                        self.assertEqual(int(row["step"]), i+1)
                        self.assertEqual(float(row["site_brier_raw"]),
                                         data["curve"]["site_brier_steps1_100"][i])
                        self.assertEqual(float(row["site_brier_display_x1000"]),
                                         1000 * float(row["site_brier_raw"]))
                if kind == "depth":
                    self.assertEqual(sum(int(row["initial_oil_voxels"]) for row in rows), 295851)

    def test_existing_output_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(ROOT / "inputs_3d", ROOT / "inputs_parent3d", Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_runtime_stops_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch("reproduce_parent3d.verify_versions", side_effect=RuntimeError("missing runtime")):
                with self.assertRaisesRegex(RuntimeError, "missing runtime"):
                    run(ROOT / "inputs_3d", ROOT / "inputs_parent3d", output)
            self.assertFalse(output.exists())

if __name__ == "__main__":
    unittest.main()
