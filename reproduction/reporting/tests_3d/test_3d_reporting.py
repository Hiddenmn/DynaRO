"""Saved-case input and renderer checks; no model or simulator is executed."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reproduce_3d import verify_inputs, verify_versions, run, compare_quantification
from reporting import read_json

class Saved3DTests(unittest.TestCase):
    def test_input_manifest(self):
        manifest = verify_inputs(ROOT / "inputs_3d")
        self.assertEqual(len(manifest["files"]), 6)
        self.assertEqual(manifest["contract"]["crop_zyx"], [[32, 96]]*3)
        self.assertEqual(manifest["contract"]["initial_local_oil"], 35740)

    def test_surface_functions_are_exact_copies(self):
        source = (ROOT / "src/surface_renderer.py").read_text(encoding="utf-8")
        expected = read_json(ROOT / "provenance/surface_functions.json")
        found = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                found[node.name] = "\n".join(source.splitlines()[start-1:node.end_lineno])
        self.assertEqual(set(found), set(expected))
        self.assertEqual(found, expected)

    def test_recorded_runtime_versions(self):
        self.assertEqual(verify_versions()["numba"], "0.65.1")

    def test_missing_runtime_stops_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch("reproduce_3d.verify_versions", side_effect=RuntimeError("missing runtime")):
                with self.assertRaisesRegex(RuntimeError, "missing runtime"):
                    run(ROOT / "inputs_3d", output)
            self.assertFalse(output.exists())

    def test_existing_output_is_not_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(ROOT / "inputs_3d", Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_bad_quantification_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure12_central64_displacement_v2.time.csv"
            path.write_text("step,simulator_percent\n0,999\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "time source data disagree"):
                compare_quantification(ROOT / "inputs_3d", Path(directory))

    def test_geometry_and_contact_from_saved_arrays(self):
        import numpy as np
        from surface_renderer import make_camera, PALETTE
        from plot_central64_displacement import contact_followup, CROP
        with np.load(ROOT / "inputs_3d/dev_parent_prediction.npz", allow_pickle=False) as data:
            initial, truth, prob = data["initial_labels"], data["truth_labels_a"], data["prediction_a"]
        self.assertEqual(int((initial[CROP] >= 3).sum()), 35740)
        self.assertEqual(int(((initial >= 3) & (truth[-1] == 2))[CROP].sum()), 17087)
        rows = contact_followup(initial, truth, prob, ROOT / "inputs_3d/dev_parent_simulator.npz")
        self.assertEqual([r["simulator_baseline_oil_sites"] for r in rows], [3581, 26236, 1973])
        self.assertEqual([r["simulator_converted_sites"] for r in rows], [2628, 3258, 0])
        self.assertEqual(rows[-1]["prediction_expected_converted_sites"], 0.0)
        self.assertEqual(make_camera(1400, az=-125., el=24.)["width"], 1484)
        np.testing.assert_array_equal(PALETTE[7], np.asarray([117, 96, 168], np.float32)/255)

if __name__ == "__main__":
    unittest.main()
