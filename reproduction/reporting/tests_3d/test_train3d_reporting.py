"""Checks for the fixed TRAIN crop; no model or simulator is executed."""
import ast
import csv
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reproduce_train3d import STEM, SELECTION, verify_inputs, compare_quantification, run
from reporting import read_json

class Training3DTests(unittest.TestCase):
    def test_fixed_selection_manifest(self):
        manifest = verify_inputs(ROOT / "inputs_train3d")
        self.assertEqual(len(manifest["files"]), 4)
        self.assertEqual(manifest["contract"]["crop_start_zyx"], [40, 16, 16])
        self.assertEqual(manifest["contract"]["remaining_counts"], [18396, 17780, 14880])
        selection = read_json(ROOT / "inputs_train3d" / SELECTION)
        self.assertEqual(selection["candidate_count"], 1117)
        self.assertEqual(selection["source_count"], 40)
        self.assertEqual(selection["minimum_parent_face_margin"], 16)

    def test_numerical_and_drawing_helpers_are_unchanged(self):
        expected = read_json(ROOT / "provenance/train_functions.json")
        actual = {}
        for name in ("plot_train50_displacement.py", "train_counts.py"):
            source = (ROOT / "src" / name).read_text(encoding="utf-8")
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef) and node.name in expected:
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    actual[node.name] = "\n".join(source.splitlines()[start-1:node.end_lineno])
        self.assertEqual(actual, expected)

    def test_all_voxel_counts_and_irreversibility(self):
        import numpy as np
        from plot_train50_displacement import selected_crop, camera
        cube, rows, record, measures = selected_crop(ROOT / "inputs_train3d" / SELECTION)
        self.assertEqual(cube.shape, (51, 64, 64, 64))
        self.assertEqual(cube.dtype, np.uint8)
        self.assertEqual(measures, ("S3", "S5"))
        self.assertEqual(record["crop_start_zyx"], [40, 16, 16])
        self.assertEqual([rows[t]["remaining_voxels"] for t in (0, 20, 50)], [18396, 17780, 14880])
        self.assertEqual(rows[-1]["converted_voxels"], 3516)
        initial_water = int((cube[0] == 2).sum())
        self.assertEqual(initial_water, 29259)
        for step, row in enumerate(rows):
            self.assertEqual(row["step"], step)
            self.assertEqual(row["remaining_voxels"], int((cube[step] >= 3).sum()))
            self.assertEqual(row["converted_voxels"], int((cube[step] == 2).sum()) - initial_water)
            self.assertEqual(row["remaining_voxels"] + row["converted_voxels"], 18396)
            self.assertEqual(row["remaining_fraction"], row["remaining_voxels"] / 18396)
            np.testing.assert_array_equal(cube[step] == 1, cube[0] == 1)
            self.assertTrue(np.all(cube[step][cube[0] == 2] == 2))
            if step:
                self.assertLessEqual(row["remaining_voxels"], rows[step-1]["remaining_voxels"])
        self.assertEqual(camera(1400)["center"].tolist(), [47.5, 47.5, 71.5])
        self.assertFalse(np.any(cube == 0))

    def test_legacy_crop_cannot_be_used_by_default(self):
        import plot_train50_displacement as figure
        self.assertEqual(figure.STEM, STEM)
        self.assertIs(inspect.signature(figure.render).parameters["selection_path"].default,
                      inspect.Parameter.empty)

    def test_changed_count_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / f"{STEM}.source.csv").write_text(
                "step,remaining_voxels\n0,999\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Training source counts disagree"):
                compare_quantification(ROOT / "inputs_train3d", output)

    def test_existing_output_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(ROOT / "inputs_train3d", Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_runtime_stops_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch("reproduce_train3d.verify_versions", side_effect=RuntimeError("missing runtime")):
                with self.assertRaisesRegex(RuntimeError, "missing runtime"):
                    run(ROOT / "inputs_train3d", output)
            self.assertFalse(output.exists())

if __name__ == "__main__":
    unittest.main()
