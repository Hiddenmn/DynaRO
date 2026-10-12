"""Fixed stack-to-cube checks; no sampling search or inference."""
import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reproduce_sampling import STEM, FULL, SLICE, EXPECTED, CASE, verify_inputs, compare_metadata, run
from reporting import read_json, digest

class SamplingTests(unittest.TestCase):
    def test_fixed_manifest(self):
        manifest = verify_inputs(ROOT / "inputs_sampling", ROOT / "inputs_3d")
        self.assertEqual(len(manifest["files"]), 3)
        self.assertEqual(len(manifest["shared_files"]), 1)
        self.assertEqual(manifest["contract"]["full_shape_zyx"], [845, 512, 512])
        self.assertFalse(manifest["contract"]["physical_units_calibrated"])

    def test_exact_array_mapping_and_source_image_size(self):
        import numpy as np
        from PIL import Image
        full = np.load(ROOT / "inputs_sampling" / FULL, mmap_mode="r", allow_pickle=False)
        with np.load(ROOT / "inputs_3d" / CASE, allow_pickle=False) as saved:
            parent = saved["initial_labels"]
        self.assertEqual(full.shape, (845, 512, 512))
        self.assertEqual(full.dtype, np.uint8)
        self.assertEqual(parent.shape, (128,)*3)
        np.testing.assert_array_equal(full[:128, 256:384, 256:384], parent)
        np.testing.assert_array_equal(full[32:96, 288:352, 288:352], parent[32:96, 32:96, 32:96])
        self.assertEqual(int((parent >= 3).sum()), 295851)
        self.assertEqual(int((parent[32:96, 32:96, 32:96] >= 3).sum()), 35740)
        with Image.open(ROOT / "inputs_sampling" / SLICE) as source:
            self.assertEqual(source.size, (1024, 1024))

    def test_preserved_functions_and_palette(self):
        fixtures = read_json(ROOT / "provenance/sampling_functions.json")
        for key, record in fixtures.items():
            filename, name = key.split("::")
            source = (ROOT / "src" / filename).read_text(encoding="utf-8")
            nodes = [n for n in ast.parse(source).body
                     if isinstance(n, ast.FunctionDef) and n.name == name]
            self.assertEqual(len(nodes), 1)
            self.assertEqual(ast.get_source_segment(source, nodes[0]), record["text"])
        import sampling_reference as reference
        self.assertEqual(digest(reference.REFERENCE), reference.FUNCTIONS_SHA256)
        ns = reference.original_functions([8, 8, 8], 2)
        self.assertEqual(ns["CLASS_COLORS"][7], "#7560A8")
        self.assertEqual(ns["POINT_HASH_MODULUS"], 3)
        self.assertEqual(set(k for k in ns if k in ("box_edges_xyz", "deterministic_points",
                                                     "add_class_layer", "add_surface_layer")),
                         {"box_edges_xyz", "deterministic_points", "add_class_layer", "add_surface_layer"})

    def test_no_saved_render_cache_input(self):
        import plot_sampling_hierarchy as figure
        self.assertEqual(list(inspect.signature(figure.render).parameters),
                         ["out", "full_path", "case_path", "original_slice"])
        self.assertEqual(figure.CAMERA, {"azimuth": -55., "elevation": 20., "projection": "orthographic"})

    def test_changed_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            changed = read_json(ROOT / "inputs_sampling" / EXPECTED)
            changed["interior_parent_bounds_zyx"] = [[31, 95]]*3
            (output / f"{STEM}.source.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Sampling geometry"):
                compare_metadata(ROOT / "inputs_sampling", output)

    def test_existing_output_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(ROOT / "inputs_sampling", ROOT / "inputs_3d", Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_runtime_stops_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch("reproduce_sampling.verify_versions", side_effect=RuntimeError("missing runtime")):
                with self.assertRaisesRegex(RuntimeError, "missing runtime"):
                    run(ROOT / "inputs_sampling", ROOT / "inputs_3d", output)
            self.assertFalse(output.exists())

if __name__ == "__main__":
    unittest.main()
