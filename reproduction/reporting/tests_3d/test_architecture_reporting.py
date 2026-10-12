"""Check saved features and preserved dataflow without importing the model."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reproduce_architecture import (STEM, FEATURES, FEATURE_META, EXPECTED_META,
                                    verify_inputs, compare_outputs, csv_rows, run)
from reporting import read_json

INPUTS = ROOT / "inputs_architecture"

class ArchitectureTests(unittest.TestCase):
    def test_fixed_feature_manifest(self):
        manifest = verify_inputs(INPUTS)
        self.assertEqual(len(manifest["files"]), 6)
        self.assertEqual(manifest["contract"]["display_step_transition"], [99, 100])
        self.assertEqual(manifest["contract"]["class_logit_channels"], [0, 1, 2, 3, 4])
        self.assertFalse(manifest["contract"]["new_inference"])

    def test_drawing_functions_and_all_dataflow_statements_unchanged(self):
        fixtures = read_json(ROOT / "provenance/architecture_functions.json")
        for original, current in (
                ("render_architecture_dataflow_v8_exposure.py", "architecture_drawing.py"),
                ("render_architecture_dataflow_v9_unified.py", "plot_architecture.py")):
            source = (ROOT / "src" / current).read_text(encoding="utf-8")
            nodes = {n.name: n for n in ast.parse(source).body
                     if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            for name, expected in fixtures[original].items():
                self.assertEqual(ast.get_source_segment(source, nodes[name]), expected)
        source = (ROOT / "src/architecture_drawing.py").read_text(encoding="utf-8")
        start = source.index("    d = Drawing()")
        self.assertEqual(source[start:start+len(fixtures["drawing_body"])], fixtures["drawing_body"])

    def test_fixed_topology_and_recurrence(self):
        objects = {row["object_id"]: row for row in csv_rows(INPUTS / "expected_objects.csv")}
        edges = csv_rows(INPUTS / "expected_edges.csv")
        self.assertEqual(len(objects), 73)
        self.assertEqual(len(edges), 24)
        for edge in edges:
            self.assertIn(edge["source"], objects)
            self.assertIn(edge["target"], objects)
            self.assertEqual(edge["direction"], "forward")
            self.assertGreaterEqual(len(json.loads(edge["route"])), 2)
        delayed = {row["edge_id"] for row in edges if "dashed" in row["style"]}
        self.assertEqual(delayed, {"H_memory", "state_memory"})
        self.assertEqual([(e["source"], e["target"]) for e in edges if e["edge_id"] == "H_memory"],
                         [("GRU", "GRU")])
        for key in ("width", "height"):
            self.assertEqual(objects["X"][key], objects["OUTPUT"][key])

    def test_all_tensor_shapes_finiteness_and_fixed_planes(self):
        import numpy as np
        meta = read_json(INPUTS / FEATURE_META)
        expected = read_json(INPUTS / EXPECTED_META)
        with np.load(INPUTS / FEATURES, allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), set(meta["tensor_shapes"]))
            for name, shape in meta["tensor_shapes"].items():
                values = saved[name]
                self.assertEqual(list(values.shape), shape, name)
                self.assertTrue(np.isfinite(values).all(), name)
            self.assertEqual(int(saved["display_step"]), 100)
            planes = {
                "conv1": saved["conv1_gelu"][:3, 64],
                "E": saved["encoder_E"][:3, 64], "S": saved["pooled_S"][:3, 4],
                "H": saved["hidden"][:3, 4], "ell": saved["local_logit"][:, 64],
                "q": saved["class_logits"][:, 4], "b": saved["cell_bias"][:, 4],
            }
            for name, selected in planes.items():
                expected_limits = expected["feature_display_limits"][name]["per_channel_limits"]
                calculated = [{"channel": k, "min": float(selected[k].min()), "max": float(selected[k].max())}
                              for k in range(len(selected)-1, -1, -1)]
                self.assertEqual(calculated, expected_limits, name)

    def test_last_step_readout_and_fixed_initial_classes(self):
        import numpy as np
        with np.load(INPUTS / FEATURES, allow_pickle=False) as d:
            labels, site, group = d["initial_labels"], d["site_index"], d["group_id"]
            oil = labels >= 3
            np.testing.assert_array_equal(site, np.flatnonzero(oil))
            z, y, x = np.unravel_index(site, labels.shape)
            expected_group = (((z//16)*8 + y//16)*8 + x//16)*5 + labels.ravel()[site] - 3
            np.testing.assert_array_equal(group, expected_group)
            np.testing.assert_array_equal(d["initial_group_inventory"], np.bincount(group, minlength=2560))
            prev, current, front, propensity, hazard = (
                d[name] for name in ("previous_occupancy", "occupancy", "front_weight", "propensity", "hazard"))
            logits = (d["local_logit"].ravel()[site] + d["cell_bias"].ravel()[group//5]
                      + d["class_logits"].transpose(1, 2, 3, 0).ravel()[group])
            # NumPy and CUDA sigmoid can differ by float32 roundoff; no stored
            # probability or figure value is replaced by this check.
            np.testing.assert_allclose(1/(1+np.exp(-logits)), propensity.ravel()[site], rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(hazard, front*propensity)
            np.testing.assert_array_equal(current, prev*(1-hazard))
            self.assertTrue(np.all((0 <= current) & (current <= prev)))
            self.assertTrue(np.all(current[~oil] == 0))
            self.assertEqual(int(oil.sum()), 460312)
            self.assertEqual(int((oil & (current >= .5)).sum()), 457150)
            total = np.bincount(group, weights=current.ravel()[site], minlength=2560)
            # Stored CUDA float32 scatter-add versus independent float64 sum.
            np.testing.assert_allclose(total, d["group_inventory"], rtol=1e-6, atol=1e-5)

    def test_exposure_uses_previous_state_and_fixed_window(self):
        import numpy as np
        with np.load(INPUTS / FEATURES, allow_pickle=False) as d:
            labels, connected, inlet = d["initial_labels"], d["connected"], d["inlet"]
            previous = d["previous_occupancy"]
            self.assertTrue(np.all(~inlet | connected))
            self.assertTrue(np.all(~connected | (labels >= 2)))
            source = np.maximum(connected.astype(np.float32)-previous, inlet.astype(np.float32))
            neighbours = np.zeros_like(source)
            for axis in range(3):
                upper, lower = [slice(None)]*3, [slice(None)]*3
                upper[axis], lower[axis] = slice(1, None), slice(None, -1)
                upper, lower = tuple(upper), tuple(lower)
                neighbours[upper] = np.maximum(neighbours[upper], source[lower])
                neighbours[lower] = np.maximum(neighbours[lower], source[upper])
            reconstructed = neighbours*d["reachable_support"]*(labels >= 3)
            np.testing.assert_array_equal(reconstructed, d["front_weight"])
            projection = d["front_weight"].max(axis=0)
            self.assertEqual(int(np.count_nonzero(projection)), 488)
            self.assertEqual(int(np.count_nonzero(projection[:40, :40])), 488)
            self.assertTrue(np.all((projection >= 0) & (projection <= 1)))

    def test_recurrent_context_is_not_future_state(self):
        import numpy as np
        with np.load(INPUTS / FEATURES, allow_pickle=False) as d:
            context = d["context"]
            np.testing.assert_array_equal(context[:8], d["pooled_S"])
            np.testing.assert_array_equal(context[13:17], np.broadcast_to(d["action"][:, None, None, None], (4, 8, 8, 8)))
            np.testing.assert_array_equal(context[17], np.full((8, 8, 8), d["time"].item(), np.float32))
            self.assertEqual(d["time"].item(), float(np.float32(100/125)))
            totals = np.bincount(d["group_id"], weights=d["previous_occupancy"].ravel()[d["site_index"]], minlength=2560)
            fractions = (totals/np.maximum(d["initial_group_inventory"], 1)).reshape(8, 8, 8, 5).transpose(3, 0, 1, 2)
            np.testing.assert_allclose(fractions, context[8:13], rtol=1e-6, atol=1e-7)

    def test_changed_edge_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for kind in ("objects", "edges"):
                text = (INPUTS / f"expected_{kind}.csv").read_text(encoding="utf-8")
                if kind == "edges":
                    text = text.replace("X_E1,X,E1", "X_E1,OUTPUT,E1", 1)
                (output / f"{STEM}.{kind}.csv").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Architecture edges"):
                compare_outputs(INPUTS, output)

    def test_existing_output_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(INPUTS, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_runtime_stops_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch("reproduce_architecture.verify_versions", side_effect=RuntimeError("missing runtime")):
                with self.assertRaisesRegex(RuntimeError, "missing runtime"):
                    run(INPUTS, output)
            self.assertFalse(output.exists())

if __name__ == "__main__":
    unittest.main()
