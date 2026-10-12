"""Small deterministic checks; no model, remote service or experiment is used."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from reporting import verify_inputs, make_main_tables, make_tables, export_tables, reduction, read_json, digest
from supplement_reporting import load_dev_metrics, moments, display_pair, dev_comparator
from reproduce import run

class SavedReportingTests(unittest.TestCase):
    def test_input_snapshot(self):
        self.assertEqual(len(verify_inputs(ROOT / "inputs")["files"]), 12)

    def test_all_main_table_values(self):
        rows, wide, tokens = make_main_tables(ROOT / "inputs")
        self.assertEqual([len(wide[k][1]) for k in ("table1", "table2", "table3")], [9, 9, 6])
        self.assertEqual(sum(len(row) for t in tokens.values() for row in t), 119)
        single = [r for r in rows["table3"] if r["group"] == "single_seed"]
        self.assertEqual(len(single), 4)
        self.assertTrue(all(r["raw_sample_sd"] is None and r["n_seeds"] == 1 for r in single))

    def test_export_without_plot_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            report = export_tables(ROOT / "inputs", Path(folder) / "tables")
            self.assertEqual(report["numeric_values_match_manuscript"], 335)
            self.assertEqual(len(list((Path(folder) / "tables").iterdir())), 21)

    def test_all_supplement_table_values(self):
        rows, wide, tokens = make_tables(ROOT / "inputs")
        keys = ("tableS1", "tableS2", "tableS3", "tableS4")
        self.assertEqual([len(wide[key][1]) for key in keys], [16, 20, 4, 6])
        self.assertEqual([sum(map(len, tokens[key])) for key in keys], [96, 80, 20, 20])
        self.assertEqual(sum(len(v) for v in tokens.values()), 70)
        self.assertEqual(sum(len(row) for table in tokens.values() for row in table), 335)
        single = [r for r in rows["tableS4"] if r["group"] == "single_seed"]
        self.assertEqual(len(single), 4)
        self.assertTrue(all(r["raw_sample_sd"] is None and r["n_seeds"] == 1 for r in single))
        self.assertEqual({r["split"] for r in rows["tableS2"]}, {"TEST"})
        for key in ("tableS1", "tableS3", "tableS4"):
            self.assertEqual({r["split"] for r in rows[key]}, {"DEV"})

    def test_packaged_base_snapshot_is_consistent(self):
        prior = read_json(ROOT / "provenance/main_inputs_manifest_v1.json")
        current = read_json(ROOT / "inputs/manifest.json")
        self.assertEqual(current["previous_manifest_sha256"],
                         digest(ROOT / "provenance/main_inputs_manifest_v1.json"))
        self.assertEqual(len(prior["files"]), 6)
        for name, record in prior["files"].items():
            self.assertEqual(record, current["files"][name])
            self.assertEqual(record["sha256"], digest(ROOT / "inputs" / name))

    def test_mismatched_seed_aggregate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aggregate disagrees"):
            moments([1, 2, 3], {"mean": 2, "seed_sd": 0})

    def test_r_squared_can_be_negative(self):
        self.assertEqual(display_pair(-0.25, 0.01, 1, 4), ["-0.2500", "0.0100"])
        with self.assertRaises(ValueError):
            display_pair(1, -0.1, 1, 4)

    def test_nonfinite_or_incomplete_seed_scores_are_rejected(self):
        for values in ([1, 2], [1, 2, float("inf")], [1, 2, float("nan")]):
            with self.assertRaises(ValueError):
                moments(values)

    def test_duplicate_dev_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder)
            source = (ROOT / "inputs/dev_secondary_metrics.csv").read_text(encoding="utf-8")
            (dest / "dev_secondary_metrics.csv").write_text(
                source + source.splitlines()[1] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate DEV"):
                load_dev_metrics(dest)

    def test_mislabeled_dev_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder)
            shutil.copyfile(ROOT / "inputs/du_dev_20260811.json", dest / "du_dev_20260812.json")
            with self.assertRaisesRegex(ValueError, "comparator protocol"):
                dev_comparator(dest, "du_gradcos", 20260812)

    def test_zero_reference_is_not_divided(self):
        with self.assertRaises(ValueError):
            reduction(0, 0)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run(ROOT / "inputs", Path(folder), tables_only=True)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_wrong_plot_version_stops_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new_run"
            with patch("reproduce.importlib.metadata.version", return_value="wrong-version"):
                with self.assertRaisesRegex(RuntimeError, "expected"):
                    run(ROOT / "inputs", output)
            self.assertFalse(output.exists())

    def test_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "inputs"
            shutil.copytree(ROOT / "inputs", dest)
            with (dest / "long_horizon.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "Saved input changed"):
                verify_inputs(dest)

if __name__ == "__main__":
    unittest.main()
