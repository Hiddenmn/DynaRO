"""Checks of geometry validation only; no scientific input or model is used."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_layout import check_layout


class LayoutChecks(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_equal_row_and_unchanged_rectangles(self):
        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
        before = [a.get_position().bounds for a in axes]
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'DYNARO_FIGURE_QA': ''}):
            result = check_layout(fig, json_out=Path(folder) / 'layout.json')
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(before, [a.get_position().bounds for a in axes])

    def test_single_axis_is_not_grid_pass(self):
        fig, _ = plt.subplots()
        with patch.dict('os.environ', {'DYNARO_FIGURE_QA': ''}):
            self.assertEqual(check_layout(fig)['status'], 'NOT_APPLICABLE')

    def test_shared_edges_of_spanning_panel(self):
        fig = plt.figure()
        grid = fig.add_gridspec(2, 2)
        for slot in (grid[0, 0], grid[0, 1], grid[1, :]):
            fig.add_subplot(slot)
        with patch.dict('os.environ', {'DYNARO_FIGURE_QA': ''}):
            result = check_layout(fig)
        self.assertTrue(any(r['check'] == 'grid-boundary-0' for r in result['checks']))

    def test_bad_width_and_overlap_are_rejected(self):
        fig, axes = plt.subplots(1, 3)
        box = axes[1].get_position()
        axes[1].set_position([box.x0, box.y0, box.width * .7, box.height])
        with self.assertRaisesRegex(ValueError, 'failed checks'):
            check_layout(fig)
        axes[1].set_position(axes[0].get_position())
        with self.assertRaisesRegex(ValueError, 'failed checks'):
            check_layout(fig)

    def test_unmeasured_groups_and_missing_labels_rejected(self):
        fig = plt.figure()
        fig.add_axes([.1, .2, .2, .5])
        fig.add_axes([.6, .2, .2, .5])
        with self.assertRaisesRegex(ValueError, 'Comparable groups'):
            check_layout(fig)
        with self.assertRaisesRegex(ValueError, 'unambiguous label'):
            check_layout(fig, row_groups=[['a', 'b']], require_panel_labels=True)


if __name__ == '__main__':
    unittest.main()
