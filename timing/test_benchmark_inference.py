import unittest
from pathlib import Path

import numpy as np
import torch

from benchmark_inference import choose_cases, controls, pore_masks, summary


class ProtocolTests(unittest.TestCase):
    def test_face_connectivity_not_diagonal(self):
        labels = np.ones((4, 4, 4), dtype=np.uint8)
        labels[0, 0, :2] = 2
        labels[1, 1, 2] = 7
        inlet, connected = pore_masks(labels)
        self.assertEqual(inlet.sum(), 1)
        self.assertEqual(connected.sum(), 2)
        self.assertFalse(connected[1, 1, 2])

    def test_control_switch_and_original_time_feature(self):
        actions, times = controls(100)
        self.assertTrue(torch.all(actions[:20, 0] == 1 / 1.45))
        self.assertTrue(torch.all(actions[20:, 0] == 1))
        self.assertEqual(torch.count_nonzero(actions[:, 1:]), 0)
        self.assertAlmostEqual(times[24, 0].item(), .5)
        self.assertAlmostEqual(times[-1, 0].item(), .8)

    def test_case_selection_uses_only_five_dev_blocks(self):
        root = Path(__file__).resolve().parents[1] / "DynaRO_release_prepared_20261009/reproduction"
        cases = choose_cases(root)
        self.assertEqual(len(cases), 5)
        self.assertTrue(all(c["split"] == "dev" for c in cases))
        self.assertEqual([c["block_id"] for c in cases], sorted(c["block_id"] for c in cases))

    def test_summary_weights_blocks_equally(self):
        rows = [{"model": "dynaro", "horizon": 25, "latency_median_s": v,
                 "peak_allocated_mib": v * 10, "peak_reserved_mib": v * 20}
                for v in [1., 3.]]
        result = summary(rows)[0]
        self.assertEqual(result["mean_case_median_s"], 2.)
        self.assertEqual(result["max_peak_allocated_mib"], 30.)


if __name__ == "__main__":
    unittest.main()
