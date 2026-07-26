import unittest

import numpy as np

from scripts.compute_stats import (
    independent_comparison,
    paired_comparison,
    prepare_paired,
)


class ComputeStatsTest(unittest.TestCase):
    def test_paired_comparison_uses_within_pair_differences(self):
        condition = np.array([8.0, 10.0, 11.0, 13.0, 15.0])
        reference = np.array([10.0, 11.0, 13.0, 14.0, 18.0])
        result = paired_comparison(
            condition,
            reference,
            n_bootstrap=500,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["mean_difference"], -1.8)
        self.assertEqual(result["effect_name"], "Cohen dz")
        self.assertLessEqual(result["ci_high"], 0.0)

    def test_paired_nonfinite_values_are_dropped_together(self):
        x, y, dropped = prepare_paired(
            np.array([1.0, np.nan, 3.0, 4.0]),
            np.array([2.0, 2.0, np.inf, 5.0]),
        )
        np.testing.assert_array_equal(x, np.array([1.0, 4.0]))
        np.testing.assert_array_equal(y, np.array([2.0, 5.0]))
        self.assertEqual(dropped, 2)

    def test_paired_comparison_rejects_misaligned_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            paired_comparison([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_independent_comparison_reports_group_sizes(self):
        result = independent_comparison(
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            n_bootstrap=500,
            rng=np.random.default_rng(11),
        )
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["n_ref"], 4)
        self.assertAlmostEqual(result["mean_difference"], -1.5)
        self.assertEqual(result["effect_name"], "Cohen d")


if __name__ == "__main__":
    unittest.main()
