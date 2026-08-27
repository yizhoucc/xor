import unittest

from scripts.build_core_stats import build_comparisons


def _row(exp_name, seed, value, signature="sig"):
    return {
        "exp_name": exp_name,
        "metric": "accuracy",
        "selection": "reported_scalar",
        "run_status": "success",
        "config_signature": signature,
        "seed": str(seed),
        "value": str(value),
    }


class BuildCoreStatsTest(unittest.TestCase):
    def test_paired_comparison_aligns_by_seed_not_row_order(self):
        rows = [
            _row("condition", 43, 0.9),
            _row("condition", 42, 0.8),
            _row("reference", 42, 0.7),
            _row("reference", 43, 0.75),
        ]
        specs = [{
            "name": "example",
            "condition": {"exp_name": "condition", "metric": "accuracy"},
            "reference": {"exp_name": "reference", "metric": "accuracy"},
            "paired": True,
            "lower_better": False,
        }]
        result = build_comparisons(rows, specs, bootstrap=100, random_seed=0)[0]
        self.assertEqual(result["common_seeds"], "42;43")
        self.assertAlmostEqual(result["mean_difference"], 0.125)
        self.assertEqual(result["direction"], "condition_better")

    def test_ambiguous_config_requires_explicit_signature(self):
        rows = [
            _row("condition", 42, 0.8, "a"),
            _row("condition", 43, 0.9, "b"),
            _row("reference", 42, 0.7, "r"),
            _row("reference", 43, 0.75, "r"),
        ]
        specs = [{
            "name": "example",
            "condition": {"exp_name": "condition", "metric": "accuracy"},
            "reference": {"exp_name": "reference", "metric": "accuracy"},
            "paired": True,
        }]
        with self.assertRaisesRegex(ValueError, "ambiguous scientific configs"):
            build_comparisons(rows, specs, bootstrap=100)


if __name__ == "__main__":
    unittest.main()
