import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_housing_scaling import WIDTHS, load_housing_scaling


class PlotHousingScalingTest(unittest.TestCase):
    def test_loads_canonical_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.csv"
            comparisons = Path(tmp) / "comparisons.csv"
            with summary.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["exp_name", "metric", "run_status", "mean", "population_sd"],
                )
                writer.writeheader()
                for width in WIDTHS:
                    writer.writerow({
                        "exp_name": f"mlp_housing_scale_2arg_w{width}",
                        "metric": "test_mse", "run_status": "success",
                        "mean": 0.2, "population_sd": 0.01,
                    })
                    writer.writerow({
                        "exp_name": f"mlp_housing_scale_relu_w{width}",
                        "metric": "test_mse", "run_status": "success",
                        "mean": 0.25, "population_sd": 0.02,
                    })
            with comparisons.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["name", "parametric_p"])
                writer.writeheader()
                for width in WIDTHS:
                    writer.writerow({
                        "name": f"housing_w{width}_innernet_vs_relu",
                        "parametric_p": 0.05,
                    })

            data = load_housing_scaling(summary, comparisons)
        self.assertEqual(sorted(data), list(WIDTHS))
        self.assertAlmostEqual(data[32]["mse_reduction_pct"], 20.0)
        self.assertEqual(data[512]["paired_t_p"], 0.05)


if __name__ == "__main__":
    unittest.main()
