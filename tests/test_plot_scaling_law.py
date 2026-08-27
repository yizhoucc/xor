import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_scaling_law import SCALES, load_scaling_data


class PlotScalingLawTest(unittest.TestCase):
    def test_values_are_loaded_from_canonical_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.csv"
            comparisons = Path(tmp) / "comparisons.csv"
            fields = ["exp_name", "metric", "run_status", "mean", "population_sd"]
            with summary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for width, spec in SCALES.items():
                    for index, method in enumerate(("gelu", "swiglu", "innernet")):
                        writer.writerow({
                            "exp_name": spec[method], "metric": "best_val_ppl",
                            "run_status": "success", "mean": 100 - width / 10 - index,
                            "population_sd": 0.5,
                        })
            with comparisons.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["name", "parametric_p"], lineterminator="\n")
                writer.writeheader()
                for spec in SCALES.values():
                    writer.writerow({"name": spec["comparison"], "parametric_p": 0.05})

            data = load_scaling_data(summary, comparisons)
            self.assertEqual(sorted(data), [64, 128, 192, 256])
            self.assertGreater(data[64]["improvement_pct"], 0)
            self.assertEqual(data[128]["paired_t_p"], 0.05)


if __name__ == "__main__":
    unittest.main()
