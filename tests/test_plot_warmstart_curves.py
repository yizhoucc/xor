import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.plot_warmstart_curves import load_curves


class PlotWarmstartCurvesTest(unittest.TestCase):
    def test_frozen_curves_are_truncated_to_common_length(self):
        runs = []
        for seed, length in ((42, 3), (43, 2)):
            runs.append({
                "seed": seed,
                "sw_ppl_phase1": [10.0, 9.0],
                "sw_ppl_phase2": [8.0, 7.0],
                "in_ppl_phase2": [8.5, 7.1],
                "ppl_swap": 12.0,
                "frozen_ppl": list(range(length)),
                "best_swiglu_final": 7.0,
            })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.p"
            with path.open("wb") as handle:
                pickle.dump({"all_results": runs}, handle)
            curves = load_curves(path)
        self.assertEqual(curves["phase1"].shape, (2, 2))
        self.assertEqual(curves["frozen"].shape, (2, 2))
        self.assertEqual(curves["seeds"], [42, 43])


if __name__ == "__main__":
    unittest.main()
