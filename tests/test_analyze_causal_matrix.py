import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.analyze_causal_matrix import analyze_result_file, expected_keys, summarize
from scripts.warmstart_free_init import InnerNetAct


class AnalyzeCausalMatrixTest(unittest.TestCase):
    def test_result_loading_and_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probes" / "bilinear_joint_s42"
            probe.mkdir(parents=True)
            checkpoint = probe / "inner_bilinear_joint_random_seed42.pth"
            torch.save(InnerNetAct(4).state_dict(), checkpoint)
            with (probe / "results.json").open("w") as handle:
                json.dump({
                    "host": "bilinear",
                    "freeze": False,
                    "seed": 42,
                    "host_ppl": 80.0,
                    "conditions": {
                        "random": {
                            "status": "completed",
                            "ppl": [79.0, 78.0],
                            "best_ppl": 78.0,
                            "checkpoint": "/cluster/path/" + checkpoint.name,
                        }
                    },
                }, handle)

            rows = analyze_result_file(probe / "results.json", grid_n=10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["best_ppl"], 78.0)
            self.assertEqual(rows[0]["n_epochs"], 2)
            self.assertIsNotNone(rows[0]["mult_r2"])
            summary = summarize(rows)
            self.assertEqual(summary["complete_conditions"], 1)
            self.assertEqual(len(summary["missing_conditions"]), len(expected_keys()) - 1)


if __name__ == "__main__":
    unittest.main()
