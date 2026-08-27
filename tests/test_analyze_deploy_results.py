import unittest

from scripts.analyze_deploy_results import summarize_payload


class AnalyzeDeployResultsTest(unittest.TestCase):
    def test_incomplete_condition_and_speedup_are_explicit(self):
        data = {"results": {
            "innernet": {"acc": [0.8, 0.9], "tput": [10.0, 10.0]},
            "distilled": {"acc": [0.75], "tput": [25.0]},
        }}
        result = summarize_payload(data, {"innernet": 100, "distilled": 90}, "acc", 2)
        self.assertTrue(result["conditions"]["innernet"]["complete"])
        self.assertFalse(result["conditions"]["distilled"]["complete"])
        self.assertEqual(
            result["comparisons"]["distilled_vs_innernet"]["throughput_ratio_condition_vs_reference"],
            2.5,
        )
        self.assertIsNone(
            result["comparisons"]["distilled_vs_innernet"]["metric_difference_condition_minus_reference"]
        )


if __name__ == "__main__":
    unittest.main()
