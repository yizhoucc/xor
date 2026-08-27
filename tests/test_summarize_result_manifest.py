import unittest

from scripts.summarize_result_manifest import aggregate_rows, find_variant_collisions


def _row(seed, value, source, config_hash="hash"):
    return {
        "exp_name": "example",
        "task_type": "classification",
        "model": "Model",
        "dataset": "dataset",
        "condition": "Model",
        "metric": "test_accuracy",
        "selection": "reported_scalar",
        "run_status": "success",
        "seed": str(seed),
        "value": str(value),
        "source_file": source,
        "config_hash": config_hash,
        "config_signature": "signature-a",
    }


class SummarizeResultManifestTest(unittest.TestCase):
    def test_unique_seeds_are_aggregated_with_both_sd_conventions(self):
        summaries, conflicts = aggregate_rows([
            _row(42, 0.8, "a"),
            _row(43, 0.9, "b"),
        ])
        self.assertEqual(conflicts, [])
        self.assertEqual(summaries[0]["audit_status"], "reportable")
        self.assertEqual(summaries[0]["seed_count"], 2)
        self.assertAlmostEqual(float(summaries[0]["mean"]), 0.85)
        self.assertAlmostEqual(float(summaries[0]["population_sd"]), 0.05)

    def test_identical_duplicate_seed_is_deduplicated(self):
        summaries, conflicts = aggregate_rows([
            _row(42, 0.8, "a"),
            _row(42, 0.8, "copy"),
        ])
        self.assertEqual(conflicts, [])
        self.assertEqual(summaries[0]["seed_count"], 1)
        self.assertEqual(summaries[0]["identical_duplicate_rows"], 1)

    def test_conflicting_duplicate_seed_withholds_summary_statistics(self):
        summaries, conflicts = aggregate_rows([
            _row(42, 0.8, "a", "hash-a"),
            _row(42, 0.9, "b", "hash-b"),
            _row(43, 0.85, "c", "hash-c"),
        ])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["seed"], "42")
        self.assertEqual(summaries[0]["audit_status"], "conflicted")
        self.assertEqual(summaries[0]["mean"], "")
        self.assertEqual(summaries[0]["conflicting_seed_count"], 1)

    def test_same_name_with_different_config_signatures_is_a_variant_collision(self):
        first = _row(42, 0.8, "a")
        second = _row(43, 0.9, "b")
        second["config_signature"] = "signature-b"
        collisions = find_variant_collisions([first, second])
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0]["signature_count"], 2)


if __name__ == "__main__":
    unittest.main()
