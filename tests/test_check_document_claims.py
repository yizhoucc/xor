import tempfile
import unittest
from pathlib import Path

from scripts.check_document_claims import check_claims


class CheckDocumentClaimsTest(unittest.TestCase):
    def test_markdown_formatting_is_ignored_but_value_is_checked(self):
        summary = [{
            "exp_name": "exp",
            "metric": "accuracy",
            "run_status": "success",
            "audit_status": "reportable",
            "condition": "model",
            "config_signature": "sig",
            "mean": "0.8",
            "population_sd": "0.01",
            "sample_sd": "0.02",
            "seed_count": "5",
        }]
        checks = [{
            "id": "claim",
            "file": "doc.md",
            "line_contains": "| Model |",
            "cells": {1: {
                "exp_name": "exp", "metric": "accuracy",
                "format": "mean_pm_pop", "scale": 100, "decimals": 2,
            }},
        }]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "doc.md").write_text("| Model | **80.00 ± 1.00** |\n")
            result = check_claims(summary, checks, Path(tmp))
        self.assertEqual(result[0]["status"], "match")


if __name__ == "__main__":
    unittest.main()
