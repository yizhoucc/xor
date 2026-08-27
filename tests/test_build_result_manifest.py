import pickle
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.build_result_manifest import _config_signature, audit_experiment, build_manifest


def _write_config(directory, **overrides):
    config = {
        "exp_name": "example",
        "task_type": "classification",
        "seed": 42,
        "dataset": {"name": "cifar10"},
        "model": {"name": "ExampleModel"},
    }
    config.update(overrides)
    with (directory / "config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle)


class BuildResultManifestTest(unittest.TestCase):
    def test_config_signature_ignores_run_metadata_but_tracks_architecture(self):
        base = {
            "exp_name": "example",
            "seed": 42,
            "run_id": "a",
            "save_dir": "exp/a",
            "use_gpu": False,
            "model": {"name": "MLP", "out_hidden_dim": [64, 64]},
        }
        rerun = {
            **base,
            "seed": 43,
            "run_id": "b",
            "save_dir": "exp/b",
            "use_gpu": True,
        }
        wider = {**rerun, "model": {"name": "MLP", "out_hidden_dim": [112, 112]}}
        self.assertEqual(_config_signature(base), _config_signature(rerun))
        self.assertNotEqual(_config_signature(base), _config_signature(wider))

    def test_scalar_test_result_is_raw_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run = root / "run_a"
            run.mkdir(parents=True)
            _write_config(run)
            (run / "COMPLETED").touch()
            with (run / "test_results.p").open("wb") as handle:
                pickle.dump({"test_accuracy": 0.75}, handle)

            inventory, metrics = build_manifest(root)
            self.assertEqual(inventory[0]["audit_status"], "raw-verified")
            self.assertEqual(metrics[0]["metric"], "test_accuracy")
            self.assertEqual(metrics[0]["seed"], 42)
            self.assertAlmostEqual(metrics[0]["value"], 0.75)

    def test_multiseed_curves_keep_best_and_final_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run = root / "run_lm"
            run.mkdir(parents=True)
            _write_config(run, task_type="language_model")
            with (run / "lm_results.p").open("wb") as handle:
                pickle.dump(
                    {"seeds": [42, 43], "all_ppl": [[10.0, 8.0, 9.0], [11.0, 7.0, 6.0]]},
                    handle,
                )

            _, metrics = build_manifest(root)
            self.assertEqual(len(metrics), 4)
            seed42 = [row for row in metrics if row["seed"] == 42]
            best = next(row for row in seed42 if row["metric"] == "best_val_ppl")
            final = next(row for row in seed42 if row["metric"] == "final_val_ppl")
            self.assertEqual((best["value"], best["selected_epoch"]), (8.0, 2))
            self.assertEqual((final["value"], final["selected_epoch"]), (9.0, 3))

    def test_completed_directory_without_result_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run = root / "run_missing"
            run.mkdir(parents=True)
            _write_config(run)
            (run / "COMPLETED").touch()

            inventory, metrics = build_manifest(root)
            self.assertEqual(inventory[0]["audit_status"], "completed-no-result")
            self.assertEqual(metrics, [])

    def test_incomplete_directory_is_not_treated_as_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run = root / "run_incomplete"
            run.mkdir(parents=True)
            _write_config(run)

            row, metrics = audit_experiment(run / "config.yaml", root)
            self.assertEqual(row["audit_status"], "incomplete")
            self.assertEqual(metrics, [])

    def test_seq_mnist_script_results_are_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run = root / "seq_run"
            run.mkdir(parents=True)
            _write_config(run, task_type="seq_mnist")
            with (run / "results.p").open("wb") as handle:
                pickle.dump([{
                    "seed": 42,
                    "best_acc": 0.98,
                    "history": {"test_acc": [0.5, 0.98, 0.97]},
                }], handle)

            inventory, metrics = build_manifest(root)
            self.assertEqual(inventory[0]["audit_status"], "raw-verified")
            self.assertEqual(
                {(row["metric"], row["value"], row["run_status"]) for row in metrics},
                {
                    ("best_test_accuracy", 0.98, "success"),
                    ("final_test_accuracy", 0.97, "success"),
                },
            )

    def test_seq_mnist_nan_run_is_explicitly_tagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run = root / "seq_run"
            run.mkdir(parents=True)
            _write_config(run, task_type="seq_mnist")
            with (run / "results.p").open("wb") as handle:
                pickle.dump([{
                    "seed": 42,
                    "best_acc": 0.1,
                    "history": {"train_loss": [2.3, float("nan")], "test_acc": [0.1]},
                }], handle)

            _, metrics = build_manifest(root)
            self.assertTrue(metrics)
            self.assertEqual({row["run_status"] for row in metrics}, {"nan"})

    def test_orphan_deploy_json_is_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run = root / "deploy"
            run.mkdir(parents=True)
            with (run / "results.json").open("w") as handle:
                import json
                json.dump({
                    "args": {"dataset": "wikitext", "num_seeds": 2},
                    "results": {"innernet": {"ppl": [10.0, 9.0]}},
                }, handle)

            inventory, metrics = build_manifest(root)
            self.assertEqual(inventory[0]["audit_status"], "raw-verified")
            self.assertIn("config.yaml unavailable", inventory[0]["notes"])
            self.assertEqual([row["seed"] for row in metrics], [42, 43])
            self.assertEqual([row["metric"] for row in metrics], ["test_ppl", "test_ppl"])


if __name__ == "__main__":
    unittest.main()
