# Result Audit Artifacts

These files are generated from structured artifacts under `exp/`:

- `experiment_inventory.csv`: one row per experiment directory containing a `config.yaml`
- `metric_manifest.csv`: one row per parsed seed, metric, and selection rule
- `audit_summary.json`: inventory counts and result-source counts
- `grouped_metric_summary.csv`: automatic per-experiment/metric mean, sample SD,
  population SD, raw seeds, and values; conflicted groups withhold statistics
- `metric_conflicts.csv`: duplicate seed values that disagree, including every
  source path and config hash
- `experiment_variant_collisions.csv`: reused experiment names that correspond
  to different scientific configurations (for example, unmatched vs
  parameter-matched widths)
- `grouped_summary_report.json`: counts of reportable and conflicted groups

Generate in order:

```bash
python scripts/build_result_manifest.py
python scripts/summarize_result_manifest.py
```

Regenerate with:

```bash
.venv/bin/python scripts/build_result_manifest.py
```

The generator parses only `test_results.p`, `lm_results.p`, `mixer_results.p`, and `rl_results.p`. It does not infer metrics from logs or copy values from result documents. The pickle files are trusted local project artifacts; do not run the generator on untrusted experiment directories.

Audit statuses:

- `raw-verified`: at least one supported metric was read from a structured result file
- `result-unparsed`: a known result file exists but has no supported metric or failed parsing
- `completed-no-result`: a `COMPLETED` marker exists without a structured result source
- `incomplete`: no `COMPLETED` marker and no structured result source

Current local exception: `exp/cnn_cifar_2arg_20260404_172455_9a5b0541` (seed 42) has `COMPLETED` and `TEST_DONE`, but its logs stop at `Starting Test` and no `test_results.p` exists. The published summary can be algebraically reconciled with a 79.68% value for this seed, but that value remains `doc-derived` until its original result source is recovered.
