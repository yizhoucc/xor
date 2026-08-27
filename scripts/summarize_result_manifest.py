"""Aggregate the canonical metric manifest without hiding duplicate conflicts.

The input is produced by ``scripts/build_result_manifest.py``. Rows are grouped
by experiment identity and metric. Repeated copies of the same seed/value are
deduplicated, while repeated seeds with different values are emitted to a
separate conflict report and make the group non-reportable.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


GROUP_FIELDS = (
    "exp_name",
    "task_type",
    "model",
    "dataset",
    "metric",
    "selection",
    "config_signature",
)
SUMMARY_FIELDS = GROUP_FIELDS + (
    "audit_status",
    "raw_row_count",
    "seed_count",
    "seeds",
    "values",
    "mean",
    "sample_sd",
    "population_sd",
    "min",
    "max",
    "identical_duplicate_rows",
    "conflicting_seed_count",
    "notes",
)
CONFLICT_FIELDS = GROUP_FIELDS + (
    "seed",
    "values",
    "source_files",
    "config_hashes",
)
VARIANT_KEY_FIELDS = tuple(field for field in GROUP_FIELDS if field != "config_signature")
VARIANT_FIELDS = VARIANT_KEY_FIELDS + (
    "signature_count",
    "config_signatures",
    "seeds",
    "values",
    "source_files",
)


def _close(a, b, tolerance):
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def aggregate_rows(rows, tolerance=1e-12):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in GROUP_FIELDS)].append(row)

    summaries = []
    conflicts = []
    for key in sorted(groups):
        group_rows = groups[key]
        by_seed = defaultdict(list)
        for row in group_rows:
            by_seed[str(row["seed"])].append(row)

        canonical = []
        duplicate_count = 0
        conflict_count = 0
        for seed in sorted(by_seed, key=lambda value: (not value.isdigit(), value)):
            seed_rows = by_seed[seed]
            values = [float(row["value"]) for row in seed_rows]
            unique_values = []
            for value in values:
                if not any(_close(value, prior, tolerance) for prior in unique_values):
                    unique_values.append(value)
            if len(unique_values) > 1:
                conflict_count += 1
                conflicts.append({
                    **dict(zip(GROUP_FIELDS, key)),
                    "seed": seed,
                    "values": ";".join(f"{value:.12g}" for value in values),
                    "source_files": ";".join(row["source_file"] for row in seed_rows),
                    "config_hashes": ";".join(row["config_hash"] for row in seed_rows),
                })
                continue
            canonical.append((seed, unique_values[0]))
            duplicate_count += len(seed_rows) - 1

        values = [value for _, value in canonical]
        reportable = conflict_count == 0 and bool(values)
        summary = {
            **dict(zip(GROUP_FIELDS, key)),
            "audit_status": "reportable" if reportable else "conflicted",
            "raw_row_count": len(group_rows),
            "seed_count": len(values),
            "seeds": ";".join(seed for seed, _ in canonical),
            "values": ";".join(f"{value:.12g}" for value in values),
            "mean": f"{statistics.mean(values):.12g}" if reportable else "",
            "sample_sd": f"{statistics.stdev(values):.12g}" if reportable and len(values) > 1 else "",
            "population_sd": f"{statistics.pstdev(values):.12g}" if reportable else "",
            "min": f"{min(values):.12g}" if reportable else "",
            "max": f"{max(values):.12g}" if reportable else "",
            "identical_duplicate_rows": duplicate_count,
            "conflicting_seed_count": conflict_count,
            "notes": "mean/SD withheld until conflicts are resolved" if conflict_count else "",
        }
        summaries.append(summary)
    return summaries, conflicts


def find_variant_collisions(rows):
    """Find reused experiment names that refer to different scientific configs."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in VARIANT_KEY_FIELDS)].append(row)

    collisions = []
    for key in sorted(groups):
        group_rows = groups[key]
        signatures = sorted({row["config_signature"] for row in group_rows})
        if len(signatures) <= 1:
            continue
        collisions.append({
            **dict(zip(VARIANT_KEY_FIELDS, key)),
            "signature_count": len(signatures),
            "config_signatures": ";".join(signatures),
            "seeds": ";".join(str(row["seed"]) for row in group_rows),
            "values": ";".join(f"{float(row['value']):.12g}" for row in group_rows),
            "source_files": ";".join(row["source_file"] for row in group_rows),
        })
    return collisions


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/audit/metric_manifest.csv")
    parser.add_argument("--output-dir", default="results/audit")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    summaries, conflicts = aggregate_rows(rows, args.tolerance)
    summary_path = output_dir / "grouped_metric_summary.csv"
    conflict_path = output_dir / "metric_conflicts.csv"
    variant_path = output_dir / "experiment_variant_collisions.csv"
    report_path = output_dir / "grouped_summary_report.json"
    _write_csv(summary_path, SUMMARY_FIELDS, summaries)
    _write_csv(conflict_path, CONFLICT_FIELDS, conflicts)
    variants = find_variant_collisions(rows)
    _write_csv(variant_path, VARIANT_FIELDS, variants)
    report = {
        "manifest": str(manifest),
        "group_count": len(summaries),
        "reportable_group_count": sum(row["audit_status"] == "reportable" for row in summaries),
        "conflicted_group_count": sum(row["audit_status"] == "conflicted" for row in summaries),
        "conflicting_seed_count": len(conflicts),
        "variant_collision_count": len(variants),
        "summary": str(summary_path),
        "conflicts": str(conflict_path),
        "variant_collisions": str(variant_path),
    }
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
