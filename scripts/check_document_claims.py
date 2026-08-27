"""Verify registered Markdown table cells against the canonical summaries."""
import argparse
import csv
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import yaml


OUTPUT_FIELDS = (
    "id",
    "file",
    "line_contains",
    "column",
    "expected",
    "observed",
    "status",
    "source_exp_name",
    "source_metric",
    "source_seed_count",
)


def _normalize_cell(value):
    return re.sub(r"[\s*`%]", "", value)


def _find_summary(rows, spec):
    selected = [
        row for row in rows
        if row["exp_name"] == spec["exp_name"]
        and row["metric"] == spec["metric"]
        and row["run_status"] == spec.get("run_status", "success")
        and row["audit_status"] == "reportable"
    ]
    if spec.get("condition"):
        selected = [row for row in selected if row["condition"] == spec["condition"]]
    if spec.get("config_signature"):
        selected = [row for row in selected if row["config_signature"] == spec["config_signature"]]
    if len(selected) != 1:
        raise ValueError(f"expected one summary row for {spec}, found {len(selected)}")
    return selected[0]


def _format_expected(row, spec):
    scale = Decimal(str(spec.get("scale", 1.0)))
    decimals = int(spec.get("decimals", 2))
    quantum = Decimal(1).scaleb(-decimals)
    mean = (Decimal(row["mean"]) * scale).quantize(quantum, rounding=ROUND_HALF_UP)
    if spec["format"] == "mean":
        return f"{mean:.{decimals}f}"
    if spec["format"] == "mean_pm_pop":
        sd = (Decimal(row["population_sd"]) * scale).quantize(quantum, rounding=ROUND_HALF_UP)
        return f"{mean:.{decimals}f}±{sd:.{decimals}f}"
    if spec["format"] == "mean_pm_sample":
        sd = (Decimal(row["sample_sd"]) * scale).quantize(quantum, rounding=ROUND_HALF_UP)
        return f"{mean:.{decimals}f}±{sd:.{decimals}f}"
    raise ValueError(f"unknown format: {spec['format']}")


def check_claims(summary_rows, checks, root):
    output = []
    for check in checks:
        path = root / check["file"]
        with path.open() as handle:
            lines = [line.rstrip("\n") for line in handle]
        if check.get("section"):
            starts = [index for index, line in enumerate(lines) if line.strip() == check["section"]]
            if len(starts) != 1:
                matching = []
            else:
                start = starts[0] + 1
                level = len(check["section"]) - len(check["section"].lstrip("#"))
                end = len(lines)
                for index in range(start, len(lines)):
                    stripped = lines[index].lstrip()
                    if stripped.startswith("#"):
                        next_level = len(stripped) - len(stripped.lstrip("#"))
                        if next_level <= level:
                            end = index
                            break
                matching = [line for line in lines[start:end] if check["line_contains"] in line]
        else:
            matching = [line for line in lines if check["line_contains"] in line]
        if len(matching) != 1:
            output.append({
                "id": check["id"], "file": check["file"],
                "line_contains": check["line_contains"], "column": "",
                "expected": "", "observed": "", "status": "line_not_unique",
                "source_exp_name": "", "source_metric": "", "source_seed_count": "",
            })
            continue
        cells = [cell.strip() for cell in matching[0].strip().strip("|").split("|")]
        for column, spec in check["cells"].items():
            column = int(column)
            row = _find_summary(summary_rows, spec)
            expected = _format_expected(row, spec)
            observed = cells[column] if column < len(cells) else ""
            status = "match" if _normalize_cell(observed) == _normalize_cell(expected) else "mismatch"
            output.append({
                "id": check["id"], "file": check["file"],
                "line_contains": check["line_contains"], "column": column,
                "expected": expected, "observed": observed, "status": status,
                "source_exp_name": spec["exp_name"], "source_metric": spec["metric"],
                "source_seed_count": row["seed_count"],
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/audit/grouped_metric_summary.csv")
    parser.add_argument("--claims", default="config/audit/document_claims.yaml")
    parser.add_argument("--output", default="results/audit/document_consistency.csv")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    with open(args.summary, newline="") as handle:
        rows = list(csv.DictReader(handle))
    with open(args.claims) as handle:
        checks = yaml.safe_load(handle)["checks"]
    output = check_claims(rows, checks, Path(args.root))
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)

    counts = {status: sum(row["status"] == status for row in output) for status in {row["status"] for row in output}}
    print(json.dumps({"cell_count": len(output), "status_counts": counts, "output": args.output}, indent=2))
    if any(row["status"] != "match" for row in output):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
