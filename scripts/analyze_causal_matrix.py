"""Aggregate causal-matrix probe results and fit closed-form operators.

The input directory is produced by ``scripts/warmstart_causal.py``.  Each
probe directory contains a ``results.json`` plus final InnerNet state dicts.
This script keeps incomplete conditions explicit, verifies the expected
5-seed matrix, and summarizes both validation PPL and learned-surface fits.
"""
import argparse
import csv
import importlib.util
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "distill_innernet",
    os.path.join(os.path.dirname(__file__), "distill_innernet.py"),
)
_distill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_distill)

FAMILIES = ("mult", "swiglu", "swiglu_sym", "poly2", "poly3")
EXPECTED_SEEDS = tuple(range(42, 47))
EXPECTED = {
    "bilinear": {
        "joint": ("random", "multiply"),
        "frozen": ("random", "multiply"),
    },
    "swiglu": {"joint": ("random", "identity", "multiply", "swiglu")},
}


def _sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _checkpoint_for(result_path, checkpoint):
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if path.exists():
        return path
    local = result_path.parent / path.name
    return local if local.exists() else None


def analyze_result_file(result_path, grid_n=200, surface_range=5.0):
    result_path = Path(result_path)
    with result_path.open() as handle:
        payload = json.load(handle)
    mode = "frozen" if payload["freeze"] else "joint"
    rows = []
    for init_name, condition in sorted(payload.get("conditions", {}).items()):
        ppls = condition.get("ppl", [])
        checkpoint = _checkpoint_for(result_path, condition.get("checkpoint", ""))
        row = {
            "host": payload["host"],
            "mode": mode,
            "seed": int(payload["seed"]),
            "init": init_name,
            "status": condition.get("status", "unknown"),
            "host_ppl": float(payload["host_ppl"]),
            "best_ppl": float(min(ppls)) if ppls else condition.get("best_ppl"),
            "final_ppl": float(ppls[-1]) if ppls else None,
            "n_epochs": len(ppls),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "result_file": str(result_path),
        }
        if checkpoint:
            net = _distill.load_innernet(checkpoint)
            a, b, z = _distill.sample_surface(
                net, -surface_range, surface_range, grid_n
            )
            fits = {family: _distill.fit(family, a, b, z) for family in FAMILIES}
            row["fits"] = fits
            row["mult_r2"] = fits["mult"]["r2"]
            row["swiglu_r2"] = fits["swiglu"]["r2"]
            row["poly3_r2"] = fits["poly3"]["r2"]
            row["operator"] = (
                "multiply" if row["mult_r2"] > row["swiglu_r2"] else "swiglu"
            )
            row["operator_r2_margin"] = abs(row["mult_r2"] - row["swiglu_r2"])
        else:
            row.update({
                "fits": None,
                "mult_r2": None,
                "swiglu_r2": None,
                "poly3_r2": None,
                "operator": None,
                "operator_r2_margin": None,
            })
        rows.append(row)
    return rows


def expected_keys():
    return {
        (host, mode, seed, init_name)
        for host, modes in EXPECTED.items()
        for mode, init_names in modes.items()
        for seed in EXPECTED_SEEDS
        for init_name in init_names
    }


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["host"], row["mode"], row["init"])].append(row)

    groups = []
    for (host, mode, init_name), group_rows in sorted(grouped.items()):
        complete = [r for r in group_rows if r["best_ppl"] is not None]
        fitted = [r for r in group_rows if r["mult_r2"] is not None]
        best = [r["best_ppl"] for r in complete]
        host_ppl = [r["host_ppl"] for r in complete]
        mult = [r["mult_r2"] for r in fitted]
        swiglu = [r["swiglu_r2"] for r in fitted]
        groups.append({
            "host": host,
            "mode": mode,
            "init": init_name,
            "n": len(complete),
            "seeds": [r["seed"] for r in complete],
            "best_ppl_mean": float(np.mean(best)) if best else None,
            "best_ppl_sample_sd": _sample_sd(best),
            "host_ppl_mean": float(np.mean(host_ppl)) if host_ppl else None,
            "ppl_delta_mean": (
                float(np.mean(np.asarray(best) - np.asarray(host_ppl))) if best else None
            ),
            "mult_r2_mean": float(np.mean(mult)) if mult else None,
            "mult_r2_sample_sd": _sample_sd(mult),
            "swiglu_r2_mean": float(np.mean(swiglu)) if swiglu else None,
            "swiglu_r2_sample_sd": _sample_sd(swiglu),
            "operator_votes": {
                "multiply": sum(r["operator"] == "multiply" for r in fitted),
                "swiglu": sum(r["operator"] == "swiglu" for r in fitted),
            },
        })

    observed = {
        (r["host"], r["mode"], r["seed"], r["init"])
        for r in rows
        if r["best_ppl"] is not None and r["checkpoint"] is not None
    }
    missing = sorted(expected_keys() - observed)
    unexpected = sorted(observed - expected_keys())
    return {
        "expected_conditions": len(expected_keys()),
        "complete_conditions": len(observed & expected_keys()),
        "missing_conditions": [list(key) for key in missing],
        "unexpected_conditions": [list(key) for key in unexpected],
        "groups": groups,
    }


def write_csv(path, rows):
    fields = [
        "host", "mode", "seed", "init", "status", "host_ppl", "best_ppl",
        "final_ppl", "n_epochs", "operator", "operator_r2_margin", "mult_r2",
        "swiglu_r2", "poly3_r2", "checkpoint", "result_file",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="causal-matrix root containing probes/*/results.json")
    parser.add_argument("--out", default="results/audit/causal_matrix_summary.json")
    parser.add_argument("--csv", default="results/audit/causal_matrix_conditions.csv")
    parser.add_argument("--grid-n", type=int, default=200)
    parser.add_argument("--surface-range", type=float, default=5.0)
    args = parser.parse_args()

    result_files = sorted(Path(args.root).glob("probes/*/results.json"))
    rows = []
    for result_file in result_files:
        rows.extend(analyze_result_file(result_file, args.grid_n, args.surface_range))
    summary = summarize(rows)
    output = {
        "root": str(Path(args.root)),
        "result_files": [str(path) for path in result_files],
        "conditions": rows,
        "summary": summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(output, handle, indent=2)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)

    print(
        f"Causal matrix: {summary['complete_conditions']}/"
        f"{summary['expected_conditions']} complete conditions"
    )
    for group in summary["groups"]:
        print(
            f"{group['host']:<8} {group['mode']:<6} {group['init']:<8} "
            f"n={group['n']} PPL={group['best_ppl_mean']} "
            f"votes={group['operator_votes']}"
        )
    print(f"Wrote {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
