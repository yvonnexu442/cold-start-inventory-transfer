#!/usr/bin/env python3
"""Finalize the frozen service-parts authority table and paper summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ai_darld_v3"
KEYS = [
    "dataset", "target_id", "cutoff", "lead_time_regime",
    "capacity_regime", "shortage_holding_ratio",
]
COMMON = KEYS + [
    "method", "actual_demand", "selected_level", "total_cost", "fill_rate_proxy",
]
GLOBAL_METHODS = ["global_quantile", "zig_mc_catboost_adapted"]
PRIMARY = "factorized_similarity_residual"
POLICY_METHODS = [
    "anchored_no_calibration_fixed",
    "anchored_no_contraction_refit",
    "factorized_no_contraction",
    "factorized_no_occurrence_calibration",
    "factorized_similarity_residual",
    "factorized_transfer",
    "shared_hurdle",
    "shared_similarity_residual",
    "similarity_complete",
    "single_complete",
    "uniform_complete",
]
SEED = 20261003


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(frame: pd.DataFrame, label: str) -> None:
    missing = sorted(set(COMMON) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    duplicate = frame.duplicated(KEYS + ["method"], keep=False)
    if duplicate.any():
        raise ValueError(f"{label} has {int(duplicate.sum())} duplicate evaluation keys")


def _comparison(reference: pd.DataFrame, comparator: pd.DataFrame) -> dict[str, float | int]:
    left = reference[KEYS + ["actual_demand", "total_cost"]]
    right = comparator[KEYS + ["actual_demand", "total_cost"]]
    joined = left.merge(right, on=KEYS, suffixes=("_reference", "_comparator"), validate="one_to_one")
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError("comparison has missing matched evaluation keys")
    if not np.allclose(joined.actual_demand_reference, joined.actual_demand_comparator):
        raise ValueError("comparison pairs different realized demand")
    target = joined.assign(
        difference=joined.total_cost_reference - joined.total_cost_comparator
    ).groupby("target_id").difference.mean().to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(target, size=(4000, len(target)), replace=True).mean(axis=1)
    return {
        "matched_rows": len(joined),
        "target_clusters": len(target),
        "difference": float(target.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-input", type=Path,
        default=ROOT / "outputs/runs/ai_darld_v3/historical_support_v4_results.parquet",
    )
    parser.add_argument(
        "--intermediate-authority", type=Path,
        default=ROOT / "outputs/runs/ai_darld_v3/corrected_authority_results.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/runs/ai_darld_v3/historical_support_v4_authority_results.parquet",
    )
    args = parser.parse_args()
    for path in (args.policy_input, args.intermediate_authority):
        if not path.exists():
            raise FileNotFoundError(f"required formal input does not exist: {path}")

    policy = pd.read_parquet(args.policy_input)
    baseline = pd.read_parquet(args.intermediate_authority)
    missing = sorted(set(COMMON + ["perturbation"]) - set(policy.columns))
    if missing:
        raise ValueError(f"policy input is missing columns: {missing}")
    _check(baseline, "intermediate authority")
    clean = policy[
        policy.perturbation.eq("none") & policy.method.isin(POLICY_METHODS)
    ][COMMON].copy()
    _check(clean, "clean policy input")
    globals_only = baseline[baseline.method.isin(GLOBAL_METHODS)][COMMON].copy()
    authority = pd.concat([clean, globals_only], ignore_index=True)
    _check(authority, "final authority")
    counts = authority.groupby(["dataset", "method"]).size().unstack(fill_value=0)
    if counts.empty or counts.nunique(axis=1).max() != 1:
        raise ValueError(f"methods do not share the formal operating grid:\n{counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    authority.to_parquet(args.output, index=False)
    summary = authority.groupby(["dataset", "method"], as_index=False).agg(
        rows=("total_cost", "size"), targets=("target_id", "nunique"),
        mean_cost=("total_cost", "mean"), service=("fill_rate_proxy", "mean"),
        tail_cost=("total_cost", lambda x: x[x >= x.quantile(0.9)].mean()),
    )
    summary.to_csv(OUT / "historical_support_v4_authority_summary.csv", index=False)
    comparisons: list[dict[str, object]] = []
    for dataset in sorted(authority.dataset.unique()):
        data = authority[authority.dataset.eq(dataset)]
        reference = data[data.method.eq(PRIMARY)]
        if reference.empty:
            raise ValueError(f"{dataset} has no {PRIMARY} rows")
        for method in sorted(set(data.method) - {PRIMARY}):
            comparisons.append({
                "dataset": dataset, "reference": PRIMARY, "comparator": method,
                **_comparison(reference, data[data.method.eq(method)]),
            })
    pd.DataFrame(comparisons).to_csv(
        OUT / "historical_support_v4_authority_comparisons.csv", index=False
    )
    manifest = {
        "schema": "historical-support-v4-authority-reproduction-v2",
        "output": str(args.output.relative_to(ROOT)),
        "inputs": {
            str(args.policy_input.relative_to(ROOT)): _sha256(args.policy_input),
            str(args.intermediate_authority.relative_to(ROOT)): _sha256(args.intermediate_authority),
        },
        "primary_policy": PRIMARY,
        "policy_methods": POLICY_METHODS,
        "global_methods_from_intermediate_authority": GLOBAL_METHODS,
        "key_columns": KEYS,
        "matched_rows_by_dataset_method": counts.to_dict(),
        "bootstrap": "target-clustered percentile, 4000 draws, seed 20261003",
    }
    (OUT / "historical_support_v4_authority_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
