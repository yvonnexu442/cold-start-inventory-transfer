#!/usr/bin/env python3
"""Combine frozen service-parts policy and global-baseline rows.

This intermediate authority file is consumed only for the rolling global
quantile and adapted ZIG--MC rows in the final historical-support authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = [
    "dataset", "target_id", "cutoff", "lead_time_regime",
    "capacity_regime", "shortage_holding_ratio",
]
COMMON = KEYS + [
    "method", "actual_demand", "selected_level", "total_cost", "fill_rate_proxy",
]
METHODS = [
    "single_complete", "uniform_complete", "similarity_complete",
    "shared_hurdle", "factorized_no_contraction",
    "factorized_no_occurrence_calibration", "factorized_transfer",
    "global_quantile", "zig_mc_catboost_adapted",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = sorted(set(COMMON) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    duplicate = frame.duplicated(KEYS + ["method"], keep=False)
    if duplicate.any():
        raise ValueError(f"{label} has {int(duplicate.sum())} duplicate evaluation keys")
    return frame


def _require_columns(frame: pd.DataFrame, label: str) -> None:
    missing = sorted(set(COMMON + ["perturbation"]) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factorized-input",
        type=Path,
        default=ROOT / "outputs/runs/ai_darld_v3/historical_support_v4_results.parquet",
    )
    parser.add_argument(
        "--global-input",
        type=Path,
        default=ROOT / "outputs/runs/ai_darld_v3/corrected_global_results.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/runs/ai_darld_v3/corrected_authority_results.parquet",
    )
    args = parser.parse_args()
    for path in (args.factorized_input, args.global_input):
        if not path.exists():
            raise FileNotFoundError(f"required formal input does not exist: {path}")

    factorized = pd.read_parquet(args.factorized_input)
    _require_columns(factorized, "policy input")
    global_rows = _validated(pd.read_parquet(args.global_input), "global input")
    clean = _validated(factorized[
        factorized.perturbation.eq("none") & factorized.method.isin(METHODS)
    ][COMMON].copy(), "clean policy input")
    global_rows = global_rows[global_rows.method.isin(METHODS)][COMMON].copy()
    authority = _validated(pd.concat([clean, global_rows], ignore_index=True), "authority")

    counts = authority.groupby(["dataset", "method"]).size().unstack(fill_value=0)
    if counts.empty or counts.nunique(axis=1).max() != 1:
        raise ValueError(f"methods do not share the formal operating grid:\n{counts}")
    demand_counts = authority.groupby(KEYS).actual_demand.nunique()
    if int(demand_counts.max()) != 1:
        raise ValueError("methods do not share realized demand on matched evaluation keys")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    authority.to_parquet(args.output, index=False)
    manifest = {
        "schema": "corrected-authority-reproduction-v2",
        "output": str(args.output.relative_to(ROOT)),
        "inputs": {
            str(args.factorized_input.relative_to(ROOT)): _sha256(args.factorized_input),
            str(args.global_input.relative_to(ROOT)): _sha256(args.global_input),
        },
        "matched_rows_by_dataset_method": counts.to_dict(),
        "key_columns": KEYS,
    }
    manifest_path = ROOT / "outputs/ai_darld_v3/corrected_authority_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output} ({len(authority)} rows)")


if __name__ == "__main__":
    main()
