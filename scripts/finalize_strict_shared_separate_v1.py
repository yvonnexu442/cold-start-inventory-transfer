#!/usr/bin/env python3
"""Validate and consolidate the frozen strict shared/separate evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "ai_darld_v3" / "strict_shared_separate_corrected_v2"
CONFIG = ROOT / "configs" / "strict_shared_separate_corrected_v2.yaml"
DATASETS = ("MAN", "BRAF", "UCI", "Favorita")
BOOTSTRAP_SEED_COST = 20261005
BOOTSTRAP_SEED_SERVICE = 20261006


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(values: pd.DataFrame, column: str, draws: int, seed: int) -> tuple[float, float]:
    clustered = values.groupby("target_id", sort=True)[column].mean().to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(clustered, size=(draws, len(clustered)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _service_parts(dataset: str) -> pd.DataFrame:
    rows = pd.read_parquet(OUTPUT / f"{dataset.lower()}_rows.parquet")
    rows = rows[rows.perturbation.eq("none")].copy()
    keys = [
        "dataset",
        "target_id",
        "cutoff",
        "lead_time_regime",
        "lead_time",
        "capacity_regime",
        "shortage_holding_ratio",
        "perturbation",
    ]
    columns = {
        "total_cost": "cost",
        "fill_rate_proxy": "service",
        "selected_level": "action",
        "actual_demand": "actual",
    }
    return _paired(rows, keys, columns, dataset)


def _retail(dataset: str) -> pd.DataFrame:
    rows = pd.read_parquet(OUTPUT / f"{dataset.lower()}_rows.parquet")
    keys = [
        column
        for column in ["dataset", "target_id", "cutoff", "horizon", "cost_ratio"]
        if column in rows.columns
    ]
    columns = {"cost": "cost", "service": "service", "action": "action", "actual": "actual"}
    return _paired(rows, keys, columns, dataset)


def _paired(
    rows: pd.DataFrame, keys: list[str], columns: dict[str, str], dataset: str
) -> pd.DataFrame:
    expected = {"strict_shared", "strict_separate"}
    if set(rows.method.unique()) != expected:
        raise ValueError(f"unexpected methods for {dataset}: {sorted(rows.method.unique())}")
    duplicated = rows.duplicated(keys + ["method"], keep=False)
    if duplicated.any():
        raise ValueError(f"duplicate rows for {dataset}: {int(duplicated.sum())}")
    normalized = rows[keys + ["method", *columns]].rename(columns=columns)
    left = normalized[normalized.method.eq("strict_separate")].drop(columns="method")
    right = normalized[normalized.method.eq("strict_shared")].drop(columns="method")
    merged = left.merge(right, on=keys, suffixes=("_separate", "_shared"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f"missing counterparts for {dataset}")
    if not np.allclose(merged.actual_separate, merged.actual_shared, atol=0.0, rtol=0.0):
        raise ValueError(f"realized demand mismatch for {dataset}")
    merged["cost_difference"] = merged.cost_separate - merged.cost_shared
    merged["service_difference"] = merged.service_separate - merged.service_shared
    return merged


def _historical_comparison(
    current: pd.DataFrame, historical_path: Path, destination: Path
) -> None:
    if not historical_path.is_file():
        raise FileNotFoundError(f"historical comparison file not found: {historical_path}")
    historical = pd.read_csv(historical_path)
    old_new = historical.merge(
        current,
        on="dataset",
        suffixes=("_historical", "_current"),
        validate="one_to_one",
    )
    for metric in (
        "separate_mean_cost",
        "shared_mean_cost",
        "cost_difference",
        "relative_difference_percent",
        "ci_low",
        "ci_high",
    ):
        old_new[f"change_{metric}"] = (
            old_new[f"{metric}_current"] - old_new[f"{metric}_historical"]
        )
    old_new.to_csv(destination, index=False)


def main(historical_comparison: Path | None = None) -> None:
    protocol_path = CONFIG
    protocol = yaml.safe_load(protocol_path.read_text())
    missing = [
        dataset
        for dataset in DATASETS
        if not (OUTPUT / f"{dataset.lower()}_rows.parquet").exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing frozen strict results: {missing}")

    comparisons = []
    summaries = []
    inputs = {}
    for dataset in DATASETS:
        path = OUTPUT / f"{dataset.lower()}_rows.parquet"
        inputs[str(path.relative_to(ROOT))] = _sha256(path)
        paired = _service_parts(dataset) if dataset in {"MAN", "BRAF"} else _retail(dataset)
        draws = int(protocol["datasets"][dataset]["bootstrap_draws"])
        cost_low, cost_high = _interval(
            paired, "cost_difference", draws, BOOTSTRAP_SEED_COST
        )
        service_low, service_high = _interval(
            paired, "service_difference", draws, BOOTSTRAP_SEED_SERVICE
        )
        shared_cost = float(paired.cost_shared.mean())
        comparisons.append(
            {
                "dataset": dataset,
                "first_policy": "strict_separate",
                "comparator": "strict_shared",
                "separate_mean_cost": float(paired.cost_separate.mean()),
                "shared_mean_cost": shared_cost,
                "cost_difference": float(paired.cost_difference.mean()),
                "relative_difference_percent": float(
                    100.0 * paired.cost_difference.mean() / shared_cost
                ),
                "ci_low": cost_low,
                "ci_high": cost_high,
                "separate_service": float(paired.service_separate.mean()),
                "shared_service": float(paired.service_shared.mean()),
                "service_difference": float(paired.service_difference.mean()),
                "service_ci_low": service_low,
                "service_ci_high": service_high,
                "separate_mean_action": float(paired.action_separate.mean()),
                "shared_mean_action": float(paired.action_shared.mean()),
                "separate_zero_order_rate": float(np.mean(paired.action_separate <= 0)),
                "shared_zero_order_rate": float(np.mean(paired.action_shared <= 0)),
                "products": int(paired.target_id.nunique()),
                "matched_rows": int(len(paired)),
                "bootstrap_draws": draws,
                "bootstrap_seed_cost": BOOTSTRAP_SEED_COST,
                "bootstrap_seed_service": BOOTSTRAP_SEED_SERVICE,
            }
        )
        for method in ("strict_shared", "strict_separate"):
            suffix = "shared" if method == "strict_shared" else "separate"
            summaries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mean_cost": float(paired[f"cost_{suffix}"].mean()),
                    "service": float(paired[f"service_{suffix}"].mean()),
                    "mean_action": float(paired[f"action_{suffix}"].mean()),
                    "zero_order_rate": float(np.mean(paired[f"action_{suffix}"] <= 0)),
                    "products": int(paired.target_id.nunique()),
                    "rows": int(len(paired)),
                }
            )

    fit_files = [OUTPUT / f"{dataset.lower()}_fits.csv" for dataset in DATASETS]
    if any(not path.exists() for path in fit_files):
        raise FileNotFoundError("one or more strict fit files are missing")
    fits = pd.concat(
        [
            pd.read_csv(path).assign(dataset=dataset)
            for path, dataset in zip(fit_files, DATASETS, strict=True)
        ],
        ignore_index=True,
    )
    expected_parameters = {"strict_shared": (4, 5), "strict_separate": (8, 9)}
    parameter_rows = []
    for method, (relation, continuous) in expected_parameters.items():
        selected = fits[fits.method.eq(method)]
        if not selected.effective_relation_parameters.eq(relation).all():
            raise ValueError(f"relation parameter-count mismatch for {method}")
        if not selected.effective_continuous_parameters.eq(continuous).all():
            raise ValueError(f"continuous parameter-count mismatch for {method}")
        parameter_rows.append(
            {
                "method": method,
                "relation_coefficients": relation,
                "calibration_intercepts": 1,
                "continuous_fitted_parameters": continuous,
                "selected_contraction_hyperparameters": 1,
                "fit_rows": len(selected),
            }
        )

    pd.DataFrame(summaries).to_csv(OUTPUT / "summary.csv", index=False)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(OUTPUT / "paired_contrasts.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(OUTPUT / "parameter_counts.csv", index=False)
    historical_output = OUTPUT / "historical_comparison.csv"
    if historical_comparison is not None:
        _historical_comparison(comparison_frame, historical_comparison, historical_output)
    elif historical_output.exists():
        historical_output.unlink()
    preexecution_path = OUTPUT / "preexecution_manifest.json"
    preexecution = json.loads(preexecution_path.read_text())
    manifest = {
        "status": "completed_four_population_frozen_protocol",
        "config": str(protocol_path.relative_to(ROOT)),
        "config_sha256": _sha256(protocol_path),
        "execution_implementation_sha256": preexecution["implementation_sha256"],
        "execution_service_runner_sha256": preexecution["service_runner_sha256"],
        "execution_retail_runner_sha256": preexecution["retail_runner_sha256"],
        "preexecution_manifest_sha256": _sha256(preexecution_path),
        "inputs": inputs,
        "fit_inputs": {
            str(path.relative_to(ROOT)): _sha256(path) for path in fit_files
        },
        "runtime_manifests": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (OUTPUT / "run_manifest.json", OUTPUT / "retail_run_manifest.json")
            if path.exists()
        },
        "datasets": list(DATASETS),
        "pairing": "one-to-one on complete dataset-specific evaluation key",
        "realized_demand_identity_checked": True,
        "bootstrap": {
            "unit": "target product with repeated contexts retained",
            "cost_seed": BOOTSTRAP_SEED_COST,
            "service_seed": BOOTSTRAP_SEED_SERVICE,
            "draws_by_dataset": {
                dataset: int(protocol["datasets"][dataset]["bootstrap_draws"])
                for dataset in DATASETS
            },
        },
        "parameter_count_identity_checked": True,
        "historical_comparison_generated": historical_comparison is not None,
        "fit_actions": {
            dataset: protocol["datasets"][dataset]["fit_action"] for dataset in DATASETS
        },
    }
    (OUTPUT / "final_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate the current four-population strict evaluation."
    )
    parser.add_argument(
        "--historical-comparison",
        type=Path,
        help="optional prior paired_contrasts.csv for a separate version comparison",
    )
    arguments = parser.parse_args()
    main(arguments.historical_comparison)
