#!/usr/bin/env python3
"""Summarize frozen operating-condition and input-quality comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/operating_condition_input_quality_v1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/ai_darld_v3/operating_condition_input_quality_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(base_seed: int, label: str) -> int:
    payload = f"{base_seed}:{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def pair_rows(
    rows: pd.DataFrame,
    *,
    key_columns: list[str],
    first_method: str,
    comparator: str,
) -> pd.DataFrame:
    selected = rows[rows["method"].isin([first_method, comparator])].copy()
    required = set(
        key_columns
        + ["method", "actual_demand", "selected_level", "total_cost", "fill_rate_proxy"]
    )
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if selected.duplicated(key_columns + ["method"]).any():
        raise ValueError("Duplicate method rows found for the declared evaluation key")
    first = selected[selected["method"].eq(first_method)].set_index(key_columns).sort_index()
    second = selected[selected["method"].eq(comparator)].set_index(key_columns).sort_index()
    if not first.index.equals(second.index):
        raise ValueError(
            "Incomplete policy pairing: "
            f"first_only={len(first.index.difference(second.index))}, "
            f"comparator_only={len(second.index.difference(first.index))}"
        )
    if not np.array_equal(
        first["actual_demand"].to_numpy(), second["actual_demand"].to_numpy()
    ):
        raise ValueError("Paired policy rows use different realized demand")
    paired = pd.DataFrame(index=first.index)
    for column in ["total_cost", "fill_rate_proxy", "selected_level"]:
        paired[f"first_{column}"] = first[column].to_numpy(float)
        paired[f"comparator_{column}"] = second[column].to_numpy(float)
    paired["actual_demand"] = first["actual_demand"].to_numpy(float)
    return paired.reset_index()


def pair_conditions(
    rows: pd.DataFrame,
    *,
    key_columns: list[str],
    perturbation: str,
    clean_condition: str,
) -> pd.DataFrame:
    selected = rows[rows["perturbation"].isin([perturbation, clean_condition])].copy()
    if selected.duplicated(key_columns + ["perturbation"]).any():
        raise ValueError(f"Duplicate condition rows for {perturbation}")
    changed = selected[selected["perturbation"].eq(perturbation)].set_index(key_columns).sort_index()
    clean = selected[selected["perturbation"].eq(clean_condition)].set_index(key_columns).sort_index()
    if not changed.index.equals(clean.index):
        raise ValueError(
            f"Incomplete clean pairing for {perturbation}: "
            f"changed_only={len(changed.index.difference(clean.index))}, "
            f"clean_only={len(clean.index.difference(changed.index))}"
        )
    if not np.array_equal(
        changed["actual_demand"].to_numpy(), clean["actual_demand"].to_numpy()
    ):
        raise ValueError(f"Realized demand differs from clean rows for {perturbation}")
    return pd.DataFrame(
        {
            **{column: changed.index.get_level_values(column) for column in key_columns},
            "changed_cost": changed["total_cost"].to_numpy(float),
            "clean_cost": clean["total_cost"].to_numpy(float),
        }
    )


def clustered_interval(
    paired: pd.DataFrame,
    *,
    difference_column: str,
    target_column: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    product = paired.groupby(target_column, sort=True, observed=True).agg(
        value_sum=(difference_column, "sum"), row_count=(difference_column, "size")
    )
    if product.empty:
        raise ValueError("No target products available for bootstrap")
    point = float(paired[difference_column].mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(product), size=(draws, len(product)))
    sums = product["value_sum"].to_numpy()[indices].sum(axis=1)
    counts = product["row_count"].to_numpy()[indices].sum(axis=1)
    estimates = sums / counts
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)


def ratio_analysis(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    first = config["authoritative_input"]["first_policy"]
    comparator = config["authoritative_input"]["comparator"]
    draws = int(config["uncertainty"]["draws"])
    seed = int(config["uncertainty"]["seed"])
    key = [
        "dataset",
        "target_id",
        "cutoff",
        "lead_time_regime",
        "capacity_regime",
        "shortage_holding_ratio",
    ]
    records: list[dict[str, Any]] = []
    selected = rows[
        rows["dataset"].isin(config["authoritative_input"]["datasets"])
        & rows["shortage_holding_ratio"].isin(
            config["ratio_stratification"]["shortage_to_holding_ratios"]
        )
    ]
    paired_all = pair_rows(
        selected, key_columns=key, first_method=first, comparator=comparator
    )
    paired_all["cost_difference"] = (
        paired_all["first_total_cost"] - paired_all["comparator_total_cost"]
    )
    for (dataset, ratio), group in paired_all.groupby(
        ["dataset", "shortage_holding_ratio"], sort=True, observed=True
    ):
        point, low, high = clustered_interval(
            group,
            difference_column="cost_difference",
            target_column="target_id",
            draws=draws,
            seed=derived_seed(seed, f"ratio:{dataset}:{ratio}"),
        )
        records.append(
            {
                "dataset": dataset,
                "shortage_holding_ratio": float(ratio),
                "first_policy": first,
                "comparator": comparator,
                "cost_difference": point,
                "ci_low": low,
                "ci_high": high,
                "first_service": float(group["first_fill_rate_proxy"].mean()),
                "comparator_service": float(group["comparator_fill_rate_proxy"].mean()),
                "first_mean_order": float(group["first_selected_level"].mean()),
                "comparator_mean_order": float(group["comparator_selected_level"].mean()),
                "products": int(group["target_id"].nunique()),
                "matched_rows": int(len(group)),
                "bootstrap_draws": draws,
                "bootstrap_seed": derived_seed(seed, f"ratio:{dataset}:{ratio}"),
            }
        )
    result = pd.DataFrame.from_records(records)
    expected = len(config["authoritative_input"]["datasets"]) * len(
        config["ratio_stratification"]["shortage_to_holding_ratios"]
    )
    if len(result) != expected:
        raise ValueError(f"Expected {expected} ratio strata, found {len(result)}")

    # The ratio strata must recover the formal full-grid means and paired difference.
    for dataset, group in paired_all.groupby("dataset", observed=True):
        weighted = result[result["dataset"].eq(dataset)]
        recovered = np.average(weighted["cost_difference"], weights=weighted["matched_rows"])
        if not np.isclose(recovered, group["cost_difference"].mean(), atol=1e-10):
            raise ValueError(f"Ratio strata do not recover the {dataset} full-grid difference")
    return result


def degradation_analysis(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    spec = config["input_quality_sensitivity"]
    slice_spec = spec["slice"]
    selected = rows[
        rows["dataset"].isin(config["authoritative_input"]["datasets"])
        & rows["method"].isin(spec["policies"])
        & rows["lead_time_regime"].eq(slice_spec["lead_time_regime"])
        & rows["shortage_holding_ratio"].eq(slice_spec["shortage_to_holding_ratio"])
        & rows["capacity_regime"].eq(slice_spec["capacity_regime"])
    ].copy()
    key = [
        "dataset",
        "target_id",
        "cutoff",
        "lead_time_regime",
        "lead_time",
        "capacity_regime",
        "shortage_holding_ratio",
        "method",
    ]
    draws = int(config["uncertainty"]["draws"])
    seed = int(config["uncertainty"]["seed"])
    records: list[dict[str, Any]] = []
    for dataset in config["authoritative_input"]["datasets"]:
        for method in spec["policies"]:
            method_rows = selected[
                selected["dataset"].eq(dataset) & selected["method"].eq(method)
            ]
            for perturbation in spec["perturbations"]:
                paired = pair_conditions(
                    method_rows,
                    key_columns=key,
                    perturbation=perturbation,
                    clean_condition=spec["clean_condition"],
                )
                paired["cost_change"] = paired["changed_cost"] - paired["clean_cost"]
                bootstrap_seed = derived_seed(seed, f"degradation:{dataset}:{method}:{perturbation}")
                point, low, high = clustered_interval(
                    paired,
                    difference_column="cost_change",
                    target_column="target_id",
                    draws=draws,
                    seed=bootstrap_seed,
                )
                records.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "perturbation": perturbation,
                        "cost_change_from_clean": point,
                        "ci_low": low,
                        "ci_high": high,
                        "clean_mean_cost": float(paired["clean_cost"].mean()),
                        "perturbed_mean_cost": float(paired["changed_cost"].mean()),
                        "products": int(paired["target_id"].nunique()),
                        "matched_rows": int(len(paired)),
                        "bootstrap_draws": draws,
                        "bootstrap_seed": bootstrap_seed,
                    }
                )
    result = pd.DataFrame.from_records(records)
    expected = (
        len(config["authoritative_input"]["datasets"])
        * len(spec["policies"])
        * len(spec["perturbations"])
    )
    if len(result) != expected:
        raise ValueError(f"Expected {expected} degradation contrasts, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = yaml.safe_load(config_path.read_text())
    input_path = ROOT / config["authoritative_input"]["path"]
    authority = pd.read_parquet(input_path)
    detailed_path = ROOT / config["authoritative_input"]["detailed_path"]
    detailed = pd.read_parquet(detailed_path)

    ratio = ratio_analysis(authority, config)
    degradation = degradation_analysis(detailed, config)
    output.mkdir(parents=True, exist_ok=True)
    ratio_path = output / "ratio_stratified.csv"
    degradation_path = output / "input_quality_sensitivity.csv"
    ratio.to_csv(ratio_path, index=False, float_format="%.10g")
    degradation.to_csv(degradation_path, index=False, float_format="%.10g")
    manifest = {
        "analysis_id": config["analysis_id"],
        "evidence_role": config["interpretation"]["evidence_role"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "inputs": {
            str(input_path.relative_to(ROOT)): sha256(input_path),
            str(detailed_path.relative_to(ROOT)): sha256(detailed_path),
        },
        "outputs": {
            str(ratio_path.relative_to(ROOT)): sha256(ratio_path),
            str(degradation_path.relative_to(ROOT)): sha256(degradation_path),
        },
        "uncertainty": config["uncertainty"],
        "ratio_rows": len(ratio),
        "degradation_rows": len(degradation),
        "notes": [
            "No policy was refit.",
            "Intervals condition on the existing fitted ensemble.",
            "Repeated evaluation contexts remain grouped within target product.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
