"""Cluster-aware statistics and decision-focused weighting calibration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def clustered_bootstrap_difference(
    paired: pd.DataFrame,
    value_column: str,
    cluster_column: str = "target_id",
    repetitions: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 20260806,
) -> tuple[float, float]:
    """Bootstrap target clusters while retaining every cutoff within a target."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if value_column not in paired or cluster_column not in paired:
        raise ValueError("paired data are missing bootstrap columns")
    groups = {
        str(cluster): group[value_column].to_numpy(dtype=float)
        for cluster, group in paired.groupby(cluster_column)
    }
    if not groups:
        raise ValueError("At least one cluster is required")
    cluster_ids = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(random_seed)
    draws = np.zeros(repetitions, dtype=float)
    for repetition in range(repetitions):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        values = np.concatenate([groups[str(cluster)] for cluster in sampled])
        draws[repetition] = values.mean()
    alpha = (1 - confidence) / 2
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def _weights(group: pd.DataFrame, mode: str, maximum_weight: float = 0.35) -> np.ndarray:
    similarity = group["metadata_similarity"].to_numpy(dtype=float)
    reliability = group["reliability_probability"].to_numpy(dtype=float)
    support = np.log1p(group["donor_nonzero_count"].to_numpy(dtype=float))
    if mode == "similarity_only":
        raw = similarity
    elif mode == "reliability_only":
        raw = reliability * support
    elif mode.startswith("combined_temperature_"):
        temperature = {
            "combined_temperature_0_5": 0.5,
            "combined_temperature_1_0": 1.0,
            "combined_temperature_2_0": 2.0,
        }[mode]
        raw = np.power(np.clip(similarity * reliability * support, 1e-12, None), 1 / temperature)
    elif mode in {"capped_combined", "reliability_fallback"}:
        raw = similarity * reliability * support
    else:
        raise ValueError(f"Unknown weighting mode: {mode}")
    weights = raw / raw.sum() if raw.sum() else np.full(len(raw), 1 / len(raw))
    if mode in {"capped_combined", "reliability_fallback"}:
        for _ in range(len(weights) + 1):
            above = weights > maximum_weight
            if not above.any():
                break
            excess = float((weights[above] - maximum_weight).sum())
            weights[above] = maximum_weight
            below = ~above
            if below.any() and weights[below].sum() > 0:
                weights[below] += excess * weights[below] / weights[below].sum()
        weights /= weights.sum()
    return weights


def select_weighting_modes(
    validation: pd.DataFrame,
    candidates: Sequence[str],
    maximum_weight: float = 0.35,
    shortage_ratio: float = 5.0,
) -> tuple[str, str, pd.DataFrame]:
    """Select forecast and decision modes using donor-only pseudo-target validation."""
    required = {
        "pseudo_target_id",
        "metadata_similarity",
        "reliability_probability",
        "donor_nonzero_count",
        "donor_expected_lead_demand",
        "pseudo_target_actual_lead_demand",
    }
    missing = required - set(validation)
    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for mode in candidates:
        errors: list[float] = []
        costs: list[float] = []
        for _, group in validation.groupby("pseudo_target_id"):
            weights = _weights(group, mode, maximum_weight)
            donor_demand = group["donor_expected_lead_demand"].to_numpy(dtype=float)
            actual = float(group["pseudo_target_actual_lead_demand"].iloc[0])
            forecast = float(np.dot(weights, donor_demand))
            demand_order = np.argsort(donor_demand)
            ordered_demand = donor_demand[demand_order]
            ordered_weights = weights[demand_order]
            quantile_index = min(
                int(
                    np.searchsorted(
                        np.cumsum(ordered_weights),
                        shortage_ratio / (1 + shortage_ratio),
                        side="left",
                    )
                ),
                len(ordered_demand) - 1,
            )
            order = float(ordered_demand[quantile_index])
            errors.append(abs(forecast - actual))
            costs.append(max(order - actual, 0) + shortage_ratio * max(actual - order, 0))
        rows.append(
            {
                "weighting_mode": mode,
                "mean_forecast_absolute_error": float(np.mean(errors)),
                "mean_decision_cost": float(np.mean(costs)),
                "pseudo_targets": validation["pseudo_target_id"].nunique(),
            }
        )
    summary = pd.DataFrame(rows)
    forecast_mode = str(
        summary.loc[summary["mean_forecast_absolute_error"].idxmin(), "weighting_mode"]
    )
    decision_mode = str(summary.loc[summary["mean_decision_cost"].idxmin(), "weighting_mode"])
    return forecast_mode, decision_mode, summary


def paired_statistical_comparisons(
    results: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    repetitions: int,
    confidence: float,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["dataset", "target_id", "cutoff", "method"]
    target = results.groupby(grouping, as_index=False).agg(
        mean_regret=("normalized_regret", "mean"),
        tail_cost=("total_normalized_cost", lambda values: float(np.quantile(values, 0.9))),
    )
    for dataset, dataset_rows in target.groupby("dataset"):
        for left, right in comparisons:
            pivot = dataset_rows[dataset_rows["method"].isin([left, right])].pivot(
                index=["target_id", "cutoff"], columns="method", values="mean_regret"
            )
            if left not in pivot or right not in pivot:
                continue
            paired = pivot[[left, right]].dropna().reset_index()
            paired["difference"] = paired[left] - paired[right]
            low, high = clustered_bootstrap_difference(
                paired,
                "difference",
                repetitions=repetitions,
                confidence=confidence,
                random_seed=random_seed + len(rows),
            )
            differences = paired["difference"].to_numpy(dtype=float)
            cluster_means = paired.groupby("target_id")["difference"].mean()
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{left} minus {right}",
                    "target_count": paired["target_id"].nunique(),
                    "cutoff_count": paired["cutoff"].nunique(),
                    "target_cutoff_rows": len(paired),
                    "mean_paired_difference": float(differences.mean()),
                    "median_paired_difference": float(np.median(differences)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "win_rate": float((differences < 0).mean()),
                    "standardized_effect": float(cluster_means.mean() / cluster_means.std(ddof=1))
                    if cluster_means.std(ddof=1) > 0
                    else 0.0,
                    "worst_decile_difference": float(
                        differences[differences >= np.quantile(differences, 0.9)].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)
