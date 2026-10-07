#!/usr/bin/env python3
# ruff: noqa: E402
"""Decompose frozen formal policy costs without refitting any policy."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cold_start_replenishment.data.spdf_pilot import (  # noqa: E402
    parse_braf,
    parse_man,
    redacted_id,
)
from cold_start_replenishment.evaluation.acceptance_enhancement import (
    _eligible_sources,  # noqa: E402
)
from cold_start_replenishment.evaluation.sprint2 import _operational_values  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cost_components(
    action: pd.Series,
    actual: pd.Series,
    holding_rate: pd.Series,
    shortage_ratio: pd.Series,
    fixed_rate: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute all three cost components independently from operating inputs."""
    holding = holding_rate * np.maximum(action - actual, 0.0)
    shortage = holding_rate * shortage_ratio * np.maximum(actual - action, 0.0)
    fixed = fixed_rate * (action > 0).astype(float)
    return holding, shortage, fixed


def _require_complete_pairs(
    rows: pd.DataFrame, *, methods: list[str] | tuple[str, ...], keys: list[str], dataset: str
) -> None:
    """Require one row per method on every evaluation key."""
    selected = rows[rows.method.isin(methods)]
    duplicates = selected.duplicated(keys + ["method"], keep=False)
    if duplicates.any():
        raise ValueError(f"duplicate method rows in {dataset}: {int(duplicates.sum())}")
    counts = selected.groupby(keys, dropna=False).method.nunique()
    incomplete = counts[counts.ne(len(methods))]
    if not incomplete.empty:
        raise ValueError(f"incomplete method pairs in {dataset}: {len(incomplete)} keys")
    expected = len(counts) * len(methods)
    if len(selected) != expected:
        raise ValueError(
            f"unexpected paired row count in {dataset}: {len(selected)} != {expected}"
        )


def _service_parts() -> pd.DataFrame:
    path = ROOT / "outputs/runs/ai_darld_v3/historical_support_v4_authority_results.parquet"
    rows = pd.read_parquet(path)
    full = yaml.safe_load((ROOT / "configs/full_scale.yaml").read_text())
    selected = {
        "BRAF": ("factorized_similarity_residual", "similarity_complete"),
        "MAN": ("factorized_similarity_residual", "similarity_complete"),
    }
    output = []
    for name, methods in selected.items():
        dataset = parse_braf() if name == "BRAF" else parse_man()
        section = full["datasets"][name]
        eligible = _eligible_sources(dataset, section)
        lookup = {}
        for source in range(len(dataset.metadata)):
            identifier = redacted_id(name, dataset.metadata.iloc[source].item_id)
            holding, fixed, _ = _operational_values(dataset, source, eligible, section)
            lookup[str(identifier)] = (float(holding), float(fixed))
        subset = rows[rows.dataset.eq(name) & rows.method.isin(methods)].copy()
        pair_keys = [
            "target_id",
            "cutoff",
            "lead_time_regime",
            "shortage_holding_ratio",
            "capacity_regime",
        ]
        _require_complete_pairs(
            subset, methods=list(methods), keys=pair_keys, dataset=name
        )
        holding_lookup = {key: value[0] for key, value in lookup.items()}
        fixed_lookup = {key: value[1] for key, value in lookup.items()}
        subset["holding_rate"] = subset.target_id.astype(str).map(holding_lookup)
        subset["fixed_rate"] = subset.target_id.astype(str).map(fixed_lookup)
        if subset[["holding_rate", "fixed_rate"]].isna().any().any():
            raise ValueError(f"missing operating-cost metadata for {name}")
        holding, shortage, fixed = _cost_components(
            subset.selected_level,
            subset.actual_demand,
            subset.holding_rate,
            subset.shortage_holding_ratio,
            subset.fixed_rate,
        )
        subset["holding_component"] = holding
        subset["shortage_component"] = shortage
        subset["fixed_component"] = fixed
        if not np.allclose(
            subset.total_cost,
            subset.holding_component + subset.shortage_component + subset.fixed_component,
            atol=1e-8,
        ):
            raise ValueError(f"component identity failed for {name}")
        subset["zero_order"] = subset.selected_level <= 0
        subset["shortage_quantity"] = np.maximum(
            subset.actual_demand - subset.selected_level, 0.0
        )
        subset["satisfaction_positive"] = np.where(
            subset.actual_demand > 0,
            np.minimum(subset.selected_level, subset.actual_demand) / subset.actual_demand,
            np.nan,
        )
        for method, group in subset.groupby("method"):
            output.append(
                {
                    "dataset": name,
                    "method": method,
                    "comparison_role": "first" if method == methods[0] else "comparator",
                    "mean_total_cost": group.total_cost.mean(),
                    "mean_holding_cost": group.holding_component.mean(),
                    "mean_shortage_cost": group.shortage_component.mean(),
                    "mean_fixed_order_cost": group.fixed_component.mean(),
                    "mean_shortage_quantity": group.shortage_quantity.mean(),
                    "mean_action": group.selected_level.mean(),
                    "zero_order_rate": group.zero_order.mean(),
                    "service_point_estimate": group.fill_rate_proxy.mean(),
                    "positive_demand_satisfaction": group.satisfaction_positive.mean(),
                    "rows": len(group),
                    "products": group.target_id.nunique(),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return pd.DataFrame(output)


def _uci() -> pd.DataFrame:
    path = ROOT / "outputs/ai_darld_v3/uci_confirmation_v3_rows.parquet"
    rows = pd.read_parquet(path)
    methods = [
        "factorized_similarity_residual",
        "shared_similarity_residual",
        "complete_similarity",
    ]
    subset = rows[rows.method.isin(methods)].copy()
    pair_keys = ["target_id", "cutoff", "horizon", "cost_ratio"]
    _require_complete_pairs(subset, methods=methods, keys=pair_keys, dataset="UCI")
    zeros = pd.Series(0.0, index=subset.index)
    holding, shortage, fixed = _cost_components(
        subset.action,
        subset.actual,
        pd.Series(1.0, index=subset.index),
        subset.cost_ratio,
        zeros,
    )
    subset["holding_component"] = holding
    subset["shortage_component"] = shortage
    subset["fixed_component"] = fixed
    if not np.allclose(
        subset.cost, subset.holding_component + subset.shortage_component, atol=1e-8
    ):
        raise ValueError("component identity failed for UCI")
    subset["zero_order"] = subset.action <= 0
    subset["shortage_quantity"] = np.maximum(subset.actual - subset.action, 0.0)
    subset["satisfaction_positive"] = np.where(
        subset.actual > 0, np.minimum(subset.action, subset.actual) / subset.actual, np.nan
    )
    return (
        subset.groupby("method", as_index=False)
        .agg(
            mean_total_cost=("cost", "mean"),
            mean_holding_cost=("holding_component", "mean"),
            mean_shortage_cost=("shortage_component", "mean"),
            mean_fixed_order_cost=("fixed_component", "mean"),
            mean_shortage_quantity=("shortage_quantity", "mean"),
            mean_action=("action", "mean"),
            zero_order_rate=("zero_order", "mean"),
            service_point_estimate=("service", "mean"),
            positive_demand_satisfaction=("satisfaction_positive", "mean"),
            rows=("cost", "size"),
            products=("target_id", "nunique"),
        )
        .assign(
            dataset="UCI",
            comparison_role=lambda x: np.where(
                x.method.eq("factorized_similarity_residual"), "first", "comparator"
            ),
            source_file=str(path.relative_to(ROOT)),
        )
    )


def main() -> None:
    output = ROOT / "outputs/ai_darld_v3/operational_cost_decomposition_v1"
    output.mkdir(parents=True, exist_ok=True)
    result = pd.concat([_service_parts(), _uci()], ignore_index=True)
    result.to_csv(output / "summary.csv", index=False)
    uci_config = yaml.safe_load(
        (ROOT / "configs/uci_online_retail_ii_external.yaml").read_text()
    )
    favorita_config = yaml.safe_load(
        (ROOT / "configs/favorita_untouched_confirmation_v1.yaml").read_text()
    )
    if float(uci_config["tasks"]["fixed_order_cost"]) != 0.0:
        raise ValueError("UCI formal protocol no longer has zero fixed-order cost")
    if float(favorita_config["tasks"]["fixed_order_cost"]) != 0.0:
        raise ValueError("Favorita formal protocol no longer has zero fixed-order cost")
    manifest = {
        "status": "completed_from_frozen_row_level_outputs",
        "no_policy_refitting": True,
        "cost_identity_checked": True,
        "fixed_cost_computed_from_operating_metadata": True,
        "zero_action_has_zero_fixed_cost_checked": True,
        "zero_fixed_cost_protocols_checked": ["BRAF", "UCI", "Favorita"],
        "complete_evaluation_key_pairs_checked": True,
        "service_is_point_estimate": True,
        "comparisons": {
            "BRAF": "factorized_similarity_residual vs similarity_complete",
            "MAN": "factorized_similarity_residual vs similarity_complete",
            "UCI": "factorized_similarity_residual vs shared_similarity_residual and complete_similarity",
        },
        "inputs": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                ROOT
                / "outputs/runs/ai_darld_v3/historical_support_v4_authority_results.parquet",
                ROOT / "outputs/ai_darld_v3/uci_confirmation_v3_rows.parquet",
            )
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
