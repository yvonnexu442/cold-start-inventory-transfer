#!/usr/bin/env python3
"""Consolidate the pre-specified strict finite-search sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/ai_darld_v3/strict_search_stability_v1"
CONFIG = ROOT / "configs/strict_search_stability_v1.yaml"
DATASETS = ("MAN", "BRAF", "UCI", "Favorita")
METHODS = ("strict_shared", "strict_separate")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys(frame: pd.DataFrame, *, include_seed: bool) -> list[str]:
    candidates = [
        "dataset",
        "target_id",
        "cutoff",
        "lead_time_regime",
        "lead_time",
        "capacity_regime",
        "shortage_holding_ratio",
        "perturbation",
        "horizon",
        "cost_ratio",
    ]
    keys = [column for column in candidates if column in frame.columns]
    if include_seed:
        keys.append("policy_seed")
    return keys


def _paired(frame: pd.DataFrame, *, include_seed: bool) -> pd.DataFrame:
    if "perturbation" in frame:
        frame = frame[frame.perturbation.eq("none")].copy()
    keys = _keys(frame, include_seed=include_seed)
    if frame.duplicated(keys + ["method"], keep=False).any():
        raise ValueError("duplicate strict sensitivity rows")
    actual = "actual_demand" if "actual_demand" in frame else "actual"
    cost = "total_cost" if "total_cost" in frame else "cost"
    service = "fill_rate_proxy" if "fill_rate_proxy" in frame else "service"
    action = "selected_level" if "selected_level" in frame else "action"
    selected = frame[keys + ["method", actual, cost, service, action]].rename(
        columns={actual: "actual", cost: "cost", service: "service", action: "action"}
    )
    separate = selected[selected.method.eq("strict_separate")].drop(columns="method")
    shared = selected[selected.method.eq("strict_shared")].drop(columns="method")
    paired = separate.merge(
        shared, on=keys, suffixes=("_separate", "_shared"), validate="one_to_one"
    )
    if len(paired) != len(separate) or len(paired) != len(shared):
        raise ValueError("missing strict sensitivity counterpart rows")
    if not np.array_equal(paired.actual_separate, paired.actual_shared):
        raise ValueError("realized outcomes differ between sensitivity policies")
    paired["cost_difference"] = paired.cost_separate - paired.cost_shared
    paired["service_difference"] = paired.service_separate - paired.service_shared
    return paired


def _interval(frame: pd.DataFrame, draws: int, seed: int) -> tuple[float, float]:
    product = frame.groupby("target_id", sort=True).cost_difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(product, size=(draws, len(product)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def main() -> None:
    protocol = yaml.safe_load(CONFIG.read_text())
    ensemble_rows = []
    seed_rows = []
    inputs: dict[str, str] = {}
    fit_frames = []
    for budget in map(int, protocol["budgets"]):
        directory = OUTPUT / f"budget_{budget}"
        for dataset in DATASETS:
            row_path = directory / f"{dataset.lower()}_rows.parquet"
            seed_path = directory / f"{dataset.lower()}_seed_rows.parquet"
            fit_path = directory / f"{dataset.lower()}_fits.csv"
            for path in (row_path, seed_path, fit_path):
                if not path.exists():
                    raise FileNotFoundError(path)
                inputs[str(path.relative_to(ROOT))] = _sha256(path)
            paired = _paired(pd.read_parquet(row_path), include_seed=False)
            draws = int(protocol["datasets"][dataset]["bootstrap_draws"])
            low, high = _interval(paired, draws, 20261005 + budget)
            shared_cost = float(paired.cost_shared.mean())
            ensemble_rows.append(
                {
                    "budget": budget,
                    "dataset": dataset,
                    "separate_mean_cost": float(paired.cost_separate.mean()),
                    "shared_mean_cost": shared_cost,
                    "cost_difference": float(paired.cost_difference.mean()),
                    "relative_difference_percent": float(
                        100 * paired.cost_difference.mean() / shared_cost
                    ),
                    "ci_low": low,
                    "ci_high": high,
                    "separate_service": float(paired.service_separate.mean()),
                    "shared_service": float(paired.service_shared.mean()),
                    "service_difference": float(paired.service_difference.mean()),
                    "separate_mean_action": float(paired.action_separate.mean()),
                    "shared_mean_action": float(paired.action_shared.mean()),
                    "products": int(paired.target_id.nunique()),
                    "matched_rows": int(len(paired)),
                    "bootstrap_draws": draws,
                    "bootstrap_seed": 20261005 + budget,
                }
            )
            seeded = _paired(pd.read_parquet(seed_path), include_seed=True)
            for seed, group in seeded.groupby("policy_seed", sort=True):
                seed_rows.append(
                    {
                        "budget": budget,
                        "dataset": dataset,
                        "policy_seed": int(seed),
                        "cost_difference": float(group.cost_difference.mean()),
                        "service_difference": float(group.service_difference.mean()),
                        "separate_mean_cost": float(group.cost_separate.mean()),
                        "shared_mean_cost": float(group.cost_shared.mean()),
                        "products": int(group.target_id.nunique()),
                        "matched_rows": int(len(group)),
                    }
                )
            fit_frames.append(pd.read_csv(fit_path).assign(budget=budget, dataset=dataset))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ensemble_rows).to_csv(OUTPUT / "ensemble_contrasts.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUTPUT / "seed_contrasts.csv", index=False)
    fits = pd.concat(fit_frames, ignore_index=True)
    fit_columns = [
        "budget",
        "dataset",
        "seed",
        "cutoff",
        "method",
        "selected_candidate",
        "contraction_alpha",
        "occurrence_logit_shift",
        "training_objective",
        "validation_objective",
        "candidate_slots",
        "unique_candidates",
        "training_evaluations",
        "validation_evaluations",
    ]
    present = [column for column in fit_columns if column in fits]
    fits[present].to_csv(OUTPUT / "fit_selection.csv", index=False)
    manifest = {
        "status": "completed_prespecified_two_budget_sensitivity",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG),
        "inputs": inputs,
        "product_clustered_interval_scope": "fixed five-seed ensemble",
        "training_randomness_scope": "reported separately in seed_contrasts.csv",
        "candidate_set_nesting_checked": True,
        "budgets": list(map(int, protocol["budgets"])),
    }
    (OUTPUT / "final_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
