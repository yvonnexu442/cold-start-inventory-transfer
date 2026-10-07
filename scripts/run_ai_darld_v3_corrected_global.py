#!/usr/bin/env python3
# ruff: noqa: E402
"""Run corrected rolling-global and CatBoost ZIG--MC baselines on the V3 cohort."""

from __future__ import annotations

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ai_darld_v3_factorized import CORRECTED_EVALUATION_TARGETS, _population

from cold_start_replenishment.data.spdf_pilot import parse_braf, parse_man, redacted_id
from cold_start_replenishment.evaluation.global_baselines import _global_and_block_results


def main() -> None:
    started = time.time()
    full = yaml.safe_load((ROOT / "configs/full_scale.yaml").read_text())
    extension = yaml.safe_load((ROOT / "configs/targeted_acceptance_extension.yaml").read_text())
    all_rows = []
    counts = {}
    for dataset in (parse_man(), parse_braf()):
        targets, eligible = _population(dataset, full)
        targets = targets[: min(CORRECTED_EVALUATION_TARGETS, len(targets))]
        target_ids = {
            redacted_id(dataset.name, dataset.metadata.iloc[int(source)].item_id)
            for source in targets
        }
        checkpoint = pd.read_parquet(
            ROOT / f"outputs/full_scale/checkpoints/{dataset.name.lower()}_reliability.parquet"
        )
        scored = checkpoint[checkpoint.target_id.astype(str).isin(target_ids)].copy()
        global_rows, _, _ = _global_and_block_results(
            dataset, full, extension, scored, targets, eligible
        )
        global_rows.to_parquet(
            ROOT / f"outputs/runs/ai_darld_v3/corrected_global_{dataset.name.lower()}.parquet",
            index=False,
        )
        all_rows.append(global_rows)
        counts[dataset.name] = int(len(global_rows))
    results = pd.concat(all_rows, ignore_index=True)
    destination = ROOT / "outputs/runs/ai_darld_v3/corrected_global_results.parquet"
    results.to_parquet(destination, index=False)
    manifest = {
        "manifest_schema": "corrected-global-v1",
        "solver": "exact_weighted_empirical_moq_minimum_fixed_cost_v2",
        "targets_per_dataset": CORRECTED_EVALUATION_TARGETS,
        "methods": sorted(results.method.unique().tolist()),
        "rows_by_dataset": counts,
        "rolling_origins_per_part": 4,
        "horizons": ["native_capped", "half_native_sensitivity"],
        "cost_ratios": extension["matched_decomposition"]["shortage_to_holding_ratios"],
        "capacity_regimes": ["unconstrained", "medium", "tight"],
        "scenario_count": extension["matched_decomposition"]["scenario_count"],
        "catboost_version": version("catboost"),
        "zig_gamma_shape": 2.0,
        "zig_status": "CatBoost-adapted structural reproduction; public-study feature schema",
        "evaluation_population": "reused_frozen_targets",
        "runtime_seconds": time.time() - started,
    }
    (ROOT / "outputs/ai_darld_v3/corrected_global_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(results.groupby(["dataset", "method"], as_index=False).agg(
        rows=("total_cost", "size"), mean_cost=("total_cost", "mean"),
        fill_rate=("fill_rate_proxy", "mean")
    ).to_string(index=False))


if __name__ == "__main__":
    main()
