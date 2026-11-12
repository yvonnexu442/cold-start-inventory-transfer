#!/usr/bin/env python3
"""Run leakage-separated exploratory matched-service evaluation on BRAF."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ai_darld_v3_direct_mixture import _features, _historical_tasks, _split, _support_matrix
from run_ai_darld_v3_factorized import _population

from cold_start_replenishment.analogs.factorized_transfer import (
    FactorizedTransferPolicy,
    complete_mixture_distribution,
    factorized_distribution,
)
from cold_start_replenishment.data.spdf_pilot import braf_similarity, parse_braf, redacted_id
from cold_start_replenishment.evaluation.acceptance_enhancement import _construct_scenarios
from cold_start_replenishment.evaluation.decision_evaluator import HeldOutDecisionEvaluator
from cold_start_replenishment.evaluation.sprint2 import _operational_values
from cold_start_replenishment.framework.objects import OperationalContext, ReplenishmentDecision
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level

CONFIG_PATH = ROOT / "configs/service_calibration_fixed_policy_braf_v2.yaml"
OUTPUT = ROOT / "outputs/ai_darld_v3/service_calibration_fixed_policy_braf_v2"
V1_CACHE = ROOT / "outputs/ai_darld_v3/matched_service_braf_exploratory_v1/policy_and_calibration_cache.joblib"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_policies(dataset, full, v3, config):
    policies: dict[int, list[FactorizedTransferPolicy]] = {}
    calibration: dict[int, list] = {}
    fit_records: list[dict] = []
    for cutoff in config["evaluation_grid"]["cutoffs"]:
        tasks, _, _, _ = _historical_tasks(
            dataset, full, v3, int(cutoff), scenario_count=120,
            random_seed=int(v3["random_seed"]),
        )
        ids = np.array(sorted({task.target_id for task in tasks}))
        ordered = np.random.default_rng(int(v3["random_seed"]) + int(cutoff)).permutation(ids)
        fit_count = int(config["fit_products_per_cutoff"])
        calibration_count = int(config["service_calibration_products_per_cutoff"])
        fit_ids = ordered[:fit_count]
        calibration_ids = set(ordered[fit_count : fit_count + calibration_count])
        fit_tasks = [task for task in tasks if task.target_id in set(fit_ids)]
        calibration[int(cutoff)] = [task for task in tasks if task.target_id in calibration_ids]
        policies[int(cutoff)] = []
        for seed in config["policy_seeds"]:
            train_ids, validation_ids, _ = _split(fit_ids, int(seed) + int(cutoff))
            policy = FactorizedTransferPolicy(
                int(seed) + int(cutoff),
                candidate_count=int(config["random_candidate_count"]),
                similarity_residual=True,
            )
            policy.fit(
                [task for task in fit_tasks if task.target_id in set(train_ids)],
                [task for task in fit_tasks if task.target_id in set(validation_ids)],
            )
            policies[int(cutoff)].append(policy)
            fit_records.append({
                "cutoff": int(cutoff), "seed": int(seed),
                "occurrence_logit_shift": float(policy.occurrence_logit_shift_),
                **{f"theta_{i}": float(value) for i, value in enumerate(policy.parameters_)},
            })
    return policies, calibration, pd.DataFrame(fit_records)


def ensemble_distribution(task, policies):
    distributions = [
        factorized_distribution(
            task, policy.parameters_, policy.occurrence_logit_shift_,
            similarity_residual=True,
        )
        for policy in policies
    ]
    values = np.concatenate([item[0] for item in distributions])
    masses = np.concatenate([item[1] / len(distributions) for item in distributions])
    return values, masses


def complete_similarity_distribution(task):
    weights = np.clip(task.donor_features[:, 0].astype(float), 0.0, None)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1 / len(weights))
    values, masses, _ = complete_mixture_distribution(task, weights)
    return values, masses


def score_action(task, values, masses, multiplier, method):
    shortage = task.holding_cost * task.cost_ratio
    level, expected_holding, expected_shortage, expected_fixed = optimal_feasible_level(
        values, masses, task.holding_cost, shortage * multiplier,
        capacity=task.capacity, minimum_order_quantity=task.minimum_order_quantity,
        fixed_order_cost=task.fixed_order_cost,
    )
    decision = ReplenishmentDecision(
        level, expected_holding, expected_shortage, method,
        {"shortage_penalty_multiplier": multiplier}, expected_fixed,
    )
    context = OperationalContext(
        task.target_id, task.lead_time, task.holding_cost, shortage,
        task.capacity, None, task.minimum_order_quantity, task.fixed_order_cost,
    )
    result = HeldOutDecisionEvaluator().evaluate(decision, task.actual_demand, context)
    return {
        "selected_level": float(level),
        "total_cost": float(result.total_cost),
        "fill_rate_proxy": float(result.fill_rate_proxy),
        "holding_cost": float(result.holding_cost),
        "shortage_cost": float(result.shortage_cost),
        "fixed_order_cost": float(result.diagnostics["fixed_order_cost"]),
    }


def calibrate(calibration_tasks, policies, config):
    records: list[dict] = []
    selections: list[dict] = []
    for cutoff, tasks in calibration_tasks.items():
        for ratio in config["evaluation_grid"]["shortage_holding_ratios"]:
            ratio_tasks = [task for task in tasks if np.isclose(task.cost_ratio, ratio)]
            complete_service = []
            for task in ratio_tasks:
                values, masses = complete_similarity_distribution(task)
                outcome = score_action(task, values, masses, 1.0, "complete_similarity")
                complete_service.append(outcome["fill_rate_proxy"])
            target_service = float(np.mean(complete_service))
            feasible: list[dict] = []
            for multiplier in config["shortage_penalty_multipliers"]:
                costs, services = [], []
                for task in ratio_tasks:
                    values, masses = ensemble_distribution(task, policies[cutoff])
                    outcome = score_action(
                        task, values, masses, float(multiplier), "component_specific_calibrated"
                    )
                    costs.append(outcome["total_cost"])
                    services.append(outcome["fill_rate_proxy"])
                row = {
                    "cutoff": cutoff, "shortage_holding_ratio": float(ratio),
                    "multiplier": float(multiplier), "calibration_rows": len(ratio_tasks),
                    "target_service": target_service, "achieved_service": float(np.mean(services)),
                    "mean_original_objective_cost": float(np.mean(costs)),
                    "feasible": bool(np.mean(services) >= target_service),
                }
                records.append(row)
                if row["feasible"]:
                    feasible.append(row)
            if feasible:
                chosen = min(feasible, key=lambda row: (row["mean_original_objective_cost"], row["multiplier"]))
                selections.append({**chosen, "status": "selected"})
            else:
                selections.append({
                    "cutoff": cutoff, "shortage_holding_ratio": float(ratio),
                    "multiplier": np.nan, "calibration_rows": len(ratio_tasks),
                    "target_service": target_service, "achieved_service": np.nan,
                    "mean_original_objective_cost": np.nan, "feasible": False,
                    "status": "infeasible_within_grid",
                })
    return pd.DataFrame(records), pd.DataFrame(selections)


def build_target_task(dataset, full, v3, target_source, target_order, cutoff, lead, ratio, capacity, eligible, allowed, similarity_row):
    demand = dataset.demand.to_numpy(float)
    section = full["datasets"][dataset.name]
    order = np.argsort(-similarity_row[allowed], kind="stable")[: int(full["representation"]["top_k"])]
    donors = allowed[order]
    histories = [demand[index, :cutoff] for index in donors]
    count = int(v3["decision_transfer_gate"]["scenario_count"])
    samples = []
    for donor in donors:
        donor_id = redacted_id(dataset.name, dataset.metadata.iloc[int(donor)].item_id)
        scenario = _construct_scenarios(
            [donor_id], [demand[donor, :cutoff]], np.ones(1), lead, count,
            int(v3["random_seed"]) + int(target_order) * 10007 + cutoff * 101 + int(donor),
            "none",
        )
        samples.append(scenario.lead_time_demand)
    matrix = np.vstack(samples)
    pool = pd.DataFrame({
        "metadata_similarity": similarity_row[donors],
        "donor_id": [redacted_id(dataset.name, dataset.metadata.iloc[int(x)].item_id) for x in donors],
    })
    holding, fixed, moq = _operational_values(dataset, int(target_source), eligible, section)
    actual = float(demand[target_source, cutoff : cutoff + lead].sum())
    return __import__("cold_start_replenishment.analogs.direct_mixture_learning", fromlist=["MixtureDecisionTask"]).MixtureDecisionTask(
        redacted_id(dataset.name, dataset.metadata.iloc[int(target_source)].item_id),
        _features(pool, matrix, histories), matrix, actual, float(ratio), int(lead), capacity,
        float(moq), float(holding), float(fixed), _support_matrix(histories, lead, count),
    )


def evaluate(dataset, full, v3, policies, selections, config):
    targets, eligible = _population(dataset, full)
    targets = np.asarray(targets[: int(config["evaluation_products"])])
    similarities = braf_similarity(dataset.metadata, targets)
    section = full["datasets"][dataset.name]
    demand = dataset.demand.to_numpy(float)
    rows: list[dict] = []
    selection_map = {
        (int(row.cutoff), float(row.shortage_holding_ratio)): float(row.multiplier)
        for row in selections.itertuples(index=False) if row.status == "selected"
    }
    allowed = np.asarray([value for value in eligible if int(value) not in set(map(int, targets))], int)
    for target_position, target_source in enumerate(targets):
        target_id = redacted_id(dataset.name, dataset.metadata.iloc[int(target_source)].item_id)
        for cutoff in config["evaluation_grid"]["cutoffs"]:
            native = min(
                max(1, int(round(float(dataset.metadata.iloc[int(target_source)].lead_time)))),
                int(section["native_lead_time_cap"]),
            )
            holding, _, moq = _operational_values(dataset, int(target_source), eligible, section)
            broad = np.sort(np.random.default_rng(int(v3["random_seed"]) + cutoff * 1009 + target_position).choice(
                allowed, size=min(int(full["representation"]["broad_pool_size"]), len(allowed)), replace=False
            ))
            for lead_name, lead in {"native_capped": native, "half_native_sensitivity": max(1, math.ceil(native / 2))}.items():
                broad_scenario = _construct_scenarios(
                    [str(x) for x in broad], [demand[x, :cutoff] for x in broad],
                    np.full(len(broad), 1 / len(broad)), lead,
                    int(v3["decision_transfer_gate"]["scenario_count"]),
                    int(v3["random_seed"]) + target_position * 10007 + cutoff * 101 + 4,
                    "capacity_reference",
                )
                capacities = {
                    "unconstrained": None,
                    "medium": max(moq, float(np.quantile(broad_scenario.lead_time_demand, 0.75))),
                    "tight": max(moq, float(np.quantile(broad_scenario.lead_time_demand, 0.50))),
                }
                base_task = build_target_task(
                    dataset, full, v3, int(target_source), int(target_position), int(cutoff),
                    int(lead), 1.0, None, eligible, allowed, similarities[target_position],
                )
                cs_values, cs_masses = ensemble_distribution(base_task, policies[int(cutoff)])
                cm_values, cm_masses = complete_similarity_distribution(base_task)
                for ratio in config["evaluation_grid"]["shortage_holding_ratios"]:
                    multiplier = selection_map.get((int(cutoff), float(ratio)))
                    if multiplier is None:
                        continue
                    for capacity_name, capacity in capacities.items():
                        task = replace(base_task, cost_ratio=float(ratio), capacity=capacity)
                        for method, values, masses, action_multiplier in (
                            ("component_specific_uncalibrated", cs_values, cs_masses, 1.0),
                            ("component_specific_matched_service", cs_values, cs_masses, multiplier),
                            ("complete_similarity", cm_values, cm_masses, 1.0),
                        ):
                            outcome = score_action(task, values, masses, action_multiplier, method)
                            rows.append({
                                "dataset": dataset.name, "target_id": target_id, "cutoff": int(cutoff),
                                "lead_time_regime": lead_name, "lead_time": int(lead),
                                "shortage_holding_ratio": float(ratio), "capacity_regime": capacity_name,
                                "method": method, "shortage_penalty_multiplier": action_multiplier,
                                "actual_demand": task.actual_demand,
                                **outcome,
                            })
    return pd.DataFrame(rows)


def summarize(rows, config):
    summary = rows.groupby("method", as_index=False).agg(
        mean_cost=("total_cost", "mean"), service=("fill_rate_proxy", "mean"),
        mean_action=("selected_level", "mean"), rows=("total_cost", "size"),
        mean_holding_cost=("holding_cost", "mean"),
        mean_shortage_cost=("shortage_cost", "mean"),
        mean_fixed_order_cost=("fixed_order_cost", "mean"),
    )
    keys = ["target_id", "cutoff", "lead_time_regime", "shortage_holding_ratio", "capacity_regime"]
    pivot = rows.pivot(index=keys, columns="method", values=["total_cost", "fill_rate_proxy", "selected_level"]).reset_index()
    pivot.columns = ["_".join(x).strip("_") for x in pivot.columns]
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    draws = int(config["bootstrap"]["draws"])
    comparisons = []
    for first, comparator in (
        ("component_specific_matched_service", "component_specific_uncalibrated"),
        ("component_specific_matched_service", "complete_similarity"),
        ("component_specific_uncalibrated", "complete_similarity"),
    ):
        differences = pd.DataFrame({
            "target_id": pivot["target_id"],
            "cost_difference": pivot[f"total_cost_{first}"] - pivot[f"total_cost_{comparator}"],
            "service_difference": pivot[f"fill_rate_proxy_{first}"] - pivot[f"fill_rate_proxy_{comparator}"],
            "action_difference": pivot[f"selected_level_{first}"] - pivot[f"selected_level_{comparator}"],
        })
        clustered = differences.groupby("target_id", as_index=False).mean(numeric_only=True)
        sampled = rng.integers(0, len(clustered), size=(draws, len(clustered)))
        row = {"first_policy": first, "comparator": comparator, "products": len(clustered), "bootstrap_draws": draws}
        for measure in ("cost", "service", "action"):
            values = clustered[f"{measure}_difference"].to_numpy()
            boot = values[sampled].mean(axis=1)
            row[f"{measure}_difference"] = float(values.mean())
            row[f"{measure}_ci_low"] = float(np.quantile(boot, 0.025))
            row[f"{measure}_ci_high"] = float(np.quantile(boot, 0.975))
        comparisons.append(row)
    comparison = pd.DataFrame(comparisons)
    return summary, comparison


def main() -> None:
    started = time.time()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    full = yaml.safe_load((ROOT / "configs/full_scale.yaml").read_text())
    v3 = yaml.safe_load((ROOT / "configs/ai_darld_v3.yaml").read_text())
    dataset = parse_braf()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path = V1_CACHE
    if cache_path.exists():
        policies, calibration_tasks, fits = joblib.load(cache_path)
    else:
        policies, calibration_tasks, fits = fit_policies(dataset, full, v3, config)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        joblib.dump((policies, calibration_tasks, fits), OUTPUT / "policy_and_calibration_cache.joblib")
    grid, selections = calibrate(calibration_tasks, policies, config)
    if not selections.status.eq("selected").all():
        raise RuntimeError("At least one cutoff-ratio calibration target is infeasible")
    rows = evaluate(dataset, full, v3, policies, selections, config)
    summary, comparison = summarize(rows, config)
    fits.to_csv(OUTPUT / "fits.csv", index=False)
    grid.to_csv(OUTPUT / "calibration_grid.csv", index=False)
    selections.to_csv(OUTPUT / "calibration_selection.csv", index=False)
    rows.to_parquet(OUTPUT / "evaluation_rows.parquet", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    comparison.to_csv(OUTPUT / "paired_comparison.csv", index=False)
    manifest = {
        "status": "completed", "analysis_label": config["analysis_label"],
        "config": str(CONFIG_PATH.relative_to(ROOT)), "config_sha256": sha256(CONFIG_PATH),
        "raw_archive_sha256": sha256(ROOT / "data/raw/spdf/Spare-Part-Demand-Forecasting-main.zip"),
        "elapsed_seconds": time.time() - started,
        "evaluation_target_outcomes_used_for_calibration": False,
        "calibration_products_per_cutoff": int(config["service_calibration_products_per_cutoff"]),
        "evaluation_products": int(config["evaluation_products"]),
        "fixed_policy_counterfactual": True,
        "scoring_uses_original_operational_costs": True,
        "interpretation": "Retrospective exploratory fixed-policy service-calibration analysis; not independent confirmation.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
