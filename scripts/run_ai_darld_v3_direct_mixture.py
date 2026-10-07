#!/usr/bin/env python3
# ruff: noqa: E402
"""Train and evaluate direct donor-mixture policies on the frozen V3 grid."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cold_start_replenishment.analogs.direct_mixture_learning import (
    DirectMixturePolicy,
    MixtureDecisionTask,
    exact_task_cost,
)
from cold_start_replenishment.data.spdf_pilot import parse_braf, parse_man, redacted_id
from cold_start_replenishment.evaluation.acceptance_enhancement import (
    _construct_scenarios,
    _prepare_dataset,
)
from cold_start_replenishment.evaluation.decision_evaluator import HeldOutDecisionEvaluator
from cold_start_replenishment.evaluation.sprint2 import (
    _build_reliability_training,
    _operational_values,
    _similarity_rows,
)
from cold_start_replenishment.evaluation.targeted_acceptance_extension import (
    _selection_sources,
)
from cold_start_replenishment.framework.objects import OperationalContext
from cold_start_replenishment.inventory.newsvendor import NewsvendorOptimizer


def _split(ids: np.ndarray, seed: int) -> tuple[set[str], set[str], set[str]]:
    values = np.array(sorted(set(map(str, ids))))
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    first = int(0.60 * len(values))
    second = int(0.80 * len(values))
    return set(values[:first]), set(values[first:second]), set(values[second:])


def _features(
    pool: pd.DataFrame, donor_lead_demand: np.ndarray,
    donor_histories: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Cutoff-visible donor summaries on the same horizon as the forecast.

    Occurrence is the event that aggregate lead demand is positive, not a
    one-period nonzero rate.  Positive magnitude is estimated from the same
    lead-demand samples, so the hurdle components define one coherent law.
    """
    samples = np.asarray(donor_lead_demand, dtype=float)
    event = np.mean(samples > 0, axis=1)
    support = (
        np.asarray([np.count_nonzero(np.asarray(row) > 0) for row in donor_histories], float)
        if donor_histories is not None
        else np.sum(samples > 0, axis=1).astype(float)
    )
    positive_mean = np.asarray(
        [float(row[row > 0].mean()) if np.any(row > 0) else 0.0 for row in samples]
    )
    return np.column_stack(
        [pool.metadata_similarity.to_numpy(float), event, support, positive_mean]
    )


def _support_matrix(
    donor_histories: list[np.ndarray], lead: int, simulation_atoms: int
) -> np.ndarray:
    """Return historical and numerical support without conflating the two."""
    rows = []
    for history in donor_histories:
        values = np.asarray(history, float)
        rows.append([
            len(values), np.count_nonzero(values > 0),
            max(len(values) - int(lead) + 1, 0), int(simulation_atoms),
        ])
    return np.asarray(rows, float)


def _historical_tasks(
    dataset,
    full,
    v3,
    cutoff: int,
    *,
    scenario_count: int | None = None,
    random_seed: int | None = None,
    excluded_donor_ids: set[str] | None = None,
):
    section = full["datasets"][dataset.name]
    eligible, targets, pseudo = _selection_sources(dataset, full)
    combined = np.concatenate([targets, pseudo])
    similarities = _similarity_rows(dataset, combined)
    lookup = {int(source): index for index, source in enumerate(combined)}
    top_k = int(full["representation"]["top_k"])
    excluded = set() if excluded_donor_ids is None else set(map(str, excluded_donor_ids))
    base = _build_reliability_training(
        dataset,
        cutoff,
        pseudo,
        set(map(int, targets)),
        eligible,
        similarities,
        lookup,
        section,
        top_k + len(excluded),
    )
    if excluded:
        base = base[~base.donor_id.astype(str).isin(excluded)].copy()
        base = (
            base.sort_values(
                ["pseudo_target_id", "metadata_similarity"], ascending=[True, False]
            )
            .groupby("pseudo_target_id", sort=False)
            .head(top_k)
            .reset_index(drop=True)
        )
    id_to_source = {
        redacted_id(dataset.name, row.item_id): int(str(index))
        for index, row in dataset.metadata.iterrows()
    }
    demand = dataset.demand.to_numpy(float)
    tasks: list[MixtureDecisionTask] = []
    # Development uses a fixed 60-scenario empirical approximation for tractable
    # non-smooth policy search; final evaluation below uses all 300 scenarios.
    count = int(scenario_count or v3["direct_mixture_learning"]["training_scenario_count"])
    seed = int(v3["random_seed"] if random_seed is None else random_seed)
    for pseudo_id, pool in base.groupby("pseudo_target_id", sort=False):
        pool = pool.sort_values("metadata_similarity", ascending=False).reset_index(drop=True)
        source = id_to_source[str(pseudo_id)]
        training_cutoff = int(pool.reliability_training_cutoff.iloc[0])
        native = int(pool.validation_end_cutoff.iloc[0]) - training_cutoff
        holding, fixed, moq = _operational_values(dataset, source, eligible, section)
        for lead in sorted({native, max(1, math.ceil(native / 2))}):
            donor_samples = []
            for donor_id in pool.donor_id.astype(str):
                donor_source = id_to_source[donor_id]
                scenario = _construct_scenarios(
                    [donor_id],
                    [demand[donor_source, :training_cutoff]],
                    np.ones(1),
                    lead,
                    count,
                    seed + cutoff * 1009 + source * 31 + donor_source + lead,
                    "direct_mixture_training",
                )
                donor_samples.append(scenario.lead_time_demand)
            matrix = np.vstack(donor_samples)
            uniform_flat = matrix.reshape(-1)
            capacities = [
                None,
                float(np.quantile(uniform_flat, 0.75)),
                float(np.quantile(uniform_flat, 0.50)),
            ]
            actual = float(demand[source, training_cutoff : training_cutoff + lead].sum())
            for ratio in map(float, v3["decision_transfer_gate"]["cost_ratios"]):
                for capacity in capacities:
                    tasks.append(
                        MixtureDecisionTask(
                            str(pseudo_id),
                            _features(pool, matrix, [demand[id_to_source[x], :training_cutoff] for x in pool.donor_id.astype(str)]),
                            matrix,
                            actual,
                            ratio,
                            lead,
                            None if capacity is None else max(float(moq), capacity),
                            float(moq),
                            holding,
                            fixed,
                            _support_matrix(
                                [demand[id_to_source[x], :training_cutoff] for x in pool.donor_id.astype(str)],
                                lead, count,
                            ),
                        )
                    )
    return tasks, base, eligible, targets


def _evaluate(dataset, full, v3, scored, eligible, targets, policies):
    section = full["datasets"][dataset.name]
    demand = dataset.demand.to_numpy(float)
    id_to_source = {
        redacted_id(dataset.name, row.item_id): int(str(index))
        for index, row in dataset.metadata.iterrows()
    }
    target_order = {int(source): index for index, source in enumerate(targets)}
    allowed = np.asarray([x for x in eligible if int(x) not in set(map(int, targets))], int)
    optimizer, evaluator = NewsvendorOptimizer(), HeldOutDecisionEvaluator()
    rows = []
    seed, count = int(v3["random_seed"]), int(v3["decision_transfer_gate"]["scenario_count"])
    for (target_id, cutoff_value), pool in scored.groupby(["target_id", "cutoff"], sort=False):
        pool = pool.sort_values("similarity_rank").reset_index(drop=True)
        cutoff = int(cutoff_value)
        source = id_to_source[str(target_id)]
        order = target_order[source]
        donor_ids = pool.donor_id.astype(str).tolist()
        histories = [demand[id_to_source[x], :cutoff] for x in donor_ids]
        native = min(
            max(1, int(round(float(dataset.metadata.iloc[source].lead_time)))),
            int(section["native_lead_time_cap"]),
        )
        holding, fixed, moq = _operational_values(dataset, source, eligible, section)
        broad_sources = np.sort(
            np.random.default_rng(seed + cutoff * 1009 + order).choice(
                allowed,
                size=min(int(full["representation"]["broad_pool_size"]), len(allowed)),
                replace=False,
            )
        )
        broad_ids = [
            redacted_id(dataset.name, dataset.metadata.iloc[x].item_id) for x in broad_sources
        ]
        for lead_name, lead in {
            "native_capped": native,
            "half_native_sensitivity": max(1, math.ceil(native / 2)),
        }.items():
            actual = float(demand[source, cutoff : cutoff + lead].sum())
            broad = _construct_scenarios(
                broad_ids,
                [demand[x, :cutoff] for x in broad_sources],
                np.full(len(broad_sources), 1 / len(broad_sources)),
                lead,
                count,
                seed + order * 10007 + cutoff * 101 + 4,
                "broad_capacity_reference",
            )
            capacities = {
                "unconstrained": None,
                "medium": max(moq, float(np.quantile(broad.lead_time_demand, 0.75))),
                "tight": max(moq, float(np.quantile(broad.lead_time_demand, 0.50))),
            }
            donor_samples = []
            for donor_id, history in zip(donor_ids, histories, strict=True):
                donor_samples.append(
                    _construct_scenarios(
                        [donor_id],
                        [history],
                        np.ones(1),
                        lead,
                        count,
                        seed + order * 10007 + cutoff * 101 + id_to_source[donor_id],
                        "direct_mixture_evaluation",
                    ).lead_time_demand
                )
            matrix = np.vstack(donor_samples)
            feature_values = _features(pool, matrix, histories)
            for ratio in map(float, v3["decision_transfer_gate"]["cost_ratios"]):
                for capacity_name, capacity in capacities.items():
                    task = MixtureDecisionTask(
                        str(target_id), feature_values, matrix, actual, ratio, lead, capacity, moq,
                        holding, fixed, _support_matrix(histories, lead, count),
                    )
                    weights = policies[cutoff].weights(task)
                    scenario = _construct_scenarios(
                        donor_ids,
                        histories,
                        weights,
                        lead,
                        count,
                        seed + order * 10007 + cutoff * 101 + 33000,
                        "direct_mixture_learning",
                    )
                    context = OperationalContext(
                        str(target_id), lead, holding, holding * ratio, capacity, None, moq, fixed
                    )
                    decision = optimizer.optimize(scenario, context)
                    result = evaluator.evaluate(decision, actual, context)
                    critical = ratio / (ratio + 1)
                    predicted = float(np.quantile(scenario.lead_time_demand, critical))
                    lower, upper = np.quantile(scenario.lead_time_demand, [0.1, 0.9])
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "target_id": target_id,
                            "cutoff": cutoff,
                            "lead_time_regime": lead_name,
                            "lead_time": lead,
                            "capacity_regime": capacity_name,
                            "shortage_holding_ratio": ratio,
                            "method": "direct_mixture_learning",
                            "actual_demand": actual,
                            "selected_level": decision.level,
                            "total_cost": result.total_cost,
                            "oracle_cost_normalized_regret": result.normalized_regret,
                            "fill_rate_proxy": result.fill_rate_proxy,
                            "critical_quantile_pinball": float(
                                (actual - predicted) * (critical - float(actual < predicted))
                            ),
                            "central_80_coverage": float(lower <= actual <= upper),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    full = yaml.safe_load((ROOT / "configs/full_scale.yaml").read_text())
    v3 = yaml.safe_load((ROOT / "configs/ai_darld_v3.yaml").read_text())
    all_results, selection_rows, confirmation_rows = [], [], []
    for dataset in (parse_man(), parse_braf()):
        checkpoint = pd.read_parquet(
            ROOT / f"outputs/full_scale/checkpoints/{dataset.name.lower()}_reliability.parquet"
        )
        _, scored, targets, eligible = _prepare_dataset(dataset, full, checkpoint)
        policies = {}
        for cutoff in map(int, full["datasets"][dataset.name]["cutoffs"]):
            tasks, _, _, _ = _historical_tasks(dataset, full, v3, cutoff)
            ids = np.array(sorted({task.target_id for task in tasks}))
            train_ids, validation_ids, confirmation_ids = _split(
                ids, int(v3["random_seed"]) + cutoff
            )
            train = [task for task in tasks if task.target_id in train_ids]
            validation = [task for task in tasks if task.target_id in validation_ids]
            confirmation = [task for task in tasks if task.target_id in confirmation_ids]
            policy = DirectMixturePolicy(
                int(v3["random_seed"]) + cutoff,
                candidate_count=int(v3["direct_mixture_learning"]["candidate_count"]),
            ).fit(train, validation)
            policies[cutoff] = policy
            uniform_parameters = np.zeros(9)
            uniform_parameters[7] = 20.0
            learned_cost = float(
                np.mean([exact_task_cost(task, policy.parameters_) for task in confirmation])
            )
            uniform_cost = float(
                np.mean([exact_task_cost(task, uniform_parameters) for task in confirmation])
            )
            selection_rows.append(
                {
                    "dataset": dataset.name,
                    "cutoff": cutoff,
                    "train_targets": len(train_ids),
                    "validation_targets": len(validation_ids),
                    "confirmation_targets": len(confirmation_ids),
                    "confirmation_direct_cost": learned_cost,
                    "confirmation_uniform_cost": uniform_cost,
                    **{f"theta_{i}": float(value) for i, value in enumerate(policy.parameters_)},
                }
            )
            confirmation_rows.extend(
                {
                    "dataset": dataset.name,
                    "cutoff": cutoff,
                    "pseudo_target_id": task.target_id,
                    "direct_cost": exact_task_cost(task, policy.parameters_),
                    "uniform_cost": exact_task_cost(task, uniform_parameters),
                }
                for task in confirmation
            )
        all_results.append(_evaluate(dataset, full, v3, scored, eligible, targets, policies))
    results = pd.concat(all_results, ignore_index=True)
    run_root = ROOT / "outputs/runs/ai_darld_v3"
    out_root = ROOT / "outputs/ai_darld_v3"
    results.to_parquet(run_root / "direct_mixture_learning_results.parquet", index=False)
    pd.DataFrame(selection_rows).to_csv(
        out_root / "direct_mixture_learning_parameters.csv", index=False
    )
    pd.DataFrame(confirmation_rows).to_parquet(
        run_root / "direct_mixture_confirmation.parquet", index=False
    )
    (out_root / "direct_mixture_learning_manifest.json").write_text(
        json.dumps(
            {
                "seed": v3["random_seed"],
                "candidate_count": int(v3["direct_mixture_learning"]["candidate_count"]) + 3,
                "training_scenario_count": int(
                    v3["direct_mixture_learning"]["training_scenario_count"]
                ),
                "evaluation_scenario_count": int(v3["decision_transfer_gate"]["scenario_count"]),
                "split": {"train": 0.6, "validation": 0.2, "confirmation": 0.2},
                "evaluation_population": "reused_frozen_targets",
            },
            indent=2,
        )
    )
    print(pd.DataFrame(selection_rows).to_string(index=False))


if __name__ == "__main__":
    main()
