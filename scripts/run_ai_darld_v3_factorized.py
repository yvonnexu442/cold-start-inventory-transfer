#!/usr/bin/env python3
# ruff: noqa: E402
"""Train five-seed factorized transfer and evaluate fixed information degradations."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ai_darld_v3_direct_mixture import _features, _historical_tasks, _split, _support_matrix

from cold_start_replenishment.analogs.decision_transfer_gate import weighted_quantile
from cold_start_replenishment.analogs.direct_mixture_learning import MixtureDecisionTask
from cold_start_replenishment.analogs.factorized_transfer import (
    FactorizedTransferPolicy,
    complete_mixture_distribution,
    factorized_distribution,
    factorized_loss,
    strict_relation_distribution,
)
from cold_start_replenishment.data.spdf_pilot import (
    braf_similarity,
    man_similarity,
    parse_braf,
    parse_man,
    redacted_id,
)
from cold_start_replenishment.evaluation.acceptance_enhancement import (
    _construct_scenarios,
    _eligible_sources,
)
from cold_start_replenishment.evaluation.decision_evaluator import HeldOutDecisionEvaluator
from cold_start_replenishment.evaluation.sprint2 import _operational_values
from cold_start_replenishment.framework.objects import (
    DecisionScenarioSet,
    OperationalContext,
    ReplenishmentDecision,
)
from cold_start_replenishment.inventory.newsvendor import (
    optimal_feasible_level,
)

SEEDS = (20260930, 20261011, 20261023, 20261037, 20261051)
POLICY_DEVELOPMENT_TARGETS = 60
CORRECTED_EVALUATION_TARGETS = 250


def _population(dataset, full):
    """Reconstruct the frozen target population without refitting unused learners."""
    section = full["datasets"][dataset.name]
    eligible = _eligible_sources(dataset, section)
    rng = np.random.default_rng(full["random_seed"] + (1 if dataset.name == "MAN" else 2))
    targets = np.sort(rng.permutation(eligible)[: int(section["sample_size"])])
    return targets, eligible


def _select(tasks, chosen):
    return [task for task in tasks if task.target_id in chosen]


def _policies(dataset, full, v3, only_cutoff=None):
    fitted, records = {}, []
    for cutoff in map(int, full["datasets"][dataset.name]["cutoffs"]):
        if only_cutoff is not None and cutoff != int(only_cutoff):
            continue
        tasks, _, _, _ = _historical_tasks(
            dataset, full, v3, cutoff, scenario_count=120, random_seed=int(v3["random_seed"])
        )
        ids = np.array(sorted({task.target_id for task in tasks}))
        # Freeze a tractable, identical development budget for every ablation.
        # The sample is selected before fitting and never uses task outcomes.
        development_rng = np.random.default_rng(int(v3["random_seed"]) + cutoff)
        ids = development_rng.permutation(ids)[: min(POLICY_DEVELOPMENT_TARGETS, len(ids))]
        tasks = _select(tasks, set(ids))
        fitted[cutoff] = {
            "factorized_transfer": [],
            "shared_hurdle": [],
            "factorized_no_contraction": [],
            "factorized_similarity_residual": [],
            "shared_similarity_residual": [],
            "anchored_no_contraction_refit": [],
        }
        for seed in SEEDS:
            train_ids, validation_ids, confirmation_ids = _split(ids, seed + cutoff)
            random_candidates = int(v3["factorized_transfer"]["random_candidate_count"])
            variants = {
                "factorized_transfer": FactorizedTransferPolicy(
                    seed + cutoff, candidate_count=random_candidates
                ),
                "shared_hurdle": FactorizedTransferPolicy(
                    seed + cutoff, candidate_count=random_candidates, shared=True
                ),
                "factorized_no_contraction": FactorizedTransferPolicy(
                    seed + cutoff, candidate_count=random_candidates, contraction=False
                ),
                "factorized_similarity_residual": FactorizedTransferPolicy(
                    seed + cutoff, candidate_count=random_candidates, similarity_residual=True
                ),
                "shared_similarity_residual": FactorizedTransferPolicy(
                    seed + cutoff,
                    candidate_count=random_candidates,
                    shared=True,
                    similarity_residual=True,
                ),
                "anchored_no_contraction_refit": FactorizedTransferPolicy(
                    seed + cutoff,
                    candidate_count=random_candidates,
                    contraction=False,
                    similarity_residual=True,
                ),
            }
            for name, candidate_policy in variants.items():
                candidate_policy.fit(_select(tasks, train_ids), _select(tasks, validation_ids))
                fitted[cutoff][name].append(candidate_policy)
            confirmation = _select(tasks, confirmation_ids)
            for name, policy in variants.items():
                chosen = int(next(row["candidate"] for row in policy.search_ if row["selected"]))
                records.append(
                    {
                        "dataset": dataset.name,
                        "cutoff": cutoff,
                        "seed": seed,
                        "policy": name,
                        "selected_candidate": chosen,
                        "occurrence_logit_shift": policy.occurrence_logit_shift_,
                        "confirmation_policy_loss": float(
                            np.mean(
                                [
                                    factorized_loss(
                                        task,
                                        policy.parameters_,
                                        occurrence_logit_shift=policy.occurrence_logit_shift_,
                                        shared=policy.shared,
                                        contraction=policy.contraction,
                                        similarity_residual=policy.similarity_residual,
                                    )
                                    for task in confirmation
                                ]
                            )
                        ),
                        **{
                            f"theta_{index}": float(value)
                            for index, value in enumerate(policy.parameters_)
                        },
                    }
                )
    return fitted, records


def _fit_cutoff_worker(dataset_name, cutoff):
    full = yaml.safe_load((ROOT / "configs/full_scale.yaml").read_text())
    v3 = yaml.safe_load((ROOT / "configs/ai_darld_v3.yaml").read_text())
    dataset = parse_man() if dataset_name == "MAN" else parse_braf()
    return _policies(dataset, full, v3, only_cutoff=cutoff)


def _parallel_policies(dataset_name, cutoffs):
    fitted, records = {}, []
    with ProcessPoolExecutor(max_workers=len(cutoffs)) as executor:
        futures = [
            executor.submit(_fit_cutoff_worker, dataset_name, int(cutoff)) for cutoff in cutoffs
        ]
        for cutoff, future in zip(cutoffs, futures, strict=True):
            cutoff_policies, cutoff_records = future.result()
            fitted.update(cutoff_policies)
            records.extend(cutoff_records)
            print(f"fitted {dataset_name} cutoff {cutoff}", flush=True)
    return fitted, records


def _scenario_from_distribution(values, masses, count, seed, lead, method):
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=count, p=masses / masses.sum())
    trajectories = np.zeros((count, lead))
    trajectories[:, -1] = draws
    return DecisionScenarioSet(trajectories, lead, method)


def _perturb(task, name, rng):
    features, scenarios = task.donor_features.copy(), task.donor_scenarios.copy()
    support = None if task.donor_support is None else task.donor_support.copy()
    if name == "remove_top_similarity":
        keep = np.arange(len(features)) != int(np.argmax(features[:, 0]))
        features, scenarios = features[keep], scenarios[keep]
        if support is not None:
            support = support[keep]
    elif name == "within_pool_similarity_perturbation":
        similarity = np.clip(features[:, 0] + rng.normal(0, 0.10, len(features)), 0, 1)
        missing = rng.random(len(features)) < 0.20
        similarity[missing] = float(np.mean(similarity[~missing])) if np.any(~missing) else 0.0
        features[:, 0] = similarity
    return MixtureDecisionTask(
        task.target_id,
        features,
        scenarios,
        task.actual_demand,
        task.cost_ratio,
        task.lead_time,
        task.capacity,
        task.minimum_order_quantity,
        task.holding_cost,
        task.fixed_order_cost,
        support,
    )


def _end_to_end_pool(dataset, source, allowed, clean_pool, *, level, seed, top_k):
    """Perturb raw target metadata before retrieval with a frozen fallback rule."""
    settings = {
        "low": (0.10, 0.10, 0.15),
        "high": (0.25, 0.30, 0.35),
    }
    noise_sd, missing_fraction, token_dropout = settings[level]
    rng = np.random.default_rng(seed)
    metadata = dataset.metadata.copy(deep=True)
    numeric = (
        ["lead_time", "cost_price", "inventory_cost", "moq"]
        if dataset.name == "MAN"
        else ["lead_time", "price"]
    )
    for column in numeric:
        value = max(float(metadata.at[source, column]), 0.0)
        value = float(np.expm1(np.log1p(value) + rng.normal(0.0, noise_sd)))
        if rng.random() < missing_fraction:
            value = float(pd.to_numeric(metadata.iloc[allowed][column], errors="coerce").median())
        metadata.at[source, column] = max(value, 0.0)
    text_column = "product_group" if dataset.name == "MAN" else "description"
    tokens = str(metadata.at[source, text_column]).split()
    kept = [token for token in tokens if rng.random() >= token_dropout]
    metadata.at[source, text_column] = " ".join(kept) if kept else "__MISSING__"
    similarities = (
        man_similarity(metadata, np.asarray([source], dtype=int))[0]
        if dataset.name == "MAN"
        else braf_similarity(metadata, np.asarray([source], dtype=int))[0]
    )
    order = np.argsort(-similarities[allowed], kind="stable")[:top_k]
    donors = np.asarray(allowed, dtype=int)[order]
    clean_ids = set(clean_pool.donor_id.astype(str))
    records = pd.DataFrame(
        {
            "donor_id": [
                redacted_id(dataset.name, dataset.metadata.iloc[int(index)].item_id)
                for index in donors
            ],
            "metadata_similarity": similarities[donors],
            "similarity_rank": np.arange(1, len(donors) + 1),
        }
    )
    overlap = len(clean_ids & set(records.donor_id.astype(str))) / max(len(clean_ids), 1)
    return records, float(overlap)


def _evaluate(
    dataset,
    full,
    v3,
    scored,
    eligible,
    targets,
    policies,
    *,
    perturbations=None,
    metadata_seed=20260930,
    only_methods=None,
):
    section, demand = full["datasets"][dataset.name], dataset.demand.to_numpy(float)
    id_to_source = {
        redacted_id(dataset.name, row.item_id): int(str(index))
        for index, row in dataset.metadata.iterrows()
    }
    target_order = {int(source): index for index, source in enumerate(targets)}
    allowed = np.asarray([x for x in eligible if int(x) not in set(map(int, targets))], int)
    evaluator, rows = HeldOutDecisionEvaluator(), []
    degraded_pool_cache = {}
    count = int(v3["decision_transfer_gate"]["scenario_count"])
    for (target_id, cutoff_value), pool in scored.groupby(["target_id", "cutoff"], sort=False):
        pool = pool.sort_values("similarity_rank").reset_index(drop=True)
        cutoff, source = int(cutoff_value), id_to_source[str(target_id)]
        order = target_order[source]
        donor_ids = pool.donor_id.astype(str).tolist()
        histories = [demand[id_to_source[x], :cutoff] for x in donor_ids]
        degraded_pools = {}
        for level_index, level in enumerate(("low", "high"), start=1):
            cache_key = (source, level)
            if cache_key not in degraded_pool_cache:
                degraded_pool_cache[cache_key] = _end_to_end_pool(
                    dataset,
                    source,
                    allowed,
                    pool,
                    level=level,
                    seed=metadata_seed + order * 10007 + level_index,
                    top_k=int(full["representation"]["top_k"]),
                )
            degraded_pools[level] = degraded_pool_cache[cache_key]
        native = min(
            max(1, int(round(float(dataset.metadata.iloc[source].lead_time)))),
            int(section["native_lead_time_cap"]),
        )
        holding, fixed, moq = _operational_values(dataset, source, eligible, section)
        broad_sources = np.sort(
            np.random.default_rng(20260930 + cutoff * 1009 + order).choice(
                allowed,
                size=min(int(full["representation"]["broad_pool_size"]), len(allowed)),
                replace=False,
            )
        )
        for lead_name, lead in {
            "native_capped": native,
            "half_native_sensitivity": max(1, math.ceil(native / 2)),
        }.items():
            actual = float(demand[source, cutoff : cutoff + lead].sum())
            broad = _construct_scenarios(
                [str(x) for x in broad_sources],
                [demand[x, :cutoff] for x in broad_sources],
                np.full(len(broad_sources), 1 / len(broad_sources)),
                lead,
                count,
                20260930 + order * 10007 + cutoff * 101 + 4,
                "capacity_reference",
            )
            capacities = {
                "unconstrained": None,
                "medium": max(moq, float(np.quantile(broad.lead_time_demand, 0.75))),
                "tight": max(moq, float(np.quantile(broad.lead_time_demand, 0.50))),
            }
            matrices = {}
            for perturbation in ("none", "donor_history_half"):
                donor_samples = []
                for donor_id, history in zip(donor_ids, histories, strict=True):
                    visible = (
                        history if perturbation == "none" else history[-max(1, len(history) // 2) :]
                    )
                    donor_samples.append(
                        _construct_scenarios(
                            [donor_id],
                            [visible],
                            np.ones(1),
                            lead,
                            count,
                            20260930 + order * 10007 + cutoff * 101 + id_to_source[donor_id],
                            perturbation,
                        ).lead_time_demand
                    )
                matrices[perturbation] = np.vstack(donor_samples)
            degraded_matrices = {}
            for level, (degraded_pool, overlap) in degraded_pools.items():
                degraded_ids = degraded_pool.donor_id.astype(str).tolist()
                degraded_samples = []
                for donor_id in degraded_ids:
                    degraded_samples.append(
                        _construct_scenarios(
                            [donor_id],
                            [demand[id_to_source[donor_id], :cutoff]],
                            np.ones(1),
                            lead,
                            count,
                            20260930 + order * 10007 + cutoff * 101 + id_to_source[donor_id],
                            f"metadata_e2e_{level}",
                        ).lead_time_demand
                    )
                degraded_matrices[level] = (degraded_pool, np.vstack(degraded_samples), overlap)
            features = _features(pool, matrices["none"], histories)
            for ratio in map(float, v3["decision_transfer_gate"]["cost_ratios"]):
                for capacity_name, capacity in capacities.items():
                    base = MixtureDecisionTask(
                        str(target_id),
                        features,
                        matrices["none"],
                        actual,
                        ratio,
                        lead,
                        capacity,
                        moq,
                        holding,
                        fixed,
                        _support_matrix(histories, lead, count),
                    )
                    for perturbation in perturbations or (
                        "none",
                        "donor_history_half",
                        "remove_top_similarity",
                        "within_pool_similarity_perturbation",
                        "metadata_e2e_low",
                        "metadata_e2e_high",
                    ):
                        if perturbation != "none" and not (
                            lead_name == "native_capped"
                            and np.isclose(ratio, 5.0)
                            and capacity_name == "unconstrained"
                        ):
                            continue
                        task = base
                        if perturbation == "donor_history_half":
                            task = MixtureDecisionTask(
                                str(target_id),
                                _features(
                                    pool,
                                    matrices[perturbation],
                                    [
                                        history[-max(1, len(history) // 2) :]
                                        for history in histories
                                    ],
                                ),
                                matrices[perturbation],
                                actual,
                                ratio,
                                lead,
                                capacity,
                                moq,
                                holding,
                                fixed,
                                _support_matrix(
                                    [
                                        history[-max(1, len(history) // 2) :]
                                        for history in histories
                                    ],
                                    lead,
                                    count,
                                ),
                            )
                        elif perturbation.startswith("metadata_e2e_"):
                            level = perturbation.removeprefix("metadata_e2e_")
                            degraded_pool, degraded_matrix, donor_pool_overlap = degraded_matrices[
                                level
                            ]
                            task = MixtureDecisionTask(
                                str(target_id),
                                _features(
                                    degraded_pool,
                                    degraded_matrix,
                                    [
                                        demand[id_to_source[x], :cutoff]
                                        for x in degraded_pool.donor_id.astype(str)
                                    ],
                                ),
                                degraded_matrix,
                                actual,
                                ratio,
                                lead,
                                capacity,
                                moq,
                                holding,
                                fixed,
                                _support_matrix(
                                    [
                                        demand[id_to_source[x], :cutoff]
                                        for x in degraded_pool.donor_id.astype(str)
                                    ],
                                    lead,
                                    count,
                                ),
                            )
                        elif perturbation != "none":
                            task = _perturb(
                                base,
                                perturbation,
                                np.random.default_rng(20260930 + order * 37 + cutoff + int(ratio)),
                            )
                        donor_pool_overlap = (
                            donor_pool_overlap if perturbation.startswith("metadata_e2e_") else 1.0
                        )
                        distributions = [
                            factorized_distribution(
                                task, policy.parameters_, policy.occurrence_logit_shift_
                            )
                            for policy in policies[cutoff]["factorized_transfer"]
                        ]
                        uncalibrated = [
                            factorized_distribution(task, policy.parameters_)
                            for policy in policies[cutoff]["factorized_transfer"]
                        ]
                        methods = []
                        for seed, (values, masses, diagnostics) in zip(
                            SEEDS, distributions, strict=True
                        ):
                            methods.append((f"factorized_seed_{seed}", values, masses, diagnostics))
                        ensemble_values = np.concatenate([item[0] for item in distributions])
                        ensemble_masses = np.concatenate(
                            [item[1] / len(distributions) for item in distributions]
                        )
                        methods.append(
                            ("factorized_transfer", ensemble_values, ensemble_masses, {})
                        )
                        for variant in (
                            "shared_hurdle",
                            "factorized_no_contraction",
                            "factorized_similarity_residual",
                            "shared_similarity_residual",
                            "anchored_no_contraction_refit",
                        ):
                            variant_distributions = [
                                factorized_distribution(
                                    task,
                                    policy.parameters_,
                                    policy.occurrence_logit_shift_,
                                    shared=policy.shared,
                                    contraction=policy.contraction,
                                    similarity_residual=policy.similarity_residual,
                                )
                                for policy in policies[cutoff][variant]
                            ]
                            methods.append(
                                (
                                    variant,
                                    np.concatenate([item[0] for item in variant_distributions]),
                                    np.concatenate(
                                        [
                                            item[1] / len(variant_distributions)
                                            for item in variant_distributions
                                        ]
                                    ),
                                    {},
                                )
                            )
                        for variant in ("strict_shared", "strict_separate"):
                            if variant not in policies[cutoff]:
                                continue
                            strict_distributions = [
                                strict_relation_distribution(
                                    task,
                                    policy.coefficients_,
                                    policy.occurrence_logit_shift_,
                                    shared=policy.shared,
                                    contraction_alpha=policy.contraction_alpha_,
                                    similarity_residual=policy.similarity_residual,
                                )
                                for policy in policies[cutoff][variant]
                            ]
                            methods.append(
                                (
                                    variant,
                                    np.concatenate([item[0] for item in strict_distributions]),
                                    np.concatenate(
                                        [
                                            item[1] / len(strict_distributions)
                                            for item in strict_distributions
                                        ]
                                    ),
                                    {},
                                )
                            )
                        methods.append(
                            (
                                "factorized_no_occurrence_calibration",
                                np.concatenate([item[0] for item in uncalibrated]),
                                np.concatenate(
                                    [item[1] / len(uncalibrated) for item in uncalibrated]
                                ),
                                {},
                            )
                        )
                        anchored_uncalibrated = []
                        for policy in policies[cutoff]["factorized_similarity_residual"]:
                            anchored_uncalibrated.append(
                                factorized_distribution(
                                    task, policy.parameters_, 0.0, similarity_residual=True
                                )
                            )
                        methods.append(
                            (
                                "anchored_no_calibration_fixed",
                                np.concatenate([item[0] for item in anchored_uncalibrated]),
                                np.concatenate(
                                    [
                                        item[1] / len(anchored_uncalibrated)
                                        for item in anchored_uncalibrated
                                    ]
                                ),
                                {},
                            )
                        )
                        uniform_parameters = np.zeros(10)
                        uniform_parameters[[6, 8]] = 20
                        uniform_values, uniform_masses, _ = factorized_distribution(
                            task, uniform_parameters
                        )
                        methods.append(("uniform_hurdle", uniform_values, uniform_masses, {}))
                        uniform_complete = complete_mixture_distribution(
                            task, np.full(len(task.donor_scenarios), 1 / len(task.donor_scenarios))
                        )
                        similarity = np.maximum(task.donor_features[:, 0], 0)
                        if similarity.sum() <= 0:
                            similarity = np.ones(len(similarity))
                        similarity_complete = complete_mixture_distribution(task, similarity)
                        methods.extend(
                            [
                                (
                                    "single_complete",
                                    *complete_mixture_distribution(
                                        task,
                                        np.eye(len(task.donor_scenarios), dtype=float)[
                                            int(np.argmax(task.donor_features[:, 0]))
                                        ],
                                    ),
                                ),
                                ("uniform_complete", *uniform_complete),
                                ("similarity_complete", *similarity_complete),
                            ]
                        )
                        if only_methods is not None:
                            methods = [item for item in methods if item[0] in set(only_methods)]
                        for method, values, masses, diagnostics in methods:
                            context = OperationalContext(
                                str(target_id),
                                lead,
                                holding,
                                holding * ratio,
                                capacity,
                                None,
                                moq,
                                fixed,
                            )
                            level, expected_holding, expected_shortage, expected_fixed = (
                                optimal_feasible_level(
                                    values,
                                    masses,
                                    holding,
                                    holding * ratio,
                                    capacity=capacity,
                                    minimum_order_quantity=moq,
                                    fixed_order_cost=fixed,
                                )
                            )
                            decision = ReplenishmentDecision(
                                level,
                                expected_holding,
                                expected_shortage,
                                method,
                                {"exact_weighted_empirical": True},
                                expected_fixed,
                            )
                            result = evaluator.evaluate(decision, actual, context)
                            critical = ratio / (ratio + 1)
                            quantile = weighted_quantile(values, masses, critical)
                            tail = weighted_quantile(values, masses, 0.9)
                            rows.append(
                                {
                                    "dataset": dataset.name,
                                    "target_id": target_id,
                                    "cutoff": cutoff,
                                    "lead_time_regime": lead_name,
                                    "lead_time": lead,
                                    "capacity_regime": capacity_name,
                                    "shortage_holding_ratio": ratio,
                                    "perturbation": perturbation,
                                    "method": method,
                                    "actual_demand": actual,
                                    "selected_level": decision.level,
                                    "total_cost": result.total_cost,
                                    "fill_rate_proxy": result.fill_rate_proxy,
                                    "critical_quantile_pinball": float(
                                        (actual - quantile) * (critical - float(actual < quantile))
                                    ),
                                    "tail_90_coverage": float(actual <= tail),
                                    "donor_pool_overlap": donor_pool_overlap,
                                    **diagnostics,
                                }
                            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("MAN", "BRAF"))
    parser.add_argument("--output-tag", default="historical_support_v4")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Explicitly reuse validated per-dataset outputs instead of recomputing them.",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Combine existing MAN and BRAF outputs for --output-tag without fitting policies.",
    )
    args = parser.parse_args()
    started = time.time()
    full = yaml.safe_load((ROOT / "configs/full_scale.yaml").read_text())
    v3 = yaml.safe_load((ROOT / "configs/ai_darld_v3.yaml").read_text())
    all_rows, parameters = [], []
    datasets = (parse_man(), parse_braf())
    if args.dataset:
        datasets = tuple(dataset for dataset in datasets if dataset.name == args.dataset)
    for dataset in datasets:
        checkpoint_path = (
            ROOT / f"outputs/runs/ai_darld_v3/{args.output_tag}_{dataset.name.lower()}.parquet"
        )
        if checkpoint_path.exists() and (args.reuse_existing or args.assemble_only):
            all_rows.append(pd.read_parquet(checkpoint_path))
            parameter_path = (
                ROOT
                / f"outputs/ai_darld_v3/{args.output_tag}_parameters_{dataset.name.lower()}.csv"
            )
            if parameter_path.exists():
                saved = pd.read_csv(parameter_path)
                parameters.extend(saved[saved.dataset.eq(dataset.name)].to_dict("records"))
            print(f"resumed {dataset.name} from {checkpoint_path}")
            continue
        if checkpoint_path.exists():
            raise FileExistsError(
                f"{checkpoint_path} already exists; use --reuse-existing to reuse it "
                "or choose a new --output-tag"
            )
        if args.assemble_only:
            raise FileNotFoundError(
                f"--assemble-only requires the existing dataset output {checkpoint_path}"
            )
        checkpoint = pd.read_parquet(
            ROOT / f"outputs/full_scale/checkpoints/{dataset.name.lower()}_reliability.parquet"
        )
        targets, eligible = _population(dataset, full)
        targets = np.asarray(targets[: min(CORRECTED_EVALUATION_TARGETS, len(targets))])
        # Factorized transfer uses the frozen retrieval rows and their metadata
        # similarity/support summaries, not the nonlinear reliability column.
        target_ids = {
            redacted_id(dataset.name, dataset.metadata.iloc[int(source)].item_id)
            for source in targets
        }
        scored = checkpoint[checkpoint.target_id.astype(str).isin(target_ids)].copy()
        if args.dataset:
            policies, records = _parallel_policies(
                dataset.name, full["datasets"][dataset.name]["cutoffs"]
            )
        else:
            policies, records = _policies(dataset, full, v3)
        parameters.extend(records)
        dataset_rows = _evaluate(dataset, full, v3, scored, eligible, targets, policies)
        dataset_rows.to_parquet(checkpoint_path, index=False)
        pd.DataFrame(records).to_csv(
            ROOT / f"outputs/ai_darld_v3/{args.output_tag}_parameters_{dataset.name.lower()}.csv",
            index=False,
        )
        all_rows.append(dataset_rows)
    if args.dataset:
        (
            ROOT / f"outputs/ai_darld_v3/{args.output_tag}_run_{args.dataset.lower()}.json"
        ).write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "output_tag": args.output_tag,
                    "elapsed_seconds": time.time() - started,
                    "candidate_count": int(v3["factorized_transfer"]["random_candidate_count"]) + 4,
                    "policy_seeds": SEEDS,
                    "status": "completed",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"saved {args.output_tag} checkpoint for {args.dataset}")
        return
    results = pd.concat(all_rows, ignore_index=True)
    results.to_parquet(
        ROOT / f"outputs/runs/ai_darld_v3/{args.output_tag}_results.parquet", index=False
    )
    pd.DataFrame(parameters).to_csv(
        ROOT / f"outputs/ai_darld_v3/{args.output_tag}_parameters.csv", index=False
    )
    summary = results.groupby(["dataset", "perturbation", "method"], as_index=False).agg(
        mean_cost=("total_cost", "mean"),
        tail_cost=("total_cost", lambda x: float(x[x >= x.quantile(0.9)].mean())),
        fill_rate=("fill_rate_proxy", "mean"),
        pinball=("critical_quantile_pinball", "mean"),
        tail_coverage=("tail_90_coverage", "mean"),
    )
    summary.to_csv(ROOT / f"outputs/ai_darld_v3/{args.output_tag}_summary.csv", index=False)
    (ROOT / f"outputs/ai_darld_v3/{args.output_tag}_manifest.json").write_text(
        json.dumps(
            {
                "seeds": SEEDS,
                "manifest_schema": "historical-support-v4-100candidate",
                "solver": "exact_weighted_empirical_moq_minimum_fixed_cost_v2",
                "distribution_features": "horizon_consistent_occurrence_positive_size_raw_support_v3",
                "support_interface": (
                    "occurrence contraction uses raw effective historical windows; "
                    "magnitude contraction uses raw positive historical periods; "
                    "simulation atoms are recorded separately and do not change contraction"
                ),
                "added_policy": "component-specific residual scores around complete-similarity anchor",
                "target_feature_scope": "decision_time_static_metadata_only",
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit_learn": version("scikit-learn"),
                "catboost": version("catboost"),
                "training_scenarios": 120,
                "evaluation_scenarios": 300,
                "candidate_count_total": int(v3["factorized_transfer"]["random_candidate_count"])
                + 4,
                "candidate_count_random": int(v3["factorized_transfer"]["random_candidate_count"]),
                "candidate_count_fixed_anchors": 4,
                "policy_development_targets_per_dataset_cutoff": POLICY_DEVELOPMENT_TARGETS,
                "corrected_evaluation_targets_per_dataset": CORRECTED_EVALUATION_TARGETS,
                "cutoffs": {name: full["datasets"][name]["cutoffs"] for name in ("MAN", "BRAF")},
                "clean_operating_grid": {
                    "lead_time_regimes": ["native_capped", "half_native_sensitivity"],
                    "shortage_holding_ratios": [2, 5, 10],
                    "capacity_regimes": ["unconstrained", "medium", "tight"],
                },
                "degradation_operating_slice": "native_capped, shortage_holding_ratio=5, unconstrained",
                "corrected_evaluation_scope": "full balanced clean grid plus frozen degradation slice",
                "perturbations": [
                    "donor_history_half",
                    "remove_top_similarity",
                    "within_pool_similarity_perturbation",
                    "metadata_e2e_low",
                    "metadata_e2e_high",
                ],
                "metadata_perturbation_scope": {
                    "within_pool_similarity_perturbation": "retrieval set held fixed",
                    "metadata_e2e_low": "raw target fields perturbed before retrieval; frozen model",
                    "metadata_e2e_high": "raw target fields perturbed before retrieval; frozen model",
                },
                "metadata_missing_fallback": "eligible-donor median for numeric fields; __MISSING__ for emptied text",
                "evaluation_population": "reused_frozen_targets",
                "superseded_evidence": [
                    "round_unconstrained_quantile_then_capacity_clip",
                    "single_period_occurrence_feature",
                    "legacy_full_grid_outputs_before_2026_10_01",
                ],
                "aggregation_runtime_seconds": time.time() - started,
                "runtime_note": "Dataset/cutoff fits are checkpointed; wall time is measured by the current run.",
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
