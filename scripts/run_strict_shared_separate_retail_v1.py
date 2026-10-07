#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the frozen strict comparison on UCI and Favorita formal populations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUTPUT = ROOT / "outputs/ai_darld_v3/strict_shared_separate_corrected_v2"
CONFIG = ROOT / "configs/strict_shared_separate_corrected_v2.yaml"

from run_favorita_untouched_confirmation import (  # noqa: E402
    _build_policy_tasks as _favorita_tasks,
)
from run_favorita_untouched_confirmation import (
    _cutoff_index,
    _eligible_donors,
    _fit_metadata,
    _prepare_panel,
    _pseudo_targets,
)
from run_favorita_untouched_confirmation import (
    _make_task as _favorita_task,
)
from run_favorita_untouched_confirmation import (
    _nearest as _favorita_nearest,
)
from run_uci_confirmation_v2 import _ordered_eligible  # noqa: E402
from run_uci_online_retail_ii_external import (  # noqa: E402
    _build_tasks as _uci_tasks,
)
from run_uci_online_retail_ii_external import (
    _features as _uci_features,
)
from run_uci_online_retail_ii_external import (
    _fit_text,
    _horizon_atoms,
    _support_matrix,
)
from run_uci_online_retail_ii_external import (
    _nearest as _uci_nearest,
)

from cold_start_replenishment.analogs.direct_mixture_learning import (  # noqa: E402
    MixtureDecisionTask,
)
from cold_start_replenishment.analogs.factorized_transfer import (  # noqa: E402
    StrictRelationPolicy,
    strict_relation_distribution,
)
from cold_start_replenishment.data.favorita import frozen_item_roles  # noqa: E402
from cold_start_replenishment.data.uci_online_retail_ii import (  # noqa: E402
    load_online_retail_ii,
)
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level  # noqa: E402

_TRAINING = None
_VALIDATION = None


def _init_fit_worker(training, validation):
    global _TRAINING, _VALIDATION
    _TRAINING, _VALIDATION = training, validation


def _fit_job(arguments):
    seed, method, shared, anchored, checkpoint, search_budget = arguments
    policy = StrictRelationPolicy(
        int(seed),
        shared=shared,
        similarity_residual=anchored,
        search_budget=search_budget,
        nested_candidate_stream=search_budget is not None,
    ).fit(_TRAINING, _VALIDATION)
    selected = next(row for row in policy.search_ if row["selected"])
    result = {
        "seed": int(seed),
        "method": method,
        "shared": bool(shared),
        "anchored": bool(anchored),
        "coefficients": policy.coefficients_.tolist(),
        "contraction_alpha": float(policy.contraction_alpha_),
        "occurrence_logit_shift": float(policy.occurrence_logit_shift_),
        "selected_candidate": int(selected["candidate"]),
        "training_objective": float(selected["training_objective"]),
        "validation_objective": float(selected["validation_objective"]),
        "effective_relation_parameters": int(policy.effective_relation_parameters),
        "effective_continuous_parameters": int(policy.effective_continuous_parameters),
        "search_budget": search_budget,
        "candidate_slots": int(len(policy.candidates())),
        "unique_candidates": int(len(np.unique(policy.candidates(), axis=0))),
        "training_evaluations": int(
            len(np.unique(policy.candidates(), axis=0)) * len(policy.contraction_grid)
        ),
        "validation_evaluations": int(
            policy.shortlist_size * len(policy.contraction_grid)
        ),
    }
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True) + "\n")
    temporary.replace(checkpoint)
    return result


def _fit_ensemble(
    training, validation, seeds, *, anchored, dataset, search_budget: int | None = None
):
    policies = {"strict_shared": [], "strict_separate": []}
    fits = []
    jobs = []
    results = []
    for seed in seeds:
        for method, shared in (("strict_shared", True), ("strict_separate", False)):
            checkpoint = (
                OUTPUT
                / "fit_checkpoints"
                / f"{dataset.lower()}_{seed}_{method}.json"
            )
            if checkpoint.exists():
                results.append(json.loads(checkpoint.read_text()))
            else:
                jobs.append(
                    (seed, method, shared, anchored, str(checkpoint), search_budget)
                )
    if jobs:
        with ProcessPoolExecutor(
            max_workers=min(6, len(jobs)),
            initializer=_init_fit_worker,
            initargs=(training, validation),
        ) as executor:
            results.extend(executor.map(_fit_job, jobs))
    for result in results:
        seed = int(result["seed"])
        method = str(result["method"])
        policy = StrictRelationPolicy(
            int(seed),
            shared=bool(result["shared"]),
            similarity_residual=bool(result["anchored"]),
            search_budget=search_budget,
            nested_candidate_stream=search_budget is not None,
        )
        policy.coefficients_ = np.asarray(result["coefficients"], float)
        policy.contraction_alpha_ = float(result["contraction_alpha"])
        policy.occurrence_logit_shift_ = float(result["occurrence_logit_shift"])
        policies[method].append(policy)
        fits.append(
            {
                "seed": seed,
                "method": method,
                "selected_candidate": result["selected_candidate"],
                "contraction_alpha": policy.contraction_alpha_,
                "occurrence_logit_shift": policy.occurrence_logit_shift_,
                "training_objective": result["training_objective"],
                "validation_objective": result["validation_objective"],
                "effective_relation_parameters": result[
                    "effective_relation_parameters"
                ],
                "effective_continuous_parameters": result[
                    "effective_continuous_parameters"
                ],
                "search_budget": result.get("search_budget"),
                "candidate_slots": result.get("candidate_slots", 100),
                "unique_candidates": result.get("unique_candidates", 100),
                "training_evaluations": result.get("training_evaluations", 400),
                "validation_evaluations": result.get("validation_evaluations", 48),
                **{
                    f"coefficient_{i}": float(value)
                    for i, value in enumerate(policy.coefficients_)
                },
            }
        )
    for method in policies:
        policies[method].sort(key=lambda policy: policy.random_seed)
    return policies, fits


def _ensemble(task, policies):
    values, masses, events = [], [], []
    for policy in policies:
        v, m, d = strict_relation_distribution(
            task,
            policy.coefficients_,
            policy.occurrence_logit_shift_,
            shared=policy.shared,
            contraction_alpha=policy.contraction_alpha_,
            similarity_residual=policy.similarity_residual,
        )
        values.extend(v)
        masses.extend(m / len(policies))
        events.append(d["event_probability"])
    return np.asarray(values), np.asarray(masses), float(np.mean(events))


def _score(task, method, values, masses, event):
    action = float(optimal_feasible_level(values, masses, 1.0, float(task.cost_ratio))[0])
    actual = float(task.actual_demand)
    ratio = float(task.cost_ratio)
    holding = max(action - actual, 0.0)
    shortage = ratio * max(actual - action, 0.0)
    return {
        "method": method,
        "actual": actual,
        "action": action,
        "cost": holding + shortage,
        "holding_cost": holding,
        "shortage_cost": shortage,
        "fixed_cost": 0.0,
        "service": 1.0 if actual <= 0 else min(action, actual) / actual,
        "zero_action": float(action <= 0),
        "event_probability": event,
    }


def run_uci(protocol, *, search_budget: int | None = None):
    base = yaml.safe_load((ROOT / "configs/uci_online_retail_ii_external.yaml").read_text())
    frozen = yaml.safe_load((ROOT / "configs/mechanism_confirmation_v3.yaml").read_text())[
        "uci_confirmation_v3"
    ]
    panel = load_online_retail_ii(ROOT / base["source"]["workbook"])
    ordered = _ordered_eligible(panel, base)
    confirmation_manifest = pd.read_csv(
        ROOT / "outputs/ai_darld_v3/uci_confirmation_v3_split_manifest.csv"
    )
    old = pd.read_csv(ROOT / "outputs/ai_darld_v3/uci_online_retail_ii_split_manifest.csv")
    id_to_index = {value: index for index, value in enumerate(panel.product_ids)}
    development = np.asarray(
        [id_to_index[x] for x in old.loc[old.role.eq("development"), "product_id"]], int
    )
    validation = np.asarray(
        [id_to_index[x] for x in old.loc[old.role.eq("validation"), "product_id"]], int
    )
    targets = np.asarray([id_to_index[x] for x in confirmation_manifest.product_id], int)
    expected = set(ordered.product_id.astype(str))
    assert set(confirmation_manifest.product_id.astype(str)) <= expected
    similarity, x, _, _ = _fit_text(
        panel.descriptions, development, int(base["global"]["text_svd_components"])
    )
    policy_dev = development[: int(frozen["policy_development_products"])]
    training = _uci_tasks(
        policy_dev,
        development,
        similarity,
        panel.quantities,
        base,
        base["tasks"]["development_cutoffs"],
        x,
    )
    validation_tasks = _uci_tasks(
        validation,
        development,
        similarity,
        panel.quantities,
        base,
        base["tasks"]["development_cutoffs"],
        x,
    )
    policies, fits = _fit_ensemble(
        training,
        validation_tasks,
        protocol["datasets"]["UCI"]["policy_seeds"],
        anchored=True,
        dataset="UCI",
        search_budget=search_budget,
    )
    rows = []
    seed_rows = []
    k = int(base["tasks"]["donors_per_target"])
    atoms = int(base["tasks"]["donor_empirical_windows"])
    for target in targets:
        donors, similarity_values = _uci_nearest(similarity, int(target), development, k)
        for cutoff in frozen["cutoffs"]:
            histories = [np.asarray(panel.quantities[d, :cutoff], float) for d in donors]
            for horizon in frozen["demand_windows_weeks"]:
                scenarios = np.vstack(
                    [
                        _horizon_atoms(panel.quantities[d], int(cutoff), int(horizon), atoms)
                        for d in donors
                    ]
                )
                actual = float(panel.quantities[target, cutoff : cutoff + horizon].sum())
                for ratio in frozen["shortage_holding_ratios"]:
                    task = MixtureDecisionTask(
                        str(target),
                        _uci_features(similarity_values, scenarios, histories),
                        scenarios,
                        actual,
                        float(ratio),
                        int(horizon),
                        None,
                        None,
                        1.0,
                        0.0,
                        _support_matrix(histories, int(horizon), atoms),
                    )
                    for method in policies:
                        values, masses, event = _ensemble(task, policies[method])
                        rows.append(
                            {
                                "dataset": "UCI",
                                "target_id": panel.product_ids[target],
                                "cutoff": cutoff,
                                "horizon": horizon,
                                "cost_ratio": ratio,
                                **_score(task, method, values, masses, event),
                            }
                        )
                        if search_budget is not None:
                            for policy in policies[method]:
                                values, masses, diagnostics = strict_relation_distribution(
                                    task,
                                    policy.coefficients_,
                                    policy.occurrence_logit_shift_,
                                    shared=policy.shared,
                                    contraction_alpha=policy.contraction_alpha_,
                                    similarity_residual=policy.similarity_residual,
                                )
                                seed_rows.append(
                                    {
                                        "dataset": "UCI",
                                        "target_id": panel.product_ids[target],
                                        "cutoff": cutoff,
                                        "horizon": horizon,
                                        "cost_ratio": ratio,
                                        "policy_seed": policy.random_seed,
                                        **_score(
                                            task,
                                            method,
                                            values,
                                            masses,
                                            diagnostics["event_probability"],
                                        ),
                                    }
                                )
    return pd.DataFrame(rows), pd.DataFrame(fits), pd.DataFrame(seed_rows)


def run_favorita(protocol, *, search_budget: int | None = None):
    cfg = yaml.safe_load((ROOT / "configs/favorita_untouched_confirmation_v1.yaml").read_text())
    panel = _prepare_panel(cfg)
    item_to_index = {int(item): i for i, item in enumerate(panel.item_ids)}
    role_items = frozen_item_roles(
        panel.item_ids,
        salt=cfg["roles"]["salt"],
        confirmation_count=int(cfg["roles"]["confirmation_target_count"]),
    )
    roles = {
        name: np.asarray([item_to_index[int(item)] for item in values], int)
        for name, values in role_items.items()
    }
    cutoff_indices = [_cutoff_index(panel.dates, value) for value in cfg["tasks"]["cutoffs"]]
    encoded, _, _ = _fit_metadata(panel, roles["development"])
    # Reuse the frozen pseudo-target identities recorded by the authoritative run.
    selection = pd.read_csv(
        ROOT / "outputs/ai_darld_v3/favorita_confirmation_v1/policy_selection.csv"
    )
    pseudo_count = int(cfg["roles"]["pseudo_target_count_per_cutoff"])
    pseudo = _pseudo_targets(
        roles["development"],
        panel.item_ids,
        cfg["roles"]["salt"] + "|pseudo-target",
        pseudo_count,
    )
    train_count = int(
        round(
            len(pseudo)
            * float(cfg["methods"]["component_specific_policy"]["pseudo_target_split"][0])
        )
    )
    validation_count = int(
        round(
            len(pseudo)
            * float(cfg["methods"]["component_specific_policy"]["pseudo_target_split"][1])
        )
    )
    training = _favorita_tasks(
        pseudo[:train_count], roles["development"], encoded, panel.quantities, cutoff_indices, cfg
    )
    validation = _favorita_tasks(
        pseudo[train_count : train_count + validation_count],
        roles["development"],
        encoded,
        panel.quantities,
        cutoff_indices,
        cfg,
    )
    policies, fits = _fit_ensemble(
        training,
        validation,
        protocol["datasets"]["Favorita"]["policy_seeds"],
        anchored=False,
        dataset="Favorita",
        search_budget=search_budget,
    )
    assert len(selection) > 0
    rows = []
    seed_rows = []
    task_cfg = cfg["tasks"]
    for target in roles["confirmation"]:
        for cutoff_label, cutoff in zip(task_cfg["cutoffs"], cutoff_indices, strict=True):
            eligible = _eligible_donors(
                panel.quantities,
                roles["development"],
                cutoff,
                int(task_cfg["required_history_days"]),
                int(cfg["eligibility"]["donor_minimum_positive_history_days"]),
            )
            donors, similarity = _favorita_nearest(
                encoded, int(target), eligible, int(task_cfg["donors_per_target"])
            )
            for horizon in task_cfg["demand_horizons_days"]:
                for ratio in task_cfg["shortage_holding_ratios"]:
                    task = _favorita_task(
                        int(target),
                        donors,
                        similarity,
                        panel.quantities,
                        cutoff,
                        int(horizon),
                        float(ratio),
                        int(task_cfg["required_history_days"]),
                        int(task_cfg["donor_empirical_atoms"]),
                    )
                    for method in policies:
                        values, masses, event = _ensemble(task, policies[method])
                        rows.append(
                            {
                                "dataset": "Favorita",
                                "target_id": int(panel.item_ids[target]),
                                "cutoff": cutoff_label,
                                "horizon": horizon,
                                "cost_ratio": ratio,
                                **_score(task, method, values, masses, event),
                            }
                        )
                        if search_budget is not None:
                            for policy in policies[method]:
                                values, masses, diagnostics = strict_relation_distribution(
                                    task,
                                    policy.coefficients_,
                                    policy.occurrence_logit_shift_,
                                    shared=policy.shared,
                                    contraction_alpha=policy.contraction_alpha_,
                                    similarity_residual=policy.similarity_residual,
                                )
                                seed_rows.append(
                                    {
                                        "dataset": "Favorita",
                                        "target_id": int(panel.item_ids[target]),
                                        "cutoff": cutoff_label,
                                        "horizon": horizon,
                                        "cost_ratio": ratio,
                                        "policy_seed": policy.random_seed,
                                        **_score(
                                            task,
                                            method,
                                            values,
                                            masses,
                                            diagnostics["event_probability"],
                                        ),
                                    }
                                )
    return pd.DataFrame(rows), pd.DataFrame(fits), pd.DataFrame(seed_rows)


def _interval(merged, column, draws, seed):
    cluster = merged.groupby("target_id")[column].mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws_array = rng.choice(cluster, size=(draws, len(cluster)), replace=True).mean(axis=1)
    return float(np.quantile(draws_array, 0.025)), float(np.quantile(draws_array, 0.975))


def summarize(rows, protocol):
    results = []
    for dataset in sorted(rows.dataset.unique()):
        subset = rows[rows.dataset.eq(dataset)]
        key = [
            c
            for c in ["dataset", "target_id", "cutoff", "horizon", "cost_ratio"]
            if c in subset.columns
        ]
        left = subset[subset.method.eq("strict_separate")].drop(columns="method")
        right = subset[subset.method.eq("strict_shared")].drop(columns="method")
        merged = left.merge(right, on=key, suffixes=("_separate", "_shared"), validate="one_to_one")
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"incomplete pairing for {dataset}")
        merged["cost_difference"] = merged.cost_separate - merged.cost_shared
        merged["service_difference"] = merged.service_separate - merged.service_shared
        draws = int(protocol["datasets"][dataset]["bootstrap_draws"])
        low, high = _interval(merged, "cost_difference", draws, 20261005)
        slow, shigh = _interval(merged, "service_difference", draws, 20261006)
        base = float(merged.cost_shared.mean())
        results.append(
            {
                "dataset": dataset,
                "separate_mean_cost": float(merged.cost_separate.mean()),
                "shared_mean_cost": base,
                "cost_difference": float(merged.cost_difference.mean()),
                "relative_difference_percent": float(100 * merged.cost_difference.mean() / base),
                "ci_low": low,
                "ci_high": high,
                "separate_service": float(merged.service_separate.mean()),
                "shared_service": float(merged.service_shared.mean()),
                "service_difference": float(merged.service_difference.mean()),
                "service_ci_low": slow,
                "service_ci_high": shigh,
                "separate_mean_action": float(merged.action_separate.mean()),
                "shared_mean_action": float(merged.action_shared.mean()),
                "separate_zero_order_rate": float(merged.zero_action_separate.mean()),
                "shared_zero_order_rate": float(merged.zero_action_shared.mean()),
                "products": int(merged.target_id.nunique()),
                "matched_rows": len(merged),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(results)


def _ensemble_row_files(output: Path) -> list[Path]:
    """Return ensemble rows without per-seed sensitivity rows."""
    return [
        path
        for path in sorted(output.glob("*_rows.parquet"))
        if path.stem in {"uci_rows", "favorita_rows"}
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["UCI", "Favorita", "all"], default="all")
    parser.add_argument("--search-budget", type=int, choices=[100, 200])
    args = parser.parse_args()
    started = time.time()
    global OUTPUT, CONFIG
    if args.search_budget is not None:
        CONFIG = ROOT / "configs/strict_search_stability_v1.yaml"
        OUTPUT = (
            ROOT
            / "outputs/ai_darld_v3/strict_search_stability_v1"
            / f"budget_{args.search_budget}"
        )
    protocol = yaml.safe_load(CONFIG.read_text())
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    selected = ["UCI", "Favorita"] if args.dataset == "all" else [args.dataset]
    for dataset in selected:
        rows, fits, seed_rows = (
            run_uci(protocol, search_budget=args.search_budget)
            if dataset == "UCI"
            else run_favorita(protocol, search_budget=args.search_budget)
        )
        rows.to_parquet(output / f"{dataset.lower()}_rows.parquet", index=False)
        fits.to_csv(output / f"{dataset.lower()}_fits.csv", index=False)
        if args.search_budget is not None:
            seed_rows.to_parquet(
                output / f"{dataset.lower()}_seed_rows.parquet", index=False
            )
    rows = pd.concat(
        [
            pd.read_parquet(path)
            for path in _ensemble_row_files(output)
        ],
        ignore_index=True,
    )
    summarize(rows, protocol).to_csv(output / "retail_paired_contrasts.csv", index=False)
    (output / "retail_run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "datasets": selected,
                "search_budget": args.search_budget,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
