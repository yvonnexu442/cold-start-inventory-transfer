#!/usr/bin/env python3
"""Run the frozen strict coefficient-tying comparison on formal populations."""

from __future__ import annotations

import argparse
import hashlib
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

from run_ai_darld_v3_direct_mixture import _historical_tasks, _split  # noqa: E402
from run_ai_darld_v3_factorized import (  # noqa: E402
    POLICY_DEVELOPMENT_TARGETS,
    _evaluate,
    _population,
    _select,
)

from cold_start_replenishment.analogs.factorized_transfer import (  # noqa: E402
    FactorizedTransferPolicy,
    StrictRelationPolicy,
)
from cold_start_replenishment.data.spdf_pilot import (  # noqa: E402
    parse_braf,
    parse_man,
    redacted_id,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_service_job(arguments):
    seed, cutoff, name, shared, training, validation, checkpoint, search_budget = arguments
    policy = StrictRelationPolicy(
        seed + cutoff,
        shared=shared,
        similarity_residual=True,
        search_budget=search_budget,
        nested_candidate_stream=search_budget is not None,
    ).fit(training, validation)
    chosen = next(row for row in policy.search_ if row["selected"])
    result = {
        "seed": int(seed),
        "cutoff": int(cutoff),
        "method": name,
        "shared": bool(shared),
        "coefficients": policy.coefficients_.tolist(),
        "contraction_alpha": float(policy.contraction_alpha_),
        "occurrence_logit_shift": float(policy.occurrence_logit_shift_),
        "selected_candidate": int(chosen["candidate"]),
        "training_objective": float(chosen["training_objective"]),
        "validation_objective": float(chosen["validation_objective"]),
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


def _fit_service(dataset, full, v3, protocol, *, search_budget: int | None = None):
    fitted, records = {}, []
    seeds = list(map(int, protocol["datasets"][dataset.name]["policy_seeds"]))
    for cutoff in map(int, full["datasets"][dataset.name]["cutoffs"]):
        tasks, _, _, _ = _historical_tasks(
            dataset, full, v3, cutoff, scenario_count=120, random_seed=int(v3["random_seed"])
        )
        ids = np.asarray(sorted({task.target_id for task in tasks}))
        ids = np.random.default_rng(int(v3["random_seed"]) + cutoff).permutation(ids)[
            : min(POLICY_DEVELOPMENT_TARGETS, len(ids))
        ]
        tasks = _select(tasks, set(ids))
        fitted[cutoff] = {"strict_shared": [], "strict_separate": []}
        jobs = []
        cached_results = []
        for seed in seeds:
            train_ids, validation_ids, _ = _split(ids, seed + cutoff)
            for name, shared in (("strict_shared", True), ("strict_separate", False)):
                checkpoint = (
                    OUTPUT
                    / "fit_checkpoints"
                    / f"{dataset.name.lower()}_{cutoff}_{seed}_{name}.json"
                )
                if checkpoint.exists():
                    cached_results.append(json.loads(checkpoint.read_text()))
                else:
                    jobs.append(
                        (
                            seed,
                            cutoff,
                            name,
                            shared,
                            _select(tasks, train_ids),
                            _select(tasks, validation_ids),
                            str(checkpoint),
                            search_budget,
                        )
                    )
        if jobs:
            with ProcessPoolExecutor(max_workers=min(6, len(jobs))) as executor:
                cached_results.extend(executor.map(_fit_service_job, jobs))
        for result in cached_results:
            seed = int(result["seed"])
            name = str(result["method"])
            shared = bool(result["shared"])
            policy = StrictRelationPolicy(
                seed + cutoff,
                shared=shared,
                similarity_residual=True,
                search_budget=search_budget,
                nested_candidate_stream=search_budget is not None,
            )
            policy.coefficients_ = np.asarray(result["coefficients"], float)
            policy.contraction_alpha_ = float(result["contraction_alpha"])
            policy.occurrence_logit_shift_ = float(result["occurrence_logit_shift"])
            fitted[cutoff][name].append(policy)
            records.append(
                {
                    "dataset": dataset.name,
                    "cutoff": cutoff,
                    "seed": seed,
                    "method": name,
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
        for name in ("strict_shared", "strict_separate"):
            fitted[cutoff][name].sort(key=lambda policy: policy.random_seed)
        authority = pd.read_csv(
            ROOT
            / f"outputs/ai_darld_v3/historical_support_v4_parameters_{dataset.name.lower()}.csv"
        )
        variants = {
            "factorized_transfer": (False, True, False),
            "shared_hurdle": (True, True, False),
            "factorized_no_contraction": (False, False, False),
            "factorized_similarity_residual": (False, True, True),
            "shared_similarity_residual": (True, True, True),
            "anchored_no_contraction_refit": (False, False, True),
        }
        for name, (shared, contraction, residual) in variants.items():
            fitted[cutoff][name] = []
            selected = authority[(authority.cutoff == cutoff) & (authority.policy == name)]
            for row in selected.itertuples(index=False):
                policy = FactorizedTransferPolicy(
                    int(row.seed),
                    shared=shared,
                    contraction=contraction,
                    similarity_residual=residual,
                )
                policy.parameters_ = np.asarray(
                    [getattr(row, f"theta_{index}") for index in range(10)], float
                )
                policy.occurrence_logit_shift_ = float(row.occurrence_logit_shift)
                fitted[cutoff][name].append(policy)
    return fitted, records


def _cluster_interval(
    frame: pd.DataFrame, value: str, draws: int, seed: int
) -> tuple[float, float]:
    cluster = frame.groupby("target_id")[value].mean().to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(cluster, size=(draws, len(cluster)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _summarize(rows: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = rows[rows.perturbation.eq("none")].copy()
    summary = clean.groupby(["dataset", "method"], as_index=False).agg(
        mean_cost=("total_cost", "mean"),
        service=("fill_rate_proxy", "mean"),
        mean_action=("selected_level", "mean"),
        zero_order_rate=("selected_level", lambda x: float(np.mean(x <= 0))),
        rows=("total_cost", "size"),
        products=("target_id", "nunique"),
    )
    comparisons = []
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
    for dataset in sorted(clean.dataset.unique()):
        left = clean[(clean.dataset == dataset) & (clean.method == "strict_separate")]
        right = clean[(clean.dataset == dataset) & (clean.method == "strict_shared")]
        merged = left.merge(
            right, on=keys, suffixes=("_separate", "_shared"), validate="one_to_one"
        )
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"incomplete strict pairing for {dataset}")
        merged["cost_difference"] = merged.total_cost_separate - merged.total_cost_shared
        merged["service_difference"] = (
            merged.fill_rate_proxy_separate - merged.fill_rate_proxy_shared
        )
        draws = int(protocol["datasets"][dataset]["bootstrap_draws"])
        low, high = _cluster_interval(merged, "cost_difference", draws, 20261005)
        slow, shigh = _cluster_interval(merged, "service_difference", draws, 20261006)
        shared_cost = float(merged.total_cost_shared.mean())
        comparisons.append(
            {
                "dataset": dataset,
                "first_policy": "strict_separate",
                "comparator": "strict_shared",
                "separate_mean_cost": float(merged.total_cost_separate.mean()),
                "shared_mean_cost": shared_cost,
                "cost_difference": float(merged.cost_difference.mean()),
                "relative_difference_percent": float(
                    100 * merged.cost_difference.mean() / shared_cost
                ),
                "ci_low": low,
                "ci_high": high,
                "separate_service": float(merged.fill_rate_proxy_separate.mean()),
                "shared_service": float(merged.fill_rate_proxy_shared.mean()),
                "service_difference": float(merged.service_difference.mean()),
                "service_ci_low": slow,
                "service_ci_high": shigh,
                "separate_mean_action": float(merged.selected_level_separate.mean()),
                "shared_mean_action": float(merged.selected_level_shared.mean()),
                "separate_zero_order_rate": float(np.mean(merged.selected_level_separate <= 0)),
                "shared_zero_order_rate": float(np.mean(merged.selected_level_shared <= 0)),
                "products": int(merged.target_id.nunique()),
                "matched_rows": len(merged),
                "bootstrap_draws": draws,
            }
        )
    return summary, pd.DataFrame(comparisons)


def _ensemble_row_files(output: Path) -> list[Path]:
    """Return ensemble rows without per-seed sensitivity rows."""
    return [
        path
        for path in sorted(output.glob("*_rows.parquet"))
        if path.stem in {"man_rows", "braf_rows"}
    ]


def run_service(
    dataset_name: str, protocol: dict, *, search_budget: int | None = None
) -> None:
    full = yaml.safe_load((ROOT / "configs/full_scale.yaml").read_text())
    v3 = yaml.safe_load((ROOT / "configs/ai_darld_v3.yaml").read_text())
    dataset = parse_man() if dataset_name == "MAN" else parse_braf()
    policies, fits = _fit_service(
        dataset, full, v3, protocol, search_budget=search_budget
    )
    checkpoint = pd.read_parquet(
        ROOT / f"outputs/full_scale/checkpoints/{dataset.name.lower()}_reliability.parquet"
    )
    targets, eligible = _population(dataset, full)
    targets = np.asarray(targets[:250])
    target_ids = {
        redacted_id(dataset.name, dataset.metadata.iloc[int(source)].item_id) for source in targets
    }
    scored = checkpoint[checkpoint.target_id.astype(str).isin(target_ids)].copy()
    rows = _evaluate(
        dataset,
        full,
        v3,
        scored,
        eligible,
        targets,
        policies,
        perturbations=("none",),
        only_methods=("strict_shared", "strict_separate"),
    )
    out = OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / f"{dataset_name.lower()}_rows.parquet", index=False)
    if search_budget is not None:
        seed_frames = []
        seeds = list(map(int, protocol["datasets"][dataset.name]["policy_seeds"]))
        for seed_index, seed in enumerate(seeds):
            seed_policies = {}
            for cutoff, variants in policies.items():
                seed_policies[cutoff] = {
                    name: (
                        [items[seed_index]]
                        if name in {"strict_shared", "strict_separate"}
                        else items
                    )
                    for name, items in variants.items()
                }
            seed_rows = _evaluate(
                dataset,
                full,
                v3,
                scored,
                eligible,
                targets,
                seed_policies,
                perturbations=("none",),
                only_methods=("strict_shared", "strict_separate"),
            )
            seed_rows["policy_seed"] = seed
            seed_frames.append(seed_rows)
        pd.concat(seed_frames, ignore_index=True).to_parquet(
            out / f"{dataset_name.lower()}_seed_rows.parquet", index=False
        )
    pd.DataFrame(fits).to_csv(out / f"{dataset_name.lower()}_fits.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MAN", "BRAF", "all"], default="all")
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
    config_path = CONFIG
    protocol = yaml.safe_load(config_path.read_text())
    chosen = ["MAN", "BRAF"] if args.dataset == "all" else [args.dataset]
    for dataset in chosen:
        run_service(dataset, protocol, search_budget=args.search_budget)
    out = OUTPUT
    row_files = _ensemble_row_files(out)
    rows = pd.concat([pd.read_parquet(path) for path in row_files], ignore_index=True)
    summary, comparisons = _summarize(rows, protocol)
    summary.to_csv(out / "summary.csv", index=False)
    comparisons.to_csv(out / "paired_contrasts.csv", index=False)
    manifest = {
        "status": "partial_service_parts"
        if set(chosen) != {"MAN", "BRAF"}
        else "service_parts_completed",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha(config_path),
        "source_commit": "8e6e467958889b3a1c80aed6da1f7256bd9af98d",
        "datasets_completed": chosen,
        "search_budget": args.search_budget,
        "elapsed_seconds": time.time() - started,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
