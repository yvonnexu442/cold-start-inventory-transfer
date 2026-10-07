"""Prespecified post-primary baseline and sensitivity experiments."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from sklearn.metrics import brier_score_loss, roc_auc_score  # type: ignore[import-untyped]

from cold_start_replenishment.analogs.reliability import (
    ReliabilityCalibrator,
    expected_calibration_error,
)
from cold_start_replenishment.data.spdf_pilot import (
    PilotDataset,
    parse_braf,
    parse_man,
    redacted_id,
)
from cold_start_replenishment.demand.reliability_scenarios import normalized_capped_weights
from cold_start_replenishment.evaluation.acceptance_enhancement import (
    _construct_scenarios,
    _markdown_table,
    _prepare_dataset,
    _write_summary,
)
from cold_start_replenishment.evaluation.decision_evaluator import HeldOutDecisionEvaluator
from cold_start_replenishment.evaluation.sprint2 import (
    _build_reliability_training,
    _eligible_sources,
    _operational_values,
    _similarity_rows,
)
from cold_start_replenishment.evaluation.sprint2_statistics import (
    clustered_bootstrap_difference,
)
from cold_start_replenishment.framework.objects import OperationalContext
from cold_start_replenishment.inventory.newsvendor import NewsvendorOptimizer
from cold_start_replenishment.paths import resolve_repo_path

METHODS = (
    "single_analog",
    "uniform_multi",
    "shuffled_forecast_reliability",
    "forecast_reliability_only",
    "forecast_combined",
    "shuffled_decision_reliability",
    "decision_reliability_only",
    "decision_combined",
)

METRICS = (
    "oracle_cost_normalized_regret",
    "raw_regret",
    "realized_demand_scaled_regret",
    "donor_pool_scaled_regret",
)


def load_targeted_extension_config() -> dict[str, Any]:
    """Load the outcome-frozen extension configuration."""

    with resolve_repo_path("configs/targeted_acceptance_extension.yaml").open(
        encoding="utf-8"
    ) as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))


def validate_targeted_extension_protocol(config: dict[str, Any] | None = None) -> dict[str, bool]:
    """Return deterministic checks for the frozen protocol."""

    frozen = config or load_targeted_extension_config()
    decomposition = frozen["matched_decomposition"]
    supervision = frozen["decision_reliability_supervision"]
    return {
        "status_frozen": frozen["status"] == "FROZEN BEFORE TARGETED EXTENSION EXECUTION",
        "datasets_fixed": frozen["datasets"] == ["MAN", "BRAF"],
        "matched_pool": decomposition["top_k"] == 10,
        "matched_scenarios": decomposition["scenario_count"] == 300,
        "all_methods_declared": set(METHODS) == set(decomposition["methods"]),
        "grouped_cross_fitting": supervision["folds"] == 5,
        "evaluation_tuning_forbidden": supervision["tuning_on_evaluation_targets"] == "forbidden",
        "bootstrap_clustered": frozen["statistics"]["cluster"] == "target_id",
        "all_results_retained": frozen["reporting_rules"]["retain_all_prespecified_results"]
        is True,
    }


def _selection_sources(
    dataset: PilotDataset, full_config: dict[str, Any]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    section = full_config["datasets"][dataset.name]
    eligible = _eligible_sources(dataset, section)
    rng = np.random.default_rng(
        int(full_config["random_seed"]) + (1 if dataset.name == "MAN" else 2)
    )
    shuffled = rng.permutation(eligible)
    target_count = int(section["sample_size"])
    pseudo_count = int(section["pseudo_target_count"])
    targets = np.sort(shuffled[:target_count])
    pseudo = np.sort(shuffled[target_count : target_count + pseudo_count])
    return eligible, targets, pseudo


def _decision_cost_labels(
    training: pd.DataFrame,
    dataset: PilotDataset,
    cutoff: int,
    scenario_count: int,
    ratios: list[float],
    seed: int,
) -> pd.DataFrame:
    """Replace forecast-error labels with frozen pseudo-target decision-cost labels."""

    demand = dataset.demand.to_numpy(dtype=float)
    id_to_source = {
        redacted_id(dataset.name, row.item_id): int(str(index))
        for index, row in dataset.metadata.iterrows()
    }
    labeled = training.copy()
    costs = np.zeros(len(labeled), dtype=float)
    optimizer = NewsvendorOptimizer()
    evaluator = HeldOutDecisionEvaluator()
    for position, row in enumerate(labeled.itertuples(index=False)):
        donor_id = str(row.donor_id)
        donor_source = id_to_source[donor_id]
        training_cutoff = int(str(row.reliability_training_cutoff))
        actual = float(str(row.pseudo_target_actual_lead_demand))
        lead = max(1, int(str(row.validation_end_cutoff)) - training_cutoff)
        # Match the source builder: native lead is capped by the validation window.
        pseudo_source = id_to_source[str(row.pseudo_target_id)]
        native = max(1, int(round(float(dataset.metadata.iloc[pseudo_source]["lead_time"]))))
        lead = min(native, lead)
        history = np.asarray(demand[donor_source, :training_cutoff], dtype=float)
        scenario = _construct_scenarios(
            [donor_id],
            [history],
            np.ones(1),
            lead,
            scenario_count,
            seed + cutoff * 1009 + pseudo_source * 31 + donor_source,
            "decision_label_single_donor",
        )
        ratio_costs: list[float] = []
        for ratio in ratios:
            context = OperationalContext(
                str(row.pseudo_target_id),
                lead,
                1.0,
                float(ratio),
                None,
                None,
                None,
                0.0,
            )
            decision = optimizer.optimize(scenario, context)
            ratio_costs.append(evaluator.evaluate(decision, actual, context).total_cost)
        costs[position] = float(np.mean(ratio_costs))
    labeled["decision_supervision_cost"] = costs
    labeled["useful"] = labeled.groupby("pseudo_target_id")["decision_supervision_cost"].transform(
        lambda values: (values <= values.median()).astype(int)
    )
    return labeled


def _fit_decision_reliability(
    dataset: PilotDataset,
    full_config: dict[str, Any],
    extension_config: dict[str, Any],
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    section = full_config["datasets"][dataset.name]
    eligible, targets, pseudo = _selection_sources(dataset, full_config)
    combined = np.concatenate([targets, pseudo])
    similarities = _similarity_rows(dataset, combined)
    lookup = {int(source): index for index, source in enumerate(combined)}
    excluded = {int(value) for value in targets}
    supervision = extension_config["decision_reliability_supervision"]
    diagnostics: list[dict[str, Any]] = []
    output: list[pd.DataFrame] = []
    for cutoff_value, group in scored.groupby("cutoff", sort=False):
        cutoff = int(str(cutoff_value))
        base_training = _build_reliability_training(
            dataset,
            cutoff,
            pseudo,
            excluded,
            eligible,
            similarities,
            lookup,
            section,
            int(extension_config["matched_decomposition"]["top_k"]),
        )
        training = _decision_cost_labels(
            base_training,
            dataset,
            cutoff,
            int(supervision["scenario_count"]),
            [float(value) for value in supervision["shortage_to_holding_ratios"]],
            int(extension_config["random_seed"]),
        )
        model = ReliabilityCalibrator(
            float(supervision["logistic_regularization_c"]),
            int(supervision["folds"]),
            int(extension_config["random_seed"]) + cutoff,
        ).fit(training)
        probabilities = np.asarray(model.cross_fitted_probabilities_, dtype=float)
        labels = training["useful"].to_numpy(dtype=int)
        diagnostics.append(
            {
                "dataset": dataset.name,
                "cutoff": cutoff,
                "pseudo_targets": training["pseudo_target_id"].nunique(),
                "rows": len(training),
                "positive_rate": labels.mean(),
                "brier_score": brier_score_loss(labels, probabilities),
                "ece": expected_calibration_error(probabilities, labels),
                "auc": roc_auc_score(labels, probabilities),
                "mean_decision_supervision_cost": training["decision_supervision_cost"].mean(),
            }
        )
        part = group.copy()
        part["decision_reliability_probability"] = model.predict(part)
        output.append(part)
    return pd.concat(output, ignore_index=True), pd.DataFrame(diagnostics)


def _method_weights(
    similarity: NDArray[np.float64],
    forecast: NDArray[np.float64],
    decision: NDArray[np.float64],
    support: NDArray[np.float64],
    method: str,
    cap: float,
    shuffled_forecast: NDArray[np.float64],
    shuffled_decision: NDArray[np.float64],
) -> NDArray[np.float64]:
    support_weight = np.log1p(support)
    if method == "single_analog":
        weights = np.zeros(len(similarity), dtype=float)
        weights[int(np.argmax(similarity))] = 1.0
        return weights
    if method == "uniform_multi":
        raw = np.ones(len(similarity), dtype=float)
    elif method == "shuffled_forecast_reliability":
        raw = shuffled_forecast * support_weight
    elif method == "forecast_reliability_only":
        raw = forecast * support_weight
    elif method == "forecast_combined":
        raw = similarity * forecast * support_weight
    elif method == "shuffled_decision_reliability":
        raw = shuffled_decision * support_weight
    elif method == "decision_reliability_only":
        raw = decision * support_weight
    elif method == "decision_combined":
        raw = similarity * decision * support_weight
    else:
        raise ValueError(f"Unknown targeted-extension method: {method}")
    return normalized_capped_weights(np.asarray(raw, dtype=np.float64), cap)


def _evaluate_dataset(
    dataset: PilotDataset,
    full_config: dict[str, Any],
    extension_config: dict[str, Any],
    scored: pd.DataFrame,
    target_sources: NDArray[np.int64],
    eligible: NDArray[np.int64],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    section = full_config["datasets"][dataset.name]
    decomposition = extension_config["matched_decomposition"]
    demand = dataset.demand.to_numpy(dtype=float)
    id_to_source = {
        redacted_id(dataset.name, row.item_id): int(str(index))
        for index, row in dataset.metadata.iterrows()
    }
    excluded = {int(value) for value in target_sources}
    allowed = np.asarray([source for source in eligible if int(source) not in excluded], dtype=int)
    target_order = {int(source): index for index, source in enumerate(target_sources)}
    count = int(decomposition["scenario_count"])
    cap = float(decomposition["maximum_donor_weight"])
    seed = int(extension_config["random_seed"])
    evaluator = HeldOutDecisionEvaluator()
    optimizer = NewsvendorOptimizer()
    rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    for (target_id, cutoff_value), pool in scored.groupby(["target_id", "cutoff"], sort=False):
        pool = pool.sort_values("similarity_rank")
        target_source = id_to_source[str(target_id)]
        order = target_order[target_source]
        cutoff = int(str(cutoff_value))
        donor_ids = pool["donor_id"].astype(str).tolist()
        donor_sources = [id_to_source[value] for value in donor_ids]
        histories = [np.asarray(demand[source, :cutoff], dtype=float) for source in donor_sources]
        similarity = pool["metadata_similarity"].to_numpy(dtype=float)
        forecast = pool["reliability_probability"].to_numpy(dtype=float)
        decision_reliability = pool["decision_reliability_probability"].to_numpy(dtype=float)
        support = pool["donor_history_support"].to_numpy(dtype=float)
        shuffle_forecast = np.random.default_rng(
            seed + cutoff * 1009 + order + 700_000
        ).permutation(forecast)
        shuffle_decision = np.random.default_rng(
            seed + cutoff * 1009 + order + 900_000
        ).permutation(decision_reliability)
        native = min(
            max(1, int(round(float(dataset.metadata.iloc[target_source]["lead_time"])))),
            int(section["native_lead_time_cap"]),
        )
        lead_times = {
            "native_capped": native,
            "half_native_sensitivity": max(1, math.ceil(native / 2)),
        }
        holding, fixed, moq = _operational_values(dataset, target_source, eligible, section)
        broad_sources = np.sort(
            np.random.default_rng(seed + cutoff * 1009 + order).choice(
                allowed,
                size=min(int(full_config["representation"]["broad_pool_size"]), len(allowed)),
                replace=False,
            )
        )
        broad_ids = [
            redacted_id(dataset.name, dataset.metadata.iloc[source]["item_id"])
            for source in broad_sources
        ]
        broad_histories = [
            np.asarray(demand[source, :cutoff], dtype=float) for source in broad_sources
        ]
        for lead_regime, lead in lead_times.items():
            actual = float(demand[target_source, cutoff : cutoff + lead].sum())
            donor_scale = max(
                float(np.median([history.mean() * lead for history in histories])), 1.0
            )
            broad = _construct_scenarios(
                broad_ids,
                broad_histories,
                np.full(len(broad_ids), 1 / len(broad_ids)),
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
            scenarios = {}
            for method in METHODS:
                weights = _method_weights(
                    similarity,
                    forecast,
                    decision_reliability,
                    support,
                    method,
                    cap,
                    shuffle_forecast,
                    shuffle_decision,
                )
                scenario = _construct_scenarios(
                    donor_ids,
                    histories,
                    weights,
                    lead,
                    count,
                    seed + order * 10007 + cutoff * 101 + 20_000,
                    method,
                )
                scenarios[method] = scenario
                lower, upper = np.quantile(scenario.lead_time_demand, [0.1, 0.9])
                scenario_rows.append(
                    {
                        "dataset": dataset.name,
                        "target_id": target_id,
                        "cutoff": cutoff,
                        "lead_time_regime": lead_regime,
                        "method": method,
                        "coverage": float(lower <= actual <= upper),
                        "scenario_diversity": float(scenario.lead_time_demand.std()),
                        **scenario.diagnostics,
                    }
                )
            oracle = _construct_scenarios(
                [str(target_id)],
                [np.asarray(demand[target_source, :cutoff], dtype=float)],
                np.ones(1),
                lead,
                count,
                seed + order * 10007 + cutoff * 101 + 999,
                "oracle_target_history",
            )
            for ratio in decomposition["shortage_to_holding_ratios"]:
                for capacity_regime, capacity in capacities.items():
                    context = OperationalContext(
                        str(target_id),
                        lead,
                        holding,
                        holding * float(ratio),
                        capacity,
                        None,
                        moq,
                        fixed,
                    )
                    reference = optimizer.optimize(oracle, context)
                    for method, scenario in scenarios.items():
                        decision = optimizer.optimize(scenario, context)
                        evaluation = evaluator.evaluate(decision, actual, context, reference)
                        raw_regret = float(evaluation.regret or 0.0)
                        oracle_cost = evaluation.total_cost - raw_regret
                        rows.append(
                            {
                                "dataset": dataset.name,
                                "target_id": target_id,
                                "cutoff": cutoff,
                                "lead_time_regime": lead_regime,
                                "lead_time": lead,
                                "capacity_regime": capacity_regime,
                                "shortage_holding_ratio": float(ratio),
                                "method": method,
                                "actual_demand": actual,
                                "donor_pool_scale": donor_scale,
                                "selected_level": decision.level,
                                "total_cost": evaluation.total_cost,
                                "oracle_target_history_cost": oracle_cost,
                                "oracle_cost_normalized_regret": evaluation.normalized_regret,
                                "raw_regret": raw_regret,
                                "realized_demand_scaled_regret": raw_regret / max(actual, 1.0),
                                "donor_pool_scaled_regret": raw_regret / donor_scale,
                                "fill_rate_proxy": evaluation.fill_rate_proxy,
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(scenario_rows)


def _trimmed_mean(values: NDArray[np.float64], fraction: float) -> float:
    ordered = np.sort(values)
    trim = int(math.floor(len(ordered) * fraction))
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return float(retained.mean())


def _clustered_trimmed_interval(
    frame: pd.DataFrame,
    repetitions: int,
    seed: int,
    fraction: float,
) -> tuple[float, float, float]:
    target_effects = frame.groupby("target_id", sort=False)["difference"].mean()
    values = target_effects.to_numpy(dtype=float)
    estimate = _trimmed_mean(values, fraction)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = rng.choice(values, size=len(values), replace=True)
        draws[index] = _trimmed_mean(sample, fraction)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def _paired_summaries(
    results: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = config["matched_decomposition"]["primary_comparisons"]
    repetitions = int(config["statistics"]["bootstrap_repetitions"])
    seed = int(config["random_seed"])
    fraction = float(config["metric_robustness"]["trimming_fraction"])
    index = [
        "target_id",
        "cutoff",
        "lead_time_regime",
        "capacity_regime",
        "shortage_holding_ratio",
    ]
    standard_rows: list[dict[str, Any]] = []
    trimmed_rows: list[dict[str, Any]] = []
    for dataset, group in results.groupby("dataset"):
        for left, right in comparisons:
            selected = group[group["method"].isin([left, right])]
            for metric in METRICS:
                pivot = selected.pivot(index=index, columns="method", values=metric)
                if left not in pivot or right not in pivot:
                    continue
                paired = pivot[[left, right]].dropna().reset_index()
                paired["difference"] = paired[left] - paired[right]
                low, high = clustered_bootstrap_difference(
                    paired,
                    "difference",
                    repetitions=repetitions,
                    random_seed=seed + len(standard_rows),
                )
                differences = paired["difference"].to_numpy(dtype=float)
                target_effect = paired.groupby("target_id")["difference"].mean()
                sd = float(target_effect.std(ddof=1))
                standard_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{left} minus {right}",
                        "metric": metric,
                        "targets": paired["target_id"].nunique(),
                        "rows": len(paired),
                        "mean_difference": differences.mean(),
                        "median_difference": np.median(differences),
                        "ci_low": low,
                        "ci_high": high,
                        "win_rate": (differences < 0).mean(),
                        "standardized_target_effect": (
                            float(target_effect.mean() / sd) if sd > 0 else np.nan
                        ),
                    }
                )
            primary = selected.pivot(
                index=index, columns="method", values="oracle_cost_normalized_regret"
            )
            paired_primary = primary[[left, right]].dropna().reset_index()
            paired_primary["difference"] = paired_primary[left] - paired_primary[right]
            estimate, low, high = _clustered_trimmed_interval(
                paired_primary,
                repetitions,
                seed + len(trimmed_rows) + 50_000,
                fraction,
            )
            trimmed_rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{left} minus {right}",
                    "metric": "target_cluster_trimmed_mean_5_percent",
                    "targets": paired_primary["target_id"].nunique(),
                    "trimmed_mean_difference": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(standard_rows), pd.DataFrame(trimmed_rows)


def _method_summary(results: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    outcomes = results.groupby(["dataset", "method"], as_index=False).agg(
        mean_primary_regret=("oracle_cost_normalized_regret", "mean"),
        median_primary_regret=("oracle_cost_normalized_regret", "median"),
        mean_raw_regret=("raw_regret", "mean"),
        mean_demand_scaled_regret=("realized_demand_scaled_regret", "mean"),
        mean_pool_scaled_regret=("donor_pool_scaled_regret", "mean"),
        mean_cost=("total_cost", "mean"),
        fill_rate=("fill_rate_proxy", "mean"),
    )
    diagnostics = scenarios.groupby(["dataset", "method"], as_index=False).agg(
        coverage=("coverage", "mean"),
        diversity=("scenario_diversity", "mean"),
        effective_donors=("effective_donor_count", "mean"),
        maximum_donor_share=("maximum_donor_share", "mean"),
    )
    return outcomes.merge(diagnostics, on=["dataset", "method"], how="left")


def run_targeted_acceptance_extension() -> dict[str, Any]:
    """Run all outcome-frozen targeted extensions and write aggregate artifacts."""

    extension = load_targeted_extension_config()
    checks = validate_targeted_extension_protocol(extension)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Targeted extension protocol validation failed: {failed}")
    with resolve_repo_path("configs/full_scale.yaml").open(encoding="utf-8") as handle:
        full_config = cast(dict[str, Any], yaml.safe_load(handle))
    output = resolve_repo_path(str(extension["output_path"]))
    output.mkdir(parents=True, exist_ok=True)
    run_output = resolve_repo_path("outputs/runs/targeted_acceptance_extension")
    run_output.mkdir(parents=True, exist_ok=True)
    all_results: list[pd.DataFrame] = []
    all_scenarios: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    for dataset in (parse_man(), parse_braf()):
        checkpoint = pd.read_parquet(
            resolve_repo_path(
                f"outputs/full_scale/checkpoints/{dataset.name.lower()}_reliability.parquet"
            )
        )
        _, forecast_scored, target_sources, eligible = _prepare_dataset(
            dataset, full_config, checkpoint
        )
        scored, diagnostics = _fit_decision_reliability(
            dataset, full_config, extension, forecast_scored
        )
        results, scenarios = _evaluate_dataset(
            dataset, full_config, extension, scored, target_sources, eligible
        )
        all_results.append(results)
        all_scenarios.append(scenarios)
        all_diagnostics.append(diagnostics)
    results = pd.concat(all_results, ignore_index=True)
    scenarios = pd.concat(all_scenarios, ignore_index=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    paired, trimmed = _paired_summaries(results, extension)
    methods = _method_summary(results, scenarios)
    results.to_parquet(run_output / "decision_results.parquet", index=False)
    scenarios.to_parquet(run_output / "scenario_diagnostics.parquet", index=False)
    paired.to_csv(output / "paired_metric_robustness.csv", index=False)
    trimmed.to_csv(output / "trimmed_mean_robustness.csv", index=False)
    methods.to_csv(output / "method_summary.csv", index=False)
    diagnostics.to_csv(output / "decision_reliability_metrics.csv", index=False)
    primary = paired[paired["metric"] == "oracle_cost_normalized_regret"]
    _write_summary(
        output / "targeted_extension_summary.md",
        "Prespecified Baseline and Sensitivity Extension",
        str(extension["reporting_rules"]["label"]),
        "## Protocol checks\n\n"
        + _markdown_table(pd.DataFrame([checks]))
        + "\n\n## Primary matched comparisons\n\n"
        + _markdown_table(primary)
        + "\n\n## Alternative metric sensitivity\n\n"
        + _markdown_table(paired)
        + "\n\n## Target-cluster trimmed sensitivity\n\n"
        + _markdown_table(trimmed)
        + "\n\n## Decision-aligned reliability supervision\n\n"
        + _markdown_table(diagnostics)
        + "\n\n## Method outcomes\n\n"
        + _markdown_table(methods),
    )
    return {
        "results": results,
        "scenarios": scenarios,
        "paired": paired,
        "trimmed": trimmed,
        "methods": methods,
        "decision_reliability": diagnostics,
        "checks": checks,
        "output": output,
    }
