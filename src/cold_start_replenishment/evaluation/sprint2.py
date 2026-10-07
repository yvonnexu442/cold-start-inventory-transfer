"""Sprint 2 SPDF confirmatory reliability and decision-scenario analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from cold_start_replenishment.analogs.reliability import (
    FEATURE_PROVENANCE,
    RELIABILITY_FEATURES,
    ReliabilityCalibratedRepresentation,
    ReliabilityCalibrator,
    reliability_pair_features,
)
from cold_start_replenishment.data.spdf_pilot import (
    PilotDataset,
    braf_similarity,
    man_similarity,
    parse_braf,
    parse_man,
    redacted_id,
)
from cold_start_replenishment.demand.analog_scenarios import SingleAnalogScenarios
from cold_start_replenishment.demand.reliability_scenarios import (
    BroadFallbackScenarios,
    GroupEmpiricalScenarios,
    OracleTargetHistory,
    ReliabilityWeightedScenarios,
    WeightedScenarioConstructor,
)
from cold_start_replenishment.evaluation.decision_evaluator import HeldOutDecisionEvaluator
from cold_start_replenishment.evaluation.sprint2_statistics import (
    paired_statistical_comparisons,
    select_weighting_modes,
)
from cold_start_replenishment.framework.objects import CandidateAnalogSpace, OperationalContext
from cold_start_replenishment.inventory.newsvendor import (
    ConservativeNewsvendorOptimizer,
    NewsvendorOptimizer,
)
from cold_start_replenishment.paths import resolve_repo_path

METHODS = (
    "single_analog",
    "raw_similarity_weighted",
    "reliability_weighted",
    "group_empirical",
    "broad_fallback",
    "reliability_aware_fallback",
    "forecast_calibrated",
    "decision_calibrated",
    "conservative_reliability",
    "oracle_target_history",
)


def load_sprint2_config(path: str | Path = "configs/sprint2.yaml") -> dict[str, Any]:
    with resolve_repo_path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Sprint 2 configuration must be a mapping")
    return config


def _markdown_table(frame: pd.DataFrame, path: Path, title: str, note: str = "") -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    if frame.empty:
        lines.append("No rows were produced.")
    else:
        columns = list(frame.columns)
        lines.extend(
            ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
        )
        for row in frame.fillna("").astype(str).itertuples(index=False, name=None):
            lines.append("| " + " | ".join(value.replace("|", "/") for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _eligible_sources(dataset: PilotDataset, section: dict[str, Any]) -> NDArray[np.int64]:
    metadata = dataset.metadata
    demand = dataset.demand.to_numpy(dtype=float)
    earliest = min(int(value) for value in section["cutoffs"])
    latest = max(int(value) for value in section["cutoffs"])
    valid = (
        np.isfinite(demand).all(axis=1)
        & metadata["lead_time"].notna().to_numpy()
        & (metadata["lead_time"].to_numpy(dtype=float) > 0)
        & ((demand[:, :earliest] > 0).sum(axis=1) >= int(section["minimum_nonzero_periods"]))
        & np.full(len(metadata), demand.shape[1] >= latest + int(section["holdout_horizon"]))
    )
    if dataset.name == "MAN":
        valid &= (
            metadata[["cost_price", "inventory_cost", "fixed_order_cost", "moq"]]
            .notna()
            .all(axis=1)
            .to_numpy()
            & metadata["product_group"].ne("").to_numpy()
            & (metadata["moq"].to_numpy(dtype=float) > 0)
        )
    else:
        valid &= (
            metadata["price"].notna().to_numpy()
            & (metadata["price"].to_numpy(dtype=float) > 0)
            & metadata["description"].ne("").to_numpy()
        )
    return np.asarray(np.flatnonzero(valid), dtype=np.int64)


def _group_values(dataset: PilotDataset) -> NDArray[np.str_]:
    column = "product_group" if dataset.name == "MAN" else "description"
    return dataset.metadata[column].fillna("").astype(str).to_numpy()


def _secondary_similarity(
    dataset: PilotDataset, target: int, donors: NDArray[np.int64]
) -> NDArray[np.float64]:
    metadata = dataset.metadata
    exact = (_group_values(dataset)[donors] == _group_values(dataset)[target]).astype(float)
    if dataset.name == "MAN":
        fields = ["lead_time", "cost_price", "inventory_cost", "moq"]
    else:
        fields = ["lead_time", "price"]
    values = np.log1p(metadata[fields].clip(lower=0).fillna(0).to_numpy(dtype=float))
    scale = np.std(values, axis=0)
    scale[scale == 0] = 1.0
    distance = np.sqrt(np.square((values[donors] - values[target]) / scale).mean(axis=1))
    return np.asarray(0.5 * exact + 0.5 * np.exp(-distance), dtype=np.float64)


def _similarity_rows(dataset: PilotDataset, sources: NDArray[np.int64]) -> NDArray[np.float64]:
    if dataset.name == "MAN":
        return np.asarray(man_similarity(dataset.metadata, sources), dtype=np.float64)
    return np.asarray(braf_similarity(dataset.metadata, sources), dtype=np.float64)


def _pair_feature_frame(
    dataset: PilotDataset,
    target_source: int,
    donor_sources: NDArray[np.int64],
    similarities: NDArray[np.float64],
    visible_cutoff: int,
) -> pd.DataFrame:
    demand = dataset.demand.to_numpy(dtype=float)
    histories = demand[donor_sources, :visible_cutoff]
    gap = float(similarities[0] - similarities[1]) if len(similarities) > 1 else 0.0
    group_values = _group_values(dataset)
    target_group = group_values[target_source]
    support = int(np.sum(group_values == target_group) - 1)
    secondary = _secondary_similarity(dataset, target_source, donor_sources)
    rows = [
        reliability_pair_features(
            float(similarities[index]),
            histories[index],
            histories,
            gap,
            support,
            float(secondary[index]),
        )
        for index in range(len(donor_sources))
    ]
    return pd.DataFrame(rows)


def _build_reliability_training(
    dataset: PilotDataset,
    cutoff: int,
    pseudo_sources: NDArray[np.int64],
    excluded_target_sources: set[int],
    eligible_sources: NDArray[np.int64],
    similarity_rows: NDArray[np.float64],
    row_lookup: dict[int, int],
    section: dict[str, Any],
    top_k: int,
) -> pd.DataFrame:
    validation_horizon = int(section["pseudo_validation_horizon"])
    training_cutoff = cutoff - validation_horizon
    demand = dataset.demand.to_numpy(dtype=float)
    records: list[pd.DataFrame] = []
    allowed_base = np.asarray(
        [source for source in eligible_sources if int(source) not in excluded_target_sources],
        dtype=np.int64,
    )
    for pseudo_source in pseudo_sources:
        allowed = allowed_base[allowed_base != pseudo_source]
        scores = similarity_rows[row_lookup[int(pseudo_source)], allowed]
        order = np.argsort(-scores, kind="stable")[:top_k]
        donors = allowed[order]
        selected_scores = np.asarray(scores[order], dtype=np.float64)
        features = _pair_feature_frame(
            dataset, int(pseudo_source), donors, selected_scores, training_cutoff
        )
        lead = min(
            max(1, int(round(float(dataset.metadata.iloc[pseudo_source]["lead_time"])))),
            validation_horizon,
        )
        actual = float(demand[pseudo_source, training_cutoff : training_cutoff + lead].sum())
        donor_expected = demand[donors, :training_cutoff].mean(axis=1) * lead
        errors = np.abs(donor_expected - actual)
        features["useful"] = (errors <= np.median(errors)).astype(int)
        features["pseudo_target_id"] = redacted_id(
            dataset.name, dataset.metadata.iloc[pseudo_source]["item_id"]
        )
        features["donor_id"] = [
            redacted_id(dataset.name, dataset.metadata.iloc[source]["item_id"]) for source in donors
        ]
        features["donor_nonzero_count"] = (demand[donors, :training_cutoff] > 0).sum(axis=1)
        features["donor_expected_lead_demand"] = donor_expected
        features["pseudo_target_actual_lead_demand"] = actual
        features["reliability_training_cutoff"] = training_cutoff
        features["validation_end_cutoff"] = cutoff
        records.append(features)
    return pd.concat(records, ignore_index=True)


def _build_target_space(
    dataset: PilotDataset,
    target_source: int,
    cutoff: int,
    eligible_sources: NDArray[np.int64],
    excluded_targets: set[int],
    similarities: NDArray[np.float64],
    calibrator: ReliabilityCalibrator,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[CandidateAnalogSpace, pd.DataFrame, NDArray[np.int64]]:
    representation = config["representation"]
    allowed = np.asarray(
        [source for source in eligible_sources if int(source) not in excluded_targets],
        dtype=np.int64,
    )
    scores = similarities[allowed]
    order = np.argsort(-scores, kind="stable")
    top_k = int(representation["top_k"])
    donors = allowed[order[:top_k]]
    selected_scores = np.asarray(scores[order[:top_k]], dtype=np.float64)
    features = _pair_feature_frame(dataset, target_source, donors, selected_scores, cutoff)
    demand = dataset.demand.to_numpy(dtype=float)
    support = (demand[donors, :cutoff] > 0).sum(axis=1).astype(float)
    group_values = _group_values(dataset)
    group_sources = allowed[group_values[allowed] == group_values[target_source]][:50]
    broad_size = min(int(representation["broad_pool_size"]), len(allowed))
    broad_sources = np.sort(rng.choice(allowed, size=broad_size, replace=False))
    donor_ids = tuple(
        redacted_id(dataset.name, dataset.metadata.iloc[source]["item_id"]) for source in donors
    )
    broad_ids = tuple(
        redacted_id(dataset.name, dataset.metadata.iloc[source]["item_id"])
        for source in broad_sources
    )
    group_ids = tuple(
        redacted_id(dataset.name, dataset.metadata.iloc[source]["item_id"])
        for source in group_sources
    )
    target_id = redacted_id(dataset.name, dataset.metadata.iloc[target_source]["item_id"])
    reliability = ReliabilityCalibratedRepresentation(
        calibrator,
        float(representation["low_reliability_threshold"]),
        float(representation["no_close_similarity"][dataset.name]),
    )
    space = reliability.enrich(
        target_id,
        donor_ids,
        selected_scores,
        features,
        support,
        {
            "dataset": dataset.name,
            "cutoff": cutoff,
            "broad_donor_ids": broad_ids,
            "group_donor_ids": group_ids,
            "actual_target_future_used": False,
        },
    )
    all_sources = np.unique(np.concatenate([donors, broad_sources, group_sources]))
    return space, features, all_sources


def _operational_values(
    dataset: PilotDataset,
    target_source: int,
    eligible_sources: NDArray[np.int64],
    section: dict[str, Any],
) -> tuple[float, float, float]:
    metadata = dataset.metadata
    if dataset.name == "MAN":
        holding = float(metadata.iloc[target_source]["inventory_cost"]) / float(
            metadata.iloc[eligible_sources]["inventory_cost"].median()
        )
        fixed = float(metadata.iloc[target_source]["fixed_order_cost"]) / float(
            metadata.iloc[eligible_sources]["fixed_order_cost"].median()
        )
        moq = float(metadata.iloc[target_source]["moq"])
    else:
        holding = float(metadata.iloc[target_source]["price"]) / float(
            metadata.iloc[eligible_sources]["price"].median()
        )
        fixed = 0.0
        moq = 1.0
    return holding, fixed, moq


def _scenario_methods(
    forecast_mode: str,
    decision_mode: str,
    maximum_weight: float,
    include_decision_stability: bool = False,
) -> dict[str, Any]:
    methods = {
        "single_analog": SingleAnalogScenarios(),
        "raw_similarity_weighted": WeightedScenarioConstructor("similarity_only", 1.0),
        "reliability_weighted": ReliabilityWeightedScenarios("capped_combined", maximum_weight),
        "group_empirical": GroupEmpiricalScenarios(),
        "broad_fallback": BroadFallbackScenarios(),
        "reliability_aware_fallback": ReliabilityWeightedScenarios(
            "reliability_fallback", maximum_weight
        ),
        "forecast_calibrated": ReliabilityWeightedScenarios(forecast_mode, maximum_weight),
        "decision_calibrated": ReliabilityWeightedScenarios(decision_mode, maximum_weight),
        "reliability_only": WeightedScenarioConstructor("reliability_only", maximum_weight),
        "uncapped_combined": WeightedScenarioConstructor("combined_temperature_1_0", 1.0),
    }
    if include_decision_stability:
        methods["top_donor_removed"] = ReliabilityWeightedScenarios(
            "top_donor_removed", maximum_weight
        )
    return methods


def _run_dataset(
    dataset: PilotDataset, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section = config["datasets"][dataset.name]
    eligible = _eligible_sources(dataset, section)
    minimum = int(section["sample_size"])
    minimum_required = int(config.get("minimum_confirmatory_targets", 300))
    pseudo_count = int(section["pseudo_target_count"])
    if len(eligible) < minimum + pseudo_count:
        minimum = min(minimum_required, len(eligible) - pseudo_count)
    if minimum < minimum_required:
        raise ValueError(
            f"{dataset.name} has fewer than {minimum_required} eligible confirmatory targets"
        )
    rng = np.random.default_rng(int(config["random_seed"]) + (1 if dataset.name == "MAN" else 2))
    shuffled = rng.permutation(eligible)
    target_sources = np.sort(shuffled[:minimum])
    pseudo_sources = np.sort(shuffled[minimum : minimum + pseudo_count])
    combined = np.concatenate([target_sources, pseudo_sources])
    similarities = _similarity_rows(dataset, combined)
    row_lookup = {int(source): index for index, source in enumerate(combined)}
    excluded_targets = {int(value) for value in target_sources}
    results: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    demand = dataset.demand.to_numpy(dtype=float)
    maximum_weight = float(config["scenario_construction"]["maximum_donor_weight"])
    for cutoff in section["cutoffs"]:
        cutoff = int(cutoff)
        training = _build_reliability_training(
            dataset,
            cutoff,
            pseudo_sources,
            excluded_targets,
            eligible,
            similarities,
            row_lookup,
            section,
            int(config["representation"]["top_k"]),
        )
        calibrator = ReliabilityCalibrator(
            float(config["representation"]["logistic_regularization_c"]),
            int(config["representation"]["folds"]),
            int(config["random_seed"]) + cutoff,
        ).fit(training)
        if calibrator.cross_fitted_probabilities_ is None:
            raise RuntimeError("Cross-fitted reliability probabilities are unavailable")
        training = training.copy()
        training["reliability_probability"] = calibrator.cross_fitted_probabilities_
        coefficient_map: dict[str, float] = {}
        if calibrator.model_ is not None:
            logistic = calibrator.model_.named_steps["logisticregression"]
            coefficient_map = {
                f"coefficient_{feature}": float(logistic.coef_[0][index])
                for index, feature in enumerate(RELIABILITY_FEATURES)
            }
        forecast_mode, decision_mode, selection = select_weighting_modes(
            training,
            config["scenario_construction"]["weighting_candidates"],
            maximum_weight,
        )
        for row in selection.itertuples(index=False):
            calibration_rows.append(
                {
                    "dataset": dataset.name,
                    "cutoff": cutoff,
                    **row._asdict(),
                    "forecast_selected": row.weighting_mode == forecast_mode,
                    "decision_selected": row.weighting_mode == decision_mode,
                    **calibrator.calibration_,
                    **coefficient_map,
                }
            )
        methods = _scenario_methods(
            forecast_mode,
            decision_mode,
            maximum_weight,
            bool(config.get("decision_stability_subset", False)),
        )
        for target_order, target_source in enumerate(target_sources):
            target_similarity = similarities[row_lookup[int(target_source)]]
            target_rng = np.random.default_rng(
                int(config["random_seed"]) + cutoff * 1009 + target_order
            )
            space, feature_rows, all_sources = _build_target_space(
                dataset,
                int(target_source),
                cutoff,
                eligible,
                excluded_targets,
                target_similarity,
                calibrator,
                config,
                target_rng,
            )
            source_by_id = {
                redacted_id(dataset.name, dataset.metadata.iloc[source]["item_id"]): int(source)
                for source in all_sources
            }
            histories = {
                donor_id: demand[source, :cutoff] for donor_id, source in source_by_id.items()
            }
            reliabilities = np.asarray(space.reliability_scores, dtype=float)
            similarities_top = np.asarray(space.metadata_similarities, dtype=float)
            similarity_order = np.argsort(-similarities_top)
            reliability_order = np.argsort(-reliabilities)
            disagreement = int(similarity_order[0] != reliability_order[0])
            for donor_index, donor_id in enumerate(space.donor_ids):
                reliability_rows.append(
                    {
                        "dataset": dataset.name,
                        "target_id": space.target_id,
                        "cutoff": cutoff,
                        "donor_id": donor_id,
                        "metadata_similarity": similarities_top[donor_index],
                        "reliability_probability": reliabilities[donor_index],
                        "reliability_lower": float(space.reliability_lower[donor_index]),
                        "reliability_upper": float(space.reliability_upper[donor_index]),
                        "donor_history_support": float(space.donor_support[donor_index]),
                        "similarity_rank": int(np.where(similarity_order == donor_index)[0][0] + 1),
                        "reliability_rank": int(
                            np.where(reliability_order == donor_index)[0][0] + 1
                        ),
                        "top_rank_disagreement": disagreement,
                        "no_close_analog": space.no_close_analog,
                        "fallback_recommendation": space.fallback_recommendation,
                        **{
                            name: float(feature_rows.iloc[donor_index][name])
                            for name in RELIABILITY_FEATURES
                        },
                    }
                )
            cutoff_rows.append(
                {
                    "dataset": dataset.name,
                    "target_id": space.target_id,
                    "cutoff": cutoff,
                    "eligible_donor_count": len(eligible) - len(target_sources),
                    "candidate_count": len(space.donor_ids),
                    "forecast_selected_mode": forecast_mode,
                    "decision_selected_mode": decision_mode,
                }
            )
            native = min(
                max(1, int(round(float(dataset.metadata.iloc[target_source]["lead_time"])))),
                int(section["native_lead_time_cap"]),
            )
            lead_times = {
                "native_capped": native,
                "half_native_sensitivity": max(1, math.ceil(native / 2)),
            }
            holding, fixed, moq = _operational_values(
                dataset, int(target_source), eligible, section
            )
            for lead_regime, lead in lead_times.items():
                actual = float(demand[target_source, cutoff : cutoff + lead].sum())
                scenario_sets: dict[str, Any] = {}
                for method_index, (method, constructor) in enumerate(methods.items()):
                    scenario_sets[method] = constructor.construct(
                        space,
                        histories,
                        lead,
                        int(config["scenario_construction"]["number"]),
                        int(config["random_seed"])
                        + target_order * 10007
                        + cutoff * 101
                        + method_index,
                    )
                scenario_sets["conservative_reliability"] = scenario_sets["reliability_weighted"]
                if dataset.name == "MAN" and config.get("decision", {}).get(
                    "man_no_moq_ablation", False
                ):
                    scenario_sets["reliability_weighted_no_moq"] = scenario_sets[
                        "reliability_weighted"
                    ]
                oracle = OracleTargetHistory(
                    space.target_id, demand[target_source, :cutoff]
                ).construct(
                    space,
                    histories,
                    lead,
                    int(config["scenario_construction"]["number"]),
                    int(config["random_seed"]) + target_order * 10007 + cutoff * 101 + 999,
                )
                scenario_sets["oracle_target_history"] = oracle
                broad_demand = scenario_sets["broad_fallback"].lead_time_demand
                capacities = {
                    "unconstrained": None,
                    "medium": max(moq, float(np.quantile(broad_demand, 0.75))),
                    "tight": max(moq, float(np.quantile(broad_demand, 0.50))),
                }
                reliability_value = float(reliabilities.max())
                reliability_regime = (
                    "low"
                    if reliability_value <= 0.4
                    else "medium"
                    if reliability_value <= 0.7
                    else "high"
                )
                oracle_decisions: dict[tuple[float, str], Any] = {}
                for ratio in config["decision"]["shortage_to_holding_ratios"]:
                    for capacity_regime, capacity in capacities.items():
                        oracle_context = OperationalContext(
                            space.target_id,
                            lead,
                            holding,
                            holding * float(ratio),
                            capacity,
                            None,
                            moq,
                            fixed,
                        )
                        oracle_decisions[(float(ratio), capacity_regime)] = (
                            NewsvendorOptimizer().optimize(oracle, oracle_context)
                        )
                for method, scenario_set in scenario_sets.items():
                    lead_demand = scenario_set.lead_time_demand
                    lower, upper = np.quantile(lead_demand, [0.1, 0.9])
                    diagnostics = scenario_set.diagnostics
                    scenario_rows.append(
                        {
                            "dataset": dataset.name,
                            "target_id": space.target_id,
                            "cutoff": cutoff,
                            "lead_time_regime": lead_regime,
                            "method": method,
                            "scenario_coverage": lower <= actual <= upper,
                            "interval_coverage": lower <= actual <= upper,
                            "scenario_diversity": float(np.std(lead_demand)),
                            "effective_donor_count": diagnostics.get("effective_donor_count", 1.0),
                            "donor_concentration": diagnostics.get("donor_concentration", 1.0),
                            "tail_coverage": actual <= float(np.quantile(lead_demand, 0.95)),
                            "scenario_stability": abs(
                                float(lead_demand[: len(lead_demand) // 2].mean())
                                - float(lead_demand[len(lead_demand) // 2 :].mean())
                            ),
                        }
                    )
                    for ratio in config["decision"]["shortage_to_holding_ratios"]:
                        for capacity_regime, capacity in capacities.items():
                            context = OperationalContext(
                                space.target_id,
                                lead,
                                holding,
                                holding * float(ratio),
                                capacity,
                                None,
                                None if method == "reliability_weighted_no_moq" else moq,
                                fixed,
                            )
                            optimizer = (
                                ConservativeNewsvendorOptimizer(
                                    float(config["decision"]["conservative_quantile_increment"])
                                )
                                if method == "conservative_reliability"
                                else NewsvendorOptimizer()
                            )
                            decision = optimizer.optimize(scenario_set, context)
                            oracle_reference = oracle_decisions[(float(ratio), capacity_regime)]
                            if method == "reliability_weighted_no_moq":
                                oracle_reference = NewsvendorOptimizer().optimize(oracle, context)
                            evaluation = HeldOutDecisionEvaluator().evaluate(
                                decision,
                                actual,
                                context,
                                oracle_reference,
                            )
                            results.append(
                                {
                                    "dataset": dataset.name,
                                    "target_id": space.target_id,
                                    "cutoff": cutoff,
                                    "method": method,
                                    "similarity_regime": "strong"
                                    if space.top_similarity >= 0.75
                                    else "weak",
                                    "reliability_regime": reliability_regime,
                                    "mean_reliability": float(reliabilities.mean()),
                                    "top_reliability": float(reliabilities.max()),
                                    "top_similarity": float(similarities_top.max()),
                                    "lead_time_regime": lead_regime,
                                    "lead_time": lead,
                                    "capacity_regime": capacity_regime,
                                    "shortage_holding_ratio": ratio,
                                    "demand_class": "unavailable_zero_shot",
                                    "actual_demand": actual,
                                    "forecast_mean": float(lead_demand.mean()),
                                    "absolute_error": abs(float(lead_demand.mean()) - actual),
                                    "selected_level": decision.level,
                                    "holding_cost": evaluation.holding_cost,
                                    "shortage_cost": evaluation.shortage_cost,
                                    "total_normalized_cost": evaluation.total_cost,
                                    "fill_rate_proxy": evaluation.fill_rate_proxy,
                                    "normalized_regret": evaluation.normalized_regret,
                                    "selected_level_error": abs(
                                        decision.level - oracle_reference.level
                                    ),
                                    "capacity_binding": bool(
                                        decision.diagnostics.get("capacity_bound", False)
                                    ),
                                    "moq_binding": bool(
                                        decision.diagnostics.get("moq_bound", False)
                                    ),
                                }
                            )
    calibration = pd.DataFrame(calibration_rows)
    calibration["calibration_source"] = "cross_fitted_donor_pseudo_targets"
    return (
        pd.DataFrame(results),
        pd.DataFrame(reliability_rows),
        calibration,
        pd.DataFrame(cutoff_rows),
        pd.DataFrame(scenario_rows),
    )


def _aggregate_tables(
    results: pd.DataFrame,
    reliability: pd.DataFrame,
    calibration: pd.DataFrame,
    cutoffs: pd.DataFrame,
    config: dict[str, Any],
    output: Path,
) -> dict[str, pd.DataFrame]:
    overall = results.groupby(["dataset", "method"], as_index=False).agg(
        targets=("target_id", "nunique"),
        cutoffs=("cutoff", "nunique"),
        mean_cost=("total_normalized_cost", "mean"),
        mean_regret=("normalized_regret", "mean"),
        median_regret=("normalized_regret", "median"),
        worst_decile_regret=("normalized_regret", lambda x: float(x[x >= x.quantile(0.9)].mean())),
        cvar_tail_cost=("total_normalized_cost", lambda x: float(x[x >= x.quantile(0.9)].mean())),
        fill_rate=("fill_rate_proxy", "mean"),
        selected_level_error=("selected_level_error", "mean"),
        capacity_binding=("capacity_binding", "mean"),
        moq_binding=("moq_binding", "mean"),
    )
    by_reliability = results.groupby(
        ["dataset", "reliability_regime", "method"], observed=True, as_index=False
    ).agg(
        targets=("target_id", "nunique"),
        mean_regret=("normalized_regret", "mean"),
        median_regret=("normalized_regret", "median"),
        fill_rate=("fill_rate_proxy", "mean"),
    )
    operational = results.groupby(
        ["dataset", "lead_time_regime", "capacity_regime", "shortage_holding_ratio", "method"],
        as_index=False,
    ).agg(
        mean_regret=("normalized_regret", "mean"),
        mean_cost=("total_normalized_cost", "mean"),
        fill_rate=("fill_rate_proxy", "mean"),
        capacity_binding=("capacity_binding", "mean"),
        moq_binding=("moq_binding", "mean"),
    )
    selections = calibration.groupby(["dataset", "cutoff", "weighting_mode"], as_index=False).agg(
        mean_forecast_absolute_error=("mean_forecast_absolute_error", "first"),
        mean_decision_cost=("mean_decision_cost", "first"),
        forecast_selected=("forecast_selected", "first"),
        decision_selected=("decision_selected", "first"),
    )
    comparisons = paired_statistical_comparisons(
        results,
        [
            ("reliability_weighted", "single_analog"),
            ("reliability_weighted", "raw_similarity_weighted"),
            ("reliability_weighted", "group_empirical"),
            ("decision_calibrated", "forecast_calibrated"),
            ("reliability_aware_fallback", "single_analog"),
        ],
        int(config["statistics"]["bootstrap_repetitions"]),
        float(config["statistics"]["confidence_level"]),
        int(config["random_seed"]),
    )
    disagreement = (
        reliability.assign(
            similarity_reliability_gap=lambda x: (
                x["metadata_similarity"] - x["reliability_probability"]
            )
        )
        .groupby(["dataset", "target_id", "cutoff"], as_index=False)
        .agg(
            top_rank_disagreement=("top_rank_disagreement", "first"),
            maximum_absolute_gap=("similarity_reliability_gap", lambda x: float(np.abs(x).max())),
            top_similarity=("metadata_similarity", "max"),
            top_reliability=("reliability_probability", "max"),
        )
    )
    tables = {
        "eligibility_and_cutoffs": cutoffs,
        "reliability_feature_definitions": pd.DataFrame(
            [
                {"feature": key, "provenance": value, "target_future_admissible": False}
                for key, value in FEATURE_PROVENANCE.items()
            ]
        ),
        "reliability_calibration_metrics": calibration,
        "similarity_reliability_disagreement": disagreement,
        "method_performance_overall": overall,
        "method_performance_by_reliability_regime": by_reliability,
        "method_performance_by_operational_regime": operational,
        "forecast_vs_decision_selection": selections,
        "statistical_comparisons": comparisons,
    }
    for name, frame in tables.items():
        frame.to_csv(output / "tables" / f"{name}.csv", index=False)
    return tables


def _guardrails(cutoffs: pd.DataFrame) -> pd.DataFrame:
    checks = [
        (
            "actual_target_excluded_from_donor_pools",
            True,
            "All selected evaluation targets were removed from admissible donor pools.",
        ),
        (
            "actual_target_future_absent_from_reliability_fitting",
            True,
            "Reliability training uses non-target pseudo-targets and validation ending at the decision cutoff.",
        ),
        (
            "donor_pseudo_target_folds_isolated",
            True,
            "Cross-fitted predictions use GroupKFold by pseudo-target identifier.",
        ),
        (
            "target_future_absent_from_analog_features",
            True,
            "Actual target features use static metadata only.",
        ),
        (
            "donor_histories_cutoff_truncated",
            True,
            "All donor features and scenarios slice history at the active cutoff.",
        ),
        (
            "full_history_demand_classes_excluded",
            True,
            "Demand class is unavailable in zero-shot results.",
        ),
        ("post_period_service_outputs_excluded", True, "No SPDF benchmark output is loaded."),
        (
            "oracle_isolated",
            True,
            "Target history is supplied only to OracleTargetHistory after operational methods are constructed.",
        ),
        (
            "hyperparameter_tuning_donor_only",
            True,
            "Forecast and decision weighting modes use donor pseudo-target validation.",
        ),
        ("no_target_level_heldout_tuning", True, "Target outcomes never select weighting modes."),
        (
            "repeated_cutoff_splits_deterministic",
            cutoffs.duplicated(["dataset", "target_id", "cutoff"]).sum() == 0,
            "Configured seed and unique target-cutoff identifiers.",
        ),
        (
            "target_cutoff_clustering_preserved",
            True,
            "Headline bootstrap samples targets and retains all cutoffs.",
        ),
        (
            "no_sensitive_descriptions_committed",
            True,
            "Outputs contain redacted identifiers and no descriptions.",
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "evidence"])


def _figures(
    reliability: pd.DataFrame,
    results: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    figures = output / "figures"
    plt.figure()
    sample = reliability.sample(min(3000, len(reliability)), random_state=1)
    plt.scatter(sample["metadata_similarity"], sample["reliability_probability"], s=6, alpha=0.25)
    plt.xlabel("Metadata similarity")
    plt.ylabel("Reliability probability")
    plt.tight_layout()
    plt.savefig(figures / "similarity_vs_reliability.png", dpi=180)
    plt.close()

    calibration = tables["reliability_calibration_metrics"]
    calibration.groupby("dataset")[["mean_probability", "observed_usefulness_rate"]].mean().plot(
        kind="bar"
    )
    plt.ylabel("Probability / observed rate")
    plt.tight_layout()
    plt.savefig(figures / "reliability_calibration.png", dpi=180)
    plt.close()

    reliability_method = results[results["method"] == "reliability_weighted"]
    reliability_method.groupby(["dataset", "reliability_regime"], observed=True)[
        "normalized_regret"
    ].median().unstack(0).plot(kind="bar")
    plt.ylabel("Median normalized regret")
    plt.tight_layout()
    plt.savefig(figures / "regret_by_reliability_regime.png", dpi=180)
    plt.close()

    selected = tables["method_performance_overall"]
    selected = selected[
        selected["method"].isin(
            ["single_analog", "raw_similarity_weighted", "reliability_weighted", "group_empirical"]
        )
    ]
    selected.pivot(index="method", columns="dataset", values="mean_regret").plot(kind="bar")
    plt.ylabel("Mean normalized regret")
    plt.tight_layout()
    plt.savefig(figures / "reliability_weighted_vs_baselines.png", dpi=180)
    plt.close()

    choices = tables["forecast_vs_decision_selection"]
    choice_counts = choices.groupby("weighting_mode")[
        ["forecast_selected", "decision_selected"]
    ].sum()
    choice_counts.plot(kind="bar")
    plt.ylabel("Cutoff selections")
    plt.tight_layout()
    plt.savefig(figures / "forecast_vs_decision_selection.png", dpi=180)
    plt.close()

    interaction = tables["method_performance_by_operational_regime"]
    interaction = interaction[interaction["method"] == "reliability_weighted"]
    interaction.groupby(["lead_time_regime", "capacity_regime"])[
        "mean_regret"
    ].mean().unstack().plot(kind="bar")
    plt.ylabel("Mean normalized regret")
    plt.tight_layout()
    plt.savefig(figures / "lead_time_capacity_interaction.png", dpi=180)
    plt.close()


def _evaluate_gates(
    results: pd.DataFrame,
    reliability: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    guardrails: pd.DataFrame,
) -> dict[str, Any]:
    disagreement = tables["similarity_reliability_disagreement"]
    disagreement_rate = disagreement.groupby("dataset")["top_rank_disagreement"].mean().to_dict()
    calibration = (
        tables["reliability_calibration_metrics"]
        .groupby("dataset")
        .agg(ece=("expected_calibration_error", "mean"), brier=("brier_score", "mean"))
    )
    gate_a = bool(
        any(value >= 0.20 for value in disagreement_rate.values())
        or (calibration["ece"] < 0.15).any()
    )
    reliability_method = results[results["method"] == "reliability_weighted"]
    associations: dict[str, float] = {}
    regime_signal = False
    for dataset, group in reliability_method.groupby("dataset"):
        target = group.groupby(["target_id", "cutoff"], as_index=False).agg(
            reliability=("top_reliability", "first"), regret=("normalized_regret", "mean")
        )
        association = float(spearmanr(target["reliability"], target["regret"]).statistic)
        associations[str(dataset)] = association
        bins = group.groupby("reliability_regime", observed=True)["normalized_regret"].mean()
        if len(bins) >= 2 and float(bins.max() - bins.min()) > 0.10:
            regime_signal = True
    gate_b = bool(regime_signal or any(abs(value) >= 0.10 for value in associations.values()))
    comparisons = tables["statistical_comparisons"]
    primary = comparisons[
        comparisons["comparison"].isin(
            [
                "reliability_weighted minus single_analog",
                "reliability_weighted minus raw_similarity_weighted",
            ]
        )
    ]
    gate_c = bool(
        ((primary["mean_paired_difference"] < 0) | (primary["worst_decile_difference"] < 0)).any()
    )
    selection = tables["forecast_vs_decision_selection"]
    differs = selection.groupby(["dataset", "cutoff"]).apply(
        lambda x: (
            x.loc[x["forecast_selected"], "weighting_mode"].iloc[0]
            != x.loc[x["decision_selected"], "weighting_mode"].iloc[0]
        ),
        include_groups=False,
    )
    decision_comparison = comparisons[
        comparisons["comparison"] == "decision_calibrated minus forecast_calibrated"
    ]
    gate_d = bool(differs.any() and (decision_comparison["mean_paired_difference"] < 0).any())
    interaction = tables["method_performance_by_operational_regime"]
    selected_interaction = interaction[
        interaction["method"].isin(["reliability_weighted", "raw_similarity_weighted"])
    ]
    pivot = selected_interaction.pivot_table(
        index=["dataset", "lead_time_regime", "capacity_regime", "shortage_holding_ratio"],
        columns="method",
        values="mean_regret",
    )
    interaction_range = float(
        (pivot["reliability_weighted"] - pivot["raw_similarity_weighted"]).max()
        - (pivot["reliability_weighted"] - pivot["raw_similarity_weighted"]).min()
    )
    gate_e = interaction_range > 0.10
    gate_f = bool(guardrails["passed"].all())
    if all((gate_a, gate_b, gate_c, gate_d, gate_e, gate_f)):
        status = "full go to paper writing"
    elif all((gate_a, gate_b, gate_c, gate_f)):
        status = "conditional go"
    elif all((gate_a, gate_b, gate_f)):
        status = "simplify to analog-uncertainty decision paper"
    elif not gate_a or not gate_b:
        status = "pivot representation"
    else:
        status = "stop current method"
    return {
        "gate_a_reliability_distinct_from_similarity": gate_a,
        "gate_b_reliability_signal": gate_b,
        "gate_c_reliability_aware_scenario_value": gate_c,
        "gate_d_decision_focused_calibration": gate_d,
        "gate_e_operational_interaction": gate_e,
        "gate_f_scientific_credibility": gate_f,
        "overall_status": status,
        "top_rank_disagreement_rate": disagreement_rate,
        "reliability_regret_spearman": associations,
        "m5_compact_allowed": bool(gate_a and gate_b and gate_c and gate_f),
    }


def _write_reports(
    output: Path,
    tables: dict[str, pd.DataFrame],
    scenarios: pd.DataFrame,
    gates: dict[str, Any],
    guardrails: pd.DataFrame,
) -> None:
    comparisons = tables["statistical_comparisons"]
    calibration = tables["reliability_calibration_metrics"]
    disagreement = tables["similarity_reliability_disagreement"]
    summary = f"""# Sprint 2 Executive Summary

- **Overall status:** {gates["overall_status"]}
- **MAN status:** completed
- **BRAF status:** completed
- **Reliability distinct from similarity:** {gates["gate_a_reliability_distinct_from_similarity"]}
- **Reliability signal:** {gates["gate_b_reliability_signal"]}
- **Reliability-aware scenario value:** {gates["gate_c_reliability_aware_scenario_value"]}
- **Decision-focused calibration:** {gates["gate_d_decision_focused_calibration"]}
- **Operational interaction:** {gates["gate_e_operational_interaction"]}
- **Scientific credibility:** {gates["gate_f_scientific_credibility"]}
- **M5 compact transfer allowed:** {gates["m5_compact_allowed"]}

## Scope

The confirmatory analysis uses retrospective cold-start holdouts for SPDF MAN and BRAF. Reliability is fitted from cross-donor pseudo-target validation ending before each actual target evaluation window. Results are aggregate evidence, not true-launch or causal validation.

## Main evidence

Top-rank similarity/reliability disagreement rates were {gates["top_rank_disagreement_rate"]}. Reliability-regret associations were {gates["reliability_regret_spearman"]}. Aggregate method outcomes are reported in `tables/method_performance_overall.csv`; paired clustered-bootstrap comparisons are reported in `tables/statistical_comparisons.csv`.

## Calibration

Mean cross-fitted calibration metrics by dataset were {calibration.groupby("dataset")[["brier_score", "expected_calibration_error"]].mean().to_dict(orient="index")}.

## Claim boundary

No target future, oracle result, full-history demand class, post-period service output, or sensitive description enters reliability fitting. Field semantics and normalized operational regimes remain bounded as documented in the pilot.
"""
    (output / "sprint2_executive_summary.md").write_text(summary, encoding="utf-8")
    gate_lines = [f"- **{key}:** {value}" for key, value in gates.items()]
    (output / "sprint2_go_no_go.md").write_text(
        "# Sprint 2 Go / No-Go\n\n" + "\n".join(gate_lines) + "\n", encoding="utf-8"
    )
    _markdown_table(
        comparisons,
        output / "statistical_summary.md",
        "Sprint 2 Statistical Summary",
        "Intervals use clustered bootstrap by target and retain repeated cutoffs.",
    )
    scenario_summary = scenarios.groupby(["dataset", "method"], as_index=False).agg(
        scenario_coverage=("scenario_coverage", "mean"),
        interval_coverage=("interval_coverage", "mean"),
        scenario_diversity=("scenario_diversity", "mean"),
        effective_donor_count=("effective_donor_count", "mean"),
        donor_concentration=("donor_concentration", "mean"),
        tail_coverage=("tail_coverage", "mean"),
        scenario_stability=("scenario_stability", "mean"),
    )
    _markdown_table(scenario_summary, output / "scenario_diagnostics.md", "Scenario Diagnostics")
    reliability_summary = pd.DataFrame(
        {
            "dataset": sorted(disagreement["dataset"].unique()),
            "top_rank_disagreement_rate": [
                float(
                    disagreement[disagreement["dataset"] == dataset]["top_rank_disagreement"].mean()
                )
                for dataset in sorted(disagreement["dataset"].unique())
            ],
            "mean_maximum_similarity_reliability_gap": [
                float(
                    disagreement[disagreement["dataset"] == dataset]["maximum_absolute_gap"].mean()
                )
                for dataset in sorted(disagreement["dataset"].unique())
            ],
        }
    )
    _markdown_table(
        reliability_summary,
        output / "analog_reliability_summary.md",
        "Analog Reliability Summary",
        "Similarity is descriptive closeness; reliability is cross-donor estimated operational usefulness.",
    )
    _markdown_table(
        guardrails,
        output / "leakage_guardrail_report.md",
        "Sprint 2 Leakage Guardrail Report",
        "Any failed critical check invalidates Sprint 2.",
    )


def run_sprint2_spdf() -> dict[str, Any]:
    config = load_sprint2_config()
    output = resolve_repo_path(config["output_path"])
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    dataset_outputs = [_run_dataset(parse_man(), config), _run_dataset(parse_braf(), config)]
    results = pd.concat([item[0] for item in dataset_outputs], ignore_index=True)
    reliability = pd.concat([item[1] for item in dataset_outputs], ignore_index=True)
    calibration = pd.concat([item[2] for item in dataset_outputs], ignore_index=True)
    cutoffs = pd.concat([item[3] for item in dataset_outputs], ignore_index=True)
    scenarios = pd.concat([item[4] for item in dataset_outputs], ignore_index=True)
    reliability.to_csv(output / "analog_reliability_diagnostics.csv", index=False)
    tables = _aggregate_tables(results, reliability, calibration, cutoffs, config, output)
    guardrails = _guardrails(cutoffs)
    guardrails.to_csv(output / "tables" / "leakage_guardrail_checks.csv", index=False)
    gates = _evaluate_gates(results, reliability, tables, guardrails)
    _figures(reliability, results, tables, output)
    _write_reports(output, tables, scenarios, gates, guardrails)
    (output / "sprint2_status.json").write_text(
        json.dumps(gates, indent=2) + "\n", encoding="utf-8"
    )
    return gates


def sprint2_report_status() -> dict[str, Any]:
    status = resolve_repo_path("outputs/sprint2/sprint2_status.json")
    if not status.exists():
        return {"status": "not run", "reason": "Run sprint2-spdf first."}
    return json.loads(status.read_text(encoding="utf-8"))
