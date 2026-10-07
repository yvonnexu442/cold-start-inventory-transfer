"""Targeted acceptance-risk analyses using the frozen full-scale population."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.stats import spearmanr  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    brier_score_loss,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from cold_start_replenishment.analogs.reliability import (
    RELIABILITY_FEATURES,
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
from cold_start_replenishment.framework.interfaces import DecisionOptimizationModule
from cold_start_replenishment.framework.objects import (
    DecisionScenarioSet,
    OperationalContext,
)
from cold_start_replenishment.inventory.newsvendor import NewsvendorOptimizer
from cold_start_replenishment.inventory.risk_aware import RiskAwareNewsvendorOptimizer
from cold_start_replenishment.paths import resolve_repo_path

CONTROL_METHODS = (
    "uniform_multi",
    "similarity_support",
    "reliability_only",
    "combined_primary",
    "shuffled_reliability",
    "hgb_combined",
    "single_analog",
)

COMPARISONS = (
    ("reliability_only", "uniform_multi"),
    ("reliability_only", "shuffled_reliability"),
    ("combined_primary", "shuffled_reliability"),
    ("similarity_support", "uniform_multi"),
    ("combined_primary", "single_analog"),
    ("hgb_combined", "single_analog"),
    ("hgb_combined", "similarity_support"),
)


class NonlinearReliabilityCalibrator:
    """Group-cross-fitted HGB learner with nested sigmoid probability calibration."""

    def __init__(self, folds: int, random_seed: int) -> None:
        self.folds = folds
        self.random_seed = random_seed
        self.model_: HistGradientBoostingClassifier | None = None
        self.sigmoid_: LogisticRegression | None = None
        self.cross_fitted_probabilities_: NDArray[np.float64] | None = None
        self.calibration_: dict[str, float] = {}

    def _model(self, seed: int) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=seed,
        )

    @staticmethod
    def _fit_sigmoid(raw: NDArray[np.float64], labels: NDArray[np.int64]) -> LogisticRegression:
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(raw[:, None], labels)
        return model

    def fit(self, training: pd.DataFrame) -> NonlinearReliabilityCalibrator:
        x = training.loc[:, RELIABILITY_FEATURES].to_numpy(dtype=float)
        y = training["useful"].to_numpy(dtype=np.int64)
        groups = training["pseudo_target_id"].astype(str).to_numpy()
        outer = GroupKFold(min(self.folds, len(np.unique(groups))))
        calibrated = np.zeros(len(training), dtype=float)
        for fold, (train_index, test_index) in enumerate(outer.split(x, y, groups)):
            inner_groups = groups[train_index]
            inner = GroupKFold(min(self.folds - 1, len(np.unique(inner_groups))))
            inner_raw = np.zeros(len(train_index), dtype=float)
            for inner_fold, (fit_pos, cal_pos) in enumerate(
                inner.split(x[train_index], y[train_index], inner_groups)
            ):
                learner = self._model(self.random_seed + fold * 31 + inner_fold)
                learner.fit(x[train_index][fit_pos], y[train_index][fit_pos])
                inner_raw[cal_pos] = learner.predict_proba(x[train_index][cal_pos])[:, 1]
            sigmoid = self._fit_sigmoid(inner_raw, y[train_index])
            learner = self._model(self.random_seed + fold)
            learner.fit(x[train_index], y[train_index])
            raw_test = learner.predict_proba(x[test_index])[:, 1]
            calibrated[test_index] = sigmoid.predict_proba(raw_test[:, None])[:, 1]

        full_raw = np.zeros(len(training), dtype=float)
        for fold, (train_index, test_index) in enumerate(outer.split(x, y, groups)):
            learner = self._model(self.random_seed + 100 + fold)
            learner.fit(x[train_index], y[train_index])
            full_raw[test_index] = learner.predict_proba(x[test_index])[:, 1]
        self.sigmoid_ = self._fit_sigmoid(full_raw, y)
        self.model_ = self._model(self.random_seed + 999)
        self.model_.fit(x, y)
        self.cross_fitted_probabilities_ = np.asarray(calibrated, dtype=np.float64)
        self.calibration_ = {
            "brier_score": float(brier_score_loss(y, calibrated)),
            "expected_calibration_error": expected_calibration_error(calibrated, y),
            "auc": float(roc_auc_score(y, calibrated)),
        }
        return self

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        if self.model_ is None or self.sigmoid_ is None:
            raise RuntimeError("Fit NonlinearReliabilityCalibrator before prediction")
        x = features.loc[:, RELIABILITY_FEATURES].to_numpy(dtype=float)
        raw = self.model_.predict_proba(x)[:, 1]
        return np.asarray(self.sigmoid_.predict_proba(raw[:, None])[:, 1], dtype=np.float64)


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "No estimable rows."
    rendered = frame.copy()
    for column in rendered.select_dtypes(include=["number"]).columns:
        rendered[column] = rendered[column].map(lambda value: f"{value:.{digits}f}")
    columns = list(rendered.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rendered.fillna("").astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_summary(path: Path, title: str, label: str, body: str) -> None:
    path.write_text(
        f"# {title}\n\n**Analysis label:** {label}\n\n{body.rstrip()}\n", encoding="utf-8"
    )


def _construct_scenarios(
    donor_ids: list[str],
    histories: list[NDArray[np.float64]],
    weights: NDArray[np.float64],
    horizon: int,
    count: int,
    seed: int,
    method: str,
) -> DecisionScenarioSet:
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(donor_ids), size=count, p=weights)
    scenarios = np.zeros((count, horizon), dtype=float)
    sources = [donor_ids[int(position)] for position in selected]
    for position in np.unique(selected):
        rows = np.flatnonzero(selected == position)
        history = histories[int(position)]
        probability = float((history > 0).mean())
        occurrences = rng.random((len(rows), horizon)) < probability
        positive = history[history > 0]
        if len(positive):
            sizes = rng.choice(positive, size=(len(rows), horizon), replace=True)
            scenarios[rows] = occurrences * sizes
    entropy = float(-(weights * np.log(np.clip(weights, 1e-12, None))).sum())
    return DecisionScenarioSet(
        scenarios,
        horizon,
        method,
        {
            "effective_donor_count": float(1 / np.square(weights).sum()),
            "donor_entropy": entropy,
            "maximum_donor_share": float(weights.max()),
        },
        tuple(sources),
        np.full(count, 1 / count),
        "frozen_cutoff_visible_donor_histories",
    )


def _weights(
    similarity: NDArray[np.float64],
    reliability: NDArray[np.float64],
    support: NDArray[np.float64],
    mode: str,
    maximum: float,
) -> NDArray[np.float64]:
    support_weight = np.log1p(support)
    if mode == "uniform_multi":
        raw = np.ones(len(similarity))
    elif mode == "similarity_support":
        raw = similarity * support_weight
    elif mode == "reliability_only":
        raw = reliability * support_weight
    elif mode in {"combined_primary", "shuffled_reliability", "hgb_combined"}:
        raw = similarity * reliability * support_weight
    elif mode == "single_analog":
        raw = np.zeros(len(similarity))
        raw[int(np.argmax(similarity))] = 1.0
        return raw
    else:
        raise ValueError(f"Unknown control mode: {mode}")
    return normalized_capped_weights(np.asarray(raw, dtype=np.float64), maximum)


def _prepare_dataset(
    dataset: PilotDataset, config: dict[str, Any], reliability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, NDArray[np.int64], NDArray[np.int64]]:
    section = config["datasets"][dataset.name]
    eligible = _eligible_sources(dataset, section)
    rng = np.random.default_rng(config["random_seed"] + (1 if dataset.name == "MAN" else 2))
    shuffled = rng.permutation(eligible)
    target_sources = np.sort(shuffled[: int(section["sample_size"])])
    pseudo_sources = np.sort(
        shuffled[
            int(section["sample_size"]) : int(section["sample_size"])
            + int(section["pseudo_target_count"])
        ]
    )
    combined = np.concatenate([target_sources, pseudo_sources])
    similarities = _similarity_rows(dataset, combined)
    row_lookup = {int(source): index for index, source in enumerate(combined)}
    excluded_targets = {int(value) for value in target_sources}
    learners: dict[int, NonlinearReliabilityCalibrator] = {}
    diagnostics: list[dict[str, Any]] = []
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
        logistic = ReliabilityCalibrator(
            float(config["representation"]["logistic_regularization_c"]),
            int(config["representation"]["folds"]),
            int(config["random_seed"]) + cutoff,
        ).fit(training)
        nonlinear = NonlinearReliabilityCalibrator(
            int(config["representation"]["folds"]), int(config["random_seed"]) + cutoff
        ).fit(training)
        y = training["useful"].to_numpy(dtype=int)
        logistic_prob = np.asarray(logistic.cross_fitted_probabilities_, dtype=float)
        diagnostics.extend(
            [
                {
                    "dataset": dataset.name,
                    "cutoff": cutoff,
                    "learner": "logistic",
                    "brier_score": brier_score_loss(y, logistic_prob),
                    "ece": expected_calibration_error(logistic_prob, y),
                    "auc": roc_auc_score(y, logistic_prob),
                },
                {
                    "dataset": dataset.name,
                    "cutoff": cutoff,
                    "learner": "calibrated_hgb",
                    "brier_score": nonlinear.calibration_["brier_score"],
                    "ece": nonlinear.calibration_["expected_calibration_error"],
                    "auc": nonlinear.calibration_["auc"],
                },
            ]
        )
        learners[cutoff] = nonlinear

    scored: list[pd.DataFrame] = []
    for cutoff, group in reliability.groupby("cutoff"):
        cutoff_value = int(str(cutoff))
        part = group.copy()
        part["hgb_reliability"] = learners[cutoff_value].predict(part)
        scored.append(part)
        for _, pool in part.groupby(["target_id", "cutoff"]):
            diagnostics.append(
                {
                    "dataset": dataset.name,
                    "cutoff": cutoff_value,
                    "learner": "target_pool_comparison",
                    "brier_score": np.nan,
                    "ece": np.nan,
                    "auc": np.nan,
                    "top_donor_stability": float(
                        pool.loc[pool["reliability_probability"].idxmax(), "donor_id"]
                        == pool.loc[pool["hgb_reliability"].idxmax(), "donor_id"]
                    ),
                    "hgb_similarity_disagreement": float(
                        pool.loc[pool["metadata_similarity"].idxmax(), "donor_id"]
                        != pool.loc[pool["hgb_reliability"].idxmax(), "donor_id"]
                    ),
                    "hgb_similarity_rank_correlation": float(
                        spearmanr(pool["metadata_similarity"], pool["hgb_reliability"]).statistic
                    ),
                    "logistic_hgb_rank_correlation": float(
                        spearmanr(
                            pool["reliability_probability"], pool["hgb_reliability"]
                        ).statistic
                    ),
                }
            )
    learner_diagnostics = pd.DataFrame(diagnostics)
    learner_diagnostics = learner_diagnostics[
        learner_diagnostics["learner"].isin(["logistic", "calibrated_hgb"])
    ].reset_index(drop=True)
    return learner_diagnostics, pd.concat(scored, ignore_index=True), target_sources, eligible


def _evaluate_dataset(
    dataset: PilotDataset,
    config: dict[str, Any],
    reliability: pd.DataFrame,
    target_sources: NDArray[np.int64],
    eligible: NDArray[np.int64],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    section = config["datasets"][dataset.name]
    demand = dataset.demand.to_numpy(dtype=float)
    id_to_source = {
        redacted_id(dataset.name, row.item_id): int(str(index))
        for index, row in dataset.metadata.iterrows()
    }
    excluded = set(int(value) for value in target_sources)
    allowed = np.asarray([source for source in eligible if int(source) not in excluded], dtype=int)
    target_order = {int(source): index for index, source in enumerate(target_sources)}
    count = int(config["scenario_construction"]["number"])
    maximum = float(config["scenario_construction"]["maximum_donor_weight"])
    seed = int(config["random_seed"])
    decision_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    evaluator = HeldOutDecisionEvaluator()
    optimizers: dict[str, DecisionOptimizationModule] = {
        "expected_cost": NewsvendorOptimizer(),
        "risk_aware": RiskAwareNewsvendorOptimizer(alpha=0.9, risk_weight=0.25),
    }

    for (target_id, cutoff), pool in reliability.groupby(["target_id", "cutoff"], sort=False):
        pool = pool.sort_values("similarity_rank")
        target_source = id_to_source[str(target_id)]
        order = target_order[target_source]
        cutoff = int(str(cutoff))
        donor_ids = pool["donor_id"].astype(str).tolist()
        donor_sources = [id_to_source[donor_id] for donor_id in donor_ids]
        histories = [np.asarray(demand[source, :cutoff], dtype=float) for source in donor_sources]
        similarity = pool["metadata_similarity"].to_numpy(dtype=float)
        logistic = pool["reliability_probability"].to_numpy(dtype=float)
        hgb = pool["hgb_reliability"].to_numpy(dtype=float)
        support = pool["donor_history_support"].to_numpy(dtype=float)
        shuffle_rng = np.random.default_rng(seed + cutoff * 1009 + order + 700_000)
        shuffled_reliability = shuffle_rng.permutation(logistic)
        reliability_by_method = {
            "uniform_multi": logistic,
            "similarity_support": logistic,
            "reliability_only": logistic,
            "combined_primary": logistic,
            "shuffled_reliability": shuffled_reliability,
            "hgb_combined": hgb,
            "single_analog": logistic,
        }
        native = min(
            max(1, int(round(float(dataset.metadata.iloc[target_source]["lead_time"])))),
            int(section["native_lead_time_cap"]),
        )
        lead_times = {
            "native_capped": native,
            "half_native_sensitivity": max(1, math.ceil(native / 2)),
        }
        holding, fixed, moq = _operational_values(dataset, target_source, eligible, section)

        broad_rng = np.random.default_rng(seed + cutoff * 1009 + order)
        broad_sources = np.sort(
            broad_rng.choice(
                allowed,
                size=min(int(config["representation"]["broad_pool_size"]), len(allowed)),
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
            scenario_sets: dict[str, DecisionScenarioSet] = {}
            for method in CONTROL_METHODS:
                weights = _weights(
                    similarity,
                    reliability_by_method[method],
                    support,
                    method,
                    maximum,
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
                scenario_sets[method] = scenario
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
            for ratio in config["decision"]["shortage_to_holding_ratios"]:
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
                    for objective, optimizer in optimizers.items():
                        # The matched-control extension uses the full frozen operational
                        # grid. Objective portability is deliberately bounded to one
                        # prespecified, interpretable slice to avoid turning this into a
                        # large optimization study.
                        if objective == "risk_aware" and not (
                            lead_regime == "native_capped"
                            and float(ratio) == 5.0
                            and capacity_regime == "unconstrained"
                        ):
                            continue
                        oracle_decision = optimizer.optimize(oracle, context)
                        for method, scenario in scenario_sets.items():
                            decision = optimizer.optimize(scenario, context)
                            evaluation = evaluator.evaluate(
                                decision, actual, context, oracle_decision
                            )
                            decision_rows.append(
                                {
                                    "dataset": dataset.name,
                                    "target_id": target_id,
                                    "cutoff": cutoff,
                                    "lead_time_regime": lead_regime,
                                    "lead_time": lead,
                                    "capacity_regime": capacity_regime,
                                    "shortage_holding_ratio": ratio,
                                    "objective": objective,
                                    "method": method,
                                    "selected_level": decision.level,
                                    "total_normalized_cost": evaluation.total_cost,
                                    "normalized_regret": evaluation.normalized_regret,
                                    "fill_rate_proxy": evaluation.fill_rate_proxy,
                                }
                            )
    return pd.DataFrame(decision_rows), pd.DataFrame(scenario_rows)


def _paired_summary(results: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    context = [
        "target_id",
        "cutoff",
        "lead_time_regime",
        "capacity_regime",
        "shortage_holding_ratio",
        "objective",
    ]
    for (dataset, objective), group in results.groupby(["dataset", "objective"]):
        for left, right in COMPARISONS:
            pivot = group[group["method"].isin([left, right])].pivot(
                index=context, columns="method", values=["normalized_regret", "fill_rate_proxy"]
            )
            if ("normalized_regret", left) not in pivot or (
                "normalized_regret",
                right,
            ) not in pivot:
                continue
            multi_columns = cast(pd.MultiIndex, pivot.columns)
            pivot.columns = [
                f"{metric}__{method}" for metric, method in multi_columns.to_flat_index()
            ]
            paired = pivot.reset_index()
            paired["difference"] = (
                paired[f"normalized_regret__{left}"] - paired[f"normalized_regret__{right}"]
            )
            low, high = clustered_bootstrap_difference(
                paired,
                "difference",
                repetitions=repetitions,
                random_seed=seed + len(rows),
            )
            differences = paired["difference"].to_numpy(dtype=float)
            target_effect = paired.groupby("target_id")["difference"].mean()
            effect_sd = float(target_effect.std(ddof=1))
            rows.append(
                {
                    "dataset": dataset,
                    "objective": objective,
                    "comparison": f"{left} minus {right}",
                    "mean_difference": differences.mean(),
                    "median_difference": np.median(differences),
                    "ci_low": low,
                    "ci_high": high,
                    "win_rate": (differences < 0).mean(),
                    "standardized_effect": (
                        float(target_effect.mean() / effect_sd) if effect_sd > 0 else np.nan
                    ),
                    "fill_rate_difference": (
                        paired[f"fill_rate_proxy__{left}"] - paired[f"fill_rate_proxy__{right}"]
                    ).mean(),
                }
            )
    return pd.DataFrame(rows)


def _method_summary(results: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    method = results.groupby(["dataset", "objective", "method"], as_index=False).agg(
        mean_regret=("normalized_regret", "mean"),
        median_regret=("normalized_regret", "median"),
        fill_rate=("fill_rate_proxy", "mean"),
        mean_cost=("total_normalized_cost", "mean"),
        cvar_style_cost=(
            "total_normalized_cost",
            lambda values: float(values[values >= values.quantile(0.9)].mean()),
        ),
    )
    diagnostics = scenarios.groupby(["dataset", "method"], as_index=False).agg(
        scenario_coverage=("coverage", "mean"),
        scenario_diversity=("scenario_diversity", "mean"),
        effective_donor_count=("effective_donor_count", "mean"),
        entropy=("donor_entropy", "mean"),
        maximum_donor_share=("maximum_donor_share", "mean"),
    )
    return method.merge(diagnostics, on=["dataset", "method"], how="left")


def _learner_pool_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in scored.groupby("dataset"):
        pools = []
        for _, pool in group.groupby(["target_id", "cutoff"]):
            pools.append(
                {
                    "top_stability": float(
                        pool.loc[pool["reliability_probability"].idxmax(), "donor_id"]
                        == pool.loc[pool["hgb_reliability"].idxmax(), "donor_id"]
                    ),
                    "hgb_similarity_disagreement": float(
                        pool.loc[pool["metadata_similarity"].idxmax(), "donor_id"]
                        != pool.loc[pool["hgb_reliability"].idxmax(), "donor_id"]
                    ),
                    "similarity_rank_correlation": spearmanr(
                        pool["metadata_similarity"], pool["hgb_reliability"]
                    ).statistic,
                    "logistic_hgb_rank_correlation": spearmanr(
                        pool["reliability_probability"], pool["hgb_reliability"]
                    ).statistic,
                }
            )
        frame = pd.DataFrame(pools)
        summary: dict[str, Any] = {"dataset": dataset}
        summary.update({str(key): value for key, value in frame.mean().to_dict().items()})
        rows.append(summary)
    return pd.DataFrame(rows)


def _planner_actionability(
    results: pd.DataFrame,
    scenarios: pd.DataFrame,
    reliability: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = results[results["objective"] == "expected_cost"].copy()
    index = [
        "dataset",
        "target_id",
        "cutoff",
        "lead_time_regime",
        "lead_time",
        "capacity_regime",
        "shortage_holding_ratio",
    ]
    pivot = (
        base[base["method"].isin(["single_analog", "combined_primary"])]
        .pivot(index=index, columns="method", values="normalized_regret")
        .reset_index()
    )
    pivot["single_minus_multi"] = pivot["single_analog"] - pivot["combined_primary"]
    pool_rows: list[dict[str, Any]] = []
    for keys, pool in reliability.groupby(["dataset", "target_id", "cutoff"]):
        similarity = np.sort(pool["metadata_similarity"].to_numpy())[::-1]
        reliability_values = np.sort(pool["reliability_probability"].to_numpy())[::-1]
        pool_rows.append(
            {
                "dataset": keys[0],
                "target_id": keys[1],
                "cutoff": keys[2],
                "top_disagreement": float(pool["top_rank_disagreement"].iloc[0]),
                "similarity_margin": similarity[0] - similarity[1],
                "reliability_margin": reliability_values[0] - reliability_values[1],
                "mean_visible_support": pool["donor_history_support"].mean(),
                "pool_dispersion": pool["pool_mean_dispersion"].iloc[0],
            }
        )
    pivot = pivot.merge(pd.DataFrame(pool_rows), on=["dataset", "target_id", "cutoff"])
    combined_diag = scenarios[scenarios["method"] == "combined_primary"].drop_duplicates(
        ["dataset", "target_id", "cutoff", "lead_time_regime"]
    )
    pivot = pivot.merge(
        combined_diag[
            [
                "dataset",
                "target_id",
                "cutoff",
                "lead_time_regime",
                "effective_donor_count",
                "donor_entropy",
                "maximum_donor_share",
                "scenario_diversity",
            ]
        ],
        on=["dataset", "target_id", "cutoff", "lead_time_regime"],
    )
    regime_rows: list[dict[str, Any]] = []
    for dataset, group in pivot.groupby("dataset"):
        medians = {
            column: group[column].median()
            for column in (
                "similarity_margin",
                "mean_visible_support",
                "scenario_diversity",
                "donor_entropy",
            )
        }
        regimes = {
            "top donors disagree": group["top_disagreement"] == 1,
            "top donors agree": group["top_disagreement"] == 0,
            "low similarity margin": group["similarity_margin"] <= medians["similarity_margin"],
            "high similarity margin": group["similarity_margin"] > medians["similarity_margin"],
            "low visible support": group["mean_visible_support"] <= medians["mean_visible_support"],
            "high visible support": group["mean_visible_support"] > medians["mean_visible_support"],
            "high scenario dispersion": group["scenario_diversity"] > medians["scenario_diversity"],
            "low scenario dispersion": group["scenario_diversity"] <= medians["scenario_diversity"],
            "native lead time": group["lead_time_regime"] == "native_capped",
            "half lead time": group["lead_time_regime"] == "half_native_sensitivity",
            "tight capacity": group["capacity_regime"] == "tight",
            "unconstrained capacity": group["capacity_regime"] == "unconstrained",
            "high shortage ratio": group["shortage_holding_ratio"] == 10,
            "low shortage ratio": group["shortage_holding_ratio"] == 2,
        }
        for label, mask in regimes.items():
            subset = group.loc[mask]
            low, high = clustered_bootstrap_difference(
                subset,
                "single_minus_multi",
                repetitions=repetitions,
                random_seed=seed + len(regime_rows),
            )
            cutoff_effects = subset.groupby("cutoff")["single_minus_multi"].mean()
            regime_rows.append(
                {
                    "dataset": dataset,
                    "regime": label,
                    "rows": len(subset),
                    "mean_single_minus_multi": subset["single_minus_multi"].mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "cutoffs_positive": int((cutoff_effects > 0).sum()),
                    "cutoff_count": len(cutoff_effects),
                }
            )

    model_rows: list[dict[str, Any]] = []
    numeric = [
        "similarity_margin",
        "reliability_margin",
        "mean_visible_support",
        "pool_dispersion",
        "lead_time",
        "shortage_holding_ratio",
        "effective_donor_count",
        "donor_entropy",
        "maximum_donor_share",
        "scenario_diversity",
        "top_disagreement",
    ]
    categorical = ["lead_time_regime", "capacity_regime"]
    for dataset, group in pivot.groupby("dataset"):
        predictions = np.zeros(len(group), dtype=float)
        groups = group["target_id"].astype(str).to_numpy()
        splitter = GroupKFold(5)
        for train_index, test_index in splitter.split(group, groups=groups):
            transformer = ColumnTransformer(
                [
                    ("numeric", StandardScaler(), numeric),
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                ]
            )
            model = make_pipeline(transformer, Ridge(alpha=10.0))
            model.fit(
                group.iloc[train_index][numeric + categorical],
                group.iloc[train_index]["single_minus_multi"],
            )
            predictions[test_index] = model.predict(group.iloc[test_index][numeric + categorical])
        model_rows.append(
            {
                "dataset": dataset,
                "out_of_fold_r2": r2_score(group["single_minus_multi"], predictions),
                "out_of_fold_spearman": spearmanr(
                    group["single_minus_multi"], predictions
                ).statistic,
            }
        )
    return pd.DataFrame(regime_rows), pd.DataFrame(model_rows)


def run_acceptance_enhancement() -> dict[str, Any]:
    with resolve_repo_path("configs/full_scale.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = resolve_repo_path("outputs/acceptance_enhancement")
    output.mkdir(parents=True, exist_ok=True)
    repetitions = int(config["statistics"]["bootstrap_repetitions"])
    seed = int(config["random_seed"])
    all_results: list[pd.DataFrame] = []
    all_scenarios: list[pd.DataFrame] = []
    all_scored: list[pd.DataFrame] = []
    learner_frames: list[pd.DataFrame] = []

    for dataset in (parse_man(), parse_braf()):
        checkpoint = pd.read_parquet(
            resolve_repo_path(
                f"outputs/full_scale/checkpoints/{dataset.name.lower()}_reliability.parquet"
            )
        )
        learner, scored, target_sources, eligible = _prepare_dataset(dataset, config, checkpoint)
        learner_frames.append(learner)
        all_scored.append(scored)
        # Reconstruct cross-fitted metrics from the frozen calibration table for logistic,
        # and use the target-pool comparison below for learner stability.
        results, scenarios = _evaluate_dataset(dataset, config, scored, target_sources, eligible)
        all_results.append(results)
        all_scenarios.append(scenarios)

    results = pd.concat(all_results, ignore_index=True)
    scenarios = pd.concat(all_scenarios, ignore_index=True)
    scored = pd.concat(all_scored, ignore_index=True)
    paired = _paired_summary(results, repetitions, seed)
    methods = _method_summary(results, scenarios)
    pool_summary = _learner_pool_summary(scored)

    learner = pd.concat(learner_frames, ignore_index=True)
    planner_regimes, planner_model = _planner_actionability(
        results, scenarios, scored, repetitions, seed
    )
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    methods.to_csv(output / "method_summary.csv", index=False)
    learner.to_csv(output / "learner_metrics.csv", index=False)
    pool_summary.to_csv(output / "learner_pool_stability.csv", index=False)
    planner_regimes.to_csv(output / "planner_regimes.csv", index=False)
    planner_model.to_csv(output / "planner_cross_fitted_model.csv", index=False)

    expected_pairs = paired[paired["objective"] == "expected_cost"]
    risk_pairs = paired[paired["objective"] == "risk_aware"]
    _write_summary(
        output / "diversification_control_summary.md",
        "Diversification Control Summary",
        "CONFIRMATORY EXTENSION",
        "Matched top-10 donor pools, scenario counts, operational grids, and maximum donor shares are held fixed. Negative differences favor the first method.\n\n"
        + _markdown_table(expected_pairs)
        + "\n\n## Scenario matching diagnostics\n\n"
        + _markdown_table(
            methods[methods["objective"] == "expected_cost"][
                [
                    "dataset",
                    "method",
                    "scenario_coverage",
                    "effective_donor_count",
                    "entropy",
                    "maximum_donor_share",
                ]
            ]
        ),
    )
    _write_summary(
        output / "learner_robustness_summary.md",
        "Learner-Class Robustness Summary",
        "CONFIRMATORY EXTENSION",
        "The supervision, features, pseudo-target groups, cutoffs, and target exclusion are identical across learners.\n\n"
        + _markdown_table(learner)
        + "\n\n## Target-pool stability\n\n"
        + _markdown_table(pool_summary)
        + "\n\n## Downstream comparisons\n\n"
        + _markdown_table(
            expected_pairs[expected_pairs["comparison"].str.startswith("hgb_combined")]
        ),
    )
    _write_summary(
        output / "or_objective_portability_summary.md",
        "OR Objective Portability Summary",
        "CONFIRMATORY EXTENSION",
        "The second formulation minimizes expected scenario cost plus 0.25 times scenario CVaR at alpha 0.90. It uses the same scenarios and constraints and is not presented as optimization novelty. To keep the extension bounded, portability is evaluated on the prespecified native-lead-time, shortage-to-holding ratio 5, unconstrained-capacity slice; the expected-cost matched controls retain the complete frozen operational grid.\n\n"
        + _markdown_table(risk_pairs)
        + "\n\n## Method outcomes under both objectives\n\n"
        + _markdown_table(methods),
    )
    _write_summary(
        output / "planner_actionability_summary.md",
        "Planner Actionability Summary",
        "EXPLORATORY",
        "Positive effects mean single-analog commitment has higher regret than the primary multi-analog method. All features are available before the target outcome; learned associations use target-grouped cross-fitting.\n\n"
        + _markdown_table(planner_model)
        + "\n\n## Prespecified regime summaries\n\n"
        + _markdown_table(planner_regimes),
    )

    return {
        "paired": paired,
        "methods": methods,
        "learner": learner,
        "pool_summary": pool_summary,
        "planner_regimes": planner_regimes,
        "planner_model": planner_model,
        "output": output,
    }
