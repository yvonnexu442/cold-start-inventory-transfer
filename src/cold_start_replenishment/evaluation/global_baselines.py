"""Global and structural baselines on the frozen target populations and grids."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.metrics import mean_absolute_error, mean_pinball_loss  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional comparator dependency
    CatBoostClassifier = CatBoostRegressor = None  # type: ignore[assignment,misc]

from cold_start_replenishment.data.spdf_pilot import PilotDataset, parse_braf, parse_man
from cold_start_replenishment.evaluation.acceptance_enhancement import (
    _construct_scenarios,
    _prepare_dataset,
)
from cold_start_replenishment.evaluation.decision_evaluator import HeldOutDecisionEvaluator
from cold_start_replenishment.evaluation.sprint2 import _operational_values
from cold_start_replenishment.evaluation.sprint2_statistics import clustered_bootstrap_difference
from cold_start_replenishment.evaluation.targeted_acceptance_extension import (
    _method_weights,
)
from cold_start_replenishment.framework.objects import DecisionScenarioSet, OperationalContext
from cold_start_replenishment.inventory.newsvendor import NewsvendorOptimizer
from cold_start_replenishment.paths import resolve_repo_path

QUANTILES = (0.05, 0.10, 0.25, 0.50, 2 / 3, 0.75, 5 / 6, 0.90, 10 / 11, 0.95)
BLOCK_METHODS = ("single_analog", "uniform_multi", "forecast_combined")


def _load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    with resolve_repo_path("configs/full_scale.yaml").open(encoding="utf-8") as handle:
        full = cast(dict[str, Any], yaml.safe_load(handle))
    with resolve_repo_path("configs/targeted_acceptance_extension.yaml").open(
        encoding="utf-8"
    ) as handle:
        extension = cast(dict[str, Any], yaml.safe_load(handle))
    return full, extension


def _global_feature_spec(dataset: PilotDataset) -> tuple[list[str], list[str]]:
    if dataset.name == "MAN":
        return ["lead_time", "cost_price", "forecast_horizon"], ["product_group"]
    return ["lead_time", "price", "forecast_horizon"], ["description"]


def build_global_training_frame(
    dataset: PilotDataset,
    cutoff: int,
    eligible_sources: NDArray[np.int64],
    evaluation_targets: NDArray[np.int64],
    native_lead_cap: int,
) -> pd.DataFrame:
    """Build leakage-safe donor examples whose labels end at the evaluation cutoff."""

    excluded = set(int(value) for value in evaluation_targets)
    sources = np.asarray(
        [source for source in eligible_sources if int(source) not in excluded], dtype=np.int64
    )
    metadata = dataset.metadata.iloc[sources].copy().reset_index(drop=True)
    leads = np.clip(
        np.rint(metadata["lead_time"].to_numpy(dtype=float)).astype(int), 1, native_lead_cap
    )
    demand = dataset.demand.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    # Four earlier origins per part and both deployed horizons.  Every label is
    # fully realized by the current cutoff; evaluation targets remain excluded.
    for row_index, (source, native) in enumerate(zip(sources, leads, strict=True)):
        for horizon in sorted({int(native), max(1, math.ceil(int(native) / 2))}):
            latest = cutoff - horizon
            origins = np.linspace(max(0, latest - 3 * horizon), latest, 4, dtype=int)
            for origin in np.unique(origins):
                row = metadata.iloc[row_index].to_dict()
                row.update(
                    {
                        "source_index": int(source),
                        "forecast_horizon": horizon,
                        "label_start": int(origin),
                        "label_end": int(origin + horizon),
                        "lead_demand": float(demand[source, origin : origin + horizon].sum()),
                    }
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["is_evaluation_target"] = frame["source_index"].isin(excluded)
    return frame


class GlobalQuantileBaseline:
    """Transparent metadata-conditioned global lead-demand quantile learner."""

    def __init__(self, dataset: PilotDataset, random_seed: int) -> None:
        numeric, categorical = _global_feature_spec(dataset)
        self.features = numeric + categorical
        self.numeric = numeric
        self.categorical = categorical
        self.random_seed = random_seed
        self.models: dict[float, Pipeline] = {}

    def fit(self, training: pd.DataFrame) -> GlobalQuantileBaseline:
        transformer = ColumnTransformer(
            [
                ("numeric", StandardScaler(), self.numeric),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False),
                    self.categorical,
                ),
            ]
        )
        for index, quantile in enumerate(QUANTILES):
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=self.random_seed + index,
            )
            pipeline = Pipeline([("features", transformer), ("model", model)])
            pipeline.fit(training[self.features], training["lead_demand"])
            self.models[quantile] = pipeline
        return self

    def predict_quantiles(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        raw = np.column_stack(
            [self.models[quantile].predict(frame[self.features]) for quantile in QUANTILES]
        )
        return np.maximum.accumulate(np.clip(raw, 0.0, None), axis=1)


class StructuralZIGBaseline:
    """CatBoost-adapted horizon-level reproduction of Nathan et al. (2026).

    The occurrence classifier and conditional-positive regressor use CatBoost,
    as in the published framework.  The present rolling labels and public-data
    feature schema remain study-specific, so this is an adapted structural
    reproduction rather than an exact reconstruction of the article pipeline.
    """

    def __init__(self, dataset: PilotDataset, random_seed: int, shape: float = 2.0) -> None:
        numeric, categorical = _global_feature_spec(dataset)
        self.features = numeric + categorical
        self.numeric = numeric
        self.categorical = categorical
        self.random_seed = random_seed
        self.shape = shape
        self.occurrence: Any | None = None
        self.magnitude: Any | None = None

    def _frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame[self.features].copy()
        for column in self.categorical:
            output[column] = output[column].fillna("__MISSING__").astype(str)
        for column in self.numeric:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
        return output

    def fit(self, training: pd.DataFrame) -> StructuralZIGBaseline:
        if CatBoostClassifier is None or CatBoostRegressor is None:
            raise RuntimeError("CatBoost is required for the adapted ZIG--MC comparator")
        x = self._frame(training)
        categorical_indices = [self.features.index(column) for column in self.categorical]
        event = (training["lead_demand"].to_numpy(float) > 0).astype(int)
        self.occurrence = CatBoostClassifier(
            iterations=180,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            random_seed=self.random_seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        ).fit(x, event, cat_features=categorical_indices)
        positive = training[training.lead_demand > 0].copy()
        positive["log_magnitude"] = np.log1p(positive["lead_demand"])
        self.magnitude = CatBoostRegressor(
            iterations=180,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=self.random_seed + 1,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        ).fit(
            self._frame(positive),
            positive["log_magnitude"],
            cat_features=categorical_indices,
        )
        return self

    def predict_parameters(
        self, frame: pd.DataFrame
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.occurrence is None or self.magnitude is None:
            raise RuntimeError("fit StructuralZIGBaseline before prediction")
        x = self._frame(frame)
        probability = self.occurrence.predict_proba(x)[:, 1]
        mean = np.expm1(self.magnitude.predict(x))
        return np.clip(probability, 1e-4, 1 - 1e-4), np.clip(mean, 1e-4, None)

    def scenarios(
        self, frame: pd.DataFrame, count: int, seed: int, horizon: int
    ) -> DecisionScenarioSet:
        probability, mean = self.predict_parameters(frame)
        rng = np.random.default_rng(seed)
        event = rng.random(count) < probability[0]
        magnitude = rng.gamma(self.shape, mean[0] / self.shape, size=count)
        values = event * magnitude
        trajectories = np.zeros((count, horizon), dtype=float)
        trajectories[:, -1] = values
        return DecisionScenarioSet(
            trajectories,
            horizon,
            "zig_mc_catboost_adapted",
            {
                "occurrence_probability": float(probability[0]),
                "positive_mean": float(mean[0]),
                "gamma_shape": self.shape,
            },
        )


def quantiles_to_scenarios(
    quantiles: NDArray[np.float64], horizon: int, count: int, seed: int, method: str
) -> DecisionScenarioSet:
    """Interpolate a compact quantile grid into deterministic Monte Carlo draws."""

    values = np.maximum.accumulate(np.clip(np.asarray(quantiles, dtype=float), 0.0, None))
    rng = np.random.default_rng(seed)
    probabilities = rng.random(count)
    samples = np.interp(
        probabilities, np.asarray(QUANTILES), values, left=values[0], right=values[-1]
    )
    trajectories = np.zeros((count, horizon), dtype=float)
    trajectories[:, 0] = samples
    return DecisionScenarioSet(trajectories, horizon, method, {"quantile_grid": list(QUANTILES)})


def contiguous_block_scenarios(
    donor_ids: list[str],
    histories: list[NDArray[np.float64]],
    weights: NDArray[np.float64],
    horizon: int,
    count: int,
    seed: int,
    method: str,
) -> DecisionScenarioSet:
    """Sample one donor and one cutoff-visible contiguous block per scenario."""

    if horizon <= 0 or count <= 0:
        raise ValueError("horizon and count must be positive")
    if any(len(history) < horizon for history in histories):
        raise ValueError("Every donor must have at least one cutoff-visible block")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(donor_ids), size=count, p=weights)
    scenarios = np.zeros((count, horizon), dtype=float)
    sources: list[str] = []
    starts: list[int] = []
    for row, position in enumerate(selected):
        history = histories[int(position)]
        start = int(rng.integers(0, len(history) - horizon + 1))
        scenarios[row] = history[start : start + horizon]
        sources.append(donor_ids[int(position)])
        starts.append(start)
    return DecisionScenarioSet(
        scenarios,
        horizon,
        method,
        {
            "sampler": "contiguous_donor_block",
            "maximum_block_end": max(start + horizon for start in starts),
            "effective_donor_count": float(1.0 / np.square(weights).sum()),
        },
        tuple(sources),
        np.full(count, 1.0 / count),
        "cutoff_visible_contiguous_donor_blocks",
    )


def reference_cost_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic within-dataset reference-cost quantile strata."""

    output = frame.copy()
    labels = ["bottom_10", "10_25", "25_50", "50_75", "top_25"]
    output["reference_cost_stratum"] = output.groupby("dataset", group_keys=False)[
        "oracle_target_history_cost"
    ].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), [0, 0.1, 0.25, 0.5, 0.75, 1], labels=labels
        )
    )
    return output


def holm_adjust(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Deterministic Holm family-wise-error adjustment."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (len(values) - rank) * values[position])
        adjusted[position] = min(running, 1.0)
    return adjusted


def candidate_order_is_deterministic(scores: NDArray[np.float64], top_k: int) -> bool:
    first = np.argsort(-np.asarray(scores), kind="stable")[:top_k]
    second = np.argsort(-np.asarray(scores), kind="stable")[:top_k]
    return bool(np.array_equal(first, second))


def _coverage(actual: float, samples: NDArray[np.float64]) -> tuple[float, float]:
    lower, upper = np.quantile(samples, [0.1, 0.9])
    tail_lower, tail_upper = np.quantile(samples, [0.05, 0.95])
    return float(lower <= actual <= upper), float(tail_lower <= actual <= tail_upper)


def _global_and_block_results(
    dataset: PilotDataset,
    full: dict[str, Any],
    extension: dict[str, Any],
    scored: pd.DataFrame,
    targets: NDArray[np.int64],
    eligible: NDArray[np.int64],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section = full["datasets"][dataset.name]
    demand = dataset.demand.to_numpy(dtype=float)
    id_to_source = (
        {
            str(row.target_id): int(str(row.target_source))
            for row in scored[["target_id", "target_source"]].drop_duplicates().itertuples()
        }
        if "target_source" in scored
        else {}
    )
    if not id_to_source:
        from cold_start_replenishment.data.spdf_pilot import redacted_id

        id_to_source = {
            redacted_id(dataset.name, row.item_id): int(str(index))
            for index, row in dataset.metadata.iterrows()
        }
    seed = int(extension["random_seed"])
    count = int(extension["matched_decomposition"]["scenario_count"])
    cap = float(extension["matched_decomposition"]["maximum_donor_weight"])
    excluded = set(int(value) for value in targets)
    allowed = np.asarray([source for source in eligible if int(source) not in excluded], dtype=int)
    target_order = {int(source): index for index, source in enumerate(targets)}
    optimizer = NewsvendorOptimizer()
    evaluator = HeldOutDecisionEvaluator()
    global_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    learners: dict[int, GlobalQuantileBaseline] = {}
    zig_learners: dict[int, StructuralZIGBaseline] = {}
    prediction_cache: dict[tuple[int, int, int], NDArray[np.float64]] = {}
    for cutoff in section["cutoffs"]:
        training = build_global_training_frame(
            dataset, int(cutoff), eligible, targets, int(section["native_lead_time_cap"])
        )
        if training["is_evaluation_target"].any() or int(training["label_end"].max()) > int(cutoff):
            raise RuntimeError("Global baseline information boundary violation")
        learners[int(cutoff)] = GlobalQuantileBaseline(dataset, seed + int(cutoff)).fit(training)
        zig_learners[int(cutoff)] = StructuralZIGBaseline(dataset, seed + 10_000 + int(cutoff)).fit(
            training
        )
        prediction_rows = []
        prediction_keys = []
        for target_source in targets:
            native = min(
                max(1, int(round(float(dataset.metadata.iloc[target_source]["lead_time"])))),
                int(section["native_lead_time_cap"]),
            )
            for lead in sorted({native, max(1, math.ceil(native / 2))}):
                row = dataset.metadata.iloc[int(target_source)].to_dict()
                row["forecast_horizon"] = lead
                prediction_rows.append(row)
                prediction_keys.append((int(cutoff), int(target_source), lead))
        predictions = learners[int(cutoff)].predict_quantiles(pd.DataFrame(prediction_rows))
        prediction_cache.update(dict(zip(prediction_keys, predictions, strict=True)))

    for (target_id, cutoff_value), pool in scored.groupby(["target_id", "cutoff"], sort=False):
        pool = pool.sort_values("similarity_rank")
        target_source = id_to_source[str(target_id)]
        target_position = target_order[target_source]
        cutoff = int(str(cutoff_value))
        donor_ids = pool["donor_id"].astype(str).tolist()
        donor_sources = [id_to_source[donor_id] for donor_id in donor_ids]
        histories = [np.asarray(demand[source, :cutoff], dtype=float) for source in donor_sources]
        similarity = pool["metadata_similarity"].to_numpy(dtype=float)
        reliability = pool["reliability_probability"].to_numpy(dtype=float)
        support = pool["donor_history_support"].to_numpy(dtype=float)
        native = min(
            max(1, int(round(float(dataset.metadata.iloc[target_source]["lead_time"])))),
            int(section["native_lead_time_cap"]),
        )
        lead_times = {
            "native_capped": native,
            "half_native_sensitivity": max(1, math.ceil(native / 2)),
        }
        holding, fixed, moq = _operational_values(dataset, target_source, eligible, section)
        for lead_regime, lead in lead_times.items():
            predicted = prediction_cache[(cutoff, target_source, lead)]
            global_scenario = quantiles_to_scenarios(
                predicted,
                lead,
                count,
                seed + target_position * 10007 + cutoff * 101 + 61_000,
                "global_quantile",
            )
            target_frame = dataset.metadata.iloc[[target_source]].copy()
            target_frame["forecast_horizon"] = lead
            zig_scenario = zig_learners[cutoff].scenarios(
                target_frame, count, seed + target_position * 10007 + cutoff * 101 + 61_500, lead
            )
            pooled_histories = [
                np.asarray(demand[source, :cutoff], dtype=float) for source in allowed
            ]
            pooled_ids = [str(source) for source in allowed]
            pooled_scenario = _construct_scenarios(
                pooled_ids,
                pooled_histories,
                np.full(len(allowed), 1 / len(allowed)),
                lead,
                count,
                seed + target_position * 10007 + cutoff * 101 + 62_000,
                "group_empirical",
            )
            actual = float(demand[target_source, cutoff : cutoff + lead].sum())
            for name, scenario in (
                ("global_quantile", global_scenario),
                ("zig_mc_catboost_adapted", zig_scenario),
                ("group_empirical", pooled_scenario),
            ):
                cover, tail = _coverage(actual, scenario.lead_time_demand)
                diagnostic_rows.append(
                    {
                        "dataset": dataset.name,
                        "target_id": target_id,
                        "cutoff": cutoff,
                        "lead_time_regime": lead_regime,
                        "sampler": "global",
                        "method": name,
                        "coverage": cover,
                        "tail_coverage": tail,
                        "dispersion": float(scenario.lead_time_demand.std()),
                        "mae": mean_absolute_error(
                            [actual], [float(np.median(scenario.lead_time_demand))]
                        ),
                        "pinball_loss": float(
                            np.mean(
                                [
                                    mean_pinball_loss([actual], [value], alpha=q)
                                    for q, value in zip(QUANTILES, predicted, strict=True)
                                ]
                            )
                        )
                        if name == "global_quantile"
                        else np.nan,
                    }
                )

            weights_by_method = {
                method: _method_weights(
                    similarity,
                    reliability,
                    reliability,
                    support,
                    method,
                    cap,
                    reliability,
                    reliability,
                )
                for method in BLOCK_METHODS
            }
            block_scenarios = {
                method: contiguous_block_scenarios(
                    donor_ids,
                    histories,
                    weights,
                    lead,
                    count,
                    seed + target_position * 10007 + cutoff * 101 + 71_000,
                    method,
                )
                for method, weights in weights_by_method.items()
            }
            iid_scenarios = {
                method: _construct_scenarios(
                    donor_ids,
                    histories,
                    weights,
                    lead,
                    count,
                    seed + target_position * 10007 + cutoff * 101 + 71_000,
                    method,
                )
                for method, weights in weights_by_method.items()
            }
            for sampler, scenarios in (
                ("iid_occurrence_size", iid_scenarios),
                ("contiguous_block", block_scenarios),
            ):
                for method, scenario in scenarios.items():
                    cover, tail = _coverage(actual, scenario.lead_time_demand)
                    diagnostic_rows.append(
                        {
                            "dataset": dataset.name,
                            "target_id": target_id,
                            "cutoff": cutoff,
                            "lead_time_regime": lead_regime,
                            "sampler": sampler,
                            "method": method,
                            "coverage": cover,
                            "tail_coverage": tail,
                            "dispersion": float(scenario.lead_time_demand.std()),
                            "intermittency": float(
                                np.mean([np.mean(history == 0) for history in histories])
                            ),
                        }
                    )

            broad = pooled_scenario
            capacities = {
                "unconstrained": None,
                "medium": max(moq, float(np.quantile(broad.lead_time_demand, 0.75))),
                "tight": max(moq, float(np.quantile(broad.lead_time_demand, 0.50))),
            }
            oracle = _construct_scenarios(
                [str(target_id)],
                [np.asarray(demand[target_source, :cutoff], dtype=float)],
                np.ones(1),
                lead,
                count,
                seed + target_position * 10007 + cutoff * 101 + 999,
                "oracle_target_history",
            )
            donor_scale = max(
                float(np.median([history.mean() * lead for history in histories])), 1.0
            )
            for ratio in extension["matched_decomposition"]["shortage_to_holding_ratios"]:
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
                    for name, scenario in (
                        ("global_quantile", global_scenario),
                        ("zig_mc_catboost_adapted", zig_scenario),
                        ("group_empirical", pooled_scenario),
                    ):
                        decision = optimizer.optimize(scenario, context)
                        evaluation = evaluator.evaluate(decision, actual, context, reference)
                        raw = float(evaluation.regret or 0.0)
                        global_rows.append(
                            {
                                "dataset": dataset.name,
                                "target_id": target_id,
                                "cutoff": cutoff,
                                "lead_time_regime": lead_regime,
                                "lead_time": lead,
                                "capacity_regime": capacity_regime,
                                "shortage_holding_ratio": float(ratio),
                                "method": name,
                                "actual_demand": actual,
                                "donor_pool_scale": donor_scale,
                                "selected_level": decision.level,
                                "total_cost": evaluation.total_cost,
                                "oracle_target_history_cost": evaluation.total_cost - raw,
                                "oracle_cost_normalized_regret": evaluation.normalized_regret,
                                "raw_regret": raw,
                                "realized_demand_scaled_regret": raw / max(actual, 1.0),
                                "donor_pool_scaled_regret": raw / donor_scale,
                                "fill_rate_proxy": evaluation.fill_rate_proxy,
                            }
                        )
                    for sampler, scenarios in (
                        ("iid_occurrence_size", iid_scenarios),
                        ("contiguous_block", block_scenarios),
                    ):
                        for method, scenario in scenarios.items():
                            decision = optimizer.optimize(scenario, context)
                            evaluation = evaluator.evaluate(decision, actual, context, reference)
                            raw = float(evaluation.regret or 0.0)
                            block_rows.append(
                                {
                                    "dataset": dataset.name,
                                    "target_id": target_id,
                                    "cutoff": cutoff,
                                    "lead_time_regime": lead_regime,
                                    "lead_time": lead,
                                    "capacity_regime": capacity_regime,
                                    "shortage_holding_ratio": float(ratio),
                                    "sampler": sampler,
                                    "method": method,
                                    "total_cost": evaluation.total_cost,
                                    "oracle_target_history_cost": evaluation.total_cost - raw,
                                    "normalized_regret": evaluation.normalized_regret,
                                    "raw_regret": raw,
                                    "holding_cost": evaluation.holding_cost,
                                    "shortage_cost": evaluation.shortage_cost,
                                    "fill_rate_proxy": evaluation.fill_rate_proxy,
                                }
                            )
    return pd.DataFrame(global_rows), pd.DataFrame(block_rows), pd.DataFrame(diagnostic_rows)


def _paired(frame: pd.DataFrame, left: str, right: str, metric: str, seed: int) -> dict[str, Any]:
    index = ["target_id", "cutoff", "lead_time_regime", "capacity_regime", "shortage_holding_ratio"]
    pivot = frame[frame["method"].isin([left, right])].pivot(
        index=index, columns="method", values=metric
    )
    paired = pivot[[left, right]].dropna().reset_index()
    paired["difference"] = paired[left] - paired[right]
    low, high = clustered_bootstrap_difference(
        paired, "difference", repetitions=2000, random_seed=seed
    )
    return {
        "comparison": f"{left} minus {right}",
        "metric": metric,
        "mean": paired["difference"].mean(),
        "median": paired["difference"].median(),
        "ci_low": low,
        "ci_high": high,
        "win_rate": (paired["difference"] < 0).mean(),
    }


def _write_markdown(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    body = [f"# {title}", ""]
    for heading, content in sections:
        body.extend([f"## {heading}", "", content.rstrip(), ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No estimable rows."
    rendered = frame.copy()
    for column in rendered.select_dtypes(include=["number"]).columns:
        rendered[column] = rendered[column].map(lambda value: f"{value:.5f}")
    columns = list(rendered.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rendered.fillna("").astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def run_global_baseline_enhancement() -> dict[str, Any]:
    full, extension = _load_configs()
    output = resolve_repo_path("outputs/ai_darld_v3/global_baseline_enhancement")
    output.mkdir(parents=True, exist_ok=True)
    run_output = resolve_repo_path("outputs/runs/global_baseline_enhancement")
    run_output.mkdir(parents=True, exist_ok=True)
    global_cache = run_output / "global_decision_results.parquet"
    block_cache = run_output / "sampler_decision_results.parquet"
    diagnostic_cache = run_output / "forecast_diagnostics.parquet"
    global_parts: list[pd.DataFrame] = []
    block_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    determinism: list[dict[str, Any]] = []
    for dataset in (parse_man(), parse_braf()):
        checkpoint = pd.read_parquet(
            resolve_repo_path(
                f"outputs/full_scale/checkpoints/{dataset.name.lower()}_reliability.parquet"
            )
        )
        _, scored, targets, eligible = _prepare_dataset(dataset, full, checkpoint)
        if not (global_cache.exists() and block_cache.exists() and diagnostic_cache.exists()):
            global_rows, block_rows, diagnostics = _global_and_block_results(
                dataset, full, extension, scored, targets, eligible
            )
            global_parts.append(global_rows)
            block_parts.append(block_rows)
            diagnostic_parts.append(diagnostics)
        for _, pool in scored.groupby(["target_id", "cutoff"]):
            determinism.append(
                {
                    "dataset": dataset.name,
                    "deterministic": candidate_order_is_deterministic(
                        pool["metadata_similarity"].to_numpy(dtype=float), 10
                    ),
                    "pool_size": len(pool),
                }
            )
    if global_parts:
        global_rows = pd.concat(global_parts, ignore_index=True)
        block_rows = pd.concat(block_parts, ignore_index=True)
        diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
        global_rows.to_parquet(global_cache, index=False)
        block_rows.to_parquet(block_cache, index=False)
        diagnostics.to_parquet(diagnostic_cache, index=False)
    else:
        global_rows = pd.read_parquet(global_cache)
        block_rows = pd.read_parquet(block_cache)
        diagnostics = pd.read_parquet(diagnostic_cache)

    existing = pd.read_parquet(
        resolve_repo_path("outputs/runs/targeted_acceptance_extension/decision_results.parquet")
    )
    combined = pd.concat([existing, global_rows], ignore_index=True)
    comparison_rows: list[dict[str, Any]] = []
    for dataset_name, group in combined.groupby("dataset"):
        for left in ("uniform_multi", "forecast_combined", "group_empirical"):
            for metric in ("oracle_cost_normalized_regret", "raw_regret", "fill_rate_proxy"):
                row = _paired(
                    group,
                    left,
                    "global_quantile",
                    metric,
                    int(extension["random_seed"]) + len(comparison_rows),
                )
                row["dataset"] = dataset_name
                comparison_rows.append(row)
    global_comparisons = pd.DataFrame(comparison_rows)
    global_comparisons.to_csv(output / "global_comparisons.csv", index=False)

    forecast_summary = (
        diagnostics[diagnostics["sampler"] == "global"]
        .groupby(["dataset", "method"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            tail_coverage=("tail_coverage", "mean"),
            dispersion=("dispersion", "mean"),
            mae=("mae", "mean"),
            pinball_loss=("pinball_loss", "mean"),
        )
    )
    _write_markdown(
        output / "global_zero_history_baseline.md",
        "Global Zero-History Baseline",
        [
            (
                "Information boundary",
                "Training excludes every evaluation target. Each donor label is its own lead-time demand ending at, never after, the evaluation cutoff. Inference uses static target metadata only. Seven HGB quantile models use quantiles 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, and 0.95; monotone quantiles are interpolated to 300 scenarios and passed to the identical newsvendor layer.",
            ),
            ("Forecast diagnostics", _table(forecast_summary)),
            ("Decision comparisons", _table(global_comparisons)),
        ],
    )

    sampler_summary = block_rows.groupby(["dataset", "sampler", "method"], as_index=False).agg(
        normalized_regret=("normalized_regret", "mean"),
        raw_regret=("raw_regret", "mean"),
        holding=("holding_cost", "mean"),
        shortage=("shortage_cost", "mean"),
        fill_rate=("fill_rate_proxy", "mean"),
    )
    sampler_diag = (
        diagnostics[diagnostics["sampler"].isin(["iid_occurrence_size", "contiguous_block"])]
        .groupby(["dataset", "sampler", "method"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            tail_coverage=("tail_coverage", "mean"),
            dispersion=("dispersion", "mean"),
        )
    )
    _write_markdown(
        output / "current_scenario_sampler_definition.md",
        "Current Scenario Sampler Definition",
        [
            (
                "Verified behavior",
                "For each scenario, one donor is selected from the method weights. Conditional on that donor, every lead-time period draws an IID Bernoulli occurrence using the donor's cutoff-visible nonzero rate and, on occurrence, independently samples a positive size with replacement. Donor identity is fixed within a scenario; chronological order, autocorrelation, burst structure, occurrence-size dependence, and lifecycle position are not preserved. Period values are summed to lead-time demand by DecisionScenarioSet.",
            ),
            (
                "Terminology implication",
                "The manuscript should replace `weighted donor trajectories` with `donor-conditioned empirical occurrence-and-size scenarios` or `weighted empirical demand scenarios`.",
            ),
        ],
    )
    _write_markdown(
        output / "scenario_sampler_robustness.md",
        "Temporal-Structure-Preserving Scenario Robustness",
        [
            (
                "Method",
                "Each scenario samples one donor, then a uniformly located contiguous block of length equal to the decision horizon from that donor's cutoff-visible history. Ordering and within-block dependence are preserved. All frozen histories exceed the requested horizon, so no fallback was invoked.",
            ),
            ("Forecast diagnostics", _table(sampler_diag)),
            ("Decision outcomes", _table(sampler_summary)),
        ],
    )

    comparisons = [
        ("uniform_multi", "single_analog"),
        ("forecast_reliability_only", "shuffled_forecast_reliability"),
        ("forecast_combined", "single_analog"),
    ]
    influence_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for dataset_name, group in existing.groupby("dataset"):
        for left, right in comparisons:
            index = [
                "dataset",
                "target_id",
                "cutoff",
                "lead_time_regime",
                "capacity_regime",
                "shortage_holding_ratio",
            ]
            columns = ["oracle_cost_normalized_regret", "raw_regret", "oracle_target_history_cost"]
            selected = group[group["method"].isin([left, right])]
            pivots = {
                column: selected.pivot(index=index, columns="method", values=column)
                for column in columns
            }
            paired = pivots["oracle_cost_normalized_regret"][[left, right]].dropna().reset_index()
            paired["normalized_difference"] = paired[left] - paired[right]
            raw = pivots["raw_regret"][[left, right]].dropna().reset_index()
            paired["raw_difference"] = raw[left] - raw[right]
            reference = selected[selected["method"] == left][index + ["oracle_target_history_cost"]]
            paired = paired.merge(reference, on=index, how="left")
            paired = reference_cost_strata(paired)
            overall = paired["normalized_difference"].mean()
            for stratum, rows in paired.groupby("reference_cost_stratum", observed=True):
                influence_rows.append(
                    {
                        "dataset": dataset_name,
                        "comparison": f"{left} minus {right}",
                        "stratum": str(stratum),
                        "rows": len(rows),
                        "share": len(rows) / len(paired),
                        "mean_raw_difference": rows["raw_difference"].mean(),
                        "mean_normalized_difference": rows["normalized_difference"].mean(),
                        "contribution_to_overall_mean": rows["normalized_difference"].sum()
                        / len(paired),
                        "contribution_fraction": (rows["normalized_difference"].sum() / len(paired))
                        / overall
                        if overall
                        else np.nan,
                        "median": rows["normalized_difference"].median(),
                        "win_rate": (rows["normalized_difference"] < 0).mean(),
                    }
                )
            for excluded_share in (0.0, 0.01, 0.05, 0.10):
                # Stable rank exclusion removes the requested fraction exactly even
                # when many reference costs are tied at zero.
                remove = int(math.floor(len(paired) * excluded_share))
                ranked = paired.sort_values(
                    ["oracle_target_history_cost", "target_id", "cutoff"], kind="stable"
                )
                retained = ranked.iloc[remove:]
                exclusion_rows.append(
                    {
                        "dataset": dataset_name,
                        "comparison": f"{left} minus {right}",
                        "bottom_excluded": excluded_share,
                        "rows": len(retained),
                        "mean_normalized_difference": retained["normalized_difference"].mean(),
                        "mean_raw_difference": retained["raw_difference"].mean(),
                    }
                )
    influence = pd.DataFrame(influence_rows)
    exclusion = pd.DataFrame(exclusion_rows)
    influence.to_csv(output / "reference_cost_strata.csv", index=False)
    exclusion.to_csv(output / "reference_cost_exclusion.csv", index=False)
    existing_metrics = pd.read_csv(
        resolve_repo_path("outputs/targeted_acceptance_extension/paired_metric_robustness.csv")
    )
    _write_markdown(
        output / "regret_influence_analysis.md",
        "Reference-Cost Influence Analysis",
        [
            ("Prespecified strata", _table(influence)),
            ("Bottom-tail exclusion", _table(exclusion)),
            ("Existing alternative metrics", _table(existing_metrics)),
        ],
    )

    deterministic = pd.DataFrame(determinism)
    spec = """MAN uses normalized exact product-group agreement plus log-scaled numeric proximity over lead time, cost price, inventory cost, and MOQ for the secondary score; the primary representation is produced by `man_similarity`. BRAF uses normalized description TF-IDF/cosine similarity plus log-scaled lead-time and price proximity; the primary representation is produced by `braf_similarity`. Eligible donors must satisfy finite demand, positive lead time, at least five pre-earliest-cutoff nonzero periods, sufficient holdout length, complete dataset-specific operational fields, and nonempty group/description. All evaluation targets are excluded from all donor pools. The top 10 candidates are selected by descending metadata similarity with NumPy stable mergesort; original source order is the deterministic tie-break. No similarity threshold is applied. The eligible populations always provide at least 10 donors, so no fallback or stochastic pool completion is used."""
    _write_markdown(
        output / "candidate_construction_spec.md",
        "Candidate Analog Construction Specification",
        [
            ("Technical specification", spec),
            (
                "Determinism check",
                _table(
                    deterministic.groupby("dataset", as_index=False).agg(
                        pools=("deterministic", "size"),
                        deterministic_share=("deterministic", "mean"),
                        candidate_pool_size=("pool_size", "median"),
                    )
                ),
            ),
        ],
    )

    central = existing_metrics[
        (
            existing_metrics["comparison"].isin(
                [
                    "uniform_multi minus single_analog",
                    "forecast_reliability_only minus shuffled_forecast_reliability",
                ]
            )
        )
        & (existing_metrics["metric"] == "oracle_cost_normalized_regret")
    ].copy()
    target_z = central["mean_difference"].abs() / (
        (central["ci_high"] - central["ci_low"]) / (2 * 1.96)
    )
    central["nominal_p"] = 2 * norm.sf(target_z)
    central["holm_p"] = holm_adjust(central["nominal_p"].to_numpy(dtype=float))
    central["holm_survives_0_05"] = central["holm_p"] < 0.05
    central.to_csv(output / "holm_headline_family.csv", index=False)
    scope = "The dataset-level mean gives equal weight to retained target/cutoff/operational-context rows in the frozen experimental grid. This is an experimental-grid estimand, not an estimate of deployment-environment frequencies. Target-clustered bootstrap intervals capture variation over sampled targets while retaining each target's repeated cutoffs and contexts; they do not capture uncertainty over datasets, organizations, or future regime distributions. The two central estimands for interpretation are (1) uniform multi-analog minus single analog and (2) learned forecast reliability minus a distribution-matched shuffled-score control. Other comparisons are secondary or exploratory and are not represented as preregistered."
    _write_markdown(
        output / "inference_scope.md",
        "Inference Scope and Multiplicity Sensitivity",
        [("Estimand scope", scope), ("Holm sensitivity", _table(central))],
    )

    summary = {
        "global_rows": len(global_rows),
        "block_rows": len(block_rows),
        "diagnostic_rows": len(diagnostics),
        "candidate_deterministic": bool(deterministic["deterministic"].all()),
        "outputs": str(output),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
