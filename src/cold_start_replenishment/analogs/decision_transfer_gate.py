"""Decision-cost transfer gate with validation-selected uniform shrinkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]

from cold_start_replenishment.analogs.reliability import RELIABILITY_FEATURES
from cold_start_replenishment.demand.reliability_scenarios import normalized_capped_weights
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level

GATE_FEATURES = (*RELIABILITY_FEATURES, "log_cost_ratio")


def _softmax_negative(values: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    centered = np.asarray(values, dtype=float) - float(np.min(values))
    scale = max(float(np.std(centered)), 1e-6)
    logits = -centered / (scale * temperature)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def shrink_weights(
    learned: NDArray[np.float64], alpha: float, maximum_weight: float
) -> NDArray[np.float64]:
    """Shrink donor mass toward uniform, then enforce the common donor cap."""
    values = np.asarray(learned, dtype=float)
    uniform = np.full(len(values), 1.0 / len(values))
    shrunk = (1.0 - alpha) * values + alpha * uniform
    return normalized_capped_weights(shrunk, maximum_weight)


def stable_cost_weights(
    predicted_costs: NDArray[np.float64],
    temperature: float,
    shrinkage: float,
    maximum_weight: float,
) -> NDArray[np.float64]:
    """Map finite cost scores to a capped simplex in a fixed numerical order."""
    values = np.asarray(predicted_costs, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("predicted donor costs must be a nonempty finite vector")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1]")
    if not 1 / len(values) <= maximum_weight <= 1:
        raise ValueError("maximum_weight is infeasible for the donor count")
    learned = _softmax_negative(values, temperature)
    return shrink_weights(learned, shrinkage, maximum_weight)


def weighted_quantile(
    values: NDArray[np.float64], weights: NDArray[np.float64], probability: float
) -> float:
    """Return the left-continuous weighted empirical quantile."""
    x = np.asarray(values, dtype=float)
    mass = np.asarray(weights, dtype=float)
    if x.ndim != 1 or mass.shape != x.shape or len(x) == 0:
        raise ValueError("values and weights must be aligned nonempty vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("quantile inputs must be finite with nonnegative mass")
    total = float(mass.sum())
    if total <= 0 or not 0 <= probability <= 1:
        raise ValueError("positive mass and probability in [0, 1] are required")
    order = np.argsort(x, kind="mergesort")
    cumulative = np.cumsum(mass[order] / total)
    index = min(int(np.searchsorted(cumulative, probability, side="left")), len(x) - 1)
    return float(x[order][index])


def exact_mixture_newsvendor_action(
    donor_scenarios: NDArray[np.float64],
    donor_weights: NDArray[np.float64],
    shortage_holding_ratio: float,
    *,
    capacity: float | None = None,
    minimum_order_quantity: float | None = None,
    holding_cost: float = 1.0,
    fixed_order_cost: float = 0.0,
) -> float:
    """Solve the empirical newsvendor for a weighted mixture of donor samples."""
    scenarios = np.asarray(donor_scenarios, dtype=float)
    weights = np.asarray(donor_weights, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[0] != len(weights):
        raise ValueError("donor scenarios must have one row per donor weight")
    if shortage_holding_ratio <= 0:
        raise ValueError("shortage_holding_ratio must be positive")
    flat = scenarios.reshape(-1)
    sample_mass = np.repeat(weights / scenarios.shape[1], scenarios.shape[1])
    action, _, _, _ = optimal_feasible_level(
        flat,
        sample_mass,
        holding_cost,
        holding_cost * shortage_holding_ratio,
        capacity=capacity,
        minimum_order_quantity=minimum_order_quantity,
        fixed_order_cost=fixed_order_cost,
    )
    return action


@dataclass
class DecisionTransferGate:
    """Shared donor-cost regressor and task-level softmax mixture gate."""

    folds: int
    random_seed: int
    learner_grid: list[dict[str, int]]
    temperatures: list[float]
    shrinkages: list[float]
    learning_rate: float = 0.05
    max_iter: int = 120
    maximum_weight: float = 0.35

    def __post_init__(self) -> None:
        self.model_: HistGradientBoostingRegressor | None = None
        self.selected_: dict[str, Any] = {}
        self.validation_: pd.DataFrame | None = None

    def _model(self, params: dict[str, int], seed: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )

    @staticmethod
    def _task_loss(
        frame: pd.DataFrame,
        prediction_column: str,
        temperature: float,
        shrinkage: float,
        maximum_weight: float,
    ) -> float:
        losses: list[float] = []
        for _, task in frame.groupby(["pseudo_target_id", "cost_ratio"], sort=False):
            learned = _softmax_negative(
                task[prediction_column].to_numpy(dtype=float), temperature
            )
            weights = shrink_weights(learned, shrinkage, maximum_weight)
            losses.append(float(np.dot(weights, task["relative_cost"].to_numpy(dtype=float))))
        return float(np.mean(losses))

    def fit(self, training: pd.DataFrame) -> DecisionTransferGate:
        required = [*GATE_FEATURES, "relative_cost", "pseudo_target_id", "cost_ratio"]
        missing = [column for column in required if column not in training]
        if missing:
            raise ValueError(f"Missing decision-gate training columns: {missing}")
        x = training.loc[:, GATE_FEATURES].to_numpy(dtype=float)
        y = np.log1p(training["relative_cost"].to_numpy(dtype=float))
        groups = training["pseudo_target_id"].astype(str).to_numpy()
        splits = min(self.folds, len(np.unique(groups)))
        candidates: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for grid_index, params in enumerate(self.learner_grid):
            oof = np.zeros(len(training), dtype=float)
            for fold, (train_idx, val_idx) in enumerate(GroupKFold(splits).split(x, y, groups)):
                model = self._model(params, self.random_seed + grid_index * 100 + fold)
                model.fit(x[train_idx], y[train_idx])
                oof[val_idx] = np.expm1(model.predict(x[val_idx]))
            scored = training.copy()
            scored["oof_predicted_relative_cost"] = np.clip(oof, 0.0, None)
            for temperature in self.temperatures:
                for shrinkage in self.shrinkages:
                    loss = self._task_loss(
                        scored,
                        "oof_predicted_relative_cost",
                        float(temperature),
                        float(shrinkage),
                        self.maximum_weight,
                    )
                    record = {
                        **params,
                        "temperature": float(temperature),
                        "shrinkage": float(shrinkage),
                        "validation_relative_cost": loss,
                    }
                    candidates.append(record)
                    if best is None or loss < float(best["validation_relative_cost"]):
                        best = record
        if best is None:
            raise RuntimeError("No decision-gate candidate was evaluated")
        params = {
            "max_leaf_nodes": int(best["max_leaf_nodes"]),
            "min_samples_leaf": int(best["min_samples_leaf"]),
        }
        self.model_ = self._model(params, self.random_seed + 9000)
        self.model_.fit(x, y)
        self.selected_ = dict(best)
        self.validation_ = pd.DataFrame(candidates)
        return self

    def predict_costs(self, features: pd.DataFrame, cost_ratio: float) -> NDArray[np.float64]:
        if self.model_ is None:
            raise RuntimeError("Fit DecisionTransferGate before prediction")
        frame = features.copy()
        frame["log_cost_ratio"] = np.log(float(cost_ratio))
        values = frame.loc[:, GATE_FEATURES].to_numpy(dtype=float)
        return np.asarray(np.clip(np.expm1(self.model_.predict(values)), 0.0, None), dtype=float)

    def weights(
        self, features: pd.DataFrame, cost_ratio: float, *, shrink: bool = True
    ) -> NDArray[np.float64]:
        predictions = self.predict_costs(features, cost_ratio)
        alpha = float(self.selected_["shrinkage"]) if shrink else 0.0
        return stable_cost_weights(
            predictions,
            float(self.selected_["temperature"]),
            alpha,
            self.maximum_weight,
        )
