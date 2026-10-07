"""Low-dimensional direct learning of donor-mixture replenishment policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cold_start_replenishment.analogs.decision_transfer_gate import (
    exact_mixture_newsvendor_action,
)


@dataclass(frozen=True)
class MixtureDecisionTask:
    """One historical pseudo-cold-start decision with deployment-valid inputs."""

    target_id: str
    donor_features: NDArray[np.float64]
    donor_scenarios: NDArray[np.float64]
    actual_demand: float
    cost_ratio: float
    lead_time: int
    capacity: float | None
    minimum_order_quantity: float | None
    holding_cost: float = 1.0
    fixed_order_cost: float = 0.0
    # Per donor: [visible periods, positive visible periods, effective horizon windows,
    # simulation atoms].  This separates historical evidence from numerical
    # approximation size.  Legacy callers may omit it, but formal factorized
    # experiments provide it.
    donor_support: NDArray[np.float64] | None = None
    # Optional deployable target-donor relation features.  These are separate
    # from outcome summaries so a relation learner can be compared with the
    # aggregate-similarity policy without receiving latent or future data.
    donor_relations: NDArray[np.float64] | None = None


def _standardize_within_pool(values: NDArray[np.float64]) -> NDArray[np.float64]:
    center = values - np.mean(values, axis=0, keepdims=True)
    scale = np.std(center, axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return center / scale


def direct_mixture_weights(
    donor_features: NDArray[np.float64],
    parameters: NDArray[np.float64],
    *,
    cost_ratio: float,
    lead_time: int,
    capacity: float | None,
) -> NDArray[np.float64]:
    """Generate simplex weights and task-conditioned contraction to uniform.

    Four donor features are used: similarity, nonzero rate, log support, and
    expected lead demand.  Context interactions expose the cost ratio, lead
    time, and capacity to the policy.  The final two parameters control a
    contraction whose signal is the predecision dispersion of donor scores.
    """
    features = np.asarray(donor_features, dtype=float)
    theta = np.asarray(parameters, dtype=float)
    if features.ndim != 2 or features.shape[1] != 4 or theta.shape != (9,):
        raise ValueError("expected Kx4 donor features and nine policy parameters")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(theta)):
        raise ValueError("mixture-policy inputs must be finite")
    z = _standardize_within_pool(features)
    log_ratio = float(np.log(max(cost_ratio, 1e-8)))
    log_lead = float(np.log1p(max(lead_time, 1)))
    expected_scale = float(np.mean(np.maximum(features[:, 3], 0.0)))
    capacity_ratio = 1.0 if capacity is None else float(capacity / max(expected_scale, 1.0))
    score = (
        z @ theta[:4]
        + theta[4] * z[:, 3] * log_ratio
        + theta[5] * z[:, 0] * log_lead
        + theta[6] * z[:, 3] * np.clip(capacity_ratio, 0.0, 5.0)
    )
    score -= float(np.max(score))
    learned = np.exp(np.clip(score, -40.0, 0.0))
    learned /= learned.sum()
    dispersion = float(np.std(score))
    logit = float(np.clip(theta[7] + theta[8] * dispersion, -20.0, 20.0))
    contraction = 1.0 / (1.0 + np.exp(-logit))
    uniform = np.full(len(learned), 1.0 / len(learned))
    weights = (1.0 - contraction) * learned + contraction * uniform
    return np.asarray(weights / weights.sum(), dtype=float)


def exact_task_cost(task: MixtureDecisionTask, parameters: NDArray[np.float64]) -> float:
    weights = direct_mixture_weights(
        task.donor_features,
        parameters,
        cost_ratio=task.cost_ratio,
        lead_time=task.lead_time,
        capacity=task.capacity,
    )
    action = exact_mixture_newsvendor_action(
        task.donor_scenarios,
        weights,
        task.cost_ratio,
        capacity=task.capacity,
        minimum_order_quantity=task.minimum_order_quantity,
        holding_cost=task.holding_cost,
        fixed_order_cost=task.fixed_order_cost,
    )
    return float(
        task.holding_cost * max(action - task.actual_demand, 0.0)
        + task.holding_cost * task.cost_ratio * max(task.actual_demand - action, 0.0)
        + (task.fixed_order_cost if action > 0 else 0.0)
    )


@dataclass
class DirectMixturePolicy:
    """Finite, reproducible search against exact historical decision cost."""

    random_seed: int
    candidate_count: int = 96
    regularization: float = 1e-3

    def __post_init__(self) -> None:
        self.parameters_: NDArray[np.float64] | None = None
        self.search_: list[dict[str, float | int]] = []

    def candidate_parameters(self) -> NDArray[np.float64]:
        rng = np.random.default_rng(self.random_seed)
        candidates = rng.normal(0.0, 1.0, size=(self.candidate_count, 9))
        candidates[:, 7] -= 1.5  # begin with modest, not forced, contraction
        anchors = np.zeros((3, 9), dtype=float)
        anchors[0, 7] = 20.0  # uniform policy
        anchors[1, 0] = 1.0
        anchors[1, 7] = -20.0  # similarity-only policy
        anchors[2, 3] = -1.0
        anchors[2, 7] = -20.0  # prefer smaller expected demand
        return np.vstack([anchors, candidates])

    def fit(
        self,
        training_tasks: list[MixtureDecisionTask],
        validation_tasks: list[MixtureDecisionTask],
    ) -> DirectMixturePolicy:
        if not training_tasks or not validation_tasks:
            raise ValueError("nonempty training and validation tasks are required")
        records: list[dict[str, float | int]] = []
        candidates = self.candidate_parameters()
        for index, parameters in enumerate(candidates):
            train_cost = np.mean(
                [
                    exact_task_cost(task, parameters) / max(task.actual_demand, 1.0)
                    for task in training_tasks
                ]
            )
            objective = float(train_cost + self.regularization * np.mean(parameters**2))
            records.append({"candidate": index, "training_objective": objective})
        shortlist = sorted(records, key=lambda row: float(row["training_objective"]))[:12]
        best_index = -1
        best_validation = np.inf
        for row in shortlist:
            index = int(row["candidate"])
            validation = float(
                np.mean(
                    [
                        exact_task_cost(task, candidates[index]) / max(task.actual_demand, 1.0)
                        for task in validation_tasks
                    ]
                )
            )
            row["validation_cost"] = validation
            if validation < best_validation:
                best_validation = validation
                best_index = index
        self.parameters_ = candidates[best_index].copy()
        for row in records:
            row["selected"] = int(row["candidate"]) == best_index
        self.search_ = records
        return self

    def weights(self, task: MixtureDecisionTask) -> NDArray[np.float64]:
        if self.parameters_ is None:
            raise RuntimeError("fit the direct mixture policy before prediction")
        return direct_mixture_weights(
            task.donor_features,
            self.parameters_,
            cost_ratio=task.cost_ratio,
            lead_time=task.lead_time,
            capacity=task.capacity,
        )
