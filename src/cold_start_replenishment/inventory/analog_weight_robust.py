"""Distributionally robust decisions under uncertainty in donor relevance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cold_start_replenishment.framework.objects import (
    DecisionScenarioSet,
    OperationalContext,
    ReplenishmentDecision,
)


def worst_case_donor_weights(
    donor_costs: NDArray[np.float64],
    nominal_weights: NDArray[np.float64],
    radius: float,
    maximum_weight: float | None = None,
) -> NDArray[np.float64]:
    """Maximize linear donor cost over a total-variation ambiguity set.

    Total variation is one half of the L1 distance. The exact solution moves
    probability mass from the lowest-cost donors to the highest-cost donors,
    respecting the simplex and optional component cap.
    """

    costs = np.asarray(donor_costs, dtype=float)
    nominal = np.asarray(nominal_weights, dtype=float)
    if costs.ndim != 1 or nominal.ndim != 1 or len(costs) != len(nominal) or not len(costs):
        raise ValueError("donor_costs and nominal_weights must be aligned nonempty vectors")
    if not np.all(np.isfinite(costs)) or not np.all(np.isfinite(nominal)):
        raise ValueError("costs and weights must be finite")
    if np.any(nominal < 0) or not np.isclose(nominal.sum(), 1.0):
        raise ValueError("nominal_weights must be nonnegative and sum to one")
    if not 0 <= radius <= 1:
        raise ValueError("radius must lie in [0, 1]")
    cap = 1.0 if maximum_weight is None else float(maximum_weight)
    if cap < 1 / len(nominal) - 1e-12 or cap > 1:
        raise ValueError("maximum_weight is infeasible for the donor count")
    if np.any(nominal > cap + 1e-12):
        raise ValueError("nominal_weights violate maximum_weight")

    result = nominal.copy()
    remaining = float(radius)
    receivers = np.argsort(-costs, kind="stable")
    donors = np.argsort(costs, kind="stable")
    for receiver in receivers:
        if remaining <= 1e-12:
            break
        receiving_capacity = cap - result[receiver]
        if receiving_capacity <= 1e-12:
            continue
        for donor in donors:
            if remaining <= 1e-12 or receiving_capacity <= 1e-12:
                break
            if donor == receiver or costs[donor] >= costs[receiver] - 1e-15:
                continue
            moved = min(remaining, receiving_capacity, result[donor])
            if moved <= 0:
                continue
            result[donor] -= moved
            result[receiver] += moved
            remaining -= moved
            receiving_capacity -= moved
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class AnalogWeightRobustOptimizer:
    """Exact finite robust newsvendor optimizer over donor-weight ambiguity."""

    radius: float
    maximum_donor_weight: float | None = 0.35

    def __post_init__(self) -> None:
        if not 0 <= self.radius <= 1:
            raise ValueError("radius must lie in [0, 1]")

    @staticmethod
    def _donor_arrays(
        scenarios: DecisionScenarioSet,
    ) -> tuple[tuple[str, ...], NDArray[np.float64], list[NDArray[np.int64]]]:
        if scenarios.source_ids is None:
            raise ValueError("Robust optimization requires a source_id for every scenario")
        probabilities = scenarios.scenario_probabilities
        if probabilities is None:
            probabilities = np.full(len(scenarios.values), 1 / len(scenarios.values))
        donor_ids = tuple(dict.fromkeys(scenarios.source_ids))
        source = np.asarray(scenarios.source_ids, dtype=object)
        groups = [np.flatnonzero(source == donor_id) for donor_id in donor_ids]
        nominal = np.asarray([probabilities[group].sum() for group in groups], dtype=float)
        return donor_ids, np.asarray(nominal / nominal.sum(), dtype=np.float64), groups

    @staticmethod
    def _candidate_levels(
        demand: NDArray[np.float64], context: OperationalContext
    ) -> NDArray[np.float64]:
        raw = np.unique(np.concatenate([np.array([0.0]), demand]))
        if context.minimum_order_quantity is not None:
            moq = context.minimum_order_quantity
            raw = np.where(raw > 0, np.ceil(raw / moq) * moq, 0.0)
        if context.capacity is not None:
            raw = np.minimum(raw, context.capacity)
            raw = np.concatenate([raw, np.array([context.capacity])])
        return np.asarray(np.unique(raw), dtype=np.float64)

    def optimize(
        self, scenarios: DecisionScenarioSet, context: OperationalContext
    ) -> ReplenishmentDecision:
        if scenarios.horizon != context.horizon:
            raise ValueError("Scenario and operational horizons must match")
        donor_ids, nominal, groups = self._donor_arrays(scenarios)
        demand = scenarios.lead_time_demand
        probabilities = scenarios.scenario_probabilities
        if probabilities is None:
            probabilities = np.full(len(demand), 1 / len(demand))
        within = [probabilities[group] / probabilities[group].sum() for group in groups]
        candidates = self._candidate_levels(demand, context)
        excess = np.maximum(candidates[:, None] - demand[None, :], 0)
        deficit = np.maximum(demand[None, :] - candidates[:, None], 0)
        holding_matrix = np.column_stack(
            [
                excess[:, group] @ weights * context.holding_cost
                for group, weights in zip(groups, within, strict=True)
            ]
        )
        shortage_matrix = np.column_stack(
            [
                deficit[:, group] @ weights * context.shortage_cost
                for group, weights in zip(groups, within, strict=True)
            ]
        )
        robust_costs = np.empty(len(candidates), dtype=float)
        adversarial: list[NDArray[np.float64]] = []
        for position in range(len(candidates)):
            holding = holding_matrix[position]
            shortage = shortage_matrix[position]
            worst = worst_case_donor_weights(
                holding + shortage, nominal, self.radius, self.maximum_donor_weight
            )
            robust_costs[position] = float(np.dot(worst, holding + shortage))
            adversarial.append(worst)
        selected = int(np.argmin(robust_costs))
        level = float(candidates[selected])
        worst = adversarial[selected]
        expected_holding = float(np.dot(worst, holding_matrix[selected]))
        expected_shortage = float(np.dot(worst, shortage_matrix[selected]))
        return ReplenishmentDecision(
            level,
            expected_holding,
            expected_shortage,
            "analog_weight_robust_newsvendor",
            {
                "ambiguity_radius": self.radius,
                "donor_ids": donor_ids,
                "nominal_donor_weights": nominal.tolist(),
                "worst_case_donor_weights": worst.tolist(),
                "total_variation_used": float(0.5 * np.abs(worst - nominal).sum()),
                "candidate_levels": len(candidates),
                "capacity_bound": context.capacity is not None
                and math.isclose(level, context.capacity),
            },
            context.fixed_order_cost if level > 0 else 0.0,
        )
