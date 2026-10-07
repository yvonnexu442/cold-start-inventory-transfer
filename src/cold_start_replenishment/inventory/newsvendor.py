"""Sprint 1 implementation of the Decision Optimization responsibility."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cold_start_replenishment.framework.objects import (
    DecisionScenarioSet,
    OperationalContext,
    ReplenishmentDecision,
)


def optimal_feasible_level(
    demand: NDArray[np.float64],
    probabilities: NDArray[np.float64] | None,
    holding_cost: float,
    shortage_cost: float,
    *,
    capacity: float | None = None,
    minimum_order_quantity: float | None = None,
    fixed_order_cost: float = 0.0,
) -> tuple[float, float, float, float]:
    """Minimize empirical newsvendor cost on the actual feasible set.

    The source field is a minimum positive order quantity, not a batch
    multiple.  Hence Q={0} union [m,U] (or [0,U] without an MOQ).  The
    piecewise-linear objective attains an optimum at zero, an admissible
    demand atom, the MOQ boundary, or the capacity boundary.
    """
    values = np.asarray(demand, dtype=float).reshape(-1)
    if len(values) == 0 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("demand must be a nonempty finite nonnegative vector")
    if holding_cost <= 0 or shortage_cost <= 0 or fixed_order_cost < 0:
        raise ValueError("costs must be positive, with nonnegative fixed cost")
    if capacity is not None and capacity < 0:
        raise ValueError("capacity must be nonnegative")
    if minimum_order_quantity is not None and minimum_order_quantity <= 0:
        raise ValueError("minimum_order_quantity must be positive")
    if probabilities is None:
        mass = np.full(len(values), 1.0 / len(values))
    else:
        mass = np.asarray(probabilities, dtype=float).reshape(-1)
        if mass.shape != values.shape or np.any(mass < 0) or not np.all(np.isfinite(mass)):
            raise ValueError("probabilities must align and be finite nonnegative")
        if float(mass.sum()) <= 0:
            raise ValueError("probabilities must have positive mass")
        mass = mass / mass.sum()
    upper = np.inf if capacity is None else float(capacity)
    lower = 0.0 if minimum_order_quantity is None else float(minimum_order_quantity)
    # The positive-order objective is convex.  Its unconstrained smallest
    # minimizer is the weighted critical quantile, and projection onto the
    # interval [m,U] is therefore exact.  Fixed order cost is constant on the
    # positive branch, so only that projected minimizer and q=0 need comparing.
    feasible = np.asarray([0.0])
    if upper > 0 and (minimum_order_quantity is None or upper >= lower):
        order = np.argsort(values, kind="stable")
        ordered_values, ordered_mass = values[order], mass[order]
        critical = shortage_cost / (holding_cost + shortage_cost)
        index = min(int(np.searchsorted(np.cumsum(ordered_mass), critical, side="left")), len(values) - 1)
        positive = max(float(ordered_values[index]), lower)
        positive = min(positive, upper)
        feasible = np.unique(np.asarray([0.0, positive], dtype=float))
    q = feasible[:, None]
    holding = ((q - values).clip(min=0) * mass).sum(axis=1) * holding_cost
    shortage = ((values - q).clip(min=0) * mass).sum(axis=1) * shortage_cost
    fixed = (feasible > 0).astype(float) * fixed_order_cost
    objective = holding + shortage + fixed
    # Smallest minimizer is stable and avoids unnecessary stock on flat faces.
    index = int(np.flatnonzero(np.isclose(objective, objective.min(), rtol=1e-12, atol=1e-12))[0])
    return (
        float(feasible[index]),
        float(holding[index]),
        float(shortage[index]),
        float(fixed[index]),
    )


class NewsvendorOptimizer:
    """Transparent single-period optimizer behind a replaceable interface."""

    def optimize(
        self, scenarios: DecisionScenarioSet, context: OperationalContext
    ) -> ReplenishmentDecision:
        if scenarios.horizon != context.horizon:
            raise ValueError("Scenario and operational horizons must match")
        demand = scenarios.lead_time_demand
        level, holding, shortage, fixed = optimal_feasible_level(
            demand,
            scenarios.scenario_probabilities,
            context.holding_cost,
            context.shortage_cost,
            capacity=context.capacity,
            minimum_order_quantity=context.minimum_order_quantity,
            fixed_order_cost=context.fixed_order_cost,
        )
        moq_bound = bool(context.minimum_order_quantity is not None and level > 0 and level <= context.minimum_order_quantity)
        capacity_bound = bool(context.capacity is not None and np.isclose(level, context.capacity))
        return ReplenishmentDecision(
            level,
            holding,
            shortage,
            "newsvendor",
            {
                "critical_ratio": context.shortage_cost
                / (context.holding_cost + context.shortage_cost),
                "moq_bound": moq_bound,
                "capacity_bound": capacity_bound,
            },
            fixed,
        )


class ConservativeNewsvendorOptimizer(NewsvendorOptimizer):
    """Supplementary risk-aware baseline using a transparent quantile increment."""

    def __init__(self, quantile_increment: float = 0.05) -> None:
        if not 0 <= quantile_increment <= 0.25:
            raise ValueError("quantile_increment must lie in [0, 0.25]")
        self.quantile_increment = quantile_increment

    def optimize(
        self, scenarios: DecisionScenarioSet, context: OperationalContext
    ) -> ReplenishmentDecision:
        critical = context.shortage_cost / (context.holding_cost + context.shortage_cost)
        adjusted = min(critical + self.quantile_increment, 1.0)
        adjusted_context = OperationalContext(
            context.target_id,
            context.horizon,
            context.holding_cost,
            context.shortage_cost,
            context.capacity,
            adjusted,
            context.minimum_order_quantity,
            context.fixed_order_cost,
        )
        decision = super().optimize(scenarios, adjusted_context)
        diagnostics = {
            **decision.diagnostics,
            "conservative_quantile_increment": self.quantile_increment,
        }
        return ReplenishmentDecision(
            decision.level,
            decision.expected_holding_cost,
            decision.expected_shortage_cost,
            "conservative_newsvendor",
            diagnostics,
            decision.expected_fixed_order_cost,
        )
