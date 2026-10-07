"""Transparent risk-aware base-stock optimization for portability analysis."""

from __future__ import annotations

import math

import numpy as np

from cold_start_replenishment.framework.objects import (
    DecisionScenarioSet,
    OperationalContext,
    ReplenishmentDecision,
)


class RiskAwareNewsvendorOptimizer:
    """Minimize expected scenario cost plus a CVaR penalty.

    This is a supplementary objective-portability test, not an optimization
    novelty claim.
    """

    def __init__(self, alpha: float = 0.9, risk_weight: float = 0.25) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if risk_weight < 0:
            raise ValueError("risk_weight must be nonnegative")
        self.alpha = alpha
        self.risk_weight = risk_weight

    def optimize(
        self, scenarios: DecisionScenarioSet, context: OperationalContext
    ) -> ReplenishmentDecision:
        if scenarios.horizon != context.horizon:
            raise ValueError("Scenario and operational horizons must match")
        demand = scenarios.lead_time_demand
        candidates = np.unique(np.concatenate(([0.0], demand)))
        if context.minimum_order_quantity is not None:
            moq = context.minimum_order_quantity
            candidates = np.unique(np.where(candidates > 0, np.ceil(candidates / moq) * moq, 0.0))
        if context.capacity is not None:
            candidates = np.unique(np.minimum(candidates, context.capacity))

        best: tuple[float, float, float, float, float] | None = None
        tail_count = max(1, math.ceil((1 - self.alpha) * len(demand)))
        fixed = context.fixed_order_cost
        for level in candidates:
            holding = np.maximum(level - demand, 0) * context.holding_cost
            shortage = np.maximum(demand - level, 0) * context.shortage_cost
            scenario_cost = holding + shortage + (fixed if level > 0 else 0.0)
            expected = float(scenario_cost.mean())
            cvar = float(np.sort(scenario_cost)[-tail_count:].mean())
            objective = expected + self.risk_weight * cvar
            candidate = (
                objective,
                float(level),
                float(holding.mean()),
                float(shortage.mean()),
                cvar,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("No feasible base-stock candidates")
        objective, level, holding, shortage, cvar = best
        return ReplenishmentDecision(
            level,
            holding,
            shortage,
            "risk_aware_newsvendor",
            {
                "alpha": self.alpha,
                "risk_weight": self.risk_weight,
                "scenario_cvar": cvar,
                "risk_objective": objective,
                "capacity_bound": bool(context.capacity is not None and level >= context.capacity),
                "moq_bound": bool(
                    context.minimum_order_quantity is not None
                    and level > 0
                    and np.isclose(level % context.minimum_order_quantity, 0)
                ),
            },
            context.fixed_order_cost if level > 0 else 0.0,
        )
