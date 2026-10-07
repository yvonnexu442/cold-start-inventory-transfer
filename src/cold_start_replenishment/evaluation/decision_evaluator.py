"""Held-out evaluation implementation, isolated from upstream modules."""

from __future__ import annotations

from cold_start_replenishment.framework.objects import (
    DecisionEvaluation,
    OperationalContext,
    ReplenishmentDecision,
)
from cold_start_replenishment.inventory.cost import realized_cost


class HeldOutDecisionEvaluator:
    """Evaluate operational consequences after a decision is frozen."""

    def evaluate(
        self,
        decision: ReplenishmentDecision,
        realized_demand: float,
        context: OperationalContext,
        reference_decision: ReplenishmentDecision | None = None,
    ) -> DecisionEvaluation:
        if realized_demand < 0:
            raise ValueError("realized_demand must be nonnegative")
        cost = realized_cost(
            decision.level, realized_demand, context.holding_cost, context.shortage_cost
        )
        fixed_cost = context.fixed_order_cost if decision.level > 0 else 0.0
        total_cost = cost.total + fixed_cost
        fill_rate = (
            1.0 if realized_demand == 0 else min(decision.level, realized_demand) / realized_demand
        )
        regret: float | None = None
        normalized_regret: float | None = None
        if reference_decision is not None:
            reference_cost = realized_cost(
                reference_decision.level,
                realized_demand,
                context.holding_cost,
                context.shortage_cost,
            ).total + (context.fixed_order_cost if reference_decision.level > 0 else 0.0)
            regret = total_cost - reference_cost
            normalized_regret = regret / max(reference_cost, 1.0)
        return DecisionEvaluation(
            realized_demand,
            cost.holding,
            cost.shortage,
            total_cost,
            float(fill_rate),
            regret,
            normalized_regret,
            {"decision_source": decision.source_module, "fixed_order_cost": fixed_cost},
        )
