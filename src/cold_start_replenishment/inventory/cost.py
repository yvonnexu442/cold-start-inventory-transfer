from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    holding: float
    shortage: float

    @property
    def total(self) -> float:
        return self.holding + self.shortage


def realized_cost(
    stock: float, demand: float, holding_cost: float, shortage_cost: float
) -> CostBreakdown:
    if min(stock, demand, holding_cost, shortage_cost) < 0:
        raise ValueError("Stock, demand, and costs must be nonnegative")
    return CostBreakdown(
        holding_cost * max(stock - demand, 0), shortage_cost * max(demand - stock, 0)
    )
