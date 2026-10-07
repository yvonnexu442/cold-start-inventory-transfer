from cold_start_replenishment.inventory.cost import realized_cost
from cold_start_replenishment.inventory.metrics import fill_rate_proxy


def test_realized_cost_and_fill_rate() -> None:
    assert realized_cost(8, 5, 2, 10).total == 6
    assert realized_cost(2, 5, 2, 10).total == 30
    assert fill_rate_proxy(2, 4) == 0.5
