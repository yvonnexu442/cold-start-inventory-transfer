def fill_rate_proxy(stock: float, demand: float) -> float:
    if stock < 0 or demand < 0:
        raise ValueError("Stock and demand must be nonnegative")
    return 1.0 if demand == 0 else min(stock, demand) / demand


def normalized_regret(cost: float, oracle_cost: float, epsilon: float = 1e-12) -> float:
    return (cost - oracle_cost) / max(abs(oracle_cost), epsilon)
