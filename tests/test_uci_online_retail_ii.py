from __future__ import annotations

import numpy as np
import pandas as pd

from cold_start_replenishment.data.uci_online_retail_ii import OnlineRetailIIPanel, frozen_product_split


def test_frozen_split_is_deterministic_and_disjoint() -> None:
    products = pd.DataFrame({
        "product_id": [f"P{i}" for i in range(12)],
        "first_week": [0] * 12,
        "last_week": [10] * 12,
        "transaction_rows": [3] * 12,
        "total_quantity": [5.0] * 12,
        "description": [f"item {i}" for i in range(12)],
    })
    panel = OnlineRetailIIPanel(
        pd.date_range("2020-01-01", periods=11, freq="7D"),
        products.product_id.to_numpy(), products.description.to_numpy(),
        np.zeros((12, 11)), products, {},
    )
    one = frozen_product_split(panel, seed=7, first_week_at_most=0, last_week_at_least=10,
                               development_products=5, validation_products=3, test_products=4)
    two = frozen_product_split(panel, seed=7, first_week_at_most=0, last_week_at_least=10,
                               development_products=5, validation_products=3, test_products=4)
    pd.testing.assert_frame_equal(one, two)
    assert one.product_id.nunique() == 12
    assert one.groupby("role").product_id.nunique().sum() == 12
