from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cold_start_replenishment.data.favorita import frozen_item_roles, load_favorita_panel


def test_protocol_hashing_uses_item_identifier_not_row_position() -> None:
    item_ids = np.asarray([9001, 101, 7007])
    roles = frozen_item_roles(item_ids, salt="identifier-test", confirmation_count=1)
    assigned = np.concatenate(list(roles.values()))
    assert set(assigned).issubset(set(item_ids))


def test_daily_net_negative_is_clipped_after_store_aggregation(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "item_nbr": [1, 2],
            "family": ["A", "B"],
            "class": [1, 2],
            "perishable": [0, 1],
        }
    ).to_csv(tmp_path / "items.csv", index=False)
    pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["X", "Y"],
            "state": ["S", "S"],
            "type": ["A", "B"],
            "cluster": [1, 2],
        }
    ).to_csv(tmp_path / "stores.csv", index=False)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2017-01-01", "2017-01-01", "2017-01-02"],
            "store_nbr": [1, 2, 1],
            "item_nbr": [1, 1, 2],
            "unit_sales": [2.0, -5.0, 4.0],
            "onpromotion": [False, False, False],
        }
    ).to_csv(tmp_path / "train.csv", index=False)
    panel = load_favorita_panel(
        tmp_path, calendar_start="2017-01-01", calendar_end="2017-01-03", chunksize=2
    )
    assert panel.quantities.shape == (2, 3)
    assert np.array_equal(panel.quantities[0], [0.0, 0.0, 0.0])
    assert np.array_equal(panel.quantities[1], [0.0, 4.0, 0.0])
    assert panel.negative_diagnostic["negative_net_item_days_before_clipping"] == 1
    assert panel.negative_diagnostic["negative_transaction_rows"] == 1


def test_frozen_roles_are_deterministic_and_disjoint() -> None:
    items = np.arange(1, 2001)
    first = frozen_item_roles(items, salt="test", confirmation_count=100)
    second = frozen_item_roles(items[::-1], salt="test", confirmation_count=100)
    for role in first:
        assert np.array_equal(first[role], second[role])
    assert set(first["development"]).isdisjoint(first["validation"])
    assert set(first["development"]).isdisjoint(first["confirmation"])
    assert set(first["validation"]).isdisjoint(first["confirmation"])
