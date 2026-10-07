"""Frozen adapter for the Corporacion Favorita confirmation population."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FavoritaPanel:
    item_ids: np.ndarray
    dates: pd.DatetimeIndex
    quantities: np.ndarray
    metadata: pd.DataFrame
    store_diagnostic: pd.DataFrame
    negative_diagnostic: dict[str, float | int]


def stable_item_hash(item_nbr: int, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}|{int(item_nbr)}".encode()).hexdigest()
    return int(digest, 16)


def frozen_item_roles(
    item_ids: np.ndarray,
    *,
    salt: str,
    confirmation_count: int,
) -> dict[str, np.ndarray]:
    records = []
    for item in map(int, item_ids):
        value = stable_item_hash(item, salt)
        bucket = value % 100
        role = "development" if bucket <= 69 else ("validation" if bucket <= 79 else "confirmation")
        records.append((item, role, value))
    frame = pd.DataFrame(records, columns=["item_nbr", "role", "hash_value"])
    confirmation = (
        frame[frame.role.eq("confirmation")]
        .sort_values(["hash_value", "item_nbr"])
        .head(int(confirmation_count))
    )
    if len(confirmation) < int(confirmation_count):
        raise ValueError("fewer schema-eligible confirmation items than frozen target count")
    return {
        "development": np.sort(
            frame.loc[frame.role.eq("development"), "item_nbr"].to_numpy(int)
        ),
        "validation": np.sort(
            frame.loc[frame.role.eq("validation"), "item_nbr"].to_numpy(int)
        ),
        "confirmation": confirmation.item_nbr.to_numpy(int),
    }


def load_favorita_panel(
    raw_dir: Path,
    *,
    calendar_start: str,
    calendar_end: str,
    chunksize: int = 2_000_000,
) -> FavoritaPanel:
    """Aggregate stores by item/day, clip net negatives, then add calendar zeros."""
    items_path = raw_dir / "items.csv"
    stores_path = raw_dir / "stores.csv"
    train_path = raw_dir / "train.csv"
    items = pd.read_csv(items_path)
    stores = pd.read_csv(stores_path)
    required_items = {"item_nbr", "family", "class", "perishable"}
    required_stores = {"store_nbr", "city", "state", "type", "cluster"}
    if not required_items.issubset(items.columns):
        raise ValueError("items.csv does not match frozen schema")
    if not required_stores.issubset(stores.columns):
        raise ValueError("stores.csv does not match frozen schema")
    if items.item_nbr.duplicated().any():
        raise ValueError("duplicate item metadata")

    start = pd.Timestamp(calendar_start)
    end = pd.Timestamp(calendar_end)
    start_text = start.strftime("%Y-%m-%d")
    end_text = end.strftime("%Y-%m-%d")
    aggregates: list[pd.DataFrame] = []
    store_pairs: list[pd.DataFrame] = []
    transaction_rows = 0
    negative_transaction_rows = 0
    negative_transaction_units = 0.0
    columns = ["date", "store_nbr", "item_nbr", "unit_sales"]
    dtypes = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}
    for chunk in pd.read_csv(train_path, usecols=columns, dtype=dtypes, chunksize=chunksize):
        mask = chunk.date.between(start_text, end_text)
        chunk = chunk.loc[mask]
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk.date, format="%Y-%m-%d")
        transaction_rows += len(chunk)
        negative = chunk.unit_sales.lt(0)
        negative_transaction_rows += int(negative.sum())
        negative_transaction_units += float(-chunk.loc[negative, "unit_sales"].sum())
        aggregates.append(
            chunk.groupby(["item_nbr", "date"], as_index=False, sort=False).unit_sales.sum()
        )
        store_pairs.append(chunk[["item_nbr", "store_nbr"]].drop_duplicates())

    if not aggregates:
        raise ValueError("no transactions in frozen observation calendar")
    daily = (
        pd.concat(aggregates, ignore_index=True)
        .groupby(["item_nbr", "date"], as_index=False, sort=False)
        .unit_sales.sum()
    )
    observed_item_days = len(daily)
    negative_daily = daily.unit_sales.lt(0)
    affected_daily = int(negative_daily.sum())
    net_units_clipped = float(-daily.loc[negative_daily, "unit_sales"].sum())
    daily.loc[negative_daily, "unit_sales"] = 0.0

    metadata = items[["item_nbr", "family", "class", "perishable"]].copy()
    metadata["family"] = metadata.family.fillna("__MISSING__").astype(str)
    metadata["class"] = metadata["class"].fillna(-1).astype(int).astype(str)
    metadata["perishable"] = metadata.perishable.fillna(-1).astype(int).astype(str)
    item_ids = np.sort(metadata.item_nbr.to_numpy(int))
    dates = pd.date_range(start, end, freq="D")
    item_index = pd.Index(item_ids, name="item_nbr")
    pivot = daily.pivot(index="item_nbr", columns="date", values="unit_sales")
    pivot = pivot.reindex(index=item_index, columns=dates, fill_value=0.0).fillna(0.0)
    quantities = pivot.to_numpy(dtype=np.float32)

    pairs = pd.concat(store_pairs, ignore_index=True).drop_duplicates()
    store_diagnostic = (
        pairs.groupby("item_nbr", as_index=False)
        .store_nbr.nunique()
        .rename(columns={"store_nbr": "observed_store_count"})
        .merge(metadata[["item_nbr"]], how="right", on="item_nbr")
        .fillna({"observed_store_count": 0})
    )
    all_calendar_item_days = len(item_ids) * len(dates)
    diagnostic: dict[str, float | int] = {
        "transaction_rows": int(transaction_rows),
        "negative_transaction_rows": int(negative_transaction_rows),
        "negative_transaction_row_fraction": float(
            negative_transaction_rows / transaction_rows
        ),
        "negative_transaction_units_absolute": float(negative_transaction_units),
        "observed_item_days_before_calendar_fill": int(observed_item_days),
        "negative_net_item_days_before_clipping": int(affected_daily),
        "negative_net_observed_item_day_fraction": float(
            affected_daily / observed_item_days
        ),
        "all_calendar_item_days": int(all_calendar_item_days),
        "negative_net_all_calendar_item_day_fraction": float(
            affected_daily / all_calendar_item_days
        ),
        "net_units_clipped": net_units_clipped,
        "explicit_zero_item_days": int(np.count_nonzero(quantities == 0)),
        "explicit_zero_item_day_fraction": float(np.mean(quantities == 0)),
    }
    return FavoritaPanel(
        item_ids=item_ids,
        dates=dates,
        quantities=quantities,
        metadata=metadata.set_index("item_nbr").loc[item_ids].reset_index(),
        store_diagnostic=store_diagnostic,
        negative_diagnostic=diagnostic,
    )
