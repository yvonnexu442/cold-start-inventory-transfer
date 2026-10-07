"""Adapter for the UCI Online Retail II workbook.

The weekly panel represents recorded purchase quantities. A blank product-week
is encoded as zero recorded sales, not as proof that the item was available or
that latent demand was zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OnlineRetailIIPanel:
    weeks: pd.DatetimeIndex
    product_ids: np.ndarray
    descriptions: np.ndarray
    quantities: np.ndarray
    product_table: pd.DataFrame
    audit: dict[str, int | str]


def _normalise_code(value: object) -> str:
    return str(value).strip().upper()


def load_online_retail_ii(
    workbook: str | Path,
    *,
    excluded_codes: tuple[str, ...] = (
        "POST", "DOT", "M", "BANK_CHARGES", "AMAZONFEE", "CRUK", "D", "S",
        "ADJUST", "ADJUST2", "C2",
    ),
) -> OnlineRetailIIPanel:
    """Load, clean, and weekly-aggregate both workbook sheets."""
    path = Path(workbook)
    frames = pd.read_excel(path, sheet_name=None)
    raw = pd.concat(
        [frame.assign(source_sheet=name) for name, frame in frames.items()],
        ignore_index=True,
    )
    raw_rows = len(raw)
    data = raw.drop_duplicates().copy()
    duplicate_rows = raw_rows - len(data)
    data["StockCode"] = data["StockCode"].map(_normalise_code)
    invoice = data["Invoice"].astype(str).str.strip().str.upper()
    cancelled = invoice.str.startswith("C")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    valid_description = data["Description"].notna() & data["Description"].astype(str).str.strip().ne("")
    product_code = data["StockCode"].str.contains(r"\d", regex=True, na=False)
    valid = (
        ~cancelled
        & data["Quantity"].gt(0)
        & data["Price"].gt(0)
        & data["InvoiceDate"].notna()
        & valid_description
        & product_code
        & ~data["StockCode"].isin(excluded_codes)
    )
    clean = data.loc[valid].copy()
    clean["Description"] = clean["Description"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    start = clean["InvoiceDate"].min().normalize()
    clean["week"] = ((clean["InvoiceDate"].dt.normalize() - start).dt.days // 7).astype(int)
    earliest = (
        clean.sort_values(["InvoiceDate", "StockCode"], kind="stable")
        .drop_duplicates("StockCode")
        .set_index("StockCode")["Description"]
    )
    activity = clean.groupby("StockCode").agg(
        first_week=("week", "min"), last_week=("week", "max"),
        transaction_rows=("Quantity", "size"), total_quantity=("Quantity", "sum"),
    )
    products = activity.join(earliest.rename("description"), how="inner").sort_index()
    weeks = pd.date_range(start, periods=int(clean["week"].max()) + 1, freq="7D")
    grouped = clean.groupby(["StockCode", "week"], sort=False)["Quantity"].sum()
    matrix = np.zeros((len(products), len(weeks)), dtype=float)
    row = {code: index for index, code in enumerate(products.index)}
    for (code, week), quantity in grouped.items():
        matrix[row[code], int(week)] = float(quantity)
    audit = {
        "raw_rows": raw_rows,
        "exact_duplicate_rows_removed": duplicate_rows,
        "cancelled_rows_excluded": int(cancelled.sum()),
        "clean_transaction_rows": len(clean),
        "products": len(products),
        "weeks": len(weeks),
        "start_date": str(weeks.min().date()),
        "end_date": str(weeks.max().date()),
    }
    return OnlineRetailIIPanel(
        weeks, products.index.to_numpy(str), products["description"].to_numpy(str),
        matrix, products.reset_index(names="product_id"), audit,
    )


def frozen_product_split(
    panel: OnlineRetailIIPanel,
    *,
    seed: int,
    first_week_at_most: int,
    last_week_at_least: int,
    development_products: int,
    validation_products: int,
    test_products: int,
) -> pd.DataFrame:
    """Mechanically assign products before any predictive evaluation."""
    table = panel.product_table
    eligible = table.loc[
        table.first_week.le(first_week_at_most) & table.last_week.ge(last_week_at_least)
    ].copy()
    eligible["split_key"] = eligible.product_id.map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    eligible = eligible.sort_values(["split_key", "product_id"], kind="stable")
    need = development_products + validation_products + test_products
    if len(eligible) < need:
        raise ValueError(f"eligible cohort has {len(eligible)} products; {need} requested")
    eligible = eligible.iloc[:need].copy()
    eligible["role"] = (
        ["development"] * development_products
        + ["validation"] * validation_products
        + ["external_test"] * test_products
    )
    return eligible[["product_id", "role", "first_week", "last_week", "transaction_rows", "split_key"]]
