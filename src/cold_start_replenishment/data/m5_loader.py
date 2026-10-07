from pathlib import Path

import pandas as pd

from cold_start_replenishment.data.validation import frame_metadata, validate_required_columns
from cold_start_replenishment.utils.logging import get_logger, log_frame_summary

LOGGER = get_logger(__name__)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"M5 file not found: {path}. Download it manually into data/raw/m5/."
        )
    return path


def read_csv(
    path: Path, required_columns: tuple[str, ...] = (), usecols: list[str] | None = None
) -> pd.DataFrame:
    frame = pd.read_csv(require_file(path), usecols=usecols)
    validate_required_columns(frame, required_columns, path.name)
    log_frame_summary(LOGGER, path.name, frame)
    return frame


def dry_run_metadata(
    path: Path, required_columns: tuple[str, ...] = (), date_columns: tuple[str, ...] = ()
) -> dict[str, object]:
    frame = read_csv(path, required_columns)
    return frame_metadata(frame, date_columns)


def inspect_sales(sales: pd.DataFrame) -> dict[str, object]:
    hierarchy = [c for c in ("state_id", "store_id", "cat_id", "dept_id", "item_id") if c in sales]
    demand_columns = [c for c in sales if str(c).startswith("d_")]
    values = sales[demand_columns] if demand_columns else pd.DataFrame(index=sales.index)
    return {
        "hierarchy_coverage": {c: int(sales[c].nunique()) for c in hierarchy},
        "item_store_series_count": len(sales),
        "nonzero_demand_frequency": float((values > 0).to_numpy().mean())
        if not values.empty
        else None,
        "leading_zero_periods": [
            int((row.to_numpy() == 0).cumprod().sum()) for _, row in values.iterrows()
        ]
        if not values.empty
        else [],
        "active_window_lengths": [int((row.to_numpy() > 0).sum()) for _, row in values.iterrows()]
        if not values.empty
        else [],
    }


def inspect_price_coverage(prices: pd.DataFrame) -> dict[str, object]:
    validate_required_columns(prices, ("store_id", "item_id", "wm_yr_wk", "sell_price"), "prices")
    groups = prices.groupby(["store_id", "item_id"], observed=True)
    return {
        "priced_series": int(groups.ngroups),
        "missing_price_periods": int(prices["sell_price"].isna().sum()),
        "price_supported_availability_onsets": int(groups["wm_yr_wk"].min().notna().sum()),
    }


def evaluation_horizon_eligible(active_lengths: pd.Series, minimum: int) -> pd.Series:
    return active_lengths >= minimum
