from collections.abc import Iterable

import pandas as pd


class SchemaValidationError(ValueError):
    """A tabular input does not satisfy its declared contract."""


def validate_required_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaValidationError(f"{name} is missing required columns: {', '.join(missing)}")


def frame_metadata(frame: pd.DataFrame, date_columns: Iterable[str] = ()) -> dict[str, object]:
    ranges: dict[str, dict[str, str | None]] = {}
    for column in date_columns:
        if column in frame:
            values = pd.to_datetime(frame[column], errors="coerce")
            ranges[column] = {
                "minimum": None if values.isna().all() else str(values.min()),
                "maximum": None if values.isna().all() else str(values.max()),
            }
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "date_ranges": ranges,
    }
