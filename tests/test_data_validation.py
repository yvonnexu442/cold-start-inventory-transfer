import pandas as pd
import pytest

from cold_start_replenishment.data.m5_loader import require_file
from cold_start_replenishment.data.validation import (
    SchemaValidationError,
    validate_required_columns,
)


def test_missing_file_reports_path(tmp_path) -> None:
    path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match=str(path)):
        require_file(path)


def test_schema_validation() -> None:
    with pytest.raises(SchemaValidationError, match="demand"):
        validate_required_columns(pd.DataFrame({"id": [1]}), ("id", "demand"), "test")
