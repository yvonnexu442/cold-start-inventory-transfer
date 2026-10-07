from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_operational_cost_decomposition.py"
)
SPEC = importlib.util.spec_from_file_location("operational_cost_decomposition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _paired_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"target": "a", "context": 1, "method": "first"},
            {"target": "a", "context": 1, "method": "second"},
            {"target": "a", "context": 2, "method": "first"},
            {"target": "a", "context": 2, "method": "second"},
        ]
    )


def test_complete_pairs_accept_exact_method_key_grid() -> None:
    MODULE._require_complete_pairs(
        _paired_rows(),
        methods=["first", "second"],
        keys=["target", "context"],
        dataset="synthetic",
    )


def test_complete_pairs_reject_missing_counterpart() -> None:
    with pytest.raises(ValueError, match="incomplete method pairs"):
        MODULE._require_complete_pairs(
            _paired_rows().iloc[:-1],
            methods=["first", "second"],
            keys=["target", "context"],
            dataset="synthetic",
        )


def test_complete_pairs_reject_duplicate_method_row() -> None:
    duplicated = pd.concat([_paired_rows(), _paired_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate method rows"):
        MODULE._require_complete_pairs(
            duplicated,
            methods=["first", "second"],
            keys=["target", "context"],
            dataset="synthetic",
        )


def test_cost_components_are_independent_and_zero_action_has_no_fixed_cost() -> None:
    action = pd.Series([0.0, 7.0, 2.0])
    actual = pd.Series([4.0, 3.0, 5.0])
    holding = pd.Series([2.0, 2.0, 2.0])
    ratio = pd.Series([5.0, 5.0, 5.0])
    fixed = pd.Series([3.0, 3.0, 3.0])
    h, s, k = MODULE._cost_components(action, actual, holding, ratio, fixed)
    pd.testing.assert_series_equal(h, pd.Series([0.0, 8.0, 0.0]))
    pd.testing.assert_series_equal(s, pd.Series([40.0, 0.0, 30.0]))
    pd.testing.assert_series_equal(k, pd.Series([0.0, 3.0, 3.0]))
    total = h + s + k
    pd.testing.assert_series_equal(total, pd.Series([40.0, 11.0, 33.0]))
