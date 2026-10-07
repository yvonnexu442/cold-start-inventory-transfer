from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plot_evidence_increment_synthesis.py"
SPEC = importlib.util.spec_from_file_location("evidence_increment_synthesis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def toy_rows() -> pd.DataFrame:
    records = []
    for target, first, comparator in (
        ("a", [8.0, 10.0], [10.0, 10.0]),
        ("b", [18.0, 20.0], [20.0, 20.0]),
    ):
        for context, (first_cost, comparator_cost) in enumerate(
            zip(first, comparator, strict=True)
        ):
            records.extend(
                [
                    {
                        "target": target,
                        "context": context,
                        "method": "first",
                        "cost": first_cost,
                        "actual": 3.0 + context,
                    },
                    {
                        "target": target,
                        "context": context,
                        "method": "comparator",
                        "cost": comparator_cost,
                        "actual": 3.0 + context,
                    },
                ]
            )
    return pd.DataFrame(records)


def test_pairing_and_sign_convention() -> None:
    paired = MODULE.validate_and_pair(
        toy_rows(),
        key_columns=["target", "context"],
        method_column="method",
        cost_column="cost",
        actual_column="actual",
        first_method="first",
        comparator_method="comparator",
    )
    assert len(paired) == 4
    assert (
        MODULE.relative_difference(paired.first_cost.to_numpy(), paired.comparator_cost.to_numpy())
        < 0
    )


def test_duplicate_and_unmatched_rows_fail() -> None:
    rows = toy_rows()
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.validate_and_pair(
            duplicate,
            key_columns=["target", "context"],
            method_column="method",
            cost_column="cost",
            actual_column="actual",
            first_method="first",
            comparator_method="comparator",
        )
    unmatched = rows.drop(index=0)
    with pytest.raises(ValueError, match="Unmatched"):
        MODULE.validate_and_pair(
            unmatched,
            key_columns=["target", "context"],
            method_column="method",
            cost_column="cost",
            actual_column="actual",
            first_method="first",
            comparator_method="comparator",
        )


def test_clustered_bootstrap_is_deterministic_and_retains_contexts() -> None:
    paired = MODULE.validate_and_pair(
        toy_rows(),
        key_columns=["target", "context"],
        method_column="method",
        cost_column="cost",
        actual_column="actual",
        first_method="first",
        comparator_method="comparator",
    )
    point_a, draws_a = MODULE.clustered_bootstrap(paired, product_column="target", draws=50, seed=7)
    point_b, draws_b = MODULE.clustered_bootstrap(paired, product_column="target", draws=50, seed=7)
    assert point_a == point_b
    np.testing.assert_array_equal(draws_a, draws_b)
    expected = (
        100 * (np.mean([8, 10, 18, 20]) - np.mean([10, 10, 20, 20])) / np.mean([10, 10, 20, 20])
    )
    assert point_a == pytest.approx(expected)


def test_frozen_configuration_has_three_panels_and_four_datasets() -> None:
    config = MODULE.json.loads(MODULE.DEFAULT_CONFIG.read_text())
    assert config["panel_order"] == ["A", "B", "C"]
    assert config["dataset_order"] == ["BRAF", "MAN", "UCI", "Favorita"]
    assert config["bootstrap_draws"] == 5000
    compact = MODULE.ROOT / config["compact_source"]
    assert compact.exists()
    tracked = MODULE.ROOT / ".git"
    assert tracked.exists()
    rows = pd.read_csv(compact)
    assert set(rows.dataset) == {"BRAF", "MAN", "UCI", "Favorita"}
    assert set(rows.policy) == {
        "complete_similarity",
        "single_donor",
        "component_specific_transfer",
        "matched_shared_relation",
    }


def test_compact_pairing_and_product_sum_bootstrap() -> None:
    compact = pd.DataFrame(
        {
            "dataset": ["Toy"] * 4,
            "redacted_product_id": ["a", "a", "b", "b"],
            "bootstrap_order": [0, 0, 1, 1],
            "policy": ["first", "comparator", "first", "comparator"],
            "cost_sum": [18.0, 20.0, 38.0, 40.0],
            "context_row_count": [2, 2, 2, 2],
        }
    )
    paired = MODULE.load_compact_pairs(
        compact, dataset="Toy", first_policy="first", comparator="comparator"
    )
    point_a, draws_a = MODULE.clustered_bootstrap_product_sums(
        paired, draws=50, seed=7
    )
    point_b, draws_b = MODULE.clustered_bootstrap_product_sums(
        paired, draws=50, seed=7
    )
    assert point_a == pytest.approx(-6.6666666667)
    np.testing.assert_array_equal(draws_a, draws_b)


def test_default_compact_source_is_git_tracked() -> None:
    import subprocess

    config = MODULE.json.loads(MODULE.DEFAULT_CONFIG.read_text())
    source = config["compact_source"]
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", source],
        cwd=MODULE.ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
