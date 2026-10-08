from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_ai_darld_v3_corrected import KEYS, _validated
from scripts.analyze_support_v3_similarity_residual import _comparison


def _rows(method: str, offset: float = 0.0) -> pd.DataFrame:
    records = []
    for target in ("a", "b"):
        for cutoff in (10, 20):
            records.append(
                {
                    "dataset": "MAN",
                    "target_id": target,
                    "cutoff": cutoff,
                    "lead_time_regime": "native_capped",
                    "capacity_regime": "unconstrained",
                    "shortage_holding_ratio": 5.0,
                    "method": method,
                    "actual_demand": float(cutoff),
                    "selected_level": 1.0,
                    "total_cost": float(cutoff) + offset,
                    "fill_rate_proxy": 0.5,
                }
            )
    return pd.DataFrame(records)


def test_authority_validation_rejects_duplicate_complete_keys() -> None:
    frame = _rows("factorized_similarity_residual")
    with pytest.raises(ValueError, match="duplicate evaluation keys"):
        _validated(pd.concat([frame, frame.iloc[[0]]]), "test")


def test_authority_comparison_requires_matched_demand_and_keys() -> None:
    reference = _rows("factorized_similarity_residual")
    comparator = _rows("similarity_complete", 1.0)
    result = _comparison(reference, comparator)
    assert result["matched_rows"] == 4
    assert result["target_clusters"] == 2
    assert result["difference"] == pytest.approx(-1.0)

    comparator.loc[0, "actual_demand"] = 999.0
    with pytest.raises(ValueError, match="different realized demand"):
        _comparison(reference, comparator)


def test_formal_runner_defaults_and_public_scripts_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_ai_darld_v3_factorized.py").read_text()
    assert 'default="historical_support_v4"' in source
    assert "--assemble-only" in source
    assert "--reuse-existing" in source
    for name in (
        "analyze_ai_darld_v3_corrected.py",
        "analyze_support_v3_similarity_residual.py",
    ):
        assert (root / "scripts" / name).is_file()


def test_complete_key_contains_operating_context() -> None:
    assert KEYS == [
        "dataset",
        "target_id",
        "cutoff",
        "lead_time_regime",
        "capacity_regime",
        "shortage_holding_ratio",
    ]
