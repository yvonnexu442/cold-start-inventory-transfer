from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "finalize_strict_search_stability_v1.py"
)
SPEC = importlib.util.spec_from_file_location("strict_search_stability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> pd.DataFrame:
    records = []
    for target, actual in (("a", 2.0), ("b", 4.0)):
        for method, cost, service, action in (
            ("strict_shared", 5.0, 0.6, 2.0),
            ("strict_separate", 4.0, 0.7, 3.0),
        ):
            records.append(
                {
                    "dataset": "X",
                    "target_id": target,
                    "cutoff": 10,
                    "horizon": 2,
                    "cost_ratio": 5,
                    "method": method,
                    "actual": actual,
                    "cost": cost,
                    "service": service,
                    "action": action,
                }
            )
    return pd.DataFrame(records)


def test_stability_pairing_preserves_products_and_sign() -> None:
    paired = MODULE._paired(_rows(), include_seed=False)
    assert len(paired) == 2
    assert paired.target_id.nunique() == 2
    assert paired.cost_difference.tolist() == [-1.0, -1.0]
    assert paired.service_difference.tolist() == pytest.approx([0.1, 0.1])


def test_stability_pairing_rejects_missing_and_duplicate_rows() -> None:
    with pytest.raises(ValueError, match="missing"):
        MODULE._paired(_rows().iloc[:-1], include_seed=False)
    duplicate = pd.concat([_rows(), _rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        MODULE._paired(duplicate, include_seed=False)


def test_runner_ensemble_file_selection_excludes_seed_rows(tmp_path: Path) -> None:
    service = _load_script("run_strict_shared_separate_v1.py")
    retail = _load_script("run_strict_shared_separate_retail_v1.py")
    for name in (
        "man_rows.parquet",
        "man_seed_rows.parquet",
        "braf_rows.parquet",
        "uci_rows.parquet",
        "uci_seed_rows.parquet",
        "favorita_rows.parquet",
    ):
        (tmp_path / name).touch()
    assert [p.name for p in service._ensemble_row_files(tmp_path)] == [
        "braf_rows.parquet",
        "man_rows.parquet",
    ]
    assert [p.name for p in retail._ensemble_row_files(tmp_path)] == [
        "favorita_rows.parquet",
        "uci_rows.parquet",
    ]
