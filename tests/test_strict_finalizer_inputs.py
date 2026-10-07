from __future__ import annotations

from pathlib import Path


def test_strict_finalizer_has_no_implicit_legacy_input() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/finalize_strict_shared_separate_v1.py"
    ).read_text(encoding="utf-8")
    assert "strict_shared_separate_v1/paired_contrasts.csv" not in source
    assert "--historical-comparison" in source


def test_readme_generates_all_search_budget_inputs() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    for runner in (
        "run_strict_shared_separate_v1.py",
        "run_strict_shared_separate_retail_v1.py",
    ):
        assert f"{runner} --search-budget 100" in readme
        assert f"{runner} --search-budget 200" in readme
