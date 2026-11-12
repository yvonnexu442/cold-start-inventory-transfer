import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/ai_darld_v3/service_calibration_fixed_policy_braf_v2"


def test_released_fixed_policy_service_calibration_outputs_are_consistent():
    summary = pd.read_csv(OUTPUT / "summary.csv").set_index("method")
    comparisons = pd.read_csv(OUTPUT / "paired_comparison.csv")
    selections = pd.read_csv(OUTPUT / "calibration_selection.csv")
    manifest = json.loads((OUTPUT / "manifest.json").read_text())

    assert set(summary.index) == {
        "complete_similarity",
        "component_specific_matched_service",
        "component_specific_uncalibrated",
    }
    assert summary["rows"].eq(13_500).all()
    assert selections.shape[0] == 9
    assert selections["status"].eq("selected").all()
    assert selections["feasible"].all()
    assert manifest["fixed_policy_counterfactual"] is True
    assert manifest["scoring_uses_original_operational_costs"] is True
    assert manifest["evaluation_target_outcomes_used_for_calibration"] is False

    effect = comparisons[
        comparisons["comparator"].eq("component_specific_uncalibrated")
    ].iloc[0]
    assert effect["first_policy"] == "component_specific_matched_service"
    assert effect["cost_difference"] > 0
    assert effect["service_difference"] > 0
    assert effect["action_difference"] > 0

    complete = comparisons[
        comparisons["first_policy"].eq("component_specific_matched_service")
        & comparisons["comparator"].eq("complete_similarity")
    ].iloc[0]
    assert complete["cost_ci_low"] > 0
    assert complete["service_ci_low"] > 0
