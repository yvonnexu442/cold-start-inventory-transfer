import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_worked_example_is_complete_and_feasible():
    weights = pd.read_csv(ROOT / "outputs/ai_darld_v3/worked_example.csv")
    summary = json.loads((ROOT / "outputs/ai_darld_v3/worked_example.json").read_text())

    assert len(weights) == 10
    for column in (
        "occurrence_weight",
        "magnitude_weight",
        "complete_similarity_weight",
    ):
        assert np.isclose(weights[column].sum(), 1.0)
        assert (weights[column] >= 0).all()
    assert np.isclose(summary["distribution_mass_sum"], 1.0)
    assert np.isclose(summary["complete_distribution_mass_sum"], 1.0)
    assert 0 <= summary["pre_calibration_event_probability"] <= 1
    assert 0 <= summary["predicted_event_probability"] <= 1
    assert 0 <= summary["complete_similarity_event_probability"] <= 1
    assert summary["selected_action"] >= summary["minimum_order_quantity"]
    assert summary["complete_similarity_selected_action"] >= summary["minimum_order_quantity"]
    assert summary["capacity"] is None


def test_worked_example_selection_rule_is_outcome_independent():
    manifest = json.loads(
        (ROOT / "outputs/ai_darld_v3/worked_example_manifest.json").read_text()
    )
    assert manifest["selection_frozen_before_outcome_inspection"] is True
    assert manifest["future_target_demand_used_for_selection_or_action"] is False
    assert manifest["formal_policy"] == "factorized_similarity_residual"
    assert manifest["comparator"] == "complete_similarity"
