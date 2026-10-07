#!/usr/bin/env python3
"""Generate the deterministic worked example used in the V3 manuscript."""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ai_darld_v3_direct_mixture import _features, _support_matrix
from run_ai_darld_v3_factorized import _population

from cold_start_replenishment.analogs.direct_mixture_learning import MixtureDecisionTask
from cold_start_replenishment.analogs.factorized_transfer import (
    complete_mixture_distribution,
    factorized_distribution,
    factorized_weights,
)
from cold_start_replenishment.data.spdf_pilot import parse_braf, redacted_id
from cold_start_replenishment.evaluation.acceptance_enhancement import _construct_scenarios
from cold_start_replenishment.evaluation.sprint2 import _operational_values
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level


def main():
    dataset = parse_braf()
    full = yaml.safe_load((ROOT / "configs/full_scale.yaml").read_text())
    targets, eligible = _population(dataset, full)
    targets = np.asarray(targets[:250])
    # Predeclared case rule: lexicographically first redacted BRAF target,
    # earliest cutoff, native horizon, ratio five, unconstrained capacity.
    target_pairs = sorted(
        (redacted_id(dataset.name, dataset.metadata.iloc[int(source)].item_id), int(source))
        for source in targets
    )
    target_id, source = target_pairs[0]
    cutoff = int(full["datasets"][dataset.name]["cutoffs"][0])
    checkpoint = pd.read_parquet(ROOT / "outputs/full_scale/checkpoints/braf_reliability.parquet")
    pool = checkpoint[
        checkpoint.target_id.astype(str).eq(target_id) & checkpoint.cutoff.eq(cutoff)
    ].sort_values("similarity_rank").reset_index(drop=True)
    id_to_source = {
        redacted_id(dataset.name, row.item_id): int(str(index))
        for index, row in dataset.metadata.iterrows()
    }
    target_order = {int(value): index for index, value in enumerate(targets)}
    order = target_order[source]
    lead = min(
        max(1, int(round(float(dataset.metadata.iloc[source].lead_time)))),
        int(full["datasets"][dataset.name]["native_lead_time_cap"]),
    )
    demand = dataset.demand.to_numpy(float)
    donor_ids = pool.donor_id.astype(str).tolist()
    samples, donor_histories = [], []
    for donor_id in donor_ids:
        donor_source = id_to_source[donor_id]
        history = demand[donor_source, :cutoff]
        donor_histories.append(history)
        samples.append(_construct_scenarios(
            [donor_id], [history], np.ones(1), lead, 300,
            20260930 + order * 10007 + cutoff * 101 + donor_source,
            "worked_example",
        ).lead_time_demand)
    matrix = np.vstack(samples)
    holding, fixed, moq = _operational_values(dataset, source, eligible, full["datasets"][dataset.name])
    task = MixtureDecisionTask(
        target_id, _features(pool, matrix, donor_histories), matrix,
        float(demand[source, cutoff:cutoff + lead].sum()), 5.0, lead, None,
        moq, holding, fixed, donor_support=_support_matrix(donor_histories, lead, 300),
    )
    parameters = pd.read_csv(ROOT / "outputs/ai_darld_v3/historical_support_v4_parameters_braf.csv")
    parameters = parameters[parameters.dataset.eq("BRAF") & parameters.cutoff.eq(cutoff)]

    def fitted_distribution(policy_name: str, *, shared: bool, similarity_residual: bool):
        rows = parameters[parameters.policy.eq(policy_name)].sort_values("seed")
        occurrence_rows, magnitude_rows, fitted = [], [], []
        raw_events, calibrated_events, alpha_occ_rows, alpha_mag_rows = [], [], [], []
        for _, row in rows.iterrows():
            theta = row[[f"theta_{i}" for i in range(10)]].to_numpy(float)
            wo, wm, alpha_occ, alpha_mag = factorized_weights(
                task.donor_features, theta, donor_support=task.donor_support,
                shared=shared, similarity_residual=similarity_residual,
            )
            occurrence_rows.append(wo)
            magnitude_rows.append(wm)
            distribution = factorized_distribution(
                    task, theta, float(row.occurrence_logit_shift), shared=shared,
                    similarity_residual=similarity_residual,
                )
            fitted.append(distribution)
            raw_events.append(float(wo @ np.mean(task.donor_scenarios > 0, axis=1)))
            calibrated_events.append(float(distribution[2]["event_probability"]))
            alpha_occ_rows.append(alpha_occ)
            alpha_mag_rows.append(alpha_mag)
        fitted_values = np.concatenate([item[0] for item in fitted])
        fitted_masses = np.concatenate([item[1] / len(fitted) for item in fitted])
        return (
            np.mean(occurrence_rows, axis=0),
            np.mean(magnitude_rows, axis=0),
            fitted_values,
            fitted_masses,
            float(np.mean(raw_events)),
            float(np.mean(calibrated_events)),
            float(np.mean(alpha_occ_rows)),
            float(np.mean(alpha_mag_rows)),
        )

    occurrence, magnitude, values, masses, raw_event, calibrated_event, alpha_occ, alpha_mag = fitted_distribution(
        "factorized_similarity_residual", shared=False, similarity_residual=True
    )
    action, expected_holding, expected_shortage, expected_fixed = optimal_feasible_level(
        values, masses, holding, holding * 5,
        minimum_order_quantity=moq, fixed_order_cost=fixed,
    )
    similarity_weights = np.maximum(task.donor_features[:, 0], 0.0)
    if similarity_weights.sum() <= 0:
        similarity_weights = np.ones(len(similarity_weights))
    similarity_weights = similarity_weights / similarity_weights.sum()
    complete_values, complete_masses, _ = complete_mixture_distribution(task, similarity_weights)
    complete_action, _, _, _ = optimal_feasible_level(
        complete_values, complete_masses, holding, holding * 5,
        minimum_order_quantity=moq, fixed_order_cost=fixed,
    )
    event = np.mean(matrix > 0, axis=1)
    positive = np.asarray([row[row > 0].mean() if np.any(row > 0) else 0 for row in matrix])
    example = pd.DataFrame({
        "donor_id": donor_ids,
        "metadata_similarity": pool.metadata_similarity.to_numpy(float),
        "lead_event_probability": event,
        "positive_lead_mean": positive,
        "occurrence_weight": occurrence,
        "magnitude_weight": magnitude,
        "complete_similarity_weight": similarity_weights,
    })
    example["display"] = example.occurrence_weight.rank(ascending=False, method="first").le(2) | example.magnitude_weight.rank(ascending=False, method="first").le(2)
    example.to_csv(ROOT / "outputs/ai_darld_v3/worked_example.csv", index=False)
    summary = {
        "selection_rule": "lexicographically first redacted BRAF target; earliest cutoff; native horizon; ratio 5; unconstrained",
        "target_id": target_id,
        "target_description": str(dataset.metadata.iloc[source].description),
        "target_lead_time_metadata": float(dataset.metadata.iloc[source].lead_time),
        "target_price_metadata": float(dataset.metadata.iloc[source].price),
        "cutoff": cutoff,
        "lead_time": lead,
        "pre_calibration_event_probability": raw_event,
        "predicted_event_probability": calibrated_event,
        "conditional_positive_mean": float(np.sum(values[values > 0] * masses[values > 0]) / masses[values > 0].sum()),
        "predicted_mean": float(np.sum(values * masses)),
        "selected_action": action,
        "mean_occurrence_contraction": alpha_occ,
        "mean_positive_size_contraction": alpha_mag,
        "complete_similarity_event_probability": float(complete_masses[complete_values > 0].sum()),
        "complete_similarity_conditional_positive_mean": float(
            np.sum(complete_values[complete_values > 0] * complete_masses[complete_values > 0])
            / complete_masses[complete_values > 0].sum()
        ),
        "complete_similarity_predicted_mean": float(np.sum(complete_values * complete_masses)),
        "complete_similarity_selected_action": complete_action,
        "occurrence_weight_sum": float(occurrence.sum()),
        "magnitude_weight_sum": float(magnitude.sum()),
        "complete_similarity_weight_sum": float(similarity_weights.sum()),
        "distribution_mass_sum": float(masses.sum()),
        "complete_distribution_mass_sum": float(complete_masses.sum()),
        "minimum_order_quantity": float(moq),
        "capacity": None,
        "holding_cost": float(holding),
        "shortage_cost": float(holding * 5),
        "fixed_order_cost": float(fixed),
        "expected_holding_cost": expected_holding,
        "expected_shortage_cost": expected_shortage,
        "expected_fixed_cost": expected_fixed,
        "actual_lead_demand": task.actual_demand,
    }
    (ROOT / "outputs/ai_darld_v3/worked_example.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "analysis": "illustrative walkthrough of an existing formal fitted policy",
        "selection_frozen_before_outcome_inspection": True,
        "selection_rule": summary["selection_rule"],
        "formal_policy": "factorized_similarity_residual",
        "comparator": "complete_similarity",
        "dataset": "BRAF",
        "ratio": 5.0,
        "capacity": None,
        "future_target_demand_used_for_selection_or_action": False,
        "weights": "post-contraction, averaged across the five formal fitted seeds",
        "event_probability": "ensemble mean after each seed's validation-fitted logit intercept",
        "positive_law": "horizon lead-demand atoms; magnitude weights renormalized over positive-support donors",
        "parameter_file": "outputs/ai_darld_v3/historical_support_v4_parameters_braf.csv",
        "parameter_file_sha256": hashlib.sha256((ROOT / "outputs/ai_darld_v3/historical_support_v4_parameters_braf.csv").read_bytes()).hexdigest(),
        "checkpoint_file": "outputs/full_scale/checkpoints/braf_reliability.parquet",
        "checkpoint_file_sha256": hashlib.sha256((ROOT / "outputs/full_scale/checkpoints/braf_reliability.parquet").read_bytes()).hexdigest(),
    }
    (ROOT / "outputs/ai_darld_v3/worked_example_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(example[example.display].to_string(index=False))


if __name__ == "__main__":
    main()
