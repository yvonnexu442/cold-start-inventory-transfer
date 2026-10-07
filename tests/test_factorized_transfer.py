import numpy as np

from cold_start_replenishment.analogs.direct_mixture_learning import MixtureDecisionTask
from cold_start_replenishment.analogs.factorized_transfer import (
    CompleteMixtureCorrectionPolicy,
    FactorizedTransferPolicy,
    complete_mixture_distribution,
    complete_preserving_distribution,
    factorized_distribution,
    factorized_weights,
)
from cold_start_replenishment.analogs.relation_feature_transfer import (
    RelationFeaturePolicy,
    relation_distribution,
)
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level


def _task() -> MixtureDecisionTask:
    return MixtureDecisionTask(
        "p",
        np.asarray([[0.9, 0.1, 2.0, 2.0], [0.4, 0.7, 3.0, 9.0]]),
        np.asarray([[0, 0, 2, 3], [0, 8, 10, 12]], float),
        7.0,
        5.0,
        4,
        None,
        None,
    )


def test_factorized_weights_are_two_simplexes() -> None:
    occurrence, magnitude, alpha_occ, alpha_mag = factorized_weights(
        _task().donor_features, np.zeros(10)
    )
    assert np.isclose(occurrence.sum(), 1)
    assert np.isclose(magnitude.sum(), 1)
    assert np.all(occurrence >= 0) and np.all(magnitude >= 0)
    assert 0 <= alpha_occ <= 1 and 0 <= alpha_mag <= 1


def test_factorized_distribution_is_normalized() -> None:
    _, mass, diagnostics = factorized_distribution(_task(), np.zeros(10))
    assert np.isclose(mass.sum(), 1)
    assert 0 <= diagnostics["event_probability"] <= 1


def test_factorized_search_is_reproducible() -> None:
    first = FactorizedTransferPolicy(7, candidate_count=3).fit([_task()], [_task()])
    second = FactorizedTransferPolicy(7, candidate_count=3).fit([_task()], [_task()])
    assert np.allclose(first.parameters_, second.parameters_)


def test_support_is_raw_and_transformed_once() -> None:
    task = _task()
    theta = np.zeros(10)
    theta[7] = theta[9] = 1.0
    _, _, alpha_occ, alpha_mag = factorized_weights(task.donor_features, theta)
    expected = 1 / (1 + np.exp(-(-np.log1p(2.5) + np.std(task.donor_features[:, 1]))))
    assert np.isclose(alpha_occ, expected)
    assert 0 <= alpha_mag <= 1


def test_similarity_residual_zero_scores_recover_similarity_before_contraction() -> None:
    theta = np.zeros(10)
    occurrence, magnitude, _, _ = factorized_weights(
        _task().donor_features, theta, contraction=False, similarity_residual=True
    )
    expected = np.asarray([0.9, 0.4]) / 1.3
    assert np.allclose(occurrence, expected)
    assert np.allclose(magnitude, expected)


def test_similarity_residual_all_zero_similarity_falls_back_to_uniform() -> None:
    features = _task().donor_features.copy()
    features[:, 0] = 0.0
    occurrence, magnitude, _, _ = factorized_weights(
        features, np.zeros(10), contraction=False, similarity_residual=True
    )
    assert np.allclose(occurrence, [0.5, 0.5])
    assert np.allclose(magnitude, [0.5, 0.5])


def test_positive_atom_mass_uses_positive_simulation_count() -> None:
    task = _task()
    object.__setattr__(
        task,
        "donor_support",
        np.asarray([[20, 19, 17, 4], [20, 1, 17, 4]], float),
    )
    values, masses, diagnostics = factorized_distribution(
        task, np.zeros(10), contraction=False
    )
    event = diagnostics["event_probability"]
    assert np.allclose(masses[values == 2], event * 0.5 / 2)
    assert np.allclose(masses[values == 8], event * 0.5 / 3)


def _collapsed(values: np.ndarray, masses: np.ndarray) -> dict[float, float]:
    return {
        float(value): float(masses[values == value].sum())
        for value in np.unique(values)
    }


def test_zero_correction_exactly_recovers_complete_mixtures_and_actions() -> None:
    task = _task()
    for anchor, baseline in (
        ("uniform", np.asarray([0.5, 0.5])),
        ("similarity", np.asarray([0.9, 0.4]) / 1.3),
    ):
        expected_values, expected_masses, _ = complete_mixture_distribution(task, baseline)
        expected_action = optimal_feasible_level(
            expected_values, expected_masses, 1.0, task.cost_ratio
        )[0]
        for shared in (True, False):
            values, masses, _ = complete_preserving_distribution(
                task, np.zeros(10), anchor=anchor, shared=shared
            )
            actual_distribution = _collapsed(values, masses)
            expected_distribution = _collapsed(expected_values, expected_masses)
            assert actual_distribution.keys() == expected_distribution.keys()
            assert np.allclose(
                list(actual_distribution.values()), list(expected_distribution.values())
            )
            action = optimal_feasible_level(values, masses, 1.0, task.cost_ratio)[0]
            assert action == expected_action


def test_complete_correction_handles_zero_similarity_and_zero_events() -> None:
    task = _task()
    features = task.donor_features.copy()
    features[:, 0] = 0.0
    scenarios = np.zeros_like(task.donor_scenarios)
    boundary = MixtureDecisionTask(
        task.target_id,
        features,
        scenarios,
        task.actual_demand,
        task.cost_ratio,
        task.lead_time,
        task.capacity,
        task.minimum_order_quantity,
    )
    values, masses, diagnostics = complete_preserving_distribution(
        boundary, np.zeros(10), anchor="similarity", shared=False
    )
    assert np.array_equal(values, [0.0])
    assert np.array_equal(masses, [1.0])
    assert diagnostics["event_probability"] == 0.0


def test_complete_correction_policy_selects_only_development_choices() -> None:
    policy = CompleteMixtureCorrectionPolicy(7, candidate_count=2).fit(
        [_task(), _task()], [_task()]
    )
    assert policy.anchor_ in {"uniform", "similarity"}
    assert policy.calibration_weight_ in {0.0, 0.05, 0.10}


def test_joint_selection_keeps_exact_complete_anchor_available() -> None:
    policy = CompleteMixtureCorrectionPolicy(
        7,
        candidate_count=2,
        joint_validation_calibration=True,
        exact_anchor_fallback=True,
        occurrence_shift_grid=(0.0,),
    ).fit([_task(), _task()], [_task()])
    exact_rows = [
        row for row in policy.search_
        if int(row["candidate"]) == 0 and float(row["occurrence_logit_shift"]) == 0.0
    ]
    assert {str(row["anchor"]) for row in exact_rows} == {"uniform", "similarity"}
    assert policy.occurrence_logit_shift_ == 0.0


def test_contraction_ignores_simulation_atom_count_when_history_support_matches() -> None:
    task = _task()
    support = np.asarray([[20, 2, 17, 60], [20, 4, 17, 60]], float)
    first = factorized_weights(task.donor_features, np.zeros(10), donor_support=support)[2:]
    support[:, 3] = 600
    second = factorized_weights(task.donor_features, np.zeros(10), donor_support=support)[2:]
    assert np.allclose(first, second)


def test_relation_feature_policy_produces_normalized_distribution() -> None:
    tasks=[]
    for _ in range(8):
        task=_task()
        object.__setattr__(task, "donor_relations", np.asarray([[.9, 1, .1],[.4,.1,1.0]]))
        tasks.append(task)
    policy=RelationFeaturePolicy(7,candidate_count=4).fit(tasks[:5],tasks[5:])
    values,masses,diag=relation_distribution(tasks[0],policy.parameters_)
    assert values.shape == masses.shape
    assert np.isclose(masses.sum(),1.0)
    assert np.isclose(diag["occurrence_weights"].sum(),1.0)
