from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from cold_start_replenishment.analogs.direct_mixture_learning import MixtureDecisionTask
from cold_start_replenishment.analogs.factorized_transfer import (
    _STRICT_ATOM_CACHE,
    _STRICT_ORDER_CACHE,
    StrictRelationPolicy,
    _strict_action_uncached,
    _strict_cached_action,
    _strict_task_atoms_uncached,
    strict_relation_distribution,
    strict_relation_weights,
)
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level

FEATURES = np.asarray(
    [
        [0.9, 0.2, 4.0, 2.0],
        [0.4, 0.7, 9.0, 8.0],
        [0.1, 0.4, 2.0, 5.0],
    ]
)


def test_strict_shared_weights_are_identical_and_normalized() -> None:
    occurrence, magnitude = strict_relation_weights(
        FEATURES,
        np.asarray([0.2, -0.3, 0.5, 0.8]),
        shared=True,
        contraction_alpha=0.25,
        similarity_residual=True,
    )
    np.testing.assert_allclose(occurrence, magnitude)
    np.testing.assert_allclose([occurrence.sum(), magnitude.sum()], [1.0, 1.0])


def test_strict_separate_nested_recovery() -> None:
    beta = np.asarray([0.2, -0.3, 0.5, 0.8])
    shared = strict_relation_weights(
        FEATURES,
        beta,
        shared=True,
        contraction_alpha=0.5,
        similarity_residual=False,
    )
    separate = strict_relation_weights(
        FEATURES,
        np.r_[beta, beta],
        shared=False,
        contraction_alpha=0.5,
        similarity_residual=False,
    )
    np.testing.assert_allclose(shared[0], separate[0])
    np.testing.assert_allclose(shared[1], separate[1])


def test_zero_residual_recovers_anchor_before_common_contraction() -> None:
    expected = FEATURES[:, 0] / FEATURES[:, 0].sum()
    occurrence, magnitude = strict_relation_weights(
        FEATURES,
        np.zeros(4),
        shared=True,
        contraction_alpha=0.0,
        similarity_residual=True,
    )
    np.testing.assert_allclose(occurrence, expected)
    np.testing.assert_allclose(magnitude, expected)


def test_common_contraction_and_parameter_counts() -> None:
    shared = StrictRelationPolicy(7, shared=True, similarity_residual=True)
    separate = StrictRelationPolicy(7, shared=False, similarity_residual=True)
    assert shared.effective_relation_parameters == 4
    assert separate.effective_relation_parameters == 8
    assert shared.effective_continuous_parameters == 5
    assert separate.effective_continuous_parameters == 9
    assert shared.contraction_grid == separate.contraction_grid == (0.0, 0.25, 0.5, 0.75)
    assert shared.candidates().shape == (100, 4)
    assert separate.candidates().shape == (100, 8)
    np.testing.assert_array_equal(
        shared.candidates(), StrictRelationPolicy(7, True, True).candidates()
    )


@pytest.mark.parametrize("budget", [100, 200])
def test_nested_search_contains_all_shared_unique_candidates(budget: int) -> None:
    shared = StrictRelationPolicy(
        7,
        shared=True,
        similarity_residual=True,
        search_budget=budget,
        nested_candidate_stream=True,
    ).candidates()
    separate = StrictRelationPolicy(
        7,
        shared=False,
        similarity_residual=True,
        search_budget=budget,
        nested_candidate_stream=True,
    ).candidates()
    assert shared.shape == (budget, 4)
    assert separate.shape == (budget, 8)
    shared_unique = np.unique(shared, axis=0)
    assert len(shared_unique) == budget // 2
    separate_tied = separate[np.all(separate[:, :4] == separate[:, 4:], axis=1), :4]
    assert len(separate_tied) == budget // 2
    assert {row.tobytes() for row in shared_unique} == {
        row.tobytes() for row in separate_tied
    }


def test_nested_candidate_stream_is_budget_nested() -> None:
    for shared in (True, False):
        small = StrictRelationPolicy(
            17,
            shared=shared,
            similarity_residual=True,
            search_budget=100,
            nested_candidate_stream=True,
        ).candidates()
        large = StrictRelationPolicy(
            17,
            shared=shared,
            similarity_residual=True,
            search_budget=200,
            nested_candidate_stream=True,
        ).candidates()
        if shared:
            small_unique = np.unique(small, axis=0)
            large_unique = np.unique(large, axis=0)
            assert {row.tobytes() for row in small_unique} <= {
                row.tobytes() for row in large_unique
            }
        else:
            small_tied = small[:50]
            large_tied = large[:100]
            small_untied = small[50:]
            large_untied = large[100:]
            assert {row.tobytes() for row in small_tied} <= {
                row.tobytes() for row in large_tied
            }
            assert {row.tobytes() for row in small_untied} <= {
                row.tobytes() for row in large_untied
            }


def test_prediction_does_not_use_evaluation_target_outcome() -> None:
    scenarios = np.asarray([[0.0, 1.0, 2.0], [0.0, 0.0, 5.0], [1.0, 3.0, 4.0]])
    common = dict(
        target_id="hidden",
        donor_features=FEATURES,
        donor_scenarios=scenarios,
        cost_ratio=5.0,
        lead_time=2,
        capacity=None,
        minimum_order_quantity=None,
    )
    first = MixtureDecisionTask(actual_demand=0.0, **common)
    second = MixtureDecisionTask(actual_demand=99.0, **common)
    arguments = dict(
        coefficients=np.asarray([0.1, -0.2, 0.3, 0.4]),
        shared=True,
        contraction_alpha=0.25,
        similarity_residual=True,
    )
    first_distribution = strict_relation_distribution(first, **arguments)
    second_distribution = strict_relation_distribution(second, **arguments)
    np.testing.assert_allclose(first_distribution[0], second_distribution[0])
    np.testing.assert_allclose(first_distribution[1], second_distribution[1])


def test_vectorized_atoms_match_explicit_donor_mass_construction() -> None:
    scenarios = np.asarray([[0.0, 1.0, 2.0], [0.0, 0.0, 5.0], [1.0, 3.0, 4.0]])
    task = MixtureDecisionTask(
        target_id="hidden",
        donor_features=FEATURES,
        donor_scenarios=scenarios,
        actual_demand=4.0,
        cost_ratio=5.0,
        lead_time=2,
        capacity=None,
        minimum_order_quantity=None,
    )
    coefficients = np.asarray([0.1, -0.2, 0.3, 0.4])
    occurrence, magnitude = strict_relation_weights(
        FEATURES,
        coefficients,
        shared=True,
        contraction_alpha=0.25,
        similarity_residual=True,
    )
    values, masses, diagnostics = strict_relation_distribution(
        task,
        coefficients,
        shared=True,
        contraction_alpha=0.25,
        similarity_residual=True,
    )
    event = float(occurrence @ np.mean(scenarios > 0, axis=1))
    expected_values = [0.0]
    expected_masses = [1.0 - event]
    active = np.flatnonzero(np.any(scenarios > 0, axis=1))
    active_weights = magnitude[active] / magnitude[active].sum()
    for donor, donor_mass in zip(active, active_weights, strict=True):
        positive = scenarios[donor][scenarios[donor] > 0]
        expected_values.extend(positive.tolist())
        expected_masses.extend(np.full(len(positive), event * donor_mass / len(positive)))
    np.testing.assert_allclose(values, expected_values)
    np.testing.assert_allclose(masses, expected_masses)
    np.testing.assert_allclose(diagnostics["event_probability"], event)


def test_cached_search_action_matches_formal_solver() -> None:
    task = MixtureDecisionTask(
        "target",
        FEATURES,
        np.asarray([[0.0, 1.0, 2.0], [0.0, 0.0, 5.0], [1.0, 3.0, 4.0]]),
        3.0,
        5.0,
        2,
        6.0,
        2.0,
        1.5,
        0.7,
    )
    values, masses, _ = strict_relation_distribution(
        task,
        np.asarray([0.1, -0.2, 0.3, 0.4]),
        shared=True,
        contraction_alpha=0.25,
        similarity_residual=True,
    )
    expected = optimal_feasible_level(
        values,
        masses,
        task.holding_cost,
        task.holding_cost * task.cost_ratio,
        capacity=task.capacity,
        minimum_order_quantity=task.minimum_order_quantity,
        fixed_order_cost=task.fixed_order_cost,
    )[0]
    assert _strict_cached_action(task, values, masses) == expected


def _task(scenarios: np.ndarray, **overrides: float | None) -> MixtureDecisionTask:
    arguments = {
        "target_id": "cache-test",
        "donor_features": FEATURES[: len(scenarios)],
        "donor_scenarios": np.asarray(scenarios, float),
        "actual_demand": 3.0,
        "cost_ratio": 5.0,
        "lead_time": 2,
        "capacity": None,
        "minimum_order_quantity": None,
        "holding_cost": 1.0,
        "fixed_order_cost": 0.0,
    }
    arguments.update(overrides)
    return MixtureDecisionTask(**arguments)


def test_task_cache_is_identity_checked_and_cleaned() -> None:
    coefficients = np.asarray([0.1, -0.2, 0.3, 0.4])
    expected_atoms = []
    references = []
    for offset in range(40):
        task = _task(np.asarray([[0.0, 1.0 + offset], [0.0, 3.0 + offset]]))
        references.append(weakref.ref(task))
        expected = _strict_task_atoms_uncached(task)
        actual = strict_relation_distribution(
            task,
            coefficients,
            shared=True,
            contraction_alpha=0.25,
            similarity_residual=True,
        )
        np.testing.assert_allclose(actual[0], expected[0])
        expected_atoms.append(expected[0].copy())
        del task
        gc.collect()
    assert all(reference() is None for reference in references)
    assert not _STRICT_ATOM_CACHE
    assert not _STRICT_ORDER_CACHE
    assert len({tuple(values) for values in expected_atoms}) == len(expected_atoms)


def test_stale_raw_id_entry_cannot_cross_tasks() -> None:
    stale = _task(np.asarray([[0.0, 1.0], [0.0, 2.0]]))
    current = _task(np.asarray([[0.0, 11.0], [0.0, 22.0]]))
    stale_values, stale_donors, stale_counts = _strict_task_atoms_uncached(stale)
    _STRICT_ATOM_CACHE[id(current)] = (
        weakref.ref(stale),
        stale_values,
        stale_donors,
        stale_counts,
    )
    values, _, _ = strict_relation_distribution(
        current,
        np.zeros(4),
        shared=True,
        contraction_alpha=0.0,
        similarity_residual=False,
    )
    np.testing.assert_allclose(values, [0.0, 11.0, 22.0])
    assert _STRICT_ATOM_CACHE[id(current)][0]() is current


def test_cached_and_uncached_distribution_inputs_and_order_are_identical() -> None:
    coefficients = np.asarray([0.4, -0.1, 0.2, 0.7])
    tasks = [
        _task(np.asarray([[0.0, 1.0, 2.0], [0.0, 0.0, 4.0]])),
        _task(np.asarray([[0.0, 0.0, 0.0], [0.0, 5.0, 7.0]])),
    ]
    forward = []
    for task in tasks:
        values, masses, diagnostics = strict_relation_distribution(
            task,
            coefficients,
            shared=True,
            contraction_alpha=0.5,
            similarity_residual=False,
        )
        ref_values, _, _ = _strict_task_atoms_uncached(task)
        np.testing.assert_allclose(values, ref_values)
        forward.append((values.copy(), masses.copy(), diagnostics["event_probability"]))
    for task, expected in zip(reversed(tasks), reversed(forward), strict=True):
        actual = strict_relation_distribution(
            task,
            coefficients,
            shared=True,
            contraction_alpha=0.5,
            similarity_residual=False,
        )
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
        assert actual[2]["event_probability"] == expected[2]


@pytest.mark.parametrize(
    ("scenarios", "minimum", "capacity", "fixed"),
    [
        (np.zeros((2, 4)), None, None, 0.0),
        (np.asarray([[0.0, 2.0, 6.0], [0.0, 0.0, 4.0]]), 3.0, 5.0, 2.5),
        (np.asarray([[0.0, 2.0, 6.0], [0.0, 0.0, 4.0]]), 8.0, 5.0, 2.5),
    ],
)
def test_cached_action_matches_reference_on_boundaries(
    scenarios: np.ndarray, minimum: float | None, capacity: float | None, fixed: float
) -> None:
    task = _task(
        scenarios,
        minimum_order_quantity=minimum,
        capacity=capacity,
        fixed_order_cost=fixed,
    )
    values, masses, _ = strict_relation_distribution(
        task,
        np.asarray([0.2, -0.1, 0.5, 0.3]),
        shared=True,
        contraction_alpha=0.25,
        similarity_residual=True,
    )
    assert _strict_cached_action(task, values, masses) == _strict_action_uncached(
        task, values, masses
    )


def test_cached_scenarios_are_immutable() -> None:
    task = _task(np.asarray([[0.0, 1.0], [0.0, 2.0]]))
    strict_relation_distribution(
        task,
        np.zeros(4),
        shared=True,
        contraction_alpha=0.0,
        similarity_residual=False,
    )
    with pytest.raises(ValueError):
        task.donor_scenarios[0, 1] = 99.0
