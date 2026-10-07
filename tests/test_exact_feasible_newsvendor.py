import numpy as np

from cold_start_replenishment.analogs.direct_mixture_learning import MixtureDecisionTask
from cold_start_replenishment.analogs.factorized_transfer import (
    complete_mixture_distribution,
    factorized_distribution,
)
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level


def test_upward_moq_rounding_is_not_the_optimum() -> None:
    # A minimum positive quantity is not a batch multiple: q=10 is feasible.
    q, *_ = optimal_feasible_level(
        np.array([0.0, 10.0]), np.array([0.2, 0.8]), 1.0, 4.0,
        minimum_order_quantity=6.0,
    )
    assert q == 10.0


def test_capacity_below_moq_leaves_only_zero() -> None:
    q, *_ = optimal_feasible_level(
        np.array([8.0]), None, 1.0, 10.0, capacity=4.0,
        minimum_order_quantity=5.0,
    )
    assert q == 0.0


def test_zero_demand_and_fixed_cost_boundary_choose_zero() -> None:
    assert optimal_feasible_level(np.zeros(3), None, 1.0, 5.0)[0] == 0.0
    assert optimal_feasible_level(
        np.array([0.0, 4.0]), None, 1.0, 1.0, fixed_order_cost=10.0
    )[0] == 0.0


def test_complete_uniform_is_not_uniform_hurdle_when_events_differ() -> None:
    task = MixtureDecisionTask(
        "x", np.array([[1.0, .2, 1.0, 10.0], [1.0, .8, 1.0, 20.0]]),
        np.array([[0., 0., 0., 0., 10.], [0., 20., 20., 20., 20.]]),
        0., 5., 1, None, None,
    )
    complete_values, complete_mass, complete_diag = complete_mixture_distribution(
        task, np.array([.5, .5])
    )
    theta = np.zeros(10)
    theta[[6, 8]] = 20
    hurdle_values, hurdle_mass, hurdle_diag = factorized_distribution(task, theta)
    assert np.isclose(complete_diag["event_probability"], .5)
    assert np.isclose(np.sum(complete_values * complete_mass), 9.0)
    assert np.isclose(hurdle_diag["event_probability"], .5)
    assert np.isclose(np.sum(hurdle_values * hurdle_mass), 7.5, atol=1e-7)
