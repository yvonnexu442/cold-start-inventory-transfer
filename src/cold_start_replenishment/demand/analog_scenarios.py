"""Sprint 1 scenario construction implementations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from cold_start_replenishment.analogs.similarity import normalized_weights
from cold_start_replenishment.framework.objects import CandidateAnalogSpace, DecisionScenarioSet


def _validate_histories(
    analog_space: CandidateAnalogSpace,
    donor_histories: Mapping[str, NDArray[np.float64]],
    horizon: int,
    n_scenarios: int,
) -> dict[str, NDArray[np.float64]]:
    if horizon <= 0 or n_scenarios <= 0:
        raise ValueError("horizon and n_scenarios must be positive")
    selected: dict[str, NDArray[np.float64]] = {}
    for donor_id in analog_space.donor_ids:
        if donor_id not in donor_histories:
            raise ValueError(f"Missing cutoff-limited history for donor: {donor_id}")
        history = np.asarray(donor_histories[donor_id], dtype=np.float64)
        if history.ndim != 1 or len(history) < horizon:
            raise ValueError(f"Donor history for {donor_id} is shorter than horizon")
        if not np.all(np.isfinite(history)) or np.any(history < 0):
            raise ValueError(f"Donor history for {donor_id} must be finite and nonnegative")
        selected[donor_id] = history
    return selected


def _sample_windows(
    donor_ids: tuple[str, ...],
    donor_histories: Mapping[str, NDArray[np.float64]],
    donor_probabilities: NDArray[np.float64],
    horizon: int,
    n_scenarios: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], list[str]]:
    donor_positions = rng.choice(len(donor_ids), size=n_scenarios, p=donor_probabilities)
    scenarios: list[NDArray[np.float64]] = []
    sampled_donors: list[str] = []
    for position in donor_positions:
        donor_id = donor_ids[int(position)]
        history = donor_histories[donor_id]
        start = int(rng.integers(0, len(history) - horizon + 1))
        scenarios.append(history[start : start + horizon])
        sampled_donors.append(donor_id)
    return np.stack(scenarios), sampled_donors


class SingleAnalogScenarios:
    """Baseline that treats the top-ranked analog as the only evidence source."""

    def construct(
        self,
        analog_space: CandidateAnalogSpace,
        donor_histories: Mapping[str, NDArray[np.float64]],
        horizon: int,
        n_scenarios: int,
        random_state: int | np.random.Generator | None = None,
    ) -> DecisionScenarioSet:
        histories = _validate_histories(analog_space, donor_histories, horizon, n_scenarios)
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        donor_ids = analog_space.donor_ids[:1]
        scenarios, sampled = _sample_windows(
            donor_ids,
            histories,
            np.ones(1, dtype=np.float64),
            horizon,
            n_scenarios,
            rng,
        )
        return DecisionScenarioSet(
            scenarios,
            horizon,
            "single_analog",
            {"sampled_donor_count": 1, "sampled_donors": sorted(set(sampled))},
        )


class WeightedAnalogScenarios:
    """First implementation of uncertain multi-analog scenario construction."""

    def construct(
        self,
        analog_space: CandidateAnalogSpace,
        donor_histories: Mapping[str, NDArray[np.float64]],
        horizon: int,
        n_scenarios: int,
        random_state: int | np.random.Generator | None = None,
    ) -> DecisionScenarioSet:
        histories = _validate_histories(analog_space, donor_histories, horizon, n_scenarios)
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        weights = normalized_weights(analog_space.relevance_scores)
        scenarios, sampled = _sample_windows(
            analog_space.donor_ids,
            histories,
            weights,
            horizon,
            n_scenarios,
            rng,
        )
        return DecisionScenarioSet(
            scenarios,
            horizon,
            "weighted_analog_mixture",
            {
                "sampled_donor_count": len(set(sampled)),
                "available_donor_count": len(analog_space.donor_ids),
                "normalized_weights": weights.tolist(),
            },
        )
