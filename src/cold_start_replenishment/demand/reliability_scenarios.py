"""Reliability-aware Sprint 2 scenario construction implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cold_start_replenishment.framework.objects import CandidateAnalogSpace, DecisionScenarioSet


def normalized_capped_weights(
    values: NDArray[np.float64], maximum_weight: float | None = None
) -> NDArray[np.float64]:
    raw = np.clip(np.asarray(values, dtype=float), 0, None)
    if raw.ndim != 1 or len(raw) == 0:
        raise ValueError("weights must be a nonempty vector")
    weights = raw / raw.sum() if raw.sum() > 0 else np.full(len(raw), 1 / len(raw))
    if maximum_weight is None:
        return np.asarray(weights, dtype=np.float64)
    if not 1 / len(weights) <= maximum_weight <= 1:
        raise ValueError("maximum_weight is infeasible for the donor count")
    free = np.ones(len(weights), dtype=bool)
    result = np.zeros(len(weights), dtype=float)
    remaining = 1.0
    while free.any():
        proposal = weights[free] / weights[free].sum() * remaining
        exceeds = proposal > maximum_weight + 1e-12
        free_indices = np.flatnonzero(free)
        if not exceeds.any():
            result[free] = proposal
            break
        fixed = free_indices[exceeds]
        result[fixed] = maximum_weight
        free[fixed] = False
        remaining = 1.0 - result.sum()
    return np.asarray(result / result.sum(), dtype=np.float64)


def effective_donor_count(weights: NDArray[np.float64]) -> float:
    values = np.asarray(weights, dtype=float)
    return float(1.0 / np.square(values).sum())


def scenario_weights(
    analog_space: CandidateAnalogSpace,
    mode: str,
    maximum_weight: float = 0.35,
) -> NDArray[np.float64]:
    similarity = (
        analog_space.metadata_similarities
        if analog_space.metadata_similarities is not None
        else analog_space.relevance_scores
    )
    reliability = (
        analog_space.reliability_scores
        if analog_space.reliability_scores is not None
        else analog_space.relevance_scores
    )
    support = (
        np.log1p(analog_space.donor_support)
        if analog_space.donor_support is not None
        else np.ones(len(similarity))
    )
    if mode == "similarity_only":
        raw = similarity
        cap = None
    elif mode == "reliability_only":
        raw = reliability * support
        cap = None
    elif mode.startswith("combined_temperature_"):
        temperature = float(mode.rsplit("_", 2)[-2] + "." + mode.rsplit("_", 1)[-1])
        raw = np.power(np.clip(similarity * reliability * support, 1e-12, None), 1 / temperature)
        cap = None
    elif mode in {"capped_combined", "reliability_fallback", "top_donor_removed"}:
        raw = similarity * reliability * support
        if mode == "top_donor_removed" and len(raw) > 1:
            raw = raw.copy()
            raw[int(np.argmax(similarity))] = 0.0
        cap = maximum_weight
    else:
        raise ValueError(f"Unknown scenario weighting mode: {mode}")
    return normalized_capped_weights(np.asarray(raw, dtype=np.float64), cap)


def _occurrence_size_trajectories(
    donor_ids: tuple[str, ...],
    histories: Mapping[str, NDArray[np.float64]],
    weights: NDArray[np.float64],
    horizon: int,
    n_scenarios: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    if horizon <= 0 or n_scenarios <= 0:
        raise ValueError("horizon and n_scenarios must be positive")
    selected = rng.choice(len(donor_ids), size=n_scenarios, p=weights)
    scenarios = np.zeros((n_scenarios, horizon), dtype=float)
    sources: list[str] = []
    for row, position in enumerate(selected):
        donor_id = donor_ids[int(position)]
        if donor_id not in histories:
            raise ValueError(f"Missing cutoff history for donor: {donor_id}")
        history = np.asarray(histories[donor_id], dtype=float)
        if history.ndim != 1 or len(history) == 0 or np.any(history < 0):
            raise ValueError("Donor histories must be nonempty nonnegative vectors")
        probability = float((history > 0).mean())
        occurrences = rng.random(horizon) < probability
        positive = history[history > 0]
        if len(positive):
            scenarios[row] = occurrences * rng.choice(positive, size=horizon, replace=True)
        sources.append(donor_id)
    return scenarios, tuple(sources)


@dataclass(frozen=True)
class WeightedScenarioConstructor:
    """Shared transparent implementation for prespecified weighting modes."""

    mode: str
    maximum_weight: float = 0.35
    low_reliability_threshold: float = 0.35

    def construct(
        self,
        analog_space: CandidateAnalogSpace,
        donor_histories: Mapping[str, NDArray[np.float64]],
        horizon: int,
        n_scenarios: int,
        random_state: int | np.random.Generator | None = None,
    ) -> DecisionScenarioSet:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        donor_ids = analog_space.donor_ids
        fallback_used = False
        if self.mode == "reliability_fallback" and (
            analog_space.fallback_recommendation == "broad_fallback"
        ):
            broad_ids = tuple(analog_space.diagnostics.get("broad_donor_ids", ()))
            if broad_ids:
                donor_ids = broad_ids
                weights = np.full(len(donor_ids), 1 / len(donor_ids), dtype=np.float64)
                fallback_used = True
            else:
                weights = scenario_weights(analog_space, self.mode, self.maximum_weight)
        else:
            weights = scenario_weights(analog_space, self.mode, self.maximum_weight)
        scenarios, sources = _occurrence_size_trajectories(
            donor_ids, donor_histories, weights, horizon, n_scenarios, rng
        )
        probabilities = np.full(n_scenarios, 1 / n_scenarios, dtype=np.float64)
        return DecisionScenarioSet(
            scenarios,
            horizon,
            self.mode,
            {
                "donor_weights": weights.tolist(),
                "effective_donor_count": effective_donor_count(weights),
                "donor_concentration": float(np.square(weights).sum()),
                "maximum_donor_weight": float(weights.max()),
                "fallback_used": fallback_used,
            },
            source_ids=sources,
            scenario_probabilities=probabilities,
            reliability_provenance="cross_donor_pseudo_target_calibration",
        )


class RawSimilarityWeightedScenarios(WeightedScenarioConstructor):
    def __init__(self) -> None:
        super().__init__("similarity_only", maximum_weight=1.0)


class ReliabilityWeightedScenarios(WeightedScenarioConstructor):
    def __init__(self, mode: str = "capped_combined", maximum_weight: float = 0.35) -> None:
        super().__init__(mode, maximum_weight)


class GroupEmpiricalScenarios(WeightedScenarioConstructor):
    def __init__(self) -> None:
        super().__init__("similarity_only", maximum_weight=1.0)

    def construct(
        self,
        analog_space: CandidateAnalogSpace,
        donor_histories: Mapping[str, NDArray[np.float64]],
        horizon: int,
        n_scenarios: int,
        random_state: int | np.random.Generator | None = None,
    ) -> DecisionScenarioSet:
        group_ids = tuple(analog_space.diagnostics.get("group_donor_ids", ()))
        if not group_ids:
            group_ids = analog_space.donor_ids
        group_space = CandidateAnalogSpace(
            analog_space.target_id, group_ids, np.ones(len(group_ids), dtype=np.float64)
        )
        return WeightedScenarioConstructor("similarity_only", 1.0).construct(
            group_space, donor_histories, horizon, n_scenarios, random_state
        )


class BroadFallbackScenarios(GroupEmpiricalScenarios):
    def construct(
        self,
        analog_space: CandidateAnalogSpace,
        donor_histories: Mapping[str, NDArray[np.float64]],
        horizon: int,
        n_scenarios: int,
        random_state: int | np.random.Generator | None = None,
    ) -> DecisionScenarioSet:
        broad_ids = tuple(analog_space.diagnostics.get("broad_donor_ids", ()))
        if not broad_ids:
            broad_ids = analog_space.donor_ids
        broad_space = CandidateAnalogSpace(
            analog_space.target_id, broad_ids, np.ones(len(broad_ids), dtype=np.float64)
        )
        result = WeightedScenarioConstructor("similarity_only", 1.0).construct(
            broad_space, donor_histories, horizon, n_scenarios, random_state
        )
        return DecisionScenarioSet(
            result.values,
            horizon,
            "broad_fallback",
            result.diagnostics,
            result.source_ids,
            result.scenario_probabilities,
            "broad_admissible_pool",
        )


@dataclass(frozen=True)
class OracleTargetHistory:
    """Evaluation-only scenario constructor; never registered as an operational method."""

    target_id: str
    target_history: NDArray[np.float64]

    def construct(
        self,
        analog_space: CandidateAnalogSpace,
        donor_histories: Mapping[str, NDArray[np.float64]],
        horizon: int,
        n_scenarios: int,
        random_state: int | np.random.Generator | None = None,
    ) -> DecisionScenarioSet:
        if analog_space.target_id != self.target_id:
            raise ValueError("Oracle target mismatch")
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        scenarios, sources = _occurrence_size_trajectories(
            (self.target_id,),
            {self.target_id: np.asarray(self.target_history, dtype=np.float64)},
            np.ones(1, dtype=np.float64),
            horizon,
            n_scenarios,
            rng,
        )
        return DecisionScenarioSet(
            scenarios,
            horizon,
            "oracle_target_history",
            {"evaluation_only": True},
            sources,
            np.full(n_scenarios, 1 / n_scenarios),
            "hidden_target_history_evaluation_only",
        )
