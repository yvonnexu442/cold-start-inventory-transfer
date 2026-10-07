"""Scientific objects passed between replaceable framework modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OperationalContext:
    """Decision-time target context; no target demand belongs in this object."""

    target_id: str
    horizon: int
    holding_cost: float
    shortage_cost: float
    capacity: float | None = None
    service_level: float | None = None
    minimum_order_quantity: float | None = None
    fixed_order_cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must be nonempty")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.holding_cost <= 0 or self.shortage_cost <= 0:
            raise ValueError("holding_cost and shortage_cost must be positive")
        if self.capacity is not None and self.capacity < 0:
            raise ValueError("capacity must be nonnegative")
        if self.service_level is not None and not 0 <= self.service_level <= 1:
            raise ValueError("service_level must lie in [0, 1]")
        if self.minimum_order_quantity is not None and self.minimum_order_quantity <= 0:
            raise ValueError("minimum_order_quantity must be positive")
        if self.fixed_order_cost < 0:
            raise ValueError("fixed_order_cost must be nonnegative")


@dataclass(frozen=True)
class CandidateAnalogSpace:
    """Admissible donors and their uncertain metadata relevance."""

    target_id: str
    donor_ids: tuple[str, ...]
    relevance_scores: NDArray[np.float64]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata_similarities: NDArray[np.float64] | None = None
    reliability_scores: NDArray[np.float64] | None = None
    reliability_lower: NDArray[np.float64] | None = None
    reliability_upper: NDArray[np.float64] | None = None
    donor_support: NDArray[np.float64] | None = None
    feature_provenance: dict[str, str] = field(default_factory=dict)
    no_close_analog: bool = False
    fallback_recommendation: str | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.relevance_scores, dtype=np.float64)
        if not self.target_id:
            raise ValueError("target_id must be nonempty")
        if len(self.donor_ids) == 0:
            raise ValueError("candidate analog space must contain at least one donor")
        if len(set(self.donor_ids)) != len(self.donor_ids):
            raise ValueError("donor_ids must be unique")
        if self.target_id in self.donor_ids:
            raise ValueError("target must be excluded from donor_ids")
        if scores.ndim != 1 or len(scores) != len(self.donor_ids):
            raise ValueError("relevance_scores must align with donor_ids")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError("relevance_scores must be finite and nonnegative")
        object.__setattr__(self, "relevance_scores", scores)
        for name in (
            "metadata_similarities",
            "reliability_scores",
            "reliability_lower",
            "reliability_upper",
            "donor_support",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 1 or len(array) != len(self.donor_ids):
                raise ValueError(f"{name} must align with donor_ids")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            if name != "donor_support" and np.any((array < 0) | (array > 1)):
                raise ValueError(f"{name} must lie in [0, 1]")
            if name == "donor_support" and np.any(array < 0):
                raise ValueError("donor_support must be nonnegative")
            object.__setattr__(self, name, array)

    @property
    def top_similarity(self) -> float:
        return float(np.max(self.relevance_scores))

    @property
    def ambiguity(self) -> float:
        """A transparent diagnostic, not a calibrated probability."""
        ordered = np.sort(self.relevance_scores)[::-1]
        if len(ordered) == 1:
            return 0.0
        return float(1.0 - max(ordered[0] - ordered[1], 0.0))


@dataclass(frozen=True)
class DecisionScenarioSet:
    """Nonnegative demand trajectories consumed by decision optimization."""

    values: NDArray[np.float64]
    horizon: int
    source_module: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    source_ids: tuple[str, ...] | None = None
    scenario_probabilities: NDArray[np.float64] | None = None
    reliability_provenance: str | None = None
    operational_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != self.horizon:
            raise ValueError("values must have shape (n_scenarios, horizon)")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("scenario values must be finite and nonnegative")
        if not self.source_module:
            raise ValueError("source_module must be nonempty")
        if self.source_ids is not None and len(self.source_ids) != values.shape[0]:
            raise ValueError("source_ids must align with scenarios")
        if self.scenario_probabilities is not None:
            probabilities = np.asarray(self.scenario_probabilities, dtype=np.float64)
            if probabilities.ndim != 1 or len(probabilities) != values.shape[0]:
                raise ValueError("scenario_probabilities must align with scenarios")
            if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
                raise ValueError("scenario_probabilities must be nonnegative and sum to one")
            object.__setattr__(self, "scenario_probabilities", probabilities)
        object.__setattr__(self, "values", values)

    @property
    def lead_time_demand(self) -> NDArray[np.float64]:
        return np.asarray(self.values.sum(axis=1), dtype=np.float64)


@dataclass(frozen=True)
class ReplenishmentDecision:
    """A feasible action produced before the target outcome is observed."""

    level: float
    expected_holding_cost: float
    expected_shortage_cost: float
    source_module: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    expected_fixed_order_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("decision level must be nonnegative")
        if self.expected_holding_cost < 0 or self.expected_shortage_cost < 0:
            raise ValueError("expected costs must be nonnegative")
        if self.expected_fixed_order_cost < 0:
            raise ValueError("expected_fixed_order_cost must be nonnegative")

    @property
    def expected_total_cost(self) -> float:
        return (
            self.expected_holding_cost
            + self.expected_shortage_cost
            + self.expected_fixed_order_cost
        )


@dataclass(frozen=True)
class DecisionEvaluation:
    """Held-out operational evaluation; never an input to upstream modules."""

    realized_demand: float
    holding_cost: float
    shortage_cost: float
    total_cost: float
    fill_rate_proxy: float
    regret: float | None
    normalized_regret: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
