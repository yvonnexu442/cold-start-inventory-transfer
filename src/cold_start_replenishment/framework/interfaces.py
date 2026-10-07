"""Replaceable interfaces aligned with frozen scientific responsibilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cold_start_replenishment.framework.objects import (
    CandidateAnalogSpace,
    DecisionEvaluation,
    DecisionScenarioSet,
    OperationalContext,
    ReplenishmentDecision,
)


@runtime_checkable
class RepresentationModule(Protocol):
    """Transform admissible metadata into uncertain historical evidence."""

    def build_analog_space(
        self, target_id: str, metadata: pd.DataFrame
    ) -> CandidateAnalogSpace: ...


@runtime_checkable
class ScenarioConstructionModule(Protocol):
    """Transform uncertain analog evidence into operational futures."""

    def construct(
        self,
        analog_space: CandidateAnalogSpace,
        donor_histories: Mapping[str, NDArray[np.float64]],
        horizon: int,
        n_scenarios: int,
        random_state: int | np.random.Generator | None = None,
    ) -> DecisionScenarioSet: ...


@runtime_checkable
class DecisionOptimizationModule(Protocol):
    """Transform scenarios and context into a feasible action."""

    def optimize(
        self, scenarios: DecisionScenarioSet, context: OperationalContext
    ) -> ReplenishmentDecision: ...


@runtime_checkable
class DecisionEvaluationModule(Protocol):
    """Evaluate a frozen decision on a held-out realized outcome."""

    def evaluate(
        self,
        decision: ReplenishmentDecision,
        realized_demand: float,
        context: OperationalContext,
        reference_decision: ReplenishmentDecision | None = None,
    ) -> DecisionEvaluation: ...
