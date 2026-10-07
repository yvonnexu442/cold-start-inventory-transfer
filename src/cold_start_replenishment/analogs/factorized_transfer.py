"""Evidence-adaptive transfer of lead-demand occurrence and positive magnitude."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cold_start_replenishment.analogs.direct_mixture_learning import MixtureDecisionTask
from cold_start_replenishment.inventory.newsvendor import optimal_feasible_level


def _zscore(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    return centered / np.where(scale < 1e-8, 1.0, scale)


def _softmax(score: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = np.clip(score - score.max(), -40.0, 0.0)
    weights = np.exp(shifted)
    return weights / weights.sum()


def factorized_weights(
    donor_features: NDArray[np.float64],
    parameters: NDArray[np.float64],
    *,
    donor_support: NDArray[np.float64] | None = None,
    shared: bool = False,
    contraction: bool = True,
    similarity_residual: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Return occurrence and positive-size weights with observable shrinkage.

    Feature column two is the raw number of positive visible history periods.
    It is transformed exactly once here.  ``donor_support`` separates visible
    periods, positive periods, effective horizon windows, and simulation atoms.
    When ``similarity_residual`` is enabled, both relations
    start from complete-similarity log weights and learn component-specific
    residual scores.
    """
    features = np.asarray(donor_features, float)
    theta = np.asarray(parameters, float)
    if features.ndim != 2 or features.shape[1] != 4 or theta.shape != (10,):
        raise ValueError("expected Kx4 features and ten parameters")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(theta)):
        raise ValueError("factorized transfer inputs must be finite")
    if np.any(features[:, 2] < 0):
        raise ValueError("raw donor support must be nonnegative")
    transformed = features.copy()
    transformed[:, 2] = np.log1p(transformed[:, 2])
    z = _zscore(transformed)
    occurrence_score = z[:, [0, 1, 2]] @ theta[:3]
    magnitude_score = z[:, [0, 2, 3]] @ theta[3:6]
    if similarity_residual:
        similarity = np.maximum(features[:, 0], 0.0)
        if similarity.sum() <= 0:
            similarity = np.ones(len(similarity))
        anchor = np.log(np.clip(similarity / similarity.sum(), 1e-8, 1.0))
        occurrence_score = occurrence_score + anchor
        magnitude_score = magnitude_score + anchor
    occurrence = _softmax(occurrence_score)
    magnitude = _softmax(magnitude_score)
    if shared:
        magnitude = occurrence.copy()
    if donor_support is None:
        occurrence_support = magnitude_support = float(np.mean(features[:, 2]))
    else:
        evidence = np.asarray(donor_support, float)
        if evidence.shape != (len(features), 4) or np.any(evidence < 0):
            raise ValueError("donor_support must be a nonnegative Kx4 matrix")
        # Occurrence has one Bernoulli observation per effective horizon window;
        # positive magnitude has one observation per positive visible period.
        occurrence_support = float(np.mean(evidence[:, 2]))
        magnitude_support = float(np.mean(evidence[:, 1]))
    occurrence_disagreement = float(np.std(features[:, 1]))
    size_level = np.maximum(features[:, 3], 0.0)
    size_disagreement = float(np.std(size_level) / max(np.mean(size_level), 1e-8))
    occurrence_logit = np.clip(
        theta[6] - theta[7] * np.log1p(occurrence_support) + occurrence_disagreement, -20, 20
    )
    magnitude_logit = np.clip(
        theta[8] - theta[9] * np.log1p(magnitude_support) + size_disagreement, -20, 20
    )
    alpha_occ = float(1 / (1 + np.exp(-occurrence_logit)))
    alpha_mag = float(1 / (1 + np.exp(-magnitude_logit)))
    if shared:
        alpha_mag = alpha_occ
    if not contraction:
        alpha_occ = alpha_mag = 0.0
    uniform = np.full(len(features), 1 / len(features))
    occurrence = (1 - alpha_occ) * occurrence + alpha_occ * uniform
    magnitude = (1 - alpha_mag) * magnitude + alpha_mag * uniform
    return occurrence / occurrence.sum(), magnitude / magnitude.sum(), alpha_occ, alpha_mag


def factorized_distribution(
    task: MixtureDecisionTask,
    parameters: NDArray[np.float64],
    occurrence_logit_shift: float = 0.0,
    *,
    shared: bool = False,
    contraction: bool = True,
    similarity_residual: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    """Build a lead-demand hurdle distribution without donor-index sampling."""
    occurrence_weights, magnitude_weights, alpha_occ, alpha_mag = factorized_weights(
        task.donor_features,
        parameters,
        donor_support=task.donor_support,
        shared=shared,
        contraction=contraction,
        similarity_residual=similarity_residual,
    )
    donor_event = np.mean(task.donor_scenarios > 0, axis=1)
    event_probability = float(np.clip(occurrence_weights @ donor_event, 0.0, 1.0))
    probability_logit = np.log(
        np.clip(event_probability, 1e-8, 1 - 1e-8) / np.clip(1 - event_probability, 1e-8, 1)
    )
    event_probability = float(
        1 / (1 + np.exp(-np.clip(probability_logit + occurrence_logit_shift, -20, 20)))
    )
    values = [0.0]
    masses = [1 - event_probability]
    active = [i for i, samples in enumerate(task.donor_scenarios) if np.any(samples > 0)]
    if active and event_probability > 0:
        active_weights = magnitude_weights[active]
        active_weights = active_weights / active_weights.sum()
        for donor, donor_mass in zip(active, active_weights, strict=True):
            positive_simulation_atoms = task.donor_scenarios[donor][task.donor_scenarios[donor] > 0]
            n_positive_simulation_atoms = len(positive_simulation_atoms)
            values.extend(positive_simulation_atoms.tolist())
            masses.extend(
                np.full(
                    n_positive_simulation_atoms,
                    event_probability * donor_mass / n_positive_simulation_atoms,
                ).tolist()
            )
    return (
        np.asarray(values),
        np.asarray(masses),
        {
            "event_probability": event_probability,
            "occurrence_shrinkage": alpha_occ,
            "magnitude_shrinkage": alpha_mag,
            "occurrence_effective_donors": float(1 / np.square(occurrence_weights).sum()),
            "magnitude_effective_donors": float(1 / np.square(magnitude_weights).sum()),
        },
    )


def strict_relation_weights(
    donor_features: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    *,
    shared: bool,
    contraction_alpha: float,
    similarity_residual: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Weights for the frozen coefficient-tying comparison.

    Both branches see the same four deployable features and use the same
    contraction coefficient.  The only model-class difference is whether the
    four coefficients are tied or estimated separately.
    """
    features = np.asarray(donor_features, float)
    beta = np.asarray(coefficients, float)
    expected = (4,) if shared else (8,)
    if features.ndim != 2 or features.shape[1] != 4 or beta.shape != expected:
        raise ValueError(f"expected Kx4 features and coefficients shaped {expected}")
    if not np.isfinite(features).all() or not np.isfinite(beta).all():
        raise ValueError("strict relation inputs must be finite")
    if np.any(features[:, 2] < 0) or not 0 <= contraction_alpha <= 1:
        raise ValueError("support must be nonnegative and contraction_alpha in [0,1]")
    transformed = features.copy()
    transformed[:, 2] = np.log1p(transformed[:, 2])
    design = _zscore(transformed)
    occurrence_beta = beta[:4]
    magnitude_beta = occurrence_beta if shared else beta[4:]
    occurrence_score = design @ occurrence_beta
    magnitude_score = design @ magnitude_beta
    if similarity_residual:
        similarity = np.maximum(features[:, 0], 0.0)
        if similarity.sum() <= 0:
            similarity = np.ones(len(similarity))
        anchor = np.log(np.clip(similarity / similarity.sum(), 1e-8, 1.0))
        occurrence_score += anchor
        magnitude_score += anchor
    occurrence = _softmax(occurrence_score)
    magnitude = _softmax(magnitude_score)
    uniform = np.full(len(features), 1.0 / len(features))
    occurrence = (1 - contraction_alpha) * occurrence + contraction_alpha * uniform
    magnitude = (1 - contraction_alpha) * magnitude + contraction_alpha * uniform
    return occurrence / occurrence.sum(), magnitude / magnitude.sum()


def strict_relation_distribution(
    task: MixtureDecisionTask,
    coefficients: NDArray[np.float64],
    occurrence_logit_shift: float = 0.0,
    *,
    shared: bool,
    contraction_alpha: float,
    similarity_residual: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    """Construct the hurdle law used by the strict nested comparison."""
    occurrence_weights, magnitude_weights = strict_relation_weights(
        task.donor_features,
        coefficients,
        shared=shared,
        contraction_alpha=contraction_alpha,
        similarity_residual=similarity_residual,
    )
    donor_event = np.mean(task.donor_scenarios > 0, axis=1)
    raw_event = float(np.clip(occurrence_weights @ donor_event, 0.0, 1.0))
    logit = np.log(np.clip(raw_event, 1e-8, 1 - 1e-8) / np.clip(1 - raw_event, 1e-8, 1))
    event_probability = float(1 / (1 + np.exp(-np.clip(logit + occurrence_logit_shift, -20, 20))))
    values, atom_donors, positive_counts = _strict_task_atoms(task)
    masses = np.zeros(len(values), float)
    masses[0] = 1 - event_probability
    if len(atom_donors) and event_probability > 0:
        active = positive_counts > 0
        active_weights = magnitude_weights[active]
        active_weights /= active_weights.sum()
        donor_mass = np.zeros(len(positive_counts), float)
        donor_mass[active] = active_weights
        masses[1:] = event_probability * donor_mass[atom_donors] / positive_counts[atom_donors]
    return (
        values,
        masses,
        {
            "event_probability": event_probability,
            "contraction_alpha": float(contraction_alpha),
            "occurrence_effective_donors": float(1 / np.square(occurrence_weights).sum()),
            "magnitude_effective_donors": float(1 / np.square(magnitude_weights).sum()),
        },
    )


_StrictAtomEntry = tuple[
    weakref.ReferenceType[MixtureDecisionTask],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]
_STRICT_ATOM_CACHE: dict[int, _StrictAtomEntry] = {}


def _drop_strict_task_cache(key: int, reference: weakref.ReferenceType[object]) -> None:
    """Remove only the cache entry created by the finalized task."""
    atom_entry = _STRICT_ATOM_CACHE.get(key)
    if atom_entry is not None and atom_entry[0] is reference:
        _STRICT_ATOM_CACHE.pop(key, None)
    order_entry = _STRICT_ORDER_CACHE.get(key)
    if order_entry is not None and order_entry[0] is reference:
        _STRICT_ORDER_CACHE.pop(key, None)


def _task_reference(task: MixtureDecisionTask, key: int) -> weakref.ReferenceType[MixtureDecisionTask]:
    return weakref.ref(task, lambda reference: _drop_strict_task_cache(key, reference))


def _strict_task_atoms_uncached(
    task: MixtureDecisionTask,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Reference atom construction without shared state."""
    positive_counts = np.sum(task.donor_scenarios > 0, axis=1).astype(np.int64)
    atom_blocks = []
    donor_blocks = []
    for donor, samples in enumerate(task.donor_scenarios):
        positive = np.asarray(samples[samples > 0], float)
        if len(positive):
            atom_blocks.append(positive)
            donor_blocks.append(np.full(len(positive), donor, dtype=np.int64))
    positive_atoms = np.concatenate(atom_blocks) if atom_blocks else np.empty(0, float)
    atom_donors = np.concatenate(donor_blocks) if donor_blocks else np.empty(0, np.int64)
    values = np.concatenate([np.zeros(1, float), positive_atoms])
    for array in (values, atom_donors, positive_counts):
        array.setflags(write=False)
    return values, atom_donors, positive_counts


def _strict_task_atoms(
    task: MixtureDecisionTask,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Cache task-stable atoms with identity validation and weak cleanup.

    ``MixtureDecisionTask`` is frozen.  The scenario array is made read-only on
    first use so the cached atom law cannot silently outlive an in-place input
    mutation.  Replacing the task creates a new object and therefore a new
    identity-checked entry; finalized tasks remove their own entries.
    """
    key = id(task)
    cached = _STRICT_ATOM_CACHE.get(key)
    if cached is not None and cached[0]() is task:
        return cached[1], cached[2], cached[3]
    task.donor_scenarios.setflags(write=False)
    values, atom_donors, positive_counts = _strict_task_atoms_uncached(task)
    reference = _task_reference(task, key)
    _STRICT_ATOM_CACHE[key] = (reference, values, atom_donors, positive_counts)
    return values, atom_donors, positive_counts


def strict_relation_loss(
    task: MixtureDecisionTask,
    coefficients: NDArray[np.float64],
    calibration_weight: float = 0.10,
    occurrence_logit_shift: float = 0.0,
    *,
    shared: bool,
    contraction_alpha: float,
    similarity_residual: bool,
) -> float:
    values, masses, diagnostics = strict_relation_distribution(
        task,
        coefficients,
        occurrence_logit_shift,
        shared=shared,
        contraction_alpha=contraction_alpha,
        similarity_residual=similarity_residual,
    )
    action = _strict_cached_action(task, values, masses)
    y = task.actual_demand
    decision = (
        task.holding_cost * max(action - y, 0.0)
        + task.holding_cost * task.cost_ratio * max(y - action, 0.0)
        + (task.fixed_order_cost if action > 0 else 0.0)
    )
    brier = (diagnostics["event_probability"] - float(y > 0)) ** 2
    positive_mass = masses[values > 0].sum()
    positive_mean = (
        float(np.sum(values[values > 0] * masses[values > 0]) / positive_mass)
        if positive_mass
        else 0.0
    )
    size_error = abs(positive_mean - y) / max(y, 1.0) if y > 0 else 0.0
    return float(decision / max(y, 1.0) + calibration_weight * (brier + size_error))


_StrictOrderEntry = tuple[
    weakref.ReferenceType[MixtureDecisionTask], NDArray[np.float64], NDArray[np.int64]
]
_STRICT_ORDER_CACHE: dict[int, _StrictOrderEntry] = {}


def _strict_action_uncached(
    task: MixtureDecisionTask, values: NDArray[np.float64], masses: NDArray[np.float64]
) -> float:
    """Reference action from the formal feasible-action solver."""
    return float(
        optimal_feasible_level(
            values,
            masses,
            task.holding_cost,
            task.holding_cost * task.cost_ratio,
            capacity=task.capacity,
            minimum_order_quantity=task.minimum_order_quantity,
            fixed_order_cost=task.fixed_order_cost,
        )[0]
    )


def _strict_cached_action(
    task: MixtureDecisionTask, values: NDArray[np.float64], masses: NDArray[np.float64]
) -> float:
    """Exact feasible action with task-stable atom ordering cached for search."""
    key = id(task)
    cached = _STRICT_ORDER_CACHE.get(key)
    if cached is None or cached[0]() is not task or not np.array_equal(cached[1], values):
        cached_values = values.copy()
        cached_values.setflags(write=False)
        order = np.argsort(values, kind="stable")
        order.setflags(write=False)
        reference = _task_reference(task, key)
        cached = (reference, cached_values, order)
        _STRICT_ORDER_CACHE[key] = cached
    order = cached[2]
    mass = masses / masses.sum()
    critical = task.cost_ratio / (1.0 + task.cost_ratio)
    index = min(
        int(np.searchsorted(np.cumsum(mass[order]), critical, side="left")), len(values) - 1
    )
    lower = 0.0 if task.minimum_order_quantity is None else float(task.minimum_order_quantity)
    upper = np.inf if task.capacity is None else float(task.capacity)
    candidates = [0.0]
    if upper > 0 and (task.minimum_order_quantity is None or upper >= lower):
        candidates.append(min(max(float(values[order[index]]), lower), upper))
    q = np.asarray(candidates)[:, None]
    holding = ((q - values).clip(min=0) * mass).sum(axis=1) * task.holding_cost
    shortage = ((values - q).clip(min=0) * mass).sum(axis=1) * task.holding_cost * task.cost_ratio
    fixed = (np.asarray(candidates) > 0) * task.fixed_order_cost
    objective = holding + shortage + fixed
    return float(
        candidates[
            int(np.flatnonzero(np.isclose(objective, objective.min(), rtol=1e-12, atol=1e-12))[0])
        ]
    )


def complete_mixture_distribution(
    task: MixtureDecisionTask, donor_weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    """Return the exact empirical mixture sum_j v_j F_j.

    Unlike a hurdle model with uniform conditional-positive weights, this
    construction automatically weights positive donor laws in proportion to
    ``v_j p_j`` and therefore preserves every donor's zero mass.
    """
    weights = np.asarray(donor_weights, float)
    if weights.shape != (len(task.donor_scenarios),) or np.any(weights < 0):
        raise ValueError("donor_weights must be a nonnegative K-vector")
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("donor_weights must have positive finite mass")
    weights = weights / weights.sum()
    n = task.donor_scenarios.shape[1]
    values = task.donor_scenarios.reshape(-1)
    masses = np.repeat(weights / n, n)
    event = float(masses[values > 0].sum())
    return (
        values,
        masses,
        {
            "event_probability": event,
            "occurrence_shrinkage": 0.0,
            "magnitude_shrinkage": 0.0,
            "occurrence_effective_donors": float(1 / np.square(weights).sum()),
            "magnitude_effective_donors": float(1 / np.square(weights).sum()),
        },
    )


def _complete_anchor(task: MixtureDecisionTask, anchor: str) -> NDArray[np.float64]:
    """Return deployable complete-mixture donor weights."""
    k = len(task.donor_scenarios)
    if anchor == "uniform":
        return np.full(k, 1.0 / k)
    if anchor != "similarity":
        raise ValueError("anchor must be 'uniform' or 'similarity'")
    similarity = np.maximum(np.asarray(task.donor_features[:, 0], float), 0.0)
    if similarity.sum() <= 0:
        return np.full(k, 1.0 / k)
    return similarity / similarity.sum()


def complete_preserving_weights(
    task: MixtureDecisionTask,
    parameters: NDArray[np.float64],
    *,
    anchor: str,
    shared: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Correct a complete mixture while nesting its exact hurdle representation.

    The occurrence anchor is the complete-mixture donor weight ``v`` and the
    conditional-positive anchor is proportional to ``v * p``.  In the shared
    model one residual corrects complete donor weights before both hurdle
    components are derived.  In the separate model the positive component has
    its own residual and contraction, both relative to the complete anchor.
    """
    theta = np.asarray(parameters, float)
    if theta.shape != (10,) or not np.isfinite(theta).all():
        raise ValueError("expected ten finite correction parameters")
    base = _complete_anchor(task, anchor)
    event = np.mean(np.asarray(task.donor_scenarios, float) > 0, axis=1)
    z = _zscore(np.asarray(task.donor_features, float))
    occurrence_learned = _softmax(np.log(np.clip(base, 1e-8, 1.0)) + z @ theta[:4])
    alpha_occ = float(1.0 / (1.0 + np.exp(-np.clip(theta[8], -20, 20))))
    occurrence = (1.0 - alpha_occ) * occurrence_learned + alpha_occ * base

    positive_anchor_mass = base * event
    if positive_anchor_mass.sum() <= 0:
        return occurrence / occurrence.sum(), np.zeros_like(base), alpha_occ, alpha_occ
    positive_anchor = positive_anchor_mass / positive_anchor_mass.sum()
    if shared:
        # A shared residual defines another complete donor mixture.  Derive its
        # conditional-positive weights after contracting complete donor weights.
        positive_mass = occurrence * event
        magnitude = positive_mass / positive_mass.sum()
        alpha_mag = alpha_occ
    else:
        magnitude_learned = _softmax(np.log(np.clip(positive_anchor, 1e-8, 1.0)) + z @ theta[4:8])
        alpha_mag = float(1.0 / (1.0 + np.exp(-np.clip(theta[9], -20, 20))))
        magnitude = (1.0 - alpha_mag) * magnitude_learned + alpha_mag * positive_anchor
    return occurrence / occurrence.sum(), magnitude / magnitude.sum(), alpha_occ, alpha_mag


def complete_preserving_distribution(
    task: MixtureDecisionTask,
    parameters: NDArray[np.float64],
    occurrence_logit_shift: float = 0.0,
    *,
    anchor: str,
    shared: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    """Build the corrected hurdle law, exactly nesting a complete mixture."""
    occurrence, magnitude, alpha_occ, alpha_mag = complete_preserving_weights(
        task, parameters, anchor=anchor, shared=shared
    )
    donor_event = np.mean(task.donor_scenarios > 0, axis=1)
    event_probability = float(occurrence @ donor_event)
    if occurrence_logit_shift:
        logit = np.log(
            np.clip(event_probability, 1e-8, 1 - 1e-8) / np.clip(1 - event_probability, 1e-8, 1)
        )
        event_probability = float(
            1 / (1 + np.exp(-np.clip(logit + occurrence_logit_shift, -20, 20)))
        )
    values = [0.0]
    masses = [1.0 - event_probability]
    if event_probability > 0 and magnitude.sum() > 0:
        for donor, donor_mass in enumerate(magnitude):
            positive = task.donor_scenarios[donor][task.donor_scenarios[donor] > 0]
            if donor_mass <= 0 or len(positive) == 0:
                continue
            values.extend(positive.tolist())
            masses.extend(
                np.full(len(positive), event_probability * donor_mass / len(positive)).tolist()
            )
    mass = np.asarray(masses, float)
    mass /= mass.sum()
    return (
        np.asarray(values, float),
        mass,
        {
            "event_probability": event_probability,
            "occurrence_shrinkage": alpha_occ,
            "magnitude_shrinkage": alpha_mag,
            "occurrence_effective_donors": float(1 / np.square(occurrence).sum()),
            "magnitude_effective_donors": (
                float(1 / np.square(magnitude).sum()) if magnitude.sum() > 0 else 0.0
            ),
        },
    )


def factorized_action(
    task: MixtureDecisionTask, parameters: NDArray[np.float64], occurrence_logit_shift: float = 0.0
) -> float:
    values, masses, _ = factorized_distribution(task, parameters, occurrence_logit_shift)
    action, _, _, _ = optimal_feasible_level(
        values,
        masses,
        task.holding_cost,
        task.holding_cost * task.cost_ratio,
        capacity=task.capacity,
        minimum_order_quantity=task.minimum_order_quantity,
        fixed_order_cost=task.fixed_order_cost,
    )
    return action


def factorized_loss(
    task: MixtureDecisionTask,
    parameters: NDArray[np.float64],
    calibration_weight: float = 0.10,
    occurrence_logit_shift: float = 0.0,
    *,
    shared: bool = False,
    contraction: bool = True,
    similarity_residual: bool = False,
) -> float:
    values, masses, diagnostics = factorized_distribution(
        task,
        parameters,
        occurrence_logit_shift,
        shared=shared,
        contraction=contraction,
        similarity_residual=similarity_residual,
    )
    action, _, _, _ = optimal_feasible_level(
        values,
        masses,
        task.holding_cost,
        task.holding_cost * task.cost_ratio,
        capacity=task.capacity,
        minimum_order_quantity=task.minimum_order_quantity,
        fixed_order_cost=task.fixed_order_cost,
    )
    decision = (
        task.holding_cost * max(action - task.actual_demand, 0.0)
        + task.holding_cost * task.cost_ratio * max(task.actual_demand - action, 0.0)
        + (task.fixed_order_cost if action > 0 else 0.0)
    )
    actual_event = float(task.actual_demand > 0)
    brier = (diagnostics["event_probability"] - actual_event) ** 2
    if actual_event and np.any(values > 0):
        positive_mean = float(np.sum(values * masses) / max(masses[values > 0].sum(), 1e-12))
        size_error = abs(positive_mean - task.actual_demand) / max(task.actual_demand, 1.0)
    else:
        size_error = 0.0
    return float(
        decision / max(task.actual_demand, 1.0) + calibration_weight * (brier + size_error)
    )


def complete_preserving_loss(
    task: MixtureDecisionTask,
    parameters: NDArray[np.float64],
    *,
    anchor: str,
    shared: bool,
    objective_mode: str,
    decision_scale: float,
    size_scale: float,
    calibration_weight: float,
    occurrence_logit_shift: float = 0.0,
) -> float:
    """Development loss for a complete-mixture-preserving correction."""
    values, masses, diagnostics = complete_preserving_distribution(
        task,
        parameters,
        occurrence_logit_shift,
        anchor=anchor,
        shared=shared,
    )
    action = optimal_feasible_level(
        values,
        masses,
        task.holding_cost,
        task.holding_cost * task.cost_ratio,
        capacity=task.capacity,
        minimum_order_quantity=task.minimum_order_quantity,
        fixed_order_cost=task.fixed_order_cost,
    )[0]
    y = task.actual_demand
    decision = (
        task.holding_cost * max(action - y, 0.0)
        + task.holding_cost * task.cost_ratio * max(y - action, 0.0)
        + (task.fixed_order_cost if action > 0 else 0.0)
    )
    if objective_mode == "realized_y":
        cost_denominator = max(y, 1.0)
        size_denominator = max(y, 1.0)
    elif objective_mode == "training_scale":
        cost_denominator = max(decision_scale, 1e-8)
        size_denominator = max(size_scale, 1e-8)
    else:
        raise ValueError("objective_mode must be 'realized_y' or 'training_scale'")
    brier = (diagnostics["event_probability"] - float(y > 0)) ** 2
    positive_mass = masses[values > 0].sum()
    positive_mean = (
        float(np.sum(values[values > 0] * masses[values > 0]) / positive_mass)
        if positive_mass > 0
        else 0.0
    )
    size_error = abs(positive_mean - y) / size_denominator if y > 0 else 0.0
    return float(decision / cost_denominator + calibration_weight * (brier + size_error))


@dataclass
class CompleteMixtureCorrectionPolicy:
    """Select a correction nested around uniform or similarity complete mixtures."""

    random_seed: int
    candidate_count: int = 24
    shared: bool = False
    objective_mode: str = "training_scale"
    calibration_weight_grid: tuple[float, ...] = (0.0, 0.05, 0.10)
    anchors: tuple[str, ...] = ("uniform", "similarity")
    shortlist_size: int = 12
    joint_validation_calibration: bool = False
    exact_anchor_fallback: bool = False
    occurrence_shift_grid: tuple[float, ...] = tuple(np.linspace(-1.5, 1.5, 13))

    def __post_init__(self) -> None:
        self.parameters_: NDArray[np.float64] | None = None
        self.anchor_: str | None = None
        self.calibration_weight_: float | None = None
        self.occurrence_logit_shift_: float = 0.0
        self.decision_scale_: float = 1.0
        self.size_scale_: float = 1.0
        self.search_: list[dict[str, float | int | bool | str]] = []

    def candidates(self) -> NDArray[np.float64]:
        rng = np.random.default_rng(self.random_seed)
        random = rng.normal(0.0, 0.65, size=(self.candidate_count, 10))
        random[:, 8:10] -= 0.5
        return np.vstack([np.zeros((1, 10)), random])

    def fit(
        self,
        training: list[MixtureDecisionTask],
        validation: list[MixtureDecisionTask],
    ) -> CompleteMixtureCorrectionPolicy:
        if not training or not validation:
            raise ValueError("nonempty training and validation tasks are required")
        zero_cost = [
            task.holding_cost * task.cost_ratio * task.actual_demand
            for task in training
            if task.actual_demand > 0
        ]
        positive_demand = [task.actual_demand for task in training if task.actual_demand > 0]
        self.decision_scale_ = float(np.median(zero_cost)) if zero_cost else 1.0
        self.size_scale_ = float(np.median(positive_demand)) if positive_demand else 1.0
        candidates = self.candidates()
        records: list[dict[str, float | int | bool | str]] = []
        finalists: list[dict[str, float | int | bool | str]] = []
        for anchor in self.anchors:
            for calibration_weight in self.calibration_weight_grid:
                group = []
                for index, parameters in enumerate(candidates):
                    value = float(
                        np.mean(
                            [
                                complete_preserving_loss(
                                    task,
                                    parameters,
                                    anchor=anchor,
                                    shared=self.shared,
                                    objective_mode=self.objective_mode,
                                    decision_scale=self.decision_scale_,
                                    size_scale=self.size_scale_,
                                    calibration_weight=calibration_weight,
                                )
                                for task in training
                            ]
                        )
                    )
                    row: dict[str, float | int | bool | str] = {
                        "anchor": anchor,
                        "calibration_weight": calibration_weight,
                        "candidate": index,
                        "training_objective": value,
                        "occurrence_logit_shift": 0.0,
                        "exact_anchor_available": bool(self.exact_anchor_fallback and index == 0),
                    }
                    records.append(row)
                    group.append(row)
                shortlist = sorted(group, key=lambda row: float(row["training_objective"]))[
                    : self.shortlist_size
                ]
                if self.exact_anchor_fallback and not any(
                    int(row["candidate"]) == 0 for row in shortlist
                ):
                    shortlist.append(next(row for row in group if int(row["candidate"]) == 0))
                for row in shortlist:
                    index = int(row["candidate"])
                    shifts = (
                        self.occurrence_shift_grid if self.joint_validation_calibration else (0.0,)
                    )
                    for shift in shifts:
                        # Candidate zero with zero shift is the exact complete
                        # mixture anchor.  Keeping it in validation prevents a
                        # post-hoc calibration step from silently removing the
                        # strongest simple policy from the learned policy set.
                        validation_value = float(
                            np.mean(
                                [
                                    complete_preserving_loss(
                                        task,
                                        candidates[index],
                                        anchor=anchor,
                                        shared=self.shared,
                                        objective_mode=self.objective_mode,
                                        decision_scale=self.decision_scale_,
                                        size_scale=self.size_scale_,
                                        calibration_weight=calibration_weight,
                                        occurrence_logit_shift=float(shift),
                                    )
                                    for task in validation
                                ]
                            )
                        )
                        finalist = dict(row)
                        finalist["validation_objective"] = validation_value
                        finalist["occurrence_logit_shift"] = float(shift)
                        finalists.append(finalist)
        best = min(finalists, key=lambda row: float(row["validation_objective"]))
        self.parameters_ = candidates[int(best["candidate"])].copy()
        self.anchor_ = str(best["anchor"])
        self.calibration_weight_ = float(best["calibration_weight"])
        if self.joint_validation_calibration:
            self.occurrence_logit_shift_ = float(best["occurrence_logit_shift"])
        else:
            shifts = np.asarray(self.occurrence_shift_grid, float)
            brier = []
            for shift in shifts:
                errors = []
                for task in validation:
                    diagnostics = complete_preserving_distribution(
                        task,
                        self.parameters_,
                        float(shift),
                        anchor=self.anchor_,
                        shared=self.shared,
                    )[2]
                    errors.append(
                        (diagnostics["event_probability"] - float(task.actual_demand > 0)) ** 2
                    )
                brier.append(float(np.mean(errors)))
            self.occurrence_logit_shift_ = float(shifts[int(np.argmin(brier))])
        for row in records:
            row["selected"] = (
                row["anchor"] == self.anchor_
                and int(row["candidate"]) == int(best["candidate"])
                and float(row["calibration_weight"]) == self.calibration_weight_
            )
            if row["selected"]:
                row["validation_objective"] = float(best["validation_objective"])
                row["occurrence_logit_shift"] = self.occurrence_logit_shift_
        self.search_ = records
        return self


@dataclass
class FactorizedTransferPolicy:
    random_seed: int
    candidate_count: int = 24
    calibration_weight: float = 0.10
    shared: bool = False
    contraction: bool = True
    similarity_residual: bool = False

    def __post_init__(self) -> None:
        self.parameters_: NDArray[np.float64] | None = None
        self.occurrence_logit_shift_: float = 0.0
        self.search_: list[dict[str, float | int | bool]] = []

    def candidates(self) -> NDArray[np.float64]:
        rng = np.random.default_rng(self.random_seed)
        random = rng.normal(0, 0.8, size=(self.candidate_count, 10))
        random[:, [7, 9]] = np.abs(random[:, [7, 9]])
        anchors = np.zeros((4, 10))
        anchors[0, [6, 8]] = 20.0  # uniform occurrence and magnitude
        anchors[1, [0, 3]] = 1.0  # shared similarity
        anchors[1, [6, 8]] = -20.0
        anchors[2, 1] = 1.0  # occurrence frequency; uniform magnitude
        anchors[2, 6] = -20.0
        anchors[2, 8] = 20.0
        anchors[3, 0] = 1.0  # similarity occurrence; expected-size magnitude
        anchors[3, 5] = 1.0
        anchors[3, [6, 8]] = -20.0
        return np.vstack([anchors, random])

    def fit(
        self, training: list[MixtureDecisionTask], validation: list[MixtureDecisionTask]
    ) -> FactorizedTransferPolicy:
        candidates = self.candidates()
        records = []
        for index, parameters in enumerate(candidates):
            train = float(
                np.mean(
                    [
                        factorized_loss(
                            task,
                            parameters,
                            self.calibration_weight,
                            shared=self.shared,
                            contraction=self.contraction,
                            similarity_residual=self.similarity_residual,
                        )
                        for task in training
                    ]
                )
            )
            records.append({"candidate": index, "training_objective": train})
        shortlist = sorted(records, key=lambda row: float(row["training_objective"]))[:12]
        best, best_value = -1, np.inf
        for row in shortlist:
            index = int(row["candidate"])
            value = float(
                np.mean(
                    [
                        factorized_loss(
                            task,
                            candidates[index],
                            self.calibration_weight,
                            shared=self.shared,
                            contraction=self.contraction,
                            similarity_residual=self.similarity_residual,
                        )
                        for task in validation
                    ]
                )
            )
            row["validation_objective"] = value
            if value < best_value:
                best, best_value = index, value
        self.parameters_ = candidates[best].copy()
        shifts = np.linspace(-1.5, 1.5, 13)
        brier = []
        for shift in shifts:
            errors = []
            for task in validation:
                _, _, diagnostics = factorized_distribution(
                    task,
                    self.parameters_,
                    float(shift),
                    shared=self.shared,
                    contraction=self.contraction,
                    similarity_residual=self.similarity_residual,
                )
                errors.append(
                    (diagnostics["event_probability"] - float(task.actual_demand > 0)) ** 2
                )
            brier.append(float(np.mean(errors)))
        self.occurrence_logit_shift_ = float(shifts[int(np.argmin(brier))])
        for row in records:
            row["selected"] = int(row["candidate"]) == best
        self.search_ = records
        return self


@dataclass
class StrictRelationPolicy:
    """Frozen nested policy for coefficient tying versus untying."""

    random_seed: int
    shared: bool
    similarity_residual: bool
    candidate_count: int = 96
    shortlist_size: int = 12
    calibration_weight: float = 0.10
    contraction_grid: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
    search_budget: int | None = None
    nested_candidate_stream: bool = False

    def __post_init__(self) -> None:
        self.coefficients_: NDArray[np.float64] | None = None
        self.contraction_alpha_: float | None = None
        self.occurrence_logit_shift_: float = 0.0
        self.search_: list[dict[str, float | int | bool]] = []

    @property
    def effective_relation_parameters(self) -> int:
        return 4 if self.shared else 8

    @property
    def effective_continuous_parameters(self) -> int:
        return self.effective_relation_parameters + 1  # validation-fitted logit intercept

    def candidates(self) -> NDArray[np.float64]:
        if self.nested_candidate_stream:
            budget = 100 if self.search_budget is None else int(self.search_budget)
            if budget not in {100, 200}:
                raise ValueError("nested search_budget must be 100 or 200")
            unique_tied_count = budget // 2
            fixed = np.vstack([np.zeros(4), np.eye(4)[:3]])
            tied_rng = np.random.default_rng(self.random_seed)
            tied_random = tied_rng.normal(0.0, 0.8, size=(unique_tied_count - len(fixed), 4))
            tied = np.vstack([fixed, tied_random])
            if self.shared:
                # Equal candidate-slot accounting is maintained by a declared
                # duplicate pass.  Shortlisting de-duplicates coefficient
                # vectors, so duplicates cannot crowd out a distinct policy.
                return np.vstack([tied, tied])
            untied_rng = np.random.default_rng(self.random_seed + 10_000_019)
            untied = untied_rng.normal(0.0, 0.8, size=(budget - len(tied), 8))
            return np.vstack([np.hstack([tied, tied]), untied])
        rng = np.random.default_rng(self.random_seed)
        base = np.vstack([np.zeros(4), np.eye(4)[:3]])
        if self.shared:
            random = rng.normal(0.0, 0.8, size=(self.candidate_count, 4))
            return np.vstack([base, random])
        fixed = np.hstack([base, base])
        random = rng.normal(0.0, 0.8, size=(self.candidate_count, 8))
        return np.vstack([fixed, random])

    def fit(
        self, training: list[MixtureDecisionTask], validation: list[MixtureDecisionTask]
    ) -> StrictRelationPolicy:
        if not training or not validation:
            raise ValueError("nonempty training and validation tasks are required")
        candidates = self.candidates()
        records: list[dict[str, float | int | bool]] = []
        finalists: list[dict[str, float | int | bool]] = []
        for alpha in self.contraction_grid:
            group = []
            objective_cache: dict[bytes, float] = {}
            for index, coefficients in enumerate(candidates):
                signature = coefficients.tobytes()
                train = objective_cache.get(signature)
                if train is None:
                    train = float(
                        np.mean(
                            [
                                strict_relation_loss(
                                    task,
                                    coefficients,
                                    self.calibration_weight,
                                    shared=self.shared,
                                    contraction_alpha=alpha,
                                    similarity_residual=self.similarity_residual,
                                )
                                for task in training
                            ]
                        )
                    )
                    objective_cache[signature] = train
                row: dict[str, float | int | bool] = {
                    "candidate": index,
                    "contraction_alpha": alpha,
                    "training_objective": train,
                }
                records.append(row)
                group.append(row)
            ordered = sorted(group, key=lambda row: float(row["training_objective"]))
            if self.nested_candidate_stream:
                unique = []
                seen: set[bytes] = set()
                for row in ordered:
                    signature = candidates[int(row["candidate"])].tobytes()
                    if signature not in seen:
                        seen.add(signature)
                        unique.append(row)
                    if len(unique) == self.shortlist_size:
                        break
                finalists.extend(unique)
            else:
                finalists.extend(ordered[: self.shortlist_size])
        best: dict[str, float | int | bool] | None = None
        best_value = np.inf
        for row in finalists:
            index = int(row["candidate"])
            alpha = float(row["contraction_alpha"])
            value = float(
                np.mean(
                    [
                        strict_relation_loss(
                            task,
                            candidates[index],
                            self.calibration_weight,
                            shared=self.shared,
                            contraction_alpha=alpha,
                            similarity_residual=self.similarity_residual,
                        )
                        for task in validation
                    ]
                )
            )
            row["validation_objective"] = value
            if value < best_value:
                best, best_value = row, value
        assert best is not None
        self.coefficients_ = candidates[int(best["candidate"])].copy()
        self.contraction_alpha_ = float(best["contraction_alpha"])
        shifts = np.linspace(-1.5, 1.5, 13)
        brier = []
        for shift in shifts:
            brier.append(
                float(
                    np.mean(
                        [
                            (
                                strict_relation_distribution(
                                    task,
                                    self.coefficients_,
                                    float(shift),
                                    shared=self.shared,
                                    contraction_alpha=self.contraction_alpha_,
                                    similarity_residual=self.similarity_residual,
                                )[2]["event_probability"]
                                - float(task.actual_demand > 0)
                            )
                            ** 2
                            for task in validation
                        ]
                    )
                )
            )
        self.occurrence_logit_shift_ = float(shifts[int(np.argmin(brier))])
        for row in records:
            row["selected"] = (
                int(row["candidate"]) == int(best["candidate"])
                and float(row["contraction_alpha"]) == self.contraction_alpha_
            )
            if row["selected"]:
                row["validation_objective"] = best_value
                row["occurrence_logit_shift"] = self.occurrence_logit_shift_
        self.search_ = records
        return self
