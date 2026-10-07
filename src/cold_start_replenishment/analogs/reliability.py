"""Transparent, leakage-controlled analog reliability estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss  # type: ignore[import-untyped]
from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from cold_start_replenishment.framework.objects import CandidateAnalogSpace

RELIABILITY_FEATURES = (
    "metadata_similarity",
    "donor_history_length_log",
    "donor_nonzero_count_log",
    "donor_nonzero_rate",
    "donor_adi_inverse",
    "donor_positive_cv2_log",
    "donor_stability",
    "pool_mean_dispersion",
    "top_similarity_gap",
    "group_support_log",
    "target_pool_distance",
    "secondary_similarity_agreement",
)

FEATURE_PROVENANCE = {
    "metadata_similarity": "admissible static metadata",
    "donor_history_length_log": "donor history before reliability training cutoff",
    "donor_nonzero_count_log": "donor history before reliability training cutoff",
    "donor_nonzero_rate": "donor history before reliability training cutoff",
    "donor_adi_inverse": "donor history before reliability training cutoff",
    "donor_positive_cv2_log": "donor history before reliability training cutoff",
    "donor_stability": "analytic resampling proxy from donor cutoff history",
    "pool_mean_dispersion": "candidate donor cutoff histories",
    "top_similarity_gap": "admissible static metadata candidate pool",
    "group_support_log": "admissible static metadata group",
    "target_pool_distance": "admissible static metadata candidate pool",
    "secondary_similarity_agreement": "two admissible static metadata definitions",
}


def donor_history_features(history: NDArray[np.float64]) -> dict[str, float]:
    """Compute reliability features using only a donor's visible history."""
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise ValueError("donor history must be a nonempty nonnegative vector")
    positive = values[values > 0]
    nonzero = len(positive)
    nonzero_rate = nonzero / len(values)
    adi_inverse = nonzero_rate
    if nonzero > 1 and positive.mean() > 0:
        cv2 = float((positive.std(ddof=1) / positive.mean()) ** 2)
    else:
        cv2 = 0.0
    mean = float(values.mean())
    stability = float(1.0 / (1.0 + values.std(ddof=0) / max(mean, 1e-8)))
    return {
        "donor_history_length_log": float(np.log1p(len(values))),
        "donor_nonzero_count_log": float(np.log1p(nonzero)),
        "donor_nonzero_rate": float(nonzero_rate),
        "donor_adi_inverse": float(adi_inverse),
        "donor_positive_cv2_log": float(np.log1p(cv2)),
        "donor_stability": stability,
    }


def reliability_pair_features(
    metadata_similarity: float,
    donor_history: NDArray[np.float64],
    pool_histories: NDArray[np.float64],
    top_similarity_gap: float,
    group_support: int,
    secondary_similarity: float,
) -> dict[str, float]:
    """Build one target–donor feature row without target outcomes."""
    donor = donor_history_features(donor_history)
    pool_means = np.asarray(pool_histories, dtype=float).mean(axis=1)
    dispersion = float(pool_means.std() / max(pool_means.mean(), 1e-8))
    return {
        "metadata_similarity": float(metadata_similarity),
        **donor,
        "pool_mean_dispersion": float(np.log1p(dispersion)),
        "top_similarity_gap": float(max(top_similarity_gap, 0.0)),
        "group_support_log": float(np.log1p(group_support)),
        "target_pool_distance": float(1.0 - metadata_similarity),
        "secondary_similarity_agreement": float(
            1.0 - abs(metadata_similarity - secondary_similarity)
        ),
    }


def expected_calibration_error(
    probabilities: NDArray[np.float64], labels: NDArray[np.float64], bins: int = 10
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(labels)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lower) & (
            (probabilities <= upper) if upper == 1 else (probabilities < upper)
        )
        if mask.any():
            value += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(value if total else np.nan)


@dataclass
class ReliabilityCalibrator:
    """Cross-fitted regularized logistic reliability model."""

    regularization_c: float = 1.0
    folds: int = 5
    random_seed: int = 20260806

    def __post_init__(self) -> None:
        self.model_: Any | None = None
        self.calibration_: dict[str, float] = {}
        self.cross_fitted_probabilities_: NDArray[np.float64] | None = None

    def _model(self) -> Any:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.regularization_c, max_iter=1000, random_state=self.random_seed
            ),
        )

    def fit(self, training: pd.DataFrame) -> ReliabilityCalibrator:
        required = [*RELIABILITY_FEATURES, "useful", "pseudo_target_id"]
        missing = [column for column in required if column not in training]
        if missing:
            raise ValueError(f"Missing reliability training columns: {missing}")
        if training["useful"].nunique() < 2:
            raise ValueError("Reliability supervision must contain both classes")
        x = training.loc[:, RELIABILITY_FEATURES].to_numpy(dtype=float)
        y = training["useful"].to_numpy(dtype=int)
        groups = training["pseudo_target_id"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        splits = min(self.folds, len(unique_groups))
        if splits < 2:
            raise ValueError("At least two pseudo-target groups are required")
        probabilities = np.zeros(len(training), dtype=float)
        for train_index, test_index in GroupKFold(splits).split(x, y, groups):
            model = self._model()
            model.fit(x[train_index], y[train_index])
            probabilities[test_index] = model.predict_proba(x[test_index])[:, 1]
        self.calibration_ = {
            "brier_score": float(brier_score_loss(y, probabilities)),
            "expected_calibration_error": expected_calibration_error(probabilities, y),
            "mean_probability": float(probabilities.mean()),
            "observed_usefulness_rate": float(y.mean()),
            "cross_fitted_rows": float(len(training)),
            "cross_fitted_pseudo_targets": float(len(unique_groups)),
        }
        self.cross_fitted_probabilities_ = np.asarray(probabilities, dtype=np.float64)
        self.model_ = self._model()
        self.model_.fit(x, y)
        return self

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        if self.model_ is None:
            raise RuntimeError("Fit ReliabilityCalibrator before prediction")
        values = features.loc[:, RELIABILITY_FEATURES].to_numpy(dtype=float)
        return np.asarray(self.model_.predict_proba(values)[:, 1], dtype=np.float64)


@dataclass(frozen=True)
class ReliabilityCalibratedRepresentation:
    """Enrich a metadata candidate pool with calibrated donor reliability."""

    calibrator: ReliabilityCalibrator
    low_reliability_threshold: float = 0.35
    no_close_similarity: float = 0.15

    def enrich(
        self,
        target_id: str,
        donor_ids: tuple[str, ...],
        similarities: NDArray[np.float64],
        feature_rows: pd.DataFrame,
        donor_support: NDArray[np.float64],
        diagnostics: dict[str, Any] | None = None,
    ) -> CandidateAnalogSpace:
        reliability = self.calibrator.predict(feature_rows)
        sample_size = max(int(self.calibrator.calibration_.get("cross_fitted_rows", 1)), 1)
        standard_error = np.sqrt(np.clip(reliability * (1 - reliability) / sample_size, 0, None))
        lower = np.clip(reliability - 1.96 * standard_error, 0, 1)
        upper = np.clip(reliability + 1.96 * standard_error, 0, 1)
        no_close = float(np.max(similarities)) < self.no_close_similarity
        all_low = bool(np.max(reliability) < self.low_reliability_threshold)
        fallback = "broad_fallback" if no_close or all_low else "reliability_weighted"
        return CandidateAnalogSpace(
            target_id,
            donor_ids,
            reliability,
            diagnostics or {},
            metadata_similarities=similarities,
            reliability_scores=reliability,
            reliability_lower=lower,
            reliability_upper=upper,
            donor_support=donor_support,
            feature_provenance=FEATURE_PROVENANCE,
            no_close_analog=no_close,
            fallback_recommendation=fallback,
        )
