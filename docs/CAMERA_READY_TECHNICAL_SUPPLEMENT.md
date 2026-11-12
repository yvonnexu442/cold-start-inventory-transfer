# AI-DARLD 2026 Camera-Ready Technical Supplement

This document is the stable public destination for method details and auxiliary results that are necessary for interpretation or reproduction but would interrupt the paper's decision-centered narrative. It describes the final camera-ready evidence; it is not a history of manuscript versions.

## 1. Policy families and demand construction

The paper compares three policy families.

- **Global prediction** fits a mapping from the historical product panel and target metadata to demand quantiles or a two-part parametric demand law. It does not retrieve ten donors for each target.
- **Complete transfer** retrieves ten eligible historical products and retains each donor's complete empirical lead-demand distribution. Uniform and metadata-similarity weights form mixture laws.
- **Learned transfer** retrieves the same type of donor pool but learns how donor information should be weighted. The shared policy uses one learned mapping for occurrence and positive size before donors without positive samples are removed and the remaining positive-size weights are renormalized. The component-specific policy learns separate mappings.

For learned transfer, let `raw_p` be the occurrence probability obtained from donor occurrence rates. The reported policy applies a validation-fitted logit intercept before constructing the hurdle law:

```text
p = logistic(logit(raw_p) + occurrence_logit_shift)
F = (1 - p) * point_mass_at_zero + p * positive_size_mixture
```

Positive demand size is realized demand conditional on demand being positive. It is not the inventory order. The order is selected later by the common constrained optimizer.

With the similarity-residual parameterization, learned residual scores are added to clipped and renormalized similarity log weights. A zero residual therefore recovers the pre-shrinkage similarity anchor. Subsequent evidence-dependent shrinkage can still move the final weights toward uniform.

## 2. Fitting and decision objective

Historical products are hidden in turn to form leave-one-product-out training tasks. Evaluation products are absent from retrieval, fitting, preprocessing, calibration, and model selection. Five fitted policies are ensembled. The learned objective combines normalized inventory cost with occurrence and positive-size probability terms. Candidate counts, seeds, and dataset-specific settings are frozen in `configs/ai_darld_v3.yaml` and the result-family configurations.

All policies ultimately minimize the same holding, shortage, and fixed-order objective over the feasible order set, including zero, MOQ, and capacity where available. A service-calibration multiplier changes only the shortage penalty used to choose an order. Evaluation always uses the original operating costs.

## 3. Fixed-policy service-calibration experiment

### Question

For one fitted component-specific BRAF policy, how does validation-selected service calibration change its order, service, and original-objective cost, and how does the calibrated procedure compare with complete similarity?

### Data roles and frozen computation

- 60 historical pseudo-target products per cutoff fit the component-specific relation.
- The next 20 products in a frozen outcome-independent permutation select the shortage multiplier.
- 250 further products are used only for evaluation.
- Five seeds search 24 random parameter vectors plus four fixed anchors.
- Candidate multipliers are `1, 1.25, 1.5, 2, 3, 4, 6, 8`.
- Calibration is performed separately by cutoff and shortage/holding ratio while pooling the two lead-time and three capacity regimes.
- The service metric is the task-level mean served-demand proxy: 1 for zero realized demand and `min(order, demand) / demand` otherwise. It is not a pooled fill rate.
- The selected multiplier is the lowest-original-cost feasible multiplier whose calibration service reaches complete similarity's calibration service. All nine cells are feasible.
- Evaluation intervals resample the 250 target products and retain all repeated operating contexts; 4,000 deterministic draws use seed 20261112.

### Results

| Policy | Mean original cost | Service | Mean order | Holding | Shortage | Fixed |
|---|---:|---:|---:|---:|---:|---:|
| Component-specific, multiplier 1 | 22.07 | .808 | 7.26 | 5.37 | 16.69 | 0.00 |
| Same fitted policy, validation-selected multiplier | 25.86 | .829 | 12.17 | 10.99 | 14.87 | 0.00 |
| Complete similarity, multiplier 1 | 24.05 | .825 | 11.41 | 8.93 | 15.12 | 0.00 |

Calibration of the fixed learned policy increases its mean order by 4.91 units (95% product-clustered interval `[4.20, 5.69]`) and service by .0216 (`[.0186, .0247]`). Scored at the original operating costs, it increases cost by 3.80 (`[1.25, 7.87]`): holding cost rises more than shortage cost falls. Against complete similarity, the service-calibrated learned policy has service higher by .0041 (`[.0019, .0064]`) and cost higher by 1.81 (`[.06, 4.98]`).

This is a retrospective exploratory analysis because the BRAF evaluation population had already been examined during study development. Evaluation outcomes do not select the fitted model, multiplier, or service target. The experiment uses a smaller fitting population and search budget than the main BRAF comparison; only contrasts within this experiment are interpretable.

Files:

- `configs/service_calibration_fixed_policy_braf_v2.yaml`
- `scripts/run_matched_service_braf_exploratory.py`
- `outputs/ai_darld_v3/service_calibration_fixed_policy_braf_v2/summary.csv`
- `outputs/ai_darld_v3/service_calibration_fixed_policy_braf_v2/paired_comparison.csv`
- `outputs/ai_darld_v3/service_calibration_fixed_policy_braf_v2/calibration_selection.csv`
- `outputs/ai_darld_v3/service_calibration_fixed_policy_braf_v2/manifest.json`

## 4. Targeted component checks

The existing paired BRAF and MAN runs cover only part of the requested component analysis.

| Component check | Fit type | BRAF full vs removal: cost / service | MAN full vs removal: cost / service | Full-minus-removal cost interval | Interpretation |
|---|---|---:|---:|---:|---|
| Similarity anchor | separately fitted comparison | 23.14/.806 vs 23.12/.806 | 340.31/.615 vs 340.39/.615 | BRAF 0.02 `[-.14,.20]`; MAN -0.08 `[-.33,.13]` | No clear cost change in either dataset. |
| Shrinkage toward uniform | removal refitted with the similarity anchor | 23.14/.806 vs 22.85/.802 | 340.31/.615 vs 345.26/.601 | BRAF 0.29 `[.05,.55]`; MAN -4.95 `[-6.31,-3.81]` | Removing shrinkage slightly lowers BRAF cost but raises MAN cost; the direction is not universal. |
| Occurrence-logit calibration | fixed fitted relation, intercept set to zero | 23.14/.806 vs 23.06/.800 | 340.31/.615 vs 344.46/.585 | BRAF 0.14 `[-.06,.36]`; MAN -4.17 `[-5.27,-3.15]` | Little BRAF cost change; calibration is associated with lower MAN cost and higher service. |

Differences are component-specific full policy minus the stated comparator, so negative favors the full policy. Full costs and service are in `historical_support_v4_authority_summary.csv`; paired intervals are in `historical_support_v4_authority_comparisons.csv`.

The similarity-anchor comparison is not a fixed-parameter removal, and the shrinkage comparison is a refit under the ablated model. They therefore include optimization variability. Five fitted seeds are represented in the policy ensembles, but no seed-wise ablation interval was consolidated for these checks. No unified auxiliary-loss-only refit has been completed. These one-at-a-time checks do not identify component interactions or independent causal contributions.

## 5. Search stability and strict coefficient sharing

The full matched shared/component-specific comparison changes the relation design as a whole. The strict control holds the feature set, similarity anchor, shrinkage, occurrence calibration, loss, donor pools, seeds, and solver fixed and changes only coefficient sharing. Absolute costs, service, paired differences, seed results, and search-budget results are under:

- `outputs/ai_darld_v3/strict_shared_separate_corrected_v2/`
- `outputs/ai_darld_v3/strict_search_stability_v1/`

The direction changes across budgets on MAN and BRAF and varies across seeds. The strict evidence therefore supports a selective, search-sensitive benefit rather than a stable advantage.

## 6. Temporal dependence sensitivity

The main learned-transfer sampler independently resamples cutoff-visible periods and therefore assumes exchangeability. An earlier diagnostic for a different complete-transfer policy family compared IID occurrence/size sampling with contiguous donor blocks, but it does not evaluate the final component-specific hurdle law and is not used as evidence that the learned policy is robust to dependence.

A valid component-specific dependence analysis must specify how separate occurrence and positive-size donor relations map to a time-ordered block without silently replacing the learned law with one donor-weight vector. That analysis was not run. Autocorrelation and seasonality remain unresolved limitations; no claim of temporal robustness is made.

## 7. Result identities

- Main service-parts results: `historical_support_v4_authority_*`.
- Online Retail II: `uci_confirmation_v3_*`; earlier `uci_online_retail_ii_external_*` files are not the paper table.
- Favorita: `favorita_confirmation_v1/` with corrected item identities.
- Strict coefficient-sharing: `strict_shared_separate_corrected_v2/`, not the earlier strict v1 outputs.
- Fixed-policy service calibration: `service_calibration_fixed_policy_braf_v2/`.

Numbers from these result families must not be exchanged merely because rounded costs appear similar.
