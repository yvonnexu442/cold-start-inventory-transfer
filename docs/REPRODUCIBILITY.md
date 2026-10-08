# Reproducibility Notes

This document records the public reproduction path for the released experiment
snapshot. It separates checks that can be run from included outputs from full
training and evaluation steps that require the original datasets.

## Quick Start From Included Outputs

The included aggregate outputs are enough to inspect the principal tables,
strict controls, search sensitivity, operating-condition summaries, cost
decomposition, and the de-identified cross-dataset evidence-increment inputs.

```bash
python -m pip install -e ".[dev,published-baseline]"
python scripts/plot_evidence_increment_synthesis.py
pytest --no-cov
```

The plotting command recomputes:

- `outputs/ai_darld_v3/evidence_increment_synthesis/relative_cost_contrasts.csv`
- `outputs/ai_darld_v3/evidence_increment_synthesis/bootstrap_draw_summary.csv`
- `outputs/ai_darld_v3/evidence_increment_synthesis/evidence_increment_forest.pdf`
- `outputs/ai_darld_v3/evidence_increment_synthesis/manifest.json`

It uses only
`outputs/ai_darld_v3/evidence_increment_synthesis/product_level_cost_inputs.csv`,
whose product identifiers are redacted.

## Full Training and Evaluation From Raw Data

Full reruns require local copies of the MAN/BRAF, Online Retail II, and
Corporacion Favorita datasets. The raw files are not redistributed here; see
`docs/dataset_download_instructions.md`.

The formal entry points are listed in `README.md`. For MAN/BRAF, place the
downloaded archive at
`data/raw/spdf/Spare-Part-Demand-Forecasting-main.zip`, then run
`python scripts/generate_service_parts_checkpoints.py --prepare-only` followed
by `python scripts/generate_service_parts_checkpoints.py --dataset all`. The
preparation step verifies `MAN.xlsx` and `BRAF.xls`, extracts them into the
parser's frozen interim directory, and refuses to overwrite different existing
content. The evaluation command reads
the source workbooks through the frozen `configs/full_scale.yaml` protocol and
writes `outputs/full_scale/checkpoints/man_reliability.parquet` and
`braf_reliability.parquet`, together with the companion results, calibration,
cutoff, scenario, and status files consumed downstream. Then run the service-
parts, retail, strict-control, and summary commands in the README order. The
service-parts sequence deliberately separates policy generation, global
baselines, validated combination, and final authority summarization. Every
combination checks complete evaluation keys, duplicate rows, realized demand,
methods, and the common operating grid.

The default strict-control run and the search-sensitivity runs use separate
output directories. Search sensitivity requires both service-parts and retail
runners at budgets 100 and 200 before its finalizer is called. The current
strict finalizer has no dependency on unpublished prior results; a prior
paired-contrast file may be supplied explicitly only when a version comparison
is desired.

Full execution requires the original datasets and can be computationally
substantial. Without those files, the included aggregate outputs and manifests
support the summary-and-figure path but do not constitute a full training rerun.

## Target-Time Decision Procedure

Donor-transfer policies produce a finite demand law at the target-time
information boundary. The common solver then evaluates feasible order
quantities under holding cost, shortage cost, fixed-order cost,
minimum-order quantity, capacity when present, and a zero-action option.

For MAN/BRAF, the global prediction baseline uses dataset-level quantile levels
to build an interpolated finite scenario law, with endpoint handling at the
grid edges, and passes that law into the same constrained solver. Online Retail
II and Favorita have no MOQ, capacity, or fixed-order cost in their frozen
protocols; their global baseline uses the cost-ratio critical quantile directly,
which is the corresponding unconstrained newsvendor action.

## Original Configuration Versus Strict Coefficient-Untying

The complete component-specific configuration and the matched shared
configuration differ in more than coefficient tying. Their comparison includes
the full design choices in the frozen configuration, including scoring features
and shrinkage behavior.

The strict coefficient-untying control isolates one narrower question: under a
common feature set, common contraction rule, common candidate protocol, and
common evaluation interface, what changes when occurrence and conditional-
positive relation coefficients are untied? The corresponding outputs are under
`outputs/ai_darld_v3/strict_shared_separate_corrected_v2/`.

## Search Budget and Actual Computation

The strict search-sensitivity run compares nominal candidate budgets while
recording the actual candidate accounting. The separate model has greater
effective parameter capacity, and tied candidates are shared through the nested
candidate stream. The search-sensitivity outputs are therefore interpreted as
finite-search sensitivity, not as a universal separation advantage.

Relevant files:

- `configs/strict_search_stability_v1.yaml`
- `outputs/ai_darld_v3/strict_search_stability_v1/ensemble_contrasts.csv`
- `outputs/ai_darld_v3/strict_search_stability_v1/seed_contrasts.csv`
- `outputs/ai_darld_v3/strict_search_stability_v1/fit_selection.csv`

## Product-Clustered Intervals

Reported paired intervals resample target products while retaining each
product's repeated evaluation contexts. They quantify uncertainty over the
sampled target products within the frozen retrospective grid. They do not
represent uncertainty over future organizations, deployment frequencies, or
unobserved datasets.

## Favorita Product-Identifier Correction

The Favorita execution uses the corrected item-identifier split and preserves
the pre-specified evaluation protocol: development, validation, and confirmation
roles are deterministic; raw competition data are not redistributed; and source
hashes are recorded in
`outputs/ai_darld_v3/favorita_confirmation_v1/run_manifest.json`.

The correction concerns product identity handling. It does not convert the
retrospective confirmation population into a real deployment study.

## Retrospective Evaluation and Reused Populations

All evaluations are retrospective pseudo-launch studies. They test how policies
would act at a zero-history cutoff using historical donors available before the
target outcome window. They do not validate an automatic policy selector or a
production deployment.

Some strict and sensitivity summaries reuse fixed populations to isolate the
comparison under study. This reuse is documented in the corresponding manifests
and should be considered when comparing intervals across result families.

## Dataset-Specific Global-Quantile Interfaces

The service-parts and retail protocols share the same holding--shortage
objective but use its appropriate computational interface. MAN/BRAF require a
scenario law for fixed cost, MOQ, capacity, and zero-action comparisons. Online
Retail II and Favorita use the objective's critical quantile directly because
those additional constraints are absent.

The final service-parts aggregate output is
`outputs/ai_darld_v3/historical_support_v4_authority_summary.csv`, with paired
intervals in `historical_support_v4_authority_comparisons.csv`.

## Online Retail II Result Identity

The current paper table is generated by `scripts/run_uci_confirmation_v2.py`
and released as `outputs/ai_darld_v3/uci_confirmation_v3_summary.csv`. The
matching `uci_confirmation_v3_split_manifest.csv`,
`uci_confirmation_v3_policy_selection.csv`, and
`uci_confirmation_v3_manifest.json` fix the isolated target cohort, fitted
policies, method set, and row counts. Paired differences are
component-specific minus comparator. Their percentile intervals use 2,000
deterministic draws (seed 20261103), resampling the target product as the
cluster and retaining its cutoff, horizon, and cost-ratio contexts.

`run_uci_online_retail_ii_external.py` constructs the predecessor population
and development roles required by the final run. Its released
`uci_online_retail_ii_external_summary.csv` documents that earlier evaluation;
it is not the source of the current paper table. Full row-level outputs are
regenerated from Online Retail II and are not redistributed.

## Service-Parts Result Lineage

The formal chain is:

1. `generate_service_parts_checkpoints.py` prepares the workbooks and local
   reliability checkpoints.
2. `run_ai_darld_v3_factorized.py --output-tag historical_support_v4` generates
   the component-specific, shared, and complete-transfer rows. MAN and BRAF are
   run separately and then combined with `--assemble-only`.
3. `run_ai_darld_v3_corrected_global.py` generates rolling global-quantile and
   adapted ZIG--MC rows.
4. `analyze_ai_darld_v3_corrected.py` validates and combines those families into
   an intermediate authority file.
5. `analyze_support_v3_similarity_residual.py` takes the final learned/complete
   rows from the historical-support run, adds only the validated global rows,
   and writes the authority parquet, main summary, and paired intervals.
6. `analyze_operational_cost_decomposition.py` and
   `analyze_operating_condition_input_quality.py` consume that authority file
   for the paper's cost and operating-condition interpretations.

Existing output tags are not reused silently: pass `--reuse-existing` only
after verifying their manifests and inputs. `--assemble-only` requires both
per-dataset outputs and performs no fitting.

## Public Snapshot Scope

Included:

- formal method code needed by the released scripts;
- scripts/configurations for the main baselines, learned transfer, strict
  control, search sensitivity, external populations, operating-condition
  summaries, cost decomposition, and walkthrough;
- aggregate outputs, manifests, and redacted product-level forest-plot inputs;
- tests that exercise the shared solver, relation laws, data adapters, result
  pairing, and released artifacts.

Excluded:

- raw datasets and third-party archives;
- manuscript LaTeX, PDFs, and paper build products;
- row-level outputs with public product identifiers or third-party competition
  item identifiers;
- development-only working records and obsolete exploratory branches.

The released UCI and Favorita role manifests retain public catalog item keys so
that the frozen product assignments can be reconstructed from the original
datasets. Other released product-level figure inputs use stable redacted keys.
