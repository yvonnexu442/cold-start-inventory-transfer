# Cold-Start Inventory Transfer

This repository is a public reproducibility snapshot for experiments on initial
inventory decisions when a target product has no demand history. The central
question is how metadata and cutoff-valid historical products can be used to
construct demand representations that feed a common constrained inventory
decision layer.

The experiments compare three evidence strategies under the same target-time
information boundary:

- `global prediction`: a dataset-level quantile grid is converted to demand
  scenarios and passed to the common solver.
- `complete donor distribution transfer`: empirical donor demand distributions
  are transferred as complete mixture laws.
- `learned transfer`: occurrence and conditional-positive donor relations are
  learned on historical pseudo-launches, then converted to the same decision
  interface.

The repository contains code, frozen configurations, aggregate outputs,
manifests, and tests needed to inspect the reported results. It does not contain
the manuscript source, PDFs, or raw third-party datasets. Row-level evaluation
outputs derived from the source datasets remain local.

## Data

Raw data must be downloaded from the original sources and kept under `data/raw/`
according to `docs/dataset_download_instructions.md`.

- MAN and BRAF service-parts workbooks come from the Spare-Part Demand
  Forecasting benchmark.
- Online Retail II comes from the UCI Machine Learning Repository.
- Corporacion Favorita comes from the official Kaggle competition and is not
  redistributed here.

The included outputs are aggregate artifacts and compact figure inputs suitable
for checking the published tables and plots. The forest-plot product keys are
redacted. The UCI and Favorita split manifests retain the source-dataset product
keys needed to reconstruct frozen roles; these are catalog identifiers, not
personal identifiers. Full training and row-level evaluation require the
original data.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,published-baseline]"
```

## Quick Start From Included Outputs

Rebuild the public evidence-increment table and forest-plot data from the
included de-identified product-level inputs:

```bash
python scripts/plot_evidence_increment_synthesis.py
```

Inspect the main aggregate outputs:

```bash
python - <<'PY'
import pandas as pd

for path in [
    "outputs/ai_darld_v3/corrected_method_summary.csv",
    "outputs/ai_darld_v3/historical_support_v4_authority_summary.csv",
    "outputs/ai_darld_v3/strict_shared_separate_corrected_v2/summary.csv",
    "outputs/ai_darld_v3/strict_search_stability_v1/ensemble_contrasts.csv",
    "outputs/ai_darld_v3/operational_cost_decomposition_v1/summary.csv",
]:
    print(f"\n== {path} ==")
    print(pd.read_csv(path).head())
PY
```

This path checks and summarizes the released outputs; it does not rerun model
training.

## Full Training and Evaluation

After placing the raw data as documented, run the frozen entry points in this
order. The first command reconstructs the local MAN/BRAF checkpoint files used
by the subsequent service-parts scripts.

```bash
# Extract MAN.xlsx/BRAF.xls from the downloaded ZIP and validate the paths
python scripts/generate_service_parts_checkpoints.py --prepare-only

# MAN/BRAF source preparation and local reliability checkpoints
python scripts/generate_service_parts_checkpoints.py --dataset all

# MAN/BRAF component-specific transfer and complete-distribution baselines.
# Per-dataset runs are explicit; assembly never silently reuses stale files.
python scripts/run_ai_darld_v3_factorized.py --dataset MAN --output-tag historical_support_v4
python scripts/run_ai_darld_v3_factorized.py --dataset BRAF --output-tag historical_support_v4
python scripts/run_ai_darld_v3_factorized.py --output-tag historical_support_v4 --assemble-only

# Rolling global-quantile and adapted ZIG--MC rows
python scripts/run_ai_darld_v3_corrected_global.py

# Validated result combination and the final service-parts authority summaries
python scripts/analyze_ai_darld_v3_corrected.py
python scripts/analyze_support_v3_similarity_residual.py

# Supplementary direct-mixture run
python scripts/run_ai_darld_v3_direct_mixture.py

# Online Retail II population construction, final isolated-product evaluation,
# and Favorita external population
python scripts/run_uci_online_retail_ii_external.py
python scripts/run_uci_confirmation_v2.py
python scripts/run_favorita_untouched_confirmation.py
python scripts/analyze_favorita_confirmation.py

# Default strict coefficient-untying control
python scripts/run_strict_shared_separate_v1.py
python scripts/run_strict_shared_separate_retail_v1.py
python scripts/finalize_strict_shared_separate_v1.py

# Search sensitivity: service-parts and retail outputs for both frozen budgets
python scripts/run_strict_shared_separate_v1.py --search-budget 100
python scripts/run_strict_shared_separate_retail_v1.py --search-budget 100
python scripts/run_strict_shared_separate_v1.py --search-budget 200
python scripts/run_strict_shared_separate_retail_v1.py --search-budget 200
python scripts/finalize_strict_search_stability_v1.py

# Operating-condition, cost-decomposition, and walkthrough summaries
python scripts/analyze_operating_condition_input_quality.py
python scripts/analyze_operational_cost_decomposition.py
python scripts/generate_ai_darld_v3_worked_example.py
```

The Favorita runner keeps its historical filename so the frozen protocol and
manifests remain executable; the reported study describes it as a pre-specified
external evaluation. Local row-level outputs and checkpoints are regenerated
from the source datasets and are not redistributed. The included manifests
record formal output identities and source-file hashes.

The preparation command expects
`data/raw/spdf/Spare-Part-Demand-Forecasting-main.zip` and extracts the two
required workbooks to the parser's frozen interim path. Repeated execution
verifies identical contents and refuses to overwrite a different local file.
The default strict run writes `strict_shared_separate_corrected_v2`; each
search-budget run writes its own `strict_search_stability_v1/budget_<N>`
directory, so these workflows do not overwrite one another. A historical
version comparison is optional:

```bash
python scripts/finalize_strict_shared_separate_v1.py \
  --historical-comparison path/to/prior_paired_contrasts.csv
```

## Result Map

- Main service-parts table and paired intervals:
  `outputs/ai_darld_v3/historical_support_v4_authority_summary.csv` and
  `outputs/ai_darld_v3/historical_support_v4_authority_comparisons.csv`, generated
  by `analyze_support_v3_similarity_residual.py` from
  `historical_support_v4_results.parquet` plus the validated global-baseline rows.
- Intermediate service-parts policy/global combination:
  `outputs/runs/ai_darld_v3/corrected_authority_results.parquet`, generated by
  `analyze_ai_darld_v3_corrected.py`. The final authority uses only its rolling
  global-quantile and adapted ZIG--MC rows; learned and complete-transfer rows
  come from `historical_support_v4_results.parquet`.
- Current Online Retail II table:
  `outputs/ai_darld_v3/uci_confirmation_v3_summary.csv`, generated by
  `run_uci_confirmation_v2.py` after the population-construction run above.
  It reports component-specific minus comparator differences with 2,000
  percentile-bootstrap draws (seed 20261103) that resample target products and
  retain each product's repeated evaluation contexts. The accompanying frozen
  split, policy selection, and run identity are in the matching
  `uci_confirmation_v3_*` files. The earlier
  `uci_online_retail_ii_external_summary.csv` is retained for provenance and is
  not the source of the current paper table.
- Favorita table:
  `outputs/ai_darld_v3/favorita_confirmation_v1/summary.csv`, with paired
  comparisons in the same released result directory.
- Strict coefficient-untying control:
  `outputs/ai_darld_v3/strict_shared_separate_corrected_v2/`.
- Search sensitivity:
  `outputs/ai_darld_v3/strict_search_stability_v1/`.
- Operating-condition stratification:
  `outputs/ai_darld_v3/operating_condition_input_quality_v1/`.
- Cost decomposition:
  `outputs/ai_darld_v3/operational_cost_decomposition_v1/`.
- Cross-dataset evidence-increment forest plot inputs:
  `outputs/ai_darld_v3/evidence_increment_synthesis/`.
- Walkthrough case:
  `outputs/ai_darld_v3/worked_example.json` and
  `outputs/ai_darld_v3/worked_example.csv`.

## Key Protocol Conditions

The retrospective evaluations hide target demand history at the pseudo-launch
cutoff. Donor pools are formed only from cutoff-valid historical products. All
policies are evaluated through the same inventory solver with fixed-order cost,
minimum order quantity, capacity when applicable, and a zero-action option.

For MAN/BRAF, the global quantile baseline interpolates its quantile grid into
a finite scenario law before entering the common constrained solver. The
Online Retail II and Favorita protocols have no MOQ, capacity, or fixed-order
cost and therefore use the cost-ratio critical quantile directly; this is the
equivalent newsvendor action for those unconstrained settings.

Additional provenance for strict controls, search budgets, product-clustered
intervals, the Favorita identifier correction, and reused populations is in
`docs/REPRODUCIBILITY.md`.

## Tests

```bash
pytest --no-cov
```

The tests cover the common solver, factorized and strict relation laws, selected
data adapters, result pairing, cost-decomposition helpers, and the released
walkthrough artifacts.

## License and Third-Party Rights

Code in this repository is released under the MIT License. Third-party datasets
retain their original licenses and terms and are not relicensed here. Raw data
and some row-level derived artifacts are intentionally absent.
