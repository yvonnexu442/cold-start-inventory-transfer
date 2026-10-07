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
# MAN/BRAF source preparation and local reliability checkpoints
python scripts/generate_service_parts_checkpoints.py --dataset all

# MAN/BRAF component-specific transfer and common baselines
python scripts/run_ai_darld_v3_factorized.py
python scripts/run_ai_darld_v3_corrected_global.py
python scripts/run_ai_darld_v3_direct_mixture.py

# Online Retail II and Favorita external populations
python scripts/run_uci_online_retail_ii_external.py
python scripts/run_uci_confirmation_v2.py
python scripts/run_favorita_untouched_confirmation.py
python scripts/analyze_favorita_confirmation.py

# Strict coefficient-untying control and search sensitivity
python scripts/run_strict_shared_separate_v1.py
python scripts/run_strict_shared_separate_retail_v1.py
python scripts/finalize_strict_shared_separate_v1.py
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

## Result Map

- Main service-parts summaries:
  `outputs/ai_darld_v3/corrected_method_summary.csv` and
  `outputs/ai_darld_v3/corrected_target_clustered_comparisons.csv`.
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

The global quantile baseline uses a dataset-level quantile grid to create a
finite scenario law before entering the shared solver. A single critical
quantile is only a simplification for unconstrained textbook newsvendor cases;
the formal constrained evaluations compare expected costs over the scenario
law.

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
