# Public Snapshot Manifest

This repository was created as a clean public reproducibility snapshot with a
new git history. It is not a fork of the development repository.

## Included Content

- Core Python modules for donor construction, transfer policies, demand-law
  construction, and the common inventory solver.
- Formal experiment scripts and configurations for the reported baselines,
  learned transfer runs, strict coefficient-untying control, search sensitivity,
  operating-condition analysis, cost decomposition, Favorita confirmation, UCI
  Online Retail II evaluation, and the walkthrough case.
- Aggregate outputs and public manifests needed to understand the reported
  tables and sensitivity results.
- Redacted product-level inputs for the cross-dataset evidence-increment
  synthesis.
- Tests for the solver, relation laws, data adapters, result pairing, cost
  components, and released walkthrough artifacts.

## Excluded Content

- Raw third-party datasets.
- Manuscript source, PDFs, LaTeX build files, and submission packages.
- Full row-level evaluation outputs derived from third-party datasets.
- Development-only working records and obsolete exploratory branches.

## Provenance Retained

The public documentation keeps the scientific provenance needed to interpret the
formal results:

- complete configuration versus strict coefficient-untying control;
- finite search budget and actual candidate accounting;
- product-clustered interval scope;
- Favorita product-identifier correction;
- retrospective pseudo-launch design and reused populations;
- dataset-specific global-quantile interfaces.

The UCI and Favorita role manifests retain source catalog item keys required to
reconstruct the frozen assignments. The compact forest-plot inputs instead use
stable redacted product keys.
