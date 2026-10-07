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
- De-identified product-level inputs for the cross-dataset evidence-increment
  synthesis.
- Tests for the solver, relation laws, data adapters, result pairing, cost
  components, and released walkthrough artifacts.

## Excluded Content

- Raw third-party datasets.
- Manuscript source, PDFs, LaTeX build files, and submission packages.
- Row-level outputs that retain public product codes, competition item IDs, or
  other identifiers unsuitable for redistribution.
- Internal review notes, development archives, scoring ledgers, audit memos, and
  revision plans.

## Provenance Retained

The public documentation keeps the scientific provenance needed to interpret the
formal results:

- complete configuration versus strict coefficient-untying control;
- finite search budget and actual candidate accounting;
- product-clustered interval scope;
- Favorita product-identifier correction;
- retrospective pseudo-launch design and reused populations;
- dataset-specific global-quantile interfaces.

These notes are research provenance, not a separate audit report.
