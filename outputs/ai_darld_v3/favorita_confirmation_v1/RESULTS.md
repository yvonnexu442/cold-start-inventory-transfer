# Favorita untouched-confirmation result

## Evidence identity

- The dataset and product population had no repository or outcome-use history
  before the protocol commit `04cf0e9`.
- The observation-window feasibility amendment (`1470118`) occurred before
  policy fitting or outcome scoring.
- The first complete execution used row positions instead of `item_nbr` for
  pseudo-target hashing. That superseded run is not included in this public
  snapshot; the correction is recorded here as scientific provenance.
- The authoritative execution uses product identifiers (`67c8d11`) and changes
  no method, target cohort, outcome, context, or analysis rule in response to
  the first execution.

## Data handling

The frozen calendar contains 44,661,573 transaction rows. Negative transactions
represent 0.00653% of rows. After store aggregation, 52 net-negative item-days
were clipped to zero: 0.00319% of observed item-days and 0.00288% of all common-
calendar item-days. Store information is used only to report aggregation
coverage. The model uses `family`, `class`, and `perishable`.

## Primary findings

The rolling horizon-matched global quantile policy has the lowest mean cost
(7,582.40). Component-specific transfer has mean cost 8,217.23. Its paired cost
difference versus the global policy is 634.83 (95% product-clustered interval
[-93.10, 1,416.00]), so this population does not detect a clear learned-policy
advantage over the strongest comparator.

Component-specific transfer has lower point-estimate cost than complete
similarity (difference -247.42; -2.9%; interval [-1,474.11, 1,077.12]) and
complete uniform (-303.72; -3.6%; [-1,543.07, 1,033.96]), while both complete
mixtures have materially higher service point estimates (0.950 and 0.951 versus
0.857). Component-specific transfer is also lower-cost than its matched shared
control by 1,065.28 (-11.5%), but the interval [-2,385.15, 485.39] spans zero.
It clearly improves on single-donor transfer (-1,695.16; [-2,952.28, -605.81]).

## Frozen triage

This is **Confirmation C**. The isolated population supports the strategy-level
empirical contribution: global prediction and complete multi-donor transfer are
strong practical policies, and aggregation clearly improves on a single donor.
It does not independently confirm a competitive or matched component-specific
gain. The cost-service ordering also reinforces the need to report both metrics.
