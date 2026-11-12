# Fixed-Policy Service Calibration Validation

- Three methods have exactly 13,500 evaluation rows each.
- Every target/cutoff/lead-time/cost-ratio/capacity key has one row per method.
- All three methods use the same realized demand at each key.
- The calibrated and uncalibrated component-specific arms use the same fitted distributions; only the shortage multiplier used for action selection differs.
- Every reported total equals realized holding plus shortage plus fixed-order cost.
- Evaluation uses the original shortage cost, not the multiplier-adjusted action-selection cost.
- The calibration grid and selection use historical calibration products only.
- All nine cutoff/cost-ratio cells are feasible within the frozen multiplier grid.
- Product-clustered intervals retain all 54 repeated contexts per evaluation product.

Status: verified for the completed retrospective exploratory run.
