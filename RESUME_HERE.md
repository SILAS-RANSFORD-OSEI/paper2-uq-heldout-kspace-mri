# Resume Paper 2 Here

## Current state

P2-Exp000 and P2-Exp001A-D are complete and passed.

The complete 281-volume Paper 2 split is frozen.

- D_fit: 181 volumes
- D_dev: 20 volumes
- D_cal: 40 volumes
- D_test: 40 volumes

Split file:

`data/splits/paper2_split.csv`

Split algorithm:

`width_coil_ilp_slice_balance_v1.0`

Split SHA-256:

`ca855cfa07e878b8b582b8decd0c96b9b80ffe98003c6734405db7d2c1dcc81a`

D_test remains a locked reused evaluation cohort, not a fresh
test cohort.

Only volume-level separation is claimed. The available
patient_id was not independently validated as a cross-volume
clinical identifier.

The final test barrier remains CLOSED.

## Next action

Run P2-Exp002A on CPU:

1. implement split-aware dataset and loader interfaces;
2. enforce role-level access controls;
3. prevent D_cal and D_test from entering gradient fitting;
4. prevent D_test from model selection, thresholding, or
   calibration;
5. add leakage and manifest-consistency tests.

## Final test barrier

`CLOSED`
