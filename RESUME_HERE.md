# Resume Paper 2 Here

## Current state

P2-Exp000, P2-Exp001A-D, and P2-Exp002A are complete.

The frozen split remains:

- D_fit: 181 volumes
- D_dev: 20 volumes
- D_cal: 40 volumes
- D_test: 40 volumes

Split SHA-256:

`ca855cfa07e878b8b582b8decd0c96b9b80ffe98003c6734405db7d2c1dcc81a`

The role-aware access contract is implemented in:

`src/paper2_uq_mri/split_access.py`

Access is restricted as follows:

- gradient fitting: D_fit only
- model selection: D_dev only
- calibration: D_cal only
- final evaluation: D_test only after explicit barrier opening

All 4,462 cached slices were indexed from the local SSD.
One D_fit sample was opened for a semantic smoke test.
No D_test cache array has been opened in Paper 2.

The final-test barrier remains CLOSED.

## Next action

Run P2-Exp002B on CPU:

1. test role-aware minibatch construction;
2. confirm the three-channel predictor input contract;
3. verify target and spatial-shape alignment;
4. test heterogeneous matrix sizes safely;
5. use D_fit and D_dev only.

## Final test barrier

`CLOSED`
