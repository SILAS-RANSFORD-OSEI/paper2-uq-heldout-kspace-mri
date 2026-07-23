# Resume Paper 2 Here

## Completed through P2-Exp002E

The batching interface exposes:

- `C_v`: exactly three predictor channels.
- `u_risk_v`: one-channel risk-learning target.
- `M_soft`: one-channel support tensor used only for loss
  weighting.

`M_soft` is mapped exactly from source cache channel 3 and
is constrained to [0, 1]. It does not enter the predictor.

Training started: NO
Gradient updates: 0
D_cal arrays opened: 0
D_test arrays opened: 0
Final-test barrier: CLOSED

## Next stage

P2-Exp003A — train the deterministic C0 control.
