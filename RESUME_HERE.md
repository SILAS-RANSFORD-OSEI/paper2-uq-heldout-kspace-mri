# Resume Paper 2 Here

## Completed through P2-Exp002D

The Paper 2 trainable model family is frozen:

- C0: deterministic A4 control.
- U1: MC-dropout with p=0.1
  and 20 evaluation passes.
- U2a: three-member point-predictor ensemble.
- U2b: three-member probabilistic deep ensemble.

All trainable neural models accept exactly three-channel C_v.

All trainable models start from independent random
initialization. The frozen Paper 1 A4 checkpoint is retained
only as a compatibility reference and is not used to initialize
Paper 2 models because it was trained on the original
201-volume cohort containing the current D_dev subset.

D_fit arrays opened: 0
D_dev arrays opened: 0
D_cal arrays opened: 0
D_test arrays opened: 0
Gradient updates: 0
Final-test barrier: CLOSED

## Next stage

P2-Exp003A — train the deterministic C0 control using D_fit
for gradient updates and D_dev for model selection.

Switch the Colab runtime to an NVIDIA T4 GPU or better before
running P2-Exp003A.
