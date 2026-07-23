# Resume Paper 2 Here

## Completed through P2-Exp003A

The deterministic C0 control was trained successfully.

- Valid run precision: FP32
- GPU: NVIDIA L4
- Completed epoch: 42
- Early stopping: yes
- Selected epoch: 32
- Best D_dev support-weighted MAE:
  0.20043605
- Gradient updates: 30,198
- D_cal arrays opened: 0
- D_test arrays opened: 0
- Final-test barrier: CLOSED

The preliminary FP16 AMP attempt overflowed before completing
epoch 1 and did not produce a valid training checkpoint.

## Next stage

P2-Exp003B — train the U1 MC-dropout model using D_fit
for gradient updates and D_dev for model selection.
