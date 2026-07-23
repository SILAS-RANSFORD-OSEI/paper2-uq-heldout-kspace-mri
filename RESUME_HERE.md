# Resume Paper 2 Here

## Completed through P2-Exp003B

U1 MC-dropout training was completed and audited.

- GPU: NVIDIA L4
- Precision: FP32
- Dropout probability: 0.10
- MC passes: 20
- Completed epoch: 17
- Selected epoch: 7
- Best D_dev MC-mean MAE:
  0.21612471
- Best D_dev deterministic MAE:
  0.21543331
- Supported MC variance mean:
  0.0295678931
- Gradient updates: 12,223
- D_cal arrays opened: 0
- D_test arrays opened: 0
- Final-test barrier: CLOSED

## Next stage

P2-Exp003C — train the three-member U2a
point-predictor ensemble using D_fit for gradient
updates and D_dev for model selection.
