# Resume Paper 2 Here

## Completed through P2-Exp003C

The three-member U2a point-predictor ensemble was
trained and independently audited.

- GPU: NVIDIA L4
- Precision: FP32
- Member seeds: (20260720, 20260721, 20260722)
- Best epochs: (32, 31, 30)
- Member D_dev MAEs:
  (0.20043605, 0.20002255, 0.20034162)
- Ensemble D_dev MAE:
  0.19779335
- Supported between-model variance:
  0.0018016411
- Positive-variance batches:
  80/80
- Total gradient updates:
  88,437
- D_cal arrays opened: 0
- D_test arrays opened: 0
- Final-test barrier: CLOSED

## Next stage

P2-Exp003D — train the three-member U2b
probabilistic deep ensemble using D_fit for gradient
updates and D_dev for independent member selection.
