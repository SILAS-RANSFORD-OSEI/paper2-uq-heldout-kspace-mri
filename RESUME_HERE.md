# Resume Paper 2 Here

## Completed through P2-Exp003D

The three-member U2b probabilistic deep ensemble was
trained, resumed safely after interruption, and audited.

- GPU: NVIDIA L4
- Precision: FP32
- Member seeds: (20260720, 20260721, 20260722)
- Completed epochs: (42, 46, 52)
- Best epochs: (32, 36, 42)
- Member D_dev NLLs:
  (-0.44048848, -0.45470173, -0.44688776)
- Member D_dev mean MAEs:
  (0.2051494, 0.20154278, 0.20304239)
- Ensemble D_dev mean MAE:
  0.20115886
- Moment-matched ensemble D_dev NLL:
  -0.46601757
- Supported within-model variance:
  0.0788495996
- Supported between-model variance:
  0.0015053657
- Supported total predictive variance:
  0.0803549655
- Total gradient updates:
  100,660
- D_cal arrays opened during training: 0
- D_test arrays opened: 0
- Final-test barrier: CLOSED

## Next stage

P2-Exp004A — generate frozen neural predictions and
uncertainty outputs on D_cal. D_test remains closed.
