# Resume Paper 2 Here

## Current state

P2-Exp000, P2-Exp001A-D, P2-Exp002A, and P2-Exp002B
are complete and passed.

## Frozen batching contract

Predictor input:

`C_v = cache_input_6ch[0:3]`

Semantic cache target attribute:

`u_risk`

Scientific target symbol used downstream:

`u_risk_v`

Translation occurs at:

`paper2_uq_mri.batching.tensorize_semantic_sample`

No raw target or legacy cache key propagates into model code.

D_cal arrays opened in P2-Exp002B: 0
D_test arrays opened in P2-Exp002B: 0
D_test predictions generated: No

The final-test barrier remains CLOSED.

## Next action

Run P2-Exp002C on CPU:

1. load the canonical frozen A4 checkpoint;
2. reconstruct its exact architecture;
3. verify checkpoint-key and tensor-shape compatibility;
4. run forward smoke tests on D_fit and D_dev only;
5. do not open D_cal or D_test.

## Final test barrier

`CLOSED`
