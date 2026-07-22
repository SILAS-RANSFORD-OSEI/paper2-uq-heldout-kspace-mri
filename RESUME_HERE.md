# Resume Paper 2 Here

## Completed through P2-Exp002C

The exact frozen Paper 1 A4 model was verified as:

`fourway_mri.reliability_model.ReliabilityUNetSmall`

Constructor:

- `in_channels=6`
- `out_channels=1`
- `base_channels=8`

Checkpoint SHA-256:

`cc1bf4c79522d6a2b9a4406461273252dd3e4621a42be2da0dd13742ce5c5cc1`

Strict loading passed with zero missing and zero unexpected
keys.

Paper 2 exposes only three-channel `C_v`.
`A4ThreeChannelAdapter` internally appends three exact-zero
channels to reproduce the Paper 1 A4 ablation.

Forward equivalence and deterministic repeat checks passed
across all D_fit and D_dev matrix sizes.

D_cal arrays opened: 0
D_test arrays opened: 0
Gradient updates: 0
Final-test barrier: CLOSED

## Next stage

P2-Exp002D — freeze the C0, U1, U2a and U2b architecture
contracts before any training.
