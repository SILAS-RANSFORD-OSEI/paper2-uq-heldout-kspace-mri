# Resume Paper 2 Here

## Current state

P2-Exp000 is complete and passed.

P2-Exp001A and P2-Exp001B are complete and passed.

The Paper 2 split policy is frozen, but no volume IDs have been
assigned.

The final test barrier remains CLOSED.

## Frozen allocation

- D_fit: 181 volumes from Paper 1 train;
- D_dev: 20 volumes from Paper 1 train;
- D_cal: all 40 Paper 1 calibration volumes;
- D_test: all 40 Paper 1 test volumes.

D_test is a locked reused evaluation cohort, not a fresh test
cohort.

## Next action

Run P2-Exp001C on CPU:

1. audit outcome-blind acquisition metadata for the 201 Paper 1
   training volumes;
2. freeze a deterministic stratification rule using seed 20260720;
3. assign 20 volumes to D_dev and 181 to D_fit;
4. verify balance, uniqueness, and zero overlap;
5. commit the split before training any Paper 2 model.

## Read first

`reports/handoffs/PAPER2_HANDOFF_2026-07-20.md`

## Latest pre-handoff commit

`b62bb6b69cdc3862365dfeb25cd20e258aa879d0`

## Final test barrier

`CLOSED`
