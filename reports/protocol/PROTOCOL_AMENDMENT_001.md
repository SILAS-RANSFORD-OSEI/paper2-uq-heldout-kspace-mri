# Protocol Amendment 001

## Title

Reclassification of the Paper 2 final evaluation cohort

## Status

Frozen before Paper 2 volume assignment and before generation
of Paper 2 test predictions.

## Original intention

The preliminary Paper 2 plan described a fresh 40-volume final
test cohort within a 181/20/40/40 allocation.

## Feasibility finding

The audited reusable population contained the same 281 volumes
used in Paper 1:

- 201 Paper 1 training volumes;
- 40 Paper 1 calibration volumes;
- 40 Paper 1 test volumes.

No allocation of these existing cached volumes could simultaneously
provide a cohort that:

1. was new to Paper 2 study design;
2. had not influenced Paper 1 fitting or model selection; and
3. could reuse the validated reconstruction and reliability cache.

## Amended policy

The Paper 1 roles are preserved:

- D_fit: 181 volumes selected from the Paper 1 training split;
- D_dev: the remaining 20 Paper 1 training volumes;
- D_cal: all 40 Paper 1 calibration volumes;
- D_test: all 40 Paper 1 test volumes.

D_test is designated a **locked reused evaluation cohort**.

## Scientific rationale

This strategy is the only evaluated option that preserves
independence from Paper 1 SSDU and A4 gradient fitting,
checkpoint selection, and calibration while retaining the fully
audited cache.

Selecting a new test cohort from Paper 1 training or calibration
data would introduce stronger upstream leakage unless the SSDU
reconstruction, A4 predictor, and reliability cache were rebuilt.

## Claim limitation

The study may state that D_test was excluded from all Paper 2
fitting, model selection, threshold selection, and calibration.

The study must not describe D_test as:

- fresh;
- previously unseen;
- external;
- prospectively untouched; or
- independent of Paper 2 study design.

Paper 1 results from this cohort informed the motivation and
design of Paper 2. This limitation must be disclosed in the
manuscript.

## Safeguards

The Paper 2 protocol, methods, endpoint definitions, calibration
rules, and analysis code will be frozen before Paper 2 test
predictions are generated. The test barrier remains closed.
