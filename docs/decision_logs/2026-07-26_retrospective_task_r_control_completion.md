# Decision log: retrospective Task R control completion

**Recorded:** 2026-07-26T18:47:24.651474+00:00  
**Parent repository commit:** `3d4cd549f04f81e875215135609566cbdc1f70d9`  
**Branch:** `retrospective-task-r-controls-v1.0`

## Status

This is a retrospective completion of prespecified Paper 2 controls after
the existing D_test uncertainty results had already been inspected.

It is not a new confirmatory evaluation and must not be described as one.

## Reason

The completed Paper 2 package evaluated U1, U2a, and U2b for Task R but
omitted:

1. the deterministic C0 mean prediction as a Task R score; and
2. the six prespecified deterministic descriptors B1-B6.

The Paper 2 protocol had defined these controls before implementation.

## Frozen source artifacts

Paper 1 cache:

`/content/drive/MyDrive/FOUR WAY MRI RESEARCH/outputs/exp007_train_reliability_cnn_v2_full/exp006_reliability_cache_full/cache`

Paper 2 aligned D_test chunks:

`/content/drive/MyDrive/Paper2_UQ_Heldout_KSpace_MRI/outputs/exp006_dtest_final/release_pending`

## Verified alignment

Cross-paper Audit 004 established exact alignment for:

- 40 volumes;
- 636 slices;
- 168,048,640 pixels;
- pixel coordinates;
- M_soft support weights;
- target_u_hold values.

Maximum support difference: 0.0  
Maximum target difference: 0.0

## Frozen Task R reference

Primary threshold:

`tau_hold = 1.7366089820861816`

Common label:

`h_v(p) = 1[u_hold,v(p) >= tau_hold]`

Evaluation support:

`M_soft,v(p) > 0`

Independent summary unit:

volume

## Frozen submitted scores

- C0: `c0_mean`
- U1: `u1_variance`
- U2a: `u2a_between_model_variance`
- U2b total: `u2b_total_predictive_variance`
- U2b within: `u2b_within_model_variance`
- U2b between: `u2b_between_model_variance`

Deterministic descriptors:

- B1: cache channel 0 = normalized `|x_hat|`
- B2: cache channel 1 = normalized `|x0|`
- B3: cache channel 2 = normalized `|x_hat - x0|`
- B4: finite-difference gradient magnitude of B1
- B5: cache channel 4 = analytical PSF descriptor
- B6: cache channel 5 = normalized q_PSF / gain envelope

## Direction and transformation freeze

Larger raw values are submitted as higher Task R scores for C0 and B1-B6.

The following are prohibited:

- D_test-dependent sign flipping;
- reciprocal or logarithmic descriptor transformations;
- outcome-dependent smoothing;
- descriptor-specific threshold fitting;
- omission of an unfavourable descriptor;
- choosing only a favourable pairwise comparison.

All six descriptors will be reported individually.

The phrase “strongest deterministic descriptor” may describe the highest
observed estimate only after all six results are presented. Inferential
comparisons will retain all six descriptor contrasts rather than treating
the observed maximum as a uniquely prespecified comparator.

## Endpoints

Primary added endpoint:

- per-volume support-weighted Task R AUPRC at frozen q90.

Secondary:

- per-volume AUROC;
- achieved prevalence;
- paired differences;
- 1,000 paired volume-bootstrap confidence intervals;
- volume-level win rates;
- q85 and q95 threshold sensitivity.

## Governance

No model training, model selection, reconstruction, target regeneration,
or calibration modification is permitted for this completion.
