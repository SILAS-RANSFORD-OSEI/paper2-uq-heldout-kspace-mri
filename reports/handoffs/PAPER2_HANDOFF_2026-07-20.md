# Paper 2 Project Handoff

## Project identity

**Working title**

Benchmarking model-derived uncertainty against held-out
k-space residual risk in self-supervised accelerated MRI
reconstruction

**Repository**

`paper2-uq-heldout-kspace-mri`

**Current branch**

`main`

**Handoff date**

20 July 2026

## Core research question

How faithfully do model-derived uncertainty estimates identify
independently measured residual risk when accelerated MRI
reconstruction is evaluated without a fully sampled supervisory
target?

## Scientific foundation inherited from Paper 1

Paper 1 introduced a four-way partition of acquired k-space:

- reconstruction input: Theta;
- self-supervised reconstruction loss: Lambda_rec;
- residual-risk learning: Lambda_risk;
- independent verification: Lambda_hold.

Paper 2 benchmarks uncertainty estimates against the independent
held-out residual-risk target derived from Lambda_hold.

## Planned Paper 2 methods

- C0: deterministic A4 control;
- U1: Monte Carlo dropout;
- U2a: point-predictor ensemble;
- U2b: probabilistic deep ensemble;
- B1-B6: deterministic image and reconstruction descriptors.

## Evaluation tasks

### Task P — Mean prediction quality

Primary descriptive endpoint: mean absolute error between the
predicted mean and the independent held-out risk target.

### Task R — High-risk identification

Primary endpoint: calibration-fixed area under the
precision-recall curve for identifying high held-out residual risk.

### Task E — Error ranking

Primary endpoint: normalized area under the sparsification error
curve for ranking absolute prediction deviation.

## Reporting framework

- CLAIM 2024 is the primary reporting framework.
- TRIPOD+AI principles will be used selectively.
- STARD-AI is not the governing framework.
- Final reporting will use volume-level bootstrap confidence
  intervals.
- The final evaluation cohort must not be opened before the
  protocol-defined test-opening conditions are satisfied.

## Completed repository and protocol initialization

The following were created and validated:

- repository directory structure;
- Python package skeleton;
- environment files;
- continuous-integration workflow;
- artifact policy;
- private development license;
- citation metadata;
- protocol version 1.1;
- method and endpoint registries;
- data-governance contract;
- notation registry version 1.1.

The notation registry contains 40 canonical symbols and passed all
validation checks.

## Canonical Paper 1 implementation source

Repository:

`SILAS-RANSFORD-OSEI/fourway-ssdu-reliability-mri-v2`

Frozen source commit:

`da563ead8fb653539e1eeca29248b31f0121ca12`

The similarly named submission-materials repository is not an
implementation source and must not be substituted.

## Canonical Paper 1 reusable artifacts

### SSDU checkpoint

Relative path:

`outputs/exp004_train_ssdu_v4_full/best_model.pt`

SHA-256:

`f63931fdd06ce52455c4dc245845ac4fd0bb9b565ae03b8c09f18c90df67fd00`

Selection basis:

Unique highest downstream provenance score.

### A4 checkpoint

Relative path:

`outputs/exp008_reliability_ablation_full/A4_image_only/best_model.pt`

SHA-256:

`cc1bf4c79522d6a2b9a4406461273252dd3e4621a42be2da0dd13742ce5c5cc1`

Selection basis:

Canonical A4 image-only path with its companion summary and
final-split metrics.

## P2-Exp000 — Completed

P2-Exp000 audited the Paper 1 source, artifacts, checkpoints,
reliability-cache manifest, NPZ schema, semantic meaning, and all
cached files.

Status: **PASS**

### Dataset and cache population

- volumes: 281;
- cached slices: 4,462;
- Paper 1 training volumes: 201;
- Paper 1 calibration volumes: 40;
- Paper 1 test volumes: 40;
- training slices: 3,190;
- calibration slices: 636;
- test slices: 636.

No volume overlap was found across the Paper 1 splits.

### Full-cache integrity result

All 4,462 registered NPZ archives passed:

- file existence;
- unique manifest resolution;
- ZIP integrity;
- numerical loading;
- no object arrays;
- no non-finite values;
- frozen key-set contract;
- frozen dimensional-template contract;
- support-mask bounds;
- nonlinear target-transformation verification.

Failed archives: **0**

### Matrix-size distribution

- 640 x 272: 16 slices;
- 640 x 320: 1,660 slices;
- 768 x 392: 176 slices;
- 768 x 396: 2,610 slices.

### Target-transformation audit

Maximum absolute reconstruction error:

`0.0018019676`

Minimum Pearson correlation:

`0.999999424`

These discrepancies are consistent with independent float16
storage of the pre-log and log-transformed targets.

## Frozen reliability-cache semantic contract

Legacy NPZ keys are translated immediately at the loader boundary.

- legacy `x` maps to `cache_input_6ch`;
- legacy `x[0:3]` maps to predictor input C_v;
- legacy `y` maps to u_risk,v;
- legacy `y_raw` maps to z_risk,v.

The legacy storage key `y` must never be used as the Paper 2
scientific symbol y, because y is reserved for measured k-space.

### Six-channel input order

0. robust-normalized reconstruction magnitude;
1. robust-normalized zero-filled magnitude;
2. robust-normalized intervention magnitude;
3. soft anatomical support mask;
4. analytical PSF from Lambda_risk;
5. sensitivity-aware PSF/gain descriptor.

A4 uses channels 0, 1, and 2 only.

### Risk-learning target transformation

The legacy `y_raw` field is the support-masked PSF-normalized
residual-risk quantity before final normalization.

The legacy `y` field is produced by:

`log1p(10 * z_risk / (Q_0.99(z_risk) + 1e-8))`

The provisional hypothesis that `y` was only a scalar multiple of
`y_raw` was correctly rejected.

## P2-Exp001A — Completed

A prior-exposure and split-feasibility audit was performed before
assigning any Paper 2 volume IDs.

Status: **PASS**

### Exposure of Paper 1 training volumes

- used for SSDU gradient fitting;
- used for A4 gradient fitting;
- not used for Paper 1 reported evaluation.

### Exposure of Paper 1 calibration volumes

- used for SSDU checkpoint selection;
- used for A4 checkpoint selection;
- used for Paper 1 calibration.

### Exposure of Paper 1 test volumes

- not used for gradient fitting;
- not used for checkpoint selection;
- not used for calibration;
- used for Paper 1 reported evaluation;
- Paper 1 results informed Paper 2 motivation and design.

## P2-Exp001B — Completed

The split policy was frozen before assigning volume IDs.

Selected strategy:

`S1_preserve_paper1_calibration_and_test`

Planned allocation:

- D_fit: 181 volumes from Paper 1 training;
- D_dev: 20 volumes from Paper 1 training;
- D_cal: all 40 Paper 1 calibration volumes;
- D_test: all 40 Paper 1 test volumes.

D_test designation:

**locked reused evaluation cohort**

## Required claim limitation

D_test may be described as:

- excluded from Paper 1 SSDU and A4 gradient fitting;
- excluded from Paper 1 checkpoint selection and calibration;
- excluded from all Paper 2 fitting, selection, thresholding,
  and calibration;
- locked before Paper 2 test predictions are generated.

D_test must not be described as:

- a fresh Paper 2 test cohort;
- previously unseen;
- external validation;
- prospectively untouched;
- independent of Paper 2 study design.

The manuscript must disclose that the same cohort was reported in
Paper 1 and that those results informed Paper 2 motivation and
design.

## Current governance state

- P2-Exp000: completed, PASS;
- P2-Exp001A: completed, PASS;
- P2-Exp001B: completed, PASS;
- Paper 2 volume IDs assigned: no;
- D_fit/D_dev split created: no;
- Paper 2 models trained: no;
- uncertainty calibration frozen: no;
- Paper 2 test predictions generated: no;
- final test barrier: CLOSED.

## Exact next experiment

Continue with **P2-Exp001C: acquisition-metadata stratification
audit**.

The next experiment must:

1. inspect only outcome-blind acquisition and structural metadata
   for the 201 Paper 1 training volumes;
2. identify usable stratification variables such as matrix width,
   matrix height, coil count, and slice count;
3. exclude u_risk, u_hold, predictions, errors, uncertainty scores,
   and performance outcomes;
4. freeze a deterministic stratification algorithm using seed
   20260720;
5. assign exactly 20 volumes to D_dev and 181 to D_fit;
6. audit distribution balance and zero overlap;
7. commit the volume-level split before any Paper 2 model fitting.

## Runtime guidance

Continue on CPU for:

- metadata audit;
- split construction;
- leakage tests;
- endpoint unit tests;
- calibration-rule implementation.

A GPU will first be required when fitting or evaluating the
trainable Paper 2 neural methods, including Monte Carlo dropout and
ensemble members.

## Important filesystem locations

Paper 2 repository:

`/content/paper2-uq-heldout-kspace-mri`

Paper 2 Drive root:

`/content/drive/MyDrive/Paper2_UQ_Heldout_KSpace_MRI`

Paper 1 source repository:

`/content/fourway-ssdu-reliability-mri-v2`

Paper 1 Drive root:

`/content/drive/MyDrive/FOUR WAY MRI RESEARCH`

Local staged cache:

`/content/paper1_reliability_cache_local`

The local staged cache is approximately 10.20 GB and is temporary.
It is not tracked by Git and will disappear when the Colab runtime
is reset.

## Commit and tag history before this handoff

- `d6999f1` — protocol and repository initialization;
  tag `protocol-v1.1`.
- `2e4536f` — canonical Paper 1 provenance;
  tag `exp000-provenance-v1.2`.
- `979b6af` — complete Paper 1 cache and provenance audit;
  tag `exp000-complete-v1.0`.
- `b62bb6b` — frozen Paper 2 split policy;
  tag `exp001-policy-v1.0`.

## Restart procedure

1. Start a new Colab runtime.
2. Mount Google Drive.
3. Clone the Paper 2 repository.
4. Check out `main`.
5. Confirm the latest handoff tag.
6. Recreate the standard project path variables.
7. Clone the canonical Paper 1 source at commit
   `da563ead8fb653539e1eeca29248b31f0121ca12`.
8. Run the repository tests.
9. Read `RESUME_HERE.md`.
10. Begin P2-Exp001C on CPU.

## Non-negotiable safeguards

- Do not open the Paper 2 test barrier.
- Do not calculate Paper 2 test predictions.
- Do not use outcome variables to choose D_dev.
- Do not alter the frozen notation contract silently.
- Do not substitute another Paper 1 repository or checkpoint.
- Do not describe D_test as fresh or previously unseen.
- Record every protocol change in the decision log.
