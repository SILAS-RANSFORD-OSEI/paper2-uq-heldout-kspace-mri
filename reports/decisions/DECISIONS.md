# Paper 2 Decision Log

All changes made after protocol freeze must be recorded here.

## Entry template

- **Date:** YYYY-MM-DD
- **Decision ID:** DXXX
- **Original rule:**
- **Revised rule:**
- **Reason:**
- **Data inspected before decision:**
- **Effect on reproducibility:**
- **Approved protocol version:**

## D001 — Freeze Paper 2 protocol v1.1

- **Date:** 2026-07-20
- **Decision ID:** D001
- **Original rule:** Paper 2 design remained provisional.
- **Revised rule:** Protocol v1.1, notation registry v1.1, method registry v1.0, endpoint registry v1.0, and data governance v1.0 are frozen before implementation.
- **Reason:** Prevent outcome-driven changes and notation drift.
- **Data inspected before decision:** Paper 1 aggregate results and methodological literature; no Paper 2 final-test result.
- **Effect on reproducibility:** Establishes machine-readable pre-implementation contracts.
- **Approved protocol version:** v1.1

## D002 — Freeze reliability-cache semantic contract

- **Date:** 2026-07-20
- **Decision ID:** D002
- **Original rule:** Cache validity was provisionally assessed using complete fixed-shape signatures.
- **Revised rule:** The frozen schema is template-based: `x` has shape `(6,H,W)` and `y`/`y_raw` have shape `(H,W)`, allowing acquisition-matrix heterogeneity.
- **Semantic translation:** Legacy `y` maps to `u_risk`; legacy `y_raw` maps to `z_risk`; legacy `x[:3]` maps to predictor input `C_v`.
- **Reason:** Paper 1 source tracing established four legitimate spatial sizes and a common semantic schema.
- **Data inspected before decision:** 36 stratified NPZ archives, the cache writer, helper functions, configuration, and the canonical A4 checkpoint.
- **Effect on reproducibility:** Prevents false schema failures and confines ambiguous storage names to one loader.
- **Final-test data inspected:** No Paper 2 final-test predictions or performance results.

## D003 — Close P2-Exp000 after full-cache audit

- **Date:** 2026-07-20
- **Decision ID:** D003
- **Decision:** P2-Exp000 was completed with PASS after all 4,462 registered cache archives passed the frozen semantic and numerical integrity contract.
- **Accepted investigative findings:** Legitimate matrix-size heterogeneity and rejection of the provisional scalar `y`/`y_raw` hypothesis.
- **Resolution:** A template-based cache schema and the source-confirmed nonlinear target transformation were frozen in cache contract v1.0.
- **Paper 1 source commit:** `da563ead8fb653539e1eeca29248b31f0121ca12`.
- **Final-test data inspected:** No Paper 2 final-test predictions or performance results.
- **Next permitted experiment:** P2-Exp001 split construction and leakage audit.

## D004 — Preserve Paper 1 calibration and test roles

- **Date:** 2026-07-20
- **Decision ID:** D004
- **Selected strategy:** `S1_preserve_paper1_calibration_and_test`.
- **Allocation policy:** D_fit and D_dev will be selected only from the 201 Paper 1 training volumes; D_cal will contain all 40 Paper 1 calibration volumes; D_test will contain all 40 Paper 1 test volumes.
- **D_test designation:** Locked reused evaluation cohort.
- **Reason:** This is the only audited strategy that preserves independence from Paper 1 fitting, checkpoint selection, and calibration without rebuilding upstream models and cache artifacts.
- **Limitation:** The cohort was reported in Paper 1 and its results informed Paper 2 motivation and design. It must not be described as a fresh or previously unseen Paper 2 test cohort.
- **Safeguard:** All Paper 2 methods, calibration rules, and endpoint code must be frozen before test predictions are generated.
- **Volume IDs assigned at this decision:** No.
- **Final test barrier:** Closed.

## D005 — Freeze the outcome-blind Paper 2 volume split

- **Date:** 2026-07-21
- **Decision ID:** D005
- **Algorithm:** `width_coil_ilp_slice_balance_v1.0`.
- **Optimization:** Deterministic mixed-integer linear programming using SciPy/HiGHS.
- **Seed:** `20260720`.
- **Allocation:** D_fit=181, D_dev=20, D_cal=40, D_test=40.
- **Exact D_dev width quotas:** 272=0, 320=7, 392=1, 396=12.
- **Coarse coil groups:** L<=8, M=12-14, H>=16.
- **Exact D_dev coil-group quotas:** L=5, M=2, H=13.
- **Secondary objective:** Slice-count category and mean-depth balance.
- **Statistical tests:** KS and chi-square tests were descriptive only; p-values were not used for acceptance.
- **Patient grouping:** The available patient_id was unique per volume but was not independently validated as a cross-volume clinical identifier. Only volume-level separation is claimed.
- **Outcome inputs:** None. No risk target, prediction, uncertainty estimate, cache NPZ array, error, or performance result entered the assignment algorithm.
- **Split SHA-256:** `ca855cfa07e878b8b582b8decd0c96b9b80ffe98003c6734405db7d2c1dcc81a`.
- **D_test designation:** Locked reused evaluation cohort.
- **Final test barrier:** Closed.

## D006 — Enforce purpose-specific split access

- **Date:** 2026-07-21
- **Decision ID:** D006
- **Gradient fitting:** D_fit only.
- **Model selection:** D_dev only.
- **Calibration:** D_cal only.
- **Final evaluation:** D_test only after an explicit test-barrier opening.
- **Current barrier:** Closed.
- **D_test arrays opened in P2-Exp002A:** Zero.
- **D_test predictions generated:** No.
- **Implementation:** `src/paper2_uq_mri/split_access.py`.
