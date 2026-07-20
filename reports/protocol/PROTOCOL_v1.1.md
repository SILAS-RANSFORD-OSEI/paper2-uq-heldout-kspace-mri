# Paper 2 Protocol v1.1

## Status

**Frozen before implementation**

## Working title

**Benchmarking model-derived uncertainty against held-out k-space
residual risk in self-supervised accelerated MRI reconstruction**

## Primary research question

Do practical model-derived uncertainty scores identify spatial
regions with elevated independent held-out k-space residual risk
in self-supervised accelerated brain MRI?

## Relationship to Paper 1

Paper 1 constructed an independent measurement-derived
residual-risk target using the four-way partition

\[
\Omega^{\mathrm{acq}}_v
=
\Theta_v
\cup
\Lambda_{\mathrm{rec},v}
\cup
\Lambda_{\mathrm{risk},v}
\cup
\Lambda_{\mathrm{hold},v}.
\]

Paper 2 benchmarks model-derived uncertainty against the independent
verification target \(u_{\mathrm{hold},v}\).

## Permitted primary claim

This study evaluates whether practical model-derived uncertainty
scores align with an independent held-out k-space residual-risk
target in self-supervised accelerated MRI reconstruction.

## Prohibited claims

The study must not claim that:

- \(u_{\mathrm{hold},v}\) is true reconstruction error;
- \(u_{\mathrm{hold},v}\) is clinical or diagnostic ground truth;
- uncertainty alignment guarantees safety;
- poor alignment proves clinical unsafety;
- the study is a complete benchmark of MRI uncertainty;
- the study provides external validation.

## Core methods

| Method ID | Symbol suffix | Method |
|---|---|---|
| C0 | C0 | Deterministic A4 residual-risk predictor |
| U1 | MC | MC-dropout predictive dispersion |
| U2a | PE | Point-predictor ensemble disagreement |
| U2b | DE | Probabilistic deep ensemble |
| B1 | B1 | Reconstruction magnitude |
| B2 | B2 | Zero-filled magnitude |
| B3 | B3 | Reconstruction intervention magnitude |
| B4 | B4 | Image-gradient magnitude |
| B5 | B5 | Analytical PSF descriptor |
| B6 | B6 | PSF gain-envelope descriptor |

## Data sets

\[
D_{\mathrm{fit}}=181,\qquad
D_{\mathrm{dev}}=20,\qquad
D_{\mathrm{cal}}=40,\qquad
D_{\mathrm{test}}=40.
\]

- \(D_{\mathrm{fit}}\): gradient-based fitting.
- \(D_{\mathrm{dev}}\): early stopping and method development.
- \(D_{\mathrm{cal}}\): thresholds and calibration.
- \(D_{\mathrm{test}}\): final independent internal testing only.

The Paper 2 final test set must remain closed until all methods,
thresholds, seeds, and metrics are frozen.

## Evaluation tasks

### Task P — direct prediction

The mean residual-risk prediction is compared with
\(u_{\mathrm{hold},v}\).

Primary descriptive endpoint:

\[
\operatorname{MAE}
\left(
\mu_{j,v},
u_{\mathrm{hold},v}
\right).
\]

### Task R — residual-risk localization

Define

\[
\tau_{\mathrm{hold}}
=
Q_{0.90}
\left(
u_{\mathrm{hold},v}
\mid
D_{\mathrm{cal}}
\right),
\]

and

\[
h_v(p)
=
\mathbf{1}
\left[
u_{\mathrm{hold},v}(p)
\geq
\tau_{\mathrm{hold}}
\right].
\]

Primary endpoint:

\[
\operatorname{AUPRC}
\left(
U_{j,v},
h_v
\right).
\]

The same calibration-fixed threshold is used for every method,
test volume, and bootstrap replicate.

### Task E — prediction-deviation awareness

Define

\[
d_{j,v}(p)
=
\left|
u_{\mathrm{hold},v}(p)
-
\mu_{j,v}(p)
\right|.
\]

Primary endpoint:

\[
\operatorname{AUSE}
\left(
U_{j,v},
d_{j,v}
\right).
\]

Lower AUSE is better.

## Statistical analysis

- The volume is the independent resampling unit.
- Use 1,000 paired volume-level bootstrap replicates.
- Use identical bootstrap volume indices for all methods.
- Report point estimates and percentile 95% confidence intervals.
- The threshold \(\tau_{\mathrm{hold}}\) remains fixed inside the
  bootstrap.
- Do not recompute a top-decile threshold from the test data.
- Effect estimates and confidence intervals are primary.
- Exploratory analyses must be labelled exploratory.

## Reporting framework

- CLAIM 2024 is the governing medical-imaging AI reporting framework.
- Relevant TRIPOD+AI principles may be adopted without claiming
  full compliance.
- STARD-AI is not the governing checklist because this is not a
  diagnostic-accuracy study.
- Use the phrase **independent internal testing**, not external
  validation.
- Every symbol must follow the canonical notation registry.

## Repository rules

1. Every experiment uses a committed configuration.
2. Every scientific output records provenance.
3. Large data, caches, and private checkpoints are excluded from Git.
4. All reported figures and tables are generated from scripts.
5. No result is reported from notebook-only code.
6. Changes after protocol freeze are entered in the decision log.
7. No final test result is inspected before calibration freeze.

## Experiment sequence

| Experiment | Purpose |
|---|---|
| P2-Exp000 | Environment, cache, checkpoint, and artifact audit |
| P2-Exp001 | Paper 2 split construction and leakage audit |
| P2-Exp002 | Deterministic C0 control |
| P2-Exp003 | MC-dropout predictive dispersion |
| P2-Exp004 | Point-predictor ensemble |
| P2-Exp005 | Probabilistic deep ensemble |
| P2-Exp006 | Deterministic descriptors |
| P2-Exp007 | Calibration freeze |
| P2-Exp008 | Final test and paired bootstrap |
| P2-Exp009 | Holdout-line subsampling stability |
| P2-Exp010 | Conditional second full mask realization |
| P2-Exp011 | Conditional uncertainty-scaled conformal intervals |
| P2-Exp012 | Publication tables, figures, and checklist |

## Hardware progression

- CPU: repository, audits, manifests, tests, and calibration code.
- T4 or L4: C0 and early U1 development.
- L4, A100, or sequential T4 runs: final five-member ensembles.
