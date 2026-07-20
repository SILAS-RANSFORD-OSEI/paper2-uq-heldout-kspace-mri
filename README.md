# Paper 2: Uncertainty versus Held-Out K-Space Residual Risk

## Working title

**Benchmarking model-derived uncertainty against held-out k-space
residual risk in self-supervised accelerated MRI reconstruction**

## Central research question

Do practical model-derived uncertainty scores identify spatial
regions with elevated independent held-out k-space residual risk
in self-supervised accelerated brain MRI?

## Relationship to Paper 1

Paper 1 introduced a four-way acquired k-space partition:

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

Paper 2 uses the independent target derived from
\(\Lambda_{\mathrm{hold},v}\) to benchmark uncertainty scores.

Paper 2 does **not** claim that the held-out residual-risk target is:

- true image-domain reconstruction error;
- clinical uncertainty;
- diagnostic ground truth;
- a guarantee of reconstruction safety.

## Core methods

| Method ID | Symbol suffix | Method |
|---|---|---|
| C0 | `C0` | Deterministic A4 residual-risk predictor |
| U1 | `MC` | MC-dropout predictive dispersion |
| U2a | `PE` | Point-predictor ensemble disagreement |
| U2b | `DE` | Probabilistic deep ensemble |
| B1–B6 | `B1`–`B6` | Deterministic image and geometry descriptors |

## Evaluation tasks

### Task P — prediction quality

Evaluates the mean residual-risk prediction against the independent
target \(u_{\mathrm{hold},v}\).

### Task R — residual-risk localization

Evaluates whether uncertainty score \(U_{j,v}\) identifies pixels or
pooled regions with elevated \(u_{\mathrm{hold},v}\).

Primary endpoint: calibration-fixed high-risk AUPRC.

### Task E — prediction-error awareness

Evaluates whether \(U_{j,v}\) ranks the absolute prediction deviation

\[
d_{j,v}
=
\left|
u_{\mathrm{hold},v}
-
\mu_{j,v}
\right|.
\]

Primary endpoint: normalized area under the sparsification error curve
(AUSE).

## Data governance

The preferred Paper 2 split is:

| Cohort | Volumes | Use |
|---|---:|---|
| \(D_{\mathrm{fit}}\) | 181 | Model fitting |
| \(D_{\mathrm{dev}}\) | 20 | Early stopping and method development |
| \(D_{\mathrm{cal}}\) | 40 | Threshold and interval calibration |
| \(D_{\mathrm{test}}\) | 40 | Final independent internal testing |

The final test cohort remains closed until all models, thresholds,
metrics, seeds, and calibration artifacts are frozen.

## Reproducibility rules

1. Every experiment is driven by a committed YAML configuration.
2. Every result directory stores the resolved configuration and provenance.
3. Large MRI data, caches, and private checkpoints are never committed.
4. All reported metrics are calculated at volume level or use
   volume-level resampling.
5. All figures and tables are generated from committed scripts.
6. One symbol has one meaning throughout code, equations, tables,
   figures, and manuscript text.
7. Changes after protocol freeze are recorded in
   `reports/decisions/DECISIONS.md`.

## Current status

Repository foundation stage.

No Paper 2 model has been trained and no final Paper 2 test result has
been inspected.

## Hardware progression

- Repository creation, audits, split generation, and unit tests: CPU.
- C0 and initial U1 development: T4 or L4 GPU.
- Final five-member ensembles: L4, A100, or sequential T4 training,
  depending on measured memory and runtime.
