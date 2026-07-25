# P2-Exp004B calibration freeze v1.1

This directory freezes the precision-audited calibration rules
computed exclusively from `D_cal`.

## Frozen identifiers

- Parent commit: `21f48ea244e2a4493474a5e77dfc75c7367938cd`
- Artifact SHA-256: `f39b8274006328bd7a3b3dd74f91cb496957c25d705316ac89a9bcaebb4058ed`
- Schema: `exp004b-calibration-v1.1`
- Tag: `exp004b-calibration-v1.1`

## Quantile implementation

- Observations: float32
- Order statistics: exact
- Linear interpolation: float64
- Stored thresholds: float64
- Label operator: `>=`

## Task R

One common `u_hold` threshold is applied to all uncertainty
methods.

- Q0.85: `1.4107755422592163`
- Q0.90 primary: `1.7366089820861816`
- Q0.95: `2.0520856380462646`

## Task E

The primary endpoint is normalized AUSE using continuous
deviation:

`abs(u_hold - raw method mean)`

Method-specific Q0.90 thresholds are retained only for secondary
binary diagnostics. Their binary AUPRC values must not be used
for direct cross-method ranking because the labels differ by
method.

## Statistical inference

The operational thresholds pool supported calibration pixels.
Statistical inference, bootstrapping, confidence intervals, and
paired method comparisons use the volume as the resampling unit.
Pixels are not treated as independent experimental subjects.

## Governance

No model training, model inference, descriptor selection,
performance-based threshold tuning, or `D_test` access occurred
during calibration.
