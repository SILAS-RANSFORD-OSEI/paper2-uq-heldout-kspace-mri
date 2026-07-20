# P2-Exp000 Completion Report

## Status

**PASS**

P2-Exp000 completed the provenance, checkpoint, manifest,
semantic, and full-cache integrity audit required before
constructing the Paper 2 data split.

## Canonical Paper 1 source

- Repository:
  `SILAS-RANSFORD-OSEI/fourway-ssdu-reliability-mri-v2`
- Commit:
  `da563ead8fb653539e1eeca29248b31f0121ca12`
- Canonical SSDU checkpoint:
  `outputs/exp004_train_ssdu_v4_full/best_model.pt`
- Canonical A4 checkpoint:
  `outputs/exp008_reliability_ablation_full/A4_image_only/best_model.pt`

## Reliability-cache audit

- Manifest rows: **4,462**
- Files passed: **4,462**
- Files failed: **0**
- Volumes: **281**
- Paper 1 split rows:
  - Train: **3,190**
  - Calibration: **636**
  - Test: **636**

## Matrix-size distribution

- `640 × 272`: **16** slices
- `640 × 320`: **1,660** slices
- `768 × 392`: **176** slices
- `768 × 396`: **2,610** slices

## Semantic contract

Legacy NPZ fields were translated at the cache boundary:

- `x` → six-channel legacy input tensor
- `x[0:3]` → predictor input \(C_v\)
- `y` → \(u_{\mathrm{risk},v}\)
- `y_raw` → \(z_{\mathrm{risk},v}\)

The legacy storage names do not propagate into Paper 2
scientific analysis code.

## Target-transform verification

- Maximum absolute reconstruction error:
  **0.0018019676**
- Maximum file-level 99th-percentile error:
  **0.0014529228**
- Minimum Pearson correlation:
  **0.999999424**

## Resolved investigative warnings

Two exploratory checks produced warnings that were resolved
before completion:

1. Complete fixed-shape signatures differed because fastMRI
   matrix dimensions varied. The correct frozen schema is
   template-based: `(6,H,W)` for the input and `(H,W)` for
   each target.
2. The hypothesis that legacy `y` was a scalar multiple of
   legacy `y_raw` was rejected. Source tracing established
   the correct nonlinear quantile-normalized logarithmic
   transformation.

Neither warning represents an unresolved cache defect.

## Governance state

- Paper 2 split constructed: **No**
- Paper 2 models trained: **No**
- Calibration frozen: **No**
- Final test barrier: **Closed**
