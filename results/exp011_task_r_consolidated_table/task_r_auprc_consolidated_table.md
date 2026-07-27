| Method | Category | $q_{85}$ AUPRC [95% CI] | $q_{90}$ AUPRC [95% CI] | $q_{95}$ AUPRC [95% CI] |
|---|---|---:|---:|---:|
| C0: Direct residual-risk predictor | Direct predictor | **0.929 [0.914, 0.944]** | **0.828 [0.801, 0.854]** | **0.623 [0.594, 0.652]** |
| U1: MC-dropout uncertainty | Model uncertainty | 0.856 [0.834, 0.877] | 0.714 [0.686, 0.742] | 0.482 [0.456, 0.507] |
| U2a: Point-ensemble between-model variance | Model uncertainty | 0.618 [0.590, 0.645] | 0.399 [0.377, 0.423] | 0.195 [0.185, 0.206] |
| U2b: Probabilistic-ensemble total predictive variance | Model uncertainty | 0.690 [0.649, 0.729] | 0.424 [0.393, 0.455] | 0.191 [0.179, 0.204] |
| B1: Reconstructed-image magnitude | Deterministic descriptor | 0.766 [0.738, 0.793] | 0.577 [0.550, 0.605] | 0.341 [0.326, 0.358] |
| B2: Zero-filled image magnitude | Deterministic descriptor | 0.769 [0.741, 0.797] | 0.580 [0.554, 0.609] | 0.344 [0.328, 0.361] |
| B3: Reconstruction–zero-filled discrepancy | Deterministic descriptor | 0.569 [0.549, 0.591] | 0.433 [0.411, 0.456] | 0.259 [0.244, 0.273] |
| B4: Reconstructed-image gradient magnitude | Deterministic descriptor | 0.758 [0.730, 0.786] | 0.575 [0.542, 0.611] | 0.327 [0.304, 0.354] |
| B5: Analytical PSF descriptor | Deterministic descriptor | 0.510 [0.488, 0.532] | 0.351 [0.330, 0.373] | 0.181 [0.169, 0.194] |
| B6: qPSF/gain descriptor | Deterministic descriptor | 0.750 [0.720, 0.778] | 0.532 [0.503, 0.562] | 0.287 [0.271, 0.305] |

**Table note.** Values are mean volume-level support-weighted AUPRC with 95% paired volume-bootstrap confidence intervals across 40 test volumes. Thresholds were fixed from the calibration split. Higher values indicate stronger localization of elevated measurement-derived residual-consistency risk. Absolute AUPRC values should not be interpreted as directly comparable across thresholds because high-risk prevalence changes with the threshold.
