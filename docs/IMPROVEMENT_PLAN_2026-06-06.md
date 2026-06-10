# 2026-06-06 Improvement Plan: Golden Subset on YJ3500

## Summary

Current best submission is fixed as the anchor:

- File: `data/submissions/nir_ms_target2_yeojohnson_pca20_ridge3500.csv`
- Public: `14.008044533067784`
- Pipeline: `SG9 smoothing -> SNV -> PCA20 -> auto Yeo-Johnson target -> Ridge alpha=3500`

The next main direction is **golden training subset / data screening**, inspired by SS4GG MIR Soil Spectroscopy. The transferable idea is not PLSR itself, but that a simple spectral model can win when trained on a more reliable subset of calibration samples.

Do not start 2026-06-06 with `YJ3600`. Keep it as a fallback only. The first implementation should test whether removing or downweighting unreliable train samples improves the already strong YJ3500 base.

## Research Basis

### SS4GG MIR Soil Spectroscopy

The MIR competition winner used a golden training subset, Savitzky-Golay smoothing, and PLSR, achieving final RMSE `8.325 wt%` across dissimilar MIR instruments. The official summary explicitly notes that relatively simple models can perform well with proper data screening and preprocessing.

Source: https://soilspectroscopy.org/community-data-science-competition-results/

### SS4GG NIR Soil Spectroscopy

The NIR competition winners improved over spectra-only baselines by using contextual covariates and localized modeling. This supports the broader principle that domain/context handling matters more than blindly increasing model complexity.

Source: https://soilspectroscopy.org/community-data-science-competition-results/

### General Spectroscopy Modeling

PLSR and related low-dimensional linear methods remain standard because spectra are high-dimensional and collinear. Sample selection, preprocessing choice, and calibration-set quality are recurring performance levers.

Sources:
- https://whrc.github.io/Soil-Predictions-MIR/plsr-models.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10857020/
- https://zenodo.org/records/7599269

## Decisions From 2026-06-05

Keep:

- `SG9 + SNV + PCA20 + auto Yeo-Johnson + Ridge`.
- `YJ3500` as current best and comparison anchor.
- Conservative candidates whose predictions stay close to `YJ3500`.

Reject for now:

- Manual Yeo-Johnson lambda tuning. `lambda=0.03, alpha=3500` worsened to Public `14.03706389176113`.
- EMSC-lite and Tweedie/Gamma GLM. Local diagnostics moved predictions too far from the best.
- Full PLS/MSC/wavelength-selection replacement. Prior Public and local diagnostics were unstable.
- Blending as a main direction.

## Experiment Families

### 1. `golden_hard_trim`

Build group-OOF predictions using the current YJ pipeline, score train samples by absolute OOF residual, then exclude the worst samples and refit.

Grid:

- Trim rates: `2%`, `3%`, `5%`, `8%`
- Ridge alpha: `3400`, `3500`, `3600`
- PCA components: start with `20`; only test `18/22` after a promising trim appears

Priority:

- First inspect `3%` and `5%`.
- Treat `8%` as diagnostic; likely too aggressive for a first submission.

### 2. `golden_resid_leverage_trim`

Combine OOF residual with PCA leverage.

Safe variants:

- Remove samples where residual is top `5%` AND leverage is top `10%`.
- Remove samples where residual is top `3%` AND leverage is top `15%`.

Attack variants, diagnostic only:

- residual top `5%` OR leverage top `1%`.
- Any rule removing a large fraction from one species.

### 3. `golden_soft_weight`

Instead of removing samples, downweight suspicious samples.

Grid:

- Residual top: `3%`, `5%`, `8%`
- Weights: `0.7`, `0.5`, `0.3`
- Alpha: `3500`, optionally `3600`

Use when hard trim changes predictions too much.

### 4. `neighbor_label_inconsistency`

In SG9+SNV+PCA space, compute k-nearest train neighbors and flag labels that are inconsistent with their local neighborhood.

Grid:

- k: `10`, `20`
- Score: `abs(y - median(neighbor_y))`
- Use only as an AND condition with high OOF residual.

Do not submit a candidate based only on neighbor inconsistency.

## Implementation Plan

Create a dedicated script rather than further expanding `nir_foundation_experiments.py`:

- New script: `scripts/nir_golden_subset_search.py`
- Inputs:
  - `data/raw/train.csv`
  - `data/raw/test.csv`
  - `data/raw/sample_submit.csv`
  - current best submission for diagnostics
- Outputs:
  - `data/submissions/nir_golden_*.csv`
  - `outputs/nir_golden_subset/YYYYMMDD_HHMMSS/golden_summary.csv`
  - `outputs/nir_golden_subset/YYYYMMDD_HHMMSS/golden_summary.json`
  - optional `removed_samples_*.csv`

Core functions:

- `make_base_features`: SG9 smoothing, SNV.
- `fit_predict_yj_ridge`: PCA + auto Yeo-Johnson + Ridge.
- `make_group_oof`: GroupKFold by `species number`.
- `compute_leverage`: PCA score Mahalanobis or squared standardized PCA score norm.
- `compute_neighbor_inconsistency`: kNN in PCA space.
- `build_candidates`: hard trim, soft weight, residual+leverage, residual+neighbor.
- `diagnostics`: prediction distribution, best-diff RMSE, max diff, correlation, decile deltas, species removal counts.

## Submission Gates

A candidate can be considered for submission only if all are true:

- CSV has 550 rows, 2 columns, no header.
- `sample_submit.csv` order matches exactly.
- No negative predictions.
- Difference vs `YJ3500`:
  - RMSE between `0.08` and `0.50`
  - correlation `>= 0.999`
  - max absolute difference preferably `< 2.0`
- Prediction distribution remains close:
  - median does not shift by more than about `0.5`
  - max/min do not collapse or expand sharply
- CV is not used for ranking, only for failure detection:
  - group-species RMSE not worse than `YJ3500` by more than about `0.3`
  - species-15-excluded RMSE not worse by more than about `0.3`
- Removed/downweighted samples are not dominated by a single species.

## First Submission Decision Tree

1. If hard trim `3%` or `5%` produces a stable candidate:
   - best-diff RMSE `0.10-0.35`
   - corr `>= 0.9995`
   - CV not worse
   - no species concentration
   Then submit that as slot 1.

2. If hard trim is too aggressive but soft weight is stable:
   - submit `golden_soft_weight` with weight `0.5` or `0.7`.

3. If residual+leverage AND is stable and more conservative than residual-only trim:
   - submit residual+leverage AND.

4. If all golden subset candidates are unstable:
   - do not spend a submission on them.
   - fallback candidate remains `YJ3600`, but only if the user explicitly wants a small continuation test.

## Avoid

- Starting the day with `YJ3600`.
- Repeating manual Yeo-Johnson lambda tuning.
- Large hard trim such as `10%+`.
- Species-level removal.
- Full PLS/MSC/EMSC replacement.
- Neural nets, CNN residuals, or complex local models as submission candidates.
- Blends as the primary explanation for a submission.

## Tomorrow's First Work Block

1. Implement `scripts/nir_golden_subset_search.py`.
2. Generate golden candidates.
3. Review `golden_summary.csv`.
4. Pick exactly one slot-1 candidate or hold submissions.
5. Record every Public result in `outputs/nir_foundation/public_log.csv`.

The intended slot-1 target is not predetermined. It should be selected from the generated golden subset diagnostics, with `YJ3500` protected as the best known anchor.
