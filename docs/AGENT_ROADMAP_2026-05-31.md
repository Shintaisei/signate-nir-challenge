# Agent Roadmap 2026-05-31

## Baseline Facts

- Protected fallback: `candidate_linear_1f.csv`.
- Historical Public best: `candidate_linear_1f` = `17.496`.
- Current local baseline:
  - `candidate_linear_1f` group_species RMSE = `32.463523`.
  - `candidate_linear_1f` moisture_quantile RMSE = `31.046655`.
- Train/test species overlap is `0`.
- Several local-good transformations have already failed Public, especially stronger SNV/diff/PLS/top-k style moves.

## Operating Thesis

Do not chase local CV with broad preprocessing or many wavelengths. The safest score-improvement path is to keep the raw single-wavelength structure and test small, physically plausible perturbations whose prediction diff from `candidate_linear_1f` is controlled.

Update after today's Public feedback:

- `candidate_linear_1f_nn_bias` worsened Public to `21.35722898556263`.
- The opposite direction, `candidate_linear_1f_anti_nn_bias_wm025`, improved to `17.150798122243412`.
- Refinement `candidate_linear_1f_anti_nn_bias_wm040` improved further to `17.084250210869193`.
- Therefore anti-nn-bias blending is now a validated base method.

Planning rule:

- Do not spend the main exploration loop only fine-tuning `w`.
- Fine tuning around `w=-0.40..-0.43` is a late-stage optimization / final-slot task.
- While submissions remain, prioritize discovering a second base method that can produce another meaningful Public move.
- A useful new base method should introduce a new correction direction or modeling assumption, not just another nearby weight, threshold, or index.

## Initial High-Value Experiments

### 0. Validated Anti-NN-Bias Blend

This is the current best discovered family:

```text
pred = candidate_linear_1f + w * (candidate_linear_1f_nn_bias - candidate_linear_1f)
```

Measured Public:

- `w=1.00`: `21.35722898556263`
- `w=0.00`: `17.496`
- `w=-0.25`: `17.150798122243412`
- `w=-0.40`: `17.084250210869193`

Use this as the current fallback improvement family. Do not make it the only
active exploration track until the endgame.

### 1. Raw Index-616 Neighborhood

Test only tiny variants around the current raw single-wavelength signal.

Candidate forms:

- Fixed indices `614..618`.
- Mean or median of `614..618`.
- Median or trimmed mean of predictions from neighboring one-feature linear models.

Why:

- Preserves the known Public-winning structure.
- Low implementation cost.
- Small prediction diff is easy to gate.

Gate:

- `0.1 <= public_diff <= 5.0`.
- `moisture_quantile RMSE <= 31.5` preferred.
- `group_species RMSE <= 34.0`.

### 2. Very Weak SNV Blend

Existing `blend_snv25` had worse Public than baseline and `blend_snv50` moves too far from the anchor, but a much weaker blend may reduce local variance without destroying domain alignment.

Candidate forms:

- raw 97.5% + SNV 2.5%.
- raw 95% + SNV 5%.
- raw 90% + SNV 10%.

Gate:

- `public_diff < 5.0`.
- local metrics must not improve only by moving far away from `candidate_linear_1f`.

### 3. One-Feature Robust / Nonlinear Calibration

Keep the same core feature but make the target mapping slightly more robust.

Candidate forms:

- Quadratic single-feature regression.
- Huber single-feature regression.
- Isotonic calibration only if prediction range remains close to baseline.

Gate:

- Prediction range cannot expand materially beyond current `18.7414..210.1525` without Plan approval.
- `public_diff < 8.0`.
- Worst-fold or high-moisture OOF should not worsen.

### 4. Local Mean/Scale Calibration Transfer

Use only index-616 neighborhood statistics to align train/test raw distributions. Avoid whole-spectrum CORAL or full SNV-style corrections.

Candidate forms:

- Standardize train/test using `idx616±0`, `±3`, `±5`, `±9` windows.
- Fit the usual one-feature model after local alignment.

Gate:

- Strongly reject if `public_diff >= 10`.
- Reviewer must check for test-target leakage. Using test spectra is allowed only for unsupervised distribution alignment.

## Rejected Or Low-Priority Paths

- Broad PLS/SNV/diff/top-k models: too much evidence of Public degradation.
- Large group-curve or species-transfer corrections: risky because train/test species do not overlap.
- Nearest-train-species correction without tight shrinkage: can learn species/domain artifacts rather than moisture.
- Public submission based only on local CV improvement.

## Next Executor Task

Implement the raw index-616 neighborhood experiment first.

Suggested ownership:

- `src/pipeline/models/single_feature_linear.py`
- `src/experiments/registry.py`

Required experiment names:

- `candidate_linear_idx614_raw`
- `candidate_linear_idx615_raw`
- `candidate_linear_idx617_raw`
- `candidate_linear_idx618_raw`
- `candidate_linear_idx616_mean5_raw`
- `candidate_linear_idx616_pred_median5`

Required commands:

```powershell
python run.py <experiment> --cv group_species
python run.py <experiment> --cv moisture_quantile
python run.py --public-diff <experiment>
```

The Plan Agent should narrow this list before Executor starts if runtime or implementation cost matters.
