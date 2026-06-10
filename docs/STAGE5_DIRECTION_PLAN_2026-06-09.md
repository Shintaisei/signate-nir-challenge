# Stage5 Direction Plan 2026-06-09

## Current State

Current best Public:

- `13.922966797992677`
- File: `data/submissions/nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv`

The 2026-06-09 five-submission queue all improved the prior best. The most aggressive candidate, q5, won.

Interpretation:

- The `test-near golden PLS residual + cluster-capped correction` direction is still alive.
- The prior gate was probably too conservative.
- Stronger OOF residual corrections can work if concentration guardrails remain strict.
- The next search must avoid only re-amplifying q5 rows.

## Research Readout

Relevant external findings:

- NIR chemometrics literature explicitly treats `LOCAL` as PLS with a closest-neighbour calibration subset for each predicted sample. This supports the current test-near/local PLS direction.
  - Source: https://journals.sagepub.com/doi/10.1255/jnirs.283
- Soil spectroscopy MBL/local modeling builds a model per prediction sample from nearest reference neighbours; variants include PLS and weighted average PLS.
  - Source: https://whrc.github.io/Soil-Predictions-MIR/mbl-models.html
- Chemometrics sample selection commonly uses Kennard-Stone/SPXY to improve spectral-space coverage and reduce extrapolation risk.
  - Source: https://chemotools.org/explore/astartes.html

Local implication:

- Continue using local/test-near PLS residual correction.
- Add a separate hard-case detector axis, but use it as a residual gate, not as a direct prediction replacement.
- Keep spectral-space diversity and concentration controls because sample selection and local modeling can overfit uncovered regions.

## Local Diagnostics

Diagnostic script:

- `scripts/nir_stage5_direction_diagnostics.py`

Outputs:

- `outputs/stage5_direction/stage5_direction_summary.json`
- `outputs/stage5_direction/stage5_row_diagnostics.csv`
- `outputs/stage5_direction/q5_changed_rows.csv`
- `outputs/stage5_direction/uncorrected_high_detector_rows.csv`
- `outputs/stage5_direction/pred_bin_detector_summary.csv`
- `outputs/stage5_direction/detector_component_corr.csv`
- `outputs/stage5_direction/test_cluster64_detector_summary.csv`

Key findings:

- q5 changed `22` rows.
- q1-q5 changed any row count: `58`.
- Never changed rows: `492`.
- q5 changed detector mean: `0.495`.
- Top 20 never-changed detector mean: `0.905`.
- `detector_score` vs `abs(q5_diff)` correlation: `-0.0067`.

This means the new detector is not simply rediscovering q5 rows. It points to a separate region.

Top uncorrected detector rows:

- Top 50 uncorrected detector rows: `40` チーク, `10` ケヤキ.
- Top clusters: cluster64 `17` has `21`, cluster64 `61` has `15`, cluster64 `37` has `8`.
- The highest uncorrected region is a contiguous チーク low-prediction band around sample numbers `938-963`.

q5 behavior:

- q5 mainly raised high-moisture rows in クスノキ, ケヤキ, スギ, タモ.
- q5 also lowered a few チーク low-prediction rows around sample numbers `959-962`.
- The detector says many neighbouring チーク rows remain suspicious and uncorrected.

## Reviewer Constraints

Do not use directly as hard submission logic:

- species labels
- sample number
- changed-any-after-Public as a selection rule

Allowed:

- Use species/sample/change status for diagnostics.
- Use unsupervised detector scores as a test-side diagnostic.
- For candidate generation, create fold-local OOF detector scores and combine them with residual signal.

Stage5 candidate gate:

- Anchor: q5 current best.
- Diff RMSE vs q5: `0.006-0.020`
- Max diff vs q5: normal `<=0.14`, attack `<=0.16`
- Changed rows: `16-26`
- New rows vs q5: `>=6`
- Prior combined correction corr: `<=0.75`, ideal `<0.65`
- q5 diff corr: `<=0.80`, ideal `<0.65`
- Species shift diagnostic: `<=0.008`
- Corrected species max: `<=6`
- Top10 species max: `<=3`
- Clip saturation: `0`
- OOF delta vs q5: normal `<= -0.0035`, attack `<= -0.005`
- Groups improved/active: `>=9/13`
- Signal corr: `>=0.30`

## Implementation Priority

### 1. Stage5 q5-Anchored Residual Search

Purpose:

- Continue the proven direction, but require lower correlation with q5 rows.

Implementation:

- Add q5 as a new anchor in `scripts/nir_slot1_testnear_branch_search.py`.
- Reconstruct q5 OOF by applying the Stage4 q5 correction on top of Stage3 OOF.
- Search branch variants around q5:
  - cluster cap `2/3`
  - gate top `3.5-5.0%`
  - shrink `0.012-0.05`
  - clip `0.14-0.20`
  - keep fraction `0.55-0.80`
  - PLS components `4-6`
- Add q5 correlation diagnostics:
  - q5 diff corr
  - changed-row overlap with q5
  - new rows vs q5

Submit only if it is not merely q5 amplification.

### 2. Detector-Gated Residual Search

Purpose:

- Use the uncorrected high-detector チーク/ケヤキ region without hardcoding species/sample number.

Implementation:

- Build fold-local detector features:
  - PCA Q residual
  - train-neighbour distance
  - LOF score
  - IsolationForest score
  - rank-averaged detector score
- For test, fit detector on train spectra and score test spectra.
- Candidate gate should combine:
  - residual branch signal magnitude
  - detector score top fraction
  - cluster cap
  - q5 low-overlap guard
- Start with detector only as an intersection filter:
  - `abs(signal) top X% AND detector top Y%`
  - `abs(signal) top X% OR detector top very small Y%` only as attack branch.

Do not use detector alone to decide correction direction.

### 3. Local MBL-Style Branch

Purpose:

- Move beyond global branch predictions by fitting per-test local models.

Implementation:

- Build local predictions in SNV/PCA/PLS space:
  - nearest train neighbours `k=40/60/80/100`
  - optional distance weighting
  - local Ridge/PLS residual branch
- Use output only as a residual signal against q5.
- Do not directly replace q5 predictions.

Priority:

- Third priority because it is more expensive and easier to overfit.
- Worth trying if Stage5 and detector-gated residual search saturate.

## Next Candidate Families

1. `s5_q5_anchor_residual`
   - q5 anchor, standard residual branch, stricter q5-overlap guard.
   - Most likely to give another small improvement.

2. `s5_detector_intersection`
   - q5 anchor, branch signal intersected with fold-local detector top rows.
   - Best chance to avoid局所解 because it targets rows q5 did not explain.

3. `s5_detector_teak_lowband_attack`
   - Same as detector intersection, but tuned to allow low-prediction detector rows.
   - Must avoid species/sample-number hardcoding; use prediction band + detector + cluster cap.
   - Attack only.

4. `s5_mbl_signal`
   - Local nearest-neighbour PLS/Ridge residual signal.
   - Use only if it passes OOF and q5-correlation gates.

## Immediate Next Work

1. Patch the Stage4 search script to support a q5 anchor and q5 OOF reconstruction.
2. Implement fold-local detector scores.
3. Run two grids:
   - q5-anchored standard Stage5
   - q5-anchored detector-intersection Stage5
4. Select at most 3 candidates:
   - one safe q5 continuation
   - one detector-diverse candidate
   - one attack candidate
5. Do not submit until the gate report is reviewed.
