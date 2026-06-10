# 2026-06-07 Action List

## Current Protected Anchor

- Public best: `13.986468676098955`
- File: `data/submissions/nir_yj_oof_affine_s0p10_mc1.csv`
- Method: `SG9 -> SNV -> PCA20 -> Yeo-Johnson target -> Ridge3500`, plus OOF affine correction.

## Submit-Ready Candidate From 2026-06-06

Primary candidate:

- File: `data/submissions/nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p1_mc1.csv`
- Method: current anchor + tiny residual distillation from `MSC + SG9 + raw-target PLS4`
- Diagnostics:
  - anchor diff RMSE: `0.0684`
  - max diff: `0.1311`
  - max species mean shift: `0.0207`
  - OOF delta: `-0.0227`
  - negative count: `0`

Safety candidate:

- File: `data/submissions/nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p08_mc1.csv`
- Method: same as primary, smaller clip
- Diagnostics:
  - anchor diff RMSE: `0.0569`
  - max diff: `0.1057`
  - max species mean shift: `0.0187`
  - OOF delta: `-0.0178`
  - negative count: `0`

## New Ideas To Add

### 1. Test-Like Golden Subset

The previous golden/robust subset search mainly judged train samples from train-only quality signals:

- OOF residual
- PCA leverage
- local label inconsistency
- KS/SPXY representative weighting

The untested variant is different: use train+test spectral proximity to find train samples that look like the test distribution.

Planned workflow:

1. Build unsupervised representations with `SG9+SNV` and `MSC+SG9`.
2. Fit PCA or PLS-like latent space on train+test spectra without using test targets.
3. Cluster train+test together, or compute kNN distances from each train sample to test samples.
4. Score train samples by `test_likeness`.
5. Build golden subset or weights:
   - high `test_likeness`: keep or upweight
   - low `test_likeness`: downweight
6. Train branch models:
   - `MSC+SG9 PLS4`
   - `SG9+SNV PLS4`
   - optional `SG9+SNV PCA20 YJ Ridge3500`
7. Do not submit branch predictions directly.
8. Distill branch residual signal into the current anchor with small shrink/clip.

Gates:

- anchor diff RMSE: `0.03-0.12`
- max diff: `<0.18`
- max species mean shift: `<0.04`
- mean diff near `0`
- negative count: `0`
- branch signal should beat or complement the current `MSC+SG9 PLS4` signal.
- reject if selected/upweighted train samples are dominated by one train species.

### 2. Index/Sample-Order Moisture Monotonicity

Hypothesis to test:

> If the index/sample number goes down, moisture also goes down.

Equivalently, within a comparable block such as each test species:

```text
lower sample/index number -> lower predicted moisture
higher sample/index number -> higher predicted moisture
```

This is the opposite direction from the older `sort_decreasing` postprocess, so both directions must be diagnosed explicitly.

Planned workflow:

1. Start from the current anchor and the PLS residual-distillation candidates.
2. For each test species block, inspect predicted moisture vs `sample number`.
3. Create constrained postprocess variants:
   - monotonic increasing by sample number within species
   - monotonic decreasing by sample number within species
   - soft monotonic correction, not full sort
4. Prefer isotonic or shrinked rank correction over hard full sorting.
5. Compare against anchor drift and species shift.

Candidate variants:

- hard increasing sort within test species:
  - lower sample number gets lower prediction
  - higher sample number gets higher prediction
- hard decreasing sort within test species:
  - old direction, kept only as diagnostic
- soft increasing sort:
  - `pred_new = (1-w) * pred + w * monotonic_sorted_pred`
  - try `w=0.05, 0.10, 0.20`
- isotonic increasing within species:
  - fit no-label isotonic projection to predictions ordered by sample number
  - then shrink toward original prediction

Gates:

- anchor diff RMSE: `<0.10` for soft variants, `<0.20` for diagnostic hard sort
- max diff: `<0.25`
- max species mean shift: `<0.04`
- top/bottom decile shift: `<0.08`
- no prediction range collapse
- reject if hard sorting changes too many samples or reverses high-confidence extremes.

## Recommended Work Order

1. Run final gate for the submit-ready `MSC+SG9 PLS4` candidate.
2. Implement/test `test-like golden subset` branch distillation.
3. Review results with the reviewer before any submission.
4. Implement/test sample/index monotonicity postprocess.
5. Review results with the reviewer before any submission.
6. Submit only one candidate at a time.

## Current Priority

Do not replace the current anchor. Treat every new idea as:

```text
current anchor + small, reviewed correction
```

The best known correction remains `MSC+SG9 raw-target PLS4 residual distillation`.

## Local Exploration Results From 2026-06-06 Evening

### Test-Like Golden Subset

Implemented:

- `scripts/nir_testlike_golden_distill_search.py`
- `scripts/nir_testlike_golden_focused_search.py`

Main outputs:

- `outputs/nir_testlike_golden_distill/20260606_184903/testlike_golden_distill_summary.csv`
- `outputs/nir_testlike_golden_distill/20260606_191547/testlike_golden_distill_summary.csv`

Result:

- Useful branch exists, but it does not beat the existing `MSC+SG9 PLS4 residual` candidate.
- Best safe branch:
  - `test-like golden keep80/c3 s0.03 c0.08`
  - anchor diff RMSE: `0.0641`
  - max diff: `0.0980`
  - max species mean shift: `0.0207`
  - OOF delta: `-0.0174`
  - signal/residual corr: `0.3209`
- Alternate:
  - `test-like golden keep80/c3 s0.02 c0.10`
  - anchor diff RMSE: `0.0730`
  - max diff: `0.1228`
  - max species mean shift: `0.0256`
  - OOF delta: `-0.0200`

Reject:

- `keep75/c2` and `keep80/c2`
  - corr is high, but species shift is too large.
- Ridge/YJ test-like branches with negative beta/corr.
  - They look like inverse correction around the current YJ/Ridge anchor, not a new robust signal.

### Sample/Index Monotonicity

Implemented:

- `scripts/nir_sample_order_monotonic_search.py`

Output:

- `outputs/nir_sample_order_monotonic/20260606_191206/sample_order_monotonic_summary.csv`

Result:

- User hypothesis direction was rejected locally:
  - `sample_number` increasing -> moisture increasing
  - anchor soft increasing `w0.03`: RMSE drift `1.5425`, max drift `3.8710`
  - anchor isotonic increasing `w0.03`: RMSE drift `0.8724`, max drift `3.4817`
- The opposite direction is already mostly satisfied and only gives tiny correction:
  - anchor isotonic decreasing `w0.03`: RMSE drift `0.0183`, max drift `0.2193`
  - PLS primary + isotonic decreasing `w0.03`: anchor RMSE `0.0708`, max `0.2883`, species `0.0207`

Decision:

- Monotonic increasing is rejected for submission.
- Decreasing isotonic is a polish idea only, not a root method.

### Bayesian / Robust Linear Branches

Implemented:

- `scripts/nir_bayes_branch_distill_search.py`

Output:

- `outputs/nir_bayes_branch_distill/20260606_192219/bayes_branch_distill_summary.csv`

Result:

- `BayesianRidge`, `ARDRegression`, and `LassoLarsIC` did not produce a strong independent signal.
- Best practical branch was `Huber raw SG9/SNV PCA15`:
  - anchor diff RMSE: `0.0227`
  - max diff: `0.0721`
  - max species mean shift: `0.0183`
  - OOF delta: `-0.0134`
  - signal/residual corr: `0.0971`
  - correction corr with PLS primary: about `0.80`

Decision:

- Keep Huber as a weak insurance candidate.
- Do not prioritize Bayesian/ARD for submission; too close to the PLS direction or too small.

### Nonlinear Latent Branches

Implemented:

- `scripts/nir_kernel_branch_distill_search.py`

Output:

- `outputs/nir_kernel_branch_distill/20260606_193058/kernel_branch_distill_summary.csv`

Result:

- `KernelRidge RBF` gave large OOF improvements in some cases, but most useful-looking rows had negative beta/corr and are rejected.
- `SVR RBF` produced a small but more independent signal.
- Best safe nonlinear candidate:
  - `SVR raw MSC+SG9 PCA12 C3 epsilon0.5 s0.01 c0.08`
  - anchor diff RMSE: `0.0276`
  - max diff: `0.0663`
  - max species mean shift: `0.0337`
  - OOF delta: `-0.0049`
  - signal/residual corr: `0.0961`
  - correction corr with PLS primary: about `0.37`

Decision:

- Keep SVR as a weak but somewhat independent nonlinear candidate.
- Reject negative-beta KRR rows even when OOF delta is large.

## Submission Priority For 2026-06-07

1. `data/submissions/nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p1_mc1.csv`
   - First candidate.
   - Current anchor + `MSC+SG9 raw-target PLS4` residual distillation.
   - Best balance of OOF delta, drift, and simple explanation.

2. `test-like golden keep80/c3 s0.03 c0.08`
   - Safe second candidate if the first candidate underperforms or if a different signal is needed.
   - File: `data/submissions/nir_20260607_tlg_keep80_c3_s0p03_c0p08.csv`

3. `test-like golden keep85/c4 s0.015 c0.10`
   - More aggressive golden-subset branch.
   - Use only after reviewing Public result from candidate 1 or 2.
   - File: `data/submissions/nir_20260607_tlg_keep85_c4_s0p015_c0p10.csv`

4. `SVR raw MSC+SG9 PCA12 C3 epsilon0.5 s0.01 c0.08`
   - Small nonlinear independent signal.
   - Lower priority because OOF delta is weak.
   - File: `data/submissions/nir_20260607_svr_msc_sg9_p12_safe.csv`

5. PLS primary + decreasing isotonic `w0.03`
   - Polish only.
   - Use only if Public indicates PLS primary direction is correct and a tiny postprocess is worth testing.

6. `Huber raw SG9/SNV PCA15`
   - Weak insurance candidate.
   - Not a root-method priority.
   - File: `data/submissions/nir_20260607_huber_sg9_snv_p15.csv`

## Submission File Checks

All copied 2026-06-07 candidates passed:

- `sample_submit.csv` order match: yes
- shape: `550x2`
- negative predictions: `0`

Diagnostics vs current anchor:

| File | Diff RMSE | Max Diff | Species Shift |
|---|---:|---:|---:|
| `nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p1_mc1.csv` | `0.068368` | `0.131052` | `0.020727` |
| `nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p08_mc1.csv` | `0.056861` | `0.105657` | `0.018664` |
| `nir_20260607_tlg_keep80_c3_s0p03_c0p08.csv` | `0.064066` | `0.097973` | `0.020747` |
| `nir_20260607_tlg_keep85_c4_s0p015_c0p10.csv` | `0.071756` | `0.150347` | `0.048701` |
| `nir_20260607_svr_msc_sg9_p12_safe.csv` | `0.027598` | `0.066278` | `0.033725` |
| `nir_20260607_huber_sg9_snv_p15.csv` | `0.022686` | `0.072129` | `0.018290` |

## Reviewer Decisions

- Reviewer: Boole (`019e9c52-9b48-7a53-88fe-e550571cf458`)
- Confirmed:
  - PLS primary remains first submission candidate.
  - test-like golden is useful but second priority.
  - monotonic increasing should be rejected.
  - Bayes/Huber is not independent enough.
  - SVR is worth retaining as a weak nonlinear branch.
  - KRR negative-beta rows are rejected despite attractive OOF deltas.

## 2026-06-06 Late Broad Search Update

New scripts:

- `scripts/nir_broad_signal_search.py`
- `scripts/nir_adv_golden_focused_search.py`

Broad search result:

- `730` broad candidates completed.
- Top family was `adv_weighted_golden`.
- Interpretation:
  - Build a target-free train-vs-test domain classifier.
  - Rank train rows by test-likeness.
  - Weight the regression branch toward test-like training rows.
  - Distill only a small correction into the current public-best anchor.

Focused search result:

- `1080` focused adversarial-golden candidates completed.
- Best safe-ranked candidate:
  - `nir_advgold_huber_msc_sg9_logistic_dc6_w0p65to1p65_c6_b0p4452_s0p018_c0p08_mc1`
  - diff RMSE: `0.05945`
  - max diff: `0.08646`
  - species shift: `0.00862`
  - OOF delta: `-0.01377`
  - weight audit: `ESS/n=0.941`, `corr(weight,y)=-0.075`, max weighted species share `0.164`
- Best aggressive candidate:
  - `nir_advgold_huber_msc_sg9_rf_dc6_w0p5to2_c5_b1_s0p02_c0p1_mc1`
  - copied submission file: `data/submissions/nir_20260606_advgold_huber_msc_sg9_rf_dc6_c5_s0p02_c0p1.csv`
  - diff RMSE: `0.08214`
  - max diff: `0.11236`
  - species shift: `0.01848`
  - OOF delta: `-0.02454`
  - fold improvement: `4/5`
  - top-10 correction concentration: max species count `3/10`
  - weight audit: `ESS/n=0.893`, max weighted species share `0.175`, but `corr(weight,y)=0.310`

Decision:

- `adv_weighted_golden` is the most promising new root direction found today.
- The aggressive RF version is locally strongest, but the weight-target correlation is too high for a clean submit without accepting extra risk.
- The logistic version is safer and passes weight audit, but it is weaker than the existing operator-PLS residual.

Updated submission posture:

1. If the goal is the next highest-upside single submit, use `data/submissions/nir_20260606_advgold_huber_msc_sg9_rf_dc6_c5_s0p02_c0p1.csv` only with the known `corr(weight,y)=0.310` risk.
2. If the goal is avoiding wasted submits, keep the existing operator-PLS residual as the first candidate and use adversarial-golden as the next family to refine.
3. Do not submit the weak safe adversarial-golden candidate unless a conservative non-PLS-like sanity check is needed.

Final reviewer decision:

- Reviewer: Aristotle (`019e9d37-36e8-72e3-82d5-de456fb10b93`)
- Adopt the aggressive adversarial-golden candidate as the single next submit candidate.
- Rationale:
  - It has the strongest OOF delta among today's focused candidates.
  - It passes ESS, species-share, fold-stability, top-correction concentration, and fallback checks.
  - The only major warning is `corr(weight,y)=0.310`, so the submit should be treated as an attack on the golden-subset/domain-weighting hypothesis, not a guaranteed safe polish.

## 2026-06-07 Public Feedback And Follow-Up

Submitted aggressive adversarial-golden:

- File: `data/submissions/nir_20260606_advgold_huber_msc_sg9_rf_dc6_c5_s0p02_c0p1.csv`
- Public: `13.993686373474532`
- Result: slight worse than current best `13.986468676098955`
- Interpretation:
  - The weighted-golden branch did not collapse.
  - The likely failure mode is over-aggressive correction and/or `corr(weight,y)=0.310`.

Follow-up submitted candidate:

- File: `data/submissions/nir_20260607_advgold_rf_shrink_s0p012_c0p06.csv`
- Method: same RF weighted-golden Huber branch, smaller residual correction.
- Public: `13.990746861581181`
- Diagnostics:
  - diff RMSE: `0.049285`
  - max diff: `0.067418`
  - species shift: `0.01109`
  - OOF delta: `-0.014752`
- Reviewer decision:
  - If continuing weighted-golden today, this is the best next test.
  - It specifically tests the hypothesis that the prior Public miss was overcorrection.
  - Warning remains: same branch, so `corr(weight,y)=0.310` risk is not removed.

Post-result reviewer update:

- Reviewer: Aristotle (`019e9d37-36e8-72e3-82d5-de456fb10b93`)
- Decision: stop submitting RF weighted-golden variants.
- Reason:
  - Aggressive RF: `13.993686373474532`
  - Shrunk RF: `13.990746861581181`
  - Shrinking improved the miss but still did not beat current best `13.986468676098955`.
  - Further shrink is likely to become a near-anchor no-op rather than a winning correction.
- Future weighted-golden work is internal-only until it satisfies:
  - `corr(weight,y) <= 0.15-0.20`
  - `ESS/n > 0.85`
  - fold improvement at least `4/5`
  - lower correlation with existing PLS residual, or a clearly better Public-risk story.

## 2026-06-07 Operator Residual Public Feedback

Submitted operator residual primary:

- File: `data/submissions/nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p1_mc1.csv`
- Public: `13.990521171208956`
- Result: worse than current best `13.986468676098955`
- Interpretation:
  - Weighted-golden and operator-residual small corrections both landed around `13.99`.
  - This suggests the current anchor-protected residual-correction family is locally saturated.
  - Further shrink/clip tuning is likely to become near-anchor no-op rather than a meaningful rank move.

Action update:

- Stop submitting small anchor residual corrections unless a new branch has clearly different signal and stronger internal evidence.
- Next work should be internal-only and should look for a genuinely different base method or a revised anchor, not another small correction around the same anchor.

## 2026-06-07 Broad Local Search Update

Current best remains:

- `data/submissions/nir_yj_oof_affine_s0p10_mc1.csv`
- Public: `13.986468676098955`
- Method: `SG9 -> SNV -> PCA20 -> Yeo-Johnson Ridge alpha=3500 -> OOF affine shrink=0.10`

Broad families tested locally:

- Nonlinear latent direct / calibrated direct:
  - Scripts: `scripts/nir_nonlinear_latent_guard_search.py`, `scripts/nir_nonlinear_calibrated_guard_search.py`
  - Result: stopped.
  - Reason: some OOF signal appeared, but test predictions were badly miscalibrated (`diff` often 5-30+, species/range collapse). OOF calibration did not recover a submit-safe distribution.

- Spectral feature direct replacement:
  - Script: `scripts/nir_spectral_feature_direct_guard_search.py`
  - Result: stopped.
  - Reason: DCT/contrast/shape direct models often improved OOF but produced large test shifts (`diff` 10+ for many candidates).

- Base estimator shape augmentation:
  - Script: `scripts/nir_base_shape_aug_guard_search.py`
  - Result: best broad candidate found today.
  - Candidate copied to:
    - `data/submissions/nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv`
  - Method:
    - keep base `SG9 -> SNV -> PCA18 -> YJ Ridge alpha=3500`
    - concatenate `shape14` spectral summary features scaled by `0.5`
    - apply OOF affine shrink `0.12`
  - Diagnostics vs current best anchor:
    - diff RMSE `0.1730`
    - max diff `1.0762`
    - species shift `0.0758`
    - bad-alpha3000 corr `0.3186`
    - weighted/operator corr `0.4540 / 0.4097`
    - affine OOF delta `-0.0457`, fold improvement `4/5`, worst fold `+0.0471`
    - range ratio `1.0083`, std ratio `1.0040`
    - top10 abs diff concentration: max species count `5/10` (`species 7`)

Reviewer decision:

- If submitting one broad, non-local candidate, use `nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv`.
- Treat it as risky because `max_diff > 1.0` and top10 species concentration are outside the conservative gate.
- Do not submit near-anchor variants (`diff < 0.05`); they are likely no-op submissions.

## 2026-06-07 Public Update And Test-Like Golden Rework

New current best:

- File: `data/submissions/nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv`
- Public: `13.94729351406489`
- Interpretation:
  - `PCA18 + shape14(scale=0.5)` improved the old `13.986468676098955` anchor.
  - The improvement is meaningful but still a near-anchor refinement, not a new model family.

New golden/test-like search:

- Script: `scripts/nir_testlike_weight_shape_search.py`
- Base estimator held fixed around the current best:
  - `SG9/SNV -> PCA/shape14 -> Yeo-Johnson Ridge -> affine shrink`
- Changed only `sample_weight`.
- Signals tested:
  - test-likeness from kNN distance to test spectra in the current latent+shape feature space
  - Yeo-Johnson OOF residual downweight
  - weak species mean cap / full species balancing

Findings:

- Full species balancing removed most of the signal; do not use as the main path.
- Residual downweight plus test-like upweight produced strong nested OOF deltas, but only `2/5` folds improved and weight concentration remained; keep as research evidence, not first submission.
- Best attack candidate:
  - `data/submissions/nir_tlw_knnk10_top0p2_up1p15_g1_20260607.csv`
  - Method: test-like kNN `k=10`, top `20%` train upweighted to max `1.15`, no residual downweight
  - Diagnostics:
    - diff RMSE `0.0576`
    - max diff `0.1434`
    - species shift `0.0665`
    - direct OOF delta `-0.0079`
    - nested affine OOF delta `-0.0085`
    - fold improvement `4/5`
    - weight-y corr about `-0.002`
    - remaining risk: top10 species concentration and high-weight concentration
- Safer but smaller candidate:
  - `data/submissions/nir_tlw_acap_k10_top0p15_up1p12_p20_20260607.csv`
  - Method: kNN `k=10`, top `15%`, max upweight `1.12`, `PCA20`
  - Diagnostics:
    - diff RMSE `0.0425`
    - max diff `0.1154`
    - species shift `0.0506`
    - nested affine OOF delta `-0.0139`
    - fold improvement `3/5`
    - risk: still top10/high-weight concentration, and the diff may be too small to move Public much

Reviewer decision:

- Prefer the original attack candidate if spending a submission:
  - `nir_tlw_knnk10_top0p2_up1p15_g1_20260607.csv`
- Reason:
  - It has the best balance of movement, small max shift, and `4/5` fold stability.
  - The residual-downweight B candidates are internally impressive but too fold-local.
- If preserving submissions is more important than attacking, hold both and continue improving the concentration control.

## 2026-06-07 Direction Switch: Minority / High-Impact Local Correction

Public feedback:

- Submitted: `data/submissions/nir_tlw_knnk10_top0p2_up1p15_g1_20260607.csv`
- Public: `13.949344918768173`
- Current best remains:
  - `data/submissions/nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv`
  - Public `13.94729351406489`
- Interpretation:
  - Broad test-like soft weighting is effectively saturated.
  - The next useful direction is not global distribution matching, but minority/high-impact test regions.

External research summary:

- NIR chemometrics has a long-running local calibration line:
  - LOCAL / locally weighted PLS chooses similar calibration samples for each unknown sample.
  - Locally biased regression applies local skew/bias correction to a global model.
  - Minority/rare spectral settings can be improved by explicitly increasing local/rare information, but this must be tightly guarded.

Local EDA:

- Current test prediction quantiles:
  - median `20.20`
  - p90 `87.88`
  - p95 `98.55`
  - max `141.41`
- Train target is much wider:
  - p95 `152.70`
  - max `298.58`
- High prediction top 30 test rows:
  - species 2: `20`
  - species 18: `4`
  - species 7: `4`
  - species 9/10: `1` each
- Most isolated top 30 test rows in current feature space:
  - species 6: `20`
  - species 7: `9`
  - species 10: `1`
- Within every test species, `sample number` and current prediction have strong negative correlation:
  - species 2: `-0.877`
  - species 6: `-0.985`
  - species 7: `-0.695`
  - species 9: `-0.911`
  - species 10: `-0.837`
  - species 18: `-0.930`

Reviewer-prioritized next directions:

1. High-prediction / top-decile gated local correction.
   - Do not condition directly on species 2.
   - Select by anchor prediction top `5-10%`, local branch disagreement top `10%`, or high local uncertainty.
   - If species 2 appears often, that is an outcome, not a hand-coded rule.
2. Species 7 isolated high end.
   - Interesting, especially sample numbers around `544-549`.
   - Do not use raw sample IDs directly in a submit candidate.
   - Use general conditions such as isolated + high disagreement + edge-like prediction.
3. Species 6 isolated region.
   - Diagnose first.
   - It may be an entire domain-shifted test species rather than a small correctable minority.
4. Sample-order monotonic correction.
   - Use only as a diagnostic or weak gate.
   - The anchor already captures strong monotonic structure, so direct sorting is risky.

First implementation plan:

- Keep the current best anchor unchanged.
- Build local branch predictions:
  - feature spaces:
    - current `SG9/SNV -> PCA18 + shape14`
    - optional `MSC+SG9 -> PCA`
  - local models:
    - kNN local Ridge / PLS
    - `k=40/80/120`
    - distance weighted
- Learn correction by OOF:
  - `delta = local_pred - anchor_pred`
  - fit one small positive beta from OOF residuals
  - apply only to gated test rows
- Gate candidates:
  - anchor prediction top `5/10%`
  - local-anchor disagreement top `10/15%`
  - isolated/high-distance top `10%`
  - max corrected rows `<=15%`
- Correction limits:
  - shrink `0.02/0.04/0.06`
  - clip `0.10/0.20/0.30`

Hard gates:

- sample order OK, no negative predictions
- corrected rows `<=15%`
- max correction `<0.35`
- anchor diff RMSE `0.02-0.12`
- species mean shift `<0.06`
- top10 correction species concentration `<=4/10`
- OOF delta `<=0.01`
- improved folds `>=3/5`
- beta positive
- bad-alpha3000 corr `<0.35`

Decision:

- Next implementation should be **gated local residual correction**, not local full replacement.
- First target is high-prediction / high-disagreement minority rows.
- Avoid direct species or sample-number rules in submit candidates.

## 2026-06-07 Stage-1.5 / Stage-2 Hard-Case Detector Results

Implemented:

- `scripts/nir_hard_case_detector_diagnostics.py`
- `scripts/nir_stage2_hardcase_correction_search.py`

Stage-1.5 hard-case detector:

- Label:
  - top `20%` current-anchor OOF absolute residual
- Features:
  - anchor prediction and rank
  - local branch disagreement
  - local isolation distance
  - branch disagreement summary
- Best detector:
  - LogisticRegression, hard label top `20%`
  - top `5%` detector gate
- OOF diagnostics:
  - AUC `0.781`
  - AP `0.507`
  - precision@top5% `0.742`
  - selected residual lift `2.39`
  - fold precision min `0.571`
  - fold precision mean `0.769`
- Test top5% gate:
  - rows: `28`
  - species counts: `{2:5, 6:10, 7:9, 9:2, 10:1, 18:1}`

Stage-2 correction:

- Gate:
  - LogisticRegression hard-case score top `5%`
- Branch:
  - local raw Ridge, `k=40`, alpha `1`
- Correction:
  - `anchor + clip(shrink * beta * (local - anchor))`
  - beta from OOF: `0.8455`
  - signal/residual corr on gate: `0.7227`
- Internal results:
  - clip `0.10`
    - output: `data/submissions/nir_s2_hardcase_logreg80_top5_rawk40_clip0p10_20260607.csv`
    - diff RMSE `0.0226`
    - max diff `0.10`
    - species shift `0.0159`
    - OOF delta `-0.0063`
    - folds `5/5`
  - clip `0.15`
    - output: `data/submissions/nir_s2_hardcase_logreg80_top5_rawk40_clip0p15_20260607.csv`
    - diff RMSE `0.0338`
    - max diff `0.15`
    - species shift `0.0238`
    - OOF delta `-0.0094`
    - folds `5/5`
  - clip `0.20`
    - diff RMSE `0.0451`
    - max diff `0.20`
    - species shift `0.0317`
    - OOF delta `-0.0122`
    - folds `5/5`

Reviewer decision:

- Stage2 is worth considering.
- First candidate if submitting:
  - `nir_s2_hardcase_logreg80_top5_rawk40_clip0p15_20260607.csv`
- Safer alternative:
  - `nir_s2_hardcase_logreg80_top5_rawk40_clip0p10_20260607.csv`
- Caveat:
  - clip `0.15` corrections are all positive and all hit the clip limit.
  - This supports the high-moisture underprediction hypothesis, but increases public-risk.
  - If spending a cautious submission, use clip `0.10`; if attacking, use clip `0.15`.
