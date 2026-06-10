# 2026-06-08 Submission Plan

## Current Protected Best

- Public best: `13.94729351406489`
- File: `data/submissions/nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv`
- Method: `SG9/SNV -> PCA18 + shape14(scale=0.5) -> Yeo-Johnson Ridge alpha3500 -> OOF affine shrink0.12`

## Research/Review Conclusion

The next promising direction is not another global model replacement. Use the current anchor and only correct hard cases.

External spectroscopy patterns supporting this:

- LOCAL / local PLS: predict each unknown sample from spectrally similar calibration samples.
- Locally-Biased Regression: keep the global calibration and estimate a small local bias/skew correction.
- SS4GG soil spectroscopy: winner patterns emphasized trusted samples, smoothing, PLSR, and localization over large generic models.

Reviewer conclusion:

- Prioritize gate improvement before more aggressive residual models.
- Treat one-sided positive correction as a warning, not an automatic reject, because the hard-case detector appears to identify underpredicted high-moisture rows.
- After the 2026-06-07 evening search, prefer the no-anchor-distance `aw0` variant over `aw0p35`.
- Submit one at a time.

## 2026-06-07 Additional Local Search

Three promising families were implemented and screened with strict gates.

1. Signed residual detector gates
   - Added `signed_pos`, `signed_impact_abs`, and `signed_impact_pos`.
   - Result: high positive precision, but the best candidates were too close to the anchor.
   - Best representative: diff RMSE `0.007912`, max diff `0.051201`, OOF delta `-0.001169`, folds `4/5`, but `signal_residual_corr` on gate was negative.
   - Decision: do not use as tomorrow's first submission.

2. Test-like golden quality filtering
   - Added `testlike`, `stable`, and `testlike_stable` train quality modes with `quality_keep` and `quality_power`.
   - Result: `testlike_stable` frequently ranked well, but mostly by becoming conservative / near-anchor.
   - Decision: keep as a diagnostic tool, not a submission candidate yet.

3. Safe local weighted PLS residual
   - Added `pls1`, `pls2`, and `pls3` local residual models.
   - Best strict pass: PLS1, `k=120`, `quality=testlike_stable`, shrink `0.003`, clip `0.15`.
   - Diagnostics: risk `0.413587`, diff RMSE `0.018180`, max diff `0.113068`, OOF delta `-0.004422`, folds `4/5`.
   - Decision: too risky for Slot 1. Use only as an attack branch after a positive Public signal.

4. Hard/high mixture-of-experts search
   - Added `scripts/nir_hardcase_moe_search.py`.
   - Tested stronger detectors for `hard80`, `poshard70`, `high80`, and `under_high`.
   - Best detector signal was `high80_rf`: it finds high-moisture hard cases with high OOF precision.
   - Density-ratio weighted global experts were rejected: ESS was too low, weight-target correlation was high, and expert signal correlation was weak/negative.
   - Local YJ experts worked but have more extrapolation risk.
   - The best conditional attack branch is `high80_rf + lres_pls1`: it keeps the anchor fixed and applies a small correction only to high-moisture hard-case rows.

## Tomorrow Slot 1: Recommended

- File: `data/submissions/nir_s2_lbr_impact_abs_wmean_k40_aw0_top5_s0p003_20260607.csv`
- Experiment: `nir_s2lbr_impact_abs_top0p05_lres_current_wmean_k40_aw0_a0_s0p003_c0p2_soft0`
- Method:
  - Anchor fixed.
  - Hard-case gate: `impact_abs = rank(detector_score) * rank(abs(local_residual_estimate))`, top 5%.
  - Local residual model: weighted mean of anchor OOF residuals.
  - Feature space: `SG9/SNV PCA18 + shape14`; no anchor prediction distance.
  - `k=40`, `shrink=0.003`, `clip=0.2`.

Diagnostics:

- Rows changed: `28`
- Negative predictions: `0`
- Anchor diff RMSE: `0.020080`
- Max diff: `0.152896`
- Clip saturation: `0`
- Correction std on gate: `0.017576`
- Max species mean shift: `0.012482`
- Corrected species max count: `9`
- Top10 correction species max count: `4`
- Signal/residual corr gate/all: `0.484109 / 0.280095`
- OOF delta: `-0.005123`
- Improved folds: `5/5`
- Worst fold delta: `-0.000491`
- Strict reject reason: `one_sided_correction` only.

Risk notes:

- All corrected rows move upward, so this is still an underprediction-correction hypothesis.
- The correction is deliberately small; even if it works, expected Public improvement is likely incremental rather than dramatic.

Submission memo:

```text
hard-case detector impact_abs top5; local residual weighted mean k40; anchor fixed; no anchor_pred distance; shrink0.003; clip0.2; 28 rows corrected
```

## Slot 1 Public Result

- Submitted file: `data/submissions/nir_s2_lbr_impact_abs_wmean_k40_aw0_top5_s0p003_20260607.csv`
- Public score: `13.93887312657075`
- Interpretation: small positive move from the protected `13.94729351406489` anchor. This validates the upward hard-case local-residual direction, but the effect size is incremental.
- Next decision: `high80_rf + lres_pls1` remains a plausible conditional attack candidate, because Slot 1 did not reject the hard-case upward-correction hypothesis.

## Slot 2 Public Result

- Submitted file: `data/submissions/nir_hmoe_high80rf_lrespls1_top6p5_s0p0035_c0p2_20260607.csv`
- Public score: `13.943731590451208`
- Interpretation: worse than Slot 1, but still slightly better than the protected `13.94729351406489` anchor.
- Decision: `high80_rf` as a detector is useful, but the more aggressive high-moisture specialist correction is not better than the simpler `impact_abs + wmean` correction. Do not keep increasing this branch without new evidence.

## Tomorrow Slot 1 Backup

- File: `data/submissions/nir_s2_lbr_impact_abs_wmean_k40_aw0p35_top5_s0p003_20260607.csv`
- Experiment: `nir_s2lbr_impact_abs_top0p05_lres_current_wmean_k40_aw0p35_a0_s0p003_c0p2_soft0`
- Difference from Slot 1: feature distance includes `anchor_pred weight=0.35`.
- Diagnostics: risk `0.136493`, diff RMSE `0.020119`, max diff `0.152976`, OOF delta `-0.005057`, folds `4/5`.
- Use only if there is a specific reason to prefer anchor-prediction-localized distance. Otherwise Slot 1 `aw0` is slightly cleaner and stronger internally.

## Tomorrow Slot 2: Conditional Attack

Use only if Slot 1 improves or is effectively tied.

- File: `data/submissions/nir_hmoe_high80rf_lrespls1_top6p5_s0p0035_c0p2_20260607.csv`
- Experiment: `nir_hmoe_high80_rf_top0p065_lres_pls1_k40_s0p0035_c0p2`
- Method:
  - Anchor fixed.
  - Hard/high gate: `high80_rf`, top 6.5%.
  - Expert signal: local residual PLS1, `k=40`.
  - Correction: calibrated residual correction, `shrink=0.0035`, `clip=0.2`.
  - This is a high-moisture hard-case specialist, not a global replacement.

Diagnostics:

- Rows changed: `36`
- Negative predictions: `0`
- Anchor diff RMSE: `0.023168`
- Max diff: `0.200000`
- Clip saturation: `0.027778`
- Max species mean shift: `0.008687`
- Corrected species max count: `12`
- Top10 correction species max count: `3`
- Signal/residual corr: `0.156641`
- OOF delta: `-0.005719`
- Improved groups: `8/13`
- Hard precision on gate: `0.534884`
- Positive precision on gate: `0.848837`
- High-moisture precision on gate: `0.965116`
- Residual lift on gate: `1.710203`
- Bad-alpha correlation: `0.204283`

Reviewer note:

- Use only after Slot 1 is non-worse or improves.
- If Slot 1 worsens, this is the same upward correction family and should be held.
- This is more attack-oriented than Slot 1, but less risky than the raw-k80 high80_rf branch.

Submission memo:

```text
high80_rf hard/high gate top6.5; local residual PLS1 k40; anchor fixed; shrink0.0035; clip0.2; 36 high-moisture hard-case rows corrected
```

## Deprecated Attack Branches

- `data/submissions/nir_s2_lwpls1_impact_abs_k40_aw0p35_top7p5_s0p003_20260607.csv`
  - Kept for reference only.
  - Strong OOF, but max diff `0.300000` and older gate make it less attractive than the new high80_rf branch.
- Raw-k80 high80_rf branch:
  - Stronger OOF than Slot 2, but higher public-risk score and more species concentration risk.
- Density-ratio weighted experts:
  - Rejected for submission. ESS was too low and target correlation was too high.
- Local YJ high80_rf:
  - Useful diagnostic, but kept below raw/local-residual branches because of extrapolation risk.

## If Slot 1 Worsens

Do not immediately submit the older detector-top5 k120 candidate. It is the same direction but weaker than the impact-gate candidate.

Instead:

1. Pause submission.
2. Review the public delta against current best.
3. Continue internal search around gate quality or a different residual signal.

## Files Created/Updated

- `scripts/nir_stage2_local_residual_model_search.py`
- `data/submissions/nir_s2_lbr_impact_abs_wmean_k40_aw0_top5_s0p003_20260607.csv`
- `data/submissions/nir_s2_lbr_wmean_k120_aw0p35_top5_s0p005_20260607.csv`
- `data/submissions/nir_s2_lbr_impact_abs_wmean_k40_aw0p35_top5_s0p003_20260607.csv`
- `data/submissions/nir_s2_lwpls1_impact_abs_k40_aw0p35_top7p5_s0p003_20260607.csv`
- `scripts/nir_hardcase_moe_search.py`
- `data/submissions/nir_hmoe_high80rf_lrespls1_top6p5_s0p0035_c0p2_20260607.csv`

## 2026-06-08 Post-Public Exploration

Current best before this round:

- Public best: `13.93887312657075`
- File: `data/submissions/nir_s2_lbr_impact_abs_wmean_k40_aw0_top5_s0p003_20260607.csv`

Slot 2 result:

- Public score: `13.943731590451208`
- Interpretation: `high80_rf + lres_pls1` was still better than the older protected anchor, but worse than Slot 1. The detector was not rejected, but the broader/high-moisture PLS correction expanded into too many different rows.

Post-public gate search:

- Script: `scripts/nir_post_public_gate_search.py`
- Output: `outputs/nir_post_public_gate/20260608_085536`
- Result: 864 candidates, all rejected by strict gate.
- Main finding:
  - Pure Slot1 neighborhood is saturated.
  - Stronger Slot1 versions improve OOF but become public-tuning risk.
  - Slot1 x high80 product gates add 8-12 Slot2-only rows and repeat the Slot2 failure pattern.

Broader test-near golden branch search:

- Script: `scripts/nir_slot1_testnear_branch_search.py`
- Output: `outputs/nir_slot1_testnear_branch/20260608_091605`
- Research basis:
  - NIR/soil spectroscopy winners often rely on trusted calibration subsets, test-near/local calibration, SG/SNV preprocessing, and PLS rather than large generic models.
  - Therefore the branch was implemented as a residual signal only; it never replaces the Slot1 anchor directly.

Submitted candidate:

- File: `data/submissions/nir_s1tn_pls_knncluster_k6_q13_c5_clcap3_f4_s004_c018_20260608.csv`
- Public score: `13.934633170499714`
- Interpretation: new best. The gain is small, but it validates the broader `test-near golden PLS residual signal + unsupervised cluster-capped gate` direction over further Slot1/Slot2 local tweaks.
- Source candidate:
  - `outputs/nir_slot1_testnear_branch/20260608_091605/candidates/nir_s1tn_pls_sg9_snv_knn_cluster_k6_q1p3_c5_abs_signal_cluster3_f0p04_s0p04_c0p18_b0p0959437.csv`
- Method:
  - Slot1 anchor fixed.
  - Branch signal: `SG9/SNV + test-near knn_cluster keep60 + residual-quality power 1.3 + PLS5`.
  - Gate: absolute branch-vs-Slot1 signal top 4%.
  - Risk control: unsupervised current-feature cluster cap 3, no species/sample-number direct rule.
  - Correction: `shrink=0.04`, `clip=0.18`.

Diagnostics:

- Rows changed vs Slot1: `22`
- Negative predictions: `0`
- Slot1 diff RMSE: `0.016226`
- Slot1 max diff: `0.153977`
- Slot2-only changed rows: `0`
- Max species mean shift: `0.005531`
- Corrected species max count: `6`
- Top10 correction species max count: `3`
- Clip saturation: `0`
- OOF delta vs Slot1: `-0.007808`
- Improved groups: `8/13`
- Signal/residual correlation on gate: `0.369869`

Submission memo:

```text
Slot1 anchor fixed; test-near golden PLS5 residual signal; abs-signal top4% gate with unsupervised current-feature cluster cap3; shrink0.04 clip0.18; 22 rows corrected
```

## Next Candidate After New Best

After Public `13.934633170499714`, the validated direction was continued as a second-stage residual search.

Plan/review conclusion:

- Do not submit simple stronger variants of the first-stage candidate.
- Reconstruct the current-best OOF anchor.
- Fit another test-near golden PLS residual signal against `y - current_best_oof`.
- Keep output as current-best anchor plus a small gated correction.

Final saved candidate:

- File: `data/submissions/nir_s2tn_curbest_pls_k75_q13_c5_clcap2_f045_s002_c018_20260608.csv`
- Public score: `13.93023150835737`
- Interpretation: new best again. The gain is small, but the second-stage current-best anchored residual correction also worked on Public, validating the iterative `test-near golden PLS residual + unsupervised cluster-capped gate` direction.
- Source candidate:
  - `outputs/nir_slot1_testnear_branch/20260608_102426/candidates/nir_s1tn_pls_sg9_snv_knn_cluster_k75_q1p3_c5_abs_signal_cluster2_f0p045_s0p02_c0p18_b0p184952.csv`
- Method:
  - Current-best anchor fixed.
  - Second-stage branch signal: `SG9/SNV + test-near knn_cluster keep75 + residual-quality power 1.3 + PLS5`.
  - Gate: absolute branch-vs-current-best signal top 4.5%.
  - Risk control: unsupervised current-feature cluster cap 2.
  - Correction: `shrink=0.02`, `clip=0.18`.

Diagnostics vs current best:

- Rows changed: `25`
- Negative predictions: `0`
- Diff RMSE: `0.015216`
- Max diff: `0.117097`
- Slot2-only changed rows: `0`
- Max species mean shift: `0.008588`
- Corrected species max count: `6`
- Top10 correction species max count: `4`
- Clip saturation: `0`
- OOF delta vs current best: `-0.005932`
- Improved groups: `9/13`
- Signal/residual correlation on gate: `0.361225`

Reviewer decision:

- This is the next saved submission candidate.
- Stronger `shrink=0.025` remains an attack backup but is not preferred because species shift and max diff are higher for only a small additional OOF gain.

Submission memo:

```text
current-best anchor fixed; second-stage test-near golden PLS5 residual; k75 q1.3; abs-signal top4.5 cluster cap2; shrink0.02 clip0.18; 25 rows corrected
```

## Third-Stage Candidate

After Public `13.93023150835737`, the same direction was continued once more, but with stricter anti-stacking controls.

Plan/review conclusion:

- Promote the second-stage Public best to the current anchor.
- Reconstruct OOF with both successful corrections:
  - stage 1: `keep60 q1.3 PLS5 cluster cap3`
  - stage 2: `keep75 q1.3 PLS5 cluster cap2`
- Search third-stage residuals with tighter gates:
  - cap 1/2 only
  - low shrink first
  - prefer PLS3/4 over PLS5
  - require overlap/correlation audit against prior corrections

Submitted candidate:

- File: `data/submissions/nir_s3tn_curbest_pls_k6_q16_c4_clcap2_f04_s0012_c018_20260608.csv`
- Public score: `13.926798195477131`
- Interpretation: new best again. The third-stage residual correction also worked on Public, so the iterative current-best anchored `test-near golden PLS residual + cluster-capped gate` direction remains productive. Further improvement is plausible, but the next step should prioritize gate novelty and concentration control over simply adding a fourth similar correction.
- Source candidate:
  - `outputs/nir_slot1_testnear_branch/20260608_114213/candidates/nir_s1tn_pls_sg9_snv_knn_cluster_k6_q1p6_c4_abs_signal_cluster2_f0p04_s0p012_c0p18_b0p364905.csv`
- Method:
  - Current-best anchor fixed.
  - Third-stage branch signal: `SG9/SNV + test-near knn_cluster keep60 + residual-quality power 1.6 + PLS4`.
  - Gate: absolute branch-vs-current-best signal top 4%.
  - Risk control: unsupervised current-feature cluster cap 2.
  - Correction: `shrink=0.012`, `clip=0.18`.

Diagnostics vs current best:

- Rows changed: `22`
- Negative predictions: `0`
- Diff RMSE: `0.013351`
- Max diff: `0.109150`
- Slot2-only changed rows: `0`
- Overlap with stage1+2 corrected rows: `14`
- New rows: `8`
- Correlation with combined stage1+2 correction: `0.598571`
- Max species mean shift: `0.003866`
- Corrected species max count: `6`
- Top10 correction species max count: `3`
- Clip saturation: `0`
- OOF delta vs current best: `-0.004677`
- Improved groups: `10/13`
- Signal/residual correlation on gate: `0.381155`

Submission memo:

```text
current-best anchor fixed; third-stage test-near golden PLS4 residual; keep60 q1.6; abs-signal top4 cluster cap2; shrink0.012 clip0.18; 22 rows corrected
```

## Fourth-Stage Candidate For Next Submission

After Public `13.926798195477131`, additional exploration was run with a stricter reviewer gate:

- Do not use test species cap for gating.
- Reject candidates that only amplify prior Stage1/2/3 corrections.
- Prefer unsupervised diversity constraints: current-feature cluster cap and low prior-correction correlation.
- Keep Stage3 as the fixed anchor and only add a small residual correction.

Explored and rejected:

- Stage4 `cluster_count=32`: no strict pass. Near candidates had good OOF but still concentrated by species (`corrected_species_max=7`, `top10=4`).
- Stage3-anchored local weighted residual: no pass. Main issues were clip saturation, one-sided correction, and species concentration.
- SPXY/Kennard-Stone direct reweighting: too far from Stage3 anchor (`diff RMSE > 0.2`), not suitable as a conservative submission.
- Test species cap: explicitly rejected by reviewer because train/test species are non-overlapping and this would be too close to Public/Private distribution shaping.
- Feature-space spacing gate: implemented as an unsupervised diversity check and tested on the `k65 q1.0 PLS5` branch. It did not produce a better replacement. The only near-pass spacing variants were identical to the saved Stage4 candidate, while spacing-only candidates had weaker OOF/group behavior.

Saved next candidate:

- File: `data/submissions/nir_s4tn_curbest_pls_k65_q1_c5_clcap1_f035_s002_c018_20260608.csv`
- Source candidate:
  - `outputs/nir_slot1_testnear_branch/20260608_221043/candidates/nir_s1tn_pls_sg9_snv_knn_cluster_k65_q1_c5_abs_signal_cluster1_f0p035_s0p02_c0p18_b0p102886.csv`
- Submission status:
  - Not submitted on 2026-06-08 because SIGNATE returned `1日の投稿数を超えました。`
  - This is the first candidate to submit when the daily quota resets.

Method:

- Stage3 current-best anchor fixed.
- Fourth-stage branch signal: `SG9/SNV + test-near knn_cluster keep65 + residual-quality power 1.0 + PLS5`.
- Gate: absolute branch-vs-current-best signal top 3.5%.
- Risk control: unsupervised current-feature `cluster_count=64`, cluster cap 1.
- Correction: `shrink=0.02`, `clip=0.18`.

Diagnostics vs Stage3 current best:

- Rows changed: `19`
- Negative predictions: `0`
- Diff RMSE: `0.008506`
- Max diff: `0.081698`
- New rows vs prior combined correction: `7`
- Correlation with prior combined correction: `0.540290`
- Max species mean shift: `0.004113`
- Corrected species max count: `4`
- Top10 correction species max count: `3`
- Clip saturation: `0`
- OOF delta vs Stage3: `-0.003101`
- Improved groups: `9/13`
- Signal/residual correlation on gate: `0.301130`

Submission memo:

```text
Stage4 current-best anchor fixed; test-near PLS5 residual; k65 q1.0; cluster64 cap1 abs-signal top3.5%; shrink0.02 clip0.18; 19 rows corrected
```

## Tomorrow Five-Candidate Queue

After the Stage4 candidate was saved, a broader same-family search was run for the next daily reset:

- Additional cluster counts: `48` and `80`
- Anchor: fixed to the current Public best Stage3 file
- Family: `test-near golden PLS residual + cluster-capped correction`
- Gate rule: keep `slot2_only_rows` as a diagnostic, not as a hard reject, because the current Stage3 construction no longer uses the old Slot2/HMOE branch.
- Diversity rule: do not keep candidates with both high correction correlation and high changed-row overlap.

Generated manifest:

- `outputs/tomorrow_5queue_manifest_20260608.csv`
- `outputs/tomorrow_5queue_pairwise_20260608.csv`

Submission order:

1. `data/submissions/nir_tomorrow_q1_s4tn_safe_s4_k65_q1_c5_cl1_f0p035_s0p02_c0p18_b0p102886_20260608.csv`
   - Role: safest Stage4 continuation.
   - Diff RMSE/max vs current: `0.008506 / 0.081698`
   - Changed/new rows: `19 / 7`
   - OOF delta: `-0.003101`
   - Submitted on 2026-06-09.
   - Public score: `13.924977185811285`
   - Result: new best. The Stage4 current-best anchored correction remains Public-positive.

2. `data/submissions/nir_tomorrow_q2_s4tn_safe_diverse_k75_k75_q1p6_c5_cl1_f0p04_s0p02_c0p18_b0p167011_20260608.csv`
   - Role: safer diverse branch.
   - Diff RMSE/max: `0.010339 / 0.092673`
   - S4 diff corr / changed overlap: `0.293615 / 0.138889`
   - Changed/new rows: `22 / 8`
   - OOF delta: `-0.004569`
   - Submitted on 2026-06-09.
   - Public score: `13.924920044963732`
   - Result: improved over q1 by a tiny margin, and improved over previous best.

3. `data/submissions/nir_tomorrow_q3_s4tn_safe_diverse_k55_k55_q1_c5_cl1_f0p04_s0p025_c0p18_b0p115147_20260608.csv`
   - Role: second safe diverse branch with more new rows.
   - Diff RMSE/max: `0.011520 / 0.099533`
   - S4 diff corr / changed overlap: `0.253330 / 0.171429`
   - Changed/new rows: `22 / 12`
   - OOF delta: `-0.003793`
   - Submitted on 2026-06-09.
   - Public score: `13.924869001891032`
   - Result: improved over q2 by a tiny margin, and improved over previous best.

4. `data/submissions/nir_tomorrow_q4_s4tn_attack_lite_lowcorr_k55_q1p3_c4_cl1_f0p04_s0p012_c0p18_b0p442354_20260608.csv`
   - Role: attack-lite, lower prior-correction correlation and stronger signal correlation.
   - Diff RMSE/max: `0.014462 / 0.116963`
   - S4 diff corr / changed overlap: `0.284862 / 0.138889`
   - Changed/new rows: `22 / 12`
   - OOF delta: `-0.005133`
   - Use only if q1/q2 indicate the Stage4 direction is still Public-positive.
   - Submitted on 2026-06-09.
   - Public score: `13.924188284374708`
   - Result: improved clearly over q1-q3. The less conservative, lower-correlation branch was rewarded.

5. `data/submissions/nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv`
   - Role: high-OOF attack branch with cluster cap 2.
   - Diff RMSE/max: `0.015773 / 0.134512`
   - S4 diff corr / changed overlap: `0.361036 / 0.205882`
   - Changed/new rows: `22 / 7`
   - OOF delta: `-0.007160`
   - Highest local improvement, but max diff is attack-range. Submit only after positive Public evidence from earlier queue items.
   - Submitted on 2026-06-09.
   - Public score: `13.922966797992677`
   - Result: new best. The most aggressive high-OOF cluster2 branch won.

Pre-submit checks for all five:

- `sample_submit.csv` order: OK
- Shape: `550 x 2`
- Negative predictions: `0`
- Top10 corrected species max: `3`
- Clip saturation: `0`

Decision rule:

- If q1 improves Public: submit q2 next.
- If q1 is flat but not clearly bad: q2 is still acceptable because it corrects a different row set.
- If q1 worsens: stop q4/q5 and only consider q2/q3 after reviewing the changed-row diagnostics.
- If q2 improves: q3 or q4 becomes the next candidate depending on how much improvement remains.
- q5 is not a default fifth submission. It is reserved for when earlier Stage4-family candidates confirm that the hard-case correction direction is still rewarded by Public.

## Public Readout After Five Submissions

All five candidates improved over the previous best `13.926798195477131`.

| Rank | Bucket | Public | Gain vs previous best |
|---:|---|---:|---:|
| q1 | safe_s4 | `13.924977185811285` | `0.001821` |
| q2 | safe_diverse_k75 | `13.924920044963732` | `0.001878` |
| q3 | safe_diverse_k55 | `13.924869001891032` | `0.001929` |
| q4 | attack_lite_lowcorr | `13.924188284374708` | `0.002610` |
| q5 | attack_high_oof_cluster2 | `13.922966797992677` | `0.003831` |

Interpretation:

- The `test-near golden PLS residual + cluster-capped correction` direction is still productive.
- q5 winning means the gate was likely too conservative before. Moderate attack-range corrections can be Public-positive when OOF delta is strong and top10 species concentration is controlled.
- In this five-point result, Public gain aligns best with stronger OOF improvement and larger allowed correction. This is a weak sample statistically, but it is directionally useful.
- `top10_species_max=3`, `clip saturation=0`, and nonzero row diversity remain important guardrails.

Next work should avoid just repeating q5 with slightly different shrink. Use q5 as the new anchor, but explore two axes in parallel:

1. Stage5 continuation:
   - Re-anchor on q5.
   - Search another residual layer with cluster cap `2/3`, max diff up to roughly `0.16`, and top10 species max still fixed at `3`.
   - Prefer branches with OOF delta stronger than `-0.004` and changed-row overlap not dominated by q5.

2. Hard-case detector broadening:
   - Keep the same correction framework, but change how correction targets are detected.
   - Candidate detectors: local PLS residual signal, KNN/local ridge disagreement, distance to test-near golden subset, ensemble uncertainty, and unsupervised outlier score.
   - The goal is to find new hard rows, not only amplify the q5 rows.
