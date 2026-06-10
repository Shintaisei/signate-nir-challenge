# 2026-06-06 Research Hypotheses For NIR Moisture Improvement

## Current State

- Public best: `13.986468676098955`
- Protected anchor: `SG9 -> SNV -> PCA20 -> Yeo-Johnson target -> Ridge3500 + OOF affine`
- Best local candidate: anchor plus small `MSC + SG9 + raw-target PLS4` residual distillation
- Secondary local candidate: test-like golden subset `keep80/c3`

The next goal is not a blend. The goal is to find a new, defensible signal that can improve the protected anchor without relying on Public leaderboard guessing.

## Research Round 1

External references point to five recurring winning or robust categories:

1. `SG/SNV/MSC + compact PLSR`
   - Soil and wood NIR work repeatedly uses scatter correction, Savitzky-Golay smoothing or derivatives, and PLSR.
   - This matches our current best local correction.
2. Calibration-set quality / golden subset
   - SS4GG MIR winner used a golden training subset, Savitzky-Golay smoothing, and PLSR.
   - The official summary says simple models can work well with proper data screening and preprocessing.
3. Local or context-aware calibration
   - SS4GG NIR winners used geocovariates/context and clustered/local modeling.
   - Our competition has no external covariates, so only spectral proximity can be used.
4. Orthogonalization / calibration-transfer style preprocessing
   - EPO/OSC/TOP style methods remove spectral directions dominated by external nuisance variation.
   - In this project, train/test species non-overlap can be treated as a domain-shift nuisance signal.
5. Lightweight nonlinear model on chemometric latent variables
   - PLSELM and TabPFN-style papers support small-data nonlinear calibration after good preprocessing or latent compression.
   - Direct raw black-box modeling is still risky.

## Review Round 1

Rejected or de-prioritized after comparing with local history:

- Broad full-spectrum PLS replacement
  - Already unstable in Public/local history.
  - Keep only as branch residual, not as anchor replacement.
- Species/context routing
  - Train/test species are non-overlapping, so any species-specific target transfer is high Public-overfit risk.
- Hard local kNN model
  - Prior local branch had neighbor signal, but weak OOF and species bias.
- Sample-number monotonic increasing
  - Local diagnostic moved predictions far from anchor and likely has wrong direction.
- Large neural/CNN/Transformer model
  - Literature can justify it when covariates or many samples exist, but this dataset is too small for stable direct training.

## Research Round 2: Implementable Hypotheses

### H1. Operator-Adaptive PLS/Ridge Residual

Idea:

- Instead of choosing one preprocessing manually, create a small bank of linear spectral operators and let compact PLS/Ridge choose useful directions.
- Keep SNV/MSC/EMSC-like fitted corrections as fold-local branches to avoid leakage.
- Use the resulting branch only as a small residual correction to the anchor.

Why it fits:

- Recent operator-adaptive NIR work argues preprocessing selection can be folded into calibration, and reports compact PLS/Ridge variants competitive with expensive preprocessing search.
- Our current best local correction is already a single-operator version: `MSC + SG9 + PLS4`.

Concrete local test:

- Operators: identity, SG window `7/9/11`, first derivative, second derivative, detrend, local mean removal, low/high spectral bands.
- Models: PLS components `2-6`, Ridge on operator PCA `12/16/20`.
- Evaluate each as `branch - anchor`, then distill with small shrink/clip.

Reject if:

- OOF delta does not beat existing PLS residual by at least `0.005`.
- Signal/residual corr `< 0.30` for PLS-like branches.
- Anchor diff RMSE `> 0.12`, max diff `> 0.20`, or species mean shift `> 0.04`.
- The selected operator only works because of one train species or one narrow wavelength artifact.

Priority: high.

### H2. Test-Like Golden Subset v2

Idea:

- Improve the current test-like golden subset, but make it less brittle.
- Select or weight train samples by proximity to test spectra in multiple latent spaces, not just one PCA/kNN score.

Why it fits:

- SS4GG MIR winner demonstrates that a screened calibration subset plus SG+PLSR can beat more complex models.
- Our first test-like golden subset produced usable but not best residual signal, so the concept has not failed.

Concrete local test:

- Representations: `MSC+SG9`, `SG9+SNV`, `SG9+SNV+d1`, and raw absorbance slope.
- Selection scores: kNN-to-test, cluster co-membership, density ratio, and leverage.
- Combine by rank aggregation, not one hard rule.
- Try soft weights first; hard keep only `70/80/90%`.
- Branch: raw-target PLS `3/4/5` and YJ Ridge only if beta/corr is positive.

Reject if:

- Any fold-local selection uses validation labels or global train+valid transforms.
- Selected set has a species share concentration above prior gates.
- Ridge/YJ branch has negative beta or negative residual corr.
- OOF gain is smaller than existing PLS residual while anchor drift is larger.

Priority: high.

### H3. EPO/OSC-Style Domain-Shift Orthogonalization

Idea:

- Estimate spectral directions that separate train and test, or high-shift train species from test-like train species.
- Remove only the top nuisance component(s), then refit compact PLS/Ridge.

Why it fits:

- EPO removes a parasitic spectral subspace caused by an external parameter; train/test species shift can be treated as a nuisance variation if estimated without test labels.
- This is a root preprocessing idea, not a leaderboard blend.

Concrete local test:

- Build train/test domain classifier or PCA on mean differences in preprocessed spectra.
- Remove `1-3` nuisance directions from `X`.
- Fit `PLS3/4/5` raw target and `YJ Ridge PCA20`.
- Distill only if OOF branch signal is positive.

Reject if:

- Removing a component collapses prediction range.
- Branch OOF worsens by more than `0.3` RMSE vs comparable branch.
- Anchor diff RMSE exceeds `0.12` after small distillation.
- Nuisance direction is strongly correlated with target in train; that would remove moisture signal.

Priority: medium-high.

### H4. Latent PLSELM / Random Feature Calibration

Idea:

- Use PLS scores as low-dimensional features, then fit a lightweight nonlinear random-feature model.
- This approximates PLSELM without requiring a large deep model.

Why it fits:

- PLSELM literature specifically targets low-data NIR calibration and combines PLS score matrices with ensemble extreme learning machines.
- Our prior SVR branch found a small nonlinear signal, but not enough.

Concrete local test:

- Preprocess: `MSC+SG9`, `SG9+SNV`.
- PLS scores: components `3-8`.
- Model: fixed random tanh/ReLU features + Ridge, 20 seeds, hidden units `16/32/64`.
- Average only within the model family to reduce random-feature noise, then distill as branch residual.

Reject if:

- Seed variance of test corrections is high: candidate-to-candidate RMSE `> 0.05`.
- OOF gain comes with negative beta or low residual corr.
- Corrections are nearly identical to SVR/PLS residual but weaker.
- Any direct model prediction moves far from anchor.

Priority: medium.

### H5. Uncertainty / Applicability-Domain Gating

Idea:

- Do not change all test samples equally.
- Apply residual correction only where the branch is inside its applicability domain, based on train/test spectral proximity and ensemble agreement.

Why it fits:

- Calibration transfer and small-data chemometrics emphasize robustness and representative calibration sets.
- Our PLS correction may be strongest for test-like samples and risky for out-of-domain samples.

Concrete local test:

- Compute per-test reliability from:
  - distance to selected golden train subset
  - branch ensemble variance
  - leverage in PLS/PCA space
- Gate correction by reliability: `correction *= reliability`.
- Compare to current uniform shrink/clip distillation.

Reject if:

- Gate mainly encodes test species blocks and creates species mean shift.
- Gate removes almost all correction and becomes a no-op.
- Gate increases max correction or creates local discontinuities.

Priority: medium.

## Review Round 2: Priority

Tomorrow's local experiment priority:

1. `H1 operator-adaptive residual`
   - Most directly connected to current winning local signal.
   - Highest chance to improve without changing the anchor philosophy.
2. `H2 test-like golden subset v2`
   - Still the best root hypothesis from similar competitions.
   - Needs better rank aggregation and soft weighting.
3. `H3 EPO/OSC domain orthogonalization`
   - More novel and root-level.
   - Risk is accidentally removing moisture signal.
4. `H5 applicability-domain gating`
   - Useful as safety layer after H1/H2/H3 branch signals exist.
5. `H4 latent PLSELM/random-feature branch`
   - Worth trying if time allows, but treat as residual only.

## Submission Policy From This Research

Do not submit direct replacements tomorrow unless they clearly dominate. Submission candidates should be:

```text
protected anchor + one reviewed, small, independently justified correction
```

Minimum submit gate:

- CSV order and shape valid.
- Negative predictions: `0`.
- Anchor diff RMSE: `0.03-0.12`.
- Max diff: `<0.20`, unless explicitly reviewed.
- Max test species mean shift: `<0.04`.
- OOF delta better than existing anchor.
- If competing against current local primary, either:
  - OOF delta improves by `>= 0.005`, or
  - signal is clearly independent and safer than current PLS residual.

## Sources Checked

- SS4GG community competition results: NIR winners used context/local modeling; MIR winner used golden subset + Savitzky-Golay + PLSR.
- EPO-PLS: external parameter orthogonalization removes nuisance spectral subspace and improved robustness under external variation.
- Operator-adaptive PLS/Ridge: recent NIR benchmark folds linear preprocessing operator selection into compact calibration.
- PLSELM: low-data NIR calibration method using PLS score matrices plus ensemble extreme learning machine.
- TabPFN NIR calibration benchmark: preprocessing-optimized TabPFN can complement chemometric workflows, but outliers/extrapolation still favor classical chemometrics.

## Final Review Round: Narrowed Implementation Hypotheses

Reviewer: Planck (`019e9d19-69db-7ca3-af96-69a7bd1fdb91`)

## Research Round 3: Outside Classic Chemometrics

The previous plan was too close to textbook NIR chemometrics. A broader search across GCMS competitions, sensor-drift papers, tabular distribution-shift practice, and spectroscopy GitHub projects suggests several non-textbook ideas that are still implementable in this small NIR project.

### O1. Multi-View Preprocessing As Channels, Then Compress

External pattern:

- Mars Spectrometry 2 GCMS winning solutions did not rely on one canonical preprocessing. The 1st and 2nd place summaries used ensembles over multiple preprocessed 1D/2D representations; 3rd place combined CNN models with logistic/ridge/statistical features.
- The transferable idea is not a CNN. It is treating preprocessing variants as separate measurement views and letting a compact model decide which cross-view signal survives.

Project translation:

- Build a small multi-view feature table from the same 20 wavelengths:
  - raw, SNV, MSC, SG9, first difference, second difference, detrend
  - per-view PCA/PLS scores, not all raw expanded columns
  - cross-view consistency features: mean, std, slope agreement, high-low contrast agreement
- Fit Ridge/ElasticNet/Huber on this compressed multi-view table, then distill only if it produces a positive residual branch.

Review:

- This is the best non-textbook extension of the current operator-bank residual.
- It avoids pretending one preprocessing is correct.
- Reject if the feature expansion makes OOF look good but branch residual corr is weak, or if test correction is dominated by one synthetic view.

Priority: high.

### O2. Adversarial-Validation Weighting

External pattern:

- Tabular competition practice often trains a classifier to separate train from test, then uses the result to diagnose or weight examples under distribution shift.
- A credit-scoring dataset-shift paper also describes selecting training samples closer to predicted/test data via adversarial validation.

Project translation:

- Fit a fold-local train-vs-test classifier in spectral latent space.
- Use its output only as a calibration weight/reliability score:
  - high test-likeness train samples get more weight
  - very non-test-like train samples are downweighted, not removed
  - final test correction is gated by how confidently the branch sits inside that applicability domain
- Test on PLS/Ridge residual branches, not direct replacement.

Review:

- This is a stronger, more general version of test-like golden subset v2.
- Main risk is Public overfit if the classifier learns sample-order/species artifacts. Fold-local OOF and species-shift gates are mandatory.

Priority: high.

### O3. Drift-Ensemble Residuals

External pattern:

- Gas sensor drift literature uses weighted ensembles of models trained under different time/drift conditions; component correction and ensembles are complementary.
- This maps better to our train/test species shift than classic PLS alone.

Project translation:

- Treat train species clusters or spectral clusters as pseudo-domains.
- Train several small calibrators, each leaving out or emphasizing a pseudo-domain.
- At test time, weight branch residuals by similarity between the test sample and each pseudo-domain.

Review:

- This is promising only if it stays soft. Hard routing already looked weak in earlier local experiments.
- Reject if one species/cluster receives a large unique offset or if per-domain residuals disagree strongly.

Priority: medium-high.

### O4. Peak/Event-Like Features

External pattern:

- GCMS winners engineered features around peaks/events, and some models used event-detection style networks.
- With only 20 wavelengths, we cannot do real peak detection, but we can create event-like descriptors.

Project translation:

- Add features such as:
  - local maxima/minima indicators after SG
  - largest adjacent drop/rise
  - curvature sign pattern
  - ratios and differences around known water-sensitive bands if identifiable
  - rank/order pattern of the 20 wavelength intensities
- Fit compact Ridge/PLS/ElasticNet branch.

Review:

- Useful as independent signal search, but likely weaker than multi-view/adversarial weighting.
- Keep as branch residual only.

Priority: medium.

### O5. Frequency-Basis Compression

External pattern:

- Some chemical-sensor drift work uses transforms such as DCT with neural/convolutional models to handle smooth drift.
- For 20-dimensional spectra, DCT/wavelet-like bases are cheap and less overfit-prone than neural models.

Project translation:

- Build DCT coefficients of raw/SNV/MSC spectra.
- Combine low-frequency coefficients with slope/curvature coefficients.
- Try Ridge/Huber/YJ Ridge and residual distillation.

Review:

- This is easy to implement and may capture baseline/shape separately.
- It is lower priority because PCA already captures smooth variation, but DCT coefficients are more interpretable and may regularize better.

Priority: medium-low.

### O6. Leakage-Like Structure Audit, Not Leakage Exploitation

External pattern:

- Some tabular Kaggle competitions were won or distorted by hidden leakage or row-order structure. This is dangerous but the audit mindset is useful.

Project translation:

- Audit whether `sample_number`, row order, duplicate/near-duplicate spectra, or train/test batching creates a stable structure.
- Do not submit hard row-order sorting. Use findings only to design validation folds or tiny reliability gates.

Review:

- We already tested sample-number monotonicity. The user hypothesis direction moved predictions far from anchor, so it is rejected for submission.
- Continue audit for validation design only.

Priority: diagnostic only.

## Review Round 3: Updated Priority

1. `O1 multi-view compressed residual`
   - Broadens operator-bank beyond textbook preprocessing.
   - Directly implementable with current scripts.
2. `O2 adversarial-validation weighting`
   - Best way to operationalize train/test shift without labels.
   - Merge with test-like golden v2 rather than treat as separate.
3. `O3 drift-ensemble residuals`
   - Soft pseudo-domain ensemble is worth trying; hard routing remains rejected.
4. `O4 peak/event-like features`
   - Good independent feature family, but likely smaller signal.
5. `O5 DCT/frequency compression`
   - Cheap exploratory branch.
6. `O6 leakage-like structure audit`
   - Validation/diagnostic only, not a submission mechanism.

Submission discipline:

- Do not submit any direct model replacement from this round.
- Every new idea must first produce an OOF residual branch and pass the existing anchor-drift gates.
- The two highest-value implementation targets are:
  - multi-view compressed residual
  - adversarial-validation weighted golden residual

Additional sources checked:

- DrivenData Mars Spectrometry and Mars Spectrometry 2 GCMS competition pages and winner repositories.
- Gas sensor drift compensation using classifier ensembles.
- Gas sensor calibration with minimal experiments / active sampling.
- DCT-based drift compensation in chemical sensors.
- Tabular adversarial-validation / distribution-shift references and Kaggle solution collections.

Final shortlist:

1. `Operator-bank residual + applicability gating`
   - Highest priority.
   - This is the most natural extension of the current local winner, `MSC+SG9+PLS4 residual`.
   - Test a compact operator bank:
     - `SG7/9/11`
     - first/second derivative
     - detrend
     - local contrast
     - bin contrast
   - Model each branch with `PLS2-6` or `Ridge PCA12-20`.
   - Distill only `branch - anchor`.
   - Add reliability gating so that correction is stronger only for test samples inside the branch applicability domain.
   - Reject if:
     - OOF delta does not beat the existing PLS residual by at least `0.005`
     - signal/residual corr `< 0.30`
     - anchor diff RMSE `> 0.12`
     - max diff `> 0.20`
     - species mean shift `> 0.04`
     - gating becomes a no-op or only moves one species block.

2. `Test-like golden subset v2`
   - Second priority.
   - The SS4GG MIR winner makes this the best root hypothesis from similar competitions.
   - The first simple kNN keep version was useful but not best, so v2 should use rank aggregation:
     - test-neighbor distance
     - cluster overlap
     - leverage
     - OOF residual stability
     - mild downweighting of target extremes
   - Use soft weights before hard trimming.
   - Main branch: `MSC+SG9 PLS3/4/5`.
   - Reject if:
     - selection is not fold-local
     - selected/weighted train samples over-concentrate in one species
     - YJ/Ridge branch has negative beta or negative corr
     - OOF is weaker than existing PLS residual while drift is larger
     - weight/keep rules remove only one side of the target distribution.

3. `EPO/TOP-lite domain-shift orthogonalization`
   - Third priority, but the most root-level new preprocessing idea.
   - Treat train/test species non-overlap as a nuisance spectral shift.
   - Estimate 1-2 train/test domain directions using:
     - PCA mean difference
     - logistic domain classifier coefficient
     - PLS/domain direction
   - Remove only those directions, then fit `PLS3/4/5` or `YJ Ridge PCA20`.
   - Use residual distillation only.
   - Reject if:
     - removed direction is highly correlated with `y` in train
     - prediction range collapses
     - branch OOF is worse than comparable branch by more than `0.3`
     - correction concentrates in one species block.

4. `Small contrast / bin feature residual`
   - Fourth priority.
   - This is the practical low-dimensional version of CARS/SPA/iPLS for a 20-feature spectrum.
   - Build:
     - adjacent differences
     - adjacent ratios
     - local bin mean differences
     - front/back spectral contrasts
   - Feed into `PCA/Ridge` or `PLS2-5`.
   - Reject if:
     - it collapses back to a single-index/band616-style rule
     - species shift is large
     - anchor diff RMSE `> 0.10` with weak OOF improvement.

Defer:

- `PLSELM`, `TabPFN`, compact CNN, and other neural/foundation-model style methods.
- They are interesting diagnostics, but local SVR was already weak, and this dataset is too small and sparse for them to be the next submission track.

## Source Links

- SS4GG competition results: https://soilspectroscopy.org/community-data-science-competition-results/
- Representative calibration sample selection with pretreatments: https://www.mdpi.com/2072-4292/11/4/450
- Operator-adaptive PLS/Ridge: https://arxiv.org/abs/2605.13587
- OSC calibration transfer in NIR: https://pubs.acs.org/doi/10.1021/ac035382g
- OSC concept for NIR calibration transfer: https://www.sciencedirect.com/science/article/pii/S0169743998001129
- Africa Soil Kaggle notes: https://blog.booleanbiotech.com/kaggle_africa_soil_prediction
- Variable selection / CARS-SPA overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC8201019/
- TabPFN NIR calibration benchmark: https://arxiv.org/abs/2605.21544

## Broad Search Round: Outside Textbook Chemometrics

The previous shortlist was useful but too close to standard chemometrics. A broader search checked:

- mass spectrometry / gas chromatography competitions
- Raman spectroscopy transfer-learning writeups
- gas sensor drift compensation
- dataset-shift / adversarial-validation ideas from Kaggle-style workflows
- spectral augmentation / alignment / baseline correction outside NIR

### New Hypothesis A. Multi-View Compressed Residual

Idea:

- Do not choose one preprocessing operator.
- Build several compact views of the same 20-point spectrum:
  - raw
  - SNV
  - MSC
  - SG-smoothed
  - first/second differences
  - detrended residual
  - DCT low/high-frequency components
  - local/bin contrasts
- Compress each view separately, then search for branch residual signal with compact `PLS`, `Ridge`, or `Huber`.
- Distill only the branch residual into the protected anchor.

Why this is less textbook:

- Mars GCMS and spectrometry competition solutions often convert raw instrument signals into alternative representations instead of relying on one canonical preprocessing path.
- For this project, a 2D image CNN is not appropriate because the spectrum has only about 20 effective features, but the representation-diversity idea is portable.

Implementation unit:

- Create a feature bank, then run:
  - per-view PLS
  - concatenated low-dimensional view PCA + Ridge
  - Huber/Ridge on view summaries
- Compare correction correlation with the current `MSC+SG9 PLS4` residual.

Reject if:

- It is just a noisier version of current PLS residual.
- Anchor diff RMSE `> 0.12`, max diff `> 0.20`, species shift `> 0.04`.
- It depends on a single raw index or recreates the failed `band616` style.

Priority: very high.

### New Hypothesis B. Adversarial-Validation Weighted Golden

Idea:

- Train a classifier to distinguish train spectra from test spectra.
- Use the classifier score as a soft weight:
  - train samples that look test-like get higher weight
  - out-of-domain train samples get lower weight
- This is a stronger version of `test-like golden subset v2`.

Why this is less textbook:

- This comes from dataset-shift/Kaggle practice rather than chemometric calibration-set selection.
- Gas sensor drift work also uses domain adaptation and weighted ensembles to handle sensor/domain drift.

Implementation unit:

- Fold-local adversarial validation:
  - in OOF, valid fold acts as pseudo-test
  - final fit may use unlabeled test spectra only for unsupervised/domain weighting
- Classifiers:
  - logistic regression
  - shallow random forest or calibrated linear SVM only if stable
- Convert domain score to bounded sample weights, not hard keep/drop.
- Fit weighted `MSC+SG9 PLS3/4/5` and `SG9+SNV PLS3/4/5`.

Reject if:

- Domain AUC is too low: no useful shift signal.
- Domain AUC is too high and weights collapse onto very few species.
- Weighted set target distribution loses high/low moisture coverage.
- OOF improves only by exploiting validation labels or non-fold-local transforms.

Priority: very high.

### New Hypothesis C. Drift-Ensemble Residual

Idea:

- Treat train species or train spectral clusters as pseudo-domains.
- Train one small branch model per pseudo-domain.
- Weight their residual corrections for each test sample by spectral/domain proximity.

Why this is less textbook:

- Gas sensor drift compensation uses weighted classifier ensembles trained under different drift states.
- This maps naturally to `species/cluster as source domains`, but avoids hard routing.

Implementation unit:

- Build `K=4-8` train spectral clusters in fold-local latent space.
- Fit small PLS/Ridge branch for each cluster or cluster-neighborhood.
- Compute soft test weights by distance to cluster centroids.
- Combine branch corrections, then distill into anchor.

Reject if:

- It becomes hard routing.
- One cluster dominates predictions.
- Species shift exceeds `0.04`.
- OOF gain is weaker than existing PLS residual with higher drift.

Priority: medium-high.

### New Hypothesis D. Peak/Event-Like Spectral Features

Idea:

- Borrow from GCMS/Raman peak/event features, but shrink to 20 wavelengths.
- Features:
  - max adjacent difference
  - max curvature
  - rank pattern
  - local high-low contrasts
  - area-like sums over small bins

Why this is less textbook:

- This is feature engineering from instrument-signal competitions rather than PLS-first calibration.

Implementation unit:

- Use these features as an auxiliary branch, not as direct replacement.
- Fit `Ridge`, `Huber`, or `PLS2-5`.

Reject if:

- It turns into one-index leaderboard chasing.
- It increases test species shift.
- Signal/residual corr is low or negative.

Priority: medium.

### New Hypothesis E. Noise / Shift Robust Residual

Idea:

- Train branch models on small realistic perturbations:
  - additive offset
  - multiplicative scale
  - local wavelength jitter approximation
  - small Gaussian noise
- Average or fit a model robust to these perturbations.

Why this is less textbook:

- Spectroscopy augmentation papers and Raman competition writeups rely on augmentation to avoid overfitting in low-data spectral tasks.

Implementation unit:

- Apply perturbations only inside training folds.
- Keep labels unchanged.
- Use augmented branch to produce OOF/test predictions, then distill.

Reject if:

- OOF improves but test correction is almost identical to current PLS residual.
- Seed variance is high.
- Perturbation scale becomes a hidden hyperparameter overfit.

Priority: medium.

### Broad-Search Priority Update

Replace the older implementation priority with:

1. `Multi-view compressed residual`
2. `Adversarial-validation weighted golden`
3. `Operator-bank residual + gating`
4. `EPO/TOP-lite domain orthogonalization`
5. `Drift-ensemble residual`
6. `Peak/event-like features`
7. `Noise/shift robust residual`

Defer:

- full CNN/image-style models
- TabPFN/direct foundation model prediction
- large neural transfer learning
- hard cluster routing

These are either too large for the 20-feature setting or too risky under train/test species non-overlap.

Additional broad-search sources:

- Mars Spectrometry 2 GCMS competition: https://www.drivendata.org/competitions/97/nasa-mars-gcms/
- Mars GCMS benchmark and feature/image representation: https://blog.drivendata.org/blog/mars-spectrometry-gcms-benchmark
- Mars Spectrometry 2 second-place report: https://arxiv.org/abs/2403.15990
- Raman DIG-4-BIO transfer-learning writeup: https://aterrail.com/blog/post_2025_08_1_dig4bio/
- Gas sensor drift classifier ensembles: https://www.sciencedirect.com/science/article/pii/S0925400512002018
- Gas sensor drift domain adaptation examples: https://pmc.ncbi.nlm.nih.gov/articles/PMC4481969/
- Spectroscopy augmentation by noise addition: https://www.sciencedirect.com/science/article/pii/S000326700401428X
- Spectral alignment / baseline correction overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC3085421/
