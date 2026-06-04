# Segment Feature Search 2026-06-02

## Goal

Explore the user's hypothesis that local wavelength cutouts and shape-derived
features can route samples better than whole-spectrum clustering. The target was
to move `group_species` RMSE toward 10 without using train/test species leakage.

## Implemented Search

Script:

```bash
python scripts/segment_feature_search.py --preset quick
python scripts/segment_feature_search.py --preset focused --only <recipe>
```

Outputs:

```text
outputs/segment_feature_search.csv
outputs/segment_feature_anchor_search.csv
outputs/segment_feature_focused3_*.csv
outputs/segment_feature_*_oof/
```

Feature families:

- window stats: `mean`, `std`, `range`, `slope`, `edge`
- adjacent-window contrast ratios
- anchor-relative cutout features: `segment_mean - anchor`, `segment_mean / anchor`, symmetric ratio
- local anchor shape: mean, std, range, slope, curvature, left-right difference, peak-baseline, area
- preprocessors: `raw`, `smooth5`, `snv`, `diff1`

Models:

- global Ridge / PLS / Huber
- KMeans routing on the segment features, then cluster-wise Ridge1000
- transductive KMeans routing, using train+validation/test `X` only for cluster boundaries and labels only for regression

## Main Results

Whole-spectrum sample clustering was weak:

| approach | group_species RMSE |
| --- | ---: |
| sample waveform clustering | 30.85 |
| KNN waveform baseline | 29.04 |

Local segment/cutout features are clearly better:

| candidate | group_species RMSE | worst group RMSE | worst group bias |
| --- | ---: | ---: | ---: |
| `smooth5_w20_s10_multi_anchor_k80__transductive_cluster_k5` | 21.0219 | 58.3785 | -46.3371 |
| `smooth5_w20_s10_multi_anchor_k80__cluster_k5` | 21.7973 | 59.6965 | -47.3337 |
| `snv_w20_s10_anchor_ratio_k160__cluster_k3` | 21.9523 | 58.1534 | -37.8627 |
| `diff1_w20_s10_multi_anchor_k80__transductive_cluster_k5` | 22.1812 | 58.8193 | not recorded in summary |

Current best local segment route:

```text
features: smooth5 spectrum
window: 20 wavelengths
step: 10
selected features: top 80 by train-fold correlation, plus anchor-neighborhood keep-ins
anchors: index 616, 1332, 1485
routing: transductive KMeans k=5 on segment features
model: cluster-wise StandardScaler + Ridge(alpha=1000)
group_species RMSE: 21.021921989753857
```

Best model group breakdown:

| group | n | RMSE | bias |
| --- | ---: | ---: | ---: |
| ベイスギ | 112 | 58.3785 | -46.3371 |
| チェリー | 88 | 28.7126 | 18.3015 |
| ホワイトオーク | 51 | 19.6126 | -1.3217 |
| ヒノキ | 141 | 14.7343 | 9.7188 |
| クリ | 91 | 10.9589 | -7.3635 |
| ウエンジ | 87 | 10.1663 | 7.5037 |
| イチョウ | 94 | 10.0252 | -1.0217 |
| ベイマツ | 93 | 9.9356 | 4.1506 |
| 米ヒバ | 68 | 9.6020 | -6.8755 |
| スプルース | 97 | 8.6586 | 4.7464 |
| トチ | 183 | 8.3766 | 2.9183 |
| ウォールナット | 110 | 8.2624 | -0.3344 |
| ナラ | 107 | 8.2602 | -1.5288 |

## Interpretation

The hypothesis is partially validated. Local cutout/shape features route the
problem much better than whole-waveform clustering, and most held-out species
are already near RMSE 8 to 11.

The blocker is not general feature weakness. It is level calibration for a few
held-out species, especially ベイスギ and チェリー. The best new route improves
average RMSE from 21.95 to 21.02, but the worst-group RMSE remains around 58.
That prevents any realistic RMSE10 claim.

Adding too many arbitrary anchors made results worse. Restricting anchors to
high-correlation areas worked better:

- raw/smooth anchor: index 616
- diff1 anchor: index 1332
- snv anchor: index 1485

The best current recipe uses `smooth5`, not `snv` or `diff1`, which suggests the
useful signal is still close to stable absorbance-level shape rather than
aggressive normalization.

## Next Search Axis

Use `smooth5_w20_s10_multi_anchor_k80__transductive_cluster_k5` as the local
segment baseline. The next improvement should target worst-group bias directly:

- route by features that separate ベイスギ-like and チェリー-like bias modes
- add a fold-safe bias predictor using segment descriptors, not species labels
- select by `guard_score`, not only average RMSE
- require progress on both `group_species_rmse < 20` and `worst_group_rmse < 45`

Current conclusion: the wavelength-cutout grouping line is worth continuing,
but it has not solved the RMSE10 objective yet.
