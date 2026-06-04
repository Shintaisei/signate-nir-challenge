# Train/Test Distribution EDA 2026-06-02

## Purpose

Identify which train species distributions are closest to each test species
under multiple spectral views. This is EDA only: test labels are not used, and
train targets are used only to audit whether distribution matching is reliable.

## Script

```bash
python scripts/train_test_distribution_eda.py --out-dir outputs/distribution_eda_readable
```

Main outputs:

- `outputs/distribution_eda_readable/test_species_match_consensus.csv`
- `outputs/distribution_eda_readable/nearest_train_species_by_test_species.csv`
- `outputs/distribution_eda_readable/train_test_species_distribution_pairs.csv`
- `outputs/distribution_eda_readable/mixed_cluster_composition.csv`
- `outputs/distribution_eda_readable/pseudo_test_alignment_score.csv`

The CSV files are UTF-8. PowerShell `Get-Content` may display Japanese labels
as mojibake, but Python/Excel with UTF-8 reads them correctly.

## Views Tested

- `raw_pca20`
- `smooth5_pca20`
- `snv_pca20`
- `center_pca20`
- `diff1_pca20`
- `diff2_pca20`
- `smooth5_segment_shape_pca20`
- `snv_segment_shape_pca20`
- `diff1_segment_shape_pca20`

For each view, the script computes:

- test species -> nearest train species by centroid cosine distance
- test species -> train species distribution distance using an energy-distance style score
- mixed train+test KMeans cluster composition
- leave-one-train-species pseudo-test alignment

## Test Species Consensus

Votes count how often a train species appears in the top-3 distribution matches
across the 9 views.

| test species | strongest train distribution matches |
| --- | --- |
| クスノキ | クリ 9, 米ヒバ 6, チェリー 6, スプルース 4 |
| ケヤキ | チェリー 8, 米ヒバ 5, クリ 5, スプルース 3 |
| スギ | スプルース 8, ヒノキ 5, イチョウ 3, クリ 3, ベイスギ 2 |
| タモ | スプルース 7, クリ 7, 米ヒバ 6, イチョウ 3 |
| チーク | ヒノキ 6, チェリー 5, ベイスギ 4, 米ヒバ 3 |
| ヤマザクラ | 米ヒバ 7, クリ 7, チェリー 6, イチョウ 3 |

This should be treated as a soft distribution map, not a one-to-one species
transfer rule.

## Pseudo-Test Reliability

Each train species was held out as pseudo-test. Matching was done with `X` only,
then the target mean of the matched train species was compared against the
held-out species target mean.

| view | mean abs target mean gap | median | max |
| --- | ---: | ---: | ---: |
| `snv_segment_shape_pca20` | 12.9214 | 10.1710 | 41.7578 |
| `snv_pca20` | 15.2645 | 11.5905 | 41.7578 |
| `diff2_pca20` | 16.2530 | 10.1710 | 57.7034 |
| `center_pca20` | 18.7148 | 10.2126 | 62.5295 |
| `diff1_pca20` | 18.8773 | 18.2210 | 41.7578 |
| `diff1_segment_shape_pca20` | 19.1894 | 13.7712 | 62.5295 |
| `smooth5_segment_shape_pca20` | 22.6276 | 19.9677 | 48.7073 |
| `raw_pca20` | 22.8251 | 20.7716 | 48.7073 |
| `smooth5_pca20` | 22.8251 | 20.7716 | 48.7073 |

The best matching view for target-level reliability is
`snv_segment_shape_pca20`. Raw/smooth PCA gives visually close species, but the
target mean gap is much worse.

## Interpretation

The EDA supports using spectral-shape distribution matching, especially
SNV-based segment shape, for routing or diagnostics. It does not support direct
target mean transfer from the nearest train species. Even the best view has
large worst-case pseudo-test gaps.

The next modeling use should be conservative:

- Use the consensus map to choose candidate routing features.
- Use pseudo-test gap as a reliability penalty.
- Do not copy train species bias directly to test species.
- Prefer weak/shrunk corrections only when multiple views agree.
