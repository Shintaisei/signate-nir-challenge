# Cluster Quality Search 2026-06-02

## Goal

Find a preprocessing and clustering view that cleanly groups test species before
deciding which predictor or correction to route into each group.

## Script

```bash
python scripts/cluster_quality_search.py --preset quick --out-dir outputs/cluster_quality
```

Outputs:

- `outputs/cluster_quality/cluster_quality_summary.csv`
- `outputs/cluster_quality/cluster_quality_details.csv`
- `outputs/cluster_quality/test_species_cluster_map.csv`
- `outputs/cluster_quality/top.json`

## Scoring

`clean_score` combines:

- test cluster purity
- test species concentration
- whether each test cluster also contains enough train samples
- silhouette score

This is not a prediction metric. It is for finding routing structure.

## Best View

Best candidate:

```text
view: diff2_pca5
clusterer: kmeans_k8
clean_score: 0.9663
test_cluster_purity: 0.9036
test_species_concentration: 1.0000
mixed_test_coverage: 1.0000
silhouette: 0.5409
```

Interpretation: second derivative features compressed to PCA5 produce very
clean test grouping. Every test species is fully concentrated into one cluster,
but ケヤキ and ヤマザクラ share a cluster.

## Best Cluster Map

Using `diff2_pca5 + KMeans(k=8)`:

| test species | share in top cluster | top train species in same cluster |
| --- | ---: | --- |
| クスノキ | 1.000 | スプルース |
| ケヤキ | 1.000 | クリ, ベイマツ, イチョウ |
| スギ | 1.000 | イチョウ |
| タモ | 1.000 | ウエンジ |
| チーク | 1.000 | トチ, ウォールナット, ナラ, チェリー |
| ヤマザクラ | 1.000 | クリ, ベイマツ, イチョウ |

The natural routing structure is effectively five groups:

```text
クスノキ
ケヤキ + ヤマザクラ
スギ
タモ
チーク
```

## Practical Use

This clustering is cleaner than the previous raw/smooth/SNV views and is a good
candidate for routing. It should not directly imply target-level correction,
because earlier pseudo-test EDA showed nearest-shape species can still have a
large target mean gap.

Recommended next use:

- Use `diff2_pca5 + kmeans_k8` only as a routing key.
- Fit or select predictors by route.
- Treat ケヤキ and ヤマザクラ as one difficult shared route unless a later view
  separates them cleanly.
- For チーク, do not rely only on ベイスギ similarity; in this cleaner diff2 view,
  チーク aligns more with トチ/ウォールナット/ナラ/チェリー.
