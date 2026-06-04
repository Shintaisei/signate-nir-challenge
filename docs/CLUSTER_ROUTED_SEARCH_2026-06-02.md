# Cluster Routed Search Notes 2026-06-02

## Goal

Find whether waveform-cluster routing plus richer wavelength features can push local
`group_species` RMSE toward 10 without leakage.

## Implementation

- Added `scripts/cluster_routed_search.py`.
- Outer validation uses species-group folds.
- For each outer fold, clustering and per-cluster feature/model choice are fit only on the outer train rows.
- Validation rows are assigned by the fold-local clusterer, then predicted by the selected route for that cluster.
- No submission file is generated.

## Results

The first routed run was:

```text
python scripts/cluster_routed_search.py --clusters 3 --cluster-preps raw --inner-splits 2 --max-routes 1 --preset quick
```

Result:

```text
cluster_raw_k3 group_species_rmse = 36.951986952480084
worst_group_rmse = 69.95858235023334
fallback_count = 0
```

This is worse than the existing non-routed baselines. The route selector often chose
SNV/window or SNV hybrid features inside clusters, but the outer fold error remained
large.

Representative fixed-rule checks:

```text
global snv top320 ridge1000 group_species_rmse = 26.798887030676152
cluster snv k3 + snv top320 ridge1000 group_species_rmse = 33.385393391640356
cluster snv k5 + snv top320 ridge1000 group_species_rmse = 28.769138368516078
```

So the waveform clustering did not improve the strongest tested top-correlation
direction. It made the route less stable under held-out species.

## Feature Expansion Findings

Ridge on correlation-selected features did not approach RMSE 10.

Best representative results:

```text
snv top320 ridge1000: 26.798887030676152
snv top80 ridge1000: 29.492240165775275
snv top20 ridge1000: 29.676623455475976
snv top5 ridge1000: 30.396623811856234
raw/smooth5 band616 radius 12-20 ridge1000: about 32.15
smooth5 band616 radius5 PLS2: 33.82080643061923 in this fold script
```

This supports the current interpretation: adding more correlated wavelengths helps
some compared with narrow ridge/band baselines, but it mainly learns species/scatter
structure and still fails held-out species generalization.

## Dataset Constraint

Train and test species are disjoint:

```text
train species: 1,3,4,5,8,11,12,13,14,15,16,17,19
test species: 2,6,7,9,10,18
```

This makes species-specific local models impossible to transfer directly. A route
must generalize to unseen species purely from spectral shape.

## Conclusion

RMSE 10 on `group_species` does not look reachable with the tried family:
top-correlation expansion, band expansion, and wave-cluster routing. The best tested
expanded-feature route is still around 26.8, and clustering worsens it.

The next promising local-only direction is not more routing, but a stricter target:
identify which held-out species/folds cause the 60-80 RMSE tail, then build a
domain-invariant correction around the protected `band616` anchor. The cluster route
should stay as a diagnostic, not as a candidate submission path yet.
