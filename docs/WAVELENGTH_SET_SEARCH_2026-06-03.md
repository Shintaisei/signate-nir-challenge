# Wavelength Set Search 2026-06-03

Large-scale diagnostic EDA was run with `scripts/wavelength_set_search.py`.

- Output: `outputs/wavelength_set_search_20260603_0049`
- Window features scanned: 74,823
- Test target usage: none. Test rows were used only for X/meta distribution checks.
- CV for OOF set checks: `group_kfold(..., group_col="species number", n_splits=5)`
- Ranking metric for sets: `guard_score = rmse + 0.10 * worst_group_rmse + 0.15 * abs(worst_group_bias)`

## Strongest Individual Regions

The strongest usable region is still the SNV/difference-shaped area around raw index 1325-1360.

| Region | Good transforms | Example key | Signal |
|---|---|---|---|
| 4752-4825 nm | `snv:slope`, `snv:edge`, `snv_diff1:mean` | `snv:slope:1344:1360` | pearson -0.9075, species mean abs corr 0.9768 |
| 4837-4856 nm | `snv_diff1:mean`, `snv:center_minus_mean` | `snv_diff1:mean:1332:1338` | pearson -0.8836, species mean abs corr 0.9743 |
| 6479-6491 nm | `snv:slope/std/range/edge` | `snv:slope:908:912` | pearson -0.8808, low transfer distance |
| 5369-5427 nm | `snv_diff1:mean`, `snv:slope` | `snv_diff1:mean:1184:1192` | pearson 0.8811, species mean abs corr 0.9728 |
| 7220-7525 nm | `snv_diff1:slope`, `snv:center_minus_mean` | `snv_diff1:slope:640:720` | pearson 0.8933, species mean abs corr 0.9718 |
| 6788-6846 nm | `snv:edge`, `snv:center_minus_mean` | `snv:edge:816:832` | pearson -0.8189, secondary support |
| 4212-4285 nm | `snv:mean` | `snv:mean:1480:1500` | pearson about -0.82, useful more as residual/bias feature |

## Best OOF Feature Set

Best set:

- `center_minus_mean_all_views_diverse16 + ElasticNet(alpha=0.03, l1_ratio=0.2)`
- RMSE: 19.8346
- Guard score: 29.0917
- Worst group: `15:ベイスギ`, RMSE 49.7411, bias -28.5532

Features:

- `snv:center_minus_mean:1332:1336`
- `snv:center_minus_mean:1352:1356`
- `snv:center_minus_mean:680:720`
- `snv:center_minus_mean:1192:1196`
- `snv:center_minus_mean:908:912`
- `snv:center_minus_mean:708:712`
- `raw:center_minus_mean:1344:1348`
- `smooth5:center_minus_mean:1330:1340`
- `raw:center_minus_mean:1332:1336`
- `smooth5:center_minus_mean:1344:1348`
- `smooth5:center_minus_mean:576:580`
- `snv:center_minus_mean:828:832`
- `smooth5:center_minus_mean:1320:1324`
- `raw:center_minus_mean:1320:1324`
- `smooth5_diff1:center_minus_mean:1200:1440`
- `snv:center_minus_mean:1320:1324`

Interpretation: the best set is not just the strongest local peak. It uses local shape deviation around several separated wavelength regions. This supports a modeling direction where compact multi-region shape features are added to the existing routed/segment models.

## Modeling Implications

1. Add a compact `center_minus_mean`/local shape feature family around indices 576-580, 680-720, 828-832, 908-912, 1192-1196, 1320-1360, and optionally 1480-1500.
2. Keep the 4750-4860 nm SNV slope/edge/diff1 family as the main water-sensitive feature group.
3. Use 6479-6491 nm and 5369-5427 nm as secondary independent bands; they are strong and not merely duplicates of band 616.
4. Treat 4212-4285 nm (`idx1485`) as a residual/bias feature rather than the main predictor.
5. Continue to rank by guard score, because the best RMSE still has a large ベイスギ bias and plain RMSE can hide this.

