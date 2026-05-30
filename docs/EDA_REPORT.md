# EDA レポート（2026-05-29）

Public ベスト: `candidate_linear_1f`（**17.496**、再現確認済み）

## 1. データ概要

- train 1322 / test 550、樹種名 overlap **0**
- 含水率: mean≈49.9、std≈49.5（`outputs/logs/data_profile.json`）
- raw 1f 波長 index **616**（|corr|≈0.779）、snv_diff1 は index **1350**（|corr|≈0.907）— **別バンド**

## 2. test 6樹種の予測レンジ（`eda_test_predictions_candidate_linear_1f.json`）

| test樹種 | 最近傍 train | pred mean | NN train mean | 判定 |
|----------|-------------|-----------|---------------|------|
| スギ | ヒノキ | 33.3 | 39.6 | **compressed**（std 0.49×） |
| チーク | ヒノキ | 33.1 | 39.6 | compressed |
| ケヤキ | スプルース | 44.6 | 37.1 | compressed |
| クスノキ | 米ヒバ | 54.7 | 70.1 | compressed + mean↓ |
| タモ | ナラ | 50.6 | 41.8 | **expanded**（std 1.64×） |
| ヤマザクラ | ナラ | 48.6 | 41.8 | expanded |

**回答Q1**: 4/6 樹種で **レンジ圧縮**、2/6 で拡大。Public 17.5 は「圧縮しつつ全体として当たる」パターンの可能性。

## 3. raw vs snv の波長（`eda_wavelength_profile.json`）

- raw と snv_diff1 は **index が 734 離れている**（616 vs 1350）
- snv は train 上の |corr| が高い（0.91 vs 0.78）が **Public では悪化**（~25）
- **回答Q2**: 主因は **波長バンドの違い** + snv 前処理が test ドメインと合わない可能性（相関↑≠Public↑）

## 4. train OOF 誤差構造（`eda_error_decomposition_candidate_linear_1f.json`）

最悪セル（樹種×含水率ビン）:

| 樹種 | ビン | RMSE | bias |
|------|------|------|------|
| ベイスギ | 高含水 | 120 | +116 |
| ホワイトオーク | 高含水 | 59 | -52 |
| ベイマツ | 高含水 | 48 | -15 |

**回答Q3**: train 外れ（ベイスギ等）の bias は test 最近傍（ヒノキ・ナラ等）と **直接は一致しない**。最近傍樹種の平均残差を test に移植する補正は要検証。

## 5. ローカル指標と Public（`eda_metric_alignment.json`）

| 指標 | Public との Pearson |
|------|---------------------|
| group_species | **+0.52** |
| leave_one_species | -0.06 |
| random | -0.81 |

**回答Q4**: 主指標は **group_species** を第一、**moisture_quantile** を第二（1f=31.05、Public 17.5 のギャップは大きいが外挿向け）。**LOO は信頼しない**。

## 6. EDA 結論（方針への入力）

1. **単一波長 raw（616付近）を維持** — 周辺波長も相関が固い（615–626）
2. **レンジ圧縮が主課題** — global affine だけでは不十分（range_cal 棄却済み）
3. **test 樹種ごとに NN train の残差バイアス補正** が最有力（H-A）
4. **固定波長 + 含水率ビン slope** は第二候補（H-B v2）
5. snv / PLS / Group curve は Public 実測で棄却
