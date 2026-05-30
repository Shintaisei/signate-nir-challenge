# 類似文献・コンペ調査メモ（2026-05-29）

## 文献の一般論

| 手法 | 文献での位置づけ | 本コンペでの実測 |
|------|------------------|------------------|
| **PLS + SNV/MSC + 微分** | 木材含水率 NIR の標準（R² 0.9 前後の報告多数） | PLS+snv_diff1 → Public **35.7**（悪化） |
| **波長選択 + PLS** | CARS/MRF 等で変数選択 | TopK/PLS 系は Public 21〜35 台 |
| **単回帰・少変数** | 報告は少ないが過学習リスク低 | **raw 1波長 → Public 17.5（ベスト）** |

参照:
- [Scientific Reports: NIR for soil-mixed woody biomass MC](https://www.nature.com/articles/s41598-026-36901-8) — SNV+SG+PLSR
- [Wood and Fiber Science: Korean pine MC](https://wfs.swst.org/index.php/wfs/article/view/1851) — 微分+PLS、含水率帯でモデル分割
- [USDA FPL: handheld NIR MC](https://www.fpl.fs.usda.gov/documnts/pdf2024/fpl_2024_thapa001.pdf) — 前処理の組合せ最適化が性能を左右

## 本コンペ固有の教訓

1. **train 内相関が高い前処理 ≠ Public 改善**（snv_diff1: |corr| 0.91 vs raw 0.78、Public は逆）
2. **樹種非重複** → 文献の「同一材種校正」前提と異なる。スペクトル最近傍は cos>0.99 だが、**樹種別モデル移植は失敗**（nearest_train RMSE≈120）
3. **含水率帯によるモデル分割**は文献で有効例あり（FSP 上下）→ H-B v2 の根拠
4. **レンジ圧縮**は EDA で test 4/6 樹種が compressed → バイアス/スケール補正が論点

## 採用 / 保留 / 棄却

| 手法 | 判定 | 理由 |
|------|------|------|
| raw 1波長（index 616 付近） | **採用（維持）** | Public 17.496 再現 |
| SNV+diff1 単波長 | **棄却** | Public ~25 一貫 |
| PLS / TopK Ridge | **棄却** | Public 悪化実績 |
| Group curve / blend | **棄却** | Public 21.17（1f より悪い） |
| ブレンド・clip | **棄却** | LOO↓ Public↑ |
| 含水率帯別 slope（固定波長） | **保留→実装** | H-B v2 |
| NN train 樹種の残差バイアス | **保留→実装** | H-A |
| MSC で波長選択→raw 回帰 | **保留→実装** | H-C |
| 樹種ごと別波長モデル | **棄却** | nearest_train 系の失敗 |

## SIGNATE / コンペ類似

- 近赤外研究会チャレンジ: 木材含水率 RMSE、train/test 樹種非重複
- 一般的な chemometrics コンペでは PLS が強いが、**ドメインシフトがある本データでは単純化が勝った**

## 実装への示唆

- 複雑化より **1 波長 + ドメイン適応後処理**（バイアス・ビン slope）
- ローカル CV は **group_species** を主、LOO は補助のみ
- 提出は `public_diff` 0〜10 かつ group_species 改善時のみ
