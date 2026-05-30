# 戦略 V2（EDA + 文献調査後 2026-05-29）

## 確定ファクト

- Public ベスト: **`candidate_linear_1f`** = 17.496（5/27・5/29 再提出で再現）
- snv_diff1: Public ~24.95（LOO は良いが提出不可）
- Private: **`candidate_linear_1f.csv`** を必ず選択

## 実装する仮説（3本）

### H-A: `candidate_linear_1f_nn_bias`（最優先）

**内容**: test 樹種 → スペクトル最近傍 train 樹種の **平均残差**を引く。

- predictor: `nearest_train_species_bias_corrected`
- preprocessor: `spectral`
- EDA根拠: test 4/6 compressed、train ベイスギ等に大 bias

**棄却条件**: public_diff > 10 または group_species RMSE が 1f より +3 悪化

### H-B: `candidate_linear_1f_moisture_bins_v2`（第二）

**内容**: 波長 **616 固定**、含水率4ビンごとに slope/intercept のみ（global への縮小あり）。

- predictor: `moisture_bin_fixed_wavelength`
- preprocessor: `spectral`

**棄却条件**: moisture_bins v1 と同様に local RMSE 爆発、public_diff > 10

### H-C: `candidate_linear_1f_msc_select_raw`（第三）

**内容**: MSC 特徴で波長選択、**raw スペクトル**で回帰。

- predictor: `msc_select_raw_linear`
- preprocessor: `spectral`（内部で MSC 行列も fit 時に構築）

**棄却条件**: 選択 index が 616 から大きく外れ public_diff > 10

## 評価プロトコル

| 用途 | 指標 |
|------|------|
| ローカル主 | `group_species` CV |
| ローカル副 | `moisture_quantile` CV（≤ 31.05 + 2） |
| 提出判断 | `public_diff` 0 < d < 10、上記 CV |
| 成功 | Public **< 17.496** |

```bash
python run.py <exp> --cv group_species
python run.py <exp> --cv moisture_quantile
python run.py --public-diff <exp>
python scripts/phase_gate_check.py --phase all  # 拡張後
```

## 提出ルール

- 週 **最大 1 本**、ゲート通過かつ Public 改善見込みのみ
- 今回バッチ: 3 本までローカル検証 → **Public 実測は 1 本だけ**（最良候補）

## フェーズ4 実装結果（2026-05-29）

| 実験 | public_diff | moisture_cv | group_species_cv | ゲート | 判定 |
|------|-------------|-------------|------------------|--------|------|
| `candidate_linear_1f_nn_bias` | **8.41** | **25.93** | **28.44** | PASS | 探索提出候補（1f: mq=31.05, gs=32.46） |
| `candidate_linear_1f_moisture_bins_v2` | 223.2 | 151.95 | 130.37 | fail | **棄却**（ビン slope でも悪化） |
| `candidate_linear_1f_msc_select_raw` | 107.5 | 120.66 | 114.67 | fail | **棄却**（MSC 選択 + raw 回帰は不安定） |

- **H-A** のみ `scripts/phase_gate_check.py --phase 4` を通過。ローカル CV は 1f より改善。
- **H-B / H-C** は STRATEGY 上の棄却条件どおりローカルで破綻。
- Public &lt; 17.496 は未確認。**提出する場合**（週1枠）:

```bash
python run.py candidate_linear_1f_nn_bias --submit
```

- 提出しない場合は Private / Public ベストとも `candidate_linear_1f` を維持。

## やらないこと

- snv_diff1 再提出、ブレンド、LOO 最適化のみの改良、同一予測の再提出
