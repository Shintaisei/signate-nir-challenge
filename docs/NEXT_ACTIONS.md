# 運用方針（2026-05-30）

**戦略議論の起点**: [docs/RETROSPECTIVE_2026-05-30.md](RETROSPECTIVE_2026-05-30.md)（データの性質・前処理/モデル/指標の振り返り）

## 本日の最終候補

**詳細: [docs/DAILY_FINAL_2026-05-29.md](DAILY_FINAL_2026-05-29.md)**  
**5枠 CSV: `outputs/daily_candidates.csv`**（`python3 run.py --top5`）

| 優先 | 実験 | 用途 |
|------|------|------|
| 必須 | `candidate_linear_1f` | Private / Public 保険（17.496） |
| 探索① | `candidate_linear_1f_blend_snv25` | ゲート PASS・pd=4.25 |
| 探索② | `candidate_linear_1f_nn_bias` | ゲート PASS・pd=8.41 |
| 攻め | `candidate_linear_1f_blend_snv25_nn_bias` | ローカル最良・pd=10.18 |
| 保険 | `group_curve_blend_snv_diff1` | Private 第2（Public 21.17） |

## 提出状況（2026-05-29）

| # | 実験 | Public | 状態 |
|---|------|--------|------|
| 1–4 | 2f_stable / blend_2pct / snv_diff1 / **1f** | 24.46 / 24.61 / 24.95 / **17.496** | 済 |
| **5** | **`candidate_linear_1f_blend_snv25`** | **18.558** | **提出済**（1f より +1.06 悪化） |

**結論**: 前処理調整はローカル CV 改善したが Public は **17.496 未達**。Private / Public ベストは **`candidate_linear_1f`** 維持。

Public 記録済み。再記録不要:

```bash
python3 run.py --record-public candidate_linear_1f_blend_snv25 18.558018446661805
```

Private は引き続き **`candidate_linear_1f.csv`** を選択。

## 提出コマンド（参考）
