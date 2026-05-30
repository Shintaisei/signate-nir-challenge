# 本日の最終候補（2026-05-29）

Public ベスト: **`candidate_linear_1f`** = **17.496**（再現済み）

## 5枠プラン（`outputs/daily_candidates.csv`）

| 枠 | 実験 | 役割 | public_diff | moisture_cv | group_species | ゲート |
|----|------|------|-------------|-------------|---------------|--------|
| 1 | `candidate_linear_1f` | **Private 必須**・Public 保険 | 0 | 31.05 | 32.46 | 基準 |
| 2 | `candidate_linear_1f_blend_snv25` | **今日の探索 #1**（前処理） | ~4.25 | 29.92 | 31.85 | PASS |
| 3 | `candidate_linear_1f_blend_snv25_nn_bias` | **ローカル最良**（前処理+補正） | **10.18** | **23.79** | **30.79** | pd 僅かに超過 |
| 4 | `candidate_linear_1f_nn_bias` | 代替探索（補正のみ） | 8.41 | 25.93 | **28.44** | PASS |
| 5 | `group_curve_blend_snv_diff1` | Private 第2・Public **21.17** | 大 | — | — | 保険のみ |

## 今日の推奨アクション

### Private（2枠選択）

1. **`candidate_linear_1f.csv`**（必須）
2. もう1枠も **`candidate_linear_1f.csv`** または `group_curve_blend_snv_diff1.csv`（21.17）

### Public 探索提出（週1本まで）

**第1候補（ゲート厳守）: `candidate_linear_1f_blend_snv25`**

- `public_diff` **4.25**、moisture_cv **29.92** — ゲート PASS
- 前処理のみの変更でリスクが最も低い

**第2候補（ゲート PASS）: `candidate_linear_1f_nn_bias`**

- `public_diff` **8.41**、group_species **28.44** — 補正のみ

**攻め枠（ローカル最良・ゲート pd 僅超）: `candidate_linear_1f_blend_snv25_nn_bias`**

- mq **23.79** / gs **30.79** で全候補最良
- `public_diff` **10.18**（上限 10 をわずかに超過）— 枠に余裕があれば検討

**提出しない**

- `smooth3/5`（1f と同一）、`blend_snv50`、`center`/`l2norm`、`moisture_bins` 系

```bash
# daily_candidates 再生成
python3 run.py --top5

# 探索提出（1本）
python3 run.py candidate_linear_1f_blend_snv25_nn_bias --submit
python3 run.py --record-public candidate_linear_1f_blend_snv25_nn_bias <score>
```

## ローカル指標まとめ（1f 基準）

| 実験 | moisture_cv | group_species | 備考 |
|------|-------------|---------------|------|
| 1f | 31.05 | 32.46 | Public 17.496 |
| blend_snv25 | 29.92 | 31.85 | 前処理のみ |
| **blend_snv25_nn_bias** | **23.79** | **30.79** | **本日最有力** |
| nn_bias | 25.93 | 28.44 | 補正のみ |
