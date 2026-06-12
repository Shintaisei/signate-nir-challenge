# 現在の方向性と次アクション

## 現在の基本方針

q5 を新しい protected anchor として扱います。

```text
q5 = nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv
Public = 13.922966797992677
```

次の改善は、q5 の再増幅ではなく、q5 が説明できていない hard rows を見つけて小さく補正する方向です。

## 今やっている修正方向

### 1. q5 anchor residual search

目的:

- q5 を anchor にして、もう一段だけ residual correction を足す。
- q5 と同じ変更行をなぞるだけの候補を reject する。

主な gate:

- diff RMSE vs q5: `0.006-0.020`
- max diff vs q5: normal `<=0.14`, attack `<=0.16`
- changed rows: `16-26`
- q5 new rows: `>=6`
- q5 increment corr: ideal `<0.65`, hard limit `<=0.80`
- q5 overlap: hard limit `<=0.65`
- species shift: `<=0.008`
- top10 species max: `<=3`
- clip saturation: `0`
- OOF delta vs q5: normal `<= -0.0035`, attack `<= -0.005`

### 2. detector-gated residual

目的:

- q5 が触っていない suspicious rows を、species/sample number の hardcode なしで拾う。
- PCA Q residual、KNN distance、LOF、IsolationForest を rank average した detector を使う。

重要な診断:

- q5 changed rows: `22`
- never changed rows: `492`
- detector と abs(q5 diff) の相関: `-0.0067`
- つまり detector は q5 の単純な再発見ではなく、別領域を見ている。

使い方:

- detector だけで補正方向を決めない。
- residual branch signal と detector の intersection として使う。
- species/sample number は診断表示だけに使い、candidate gate へ直接入れない。

### 3. local MBL-style branch

目的:

- NIR/soil spectroscopy の local modeling / MBL の発想を使う。
- 各 test sample に近い train neighbours から local PLS/Ridge signal を作る。

制約:

- q5 の置換ではなく residual signal として使う。
- nearest train species の residual をそのまま転写しない。
- test target、Public 後に推定した label、leakage 由来の label は使わない。通常の train `y` は supervised 学習として使う。

## 次に渡すべき具体アクション

1. `artifacts/stage5_q5_candidate_manifest_20260609.csv` の cand1-cand3 を、Public 未確認候補としてレビューする。
2. q5 との差分が小さく、q5_new rows が多く、q5 corr が低い候補を優先する。
3. `scripts/nir_next5_*` の候補群は、q5 後の探索ログとして見る。ただし Public 確認済み best と混同しない。
4. 次の候補を提出する前に、以下を必ず確認する。
   - 550 rows / 2 cols / no header
   - negative predictions = 0
   - diff RMSE / max diff vs q5
   - changed rows and q5_new rows
   - q5 corr and q5 overlap
   - species shift and top10 species max
   - OOF delta vs q5
   - candidate が q5 の単純増幅ではないこと

## やらないこと

- q5 の shrink だけを少し増やす単純 sweep に寄りすぎない。
- species name / sample number / Public 後の changed row を直接条件にして提出候補を作らない。
- local CV が良いだけの broad high-dimensional model を提出しない。
- q5 以後の未確認候補を「最高スコア」として扱わない。

## 推奨レビュー観点

レビュー担当には以下を厳しめに見てもらってください。

- 最高スコアの根拠ファイルと数値が一致しているか。
- 再現コマンドが実際の中心スクリプトと整合しているか。
- 提出 CSV を raw data や label と混同していないか。
- 「Public confirmed」と「generated candidate」が明確に分離されているか。
- 次アクションが q5 過適合を避ける形になっているか。
- target leakage のリスクがないか。
