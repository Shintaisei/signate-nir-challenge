# 試行履歴の凝縮まとめ

## 1. 初期ベースライン

最初の安定ベースは `candidate_linear_1f` でした。

- Public: `17.496`
- 内容: raw の単一波長付近を使った線形モデル
- 意味: 複雑な前処理や多波長モデルより、raw の局所的な水分シグナルが強かった。

## 2. anti-NN-bias が一度改善

`candidate_linear_1f_nn_bias` は local では良く見えましたが Public は悪化しました。

- `candidate_linear_1f_nn_bias`: Public `21.35722898556263`

その逆方向へ振った anti-NN-bias が改善しました。

- `candidate_linear_1f_anti_nn_bias_wm025`: Public `17.150798122243412`
- `candidate_linear_1f_anti_nn_bias_wm040`: Public `17.084250210869193`
- `candidate_linear_1f_anti_nn_bias_wm045`: Public `17.085938670600243`

学び:

- local CV の改善方向と Public 改善方向が逆になることがある。
- Public feedback を使って「local が示す補正方向の逆」が有効になるケースがある。

## 3. band616 PLS が大きく改善

`candidate_band616_pls_c2_smooth5` が確認済みで大きく改善しました。

- Public: `16.151771224841994`
- 内容: smooth5 + band616 radius5 + PLS c2

学び:

- index 616 近傍は引き続き強い。
- PLS/Savitzky-Golay/MSC/SNV は候補として有効だが、広げすぎると Public 悪化しやすい。

## 4. 広い特徴・並べ替え・通常 CV 追従は危険

Public 悪化例:

- `candidate_multiview_pca100_ridge300`: Public `24.178044669023276`
- `candidate_waveset_cmm16_elastic_sort`: Public `24.06609093687482`
- `candidate_waveset_cmm16_pls4_sort`: Public `24.326447281849173`
- `candidate_band616_pls_c2_smooth5_sort`: Public `17.631897102099344`

学び:

- train/test species が disjoint なので、普通の random CV や広い特徴セットは Public に直結しにくい。
- species/sample order を hard に使う処理は危険。診断には使えるが、提出ロジックの直接条件にはしない。

## 5. 13 点台への移行

6/6-6/7 頃に anchor residual correction 系で Public が 13 点台まで改善しました。

記録上の例:

- `nir_20260606_advgold_huber_msc_sg9_rf_dc6_c5_s0p02_c0p1`: Public `13.993686373474532`
- `nir_20260607_advgold_rf_shrink_s0p012_c0p06`: Public `13.990746861581181`
- `nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p1_mc1`: Public `13.990521171208956`

この段階で「既存アンカーを少しだけ補正する」方向が明確になりました。

## 6. Stage4 q1-q5 キューで現在最高へ

2026-06-09 の 5 本は全てそれまでの best `13.926798195477131` を改善しました。

| Queue | Role | Public | Gain |
| --- | --- | ---: | ---: |
| q1 | safe_s4 | `13.924977185811285` | `0.001821` |
| q2 | safe_diverse_k75 | `13.924920044963732` | `0.001878` |
| q3 | safe_diverse_k55 | `13.924869001891032` | `0.001929` |
| q4 | attack_lite_lowcorr | `13.924188284374708` | `0.002610` |
| q5 | attack_high_oof_cluster2 | `13.922966797992677` | `0.003831` |

勝ち筋:

- `test-near golden PLS residual`
- correction は cluster cap で集中を抑える
- anchor から小さく、少数行だけ動かす
- OOF delta が強い候補を優先する
- ただし species concentration と max diff を監視する

## 7. q5 後の探索

q5 後には以下の方向が走っています。

- `stage5_q5_anchor_residual`: q5 を anchor にした追加 residual
- `detector-gated residual`: q5 が触っていない hard rows の検出
- `local MBL-style branch`: local nearest-neighbour PLS/Ridge を residual signal として使う
- `nir_next5_*`: q5 + q5cand2 を旧 best とみなした次候補群

ただし、ローカル記録上の確認済み Public best はまだ q5 です。q5 後候補は未確認または別途確認が必要です。

## 重要な反省

- local CV は最適化対象ではなく、破綻検知と比較材料。
- 強い補正ほど良いわけではない。小さい correction を、根拠のある少数行へ当てる。
- species/sample number は診断に使う。提出ロジックの hard condition にしない。
- q5 の成功をそのまま増幅すると、Public に過適合するリスクが高い。
