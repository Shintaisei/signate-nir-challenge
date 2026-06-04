# 評価指標見直しメモ 2026-06-02

## 目的

今日の提出結果から、ローカルCVだけではPublicを説明できないことがはっきりした。目的は、次の候補を出す前に「Publicで大外ししそうな候補」を落とし、過去にうまくいった手法・失敗した手法を同じ評価軸で説明できる状態にすること。

現時点の保護アンカーは `candidate_band616_pls_c2_smooth5`。Publicは `16.151771224841994`。

## 結論

使うべき評価は、ローカルCVの順位付けではなく、band616アンカーからのテスト分布上のズレを補正込みで見る評価。

推奨する実運用は次の3段階。

1. `scripts/candidate_risk_report.py` で raw anchor drift を見る。
2. `scripts/adjusted_drift_report.py` で、global補正・species補正後も形がズレているかを見る。
3. `scripts/calibrated_public_estimate.py --audit` の考え方で、測定済みPublicから作った粗いPublic推定を参考値として見る。

ローカルの `group_species` / `moisture_quantile` / `random` は、候補の過学習検知には使えるが、今回のPublic順位をそのまま予測する指標としては使わない。

## なぜローカル評価がズレたか

測定済み8行では、Publicと強く相関したのはアンカーからのズレだった。

| 指標 | Public悪化との関係 | 解釈 |
| --- | ---: | --- |
| `max_abs_species_mean_shift` | 強い正相関 | テストspeciesごとの平均予測がアンカーからズレるほど危険 |
| `raw_diff_rmse` | 強い正相関 | アンカーから大きく離れる候補ほど危険 |
| `mean_std_aligned_rmse` | 強い正相関 | 全体の平均・分散を直してもズレる候補は危険 |
| `species_mean_std_aligned_rmse` | 中程度の正相関 | speciesごとの水準を直しても形が違う候補はかなり危険 |
| `group_species` | 逆方向に相関 | 今回はローカルgroupが良いほどPublicが悪いケースがある |

特に `candidate_multiview_pca100_ridge300` は、ローカルでは非常に良く見えたが、Publicは `24.178044669023276`。これは、multiview PCA/Ridge が水分ではなく species / scatter / baseline / instrument 差を拾い、ローカルCVでは同種内補間として効いたためと見ている。

## 補正済み評価の読み方

`adjusted_drift_report.py` の各値は、正解RMSEではなく「候補予測とアンカー予測の距離」。

| 補正 | 何を見るか |
| --- | --- |
| raw | 候補がアンカーからどれだけ動いたか |
| mean aligned | 全体平均だけ合わせてもまだズレるか |
| mean/std aligned | 全体平均・分散を合わせても形が違うか |
| species mean aligned | テストspeciesごとの平均を合わせても形が違うか |
| species mean/std aligned | speciesごとの平均・分散まで合わせても残る形のズレ |

`candidate_raw_window10pca1_ridge1000` の `species_mean_std_aligned_rmse = 2.2929` は、Public RMSEが2.29という意味ではない。speciesごとの平均・分散をアンカーに合わせた後、予測形状がアンカーに近いという意味。実Publicは `17.711250233750594` なので、単体ではアンカーより悪い。

## 過去候補の説明

| 候補 | Public | 評価上の分類 | 判断 |
| --- | ---: | --- | --- |
| `candidate_band616_pls_c2_smooth5` | 16.1518 | 保護アンカー | 現在の基準。狭いband616周辺が転移しやすい水分信号を拾っている |
| `candidate_linear_1f_anti_nn_bias_wm040` | 17.0843 | アンカーよりは悪いが大崩れしない | 1fより改善。ただしspecies biasが大きく、主軸にはしにくい |
| `candidate_linear_1f_anti_nn_bias_wm025` | 17.1508 | 同上 | anti方向は direct nn より正しいが、band616は超えない |
| `candidate_linear_1f_anti_snv25_pred_wm025` | 17.3841 | 軽い補正候補 | mildには効くが、強く使うとspecies driftが出る |
| `candidate_raw_window10pca1_ridge1000` | 17.7113 | 形は近いが水準がズレる | catastrophicではない。ただし単体では弱い |
| `candidate_band616_anchor_r8_blend_w005` | 17.8623 | アンカー近傍だが方向が悪い | raw driftが小さくても改善保証はないことを示した |
| `candidate_linear_1f_nn_bias` | 21.3572 | transductive bias失敗 | nearest species補正の向きがPublicでは逆に働いた |
| `candidate_multiview_pca100_ridge300` | 24.1780 | 形そのものが壊れている | broad multiview PCAは危険。補正後も大きくズレる |

## 採用する評価ゲート

次のルールで候補を落とす。

| 条件 | 判断 |
| --- | --- |
| `raw_diff_rmse > 10` | 原則reject。multiview/global PCA系の大外しを防ぐ |
| `max_abs_species_mean_shift > 12` | 原則reject。テストspeciesの水準が動きすぎ |
| `species_mean_std_aligned_rmse > 6` | 原則reject。補正しても予測形状が違う |
| negative predictionが出る | 原則clip後も再評価。multiviewでは危険サイン |
| raw driftが小さいだけ | accept理由にしない。r8 blendで失敗済み |
| calibrated estimateがアンカー未満 | 参考にするが単独採用しない。測定済み8行なので不確実 |

このゲートは「Publicを正確に当てる」よりも「大外しを避ける」目的で使う。Public推定のLOOは MAE `0.8534`, RMSE `1.0501` だが、近傍候補では外れるので、誤差は最低でも1.0から1.5程度を見る。

## 指標探索の追加結果

`scripts/metric_alignment_report.py` を追加し、測定済みPublic 8件で従来指標と新指標の順位ズレを比較した。出力は `outputs/metric_alignment_report.md`。

| 指標 | Public順位との平均ズレ | 結論 |
| --- | ---: | --- |
| `group_species` | 4.00 rank | 使わない。multiviewを1位に置いてしまう |
| `moisture_quantile` | 4.00 rank | 使わない。今回のPublicと逆方向 |
| `worst_species_rmse` | 3.75 rank | 使わない。anchorが悪く見える |
| `raw_diff_rmse` | 1.25 rank | 大外し検知に使える |
| `max_abs_species_mean_shift` | 1.25 rank | 大外し検知に強い |
| `mean_std_diff_rmse` | 1.25 rank | 大外し検知に強い |
| `species_mean_std_diff_rmse` | 1.75 rank | 形状破綻の診断に使える |
| `corr_to_anchor` | 1.25 rank | 補助指標として使える |

サブエージェント側の独立確認でも、厳しめの提出前ゲートとして次が提案された。

```text
reject if:
  raw_diff_rmse > 8
  or max_abs_species_mean_shift > 12
  or mean_std_diff_rmse > 7
  or negative predictionあり
```

このルールは測定済み8件では、Public `20` 超え事故の `candidate_linear_1f_nn_bias` と `candidate_multiview_pca100_ridge300` を両方弾き、誤rejectはなかった。ただし、`candidate_band616_anchor_r8_blend_w005` は通過したのにPublic `17.8623` へ悪化した。したがって、このゲートは「事故回避」用であり、「改善保証」用ではない。

## 分類器・補正器の再評価

分類器/補正器を新指標で見直すと、過去のPublic結果をかなり説明できる。

| 候補 | Public | raw drift | species shift | shape drift | calibrated est | 新指標での判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `candidate_linear_1f` | 17.496 | 2.662 | 3.403 | 1.438 | 17.396 | pass。Publicも悪くない |
| `candidate_linear_1f_nn_bias` | 21.357 | 8.825 | 16.397 | 1.438 | 20.278 | reject。species shiftで事故を検知 |
| `candidate_linear_1f_anti_nn_bias_wm025` | 17.151 | 3.388 | 5.748 | 1.438 | 17.569 | pass。Public改善と整合 |
| `candidate_linear_1f_anti_nn_bias_wm040` | 17.084 | 4.284 | 7.155 | 1.438 | 17.834 | pass。Public改善と整合 |
| `candidate_linear_1f_anti_snv25_pred_wm025` | 17.384 | 2.658 | 3.423 | 1.448 | 17.246 | pass。Public中位と整合 |
| `candidate_raw_window10pca1_ridge1000` | 17.711 | 4.676 | 7.577 | 2.293 | 18.786 | passだが弱い |
| `candidate_raw_window20pca1_ridge1000` | 未提出 | 3.452 | 6.087 | 1.721 | 18.146 | passだが単体改善期待は低い |
| `candidate_raw_window10pca2_ridge1000` | 未提出 | 21.767 | 26.030 | 10.694 | 25.403 | reject。PCA2で形が壊れる |
| `candidate_raw_pca50_ridge100` | 未提出 | 16.663 | 18.023 | 10.493 | 22.684 | reject。広域PCAは危険 |
| `candidate_multiview_pca100_ridge300` | 24.178 | 23.466 | 25.927 | 14.025 | 23.988 | reject。Public事故と一致 |
| `candidate_band616_anchor_r8_blend_w005` | 17.862 | 0.479 | 0.579 | 0.178 | 16.690 | passしたが失敗。方向禁止が必要 |

この表から、direct `nn_bias` はローカルでは良かったが、test speciesの平均シフトが大きいため失敗したと説明できる。逆に `anti_nn` と `anti_snv` はローカルでは悪く見えても、アンカーとの形状が近いためPublic 17台前半から中盤に収まったと説明できる。

追加で `scripts/all_candidate_drift_table.py` を作成し、保存済み全submissionに同じ評価を当てた。出力は `outputs/all_candidate_drift_table.csv`。未提出候補も同じゲートで確認できる。

## 次の探索方針

ブレンドを主軸にしない前提なら、広い特徴量をそのまま増やす方向は一旦止める。次に試すなら、次の条件を満たす単体モデルだけに絞る。

1. band616アンカーから大きく離れない。
2. `species_mean_std_aligned_rmse` が低い。
3. 予測平均・species平均がアンカーから大きくズレない。
4. ローカルCVが良くても、anchor drift が悪ければ提出しない。

現時点では、`window10/20 PCA1` は「形は近いが単体では弱い」候補。`PCA2`、full raw PCA、multiview PCAは、特徴量を増やす方向として魅力はあるが、測定済み結果ではPublicに対して危険。

## 今日の追加提出結果

攻め優先で3枠を使い、評価指標の妥当性を追加検証した。

| 候補 | Public | 事前推定 | 結果 | 学び |
| --- | ---: | ---: | --- | --- |
| `candidate_linear_1f_anti_snv25_pred_wm050` | 17.3368 | 17.394 | wm025の17.384から改善 | anti-SNVはwm050まで少し伸びるが、anti-NN wm040には届かない |
| `candidate_linear_1f_anti_nn_bias_wm045` | 17.0859 | 17.955 | wm040の17.0843とほぼ同点 | anti-NNはwm040付近がピーク。wm050へ進む価値は低い |
| `candidate_raw_window20pca1_ridge1000` | 17.5032 | 18.106 | window10 PCA1の17.711から改善 | window幅20 + PCA1はwindow10より安定。PCA2/full PCAは引き続きreject |

測定済みPublicは11件になり、`calibrated_public_estimate.py --audit` は LOO MAE `0.7593`、RMSE `0.9437`。まだ近傍候補の微差は外すが、大外し検知と候補ファミリーの方向判断には使える。
