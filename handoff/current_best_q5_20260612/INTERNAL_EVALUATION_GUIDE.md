# 内部評価の考え方

このコンペでは、内部評価を「Public を正確に予測する指標」として使わない。内部評価の目的は次の 3 つです。

1. Public で大外ししそうな候補を落とす。
2. 同じ候補ファミリー内で、相対的に安全な候補を選ぶ。
3. Public feedback と矛盾した local 結果を、次の仮説へ反映する。

現在の protected anchor は q5 です。

```text
q5 Public = 13.922966797992677
q5 file = nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv
```

## なぜ普通の CV を信用しすぎないか

train species と test species は overlap しません。そのため、普通の random CV や train species 内の補間性能は Public とズレやすいです。

実際に、測定済み Public では以下のような逆転が起きています。

| 例 | local の見え方 | Public |
| --- | --- | ---: |
| `candidate_linear_1f_nn_bias` | local は良い | `21.35722898556263` へ悪化 |
| `candidate_multiview_pca100_ridge300` | local CV 上位 | `24.178044669023276` へ大悪化 |
| `candidate_linear_1f_anti_nn_bias_wm040` | local は悪く見える | `17.084250210869193` へ改善 |
| `candidate_band616_anchor_r8_blend_w005` | anchor drift は小さい | `17.86230359396826` へ悪化 |
| q5 | attack-range だが guard 内 | `13.922966797992677` で現 best |

結論:

- `group_species`、`moisture_quantile`、`random` は最終 ranker ではない。
- local CV は破綻検知、OOF delta、方向性診断に使う。
- Public history と anchor からのズレを必ず合わせて読む。

## 内部評価の階層

### 1. Format and sanity gate

提出前に必ず確認する最低条件です。

| Check | Pass condition |
| --- | --- |
| CSV shape | `550 x 2` |
| Header | no header |
| sample order | `sample_submit.csv` と同じ |
| negative predictions | `0` が基本。出る場合は clip 後に再評価 |
| raw data / labels | handoff や提出 CSV に raw data / target label を混ぜない |

### 2. Local CV and OOF gate

local CV は Public 予測ではなく、候補が壊れていないかを見るために使います。

| Metric | 使い方 |
| --- | --- |
| `group_species` | train species holdout で破綻していないかを見る |
| `moisture_quantile` | 水分レンジごとの偏りを見る |
| `random` | IID 的な学習破綻だけを見る。優先度は低い |
| OOF delta vs anchor | 同じ候補ファミリー内の相対比較に使う |
| improved group count | correction が一部 group だけに効いていないかを見る |
| signal/residual corr | gate で拾った行の correction signal が residual と同じ向きかを見る |

重要:

- local CV が良いだけでは提出理由になりません。
- q5 以後は `OOF delta vs q5` を見る。通常は `<= -0.0035`、attack 候補は `<= -0.005` を目安にする。
- OOF が良くても、anchor drift や q5 overlap が悪ければ reject する。

### 3. Anchor drift gate

過去の失敗は「正解 RMSE」よりも「保護 anchor からのズレ」で説明できることが多いです。

古い band616 anchor 時代の測定済み 8 件では、Public 悪化と相関が強かったのは以下です。

| Metric | 役割 |
| --- | --- |
| `raw_diff_rmse` | anchor から動きすぎていないか |
| `max_abs_species_mean_shift` | test species ごとの予測平均が動きすぎていないか |
| `mean_std_diff_rmse` | 全体平均・分散を合わせても形が違うか |
| `species_mean_std_diff_rmse` | species ごとに平均・分散を合わせても形が違うか |
| `corr_to_anchor` | anchor と予測形状が近いか |

当時の reject 目安:

```text
raw_diff_rmse > 8-10
or max_abs_species_mean_shift > 12
or mean_std_diff_rmse > 7
or species_mean_std_diff_rmse > 6
or negative predictionあり
```

これは「改善保証」ではなく「事故回避」です。`candidate_band616_anchor_r8_blend_w005` は drift が小さくても Public `17.86230359396826` に悪化しました。近いだけでは足りず、correction の方向が合っている必要があります。

### 4. q5 anchor gate

現在は q5 が protected anchor です。候補は q5 からの追加 correction として評価します。

| Metric | Target |
| --- | --- |
| diff RMSE vs q5 | `0.006-0.020` |
| max diff vs q5 | normal `<=0.14`, attack `<=0.16` |
| changed rows | `16-26` |
| q5 new rows | `>=6` |
| q5 increment corr | ideal `<0.65`, hard limit `<=0.80` |
| q5 overlap | `<=0.65` |
| species shift | `<=0.008` |
| corrected species max | `<=6` |
| top10 species max | `<=3` |
| clip saturation | `0` |
| OOF delta vs q5 | normal `<= -0.0035`, attack `<= -0.005` |
| improved/active groups | `>=9/13` |
| signal corr | `>=0.30` |

解釈:

- q5 と同じ行をさらに増幅する候補は危険。
- q5 が触っていない hard rows を拾う候補を優先する。
- attack 候補は、Public-positive な同系列の前段結果がある時だけ検討する。

### q5 gate の列対応

q5 後候補の確認には、まず `artifacts/stage5_q5_candidate_manifest_20260609.csv` を見ます。列名と意味は以下です。

| Gate | CSV column | 定義 / 読み方 |
| --- | --- | --- |
| diff RMSE vs q5 | `slot1_diff_rmse` | 候補予測と q5 予測の RMSE。小さすぎると何もしておらず、大きすぎると anchor から離れすぎ |
| max diff vs q5 | `slot1_diff_max_abs` | 1 行あたりの最大補正量。attack でも `0.16` を超えたら危険 |
| changed rows | `changed_count` | q5 から実質的に変えた test rows 数 |
| q5 new rows | `q5_new_changed_count` | q5 が変えていなかった rows を新たに変えた数。q5 再増幅でないかを見る主指標 |
| q5 increment corr | `q5_increment_corr` | 候補の q5 からの差分と、q5 自身の差分方向の相関。高すぎると q5 の同方向増幅 |
| q5 overlap | `q5_overlap_frac` | 候補 changed rows のうち q5 changed rows と重なる割合 |
| species shift | `species_shift` | test species ごとの平均予測 shift の最大系診断。水準を動かしすぎていないか |
| corrected species max | `corrected_species_max` | correction 対象が 1 species に集中していないか |
| top10 species max | `top10_species_max` | 補正量上位 10 件が 1 species に集中していないか |
| clip saturation | `clip_saturation_frac` | clip 上限に張り付いた correction の割合。`0` 以外は危険 |
| OOF delta vs q5 | `oof_delta_vs_q5` | q5 OOF からの RMSE 差。負なら OOF 改善 |
| improved/active groups | `groups` | 改善または有効な group 数。`9/13` 以上が目安 |
| signal corr | `signal_corr_gate` | gate で拾った signal と residual の相関 |

運用例:

- `q5_new_changed_count` が多く、`q5_increment_corr` と `q5_overlap_frac` が低い候補を優先する。
- `oof_delta_vs_q5` が強くても、`slot1_diff_max_abs`、`species_shift`、`top10_species_max` が悪ければ reject する。
- `slot1_diff_rmse` が範囲内でも、`q5_new_changed_count` が少ない候補は q5 の焼き直しとして扱う。

## Target-domain / distribution 評価

test species の target label はありません。使えるのは test-side の X、species label、sample number などの公開特徴だけです。

許可:

- test X を使った unsupervised grouping
- train/test distribution distance
- fold-local detector score
- pseudo-test simulation
- species/sample number を診断表示に使う

禁止:

- test target を使う
- Public 後に推定した label を使う
- species name / sample number / Public 後の changed row を hard submission rule にする
- nearest train species の residual や target mean をそのまま test species へ転写する

分布 EDA の結論:

- SNV segment shape 系は train/test distribution matching の診断に使える。
- ただし pseudo-test でも target mean gap は大きい。
- nearest species は「診断」には使えるが、「補正方向の直接コピー」には使わない。

### pseudo-test simulation の作り方

実行元:

```powershell
python scripts/train_test_distribution_eda.py --out-dir outputs/distribution_eda_readable
```

主な出力:

| Output | 見ること |
| --- | --- |
| `outputs/distribution_eda_readable/pseudo_test_alignment_score.csv` | train species を 1 つ holdout し、残り train の X 分布だけで nearest species を推定した時の target mean gap |
| `outputs/distribution_eda_readable/test_species_match_consensus.csv` | test species が複数 spectral view でどの train species に近いか |
| `outputs/distribution_eda_readable/train_test_species_distribution_pairs.csv` | view ごとの distribution distance |
| `outputs/distribution_eda_readable/summary.json` | 使った view と注意書き |

手順:

1. train species を 1 つ pseudo-test として holdout する。
2. holdout した species の `y` は matching には使わない。
3. 残り train だけで spectral view を fit し、X 分布で nearest train species を探す。
4. holdout species の target mean と、matched train species の target mean gap を監査値として見る。
5. gap が大きい view や nearest species は、test 補正方向の根拠に使わない。

失敗扱い:

- pseudo-test target mean gap が大きい view に依存する。
- gap は絶対値だけで切らず、view 間の相対順位と worst-case gap を見る。`snv_segment_shape_pca20` のように相対的に良い view でも worst-case は大きいので、単独根拠にしない。
- 1 つの nearest species だけを根拠に correction 方向を決める。
- raw/smooth PCA の見た目の近さだけで target mean transfer する。

## Detector 評価

q5 後の detector は、q5 が拾っていない suspicious rows を探すための補助です。

実行元:

```powershell
python scripts/nir_stage5_direction_diagnostics.py
```

handoff にコピーした根拠:

| Artifact | 内容 |
| --- | --- |
| `artifacts/stage5_direction_summary.json` | q5 changed count、detector corr、top uncorrected rows などの要約 |
| `artifacts/top_uncorrected_detector_species.csv` | detector 上位未補正 rows の species 集計 |
| `artifacts/top_uncorrected_detector_clusters.csv` | detector 上位未補正 rows の cluster 集計 |

repo 側の詳細出力:

| Output | 内容 |
| --- | --- |
| `outputs/stage5_direction/stage5_row_diagnostics.csv` | row 単位の q1-q5 diff、detector score、cluster |
| `outputs/stage5_direction/q5_changed_rows.csv` | q5 が実際に変えた rows |
| `outputs/stage5_direction/uncorrected_high_detector_rows.csv` | q5 が触っていない detector high rows |
| `outputs/stage5_direction/pred_bin_detector_summary.csv` | prediction bin ごとの detector 傾向 |
| `outputs/stage5_direction/detector_component_corr.csv` | detector component の相関 |

確認済み診断:

- q5 changed rows: `22`
- q1-q5 changed any: `58`
- never changed rows: `492`
- detector と `abs(q5_diff)` の相関: `-0.0067`
- top uncorrected detector rows は q5 の単純再発見ではない。

使い方:

- detector だけで correction 方向を決めない。
- residual branch signal と detector の intersection として使う。
- detector high かつ q5 untouched な領域は候補探索の優先対象。

## Public history の使い方

Public feedback は少数サンプルですが、このリポジトリでは local CV より重要な制約です。

q1-q5 の Public readout:

| Queue | Role | Public | Gain vs previous best |
| --- | --- | ---: | ---: |
| q1 | safe_s4 | `13.924977185811285` | `0.001821` |
| q2 | safe_diverse_k75 | `13.924920044963732` | `0.001878` |
| q3 | safe_diverse_k55 | `13.924869001891032` | `0.001929` |
| q4 | attack_lite_lowcorr | `13.924188284374708` | `0.002610` |
| q5 | attack_high_oof_cluster2 | `13.922966797992677` | `0.003831` |

使い方:

- 同じファミリーで Public が連続改善したら、その方向は alive とみなす。
- q1-q5 は全て改善したため、`test-near golden PLS residual + cluster-capped correction` は alive。
- q5 が最も良かったため、以前の gate は保守的すぎた可能性がある。
- ただし、残り探索を q5 の同一方向だけに寄せると Public 過適合しやすい。

## 提出判断フロー

1. Format and sanity gate を通す。
2. q5 との差分と changed row を見る。
3. OOF delta vs q5 と improved groups を見る。
4. q5 corr / q5 overlap / q5 new rows を見る。
5. species shift / corrected species max / top10 species max を見る。
6. detector や distribution EDA は、hardcode ではなく診断として使う。
7. Public confirmed と未確認 generated candidate を混同しない。
8. レビューで leakage、過剰な q5 増幅、local CV 過信がないか確認する。

## 評価で伝えるべき一文

チームへは次のように伝えるのが一番安全です。

```text
この内部評価は Public を当てるものではなく、q5 anchor から安全に小さく動かすための事故回避・候補選別ルールです。local CV は診断、anchor drift と q5 overlap はリスク管理、Public history は方向判断に使います。
```
