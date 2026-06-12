# 最高スコア q5 の再現メモ

## 現在最高

| 項目 | 値 |
| --- | --- |
| Public score | `13.922966797992677` |
| Candidate bucket | `attack_high_oof_cluster2` |
| Submission | `nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv` |
| Copied CSV | `artifacts/current_best_q5_submission.csv` |
| Row format | `550 x 2`, no header |
| Negative predictions | `0` |
| Confirming file | `artifacts/tomorrow_5queue_public_results_20260609.csv` |

## 実行ファイル

中心スクリプト:

```text
scripts/nir_slot1_testnear_branch_search.py
```

この引き継ぎフォルダ内にもコピーを置いています。

```text
handoff/current_best_q5_20260612/scripts/nir_slot1_testnear_branch_search.py
```

元のリポジトリ側スクリプトには、q5 を固定アンカーとして再構成する関数 `make_q5_oof` も追加済みです。q5 自体は `--anchor-current-best` の Stage4 探索で生成された候補です。

## q5 候補プール再生成コマンド

ログ上の q5 生成 run は `outputs/nir_slot1_testnear_branch/20260608_232629/` に保存されています。対応する実行形は以下です。

```powershell
python scripts/nir_slot1_testnear_branch_search.py --focus-best --anchor-current-best --cluster-count 80
```

同じコマンドを `scripts/run_reproduce_q5_pool.ps1` に入れています。

注意:

- ラッパーは repo 側の `scripts/nir_slot1_testnear_branch_search.py` を実行します。handoff 内コピーを直接実行すると、script 内の `ROOT` と import 前提が変わるためです。
- 実行前に repo 側スクリプトと handoff 内コピーの SHA256 が一致することを検証します。一致しない場合は停止します。
- 実行すると新しい timestamp の `outputs/nir_slot1_testnear_branch/<timestamp>/` が作られます。
- 完全なファイル名は beta 推定値などに依存するため、まず生成後の `candidates.csv` / `candidates/` を確認してください。
- q5 の提出済み CSV 自体は `artifacts/current_best_q5_submission.csv` に固定コピー済みです。

検証済み SHA256:

- `scripts/nir_slot1_testnear_branch_search.py`: `81B9508A89104C52F6B38E19E1F6FFDDD8D2967BAD23480F0E268A732C4E143D`
- `artifacts/current_best_q5_submission.csv`: `FD7EAE6611396E6EFAFCE5602D80882B2C61F194E95883D935F3694B5AC258E4`

## q5 の診断値

`artifacts/tomorrow_5queue_public_results_20260609.csv` より:

| Metric | Value |
| --- | ---: |
| diff RMSE vs prior current best | `0.0157734961859651` |
| max diff | `0.1345116642963404` |
| changed rows | `22` |
| new changed rows | `7` |
| prior correction corr | `0.6090718895429249` |
| Stage4 diff corr | `0.3610356149809532` |
| Stage4 changed Jaccard | `0.2058823529411764` |
| species shift | `0.0048360174495772` |
| corrected species max | `5` |
| top10 species max | `3` |
| clip saturation | `0.0` |
| OOF delta vs prior current best | `-0.0071600273955922` |
| signal corr gate | `0.326600340955879` |
| pred min / mean / median / max | `11.308120081846216 / 36.33667307332791 / 20.200266786472056 / 141.50533599590466` |

## なぜ q5 が勝ったか

- q1-q5 の 5 本すべてが、それまでの best `13.926798195477131` を改善した。
- 最も aggressive な q5 が一番良かったため、以前の gate は保守的すぎた可能性が高い。
- ただし q5 は全体を動かしていない。22 行だけを小さく補正し、最大差分も `0.1345` に抑えている。
- `top10_species_max=3`、`clip_saturation=0`、species shift 小、という安全策は効いている。

## 再現前提

- `data/raw/` と `data/submissions/` は `.gitignore` 対象です。別環境では元データとアンカー CSV 群を配置してください。
- 特に以下は必要です。
  - `data/raw/train.csv`
  - `data/raw/test.csv`
  - `data/raw/sample_submit.csv`
  - `data/submissions/nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv`
  - `data/submissions/nir_s2_lbr_impact_abs_wmean_k40_aw0_top5_s0p003_20260607.csv`
  - `data/submissions/nir_hmoe_high80rf_lrespls1_top6p5_s0p0035_c0p2_20260607.csv`
  - `data/submissions/nir_s1tn_pls_knncluster_k6_q13_c5_clcap3_f4_s004_c018_20260608.csv`
  - `data/submissions/nir_s2tn_curbest_pls_k75_q13_c5_clcap2_f045_s002_c018_20260608.csv`
  - `data/submissions/nir_s3tn_curbest_pls_k6_q16_c4_clcap2_f04_s0012_c018_20260608.csv`
  - `data/submissions/nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv`
- このフォルダには q5 の提出 CSV コピーはありますが、raw データは含めていません。
- `Q5_BEST` も起動時に読まれます。q5 候補プールの再生成では既存 q5 CSV を比較・検証用に配置してください。
- `artifacts/*.csv` の一部には生成当時の絶対パスが残っています。根拠ログとしてそのまま残し、相対パスはこのドキュメント側で併記しています。
