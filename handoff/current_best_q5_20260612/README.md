# SIGNATE NIR 引き継ぎ: current best q5

作成日: 2026-06-12

このフォルダは、チームメンバーへ現在の最高 Public スコア、再現に必要な実行ファイル、これまでの試行履歴、今後の修正方向を一括で渡すための引き継ぎパックです。

## 結論

- 現在の確認済み最高 Public: `13.922966797992677`
- 提出ファイル: `artifacts/current_best_q5_submission.csv`
- 元ファイル: `data/submissions/nir_tomorrow_q5_s4tn_attack_high_oof_cluster2_k6_q1p6_c5_cl2_f0p04_s0p04_c0p18_b0p101219_20260608.csv`
- 生成系の中心スクリプト: `scripts/nir_slot1_testnear_branch_search.py`
- 勝ち筋: `test-near golden PLS residual + cluster-capped correction`
- 現在の次方向: q5 を新アンカーにして、q5 の同じ行を増幅するだけの探索を避ける。

## Public confirmed と未確認候補

| 区分 | ファイル / 系列 | 扱い |
| --- | --- | --- |
| Public confirmed best | `artifacts/current_best_q5_submission.csv` | 最高 Public `13.922966797992677` として共有してよい |
| Public confirmed queue | `artifacts/tomorrow_5queue_public_results_20260609.csv` | q1-q5 全ての Public 結果。q5 が勝者 |
| Generated, not current confirmed best | `artifacts/stage5_q5_candidate_manifest_20260609.csv` | q5 後の候補。提出済み最高とは混同しない |
| Generated, not current confirmed best | repo 側 `scripts/nir_next5_*.py` の出力群 | 次探索ログ。Public 確認が別途必要 |

## フォルダ構成

| Path | 用途 |
| --- | --- |
| `README.md` | 最初に読む概要 |
| `BEST_SCORE_REPRODUCTION.md` | 最高スコア q5 の再現情報、コマンド、検証 |
| `INTERNAL_EVALUATION_GUIDE.md` | 内部評価の考え方、CV/OOF/anchor drift/q5 gate の使い分け |
| `EXPERIMENT_HISTORY_CONDENSED.md` | これまでの探索を圧縮した履歴 |
| `CURRENT_DIRECTION_AND_NEXT_ACTIONS.md` | 今やるべき方向性とレビュー基準 |
| `REVIEW_LOG.md` | レビューエージェントの厳しめレビュー記録 |
| `scripts/nir_slot1_testnear_branch_search.py` | q5 系候補を生成した中心実行ファイルのコピー |
| `scripts/run_reproduce_q5_pool.ps1` | q5 候補プールを再生成するためのラッパー |
| `artifacts/current_best_q5_submission.csv` | 現在最高 Public の提出 CSV コピー |
| `artifacts/tomorrow_5queue_public_results_20260609.csv` | q1-q5 の Public 結果 |
| `artifacts/tomorrow_5queue_manifest_20260608.csv` | q1-q5 生成時の候補マニフェスト |
| `artifacts/stage5_q5_candidate_manifest_20260609.csv` | q5 以後の Stage5 候補マニフェスト |
| `artifacts/stage5_direction_summary.json` | q5 後 detector / changed-row 診断の要約 |
| `artifacts/top_uncorrected_detector_species.csv` | detector 上位未補正 rows の species 集計 |
| `artifacts/top_uncorrected_detector_clusters.csv` | detector 上位未補正 rows の cluster 集計 |

## 固定コピーの検証値

| 対象 | SHA256 |
| --- | --- |
| q5 生成スクリプトコピー | `81B9508A89104C52F6B38E19E1F6FFDDD8D2967BAD23480F0E268A732C4E143D` |
| q5 提出 CSV コピー | `FD7EAE6611396E6EFAFCE5602D80882B2C61F194E95883D935F3694B5AC258E4` |

## 相手にまず伝えること

1. q5 が現在の確認済み最高スコアです。q5 より後の `nir_next5_*` は候補生成済みですが、ローカル記録上は Public 確認済みの最高ではありません。
2. Public は local CV とかなりズレます。local CV は破綻検知であり、Public の直接最適化指標ではありません。
3. q5 の勝ち筋は「全体置換」ではなく、既存アンカーから 22 行だけ小さく補正したことです。
4. 次は q5 をアンカーにします。ただし q5 と同じ行の再増幅は避け、q5 未説明の hard rows を狙う必要があります。
5. 提出は自動で行わず、候補ごとにゲートとレビューを通してから判断してください。
