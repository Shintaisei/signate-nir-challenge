# Review log

このフォルダは、厳しめレビューを 2 回通す前提で作成しています。

## Review pass 1

Status: failed, fixed before pass 2

Reviewer: `Arendt`

Blocker:

- `REVIEW_LOG.md` が pending のままで、2 回レビュー突破済みとは言えない。

Required fixes applied:

- README に `Public confirmed` と未確認候補の小表を追加した。
- `run_reproduce_q5_pool.ps1` は repo 側スクリプトを実行する前提だと明記し、実行前 SHA256 一致チェックを追加した。
- `target labels は使わない` を、`test target / Public 後推定 label / leakage 由来 label は使わない。通常の train y は使う` に修正した。
- q5 スクリプトコピーと q5 提出 CSV コピーの SHA256 を README / reproduction doc に追加した。

## Review pass 2

Status: passed, follow-up fix applied

Reviewer: `Arendt`

Blocker:

- なし。

Should fix:

- 別環境で再現する際に必要な `data/submissions/` 配置ファイル名が曖昧。
- `Q5_BEST` も起動時に読まれる点を明示した方がよい。

Fix applied:

- `BEST_SCORE_REPRODUCTION.md` に script が読む `BASE_ANCHOR` / `SLOT1` / `SLOT2` / `FIRST_STAGE_BEST` / `SECOND_STAGE_BEST` / `CURRENT_BEST` / `Q5_BEST` の具体ファイル名を追記した。

## Review pass 3

Status: passed

Reviewer: `Arendt`

Blocker:

- なし。

Should fix:

- なし。

Nice to have:

- README の `REVIEW_LOG.md` 説明を、pass 3 まで含めても自然な表現にするとよりよい。
- GitHub 経由で共有するなら `handoff/` を add / commit / push する。

Final pass notes:

- 最高 Public `13.922966797992677`、q5 file、提出 CSV コピー、Public confirmed と未確認候補の分離、再現前提、leakage 注意はいずれもチーム引き継ぎとして十分。
- q5 CSV: 550 rows / 2 cols / no header / negative count 0。
- script SHA256 一致: `81B9508A89104C52F6B38E19E1F6FFDDD8D2967BAD23480F0E268A732C4E143D`。
- q5 CSV SHA256 一致: `FD7EAE6611396E6EFAFCE5602D80882B2C61F194E95883D935F3694B5AC258E4`。
- `run_reproduce_q5_pool.ps1`: syntax ok。

## Review pass 4: internal evaluation guide

Status: failed, fixed

Reviewer: `Nietzsche`

Should fix:

- q5 gate の列対応がなく、`stage5_q5_candidate_manifest_20260609.csv` のどの列を見るかが曖昧。
- pseudo-test simulation の作り方と失敗扱いが薄い。
- detector の実行元スクリプトと出力先が明記されていない。

Fix applied:

- `INTERNAL_EVALUATION_GUIDE.md` を追加し、local CV / OOF / anchor drift / q5 gate / target-domain simulation / detector / Public history の役割分担を整理した。
- q5 gate と `stage5_q5_candidate_manifest_20260609.csv` の列対応表を追加した。
- pseudo-test simulation の手順、出力、失敗扱いを追加した。
- detector 診断の実行元 `scripts/nir_stage5_direction_diagnostics.py` と handoff artifacts / repo outputs の対応を追加した。
- `stage5_direction_summary.json`、`top_uncorrected_detector_species.csv`、`top_uncorrected_detector_clusters.csv` を artifacts に追加した。

## Review pass 5: internal evaluation guide final

Status: passed

Reviewer: `Nietzsche`

Blocker:

- なし。

Should fix:

- なし。

Nice-to-have applied:

- `q5 corr` 表記を `q5 increment corr` に寄せた。
- pseudo-test gap は絶対値だけでなく view 間の相対順位と worst-case gap を見る、と追記した。

Final pass notes:

- local CV / OOF / anchor drift / q5 gate / target-domain simulation / detector / Public history の役割分担は説明済み。
- q5 gate 列対応、pseudo-test 手順、detector 実行元と artifact 対応により、運用の再現性は実用水準。
- `stage5_direction_summary.json` は parse 可能で、`q5_changed_count=22`、`changed_any_count=58`、`never_changed_count=492`、`detector_abs_q5_corr=-0.006701...` がガイド記述と整合。
