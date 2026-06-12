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
