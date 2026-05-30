#!/usr/bin/env python3
"""
実験の実行エントリポイント。

例:
  python run.py --list
  python run.py baseline_mean --cv group_species
  python run.py baseline_mean --cv group_species --submit
  python run.py --rank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.registry import EXPERIMENTS, resolve_experiment  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.leaderboard import (  # noqa: E402
    ranked_best_rows,
    ranked_rows,
    write_daily_candidates,
    write_public_ranked_candidates,
)
from pipeline.public_score import (  # noqa: E402
    ANCHOR_EXPERIMENT,
    PUBLIC_BEST_EXPERIMENT,
    fit_public_proxy,
    print_public_comparison,
    print_public_ranked_rows,
    record_public_score,
    summarize_anchor_submission_diff,
)
from pipeline.runner import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="近赤外研究会コンペ: 学習・評価・提出パイプライン")
    parser.add_argument(
        "experiment",
        nargs="?",
        help=f"実験名 ({', '.join(EXPERIMENTS)})",
    )
    parser.add_argument("--submit", action="store_true", help="SIGNATE に提出する")
    parser.add_argument("--memo", default=None, help="提出メモ（省略時は実験定義の memo）")
    parser.add_argument("--list", action="store_true", help="登録済み実験を表示")
    parser.add_argument(
        "--cv",
        choices=[
            "group_species",
            "leave_one_species",
            "repeated_leave_one_species",
            "moisture_quantile",
            "random",
        ],
        default=None,
        help="ローカル評価のCV戦略",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="CV分割数（group/random用）")
    parser.add_argument("--seed", type=int, default=42, help="random CV用seed")
    parser.add_argument("--rank", action="store_true", help="leaderboard.csv を composite_score 順に表示")
    parser.add_argument(
        "--calibrate-public",
        action="store_true",
        help="public_compare.csv から Public proxy モデルを再学習",
    )
    parser.add_argument("--top5", action="store_true", help="提出候補上位5件を daily_candidates.csv に保存")
    parser.add_argument(
        "--rank-public",
        action="store_true",
        help="Public校正スコア順に表示（ブラックリストは末尾）",
    )
    parser.add_argument(
        "--top5-public",
        action="store_true",
        help="Public校正スコア順の提出候補5件を保存（ブラックリスト除外）",
    )
    parser.add_argument(
        "--anchor-diff",
        metavar="EXPERIMENT",
        help=f"開発アンカー({ANCHOR_EXPERIMENT})との予測差分サマリを表示",
    )
    parser.add_argument(
        "--public-diff",
        metavar="EXPERIMENT",
        help=f"Publicベスト({PUBLIC_BEST_EXPERIMENT})との予測差分サマリを表示",
    )
    parser.add_argument(
        "--record-public",
        nargs=2,
        metavar=("EXPERIMENT", "SCORE"),
        help="Publicスコアを記録する: --record-public experiment score",
    )
    parser.add_argument("--compare-public", action="store_true", help="Publicスコアとローカル評価を比較表示")
    args = parser.parse_args()

    config = load_config()

    if args.list:
        for name, spec in EXPERIMENTS.items():
            print(f"{name:20s}  {spec.preprocessor} + {spec.predictor}")
        return

    if args.rank:
        print_ranked_rows(config, limit=None)
        return

    if args.top5:
        path = write_daily_candidates(config, limit=5)
        print(f"saved submission plan (5 slots): {path}")
        print_public_ranked_rows(config, limit=10)
        return

    if args.rank_public:
        print_public_ranked_rows(config, limit=None)
        return

    if args.top5_public:
        path = write_public_ranked_candidates(config, limit=5)
        print(f"saved (simple 1-feature only, blacklist excluded): {path}")
        _print_csv_rows(path)
        return

    if args.public_diff:
        summary = summarize_anchor_submission_diff(
            config,
            args.public_diff,
            anchor_experiment=PUBLIC_BEST_EXPERIMENT,
        )
        print(f"public_best: {PUBLIC_BEST_EXPERIMENT}")
        print(f"candidate: {args.public_diff}")
        print(f"public_best_path: {summary.get('anchor_path')}")
        print(f"candidate_path: {summary.get('candidate_path')}")
        if summary.get("error"):
            print(f"error: {summary['error']}")
            return
        if summary.get("diff_rmse") is None:
            print("diff: unavailable (submission CSV missing or row mismatch)")
            return
        print(f"diff_rmse: {summary['diff_rmse']:.6f}")
        print(f"changed_rows: {summary['changed_rows']} / 550")
        return

    if args.anchor_diff:
        summary = summarize_anchor_submission_diff(config, args.anchor_diff)
        print(f"anchor: {ANCHOR_EXPERIMENT}")
        print(f"candidate: {args.anchor_diff}")
        print(f"anchor_path: {summary.get('anchor_path')}")
        print(f"candidate_path: {summary.get('candidate_path')}")
        if summary.get("error"):
            print(f"error: {summary['error']}")
            return
        if summary.get("diff_rmse") is None:
            print("diff: unavailable (submission CSV missing or row mismatch)")
            return
        print(f"diff_rmse: {summary['diff_rmse']:.6f}")
        print(f"changed_rows: {summary['changed_rows']} / 550")
        return

    if args.record_public:
        experiment_name, public_score = args.record_public
        path = record_public_score(experiment_name, float(public_score), config=config, memo=args.memo or "")
        print(f"saved: {path}")
        print_public_comparison(config)
        return

    if args.compare_public:
        print_public_comparison(config)
        return

    if args.calibrate_public:
        payload = fit_public_proxy(config)
        print(f"public proxy model: {payload.get('model')} (n_samples={payload.get('n_samples')})")
        if payload.get("model") == "linear":
            print(f"  intercept={payload.get('intercept')}")
            print(f"  coefficients={payload.get('coefficients')}")
        return

    if not args.experiment:
        parser.error("experiment を指定するか --list / --rank / --top5 を使ってください")

    preprocessor, predictor, default_memo = resolve_experiment(args.experiment)
    run_experiment(
        args.experiment,
        preprocessor,
        predictor,
        config=config,
        submit=args.submit,
        memo=args.memo or default_memo,
        cv_strategy=args.cv,
        n_splits=args.n_splits,
        seed=args.seed,
    )


def _print_csv_rows(path: Path) -> None:
    import csv

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("(empty)")
        return
    header = (
        f"{'rank':>4}  {'pub_aligned':>12}  {'loo':>10}  {'anchor_d':>10}  "
        f"{'chg':>5}  {'source':<22}  experiment"
    )
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, start=1):
        print(
            f"{i:>4}  "
            f"{row.get('public_aligned_score', ''):>12}  "
            f"{row.get('leave_one_species_rmse', ''):>10}  "
            f"{row.get('anchor_diff_rmse', ''):>10}  "
            f"{row.get('anchor_changed_rows', ''):>5}  "
            f"{row.get('public_score_source', ''):<22}  "
            f"{row.get('experiment_name', '')}"
        )


def print_ranked_rows(config, *, limit: int | None, best_only: bool = False) -> None:
    rows = ranked_best_rows(config, limit=limit) if best_only else ranked_rows(config, limit=limit)
    if not rows:
        print("leaderboard is empty. Run an experiment with --cv first.")
        return

    header = (
        f"{'rank':>4}  {'composite':>10}  {'loo_shift':>10}  {'proxy':>10}  "
        f"{'mean':>10}  {'experiment':<32}  {'cv':<22}"
    )
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, start=1):
        composite = row.get("composite_score", "") or ""
        loo_shift = row.get("species_shift_rmse", "") or ""
        proxy = row.get("public_proxy_rmse", "") or ""
        print(
            f"{i:>4}  "
            f"{composite:>10}  "
            f"{loo_shift:>10}  "
            f"{proxy:>10}  "
            f"{float(row['mean_score']):>10.6f}  "
            f"{row['experiment_name']:<32}  "
            f"{row['cv_strategy']:<22}"
        )


if __name__ == "__main__":
    main()
