"""ローカル実験ランキングの保存と表示。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from pipeline.composite_score import compute_row_composite_metrics
from pipeline.config import CompetitionConfig, load_config
from pipeline.public_score import (
    PUBLIC_CANDIDATE_COLUMNS,
    SUBMISSION_PLAN_EXPERIMENTS,
    build_public_rank_rows,
    summarize_anchor_submission_diff,
)


LEADERBOARD_COLUMNS = [
    "timestamp",
    "experiment_name",
    "cv_strategy",
    "primary_metric",
    "mean_score",
    "std_score",
    "oof_rmse",
    "oof_mae",
    "oof_r2",
    "species_shift_rmse",
    "public_proxy_rmse",
    "worst_species_rmse",
    "composite_score",
    "preprocessor",
    "predictor",
    "n_features",
    "submission_path",
    "log_path",
    "memo",
]


def leaderboard_path(config: CompetitionConfig | None = None) -> Path:
    config = config or load_config()
    return config.outputs_dir / "leaderboard.csv"


def append_leaderboard_row(
    config: CompetitionConfig,
    *,
    result,
    cv_result: dict[str, object],
    memo: str,
) -> None:
    path = leaderboard_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    oof_scores = cv_result.get("oof_scores", {})
    cv_strategy = str(cv_result.get("strategy", ""))
    mean_score = float(cv_result["mean_score"]) if cv_result.get("mean_score") is not None else 0.0
    worst_species = _worst_species_from_cv(cv_result)
    composite_fields = compute_row_composite_metrics(
        config,
        experiment_name=result.experiment_name,
        preprocessor=result.preprocessor,
        predictor=result.predictor,
        cv_strategy=cv_strategy,
        mean_score=mean_score,
        worst_species_rmse=worst_species,
    )
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": result.experiment_name,
        "cv_strategy": cv_strategy,
        "primary_metric": cv_result.get("primary_metric", "rmse"),
        "mean_score": cv_result.get("mean_score", ""),
        "std_score": cv_result.get("std_score", ""),
        "oof_rmse": oof_scores.get("rmse", ""),
        "oof_mae": oof_scores.get("mae", ""),
        "oof_r2": oof_scores.get("r2", ""),
        **{k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in composite_fields.items()},
        "preprocessor": result.preprocessor,
        "predictor": result.predictor,
        "n_features": result.n_features,
        "submission_path": str(result.submission_path),
        "log_path": str(result.log_path),
        "memo": memo,
    }

    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_leaderboard(config: CompetitionConfig | None = None) -> list[dict[str, str]]:
    path = leaderboard_path(config)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ranked_rows(config: CompetitionConfig | None = None, *, limit: int | None = None) -> list[dict[str, str]]:
    rows = load_leaderboard(config)
    rows = [row for row in rows if row.get("mean_score")]
    rows.sort(key=_row_sort_key)
    if limit is not None:
        return rows[:limit]
    return rows


def ranked_best_rows(config: CompetitionConfig | None = None, *, limit: int | None = None) -> list[dict[str, str]]:
    """実験ごとに最良行を1件（composite_score 優先）。"""
    rows = load_leaderboard(config)
    rows = [row for row in rows if row.get("mean_score")]
    best_by_experiment: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row["experiment_name"]
        current = best_by_experiment.get(key)
        if current is None or _row_sort_key(row) < _row_sort_key(current):
            best_by_experiment[key] = row
    best_rows = sorted(best_by_experiment.values(), key=_row_sort_key)
    if limit is not None:
        return best_rows[:limit]
    return best_rows


def _row_sort_key(row: dict[str, str]) -> tuple[int, float, float]:
    composite = row.get("composite_score", "")
    if composite not in {"", None}:
        return (0, float(composite), float(row.get("mean_score", "inf") or "inf"))
    return (1, float(row.get("mean_score", "inf") or "inf"), 0.0)


def _worst_species_from_cv(cv_result: dict[str, object]) -> float | None:
    summary = cv_result.get("oof_by_species")
    if not isinstance(summary, list):
        return None
    values = [float(item["rmse"]) for item in summary if item.get("rmse") is not None]
    return max(values) if values else None


def write_daily_candidates(config: CompetitionConfig | None = None, *, limit: int = 5) -> Path:
    """次回5枠テンプレ（保険2 + 単一変更探索）を daily_candidates.csv に書く。"""
    config = config or load_config()
    plan_names = SUBMISSION_PLAN_EXPERIMENTS[:limit]
    rows = _rows_for_experiment_plan(config, plan_names)
    path = config.outputs_dir / "daily_candidates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLIC_CANDIDATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


AUTO_SUBMIT_PREDICTORS = frozenset(
    {
        "single_feature_linear",
        "single_feature_stable_wavelength",
        "single_feature_loo_consensus_wavelength",
        "single_feature_loo_rmse_wavelength",
        "single_feature_second_wavelength",
        "dual_prep_anchor_raw_blend",
        "single_feature_top3_median",
        "single_feature_linear_oof_intercept",
        "dual_stable_wavelength_ridge",
    },
)


def write_public_ranked_candidates(config: CompetitionConfig | None = None, *, limit: int = 5) -> Path:
    """Public校正スコア順（ブラックリスト除外・単純1特徴系のみ）の提出候補を保存する。"""
    config = config or load_config()
    rows = [
        row
        for row in build_public_rank_rows(config)
        if row.get("blacklisted") != "yes" and row.get("predictor") in AUTO_SUBMIT_PREDICTORS
    ][:limit]
    path = config.outputs_dir / "daily_candidates_public_ranked.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLIC_CANDIDATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _rows_for_experiment_plan(config: CompetitionConfig, experiment_names: list[str]) -> list[dict[str, str]]:
    ranked = {row["experiment_name"]: row for row in build_public_rank_rows(config)}
    slot_memos = [
        "slot1: Private必須・Public保険 (17.496)",
        "slot2: 今日の探索#1 raw75%+SNV25%",
        "slot3: 今日の探索#2 blend_snv25 + NN bias補正",
        "slot4: 今日の探索#3 NN train残差補正のみ",
        "slot5: Private第2・Public 21.17 保険",
    ]
    rows: list[dict[str, str]] = []
    for i, name in enumerate(experiment_names):
        if name in ranked:
            row = dict(ranked[name])
        else:
            row = {"experiment_name": name}
        row["memo"] = slot_memos[i] if i < len(slot_memos) else row.get("memo", "")
        anchor_diff = summarize_anchor_submission_diff(config, name)
        if anchor_diff.get("diff_rmse") is not None:
            row["anchor_diff_rmse"] = f"{anchor_diff['diff_rmse']:.6f}"
        if anchor_diff.get("changed_rows") is not None:
            row["anchor_changed_rows"] = str(anchor_diff["changed_rows"])
        rows.append(row)
    return rows
