"""複合ローカル指標（Public proxy + 樹種シフト耐性）。"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pipeline.config import CompetitionConfig, load_config
from pipeline.public_score import (
    aggregate_local_scores_by_experiment,
    estimate_public_aligned_score,
    load_known_public_scores,
    summarize_anchor_submission_diff,
)


DEFAULT_COMPOSITE_WEIGHTS = {
    "species_shift": 0.5,
    "public_proxy": 0.3,
    "worst_species": 0.2,
}


def composite_config_path(config: CompetitionConfig | None = None) -> Path:
    config = config or load_config()
    return config.outputs_dir / "composite_weights.json"


def load_composite_weights(config: CompetitionConfig | None = None) -> dict[str, float]:
    path = composite_config_path(config)
    if not path.exists():
        return dict(DEFAULT_COMPOSITE_WEIGHTS)
    data = json.loads(path.read_text(encoding="utf-8"))
    weights = dict(DEFAULT_COMPOSITE_WEIGHTS)
    weights.update({k: float(v) for k, v in data.get("weights", {}).items()})
    return weights


def worst_species_rmse_from_oof_summary(oof_by_species: list[dict[str, object]] | None) -> float | None:
    if not oof_by_species:
        return None
    values = [float(item["rmse"]) for item in oof_by_species if item.get("rmse") is not None]
    return max(values) if values else None


def compute_row_composite_metrics(
    config: CompetitionConfig,
    *,
    experiment_name: str,
    preprocessor: str,
    predictor: str,
    cv_strategy: str,
    mean_score: float,
    worst_species_rmse: float | None,
) -> dict[str, str | float]:
    """1 leaderboard 行分の複合指標を計算する。"""
    local_by_exp = aggregate_local_scores_by_experiment(config)
    local_scores = dict(local_by_exp.get(experiment_name, {}))
    if cv_strategy and cv_strategy not in local_scores:
        local_scores[cv_strategy] = mean_score

    known_public = load_known_public_scores(config)
    public_proxy, _source = estimate_public_aligned_score(
        experiment_name,
        local_scores,
        preprocessor=preprocessor,
        predictor=predictor,
        known_public=known_public,
    )

    species_shift = (
        local_scores.get("group_species")
        or local_scores.get("leave_one_species")
        or local_scores.get("repeated_leave_one_species")
    )
    if species_shift is None and cv_strategy in {
        "group_species",
        "leave_one_species",
        "repeated_leave_one_species",
    }:
        species_shift = mean_score

    worst = worst_species_rmse
    composite = compute_composite_score(
        species_shift_rmse=species_shift,
        public_proxy_rmse=public_proxy,
        worst_species_rmse=worst,
        config=config,
    )

    return {
        "species_shift_rmse": "" if species_shift is None else species_shift,
        "public_proxy_rmse": "" if public_proxy is None else public_proxy,
        "worst_species_rmse": "" if worst is None else worst,
        "composite_score": "" if composite is None else composite,
    }


def compute_composite_score(
    *,
    species_shift_rmse: float | None,
    public_proxy_rmse: float | None,
    worst_species_rmse: float | None,
    config: CompetitionConfig | None = None,
) -> float | None:
    weights = load_composite_weights(config)
    parts: list[tuple[float, float]] = []
    if species_shift_rmse is not None:
        parts.append((species_shift_rmse, weights["species_shift"]))
    if public_proxy_rmse is not None:
        parts.append((public_proxy_rmse, weights["public_proxy"]))
    if worst_species_rmse is not None:
        parts.append((worst_species_rmse, weights["worst_species"]))
    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return sum(v * w for v, w in parts) / total_w


def enrich_experiment_composite(
    config: CompetitionConfig,
    experiment_name: str,
    *,
    preprocessor: str = "",
    predictor: str = "",
) -> dict[str, str | float]:
    """実験単位で3 CV を集約した複合指標（ランキング用）。"""
    local_scores = aggregate_local_scores_by_experiment(config).get(experiment_name, {})
    known_public = load_known_public_scores(config)
    public_proxy, _ = estimate_public_aligned_score(
        experiment_name,
        local_scores,
        preprocessor=preprocessor,
        predictor=predictor,
        known_public=known_public,
    )

    species_shift = (
        local_scores.get("group_species")
        or local_scores.get("leave_one_species")
        or local_scores.get("repeated_leave_one_species")
    )
    worst = _worst_species_from_leaderboard(config, experiment_name)
    composite = compute_composite_score(
        species_shift_rmse=species_shift,
        public_proxy_rmse=public_proxy,
        worst_species_rmse=worst,
        config=config,
    )
    anchor_diff = summarize_anchor_submission_diff(config, experiment_name)

    return {
        "species_shift_rmse": species_shift if species_shift is not None else "",
        "public_proxy_rmse": public_proxy if public_proxy is not None else "",
        "worst_species_rmse": worst if worst is not None else "",
        "composite_score": composite if composite is not None else "",
        "anchor_diff_rmse": anchor_diff.get("diff_rmse") if anchor_diff.get("diff_rmse") is not None else "",
    }


def _worst_species_from_leaderboard(config: CompetitionConfig, experiment_name: str) -> float | None:
    from pipeline.leaderboard import load_leaderboard

    worst: float | None = None
    for row in load_leaderboard(config):
        if row.get("experiment_name") != experiment_name:
            continue
        value = row.get("worst_species_rmse", "")
        if value in {"", None}:
            continue
        v = float(value)
        worst = v if worst is None else max(worst, v)
    return worst
