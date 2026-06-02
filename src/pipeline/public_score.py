"""Publicスコアを手入力で記録し、ローカル評価との差を比較・再校正する。"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import CompetitionConfig, load_config

MIN_PROXY_FIT_SAMPLES = 5
PROXY_FEATURE_KEYS = ("loo", "group", "random")
PROXY_PREDICT_MIN = 5.0
PROXY_PREDICT_MAX = 50.0


def _load_leaderboard(config: CompetitionConfig | None = None) -> list[dict[str, str]]:
    from pipeline.leaderboard import load_leaderboard

    return load_leaderboard(config)

ANCHOR_EXPERIMENT = "candidate_linear_1f_snv_diff1"
PUBLIC_BEST_EXPERIMENT = "candidate_linear_1f"
PUBLIC_BASELINE_SCORES = {
    # Historical measured Public scores kept as fixed anchors even if not present
    # in outputs/public_compare.csv for the current machine.
    "candidate_linear_1f": 17.496,
    "candidate_linear_1f_blend_snv25": 18.558018446661805,
}

SUBMISSION_BLACKLIST_SUBSTRINGS = (
    "group_curve",
    "meta_monotonic",
    "topk_quantile",
    "clip",
    "lowtail",
    "low_tail",
)

# (leave_one_species, group_species, random)
WEIGHTS_ANCHOR_SNV_DIFF1 = (0.55, 0.25, 0.20)
WEIGHTS_SIMPLE_LINEAR = (0.35, 0.35, 0.30)
WEIGHTS_COMPLEX = (0.15, 0.35, 0.50)

PUBLIC_COLUMNS = [
    "timestamp",
    "experiment_name",
    "submission_name",
    "public_score",
    "group_species_rmse",
    "leave_one_species_rmse",
    "random_rmse",
    "best_local_metric",
    "best_local_score",
    "local_public_gap",
    "submission_path",
    "memo",
]

PUBLIC_CANDIDATE_COLUMNS = [
    "timestamp",
    "experiment_name",
    "cv_strategy",
    "primary_metric",
    "mean_score",
    "std_score",
    "oof_rmse",
    "oof_mae",
    "oof_r2",
    "preprocessor",
    "predictor",
    "n_features",
    "submission_path",
    "log_path",
    "memo",
    "public_aligned_score",
    "public_score_source",
    "leave_one_species_rmse",
    "group_species_rmse",
    "random_rmse",
    "anchor_diff_rmse",
    "anchor_changed_rows",
    "blacklisted",
]

# 次回5枠テンプレ（2026-05-29 確定: 保険 + 前処理/補正探索）
SUBMISSION_PLAN_EXPERIMENTS = [
    "candidate_linear_1f",
    "candidate_linear_1f_blend_snv25",
    "candidate_linear_1f_blend_snv25_nn_bias",
    "candidate_linear_1f_nn_bias",
    "group_curve_blend_snv_diff1",
]


def public_compare_path(config: CompetitionConfig | None = None) -> Path:
    config = config or load_config()
    return config.outputs_dir / "public_compare.csv"


def public_proxy_path(config: CompetitionConfig | None = None) -> Path:
    config = config or load_config()
    return config.outputs_dir / "public_proxy.json"


def fit_public_proxy(config: CompetitionConfig | None = None) -> dict[str, object]:
    """public_compare.csv の実測から Public 推定モデルを学習する。"""
    config = config or load_config()
    rows = load_public_comparison(config)
    samples: list[tuple[list[float], float]] = []

    for row in rows:
        public = row.get("public_score", "")
        if public in {"", None}:
            continue
        loo = _float_or_none(row.get("leave_one_species_rmse"))
        group = _float_or_none(row.get("group_species_rmse"))
        random = _float_or_none(row.get("random_rmse"))
        if loo is None and group is None and random is None:
            continue
        features = [
            loo if loo is not None else 0.0,
            group if group is not None else 0.0,
            random if random is not None else 0.0,
        ]
        samples.append((features, float(public)))

    payload: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "min_samples": MIN_PROXY_FIT_SAMPLES,
        "feature_keys": list(PROXY_FEATURE_KEYS),
    }

    if len(samples) < MIN_PROXY_FIT_SAMPLES:
        payload["model"] = "fallback"
        payload["reason"] = "insufficient_measured_public_rows"
    else:
        coeffs = _fit_linear_regression(samples)
        payload["model"] = "linear"
        payload["intercept"] = coeffs[0]
        payload["coefficients"] = dict(zip(PROXY_FEATURE_KEYS, coeffs[1:]))

    path = public_proxy_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_public_proxy_model(config: CompetitionConfig | None = None) -> dict[str, object] | None:
    path = public_proxy_path(config)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model") != "linear":
        return None
    return data


def predict_public_proxy(
    local_scores: dict[str, float],
    *,
    anchor_diff_rmse: float | None = None,
    config: CompetitionConfig | None = None,
) -> float | None:
    model = load_public_proxy_model(config)
    if model is None:
        return None
    intercept = float(model["intercept"])
    coeffs = model["coefficients"]
    values = {
        "loo": (
            local_scores.get("leave_one_species")
            or local_scores.get("moisture_quantile")
            or local_scores.get("repeated_leave_one_species")
        ),
        "group": local_scores.get("group_species"),
        "random": local_scores.get("random"),
    }
    if values["loo"] is None and values["group"] is None and values["random"] is None:
        return None
    score = intercept
    for key in PROXY_FEATURE_KEYS:
        val = values[key]
        if val is None:
            val = 0.0
        score += float(coeffs[key]) * float(val)
    if not (PROXY_PREDICT_MIN <= score <= PROXY_PREDICT_MAX):
        return None
    return score


def load_known_public_scores(config: CompetitionConfig | None = None) -> dict[str, float]:
    scores: dict[str, float] = dict(PUBLIC_BASELINE_SCORES)
    for row in load_public_comparison(config):
        name = row.get("experiment_name", "")
        public = row.get("public_score", "")
        if name and public not in {"", None}:
            scores[name] = float(public)
    return scores


def aggregate_local_scores_by_experiment(config: CompetitionConfig | None = None) -> dict[str, dict[str, float]]:
    config = config or load_config()
    out: dict[str, dict[str, float]] = {}
    for row in _load_leaderboard(config):
        name = row.get("experiment_name", "")
        strategy = row.get("cv_strategy", "")
        score = row.get("mean_score", "")
        if not name or not strategy or not score:
            continue
        out.setdefault(name, {})[strategy] = float(score)
    return out


def is_blacklisted(*, experiment_name: str, predictor: str = "") -> bool:
    text = f"{experiment_name} {predictor}".lower()
    return any(token in text for token in SUBMISSION_BLACKLIST_SUBSTRINGS)


def _weighted_score(local_scores: dict[str, float], weights: tuple[float, float, float]) -> float | None:
    keys = ("leave_one_species", "group_species", "random")
    parts: list[tuple[float, float]] = []
    for key, weight in zip(keys, weights):
        if key in local_scores:
            parts.append((local_scores[key], weight))
    if not parts:
        return None
    total_weight = sum(weight for _, weight in parts)
    return sum(score * weight for score, weight in parts) / total_weight


def _median_score(local_scores: dict[str, float]) -> float | None:
    values = [local_scores[k] for k in ("leave_one_species", "group_species", "random") if k in local_scores]
    if not values:
        return None
    values.sort()
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def estimate_public_aligned_score(
    experiment_name: str,
    local_scores: dict[str, float],
    *,
    preprocessor: str = "",
    predictor: str = "",
    known_public: dict[str, float] | None = None,
    config: CompetitionConfig | None = None,
) -> tuple[float | None, str]:
    known_public = known_public if known_public is not None else {}
    if experiment_name in known_public:
        return known_public[experiment_name], "measured"

    config = config or load_config()
    anchor_diff = summarize_anchor_submission_diff(config, experiment_name).get("diff_rmse")
    feedback_score = _estimate_from_public_feedback(experiment_name, known_public)
    if feedback_score is not None:
        return feedback_score, "public_feedback_quadratic"

    fitted = predict_public_proxy(
        local_scores,
        anchor_diff_rmse=float(anchor_diff) if anchor_diff is not None else None,
        config=config,
    )
    if fitted is not None:
        return fitted, "fitted"

    if (
        predictor
        in (
            "single_feature_linear",
            "single_feature_loo_consensus_wavelength",
            "single_feature_loo_rmse_wavelength",
            "single_feature_second_wavelength",
            "single_feature_linear_oof_intercept",
            "single_feature_stable_wavelength",
        )
        and preprocessor == "spectral_snv_diff1"
    ):
        score = _weighted_score(local_scores, WEIGHTS_ANCHOR_SNV_DIFF1)
        return (score, "estimated_anchor_snv_diff1") if score is not None else (None, "missing")

    if predictor == "single_feature_linear":
        score = _weighted_score(local_scores, WEIGHTS_SIMPLE_LINEAR)
        return (score, "estimated_simple_linear") if score is not None else (None, "missing")

    complex_markers = (
        "group_curve",
        "meta_monotonic",
        "topk_",
        "topk6",
        "topk8",
        "topk12",
        "topk24",
        "pls_",
        "pls4",
        "pls6",
        "hgbdt",
    )
    text = f"{experiment_name} {predictor} {preprocessor}".lower()
    if any(marker in text for marker in complex_markers):
        score = _weighted_score(local_scores, WEIGHTS_COMPLEX)
        return (score, "estimated_complex") if score is not None else (None, "missing")

    score = _median_score(local_scores)
    return (score, "estimated_median") if score is not None else (None, "missing")


def _estimate_from_public_feedback(experiment_name: str, known_public: dict[str, float]) -> float | None:
    weight = _anti_nn_bias_weight(experiment_name)
    if weight is None:
        return None

    baseline = known_public.get("candidate_linear_1f")
    nn_bias = known_public.get("candidate_linear_1f_nn_bias")
    if baseline is None or nn_bias is None:
        return None

    points: list[tuple[float, float]] = [(0.0, baseline), (1.0, nn_bias)]
    for name, score in known_public.items():
        known_weight = _anti_nn_bias_weight(name)
        if known_weight is not None:
            points.append((known_weight, score))
    if len(points) < 3:
        return None

    # Public feedback is one-dimensional for these candidates:
    # pred(w) = baseline + w * (nn_bias - baseline).
    # Since RMSE^2 along a linear prediction path is quadratic in w, fit
    # score^2 = a*w^2 + b*w + c using all measured Public points.
    a, b, c = _fit_quadratic_score_squared(points)
    estimate_sq = a * weight * weight + b * weight + c
    if estimate_sq <= 0:
        return None
    return math.sqrt(estimate_sq)


def _anti_nn_bias_weight(experiment_name: str) -> float | None:
    prefix = "candidate_linear_1f_anti_nn_bias_w"
    if not experiment_name.startswith(prefix):
        return None
    suffix = experiment_name[len(prefix):]
    if not suffix.startswith("m"):
        return None
    digits = suffix[1:]
    if not digits.isdigit():
        return None
    return -float(digits) / 100.0


def _fit_quadratic_score_squared(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares fit of score^2 = a*w^2 + b*w + c."""
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for weight, score in points:
        row = [weight * weight, weight, 1.0]
        y = score * score
        for i, xi in enumerate(row):
            xty[i] += xi * y
            for j, xj in enumerate(row):
                xtx[i][j] += xi * xj
    coeffs = _solve_linear_system(xtx, xty)
    return coeffs[0], coeffs[1], coeffs[2]


def build_public_rank_rows(config: CompetitionConfig | None = None) -> list[dict[str, str]]:
    config = config or load_config()
    known_public = load_known_public_scores(config)
    local_by_exp = aggregate_local_scores_by_experiment(config)

    meta_by_exp: dict[str, dict[str, str]] = {}
    for row in _load_leaderboard(config):
        name = row.get("experiment_name", "")
        if name and name not in meta_by_exp:
            meta_by_exp[name] = row

    enriched: list[dict[str, object]] = []
    for experiment_name, local_scores in local_by_exp.items():
        meta = meta_by_exp.get(experiment_name, {})
        preprocessor = str(meta.get("preprocessor", ""))
        predictor = str(meta.get("predictor", ""))
        aligned, source = estimate_public_aligned_score(
            experiment_name,
            local_scores,
            preprocessor=preprocessor,
            predictor=predictor,
            known_public=known_public,
            config=config,
        )
        loo = local_scores.get("leave_one_species")
        anchor_diff = summarize_anchor_submission_diff(config, experiment_name)
        enriched.append(
            {
                "experiment_name": experiment_name,
                "preprocessor": preprocessor,
                "predictor": predictor,
                "public_aligned_score": aligned,
                "public_score_source": source,
                "leave_one_species_rmse": loo,
                "group_species_rmse": local_scores.get("group_species"),
                "random_rmse": local_scores.get("random"),
                "anchor_diff_rmse": anchor_diff.get("diff_rmse"),
                "anchor_changed_rows": anchor_diff.get("changed_rows"),
                "blacklisted": is_blacklisted(experiment_name=experiment_name, predictor=predictor),
                "meta": meta,
            }
        )

    enriched.sort(
        key=lambda row: (
            1 if row["blacklisted"] else 0,
            float("inf") if row["public_aligned_score"] is None else float(row["public_aligned_score"]),
            float("inf") if row["leave_one_species_rmse"] is None else float(row["leave_one_species_rmse"]),
        )
    )

    rows: list[dict[str, str]] = []
    for item in enriched:
        meta = item["meta"]
        row = {key: str(meta.get(key, "")) for key in PUBLIC_CANDIDATE_COLUMNS if key in meta}
        row["experiment_name"] = str(item["experiment_name"])
        row["preprocessor"] = str(item["preprocessor"])
        row["predictor"] = str(item["predictor"])
        row["public_aligned_score"] = "" if item["public_aligned_score"] is None else f"{item['public_aligned_score']:.6f}"
        row["public_score_source"] = str(item["public_score_source"])
        row["leave_one_species_rmse"] = "" if item["leave_one_species_rmse"] is None else f"{item['leave_one_species_rmse']:.6f}"
        row["group_species_rmse"] = "" if item["group_species_rmse"] is None else f"{item['group_species_rmse']:.6f}"
        row["random_rmse"] = "" if item["random_rmse"] is None else f"{item['random_rmse']:.6f}"
        row["anchor_diff_rmse"] = "" if item["anchor_diff_rmse"] is None else f"{item['anchor_diff_rmse']:.6f}"
        row["anchor_changed_rows"] = str(item["anchor_changed_rows"])
        row["blacklisted"] = "yes" if item["blacklisted"] else "no"
        rows.append(row)
    return rows


def summarize_anchor_submission_diff(
    config: CompetitionConfig | None = None,
    experiment_name: str = "",
    *,
    anchor_experiment: str = ANCHOR_EXPERIMENT,
) -> dict[str, object]:
    config = config or load_config()
    anchor_path = _submission_path_for_experiment(config, anchor_experiment)
    candidate_path = _submission_path_for_experiment(config, experiment_name)
    if not anchor_path.exists() or not candidate_path.exists():
        return {"diff_rmse": None, "changed_rows": None, "anchor_path": str(anchor_path), "candidate_path": str(candidate_path)}

    anchor_rows = _read_submission(anchor_path)
    candidate_rows = _read_submission(candidate_path)
    if len(anchor_rows) != len(candidate_rows):
        return {
            "diff_rmse": None,
            "changed_rows": None,
            "error": "row_count_mismatch",
            "anchor_path": str(anchor_path),
            "candidate_path": str(candidate_path),
        }

    diffs: list[float] = []
    changed = 0
    for (_, ay), (_, cy) in zip(anchor_rows, candidate_rows):
        diff = cy - ay
        diffs.append(diff)
        if abs(diff) > 1e-9:
            changed += 1
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else None
    return {
        "diff_rmse": rmse,
        "changed_rows": changed,
        "anchor_path": str(anchor_path),
        "candidate_path": str(candidate_path),
    }


def record_public_score(
    experiment_name: str,
    public_score: float,
    *,
    config: CompetitionConfig | None = None,
    memo: str = "",
) -> Path:
    config = config or load_config()
    local_scores = _latest_local_scores(config, experiment_name)
    best_metric, best_score = _closest_metric(local_scores, public_score)
    submission_path = _latest_submission_path(config, experiment_name)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "submission_name": Path(submission_path).name if submission_path else "",
        "public_score": public_score,
        "group_species_rmse": local_scores.get("group_species", ""),
        "leave_one_species_rmse": local_scores.get("leave_one_species", ""),
        "random_rmse": local_scores.get("random", ""),
        "best_local_metric": best_metric,
        "best_local_score": best_score if best_score is not None else "",
        "local_public_gap": abs(best_score - public_score) if best_score is not None else "",
        "submission_path": submission_path,
        "memo": memo,
    }

    path = public_compare_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    fit_public_proxy(config)
    return path


def load_public_comparison(config: CompetitionConfig | None = None) -> list[dict[str, str]]:
    path = public_compare_path(config)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def print_public_comparison(config: CompetitionConfig | None = None) -> None:
    rows = load_public_comparison(config)
    if not rows:
        print("public_compare.csv is empty. Use --record-public first.")
        return

    header = f"{'experiment':<26} {'public':>10} {'group':>10} {'loo':>10} {'random':>10} {'best':<18} {'gap':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['experiment_name']:<26} "
            f"{_fmt(row['public_score']):>10} "
            f"{_fmt(row['group_species_rmse']):>10} "
            f"{_fmt(row['leave_one_species_rmse']):>10} "
            f"{_fmt(row['random_rmse']):>10} "
            f"{row['best_local_metric']:<18} "
            f"{_fmt(row['local_public_gap']):>10}"
        )


def print_public_ranked_rows(config: CompetitionConfig | None = None, *, limit: int | None = None) -> None:
    rows = build_public_rank_rows(config)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        print("no experiments with local scores.")
        return

    header = (
        f"{'rank':>4}  {'pub_aligned':>12}  {'loo':>10}  {'anchor_d':>10}  "
        f"{'chg':>5}  {'blk':>4}  {'source':<22}  experiment"
    )
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, start=1):
        print(
            f"{i:>4}  "
            f"{row['public_aligned_score']:>12}  "
            f"{row['leave_one_species_rmse']:>10}  "
            f"{row['anchor_diff_rmse']:>10}  "
            f"{row['anchor_changed_rows']:>5}  "
            f"{row['blacklisted']:>4}  "
            f"{row['public_score_source']:<22}  "
            f"{row['experiment_name']}"
        )


def _submission_path_for_experiment(config: CompetitionConfig, experiment_name: str) -> Path:
    local = config.submissions_dir / f"{experiment_name}.csv"
    if local.exists():
        return local
    path = _latest_submission_path(config, experiment_name)
    if path and Path(path).exists():
        return Path(path)
    return local


def _read_submission(path: Path) -> list[tuple[str, float]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [(row[0], float(row[1])) for row in csv.reader(f)]


def _latest_local_scores(config: CompetitionConfig, experiment_name: str) -> dict[str, float]:
    rows = [
        row for row in _load_leaderboard(config)
        if row.get("experiment_name") == experiment_name and row.get("mean_score")
    ]
    scores: dict[str, float] = {}
    for row in rows:
        scores[row["cv_strategy"]] = float(row["mean_score"])
    return scores


def _latest_submission_path(config: CompetitionConfig, experiment_name: str) -> str:
    local = config.submissions_dir / f"{experiment_name}.csv"
    if local.exists():
        return str(local)
    rows = [
        row for row in _load_leaderboard(config)
        if row.get("experiment_name") == experiment_name and row.get("submission_path")
    ]
    for row in reversed(rows):
        path = Path(str(row["submission_path"]))
        if path.exists():
            return str(path)
    return str(local) if local.exists() else ""


def _closest_metric(local_scores: dict[str, float], public_score: float) -> tuple[str, float | None]:
    if not local_scores:
        return "", None
    metric, score = min(local_scores.items(), key=lambda item: abs(item[1] - public_score))
    return metric, score


def _fmt(value: object) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.5f}"


def _float_or_none(value: object) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def _fit_linear_regression(samples: list[tuple[list[float], float]]) -> list[float]:
    """最小二乗: [intercept, coef_1, ...]。"""
    n_features = len(samples[0][0])
    # 拡張特徴 [1, x1, x2, ...]
    xtx = [[0.0] * (n_features + 1) for _ in range(n_features + 1)]
    xty = [0.0] * (n_features + 1)
    for features, y in samples:
        row = [1.0, *features]
        for i, xi in enumerate(row):
            xty[i] += xi * y
            for j, xj in enumerate(row):
                xtx[i][j] += xi * xj
    return _solve_linear_system(xtx, xty)


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """ガウス消去（小規模用）。"""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[pivot][col]):
                pivot = row
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        if abs(pivot_val) < 1e-12:
            continue
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot_val
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    solution = [0.0] * n
    for col in reversed(range(n)):
        if abs(aug[col][col]) < 1e-12:
            solution[col] = 0.0
            continue
        solution[col] = (aug[col][n] - sum(aug[col][j] * solution[j] for j in range(col + 1, n))) / aug[col][col]
    return solution
