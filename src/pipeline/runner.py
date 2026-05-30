"""前処理 → CV評価 → 学習 → 推論 → 提出ファイル生成をつなぐ。"""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import CompetitionConfig, load_config
from pipeline.cv import Fold, make_cv_splits
from pipeline.data import load_raw_data
from pipeline.leaderboard import append_leaderboard_row
from pipeline.metrics import regression_scores
from pipeline.submission import build_submission_frame, write_submission
from pipeline.types import Predictor, Preprocessor, Rows, Vector


@dataclass
class RunResult:
    experiment_name: str
    submission_path: Path
    log_path: Path
    preprocessor: str
    predictor: str
    n_train: int
    n_test: int
    n_features: int
    pred_min: float
    pred_max: float
    pred_mean: float
    cv_strategy: str | None
    primary_metric: str | None
    mean_score: float | None
    std_score: float | None


def run_experiment(
    experiment_name: str,
    preprocessor: Preprocessor,
    predictor: Predictor,
    *,
    config: CompetitionConfig | None = None,
    submit: bool = False,
    memo: str | None = None,
    submission_filename: str | None = None,
    cv_strategy: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> RunResult:
    config = config or load_config()
    raw = load_raw_data(config)

    cv_result = None
    if cv_strategy:
        cv_result = evaluate_cv(
            raw.train,
            preprocessor,
            predictor,
            config=config,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            seed=seed,
        )

    preprocessor.fit(raw.train, target_col=config.target_col, meta_cols=config.meta_cols)
    X_train, y_train = preprocessor.transform_train(raw.train)
    X_test = preprocessor.transform_test(raw.test)

    if hasattr(predictor, "set_train_context"):
        predictor.set_train_context(raw.train, config)
    if hasattr(predictor, "set_group_labels"):
        predictor.set_group_labels([str(row["樹種"]) for row in raw.train])
    predictor.fit(X_train, y_train)
    if hasattr(predictor, "predict_test"):
        predictions = _predict_on_rows(predictor, raw.test, X_test)
    else:
        predictions = predictor.predict(X_test)

    submission_name = submission_filename or f"{experiment_name}.csv"
    submission_path = config.submissions_dir / submission_name
    frame = build_submission_frame(raw.test, predictions, id_col=config.id_col)
    write_submission(frame, submission_path, id_col=config.id_col)

    result = RunResult(
        experiment_name=experiment_name,
        submission_path=submission_path,
        log_path=config.outputs_dir / "logs" / f"{experiment_name}.json",
        preprocessor=preprocessor.name,
        predictor=predictor.name,
        n_train=len(raw.train),
        n_test=len(raw.test),
        n_features=len(X_train[0]) if X_train else 0,
        pred_min=min(predictions),
        pred_max=max(predictions),
        pred_mean=_mean(predictions),
        cv_strategy=cv_strategy,
        primary_metric="rmse" if cv_result else None,
        mean_score=cv_result["mean_score"] if cv_result else None,
        std_score=cv_result["std_score"] if cv_result else None,
    )

    if cv_result:
        oof_path = _write_oof_predictions(config, experiment_name=experiment_name, cv_result=cv_result)
        cv_result["oof_path"] = str(oof_path)

    result.log_path = _write_run_log(config, result=result, cv_result=cv_result, submit=submit, memo=memo)

    if cv_result:
        append_leaderboard_row(config, result=result, cv_result=cv_result, memo=memo or "")

    _print_summary(result)

    if submit:
        from pipeline.signate_submit import submit_to_signate

        submit_memo = memo or f"{experiment_name}: {preprocessor.name} + {predictor.name}"
        submit_to_signate(submission_path, submit_memo, config=config)

    return result


def evaluate_cv(
    train: Rows,
    preprocessor: Preprocessor,
    predictor: Predictor,
    *,
    config: CompetitionConfig,
    cv_strategy: str,
    n_splits: int,
    seed: int,
) -> dict[str, object]:
    folds = make_cv_splits(
        train,
        strategy=cv_strategy,
        group_col="樹種",
        target_col=config.target_col,
        n_splits=n_splits,
        seed=seed,
    )

    y_true_all: Vector = []
    y_pred_all: Vector = []
    fold_scores: list[dict[str, object]] = []
    oof_predictions: list[dict[str, object]] = []

    for fold in folds:
        fold_preprocessor = copy.deepcopy(preprocessor)
        fold_predictor = copy.deepcopy(predictor)
        train_fold = [train[i] for i in fold.train_idx]
        valid_fold = [train[i] for i in fold.valid_idx]

        fold_preprocessor.fit(train_fold, target_col=config.target_col, meta_cols=config.meta_cols)
        X_train, y_train = fold_preprocessor.transform_train(train_fold)
        y_valid = [float(row[config.target_col]) for row in valid_fold]

        if hasattr(fold_predictor, "set_train_context"):
            fold_predictor.set_train_context(train_fold, config)
        if hasattr(fold_predictor, "set_group_labels"):
            fold_predictor.set_group_labels([str(row["樹種"]) for row in train_fold])
        fold_predictor.fit(X_train, y_train)
        if hasattr(fold_predictor, "predict_test"):
            X_valid = fold_preprocessor.transform_test(valid_fold)
            pred_valid = _predict_on_rows(fold_predictor, valid_fold, X_valid)
        else:
            X_valid = fold_preprocessor.transform_test(valid_fold)
            pred_valid = fold_predictor.predict(X_valid)
        scores = regression_scores(y_valid, pred_valid)

        y_true_all.extend(y_valid)
        y_pred_all.extend(pred_valid)
        for row, y_true, y_pred in zip(valid_fold, y_valid, pred_valid):
            error = y_pred - y_true
            oof_predictions.append(
                {
                    "fold_id": fold.fold_id,
                    config.id_col: row[config.id_col],
                    "樹種": row.get("樹種", ""),
                    "species number": row.get("species number", ""),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "error": error,
                    "abs_error": abs(error),
                }
            )
        fold_scores.append(_fold_summary(fold, scores, len(train_fold), len(valid_fold)))

    oof_scores = regression_scores(y_true_all, y_pred_all)
    fold_rmse = [float(item["rmse"]) for item in fold_scores]

    return {
        "strategy": cv_strategy,
        "primary_metric": "rmse",
        "mean_score": _mean(fold_rmse),
        "std_score": _std(fold_rmse),
        "oof_scores": oof_scores,
        "fold_scores": fold_scores,
        "oof_by_species": _summarize_oof_by_species(oof_predictions),
        "oof_predictions": oof_predictions,
    }


def _fold_summary(fold: Fold, scores: dict[str, float], n_train: int, n_valid: int) -> dict[str, object]:
    return {
        "fold_id": fold.fold_id,
        "n_train": n_train,
        "n_valid": n_valid,
        "valid_groups": list(fold.valid_groups),
        **scores,
    }


def _write_run_log(
    config: CompetitionConfig,
    *,
    result: RunResult,
    cv_result: dict[str, object] | None,
    submit: bool,
    memo: str | None,
) -> Path:
    logs_dir = config.outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{result.experiment_name}.json"
    log_cv_result = _compact_cv_result(cv_result)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": result.experiment_name,
        "preprocessor": result.preprocessor,
        "predictor": result.predictor,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_features": result.n_features,
        "prediction_summary": {
            "min": result.pred_min,
            "max": result.pred_max,
            "mean": result.pred_mean,
        },
        "submission_path": str(result.submission_path),
        "submit": submit,
        "memo": memo,
        "cv": log_cv_result,
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return log_path


def _compact_cv_result(cv_result: dict[str, object] | None) -> dict[str, object] | None:
    if cv_result is None:
        return None
    compact = dict(cv_result)
    compact.pop("oof_predictions", None)
    return compact


def _write_oof_predictions(
    config: CompetitionConfig,
    *,
    experiment_name: str,
    cv_result: dict[str, object],
) -> Path:
    rows = cv_result.get("oof_predictions", [])
    path = config.outputs_dir / "oof" / f"{experiment_name}_{cv_result.get('strategy', 'cv')}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = ["fold_id", config.id_col, "樹種", "species number", "y_true", "y_pred", "error", "abs_error"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)  # type: ignore[arg-type]
    return path


def _summarize_oof_by_species(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_species: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_species.setdefault(str(row["樹種"]), []).append(row)

    summary: list[dict[str, object]] = []
    for species, items in sorted(by_species.items()):
        y_true = [float(row["y_true"]) for row in items]
        y_pred = [float(row["y_pred"]) for row in items]
        errors = [float(row["error"]) for row in items]
        summary.append(
            {
                "樹種": species,
                "count": len(items),
                "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
                "mae": sum(abs(error) for error in errors) / len(errors),
                "bias": sum(errors) / len(errors),
                "y_true_mean": _mean(y_true),
                "y_pred_mean": _mean(y_pred),
            }
        )
    summary.sort(key=lambda row: float(row["rmse"]), reverse=True)
    return summary


def _print_summary(result: RunResult) -> None:
    print(f"experiment : {result.experiment_name}")
    print(f"pipeline   : {result.preprocessor} -> {result.predictor}")
    print(f"train/test : {result.n_train} / {result.n_test}  features={result.n_features}")
    if result.mean_score is not None:
        print(
            f"cv         : {result.cv_strategy} {result.primary_metric}="
            f"{result.mean_score:.6f} +/- {result.std_score:.6f}"
        )
    print(f"pred range : {result.pred_min:.4f} .. {result.pred_max:.4f} (mean={result.pred_mean:.4f})")
    print(f"submission : {result.submission_path}")
    print(f"log        : {result.log_path}")


def _predict_on_rows(predictor, rows: Rows, X: Matrix) -> Vector:
    import inspect

    params = list(inspect.signature(predictor.predict_test).parameters)
    if len(params) >= 2:
        return predictor.predict_test(rows, X)
    return predictor.predict_test(rows)


def _mean(values: Vector) -> float:
    return sum(values) / len(values)


def _std(values: Vector) -> float:
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))
