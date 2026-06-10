#!/usr/bin/env python3
"""Stage-1.5 hard-case detector diagnostics.

Before building a second-stage predictor, test whether we can identify samples
that are likely to hurt RMSE using only features available at test time.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nir_base_shape_aug_guard_search as bsa
import nir_gated_local_correction_search as glc
import nir_nonlinear_latent_guard_search as nl
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_hard_case_detector"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_baseaug_shape14_sc0p5_p18_a3500_affs0p12.csv"


@dataclass(frozen=True)
class BranchDiag:
    name: str
    local_target: str
    k: int
    alpha: float


def main() -> None:
    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building current anchor OOF", flush=True)
    anchor_spec = bsa.AugSpec(
        name="current_shape_anchor",
        pca_components=18,
        alpha=3500.0,
        shape_set="shape14",
        shape_scale=0.50,
        affine_shrink=0.12,
        affine_clip=0.40,
        mean_center=True,
    )
    anchor_oof = bsa.make_nested_affine_oof(anchor_spec, data["X_train"], data["y"], data["groups"])
    residual_abs = np.abs(data["y"] - anchor_oof)
    print(f"anchor_oof_rmse={rmse(data['y'], anchor_oof):.6f}", flush=True)

    branch_specs = [
        BranchDiag("local_raw_k40_a1", "raw", 40, 1.0),
        BranchDiag("local_raw_k80_a1", "raw", 80, 1.0),
        BranchDiag("local_yj_k40_a10", "yj", 40, 10.0),
        BranchDiag("local_yj_k80_a10", "yj", 80, 10.0),
    ]

    feature_frames_train: list[pd.DataFrame] = [base_features(anchor_oof, residual_abs, prefix="anchor")]
    feature_frames_test: list[pd.DataFrame] = [base_features(anchor_test, None, prefix="anchor")]
    branch_rows: list[dict[str, object]] = []

    for branch in branch_specs:
        print(f"branch {branch.name}", flush=True)
        spec = glc.LocalSpec(
            name=branch.name,
            local_target=branch.local_target,
            k=branch.k,
            alpha=branch.alpha,
        )
        local_oof, iso_oof = glc.make_local_oof(spec, data)
        local_test, iso_test = glc.fit_predict_local(spec, data["X_train"], data["y"], data["X_test"])
        signal_oof = local_oof - anchor_oof
        signal_test = local_test - anchor_test
        branch_rows.append(
            {
                "branch": branch.name,
                "local_oof_rmse": rmse(data["y"], local_oof),
                "delta_rmse_vs_anchor": rmse(data["y"], local_oof) - rmse(data["y"], anchor_oof),
                "signal_residual_corr": nl.safe_corr(signal_oof, data["y"] - anchor_oof),
                "abs_signal_residual_abs_corr": nl.safe_corr(np.abs(signal_oof), residual_abs),
                "test_signal_abs_mean": float(np.mean(np.abs(signal_test))),
                "test_iso_mean": float(np.mean(iso_test)),
            }
        )
        feature_frames_train.append(branch_features(branch.name, signal_oof, iso_oof, local_oof))
        feature_frames_test.append(branch_features(branch.name, signal_test, iso_test, local_test))

    X_train_df = pd.concat(feature_frames_train, axis=1)
    X_test_df = pd.concat(feature_frames_test, axis=1)
    X_train_df = add_rank_features(X_train_df)
    X_test_df = add_rank_features(X_test_df)

    rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    for hard_q in [0.80, 0.85, 0.90]:
        y_hard = residual_abs >= np.quantile(residual_abs, hard_q)
        for model_name in ["logreg", "rf"]:
            oof_score = detector_oof_score(model_name, X_train_df, y_hard, data["groups"])
            full_score = detector_full_score(model_name, X_train_df, y_hard, X_test_df)
            rows.extend(evaluate_scores(model_name, hard_q, oof_score, y_hard, residual_abs, data["groups"]))
            score_frames.append(test_score_frame(model_name, hard_q, full_score, anchor_test, test_species, data["test_ids"]))

    branch_df = pd.DataFrame(branch_rows)
    summary_df = pd.DataFrame(rows).sort_values(["hard_q", "model", "k_frac"])
    test_scores_df = pd.concat(score_frames, ignore_index=True)

    branch_df.to_csv(out_dir / "branch_diagnostics.csv", index=False)
    summary_df.to_csv(out_dir / "detector_summary.csv", index=False)
    test_scores_df.to_csv(out_dir / "test_hard_scores.csv", index=False)
    with (out_dir / "detector_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "branch_diagnostics": branch_rows,
                "detector_summary": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nBranch diagnostics:")
    print(branch_df.to_string(index=False))
    print("\nDetector summary:")
    print(summary_df.to_string(index=False))
    print("\nTop test hard scores:")
    for (model_name, hard_q), group in test_scores_df.groupby(["model", "hard_q"]):
        top = group.sort_values("hard_score", ascending=False).head(15)
        print(f"\n{model_name} hard_q={hard_q}")
        print(top[["sample_number", "species_number", "anchor_pred", "hard_score"]].to_string(index=False))
    print(f"saved hard-case detector diagnostics: {out_dir}")


def base_features(pred: np.ndarray, residual_abs: np.ndarray | None, *, prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            f"{prefix}_pred": pred,
            f"{prefix}_pred_abs": np.abs(pred),
            f"{prefix}_pred_rank": pd.Series(pred).rank(pct=True).to_numpy(),
        }
    )
    return df


def branch_features(name: str, signal: np.ndarray, iso: np.ndarray, local_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{name}_local_pred": local_pred,
            f"{name}_signal": signal,
            f"{name}_abs_signal": np.abs(signal),
            f"{name}_iso": iso,
            f"{name}_abs_signal_rank": pd.Series(np.abs(signal)).rank(pct=True).to_numpy(),
            f"{name}_iso_rank": pd.Series(iso).rank(pct=True).to_numpy(),
        }
    )


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    abs_signal_cols = [c for c in numeric_cols if c.endswith("_abs_signal")]
    iso_cols = [c for c in numeric_cols if c.endswith("_iso")]
    if abs_signal_cols:
        out["max_abs_signal"] = out[abs_signal_cols].max(axis=1)
        out["mean_abs_signal"] = out[abs_signal_cols].mean(axis=1)
        out["max_abs_signal_rank"] = out["max_abs_signal"].rank(pct=True).to_numpy()
    if iso_cols:
        out["max_iso"] = out[iso_cols].max(axis=1)
        out["mean_iso"] = out[iso_cols].mean(axis=1)
        out["max_iso_rank"] = out["max_iso"].rank(pct=True).to_numpy()
    return out


def detector_oof_score(model_name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    score = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        model, scaler = fit_detector(model_name, X.iloc[train_idx], y[train_idx])
        score[valid_idx] = predict_detector(model, scaler, X.iloc[valid_idx])
    return score


def detector_full_score(model_name: str, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    model, scaler = fit_detector(model_name, X, y)
    return predict_detector(model, scaler, X_test)


def fit_detector(model_name: str, X: pd.DataFrame, y: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if model_name == "logreg":
        model = LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42)
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(model_name)
    model.fit(Xs, y)
    return model, scaler


def predict_detector(model, scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


def evaluate_scores(
    model_name: str,
    hard_q: float,
    score: np.ndarray,
    y_hard: np.ndarray,
    residual_abs: np.ndarray,
    groups: np.ndarray,
) -> list[dict[str, object]]:
    auc = safe_auc(y_hard, score)
    ap = float(average_precision_score(y_hard, score))
    rows = []
    for k_frac in [0.05, 0.10, 0.15, 0.20]:
        k = max(1, int(round(len(score) * k_frac)))
        idx = np.argsort(score)[-k:]
        selected = np.zeros(len(score), dtype=bool)
        selected[idx] = True
        fold_hit_rates = []
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        for _, valid_idx in splitter.split(np.zeros((len(groups), 1)), y_hard, groups):
            if np.sum(selected[valid_idx]) == 0:
                continue
            fold_hit_rates.append(float(np.mean(y_hard[valid_idx][selected[valid_idx]])))
        rows.append(
            {
                "model": model_name,
                "hard_q": hard_q,
                "auc": auc,
                "average_precision": ap,
                "k_frac": k_frac,
                "precision_at_k": float(np.mean(y_hard[idx])),
                "selected_residual_mean": float(np.mean(residual_abs[idx])),
                "overall_residual_mean": float(np.mean(residual_abs)),
                "selected_residual_lift": float(np.mean(residual_abs[idx]) / np.mean(residual_abs)),
                "fold_precision_min": float(np.min(fold_hit_rates)) if fold_hit_rates else 0.0,
                "fold_precision_mean": float(np.mean(fold_hit_rates)) if fold_hit_rates else 0.0,
            }
        )
    return rows


def test_score_frame(
    model_name: str,
    hard_q: float,
    score: np.ndarray,
    anchor_test: np.ndarray,
    test_species: np.ndarray,
    test_ids: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "hard_q": hard_q,
            "sample_number": test_ids,
            "species_number": test_species,
            "anchor_pred": anchor_test,
            "hard_score": score,
            "score_rank": pd.Series(score).rank(pct=True).to_numpy(),
        }
    )


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.0
    return float(roc_auc_score(y, score))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))


if __name__ == "__main__":
    main()
