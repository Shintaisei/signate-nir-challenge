#!/usr/bin/env python3
"""Fold-safe CV search around the true Yeo-Johnson Ridge3500 anchor."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import PowerTransformer


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "nir_true_anchor_cv"
CURRENT_BEST = SUBMISSION_DIR / "nir_ms_target2_yeojohnson_pca20_ridge3500.csv"


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    alpha: float
    n_components: int = 20
    orthogonal_pc_remove: int = 0


def main() -> None:
    data = load_data()
    current = pd.read_csv(CURRENT_BEST, header=None)
    if not np.array_equal(data["test_ids"], current[0].to_numpy()):
        raise ValueError("current-best sample order mismatch")

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs()
    cv_cache: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    best = current[1].to_numpy(float)
    for spec in specs:
        pred = fit_full(spec, data)
        pred = np.clip(pred, 0, None)
        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        cv_cache[spec.name] = {
            **cv_score(spec, data, exclude_species=None),
            **cv_score(spec, data, exclude_species=15),
        }
        row = diagnostics(spec, pred, best, path)
        row.update(cv_cache[spec.name])
        rows.append(row)
        print_one(row)

    base = cv_cache["nir_ta_base_yj_pca20_ridge3500"]
    for row in rows:
        row["cv_group_delta_vs_base3500"] = (
            float(row["cv_group_species_rmse"]) - base["cv_group_species_rmse"]
        )
        row["cv_exclude15_delta_vs_base3500"] = (
            float(row["cv_exclude15_rmse"]) - base["cv_exclude15_rmse"]
        )

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "true_anchor_cv_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "true_anchor_cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop candidates:")
    for row in rows_sorted:
        print(
            f"{row['experiment']}: {row['family']} rmse={row['best_diff_rmse']:.4f} "
            f"max={row['best_diff_max_abs']:.4f} low={row['bottom_decile_delta']:.4f} "
            f"top={row['top_decile_delta']:.4f} "
            f"cv_group_delta={row['cv_group_delta_vs_base3500']:.4f} "
            f"cv_ex15_delta={row['cv_exclude15_delta_vs_base3500']:.4f}"
        )
    print(f"saved true-anchor CV diagnostics: {out_dir}")


def build_specs() -> list[Spec]:
    specs = [
        Spec("nir_ta_base_yj_pca20_ridge3500", "base", 3500.0),
        Spec("nir_ta_base_yj_pca20_ridge3600", "base", 3600.0),
    ]
    for remove in [1, 2]:
        for alpha in [3400.0, 3500.0, 3600.0]:
            specs.append(
                Spec(
                    name=f"nir_ta_orthopcrem{remove}_pca20_yj_ridge{int(alpha)}",
                    family="orthogonal_pc_removal",
                    alpha=alpha,
                    orthogonal_pc_remove=remove,
                )
            )
    return specs


def load_data() -> dict[str, np.ndarray]:
    train = pd.read_csv(TRAIN_PATH, encoding="cp932")
    test = pd.read_csv(TEST_PATH, encoding="cp932")
    sample = pd.read_csv(SAMPLE_PATH, header=None)
    feature_cols = [c for c in train.columns if is_float_like(c)]
    target_cols = [
        c
        for c in train.columns
        if c not in feature_cols
        and c not in {"sample number", "species number"}
        and pd.api.types.is_numeric_dtype(train[c])
    ]
    if len(target_cols) != 1:
        raise ValueError(f"could not infer target column, candidates={target_cols}")
    ids = test["sample number"].to_numpy()
    if not np.array_equal(ids, sample[0].to_numpy()):
        raise ValueError("sample_submit order mismatch")
    return {
        "X_train": train[feature_cols].to_numpy(float),
        "X_test": test[feature_cols].to_numpy(float),
        "y": train[target_cols[0]].to_numpy(float),
        "groups": train["species number"].to_numpy(),
        "test_ids": ids,
    }


def is_float_like(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def make_base_features(X: np.ndarray) -> np.ndarray:
    return snv(savgol_filter(X, 9, 2, axis=1, mode="interp"))


def snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / np.where(std == 0, 1.0, std)


def fit_full(spec: Spec, data: dict[str, np.ndarray]) -> np.ndarray:
    X_train = make_base_features(data["X_train"])
    X_test = make_base_features(data["X_test"])
    if spec.orthogonal_pc_remove:
        yj = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(
            data["y"].reshape(-1, 1)
        ).ravel()
        X_train, X_test = remove_y_orthogonal_pcs(
            X_train,
            X_test,
            yj,
            n_remove=spec.orthogonal_pc_remove,
            search_components=12,
        )
    return fit_predict_yj_ridge(
        X_train,
        data["y"],
        X_test,
        alpha=spec.alpha,
        n_components=spec.n_components,
    )


def cv_score(spec: Spec, data: dict[str, np.ndarray], *, exclude_species: int | None) -> dict[str, float]:
    X_raw = data["X_train"]
    y = data["y"]
    groups = data["groups"]
    if exclude_species is None:
        mask = np.ones(len(y), dtype=bool)
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        splits = splitter.split(X_raw[mask], y[mask], groups[mask])
        prefix = "cv_group_species"
    else:
        mask = groups != exclude_species
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = splitter.split(X_raw[mask], y[mask])
        prefix = "cv_exclude15"

    X_m = X_raw[mask]
    y_m = y[mask]
    rmses: list[float] = []
    for train_idx, valid_idx in splits:
        X_train = make_base_features(X_m[train_idx])
        X_valid = make_base_features(X_m[valid_idx])
        y_train = y_m[train_idx]
        y_valid = y_m[valid_idx]
        if spec.orthogonal_pc_remove:
            yj_train = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(
                y_train.reshape(-1, 1)
            ).ravel()
            X_train, X_valid = remove_y_orthogonal_pcs(
                X_train,
                X_valid,
                yj_train,
                n_remove=spec.orthogonal_pc_remove,
                search_components=12,
            )
        pred = fit_predict_yj_ridge(
            X_train,
            y_train,
            X_valid,
            alpha=spec.alpha,
            n_components=spec.n_components,
        )
        pred = np.clip(pred, 0, None)
        rmses.append(math.sqrt(mean_squared_error(y_valid, pred)))
    return {f"{prefix}_rmse": float(np.mean(rmses)), f"{prefix}_std": float(np.std(rmses))}


def fit_predict_yj_ridge(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float,
    n_components: int,
) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    model = Ridge(alpha=alpha)
    model.fit(Z_train, yt)
    pred_t = model.predict(Z_test)
    return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()


def remove_y_orthogonal_pcs(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_transformed: np.ndarray,
    *,
    n_remove: int,
    search_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = X_train.mean(axis=0, keepdims=True)
    Xc = X_train - x_mean
    Tc = X_test - x_mean
    n_components = min(search_components, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(Xc)
    loadings = pca.components_
    corr = np.array([abs(np.corrcoef(scores[:, i], y_transformed)[0, 1]) for i in range(scores.shape[1])])
    remove = np.argsort(corr)[:n_remove]
    Xc_new = Xc.copy()
    Tc_new = Tc.copy()
    for idx in remove:
        loading = loadings[idx]
        Xc_new -= np.outer(Xc_new @ loading, loading)
        Tc_new -= np.outer(Tc_new @ loading, loading)
    return Xc_new + x_mean, Tc_new + x_mean


def diagnostics(spec: Spec, pred: np.ndarray, best: np.ndarray, path: Path) -> dict[str, object]:
    diff = pred - best
    lo = best <= np.quantile(best, 0.10)
    hi = best >= np.quantile(best, 0.90)
    return {
        "experiment": spec.name,
        "family": spec.family,
        "submission_path": str(path),
        "alpha": spec.alpha,
        "n_components": spec.n_components,
        "orthogonal_pc_remove": spec.orthogonal_pc_remove,
        "pred_min": float(np.min(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
        "pred_std": float(np.std(pred)),
        "negative_count": int((pred < 0).sum()),
        "best_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "best_diff_max_abs": float(np.max(np.abs(diff))),
        "best_diff_mean": float(np.mean(diff)),
        "best_corr": float(np.corrcoef(best, pred)[0, 1]),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "range_ratio_vs_best": float((np.max(pred) - np.min(pred)) / (np.max(best) - np.min(best))),
    }


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    group_delta = float(row["cv_group_delta_vs_base3500"])
    ex15_delta = float(row["cv_exclude15_delta_vs_base3500"])
    rmse = float(row["best_diff_rmse"])
    max_abs = float(row["best_diff_max_abs"])
    corr = float(row["best_corr"])
    if corr < 0.99999 or max_abs > 1.0:
        return (10.0, max_abs, -corr)
    cv_penalty = max(group_delta, 0.0) + max(ex15_delta, 0.0)
    movement_penalty = abs(rmse - 0.18)
    return (cv_penalty + movement_penalty, max_abs, -corr)


def print_one(row: dict[str, object]) -> None:
    print(
        f"{row['experiment']}: rmse={row['best_diff_rmse']:.4f} "
        f"max={row['best_diff_max_abs']:.4f} corr={row['best_corr']:.6f} "
        f"low={row['bottom_decile_delta']:.4f} top={row['top_decile_delta']:.4f} "
        f"group={row['cv_group_species_rmse']:.4f} ex15={row['cv_exclude15_rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
