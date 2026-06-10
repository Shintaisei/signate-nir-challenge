#!/usr/bin/env python3
"""Search conservative extensions around the current Yeo-Johnson Ridge anchor."""

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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "nir_yj_extension"
CURRENT_BEST = SUBMISSION_DIR / "nir_ms_target2_yeojohnson_pca20_ridge3500.csv"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    X_train: np.ndarray
    X_test: np.ndarray
    alpha: float
    n_components: int
    memo: str
    sample_weight: np.ndarray | None = None


def main() -> None:
    data = load_data()
    current = pd.read_csv(CURRENT_BEST, header=None)
    ids = data["test_ids"]
    if not np.array_equal(ids, current[0].to_numpy()):
        raise ValueError("current best sample order mismatch")

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(data)
    rows = []
    for cand in candidates:
        pred = fit_predict_yj_ridge(
            cand.X_train,
            data["y"],
            cand.X_test,
            alpha=cand.alpha,
            n_components=cand.n_components,
            sample_weight=cand.sample_weight,
        )
        pred = np.clip(pred, 0, None)
        submission = pd.DataFrame({0: ids, 1: pred})
        path = SUBMISSION_DIR / f"{cand.name}.csv"
        submission.to_csv(path, index=False, header=False)
        row = diagnostics(cand, pred, current[1].to_numpy(dtype=float), path)
        rows.append(row)
        print(
            f"{cand.name}: family={cand.family} rmse={row['best_diff_rmse']:.4f} "
            f"corr={row['best_corr']:.6f} low={row['bottom_decile_delta']:.4f} "
            f"top={row['top_decile_delta']:.4f} range={row['range_ratio_vs_best']:.4f}"
        )

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "extension_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "extension_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop conservative candidates:")
    for row in rows_sorted[:20]:
        print(
            f"{row['experiment']}: {row['family']} rmse={row['best_diff_rmse']:.4f} "
            f"maxabs={row['best_diff_max_abs']:.4f} corr={row['best_corr']:.6f} "
            f"low={row['bottom_decile_delta']:.4f} top={row['top_decile_delta']:.4f} "
            f"range={row['range_ratio_vs_best']:.4f}"
        )
    print(f"saved extension diagnostics: {out_dir}")


def load_data() -> dict[str, np.ndarray]:
    train = pd.read_csv(TRAIN_PATH, encoding="cp932")
    test = pd.read_csv(TEST_PATH, encoding="cp932")
    sample = pd.read_csv(SAMPLE_PATH, header=None)
    feature_cols = [c for c in train.columns if is_float_like(c)]
    if feature_cols != [c for c in test.columns if is_float_like(c)]:
        raise ValueError("train/test feature columns mismatch")
    ids = test["sample number"].to_numpy()
    if not np.array_equal(ids, sample[0].to_numpy()):
        raise ValueError("sample_submit order mismatch")
    return {
        "X_train": train[feature_cols].to_numpy(dtype=float),
        "X_test": test[feature_cols].to_numpy(dtype=float),
        "y": train["含水率"].to_numpy(dtype=float),
        "groups": train["species number"].to_numpy(),
        "test_ids": ids,
    }


def is_float_like(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def build_candidates(data: dict[str, np.ndarray]) -> list[Candidate]:
    X_raw = data["X_train"]
    T_raw = data["X_test"]
    y = data["y"]

    base_train = snv(savgol_filter(X_raw, 9, 2, axis=1, mode="interp"))
    base_test = snv(savgol_filter(T_raw, 9, 2, axis=1, mode="interp"))
    yj = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(y.reshape(-1, 1)).ravel()

    cands: list[Candidate] = []
    for alpha in [3500.0, 3600.0]:
        cands.append(
            Candidate(
                name=f"nir_ext_base_yj_pca20_ridge{int(alpha)}",
                family="base_check",
                X_train=base_train,
                X_test=base_test,
                alpha=alpha,
                n_components=20,
                memo=f"Base SG9+SNV YJ Ridge alpha={alpha:g}",
            )
        )

    for window in [11, 15, 21]:
        d1_train = savgol_filter(snv(X_raw), window, 2, deriv=1, axis=1, mode="interp")
        d1_test = savgol_filter(snv(T_raw), window, 2, deriv=1, axis=1, mode="interp")
        d2_train = savgol_filter(snv(X_raw), window, 2, deriv=2, axis=1, mode="interp")
        d2_test = savgol_filter(snv(T_raw), window, 2, deriv=2, axis=1, mode="interp")
        for deriv_name, E_train, E_test in [("d1", d1_train, d1_test), ("d2", d2_train, d2_test)]:
            for n_components in [20, 22, 25, 30]:
                Xtr = np.hstack([base_train, E_train])
                Xte = np.hstack([base_test, E_test])
                cands.append(
                    Candidate(
                        name=f"nir_ext_stack_{deriv_name}w{window}_pca{n_components}_yj_ridge3500",
                        family="derivative_stack",
                        X_train=Xtr,
                        X_test=Xte,
                        alpha=3500.0,
                        n_components=n_components,
                        memo=f"SG9+SNV stacked with {deriv_name} window={window}",
                    )
                )

    for top_k in [32, 64, 96, 128, 192]:
        selected = top_wavelengths_by_quartile_delta(base_train, y, top_k)
        Xtr = base_train[:, selected]
        Xte = base_test[:, selected]
        for n_components in [8, 12, 16, min(20, top_k)]:
            cands.append(
                Candidate(
                    name=f"nir_ext_qdelta_top{top_k}_pca{n_components}_yj_ridge3500",
                    family="quartile_wavelength_selection",
                    X_train=Xtr,
                    X_test=Xte,
                    alpha=3500.0,
                    n_components=n_components,
                    memo=f"Top {top_k} wavelengths by high/low target quartile delta",
                )
            )

    for n_remove in [1, 2]:
        Xtr_osc, Xte_osc = remove_y_orthogonal_pcs(base_train, base_test, yj, n_remove=n_remove, search_components=12)
        for alpha in [3400.0, 3500.0, 3600.0]:
            cands.append(
                Candidate(
                    name=f"nir_ext_orthopcrem{n_remove}_pca20_yj_ridge{int(alpha)}",
                    family="orthogonal_pc_removal",
                    X_train=Xtr_osc,
                    X_test=Xte_osc,
                    alpha=alpha,
                    n_components=20,
                    memo=f"Remove {n_remove} high-variance PCs least correlated with YJ target",
                )
            )

    for label, weight in sample_weight_variants(y).items():
        cands.append(
            Candidate(
                name=f"nir_ext_weight_{label}_pca20_yj_ridge3500",
                family="weighted_yj_ridge",
                X_train=base_train,
                X_test=base_test,
                alpha=3500.0,
                n_components=20,
                memo=f"YJ Ridge with sample weights {label}",
                sample_weight=weight,
            )
        )

    return cands


def fit_predict_yj_ridge(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float,
    n_components: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    model = Ridge(alpha=alpha)
    model.fit(Z_train, yt, sample_weight=sample_weight)
    pred_t = model.predict(Z_test)
    return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()


def snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def top_wavelengths_by_quartile_delta(X: np.ndarray, y: np.ndarray, top_k: int) -> np.ndarray:
    lo = y <= np.quantile(y, 0.25)
    hi = y >= np.quantile(y, 0.75)
    score = np.abs(X[hi].mean(axis=0) - X[lo].mean(axis=0))
    return np.argsort(score)[-top_k:]


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
    pca = PCA(n_components=search_components, random_state=42)
    scores = pca.fit_transform(Xc)
    loadings = pca.components_
    corr = np.array([abs(np.corrcoef(scores[:, i], y_transformed)[0, 1]) for i in range(scores.shape[1])])
    order = np.argsort(corr)
    remove = order[:n_remove]
    Xc_new = Xc.copy()
    Tc_new = Tc.copy()
    for idx in remove:
        loading = loadings[idx]
        Xc_new -= np.outer(Xc_new @ loading, loading)
        Tc_new -= np.outer(Tc_new @ loading, loading)
    return Xc_new + x_mean, Tc_new + x_mean


def sample_weight_variants(y: np.ndarray) -> dict[str, np.ndarray]:
    pct = pd.Series(y).rank(pct=True).to_numpy()
    variants: dict[str, np.ndarray] = {}
    w = np.ones_like(y, dtype=float)
    w[pct >= 0.80] = 0.9
    w[pct <= 0.20] = 1.1
    variants["low110_high090"] = normalize_weight(w)
    w = np.ones_like(y, dtype=float)
    w[pct >= 0.80] = 0.85
    w[pct <= 0.20] = 1.15
    variants["low115_high085"] = normalize_weight(w)
    w = np.ones_like(y, dtype=float)
    w[pct >= 0.90] = 0.8
    w[pct <= 0.10] = 1.1
    variants["tail_low110_high080"] = normalize_weight(w)
    w = np.ones_like(y, dtype=float)
    w[pct >= 0.90] = 1.1
    w[pct <= 0.10] = 0.9
    variants["tail_low090_high110"] = normalize_weight(w)
    return variants


def normalize_weight(w: np.ndarray) -> np.ndarray:
    return w / np.mean(w)


def diagnostics(cand: Candidate, pred: np.ndarray, best: np.ndarray, path: Path) -> dict[str, object]:
    diff = pred - best
    lo = best <= np.quantile(best, 0.10)
    hi = best >= np.quantile(best, 0.90)
    return {
        "experiment": cand.name,
        "family": cand.family,
        "submission_path": str(path),
        "memo": cand.memo,
        "alpha": cand.alpha,
        "n_components": cand.n_components,
        "pred_min": float(np.min(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
        "pred_std": float(np.std(pred)),
        "negative_count": int((pred < 0).sum()),
        "best_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "best_diff_max_abs": float(np.max(np.abs(diff))),
        "best_corr": float(np.corrcoef(best, pred)[0, 1]),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "range_ratio_vs_best": float((np.max(pred) - np.min(pred)) / (np.max(best) - np.min(best))),
    }


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    corr = float(row["best_corr"])
    rmse = float(row["best_diff_rmse"])
    max_abs = float(row["best_diff_max_abs"])
    range_ratio = float(row["range_ratio_vs_best"])
    low = float(row["bottom_decile_delta"])
    top = float(row["top_decile_delta"])
    if corr < 0.995 or range_ratio < 0.92 or range_ratio > 1.04 or max_abs > 5.0:
        return (10.0, max_abs, -corr)
    direction_penalty = abs(max(low, 0.0)) * 0.1 + abs(min(top, 0.0)) * 0.1
    novelty = abs(rmse - 0.25)
    return (direction_penalty + novelty + abs(range_ratio - 1.0), max_abs, -corr)


if __name__ == "__main__":
    main()
