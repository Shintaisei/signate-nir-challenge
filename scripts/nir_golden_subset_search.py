#!/usr/bin/env python3
"""Golden-subset search around the Yeo-Johnson Ridge anchor.

The goal is to keep the current strong base model unchanged and only adjust
which training samples are trusted. This follows the SS4GG-style data-screening
idea: a simple spectral model can improve if noisy calibration samples are
removed or downweighted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_SUBMIT_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "nir_golden_subset"
CURRENT_BEST = SUBMISSION_DIR / "nir_ms_target2_yeojohnson_pca20_ridge3500.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    alpha: float = 3500.0
    pca_components: int = 20
    trim_pct: float | None = None
    weight_value: float | None = None
    residual_pct: float | None = None
    leverage_pct: float | None = None
    neighbor_pct: float | None = None
    neighbor_k: int = 20
    residual_score: str = "raw"


@dataclass
class SampleScores:
    oof_pred: np.ndarray
    residual_abs: np.ndarray
    residual_yj_abs: np.ndarray
    leverage: np.ndarray
    neighbor_inconsistency: dict[int, np.ndarray]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-top", type=int, default=12, help="Run nested group CV for top N candidates.")
    parser.add_argument("--no-cv", action="store_true", help="Skip nested CV diagnostics.")
    args = parser.parse_args()

    data = load_data()
    current_best = pd.read_csv(CURRENT_BEST, header=None)
    if not np.array_equal(data["test_ids"], current_best[0].to_numpy()):
        raise ValueError("current-best sample order mismatch")

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = make_base_features(data["X_train"])
    X_test = make_base_features(data["X_test"])
    scores = score_samples(X_train, data["y"], data["groups"])
    specs = build_specs()

    rows = []
    removal_rows = []
    for spec in specs:
        pred, keep_mask, sample_weight = fit_full_candidate(spec, X_train, data["y"], X_test, scores)
        path = SUBMISSION_DIR / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)

        row = diagnostics(
            spec,
            pred,
            current_best[1].to_numpy(dtype=float),
            path,
            keep_mask,
            sample_weight,
            data["groups"],
        )
        rows.append(row)
        removal_rows.extend(species_removal_rows(spec, keep_mask, sample_weight, data["groups"]))
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)

    if not args.no_cv and args.cv_top > 0:
        cv_specs = pick_cv_specs(rows_sorted, specs, args.cv_top)
        cv_map = {}
        for spec in cv_specs:
            cv_group = nested_cv(spec, data, exclude_species=None)
            cv_ex15 = nested_cv(spec, data, exclude_species=15)
            cv_map[spec.name] = {
                "cv_group_species_rmse": cv_group[0],
                "cv_group_species_std": cv_group[1],
                "cv_exclude15_rmse": cv_ex15[0],
                "cv_exclude15_std": cv_ex15[1],
            }
            print(
                f"CV {spec.name}: group={cv_group[0]:.4f}+/-{cv_group[1]:.4f} "
                f"exclude15={cv_ex15[0]:.4f}+/-{cv_ex15[1]:.4f}"
            )
        for row in rows_sorted:
            row.update(
                cv_map.get(
                    str(row["experiment"]),
                    {
                        "cv_group_species_rmse": None,
                        "cv_group_species_std": None,
                        "cv_exclude15_rmse": None,
                        "cv_exclude15_std": None,
                    },
                )
            )

    write_outputs(out_dir, rows_sorted, removal_rows, scores, data["groups"])
    print_top(rows_sorted)
    print(f"saved golden-subset diagnostics: {out_dir}")


def load_data() -> dict[str, np.ndarray]:
    train = pd.read_csv(TRAIN_PATH, encoding="cp932")
    test = pd.read_csv(TEST_PATH, encoding="cp932")
    sample_submit = pd.read_csv(SAMPLE_SUBMIT_PATH, header=None)

    feature_cols = [c for c in train.columns if is_float_like(c)]
    if feature_cols != [c for c in test.columns if is_float_like(c)]:
        raise ValueError("train/test spectral feature columns mismatch")

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
    if not np.array_equal(ids, sample_submit[0].to_numpy()):
        raise ValueError("test sample order does not match sample_submit.csv")

    return {
        "X_train": train[feature_cols].to_numpy(dtype=float),
        "X_test": test[feature_cols].to_numpy(dtype=float),
        "y": train[target_cols[0]].to_numpy(dtype=float),
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
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def build_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = [
        CandidateSpec("nir_golden_base_yj3500", "base", alpha=3500.0),
        CandidateSpec("nir_golden_base_yj3600", "base", alpha=3600.0),
    ]
    for trim_pct in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for alpha in [3400.0, 3500.0, 3600.0]:
            specs.append(
                CandidateSpec(
                    name=f"nir_golden_hard_resid{pct_tag(trim_pct)}_yj{int(alpha)}",
                    family="golden_hard_trim",
                    alpha=alpha,
                    trim_pct=trim_pct,
                )
            )
    for trim_pct in [0.0025, 0.005, 0.01, 0.02]:
        for alpha in [3500.0, 3600.0]:
            specs.append(
                CandidateSpec(
                    name=f"nir_golden_micro_yjres{pct_tag(trim_pct)}_yj{int(alpha)}",
                    family="golden_hard_trim",
                    alpha=alpha,
                    trim_pct=trim_pct,
                    residual_score="yj",
                )
            )
    for residual_pct in [0.03, 0.05, 0.08]:
        for leverage_pct in [0.05, 0.10, 0.15]:
            for alpha in [3500.0, 3600.0]:
                specs.append(
                    CandidateSpec(
                        name=f"nir_golden_and_resid{pct_tag(residual_pct)}_lev{pct_tag(leverage_pct)}_yj{int(alpha)}",
                        family="golden_resid_leverage_trim",
                        alpha=alpha,
                        residual_pct=residual_pct,
                        leverage_pct=leverage_pct,
                    )
                )
    for residual_pct in [0.005, 0.01, 0.02]:
        for leverage_pct in [0.05, 0.10]:
            for alpha in [3500.0, 3600.0]:
                specs.append(
                    CandidateSpec(
                        name=f"nir_golden_and_yjres{pct_tag(residual_pct)}_lev{pct_tag(leverage_pct)}_yj{int(alpha)}",
                        family="golden_resid_leverage_trim",
                        alpha=alpha,
                        residual_pct=residual_pct,
                        leverage_pct=leverage_pct,
                        residual_score="yj",
                    )
                )
    for residual_pct in [0.03, 0.05, 0.08]:
        for weight_value in [0.7, 0.5, 0.3]:
            for alpha in [3500.0, 3600.0]:
                specs.append(
                    CandidateSpec(
                        name=f"nir_golden_soft_resid{pct_tag(residual_pct)}_w{weight_tag(weight_value)}_yj{int(alpha)}",
                        family="golden_soft_weight",
                        alpha=alpha,
                        residual_pct=residual_pct,
                        weight_value=weight_value,
                    )
                )
    for residual_pct in [0.005, 0.01, 0.02, 0.03]:
        for weight_value in [0.95, 0.90, 0.85, 0.80]:
            for alpha in [3500.0, 3600.0]:
                specs.append(
                    CandidateSpec(
                        name=f"nir_golden_soft_yjres{pct_tag(residual_pct)}_w{weight_tag(weight_value)}_yj{int(alpha)}",
                        family="golden_soft_weight",
                        alpha=alpha,
                        residual_pct=residual_pct,
                        weight_value=weight_value,
                        residual_score="yj",
                    )
                )
    for residual_pct in [0.03, 0.05]:
        for neighbor_pct in [0.05, 0.10]:
            for k in [10, 20]:
                specs.append(
                    CandidateSpec(
                        name=f"nir_golden_and_resid{pct_tag(residual_pct)}_nn{k}_{pct_tag(neighbor_pct)}_yj3500",
                        family="golden_resid_neighbor_trim",
                        alpha=3500.0,
                        residual_pct=residual_pct,
                        neighbor_pct=neighbor_pct,
                        neighbor_k=k,
                    )
                )
    for residual_pct in [0.005, 0.01, 0.02]:
        for neighbor_pct in [0.05, 0.10]:
            for k in [10, 20]:
                specs.append(
                    CandidateSpec(
                        name=f"nir_golden_and_yjres{pct_tag(residual_pct)}_nn{k}_{pct_tag(neighbor_pct)}_yj3500",
                        family="golden_resid_neighbor_trim",
                        alpha=3500.0,
                        residual_pct=residual_pct,
                        neighbor_pct=neighbor_pct,
                        neighbor_k=k,
                        residual_score="yj",
                    )
                )
    return specs


def pct_tag(value: float) -> str:
    pct = value * 100
    if pct < 1:
        return f"0p{int(round(pct * 100)):02d}"
    return str(int(round(pct))).zfill(2)


def weight_tag(value: float) -> str:
    return str(value).replace(".", "p")


def score_samples(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> SampleScores:
    oof_pred, oof_pred_yj = make_group_oof(X, y, groups)
    yj_target = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(y.reshape(-1, 1)).ravel()
    leverage, pca_scores = compute_leverage(X)
    return SampleScores(
        oof_pred=oof_pred,
        residual_abs=np.abs(y - oof_pred),
        residual_yj_abs=np.abs(yj_target - oof_pred_yj),
        leverage=leverage,
        neighbor_inconsistency={
            10: compute_neighbor_inconsistency(pca_scores, y, k=10),
            20: compute_neighbor_inconsistency(pca_scores, y, k=20),
        },
    )


def make_group_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oof = np.empty_like(y, dtype=float)
    oof_yj = np.empty_like(y, dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred, pred_yj = fit_predict_yj_ridge(
            X[train_idx],
            y[train_idx],
            X[valid_idx],
            alpha=3500.0,
            n_components=20,
            return_transformed=True,
        )
        oof[valid_idx] = pred
        oof_yj[valid_idx] = pred_yj
    return oof, oof_yj


def compute_leverage(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = PCA(n_components=20, random_state=42).fit_transform(X)
    scaled = StandardScaler().fit_transform(scores)
    leverage = np.sum(scaled**2, axis=1)
    return leverage, scaled


def compute_neighbor_inconsistency(scores: np.ndarray, y: np.ndarray, *, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(scores)
    _, indices = nn.kneighbors(scores)
    neighbor_y = y[indices[:, 1:]]
    return np.abs(y - np.median(neighbor_y, axis=1))


def fit_full_candidate(
    spec: CandidateSpec,
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    scores: SampleScores,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    keep_mask, sample_weight = build_keep_and_weight(spec, scores, len(y))
    if sample_weight is None:
        pred = fit_predict_yj_ridge(
            X_train[keep_mask],
            y[keep_mask],
            X_test,
            alpha=spec.alpha,
            n_components=spec.pca_components,
        )
    else:
        pred = fit_predict_yj_ridge(
            X_train,
            y,
            X_test,
            alpha=spec.alpha,
            n_components=spec.pca_components,
            sample_weight=sample_weight,
        )
    return np.clip(pred, 0, None), keep_mask, sample_weight


def build_keep_and_weight(
    spec: CandidateSpec,
    scores: SampleScores,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    keep = np.ones(n_samples, dtype=bool)
    weight = None
    if spec.family == "base":
        return keep, weight
    if spec.family == "golden_hard_trim":
        keep &= ~top_mask(residual_values(scores, spec), spec.trim_pct or 0.0)
        return keep, weight
    if spec.family == "golden_resid_leverage_trim":
        remove = top_mask(residual_values(scores, spec), spec.residual_pct or 0.0) & top_mask(
            scores.leverage,
            spec.leverage_pct or 0.0,
        )
        keep &= ~remove
        return keep, weight
    if spec.family == "golden_resid_neighbor_trim":
        remove = top_mask(residual_values(scores, spec), spec.residual_pct or 0.0) & top_mask(
            scores.neighbor_inconsistency[spec.neighbor_k],
            spec.neighbor_pct or 0.0,
        )
        keep &= ~remove
        return keep, weight
    if spec.family == "golden_soft_weight":
        weight = np.ones(n_samples, dtype=float)
        weight[top_mask(residual_values(scores, spec), spec.residual_pct or 0.0)] = float(spec.weight_value)
        weight /= np.mean(weight)
        return keep, weight
    raise ValueError(f"unknown candidate family: {spec.family}")


def residual_values(scores: SampleScores, spec: CandidateSpec) -> np.ndarray:
    if spec.residual_score == "raw":
        return scores.residual_abs
    if spec.residual_score == "yj":
        return scores.residual_yj_abs
    raise ValueError(f"unknown residual score: {spec.residual_score}")


def top_mask(values: np.ndarray, pct: float) -> np.ndarray:
    if pct <= 0:
        return np.zeros(len(values), dtype=bool)
    threshold = np.quantile(values, 1.0 - pct)
    return values >= threshold


def fit_predict_yj_ridge(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float,
    n_components: int,
    sample_weight: np.ndarray | None = None,
    return_transformed: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    model = Ridge(alpha=alpha)
    model.fit(Z_train, yt, sample_weight=sample_weight)
    pred_yj = model.predict(Z_test)
    pred = transformer.inverse_transform(pred_yj.reshape(-1, 1)).ravel()
    if return_transformed:
        return pred, pred_yj
    return pred


def diagnostics(
    spec: CandidateSpec,
    pred: np.ndarray,
    best: np.ndarray,
    path: Path,
    keep_mask: np.ndarray,
    sample_weight: np.ndarray | None,
    groups: np.ndarray,
) -> dict[str, object]:
    diff = pred - best
    lo = best <= np.quantile(best, 0.10)
    hi = best >= np.quantile(best, 0.90)
    removed = ~keep_mask
    removed_by_species = {
        str(g): int(np.sum(removed & (groups == g)))
        for g in sorted(np.unique(groups))
        if int(np.sum(removed & (groups == g))) > 0
    }
    max_species_removed_rate = 0.0
    if np.any(removed):
        rates = [
            float(np.sum(removed & (groups == g)) / np.sum(groups == g))
            for g in np.unique(groups)
        ]
        max_species_removed_rate = max(rates)
    return {
        "experiment": spec.name,
        "family": spec.family,
        "submission_path": str(path),
        "alpha": spec.alpha,
        "pca_components": spec.pca_components,
        "trim_pct": spec.trim_pct,
        "residual_pct": spec.residual_pct,
        "leverage_pct": spec.leverage_pct,
        "neighbor_pct": spec.neighbor_pct,
        "neighbor_k": spec.neighbor_k,
        "weight_value": spec.weight_value,
        "residual_score": spec.residual_score,
        "pred_min": float(np.min(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
        "pred_std": float(np.std(pred)),
        "negative_count": int(np.sum(pred < 0)),
        "best_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "best_diff_max_abs": float(np.max(np.abs(diff))),
        "best_corr": float(np.corrcoef(best, pred)[0, 1]),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "range_ratio_vs_best": float((np.max(pred) - np.min(pred)) / (np.max(best) - np.min(best))),
        "removed_count": int(np.sum(removed)),
        "removed_rate": float(np.mean(removed)),
        "min_sample_weight": None if sample_weight is None else float(np.min(sample_weight)),
        "max_sample_weight": None if sample_weight is None else float(np.max(sample_weight)),
        "max_species_removed_rate": max_species_removed_rate,
        "removed_by_species_json": json.dumps(removed_by_species, ensure_ascii=False),
    }


def species_removal_rows(
    spec: CandidateSpec,
    keep_mask: np.ndarray,
    sample_weight: np.ndarray | None,
    groups: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    removed = ~keep_mask
    for group in sorted(np.unique(groups)):
        mask = groups == group
        row = {
            "experiment": spec.name,
            "family": spec.family,
            "species": int(group),
            "n": int(np.sum(mask)),
            "removed": int(np.sum(mask & removed)),
            "removed_rate": float(np.sum(mask & removed) / np.sum(mask)),
            "mean_weight": None if sample_weight is None else float(np.mean(sample_weight[mask])),
            "min_weight": None if sample_weight is None else float(np.min(sample_weight[mask])),
        }
        rows.append(row)
    return rows


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    corr = float(row["best_corr"])
    rmse = float(row["best_diff_rmse"])
    max_abs = float(row["best_diff_max_abs"])
    range_ratio = float(row["range_ratio_vs_best"])
    removed_rate = float(row["removed_rate"])
    max_species_removed = float(row["max_species_removed_rate"])

    hard_fail = 0.0
    if corr < 0.999 or rmse < 0.04 or rmse > 0.60 or max_abs > 2.5:
        hard_fail += 10.0
    if range_ratio < 0.97 or range_ratio > 1.03:
        hard_fail += 10.0
    if max_species_removed > 0.25:
        hard_fail += 5.0
    target_movement = abs(rmse - 0.22)
    species_penalty = max(0.0, max_species_removed - 0.15)
    removal_penalty = max(0.0, removed_rate - 0.06)
    return (hard_fail, target_movement + species_penalty + removal_penalty, max_abs, -corr)


def pick_cv_specs(
    rows_sorted: list[dict[str, object]],
    specs: list[CandidateSpec],
    cv_top: int,
) -> list[CandidateSpec]:
    spec_by_name = {spec.name: spec for spec in specs}
    picked: list[CandidateSpec] = []
    for name in ["nir_golden_base_yj3500", "nir_golden_base_yj3600"]:
        picked.append(spec_by_name[name])
    for row in rows_sorted:
        if len(picked) >= cv_top + 2:
            break
        name = str(row["experiment"])
        if name not in {spec.name for spec in picked}:
            picked.append(spec_by_name[name])
    return picked


def nested_cv(
    spec: CandidateSpec,
    data: dict[str, np.ndarray],
    *,
    exclude_species: int | None,
) -> tuple[float, float]:
    groups_all = data["groups"]
    mask = np.ones(len(groups_all), dtype=bool)
    if exclude_species is not None:
        mask &= groups_all != exclude_species
    X_raw = data["X_train"][mask]
    y = data["y"][mask]
    groups = groups_all[mask]

    rmses = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X_raw, y, groups):
        X_train = make_base_features(X_raw[train_idx])
        X_valid = make_base_features(X_raw[valid_idx])
        y_train = y[train_idx]
        scores = score_samples(X_train, y_train, groups[train_idx])
        keep_mask, sample_weight = build_keep_and_weight(spec, scores, len(y_train))
        if sample_weight is None:
            pred = fit_predict_yj_ridge(
                X_train[keep_mask],
                y_train[keep_mask],
                X_valid,
                alpha=spec.alpha,
                n_components=spec.pca_components,
            )
        else:
            pred = fit_predict_yj_ridge(
                X_train,
                y_train,
                X_valid,
                alpha=spec.alpha,
                n_components=spec.pca_components,
                sample_weight=sample_weight,
            )
        rmses.append(math.sqrt(mean_squared_error(y[valid_idx], np.clip(pred, 0, None))))
    return float(np.mean(rmses)), float(np.std(rmses))


def write_outputs(
    out_dir: Path,
    rows_sorted: list[dict[str, object]],
    removal_rows: list[dict[str, object]],
    scores: SampleScores,
    groups: np.ndarray,
) -> None:
    if not rows_sorted:
        raise ValueError("no rows to write")
    with (out_dir / "golden_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "golden_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    with (out_dir / "species_screening_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(removal_rows[0]))
        writer.writeheader()
        writer.writerows(removal_rows)

    score_df = pd.DataFrame(
        {
            "species": groups,
            "oof_pred": scores.oof_pred,
            "residual_abs": scores.residual_abs,
            "residual_yj_abs": scores.residual_yj_abs,
            "leverage": scores.leverage,
            "neighbor_inconsistency_k10": scores.neighbor_inconsistency[10],
            "neighbor_inconsistency_k20": scores.neighbor_inconsistency[20],
        }
    )
    score_df.to_csv(out_dir / "sample_screening_scores.csv", index=False)


def print_one(row: dict[str, object]) -> None:
    print(
        f"{row['experiment']}: family={row['family']} rmse={row['best_diff_rmse']:.4f} "
        f"max={row['best_diff_max_abs']:.4f} corr={row['best_corr']:.6f} "
        f"low={row['bottom_decile_delta']:.4f} top={row['top_decile_delta']:.4f} "
        f"range={row['range_ratio_vs_best']:.4f} removed={row['removed_count']}"
    )


def print_top(rows_sorted: list[dict[str, object]]) -> None:
    print("\nTop gated candidates:")
    for row in rows_sorted[:20]:
        print_one(row)
        if row.get("cv_group_species_rmse") is not None:
            print(
                f"  cv_group={row['cv_group_species_rmse']:.4f}+/-{row['cv_group_species_std']:.4f} "
                f"cv_ex15={row['cv_exclude15_rmse']:.4f}+/-{row['cv_exclude15_std']:.4f}"
            )


if __name__ == "__main__":
    main()
