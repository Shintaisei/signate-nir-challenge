#!/usr/bin/env python3
"""SPXY/Kennard-Stone representative-weight search around the current best.

The current best uses a simple Yeo-Johnson residual soft weight. This script
keeps that idea as the anchor-like base and adds a separate chemometric sample
selection idea: samples selected late by Kennard-Stone/SPXY are more redundant
in the calibration space, so they receive a weak continuous downweight.

No test targets or Public scores are used to construct weights.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_SUBMIT_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "nir_spxy_weight"
CURRENT_BEST = SUBMISSION_DIR / "nir_s3tn_curbest_pls_k6_q16_c4_clcap2_f04_s0012_c018_20260608.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    distance_mode: str
    alpha: float = 3500.0
    pca_components: int = 20
    spxy_x_weight: float = 0.5
    late_pct: float = 0.10
    late_min_weight: float = 0.90
    gamma: float = 1.0
    anchor_weight: bool = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-top", type=int, default=18)
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()

    data = load_data()
    current_best = pd.read_csv(CURRENT_BEST, header=None)
    if not np.array_equal(data["test_ids"], current_best[0].to_numpy()):
        raise ValueError("current-best sample order mismatch")

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    X_train = make_base_features(data["X_train"])
    X_test = make_base_features(data["X_test"])
    y = data["y"]
    anchor_weights = make_anchor_like_weights(X_train, y, data["groups"])
    specs = build_specs()

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for spec in specs:
        weights = make_weights(spec, X_train, y, data["groups"], anchor_weights)
        raw_pred = fit_predict_yj_ridge(
            X_train,
            y,
            X_test,
            alpha=spec.alpha,
            n_components=spec.pca_components,
            sample_weight=weights,
        )
        pred = np.clip(raw_pred, 0, None)
        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        rows.append(diagnostics(spec, pred, np.asarray(raw_pred), current_best[1].to_numpy(float), path, weights, data["groups"]))
        weight_rows.extend(species_weight_rows(spec, weights, data["groups"]))
        print_one(rows[-1])

    rows_sorted = sorted(rows, key=rank_key)
    if not args.no_cv and args.cv_top > 0:
        picked = pick_cv_specs(rows_sorted, specs, args.cv_top)
        cv_map: dict[str, dict[str, float]] = {}
        for spec in picked:
            group = nested_cv(spec, data, exclude_species=None)
            ex15 = nested_cv(spec, data, exclude_species=15)
            cv_map[spec.name] = {
                "cv_group_species_rmse": group[0],
                "cv_group_species_std": group[1],
                "cv_exclude15_rmse": ex15[0],
                "cv_exclude15_std": ex15[1],
            }
            print(
                f"CV {spec.name}: group={group[0]:.4f}+/-{group[1]:.4f} "
                f"exclude15={ex15[0]:.4f}+/-{ex15[1]:.4f}"
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
        anchor_cv = cv_map.get("nir_spxy_anchor_like_yjres03_w085_ridge3500")
        if anchor_cv is not None:
            for row in rows_sorted:
                if row.get("cv_group_species_rmse") is not None:
                    row["cv_group_delta_vs_anchor"] = (
                        float(row["cv_group_species_rmse"]) - anchor_cv["cv_group_species_rmse"]
                    )
                    row["cv_exclude15_delta_vs_anchor"] = (
                        float(row["cv_exclude15_rmse"]) - anchor_cv["cv_exclude15_rmse"]
                    )
                else:
                    row["cv_group_delta_vs_anchor"] = None
                    row["cv_exclude15_delta_vs_anchor"] = None
        rows_sorted = sorted(rows_sorted, key=rank_key)

    write_outputs(out_dir, rows_sorted, weight_rows)
    print_top(rows_sorted)
    print(f"saved SPXY-weight diagnostics: {out_dir}")


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


def build_specs() -> list[CandidateSpec]:
    specs = [
        CandidateSpec(
            name="nir_spxy_anchor_like_yjres03_w085_ridge3500",
            distance_mode="anchor_only",
            anchor_weight=True,
        ),
    ]
    for distance_mode, x_weight in [("ks", 1.0), ("spxy75", 0.75), ("spxy50", 0.50)]:
        for late_pct in [0.05, 0.08, 0.12, 0.20]:
            for late_min_weight in [0.85, 0.90, 0.95]:
                for gamma in [0.5, 1.0, 2.0]:
                    for alpha in [3400.0, 3500.0, 3600.0]:
                        name = (
                            f"nir_spxy_{distance_mode}_late{pct_tag(late_pct)}_"
                            f"w{weight_tag(late_min_weight)}_g{gamma_tag(gamma)}_"
                            f"pca20_ridge{int(alpha)}"
                        )
                        specs.append(
                            CandidateSpec(
                                name=name,
                                distance_mode=distance_mode,
                                spxy_x_weight=x_weight,
                                late_pct=late_pct,
                                late_min_weight=late_min_weight,
                                gamma=gamma,
                                alpha=alpha,
                            )
                        )
    return specs


def pct_tag(value: float) -> str:
    pct = value * 100
    if pct < 1:
        return f"0p{int(round(pct * 100)):02d}"
    return str(int(round(pct))).zfill(2)


def weight_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def gamma_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def make_anchor_like_weights(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    residual_yj_abs = make_group_yj_oof_residual(X, y, groups)
    weights = np.ones(len(y), dtype=float)
    weights[top_mask(residual_yj_abs, 0.03)] = 0.85
    weights /= np.mean(weights)
    return weights


def make_group_yj_oof_residual(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    residual = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pca = PCA(n_components=20, random_state=42)
        Z_train = pca.fit_transform(X[train_idx])
        Z_valid = pca.transform(X[valid_idx])
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        y_train_t = transformer.fit_transform(y[train_idx].reshape(-1, 1)).ravel()
        y_valid_t = transformer.transform(y[valid_idx].reshape(-1, 1)).ravel()
        model = Ridge(alpha=3500.0)
        model.fit(Z_train, y_train_t)
        pred_t = model.predict(Z_valid)
        residual[valid_idx] = np.abs(y_valid_t - pred_t)
    return residual


def top_mask(values: np.ndarray, pct: float) -> np.ndarray:
    if pct <= 0:
        return np.zeros(len(values), dtype=bool)
    return values >= np.quantile(values, 1.0 - pct)


def make_weights(
    spec: CandidateSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    anchor_weights: np.ndarray,
) -> np.ndarray:
    weights = anchor_weights.copy() if spec.anchor_weight else np.ones(len(y), dtype=float)
    if spec.distance_mode == "anchor_only":
        weights /= np.mean(weights)
        return weights

    order = kennard_stone_order(X, y, spec.spxy_x_weight)
    selection_pct = np.empty(len(y), dtype=float)
    selection_pct[order] = np.arange(len(y), dtype=float) / max(len(y) - 1, 1)
    tail_start = 1.0 - spec.late_pct
    tail = np.clip((selection_pct - tail_start) / max(spec.late_pct, 1e-12), 0.0, 1.0)
    penalty = tail**spec.gamma
    spxy_weights = 1.0 - (1.0 - spec.late_min_weight) * penalty
    weights *= spxy_weights
    weights /= np.mean(weights)
    return weights


def kennard_stone_order(X: np.ndarray, y: np.ndarray, x_weight: float) -> np.ndarray:
    pca_scores = PCA(n_components=20, random_state=42).fit_transform(X)
    Xs = StandardScaler().fit_transform(pca_scores)
    dx = pairwise_distances(Xs, metric="euclidean")
    dx /= np.max(dx) if np.max(dx) > 0 else 1.0
    if x_weight >= 0.999:
        d = dx
    else:
        yt = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(y.reshape(-1, 1))
        dy = pairwise_distances(yt, metric="euclidean")
        dy /= np.max(dy) if np.max(dy) > 0 else 1.0
        d = x_weight * dx + (1.0 - x_weight) * dy

    n = len(y)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    selected = [int(i), int(j)] if i != j else [int(i)]
    remaining = np.ones(n, dtype=bool)
    remaining[selected] = False
    min_dist = np.min(d[:, selected], axis=1)
    while len(selected) < n:
        masked = np.where(remaining, min_dist, -np.inf)
        nxt = int(np.argmax(masked))
        selected.append(nxt)
        remaining[nxt] = False
        min_dist = np.minimum(min_dist, d[:, nxt])
    return np.asarray(selected, dtype=int)


def fit_predict_yj_ridge(
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float,
    n_components: int,
    sample_weight: np.ndarray | None,
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


def diagnostics(
    spec: CandidateSpec,
    pred: np.ndarray,
    raw_pred: np.ndarray,
    anchor: np.ndarray,
    path: Path,
    weights: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    diff = pred - anchor
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    low_weight = weights < np.quantile(weights, 0.10)
    high_weight = weights > np.quantile(weights, 0.90)
    species_low = {
        str(g): int(np.sum(low_weight & (groups == g)))
        for g in sorted(np.unique(groups))
        if int(np.sum(low_weight & (groups == g))) > 0
    }
    species_high = {
        str(g): int(np.sum(high_weight & (groups == g)))
        for g in sorted(np.unique(groups))
        if int(np.sum(high_weight & (groups == g))) > 0
    }
    effective_n = float((np.sum(weights) ** 2) / np.sum(weights**2))
    max_low_weight_species_rate = max(
        float(np.sum(low_weight & (groups == g)) / np.sum(groups == g)) for g in np.unique(groups)
    )
    max_high_weight_species_rate = max(
        float(np.sum(high_weight & (groups == g)) / np.sum(groups == g)) for g in np.unique(groups)
    )
    return {
        "experiment": spec.name,
        "submission_path": str(path),
        "distance_mode": spec.distance_mode,
        "alpha": spec.alpha,
        "pca_components": spec.pca_components,
        "spxy_x_weight": spec.spxy_x_weight,
        "late_pct": spec.late_pct,
        "late_min_weight": spec.late_min_weight,
        "gamma": spec.gamma,
        "anchor_weight": spec.anchor_weight,
        "raw_pred_min": float(np.min(raw_pred)),
        "raw_negative_count": int(np.sum(raw_pred < 0)),
        "pred_min": float(np.min(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
        "pred_std": float(np.std(pred)),
        "negative_count": int(np.sum(pred < 0)),
        "anchor_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": float(np.corrcoef(anchor, pred)[0, 1]),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "range_ratio_vs_anchor": float((np.max(pred) - np.min(pred)) / (np.max(anchor) - np.min(anchor))),
        "min_sample_weight": float(np.min(weights)),
        "p10_sample_weight": float(np.quantile(weights, 0.10)),
        "mean_sample_weight": float(np.mean(weights)),
        "max_sample_weight": float(np.max(weights)),
        "effective_n": effective_n,
        "effective_n_ratio": effective_n / len(weights),
        "max_low_weight_species_rate": max_low_weight_species_rate,
        "max_high_weight_species_rate": max_high_weight_species_rate,
        "low_weight_by_species_json": json.dumps(species_low, ensure_ascii=False),
        "high_weight_by_species_json": json.dumps(species_high, ensure_ascii=False),
    }


def species_weight_rows(spec: CandidateSpec, weights: np.ndarray, groups: np.ndarray) -> list[dict[str, object]]:
    low_weight = weights < np.quantile(weights, 0.10)
    high_weight = weights > np.quantile(weights, 0.90)
    rows = []
    for group in sorted(np.unique(groups)):
        mask = groups == group
        rows.append(
            {
                "experiment": spec.name,
                "distance_mode": spec.distance_mode,
                "species": int(group),
                "n": int(np.sum(mask)),
                "mean_weight": float(np.mean(weights[mask])),
                "min_weight": float(np.min(weights[mask])),
                "max_weight": float(np.max(weights[mask])),
                "low_weight_count": int(np.sum(mask & low_weight)),
                "low_weight_rate": float(np.sum(mask & low_weight) / np.sum(mask)),
                "high_weight_count": int(np.sum(mask & high_weight)),
                "high_weight_rate": float(np.sum(mask & high_weight) / np.sum(mask)),
            }
        )
    return rows


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    rmse = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    corr = float(row["anchor_corr"])
    low_delta = float(row["bottom_decile_delta"])
    top_delta = float(row["top_decile_delta"])
    range_ratio = float(row["range_ratio_vs_anchor"])
    species_rate = float(row["max_low_weight_species_rate"])
    high_species_rate = float(row["max_high_weight_species_rate"])
    cv_group = row.get("cv_group_species_rmse")
    cv_ex15 = row.get("cv_exclude15_rmse")
    cv_group_delta = row.get("cv_group_delta_vs_anchor")
    cv_ex15_delta = row.get("cv_exclude15_delta_vs_anchor")
    hard_fail = 0.0
    if corr < 0.99995 or rmse < 0.08 or rmse > 0.40 or max_abs > 2.0:
        hard_fail += 10.0
    if low_delta < -0.5 or low_delta > 0.5 or top_delta < -0.8 or top_delta > 0.8:
        hard_fail += 6.0
    if range_ratio < 0.985 or range_ratio > 1.015:
        hard_fail += 5.0
    if species_rate > 0.35:
        hard_fail += 4.0
    if high_species_rate > 0.35:
        hard_fail += 4.0
    if str(row["distance_mode"]) == "spxy50":
        hard_fail += 2.0
    if cv_group_delta is not None and not pd.isna(cv_group_delta) and float(cv_group_delta) > 0.12:
        hard_fail += 4.0
    if cv_ex15_delta is not None and not pd.isna(cv_ex15_delta) and float(cv_ex15_delta) > 0.12:
        hard_fail += 4.0
    movement = abs(rmse - 0.22)
    cv_penalty = 0.0
    if cv_group_delta is not None and not pd.isna(cv_group_delta):
        cv_penalty += max(0.0, float(cv_group_delta))
    elif cv_group is not None and not pd.isna(cv_group):
        cv_penalty += max(0.0, float(cv_group) - 25.72)
    if cv_ex15_delta is not None and not pd.isna(cv_ex15_delta):
        cv_penalty += max(0.0, float(cv_ex15_delta))
    elif cv_ex15 is not None and not pd.isna(cv_ex15):
        cv_penalty += max(0.0, float(cv_ex15) - 19.98)
    return (hard_fail, movement + cv_penalty, max_abs, species_rate, -corr)


def pick_cv_specs(rows_sorted: list[dict[str, object]], specs: list[CandidateSpec], cv_top: int) -> list[CandidateSpec]:
    spec_by_name = {spec.name: spec for spec in specs}
    picked_names = ["nir_spxy_anchor_like_yjres03_w085_ridge3500"]
    for mode in ["ks", "spxy75", "spxy50"]:
        for row in rows_sorted:
            name = str(row["experiment"])
            if row.get("distance_mode") == mode and name not in picked_names:
                picked_names.append(name)
                break
    for row in rows_sorted:
        if len(picked_names) >= cv_top:
            break
        name = str(row["experiment"])
        if name not in picked_names:
            picked_names.append(name)
    return [spec_by_name[name] for name in picked_names if name in spec_by_name]


def nested_cv(spec: CandidateSpec, data: dict[str, np.ndarray], *, exclude_species: int | None) -> tuple[float, float]:
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
        train_groups = groups[train_idx]
        anchor_weights = make_anchor_like_weights(X_train, y_train, train_groups)
        weights = make_weights(spec, X_train, y_train, train_groups, anchor_weights)
        pred = fit_predict_yj_ridge(
            X_train,
            y_train,
            X_valid,
            alpha=spec.alpha,
            n_components=spec.pca_components,
            sample_weight=weights,
        )
        rmses.append(math.sqrt(mean_squared_error(y[valid_idx], np.clip(pred, 0, None))))
    return float(np.mean(rmses)), float(np.std(rmses))


def write_outputs(out_dir: Path, rows: list[dict[str, object]], weight_rows: list[dict[str, object]]) -> None:
    with (out_dir / "spxy_weight_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "spxy_weight_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with (out_dir / "species_weight_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)


def print_one(row: dict[str, object]) -> None:
    print(
        f"{row['experiment']}: rmse={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} corr={row['anchor_corr']:.6f} "
        f"low={row['bottom_decile_delta']:.4f} top={row['top_decile_delta']:.4f} "
        f"range={row['range_ratio_vs_anchor']:.4f} effN={row['effective_n_ratio']:.4f}"
    )


def print_top(rows: list[dict[str, object]]) -> None:
    print("\nTop SPXY-weight candidates:")
    for row in rows[:20]:
        print_one(row)
        if row.get("cv_group_species_rmse") is not None:
            print(
                f"  CV group={row['cv_group_species_rmse']:.4f}+/-{row['cv_group_species_std']:.4f} "
                f"exclude15={row['cv_exclude15_rmse']:.4f}+/-{row['cv_exclude15_std']:.4f}"
            )


if __name__ == "__main__":
    main()
