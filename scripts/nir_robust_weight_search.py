#!/usr/bin/env python3
"""Robust sample-weight search around the current golden-subset anchor.

This extends the first successful golden-subset idea from a binary "top N
samples get one lower weight" rule into a continuous calibration-sample
confidence model. The model itself stays intentionally simple:

    SG9 smoothing -> SNV -> PCA -> auto Yeo-Johnson target -> Ridge

Only sample weights are changed. Candidate scores are built from OOF residuals,
PCA leverage, and local-neighbor label inconsistency in spectral latent space.
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
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_SUBMIT_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "nir_robust_weight"
CURRENT_BEST = SUBMISSION_DIR / "nir_golden_soft_yjres03_w0p85_yj3500.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    score_key: str
    alpha: float = 3500.0
    pca_components: int = 20
    top_pct: float = 0.05
    min_weight: float = 0.85
    gamma: float = 1.0
    mode: str = "tail_power"


@dataclass
class SampleScores:
    oof_pred: np.ndarray
    residual_abs: np.ndarray
    residual_yj_abs: np.ndarray
    leverage: np.ndarray
    neighbor10: np.ndarray
    neighbor20: np.ndarray


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
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = make_base_features(data["X_train"])
    X_test = make_base_features(data["X_test"])
    scores = score_samples(X_train, data["y"], data["groups"])

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    specs = build_specs()
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        weights = make_weights(spec, scores)
        raw_pred = fit_predict_yj_ridge(
            X_train,
            data["y"],
            X_test,
            alpha=spec.alpha,
            n_components=spec.pca_components,
            sample_weight=weights,
        )
        pred = np.clip(raw_pred, 0, None)
        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        rows.append(
            diagnostics(
                spec,
                pred,
                np.asarray(raw_pred),
                current_best[1].to_numpy(dtype=float),
                path,
                weights,
                data["groups"],
            )
        )
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
        rows_sorted = sorted(rows_sorted, key=rank_key)

    write_outputs(out_dir, rows_sorted, weight_rows, scores, data["groups"])
    print_top(rows_sorted)
    print(f"saved robust-weight diagnostics: {out_dir}")


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
    return (X - mean) / np.where(std == 0, 1.0, std)


def build_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = [
        CandidateSpec(
            name="nir_rw_anchor_like_yj_top03_w0p85_pca20_ridge3500",
            score_key="yj",
            alpha=3500.0,
            top_pct=0.03,
            min_weight=0.85,
            gamma=0.0,
            mode="binary_tail",
        )
    ]
    score_keys = [
        "yj",
        "yj_lev",
        "yj_nn20",
        "yj_lev_nn20",
        "raw_yj_lev",
        "lev_nn20",
    ]
    for score_key in score_keys:
        for top_pct in [0.03, 0.05, 0.08, 0.12]:
            for min_weight in [0.70, 0.78, 0.85, 0.90]:
                for gamma in [0.5, 1.0, 2.0]:
                    for alpha in [3400.0, 3500.0, 3600.0]:
                        name = (
                            f"nir_rw_{score_key}_top{pct_tag(top_pct)}_"
                            f"w{weight_tag(min_weight)}_g{gamma_tag(gamma)}_"
                            f"pca20_ridge{int(alpha)}"
                        )
                        specs.append(
                            CandidateSpec(
                                name=name,
                                score_key=score_key,
                                alpha=alpha,
                                top_pct=top_pct,
                                min_weight=min_weight,
                                gamma=gamma,
                            )
                        )
    for score_key in ["yj_lev", "yj_lev_nn20"]:
        for top_pct in [0.05, 0.08]:
            for min_weight in [0.78, 0.85]:
                for pca_components in [18, 22, 25]:
                    name = (
                        f"nir_rw_{score_key}_top{pct_tag(top_pct)}_"
                        f"w{weight_tag(min_weight)}_g1_pca{pca_components}_ridge3500"
                    )
                    specs.append(
                        CandidateSpec(
                            name=name,
                            score_key=score_key,
                            pca_components=pca_components,
                            top_pct=top_pct,
                            min_weight=min_weight,
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


def score_samples(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> SampleScores:
    oof, residual_yj_abs = make_group_oof(X, y, groups)
    leverage, pca_scores = compute_leverage(X)
    return SampleScores(
        oof_pred=oof,
        residual_abs=np.abs(y - oof),
        residual_yj_abs=residual_yj_abs,
        leverage=leverage,
        neighbor10=compute_neighbor_inconsistency(pca_scores, y, k=10),
        neighbor20=compute_neighbor_inconsistency(pca_scores, y, k=20),
    )


def make_group_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oof = np.empty_like(y, dtype=float)
    residual_yj_abs = np.empty_like(y, dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pca = PCA(n_components=20, random_state=42)
        Z_train = pca.fit_transform(X[train_idx])
        Z_valid = pca.transform(X[valid_idx])
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        y_train_yj = transformer.fit_transform(y[train_idx].reshape(-1, 1)).ravel()
        y_valid_yj = transformer.transform(y[valid_idx].reshape(-1, 1)).ravel()
        model = Ridge(alpha=3500.0)
        model.fit(Z_train, y_train_yj)
        pred_yj = model.predict(Z_valid)
        pred = transformer.inverse_transform(pred_yj.reshape(-1, 1)).ravel()
        oof[valid_idx] = pred
        residual_yj_abs[valid_idx] = np.abs(y_valid_yj - pred_yj)
    return oof, residual_yj_abs


def compute_leverage(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = PCA(n_components=20, random_state=42).fit_transform(X)
    scaled = StandardScaler().fit_transform(scores)
    return np.sum(scaled**2, axis=1), scaled


def compute_neighbor_inconsistency(scores: np.ndarray, y: np.ndarray, *, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(scores)
    _, indices = nn.kneighbors(scores)
    return np.abs(y - np.median(y[indices[:, 1:]], axis=1))


def make_weights(spec: CandidateSpec, scores: SampleScores) -> np.ndarray:
    score = composite_score(spec.score_key, scores)
    rank = percentile_rank(score)
    tail_start = 1.0 - spec.top_pct
    if spec.mode == "binary_tail":
        penalty = (rank >= tail_start).astype(float)
    elif spec.mode == "tail_power":
        tail = np.clip((rank - tail_start) / max(spec.top_pct, 1e-12), 0.0, 1.0)
        penalty = tail**spec.gamma
    else:
        raise ValueError(f"unknown weight mode: {spec.mode}")
    weights = 1.0 - (1.0 - spec.min_weight) * penalty
    weights /= np.mean(weights)
    return weights


def composite_score(score_key: str, scores: SampleScores) -> np.ndarray:
    parts = {
        "raw": percentile_rank(scores.residual_abs),
        "yj": percentile_rank(scores.residual_yj_abs),
        "lev": percentile_rank(scores.leverage),
        "nn10": percentile_rank(scores.neighbor10),
        "nn20": percentile_rank(scores.neighbor20),
    }
    if score_key == "yj":
        return parts["yj"]
    if score_key == "yj_lev":
        return 0.70 * parts["yj"] + 0.30 * parts["lev"]
    if score_key == "yj_nn20":
        return 0.70 * parts["yj"] + 0.30 * parts["nn20"]
    if score_key == "yj_lev_nn20":
        return 0.55 * parts["yj"] + 0.25 * parts["lev"] + 0.20 * parts["nn20"]
    if score_key == "raw_yj_lev":
        return 0.35 * parts["raw"] + 0.45 * parts["yj"] + 0.20 * parts["lev"]
    if score_key == "lev_nn20":
        return 0.55 * parts["lev"] + 0.45 * parts["nn20"]
    raise ValueError(f"unknown score key: {score_key}")


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    return ranks / (len(values) - 1)


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
    raw_pred: np.ndarray,
    anchor: np.ndarray,
    path: Path,
    weights: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    diff = pred - anchor
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    effective_n = float((np.sum(weights) ** 2) / np.sum(weights**2))
    low_weight = weights < np.quantile(weights, 0.10)
    species_low = {
        str(g): int(np.sum(low_weight & (groups == g)))
        for g in sorted(np.unique(groups))
        if int(np.sum(low_weight & (groups == g))) > 0
    }
    max_low_weight_species_rate = max(
        float(np.sum(low_weight & (groups == g)) / np.sum(groups == g)) for g in np.unique(groups)
    )
    return {
        "experiment": spec.name,
        "submission_path": str(path),
        "score_key": spec.score_key,
        "alpha": spec.alpha,
        "pca_components": spec.pca_components,
        "top_pct": spec.top_pct,
        "min_weight_raw": spec.min_weight,
        "gamma": spec.gamma,
        "mode": spec.mode,
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
        "low_weight_by_species_json": json.dumps(species_low, ensure_ascii=False),
    }


def species_weight_rows(spec: CandidateSpec, weights: np.ndarray, groups: np.ndarray) -> list[dict[str, object]]:
    rows = []
    low_weight = weights < np.quantile(weights, 0.10)
    for group in sorted(np.unique(groups)):
        mask = groups == group
        rows.append(
            {
                "experiment": spec.name,
                "score_key": spec.score_key,
                "species": int(group),
                "n": int(np.sum(mask)),
                "mean_weight": float(np.mean(weights[mask])),
                "min_weight": float(np.min(weights[mask])),
                "low_weight_count": int(np.sum(mask & low_weight)),
                "low_weight_rate": float(np.sum(mask & low_weight) / np.sum(mask)),
            }
        )
    return rows


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    rmse = float(row["anchor_diff_rmse"])
    corr = float(row["anchor_corr"])
    max_abs = float(row["anchor_diff_max_abs"])
    top_delta = float(row["top_decile_delta"])
    range_ratio = float(row["range_ratio_vs_anchor"])
    species_rate = float(row["max_low_weight_species_rate"])
    cv_group = row.get("cv_group_species_rmse")
    cv_ex15 = row.get("cv_exclude15_rmse")
    hard_fail = 0.0
    if corr < 0.99995 or rmse < 0.08 or rmse > 0.55 or max_abs > 2.0:
        hard_fail += 10.0
    if top_delta < -1.0 or top_delta > 0.45:
        hard_fail += 6.0
    if range_ratio < 0.985 or range_ratio > 1.015:
        hard_fail += 5.0
    if species_rate > 0.35:
        hard_fail += 4.0
    if cv_group is not None and not pd.isna(cv_group) and float(cv_group) > 25.85:
        hard_fail += 4.0
    if cv_ex15 is not None and not pd.isna(cv_ex15) and float(cv_ex15) > 20.15:
        hard_fail += 4.0
    # Prefer real movement without collapsing the high-moisture side.
    movement = abs(rmse - 0.24)
    top_penalty = max(0.0, abs(top_delta) - 0.65)
    cv_penalty = 0.0
    if cv_group is not None and not pd.isna(cv_group):
        cv_penalty += max(0.0, float(cv_group) - 25.72)
    if cv_ex15 is not None and not pd.isna(cv_ex15):
        cv_penalty += max(0.0, float(cv_ex15) - 19.98)
    return (hard_fail, movement + top_penalty + cv_penalty, max_abs, species_rate, -corr)


def pick_cv_specs(rows_sorted: list[dict[str, object]], specs: list[CandidateSpec], cv_top: int) -> list[CandidateSpec]:
    spec_by_name = {spec.name: spec for spec in specs}
    return [spec_by_name[str(row["experiment"])] for row in rows_sorted[:cv_top]]


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
        scores = score_samples(X_train, y_train, groups[train_idx])
        weights = make_weights(spec, scores)
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


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, object]],
    weight_rows: list[dict[str, object]],
    scores: SampleScores,
    groups: np.ndarray,
) -> None:
    with (out_dir / "robust_weight_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "robust_weight_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with (out_dir / "species_weight_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)
    score_df = pd.DataFrame(
        {
            "species": groups,
            "oof_pred": scores.oof_pred,
            "residual_abs": scores.residual_abs,
            "residual_yj_abs": scores.residual_yj_abs,
            "leverage": scores.leverage,
            "neighbor10": scores.neighbor10,
            "neighbor20": scores.neighbor20,
        }
    )
    score_df.to_csv(out_dir / "sample_scores.csv", index=False)


def print_one(row: dict[str, object]) -> None:
    print(
        f"{row['experiment']}: rmse={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} corr={row['anchor_corr']:.6f} "
        f"low={row['bottom_decile_delta']:.4f} top={row['top_decile_delta']:.4f} "
        f"range={row['range_ratio_vs_anchor']:.4f} effN={row['effective_n_ratio']:.4f}"
    )


def print_top(rows: list[dict[str, object]]) -> None:
    print("\nTop robust-weight candidates:")
    for row in rows[:20]:
        print_one(row)
        if row.get("cv_group_species_rmse") is not None:
            print(
                f"  CV group={row['cv_group_species_rmse']:.4f}+/-{row['cv_group_species_std']:.4f} "
                f"exclude15={row['cv_exclude15_rmse']:.4f}+/-{row['cv_exclude15_std']:.4f}"
            )


if __name__ == "__main__":
    main()
