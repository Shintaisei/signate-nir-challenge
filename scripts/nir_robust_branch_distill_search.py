#!/usr/bin/env python3
"""Distill robust/golden-subset branch signals into the current Public-best anchor."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
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
from sklearn.preprocessing import PowerTransformer


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_SCRIPT = ROOT / "scripts" / "nir_operator_branch_distill_search.py"
OUTPUT_ROOT = ROOT / "outputs" / "nir_robust_branch_distill"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"


@dataclass(frozen=True)
class RobustSpec:
    name: str
    mode: str
    top_pct: float
    min_weight: float
    alpha: float = 3500.0
    n_components: int = 20


@dataclass(frozen=True)
class DistillSpec:
    shrink: float
    clip: float
    mean_center: bool = True


def main() -> None:
    op = load_operator_module()
    data = op.load_data()
    anchor = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor[0].to_numpy()):
        raise ValueError("anchor sample order mismatch")
    anchor_test = anchor[1].to_numpy(float)

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF")
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    residual = data["y"] - anchor_oof
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}")

    base_scores = make_oof_scores(data["X_train"], data["y"], data["groups"])
    specs = build_specs()
    distills = [
        DistillSpec(0.02, 0.08),
        DistillSpec(0.03, 0.08),
        DistillSpec(0.02, 0.10),
        DistillSpec(0.03, 0.10),
        DistillSpec(0.04, 0.10),
        DistillSpec(0.03, 0.12),
        DistillSpec(0.04, 0.12),
        DistillSpec(0.05, 0.12),
    ]

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for spec in specs:
        print(f"branch {spec.name}", flush=True)
        full_weights = make_weights(spec, base_scores)
        branch_oof = make_branch_oof(spec, data["X_train"], data["y"], data["groups"])
        branch_test = fit_predict_weighted_branch(spec, data["X_train"], data["y"], data["X_test"], full_weights)
        signal_oof = branch_oof - anchor_oof
        signal_test = branch_test - anchor_test
        beta = fit_beta(signal_oof, residual)
        signal_corr = safe_corr(signal_oof, residual)
        branch_oof_rmse = rmse(data["y"], branch_oof)
        weight_rows.extend(make_weight_rows(spec, full_weights, data["groups"]))
        for distill in distills:
            correction_oof = make_correction(beta, signal_oof, distill)
            corrected_oof = anchor_oof + correction_oof
            correction_test = make_correction(beta, signal_test, distill)
            pred = np.clip(anchor_test + correction_test, 0, None)
            name = (
                f"nir_rbd_{spec.name}_b{tag(beta)}_"
                f"s{tag(distill.shrink)}_c{tag(distill.clip)}_mc1"
            )
            path = candidate_dir / f"{name}.csv"
            pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
            row = diagnostics(
                name=name,
                spec=spec,
                distill=distill,
                pred=pred,
                anchor=anchor_test,
                y=data["y"],
                corrected_oof=corrected_oof,
                anchor_oof_rmse=anchor_oof_rmse,
                branch_oof_rmse=branch_oof_rmse,
                beta=beta,
                signal_corr=signal_corr,
                path=path,
            )
            rows.append(row)
            print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "robust_branch_distill_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "robust_branch_distill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    with (out_dir / "robust_branch_weights.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)

    print("\nTop candidates:")
    for row in rows_sorted[:25]:
        print(
            f"{row['experiment']}: mode={row['mode']} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
            f"corr={row['signal_residual_corr']:.4f}"
        )
    print(f"saved robust-branch distillation diagnostics: {out_dir}")


def load_operator_module():
    spec = importlib.util.spec_from_file_location("nir_operator_branch_distill_search", OPERATOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load operator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_specs() -> list[RobustSpec]:
    specs: list[RobustSpec] = []
    for mode in ["resid_yj", "leverage", "nn_inconsistency", "resid_lev", "resid_nn", "resid_lev_nn"]:
        for top_pct in [0.02, 0.03, 0.05, 0.08, 0.12]:
            for min_weight in [0.65, 0.75, 0.85, 0.92]:
                specs.append(
                    RobustSpec(
                        name=f"{mode}_top{pct_tag(top_pct)}_w{tag(min_weight)}",
                        mode=mode,
                        top_pct=top_pct,
                        min_weight=min_weight,
                    )
                )
    return specs


def make_oof_scores(X_raw: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    X = make_features(X_raw)
    oof = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred, _ = fit_predict_base(X[train_idx], y[train_idx], X[valid_idx], np.ones(len(train_idx)))
        oof[valid_idx] = pred
    residual = np.abs(y - oof)

    pca = PCA(n_components=min(20, X.shape[0] - 1, X.shape[1]), random_state=42)
    Z = pca.fit_transform(X)
    leverage = np.sum((Z / np.where(np.std(Z, axis=0) == 0, 1.0, np.std(Z, axis=0))) ** 2, axis=1)

    nn = NearestNeighbors(n_neighbors=min(21, len(y)), metric="euclidean")
    nn.fit(Z)
    indices = nn.kneighbors(Z, return_distance=False)[:, 1:]
    neighbor_mean = np.array([np.mean(y[idx]) for idx in indices])
    nn_inconsistency = np.abs(y - neighbor_mean)

    return {
        "resid_yj": scale01(residual),
        "leverage": scale01(leverage),
        "nn_inconsistency": scale01(nn_inconsistency),
        "resid_lev": scale01(0.65 * scale01(residual) + 0.35 * scale01(leverage)),
        "resid_nn": scale01(0.65 * scale01(residual) + 0.35 * scale01(nn_inconsistency)),
        "resid_lev_nn": scale01(
            0.50 * scale01(residual) + 0.25 * scale01(leverage) + 0.25 * scale01(nn_inconsistency)
        ),
    }


def make_branch_oof(spec: RobustSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        fold_scores = make_oof_scores(X[train_idx], y[train_idx], groups[train_idx])
        weights = make_weights(spec, fold_scores)
        pred[valid_idx] = fit_predict_weighted_branch(spec, X[train_idx], y[train_idx], X[valid_idx], weights)
    return np.clip(pred, 0, None)


def fit_predict_weighted_branch(
    spec: RobustSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    X_train = make_features(X_train_raw)
    X_pred = make_features(X_pred_raw)
    return fit_predict_base(X_train, y, X_pred, weights, spec.n_components, spec.alpha)[0]


def fit_predict_base(
    X_train: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
    weights: np.ndarray,
    n_components: int = 20,
    alpha: float = 3500.0,
) -> tuple[np.ndarray, Ridge]:
    pca = PCA(n_components=min(n_components, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_pred = pca.transform(X_pred)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    model = Ridge(alpha=alpha)
    model.fit(Z_train, yt, sample_weight=weights)
    pred_t = model.predict(Z_pred)
    pred = transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()
    return pred, model


def make_features(X: np.ndarray) -> np.ndarray:
    return snv(savgol_filter(X, 9, 2, axis=1, mode="interp"))


def snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / np.where(std == 0, 1.0, std)


def make_weights(spec: RobustSpec, scores: dict[str, np.ndarray]) -> np.ndarray:
    score = scores[spec.mode]
    threshold = np.quantile(score, 1.0 - spec.top_pct)
    tail = np.clip((score - threshold) / max(1.0 - threshold, 1e-12), 0, 1)
    weights = 1.0 - (1.0 - spec.min_weight) * tail
    return np.clip(weights, spec.min_weight, 1.0)


def make_weight_rows(spec: RobustSpec, weights: np.ndarray, groups: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for group in sorted(np.unique(groups)):
        mask = groups == group
        rows.append(
            {
                "experiment": spec.name,
                "species": int(group),
                "n": int(np.sum(mask)),
                "mean_weight": float(np.mean(weights[mask])),
                "min_weight": float(np.min(weights[mask])),
                "downweighted_count": int(np.sum(weights[mask] < 0.999)),
            }
        )
    return rows


def scale01(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def fit_beta(signal: np.ndarray, residual: np.ndarray) -> float:
    denom = float(np.dot(signal, signal))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(float(np.dot(signal, residual) / denom), -1.0, 1.0))


def make_correction(beta: float, signal: np.ndarray, spec: DistillSpec) -> np.ndarray:
    delta = spec.shrink * beta * signal
    delta = np.clip(delta, -spec.clip, spec.clip)
    if spec.mean_center:
        delta = delta - np.mean(delta)
    return delta


def diagnostics(
    *,
    name: str,
    spec: RobustSpec,
    distill: DistillSpec,
    pred: np.ndarray,
    anchor: np.ndarray,
    y: np.ndarray,
    corrected_oof: np.ndarray,
    anchor_oof_rmse: float,
    branch_oof_rmse: float,
    beta: float,
    signal_corr: float,
    path: Path,
) -> dict[str, object]:
    diff = pred - anchor
    species = pd.read_csv(ROOT / "data" / "raw" / "test.csv", encoding="cp932")["species number"].to_numpy()
    species_shift = max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    corrected_oof_rmse = rmse(y, corrected_oof)
    return {
        "experiment": name,
        "branch": spec.name,
        "mode": spec.mode,
        "top_pct": spec.top_pct,
        "min_weight": spec.min_weight,
        "alpha": spec.alpha,
        "n_components": spec.n_components,
        "shrink": distill.shrink,
        "clip": distill.clip,
        "mean_center": distill.mean_center,
        "beta": beta,
        "signal_residual_corr": signal_corr,
        "anchor_oof_rmse": anchor_oof_rmse,
        "branch_oof_rmse": branch_oof_rmse,
        "corrected_oof_rmse": corrected_oof_rmse,
        "oof_delta_vs_anchor": corrected_oof_rmse - anchor_oof_rmse,
        "anchor_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": safe_corr(pred, anchor),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "max_abs_species_mean_shift": species_shift,
        "negative_count": int(np.sum(pred < 0)),
        "submission_path": str(path),
    }


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    rmse_diff = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    mean = abs(float(row["anchor_diff_mean"]))
    low = abs(float(row["bottom_decile_delta"]))
    top = abs(float(row["top_decile_delta"]))
    oof_delta = float(row["oof_delta_vs_anchor"])
    corr = abs(float(row["signal_residual_corr"]))
    penalty = 0.0
    if rmse_diff < 0.03 or rmse_diff > 0.12:
        penalty += 5.0
    if max_abs > 0.25 or species > 0.06 or mean > 0.015 or low > 0.12 or top > 0.12:
        penalty += 5.0
    if corr < 0.05:
        penalty += 1.0
    return (penalty + max(oof_delta, 0.0) + abs(rmse_diff - 0.06) + species, max_abs, species)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def pct_tag(value: float) -> str:
    return tag(value).replace("0p", "")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
        f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
        f"corr={row['signal_residual_corr']:.4f}"
    )


if __name__ == "__main__":
    main()
