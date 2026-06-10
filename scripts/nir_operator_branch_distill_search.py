#!/usr/bin/env python3
"""Distill operator/preprocessing branch signals into tiny corrections of the 13.986 anchor."""

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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "nir_operator_branch_distill"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"


@dataclass(frozen=True)
class BranchSpec:
    name: str
    preprocess: str
    model: str
    n_components: int
    alpha: float = 3500.0
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class DistillSpec:
    shrink: float
    clip: float
    mean_center: bool = True


def main() -> None:
    data = load_data()
    anchor = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor[0].to_numpy()):
        raise ValueError("anchor sample order mismatch")
    anchor_test = anchor[1].to_numpy(float)

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF")
    anchor_oof = make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    residual = data["y"] - anchor_oof
    base_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={base_oof_rmse:.6f}")

    rows: list[dict[str, object]] = []
    specs = build_branch_specs()
    distill_specs = [
        DistillSpec(0.02, 0.08),
        DistillSpec(0.03, 0.08),
        DistillSpec(0.04, 0.08),
        DistillSpec(0.02, 0.10),
        DistillSpec(0.03, 0.10),
        DistillSpec(0.04, 0.10),
        DistillSpec(0.03, 0.12),
        DistillSpec(0.04, 0.12),
        DistillSpec(0.05, 0.10),
        DistillSpec(0.08, 0.10),
        DistillSpec(0.05, 0.15),
        DistillSpec(0.08, 0.15),
        DistillSpec(0.10, 0.15),
        DistillSpec(0.05, 0.20),
        DistillSpec(0.08, 0.20),
    ]

    for branch in specs:
        print(f"branch {branch.name}", flush=True)
        branch_oof = make_branch_oof(branch, data["X_train"], data["y"], data["groups"])
        branch_test = fit_predict_branch(branch, data["X_train"], data["y"], data["X_test"])
        signal_oof = branch_oof - anchor_oof
        signal_test = branch_test - anchor_test
        beta = fit_beta(signal_oof, residual)
        signal_corr = safe_corr(signal_oof, residual)
        branch_rmse = rmse(data["y"], branch_oof)
        for distill in distill_specs:
            correction_oof = make_correction(beta, signal_oof, distill)
            corrected_oof = anchor_oof + correction_oof
            correction_test = make_correction(beta, signal_test, distill)
            pred = np.clip(anchor_test + correction_test, 0, None)
            name = (
                f"nir_opdist_{branch.name}_b{tag(beta)}_"
                f"s{tag(distill.shrink)}_c{tag(distill.clip)}_mc1"
            )
            path = candidate_dir / f"{name}.csv"
            pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
            row = diagnostics(
                name=name,
                branch=branch,
                distill=distill,
                pred=pred,
                anchor=anchor_test,
                oof_pred=corrected_oof,
                y=data["y"],
                anchor_oof=anchor_oof,
                branch_oof=branch_oof,
                beta=beta,
                signal_corr=signal_corr,
                branch_oof_rmse=branch_rmse,
                base_oof_rmse=base_oof_rmse,
                path=path,
            )
            rows.append(row)
            print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "operator_branch_distill_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "operator_branch_distill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop candidates:")
    for row in rows_sorted[:25]:
        print(
            f"{row['experiment']}: branch={row['branch']} rmse={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
            f"sigcorr={row['signal_residual_corr']:.4f}"
        )
    print(f"saved operator-branch distillation diagnostics: {out_dir}")


def build_branch_specs() -> list[BranchSpec]:
    specs: list[BranchSpec] = []
    for n_components in [15, 18, 22, 25]:
        for alpha in [3300.0, 3500.0, 3700.0]:
            specs.append(
                BranchSpec(
                    name=f"ridge_yj_sg9snv_pca{n_components}_a{int(alpha)}",
                    preprocess="sg9_snv",
                    model="ridge_yj",
                    n_components=n_components,
                    alpha=alpha,
                )
            )
    for deriv in ["d1", "d2"]:
        for n_components in [22, 25]:
            specs.append(
                BranchSpec(
                    name=f"ridge_yj_stack{deriv}_pca{n_components}_a3500",
                    preprocess=f"sg9_snv_stack_{deriv}",
                    model="ridge_yj",
                    n_components=n_components,
                    alpha=3500.0,
                )
            )
    for prep in [
        "raw",
        "sg7_snv",
        "sg9_snv",
        "sg11_snv",
        "msc_sg7",
        "msc_sg9",
        "msc_sg11",
        "sg9_snv_d1",
    ]:
        for model in ["pls_raw", "pls_yj"]:
            for n_components in [3, 4, 5, 6, 8, 10, 12]:
                specs.append(
                    BranchSpec(
                        name=f"{model}_{prep}_c{n_components}",
                        preprocess=prep,
                        model=model,
                        n_components=n_components,
                    )
                )
    interval_specs = [
        ("raw", 0, 6),
        ("raw", 4, 12),
        ("raw", 8, 16),
        ("raw", 12, 20),
        ("sg9_snv", 0, 6),
        ("sg9_snv", 4, 12),
        ("sg9_snv", 8, 16),
        ("sg9_snv", 12, 20),
        ("msc_sg9", 0, 6),
        ("msc_sg9", 4, 12),
        ("msc_sg9", 8, 16),
        ("msc_sg9", 12, 20),
    ]
    for prep, start, end in interval_specs:
        width = end - start
        for n_components in [2, 3, 4, 5]:
            if n_components >= width:
                continue
            specs.append(
                BranchSpec(
                    name=f"pls_raw_{prep}_i{start}_{end}_c{n_components}",
                    preprocess=prep,
                    model="pls_raw",
                    n_components=n_components,
                    start=start,
                    end=end,
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
        "test_species": test["species number"].to_numpy(),
    }


def is_float_like(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def make_current_anchor_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        X_train_raw = X[train_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx]
        valid_base = fit_predict_base_yj(X_train_raw, y_train, X[valid_idx])
        inner_oof = make_base_oof(X_train_raw, y_train, groups_train)
        params = fit_affine_params(inner_oof, y_train)
        pred[valid_idx] = apply_affine_correction(valid_base, params, shrink=0.10, clip=0.4, mean_center=True)
    return np.clip(pred, 0, None)


def make_base_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx] = fit_predict_base_yj(X[train_idx], y[train_idx], X[valid_idx])
    return np.clip(pred, 0, None)


def fit_predict_base_yj(X_train_raw: np.ndarray, y: np.ndarray, X_pred_raw: np.ndarray) -> np.ndarray:
    spec = BranchSpec("base", "sg9_snv", "ridge_yj", 20, 3500.0)
    return fit_predict_branch(spec, X_train_raw, y, X_pred_raw)


def fit_affine_params(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(pred, y, deg=1)
    return float(intercept), float(slope)


def apply_affine_correction(
    pred: np.ndarray,
    params: tuple[float, float],
    *,
    shrink: float,
    clip: float,
    mean_center: bool,
) -> np.ndarray:
    intercept, slope = params
    delta = shrink * (intercept + slope * pred - pred)
    delta = np.clip(delta, -clip, clip)
    if mean_center:
        delta = delta - np.mean(delta)
    return pred + delta


def make_branch_oof(branch: BranchSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx] = fit_predict_branch(branch, X[train_idx], y[train_idx], X[valid_idx])
    return np.clip(pred, 0, None)


def fit_predict_branch(
    branch: BranchSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> np.ndarray:
    X_train, X_pred = preprocess_pair(X_train_raw, X_pred_raw, branch.preprocess)
    if branch.start is not None or branch.end is not None:
        if branch.start is None or branch.end is None:
            raise ValueError("interval branch requires both start and end")
        X_train = X_train[:, branch.start : branch.end]
        X_pred = X_pred[:, branch.start : branch.end]
    if branch.model == "ridge_yj":
        return fit_predict_ridge_yj(X_train, y, X_pred, branch.n_components, branch.alpha)
    if branch.model == "pls_raw":
        pls = PLSRegression(n_components=min(branch.n_components, X_train.shape[0] - 1, X_train.shape[1]), scale=True)
        pls.fit(X_train, y)
        return pls.predict(X_pred).ravel()
    if branch.model == "pls_yj":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
        pls = PLSRegression(n_components=min(branch.n_components, X_train.shape[0] - 1, X_train.shape[1]), scale=True)
        pls.fit(X_train, yt)
        pred_t = pls.predict(X_pred).ravel()
        return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()
    raise ValueError(branch.model)


def fit_predict_ridge_yj(
    X_train: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
    n_components: int,
    alpha: float,
) -> np.ndarray:
    pca = PCA(n_components=min(n_components, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_pred = pca.transform(X_pred)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    model = Ridge(alpha=alpha)
    model.fit(Z_train, yt)
    pred_t = model.predict(Z_pred)
    return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()


def preprocess_pair(X_train: np.ndarray, X_pred: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "raw":
        return X_train, X_pred
    if name == "sg7_snv":
        return snv(sg(X_train, 7)), snv(sg(X_pred, 7))
    if name == "sg9_snv":
        return snv(sg(X_train, 9)), snv(sg(X_pred, 9))
    if name == "sg11_snv":
        return snv(sg(X_train, 11)), snv(sg(X_pred, 11))
    if name == "sg9_snv_d1":
        return sg(snv(X_train), 11, deriv=1), sg(snv(X_pred), 11, deriv=1)
    if name == "sg9_snv_stack_d1":
        return (
            np.hstack([snv(sg(X_train, 9)), sg(snv(X_train), 11, deriv=1)]),
            np.hstack([snv(sg(X_pred, 9)), sg(snv(X_pred), 11, deriv=1)]),
        )
    if name == "sg9_snv_stack_d2":
        return (
            np.hstack([snv(sg(X_train, 9)), sg(snv(X_train), 11, deriv=2)]),
            np.hstack([snv(sg(X_pred, 9)), sg(snv(X_pred), 11, deriv=2)]),
        )
    if name == "msc_sg9":
        train = sg(X_train, 9)
        pred = sg(X_pred, 9)
        return msc_pair(train, pred)
    if name == "msc_sg7":
        train = sg(X_train, 7)
        pred = sg(X_pred, 7)
        return msc_pair(train, pred)
    if name == "msc_sg11":
        train = sg(X_train, 11)
        pred = sg(X_pred, 11)
        return msc_pair(train, pred)
    raise ValueError(name)


def sg(X: np.ndarray, window: int, deriv: int = 0) -> np.ndarray:
    return savgol_filter(X, window, 2, deriv=deriv, axis=1, mode="interp")


def snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / np.where(std == 0, 1.0, std)


def msc_pair(X_train: np.ndarray, X_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = X_train.mean(axis=0)
    return msc_transform(X_train, reference), msc_transform(X_pred, reference)


def msc_transform(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_centered = reference - reference.mean()
    denom = float(np.dot(ref_centered, ref_centered))
    if denom == 0.0:
        raise ValueError("MSC reference variance is zero")
    out = np.empty_like(X)
    for i, row in enumerate(X):
        row_centered = row - row.mean()
        slope = float(np.dot(row_centered, ref_centered) / denom)
        intercept = float(row.mean() - slope * reference.mean())
        if abs(slope) < 1e-12:
            slope = 1.0
        out[i] = (row - intercept) / slope
    return out


def fit_beta(signal: np.ndarray, residual: np.ndarray) -> float:
    denom = float(np.dot(signal, signal))
    if denom < 1e-12:
        return 0.0
    beta = float(np.dot(signal, residual) / denom)
    return float(np.clip(beta, -1.0, 1.0))


def make_correction(beta: float, signal: np.ndarray, spec: DistillSpec) -> np.ndarray:
    delta = spec.shrink * beta * signal
    delta = np.clip(delta, -spec.clip, spec.clip)
    if spec.mean_center:
        delta = delta - np.mean(delta)
    return delta


def diagnostics(
    *,
    name: str,
    branch: BranchSpec,
    distill: DistillSpec,
    pred: np.ndarray,
    anchor: np.ndarray,
    oof_pred: np.ndarray,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    branch_oof: np.ndarray,
    beta: float,
    signal_corr: float,
    branch_oof_rmse: float,
    base_oof_rmse: float,
    path: Path,
) -> dict[str, object]:
    diff = pred - anchor
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    species = pd.read_csv(TEST_PATH, encoding="cp932")["species number"].to_numpy()
    species_shift = max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))
    corrected_oof_rmse = rmse(y, oof_pred)
    return {
        "experiment": name,
        "branch": branch.name,
        "preprocess": branch.preprocess,
        "model": branch.model,
        "n_components": branch.n_components,
        "alpha": branch.alpha,
        "start": branch.start,
        "end": branch.end,
        "submission_path": str(path),
        "shrink": distill.shrink,
        "clip": distill.clip,
        "mean_center": distill.mean_center,
        "beta": beta,
        "signal_residual_corr": signal_corr,
        "anchor_oof_rmse": base_oof_rmse,
        "branch_oof_rmse": branch_oof_rmse,
        "corrected_oof_rmse": corrected_oof_rmse,
        "oof_delta_vs_anchor": corrected_oof_rmse - base_oof_rmse,
        "branch_delta_vs_anchor": branch_oof_rmse - base_oof_rmse,
        "anchor_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": safe_corr(pred, anchor),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "range_ratio_vs_anchor": float((np.max(pred) - np.min(pred)) / (np.max(anchor) - np.min(anchor))),
        "max_abs_species_mean_shift": species_shift,
        "negative_count": int(np.sum(pred < 0)),
    }


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    rmse_diff = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    mean = abs(float(row["anchor_diff_mean"]))
    low = abs(float(row["bottom_decile_delta"]))
    top = abs(float(row["top_decile_delta"]))
    oof_delta = float(row["oof_delta_vs_anchor"])
    penalty = 0.0
    if rmse_diff < 0.03 or rmse_diff > 0.12:
        penalty += 5.0
    if max_abs > 0.25 or species > 0.06 or mean > 0.015 or low > 0.12 or top > 0.12:
        penalty += 5.0
    if oof_delta > 0.02:
        penalty += 3.0
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


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
        f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
        f"corr={row['signal_residual_corr']:.4f}"
    )


if __name__ == "__main__":
    main()
