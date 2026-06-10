#!/usr/bin/env python3
"""RBF kernel latent branch distillation around the current NIR anchor."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVR

import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_kernel_branch_distill"
CURRENT_ANCHOR = ROOT / "data" / "submissions" / "nir_yj_oof_affine_s0p10_mc1.csv"


@dataclass(frozen=True)
class KernelSpec:
    name: str
    preprocess: str
    model: str
    target: str
    n_components: int
    alpha: float = math.nan
    gamma_mult: float = 1.0
    c: float = math.nan
    epsilon: float = math.nan


def main() -> None:
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
    base_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={base_oof_rmse:.6f}")

    rows: list[dict[str, object]] = []
    for spec in build_specs():
        print(f"branch {spec.name}", flush=True)
        branch_oof, oof_meta = make_branch_oof(spec, data["X_train"], data["y"], data["groups"])
        branch_test, test_meta = fit_predict_branch(spec, data["X_train"], data["y"], data["X_test"])
        signal_oof = branch_oof - anchor_oof
        signal_test = branch_test - anchor_test
        beta = op.fit_beta(signal_oof, residual)
        signal_corr = op.safe_corr(signal_oof, residual)
        branch_oof_rmse = rmse(data["y"], branch_oof)
        for distill in build_distills():
            correction_oof = op.make_correction(beta, signal_oof, distill)
            corrected_oof = anchor_oof + correction_oof
            correction_test = op.make_correction(beta, signal_test, distill)
            pred = np.clip(anchor_test + correction_test, 0, None)
            name = (
                f"nir_kernel_{spec.name}_b{op.tag(beta)}_"
                f"s{op.tag(distill.shrink)}_c{op.tag(distill.clip)}_mc1"
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
                branch_oof=branch_oof,
                anchor_oof_rmse=base_oof_rmse,
                branch_oof_rmse=branch_oof_rmse,
                beta=beta,
                signal_corr=signal_corr,
                path=path,
                oof_meta=oof_meta,
                test_meta=test_meta,
            )
            rows.append(row)
            print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "kernel_branch_distill_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "kernel_branch_distill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop candidates:")
    for row in rows_sorted[:30]:
        print(
            f"{row['experiment']}: branch={row['branch']} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
            f"corr={row['signal_residual_corr']:.4f} gamma={row['test_gamma']:.5f}"
        )
    print(f"saved kernel branch diagnostics: {out_dir}")


def build_specs() -> list[KernelSpec]:
    specs: list[KernelSpec] = []
    for preprocess in ["sg9_snv", "msc_sg9"]:
        for n_components in [8, 12, 15]:
            for alpha in [10.0, 30.0, 100.0]:
                for gamma_mult in [0.5, 1.0, 2.0]:
                    specs.append(
                        KernelSpec(
                            name=(
                                f"krr_raw_{preprocess}_p{n_components}_"
                                f"a{int(alpha)}_gm{op.tag(gamma_mult)}"
                            ),
                            preprocess=preprocess,
                            model="krr",
                            target="raw",
                            n_components=n_components,
                            alpha=alpha,
                            gamma_mult=gamma_mult,
                        )
                    )
            for c in [1.0, 3.0]:
                for epsilon in [0.2, 0.5]:
                    specs.append(
                        KernelSpec(
                            name=(
                                f"svr_raw_{preprocess}_p{n_components}_"
                                f"C{op.tag(c)}_e{op.tag(epsilon)}"
                            ),
                            preprocess=preprocess,
                            model="svr",
                            target="raw",
                            n_components=n_components,
                            c=c,
                            epsilon=epsilon,
                        )
                    )
        specs.append(
            KernelSpec(
                name=f"krr_yj_{preprocess}_p12_a30_gm1",
                preprocess=preprocess,
                model="krr",
                target="yj",
                n_components=12,
                alpha=30.0,
                gamma_mult=1.0,
            )
        )
    return specs


def build_distills() -> list[op.DistillSpec]:
    return [
        op.DistillSpec(0.010, 0.08),
        op.DistillSpec(0.015, 0.08),
        op.DistillSpec(0.020, 0.08),
        op.DistillSpec(0.015, 0.10),
        op.DistillSpec(0.020, 0.10),
        op.DistillSpec(0.030, 0.10),
    ]


def make_branch_oof(
    spec: KernelSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    pred = np.empty(len(y), dtype=float)
    gammas: list[float] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        fold_pred, meta = fit_predict_branch(spec, X[train_idx], y[train_idx], X[valid_idx])
        pred[valid_idx] = fold_pred
        gammas.append(float(meta["gamma"]))
    return np.clip(pred, 0, None), {
        "oof_gamma_median": float(np.median(gammas)),
        "oof_gamma_min": float(np.min(gammas)),
        "oof_gamma_max": float(np.max(gammas)),
    }


def fit_predict_branch(
    spec: KernelSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    pca = PCA(n_components=min(spec.n_components, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_pred = pca.transform(X_pred)
    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Z_train)
    Z_pred = scaler.transform(Z_pred)
    gamma = estimate_gamma(Z_train, spec.gamma_mult)

    transformer: PowerTransformer | None = None
    y_fit = y
    if spec.target == "yj":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        y_fit = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    elif spec.target != "raw":
        raise ValueError(spec.target)

    if spec.model == "krr":
        model = KernelRidge(alpha=spec.alpha, kernel="rbf", gamma=gamma)
    elif spec.model == "svr":
        model = SVR(kernel="rbf", C=spec.c, epsilon=spec.epsilon, gamma=gamma)
    else:
        raise ValueError(spec.model)
    model.fit(Z_train, y_fit)
    pred = model.predict(Z_pred)
    if transformer is not None:
        pred = transformer.inverse_transform(np.asarray(pred).reshape(-1, 1)).ravel()
    return np.asarray(pred, dtype=float).ravel(), {"gamma": gamma}


def estimate_gamma(Z_train: np.ndarray, multiplier: float) -> float:
    if len(Z_train) > 600:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Z_train), size=600, replace=False)
        sample = Z_train[idx]
    else:
        sample = Z_train
    distances = pdist(sample, metric="euclidean")
    positive = distances[distances > 1e-12]
    median = float(np.median(positive)) if len(positive) else 1.0
    return float(multiplier / max(median * median, 1e-12))


def diagnostics(
    *,
    name: str,
    spec: KernelSpec,
    distill: op.DistillSpec,
    pred: np.ndarray,
    anchor: np.ndarray,
    y: np.ndarray,
    corrected_oof: np.ndarray,
    branch_oof: np.ndarray,
    anchor_oof_rmse: float,
    branch_oof_rmse: float,
    beta: float,
    signal_corr: float,
    path: Path,
    oof_meta: dict[str, object],
    test_meta: dict[str, object],
) -> dict[str, object]:
    diff = pred - anchor
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    species_shift = max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))
    corrected_oof_rmse = rmse(y, corrected_oof)
    return {
        "experiment": name,
        "branch": spec.name,
        "preprocess": spec.preprocess,
        "model": spec.model,
        "target": spec.target,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "gamma_mult": spec.gamma_mult,
        "C": spec.c,
        "epsilon": spec.epsilon,
        "submission_path": str(path),
        "shrink": distill.shrink,
        "clip": distill.clip,
        "mean_center": distill.mean_center,
        "beta": beta,
        "signal_residual_corr": signal_corr,
        "anchor_oof_rmse": anchor_oof_rmse,
        "branch_oof_rmse": branch_oof_rmse,
        "corrected_oof_rmse": corrected_oof_rmse,
        "oof_delta_vs_anchor": corrected_oof_rmse - anchor_oof_rmse,
        "branch_delta_vs_anchor": branch_oof_rmse - anchor_oof_rmse,
        "anchor_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": op.safe_corr(pred, anchor),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "range_ratio_vs_anchor": float((np.max(pred) - np.min(pred)) / (np.max(anchor) - np.min(anchor))),
        "max_abs_species_mean_shift": species_shift,
        "negative_count": int(np.sum(pred < 0)),
        "oof_gamma_median": oof_meta["oof_gamma_median"],
        "oof_gamma_min": oof_meta["oof_gamma_min"],
        "oof_gamma_max": oof_meta["oof_gamma_max"],
        "test_gamma": test_meta["gamma"],
    }


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    diff = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    mean = abs(float(row["anchor_diff_mean"]))
    oof_delta = float(row["oof_delta_vs_anchor"])
    beta = float(row["beta"])
    corr = float(row["signal_residual_corr"])
    penalty = 0.0
    if diff < 0.02 or diff > 0.12:
        penalty += 5.0
    if max_abs > 0.25 or species > 0.05 or mean > 0.015:
        penalty += 5.0
    if beta <= 0.0 or corr <= 0.0:
        penalty += 10.0
    if float(row["branch_oof_rmse"]) > float(row["anchor_oof_rmse"]) + 15.0:
        penalty += 3.0
    return (penalty + max(oof_delta, 0.0) + abs(diff - 0.06) + species, max_abs, species)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
        f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
        f"corr={row['signal_residual_corr']:.4f}"
    )


if __name__ == "__main__":
    main()
