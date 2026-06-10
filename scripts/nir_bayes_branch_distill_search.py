#!/usr/bin/env python3
"""Bayesian/robust linear branch distillation around the current NIR anchor."""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ARDRegression, BayesianRidge, HuberRegressor, LassoLarsIC
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_bayes_branch_distill"
CURRENT_ANCHOR = ROOT / "data" / "submissions" / "nir_yj_oof_affine_s0p10_mc1.csv"


@dataclass(frozen=True)
class BayesSpec:
    name: str
    preprocess: str
    model: str
    target: str
    n_components: int
    alpha: float = math.nan
    start: int | None = None
    end: int | None = None


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
                f"nir_bayes_{spec.name}_b{op.tag(beta)}_"
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
                oof_pred=corrected_oof,
                y=data["y"],
                anchor_oof=anchor_oof,
                branch_oof=branch_oof,
                beta=beta,
                signal_corr=signal_corr,
                branch_oof_rmse=branch_oof_rmse,
                base_oof_rmse=base_oof_rmse,
                path=path,
                oof_meta=oof_meta,
                test_meta=test_meta,
            )
            rows.append(row)
            print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "bayes_branch_distill_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "bayes_branch_distill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop candidates:")
    for row in rows_sorted[:30]:
        print(
            f"{row['experiment']}: branch={row['branch']} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
            f"corr={row['signal_residual_corr']:.4f} active={row['active_coef_median']}"
        )
    print(f"saved bayes branch diagnostics: {out_dir}")


def build_specs() -> list[BayesSpec]:
    specs: list[BayesSpec] = []
    for preprocess in ["msc_sg9", "sg9_snv"]:
        for target in ["raw", "yj"]:
            for n_components in [15, 20, 25]:
                for model in ["bayesridge", "ard", "lassolars_aic", "lassolars_bic", "huber"]:
                    specs.append(
                        BayesSpec(
                            name=f"{model}_{target}_{preprocess}_p{n_components}",
                            preprocess=preprocess,
                            model=model,
                            target=target,
                            n_components=n_components,
                        )
                    )
    return specs


def build_distills() -> list[op.DistillSpec]:
    return [
        op.DistillSpec(0.015, 0.08),
        op.DistillSpec(0.020, 0.08),
        op.DistillSpec(0.030, 0.08),
        op.DistillSpec(0.015, 0.10),
        op.DistillSpec(0.020, 0.10),
        op.DistillSpec(0.030, 0.10),
    ]


def make_branch_oof(
    spec: BayesSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    pred = np.empty(len(y), dtype=float)
    active_counts: list[int] = []
    warned = 0
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        fold_pred, meta = fit_predict_branch(spec, X[train_idx], y[train_idx], X[valid_idx])
        pred[valid_idx] = fold_pred
        active_counts.append(int(meta["active_coef_count"]))
        warned += int(meta["convergence_warning"])
    return np.clip(pred, 0, None), {
        "active_coef_median": float(np.median(active_counts)),
        "active_coef_min": int(np.min(active_counts)),
        "oof_convergence_warnings": warned,
    }


def fit_predict_branch(
    spec: BayesSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    pca = PCA(n_components=min(spec.n_components, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_pred = pca.transform(X_pred)

    transformer: PowerTransformer | None = None
    y_fit = y
    if spec.target == "yj":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        y_fit = transformer.fit_transform(y.reshape(-1, 1)).ravel()
    elif spec.target != "raw":
        raise ValueError(spec.target)

    model = build_model(spec.model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(Z_train, y_fit)
    pred = model.predict(Z_pred)
    if transformer is not None:
        pred = transformer.inverse_transform(np.asarray(pred).reshape(-1, 1)).ravel()

    coef = extract_coef(model)
    active = int(np.sum(np.abs(coef) > 1e-8)) if coef is not None else spec.n_components
    warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return np.asarray(pred, dtype=float).ravel(), {
        "active_coef_count": active,
        "convergence_warning": int(warned),
    }


def build_model(name: str):
    if name == "bayesridge":
        return BayesianRidge(max_iter=1000, tol=1e-5)
    if name == "ard":
        return ARDRegression(max_iter=1000, tol=1e-5)
    if name == "lassolars_aic":
        return make_pipeline(StandardScaler(), LassoLarsIC(criterion="aic"))
    if name == "lassolars_bic":
        return make_pipeline(StandardScaler(), LassoLarsIC(criterion="bic"))
    if name == "huber":
        return make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000))
    raise ValueError(name)


def extract_coef(model) -> np.ndarray | None:
    coef = getattr(model, "coef_", None)
    if coef is not None:
        return np.asarray(coef)
    if hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
        coef = getattr(final, "coef_", None)
        if coef is not None:
            return np.asarray(coef)
    return None


def diagnostics(
    *,
    name: str,
    spec: BayesSpec,
    distill: op.DistillSpec,
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
    oof_meta: dict[str, object],
    test_meta: dict[str, object],
) -> dict[str, object]:
    diff = pred - anchor
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    species_shift = max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))
    corrected_oof_rmse = rmse(y, oof_pred)
    return {
        "experiment": name,
        "branch": spec.name,
        "preprocess": spec.preprocess,
        "model": spec.model,
        "target": spec.target,
        "n_components": spec.n_components,
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
        "anchor_corr": op.safe_corr(pred, anchor),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "range_ratio_vs_anchor": float((np.max(pred) - np.min(pred)) / (np.max(anchor) - np.min(anchor))),
        "max_abs_species_mean_shift": species_shift,
        "negative_count": int(np.sum(pred < 0)),
        "active_coef_median": oof_meta["active_coef_median"],
        "active_coef_min": oof_meta["active_coef_min"],
        "test_active_coef_count": test_meta["active_coef_count"],
        "oof_convergence_warnings": oof_meta["oof_convergence_warnings"],
        "test_convergence_warning": test_meta["convergence_warning"],
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
    if int(row["oof_convergence_warnings"]) or int(row["test_convergence_warning"]):
        penalty += 5.0
    return (penalty + max(oof_delta, 0.0) + abs(diff - 0.06) + species, max_abs, species)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
        f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
        f"corr={row['signal_residual_corr']:.4f} active={row['active_coef_median']}"
    )


if __name__ == "__main__":
    main()
