#!/usr/bin/env python3
"""Rebuild direct anchors around the current YJ Ridge family.

This is intentionally not another tiny residual-branch search. Each candidate
fits a full direct model, then optionally applies that model's own OOF-affine
calibration. Broad rows use standard OOF for fast screening; the top rows get
nested affine evaluation to avoid ranking on an optimistic correction.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_anchor_rebuild"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"


@dataclass(frozen=True)
class AnchorSpec:
    name: str
    preprocess: str
    model: str
    target: str
    n_components: int
    alpha: float = 3500.0
    affine_shrink: float = 0.10
    affine_clip: float = 0.40
    mean_center: bool = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-top", type=int, default=80, help="Number of rows to nested-evaluate.")
    parser.add_argument("--max-specs", type=int, default=None, help="Optional cap for quick smoke runs.")
    args = parser.parse_args()

    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor nested OOF")
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}")
    assert_anchor_reproduction(data, anchor_test)

    specs = build_specs()
    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    print(f"specs={len(specs)}")

    rows: list[dict[str, object]] = []
    row_by_name: dict[str, dict[str, object]] = {}
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        try:
            direct_oof = make_oof(spec, data["X_train"], data["y"], data["groups"])
            direct_test = fit_predict_spec(spec, data["X_train"], data["y"], data["X_test"])
            params = fit_affine_params(direct_oof, data["y"])
            affine_oof_fast = np.clip(apply_affine(direct_oof, params, spec), 0, None)
            affine_test = np.clip(apply_affine(direct_test, params, spec), 0, None)
        except Exception as exc:
            row = failed_row(spec, exc)
            rows.append(row)
            row_by_name[spec.name] = row
            print(f"  SKIP {type(exc).__name__}: {exc}", flush=True)
            continue

        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: affine_test}).to_csv(path, index=False, header=False)
        row = diagnostics(
            spec=spec,
            pred=affine_test,
            direct_test=direct_test,
            anchor_test=anchor_test,
            y=data["y"],
            groups=data["groups"],
            direct_oof=direct_oof,
            affine_oof=affine_oof_fast,
            anchor_oof=anchor_oof,
            anchor_oof_rmse=anchor_oof_rmse,
            path=path,
            nested=False,
        )
        rows.append(row)
        row_by_name[spec.name] = row
        cache[spec.name] = (direct_oof, direct_test)
        print_one(row)

    picked = pick_nested(rows, args.cv_top)
    print(f"\nnested candidates={len(picked)}")
    for name in picked:
        spec = next(item for item in specs if item.name == name)
        row = row_by_name[name]
        if row.get("status") != "ok":
            continue
        direct_oof, direct_test = cache[name]
        nested_oof = make_nested_affine_oof(spec, data["X_train"], data["y"], data["groups"])
        params = fit_affine_params(direct_oof, data["y"])
        affine_test = np.clip(apply_affine(direct_test, params, spec), 0, None)
        path = candidate_dir / f"{spec.name}_nested_ranked.csv"
        pd.DataFrame({0: data["test_ids"], 1: affine_test}).to_csv(path, index=False, header=False)
        row.update(
            diagnostics(
                spec=spec,
                pred=affine_test,
                direct_test=direct_test,
                anchor_test=anchor_test,
                y=data["y"],
                groups=data["groups"],
                direct_oof=direct_oof,
                affine_oof=nested_oof,
                anchor_oof=anchor_oof,
                anchor_oof_rmse=anchor_oof_rmse,
                path=path,
                nested=True,
            )
        )
        print("nested", end=" ")
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "anchor_rebuild_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "anchor_rebuild_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop candidates:")
    for row in rows_sorted[:40]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: pre={row['preprocess']} model={row['model']} target={row['target']} "
            f"pca={row['n_components']} alpha={row['alpha']} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"direct_delta={row['direct_oof_delta_vs_anchor']:.4f} "
            f"affine_delta={row['affine_oof_delta_vs_anchor']:.4f} "
            f"nested={row['nested_evaluated']} range={row['range_ratio_vs_anchor']:.4f}"
        )
    print(f"saved anchor rebuild diagnostics: {out_dir}")


def build_specs() -> list[AnchorSpec]:
    specs: list[AnchorSpec] = []
    preprocesses = [
        "sg9_snv",
        "sg7_snv",
        "sg11_snv",
        "snv",
        "msc_sg7",
        "msc_sg9",
        "msc_sg11",
        "detrend_snv",
        "sg9_snv_d1",
        "sg9_snv_d2",
        "sg9_snv_stack_d1",
        "sg9_snv_stack_d2",
    ]
    for preprocess in preprocesses:
        for target in ["yj", "log1p", "raw"]:
            for n_components in [10, 12, 15, 18, 20, 24, 28, 32]:
                for alpha in [2500.0, 3000.0, 3500.0, 4000.0, 5000.0, 6500.0, 8000.0, 10000.0]:
                    specs.append(
                        AnchorSpec(
                            name=f"nir_anchor_{preprocess}_ridge_{target}_p{n_components}_a{int(alpha)}_affs10",
                            preprocess=preprocess,
                            model="ridge",
                            target=target,
                            n_components=n_components,
                            alpha=alpha,
                            affine_shrink=0.10,
                        )
                    )
    for preprocess in ["sg9_snv", "sg11_snv", "msc_sg9", "detrend_snv", "sg9_snv_stack_d1"]:
        for target in ["raw", "yj"]:
            for n_components in [3, 4, 5, 6, 8]:
                specs.append(
                    AnchorSpec(
                        name=f"nir_anchor_{preprocess}_pls_{target}_c{n_components}_affs10",
                        preprocess=preprocess,
                        model="pls",
                        target=target,
                        n_components=n_components,
                        alpha=0.0,
                        affine_shrink=0.10,
                    )
                )
    for preprocess in ["sg9_snv", "msc_sg9", "detrend_snv"]:
        for target in ["raw", "yj"]:
            for n_components in [12, 15, 20, 24]:
                specs.append(
                    AnchorSpec(
                        name=f"nir_anchor_{preprocess}_huber_{target}_p{n_components}_affs10",
                        preprocess=preprocess,
                        model="huber",
                        target=target,
                        n_components=n_components,
                        alpha=0.0001,
                        affine_shrink=0.10,
                    )
                )
    return specs


def make_oof(spec: AnchorSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx] = fit_predict_spec(spec, X[train_idx], y[train_idx], X[valid_idx])
    return np.clip(pred, 0, None)


def make_nested_affine_oof(spec: AnchorSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        X_train = X[train_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx]
        valid_direct = fit_predict_spec(spec, X_train, y_train, X[valid_idx])
        inner_oof = make_oof(spec, X_train, y_train, groups_train)
        params = fit_affine_params(inner_oof, y_train)
        pred[valid_idx] = apply_affine(valid_direct, params, spec)
    return np.clip(pred, 0, None)


def fit_predict_spec(spec: AnchorSpec, X_train_raw: np.ndarray, y: np.ndarray, X_pred_raw: np.ndarray) -> np.ndarray:
    X_train, X_pred = preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    if spec.model == "ridge":
        return inverse_target(spec.target, y, fit_predict_latent_ridge(X_train, y, X_pred, spec))
    if spec.model == "pls":
        yt = transform_target(spec.target, y)
        comps = min(spec.n_components, X_train.shape[0] - 1, X_train.shape[1])
        model = PLSRegression(n_components=max(1, comps), scale=True)
        model.fit(X_train, yt)
        return inverse_target(spec.target, y, model.predict(X_pred).ravel())
    if spec.model == "huber":
        yt = transform_target(spec.target, y)
        Z_train, Z_pred = pca_features(X_train, X_pred, spec.n_components)
        model = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000))
        model.fit(Z_train, yt)
        return inverse_target(spec.target, y, model.predict(Z_pred))
    raise ValueError(spec.model)


def fit_predict_latent_ridge(X_train: np.ndarray, y: np.ndarray, X_pred: np.ndarray, spec: AnchorSpec) -> np.ndarray:
    yt = transform_target(spec.target, y)
    Z_train, Z_pred = pca_features(X_train, X_pred, spec.n_components)
    model = Ridge(alpha=spec.alpha)
    model.fit(Z_train, yt)
    return model.predict(Z_pred)


def pca_features(X_train: np.ndarray, X_pred: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    comps = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=max(1, comps), random_state=42)
    return pca.fit_transform(X_train), pca.transform(X_pred)


def transform_target(target: str, y: np.ndarray) -> np.ndarray:
    if target == "raw":
        return y
    if target == "log1p":
        return np.log1p(y)
    if target == "yj":
        return PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(y.reshape(-1, 1)).ravel()
    raise ValueError(target)


def inverse_target(target: str, y_fit: np.ndarray, pred_t: np.ndarray) -> np.ndarray:
    if target == "raw":
        return pred_t
    if target == "log1p":
        return np.expm1(pred_t)
    if target == "yj":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        transformer.fit(y_fit.reshape(-1, 1))
        return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()
    raise ValueError(target)


def preprocess_pair(X_train: np.ndarray, X_pred: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if name in {"raw", "sg7_snv", "sg9_snv", "sg11_snv", "sg9_snv_stack_d1", "sg9_snv_stack_d2", "msc_sg7", "msc_sg9", "msc_sg11"}:
        return op.preprocess_pair(X_train, X_pred, name)
    if name == "snv":
        return op.snv(X_train), op.snv(X_pred)
    if name == "detrend_snv":
        return detrend_rows(op.snv(X_train)), detrend_rows(op.snv(X_pred))
    if name == "sg9_snv_d2":
        return op.sg(op.snv(X_train), 11, deriv=2), op.sg(op.snv(X_pred), 11, deriv=2)
    raise ValueError(name)


def detrend_rows(X: np.ndarray) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, X.shape[1])
    design = np.vstack([np.ones_like(grid), grid]).T
    pinv = np.linalg.pinv(design)
    trend = design @ (pinv @ X.T)
    return X - trend.T


def fit_affine_params(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(pred, y, deg=1)
    return float(intercept), float(slope)


def apply_affine(pred: np.ndarray, params: tuple[float, float], spec: AnchorSpec) -> np.ndarray:
    intercept, slope = params
    delta = spec.affine_shrink * (intercept + slope * pred - pred)
    delta = np.clip(delta, -spec.affine_clip, spec.affine_clip)
    if spec.mean_center:
        delta = delta - np.mean(delta)
    return pred + delta


def diagnostics(
    *,
    spec: AnchorSpec,
    pred: np.ndarray,
    direct_test: np.ndarray,
    anchor_test: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    direct_oof: np.ndarray,
    affine_oof: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_oof_rmse: float,
    path: Path,
    nested: bool,
) -> dict[str, object]:
    diff = pred - anchor_test
    direct_diff = direct_test - anchor_test
    species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    direct_rmse = rmse(y, direct_oof)
    affine_rmse = rmse(y, affine_oof)
    direct_fold = fold_delta_stats(y, direct_oof, anchor_oof, groups)
    affine_fold = fold_delta_stats(y, affine_oof, anchor_oof, groups)
    return {
        "status": "ok",
        "experiment": spec.name,
        "submission_path": str(path),
        "preprocess": spec.preprocess,
        "model": spec.model,
        "target": spec.target,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "affine_shrink": spec.affine_shrink,
        "affine_clip": spec.affine_clip,
        "mean_center": spec.mean_center,
        "nested_evaluated": nested,
        "anchor_oof_rmse": anchor_oof_rmse,
        "direct_oof_rmse": direct_rmse,
        "affine_oof_rmse": affine_rmse,
        "direct_oof_delta_vs_anchor": direct_rmse - anchor_oof_rmse,
        "affine_oof_delta_vs_anchor": affine_rmse - anchor_oof_rmse,
        "direct_improved_fold_count": direct_fold["improved_count"],
        "direct_worst_fold_delta": direct_fold["worst_delta"],
        "direct_fold_delta_std": direct_fold["delta_std"],
        "affine_improved_fold_count": affine_fold["improved_count"],
        "affine_worst_fold_delta": affine_fold["worst_delta"],
        "affine_fold_delta_std": affine_fold["delta_std"],
        "direct_oof_corr_with_anchor": safe_corr(direct_oof, anchor_oof),
        "affine_oof_corr_with_anchor": safe_corr(affine_oof, anchor_oof),
        "pred_min": float(np.min(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_median": float(np.median(pred)),
        "pred_max": float(np.max(pred)),
        "pred_std": float(np.std(pred)),
        "negative_count": int(np.sum(pred < 0)),
        "anchor_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "anchor_direct_diff_rmse": float(math.sqrt(np.mean(direct_diff**2))),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": safe_corr(pred, anchor_test),
        "range_ratio_vs_anchor": float((np.max(pred) - np.min(pred)) / (np.max(anchor_test) - np.min(anchor_test))),
        "max_abs_species_mean_shift": max_abs_species_shift(diff, species),
    }


def failed_row(spec: AnchorSpec, exc: Exception) -> dict[str, object]:
    return {
        "status": "failed",
        "experiment": spec.name,
        "preprocess": spec.preprocess,
        "model": spec.model,
        "target": spec.target,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "nested_evaluated": False,
        "anchor_diff_rmse": float("inf"),
        "anchor_diff_max_abs": float("inf"),
        "max_abs_species_mean_shift": float("inf"),
        "affine_oof_delta_vs_anchor": float("inf"),
        "direct_oof_delta_vs_anchor": float("inf"),
        "direct_improved_fold_count": 0,
        "affine_improved_fold_count": 0,
    }


def pick_nested(rows: list[dict[str, object]], limit: int) -> list[str]:
    ok = [row for row in rows if row.get("status") == "ok"]
    ranked = sorted(ok, key=rough_rank_key)
    picked: list[str] = []
    for row in ranked:
        if len(picked) >= limit:
            break
        picked.append(str(row["experiment"]))
    return picked


def rough_rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    diff = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    affine_delta = float(row["affine_oof_delta_vs_anchor"])
    direct_delta = float(row["direct_oof_delta_vs_anchor"])
    corr = float(row["anchor_corr"])
    penalty = 0.0
    if diff < 0.03 or diff > 1.5:
        penalty += 3.0
    if max_abs > 4.0 or species > 0.20 or corr < 0.998:
        penalty += 5.0
    if direct_delta > 1.0:
        penalty += 2.0
    if int(row.get("direct_improved_fold_count", 0) or 0) < 3:
        penalty += 1.0
    return (penalty + max(affine_delta, 0.0) + 0.2 * max(direct_delta, 0.0) + 0.05 * species + abs(diff - 0.25) * 0.02, max_abs, species)


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"))
    base = rough_rank_key(row)
    if not bool(row.get("nested_evaluated")):
        return (base[0] + 20.0, base[1], base[2])
    nested_delta = float(row["affine_oof_delta_vs_anchor"])
    diff = float(row["anchor_diff_rmse"])
    species = float(row["max_abs_species_mean_shift"])
    max_abs = float(row["anchor_diff_max_abs"])
    penalty = 0.0
    if nested_delta > 0.0:
        penalty += 2.0 + nested_delta
    if int(row.get("direct_improved_fold_count", 0) or 0) < 3:
        penalty += 1.5
    if int(row.get("affine_improved_fold_count", 0) or 0) < 3:
        penalty += 1.5
    if diff < 0.04 or diff > 1.2:
        penalty += 2.0
    if max_abs > 3.0 or species > 0.15:
        penalty += 5.0
    return (penalty + max(nested_delta, -2.0) + 0.03 * species + abs(diff - 0.20) * 0.02, max_abs, species)


def max_abs_species_shift(diff: np.ndarray, species: np.ndarray) -> float:
    return max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def fold_delta_stats(y: np.ndarray, pred: np.ndarray, anchor_pred: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    deltas: list[float] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for _, valid_idx in splitter.split(np.zeros((len(y), 1)), y, groups):
        deltas.append(rmse(y[valid_idx], pred[valid_idx]) - rmse(y[valid_idx], anchor_pred[valid_idx]))
    arr = np.asarray(deltas, dtype=float)
    return {
        "improved_count": int(np.sum(arr < 0.0)),
        "worst_delta": float(np.max(arr)),
        "delta_std": float(np.std(arr)),
    }


def assert_anchor_reproduction(data: dict[str, np.ndarray], anchor_test: np.ndarray) -> None:
    spec = AnchorSpec(
        name="anchor_reproduction",
        preprocess="sg9_snv",
        model="ridge",
        target="yj",
        n_components=20,
        alpha=3500.0,
        affine_shrink=0.10,
        affine_clip=0.40,
        mean_center=True,
    )
    direct_oof = make_oof(spec, data["X_train"], data["y"], data["groups"])
    direct_test = fit_predict_spec(spec, data["X_train"], data["y"], data["X_test"])
    params = fit_affine_params(direct_oof, data["y"])
    pred = np.clip(apply_affine(direct_test, params, spec), 0, None)
    diff = pred - anchor_test
    rmse_diff = float(math.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    print(f"anchor reproduction diff_rmse={rmse_diff:.12f} max_abs={max_abs:.12f}")
    if max_abs > 1e-8:
        raise ValueError("anchor reproduction failed; search is not comparable to current best")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
        f"direct_delta={row['direct_oof_delta_vs_anchor']:.4f} "
        f"affine_delta={row['affine_oof_delta_vs_anchor']:.4f} "
        f"nested={row['nested_evaluated']}"
    )


if __name__ == "__main__":
    main()
