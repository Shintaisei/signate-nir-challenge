#!/usr/bin/env python3
"""Direct spectral feature-engineering search under Public-risk guards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler

import nir_broad_signal_search as broad
import nir_nonlinear_latent_guard_search as nl
import nir_operator_branch_distill_search as op


warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_spectral_feature_direct_guard"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"


@dataclass(frozen=True)
class SpectralSpec:
    name: str
    feature_set: str
    model: str
    n_components: int
    alpha: float = math.nan
    l1_ratio: float = math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-specs", type=int, default=None)
    parser.add_argument(
        "--feature-sets",
        default="mv_core,mv_dct,mv_contrast,event,shape_compact,shape_ratios,shape_deriv,shape_all",
    )
    args = parser.parse_args()

    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)
    refs = nl.load_reference_diffs(anchor_test)
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF", flush=True)
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}", flush=True)

    specs = build_specs([part.strip() for part in args.feature_sets.split(",") if part.strip()])
    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    print(f"specs={len(specs)}", flush=True)

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        try:
            oof = np.clip(make_oof(spec, data["X_train"], data["y"], data["groups"]), 0, None)
            pred = np.clip(fit_predict_spec(spec, data["X_train"], data["y"], data["X_test"]), 0, None)
        except Exception as exc:
            row = failed_row(spec, exc)
            rows.append(row)
            print(f"  SKIP {type(exc).__name__}: {exc}", flush=True)
            continue

        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row = diagnostics(
            spec=spec,
            path=path,
            pred=pred,
            oof=oof,
            anchor_test=anchor_test,
            anchor_oof=anchor_oof,
            anchor_oof_rmse=anchor_oof_rmse,
            y=data["y"],
            groups=data["groups"],
            test_species=test_species,
            refs=refs,
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "spectral_feature_direct_guard_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "spectral_feature_direct_guard_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop spectral direct candidates:")
    for row in rows_sorted[:80]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"badcorr={row['corr_diff_bad_alpha3000']:.4f} opcorr={row['corr_diff_operator_residual']:.4f} "
            f"direct={row['direct_oof_delta_vs_anchor']:.4f} fold={row['direct_improved_fold_count']}/5"
        )
    print(f"saved spectral feature diagnostics: {out_dir}")


def build_specs(feature_sets: list[str]) -> list[SpectralSpec]:
    specs: list[SpectralSpec] = []
    for feature_set in feature_sets:
        for n_components in [4, 6, 8, 12, 16, 20]:
            if feature_set == "event" and n_components > 12:
                continue
            for alpha in [2500.0, 3500.0, 5000.0, 8000.0]:
                specs.append(
                    SpectralSpec(
                        name=f"nir_spfeat_{feature_set}_ridge_yj_p{n_components}_a{int(alpha)}",
                        feature_set=feature_set,
                        model="ridge_yj",
                        n_components=n_components,
                        alpha=alpha,
                    )
                )
                specs.append(
                    SpectralSpec(
                        name=f"nir_spfeat_{feature_set}_ridge_raw_p{n_components}_a{int(alpha)}",
                        feature_set=feature_set,
                        model="ridge_raw",
                        n_components=n_components,
                        alpha=alpha,
                    )
                )
            specs.append(
                SpectralSpec(
                    name=f"nir_spfeat_{feature_set}_huber_raw_p{n_components}",
                    feature_set=feature_set,
                    model="huber_raw",
                    n_components=n_components,
                    alpha=0.0001,
                )
            )
            specs.append(
                SpectralSpec(
                    name=f"nir_spfeat_{feature_set}_pls_raw_c{n_components}",
                    feature_set=feature_set,
                    model="pls_raw",
                    n_components=n_components,
                )
            )
            for alpha in [0.001, 0.003]:
                specs.append(
                    SpectralSpec(
                        name=f"nir_spfeat_{feature_set}_elastic_raw_p{n_components}_a{nl.tag(alpha)}",
                        feature_set=feature_set,
                        model="elastic_raw",
                        n_components=n_components,
                        alpha=alpha,
                        l1_ratio=0.15,
                    )
                )
    return specs


def make_oof(spec: SpectralSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx] = fit_predict_spec(spec, X[train_idx], y[train_idx], X[valid_idx])
    return pred


def fit_predict_spec(spec: SpectralSpec, X_train_raw: np.ndarray, y: np.ndarray, X_pred_raw: np.ndarray) -> np.ndarray:
    X_train, X_pred = make_features_pair(X_train_raw, X_pred_raw, spec.feature_set)
    if spec.model == "pls_raw":
        comps = min(spec.n_components, X_train.shape[0] - 1, X_train.shape[1])
        model = PLSRegression(n_components=max(1, comps), scale=True)
        model.fit(X_train, y)
        return model.predict(X_pred).ravel()

    Z_train, Z_pred = scaled_pca(X_train, X_pred, spec.n_components)
    if spec.model == "ridge_yj":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
        model = Ridge(alpha=spec.alpha)
        model.fit(Z_train, yt)
        pred_t = model.predict(Z_pred)
        return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()
    if spec.model == "ridge_raw":
        model = Ridge(alpha=spec.alpha)
        model.fit(Z_train, y)
        return model.predict(Z_pred)
    if spec.model == "huber_raw":
        model = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000)
        model.fit(Z_train, y)
        return model.predict(Z_pred)
    if spec.model == "elastic_raw":
        model = ElasticNet(alpha=spec.alpha, l1_ratio=spec.l1_ratio, max_iter=10000, random_state=42)
        model.fit(Z_train, y)
        return model.predict(Z_pred)
    raise ValueError(spec.model)


def scaled_pca(X_train: np.ndarray, X_pred: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    Z_train0 = scaler.fit_transform(X_train)
    Z_pred0 = scaler.transform(X_pred)
    comps = min(n_components, Z_train0.shape[0] - 1, Z_train0.shape[1])
    pca = PCA(n_components=max(1, comps), random_state=42)
    return pca.fit_transform(Z_train0), pca.transform(Z_pred0)


def make_features_pair(X_train_raw: np.ndarray, X_pred_raw: np.ndarray, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    if feature_set in {"mv_core", "mv_dct", "mv_contrast", "event"}:
        views_train, views_pred = broad.make_views(X_train_raw, X_pred_raw, feature_set)
        return np.hstack(views_train), np.hstack(views_pred)

    raw_train, raw_pred = X_train_raw, X_pred_raw
    snv_train, snv_pred = op.snv(raw_train), op.snv(raw_pred)
    sg_train, sg_pred = op.snv(op.sg(raw_train, 9)), op.snv(op.sg(raw_pred, 9))
    d1_train, d1_pred = diff_pad(sg_train, 1), diff_pad(sg_pred, 1)
    d2_train, d2_pred = diff_pad(sg_train, 2), diff_pad(sg_pred, 2)
    seg_train, seg_pred = segment_features(sg_train), segment_features(sg_pred)
    ratio_train, ratio_pred = ratio_features(sg_train), ratio_features(sg_pred)
    dct_train, dct_pred = broad.dct_features(snv_train), broad.dct_features(snv_pred)
    event_train, event_pred = broad.event_features(sg_train), broad.event_features(sg_pred)
    contrast_train, contrast_pred = broad.contrast_features(sg_train), broad.contrast_features(sg_pred)

    if feature_set == "shape_compact":
        return (
            np.hstack([seg_train, event_train, dct_train[:, :8]]),
            np.hstack([seg_pred, event_pred, dct_pred[:, :8]]),
        )
    if feature_set == "shape_ratios":
        return (
            np.hstack([seg_train, ratio_train, contrast_train]),
            np.hstack([seg_pred, ratio_pred, contrast_pred]),
        )
    if feature_set == "shape_deriv":
        return (
            np.hstack([d1_train, d2_train, event_train, dct_train]),
            np.hstack([d1_pred, d2_pred, event_pred, dct_pred]),
        )
    if feature_set == "shape_all":
        return (
            np.hstack([sg_train, d1_train, d2_train, seg_train, ratio_train, event_train, contrast_train, dct_train]),
            np.hstack([sg_pred, d1_pred, d2_pred, seg_pred, ratio_pred, event_pred, contrast_pred, dct_pred]),
        )
    raise ValueError(feature_set)


def diff_pad(X: np.ndarray, order: int) -> np.ndarray:
    out = X.copy()
    for _ in range(order):
        d = np.diff(out, axis=1)
        out = np.hstack([d[:, :1], d])
    return out


def segment_features(X: np.ndarray) -> np.ndarray:
    parts: list[np.ndarray] = []
    for bins in [4, 5, 10]:
        for idx in np.array_split(np.arange(X.shape[1]), bins):
            parts.append(np.mean(X[:, idx], axis=1))
            parts.append(np.std(X[:, idx], axis=1))
    return np.vstack(parts).T


def ratio_features(X: np.ndarray) -> np.ndarray:
    feats: list[np.ndarray] = []
    for step in [1, 2, 3]:
        a = X[:, step:]
        b = X[:, :-step]
        feats.append(a / safe_denom(b))
        feats.append(a - b)
    seg = []
    for idx in np.array_split(np.arange(X.shape[1]), 5):
        seg.append(np.mean(X[:, idx], axis=1))
    seg_arr = np.vstack(seg).T
    for i in range(seg_arr.shape[1]):
        for j in range(i + 1, seg_arr.shape[1]):
            feats.append(((seg_arr[:, i] - seg_arr[:, j])[:, None]))
            feats.append((seg_arr[:, i] / safe_denom(seg_arr[:, j]))[:, None])
    out = np.hstack(feats)
    out = np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0)
    return np.clip(out, -20.0, 20.0)


def safe_denom(values: np.ndarray) -> np.ndarray:
    signs = np.where(values < 0, -1.0, 1.0)
    return np.where(np.abs(values) < 1e-3, signs * 1e-3, values)


def diagnostics(
    *,
    spec: SpectralSpec,
    path: Path,
    pred: np.ndarray,
    oof: np.ndarray,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_oof_rmse: float,
    y: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    refs: dict[str, np.ndarray],
) -> dict[str, object]:
    diff = pred - anchor_test
    oof_rmse = rmse(y, oof)
    fold = nl.fold_delta_stats(y, oof, anchor_oof, groups)
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    row: dict[str, object] = {
        "status": "ok",
        "experiment": spec.name,
        "submission_path": str(path),
        "feature_set": spec.feature_set,
        "model": spec.model,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "l1_ratio": spec.l1_ratio,
        "anchor_oof_rmse": anchor_oof_rmse,
        "direct_oof_rmse": oof_rmse,
        "direct_oof_delta_vs_anchor": oof_rmse - anchor_oof_rmse,
        "direct_improved_fold_count": fold["improved_count"],
        "direct_worst_fold_delta": fold["worst_delta"],
        "direct_fold_delta_std": fold["delta_std"],
        "seed_pred_std_mean": 0.0,
        "seed_pred_std_max": 0.0,
        "seed_pred_rmse_mean": 0.0,
        "pred_min": float(pred.min()),
        "pred_mean": float(pred.mean()),
        "pred_median": float(np.median(pred)),
        "pred_max": float(pred.max()),
        "pred_std": float(np.std(pred)),
        "pred_std_ratio_vs_anchor": float(np.std(pred) / np.std(anchor_test)),
        "negative_count": int(np.sum(pred < 0)),
        "anchor_diff_rmse": nl.rmse(diff),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": nl.safe_corr(pred, anchor_test),
        "max_abs_species_mean_shift": nl.max_abs_species_shift(diff, test_species),
        "range_ratio_vs_anchor": float((pred.max() - pred.min()) / (anchor_test.max() - anchor_test.min())),
        "bottom_decile_shift": float(np.mean(diff[lo])),
        "top_decile_shift": float(np.mean(diff[hi])),
        "decile_gap_top_minus_bottom": float(np.mean(diff[hi]) - np.mean(diff[lo])),
        "sample_order_corr": nl.safe_corr(diff, np.arange(len(diff), dtype=float)),
        "top10_abs_max_species_count": int(top_species.iloc[0]),
        "top10_abs_species_set": ",".join(map(str, sorted(top_species.index.tolist()))),
    }
    for ref_name, ref_diff in refs.items():
        row[f"corr_diff_{ref_name}"] = nl.safe_corr(diff, ref_diff)
        row[f"rmse_diff_{ref_name}"] = nl.rmse(diff - ref_diff)
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else "pass"
    row["submit_gate"] = len(reasons) == 0
    row["public_failure_risk"] = nl.public_failure_risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons = nl.reject_reasons(row)
    if not (0.85 <= float(row["pred_std_ratio_vs_anchor"]) <= 1.15):
        reasons.append("pred_std_ratio_shift")
    return reasons


def failed_row(spec: SpectralSpec, exc: Exception) -> dict[str, object]:
    return {
        "status": "failed",
        "experiment": spec.name,
        "feature_set": spec.feature_set,
        "model": spec.model,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "submit_gate": False,
        "reject_reasons": "failed",
        "public_failure_risk": float("inf"),
        "direct_oof_delta_vs_anchor": float("inf"),
        "direct_improved_fold_count": 0,
        "anchor_diff_rmse": float("inf"),
        "anchor_diff_max_abs": float("inf"),
    }


def rmse(a: np.ndarray, b: np.ndarray | None = None) -> float:
    if b is None:
        values = np.asarray(a, dtype=float)
        return float(math.sqrt(np.mean(values**2)))
    return float(math.sqrt(mean_squared_error(a, b)))


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"))
    penalty = 0.0 if bool(row.get("submit_gate")) else 20.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.8 * max(float(row["direct_oof_delta_vs_anchor"]), 0.0)
        + 0.2 * abs(float(row["anchor_diff_rmse"]) - 0.18),
        float(row["direct_oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} risk={row['public_failure_risk']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} badcorr={row['corr_diff_bad_alpha3000']:.4f} "
        f"direct={row['direct_oof_delta_vs_anchor']:.4f} fold={row['direct_improved_fold_count']}/5 "
        f"reasons={row['reject_reasons']}"
    )


if __name__ == "__main__":
    main()
