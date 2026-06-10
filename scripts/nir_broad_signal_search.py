#!/usr/bin/env python3
"""Broad non-textbook residual-signal search for the NIR challenge.

The search tests representation-diversity and domain-shift ideas gathered from
spectrometry competitions, Raman transfer writeups, gas-sensor drift work, and
adversarial-validation practice. Every candidate is evaluated only as a small
correction to the protected public-best anchor.
"""

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
from scipy.fft import dct
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.exceptions import ConvergenceWarning

import nir_operator_branch_distill_search as op


warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_broad_signal_search"
CURRENT_ANCHOR = ROOT / "data" / "submissions" / "nir_yj_oof_affine_s0p10_mc1.csv"
PLS_PRIMARY = ROOT / "data" / "submissions" / "nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p1_mc1.csv"


@dataclass(frozen=True)
class BroadSpec:
    name: str
    family: str
    feature_set: str
    model: str
    n_components: int
    alpha: float = 1000.0
    preprocess: str = "sg9_snv"
    weight_mode: str = "none"
    clusters: int = 5
    seed: int = 42


def main() -> None:
    data = op.load_data()
    anchor = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor[0].to_numpy()):
        raise ValueError("anchor sample order mismatch")
    anchor_test = anchor[1].to_numpy(float)

    pls_primary = pd.read_csv(PLS_PRIMARY, header=None)
    if not np.array_equal(data["test_ids"], pls_primary[0].to_numpy()):
        raise ValueError("PLS primary order mismatch")
    pls_primary_delta = pls_primary[1].to_numpy(float) - anchor_test

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF")
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    residual = data["y"] - anchor_oof
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}")

    rows: list[dict[str, object]] = []
    for spec in build_specs():
        print(f"branch {spec.name}", flush=True)
        try:
            branch_oof, oof_meta = make_branch_oof(spec, data["X_train"], data["y"], data["groups"])
            branch_test, test_meta = fit_predict_branch(spec, data["X_train"], data["y"], data["X_test"])
        except Exception as exc:
            print(f"  SKIP {spec.name}: {exc}", flush=True)
            rows.append(failed_row(spec, exc))
            continue
        signal_oof = branch_oof - anchor_oof
        signal_test = branch_test - anchor_test
        beta = op.fit_beta(signal_oof, residual)
        signal_corr = op.safe_corr(signal_oof, residual)
        branch_oof_rmse = rmse(data["y"], branch_oof)

        for distill in build_distills(spec):
            correction_oof = op.make_correction(beta, signal_oof, distill)
            corrected_oof = anchor_oof + correction_oof
            correction_test = op.make_correction(beta, signal_test, distill)
            pred = np.clip(anchor_test + correction_test, 0, None)
            name = (
                f"nir_broad_{spec.name}_b{op.tag(beta)}_"
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
                anchor_oof_rmse=anchor_oof_rmse,
                branch_oof_rmse=branch_oof_rmse,
                beta=beta,
                signal_corr=signal_corr,
                path=path,
                oof_meta=oof_meta,
                test_meta=test_meta,
                pls_primary_delta=pls_primary_delta,
            )
            rows.append(row)
            print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    if not rows_sorted:
        raise RuntimeError("no broad-search candidates were produced")
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "broad_signal_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "broad_signal_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop candidates:")
    for row in rows_sorted[:40]:
        print(
            f"{row['experiment']}: family={row['family']} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
            f"corr={row['signal_residual_corr']:.4f} pls_corr={row['corr_with_pls_primary']:.4f}"
        )
    print(f"saved broad diagnostics: {out_dir}")


def build_specs() -> list[BroadSpec]:
    specs: list[BroadSpec] = []

    for feature_set in ["mv_core", "mv_dct", "mv_contrast", "event"]:
        for model in ["ridge_yj", "ridge_raw", "huber_raw", "pls_raw"]:
            for n_components in [4, 6, 8, 12]:
                if model == "pls_raw" and n_components > 6:
                    continue
                specs.append(
                    BroadSpec(
                        name=f"{model}_{feature_set}_c{n_components}",
                        family="multi_view",
                        feature_set=feature_set,
                        model=model,
                        n_components=n_components,
                        alpha=1200.0,
                    )
                )

    for preprocess in ["msc_sg9", "sg9_snv"]:
        for model in ["adv_ridge_yj", "adv_ridge_raw", "adv_huber_raw", "adv_pls_raw"]:
            for n_components in [3, 4, 5, 8]:
                if model == "adv_pls_raw" and n_components > 5:
                    continue
                for weight_mode in ["logistic", "rf"]:
                    specs.append(
                        BroadSpec(
                            name=f"{model}_{preprocess}_{weight_mode}_c{n_components}",
                            family="adv_weighted_golden",
                            feature_set="domain_pca",
                            model=model,
                            n_components=n_components,
                            preprocess=preprocess,
                            weight_mode=weight_mode,
                            alpha=1200.0,
                        )
                    )

    for preprocess in ["msc_sg9", "sg9_snv"]:
        for clusters in [4, 6, 8]:
            for model in ["drift_pls_raw", "drift_ridge_yj"]:
                for n_components in [3, 4, 5]:
                    specs.append(
                        BroadSpec(
                            name=f"{model}_{preprocess}_k{clusters}_c{n_components}",
                            family="drift_ensemble",
                            feature_set="cluster_pca",
                            model=model,
                            n_components=n_components,
                            preprocess=preprocess,
                            clusters=clusters,
                            alpha=1600.0,
                        )
                    )

    for feature_set in ["event", "mv_contrast"]:
        for model in ["noise_pls_raw", "noise_ridge_yj"]:
            for n_components in [3, 4, 5, 8]:
                if model == "noise_pls_raw" and n_components > 5:
                    continue
                specs.append(
                    BroadSpec(
                        name=f"{model}_{feature_set}_c{n_components}",
                        family="noise_robust",
                        feature_set=feature_set,
                        model=model,
                        n_components=n_components,
                        alpha=1600.0,
                    )
                )
    return specs


def build_distills(spec: BroadSpec) -> list[op.DistillSpec]:
    if spec.family in {"drift_ensemble", "noise_robust"}:
        return [op.DistillSpec(0.01, 0.08), op.DistillSpec(0.015, 0.08), op.DistillSpec(0.02, 0.10)]
    return [
        op.DistillSpec(0.015, 0.08),
        op.DistillSpec(0.02, 0.08),
        op.DistillSpec(0.03, 0.08),
        op.DistillSpec(0.02, 0.10),
        op.DistillSpec(0.03, 0.10),
    ]


def make_branch_oof(
    spec: BroadSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    pred = np.empty(len(y), dtype=float)
    meta_rows: list[dict[str, float]] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        fold_pred, meta = fit_predict_branch(spec, X[train_idx], y[train_idx], X[valid_idx])
        pred[valid_idx] = fold_pred
        meta_rows.append(meta)
    return np.clip(pred, 0, None), summarize_meta(meta_rows)


def fit_predict_branch(
    spec: BroadSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if spec.family == "multi_view":
        X_train, X_pred = make_multiview_features(X_train_raw, X_pred_raw, spec.feature_set, spec.n_components)
        return fit_predict_model(spec.model, X_train, y, X_pred, spec.n_components, spec.alpha), {}
    if spec.family == "adv_weighted_golden":
        return fit_predict_adv_weighted(spec, X_train_raw, y, X_pred_raw)
    if spec.family == "drift_ensemble":
        return fit_predict_drift_ensemble(spec, X_train_raw, y, X_pred_raw)
    if spec.family == "noise_robust":
        return fit_predict_noise_robust(spec, X_train_raw, y, X_pred_raw)
    raise ValueError(spec.family)


def fit_predict_model(
    model_name: str,
    X_train: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
    n_components: int,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    pred = fit_predict_model_raw(model_name, X_train, y, X_pred, n_components, alpha, sample_weight)
    return finite_or_fill(pred, np.mean(y))


def fit_predict_model_raw(
    model_name: str,
    X_train: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
    n_components: int,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if model_name in {"ridge_yj", "adv_ridge_yj", "drift_ridge_yj", "noise_ridge_yj"}:
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        yt = transformer.fit_transform(y.reshape(-1, 1)).ravel()
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(X_train, yt, ridge__sample_weight=sample_weight)
        pred_t = pipe.predict(X_pred)
        return transformer.inverse_transform(pred_t.reshape(-1, 1)).ravel()
    if model_name in {"ridge_raw", "adv_ridge_raw"}:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(X_train, y, ridge__sample_weight=sample_weight)
        return pipe.predict(X_pred)
    if model_name in {"huber_raw", "adv_huber_raw"}:
        pipe = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000))
        pipe.fit(X_train, y, huberregressor__sample_weight=sample_weight)
        return pipe.predict(X_pred)
    if model_name in {"pls_raw", "adv_pls_raw", "drift_pls_raw", "noise_pls_raw"}:
        if sample_weight is not None:
            X_fit, y_fit = replicate_by_weight(X_train, y, sample_weight)
        else:
            X_fit, y_fit = X_train, y
        comps = min(n_components, X_fit.shape[0] - 1, X_fit.shape[1])
        pls = PLSRegression(n_components=max(1, comps), scale=True)
        pls.fit(X_fit, y_fit)
        return pls.predict(X_pred).ravel()
    raise ValueError(model_name)


def make_multiview_features(
    X_train_raw: np.ndarray,
    X_pred_raw: np.ndarray,
    feature_set: str,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    views_train, views_pred = make_views(X_train_raw, X_pred_raw, feature_set)
    train_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    per_view_components = max(1, min(n_components, 4))
    for X_train, X_pred in zip(views_train, views_pred):
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(X_train)
        Z_pred = scaler.transform(X_pred)
        comps = min(per_view_components, Z_train.shape[0] - 1, Z_train.shape[1])
        if comps < Z_train.shape[1]:
            pca = PCA(n_components=max(1, comps), random_state=42)
            train_parts.append(pca.fit_transform(Z_train))
            pred_parts.append(pca.transform(Z_pred))
        else:
            train_parts.append(Z_train)
            pred_parts.append(Z_pred)
    return np.hstack(train_parts), np.hstack(pred_parts)


def make_views(
    X_train_raw: np.ndarray,
    X_pred_raw: np.ndarray,
    feature_set: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    raw_train, raw_pred = X_train_raw, X_pred_raw
    snv_train, snv_pred = op.snv(raw_train), op.snv(raw_pred)
    sg_train, sg_pred = op.snv(op.sg(raw_train, 9)), op.snv(op.sg(raw_pred, 9))
    msc_train, msc_pred = op.preprocess_pair(raw_train, raw_pred, "msc_sg9")
    d1_train, d1_pred = diff_pad(sg_train, 1), diff_pad(sg_pred, 1)
    d2_train, d2_pred = diff_pad(sg_train, 2), diff_pad(sg_pred, 2)
    det_train, det_pred = detrend_rows(snv_train), detrend_rows(snv_pred)
    dct_train, dct_pred = dct_features(snv_train), dct_features(snv_pred)
    event_train, event_pred = event_features(sg_train), event_features(sg_pred)
    contrast_train, contrast_pred = contrast_features(sg_train), contrast_features(sg_pred)

    if feature_set == "mv_core":
        return (
            [raw_train, snv_train, sg_train, msc_train, d1_train, event_train],
            [raw_pred, snv_pred, sg_pred, msc_pred, d1_pred, event_pred],
        )
    if feature_set == "mv_dct":
        return (
            [snv_train, det_train, dct_train, d1_train, d2_train, contrast_train],
            [snv_pred, det_pred, dct_pred, d1_pred, d2_pred, contrast_pred],
        )
    if feature_set == "mv_contrast":
        return (
            [sg_train, d1_train, d2_train, event_train, contrast_train],
            [sg_pred, d1_pred, d2_pred, event_pred, contrast_pred],
        )
    if feature_set == "event":
        return ([event_train, contrast_train], [event_pred, contrast_pred])
    raise ValueError(feature_set)


def fit_predict_adv_weighted(
    spec: BroadSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(np.vstack([X_train, X_pred]))
    pca = PCA(n_components=min(10, X_all.shape[0] - 1, X_all.shape[1]), random_state=42)
    Z_all = pca.fit_transform(X_all)
    Z_train = Z_all[: len(X_train)]
    Z_pred = Z_all[len(X_train) :]
    labels = np.r_[np.zeros(len(X_train)), np.ones(len(X_pred))]
    if spec.weight_mode == "logistic":
        clf = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=spec.seed)
    elif spec.weight_mode == "rf":
        clf = RandomForestClassifier(
            n_estimators=80,
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=spec.seed,
        )
    else:
        raise ValueError(spec.weight_mode)
    clf.fit(Z_all, labels)
    domain_score_train = clf.predict_proba(Z_train)[:, 1]
    domain_score_all = clf.predict_proba(Z_all)[:, 1]
    auc = safe_auc(labels, domain_score_all)
    weights = bounded_rank_weights(domain_score_train, low=0.65, high=1.65)
    X_model_train, X_model_pred = make_adv_model_features(X_train, X_pred, spec.n_components)
    pred = fit_predict_model(spec.model, X_model_train, y, X_model_pred, spec.n_components, spec.alpha, weights)
    return pred, {
        "domain_auc": auc,
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_std": float(np.std(weights)),
    }


def make_adv_model_features(X_train: np.ndarray, X_pred: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_train)
    Z_pred = scaler.transform(X_pred)
    comps = min(max(n_components, 3), Z_train.shape[0] - 1, Z_train.shape[1])
    pca = PCA(n_components=max(1, comps), random_state=42)
    return pca.fit_transform(Z_train), pca.transform(Z_pred)


def fit_predict_drift_ensemble(
    spec: BroadSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    X_train, X_pred = op.preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    Z_train, Z_pred = make_adv_model_features(X_train, X_pred, max(6, spec.n_components))
    k = min(spec.clusters, len(X_train) // 30)
    kmeans = KMeans(n_clusters=max(2, k), n_init=20, random_state=spec.seed)
    cluster = kmeans.fit_predict(Z_train)
    distances = ((Z_pred[:, None, :] - kmeans.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    weights_pred = softmax(-distances / max(float(np.median(distances)), 1e-6), axis=1)

    train_dist = ((Z_train[:, None, :] - kmeans.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    weights_train = softmax(-train_dist / max(float(np.median(train_dist)), 1e-6), axis=1)

    fallback_pred = fit_predict_model(spec.model, Z_train, y, Z_pred, spec.n_components, spec.alpha)
    fallback_train = fit_predict_model(spec.model, Z_train, y, Z_train, spec.n_components, spec.alpha)
    pred_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    fallback_pred_count = 0
    fallback_train_count = 0
    cluster_subset_count = 0
    for c in range(kmeans.n_clusters):
        idx = cluster == c
        if np.sum(idx) < max(25, spec.n_components + 3):
            idx = weights_train[:, c] >= np.quantile(weights_train[:, c], 0.70)
            cluster_subset_count += 1
        X_fit, y_fit = Z_train[idx], y[idx]
        raw_pred = fit_predict_model_raw(spec.model, X_fit, y_fit, Z_pred, spec.n_components, spec.alpha)
        raw_train = fit_predict_model_raw(spec.model, X_fit, y_fit, Z_train, spec.n_components, spec.alpha)
        fallback_pred_count += int(np.sum(~np.isfinite(raw_pred)))
        fallback_train_count += int(np.sum(~np.isfinite(raw_train)))
        pred_parts.append(finite_or_fill(raw_pred, fallback_pred))
        train_parts.append(finite_or_fill(raw_train, fallback_train))
    pred_matrix = np.vstack(pred_parts).T
    train_matrix = np.vstack(train_parts).T
    pred = finite_or_fill(np.sum(weights_pred * pred_matrix, axis=1), fallback_pred)
    train_pred = finite_or_fill(np.sum(weights_train * train_matrix, axis=1), fallback_train)
    return pred, {
        "cluster_count": float(kmeans.n_clusters),
        "cluster_weight_max_mean": float(np.mean(np.max(weights_pred, axis=1))),
        "train_self_rmse": rmse(y, train_pred),
        "cluster_subset_count": float(cluster_subset_count),
        "cluster_fallback_pred_rate": float(fallback_pred_count / max(1, len(Z_pred) * kmeans.n_clusters)),
        "cluster_fallback_train_rate": float(fallback_train_count / max(1, len(Z_train) * kmeans.n_clusters)),
    }


def fit_predict_noise_robust(
    spec: BroadSpec,
    X_train_raw: np.ndarray,
    y: np.ndarray,
    X_pred_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(spec.seed)
    preds: list[np.ndarray] = []
    for scale in [0.0, 0.003, 0.006, 0.010]:
        X_aug = perturb_spectra(X_train_raw, rng, scale)
        X_train, X_pred = make_multiview_features(X_aug, X_pred_raw, spec.feature_set, spec.n_components)
        preds.append(fit_predict_model(spec.model, X_train, y, X_pred, spec.n_components, spec.alpha))
    pred_matrix = np.vstack(preds)
    return np.mean(pred_matrix, axis=0), {
        "noise_pred_std_mean": float(np.mean(np.std(pred_matrix, axis=0))),
        "noise_pred_std_max": float(np.max(np.std(pred_matrix, axis=0))),
    }


def perturb_spectra(X: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    if scale == 0.0:
        return X.copy()
    offset = rng.normal(0.0, scale, size=(len(X), 1))
    mult = 1.0 + rng.normal(0.0, scale, size=(len(X), 1))
    noise = rng.normal(0.0, scale, size=X.shape)
    shifted = np.empty_like(X)
    grid = np.arange(X.shape[1], dtype=float)
    for i, row in enumerate(X):
        jitter = rng.normal(0.0, scale * 2.0)
        shifted[i] = np.interp(grid + jitter, grid, row, left=row[0], right=row[-1])
    return shifted * mult + offset + noise


def diff_pad(X: np.ndarray, order: int) -> np.ndarray:
    out = X.copy()
    for _ in range(order):
        d = np.diff(out, axis=1)
        out = np.hstack([d[:, :1], d])
    return out


def detrend_rows(X: np.ndarray) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, X.shape[1])
    design = np.vstack([np.ones_like(grid), grid]).T
    pinv = np.linalg.pinv(design)
    trend = design @ (pinv @ X.T)
    return X - trend.T


def dct_features(X: np.ndarray) -> np.ndarray:
    coeff = dct(X, type=2, norm="ortho", axis=1)
    low = coeff[:, : min(8, coeff.shape[1])]
    high_energy = np.sqrt(np.cumsum(coeff[:, ::-1] ** 2, axis=1)[:, : min(6, coeff.shape[1])])
    return np.hstack([low, high_energy])


def event_features(X: np.ndarray) -> np.ndarray:
    d1 = np.diff(X, axis=1)
    d2 = np.diff(X, n=2, axis=1)
    ranks = np.argsort(np.argsort(X, axis=1), axis=1) / max(1, X.shape[1] - 1)
    features = [
        np.max(X, axis=1),
        np.min(X, axis=1),
        np.ptp(X, axis=1),
        np.max(np.abs(d1), axis=1),
        np.mean(np.abs(d1), axis=1),
        np.max(np.abs(d2), axis=1),
        np.mean(np.abs(d2), axis=1),
        np.argmax(X, axis=1) / max(1, X.shape[1] - 1),
        np.argmin(X, axis=1) / max(1, X.shape[1] - 1),
        np.mean(ranks[:, : X.shape[1] // 2], axis=1) - np.mean(ranks[:, X.shape[1] // 2 :], axis=1),
    ]
    return np.vstack(features).T


def contrast_features(X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    bins = [
        (0, n // 4),
        (n // 4, n // 2),
        (n // 2, 3 * n // 4),
        (3 * n // 4, n),
        (0, n // 2),
        (n // 2, n),
    ]
    means = [np.mean(X[:, a:b], axis=1) for a, b in bins if b > a]
    feats: list[np.ndarray] = means.copy()
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            feats.append(means[i] - means[j])
            feats.append((means[i] + 1e-6) / (means[j] + 1e-6))
    return np.vstack(feats).T


def bounded_rank_weights(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    ranks = (ranks - 1.0) / max(1.0, len(scores) - 1.0)
    weights = low + (high - low) * ranks
    return weights / np.mean(weights)


def replicate_by_weight(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q1, q2 = np.quantile(weights, [0.40, 0.78])
    reps = np.ones(len(weights), dtype=int)
    reps[weights >= q1] = 2
    reps[weights >= q2] = 3
    return np.repeat(X, reps, axis=0), np.repeat(y, reps, axis=0)


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def safe_auc(labels: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, score))
    except ValueError:
        return 0.5


def summarize_meta(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_max"] = float(np.max(values))
        out[f"{key}_min"] = float(np.min(values))
    return out


def diagnostics(
    *,
    name: str,
    spec: BroadSpec,
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
    oof_meta: dict[str, float],
    test_meta: dict[str, float],
    pls_primary_delta: np.ndarray,
) -> dict[str, object]:
    diff = pred - anchor
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()
    species_shift = max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))
    corrected_oof_rmse = rmse(y, corrected_oof)
    row: dict[str, object] = {
        "status": "ok",
        "experiment": name,
        "family": spec.family,
        "branch": spec.name,
        "feature_set": spec.feature_set,
        "model": spec.model,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "preprocess": spec.preprocess,
        "weight_mode": spec.weight_mode,
        "clusters": spec.clusters,
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
        "corr_with_pls_primary": op.safe_corr(diff, pls_primary_delta),
    }
    for key, value in oof_meta.items():
        row[f"oof_{key}"] = value
    for key, value in test_meta.items():
        row[f"test_{key}"] = value
    return row


def failed_row(spec: BroadSpec, exc: Exception) -> dict[str, object]:
    return {
        "status": "failed",
        "experiment": f"FAILED_{spec.name}",
        "family": spec.family,
        "branch": spec.name,
        "feature_set": spec.feature_set,
        "model": spec.model,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "preprocess": spec.preprocess,
        "weight_mode": spec.weight_mode,
        "clusters": spec.clusters,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "anchor_diff_rmse": float("inf"),
        "anchor_diff_max_abs": float("inf"),
        "max_abs_species_mean_shift": float("inf"),
        "anchor_diff_mean": float("inf"),
        "oof_delta_vs_anchor": float("inf"),
        "branch_oof_rmse": float("inf"),
        "anchor_oof_rmse": float("inf"),
        "beta": 0.0,
        "signal_residual_corr": 0.0,
        "corr_with_pls_primary": 0.0,
    }


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"))
    diff = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    mean = abs(float(row["anchor_diff_mean"]))
    oof_delta = float(row["oof_delta_vs_anchor"])
    beta = float(row["beta"])
    corr = float(row["signal_residual_corr"])
    pls_corr = abs(float(row["corr_with_pls_primary"]))
    penalty = 0.0
    if diff < 0.02 or diff > 0.12:
        penalty += 4.0
    if max_abs > 0.20 or species > 0.04 or mean > 0.015:
        penalty += 5.0
    if beta <= 0.0 or corr <= 0.0:
        penalty += 8.0
    if float(row["branch_oof_rmse"]) > float(row["anchor_oof_rmse"]) + 20.0:
        penalty += 2.0
    fallback_rate = max(
        float(row.get("test_cluster_fallback_pred_rate", 0.0) or 0.0),
        float(row.get("oof_cluster_fallback_pred_rate_max", 0.0) or 0.0),
        float(row.get("oof_cluster_fallback_train_rate_max", 0.0) or 0.0),
    )
    if fallback_rate > 0.05:
        penalty += 20.0
    elif fallback_rate > 0.02:
        penalty += 8.0
    return (penalty + max(oof_delta, 0.0) + abs(diff - 0.06) + species + 0.01 * pls_corr, max_abs, species)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    pred = finite_or_fill(pred, np.mean(y))
    return float(math.sqrt(mean_squared_error(y, pred)))


def finite_or_fill(pred: np.ndarray, fallback: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(pred, dtype=float).ravel()
    if np.all(np.isfinite(arr)):
        return arr
    fallback_arr = np.asarray(fallback, dtype=float)
    if fallback_arr.ndim == 0:
        fill = np.full(arr.shape, float(fallback_arr))
    else:
        fill = fallback_arr.ravel()
        if fill.shape != arr.shape:
            fill = np.full(arr.shape, float(np.nanmean(fill)))
    fill = np.where(np.isfinite(fill), fill, float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0)
    return np.where(np.isfinite(arr), arr, fill)


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
        f"oof_delta={row['oof_delta_vs_anchor']:.4f} beta={row['beta']:.4f} "
        f"corr={row['signal_residual_corr']:.4f} pls_corr={row['corr_with_pls_primary']:.4f}"
    )


if __name__ == "__main__":
    main()
