#!/usr/bin/env python3
"""Fast model grid on a fixed diff2-PCA cluster routing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402
from pipeline.cv import group_kfold  # noqa: E402
from pipeline.data import load_raw_data  # noqa: E402
from pipeline.metrics import rmse  # noqa: E402


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    prep: str
    kind: str
    k: int = 0
    radius: int = 0
    window: int = 0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "routed_model_grid.csv")
    parser.add_argument("--oof-dir", type=Path, default=ROOT / "outputs" / "routed_model_grid_oof")
    parser.add_argument("--cluster-prep", default="diff2")
    parser.add_argument("--cluster-components", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config()
    raw = load_raw_data(cfg)
    feature_cols = [c for c in raw.train[0] if c not in set(cfg.meta_cols) | {cfg.target_col}]
    X = np.asarray([[float(row[c]) for c in feature_cols] for row in raw.train], dtype=float)
    y = np.asarray([float(row[cfg.target_col]) for row in raw.train], dtype=float)
    groups = np.asarray([str(row["species number"]) for row in raw.train], dtype=object)
    folds = group_kfold(raw.train, group_col="species number", n_splits=5)

    args.oof_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best: tuple[float, str, list[dict[str, object]]] | None = None
    for feature in _features():
        for model in _models():
            print(f"{feature.name} + {model.name}", flush=True)
            summary, oof = _evaluate(X, y, groups, folds, args, feature, model)
            rows.append(summary)
            if best is None or float(summary["group_species_rmse"]) < best[0]:
                best = (float(summary["group_species_rmse"]), str(summary["name"]), oof)
            print(
                f"  rmse={float(summary['group_species_rmse']):.4f} "
                f"worst={float(summary['worst_group_rmse']):.4f}",
                flush=True,
            )
    rows.sort(key=lambda row: float(row["group_species_rmse"]))
    _write_csv(args.out, rows)
    args.out.with_suffix(".top.json").write_text(json.dumps(rows[:20], ensure_ascii=False, indent=2), encoding="utf-8")
    if best is not None:
        _write_csv(args.oof_dir / f"{best[1]}.csv", best[2])
        print(f"best={best[1]} rmse={best[0]:.6f}")


def _evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds,
    args,
    feature: FeatureSpec,
    model_spec: ModelSpec,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    oof = []
    y_true_all = []
    y_pred_all = []
    for fold in folds:
        train_idx = np.asarray(fold.train_idx, dtype=int)
        valid_idx = np.asarray(fold.valid_idx, dtype=int)
        clusterer = _fit_clusterer(_prep(X[train_idx], args.cluster_prep), args.clusters, args.cluster_components)
        train_clusters = clusterer.predict(_prep(X[train_idx], args.cluster_prep))
        valid_clusters = clusterer.predict(_prep(X[valid_idx], args.cluster_prep))
        fallback = _fit_model(_make_features(X[train_idx], y[train_idx], X[train_idx], feature)[0], y[train_idx], model_spec)
        pred = np.zeros(len(valid_idx), dtype=float)
        for cluster_id in sorted(set(train_clusters.tolist())):
            local_train = np.where(train_clusters == cluster_id)[0]
            local_valid = np.where(valid_clusters == cluster_id)[0]
            if len(local_valid) == 0:
                continue
            if len(local_train) < 40:
                X_valid_feat = _make_features(X[train_idx], y[train_idx], X[valid_idx][local_valid], feature)[1]
                pred[local_valid] = fallback.predict(X_valid_feat)
                continue
            X_train_feat, X_valid_feat = _make_features(
                X[train_idx][local_train],
                y[train_idx][local_train],
                X[valid_idx][local_valid],
                feature,
            )
            model = _fit_model(X_train_feat, y[train_idx][local_train], model_spec)
            pred[local_valid] = model.predict(X_valid_feat)
        pred = np.clip(pred, float(np.min(y[train_idx])), float(np.max(y[train_idx])))
        y_true_all.extend(y[valid_idx].tolist())
        y_pred_all.extend(pred.tolist())
        for idx, true, value, cluster_id in zip(valid_idx, y[valid_idx], pred, valid_clusters):
            error = float(value - true)
            oof.append(
                {
                    "fold": fold.fold_id,
                    "row_index": int(idx),
                    "group": str(groups[idx]),
                    "cluster": int(cluster_id),
                    "y_true": float(true),
                    "y_pred": float(value),
                    "error": error,
                    "abs_error": abs(error),
                }
            )
    summary = _summary(feature, model_spec, y_true_all, y_pred_all, oof)
    return summary, oof


def _fit_clusterer(X: np.ndarray, k: int, n_components: int):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1), random_state=42),
        KMeans(n_clusters=k, random_state=42, n_init=30),
    ).fit(X)


def _make_features(X_train: np.ndarray, y_train: np.ndarray, X_apply: np.ndarray, spec: FeatureSpec):
    Z_train = _prep(X_train, spec.prep)
    Z_apply = _prep(X_apply, spec.prep)
    if spec.kind == "top":
        cols = _top_corr_columns(Z_train, y_train, spec.k)
        return Z_train[:, cols], Z_apply[:, cols]
    if spec.kind == "band616":
        cols = _band_columns(Z_train.shape[1], 616, spec.radius)
        return Z_train[:, cols], Z_apply[:, cols]
    if spec.kind == "window_pca":
        return _window_pca(Z_train, Z_apply, spec.window)
    raise ValueError(spec.kind)


def _fit_model(X: np.ndarray, y: np.ndarray, spec: ModelSpec):
    if spec.kind == "ridge100":
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    elif spec.kind == "ridge1000":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
    elif spec.kind == "ridge3000":
        model = make_pipeline(StandardScaler(), Ridge(alpha=3000.0))
    elif spec.kind == "pls2":
        model = make_pipeline(StandardScaler(), PLSRegression(n_components=min(2, X.shape[1], X.shape[0] - 1)))
    elif spec.kind == "huber":
        model = make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000))
    elif spec.kind == "elastic001":
        model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=10000))
    elif spec.kind == "elastic003":
        model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.3, max_iter=10000))
    elif spec.kind == "elastic010":
        model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.10, l1_ratio=0.5, max_iter=10000))
    else:
        raise ValueError(spec.kind)
    model.fit(X, y)
    return model


def _features() -> list[FeatureSpec]:
    return [
        FeatureSpec("smooth5_band616_r5", "smooth5", "band616", radius=5),
        FeatureSpec("smooth5_band616_r12", "smooth5", "band616", radius=12),
        FeatureSpec("raw_band616_r5", "raw", "band616", radius=5),
        FeatureSpec("raw_top12", "raw", "top", k=12),
        FeatureSpec("smooth5_top12", "smooth5", "top", k=12),
        FeatureSpec("snv_top12", "snv", "top", k=12),
        FeatureSpec("diff1_top12", "diff1", "top", k=12),
        FeatureSpec("diff2_top12", "diff2", "top", k=12),
        FeatureSpec("raw_window20pca1", "raw", "window_pca", window=20),
        FeatureSpec("smooth5_window20pca1", "smooth5", "window_pca", window=20),
        FeatureSpec("snv_window20pca1", "snv", "window_pca", window=20),
        FeatureSpec("diff1_window20pca1", "diff1", "window_pca", window=20),
        FeatureSpec("diff2_window20pca1", "diff2", "window_pca", window=20),
    ]


def _models() -> list[ModelSpec]:
    return [
        ModelSpec("ridge100", "ridge100"),
        ModelSpec("ridge1000", "ridge1000"),
        ModelSpec("ridge3000", "ridge3000"),
        ModelSpec("pls2", "pls2"),
        ModelSpec("huber", "huber"),
        ModelSpec("elastic001_l01", "elastic001"),
        ModelSpec("elastic003_l03", "elastic003"),
        ModelSpec("elastic010_l05", "elastic010"),
    ]


def _prep(X: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return X
    if name == "smooth5":
        p = np.pad(X, ((0, 0), (2, 2)), mode="edge")
        return (p[:, :-4] + p[:, 1:-3] + p[:, 2:-2] + p[:, 3:-1] + p[:, 4:]) / 5.0
    if name == "snv":
        return (X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-12)
    if name == "diff1":
        return np.diff(X, axis=1)
    if name == "diff2":
        return np.diff(X, n=2, axis=1)
    raise ValueError(name)


def _top_corr_columns(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    x = X - np.mean(X, axis=0, keepdims=True)
    y0 = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x, axis=0) * float(np.sum(y0 * y0)))
    corr = (x.T @ y0) / np.maximum(denom, 1e-12)
    return np.argsort(np.abs(corr))[::-1][: min(k, X.shape[1])].astype(int)


def _band_columns(n: int, center: int, radius: int) -> np.ndarray:
    return np.arange(max(0, center - radius), min(n, center + radius + 1), dtype=int)


def _window_pca(X_train: np.ndarray, X_apply: np.ndarray, window: int):
    train_blocks = []
    apply_blocks = []
    for start in range(0, X_train.shape[1], window):
        end = min(start + window, X_train.shape[1])
        model = make_pipeline(StandardScaler(), PCA(n_components=1, random_state=42))
        train_blocks.append(model.fit_transform(X_train[:, start:end]))
        apply_blocks.append(model.transform(X_apply[:, start:end]))
    return np.hstack(train_blocks), np.hstack(apply_blocks)


def _summary(feature: FeatureSpec, model: ModelSpec, y_true, y_pred, oof):
    errors = np.asarray(y_pred) - np.asarray(y_true)
    worst = _worst_group(oof)
    return {
        "name": f"{feature.name}__{model.name}",
        "feature": feature.name,
        "model": model.name,
        "group_species_rmse": rmse(y_true, y_pred),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "worst_group": worst["group"],
        "worst_group_rmse": worst["rmse"],
        "worst_group_bias": worst["bias"],
    }


def _worst_group(oof):
    best = {"group": "", "rmse": -1.0, "bias": 0.0}
    for group in sorted({str(row["group"]) for row in oof}):
        rows = [row for row in oof if str(row["group"]) == group]
        errors = [float(row["error"]) for row in rows]
        value = math.sqrt(sum(e * e for e in errors) / len(errors))
        if value > float(best["rmse"]):
            best = {"group": group, "rmse": value, "bias": sum(errors) / len(errors)}
    return best


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
