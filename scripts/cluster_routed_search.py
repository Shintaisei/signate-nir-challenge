#!/usr/bin/env python3
"""Waveform-cluster routed local search.

This script is intentionally local-only: it writes diagnostics, not submission
files. Outer group-species folds own all validation estimates; clustering,
feature selection, and per-cluster model choice are fit only on each outer
train fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
from pipeline.cv import Fold, group_kfold  # noqa: E402
from pipeline.data import load_raw_data  # noqa: E402
from pipeline.metrics import rmse  # noqa: E402


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    prep: str
    k: int = 0
    radius: int = 0
    window: int = 0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[], object]


@dataclass
class FittedRoute:
    cluster_id: int
    n_train: int
    feature_spec: FeatureSpec
    model_spec: ModelSpec
    model: object
    columns: np.ndarray | None
    transformer: object | None
    inner_rmse: float
    inner_penalty_score: float


@dataclass
class SearchResult:
    route_name: str
    cluster_prep: str
    n_clusters: int
    outer_rmse: float
    outer_mae: float
    worst_group_rmse: float
    worst_group_bias: float
    fallback_count: int
    route_summary: list[dict[str, object]]
    oof: list[dict[str, object]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="3,5,8")
    parser.add_argument("--cluster-preps", default="raw,smooth5,snv")
    parser.add_argument("--cluster-components", type=int, default=10)
    parser.add_argument("--max-routes", type=int, default=0, help="debug limit after route expansion")
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--preset", choices=["quick", "elastic", "full"], default="quick")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "cluster_routed_search.csv")
    parser.add_argument("--oof-dir", type=Path, default=ROOT / "outputs" / "cluster_routed_oof")
    args = parser.parse_args()

    config = load_config()
    raw = load_raw_data(config)
    target_col = config.target_col if config.target_col in raw.train[0] else "含水率"
    group_col = config.meta_cols[1] if config.meta_cols[1] in raw.train[0] else "species number"
    meta_cols = set(config.meta_cols) | {"樹種", target_col}
    feature_cols = [c for c in raw.train[0] if c not in meta_cols]
    X_all = _matrix(raw.train, feature_cols)
    y_all = np.asarray([float(row[target_col]) for row in raw.train], dtype=float)
    groups = np.asarray([str(row[group_col]) for row in raw.train], dtype=object)

    cluster_preps = [value.strip() for value in args.cluster_preps.split(",") if value.strip()]
    cluster_counts = [int(value) for value in args.clusters.split(",") if value.strip()]
    routes = [(prep, k) for prep in cluster_preps for k in cluster_counts]
    if args.max_routes > 0:
        routes = routes[: args.max_routes]

    feature_specs = _feature_specs(args.preset)
    model_specs = _model_specs(args.preset)
    outer_folds = group_kfold(raw.train, group_col=group_col, n_splits=5)

    args.oof_dir.mkdir(parents=True, exist_ok=True)
    results: list[SearchResult] = []
    for cluster_prep, n_clusters in routes:
        print(f"route cluster_prep={cluster_prep} n_clusters={n_clusters}", flush=True)
        result = _evaluate_route(
            X_all,
            y_all,
            groups,
            outer_folds,
            cluster_prep=cluster_prep,
            n_clusters=n_clusters,
            cluster_components=args.cluster_components,
            feature_specs=feature_specs,
            model_specs=model_specs,
            inner_splits=args.inner_splits,
        )
        results.append(result)
        _write_oof(args.oof_dir / f"{result.route_name}.csv", result.oof)
        print(
            f"  rmse={result.outer_rmse:.4f} worst={result.worst_group_rmse:.4f} "
            f"fallback={result.fallback_count}",
            flush=True,
        )

    results.sort(key=lambda r: r.outer_rmse)
    _write_summary(args.out, results)
    _write_json(args.out.with_suffix(".json"), results[:10])
    print(args.out)
    if results:
        best = results[0]
        print(f"best={best.route_name} group_species_rmse={best.outer_rmse:.6f}")


def _evaluate_route(
    X_all: np.ndarray,
    y_all: np.ndarray,
    groups: np.ndarray,
    outer_folds: list[Fold],
    *,
    cluster_prep: str,
    n_clusters: int,
    cluster_components: int,
    feature_specs: list[FeatureSpec],
    model_specs: list[ModelSpec],
    inner_splits: int,
) -> SearchResult:
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    oof: list[dict[str, object]] = []
    route_summary: list[dict[str, object]] = []
    fallback_count = 0

    for fold in outer_folds:
        train_idx = np.asarray(fold.train_idx, dtype=int)
        valid_idx = np.asarray(fold.valid_idx, dtype=int)
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        groups_train = groups[train_idx]
        X_valid = X_all[valid_idx]
        y_valid = y_all[valid_idx]

        cluster_model = _fit_clusterer(_prep(X_train, cluster_prep), n_clusters, n_components=cluster_components)
        train_clusters = _assign_cluster(cluster_model, _prep(X_train, cluster_prep))
        valid_clusters = _assign_cluster(cluster_model, _prep(X_valid, cluster_prep))
        fallback = _fit_global_fallback(X_train, y_train)

        fold_pred = np.zeros(len(valid_idx), dtype=float)
        for cluster_id in sorted(set(train_clusters.tolist())):
            cluster_train_local = np.where(train_clusters == cluster_id)[0]
            if len(cluster_train_local) < 30:
                continue
            fitted = _fit_best_cluster_route(
                X_train,
                y_train,
                groups_train,
                cluster_train_local,
                cluster_id=cluster_id,
                feature_specs=feature_specs,
                model_specs=model_specs,
                inner_splits=inner_splits,
            )
            valid_local = np.where(valid_clusters == cluster_id)[0]
            if len(valid_local) == 0:
                continue
            fold_pred[valid_local] = _predict_with_route(fitted, X_valid[valid_local])
            route_summary.append(
                {
                    "fold": fold.fold_id,
                    "cluster": cluster_id,
                    "n_train": fitted.n_train,
                    "n_valid": int(len(valid_local)),
                    "feature": fitted.feature_spec.name,
                    "model": fitted.model_spec.name,
                    "inner_rmse": fitted.inner_rmse,
                    "inner_penalty_score": fitted.inner_penalty_score,
                }
            )

        missing = fold_pred == 0.0
        if np.any(missing):
            fold_pred[missing] = fallback.predict(X_valid[missing])
            fallback_count += int(np.sum(missing))
        fold_pred = np.clip(fold_pred, float(np.min(y_train)), float(np.max(y_train)))

        y_true_all.extend(y_valid.tolist())
        y_pred_all.extend(fold_pred.tolist())
        for idx, y_true, y_pred, cluster_id in zip(valid_idx, y_valid, fold_pred, valid_clusters):
            error = float(y_pred - y_true)
            oof.append(
                {
                    "fold": fold.fold_id,
                    "row_index": int(idx),
                    "group": str(groups[idx]),
                    "cluster": int(cluster_id),
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "error": error,
                    "abs_error": abs(error),
                }
            )

    group_stats = _group_stats(oof)
    return SearchResult(
        route_name=f"cluster_{cluster_prep}_k{n_clusters}",
        cluster_prep=cluster_prep,
        n_clusters=n_clusters,
        outer_rmse=rmse(y_true_all, y_pred_all),
        outer_mae=float(np.mean(np.abs(np.asarray(y_pred_all) - np.asarray(y_true_all)))),
        worst_group_rmse=max(item["rmse"] for item in group_stats) if group_stats else float("nan"),
        worst_group_bias=max(group_stats, key=lambda item: abs(item["bias"]))["bias"] if group_stats else float("nan"),
        fallback_count=fallback_count,
        route_summary=route_summary,
        oof=oof,
    )


def _fit_clusterer(X: np.ndarray, n_clusters: int, *, n_components: int):
    n_components = min(n_components, X.shape[1], max(1, X.shape[0] - 1))
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=42),
        KMeans(n_clusters=min(n_clusters, X.shape[0]), random_state=42, n_init=20),
    )
    model.fit(X)
    return model


def _assign_cluster(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=int)


def _fit_best_cluster_route(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    groups_outer_train: np.ndarray,
    cluster_local_idx: np.ndarray,
    *,
    cluster_id: int,
    feature_specs: list[FeatureSpec],
    model_specs: list[ModelSpec],
    inner_splits: int,
) -> FittedRoute:
    X_cluster = X_outer_train[cluster_local_idx]
    y_cluster = y_outer_train[cluster_local_idx]
    groups_cluster = groups_outer_train[cluster_local_idx]
    specs = _restricted_feature_specs(feature_specs, len(y_cluster))
    inner_folds = _inner_folds(groups_cluster, inner_splits)

    best: tuple[float, float, FeatureSpec, ModelSpec] | None = None
    for feature_spec in specs:
        for model_spec in model_specs:
            y_true: list[float] = []
            y_pred: list[float] = []
            for train_local, valid_local in inner_folds:
                try:
                    fitted = _fit_feature_model(X_cluster[train_local], y_cluster[train_local], feature_spec, model_spec)
                    pred = _predict_with_route(fitted, X_cluster[valid_local])
                except Exception:
                    y_true = []
                    y_pred = []
                    break
                pred = np.clip(pred, float(np.min(y_cluster[train_local])), float(np.max(y_cluster[train_local])))
                y_true.extend(y_cluster[valid_local].tolist())
                y_pred.extend(pred.tolist())
            if not y_true:
                continue
            inner_rmse = rmse(y_true, y_pred)
            penalty = inner_rmse + 0.02 * _max_abs_group_bias(groups_cluster, np.asarray(y_true), np.asarray(y_pred), inner_folds)
            if best is None or penalty < best[0]:
                best = (penalty, inner_rmse, feature_spec, model_spec)

    if best is None:
        feature_spec = FeatureSpec("band616_r5", "band", "smooth5", radius=5)
        model_spec = ModelSpec("pls2", lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=2)))
        best = (float("inf"), float("inf"), feature_spec, model_spec)

    penalty, inner_rmse, feature_spec, model_spec = best
    fitted = _fit_feature_model(X_cluster, y_cluster, feature_spec, model_spec)
    fitted.cluster_id = cluster_id
    fitted.inner_rmse = inner_rmse
    fitted.inner_penalty_score = penalty
    return fitted


def _fit_feature_model(X: np.ndarray, y: np.ndarray, feature_spec: FeatureSpec, model_spec: ModelSpec) -> FittedRoute:
    X_prep = _prep(X, feature_spec.prep)
    columns = None
    transformer = None
    if feature_spec.kind == "top":
        columns = _top_corr_columns(X_prep, y, feature_spec.k)
        X_feat = X_prep[:, columns]
    elif feature_spec.kind == "band":
        columns = _band_columns(X_prep.shape[1], 616, feature_spec.radius)
        X_feat = X_prep[:, columns]
    elif feature_spec.kind == "hybrid":
        top = _top_corr_columns(X_prep, y, feature_spec.k)
        band = _band_columns(X_prep.shape[1], 616, feature_spec.radius)
        columns = np.asarray(sorted(set(top.tolist()) | set(band.tolist())), dtype=int)
        X_feat = X_prep[:, columns]
    elif feature_spec.kind == "window_pca":
        transformer = _WindowPcaTransformer(feature_spec.window)
        X_feat = transformer.fit_transform(X_prep)
    else:
        raise ValueError(f"unknown feature kind: {feature_spec.kind}")

    model = model_spec.build()
    model.fit(X_feat, y)
    return FittedRoute(
        cluster_id=-1,
        n_train=len(y),
        feature_spec=feature_spec,
        model_spec=model_spec,
        model=model,
        columns=columns,
        transformer=transformer,
        inner_rmse=float("nan"),
        inner_penalty_score=float("nan"),
    )


def _predict_with_route(route: FittedRoute, X: np.ndarray) -> np.ndarray:
    X_prep = _prep(X, route.feature_spec.prep)
    if route.transformer is not None:
        X_feat = route.transformer.transform(X_prep)
    elif route.columns is not None:
        X_feat = X_prep[:, route.columns]
    else:
        raise RuntimeError("route has no columns or transformer")
    return np.asarray(route.model.predict(X_feat), dtype=float).reshape(-1)


def _fit_global_fallback(X: np.ndarray, y: np.ndarray):
    X_smooth = _prep(X, "smooth5")
    cols = _band_columns(X_smooth.shape[1], 616, 5)
    model = make_pipeline(StandardScaler(), PLSRegression(n_components=2))
    model.fit(X_smooth[:, cols], y)

    class _Fallback:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict(_prep(values, "smooth5")[:, cols]), dtype=float).reshape(-1)

    return _Fallback()


class _WindowPcaTransformer:
    def __init__(self, window: int) -> None:
        self.window = window
        self.blocks: list[tuple[int, int, object]] = []

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        outputs = []
        self.blocks = []
        for start in range(0, X.shape[1], self.window):
            end = min(start + self.window, X.shape[1])
            model = make_pipeline(StandardScaler(), PCA(n_components=1, random_state=42))
            block = model.fit_transform(X[:, start:end])
            outputs.append(block)
            self.blocks.append((start, end, model))
        return np.hstack(outputs)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([model.transform(X[:, start:end]) for start, end, model in self.blocks])


def _feature_specs(preset: str) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    if preset in {"quick", "elastic"}:
        for prep in ("raw", "smooth5", "snv", "diff1", "diff2"):
            for k in (5, 12):
                specs.append(FeatureSpec(f"{prep}_top{k}", "top", prep, k=k))
            for radius in (5, 12):
                specs.append(FeatureSpec(f"{prep}_band616_r{radius}", "band", prep, radius=radius))
            specs.append(FeatureSpec(f"{prep}_hybrid_r5_top8", "hybrid", prep, k=8, radius=5))
            specs.append(FeatureSpec(f"{prep}_window20pca1", "window_pca", prep, window=20))
        return specs

    for prep in ("raw", "smooth5", "snv", "center", "diff1", "diff2"):
        for k in (5, 12, 20):
            specs.append(FeatureSpec(f"{prep}_top{k}", "top", prep, k=k))
        for radius in (5, 12, 20):
            specs.append(FeatureSpec(f"{prep}_band616_r{radius}", "band", prep, radius=radius))
        specs.append(FeatureSpec(f"{prep}_hybrid_r5_top8", "hybrid", prep, k=8, radius=5))
    for prep in ("raw", "smooth5", "snv", "diff1", "diff2"):
        for window in (20, 40):
            specs.append(FeatureSpec(f"{prep}_window{window}pca1", "window_pca", prep, window=window))
    return specs


def _restricted_feature_specs(specs: list[FeatureSpec], n: int) -> list[FeatureSpec]:
    if n >= 80:
        return specs
    out = []
    for spec in specs:
        if spec.kind == "top" and spec.k <= 5:
            out.append(spec)
        elif spec.kind == "band" and spec.radius <= 8:
            out.append(spec)
        elif spec.kind == "hybrid" and spec.radius <= 5 and spec.k <= 8:
            out.append(spec)
    return out


def _model_specs(preset: str) -> list[ModelSpec]:
    if preset == "quick":
        return [
            ModelSpec("ridge100", lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
            ModelSpec("ridge1000", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1000.0))),
            ModelSpec("ridge3000", lambda: make_pipeline(StandardScaler(), Ridge(alpha=3000.0))),
            ModelSpec("pls2", lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=2))),
        ]
    if preset == "elastic":
        return [
            ModelSpec("ridge100", lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
            ModelSpec("ridge1000", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1000.0))),
            ModelSpec("ridge3000", lambda: make_pipeline(StandardScaler(), Ridge(alpha=3000.0))),
            ModelSpec("pls2", lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=2))),
            ModelSpec("huber", lambda: make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000))),
            ModelSpec("elastic001_l01", lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=10000))),
            ModelSpec("elastic003_l03", lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.3, max_iter=10000))),
            ModelSpec("elastic010_l05", lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.10, l1_ratio=0.5, max_iter=10000))),
        ]
    return [
        ModelSpec("ridge30", lambda: make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ModelSpec("ridge100", lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ModelSpec("ridge1000", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1000.0))),
        ModelSpec("ridge3000", lambda: make_pipeline(StandardScaler(), Ridge(alpha=3000.0))),
        ModelSpec("pls1", lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=1))),
        ModelSpec("pls2", lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=2))),
        ModelSpec("pls4", lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=4))),
        ModelSpec("huber", lambda: make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000))),
        ModelSpec("elastic001", lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=10000))),
        ModelSpec("elastic003", lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.3, max_iter=10000))),
    ]


def _inner_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = sorted(set(groups.tolist()))
    if len(unique) >= 2:
        folds: list[list[str]] = [[] for _ in range(min(n_splits, len(unique)))]
        for i, group in enumerate(unique):
            folds[i % len(folds)].append(group)
        out = []
        all_idx = np.arange(len(groups))
        for fold_groups in folds:
            valid = np.asarray([i for i, g in enumerate(groups) if g in set(fold_groups)], dtype=int)
            train = np.asarray([i for i in all_idx if i not in set(valid.tolist())], dtype=int)
            if len(train) > 5 and len(valid) > 0:
                out.append((train, valid))
        if out:
            return out
    idx = np.arange(len(groups))
    chunks = np.array_split(idx, min(n_splits, len(idx)))
    return [(np.setdiff1d(idx, chunk), chunk) for chunk in chunks if len(chunk) > 0 and len(idx) - len(chunk) > 5]


def _max_abs_group_bias(
    groups: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    valid_groups: list[str] = []
    for _train, valid in folds:
        valid_groups.extend(groups[valid].tolist())
    if len(valid_groups) != len(y_true):
        return 0.0
    errors = y_pred - y_true
    return max(abs(float(np.mean(errors[np.asarray(valid_groups) == group]))) for group in set(valid_groups))


def _top_corr_columns(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    x = X - np.mean(X, axis=0, keepdims=True)
    y0 = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x, axis=0) * float(np.sum(y0 * y0)))
    corr = np.divide(x.T @ y0, np.maximum(denom, 1e-12))
    order = np.argsort(np.abs(corr))[::-1]
    return np.asarray(order[: min(k, X.shape[1])], dtype=int)


def _band_columns(n_features: int, center: int, radius: int) -> np.ndarray:
    start = max(0, center - radius)
    end = min(n_features, center + radius + 1)
    return np.arange(start, end, dtype=int)


def _prep(X: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return X
    if name == "smooth5":
        return _smooth5(X)
    if name == "snv":
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.maximum(np.std(X, axis=1, keepdims=True), 1e-12)
        return (X - mean) / std
    if name == "center":
        return X - np.mean(X, axis=1, keepdims=True)
    if name == "diff1":
        return np.diff(X, axis=1)
    if name == "diff2":
        return np.diff(X, n=2, axis=1)
    raise ValueError(f"unknown prep: {name}")


def _smooth5(X: np.ndarray) -> np.ndarray:
    if X.shape[1] < 5:
        return X
    padded = np.pad(X, ((0, 0), (2, 2)), mode="edge")
    return (
        padded[:, 0:-4]
        + padded[:, 1:-3]
        + padded[:, 2:-2]
        + padded[:, 3:-1]
        + padded[:, 4:]
    ) / 5.0


def _matrix(rows: list[dict[str, str]], feature_cols: list[str]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in feature_cols] for row in rows], dtype=float)


def _group_stats(oof: list[dict[str, object]]) -> list[dict[str, float]]:
    out = []
    for group in sorted({str(row["group"]) for row in oof}):
        rows = [row for row in oof if str(row["group"]) == group]
        y_true = [float(row["y_true"]) for row in rows]
        y_pred = [float(row["y_pred"]) for row in rows]
        errors = [p - t for t, p in zip(y_true, y_pred)]
        out.append(
            {
                "group": group,
                "n": len(rows),
                "rmse": rmse(y_true, y_pred),
                "bias": float(np.mean(errors)),
            }
        )
    return out


def _write_summary(path: Path, results: list[SearchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "route_name",
                "cluster_prep",
                "n_clusters",
                "group_species_rmse",
                "mae",
                "worst_group_rmse",
                "worst_group_bias",
                "fallback_count",
                "route_summary_json",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "route_name": result.route_name,
                    "cluster_prep": result.cluster_prep,
                    "n_clusters": result.n_clusters,
                    "group_species_rmse": result.outer_rmse,
                    "mae": result.outer_mae,
                    "worst_group_rmse": result.worst_group_rmse,
                    "worst_group_bias": result.worst_group_bias,
                    "fallback_count": result.fallback_count,
                    "route_summary_json": json.dumps(result.route_summary, ensure_ascii=False),
                }
            )


def _write_json(path: Path, results: list[SearchResult]) -> None:
    payload = [
        {
            "route_name": result.route_name,
            "group_species_rmse": result.outer_rmse,
            "mae": result.outer_mae,
            "worst_group_rmse": result.worst_group_rmse,
            "worst_group_bias": result.worst_group_bias,
            "fallback_count": result.fallback_count,
            "route_summary": result.route_summary,
        }
        for result in results
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_oof(path: Path, oof: list[dict[str, object]]) -> None:
    if not oof:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(oof[0]))
        writer.writeheader()
        writer.writerows(oof)


if __name__ == "__main__":
    main()
