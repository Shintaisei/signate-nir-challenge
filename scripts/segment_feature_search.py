#!/usr/bin/env python3
"""Search cutout/segment-derived spectral features under group-species CV."""

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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
class FeatureRecipe:
    name: str
    prep: str
    window: int
    step: int
    stats: tuple[str, ...]
    select_k: int
    include_band616: bool = True
    include_ratios: bool = False
    include_anchor_ratios: bool = False
    include_local616: bool = False
    anchor_centers: tuple[int, ...] = (616,)


@dataclass(frozen=True)
class ModelRecipe:
    name: str
    kind: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "focused", "full"], default="quick")
    parser.add_argument("--only", default="", help="comma-separated substrings to keep feature recipes")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "segment_feature_search.csv")
    parser.add_argument("--oof-dir", type=Path, default=ROOT / "outputs" / "segment_feature_oof")
    args = parser.parse_args()

    config = load_config()
    raw = load_raw_data(config)
    feature_cols = [c for c in raw.train[0] if c not in set(config.meta_cols) | {config.target_col}]
    wavelengths = np.asarray([float(c) for c in feature_cols], dtype=float)
    X = np.asarray([[float(row[c]) for c in feature_cols] for row in raw.train], dtype=float)
    y = np.asarray([float(row[config.target_col]) for row in raw.train], dtype=float)
    groups = np.asarray([str(row[config.meta_cols[1]]) for row in raw.train], dtype=object)
    folds = group_kfold(raw.train, group_col=config.meta_cols[1], n_splits=5)

    recipes = _feature_recipes(args.preset)
    if args.only:
        needles = [value.strip() for value in args.only.split(",") if value.strip()]
        recipes = [recipe for recipe in recipes if any(needle in recipe.name for needle in needles)]
    models = _model_recipes(args.preset)
    args.oof_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_oof: tuple[str, list[dict[str, object]]] | None = None
    best_rmse = float("inf")
    for recipe in recipes:
        print(f"features {recipe.name}", flush=True)
        for model in models:
            result = _evaluate_global(X, y, groups, folds, wavelengths, recipe, model)
            rows.append(result["summary"])
            if float(result["summary"]["group_species_rmse"]) < best_rmse:
                best_rmse = float(result["summary"]["group_species_rmse"])
                best_oof = (f"{recipe.name}__{model.name}", result["oof"])
            print(
                f"  {model.name}: rmse={float(result['summary']['group_species_rmse']):.4f} "
                f"worst={float(result['summary']['worst_group_rmse']):.4f}",
                flush=True,
            )

    for recipe in recipes:
        if recipe.select_k > 160:
            continue
        for k in (3, 5, 8):
            result = _evaluate_cluster_fixed(X, y, groups, folds, wavelengths, recipe, n_clusters=k)
            rows.append(result["summary"])
            if float(result["summary"]["group_species_rmse"]) < best_rmse:
                best_rmse = float(result["summary"]["group_species_rmse"])
                best_oof = (f"{recipe.name}__cluster_k{k}", result["oof"])
            print(
                f"  cluster k={k}: rmse={float(result['summary']['group_species_rmse']):.4f} "
                f"worst={float(result['summary']['worst_group_rmse']):.4f}",
                flush=True,
            )
        if args.preset in {"focused", "full"}:
            for k in (2, 3, 5):
                result = _evaluate_cluster_fixed(
                    X,
                    y,
                    groups,
                    folds,
                    wavelengths,
                    recipe,
                    n_clusters=k,
                    transductive=True,
                )
                rows.append(result["summary"])
                if float(result["summary"]["group_species_rmse"]) < best_rmse:
                    best_rmse = float(result["summary"]["group_species_rmse"])
                    best_oof = (f"{recipe.name}__transductive_cluster_k{k}", result["oof"])
                print(
                    f"  transductive cluster k={k}: rmse={float(result['summary']['group_species_rmse']):.4f} "
                    f"worst={float(result['summary']['worst_group_rmse']):.4f}",
                    flush=True,
                )

    rows.sort(key=lambda row: float(row["guard_score"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if best_oof is not None:
        name, oof = best_oof
        _write_oof(args.oof_dir / f"{name}.csv", oof)
    args.out.with_suffix(".top.json").write_text(json.dumps(rows[:20], ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)
    print(f"best_rmse={best_rmse:.6f}")


def _evaluate_global(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds,
    wavelengths: np.ndarray,
    recipe: FeatureRecipe,
    model_recipe: ModelRecipe,
) -> dict[str, object]:
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    oof: list[dict[str, object]] = []
    for fold in folds:
        train_idx = np.asarray(fold.train_idx, dtype=int)
        valid_idx = np.asarray(fold.valid_idx, dtype=int)
        builder = _fit_feature_builder(X[train_idx], y[train_idx], wavelengths, recipe)
        X_train = builder.transform(X[train_idx])
        X_valid = builder.transform(X[valid_idx])
        model = _build_model(model_recipe)
        model.fit(X_train, y[train_idx])
        pred = np.asarray(model.predict(X_valid), dtype=float).reshape(-1)
        pred = np.clip(pred, float(np.min(y[train_idx])), float(np.max(y[train_idx])))
        _append_oof(oof, fold.fold_id, valid_idx, groups, y[valid_idx], pred, cluster=None)
        y_true_all.extend(y[valid_idx].tolist())
        y_pred_all.extend(pred.tolist())
    summary = _summary(f"{recipe.name}__{model_recipe.name}", recipe.name, model_recipe.name, y_true_all, y_pred_all, oof)
    return {"summary": summary, "oof": oof}


def _evaluate_cluster_fixed(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds,
    wavelengths: np.ndarray,
    recipe: FeatureRecipe,
    *,
    n_clusters: int,
    transductive: bool = False,
) -> dict[str, object]:
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    oof: list[dict[str, object]] = []
    for fold in folds:
        train_idx = np.asarray(fold.train_idx, dtype=int)
        valid_idx = np.asarray(fold.valid_idx, dtype=int)
        builder = _fit_feature_builder(X[train_idx], y[train_idx], wavelengths, recipe)
        X_train = builder.transform(X[train_idx])
        X_valid = builder.transform(X[valid_idx])
        clusterer = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(10, X_train.shape[1], X_train.shape[0] - 1), random_state=42),
            KMeans(n_clusters=min(n_clusters, len(train_idx)), random_state=42, n_init=20),
        )
        cluster_fit = np.vstack([X_train, X_valid]) if transductive else X_train
        clusterer.fit(cluster_fit)
        train_clusters = np.asarray(clusterer.predict(X_train), dtype=int)
        valid_clusters = np.asarray(clusterer.predict(X_valid), dtype=int)
        pred = np.zeros(len(valid_idx), dtype=float)
        fallback = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
        fallback.fit(X_train, y[train_idx])
        for cluster_id in sorted(set(train_clusters.tolist())):
            local_train = np.where(train_clusters == cluster_id)[0]
            local_valid = np.where(valid_clusters == cluster_id)[0]
            if len(local_valid) == 0:
                continue
            if len(local_train) < 40:
                pred[local_valid] = fallback.predict(X_valid[local_valid])
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
            model.fit(X_train[local_train], y[train_idx][local_train])
            pred[local_valid] = model.predict(X_valid[local_valid])
        pred = np.clip(pred, float(np.min(y[train_idx])), float(np.max(y[train_idx])))
        _append_oof(oof, fold.fold_id, valid_idx, groups, y[valid_idx], pred, cluster=valid_clusters)
        y_true_all.extend(y[valid_idx].tolist())
        y_pred_all.extend(pred.tolist())
    prefix = "transductive_cluster" if transductive else "cluster"
    name = f"{recipe.name}__{prefix}_k{n_clusters}"
    summary = _summary(name, recipe.name, f"{prefix}_ridge1000_k{n_clusters}", y_true_all, y_pred_all, oof)
    return {"summary": summary, "oof": oof}


class _FeatureBuilder:
    def __init__(self, recipe: FeatureRecipe, columns: np.ndarray, means: np.ndarray, stds: np.ndarray) -> None:
        self.recipe = recipe
        self.columns = columns
        self.means = means
        self.stds = stds

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = _segment_features(_prep(X, self.recipe.prep), self.recipe)
        features = (features - self.means) / self.stds
        return features[:, self.columns]


def _fit_feature_builder(X: np.ndarray, y: np.ndarray, wavelengths: np.ndarray, recipe: FeatureRecipe) -> _FeatureBuilder:
    raw_features = _segment_features(_prep(X, recipe.prep), recipe)
    means = np.mean(raw_features, axis=0)
    stds = np.maximum(np.std(raw_features, axis=0), 1e-12)
    scaled = (raw_features - means) / stds
    selected = _top_corr_columns(scaled, y, min(recipe.select_k, scaled.shape[1]))
    if recipe.include_band616:
        band_feature_cols = _feature_columns_near_616(recipe, wavelengths, raw_features.shape[1])
        selected = np.asarray(sorted(set(selected.tolist()) | set(band_feature_cols.tolist())), dtype=int)
    return _FeatureBuilder(recipe, selected, means, stds)


def _segment_features(X: np.ndarray, recipe: FeatureRecipe) -> np.ndarray:
    blocks = []
    n = X.shape[1]
    x_axis_cache: dict[int, np.ndarray] = {}
    for start in range(0, n, recipe.step):
        end = min(start + recipe.window, n)
        if end - start < 3:
            continue
        block = X[:, start:end]
        vals = []
        if "mean" in recipe.stats:
            vals.append(np.mean(block, axis=1))
        if "std" in recipe.stats:
            vals.append(np.std(block, axis=1))
        if "min" in recipe.stats:
            vals.append(np.min(block, axis=1))
        if "max" in recipe.stats:
            vals.append(np.max(block, axis=1))
        if "slope" in recipe.stats:
            width = block.shape[1]
            if width not in x_axis_cache:
                axis = np.arange(width, dtype=float)
                x_axis_cache[width] = axis - axis.mean()
            axis = x_axis_cache[width]
            denom = float(np.sum(axis * axis))
            vals.append((block - block.mean(axis=1, keepdims=True)) @ axis / max(denom, 1e-12))
        if "range" in recipe.stats:
            vals.append(np.max(block, axis=1) - np.min(block, axis=1))
        if "edge" in recipe.stats:
            vals.append(block[:, -1] - block[:, 0])
        if vals:
            blocks.append(np.vstack(vals).T)
    features = np.hstack(blocks)
    if recipe.include_ratios:
        # Add stable local contrast features between adjacent segment means.
        means = features[:, :: len(recipe.stats)] if recipe.stats and recipe.stats[0] == "mean" else features[:, :0]
        if means.shape[1] > 1:
            denom = np.maximum(np.abs(means[:, :-1]) + np.abs(means[:, 1:]), 1e-12)
            features = np.hstack([features, (means[:, 1:] - means[:, :-1]) / denom])
    if recipe.include_anchor_ratios:
        # Use segment means as stable cutout levels, then express each cutout
        # relative to proven/selected anchor neighborhoods.
        segment_means = []
        for start in range(0, n, recipe.step):
            end = min(start + recipe.window, n)
            if end - start < 3:
                continue
            segment_means.append(np.mean(X[:, start:end], axis=1))
        if segment_means:
            means = np.vstack(segment_means).T
            anchor_blocks = []
            for center in recipe.anchor_centers:
                anchor = _band_anchor(X, center=center, radius=5)
                anchor_blocks.extend(
                    [
                        means - anchor[:, None],
                        means / np.maximum(np.abs(anchor[:, None]), 1e-12),
                        (means - anchor[:, None])
                        / np.maximum(np.abs(means) + np.abs(anchor[:, None]), 1e-12),
                    ]
                )
            features = np.hstack([features, *anchor_blocks])
    if recipe.include_local616:
        features = np.hstack([features, _local_anchor_features(X, recipe.anchor_centers)])
    return features


def _band_anchor(X: np.ndarray, *, center: int, radius: int) -> np.ndarray:
    start = max(0, center - radius)
    end = min(X.shape[1], center + radius + 1)
    return np.mean(X[:, start:end], axis=1)


def _local_anchor_features(X: np.ndarray, centers: tuple[int, ...]) -> np.ndarray:
    cols = []
    for center in centers:
        for radius in (2, 5, 10, 20, 40):
            start = max(0, center - radius)
            end = min(X.shape[1], center + radius + 1)
            block = X[:, start:end]
            mid = block.shape[1] // 2
            left = block[:, : max(1, mid)]
            right = block[:, min(block.shape[1], mid + 1) :]
            if right.shape[1] == 0:
                right = block[:, -1:]
            axis = np.arange(block.shape[1], dtype=float)
            axis = axis - axis.mean()
            denom = max(float(np.sum(axis * axis)), 1e-12)
            slope = (block - block.mean(axis=1, keepdims=True)) @ axis / denom
            curvature_axis = axis * axis - np.mean(axis * axis)
            curvature = (block - block.mean(axis=1, keepdims=True)) @ curvature_axis / max(
                float(np.sum(curvature_axis * curvature_axis)),
                1e-12,
            )
            local_mean = np.mean(block, axis=1)
            local_std = np.std(block, axis=1)
            local_range = np.max(block, axis=1) - np.min(block, axis=1)
            left_right = np.mean(left, axis=1) - np.mean(right, axis=1)
            peak_base = block[:, mid] - (np.mean(left, axis=1) + np.mean(right, axis=1)) / 2.0
            area = np.trapz(block, axis=1)
            cols.extend([local_mean, local_std, local_range, slope, curvature, left_right, peak_base, area])
    return np.vstack(cols).T


def _feature_columns_near_616(recipe: FeatureRecipe, wavelengths: np.ndarray, n_features: int) -> np.ndarray:
    # Approximate by segment index around raw feature index 616. This intentionally
    # keeps the proven band area in the candidate pool.
    stat_count = max(1, len(recipe.stats))
    segment_idx = 616 // recipe.step
    cols = []
    for seg in range(max(0, segment_idx - 2), segment_idx + 3):
        for offset in range(stat_count):
            col = seg * stat_count + offset
            if 0 <= col < n_features:
                cols.append(col)
    return np.asarray(cols, dtype=int)


def _feature_recipes(preset: str) -> list[FeatureRecipe]:
    stats_basic = ("mean", "slope", "std")
    stats_shape = ("mean", "slope", "std", "range", "edge")
    anchor_centers = (616, 1332, 1485)
    if preset == "focused":
        recipes = []
        for prep in ("raw", "smooth5", "snv", "diff1"):
            for window, step in ((10, 5), (20, 10), (40, 20), (80, 40)):
                for k in (80, 160, 320):
                    recipes.append(
                        FeatureRecipe(
                            f"{prep}_w{window}_s{step}_multi_anchor_k{k}",
                            prep,
                            window,
                            step,
                            stats_shape,
                            k,
                            include_ratios=True,
                            include_anchor_ratios=True,
                            include_local616=True,
                            anchor_centers=anchor_centers,
                        )
                    )
        for window, step in ((10, 5), (20, 10), (40, 20), (80, 40)):
            recipes.append(
                FeatureRecipe(
                    f"diff1_w{window}_s{step}_multi_anchor_basic_k160",
                    "diff1",
                    window,
                    step,
                    stats_basic,
                    160,
                    include_ratios=True,
                    include_anchor_ratios=True,
                    include_local616=True,
                    anchor_centers=anchor_centers,
                )
            )
        return recipes
    if preset == "quick":
        return [
            FeatureRecipe("raw_w10_s5_shape_k80", "raw", 10, 5, stats_shape, 80, include_ratios=True),
            FeatureRecipe("raw_w20_s10_shape_k80", "raw", 20, 10, stats_shape, 80, include_ratios=True),
            FeatureRecipe("smooth5_w10_s5_shape_k80", "smooth5", 10, 5, stats_shape, 80, include_ratios=True),
            FeatureRecipe("smooth5_w20_s10_shape_k80", "smooth5", 20, 10, stats_shape, 80, include_ratios=True),
            FeatureRecipe("snv_w10_s5_shape_k120", "snv", 10, 5, stats_shape, 120, include_ratios=True),
            FeatureRecipe("snv_w20_s10_shape_k120", "snv", 20, 10, stats_shape, 120, include_ratios=True),
            FeatureRecipe("diff1_w20_s10_basic_k80", "diff1", 20, 10, stats_basic, 80, include_ratios=False),
            FeatureRecipe(
                "snv_w20_s10_anchor_ratio_k160",
                "snv",
                20,
                10,
                stats_shape,
                160,
                include_ratios=True,
                include_anchor_ratios=True,
                include_local616=True,
            ),
            FeatureRecipe(
                "raw_w20_s10_anchor_ratio_k160",
                "raw",
                20,
                10,
                stats_shape,
                160,
                include_ratios=True,
                include_anchor_ratios=True,
                include_local616=True,
            ),
            FeatureRecipe(
                "diff1_w20_s10_anchor_shape_k160",
                "diff1",
                20,
                10,
                stats_shape,
                160,
                include_ratios=True,
                include_anchor_ratios=True,
                include_local616=True,
            ),
        ]
    return [
        FeatureRecipe(
            f"{prep}_w{window}_s{step}_shape_k{k}",
            prep,
            window,
            step,
            stats_shape,
            k,
            include_ratios=True,
            include_anchor_ratios=True,
            include_local616=True,
            anchor_centers=anchor_centers,
        )
        for prep in ("raw", "smooth5", "snv", "center", "diff1")
        for window, step in ((8, 4), (10, 5), (20, 10), (40, 20), (80, 40))
        for k in (40, 80, 160, 320)
    ]


def _model_recipes(preset: str) -> list[ModelRecipe]:
    models = [
        ModelRecipe("ridge100", "ridge100"),
        ModelRecipe("ridge1000", "ridge1000"),
        ModelRecipe("ridge3000", "ridge3000"),
        ModelRecipe("pls2", "pls2"),
        ModelRecipe("pls4", "pls4"),
        ModelRecipe("huber", "huber"),
    ]
    if preset in {"quick", "focused"}:
        return models
    return models + [
        ModelRecipe("elastic003", "elastic003"),
        ModelRecipe("extratrees", "extratrees"),
        ModelRecipe("rf", "rf"),
    ]


def _build_model(recipe: ModelRecipe):
    if recipe.kind == "ridge100":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if recipe.kind == "ridge1000":
        return make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
    if recipe.kind == "ridge3000":
        return make_pipeline(StandardScaler(), Ridge(alpha=3000.0))
    if recipe.kind == "pls2":
        return make_pipeline(StandardScaler(), PLSRegression(n_components=2))
    if recipe.kind == "pls4":
        return make_pipeline(StandardScaler(), PLSRegression(n_components=4))
    if recipe.kind == "huber":
        return make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000))
    if recipe.kind == "elastic003":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=10000))
    if recipe.kind == "extratrees":
        return ExtraTreesRegressor(n_estimators=200, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=-1)
    if recipe.kind == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=-1)
    raise ValueError(recipe.kind)


def _prep(X: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return X
    if name == "smooth5":
        p = np.pad(X, ((0, 0), (2, 2)), mode="edge")
        return (p[:, :-4] + p[:, 1:-3] + p[:, 2:-2] + p[:, 3:-1] + p[:, 4:]) / 5.0
    if name == "snv":
        return (X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-12)
    if name == "center":
        return X - X.mean(axis=1, keepdims=True)
    if name == "diff1":
        return np.diff(X, axis=1)
    raise ValueError(name)


def _top_corr_columns(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    x = X - X.mean(axis=0, keepdims=True)
    y0 = y - y.mean()
    denom = np.sqrt(np.sum(x * x, axis=0) * float(np.sum(y0 * y0)))
    corr = (x.T @ y0) / np.maximum(denom, 1e-12)
    return np.argsort(np.abs(corr))[::-1][:k].astype(int)


def _append_oof(
    oof: list[dict[str, object]],
    fold: int,
    indices: np.ndarray,
    groups: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    cluster,
) -> None:
    for pos, idx in enumerate(indices):
        error = float(y_pred[pos] - y_true[pos])
        oof.append(
            {
                "fold": fold,
                "row_index": int(idx),
                "group": str(groups[idx]),
                "cluster": "" if cluster is None else int(cluster[pos]),
                "y_true": float(y_true[pos]),
                "y_pred": float(y_pred[pos]),
                "error": error,
                "abs_error": abs(error),
            }
        )


def _summary(
    name: str,
    features: str,
    model: str,
    y_true: list[float],
    y_pred: list[float],
    oof: list[dict[str, object]],
) -> dict[str, object]:
    errors = np.asarray(y_pred) - np.asarray(y_true)
    worst = _worst_group(oof)
    group_rmse = rmse(y_true, y_pred)
    guard_score = group_rmse + 0.10 * float(worst["rmse"]) + 0.15 * abs(float(worst["bias"]))
    return {
        "name": name,
        "features": features,
        "model": model,
        "group_species_rmse": group_rmse,
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "worst_group": worst["group"],
        "worst_group_rmse": worst["rmse"],
        "worst_group_bias": worst["bias"],
        "guard_score": guard_score,
    }


def _worst_group(oof: list[dict[str, object]]) -> dict[str, object]:
    best = {"group": "", "rmse": -1.0, "bias": 0.0}
    for group in sorted({str(row["group"]) for row in oof}):
        rows = [row for row in oof if str(row["group"]) == group]
        errors = [float(row["error"]) for row in rows]
        value = math.sqrt(sum(e * e for e in errors) / len(errors))
        if value > float(best["rmse"]):
            best = {"group": group, "rmse": value, "bias": sum(errors) / len(errors)}
    return best


def _write_oof(path: Path, oof: list[dict[str, object]]) -> None:
    if not oof:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(oof[0]))
        writer.writeheader()
        writer.writerows(oof)


if __name__ == "__main__":
    main()
