#!/usr/bin/env python3
"""One-hour comprehensive EDA sweep for the NIR moisture task.

The script is diagnostic only. It uses train targets for train-only correlation
and OOF error analysis, and uses test rows only through available X/meta fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import pairwise_distances
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


SPECIES_NAMES = {
    "1": "イチョウ",
    "2": "クスノキ",
    "3": "ウエンジ",
    "4": "ウォールナット",
    "5": "クリ",
    "6": "ケヤキ",
    "7": "スギ",
    "8": "スプルース",
    "9": "タモ",
    "10": "チーク",
    "11": "チェリー",
    "12": "トチ",
    "13": "ナラ",
    "14": "ヒノキ",
    "15": "ベイスギ",
    "16": "米ヒバ",
    "17": "ベイマツ",
    "18": "ヤマザクラ",
    "19": "ホワイトオーク",
}


@dataclass(frozen=True)
class ViewSpec:
    name: str
    prep: str
    representation: str = "full"
    n_components: int = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-budget-minutes", type=float, default=60.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    started = time.monotonic()
    deadline = started + args.time_budget_minutes * 60.0
    out_dir = args.out_dir or ROOT / "outputs" / f"comprehensive_eda_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    raw = load_raw_data(cfg)
    feature_cols = [c for c in raw.train[0] if c not in set(cfg.meta_cols) | {cfg.target_col}]
    wavelengths = np.asarray([float(c) for c in feature_cols], dtype=float)
    X_train = _matrix(raw.train, feature_cols)
    X_test = _matrix(raw.test, feature_cols)
    y = np.asarray([float(row[cfg.target_col]) for row in raw.train], dtype=float)
    train_species = np.asarray([_species_label(row) for row in raw.train], dtype=object)
    test_species = np.asarray([_species_label(row) for row in raw.test], dtype=object)
    train_sample = np.asarray([float(row[cfg.id_col]) for row in raw.train], dtype=float)
    test_sample = np.asarray([float(row[cfg.id_col]) for row in raw.test], dtype=float)

    manifest: dict[str, object] = {
        "out_dir": str(out_dir),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "time_budget_minutes": args.time_budget_minutes,
        "leakage_note": "Test targets are never used. Test rows are used only through X/sample number/species number.",
    }

    train_corr_rows: list[dict[str, object]] = []
    species_corr_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    sample_order_rows: list[dict[str, object]] = []
    monotonic_rows: list[dict[str, object]] = []
    oof_error_rows: list[dict[str, object]] = []

    print("train feature correlations", flush=True)
    for view in _correlation_views():
        if _expired(deadline):
            break
        Z = _prep(X_train, view.prep)
        train_corr_rows.extend(_train_correlations(view.name, Z, wavelengths, y, top_k=args.top_k))
        species_corr_rows.extend(_specieswise_correlations(view.name, Z, wavelengths, y, train_species, top_k=20))
        anchor_rows.extend(_anchor_neighborhood(view.name, Z, wavelengths, y))
    _write_csv(out_dir / "train_feature_correlations.csv", train_corr_rows)
    _write_csv(out_dir / "specieswise_correlations.csv", species_corr_rows)
    _write_csv(out_dir / "anchor_neighborhood_correlations.csv", anchor_rows)

    print("train/test distribution and clustering", flush=True)
    for view in _distribution_views():
        if _expired(deadline):
            break
        train_repr, test_repr, explained = _fit_representation(X_train, X_test, view)
        distribution_rows.extend(
            _distribution_matches(view.name, train_repr, test_repr, train_species, test_species, y, explained)
        )
        if view.name == "diff2_pca5":
            for k in (5, 8, 10):
                cluster_rows.extend(_cluster_composition(view.name, train_repr, test_repr, train_species, test_species, k))
    _write_csv(out_dir / "train_test_distribution_matches.csv", distribution_rows)
    _write_csv(out_dir / "cluster_composition_k5_k8_k10.csv", cluster_rows)

    print("sample-order monotonicity", flush=True)
    sample_order_rows = _sample_order_monotonicity(train_sample, y, train_species)
    _write_csv(out_dir / "sample_order_monotonicity.csv", sample_order_rows)

    print("monotonic postprocess audit", flush=True)
    monotonic_rows = _monotonic_postprocess_audit(raw.train, cfg.id_col)
    _write_csv(out_dir / "monotonic_postprocess_audit.csv", monotonic_rows)

    print("oof error correlation audit", flush=True)
    train_aux = _auxiliary_train_features(X_train, y, train_sample, train_species)
    oof_error_rows = _oof_error_correlation_audit(raw.train, cfg.id_col, train_aux)
    _write_csv(out_dir / "oof_error_correlation_audit.csv", oof_error_rows)

    summary = _build_summary(
        train_corr_rows,
        species_corr_rows,
        anchor_rows,
        distribution_rows,
        cluster_rows,
        sample_order_rows,
        monotonic_rows,
        oof_error_rows,
        elapsed=time.monotonic() - started,
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_dir}", flush=True)


def _matrix(rows: list[dict[str, str]], feature_cols: list[str]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in feature_cols] for row in rows], dtype=float)


def _species_label(row: dict[str, str]) -> str:
    number = str(row.get("species number", "")).strip()
    return f"{number}:{SPECIES_NAMES.get(number, f'species_{number}')}"


def _correlation_views() -> list[ViewSpec]:
    return [
        ViewSpec("raw", "raw"),
        ViewSpec("smooth5", "smooth5"),
        ViewSpec("center", "center"),
        ViewSpec("snv", "snv"),
        ViewSpec("diff1", "diff1"),
        ViewSpec("diff2", "diff2"),
        ViewSpec("smooth5_diff1", "smooth5_diff1"),
        ViewSpec("snv_diff1", "snv_diff1"),
    ]


def _distribution_views() -> list[ViewSpec]:
    specs = []
    for prep in ("raw", "smooth5", "snv", "center", "diff1", "diff2"):
        for n_components in (5, 10, 20):
            specs.append(ViewSpec(f"{prep}_pca{n_components}", prep, "pca", n_components))
    for prep in ("smooth5", "snv", "diff1", "diff2"):
        for n_components in (5, 10, 20):
            specs.append(ViewSpec(f"{prep}_segment_pca{n_components}", prep, "segment", n_components))
    return specs


def _prep(X: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return X
    if name == "smooth5":
        p = np.pad(X, ((0, 0), (2, 2)), mode="edge")
        return (p[:, :-4] + p[:, 1:-3] + p[:, 2:-2] + p[:, 3:-1] + p[:, 4:]) / 5.0
    if name == "center":
        return X - np.mean(X, axis=1, keepdims=True)
    if name == "snv":
        return (X - np.mean(X, axis=1, keepdims=True)) / np.maximum(np.std(X, axis=1, keepdims=True), 1e-12)
    if name == "diff1":
        return np.diff(X, axis=1)
    if name == "diff2":
        return np.diff(X, n=2, axis=1)
    if name == "smooth5_diff1":
        return np.diff(_prep(X, "smooth5"), axis=1)
    if name == "snv_diff1":
        return np.diff(_prep(X, "snv"), axis=1)
    raise ValueError(name)


def _segment_features(X: np.ndarray, *, window: int = 20, step: int = 10) -> np.ndarray:
    blocks = []
    axis_cache: dict[int, np.ndarray] = {}
    for start in range(0, X.shape[1], step):
        end = min(start + window, X.shape[1])
        if end - start < 3:
            continue
        block = X[:, start:end]
        width = block.shape[1]
        if width not in axis_cache:
            axis = np.arange(width, dtype=float)
            axis_cache[width] = axis - np.mean(axis)
        axis = axis_cache[width]
        denom = max(float(np.sum(axis * axis)), 1e-12)
        slope = (block - block.mean(axis=1, keepdims=True)) @ axis / denom
        curve_axis = axis * axis - np.mean(axis * axis)
        curve = (block - block.mean(axis=1, keepdims=True)) @ curve_axis / max(
            float(np.sum(curve_axis * curve_axis)), 1e-12
        )
        blocks.append(
            np.vstack(
                [
                    np.mean(block, axis=1),
                    np.std(block, axis=1),
                    np.max(block, axis=1) - np.min(block, axis=1),
                    slope,
                    curve,
                    block[:, -1] - block[:, 0],
                ]
            ).T
        )
    return np.hstack(blocks)


def _pearson_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = X - np.mean(X, axis=0, keepdims=True)
    y0 = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x, axis=0) * float(np.sum(y0 * y0)))
    return (x.T @ y0) / np.maximum(denom, 1e-12)


def _spearman_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Approximate enough for EDA. Ranking all columns is still cheap at this size.
    ranks = np.apply_along_axis(_rank, 0, X)
    return _pearson_columns(ranks, _rank(y))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _train_correlations(view: str, X: np.ndarray, wavelengths: np.ndarray, y: np.ndarray, *, top_k: int) -> list[dict[str, object]]:
    pearson = _pearson_columns(X, y)
    spearman = _spearman_columns(X, y)
    order = np.argsort(np.abs(pearson))[::-1][:top_k]
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "view": view,
                "rank": rank,
                "index": int(idx),
                "wavelength": _wavelength(wavelengths, idx),
                "pearson": float(pearson[idx]),
                "abs_pearson": abs(float(pearson[idx])),
                "spearman": float(spearman[idx]),
                "abs_spearman": abs(float(spearman[idx])),
            }
        )
    return rows


def _specieswise_correlations(
    view: str, X: np.ndarray, wavelengths: np.ndarray, y: np.ndarray, species: np.ndarray, *, top_k: int
) -> list[dict[str, object]]:
    rows = []
    for label in sorted(set(species.tolist()), key=_sort_label):
        idx = np.where(species == label)[0]
        if len(idx) < 10:
            continue
        corr = _pearson_columns(X[idx], y[idx])
        order = np.argsort(np.abs(corr))[::-1][:top_k]
        for rank, col in enumerate(order, start=1):
            rows.append(
                {
                    "view": view,
                    "species": label,
                    "n": int(len(idx)),
                    "rank": rank,
                    "index": int(col),
                    "wavelength": _wavelength(wavelengths, col),
                    "pearson": float(corr[col]),
                    "abs_pearson": abs(float(corr[col])),
                }
            )
    return rows


def _anchor_neighborhood(view: str, X: np.ndarray, wavelengths: np.ndarray, y: np.ndarray) -> list[dict[str, object]]:
    corr = _pearson_columns(X, y)
    rows = []
    for center in (616, 1332, 1485):
        for idx in range(max(0, center - 20), min(X.shape[1], center + 21)):
            rows.append(
                {
                    "view": view,
                    "anchor": center,
                    "index": idx,
                    "offset": idx - center,
                    "wavelength": _wavelength(wavelengths, idx),
                    "pearson": float(corr[idx]),
                    "abs_pearson": abs(float(corr[idx])),
                }
            )
    return rows


def _fit_representation(X_train: np.ndarray, X_test: np.ndarray, view: ViewSpec) -> tuple[np.ndarray, np.ndarray, float]:
    Z_train = _prep(X_train, view.prep)
    Z_test = _prep(X_test, view.prep)
    if view.representation == "segment":
        Z_train = _segment_features(Z_train)
        Z_test = _segment_features(Z_test)
    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Z_train)
    Z_test = scaler.transform(Z_test)
    n_components = min(view.n_components, Z_train.shape[1], Z_train.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    train_repr = pca.fit_transform(Z_train)
    test_repr = pca.transform(Z_test)
    return train_repr, test_repr, float(np.sum(pca.explained_variance_ratio_))


def _distribution_matches(
    view: str,
    train_repr: np.ndarray,
    test_repr: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
    y: np.ndarray,
    explained: float,
) -> list[dict[str, object]]:
    train_centroids = _centroids(train_repr, train_species)
    test_centroids = _centroids(test_repr, test_species)
    train_labels = list(train_centroids)
    train_matrix = np.vstack([train_centroids[label] for label in train_labels])
    rows = []
    for test_label, centroid in test_centroids.items():
        d = pairwise_distances(centroid[None, :], train_matrix, metric="cosine")[0]
        order = np.argsort(d)
        for rank, pos in enumerate(order[:8], start=1):
            train_label = train_labels[int(pos)]
            train_idx = train_species == train_label
            rows.append(
                {
                    "view": view,
                    "explained": explained,
                    "test_species": test_label,
                    "test_n": int(np.sum(test_species == test_label)),
                    "rank": rank,
                    "train_species": train_label,
                    "train_n": int(np.sum(train_idx)),
                    "cosine_distance": float(d[pos]),
                    "train_target_mean": float(np.mean(y[train_idx])),
                    "train_target_std": float(np.std(y[train_idx])),
                }
            )
    return rows


def _cluster_composition(
    view: str, train_repr: np.ndarray, test_repr: np.ndarray, train_species: np.ndarray, test_species: np.ndarray, k: int
) -> list[dict[str, object]]:
    X = np.vstack([train_repr, test_repr])
    domain = np.asarray(["train"] * len(train_repr) + ["test"] * len(test_repr), dtype=object)
    species = np.concatenate([train_species, test_species])
    labels = KMeans(n_clusters=k, random_state=42, n_init=30).fit_predict(X)
    rows = []
    for cluster_id in sorted(set(labels.tolist())):
        idx = labels == cluster_id
        train_items = species[idx & (domain == "train")].tolist()
        test_items = species[idx & (domain == "test")].tolist()
        rows.append(
            {
                "view": view,
                "k": k,
                "cluster": int(cluster_id),
                "n": int(np.sum(idx)),
                "train_n": len(train_items),
                "test_n": len(test_items),
                "test_ratio": float(len(test_items) / max(1, int(np.sum(idx)))),
                "top_train_species": _top_counts(train_items, 8),
                "top_test_species": _top_counts(test_items, 8),
            }
        )
    return rows


def _sample_order_monotonicity(sample: np.ndarray, y: np.ndarray, species: np.ndarray) -> list[dict[str, object]]:
    rows = []
    rows.append(_sample_order_row("ALL", sample, y))
    for label in sorted(set(species.tolist()), key=_sort_label):
        idx = np.where(species == label)[0]
        rows.append(_sample_order_row(label, sample[idx], y[idx]))
    return rows


def _sample_order_row(label: str, sample: np.ndarray, y: np.ndarray) -> dict[str, object]:
    order = np.argsort(sample)
    x = sample[order]
    yy = y[order]
    diffs = np.diff(yy)
    pear = _corr(x, yy)
    spear = _corr(_rank(x), _rank(yy))
    slope = float(np.polyfit(x, yy, deg=1)[0]) if len(x) > 1 else 0.0
    return {
        "species": label,
        "n": int(len(y)),
        "pearson": pear,
        "spearman": spear,
        "linear_slope": slope,
        "decrease_rate": float(np.mean(diffs < 0)) if len(diffs) else 0.0,
        "median_adjacent_delta": float(np.median(diffs)) if len(diffs) else 0.0,
        "mean_adjacent_delta": float(np.mean(diffs)) if len(diffs) else 0.0,
        "first_target": float(yy[0]) if len(yy) else 0.0,
        "last_target": float(yy[-1]) if len(yy) else 0.0,
    }


def _monotonic_postprocess_audit(train_rows: list[dict[str, str]], id_col: str) -> list[dict[str, object]]:
    sample_by_row = {i: float(row[id_col]) for i, row in enumerate(train_rows)}
    rows = []
    for path in _oof_files():
        if not path.exists():
            continue
        oof = list(csv.DictReader(path.open(encoding="utf-8")))
        y = np.asarray([float(row["y_true"]) for row in oof], dtype=float)
        pred = np.asarray([float(row["y_pred"]) for row in oof], dtype=float)
        groups = np.asarray([str(row["group"]) for row in oof], dtype=object)
        sample = np.asarray([sample_by_row[int(row["row_index"])] for row in oof], dtype=float)
        variants = {
            "base": pred,
            "isotonic_decreasing": _isotonic_decreasing(pred, sample, groups),
            "sort_decreasing": _sort_decreasing(pred, sample, groups),
        }
        for name, values in variants.items():
            errors = values - y
            worst = _worst_group(groups, y, values)
            rows.append(
                {
                    "oof_file": path.name,
                    "variant": name,
                    "rmse": rmse(y.tolist(), values.tolist()),
                    "mae": float(np.mean(np.abs(errors))),
                    "bias": float(np.mean(errors)),
                    "worst_group": worst["group"],
                    "worst_group_rmse": worst["rmse"],
                    "worst_group_bias": worst["bias"],
                }
            )
    return rows


def _oof_error_correlation_audit(
    train_rows: list[dict[str, str]], id_col: str, aux: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    sample_by_row = {i: float(row[id_col]) for i, row in enumerate(train_rows)}
    species_by_row = {i: _species_label(row) for i, row in enumerate(train_rows)}
    rows = []
    for path in _oof_files():
        if not path.exists():
            continue
        oof = list(csv.DictReader(path.open(encoding="utf-8")))
        row_idx = np.asarray([int(row["row_index"]) for row in oof], dtype=int)
        error = np.asarray([float(row["error"]) for row in oof], dtype=float)
        abs_error = np.abs(error)
        samples = np.asarray([sample_by_row[int(i)] for i in row_idx], dtype=float)
        species = np.asarray([species_by_row[int(i)] for i in row_idx], dtype=object)
        feature_map = {
            "sample_number": samples,
            "sample_rank_within_species": _rank_within_groups(samples, species),
        }
        for key, values in aux.items():
            feature_map[key] = values[row_idx]
        for name, values in feature_map.items():
            rows.append(
                {
                    "oof_file": path.name,
                    "feature": name,
                    "corr_error": _corr(values, error),
                    "corr_abs_error": _corr(values, abs_error),
                    "spearman_error": _corr(_rank(values), _rank(error)),
                    "spearman_abs_error": _corr(_rank(values), _rank(abs_error)),
                }
            )
    rows.sort(key=lambda row: abs(float(row["corr_abs_error"])), reverse=True)
    return rows


def _auxiliary_train_features(X: np.ndarray, y: np.ndarray, sample: np.ndarray, species: np.ndarray) -> dict[str, np.ndarray]:
    aux: dict[str, np.ndarray] = {}
    raw = X
    smooth = _prep(X, "smooth5")
    snv = _prep(X, "snv")
    diff1 = _prep(X, "diff1")
    diff2 = _prep(X, "diff2")
    aux["raw_band616"] = raw[:, min(616, raw.shape[1] - 1)]
    aux["smooth5_band616"] = smooth[:, min(616, smooth.shape[1] - 1)]
    aux["snv_band1485"] = snv[:, min(1485, snv.shape[1] - 1)]
    aux["diff1_band1332"] = diff1[:, min(1332, diff1.shape[1] - 1)]
    aux["diff2_band1332"] = diff2[:, min(1332, diff2.shape[1] - 1)]
    aux["sample_number"] = sample
    aux["sample_rank_within_species"] = _rank_within_groups(sample, species)
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=5, random_state=42),
    )
    pcs = model.fit_transform(diff2)
    for i in range(pcs.shape[1]):
        aux[f"diff2_pca5_pc{i+1}"] = pcs[:, i]
    return aux


def _oof_files() -> list[Path]:
    return [
        ROOT / "outputs" / "segment_feature_focused3_smooth5w20_k80_oof" / "smooth5_w20_s10_multi_anchor_k80__transductive_cluster_k5.csv",
        ROOT / "outputs" / "segment_feature_anchor_oof" / "snv_w20_s10_anchor_ratio_k160__cluster_k3.csv",
        ROOT / "outputs" / "routed_model_grid_diff2pca5_k8_oof" / "snv_window20pca1__elastic010_l05.csv",
        ROOT / "outputs" / "cluster_routed_diff2pca5_k8_oof" / "cluster_diff2_k8.csv",
    ]


def _isotonic_decreasing(pred: np.ndarray, sample: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = pred.copy()
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        if len(idx) < 3:
            continue
        order = idx[np.argsort(sample[idx])]
        out[order] = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(sample[order], pred[order])
    return out


def _sort_decreasing(pred: np.ndarray, sample: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = pred.copy()
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        order = idx[np.argsort(sample[idx])]
        out[order] = np.sort(pred[idx])[::-1]
    return out


def _rank_within_groups(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        order = idx[np.argsort(values[idx])]
        denom = max(len(order) - 1, 1)
        for rank, pos in enumerate(order):
            out[pos] = rank / denom
    return out


def _centroids(X: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    return {label: np.mean(X[labels == label], axis=0) for label in sorted(set(labels.tolist()), key=_sort_label)}


def _worst_group(groups: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    best = {"group": "", "rmse": -1.0, "bias": 0.0}
    for group in sorted(set(groups.tolist())):
        idx = groups == group
        errors = y_pred[idx] - y_true[idx]
        value = rmse(y_true[idx].tolist(), y_pred[idx].tolist())
        if value > float(best["rmse"]):
            best = {"group": group, "rmse": value, "bias": float(np.mean(errors))}
    return best


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _wavelength(wavelengths: np.ndarray, idx: int) -> float:
    if 0 <= idx < len(wavelengths):
        return float(wavelengths[idx])
    return float("nan")


def _top_counts(items: list[str], n: int) -> str:
    return "; ".join(f"{label}={count}" for label, count in Counter(items).most_common(n))


def _sort_label(label: str) -> tuple[int, str]:
    prefix = label.split(":", 1)[0]
    try:
        return (int(prefix), label)
    except ValueError:
        return (10_000, label)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    train_corr_rows: list[dict[str, object]],
    species_corr_rows: list[dict[str, object]],
    anchor_rows: list[dict[str, object]],
    distribution_rows: list[dict[str, object]],
    cluster_rows: list[dict[str, object]],
    sample_order_rows: list[dict[str, object]],
    monotonic_rows: list[dict[str, object]],
    oof_error_rows: list[dict[str, object]],
    *,
    elapsed: float,
) -> str:
    lines = [
        "# Comprehensive EDA Search",
        "",
        f"- elapsed_seconds: {elapsed:.1f}",
        "- leakage: test targets were not used; test rows were used only through X/meta.",
        "",
        "## Top Train Correlations",
        "",
    ]
    for row in sorted(train_corr_rows, key=lambda r: float(r["abs_pearson"]), reverse=True)[:15]:
        lines.append(
            f"- {row['view']} idx={row['index']} wl={float(row['wavelength']):.2f} "
            f"pearson={float(row['pearson']):.4f} spearman={float(row['spearman']):.4f}"
        )
    lines.extend(["", "## Anchor Neighborhood Highlights", ""])
    for row in sorted(anchor_rows, key=lambda r: float(r["abs_pearson"]), reverse=True)[:15]:
        lines.append(
            f"- {row['view']} anchor={row['anchor']} idx={row['index']} "
            f"offset={row['offset']} pearson={float(row['pearson']):.4f}"
        )
    lines.extend(["", "## Train/Test Distribution Matches", ""])
    seen = set()
    for row in sorted(distribution_rows, key=lambda r: (str(r["test_species"]), int(r["rank"]))):
        key = (row["view"], row["test_species"])
        if row["rank"] == 1 and row["view"] in {"diff2_pca5", "snv_segment_pca20", "smooth5_pca20"}:
            lines.append(
                f"- {row['view']} {row['test_species']} -> {row['train_species']} "
                f"distance={float(row['cosine_distance']):.4f} train_target_mean={float(row['train_target_mean']):.2f}"
            )
            seen.add(key)
    lines.extend(["", "## Cluster Composition Highlights", ""])
    for row in [r for r in cluster_rows if str(r["view"]) == "diff2_pca5" and int(r["k"]) in {5, 8} and int(r["test_n"]) > 0]:
        lines.append(
            f"- k={row['k']} cluster={row['cluster']} train=[{row['top_train_species']}] "
            f"test=[{row['top_test_species']}]"
        )
    lines.extend(["", "## Sample Order", ""])
    for row in sample_order_rows:
        if row["species"] == "ALL" or abs(float(row["spearman"])) > 0.95:
            lines.append(
                f"- {row['species']}: n={row['n']} spearman={float(row['spearman']):.4f} "
                f"decrease_rate={float(row['decrease_rate']):.3f}"
            )
    lines.extend(["", "## Monotonic Postprocess Audit", ""])
    for row in sorted(monotonic_rows, key=lambda r: (str(r["oof_file"]), float(r["rmse"]))):
        lines.append(
            f"- {row['oof_file']} {row['variant']}: rmse={float(row['rmse']):.4f} "
            f"worst={row['worst_group']}:{float(row['worst_group_rmse']):.4f}"
        )
    lines.extend(["", "## OOF Error Correlation Leads", ""])
    for row in sorted(oof_error_rows, key=lambda r: abs(float(r["corr_abs_error"])), reverse=True)[:20]:
        lines.append(
            f"- {row['oof_file']} feature={row['feature']} "
            f"corr_abs_error={float(row['corr_abs_error']):.4f} corr_error={float(row['corr_error']):.4f}"
        )
    lines.extend(
        [
            "",
            "## Suggested Next Experiments",
            "",
            "- Implement species-wise sample-number monotonic postprocess for the current best candidate.",
            "- Use diff2_pca5 cluster only as a routing/diagnostic key, not as a direct target correction.",
            "- Prioritize weak/shrunk corrections based on OOF error correlations rather than full high-capacity models.",
        ]
    )
    return "\n".join(lines) + "\n"


def _expired(deadline: float) -> bool:
    return time.monotonic() >= deadline


if __name__ == "__main__":
    main()
