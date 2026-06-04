#!/usr/bin/env python3
"""Large wavelength-set EDA for the NIR moisture task.

This script is diagnostic. It searches contiguous wavelength windows and
derived band sets under several preprocessing views, then runs lightweight OOF
checks so the output is closer to "usable feature sets" than raw correlation.
Test rows are used only through X/meta distribution diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, Ridge
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
class WindowFeature:
    key: str
    view: str
    stat: str
    start: int
    end: int
    wl_start: float
    wl_end: float
    width: int
    values_train: np.ndarray
    values_test: np.ndarray


@dataclass(frozen=True)
class FeatureSet:
    name: str
    keys: tuple[str, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--time-budget-minutes", type=float, default=90.0)
    parser.add_argument("--top-windows", type=int, default=1200)
    parser.add_argument("--top-sets", type=int, default=80)
    parser.add_argument("--mi-top", type=int, default=250)
    args = parser.parse_args()

    started = time.monotonic()
    deadline = started + args.time_budget_minutes * 60.0
    out_dir = args.out_dir or ROOT / "outputs" / f"wavelength_set_search_{datetime.now().strftime('%Y%m%d_%H%M')}"
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

    print("window candidate scan", flush=True)
    windows = _build_window_features(X_train, X_test, wavelengths, deadline)
    window_rows = _score_windows(windows, y, train_species, test_species, train_sample, args.mi_top)
    window_rows.sort(key=lambda row: float(row["usable_score"]), reverse=True)
    _write_csv(out_dir / "window_candidates.csv", window_rows)

    print("region aggregation", flush=True)
    region_rows = _aggregate_regions(window_rows)
    _write_csv(out_dir / "region_candidates.csv", region_rows)

    print("feature set construction", flush=True)
    by_key = {w.key: w for w in windows}
    feature_sets = _make_feature_sets(window_rows, region_rows, args.top_sets)
    set_rows = _score_feature_sets(feature_sets, by_key, y, train_species, raw.train, cfg, deadline)
    set_rows.sort(key=lambda row: (float(row["guard_score"]), float(row["rmse"])))
    _write_csv(out_dir / "feature_set_oof.csv", set_rows)

    print("species-specific bands", flush=True)
    species_rows = _species_specific_windows(window_rows)
    _write_csv(out_dir / "species_specific_windows.csv", species_rows)

    summary = _summary(window_rows, region_rows, set_rows, species_rows, time.monotonic() - started)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    manifest = {
        "out_dir": str(out_dir),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_raw_features": int(X_train.shape[1]),
        "n_window_features": len(windows),
        "n_feature_sets": len(feature_sets),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "leakage_note": "Test targets are never used. Test rows are used only for X/meta distribution diagnostics.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_dir}", flush=True)


def _matrix(rows: list[dict[str, str]], feature_cols: list[str]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in feature_cols] for row in rows], dtype=float)


def _species_label(row: dict[str, str]) -> str:
    number = str(row.get("species number", "")).strip()
    return f"{number}:{SPECIES_NAMES.get(number, f'species_{number}')}"


def _prep(X: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return X
    if name == "smooth5":
        p = np.pad(X, ((0, 0), (2, 2)), mode="edge")
        return (p[:, :-4] + p[:, 1:-3] + p[:, 2:-2] + p[:, 3:-1] + p[:, 4:]) / 5.0
    if name == "smooth11":
        p = np.pad(X, ((0, 0), (5, 5)), mode="edge")
        return sum(p[:, i : i + X.shape[1]] for i in range(11)) / 11.0
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
    if name == "snv_diff2":
        return np.diff(_prep(X, "snv"), n=2, axis=1)
    raise ValueError(name)


def _views() -> tuple[str, ...]:
    return ("raw", "smooth5", "snv", "diff1", "diff2", "smooth5_diff1", "snv_diff1", "snv_diff2")


def _window_grid(n: int) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    for width, step in ((4, 4), (6, 6), (8, 8), (10, 10), (16, 16), (20, 20), (32, 32), (40, 40), (64, 64), (80, 80), (128, 128), (160, 160), (240, 240)):
        for start in range(0, n - 2, step):
            end = min(start + width, n)
            if end - start >= 3:
                specs.append((start, end))
    return specs


def _build_window_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    wavelengths: np.ndarray,
    deadline: float,
) -> list[WindowFeature]:
    features: list[WindowFeature] = []
    for view in _views():
        if time.monotonic() > deadline:
            break
        Z_train = _prep(X_train, view)
        Z_test = _prep(X_test, view)
        n = Z_train.shape[1]
        for start, end in _window_grid(n):
            block_train = Z_train[:, start:end]
            block_test = Z_test[:, start:end]
            stat_values = _window_stats(block_train, block_test)
            for stat, (train_values, test_values) in stat_values.items():
                key = f"{view}:{stat}:{start}:{end}"
                wl_start = _wavelength(wavelengths, start)
                wl_end = _wavelength(wavelengths, end - 1)
                features.append(
                    WindowFeature(
                        key=key,
                        view=view,
                        stat=stat,
                        start=start,
                        end=end,
                        wl_start=wl_start,
                        wl_end=wl_end,
                        width=end - start,
                        values_train=train_values,
                        values_test=test_values,
                    )
                )
    return features


def _window_stats(block_train: np.ndarray, block_test: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    def slope(block: np.ndarray) -> np.ndarray:
        axis = np.arange(block.shape[1], dtype=float)
        axis = axis - axis.mean()
        denom = max(float(np.sum(axis * axis)), 1e-12)
        return (block - block.mean(axis=1, keepdims=True)) @ axis / denom

    def curvature(block: np.ndarray) -> np.ndarray:
        axis = np.arange(block.shape[1], dtype=float)
        axis = axis - axis.mean()
        curve = axis * axis - np.mean(axis * axis)
        denom = max(float(np.sum(curve * curve)), 1e-12)
        return (block - block.mean(axis=1, keepdims=True)) @ curve / denom

    mid_train = block_train[:, block_train.shape[1] // 2]
    mid_test = block_test[:, block_test.shape[1] // 2]
    edge_train = block_train[:, -1] - block_train[:, 0]
    edge_test = block_test[:, -1] - block_test[:, 0]
    return {
        "mean": (np.mean(block_train, axis=1), np.mean(block_test, axis=1)),
        "std": (np.std(block_train, axis=1), np.std(block_test, axis=1)),
        "range": (np.max(block_train, axis=1) - np.min(block_train, axis=1), np.max(block_test, axis=1) - np.min(block_test, axis=1)),
        "slope": (slope(block_train), slope(block_test)),
        "curvature": (curvature(block_train), curvature(block_test)),
        "edge": (edge_train, edge_test),
        "center_minus_mean": (mid_train - np.mean(block_train, axis=1), mid_test - np.mean(block_test, axis=1)),
    }


def _score_windows(
    windows: list[WindowFeature],
    y: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
    train_sample: np.ndarray,
    mi_top: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    X_corr = np.vstack([_standardize(w.values_train) for w in windows]).T
    corr = _corr_columns(X_corr, y)
    spearman = _corr_columns(_rank_columns(X_corr), _rank(y))
    sample_corr = _corr_columns(X_corr, train_sample)
    mi_indices = np.argsort(-np.abs(corr))[: min(mi_top, X_corr.shape[1])]
    mi_values: dict[int, float] = {}
    if len(mi_indices):
        mi = mutual_info_regression(X_corr[:, mi_indices], y, random_state=42)
        mi_values = {int(idx): float(value) for idx, value in zip(mi_indices, mi)}

    species_strengths = _species_strengths(X_corr, y, train_species)
    for i, w in enumerate(windows):
        z_train = _standardize(w.values_train)
        z_test = (w.values_test - float(np.mean(w.values_train))) / max(float(np.std(w.values_train)), 1e-12)
        transfer = _transfer_score(z_train, z_test, train_species, test_species)
        test_sep = _test_species_separation(z_test, test_species)
        abs_corr = abs(float(corr[i]))
        abs_spearman = abs(float(spearman[i]))
        species_abs = species_strengths[i]
        shift_penalty = min(2.5, abs(float(transfer["test_global_shift"]))) + 0.35 * min(3.0, abs(float(transfer["mean_nearest_distance"])))
        usable_score = 0.42 * abs_corr + 0.22 * abs_spearman + 0.18 * species_abs + 0.10 * min(1.0, test_sep) + 0.08 * min(1.0, mi_values.get(i, 0.0)) - 0.08 * shift_penalty
        rows.append(
            {
                "key": w.key,
                "view": w.view,
                "stat": w.stat,
                "start": w.start,
                "end": w.end,
                "width": w.width,
                "wl_start": w.wl_start,
                "wl_end": w.wl_end,
                "pearson": float(corr[i]),
                "abs_pearson": abs_corr,
                "spearman": float(spearman[i]),
                "abs_spearman": abs_spearman,
                "mutual_info": mi_values.get(i, 0.0),
                "species_mean_abs_corr": species_abs,
                "sample_abs_corr": abs(float(sample_corr[i])),
                "test_species_separation": test_sep,
                "test_global_shift": transfer["test_global_shift"],
                "mean_nearest_distance": transfer["mean_nearest_distance"],
                "nearest_train_species_by_test": transfer["nearest_train_species_by_test"],
                "usable_score": usable_score,
            }
        )
    return rows


def _species_strengths(X: np.ndarray, y: np.ndarray, species: np.ndarray) -> np.ndarray:
    acc = np.zeros(X.shape[1], dtype=float)
    total = 0
    for sp in sorted(set(species)):
        mask = species == sp
        if int(mask.sum()) < 8:
            continue
        acc += np.abs(_corr_columns(X[mask], y[mask]))
        total += 1
    return acc / max(total, 1)


def _transfer_score(
    z_train: np.ndarray,
    z_test: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
) -> dict[str, object]:
    train_means = {sp: float(np.mean(z_train[train_species == sp])) for sp in sorted(set(train_species))}
    nearest: dict[str, str] = {}
    distances = []
    for tsp in sorted(set(test_species)):
        test_mean = float(np.mean(z_test[test_species == tsp]))
        best_sp, best_dist = min(
            ((sp, abs(test_mean - train_mean)) for sp, train_mean in train_means.items()),
            key=lambda item: item[1],
        )
        nearest[str(tsp)] = str(best_sp)
        distances.append(best_dist)
    return {
        "test_global_shift": float(np.mean(z_test)),
        "mean_nearest_distance": float(np.mean(distances)) if distances else 0.0,
        "nearest_train_species_by_test": "; ".join(f"{k}->{v}" for k, v in nearest.items()),
    }


def _test_species_separation(z_test: np.ndarray, test_species: np.ndarray) -> float:
    means = []
    within = []
    for sp in sorted(set(test_species)):
        vals = z_test[test_species == sp]
        means.append(float(np.mean(vals)))
        within.append(float(np.var(vals)))
    if len(means) <= 1:
        return 0.0
    return float(np.var(means) / max(np.mean(within), 1e-12))


def _aggregate_regions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        center = (int(row["start"]) + int(row["end"])) // 2
        bucket = 25 * round(center / 25)
        buckets[(str(row["view"]), str(row["stat"]), bucket)].append(row)
    out = []
    for (view, stat, bucket), items in buckets.items():
        top = sorted(items, key=lambda row: float(row["usable_score"]), reverse=True)[:10]
        best = top[0]
        starts = [int(row["start"]) for row in top]
        ends = [int(row["end"]) for row in top]
        out.append(
            {
                "view": view,
                "stat": stat,
                "bucket_center_index": bucket,
                "start_min": min(starts),
                "end_max": max(ends),
                "wl_start": best["wl_start"],
                "wl_end": best["wl_end"],
                "n_support": len(items),
                "best_key": best["key"],
                "best_usable_score": best["usable_score"],
                "best_abs_pearson": best["abs_pearson"],
                "best_abs_spearman": best["abs_spearman"],
                "mean_top_usable_score": float(np.mean([float(row["usable_score"]) for row in top])),
                "mean_top_transfer_distance": float(np.mean([float(row["mean_nearest_distance"]) for row in top])),
            }
        )
    out.sort(key=lambda row: float(row["mean_top_usable_score"]), reverse=True)
    return out


def _make_feature_sets(
    window_rows: list[dict[str, object]],
    region_rows: list[dict[str, object]],
    top_sets: int,
) -> list[FeatureSet]:
    sets: list[FeatureSet] = []
    for view in _views():
        view_keys = _diverse_keys([row for row in window_rows if row["view"] == view], 12)
        if view_keys:
            sets.append(FeatureSet(f"{view}_diverse12", tuple(view_keys)))
    for stat in ("mean", "slope", "edge", "curvature", "center_minus_mean"):
        stat_keys = _diverse_keys([row for row in window_rows if row["stat"] == stat], 16)
        if stat_keys:
            sets.append(FeatureSet(f"{stat}_all_views_diverse16", tuple(stat_keys)))
    for k in (8, 16, 32, 64):
        sets.append(FeatureSet(f"top_usable{k}", tuple(str(row["key"]) for row in window_rows[:k])))
        low_shift = [row for row in window_rows if abs(float(row["test_global_shift"])) < 0.75 and float(row["mean_nearest_distance"]) < 1.25]
        sets.append(FeatureSet(f"low_shift_top{k}", tuple(str(row["key"]) for row in low_shift[:k])))
    for center, label in ((616, "band616"), (1332, "band1332"), (1350, "snv_diff1_peak1350"), (1485, "band1485")):
        near = [row for row in window_rows if abs(((int(row["start"]) + int(row["end"])) // 2) - center) <= 35]
        sets.append(FeatureSet(f"{label}_near_top24", tuple(str(row["key"]) for row in near[:24])))
    for region_k in (8, 16, 32):
        keys = [str(row["best_key"]) for row in region_rows[:region_k]]
        sets.append(FeatureSet(f"top_regions{region_k}", tuple(keys)))
    return [s for s in sets[:top_sets] if s.keys]


def _diverse_keys(rows: list[dict[str, object]], k: int) -> list[str]:
    keys = []
    taken: list[tuple[int, int]] = []
    for row in rows:
        start = int(row["start"])
        end = int(row["end"])
        center = (start + end) // 2
        if any(abs(center - old_center) < max(10, (end - start) // 2) and str(row["view"]) == old_view for old_center, old_view in taken):
            continue
        keys.append(str(row["key"]))
        taken.append((center, str(row["view"])))
        if len(keys) >= k:
            break
    return keys


def _score_feature_sets(
    feature_sets: list[FeatureSet],
    by_key: dict[str, WindowFeature],
    y: np.ndarray,
    train_species: np.ndarray,
    train_rows: list[dict[str, str]],
    cfg,
    deadline: float,
) -> list[dict[str, object]]:
    folds = group_kfold(train_rows, group_col="species number", n_splits=5)
    rows = []
    for feature_set in feature_sets:
        if time.monotonic() > deadline:
            break
        keys = [key for key in feature_set.keys if key in by_key]
        if not keys:
            continue
        X = np.vstack([by_key[key].values_train for key in keys]).T
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        for model_name in ("ridge100", "ridge1000", "elastic003", "pls2", "pls4"):
            pred = np.zeros_like(y, dtype=float)
            oof = []
            for fold in folds:
                model = _model(model_name, X[fold.train_idx].shape)
                model.fit(X[fold.train_idx], y[fold.train_idx])
                fold_pred = np.asarray(model.predict(X[fold.valid_idx]), dtype=float).reshape(-1)
                pred[fold.valid_idx] = fold_pred
                for idx, value in zip(fold.valid_idx, fold_pred):
                    oof.append(
                        {
                            "species": train_species[idx],
                            "y": float(y[idx]),
                            "pred": float(value),
                        }
                    )
            worst = _worst_group(oof)
            base_rmse = rmse(y.tolist(), pred.tolist())
            bias = float(np.mean(pred - y))
            guard = base_rmse + 0.10 * float(worst["rmse"]) + 0.15 * abs(float(worst["bias"]))
            rows.append(
                {
                    "set_name": feature_set.name,
                    "model": model_name,
                    "n_features": len(keys),
                    "rmse": base_rmse,
                    "mae": float(np.mean(np.abs(pred - y))),
                    "bias": bias,
                    "worst_group": worst["group"],
                    "worst_group_rmse": worst["rmse"],
                    "worst_group_bias": worst["bias"],
                    "guard_score": guard,
                    "keys": " | ".join(keys),
                }
            )
    return rows


def _model(name: str, shape: tuple[int, int]):
    if name == "ridge100":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if name == "ridge1000":
        return make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
    if name == "elastic003":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=50000, tol=1e-3))
    if name == "pls2":
        return make_pipeline(StandardScaler(), PLSRegression(n_components=min(2, shape[1], shape[0] - 1)))
    if name == "pls4":
        return make_pipeline(StandardScaler(), PLSRegression(n_components=min(4, shape[1], shape[0] - 1)))
    raise ValueError(name)


def _worst_group(oof: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in oof:
        grouped[str(row["species"])].append((float(row["y"]), float(row["pred"])))
    worst = {"group": "", "rmse": -1.0, "bias": 0.0}
    for group, pairs in grouped.items():
        ys = [p[0] for p in pairs]
        ps = [p[1] for p in pairs]
        value = rmse(ys, ps)
        if value > float(worst["rmse"]):
            worst = {"group": group, "rmse": value, "bias": float(np.mean(np.asarray(ps) - np.asarray(ys)))}
    return worst


def _species_specific_windows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if float(row["species_mean_abs_corr"]) >= 0.82 or float(row["abs_pearson"]) >= 0.88:
            out.append(row)
        if len(out) >= 300:
            break
    return out


def _corr_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xc = X - np.mean(X, axis=0, keepdims=True)
    yc = y - float(np.mean(y))
    denom = np.sqrt(np.sum(Xc * Xc, axis=0) * float(np.sum(yc * yc)))
    return np.divide(Xc.T @ yc, denom, out=np.zeros(X.shape[1], dtype=float), where=denom > 1e-12)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _rank_columns(X: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(_rank, 0, X)


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - float(np.mean(values))) / max(float(np.std(values)), 1e-12)


def _wavelength(wavelengths: np.ndarray, idx: int) -> float:
    idx = max(0, min(int(idx), len(wavelengths) - 1))
    return float(wavelengths[idx])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    window_rows: list[dict[str, object]],
    region_rows: list[dict[str, object]],
    set_rows: list[dict[str, object]],
    species_rows: list[dict[str, object]],
    elapsed: float,
) -> str:
    lines = [
        "# Wavelength Set Search",
        "",
        f"- elapsed_seconds: {elapsed:.1f}",
        "- leakage: test targets were not used; test rows were used only through X/meta.",
        "",
        "## Top Window Candidates",
        "",
    ]
    for row in window_rows[:25]:
        lines.append(
            f"- {row['key']} wl={float(row['wl_start']):.2f}-{float(row['wl_end']):.2f} "
            f"score={float(row['usable_score']):.4f} pearson={float(row['pearson']):.4f} "
            f"species_abs={float(row['species_mean_abs_corr']):.4f} transfer={float(row['mean_nearest_distance']):.4f}"
        )
    lines.extend(["", "## Top Regions", ""])
    for row in region_rows[:20]:
        lines.append(
            f"- {row['view']} {row['stat']} idx~{row['bucket_center_index']} "
            f"wl={float(row['wl_start']):.2f}-{float(row['wl_end']):.2f} "
            f"score={float(row['mean_top_usable_score']):.4f} key={row['best_key']}"
        )
    lines.extend(["", "## OOF Feature Sets", ""])
    for row in set_rows[:20]:
        lines.append(
            f"- {row['set_name']} {row['model']}: rmse={float(row['rmse']):.4f} "
            f"guard={float(row['guard_score']):.4f} worst={row['worst_group']}:{float(row['worst_group_rmse']):.4f}"
        )
    lines.extend(["", "## Species/Strong Windows", ""])
    for row in species_rows[:20]:
        lines.append(
            f"- {row['key']} wl={float(row['wl_start']):.2f}-{float(row['wl_end']):.2f} "
            f"pearson={float(row['pearson']):.4f} species_abs={float(row['species_mean_abs_corr']):.4f}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
