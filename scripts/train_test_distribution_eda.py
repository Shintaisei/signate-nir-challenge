#!/usr/bin/env python3
"""EDA for train/test species distribution matching under spectral views."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402
from pipeline.data import load_raw_data  # noqa: E402


@dataclass(frozen=True)
class View:
    name: str
    prep: str
    representation: str
    n_components: int = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "distribution_eda")
    parser.add_argument("--sample-limit", type=int, default=120, help="max rows per species for pairwise distribution distance")
    args = parser.parse_args()

    config = load_config()
    raw = load_raw_data(config)
    feature_cols = [c for c in raw.train[0] if c not in set(config.meta_cols) | {config.target_col}]
    X_train = np.asarray([[float(row[c]) for c in feature_cols] for row in raw.train], dtype=float)
    X_test = np.asarray([[float(row[c]) for c in feature_cols] for row in raw.test], dtype=float)
    train_species = np.asarray([_species_label(row, config.meta_cols) for row in raw.train], dtype=object)
    test_species = np.asarray([_species_label(row, config.meta_cols) for row in raw.test], dtype=object)
    y_train = np.asarray([float(row[config.target_col]) for row in raw.train], dtype=float)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    views = _views()
    nearest_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    pseudo_rows: list[dict[str, object]] = []

    for view in views:
        print(f"view {view.name}", flush=True)
        train_repr, test_repr, explained = _fit_view(X_train, X_test, view)
        nearest_rows.extend(_nearest_species_rows(view, train_repr, test_repr, train_species, test_species, y_train))
        pair_rows.extend(
            _distribution_pair_rows(
                view,
                train_repr,
                test_repr,
                train_species,
                test_species,
                y_train,
                sample_limit=args.sample_limit,
            )
        )
        cluster_rows.extend(_cluster_rows(view, train_repr, test_repr, train_species, test_species))
        embedding_rows.extend(_embedding_rows(view, train_repr, test_repr, train_species, test_species))
        pseudo_rows.extend(_pseudo_test_alignment_rows(view, X_train, train_species, y_train, sample_limit=args.sample_limit))
        print(f"  explained={explained:.4f}", flush=True)

    _write_csv(args.out_dir / "nearest_train_species_by_test_species.csv", nearest_rows)
    _write_csv(args.out_dir / "train_test_species_distribution_pairs.csv", pair_rows)
    _write_csv(args.out_dir / "mixed_cluster_composition.csv", cluster_rows)
    _write_csv(args.out_dir / "species_centroid_embedding.csv", embedding_rows)
    _write_csv(args.out_dir / "pseudo_test_alignment_score.csv", pseudo_rows)

    consensus = _consensus_rows(pair_rows)
    _write_csv(args.out_dir / "test_species_match_consensus.csv", consensus)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "n_features": int(X_train.shape[1]),
                "train_species": dict(Counter(train_species.tolist())),
                "test_species": dict(Counter(test_species.tolist())),
                "views": [view.name for view in views],
                "pseudo_test_note": "Each train species is held out, represented with transforms fit on the remaining train rows only, then matched by X distribution.",
                "note": "All matching uses X only for test rows; train targets are reported only as context.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {args.out_dir}")


def _species_label(row: dict[str, str], meta_cols: tuple[str, ...]) -> str:
    number = str(row.get("species number", "")).strip()
    name = _SPECIES_NAMES.get(number, f"species_{number}")
    return f"{number}:{name}"


_SPECIES_NAMES = {
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


def _views() -> list[View]:
    return [
        View("raw_pca20", "raw", "pca", 20),
        View("smooth5_pca20", "smooth5", "pca", 20),
        View("snv_pca20", "snv", "pca", 20),
        View("center_pca20", "center", "pca", 20),
        View("diff1_pca20", "diff1", "pca", 20),
        View("diff2_pca20", "diff2", "pca", 20),
        View("smooth5_segment_shape_pca20", "smooth5", "segment", 20),
        View("snv_segment_shape_pca20", "snv", "segment", 20),
        View("diff1_segment_shape_pca20", "diff1", "segment", 20),
    ]


def _fit_view(X_train: np.ndarray, X_test: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, float]:
    Z_train = _prep(X_train, view.prep)
    Z_test = _prep(X_test, view.prep)
    if view.representation == "segment":
        Z_train = _segment_shape_features(Z_train)
        Z_test = _segment_shape_features(Z_test)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(Z_train)
    test_scaled = scaler.transform(Z_test)
    n_components = min(view.n_components, train_scaled.shape[1], train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    train_repr = pca.fit_transform(train_scaled)
    test_repr = pca.transform(test_scaled)
    return train_repr, test_repr, float(np.sum(pca.explained_variance_ratio_))


def _fit_view_train_valid(X_train: np.ndarray, X_valid: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray]:
    train_repr, valid_repr, _ = _fit_view(X_train, X_valid, view)
    return train_repr, valid_repr


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
    if name == "diff2":
        return np.diff(X, n=2, axis=1)
    raise ValueError(name)


def _segment_shape_features(X: np.ndarray, *, window: int = 20, step: int = 10) -> np.ndarray:
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
            axis_cache[width] = axis - axis.mean()
        axis = axis_cache[width]
        denom = max(float(np.sum(axis * axis)), 1e-12)
        slope = (block - block.mean(axis=1, keepdims=True)) @ axis / denom
        blocks.append(
            np.vstack(
                [
                    np.mean(block, axis=1),
                    np.std(block, axis=1),
                    np.max(block, axis=1) - np.min(block, axis=1),
                    slope,
                    block[:, -1] - block[:, 0],
                ]
            ).T
        )
    return np.hstack(blocks)


def _nearest_species_rows(
    view: View,
    train_repr: np.ndarray,
    test_repr: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
    y_train: np.ndarray,
) -> list[dict[str, object]]:
    train_centroids = _centroids(train_repr, train_species)
    test_centroids = _centroids(test_repr, test_species)
    train_labels = list(train_centroids)
    train_matrix = np.vstack([train_centroids[label] for label in train_labels])
    rows = []
    for test_label, centroid in test_centroids.items():
        distances = pairwise_distances(centroid[None, :], train_matrix, metric="cosine")[0]
        order = np.argsort(distances)
        top = [train_labels[int(i)] for i in order[:5]]
        rows.append(
            {
                "view": view.name,
                "test_species": test_label,
                "test_n": int(np.sum(test_species == test_label)),
                "nearest_train_1": top[0],
                "distance_1": float(distances[order[0]]),
                "nearest_train_2": top[1],
                "distance_2": float(distances[order[1]]),
                "nearest_train_3": top[2],
                "distance_3": float(distances[order[2]]),
                "nearest_train_4": top[3],
                "distance_4": float(distances[order[3]]),
                "nearest_train_5": top[4],
                "distance_5": float(distances[order[4]]),
                "nearest_train_1_target_mean": _target_mean_for_species(top[0], train_species, y_train),
            }
        )
    return rows


def _distribution_pair_rows(
    view: View,
    train_repr: np.ndarray,
    test_repr: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
    y_train: np.ndarray,
    *,
    sample_limit: int,
) -> list[dict[str, object]]:
    rows = []
    for test_label in sorted(set(test_species.tolist()), key=_sort_label):
        test_idx = np.where(test_species == test_label)[0]
        Xt = _limited(test_repr[test_idx], sample_limit)
        test_spread = _mean_pairwise_distance(Xt)
        scored = []
        for train_label in sorted(set(train_species.tolist()), key=_sort_label):
            train_idx = np.where(train_species == train_label)[0]
            Xr = _limited(train_repr[train_idx], sample_limit)
            cross = float(np.mean(pairwise_distances(Xt, Xr, metric="euclidean")))
            train_spread = _mean_pairwise_distance(Xr)
            energy = max(0.0, 2.0 * cross - test_spread - train_spread)
            centroid_cosine = float(
                pairwise_distances(np.mean(Xt, axis=0, keepdims=True), np.mean(Xr, axis=0, keepdims=True), metric="cosine")[
                    0, 0
                ]
            )
            scored.append(
                {
                    "view": view.name,
                    "test_species": test_label,
                    "train_species": train_label,
                    "test_n": int(len(test_idx)),
                    "train_n": int(len(train_idx)),
                    "energy_distance": energy,
                    "cross_mean_distance": cross,
                    "centroid_cosine_distance": centroid_cosine,
                    "train_target_mean": _target_mean_for_species(train_label, train_species, y_train),
                    "train_target_std": _target_std_for_species(train_label, train_species, y_train),
                }
            )
        scored.sort(key=lambda row: (float(row["energy_distance"]), float(row["centroid_cosine_distance"])))
        for rank, row in enumerate(scored[:8], start=1):
            row["rank"] = rank
            rows.append(row)
    return rows


def _pseudo_test_alignment_rows(
    view: View,
    X: np.ndarray,
    species: np.ndarray,
    y: np.ndarray,
    *,
    sample_limit: int,
) -> list[dict[str, object]]:
    rows = []
    labels = sorted(set(species.tolist()), key=_sort_label)
    for heldout in labels:
        train_idx = np.where(species != heldout)[0]
        valid_idx = np.where(species == heldout)[0]
        train_repr, valid_repr = _fit_view_train_valid(X[train_idx], X[valid_idx], view)
        train_species = species[train_idx]
        heldout_species = np.asarray([heldout] * len(valid_idx), dtype=object)
        scored = _distribution_pair_rows(
            view,
            train_repr,
            valid_repr,
            train_species,
            heldout_species,
            y[train_idx],
            sample_limit=sample_limit,
        )
        top = [row for row in scored if int(row["rank"]) == 1][0]
        nearest = str(top["train_species"])
        heldout_mean = float(np.mean(y[valid_idx]))
        nearest_mean = _target_mean_for_species(nearest, species, y)
        heldout_std = float(np.std(y[valid_idx]))
        nearest_std = _target_std_for_species(nearest, species, y)
        rows.append(
            {
                "view": view.name,
                "heldout_species": heldout,
                "heldout_n": int(len(valid_idx)),
                "nearest_train_species": nearest,
                "energy_distance": top["energy_distance"],
                "centroid_cosine_distance": top["centroid_cosine_distance"],
                "heldout_target_mean": heldout_mean,
                "nearest_target_mean": nearest_mean,
                "target_mean_gap": nearest_mean - heldout_mean,
                "abs_target_mean_gap": abs(nearest_mean - heldout_mean),
                "heldout_target_std": heldout_std,
                "nearest_target_std": nearest_std,
                "target_std_gap": nearest_std - heldout_std,
            }
        )
    return rows


def _cluster_rows(
    view: View,
    train_repr: np.ndarray,
    test_repr: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    combined = np.vstack([train_repr, test_repr])
    domains = np.asarray(["train"] * len(train_repr) + ["test"] * len(test_repr), dtype=object)
    species = np.concatenate([train_species, test_species])
    for k in (6, 8, 13, 18):
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = clusterer.fit_predict(combined)
        for cluster_id in range(k):
            idx = np.where(labels == cluster_id)[0]
            if len(idx) == 0:
                continue
            domain_counts = Counter(domains[idx].tolist())
            train_items = [species[i] for i in idx if domains[i] == "train"]
            test_items = [species[i] for i in idx if domains[i] == "test"]
            rows.append(
                {
                    "view": view.name,
                    "k": k,
                    "cluster": cluster_id,
                    "n": int(len(idx)),
                    "train_n": int(domain_counts.get("train", 0)),
                    "test_n": int(domain_counts.get("test", 0)),
                    "test_ratio": float(domain_counts.get("test", 0) / len(idx)),
                    "top_train_species": _top_counts(train_items, 5),
                    "top_test_species": _top_counts(test_items, 5),
                }
            )
    return rows


def _embedding_rows(
    view: View,
    train_repr: np.ndarray,
    test_repr: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for domain, X, species in (("train", train_repr, train_species), ("test", test_repr, test_species)):
        for label, centroid in _centroids(X, species).items():
            rows.append(
                {
                    "view": view.name,
                    "domain": domain,
                    "species": label,
                    "n": int(np.sum(species == label)),
                    "pc1": float(centroid[0]) if len(centroid) > 0 else 0.0,
                    "pc2": float(centroid[1]) if len(centroid) > 1 else 0.0,
                    "pc3": float(centroid[2]) if len(centroid) > 2 else 0.0,
                }
            )
    return rows


def _consensus_rows(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        if int(row["rank"]) <= 3:
            grouped[str(row["test_species"])].append(row)
    rows = []
    for test_label, items in grouped.items():
        votes = Counter(str(row["train_species"]) for row in items)
        best = votes.most_common(5)
        rows.append(
            {
                "test_species": test_label,
                "votes_top_train_species": "; ".join(f"{label}={count}" for label, count in best),
                "n_views": len(set(str(row["view"]) for row in items)),
                "note": "votes count top-3 distribution matches across all views",
            }
        )
    return sorted(rows, key=lambda row: _sort_label(str(row["test_species"])))


def _centroids(X: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    return {label: np.mean(X[labels == label], axis=0) for label in sorted(set(labels.tolist()), key=_sort_label)}


def _target_mean_for_species(label: str, train_species: np.ndarray, y_train: np.ndarray) -> float:
    idx = train_species == label
    return float(np.mean(y_train[idx])) if np.any(idx) else float("nan")


def _target_std_for_species(label: str, train_species: np.ndarray, y_train: np.ndarray) -> float:
    idx = train_species == label
    return float(np.std(y_train[idx])) if np.any(idx) else float("nan")


def _mean_pairwise_distance(X: np.ndarray) -> float:
    if len(X) <= 1:
        return 0.0
    d = pairwise_distances(X, X, metric="euclidean")
    mask = ~np.eye(len(X), dtype=bool)
    return float(np.mean(d[mask]))


def _limited(X: np.ndarray, limit: int) -> np.ndarray:
    if len(X) <= limit:
        return X
    # Deterministic evenly spaced subset keeps the script reproducible.
    idx = np.linspace(0, len(X) - 1, limit).round().astype(int)
    return X[idx]


def _top_counts(items: list[str], n: int) -> str:
    return "; ".join(f"{label}={count}" for label, count in Counter(items).most_common(n))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sort_label(label: str) -> tuple[int, str]:
    prefix = label.split(":", 1)[0]
    try:
        return (int(prefix), label)
    except ValueError:
        return (10_000, label)


if __name__ == "__main__":
    main()
