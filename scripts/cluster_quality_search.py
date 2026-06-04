#!/usr/bin/env python3
"""Search spectral views that give clean train/test clustering structure."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402
from pipeline.data import load_raw_data  # noqa: E402


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


@dataclass(frozen=True)
class ViewSpec:
    name: str
    prep: str
    rep: str
    n_components: int


@dataclass(frozen=True)
class ClusterSpec:
    name: str
    kind: str
    k: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "cluster_quality")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    cfg = load_config()
    raw = load_raw_data(cfg)
    feature_cols = [c for c in raw.train[0] if c not in set(cfg.meta_cols) | {cfg.target_col}]
    X_train = np.asarray([[float(row[c]) for c in feature_cols] for row in raw.train], dtype=float)
    X_test = np.asarray([[float(row[c]) for c in feature_cols] for row in raw.test], dtype=float)
    train_species = np.asarray([_species_label(row) for row in raw.train], dtype=object)
    test_species = np.asarray([_species_label(row) for row in raw.test], dtype=object)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    species_rows: list[dict[str, object]] = []

    for view in _views(args.preset):
        print(f"view {view.name}", flush=True)
        Xtr, Xte, explained = _fit_view(X_train, X_test, view)
        X = np.vstack([Xtr, Xte])
        domain = np.asarray(["train"] * len(Xtr) + ["test"] * len(Xte), dtype=object)
        species = np.concatenate([train_species, test_species])
        for spec in _cluster_specs(args.preset):
            labels = _cluster(X, spec)
            summary = _summary_row(view, spec, X, labels, domain, species, explained)
            summary_rows.append(summary)
            detail_rows.extend(_cluster_detail_rows(view, spec, labels, domain, species))
            species_rows.extend(_test_species_rows(view, spec, labels, train_species, test_species))
            print(
                f"  {spec.name}: score={float(summary['clean_score']):.4f} "
                f"purity={float(summary['test_cluster_purity']):.4f} "
                f"concentration={float(summary['test_species_concentration']):.4f}",
                flush=True,
            )

    summary_rows.sort(key=lambda row: float(row["clean_score"]), reverse=True)
    _write_csv(args.out_dir / "cluster_quality_summary.csv", summary_rows)
    _write_csv(args.out_dir / "cluster_quality_details.csv", detail_rows)
    _write_csv(args.out_dir / "test_species_cluster_map.csv", species_rows)
    (args.out_dir / "top.json").write_text(
        json.dumps(summary_rows[:20], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {args.out_dir}")


def _species_label(row: dict[str, str]) -> str:
    number = str(row.get("species number", "")).strip()
    return f"{number}:{_SPECIES_NAMES.get(number, f'species_{number}')}"


def _views(preset: str) -> list[ViewSpec]:
    components = (5, 10, 20) if preset == "quick" else (3, 5, 10, 20, 40)
    base = []
    for prep in ("raw", "smooth5", "snv", "center", "diff1", "diff2"):
        for n_components in components:
            base.append(ViewSpec(f"{prep}_pca{n_components}", prep, "pca", n_components))
    for prep in ("smooth5", "snv", "diff1", "diff2"):
        for n_components in components:
            base.append(ViewSpec(f"{prep}_segment_pca{n_components}", prep, "segment", n_components))
    return base


def _cluster_specs(preset: str) -> list[ClusterSpec]:
    ks = (6, 8, 10, 13, 18) if preset == "quick" else (4, 5, 6, 8, 10, 13, 18, 24, 32)
    specs = []
    for k in ks:
        specs.append(ClusterSpec(f"kmeans_k{k}", "kmeans", k))
        specs.append(ClusterSpec(f"ward_k{k}", "ward", k))
        if preset == "full":
            specs.append(ClusterSpec(f"gmm_k{k}", "gmm", k))
    return specs


def _fit_view(X_train: np.ndarray, X_test: np.ndarray, view: ViewSpec) -> tuple[np.ndarray, np.ndarray, float]:
    Ztr = _prep(X_train, view.prep)
    Zte = _prep(X_test, view.prep)
    if view.rep == "segment":
        Ztr = _segment_features(Ztr)
        Zte = _segment_features(Zte)
    scaler = StandardScaler()
    Ztr = scaler.fit_transform(Ztr)
    Zte = scaler.transform(Zte)
    n_components = min(view.n_components, Ztr.shape[1], Ztr.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    Xtr = pca.fit_transform(Ztr)
    Xte = pca.transform(Zte)
    return Xtr, Xte, float(np.sum(pca.explained_variance_ratio_))


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
            axis_cache[width] = axis - axis.mean()
        axis = axis_cache[width]
        denom = max(float(np.sum(axis * axis)), 1e-12)
        slope = (block - block.mean(axis=1, keepdims=True)) @ axis / denom
        curve_axis = axis * axis - np.mean(axis * axis)
        curvature = (block - block.mean(axis=1, keepdims=True)) @ curve_axis / max(
            float(np.sum(curve_axis * curve_axis)),
            1e-12,
        )
        blocks.append(
            np.vstack(
                [
                    np.mean(block, axis=1),
                    np.std(block, axis=1),
                    np.max(block, axis=1) - np.min(block, axis=1),
                    slope,
                    curvature,
                    block[:, -1] - block[:, 0],
                ]
            ).T
        )
    return np.hstack(blocks)


def _cluster(X: np.ndarray, spec: ClusterSpec) -> np.ndarray:
    if spec.kind == "kmeans":
        return KMeans(n_clusters=spec.k, random_state=42, n_init=30).fit_predict(X)
    if spec.kind == "ward":
        return AgglomerativeClustering(n_clusters=spec.k, linkage="ward").fit_predict(X)
    if spec.kind == "gmm":
        return GaussianMixture(n_components=spec.k, covariance_type="diag", random_state=42).fit_predict(X)
    raise ValueError(spec.kind)


def _summary_row(
    view: ViewSpec,
    spec: ClusterSpec,
    X: np.ndarray,
    labels: np.ndarray,
    domain: np.ndarray,
    species: np.ndarray,
    explained: float,
) -> dict[str, object]:
    test_mask = domain == "test"
    train_mask = domain == "train"
    test_purity = _cluster_purity(labels[test_mask], species[test_mask])
    train_purity = _cluster_purity(labels[train_mask], species[train_mask])
    test_concentration = _species_concentration(labels[test_mask], species[test_mask])
    mixed_coverage = _mixed_test_coverage(labels, domain)
    sil = _safe_silhouette(X, labels)
    # This favors test species that form clean clusters while still sharing
    # enough clusters with train samples to allow a routed predictor later.
    clean_score = (
        0.35 * test_purity
        + 0.35 * test_concentration
        + 0.20 * mixed_coverage
        + 0.10 * max(0.0, min(1.0, (sil + 0.1) / 0.6))
    )
    return {
        "view": view.name,
        "clusterer": spec.name,
        "k": spec.k,
        "explained": explained,
        "test_cluster_purity": test_purity,
        "test_species_concentration": test_concentration,
        "train_cluster_purity": train_purity,
        "mixed_test_coverage": mixed_coverage,
        "silhouette": sil,
        "clean_score": clean_score,
    }


def _cluster_purity(labels: np.ndarray, species: np.ndarray) -> float:
    total = len(labels)
    score = 0
    for cluster_id in sorted(set(labels.tolist())):
        idx = labels == cluster_id
        if np.any(idx):
            score += Counter(species[idx].tolist()).most_common(1)[0][1]
    return float(score / total)


def _species_concentration(labels: np.ndarray, species: np.ndarray) -> float:
    total = len(labels)
    score = 0
    for label in sorted(set(species.tolist())):
        idx = species == label
        score += Counter(labels[idx].tolist()).most_common(1)[0][1]
    return float(score / total)


def _mixed_test_coverage(labels: np.ndarray, domain: np.ndarray) -> float:
    test_total = int(np.sum(domain == "test"))
    covered = 0
    for cluster_id in sorted(set(labels.tolist())):
        idx = labels == cluster_id
        if np.sum(idx & (domain == "test")) > 0 and np.sum(idx & (domain == "train")) >= 10:
            covered += int(np.sum(idx & (domain == "test")))
    return float(covered / max(test_total, 1))


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    try:
        if len(set(labels.tolist())) < 2:
            return 0.0
        sample = min(1000, len(labels))
        return float(silhouette_score(X, labels, sample_size=sample, random_state=42))
    except Exception:
        return 0.0


def _cluster_detail_rows(
    view: ViewSpec,
    spec: ClusterSpec,
    labels: np.ndarray,
    domain: np.ndarray,
    species: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for cluster_id in sorted(set(labels.tolist())):
        idx = labels == cluster_id
        train_items = species[idx & (domain == "train")].tolist()
        test_items = species[idx & (domain == "test")].tolist()
        rows.append(
            {
                "view": view.name,
                "clusterer": spec.name,
                "cluster": int(cluster_id),
                "n": int(np.sum(idx)),
                "train_n": len(train_items),
                "test_n": len(test_items),
                "test_ratio": float(len(test_items) / max(int(np.sum(idx)), 1)),
                "top_train_species": _top_counts(train_items, 6),
                "top_test_species": _top_counts(test_items, 6),
            }
        )
    return rows


def _test_species_rows(
    view: ViewSpec,
    spec: ClusterSpec,
    labels: np.ndarray,
    train_species: np.ndarray,
    test_species: np.ndarray,
) -> list[dict[str, object]]:
    train_labels = labels[: len(train_species)]
    test_labels = labels[len(train_species) :]
    rows = []
    for label in sorted(set(test_species.tolist()), key=_sort_label):
        idx = test_species == label
        counts = Counter(test_labels[idx].tolist())
        top_cluster, top_n = counts.most_common(1)[0]
        train_in_cluster = train_species[train_labels == top_cluster].tolist()
        test_in_cluster = test_species[test_labels == top_cluster].tolist()
        rows.append(
            {
                "view": view.name,
                "clusterer": spec.name,
                "test_species": label,
                "test_n": int(np.sum(idx)),
                "top_cluster": int(top_cluster),
                "top_cluster_share": float(top_n / np.sum(idx)),
                "top_cluster_train_n": len(train_in_cluster),
                "top_cluster_test_n": len(test_in_cluster),
                "top_train_species_in_cluster": _top_counts(train_in_cluster, 6),
                "top_test_species_in_cluster": _top_counts(test_in_cluster, 6),
            }
        )
    return rows


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
