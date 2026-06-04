#!/usr/bin/env python3
"""Domain-weighted OOF evaluation.

Weights held-out train species by how close their spectrum distribution is to
the unlabeled test species distribution after a chosen lightweight transform.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", help="Experiment with group_species OOF")
    parser.add_argument("--transform", default="smooth5", choices=["raw", "center", "snv", "smooth5", "diff1"])
    parser.add_argument("--all-public", action="store_true", help="Evaluate all experiments in public_compare.csv")
    args = parser.parse_args()

    config = load_config()
    train_rows = _read_rows(config.train_path, config.encoding)
    test_rows = _read_rows(config.test_path, config.encoding)
    feature_cols = [c for c in train_rows[0] if c not in set(config.meta_cols) | {config.target_col}]

    train_x = _transform(_matrix(train_rows, feature_cols), args.transform)
    test_x = _transform(_matrix(test_rows, feature_cols), args.transform)
    train_groups = [str(row["species number"]) for row in train_rows]
    test_groups = [str(row["species number"]) for row in test_rows]
    weights = _domain_weights(train_x, train_groups, test_x, test_groups)

    experiments = [args.experiment] if args.experiment else []
    if args.all_public:
        experiments = _public_experiments(config.outputs_dir / "public_compare.csv")
    if not experiments:
        raise SystemExit("provide experiment or --all-public")

    rows = []
    for exp in experiments:
        metrics = _weighted_oof_metrics(config.outputs_dir / "oof" / f"{exp}_group_species.csv", weights)
        if not metrics:
            continue
        metrics["experiment"] = exp
        rows.append(metrics)

    _print_rows(rows)


def _read_rows(path: Path, encoding: str) -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


def _matrix(rows: list[dict[str, str]], cols: list[str]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in cols] for row in rows], dtype=float)


def _transform(x: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return x
    if name == "center":
        return x - x.mean(axis=1, keepdims=True)
    if name == "snv":
        centered = x - x.mean(axis=1, keepdims=True)
        scale = x.std(axis=1, ddof=1, keepdims=True)
        return centered / np.maximum(scale, 1e-12)
    if name == "smooth5":
        return _moving_average(x, 5)
    if name == "diff1":
        return np.diff(x, axis=1)
    raise ValueError(name)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        start = max(0, j - radius)
        end = min(x.shape[1], j + radius + 1)
        out[:, j] = x[:, start:end].mean(axis=1)
    return out


def _domain_weights(
    train_x: np.ndarray,
    train_groups: list[str],
    test_x: np.ndarray,
    test_groups: list[str],
) -> dict[str, float]:
    combined = np.vstack([train_x, test_x])
    mean = combined.mean(axis=0)
    std = np.maximum(combined.std(axis=0), 1e-12)
    train_z = (train_x - mean) / std
    test_z = (test_x - mean) / std

    train_centroids = _centroids(train_z, train_groups)
    test_centroids = _centroids(test_z, test_groups)
    distances: dict[str, float] = {}
    all_dists: list[float] = []
    for group, center in train_centroids.items():
        d = min(float(np.linalg.norm(center - tc) / math.sqrt(len(center))) for tc in test_centroids.values())
        distances[group] = d
        all_dists.append(d)
    bandwidth = max(float(np.median(all_dists)), 1e-6)
    raw = {g: math.exp(-0.5 * (d / bandwidth) ** 2) for g, d in distances.items()}
    total = sum(raw.values()) or 1.0
    return {g: v / total for g, v in raw.items()}


def _centroids(x: np.ndarray, groups: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for group in sorted(set(groups)):
        idx = [i for i, current in enumerate(groups) if current == group]
        out[group] = x[idx].mean(axis=0)
    return out


def _weighted_oof_metrics(path: Path, weights: dict[str, float]) -> dict[str, float] | None:
    if not path.exists():
        return None
    by_group: dict[str, list[float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_group.setdefault(str(row["species number"]), []).append(float(row["error"]))
    weighted_mse = 0.0
    weighted_abs_bias = 0.0
    used_weight = 0.0
    worst_rmse = 0.0
    worst_bias = 0.0
    for group, errors in by_group.items():
        mse = sum(e * e for e in errors) / len(errors)
        bias = sum(errors) / len(errors)
        w = weights.get(group, 0.0)
        weighted_mse += w * mse
        weighted_abs_bias += w * abs(bias)
        used_weight += w
        worst_rmse = max(worst_rmse, math.sqrt(mse))
        worst_bias = max(worst_bias, abs(bias))
    if used_weight <= 0:
        return None
    return {
        "domain_rmse": math.sqrt(weighted_mse / used_weight),
        "domain_abs_bias": weighted_abs_bias / used_weight,
        "worst_species_rmse": worst_rmse,
        "worst_species_abs_bias": worst_bias,
    }


def _public_experiments(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return [row["experiment_name"] for row in csv.DictReader(f)]


def _print_rows(rows: list[dict[str, float | str]]) -> None:
    print(f"{'experiment':45s} {'domain':>9} {'bias':>9} {'worst':>9} {'worst_b':>9}")
    for row in sorted(rows, key=lambda r: float(r["domain_rmse"])):
        print(
            f"{str(row['experiment'])[:45]:45s} "
            f"{float(row['domain_rmse']):9.4f} "
            f"{float(row['domain_abs_bias']):9.4f} "
            f"{float(row['worst_species_rmse']):9.4f} "
            f"{float(row['worst_species_abs_bias']):9.4f}"
        )


if __name__ == "__main__":
    main()
