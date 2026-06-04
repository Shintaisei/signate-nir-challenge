#!/usr/bin/env python3
"""Estimate Public score from measured submissions and anchor-drift features."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402
from pipeline.public_score import PUBLIC_BEST_EXPERIMENT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("--anchor", default=PUBLIC_BEST_EXPERIMENT)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    config = load_config()
    measured = _measured_rows(config, args.anchor)
    if len(measured) < 4:
        raise SystemExit("not enough measured Public rows")

    names = [row[0] for row in measured]
    x = np.asarray([row[1] for row in measured], dtype=float)
    y = np.asarray([row[2] for row in measured], dtype=float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    if args.audit:
        print(f"measured rows: {len(measured)}")
        errors = []
        for i, name in enumerate(names):
            keep = [j for j in range(len(names)) if j != i]
            model.fit(x[keep], y[keep])
            pred = float(model.predict(x[[i]])[0])
            err = pred - y[i]
            errors.append(err)
            print(f"{name:45s} public={y[i]:9.4f} loo_est={pred:9.4f} err={err:8.4f}")
        print(f"LOO MAE={np.mean(np.abs(errors)):.4f} RMSE={np.sqrt(np.mean(np.asarray(errors) ** 2)):.4f}")
        print()

    if args.experiment:
        model.fit(x, y)
        features = np.asarray([_drift_features(config, args.experiment, args.anchor)], dtype=float)
        pred = float(model.predict(features)[0])
        print(f"experiment: {args.experiment}")
        print(f"anchor    : {args.anchor}")
        print(f"features  : diff_rmse={features[0,0]:.6f}, abs_bias={features[0,1]:.6f}, max_species_shift={features[0,2]:.6f}")
        print(f"calibrated_public_estimate: {pred:.6f}")


def _measured_rows(config, anchor: str) -> list[tuple[str, list[float], float]]:
    rows: list[tuple[str, list[float], float]] = []
    path = config.outputs_dir / "public_compare.csv"
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            exp = row["experiment_name"]
            try:
                public = float(row["public_score"])
                features = _drift_features(config, exp, anchor)
            except (FileNotFoundError, ValueError):
                continue
            rows.append((exp, features, public))
    return rows


def _drift_features(config, experiment: str, anchor: str) -> list[float]:
    candidate = _read_submission(config.submissions_dir / f"{experiment}.csv")
    base = _read_submission(config.submissions_dir / f"{anchor}.csv")
    ids = sorted(candidate, key=_sort_key)
    if set(ids) != set(base):
        raise ValueError("submission ids do not match")
    diff = np.asarray([candidate[i] - base[i] for i in ids], dtype=float)
    max_species_shift = _max_species_shift(config.test_path, config.id_col, ids, diff)
    return [
        float(np.sqrt(np.mean(diff * diff))),
        abs(float(diff.mean())),
        max_species_shift,
    ]


def _read_submission(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        return {str(row[0]): float(row[1]) for row in csv.reader(f) if row}


def _max_species_shift(test_path: Path, id_col: str, ids: list[str], diff: np.ndarray) -> float:
    with test_path.open(encoding="cp932", newline="") as f:
        rows = list(csv.DictReader(f))
    species = {str(row[id_col]): str(row.get("species number", "")) for row in rows}
    grouped: dict[str, list[float]] = {}
    for sample_id, value in zip(ids, diff):
        grouped.setdefault(species.get(sample_id, ""), []).append(float(value))
    return max(abs(sum(values) / len(values)) for values in grouped.values())


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    main()
