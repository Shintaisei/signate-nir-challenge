#!/usr/bin/env python3
"""Compare a candidate to the protected anchor after simple calibrations.

The goal is diagnostic: separate global level/scale drift from within-species
shape drift before using anchor-difference metrics.
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
from pipeline.public_score import PUBLIC_BEST_EXPERIMENT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--anchor", default=PUBLIC_BEST_EXPERIMENT)
    args = parser.parse_args()

    config = load_config()
    candidate = _read_submission(config.submissions_dir / f"{args.experiment}.csv")
    anchor = _read_submission(config.submissions_dir / f"{args.anchor}.csv")
    ids = sorted(candidate, key=_sort_key)
    if set(ids) != set(anchor):
        raise SystemExit("candidate and anchor ids do not match")

    pred = np.asarray([candidate[i] for i in ids], dtype=float)
    base = np.asarray([anchor[i] for i in ids], dtype=float)
    species = _test_species(config.test_path, config.id_col, ids)

    variants = {
        "raw": pred,
        "mean_aligned": pred - (pred.mean() - base.mean()),
        "mean_std_aligned": _mean_std_align(pred, base),
        "species_mean_aligned": _species_mean_align(pred, base, species),
        "species_mean_std_aligned": _species_mean_std_align(pred, base, species),
    }

    print(f"candidate: {args.experiment}")
    print(f"anchor   : {args.anchor}")
    print(f"{'variant':24s} {'diff_rmse':>10} {'diff_mae':>10} {'bias':>10} {'max_species_bias':>16} {'corr':>8}")
    for name, values in variants.items():
        diff = values - base
        max_species_bias = _max_species_bias(diff, species)
        corr = float(np.corrcoef(values, base)[0, 1]) if len(values) > 1 else float("nan")
        print(
            f"{name:24s} "
            f"{_rmse(diff):10.4f} "
            f"{float(np.mean(np.abs(diff))):10.4f} "
            f"{float(diff.mean()):10.4f} "
            f"{max_species_bias:16.4f} "
            f"{corr:8.4f}"
        )

    print()
    print("by species after species_mean_aligned:")
    adjusted = variants["species_mean_aligned"]
    diff = adjusted - base
    print(f"{'species':>8} {'n':>4} {'rmse':>10} {'mae':>10} {'bias':>10} {'corr':>8}")
    for group in sorted(set(species), key=_sort_key):
        idx = [i for i, value in enumerate(species) if value == group]
        d = diff[idx]
        v = adjusted[idx]
        b = base[idx]
        corr = float(np.corrcoef(v, b)[0, 1]) if len(idx) > 1 else float("nan")
        print(
            f"{group:>8} {len(idx):>4} "
            f"{_rmse(d):10.4f} "
            f"{float(np.mean(np.abs(d))):10.4f} "
            f"{float(d.mean()):10.4f} "
            f"{corr:8.4f}"
        )


def _read_submission(path: Path) -> dict[str, float]:
    if not path.exists():
        raise SystemExit(f"missing submission: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return {str(row[0]): float(row[1]) for row in csv.reader(f) if row}


def _test_species(test_path: Path, id_col: str, ids: list[str]) -> list[str]:
    with test_path.open(encoding="cp932", newline="") as f:
        rows = list(csv.DictReader(f))
    species = {str(row[id_col]): str(row.get("species number", "")) for row in rows}
    return [species.get(i, "") for i in ids]


def _mean_std_align(pred: np.ndarray, base: np.ndarray) -> np.ndarray:
    pred_std = max(float(pred.std()), 1e-12)
    return (pred - pred.mean()) / pred_std * float(base.std()) + float(base.mean())


def _species_mean_align(pred: np.ndarray, base: np.ndarray, species: list[str]) -> np.ndarray:
    out = pred.copy()
    for group in set(species):
        idx = np.asarray([i for i, value in enumerate(species) if value == group], dtype=int)
        out[idx] = pred[idx] - (pred[idx].mean() - base[idx].mean())
    return out


def _species_mean_std_align(pred: np.ndarray, base: np.ndarray, species: list[str]) -> np.ndarray:
    out = pred.copy()
    for group in set(species):
        idx = np.asarray([i for i, value in enumerate(species) if value == group], dtype=int)
        p = pred[idx]
        b = base[idx]
        out[idx] = (p - p.mean()) / max(float(p.std()), 1e-12) * float(b.std()) + float(b.mean())
    return out


def _max_species_bias(diff: np.ndarray, species: list[str]) -> float:
    return max(abs(float(diff[[i for i, s in enumerate(species) if s == group]].mean())) for group in set(species))


def _rmse(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values * values))))


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    main()
