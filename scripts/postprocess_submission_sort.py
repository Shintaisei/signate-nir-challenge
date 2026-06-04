#!/usr/bin/env python3
"""Apply species-wise sample-number decreasing sort to existing submissions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402
from pipeline.data import load_raw_data  # noqa: E402
from pipeline.submission import write_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", help="Existing submission name or path")
    parser.add_argument("--suffix", default="_sort")
    args = parser.parse_args()

    cfg = load_config()
    raw = load_raw_data(cfg)
    path = Path(args.submission)
    if not path.exists():
        path = cfg.submissions_dir / args.submission
    if not path.exists() and not str(path).endswith(".csv"):
        path = cfg.submissions_dir / f"{args.submission}.csv"
    if not path.exists():
        raise FileNotFoundError(args.submission)

    pred_by_id = _read_submission(path)
    sample = np.asarray([float(row[cfg.id_col]) for row in raw.test], dtype=float)
    species = np.asarray([str(row.get("species number", "")) for row in raw.test], dtype=object)
    ids = [int(row[cfg.id_col]) for row in raw.test]
    pred = np.asarray([pred_by_id[i] for i in ids], dtype=float)
    sorted_pred = _sort_decreasing(pred, sample, species)

    stem = path.stem
    out_path = cfg.submissions_dir / f"{stem}{args.suffix}.csv"
    write_submission(list(zip(ids, sorted_pred.tolist())), out_path, id_col=cfg.id_col)
    diff = float(np.sqrt(np.mean((sorted_pred - pred) ** 2)))
    print(f"saved: {out_path}")
    print(f"diff_rmse_from_source: {diff:.6f}")
    print(f"min={float(np.min(sorted_pred)):.6f} max={float(np.max(sorted_pred)):.6f} mean={float(np.mean(sorted_pred)):.6f}")


def _read_submission(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            out[int(float(row[0]))] = float(row[1])
    return out


def _sort_decreasing(pred: np.ndarray, sample: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = pred.copy()
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        order = idx[np.argsort(sample[idx])]
        out[order] = np.sort(pred[idx])[::-1]
    return out


if __name__ == "__main__":
    main()
