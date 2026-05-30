#!/usr/bin/env python3
"""OOF を樹種 × 含水率4分位で分解する。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eda.eda_common import setup_src_path, write_json  # noqa: E402
from pipeline.config import load_config  # noqa: E402


def main() -> None:
    setup_src_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="candidate_linear_1f")
    parser.add_argument("--cv", default="leave_one_species")
    args = parser.parse_args()

    config = load_config()
    oof_path = config.outputs_dir / "oof" / f"{args.experiment}_{args.cv}.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF not found: {oof_path}. Run: python run.py {args.experiment} --cv {args.cv}")

    rows = _read_oof(oof_path)
    edges = _moisture_edges([float(r["y_true"]) for r in rows], n_bins=4)

    matrix: list[dict[str, object]] = []
    species_list = sorted({str(r["樹種"]) for r in rows})
    for species in species_list:
        species_rows = [r for r in rows if str(r["樹種"]) == species]
        for b in range(4):
            lo, hi = edges[b], edges[b + 1]
            if b == 3:
                chunk = [r for r in species_rows if float(r["y_true"]) >= lo]
            else:
                chunk = [r for r in species_rows if lo <= float(r["y_true"]) < hi]
            if not chunk:
                continue
            matrix.append(
                {
                    "樹種": species,
                    "moisture_bin": b,
                    "y_range": [lo, hi],
                    "n": len(chunk),
                    **_metrics(chunk),
                }
            )

    worst = sorted(matrix, key=lambda r: float(r["rmse"]), reverse=True)[:15]
    report = {
        "experiment": args.experiment,
        "cv": args.cv,
        "oof_path": str(oof_path),
        "moisture_edges": edges,
        "matrix": matrix,
        "worst_cells": worst,
    }
    out = config.outputs_dir / "logs" / f"eda_error_decomposition_{args.experiment}.json"
    write_json(out, report)
    print(f"saved: {out}")
    for row in worst[:5]:
        print(f"  {row['樹種']} bin={row['moisture_bin']} rmse={row['rmse']:.2f} bias={row['bias']:.2f}")


def _read_oof(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _moisture_edges(values: list[float], n_bins: int) -> list[float]:
    sorted_y = sorted(values)
    return [
        sorted_y[0] - 1e-6,
        *[_quantile(sorted_y, i / n_bins) for i in range(1, n_bins)],
        sorted_y[-1] + 1e-6,
    ]


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (1 - (pos - lo)) + values[hi] * (pos - lo)


def _metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    errors = [float(r["y_true"]) - float(r["y_pred"]) for r in rows]
    sq = [e * e for e in errors]
    return {
        "rmse": math.sqrt(sum(sq) / len(sq)),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "bias": sum(errors) / len(errors),
    }


if __name__ == "__main__":
    main()
