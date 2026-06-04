#!/usr/bin/env python3
"""Conservative Public-risk score from local diagnostics.

This is not a leaderboard predictor. It is a pre-submission guard that catches
domain-shift failures seen in measured Public history.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

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
        raise SystemExit("candidate and anchor submission ids do not match")

    pred = [candidate[i] for i in ids]
    base = [anchor[i] for i in ids]
    diff = [p - b for p, b in zip(pred, base)]
    species_shifts = _species_mean_shifts(config.test_path, config.id_col, ids, diff)

    diff_rmse = _rmse(diff)
    mean_shift = _mean(diff)
    max_species_shift = max(abs(v) for v in species_shifts.values()) if species_shifts else abs(mean_shift)
    pred_mean_shift = _mean(pred) - _mean(base)
    neg_count = sum(1 for value in pred if value < 0)

    guard_score = (
        16.151771
        + 0.12 * diff_rmse
        + 0.18 * abs(mean_shift)
        + 0.18 * max_species_shift
        + 0.08 * abs(pred_mean_shift)
        + 0.20 * neg_count
    )

    reasons: list[str] = []
    if diff_rmse > 10:
        reasons.append("reject: anchor diff RMSE > 10")
    if abs(mean_shift) > 5:
        reasons.append("reject: global mean shift > 5")
    if max_species_shift > 12:
        reasons.append("reject: test-species mean shift > 12")
    if neg_count:
        reasons.append("warn: negative predictions")
    if not reasons:
        reasons.append("pass distribution guard")

    print(f"experiment: {args.experiment}")
    print(f"anchor    : {args.anchor}")
    print(f"guard_score_rough: {guard_score:.6f}")
    print(f"diff_rmse : {diff_rmse:.6f}")
    print(f"mean_shift: {mean_shift:.6f}")
    print(f"max_species_mean_shift: {max_species_shift:.6f}")
    print(f"pred_mean_shift: {pred_mean_shift:.6f}")
    print(f"negative_predictions: {neg_count}")
    print("decision  : " + "; ".join(reasons))


def _read_submission(path: Path) -> dict[str, float]:
    if not path.exists():
        raise SystemExit(f"missing submission: {path}")
    out: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if row:
                out[str(row[0])] = float(row[1])
    return out


def _species_mean_shifts(test_path: Path, id_col: str, ids: list[str], diff: list[float]) -> dict[str, float]:
    with test_path.open(encoding="cp932", newline="") as f:
        rows = list(csv.DictReader(f))
    species_by_id = {str(row[id_col]): str(row.get("species number", "")) for row in rows}
    grouped: dict[str, list[float]] = {}
    for sample_id, value in zip(ids, diff):
        grouped.setdefault(species_by_id.get(sample_id, ""), []).append(value)
    return {group: _mean(values) for group, values in grouped.items() if values}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    main()
