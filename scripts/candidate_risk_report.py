#!/usr/bin/env python3
"""Submission risk report against the protected Public-best anchor."""

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
    candidate_path = config.submissions_dir / f"{args.experiment}.csv"
    anchor_path = config.submissions_dir / f"{args.anchor}.csv"
    if not candidate_path.exists():
        raise SystemExit(f"missing candidate submission: {candidate_path}")
    if not anchor_path.exists():
        raise SystemExit(f"missing anchor submission: {anchor_path}")

    candidate = _read_submission(candidate_path)
    anchor = _read_submission(anchor_path)
    ids = sorted(set(candidate) & set(anchor), key=_sort_key)
    if len(ids) != len(candidate) or len(ids) != len(anchor):
        raise SystemExit("candidate/anchor ids do not match")

    pred = [candidate[i] for i in ids]
    base = [anchor[i] for i in ids]
    diff = [p - b for p, b in zip(pred, base)]

    print(f"candidate: {args.experiment}")
    print(f"anchor   : {args.anchor}")
    print(f"rows     : {len(ids)}")
    print(
        "pred     : "
        f"min={min(pred):.6f} max={max(pred):.6f} mean={_mean(pred):.6f} "
        f"neg={sum(1 for x in pred if x < 0)}"
    )
    print(
        "diff     : "
        f"rmse={_rmse(diff):.6f} mean={_mean(diff):.6f} "
        f"min={min(diff):.6f} max={max(diff):.6f}"
    )
    print()
    print("by test species number:")
    print(f"{'species':>8} {'n':>4} {'base_mean':>10} {'pred_mean':>10} {'diff_mean':>10} {'diff_rmse':>10} {'neg':>4}")
    meta = _read_test_meta(config.test_path, config.id_col)
    groups: dict[str, list[str]] = {}
    for sample_id in ids:
        groups.setdefault(meta.get(sample_id, ""), []).append(sample_id)
    for group, group_ids in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
        group_base = [anchor[i] for i in group_ids]
        group_pred = [candidate[i] for i in group_ids]
        group_diff = [p - b for p, b in zip(group_pred, group_base)]
        print(
            f"{group:>8} {len(group_ids):>4} "
            f"{_mean(group_base):>10.4f} {_mean(group_pred):>10.4f} "
            f"{_mean(group_diff):>10.4f} {_rmse(group_diff):>10.4f} "
            f"{sum(1 for x in group_pred if x < 0):>4}"
        )


def _read_submission(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            out[str(row[0])] = float(row[1])
    return out


def _read_test_meta(path: Path, id_col: str) -> dict[str, str]:
    with path.open(encoding="cp932", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    species_col = "species number" if "species number" in rows[0] else ""
    return {str(row[id_col]): str(row.get(species_col, "")) for row in rows}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(x * x for x in values) / len(values))


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    main()
