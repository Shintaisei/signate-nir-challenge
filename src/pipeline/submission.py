"""提出 CSV の生成。"""

from __future__ import annotations

import csv
from pathlib import Path

from pipeline.types import Rows, Vector


def build_submission_frame(
    test: Rows,
    predictions: Vector,
    *,
    id_col: str,
) -> list[tuple[int, float]]:
    return [(int(row[id_col]), float(pred)) for row, pred in zip(test, predictions)]


def write_submission(
    frame: list[tuple[int, float]],
    path: Path,
    *,
    id_col: str = "sample number",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(frame)
    return path
