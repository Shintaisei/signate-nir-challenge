#!/usr/bin/env python3
"""leaderboard / public_compare 内の旧パスを現リポジトリルートへ置換する。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_MARKERS = (
    "金融AI/competitions/近赤外研究会",
    "近赤外研究会",
)


def _rewrite(value: str) -> str:
    out = value
    for marker in OLD_MARKERS:
        if marker in out:
            idx = out.find(marker)
            suffix = out[idx + len(marker) :].lstrip("/")
            out = str(ROOT / suffix) if suffix else str(ROOT)
    return out


def migrate_csv(path: Path, path_columns: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        return 0
    changed = 0
    for row in rows:
        for col in path_columns:
            if col not in row or not row[col]:
                continue
            new_val = _rewrite(row[col])
            if new_val != row[col]:
                row[col] = new_val
                changed += 1
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main() -> None:
    outputs = ROOT / "outputs"
    n1 = migrate_csv(outputs / "leaderboard.csv", ("submission_path", "log_path"))
    n2 = migrate_csv(outputs / "public_compare.csv", ("submission_path",))
    print(f"migrated path cells: leaderboard={n1}, public_compare={n2}")


if __name__ == "__main__":
    main()
