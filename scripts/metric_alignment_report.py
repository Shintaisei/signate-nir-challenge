#!/usr/bin/env python3
"""Compare local and anchor-drift metrics against measured Public scores."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "outputs" / "public_diagnostic_table.csv"
DEFAULT_OUT = ROOT / "outputs" / "metric_alignment_report.md"


METRICS: list[tuple[str, str, str]] = [
    ("group_species", "asc", "traditional"),
    ("moisture_quantile", "asc", "traditional"),
    ("random", "asc", "traditional"),
    ("worst_species_rmse", "asc", "traditional"),
    ("raw_diff_rmse", "asc", "anchor_drift"),
    ("max_abs_species_mean_shift", "asc", "anchor_drift"),
    ("mean_std_diff_rmse", "asc", "adjusted_drift"),
    ("species_mean_diff_rmse", "asc", "adjusted_drift"),
    ("species_mean_std_diff_rmse", "asc", "adjusted_drift"),
    ("corr_to_anchor", "desc", "anchor_drift"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = _read_rows(args.table)
    measured = [row for row in rows if _is_num(row.get("public"))]
    if len(measured) < 4:
        raise SystemExit("not enough measured rows")

    lines: list[str] = []
    lines.append("# Metric Alignment Report")
    lines.append("")
    lines.append(f"Measured Public rows: {len(measured)}")
    lines.append("")
    lines.append("## Public Rows")
    lines.append("")
    lines.append("| public_rank | experiment | Public | group_species | raw_diff | max_species_shift | species_mean_std_diff | corr_to_anchor |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, row in enumerate(sorted(measured, key=lambda r: float(r["public"])), start=1):
        lines.append(
            "| "
            f"{rank} | {row['experiment']} | {_fmt(row, 'public')} | {_fmt(row, 'group_species')} | "
            f"{_fmt(row, 'raw_diff_rmse')} | {_fmt(row, 'max_abs_species_mean_shift')} | "
            f"{_fmt(row, 'species_mean_std_diff_rmse')} | {_fmt(row, 'corr_to_anchor')} |"
        )

    lines.append("")
    lines.append("## Metric Correlations")
    lines.append("")
    lines.append("| metric | family | n | pearson_vs_public | spearman_vs_public | direction |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    summaries = []
    for metric, direction, family in METRICS:
        values = [(float(row[metric]), float(row["public"])) for row in measured if _is_num(row.get(metric))]
        if len(values) < 4:
            continue
        x = np.asarray([v[0] for v in values], dtype=float)
        y = np.asarray([v[1] for v in values], dtype=float)
        if direction == "desc":
            x_for_rank = -x
            direction_label = "higher is better"
        else:
            x_for_rank = x
            direction_label = "lower is better"
        pearson = _corr(x if direction == "asc" else -x, y)
        spearman = _corr(_rank(x_for_rank), _rank(y))
        summaries.append((metric, family, len(values), pearson, spearman, direction_label))
    for metric, family, n, pearson, spearman, direction_label in sorted(summaries, key=lambda s: abs(s[4]), reverse=True):
        lines.append(f"| {metric} | {family} | {n} | {pearson:.3f} | {spearman:.3f} | {direction_label} |")

    lines.append("")
    lines.append("## Rank Check")
    lines.append("")
    lines.append("Public上位をどれだけ上位に置けるかを見る。rank_errorはPublic rankとの差の絶対値平均。小さいほど整合的。")
    lines.append("")
    lines.append("| metric | n | mean_rank_error | worst_rank_error | top3_by_metric |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    public_rank = {row["experiment"]: rank for rank, row in enumerate(sorted(measured, key=lambda r: float(r["public"])), start=1)}
    for metric, direction, _family in METRICS:
        available = [row for row in measured if _is_num(row.get(metric))]
        if len(available) < 4:
            continue
        reverse = direction == "desc"
        ranked = sorted(available, key=lambda r: float(r[metric]), reverse=reverse)
        errors = [abs(rank - public_rank[row["experiment"]]) for rank, row in enumerate(ranked, start=1)]
        top3 = ", ".join(row["experiment"] for row in ranked[:3])
        lines.append(f"| {metric} | {len(available)} | {np.mean(errors):.2f} | {max(errors):.0f} | {top3} |")

    lines.append("")
    lines.append("## Practical Interpretation")
    lines.append("")
    lines.append("- `group_species` and `moisture_quantile` are not reliable rankers in the measured set; multiview was locally best but Public worst.")
    lines.append("- `raw_diff_rmse` and `max_abs_species_mean_shift` are useful rejection gates: they catch multiview and nn-bias failures.")
    lines.append("- Corrected drift is useful for diagnosis. If `species_mean_std_diff_rmse` remains high, the candidate shape is different from the anchor even after species-level calibration.")
    lines.append("- Close-to-anchor alone is not enough for improvement. The r8 5% blend had tiny drift but worsened Public, so drift metrics should reject risk, not guarantee upside.")
    lines.append("- Current best use: gate by anchor drift first, then compare only candidates that pass against domain knowledge and measured family history.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _is_num(value: object) -> bool:
    try:
        if value is None or value == "":
            return False
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(row: dict[str, str], key: str) -> str:
    if not _is_num(row.get(key)):
        return ""
    return f"{float(row[key]):.4f}"


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


if __name__ == "__main__":
    main()

