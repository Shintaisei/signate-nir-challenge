#!/usr/bin/env python3
"""Public 実測とローカル CV 指標の整合性を分析する。"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eda.eda_common import setup_src_path, write_json  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.leaderboard import load_leaderboard  # noqa: E402


def main() -> None:
    setup_src_path()
    config = load_config()
    public_path = config.outputs_dir / "public_compare.csv"
    public_rows = list(csv.DictReader(public_path.open(encoding="utf-8")))

    # 実験ごとに最新 Public
    latest_public: dict[str, float] = {}
    for row in public_rows:
        name = row["experiment_name"]
        latest_public[name] = float(row["public_score"])

    leaderboard = load_leaderboard(config)
    by_exp: dict[str, dict[str, float]] = {}
    for row in leaderboard:
        name = row.get("experiment_name", "")
        if not name:
            continue
        strategy = row.get("cv_strategy", "")
        score = row.get("mean_score", "")
        if not score:
            continue
        by_exp.setdefault(name, {})[strategy] = float(score)

    records: list[dict[str, object]] = []
    for exp, public in latest_public.items():
        local = by_exp.get(exp, {})
        records.append(
            {
                "experiment": exp,
                "public_score": public,
                "leave_one_species": local.get("leave_one_species"),
                "group_species": local.get("group_species"),
                "random": local.get("random"),
                "moisture_quantile": local.get("moisture_quantile"),
            }
        )

    correlations = _correlations(records)
    recommendation = _recommend_metric(correlations)

    report = {
        "n_public_records": len(latest_public),
        "records": records,
        "correlations_with_public": correlations,
        "recommended_local_metric": recommendation,
    }
    out = config.outputs_dir / "logs" / "eda_metric_alignment.json"
    write_json(out, report)
    print(f"saved: {out}")
    print(f"recommended local metric: {recommendation}")
    for metric, corr in correlations.items():
        print(f"  {metric}: pearson={corr}")


def _correlations(records: list[dict[str, object]]) -> dict[str, float | None]:
    public = [float(r["public_score"]) for r in records]
    out: dict[str, float | None] = {}
    for key in ("leave_one_species", "group_species", "random", "moisture_quantile"):
        local = [float(r[key]) for r in records if r.get(key) is not None]
        pub = [float(r["public_score"]) for r in records if r.get(key) is not None]
        if len(local) < 3:
            out[key] = None
            continue
        out[key] = round(_pearson(pub, local), 4)
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / max(den, 1e-12)


def _recommend_metric(correlations: dict[str, float | None]) -> str:
    valid = {k: abs(v) for k, v in correlations.items() if v is not None}
    if not valid:
        return "moisture_quantile"
    # 符号付きで Public に近い（正の相関）を優先
    signed = {k: v for k, v in correlations.items() if v is not None}
    positive = {k: v for k, v in signed.items() if v > 0}
    if positive:
        return max(positive.items(), key=lambda x: x[1])[0]
    return min(valid.items(), key=lambda x: x[1])[0]


if __name__ == "__main__":
    main()
