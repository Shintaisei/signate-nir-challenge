"""OOF予測を樹種・目的値レンジ・sample順で診断する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="OOF誤差診断")
    parser.add_argument("experiment", help="例: topk6_ridge_snv_diff1")
    parser.add_argument("--cv", default="leave_one_species", help="CV名")
    args = parser.parse_args()

    config = load_config()
    oof_path = config.outputs_dir / "oof" / f"{args.experiment}_{args.cv}.csv"
    rows = _read_rows(oof_path)
    report = {
        "experiment": args.experiment,
        "cv": args.cv,
        "oof_path": str(oof_path),
        "overall": _summarize(rows),
        "by_species": _by_species(rows),
        "by_species_range": _by_species_range(rows),
        "by_target_bin": _by_target_bin(rows),
        "by_species_position": _by_species_position(rows),
    }

    out_path = config.outputs_dir / "logs" / f"{args.experiment}_{args.cv}_oof_analysis.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    for item in report["by_species"][:8]:
        print(
            f"{item['樹種']}: n={item['count']} rmse={item['rmse']:.4f} "
            f"mae={item['mae']:.4f} bias={item['bias']:.4f}"
        )
    print("hard species position bins:")
    for item in report["by_species_position"]:
        if item["樹種"] in {"ベイスギ", "チェリー", "ホワイトオーク", "ナラ"}:
            print(
                f"{item['樹種']} bin={item['position_bin']} rmse={item['rmse']:.4f} "
                f"bias={item['bias']:.4f} true={item['y_min']:.1f}-{item['y_max']:.1f} "
                f"pred={item['pred_min']:.1f}-{item['pred_max']:.1f}"
            )


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["sample number"] = int(str(row["sample number"]))
        row["species number"] = int(str(row["species number"]))
        row["y_true"] = float(str(row["y_true"]))
        row["y_pred"] = float(str(row["y_pred"]))
        row["error"] = float(str(row["error"]))
        row["abs_error"] = float(str(row["abs_error"]))
    return rows


def _summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    errors = [float(row["error"]) for row in rows]
    abs_errors = [abs(error) for error in errors]
    return {
        "count": float(len(rows)),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "mae": sum(abs_errors) / len(abs_errors),
        "bias": sum(errors) / len(errors),
    }


def _by_species(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["樹種"]), []).append(row)
    out = [{"樹種": species, **_summarize(items)} for species, items in grouped.items()]
    out.sort(key=lambda row: float(row["rmse"]), reverse=True)
    return out


def _by_species_range(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["樹種"]), []).append(row)

    out: list[dict[str, object]] = []
    for species, items in grouped.items():
        y_true = [float(row["y_true"]) for row in items]
        y_pred = [float(row["y_pred"]) for row in items]
        true_range = max(y_true) - min(y_true)
        pred_range = max(y_pred) - min(y_pred)
        out.append(
            {
                "樹種": species,
                "true_min": min(y_true),
                "true_max": max(y_true),
                "true_range": true_range,
                "pred_min": min(y_pred),
                "pred_max": max(y_pred),
                "pred_range": pred_range,
                "range_gap": pred_range - true_range,
            }
        )
    out.sort(key=lambda row: abs(float(row["range_gap"])), reverse=True)
    return out


def _by_target_bin(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sorted_rows = sorted(rows, key=lambda row: float(row["y_true"]))
    n = len(sorted_rows)
    out: list[dict[str, object]] = []
    for i in range(5):
        start = n * i // 5
        end = n * (i + 1) // 5
        chunk = sorted_rows[start:end]
        summary = _summarize(chunk)
        out.append(
            {
                "bin": i,
                "y_min": min(float(row["y_true"]) for row in chunk),
                "y_max": max(float(row["y_true"]) for row in chunk),
                "pred_min": min(float(row["y_pred"]) for row in chunk),
                "pred_max": max(float(row["y_pred"]) for row in chunk),
                **summary,
            }
        )
    return out


def _by_species_position(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["樹種"]), []).append(row)

    for species, items in grouped.items():
        items = sorted(items, key=lambda row: int(row["sample number"]))
        n = len(items)
        for i in range(4):
            start = n * i // 4
            end = n * (i + 1) // 4
            chunk = items[start:end]
            out.append(
                {
                    "樹種": species,
                    "position_bin": i,
                    "y_min": min(float(row["y_true"]) for row in chunk),
                    "y_max": max(float(row["y_true"]) for row in chunk),
                    "pred_min": min(float(row["y_pred"]) for row in chunk),
                    "pred_max": max(float(row["y_pred"]) for row in chunk),
                    **_summarize(chunk),
                }
            )
    out.sort(key=lambda row: (str(row["樹種"]), int(row["position_bin"])))
    return out


if __name__ == "__main__":
    main()
