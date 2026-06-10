#!/usr/bin/env python3
"""Sample-number monotonicity diagnostics for NIR submissions."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submit.csv"
SUBMISSION_DIR = ROOT / "data" / "submissions"
OUTPUT_ROOT = ROOT / "outputs" / "nir_sample_order_monotonic"
ANCHOR_NAME = "nir_yj_oof_affine_s0p10_mc1"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class MonoSpec:
    method: str
    direction: str
    weight: float


def main() -> None:
    data = load_data()
    anchor = read_submission(SUBMISSION_DIR / f"{ANCHOR_NAME}.csv", data["ids"])
    sources = build_sources()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    for source in sources:
        pred = read_submission(source.path, data["ids"])
        block_rows.extend(source_block_diagnostics(source.name, pred, data))
        for spec in build_specs():
            adjusted = apply_monotonic(pred, data["sample"], data["species"], spec)
            name = f"nir_mono_{source.name}_{spec.method}_{spec.direction}_w{tag(spec.weight)}"
            path = candidate_dir / f"{name}.csv"
            pd.DataFrame({0: data["ids"], 1: adjusted}).to_csv(path, index=False, header=False)
            row = diagnostics(
                name=name,
                source=source,
                spec=spec,
                pred=adjusted,
                source_pred=pred,
                anchor=anchor,
                data=data,
                path=path,
            )
            rows.append(row)
            print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "sample_order_monotonic_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "sample_order_monotonic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    with (out_dir / "source_block_diagnostics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(block_rows[0]))
        writer.writeheader()
        writer.writerows(block_rows)

    print("\nTop candidates:")
    for row in rows_sorted[:30]:
        print(
            f"{row['experiment']}: source={row['source']} method={row['method']} "
            f"dir={row['direction']} w={row['weight']} anchor_rmse={row['anchor_diff_rmse']:.4f} "
            f"src_rmse={row['source_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
            f"sp={row['max_abs_species_mean_shift']:.4f} range_ratio={row['range_ratio_vs_source']:.4f}"
        )
    print(f"saved monotonic diagnostics: {out_dir}")


def build_sources() -> list[SourceSpec]:
    names = [
        ANCHOR_NAME,
        "nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p1_mc1",
        "nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p08_mc1",
    ]
    return [SourceSpec(name, SUBMISSION_DIR / f"{name}.csv") for name in names if (SUBMISSION_DIR / f"{name}.csv").exists()]


def build_specs() -> list[MonoSpec]:
    specs: list[MonoSpec] = []
    for direction in ["increasing", "decreasing"]:
        specs.append(MonoSpec("hard_sort", direction, 1.0))
        for weight in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            specs.append(MonoSpec("soft_sort", direction, weight))
        for weight in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            specs.append(MonoSpec("isotonic", direction, weight))
    return specs


def load_data() -> dict[str, np.ndarray]:
    test = pd.read_csv(TEST_PATH, encoding="cp932")
    sample = pd.read_csv(SAMPLE_PATH, header=None)
    ids = test["sample number"].to_numpy()
    if not np.array_equal(ids, sample[0].to_numpy()):
        raise ValueError("sample order mismatch")
    return {
        "ids": ids,
        "sample": test["sample number"].to_numpy(float),
        "species": test["species number"].to_numpy(),
    }


def read_submission(path: Path, ids: np.ndarray) -> np.ndarray:
    values: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if row:
                values[int(float(row[0]))] = float(row[1])
    return np.asarray([values[int(i)] for i in ids], dtype=float)


def apply_monotonic(pred: np.ndarray, sample: np.ndarray, species: np.ndarray, spec: MonoSpec) -> np.ndarray:
    target = pred.copy()
    for group in np.unique(species):
        idx = np.where(species == group)[0]
        order = idx[np.argsort(sample[idx], kind="mergesort")]
        y = pred[order]
        if spec.method in {"hard_sort", "soft_sort"}:
            sorted_y = np.sort(y)
            if spec.direction == "decreasing":
                sorted_y = sorted_y[::-1]
            target[order] = sorted_y
        elif spec.method == "isotonic":
            x = np.arange(len(order), dtype=float)
            increasing = spec.direction == "increasing"
            iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
            target[order] = iso.fit_transform(x, y)
        else:
            raise ValueError(spec.method)
    if spec.method == "hard_sort":
        return target
    adjusted = (1.0 - spec.weight) * pred + spec.weight * target
    return adjusted


def diagnostics(
    *,
    name: str,
    source: SourceSpec,
    spec: MonoSpec,
    pred: np.ndarray,
    source_pred: np.ndarray,
    anchor: np.ndarray,
    data: dict[str, np.ndarray],
    path: Path,
) -> dict[str, object]:
    diff_anchor = pred - anchor
    diff_source = pred - source_pred
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    species = data["species"]
    return {
        "experiment": name,
        "source": source.name,
        "method": spec.method,
        "direction": spec.direction,
        "weight": spec.weight,
        "anchor_diff_rmse": rmse0(diff_anchor),
        "anchor_diff_max_abs": float(np.max(np.abs(diff_anchor))),
        "anchor_diff_mean": float(np.mean(diff_anchor)),
        "source_diff_rmse": rmse0(diff_source),
        "source_diff_max_abs": float(np.max(np.abs(diff_source))),
        "max_abs_species_mean_shift": max(abs(float(np.mean(diff_anchor[species == group]))) for group in np.unique(species)),
        "bottom_decile_delta": float(np.mean(diff_anchor[lo])),
        "top_decile_delta": float(np.mean(diff_anchor[hi])),
        "range_ratio_vs_source": float((np.max(pred) - np.min(pred)) / max(np.max(source_pred) - np.min(source_pred), 1e-12)),
        "negative_count": int(np.sum(pred < 0)),
        "submission_path": str(path),
    }


def source_block_diagnostics(source_name: str, pred: np.ndarray, data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for group in sorted(np.unique(data["species"])):
        idx = np.where(data["species"] == group)[0]
        order = idx[np.argsort(data["sample"][idx], kind="mergesort")]
        x = data["sample"][order]
        y = pred[order]
        rows.append(
            {
                "source": source_name,
                "species": int(group),
                "n": int(len(order)),
                "sample_pred_corr": safe_corr(x, y),
                "first_pred": float(y[0]),
                "last_pred": float(y[-1]),
                "pred_slope_ols": float(np.polyfit(x, y, 1)[0]) if len(order) > 1 else 0.0,
                "violations_increasing": int(np.sum(np.diff(y) < 0)),
                "violations_decreasing": int(np.sum(np.diff(y) > 0)),
            }
        )
    return rows


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    method = str(row["method"])
    direction = str(row["direction"])
    anchor_rmse = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    low = abs(float(row["bottom_decile_delta"]))
    top = abs(float(row["top_decile_delta"]))
    range_ratio = float(row["range_ratio_vs_source"])
    penalty = 0.0
    if method == "hard_sort":
        penalty += 5.0
    if direction != "increasing":
        penalty += 0.25
    if anchor_rmse > 0.10 or max_abs > 0.25 or species > 0.04 or low > 0.08 or top > 0.08:
        penalty += 5.0
    if range_ratio < 0.95 or range_ratio > 1.05:
        penalty += 2.0
    return (penalty + anchor_rmse + species, max_abs, species)


def rmse0(values: np.ndarray) -> float:
    return float(math.sqrt(np.mean(values**2)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: anchor_rmse={row['anchor_diff_rmse']:.4f} "
        f"src_rmse={row['source_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} range={row['range_ratio_vs_source']:.4f}"
    )


if __name__ == "__main__":
    main()
