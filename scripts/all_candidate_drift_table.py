#!/usr/bin/env python3
"""Build an anchor-drift table for every saved submission."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402
from pipeline.public_score import PUBLIC_BEST_EXPERIMENT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default=PUBLIC_BEST_EXPERIMENT)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "all_candidate_drift_table.csv")
    args = parser.parse_args()

    config = load_config()
    anchor = _read_submission(config.submissions_dir / f"{args.anchor}.csv")
    ids = sorted(anchor, key=_sort_key)
    base = np.asarray([anchor[i] for i in ids], dtype=float)
    species = _test_species(config.test_path, config.id_col, ids)
    public = _public_scores(config.outputs_dir / "public_compare.csv")
    local = _local_scores(config.outputs_dir / "logs")
    calibrator = _calibrator(config, args.anchor)

    rows = []
    for path in sorted(config.submissions_dir.glob("*.csv")):
        name = path.stem
        try:
            pred_map = _read_submission(path)
        except ValueError:
            continue
        if set(pred_map) != set(anchor):
            continue
        pred = np.asarray([pred_map[i] for i in ids], dtype=float)
        features = _features(pred, base, species)
        calibrated = float(calibrator.predict(np.asarray([[features["raw_diff_rmse"], abs(features["bias"]), features["max_abs_species_mean_shift"]]], dtype=float))[0])
        guard = _guard(features, pred, base)
        row = {
            "experiment": name,
            "public": public.get(name, ""),
            "group_species": local.get(name, {}).get("group_species", ""),
            "moisture_quantile": local.get(name, {}).get("moisture_quantile", ""),
            "random": local.get(name, {}).get("random", ""),
            **features,
            "guard_score_rough": guard,
            "calibrated_public_estimate": calibrated,
            "decision": _decision(features, pred),
        }
        rows.append(row)

    rows.sort(key=lambda r: (float(r["calibrated_public_estimate"]), float(r["raw_diff_rmse"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


def _features(pred: np.ndarray, base: np.ndarray, species: list[str]) -> dict[str, float]:
    diff = pred - base
    mean_std = _mean_std_align(pred, base) - base
    species_mean = _species_mean_align(pred, base, species) - base
    species_mean_std = _species_mean_std_align(pred, base, species) - base
    return {
        "raw_diff_rmse": _rmse(diff),
        "bias": float(diff.mean()),
        "max_abs_species_mean_shift": _max_species_bias(diff, species),
        "mean_std_diff_rmse": _rmse(mean_std),
        "species_mean_diff_rmse": _rmse(species_mean),
        "species_mean_std_diff_rmse": _rmse(species_mean_std),
        "corr_to_anchor": _corr(pred, base),
        "pred_mean": float(pred.mean()),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
        "pred_neg": float(np.sum(pred < 0)),
    }


def _decision(features: dict[str, float], pred: np.ndarray) -> str:
    reasons = []
    if features["raw_diff_rmse"] > 10:
        reasons.append("reject_raw_drift")
    if abs(features["bias"]) > 5:
        reasons.append("reject_global_bias")
    if features["max_abs_species_mean_shift"] > 12:
        reasons.append("reject_species_shift")
    if features["species_mean_std_diff_rmse"] > 6:
        reasons.append("reject_shape_drift")
    if float(np.sum(pred < 0)) > 0:
        reasons.append("warn_negative")
    return ";".join(reasons) if reasons else "pass_guard"


def _guard(features: dict[str, float], pred: np.ndarray, base: np.ndarray) -> float:
    return float(
        16.151771
        + 0.12 * features["raw_diff_rmse"]
        + 0.18 * abs(features["bias"])
        + 0.18 * features["max_abs_species_mean_shift"]
        + 0.08 * abs(float(pred.mean() - base.mean()))
        + 0.20 * float(np.sum(pred < 0))
    )


def _calibrator(config, anchor: str):
    rows = []
    public = _public_scores(config.outputs_dir / "public_compare.csv")
    anchor_map = _read_submission(config.submissions_dir / f"{anchor}.csv")
    ids = sorted(anchor_map, key=_sort_key)
    base = np.asarray([anchor_map[i] for i in ids], dtype=float)
    species = _test_species(config.test_path, config.id_col, ids)
    for name, score in public.items():
        path = config.submissions_dir / f"{name}.csv"
        if not path.exists():
            continue
        pred_map = _read_submission(path)
        if set(pred_map) != set(anchor_map):
            continue
        pred = np.asarray([pred_map[i] for i in ids], dtype=float)
        f = _features(pred, base, species)
        rows.append(([f["raw_diff_rmse"], abs(f["bias"]), f["max_abs_species_mean_shift"]], score))
    if len(rows) < 4:
        raise SystemExit("not enough measured rows for calibrator")
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x, y)
    return model


def _read_submission(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if row:
                out[str(row[0])] = float(row[1])
    return out


def _public_scores(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["experiment_name"]] = float(row["public_score"])
            except (KeyError, ValueError):
                continue
    return out


def _local_scores(log_dir: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in log_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        name = str(data.get("experiment_name") or path.stem)
        metrics = out.setdefault(name, {})
        for cv_result in data.get("cv_results", []):
            cv_name = cv_result.get("cv_name") or cv_result.get("splitter")
            value = cv_result.get("rmse")
            if cv_name and isinstance(value, (int, float)):
                metrics[str(cv_name)] = float(value)
    return out


def _test_species(test_path: Path, id_col: str, ids: list[str]) -> list[str]:
    with test_path.open(encoding="cp932", newline="") as f:
        rows = list(csv.DictReader(f))
    species = {str(row[id_col]): str(row.get("species number", "")) for row in rows}
    return [species.get(i, "") for i in ids]


def _mean_std_align(pred: np.ndarray, base: np.ndarray) -> np.ndarray:
    return (pred - pred.mean()) / max(float(pred.std()), 1e-12) * float(base.std()) + float(base.mean())


def _species_mean_align(pred: np.ndarray, base: np.ndarray, species: list[str]) -> np.ndarray:
    out = pred.copy()
    for group in set(species):
        idx = np.asarray([i for i, value in enumerate(species) if value == group], dtype=int)
        out[idx] = pred[idx] - (pred[idx].mean() - base[idx].mean())
    return out


def _species_mean_std_align(pred: np.ndarray, base: np.ndarray, species: list[str]) -> np.ndarray:
    out = pred.copy()
    for group in set(species):
        idx = np.asarray([i for i, value in enumerate(species) if value == group], dtype=int)
        p = pred[idx]
        b = base[idx]
        out[idx] = (p - p.mean()) / max(float(p.std()), 1e-12) * float(b.std()) + float(b.mean())
    return out


def _max_species_bias(diff: np.ndarray, species: list[str]) -> float:
    return max(abs(float(diff[[i for i, s in enumerate(species) if s == group]].mean())) for group in set(species))


def _rmse(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values * values))))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    main()

