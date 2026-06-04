#!/usr/bin/env python3
"""Build submission candidates from the wavelength-set EDA landscape."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402
from pipeline.cv import group_kfold  # noqa: E402
from pipeline.data import load_raw_data  # noqa: E402
from pipeline.metrics import rmse  # noqa: E402
from pipeline.submission import build_submission_frame, write_submission  # noqa: E402


CMM16_KEYS = (
    "snv:center_minus_mean:1332:1336",
    "snv:center_minus_mean:1352:1356",
    "snv:center_minus_mean:680:720",
    "snv:center_minus_mean:1192:1196",
    "snv:center_minus_mean:908:912",
    "snv:center_minus_mean:708:712",
    "raw:center_minus_mean:1344:1348",
    "smooth5:center_minus_mean:1330:1340",
    "raw:center_minus_mean:1332:1336",
    "smooth5:center_minus_mean:1344:1348",
    "smooth5:center_minus_mean:576:580",
    "snv:center_minus_mean:828:832",
    "smooth5:center_minus_mean:1320:1324",
    "raw:center_minus_mean:1320:1324",
    "smooth5_diff1:center_minus_mean:1200:1440",
    "snv:center_minus_mean:1320:1324",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    raw = load_raw_data(cfg)
    feature_cols = [c for c in raw.train[0] if c not in set(cfg.meta_cols) | {cfg.target_col}]
    X_train_raw = _matrix(raw.train, feature_cols)
    X_test_raw = _matrix(raw.test, feature_cols)
    y = np.asarray([float(row[cfg.target_col]) for row in raw.train], dtype=float)
    train_species = np.asarray([str(row.get("species number", "")) for row in raw.train], dtype=object)
    test_species = np.asarray([str(row.get("species number", "")) for row in raw.test], dtype=object)
    train_sample = np.asarray([float(row[cfg.id_col]) for row in raw.train], dtype=float)
    test_sample = np.asarray([float(row[cfg.id_col]) for row in raw.test], dtype=float)

    X_train = _features_from_keys(X_train_raw, CMM16_KEYS)
    X_test = _features_from_keys(X_test_raw, CMM16_KEYS)

    out_dir = args.out_dir or cfg.outputs_dir / f"waveset_submission_candidates_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("candidate_waveset_cmm16_elastic_sort", _build_model("elastic")),
        ("candidate_waveset_cmm16_pls4_sort", _build_model("pls4")),
    ]
    summary_rows: list[dict[str, object]] = []
    for name, model in candidates:
        cv_payload = _evaluate_oof(name, model, X_train, y, train_species, train_sample, raw.train)
        fitted = _build_model("elastic" if "elastic" in name else "pls4")
        fitted.fit(X_train, y)
        pred = np.asarray(fitted.predict(X_test), dtype=float).reshape(-1)
        pred = _sort_decreasing(pred, test_sample, test_species)
        pred = np.clip(pred, 0.0, float(np.max(y)))

        submission_path = cfg.submissions_dir / f"{name}.csv"
        write_submission(build_submission_frame(raw.test, pred.tolist(), id_col=cfg.id_col), submission_path, id_col=cfg.id_col)

        log_path = out_dir / f"{name}.json"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_name": name,
            "submission_path": str(submission_path),
            "feature_keys": list(CMM16_KEYS),
            "model": "elastic003" if "elastic" in name else "pls4",
            "postprocess": "sort_decreasing_by_species_number_sample_number",
            "prediction_summary": {
                "min": float(np.min(pred)),
                "max": float(np.max(pred)),
                "mean": float(np.mean(pred)),
            },
            "cv": cv_payload["summary"],
        }
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(out_dir / f"{name}_oof.csv", cv_payload["oof"])
        summary_rows.append(
            {
                "experiment": name,
                "submission_path": str(submission_path),
                "log_path": str(log_path),
                **cv_payload["summary"],
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "pred_mean": float(np.mean(pred)),
            }
        )

    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.md").write_text(_summary_markdown(summary_rows), encoding="utf-8")
    print(f"saved: {out_dir}")
    for row in summary_rows:
        print(
            f"{row['experiment']}: rmse={float(row['sort_rmse']):.4f} "
            f"guard={float(row['sort_guard_score']):.4f} path={row['submission_path']}"
        )


def _matrix(rows: list[dict[str, str]], feature_cols: list[str]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in feature_cols] for row in rows], dtype=float)


def _prep(X: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return X
    if name == "smooth5":
        p = np.pad(X, ((0, 0), (2, 2)), mode="edge")
        return (p[:, :-4] + p[:, 1:-3] + p[:, 2:-2] + p[:, 3:-1] + p[:, 4:]) / 5.0
    if name == "snv":
        return (X - np.mean(X, axis=1, keepdims=True)) / np.maximum(np.std(X, axis=1, keepdims=True), 1e-12)
    if name == "smooth5_diff1":
        return np.diff(_prep(X, "smooth5"), axis=1)
    raise ValueError(name)


def _features_from_keys(X: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}
    cols = []
    for key in keys:
        view, stat, start_text, end_text = key.split(":")
        if view not in cache:
            cache[view] = _prep(X, view)
        Z = cache[view]
        start = int(start_text)
        end = min(int(end_text), Z.shape[1])
        block = Z[:, start:end]
        if stat != "center_minus_mean":
            raise ValueError(f"unsupported stat: {stat}")
        cols.append(block[:, block.shape[1] // 2] - np.mean(block, axis=1))
    return np.vstack(cols).T


def _build_model(kind: str):
    if kind == "elastic":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=50000, tol=1e-3))
    if kind == "pls4":
        return make_pipeline(StandardScaler(), PLSRegression(n_components=4))
    raise ValueError(kind)


def _evaluate_oof(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    species: np.ndarray,
    sample: np.ndarray,
    train_rows: list[dict[str, str]],
) -> dict[str, object]:
    folds = group_kfold(train_rows, group_col="species number", n_splits=5)
    pred = np.zeros_like(y, dtype=float)
    for fold in folds:
        fold_model = _build_model("elastic" if "elastic" in name else "pls4")
        fold_model.fit(X[fold.train_idx], y[fold.train_idx])
        pred[fold.valid_idx] = np.asarray(fold_model.predict(X[fold.valid_idx]), dtype=float).reshape(-1)
    sorted_pred = _sort_decreasing(pred, sample, species)
    base_worst = _worst_group(y, pred, species)
    sort_worst = _worst_group(y, sorted_pred, species)
    oof = []
    for i, (yy, pp, ss) in enumerate(zip(y, pred, sorted_pred)):
        oof.append(
            {
                "row_index": i,
                "species_number": species[i],
                "sample_number": sample[i],
                "y_true": float(yy),
                "y_pred_base": float(pp),
                "y_pred_sort": float(ss),
                "error_base": float(pp - yy),
                "error_sort": float(ss - yy),
            }
        )
    summary = {
        "base_rmse": rmse(y.tolist(), pred.tolist()),
        "base_mae": float(np.mean(np.abs(pred - y))),
        "base_bias": float(np.mean(pred - y)),
        "base_worst_group": base_worst["group"],
        "base_worst_group_rmse": base_worst["rmse"],
        "base_worst_group_bias": base_worst["bias"],
        "base_guard_score": _guard_score(y, pred, species),
        "sort_rmse": rmse(y.tolist(), sorted_pred.tolist()),
        "sort_mae": float(np.mean(np.abs(sorted_pred - y))),
        "sort_bias": float(np.mean(sorted_pred - y)),
        "sort_worst_group": sort_worst["group"],
        "sort_worst_group_rmse": sort_worst["rmse"],
        "sort_worst_group_bias": sort_worst["bias"],
        "sort_guard_score": _guard_score(y, sorted_pred, species),
    }
    return {"summary": summary, "oof": oof}


def _sort_decreasing(pred: np.ndarray, sample: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = pred.copy()
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        order = idx[np.argsort(sample[idx])]
        out[order] = np.sort(pred[idx])[::-1]
    return out


def _worst_group(y: np.ndarray, pred: np.ndarray, species: np.ndarray) -> dict[str, object]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, sp in enumerate(species):
        grouped[str(sp)].append(i)
    worst = {"group": "", "rmse": -1.0, "bias": 0.0}
    for group, idx_list in grouped.items():
        idx = np.asarray(idx_list, dtype=int)
        value = rmse(y[idx].tolist(), pred[idx].tolist())
        if value > float(worst["rmse"]):
            worst = {"group": group, "rmse": value, "bias": float(np.mean(pred[idx] - y[idx]))}
    return worst


def _guard_score(y: np.ndarray, pred: np.ndarray, species: np.ndarray) -> float:
    worst = _worst_group(y, pred, species)
    return rmse(y.tolist(), pred.tolist()) + 0.10 * float(worst["rmse"]) + 0.15 * abs(float(worst["bias"]))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(rows: list[dict[str, object]]) -> str:
    lines = ["# Waveset Submission Candidates", ""]
    for row in rows:
        lines.append(
            f"- `{row['experiment']}`: sort_rmse={float(row['sort_rmse']):.4f}, "
            f"sort_guard={float(row['sort_guard_score']):.4f}, "
            f"worst={row['sort_worst_group']}:{float(row['sort_worst_group_rmse']):.4f}, "
            f"path={row['submission_path']}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
