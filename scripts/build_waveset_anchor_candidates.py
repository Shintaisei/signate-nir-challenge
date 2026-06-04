#!/usr/bin/env python3
"""Build conservative candidates that add wavelength-set features to anchors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
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


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_kind: str
    alpha: float
    sort: bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--submit", action="store_true")
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

    out_dir = args.out_dir or cfg.outputs_dir / f"waveset_anchor_candidates_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate("candidate_rawwin20pca1_cmm16_ridge1000", "window20pca1_cmm16", 1000.0, False),
        Candidate("candidate_rawwin20pca1_cmm16_ridge3000", "window20pca1_cmm16", 3000.0, False),
        Candidate("candidate_band616_cmm16_ridge10000", "band616_cmm16", 10000.0, False),
        Candidate("candidate_band616_cmm16_ridge10000_sort", "band616_cmm16", 10000.0, True),
    ]

    rows = []
    for candidate in candidates:
        cv_payload = _evaluate_candidate(candidate, X_train_raw, y, train_species, train_sample, raw.train)
        X_train = _make_features_fit_all(candidate.feature_kind, X_train_raw, X_train_raw)
        X_test = _make_features_fit_all(candidate.feature_kind, X_train_raw, X_test_raw)
        model = _model(candidate.alpha)
        model.fit(X_train, y)
        pred = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
        if candidate.sort:
            pred = _sort_decreasing(pred, test_sample, test_species)
        pred = np.clip(pred, 0.0, float(np.max(y)))
        submission_path = cfg.submissions_dir / f"{candidate.name}.csv"
        write_submission(build_submission_frame(raw.test, pred.tolist(), id_col=cfg.id_col), submission_path, id_col=cfg.id_col)
        log_path = out_dir / f"{candidate.name}.json"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_name": candidate.name,
            "feature_kind": candidate.feature_kind,
            "alpha": candidate.alpha,
            "sort": candidate.sort,
            "feature_keys": list(CMM16_KEYS),
            "submission_path": str(submission_path),
            "prediction_summary": {
                "min": float(np.min(pred)),
                "max": float(np.max(pred)),
                "mean": float(np.mean(pred)),
            },
            "cv": cv_payload["summary"],
        }
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(out_dir / f"{candidate.name}_oof.csv", cv_payload["oof"])
        rows.append(
            {
                "experiment": candidate.name,
                "submission_path": str(submission_path),
                "log_path": str(log_path),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "pred_mean": float(np.mean(pred)),
                **cv_payload["summary"],
            }
        )

    _write_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.md").write_text(_summary(rows), encoding="utf-8")
    print(f"saved: {out_dir}")
    for row in rows:
        print(
            f"{row['experiment']}: rmse={float(row['eval_rmse']):.4f} "
            f"guard={float(row['guard_score']):.4f} path={row['submission_path']}"
        )

    if args.submit:
        from pipeline.signate_submit import submit_to_signate

        for row in rows:
            submit_to_signate(Path(str(row["submission_path"])), str(row["experiment"]), config=cfg)


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


def _cmm16_features(X: np.ndarray) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}
    cols = []
    for key in CMM16_KEYS:
        view, stat, start_text, end_text = key.split(":")
        if stat != "center_minus_mean":
            raise ValueError(stat)
        if view not in cache:
            cache[view] = _prep(X, view)
        Z = cache[view]
        start = int(start_text)
        end = min(int(end_text), Z.shape[1])
        block = Z[:, start:end]
        cols.append(block[:, block.shape[1] // 2] - np.mean(block, axis=1))
    return np.vstack(cols).T


def _window20_pca1_fit_transform(X_fit: np.ndarray, X_apply: np.ndarray) -> np.ndarray:
    cols = []
    for start in range(0, X_fit.shape[1], 20):
        end = min(start + 20, X_fit.shape[1])
        pca = PCA(n_components=1, random_state=42)
        pca.fit(X_fit[:, start:end])
        cols.append(pca.transform(X_apply[:, start:end]).reshape(-1))
    return np.vstack(cols).T


def _make_features_fit_all(kind: str, X_fit_raw: np.ndarray, X_apply_raw: np.ndarray) -> np.ndarray:
    cmm = _cmm16_features(X_apply_raw)
    if kind == "window20pca1_cmm16":
        win = _window20_pca1_fit_transform(X_fit_raw, X_apply_raw)
        return np.hstack([win, cmm])
    if kind == "band616_cmm16":
        band = X_apply_raw[:, [616]]
        return np.hstack([band, cmm])
    raise ValueError(kind)


def _model(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _evaluate_candidate(
    candidate: Candidate,
    X_raw: np.ndarray,
    y: np.ndarray,
    species: np.ndarray,
    sample: np.ndarray,
    train_rows: list[dict[str, str]],
) -> dict[str, object]:
    folds = group_kfold(train_rows, group_col="species number", n_splits=5)
    pred = np.zeros_like(y, dtype=float)
    for fold in folds:
        X_tr = _make_features_fit_all(candidate.feature_kind, X_raw[fold.train_idx], X_raw[fold.train_idx])
        X_va = _make_features_fit_all(candidate.feature_kind, X_raw[fold.train_idx], X_raw[fold.valid_idx])
        model = _model(candidate.alpha)
        model.fit(X_tr, y[fold.train_idx])
        pred[fold.valid_idx] = np.asarray(model.predict(X_va), dtype=float).reshape(-1)
    eval_pred = _sort_decreasing(pred, sample, species) if candidate.sort else pred
    worst = _worst_group(y, eval_pred, species)
    oof = []
    for i, (yy, pp, ep) in enumerate(zip(y, pred, eval_pred)):
        oof.append(
            {
                "row_index": i,
                "species_number": species[i],
                "sample_number": sample[i],
                "y_true": float(yy),
                "y_pred_base": float(pp),
                "y_pred_eval": float(ep),
                "error_eval": float(ep - yy),
            }
        )
    summary = {
        "base_rmse": rmse(y.tolist(), pred.tolist()),
        "eval_rmse": rmse(y.tolist(), eval_pred.tolist()),
        "eval_mae": float(np.mean(np.abs(eval_pred - y))),
        "eval_bias": float(np.mean(eval_pred - y)),
        "worst_group": worst["group"],
        "worst_group_rmse": worst["rmse"],
        "worst_group_bias": worst["bias"],
        "guard_score": rmse(y.tolist(), eval_pred.tolist()) + 0.10 * float(worst["rmse"]) + 0.15 * abs(float(worst["bias"])),
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, object]]) -> str:
    lines = ["# Waveset Anchor Candidates", ""]
    for row in rows:
        lines.append(
            f"- `{row['experiment']}`: rmse={float(row['eval_rmse']):.4f}, "
            f"guard={float(row['guard_score']):.4f}, "
            f"worst={row['worst_group']}:{float(row['worst_group_rmse']):.4f}, "
            f"path={row['submission_path']}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
