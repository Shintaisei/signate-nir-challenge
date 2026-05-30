"""EDA スクリプト共通ユーティリティ。"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from pipeline.config import CompetitionConfig, load_config
from pipeline.data import load_raw_data
from pipeline.preprocessors import get_preprocessor
from pipeline.types import Matrix, Rows, Vector


def setup_src_path() -> None:
    import sys

    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    mean = sum(values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "median": sorted(values)[len(values) // 2],
        "std": math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)),
        "count": float(len(values)),
    }


def read_submission(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            out[int(row[0])] = float(row[1])
    return out


def mean_spectrum_by_group(rows: Rows, X: Matrix, group_col: str) -> dict[str, list[float]]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for row, features in zip(rows, X):
        grouped[str(row[group_col])].append(features)
    return {g: _mean_vector(samples) for g, samples in grouped.items()}


def _mean_vector(samples: list[list[float]]) -> list[float]:
    if not samples:
        return []
    dim = len(samples[0])
    return [sum(row[j] for row in samples) / len(samples) for j in range(dim)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / max(na * nb, 1e-12)


def nearest_train_species(
    test_means: dict[str, list[float]],
    train_means: dict[str, list[float]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for test_species, test_vec in test_means.items():
        scored = [(name, cosine(test_vec, vec)) for name, vec in train_means.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_train, best_sim = scored[0]
        out[test_species] = {
            "nearest_train_species": best_train,
            "cosine_similarity": round(best_sim, 4),
            "top3": [{"train": n, "cos": round(s, 4)} for n, s in scored[:3]],
        }
    return out


def target_by_species(rows: Rows, target_col: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["樹種"])].append(float(row[target_col]))
    return {s: numeric_summary(vals) for s, vals in sorted(grouped.items())}


def load_experiment_predictions(
    experiment_name: str,
    config: CompetitionConfig | None = None,
) -> tuple[Rows, Rows, dict[int, float]]:
    from experiments.registry import resolve_experiment

    config = config or load_config()
    raw = load_raw_data(config)
    preprocessor, predictor, _ = resolve_experiment(experiment_name)

    preprocessor.fit(raw.train, target_col=config.target_col, meta_cols=config.meta_cols)
    X_train, y_train = preprocessor.transform_train(raw.train)
    X_test = preprocessor.transform_test(raw.test)

    if hasattr(predictor, "set_train_context"):
        predictor.set_train_context(raw.train, config)
    if hasattr(predictor, "set_group_labels"):
        predictor.set_group_labels([str(row["樹種"]) for row in raw.train])
    predictor.fit(X_train, y_train)
    if hasattr(predictor, "predict_test"):
        preds = predictor.predict_test(raw.test, X_test)
    else:
        preds = predictor.predict(X_test)

    pred_map = {int(row["sample number"]): float(p) for row, p in zip(raw.test, preds)}
    return raw.train, raw.test, pred_map


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
