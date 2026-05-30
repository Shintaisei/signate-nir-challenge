#!/usr/bin/env python3
"""train/test のドメインギャップを定量化し、根本改善の仮説を出力する。"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config  # noqa: E402
from pipeline.data import load_raw_data  # noqa: E402
from pipeline.preprocessors import get_preprocessor  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="train/test ドメインギャップ分析")
    parser.add_argument(
        "--preprocessor",
        default="spectral_snv_diff1",
        help="前処理名（例: spectral, spectral_snv_diff1）",
    )
    parser.add_argument(
        "--output",
        default="",
        help="出力JSONパス（省略時は outputs/logs/domain_shift_report_<prep>.json）",
    )
    args = parser.parse_args()

    config = load_config()
    raw = load_raw_data(config)
    prep_name = args.preprocessor
    prep = get_preprocessor(prep_name)
    prep.fit(raw.train, target_col=config.target_col, meta_cols=config.meta_cols)
    X_train, y_train = prep.transform_train(raw.train)
    X_test = prep.transform_test(raw.test)

    train_species_means = _mean_spectrum_by_group(raw.train, X_train, "樹種")
    test_species_means = _mean_spectrum_by_group(raw.test, X_test, "樹種")

    nn_map = _nearest_train_species(test_species_means, train_species_means)
    moisture_overlap = _moisture_range_overlap(raw.train, config.target_col)

    anchor_idx = _anchor_wavelength_index(X_train, y_train)
    train_target = _numeric_summary(y_train)
    test_pred_anchor = _predict_anchor(X_train, y_train, X_test, anchor_idx)
    test_pred_summary = _numeric_summary(test_pred_anchor)

    report = {
        "summary": {
            "preprocessor": prep_name,
            "train_species_n": len(train_species_means),
            "test_species_n": len(test_species_means),
            "species_name_overlap": 0,
            "anchor_wavelength_index": anchor_idx,
        },
        "train_target": train_target,
        "test_anchor_prediction": test_pred_summary,
        "prediction_vs_train_target": {
            "mean_shift": test_pred_summary["mean"] - train_target["mean"],
            "std_ratio": test_pred_summary["std"] / max(train_target["std"], 1e-12),
            "range_gap": (test_pred_summary["max"] - test_pred_summary["min"])
            - (train_target["max"] - train_target["min"]),
        },
        "moisture_range_overlap": moisture_overlap,
        "test_to_nearest_train_species": nn_map,
        "train_species_target": _target_by_species(raw.train, config.target_col),
        "hypotheses": _build_hypotheses(
            nn_map=nn_map,
            moisture_overlap=moisture_overlap,
            pred_summary=test_pred_summary,
            train_target=train_target,
        ),
    }

    if args.output:
        out = Path(args.output)
    else:
        safe = prep_name.replace("/", "_")
        out = config.outputs_dir / "logs" / f"domain_shift_report_{safe}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report)
    print(f"saved: {out}")


def _mean_spectrum_by_group(
    rows: list[dict[str, str]],
    X: list[list[float]],
    group_col: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for row, features in zip(rows, X):
        grouped[str(row[group_col])].append(features)
    return {group: _mean_vector(samples) for group, samples in grouped.items()}


def _mean_vector(samples: list[list[float]]) -> list[float]:
    if not samples:
        return []
    dim = len(samples[0])
    return [sum(row[j] for row in samples) / len(samples) for j in range(dim)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / max(na * nb, 1e-12)


def _nearest_train_species(
    test_means: dict[str, list[float]],
    train_means: dict[str, list[float]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for test_species, test_vec in sorted(test_means.items()):
        scored = [
            (train_species, _cosine(test_vec, train_vec))
            for train_species, train_vec in train_means.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        best_train, best_sim = scored[0]
        out.append(
            {
                "test_species": test_species,
                "nearest_train_species": best_train,
                "cosine_similarity": round(best_sim, 4),
                "top3": [{"train": name, "cos": round(sim, 4)} for name, sim in scored[:3]],
            }
        )
    out.sort(key=lambda row: float(row["cosine_similarity"]))
    return out


def _moisture_range_overlap(train: list[dict[str, str]], target_col: str) -> dict[str, float]:
    values = [float(row[target_col]) for row in train]
    lo, hi = min(values), max(values)
    return {
        "train_min": lo,
        "train_max": hi,
        "train_mean": sum(values) / len(values),
        "train_std": math.sqrt(sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)),
    }


def _anchor_wavelength_index(X: list[list[float]], y: list[float]) -> int:
    n = len(y)
    y_mean = sum(y) / n
    y_std = math.sqrt(sum((yi - y_mean) ** 2 for yi in y) / max(n - 1, 1))
    best_idx = 0
    best_corr = -1.0
    for j in range(len(X[0])):
        col = [row[j] for row in X]
        x_mean = sum(col) / n
        x_std = math.sqrt(sum((xi - x_mean) ** 2 for xi in col) / max(n - 1, 1))
        cov = sum((col[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / max(n - 1, 1)
        corr = abs(cov / max(x_std * y_std, 1e-12))
        if corr > best_corr:
            best_corr = corr
            best_idx = j
    return best_idx


def _predict_anchor(
    X_train: list[list[float]],
    y_train: list[float],
    X_test: list[list[float]],
    feature_idx: int,
) -> list[float]:
    x = [row[feature_idx] for row in X_train]
    x_mean = sum(x) / len(x)
    y_mean = sum(y_train) / len(y_train)
    x_var = sum((xi - x_mean) ** 2 for xi in x)
    if x_var <= 1e-12:
        slope = 0.0
    else:
        slope = sum((x[i] - x_mean) * (y_train[i] - y_mean) for i in range(len(x))) / x_var
    intercept = y_mean - slope * x_mean
    return [intercept + slope * row[feature_idx] for row in X_test]


def _target_by_species(train: list[dict[str, str]], target_col: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in train:
        grouped[str(row["樹種"])].append(float(row[target_col]))
    return {species: _numeric_summary(values) for species, values in sorted(grouped.items())}


def _numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "median": sorted(values)[len(values) // 2],
        "max": max(values),
        "std": math.sqrt(sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)),
        "count": float(len(values)),
    }


def _build_hypotheses(
    *,
    nn_map: list[dict[str, object]],
    moisture_overlap: dict[str, float],
    pred_summary: dict[str, float],
    train_target: dict[str, float],
) -> list[dict[str, str]]:
    low_sim = [row for row in nn_map if float(row["cosine_similarity"]) < 0.85]
    mean_shift = pred_summary["mean"] - train_target["mean"]
    hypotheses = [
        {
            "id": "H1_moisture_cv",
            "priority": "high",
            "title": "樹種LOOではなく含水率層化CVで評価",
            "rationale": "train/test樹種非重複のためLOOはPublicと乖離。含水率軸の外挿をローカルで模倣する。",
            "action": "python run.py <exp> --cv moisture_quantile",
        },
        {
            "id": "H2_spectral_transfer",
            "priority": "high",
            "title": "test樹種→最近傍train樹種のスペクトルテンプレート補正",
            "rationale": f"低類似test樹種 {len(low_sim)} / {len(nn_map)} 件。樹種名ではなくスペクトル距離で知識移転。",
            "action": "新predictor: nearest_train_species_linear（実装候補）",
        },
        {
            "id": "H3_range_calibration",
            "priority": "medium",
            "title": "予測レンジの系統バイアス補正",
            "rationale": f"test予測meanシフト={mean_shift:.2f}, range_gap指標要OOF確認。",
            "action": "OOFで range/bias 補正（clipはPublic悪化済みのため線形のみ）",
        },
        {
            "id": "H4_stop_blend",
            "priority": "done",
            "title": "ブレンド・微調整は停止",
            "rationale": "Public同点のraw1fを混ぜても24.6。根本はドメインシフト。",
            "action": "提出はアンカーのみ",
        },
    ]
    return hypotheses


def _print_summary(report: dict[str, object]) -> None:
    print("=== domain shift summary ===")
    pred = report["test_anchor_prediction"]
    train = report["train_target"]
    print(f"train target mean={train['mean']:.2f} std={train['std']:.2f}")
    print(f"test pred(anchor) mean={pred['mean']:.2f} std={pred['std']:.2f}")
    shift = report["prediction_vs_train_target"]
    print(f"mean_shift={shift['mean_shift']:.2f} std_ratio={shift['std_ratio']:.3f}")
    print("\nnearest train species (lowest cosine first):")
    for row in report["test_to_nearest_train_species"][:6]:
        print(
            f"  test={row['test_species']:<8} -> train={row['nearest_train_species']:<8} "
            f"cos={row['cosine_similarity']}"
        )
    print("\nhypotheses:")
    for h in report["hypotheses"]:
        print(f"  [{h['priority']}] {h['id']}: {h['title']}")


if __name__ == "__main__":
    main()
