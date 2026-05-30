#!/usr/bin/env python3
"""test 樹種別の予測分布と最近傍 train 樹種の含水率レンジを比較する。"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eda.eda_common import (  # noqa: E402
    load_experiment_predictions,
    mean_spectrum_by_group,
    nearest_train_species,
    numeric_summary,
    setup_src_path,
    target_by_species,
    write_json,
)
from pipeline.config import load_config  # noqa: E402
from pipeline.data import load_raw_data  # noqa: E402
from pipeline.preprocessors import get_preprocessor  # noqa: E402


def main() -> None:
    setup_src_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="candidate_linear_1f")
    parser.add_argument("--preprocessor", default="spectral")
    args = parser.parse_args()

    config = load_config()
    train, test, pred_map = load_experiment_predictions(args.experiment, config)

    prep = get_preprocessor(args.preprocessor)
    prep.fit(train, target_col=config.target_col, meta_cols=config.meta_cols)
    X_train, _ = prep.transform_train(train)
    X_test = prep.transform_test(test)
    train_means = mean_spectrum_by_group(train, X_train, "樹種")
    test_means = mean_spectrum_by_group(test, X_test, "樹種")
    nn = nearest_train_species(test_means, train_means)
    train_target = target_by_species(train, config.target_col)

    by_test_species: list[dict[str, object]] = []
    grouped_preds: dict[str, list[float]] = defaultdict(list)
    for row in test:
        species = str(row["樹種"])
        grouped_preds[species].append(pred_map[int(row["sample number"])])

    for species in sorted(grouped_preds):
        preds = grouped_preds[species]
        pred_sum = numeric_summary(preds)
        nn_info = nn[species]
        ref_species = str(nn_info["nearest_train_species"])
        ref_target = train_target[ref_species]
        pred_sum["vs_nn_train_mean_shift"] = pred_sum["mean"] - ref_target["mean"]
        pred_sum["vs_nn_train_std_ratio"] = pred_sum["std"] / max(ref_target["std"], 1e-12)
        by_test_species.append(
            {
                "test_species": species,
                "n_samples": len(preds),
                "prediction": pred_sum,
                "nearest_train": nn_info,
                "nn_train_moisture": ref_target,
                "range_assessment": _assess_range(pred_sum, ref_target),
            }
        )

    report = {
        "experiment": args.experiment,
        "preprocessor_for_nn": args.preprocessor,
        "by_test_species": by_test_species,
    }
    out = config.outputs_dir / "logs" / f"eda_test_predictions_{args.experiment}.json"
    write_json(out, report)
    print(f"saved: {out}")
    for row in by_test_species:
        print(
            f"  {row['test_species']}: pred_mean={row['prediction']['mean']:.1f} "
            f"nn={row['nearest_train']['nearest_train_species']} "
            f"assessment={row['range_assessment']}"
        )


def _assess_range(pred: dict[str, float], ref: dict[str, float]) -> str:
    if pred["std"] < 0.85 * ref["std"]:
        return "compressed"
    if pred["std"] > 1.15 * ref["std"]:
        return "expanded"
    if abs(pred["mean"] - ref["mean"]) > 0.15 * max(ref["std"], 1.0):
        return "mean_shift"
    return "aligned"


if __name__ == "__main__":
    main()
