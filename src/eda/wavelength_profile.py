#!/usr/bin/env python3
"""波長ごとの含水率相関プロファイルと選択波長の比較。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eda.eda_common import setup_src_path, write_json  # noqa: E402
from experiments.registry import resolve_experiment  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.data import load_raw_data  # noqa: E402
from pipeline.models.single_feature_linear import _select_top1_index  # noqa: E402
from pipeline.preprocessors import get_preprocessor  # noqa: E402


def main() -> None:
    setup_src_path()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessors",
        default="spectral,spectral_snv_diff1",
        help="カンマ区切り",
    )
    args = parser.parse_args()

    config = load_config()
    raw = load_raw_data(config)
    y = [float(row[config.target_col]) for row in raw.train]

    profiles: dict[str, object] = {}
    for prep_name in args.preprocessors.split(","):
        prep_name = prep_name.strip()
        prep = get_preprocessor(prep_name)
        prep.fit(raw.train, target_col=config.target_col, meta_cols=config.meta_cols)
        X, _ = prep.transform_train(raw.train)
        corrs = _abs_correlations(X, y)
        top20 = sorted(enumerate(corrs), key=lambda x: x[1], reverse=True)[:20]
        selected = _select_top1_index(X, y)
        profiles[prep_name] = {
            "n_features": len(corrs),
            "selected_index": selected,
            "selected_abs_corr": corrs[selected],
            "top20": [{"index": i, "abs_corr": round(c, 6)} for i, c in top20],
        }

    _, predictor, _ = resolve_experiment("candidate_linear_1f")
    _, predictor_snv, _ = resolve_experiment("candidate_linear_1f_snv_diff1")

    report = {
        "profiles": profiles,
        "comparison": {
            "raw_selected": profiles.get("spectral", {}).get("selected_index"),
            "snv_diff1_selected": profiles.get("spectral_snv_diff1", {}).get("selected_index"),
            "index_delta": None,
        },
    }
    raw_i = report["comparison"]["raw_selected"]
    snv_i = report["comparison"]["snv_diff1_selected"]
    if raw_i is not None and snv_i is not None:
        report["comparison"]["index_delta"] = int(snv_i) - int(raw_i)

    out = config.outputs_dir / "logs" / "eda_wavelength_profile.json"
    write_json(out, report)
    print(f"saved: {out}")
    for name, prof in profiles.items():
        print(f"  {name}: idx={prof['selected_index']} corr={prof['selected_abs_corr']:.4f}")


def _abs_correlations(X: list[list[float]], y: list[float]) -> list[float]:
    n = len(y)
    y_mean = sum(y) / n
    y_std = math.sqrt(sum((yi - y_mean) ** 2 for yi in y) / max(n - 1, 1))
    out: list[float] = []
    for j in range(len(X[0])):
        col = [row[j] for row in X]
        x_mean = sum(col) / n
        x_std = math.sqrt(sum((xi - x_mean) ** 2 for xi in col) / max(n - 1, 1))
        if x_std <= 1e-12 or y_std <= 1e-12:
            out.append(0.0)
            continue
        cov = sum((col[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / max(n - 1, 1)
        out.append(abs(cov / (x_std * y_std)))
    return out


if __name__ == "__main__":
    main()
