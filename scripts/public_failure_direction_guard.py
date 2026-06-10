#!/usr/bin/env python3
"""Rank submission candidates by measured Public-failure direction risk.

This guard is deliberately conservative. It does not predict the leaderboard;
it rejects candidates whose anchor-diff shape resembles known Public failures.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "data" / "submissions"
ANCHOR_PATH = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"
BAD_ALPHA3000_PATH = SUBMISSION_DIR / "nir_sg9_snv_pca20_ridge3000.csv"
WEIGHTED_GOLDEN_PATH = SUBMISSION_DIR / "nir_20260607_advgold_rf_shrink_s0p012_c0p06.csv"
OPERATOR_RESIDUAL_PATH = SUBMISSION_DIR / "nir_opdist_pls_raw_msc_sg9_c4_b0p5591_s0p02_c0p1_mc1.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
PUBLIC_COMPARE_PATH = ROOT / "outputs" / "public_compare.csv"
OUTPUT_ROOT = ROOT / "outputs" / "public_failure_guard"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", action="append", type=Path, default=[], help="Extra directory of CSVs.")
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()

    anchor = read_submission(ANCHOR_PATH)
    anchor_ids = anchor[0]
    anchor_pred = anchor[1]
    refs = {
        "bad_alpha3000": read_submission(BAD_ALPHA3000_PATH)[1] - anchor_pred,
        "weighted_golden": read_submission(WEIGHTED_GOLDEN_PATH)[1] - anchor_pred,
        "operator_residual": read_submission(OPERATOR_RESIDUAL_PATH)[1] - anchor_pred,
    }
    test = pd.read_csv(TEST_PATH, encoding="cp932")
    species = test["species number"].to_numpy()
    if len(species) != len(anchor_pred):
        raise ValueError("test species length mismatch")

    public = read_public_scores(PUBLIC_COMPARE_PATH)
    paths = collect_paths(args.candidate_dir)
    rows: list[dict[str, object]] = []
    for path in paths:
        try:
            ids, pred = read_submission(path)
            if not np.array_equal(ids, anchor_ids):
                continue
        except Exception:
            continue
        diff = pred - anchor_pred
        row = diagnostics(path, pred, anchor_pred, diff, refs, species)
        row["public_score"] = public.get(path.stem, "")
        rows.append(row)

    ranked = sorted(rows, key=rank_key)
    out_dir = OUTPUT_ROOT / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "public_failure_guard.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in ranked for key in row}))
        writer.writeheader()
        writer.writerows(ranked)

    print(f"candidates={len(ranked)}")
    print(f"saved: {out_path}")
    print("\nLowest-risk candidates:")
    for row in ranked[: args.top]:
        print(
            f"{row['experiment']}: risk={row['public_failure_risk']:.4f} gate={row['guard_gate']} "
            f"badcorr={row['corr_diff_bad_alpha3000']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"gap={row['decile_gap_top_minus_bottom']:.4f} top10sp={row['top10_abs_max_species_count']} "
            f"public={row['public_score']} reasons={row['reject_reasons']}"
        )


def collect_paths(extra_dirs: list[Path]) -> list[Path]:
    paths = list(SUBMISSION_DIR.glob("*.csv"))
    for directory in extra_dirs:
        resolved = directory if directory.is_absolute() else ROOT / directory
        if resolved.exists():
            paths.extend(resolved.glob("*.csv"))
    return sorted({path.resolve() for path in paths})


def diagnostics(
    path: Path,
    pred: np.ndarray,
    anchor: np.ndarray,
    diff: np.ndarray,
    refs: dict[str, np.ndarray],
    species: np.ndarray,
) -> dict[str, object]:
    lo = anchor <= np.quantile(anchor, 0.10)
    hi = anchor >= np.quantile(anchor, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(species[top10]).value_counts()
    species_shift = max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))
    range_ratio = float((pred.max() - pred.min()) / (anchor.max() - anchor.min()))
    row: dict[str, object] = {
        "experiment": path.stem,
        "submission_path": str(path),
        "anchor_diff_rmse": rmse(diff),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "max_abs_species_mean_shift": species_shift,
        "range_ratio_vs_anchor": range_ratio,
        "pred_min": float(pred.min()),
        "pred_mean": float(pred.mean()),
        "pred_median": float(np.median(pred)),
        "pred_max": float(pred.max()),
        "negative_count": int(np.sum(pred < 0)),
        "bottom_decile_shift": float(np.mean(diff[lo])),
        "top_decile_shift": float(np.mean(diff[hi])),
        "decile_gap_top_minus_bottom": float(np.mean(diff[hi]) - np.mean(diff[lo])),
        "sample_order_corr": safe_corr(diff, np.arange(len(diff), dtype=float)),
        "top10_abs_max_species_count": int(top_species.iloc[0]),
        "top10_abs_species_set": ",".join(map(str, sorted(top_species.index.tolist()))),
    }
    for name, ref_diff in refs.items():
        row[f"corr_diff_{name}"] = safe_corr(diff, ref_diff)
        row[f"rmse_diff_{name}"] = rmse(diff - ref_diff)
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else "pass"
    row["guard_gate"] = len(reasons) == 0
    row["public_failure_risk"] = public_failure_risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    diff = float(row["anchor_diff_rmse"])
    bad_corr = float(row["corr_diff_bad_alpha3000"])
    decile_gap = abs(float(row["decile_gap_top_minus_bottom"]))
    species_shift = float(row["max_abs_species_mean_shift"])
    max_abs = float(row["anchor_diff_max_abs"])
    range_ratio = float(row["range_ratio_vs_anchor"])
    top10_species = int(row["top10_abs_max_species_count"])

    if diff < 0.03:
        reasons.append("near_anchor")
    if diff > 1.00:
        reasons.append("anchor_diff_gt1")
    if max_abs > 3.00:
        reasons.append("max_diff_gt3")
    if species_shift > 0.35:
        reasons.append("species_shift_gt0p35")
    if bad_corr > 0.60:
        reasons.append("bad_alpha3000_corr_gt0p60")
    if decile_gap > 0.50:
        reasons.append("decile_gap_gt0p50")
    if top10_species >= 5:
        reasons.append("top10_species_concentration")
    if not (0.92 <= range_ratio <= 1.08):
        reasons.append("range_shift")
    if int(row["negative_count"]) > 0:
        reasons.append("negative")
    return reasons


def public_failure_risk(row: dict[str, object]) -> float:
    return float(
        2.0 * max(float(row["corr_diff_bad_alpha3000"]), 0.0)
        + 0.8 * abs(float(row["decile_gap_top_minus_bottom"]))
        + 0.7 * abs(float(row["sample_order_corr"]))
        + 0.8 * float(row["max_abs_species_mean_shift"])
        + 0.08 * float(row["anchor_diff_max_abs"])
        + (0.5 if int(row["top10_abs_max_species_count"]) >= 5 else 0.0)
        + (0.4 if not (0.92 <= float(row["range_ratio_vs_anchor"]) <= 1.08) else 0.0)
    )


def read_submission(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None)
    return df[0].to_numpy(), df[1].to_numpy(float)


def read_public_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "experiment_name" not in df or "public_score" not in df:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        try:
            out[str(row["experiment_name"])] = float(row["public_score"])
        except (TypeError, ValueError):
            pass
    return out


def rmse(values: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    penalty = 0.0 if bool(row["guard_gate"]) else 10.0
    return (
        penalty + float(row["public_failure_risk"]),
        float(row["anchor_diff_rmse"]),
        float(row["anchor_diff_max_abs"]),
    )


if __name__ == "__main__":
    main()
