#!/usr/bin/env python3
"""Calibrated direct nonlinear latent search.

The calibration is fitted from the nonlinear model's own OOF predictions to y.
It is not an anchor blend: the anchor is used only for diagnostics and gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

import nir_nonlinear_latent_guard_search as nl
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_nonlinear_calibrated_guard"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-specs", type=int, default=None)
    parser.add_argument("--families", default="svr,krr,rff,elm")
    args = parser.parse_args()

    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)
    refs = nl.load_reference_diffs(anchor_test)
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF", flush=True)
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}", flush=True)

    families = set(filter(None, (part.strip() for part in args.families.split(","))))
    specs = build_specs(families)
    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    print(f"specs={len(specs)}", flush=True)

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        try:
            raw_oof = np.clip(nl.make_oof(spec, data["X_train"], data["y"], data["groups"]), 0, None)
            raw_test, test_meta = nl.fit_predict_spec_with_meta(spec, data["X_train"], data["y"], data["X_test"])
            raw_test = np.clip(raw_test, 0, None)
            if nl.rmse(raw_test - anchor_test) > 10.0:
                raise ValueError("uncalibrated test diff too large for conservative calibration")
            cal_oof, cal_meta = make_calibrated_oof(raw_oof, data["y"], data["groups"])
            full_intercept, full_slope = fit_affine(raw_oof, data["y"])
            full_slope_used = float(np.clip(full_slope, 0.5, 1.2))
            pred = np.clip(apply_affine(raw_test, full_intercept, full_slope_used), 0, None)
        except Exception as exc:
            row = failed_row(spec, exc)
            rows.append(row)
            print(f"  SKIP {type(exc).__name__}: {exc}", flush=True)
            continue

        path = candidate_dir / f"{spec.name}_oofcal.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row = diagnostics(
            spec=spec,
            path=path,
            pred=pred,
            oof=np.clip(cal_oof, 0, None),
            raw_test=raw_test,
            raw_oof=raw_oof,
            anchor_test=anchor_test,
            anchor_oof=anchor_oof,
            anchor_oof_rmse=anchor_oof_rmse,
            y=data["y"],
            groups=data["groups"],
            test_species=test_species,
            refs=refs,
            test_meta=test_meta,
            cal_meta={
                **cal_meta,
                "full_intercept": full_intercept,
                "full_slope_raw": full_slope,
                "full_slope_used": full_slope_used,
            },
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "nonlinear_calibrated_guard_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "nonlinear_calibrated_guard_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop calibrated nonlinear candidates:")
    for row in rows_sorted[:80]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"badcorr={row['corr_diff_bad_alpha3000']:.4f} direct={row['direct_oof_delta_vs_anchor']:.4f} "
            f"fold={row['direct_improved_fold_count']}/5 worst={row['direct_worst_fold_delta']:.4f} "
            f"slope={row['full_slope_raw']:.4f}"
        )
    print(f"saved calibrated nonlinear diagnostics: {out_dir}")


def build_specs(families: set[str]) -> list[nl.NonlinearSpec]:
    specs: list[nl.NonlinearSpec] = []
    preprocesses = ["sg9_snv", "msc_sg9", "detrend_snv", "sg9_snv_stack_d1", "sg9_snv_d1"]
    latent_grid = [("pls", 3), ("pls", 5), ("pls", 8), ("pca", 12)]
    for preprocess in preprocesses:
        for latent, n_components in latent_grid:
            prefix = f"nir_nlcal_{preprocess}_{latent}{n_components}_raw"
            if "svr" in families:
                for c in [1.0, 3.0, 10.0]:
                    for epsilon in [0.10, 0.20]:
                        for gamma in ["scale", "0.03"]:
                            specs.append(
                                nl.NonlinearSpec(
                                    name=f"{prefix}_svr_C{nl.tag(c)}_e{nl.tag(epsilon)}_g{nl.tag_gamma(gamma)}",
                                    preprocess=preprocess,
                                    latent=latent,
                                    n_components=n_components,
                                    target="raw",
                                    model="svr",
                                    c=c,
                                    epsilon=epsilon,
                                    gamma=gamma,
                                )
                            )
            if "krr" in families:
                for alpha in [30.0, 100.0]:
                    for gamma in ["0.03", "0.10"]:
                        specs.append(
                            nl.NonlinearSpec(
                                name=f"{prefix}_krr_a{nl.tag(alpha)}_g{nl.tag_gamma(gamma)}",
                                preprocess=preprocess,
                                latent=latent,
                                n_components=n_components,
                                target="raw",
                                model="krr",
                                alpha=alpha,
                                gamma=gamma,
                            )
                        )
            if "rff" in families:
                for alpha in [1000.0, 3000.0]:
                    specs.append(
                        nl.NonlinearSpec(
                            name=f"{prefix}_rff_nf128_a{nl.tag(alpha)}_g0p03",
                            preprocess=preprocess,
                            latent=latent,
                            n_components=n_components,
                            target="raw",
                            model="rff",
                            alpha=alpha,
                            gamma="0.03",
                            n_features=128,
                            seed=11,
                            seed_count=3,
                        )
                    )
            if "elm" in families:
                for alpha in [1000.0, 3000.0]:
                    specs.append(
                        nl.NonlinearSpec(
                            name=f"{prefix}_elm_h64_a{nl.tag(alpha)}",
                            preprocess=preprocess,
                            latent=latent,
                            n_components=n_components,
                            target="raw",
                            model="elm",
                            alpha=alpha,
                            hidden=64,
                            seed=17,
                            seed_count=5,
                        )
                    )
    return specs


def make_calibrated_oof(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    out = np.empty(len(y), dtype=float)
    intercepts: list[float] = []
    slopes_raw: list[float] = []
    slopes_used: list[float] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(pred.reshape(-1, 1), y, groups):
        intercept, slope = fit_affine(pred[train_idx], y[train_idx])
        slope_used = float(np.clip(slope, 0.5, 1.2))
        out[valid_idx] = apply_affine(pred[valid_idx], intercept, slope_used)
        intercepts.append(intercept)
        slopes_raw.append(slope)
        slopes_used.append(slope_used)
    return out, {
        "cal_intercept_median": float(np.median(intercepts)),
        "cal_intercept_max_abs": float(np.max(np.abs(intercepts))),
        "cal_slope_raw_median": float(np.median(slopes_raw)),
        "cal_slope_raw_min": float(np.min(slopes_raw)),
        "cal_slope_raw_max": float(np.max(slopes_raw)),
        "cal_slope_used_median": float(np.median(slopes_used)),
    }


def fit_affine(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(pred, y, deg=1)
    return float(intercept), float(slope)


def apply_affine(pred: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return intercept + slope * pred


def diagnostics(
    *,
    spec: nl.NonlinearSpec,
    path: Path,
    pred: np.ndarray,
    oof: np.ndarray,
    raw_test: np.ndarray,
    raw_oof: np.ndarray,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_oof_rmse: float,
    y: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    refs: dict[str, np.ndarray],
    test_meta: dict[str, float],
    cal_meta: dict[str, float],
) -> dict[str, object]:
    diff = pred - anchor_test
    raw_diff = raw_test - anchor_test
    oof_rmse = rmse(y, oof)
    raw_oof_rmse = rmse(y, raw_oof)
    fold = nl.fold_delta_stats(y, oof, anchor_oof, groups)
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    row: dict[str, object] = {
        "status": "ok",
        "experiment": f"{spec.name}_oofcal",
        "submission_path": str(path),
        "preprocess": spec.preprocess,
        "latent": spec.latent,
        "n_components": spec.n_components,
        "target": spec.target,
        "model": spec.model,
        "alpha": spec.alpha,
        "gamma": spec.gamma,
        "C": spec.c,
        "epsilon": spec.epsilon,
        "n_features": spec.n_features,
        "hidden": spec.hidden,
        "anchor_oof_rmse": anchor_oof_rmse,
        "raw_direct_oof_rmse": raw_oof_rmse,
        "raw_direct_oof_delta_vs_anchor": raw_oof_rmse - anchor_oof_rmse,
        "direct_oof_rmse": oof_rmse,
        "direct_oof_delta_vs_anchor": oof_rmse - anchor_oof_rmse,
        "direct_improved_fold_count": fold["improved_count"],
        "direct_worst_fold_delta": fold["worst_delta"],
        "direct_fold_delta_std": fold["delta_std"],
        "raw_anchor_diff_rmse": nl.rmse(raw_diff),
        "raw_anchor_diff_max_abs": float(np.max(np.abs(raw_diff))),
        "seed_pred_std_mean": test_meta["seed_pred_std_mean"],
        "seed_pred_std_max": test_meta["seed_pred_std_max"],
        "seed_pred_rmse_mean": test_meta["seed_pred_rmse_mean"],
        "pred_min": float(pred.min()),
        "pred_mean": float(pred.mean()),
        "pred_median": float(np.median(pred)),
        "pred_max": float(pred.max()),
        "pred_std": float(np.std(pred)),
        "pred_std_ratio_vs_anchor": float(np.std(pred) / np.std(anchor_test)),
        "negative_count": int(np.sum(pred < 0)),
        "anchor_diff_rmse": nl.rmse(diff),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": nl.safe_corr(pred, anchor_test),
        "max_abs_species_mean_shift": nl.max_abs_species_shift(diff, test_species),
        "range_ratio_vs_anchor": float((pred.max() - pred.min()) / (anchor_test.max() - anchor_test.min())),
        "bottom_decile_shift": float(np.mean(diff[lo])),
        "top_decile_shift": float(np.mean(diff[hi])),
        "decile_gap_top_minus_bottom": float(np.mean(diff[hi]) - np.mean(diff[lo])),
        "sample_order_corr": nl.safe_corr(diff, np.arange(len(diff), dtype=float)),
        "top10_abs_max_species_count": int(top_species.iloc[0]),
        "top10_abs_species_set": ",".join(map(str, sorted(top_species.index.tolist()))),
        **cal_meta,
    }
    for ref_name, ref_diff in refs.items():
        row[f"corr_diff_{ref_name}"] = nl.safe_corr(diff, ref_diff)
        row[f"rmse_diff_{ref_name}"] = nl.rmse(diff - ref_diff)
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else "pass"
    row["submit_gate"] = len(reasons) == 0
    row["public_failure_risk"] = nl.public_failure_risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons = nl.reject_reasons(row)
    if float(row["raw_anchor_diff_rmse"]) > 10.0:
        reasons.append("raw_test_diff_gt10")
    if abs(float(row["full_intercept"])) > 60.0:
        reasons.append("intercept_abs_gt60")
    slope = float(row["full_slope_raw"])
    if not (0.5 <= slope <= 1.2):
        reasons.append("slope_outside_0p5_1p2")
    if float(row["cal_slope_raw_min"]) < 0.35 or float(row["cal_slope_raw_max"]) > 1.5:
        reasons.append("fold_slope_unstable")
    if not (0.85 <= float(row["pred_std_ratio_vs_anchor"]) <= 1.15):
        reasons.append("pred_std_ratio_shift")
    return reasons


def failed_row(spec: nl.NonlinearSpec, exc: Exception) -> dict[str, object]:
    row = nl.failed_row(spec, exc)
    row["experiment"] = f"{spec.name}_oofcal"
    return row


def rmse(a: np.ndarray, b: np.ndarray | None = None) -> float:
    if b is None:
        values = np.asarray(a, dtype=float)
        return float(math.sqrt(np.mean(values**2)))
    return float(math.sqrt(mean_squared_error(a, b)))


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"))
    penalty = 0.0 if bool(row.get("submit_gate")) else 20.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.8 * max(float(row["direct_oof_delta_vs_anchor"]), 0.0)
        + 0.2 * abs(float(row["anchor_diff_rmse"]) - 0.18),
        float(row["direct_oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} risk={row['public_failure_risk']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} badcorr={row['corr_diff_bad_alpha3000']:.4f} "
        f"direct={row['direct_oof_delta_vs_anchor']:.4f} fold={row['direct_improved_fold_count']}/5 "
        f"slope={row['full_slope_raw']:.4f} reasons={row['reject_reasons']}"
    )


if __name__ == "__main__":
    main()
