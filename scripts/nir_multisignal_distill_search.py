#!/usr/bin/env python3
"""Combine a strong PLS residual signal with a tiny robust/golden signal."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_SCRIPT = ROOT / "scripts" / "nir_operator_branch_distill_search.py"
ROBUST_SCRIPT = ROOT / "scripts" / "nir_robust_branch_distill_search.py"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"
OUTPUT_ROOT = ROOT / "outputs" / "nir_multisignal_distill"


@dataclass(frozen=True)
class MultiSpec:
    pls_name: str
    robust_name: str
    pls_shrink: float
    robust_shrink: float
    pls_clip: float
    robust_clip: float
    total_clip: float


def main() -> None:
    op = load_module("nir_operator_branch_distill_search", OPERATOR_SCRIPT)
    rb = load_module("nir_robust_branch_distill_search", ROBUST_SCRIPT)
    data = op.load_data()
    anchor = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor[0].to_numpy()):
        raise ValueError("anchor sample order mismatch")
    anchor_test = anchor[1].to_numpy(float)

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF")
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    residual = data["y"] - anchor_oof
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}")

    pls_specs = [
        op.BranchSpec("pls_raw_msc_sg11_c4", "msc_sg11", "pls_raw", 4),
        op.BranchSpec("pls_raw_msc_sg9_c4", "msc_sg9", "pls_raw", 4),
        op.BranchSpec("pls_raw_msc_sg7_c4", "msc_sg7", "pls_raw", 4),
    ]
    robust_specs = [
        rb.RobustSpec("resid_lev_top12_w0p65", "resid_lev", 0.12, 0.65),
        rb.RobustSpec("resid_yj_top12_w0p65", "resid_yj", 0.12, 0.65),
        rb.RobustSpec("resid_lev_top08_w0p65", "resid_lev", 0.08, 0.65),
        rb.RobustSpec("resid_lev_nn_top12_w0p65", "resid_lev_nn", 0.12, 0.65),
    ]
    base_scores = rb.make_oof_scores(data["X_train"], data["y"], data["groups"])

    pls_signals = {}
    for spec in pls_specs:
        print(f"PLS signal {spec.name}")
        oof = op.make_branch_oof(spec, data["X_train"], data["y"], data["groups"])
        test = op.fit_predict_branch(spec, data["X_train"], data["y"], data["X_test"])
        signal_oof = oof - anchor_oof
        signal_test = test - anchor_test
        pls_signals[spec.name] = {
            "oof": signal_oof,
            "test": signal_test,
            "beta": fit_beta(signal_oof, residual),
            "corr": safe_corr(signal_oof, residual),
            "branch_oof_rmse": rmse(data["y"], oof),
        }

    robust_signals = {}
    for spec in robust_specs:
        print(f"robust signal {spec.name}")
        weights = rb.make_weights(spec, base_scores)
        oof = rb.make_branch_oof(spec, data["X_train"], data["y"], data["groups"])
        test = rb.fit_predict_weighted_branch(spec, data["X_train"], data["y"], data["X_test"], weights)
        signal_oof = oof - anchor_oof
        signal_test = test - anchor_test
        robust_signals[spec.name] = {
            "oof": signal_oof,
            "test": signal_test,
            "beta": fit_beta(signal_oof, residual),
            "corr": safe_corr(signal_oof, residual),
            "branch_oof_rmse": rmse(data["y"], oof),
        }

    rows: list[dict[str, object]] = []
    for pls_name in pls_signals:
        for robust_name in robust_signals:
            for pls_shrink in [0.02, 0.025, 0.03]:
                for robust_shrink in [0.005, 0.01, 0.015, 0.02, 0.03]:
                    for robust_clip in [0.04, 0.06, 0.08]:
                        spec = MultiSpec(
                            pls_name=pls_name,
                            robust_name=robust_name,
                            pls_shrink=pls_shrink,
                            robust_shrink=robust_shrink,
                            pls_clip=0.10,
                            robust_clip=robust_clip,
                            total_clip=0.14,
                        )
                        row = build_candidate(
                            spec,
                            pls_signals[pls_name],
                            robust_signals[robust_name],
                            anchor_test,
                            anchor_oof,
                            residual,
                            data,
                            candidate_dir,
                            anchor_oof_rmse,
                        )
                        rows.append(row)
                        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    with (out_dir / "multisignal_distill_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0]))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "multisignal_distill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop candidates:")
    for row in rows_sorted[:25]:
        print(
            f"{row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"oof_delta={row['oof_delta_vs_anchor']:.4f} "
            f"pls={row['pls_name']} robust={row['robust_name']}"
        )
    print(f"saved multisignal diagnostics: {out_dir}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_candidate(
    spec: MultiSpec,
    pls_signal: dict[str, np.ndarray | float],
    robust_signal: dict[str, np.ndarray | float],
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    residual: np.ndarray,
    data: dict[str, np.ndarray],
    candidate_dir: Path,
    anchor_oof_rmse: float,
) -> dict[str, object]:
    pls_oof = make_correction(
        float(pls_signal["beta"]),
        np.asarray(pls_signal["oof"]),
        spec.pls_shrink,
        spec.pls_clip,
        mean_center=False,
    )
    robust_oof = make_correction(
        float(robust_signal["beta"]),
        np.asarray(robust_signal["oof"]),
        spec.robust_shrink,
        spec.robust_clip,
        mean_center=False,
    )
    correction_oof = np.clip(pls_oof + robust_oof, -spec.total_clip, spec.total_clip)
    correction_oof = correction_oof - np.mean(correction_oof)
    corrected_oof = anchor_oof + correction_oof

    pls_test = make_correction(
        float(pls_signal["beta"]),
        np.asarray(pls_signal["test"]),
        spec.pls_shrink,
        spec.pls_clip,
        mean_center=False,
    )
    robust_test = make_correction(
        float(robust_signal["beta"]),
        np.asarray(robust_signal["test"]),
        spec.robust_shrink,
        spec.robust_clip,
        mean_center=False,
    )
    correction_test = np.clip(pls_test + robust_test, -spec.total_clip, spec.total_clip)
    correction_test = correction_test - np.mean(correction_test)
    pred = np.clip(anchor_test + correction_test, 0, None)

    name = (
        f"nir_msd_{spec.pls_name}_ps{tag(spec.pls_shrink)}_"
        f"{spec.robust_name}_rs{tag(spec.robust_shrink)}_rc{tag(spec.robust_clip)}_tc{tag(spec.total_clip)}"
    )
    path = candidate_dir / f"{name}.csv"
    pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)

    diff = pred - anchor_test
    species = pd.read_csv(ROOT / "data" / "raw" / "test.csv", encoding="cp932")["species number"].to_numpy()
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    return {
        "experiment": name,
        "pls_name": spec.pls_name,
        "robust_name": spec.robust_name,
        "pls_shrink": spec.pls_shrink,
        "robust_shrink": spec.robust_shrink,
        "pls_clip": spec.pls_clip,
        "robust_clip": spec.robust_clip,
        "total_clip": spec.total_clip,
        "pls_beta": float(pls_signal["beta"]),
        "robust_beta": float(robust_signal["beta"]),
        "pls_corr": float(pls_signal["corr"]),
        "robust_corr": float(robust_signal["corr"]),
        "signal_corr": safe_corr(np.asarray(pls_signal["oof"]), np.asarray(robust_signal["oof"])),
        "oof_delta_vs_anchor": rmse(data["y"], corrected_oof) - anchor_oof_rmse,
        "anchor_diff_rmse": float(math.sqrt(np.mean(diff**2))),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": safe_corr(pred, anchor_test),
        "bottom_decile_delta": float(np.mean(diff[lo])),
        "top_decile_delta": float(np.mean(diff[hi])),
        "max_abs_species_mean_shift": max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species)),
        "negative_count": int(np.sum(pred < 0)),
        "submission_path": str(path),
    }


def fit_beta(signal: np.ndarray, residual: np.ndarray) -> float:
    denom = float(np.dot(signal, signal))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(float(np.dot(signal, residual) / denom), -1.0, 1.0))


def make_correction(beta: float, signal: np.ndarray, shrink: float, clip: float, *, mean_center: bool) -> np.ndarray:
    delta = shrink * beta * signal
    delta = np.clip(delta, -clip, clip)
    if mean_center:
        delta = delta - np.mean(delta)
    return delta


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    rmse_diff = float(row["anchor_diff_rmse"])
    max_abs = float(row["anchor_diff_max_abs"])
    species = float(row["max_abs_species_mean_shift"])
    mean = abs(float(row["anchor_diff_mean"]))
    low = abs(float(row["bottom_decile_delta"]))
    top = abs(float(row["top_decile_delta"]))
    oof_delta = float(row["oof_delta_vs_anchor"])
    penalty = 0.0
    if rmse_diff < 0.05 or rmse_diff > 0.10:
        penalty += 5.0
    if max_abs > 0.16 or species > 0.03 or mean > 0.015 or low > 0.08 or top > 0.08:
        penalty += 5.0
    return (penalty + max(oof_delta, 0.0) + abs(rmse_diff - 0.07) + species, max_abs, species)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: diff={row['anchor_diff_rmse']:.4f} "
        f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
        f"oof_delta={row['oof_delta_vs_anchor']:.4f} sigcorr={row['signal_corr']:.4f}"
    )


if __name__ == "__main__":
    main()
