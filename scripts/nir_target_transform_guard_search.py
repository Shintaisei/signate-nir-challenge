#!/usr/bin/env python3
"""Focused target-transform search under current-anchor Public-failure gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer

import nir_anchor_rebuild_search as ar
import nir_operator_branch_distill_search as op


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "nir_target_transform_guard"
SUBMISSION_DIR = ROOT / "data" / "submissions"
CURRENT_ANCHOR = SUBMISSION_DIR / "nir_yj_oof_affine_s0p10_mc1.csv"
BAD_ALPHA3000 = SUBMISSION_DIR / "nir_sg9_snv_pca20_ridge3000.csv"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    preprocess: str
    target: str
    n_components: int
    alpha: float
    lam: float | None = None


@dataclass(frozen=True)
class TargetTransform:
    z: np.ndarray
    inverse: object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-specs", type=int, default=None)
    args = parser.parse_args()

    data = op.load_data()
    anchor_df = pd.read_csv(CURRENT_ANCHOR, header=None)
    if not np.array_equal(data["test_ids"], anchor_df[0].to_numpy()):
        raise ValueError("current anchor sample order mismatch")
    anchor_test = anchor_df[1].to_numpy(float)
    bad_diff = pd.read_csv(BAD_ALPHA3000, header=None)[1].to_numpy(float) - anchor_test
    test_species = pd.read_csv(op.TEST_PATH, encoding="cp932")["species number"].to_numpy()

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print("building current-anchor OOF")
    anchor_oof = op.make_current_anchor_oof(data["X_train"], data["y"], data["groups"])
    anchor_oof_rmse = rmse(data["y"], anchor_oof)
    print(f"anchor_oof_rmse={anchor_oof_rmse:.6f}")

    specs = build_specs()
    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    print(f"specs={len(specs)}")

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name}", flush=True)
        try:
            pred = fit_predict_spec(spec, data["X_train"], data["y"], data["X_test"])
            oof = make_oof(spec, data["X_train"], data["y"], data["groups"])
            pred = np.clip(pred, 0, None)
            oof = np.clip(oof, 0, None)
        except Exception as exc:
            row = failed_row(spec, exc)
            rows.append(row)
            print(f"  SKIP {type(exc).__name__}: {exc}")
            continue

        path = candidate_dir / f"{spec.name}.csv"
        pd.DataFrame({0: data["test_ids"], 1: pred}).to_csv(path, index=False, header=False)
        row = diagnostics(
            spec=spec,
            path=path,
            pred=pred,
            oof=oof,
            anchor_test=anchor_test,
            anchor_oof=anchor_oof,
            anchor_oof_rmse=anchor_oof_rmse,
            y=data["y"],
            groups=data["groups"],
            test_species=test_species,
            bad_diff=bad_diff,
        )
        rows.append(row)
        print_one(row)

    rows_sorted = sorted(rows, key=rank_key)
    fieldnames = sorted({key for row in rows_sorted for key in row})
    with (out_dir / "target_transform_guard_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (out_dir / "target_transform_guard_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print("\nTop target-transform candidates:")
    for row in rows_sorted[:60]:
        if row.get("status") != "ok":
            continue
        print(
            f"{row['experiment']}: gate={row['submit_gate']} reasons={row['reject_reasons']} "
            f"risk={row['public_failure_risk']:.4f} diff={row['anchor_diff_rmse']:.4f} "
            f"max={row['anchor_diff_max_abs']:.4f} sp={row['max_abs_species_mean_shift']:.4f} "
            f"badcorr={row['corr_diff_bad_alpha3000']:.4f} direct={row['direct_oof_delta_vs_anchor']:.4f} "
            f"fold={row['direct_improved_fold_count']}/5 range={row['range_ratio_vs_anchor']:.4f}"
        )
    print(f"saved target-transform diagnostics: {out_dir}")


def build_specs() -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    preprocesses = ["sg9_snv", "sg7_snv", "sg11_snv", "snv", "detrend_snv"]
    targets: list[tuple[str, float | None]] = [("boxcox_auto", None), ("yj_auto", None)]
    targets.extend(("boxcox_manual", lam) for lam in [0.0, 0.03, 0.05, 0.08, 0.092, 0.10, 0.12, 0.15, 0.20])
    targets.extend(("yj_manual", lam) for lam in [0.03, 0.047, 0.07])
    for preprocess in preprocesses:
        for target, lam in targets:
            for n_components in [15, 18, 20, 22, 24, 28]:
                for alpha in [3400.0, 3500.0, 3600.0, 3700.0, 4000.0]:
                    tag = target if lam is None else f"{target}_l{value_tag(lam)}"
                    specs.append(
                        TargetSpec(
                            name=f"nir_tg_{preprocess}_{tag}_p{n_components}_a{int(alpha)}",
                            preprocess=preprocess,
                            target=target,
                            lam=lam,
                            n_components=n_components,
                            alpha=alpha,
                        )
                    )
    return specs


def make_oof(spec: TargetSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, valid_idx in splitter.split(X, y, groups):
        pred[valid_idx] = fit_predict_spec(spec, X[train_idx], y[train_idx], X[valid_idx])
    return pred


def fit_predict_spec(spec: TargetSpec, X_train_raw: np.ndarray, y: np.ndarray, X_pred_raw: np.ndarray) -> np.ndarray:
    X_train, X_pred = ar.preprocess_pair(X_train_raw, X_pred_raw, spec.preprocess)
    pca = PCA(n_components=spec.n_components, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_pred = pca.transform(X_pred)
    transformed = transform_target(spec, y)
    model = Ridge(alpha=spec.alpha)
    model.fit(Z_train, transformed.z)
    pred_z = model.predict(Z_pred)
    return transformed.inverse(pred_z)


def transform_target(spec: TargetSpec, y: np.ndarray) -> TargetTransform:
    y = np.asarray(y, dtype=float)
    if spec.target == "yj_auto":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        z = transformer.fit_transform(y.reshape(-1, 1)).ravel()
        return TargetTransform(z, lambda pred, tr=transformer: tr.inverse_transform(np.asarray(pred).reshape(-1, 1)).ravel())
    if spec.target == "boxcox_auto":
        transformer = PowerTransformer(method="box-cox", standardize=True)
        z = transformer.fit_transform(y.reshape(-1, 1)).ravel()
        return TargetTransform(z, lambda pred, tr=transformer: tr.inverse_transform(np.asarray(pred).reshape(-1, 1)).ravel())
    if spec.target == "boxcox_manual":
        z_raw = manual_boxcox(y, float(spec.lam))
        mean = float(z_raw.mean())
        std = float(z_raw.std() or 1.0)
        z = (z_raw - mean) / std
        return TargetTransform(z, lambda pred, lam=float(spec.lam), m=mean, s=std: inv_manual_boxcox(np.asarray(pred) * s + m, lam))
    if spec.target == "yj_manual":
        z_raw = manual_yeojohnson_positive(y, float(spec.lam))
        mean = float(z_raw.mean())
        std = float(z_raw.std() or 1.0)
        z = (z_raw - mean) / std
        return TargetTransform(z, lambda pred, lam=float(spec.lam), m=mean, s=std: inv_manual_yeojohnson_positive(np.asarray(pred) * s + m, lam))
    raise ValueError(spec.target)


def manual_boxcox(y: np.ndarray, lam: float) -> np.ndarray:
    if np.any(y <= 0):
        raise ValueError("Box-Cox requires positive target")
    if abs(lam) < 1e-12:
        return np.log(y)
    return (np.power(y, lam) - 1.0) / lam


def inv_manual_boxcox(z: np.ndarray, lam: float) -> np.ndarray:
    if abs(lam) < 1e-12:
        return np.exp(z)
    return np.power(np.maximum(lam * z + 1.0, 1e-12), 1.0 / lam)


def manual_yeojohnson_positive(y: np.ndarray, lam: float) -> np.ndarray:
    if abs(lam) < 1e-12:
        return np.log1p(y)
    return (np.power(y + 1.0, lam) - 1.0) / lam


def inv_manual_yeojohnson_positive(z: np.ndarray, lam: float) -> np.ndarray:
    if abs(lam) < 1e-12:
        return np.expm1(z)
    return np.power(np.maximum(lam * z + 1.0, 1e-12), 1.0 / lam) - 1.0


def diagnostics(
    *,
    spec: TargetSpec,
    path: Path,
    pred: np.ndarray,
    oof: np.ndarray,
    anchor_test: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_oof_rmse: float,
    y: np.ndarray,
    groups: np.ndarray,
    test_species: np.ndarray,
    bad_diff: np.ndarray,
) -> dict[str, object]:
    diff = pred - anchor_test
    oof_rmse = rmse(y, oof)
    fold = fold_delta_stats(y, oof, anchor_oof, groups)
    lo = anchor_test <= np.quantile(anchor_test, 0.10)
    hi = anchor_test >= np.quantile(anchor_test, 0.90)
    top10 = np.argsort(np.abs(diff))[-10:]
    top_species = pd.Series(test_species[top10]).value_counts()
    row: dict[str, object] = {
        "status": "ok",
        "experiment": spec.name,
        "submission_path": str(path),
        "preprocess": spec.preprocess,
        "target": spec.target,
        "lambda": "" if spec.lam is None else spec.lam,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "anchor_oof_rmse": anchor_oof_rmse,
        "direct_oof_rmse": oof_rmse,
        "direct_oof_delta_vs_anchor": oof_rmse - anchor_oof_rmse,
        "direct_improved_fold_count": fold["improved_count"],
        "direct_worst_fold_delta": fold["worst_delta"],
        "direct_fold_delta_std": fold["delta_std"],
        "pred_min": float(pred.min()),
        "pred_mean": float(pred.mean()),
        "pred_median": float(np.median(pred)),
        "pred_max": float(pred.max()),
        "negative_count": int(np.sum(pred < 0)),
        "anchor_diff_rmse": rmse(diff),
        "anchor_diff_max_abs": float(np.max(np.abs(diff))),
        "anchor_diff_mean": float(np.mean(diff)),
        "anchor_corr": safe_corr(pred, anchor_test),
        "max_abs_species_mean_shift": max_abs_species_shift(diff, test_species),
        "range_ratio_vs_anchor": float((pred.max() - pred.min()) / (anchor_test.max() - anchor_test.min())),
        "corr_diff_bad_alpha3000": safe_corr(diff, bad_diff),
        "bottom_decile_shift": float(np.mean(diff[lo])),
        "top_decile_shift": float(np.mean(diff[hi])),
        "decile_gap_top_minus_bottom": float(np.mean(diff[hi]) - np.mean(diff[lo])),
        "sample_order_corr": safe_corr(diff, np.arange(len(diff), dtype=float)),
        "top10_abs_max_species_count": int(top_species.iloc[0]),
        "top10_abs_species_set": ",".join(map(str, sorted(top_species.index.tolist()))),
    }
    reasons = reject_reasons(row)
    row["reject_reasons"] = "|".join(reasons) if reasons else "pass"
    row["submit_gate"] = len(reasons) == 0
    row["public_failure_risk"] = public_failure_risk(row)
    return row


def reject_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    diff = float(row["anchor_diff_rmse"])
    if diff < 0.05:
        reasons.append("near_anchor")
    if diff > 0.25:
        reasons.append("anchor_diff_gt0p25")
    if float(row["anchor_diff_max_abs"]) > 1.50:
        reasons.append("max_diff_gt1p5")
    if float(row["max_abs_species_mean_shift"]) > 0.18:
        reasons.append("species_shift_gt0p18")
    if float(row["max_abs_species_mean_shift"]) > 0.12:
        reasons.append("species_shift_caution")
    if float(row["corr_diff_bad_alpha3000"]) > 0.40:
        reasons.append("bad_alpha3000_corr_gt0p40")
    if abs(float(row["decile_gap_top_minus_bottom"])) > 0.50:
        reasons.append("decile_gap_gt0p50")
    if int(row["top10_abs_max_species_count"]) > 4:
        reasons.append("top10_species_concentration")
    if not (0.90 <= float(row["range_ratio_vs_anchor"]) <= 1.10):
        reasons.append("range_shift")
    if float(row["direct_oof_delta_vs_anchor"]) > 0.10:
        reasons.append("direct_oof_bad")
    if int(row["direct_improved_fold_count"]) < 3:
        reasons.append("direct_folds_lt3")
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
        + (0.5 if int(row["top10_abs_max_species_count"]) > 4 else 0.0)
        + 0.5 * max(float(row["direct_oof_delta_vs_anchor"]), 0.0)
    )


def failed_row(spec: TargetSpec, exc: Exception) -> dict[str, object]:
    return {
        "status": "failed",
        "experiment": spec.name,
        "preprocess": spec.preprocess,
        "target": spec.target,
        "lambda": "" if spec.lam is None else spec.lam,
        "n_components": spec.n_components,
        "alpha": spec.alpha,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "submit_gate": False,
        "reject_reasons": "failed",
        "public_failure_risk": float("inf"),
    }


def fold_delta_stats(y: np.ndarray, pred: np.ndarray, anchor_pred: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    deltas: list[float] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for _, valid_idx in splitter.split(np.zeros((len(y), 1)), y, groups):
        deltas.append(rmse(y[valid_idx], pred[valid_idx]) - rmse(y[valid_idx], anchor_pred[valid_idx]))
    arr = np.asarray(deltas)
    return {
        "improved_count": int(np.sum(arr < 0)),
        "worst_delta": float(np.max(arr)),
        "delta_std": float(np.std(arr)),
    }


def max_abs_species_shift(diff: np.ndarray, species: np.ndarray) -> float:
    return max(abs(float(np.mean(diff[species == group]))) for group in np.unique(species))


def rmse(a: np.ndarray, b: np.ndarray | None = None) -> float:
    if b is None:
        values = np.asarray(a, dtype=float)
        return float(math.sqrt(np.mean(values**2)))
    return float(math.sqrt(mean_squared_error(a, b)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    if row.get("status") != "ok":
        return (999.0, float("inf"), float("inf"))
    penalty = 0.0 if bool(row.get("submit_gate")) else 20.0
    return (
        penalty
        + float(row["public_failure_risk"])
        + 0.6 * max(float(row["direct_oof_delta_vs_anchor"]), 0.0)
        + abs(float(row["anchor_diff_rmse"]) - 0.14) * 0.4,
        float(row["direct_oof_delta_vs_anchor"]),
        float(row["anchor_diff_max_abs"]),
    )


def value_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def print_one(row: dict[str, object]) -> None:
    print(
        f"  {row['experiment']}: gate={row['submit_gate']} risk={row['public_failure_risk']:.4f} "
        f"diff={row['anchor_diff_rmse']:.4f} max={row['anchor_diff_max_abs']:.4f} "
        f"sp={row['max_abs_species_mean_shift']:.4f} badcorr={row['corr_diff_bad_alpha3000']:.4f} "
        f"direct={row['direct_oof_delta_vs_anchor']:.4f} fold={row['direct_improved_fold_count']}/5 "
        f"reasons={row['reject_reasons']}"
    )


if __name__ == "__main__":
    main()
